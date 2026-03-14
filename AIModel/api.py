"""
Energy & Carbon Forecasting Service

Prophet-based REST API that loads all 15 schema metrics, fits 7 Prophet models
at startup, and exposes forecasts consumable by schedulers, dashboards, or agents.

Usage (Docker):
    docker compose up --build

Endpoints:
    POST /data                               — ingest new measurements (EVIDEN push)
    GET /health                              — model status, readiness, and L.1801 compliance
    GET /forecast/all?horizon=1             — all metrics in one response (default 60 min)
    GET /forecast/sci?horizon=1             — SCI (derived, kgCO2e/req) with functional unit
    GET /forecast/peak?metric=consumption   — peak provisioning value (yhat_upper) for HPA/KEDA
    GET /forecast/{metric}?horizon=1        — single metric forecast
    GET /optimal-window?horizon_days=7       — best low-carbon scheduling windows
    GET /anomalies?metric=consumption        — point anomalies vs Prophet bounds
    GET /investigation-leads                 — ranked recurring anomaly patterns
    GET /metrics/prometheus                  — Prometheus text format with confidence bands

Available metrics for /forecast/{metric}:
    consumption, carbonEmissions, carbonIntensityFactor,
    greenConsumptionPercentage, cpuUtilization, functionalUnit, cost
"""
import os
import logging
import warnings
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal, NamedTuple, Optional

# Suppress noisy loggers before prophet imports
logging.getLogger("prophet").setLevel(logging.WARNING)
_cmdstan = logging.getLogger("cmdstanpy")
_cmdstan.setLevel(logging.ERROR)
_cmdstan.addHandler(logging.NullHandler())
_cmdstan.propagate = False
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from data_generator import generate_energy_carbon_data
from data_loader import load_and_validate, build_pipeline_config
from prophet_model import EnergyProphet
from anomaly_detector import detect_point_anomalies, find_recurring_patterns, build_investigation_leads
from model_registry import get_best_model_key
from physics_constraint import derive_carbon_emissions

try:
    from timesfm_model import EnergyTimesFM
    _TIMESFM_AVAILABLE = True
except ImportError:
    _TIMESFM_AVAILABLE = False
    # Prophet-only mode; timesfm_flagged=False, confidence="prophet-only" on all anomalies

log = logging.getLogger("api")

# Node name used when no ?node= query param is supplied.
# All startup models are stored under this key; unknown nodes fall back to it.
_DEFAULT_NODE = "_default"

# ── Internal types ───────────────────────────────────────────────────────────────

_ModelKey = Literal["formula", "timesfm", "prophet"]


class _ForecastResult(NamedTuple):
    """Return type for _run_best_forecast: forecast DataFrame + which model served it."""
    data:       pd.DataFrame
    model_used: _ModelKey


# ── Config ──────────────────────────────────────────────────────────────────────

DATA_PATH = os.environ.get("DATA_PATH", "data/sample_energy_data.csv")

UNIT_MAP: dict[str, str] = {
    "consumption":                "kWh/h",
    "carbonEmissions":            "kgCO2e/h",
    "carbonIntensityFactor":      "kgCO2/kWh",
    "greenConsumptionPercentage": "%",
    "cpuUtilization":             "fraction",
    "functionalUnit":             "req/h",
    "cost":                       "EUR/h",
    "softwareCarbonIntensity":    "kgCO2e/req",
}

# Metrics the service fits Prophet models for at startup.
# Only metrics present in the loaded data are actually fit (others are skipped).
FIT_METRICS: list[str] = [
    "consumption",
    "carbonEmissions",
    "carbonIntensityFactor",
    "greenConsumptionPercentage",
    "cpuUtilization",
    "functionalUnit",
    "cost",
]

# Model routing is defined in model_registry.py — see get_best_model_key()

# ── App state ────────────────────────────────────────────────────────────────────
# Outer key: node name (str). Inner dict: models, df, caches for that node.
# All startup models live under _DEFAULT_NODE. New nodes are created on first
# POST /data?node=<name> and receive a background refit on their own data.

_state: dict[str, dict] = {}


# ── Lifespan ─────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data and fit Prophet models on startup."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    # 1. Load or generate data
    if os.path.exists(DATA_PATH):
        log.info("Loading data from %s", DATA_PATH)
        try:
            df = load_and_validate(DATA_PATH)
        except (FileNotFoundError, ValueError) as exc:
            log.error("Data validation failed: %s — falling back to synthetic data", exc)
            df = generate_energy_carbon_data()
    else:
        log.info("No data found — generating synthetic data …")
        os.makedirs(os.path.dirname(DATA_PATH) or ".", exist_ok=True)
        df = generate_energy_carbon_data()
        df.to_csv(DATA_PATH, index=False)
        log.info("Saved synthetic data to %s", DATA_PATH)

    config = build_pipeline_config(df)

    # 2. Fit one Prophet model per metric (skips metrics absent from data)
    models: dict[str, EnergyProphet] = {}
    for metric in FIT_METRICS:
        if metric not in df.columns:
            log.warning("Column '%s' not in data — skipping.", metric)
            continue
        regs = config.regressor_map.get(metric, [])
        label = f" + regressors {regs}" if regs else ""
        log.info("Fitting Prophet  %-28s%s …", f"'{metric}'", label)
        m = EnergyProphet(regressors=regs)
        m.fit(df, metric)
        models[metric] = m

    # 3. Store TimesFM context for each metric (no model weights loaded yet — lazy)
    timesfm_models: dict = {}
    if _TIMESFM_AVAILABLE:
        for metric in FIT_METRICS:
            if metric in df.columns:
                tfm = EnergyTimesFM()
                tfm.fit(df, metric)   # stores history only; no model load yet
                timesfm_models[metric] = tfm
        log.info(
            "TimesFM context stored for %d metrics (weights lazy-loaded on first /anomalies)",
            len(timesfm_models),
        )

    _state[_DEFAULT_NODE] = {
        "df":                         df,
        "models":                     models,
        "config":                     config,
        "insample_forecasts":         {},   # lazy-populated on first /anomalies request
        "timesfm_models":             timesfm_models,
        "timesfm_insample_forecasts": {},   # lazy-populated on first /anomalies call per metric
        "fitted_at":                  datetime.utcnow().isoformat() + "Z",
        "data_rows":                  len(df),
    }
    log.info("Service ready — %d rows, models: %s", len(df), list(models.keys()))

    yield  # service runs here

    _state.clear()


# ── App ──────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Energy & Carbon Forecasting Service",
    description=(
        "Prophet-based forecasting for all Green Software Foundation SCI metrics. "
        "Loads all 15 schema columns; fits 7 Prophet models directly; derives "
        "softwareCarbonIntensity (SCI) from carbonEmissions / functionalUnit."
    ),
    version="0.3.0",
    lifespan=lifespan,
)


# ── Response models ──────────────────────────────────────────────────────────────

class ForecastPoint(BaseModel):
    ds:         str
    yhat:       float
    yhat_lower: float
    yhat_upper: float


class CarbonBreakdown(BaseModel):
    """L.1801-aligned carbon accounting metadata for carbonEmissions / SCI forecasts."""
    formula:           str    # human-readable formula used to derive carbonEmissions
    embodied_kgco2e_h: float  # hardware amortization constant (kgCO2e/h)
    sci_standard:      str    # ratified standard reference
    l1801_stage:       str    # L.1801 lifecycle stage this formula covers


class ForecastResponse(BaseModel):
    node:                        Optional[str]
    metric:                      str
    unit:                        str
    model_used:                  _ModelKey   # "formula" | "timesfm" | "prophet"
    horizon_hours:               int
    generated_at:                str
    predictions:                 list[ForecastPoint]
    carbon_breakdown:            Optional[CarbonBreakdown] = None
    functional_unit_declaration: Optional[str] = None


class SchedulingWindow(BaseModel):
    date:                 str
    start_hour:           int
    end_hour:             int
    avg_predicted_carbon: float
    vs_daily_mean_pct:    float   # negative = greener than the day's average
    unit:                 str


class OptimalWindowResponse(BaseModel):
    node:         Optional[str]
    model_used:   _ModelKey   # model used for the carbonEmissions forecast
    horizon_days: int
    window_hours: int
    generated_at: str
    windows:      list[SchedulingWindow]


class L1801Compliance(BaseModel):
    """Partial ITU-T L.1801 compliance declaration (approved 2026-02-06).

    L.1801 §4.3 requires omissions to be declared explicitly — partial compliance
    with a transparency statement is preferred over silent non-compliance.
    """
    standard:           str
    version:            str        # ISO date of standard approval
    partial_compliance: bool
    implemented:        list[str]  # lifecycle stages / capabilities already satisfied
    pending_gaps:       list[str]  # declared omissions
    sci_standard:       str        # ISO/IEC 21031:2024 (SCI formula)


class PeakForecastResponse(BaseModel):
    """Scheduler-facing peak provisioning values for HPA/KEDA integration.

    provision_for is yhat_upper at the forecast peak — the conservative ceiling
    a Kubernetes HPA or KEDA ScaledObject should pre-provision for. Use this
    value as the HPA metric threshold to prevent brownouts during demand spikes.
    """
    node:                Optional[str]
    metric:              str
    unit:                str
    horizon_hours:       int
    provision_for:       float   # max yhat_upper in the forecast window — provision at least this
    expected:            float   # yhat at the peak hour — best estimate
    floor:               float   # yhat_lower at the peak hour — minimum expected
    confidence_band_pct: float   # (provision_for − floor) / expected × 100 — width of uncertainty
    peak_hour:           str     # ISO 8601 timestamp of the forecast peak
    model_used:          _ModelKey
    generated_at:        str


class HealthResponse(BaseModel):
    status:           str
    fitted_at:        str
    data_rows:        int
    models_ready:     list[str]
    nodes:            list[str]          # all known node keys (includes "_default")
    l1801_compliance: L1801Compliance


class AnomalyPoint(BaseModel):
    ds:                   str
    actual:               float
    expected:             float
    upper_bound:          float
    lower_bound:          float
    excess_pct:           float
    direction:            str    # "excess" | "deficit"
    hour_of_day:          int
    is_weekend:           bool
    carbon_intensity:     float
    excess_carbon_kgco2e: float
    excess_cost_eur:      float
    timesfm_flagged:      bool   # True if TimesFM also flags this timestamp as anomalous
    confidence:           str    # "high" | "prophet-only" | "timesfm-only"


class AnomalyResponse(BaseModel):
    node:            Optional[str]
    metric:          str
    unit:            str
    direction:       str
    lookback_days:   int
    total_anomalies: int
    generated_at:    str
    anomalies:       list[AnomalyPoint]


class InvestigationLeadPoint(BaseModel):
    rank:                        int
    metric:                      str
    unit:                        str
    pattern_summary:             str
    occurrences:                 int
    frequency_pct:               float
    avg_excess_pct:              float
    total_excess_carbon_kgco2e:  float
    total_excess_cost_eur:       float
    optimal_window_start_hour:   int
    optimal_window_end_hour:     int
    estimated_carbon_saving_pct: float
    estimated_cost_saving_pct:   float
    carbon_context:              str
    example_timestamps:          list[str]


class InvestigationLeadsResponse(BaseModel):
    lookback_days: int
    top_n:         int
    generated_at:  str
    leads:         list[InvestigationLeadPoint]


class DataRow(BaseModel):
    """A single measurement row pushed by an external system (e.g. EVIDEN).

    Only `ds` and `consumption` are required — the pipeline degrades gracefully
    when optional columns are absent, matching data_loader.py's minimum schema.
    """
    ds:                       str            = Field(..., description="ISO 8601 timestamp, e.g. '2025-01-01T14:00:00'")
    consumption:              float          = Field(..., ge=0, description="Energy consumption (kWh/h)")
    carbonEmissions:          Optional[float] = Field(None, ge=0)
    carbonIntensityFactor:    Optional[float] = Field(None, ge=0)
    functionalUnit:           Optional[float] = Field(None, ge=0)
    cost:                     Optional[float] = Field(None, ge=0)
    greenConsumptionPercentage: Optional[float] = Field(None, ge=0, le=100)
    cpuUtilization:           Optional[float] = Field(None, ge=0, le=1)
    operationalEmissions:     Optional[float] = Field(None, ge=0)
    embodiedEmissions:        Optional[float] = Field(None, ge=0)
    totalConsumption:         Optional[float] = Field(None, ge=0)
    totalCost:                Optional[float] = Field(None, ge=0)
    softwareCarbonIntensity:  Optional[float] = Field(None, ge=0)
    timeWindow:               Optional[int]   = Field(None, ge=0)
    measurementSource:        Optional[str]   = None


class DataIngestionRequest(BaseModel):
    rows: list[DataRow] = Field(..., min_length=1, description="One or more measurement rows")


class DataIngestionResponse(BaseModel):
    rows_accepted:  int
    total_rows:     int
    models_status:  str  # "refit_scheduled"


# ── L.1801 compliance declaration (static — does not change at runtime) ──────────
# ITU-T L.1801 §4.3: omissions must be declared explicitly.
# Partial compliance with a transparency statement is preferred over silent non-compliance.

_L1801_COMPLIANCE = L1801Compliance(
    standard="ITU-T L.1801",
    version="2026-02-06",
    partial_compliance=True,
    implemented=[
        "Stage3_Operation: energy consumption monitored and forecast (consumption kWh/h)",
        "Stage3_Operation: carbon emissions forecast via physics formula E×CIF+M",
        "SCI formula implemented (ISO/IEC 21031:2024): SCI = (E×I + M) / R",
        "Functional unit declared in /forecast/sci (R = req/h)",
        "Confidence intervals propagated through all forecasts (yhat_lower, yhat_upper)",
        "Model provenance exposed via model_used label on every forecast response",
    ],
    pending_gaps=[
        "Stage1_Training: Prophet and TimesFM training emissions not tracked",
        "Stage2_Deployment: container build and inference serving emissions not tracked",
        "Stage4_Disposal: model decommissioning and hardware disposal not modelled",
        "Hardware embodied carbon M=0.002 from engineering estimate, not Boavizta LCA",
        "WUE (Water Usage Effectiveness) of data centre not measured",
        "Second-order network and cooling effects not modelled",
    ],
    sci_standard="ISO/IEC 21031:2024",
)


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _require_state(node: Optional[str] = None) -> dict:
    """Return the state dict for `node`, falling back to the default node.

    When `node` is None or not yet known, returns the default node's state.
    This makes per-node operations fail-safe: an unknown node transparently
    uses the global default models until its own background refit completes.
    """
    key = node if node is not None else _DEFAULT_NODE
    state = _state.get(key) or _state.get(_DEFAULT_NODE)
    if state is None:
        raise HTTPException(status_code=503, detail="Models not ready yet.")
    return state


def _refit_models_in_background(df: pd.DataFrame, node: str = _DEFAULT_NODE) -> None:
    """Refit all Prophet models for `node`; atomically replaces old models when done."""
    config = build_pipeline_config(df)
    new_models: dict[str, EnergyProphet] = {}

    for metric in FIT_METRICS:
        if metric not in df.columns:
            continue
        regs = config.regressor_map.get(metric, [])
        m = EnergyProphet(regressors=regs)
        try:
            m.fit(df, metric)
            new_models[metric] = m
            log.info("Background refit OK: '%s' (node=%s)", metric, node)
        except Exception:
            log.exception("Background refit failed for '%s' (node=%s) — keeping old model", metric, node)
            old = _state.get(node, {}).get("models", {}).get(metric)
            if old is not None:
                new_models[metric] = old

    # Guard: service might have shut down between task creation and execution
    if node not in _state:
        return

    # Atomically swap in new models and flush all stale caches for this node only
    _state[node]["models"] = new_models
    _state[node]["config"] = config
    _state[node]["insample_forecasts"] = {}
    _state[node]["timesfm_insample_forecasts"] = {}
    log.info("Background refit complete — %d Prophet models updated (node=%s)", len(new_models), node)


def _build_regressor_future_df(
    horizon: int, regressors: list[str], node: Optional[str] = None
) -> pd.DataFrame:
    """Build historical + forecast regressor values for chained forecasting."""
    state = _require_state(node)
    df = state["df"]

    # Start with historical regressor values (actual, known timestamps)
    hist_cols = ["ds"] + [r for r in regressors if r in df.columns]
    combined = df[hist_cols].copy()
    combined["ds"] = pd.to_datetime(combined["ds"])

    # Forecast each regressor and append its future predictions
    for reg in regressors:
        if reg not in state["models"]:
            log.warning(
                "No model for regressor '%s' — future values will be zero-filled.", reg
            )
            continue
        reg_fc = _run_forecast(reg, horizon, node)
        future_rows = pd.DataFrame({
            "ds": pd.to_datetime(reg_fc["ds"]),
            reg:  reg_fc["yhat"].values,
        })
        combined = pd.concat([combined, future_rows], ignore_index=True)

    return combined


def _run_forecast(metric: str, horizon: int, node: Optional[str] = None) -> pd.DataFrame:
    """Return the future `horizon` rows from Prophet for `metric` (with chained regressors)."""
    state = _require_state(node)
    if metric not in state["models"]:
        available = list(state["models"].keys())
        raise HTTPException(
            status_code=404,
            detail=f"No model for '{metric}'. Available: {available}",
        )
    model: EnergyProphet = state["models"][metric]
    regressors = state["config"].regressor_map.get(metric, [])

    if regressors:
        # Chained forecasting: forecast regressors first, inject as future_df
        future_df = _build_regressor_future_df(horizon, regressors, node)
    else:
        future_df = state["df"]

    forecast = model.predict(periods=horizon, future_df=future_df)
    return forecast.tail(horizon).reset_index(drop=True)


def _run_timesfm_forecast(metric: str, horizon: int, node: Optional[str] = None) -> pd.DataFrame:
    """Run TimesFM forecast for a single metric. Returns future-only DataFrame."""
    tfm_models = _require_state(node).get("timesfm_models", {})
    if metric not in tfm_models:
        raise HTTPException(status_code=503, detail=f"TimesFM model for {metric} not available")
    model = tfm_models[metric]
    fc = model.predict(horizon)   # returns history + future rows
    return fc.tail(horizon).reset_index(drop=True)


def _run_formula_forecast(horizon: int, node: Optional[str] = None) -> pd.DataFrame:
    """Derive carbonEmissions analytically: consumption × carbonIntensityFactor + 0.002."""
    consumption_fc = _run_forecast("consumption", horizon, node)
    cif_fc         = _run_forecast("carbonIntensityFactor", horizon, node)
    return derive_carbon_emissions(consumption_fc, cif_fc)


def _run_best_forecast(metric: str, horizon: int, node: Optional[str] = None) -> _ForecastResult:
    """Route to the best available model for a metric (see model_registry.py)."""
    state = _require_state(node)
    available: set[str] = set()
    if "consumption" in state["models"] and "carbonIntensityFactor" in state["models"]:
        available.add("formula")
    if _TIMESFM_AVAILABLE and metric in state.get("timesfm_models", {}):
        available.add("timesfm")
    if metric in state["models"]:
        available.add("prophet")

    best = get_best_model_key(metric, available)

    if best == "formula":
        try:
            return _ForecastResult(_run_formula_forecast(horizon, node), "formula")
        except Exception:
            log.warning("Formula forecast failed for '%s' — falling back", metric)
    if best in ("timesfm", "formula"):   # formula fell through
        try:
            return _ForecastResult(_run_timesfm_forecast(metric, horizon, node), "timesfm")
        except Exception:
            log.warning("TimesFM forecast failed for '%s' — falling back to Prophet", metric)
    return _ForecastResult(_run_forecast(metric, horizon, node), "prophet")


_CARBON_BREAKDOWN = CarbonBreakdown(
    formula="carbonEmissions = consumption × carbonIntensityFactor + 0.002",
    embodied_kgco2e_h=0.002,
    sci_standard="ISO/IEC 21031:2024",
    l1801_stage="Stage3_Operation",
)


def _to_response(
    metric: str,
    horizon: int,
    fc: pd.DataFrame,
    model_used: _ModelKey,
    node: Optional[str] = None,
    include_breakdown: bool = False,
) -> ForecastResponse:
    predictions = [
        ForecastPoint(
            ds=pd.Timestamp(row["ds"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
            yhat=      round(float(row["yhat"]),       6),
            yhat_lower=round(float(row["yhat_lower"]), 6),
            yhat_upper=round(float(row["yhat_upper"]), 6),
        )
        for _, row in fc.iterrows()
    ]
    breakdown = _CARBON_BREAKDOWN if (include_breakdown and metric == "carbonEmissions") else None
    return ForecastResponse(
        node=node,
        metric=metric,
        unit=UNIT_MAP.get(metric, ""),
        model_used=model_used,
        horizon_hours=horizon,
        generated_at=datetime.utcnow().isoformat() + "Z",
        predictions=predictions,
        carbon_breakdown=breakdown,
    )


_SCI_CARBON_BREAKDOWN = CarbonBreakdown(
    formula="SCI = ((consumption × carbonIntensityFactor + 0.002) / functionalUnit)",
    embodied_kgco2e_h=0.002,
    sci_standard="ISO/IEC 21031:2024",
    l1801_stage="Stage3_Operation",
)

_SCI_FUNCTIONAL_UNIT = "req/h — requests per hour (the SCI denominator R in ISO/IEC 21031:2024)"


def _build_sci_response(
    horizon: int, node: Optional[str] = None, include_breakdown: bool = False
) -> ForecastResponse:
    """Derive SCI with propagated uncertainty from two component forecasts."""
    carbon_result = _run_best_forecast("carbonEmissions", horizon, node)
    carbon_fc     = carbon_result.data
    request_fc    = _run_forecast("functionalUnit", horizon, node)

    predictions = []
    for (_, c_row), (_, r_row) in zip(carbon_fc.iterrows(), request_fc.iterrows()):
        r_hat   = max(float(r_row["yhat"]),       1.0)
        r_upper = max(float(r_row["yhat_upper"]), 1.0)
        r_lower = max(float(r_row["yhat_lower"]), 1.0)

        predictions.append(ForecastPoint(
            ds=pd.Timestamp(c_row["ds"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
            yhat=       round(float(c_row["yhat"])       / r_hat,   8),
            yhat_lower= round(float(c_row["yhat_lower"]) / r_upper, 8),  # best SCI
            yhat_upper= round(float(c_row["yhat_upper"]) / r_lower, 8),  # worst SCI
        ))

    breakdown = _SCI_CARBON_BREAKDOWN if include_breakdown else None
    return ForecastResponse(
        node=node,
        metric="softwareCarbonIntensity",
        unit=UNIT_MAP["softwareCarbonIntensity"],
        model_used=carbon_result.model_used,
        horizon_hours=horizon,
        generated_at=datetime.utcnow().isoformat() + "Z",
        predictions=predictions,
        carbon_breakdown=breakdown,
        functional_unit_declaration=_SCI_FUNCTIONAL_UNIT,
    )


def _get_insample_forecast(metric: str, node: Optional[str] = None) -> "pd.DataFrame":
    """Return in-sample Prophet predictions for `metric`, lazy-cached per node."""
    state = _require_state(node)
    if metric not in state["models"]:
        available = list(state["models"].keys())
        raise HTTPException(
            status_code=404,
            detail=f"No model for '{metric}'. Available: {available}",
        )

    cache = state["insample_forecasts"]
    if metric not in cache:
        model: EnergyProphet = state["models"][metric]
        df = state["df"]
        # periods=0 → make_future_dataframe returns only the training timestamps.
        # Passing future_df=df supplies historical regressor values (e.g., functionalUnit).
        insample = model.predict(periods=0, future_df=df)
        cache[metric] = insample[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        log.info(
            "In-sample forecast cached for '%s' (%d rows)", metric, len(cache[metric])
        )

    return cache[metric]


def _compute_timesfm_rolling_insample(metric: str, node: Optional[str] = None) -> "pd.DataFrame":
    """Rolling CV for TimesFM in-sample predictions (step=24h, min_context=168h).

    Returns an empty DataFrame if TimesFM is unavailable or metric has no context.
    """
    _empty = pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])
    state = _require_state(node)
    if metric not in state.get("timesfm_models", {}):
        return _empty

    tfm = state["timesfm_models"][metric]
    log.info("Computing TimesFM rolling CV for '%s' (first call — may take ~60s)…", metric)

    tfm._ensure_model()

    history        = tfm._history         # numpy array, shape (n,)
    history_dates  = tfm._history_dates   # pd.Series of datetimes
    step           = 24
    min_context    = 168
    context_window = tfm.context_len      # 512

    n    = len(history)
    rows = []

    for pos in range(min_context, n, step):
        ctx_start = max(0, pos - context_window)
        ctx       = history[ctx_start:pos]

        horizon = min(step, n - pos)
        if horizon <= 0:
            break

        point, quantile = tfm._model.forecast([ctx], freq=[0])

        for i in range(horizon):
            rows.append({
                "ds":         history_dates.iloc[pos + i],
                "yhat":       float(point[0, i]),
                "yhat_lower": float(quantile[0, i, 0]),   # 10th percentile
                "yhat_upper": float(quantile[0, i, 8]),   # 90th percentile
            })

    if not rows:
        return _empty

    result = pd.DataFrame(rows)
    log.info("TimesFM rolling CV complete for '%s' (%d rows)", metric, len(result))
    return result


def _get_timesfm_insample_forecast(metric: str, node: Optional[str] = None) -> "pd.DataFrame":
    """Return TimesFM rolling-CV in-sample predictions for `metric`, lazy-cached per node."""
    state = _require_state(node)
    cache = state["timesfm_insample_forecasts"]
    if metric not in cache:
        cache[metric] = _compute_timesfm_rolling_insample(metric, node)
        log.info(
            "TimesFM in-sample forecast cached for '%s' (%d rows)",
            metric, len(cache[metric]),
        )
    return cache[metric]


def _anomaly_to_point(a, timesfm_flagged: bool = False, confidence: str = "prophet-only") -> "AnomalyPoint":
    """Convert an Anomaly dataclass to its Pydantic response model."""
    direction = "excess" if a.excess_pct >= 0 else "deficit"
    return AnomalyPoint(
        ds=                   a.ds.strftime("%Y-%m-%dT%H:%M:%SZ"),
        actual=               round(a.actual,               6),
        expected=             round(a.expected,             6),
        upper_bound=          round(a.upper_bound,          6),
        lower_bound=          round(a.lower_bound,          6),
        excess_pct=           round(a.excess_pct,           2),
        direction=            direction,
        hour_of_day=          a.hour_of_day,
        is_weekend=           a.is_weekend,
        carbon_intensity=     round(a.carbon_intensity,     4),
        excess_carbon_kgco2e= round(a.excess_carbon_kgco2e, 6),
        excess_cost_eur=      round(a.excess_cost_eur,       6),
        timesfm_flagged=      timesfm_flagged,
        confidence=           confidence,
    )


def _lead_to_point(lead) -> InvestigationLeadPoint:
    """Convert an InvestigationLead dataclass to its Pydantic response model."""
    return InvestigationLeadPoint(
        rank=                        lead.rank,
        metric=                      lead.metric,
        unit=                        UNIT_MAP.get(lead.metric, ""),
        pattern_summary=             lead.pattern_summary,
        occurrences=                 lead.occurrences,
        frequency_pct=               round(lead.frequency_pct,               1),
        avg_excess_pct=              round(lead.avg_excess_pct,              2),
        total_excess_carbon_kgco2e=  round(lead.total_excess_carbon_kgco2e,  6),
        total_excess_cost_eur=       round(lead.total_excess_cost_eur,        4),
        optimal_window_start_hour=   lead.optimal_window_start_hour,
        optimal_window_end_hour=     lead.optimal_window_end_hour,
        estimated_carbon_saving_pct= round(lead.estimated_carbon_saving_pct, 1),
        estimated_cost_saving_pct=   round(lead.estimated_cost_saving_pct,   1),
        carbon_context=              lead.carbon_context,
        example_timestamps=          lead.example_timestamps,
    )


# ── Endpoints ────────────────────────────────────────────────────────────────────
# NOTE: /forecast/all and /forecast/sci must be defined BEFORE /forecast/{metric}
# so FastAPI does not swallow them as path parameters.

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Service liveness, model readiness, and ITU-T L.1801 compliance status."""
    state = _require_state()   # returns default node state
    return HealthResponse(
        status="ok",
        fitted_at=state["fitted_at"],
        data_rows=state["data_rows"],
        models_ready=list(state["models"].keys()),
        nodes=list(_state.keys()),
        l1801_compliance=_L1801_COMPLIANCE,
    )


@app.get("/forecast/all", response_model=dict[str, ForecastResponse], tags=["Forecast", "Scheduler"])
def forecast_all(
    horizon: int = Query(1, ge=1, le=168, description="Hours ahead to forecast (default 1 = next 60 min)"),
    node: Optional[str] = Query(None, description="Node identifier (placeholder — per-node models pending Thanos integration)"),
):
    """
    Fetch forecasts for every available metric in one request.

    Returns a dict keyed by metric name. Includes all 7 directly-fit metrics
    plus the derived `softwareCarbonIntensity` (SCI).
    Suitable for feeding downstream AI models or dashboards.
    """
    state = _require_state(node)
    result: dict[str, ForecastResponse] = {}

    for metric in state["models"]:
        r = _run_best_forecast(metric, horizon, node)
        result[metric] = _to_response(metric, horizon, r.data, r.model_used, node=node)

    # Append derived SCI
    result["softwareCarbonIntensity"] = _build_sci_response(horizon, node=node)

    return result


@app.get("/forecast/sci", response_model=ForecastResponse, tags=["Forecast", "Scheduler"])
def forecast_sci(
    horizon: int = Query(1, ge=1, le=168, description="Hours ahead to forecast (default 1 = next 60 min)"),
    include_breakdown: bool = Query(False, description="Include L.1801 carbon breakdown metadata in the response"),
    node: Optional[str] = Query(None, description="Node identifier (placeholder — per-node models pending Thanos integration)"),
):
    """
    Forecast Software Carbon Intensity (kgCO2e/req) for the next N hours.

    SCI is derived as carbonEmissions / max(functionalUnit, 1).
    Never fit directly — ratio stability requires deriving from components.

    Uncertainty intervals are propagated from both component forecasts:
      yhat_lower = carbon_lower / max(request_upper, 1)   (best-case SCI)
      yhat_upper = carbon_upper / max(request_lower, 1)   (worst-case SCI)

    Always includes `functional_unit_declaration` for L.1801 §4.3 transparency.
    Use `?include_breakdown=true` to also include the SCI formula decomposition.
    """
    return _build_sci_response(horizon, node=node, include_breakdown=include_breakdown)


@app.get("/forecast/peak", response_model=PeakForecastResponse, tags=["Forecast", "Scheduler"])
def forecast_peak(
    metric: str = Query("consumption", description="Metric to forecast. Defaults to 'consumption' for HPA provisioning."),
    horizon: int = Query(1, ge=1, le=168, description="Forecast window in hours. Peak yhat_upper across all hours is returned."),
    node: Optional[str] = Query(None, description="Node identifier (placeholder — per-node models pending Thanos integration)"),
):
    """
    Return the peak provisioning value (yhat_upper) for HPA/KEDA auto-scaling.

    Scans the full forecast horizon and returns the hour with the maximum yhat_upper —
    the conservative ceiling a Kubernetes HPA or KEDA ScaledObject should pre-provision
    for. This eliminates brownout windows by provisioning ahead of demand peaks.

    Response fields:
      provision_for       — yhat_upper at the peak hour (set as HPA metric threshold)
      expected            — yhat at the peak hour (Prophet best-estimate)
      floor               — yhat_lower at the peak hour (minimum expected)
      confidence_band_pct — (provision_for − floor) / expected × 100
      peak_hour           — ISO 8601 timestamp when the peak is forecast to occur

    Example KEDA ScaledObject uses ai_forecast_consumption_kwh_upper from
    /metrics/prometheus as the trigger metric for the same effect.
    """
    r = _run_best_forecast(metric, horizon, node)
    fc = r.data

    # Find the hour within the forecast window with the highest yhat_upper
    peak_idx = int(fc["yhat_upper"].idxmax())
    peak_row = fc.loc[peak_idx]

    provision_for = round(float(peak_row["yhat_upper"]), 6)
    expected      = round(float(peak_row["yhat"]),       6)
    floor         = round(float(peak_row["yhat_lower"]), 6)

    band_pct = round((provision_for - floor) / expected * 100, 1) if expected > 0 else 0.0

    return PeakForecastResponse(
        node=node,
        metric=metric,
        unit=UNIT_MAP.get(metric, ""),
        horizon_hours=horizon,
        provision_for=provision_for,
        expected=expected,
        floor=floor,
        confidence_band_pct=band_pct,
        peak_hour=pd.Timestamp(peak_row["ds"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
        model_used=r.model_used,
        generated_at=datetime.utcnow().isoformat() + "Z",
    )


@app.get("/forecast/{metric}", response_model=ForecastResponse, tags=["Forecast", "Scheduler"])
def forecast_metric(
    metric: str,
    horizon: int = Query(1, ge=1, le=168, description="Hours ahead to forecast (default 1 = next 60 min)"),
    include_breakdown: bool = Query(False, description="Include L.1801 carbon breakdown metadata in the response"),
    node: Optional[str] = Query(None, description="Node identifier (placeholder — per-node models pending Thanos integration)"),
):
    """
    Forecast any directly-fit metric for the next N hours.

    Available metrics: consumption, carbonEmissions, carbonIntensityFactor,
    greenConsumptionPercentage, cpuUtilization, functionalUnit, cost.

    For SCI (derived from two models) use /forecast/sci.
    For all metrics at once use /forecast/all.
    Use `?include_breakdown=true` on carbonEmissions to get L.1801 formula metadata.
    """
    r = _run_best_forecast(metric, horizon, node)
    return _to_response(metric, horizon, r.data, r.model_used, node=node, include_breakdown=include_breakdown)


@app.get("/optimal-window", response_model=OptimalWindowResponse, tags=["Scheduling", "Scheduler"])
def optimal_window(
    horizon_days: int = Query(7,  ge=1, le=14, description="Days to scan"),
    window_hours: int = Query(6,  ge=1, le=12, description="Window length in hours"),
    node: Optional[str] = Query(None, description="Node identifier (placeholder — per-node models pending Thanos integration)"),
):
    """
    Find the lowest-carbon scheduling window for each day over the next N days.

    Uses the carbon emissions forecast to identify the contiguous `window_hours`-long
    block with the lowest predicted emissions per day. Use this to defer batch
    jobs or maintenance to greener time windows.

    vs_daily_mean_pct < 0 means the window is greener than the day's average.
    """
    horizon_hours = horizon_days * 24
    carbon_result = _run_best_forecast("carbonEmissions", horizon_hours, node)
    carbon_fc = carbon_result.data.copy()
    carbon_fc["ds"]   = pd.to_datetime(carbon_fc["ds"])
    carbon_fc["date"] = carbon_fc["ds"].dt.date
    carbon_fc["hour"] = carbon_fc["ds"].dt.hour

    daily_mean = float(carbon_fc["yhat"].mean())
    windows = []

    for day, group in carbon_fc.groupby("date"):
        group = group.reset_index(drop=True)
        n = len(group)

        best_start = 0
        best_avg   = float("inf")
        for start in range(n - window_hours + 1):
            avg = float(group.iloc[start : start + window_hours]["yhat"].mean())
            if avg < best_avg:
                best_avg   = avg
                best_start = int(group.iloc[start]["hour"])

        vs_mean = round((best_avg - daily_mean) / daily_mean * 100, 1)
        windows.append(SchedulingWindow(
            date=str(day),
            start_hour=best_start,
            end_hour=(best_start + window_hours) % 24,   # wrap midnight; matches anomaly_detector convention
            avg_predicted_carbon=round(best_avg, 6),
            vs_daily_mean_pct=vs_mean,
            unit="kgCO2e/h",
        ))

    return OptimalWindowResponse(
        node=node,
        model_used=carbon_result.model_used,
        horizon_days=horizon_days,
        window_hours=window_hours,
        generated_at=datetime.utcnow().isoformat() + "Z",
        windows=windows,
    )


@app.post("/data", response_model=DataIngestionResponse, status_code=202, tags=["Data", "Scheduler"])
def ingest_data(
    payload: DataIngestionRequest,
    background_tasks: BackgroundTasks,
    node: Optional[str] = Query(None, description="Node identifier. Rows are stored under this node; a separate Prophet refit is scheduled for that node's dataset. Unknown nodes are initialised automatically."),
):
    """
    Ingest new measurement rows. Returns 202 immediately; Prophet refit runs async.

    Required fields: `ds` (ISO 8601) and `consumption` (kWh/h). All others optional.
    Use `?node=<name>` to scope data and refit to a specific node.
    """
    node_key = node if node is not None else _DEFAULT_NODE
    default_state = _require_state()   # raises 503 if not ready

    # First time seeing this node — bootstrap state from the default node so
    # forecasts are immediately available before the node-specific refit finishes.
    if node_key not in _state:
        _state[node_key] = {
            "df":                         default_state["df"].copy(),   # context for regressors until node-specific data arrives
            "models":                     dict(default_state["models"]),
            "config":                     default_state["config"],
            "insample_forecasts":         {},
            "timesfm_models":             dict(default_state.get("timesfm_models", {})),
            "timesfm_insample_forecasts": {},
            "fitted_at":                  datetime.utcnow().isoformat() + "Z",
            "data_rows":                  0,
        }
        log.info("New node '%s' initialised — borrowing default models until refit completes", node_key)

    state = _state[node_key]

    # Build a DataFrame from the incoming rows, preserving only non-None fields
    records = [row.model_dump(exclude_none=True) for row in payload.rows]
    new_df = pd.DataFrame(records)
    new_df["ds"] = pd.to_datetime(new_df["ds"])

    # Merge into the node's dataset, keeping chronological order
    updated_df = (
        pd.concat([state["df"], new_df], ignore_index=True)
        .sort_values("ds")
        .reset_index(drop=True)
    )
    state["df"] = updated_df
    state["data_rows"] = len(updated_df)

    # Update TimesFM history contexts (instant — only stores numpy array)
    if _TIMESFM_AVAILABLE:
        for metric, tfm in state.get("timesfm_models", {}).items():
            if metric in updated_df.columns:
                tfm.fit(updated_df, metric)
        state["timesfm_insample_forecasts"] = {}

    # Flush Prophet in-sample cache (stale after new data)
    state["insample_forecasts"] = {}

    # Refit Prophet models asynchronously for this node — old models keep serving
    background_tasks.add_task(_refit_models_in_background, updated_df.copy(), node_key)

    return DataIngestionResponse(
        rows_accepted=len(payload.rows),
        total_rows=len(updated_df),
        models_status="refit_scheduled",
    )


@app.get("/anomalies", response_model=AnomalyResponse, tags=["Anomaly Detection", "Frontend"])
def anomalies_endpoint(
    metric:       str = Query(..., description="Metric to analyse, e.g. 'consumption' or 'carbonEmissions'"),
    lookback_days: int = Query(30, ge=1, le=365, description="Days of historical data to scan"),
    direction: Literal["excess", "deficit", "both"] = Query(
        "excess", description="'excess' → actual > yhat_upper; 'deficit' → actual < yhat_lower; 'both'"
    ),
    node: Optional[str] = Query(None, description="Node identifier (placeholder — per-node models pending Thanos integration)"),
):
    """
    Detect point anomalies for a metric over the past N days.

    Compares actual historical values against the Prophet in-sample 95% confidence
    interval. An anomaly occurs when actual > yhat_upper (excess) or
    actual < yhat_lower (deficit).

    In-sample forecasts are lazy-computed on the first call per metric and cached
    for subsequent requests — no re-fitting required.

    Example: GET /anomalies?metric=consumption&lookback_days=30&direction=excess
    """
    state = _require_state(node)
    df = state["df"].copy()
    df["ds"] = pd.to_datetime(df["ds"])

    # Validate metric before the expensive in-sample forecast call
    if metric not in state["models"]:
        available = list(state["models"].keys())
        raise HTTPException(
            status_code=404,
            detail=f"No model for '{metric}'. Available: {available}",
        )

    # Restrict to the lookback window (most-recent N days of history)
    cutoff = df["ds"].max() - pd.Timedelta(days=lookback_days)
    df_window = df[df["ds"] > cutoff].copy()

    # Retrieve (or lazily compute) in-sample forecast, then filter to window
    df_forecast = _get_insample_forecast(metric, node).copy()
    df_forecast["ds"] = pd.to_datetime(df_forecast["ds"])
    df_forecast_window = df_forecast[df_forecast["ds"] > cutoff]

    anomalies = detect_point_anomalies(df_window, df_forecast_window, metric, direction=direction)

    # --- Dual-model confidence scoring ---
    if _TIMESFM_AVAILABLE and metric in state.get("timesfm_models", {}):
        tfm_fc = _get_timesfm_insample_forecast(metric, node).copy()
        tfm_fc["ds"] = pd.to_datetime(tfm_fc["ds"])
        tfm_fc_window = tfm_fc[tfm_fc["ds"] > cutoff].copy()

        # Run the same anomaly detector against TimesFM bounds instead of Prophet bounds.
        # This produces a fully independent set of flagged timestamps.
        tfm_anomalies: list = (
            detect_point_anomalies(df_window, tfm_fc_window, metric, direction=direction)
            if not tfm_fc_window.empty else []
        )

        # Build lookup sets (floor to hour to handle any sub-second rounding in joins)
        tfm_flagged_ts     = {pd.Timestamp(a.ds).floor("h") for a in tfm_anomalies}
        prophet_flagged_ts = {pd.Timestamp(a.ds).floor("h") for a in anomalies}

        points: list[AnomalyPoint] = []

        # Prophet anomalies: "high" if TimesFM also flags the same hour, else "prophet-only"
        for a in anomalies:
            a_ts   = pd.Timestamp(a.ds).floor("h")
            is_tfm = a_ts in tfm_flagged_ts
            conf   = "high" if is_tfm else "prophet-only"
            points.append(_anomaly_to_point(a, timesfm_flagged=is_tfm, confidence=conf))

        # TimesFM-only anomalies: Prophet's baseline was not breached but TimesFM signals
        # something unexpected — a weaker but independent signal worth surfacing.
        for a in tfm_anomalies:
            if pd.Timestamp(a.ds).floor("h") not in prophet_flagged_ts:
                points.append(_anomaly_to_point(a, timesfm_flagged=True, confidence="timesfm-only"))

        # Sort: "high" (consensus) → "prophet-only" → "timesfm-only", then by time
        _conf_rank = {"high": 0, "prophet-only": 1, "timesfm-only": 2}
        points.sort(key=lambda p: (_conf_rank[p.confidence], p.ds))
    else:
        points = [_anomaly_to_point(a) for a in anomalies]

    return AnomalyResponse(
        node=           node,
        metric=         metric,
        unit=           UNIT_MAP.get(metric, ""),
        direction=      direction,
        lookback_days=  lookback_days,
        total_anomalies=len(points),   # includes TimesFM-only anomalies when available
        generated_at=   datetime.utcnow().isoformat() + "Z",
        anomalies=      points,
    )


@app.get("/investigation-leads", response_model=InvestigationLeadsResponse, tags=["Anomaly Detection", "Frontend"])
def investigation_leads_endpoint(
    lookback_days:   int = Query(30, ge=1, le=365, description="Days of historical data to scan"),
    top_n:           int = Query(5,  ge=1, le=20,  description="Maximum leads to return"),
    min_occurrences: int = Query(3,  ge=1,          description="Minimum occurrences to form a recurring pattern"),
    node: Optional[str] = Query(None, description="Node identifier (placeholder — accepted for API consistency but not echoed; per-node patterns pending Thanos integration)"),
):
    """
    Rank the top recurring excess anomaly patterns with rescheduling savings estimates.

    Groups anomalies by (hour_of_day × weekday/weekend), ranks by total excess carbon.
    Each lead includes the optimal low-carbon window and estimated carbon/cost savings.
    """
    state = _require_state(node)
    df = state["df"].copy()
    df["ds"] = pd.to_datetime(df["ds"])

    cutoff = df["ds"].max() - pd.Timedelta(days=lookback_days)
    df_window = df[df["ds"] > cutoff].copy()

    # Collect excess anomalies across the two primary carbon-relevant metrics
    all_anomalies = []
    for metric in ["consumption", "carbonEmissions"]:
        if metric not in state["models"]:
            continue
        df_forecast = _get_insample_forecast(metric, node).copy()
        df_forecast["ds"] = pd.to_datetime(df_forecast["ds"])
        df_forecast_window = df_forecast[df_forecast["ds"] > cutoff]
        metric_anomalies = detect_point_anomalies(
            df_window, df_forecast_window, metric, direction="excess"
        )
        all_anomalies.extend(metric_anomalies)

    patterns = find_recurring_patterns(all_anomalies, df_window, min_occurrences=min_occurrences)
    leads    = build_investigation_leads(patterns, df_window, top_n=top_n)

    return InvestigationLeadsResponse(
        lookback_days=lookback_days,
        top_n=        top_n,
        generated_at= datetime.utcnow().isoformat() + "Z",
        leads=        [_lead_to_point(lead) for lead in leads],
    )


def _metric_to_prometheus_block(
    prom_name: str,
    help_text: str,
    fc: pd.DataFrame,
    node_label: str,
    model_used: _ModelKey,
    horizon: int,
) -> str:
    """Build a Prometheus text-format block for a single metric's future forecast rows.

    Emits three metric families: point forecast (yhat), upper bound (yhat_upper),
    and lower bound (yhat_lower). This exposes the full confidence band so that
    KEDA ScaledObjects can use yhat_upper as the HPA provisioning threshold.

    Each row becomes a separate time series with `forecast_ts` as a label so that
    future timestamps remain queryable even though Prometheus scraping uses wall time.
    `horizon` encodes the forecast window length (e.g. "6h") for PromQL filtering.
    `model_used` is exposed as a label so dashboards can filter or alert by routing.
    """
    horizon_label = f"{horizon}h"

    def _series_block(name: str, col: str, description_suffix: str) -> list[str]:
        lines = [
            f"# HELP {name} {help_text}{description_suffix}",
            f"# TYPE {name} gauge",
        ]
        for _, row in fc.iterrows():
            ts = pd.Timestamp(row["ds"]).strftime("%Y-%m-%dT%H:%M:%SZ")
            lines.append(
                f'{name}{{node="{node_label}",forecast_ts="{ts}",'
                f'horizon="{horizon_label}",model_used="{model_used}"}}'
                f" {row[col]:.6f}"
            )
        return lines

    parts = (
        _series_block(prom_name,              "yhat",       "")
        + [""]
        + _series_block(f"{prom_name}_upper", "yhat_upper", " (upper confidence bound — use as HPA threshold)")
        + [""]
        + _series_block(f"{prom_name}_lower", "yhat_lower", " (lower confidence bound)")
    )
    return "\n".join(parts)


@app.get(
    "/metrics/prometheus",
    tags=["Prometheus", "Scheduler"],
    summary="Prometheus text format forecast for HPA integration",
)
def prometheus_metrics(
    horizon: int = Query(1, ge=1, le=168, description="Forecast horizon (hours)"),
    node: Optional[str] = Query(None, description="Node identifier"),
) -> Response:
    """
    Expose consumption and carbonEmissions forecasts in Prometheus text format.

    Designed for EVIDEN's HPA (Horizontal Pod Autoscaler) which scrapes Prometheus.
    Because Prometheus scraping uses the current time as the native timestamp,
    future forecast timestamps are encoded as label values (`forecast_ts`) so that
    each forecast point becomes a queryable time series.

    Returns text/plain in Prometheus exposition format (version 0.0.4).
    """
    state = _require_state(node)
    node_label = node if node is not None else "global"

    _PROM_METRICS = [
        ("consumption",     "ai_forecast_consumption_kwh",        "Predicted energy consumption (kWh/h)"),
        ("carbonEmissions", "ai_forecast_carbon_emissions_kgco2e", "Predicted carbon emissions (kgCO2e/h)"),
    ]

    blocks = []
    for metric, prom_name, help_text in _PROM_METRICS:
        if metric not in state["models"]:
            continue
        r = _run_best_forecast(metric, horizon, node)   # honours formula→TimesFM→Prophet routing
        block = _metric_to_prometheus_block(prom_name, help_text, r.data, node_label, r.model_used, horizon)
        blocks.append(block)

    content = "\n\n".join(blocks) + "\n"
    return Response(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")
