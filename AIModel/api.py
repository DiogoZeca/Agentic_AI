"""
Energy & Carbon Forecasting Service

Prophet-based REST API that loads all 15 schema metrics, fits 7 Prophet models
at startup, and exposes forecasts consumable by schedulers, dashboards, or agents.

Usage (Docker):
    docker compose up --build

Endpoints:
    GET /health                              — model status and readiness
    GET /forecast/all?horizon=24             — all metrics in one response
    GET /forecast/sci?horizon=24             — SCI (derived, kgCO2e/req)
    GET /forecast/{metric}?horizon=24        — single metric forecast
    GET /optimal-window?horizon_days=7       — best low-carbon scheduling windows

Available metrics for /forecast/{metric}:
    consumption, carbonEmissions, carbonIntensityFactor,
    greenConsumptionPercentage, cpuUtilization, functionalUnit, cost
"""
import os
import logging
import warnings
from contextlib import asynccontextmanager
from datetime import datetime

# Suppress noisy loggers before prophet imports
logging.getLogger("prophet").setLevel(logging.WARNING)
_cmdstan = logging.getLogger("cmdstanpy")
_cmdstan.setLevel(logging.ERROR)
_cmdstan.addHandler(logging.NullHandler())
_cmdstan.propagate = False
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from data_generator import generate_energy_carbon_data
from data_loader import load_and_validate, build_pipeline_config
from prophet_model import EnergyProphet

log = logging.getLogger("api")

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

# ── App state ────────────────────────────────────────────────────────────────────

_state: dict = {}


# ── Lifespan ─────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data and fit Prophet models on startup."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    # 1. Load or generate data
    if os.path.exists(DATA_PATH):
        log.info("Loading data from %s", DATA_PATH)
        df = load_and_validate(DATA_PATH)
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

    _state.update({
        "df":        df,
        "models":    models,
        "config":    config,
        "fitted_at": datetime.utcnow().isoformat() + "Z",
        "data_rows": len(df),
    })
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
    version="0.2.0",
    lifespan=lifespan,
)


# ── Response models ──────────────────────────────────────────────────────────────

class ForecastPoint(BaseModel):
    ds:         str
    yhat:       float
    yhat_lower: float
    yhat_upper: float


class ForecastResponse(BaseModel):
    metric:        str
    unit:          str
    horizon_hours: int
    generated_at:  str
    predictions:   list[ForecastPoint]


class SchedulingWindow(BaseModel):
    date:                 str
    start_hour:           int
    end_hour:             int
    avg_predicted_carbon: float
    vs_daily_mean_pct:    float   # negative = greener than the day's average
    unit:                 str


class OptimalWindowResponse(BaseModel):
    horizon_days: int
    window_hours: int
    generated_at: str
    windows:      list[SchedulingWindow]


class HealthResponse(BaseModel):
    status:       str
    fitted_at:    str
    data_rows:    int
    models_ready: list[str]


# ── Helpers ──────────────────────────────────────────────────────────────────────

def _require_state() -> dict:
    if not _state:
        raise HTTPException(status_code=503, detail="Models not ready yet.")
    return _state


def _run_forecast(metric: str, horizon: int) -> pd.DataFrame:
    """Return the future `horizon` rows from Prophet for `metric`."""
    state = _require_state()
    if metric not in state["models"]:
        available = list(state["models"].keys())
        raise HTTPException(
            status_code=404,
            detail=f"No model for '{metric}'. Available: {available}",
        )
    model: EnergyProphet = state["models"][metric]
    forecast = model.predict(periods=horizon, future_df=state["df"])
    return forecast.tail(horizon).reset_index(drop=True)


def _to_response(metric: str, horizon: int, fc: pd.DataFrame) -> ForecastResponse:
    predictions = [
        ForecastPoint(
            ds=str(row["ds"]),
            yhat=      round(float(row["yhat"]),       6),
            yhat_lower=round(float(row["yhat_lower"]), 6),
            yhat_upper=round(float(row["yhat_upper"]), 6),
        )
        for _, row in fc.iterrows()
    ]
    return ForecastResponse(
        metric=metric,
        unit=UNIT_MAP.get(metric, ""),
        horizon_hours=horizon,
        generated_at=datetime.utcnow().isoformat() + "Z",
        predictions=predictions,
    )


def _build_sci_response(horizon: int) -> ForecastResponse:
    """Derive SCI with propagated uncertainty from two component forecasts."""
    carbon_fc  = _run_forecast("carbonEmissions", horizon)
    request_fc = _run_forecast("functionalUnit",  horizon)

    predictions = []
    for (_, c_row), (_, r_row) in zip(carbon_fc.iterrows(), request_fc.iterrows()):
        r_hat   = max(float(r_row["yhat"]),       1.0)
        r_upper = max(float(r_row["yhat_upper"]), 1.0)
        r_lower = max(float(r_row["yhat_lower"]), 1.0)

        predictions.append(ForecastPoint(
            ds=str(c_row["ds"]),
            yhat=       round(float(c_row["yhat"])       / r_hat,   8),
            yhat_lower= round(float(c_row["yhat_lower"]) / r_upper, 8),  # best SCI
            yhat_upper= round(float(c_row["yhat_upper"]) / r_lower, 8),  # worst SCI
        ))

    return ForecastResponse(
        metric="softwareCarbonIntensity",
        unit=UNIT_MAP["softwareCarbonIntensity"],
        horizon_hours=horizon,
        generated_at=datetime.utcnow().isoformat() + "Z",
        predictions=predictions,
    )


# ── Endpoints ────────────────────────────────────────────────────────────────────
# NOTE: /forecast/all and /forecast/sci must be defined BEFORE /forecast/{metric}
# so FastAPI does not swallow them as path parameters.

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Service liveness and model readiness check."""
    state = _require_state()
    return HealthResponse(
        status="ok",
        fitted_at=state["fitted_at"],
        data_rows=state["data_rows"],
        models_ready=list(state["models"].keys()),
    )


@app.get("/forecast/all", response_model=dict[str, ForecastResponse], tags=["Forecast"])
def forecast_all(
    horizon: int = Query(24, ge=1, le=168, description="Hours ahead to forecast"),
):
    """
    Fetch forecasts for every available metric in one request.

    Returns a dict keyed by metric name. Includes all 7 directly-fit metrics
    plus the derived `softwareCarbonIntensity` (SCI).
    Suitable for feeding downstream AI models or dashboards.
    """
    state = _require_state()
    result: dict[str, ForecastResponse] = {}

    for metric in state["models"]:
        fc = _run_forecast(metric, horizon)
        result[metric] = _to_response(metric, horizon, fc)

    # Append derived SCI
    result["softwareCarbonIntensity"] = _build_sci_response(horizon)

    return result


@app.get("/forecast/sci", response_model=ForecastResponse, tags=["Forecast"])
def forecast_sci(
    horizon: int = Query(24, ge=1, le=168, description="Hours ahead to forecast"),
):
    """
    Forecast Software Carbon Intensity (kgCO2e/req) for the next N hours.

    SCI is derived as carbonEmissions / max(functionalUnit, 1).
    Never fit directly — ratio stability requires deriving from components.

    Uncertainty intervals are propagated from both component forecasts:
      yhat_lower = carbon_lower / max(request_upper, 1)   (best-case SCI)
      yhat_upper = carbon_upper / max(request_lower, 1)   (worst-case SCI)
    """
    return _build_sci_response(horizon)


@app.get("/forecast/{metric}", response_model=ForecastResponse, tags=["Forecast"])
def forecast_metric(
    metric: str,
    horizon: int = Query(24, ge=1, le=168, description="Hours ahead to forecast"),
):
    """
    Forecast any directly-fit metric for the next N hours.

    Available metrics: consumption, carbonEmissions, carbonIntensityFactor,
    greenConsumptionPercentage, cpuUtilization, functionalUnit, cost.

    For SCI (derived from two models) use /forecast/sci.
    For all metrics at once use /forecast/all.
    """
    fc = _run_forecast(metric, horizon)
    return _to_response(metric, horizon, fc)


@app.get("/optimal-window", response_model=OptimalWindowResponse, tags=["Scheduling"])
def optimal_window(
    horizon_days: int = Query(7,  ge=1, le=14, description="Days to scan"),
    window_hours: int = Query(6,  ge=1, le=12, description="Window length in hours"),
):
    """
    Find the lowest-carbon scheduling window for each day over the next N days.

    Uses the carbon emissions forecast to identify the contiguous `window_hours`-long
    block with the lowest predicted emissions per day. Use this to defer batch
    jobs or maintenance to greener time windows.

    vs_daily_mean_pct < 0 means the window is greener than the day's average.
    """
    horizon_hours = horizon_days * 24
    carbon_fc = _run_forecast("carbonEmissions", horizon_hours)
    carbon_fc = carbon_fc.copy()
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
            end_hour=best_start + window_hours,
            avg_predicted_carbon=round(best_avg, 6),
            vs_daily_mean_pct=vs_mean,
            unit="kgCO2e/h",
        ))

    return OptimalWindowResponse(
        horizon_days=horizon_days,
        window_hours=window_hours,
        generated_at=datetime.utcnow().isoformat() + "Z",
        windows=windows,
    )
