"""FastAPI service for CPU spike prediction.

Loads model artefacts once at startup (lifespan context), then serves
per-request inference via predict_with_artifacts() — no disk reads per call.

Endpoints
---------
  GET  /health   — liveness probe (always 200 if process is alive)
  GET  /ready    — readiness probe (200 only when artefacts are loaded)
  POST /predict  — full per-machine inference (probabilities, imminence, SHAP)
  POST /summary  — scheduler-facing summary: spike counts per horizon + per-node status

Both /predict and /summary share the same EWMA and debounce state. Use one
endpoint consistently per deployment cycle — calling both in the same cycle
will double-update the per-machine smoothing and consecutive-alarm counters.

Example request (POST /predict or POST /summary):
  {
    "rows": [
      {"machine_id": 1, "bucket": 100, "time_us": 30000000000,
       "total_cpu": 0.42, "peak_cpu": 0.61, "total_mem": 1.2,
       "peak_mem": 2.1, "disk_io": 0.0, "n_tasks": 4},
      ...
    ]
  }
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from spike.predict import _Artifacts, _load_artifacts, predict_with_artifacts

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Global state (populated in lifespan / updated on every request) ───────────

_artifacts: _Artifacts | None = None

# Per-machine consecutive-spike counter.  Resets to 0 when a machine does not
# spike; increments on each spiking prediction.  Kept in memory — resets on
# restart, which is acceptable: a short gap after a restart simply requires
# the machine to spike again before the alarm fires.
_consecutive_alarms: dict[int, int] = {}

# Minimum number of consecutive spiking predictions before is_spike fires.
# Prevents single-shot noise from triggering scheduler actions.
# Override via ALARM_MIN_CONSECUTIVE env var; set to 1 to disable debouncing.
_ALARM_MIN_CONSECUTIVE: int = int(os.environ.get("ALARM_MIN_CONSECUTIVE", "2"))

# Per-machine EWMA state: smoothed p_spike = p_moderate + p_severe.
# Initialised to the raw score on first encounter — no warm-up period needed.
# Resets on restart (same trade-off as _consecutive_alarms).
_ewma_scores: dict[int, float] = {}

# Exponential smoothing factor.  0 < alpha <= 1.
#   alpha = 1.0  →  no smoothing (EWMA = raw score, EWMA filter disabled).
#   alpha = 0.5  →  each cycle contributes 50 %; last 3 cycles carry ~87.5 %.
#   alpha = 0.3  →  slower response; 10-cycle half-life.
# Override via EWMA_ALPHA env var.
_EWMA_ALPHA: float = max(0.0, min(1.0, float(os.environ.get("EWMA_ALPHA", "0.5"))))


# ── Lifespan (startup + shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artefacts on startup; release on shutdown."""
    global _artifacts
    import os
    model_dir = os.environ.get("MODEL_DIR", "/app/data/full_run/spike")
    log.info("Loading artefacts from %s …", model_dir)
    # Let exceptions propagate — Starlette sends lifespan.startup.failed to
    # uvicorn, which sets should_exit=True and terminates with a non-zero exit
    # code.  A container restart policy then handles recovery.  Swallowing the
    # exception leaves the process alive but permanently broken (_artifacts=None),
    # which fools liveness probes into never restarting the container.
    _artifacts = _load_artifacts(__import__("pathlib").Path(model_dir))
    log.info("Artefacts loaded — service ready.")
    yield
    log.info("Shutting down.")


# ── EWMA smoothing ────────────────────────────────────────────────────────────


def _apply_ewma_smoothing(
    result: dict,
    state:  dict[int, float],
    alpha:  float,
) -> dict:
    """Dampen single-cycle probability spikes via per-machine EWMA.

    Maintains a per-machine exponentially weighted moving average of
    ``p_spike = p_moderate + p_severe`` across requests.  When the smoothed
    score falls below the trained alarm threshold, the alarm fields are
    suppressed — EWMA can only *remove* alarms, never *create* ones.

    The smoothed score is written to ``p_spike_smoothed`` for transparency
    (useful for dashboard and scheduler debugging).

    Cold-start predictions are skipped — they carry no probability output.

    Parameters
    ----------
    result : envelope dict returned by predict_with_artifacts().
    state  : mutable EWMA score dict shared across requests
             (machine_id → smoothed p_spike).
    alpha  : smoothing factor in (0, 1].  1.0 = no smoothing (pass-through).
    """
    for pred in result["predictions"]:
        mid = pred["machine_id"]

        if pred["status"] == "cold_start" or pred.get("p_moderate") is None:
            pred["p_spike_smoothed"] = None
            continue

        p_spike_raw = pred["p_moderate"] + pred["p_severe"]

        # First encounter: seed the EWMA with the raw score so there is no
        # artificial warm-up suppression on machines the API has not seen yet.
        prev_smoothed       = state.get(mid, p_spike_raw)
        smoothed            = alpha * p_spike_raw + (1.0 - alpha) * prev_smoothed
        state[mid]          = smoothed
        pred["p_spike_smoothed"] = round(smoothed, 6)

        # Suppress alarm when the smoothed signal is below the alarm threshold.
        # Using the per-prediction alarm_threshold keeps this consistent with
        # how is_severe was originally determined during training.
        alarm_threshold = pred.get("alarm_threshold", 0.5)
        if smoothed < alarm_threshold:
            pred["is_spike"]           = False
            pred["is_severe"]          = False
            pred["recommended_action"] = "normal"

    return result


# ── Alarm debounce ────────────────────────────────────────────────────────────


def _apply_alarm_debounce(
    result:          dict,
    state:           dict[int, int],
    min_consecutive: int,
) -> dict:
    """Suppress single-shot alarms; require min_consecutive spikes to fire.

    Mutates ``state`` (machine_id → consecutive spike count) in place and
    adds a ``consecutive_alarms`` field to every prediction dict.

    Suppression only overrides the actionable fields consumed by the scheduler
    (``is_spike``, ``is_severe``, ``recommended_action``).  Raw probabilities
    and the ``imminence`` detail block are left unchanged so the dashboard can
    still render the underlying model signal.

    Parameters
    ----------
    result          : envelope dict returned by predict_with_artifacts().
    state           : mutable counter dict shared across requests.
    min_consecutive : number of consecutive spiking predictions required
                      before the alarm fires.  1 = no suppression.
    """
    for pred in result["predictions"]:
        mid = pred["machine_id"]

        if pred["status"] == "cold_start":
            pred["consecutive_alarms"] = 0
            continue

        if pred.get("is_spike"):
            state[mid] = state.get(mid, 0) + 1
        else:
            state[mid] = 0

        count = state[mid]
        pred["consecutive_alarms"] = count

        # Suppress until the streak meets the minimum threshold.
        # count == 0 means no spike this cycle — nothing to suppress.
        if 0 < count < min_consecutive:
            pred["is_spike"]           = False
            pred["is_severe"]          = False
            pred["recommended_action"] = "normal"

    return result


# ── Summary builder ───────────────────────────────────────────────────────────

_SEVERITY_LABELS: dict[int, str] = {0: "no_spike", 1: "moderate", 2: "severe"}


def _build_summary_response(result: dict) -> dict:
    """Aggregate a full predict_with_artifacts result into the scheduler summary format.

    summary.spikes_in_Xm  = machines predicted to spike within X minutes.
    summary.nodes_requiring_action = machines with recommended_action in
                                     {preempt_now, migrate_jobs} — direct
                                     cluster-level signal for the scheduler.

    per_node fields for scheduler integration:
      spike_60m        — 60m severity model verdict (sustained spike risk)
      spike_imminent   — True if any binary horizon model (15m/30m/45m) alarmed
      alarm_source     — which model drove recommended_action
      scheduler_score  — 0-100; higher = safer; maps directly to K8s Score plugin
      recommended_action — what the scheduler should do with this node

    Cold-start machines are excluded from summary counts but present in per_node.
    """
    counts: dict[str, int] = {
        "spikes_in_15m": 0,
        "spikes_in_30m": 0,
        "spikes_in_45m": 0,
        "spikes_in_60m": 0,
    }
    per_node = []

    for pred in result["predictions"]:
        imm    = pred.get("imminence") or {}
        status = pred.get("status", "")

        if status != "cold_start":
            if imm.get("15m", {}).get("is_spike"):
                counts["spikes_in_15m"] += 1
            if imm.get("30m", {}).get("is_spike"):
                counts["spikes_in_30m"] += 1
            if imm.get("45m", {}).get("is_spike"):
                counts["spikes_in_45m"] += 1
            if pred.get("is_spike"):
                counts["spikes_in_60m"] += 1

        # Earliest alarming horizon — determines scheduler lead time.
        horizon: str | None = None
        if status != "cold_start":
            for h in ("15m", "30m", "45m"):
                if imm.get(h, {}).get("is_spike"):
                    horizon = h
                    break
            if horizon is None and pred.get("is_spike"):
                horizon = "60m"

        # spike_imminent: any short-horizon binary model alarmed.
        spike_imminent: bool | None = None
        if status != "cold_start":
            spike_imminent = any(
                imm.get(h, {}).get("is_spike") for h in ("15m", "30m", "45m")
            )

        # alarm_source: which model drove recommended_action — removes ambiguity
        # when spike_60m and spike_imminent disagree.
        if status == "cold_start":
            alarm_source: str | None = None
        elif horizon in ("15m", "30m", "45m"):
            alarm_source = f"binary_{horizon}"
        elif pred.get("is_spike"):
            alarm_source = "60m_severity"
        else:
            alarm_source = "none"

        # scheduler_score: 0-100 (100 = safest). Plugs directly into the K8s
        # Score plugin — no transformation required by the integrator.
        p_smoothed = pred.get("p_spike_smoothed")
        scheduler_score: int | None = (
            round(100 * (1.0 - p_smoothed)) if p_smoothed is not None else None
        )

        sev_cls = pred.get("severity_class")
        per_node.append({
            "machine_id":         pred["machine_id"],
            "status":             status,
            "spike_60m":          pred.get("is_spike"),
            "spike_imminent":     spike_imminent,
            "severity":           _SEVERITY_LABELS.get(sev_cls) if sev_cls is not None else None,
            "alarm_source":       alarm_source,
            "scheduler_score":    scheduler_score,
            "horizon":            horizon,
            "recommended_action": pred.get("recommended_action"),
            "p_spike_smoothed":   pred.get("p_spike_smoothed"),
            "consecutive_alarms": pred.get("consecutive_alarms"),
        })

    counts["nodes_requiring_action"] = sum(
        1 for n in per_node
        if n.get("recommended_action") in ("preempt_now", "migrate_jobs")
    )

    return {
        "summary":             counts,
        "per_node":            per_node,
        "predicted_at":        result.get("predicted_at"),
        "machines_total":      result.get("machines_total"),
        "machines_predicted":  result.get("machines_predicted"),
        "machines_cold_start": result.get("machines_cold_start"),
    }


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "CPU Spike Predictor",
    description = "XGBoost severity-tier classifier for CPU spike prediction.",
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ── Pydantic models ───────────────────────────────────────────────────────────

class ObservationRow(BaseModel):
    machine_id: int
    bucket:     int
    time_us:    int
    total_cpu:  float
    peak_cpu:   float
    total_mem:  float
    peak_mem:   float
    disk_io:    float
    n_tasks:    int


class PredictRequest(BaseModel):
    rows: list[ObservationRow] = Field(..., min_length=1)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.model_dump() for r in self.rows])


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Liveness probe — always 200 if the process is running."""
    return {"status": "alive"}


@app.get("/ready", tags=["ops"])
def ready() -> dict[str, Any]:
    """Readiness probe — 503 until artefacts are loaded."""
    if _artifacts is None:
        raise HTTPException(status_code=503, detail="Artefacts not yet loaded.")
    return {
        "status":          "ready",
        "horizons":        sorted(_artifacts.boosters.keys()),
        "known_machines":  int(len(_artifacts.thresholds)),
        "feature_count":   int(len(_artifacts.feature_cols)),
        "trained_at":      _artifacts.trained_at,
        "calibrated":      _artifacts.calibrators_60m is not None,
    }


@app.post("/predict", tags=["inference"])
def predict(request: PredictRequest) -> dict[str, Any]:
    """Run spike prediction on a rolling 120-min CPU window.

    Send the last 24 buckets (120 min) per machine.  Returns per-machine
    severity probabilities (p_no_spike, p_moderate, p_severe), severity_class
    (0/1/2), and is_severe flag based on the learned alarm threshold.

    Machines with fewer than 12 buckets are returned with null probabilities
    and status=cold_start.
    """
    if _artifacts is None:
        raise HTTPException(status_code=503, detail="Artefacts not yet loaded.")

    try:
        input_df = request.to_dataframe()
        result   = predict_with_artifacts(input_df, _artifacts)
        result   = _apply_ewma_smoothing(result, _ewma_scores, _EWMA_ALPHA)
        result   = _apply_alarm_debounce(result, _consecutive_alarms, _ALARM_MIN_CONSECUTIVE)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    # jsonable_encoder converts numpy types (int64, float32, bool_) to Python
    # primitives — FastAPI does not do this automatically for nested dicts.
    return jsonable_encoder(result)


@app.post("/summary", tags=["inference"])
def summary(request: PredictRequest) -> dict[str, Any]:
    """Scheduler-facing spike summary.

    Runs the same inference pipeline as /predict and returns:
      - summary: spike count per horizon (0 = no alarms — best score)
      - per_node: per-machine status, severity, and earliest alarming horizon

    The scheduler uses summary.spikes_in_Xm as a cluster health score and
    per_node to decide which specific machines to migrate or defer work from.

    Send the last 24 buckets (120 min) per machine.
    Cold-start machines are excluded from summary counts but present in per_node.
    """
    if _artifacts is None:
        raise HTTPException(status_code=503, detail="Artefacts not yet loaded.")

    try:
        input_df = request.to_dataframe()
        result   = predict_with_artifacts(input_df, _artifacts)
        result   = _apply_ewma_smoothing(result, _ewma_scores, _EWMA_ALPHA)
        result   = _apply_alarm_debounce(result, _consecutive_alarms, _ALARM_MIN_CONSECUTIVE)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    return jsonable_encoder(_build_summary_response(result))
