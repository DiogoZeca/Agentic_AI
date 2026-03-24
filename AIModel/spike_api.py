"""FastAPI service for CPU spike prediction.

Loads model artefacts once at startup (lifespan context), then serves
per-request inference via predict_with_artifacts() — no disk reads per call.

Endpoints
---------
  GET  /health   — liveness probe (always 200 if process is alive)
  GET  /ready    — readiness probe (200 only when artefacts are loaded)
  POST /predict  — run inference on a rolling window of CPU observations

Example request (POST /predict):
  {
    "model_dir": "/app/data/full_run/models/spike",
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
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from predict_spike import _Artifacts, _load_artifacts, predict_with_artifacts

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Global artefacts store (populated in lifespan) ────────────────────────────

_artifacts: _Artifacts | None = None


# ── Lifespan (startup + shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artefacts on startup; release on shutdown."""
    global _artifacts
    import os
    model_dir = os.environ.get("MODEL_DIR", "/app/data/full_run/models/spike")
    log.info("Loading artefacts from %s …", model_dir)
    try:
        _artifacts = _load_artifacts(__import__("pathlib").Path(model_dir))
        log.info("Artefacts loaded — service ready.")
    except Exception as exc:
        log.error("Failed to load artefacts: %s", exc)
        # Don't crash the process — /ready will return 503 until artefacts load.
    yield
    log.info("Shutting down.")


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
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    # jsonable_encoder converts numpy types (int64, float32, bool_) to Python
    # primitives — FastAPI does not do this automatically for nested dicts.
    return jsonable_encoder(result)
