"""CPU Power Spike Prediction API.

Loads a pre-trained artefact at startup and serves spike predictions:
given CPU% → predicted Power (W) + is_spike flag via O(1) lookup table.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from cpu_power_model import CpuPowerModel

log = logging.getLogger("cpu_power_api")
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

MODELS_PATH = os.environ.get("MODELS_PATH", "models/winner.json")

_model: Optional[CpuPowerModel] = None


def _require_model() -> CpuPowerModel:
    if _model is None:
        raise RuntimeError("Model not initialised")
    return _model


# ── App ────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = CpuPowerModel.load(MODELS_PATH)
    log.info("Model ready — %d CPU types", len(_model.available_types()))
    yield


app = FastAPI(title="CPU Power Spike Prediction", version="1.0.0", lifespan=lifespan)


# ── Response models ────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    cpu_type: str
    cpu_pct: float
    power_w: float
    power_lower_w: float
    power_upper_w: float
    is_spike: bool
    spike_threshold_w: float


class BatchItem(BaseModel):
    cpu_pct: float = Field(..., ge=0, le=100)
    cpu_type: str = "unknown"


class BatchRequest(BaseModel):
    predictions: list[BatchItem]


class CpuTypeInfo(BaseModel):
    cpu_type: str
    idle_w: float
    full_w: float
    spike_threshold_w: float
    mean_std_w: float


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    model = _require_model()
    return {
        "status": "ready",
        "cpu_types_loaded": len(model.available_types()),
        "available_cpu_types": model.available_types(),
    }


@app.get("/predict", response_model=PredictionResponse)
def predict(
    cpu_pct: float = Query(..., ge=0, le=100, description="CPU utilization 0–100"),
    cpu_type: str  = Query("unknown", description="CPU model (e.g. intel-xeon-e5420)"),
):
    model = _require_model()
    result = model.predict(cpu_pct, cpu_type)
    return result.__dict__


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(body: BatchRequest):
    model = _require_model()
    return [model.predict(item.cpu_pct, item.cpu_type).__dict__ for item in body.predictions]


@app.get("/models", response_model=list[CpuTypeInfo])
def list_models():
    model = _require_model()
    return [
        CpuTypeInfo(cpu_type=ct, **model.stats(ct))
        for ct in model.available_types()
    ]
