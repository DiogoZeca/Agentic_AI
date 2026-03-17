"""CPU power model — loads a trained ML artefact and serves predictions via O(1) lookup.

At startup, CpuPowerModel.load() reads the winner manifest, loads the winning
trainer artefact (XGBoost or MLP), pre-computes all 1 111 predictions
(11 CPU types × 101 cpu_pct values), applies isotonic post-processing to enforce
physical monotonicity, and stores everything in an in-memory dict.

Request-time prediction is a pure dict lookup — no model forward pass overhead.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from model_trainer import MLPPowerTrainer, XGBoostPowerTrainer

# Lookup grid: integer cpu_pct values that match the training data resolution.
_PCT_GRID: list[int] = list(range(101))   # 0, 1, 2, ..., 100


@dataclass(frozen=True)
class PowerPrediction:
    cpu_type: str
    cpu_pct: float
    power_w: float
    power_lower_w: float
    power_upper_w: float
    is_spike: bool
    spike_threshold_w: float


class CpuPowerModel:
    """Inference-only CPU power model loaded from a trained artefact.

    Usage
    -----
        model = CpuPowerModel.load("models/winner.json")
        pred  = model.predict(cpu_pct=75.0, cpu_type="intel-xeon-e5420")

    Design
    ------
    - load()         reads winner.json, loads the winning trainer + metadata.json
    - _build_lookup  runs one batch forward pass → isotonic smoothing → dict
    - predict()      is a pure (cpu_type, int(round(cpu_pct))) dict lookup
    """

    def __init__(self) -> None:
        self._lookup:   dict[tuple[str, int], PowerPrediction] = {}
        self._metadata: dict[str, dict] = {}

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, manifest_path: str | Path) -> "CpuPowerModel":
        """Load the winner artefact and build the full prediction lookup table.

        Parameters
        ----------
        manifest_path : path to winner.json produced by train.py
        """
        manifest_path = Path(manifest_path)
        with open(manifest_path) as f:
            manifest = json.load(f)

        model_dir  = manifest_path.parent / manifest["dir"]
        model_name = manifest["model"]

        if model_name == "xgboost":
            trainer: XGBoostPowerTrainer | MLPPowerTrainer = XGBoostPowerTrainer.load(model_dir)
        elif model_name == "mlp":
            trainer = MLPPowerTrainer.load(model_dir)
        else:
            raise ValueError(f"Unknown model name in manifest: {model_name!r}")

        metadata_path = manifest_path.parent / "metadata.json"
        with open(metadata_path) as f:
            metadata: dict[str, dict] = json.load(f)

        inst = cls()
        inst._metadata = metadata
        inst._lookup   = cls._build_lookup(trainer, metadata)
        return inst

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(self, cpu_pct: float, cpu_type: str = "unknown") -> PowerPrediction:
        """Return a PowerPrediction for the given cpu_pct and cpu_type.

        Unknown cpu_types fall back silently to "unknown".
        Fractional cpu_pct values are rounded to the nearest integer
        (training data resolution = 1 percentage point).
        """
        if cpu_type not in self._metadata:
            cpu_type = "unknown"
        pct_key = int(round(min(max(float(cpu_pct), 0.0), 100.0)))
        return self._lookup[(cpu_type, pct_key)]

    # ── Metadata ───────────────────────────────────────────────────────────────

    def available_types(self) -> list[str]:
        return sorted(self._metadata.keys())

    def stats(self, cpu_type: str) -> dict:
        """Return the 4 per-CPU-type constants exposed by the /models endpoint.

        Returns only the API contract keys — extra metadata fields (e.g.
        dynamic_range_w) are intentionally omitted to keep the response stable.
        """
        meta = self._metadata.get(cpu_type, {})
        return {k: meta[k] for k in ("idle_w", "full_w", "spike_threshold_w", "mean_std_w") if k in meta}

    # ── Private ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_lookup(
        trainer: XGBoostPowerTrainer | MLPPowerTrainer,
        metadata: dict[str, dict],
    ) -> dict[tuple[str, int], PowerPrediction]:
        """Pre-compute predictions for every (cpu_type, cpu_pct) pair.

        Steps
        -----
        1. Build an 1 111-row feature DataFrame (11 types × 101 pct values)
           in the exact column layout expected by both trainers.
        2. Run a single batch forward pass — one predict_power() call.
        3. Reshape to (n_types, 101) for per-type isotonic post-processing.
        4. Isotonic regression enforces physical monotonicity: power must never
           decrease as CPU% increases. Small XGBoost tree reversals are fixed here.
        5. Store as (cpu_type, pct_int) → PowerPrediction.
        """
        cpu_types = sorted(metadata.keys())
        n_pcts    = len(_PCT_GRID)

        # ── Step 1: feature matrix ─────────────────────────────────────────────
        rows: list[dict] = []
        for cpu_type in cpu_types:
            meta        = metadata[cpu_type]
            idle_w      = meta["idle_w"]
            dyn_range_w = meta["dynamic_range_w"]
            for pct in _PCT_GRID:
                p = float(pct)
                rows.append({
                    "cpu_pct":         p,
                    "cpu_pct_sq":      p * p,
                    "cpu_pct_cube":    p * p * p,
                    "sqrt_cpu_pct":    math.sqrt(p),
                    "log_cpu_pct":     math.log1p(p),
                    "idle_w":          idle_w,
                    "dynamic_range_w": dyn_range_w,
                    "CPUTYPE":         cpu_type,
                })
        X = pd.DataFrame(rows)

        # ── Step 2: single batch forward pass ──────────────────────────────────
        raw_preds = trainer.predict_power(X)   # shape: (n_types × n_pcts,)

        # ── Step 3: reshape ────────────────────────────────────────────────────
        raw_matrix = raw_preds.reshape(len(cpu_types), n_pcts)

        # ── Steps 4 + 5: isotonic smoothing + dict construction ────────────────
        isotonic  = IsotonicRegression(increasing=True, out_of_bounds="clip")
        pcts_arr  = np.array(_PCT_GRID, dtype=float)
        lookup: dict[tuple[str, int], PowerPrediction] = {}

        for i, cpu_type in enumerate(cpu_types):
            meta              = metadata[cpu_type]
            spike_threshold_w = meta["spike_threshold_w"]
            mean_std_w        = meta["mean_std_w"]

            # Enforce monotonicity; clip to physical floor (power ≥ 0)
            smooth = np.maximum(0.0, isotonic.fit_transform(pcts_arr, raw_matrix[i]))

            for j, pct in enumerate(_PCT_GRID):
                power_w = float(round(smooth[j], 3))
                lookup[(cpu_type, pct)] = PowerPrediction(
                    cpu_type          = cpu_type,
                    cpu_pct           = float(pct),
                    power_w           = power_w,
                    power_lower_w     = round(max(0.0, power_w - 2.0 * mean_std_w), 3),
                    power_upper_w     = round(power_w + 2.0 * mean_std_w, 3),
                    is_spike          = power_w >= spike_threshold_w,
                    spike_threshold_w = spike_threshold_w,
                )

        return lookup
