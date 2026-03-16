"""CPU power model — quadratic regression trained on cpu_data.dat.

Maps (cpu_type, cpu_pct) → Power (Watts) with spike detection.
Falls back to the 'unknown' model when an unrecognised CPU type is given.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# A reading is a spike when its predicted power is in the top 25% of the
# CPU's dynamic range (idle → full load).
_SPIKE_FRACTION = 0.75


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
    """Per-CPU-type quadratic model: Power(W) = f(CPU%).

    Fit using NPTS as sample weights so high-observation buckets
    (e.g. idle and full-load) dominate the regression.
    Uncertainty bounds use the real per-bucket std dev recovered from SUM2.
    """

    def __init__(self) -> None:
        self._models: dict[str, Pipeline] = {}
        self._stats: dict[str, dict] = {}

    # ── Training ───────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "CpuPowerModel":
        for cpu_type, group in df.groupby("CPUTYPE"):
            group = group.sort_values("CPUPCT")
            X = group[["CPUPCT"]].values.astype(float)
            y = group["AVGPOWER"].values.astype(float)
            weights = group["NPTS"].values.astype(float)

            model = Pipeline([
                ("poly", PolynomialFeatures(degree=2, include_bias=True)),
                ("reg", LinearRegression()),
            ])
            model.fit(X, y, reg__sample_weight=weights)
            self._models[cpu_type] = model

            # Idle and full-load power from measured data (prefer direct lookup
            # over the polynomial extrapolation at the boundaries).
            pcts = group["CPUPCT"].values
            avgs = group["AVGPOWER"].values
            idle_w = float(avgs[pcts == 0][0]) if 0 in pcts else float(model.predict([[0]])[0])
            full_w = float(avgs[pcts == 100][0]) if 100 in pcts else float(model.predict([[100]])[0])

            # Weighted mean std dev recovered from SUM2 (population variance).
            npts = group["NPTS"].values.astype(float)
            variance = group["SUM2"].values / npts - avgs ** 2
            std_devs = np.sqrt(np.maximum(0.0, variance))
            mean_std_w = float(np.average(std_devs, weights=npts))

            self._stats[cpu_type] = {
                "idle_w": round(idle_w, 3),
                "full_w": round(full_w, 3),
                "mean_std_w": round(mean_std_w, 3),
                "spike_threshold_w": round(idle_w + _SPIKE_FRACTION * (full_w - idle_w), 3),
            }

        return self

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(self, cpu_pct: float, cpu_type: str = "unknown") -> PowerPrediction:
        if cpu_type not in self._models:
            cpu_type = "unknown"

        model = self._models[cpu_type]
        stats = self._stats[cpu_type]

        power_w = float(model.predict([[cpu_pct]])[0])
        std = stats["mean_std_w"]

        return PowerPrediction(
            cpu_type=cpu_type,
            cpu_pct=cpu_pct,
            power_w=round(power_w, 3),
            power_lower_w=round(max(0.0, power_w - 2 * std), 3),
            power_upper_w=round(power_w + 2 * std, 3),
            is_spike=power_w >= stats["spike_threshold_w"],
            spike_threshold_w=stats["spike_threshold_w"],
        )

    # ── Metadata ───────────────────────────────────────────────────────────────

    def available_types(self) -> list[str]:
        return sorted(self._models.keys())

    def stats(self, cpu_type: str) -> dict:
        return dict(self._stats.get(cpu_type, {}))
