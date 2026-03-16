"""Feature engineering for CPU power model training.

Transforms cpu_data.dat into a training-ready feature matrix:
  - Smooths noisy AVGPOWER per CPU type using isotonic regression (training only)
  - Computes per-CPU-type constants: idle_w, full_w, dynamic_range_w, spike_threshold_w
  - Builds 7 numeric features + raw CPUTYPE string column
  - Applies per-CPU-type NPTS weight normalisation (Option 1)

CPUTYPE encoding is intentionally left to each trainer:
  XGBoost → pandas Categorical + enable_categorical=True
  MLP     → one-hot encode (11 binary columns) + StandardScaler on numerics

Isotonic regression is a preprocessing artefact only.
It is NOT part of the inference pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from data_loader import load_cpu_power_data

_SPIKE_FRACTION = 0.75          # power >= idle + 75% of dynamic range = spike
_MODELS_DIR = Path("models")


def build_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, dict]:
    """Transform raw cpu_data.dat DataFrame into ML-ready features.

    Parameters
    ----------
    df : validated DataFrame from load_cpu_power_data()

    Returns
    -------
    X        : (N × 8) — 7 numeric features + CPUTYPE string column
    y_reg    : smooth power regression target (Watts)
    y_clf    : spike classification target (bool)
    weights  : per-CPU-type normalised NPTS sample weights
    metadata : {cpu_type → {idle_w, full_w, dynamic_range_w, spike_threshold_w}}
    """
    df = df.copy()

    # ── Stage 2: per-CPU-type isotonic smoothing ──────────────────────────────
    # Enforces the physical constraint: power never decreases as CPU% increases.
    # The noisy raw AVGPOWER is never seen by the ML model — only smooth_power_w.
    df["smooth_power_w"] = np.nan

    for cpu_type, group in df.groupby("CPUTYPE"):
        ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
        smoothed = ir.fit_transform(
            group["CPUPCT"].values.astype(float),
            group["AVGPOWER"].values.astype(float),
            sample_weight=group["NPTS"].values.astype(float),
        )
        df.loc[group.index, "smooth_power_w"] = smoothed

    # ── Stage 3: per-CPU-type constants ──────────────────────────────────────
    # Computed from smoothed values so boundary noise doesn't skew the threshold.
    # These constants become features that let the model generalise to new hardware.
    metadata: dict[str, dict] = {}
    df["idle_w"] = np.nan
    df["dynamic_range_w"] = np.nan

    for cpu_type, group in df.groupby("CPUTYPE"):
        at_0   = group[group["CPUPCT"] == 0]["smooth_power_w"]
        at_100 = group[group["CPUPCT"] == 100]["smooth_power_w"]

        idle_w = float(at_0.iloc[0]) if len(at_0) else float(group["smooth_power_w"].min())
        full_w = float(at_100.iloc[0]) if len(at_100) else float(group["smooth_power_w"].max())
        dynamic_range_w   = full_w - idle_w
        spike_threshold_w = idle_w + _SPIKE_FRACTION * dynamic_range_w

        metadata[cpu_type] = {
            "idle_w":            round(idle_w, 3),
            "full_w":            round(full_w, 3),
            "dynamic_range_w":   round(dynamic_range_w, 3),
            "spike_threshold_w": round(spike_threshold_w, 3),
        }
        df.loc[group.index, "idle_w"]          = idle_w
        df.loc[group.index, "dynamic_range_w"] = dynamic_range_w

    # ── Stage 4: feature matrix ───────────────────────────────────────────────
    cpu_pct = df["CPUPCT"].astype(float)

    X = pd.DataFrame(
        {
            "cpu_pct":         cpu_pct,
            "cpu_pct_sq":      cpu_pct ** 2,
            "cpu_pct_cube":    cpu_pct ** 3,
            "sqrt_cpu_pct":    np.sqrt(cpu_pct),
            "log_cpu_pct":     np.log1p(cpu_pct),
            "idle_w":          df["idle_w"],
            "dynamic_range_w": df["dynamic_range_w"],
            "CPUTYPE":         df["CPUTYPE"],   # raw string — trainers encode this
        },
        index=df.index,
    )

    y_reg = df["smooth_power_w"].rename("power_w")

    thresholds = df["CPUTYPE"].map(
        {ct: meta["spike_threshold_w"] for ct, meta in metadata.items()}
    )
    y_clf = (df["smooth_power_w"] >= thresholds).rename("is_spike")

    # ── Stage 5: per-CPU-type normalised weights ──────────────────────────────
    # Each CPU type contributes equal total weight (1.0) to training.
    # Within a type rows are weighted proportionally to NPTS.
    # Prevents intel-xeon-e5420 (10M samples) from dominating the other 10 types.
    df["weight"] = np.nan
    for cpu_type, group in df.groupby("CPUTYPE"):
        total = float(group["NPTS"].sum())
        df.loc[group.index, "weight"] = group["NPTS"] / total

    weights = df["weight"].rename("weight")

    return X, y_reg, y_clf, weights, metadata


# ── Metadata persistence ──────────────────────────────────────────────────────

def save_metadata(metadata: dict, path: Path | str = _MODELS_DIR / "metadata.json") -> None:
    """Persist per-CPU-type constants needed at inference time."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_metadata(path: Path | str = _MODELS_DIR / "metadata.json") -> dict:
    with open(path) as f:
        return json.load(f)


# ── Convenience entry point ───────────────────────────────────────────────────

def load_and_build(filepath: str = "data/cpu_data.dat") -> tuple:
    """Load cpu_data.dat and return all feature engineering outputs."""
    return build_features(load_cpu_power_data(filepath))


if __name__ == "__main__":
    X, y_reg, y_clf, weights, metadata = load_and_build()
    save_metadata(metadata)

    print(f"Feature matrix : {X.shape}")
    print(f"Regression target (power_w)  : min={y_reg.min():.1f}W  max={y_reg.max():.1f}W")
    print(f"Spike rate      : {y_clf.mean():.1%}")
    print(f"Weight sum      : {weights.sum():.1f}  (= {len(metadata)} CPU types)")
    print(f"Metadata saved  : {_MODELS_DIR / 'metadata.json'}")
    print()
    print("Per-CPU spike rates:")
    for ct, meta in sorted(metadata.items()):
        mask = X["CPUTYPE"] == ct
        rate = y_clf[mask].mean()
        print(f"  {ct:<35}  {rate:.1%}  threshold={meta['spike_threshold_w']:.2f}W")
