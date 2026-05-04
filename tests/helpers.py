"""Shared test scaffolding for spike inference tests.

Provides minimal fake model directories and synthetic DataFrames so tests
can run without real training artefacts.  Imported by test_predict.py,
test_daemon_slo.py, and test_version.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from spike.predict import _Artifacts, _load_artifacts
from spike.classifier import _X_COLS


# ── Synthetic input data ──────────────────────────────────────────────────────


def make_agg_row(
    machine_id: int,
    bucket:     int,
    total_cpu:  float = 0.15,
    peak_cpu:   float = 0.20,
    total_mem:  float = 0.10,
    peak_mem:   float = 0.20,
    disk_io:    float = 0.01,
    n_tasks:    int   = 5,
) -> dict:
    """One cluster_agg row with sensible defaults."""
    return {
        "machine_id": machine_id,
        "bucket":     bucket,
        "time_us":    bucket * 300_000_000,
        "total_cpu":  total_cpu,
        "peak_cpu":   peak_cpu,
        "total_mem":  total_mem,
        "peak_mem":   peak_mem,
        "disk_io":    disk_io,
        "n_tasks":    n_tasks,
    }


def make_window_df(
    machine_ids:  list[int] = (1, 2),
    n_buckets:    int        = 24,
    start_bucket: int        = 100,
) -> pd.DataFrame:
    """Minimal valid input DataFrame with n_buckets per machine."""
    rows = [
        make_agg_row(mid, start_bucket + i)
        for mid in machine_ids
        for i in range(n_buckets)
    ]
    df = pd.DataFrame(rows)
    df["machine_id"] = df["machine_id"].astype("int64")
    df["bucket"]     = df["bucket"].astype("int64")
    df["time_us"]    = df["time_us"].astype("int64")
    df["n_tasks"]    = df["n_tasks"].astype("int32")
    for col in ("total_cpu", "peak_cpu", "total_mem", "peak_mem", "disk_io"):
        df[col] = df[col].astype("float32")
    return df


# ── Fake model directory builder ──────────────────────────────────────────────


def train_tiny_booster(feature_cols: list[str], binary: bool = False) -> xgb.Booster:
    """5-tree XGBoost booster trained on random synthetic data."""
    rng = np.random.default_rng(0)
    X   = rng.standard_normal((30, len(feature_cols))).astype("float32")
    if binary:
        y      = rng.integers(0, 2, size=30)
        params = {"objective": "binary:logistic", "eval_metric": "aucpr", "verbosity": 0}
    else:
        y      = rng.integers(0, 3, size=30)
        params = {
            "objective":   "multi:softprob",
            "num_class":   3,
            "eval_metric": "mlogloss",
            "verbosity":   0,
        }
    return xgb.train(params, xgb.DMatrix(X, label=y, feature_names=feature_cols), num_boost_round=5)


def write_horizon_dir(
    parent_dir:      Path,
    dir_name:        str,
    feature_cols:    list[str],
    alarm_threshold: float,
    binary:          bool = False,
) -> None:
    """Write one horizon sub-directory: model weights + meta + config."""
    h_dir = parent_dir / dir_name
    h_dir.mkdir(parents=True, exist_ok=True)

    booster = train_tiny_booster(feature_cols, binary=binary)
    booster.save_model(str(h_dir / "spike_model.json"))

    (h_dir / "spike_model.meta.json").write_text(json.dumps({
        "feature_cols":    feature_cols,
        "xgboost_version": xgb.__version__,
    }))
    (h_dir / "spike_config.json").write_text(json.dumps({
        "alarm_threshold": alarm_threshold,
        "inference":       {"feature_cols": feature_cols},
    }))


def make_fake_artifacts(
    tmp_path:             Path,
    feature_cols:         list[str] | None = None,
    alarm_threshold:      float            = 0.25,
    global_threshold:     float            = 0.35,
    global_threshold_p99: float            = 0.40,
    thresholds:           dict[int, float] | None = None,
    thresholds_p99:       dict[int, float] | None = None,
    include_binary_horizons: bool = True,
    include_severe_ovr:      bool = False,
) -> tuple[Path, _Artifacts]:
    """Build a minimal fake model directory and return (model_dir, artifacts).

    Trains tiny 5-tree XGBoost models so _load_artifacts() can be called
    without a real training run.  The 60m config includes trained_at so
    get_model_version() returns a proper version string in versioning tests.
    """
    feature_cols   = feature_cols   or list(_X_COLS)
    thresholds     = thresholds     or {1: 0.30, 2: 0.32}
    thresholds_p99 = thresholds_p99 or {k: v * 1.15 for k, v in thresholds.items()}

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 60m 3-class severity model (always required)
    write_horizon_dir(models_dir, "spike", feature_cols, alarm_threshold, binary=False)
    model_dir = models_dir / "spike"

    # Optional short-horizon binary models
    if include_binary_horizons:
        write_horizon_dir(models_dir, "spike_15m", feature_cols, alarm_threshold * 0.8, binary=True)
    if include_severe_ovr:
        write_horizon_dir(models_dir, "spike_severe_ovr", feature_cols, alarm_threshold * 0.9, binary=True)

    # Per-machine p95/p99 thresholds
    thresh_df = pd.DataFrame([
        {
            "machine_id":    mid,
            "threshold_p95": thr,
            "threshold_p99": thresholds_p99.get(mid, thr * 1.15),
        }
        for mid, thr in thresholds.items()
    ])
    thresh_df["machine_id"]    = thresh_df["machine_id"].astype("int64")
    thresh_df["threshold_p95"] = thresh_df["threshold_p95"].astype("float32")
    thresh_df["threshold_p99"] = thresh_df["threshold_p99"].astype("float32")
    thresh_df.to_parquet(model_dir / "spike_thresholds.parquet", index=False)

    # Rewrite 60m config with the full inference block and trained_at.
    # trained_at is required by get_model_version() to produce a timestamped version string.
    (model_dir / "spike_config.json").write_text(json.dumps({
        "alarm_threshold": alarm_threshold,
        "trained_at":      "2026-01-01T00:00:00+00:00",
        "inference": {
            "bucket_duration_seconds":       300,
            "horizon_windows":               12,
            "horizon_minutes":               60,
            "lookback_windows":              24,
            "lookback_minutes":              120,
            "min_buckets_cold_start":        12,
            "feature_cols":                  feature_cols,
            "global_threshold_p95_fallback": global_threshold,
            "global_threshold_p99_fallback": global_threshold_p99,
            "thresholds_file":               str(model_dir / "spike_thresholds.parquet"),
        },
    }))

    arts = _load_artifacts(model_dir)
    return model_dir, arts
