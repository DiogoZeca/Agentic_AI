"""Inference script for the CPU spike classifier.

2-model architecture: a 15m binary classifier (BinarySpikeClassifier) and a
60m 3-class severity classifier (SpikeClassifier).  The 30m and 45m binary
models were dropped in Fix 3 (architecture simplification).

Accepts a CSV of per-machine aggregated 5-minute CPU data (last 120 min,
24 buckets per machine) and produces spike-probability predictions for each
machine.

Input schema (cluster_agg format)
----------------------------------
  machine_id  int64   — unique machine identifier
  bucket      int64   — 5-min bucket index (monotonically increasing per machine)
  time_us     int64   — bucket * 300_000_000 (microseconds since trace epoch)
  total_cpu   float32 — duration-weighted total CPU load  (fraction of 1 core)
  peak_cpu    float32 — peak cpu_rate observed in this bucket
  total_mem   float32 — sum of canonical_mem_usage across tasks
  peak_mem    float32 — peak max_mem_usage
  disk_io     float32 — max mean_disk_io_time across tasks
  n_tasks     int32   — number of concurrent tasks in this bucket

Provide the last 120 minutes (24 buckets) per machine.  Machines with fewer
than 24 rows still receive predictions but are flagged as cold_start_degraded.
Machines with fewer than min_buckets_cold_start rows (default 12) are returned
with null probability and status=cold_start.

Output (JSON)
-------------
  predicted_at          ISO 8601 timestamp of the inference call
  model_dir             resolved path to the model artefacts directory
  horizon_minutes       60 — predictions cover the next 60 min
  machines_total        number of unique machines in the input
  machines_predicted    machines that received a probability estimate
  machines_cold_start   machines that lacked sufficient history
  predictions           list of per-machine prediction objects:
    machine_id              int
    severity_class          int (0=no_spike, 1=moderate, 2=severe) | null
    p_no_spike              float in [0, 1] | null
    p_moderate              float in [0, 1] | null
    p_severe                float in [0, 1] | null
    is_spike                bool | null — severity_class >= 1 (backward compat)
    is_severe               bool | null — p_severe >= alarm_threshold
    alarm_threshold         float | null — p_severe threshold used for is_severe
    threshold_used          float | null — per-machine p95 CPU threshold
    threshold_source        "learned" | "global_fallback" | null
    status                  "success" | "cold_start_degraded" | "cold_start"
    observations_used       int — buckets of history provided

Usage
-----
  # Predictions to stdout; logs to stderr
  python predict_spike.py --input window.csv --model-dir models/spike/

  # Also write to a file (atomic write — safe if process is killed mid-run)
  python predict_spike.py --input window.csv --model-dir models/spike/ \\
      --output predictions.json

Exit codes
----------
  0  success
  1  missing file, schema violation, or configuration mismatch
  2  empty input (no machines to predict)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

# Import the exact same feature-engineering functions used during training.
# This is the structural guarantee against training-serving skew: one module,
# one implementation, shared by both train_spike_classifier.py and this script.
from spike_feature_engineer import (
    _add_cluster_features,
    _add_time_features,
    _engineer_machine,
)
from spike_classifier import _X_COLS

# ── Logging (always to stderr — stdout is reserved for JSON output) ───────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(message)s",
    datefmt = "%H:%M:%S",
    stream  = sys.stderr,
)
log = logging.getLogger(__name__)

# ── Required input columns (cluster_agg format) ───────────────────────────────

_INPUT_COLS: tuple[str, ...] = (
    "machine_id", "bucket", "time_us",
    "total_cpu", "peak_cpu", "total_mem", "peak_mem", "disk_io", "n_tasks",
)

# Short-horizon binary models used for imminence scoring, in ascending order.
# The 60m severity model is handled separately and is always required.
_BINARY_HORIZONS_ORDERED: list[str] = ["15m"]


# ── Artefacts dataclass ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Artifacts:
    """Immutable bundle of model artefacts loaded once at startup.

    Supports 2-model inference: boosters holds one XGBoost Booster per
    horizon key ("15m", "60m").  alarm_thresholds holds the per-horizon alarm
    threshold selected on the validation set.

    Horizon "15m" is binary (P(spike) from binary:logistic).
    Horizon "60m" is 3-class (P(no_spike), P(moderate), P(severe)).
    """
    boosters:          dict[str, xgb.Booster]   # horizon_name → XGBoost Booster
    feature_cols:      list[str]                 # ordered — must match _X_COLS exactly
    alarm_thresholds:  dict[str, float]          # horizon_name → alarm threshold
    min_buckets:       int                       # < this → cold_start (no prediction)
    global_thresh:     float                     # p95 fallback for unknown machines
    global_thresh_p99: float                     # p99 fallback for unknown machines
    thresholds:        pd.Series                 # machine_id (int) → threshold_p95 (float)
    thresholds_p99:    pd.Series                 # machine_id (int) → threshold_p99 (float)
    horizon_minutes:   int                       # primary horizon for output annotation
    model_dir:         Path                      # source directory (for response metadata)
    trained_at:        str | None = None         # ISO-8601 timestamp from spike_config.json
    calibrators_60m:   list | None = None        # per-class IsotonicRegression fitted on val set


# ── Load artefacts ────────────────────────────────────────────────────────────


def _load_single_booster(model_dir: Path) -> tuple[xgb.Booster, list[str], float]:
    """Load one horizon's booster, feature list, and alarm threshold.

    Returns
    -------
    (booster, feature_cols, alarm_threshold)
    """
    model_path  = model_dir / "spike_model.json"
    meta_path   = model_dir / "spike_model.meta.json"
    config_path = model_dir / "spike_config.json"

    for p in (model_path, meta_path, config_path):
        if not p.exists():
            raise FileNotFoundError(f"Required artefact not found: {p}")

    meta   = json.loads(meta_path.read_text())
    config = json.loads(config_path.read_text())

    meta_cols   = meta["feature_cols"]
    config_cols = config.get("inference", {}).get("feature_cols", [])
    if config_cols and meta_cols != config_cols:
        raise RuntimeError(
            f"Feature column mismatch in {model_dir}: "
            "spike_model.meta.json and spike_config.json from different training runs."
        )

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    booster.set_param("nthread", 1)

    alarm_threshold = float(config["alarm_threshold"])
    return booster, meta_cols, alarm_threshold


def _load_artifacts(model_dir: Path) -> _Artifacts:
    """Load all inference artefacts from model_dir.

    The 60m model is required (``model_dir/spike_model.json``).  The binary
    15m horizon model (spike_15m/ sibling of model_dir) is loaded when present;
    if absent it is silently skipped and only the 60m model is used.

    Parameters
    ----------
    model_dir : directory containing the primary (60m) model artefacts.
                The binary 15m horizon model is expected as a sibling directory
                named ``spike_15m`` relative to ``model_dir.parent``.

    Raises
    ------
    FileNotFoundError  if any required 60m artefact file is missing.
    RuntimeError       if feature_cols in meta.json ≠ feature_cols in config.
    """
    # Load primary 60m model first — produces the most descriptive error on
    # missing directories (spike_model.json not found).
    booster_60m, meta_cols, alarm_60m = _load_single_booster(model_dir)

    thresholds_path = model_dir / "spike_thresholds.parquet"
    if not thresholds_path.exists():
        raise FileNotFoundError(f"Required artefact not found: {thresholds_path}")

    boosters: dict[str, xgb.Booster]  = {"60m": booster_60m}
    alarm_thresholds: dict[str, float] = {"60m": alarm_60m}

    # Load optional binary horizon models (sibling directories of model_dir).
    # "15m"         — short-horizon imminence scorer.
    # "severe_ovr"  — OVR severe binary detector trained end-to-end on p99
    #                 exceedances; overrides is_severe from the 60m softmax column
    #                 when present (Phase 5).
    models_parent = model_dir.parent
    for h_name in ("15m", "severe_ovr"):
        h_dir = models_parent / f"spike_{h_name}"
        if (h_dir / "spike_model.json").exists():
            try:
                b, _, alm = _load_single_booster(h_dir)
                boosters[h_name]         = b
                alarm_thresholds[h_name] = alm
                log.info("  Loaded %s binary model from %s", h_name, h_dir.name)
            except FileNotFoundError as e:
                # A companion artefact (meta.json, config.json) is missing.
                # Expected when the model hasn't been trained yet — degrade
                # gracefully, the 60m model still produces valid predictions.
                log.warning("  Optional %s model incomplete (missing artefact) — skipping: %s", h_name, e)
            except (RuntimeError, ValueError) as e:
                # Structural integrity error (feature mismatch, corrupted JSON,
                # incompatible XGBoost version).  This is unexpected — a silently
                # wrong prediction is worse than a hard failure here.
                raise RuntimeError(
                    f"Artefact integrity error in optional '{h_name}' model — "
                    f"re-train or delete corrupted artefacts. Cause: {e}"
                ) from e

    config_path = model_dir / "spike_config.json"
    config      = json.loads(config_path.read_text())
    inf         = config.get("inference", {})
    trained_at  = config.get("trained_at")

    thresholds_df  = pd.read_parquet(thresholds_path)
    thresholds     = thresholds_df.set_index("machine_id")["threshold_p95"]
    thresholds_p99 = thresholds_df.set_index("machine_id")["threshold_p99"]

    # Load optional per-class isotonic calibrators (fitted on val set during training).
    # When present, raw softmax probabilities are calibrated before threshold comparisons.
    # Absent on first-run models — inference degrades gracefully to raw probabilities.
    calibrators_60m = None
    cal_path = model_dir / "calibrators.pkl"
    if cal_path.exists():
        with open(cal_path, "rb") as _f:
            calibrators_60m = pickle.load(_f)
        log.info("  Calibrators     : loaded (%d classes)", len(calibrators_60m))
    else:
        log.info("  Calibrators     : not found — using raw softmax probabilities")

    log.info("Artefacts loaded from %s", model_dir)
    log.info("  Horizons loaded : %s", ", ".join(sorted(boosters.keys())))
    log.info("  Features        : %d columns", len(meta_cols))
    log.info("  Alarm thresholds: %s",
             "  ".join(f"{h}={t:.3f}" for h, t in sorted(alarm_thresholds.items())))
    log.info("  Known machines  : %d", len(thresholds))

    return _Artifacts(
        boosters          = boosters,
        feature_cols      = meta_cols,
        alarm_thresholds  = alarm_thresholds,
        min_buckets       = int(inf.get("min_buckets_cold_start", 12)),
        global_thresh     = float(inf.get("global_threshold_p95_fallback",
                                          float(thresholds.median()))),
        global_thresh_p99 = float(inf.get("global_threshold_p99_fallback",
                                          float(thresholds_p99.median()))),
        thresholds        = thresholds,
        thresholds_p99    = thresholds_p99,
        horizon_minutes   = int(inf.get("horizon_minutes", 60)),
        model_dir         = model_dir.resolve(),
        trained_at        = trained_at,
        calibrators_60m   = calibrators_60m,
    )


# ── Input validation ──────────────────────────────────────────────────────────


def _validate_input(df: pd.DataFrame) -> None:
    """Raise ValueError with a descriptive message on any schema violation.

    Parameters
    ----------
    df : raw DataFrame loaded from the input CSV.

    Raises
    ------
    ValueError  if required columns are missing or the DataFrame is empty.
    """
    if df.empty:
        raise ValueError("Input is empty — nothing to predict.")

    missing = [c for c in _INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing required column(s): {missing}.\n"
            f"Expected: {list(_INPUT_COLS)}"
        )


# ── Feature engineering ───────────────────────────────────────────────────────


def _build_features(
    input_df:  pd.DataFrame,
    artifacts: _Artifacts,
) -> tuple[pd.DataFrame, dict[int, str]]:
    """Build the feature matrix using the same functions used during training.

    Per-machine processing mirrors train_spike_classifier.py exactly:
      1. Per-machine loop  → _engineer_machine() fills gaps, computes lags/EWMA
      2. concat all parts
      3. _add_cluster_features() on the full frame  (cross-sectional)
      4. _add_time_features()    on the full frame  (harmonic encoding)

    At inference time there is no future window, so _add_label() is never called.

    _add_cluster_features receives train_bucket_max = max(bucket) so that ALL
    rows count as "training" — the live cluster state is always the current
    cluster state, and every bucket should contribute to cluster_cpu_p90.

    Parameters
    ----------
    input_df  : validated per-machine aggregated DataFrame.
    artifacts : loaded model artefacts.

    Returns
    -------
    (feature_df, status_map)
    feature_df  : rows for machines that produced predictions, feature columns only.
    status_map  : {machine_id: "success" | "cold_start_degraded" | "cold_start"}
    """
    lookback_windows = 24   # 120 min / 5 min per bucket

    parts:      list[pd.DataFrame] = []
    status_map: dict[int, str]     = {}

    for machine_id, group in input_df.groupby("machine_id"):
        machine_id = int(machine_id)
        n          = len(group)

        if n < artifacts.min_buckets:
            status_map[machine_id] = "cold_start"
            continue

        status_map[machine_id] = (
            "cold_start_degraded" if n < lookback_windows else "success"
        )

        threshold     = float(artifacts.thresholds.get(machine_id, artifacts.global_thresh))
        threshold_p99 = float(artifacts.thresholds_p99.get(machine_id, artifacts.global_thresh_p99))
        full_series   = _engineer_machine(group.copy(), machine_id, threshold, threshold_p99=threshold_p99)

        # Keep only original (non-gap-filled) rows — same filter as training
        original = (
            full_series[full_series["_original"]]
            .drop(columns=["_original"])
            .reset_index(drop=True)
        )
        parts.append(original)

    if not parts:
        return pd.DataFrame(columns=list(_X_COLS)), status_map

    final = pd.concat(parts, ignore_index=True)

    # Cluster-level features: pass max(bucket) as train_bucket_max so every
    # row in the live window is included in the per-bucket p90 calculation.
    final = _add_cluster_features(final, train_bucket_max=int(final["bucket"].max()))
    final = _add_time_features(final)

    # cpu_vs_p95: machine-relative normalised CPU — mirrors engineer() exactly.
    # Uses per-machine learned threshold; falls back to global for unknown machines.
    thresh_mapped = final["machine_id"].map(artifacts.thresholds).fillna(artifacts.global_thresh)
    final["cpu_vs_p95"] = np.where(
        thresh_mapped > 0,
        final["total_cpu"] / thresh_mapped,
        0.0,
    ).astype("float32")

    return final, status_map


# ── Inference ─────────────────────────────────────────────────────────────────


def _run_inference(
    feature_df: pd.DataFrame,
    input_df:   pd.DataFrame,
    status_map: dict[int, str],
    artifacts:  _Artifacts,
) -> list[dict]:
    """Run the booster and assemble per-machine prediction dicts.

    Uses booster.inplace_predict() directly, bypassing the sklearn wrapper
    and its DMatrix construction overhead.  At ~300K rows the DMatrix
    construction accounts for the majority of prediction time, so this is a
    measurable win with no functional downside.

    Column order is enforced by selecting artifacts.feature_cols explicitly —
    inplace_predict on a bare numpy array does NOT validate column order, so
    this selection is the guard against silent wrong predictions.

    Parameters
    ----------
    feature_df  : output of _build_features() (rows for non-cold-start machines).
    input_df    : original input (used to look up observations_used per machine).
    status_map  : machine_id → status string.
    artifacts   : loaded model artefacts.

    Returns
    -------
    List of prediction dicts, one entry per machine in input_df.
    """
    predictions: list[dict] = []

    # Pre-compute observation counts once — avoids an O(n_machines²) scan where
    # each machine_id triggers a full DataFrame filter over all rows.  Defined
    # here (outside the feature_df.empty guard) so the cold-start loop can use
    # it even when feature_df is entirely empty.
    obs_count = input_df.groupby("machine_id").size().to_dict()

    if not feature_df.empty:
        # Guard: all required feature columns must be present before inference.
        # Missing columns produce silent NaN predictions, which is worse than failing fast.
        missing = [c for c in artifacts.feature_cols if c not in feature_df.columns]
        if missing:
            raise ValueError(
                f"Feature DataFrame is missing {len(missing)} required column(s): {missing}. "
                "Ensure spike_feature_engineer produces the same feature set used during training."
            )

        # Enforce feature column order — critical for inplace_predict correctness
        X = feature_df[artifacts.feature_cols].astype("float32").values

        # Run all horizon boosters — collect last-bucket probability per machine
        # horizon_probs: machine_id → {"15m": p_spike, "30m": p_spike, ...}
        horizon_probs: dict[int, dict[str, float]] = {}

        # Build machine_id index for the feature_df rows
        machine_ids_col = feature_df["machine_id"].values

        for h_name, booster in artifacts.boosters.items():
            if h_name == "60m":
                # multi:softprob — inplace_predict returns (n, 3): [P(no), P(mod), P(sev)]
                raw      = booster.inplace_predict(X)
                probas_h = raw.reshape(-1, 3)
                # Apply per-class isotonic calibration when calibrators are available.
                # Calibrators are fitted on the val set during training (sklearn IsotonicRegression).
                # Renormalize after calibration so probabilities sum to 1.
                if artifacts.calibrators_60m is not None:
                    cals = artifacts.calibrators_60m
                    cal  = np.column_stack([
                        np.clip(cals[k].predict(probas_h[:, k].astype("float64")), 0.0, 1.0)
                        for k in range(3)
                    ])
                    row_sums = cal.sum(axis=1, keepdims=True)
                    probas_h = np.where(row_sums > 0, cal / row_sums, cal).astype("float32")
            else:
                # binary:logistic — use strict_shape=True to always get (n, 1),
                # eliminating version-dependent shape variance.  Without it,
                # XGBoost may return (n,) in some versions, (n,1) in others; a
                # naive reshape(-1, 2) on a (n,1) array silently corrupts values.
                raw  = booster.inplace_predict(X, strict_shape=True)  # (n, 1)
                p1   = raw[:, 0]                                        # P(spike)
                probas_h = np.stack([1 - p1, p1], axis=1)              # (n, 2)

            # For each row, record probability; keep only the LAST bucket per machine
            for i, mid in enumerate(machine_ids_col):
                mid = int(mid)
                if mid not in horizon_probs:
                    horizon_probs[mid] = {}
                # Overwrite with latest bucket (rows are ordered by bucket in feature_df)
                if h_name == "60m":
                    horizon_probs[mid]["60m_vec"] = probas_h[i]  # shape (3,)
                else:
                    horizon_probs[mid][h_name] = float(probas_h[i, 1])  # P(spike)

        for machine_id, h_probs in horizon_probs.items():
            thresh = float(artifacts.thresholds.get(machine_id, artifacts.global_thresh))
            n_obs  = int(obs_count.get(int(machine_id), 0))

            # 60m vector
            pv_60m = h_probs.get("60m_vec")
            if pv_60m is not None:
                p_no, p_mod, p_sev = float(pv_60m[0]), float(pv_60m[1]), float(pv_60m[2])
                sev_cls = int(np.argmax(pv_60m))
                p_spike_60m = p_mod + p_sev   # binary-equivalent for monotonicity
            else:
                p_no = p_mod = p_sev = None
                sev_cls     = None
                p_spike_60m = None

            # OVR severe probability (Phase 5) — dedicated binary p99 detector.
            # When present, overrides is_severe from the 60m softmax column.
            p_sev_ovr = h_probs.get("severe_ovr")
            alm_ovr   = artifacts.alarm_thresholds.get("severe_ovr", 0.5)

            # Collect binary P(spike) per short horizon
            raw_p_spikes = {
                h: h_probs.get(h)
                for h in _BINARY_HORIZONS_ORDERED
                if h in artifacts.boosters
            }

            # Monotonicity enforcement:
            # P(spike) should increase with the horizon.  Apply
            # np.maximum.accumulate across [p_15m, p_spike_60m].
            horizon_seq  = _BINARY_HORIZONS_ORDERED + ["60m"]
            p_seq_values = [raw_p_spikes.get(h) for h in _BINARY_HORIZONS_ORDERED]
            if p_spike_60m is not None:
                p_seq_values.append(p_spike_60m)
            else:
                p_seq_values.append(None)

            # Only enforce monotonicity when all values are present
            valid_p = [v for v in p_seq_values if v is not None]
            if len(valid_p) == len(p_seq_values):
                enforced = list(np.maximum.accumulate(p_seq_values).tolist())
            else:
                enforced = p_seq_values  # partial — skip enforcement

            p_enforced = {
                horizon_seq[i]: enforced[i]
                for i in range(len(horizon_seq))
                if enforced[i] is not None
            }

            # Build imminence dict
            imminence: dict = {}
            for h in _BINARY_HORIZONS_ORDERED:
                if h in artifacts.boosters:
                    p_s = p_enforced.get(h)
                    alm = artifacts.alarm_thresholds.get(h, 0.5)
                    imminence[h] = {
                        "p_spike":         round(p_s, 6) if p_s is not None else None,
                        "is_spike":        bool(p_s >= alm) if p_s is not None else None,
                        "alarm_threshold": round(alm, 4),
                    }

            if pv_60m is not None:
                alm_60m = artifacts.alarm_thresholds.get("60m", 0.5)
                # p_spike_60m_enforced = max(p_15m, p_mod+p_sev) after monotonic
                # enforcement — P(any spike in 60m), not P(severe specifically).
                p_spike_60m_enforced = p_enforced.get("60m", p_spike_60m)
                # is_severe: prefer OVR model when available (Phase 5)
                if p_sev_ovr is not None:
                    is_severe_60m = bool(p_sev_ovr >= alm_ovr)
                else:
                    is_severe_60m = bool(p_sev >= alm_60m)
                imminence["60m"] = {
                    "severity_class":  sev_cls,
                    "p_no_spike":      round(p_no,  6),
                    "p_moderate":      round(p_mod, 6),
                    "p_severe":        round(p_sev, 6),
                    "p_spike":         round(p_spike_60m_enforced, 6),
                    "is_spike":        bool(sev_cls >= 1),
                    "is_severe":       is_severe_60m,
                    "alarm_threshold": round(alm_60m, 4),
                    **({"p_severe_ovr": round(p_sev_ovr, 6), "alarm_threshold_ovr": round(alm_ovr, 4)}
                       if p_sev_ovr is not None else {}),
                }

            # Top-level is_severe also uses OVR when available
            if p_sev_ovr is not None:
                top_is_severe = bool(p_sev_ovr >= alm_ovr)
            elif p_sev is not None:
                top_is_severe = bool(p_sev >= artifacts.alarm_thresholds.get("60m", 0.5))
            else:
                top_is_severe = None

            predictions.append({
                "machine_id":       machine_id,
                "imminence":        imminence,
                # Backward-compatible top-level fields from 60m model
                "severity_class":   sev_cls,
                "p_no_spike":       round(p_no,  6) if p_no  is not None else None,
                "p_moderate":       round(p_mod, 6) if p_mod is not None else None,
                "p_severe":         round(p_sev, 6) if p_sev is not None else None,
                "p_severe_ovr":     round(p_sev_ovr, 6) if p_sev_ovr is not None else None,
                "is_spike":         bool(sev_cls >= 1) if sev_cls is not None else None,
                "is_severe":        top_is_severe,
                "alarm_threshold":  round(artifacts.alarm_thresholds.get("60m", 0.5), 4),
                "threshold_used":   round(thresh, 6),
                "threshold_source": (
                    "learned"
                    if machine_id in artifacts.thresholds.index
                    else "global_fallback"
                ),
                "status":           status_map[machine_id],
                "observations_used": n_obs,
            })

    # Append null entries for cold-start machines
    for machine_id, status in status_map.items():
        if status == "cold_start":
            n_obs = int(obs_count.get(int(machine_id), 0))
            predictions.append({
                "machine_id":       machine_id,
                "imminence":        None,
                "severity_class":   None,
                "p_no_spike":       None,
                "p_moderate":       None,
                "p_severe":         None,
                "p_severe_ovr":     None,
                "is_spike":         None,
                "is_severe":        None,
                "alarm_threshold":  None,
                "threshold_used":   None,
                "threshold_source": None,
                "status":           "cold_start",
                "observations_used": n_obs,
            })

    predictions.sort(key=lambda p: p["machine_id"])
    return predictions


# ── Public entry point ────────────────────────────────────────────────────────


def predict(input_df: pd.DataFrame, model_dir: str | Path) -> dict:
    """Run spike prediction on a pre-aggregated rolling window DataFrame.

    This is the importable entry point for callers that construct the input
    DataFrame programmatically rather than from a CSV file.

    Parameters
    ----------
    input_df  : per-machine aggregated data (cluster_agg format, last 120 min).
    model_dir : directory with spike_model.json, spike_config.json, etc.

    Returns
    -------
    Prediction envelope dict (same structure as the JSON written by the CLI).

    Raises
    ------
    ValueError       on schema violations.
    FileNotFoundError on missing artefact files.
    RuntimeError     on feature column mismatch.
    """
    _validate_input(input_df)
    artifacts   = _load_artifacts(Path(model_dir))
    feature_df, status_map = _build_features(input_df, artifacts)
    predictions = _run_inference(feature_df, input_df, status_map, artifacts)

    n_predicted   = sum(1 for p in predictions if p["status"] != "cold_start")
    n_cold_start  = sum(1 for p in predictions if p["status"] == "cold_start")

    return {
        "predicted_at":        datetime.now(timezone.utc).isoformat(),
        "model_dir":           str(Path(model_dir).resolve()),
        "horizon_minutes":     artifacts.horizon_minutes,
        "machines_total":      int(input_df["machine_id"].nunique()),
        "machines_predicted":  n_predicted,
        "machines_cold_start": n_cold_start,
        "predictions":         predictions,
    }


def predict_with_artifacts(input_df: pd.DataFrame, artifacts: _Artifacts) -> dict:
    """Run spike prediction using pre-loaded artefacts.

    Identical to predict() but skips the artefact loading step.  Use this in
    long-running services (e.g. the FastAPI server) where artefacts are loaded
    once at startup and reused across requests.

    Parameters
    ----------
    input_df  : per-machine aggregated data (cluster_agg format, last 120 min).
    artifacts : artefacts returned by _load_artifacts(), loaded at startup.

    Returns
    -------
    Same envelope dict as predict().

    Raises
    ------
    ValueError  on schema violations.
    """
    _validate_input(input_df)
    feature_df, status_map = _build_features(input_df, artifacts)
    predictions = _run_inference(feature_df, input_df, status_map, artifacts)

    n_predicted  = sum(1 for p in predictions if p["status"] != "cold_start")
    n_cold_start = sum(1 for p in predictions if p["status"] == "cold_start")

    return {
        "predicted_at":        datetime.now(timezone.utc).isoformat(),
        "model_dir":           str(artifacts.model_dir),
        "horizon_minutes":     artifacts.horizon_minutes,
        "machines_total":      int(input_df["machine_id"].nunique()),
        "machines_predicted":  n_predicted,
        "machines_cold_start": n_cold_start,
        "predictions":         predictions,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "predict_spike.py",
        description = (
            "Predict CPU spikes 60 minutes ahead for each machine in the input window.\n"
            "JSON output goes to stdout; progress logs go to stderr."
        ),
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", "-i",
        required = True,
        metavar  = "CSV",
        help     = "Per-machine aggregated data CSV (cluster_agg format, last 120 min)",
    )
    p.add_argument(
        "--model-dir", "-m",
        required = True,
        metavar  = "DIR",
        help     = "Directory containing spike_model.json, spike_config.json, "
                   "spike_model.meta.json, spike_thresholds.parquet",
    )
    p.add_argument(
        "--output", "-o",
        default = None,
        metavar = "JSON",
        help    = "Also write predictions to this file (atomic write — safe on failure)",
    )
    return p


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    input_path = Path(args.input)
    model_dir  = Path(args.model_dir)

    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        sys.exit(1)
    if not model_dir.is_dir():
        log.error("Model directory not found: %s", model_dir)
        sys.exit(1)

    log.info("═" * 62)
    log.info("  SPIKE PREDICTOR")
    log.info("  Input     : %s", input_path)
    log.info("  Model dir : %s", model_dir)
    log.info("═" * 62)

    try:
        input_df = pd.read_csv(args.input)
        _validate_input(input_df)
    except ValueError as exc:
        log.error("Input validation failed: %s", exc)
        sys.exit(1)

    if input_df["machine_id"].nunique() == 0:
        log.error("Input contains no machines — nothing to predict.")
        sys.exit(2)

    try:
        result = predict(input_df, model_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        log.error("%s", exc)
        sys.exit(1)

    log.info(
        "  Predicted : %d machines  |  cold-start skipped : %d",
        result["machines_predicted"],
        result["machines_cold_start"],
    )
    log.info("  Horizon   : %d min ahead", result["horizon_minutes"])

    output_json = json.dumps(result, indent=2)

    # Always emit to stdout for piping / capture
    print(output_json)

    # Atomic write to file: write to a temp file in the same directory,
    # then rename — guarantees the output file is either complete or absent,
    # never a partial write.
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=out_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(output_json)
            os.replace(tmp, out_path)
        except Exception:
            os.unlink(tmp)
            raise
        log.info("  Output    : %s", out_path)


if __name__ == "__main__":
    main()
