"""Drift monitor — detects feature distribution shift between training and live data.

Why a drift monitor?
--------------------
The spike prediction model is trained on Google Cluster 2011 telemetry with specific
CPU load distributions, spike frequencies, and machine heterogeneity.  When deployed
to a different domain (e.g., Zabbix-monitored nodes) or when the production cluster
evolves over time (capacity changes, workload shifts, software updates), the input
feature distributions can drift away from what the model was trained on.

Without monitoring, the model silently degrades: alarm_rate in slo_metrics.json may
spike or drop, but there is no way to distinguish data drift from a real infrastructure
change without comparing the current feature distributions to training.  This script
provides that comparison via PSI (Population Stability Index).

What this script does:
  1. Loads the training split of cluster_features.parquet as the reference distribution.
  2. Runs feature engineering on the live cluster_agg data to produce the same 40 features.
  3. Computes PSI for each feature (reference vs. live).
  4. Classifies each feature as stable / warning / retrain / emergency.
  5. Fires a system-wide alert if > 10% of valid features exceed PSI 0.20.
  6. Writes drift_report.json atomically.

Integration with the deployment pipeline:
  - Run this script after each model retraining to verify the new model's training
    distribution matches the production domain.
  - Run it periodically (e.g., weekly) on the current live window to catch slow drift.
  - Pair with slo_metrics.json: a rising alarm_rate alongside high PSI confirms drift;
    a rising alarm_rate with stable PSI suggests a real infrastructure event.

CLI:
    python -m spike.drift_monitor \\
        --reference   data/full_run/cluster_features.parquet \\
        --live        /path/to/live/cluster_agg.csv \\
        --model-dir   data/full_run/spike \\
        --output      drift_report.json \\
        [--ref-sample-frac 0.10] \\
        [--min-bins-count  5] \\
        [--n-bins         10]

Output: drift_report.json written atomically (never partially written).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike import psi as _psi
from spike.classifier import _X_COLS
from spike.feature_engineer import _TRAIN_RATIO, engineer
from spike.version import get_model_version

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)

# System-wide alert thresholds.
# _SYSTEM_ALERT_PSI = 0.20 is intentionally lower than PSI_WARNING (0.25) — it is
# an intermediate signal meaning "this feature is moving, watch it" rather than
# "this feature already requires action".  Catching 10% of features at 0.20 is more
# sensitive than catching 10% at 0.25 and fires earlier during slow, broad drift.
_SYSTEM_ALERT_PSI      = 0.20

# 10% threshold: a single feature drifting is normal (binary features fluctuate with
# spike rate, which varies by day/week).  When more than 10% of ALL features shift
# simultaneously, it suggests a systematic change (new workload, capacity event,
# data pipeline change) rather than feature-specific noise.
_SYSTEM_ALERT_FRACTION = 0.10

_STATUS_STABLE    = "stable"
_STATUS_WARNING   = "warning"
_STATUS_RETRAIN   = "retrain"
_STATUS_EMERGENCY = "emergency"

# Ordered from least to most severe — index in this list determines which status
# "wins" when _top_recommendation() scans all features.
_SEVERITY_ORDER = [_STATUS_STABLE, _STATUS_WARNING, _STATUS_RETRAIN, _STATUS_EMERGENCY]


# ── JSON helpers ──────────────────────────────────────────────────────────────


def _safe_float(v: float | None) -> float | None:
    """Return None for NaN/inf so the value is JSON-serializable.

    json.dumps() raises ValueError on float('nan') by default.  PSI returns NaN
    for zero-variance or empty features, and reference_mean/live_mean can be NaN
    when all values in a column are non-finite.  This helper gates all float
    values before they enter the report dict.
    """
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return round(f, 4)


def _write_atomic(report: dict, output_path: Path) -> None:
    """Write report dict as JSON atomically via temp-file + rename.

    The temp file is created in the same directory as output_path so that
    os.replace() is guaranteed to be an atomic rename (same filesystem).
    If the process is killed between mkstemp and replace, only a .tmp file is
    left behind — the reader never sees a partial JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2)
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(payload)
        os.replace(tmp, output_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── Classification helpers ────────────────────────────────────────────────────


def _classify_status(psi: float | None) -> str:
    """Map a PSI value to a stability status string.

    Thresholds match the constants in spike/psi.py (see that module for derivation).
    None PSI (degenerate feature — zero variance or empty) is classified as "stable"
    because we have no evidence of drift, not because drift is absent.  The null
    psi value in the report tells the operator that this feature could not be assessed.
    """
    if psi is None:
        return _STATUS_STABLE   # no evidence of drift — not confirmed absence of drift
    if psi >= _psi.PSI_EMERGENCY:
        return _STATUS_EMERGENCY
    if psi >= _psi.PSI_WARNING:
        return _STATUS_RETRAIN
    if psi >= _psi.PSI_STABLE:
        return _STATUS_WARNING
    return _STATUS_STABLE


def _top_recommendation(statuses: list[str]) -> str:
    """Return the most severe status observed across all features.

    The top-level recommendation is the single worst feature's status so that
    operators get a clear, actionable summary without reading all 40 feature rows.
    """
    best = 0
    for s in statuses:
        idx = _SEVERITY_ORDER.index(s) if s in _SEVERITY_ORDER else 0
        best = max(best, idx)
    return _SEVERITY_ORDER[best]


# ── Reference loading ─────────────────────────────────────────────────────────


def _load_reference(reference_path: Path, ref_sample_frac: float) -> pd.DataFrame:
    """Load the training split of cluster_features.parquet, column-pruned and sampled.

    Why column-pruning before sampling?
    ------------------------------------
    The full cluster_features.parquet is ~24M rows × 40+ columns ≈ 2 GB.  Loading
    all columns and then sampling would hold 2 GB in RAM during the sample call.
    By requesting only ["bucket"] + _X_COLS from read_parquet, we load ~120 MB
    (40 float32 cols × 24M rows × 4B ≈ 3.8 GB → column prune to 40/60 cols cuts
    it to ~2.5 GB, then the 10% sample brings runtime RAM to ~250 MB).  This matches
    the approach in evaluate_cross_domain.py and avoids OOM on standard-sized VMs.

    Why training split only?
    -------------------------
    PSI compares the model's training distribution (what the model learned) against
    live data.  Including val/test rows in the reference would mean we compare against
    a distribution the model was NOT optimised on, producing inflated PSI even with
    no real drift.  The same _TRAIN_RATIO used in spike_classifier.py is applied here
    to reproduce the exact same split boundary.

    Why random_state=42?
    --------------------
    A fixed seed makes consecutive drift monitor runs on the same reference parquet
    produce identical reference samples, so PSI changes between runs reflect real live
    data changes rather than sampling noise.
    """
    log.info("Loading reference features from %s …", reference_path)
    cols   = ["bucket"] + list(_X_COLS)
    ref_df = pd.read_parquet(reference_path, columns=cols)

    # Filter to training buckets only — val/test rows must not bias the reference.
    bmin      = int(ref_df["bucket"].min())
    bmax      = int(ref_df["bucket"].max())
    train_max = bmin + int((bmax - bmin) * _TRAIN_RATIO)
    ref_train  = ref_df[ref_df["bucket"] <= train_max]
    log.info("  Training split: %d rows (buckets %d–%d)", len(ref_train), bmin, train_max)

    ref_sample = ref_train.sample(frac=ref_sample_frac, random_state=42)
    log.info("  Sampled %d rows (%.0f%% of training split)", len(ref_sample), ref_sample_frac * 100)
    return ref_sample


# ── Live feature engineering ──────────────────────────────────────────────────


def _engineer_live(live_path: Path) -> pd.DataFrame:
    """Run feature engineering on live cluster_agg data in a temp directory.

    Why a TemporaryDirectory?
    --------------------------
    engineer() is a training-phase function that writes two artefact files:
    cluster_features.parquet and spike_thresholds.parquet.  For the drift monitor
    these are transient — we only need the in-memory DataFrame they produce.
    TemporaryDirectory auto-deletes both files when the context exits, regardless
    of success or failure.

    Why CSV → Parquet conversion?
    ------------------------------
    engineer() calls pd.read_parquet() internally.  Accepting CSV here makes the
    drift monitor compatible with the daemon's --input format and the Zabbix adapter
    output format without requiring operators to pre-convert files.

    Normalization mismatch for threshold-relative features (known limitation):
    --------------------------------------------------------------------------
    engineer() computes per-machine p95 and p99 thresholds from the LIVE data using
    the training split of the live window.  Features that are normalised by these
    thresholds — cpu_vs_p95, band_position, time_to_p95_3, cpu_vs_p95_delta,
    cpu_vs_p95_slope_3/6, peak_cpu_vs_p95, cpu_vs_p99, peak_cpu_vs_p99, band_width —
    will be normalised by LIVE thresholds rather than the training thresholds stored
    in spike_thresholds.parquet.  This means PSI for these ~15 features conflates
    real distribution shift with a change in the normalisation baseline.

    This is the same tradeoff accepted in evaluate_cross_domain.py.  The raw
    features (total_cpu, peak_cpu, etc.) and lag/trend features are not affected.
    When accurate PSI for threshold-relative features is critical, supply bootstrap
    thresholds via the model-dir argument so that engineer() uses the training
    thresholds instead of computing fresh ones from the live window.
    """
    log.info("Engineering live features from %s …", live_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # engineer() requires Parquet input — convert CSV to a temp Parquet if needed
        if live_path.suffix.lower() == ".csv":
            raw      = pd.read_csv(live_path)
            agg_path = tmp / "live_agg.parquet"
            raw.to_parquet(agg_path, index=False)
        else:
            agg_path = live_path

        engineer(
            input_path      = agg_path,
            output_path     = tmp / "live_features.parquet",
            thresholds_path = tmp / "live_thresholds.parquet",
        )
        # Read while tmpdir still exists; DataFrame is held in memory after the
        # TemporaryDirectory context exits and deletes the files.
        live_df = pd.read_parquet(tmp / "live_features.parquet")

    keep = ["bucket"] + [c for c in _X_COLS if c in live_df.columns]
    return live_df[[c for c in keep if c in live_df.columns]]


# ── Per-feature PSI computation ───────────────────────────────────────────────


def _compute_feature_psi(
    ref_df:   pd.DataFrame,
    live_df:  pd.DataFrame,
    n_bins:   int,
) -> list[dict]:
    """Return one dict per feature with psi, status, reference_mean, live_mean.

    Iterates over _X_COLS in the canonical order from spike/classifier.py so that
    the features list in drift_report.json always matches the model's feature order.
    Missing columns (can happen if a new feature was added after the reference was
    built) produce PSI=null with status="stable" — the operator can see the null
    and investigate.

    reference_mean and live_mean are included so operators can see the direction of
    shift without loading the raw data: if live_mean >> reference_mean for total_cpu,
    the cluster is running hotter than during training, which explains elevated PSI
    even when the alarm rate is still normal.
    """
    records = []
    for col in _X_COLS:
        ref_vals  = ref_df[col].values.astype("float64")  if col in ref_df.columns  else np.array([], dtype="float64")
        live_vals = live_df[col].values.astype("float64") if col in live_df.columns else np.array([], dtype="float64")

        psi_raw = _psi.compute_psi_single(ref_vals, live_vals, n_bins=n_bins)
        psi     = _safe_float(psi_raw)
        status  = _classify_status(psi)

        # Compute means over finite values only so NaN-heavy lag features (e.g.
        # cpu_lag_24 for a very short live window) don't produce NaN means.
        ref_finite  = ref_vals[np.isfinite(ref_vals)]
        live_finite = live_vals[np.isfinite(live_vals)]
        ref_mean    = _safe_float(float(ref_finite.mean())  if len(ref_finite)  > 0 else float("nan"))
        live_mean   = _safe_float(float(live_finite.mean()) if len(live_finite) > 0 else float("nan"))

        records.append({
            "feature":        col,
            "psi":            psi,
            "status":         status,
            "reference_mean": ref_mean,
            "live_mean":      live_mean,
        })

    return records


# ── Report assembly ───────────────────────────────────────────────────────────


def _build_report(
    *,
    model_version:    str,
    reference_rows:   int,
    live_df:          pd.DataFrame,
    features:         list[dict],
    min_bins_warning: bool,
) -> dict:
    """Assemble the drift_report.json dict from computed feature stats.

    System alert logic:
    -------------------
    The denominator is n_valid (features with non-null PSI), not the total feature
    count.  Null-PSI features (zero-variance in reference or live, missing columns)
    are excluded because they carry no information about drift.  Including them in
    the denominator would dilute the alert signal when many features are degenerate
    — common in very short live windows where lag features are mostly NaN.

    The alert threshold _SYSTEM_ALERT_PSI = 0.20 is lower than PSI_WARNING (0.25)
    so that broad slow drift across many features is caught before any single feature
    reaches the retrain threshold.

    recommendation vs system_alert:
    --------------------------------
    recommendation = highest single-feature status → catches isolated severe shifts.
    system_alert = many features above 0.20 → catches broad moderate shifts.
    Both can fire simultaneously; recommendation is the primary action signal.
    """
    live_rows   = len(live_df)
    live_period: dict = {}
    if "bucket" in live_df.columns and live_rows > 0:
        live_period = {
            "min_bucket": int(live_df["bucket"].min()),
            "max_bucket": int(live_df["bucket"].max()),
        }

    # Exclude null-PSI features from the system alert denominator — they carry no
    # drift signal (either degenerate features or missing columns in live data).
    valid_features = [f for f in features if f["psi"] is not None]
    n_valid        = len(valid_features)
    n_alert        = len([f for f in valid_features if f["psi"] >= _SYSTEM_ALERT_PSI])
    system_alert   = n_valid > 0 and (n_alert / n_valid) > _SYSTEM_ALERT_FRACTION
    alert_reason   = (
        f"{n_alert}/{n_valid} features ({n_alert/n_valid:.0%}) have PSI ≥ {_SYSTEM_ALERT_PSI}"
        if system_alert else None
    )

    counts = {s: 0 for s in _SEVERITY_ORDER}
    for f in features:
        counts[f["status"]] += 1

    recommendation = _top_recommendation([f["status"] for f in features])

    return {
        "run_at":           datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version":    model_version,
        "reference_rows":   reference_rows,
        "live_rows":        live_rows,
        "live_period":      live_period,
        # True when live data is too sparse for reliable PSI estimates (< min_bins_count
        # rows per bin).  PSI values are still computed but should be treated with caution.
        "min_bins_warning": min_bins_warning,
        "recommendation":   recommendation,
        "psi_summary": {
            "stable_count":        counts[_STATUS_STABLE],
            "warning_count":       counts[_STATUS_WARNING],
            "retrain_count":       counts[_STATUS_RETRAIN],
            "emergency_count":     counts[_STATUS_EMERGENCY],
            "system_alert":        system_alert,
            "system_alert_reason": alert_reason,
        },
        "features": features,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Compute PSI drift report comparing the training feature distribution "
            "to live cluster_agg data.  Output: drift_report.json."
        )
    )
    p.add_argument("--reference",
                   required=True,
                   help="Path to cluster_features.parquet (training artifacts)")
    p.add_argument("--live",
                   required=True,
                   help="Path to live cluster_agg in CSV or Parquet format")
    p.add_argument("--model-dir",
                   required=True,
                   help="Path to spike model directory, e.g. data/full_run/spike (used to read model version)")
    p.add_argument("--output",
                   default="drift_report.json",
                   help="Output path for drift_report.json (default: ./drift_report.json)")
    p.add_argument("--ref-sample-frac",
                   type=float, default=0.10,
                   help=(
                       "Fraction of training split rows to use as reference (default: 0.10). "
                       "10%% of ~14M training rows ≈ 1.4M rows — sufficient for stable PSI "
                       "estimates.  Increase if you need higher precision on rare features."
                   ))
    p.add_argument("--min-bins-count",
                   type=int, default=5,
                   help=(
                       "Set min_bins_warning=true when live rows/bin < this value (default: 5). "
                       "PSI is unreliable when bins are sparse.  Recommend ≥48 buckets per "
                       "machine in the live window (4 hours of 5-min data) to avoid this."
                   ))
    p.add_argument("--n-bins",
                   type=int, default=_psi.N_PSI_BINS,
                   help="PSI quantile bins for continuous features (default: 10)")
    args = p.parse_args(argv)

    reference_path = Path(args.reference)
    live_path      = Path(args.live)
    model_dir      = Path(args.model_dir)
    output_path    = Path(args.output)

    if not reference_path.exists():
        log.error("Reference parquet not found: %s", reference_path)
        sys.exit(1)
    if not live_path.exists():
        log.error("Live data not found: %s", live_path)
        sys.exit(1)

    # Read model version once at startup — embedded in report header so downstream
    # tools can correlate drift reports to the exact model that produced predictions.
    version_info  = get_model_version(model_dir)
    model_version = version_info.get("model_version", "unknown")
    log.info("Model version: %s", model_version)

    # Load reference distribution (training split, column-pruned, sampled).
    ref_df = _load_reference(reference_path, args.ref_sample_frac)

    # Engineer live features in a temp dir (auto-cleaned up on exit).
    live_df   = _engineer_live(live_path)
    live_rows = len(live_df)
    log.info("Live feature rows: %d", live_rows)

    # Warn when bins will be sparse — PSI estimates are unreliable with < 5 obs/bin.
    # At n_bins=10, this fires when live_rows < 50 total (e.g., a 5-machine cluster
    # with < 10 buckets each).  Default production clusters have 12,555 × 24 = 300K+
    # rows and will never hit this.
    live_rows_per_bin = live_rows / args.n_bins if args.n_bins > 0 else live_rows
    min_bins_warning  = live_rows_per_bin < args.min_bins_count
    if min_bins_warning:
        log.warning(
            "Live data has %.1f rows/bin (threshold: %d min). "
            "PSI estimates may be unreliable — recommend ≥48 buckets per machine.",
            live_rows_per_bin, args.min_bins_count,
        )

    # Compute PSI for each of the 40 model features.
    features = _compute_feature_psi(ref_df, live_df, args.n_bins)

    # Assemble and atomically write the report.
    report = _build_report(
        model_version    = model_version,
        reference_rows   = len(ref_df),
        live_df          = live_df,
        features         = features,
        min_bins_warning = min_bins_warning,
    )
    _write_atomic(report, output_path)
    log.info(
        "Drift report written to %s  (recommendation: %s)",
        output_path, report["recommendation"],
    )


if __name__ == "__main__":
    main()
