#!/usr/bin/env python3
"""evaluate_domain.py — Retrospective backtesting: test the spike model on your own data.

This is a self-contained evaluation tool for new deployment domains (Type B
operators).  It requires only your historical cluster_agg data and the trained
model artifacts — no reference Google training data, no manual spike labels.

Why can we evaluate without manual labels?
------------------------------------------
Spike labels are derived algorithmically from the telemetry itself: "spike" =
CPU exceeds the machine's p95 threshold for ≥1 future 5-min window in the next
60 minutes.  This is programmatic weak supervision — the same approach used by
Netflix, Meta, and Google for offline anomaly detection backtesting.  No human
annotation is required because the label definition is a deterministic rule
applied to the same CPU time series used for features (just looking forward in
time, not backward — so there is no leakage).

What this script does
---------------------
1. Load your historical cluster_agg data (CSV or Parquet).
2. Warn if any machines have fewer than _MIN_BUCKETS_WARN 5-min buckets — lag
   features (cpu_lag_24, cpu_ewma_24) need ≥24 prior buckets per machine; with
   less history the threshold estimates are also unstable.
3. Run spike/feature_engineer.py in a TemporaryDirectory — p95/p99 thresholds
   are bootstrapped from the first 60% of your data (train split), labels are
   derived from forward-looking windows.
4. Split chronologically: train 60% / val 20% / test 20%.
5. Sweep alarm thresholds on the val split (F1-max) to recommend a domain-tuned
   operating point for each model.
6. Evaluate all 5 models on the test split: PR-AUC, ROC-AUC, precision/recall
   at the recommended threshold, and alarm rate.
7. Write domain_eval_report.json and print a human-readable summary.

Minimum data requirement
------------------------
≥ 2016 five-minute buckets (~7 days) per machine for stable p99 estimation,
consistent with the AWS CloudWatch anomaly detection warm-up guideline and
with academic sample-size recommendations for the 99th percentile
(210+ observations for 95% CI).  Below 288 buckets (24h) a warning is issued.

Sibling model discovery
-----------------------
Given --model-dir data/full_run/spike, the script looks for sibling model dirs
at data/full_run/spike_15m, data/full_run/spike_30m, etc.  Only models whose
directory exists are loaded; missing horizons produce a warning and are skipped.

Usage
-----
    python evaluation/evaluate_domain.py \\
        --input     data/your_domain/cluster_agg.csv \\
        --model-dir data/full_run/spike \\
        --output    data/your_domain_eval

    python evaluation/evaluate_domain.py \\
        --input     data/your_domain/cluster_agg.parquet \\
        --model-dir data/full_run/spike \\
        --output    data/your_domain_eval
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from spike.classifier import _VAL_RATIO, _X_COLS
from spike.feature_engineer import _TRAIN_RATIO, engineer

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)

# Machines with fewer buckets than this get a data-quality warning.
# 288 = 24 hours at 5-min intervals. Lag features require ≥24 prior
# buckets; below this most lag/EWMA features are NaN and threshold
# estimates are unstable.
_MIN_BUCKETS_WARN = 288

# Google Cluster 2011 test PR-AUC — used to compute retention percentage
# and drive the deployment recommendation in the report.
_SOURCE_PR_AUC = {
    "60m"       : 0.547,
    "15m"       : 0.575,
    "30m"       : 0.564,
    "45m"       : 0.563,
    "severe_ovr": 0.543,
}

# Labels that engineer() produces for each model's positive class.
_LABEL_COLS = {
    "60m"       : "severity_in_60m",
    "15m"       : "spike_in_15m",
    "30m"       : "spike_in_30m",
    "45m"       : "spike_in_45m",
    "severe_ovr": "spike_severe_ovr",   # derived post-engineering; see _derive_ovr_label()
}


# ── Utility ───────────────────────────────────────────────────────────────────

def _safe_float(v: Any) -> float | None:
    """Return None for NaN/inf so json.dumps never sees non-serialisable floats."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (f != f or f == float("inf") or f == float("-inf")) else round(f, 4)


# ── Input normalisation ───────────────────────────────────────────────────────

def _load_agg(input_path: Path) -> tuple[pd.DataFrame, Path, tempfile.TemporaryDirectory | None]:
    """Load cluster_agg into a DataFrame and, if CSV, produce a temp Parquet copy.

    engineer() calls pd.read_parquet() internally so it cannot accept CSV
    directly.  When a CSV is provided we write it to a TemporaryDirectory and
    return the resulting Parquet path alongside the in-memory DataFrame.

    Returns
    -------
    (df, parquet_path, tmpdir_or_None)
        The caller must call tmpdir.cleanup() after engineer() has finished.
        If input was already Parquet, tmpdir is None.

    Why coerce int64 for integer columns?
        CSV serialisation loses column dtype; machine_id/bucket/time_us must be
        int64 for downstream groupby and index operations to be type-stable.
    """
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
        for col in ("machine_id", "bucket", "time_us"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("int64")
        tmpdir      = tempfile.TemporaryDirectory()
        parquet_path = Path(tmpdir.name) / "domain_agg.parquet"
        df.to_parquet(parquet_path, index=False)
        return df, parquet_path, tmpdir
    else:
        df = pd.read_parquet(input_path)
        return df, input_path, None


# ── Data-quality checks ───────────────────────────────────────────────────────

def _check_schema(df: pd.DataFrame) -> list[str]:
    """Return a list of schema error messages (empty = OK)."""
    required = {"machine_id", "bucket", "time_us", "total_cpu", "peak_cpu",
                "total_mem", "peak_mem", "disk_io", "n_tasks"}
    missing = required - set(df.columns)
    errors = []
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        errors.append("Input data is empty.")
    return errors


def _check_coverage(df: pd.DataFrame) -> list[str]:
    """Warn if any machine has fewer than _MIN_BUCKETS_WARN buckets."""
    warnings_out = []
    counts = df.groupby("machine_id")["bucket"].nunique()
    low    = counts[counts < _MIN_BUCKETS_WARN]
    if not low.empty:
        warnings_out.append(
            f"{len(low)}/{len(counts)} machines have fewer than {_MIN_BUCKETS_WARN} "
            f"buckets ({_MIN_BUCKETS_WARN * 5 / 60:.0f}h). "
            "Lag and EWMA features will be mostly NaN for these machines — "
            "PR-AUC estimates may be unreliable. "
            "Recommend ≥2016 buckets (~7 days) per machine."
        )
    return warnings_out


# ── Feature engineering ───────────────────────────────────────────────────────

def _engineer_domain(parquet_path: Path) -> pd.DataFrame:
    """Run feature engineering inside a TemporaryDirectory.

    Why TemporaryDirectory?
        engineer() writes two artefact files (cluster_features.parquet and
        spike_thresholds.parquet) as side effects — they are not optional.
        For this evaluation script those files are transient: only the
        in-memory DataFrame is needed.  TemporaryDirectory guarantees both
        files are deleted on exit, even on error.

    Why train_ratio=_TRAIN_RATIO (0.6)?
        The first 60% of the data is used to compute per-machine p95/p99
        thresholds.  Using the full dataset would leak val/test CPU
        distributions into the thresholds, inflating spike-history features
        for those periods.  Matching the training-time ratio also makes the
        thresholds comparable to what predict.py would use after bootstrap.

    Why min_future_windows=1 (K=1)?
        K=1 maximises label coverage on small datasets.  The production
        training uses K=2, but cross-domain evaluation always uses K=1 so
        that short datasets (few weeks) still yield enough positive labels
        for a meaningful PR-AUC.  This is the same choice as
        evaluate_cross_domain.py.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        engineer(
            input_path         = parquet_path,
            output_path        = tmp / "domain_features.parquet",
            thresholds_path    = tmp / "domain_thresholds.parquet",
            train_ratio        = _TRAIN_RATIO,
            min_future_windows = 1,
        )
        return pd.read_parquet(tmp / "domain_features.parquet")


def _derive_ovr_label(features_df: pd.DataFrame) -> pd.DataFrame:
    """Add spike_severe_ovr = 1 where severity_in_60m == 2 (severe class).

    The OVR severe model is a binary classifier trained on spike-positive rows
    only (moderate + severe), predicting severe vs moderate.  At inference it
    is gated by the 60m model: only rows where p(any spike) >= 0.15 reach the
    OVR model.  For evaluation purposes we derive the label from the 60m
    multiclass label without any gating — this gives an upper-bound estimate
    of performance (the gating only reduces the effective test set).
    """
    if "severity_in_60m" in features_df.columns:
        features_df = features_df.copy()
        features_df["spike_severe_ovr"] = (
            features_df["severity_in_60m"] == 2
        ).astype("float32")
    return features_df


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model(model_dir: Path) -> dict:
    """Load a single model's booster, feature list, alarm threshold, and calibrators.

    Raises FileNotFoundError if any required artefact (spike_model.json,
    spike_model.meta.json, spike_config.json) is absent.  calibrators.pkl is
    optional — only the 60m severity model ships with isotonic calibrators.
    """
    for name in ("spike_model.json", "spike_model.meta.json", "spike_config.json"):
        if not (model_dir / name).exists():
            raise FileNotFoundError(f"Required artefact not found: {model_dir / name}")

    meta   = json.loads((model_dir / "spike_model.meta.json").read_text())
    config = json.loads((model_dir / "spike_config.json").read_text())

    booster = xgb.Booster()
    booster.load_model(str(model_dir / "spike_model.json"))
    booster.set_param("nthread", 1)

    calibrators = None
    cal_path    = model_dir / "calibrators.pkl"
    if cal_path.exists():
        with open(cal_path, "rb") as f:
            calibrators = pickle.load(f)

    return {
        "booster"         : booster,
        "feature_cols"    : meta["feature_cols"],
        "alarm_threshold" : float(config["alarm_threshold"]),
        "calibrators"     : calibrators,
    }


def _discover_models(model_dir: Path) -> dict[str, dict]:
    """Load all available model variants.

    Sibling model dirs sit next to model_dir under the same parent:
        model_dir            = data/full_run/spike       → 60m severity
        model_dir.parent / spike_15m                     → 15m binary
        model_dir.parent / spike_30m                     → 30m binary
        …

    Missing sibling dirs are skipped with a warning rather than raising an
    error, so the script remains useful when only a subset of models are
    present (e.g., deployment zip with only the 60m model).
    """
    base   = model_dir.name    # e.g. "spike"
    parent = model_dir.parent  # e.g. data/full_run/

    candidates = {
        "60m"       : model_dir,
        "15m"       : parent / f"{base}_15m",
        "30m"       : parent / f"{base}_30m",
        "45m"       : parent / f"{base}_45m",
        "severe_ovr": parent / f"{base}_severe_ovr",
    }

    models: dict[str, dict] = {}
    for name, mdir in candidates.items():
        if not mdir.exists():
            log.warning("  Model dir not found — skipping %s: %s", name, mdir)
            continue
        try:
            models[name] = _load_model(mdir)
            log.info("  Loaded %-11s  alarm @ %.2f", name, models[name]["alarm_threshold"])
        except FileNotFoundError as exc:
            log.warning("  Model %s incomplete — skipping: %s", name, exc)

    return models


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _calibrate_60m(raw: np.ndarray, calibrators: list | None) -> np.ndarray:
    """Apply per-class isotonic calibration to 60m softmax output; renormalise."""
    if calibrators is None:
        return raw
    cal = np.column_stack([
        np.clip(calibrators[k].predict(raw[:, k].astype("float64")), 0.0, 1.0)
        for k in range(3)
    ])
    row_sums = cal.sum(axis=1, keepdims=True)
    uniform  = np.full_like(cal, 1.0 / 3)
    return np.where(row_sums > 0, cal / row_sums, uniform).astype("float32")


def _threshold_sweep(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float, float, float]:
    """Sweep thresholds [0.05, 0.95] and return (best_threshold, precision, recall, f1).

    F1-max on the validation split is the recommended operating point for
    initial deployment.  Operators can tune further based on their FP/FN cost
    ratio using the full precision/recall table in spike_config.json.
    """
    best_t, best_f1, best_p, best_r = 0.5, 0.0, 0.0, 0.0
    for t in np.arange(0.05, 0.96, 0.05):
        preds = (scores >= t).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_t, best_p, best_r = f1, float(t), prec, rec
    return best_t, best_p, best_r, best_f1


def _eval_60m(
    features_df: pd.DataFrame,
    val_mask:    pd.Series,
    test_mask:   pd.Series,
    model_info:  dict,
) -> dict:
    """Evaluate 60m severity model: threshold sweep on val, PR-AUC/ROC-AUC on test."""
    label_col = "severity_in_60m"
    val_df    = features_df[val_mask].dropna(subset=[label_col])
    test_df   = features_df[test_mask].dropna(subset=[label_col])

    if len(test_df) == 0:
        return {"error": "no test rows after label drop"}

    feat_cols = model_info["feature_cols"]
    X_val  = val_df[feat_cols].astype("float32").values
    X_test = test_df[feat_cols].astype("float32").values
    y_val  = val_df[label_col].astype(int).values
    y_test = test_df[label_col].astype(int).values

    raw_val  = model_info["booster"].inplace_predict(X_val).reshape(-1, 3)
    raw_test = model_info["booster"].inplace_predict(X_test).reshape(-1, 3)

    prob_val  = _calibrate_60m(raw_val,  model_info["calibrators"])
    prob_test = _calibrate_60m(raw_test, model_info["calibrators"])

    # Threshold sweep on val using P(any spike) = 1 - P(no_spike)
    p_spike_val = 1.0 - prob_val[:, 0].astype("float64")
    best_t, best_p, best_r, best_f1 = _threshold_sweep(
        (y_val > 0).astype(int), p_spike_val
    )

    # Per-class and macro PR-AUC / ROC-AUC on test
    pr_aucs, roc_aucs = [], []
    per_class: dict = {}
    for cls in range(3):
        y_bin = (y_test == cls).astype(int)
        n_pos = int(y_bin.sum())
        if n_pos == 0 or n_pos == len(y_bin):
            continue
        pa  = float(average_precision_score(y_bin, prob_test[:, cls]))
        ra  = float(roc_auc_score(y_bin, prob_test[:, cls]))
        pr_aucs.append(pa)
        roc_aucs.append(ra)
        per_class[f"class_{cls}"] = {"pr_auc": _safe_float(pa), "roc_auc": _safe_float(ra)}

    macro_pr_auc  = _safe_float(np.mean(pr_aucs))  if pr_aucs  else None
    macro_roc_auc = _safe_float(np.mean(roc_aucs)) if roc_aucs else None

    p_spike_test  = 1.0 - prob_test[:, 0].astype("float64")
    alarm_rate    = _safe_float(float((p_spike_test >= best_t).mean()))

    return {
        "n_val"            : len(val_df),
        "n_test"           : len(test_df),
        "class_rates"      : {f"class_{c}": _safe_float(float((y_test == c).mean())) for c in range(3)},
        "macro_pr_auc"     : macro_pr_auc,
        "macro_roc_auc"    : macro_roc_auc,
        "per_class"        : per_class,
        "alarm_threshold"  : _safe_float(best_t),
        "alarm_precision"  : _safe_float(best_p),
        "alarm_recall"     : _safe_float(best_r),
        "alarm_f1"         : _safe_float(best_f1),
        "alarm_rate"       : alarm_rate,
    }


def _eval_binary(
    features_df: pd.DataFrame,
    val_mask:    pd.Series,
    test_mask:   pd.Series,
    model_info:  dict,
    label_col:   str,
) -> dict:
    """Evaluate a binary horizon model: threshold sweep on val, PR-AUC on test."""
    if label_col not in features_df.columns:
        return {"error": f"label column '{label_col}' not in feature set"}

    val_df  = features_df[val_mask].dropna(subset=[label_col])
    test_df = features_df[test_mask].dropna(subset=[label_col])

    if len(val_df) < 100:
        return {"error": f"val split too small for threshold sweep: {len(val_df)} rows"}
    if len(test_df) == 0:
        return {"error": "no test rows after label drop"}

    feat_cols = model_info["feature_cols"]
    X_val  = val_df[feat_cols].astype("float32").values
    X_test = test_df[feat_cols].astype("float32").values
    y_val  = val_df[label_col].astype(int).values
    y_test = test_df[label_col].astype(int).values

    # inplace_predict returns shape (n, 1) for binary:logistic — take column 0
    p_val  = model_info["booster"].inplace_predict(X_val,  strict_shape=True)[:, 0].astype("float64")
    p_test = model_info["booster"].inplace_predict(X_test, strict_shape=True)[:, 0].astype("float64")

    best_t, best_p, best_r, best_f1 = _threshold_sweep(y_val, p_val)

    n_pos_test = int(y_test.sum())
    if n_pos_test == 0 or n_pos_test == len(y_test):
        return {
            "n_val": len(val_df), "n_test": len(test_df), "n_pos": n_pos_test,
            "pr_auc": None, "roc_auc": None,
            "error": "degenerate test labels (all same class)",
        }

    pr_auc  = _safe_float(float(average_precision_score(y_test, p_test)))
    roc_auc = _safe_float(float(roc_auc_score(y_test, p_test)))

    return {
        "n_val"          : len(val_df),
        "n_test"         : len(test_df),
        "n_pos"          : n_pos_test,
        "pos_rate"       : _safe_float(float(y_test.mean())),
        "pr_auc"         : pr_auc,
        "roc_auc"        : roc_auc,
        "alarm_threshold": _safe_float(best_t),
        "alarm_precision": _safe_float(best_p),
        "alarm_recall"   : _safe_float(best_r),
        "alarm_f1"       : _safe_float(best_f1),
        "alarm_rate"     : _safe_float(float((p_test >= best_t).mean())),
    }


# ── Recommendation ────────────────────────────────────────────────────────────

def _retention_recommendation(pr_auc: float | None, horizon: str) -> tuple[str, str]:
    """Return (retention_pct_str, action) based on PR-AUC vs Google source."""
    source = _SOURCE_PR_AUC.get(horizon)
    if pr_auc is None or source is None or source <= 0:
        return "N/A", "Cannot determine — PR-AUC not available."
    retention = pr_auc / source
    if retention >= 0.70:
        action = "Deploy. Model transfers well to this domain."
    elif retention >= 0.40:
        action = "Adapt. Tune alarm threshold or fine-tune with ~300 warm-start boosting rounds."
    else:
        action = "Retrain from scratch on domain data (use Google best_params.json as Optuna seed)."
    return f"{retention:.0%}", action


# ── Terminal summary ──────────────────────────────────────────────────────────

def _print_summary(results: dict, warnings: list[str]) -> None:
    """Print a concise human-readable summary to stdout."""
    sep = "═" * 62
    print(sep)
    print("  DOMAIN EVALUATION SUMMARY")
    print(f"  Input : {results['input_path']}")
    print(f"  Run at: {results['run_at']}")
    print(sep)

    if warnings:
        print("\n  DATA QUALITY WARNINGS:")
        for w in warnings:
            print(f"  ⚠  {w}")

    print(f"\n  Machines : {results['n_machines']}")
    print(f"  Coverage : {results.get('coverage_days', '?'):.1f} days")
    print(f"  Test rows: val={results.get('val_rows','?')}  test={results.get('test_rows','?')}")

    print("\n  Model results (test split):")
    print(f"  {'Horizon':<12} {'PR-AUC':>8} {'Source':>8} {'Retention':>10} "
          f"{'ROC-AUC':>8} {'Alarm@':>8} {'F1':>6}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*6}")

    for h_name, h_label in [
        ("60m",        "60m sev"),
        ("15m",        "15m bin"),
        ("30m",        "30m bin"),
        ("45m",        "45m bin"),
        ("severe_ovr", "OVR sev"),
    ]:
        m = results["models"].get(h_name)
        if m is None or "error" in m:
            print(f"  {h_label:<12} {'—':>8}  (skipped: {(m or {}).get('error', 'not loaded')})")
            continue

        pr_auc  = m.get("macro_pr_auc") or m.get("pr_auc")
        roc_auc = m.get("macro_roc_auc") or m.get("roc_auc")
        thr     = m.get("alarm_threshold")
        f1      = m.get("alarm_f1")
        ret, _  = _retention_recommendation(pr_auc, h_name)

        def _f(v: Any) -> str:
            return f"{v:.3f}" if isinstance(v, float) else (str(v) if v is not None else "—")

        print(f"  {h_label:<12} {_f(pr_auc):>8} {_SOURCE_PR_AUC.get(h_name, 0):>8.3f} "
              f"{ret:>10} {_f(roc_auc):>8} {_f(thr):>8} {_f(f1):>6}")

    # Overall recommendation based on 60m model
    m60 = results["models"].get("60m", {})
    pr60 = m60.get("macro_pr_auc") if m60 and "error" not in m60 else None
    _, action = _retention_recommendation(pr60, "60m")
    print(f"\n  Recommendation (based on 60m model): {action}")
    print(sep)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def evaluate(
    input_path: Path,
    model_dir:  Path,
    output_dir: Path,
) -> None:
    """Run the full domain backtesting pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_at = datetime.now(tz=timezone.utc).isoformat()

    log.info("═" * 62)
    log.info("  DOMAIN EVALUATION — %s", input_path)
    log.info("═" * 62)

    # ── Load and validate input ───────────────────────────────────────────────
    df_agg, agg_parquet, _tmpdir = _load_agg(input_path)
    try:
        schema_errors = _check_schema(df_agg)
        if schema_errors:
            for e in schema_errors:
                log.error("  Schema error: %s", e)
            sys.exit(1)

        coverage_warnings = _check_coverage(df_agg)
        for w in coverage_warnings:
            log.warning("  %s", w)

        n_machines = int(df_agg["machine_id"].nunique())
        log.info("  Machines: %d  |  Total rows: %d  |  Buckets: %d",
                 n_machines, len(df_agg), df_agg["bucket"].nunique())

        # ── Feature engineering ───────────────────────────────────────────────
        log.info("  Running feature engineering (thresholds from train split)…")
        features_df = _engineer_domain(agg_parquet)
        features_df = _derive_ovr_label(features_df)

        log.info("  Feature rows: %d  |  Label coverage: %.1f%%",
                 len(features_df),
                 100 * features_df["severity_in_60m"].notna().mean())

        # ── Chronological split ───────────────────────────────────────────────
        b_min = int(features_df["bucket"].min())
        b_max = int(features_df["bucket"].max())
        n_b   = b_max - b_min
        train_max = b_min + int(n_b * _TRAIN_RATIO)
        val_end   = b_min + int(n_b * (_TRAIN_RATIO + _VAL_RATIO))

        val_mask  = (features_df["bucket"] > train_max) & (features_df["bucket"] <= val_end)
        test_mask = features_df["bucket"] > val_end

        coverage_days = n_b * 5 / 60 / 24
        log.info("  Coverage: %.1f days  |  Val: %d rows  |  Test: %d rows",
                 coverage_days, int(val_mask.sum()), int(test_mask.sum()))

        # ── Load models ───────────────────────────────────────────────────────
        log.info("  Loading models from %s …", model_dir)
        models = _discover_models(model_dir)
        if not models:
            log.error("  No models loaded — check --model-dir path.")
            sys.exit(1)

        # ── Evaluate each model ───────────────────────────────────────────────
        model_results: dict[str, dict] = {}
        for h_name, model_info in models.items():
            label_col = _LABEL_COLS[h_name]
            log.info("  Evaluating %s …", h_name)
            if h_name == "60m":
                model_results[h_name] = _eval_60m(features_df, val_mask, test_mask, model_info)
            else:
                model_results[h_name] = _eval_binary(
                    features_df, val_mask, test_mask, model_info, label_col
                )

        # ── Build and write report ────────────────────────────────────────────
        all_warnings = coverage_warnings
        results = {
            "run_at"       : run_at,
            "input_path"   : str(input_path),
            "model_dir"    : str(model_dir),
            "n_machines"   : n_machines,
            "coverage_days": round(coverage_days, 2),
            "val_rows"     : int(val_mask.sum()),
            "test_rows"    : int(test_mask.sum()),
            "warnings"     : all_warnings,
            "models"       : model_results,
        }

        report_path = output_dir / "domain_eval_report.json"
        report_path.write_text(json.dumps(results, indent=2, default=str))
        log.info("  Report written: %s", report_path)

        _print_summary(results, all_warnings)

    finally:
        if _tmpdir is not None:
            _tmpdir.cleanup()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Retrospective backtesting: evaluate the spike model on your own cluster data. "
            "Derives spike labels programmatically from telemetry — no manual annotation needed."
        ),
    )
    p.add_argument(
        "--input", "-i",
        type    = Path,
        required = True,
        metavar = "PATH",
        help    = "Historical cluster_agg data in CSV or Parquet format.",
    )
    p.add_argument(
        "--model-dir",
        type    = Path,
        default = Path("data/full_run/spike"),
        metavar = "DIR",
        help    = "Path to the 60m model directory (default: data/full_run/spike). "
                  "Sibling models (spike_15m, spike_30m, …) are discovered automatically.",
    )
    p.add_argument(
        "--output", "-o",
        type    = Path,
        default = Path("data/domain_eval"),
        metavar = "DIR",
        help    = "Output directory for domain_eval_report.json (default: data/domain_eval).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if not args.input.exists():
        log.error("Input not found: %s — pass --input PATH to your cluster_agg file.", args.input)
        sys.exit(1)

    if not args.model_dir.exists():
        log.error(
            "Model dir not found: %s — pass --model-dir PATH to your model directory "
            "(e.g. data/full_run/spike).", args.model_dir
        )
        sys.exit(1)

    evaluate(
        input_path = args.input,
        model_dir  = args.model_dir,
        output_dir = args.output,
    )


if __name__ == "__main__":
    main()
