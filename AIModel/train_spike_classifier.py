"""CLI orchestrator for the CPU spike prediction training pipeline.

Wires together the three pipeline steps:

  Step 1  spike_preprocessor.py      CSV  →  cluster_agg.parquet
  Step 2  spike_feature_engineer.py  agg  →  cluster_features.parquet
  Step 3  spike_classifier.py        feat →  spike_model.json + artefacts

Each step caches its output.  If the output already exists the step is skipped
(unless --force or --from-step N overrides this).

Walk-forward cross-validation (--walk-forward, default on) uses
sklearn.model_selection.TimeSeriesSplit with 5 expanding folds on the training
portion.  Metrics are reported as mean ± std across folds before the final
single-model training run.  scale_pos_weight is computed per fold from that
fold's training data to avoid class-ratio leakage from future folds.

Threshold selection (automatic, no manual input required):
  find_threshold() is called on the validation split only.  The selected
  threshold and full precision/recall/F1 sweep table are written to
  spike_config.json alongside the model.

Feature importance:
  XGBoost gain importances are always printed and saved.
  SHAP values (more reliable for correlated lag features) are computed when
  the ``shap`` package is installed; silently skipped otherwise.

Reproducibility:
  All CLI arguments are written to run_config.json at the start of each run.
  Random seeds are set for numpy and XGBoost before any computation.

Output artefacts (all written to --artifacts-dir):
  cluster_agg.parquet        preprocessed aggregation (Step 1 cache)
  cluster_features.parquet   engineered features + labels (Step 2 cache)
  spike_model.json           XGBoost model (native format)
  spike_model.meta.json      feature column list + XGBoost version
  spike_config.json          threshold, split info, spike rates, CV summary
  run_config.json            all CLI args + ISO-8601 timestamp
  cv_results.csv             per-fold metrics (only with --walk-forward)
  feature_importance.csv     XGBoost gain + SHAP importances

Usage:
    python train_spike_classifier.py --data-path data/cluster_cpu_data.csv
    python train_spike_classifier.py --data-path data/cluster_cpu_data.csv --force
    python train_spike_classifier.py --data-path data/cluster_cpu_data.csv --from-step 3
    python train_spike_classifier.py --data-path data/cluster_cpu_data.csv --no-walk-forward
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from spike_preprocessor import preprocess
from spike_feature_engineer import engineer
from spike_classifier import (
    SpikeClassifier,
    _TRAIN_RATIO,
    _VAL_RATIO,
    _X_COLS,
    train as _train_model,
)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(message)s",
    datefmt = "%H:%M:%S",
    stream  = sys.stdout,
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_N_FOLDS:           int   = 5
_THRESHOLD_SWEEP:   list[float] = [round(t, 2) for t in np.arange(0.10, 1.0, 0.05).tolist()]

# ── Walk-forward cross-validation ─────────────────────────────────────────────


def _fold_metrics(
    clf:       SpikeClassifier,
    X:         pd.DataFrame,
    y:         pd.Series,
    threshold: float,
) -> dict[str, float]:
    """Evaluate classifier on one fold's val set, returning all scalar metrics."""
    proba     = clf.predict_proba(X)
    pred      = (proba >= threshold).astype(int)
    y_vals    = y.values
    n_classes = len(np.unique(y_vals))

    if n_classes < 2:
        pr_auc  = float("nan")
        roc_auc = float("nan")
    else:
        pr_auc  = float(average_precision_score(y_vals, proba))
        roc_auc = float(roc_auc_score(y_vals, proba))

    return {
        "pr_auc":    pr_auc,
        "roc_auc":   roc_auc,
        "precision": float(precision_score(y_vals, pred, zero_division=0)),
        "recall":    float(recall_score(y_vals, pred, zero_division=0)),
        "f1":        float(f1_score(y_vals, pred, zero_division=0)),
    }


def _run_walk_forward_cv(
    train_df:    pd.DataFrame,
    n_folds:     int,
    seed:        int,
    output_path: Path,
) -> dict[str, float]:
    """Run walk-forward CV on the training split.

    Uses sklearn.model_selection.TimeSeriesSplit (expanding window) so that
    each fold trains on older data and validates on the immediately following
    window — matching the deployment scenario.

    scale_pos_weight is computed per fold from that fold's training rows only,
    preventing class-ratio leakage from future folds.

    find_threshold() is called on each fold's validation data.

    Parameters
    ----------
    train_df    : training portion of cluster_features.parquet (labeled rows).
    n_folds     : number of TimeSeriesSplit folds.
    seed        : random seed passed to SpikeClassifier.
    output_path : destination for cv_results.csv.

    Returns
    -------
    dict with mean and std for each metric across folds.
    """
    log.info("─" * 62)
    log.info("  WALK-FORWARD CV  (%d folds, expanding window)", n_folds)
    log.info("─" * 62)

    tscv     = TimeSeriesSplit(n_splits=n_folds)
    X_all    = train_df[_X_COLS]
    y_all    = train_df["spike_in_60m"]
    indices  = np.arange(len(train_df))

    fold_rows: list[dict] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(indices), start=1):
        X_tr, y_tr = X_all.iloc[tr_idx], y_all.iloc[tr_idx]
        X_val, y_val = X_all.iloc[val_idx], y_all.iloc[val_idx]

        # Per-fold scale_pos_weight — never use full dataset ratio
        neg_tr = int((y_tr == 0).sum())
        pos_tr = int((y_tr == 1).sum())
        if pos_tr == 0:
            log.warning("  Fold %d: no positive examples in training portion — skipping", fold_idx)
            continue
        spw = neg_tr / pos_tr

        clf = SpikeClassifier(scale_pos_weight=spw, random_state=seed)
        clf.fit(X_tr, y_tr)

        # Threshold selected on this fold's validation data (no leakage to test)
        if y_val.sum() == 0:
            fold_thresh = 0.5
            log.warning("  Fold %d: no positive examples in validation — using threshold 0.5", fold_idx)
        else:
            fold_thresh = clf.find_threshold(X_val, y_val)

        m = _fold_metrics(clf, X_val, y_val, fold_thresh)

        log.info(
            "  Fold %d/%d  |  PR-AUC %.3f  ROC-AUC %.3f  F1 %.3f  "
            "P %.3f  R %.3f  thresh %.2f  (train=%d  val=%d  spw=%.1f)",
            fold_idx, n_folds,
            m["pr_auc"], m["roc_auc"], m["f1"],
            m["precision"], m["recall"], fold_thresh,
            len(tr_idx), len(val_idx), spw,
        )

        fold_rows.append({
            "fold":      fold_idx,
            "threshold": fold_thresh,
            "train_n":   len(tr_idx),
            "val_n":     len(val_idx),
            "spw":       round(spw, 4),
            **{k: round(v, 6) for k, v in m.items()},
        })

    if not fold_rows:
        log.warning("  No valid CV folds — skipping summary")
        return {}

    # Write per-fold CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_path, fold_rows)
    log.info("  CV results saved : %s", output_path)

    # Aggregate across folds — mean ± std
    metric_keys = ["pr_auc", "roc_auc", "precision", "recall", "f1"]
    summary: dict[str, float] = {}
    log.info("─" * 62)
    log.info("  CV SUMMARY")
    for key in metric_keys:
        vals = [r[key] for r in fold_rows if not np.isnan(r[key])]
        if not vals:
            summary[f"cv_{key}_mean"] = float("nan")
            summary[f"cv_{key}_std"]  = float("nan")
            continue
        mean = float(np.mean(vals))
        std  = float(np.std(vals))
        summary[f"cv_{key}_mean"] = mean
        summary[f"cv_{key}_std"]  = std
        log.info("  %-12s  mean %.3f  ±  std %.3f", key, mean, std)
    log.info("─" * 62)

    return summary


# ── Threshold sweep table ──────────────────────────────────────────────────────


def _threshold_sweep_table(
    clf:    SpikeClassifier,
    X_val:  pd.DataFrame,
    y_val:  pd.Series,
    total_val_rows: int,
) -> list[dict]:
    """Compute precision / recall / F1 / alarms_per_day for each threshold.

    ``alarms_per_day`` estimates how many times the model would fire per day
    on a node given the validation set density (val_rows over its time span).
    Buckets are 5 min each → 288 buckets/day.

    Parameters
    ----------
    clf             : fitted classifier.
    X_val           : validation feature DataFrame.
    y_val           : validation labels.
    total_val_rows  : total labeled validation rows (used for alarm rate scaling).
    """
    proba      = clf.predict_proba(X_val)
    buckets_per_day = 288   # 24 h × 12 buckets/h (5-min windows)
    rows: list[dict] = []

    for thresh in _THRESHOLD_SWEEP:
        pred      = (proba >= thresh).astype(int)
        y_vals    = y_val.values
        n_alarms  = int(pred.sum())
        alarm_rate = n_alarms / max(total_val_rows, 1)   # alarms per row
        alarms_per_day = round(alarm_rate * buckets_per_day, 1)
        rows.append({
            "threshold":     thresh,
            "precision":     round(float(precision_score(y_vals, pred, zero_division=0)), 4),
            "recall":        round(float(recall_score(y_vals, pred, zero_division=0)), 4),
            "f1":            round(float(f1_score(y_vals, pred, zero_division=0)), 4),
            "alarms_per_day": alarms_per_day,
        })

    return rows


def _print_threshold_table(rows: list[dict], optimal_thresh: float) -> None:
    """Print the threshold sweep table to the log."""
    log.info("  Threshold | Precision | Recall | F1    | Alarms/day")
    log.info("  ──────────┼───────────┼────────┼───────┼────────────")
    for r in rows:
        marker = "  ←  optimal" if abs(r["threshold"] - optimal_thresh) < 1e-6 else ""
        log.info(
            "    %.2f    |   %.4f  |  %.4f | %.4f | %7.1f%s",
            r["threshold"], r["precision"], r["recall"], r["f1"],
            r["alarms_per_day"], marker,
        )


# ── Feature importance ────────────────────────────────────────────────────────


def _compute_feature_importance(
    clf:         SpikeClassifier,
    X_val:       pd.DataFrame,
    output_path: Path,
) -> None:
    """Compute and save XGBoost gain + SHAP importances.

    XGBoost gain is a fast proxy.  SHAP (via TreeExplainer) is computed when
    the ``shap`` package is installed — it handles correlated lag features more
    reliably than any built-in importance type.

    ``colsample_bytree`` is intentionally left at its training value during SHAP
    computation; TreeExplainer handles feature sampling correctly via the
    Shapley value game theory guarantees.

    Parameters
    ----------
    clf         : fitted SpikeClassifier.
    X_val       : validation feature DataFrame (used for SHAP background sample).
    output_path : destination for feature_importance.csv.
    """
    booster  = clf._model.get_booster()
    gain_raw = booster.get_score(importance_type="gain")

    # Fill 0 for features that never appeared in any split (possible with
    # colsample_bytree < 1.0 and a short training run)
    gain = {col: gain_raw.get(col, 0.0) for col in _X_COLS}
    total = sum(gain.values()) or 1.0
    gain_norm = {col: v / total for col, v in gain.items()}

    rows = [
        {"feature": col, "gain_importance": round(gain_norm[col], 6), "shap_mean_abs": None}
        for col in sorted(gain_norm, key=gain_norm.get, reverse=True)
    ]

    # SHAP — optional, not a hard dependency
    try:
        import shap  # type: ignore[import]
        explainer  = shap.TreeExplainer(clf._model)
        shap_vals  = explainer.shap_values(X_val[_X_COLS].astype("float32"))
        shap_means = np.abs(shap_vals).mean(axis=0)
        shap_map   = dict(zip(_X_COLS, shap_means.tolist()))
        for r in rows:
            r["shap_mean_abs"] = round(float(shap_map[r["feature"]]), 6)
        log.info("  SHAP values computed (shap package available)")
    except ImportError:
        log.info("  shap package not installed — skipping SHAP values (gain only)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_path, rows)

    log.info("  Feature importance saved : %s", output_path)
    log.info("  Top-10 features by gain:")
    for r in rows[:10]:
        shap_str = f"  SHAP {r['shap_mean_abs']:.4f}" if r["shap_mean_abs"] is not None else ""
        log.info(
            "    %-22s  gain %.4f%s",
            r["feature"], r["gain_importance"], shap_str,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of dicts to a CSV file."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _step_needed(output_path: Path, step_num: int, from_step: int, force: bool) -> bool:
    """Return True if this step should run.

    ``--from-step N`` unconditionally skips steps before N, even if their cached
    outputs are absent.  This lets callers start from an intermediate step with
    pre-existing parquets without re-running earlier (potentially slow) steps.

    ``--force`` overrides everything and always runs.
    """
    if force:
        return True
    if step_num < from_step:
        log.info("  Skipping step %d — before --from-step %d", step_num, from_step)
        return False
    if not output_path.exists():
        return True
    log.info("  Skipping step %d — cache exists: %s", step_num, output_path)
    return False


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run(
    data_path:     Path,
    artifacts_dir: Path,
    train_ratio:   float = _TRAIN_RATIO,
    val_ratio:     float = _VAL_RATIO,
    n_folds:       int   = _N_FOLDS,
    walk_forward:  bool  = True,
    from_step:     int   = 1,
    force:         bool  = False,
    seed:          int   = 42,
) -> dict:
    """Run the full spike classifier training pipeline.

    Parameters
    ----------
    data_path     : path to cluster_cpu_data.csv (raw Google Cluster Traces).
    artifacts_dir : directory for all intermediate and output files.
    train_ratio   : fraction of buckets for the training split (default 0.6).
    val_ratio     : fraction of buckets for the validation split (default 0.2).
    n_folds       : walk-forward CV folds (default 5).
    walk_forward  : whether to run walk-forward CV before final training.
    from_step     : skip steps before this number (1=preprocess, 2=features, 3=train).
    force         : re-run all steps, ignoring existing cache files.
    seed          : random seed for reproducibility.

    Returns
    -------
    dict with all training metadata and evaluation metrics from ``spike_classifier.train()``.
    """
    np.random.seed(seed)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_dir = artifacts_dir / "models" / "spike"
    model_dir.mkdir(parents=True, exist_ok=True)

    agg_path      = artifacts_dir / "cluster_agg.parquet"
    features_path = artifacts_dir / "cluster_features.parquet"
    thresholds_path = artifacts_dir / "spike_thresholds.parquet"
    model_path    = model_dir / "spike_model.json"
    config_path   = model_dir / "spike_config.json"
    run_cfg_path  = model_dir / "run_config.json"
    cv_path       = model_dir / "cv_results.csv"
    importance_path = model_dir / "feature_importance.csv"

    t_pipeline = time.perf_counter()

    # ── Persist run config ────────────────────────────────────────────────────
    run_config = {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "data_path":    str(data_path),
        "artifacts_dir": str(artifacts_dir),
        "train_ratio":  train_ratio,
        "val_ratio":    val_ratio,
        "test_ratio":   round(1.0 - train_ratio - val_ratio, 4),
        "n_folds":      n_folds,
        "walk_forward": walk_forward,
        "from_step":    from_step,
        "force":        force,
        "seed":         seed,
    }
    run_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    run_cfg_path.write_text(json.dumps(run_config, indent=2))
    log.info("Run config saved: %s", run_cfg_path)

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    if _step_needed(agg_path, 1, from_step, force):
        log.info("═" * 62)
        log.info("  STEP 1 — Preprocessing")
        if not data_path.exists():
            raise FileNotFoundError(
                f"Raw data file not found: {data_path}\n"
                "Download: curl -O https://storage.googleapis.com/clusterdata-2011-2/"
                "task_usage/part-00000-of-00500.csv.gz"
            )
        t0 = time.perf_counter()
        preprocess(input_path=data_path, output_path=agg_path)
        log.info("  Step 1 done in %.1f min", (time.perf_counter() - t0) / 60)

    # ── Step 2: Feature engineering ───────────────────────────────────────────
    if _step_needed(features_path, 2, from_step, force):
        log.info("═" * 62)
        log.info("  STEP 2 — Feature engineering")
        t0 = time.perf_counter()
        engineer(
            input_path      = agg_path,
            output_path     = features_path,
            thresholds_path = thresholds_path,
            train_ratio     = train_ratio,
        )
        log.info("  Step 2 done in %.1f min", (time.perf_counter() - t0) / 60)

    # ── Step 3: Walk-forward CV + final training ───────────────────────────────
    log.info("═" * 62)
    log.info("  STEP 3 — Training")

    df = pd.read_parquet(features_path)
    df = df[df["spike_in_60m"].notna()].copy()
    df["spike_in_60m"] = df["spike_in_60m"].astype("int8")

    bucket_max = int(df["bucket"].max())
    train_max  = int(bucket_max * train_ratio)
    val_max    = int(bucket_max * (train_ratio + val_ratio))

    train_df = df[df["bucket"] <= train_max].reset_index(drop=True)
    val_df   = df[(df["bucket"] > train_max) & (df["bucket"] <= val_max)].reset_index(drop=True)

    # ── Walk-forward CV ───────────────────────────────────────────────────────
    cv_summary: dict[str, float] = {}
    if walk_forward:
        cv_summary = _run_walk_forward_cv(train_df, n_folds, seed, cv_path)

    # ── Final model training ──────────────────────────────────────────────────
    log.info("═" * 62)
    log.info("  FINAL TRAINING  (train=%.0f%%  val=%.0f%%  test=%.0f%%)",
             train_ratio * 100, val_ratio * 100,
             (1.0 - train_ratio - val_ratio) * 100)

    result = _train_model(
        features_path = features_path,
        model_path    = model_path,
        train_ratio   = train_ratio,
        val_ratio     = val_ratio,
    )

    # ── Threshold sweep table ─────────────────────────────────────────────────
    clf     = SpikeClassifier.load(model_path)
    X_val   = val_df[_X_COLS]
    y_val   = val_df["spike_in_60m"]

    sweep_rows = _threshold_sweep_table(clf, X_val, y_val, len(val_df))
    log.info("  THRESHOLD SWEEP  (on validation set)")
    _print_threshold_table(sweep_rows, result["optimal_threshold"])

    # ── Feature importance ────────────────────────────────────────────────────
    _compute_feature_importance(clf, X_val, importance_path)

    # ── Save spike_config.json ────────────────────────────────────────────────
    spike_config = {
        "optimal_threshold": result["optimal_threshold"],
        "train_ratio":       train_ratio,
        "val_ratio":         val_ratio,
        "test_ratio":        round(1.0 - train_ratio - val_ratio, 4),
        "train_bucket_max":  result["train_bucket_max"],
        "val_bucket_max":    result["val_bucket_max"],
        "spike_rate_train":  round(result["spike_rate_train"], 6),
        "spike_rate_val":    round(result["spike_rate_val"], 6),
        "spike_rate_test":   round(result["spike_rate_test"], 6),
        "scale_pos_weight":  round(result["scale_pos_weight"], 4),
        "final_metrics": {
            "pr_auc":               round(result["pr_auc"], 6),
            "roc_auc":              round(result["roc_auc"], 6),
            "precision_at_0.5":     round(result["precision"], 6),
            "recall_at_0.5":        round(result["recall"], 6),
            "f1_at_0.5":            round(result["f1"], 6),
            "precision_calibrated": round(result["precision_calibrated"], 6),
            "recall_calibrated":    round(result["recall_calibrated"], 6),
            "f1_calibrated":        round(result["f1_calibrated"], 6),
        },
        "cv_summary":            cv_summary,
        "threshold_sweep":       sweep_rows,
        "feature_importance_path": str(importance_path),
        "model_path":            str(model_path),
        "trained_at":            run_config["timestamp"],
    }
    config_path.write_text(json.dumps(spike_config, indent=2))
    log.info("  Spike config saved : %s", config_path)

    elapsed = (time.perf_counter() - t_pipeline) / 60
    log.info("═" * 62)
    log.info("  PIPELINE COMPLETE  (total %.1f min)", elapsed)
    log.info("  PR-AUC    : %.3f", result["pr_auc"])
    log.info("  Optimal threshold : %.2f  →  F1 %.3f  P %.3f  R %.3f",
             result["optimal_threshold"],
             result["f1_calibrated"],
             result["precision_calibrated"],
             result["recall_calibrated"])
    log.info("  Model     : %s", model_path)
    log.info("  Config    : %s", config_path)
    log.info("═" * 62)

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description     = "Train the CPU spike classifier (full pipeline).",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = (
            "examples:\n"
            "  python train_spike_classifier.py --data-path data/cluster_cpu_data.csv\n"
            "  python train_spike_classifier.py --data-path data/cluster_cpu_data.csv"
            " --force\n"
            "  python train_spike_classifier.py --data-path data/cluster_cpu_data.csv"
            " --from-step 3\n"
            "  python train_spike_classifier.py --data-path data/cluster_cpu_data.csv"
            " --no-walk-forward\n"
        ),
    )
    p.add_argument(
        "--data-path",
        required = True,
        type     = Path,
        metavar  = "PATH",
        help     = "Path to cluster_cpu_data.csv (raw Google Cluster Traces input).",
    )
    p.add_argument(
        "--artifacts-dir",
        type    = Path,
        default = Path("data"),
        metavar = "DIR",
        help    = "Directory for intermediate and output files  (default: data/).",
    )
    p.add_argument(
        "--train-ratio",
        type    = float,
        default = _TRAIN_RATIO,
        metavar = "R",
        help    = (
            f"Fraction of bucket range for training  (default: {_TRAIN_RATIO}). "
            "Must be consistent across both feature engineering and training steps."
        ),
    )
    p.add_argument(
        "--val-ratio",
        type    = float,
        default = _VAL_RATIO,
        metavar = "R",
        help    = (
            f"Fraction of bucket range for validation / threshold selection "
            f"(default: {_VAL_RATIO}).  Remaining buckets become the test set."
        ),
    )
    p.add_argument(
        "--n-folds",
        type    = int,
        default = _N_FOLDS,
        metavar = "N",
        help    = f"Walk-forward CV folds  (default: {_N_FOLDS}).",
    )
    p.add_argument(
        "--walk-forward",
        dest    = "walk_forward",
        action  = "store_true",
        default = True,
        help    = "Run walk-forward cross-validation before final training (default: on).",
    )
    p.add_argument(
        "--no-walk-forward",
        dest   = "walk_forward",
        action = "store_false",
        help   = "Skip walk-forward CV (faster; use for quick experiments).",
    )
    p.add_argument(
        "--from-step",
        type    = int,
        default = 1,
        choices = [1, 2, 3],
        metavar = "N",
        help    = "Re-run from this step onward (1=preprocess, 2=features, 3=train).",
    )
    p.add_argument(
        "--force",
        action = "store_true",
        help   = "Re-run all steps, ignoring existing cache files.",
    )
    p.add_argument(
        "--seed",
        type    = int,
        default = 42,
        metavar = "N",
        help    = "Random seed for reproducibility  (default: 42).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        data_path     = args.data_path,
        artifacts_dir = args.artifacts_dir,
        train_ratio   = args.train_ratio,
        val_ratio     = args.val_ratio,
        n_folds       = args.n_folds,
        walk_forward  = args.walk_forward,
        from_step     = args.from_step,
        force         = args.force,
        seed          = args.seed,
    )
