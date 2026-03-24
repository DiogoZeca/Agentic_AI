"""CLI orchestrator for the CPU spike prediction training pipeline.

Wires together the three pipeline steps:

  Step 1  spike_preprocessor.py      CSV  →  cluster_agg.parquet
  Step 2  spike_feature_engineer.py  agg  →  cluster_features.parquet
  Step 3  spike_classifier.py        feat →  spike_model.json + artefacts

Each step caches its output.  If the output already exists the step is skipped
(unless --force or --from-step N overrides this).

Walk-forward cross-validation (--walk-forward, default on) uses
sklearn.model_selection.TimeSeriesSplit with 5 expanding folds on the training
portion.  A gap of 12 buckets (= 60 min, matching the prediction horizon) is
inserted between each fold's train and validation windows to prevent temporal
leakage at fold boundaries.  Metrics are reported as mean ± std across folds
before the final single-model training run.  For binary horizons,
scale_pos_weight is computed per fold from that fold's training data.  For the
3-class severity model, compute_sample_weight('balanced') is used instead.

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
import gc
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import kendalltau
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from spike_preprocessor import preprocess
from spike_feature_engineer import engineer
from spike_classifier import (
    BinarySpikeClassifier,
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

_N_FOLDS:              int   = 5
_THRESHOLD_SWEEP:      list[float] = [round(t, 2) for t in np.arange(0.10, 1.0, 0.05).tolist()]
# Maximum rows used for Optuna inner-split evaluation (binary models only).
# Binary:logistic hyperparameter sensitivity plateaus ~1-2M rows; capping avoids
# OOM when the binary train split exceeds 7M rows on large datasets.
_MAX_OPTUNA_ROWS_BINARY: int = 2_000_000
# Maximum Optuna trials for binary horizon models.  Binary:logistic converges
# faster than multi:softprob — 30 trials from warm-start is sufficient.
_MAX_BINARY_TRIALS:      int = 30

# Multi-horizon training configuration (Phase 4).
# Each entry defines one model to train.  The binary=False entry (60m) uses
# SpikeClassifier (3-class); binary=True entries use BinarySpikeClassifier.
# cv_gap must equal n_windows to prevent temporal label leakage at CV fold
# boundaries (the validation window starts after a full horizon gap).
_HORIZON_CONFIGS: list[dict] = [
    {"horizon": "15m", "label_col": "spike_in_15m",    "n_windows": 3,  "cv_gap": 3,  "binary": True},
    {"horizon": "60m", "label_col": "severity_in_60m", "n_windows": 12, "cv_gap": 12, "binary": False},
]

# ── Walk-forward cross-validation ─────────────────────────────────────────────


def _fold_metrics(
    clf:             SpikeClassifier | BinarySpikeClassifier,
    X:               pd.DataFrame,
    y:               pd.Series,
    alarm_threshold: float,
    binary:          bool = False,
) -> dict[str, float]:
    """Evaluate classifier on one fold's val set.

    Returns a unified metric dict regardless of whether the classifier is
    binary or multiclass.  Binary metrics are mapped to the same key names
    so the same CV aggregation code works for both model types.

    Keys: macro_pr_auc, macro_roc_auc, weighted_f1, macro_f1.
    """
    probas = clf.predict_proba(X)
    y_vals = y.values

    if binary:
        # Binary path — shape (n, 2); column 1 = P(spike)
        p_spike   = probas[:, 1]
        pred      = (p_spike >= alarm_threshold).astype(int)
        n_present = len(np.unique(y_vals))

        if n_present < 2:
            macro_pr_auc  = float("nan")
            macro_roc_auc = float("nan")
        else:
            macro_pr_auc  = float(average_precision_score(y_vals, p_spike))
            macro_roc_auc = float(roc_auc_score(y_vals, p_spike))

        f1 = float(f1_score(y_vals, pred, zero_division=0))
        return {
            "macro_pr_auc":  macro_pr_auc,
            "macro_roc_auc": macro_roc_auc,
            "weighted_f1":   f1,
            "macro_f1":      f1,
        }

    # Multiclass path — shape (n, 3)
    pred      = np.argmax(probas, axis=1)
    n_present = len(np.unique(y_vals))

    if n_present < 2:
        macro_pr_auc  = float("nan")
        macro_roc_auc = float("nan")
    else:
        macro_pr_auc  = float(average_precision_score(y_vals, probas, average="macro"))
        macro_roc_auc = float(roc_auc_score(y_vals, probas, multi_class="ovr", average="macro"))

    return {
        "macro_pr_auc":  macro_pr_auc,
        "macro_roc_auc": macro_roc_auc,
        "weighted_f1":   float(f1_score(y_vals, pred, average="weighted", zero_division=0)),
        "macro_f1":      float(f1_score(y_vals, pred, average="macro", zero_division=0)),
    }


_CV_GAP: int = 12   # default 12 × 5-min buckets = 60 min prediction horizon


def _run_walk_forward_cv(
    train_df:    pd.DataFrame,
    n_folds:     int,
    seed:        int,
    output_path: Path,
    device:      str        = "cpu",
    model_kwargs: dict | None = None,
    label_col:   str        = "severity_in_60m",
    cv_gap:      int        = _CV_GAP,
    binary:      bool       = False,
) -> dict[str, float]:
    """Run walk-forward CV on the training split.

    Uses sklearn.model_selection.TimeSeriesSplit (expanding window) so that
    each fold trains on older data and validates on the immediately following
    window — matching the deployment scenario.

    A gap of _CV_GAP buckets (= prediction horizon) is inserted between each
    fold's train and validation windows so that label look-ahead from the last
    training rows cannot bleed into the validation window.

    scale_pos_weight is computed per fold from that fold's training rows only,
    preventing class-ratio leakage from future folds.

    find_threshold() is called on each fold's validation data.

    After all folds, a Kendall τ drift test is applied to the fold-level PR-AUC
    sequence and the result is printed to help diagnose concept drift.

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
    log.info(
        "  WALK-FORWARD CV  (%d folds, expanding window, gap=%d buckets, label=%s)",
        n_folds, cv_gap, label_col,
    )
    log.info("─" * 62)

    tscv    = TimeSeriesSplit(n_splits=n_folds, gap=cv_gap)
    X_all   = train_df[_X_COLS]
    y_all   = train_df[label_col]
    indices = np.arange(len(train_df))

    fold_rows: list[dict] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(indices), start=1):
        X_tr, y_tr   = X_all.iloc[tr_idx], y_all.iloc[tr_idx]
        X_val, y_val = X_all.iloc[val_idx], y_all.iloc[val_idx]

        n_spikes_tr = int((y_tr > 0).sum())
        if n_spikes_tr == 0:
            log.warning("  Fold %d: no spike examples in training portion — skipping", fold_idx)
            continue

        train_spike_rate = float((y_tr > 0).mean())
        val_spike_rate   = float((y_val > 0).mean())

        if binary:
            # Binary model: scale_pos_weight from fold training data only
            n_neg = int((y_tr == 0).sum())
            n_pos = max(int((y_tr == 1).sum()), 1)
            spw   = float(n_neg / n_pos)
            clf   = BinarySpikeClassifier(
                device           = device,
                random_state     = seed,
                scale_pos_weight = spw,
                **(model_kwargs or {}),
            )
            clf.fit(X_tr, y_tr)
            fold_thresh = clf.find_alarm_threshold(X_val, y_val)
        else:
            # 3-class model: XGBoost validates classes are contiguous [0..k-1]
            # Skip folds where any class is absent (common with rare severe ~1%)
            fold_classes = sorted(np.unique(y_tr).tolist())
            if fold_classes != [0, 1, 2]:
                log.warning(
                    "  Fold %d: not all 3 classes in training %s — skipping",
                    fold_idx, fold_classes,
                )
                continue

            sw_tr = compute_sample_weight("balanced", y_tr)
            clf   = SpikeClassifier(
                device       = device,
                random_state = seed,
                **(model_kwargs or {}),
            )
            clf.fit(X_tr, y_tr, sample_weight=sw_tr)

            # Alarm threshold on this fold's validation data
            if int((y_val == 2).sum()) == 0:
                fold_thresh = 0.5
                log.warning("  Fold %d: no severe examples in validation — using threshold 0.5", fold_idx)
            else:
                fold_thresh = clf.find_alarm_threshold(X_val, y_val)

        m = _fold_metrics(clf, X_val, y_val, fold_thresh, binary=binary)

        log.info(
            "  Fold %d/%d  |  MacroPR %.3f  MacroF1 %.3f  thresh %.2f  "
            "spike(tr=%.1f%%  val=%.1f%%)",
            fold_idx, n_folds,
            m["macro_pr_auc"], m["macro_f1"], fold_thresh,
            train_spike_rate * 100, val_spike_rate * 100,
        )

        fold_rows.append({
            "fold":             fold_idx,
            "alarm_threshold":  fold_thresh,
            "train_n":          len(tr_idx),
            "val_n":            len(val_idx),
            "train_spike_rate": round(train_spike_rate, 4),
            "val_spike_rate":   round(val_spike_rate, 4),
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
    metric_keys = ["macro_pr_auc", "macro_roc_auc", "weighted_f1", "macro_f1"]
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
        log.info("  %-16s  mean %.3f  ±  std %.3f", key, mean, std)

    # Drift diagnosis — Kendall τ on fold macro PR-AUC sequence
    pr_vals = [r["macro_pr_auc"] for r in fold_rows if not np.isnan(r["macro_pr_auc"])]
    if len(pr_vals) >= 3:
        tau, p_val = kendalltau(range(len(pr_vals)), pr_vals)
        if tau < -0.3:
            direction = "DECLINING (concept drift likely)"
        elif tau > 0.3:
            direction = "IMPROVING"
        else:
            direction = "STABLE"
        log.info("─" * 62)
        log.info(
            "  Drift signal  : Macro-PR is %-34s  τ=%.2f  p=%.3f",
            direction, tau, p_val,
        )
        log.info(
            "  Per-fold Macro-PR : %s",
            "  ".join(f"{v:.3f}" for v in pr_vals),
        )
        summary["cv_drift_tau"]   = round(float(tau), 4)
        summary["cv_drift_p_val"] = round(float(p_val), 4)

    log.info("─" * 62)

    return summary


# ── Threshold sweep table ──────────────────────────────────────────────────────


def _threshold_sweep_table(
    clf:    SpikeClassifier | BinarySpikeClassifier,
    X_val:  pd.DataFrame,
    y_val:  pd.Series,
    total_val_rows: int,
    binary: bool = False,
) -> list[dict]:
    """Compute precision / recall / F1 / alarms_per_day for each threshold.

    ``alarms_per_day`` estimates how many times the model would fire per day
    on a node given the validation set density (val_rows over its time span).
    Buckets are 5 min each → 288 buckets/day.

    Parameters
    ----------
    clf             : fitted classifier (SpikeClassifier or BinarySpikeClassifier).
    X_val           : validation feature DataFrame.
    y_val           : validation labels.
    total_val_rows  : total labeled validation rows (used for alarm rate scaling).
    binary          : if True, use P(spike) column 1; otherwise use P(severe) column 2.
    """
    probas          = clf.predict_proba(X_val)
    # Binary: column 1 = P(spike); 3-class: column 2 = P(severe)
    p_alarm         = probas[:, 1] if binary else probas[:, 2]
    y_binary        = (y_val.values > 0).astype(int) if binary else (y_val.values == 2).astype(int)
    buckets_per_day = 288   # 24 h × 12 buckets/h (5-min windows)
    rows: list[dict] = []

    for thresh in _THRESHOLD_SWEEP:
        pred           = (p_alarm >= thresh).astype(int)
        n_alarms       = int(pred.sum())
        alarm_rate     = n_alarms / max(total_val_rows, 1)
        alarms_per_day = round(alarm_rate * buckets_per_day, 1)
        rows.append({
            "threshold":      thresh,
            "precision":      round(float(precision_score(y_binary, pred, zero_division=0)), 4),
            "recall":         round(float(recall_score(y_binary, pred, zero_division=0)), 4),
            "f1":             round(float(f1_score(y_binary, pred, zero_division=0)), 4),
            "alarms_per_day": alarms_per_day,
        })

    return rows


def _print_threshold_table(rows: list[dict], optimal_thresh: float) -> None:
    """Print the p_severe alarm threshold sweep table to the log."""
    log.info("  p_severe  | Precision | Recall | F1    | Alarms/day")
    log.info("  ──────────┼───────────┼────────┼───────┼────────────")
    for r in rows:
        marker = "  ←  alarm" if abs(r["threshold"] - optimal_thresh) < 1e-6 else ""
        log.info(
            "    %.2f    |   %.4f  |  %.4f | %.4f | %7.1f%s",
            r["threshold"], r["precision"], r["recall"], r["f1"],
            r["alarms_per_day"], marker,
        )


# ── Feature importance ────────────────────────────────────────────────────────


def _compute_feature_importance(
    clf:         SpikeClassifier | BinarySpikeClassifier,
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

    # XGBoost stores features as "f0", "f1", ... when the model is trained on
    # a numpy array (which strips column names).  Map them back to actual names
    # so gain_raw keys match _X_COLS.
    fmap = {f"f{i}": name for i, name in enumerate(clf._feature_cols)}
    gain_raw = {fmap.get(k, k): v for k, v in gain_raw.items()}

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
        explainer = shap.TreeExplainer(clf._model)
        shap_vals = explainer.shap_values(X_val[_X_COLS].astype("float32"))
        # shap_values() return shape depends on objective:
        #   binary:logistic  → 2-D array (n_samples, n_features)
        #   multi:softprob   → list of 2-D arrays or 3-D array (n_samples, n_features, n_classes)
        if isinstance(shap_vals, list):
            # List of (n_samples, n_features) arrays — mean absolute SHAP across classes
            shap_means = np.mean([np.abs(sv) for sv in shap_vals], axis=0).mean(axis=0)
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            shap_means = np.abs(shap_vals).mean(axis=(0, 2))
        else:
            shap_means = np.abs(shap_vals).mean(axis=0)
        shap_map = dict(zip(_X_COLS, shap_means.tolist()))
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


# ── Hyperparameter search (Optuna) ────────────────────────────────────────────


class _OptunaProgressCallback:
    """Log Optuna progress every ``log_interval`` completed trials.

    Without this, a 50–100 trial run produces no output for 1–2 hours,
    making it impossible to distinguish a running search from a crashed one.

    Logs to the module logger (same stream as all other pipeline output) so
    progress is visible in both the terminal and any nohup redirect file.
    """

    def __init__(self, n_trials: int, log_interval: int = 5) -> None:
        self._n_trials     = n_trials
        self._log_interval = log_interval
        self._count        = 0

    def __call__(self, study, trial) -> None:
        self._count += 1
        if self._count % self._log_interval == 0 or self._count == self._n_trials:
            trial_val = trial.value if trial.value is not None else float("nan")
            log.info(
                "  Optuna  %3d/%d  |  best PR-AUC %.4f  |  trial %.4f",
                self._count, self._n_trials,
                study.best_value,
                trial_val,
            )


def _run_optuna_search(
    train_df:         pd.DataFrame,
    seed:             int,
    n_trials:         int,
    device:           str        = "cpu",
    output_dir:       Path | None = None,
    warmstart_params: dict | None = None,
    label_col:        str        = "severity_in_60m",
    study_name:       str        = "spike-hpo-60m",
    binary:           bool       = False,
) -> dict:
    """Bayesian hyperparameter search using Optuna TPE sampler.

    Carves an inner temporal split from ``train_df`` (first 50% of bucket range
    = inner train, second 50% = inner val) to score each trial.  The outer
    val/test sets are never touched.

    ``scale_pos_weight`` is computed from inner_train only and held fixed — it
    is not a hyperparameter (its value is determined by the data distribution,
    not by model capacity).

    Each trial trains a ``SpikeClassifier`` on the inner train split and
    evaluates PR-AUC on the inner val split.  Optuna maximises PR-AUC using
    the TPE sampler with 15 random startup trials (more than the default 10,
    recommended for 9-dimensional search spaces to give TPE a denser initial
    probability model before Bayesian optimisation begins).

    Known limitation
    ----------------
    The inner 50/50 split objective is a proxy for the 5-fold walk-forward CV
    production metric.  The gap (inner ~0.617 vs outer CV ~0.640) is accepted:
    switching to 3-fold CV in the objective would be 3× slower per trial.
    Monitor the gap across phases; if it widens beyond 0.03, revisit.

    Parameters
    ----------
    train_df         : training portion of cluster_features.parquet (labeled).
    seed             : random seed for TPE sampler and SpikeClassifier.
    n_trials         : total number of Optuna trials to run (including any
                       resumed trials from a previous interrupted run).
    device           : compute device passed to SpikeClassifier.
    output_dir       : when provided, Optuna persists each trial to a SQLite
                       database at ``output_dir/optuna.db``.  If the database
                       already exists (interrupted previous run), trials are
                       resumed automatically — only the remaining ``n_trials``
                       minus already-completed trials are run.
    warmstart_params : when provided, this parameter dict is enqueued as the
                       first trial.  Useful when re-running after a feature
                       change (Phase 2+): the previous best tree-structure
                       hyperparameters are a reasonable starting point even
                       when the feature set changes.

    Returns
    -------
    dict with keys:
        params       — best hyperparameter dict (9 keys)
        best_pr_auc  — PR-AUC of the best trial on the inner val split
        n_trials     — requested number of trials
        n_completed  — number of trials that completed without error
        resumed_from — number of trials already in storage before this run
    """
    import optuna  # lazy import — only needed when --tune-hyperparams is active
    from sklearn.metrics import average_precision_score as _ap_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── Persistent storage (SQLite) ───────────────────────────────────────────
    # Persisting to SQLite means an interrupted run can be resumed: the study
    # loads existing trials and only runs the remaining budget.  Without this,
    # a crashed 2-hour search loses all trial history.
    storage    = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        db_path  = output_dir / "optuna.db"
        storage  = f"sqlite:///{db_path}"
        log.info("  Optuna storage  : %s", db_path)

    # ── Inner temporal split ──────────────────────────────────────────────────
    # First 50% of bucket range = inner train, second 50% = inner val.
    # Mirrors the outer walk-forward logic without touching val/test.
    split_max   = int(train_df["bucket"].max() * 0.5)
    inner_train = train_df[train_df["bucket"] <= split_max].reset_index(drop=True)
    inner_val   = train_df[train_df["bucket"] >  split_max].reset_index(drop=True)

    # For binary models, cap the inner split to avoid OOM during Optuna.
    # Hyperparameter sensitivity plateaus around 2M rows; extra rows cost memory
    # and time without improving the TPE search surface.
    if binary and len(inner_train) > _MAX_OPTUNA_ROWS_BINARY:
        log.info(
            "  Optuna inner split capped : %d → %d rows (binary OOM prevention)",
            len(inner_train), _MAX_OPTUNA_ROWS_BINARY,
        )
        inner_train = inner_train.sample(
            n           = _MAX_OPTUNA_ROWS_BINARY,
            random_state = seed,
            stratify    = inner_train[label_col].astype("int8"),
        ).sort_values("bucket").reset_index(drop=True)
        inner_val = inner_val.sample(
            n           = min(_MAX_OPTUNA_ROWS_BINARY, len(inner_val)),
            random_state = seed,
            stratify    = inner_val[label_col].astype("int8"),
        ).sort_values("bucket").reset_index(drop=True)

    X_inner_tr  = inner_train[_X_COLS]
    y_inner_tr  = inner_train[label_col].astype("int8")
    X_inner_val = inner_val[_X_COLS]
    y_inner_val = inner_val[label_col].astype("int8")

    if binary:
        # Binary: scale_pos_weight from inner train
        n_neg_inner  = int((y_inner_tr == 0).sum())
        n_pos_inner  = max(int((y_inner_tr == 1).sum()), 1)
        spw_inner    = float(n_neg_inner / n_pos_inner)
    else:
        # 3-class: sample_weight from inner train
        sw_inner = compute_sample_weight("balanced", y_inner_tr)

    # ── Create or resume study ────────────────────────────────────────────────
    study = optuna.create_study(
        study_name     = study_name,
        storage        = storage,
        load_if_exists = True,   # resume if interrupted (no-op when storage=None)
        direction      = "maximize",
        sampler        = optuna.samplers.TPESampler(
            seed             = seed,
            n_startup_trials = 15,  # 15 random before Bayesian (was 10); better
                                    # initial density for 9-dimensional space
        ),
    )

    # How many trials have already been consumed (complete, failed, or pruned).
    # Excludes WAITING (enqueued but not started) so the budget isn't under-counted.
    already_done = len([t for t in study.trials
                        if t.state != optuna.trial.TrialState.WAITING])
    remaining    = max(0, n_trials - already_done)

    if already_done:
        log.info(
            "  Optuna resumed  : %d trials already complete, running %d more",
            already_done, remaining,
        )
    if remaining == 0:
        log.info("  Optuna skipped  : all %d trials already complete", n_trials)
        best = study.best_trial
        return {
            "params":            dict(best.params),
            "best_macro_pr_auc": float(best.value),
            "n_trials":          n_trials,
            "n_completed":       len(study.trials),
            "resumed_from":      already_done,
        }

    # ── Warm-start: enqueue previous best params as the first trial ───────────
    # Tree-structure hyperparameters (max_depth, regularisation, etc.) are largely
    # feature-independent, so the previous best is a good starting point even when
    # the feature set changes between phases.  TPE will score it on the new features
    # and use it to initialise its probability model.
    if warmstart_params and len(study.trials) == 0:
        study.enqueue_trial(warmstart_params)
        log.info("  Optuna warm-start : enqueueing 1 trial from previous best_params")

    log.info(
        "  Optuna search   : %d trials  |  inner split  train=%d  val=%d rows",
        remaining, len(inner_train), len(inner_val),
    )

    def objective(trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        }
        if len(np.unique(y_inner_val)) < 2:
            return 0.0
        if binary:
            clf = BinarySpikeClassifier(
                device           = device,
                random_state     = seed,
                scale_pos_weight = spw_inner,
                **params,
            )
            clf.fit(X_inner_tr, y_inner_tr)
            probas = clf.predict_proba(X_inner_val)[:, 1]  # P(spike)
            score  = float(_ap_score(y_inner_val.values, probas))
            del clf
            gc.collect()
            return score
        else:
            clf = SpikeClassifier(
                device       = device,
                random_state = seed,
                **params,
            )
            clf.fit(X_inner_tr, y_inner_tr, sample_weight=sw_inner)
            probas = clf.predict_proba(X_inner_val)  # (n, 3)
            score  = float(_ap_score(y_inner_val.values, probas, average="macro"))
            del clf
            gc.collect()
            return score

    study.optimize(
        objective,
        n_trials          = remaining,
        show_progress_bar = False,
        callbacks         = [_OptunaProgressCallback(n_trials=remaining)],
    )

    best = study.best_trial
    log.info(
        "  Optuna done     : best macro-PR-AUC %.4f  (trial %d / %d completed)",
        best.value, best.number + 1, len(study.trials),
    )

    return {
        "params":              dict(best.params),
        "best_macro_pr_auc":   float(best.value),
        "n_trials":            n_trials,
        "n_completed":         len(study.trials),
        "resumed_from":        already_done,
    }


# ── Binary horizon training helper ────────────────────────────────────────────


def _train_binary_horizon(
    df:               pd.DataFrame,
    label_col:        str,
    horizon_name:     str,
    cv_gap:           int,
    model_dir:        Path,
    features_path:    Path,
    train_ratio:      float,
    val_ratio:        float,
    n_folds:          int,
    walk_forward:     bool,
    seed:             int,
    device:           str,
    tune_hyperparams: bool,
    n_trials:         int,
    warmstart_params: dict | None = None,
) -> dict:
    """Train and evaluate a BinarySpikeClassifier for one short horizon.

    Full pipeline: optional CV → optional Optuna → final training → threshold
    sweep → feature importance → save config.  Mirrors the 60m pipeline in
    ``run()`` but uses BinarySpikeClassifier and binary metrics throughout.

    Returns
    -------
    dict with binary evaluation metrics from the test set.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    log.info("  [GC] Memory freed before %s binary model training", horizon_name)

    cv_path         = model_dir / "cv_results.csv"
    importance_path = model_dir / "feature_importance.csv"
    model_path      = model_dir / "spike_model.json"
    config_path     = model_dir / "spike_config.json"
    best_params_path = model_dir / "best_params.json"

    # Filter to rows with a valid label for this horizon
    df_h = df[df[label_col].notna()].copy()
    df_h[label_col] = df_h[label_col].astype("int8")

    bucket_max = int(df_h["bucket"].max())
    train_max  = int(bucket_max * train_ratio)
    val_max    = int(bucket_max * (train_ratio + val_ratio))

    train_df = df_h[df_h["bucket"] <= train_max].reset_index(drop=True)
    val_df   = df_h[(df_h["bucket"] > train_max) & (df_h["bucket"] <= val_max)].reset_index(drop=True)
    test_df  = df_h[df_h["bucket"] > val_max].reset_index(drop=True)

    n_neg = int((train_df[label_col] == 0).sum())
    n_pos = max(int((train_df[label_col] == 1).sum()), 1)
    spw   = float(n_neg / n_pos)

    log.info("═" * 62)
    log.info("  BINARY SPIKE CLASSIFIER — %s horizon  (%s)", horizon_name, label_col)
    log.info("  Train rows : %s  (spike %.1f%%)", f"{len(train_df):,}", 100 / (1 + spw))
    log.info("  Val rows   : %s", f"{len(val_df):,}")
    log.info("  Test rows  : %s", f"{len(test_df):,}")
    log.info("  spw        : %.1f  (n_neg/n_pos = %d/%d)", spw, n_neg, n_pos)

    # Hyperparameter resolution (same priority as 60m model)
    search_result: dict | None = None

    if tune_hyperparams:
        ws: dict | None = None
        if best_params_path.exists():
            try:
                ws = json.loads(best_params_path.read_text())["params"]
            except (KeyError, json.JSONDecodeError):
                ws = None
        if ws is None and warmstart_params is not None:
            ws = warmstart_params

        binary_n_trials = min(n_trials, _MAX_BINARY_TRIALS)
        if binary_n_trials < n_trials:
            log.info(
                "  Binary Optuna capped at %d trials (binary converges faster than multiclass)",
                binary_n_trials,
            )
        search_result = _run_optuna_search(
            train_df,
            seed,
            binary_n_trials,
            device           = device,
            output_dir       = model_dir,
            warmstart_params = ws,
            label_col        = label_col,
            study_name       = f"spike-hpo-{horizon_name}",
            binary           = True,
        )
        best_params_path.write_text(json.dumps(search_result, indent=2))
        model_kwargs: dict = search_result["params"]
    elif best_params_path.exists():
        cached       = json.loads(best_params_path.read_text())
        model_kwargs = cached["params"]
    else:
        model_kwargs = {}

    # Walk-forward CV
    cv_summary: dict[str, float] = {}
    if walk_forward:
        cv_summary = _run_walk_forward_cv(
            train_df, n_folds, seed, cv_path,
            device       = device,
            model_kwargs = model_kwargs,
            label_col    = label_col,
            cv_gap       = cv_gap,
            binary       = True,
        )

    # Final model training
    clf = BinarySpikeClassifier(
        device           = device,
        random_state     = seed,
        scale_pos_weight = spw,
        **(model_kwargs or {}),
    )
    clf.fit(train_df[_X_COLS], train_df[label_col])

    alarm_thresh = clf.find_alarm_threshold(val_df[_X_COLS], val_df[label_col])
    m = clf.evaluate(test_df[_X_COLS], test_df[label_col], alarm_threshold=alarm_thresh)

    log.info("  PR-AUC         : %.3f  ← primary metric", m["pr_auc"])
    log.info("  ROC-AUC        : %.3f", m["roc_auc"])
    log.info("  Alarm threshold: %.2f  →  P=%.3f  R=%.3f",
             alarm_thresh, m["precision"], m["recall"])

    sweep_rows = _threshold_sweep_table(clf, val_df[_X_COLS], val_df[label_col], len(val_df), binary=True)
    _print_threshold_table(sweep_rows, alarm_thresh)
    _compute_feature_importance(clf, val_df[_X_COLS], importance_path)

    clf.save(model_path)

    spike_config = {
        "horizon":            horizon_name,
        "label_col":          label_col,
        "alarm_threshold":    alarm_thresh,
        "train_ratio":        train_ratio,
        "val_ratio":          val_ratio,
        "test_ratio":         round(1.0 - train_ratio - val_ratio, 4),
        "train_bucket_max":   train_max,
        "val_bucket_max":     val_max,
        "spike_rate_train":   round(float(train_df[label_col].mean()), 6),
        "spike_rate_val":     round(float(val_df[label_col].mean()), 6),
        "spike_rate_test":    round(float(test_df[label_col].mean()), 6),
        "final_metrics": {k: round(v, 6) for k, v in m.items()},
        "cv_summary":               cv_summary,
        "alarm_threshold_sweep":    sweep_rows,
        "hyperparameter_search": {
            "enabled":           tune_hyperparams,
            "best_params":       model_kwargs if model_kwargs else None,
            "best_macro_pr_auc": search_result["best_macro_pr_auc"] if search_result else None,
            "n_completed":       search_result["n_completed"] if search_result else None,
        },
        "inference": {
            "feature_cols": list(_X_COLS),
        },
    }
    config_path.write_text(json.dumps(spike_config, indent=2))
    log.info("  Config saved : %s", config_path)

    del clf
    gc.collect()

    return {
        "horizon":         horizon_name,
        "train_rows":      len(train_df),
        "val_rows":        len(val_df),
        "test_rows":       len(test_df),
        "alarm_threshold": alarm_thresh,
        **{k: v for k, v in m.items() if k != "alarm_threshold"},
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run(
    data_path:        Path,
    artifacts_dir:    Path,
    train_ratio:      float        = _TRAIN_RATIO,
    val_ratio:        float        = _VAL_RATIO,
    n_folds:          int          = _N_FOLDS,
    walk_forward:     bool         = True,
    from_step:        int          = 1,
    force:            bool         = False,
    seed:             int          = 42,
    device:           str          = "cpu",
    target_pos_rate:  float | None = None,
    tune_hyperparams: bool         = False,
    n_trials:         int          = 30,
) -> dict:
    """Run the full spike classifier training pipeline.

    Parameters
    ----------
    data_path       : path to cluster_cpu_data.csv (raw Google Cluster Traces).
    artifacts_dir   : directory for all intermediate and output files.
    train_ratio     : fraction of buckets for the training split (default 0.6).
    val_ratio       : fraction of buckets for the validation split (default 0.2).
    n_folds         : walk-forward CV folds (default 5).
    walk_forward    : whether to run walk-forward CV before final training.
    from_step       : skip steps before this number (1=preprocess, 2=features, 3=train).
    force           : re-run all steps, ignoring existing cache files.
    seed            : random seed for reproducibility.
    target_pos_rate : expected deployment spike rate for label-shift correction.
                      See ``spike_classifier.train()`` for the full derivation.
                      When ``None`` (default) no correction is applied.

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
        "seed":            seed,
        "device":          device,
        "target_pos_rate": target_pos_rate,
        "tune_hyperparams": tune_hyperparams,
        "n_trials":         n_trials,
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
    df = df[df["severity_in_60m"].notna()].copy()
    df["severity_in_60m"] = df["severity_in_60m"].astype("int8")

    bucket_max = int(df["bucket"].max())
    train_max  = int(bucket_max * train_ratio)
    val_max    = int(bucket_max * (train_ratio + val_ratio))

    train_df = df[df["bucket"] <= train_max].reset_index(drop=True)
    val_df   = df[(df["bucket"] > train_max) & (df["bucket"] <= val_max)].reset_index(drop=True)

    # ── Hyperparameter resolution ─────────────────────────────────────────────
    # Priority:
    #   1. --tune-hyperparams → run Optuna, save best_params.json
    #   2. best_params.json exists → load cached params (no search)
    #   3. Neither → use SpikeClassifier defaults (no tuning)
    best_params_path = model_dir / "best_params.json"
    search_result: dict | None = None

    if tune_hyperparams:
        # Warm-start: if a previous best_params.json exists, pass those params
        # as trial 1 so TPE starts from a known-good region of the search space.
        # This works even after feature changes (Phase 2+): tree-structure
        # hyperparameters are largely feature-independent.
        warmstart: dict | None = None
        if best_params_path.exists():
            try:
                warmstart = json.loads(best_params_path.read_text())["params"]
            except (KeyError, json.JSONDecodeError):
                warmstart = None

        search_result = _run_optuna_search(
            train_df,
            seed,
            n_trials,
            device           = device,
            output_dir       = model_dir,
            warmstart_params = warmstart,
        )
        best_params_path.write_text(json.dumps(search_result, indent=2))
        log.info("  Best params saved  : %s", best_params_path)
        model_kwargs: dict = search_result["params"]
    elif best_params_path.exists():
        cached = json.loads(best_params_path.read_text())
        model_kwargs = cached["params"]
        log.info("  Loaded cached hyperparams from %s", best_params_path)
    else:
        model_kwargs = {}
        log.info("  Using default hyperparameters (no tuning, no cache)")

    # ── Walk-forward CV ───────────────────────────────────────────────────────
    cv_summary: dict[str, float] = {}
    if walk_forward:
        cv_summary = _run_walk_forward_cv(
            train_df, n_folds, seed, cv_path, device=device, model_kwargs=model_kwargs
        )

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
        device        = device,
        model_kwargs  = model_kwargs,
    )

    # ── Alarm threshold sweep table ───────────────────────────────────────────
    clf   = SpikeClassifier.load(model_path)
    X_val = val_df[_X_COLS]
    y_val = val_df["severity_in_60m"]

    sweep_rows = _threshold_sweep_table(clf, X_val, y_val, len(val_df))
    log.info("  ALARM THRESHOLD SWEEP  (p_severe on validation set)")
    _print_threshold_table(sweep_rows, result["alarm_threshold"])

    # ── Feature importance ────────────────────────────────────────────────────
    _compute_feature_importance(clf, X_val, importance_path)

    # ── Save spike_config.json ────────────────────────────────────────────────
    thresh_df = pd.read_parquet(thresholds_path)
    _global_p95_fallback = float(thresh_df["threshold_p95"].median())
    _global_p99_fallback = float(thresh_df["threshold_p99"].median())

    spike_config = {
        "alarm_threshold":   result["alarm_threshold"],
        "train_ratio":       train_ratio,
        "val_ratio":         val_ratio,
        "test_ratio":        round(1.0 - train_ratio - val_ratio, 4),
        "train_bucket_max":  result["train_bucket_max"],
        "val_bucket_max":    result["val_bucket_max"],
        "class_rates_train": {k: round(v, 6) for k, v in result["class_rates_train"].items()},
        "class_rates_val":   {k: round(v, 6) for k, v in result["class_rates_val"].items()},
        "class_rates_test":  {k: round(v, 6) for k, v in result["class_rates_test"].items()},
        "final_metrics": {
            "macro_pr_auc":   round(result["macro_pr_auc"], 6),
            "macro_roc_auc":  round(result["macro_roc_auc"], 6),
            "pr_auc_class_0": round(result["pr_auc_class_0"], 6),
            "pr_auc_class_1": round(result["pr_auc_class_1"], 6),
            "pr_auc_class_2": round(result["pr_auc_class_2"], 6),
            "weighted_f1":    round(result["weighted_f1"], 6),
            "macro_f1":       round(result["macro_f1"], 6),
            "alarm_precision": round(result["alarm_precision"], 6),
            "alarm_recall":    round(result["alarm_recall"], 6),
        },
        "cv_summary":            cv_summary,
        "alarm_threshold_sweep": sweep_rows,
        "hyperparameter_search": {
            "enabled":             tune_hyperparams,
            "best_params":         model_kwargs if model_kwargs else None,
            "best_macro_pr_auc":   search_result["best_macro_pr_auc"] if search_result else None,
            "n_completed":         search_result["n_completed"] if search_result else None,
            "resumed_from":        search_result["resumed_from"] if search_result else None,
            "from_cache":          not tune_hyperparams and best_params_path.exists(),
        },
        # ── Inference-time operational fields ─────────────────────────────────
        "inference": {
            "bucket_duration_seconds":         300,
            "horizon_windows":                 12,
            "horizon_minutes":                 60,
            "lookback_windows":                24,
            "lookback_minutes":                120,
            "min_buckets_cold_start":          12,
            "feature_cols":                    list(_X_COLS),
            "global_threshold_p95_fallback":   round(_global_p95_fallback, 6),
            "global_threshold_p99_fallback":   round(_global_p99_fallback, 6),
            "thresholds_file":                 str(thresholds_path),
        },
        "data": {
            "training_rows":     len(train_df),
            "training_machines": int(train_df["machine_id"].nunique()),
        },
        "environment": {
            "python_version":  sys.version.split()[0],
            "xgboost_version": xgb.__version__,
        },
        "model_path":              str(model_path),
        "feature_importance_path": str(importance_path),
        "trained_at":              run_config["timestamp"],
    }
    config_path.write_text(json.dumps(spike_config, indent=2))
    log.info("  Spike config saved : %s", config_path)

    # Release 60m model and large DataFrames from memory before training binary models
    del clf
    gc.collect()

    # ── Binary horizon models (15m only — 30m/45m dropped in Fix 3) ───────────
    # Log available RAM before starting binary training — helpful for OOM diagnosis.
    try:
        import psutil
        _ram = psutil.virtual_memory()
        log.info(
            "  Memory before binary training : %.1f%% used  (%.1f GB free)",
            _ram.percent, _ram.available / 1e9,
        )
    except ImportError:
        pass

    # Full feature DataFrame needed (includes binary label columns from Phase 4).
    # The `df` in scope here only has severity_in_60m rows; reload to include
    # rows where binary labels are valid but severity_in_60m may be NaN.
    df_all = pd.read_parquet(features_path)

    # Warm-start binary models from 60m best_params (tree structure is portable)
    binary_warmstart: dict | None = None
    if best_params_path.exists():
        try:
            binary_warmstart = json.loads(best_params_path.read_text())["params"]
        except (KeyError, json.JSONDecodeError):
            binary_warmstart = None

    binary_results: dict[str, dict] = {}
    for hcfg in _HORIZON_CONFIGS:
        if not hcfg["binary"]:
            continue  # 60m already trained above
        h_name     = hcfg["horizon"]
        h_label    = hcfg["label_col"]
        h_cv_gap   = hcfg["cv_gap"]
        h_model_dir = artifacts_dir / "models" / f"spike_{h_name}"

        log.info("  Starting binary horizon model: %s (%s)", h_name, h_label)
        h_result = _train_binary_horizon(
            df               = df_all,
            label_col        = h_label,
            horizon_name     = h_name,
            cv_gap           = h_cv_gap,
            model_dir        = h_model_dir,
            features_path    = features_path,
            train_ratio      = train_ratio,
            val_ratio        = val_ratio,
            n_folds          = n_folds,
            walk_forward     = walk_forward,
            seed             = seed,
            device           = device,
            tune_hyperparams = tune_hyperparams,
            n_trials         = n_trials,
            warmstart_params = binary_warmstart,
        )
        binary_results[h_name] = h_result

    elapsed = (time.perf_counter() - t_pipeline) / 60
    log.info("═" * 62)
    log.info("  PIPELINE COMPLETE  (total %.1f min)", elapsed)
    log.info("  60m Macro PR-AUC : %.3f", result["macro_pr_auc"])
    log.info("  60m Severe PR-AUC: %.3f", result["pr_auc_class_2"])
    log.info("  60m Alarm thresh : %.2f  →  P %.3f  R %.3f",
             result["alarm_threshold"],
             result["alarm_precision"],
             result["alarm_recall"])
    for h_name, h_res in binary_results.items():
        log.info("  %s PR-AUC        : %.3f  (alarm %.2f)",
                 h_name, h_res.get("pr_auc", float("nan")), h_res.get("alarm_threshold", 0.5))
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
    p.add_argument(
        "--device",
        default = "cpu",
        choices = ["cpu", "cuda"],
        help    = (
            "Compute device for XGBoost training  (default: cpu). "
            "Use 'cuda' on a VM with a CUDA-capable GPU — falls back to CPU "
            "automatically if no GPU is detected.  The saved model is "
            "device-agnostic and loads on any machine regardless of this setting."
        ),
    )
    p.add_argument(
        "--target-pos-rate",
        type    = float,
        default = None,
        metavar = "P",
        help    = (
            "Expected spike rate in the deployment window, e.g. 0.119 for 11.9%%. "
            "When provided, applies label-shift correction: scale_pos_weight is "
            "adjusted so XGBoost's effective positive rate during training matches "
            "this target rather than the raw training rate.  Use this when the "
            "test/deployment spike prevalence is known to differ from training "
            "(e.g. the cluster stabilises over time).  Omit to use the standard "
            "neg/pos ratio with no correction  (default: None)."
        ),
    )
    p.add_argument(
        "--tune-hyperparams",
        dest    = "tune_hyperparams",
        action  = "store_true",
        default = False,
        help    = (
            "Run Optuna Bayesian hyperparameter search before final training. "
            "Saves best_params.json alongside the model artifacts. "
            "Subsequent runs without this flag automatically load the cached params."
        ),
    )
    p.add_argument(
        "--optuna-trials",
        dest    = "n_trials",
        type    = int,
        default = 30,
        metavar = "N",
        help    = (
            "Number of Optuna trials for hyperparameter search  (default: 30). "
            "Ignored unless --tune-hyperparams is set. "
            "30 trials is a practical minimum; 50–100 yields diminishing returns."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        data_path        = args.data_path,
        artifacts_dir    = args.artifacts_dir,
        train_ratio      = args.train_ratio,
        val_ratio        = args.val_ratio,
        n_folds          = args.n_folds,
        walk_forward     = args.walk_forward,
        from_step        = args.from_step,
        force            = args.force,
        seed             = args.seed,
        device           = args.device,
        target_pos_rate  = args.target_pos_rate,
        tune_hyperparams = args.tune_hyperparams,
        n_trials         = args.n_trials,
    )
