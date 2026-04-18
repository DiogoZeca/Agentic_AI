#!/usr/bin/env python3
"""
evaluate_zabbix.py — 3-phase cross-domain evaluation: Google-trained model on Zabbix data.

Phases
------
  Phase 0 — EDA
      Per-node statistics (idle_fraction, burstiness, PMR, spike_rate, …).
      Flags potential domain-incompatibility before touching the model.

  Phase 1 — Domain shift (PSI)
      Population Stability Index for all 37 model features, comparing the
      Google Cluster 2011 training distribution to the Zabbix dataset.
      Skipped with a warning if the Google features parquet is not found.

  Phase 2 — Model evaluation
      Runs spike_feature_engineer.engineer() on Zabbix data (Zabbix-local
      p95/p99 thresholds, K=1 labels for maximum label coverage), then loads
      the production model from data/full_run/ and evaluates on the test split.

Outputs (written to data/zabbix_eval/)
-------
  evaluation_results.json  — machine-readable metrics
  evaluation_report.md     — human-readable summary

Usage (from AIModel/):
  .venv/bin/python3 evaluate_zabbix.py [--agg PATH] [--artifacts-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from spike_classifier import _VAL_RATIO, _X_COLS
from spike_feature_engineer import _TRAIN_RATIO, engineer

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)

# PR-AUC of the production model on Google Cluster test set.
_SOURCE_PR_AUC = 0.547

# EDA domain-incompatibility warning thresholds (from project_zabbix_evaluation.md).
_WARN_IDLE_FRACTION    = 0.40   # > 40% idle → p95 features behave differently
_WARN_BUSINESS_RATIO   = 3.0    # > 3× daytime vs off-hours → hour features dominate
_WARN_NODE_CORRELATION = 0.60   # > 0.6 pairwise → cluster_cpu_p90 less informative

# PSI thresholds (standard industry values).
_PSI_MINOR    = 0.10
_PSI_MODERATE = 0.25
_N_PSI_BINS   = 10

_PSI_LABEL_MAJOR    = "MAJOR"
_PSI_LABEL_MODERATE = "moderate"
_PSI_LABEL_MINOR    = "minor"

_SOURCE_PR_AUC_15M  = 0.575
_SOURCE_PR_AUC_30M  = 0.563
_SOURCE_PR_AUC_45M  = 0.562
_SOURCE_PR_AUC_OVR  = 0.339

_CLASS_LABELS = {0: "no_spike", 1: "moderate", 2: "severe"}


def _acf(x: np.ndarray, lag: int) -> float:
    """Autocorrelation at a given lag. Returns nan when series is too short."""
    if len(x) <= lag:
        return float("nan")
    c0 = float(np.var(x))
    if c0 < 1e-12:
        return 0.0
    return float(np.cov(x[:-lag], x[lag:])[0, 1] / c0)


# ── Phase 0 — EDA ─────────────────────────────────────────────────────────────

def _compute_node_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Return one EDA row per node.

    Parameters
    ----------
    df : cluster_agg schema DataFrame from zabbix_agg.parquet.
    """
    records = []
    for mid, grp in df.groupby("machine_id"):
        cpu = grp["total_cpu"].values.astype("float64")
        n   = len(cpu)
        if n == 0:
            continue

        mean_cpu = float(cpu.mean())
        std_cpu  = float(cpu.std()) if n > 1 else 0.0

        # Idle: CPU effectively zero (< 5% of mean, or < 0.01 absolute)
        idle_thresh   = max(0.01, mean_cpu * 0.05)
        idle_fraction = float((cpu < idle_thresh).mean())

        # Burstiness: (std - mean) / (std + mean); 0 = Poisson-like, 1 = bursty
        burstiness = float((std_cpu - mean_cpu) / (std_cpu + mean_cpu + 1e-9))

        pmr = float(np.percentile(cpu, 99) / (mean_cpu + 1e-9))
        cv  = float(std_cpu / (mean_cpu + 1e-9))

        # bucket % 288 gives slot-in-day; 9am = slot 108, 6pm = slot 216 (UTC)
        slot       = (grp["bucket"].values % 288).astype(int)
        business   = cpu[(slot >= 108) & (slot < 216)]
        off_hours  = cpu[(slot < 108) | (slot >= 216)]
        biz_ratio  = float(
            business.mean() / (off_hours.mean() + 1e-9)
            if len(business) > 0 and len(off_hours) > 0
            else 1.0
        )

        acf1  = _acf(cpu, 1)
        acf12 = _acf(cpu, 12)

        p95         = float(np.percentile(cpu, 95))
        spike_rate  = float((cpu > p95).mean())

        above       = (cpu > p95).astype(int)
        runs        = np.diff(np.concatenate([[0], above, [0]]))
        starts      = np.where(runs == 1)[0]
        ends        = np.where(runs == -1)[0]
        lengths     = ends - starts
        mean_ep_len = float(lengths.mean()) if len(lengths) > 0 else 0.0

        records.append({
            "machine_id"   : int(mid),
            "n_buckets"    : n,
            "mean_cpu"     : round(mean_cpu, 4),
            "std_cpu"      : round(std_cpu, 4),
            "idle_fraction": round(idle_fraction, 4),
            "burstiness_B" : round(burstiness, 4),
            "pmr"          : round(pmr, 2),
            "cv"           : round(cv, 4),
            "biz_ratio"    : round(biz_ratio, 2),
            "acf_lag1"     : round(acf1, 4),
            "acf_lag12"    : round(acf12, 4),
            "spike_rate"   : round(spike_rate, 4),
            "mean_ep_len"  : round(mean_ep_len, 2),
            "p95_cpu"      : round(p95, 4),
        })

    return pd.DataFrame(records)


def _eda_warnings(eda_df: pd.DataFrame) -> list[str]:
    """Return human-readable domain-incompatibility warnings."""
    warnings = []

    n_idle_warn = int((eda_df["idle_fraction"] > _WARN_IDLE_FRACTION).sum())
    if n_idle_warn:
        warnings.append(
            f"{n_idle_warn}/{len(eda_df)} nodes have idle_fraction > {_WARN_IDLE_FRACTION:.0%} "
            f"(Google 2011: ~15–20%). Per-machine p95 features may cluster at 0 during idle."
        )

    n_biz_warn = int((eda_df["biz_ratio"] > _WARN_BUSINESS_RATIO).sum())
    if n_biz_warn:
        warnings.append(
            f"{n_biz_warn}/{len(eda_df)} nodes have business_hours_ratio > {_WARN_BUSINESS_RATIO}x. "
            "Hour features (hour_sin/cos) will dominate — more than in Google 2011 (only 7 days)."
        )

    # Pairwise node correlation (using mean CPU per bucket as signal)
    if len(eda_df) >= 2:
        # Load machine-level CPU per bucket from the eda context — proxy: use mean_cpu column only
        # (full pairwise requires the original df; warn loosely based on spike_rate similarity)
        spike_rates = eda_df["spike_rate"].values
        sr_std = float(np.std(spike_rates))
        if sr_std < 0.02:
            warnings.append(
                f"All nodes have similar spike_rate (std={sr_std:.3f}). "
                "cluster_cpu_p90 and machine_rank_in_cluster may carry less signal "
                f"than across Google's 12,555 diverse machines."
            )

    return warnings


# ── Phase 1 — PSI ────────────────────────────────────────────────────────────

def _compute_psi_single(
    expected: np.ndarray,
    actual:   np.ndarray,
    n_bins:   int = _N_PSI_BINS,
) -> float:
    """Compute Population Stability Index for one feature.

    Bin edges derived from `expected` (Google training) quantiles so that
    every expected bin is non-empty by construction.  A small epsilon (1e-4)
    guards against log(0).

    Returns nan if the feature has zero variance in either distribution.
    """
    expected = expected[np.isfinite(expected)]
    actual   = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    # Binary features (only 0 and 1 values) → 2 exact bins
    unique_vals = np.unique(expected)
    if len(unique_vals) <= 2 and set(unique_vals).issubset({0.0, 1.0}):
        edges = np.array([-0.5, 0.5, 1.5])
    else:
        quantiles = np.linspace(0, 100, n_bins + 1)
        edges     = np.unique(np.percentile(expected, quantiles))
        if len(edges) < 2:
            return float("nan")

    eps = 1e-4
    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual,   bins=edges)

    exp_pct = (exp_counts / len(expected)).clip(eps)
    act_pct = (act_counts / len(actual)).clip(eps)

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return psi


def _compute_psi(
    google_features_path: Path,
    zabbix_features_df:   pd.DataFrame,
) -> pd.DataFrame:
    """Compute PSI for all model features (Google train vs Zabbix full dataset).

    Loads only the needed columns from the Google features parquet and
    downsamples to 10% of the training split to keep memory bounded.
    """
    log.info("Loading Google training features for PSI (10%% sample)…")
    google_df    = pd.read_parquet(google_features_path, columns=["bucket"] + list(_X_COLS))
    bmin, bmax   = int(google_df["bucket"].min()), int(google_df["bucket"].max())
    train_bucket_max = bmin + int((bmax - bmin) * _TRAIN_RATIO)
    google_train  = google_df[google_df["bucket"] <= train_bucket_max]
    google_sample = google_train.sample(frac=0.10, random_state=42)
    log.info("  Google sample: %d rows (10%% of %d training rows)",
             len(google_sample), len(google_train))

    # Zabbix: use ALL available rows (282K total — small enough for full use)
    zab_df = zabbix_features_df[[c for c in _X_COLS if c in zabbix_features_df.columns]]

    rows = []
    for col in _X_COLS:
        if col not in google_sample.columns or col not in zab_df.columns:
            rows.append({"feature": col, "psi": float("nan"), "interpretation": "missing"})
            continue

        g_vals = google_sample[col].values.astype("float64")
        z_vals = zab_df[col].values.astype("float64")
        psi    = _compute_psi_single(g_vals, z_vals)

        if np.isnan(psi):
            interp = "no variance"
        elif psi < _PSI_MINOR:
            interp = _PSI_LABEL_MINOR
        elif psi < _PSI_MODERATE:
            interp = _PSI_LABEL_MODERATE
        else:
            interp = _PSI_LABEL_MAJOR

        rows.append({"feature": col, "psi": round(psi, 4), "interpretation": interp})

    return pd.DataFrame(rows).sort_values("psi", ascending=False)


# ── Phase 2 — Model evaluation ────────────────────────────────────────────────

def _load_model(model_dir: Path) -> dict:
    """Load a single model's booster, feature cols, alarm threshold, and calibrators."""
    model_path  = model_dir / "spike_model.json"
    meta_path   = model_dir / "spike_model.meta.json"
    config_path = model_dir / "spike_config.json"

    for p in (model_path, meta_path, config_path):
        if not p.exists():
            raise FileNotFoundError(f"Required artifact not found: {p}")

    meta   = json.loads(meta_path.read_text())
    config = json.loads(config_path.read_text())

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    booster.set_param("nthread", 1)

    cal_path     = model_dir / "calibrators.pkl"
    calibrators  = None
    if cal_path.exists():
        with open(cal_path, "rb") as f:
            calibrators = pickle.load(f)

    return {
        "booster"         : booster,
        "feature_cols"    : meta["feature_cols"],
        "alarm_threshold" : float(config["alarm_threshold"]),
        "calibrators"     : calibrators,
        "config"          : config,
        "use_focal"       : bool(meta.get("use_focal_loss", False)),
    }


def _calibrate_60m(raw: np.ndarray, calibrators: list | None) -> np.ndarray:
    """Apply per-class isotonic calibration; renormalise to sum=1."""
    if calibrators is None:
        return raw
    cal = np.column_stack([
        np.clip(calibrators[k].predict(raw[:, k].astype("float64")), 0.0, 1.0)
        for k in range(3)
    ])
    row_sums = cal.sum(axis=1, keepdims=True)
    uniform  = np.full_like(cal, 1.0 / 3)
    return np.where(row_sums > 0, cal / row_sums, uniform).astype("float32")


def _eval_60m(
    features_df:  pd.DataFrame,
    test_mask:    pd.Series,
    model_info:   dict,
) -> dict:
    """Evaluate 60m severity model on the test split."""
    test_df = features_df[test_mask].dropna(subset=["severity_in_60m"])
    if len(test_df) == 0:
        return {"error": "no test rows after label drop"}

    X      = test_df[model_info["feature_cols"]].astype("float32").values
    y_true = test_df["severity_in_60m"].astype(int).values

    raw     = model_info["booster"].inplace_predict(X).reshape(-1, 3)
    probas  = _calibrate_60m(raw, model_info["calibrators"])

    # Per-class and macro PR-AUC / ROC-AUC
    results: dict[str, Any] = {"n_test": len(test_df)}
    pr_aucs, roc_aucs = [], []
    for cls in range(3):
        y_bin    = (y_true == cls).astype(int)
        n_pos    = int(y_bin.sum())
        if n_pos == 0 or n_pos == len(y_bin):
            pr_auc = float("nan")
            roc_auc = float("nan")
        else:
            pr_auc  = float(average_precision_score(y_bin, probas[:, cls]))
            roc_auc = float(roc_auc_score(y_bin, probas[:, cls]))
        results[f"pr_auc_class_{cls}"] = round(pr_auc, 4)
        results[f"roc_auc_class_{cls}"] = round(roc_auc, 4)
        if not np.isnan(pr_auc):
            pr_aucs.append(pr_auc)
        if not np.isnan(roc_auc):
            roc_aucs.append(roc_auc)

    results["macro_pr_auc"]  = round(float(np.mean(pr_aucs)),  4) if pr_aucs  else float("nan")
    results["macro_roc_auc"] = round(float(np.mean(roc_aucs)), 4) if roc_aucs else float("nan")

    class_rates = {
        f"rate_class_{c}": round(float((y_true == c).mean()), 4) for c in range(3)
    }
    results.update(class_rates)
    return results


def _eval_binary(
    features_df: pd.DataFrame,
    test_mask:   pd.Series,
    model_info:  dict,
    label_col:   str,
) -> dict:
    """Evaluate a binary model on the test split."""
    if label_col not in features_df.columns:
        return {"error": f"label column '{label_col}' not found"}

    test_df = features_df[test_mask].dropna(subset=[label_col])
    if len(test_df) == 0:
        return {"error": "no test rows after label drop"}

    X      = test_df[model_info["feature_cols"]].astype("float32").values
    y_true = test_df[label_col].astype(int).values
    n_pos  = int(y_true.sum())

    if n_pos == 0 or n_pos == len(y_true):
        return {"n_test": len(test_df), "n_pos": n_pos, "pr_auc": float("nan"), "roc_auc": float("nan")}

    raw = model_info["booster"].inplace_predict(X, strict_shape=True)  # (n, 1)
    if model_info.get("use_focal", False):
        p1 = np.clip(1.0 / (1.0 + np.exp(-raw[:, 0].astype("float64"))), 0.0, 1.0)
    else:
        p1 = raw[:, 0].astype("float64")

    pr_auc  = float(average_precision_score(y_true, p1))
    roc_auc = float(roc_auc_score(y_true, p1))
    return {
        "n_test"  : len(test_df),
        "n_pos"   : n_pos,
        "pos_rate": round(float(y_true.mean()), 4),
        "pr_auc"  : round(pr_auc,  4),
        "roc_auc" : round(roc_auc, 4),
    }


# ── Phase 3 — Recalibration ───────────────────────────────────────────────────

def _threshold_sweep(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float, float, float]:
    """Sweep thresholds [0.05, 0.95] and return (best_threshold, precision, recall, f1)."""
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


def _bootstrap_pr_auc(
    y_true:      np.ndarray,
    y_score:     np.ndarray,
    n_resamples: int = 1000,
    seed:        int = 42,
) -> tuple[float, float]:
    """95% percentile bootstrap CI for average_precision_score."""
    rng  = np.random.default_rng(seed)
    n    = len(y_true)
    boot = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if yt.sum() == 0 or yt.sum() == n:
            boot[i] = float("nan")
        else:
            boot[i] = average_precision_score(yt, ys)
    valid = boot[np.isfinite(boot)]
    if len(valid) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))


def _recalibrate_60m(
    features_df: pd.DataFrame,
    val_mask:    pd.Series,
    test_mask:   pd.Series,
    model_info:  dict,
    output_dir:  Path,
) -> dict:
    """Fit Zabbix-local OvR isotonic calibrators on val; re-evaluate on test. Saves artifacts."""
    label_col = "severity_in_60m"
    val_df    = features_df[val_mask].dropna(subset=[label_col])
    test_df   = features_df[test_mask].dropna(subset=[label_col])

    if len(val_df) < 1000:
        return {"error": f"val split too small for isotonic calibration: {len(val_df)} rows"}

    feat_cols = model_info["feature_cols"]
    X_val  = val_df[feat_cols].astype("float32").values
    X_test = test_df[feat_cols].astype("float32").values
    y_val  = val_df[label_col].astype(int).values
    y_test = test_df[label_col].astype(int).values

    raw_val  = model_info["booster"].inplace_predict(X_val).reshape(-1, 3)
    raw_test = model_info["booster"].inplace_predict(X_test).reshape(-1, 3)

    # OvR isotonic calibrators trained on Zabbix val raw XGBoost probs (replaces Google calibrators)
    zab_calibrators = [
        IsotonicRegression(out_of_bounds="clip").fit(
            raw_val[:, k].astype("float64"), (y_val == k).astype(int)
        )
        for k in range(3)
    ]

    def _apply_zab_cal(raw: np.ndarray) -> np.ndarray:
        cal = np.column_stack([
            np.clip(zab_calibrators[k].predict(raw[:, k].astype("float64")), 0.0, 1.0)
            for k in range(3)
        ])
        row_sums = cal.sum(axis=1, keepdims=True)
        return np.where(row_sums > 0, cal / row_sums, np.full_like(cal, 1.0 / 3)).astype("float32")

    zab_cal_val  = _apply_zab_cal(raw_val)
    zab_cal_test = _apply_zab_cal(raw_test)

    # F1-max threshold sweep on val, alarm on P(severe) = calibrated[:, 2]
    best_t, best_p, best_r, best_f1 = _threshold_sweep(
        (y_val == 2).astype(int), zab_cal_val[:, 2]
    )

    # Per-class metrics with bootstrap CI on test
    pr_aucs, roc_aucs = [], []
    per_class: dict = {}
    for cls in range(3):
        y_bin = (y_test == cls).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        pa      = float(average_precision_score(y_bin, zab_cal_test[:, cls]))
        ra      = float(roc_auc_score(y_bin, zab_cal_test[:, cls]))
        ci_lo, ci_hi = _bootstrap_pr_auc(y_bin, zab_cal_test[:, cls])
        pr_aucs.append(pa)
        roc_aucs.append(ra)
        per_class[f"class_{cls}"] = {
            "pr_auc":      round(pa,     4),
            "roc_auc":     round(ra,     4),
            "pr_auc_ci95": [round(ci_lo, 4), round(ci_hi, 4)],
        }

    macro_pr_auc  = round(float(np.mean(pr_aucs)),  4) if pr_aucs  else float("nan")
    macro_roc_auc = round(float(np.mean(roc_aucs)), 4) if roc_aucs else float("nan")

    # Save artifacts
    cal_path = output_dir / "zabbix_calibrators.pkl"
    with open(cal_path, "wb") as f:
        pickle.dump(zab_calibrators, f, protocol=4)

    config_out = {
        "alarm_threshold":  round(best_t,  2),
        "alarm_precision":  round(best_p,  4),
        "alarm_recall":     round(best_r,  4),
        "alarm_f1":         round(best_f1, 4),
        "macro_pr_auc":     macro_pr_auc,
        "macro_roc_auc":    macro_roc_auc,
        "per_class":        per_class,
        "n_val":            len(val_df),
        "n_test":           len(test_df),
        "calibrators_path": str(cal_path),
    }
    config_path = output_dir / "zabbix_spike_config.json"
    config_path.write_text(json.dumps(config_out, indent=2))

    log.info("  Phase 3 60m: macro PR-AUC %.3f  |  alarm @ %.2f  (P=%.3f R=%.3f F1=%.3f)",
             macro_pr_auc, best_t, best_p, best_r, best_f1)
    log.info("  Saved: %s", cal_path)
    log.info("  Saved: %s", config_path)
    return config_out


def _recalibrate_binary(
    features_df: pd.DataFrame,
    val_mask:    pd.Series,
    test_mask:   pd.Series,
    model_info:  dict,
    label_col:   str,
) -> dict:
    """Threshold sweep on Zabbix val; re-evaluate on test with bootstrap CI."""
    if label_col not in features_df.columns:
        return {"error": f"label column '{label_col}' not found"}

    val_df  = features_df[val_mask].dropna(subset=[label_col])
    test_df = features_df[test_mask].dropna(subset=[label_col])

    if len(val_df) < 100:
        return {"error": f"val split too small: {len(val_df)} rows"}

    feat_cols = model_info["feature_cols"]
    X_val  = val_df[feat_cols].astype("float32").values
    X_test = test_df[feat_cols].astype("float32").values
    y_val  = val_df[label_col].astype(int).values
    y_test = test_df[label_col].astype(int).values

    def _predict(X: np.ndarray) -> np.ndarray:
        raw = model_info["booster"].inplace_predict(X, strict_shape=True)
        if model_info.get("use_focal", False):
            return np.clip(1.0 / (1.0 + np.exp(-raw[:, 0].astype("float64"))), 0.0, 1.0)
        return raw[:, 0].astype("float64")

    p1_val  = _predict(X_val)
    p1_test = _predict(X_test)

    best_t, best_p, best_r, best_f1 = _threshold_sweep(y_val, p1_val)

    n_pos_test = int(y_test.sum())
    if n_pos_test == 0 or n_pos_test == len(y_test):
        return {"error": "degenerate test labels"}

    pr_auc  = float(average_precision_score(y_test, p1_test))
    roc_auc = float(roc_auc_score(y_test, p1_test))
    ci_lo, ci_hi = _bootstrap_pr_auc(y_test, p1_test)

    return {
        "alarm_threshold": round(best_t,  2),
        "alarm_precision": round(best_p,  4),
        "alarm_recall":    round(best_r,  4),
        "alarm_f1":        round(best_f1, 4),
        "pr_auc":          round(pr_auc,  4),
        "roc_auc":         round(roc_auc, 4),
        "pr_auc_ci95":     [round(ci_lo, 4), round(ci_hi, 4)],
        "n_val":           len(val_df),
        "n_test":          len(test_df),
    }


# ── Report generation ─────────────────────────────────────────────────────────

def _decision(pr_auc: float, source: float = _SOURCE_PR_AUC) -> tuple[str, str]:
    """Return (retention_pct_str, action) based on PR-AUC retention."""
    if np.isnan(pr_auc) or np.isnan(source) or source <= 0:
        return "N/A", "Cannot determine — source PR-AUC not available."
    retention = pr_auc / source
    if retention >= 0.70:
        action = "**Recalibrate on Zabbix val split and deploy.** Model transfers well."
    elif retention >= 0.40:
        action = "**Adapt.** Fine-tune with ~300 warm-start boosting rounds on Zabbix train split."
    else:
        action = "**Retrain from scratch** on Zabbix data (use Google best_params.json as Optuna seed)."
    return f"{retention:.0%}", action


def _generate_report(
    eda_df:       pd.DataFrame,
    eda_warnings: list[str],
    psi_df:       pd.DataFrame | None,
    results_60m:  dict,
    results_15m:  dict,
    results_30m:  dict,
    results_45m:  dict,
    results_ovr:  dict,
    recal_60m:    dict,
    recal_15m:    dict,
    recal_30m:    dict,
    recal_45m:    dict,
    recal_ovr:    dict,
    node_map:     dict,
    run_at:       str,
) -> str:
    node_names = {v: k for k, v in node_map.items()} if node_map else {}
    eda_df     = eda_df.copy()
    eda_df["node"] = eda_df["machine_id"].map(node_names).fillna(eda_df["machine_id"].astype(str))

    pr_60m  = results_60m.get("macro_pr_auc",  float("nan"))
    pr_15m  = results_15m.get("pr_auc",         float("nan"))
    pr_ovr  = results_ovr.get("pr_auc",         float("nan"))
    roc_60m = results_60m.get("macro_roc_auc",  float("nan"))
    roc_15m = results_15m.get("roc_auc",         float("nan"))
    roc_ovr = results_ovr.get("roc_auc",         float("nan"))

    ret_pct, action = _decision(pr_60m)

    lines = [
        "# Zabbix Evaluation Report",
        f"Generated: {run_at}",
        f"Source model (Google Cluster 2011 K=2 baseline): 60m PR-AUC = {_SOURCE_PR_AUC}",
        "",
        "---",
        "",
        "## Phase 0 — EDA",
        "",
    ]

    # EDA table
    eda_cols = ["node", "n_buckets", "idle_fraction", "burstiness_B",
                "pmr", "spike_rate", "biz_ratio", "acf_lag1"]
    lines.append("| " + " | ".join(eda_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(eda_cols)) + " |")
    for _, row in eda_df.iterrows():
        vals = [str(row.get(c, "—")) for c in eda_cols]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")

    if eda_warnings:
        lines.append("### ⚠️  Domain-incompatibility warnings")
        for w in eda_warnings:
            lines.append(f"- {w}")
    else:
        lines.append("✅ No domain-incompatibility warnings.")
    lines.append("")

    # PSI
    lines += ["---", "", "## Phase 1 — Domain Shift (PSI)", ""]
    if psi_df is None:
        lines.append("*Skipped — Google cluster_features.parquet not found on this machine.*")
    else:
        major  = psi_df[psi_df["interpretation"] == _PSI_LABEL_MAJOR]
        mod    = psi_df[psi_df["interpretation"] == _PSI_LABEL_MODERATE]
        minor  = psi_df[psi_df["interpretation"] == _PSI_LABEL_MINOR]
        lines.append(f"Feature shift summary: {len(major)} MAJOR / {len(mod)} moderate / {len(minor)} minor")
        lines.append("")
        if not major.empty:
            lines.append("### MAJOR shift features (PSI > 0.25) — model degraded on these:")
            lines.append("")
            lines.append("| Feature | PSI |")
            lines.append("| --- | --- |")
            for _, r in major.iterrows():
                lines.append(f"| {r['feature']} | {r['psi']} |")
            lines.append("")
        if not mod.empty:
            lines.append("### Moderate shift features (0.10 < PSI ≤ 0.25):")
            lines.append("")
            lines.append("| Feature | PSI |")
            lines.append("| --- | --- |")
            for _, r in mod.iterrows():
                lines.append(f"| {r['feature']} | {r['psi']} |")
    lines.append("")

    # Model evaluation
    lines += ["---", "", "## Phase 2 — Model Evaluation", ""]
    lines.append(f"Test split: last 20% of Zabbix data (~18 days). Labels computed with K=1 (any exceedance).")
    lines.append("")
    lines.append("| Model | Test PR-AUC | Source PR-AUC | Retention | Test ROC-AUC | N test |")
    lines.append("| --- | --- | --- | --- | --- | --- |")

    def _fmt(v: Any) -> str:
        return f"{v:.3f}" if isinstance(v, float) and not np.isnan(v) else str(v)

    pr_30m  = results_30m.get("pr_auc",  float("nan"))
    pr_45m  = results_45m.get("pr_auc",  float("nan"))
    roc_30m = results_30m.get("roc_auc", float("nan"))
    roc_45m = results_45m.get("roc_auc", float("nan"))

    ret60  = ret_pct
    ret15, _  = _decision(pr_15m, source=_SOURCE_PR_AUC_15M)
    ret30, _  = _decision(pr_30m, source=_SOURCE_PR_AUC_30M)
    ret45, _  = _decision(pr_45m, source=_SOURCE_PR_AUC_45M)
    retovr, _ = _decision(pr_ovr, source=_SOURCE_PR_AUC_OVR)

    src_30m = f"{_SOURCE_PR_AUC_30M:.3f}" if not np.isnan(_SOURCE_PR_AUC_30M) else "—"
    src_45m = f"{_SOURCE_PR_AUC_45M:.3f}" if not np.isnan(_SOURCE_PR_AUC_45M) else "—"

    lines.append(f"| 60m severity  | {_fmt(pr_60m)} | {_SOURCE_PR_AUC}     | {ret60}  | {_fmt(roc_60m)} | {results_60m.get('n_test','—')} |")
    lines.append(f"| 15m binary    | {_fmt(pr_15m)} | {_SOURCE_PR_AUC_15M} | {ret15}  | {_fmt(roc_15m)} | {results_15m.get('n_test','—')} |")
    lines.append(f"| 30m binary    | {_fmt(pr_30m)} | {src_30m}            | {ret30}  | {_fmt(roc_30m)} | {results_30m.get('n_test','—')} |")
    lines.append(f"| 45m binary    | {_fmt(pr_45m)} | {src_45m}            | {ret45}  | {_fmt(roc_45m)} | {results_45m.get('n_test','—')} |")
    lines.append(f"| OVR severe    | {_fmt(pr_ovr)} | {_SOURCE_PR_AUC_OVR} | {retovr} | {_fmt(roc_ovr)} | {results_ovr.get('n_test','—')} |")
    lines.append("")

    # Per-class 60m breakdown
    lines.append("### 60m per-class PR-AUC")
    lines.append("")
    lines.append("| Class | Label | Test PR-AUC | Class rate |")
    lines.append("| --- | --- | --- | --- |")
    for c in range(3):
        pa = results_60m.get(f"pr_auc_class_{c}", float("nan"))
        cr = results_60m.get(f"rate_class_{c}",   float("nan"))
        lines.append(f"| {c} | {_CLASS_LABELS[c]} | {_fmt(pa)} | {_fmt(cr)} |")
    lines.append("")

    # Decision
    lines += ["---", "", "## Decision", ""]
    lines.append(f"60m macro PR-AUC retention vs source: **{ret_pct}**")
    lines.append("")
    lines.append(f"→ {action}")
    lines.append("")

    # Phase 3 — Recalibration
    lines += ["---", "", "## Phase 3 — Zabbix Recalibration", ""]
    if not recal_60m or "error" in recal_60m:
        lines.append(f"*60m recalibration skipped — {(recal_60m or {}).get('error', 'not run')}*")
        lines.append("")
    else:
        lines.append("### 60m Severity (Zabbix-recalibrated)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| Macro PR-AUC | {_fmt(recal_60m.get('macro_pr_auc', float('nan')))} |")
        lines.append(f"| Macro ROC-AUC | {_fmt(recal_60m.get('macro_roc_auc', float('nan')))} |")
        lines.append(f"| Alarm threshold | {recal_60m.get('alarm_threshold', '—')} |")
        lines.append(
            f"| Alarm P / R / F1 | "
            f"{_fmt(recal_60m.get('alarm_precision', float('nan')))} / "
            f"{_fmt(recal_60m.get('alarm_recall', float('nan')))} / "
            f"{_fmt(recal_60m.get('alarm_f1', float('nan')))} |"
        )
        lines.append("")
        per_class = recal_60m.get("per_class", {})
        if per_class:
            lines.append("| Class | Label | PR-AUC | 95% CI | ROC-AUC |")
            lines.append("| --- | --- | --- | --- | --- |")
            for c in range(3):
                cm   = per_class.get(f"class_{c}", {})
                if not cm:
                    continue
                ci_c = cm.get("pr_auc_ci95", [float("nan"), float("nan")])
                lines.append(
                    f"| {c} | {_CLASS_LABELS[c]} | {_fmt(cm.get('pr_auc', float('nan')))} "
                    f"| [{_fmt(ci_c[0])}, {_fmt(ci_c[1])}] | {_fmt(cm.get('roc_auc', float('nan')))} |"
                )
            lines.append("")

    for name, recal in [
        ("15m Binary", recal_15m),
        ("30m Binary", recal_30m),
        ("45m Binary", recal_45m),
        ("OVR Severe", recal_ovr),
    ]:
        if not recal or "error" in recal:
            lines.append(f"*{name} recalibration skipped — {(recal or {}).get('error', 'not run')}*")
            lines.append("")
        else:
            lines.append(f"### {name} (Zabbix threshold)")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("| --- | --- |")
            lines.append(f"| PR-AUC | {_fmt(recal.get('pr_auc', float('nan')))} |")
            ci = recal.get("pr_auc_ci95", [float("nan"), float("nan")])
            lines.append(f"| PR-AUC 95% CI | [{_fmt(ci[0])}, {_fmt(ci[1])}] |")
            lines.append(f"| ROC-AUC | {_fmt(recal.get('roc_auc', float('nan')))} |")
            lines.append(f"| Alarm threshold | {recal.get('alarm_threshold', '—')} |")
            lines.append(
                f"| Alarm P / R / F1 | "
                f"{_fmt(recal.get('alarm_precision', float('nan')))} / "
                f"{_fmt(recal.get('alarm_recall', float('nan')))} / "
                f"{_fmt(recal.get('alarm_f1', float('nan')))} |"
            )
            lines.append("")

    return "\n".join(lines)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def evaluate(
    agg_path:      Path,
    artifacts_dir: Path,
    output_dir:    Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_at = datetime.now(tz=timezone.utc).isoformat()

    # Load node map for human-readable names in the report
    node_map_path = agg_path.parent / "node_map.json"
    node_map: dict = {}
    if node_map_path.exists():
        node_map = json.loads(node_map_path.read_text())

    log.info("═" * 62)
    log.info("  Loading Zabbix aggregated data")
    log.info("═" * 62)
    df_agg = pd.read_parquet(agg_path)
    log.info("  Rows: %d  |  Machines: %d  |  Buckets: %d",
             len(df_agg), df_agg["machine_id"].nunique(), df_agg["bucket"].nunique())

    # ── Phase 0: EDA ─────────────────────────────────────────────────────────
    log.info("═" * 62)
    log.info("  Phase 0 — EDA")
    log.info("═" * 62)
    eda_df   = _compute_node_eda(df_agg)
    warnings = _eda_warnings(eda_df)
    log.info("  EDA complete — %d nodes", len(eda_df))
    for w in warnings:
        log.warning("  ⚠️  %s", w)

    # ── Phase 1: PSI (optional) ───────────────────────────────────────────────
    log.info("═" * 62)
    log.info("  Phase 1 — Domain shift PSI")
    log.info("═" * 62)
    psi_df: pd.DataFrame | None = None
    zab_features: pd.DataFrame | None = None

    google_features_path = artifacts_dir / "cluster_features.parquet"
    if google_features_path.exists():
        try:
            log.info("  Running feature engineering on Zabbix data for PSI…")
            zab_features = engineer(
                input_path         = agg_path,
                output_path        = output_dir / "zabbix_features.parquet",
                thresholds_path    = output_dir / "zabbix_thresholds.parquet",
                min_future_windows = 1,
            )
            psi_df = _compute_psi(google_features_path, zab_features)
            major_count = int((psi_df["interpretation"] == _PSI_LABEL_MAJOR).sum())
            mod_count   = int((psi_df["interpretation"] == _PSI_LABEL_MODERATE).sum())
            log.info("  PSI: %d MAJOR / %d moderate / %d minor features",
                     major_count, mod_count,
                     int((psi_df["interpretation"] == _PSI_LABEL_MINOR).sum()))
        except Exception as exc:
            log.warning("  PSI computation failed: %s — skipping Phase 1", exc)
            psi_df = None
    else:
        log.warning(
            "  Google features parquet not found at %s — skipping Phase 1 PSI.",
            google_features_path,
        )

    # ── Phase 2: Feature engineering + model evaluation ───────────────────────
    log.info("═" * 62)
    log.info("  Phase 2 — Feature engineering + model evaluation")
    log.info("═" * 62)

    if zab_features is None:
        log.info("  Running feature engineering on Zabbix data…")
        zab_features = engineer(
            input_path         = agg_path,
            output_path        = output_dir / "zabbix_features.parquet",
            thresholds_path    = output_dir / "zabbix_thresholds.parquet",
            min_future_windows = 1,
        )

    log.info("  Feature rows: %d  |  Label coverage: %d (%.1f%%)",
             len(zab_features),
             int(zab_features["severity_in_60m"].notna().sum()),
             100 * zab_features["severity_in_60m"].notna().mean())

    # Chronological splits — train 60% / val 20% / test 20%
    b_min     = int(zab_features["bucket"].min())
    b_max     = int(zab_features["bucket"].max())
    n_b       = b_max - b_min
    train_max = b_min + int(n_b * _TRAIN_RATIO)
    val_end   = b_min + int(n_b * (_TRAIN_RATIO + _VAL_RATIO))
    val_mask  = (zab_features["bucket"] > train_max) & (zab_features["bucket"] <= val_end)
    test_mask = zab_features["bucket"] > val_end
    log.info("  Val split: %d rows  |  Test split: %d rows",
             int(val_mask.sum()), int(test_mask.sum()))

    # Derive spike_severe_ovr label (class 2 vs rest)
    if "severity_in_60m" in zab_features.columns:
        zab_features["spike_severe_ovr"] = (
            (zab_features["severity_in_60m"] == 2).astype("float32")
        )

    # Load models
    model_dirs = {
        "60m"       : artifacts_dir / "models" / "spike",
        "15m"       : artifacts_dir / "models" / "spike_15m",
        "30m"       : artifacts_dir / "models" / "spike_30m",
        "45m"       : artifacts_dir / "models" / "spike_45m",
        "severe_ovr": artifacts_dir / "models" / "spike_severe_ovr",
    }
    models: dict[str, dict] = {}
    for name, mdir in model_dirs.items():
        try:
            models[name] = _load_model(mdir)
            log.info("  Loaded model: %s  (alarm @ %.2f)",
                     name, models[name]["alarm_threshold"])
        except FileNotFoundError as e:
            log.warning("  Model %s not found: %s — skipped", name, e)

    # Evaluate
    results_60m = _eval_60m(zab_features, test_mask, models["60m"]) if "60m" in models else {}
    results_15m = (
        _eval_binary(zab_features, test_mask, models["15m"],        "spike_in_15m")
        if "15m" in models else {}
    )
    results_30m = (
        _eval_binary(zab_features, test_mask, models["30m"],        "spike_in_30m")
        if "30m" in models else {}
    )
    results_45m = (
        _eval_binary(zab_features, test_mask, models["45m"],        "spike_in_45m")
        if "45m" in models else {}
    )
    results_ovr = (
        _eval_binary(zab_features, test_mask, models["severe_ovr"], "spike_severe_ovr")
        if "severe_ovr" in models else {}
    )

    log.info("  60m macro PR-AUC : %.3f  |  ROC-AUC : %.3f",
             results_60m.get("macro_pr_auc", float("nan")),
             results_60m.get("macro_roc_auc", float("nan")))
    for label, res in [("15m", results_15m), ("30m", results_30m), ("45m", results_45m)]:
        log.info("  %s PR-AUC        : %.3f  |  ROC-AUC : %.3f",
                 label, res.get("pr_auc", float("nan")), res.get("roc_auc", float("nan")))
    log.info("  OVR severe PR-AUC: %.3f  |  ROC-AUC : %.3f",
             results_ovr.get("pr_auc", float("nan")),
             results_ovr.get("roc_auc", float("nan")))

    # ── Phase 3: Recalibration ────────────────────────────────────────────────
    log.info("═" * 62)
    log.info("  Phase 3 — Zabbix recalibration")
    log.info("═" * 62)

    recal_60m: dict = {}
    recal_15m: dict = {}
    recal_30m: dict = {}
    recal_45m: dict = {}
    recal_ovr: dict = {}

    if "60m" in models:
        recal_60m = _recalibrate_60m(
            features_df = zab_features,
            val_mask    = val_mask,
            test_mask   = test_mask,
            model_info  = models["60m"],
            output_dir  = output_dir,
        )

    for h_name, label_col, recal_dict_ref in [
        ("15m",        "spike_in_15m",    "recal_15m"),
        ("30m",        "spike_in_30m",    "recal_30m"),
        ("45m",        "spike_in_45m",    "recal_45m"),
        ("severe_ovr", "spike_severe_ovr","recal_ovr"),
    ]:
        if h_name not in models:
            continue
        result = _recalibrate_binary(
            features_df = zab_features,
            val_mask    = val_mask,
            test_mask   = test_mask,
            model_info  = models[h_name],
            label_col   = label_col,
        )
        log.info("  Phase 3 %s: PR-AUC %.3f  |  alarm @ %.2f  (P=%.3f R=%.3f F1=%.3f)",
                 h_name,
                 result.get("pr_auc", float("nan")),
                 result.get("alarm_threshold", float("nan")),
                 result.get("alarm_precision", float("nan")),
                 result.get("alarm_recall", float("nan")),
                 result.get("alarm_f1", float("nan")))
        if recal_dict_ref == "recal_15m":
            recal_15m = result
        elif recal_dict_ref == "recal_30m":
            recal_30m = result
        elif recal_dict_ref == "recal_45m":
            recal_45m = result
        else:
            recal_ovr = result

    # ── Write outputs ─────────────────────────────────────────────────────────
    log.info("═" * 62)
    log.info("  Writing outputs")
    log.info("═" * 62)

    pr_auc_60m = results_60m.get("macro_pr_auc", float("nan"))
    retention, action = _decision(pr_auc_60m)

    results_json = {
        "run_at"            : run_at,
        "source_pr_auc"     : _SOURCE_PR_AUC,
        "zabbix_agg_path"   : str(agg_path),
        "artifacts_dir"     : str(artifacts_dir),
        "eda"               : eda_df.to_dict(orient="records"),
        "eda_warnings"      : warnings,
        "psi"               : psi_df.to_dict(orient="records") if psi_df is not None else None,
        "results_60m"       : results_60m,
        "results_15m"       : results_15m,
        "results_30m"       : results_30m,
        "results_45m"       : results_45m,
        "results_severe_ovr": results_ovr,
        "decision"          : {"retention_pct": retention, "action": action},
        "recal_60m"         : recal_60m,
        "recal_15m"         : recal_15m,
        "recal_30m"         : recal_30m,
        "recal_45m"         : recal_45m,
        "recal_ovr"         : recal_ovr,
    }

    results_path = output_dir / "evaluation_results.json"
    results_path.write_text(json.dumps(results_json, indent=2, default=str))
    log.info("  Results JSON : %s", results_path)

    report_md = _generate_report(
        eda_df       = eda_df,
        eda_warnings = warnings,
        psi_df       = psi_df,
        results_60m  = results_60m,
        results_15m  = results_15m,
        results_30m  = results_30m,
        results_45m  = results_45m,
        results_ovr  = results_ovr,
        recal_60m    = recal_60m,
        recal_15m    = recal_15m,
        recal_30m    = recal_30m,
        recal_45m    = recal_45m,
        recal_ovr    = recal_ovr,
        node_map     = node_map,
        run_at       = run_at,
    )
    report_path = output_dir / "evaluation_report.md"
    report_path.write_text(report_md)
    log.info("  Report MD    : %s", report_path)

    log.info("═" * 62)
    log.info("  RESULT   60m PR-AUC: %.3f  (%.0f%% of source %.3f)",
             pr_auc_60m, pr_auc_60m / _SOURCE_PR_AUC * 100 if not np.isnan(pr_auc_60m) else float("nan"),
             _SOURCE_PR_AUC)
    log.info("  DECISION %s", action.replace("**", "").replace("*", ""))
    log.info("═" * 62)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description = "3-phase cross-domain evaluation of the spike model on Zabbix data.",
    )
    p.add_argument(
        "--agg",
        type    = Path,
        default = Path("data/zabbix_eval/zabbix_agg.parquet"),
        metavar = "PATH",
        help    = "Path to zabbix_agg.parquet (output of fetch_zabbix_data.py).",
    )
    p.add_argument(
        "--artifacts-dir",
        dest    = "artifacts_dir",
        type    = Path,
        default = Path("data/full_run"),
        metavar = "DIR",
        help    = "Directory with full_run artifacts (models/, cluster_features.parquet).",
    )
    p.add_argument(
        "--output",
        type    = Path,
        default = Path("data/zabbix_eval"),
        metavar = "DIR",
        help    = "Output directory for report and results JSON.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.agg.exists():
        log.error("Zabbix aggregated data not found: %s  — run fetch_zabbix_data.py first.", args.agg)
        sys.exit(1)

    evaluate(
        agg_path      = args.agg,
        artifacts_dir = args.artifacts_dir,
        output_dir    = args.output,
    )
