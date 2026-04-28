#!/usr/bin/env python3
"""
evaluate_baselines.py — Baseline comparison for the spike prediction paper.

Five baselines are evaluated against the same train/val/test splits and label
columns used by the XGBoost models, so results are directly comparable.

Baselines
---------
  random          — Uniform random score at test-set prevalence (sanity floor).
  persistence     — Score = total_cpu / p95 (is the machine already stressed?).
  static_thresh   — Fraction of last 4 buckets where total_cpu > p95.
  ewma_zscore     — Z-score of CPU relative to its EWMA mean/std (span=12).
  rolling_zscore  — Z-score of CPU relative to a 24-bucket rolling mean/std.
  arima           — 1-step-ahead ARIMA(2,1,0) forecast normalised by p95.
                    Fitted per machine on the training split.  Capped at
                    --arima-sample machines for the Google dataset (slow).

All baselines see only the raw cluster_agg schema (total_cpu, peak_cpu, …) plus
the per-machine p95/p99 thresholds computed from training data.  Labels are
loaded from cluster_features.parquet so both XGBoost and baselines share the
exact same ground truth.

Evaluated on Google Cluster 2011 test set by default.
Pass --zabbix to evaluate on the Zabbix dataset instead.

Usage (from AIModel/):
  .venv/bin/python3 baselines/evaluate_baselines.py
  .venv/bin/python3 baselines/evaluate_baselines.py --zabbix
  .venv/bin/python3 baselines/evaluate_baselines.py --arima-sample 200
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import average_precision_score, roc_auc_score

from spike.classifier import _VAL_RATIO
from spike.feature_engineer import _TRAIN_RATIO

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_ARIMA_ORDER    = (2, 1, 0)   # ARIMA(p,d,q) — AR(2) + first-difference; fits most CPU series
_EWMA_SPAN      = 12          # ~1 hour of history (12 × 5 min)
_ROLLING_WINDOW = 24          # ~2 hours of history for z-score baseline
_STATIC_LOOKBACK = 4          # 4 × 5 min = 20 min lookback for fraction-above-p95

_BINARY_LABELS = [
    ("spike_in_15m",  "15m"),
    ("spike_in_30m",  "30m"),
    ("spike_in_45m",  "45m"),
]
_MULTICLASS_LABEL = "severity_in_60m"
_OVR_LABEL        = "spike_severe_ovr"

# XGBoost test-set numbers for comparison (Google dataset, K=2 production)
_XGBOOST_GOOGLE: dict[str, float] = {
    "15m":  0.575,
    "30m":  0.563,
    "45m":  0.562,
    "60m":  0.547,
    "ovr":  0.339,
}
# XGBoost Zabbix recalibrated numbers
_XGBOOST_ZABBIX: dict[str, float] = {
    "15m":  0.957,
    "30m":  0.957,
    "45m":  0.957,
    "60m":  0.848,
    "ovr":  0.926,
}


# ── Baseline base class ───────────────────────────────────────────────────────

class _Baseline(ABC):
    """Abstract baseline.  Operates on the raw cluster_agg schema + p95/p99.

    Input DataFrame columns available to every baseline:
      machine_id, bucket, total_cpu, peak_cpu, total_mem, peak_mem,
      disk_io, n_tasks, p95, p99
    All rows are sorted by (machine_id, bucket) before score() is called.
    """

    name: str

    @abstractmethod
    def score(self, df: pd.DataFrame) -> pd.Series:
        """Return a stress score for every row.  Must be side-effect-free.

        Only current and past data are accessible — do NOT use future buckets.
        The shift(1) guard is applied inside each implementation where needed.

        Returns
        -------
        pd.Series with the same index as df, dtype float64, values ≥ 0.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ── Concrete baselines ────────────────────────────────────────────────────────

class RandomBaseline(_Baseline):
    """Uniform random score — sanity floor.  PR-AUC ≈ spike prevalence rate."""

    name = "random"

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def score(self, df: pd.DataFrame) -> pd.Series:
        rng = np.random.default_rng(self._seed)
        return pd.Series(rng.random(len(df)).astype("float64"), index=df.index)


class PersistenceBaseline(_Baseline):
    """Score = total_cpu / p95.

    Hypothesis: a machine already under stress will spike again.
    Normalised to p95 so scores are cross-machine comparable.
    """

    name = "persistence"

    def score(self, df: pd.DataFrame) -> pd.Series:
        return (df["total_cpu"] / df["p95"].clip(lower=1e-9)).clip(upper=20.0)


class StaticThresholdBaseline(_Baseline):
    """Score = fraction of last K buckets where total_cpu > p95.

    Represents the rule an operator would write in a monitoring tool.
    Uses a per-machine rolling mean of the binary exceedance indicator,
    shifted by 1 to exclude the current bucket from looking ahead.
    """

    name = "static_thresh"

    def __init__(self, lookback: int = _STATIC_LOOKBACK) -> None:
        self._k = lookback

    def score(self, df: pd.DataFrame) -> pd.Series:
        exceeds = (df["total_cpu"] > df["p95"]).astype("float64")

        def _rolling_frac(grp: pd.Series) -> pd.Series:
            # shift(1): predict using *previous* buckets, not current
            return grp.shift(1).rolling(self._k, min_periods=1).mean()

        return exceeds.groupby(df["machine_id"]).transform(_rolling_frac).fillna(0.0)


class EWMAZScoreBaseline(_Baseline):
    """Score = max(0, (cpu − ewma_mean) / ewma_std).

    Captures whether the current reading is anomalously high relative to the
    machine's own recent baseline.  The EWMA span is ~1 hour of history.
    Both mean and variance are shifted by 1 bucket to avoid lookahead.
    """

    name = "ewma_zscore"

    def __init__(self, span: int = _EWMA_SPAN) -> None:
        self._span = span

    def score(self, df: pd.DataFrame) -> pd.Series:
        def _z(grp: pd.DataFrame) -> pd.Series:
            cpu  = grp["total_cpu"]
            mu   = cpu.ewm(span=self._span, adjust=False).mean().shift(1)
            var  = (cpu - mu).pow(2).ewm(span=self._span, adjust=False).mean().shift(1)
            sigma = var.pow(0.5).clip(lower=1e-9)
            return ((cpu - mu) / sigma).clip(lower=0.0)

        return (
            df.groupby("machine_id", group_keys=False)
              .apply(_z)
              .astype("float64")
              .fillna(0.0)
        )


class RollingZScoreBaseline(_Baseline):
    """Score = max(0, (cpu − rolling_mean) / rolling_std).

    Uses a longer 2-hour window for mean/std estimation.
    Shifted by 1 bucket to prevent any lookahead.
    """

    name = "rolling_zscore"

    def __init__(self, window: int = _ROLLING_WINDOW) -> None:
        self._w = window

    def score(self, df: pd.DataFrame) -> pd.Series:
        def _z(grp: pd.DataFrame) -> pd.Series:
            cpu   = grp["total_cpu"]
            mu    = cpu.rolling(self._w, min_periods=4).mean().shift(1)
            sigma = cpu.rolling(self._w, min_periods=4).std().shift(1).clip(lower=1e-9)
            return ((cpu - mu) / sigma).clip(lower=0.0)

        return (
            df.groupby("machine_id", group_keys=False)
              .apply(_z)
              .astype("float64")
              .fillna(0.0)
        )


class ARIMABaseline(_Baseline):
    """Per-machine ARIMA(2,1,0) 1-step-ahead forecast, normalised by p95.

    Fit on the training split for each machine (or a random sample of
    `sample_n` machines for large datasets).  Machines not in the sample
    fall back to the PersistenceBaseline score.

    Score = max(0, forecast / p95 − 0.5) — positive when the forecast
    exceeds half of the p95 threshold, proportional to predicted overload.
    """

    name = "arima"

    def __init__(
        self,
        order:    tuple[int, int, int] = _ARIMA_ORDER,
        sample_n: int  = 500,
        seed:     int  = 42,
        train_ratio: float = _TRAIN_RATIO,
    ) -> None:
        self._order      = order
        self._sample_n   = sample_n
        self._seed       = seed
        self._train_ratio = train_ratio

    def score(self, df: pd.DataFrame) -> pd.Series:
        try:
            from statsmodels.tsa.arima.model import ARIMA as _ARIMA  # type: ignore
        except ImportError:
            log.warning("statsmodels not installed — ARIMA baseline skipped, using persistence.")
            return PersistenceBaseline().score(df)

        all_machines = df["machine_id"].unique()
        rng          = np.random.default_rng(self._seed)
        if len(all_machines) > self._sample_n:
            sampled = set(rng.choice(all_machines, self._sample_n, replace=False).tolist())
            log.info("  ARIMA: sampling %d / %d machines", self._sample_n, len(all_machines))
        else:
            sampled = set(all_machines.tolist())

        # Fallback scores for machines outside the sample
        fallback = PersistenceBaseline().score(df)
        scores   = fallback.copy()

        b_min = int(df["bucket"].min())
        b_max = int(df["bucket"].max())
        train_cutoff = b_min + int((b_max - b_min) * self._train_ratio)

        n_ok = n_fail = 0
        for mid, grp in df.groupby("machine_id"):
            if mid not in sampled:
                continue

            cpu    = grp["total_cpu"].values.astype("float64")
            buckets = grp["bucket"].values
            p95    = float(grp["p95"].iloc[0])

            train_mask = buckets <= train_cutoff
            n_train    = int(train_mask.sum())
            if n_train < 20:
                continue

            try:
                fitted = _ARIMA(cpu[:n_train], order=self._order).fit(
                    method_kwargs={"warn_convergence": False}
                )
                # In-sample fitted values for train portion
                preds = np.full(len(cpu), np.nan)
                preds[:n_train] = fitted.fittedvalues

                # Out-of-sample: apply() updates state with all observations at
                # once (O(n_test)) without re-fitting — correct 1-step-ahead.
                if len(cpu) > n_train:
                    applied  = fitted.apply(cpu[n_train:], refit=False)
                    preds[n_train:] = applied.fittedvalues

                machine_scores = pd.Series(
                    np.maximum(0.0, preds / max(p95, 1e-9) - 0.5),
                    index=grp.index,
                )
                scores.loc[grp.index] = machine_scores
                n_ok += 1
            except Exception:
                n_fail += 1

        log.info("  ARIMA: fitted %d machines OK, %d failed (fallback used)", n_ok, n_fail)
        return scores.fillna(0.0)


# ── Evaluation helpers ────────────────────────────────────────────────────────


def _eval_binary(
    y_true: np.ndarray,
    scores: np.ndarray,
    label:  str,
) -> dict[str, Any]:
    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == len(y_true):
        return {"label": label, "n_test": len(y_true), "n_pos": n_pos,
                "pr_auc": float("nan"), "roc_auc": float("nan")}

    pr_auc  = float(average_precision_score(y_true, scores))
    roc_auc = float(roc_auc_score(y_true, scores))
    return {
        "label":    label,
        "n_test":   len(y_true),
        "n_pos":    n_pos,
        "pos_rate": round(float(y_true.mean()), 4),
        "pr_auc":   round(pr_auc,  4),
        "roc_auc":  round(roc_auc, 4),
    }


def _eval_multiclass_ovr(
    y_true:  np.ndarray,
    scores:  np.ndarray,
    n_classes: int = 3,
) -> dict[str, Any]:
    pr_aucs, roc_aucs = [], []
    per_class: dict = {}
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        pa = float(average_precision_score(y_bin, scores))
        ra = float(roc_auc_score(y_bin, scores))
        pr_aucs.append(pa)
        roc_aucs.append(ra)
        per_class[f"class_{c}"] = {"pr_auc": round(pa, 4), "roc_auc": round(ra, 4)}

    return {
        "macro_pr_auc":  round(float(np.mean(pr_aucs)),  4) if pr_aucs  else float("nan"),
        "macro_roc_auc": round(float(np.mean(roc_aucs)), 4) if roc_aucs else float("nan"),
        "per_class":     per_class,
        "n_test":        len(y_true),
    }


def _evaluate_baseline(
    baseline: _Baseline,
    unified:  pd.DataFrame,
) -> dict[str, Any]:
    """Run one baseline and return all metrics.

    unified is the merged agg+feat test DataFrame (machine_id, bucket, total_cpu,
    p95, p99, + all label columns), sorted by (machine_id, bucket).
    """
    log.info("  Scoring: %s", baseline.name)
    scores = baseline.score(unified).values.astype("float64")

    results: dict[str, Any] = {"baseline": baseline.name}

    # Binary horizons
    for col, tag in _BINARY_LABELS:
        if col not in unified.columns:
            continue
        mask   = unified[col].notna().values
        y_true = unified.loc[mask, col].astype(int).values
        results[tag] = _eval_binary(y_true, scores[mask], col)

    # 60m multiclass (OvR macro PR-AUC — same score for all classes)
    if _MULTICLASS_LABEL in unified.columns:
        mask   = unified[_MULTICLASS_LABEL].notna().values
        y_true = unified.loc[mask, _MULTICLASS_LABEL].astype(int).values
        results["60m"] = _eval_multiclass_ovr(y_true, scores[mask])

    # OVR severe (binary: class 2 vs rest)
    if _MULTICLASS_LABEL in unified.columns:
        mask  = unified[_MULTICLASS_LABEL].notna().values
        y_bin = (unified.loc[mask, _MULTICLASS_LABEL].astype(int) == 2).astype(int).values
        results["ovr"] = _eval_binary(y_bin, scores[mask], "spike_severe_ovr")

    return results


# ── Report generation ─────────────────────────────────────────────────────────

def _fmt(v: Any, precision: int = 3) -> str:
    if isinstance(v, float) and not np.isnan(v):
        return f"{v:.{precision}f}"
    return str(v)


def _generate_report(
    all_results: list[dict],
    xgb_ref:     dict[str, float],
    dataset:     str,
) -> str:
    horizons  = ["15m", "30m", "45m", "60m", "ovr"]
    h_labels  = ["15m PR-AUC", "30m PR-AUC", "45m PR-AUC", "60m macro PR-AUC", "OVR PR-AUC"]

    lines = [
        f"# Baseline Comparison — {dataset}",
        "",
        "All baselines evaluated on the **test split** (last 20% chronologically).",
        "Scores computed from raw cluster_agg schema only (no engineered features).",
        "Labels and splits identical to XGBoost evaluation.",
        "",
        "## PR-AUC Comparison",
        "",
    ]

    # Header
    col_w = 20
    header = "| {:<{w}} |".format("Baseline", w=col_w)
    sep    = "| {:<{w}} |".format("---", w=col_w)
    for h in h_labels:
        header += f" {h} |"
        sep    += " --- |"
    lines += [header, sep]

    # Baseline rows
    for r in all_results:
        bname = r["baseline"]
        row   = "| {:<{w}} |".format(bname, w=col_w)
        for h in horizons:
            if h not in r:
                row += " — |"
                continue
            m = r[h]
            key = "macro_pr_auc" if h == "60m" else "pr_auc"
            row += f" {_fmt(m.get(key, float('nan')))} |"
        lines.append(row)

    # XGBoost reference row
    xgb_row = "| {:<{w}} |".format("**XGBoost (ours)**", w=col_w)
    for h in horizons:
        xgb_row += f" **{_fmt(xgb_ref.get(h, float('nan')))}** |"
    lines.append(xgb_row)
    lines.append("")

    # Per-horizon detail
    lines += ["## Per-Horizon Detail", ""]
    for h, hlabel in zip(horizons, h_labels):
        lines += [f"### {hlabel}", ""]
        lines += ["| Baseline | PR-AUC | ROC-AUC | N test | Pos rate |"]
        lines += ["| --- | --- | --- | --- | --- |"]
        for r in all_results:
            if h not in r:
                continue
            m = r[h]
            key = "macro_pr_auc" if h == "60m" else "pr_auc"
            roc_key = "macro_roc_auc" if h == "60m" else "roc_auc"
            lines.append(
                f"| {r['baseline']} "
                f"| {_fmt(m.get(key, float('nan')))} "
                f"| {_fmt(m.get(roc_key, float('nan')))} "
                f"| {m.get('n_test', '—')} "
                f"| {_fmt(m.get('pos_rate', float('nan')), 4)} |"
            )
        xgb_pa = xgb_ref.get(h, float("nan"))
        lines.append(f"| **XGBoost** | **{_fmt(xgb_pa)}** | — | — | — |")
        lines.append("")

    return "\n".join(lines)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def _load_google(artifacts_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Google cluster data: (agg, features, thresholds)."""
    agg_path  = artifacts_dir / "cluster_agg.parquet"
    feat_path = artifacts_dir / "cluster_features.parquet"
    thr_path  = artifacts_dir / "spike_thresholds.parquet"

    for p in (agg_path, feat_path, thr_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    log.info("Loading cluster_agg.parquet …")
    agg  = pd.read_parquet(agg_path)
    log.info("Loading cluster_features.parquet (labels only) …")
    # Load only label columns + keys to keep memory bounded
    label_cols = ["machine_id", "bucket"] + [
        c for c in [_MULTICLASS_LABEL] + [x for x, _ in _BINARY_LABELS]
        if True  # all needed
    ]
    feat = pd.read_parquet(feat_path, columns=label_cols)
    log.info("Loading spike_thresholds.parquet …")
    thr  = pd.read_parquet(thr_path)

    return agg, feat, thr


def _load_zabbix(
    artifacts_dir: Path,
    zabbix_dir:    Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Zabbix data: (agg, features, thresholds)."""
    agg_path  = zabbix_dir / "zabbix_agg.parquet"
    feat_path = zabbix_dir / "zabbix_features.parquet"
    thr_path  = zabbix_dir / "zabbix_thresholds.parquet"

    for p in (agg_path, feat_path, thr_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                "Run evaluate_zabbix.py first to generate Zabbix features."
            )

    log.info("Loading zabbix_agg.parquet …")
    agg  = pd.read_parquet(agg_path)
    log.info("Loading zabbix_features.parquet (labels only) …")
    label_cols = ["machine_id", "bucket"] + [_MULTICLASS_LABEL] + [x for x, _ in _BINARY_LABELS]
    feat = pd.read_parquet(feat_path, columns=label_cols)
    log.info("Loading zabbix_thresholds.parquet …")
    thr  = pd.read_parquet(thr_path)

    return agg, feat, thr


def _derive_ovr_label(feat: pd.DataFrame) -> pd.DataFrame:
    """Add spike_severe_ovr = (severity_in_60m == 2) if not already present."""
    if _OVR_LABEL not in feat.columns and _MULTICLASS_LABEL in feat.columns:
        feat = feat.copy()
        feat[_OVR_LABEL] = (feat[_MULTICLASS_LABEL] == 2).astype("float32")
        feat.loc[feat[_MULTICLASS_LABEL].isna(), _OVR_LABEL] = float("nan")
    return feat


def _split_test(
    df: pd.DataFrame,
    train_ratio: float = _TRAIN_RATIO,
    val_ratio:   float = _VAL_RATIO,
) -> pd.DataFrame:
    b_min = int(df["bucket"].min())
    b_max = int(df["bucket"].max())
    val_end = b_min + int((b_max - b_min) * (train_ratio + val_ratio))
    return df[df["bucket"] > val_end].copy()


def evaluate(
    artifacts_dir: Path,
    output_dir:    Path,
    zabbix:        bool = False,
    zabbix_dir:    Path | None = None,
    arima_sample:  int  = 500,
    skip_arima:    bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info("═" * 62)
    if zabbix:
        zdir = zabbix_dir or (artifacts_dir.parent / "zabbix_eval")
        agg, feat, thr = _load_zabbix(artifacts_dir, zdir)
        dataset_label  = "Zabbix"
        xgb_ref        = _XGBOOST_ZABBIX
    else:
        agg, feat, thr = _load_google(artifacts_dir)
        dataset_label  = "Google Cluster 2011"
        xgb_ref        = _XGBOOST_GOOGLE

    feat = _derive_ovr_label(feat)

    log.info(
        "  Loaded: agg=%d rows, feat=%d rows, %d machines",
        len(agg), len(feat), agg["machine_id"].nunique(),
    )

    # ── Merge p95/p99 thresholds onto agg ─────────────────────────────────────
    # spike_thresholds.parquet uses "threshold_p95" / "threshold_p99" columns
    thr_clean = thr[["machine_id"]].copy()
    for src, dst in [("threshold_p95", "p95"), ("threshold_p99", "p99"),
                     ("p95", "p95"), ("p99", "p99")]:
        if src in thr.columns and dst not in thr_clean.columns:
            thr_clean[dst] = thr[src].values
    agg = agg.merge(thr_clean, on="machine_id", how="left")
    agg["p95"] = agg["p95"].fillna(agg["total_cpu"].quantile(0.95))
    if "p99" not in agg.columns:
        agg["p99"] = agg["total_cpu"].quantile(0.99)
    agg = agg.sort_values(["machine_id", "bucket"]).reset_index(drop=True)

    # ── Test split + unified DataFrame ───────────────────────────────────────
    # Merge label columns from feat onto agg once — baselines score on raw
    # columns (total_cpu, p95…) and evaluate on labels from the same rows.
    label_cols = [_MULTICLASS_LABEL] + [c for c, _ in _BINARY_LABELS]
    feat_labels = feat[["machine_id", "bucket"] + label_cols].copy()

    agg_unified = agg.merge(feat_labels, on=["machine_id", "bucket"], how="inner")
    agg_unified = agg_unified.sort_values(["machine_id", "bucket"]).reset_index(drop=True)
    unified_test = _split_test(agg_unified)
    log.info("  Test split: %d rows, %d machines", len(unified_test),
             unified_test["machine_id"].nunique())

    # ── Define baselines ──────────────────────────────────────────────────────
    baselines: list[_Baseline] = [
        RandomBaseline(),
        PersistenceBaseline(),
        StaticThresholdBaseline(),
        EWMAZScoreBaseline(),
        RollingZScoreBaseline(),
    ]
    if not skip_arima:
        baselines.append(ARIMABaseline(sample_n=arima_sample))

    # ── Evaluate ──────────────────────────────────────────────────────────────
    log.info("═" * 62)
    log.info("  Evaluating %d baselines on %s test split", len(baselines), dataset_label)
    log.info("═" * 62)

    all_results: list[dict] = []
    for bl in baselines:
        log.info("─" * 40)
        result = _evaluate_baseline(bl, unified_test)
        all_results.append(result)
        pr_15m = result.get("15m", {}).get("pr_auc", float("nan"))
        pr_60m = result.get("60m", {}).get("macro_pr_auc", float("nan"))
        log.info(
            "  %-20s  15m PR-AUC=%.3f  60m macro PR-AUC=%.3f",
            bl.name, pr_15m, pr_60m,
        )

    # ── Write outputs ─────────────────────────────────────────────────────────
    log.info("═" * 62)
    suffix = "zabbix" if zabbix else "google"

    json_path = output_dir / f"baselines_results_{suffix}.json"
    json_path.write_text(json.dumps(all_results, indent=2, default=str))
    log.info("  Results JSON : %s", json_path)

    report = _generate_report(all_results, xgb_ref, dataset_label)
    md_path = output_dir / f"baselines_report_{suffix}.md"
    md_path.write_text(report)
    log.info("  Report MD    : %s", md_path)

    # Summary table to stdout
    log.info("═" * 62)
    log.info("  SUMMARY — %s", dataset_label)
    log.info("  %-20s  %8s  %8s  %8s  %8s  %8s",
             "Baseline", "15m", "30m", "45m", "60m", "OVR")
    for r in all_results:
        log.info(
            "  %-20s  %8s  %8s  %8s  %8s  %8s",
            r["baseline"],
            _fmt(r.get("15m", {}).get("pr_auc", float("nan"))),
            _fmt(r.get("30m", {}).get("pr_auc", float("nan"))),
            _fmt(r.get("45m", {}).get("pr_auc", float("nan"))),
            _fmt(r.get("60m", {}).get("macro_pr_auc", float("nan"))),
            _fmt(r.get("ovr", {}).get("pr_auc", float("nan"))),
        )
    log.info("  %-20s  %8s  %8s  %8s  %8s  %8s",
             "XGBoost (ours)",
             _fmt(xgb_ref["15m"]), _fmt(xgb_ref["30m"]),
             _fmt(xgb_ref["45m"]), _fmt(xgb_ref["60m"]),
             _fmt(xgb_ref["ovr"]))
    log.info("═" * 62)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate spike-prediction baselines for paper comparison.",
    )
    p.add_argument(
        "--artifacts-dir",
        dest    = "artifacts_dir",
        type    = Path,
        default = Path("data/full_run"),
        metavar = "DIR",
        help    = "Directory with production artifacts (cluster_agg/features/thresholds).",
    )
    p.add_argument(
        "--output",
        type    = Path,
        default = Path("data/baselines"),
        metavar = "DIR",
        help    = "Output directory for results JSON and report MD.",
    )
    p.add_argument(
        "--zabbix",
        action  = "store_true",
        help    = "Evaluate on Zabbix dataset instead of Google.",
    )
    p.add_argument(
        "--zabbix-dir",
        dest    = "zabbix_dir",
        type    = Path,
        default = None,
        metavar = "DIR",
        help    = "Path to zabbix_eval/ directory (default: <artifacts-dir>/../zabbix_eval).",
    )
    p.add_argument(
        "--arima-sample",
        dest    = "arima_sample",
        type    = int,
        default = 500,
        metavar = "N",
        help    = "Max machines to fit ARIMA on (default 500 — Google has 12,555).",
    )
    p.add_argument(
        "--skip-arima",
        dest   = "skip_arima",
        action = "store_true",
        help   = "Skip ARIMA baseline (fast mode for iteration).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        artifacts_dir = args.artifacts_dir,
        output_dir    = args.output,
        zabbix        = args.zabbix,
        zabbix_dir    = args.zabbix_dir,
        arima_sample  = args.arima_sample,
        skip_arima    = args.skip_arima,
    )
