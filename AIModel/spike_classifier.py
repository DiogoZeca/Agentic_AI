"""Train and evaluate the XGBoost binary spike classifier.

Reads cluster_features.parquet (output of spike_feature_engineer.py) and
trains an XGBoost binary classifier to predict spike_in_60m — whether any
5-min bucket in the next 60 minutes will exceed the node's p95 CPU threshold.

Design notes
------------
Train/test split is strictly time-based (by bucket), never random.  A random
split would expose future load patterns to training and inflate all metrics.
The split ratio must match the value used in spike_feature_engineer.engineer()
so that the spike thresholds (computed on training data) are not contaminated
by test-period observations.

scale_pos_weight is computed from the training split only — using the full
dataset would let test-set class distribution influence training.

Feature names are NOT preserved by XGBoost's save_model/load_model. A
companion <model>.meta.json file stores the feature column list so that the
correct column order is restored on load.

Input  : data/cluster_features.parquet
Output : data/spike_model.json       (XGBoost native format)
         data/spike_model.meta.json  (feature column list + version)

Usage:
    python spike_classifier.py
    python spike_classifier.py --features data/cluster_features.parquet \\
                                --model    data/spike_model.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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

# Columns passed to XGBoost — metadata and label excluded.
# machine_id / bucket / time_us intentionally absent: the model must generalise
# across machines and time, not memorise IDs or timestamps.
_X_COLS: list[str] = [
    "total_cpu", "peak_cpu", "total_mem", "peak_mem", "disk_io", "n_tasks",
    "cpu_lag_1", "cpu_lag_2", "cpu_lag_3", "cpu_lag_6", "cpu_lag_12", "cpu_lag_24",
    "cpu_ewma_6", "cpu_ewma_24",
    "cpu_delta_1",
    "task_dominance",
    "cpu_vs_p95",   # machine-relative: total_cpu / machine_p95 threshold
]

_TRAIN_RATIO:  float = 0.6   # must match spike_feature_engineer._TRAIN_RATIO
_VAL_RATIO:    float = 0.2   # validation slice for threshold selection (no leakage)
_MIN_POSITIVE: int   = 10   # minimum spike rows in training; fewer → degenerate model

# ── SpikeClassifier ───────────────────────────────────────────────────────────


class SpikeClassifier:
    """XGBoost binary classifier for 60-minute CPU spike prediction.

    Wraps XGBClassifier with explicit feature column management and
    save/load logic that preserves feature names, which XGBoost's native
    save_model / load_model do not preserve automatically.

    Parameters
    ----------
    scale_pos_weight : ``neg / pos`` ratio from training data.  Set to 1.0 as a
                       neutral default; ``train()`` always overrides this with the
                       measured value.
    """

    def __init__(
        self,
        scale_pos_weight: float = 1.0,
        n_estimators:     int   = 300,
        max_depth:        int   = 6,
        learning_rate:    float = 0.05,
        subsample:        float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int   = 1,    # 1–3 recommended for rare-event detection;
                                        # 10 is too conservative — minority class
                                        # samples form small leaves that 10 discards
        random_state:     int   = 42,
    ) -> None:
        self._feature_cols: list[str] = list(_X_COLS)
        self._model = xgb.XGBClassifier(
            objective        = "binary:logistic",
            eval_metric      = "logloss",
            tree_method      = "hist",   # fast CPU histogram method (not deprecated in XGBoost 2.x)
            n_estimators     = n_estimators,
            max_depth        = max_depth,
            learning_rate    = learning_rate,
            subsample        = subsample,
            colsample_bytree = colsample_bytree,
            min_child_weight = min_child_weight,
            scale_pos_weight = scale_pos_weight,
            random_state     = random_state,   # sklearn API param name (not "seed")
            n_jobs           = -1,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _to_array(self, df: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix as float32 (consistent dtype for XGBoost)."""
        return df[self._feature_cols].astype("float32").values

    # ── Public interface ──────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SpikeClassifier":
        """Fit on training data.

        Parameters
        ----------
        X : DataFrame containing at least the columns in _X_COLS.
        y : integer Series {0, 1} — spike labels, NaN-free.

        Returns
        -------
        self
        """
        self._model.fit(self._to_array(X), y.values)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return spike probability for each row.

        Returns
        -------
        1-D float64 array of shape (n,) with values in [0.0, 1.0].
        Column 1 of XGBClassifier.predict_proba (P(spike=1)) is returned.
        """
        return self._model.predict_proba(self._to_array(X))[:, 1]

    def find_threshold(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Find the decision threshold that maximises F1 on the given data.

        Scans 0.05 → 0.95 in steps of 0.05 and returns the threshold with the
        highest F1.  Always call this on **training data only** — using test
        labels to pick the threshold leaks information into the evaluation.

        Parameters
        ----------
        X : DataFrame with _X_COLS columns.
        y : integer Series {0, 1} — spike labels.

        Returns
        -------
        float threshold in (0.0, 1.0).
        """
        proba  = self.predict_proba(X)
        y_vals = y.values

        best_thresh = 0.5
        best_f1     = 0.0

        for thresh in np.arange(0.05, 1.0, 0.05):
            pred = (proba >= thresh).astype(int)
            f1   = float(f1_score(y_vals, pred, zero_division=0))
            if f1 > best_f1:
                best_f1     = f1
                best_thresh = float(thresh)

        return best_thresh

    def evaluate(
        self,
        X:         pd.DataFrame,
        y:         pd.Series,
        threshold: float = 0.5,
    ) -> dict[str, float]:
        """Evaluate on a test set.

        PR-AUC is the primary metric for imbalanced spike detection and is
        threshold-independent.  Precision / recall / F1 use the supplied
        threshold; pass the result of ``find_threshold()`` to report calibrated
        metrics alongside the default 0.5 numbers.

        If the test set contains only one class (e.g., no spikes), AUC
        metrics are undefined and returned as NaN rather than raising.

        Parameters
        ----------
        X         : DataFrame with _X_COLS columns.
        y         : integer Series {0, 1} — spike labels.
        threshold : decision threshold for precision / recall / F1 (default 0.5).

        Returns
        -------
        dict with keys: threshold, pr_auc, roc_auc, precision, recall, f1.
        """
        proba     = self.predict_proba(X)
        pred      = (proba >= threshold).astype(int)
        y_vals    = y.values
        n_classes = len(np.unique(y_vals))

        # AUC metrics are undefined when only one class is present.
        # average_precision_score returns 0.0 with a warning (not a ValueError),
        # so we must guard explicitly rather than relying on try/except.
        if n_classes < 2:
            pr_auc  = float("nan")
            roc_auc = float("nan")
        else:
            pr_auc  = float(average_precision_score(y_vals, proba))
            roc_auc = float(roc_auc_score(y_vals, proba))

        return {
            "threshold": float(threshold),
            "pr_auc":    pr_auc,
            "roc_auc":   roc_auc,
            "precision": float(precision_score(y_vals, pred, zero_division=0)),
            "recall":    float(recall_score(y_vals, pred, zero_division=0)),
            "f1":        float(f1_score(y_vals, pred, zero_division=0)),
        }

    def save(self, path: str | Path) -> None:
        """Save XGBoost model and feature-column metadata to disk.

        Two files are written:
          <path>           — XGBoost native JSON model (booster weights etc.)
          <path>.meta.json — feature column order + XGBoost version

        Feature names are NOT preserved by XGBoost's save_model.  The meta
        file ensures the correct column order is restored on ``load()``.

        Parameters
        ----------
        path : destination path for the model file (e.g. ``data/spike_model.json``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))

        meta      = {"feature_cols": self._feature_cols, "xgboost_version": xgb.__version__}
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2))

        log.info("  Model saved    : %s", path)
        log.info("  Metadata saved : %s", meta_path)

    @classmethod
    def load(cls, path: str | Path) -> "SpikeClassifier":
        """Load a SpikeClassifier from disk.

        Reads the XGBoost model and restores the feature column list from
        the companion .meta.json file.

        Parameters
        ----------
        path : path to the model file (e.g. ``data/spike_model.json``).

        Raises
        ------
        FileNotFoundError
            If the model file or its companion metadata file is missing.
        """
        path      = Path(path)
        meta_path = path.with_suffix(".meta.json")

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {meta_path}. "
                "Re-save the model with SpikeClassifier.save()."
            )

        meta              = json.loads(meta_path.read_text())
        obj               = cls()
        obj._model.load_model(str(path))
        obj._feature_cols = meta["feature_cols"]
        return obj


# ── Public entry point ────────────────────────────────────────────────────────


def train(
    features_path: str | Path = "data/cluster_features.parquet",
    model_path:    str | Path = "data/spike_model.json",
    train_ratio:   float      = _TRAIN_RATIO,
    val_ratio:     float      = _VAL_RATIO,
) -> dict:
    """Train SpikeClassifier and save the model.

    Time-based three-way split: train / val / test (default 60 % / 20 % / 20 %).

    - **train** : model is fitted here; ``scale_pos_weight`` derived here.
    - **val**   : ``find_threshold()`` is called here — the only place threshold
                  selection touches data.  Never the test set.
    - **test**  : final metrics only.  Never touched during training or threshold
                  selection.

    ``train_ratio`` **must match** the value passed to
    ``spike_feature_engineer.engineer()`` so that the per-machine p95 thresholds
    were computed from the same training window.

    Parameters
    ----------
    features_path : path to cluster_features.parquet.
    model_path    : destination for the saved model and companion meta file.
    train_ratio   : fraction of buckets used for training (default 0.6).
    val_ratio     : fraction of buckets used for validation / threshold selection
                    (default 0.2).  Remaining buckets become the test set.

    Returns
    -------
    dict with 18 keys covering split sizes, spike rates, threshold-independent
    AUC metrics, default-threshold metrics, and calibrated-threshold metrics.

    Raises
    ------
    FileNotFoundError
        If features_path does not exist.
    ValueError
        If fewer than _MIN_POSITIVE spike rows exist in the training split.
    """
    features_path = Path(features_path)
    model_path    = Path(model_path)

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    log.info("═" * 62)
    log.info("  SPIKE CLASSIFIER — training")
    log.info("  Input      : %s", features_path)
    log.info("  Model      : %s", model_path)
    log.info(
        "  Split      : train %.0f%% / val %.0f%% / test %.0f%%",
        train_ratio * 100,
        val_ratio * 100,
        (1.0 - train_ratio - val_ratio) * 100,
    )
    log.info("═" * 62)

    df = pd.read_parquet(features_path)

    # Drop rows without a valid label (last horizon windows per machine)
    df = df[df["spike_in_60m"].notna()].copy()
    df["spike_in_60m"] = df["spike_in_60m"].astype("int8")

    # Time-based three-way split — never random to prevent future-data leakage.
    # train_ratio must match spike_feature_engineer._TRAIN_RATIO so that the
    # per-machine p95 thresholds were computed from this exact training window.
    bucket_max = int(df["bucket"].max())
    train_max  = int(bucket_max * train_ratio)
    val_max    = int(bucket_max * (train_ratio + val_ratio))

    train_df = df[df["bucket"] <= train_max].reset_index(drop=True)
    val_df   = df[(df["bucket"] > train_max) & (df["bucket"] <= val_max)].reset_index(drop=True)
    test_df  = df[df["bucket"] > val_max].reset_index(drop=True)

    y_train = train_df["spike_in_60m"]
    y_val   = val_df["spike_in_60m"]
    y_test  = test_df["spike_in_60m"]

    # scale_pos_weight from training split only — val/test must not influence it
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())

    if pos < _MIN_POSITIVE:
        raise ValueError(
            f"Only {pos} positive (spike) example(s) in training data "
            f"(minimum required: {_MIN_POSITIVE}). "
            "Check that spike thresholds are not too high or train_ratio not too low."
        )

    scale_pos_weight = neg / pos

    log.info(
        "  Train rows : %s  (spike rate %.1f%%)",
        f"{len(train_df):,}", 100 * pos / (neg + pos),
    )
    log.info(
        "  Val rows   : %s  (spike rate %.1f%%)",
        f"{len(val_df):,}", 100 * float(y_val.mean()),
    )
    log.info(
        "  Test rows  : %s  (spike rate %.1f%%)",
        f"{len(test_df):,}", 100 * float(y_test.mean()),
    )
    log.info("  scale_pos_weight : %.2f", scale_pos_weight)

    t_start = time.perf_counter()
    clf = SpikeClassifier(scale_pos_weight=scale_pos_weight)
    clf.fit(train_df, y_train)
    elapsed = time.perf_counter() - t_start
    log.info("  Training time : %.1f s", elapsed)

    # Find optimal threshold on the validation set — never on train (overfits the
    # threshold to training noise) and never on test (leaks evaluation signal).
    optimal_thresh = clf.find_threshold(val_df, y_val)

    m_default    = clf.evaluate(test_df, y_test, threshold=0.5)
    m_calibrated = clf.evaluate(test_df, y_test, threshold=optimal_thresh)

    log.info("  ─────────────────────────────────────────────────────")
    log.info("  PR-AUC    : %.3f  ← primary metric (threshold-independent)",
             m_default["pr_auc"])
    log.info("  ROC-AUC   : %.3f", m_default["roc_auc"])
    log.info("  At threshold 0.50 (default):")
    log.info("    Precision : %.3f   Recall : %.3f   F1 : %.3f",
             m_default["precision"], m_default["recall"], m_default["f1"])
    log.info("  At threshold %.2f (F1-optimal, calibrated on val data):",
             optimal_thresh)
    log.info("    Precision : %.3f   Recall : %.3f   F1 : %.3f",
             m_calibrated["precision"], m_calibrated["recall"], m_calibrated["f1"])
    log.info("═" * 62)

    clf.save(model_path)

    return {
        "train_rows":           len(train_df),
        "val_rows":             len(val_df),
        "test_rows":            len(test_df),
        "train_bucket_max":     train_max,
        "val_bucket_max":       val_max,
        "spike_rate_train":     float(pos / (neg + pos)),
        "spike_rate_val":       float(y_val.mean()),
        "spike_rate_test":      float(y_test.mean()),
        "scale_pos_weight":     scale_pos_weight,
        # threshold-independent metrics
        "pr_auc":               m_default["pr_auc"],
        "roc_auc":              m_default["roc_auc"],
        # at default 0.5 threshold (test set)
        "precision":            m_default["precision"],
        "recall":               m_default["recall"],
        "f1":                   m_default["f1"],
        # at F1-optimal threshold calibrated on val data (test set)
        "optimal_threshold":    optimal_thresh,
        "precision_calibrated": m_calibrated["precision"],
        "recall_calibrated":    m_calibrated["recall"],
        "f1_calibrated":        m_calibrated["f1"],
    }


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description     = "Train the XGBoost spike classifier on cluster_features.parquet.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = (
            "examples:\n"
            "  python spike_classifier.py\n"
            "  python spike_classifier.py --features data/cluster_features.parquet\n"
            "  python spike_classifier.py --features data/cluster_features.parquet"
            " --model data/spike_model.json\n"
        ),
    )
    p.add_argument(
        "--features",
        default = "data/cluster_features.parquet",
        metavar = "PATH",
        help    = "Path to cluster_features.parquet  (default: data/cluster_features.parquet)",
    )
    p.add_argument(
        "--model",
        default = "data/spike_model.json",
        metavar = "PATH",
        help    = "Output model path  (default: data/spike_model.json)",
    )
    p.add_argument(
        "--train-ratio",
        type    = float,
        default = _TRAIN_RATIO,
        metavar = "R",
        help    = (
            f"Fraction of bucket range used for training  (default: {_TRAIN_RATIO}). "
            "Must match the value passed to spike_feature_engineer.engineer()."
        ),
    )
    p.add_argument(
        "--val-ratio",
        type    = float,
        default = _VAL_RATIO,
        metavar = "R",
        help    = (
            f"Fraction of bucket range used for validation / threshold selection "
            f"(default: {_VAL_RATIO}).  Remaining buckets become the test set."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        features_path = args.features,
        model_path    = args.model,
        train_ratio   = args.train_ratio,
        val_ratio     = args.val_ratio,
    )
