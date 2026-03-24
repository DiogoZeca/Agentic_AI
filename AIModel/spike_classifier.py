"""Train and evaluate the XGBoost 3-class severity classifier.

Reads cluster_features.parquet (output of spike_feature_engineer.py) and
trains a multi:softprob classifier to predict severity_in_60m:
  0 = no_spike   — no p95 exceedance in the next 60 minutes
  1 = moderate   — p95 exceeded but p99 not exceeded
  2 = severe     — p99 exceeded in the next 60 minutes

Design notes
------------
Train/test split is strictly time-based (by bucket), never random.  A random
split would expose future load patterns to training and inflate all metrics.
The split ratio must match the value used in spike_feature_engineer.engineer()
so that the spike thresholds (computed on training data) are not contaminated
by test-period observations.

Class balancing uses per-sample weights (compute_sample_weight('balanced', y))
computed from the training split only — using the full dataset would let
test-set class distribution influence training.

Monotone constraints are disabled for multi:softprob: a +1 constraint on a
feature forces raw scores for *all* K classes to increase, which produces
contradictory P(no_spike) behaviour post-softmax.

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
    # raw signals
    "total_cpu", "peak_cpu", "total_mem", "peak_mem", "disk_io", "n_tasks",
    # lag history (Phase 2: dropped cpu_lag_2/3/6 — combined gain 0.010)
    "cpu_lag_1", "cpu_lag_12", "cpu_lag_24",
    # trend
    "cpu_ewma_6", "cpu_ewma_24",
    # derivatives
    "cpu_delta_1", "cpu_delta_2", "cpu_rolling_std_6",
    # machine-relative
    "task_dominance",
    "cpu_vs_p95",        # total_cpu / machine p95 — > 1.0 means already spiking
    "cpu_vs_p95_delta",  # rate of change of the normalised signal
    "peak_cpu_vs_p95",   # peak_cpu / machine p95 — near-miss detection (Phase 2)
    # p99-level features (Fix 1)
    "spike_severe_now",       # is current window already above p99? (binary)
    "cpu_vs_p99",             # total_cpu / machine p99 threshold
    "peak_cpu_vs_p99",        # peak_cpu / machine p99 threshold
    "band_position",          # fractional position in moderate band [p95, p99]
    "band_width",             # width of moderate zone for this machine (scalar)
    "spike_severe_in_last_1", # was t-1 a severe spike? (binary)
    "spike_severe_in_last_3", # any severe spike in t-3..t-1? (binary)
    "spike_severe_in_last_6", # any severe spike in t-6..t-1? (binary)
    # spike history (highest expected predictive gain)
    # NOTE: all derived from total_cpu > threshold, never from spike_in_60m label
    "spike_now",              # is current window already a spike?
    "spike_in_last_1",        # was t-1 a spike?
    "spike_in_last_3",        # any spike in t-3..t-1?
    "spike_in_last_6",        # any spike in t-6..t-1?
    "time_since_last_spike",  # windows since last exceedance (capped at 24)
    "cpu_spike_rate_24",      # fraction of previous 24 windows spiking (Phase 2)
    # cluster-level (cross-sectional, same bucket t — not temporal leakage)
    "cluster_cpu_p90",          # p90 CPU across all machines at this timestamp
    "machine_rank_in_cluster",  # percentile rank of this machine at this timestamp
    # time-of-day / day-of-week (harmonic encoding)
    "hour_sin", "hour_cos",  # within-day periodicity
    "dow_sin",  "dow_cos",   # within-week periodicity
]

_TRAIN_RATIO:  float = 0.6   # must match spike_feature_engineer._TRAIN_RATIO
_VAL_RATIO:    float = 0.2   # validation slice for alarm threshold selection (no leakage)
_MIN_POSITIVE: int   = 10   # minimum spike rows in training; fewer → degenerate model
_NUM_CLASSES:  int   = 3    # 0=no_spike, 1=moderate, 2=severe

# NOTE: Monotone constraints are intentionally disabled for multi:softprob
# (SpikeClassifier).  With K-class softprob, a +1 constraint forces raw scores
# for ALL K classes to increase together.  Post-softmax, this produces
# contradictory behaviour: P(no_spike) would also increase when cpu_vs_p95
# rises — physically wrong.  No asymmetric per-class constraint is supported
# in XGBoost 2.x.
#
# For BinarySpikeClassifier (binary:logistic), monotone constraints are valid:
# a +1 constraint forces P(spike) to increase with the feature, which is
# physically correct for CPU/spike-history features.
_BINARY_MONOTONE_MAP: dict[str, int] = {
    # higher normalised CPU → more likely to spike
    "cpu_vs_p95":            +1,
    "peak_cpu_vs_p95":       +1,
    # p99-level features (Fix 5b)
    "spike_severe_now":        +1,   # current severe spike → imminent future spike
    "cpu_vs_p99":              +1,   # higher normalized CPU vs p99 → more likely spike
    "peak_cpu_vs_p99":         +1,   # higher peak vs p99 → more likely spike
    "band_position":           +1,   # higher in moderate band → more likely to become severe
    "spike_severe_in_last_1":  +1,   # recent severe history → more likely
    "spike_severe_in_last_3":  +1,
    "spike_severe_in_last_6":  +1,
    # band_width: 0 (no constraint — narrow band could mean anything)
    # currently spiking / recent spike history → more likely
    "spike_now":             +1,
    "spike_in_last_1":       +1,
    "spike_in_last_3":       +1,
    "spike_in_last_6":       +1,
    "cpu_spike_rate_24":     +1,
    # longer time since last spike → less imminent
    "time_since_last_spike": -1,
}

# Positional string format "(+1,0,-1,...)" aligned with _X_COLS order.
# String is the XGBoost-declared contract for sklearn API (Union[Dict[str,int], str]).
# Dict format would fail when training on numpy arrays (auto-names "f0","f1",... don't
# match the feature-name keys) — string format is positional and array-safe.
_BINARY_MONOTONE: tuple = tuple(_BINARY_MONOTONE_MAP.get(col, 0) for col in _X_COLS)
_BINARY_MONOTONE_STR: str = "(" + ",".join(str(v) for v in _BINARY_MONOTONE) + ")"

# ── Device helpers ────────────────────────────────────────────────────────────


def _resolve_device(device: str) -> str:
    """Validate the requested compute device and fall back gracefully.

    Accepts ``"cpu"`` or ``"cuda"``.  If ``"cuda"`` is requested but XGBoost
    cannot find a CUDA-capable GPU, logs a warning and returns ``"cpu"`` so
    the pipeline continues without crashing.

    Returns
    -------
    ``"cpu"`` or ``"cuda"``
    """
    device = device.lower()
    if device not in ("cpu", "cuda"):
        raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
    if device == "cuda":
        try:
            # Attempt a tiny DMatrix operation — raises if no CUDA device found
            import xgboost as _xgb
            _xgb.train(
                {"tree_method": "hist", "device": "cuda", "verbosity": 0},
                _xgb.DMatrix([[0]], label=[0]),
                num_boost_round=1,
            )
        except Exception:
            log.warning(
                "device='cuda' requested but no CUDA GPU found — falling back to CPU"
            )
            return "cpu"
    return device


# ── SpikeClassifier ───────────────────────────────────────────────────────────


class SpikeClassifier:
    """XGBoost 3-class severity classifier for 60-minute CPU spike prediction.

    Predicts one of three severity classes per row:
      0 = no_spike   — no p95 exceedance in the next 60 minutes
      1 = moderate   — p95 exceeded but p99 not exceeded
      2 = severe     — p99 exceeded in the next 60 minutes

    Uses multi:softprob objective which outputs a probability vector per row.
    Class balancing is applied via sample_weight in fit() rather than
    scale_pos_weight (which is binary-only).

    Wraps XGBClassifier with explicit feature column management and
    save/load logic that preserves feature names, which XGBoost's native
    save_model / load_model do not preserve automatically.
    """

    def __init__(
        self,
        n_estimators:    int   = 300,
        max_depth:       int   = 6,
        learning_rate:   float = 0.05,
        subsample:       float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int  = 1,    # 1–3 recommended for rare-event detection
        max_delta_step:  int   = 1,    # caps leaf weight updates — prevents gradient
                                       # explosion when sample_weight amplifies gradients
                                       # on imbalanced data (XGBoost issue #4204)
        gamma:           float = 0.0,  # min loss reduction required to split a node
        reg_alpha:       float = 0.0,  # L1 regularisation on leaf weights
        reg_lambda:      float = 1.0,  # L2 regularisation on leaf weights
        device:          str   = "cpu",  # "cpu" or "cuda" — XGBoost 2.x device param
        random_state:    int   = 42,
    ) -> None:
        device = _resolve_device(device)
        self._feature_cols: list[str] = list(_X_COLS)

        self._model = xgb.XGBClassifier(
            objective        = "multi:softprob",
            num_class        = _NUM_CLASSES,
            eval_metric      = "mlogloss",
            tree_method      = "hist",   # unified histogram method for CPU and GPU (XGBoost 2.x)
            device           = device,
            n_estimators     = n_estimators,
            max_depth        = max_depth,
            learning_rate    = learning_rate,
            subsample        = subsample,
            colsample_bytree = colsample_bytree,
            min_child_weight = min_child_weight,
            max_delta_step   = max_delta_step,
            gamma            = gamma,
            reg_alpha        = reg_alpha,
            reg_lambda       = reg_lambda,
            # monotone_constraints: intentionally absent — see module-level NOTE
            random_state     = random_state,
            n_jobs           = 1 if device == "cuda" else -1,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _to_array(self, df: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix as float32 (consistent dtype for XGBoost)."""
        return df[self._feature_cols].astype("float32").values

    # ── Public interface ──────────────────────────────────────────────────────

    def fit(
        self,
        X:             pd.DataFrame,
        y:             pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> "SpikeClassifier":
        """Fit on training data.

        Parameters
        ----------
        X             : DataFrame containing at least the columns in _X_COLS.
        y             : integer Series {0, 1, 2} — severity labels, NaN-free.
        sample_weight : optional per-sample weights for class balancing.
                        Use compute_sample_weight('balanced', y) to balance
                        the 3 severity classes by inverse frequency.

        Returns
        -------
        self
        """
        self._model.fit(self._to_array(X), y.values, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probability matrix for each row.

        Returns
        -------
        2-D float64 array of shape (n, 3).
        Column order: [P(no_spike), P(moderate), P(severe)].
        Rows sum to 1.0 (softmax normalisation).
        """
        return self._model.predict_proba(self._to_array(X))  # shape (n, 3)

    def find_alarm_threshold(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Find the p_severe threshold that maximises F1 for severe class detection.

        Sweeps p_severe (column 2 of predict_proba) from 0.05 to 0.95 in steps
        of 0.05.  Returns the threshold with the highest F1 on the binary
        question "is this row severe?" (class 2 vs. all others).

        Always call this on **validation data only** — using test labels to
        pick the threshold leaks information into the evaluation.

        Parameters
        ----------
        X : DataFrame with _X_COLS columns.
        y : integer Series {0, 1, 2} — severity labels.

        Returns
        -------
        float alarm threshold in (0.0, 1.0).
        """
        p_severe  = self.predict_proba(X)[:, 2]
        y_severe  = (y.values == 2).astype(int)

        best_thresh = 0.5
        best_f1     = 0.0

        for thresh in np.arange(0.05, 1.0, 0.05):
            pred = (p_severe >= thresh).astype(int)
            f1   = float(f1_score(y_severe, pred, zero_division=0))
            if f1 > best_f1:
                best_f1     = f1
                best_thresh = float(thresh)

        return best_thresh

    def evaluate(
        self,
        X:               pd.DataFrame,
        y:               pd.Series,
        alarm_threshold: float = 0.5,
    ) -> dict[str, float]:
        """Evaluate on a held-out set.

        Primary metric is macro PR-AUC — equal weight to all three severity
        classes regardless of prevalence.  Per-class PR-AUC is reported
        separately for diagnostic purposes.

        If the test set contains fewer than 2 classes, AUC metrics are
        returned as NaN rather than raising.

        Parameters
        ----------
        X               : DataFrame with _X_COLS columns.
        y               : integer Series {0, 1, 2} — severity labels.
        alarm_threshold : minimum p_severe to trigger a severe alarm (default 0.5).

        Returns
        -------
        dict with keys: macro_pr_auc, pr_auc_class_0/1/2, macro_roc_auc,
        weighted_f1, macro_f1, alarm_threshold, alarm_precision, alarm_recall.
        """
        probas    = self.predict_proba(X)          # (n, 3)
        pred      = np.argmax(probas, axis=1)
        y_vals    = y.values
        n_present = len(np.unique(y_vals))

        if n_present < 2:
            macro_pr_auc  = float("nan")
            per_class_pr  = [float("nan")] * _NUM_CLASSES
            macro_roc_auc = float("nan")
        else:
            macro_pr_auc  = float(average_precision_score(y_vals, probas, average="macro"))
            per_class_pr  = list(
                average_precision_score(y_vals, probas, average=None).tolist()
            )
            macro_roc_auc = float(
                roc_auc_score(y_vals, probas, multi_class="ovr", average="macro")
            )

        alarm_pred       = (probas[:, 2] >= alarm_threshold).astype(int)
        y_severe         = (y_vals == 2).astype(int)

        return {
            "macro_pr_auc":   macro_pr_auc,
            "pr_auc_class_0": per_class_pr[0],
            "pr_auc_class_1": per_class_pr[1],
            "pr_auc_class_2": per_class_pr[2],
            "macro_roc_auc":  macro_roc_auc,
            "weighted_f1":    float(f1_score(y_vals, pred, average="weighted", zero_division=0)),
            "macro_f1":       float(f1_score(y_vals, pred, average="macro", zero_division=0)),
            "alarm_threshold":  float(alarm_threshold),
            "alarm_precision":  float(precision_score(y_severe, alarm_pred, zero_division=0)),
            "alarm_recall":     float(recall_score(y_severe, alarm_pred, zero_division=0)),
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


# ── BinarySpikeClassifier ─────────────────────────────────────────────────────


class BinarySpikeClassifier:
    """XGBoost binary classifier for short-horizon CPU spike prediction.

    Predicts the binary question "will there be a p95 exceedance within the
    next N minutes?" where N is one of 15, 30, or 45 minutes.

    Uses binary:logistic objective (outputs a single spike probability) with
    monotone constraints re-enabled — unlike the 3-class SpikeClassifier where
    constraints are invalid.  Monotone constraints enforce that CPU/spike
    features never lower predicted spike probability, making predictions more
    interpretable and preventing adversarial feature interactions.

    Class balancing is applied via scale_pos_weight (scalar ratio) rather
    than sample_weight — simpler for binary and achieves the same effect.

    Wraps XGBClassifier with the same save/load pattern as SpikeClassifier.
    """

    def __init__(
        self,
        n_estimators:     int   = 300,
        max_depth:        int   = 6,
        learning_rate:    float = 0.05,
        subsample:        float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int   = 1,
        max_delta_step:   int   = 1,
        gamma:            float = 0.0,
        reg_alpha:        float = 0.0,
        reg_lambda:       float = 1.0,
        scale_pos_weight: float = 1.0,  # n_negative / n_positive for class balance
        device:           str   = "cpu",
        random_state:     int   = 42,
    ) -> None:
        device = _resolve_device(device)
        self._feature_cols: list[str] = list(_X_COLS)

        self._model = xgb.XGBClassifier(
            objective             = "binary:logistic",
            eval_metric           = "aucpr",
            tree_method           = "hist",
            device                = device,
            n_estimators          = n_estimators,
            max_depth             = max_depth,
            learning_rate         = learning_rate,
            subsample             = subsample,
            colsample_bytree      = colsample_bytree,
            min_child_weight      = min_child_weight,
            max_delta_step        = max_delta_step,
            gamma                 = gamma,
            reg_alpha             = reg_alpha,
            reg_lambda            = reg_lambda,
            scale_pos_weight      = scale_pos_weight,
            monotone_constraints  = _BINARY_MONOTONE_STR,
            random_state          = random_state,
            n_jobs                = 1 if device == "cuda" else -1,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _to_array(self, df: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix as float32 (consistent dtype for XGBoost)."""
        return df[self._feature_cols].astype("float32").values

    # ── Public interface ──────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BinarySpikeClassifier":
        """Fit on training data.

        Parameters
        ----------
        X : DataFrame containing at least the columns in _X_COLS.
        y : binary Series {0, 1} — spike labels, NaN-free.

        Returns
        -------
        self
        """
        self._model.fit(self._to_array(X), y.values)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probability matrix for each row.

        Returns
        -------
        2-D float64 array of shape (n, 2).
        Column order: [P(no_spike), P(spike)].
        Rows sum to 1.0.
        """
        return self._model.predict_proba(self._to_array(X))  # shape (n, 2)

    def find_alarm_threshold(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Find the p_spike threshold that maximises F1 for spike detection.

        Sweeps P(spike) (column 1 of predict_proba) from 0.05 to 0.95 in steps
        of 0.05.  Returns the threshold with the highest F1 on the binary
        "is this row a spike?" question.

        Always call this on **validation data only**.

        Parameters
        ----------
        X : DataFrame with _X_COLS columns.
        y : binary Series {0, 1} — spike labels.

        Returns
        -------
        float alarm threshold in (0.0, 1.0).
        """
        p_spike = self.predict_proba(X)[:, 1]
        y_vals  = y.values

        best_thresh = 0.5
        best_f1     = 0.0

        for thresh in np.arange(0.05, 1.0, 0.05):
            pred = (p_spike >= thresh).astype(int)
            f1   = float(f1_score(y_vals, pred, zero_division=0))
            if f1 > best_f1:
                best_f1     = f1
                best_thresh = float(thresh)

        return best_thresh

    def evaluate(
        self,
        X:               pd.DataFrame,
        y:               pd.Series,
        alarm_threshold: float = 0.5,
    ) -> dict[str, float]:
        """Evaluate on a held-out set.

        Primary metric is PR-AUC (area under the precision-recall curve),
        which is informative for imbalanced binary classification.

        Parameters
        ----------
        X               : DataFrame with _X_COLS columns.
        y               : binary Series {0, 1} — spike labels.
        alarm_threshold : minimum P(spike) to trigger an alarm (default 0.5).

        Returns
        -------
        dict with keys: pr_auc, roc_auc, f1, precision, recall, alarm_threshold.
        """
        probas  = self.predict_proba(X)    # (n, 2)
        p_spike = probas[:, 1]
        pred    = (p_spike >= alarm_threshold).astype(int)
        y_vals  = y.values

        if len(np.unique(y_vals)) < 2:
            pr_auc  = float("nan")
            roc_auc = float("nan")
        else:
            pr_auc  = float(average_precision_score(y_vals, p_spike))
            roc_auc = float(roc_auc_score(y_vals, p_spike))

        return {
            "pr_auc":          pr_auc,
            "roc_auc":         roc_auc,
            "f1":              float(f1_score(y_vals, pred, zero_division=0)),
            "precision":       float(precision_score(y_vals, pred, zero_division=0)),
            "recall":          float(recall_score(y_vals, pred, zero_division=0)),
            "alarm_threshold": float(alarm_threshold),
        }

    def save(self, path: str | Path) -> None:
        """Save XGBoost model and feature-column metadata to disk.

        Two files are written:
          <path>           — XGBoost native JSON model
          <path>.meta.json — feature column order + XGBoost version

        Parameters
        ----------
        path : destination path for the model file.
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
    def load(cls, path: str | Path) -> "BinarySpikeClassifier":
        """Load a BinarySpikeClassifier from disk.

        Parameters
        ----------
        path : path to the model file.

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
                "Re-save the model with BinarySpikeClassifier.save()."
            )

        meta              = json.loads(meta_path.read_text())
        obj               = cls()
        obj._model.load_model(str(path))
        obj._feature_cols = meta["feature_cols"]
        return obj


# ── Public entry point ────────────────────────────────────────────────────────


def train(
    features_path: str | Path  = "data/cluster_features.parquet",
    model_path:    str | Path  = "data/spike_model.json",
    train_ratio:   float       = _TRAIN_RATIO,
    val_ratio:     float       = _VAL_RATIO,
    device:        str         = "cpu",
    model_kwargs:  dict | None = None,
) -> dict:
    """Train SpikeClassifier and save the model.

    Time-based three-way split: train / val / test (default 60 % / 20 % / 20 %).

    - **train** : model is fitted here; sample_weight computed here.
    - **val**   : ``find_alarm_threshold()`` is called here — the only place
                  alarm threshold selection touches data.  Never the test set.
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
    val_ratio     : fraction of buckets used for validation / alarm threshold
                    selection (default 0.2).  Remaining buckets → test set.
    device        : compute device — ``"cpu"`` or ``"cuda"`` (default ``"cpu"``).
    model_kwargs  : optional hyperparameter overrides for SpikeClassifier.

    Returns
    -------
    dict with keys covering split sizes, class rates, threshold-independent
    AUC metrics, and alarm-threshold calibrated metrics.

    Raises
    ------
    FileNotFoundError
        If features_path does not exist.
    ValueError
        If fewer than _MIN_POSITIVE spike rows exist in the training split.
    """
    from sklearn.utils.class_weight import compute_sample_weight

    features_path = Path(features_path)
    model_path    = Path(model_path)

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    log.info("═" * 62)
    log.info("  SPIKE CLASSIFIER — training (3-class severity)")
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
    df = df[df["severity_in_60m"].notna()].copy()
    df["severity_in_60m"] = df["severity_in_60m"].astype("int8")

    # Time-based three-way split — never random to prevent future-data leakage.
    bucket_max = int(df["bucket"].max())
    train_max  = int(bucket_max * train_ratio)
    val_max    = int(bucket_max * (train_ratio + val_ratio))

    train_df = df[df["bucket"] <= train_max].reset_index(drop=True)
    val_df   = df[(df["bucket"] > train_max) & (df["bucket"] <= val_max)].reset_index(drop=True)
    test_df  = df[df["bucket"] > val_max].reset_index(drop=True)

    y_train = train_df["severity_in_60m"]
    y_val   = val_df["severity_in_60m"]
    y_test  = test_df["severity_in_60m"]

    # Validate minimum spikes in training (class 1 or 2)
    n_spikes = int((y_train > 0).sum())
    if n_spikes < _MIN_POSITIVE:
        raise ValueError(
            f"Only {n_spikes} spike (class 1 or 2) example(s) in training data "
            f"(minimum required: {_MIN_POSITIVE}). "
            "Check that spike thresholds are not too high or train_ratio not too low."
        )

    def _class_rates(y: pd.Series) -> dict[str, float]:
        n = max(len(y), 1)
        return {
            "no_spike": float((y == 0).sum() / n),
            "moderate": float((y == 1).sum() / n),
            "severe":   float((y == 2).sum() / n),
        }

    log.info("  Train rows : %s  (moderate %.1f%%  severe %.1f%%)",
             f"{len(train_df):,}",
             100 * float((y_train == 1).mean()),
             100 * float((y_train == 2).mean()))
    log.info("  Val rows   : %s  (moderate %.1f%%  severe %.1f%%)",
             f"{len(val_df):,}",
             100 * float((y_val == 1).mean()),
             100 * float((y_val == 2).mean()))
    log.info("  Test rows  : %s  (moderate %.1f%%  severe %.1f%%)",
             f"{len(test_df):,}",
             100 * float((y_test == 1).mean()),
             100 * float((y_test == 2).mean()))

    # Per-sample class balancing — training split only
    sw_train = compute_sample_weight("balanced", y_train)

    t_start = time.perf_counter()
    clf = SpikeClassifier(device=device, **(model_kwargs or {}))
    clf.fit(train_df, y_train, sample_weight=sw_train)
    elapsed = time.perf_counter() - t_start
    log.info("  Training time : %.1f s", elapsed)

    # Find alarm threshold on the validation set only
    alarm_thresh = clf.find_alarm_threshold(val_df, y_val)

    m = clf.evaluate(test_df, y_test, alarm_threshold=alarm_thresh)

    log.info("  ─────────────────────────────────────────────────────")
    log.info("  Macro PR-AUC   : %.3f  ← primary metric", m["macro_pr_auc"])
    log.info("  Macro ROC-AUC  : %.3f", m["macro_roc_auc"])
    log.info("  Per-class PR-AUC : no_spike=%.3f  moderate=%.3f  severe=%.3f",
             m["pr_auc_class_0"], m["pr_auc_class_1"], m["pr_auc_class_2"])
    log.info("  Alarm threshold  : %.2f  (p_severe)  →  P=%.3f  R=%.3f",
             alarm_thresh, m["alarm_precision"], m["alarm_recall"])
    log.info("═" * 62)

    clf.save(model_path)

    return {
        "train_rows":       len(train_df),
        "val_rows":         len(val_df),
        "test_rows":        len(test_df),
        "train_bucket_max": train_max,
        "val_bucket_max":   val_max,
        "class_rates_train": _class_rates(y_train),
        "class_rates_val":   _class_rates(y_val),
        "class_rates_test":  _class_rates(y_test),
        # primary metric (threshold-independent)
        "macro_pr_auc":     m["macro_pr_auc"],
        "macro_roc_auc":    m["macro_roc_auc"],
        "pr_auc_class_0":   m["pr_auc_class_0"],
        "pr_auc_class_1":   m["pr_auc_class_1"],
        "pr_auc_class_2":   m["pr_auc_class_2"],
        # alarm threshold calibrated on val data
        "alarm_threshold":  alarm_thresh,
        "alarm_precision":  m["alarm_precision"],
        "alarm_recall":     m["alarm_recall"],
        "weighted_f1":      m["weighted_f1"],
        "macro_f1":         m["macro_f1"],
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
    p.add_argument(
        "--device",
        default = "cpu",
        choices = ["cpu", "cuda"],
        help    = (
            "Compute device for XGBoost  (default: cpu). "
            "Use 'cuda' on a machine with a CUDA-capable GPU — falls back to "
            "CPU automatically if no GPU is found.  The saved model is "
            "device-agnostic and loads on any machine."
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
        device        = args.device,
    )
