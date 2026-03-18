"""Tests for spike_classifier.py — Step 3 of the CPU spike prediction pipeline.

Fixtures
--------
clf_data
    200-row synthetic DataFrame with all _X_COLS columns and a binary
    spike_in_60m label (~30 % positive rate).  Used for unit tests of
    SpikeClassifier directly (no parquet I/O, no time-split logic).

features_parquet
    Two-machine, 200-bucket parquet written to a tmp file, matching the
    schema that spike_feature_engineer.engineer() produces.  Used to test
    train() end-to-end.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike_classifier import (
    _MIN_POSITIVE,
    _TRAIN_RATIO,
    _VAL_RATIO,
    _X_COLS,
    SpikeClassifier,
    train,
)

# ── Synthetic data helpers ────────────────────────────────────────────────────

_N_ROWS    = 200
_N_BUCKETS = 200
_N_MACHINES = 2
_RNG       = np.random.default_rng(42)


def _make_X(n: int = _N_ROWS) -> pd.DataFrame:
    """Synthetic feature matrix — random floats in realistic ranges.

    All columns in _X_COLS are populated, including cpu_vs_p95 which is
    uniformly sampled from [0, 2] to represent values both below and above
    the machine's spike threshold.
    """
    rng  = np.random.default_rng(0)
    data = {col: rng.uniform(0.0, 1.0, n).astype("float32") for col in _X_COLS}
    data["n_tasks"]    = rng.integers(1, 20, n).astype("int32")
    data["cpu_vs_p95"] = rng.uniform(0.0, 2.0, n).astype("float32")
    return pd.DataFrame(data)


def _make_y(n: int = _N_ROWS, positive_rate: float = 0.30) -> pd.Series:
    """Binary labels with ~30 % positive rate."""
    rng = np.random.default_rng(1)
    return pd.Series(
        (rng.uniform(size=n) < positive_rate).astype("int8"),
        name="spike_in_60m",
    )


def _make_features_df(n_machines: int = _N_MACHINES, n_buckets: int = _N_BUCKETS) -> pd.DataFrame:
    """Full features DataFrame matching the schema from spike_feature_engineer."""
    rng   = np.random.default_rng(2)
    rows  = []
    for m in range(1, n_machines + 1):
        cpu = rng.uniform(0.1, 0.9, n_buckets).astype("float32")
        for b in range(1, n_buckets + 1):
            label: float
            if b > n_buckets - 12:
                label = float("nan")    # last 12 buckets per machine → NaN
            else:
                label = float(cpu[b - 1] > 0.6)

            row = {
                "machine_id": m,
                "bucket":     b,
                "time_us":    b * 300_000_000,
                "spike_in_60m": label,
            }
            for col in _X_COLS:
                if col == "total_cpu":
                    row[col] = float(cpu[b - 1])
                elif col == "cpu_vs_p95":
                    # simulate machine-relative feature: values both < 1 and > 1
                    row[col] = float(cpu[b - 1] / 0.6)
                else:
                    row[col] = rng.uniform(0.0, 1.0)
            row["n_tasks"] = int(rng.integers(1, 20))
            rows.append(row)

    df = pd.DataFrame(rows)
    df["machine_id"] = df["machine_id"].astype("int64")
    df["bucket"]     = df["bucket"].astype("int64")
    df["time_us"]    = df["time_us"].astype("int64")
    df["n_tasks"]    = df["n_tasks"].astype("int32")
    for col in _X_COLS:
        if col != "n_tasks":
            df[col] = df[col].astype("float32")
    return df


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def clf_data():
    X = _make_X()
    y = _make_y()
    return X, y


@pytest.fixture(scope="module")
def fitted_clf(clf_data):
    X, y = clf_data
    return SpikeClassifier().fit(X, y)


@pytest.fixture(scope="module")
def features_parquet(tmp_path_factory) -> Path:
    df   = _make_features_df()
    path = tmp_path_factory.mktemp("features") / "cluster_features.parquet"
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    return path


@pytest.fixture(scope="module")
def train_result(features_parquet, tmp_path_factory):
    model_p = tmp_path_factory.mktemp("model") / "spike_model.json"
    result  = train(features_parquet, model_p, train_ratio=_TRAIN_RATIO)
    return result, model_p


# ── SpikeClassifier unit tests ────────────────────────────────────────────────

class TestSpikeClassifierFit:
    def test_fit_returns_self(self, clf_data):
        X, y = clf_data
        clf  = SpikeClassifier()
        assert clf.fit(X, y) is clf

    def test_predict_proba_shape(self, fitted_clf, clf_data):
        X, _ = clf_data
        out  = fitted_clf.predict_proba(X)
        assert out.shape == (len(X),)

    def test_predict_proba_in_unit_interval(self, fitted_clf, clf_data):
        X, _ = clf_data
        out  = fitted_clf.predict_proba(X)
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_machine_id_not_in_feature_cols(self):
        """machine_id must never be passed to XGBoost."""
        assert "machine_id" not in _X_COLS

    def test_bucket_not_in_feature_cols(self):
        """bucket (timestamp proxy) must not be a feature — model must generalise in time."""
        assert "bucket" not in _X_COLS

    def test_time_us_not_in_feature_cols(self):
        assert "time_us" not in _X_COLS


class TestSpikeClassifierEvaluate:
    def test_evaluate_returns_all_keys(self, fitted_clf, clf_data):
        X, y   = clf_data
        result = fitted_clf.evaluate(X, y)
        assert set(result.keys()) == {"threshold", "pr_auc", "roc_auc", "precision", "recall", "f1"}

    def test_evaluate_default_threshold_is_half(self, fitted_clf, clf_data):
        X, y = clf_data
        assert fitted_clf.evaluate(X, y)["threshold"] == pytest.approx(0.5)

    def test_evaluate_respects_custom_threshold(self, fitted_clf, clf_data):
        X, y = clf_data
        assert fitted_clf.evaluate(X, y, threshold=0.3)["threshold"] == pytest.approx(0.3)

    def test_evaluate_metrics_in_unit_interval(self, fitted_clf, clf_data):
        X, y   = clf_data
        result = fitted_clf.evaluate(X, y)
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"{key} = {val} is outside [0, 1]"

    def test_evaluate_all_one_class_returns_nan_auc(self, fitted_clf, clf_data):
        """When test set has only one class, AUC metrics must be NaN, not raise."""
        X, _ = clf_data
        y_one_class = pd.Series(np.zeros(len(X), dtype="int8"))
        result = fitted_clf.evaluate(X, y_one_class)
        assert np.isnan(result["pr_auc"])
        assert np.isnan(result["roc_auc"])

    def test_find_threshold_returns_float_in_range(self, fitted_clf, clf_data):
        X, y   = clf_data
        thresh = fitted_clf.find_threshold(X, y)
        assert isinstance(thresh, float)
        assert 0.0 < thresh < 1.0

    def test_find_threshold_affects_precision_recall_tradeoff(self, fitted_clf, clf_data):
        """A lower threshold should increase recall (fewer missed spikes)."""
        X, y = clf_data
        high_thresh_result = fitted_clf.evaluate(X, y, threshold=0.9)
        low_thresh_result  = fitted_clf.evaluate(X, y, threshold=0.1)
        assert low_thresh_result["recall"] >= high_thresh_result["recall"]


class TestSpikeClassifierSaveLoad:
    def test_save_creates_model_file(self, fitted_clf, tmp_path):
        path = tmp_path / "model.json"
        fitted_clf.save(path)
        assert path.exists()

    def test_save_creates_meta_file(self, fitted_clf, tmp_path):
        path = tmp_path / "model.json"
        fitted_clf.save(path)
        assert path.with_suffix(".meta.json").exists()

    def test_meta_file_contains_feature_cols(self, fitted_clf, tmp_path):
        path = tmp_path / "model.json"
        fitted_clf.save(path)
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        assert meta["feature_cols"] == _X_COLS

    def test_load_restores_predictions(self, fitted_clf, clf_data, tmp_path):
        """Predictions from a loaded model must match the original."""
        X, _ = clf_data
        path = tmp_path / "model.json"
        fitted_clf.save(path)
        loaded = SpikeClassifier.load(path)
        np.testing.assert_allclose(
            fitted_clf.predict_proba(X),
            loaded.predict_proba(X),
            rtol=1e-5,
        )

    def test_load_raises_on_missing_model(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            SpikeClassifier.load(tmp_path / "nonexistent.json")

    def test_load_raises_on_missing_meta(self, fitted_clf, tmp_path):
        path = tmp_path / "model.json"
        fitted_clf.save(path)
        path.with_suffix(".meta.json").unlink()
        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            SpikeClassifier.load(path)


# ── train() integration tests ─────────────────────────────────────────────────

class TestTrainFunction:
    def test_returns_all_expected_keys(self, train_result):
        result, _ = train_result
        expected  = {
            "train_rows", "val_rows", "test_rows",
            "train_bucket_max", "val_bucket_max",
            "spike_rate_train", "spike_rate_val", "spike_rate_test",
            "scale_pos_weight",
            # threshold-independent
            "pr_auc", "roc_auc",
            # at default 0.5 threshold (test set)
            "precision", "recall", "f1",
            # at F1-optimal threshold calibrated on val, reported on test
            "optimal_threshold",
            "precision_calibrated", "recall_calibrated", "f1_calibrated",
        }
        assert expected == set(result.keys())

    def test_optimal_threshold_in_range(self, train_result):
        result, _ = train_result
        assert 0.0 < result["optimal_threshold"] < 1.0

    def test_splits_are_strictly_ordered_by_bucket(self, features_parquet):
        """Verify the 3-way time-based split: train < val < test (no overlap)."""
        df         = pd.read_parquet(features_parquet)
        df         = df[df["spike_in_60m"].notna()]
        bucket_max = int(df["bucket"].max())
        train_max  = int(bucket_max * _TRAIN_RATIO)
        val_max    = int(bucket_max * (_TRAIN_RATIO + _VAL_RATIO))
        train_bkts = df[df["bucket"] <= train_max]["bucket"]
        val_bkts   = df[(df["bucket"] > train_max) & (df["bucket"] <= val_max)]["bucket"]
        test_bkts  = df[df["bucket"] > val_max]["bucket"]
        assert train_bkts.max() <= train_max
        assert val_bkts.min()   >  train_max
        assert val_bkts.max()   <= val_max
        assert test_bkts.min()  >  val_max

    def test_scale_pos_weight_matches_training_data(self, features_parquet, train_result):
        """scale_pos_weight must equal neg/pos of training rows only (not val/test)."""
        result, _ = train_result
        df        = pd.read_parquet(features_parquet)
        df        = df[df["spike_in_60m"].notna()]
        train_max = int(df["bucket"].max() * _TRAIN_RATIO)
        y_train   = df[df["bucket"] <= train_max]["spike_in_60m"]
        neg       = int((y_train == 0).sum())
        pos       = int((y_train == 1).sum())
        assert result["scale_pos_weight"] == pytest.approx(neg / pos, rel=1e-6)

    def test_model_file_written(self, train_result):
        _, model_p = train_result
        assert model_p.exists()

    def test_meta_file_written(self, train_result):
        _, model_p = train_result
        assert model_p.with_suffix(".meta.json").exists()

    def test_missing_features_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            train(tmp_path / "nonexistent.parquet", tmp_path / "model.json")

    def test_too_few_positives_raises(self, tmp_path):
        """Train split with fewer than _MIN_POSITIVE spike rows must raise ValueError."""
        # All labels = 0 → pos = 0
        df        = _make_features_df(n_machines=1, n_buckets=50)
        df["spike_in_60m"] = df["spike_in_60m"].where(
            df["spike_in_60m"].isna(), 0.0
        )
        path = tmp_path / "no_spikes.parquet"
        df.to_parquet(path, engine="pyarrow", index=False)
        with pytest.raises(ValueError, match="positive"):
            train(path, tmp_path / "model.json")
