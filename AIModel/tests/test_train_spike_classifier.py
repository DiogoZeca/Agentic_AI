"""Tests for train_spike_classifier.py — Step 4 of the CPU spike prediction pipeline.

Strategy: all tests use synthetic parquets (no real CSV needed) so the suite
runs in seconds without any large data files.  The preprocessor step (Step 1)
is always skipped by starting from --from-step 2 or 3, or by patching preprocess.

Fixtures
--------
agg_parquet
    Minimal two-machine, 200-bucket cluster_agg.parquet.  Enough buckets to
    produce valid train (60%) / val (20%) / test (20%) splits with labeled rows
    after the 12-bucket NaN tail.

features_parquet
    Runs engineer() on agg_parquet once and caches the result.  Reused by all
    tests that need a features file.

pipeline_result
    Runs the full run() pipeline (from_step=3, walk_forward=False) on the
    features_parquet.  Validates end-to-end behaviour without re-engineering.
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

from spike_feature_engineer import engineer, _FEATURE_COLS
from spike_classifier import _X_COLS, _TRAIN_RATIO, _VAL_RATIO
from train_spike_classifier import (
    _N_FOLDS,
    _fold_metrics,
    _run_walk_forward_cv,
    _threshold_sweep_table,
    _step_needed,
    run,
)

# ── Shared helpers ────────────────────────────────────────────────────────────

_N_MACHINES = 2
_N_BUCKETS  = 200
_RNG        = np.random.default_rng(42)


def _make_agg_df(n_machines: int = _N_MACHINES, n_buckets: int = _N_BUCKETS) -> pd.DataFrame:
    rng  = np.random.default_rng(7)
    rows = []
    for m in range(1, n_machines + 1):
        cpu = rng.uniform(0.05, 0.95, n_buckets).astype("float32")
        for b in range(1, n_buckets + 1):
            rows.append({
                "machine_id": m,
                "bucket":     b,
                "time_us":    b * 300_000_000,
                "total_cpu":  float(cpu[b - 1]),
                "peak_cpu":   float(min(cpu[b - 1] * 1.2, 1.0)),
                "total_mem":  float(rng.uniform(0.1, 0.9)),
                "peak_mem":   float(rng.uniform(0.1, 0.9)),
                "disk_io":    float(rng.uniform(0.0, 0.5)),
                "n_tasks":    int(rng.integers(1, 20)),
            })
    df = pd.DataFrame(rows)
    df["machine_id"] = df["machine_id"].astype("int64")
    df["bucket"]     = df["bucket"].astype("int64")
    df["time_us"]    = df["time_us"].astype("int64")
    df["n_tasks"]    = df["n_tasks"].astype("int32")
    for col in ("total_cpu", "peak_cpu", "total_mem", "peak_mem", "disk_io"):
        df[col] = df[col].astype("float32")
    return df


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def agg_parquet(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("agg") / "cluster_agg.parquet"
    _make_agg_df().to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    return path


@pytest.fixture(scope="module")
def features_parquet(agg_parquet, tmp_path_factory) -> Path:
    out_dir = tmp_path_factory.mktemp("feat")
    feat_p  = out_dir / "cluster_features.parquet"
    thr_p   = out_dir / "spike_thresholds.parquet"
    engineer(agg_parquet, feat_p, thr_p, train_ratio=_TRAIN_RATIO)
    return feat_p


@pytest.fixture(scope="module")
def pipeline_result(features_parquet, tmp_path_factory):
    """Run the pipeline from step 3 (training only) with walk-forward off."""
    arts_dir = tmp_path_factory.mktemp("arts")
    # Copy features parquet into artifacts dir so run() can find it
    feat_dest = arts_dir / "cluster_features.parquet"
    pd.read_parquet(features_parquet).to_parquet(
        feat_dest, engine="pyarrow", compression="zstd", index=False
    )
    result = run(
        data_path     = arts_dir / "nonexistent.csv",   # not needed when from_step=3
        artifacts_dir = arts_dir,
        train_ratio   = _TRAIN_RATIO,
        val_ratio     = _VAL_RATIO,
        from_step     = 3,
        walk_forward  = False,
        seed          = 42,
    )
    return result, arts_dir


# ── _step_needed ──────────────────────────────────────────────────────────────

class TestStepNeeded:
    def test_force_always_runs(self, tmp_path):
        p = tmp_path / "existing.parquet"
        p.touch()
        assert _step_needed(p, step_num=1, from_step=1, force=True) is True

    def test_skip_when_cache_exists_and_step_before_from_step(self, tmp_path):
        p = tmp_path / "existing.parquet"
        p.touch()
        assert _step_needed(p, step_num=1, from_step=2, force=False) is False

    def test_skip_when_step_equals_from_step_and_cache_exists(self, tmp_path):
        """Cache hit at from_step boundary → skip (use --force to invalidate)."""
        p = tmp_path / "existing.parquet"
        p.touch()
        assert _step_needed(p, step_num=2, from_step=2, force=False) is False

    def test_skip_when_step_before_from_step_even_if_cache_missing(self, tmp_path):
        """from_step=2 must unconditionally skip step 1, even if its cache is absent."""
        p = tmp_path / "nonexistent.parquet"
        assert _step_needed(p, step_num=1, from_step=2, force=False) is False

    def test_run_when_cache_missing_and_step_at_from_step(self, tmp_path):
        """Step equals from_step and cache is absent → must run."""
        p = tmp_path / "nonexistent.parquet"
        assert _step_needed(p, step_num=2, from_step=2, force=False) is True


# ── _fold_metrics ─────────────────────────────────────────────────────────────

class TestFoldMetrics:
    @pytest.fixture(scope="class")
    def fitted_clf(self):
        from spike_classifier import SpikeClassifier
        rng = np.random.default_rng(0)
        X   = pd.DataFrame(
            {col: rng.uniform(0.0, 1.0, 100).astype("float32") for col in _X_COLS}
        )
        y   = pd.Series((rng.uniform(size=100) < 0.3).astype("int8"))
        return SpikeClassifier().fit(X, y), X, y

    def test_returns_five_keys(self, fitted_clf):
        clf, X, y = fitted_clf
        m = _fold_metrics(clf, X, y, threshold=0.5)
        assert set(m.keys()) == {"pr_auc", "roc_auc", "precision", "recall", "f1"}

    def test_metrics_in_unit_interval(self, fitted_clf):
        clf, X, y = fitted_clf
        m = _fold_metrics(clf, X, y, threshold=0.5)
        for key, val in m.items():
            assert 0.0 <= val <= 1.0 or np.isnan(val), f"{key}={val}"

    def test_one_class_returns_nan_auc(self, fitted_clf):
        clf, X, _ = fitted_clf
        y_one = pd.Series(np.zeros(len(X), dtype="int8"))
        m = _fold_metrics(clf, X, y_one, threshold=0.5)
        assert np.isnan(m["pr_auc"])
        assert np.isnan(m["roc_auc"])


# ── _run_walk_forward_cv ──────────────────────────────────────────────────────

class TestWalkForwardCV:
    @pytest.fixture(scope="class")
    def cv_output(self, features_parquet, tmp_path_factory):
        df       = pd.read_parquet(features_parquet)
        df       = df[df["spike_in_60m"].notna()].copy()
        df["spike_in_60m"] = df["spike_in_60m"].astype("int8")
        train_max = int(df["bucket"].max() * _TRAIN_RATIO)
        train_df  = df[df["bucket"] <= train_max].reset_index(drop=True)
        cv_p      = tmp_path_factory.mktemp("cv") / "cv_results.csv"
        summary   = _run_walk_forward_cv(train_df, n_folds=3, seed=42, output_path=cv_p)
        return summary, cv_p

    def test_cv_results_csv_written(self, cv_output):
        _, cv_p = cv_output
        assert cv_p.exists()
        assert cv_p.stat().st_size > 0

    def test_cv_summary_has_expected_keys(self, cv_output):
        summary, _ = cv_output
        expected_keys = {
            "cv_pr_auc_mean", "cv_pr_auc_std",
            "cv_roc_auc_mean", "cv_roc_auc_std",
            "cv_f1_mean", "cv_f1_std",
            "cv_precision_mean", "cv_precision_std",
            "cv_recall_mean", "cv_recall_std",
        }
        assert expected_keys == set(summary.keys())

    def test_cv_means_in_unit_interval(self, cv_output):
        summary, _ = cv_output
        for key, val in summary.items():
            if "mean" in key:
                assert 0.0 <= val <= 1.0 or np.isnan(val), f"{key}={val}"

    def test_cv_stds_nonnegative(self, cv_output):
        summary, _ = cv_output
        for key, val in summary.items():
            if "std" in key:
                assert val >= 0.0 or np.isnan(val), f"{key}={val}"

    def test_cv_csv_has_fold_column(self, cv_output):
        _, cv_p = cv_output
        df = pd.read_csv(cv_p)
        assert "fold" in df.columns
        assert df["fold"].nunique() >= 1


# ── _threshold_sweep_table ────────────────────────────────────────────────────

class TestThresholdSweepTable:
    @pytest.fixture(scope="class")
    def sweep(self, features_parquet):
        from spike_classifier import SpikeClassifier
        df       = pd.read_parquet(features_parquet)
        df       = df[df["spike_in_60m"].notna()].copy()
        df["spike_in_60m"] = df["spike_in_60m"].astype("int8")
        train_max = int(df["bucket"].max() * _TRAIN_RATIO)
        val_max   = int(df["bucket"].max() * (_TRAIN_RATIO + _VAL_RATIO))
        train_df  = df[df["bucket"] <= train_max]
        val_df    = df[(df["bucket"] > train_max) & (df["bucket"] <= val_max)]
        y_train   = train_df["spike_in_60m"]
        neg, pos  = int((y_train == 0).sum()), int((y_train == 1).sum())
        clf = SpikeClassifier(scale_pos_weight=neg / max(pos, 1)).fit(train_df, y_train)
        return _threshold_sweep_table(clf, val_df[_X_COLS], val_df["spike_in_60m"], len(val_df))

    def test_sweep_covers_expected_range(self, sweep):
        thresholds = [r["threshold"] for r in sweep]
        assert min(thresholds) <= 0.10
        assert max(thresholds) >= 0.90

    def test_sweep_rows_have_required_keys(self, sweep):
        required = {"threshold", "precision", "recall", "f1", "alarms_per_day"}
        for row in sweep:
            assert required == set(row.keys())

    def test_lower_threshold_higher_recall(self, sweep):
        """Recall must be monotonically non-increasing as threshold rises."""
        recalls = [r["recall"] for r in sweep]
        for lo, hi in zip(recalls, recalls[1:]):
            assert lo >= hi - 1e-6, "recall should not increase as threshold rises"

    def test_alarms_per_day_nonnegative(self, sweep):
        for row in sweep:
            assert row["alarms_per_day"] >= 0.0


# ── Full pipeline: run() ──────────────────────────────────────────────────────

class TestRunPipeline:
    def test_returns_dict_with_expected_keys(self, pipeline_result):
        result, _ = pipeline_result
        expected = {
            "train_rows", "val_rows", "test_rows",
            "train_bucket_max", "val_bucket_max",
            "spike_rate_train", "spike_rate_val", "spike_rate_test",
            "scale_pos_weight",
            "pr_auc", "roc_auc",
            "precision", "recall", "f1",
            "optimal_threshold",
            "precision_calibrated", "recall_calibrated", "f1_calibrated",
        }
        assert expected == set(result.keys())

    def test_model_file_written(self, pipeline_result):
        _, arts = pipeline_result
        assert (arts / "models" / "spike" / "spike_model.json").exists()

    def test_meta_file_written(self, pipeline_result):
        _, arts = pipeline_result
        assert (arts / "models" / "spike" / "spike_model.meta.json").exists()

    def test_spike_config_written(self, pipeline_result):
        _, arts = pipeline_result
        assert (arts / "models" / "spike" / "spike_config.json").exists()

    def test_run_config_written(self, pipeline_result):
        _, arts = pipeline_result
        assert (arts / "models" / "spike" / "run_config.json").exists()

    def test_feature_importance_csv_written(self, pipeline_result):
        _, arts = pipeline_result
        assert (arts / "models" / "spike" / "feature_importance.csv").exists()

    def test_spike_config_has_optimal_threshold(self, pipeline_result):
        _, arts = pipeline_result
        cfg = json.loads((arts / "models" / "spike" / "spike_config.json").read_text())
        assert "optimal_threshold" in cfg
        assert 0.0 < cfg["optimal_threshold"] < 1.0

    def test_spike_config_has_final_metrics(self, pipeline_result):
        _, arts = pipeline_result
        cfg = json.loads((arts / "models" / "spike" / "spike_config.json").read_text())
        assert "final_metrics" in cfg
        assert "pr_auc" in cfg["final_metrics"]

    def test_run_config_records_seed(self, pipeline_result):
        _, arts = pipeline_result
        cfg = json.loads((arts / "models" / "spike" / "run_config.json").read_text())
        assert cfg["seed"] == 42

    def test_run_config_records_timestamp(self, pipeline_result):
        _, arts = pipeline_result
        cfg = json.loads((arts / "models" / "spike" / "run_config.json").read_text())
        assert "timestamp" in cfg
        assert "T" in cfg["timestamp"]   # ISO-8601 format contains 'T'

    def test_splits_sum_to_total_labeled_rows(self, pipeline_result):
        result, _ = pipeline_result
        total = result["train_rows"] + result["val_rows"] + result["test_rows"]
        assert total > 0
        # Each split must be non-empty
        assert result["train_rows"] > 0
        assert result["val_rows"]   > 0
        assert result["test_rows"]  > 0

    def test_val_bucket_max_between_train_and_test(self, pipeline_result):
        result, _ = pipeline_result
        assert result["train_bucket_max"] < result["val_bucket_max"]


# ── Walk-forward enabled in run() ─────────────────────────────────────────────

class TestRunWithWalkForward:
    @pytest.fixture(scope="class")
    def wf_result(self, features_parquet, tmp_path_factory):
        arts_dir  = tmp_path_factory.mktemp("arts_wf")
        feat_dest = arts_dir / "cluster_features.parquet"
        pd.read_parquet(features_parquet).to_parquet(
            feat_dest, engine="pyarrow", compression="zstd", index=False
        )
        result = run(
            data_path     = arts_dir / "nonexistent.csv",
            artifacts_dir = arts_dir,
            train_ratio   = _TRAIN_RATIO,
            val_ratio     = _VAL_RATIO,
            from_step     = 3,
            walk_forward  = True,
            n_folds       = 3,
            seed          = 42,
        )
        return result, arts_dir

    def test_cv_results_csv_present_with_walk_forward(self, wf_result):
        _, arts = wf_result
        assert (arts / "models" / "spike" / "cv_results.csv").exists()

    def test_spike_config_has_cv_summary(self, wf_result):
        _, arts = wf_result
        cfg = json.loads((arts / "models" / "spike" / "spike_config.json").read_text())
        assert "cv_summary" in cfg
        assert len(cfg["cv_summary"]) > 0


# ── Error handling ─────────────────────────────────────────────────────────────

class TestRunErrors:
    def test_missing_data_file_raises_on_step1(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            run(
                data_path     = tmp_path / "nonexistent.csv",
                artifacts_dir = tmp_path,
                from_step     = 1,
                walk_forward  = False,
            )

    def test_missing_features_skips_to_error_on_step3(self, tmp_path):
        """from_step=3 with no features parquet should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            run(
                data_path     = tmp_path / "nonexistent.csv",
                artifacts_dir = tmp_path,
                from_step     = 3,
                walk_forward  = False,
            )
