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
    _apply_calibrators,
    _compute_calibration_metrics,
    _compute_feature_importance,
    _fold_metrics,
    _run_optuna_search,
    _run_walk_forward_cv,
    _select_alarm_threshold,
    _threshold_sweep_table,
    _step_needed,
    _train_binary_horizon,
    run,
)

# ── Shared helpers ────────────────────────────────────────────────────────────

_N_MACHINES = 2
_N_BUCKETS  = 300   # increased from 200 — needs enough rows for 3-class CV folds
_RNG        = np.random.default_rng(42)


def _make_agg_df(n_machines: int = _N_MACHINES, n_buckets: int = _N_BUCKETS) -> pd.DataFrame:
    """Generate synthetic cluster_agg data with all three severity classes.

    Uses a bimodal CPU distribution: most buckets are moderate (0.05–0.85),
    but ~5% are deliberately high (0.90–1.0) to create both moderate and severe
    spikes once the feature engineer applies per-machine p95/p99 thresholds.
    This ensures all three labels (0=no_spike, 1=moderate, 2=severe) are present,
    which is required for the 3-class walk-forward CV and Optuna tests.
    """
    rng  = np.random.default_rng(42)
    rows = []
    for m in range(1, n_machines + 1):
        cpu = rng.uniform(0.05, 0.85, n_buckets).astype("float32")
        # Inject ~5% high-CPU buckets so the p99 threshold is reachable and
        # severe labels (class 2) appear in the engineered feature parquet.
        spike_idx = rng.choice(n_buckets, size=max(1, n_buckets // 20), replace=False)
        cpu[spike_idx] = rng.uniform(0.90, 1.0, len(spike_idx)).astype("float32")
        for b in range(1, n_buckets + 1):
            rows.append({
                "machine_id": m,
                "bucket":     b,
                "time_us":    b * 300_000_000,
                "total_cpu":  float(cpu[b - 1]),
                "peak_cpu":   float(min(cpu[b - 1] * 1.1, 1.0)),
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
    import shutil
    arts_dir = tmp_path_factory.mktemp("arts")
    # Copy features parquet into artifacts dir so run() can find it
    feat_dest = arts_dir / "cluster_features.parquet"
    pd.read_parquet(features_parquet).to_parquet(
        feat_dest, engine="pyarrow", compression="zstd", index=False
    )
    # Copy spike_thresholds.parquet — written by engineer() alongside features.
    # Required by train_spike_classifier.py step 3 to compute global_threshold_fallback.
    shutil.copy(
        features_parquet.parent / "spike_thresholds.parquet",
        arts_dir / "spike_thresholds.parquet",
    )
    result = run(
        data_path             = arts_dir / "nonexistent.csv",   # not needed when from_step=3
        artifacts_dir         = arts_dir,
        train_ratio           = _TRAIN_RATIO,
        val_ratio             = _VAL_RATIO,
        from_step             = 3,
        walk_forward          = False,
        seed                  = 42,
        n_estimators          = 50,
        early_stopping_rounds = 10,
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
        y   = pd.Series(rng.choice([0, 1, 2], size=100, p=[0.70, 0.20, 0.10]).astype("int8"))
        return SpikeClassifier().fit(X, y), X, y

    @pytest.fixture(scope="class")
    def fitted_binary_clf(self):
        from spike_classifier import BinarySpikeClassifier
        rng = np.random.default_rng(0)
        X   = pd.DataFrame(
            {col: rng.uniform(0.0, 1.0, 100).astype("float32") for col in _X_COLS}
        )
        y   = pd.Series(rng.choice([0, 1], size=100, p=[0.80, 0.20]).astype("int8"))
        return BinarySpikeClassifier().fit(X, y), X, y

    def test_returns_expected_keys(self, fitted_clf):
        clf, X, y = fitted_clf
        m = _fold_metrics(clf, X, y, alarm_threshold=0.5)
        assert set(m.keys()) == {"macro_pr_auc", "macro_roc_auc", "weighted_f1", "macro_f1"}

    def test_binary_path_returns_same_keys(self, fitted_binary_clf):
        """binary=True path must return the same metric key names as the 3-class path."""
        clf, X, y = fitted_binary_clf
        m = _fold_metrics(clf, X, y, alarm_threshold=0.5, binary=True)
        assert set(m.keys()) == {"macro_pr_auc", "macro_roc_auc", "weighted_f1", "macro_f1"}

    def test_binary_path_metrics_in_unit_interval(self, fitted_binary_clf):
        clf, X, y = fitted_binary_clf
        m = _fold_metrics(clf, X, y, alarm_threshold=0.5, binary=True)
        for key, val in m.items():
            # 1e-9 tolerance: average_precision_score can return 1.0 + ε on tiny data
            assert 0.0 <= val <= 1.0 + 1e-9 or np.isnan(val), f"{key}={val}"

    def test_metrics_in_unit_interval(self, fitted_clf):
        clf, X, y = fitted_clf
        m = _fold_metrics(clf, X, y, alarm_threshold=0.5)
        for key, val in m.items():
            assert 0.0 <= val <= 1.0 or np.isnan(val), f"{key}={val}"

    def test_one_class_returns_nan_auc(self, fitted_clf):
        clf, X, _ = fitted_clf
        y_one = pd.Series(np.zeros(len(X), dtype="int8"))
        m = _fold_metrics(clf, X, y_one, alarm_threshold=0.5)
        assert np.isnan(m["macro_pr_auc"])
        assert np.isnan(m["macro_roc_auc"])


# ── _run_walk_forward_cv ──────────────────────────────────────────────────────

class TestWalkForwardCV:
    @pytest.fixture(scope="class")
    def cv_output(self, features_parquet, tmp_path_factory):
        df       = pd.read_parquet(features_parquet)
        df       = df[df["severity_in_60m"].notna()].copy()
        df["severity_in_60m"] = df["severity_in_60m"].astype("int8")
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
        required_keys = {
            "cv_macro_pr_auc_mean", "cv_macro_pr_auc_std",
            "cv_macro_roc_auc_mean", "cv_macro_roc_auc_std",
            "cv_weighted_f1_mean", "cv_weighted_f1_std",
            "cv_macro_f1_mean", "cv_macro_f1_std",
        }
        # cv_drift_tau / cv_drift_p_val are only present when >= 3 valid folds
        assert required_keys.issubset(set(summary.keys()))

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
        from sklearn.utils.class_weight import compute_sample_weight
        df       = pd.read_parquet(features_parquet)
        df       = df[df["severity_in_60m"].notna()].copy()
        df["severity_in_60m"] = df["severity_in_60m"].astype("int8")
        train_max = int(df["bucket"].max() * _TRAIN_RATIO)
        val_max   = int(df["bucket"].max() * (_TRAIN_RATIO + _VAL_RATIO))
        train_df  = df[df["bucket"] <= train_max]
        val_df    = df[(df["bucket"] > train_max) & (df["bucket"] <= val_max)]
        y_train   = train_df["severity_in_60m"]
        sw        = compute_sample_weight("balanced", y_train)
        clf = SpikeClassifier().fit(train_df, y_train, sample_weight=sw)
        return _threshold_sweep_table(clf, val_df[_X_COLS], val_df["severity_in_60m"], len(val_df))

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
            "class_rates_train", "class_rates_val", "class_rates_test",
            "macro_pr_auc", "macro_roc_auc",
            "pr_auc_class_0", "pr_auc_class_1", "pr_auc_class_2",
            "alarm_threshold", "alarm_precision", "alarm_recall",
            "weighted_f1", "macro_f1",
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

    def test_spike_config_has_alarm_threshold(self, pipeline_result):
        _, arts = pipeline_result
        cfg = json.loads((arts / "models" / "spike" / "spike_config.json").read_text())
        assert "alarm_threshold" in cfg
        assert 0.0 < cfg["alarm_threshold"] < 1.0

    def test_spike_config_has_final_metrics(self, pipeline_result):
        _, arts = pipeline_result
        cfg = json.loads((arts / "models" / "spike" / "spike_config.json").read_text())
        assert "final_metrics" in cfg
        assert "macro_pr_auc" in cfg["final_metrics"]

    def test_spike_config_has_calibration_block(self, pipeline_result):
        """spike_config.json must have a calibration block with per-class metrics."""
        _, arts = pipeline_result
        cfg = json.loads((arts / "models" / "spike" / "spike_config.json").read_text())
        assert "calibration" in cfg, "spike_config.json must have a calibration block"
        cal = cfg["calibration"]
        assert cal["method"]  == "isotonic_ovr"
        assert cal["fit_on"]  == "validation_set"
        assert "sklearn_version" in cal
        for k in ("class_0", "class_1", "class_2"):
            assert k in cal["per_class"], f"calibration.per_class must contain {k}"
            cls_metrics = cal["per_class"][k]
            assert "brier_raw" in cls_metrics
            assert "brier_cal" in cls_metrics
            assert "ece_cal"   in cls_metrics

    def test_calibrators_pkl_written(self, pipeline_result):
        """calibrators.pkl must be saved alongside the 60m model."""
        _, arts = pipeline_result
        assert (arts / "models" / "spike" / "calibrators.pkl").exists()

    def test_final_metrics_has_calibrated_pr_auc(self, pipeline_result):
        """spike_config.json final_metrics must include macro_pr_auc_calibrated."""
        _, arts = pipeline_result
        cfg = json.loads((arts / "models" / "spike" / "spike_config.json").read_text())
        assert "macro_pr_auc_calibrated" in cfg["final_metrics"], (
            "final_metrics must include macro_pr_auc_calibrated"
        )
        val = cfg["final_metrics"]["macro_pr_auc_calibrated"]
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0 or val != val  # allow NaN for tiny synthetic data

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

    def test_binary_horizon_model_dirs_written(self, pipeline_result):
        """spike_15m / 30m / 45m and spike_severe_ovr model dirs must all exist."""
        _, arts = pipeline_result
        for horizon in ("spike_15m", "spike_30m", "spike_45m"):
            assert (arts / "models" / horizon / "spike_model.json").exists(), (
                f"{horizon} model must be trained"
            )
        assert (arts / "models" / "spike_severe_ovr" / "spike_model.json").exists(), (
            "OVR severe model (Phase 5) must be trained alongside binary horizons"
        )


# ── Walk-forward enabled in run() ─────────────────────────────────────────────

class TestRunWithWalkForward:
    @pytest.fixture(scope="class")
    def wf_result(self, features_parquet, tmp_path_factory):
        import shutil
        arts_dir  = tmp_path_factory.mktemp("arts_wf")
        feat_dest = arts_dir / "cluster_features.parquet"
        pd.read_parquet(features_parquet).to_parquet(
            feat_dest, engine="pyarrow", compression="zstd", index=False
        )
        shutil.copy(
            features_parquet.parent / "spike_thresholds.parquet",
            arts_dir / "spike_thresholds.parquet",
        )
        result = run(
            data_path             = arts_dir / "nonexistent.csv",
            artifacts_dir         = arts_dir,
            train_ratio           = _TRAIN_RATIO,
            val_ratio             = _VAL_RATIO,
            from_step             = 3,
            walk_forward          = True,
            n_folds               = 3,
            seed                  = 42,
            n_estimators          = 50,
            early_stopping_rounds = 10,
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


# ── Hyperparameter search (Phase 1) ───────────────────────────────────────────

class TestHyperparameterSearch:
    @pytest.fixture(scope="class")
    def train_df(self, features_parquet):
        """Training split of the synthetic features parquet."""
        df = pd.read_parquet(features_parquet)
        df = df[df["severity_in_60m"].notna()].copy()
        df["severity_in_60m"] = df["severity_in_60m"].astype("int8")
        train_max = int(df["bucket"].max() * _TRAIN_RATIO)
        return df[df["bucket"] <= train_max].reset_index(drop=True)

    def test_search_returns_expected_param_keys(self, train_df):
        """Optuna result must contain all 8 tuned hyperparameter keys.
        n_estimators is excluded (Phase 5): final training uses 2000 + early stopping.
        """
        result        = _run_optuna_search(train_df, seed=42, n_trials=2, n_estimators=50)
        expected_keys = {
            "max_depth", "learning_rate",
            "subsample", "colsample_bytree", "min_child_weight",
            "gamma", "reg_alpha", "reg_lambda",
        }
        assert expected_keys == set(result["params"].keys())
        assert "n_estimators" not in result["params"]

    def test_search_result_structure(self, train_df, tmp_path):
        """Top-level result dict must have all expected metadata keys."""
        result = _run_optuna_search(train_df, seed=42, n_trials=2, output_dir=tmp_path, n_estimators=50)
        assert "params"            in result
        assert "best_macro_pr_auc" in result
        assert "n_trials"          in result
        assert "n_completed"       in result
        assert "resumed_from"      in result
        assert result["n_trials"]           == 2
        assert result["n_completed"]        == 2
        assert result["resumed_from"]       == 0
        assert 0.0 <= result["best_macro_pr_auc"] <= 1.0

    def test_resume_skips_completed_trials(self, train_df, tmp_path):
        """Running the search twice with the same output_dir must resume, not restart.
        Second call should report resumed_from == first call's n_completed."""
        r1 = _run_optuna_search(train_df, seed=42, n_trials=2, output_dir=tmp_path, n_estimators=50)
        r2 = _run_optuna_search(train_df, seed=42, n_trials=2, output_dir=tmp_path, n_estimators=50)
        # All trials already done — second call must skip and report resumed_from=2
        assert r2["resumed_from"] == 2
        assert r2["n_completed"]  == 2

    def test_warmstart_params_accepted(self, train_df, tmp_path):
        """warmstart_params must not raise and the enqueued trial is counted.

        Includes n_estimators in the warmstart dict (as an old best_params.json
        would) — Phase 5 code must filter it out silently before enqueue_trial().
        """
        warmstart = {
            "n_estimators": 300,  # legacy key — must be stripped before enqueue
            "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1,
            "gamma": 0.0, "reg_alpha": 0.0, "reg_lambda": 1.0,
        }
        result = _run_optuna_search(
            train_df, seed=42, n_trials=3,
            output_dir=tmp_path / "ws",
            warmstart_params=warmstart,
            n_estimators=50,
        )
        assert result["n_completed"] == 3
        assert "params" in result
        assert "n_estimators" not in result["params"]

    def test_config_has_hyperparameter_search_block_after_tuning(
        self, features_parquet, tmp_path_factory
    ):
        """spike_config.json must contain a hyperparameter_search block with
        best_params populated after a tuning run."""
        import shutil
        arts_dir  = tmp_path_factory.mktemp("arts_tune")
        feat_dest = arts_dir / "cluster_features.parquet"
        pd.read_parquet(features_parquet).to_parquet(
            feat_dest, engine="pyarrow", compression="zstd", index=False
        )
        shutil.copy(
            features_parquet.parent / "spike_thresholds.parquet",
            arts_dir / "spike_thresholds.parquet",
        )
        run(
            data_path             = arts_dir / "nonexistent.csv",
            artifacts_dir         = arts_dir,
            train_ratio           = _TRAIN_RATIO,
            val_ratio             = _VAL_RATIO,
            from_step             = 3,
            walk_forward          = False,
            seed                  = 42,
            tune_hyperparams      = True,
            n_trials              = 2,
            n_estimators          = 50,
            early_stopping_rounds = 10,
        )
        cfg = json.loads(
            (arts_dir / "models" / "spike" / "spike_config.json").read_text()
        )
        assert "hyperparameter_search" in cfg
        hs = cfg["hyperparameter_search"]
        assert hs["enabled"] is True
        assert hs["best_params"] is not None
        assert isinstance(hs["best_params"], dict)
        assert len(hs["best_params"]) == 8


# ── Calibration helpers ───────────────────────────────────────────────────────

class TestApplyCalibrators:
    """_apply_calibrators must produce a valid probability distribution."""

    @pytest.fixture(scope="class")
    def calibrators_and_probs(self):
        from sklearn.isotonic import IsotonicRegression
        rng = np.random.default_rng(0)
        # Synthetic 3-class probabilities (rows sum to 1)
        raw = rng.dirichlet(alpha=[3, 1, 0.5], size=200).astype("float32")
        y   = rng.choice([0, 1, 2], size=200, p=[0.70, 0.20, 0.10])
        cals = []
        for k in range(3):
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(raw[:, k].astype("float64"), (y == k).astype("float64"))
            cals.append(ir)
        return cals, raw, y

    def test_output_rows_sum_to_one(self, calibrators_and_probs):
        cals, raw, _ = calibrators_and_probs
        cal = _apply_calibrators(cals, raw)
        np.testing.assert_allclose(cal.sum(axis=1), np.ones(len(raw)), atol=1e-5)

    def test_output_in_unit_interval(self, calibrators_and_probs):
        cals, raw, _ = calibrators_and_probs
        cal = _apply_calibrators(cals, raw)
        assert (cal >= 0.0).all()
        assert (cal <= 1.0).all()

    def test_output_dtype_float32(self, calibrators_and_probs):
        cals, raw, _ = calibrators_and_probs
        cal = _apply_calibrators(cals, raw)
        assert cal.dtype == np.float32

    def test_zero_row_fallback_is_uniform(self):
        """Rows where all calibrators return 0 should be uniform 1/3."""
        from sklearn.isotonic import IsotonicRegression
        # Calibrators that always predict 0 (fit on all-zero target)
        cals = []
        for _ in range(3):
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit([0.0, 1.0], [0.0, 0.0])   # always returns 0
            cals.append(ir)
        raw = np.array([[0.01, 0.01, 0.01]], dtype="float32")
        cal = _apply_calibrators(cals, raw)
        np.testing.assert_allclose(cal[0], [1/3, 1/3, 1/3], atol=1e-5)


class TestSelectAlarmThreshold:
    """_select_alarm_threshold must return a float in (0, 1)."""

    def test_returns_float_in_unit_interval(self):
        rng     = np.random.default_rng(0)
        p_alarm = rng.uniform(0.0, 1.0, 300)
        y_bin   = rng.integers(0, 2, size=300)
        thresh  = _select_alarm_threshold(p_alarm, y_bin)
        assert isinstance(thresh, float)
        assert 0.0 < thresh < 1.0

    def test_selects_threshold_that_maximises_f1(self):
        """When p_alarm mirrors the labels perfectly the threshold should be
        at ~0.5 (boundary between 0 and 1) and yield perfect F1=1.0."""
        from sklearn.metrics import f1_score
        y_bin   = np.array([0, 0, 0, 1, 1, 1] * 10)
        # Scores: positives all at 0.9, negatives all at 0.1
        p_alarm = np.where(y_bin == 1, 0.9, 0.1)
        thresh  = _select_alarm_threshold(p_alarm, y_bin)
        pred    = (p_alarm >= thresh).astype(int)
        assert float(f1_score(y_bin, pred, zero_division=0)) == pytest.approx(1.0)


class TestComputeCalibrationMetrics:
    """_compute_calibration_metrics must return structurally valid dicts."""

    def test_returns_expected_keys(self):
        rng      = np.random.default_rng(0)
        raw      = rng.dirichlet([3, 1, 0.5], size=100)
        y_true   = rng.choice([0, 1, 2], size=100, p=[0.70, 0.20, 0.10])
        cal      = raw  # use raw as cal for simplicity
        metrics  = _compute_calibration_metrics(raw, y_true, cal)
        for k in ("class_0", "class_1", "class_2"):
            assert k in metrics
            for field in ("brier_raw", "brier_cal", "ece_cal"):
                assert field in metrics[k], f"Missing {field} in {k}"

    def test_brier_scores_in_unit_interval(self):
        rng    = np.random.default_rng(1)
        raw    = rng.dirichlet([3, 1, 0.5], size=200)
        y_true = rng.choice([0, 1, 2], size=200, p=[0.70, 0.20, 0.10])
        m      = _compute_calibration_metrics(raw, y_true, raw)
        for cls in ("class_0", "class_1", "class_2"):
            assert 0.0 <= m[cls]["brier_raw"] <= 1.0
            assert 0.0 <= m[cls]["ece_cal"]   <= 1.0

    def test_threshold_sweep_accepts_precomputed_probas(self, features_parquet):
        """_threshold_sweep_table with probas= must produce the same structure
        as without (internally calls predict_proba if probas=None)."""
        from spike_classifier import SpikeClassifier
        from sklearn.utils.class_weight import compute_sample_weight
        df       = pd.read_parquet(features_parquet)
        df       = df[df["severity_in_60m"].notna()].copy()
        df["severity_in_60m"] = df["severity_in_60m"].astype("int8")
        train_max = int(df["bucket"].max() * _TRAIN_RATIO)
        val_max   = int(df["bucket"].max() * (_TRAIN_RATIO + _VAL_RATIO))
        train_df  = df[df["bucket"] <= train_max]
        val_df    = df[(df["bucket"] > train_max) & (df["bucket"] <= val_max)]
        y_tr      = train_df["severity_in_60m"]
        sw        = compute_sample_weight("balanced", y_tr)
        clf       = SpikeClassifier().fit(train_df, y_tr, sample_weight=sw)
        X_val     = val_df[_X_COLS]
        y_val     = val_df["severity_in_60m"]
        raw_probs = clf.predict_proba(X_val)

        rows_auto     = _threshold_sweep_table(clf, X_val, y_val, len(val_df))
        rows_explicit = _threshold_sweep_table(clf, X_val, y_val, len(val_df),
                                               probas=raw_probs)

        assert len(rows_auto) == len(rows_explicit)
        for r_a, r_e in zip(rows_auto, rows_explicit):
            assert r_a["threshold"] == r_e["threshold"]
            assert abs(r_a["precision"] - r_e["precision"]) < 1e-6


class TestComputeFeatureImportanceFocalPath:
    """Regression tests for _compute_feature_importance with focal-loss classifiers.

    Before the bug fix, this function always called clf._model.get_booster(), which
    raises AttributeError for the focal path because clf._model (XGBClassifier) is
    never fitted — the booster lives in clf._booster.  These tests guard against
    any future regression of that fix.
    """

    def _make_X_val(self) -> pd.DataFrame:
        rng  = np.random.default_rng(99)
        data = {col: rng.uniform(0.0, 1.0, 60).astype("float32") for col in _X_COLS}
        return pd.DataFrame(data)

    def test_focal_model_produces_csv(self, tmp_path):
        """_compute_feature_importance must write feature_importance.csv for a focal model."""
        from spike_classifier import BinarySpikeClassifier
        X_val = self._make_X_val()
        rng   = np.random.default_rng(0)
        y     = pd.Series(rng.choice([0, 1], size=60, p=[0.8, 0.2]).astype("int8"))
        clf   = BinarySpikeClassifier(use_focal_loss=True, n_estimators=30)
        clf.fit(X_val, y)
        out = tmp_path / "fi.csv"
        _compute_feature_importance(clf, X_val, out)
        assert out.exists(), "feature_importance.csv not written for focal model"

    def test_focal_csv_contains_all_features(self, tmp_path):
        """Every column in _X_COLS must appear in the output (even if gain is 0)."""
        from spike_classifier import BinarySpikeClassifier
        X_val = self._make_X_val()
        rng   = np.random.default_rng(1)
        y     = pd.Series(rng.choice([0, 1], size=60, p=[0.8, 0.2]).astype("int8"))
        clf   = BinarySpikeClassifier(use_focal_loss=True, n_estimators=30)
        clf.fit(X_val, y)
        out = tmp_path / "fi.csv"
        _compute_feature_importance(clf, X_val, out)
        df_fi = pd.read_csv(out)
        assert set(df_fi["feature"]) == set(_X_COLS)

    def test_standard_model_still_works(self, tmp_path):
        """Standard (non-focal) BinarySpikeClassifier must still produce valid output."""
        from spike_classifier import BinarySpikeClassifier
        X_val = self._make_X_val()
        rng   = np.random.default_rng(2)
        y     = pd.Series(rng.choice([0, 1], size=60, p=[0.8, 0.2]).astype("int8"))
        clf   = BinarySpikeClassifier(n_estimators=30)
        clf.fit(X_val, y)
        out = tmp_path / "fi_std.csv"
        _compute_feature_importance(clf, X_val, out)
        assert out.exists()
        df_fi = pd.read_csv(out)
        assert set(df_fi["feature"]) == set(_X_COLS)


class TestFocalWithScalePosWeight:
    """scale_pos_weight must be passed through to the focal path (_booster_params).

    Option A design: focal_alpha is fixed at 0.5 (neutral) and scale_pos_weight
    continues to handle class weighting.  These tests guard against any future
    regression that drops spw back to 1.0 when focal loss is active.
    """

    def _make_imbalanced_data(self, n: int = 200, pos_rate: float = 0.025):
        rng   = np.random.default_rng(42)
        data  = {col: rng.uniform(0.0, 1.0, n).astype("float32") for col in _X_COLS}
        X     = pd.DataFrame(data)
        n_pos = max(1, int(n * pos_rate))
        y_arr = np.zeros(n, dtype="int8")
        y_arr[:n_pos] = 1
        rng.shuffle(y_arr)
        return X, pd.Series(y_arr)

    def test_booster_params_contains_scale_pos_weight(self):
        """_booster_params must carry scale_pos_weight so xgb.train() uses it."""
        from spike_classifier import BinarySpikeClassifier
        spw = 39.0
        clf = BinarySpikeClassifier(use_focal_loss=True, scale_pos_weight=spw, n_estimators=10)
        assert "scale_pos_weight" in clf._booster_params
        assert clf._booster_params["scale_pos_weight"] == spw

    def test_focal_alpha_stored_when_passed(self):
        """Explicitly passing focal_alpha=0.5 (neutral) must be stored correctly."""
        from spike_classifier import BinarySpikeClassifier
        clf = BinarySpikeClassifier(use_focal_loss=True, focal_alpha=0.5, n_estimators=10)
        assert clf._focal_alpha == 0.5

    def test_focal_with_spw_produces_higher_recall_than_no_spw(self):
        """With severe class imbalance, spw=40 focal model must recall more positives
        than spw=1 focal model — verifying the weight is actually applied."""
        from spike_classifier import BinarySpikeClassifier
        from sklearn.metrics import recall_score
        X, y = self._make_imbalanced_data(n=300, pos_rate=0.025)
        X_arr = X.values.astype("float32")
        y_arr = y.values

        clf_spw = BinarySpikeClassifier(
            use_focal_loss=True, scale_pos_weight=40.0,
            focal_alpha=0.5, focal_gamma=2.0, n_estimators=50,
        )
        clf_spw.fit(X, y)

        clf_no_spw = BinarySpikeClassifier(
            use_focal_loss=True, scale_pos_weight=1.0,
            focal_alpha=0.5, focal_gamma=2.0, n_estimators=50,
        )
        clf_no_spw.fit(X, y)

        thresh = 0.3
        pred_spw    = (clf_spw.predict_proba(X_arr)[:, 1] >= thresh).astype(int)
        pred_no_spw = (clf_no_spw.predict_proba(X_arr)[:, 1] >= thresh).astype(int)
        recall_spw    = recall_score(y_arr, pred_spw,    zero_division=0)
        recall_no_spw = recall_score(y_arr, pred_no_spw, zero_division=0)
        assert recall_spw >= recall_no_spw, (
            f"spw=40 recall {recall_spw:.3f} not >= spw=1 recall {recall_no_spw:.3f}"
        )


# ── Two-stage cascade (Stage 2 OVR severe) ────────────────────────────────────

class TestCascadeStage2:
    """cascade_stage2_ovr=True must restrict OVR severe training to spike-positive
    rows only, raising the severe class fraction well above the ~2.4% population rate.
    """

    @pytest.fixture(scope="class")
    def cascade_arts(self, features_parquet, tmp_path_factory):
        import shutil
        arts_dir = tmp_path_factory.mktemp("arts_cascade")
        pd.read_parquet(features_parquet).to_parquet(
            arts_dir / "cluster_features.parquet",
            engine="pyarrow", compression="zstd", index=False,
        )
        shutil.copy(
            features_parquet.parent / "spike_thresholds.parquet",
            arts_dir / "spike_thresholds.parquet",
        )
        run(
            data_path             = arts_dir / "nonexistent.csv",
            artifacts_dir         = arts_dir,
            train_ratio           = _TRAIN_RATIO,
            val_ratio             = _VAL_RATIO,
            from_step             = 3,
            walk_forward          = False,
            seed                  = 42,
            n_estimators          = 50,
            early_stopping_rounds = 10,
            cascade_stage2_ovr    = True,
        )
        return arts_dir

    def test_ovr_severe_model_is_written(self, cascade_arts):
        """cascade_stage2_ovr must still produce a trained OVR severe model on disk."""
        assert (cascade_arts / "models" / "spike_severe_ovr" / "spike_model.json").exists()

    def test_cascade_flag_recorded_in_config(self, cascade_arts):
        """spike_config.json must record cascade_stage2=True for auditability."""
        cfg = json.loads(
            (cascade_arts / "models" / "spike_severe_ovr" / "spike_config.json").read_text()
        )
        assert cfg.get("cascade_stage2") is True

    def test_spike_rate_train_reflects_filtered_population(self, cascade_arts):
        """Filtering to spike-positive rows raises the severe fraction well above
        the ~2.4% population rate.  Even in synthetic data the cascade rate must
        exceed 15% — a threshold that is impossible without active filtering."""
        cfg = json.loads(
            (cascade_arts / "models" / "spike_severe_ovr" / "spike_config.json").read_text()
        )
        assert cfg["spike_rate_train"] >= 0.15, (
            f"Expected cascade Stage 2 spike_rate_train >= 0.15 (well above the "
            f"~2.4%% OVR population rate), got {cfg['spike_rate_train']:.3f}. "
            "The spike-positive row filter may not be active."
        )

    def test_horizon_metadata_unchanged(self, cascade_arts):
        """Cascade mode must preserve the OVR severe horizon / label metadata."""
        cfg = json.loads(
            (cascade_arts / "models" / "spike_severe_ovr" / "spike_config.json").read_text()
        )
        assert cfg["horizon"] == "severe_ovr"
        assert cfg["label_col"] == "spike_severe_ovr"

    def test_15m_30m_45m_models_unaffected(self, cascade_arts):
        """cascade_stage2_ovr must not touch the binary horizon models."""
        for horizon in ("spike_15m", "spike_30m", "spike_45m"):
            cfg = json.loads(
                (cascade_arts / "models" / horizon / "spike_config.json").read_text()
            )
            assert cfg.get("cascade_stage2") is False, (
                f"{horizon} must not be in cascade mode"
            )
