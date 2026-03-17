"""Tests for feature_engineering.py — Stage 1 of the CPU power ML pipeline."""

import numpy as np
import pandas as pd
import pytest

from feature_engineering import build_features, load_and_build, save_metadata, load_metadata


@pytest.fixture(scope="module")
def features():
    """Load cpu_data.dat and build features once for the whole module."""
    return load_and_build("data/cpu_data.dat")


@pytest.fixture(scope="module")
def X(features):
    return features[0]


@pytest.fixture(scope="module")
def y_reg(features):
    return features[1]


@pytest.fixture(scope="module")
def y_clf(features):
    return features[2]


@pytest.fixture(scope="module")
def weights(features):
    return features[3]


@pytest.fixture(scope="module")
def metadata(features):
    return features[4]


# ── Output shapes and types ────────────────────────────────────────────────────

class TestOutputShapes:
    def test_X_has_8_columns(self, X):
        assert X.shape[1] == 8

    def test_X_columns(self, X):
        expected = {"cpu_pct", "cpu_pct_sq", "cpu_pct_cube",
                    "sqrt_cpu_pct", "log_cpu_pct", "idle_w",
                    "dynamic_range_w", "CPUTYPE"}
        assert set(X.columns) == expected

    def test_all_outputs_same_length(self, X, y_reg, y_clf, weights):
        n = len(X)
        assert len(y_reg) == n
        assert len(y_clf) == n
        assert len(weights) == n

    def test_y_reg_name(self, y_reg):
        assert y_reg.name == "power_w"

    def test_y_clf_name(self, y_clf):
        assert y_clf.name == "is_spike"

    def test_weights_name(self, weights):
        assert weights.name == "weight"

    def test_y_clf_is_bool(self, y_clf):
        assert y_clf.dtype == bool

    def test_metadata_has_11_cpu_types(self, metadata):
        assert len(metadata) == 11

    def test_metadata_keys(self, metadata):
        required_keys = {"idle_w", "full_w", "dynamic_range_w", "spike_threshold_w", "mean_std_w"}
        for cpu_type, meta in metadata.items():
            assert set(meta.keys()) == required_keys, f"{cpu_type} missing keys"


# ── Smoothed power target ──────────────────────────────────────────────────────

class TestSmoothedPower:
    def test_y_reg_no_nan(self, y_reg):
        assert not y_reg.isna().any(), "smooth_power_w contains NaN"

    def test_y_reg_all_positive(self, y_reg):
        assert (y_reg > 0).all(), "smooth_power_w has non-positive values"

    def test_smooth_power_monotone_per_cpu_type(self, X, y_reg):
        """Isotonic regression must enforce: power non-decreasing as cpu_pct increases."""
        for cpu_type, group_idx in X.groupby("CPUTYPE").groups.items():
            group = X.loc[group_idx].copy()
            group["power"] = y_reg.loc[group_idx]
            sorted_group = group.sort_values("cpu_pct")
            diffs = sorted_group["power"].diff().dropna()
            violations = (diffs < -1e-6).sum()
            assert violations == 0, (
                f"{cpu_type}: {violations} monotonicity violations after isotonic smoothing"
            )


# ── Features ──────────────────────────────────────────────────────────────────

class TestFeatureValues:
    def test_cpu_pct_range(self, X):
        assert X["cpu_pct"].min() >= 0
        assert X["cpu_pct"].max() <= 100

    def test_cpu_pct_sq_equals_square(self, X):
        expected = X["cpu_pct"] ** 2
        np.testing.assert_allclose(X["cpu_pct_sq"].values, expected.values)

    def test_cpu_pct_cube_equals_cube(self, X):
        expected = X["cpu_pct"] ** 3
        np.testing.assert_allclose(X["cpu_pct_cube"].values, expected.values)

    def test_sqrt_cpu_pct_non_negative(self, X):
        assert (X["sqrt_cpu_pct"] >= 0).all()

    def test_log_cpu_pct_non_negative(self, X):
        assert (X["log_cpu_pct"] >= 0).all()

    def test_idle_w_positive(self, X):
        assert (X["idle_w"] > 0).all()

    def test_dynamic_range_w_positive(self, X):
        assert (X["dynamic_range_w"] > 0).all()

    def test_no_nan_in_numeric_features(self, X):
        numeric_cols = [c for c in X.columns if c != "CPUTYPE"]
        assert not X[numeric_cols].isna().any().any()

    def test_cputype_column_is_string(self, X):
        # pandas 2.x may use StringDtype instead of object — both are string-like
        assert pd.api.types.is_string_dtype(X["CPUTYPE"])


# ── Weights ────────────────────────────────────────────────────────────────────

class TestWeights:
    def test_weight_sum_per_cpu_type_equals_1(self, X, weights):
        """Each CPU type must contribute total weight = 1.0 (Option 1 normalisation)."""
        for cpu_type, group_idx in X.groupby("CPUTYPE").groups.items():
            total = weights.loc[group_idx].sum()
            assert abs(total - 1.0) < 1e-6, f"{cpu_type}: weight sum = {total:.6f}, expected 1.0"

    def test_total_weight_equals_num_cpu_types(self, weights, metadata):
        n_types = len(metadata)
        total = weights.sum()
        assert abs(total - n_types) < 1e-6, f"Total weight = {total:.4f}, expected {n_types}"

    def test_weights_all_positive(self, weights):
        assert (weights > 0).all()


# ── Spike classification ───────────────────────────────────────────────────────

class TestSpikeClassification:
    def test_idle_rows_not_spike(self, X, y_clf):
        """CPU at 0% should never be classified as a spike (below threshold)."""
        idle_mask = X["cpu_pct"] == 0
        assert not y_clf[idle_mask].any(), "Some idle (cpu_pct=0) rows flagged as spike"

    def test_full_load_rows_are_spike(self, X, y_clf):
        """CPU at 100% should always be classified as a spike (top 25% of range)."""
        full_mask = X["cpu_pct"] == 100
        assert y_clf[full_mask].all(), "Some full-load (cpu_pct=100) rows NOT flagged as spike"

    def test_spike_rate_between_0_and_1(self, y_clf):
        rate = y_clf.mean()
        assert 0 < rate < 1, f"Spike rate {rate:.2%} is degenerate (all True or all False)"

    def test_spike_threshold_between_idle_and_full(self, metadata):
        for cpu_type, meta in metadata.items():
            assert meta["idle_w"] < meta["spike_threshold_w"] < meta["full_w"], (
                f"{cpu_type}: spike_threshold_w not between idle_w and full_w"
            )


# ── Metadata ───────────────────────────────────────────────────────────────────

class TestMetadata:
    def test_full_w_gt_idle_w(self, metadata):
        for cpu_type, meta in metadata.items():
            assert meta["full_w"] > meta["idle_w"], f"{cpu_type}: full_w <= idle_w"

    def test_dynamic_range_w_consistent(self, metadata):
        for cpu_type, meta in metadata.items():
            expected = meta["full_w"] - meta["idle_w"]
            assert abs(meta["dynamic_range_w"] - expected) < 0.01, (
                f"{cpu_type}: dynamic_range_w mismatch"
            )

    def test_metadata_values_are_floats(self, metadata):
        for cpu_type, meta in metadata.items():
            for key, val in meta.items():
                assert isinstance(val, float), f"{cpu_type}/{key} is {type(val)}, expected float"

    def test_all_cpu_types_in_X(self, X, metadata):
        cpu_types_in_X = set(X["CPUTYPE"].unique())
        cpu_types_in_meta = set(metadata.keys())
        assert cpu_types_in_X == cpu_types_in_meta


# ── Persistence ───────────────────────────────────────────────────────────────

class TestMetadataPersistence:
    def test_save_and_load_roundtrip(self, metadata, tmp_path):
        path = tmp_path / "test_metadata.json"
        save_metadata(metadata, path)
        loaded = load_metadata(path)
        assert set(loaded.keys()) == set(metadata.keys())
        for cpu_type in metadata:
            for key in metadata[cpu_type]:
                assert abs(loaded[cpu_type][key] - metadata[cpu_type][key]) < 1e-6
