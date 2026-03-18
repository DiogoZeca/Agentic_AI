"""Tests for spike_feature_engineer.py — Step 2 of the CPU spike prediction pipeline.

Fixtures
--------
gap_group
    Single machine, buckets [1, 2, 5, 6, 7] — 2-bucket gap at 3 and 4.
    Used to verify that lag/EWMA features treat idle windows as 0 rather
    than carrying forward the last observed value.

label_group
    Single machine, 25 contiguous buckets, spike at bucket 6 (total_cpu=0.9,
    all others 0.1). Enough rows to produce valid labels with horizon=12.

    With threshold ≈ 0.14 (p95 of the 25-row series):
      buckets 1-5  →  spike_in_60m = 1  (bucket 6 is within 12-window look-ahead)
      buckets 6-13 →  spike_in_60m = 0  (bucket 6 not in future look-ahead)
      buckets 14-25 → spike_in_60m = NaN  (last 12 rows, no full look-ahead)

agg_parquet
    Two-machine, 25-bucket parquet written to a tmp file. Used to test the
    full engineer() pipeline end-to-end.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike_feature_engineer import (
    _FEATURE_COLS,
    _HORIZON,
    _LAGS,
    _THRESHOLD_COLS,
    _add_label,
    _compute_thresholds,
    _engineer_machine,
    engineer,
)

# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_agg_row(machine_id, bucket, total_cpu=0.1, peak_cpu=0.15,
                  total_mem=0.1, peak_mem=0.2, disk_io=0.01, n_tasks=5):
    return {
        "machine_id": machine_id,
        "bucket":     bucket,
        "time_us":    bucket * 300_000_000,
        "total_cpu":  total_cpu,
        "peak_cpu":   peak_cpu,
        "total_mem":  total_mem,
        "peak_mem":   peak_mem,
        "disk_io":    disk_io,
        "n_tasks":    n_tasks,
    }


def _make_agg_df(rows: list[dict]) -> pd.DataFrame:
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
def gap_group() -> pd.DataFrame:
    """Machine 1 — buckets 1, 2, 5, 6, 7  (gap at 3 and 4)."""
    rows = [
        _make_agg_row(1, 1, total_cpu=0.3, peak_cpu=0.4),
        _make_agg_row(1, 2, total_cpu=0.4, peak_cpu=0.5),
        _make_agg_row(1, 5, total_cpu=0.5, peak_cpu=0.6),
        _make_agg_row(1, 6, total_cpu=0.6, peak_cpu=0.7),
        _make_agg_row(1, 7, total_cpu=0.7, peak_cpu=0.8),
    ]
    return _make_agg_df(rows)


@pytest.fixture(scope="module")
def label_group() -> pd.DataFrame:
    """Machine 1 — 25 contiguous buckets, spike (total_cpu=0.9) at bucket 6."""
    rows = [
        _make_agg_row(1, b, total_cpu=(0.9 if b == 6 else 0.1), peak_cpu=0.15)
        for b in range(1, 26)
    ]
    return _make_agg_df(rows)


@pytest.fixture(scope="module")
def agg_parquet(tmp_path_factory) -> Path:
    """Write a minimal two-machine cluster_agg.parquet for end-to-end tests."""
    rows = [
        _make_agg_row(m, b, total_cpu=(0.9 if b == 6 else 0.1))
        for m in (1, 2)
        for b in range(1, 26)
    ]
    df   = _make_agg_df(rows)
    path = tmp_path_factory.mktemp("agg") / "cluster_agg.parquet"
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    return path


@pytest.fixture(scope="module")
def eng_result(agg_parquet, tmp_path_factory):
    """Run engineer() once and share the result across the module."""
    out_dir = tmp_path_factory.mktemp("features")
    out_p   = out_dir / "cluster_features.parquet"
    thr_p   = out_dir / "spike_thresholds.parquet"
    df      = engineer(agg_parquet, out_p, thr_p, train_ratio=0.6, horizon=_HORIZON)
    return df, out_p, thr_p


# ── _engineer_machine: gap handling ──────────────────────────────────────────

class TestEngineerMachineGaps:
    """Problem 1 fix: idle gaps must be treated as 0, not as carry-forward."""

    @pytest.fixture(scope="class")
    def full_series(self, gap_group):
        return _engineer_machine(gap_group, 1)

    @pytest.fixture(scope="class")
    def original_rows(self, full_series):
        return full_series[full_series["_original"]].reset_index(drop=True)

    def test_full_series_has_all_buckets(self, full_series):
        """Gap-filled series must cover every bucket from min to max."""
        assert set(full_series["bucket"].tolist()) == {1, 2, 3, 4, 5, 6, 7}

    def test_original_flag_marks_source_rows(self, full_series):
        """_original must be True for buckets {1,2,5,6,7} only."""
        orig  = set(full_series.loc[full_series["_original"], "bucket"].tolist())
        filled = set(full_series.loc[~full_series["_original"], "bucket"].tolist())
        assert orig   == {1, 2, 5, 6, 7}
        assert filled == {3, 4}

    def test_lag1_at_bucket5_is_zero(self, original_rows):
        """bucket 5: lag_1 must be 0 (bucket 4 was idle gap, not carry-forward)."""
        row = original_rows[original_rows["bucket"] == 5].iloc[0]
        assert float(row["cpu_lag_1"]) == pytest.approx(0.0)

    def test_lag2_at_bucket5_is_zero(self, original_rows):
        """bucket 5: lag_2 must be 0 (bucket 3 was also an idle gap)."""
        row = original_rows[original_rows["bucket"] == 5].iloc[0]
        assert float(row["cpu_lag_2"]) == pytest.approx(0.0)

    def test_lag3_at_bucket5_is_actual_value(self, original_rows):
        """bucket 5: lag_3 should reach back to bucket 2 (total_cpu = 0.4)."""
        row = original_rows[original_rows["bucket"] == 5].iloc[0]
        assert float(row["cpu_lag_3"]) == pytest.approx(0.4, abs=1e-4)

    def test_lag1_at_bucket6_is_actual(self, original_rows):
        """bucket 6 follows bucket 5 with no gap: lag_1 = 0.5."""
        row = original_rows[original_rows["bucket"] == 6].iloc[0]
        assert float(row["cpu_lag_1"]) == pytest.approx(0.5, abs=1e-4)


# ── _engineer_machine: task_dominance ────────────────────────────────────────

class TestTaskDominance:
    """Problem 4 fix: no divide-by-zero when total_cpu == 0."""

    @pytest.fixture(scope="class")
    def full_series(self, gap_group):
        return _engineer_machine(gap_group, 1)

    def test_task_dominance_zero_when_idle(self, full_series):
        """Filled (idle) rows have total_cpu=0 — task_dominance must be 0."""
        idle = full_series[~full_series["_original"]]
        assert (idle["task_dominance"] == 0.0).all()

    def test_task_dominance_value_when_active(self, full_series):
        """Active row: task_dominance = peak_cpu / total_cpu."""
        row = full_series[full_series["bucket"] == 5].iloc[0]
        expected = 0.6 / 0.5   # peak_cpu / total_cpu for bucket 5
        assert float(row["task_dominance"]) == pytest.approx(expected, rel=1e-4)


# ── _add_label: spike labeling ────────────────────────────────────────────────

class TestAddLabel:
    """Problem 5 fix: O(n) prefix-sum labeling with correct NaN for last rows."""

    @pytest.fixture(scope="class")
    def labeled(self, label_group):
        """Full series with labels, horizon=12, threshold=0.5."""
        full = _engineer_machine(label_group, 1)
        return _add_label(full, threshold=0.5, horizon=12)

    def test_spike_label_one_when_future_exceeds(self, labeled):
        """Bucket 1 looks ahead to buckets 2-13; bucket 6 (cpu=0.9) is in range."""
        row = labeled[labeled["bucket"] == 1].iloc[0]
        assert float(row["spike_in_60m"]) == 1.0

    def test_spike_label_one_when_spike_at_horizon_edge(self, labeled):
        """Bucket 5 looks ahead to buckets 6-17; bucket 6 is still in range."""
        row = labeled[labeled["bucket"] == 5].iloc[0]
        assert float(row["spike_in_60m"]) == 1.0

    def test_spike_label_zero_when_spike_just_outside(self, labeled):
        """Bucket 6 looks ahead to buckets 7-18; bucket 6's spike is not included."""
        row = labeled[labeled["bucket"] == 6].iloc[0]
        assert float(row["spike_in_60m"]) == 0.0

    def test_spike_label_zero_when_no_future_spike(self, labeled):
        """Bucket 13 looks ahead to buckets 14-25; all are 0.1 < threshold."""
        row = labeled[labeled["bucket"] == 13].iloc[0]
        assert float(row["spike_in_60m"]) == 0.0

    def test_last_horizon_rows_are_nan(self, labeled):
        """Last 12 rows (buckets 14-25) must have spike_in_60m = NaN."""
        last_12 = labeled[labeled["bucket"] >= 14]
        assert last_12["spike_in_60m"].isna().all()
        assert len(last_12) == 12

    def test_count_of_valid_labels(self, labeled):
        """With 25 rows and horizon=12: 25-12=13 valid (non-NaN) labels."""
        assert labeled["spike_in_60m"].notna().sum() == 13


# ── _compute_thresholds ───────────────────────────────────────────────────────

class TestComputeThresholds:
    """Problem 2 + 3 fix: training-only thresholds with fallback for new machines."""

    @pytest.fixture(scope="class")
    def mixed_df(self):
        """Machine 1: low CPU in training, elevated in test; Machine 2: test-only."""
        rows = (
            # Machine 1: buckets 1-20 (train) → cpu=0.1; buckets 21-25 (test) → cpu=0.9
            [_make_agg_row(1, b, total_cpu=(0.9 if b > 20 else 0.1))
             for b in range(1, 26)]
            +
            # Machine 2: only in test portion
            [_make_agg_row(2, b, total_cpu=0.5) for b in range(21, 26)]
        )
        return _make_agg_df(rows)

    def test_threshold_uses_training_data_only(self, mixed_df):
        """Machine 1 p95 from training (all 0.1) must be below full-data p95."""
        train_max   = int(mixed_df["bucket"].max() * 0.8)   # = 20
        train_only  = _compute_thresholds(mixed_df, train_max)
        full_p95    = float(
            mixed_df[mixed_df["machine_id"] == 1]["total_cpu"].quantile(0.95)
        )
        assert float(train_only[1]) < full_p95

    def test_fallback_for_machine_absent_from_training(self, mixed_df):
        """Machine 2 (test-only) must receive the global training p95."""
        train_max     = 20
        thresholds    = _compute_thresholds(mixed_df, train_max)
        global_p95    = float(
            mixed_df[mixed_df["bucket"] <= train_max]["total_cpu"].quantile(0.95)
        )
        assert float(thresholds[2]) == pytest.approx(global_p95, rel=1e-5)

    def test_all_machines_have_threshold(self, mixed_df):
        """Every machine in the DataFrame must appear in the returned Series."""
        thresholds = _compute_thresholds(mixed_df, train_bucket_max=20)
        all_ids    = set(mixed_df["machine_id"].unique())
        assert all_ids == set(thresholds.index)


# ── Full pipeline: schema and types ──────────────────────────────────────────

class TestOutputSchema:
    def test_output_columns_match_spec(self, eng_result):
        df, _, _ = eng_result
        assert list(df.columns) == _FEATURE_COLS

    def test_machine_id_is_int64(self, eng_result):
        df, _, _ = eng_result
        assert df["machine_id"].dtype == np.int64

    def test_bucket_is_int64(self, eng_result):
        df, _, _ = eng_result
        assert df["bucket"].dtype == np.int64

    def test_n_tasks_is_int32(self, eng_result):
        df, _, _ = eng_result
        assert df["n_tasks"].dtype == np.int32

    def test_float_feature_cols_are_float32(self, eng_result):
        df, _, _ = eng_result
        float32_cols = [
            "total_cpu", "peak_cpu", "cpu_lag_1", "cpu_lag_6",
            "cpu_ewma_6", "cpu_ewma_24", "cpu_delta_1", "task_dominance",
            "cpu_vs_p95",
        ]
        for col in float32_cols:
            assert df[col].dtype == np.float32, f"{col} should be float32"

    def test_spike_label_has_expected_nan_count(self, eng_result):
        """2 machines × 12 NaN rows each = 24 NaN labels total."""
        df, _, _ = eng_result
        assert df["spike_in_60m"].isna().sum() == 2 * _HORIZON

    def test_thresholds_file_has_correct_columns(self, eng_result):
        _, _, thr_p = eng_result
        thresh = pd.read_parquet(thr_p)
        assert list(thresh.columns) == _THRESHOLD_COLS


# ── cpu_vs_p95: machine-relative normalisation ───────────────────────────────

class TestCpuVsP95:
    """cpu_vs_p95 = total_cpu / machine_p95 threshold.

    This feature gives XGBoost the machine-relative context that raw absolute
    lag values lack.  Values > 1 mean the node is already above its spike level.
    """

    def test_cpu_vs_p95_above_one_when_spiking(self, eng_result):
        """Bucket 6 has total_cpu=0.9 which exceeds the training-data p95.
        cpu_vs_p95 must therefore be > 1 at that row."""
        df, _, _ = eng_result
        spike_rows = df[(df["machine_id"] == 1) & (df["bucket"] == 6)]
        assert len(spike_rows) == 1
        assert float(spike_rows["cpu_vs_p95"].iloc[0]) > 1.0

    def test_cpu_vs_p95_below_one_when_normal(self, eng_result):
        """Bucket 1 has total_cpu=0.1 which is well below the p95 threshold."""
        df, _, _ = eng_result
        normal_rows = df[(df["machine_id"] == 1) & (df["bucket"] == 1)]
        assert float(normal_rows["cpu_vs_p95"].iloc[0]) < 1.0

    def test_cpu_vs_p95_nonnegative(self, eng_result):
        """All values must be >= 0 — total_cpu and threshold are both non-negative."""
        df, _, _ = eng_result
        assert (df["cpu_vs_p95"] >= 0.0).all()

    def test_cpu_vs_p95_consistent_with_total_cpu_and_threshold(self, eng_result):
        """cpu_vs_p95 must equal total_cpu / threshold_cpu for every row."""
        df, _, thr_p = eng_result
        thresholds = (
            pd.read_parquet(thr_p)
            .set_index("machine_id")["threshold_cpu"]
        )
        expected = (
            df["total_cpu"].values
            / df["machine_id"].map(thresholds).values
        )
        np.testing.assert_allclose(
            df["cpu_vs_p95"].values, expected, rtol=1e-4,
        )


# ── Full pipeline: parquet files written ─────────────────────────────────────

class TestParquetOutput:
    def test_features_parquet_written(self, eng_result):
        _, out_p, _ = eng_result
        assert out_p.exists()
        assert out_p.stat().st_size > 0

    def test_thresholds_parquet_written(self, eng_result):
        _, _, thr_p = eng_result
        assert thr_p.exists()
        assert thr_p.stat().st_size > 0

    def test_features_roundtrip_matches(self, eng_result):
        df, out_p, _ = eng_result
        loaded = pd.read_parquet(out_p)
        pd.testing.assert_frame_equal(df.reset_index(drop=True), loaded)


# ── Error handling ────────────────────────────────────────────────────────────

class TestErrors:
    def test_missing_input_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            engineer(tmp_path / "nonexistent.parquet", tmp_path / "out.parquet",
                     tmp_path / "thr.parquet")

    def test_empty_input_returns_empty_df(self, tmp_path):
        """Empty input parquet must produce empty feature and threshold files."""
        empty_p = tmp_path / "empty.parquet"
        pd.DataFrame(columns=[
            "machine_id", "bucket", "time_us",
            "total_cpu", "peak_cpu", "total_mem", "peak_mem", "disk_io", "n_tasks",
        ]).to_parquet(empty_p, engine="pyarrow", index=False)

        result = engineer(
            empty_p,
            tmp_path / "features.parquet",
            tmp_path / "thresholds.parquet",
        )
        assert result.empty
        assert list(result.columns) == _FEATURE_COLS
