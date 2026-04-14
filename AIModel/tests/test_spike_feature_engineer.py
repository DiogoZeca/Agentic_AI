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

    With threshold_p95 = 0.5, threshold_p99 = 0.6:
      buckets 1-5   →  severity_in_60m = 2  (bucket 6 at cpu=0.9 exceeds p99=0.6)
      buckets 6-13  →  severity_in_60m = 0  (bucket 6 not in future look-ahead)
      buckets 14-25 →  severity_in_60m = NaN  (last 12 rows, no full look-ahead)

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
    _BUCKETS_PER_DAY,
    _FEATURE_COLS,
    _HORIZON,
    _LAGS,
    _MAX_TIME_SINCE_SPIKE,
    _THRESHOLD_COLS,
    _add_cluster_features,
    _add_label,
    _compute_thresholds,
    _engineer_machine,
    engineer,
)
from spike_classifier import _X_COLS

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

    _THRESHOLD = 0.45   # between 0.4 (bucket 2) and 0.5 (bucket 5)

    @pytest.fixture(scope="class")
    def full_series(self, gap_group):
        return _engineer_machine(gap_group, 1, threshold=self._THRESHOLD)

    @pytest.fixture(scope="class")
    def original_rows(self, full_series):
        return full_series[full_series["_original"]].reset_index(drop=True)

    def test_full_series_has_all_buckets(self, full_series):
        """Gap-filled series must cover every bucket from min to max."""
        assert set(full_series["bucket"].tolist()) == {1, 2, 3, 4, 5, 6, 7}

    def test_original_flag_marks_source_rows(self, full_series):
        """_original must be True for buckets {1,2,5,6,7} only."""
        orig   = set(full_series.loc[full_series["_original"], "bucket"].tolist())
        filled = set(full_series.loc[~full_series["_original"], "bucket"].tolist())
        assert orig   == {1, 2, 5, 6, 7}
        assert filled == {3, 4}

    def test_lag1_at_bucket5_is_zero(self, original_rows):
        """bucket 5: lag_1 must be 0 (bucket 4 was idle gap, not carry-forward)."""
        row = original_rows[original_rows["bucket"] == 5].iloc[0]
        assert float(row["cpu_lag_1"]) == pytest.approx(0.0)

    def test_lag1_at_bucket6_is_actual(self, original_rows):
        """bucket 6 follows bucket 5 with no gap: lag_1 = 0.5."""
        row = original_rows[original_rows["bucket"] == 6].iloc[0]
        assert float(row["cpu_lag_1"]) == pytest.approx(0.5, abs=1e-4)


# ── _engineer_machine: task_dominance ────────────────────────────────────────

class TestTaskDominance:
    """Problem 4 fix: no divide-by-zero when total_cpu == 0."""

    @pytest.fixture(scope="class")
    def full_series(self, gap_group):
        return _engineer_machine(gap_group, 1, threshold=0.45)

    def test_task_dominance_zero_when_idle(self, full_series):
        """Filled (idle) rows have total_cpu=0 — task_dominance must be 0."""
        idle = full_series[~full_series["_original"]]
        assert (idle["task_dominance"] == 0.0).all()

    def test_task_dominance_value_when_active(self, full_series):
        """Active row: task_dominance = peak_cpu / total_cpu."""
        row      = full_series[full_series["bucket"] == 5].iloc[0]
        expected = 0.6 / 0.5   # peak_cpu / total_cpu for bucket 5
        assert float(row["task_dominance"]) == pytest.approx(expected, rel=1e-4)


# ── cpu_per_task (Phase 5) ────────────────────────────────────────────────────


class TestCpuPerTask:
    """cpu_per_task = total_cpu / max(n_tasks, 1) — per-task intensity proxy.

    Added in Phase 5 to capture machines doing more CPU work per scheduled
    task, which is predictive of upcoming spikes independent of raw load.
    """

    @pytest.fixture(scope="class")
    def full_series(self, gap_group):
        return _engineer_machine(gap_group, 1, threshold=0.45)

    def test_cpu_per_task_present(self, full_series):
        assert "cpu_per_task" in full_series.columns

    def test_cpu_per_task_is_float32(self, full_series):
        assert full_series["cpu_per_task"].dtype == np.float32

    def test_cpu_per_task_no_divide_by_zero(self, full_series):
        """All values must be finite — including filled (n_tasks=0) rows."""
        assert np.isfinite(full_series["cpu_per_task"].values).all()

    def test_cpu_per_task_value(self, full_series):
        """Spot-check: bucket 1 has total_cpu=0.3, n_tasks=5 → cpu_per_task=0.06."""
        row = full_series[full_series["bucket"] == 1].iloc[0]
        assert float(row["cpu_per_task"]) == pytest.approx(0.3 / 5, rel=1e-4)

    def test_cpu_per_task_zero_when_idle(self, full_series):
        """Gap-filled rows have total_cpu=0 → cpu_per_task must be 0."""
        idle = full_series[~full_series["_original"]]
        assert (idle["cpu_per_task"] == 0.0).all()


# ── _add_label: spike labeling ────────────────────────────────────────────────

class TestAddLabel:
    """Problem 5 fix: O(n) prefix-sum labeling with correct NaN for last rows."""

    @pytest.fixture(scope="class")
    def labeled(self, label_group):
        """Full series with labels, horizon=12, threshold_p95=0.5, threshold_p99=0.6."""
        full = _engineer_machine(label_group, 1, threshold=0.5)
        return _add_label(full, threshold_p95=0.5, threshold_p99=0.6, horizon=12)

    def test_severity_label_nonzero_when_future_exceeds(self, labeled):
        """Bucket 1 looks ahead to buckets 2-13; bucket 6 (cpu=0.9 > p99=0.6) is in range → severity 2."""
        row = labeled[labeled["bucket"] == 1].iloc[0]
        assert float(row["severity_in_60m"]) >= 1.0

    def test_severity_label_nonzero_when_spike_at_horizon_edge(self, labeled):
        """Bucket 5 looks ahead to buckets 6-17; bucket 6 still in range → severity 2."""
        row = labeled[labeled["bucket"] == 5].iloc[0]
        assert float(row["severity_in_60m"]) >= 1.0

    def test_severity_label_zero_when_spike_just_outside(self, labeled):
        """Bucket 6 looks ahead to buckets 7-18; bucket 6's spike is not included."""
        row = labeled[labeled["bucket"] == 6].iloc[0]
        assert float(row["severity_in_60m"]) == 0.0

    def test_severity_label_zero_when_no_future_spike(self, labeled):
        """Bucket 13 looks ahead to buckets 14-25; all are 0.1 < threshold."""
        row = labeled[labeled["bucket"] == 13].iloc[0]
        assert float(row["severity_in_60m"]) == 0.0

    def test_last_horizon_rows_are_nan(self, labeled):
        """Last 12 rows (buckets 14-25) must have severity_in_60m = NaN."""
        last_12 = labeled[labeled["bucket"] >= 14]
        assert last_12["severity_in_60m"].isna().all()
        assert len(last_12) == 12

    def test_count_of_valid_labels(self, labeled):
        """With 25 rows and horizon=12: 25-12=13 valid (non-NaN) labels."""
        assert labeled["severity_in_60m"].notna().sum() == 13

    def test_binary_mode_output_col(self, label_group):
        """_add_label with binary=True must write a {0,1} column under output_col."""
        full   = _engineer_machine(label_group, 1, threshold=0.5)
        result = _add_label(full, threshold_p95=0.5, threshold_p99=0.6,
                            horizon=3, output_col="spike_in_15m", binary=True)
        assert "spike_in_15m" in result.columns
        valid = result["spike_in_15m"].dropna().unique().tolist()
        assert all(v in (0.0, 1.0) for v in valid), f"Unexpected values: {valid}"

    def test_binary_mode_nan_count(self, label_group):
        """With 25 rows and horizon=3: last 3 rows must be NaN in binary label."""
        full   = _engineer_machine(label_group, 1, threshold=0.5)
        result = _add_label(full, threshold_p95=0.5, threshold_p99=0.6,
                            horizon=3, output_col="spike_in_15m", binary=True)
        assert int(result["spike_in_15m"].isna().sum()) == 3

    def test_binary_mode_detects_spike_at_horizon_edge(self, label_group):
        """Bucket 4 looks ahead 3 windows (5,6,7). Bucket 6 has cpu=0.9>p95=0.5 → spike_in_15m=1."""
        full   = _engineer_machine(label_group, 1, threshold=0.5)
        result = _add_label(full, threshold_p95=0.5, threshold_p99=0.6,
                            horizon=3, output_col="spike_in_15m", binary=True)
        row = result[result["bucket"] == 4].iloc[0]
        assert float(row["spike_in_15m"]) == pytest.approx(1.0)


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
        thresh_p95  = train_only.set_index("machine_id")["threshold_p95"]
        full_p95    = float(
            mixed_df[mixed_df["machine_id"] == 1]["total_cpu"].quantile(0.95)
        )
        assert float(thresh_p95[1]) < full_p95

    def test_fallback_for_machine_absent_from_training(self, mixed_df):
        """Machine 2 (test-only) must receive the global training p95."""
        train_max   = 20
        thresholds  = _compute_thresholds(mixed_df, train_max)
        thresh_p95  = thresholds.set_index("machine_id")["threshold_p95"]
        global_p95  = float(
            mixed_df[mixed_df["bucket"] <= train_max]["total_cpu"].quantile(0.95)
        )
        assert float(thresh_p95[2]) == pytest.approx(global_p95, rel=1e-5)

    def test_all_machines_have_threshold(self, mixed_df):
        """Every machine in the DataFrame must appear in the returned DataFrame."""
        thresholds = _compute_thresholds(mixed_df, train_bucket_max=20)
        all_ids    = set(mixed_df["machine_id"].unique())
        assert all_ids == set(thresholds["machine_id"])


# ── Spike history features ────────────────────────────────────────────────────

class TestSpikeHistoryFeatures:
    """Group A: spike_now, spike_in_last_1/3/6, time_since_last_spike.

    All features must be derived from total_cpu > threshold, NEVER from
    severity_in_60m (the future label).  The leakage firewall is:

        exc = spike_now.shift(1).fillna(0)

    No feature may depend on any window at t or later.
    """

    _THRESHOLD = 0.45   # bucket 1=0.3 (<), bucket 2=0.4 (<), bucket 5=0.5 (>), 6=0.6 (>), 7=0.7 (>)

    @pytest.fixture(scope="class")
    def series(self, gap_group):
        """Full gap-filled series with threshold=0.45."""
        return _engineer_machine(gap_group, 1, threshold=self._THRESHOLD)

    def test_spike_now_zero_below_threshold(self, series):
        """Buckets 1 (0.3) and 2 (0.4) are below threshold 0.45 → spike_now = 0."""
        for b in (1, 2):
            row = series[series["bucket"] == b].iloc[0]
            assert float(row["spike_now"]) == pytest.approx(0.0)

    def test_spike_now_one_above_threshold(self, series):
        """Buckets 5 (0.5), 6 (0.6), 7 (0.7) exceed threshold 0.45 → spike_now = 1."""
        for b in (5, 6, 7):
            row = series[series["bucket"] == b].iloc[0]
            assert float(row["spike_now"]) == pytest.approx(1.0)

    def test_spike_now_zero_for_gap_filled_rows(self, series):
        """Gap-filled rows have total_cpu=0 < any positive threshold → spike_now = 0."""
        filled = series[~series["_original"]]
        assert (filled["spike_now"] == 0.0).all()

    def test_spike_in_last_1_at_first_row_is_zero(self, series):
        """No prior window exists at row 0 → spike_in_last_1 = 0 (not NaN)."""
        first = series.iloc[0]
        assert float(first["spike_in_last_1"]) == pytest.approx(0.0)

    def test_spike_in_last_1_reflects_previous_window(self, series):
        """bucket 3 (gap-filled, cpu=0): spike_in_last_1 should equal spike_now[bucket 2].
        bucket 2 total_cpu=0.4 < 0.45 → spike_now[2]=0 → spike_in_last_1[3]=0."""
        row = series[series["bucket"] == 3].iloc[0]
        assert float(row["spike_in_last_1"]) == pytest.approx(0.0)

    def test_spike_in_last_1_after_spike(self, series):
        """bucket 6 (cpu=0.6 > 0.45 → spike_now=1) → spike_in_last_1[bucket 7] = 1."""
        row = series[series["bucket"] == 7].iloc[0]
        assert float(row["spike_in_last_1"]) == pytest.approx(1.0)

    def test_spike_in_last_3_covers_3_windows(self, series):
        """At bucket 7: windows 4,5,6 are in look-back. buckets 5,6 > threshold → 1."""
        row = series[series["bucket"] == 7].iloc[0]
        assert float(row["spike_in_last_3"]) == pytest.approx(1.0)

    def test_spike_in_last_6_binary(self, series):
        """spike_in_last_6 must be in {0.0, 1.0} — it is a binary indicator."""
        vals = series["spike_in_last_6"].unique().tolist()
        assert all(v in (0.0, 1.0) for v in vals)

    def test_time_since_capped_before_first_spike(self, series):
        """Before any spike, time_since_last_spike must equal _MAX_TIME_SINCE_SPIKE."""
        # buckets 1, 2, 3, 4 are all before the first exceedance (bucket 5)
        for b in (1, 2, 3, 4):
            row = series[series["bucket"] == b].iloc[0]
            assert float(row["time_since_last_spike"]) == pytest.approx(
                float(_MAX_TIME_SINCE_SPIKE)
            )

    def test_time_since_is_one_immediately_after_spike(self, series):
        """bucket 6 (cpu=0.6 > threshold) → time_since_last_spike[bucket 7] = 1."""
        row = series[series["bucket"] == 7].iloc[0]
        assert float(row["time_since_last_spike"]) == pytest.approx(1.0)

    def test_spike_history_never_uses_severity_in_60m(self, label_group):
        """Critical leakage test: spike_in_last_1[t] must equal spike_now[t-1],
        NOT severity_in_60m[t-1].  severity_in_60m encodes future windows t+1..t+12.

        Build a series where spike_now[t-1]=0 but severity_in_60m[t-1]=2 (because a
        spike occurs later within the label's horizon).  Verify spike_in_last_1[t]=0.
        """
        full = _engineer_machine(label_group, 1, threshold=0.5)
        labeled = _add_label(full, threshold_p95=0.5, threshold_p99=0.6, horizon=12)

        # bucket 1: total_cpu=0.1 < 0.5 → spike_now[1]=0
        #           but severity_in_60m[1]=2 (bucket 6 exceeds p99=0.6 in look-ahead)
        row_1 = labeled[labeled["bucket"] == 1].iloc[0]
        row_2 = labeled[labeled["bucket"] == 2].iloc[0]

        assert float(row_1["spike_now"])         == pytest.approx(0.0)  # not a spike
        assert float(row_1["severity_in_60m"])   >= 1.0                 # label=2 (future!)
        # spike_in_last_1[2] must be 0 (spike_now[1]=0), not based on severity_in_60m[1]
        assert float(row_2["spike_in_last_1"]) == pytest.approx(0.0)


# ── Derivative features ───────────────────────────────────────────────────────

class TestDerivativeFeatures:
    """Group B: cpu_delta_2, cpu_vs_p95_delta, cpu_rolling_std_6."""

    _THRESHOLD = 0.45

    @pytest.fixture(scope="class")
    def series(self, gap_group):
        return _engineer_machine(gap_group, 1, threshold=self._THRESHOLD)

    def test_cpu_delta_2_is_acceleration(self):
        """cpu_delta_2[t] = cpu_delta_1[t] - cpu_delta_1[t-1] on a contiguous series.

        Uses a gap-free group so each original row is adjacent in the full series
        — the relationship holds exactly without gap-fill distortion.
        """
        rows = [_make_agg_row(99, b, total_cpu=0.1 * b) for b in range(1, 8)]
        df   = _make_agg_df(rows)
        s    = _engineer_machine(df, 99, threshold=0.5)
        orig = s[s["_original"]].reset_index(drop=True)
        for i in range(1, len(orig)):
            d1_t   = float(orig.loc[i,   "cpu_delta_1"])
            d1_t_1 = float(orig.loc[i-1, "cpu_delta_1"])
            d2     = float(orig.loc[i,   "cpu_delta_2"])
            assert d2 == pytest.approx(d1_t - d1_t_1, abs=1e-4)

    def test_cpu_rolling_std_6_nonnegative(self, series):
        """Standard deviation is always >= 0."""
        assert (series["cpu_rolling_std_6"] >= 0.0).all()

    def test_cpu_rolling_std_6_zero_for_constant_series(self):
        """Constant CPU → rolling std should be 0 after the first window."""
        rows = [_make_agg_row(99, b, total_cpu=0.3) for b in range(1, 10)]
        df   = _make_agg_df(rows)
        s    = _engineer_machine(df, 99, threshold=0.5)
        orig = s[s["_original"]].reset_index(drop=True)
        # std of a constant is 0; first row returns 0 from fillna
        np.testing.assert_allclose(
            orig["cpu_rolling_std_6"].iloc[1:].values, 0.0, atol=1e-5,
        )

    def test_cpu_vs_p95_delta_zero_when_constant_cpu(self):
        """When total_cpu doesn't change, cpu_vs_p95_delta = 0."""
        rows = [_make_agg_row(99, b, total_cpu=0.3) for b in range(1, 6)]
        df   = _make_agg_df(rows)
        s    = _engineer_machine(df, 99, threshold=0.5)
        orig = s[s["_original"]].reset_index(drop=True)
        np.testing.assert_allclose(
            orig["cpu_vs_p95_delta"].iloc[1:].values, 0.0, atol=1e-5,
        )

    def test_cpu_vs_p95_delta_proportional_to_delta_1(self, series):
        """cpu_vs_p95_delta = cpu_delta_1 / threshold for threshold > 0."""
        orig = series[series["_original"]].reset_index(drop=True)
        for _, row in orig.iterrows():
            expected = float(row["cpu_delta_1"]) / self._THRESHOLD
            assert float(row["cpu_vs_p95_delta"]) == pytest.approx(expected, abs=1e-4)


# ── Phase 2 new features ──────────────────────────────────────────────────────


class TestPhase2Features:
    """Tests for features added in Phase 2.

    cpu_spike_rate_24  — fraction of previous 24 windows that were spiking
    peak_cpu_vs_p95    — peak_cpu / machine p95 threshold (near-miss detection)
    """

    _THRESHOLD = 0.5

    @pytest.fixture(scope="class")
    def series(self):
        """25 contiguous buckets with cpu rising 0.1 → 0.34 then spiking 0.7.
        Threshold 0.5: spike only at the last bucket (bucket 25, cpu=0.7).
        """
        rows = []
        for i in range(1, 25):
            cpu = round(0.1 + i * 0.01, 3)
            rows.append(_make_agg_row(1, i, total_cpu=cpu, peak_cpu=cpu + 0.05))
        rows.append(_make_agg_row(1, 25, total_cpu=0.7, peak_cpu=0.8))
        return _make_agg_df(rows)

    # ── cpu_spike_rate_24 ─────────────────────────────────────────────────────

    def test_cpu_spike_rate_24_zero_when_no_prior_spikes(self, series):
        """No past windows exceeded threshold → rate is 0 everywhere."""
        result = _engineer_machine(series, 1, threshold=self._THRESHOLD)
        # All windows before the spike bucket should have rate = 0
        pre_spike = result[result["bucket"] < 25]
        assert (pre_spike["cpu_spike_rate_24"] == 0.0).all()

    def test_cpu_spike_rate_24_stays_zero_at_spike_row(self, series):
        """Rate at the spike bucket itself looks backward only (via shift(1))."""
        result = _engineer_machine(series, 1, threshold=self._THRESHOLD)
        spike_row = result[result["bucket"] == 25].iloc[0]
        # The spike is at bucket 25; no prior buckets spiked → rate still 0
        assert float(spike_row["cpu_spike_rate_24"]) == pytest.approx(0.0)

    def test_cpu_spike_rate_24_reflects_past_spikes(self):
        """When 12 of the last 24 windows spiked, rate ≈ 0.5."""
        rows = []
        for i in range(1, 31):
            cpu = 0.8 if i <= 12 else 0.1   # first 12 buckets: spike; rest: normal
            rows.append(_make_agg_row(1, i, total_cpu=cpu, peak_cpu=cpu + 0.05))
        df = _make_agg_df(rows)
        result = _engineer_machine(df, 1, threshold=self._THRESHOLD)
        # At bucket 30 (index=29): shift(1) looks at buckets 1-29
        # Only the first 12 of those were spiking: rolling(24).mean() on last 24
        # windows = rows 6-29 → rows 6-12 spike (7 spikes), 13-29 no spike = 7/24
        last_row = result[result["bucket"] == 30].iloc[0]
        assert float(last_row["cpu_spike_rate_24"]) == pytest.approx(7 / 24, abs=1e-4)

    def test_cpu_spike_rate_24_is_float32(self, series):
        result = _engineer_machine(series, 1, threshold=self._THRESHOLD)
        assert result["cpu_spike_rate_24"].dtype == np.float32

    # ── peak_cpu_vs_p95 ───────────────────────────────────────────────────────

    def test_peak_cpu_vs_p95_proportional_to_peak(self):
        """peak_cpu_vs_p95 = peak_cpu / threshold exactly."""
        rows = [
            _make_agg_row(1, i, total_cpu=0.3, peak_cpu=0.4)
            for i in range(1, 10)
        ]
        result = _engineer_machine(_make_agg_df(rows), 1, threshold=self._THRESHOLD)
        expected = 0.4 / self._THRESHOLD
        assert np.allclose(result["peak_cpu_vs_p95"].values, expected, rtol=1e-4)

    def test_peak_cpu_vs_p95_nonnegative(self, series):
        result = _engineer_machine(series, 1, threshold=self._THRESHOLD)
        assert (result["peak_cpu_vs_p95"] >= 0.0).all()

    def test_peak_cpu_vs_p95_is_float32(self, series):
        result = _engineer_machine(series, 1, threshold=self._THRESHOLD)
        assert result["peak_cpu_vs_p95"].dtype == np.float32


# ── Cluster-level features ────────────────────────────────────────────────────

class TestClusterFeatures:
    """Group C: cluster_cpu_p90, machine_rank_in_cluster.

    These are cross-sectional features (same bucket t, all machines) — not
    temporal leakage.  They are only present in the engineer() output, not
    from _engineer_machine() which processes one machine at a time.
    """

    def test_cluster_cpu_p90_present(self, eng_result):
        df, _, _ = eng_result
        assert "cluster_cpu_p90" in df.columns

    def test_machine_rank_present(self, eng_result):
        df, _, _ = eng_result
        assert "machine_rank_in_cluster" in df.columns

    def test_cluster_cpu_p90_nonnegative(self, eng_result):
        df, _, _ = eng_result
        assert (df["cluster_cpu_p90"] >= 0.0).all()

    def test_machine_rank_in_unit_interval(self, eng_result):
        df, _, _ = eng_result
        assert (df["machine_rank_in_cluster"] >= 0.0).all()
        assert (df["machine_rank_in_cluster"] <= 1.0).all()

    def test_machine_rank_is_float32(self, eng_result):
        df, _, _ = eng_result
        assert df["machine_rank_in_cluster"].dtype == np.float32

    def test_cluster_p90_is_float32(self, eng_result):
        df, _, _ = eng_result
        assert df["cluster_cpu_p90"].dtype == np.float32

    def test_cluster_p90_consistent_within_bucket(self, eng_result):
        """All machines in the same bucket must share the same cluster_cpu_p90."""
        df, _, _ = eng_result
        # For each bucket, the cluster_cpu_p90 value should be identical across rows
        inconsistent = (
            df.groupby("bucket")["cluster_cpu_p90"]
            .nunique()
            .gt(1)
            .any()
        )
        assert not inconsistent

    def test_higher_cpu_has_higher_rank(self, eng_result):
        """Within a bucket, the machine with the highest total_cpu gets the highest rank."""
        df, _, _ = eng_result
        # bucket 6 has machine 1 and 2 both with total_cpu=0.9 (tied) — use bucket 1
        bucket_df = df[df["bucket"] == 1].copy()
        if len(bucket_df) >= 2:
            max_cpu_idx = bucket_df["total_cpu"].idxmax()
            max_rank    = bucket_df.loc[max_cpu_idx, "machine_rank_in_cluster"]
            assert max_rank == pytest.approx(bucket_df["machine_rank_in_cluster"].max(), abs=1e-5)


# ── Time features ─────────────────────────────────────────────────────────────

class TestTimeFeatures:
    """Group D: hour_sin, hour_cos only (Phase 5: dow_sin/dow_cos removed).

    dow_sin/dow_cos carried near-zero SHAP importance (0.356 gain, 0.011)
    and introduced a day-of-week temporal confound on the 7-day dataset.
    Only within-day harmonic features are retained.
    """

    def test_all_time_features_present(self, eng_result):
        df, _, _ = eng_result
        for col in ("hour_sin", "hour_cos"):
            assert col in df.columns

    def test_dow_features_absent(self, eng_result):
        """dow_sin and dow_cos must not appear after Phase 5 removal."""
        df, _, _ = eng_result
        for col in ("dow_sin", "dow_cos"):
            assert col not in df.columns, f"{col} should have been removed in Phase 5"

    def test_time_features_in_unit_interval(self, eng_result):
        """sin/cos values must always lie in [-1.0, 1.0]."""
        df, _, _ = eng_result
        for col in ("hour_sin", "hour_cos"):
            assert (df[col] >= -1.0 - 1e-6).all(), f"{col} has values below -1"
            assert (df[col] <=  1.0 + 1e-6).all(), f"{col} has values above +1"

    def test_time_features_are_float32(self, eng_result):
        df, _, _ = eng_result
        for col in ("hour_sin", "hour_cos"):
            assert df[col].dtype == np.float32, f"{col} should be float32"

    def test_same_bucket_mod_same_hour_encoding(self, eng_result):
        """Two rows with the same bucket % 288 must have identical hour_sin/cos."""
        df, _, _ = eng_result
        df = df.copy()
        df["bucket_mod"] = df["bucket"] % _BUCKETS_PER_DAY
        grp = df.groupby("bucket_mod")[["hour_sin", "hour_cos"]].nunique()
        assert (grp["hour_sin"] == 1).all()
        assert (grp["hour_cos"] == 1).all()

    def test_hour_sin_cos_pythagorean_identity(self, eng_result):
        """sin^2 + cos^2 == 1 for within-day encoding."""
        df, _, _ = eng_result
        day_norm = df["hour_sin"].astype("float64")**2 + df["hour_cos"].astype("float64")**2
        np.testing.assert_allclose(day_norm.values, 1.0, atol=1e-5)


# ── Cross-module contract ─────────────────────────────────────────────────────

def test_x_cols_subset_of_feature_cols():
    """Every column in _X_COLS must appear in engineer() output (_FEATURE_COLS).

    This is a static contract test — if a feature is added to _X_COLS (the
    classifier's training column list) but forgotten in spike_feature_engineer
    the model would silently train on NaN-filled columns.  Catches the mismatch
    before any data is processed.
    """
    missing = set(_X_COLS) - set(_FEATURE_COLS)
    assert not missing, (
        f"These columns are in _X_COLS but missing from _FEATURE_COLS: {sorted(missing)}"
    )


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
            "total_cpu", "peak_cpu", "cpu_per_task", "cpu_lag_1",
            "cpu_ewma_6", "cpu_ewma_24", "cpu_delta_1", "cpu_delta_2",
            "cpu_rolling_std_6",
            "task_dominance", "cpu_vs_p95", "cpu_vs_p95_delta", "peak_cpu_vs_p95",
            # p99-level features (Fix 1)
            "spike_severe_now", "cpu_vs_p99", "peak_cpu_vs_p99",
            "band_position", "band_width",
            "spike_severe_in_last_1", "spike_severe_in_last_3", "spike_severe_in_last_6",
            "spike_now", "spike_in_last_1", "spike_in_last_3", "spike_in_last_6",
            "time_since_last_spike", "cpu_spike_rate_24",
            "cluster_cpu_p90", "machine_rank_in_cluster",
            "hour_sin", "hour_cos",
        ]
        for col in float32_cols:
            assert df[col].dtype == np.float32, f"{col} should be float32"

    def test_severity_label_has_expected_nan_count(self, eng_result):
        """2 machines × 12 NaN rows each = 24 NaN labels total."""
        df, _, _ = eng_result
        assert df["severity_in_60m"].isna().sum() == 2 * _HORIZON

    def test_binary_label_columns_present(self, eng_result):
        """Phase 4: spike_in_15m, spike_in_30m, spike_in_45m must be in output."""
        df, _, _ = eng_result
        for col in ("spike_in_15m", "spike_in_30m", "spike_in_45m"):
            assert col in df.columns, f"{col} missing from engineer() output"

    def test_binary_labels_values_in_zero_one(self, eng_result):
        """Binary horizon labels must only contain 0, 1, or NaN — never 2."""
        df, _, _ = eng_result
        for col in ("spike_in_15m", "spike_in_30m", "spike_in_45m"):
            valid = df[col].dropna().unique().tolist()
            assert all(v in (0.0, 1.0) for v in valid), (
                f"{col} contains unexpected values: {valid}"
            )

    def test_shorter_horizon_label_has_fewer_nan_rows(self, eng_result):
        """spike_in_15m (horizon=3) should have fewer NaN rows than severity_in_60m (horizon=12)."""
        df, _, _ = eng_result
        nan_60m = int(df["severity_in_60m"].isna().sum())
        nan_15m = int(df["spike_in_15m"].isna().sum())
        assert nan_15m < nan_60m

    def test_streak_features_present_in_output(self, eng_result):
        """Streak features must be present in the full engineer() output."""
        df, _, _ = eng_result
        for col in ("current_spike_streak", "max_spike_streak_24h"):
            assert col in df.columns, f"{col} missing from engineer() output"

    def test_streak_features_are_float32(self, eng_result):
        """Streak features must be float32."""
        df, _, _ = eng_result
        for col in ("current_spike_streak", "max_spike_streak_24h"):
            assert df[col].dtype == np.float32, f"{col} should be float32"


# ── Fixtures for streak and K-of-N tests ─────────────────────────────────────

@pytest.fixture(scope="module")
def streak_group() -> pd.DataFrame:
    """Machine 1 — 20 contiguous buckets, consecutive spikes at buckets 5, 6, 7.

    With threshold=0.5 (p95):
      spike_now[5]=1, spike_now[6]=1, spike_now[7]=1
      exc (shifted by 1): exc[6]=1, exc[7]=1, exc[8]=1, exc[9]=0

    Expected current_spike_streak values (at key buckets):
      bucket 5 → 0  (exc[5] = spike_now[4] = 0, no prior spike)
      bucket 8 → 3  (exc[6,7,8] all 1: run of 3 ending at bucket 8)
      bucket 9 → 0  (exc[9] = spike_now[8] = 0, run broken)
    """
    rows = [
        _make_agg_row(1, b, total_cpu=(0.9 if 5 <= b <= 7 else 0.1))
        for b in range(1, 21)
    ]
    return _make_agg_df(rows)


@pytest.fixture(scope="module")
def two_spike_group() -> pd.DataFrame:
    """Machine 1 — 25 contiguous buckets, spikes at buckets 6 AND 8 (non-consecutive).

    Used for K-of-N label tests.
    With threshold_p95=0.5: future_mild[bucket=1] = 2 (buckets 6 and 8 in look-ahead).
      K=1 → label ≥ 1 (any spike fires)
      K=2 → label ≥ 1 (exactly 2 spikes, meets K=2)
      K=3 → label = 0 (only 2 spikes, below K=3)
    """
    rows = [
        _make_agg_row(1, b, total_cpu=(0.9 if b in (6, 8) else 0.1))
        for b in range(1, 26)
    ]
    return _make_agg_df(rows)


# ── Streak feature tests ──────────────────────────────────────────────────────

class TestStreakFeatures:
    """Steps 1 and 3: current_spike_streak, max_spike_streak_24h."""

    _THRESHOLD_P95 = 0.5   # 0.1 < threshold < 0.9 — below/above CPU values in fixture
    _THRESHOLD_P99 = 0.95  # above 0.9, so severe streak should stay 0 throughout

    @pytest.fixture(scope="class")
    def full_series(self, streak_group):
        return _engineer_machine(streak_group, 1,
                                 threshold=self._THRESHOLD_P95,
                                 threshold_p99=self._THRESHOLD_P99)

    def test_streak_zero_before_first_spike(self, full_series):
        """At bucket 5 (first spike bucket), no prior spike → current_spike_streak = 0.

        exc[5] = spike_now[4] = (0.1 > 0.5) = 0.  No consecutive run ending here.
        """
        row = full_series[full_series["bucket"] == 5].iloc[0]
        assert float(row["current_spike_streak"]) == pytest.approx(0.0)

    def test_streak_one_after_first_spike(self, full_series):
        """At bucket 6, exc[6] = spike_now[5] = 1 → streak of 1 (run just started)."""
        row = full_series[full_series["bucket"] == 6].iloc[0]
        assert float(row["current_spike_streak"]) == pytest.approx(1.0)

    def test_streak_increments_through_run(self, full_series):
        """At bucket 8, exc[6,7,8] = [1,1,1] → consecutive run of 3 ending here."""
        row = full_series[full_series["bucket"] == 8].iloc[0]
        assert float(row["current_spike_streak"]) == pytest.approx(3.0)

    def test_streak_resets_after_gap(self, full_series):
        """At bucket 9, exc[9] = spike_now[8] = 0 → run broken, streak resets to 0."""
        row = full_series[full_series["bucket"] == 9].iloc[0]
        assert float(row["current_spike_streak"]) == pytest.approx(0.0)

    def test_max_streak_captures_longest_run(self, full_series):
        """At bucket 9 (just after the run ends), max_spike_streak_24h must be ≥ 3."""
        row = full_series[full_series["bucket"] == 9].iloc[0]
        assert float(row["max_spike_streak_24h"]) >= 3.0

    def test_max_streak_not_reset_by_later_non_spike(self, full_series):
        """After the run ends, max still remembers the historical best within the window."""
        row = full_series[full_series["bucket"] == 15].iloc[0]
        # The 3-bucket run (exc at 6,7,8) is within the last 24 windows of bucket 15.
        assert float(row["max_spike_streak_24h"]) >= 3.0

    def test_streak_features_are_nonnegative(self, full_series):
        """All streak features must be ≥ 0 everywhere."""
        for col in ("current_spike_streak", "max_spike_streak_24h"):
            assert (full_series[col] >= 0).all(), f"{col} has negative values"

    def test_max_streak_ge_current_streak(self, full_series):
        """max_spike_streak_24h ≥ current_spike_streak at every row (by definition)."""
        assert (full_series["max_spike_streak_24h"] >= full_series["current_spike_streak"]).all()


# ── K-of-N label tests ────────────────────────────────────────────────────────

class TestAddLabelMinFutureWindows:
    """Step 4: min_future_windows parameter on _add_label()."""

    _P95 = 0.5
    _P99 = 0.6

    @pytest.fixture(scope="class")
    def single_spike_labeled_k1(self, label_group):
        """label_group (1 spike at b6), K=1 — should match default behaviour."""
        full = _engineer_machine(label_group, 1, threshold=self._P95)
        return _add_label(full, threshold_p95=self._P95, threshold_p99=self._P99,
                          horizon=12, min_future_windows=1)

    @pytest.fixture(scope="class")
    def single_spike_labeled_k2(self, label_group):
        """label_group (1 spike at b6), K=2."""
        full = _engineer_machine(label_group, 1, threshold=self._P95)
        return _add_label(full, threshold_p95=self._P95, threshold_p99=self._P99,
                          horizon=12, min_future_windows=2)

    @pytest.fixture(scope="class")
    def two_spike_labeled_k2(self, two_spike_group):
        """two_spike_group (spikes at b6 and b8), K=2."""
        full = _engineer_machine(two_spike_group, 1, threshold=self._P95)
        return _add_label(full, threshold_p95=self._P95, threshold_p99=self._P99,
                          horizon=12, min_future_windows=2)

    @pytest.fixture(scope="class")
    def two_spike_labeled_k3(self, two_spike_group):
        """two_spike_group (spikes at b6 and b8), K=3."""
        full = _engineer_machine(two_spike_group, 1, threshold=self._P95)
        return _add_label(full, threshold_p95=self._P95, threshold_p99=self._P99,
                          horizon=12, min_future_windows=3)

    def test_k1_matches_default_signature(self, label_group):
        """min_future_windows=1 must produce identical labels to calling without the param."""
        full     = _engineer_machine(label_group, 1, threshold=self._P95)
        default  = _add_label(full, threshold_p95=self._P95, threshold_p99=self._P99, horizon=12)
        explicit = _add_label(full, threshold_p95=self._P95, threshold_p99=self._P99,
                              horizon=12, min_future_windows=1)
        pd.testing.assert_series_equal(
            default["severity_in_60m"].reset_index(drop=True),
            explicit["severity_in_60m"].reset_index(drop=True),
        )

    def test_k1_fires_on_single_spike(self, single_spike_labeled_k1):
        """K=1 (default): bucket 1 looks ahead to 2-13; bucket 6 is in range → label ≥ 1."""
        row = single_spike_labeled_k1[single_spike_labeled_k1["bucket"] == 1].iloc[0]
        assert float(row["severity_in_60m"]) >= 1.0

    def test_k2_single_spike_no_longer_fires(self, single_spike_labeled_k2):
        """K=2: label_group has exactly 1 spike in the 60m window → future_mild=1 < 2 → label=0."""
        row = single_spike_labeled_k2[single_spike_labeled_k2["bucket"] == 1].iloc[0]
        assert float(row["severity_in_60m"]) == pytest.approx(0.0)

    def test_k2_two_spikes_fires(self, two_spike_labeled_k2):
        """K=2: two_spike_group has 2 spikes (b6, b8) in look-ahead of bucket 1 → label ≥ 1."""
        row = two_spike_labeled_k2[two_spike_labeled_k2["bucket"] == 1].iloc[0]
        assert float(row["severity_in_60m"]) >= 1.0

    def test_k3_two_spikes_no_longer_fires(self, two_spike_labeled_k3):
        """K=3: only 2 spikes in the window → future_mild=2 < 3 → label=0."""
        row = two_spike_labeled_k3[two_spike_labeled_k3["bucket"] == 1].iloc[0]
        assert float(row["severity_in_60m"]) == pytest.approx(0.0)

    def test_nan_count_unchanged_by_k(self, single_spike_labeled_k1, single_spike_labeled_k2):
        """Changing K must not affect the number of NaN rows (last horizon rows)."""
        nan_k1 = int(single_spike_labeled_k1["severity_in_60m"].isna().sum())
        nan_k2 = int(single_spike_labeled_k2["severity_in_60m"].isna().sum())
        assert nan_k1 == nan_k2

    def test_k2_binary_mode_unaffected_by_min_future_windows(self, label_group):
        """Binary short-horizon labels must be computed with their own min_future_windows=1.

        This test calls _add_label with binary=True and min_future_windows=2 directly
        to confirm the parameter works for binary mode too (the internal wiring in
        engineer() passes K=1 for binary labels, but _add_label itself must support it).
        """
        full    = _engineer_machine(label_group, 1, threshold=self._P95)
        result  = _add_label(full, threshold_p95=self._P95, threshold_p99=self._P99,
                             horizon=3, output_col="spike_in_15m", binary=True,
                             min_future_windows=2)
        # bucket 4 looks at buckets 5,6,7 — only bucket 6 spikes → future_mild=1 < 2 → 0
        row = result[result["bucket"] == 4].iloc[0]
        assert float(row["spike_in_15m"]) == pytest.approx(0.0)

    def test_thresholds_file_has_correct_columns(self, eng_result):
        _, _, thr_p = eng_result
        thresh = pd.read_parquet(thr_p)
        assert list(thresh.columns) == _THRESHOLD_COLS


# ── p99-level features (Fix 1) ────────────────────────────────────────────────


class TestP99Features:
    """Tests for the 8 p99-level features added in Fix 1.

    All features are computed inside _engineer_machine() using threshold_p99.
    """

    _THRESHOLD     = 0.5   # p95 threshold
    _THRESHOLD_P99 = 0.7   # p99 threshold

    @pytest.fixture(scope="class")
    def series_p99(self):
        """25 contiguous buckets with a severe spike at bucket 20 (cpu=0.8 > p99=0.7)
        and a moderate spike at bucket 10 (cpu=0.6 > p95=0.5 but < p99=0.7)."""
        rows = []
        for b in range(1, 26):
            if b == 20:
                cpu = 0.8   # severe spike
            elif b == 10:
                cpu = 0.6   # moderate spike
            else:
                cpu = 0.2   # normal
            rows.append(_make_agg_row(1, b, total_cpu=cpu, peak_cpu=cpu + 0.05))
        return _make_agg_df(rows)

    @pytest.fixture(scope="class")
    def series(self, series_p99):
        return _engineer_machine(series_p99, 1,
                                 threshold=self._THRESHOLD,
                                 threshold_p99=self._THRESHOLD_P99)

    def test_new_p99_features_present_in_output(self, series):
        """All 8 p99-level features must appear in the output."""
        for col in (
            "spike_severe_now", "cpu_vs_p99", "peak_cpu_vs_p99",
            "band_position", "band_width",
            "spike_severe_in_last_1", "spike_severe_in_last_3", "spike_severe_in_last_6",
        ):
            assert col in series.columns, f"Missing p99 feature: {col}"

    def test_spike_severe_now_binary(self, series):
        """spike_severe_now must be in {0.0, 1.0} only."""
        vals = series["spike_severe_now"].unique().tolist()
        assert all(v in (0.0, 1.0) for v in vals), f"Unexpected values: {vals}"

    def test_spike_severe_now_one_at_severe_bucket(self, series):
        """Bucket 20 has cpu=0.8 > p99=0.7 → spike_severe_now = 1."""
        row = series[series["bucket"] == 20].iloc[0]
        assert float(row["spike_severe_now"]) == pytest.approx(1.0)

    def test_spike_severe_now_zero_at_moderate_bucket(self, series):
        """Bucket 10 has cpu=0.6 > p95=0.5 but < p99=0.7 → spike_severe_now = 0."""
        row = series[series["bucket"] == 10].iloc[0]
        assert float(row["spike_severe_now"]) == pytest.approx(0.0)

    def test_cpu_vs_p99_nonnegative(self, series):
        """cpu_vs_p99 = total_cpu / threshold_p99 — always >= 0."""
        assert (series["cpu_vs_p99"] >= 0.0).all()

    def test_cpu_vs_p99_value(self, series):
        """At bucket 20 (cpu=0.8, p99=0.7): cpu_vs_p99 = 0.8 / 0.7 ≈ 1.143."""
        row = series[series["bucket"] == 20].iloc[0]
        assert float(row["cpu_vs_p99"]) == pytest.approx(0.8 / 0.7, rel=1e-3)

    def test_band_position_nonneg(self, series):
        """band_position is clipped to lower=0 — must always be >= 0."""
        assert (series["band_position"] >= 0.0).all()

    def test_band_position_zero_below_p95(self, series):
        """Rows with total_cpu < p95 threshold must have band_position = 0."""
        below_p95 = series[series["total_cpu"] < self._THRESHOLD]
        assert (below_p95["band_position"] == 0.0).all()

    def test_band_width_positive(self, series):
        """band_width = max(p99 - p95, 1e-6) — always > 0."""
        assert (series["band_width"] > 0.0).all()

    def test_band_width_constant_within_machine(self, series):
        """band_width is scalar per machine — all rows should have the same value."""
        vals = series["band_width"].unique()
        assert len(vals) == 1, f"band_width should be constant per machine, got {vals}"

    def test_spike_severe_now_implies_spike_now(self, series):
        """Where spike_severe_now=1, spike_now must also be 1 (p99 > p95)."""
        severe = series[series["spike_severe_now"] == 1.0]
        assert (severe["spike_now"] == 1.0).all(), (
            "spike_severe_now=1 implies spike_now=1 (since p99 > p95)"
        )

    def test_spike_severe_in_last_1_leakage_firewall(self, series):
        """spike_severe_in_last_1 at bucket 20 (the severe spike) must be 0
        because no prior bucket was severe (leakage firewall: shift(1))."""
        row = series[series["bucket"] == 20].iloc[0]
        assert float(row["spike_severe_in_last_1"]) == pytest.approx(0.0)

    def test_spike_severe_in_last_1_after_severe_spike(self, series):
        """Bucket 21 follows the severe spike at 20 → spike_severe_in_last_1 = 1."""
        row = series[series["bucket"] == 21].iloc[0]
        assert float(row["spike_severe_in_last_1"]) == pytest.approx(1.0)

    def test_p99_features_are_float32(self, series):
        """All 8 p99-level features must have dtype float32."""
        for col in (
            "spike_severe_now", "cpu_vs_p99", "peak_cpu_vs_p99",
            "band_position", "band_width",
            "spike_severe_in_last_1", "spike_severe_in_last_3", "spike_severe_in_last_6",
        ):
            assert series[col].dtype == np.float32, f"{col} should be float32"


def test_min_p99_p95_gap_enforced(agg_parquet, tmp_path_factory):
    """Fix 2: after engineer(), threshold_p99 must be >= threshold_p95 * 1.10
    for all machines."""
    out_dir = tmp_path_factory.mktemp("gap_check")
    out_p   = out_dir / "cluster_features.parquet"
    thr_p   = out_dir / "spike_thresholds.parquet"
    engineer(agg_parquet, out_p, thr_p, train_ratio=0.6, horizon=_HORIZON)
    thresh = pd.read_parquet(thr_p)
    for _, row in thresh.iterrows():
        assert float(row["threshold_p99"]) >= float(row["threshold_p95"]) * 1.10 - 1e-6, (
            f"machine {row['machine_id']}: p99={row['threshold_p99']:.4f} < "
            f"p95 * 1.10 = {row['threshold_p95'] * 1.10:.4f}"
        )


# ── cpu_vs_p95: machine-relative normalisation ───────────────────────────────

class TestCpuVsP95:
    """cpu_vs_p95 = total_cpu / machine_p95 threshold."""

    def test_cpu_vs_p95_above_one_when_spiking(self, eng_result):
        """Bucket 6 has total_cpu=0.9 which exceeds the training-data p95."""
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
        df, _, _ = eng_result
        assert (df["cpu_vs_p95"] >= 0.0).all()

    def test_cpu_vs_p95_consistent_with_total_cpu_and_threshold(self, eng_result):
        """cpu_vs_p95 must equal total_cpu / threshold_p95 for every row."""
        df, _, thr_p = eng_result
        thresholds = (
            pd.read_parquet(thr_p)
            .set_index("machine_id")["threshold_p95"]
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


# ── cluster_cpu_p90 training-only fix ─────────────────────────────────────────

class TestClusterP90TrainingOnly:
    """Verify that cluster_cpu_p90 is derived from training buckets only.

    Test scenario: two machines, two periods.

    Training (buckets 1-5, train_max=5):
      Both machines have total_cpu = 0.1 → training cluster p90 ≈ 0.1

    Test (buckets 6-10):
      Both machines have total_cpu = 0.9 → test cluster p90 = 0.9

    After the fix, test-period rows must get the TRAINING p90 (≈ 0.1),
    not the test-period p90 (0.9).
    """

    @pytest.fixture(scope="class")
    def mixed_cluster_df(self):
        """Two machines: low CPU in training, high CPU in test."""
        rows = (
            [_make_agg_row(m, b, total_cpu=0.1) for m in (1, 2) for b in range(1, 6)]
            + [_make_agg_row(m, b, total_cpu=0.9) for m in (1, 2) for b in range(6, 11)]
        )
        return _make_agg_df(rows)

    def test_training_rows_get_training_p90(self, mixed_cluster_df):
        """Training bucket rows must carry the per-bucket training p90 (≈ 0.1)."""
        result     = _add_cluster_features(mixed_cluster_df, train_bucket_max=5)
        train_rows = result[result["bucket"] <= 5]
        # Both machines have total_cpu=0.1 in training → p90 of [0.1, 0.1] = 0.1
        assert (train_rows["cluster_cpu_p90"] < 0.5).all()

    def test_test_rows_get_training_global_p90_not_test_p90(self, mixed_cluster_df):
        """Test bucket rows must NOT get 0.9 (the test-period cluster p90).
        They must get the training-period global p90 (≈ 0.1) as fallback.
        """
        result    = _add_cluster_features(mixed_cluster_df, train_bucket_max=5)
        test_rows = result[result["bucket"] > 5]
        # If the bug were present, test_rows would all have cluster_cpu_p90 ≈ 0.9
        assert (test_rows["cluster_cpu_p90"] < 0.5).all()

    def test_no_nan_in_cluster_p90(self, mixed_cluster_df):
        """cluster_cpu_p90 must be finite for every row — no NaN from the fillna."""
        result = _add_cluster_features(mixed_cluster_df, train_bucket_max=5)
        assert result["cluster_cpu_p90"].notna().all()

    def test_cluster_p90_is_float32(self, mixed_cluster_df):
        result = _add_cluster_features(mixed_cluster_df, train_bucket_max=5)
        assert result["cluster_cpu_p90"].dtype == np.float32
