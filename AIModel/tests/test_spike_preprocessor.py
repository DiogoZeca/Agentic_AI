"""Tests for spike_preprocessor.py — Step 1 of the CPU spike prediction pipeline.

Synthetic CSV has 10 rows split into two chunks (chunksize=5) to exercise the
two-pass boundary-merge logic without touching real data.

Row layout
----------
Chunk 0 (rows 0-4):
  Row 0  machine=1, bucket=2, cpu_rate=0.30, duration=300 M µs
  Row 1  machine=1, bucket=2, cpu_rate=0.20, duration=300 M µs
  Row 2  machine=2, bucket=2, cpu_rate=0.40, duration=300 M µs
  Row 3  machine=2, bucket=3, cpu_rate=1.00, duration=300 M µs  (cpu_rate > 1 — valid)
  Row 4  machine=3, bucket=4, cpu_rate=0.50, duration=300 M µs

Chunk 1 (rows 5-9):
  Row 5  machine=1, bucket=2, cpu_rate=0.60, duration=2 M µs   (2-second boundary artefact)
  Row 6  machine=1, bucket=5, cpu_rate=1.50, duration=300 M µs (multi-core, valid)
  Row 7  machine=6_301_942_525, bucket=6, cpu_rate=0.30, duration=300 M µs  (large id)
  Row 8  machine=5, bucket=7, cpu_rate=100.0, duration=300 M µs  → filtered out
  Row 9  machine=5, bucket=7, cpu_rate=0.10, duration=300 M µs

Key expected values (verified by hand):
  (m=1, bkt=2):  total_cpu = (0.3×300M + 0.2×300M + 0.6×2M) / 300M
                           = (90M + 60M + 1.2M) / 300M ≈ 0.504
                 peak_cpu  = max(0.5, 0.9, 0.8) = 0.9
                 n_tasks   = 3
  (m=2, bkt=2):  total_cpu = 0.4 × 300M / 300M = 0.4
  (m=2, bkt=3):  total_cpu = 1.0 × 300M / 300M = 1.0
  (m=3, bkt=4):  total_cpu = 0.5
  (m=1, bkt=5):  total_cpu = 1.5  (multi-core row kept)
  (m=6301942525, bkt=6):  total_cpu = 0.3  (large machine_id stored as int64)
  (m=5, bkt=7):  total_cpu = 0.1  (cpu_rate=100 row filtered; only row 9 survives)
"""

from __future__ import annotations

import io
import sys
import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike_preprocessor import (
    _BUCKET_US,
    _OUTPUT_COLS,
    _PARTIAL_COLS,
    _process_chunk,
    _merge_partials,
    _validate_schema,
    preprocess,
)

# ── Synthetic CSV ──────────────────────────────────────────────────────────────

_CSV_CONTENT = textwrap.dedent("""\
    start_time,end_time,machine_id,cpu_rate,max_cpu_rate,canonical_mem_usage,max_mem_usage,mean_disk_io_time,sample_portion
    600000000,900000000,1,0.3,0.5,0.10,0.20,0.010,0.0
    600000000,900000000,1,0.2,0.9,0.20,0.30,0.020,0.0
    600000000,900000000,2,0.4,0.6,0.15,0.25,0.010,0.0
    900000000,1200000000,2,1.0,1.2,0.30,0.40,0.030,0.0
    1200000000,1500000000,3,0.5,0.7,0.10,0.20,0.010,0.0
    600000000,602000000,1,0.6,0.8,0.05,0.10,0.005,0.0
    1500000000,1800000000,1,1.5,2.0,0.40,0.50,0.040,0.0
    1800000000,2100000000,6301942525,0.3,0.4,0.10,0.20,0.010,0.0
    2100000000,2400000000,5,100.0,120.0,0.50,0.60,0.050,0.0
    2100000000,2400000000,5,0.1,0.2,0.10,0.20,0.010,0.0
""")

_CHUNKSIZE_SMALL = 5   # forces boundary split at row 5


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory) -> Path:
    """Write the synthetic CSV to a temp file once for the whole module."""
    p = tmp_path_factory.mktemp("data") / "test_cluster.csv"
    p.write_text(_CSV_CONTENT)
    return p


@pytest.fixture(scope="module")
def output_path(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("out") / "cluster_agg.parquet"


@pytest.fixture(scope="module")
def result(csv_path, output_path) -> pd.DataFrame:
    """Run the full pipeline once; all tests share this output."""
    return preprocess(csv_path, output_path, chunksize=_CHUNKSIZE_SMALL)


# ── Schema validation ─────────────────────────────────────────────────────────

class TestValidateSchema:
    def test_valid_csv_passes(self, csv_path):
        _validate_schema(csv_path)   # must not raise

    def test_missing_column_raises(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("start_time,machine_id,cpu_rate\n")
        with pytest.raises(ValueError, match="missing expected columns"):
            _validate_schema(bad)


# ── _process_chunk ────────────────────────────────────────────────────────────

class TestProcessChunk:
    @pytest.fixture(scope="class")
    def raw_df(self) -> pd.DataFrame:
        return pd.read_csv(io.StringIO(_CSV_CONTENT))

    def test_cpu_rate_filter_removes_high_values(self, raw_df):
        """cpu_rate=100 row must be filtered out; cpu_rate=1.5 must survive."""
        chunk = raw_df.iloc[5:10].copy()
        partial = _process_chunk(chunk)
        # machine=5 should appear with only 1 task (cpu_rate=0.1 survives)
        m5 = partial[partial["machine_id"] == 5]
        assert len(m5) == 1
        assert int(m5["n_tasks"].iloc[0]) == 1

    def test_multi_core_cpu_rate_kept(self, raw_df):
        """cpu_rate=1.5 (multi-core) must not be filtered — check bucket=5 row exists."""
        chunk = raw_df.iloc[5:10].copy()
        partial = _process_chunk(chunk)
        m1_bkt5 = partial[(partial["machine_id"] == 1) & (partial["bucket"] == 5)]
        assert len(m1_bkt5) == 1
        assert m1_bkt5["n_tasks"].iloc[0] == 1

    def test_returns_partial_cols(self, raw_df):
        partial = _process_chunk(raw_df.iloc[0:5].copy())
        assert set(partial.columns) == set(_PARTIAL_COLS)

    def test_empty_chunk_after_filter(self, raw_df):
        """A chunk where all rows fail the cpu_rate filter returns empty DataFrame."""
        all_bad = raw_df.copy()
        all_bad["cpu_rate"] = 999.0
        partial = _process_chunk(all_bad)
        assert partial.empty
        assert list(partial.columns) == _PARTIAL_COLS


# ── Full pipeline output ──────────────────────────────────────────────────────

class TestOutputColumns:
    def test_output_columns_match_spec(self, result):
        assert list(result.columns) == _OUTPUT_COLS

    def test_sorted_by_machine_bucket(self, result):
        assert result["machine_id"].is_monotonic_increasing or (
            result[["machine_id", "bucket"]]
            .equals(result[["machine_id", "bucket"]].sort_values(["machine_id", "bucket"]).reset_index(drop=True))
        )


class TestOutputTypes:
    def test_machine_id_is_int64(self, result):
        assert result["machine_id"].dtype == np.int64

    def test_bucket_is_int64(self, result):
        assert result["bucket"].dtype == np.int64

    def test_time_us_is_int64(self, result):
        assert result["time_us"].dtype == np.int64

    def test_n_tasks_is_int32(self, result):
        assert result["n_tasks"].dtype == np.int32

    def test_float_cols_are_float32(self, result):
        for col in ("total_cpu", "peak_cpu", "total_mem", "peak_mem", "disk_io"):
            assert result[col].dtype == np.float32, f"{col} should be float32"


class TestOutputValues:
    def _row(self, result, machine_id, bucket):
        mask = (result["machine_id"] == machine_id) & (result["bucket"] == bucket)
        rows = result[mask]
        assert len(rows) == 1, f"Expected exactly 1 row for machine={machine_id} bucket={bucket}"
        return rows.iloc[0]

    def test_boundary_merge_total_cpu(self, result):
        """(m=1, bkt=2) spans both chunks — total_cpu must reflect all 3 tasks."""
        row = self._row(result, 1, 2)
        # 0.3×300M + 0.2×300M + 0.6×2M = 151.2M → / 300M = 0.504
        assert abs(float(row["total_cpu"]) - 0.504) < 1e-4

    def test_boundary_merge_n_tasks(self, result):
        """(m=1, bkt=2) must count all 3 tasks from both chunks."""
        row = self._row(result, 1, 2)
        assert int(row["n_tasks"]) == 3

    def test_boundary_merge_peak_cpu(self, result):
        """peak_cpu for (m=1, bkt=2) = max(0.5, 0.9, 0.8) = 0.9."""
        row = self._row(result, 1, 2)
        assert abs(float(row["peak_cpu"]) - 0.9) < 1e-4

    def test_single_window_total_cpu(self, result):
        """(m=2, bkt=3) has one full-window task at cpu_rate=1.0 → total_cpu=1.0."""
        row = self._row(result, 2, 3)
        assert abs(float(row["total_cpu"]) - 1.0) < 1e-4

    def test_multi_core_row_kept(self, result):
        """(m=1, bkt=5) cpu_rate=1.5 is valid and must appear in output."""
        row = self._row(result, 1, 5)
        assert abs(float(row["total_cpu"]) - 1.5) < 1e-4

    def test_filtered_row_absent(self, result):
        """cpu_rate=100 row must be gone; (m=5, bkt=7) has only 1 surviving task."""
        row = self._row(result, 5, 7)
        assert int(row["n_tasks"]) == 1
        assert abs(float(row["total_cpu"]) - 0.1) < 1e-4

    def test_large_machine_id_stored_correctly(self, result):
        """machine_id=6_301_942_525 exceeds int32 max; must be stored as int64."""
        large_id = 6_301_942_525
        assert large_id in result["machine_id"].values
        row = self._row(result, large_id, 6)
        assert int(row["machine_id"]) == large_id

    def test_time_us_matches_bucket(self, result):
        """time_us must equal bucket × _BUCKET_US for every row."""
        expected = result["bucket"] * _BUCKET_US
        assert (result["time_us"] == expected).all()

    def test_disk_io_uses_max(self, result):
        """(m=1, bkt=2) disk_io = max(0.010, 0.020, 0.005) = 0.020."""
        mask = (result["machine_id"] == 1) & (result["bucket"] == 2)
        disk_io = float(result.loc[mask, "disk_io"].iloc[0])
        assert abs(disk_io - 0.020) < 1e-5


# ── Parquet output ────────────────────────────────────────────────────────────

class TestParquetOutput:
    def test_parquet_file_written(self, output_path):
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_parquet_roundtrip_matches(self, result, output_path):
        loaded = pd.read_parquet(output_path)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), loaded)


# ── Error handling ────────────────────────────────────────────────────────────

class TestErrors:
    def test_missing_input_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            preprocess(tmp_path / "nonexistent.csv", tmp_path / "out.parquet")

    def test_all_filtered_returns_empty_parquet(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text(
            "start_time,end_time,machine_id,cpu_rate,max_cpu_rate,"
            "canonical_mem_usage,max_mem_usage,mean_disk_io_time,sample_portion\n"
            "600000000,900000000,1,200.0,200.0,0.1,0.2,0.01,0.0\n"
        )
        out = tmp_path / "empty.parquet"
        result = preprocess(bad_csv, out)
        assert result.empty
        assert out.exists()
        loaded = pd.read_parquet(out)
        assert list(loaded.columns) == _OUTPUT_COLS
