"""Tests for spike/bootstrap_thresholds.py."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spike.bootstrap_thresholds import (
    _DEFAULT_MIN_BUCKETS,
    _MIN_GAP_FACTOR,
    _QUANTILE_P95,
    _QUANTILE_P99,
    _compute_bootstrap_thresholds,
    main,
    run,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_cluster_agg(
    n_machines:  int   = 10,
    n_buckets:   int   = 2016,
    cpu_scale:   float = 0.5,
    seed:        int   = 42,
) -> pd.DataFrame:
    """Synthetic cluster_agg DataFrame — all required columns, valid values."""
    rng = np.random.default_rng(seed)
    rows = []
    for mid in range(n_machines):
        for b in range(n_buckets):
            cpu = float(rng.exponential(cpu_scale))
            rows.append({
                "machine_id": mid,
                "bucket":     b,
                "time_us":    b * 300_000_000,
                "total_cpu":  cpu,
                "peak_cpu":   cpu * 1.2,
                "total_mem":  float(rng.uniform(0, 0.8)),
                "peak_mem":   float(rng.uniform(0, 0.9)),
                "disk_io":    float(rng.uniform(0, 0.3)),
                "n_tasks":    int(rng.integers(1, 20)),
            })
    return pd.DataFrame(rows)


# ── TestComputeBootstrapThresholds ────────────────────────────────────────────


class TestComputeBootstrapThresholds:

    def test_output_columns(self):
        df = _make_cluster_agg(n_machines=5, n_buckets=2016)
        thresh_df, _ = _compute_bootstrap_thresholds(df, min_buckets_per_machine=100)
        assert list(thresh_df.columns) == ["machine_id", "threshold_p95", "threshold_p99"]

    def test_one_row_per_machine(self):
        df = _make_cluster_agg(n_machines=8, n_buckets=2016)
        thresh_df, _ = _compute_bootstrap_thresholds(df, min_buckets_per_machine=100)
        assert len(thresh_df) == 8
        assert thresh_df["machine_id"].nunique() == 8

    def test_p99_always_above_p95(self):
        df = _make_cluster_agg(n_machines=10, n_buckets=2016)
        thresh_df, _ = _compute_bootstrap_thresholds(df, min_buckets_per_machine=100)
        assert (thresh_df["threshold_p99"] > thresh_df["threshold_p95"]).all()

    def test_minimum_gap_enforced(self):
        """Machines where p99 ≈ p95 (very low variance) get the 10 % gap floor."""
        # All CPU values identical → p95 == p99 → gap must be enforced.
        rows = [{"machine_id": 0, "bucket": b, "total_cpu": 0.5} for b in range(500)]
        df = pd.DataFrame(rows)
        thresh_df, _ = _compute_bootstrap_thresholds(df, min_buckets_per_machine=10)
        p95 = thresh_df.loc[thresh_df["machine_id"] == 0, "threshold_p95"].item()
        p99 = thresh_df.loc[thresh_df["machine_id"] == 0, "threshold_p99"].item()
        assert p99 >= p95 * _MIN_GAP_FACTOR - 1e-9

    def test_sparse_machine_gets_global_fallback(self):
        """Machine 99 has only 5 buckets — must receive global fallback, not NaN."""
        df_rich = _make_cluster_agg(n_machines=5, n_buckets=300)
        sparse_rows = [
            {"machine_id": 99, "bucket": b, "total_cpu": 0.1}
            for b in range(5)
        ]
        df = pd.concat([df_rich, pd.DataFrame(sparse_rows)], ignore_index=True)

        thresh_df, stats = _compute_bootstrap_thresholds(df, min_buckets_per_machine=100)

        assert stats["n_machines_with_global_fallback"] == 1
        machine_99 = thresh_df[thresh_df["machine_id"] == 99]
        assert len(machine_99) == 1
        assert not machine_99["threshold_p95"].isna().any()
        assert not machine_99["threshold_p99"].isna().any()

    def test_all_machines_sparse_uses_global(self):
        """When every machine is sparse, global fallback covers all."""
        df = _make_cluster_agg(n_machines=4, n_buckets=10)
        thresh_df, stats = _compute_bootstrap_thresholds(df, min_buckets_per_machine=100)
        assert stats["n_machines_with_global_fallback"] == 4
        assert stats["n_machines_with_per_machine"] == 0
        assert len(thresh_df) == 4
        # All rows get the same global threshold values.
        assert thresh_df["threshold_p95"].nunique() == 1

    def test_stats_keys_present(self):
        df = _make_cluster_agg(n_machines=3, n_buckets=200)
        _, stats = _compute_bootstrap_thresholds(df, min_buckets_per_machine=50)
        for key in ("n_machines_total", "n_machines_with_per_machine",
                    "n_machines_with_global_fallback", "global_p95", "global_p99"):
            assert key in stats

    def test_global_p95_matches_overall_quantile(self):
        df = _make_cluster_agg(n_machines=5, n_buckets=400)
        _, stats = _compute_bootstrap_thresholds(df, min_buckets_per_machine=10)
        expected = float(df["total_cpu"].quantile(_QUANTILE_P95))
        assert abs(stats["global_p95"] - expected) < 1e-9


# ── TestRunFunction ───────────────────────────────────────────────────────────


class TestRunFunction:

    def test_writes_thresholds_parquet(self, tmp_path):
        df = _make_cluster_agg(n_machines=4, n_buckets=2016)
        csv_path = tmp_path / "agg.csv"
        df.to_csv(csv_path, index=False)

        run(csv_path, tmp_path / "domain", min_buckets_per_machine=100)

        parquet_path = tmp_path / "domain" / "spike_thresholds.parquet"
        assert parquet_path.exists()
        result = pd.read_parquet(parquet_path)
        assert list(result.columns) == ["machine_id", "threshold_p95", "threshold_p99"]
        assert len(result) == 4

    def test_writes_bootstrap_meta_json(self, tmp_path):
        df = _make_cluster_agg(n_machines=3, n_buckets=2016)
        csv_path = tmp_path / "agg.csv"
        df.to_csv(csv_path, index=False)

        run(csv_path, tmp_path / "domain", min_buckets_per_machine=100)

        meta_path = tmp_path / "domain" / "bootstrap_meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        for key in ("bootstrapped_at", "source_path", "min_buckets_per_machine",
                    "quantile_p95", "quantile_p99", "global_p95", "global_p99"):
            assert key in meta

    def test_parquet_input(self, tmp_path):
        df = _make_cluster_agg(n_machines=3, n_buckets=2016)
        pq_path = tmp_path / "agg.parquet"
        df.to_parquet(pq_path, index=False)

        run(pq_path, tmp_path / "domain", min_buckets_per_machine=100)

        assert (tmp_path / "domain" / "spike_thresholds.parquet").exists()

    def test_creates_domain_dir_if_absent(self, tmp_path):
        df = _make_cluster_agg(n_machines=2, n_buckets=300)
        csv_path = tmp_path / "agg.csv"
        df.to_csv(csv_path, index=False)

        nested = tmp_path / "a" / "b" / "c"
        assert not nested.exists()
        run(csv_path, nested, min_buckets_per_machine=10)
        assert (nested / "spike_thresholds.parquet").exists()


# ── TestCLI ───────────────────────────────────────────────────────────────────


class TestCLI:

    def test_csv_input_end_to_end(self, tmp_path):
        df = _make_cluster_agg(n_machines=3, n_buckets=2016)
        csv_path = tmp_path / "agg.csv"
        df.to_csv(csv_path, index=False)
        domain = tmp_path / "domain"

        main(["--input", str(csv_path), "--domain-dir", str(domain)])

        assert (domain / "spike_thresholds.parquet").exists()
        assert (domain / "bootstrap_meta.json").exists()

    def test_missing_input_exits_1(self, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            main(["--input", str(tmp_path / "nonexistent.csv"),
                  "--domain-dir", str(tmp_path / "domain")])
        assert exc_info.value.code == 1

    def test_invalid_min_buckets_exits_1(self, tmp_path):
        df = _make_cluster_agg(n_machines=2, n_buckets=100)
        csv_path = tmp_path / "agg.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(SystemExit) as exc_info:
            main(["--input", str(csv_path),
                  "--domain-dir", str(tmp_path / "domain"),
                  "--min-buckets", "0"])
        assert exc_info.value.code == 1

    def test_missing_required_column_exits_1(self, tmp_path):
        df = _make_cluster_agg(n_machines=2, n_buckets=100)
        df = df.drop(columns=["total_cpu"])
        csv_path = tmp_path / "bad.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(SystemExit) as exc_info:
            main(["--input", str(csv_path),
                  "--domain-dir", str(tmp_path / "domain")])
        assert exc_info.value.code == 1

    def test_custom_min_buckets(self, tmp_path):
        """--min-buckets is respected when flagging sparse machines."""
        n_rich, n_sparse = 4, 2
        df_rich   = _make_cluster_agg(n_machines=n_rich,   n_buckets=50, seed=1)
        df_sparse = _make_cluster_agg(n_machines=n_sparse, n_buckets=10, seed=2)
        # Give sparse machines different IDs.
        df_sparse["machine_id"] += n_rich
        df = pd.concat([df_rich, df_sparse], ignore_index=True)

        csv_path = tmp_path / "agg.csv"
        df.to_csv(csv_path, index=False)
        domain = tmp_path / "domain"

        main(["--input", str(csv_path),
              "--domain-dir", str(domain),
              "--min-buckets", "20"])

        meta = json.loads((domain / "bootstrap_meta.json").read_text())
        assert meta["n_machines_with_global_fallback"] == n_sparse
        assert meta["n_machines_with_per_machine"] == n_rich
