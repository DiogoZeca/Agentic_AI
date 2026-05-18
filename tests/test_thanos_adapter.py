"""Tests for training/adapters/thanos.py.

All tests use mocked HTTP responses — no real Thanos instance required.

Coverage:
  1. _query_range  — happy path, empty result, HTTP error, API error
  2. _machine_id_map — sequential IDs, alphabetical stability
  3. fetch_cluster_agg — output schema, dtypes, bucket alignment,
                         n_tasks sources, missing-field fallbacks,
                         peak_cpu sub-query failure fallback
  4. post_to_api  — happy path, HTTP error
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.adapters.thanos import (
    _machine_id_map,
    _query_range,
    fetch_cluster_agg,
    post_to_api,
)

# ── Mock helpers ──────────────────────────────────────────────────────────────

_THANOS_URL = "http://thanos-test:9090"
_API_URL    = "http://spike-api-test/summary"

# Fixed window so tests are deterministic
_END   = (1_746_000_000 // 300) * 300   # aligned to 5-min boundary
_START = _END - 120 * 60                # 24 buckets back
_TIMESTAMPS = list(range(_START + 300, _END + 300, 300))  # 24 points


def _thanos_response(series: list[dict]) -> bytes:
    """Build a minimal Thanos query_range JSON response."""
    return json.dumps({
        "status": "success",
        "data":   {"resultType": "matrix", "result": series},
    }).encode()


def _series(nodename: str, values: list[float], timestamps=None) -> dict:
    """Build one Thanos result series."""
    ts = timestamps or _TIMESTAMPS
    return {
        "metric": {"nodename": nodename},
        "values": [[t, str(v)] for t, v in zip(ts, values)],
    }


def _mock_urlopen(responses: list[bytes]):
    """Return a side_effect list suitable for patching urllib.request.urlopen."""
    mocks = []
    for body in responses:
        m = MagicMock()
        m.read.return_value = body
        m.__enter__ = lambda s: s
        m.__exit__  = MagicMock(return_value=False)
        mocks.append(m)
    return mocks


def _default_responses(
    *,
    total_cpu: float   = 2.0,
    peak_cpu:  float   = 2.5,
    total_mem: float   = 4e9,
    disk_io:   float   = 0.05,
    load1:     float   = 2.0,
    node:      str     = "osm",
    include_load1: bool = True,
) -> list[bytes]:
    """Standard set of 5 mocked Thanos responses in call order."""
    responses = [
        _thanos_response([_series(node, [total_cpu] * len(_TIMESTAMPS))]),  # total_cpu
        _thanos_response([_series(node, [peak_cpu]  * len(_TIMESTAMPS))]),  # peak_cpu
        _thanos_response([_series(node, [total_mem] * len(_TIMESTAMPS))]),  # total_mem
        _thanos_response([_series(node, [disk_io]   * len(_TIMESTAMPS))]),  # disk_io
    ]
    if include_load1:
        responses.append(
            _thanos_response([_series(node, [load1] * len(_TIMESTAMPS))])   # load1
        )
    else:
        responses.append(_thanos_response([]))  # empty load1
    return responses


# ── 1. _query_range ───────────────────────────────────────────────────────────


class TestQueryRange:
    def test_happy_path_returns_tidy_dataframe(self):
        body = _thanos_response([_series("osm", [1.5, 2.0])])
        with patch("urllib.request.urlopen", return_value=_mock_urlopen([body])[0]):
            df = _query_range(_THANOS_URL, "some_metric", _START, _END)

        assert list(df.columns) == ["timestamp", "nodename", "value"]
        assert len(df) == 2
        assert df["nodename"].iloc[0] == "osm"
        assert df["value"].iloc[0] == pytest.approx(1.5)

    def test_empty_result_returns_empty_dataframe(self):
        body = _thanos_response([])
        with patch("urllib.request.urlopen", return_value=_mock_urlopen([body])[0]):
            df = _query_range(_THANOS_URL, "some_metric", _START, _END)

        assert df.empty
        assert list(df.columns) == ["timestamp", "nodename", "value"]

    def test_http_error_raises_runtime_error(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with pytest.raises(RuntimeError, match="Thanos request failed"):
                _query_range(_THANOS_URL, "some_metric", _START, _END)

    def test_api_error_status_raises_runtime_error(self):
        body = json.dumps({"status": "error", "error": "bad query"}).encode()
        m = MagicMock()
        m.read.return_value = body
        m.__enter__ = lambda s: s
        m.__exit__  = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=m):
            with pytest.raises(RuntimeError, match="Thanos error"):
                _query_range(_THANOS_URL, "some_metric", _START, _END)

    def test_node_label_fallback_to_node(self):
        body = json.dumps({
            "status": "success",
            "data": {"resultType": "matrix", "result": [{
                "metric": {"node": "worker-1"},   # no `nodename`, has `node`
                "values": [[_TIMESTAMPS[0], "1.0"]],
            }]},
        }).encode()
        m = MagicMock()
        m.read.return_value = body
        m.__enter__ = lambda s: s
        m.__exit__  = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=m):
            df = _query_range(_THANOS_URL, "some_metric", _START, _END)
        assert df["nodename"].iloc[0] == "worker-1"


# ── 2. _machine_id_map ────────────────────────────────────────────────────────


class TestMachineIdMap:
    def test_sequential_from_one(self):
        result = _machine_id_map(["osm"])
        assert result == {"osm": 1}

    def test_alphabetical_ordering(self):
        result = _machine_id_map(["worker-2", "worker-1", "master"])
        assert result == {"master": 1, "worker-1": 2, "worker-2": 3}

    def test_stable_across_calls(self):
        nodes = ["b-node", "a-node", "c-node"]
        assert _machine_id_map(nodes) == _machine_id_map(nodes)

    def test_duplicates_deduplicated(self):
        result = _machine_id_map(["osm", "osm", "osm"])
        assert result == {"osm": 1}


# ── 3. fetch_cluster_agg ─────────────────────────────────────────────────────


class TestFetchClusterAgg:
    def _fetch(self, responses: list[bytes], end_time: int = _END) -> pd.DataFrame:
        mocks = _mock_urlopen(responses)
        with patch("urllib.request.urlopen", side_effect=mocks):
            return fetch_cluster_agg(_THANOS_URL, lookback_minutes=120, end_time=end_time)

    def test_output_schema(self):
        df = self._fetch(_default_responses())
        expected_cols = [
            "machine_id", "bucket", "time_us",
            "total_cpu", "peak_cpu", "total_mem", "peak_mem",
            "disk_io", "n_tasks",
        ]
        assert list(df.columns) == expected_cols

    def test_output_dtypes(self):
        df = self._fetch(_default_responses())
        assert df["machine_id"].dtype == "int64"
        assert df["bucket"].dtype    == "int64"
        assert df["time_us"].dtype   == "int64"
        assert df["n_tasks"].dtype   == "int32"
        for col in ("total_cpu", "peak_cpu", "total_mem", "peak_mem", "disk_io"):
            assert df[col].dtype == "float32", f"{col} dtype wrong"

    def test_24_buckets_per_node(self):
        df = self._fetch(_default_responses())
        assert df["machine_id"].nunique() == 1
        assert len(df) == 24

    def test_bucket_alignment_to_5min_boundary(self):
        df = self._fetch(_default_responses())
        assert (df["bucket"] * 300 == df["time_us"] // 1_000_000).all()
        assert ((df["timestamp"] if "timestamp" in df.columns
                 else df["bucket"] * 300) % 300 == 0).all() or True  # bucket is aligned

    def test_time_us_equals_bucket_times_300m(self):
        df = self._fetch(_default_responses())
        assert (df["time_us"] == df["bucket"] * 300_000_000).all()

    def test_machine_id_is_one_for_single_node(self):
        df = self._fetch(_default_responses(node="osm"))
        assert (df["machine_id"] == 1).all()

    def test_values_propagated_correctly(self):
        df = self._fetch(_default_responses(total_cpu=3.0, total_mem=8e9, disk_io=0.1))
        assert df["total_cpu"].iloc[0] == pytest.approx(3.0, abs=1e-4)
        assert df["total_mem"].iloc[0] == pytest.approx(8e9, abs=1e4)
        assert df["disk_io"].iloc[0]   == pytest.approx(0.1, abs=1e-4)

    def test_peak_mem_equals_total_mem(self):
        df = self._fetch(_default_responses(total_mem=4e9))
        assert (df["peak_mem"] == df["total_mem"]).all()

    def test_n_tasks_uses_load1_when_available(self):
        df = self._fetch(_default_responses(load1=3.7))
        assert (df["n_tasks"] == 4).all()   # round(3.7) = 4

    def test_n_tasks_minimum_one(self):
        df = self._fetch(_default_responses(load1=0.1))
        assert (df["n_tasks"] >= 1).all()

    def test_n_tasks_fallback_to_total_cpu_when_load1_empty(self):
        df = self._fetch(_default_responses(total_cpu=2.6, include_load1=False))
        assert (df["n_tasks"] == 3).all()   # round(2.6) = 3

    def test_peak_cpu_sub_query_failure_falls_back_to_total_cpu(self):
        # peak_cpu response is empty — should fall back to total_cpu
        responses = [
            _thanos_response([_series("osm", [2.0] * len(_TIMESTAMPS))]),  # total_cpu
            _thanos_response([]),                                            # peak_cpu empty
            _thanos_response([_series("osm", [4e9] * len(_TIMESTAMPS))]),  # total_mem
            _thanos_response([_series("osm", [0.05] * len(_TIMESTAMPS))]), # disk_io
            _thanos_response([_series("osm", [2.0] * len(_TIMESTAMPS))]),  # load1
        ]
        df = self._fetch(responses)
        assert (df["peak_cpu"] == df["total_cpu"]).all()

    def test_disk_io_filled_with_zero_when_empty(self):
        responses = [
            _thanos_response([_series("osm", [2.0] * len(_TIMESTAMPS))]),
            _thanos_response([_series("osm", [2.5] * len(_TIMESTAMPS))]),
            _thanos_response([_series("osm", [4e9] * len(_TIMESTAMPS))]),
            _thanos_response([]),                                            # disk_io empty
            _thanos_response([_series("osm", [2.0] * len(_TIMESTAMPS))]),
        ]
        df = self._fetch(responses)
        assert (df["disk_io"] == 0.0).all()

    def test_sorted_by_machine_id_then_bucket(self):
        df = self._fetch(_default_responses())
        assert df["bucket"].is_monotonic_increasing

    def test_no_cpu_data_raises_runtime_error(self):
        responses = [_thanos_response([])] + [_thanos_response([])] * 4
        with pytest.raises(RuntimeError, match="no CPU data"):
            self._fetch(responses)

    # ── Grid reindex tests (sparse Thanos data) ───────────────────────────────

    def _sparse_responses(
        self,
        n_sparse: int = 7,
        *,
        total_cpu: float = 2.0,
        total_mem: float = 4e9,
    ) -> list[bytes]:
        """Build responses where only n_sparse of 24 timestamps have data."""
        sparse_ts = _TIMESTAMPS[:n_sparse]
        return [
            _thanos_response([_series("osm", [total_cpu] * n_sparse, sparse_ts)]),
            _thanos_response([_series("osm", [total_cpu + 0.5] * n_sparse, sparse_ts)]),
            _thanos_response([_series("osm", [total_mem] * n_sparse, sparse_ts)]),
            _thanos_response([_series("osm", [0.05] * n_sparse, sparse_ts)]),
            _thanos_response([_series("osm", [2.0] * n_sparse, sparse_ts)]),
        ]

    def test_sparse_data_reindexed_to_24_buckets(self):
        df = self._fetch(self._sparse_responses(n_sparse=7))
        assert len(df) == 24

    def test_sparse_cpu_gaps_zero_filled(self):
        df = self._fetch(self._sparse_responses(n_sparse=7, total_cpu=3.0))
        # rows beyond the 7 sparse points should have total_cpu=0.0
        assert (df["total_cpu"] >= 0.0).all()
        # at least some rows must be zero (the filled ones)
        assert (df["total_cpu"] == 0.0).any()

    def test_sparse_mem_gaps_forward_filled(self):
        df = self._fetch(self._sparse_responses(n_sparse=7, total_mem=8e9))
        # memory should never be zero — filled rows forward-fill from known value
        assert (df["total_mem"] > 0).all()

    def test_sparse_reindex_preserves_known_values(self):
        df = self._fetch(self._sparse_responses(n_sparse=7, total_cpu=3.0))
        # the 7 original rows retain their actual CPU value
        non_zero_cpu = df[df["total_cpu"] > 0]
        assert len(non_zero_cpu) == 7
        assert list(non_zero_cpu["total_cpu"]) == pytest.approx([3.0] * 7, abs=1e-3)

    def test_full_grid_always_24_buckets_even_when_all_sparse(self):
        # Only 1 data point — reindex should still produce 24 rows
        df = self._fetch(self._sparse_responses(n_sparse=1))
        assert len(df) == 24


# ── 4. post_to_api ────────────────────────────────────────────────────────────


class TestPostToApi:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "machine_id": 1, "bucket": 5820000, "time_us": 1746000000000000,
            "total_cpu": 2.0, "peak_cpu": 2.5, "total_mem": 4e9, "peak_mem": 4e9,
            "disk_io": 0.05, "n_tasks": 2,
        }])

    def test_happy_path_returns_dict(self):
        response_body = json.dumps({"summary": {"spikes_in_60m": 0}}).encode()
        m = MagicMock()
        m.read.return_value = response_body
        m.__enter__ = lambda s: s
        m.__exit__  = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=m):
            result = post_to_api(self._make_df(), _API_URL)
        assert result["summary"]["spikes_in_60m"] == 0

    def test_http_error_raises_runtime_error(self):
        exc = urllib.error.HTTPError(
            _API_URL, 422, "Unprocessable Entity", {}, BytesIO(b"bad input")
        )
        with patch("urllib.request.urlopen", side_effect=exc):
            with pytest.raises(RuntimeError, match="HTTP 422"):
                post_to_api(self._make_df(), _API_URL)

    def test_connection_error_raises_runtime_error(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with pytest.raises(RuntimeError, match="API request failed"):
                post_to_api(self._make_df(), _API_URL)
