"""Tests for daemon SLO metrics writing.

Three layers of coverage:
  1. Unit — _extract_cycle_stats: alarm counting logic.
  2. Unit — _record_slo / _write_slo: deque management and file schema.
  3. Integration — run() in one-shot mode: full cycle produces slo_metrics.json.

All tests use synthetic data and the fake model directory from helpers.py;
no real training artefacts are required.
"""
from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike.daemon import (
    _extract_cycle_stats,
    _record_slo,
    _write_slo,
    run,
)
from helpers import make_fake_artifacts, make_window_df


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_result(
    machines_total:      int = 10,
    machines_predicted:  int = 9,
    machines_cold_start: int = 1,
    alarm_actions: list[str] | None = None,
) -> dict:
    """Synthetic prediction result dict for unit tests."""
    alarm_actions = alarm_actions or []
    preds = [
        {"machine_id": i, "recommended_action": act}
        for i, act in enumerate(alarm_actions)
    ]
    # Pad with "normal" entries so machine counts are consistent.
    n_padded = machines_predicted - len(alarm_actions)
    preds += [
        {"machine_id": 100 + i, "recommended_action": "normal"}
        for i in range(max(0, n_padded))
    ]
    return {
        "machines_total":      machines_total,
        "machines_predicted":  machines_predicted,
        "machines_cold_start": machines_cold_start,
        "predictions":         preds,
    }


def _make_deque(*cycle_records: dict) -> collections.deque:
    """Wrap dicts in a deque(maxlen=300) for write tests."""
    d = collections.deque(maxlen=300)
    d.extend(cycle_records)
    return d


# ── 1. _extract_cycle_stats ───────────────────────────────────────────────────


class TestExtractCycleStats:
    def test_alarm_count_matches_non_passive_actions(self):
        result = _make_result(
            machines_predicted=5,
            alarm_actions=["preempt_now", "migrate_jobs", "normal"],
        )
        stats = _extract_cycle_stats(result)
        # "preempt_now" and "migrate_jobs" are alarms; "normal" is not.
        assert stats["alarm_count"] == 2

    def test_passive_actions_not_counted(self):
        """None, "normal", and "monitor" must never increment alarm_count."""
        result = _make_result(
            machines_predicted=3,
            alarm_actions=["normal", "monitor"],
        )
        stats = _extract_cycle_stats(result)
        assert stats["alarm_count"] == 0

    def test_alarm_rate_is_alarm_count_over_predicted(self):
        result = _make_result(
            machines_predicted=10,
            alarm_actions=["preempt_now", "defer_batch"],
        )
        stats = _extract_cycle_stats(result)
        assert abs(stats["alarm_rate"] - 2 / 10) < 1e-9

    def test_alarm_rate_none_when_machines_predicted_zero(self):
        """Avoid divide-by-zero when all machines are cold-start."""
        result = _make_result(machines_total=5, machines_predicted=0, machines_cold_start=5)
        result["predictions"] = []
        stats = _extract_cycle_stats(result)
        assert stats["alarm_rate"] is None

    def test_cold_start_rate_computed_correctly(self):
        result = _make_result(machines_total=10, machines_cold_start=2, machines_predicted=8)
        stats = _extract_cycle_stats(result)
        assert abs(stats["cold_start_rate"] - 0.2) < 1e-9

    def test_cold_start_rate_none_when_machines_total_zero(self):
        result = {"machines_total": 0, "machines_predicted": 0,
                  "machines_cold_start": 0, "predictions": []}
        stats = _extract_cycle_stats(result)
        assert stats["cold_start_rate"] is None

    def test_all_expected_keys_present(self):
        result = _make_result()
        stats  = _extract_cycle_stats(result)
        for key in ("machines_total", "machines_predicted", "machines_cold_start",
                    "alarm_count", "alarm_rate", "cold_start_rate", "alarm_list"):
            assert key in stats, f"missing key: {key}"


# ── 2. _record_slo and _write_slo ────────────────────────────────────────────


class TestRecordSlo:
    def test_successful_cycle_appended(self):
        d      = collections.deque(maxlen=300)
        result = _make_result(machines_predicted=5, alarm_actions=["preempt_now"])
        _record_slo(d, cycle_index=0, elapsed=1.5, result=result)

        assert len(d) == 1
        rec = d[0]
        assert rec["cycle_index"]  == 0
        assert rec["success"]      is True
        assert rec["latency_s"]    == 1.5
        assert rec["alarm_count"]  == 1
        assert rec["alarm_rate"]   is not None

    def test_skipped_cycle_success_false(self):
        d = collections.deque(maxlen=300)
        _record_slo(d, cycle_index=1, elapsed=0.3, result=None)

        rec = d[0]
        assert rec["success"]            is False
        assert rec["alarm_count"]        is None
        assert rec["alarm_rate"]         is None
        assert rec["cold_start_rate"]    is None
        assert rec["machines_predicted"] is None

    def test_deque_evicts_oldest_at_maxlen(self):
        d = collections.deque(maxlen=3)
        for i in range(4):
            _record_slo(d, cycle_index=i, elapsed=float(i), result=None)

        assert len(d) == 3
        # The oldest (cycle_index=0) was evicted; first remaining is index=1.
        assert d[0]["cycle_index"] == 1

    def test_timestamp_is_iso8601_string(self):
        d = collections.deque(maxlen=300)
        _record_slo(d, cycle_index=0, elapsed=1.0, result=None)
        ts = d[0]["timestamp"]
        assert isinstance(ts, str)
        assert "T" in ts  # ISO 8601 separator


class TestWriteSlo:
    def test_file_created_with_required_keys(self, tmp_path):
        slo_path = tmp_path / "slo_metrics.json"
        d = _make_deque(
            {"cycle_index": 0, "timestamp": "2026-01-01T00:00:00+00:00",
             "latency_s": 1.2, "success": True,
             "machines_total": 10, "machines_predicted": 9, "machines_cold_start": 1,
             "alarm_count": 2, "alarm_rate": 0.222, "cold_start_rate": 0.1},
        )
        _write_slo(slo_path, d, model_version=None)

        assert slo_path.exists()
        data = json.loads(slo_path.read_text())
        for key in ("model_version", "summary", "cycles"):
            assert key in data, f"missing top-level key: {key}"

    def test_summary_has_required_keys(self, tmp_path):
        slo_path = tmp_path / "slo_metrics.json"
        d = _make_deque(
            {"cycle_index": 0, "timestamp": "2026-01-01T00:00:00+00:00",
             "latency_s": 1.0, "success": True,
             "machines_total": 5, "machines_predicted": 5, "machines_cold_start": 0,
             "alarm_count": 1, "alarm_rate": 0.2, "cold_start_rate": 0.0},
        )
        _write_slo(slo_path, d, model_version=None)
        summary = json.loads(slo_path.read_text())["summary"]

        for key in ("window_cycles", "success_rate", "latency_p50_s", "latency_p95_s",
                    "alarm_rate_mean", "alarm_rate_std", "cold_start_rate_mean"):
            assert key in summary, f"missing summary key: {key}"

    def test_single_cycle_std_is_none(self, tmp_path):
        """alarm_rate_std must be None (not 0.0) when there is only one data point."""
        slo_path = tmp_path / "slo_metrics.json"
        d = _make_deque(
            {"cycle_index": 0, "timestamp": "2026-01-01T00:00:00+00:00",
             "latency_s": 1.0, "success": True,
             "machines_total": 5, "machines_predicted": 5, "machines_cold_start": 0,
             "alarm_count": 1, "alarm_rate": 0.2, "cold_start_rate": 0.0},
        )
        _write_slo(slo_path, d, model_version=None)
        assert json.loads(slo_path.read_text())["summary"]["alarm_rate_std"] is None

    def test_two_cycles_std_is_populated(self, tmp_path):
        slo_path = tmp_path / "slo_metrics.json"
        d = _make_deque(
            {"cycle_index": 0, "timestamp": "2026-01-01T00:00:00+00:00",
             "latency_s": 1.0, "success": True,
             "machines_total": 5, "machines_predicted": 5, "machines_cold_start": 0,
             "alarm_count": 1, "alarm_rate": 0.2, "cold_start_rate": 0.0},
            {"cycle_index": 1, "timestamp": "2026-01-01T00:05:00+00:00",
             "latency_s": 1.5, "success": True,
             "machines_total": 5, "machines_predicted": 5, "machines_cold_start": 0,
             "alarm_count": 0, "alarm_rate": 0.0, "cold_start_rate": 0.0},
        )
        _write_slo(slo_path, d, model_version=None)
        assert json.loads(slo_path.read_text())["summary"]["alarm_rate_std"] is not None

    def test_skipped_cycle_excluded_from_alarm_rate_mean(self, tmp_path):
        """Skipped cycles (success=False) must not contribute to alarm_rate_mean."""
        slo_path = tmp_path / "slo_metrics.json"
        d = _make_deque(
            {"cycle_index": 0, "timestamp": "2026-01-01T00:00:00+00:00",
             "latency_s": 0.5, "success": False,
             "machines_total": None, "machines_predicted": None, "machines_cold_start": None,
             "alarm_count": None, "alarm_rate": None, "cold_start_rate": None},
            {"cycle_index": 1, "timestamp": "2026-01-01T00:05:00+00:00",
             "latency_s": 1.0, "success": True,
             "machines_total": 5, "machines_predicted": 5, "machines_cold_start": 0,
             "alarm_count": 1, "alarm_rate": 0.2, "cold_start_rate": 0.0},
        )
        _write_slo(slo_path, d, model_version=None)
        summary = json.loads(slo_path.read_text())["summary"]

        # Only 1 successful cycle → success_rate = 0.5
        assert abs(summary["success_rate"] - 0.5) < 1e-6
        # alarm_rate_mean computed from 1 successful cycle only
        assert abs(summary["alarm_rate_mean"] - 0.2) < 1e-6

    def test_model_version_embedded(self, tmp_path):
        slo_path = tmp_path / "slo_metrics.json"
        version  = {"model_version": "20260101T000000Z_abc123", "trained_at": "2026-01-01T00:00:00+00:00"}
        d = _make_deque(
            {"cycle_index": 0, "timestamp": "2026-01-01T00:00:00+00:00",
             "latency_s": 1.0, "success": True,
             "machines_total": 2, "machines_predicted": 2, "machines_cold_start": 0,
             "alarm_count": 0, "alarm_rate": 0.0, "cold_start_rate": 0.0},
        )
        _write_slo(slo_path, d, model_version=version)
        assert json.loads(slo_path.read_text())["model_version"] == version

    def test_parent_dir_created_automatically(self, tmp_path):
        """_write_slo must create the parent directory if it does not exist."""
        slo_path = tmp_path / "subdir" / "slo_metrics.json"
        assert not slo_path.parent.exists()

        d = _make_deque(
            {"cycle_index": 0, "timestamp": "2026-01-01T00:00:00+00:00",
             "latency_s": 1.0, "success": True,
             "machines_total": 1, "machines_predicted": 1, "machines_cold_start": 0,
             "alarm_count": 0, "alarm_rate": 0.0, "cold_start_rate": 0.0},
        )
        _write_slo(slo_path, d, model_version=None)
        assert slo_path.exists()


# ── 3. Integration — run() one-shot writes slo_metrics.json ──────────────────


class TestRunOneShotSlo:
    def test_slo_file_created_alongside_predictions(self, tmp_path):
        """run(interval=0) must write both predictions.json and slo_metrics.json."""
        model_dir, _ = make_fake_artifacts(tmp_path, thresholds={1: 0.30})
        df = make_window_df(machine_ids=[1], n_buckets=24)
        csv_path  = tmp_path / "input.csv"
        pred_path = tmp_path / "predictions.json"
        df.to_csv(csv_path, index=False)

        run(
            model_dir   = model_dir,
            output_path = pred_path,
            interval    = 0,
            input_path  = csv_path,
        )

        assert pred_path.exists(),                  "predictions.json not written"
        assert (tmp_path / "slo_metrics.json").exists(), "slo_metrics.json not written"

    def test_slo_file_schema(self, tmp_path):
        """slo_metrics.json must have model_version, summary, and a one-entry cycles list."""
        model_dir, _ = make_fake_artifacts(tmp_path, thresholds={1: 0.30})
        df = make_window_df(machine_ids=[1], n_buckets=24)
        csv_path = tmp_path / "input.csv"
        df.to_csv(csv_path, index=False)

        run(
            model_dir   = model_dir,
            output_path = tmp_path / "predictions.json",
            interval    = 0,
            input_path  = csv_path,
        )

        data = json.loads((tmp_path / "slo_metrics.json").read_text())
        assert "model_version" in data
        assert "summary"       in data
        assert "cycles"        in data
        assert len(data["cycles"]) == 1

    def test_slo_cycle_success_and_latency(self, tmp_path):
        """The single cycle record must have success=True and a non-negative latency."""
        model_dir, _ = make_fake_artifacts(tmp_path, thresholds={1: 0.30})
        df = make_window_df(machine_ids=[1], n_buckets=24)
        csv_path = tmp_path / "input.csv"
        df.to_csv(csv_path, index=False)

        run(
            model_dir   = model_dir,
            output_path = tmp_path / "predictions.json",
            interval    = 0,
            input_path  = csv_path,
        )

        cycle = json.loads((tmp_path / "slo_metrics.json").read_text())["cycles"][0]
        assert cycle["success"]   is True
        assert cycle["latency_s"] >= 0

    def test_slo_model_version_string_present(self, tmp_path):
        """model_version in slo_metrics.json must contain the version string key."""
        model_dir, _ = make_fake_artifacts(tmp_path, thresholds={1: 0.30})
        df = make_window_df(machine_ids=[1], n_buckets=24)
        csv_path = tmp_path / "input.csv"
        df.to_csv(csv_path, index=False)

        run(
            model_dir   = model_dir,
            output_path = tmp_path / "predictions.json",
            interval    = 0,
            input_path  = csv_path,
        )

        mv = json.loads((tmp_path / "slo_metrics.json").read_text())["model_version"]
        # model_version is the full version_info dict
        assert mv is not None
        assert "model_version" in mv
        assert isinstance(mv["model_version"], str)
        assert len(mv["model_version"]) > 0

    def test_predictions_contain_model_version_string(self, tmp_path):
        """predictions.json top-level must contain model_version as a string."""
        model_dir, _ = make_fake_artifacts(tmp_path, thresholds={1: 0.30})
        df = make_window_df(machine_ids=[1], n_buckets=24)
        csv_path  = tmp_path / "input.csv"
        pred_path = tmp_path / "predictions.json"
        df.to_csv(csv_path, index=False)

        run(
            model_dir   = model_dir,
            output_path = pred_path,
            interval    = 0,
            input_path  = csv_path,
        )

        pdata = json.loads(pred_path.read_text())
        assert "model_version" in pdata
        assert isinstance(pdata["model_version"], str)
        # Format: "<ts>_<6-char-hex>"
        parts = pdata["model_version"].split("_")
        assert len(parts) == 2
