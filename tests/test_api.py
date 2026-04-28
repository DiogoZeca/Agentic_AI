"""Tests for spike_api.py — EWMA smoothing and alarm debounce logic.

Tests target pure functions (_apply_ewma_smoothing, _apply_alarm_debounce)
directly given a state dict, avoiding the need for a live server or loaded
model artefacts.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike.api import _apply_alarm_debounce, _apply_ewma_smoothing


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_result(predictions: list[dict]) -> dict:
    """Minimal envelope dict matching predict_with_artifacts() output shape."""
    return {
        "batch_id":            "test-batch",
        "predicted_at":        "2026-01-01T00:00:00+00:00",
        "machines_total":      len(predictions),
        "machines_predicted":  sum(1 for p in predictions if p["status"] != "cold_start"),
        "machines_cold_start": sum(1 for p in predictions if p["status"] == "cold_start"),
        "predictions":         predictions,
    }


def _make_pred(
    machine_id:         int,
    is_spike:           bool | None = False,
    is_severe:          bool | None = False,
    recommended_action: str | None = "normal",
    status:             str = "success",
) -> dict:
    return {
        "machine_id":          machine_id,
        "is_spike":            is_spike,
        "is_severe":           is_severe,
        "recommended_action":  recommended_action,
        "status":              status,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestDebounceBasicCounting:
    """Counter increments on spikes and resets on non-spikes."""

    def test_single_spike_below_threshold_is_suppressed(self):
        state = {}
        result = _apply_alarm_debounce(
            _make_result([_make_pred(1, is_spike=True, is_severe=True,
                                     recommended_action="defer_batch")]),
            state, min_consecutive=2,
        )
        pred = result["predictions"][0]
        assert pred["is_spike"]           is False
        assert pred["is_severe"]          is False
        assert pred["recommended_action"] == "normal"
        assert pred["consecutive_alarms"] == 1

    def test_second_consecutive_spike_fires(self):
        state = {1: 1}   # already one spike from previous cycle
        result = _apply_alarm_debounce(
            _make_result([_make_pred(1, is_spike=True, is_severe=False,
                                     recommended_action="monitor")]),
            state, min_consecutive=2,
        )
        pred = result["predictions"][0]
        assert pred["is_spike"]           is True
        assert pred["recommended_action"] == "monitor"
        assert pred["consecutive_alarms"] == 2

    def test_no_spike_resets_counter(self):
        state = {1: 3}   # machine had a streak
        _apply_alarm_debounce(
            _make_result([_make_pred(1, is_spike=False)]),
            state, min_consecutive=2,
        )
        assert state[1] == 0

    def test_counter_persists_across_calls(self):
        state = {}
        # Call 1: spike fires (streak = 1, suppressed)
        _apply_alarm_debounce(
            _make_result([_make_pred(1, is_spike=True)]),
            state, min_consecutive=2,
        )
        assert state[1] == 1
        # Call 2: spike fires again (streak = 2, no longer suppressed)
        result = _apply_alarm_debounce(
            _make_result([_make_pred(1, is_spike=True)]),
            state, min_consecutive=2,
        )
        assert state[1] == 2
        assert result["predictions"][0]["is_spike"] is True


class TestDebounceEdgeCases:
    """Boundary conditions and special machine states."""

    def test_min_consecutive_1_never_suppresses(self):
        """min_consecutive=1 means every spike fires immediately."""
        state = {}
        result = _apply_alarm_debounce(
            _make_result([_make_pred(1, is_spike=True, recommended_action="monitor")]),
            state, min_consecutive=1,
        )
        pred = result["predictions"][0]
        assert pred["is_spike"]           is True
        assert pred["recommended_action"] == "monitor"
        assert pred["consecutive_alarms"] == 1

    def test_cold_start_machine_gets_zero_count_no_override(self):
        """cold_start predictions are untouched by debounce."""
        state = {}
        pred_in = _make_pred(1, is_spike=None, is_severe=None,
                              recommended_action=None, status="cold_start")
        result = _apply_alarm_debounce(
            _make_result([pred_in]), state, min_consecutive=2,
        )
        pred = result["predictions"][0]
        assert pred["consecutive_alarms"] == 0
        assert pred["is_spike"]           is None   # unchanged
        assert pred["recommended_action"] is None   # unchanged

    def test_new_machine_starts_at_zero_when_no_spike(self):
        state = {}
        result = _apply_alarm_debounce(
            _make_result([_make_pred(99, is_spike=False)]),
            state, min_consecutive=2,
        )
        assert result["predictions"][0]["consecutive_alarms"] == 0
        assert state.get(99, 0) == 0

    def test_gap_then_spike_requires_full_streak_again(self):
        """A non-spike resets; next spike must build a fresh streak."""
        state = {1: 1}   # was at 1
        # Gap cycle
        _apply_alarm_debounce(
            _make_result([_make_pred(1, is_spike=False)]),
            state, min_consecutive=2,
        )
        assert state[1] == 0
        # Single spike after gap — should be suppressed again
        result = _apply_alarm_debounce(
            _make_result([_make_pred(1, is_spike=True)]),
            state, min_consecutive=2,
        )
        assert result["predictions"][0]["is_spike"] is False
        assert result["predictions"][0]["consecutive_alarms"] == 1

    def test_multiple_machines_independent_counters(self):
        """Each machine has its own independent streak."""
        state = {}
        preds = [
            _make_pred(1, is_spike=True),   # first spike — suppressed
            _make_pred(2, is_spike=False),  # no spike
            _make_pred(3, is_spike=True),   # first spike — suppressed
        ]
        result = _apply_alarm_debounce(
            _make_result(preds), state, min_consecutive=2,
        )
        assert state[1] == 1
        assert state[2] == 0
        assert state[3] == 1
        assert result["predictions"][0]["is_spike"] is False
        assert result["predictions"][1]["is_spike"] is False   # was already False
        assert result["predictions"][2]["is_spike"] is False


class TestDebounceRawSignalPreserved:
    """Probabilities and imminence are never modified by debounce."""

    def test_probabilities_unchanged_when_suppressed(self):
        pred_in = _make_pred(1, is_spike=True)
        pred_in["p_severe"]   = 0.72
        pred_in["p_moderate"] = 0.15
        pred_in["imminence"]  = {"15m": {"p_spike": 0.68, "is_spike": True}}

        result = _apply_alarm_debounce(
            _make_result([pred_in]), {}, min_consecutive=2,
        )
        pred = result["predictions"][0]
        assert pred["p_severe"]                        == 0.72
        assert pred["p_moderate"]                      == 0.15
        assert pred["imminence"]["15m"]["p_spike"]     == 0.68
        # imminence.is_spike is NOT touched — raw model signal
        assert pred["imminence"]["15m"]["is_spike"]    is True


# ── EWMA smoothing tests ──────────────────────────────────────────────────────


def _make_spike_pred(
    machine_id:         int,
    p_moderate:         float,
    p_severe:           float,
    alarm_threshold:    float = 0.55,
    is_spike:           bool  = True,
    is_severe:          bool  = False,
    recommended_action: str   = "monitor",
    status:             str   = "success",
) -> dict:
    """Prediction dict with probability fields needed by _apply_ewma_smoothing."""
    return {
        "machine_id":          machine_id,
        "p_moderate":          p_moderate,
        "p_severe":            p_severe,
        "alarm_threshold":     alarm_threshold,
        "is_spike":            is_spike,
        "is_severe":           is_severe,
        "recommended_action":  recommended_action,
        "status":              status,
    }


class TestEWMASmoothing:
    """EWMA suppression: reduces single-cycle noise without hiding real trends."""

    def test_single_high_spike_is_suppressed_when_history_is_low(self):
        """First-encounter machine: no history → EWMA = raw → no suppression."""
        # alpha=1.0: EWMA = raw score; raw p_spike=0.70 > threshold=0.55 → not suppressed
        pred = _make_spike_pred(1, p_moderate=0.40, p_severe=0.30, alarm_threshold=0.55,
                                is_spike=True)
        result = _apply_ewma_smoothing(_make_result([pred]), {}, alpha=1.0)
        assert result["predictions"][0]["is_spike"] is True

    def test_ewma_suppresses_single_cycle_noise(self):
        """Machine with low history (smoothed=0.20) gets a one-cycle high p_spike (0.70)
        → smoothed remains below threshold → alarm suppressed."""
        state = {1: 0.20}   # previous smoothed score
        pred  = _make_spike_pred(1, p_moderate=0.40, p_severe=0.30,  # p_spike_raw=0.70
                                 alarm_threshold=0.55, is_spike=True,
                                 recommended_action="defer_batch")
        result = _apply_ewma_smoothing(_make_result([pred]), state, alpha=0.5)
        p      = result["predictions"][0]
        # smoothed = 0.5*0.70 + 0.5*0.20 = 0.45 < 0.55 → suppressed
        assert p["is_spike"]           is False
        assert p["is_severe"]          is False
        assert p["recommended_action"] == "normal"

    def test_sustained_spike_passes_through(self):
        """Machine with already-elevated history + continued high p_spike → not suppressed."""
        state = {1: 0.65}   # smoothed already above threshold
        pred  = _make_spike_pred(1, p_moderate=0.40, p_severe=0.30,  # p_spike_raw=0.70
                                 alarm_threshold=0.55, is_spike=True)
        result = _apply_ewma_smoothing(_make_result([pred]), state, alpha=0.5)
        p      = result["predictions"][0]
        # smoothed = 0.5*0.70 + 0.5*0.65 = 0.675 > 0.55 → not suppressed
        assert p["is_spike"] is True

    def test_state_updated_correctly(self):
        """EWMA state is updated after each call."""
        state = {1: 0.40}
        pred  = _make_spike_pred(1, p_moderate=0.35, p_severe=0.25,  # p_spike_raw=0.60
                                 alarm_threshold=0.55)
        _apply_ewma_smoothing(_make_result([pred]), state, alpha=0.5)
        # 0.5 * 0.60 + 0.5 * 0.40 = 0.50
        assert abs(state[1] - 0.50) < 1e-6

    def test_first_encounter_seeds_state_with_raw_score(self):
        """New machine initialises EWMA with raw p_spike — no warm-up suppression."""
        state = {}
        pred  = _make_spike_pred(99, p_moderate=0.30, p_severe=0.35,  # p_spike_raw=0.65
                                 alarm_threshold=0.55, is_spike=True)
        result = _apply_ewma_smoothing(_make_result([pred]), state, alpha=0.5)
        # smoothed = 0.5 * 0.65 + 0.5 * 0.65 = 0.65 (seeded with raw)
        assert abs(state[99] - 0.65) < 1e-6
        assert result["predictions"][0]["is_spike"] is True  # 0.65 > 0.55

    def test_p_spike_smoothed_field_added(self):
        """p_spike_smoothed is always written for non-cold-start predictions."""
        state = {}
        pred  = _make_spike_pred(1, p_moderate=0.20, p_severe=0.20, alarm_threshold=0.55)
        result = _apply_ewma_smoothing(_make_result([pred]), state, alpha=1.0)
        assert "p_spike_smoothed" in result["predictions"][0]
        assert isinstance(result["predictions"][0]["p_spike_smoothed"], float)

    def test_cold_start_untouched(self):
        """cold_start predictions: p_spike_smoothed=None, alarm fields unchanged."""
        pred = {
            "machine_id":          5,
            "status":              "cold_start",
            "is_spike":            None,
            "is_severe":           None,
            "recommended_action":  None,
        }
        result = _apply_ewma_smoothing(_make_result([pred]), {}, alpha=0.5)
        p = result["predictions"][0]
        assert p["p_spike_smoothed"] is None
        assert p["is_spike"]          is None   # unchanged

    def test_alpha_one_is_passthrough(self):
        """alpha=1.0 means EWMA = raw score; any alarm that passes raw threshold is preserved."""
        state = {1: 0.10}   # low history — would suppress at alpha=0.5
        pred  = _make_spike_pred(1, p_moderate=0.40, p_severe=0.30,  # p_spike_raw=0.70
                                 alarm_threshold=0.55, is_spike=True)
        result = _apply_ewma_smoothing(_make_result([pred]), state, alpha=1.0)
        # smoothed = 1.0 * 0.70 + 0.0 * 0.10 = 0.70 > threshold → not suppressed
        assert result["predictions"][0]["is_spike"] is True

    def test_raw_probabilities_never_modified(self):
        """p_moderate and p_severe are not touched, even when alarm is suppressed."""
        state = {1: 0.10}
        pred  = _make_spike_pred(1, p_moderate=0.40, p_severe=0.30, alarm_threshold=0.55)
        result = _apply_ewma_smoothing(_make_result([pred]), state, alpha=0.5)
        p = result["predictions"][0]
        assert p["p_moderate"] == 0.40
        assert p["p_severe"]   == 0.30

    def test_multiple_machines_independent_state(self):
        """Each machine maintains its own EWMA; one machine's suppression doesn't affect another."""
        state = {1: 0.20, 2: 0.70}
        preds = [
            _make_spike_pred(1, p_moderate=0.40, p_severe=0.30, alarm_threshold=0.55,
                             is_spike=True),  # smoothed → 0.45 → suppressed
            _make_spike_pred(2, p_moderate=0.40, p_severe=0.30, alarm_threshold=0.55,
                             is_spike=True),  # smoothed → 0.675 → not suppressed
        ]
        result = _apply_ewma_smoothing(_make_result(preds), state, alpha=0.5)
        assert result["predictions"][0]["is_spike"] is False   # machine 1 suppressed
        assert result["predictions"][1]["is_spike"] is True    # machine 2 passes through
