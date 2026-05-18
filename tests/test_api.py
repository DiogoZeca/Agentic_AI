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

from spike.api import _apply_alarm_debounce, _apply_ewma_smoothing, _build_summary_response


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


# ── _build_summary_response tests ─────────────────────────────────────────────


def _make_full_pred(
    machine_id:         int,
    is_spike:           bool | None = False,
    severity_class:     int | None  = 0,
    imminence:          dict | None = None,
    recommended_action: str | None  = "normal",
    status:             str         = "success",
    p_spike_smoothed:   float | None = None,
    consecutive_alarms: int          = 0,
) -> dict:
    """Full prediction dict matching the shape after EWMA + debounce."""
    return {
        "machine_id":          machine_id,
        "is_spike":            is_spike,
        "severity_class":      severity_class,
        "imminence":           imminence or {},
        "recommended_action":  recommended_action,
        "status":              status,
        "p_spike_smoothed":    p_spike_smoothed,
        "consecutive_alarms":  consecutive_alarms,
    }


class TestBuildSummaryResponse:
    """_build_summary_response aggregates predictions into scheduler summary format."""

    def test_no_spikes_all_counts_zero(self):
        preds = [_make_full_pred(i) for i in range(3)]
        out = _build_summary_response(_make_result(preds))
        s = out["summary"]
        assert s["spikes_in_15m"] == 0
        assert s["spikes_in_30m"] == 0
        assert s["spikes_in_45m"] == 0
        assert s["spikes_in_60m"] == 0
        assert s["nodes_requiring_action"] == 0

    def test_60m_spike_only_increments_60m_count(self):
        pred = _make_full_pred(1, is_spike=True, severity_class=1,
                               imminence={}, recommended_action="monitor")
        out = _build_summary_response(_make_result([pred]))
        s = out["summary"]
        assert s["spikes_in_60m"] == 1
        assert s["spikes_in_15m"] == 0
        assert s["spikes_in_30m"] == 0
        assert s["spikes_in_45m"] == 0

    def test_15m_alarm_increments_all_counts(self):
        imm = {
            "15m": {"is_spike": True},
            "30m": {"is_spike": True},
            "45m": {"is_spike": True},
        }
        pred = _make_full_pred(1, is_spike=True, severity_class=1, imminence=imm)
        out = _build_summary_response(_make_result([pred]))
        s = out["summary"]
        assert s["spikes_in_15m"] == 1
        assert s["spikes_in_30m"] == 1
        assert s["spikes_in_45m"] == 1
        assert s["spikes_in_60m"] == 1

    def test_cold_start_excluded_from_counts(self):
        preds = [
            _make_full_pred(1, is_spike=True, severity_class=1),
            _make_full_pred(2, is_spike=None, severity_class=None,
                            status="cold_start"),
        ]
        out = _build_summary_response(_make_result(preds))
        assert out["summary"]["spikes_in_60m"] == 1   # only machine 1

    def test_cold_start_present_in_per_node(self):
        preds = [
            _make_full_pred(1),
            _make_full_pred(2, status="cold_start", is_spike=None, severity_class=None),
        ]
        out = _build_summary_response(_make_result(preds))
        assert len(out["per_node"]) == 2
        cold = next(p for p in out["per_node"] if p["machine_id"] == 2)
        assert cold["status"]           == "cold_start"
        assert cold["horizon"]          is None
        assert cold["alarm_source"]     is None
        assert cold["spike_imminent"]   is None
        assert cold["scheduler_score"]  is None

    def test_horizon_is_earliest_alarming(self):
        imm = {"15m": {"is_spike": False}, "30m": {"is_spike": True}, "45m": {"is_spike": True}}
        pred = _make_full_pred(1, is_spike=True, imminence=imm)
        out = _build_summary_response(_make_result([pred]))
        node = out["per_node"][0]
        assert node["horizon"] == "30m"

    def test_horizon_none_when_no_spike(self):
        pred = _make_full_pred(1, is_spike=False, severity_class=0)
        out = _build_summary_response(_make_result([pred]))
        assert out["per_node"][0]["horizon"] is None

    def test_severity_labels_mapped_correctly(self):
        preds = [
            _make_full_pred(1, severity_class=0),
            _make_full_pred(2, severity_class=1, is_spike=True),
            _make_full_pred(3, severity_class=2, is_spike=True),
            _make_full_pred(4, severity_class=None, status="cold_start", is_spike=None),
        ]
        out = _build_summary_response(_make_result(preds))
        by_id = {p["machine_id"]: p for p in out["per_node"]}
        assert by_id[1]["severity"] == "no_spike"
        assert by_id[2]["severity"] == "moderate"
        assert by_id[3]["severity"] == "severe"
        assert by_id[4]["severity"] is None

    def test_all_machines_present_in_per_node(self):
        preds = [_make_full_pred(i) for i in range(5)]
        out = _build_summary_response(_make_result(preds))
        assert len(out["per_node"]) == 5

    def test_metadata_fields_forwarded(self):
        result = _make_result([_make_full_pred(1)])
        out = _build_summary_response(result)
        assert out["predicted_at"]        == result["predicted_at"]
        assert out["machines_total"]      == result["machines_total"]
        assert out["machines_predicted"]  == result["machines_predicted"]
        assert out["machines_cold_start"] == result["machines_cold_start"]

    # ── spike_60m (renamed from is_spike) ─────────────────────────────────────

    def test_spike_60m_true_when_60m_model_fires(self):
        pred = _make_full_pred(1, is_spike=True, severity_class=1)
        out  = _build_summary_response(_make_result([pred]))
        assert out["per_node"][0]["spike_60m"] is True

    def test_spike_60m_false_when_no_alarm(self):
        pred = _make_full_pred(1, is_spike=False)
        out  = _build_summary_response(_make_result([pred]))
        assert out["per_node"][0]["spike_60m"] is False

    # ── spike_imminent ─────────────────────────────────────────────────────────

    def test_spike_imminent_true_when_binary_15m_alarmed(self):
        imm  = {"15m": {"is_spike": True}, "30m": {"is_spike": False}}
        pred = _make_full_pred(1, imminence=imm)
        out  = _build_summary_response(_make_result([pred]))
        assert out["per_node"][0]["spike_imminent"] is True

    def test_spike_imminent_false_when_only_60m_fires(self):
        imm  = {"15m": {"is_spike": False}, "30m": {"is_spike": False}, "45m": {"is_spike": False}}
        pred = _make_full_pred(1, is_spike=True, imminence=imm)
        out  = _build_summary_response(_make_result([pred]))
        assert out["per_node"][0]["spike_imminent"] is False

    def test_spike_imminent_false_when_no_alarm(self):
        pred = _make_full_pred(1, is_spike=False, imminence={})
        out  = _build_summary_response(_make_result([pred]))
        assert out["per_node"][0]["spike_imminent"] is False

    def test_spike_imminent_none_for_cold_start(self):
        pred = _make_full_pred(1, status="cold_start", is_spike=None, severity_class=None)
        out  = _build_summary_response(_make_result([pred]))
        assert out["per_node"][0]["spike_imminent"] is None

    # ── alarm_source ───────────────────────────────────────────────────────────

    def test_alarm_source_binary_15m_when_15m_fires(self):
        imm  = {"15m": {"is_spike": True}}
        pred = _make_full_pred(1, imminence=imm, recommended_action="preempt_now")
        out  = _build_summary_response(_make_result([pred]))
        assert out["per_node"][0]["alarm_source"] == "binary_15m"

    def test_alarm_source_binary_30m_when_30m_earliest(self):
        imm  = {"15m": {"is_spike": False}, "30m": {"is_spike": True}}
        pred = _make_full_pred(1, imminence=imm, recommended_action="preempt_now")
        out  = _build_summary_response(_make_result([pred]))
        assert out["per_node"][0]["alarm_source"] == "binary_30m"

    def test_alarm_source_60m_severity_when_only_60m_fires(self):
        imm  = {"15m": {"is_spike": False}, "30m": {"is_spike": False}, "45m": {"is_spike": False}}
        pred = _make_full_pred(1, is_spike=True, imminence=imm, recommended_action="migrate_jobs")
        out  = _build_summary_response(_make_result([pred]))
        assert out["per_node"][0]["alarm_source"] == "60m_severity"

    def test_alarm_source_none_when_no_alarm(self):
        pred = _make_full_pred(1, is_spike=False, imminence={})
        out  = _build_summary_response(_make_result([pred]))
        assert out["per_node"][0]["alarm_source"] == "none"

    def test_alarm_source_none_for_cold_start(self):
        pred = _make_full_pred(1, status="cold_start", is_spike=None, severity_class=None)
        out  = _build_summary_response(_make_result([pred]))
        assert out["per_node"][0]["alarm_source"] is None

    # ── scheduler_score ────────────────────────────────────────────────────────

    def test_scheduler_score_100_when_p_spike_zero(self):
        pred = _make_full_pred(1, p_spike_smoothed=0.0)
        out  = _build_summary_response(_make_result([pred]))
        assert out["per_node"][0]["scheduler_score"] == 100

    def test_scheduler_score_0_when_p_spike_one(self):
        pred = _make_full_pred(1, p_spike_smoothed=1.0)
        out  = _build_summary_response(_make_result([pred]))
        assert out["per_node"][0]["scheduler_score"] == 0

    def test_scheduler_score_rounds_correctly(self):
        pred = _make_full_pred(1, p_spike_smoothed=0.519)
        out  = _build_summary_response(_make_result([pred]))
        assert out["per_node"][0]["scheduler_score"] == 48   # round(100*(1-0.519))

    def test_scheduler_score_none_when_no_smoothed_score(self):
        pred = _make_full_pred(1, p_spike_smoothed=None)
        out  = _build_summary_response(_make_result([pred]))
        assert out["per_node"][0]["scheduler_score"] is None

    # ── nodes_requiring_action ─────────────────────────────────────────────────

    def test_nodes_requiring_action_counts_preempt_and_migrate(self):
        preds = [
            _make_full_pred(1, recommended_action="preempt_now"),
            _make_full_pred(2, recommended_action="migrate_jobs"),
            _make_full_pred(3, recommended_action="monitor"),
            _make_full_pred(4, recommended_action="normal"),
        ]
        out = _build_summary_response(_make_result(preds))
        assert out["summary"]["nodes_requiring_action"] == 2

    def test_nodes_requiring_action_zero_when_all_normal(self):
        preds = [_make_full_pred(i, recommended_action="normal") for i in range(4)]
        out   = _build_summary_response(_make_result(preds))
        assert out["summary"]["nodes_requiring_action"] == 0

    def test_multiple_machines_counted_independently(self):
        # Machine 1 alarms at 15m — monotone enforcement means 30m/45m also True in real output.
        # Machine 2 only alarms at 30m+. Machine 3 has no spike.
        preds = [
            _make_full_pred(1, is_spike=True, severity_class=1,
                            imminence={"15m": {"is_spike": True},
                                       "30m": {"is_spike": True},
                                       "45m": {"is_spike": True}}),
            _make_full_pred(2, is_spike=True, severity_class=1,
                            imminence={"15m": {"is_spike": False},
                                       "30m": {"is_spike": True}}),
            _make_full_pred(3, is_spike=False, severity_class=0),
        ]
        out = _build_summary_response(_make_result(preds))
        s = out["summary"]
        assert s["spikes_in_15m"] == 1   # only machine 1
        assert s["spikes_in_30m"] == 2   # machines 1 and 2
        assert s["spikes_in_45m"] == 1   # only machine 1
        assert s["spikes_in_60m"] == 2   # machines 1 and 2
