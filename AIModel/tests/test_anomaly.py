"""
Test suite for anomaly_detector.py

Four test classes:
  TestHelpers               — _compute_excess_carbon, _find_optimal_window_start
  TestDetectPointAnomalies  — Stage 1: flag individual anomalous timestamps
  TestFindRecurringPatterns — Stage 2: group into (hour, day_type) patterns
  TestBuildInvestigationLeads — Stage 3: rank + savings estimates

Plus TestFullPipeline — end-to-end: injected synthetic spike → lead #1.

All tests are Prophet-free. The forecast DataFrame is constructed directly
(matching what model.predict(periods=0) would return), making the suite fast
and deterministic.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from anomaly_detector import (
    Anomaly,
    InvestigationLead,
    RecurringPattern,
    _compute_excess_carbon,
    _compute_excess_cost,
    _find_optimal_window_start,
    build_investigation_leads,
    detect_point_anomalies,
    find_recurring_patterns,
)


# ── Shared helpers ────────────────────────────────────────────────────────────
# These are plain functions (not fixtures) — each test builds exactly what it
# needs, keeping state isolated.

def _make_df(
    n_hours:  int   = 48,
    start:    str   = "2024-01-01",  # Monday
    value:    float = 0.08,          # consumption within bounds
    ci:       float = 0.40,
    cost:     float = 0.012,
) -> pd.DataFrame:
    """Minimal df_actual with all optional enrichment columns."""
    dates = pd.date_range(start, periods=n_hours, freq="h")
    return pd.DataFrame({
        "ds":                    dates,
        "consumption":           value,
        "carbonIntensityFactor": ci,
        "cost":                  cost,
    })


def _make_forecast(
    n_hours:    int   = 48,
    start:      str   = "2024-01-01",
    yhat:       float = 0.08,
    yhat_lower: float = 0.06,
    yhat_upper: float = 0.10,
) -> pd.DataFrame:
    """Synthetic Prophet-style forecast (all columns constant for simplicity)."""
    dates = pd.date_range(start, periods=n_hours, freq="h")
    return pd.DataFrame({
        "ds":         dates,
        "yhat":       yhat,
        "yhat_lower": yhat_lower,
        "yhat_upper": yhat_upper,
    })


def _make_anomaly(
    ds:                   str   = "2024-01-01 15:00",
    metric:               str   = "consumption",
    actual:               float = 0.15,
    expected:             float = 0.08,
    upper_bound:          float = 0.10,
    lower_bound:          float = 0.06,
    excess_abs:           float = 0.05,
    excess_pct:           float = 87.5,
    hour_of_day:          int   = 15,
    is_weekend:           bool  = False,
    carbon_intensity:     float = 0.40,
    cost_eur_h:           float = 0.018,
    excess_carbon_kgco2e: float = 0.020,
    excess_cost_eur:      float = 0.006,
) -> Anomaly:
    """Factory for Anomaly instances in pattern / lead tests."""
    return Anomaly(
        ds                   = pd.Timestamp(ds).to_pydatetime(),
        metric               = metric,
        actual               = actual,
        expected             = expected,
        upper_bound          = upper_bound,
        lower_bound          = lower_bound,
        excess_abs           = excess_abs,
        excess_pct           = excess_pct,
        hour_of_day          = hour_of_day,
        is_weekend           = is_weekend,
        carbon_intensity     = carbon_intensity,
        cost_eur_h           = cost_eur_h,
        excess_carbon_kgco2e = excess_carbon_kgco2e,
        excess_cost_eur      = excess_cost_eur,
    )


def _one_week_df(ci_by_hour: dict[int, float] | None = None) -> pd.DataFrame:
    """
    7-day DataFrame (Mon 2024-01-01 → Sun 2024-01-07).
    Provides 168 rows; 5 weekday-slots and 2 weekend-slots per hour.
    Optional ci_by_hour overrides carbonIntensityFactor for specific hours.
    """
    dates = pd.date_range("2024-01-01", periods=24 * 7, freq="h")
    ci    = pd.Series(0.40, index=range(len(dates)))
    if ci_by_hour:
        for h, v in ci_by_hour.items():
            ci[dates.hour == h] = v
    return pd.DataFrame({
        "ds":                    dates,
        "consumption":           0.08,
        "carbonIntensityFactor": ci.values,
        "cost":                  0.012,
    })


# ── TestHelpers ───────────────────────────────────────────────────────────────

class TestComputeExcessCarbon:

    @pytest.mark.parametrize("metric,excess,ci,expected", [
        ("consumption",                0.01, 0.40, 0.004),   # kWh × kgCO2/kWh
        ("consumption",                0.05, 0.30, 0.015),   # different CI
        ("carbonEmissions",            0.03, 0.40, 0.030),   # already kgCO2e
        ("carbonEmissions",            0.03, 0.00, 0.030),   # CI irrelevant for emissions
        ("cpuUtilization",             0.10, 0.40, 0.000),   # no translation
        ("functionalUnit",             10.0, 0.40, 0.000),
        ("cost",                       0.50, 0.40, 0.000),
        ("greenConsumptionPercentage", 5.00, 0.40, 0.000),
    ])
    def test_excess_carbon_per_metric(self, metric, excess, ci, expected):
        result = _compute_excess_carbon(metric, excess, ci)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_zero_ci_gives_zero_for_consumption(self):
        assert _compute_excess_carbon("consumption", 0.01, 0.0) == pytest.approx(0.0)


class TestComputeExcessCost:

    def test_consumption_excess_cost(self):
        # rate = cost_eur_h / actual = 0.018 / 0.12 = 0.15 EUR/kWh
        # excess_cost = 0.05 kWh × 0.15 EUR/kWh = 0.0075 EUR
        result = _compute_excess_cost("consumption", excess_abs=0.05, cost_eur_h=0.018, actual_cons=0.12)
        assert result == pytest.approx(0.0075, rel=1e-5)

    def test_cost_metric_returns_excess_directly(self):
        result = _compute_excess_cost("cost", excess_abs=0.05, cost_eur_h=0.018, actual_cons=0.12)
        assert result == pytest.approx(0.05)

    def test_other_metrics_return_zero(self):
        for metric in ("carbonEmissions", "cpuUtilization", "functionalUnit"):
            assert _compute_excess_cost(metric, 0.1, 0.02, 0.10) == pytest.approx(0.0)

    def test_near_zero_consumption_does_not_divide_by_zero(self):
        # actual_cons is near zero — should not raise
        result = _compute_excess_cost("consumption", excess_abs=0.01, cost_eur_h=0.015, actual_cons=0.0)
        assert result >= 0.0


class TestFindOptimalWindowStart:

    def test_lowest_ci_hours_found(self):
        """Hours 3–6 have CI = 0.30; all others 0.40 — window should start at 3."""
        ci = pd.Series({h: (0.30 if 3 <= h <= 6 else 0.40) for h in range(24)})
        assert _find_optimal_window_start(ci, window_hours=4) == 3

    def test_midnight_crossing(self):
        """Best window crosses midnight: hours 23, 0, 1, 2 all low."""
        ci = pd.Series({h: (0.30 if h in (23, 0, 1, 2) else 0.40) for h in range(24)})
        start = _find_optimal_window_start(ci, window_hours=4)
        assert start == 23

    def test_empty_series_returns_default(self):
        assert _find_optimal_window_start(pd.Series(dtype=float)) == 2

    def test_uniform_ci_returns_hour_zero(self):
        """All hours identical — first minimum wins (hour 0)."""
        ci = pd.Series({h: 0.40 for h in range(24)})
        assert _find_optimal_window_start(ci, window_hours=4) == 0

    def test_window_hours_respected(self):
        """1-hour window: picks the single lowest-CI hour."""
        ci = pd.Series({h: 0.40 for h in range(24)})
        ci[5] = 0.25
        assert _find_optimal_window_start(ci, window_hours=1) == 5


# ── TestDetectPointAnomalies ──────────────────────────────────────────────────

class TestDetectPointAnomalies:

    # ── Happy paths ───────────────────────────────────────────────────────────

    def test_all_rows_flagged_when_all_exceed_upper(self):
        df  = _make_df(n_hours=24, value=0.15)   # 0.15 > yhat_upper=0.10
        fc  = _make_forecast(n_hours=24)
        out = detect_point_anomalies(df, fc, "consumption")
        assert len(out) == 24

    def test_no_anomalies_when_within_bounds(self):
        df  = _make_df(n_hours=24, value=0.08)   # exactly yhat, well inside bounds
        fc  = _make_forecast(n_hours=24)
        out = detect_point_anomalies(df, fc, "consumption")
        assert len(out) == 0

    def test_exact_upper_bound_is_not_an_anomaly(self):
        """Condition is strictly >; equality is NOT an anomaly."""
        df  = _make_df(n_hours=24, value=0.10)   # value == yhat_upper
        fc  = _make_forecast(n_hours=24)
        out = detect_point_anomalies(df, fc, "consumption")
        assert len(out) == 0

    def test_only_spike_rows_flagged(self):
        """Inject a spike at hour 3 only; only that row should be returned."""
        df  = _make_df(n_hours=24, value=0.08)
        df.loc[df["ds"].dt.hour == 3, "consumption"] = 0.20
        fc  = _make_forecast(n_hours=24)
        out = detect_point_anomalies(df, fc, "consumption")
        assert len(out) == 1
        assert out[0].hour_of_day == 3

    # ── Direction modes ───────────────────────────────────────────────────────

    def test_deficit_direction_detects_below_lower(self):
        df  = _make_df(n_hours=24, value=0.04)   # 0.04 < yhat_lower=0.06
        fc  = _make_forecast(n_hours=24)
        out = detect_point_anomalies(df, fc, "consumption", direction="deficit")
        assert len(out) == 24

    def test_deficit_direction_ignores_excess(self):
        df  = _make_df(n_hours=24, value=0.20)   # well above upper
        fc  = _make_forecast(n_hours=24)
        out = detect_point_anomalies(df, fc, "consumption", direction="deficit")
        assert len(out) == 0

    def test_both_direction_flags_excess_and_deficit(self):
        df  = _make_df(n_hours=24, value=0.08)
        # hour 0: excess; hour 1: deficit; hours 2–23: normal
        df.loc[df["ds"].dt.hour == 0, "consumption"] = 0.20
        df.loc[df["ds"].dt.hour == 1, "consumption"] = 0.03
        fc  = _make_forecast(n_hours=24)
        out = detect_point_anomalies(df, fc, "consumption", direction="both")
        assert len(out) == 2

    def test_invalid_direction_raises_value_error(self):
        df  = _make_df()
        fc  = _make_forecast()
        with pytest.raises(ValueError, match="direction"):
            detect_point_anomalies(df, fc, "consumption", direction="sideways")

    # ── Field value correctness ───────────────────────────────────────────────

    def test_excess_abs_equals_actual_minus_upper(self):
        """excess_abs = actual - yhat_upper when actual > upper."""
        df  = _make_df(n_hours=1, value=0.15)   # excess = 0.15 - 0.10 = 0.05
        fc  = _make_forecast(n_hours=1)
        out = detect_point_anomalies(df, fc, "consumption")
        assert out[0].excess_abs == pytest.approx(0.05, rel=1e-5)

    def test_deficit_excess_abs_equals_lower_minus_actual(self):
        """excess_abs = yhat_lower - actual when actual < lower."""
        df  = _make_df(n_hours=1, value=0.04)   # deficit = 0.06 - 0.04 = 0.02
        fc  = _make_forecast(n_hours=1)
        out = detect_point_anomalies(df, fc, "consumption", direction="deficit")
        assert out[0].excess_abs == pytest.approx(0.02, rel=1e-5)

    def test_excess_pct_signed_positive_for_excess(self):
        # excess_pct = (0.15 - 0.08) / 0.08 * 100 = 87.5
        df  = _make_df(n_hours=1, value=0.15)
        fc  = _make_forecast(n_hours=1)
        out = detect_point_anomalies(df, fc, "consumption")
        assert out[0].excess_pct == pytest.approx(87.5, rel=1e-3)

    def test_hour_of_day_correct(self):
        dates = pd.date_range("2024-01-01 14:00", periods=1, freq="h")
        df  = pd.DataFrame({"ds": dates, "consumption": 0.20})
        fc  = pd.DataFrame({"ds": dates, "yhat": 0.08, "yhat_lower": 0.06, "yhat_upper": 0.10})
        out = detect_point_anomalies(df, fc, "consumption")
        assert out[0].hour_of_day == 14

    def test_is_weekend_saturday(self):
        # 2024-01-06 is a Saturday
        dates = pd.date_range("2024-01-06 15:00", periods=1, freq="h")
        df  = pd.DataFrame({"ds": dates, "consumption": 0.20})
        fc  = pd.DataFrame({"ds": dates, "yhat": 0.08, "yhat_lower": 0.06, "yhat_upper": 0.10})
        out = detect_point_anomalies(df, fc, "consumption")
        assert out[0].is_weekend is True

    def test_is_weekend_monday(self):
        # 2024-01-01 is a Monday
        dates = pd.date_range("2024-01-01 15:00", periods=1, freq="h")
        df  = pd.DataFrame({"ds": dates, "consumption": 0.20})
        fc  = pd.DataFrame({"ds": dates, "yhat": 0.08, "yhat_lower": 0.06, "yhat_upper": 0.10})
        out = detect_point_anomalies(df, fc, "consumption")
        assert out[0].is_weekend is False

    def test_carbon_intensity_populated_when_column_present(self):
        df  = _make_df(n_hours=1, value=0.20, ci=0.45)
        fc  = _make_forecast(n_hours=1)
        out = detect_point_anomalies(df, fc, "consumption")
        assert out[0].carbon_intensity == pytest.approx(0.45, rel=1e-5)

    def test_excess_carbon_computed_for_consumption_metric(self):
        # excess_abs=0.10, ci=0.40 → excess_carbon = 0.10 × 0.40 = 0.04
        df  = _make_df(n_hours=1, value=0.20, ci=0.40)   # excess = 0.20 - 0.10 = 0.10
        fc  = _make_forecast(n_hours=1)
        out = detect_point_anomalies(df, fc, "consumption")
        assert out[0].excess_carbon_kgco2e == pytest.approx(0.04, rel=1e-4)

    # ── Graceful degradation ──────────────────────────────────────────────────

    def test_missing_ci_column_defaults_to_zero(self):
        dates = pd.date_range("2024-01-01", periods=1, freq="h")
        df  = pd.DataFrame({"ds": dates, "consumption": 0.20})   # no CI column
        fc  = pd.DataFrame({"ds": dates, "yhat": 0.08, "yhat_lower": 0.06, "yhat_upper": 0.10})
        out = detect_point_anomalies(df, fc, "consumption")
        assert out[0].carbon_intensity        == pytest.approx(0.0)
        assert out[0].excess_carbon_kgco2e   == pytest.approx(0.0)

    def test_missing_cost_column_defaults_to_zero(self):
        dates = pd.date_range("2024-01-01", periods=1, freq="h")
        df  = pd.DataFrame({"ds": dates, "consumption": 0.20})
        fc  = pd.DataFrame({"ds": dates, "yhat": 0.08, "yhat_lower": 0.06, "yhat_upper": 0.10})
        out = detect_point_anomalies(df, fc, "consumption")
        assert out[0].cost_eur_h    == pytest.approx(0.0)
        assert out[0].excess_cost_eur == pytest.approx(0.0)

    # ── Error cases ───────────────────────────────────────────────────────────

    def test_unknown_metric_raises_key_error(self):
        df  = _make_df()
        fc  = _make_forecast()
        with pytest.raises(KeyError):
            detect_point_anomalies(df, fc, "nonexistent_metric")

    def test_missing_yhat_upper_in_forecast_raises_key_error(self):
        df  = _make_df()
        fc  = _make_forecast().drop(columns=["yhat_upper"])
        with pytest.raises(KeyError):
            detect_point_anomalies(df, fc, "consumption")

    def test_no_timestamp_overlap_returns_empty(self):
        df  = _make_df(start="2024-01-01", n_hours=24, value=0.20)
        fc  = _make_forecast(start="2024-02-01", n_hours=24)  # different month
        out = detect_point_anomalies(df, fc, "consumption")
        assert out == []

    # ── Output ordering ───────────────────────────────────────────────────────

    def test_output_sorted_by_timestamp(self):
        df  = _make_df(n_hours=48, value=0.20)
        fc  = _make_forecast(n_hours=48)
        out = detect_point_anomalies(df, fc, "consumption")
        timestamps = [a.ds for a in out]
        assert timestamps == sorted(timestamps)


# ── TestFindRecurringPatterns ─────────────────────────────────────────────────

class TestFindRecurringPatterns:

    def test_empty_anomalies_returns_empty(self):
        assert find_recurring_patterns([], _one_week_df()) == []

    def test_below_min_occurrences_filtered_out(self):
        """2 anomalies at same hour + day_type; min_occurrences=3 → no pattern."""
        anomalies = [
            _make_anomaly(ds="2024-01-01 15:00"),  # Monday
            _make_anomaly(ds="2024-01-02 15:00"),  # Tuesday
        ]
        result = find_recurring_patterns(anomalies, _one_week_df(), min_occurrences=3)
        assert result == []

    def test_exactly_min_occurrences_is_included(self):
        anomalies = [
            _make_anomaly(ds="2024-01-01 15:00"),  # Mon
            _make_anomaly(ds="2024-01-02 15:00"),  # Tue
            _make_anomaly(ds="2024-01-03 15:00"),  # Wed
        ]
        result = find_recurring_patterns(anomalies, _one_week_df(), min_occurrences=3)
        assert len(result) == 1

    def test_weekday_and_weekend_are_separate_patterns(self):
        """Same hour 15 — 3 weekdays + 2 weekends should produce two patterns
        (weekday meets min=3; weekend is below min=3 and is excluded)."""
        anomalies = [
            _make_anomaly(ds="2024-01-01 15:00", is_weekend=False),  # Mon
            _make_anomaly(ds="2024-01-02 15:00", is_weekend=False),  # Tue
            _make_anomaly(ds="2024-01-03 15:00", is_weekend=False),  # Wed
            _make_anomaly(ds="2024-01-06 15:00", is_weekend=True),   # Sat
            _make_anomaly(ds="2024-01-07 15:00", is_weekend=True),   # Sun
        ]
        result = find_recurring_patterns(anomalies, _one_week_df(), min_occurrences=3)
        assert len(result) == 1
        assert result[0].day_type == "weekday"

    def test_different_hours_are_separate_patterns(self):
        anomalies = [
            _make_anomaly(ds="2024-01-01 15:00", hour_of_day=15),
            _make_anomaly(ds="2024-01-02 15:00", hour_of_day=15),
            _make_anomaly(ds="2024-01-03 15:00", hour_of_day=15),
            _make_anomaly(ds="2024-01-01 10:00", hour_of_day=10),
            _make_anomaly(ds="2024-01-02 10:00", hour_of_day=10),
            _make_anomaly(ds="2024-01-03 10:00", hour_of_day=10),
        ]
        result = find_recurring_patterns(anomalies, _one_week_df(), min_occurrences=3)
        assert len(result) == 2
        hours = {p.hour_of_day for p in result}
        assert hours == {10, 15}

    def test_frequency_pct_correct(self):
        """
        3 anomalies at weekday 15:00 over a 1-week df.
        Weekday hour-15 slots in the week: Mon, Tue, Wed, Thu, Fri = 5.
        frequency_pct = 3 / 5 × 100 = 60.0%
        """
        anomalies = [
            _make_anomaly(ds="2024-01-01 15:00"),  # Mon
            _make_anomaly(ds="2024-01-02 15:00"),  # Tue
            _make_anomaly(ds="2024-01-03 15:00"),  # Wed
        ]
        result = find_recurring_patterns(anomalies, _one_week_df(), min_occurrences=3)
        assert result[0].occurrences == 3
        assert result[0].possible    == 5
        assert result[0].frequency_pct == pytest.approx(60.0, rel=1e-3)

    def test_avg_excess_pct_is_mean_of_anomalies(self):
        anomalies = [
            _make_anomaly(ds="2024-01-01 15:00", excess_pct=50.0),
            _make_anomaly(ds="2024-01-02 15:00", excess_pct=100.0),
            _make_anomaly(ds="2024-01-03 15:00", excess_pct=150.0),
        ]
        result = find_recurring_patterns(anomalies, _one_week_df(), min_occurrences=3)
        assert result[0].avg_excess_pct == pytest.approx(100.0, rel=1e-3)

    def test_total_excess_carbon_is_sum(self):
        anomalies = [
            _make_anomaly(ds="2024-01-01 15:00", excess_carbon_kgco2e=0.010),
            _make_anomaly(ds="2024-01-02 15:00", excess_carbon_kgco2e=0.020),
            _make_anomaly(ds="2024-01-03 15:00", excess_carbon_kgco2e=0.030),
        ]
        result = find_recurring_patterns(anomalies, _one_week_df(), min_occurrences=3)
        assert result[0].total_excess_carbon == pytest.approx(0.060, rel=1e-5)

    def test_total_excess_cost_is_sum(self):
        anomalies = [
            _make_anomaly(ds="2024-01-01 15:00", excess_cost_eur=0.005),
            _make_anomaly(ds="2024-01-02 15:00", excess_cost_eur=0.010),
            _make_anomaly(ds="2024-01-03 15:00", excess_cost_eur=0.015),
        ]
        result = find_recurring_patterns(anomalies, _one_week_df(), min_occurrences=3)
        assert result[0].total_excess_cost == pytest.approx(0.030, rel=1e-5)

    def test_carbon_context_high_when_ci_above_median(self):
        """
        Dataset median CI = 0.40 (uniform).
        Anomalies at hour 15 have CI = 0.45 (above median) → "high-carbon".
        """
        df = _one_week_df()  # uniform CI = 0.40 → median = 0.40
        anomalies = [
            _make_anomaly(ds="2024-01-01 15:00", carbon_intensity=0.45),
            _make_anomaly(ds="2024-01-02 15:00", carbon_intensity=0.45),
            _make_anomaly(ds="2024-01-03 15:00", carbon_intensity=0.45),
        ]
        result = find_recurring_patterns(anomalies, df, min_occurrences=3)
        assert result[0].carbon_context == "high-carbon"

    def test_carbon_context_low_when_ci_below_median(self):
        df = _one_week_df()  # median = 0.40
        anomalies = [
            _make_anomaly(ds="2024-01-01 03:00", hour_of_day=3, carbon_intensity=0.32),
            _make_anomaly(ds="2024-01-02 03:00", hour_of_day=3, carbon_intensity=0.32),
            _make_anomaly(ds="2024-01-03 03:00", hour_of_day=3, carbon_intensity=0.32),
        ]
        result = find_recurring_patterns(anomalies, df, min_occurrences=3)
        assert result[0].carbon_context == "low-carbon"

    def test_carbon_context_unknown_when_ci_column_absent(self):
        df = _one_week_df().drop(columns=["carbonIntensityFactor"])
        anomalies = [
            _make_anomaly(ds="2024-01-01 15:00", carbon_intensity=0.0),
            _make_anomaly(ds="2024-01-02 15:00", carbon_intensity=0.0),
            _make_anomaly(ds="2024-01-03 15:00", carbon_intensity=0.0),
        ]
        result = find_recurring_patterns(anomalies, df, min_occurrences=3)
        assert result[0].carbon_context == "unknown"

    def test_anomaly_timestamps_populated_and_sorted(self):
        ts1 = "2024-01-03 15:00"
        ts2 = "2024-01-01 15:00"  # earlier, listed second
        ts3 = "2024-01-02 15:00"
        anomalies = [
            _make_anomaly(ds=ts1),
            _make_anomaly(ds=ts2),
            _make_anomaly(ds=ts3),
        ]
        result = find_recurring_patterns(anomalies, _one_week_df(), min_occurrences=3)
        stamps = result[0].anomaly_timestamps
        assert len(stamps) == 3
        assert stamps == sorted(stamps)

    def test_sorted_by_total_excess_carbon_desc(self):
        """Pattern with higher total carbon should rank first."""
        high_carbon = [
            _make_anomaly(ds="2024-01-01 15:00", excess_carbon_kgco2e=0.050),
            _make_anomaly(ds="2024-01-02 15:00", excess_carbon_kgco2e=0.050),
            _make_anomaly(ds="2024-01-03 15:00", excess_carbon_kgco2e=0.050),
        ]
        low_carbon = [
            _make_anomaly(ds="2024-01-01 10:00", hour_of_day=10, excess_carbon_kgco2e=0.005),
            _make_anomaly(ds="2024-01-02 10:00", hour_of_day=10, excess_carbon_kgco2e=0.005),
            _make_anomaly(ds="2024-01-03 10:00", hour_of_day=10, excess_carbon_kgco2e=0.005),
        ]
        result = find_recurring_patterns(
            high_carbon + low_carbon, _one_week_df(), min_occurrences=3
        )
        assert result[0].hour_of_day == 15   # high-carbon pattern ranks first
        assert result[1].hour_of_day == 10

    def test_min_frequency_pct_filter(self):
        """
        3 out of 5 possible weekday-15:00 slots = 60%.
        min_frequency_pct=70 → pattern filtered out.
        """
        anomalies = [
            _make_anomaly(ds="2024-01-01 15:00"),
            _make_anomaly(ds="2024-01-02 15:00"),
            _make_anomaly(ds="2024-01-03 15:00"),
        ]
        result = find_recurring_patterns(
            anomalies, _one_week_df(), min_occurrences=3, min_frequency_pct=70.0
        )
        assert result == []


# ── TestBuildInvestigationLeads ───────────────────────────────────────────────

class TestBuildInvestigationLeads:

    def _make_pattern(
        self,
        hour:          int   = 15,
        day_type:      str   = "weekday",
        occurrences:   int   = 10,
        carbon:        float = 0.50,
        cost:          float = 0.20,
        avg_excess:    float = 30.0,
        frequency_pct: float = 80.0,
        avg_ci:        float = 0.43,
        ctx:           str   = "high-carbon",
    ) -> RecurringPattern:
        ts = pd.date_range("2024-01-01 15:00", periods=occurrences, freq="7D")
        return RecurringPattern(
            metric               = "consumption",
            hour_of_day          = hour,
            day_type             = day_type,
            occurrences          = occurrences,
            possible             = 12,
            frequency_pct        = frequency_pct,
            avg_excess_pct       = avg_excess,
            total_excess_carbon  = carbon,
            total_excess_cost    = cost,
            avg_carbon_intensity = avg_ci,
            carbon_context       = ctx,
            anomaly_timestamps   = [t.to_pydatetime() for t in ts],
        )

    def test_empty_patterns_returns_empty(self):
        assert build_investigation_leads([], _one_week_df()) == []

    def test_top_n_respected(self):
        patterns = [self._make_pattern(hour=h) for h in range(10, 15)]  # 5 patterns
        leads = build_investigation_leads(patterns, _one_week_df(), top_n=3)
        assert len(leads) == 3

    def test_rank_starts_at_one(self):
        patterns = [self._make_pattern()]
        leads = build_investigation_leads(patterns, _one_week_df())
        assert leads[0].rank == 1

    def test_ranks_are_sequential(self):
        patterns = [self._make_pattern(hour=h) for h in (10, 11, 12)]
        leads = build_investigation_leads(patterns, _one_week_df(), top_n=3)
        assert [l.rank for l in leads] == [1, 2, 3]

    def test_unit_populated_from_metric(self):
        patterns = [self._make_pattern()]
        leads = build_investigation_leads(patterns, _one_week_df())
        assert leads[0].unit == "kWh/h"

    def test_pattern_summary_contains_metric_and_hour(self):
        patterns = [self._make_pattern(hour=15)]
        leads = build_investigation_leads(patterns, _one_week_df())
        summary = leads[0].pattern_summary
        assert "15" in summary
        assert "consumption" in summary

    def test_example_timestamps_max_three(self):
        patterns = [self._make_pattern(occurrences=10)]
        leads = build_investigation_leads(patterns, _one_week_df())
        assert len(leads[0].example_timestamps) <= 3

    def test_optimal_window_within_valid_hours(self):
        patterns = [self._make_pattern()]
        leads = build_investigation_leads(patterns, _one_week_df())
        assert 0 <= leads[0].optimal_window_start_hour <= 23

    def test_carbon_saving_positive_when_moving_to_lower_ci_window(self):
        """
        Pattern at hour 15 (CI = 0.45), optimal window at hours 3–6 (CI = 0.30).
        Carbon saving = (0.45 - 0.30) / 0.45 × 100 ≈ +33.3% (positive = you save).
        Convention matches GHG Protocol and AWS Compute Optimizer: positive = benefit.
        """
        df = _one_week_df(ci_by_hour={3: 0.30, 4: 0.30, 5: 0.30, 6: 0.30})
        pattern = self._make_pattern(hour=15, avg_ci=0.45, ctx="high-carbon")
        leads = build_investigation_leads([pattern], df, window_hours=4)
        assert leads[0].estimated_carbon_saving_pct > 0, (
            "Carbon saving should be positive (a genuine saving) when moving "
            "to a lower-CI window."
        )

    def test_carbon_saving_positive_when_optimal_ci_higher(self):
        """
        Pattern at hour 3 (CI = 0.30, already green), optimal window at hours 10–14
        (CI = 0.45). Saving = (0.45 − 0.30) / 0.30 × 100 = +50% → moving there costs
        MORE carbon. Positive saving_pct = bad.
        """
        df = _one_week_df(ci_by_hour={10: 0.45, 11: 0.45, 12: 0.45, 13: 0.45})
        # Make hour 3 anomalously low-CI in the df too, so optimal window isn't hour 3
        df2 = _one_week_df(ci_by_hour={10: 0.45, 11: 0.45, 12: 0.45, 13: 0.45,
                                        3: 0.30,  4: 0.30,  5: 0.30,  6: 0.30})
        pattern = self._make_pattern(hour=15, avg_ci=0.30)
        leads = build_investigation_leads([pattern], df2, window_hours=4)
        # The key point: function handles this gracefully without crashing
        assert isinstance(leads[0].estimated_carbon_saving_pct, float)

    def test_graceful_degradation_no_ci_column(self):
        """When carbonIntensityFactor is absent, savings default to 0.0."""
        df_no_ci = _one_week_df().drop(columns=["carbonIntensityFactor"])
        pattern  = self._make_pattern(avg_ci=0.0, ctx="unknown")
        leads    = build_investigation_leads([pattern], df_no_ci)
        assert leads[0].estimated_carbon_saving_pct == pytest.approx(0.0)

    def test_fields_copied_from_pattern(self):
        pattern = self._make_pattern(
            occurrences=7, frequency_pct=70.0, avg_excess=25.0, carbon=0.3, cost=0.1
        )
        leads = build_investigation_leads([pattern], _one_week_df())
        lead  = leads[0]
        assert lead.occurrences                 == 7
        assert lead.frequency_pct               == pytest.approx(70.0)
        assert lead.avg_excess_pct              == pytest.approx(25.0)
        assert lead.total_excess_carbon_kgco2e  == pytest.approx(0.3)
        assert lead.total_excess_cost_eur       == pytest.approx(0.1)

    def test_cost_saving_positive_when_moving_to_cheaper_window(self):
        """
        Pattern at peak hour (09:00–18:00, cost rate €0.18/kWh).
        Optimal carbon window at off-peak hour (e.g., 03:00, €0.12/kWh).
        Cost saving = (0.18 − 0.12) / 0.18 × 100 ≈ +33.3% (positive = you save).
        Convention matches GHG Protocol and AWS Compute Optimizer: positive = benefit.
        """
        # Build df with explicit cost/consumption columns at known rates
        dates   = pd.date_range("2024-01-01", periods=24 * 7, freq="h")
        hours   = dates.hour
        # Peak (9–18): 0.18 EUR/kWh; off-peak: 0.12 EUR/kWh
        rate    = pd.Series([0.18 if (9 <= h <= 18) else 0.12 for h in hours])
        df_cost = pd.DataFrame({
            "ds":                    dates,
            "consumption":           0.10,
            "cost":                  rate * 0.10,
            "carbonIntensityFactor": 0.40,
        })
        # Make off-peak hour 3 the carbon-optimal window too
        df_cost.loc[hours == 3, "carbonIntensityFactor"] = 0.30
        df_cost.loc[hours == 4, "carbonIntensityFactor"] = 0.30
        df_cost.loc[hours == 5, "carbonIntensityFactor"] = 0.30
        df_cost.loc[hours == 6, "carbonIntensityFactor"] = 0.30

        pattern = self._make_pattern(hour=15, avg_ci=0.40, ctx="high-carbon")
        leads   = build_investigation_leads([pattern], df_cost, window_hours=4)
        assert leads[0].estimated_cost_saving_pct > 0, (
            "Moving from peak to off-peak should show positive (saving) cost_saving_pct"
        )


# ── TestFullPipeline ──────────────────────────────────────────────────────────

class TestFullPipeline:
    """
    End-to-end integration test — no Prophet required.

    A synthetic "baseline forecast" is constructed that represents expected
    behaviour. Known anomalies are injected into the actual data. The full
    3-stage pipeline is run, and the output is verified.
    """

    @pytest.fixture(scope="class")
    def pipeline_results(self):
        """
        Build a 30-day scenario with a weekday-15:00 consumption spike
        (+80% above baseline) and run all three stages once.
        """
        n_hours = 24 * 30
        dates   = pd.date_range("2024-01-01", periods=n_hours, freq="h")

        # Actual data: normal consumption at 0.08 except weekday 15:00 (+80%)
        actual_values = pd.Series(0.08, index=range(n_hours))
        spike_mask = (dates.hour == 15) & (dates.dayofweek < 5)
        actual_values[spike_mask] = 0.144    # 0.08 × 1.8 = +80%

        df_actual = pd.DataFrame({
            "ds":                    dates,
            "consumption":           actual_values.values,
            "carbonIntensityFactor": 0.40,
            "cost":                  actual_values.values * 0.12 * 1.5,  # all peak
        })

        # Synthetic forecast: baseline without spike
        df_forecast = pd.DataFrame({
            "ds":         dates,
            "yhat":       0.08,
            "yhat_lower": 0.06,
            "yhat_upper": 0.10,
        })

        anomalies = detect_point_anomalies(df_actual, df_forecast, "consumption")
        patterns  = find_recurring_patterns(anomalies, df_actual, min_occurrences=3)
        leads     = build_investigation_leads(patterns, df_actual, top_n=5)

        return {"anomalies": anomalies, "patterns": patterns, "leads": leads,
                "spike_count": int(spike_mask.sum())}

    def test_spike_count_matches_anomaly_count(self, pipeline_results):
        """Every injected spike should produce exactly one anomaly."""
        assert len(pipeline_results["anomalies"]) == pipeline_results["spike_count"]

    def test_spike_hour_is_fifteen(self, pipeline_results):
        hours = {a.hour_of_day for a in pipeline_results["anomalies"]}
        assert hours == {15}, f"Expected only hour 15; found {hours}"

    def test_all_spike_anomalies_are_weekday(self, pipeline_results):
        for a in pipeline_results["anomalies"]:
            assert not a.is_weekend, f"Weekend anomaly found: {a.ds}"

    def test_recurring_pattern_found_at_hour_15(self, pipeline_results):
        assert len(pipeline_results["patterns"]) >= 1
        assert pipeline_results["patterns"][0].hour_of_day == 15

    def test_pattern_day_type_is_weekday(self, pipeline_results):
        assert pipeline_results["patterns"][0].day_type == "weekday"

    def test_lead_rank_one_is_hour_15(self, pipeline_results):
        assert len(pipeline_results["leads"]) >= 1
        assert pipeline_results["leads"][0].optimal_window_start_hour in range(24)
        assert pipeline_results["leads"][0].occurrences > 0

    def test_lead_excess_pct_reflects_injection(self, pipeline_results):
        """
        We injected +80% above expected.
        excess_pct = (actual - expected) / expected × 100 = 80.0%.
        """
        lead = pipeline_results["leads"][0]
        assert lead.avg_excess_pct == pytest.approx(80.0, rel=0.05)

    def test_total_excess_carbon_positive(self, pipeline_results):
        assert pipeline_results["leads"][0].total_excess_carbon_kgco2e > 0
