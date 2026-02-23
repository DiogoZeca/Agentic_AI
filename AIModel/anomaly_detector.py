"""
Anomaly Detection Engine — pure analysis module.

No Prophet imports. No FastAPI. Takes DataFrames, returns structured dataclasses.

Three-stage pipeline
--------------------
1. detect_point_anomalies()    — flag individual hours where actual > yhat_upper
                                 (or actual < yhat_lower for deficit mode)
2. find_recurring_patterns()   — group anomalies by (hour_of_day, day_type);
                                 surfaces "same excess, same time window" signals
3. build_investigation_leads() — rank patterns by total excess carbon, add
                                 savings estimate vs the optimal low-carbon window

Caller responsibility
---------------------
The in-sample Prophet forecast (yhat ± bounds for historical timestamps) is
produced by the API/analysis layer and passed in as a DataFrame. Example:

    # api.py or carbon_analysis.py
    insample_fc = model.predict(periods=0, future_df=df)   # all historical rows
    anomalies   = detect_point_anomalies(df, insample_fc, "consumption")
    patterns    = find_recurring_patterns(anomalies, df, min_occurrences=3)
    leads       = build_investigation_leads(patterns, df, top_n=5)

This design keeps the engine fast and fully testable without Prophet.

Graceful degradation
---------------------
If `carbonIntensityFactor` or `cost` are absent from df_actual (real customer
data may not have these), all carbon/cost fields default to 0.0 and
carbon_context is "unknown". The anomaly shape detection still works.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


# ── Metric unit map (for human-readable output) ───────────────────────────────

_UNIT_MAP: dict[str, str] = {
    "consumption":                "kWh/h",
    "carbonEmissions":            "kgCO2e/h",
    "carbonIntensityFactor":      "kgCO2/kWh",
    "greenConsumptionPercentage": "%",
    "cpuUtilization":             "fraction",
    "functionalUnit":             "req/h",
    "cost":                       "EUR/h",
    "softwareCarbonIntensity":    "kgCO2e/req",
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Anomaly:
    """A single anomalous timestamp — actual value exceeded the forecast bound."""

    ds:                   datetime   # exact timestamp of the anomaly
    metric:               str        # which metric triggered it
    actual:               float      # measured value
    expected:             float      # Prophet point forecast (yhat)
    upper_bound:          float      # yhat_upper  (97.5th percentile)
    lower_bound:          float      # yhat_lower  (2.5th percentile)
    excess_abs:           float      # |actual - bound| — always positive
    excess_pct:           float      # (actual - expected) / |expected| × 100 (signed)
    hour_of_day:          int        # 0–23
    is_weekend:           bool       # Saturday or Sunday
    carbon_intensity:     float      # kgCO2/kWh at this timestamp (0.0 if absent)
    cost_eur_h:           float      # EUR/h at this timestamp (0.0 if absent)
    excess_carbon_kgco2e: float      # carbon cost of the excess energy
    excess_cost_eur:      float      # EUR cost of the excess consumption


@dataclass
class RecurringPattern:
    """
    A repeating anomaly at the same hour-of-day and day-type over the
    observation window — the core "investigation trigger."
    """

    metric:               str
    hour_of_day:          int
    day_type:             str        # "weekday" | "weekend"
    occurrences:          int        # how many anomalies in this slot
    possible:             int        # total times this slot appeared in history
    frequency_pct:        float      # occurrences / possible × 100
    avg_excess_pct:       float      # mean excess_pct across all anomalies
    total_excess_carbon:  float      # Σ excess_carbon_kgco2e across occurrences
    total_excess_cost:    float      # Σ excess_cost_eur across occurrences
    avg_carbon_intensity: float      # mean carbonIntensityFactor at anomaly hours
    carbon_context:       str        # "high-carbon" | "low-carbon" | "unknown"
    anomaly_timestamps:   list[datetime] = field(default_factory=list)


@dataclass
class InvestigationLead:
    """
    Actionable output for the EVIDEN analyst — structured investigation trigger.

    Example:
        "Weekday 15:00 consumption spike — 22 occurrences (88% of weekdays).
         Average 34% above baseline. Move to 03:00–07:00 to save −29% carbon,
         −22% cost."
    """

    rank:                        int        # 1 = highest priority
    metric:                      str
    unit:                        str
    pattern_summary:             str        # human-readable one-liner
    occurrences:                 int
    frequency_pct:               float
    avg_excess_pct:              float
    total_excess_carbon_kgco2e:  float
    total_excess_cost_eur:       float
    optimal_window_start_hour:   int        # start of lowest-carbon 4h window
    optimal_window_end_hour:     int        # start + window_hours
    estimated_carbon_saving_pct: float      # negative = saving
    estimated_cost_saving_pct:   float      # negative = saving
    carbon_context:              str
    example_timestamps:          list[str]  = field(default_factory=list)  # first 3 ds


# ── Public API ────────────────────────────────────────────────────────────────

def detect_point_anomalies(
    df_actual:   pd.DataFrame,
    df_forecast: pd.DataFrame,
    metric:      str,
    direction:   str = "excess",   # "excess" | "deficit" | "both"
) -> list[Anomaly]:
    """
    Flag timestamps where the actual metric value exceeded the Prophet forecast
    bounds.

    Args:
        df_actual:   Historical DataFrame with at least 'ds' and `metric`
                     columns. Optional enrichment: 'carbonIntensityFactor',
                     'cost', 'consumption'.
        df_forecast: In-sample Prophet forecast — must have 'ds', 'yhat',
                     'yhat_lower', 'yhat_upper'. Produced by:
                         model.predict(periods=0, future_df=df_actual)
        metric:      Column name in df_actual to compare against bounds.
        direction:   "excess"  → flag actual > yhat_upper   (over-consumption)
                     "deficit" → flag actual < yhat_lower   (under-utilisation)
                     "both"    → flag either condition

    Returns:
        List of Anomaly objects, one per anomalous hour, sorted by timestamp.
    """
    if metric not in df_actual.columns:
        raise KeyError(f"Metric '{metric}' not found in df_actual columns: {list(df_actual.columns)}")

    required_fc = {"ds", "yhat", "yhat_lower", "yhat_upper"}
    missing_fc  = required_fc - set(df_forecast.columns)
    if missing_fc:
        raise KeyError(f"df_forecast is missing columns: {missing_fc}")

    # Align on timestamp
    actual_ts   = df_actual[["ds", metric]].copy()
    actual_ts["ds"] = pd.to_datetime(actual_ts["ds"])

    forecast_ts = df_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_ts["ds"] = pd.to_datetime(forecast_ts["ds"])

    merged = actual_ts.merge(forecast_ts, on="ds", how="inner")
    if merged.empty:
        return []

    # Optional enrichment columns — default to 0.0 if absent
    has_ci   = "carbonIntensityFactor" in df_actual.columns
    has_cost = "cost" in df_actual.columns
    has_cons = "consumption" in df_actual.columns

    if has_ci:
        ci_series = df_actual.set_index(pd.to_datetime(df_actual["ds"]))["carbonIntensityFactor"]
    if has_cost:
        cost_series = df_actual.set_index(pd.to_datetime(df_actual["ds"]))["cost"]
    if has_cons:
        cons_series = df_actual.set_index(pd.to_datetime(df_actual["ds"]))["consumption"]

    # Determine anomaly mask
    actual_col = merged[metric].values
    upper_col  = merged["yhat_upper"].values
    lower_col  = merged["yhat_lower"].values
    yhat_col   = merged["yhat"].values

    if direction == "excess":
        mask = actual_col > upper_col
    elif direction == "deficit":
        mask = actual_col < lower_col
    elif direction == "both":
        mask = (actual_col > upper_col) | (actual_col < lower_col)
    else:
        raise ValueError(f"direction must be 'excess', 'deficit', or 'both'. Got '{direction}'")

    anomalies: list[Anomaly] = []
    ds_col = pd.to_datetime(merged["ds"])

    for idx in np.where(mask)[0]:
        ts          = ds_col.iloc[idx]
        actual_val  = float(actual_col[idx])
        expected    = float(yhat_col[idx])
        upper_b     = float(upper_col[idx])
        lower_b     = float(lower_col[idx])
        is_excess   = actual_val > upper_b

        excess_abs = abs(actual_val - upper_b) if is_excess else abs(lower_b - actual_val)
        denom      = abs(expected) if abs(expected) > 1e-10 else 1e-10
        excess_pct = (actual_val - expected) / denom * 100.0

        # Carbon intensity at this timestamp
        ci = float(ci_series.get(ts, 0.0)) if has_ci else 0.0

        # Cost per hour at this timestamp
        cost_h = float(cost_series.get(ts, 0.0)) if has_cost else 0.0

        # Excess carbon: only meaningful for energy and emission metrics
        exc_carbon = _compute_excess_carbon(metric, excess_abs, ci)

        # Excess cost: approximate from cost/consumption rate at this timestamp
        if has_cost and has_cons:
            cons_h = float(cons_series.get(ts, 0.0))
            exc_cost = _compute_excess_cost(metric, excess_abs, cost_h, cons_h)
        else:
            exc_cost = 0.0

        anomalies.append(Anomaly(
            ds                   = ts.to_pydatetime(),
            metric               = metric,
            actual               = round(actual_val, 8),
            expected             = round(expected, 8),
            upper_bound          = round(upper_b, 8),
            lower_bound          = round(lower_b, 8),
            excess_abs           = round(excess_abs, 8),
            excess_pct           = round(excess_pct, 2),
            hour_of_day          = int(ts.hour),
            is_weekend           = bool(ts.dayofweek >= 5),
            carbon_intensity     = round(ci, 4),
            cost_eur_h           = round(cost_h, 6),
            excess_carbon_kgco2e = round(exc_carbon, 8),
            excess_cost_eur      = round(exc_cost, 6),
        ))

    return sorted(anomalies, key=lambda a: a.ds)


def find_recurring_patterns(
    anomalies:        list[Anomaly],
    df_actual:        pd.DataFrame,
    min_occurrences:  int   = 3,
    min_frequency_pct: float = 0.0,
) -> list[RecurringPattern]:
    """
    Group anomalies by (hour_of_day, day_type) and surface recurring patterns.

    A "recurring pattern" is the same metric exceeding its expected bound at
    the same time-of-day window repeatedly. This is the "15:00 weekday spike"
    signal relevant to EVIDEN.

    Args:
        anomalies:         Output of detect_point_anomalies().
        df_actual:         Original historical DataFrame — used to compute
                           the total number of times each (hour, day_type)
                           slot appeared (denominator for frequency_pct).
        min_occurrences:   Minimum anomaly count before a pattern is reported.
                           Default 3 — avoids spurious one-off events.
        min_frequency_pct: Minimum occurrence rate (%) to report.
                           Default 0.0 (no filter; rely on min_occurrences).

    Returns:
        List of RecurringPattern objects, sorted by total_excess_carbon
        descending (highest carbon impact first).
    """
    if not anomalies:
        return []

    # Pre-compute dataset-wide median carbon intensity for context labelling
    has_ci      = "carbonIntensityFactor" in df_actual.columns
    median_ci   = float(df_actual["carbonIntensityFactor"].median()) if has_ci else None

    # Count possible occurrences per (hour, day_type) in the historical window
    ds_series  = pd.to_datetime(df_actual["ds"])
    hour_series = ds_series.dt.hour
    dow_series  = ds_series.dt.dayofweek  # 0=Mon, 6=Sun

    def _possible(hour: int, day_type: str) -> int:
        h_mask = hour_series == hour
        if day_type == "weekday":
            d_mask = dow_series < 5
        else:
            d_mask = dow_series >= 5
        return int((h_mask & d_mask).sum())

    # Group anomalies by (metric, hour_of_day, day_type)
    groups: dict[tuple[str, int, str], list[Anomaly]] = {}
    for a in anomalies:
        day_type = "weekend" if a.is_weekend else "weekday"
        key = (a.metric, a.hour_of_day, day_type)
        groups.setdefault(key, []).append(a)

    patterns: list[RecurringPattern] = []

    for (metric, hour, day_type), group in groups.items():
        if len(group) < min_occurrences:
            continue

        possible     = _possible(hour, day_type)
        freq_pct     = round(len(group) / possible * 100, 1) if possible > 0 else 0.0

        if freq_pct < min_frequency_pct:
            continue

        avg_excess   = round(float(np.mean([a.excess_pct for a in group])), 2)
        total_carbon = round(float(sum(a.excess_carbon_kgco2e for a in group)), 6)
        total_cost   = round(float(sum(a.excess_cost_eur      for a in group)), 4)
        avg_ci       = round(float(np.mean([a.carbon_intensity for a in group])), 4)

        if median_ci is None or avg_ci == 0.0:
            ctx = "unknown"
        elif avg_ci > median_ci:
            ctx = "high-carbon"
        else:
            ctx = "low-carbon"

        patterns.append(RecurringPattern(
            metric               = metric,
            hour_of_day          = hour,
            day_type             = day_type,
            occurrences          = len(group),
            possible             = possible,
            frequency_pct        = freq_pct,
            avg_excess_pct       = avg_excess,
            total_excess_carbon  = total_carbon,
            total_excess_cost    = total_cost,
            avg_carbon_intensity = avg_ci,
            carbon_context       = ctx,
            anomaly_timestamps   = sorted(a.ds for a in group),
        ))

    # Primary sort: total excess carbon desc; secondary: occurrences desc
    patterns.sort(key=lambda p: (-p.total_excess_carbon, -p.occurrences))
    return patterns


def build_investigation_leads(
    patterns:     list[RecurringPattern],
    df_actual:    pd.DataFrame,
    window_hours: int = 4,
    top_n:        int = 5,
) -> list[InvestigationLead]:
    """
    Convert recurring patterns into ranked investigation leads.

    For each pattern this function:
      - Identifies the lowest-carbon scheduling window in the day (optimal window)
      - Estimates carbon saving if the anomalous workload were moved there
      - Estimates cost saving from peak → off-peak shift

    Args:
        patterns:     Output of find_recurring_patterns().
        df_actual:    Historical DataFrame — used to compute carbon intensity
                      and cost rate profiles by hour.
        window_hours: Length of the "optimal window" block. Default 4h.
        top_n:        Maximum investigation leads to return.

    Returns:
        List of InvestigationLead, ranked 1…N by total excess carbon.
    """
    if not patterns:
        return []

    # Hourly profile tables (mean values per hour-of-day across the dataset)
    ds_dt    = pd.to_datetime(df_actual["ds"])
    hour_col = ds_dt.dt.hour

    has_ci   = "carbonIntensityFactor" in df_actual.columns
    has_cost = "cost" in df_actual.columns
    has_cons = "consumption" in df_actual.columns

    hourly_ci   = (
        df_actual.assign(_h=hour_col).groupby("_h")["carbonIntensityFactor"].mean()
        if has_ci else pd.Series(dtype=float)
    )
    hourly_cost_rate = pd.Series(dtype=float)
    if has_cost and has_cons:
        cost_rate = df_actual["cost"] / df_actual["consumption"].clip(lower=1e-9)
        hourly_cost_rate = (
            pd.DataFrame({"_h": hour_col, "rate": cost_rate})
            .groupby("_h")["rate"].mean()
        )

    # Find the optimal (lowest-carbon) scheduling window in the 24h cycle
    opt_start = _find_optimal_window_start(hourly_ci, window_hours) if has_ci else 0
    opt_hours = [(opt_start + i) % 24 for i in range(window_hours)]

    # Mean CI at optimal window
    avg_ci_opt = (
        float(hourly_ci.reindex(opt_hours).mean()) if has_ci else 0.0
    )
    # Mean cost rate at optimal window
    avg_rate_opt = (
        float(hourly_cost_rate.reindex(opt_hours).mean())
        if not hourly_cost_rate.empty else 0.0
    )

    leads: list[InvestigationLead] = []

    for rank, pat in enumerate(patterns[:top_n], start=1):
        # Carbon saving: shift from pattern CI to optimal window CI
        ci_pat = pat.avg_carbon_intensity if pat.avg_carbon_intensity > 0 else avg_ci_opt
        if ci_pat > 0 and avg_ci_opt > 0:
            carbon_saving_pct = round((avg_ci_opt - ci_pat) / ci_pat * 100, 1)
        else:
            carbon_saving_pct = 0.0

        # Cost saving: shift from pattern cost rate to optimal window rate
        avg_rate_pat = (
            float(hourly_cost_rate.get(pat.hour_of_day, avg_rate_opt))
            if not hourly_cost_rate.empty else 0.0
        )
        if avg_rate_pat > 0:
            cost_saving_pct = round((avg_rate_opt - avg_rate_pat) / avg_rate_pat * 100, 1)
        else:
            cost_saving_pct = 0.0

        summary = _build_summary(pat)
        examples = [
            ts.strftime("%Y-%m-%d %H:%M")
            for ts in pat.anomaly_timestamps[:3]
        ]

        leads.append(InvestigationLead(
            rank                        = rank,
            metric                      = pat.metric,
            unit                        = _UNIT_MAP.get(pat.metric, ""),
            pattern_summary             = summary,
            occurrences                 = pat.occurrences,
            frequency_pct               = pat.frequency_pct,
            avg_excess_pct              = pat.avg_excess_pct,
            total_excess_carbon_kgco2e  = pat.total_excess_carbon,
            total_excess_cost_eur       = pat.total_excess_cost,
            optimal_window_start_hour   = opt_start,
            optimal_window_end_hour     = (opt_start + window_hours) % 24,
            estimated_carbon_saving_pct = carbon_saving_pct,
            estimated_cost_saving_pct   = cost_saving_pct,
            carbon_context              = pat.carbon_context,
            example_timestamps          = examples,
        ))

    return leads


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_excess_carbon(metric: str, excess_abs: float, carbon_intensity: float) -> float:
    """
    Carbon cost of the excess energy above the forecast upper bound.

    - consumption anomaly:    excess_abs (kWh) × CI (kgCO2/kWh)  = kgCO2e
    - carbonEmissions anomaly: excess_abs is already kgCO2e/h → 1h window = kgCO2e
    - all other metrics:      no direct carbon translation → 0.0
    """
    if metric == "consumption":
        return excess_abs * carbon_intensity
    if metric == "carbonEmissions":
        return excess_abs          # 1-hour measurement window
    return 0.0


def _compute_excess_cost(
    metric:       str,
    excess_abs:   float,
    cost_eur_h:   float,
    actual_cons:  float,
) -> float:
    """
    EUR cost of the excess consumption above the forecast upper bound.

    Uses the actual cost/consumption rate at that timestamp as the marginal
    rate. Only meaningful for 'consumption' and 'cost' metrics.
    """
    if metric == "consumption":
        rate = cost_eur_h / max(actual_cons, 1e-9)   # EUR/kWh at this hour
        return excess_abs * rate
    if metric == "cost":
        return excess_abs                             # already in EUR
    return 0.0


def _find_optimal_window_start(hourly_ci: pd.Series, window_hours: int = 4) -> int:
    """
    Sliding-window search over 24 hours (with wrap-around) to find the
    starting hour of the contiguous block with the lowest mean carbon intensity.

    Args:
        hourly_ci:    pd.Series indexed 0–23 with mean CI per hour.
        window_hours: Block length.

    Returns:
        Starting hour (0–23) of the best window.
    """
    if hourly_ci.empty:
        return 2    # sensible default: early morning

    best_start = 0
    best_avg   = float("inf")

    for start in range(24):
        hours = [(start + i) % 24 for i in range(window_hours)]
        avg   = float(hourly_ci.reindex(hours).mean())
        if avg < best_avg:
            best_avg   = avg
            best_start = start

    return best_start


def _build_summary(pattern: RecurringPattern) -> str:
    """Human-readable one-liner for an InvestigationLead."""
    day  = pattern.day_type.capitalize()     # "Weekday" | "Weekend"
    hour = f"{pattern.hour_of_day:02d}:00"
    ctx  = f" [{pattern.carbon_context}]" if pattern.carbon_context != "unknown" else ""
    return (
        f"{day} {hour} {pattern.metric} anomaly — "
        f"{pattern.occurrences} occurrences ({pattern.frequency_pct:.0f}% of {pattern.day_type}s){ctx}"
    )


# ── Quick standalone demo ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    from data_generator import generate_energy_carbon_data
    from prophet_model import EnergyProphet

    print("Generating 90-day dataset …")
    df = generate_energy_carbon_data(periods=24 * 90)

    # ── Inject a synthetic anomaly: weekday 15:00 consumption spike ───────────
    # Every weekday at 15:00 over the last 60 days: add +50% excess
    ts = pd.to_datetime(df["ds"])
    spike_mask = (
        (ts.dt.hour == 15)
        & (ts.dt.dayofweek < 5)
        & (ts > ts.max() - pd.Timedelta(days=60))
    )
    df_spike = df.copy()
    df_spike.loc[spike_mask, "consumption"] *= 1.5
    df_spike.loc[spike_mask, "carbonEmissions"] *= 1.5

    print(f"Injected spikes at {spike_mask.sum()} weekday-15:00 timestamps.\n")

    # ── Fit Prophet on spiked data ────────────────────────────────────────────
    print("Fitting Prophet on 'consumption' …")
    model = EnergyProphet(regressors=["functionalUnit"])
    model.fit(df_spike, "consumption")          # model learns the spiked data too

    # In-sample forecast (yhat ± bounds for every training timestamp)
    print("Computing in-sample forecast …")
    insample_fc = model.predict(periods=0, future_df=df_spike)

    # ── Stage 1: point anomalies ──────────────────────────────────────────────
    print("Detecting point anomalies …")
    anomalies = detect_point_anomalies(df_spike, insample_fc, "consumption")
    print(f"  Found {len(anomalies)} point anomalies.\n")

    # ── Stage 2: recurring patterns ───────────────────────────────────────────
    print("Finding recurring patterns (min 3 occurrences) …")
    patterns = find_recurring_patterns(anomalies, df_spike, min_occurrences=3)
    print(f"  Found {len(patterns)} recurring patterns:")
    for p in patterns:
        print(
            f"    Hour {p.hour_of_day:02d}:00 {p.day_type:7s} | "
            f"{p.occurrences}/{p.possible} ({p.frequency_pct:.0f}%) | "
            f"avg +{p.avg_excess_pct:.1f}% | "
            f"total carbon {p.total_excess_carbon:.4f} kgCO2e | "
            f"{p.carbon_context}"
        )

    # ── Stage 3: investigation leads ──────────────────────────────────────────
    print("\nBuilding investigation leads (top 3) …")
    leads = build_investigation_leads(patterns, df_spike, top_n=3)
    print()
    for lead in leads:
        print(f"  [{lead.rank}] {lead.pattern_summary}")
        print(f"      Occurrences    : {lead.occurrences} ({lead.frequency_pct:.0f}%)")
        print(f"      Avg excess     : +{lead.avg_excess_pct:.1f}% above baseline")
        print(f"      Excess carbon  : {lead.total_excess_carbon_kgco2e:.4f} kgCO2e total")
        print(f"      Excess cost    : €{lead.total_excess_cost_eur:.2f} total")
        print(f"      Optimal window : {lead.optimal_window_start_hour:02d}:00–{lead.optimal_window_end_hour:02d}:00")
        print(f"      Carbon saving  : {lead.estimated_carbon_saving_pct:+.1f}%")
        print(f"      Cost saving    : {lead.estimated_cost_saving_pct:+.1f}%")
        print(f"      Examples       : {', '.join(lead.example_timestamps)}")
        print()
