"""Engineer features and spike labels from node-level 5-min aggregations.

Reads cluster_agg.parquet (output of spike_preprocessor.py) and produces:

  cluster_features.parquet  — one row per (machine_id, bucket) with all
                               features and the spike_in_60m label.
  spike_thresholds.parquet  — per-machine p95 CPU threshold, computed from
                               training windows only to prevent label leakage.

Feature set (37 columns)
------------------------
Raw signals at time t:
  total_cpu, peak_cpu, total_mem, peak_mem, disk_io, n_tasks

Lag history (CPU load N windows ago):
  cpu_lag_{1,12,24}  (Phase 2: cpu_lag_2/3/6 dropped — gain 0.010 combined)

Trend (exponentially weighted moving averages):
  cpu_ewma_6, cpu_ewma_24

Derivatives:
  cpu_delta_1       — first difference: total_cpu[t] - total_cpu[t-1]
  cpu_delta_2       — second difference (acceleration): delta_1[t] - delta_1[t-1]
  cpu_rolling_std_6 — rolling std over last 6 windows (load volatility)

Machine-relative:
  task_dominance   — peak_cpu / total_cpu (single-task monopolisation risk)
  cpu_vs_p95       — total_cpu / machine p95 threshold (> 1.0 = already spiking)
  cpu_vs_p95_delta — rate of change of the normalised signal
  peak_cpu_vs_p95  — peak_cpu / machine p95 threshold (near-miss detection)

Spike history (highest expected predictive gain):
  spike_now             — is total_cpu[t] already above threshold? (binary)
  spike_in_last_1       — was t-1 a spike? (binary)
  spike_in_last_3       — any spike in t-3..t-1? (binary)
  spike_in_last_6       — any spike in t-6..t-1? (binary)
  time_since_last_spike — windows since last exceedance (capped at 24)
  cpu_spike_rate_24     — fraction of previous 24 windows that were spiking

  LEAKAGE NOTE: all spike history features are derived exclusively from
  total_cpu[t'] > threshold for t' < t.  They never use spike_in_60m (the
  future label).  Using spike_in_60m[t-1] as a feature would inject
  knowledge of windows t..t+11 into training and is explicitly forbidden.

Cluster-level (cross-sectional, same timestamp t):
  cluster_cpu_p90        — p90 of total_cpu across all machines at bucket t
  machine_rank_in_cluster — percentile rank of this machine vs all at bucket t

Time-of-day / day-of-week (harmonic encoding from bucket index):
  hour_sin, hour_cos — sin/cos of position within 24-hour day
  dow_sin,  dow_cos  — sin/cos of position within 7-day week

Severity label (Phase 3)
------------------------
  severity_in_60m = 2  (severe)   if any window in [t+1, t+12] exceeds the machine's
                                   p99 CPU threshold (computed on training data only).
                  = 1  (moderate)  if no p99 exceedance but p95 is exceeded in [t+1, t+12].
                  = 0  (no_spike)  if all look-ahead windows are below the p95 threshold.
                  = NaN for the last 12 rows per machine (no full look-ahead).

Gap handling (key correctness concern)
---------------------------------------
Not every machine has a row in every 5-min bucket (idle periods leave gaps).
Naively shifting rows would assign the wrong lag values — a value from 30 min
ago appearing as lag_1. Before computing features each machine's time series
is reindexed to a contiguous bucket range and idle windows are filled with 0.
Features are computed on this gap-free series. Spike labels also use the
gap-free series so look-ahead operates over the correct time range, not just
the next N activity records. Only original (non-filled) rows appear in the
output.

Input  : data/cluster_agg.parquet
Output : data/cluster_features.parquet  (~1-2 GB)
         data/spike_thresholds.parquet  (small — one row per machine)

Usage:
    python spike_feature_engineer.py
    python spike_feature_engineer.py --input  data/cluster_agg.parquet \\
                                      --output data/cluster_features.parquet \\
                                      --thresholds data/spike_thresholds.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(message)s",
    datefmt = "%H:%M:%S",
    stream  = sys.stdout,
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_BUCKET_US:            int   = 300_000_000   # 5 minutes in microseconds
_HORIZON:              int   = 12            # 60-min look-ahead (12 × 5-min windows)
_LAGS:                 tuple = (1, 12, 24)   # Phase 2: dropped lag_2/3/6 (combined gain 0.010)
_EWMA_HALFLIVES:       tuple = (6, 24)
_TRAIN_RATIO:          float = 0.6    # 60% of buckets — must match spike_classifier._TRAIN_RATIO
_SPIKE_QUANTILE:       float = 0.95   # moderate exceedance boundary
_SPIKE_QUANTILE_P99:   float = 0.99   # severe exceedance boundary (Phase 3)
_MAX_TIME_SINCE_SPIKE: int   = 24    # cap for time_since_last_spike (24 × 5 min = 2 h)
_BUCKETS_PER_DAY:      int   = 288   # 24 h × 12 buckets/h (5-min windows)

# Time-interval window sizes — grouped here so tuning one value updates all usages.
# All sizes are in 5-min bucket units.  The horizon (12 buckets = 60 min) sets the
# upper bound on look-ahead; all look-back windows should stay ≤ _HORIZON to avoid
# redundancy with the lag features.
_ROLLING_STD_WINDOW:     int   = 6          # 30-min volatility window  (6 × 5-min buckets)
_CPU_DELTA_LAGS:         tuple = (1, 2)     # first and second differences of CPU load
_SPIKE_HISTORY_WINDOWS:  tuple = (1, 3, 6)  # lookback windows for p95 spike history features
_SEVERE_HISTORY_WINDOWS: tuple = (1, 3, 6)  # lookback windows for p99 spike history features

_FEATURE_COLS: list[str] = [
    # identifiers
    "machine_id", "bucket", "time_us",
    # raw signals
    "total_cpu", "peak_cpu", "total_mem", "peak_mem", "disk_io", "n_tasks",
    # derived raw: CPU load per concurrent task (Phase 5)
    "cpu_per_task",
    # lag history (Phase 2: dropped cpu_lag_2/3/6 — combined gain 0.010)
    "cpu_lag_1", "cpu_lag_12", "cpu_lag_24",
    # trend
    "cpu_ewma_6", "cpu_ewma_24",
    # derivatives
    "cpu_delta_1", "cpu_delta_2", "cpu_rolling_std_6",
    # machine-relative
    "task_dominance", "cpu_vs_p95", "cpu_vs_p95_delta",
    "cpu_vs_p95_slope_3",   # avg rate-of-change of cpu/p95 over last 3 buckets (15 min)
    "cpu_vs_p95_slope_6",   # avg rate-of-change of cpu/p95 over last 6 buckets (30 min)
    "time_to_p95_3",        # estimated buckets until cpu/p95 = 1.0 at current slope; clipped [-24, 24]
    "peak_cpu_vs_p95",      # peak_cpu / p95 threshold — near-miss detection (Phase 2)
    # spike history — derived from raw metric only, never from spike_in_60m label
    "spike_now", "spike_in_last_1", "spike_in_last_3", "spike_in_last_6",
    "time_since_last_spike",
    "cpu_spike_rate_24",    # fraction of previous 24 windows spiking (Phase 2)
    # cluster-level (cross-sectional, same bucket t)
    "cluster_cpu_p90", "machine_rank_in_cluster",
    # time-of-day (harmonic encoding; dow dropped in Phase 5 — confound on 7-day dataset)
    "hour_sin", "hour_cos",
    # p99-level features (Fix 1)
    "spike_severe_now",       # is total_cpu[t] already above p99 threshold? (binary)
    "cpu_vs_p99",             # total_cpu / machine p99 threshold
    "peak_cpu_vs_p99",        # peak_cpu / machine p99 threshold
    "band_position",          # fractional position in moderate band [p95, p99]
    "band_width",             # width of moderate zone for this machine (scalar)
    "spike_severe_in_last_1", # was t-1 a severe spike? (binary, shift(1) guarded)
    "spike_severe_in_last_3", # any severe spike in t-3..t-1? (binary)
    "spike_severe_in_last_6", # any severe spike in t-6..t-1? (binary)
    # multi-horizon binary labels (Phase 4 — binary spike flag at shorter horizons)
    "spike_in_15m",    # binary: any p95 exceedance in next  3 windows (15 min)
    "spike_in_30m",    # binary: any p95 exceedance in next  6 windows (30 min)
    "spike_in_45m",    # binary: any p95 exceedance in next  9 windows (45 min)
    # 3-class severity label (Phase 3: 0=no_spike, 1=moderate, 2=severe)
    "severity_in_60m",
]

_THRESHOLD_COLS: list[str] = ["machine_id", "threshold_p95", "threshold_p99"]

# ── Private helpers ───────────────────────────────────────────────────────────


def _compute_thresholds(df: pd.DataFrame, train_bucket_max: int) -> pd.DataFrame:
    """Compute per-machine p95 and p99 total_cpu from training windows only.

    Using only training data prevents the spike thresholds from being
    influenced by future high-load events (label leakage — Problem 2).

    Machines with no rows in the training portion receive the global
    training-data quantile as a fallback (Problem 3).

    Parameters
    ----------
    df               : full aggregated DataFrame.
    train_bucket_max : highest bucket index counted as training data.

    Returns
    -------
    pd.DataFrame with columns [machine_id, threshold_p95, threshold_p99].
    """
    train = df[df["bucket"] <= train_bucket_max]

    p95_per_machine: pd.Series = (
        train.groupby("machine_id")["total_cpu"].quantile(_SPIKE_QUANTILE)
    )
    p99_per_machine: pd.Series = (
        train.groupby("machine_id")["total_cpu"].quantile(_SPIKE_QUANTILE_P99)
    )

    global_p95 = float(train["total_cpu"].quantile(_SPIKE_QUANTILE))
    global_p99 = float(train["total_cpu"].quantile(_SPIKE_QUANTILE_P99))

    all_ids = set(df["machine_id"].unique())
    missing = all_ids - set(p95_per_machine.index)
    if missing:
        log.warning(
            "  %d machine(s) absent from training data — "
            "assigning global thresholds (p95=%.4f, p99=%.4f) as fallback",
            len(missing),
            global_p95,
            global_p99,
        )
        idx = pd.Index(sorted(missing), name="machine_id")
        p95_per_machine = pd.concat(
            [p95_per_machine, pd.Series(global_p95, index=idx, dtype="float64")]
        ).sort_index()
        p99_per_machine = pd.concat(
            [p99_per_machine, pd.Series(global_p99, index=idx, dtype="float64")]
        ).sort_index()

    # Guard: p99 must be strictly above p95 — degenerate when too few training samples.
    degenerate = p99_per_machine <= p95_per_machine
    if degenerate.any():
        n_deg = int(degenerate.sum())
        log.warning(
            "  %d machine(s) have p99 <= p95 (too few training samples) — "
            "setting p99 = p95 * 1.1",
            n_deg,
        )
        p99_per_machine = p99_per_machine.where(~degenerate, p95_per_machine * 1.1)

    thresh_df = pd.DataFrame({
        "machine_id":    p95_per_machine.index,
        "threshold_p95": p95_per_machine.values,
        "threshold_p99": p99_per_machine.values,
    })

    # Enforce minimum 10% gap between p95 and p99 to avoid degenerate moderate zones
    # (affects ~12.6% of machines with very similar p95/p99 values)
    min_gap_mask = thresh_df["threshold_p99"] < thresh_df["threshold_p95"] * 1.10
    n_adjusted = int(min_gap_mask.sum())
    if n_adjusted:
        thresh_df.loc[min_gap_mask, "threshold_p99"] = (
            thresh_df.loc[min_gap_mask, "threshold_p95"] * 1.10
        ).astype("float32")
        log.info("  Enforced min 10%% p99/p95 gap for %d machines", n_adjusted)

    return thresh_df



def _engineer_machine(
    group:          pd.DataFrame,
    machine_id:     int,
    threshold:      float,
    threshold_p99:  float = 0.0,
) -> pd.DataFrame:
    """Reindex to fill bucket gaps, then compute all features.

    Returns the *full* reindexed series (original rows plus gap-filled rows).
    A boolean column ``_original`` marks which rows were in the input; callers
    must filter on it after adding the spike label.

    Gap-filled rows use 0 for all CPU/memory/disk signals — an idle machine
    exerts no load, so 0 is the physically correct value, not a sentinel.

    Leakage discipline
    ------------------
    All spike history features are computed from ``spike_now`` (whether
    ``total_cpu[t] > threshold`` at the current window).  They are never
    derived from ``spike_in_60m``, which encodes future windows t+1..t+12.

    The single "leakage firewall" is::

        exc = spike_now.shift(1).fillna(0.0)

    Every spike history feature that looks backwards uses this shifted series,
    ensuring no information from time t ever reaches a feature for time t.

    Parameters
    ----------
    group         : rows for a single machine_id from the aggregated parquet.
    machine_id    : the machine identifier (int).
    threshold     : per-machine p95 CPU threshold (from training data only).
    threshold_p99 : per-machine p99 CPU threshold (from training data only).
                    When 0.0 (default), p99-level features are set to 0.
    """
    g = group.set_index("bucket").sort_index()

    min_b = int(g.index.min())
    max_b = int(g.index.max())

    # Remember which buckets actually had observations (Problem 1)
    original_buckets: frozenset[int] = frozenset(g.index.tolist())

    # Reindex to fill gaps with NaN, then detect originals before filling
    g = g.reindex(range(min_b, max_b + 1))
    g.index.name = "bucket"

    g["_original"]  = g.index.isin(original_buckets)
    g["machine_id"] = machine_id
    g["time_us"]    = g.index.astype("int64") * _BUCKET_US

    # Fill idle windows with 0 (Problem 1 fix)
    for col in ("total_cpu", "peak_cpu", "total_mem", "peak_mem", "disk_io"):
        g[col] = g[col].fillna(0.0)
    g["n_tasks"] = g["n_tasks"].fillna(0).astype("int32")

    # cpu_per_task: intensity per concurrent task.
    # clip(lower=1) prevents division by zero for gap-filled windows where n_tasks=0.
    # Idle windows have total_cpu=0 too, so the result is 0/1 = 0 (correct).
    g["cpu_per_task"] = (g["total_cpu"] / g["n_tasks"].clip(lower=1)).astype("float32")

    cpu = g["total_cpu"]

    # ── Lag features ──────────────────────────────────────────────────────────
    for lag in _LAGS:
        g[f"cpu_lag_{lag}"] = cpu.shift(lag).fillna(0.0)

    # ── EWMA features ─────────────────────────────────────────────────────────
    for halflife in _EWMA_HALFLIVES:
        g[f"cpu_ewma_{halflife}"] = cpu.ewm(halflife=halflife, adjust=False).mean()

    # ── Derivative features ───────────────────────────────────────────────────
    delta_1 = cpu - cpu.shift(1).fillna(0.0)
    g["cpu_delta_1"]      = delta_1
    g["cpu_delta_2"]      = (delta_1 - delta_1.shift(1).fillna(0.0)).astype("float32")

    # Rolling std over last _ROLLING_STD_WINDOW windows — captures load volatility.
    # min_periods=1: returns 0 on first row (std of one value = NaN → fillna 0).
    g["cpu_rolling_std_6"] = (
        cpu.rolling(_ROLLING_STD_WINDOW, min_periods=1).std().fillna(0.0).astype("float32")
    )

    # ── Task dominance ────────────────────────────────────────────────────────
    g["task_dominance"] = np.where(
        g["total_cpu"] > 0,
        g["peak_cpu"] / g["total_cpu"],
        0.0,
    )

    # ── Machine-relative derivative ───────────────────────────────────────────
    # cpu_vs_p95_delta = rate of change of normalised signal = delta_1 / threshold.
    # Equivalent to (cpu[t] - cpu[t-1]) / p95 — tells the model how fast the
    # machine is moving toward or away from its spike boundary.
    # cpu_vs_p95 itself is added in engineer() after threshold mapping.
    if threshold > 0:
        g["cpu_vs_p95_delta"] = (delta_1 / threshold).astype("float32")
    else:
        g["cpu_vs_p95_delta"] = 0.0

    # ── Rate-of-approach features ─────────────────────────────────────────────
    # slope_3/slope_6: avg change in cpu/p95 per bucket over the last 3 and 6 buckets.
    # time_to_p95_3: estimated buckets until cpu/p95 hits 1.0 at the current slope_3 rate.
    #   Negative = already above threshold.  24.0 = not approaching (slope ≤ 0.01).
    if threshold > 0:
        cpu_norm = cpu / threshold
        slope_3  = (cpu_norm - cpu_norm.shift(3).fillna(0.0)) / 3
        slope_6  = (cpu_norm - cpu_norm.shift(6).fillna(0.0)) / 6
        g["cpu_vs_p95_slope_3"] = slope_3.astype("float32")
        g["cpu_vs_p95_slope_6"] = slope_6.astype("float32")
        distance = 1.0 - cpu_norm
        # Default when slope ≤ 0.01: -24 if already above threshold, +24 if below.
        default = np.where(distance.values < 0, -24.0, 24.0)
        # np.divide with out/where avoids the divide-by-zero RuntimeWarning that
        # np.where triggers even on the un-selected branch (both evaluated eagerly).
        ttx = np.divide(
            distance.values, slope_3.values,
            out   = default.astype(np.float64),
            where = slope_3.values > 0.01,
        )
        g["time_to_p95_3"] = np.clip(ttx, -24.0, 24.0).astype("float32")
    else:
        g["cpu_vs_p95_slope_3"] = np.float32(0.0)
        g["cpu_vs_p95_slope_6"] = np.float32(0.0)
        g["time_to_p95_3"]      = np.float32(0.0)

    # peak_cpu / threshold — near-miss detection.
    # cpu_vs_p95 uses total_cpu (5-min average); this uses peak_cpu (instantaneous max).
    # Machines where the average looks safe but the peak grazed the threshold are
    # currently invisible to the model — this feature surfaces them.
    peak = g["peak_cpu"].values
    # np.divide with out/where avoids the spurious RuntimeWarning: divide by zero
    # that np.where triggers even when the zero-threshold branch is not selected
    # (NumPy evaluates both branches before masking).
    g["peak_cpu_vs_p95"] = np.divide(
        peak, threshold,
        out   = np.zeros(len(peak), dtype="float32"),
        where = threshold > 0,
    )

    # ── p99-level features (Fix 1) ────────────────────────────────────────────
    band_w = max(float(threshold_p99) - float(threshold), 1e-6)   # avoid div-by-zero

    spike_severe_now  = (g["total_cpu"] > threshold_p99).astype("float32")
    g["spike_severe_now"]  = spike_severe_now.values

    g["cpu_vs_p99"]        = (g["total_cpu"] / threshold_p99).astype("float32") if threshold_p99 > 0 else np.float32(0.0)
    g["peak_cpu_vs_p99"]   = (g["peak_cpu"]  / threshold_p99).astype("float32") if threshold_p99 > 0 else np.float32(0.0)

    # Fractional position in moderate band: 0=at p95, 1=at p99, >1=severe zone
    g["band_position"] = (
        ((g["total_cpu"] - threshold) / band_w).clip(lower=0)
    ).astype("float32")

    # Width of moderate zone for this machine (scalar broadcast)
    g["band_width"] = np.float32(band_w)

    # Rolling p99 exceedance history — shift(1) guards against leakage
    exc_sev = spike_severe_now.shift(1).fillna(0).astype("float32")
    g["spike_severe_in_last_1"] = exc_sev.values
    g["spike_severe_in_last_3"] = exc_sev.rolling(_SEVERE_HISTORY_WINDOWS[1], min_periods=1).max().astype("float32").values
    g["spike_severe_in_last_6"] = exc_sev.rolling(_SEVERE_HISTORY_WINDOWS[2], min_periods=1).max().astype("float32").values

    # ── Spike history features ────────────────────────────────────────────────
    # LEAKAGE FIREWALL: spike_now uses current window only (not future).
    # exc = spike_now.shift(1) ensures every downstream feature looks only
    # at windows strictly before t.  This is the single gate that prevents
    # label leakage from spike_in_60m (which encodes windows t+1..t+12).
    spike_now = (cpu > threshold).astype("float32")
    exc       = spike_now.shift(1).fillna(0.0)   # leakage firewall

    g["spike_now"]      = spike_now
    g["spike_in_last_1"] = exc
    g["spike_in_last_3"] = exc.rolling(_SPIKE_HISTORY_WINDOWS[1], min_periods=1).max().astype("float32")
    g["spike_in_last_6"] = exc.rolling(_SPIKE_HISTORY_WINDOWS[2], min_periods=1).max().astype("float32")

    # cpu_spike_rate_24: fraction of the previous 24 windows that were spiking.
    # Reuses `exc` (shift(1) applied) — same leakage firewall; never uses current t.
    # Captures "chronic spiker" machines that neither spike_in_last_6 (binary)
    # nor consecutive_spikes (unbroken run) can represent.
    g["cpu_spike_rate_24"] = (
        exc.rolling(_MAX_TIME_SINCE_SPIKE, min_periods=1).mean().fillna(0.0).astype("float32")
    )

    # time_since_last_spike: how many windows ago was the last exceedance,
    # looking strictly before t (capped at _MAX_TIME_SINCE_SPIKE).
    # Pattern: forward-fill the position of the last True in spike_now (shifted
    # by 1 to exclude t), then compute distance from current position.
    # fillna(_MAX + 1) handles "no prior spike" → gets clipped to _MAX.
    pos      = pd.Series(
        np.arange(len(g), dtype="float32"),
        index = g.index,
    )
    last_pos = pos.where(spike_now.astype(bool)).shift(1).ffill()
    g["time_since_last_spike"] = (
        (pos - last_pos)
        .fillna(_MAX_TIME_SINCE_SPIKE + 1)
        .clip(upper=_MAX_TIME_SINCE_SPIKE)
        .astype("float32")
    )

    return g.reset_index()   # "bucket" becomes a column again


def _add_label(
    df:                 pd.DataFrame,
    threshold_p95:      float,
    threshold_p99:      float,
    horizon:            int,
    output_col:         str  = "severity_in_60m",
    binary:             bool = False,
    min_future_windows: int  = 1,
) -> pd.DataFrame:
    """Add a spike label column to a full (gap-filled) machine series.

    Must be called on the full series from _engineer_machine — *not* on
    already-filtered rows — so that the look-ahead operates over the correct
    time range, regardless of whether intermediate buckets had activity.

    Uses prefix-sum passes for O(n) look-ahead:

    When ``binary=False`` (default — 3-class severity):
        output_col[i] = 2   if COUNT(total_cpu[i+1..i+horizon] > threshold_p99) >= min_future_windows
                       = 1   elif COUNT(total_cpu[i+1..i+horizon] > threshold_p95) >= min_future_windows
                       = 0   otherwise
                       = NaN for the last `horizon` rows (incomplete look-ahead)

    When ``binary=True`` (binary spike flag):
        output_col[i] = 1   if COUNT(total_cpu[i+1..i+horizon] > threshold_p95) >= min_future_windows
                       = 0   otherwise
                       = NaN for the last `horizon` rows (incomplete look-ahead)

    Parameters
    ----------
    df                 : full reindexed series for one machine, sorted by bucket.
    threshold_p95      : CPU level above which a window counts as a spike.
    threshold_p99      : CPU level above which a window is severe.
                         Ignored when binary=True.
    horizon            : number of future windows to examine (e.g. 12 = 60 min).
    output_col         : name of the label column to create (default "severity_in_60m").
    binary             : if True, produce binary {0, 1} labels using p95 only.
    min_future_windows : K-of-N threshold — at least this many future windows must
                         exceed the threshold for a positive label.  Default=1
                         preserves the original any-exceedance behaviour.
                         Use K=2 or K=3 for ablation experiments that require
                         sustained spikes before labelling a window as positive.
    """
    df  = df.copy()
    n   = len(df)
    cpu = df["total_cpu"].values

    # Prefix-sum for O(n) range-sum queries.
    def _prefix_sum(exceeds: np.ndarray) -> np.ndarray:
        cs     = np.zeros(n + 1, dtype=np.float64)
        cs[1:] = np.cumsum(exceeds)
        return cs

    cs_mild = _prefix_sum((cpu > threshold_p95).astype(np.float64))

    labels = np.full(n, np.nan)
    valid  = n - horizon
    if valid > 0:
        # future_sums[i] = sum of exceeds[i+1 .. i+horizon]
        future_mild = cs_mild[1 + horizon : 1 + horizon + valid] - cs_mild[1 : 1 + valid]
        if binary:
            labels[:valid] = (future_mild >= min_future_windows).astype(np.float64)
        else:
            cs_severe     = _prefix_sum((cpu > threshold_p99).astype(np.float64))
            future_severe = cs_severe[1 + horizon : 1 + horizon + valid] - cs_severe[1 : 1 + valid]
            labels[:valid] = np.where(
                future_severe >= min_future_windows, 2,
                np.where(future_mild >= min_future_windows, 1, 0),
            ).astype(np.float64)

    df[output_col] = labels
    return df


def _add_cluster_features(df: pd.DataFrame, train_bucket_max: int) -> pd.DataFrame:
    """Add cluster-level cross-sectional features.

    ``cluster_cpu_p90`` is derived from training buckets only, matching the
    same training-only discipline used by ``_compute_thresholds``.  Val/test
    rows receive the training-period global p90 so that the anomalous
    test-period cluster behaviour (e.g. the days-21–25 usage spike in Google
    2011) cannot inflate cluster_cpu_p90 for test rows and skew evaluation.

    At production inference time the caller computes this value from the live
    120-min window — always the current cluster state — so there is no
    train/inference mismatch.

    ``machine_rank_in_cluster`` is contemporaneous (same bucket t, all active
    machines) and is computed on the full DataFrame; it describes the cluster
    state at t from currently observable data, not any future window.

    Parameters
    ----------
    df               : post-concat DataFrame of original (non-gap-filled) rows.
    train_bucket_max : highest bucket counted as training data (inclusive).
    """
    train = df[df["bucket"] <= train_bucket_max]

    # Per-bucket p90 from training machines only.
    p90 = (
        train.groupby("bucket")["total_cpu"]
        .quantile(0.9)
        .rename("cluster_cpu_p90")
        .reset_index()
    )
    p90["cluster_cpu_p90"] = p90["cluster_cpu_p90"].astype("float32")

    # Training-period global p90 fills val/test buckets whose indices never
    # appear in the training groupby (bucket indices are monotonically
    # increasing and the periods are disjoint).
    global_train_p90 = float(train["total_cpu"].quantile(0.9))

    df = df.merge(p90, on="bucket", how="left")
    df["cluster_cpu_p90"] = (
        df["cluster_cpu_p90"].fillna(global_train_p90).astype("float32")
    )

    # Percentile rank of this machine within its bucket (0.0 = lowest, 1.0 = highest).
    # method="average" assigns tied values the average rank — prevents
    # rank distortion when many idle machines share cpu=0.
    df["machine_rank_in_cluster"] = (
        df.groupby("bucket")["total_cpu"]
        .rank(pct=True, method="average")
        .astype("float32")
    )

    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time-of-day features.

    Derived from the bucket index only — no epoch lookup required.
    ``bucket % 288`` captures within-day periodicity regardless of the
    absolute start time of the trace.

    Harmonic (sin/cos) encoding is used rather than raw integers so that
    XGBoost can learn "peak period" patterns with fewer splits: hour 23 and
    hour 0 are angularly adjacent in sin/cos space but numerically far apart
    as integers.

    Phase 5: dow_sin / dow_cos removed.  High SHAP (0.356) combined with low
    XGBoost gain (0.011) indicates a global-bias corrector via interaction
    effects, not genuine weekly periodicity.  With only 7 days of data (~23
    samples per DOW label) the signal is confound-level, not structural.

    Parameters
    ----------
    df : post-concat DataFrame.
    """
    two_pi = float(2.0 * np.pi)

    bucket_in_day = (df["bucket"] % _BUCKETS_PER_DAY).astype("float32")

    df["hour_sin"] = np.sin(two_pi * bucket_in_day / _BUCKETS_PER_DAY).astype("float32")
    df["hour_cos"] = np.cos(two_pi * bucket_in_day / _BUCKETS_PER_DAY).astype("float32")

    return df


# ── Public entry point ────────────────────────────────────────────────────────


def engineer(
    input_path:         str | Path = "data/cluster_agg.parquet",
    output_path:        str | Path = "data/cluster_features.parquet",
    thresholds_path:    str | Path = "data/spike_thresholds.parquet",
    train_ratio:        float      = _TRAIN_RATIO,
    horizon:            int        = _HORIZON,
    min_future_windows: int        = 1,
) -> pd.DataFrame:
    """Engineer features and spike labels from the node-level aggregation.

    Parameters
    ----------
    input_path         : path to cluster_agg.parquet (output of spike_preprocessor).
    output_path        : destination for cluster_features.parquet.
    thresholds_path    : destination for spike_thresholds.parquet.
    train_ratio        : fraction of the time range used to compute spike thresholds.
                         Must match spike_classifier._TRAIN_RATIO (default 0.6) so that
                         thresholds are not contaminated by val/test-period observations.
    horizon            : look-ahead windows for the spike label (default 12 = 60 min).
    min_future_windows : K-of-N threshold for severity_in_60m — at least this many
                         future windows must exceed the threshold to fire the label.
                         Default=1 preserves current any-exceedance behaviour.
                         Only applied to the 60m multiclass label; binary short-horizon
                         labels (spike_in_15m/30m/45m) always use K=1.

    Returns
    -------
    Engineered DataFrame (also written to output_path as Parquet).

    Raises
    ------
    FileNotFoundError
        If input_path does not exist.
    """
    input_path      = Path(input_path)
    output_path     = Path(output_path)
    thresholds_path = Path(thresholds_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    log.info("═" * 62)
    log.info("  SPIKE FEATURE ENGINEER — lag / EWMA / spike history / cluster")
    log.info("  Input      : %s", input_path)
    log.info("  Output     : %s", output_path)
    log.info("  Thresholds : %s", thresholds_path)
    log.info("  Horizon    : %d windows (%d min)", horizon, horizon * 5)
    log.info("  Train ratio: %.0f%%  (thresholds computed from this window only)", train_ratio * 100)
    log.info("  Label K-of-N: min_future_windows=%d", min_future_windows)
    log.info("═" * 62)

    df = pd.read_parquet(input_path)

    if df.empty:
        log.warning("Input parquet is empty — writing empty outputs.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        thresholds_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=_FEATURE_COLS).to_parquet(
            output_path, engine="pyarrow", compression="zstd", index=False
        )
        pd.DataFrame(columns=_THRESHOLD_COLS).to_parquet(
            thresholds_path, engine="pyarrow", compression="zstd", index=False
        )
        return pd.DataFrame(columns=_FEATURE_COLS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds_path.parent.mkdir(parents=True, exist_ok=True)

    df = df.sort_values(["machine_id", "bucket"]).reset_index(drop=True)

    # Compute p95 and p99 thresholds from training split only (Problem 2)
    b_min = int(df["bucket"].min())
    b_max = int(df["bucket"].max())
    train_max = b_min + int((b_max - b_min) * train_ratio)
    log.info("  Train max bucket : %d", train_max)

    thresh_df = _compute_thresholds(df, train_max)
    thresh_df["threshold_p95"] = thresh_df["threshold_p95"].astype("float32")
    thresh_df["threshold_p99"] = thresh_df["threshold_p99"].astype("float32")
    thresh_df.to_parquet(
        thresholds_path, engine="pyarrow", compression="zstd", index=False
    )
    log.info("  Thresholds saved : %d machines", len(thresh_df))

    thresholds     = thresh_df.set_index("machine_id")["threshold_p95"]
    thresholds_p99 = thresh_df.set_index("machine_id")["threshold_p99"]

    # ── Per-machine feature engineering + labeling ────────────────────────────
    n_machines = int(df["machine_id"].nunique())
    log.info("  Engineering features for %d machines …", n_machines)
    t_start = time.perf_counter()

    parts: list[pd.DataFrame] = []
    for i, (machine_id, group) in enumerate(df.groupby("machine_id")):
        threshold_p95 = float(thresholds[machine_id])
        threshold_p99 = float(thresholds_p99[machine_id])
        full_series = _engineer_machine(group.copy(), int(machine_id), threshold_p95, threshold_p99=threshold_p99)
        # 3-class severity label for the primary 60-min horizon
        labeled = _add_label(full_series, threshold_p95, threshold_p99, horizon=12,
                             output_col="severity_in_60m", binary=False,
                             min_future_windows=min_future_windows)
        # Binary spike labels for shorter horizons (Phase 4)
        labeled = _add_label(labeled, threshold_p95, threshold_p99, horizon=3,
                             output_col="spike_in_15m",  binary=True)
        labeled = _add_label(labeled, threshold_p95, threshold_p99, horizon=6,
                             output_col="spike_in_30m",  binary=True)
        labeled = _add_label(labeled, threshold_p95, threshold_p99, horizon=9,
                             output_col="spike_in_45m",  binary=True)
        original      = (
            labeled[labeled["_original"]]
            .drop(columns=["_original"])
            .reset_index(drop=True)
        )
        parts.append(original)

        if (i + 1) % 1_000 == 0:
            elapsed = (time.perf_counter() - t_start) / 60
            log.info(
                "  %5d / %d machines  |  %.1f min",
                i + 1, n_machines, elapsed,
            )

    final = pd.concat(parts, ignore_index=True)

    # ── Type casting (per-machine features) ───────────────────────────────────
    final["machine_id"] = final["machine_id"].astype("int64")
    final["bucket"]     = final["bucket"].astype("int64")
    final["time_us"]    = final["time_us"].astype("int64")
    final["n_tasks"]    = final["n_tasks"].astype("int32")

    float32_cols = [
        # raw signals
        "total_cpu", "peak_cpu", "total_mem", "peak_mem", "disk_io",
        # lags (Phase 2: cpu_lag_2/3/6 removed)
        "cpu_lag_1", "cpu_lag_12", "cpu_lag_24",
        # trend
        "cpu_ewma_6", "cpu_ewma_24",
        # derivatives (delta_2, rolling_std already cast inline)
        "cpu_delta_1",
        # machine-relative (delta, peak_vs_p95 already cast inline)
        "task_dominance",
        # spike history (all already cast inline, including cpu_spike_rate_24)
        "spike_now", "spike_in_last_1", "spike_in_last_3", "spike_in_last_6",
        "time_since_last_spike",
    ]
    for col in float32_cols:
        final[col] = final[col].astype("float32")

    # ── cpu_vs_p95: machine-relative normalised CPU ───────────────────────────
    # Computed here (not inside _engineer_machine) so the threshold mapping
    # stays a pipeline-level concern.  Fallback uses global_thresh (median of
    # per-machine thresholds) — matches the inference-time fallback in
    # predict_spike.py so unseen machines get consistent cpu_vs_p95 values.
    global_thresh = float(thresholds.median())
    thresh_mapped = final["machine_id"].map(thresholds.to_dict()).fillna(global_thresh)
    final["cpu_vs_p95"] = np.where(
        thresh_mapped > 0,
        final["total_cpu"] / thresh_mapped,
        0.0,
    ).astype("float32")

    # cpu_vs_p95_delta already computed per-machine in _engineer_machine;
    # enforce float32 here as the final type pass.
    final["cpu_vs_p95_delta"] = final["cpu_vs_p95_delta"].astype("float32")
    final["cpu_delta_2"]      = final["cpu_delta_2"].astype("float32")
    final["cpu_rolling_std_6"] = final["cpu_rolling_std_6"].astype("float32")

    # ── Cluster-level features (cross-sectional, same bucket t) ──────────────
    log.info("  Adding cluster-level features …")
    final = _add_cluster_features(final, train_max)

    # ── Time features (from bucket index, no epoch needed) ───────────────────
    log.info("  Adding time-of-day / day-of-week features …")
    final = _add_time_features(final)

    # ── Final column selection and sort ───────────────────────────────────────
    # severity_in_60m stays float64 to preserve NaN
    final = (
        final[_FEATURE_COLS]
        .sort_values(["machine_id", "bucket"])
        .reset_index(drop=True)
    )

    final.to_parquet(output_path, engine="pyarrow", compression="zstd", index=False)

    elapsed        = (time.perf_counter() - t_start) / 60
    n_labeled      = int(final["severity_in_60m"].notna().sum())
    n_no_spike     = int((final["severity_in_60m"] == 0).sum())
    n_moderate     = int((final["severity_in_60m"] == 1).sum())
    n_severe       = int((final["severity_in_60m"] == 2).sum())
    moderate_pct   = 100.0 * n_moderate / max(n_labeled, 1)
    severe_pct     = 100.0 * n_severe   / max(n_labeled, 1)

    n_spike_15m    = int((final["spike_in_15m"] == 1).sum())
    n_spike_30m    = int((final["spike_in_30m"] == 1).sum())
    n_spike_45m    = int((final["spike_in_45m"] == 1).sum())
    n_labeled_15m  = int(final["spike_in_15m"].notna().sum())

    log.info("═" * 62)
    log.info("  Done")
    log.info("  Output rows  : %s", f"{len(final):,}")
    log.info(
        "  Labeled rows : %s  (last %d per machine = NaN)",
        f"{n_labeled:,}", horizon,
    )
    log.info(
        "  Severity dist : no_spike=%s  moderate=%.1f%%  severe=%.1f%%",
        f"{n_no_spike:,}", moderate_pct, severe_pct,
    )
    log.info(
        "  Spike rates   : 15m=%.1f%%  30m=%.1f%%  45m=%.1f%%",
        100.0 * n_spike_15m / max(n_labeled_15m, 1),
        100.0 * n_spike_30m / max(n_labeled_15m, 1),
        100.0 * n_spike_45m / max(n_labeled_15m, 1),
    )
    log.info("  Output size  : %.1f MB", output_path.stat().st_size / 1_048_576)
    log.info("  Elapsed      : %.1f min", elapsed)
    log.info("═" * 62)

    return final


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description     = "Engineer spike-prediction features from cluster_agg.parquet.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = (
            "examples:\n"
            "  python spike_feature_engineer.py\n"
            "  python spike_feature_engineer.py --input data/cluster_agg.parquet\n"
            "  python spike_feature_engineer.py --input  data/cluster_agg.parquet"
            " --output data/cluster_features.parquet\n"
        ),
    )
    p.add_argument(
        "--input",
        default = "data/cluster_agg.parquet",
        metavar = "PATH",
        help    = "Path to cluster_agg.parquet  (default: data/cluster_agg.parquet)",
    )
    p.add_argument(
        "--output",
        default = "data/cluster_features.parquet",
        metavar = "PATH",
        help    = "Output features Parquet path  (default: data/cluster_features.parquet)",
    )
    p.add_argument(
        "--thresholds",
        default = "data/spike_thresholds.parquet",
        metavar = "PATH",
        help    = "Output thresholds Parquet path  (default: data/spike_thresholds.parquet)",
    )
    p.add_argument(
        "--train-ratio",
        type    = float,
        default = _TRAIN_RATIO,
        metavar = "R",
        help    = f"Fraction of buckets used for threshold computation  (default: {_TRAIN_RATIO})",
    )
    p.add_argument(
        "--horizon",
        type    = int,
        default = _HORIZON,
        metavar = "N",
        help    = f"Look-ahead windows for spike label  (default: {_HORIZON})",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    engineer(
        input_path      = args.input,
        output_path     = args.output,
        thresholds_path = args.thresholds,
        train_ratio     = args.train_ratio,
        horizon         = args.horizon,
    )
