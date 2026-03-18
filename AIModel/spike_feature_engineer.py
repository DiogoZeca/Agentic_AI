"""Engineer features and spike labels from node-level 5-min aggregations.

Reads cluster_agg.parquet (output of spike_preprocessor.py) and produces:

  cluster_features.parquet  — one row per (machine_id, bucket) with all
                               features and the spike_in_60m label.
  spike_thresholds.parquet  — per-machine p95 CPU threshold, computed from
                               training windows only to prevent label leakage.

Feature set (16 columns)
------------------------
  total_cpu, peak_cpu, total_mem, peak_mem, disk_io, n_tasks
    raw signals at time t.

  cpu_lag_{1,2,3,6,12,24}
    total_cpu shifted by N 5-min windows (history shape).

  cpu_ewma_6, cpu_ewma_24
    exponentially weighted moving averages with half-life 6 and 24 windows
    (short-term trend vs long-term baseline).

  cpu_delta_1
    first difference: total_cpu[t] - total_cpu[t-1] (rate of change).

  task_dominance
    peak_cpu / total_cpu — fraction of node CPU consumed by the single
    busiest task. High dominance indicates monopolisation risk.

Spike label
-----------
  spike_in_60m = 1  if any window in [t+1, t+12] exceeds the machine's
                     p95 CPU threshold (computed on training data only).
               = 0  if all look-ahead windows are below threshold.
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
Output : data/cluster_features.parquet  (~400-600 MB)
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

_BUCKET_US:      int   = 300_000_000   # 5 minutes in microseconds
_HORIZON:        int   = 12            # 60-min look-ahead (12 × 5-min windows)
_LAGS:           tuple = (1, 2, 3, 6, 12, 24)
_EWMA_HALFLIVES: tuple = (6, 24)
_TRAIN_RATIO:    float = 0.6   # 60 % of buckets — must match spike_classifier._TRAIN_RATIO
_SPIKE_QUANTILE: float = 0.95

_FEATURE_COLS: list[str] = [
    "machine_id", "bucket", "time_us",
    "total_cpu", "peak_cpu", "total_mem", "peak_mem", "disk_io", "n_tasks",
    "cpu_lag_1", "cpu_lag_2", "cpu_lag_3", "cpu_lag_6", "cpu_lag_12", "cpu_lag_24",
    "cpu_ewma_6", "cpu_ewma_24",
    "cpu_delta_1",
    "task_dominance",
    "cpu_vs_p95",    # total_cpu / machine_p95 threshold — gives XGBoost the machine-relative
                     # context that raw absolute lags lack.  cpu_vs_p95 > 1 means the node is
                     # already above its historical spike level right now.
    "spike_in_60m",
]

_THRESHOLD_COLS: list[str] = ["machine_id", "threshold_cpu"]

# ── Private helpers ───────────────────────────────────────────────────────────


def _compute_thresholds(df: pd.DataFrame, train_bucket_max: int) -> pd.Series:
    """Compute per-machine p95 total_cpu from training windows only.

    Using only training data prevents the spike threshold from being
    influenced by future high-load events (label leakage — Problem 2).

    Machines with no rows in the training portion receive the global
    training-data p95 as a fallback (Problem 3).

    Parameters
    ----------
    df               : full aggregated DataFrame.
    train_bucket_max : highest bucket index counted as training data.

    Returns
    -------
    pd.Series indexed by machine_id, values are float p95 thresholds.
    """
    train = df[df["bucket"] <= train_bucket_max]

    per_machine: pd.Series = (
        train
        .groupby("machine_id")["total_cpu"]
        .quantile(_SPIKE_QUANTILE)
    )

    global_p95 = float(train["total_cpu"].quantile(_SPIKE_QUANTILE))

    all_ids    = set(df["machine_id"].unique())
    missing    = all_ids - set(per_machine.index)
    if missing:
        log.warning(
            "  %d machine(s) absent from training data — "
            "assigning global p95 (%.4f) as threshold fallback",
            len(missing),
            global_p95,
        )
        fallback = pd.Series(
            global_p95,
            index  = pd.Index(sorted(missing), name="machine_id"),
            dtype  = "float64",
        )
        per_machine = pd.concat([per_machine, fallback]).sort_index()

    return per_machine


def _engineer_machine(group: pd.DataFrame, machine_id: int) -> pd.DataFrame:
    """Reindex to fill bucket gaps, then compute all lag/EWMA/delta features.

    Returns the *full* reindexed series (original rows plus gap-filled rows).
    A boolean column ``_original`` marks which rows were in the input; callers
    must filter on it after adding the spike label.

    Gap-filled rows use 0 for all CPU/memory/disk signals — an idle machine
    exerts no load, so 0 is the physically correct value, not a sentinel.

    Parameters
    ----------
    group      : rows for a single machine_id from the aggregated parquet.
    machine_id : the machine identifier (int).
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

    cpu = g["total_cpu"]

    # Lag features — correct because gaps are already filled with 0
    for lag in _LAGS:
        g[f"cpu_lag_{lag}"] = cpu.shift(lag).fillna(0.0)

    # EWMA features — idle zeros correctly drag down the moving average
    for halflife in _EWMA_HALFLIVES:
        g[f"cpu_ewma_{halflife}"] = (
            cpu.ewm(halflife=halflife, adjust=False).mean()
        )

    # Rate of change
    g["cpu_delta_1"] = cpu - cpu.shift(1).fillna(0.0)

    # Task dominance — guard against division by zero (Problem 4)
    g["task_dominance"] = np.where(
        g["total_cpu"] > 0,
        g["peak_cpu"] / g["total_cpu"],
        0.0,
    )

    return g.reset_index()   # "bucket" becomes a column again


def _add_label(
    df:        pd.DataFrame,
    threshold: float,
    horizon:   int,
) -> pd.DataFrame:
    """Add spike_in_60m to a full (gap-filled) machine series.

    Must be called on the full series from _engineer_machine — *not* on
    already-filtered rows — so that the look-ahead operates over the correct
    time range, regardless of whether intermediate buckets had activity.

    Uses a prefix-sum trick for O(n) look-ahead (Problem 5):

        spike_in_60m[i] = 1  if any(total_cpu[i+1 .. i+horizon] > threshold)
                        = NaN for the last `horizon` rows (incomplete look-ahead).

    Parameters
    ----------
    df        : full reindexed series for one machine, sorted by bucket.
    threshold : CPU level above which a window counts as a spike.
    horizon   : number of future windows to examine (default 12 = 60 min).
    """
    df = df.copy()
    n  = len(df)

    exceeds = (df["total_cpu"].values > threshold).astype(np.float64)

    # cs[i+1] = number of exceedances in exceeds[0..i]
    cs      = np.zeros(n + 1, dtype=np.float64)
    cs[1:]  = np.cumsum(exceeds)

    labels     = np.full(n, np.nan)
    valid      = n - horizon
    if valid > 0:
        # future_sums[i] = sum of exceeds[i+1 .. i+horizon]
        #                 = cs[i+1+horizon] - cs[i+1]
        future_sums = cs[1 + horizon : 1 + horizon + valid] - cs[1 : 1 + valid]
        labels[:valid] = (future_sums > 0).astype(np.float32)

    df["spike_in_60m"] = labels
    return df


# ── Public entry point ────────────────────────────────────────────────────────


def engineer(
    input_path:      str | Path = "data/cluster_agg.parquet",
    output_path:     str | Path = "data/cluster_features.parquet",
    thresholds_path: str | Path = "data/spike_thresholds.parquet",
    train_ratio:     float      = _TRAIN_RATIO,
    horizon:         int        = _HORIZON,
) -> pd.DataFrame:
    """Engineer features and spike labels from the node-level aggregation.

    Parameters
    ----------
    input_path      : path to cluster_agg.parquet (output of spike_preprocessor).
    output_path     : destination for cluster_features.parquet.
    thresholds_path : destination for spike_thresholds.parquet.
    train_ratio     : fraction of the time range used to compute spike thresholds.
                      Must match spike_classifier._TRAIN_RATIO (default 0.6) so that
                      thresholds are not contaminated by val/test-period observations.
    horizon         : look-ahead windows for the spike label (default 12 = 60 min).

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
    log.info("  SPIKE FEATURE ENGINEER — lag / EWMA / spike labels")
    log.info("  Input      : %s", input_path)
    log.info("  Output     : %s", output_path)
    log.info("  Thresholds : %s", thresholds_path)
    log.info("  Horizon    : %d windows (%d min)", horizon, horizon * 5)
    log.info("  Train ratio: %.0f%%  (thresholds computed from this window only)", train_ratio * 100)
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

    # Compute thresholds from training split only (Problem 2)
    train_max = int(df["bucket"].max() * train_ratio)
    log.info("  Train max bucket : %d", train_max)

    thresholds = _compute_thresholds(df, train_max)

    # Write thresholds
    thresh_df = thresholds.reset_index()
    thresh_df.columns = pd.Index(_THRESHOLD_COLS)
    thresh_df["threshold_cpu"] = thresh_df["threshold_cpu"].astype("float32")
    thresh_df.to_parquet(
        thresholds_path, engine="pyarrow", compression="zstd", index=False
    )
    log.info("  Thresholds saved : %d machines", len(thresh_df))

    # Feature engineering + labeling — one machine at a time (Problem 6)
    n_machines = int(df["machine_id"].nunique())
    log.info("  Engineering features for %d machines …", n_machines)
    t_start = time.perf_counter()

    parts: list[pd.DataFrame] = []
    for i, (machine_id, group) in enumerate(df.groupby("machine_id")):
        threshold   = float(thresholds[machine_id])
        full_series = _engineer_machine(group.copy(), int(machine_id))
        labeled     = _add_label(full_series, threshold, horizon)
        original    = (
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

    # Type casting
    final["machine_id"] = final["machine_id"].astype("int64")
    final["bucket"]     = final["bucket"].astype("int64")
    final["time_us"]    = final["time_us"].astype("int64")
    final["n_tasks"]    = final["n_tasks"].astype("int32")
    float32_cols = [
        "total_cpu", "peak_cpu", "total_mem", "peak_mem", "disk_io",
        "cpu_lag_1", "cpu_lag_2", "cpu_lag_3", "cpu_lag_6", "cpu_lag_12", "cpu_lag_24",
        "cpu_ewma_6", "cpu_ewma_24",
        "cpu_delta_1",
        "task_dominance",
    ]
    for col in float32_cols:
        final[col] = final[col].astype("float32")

    # cpu_vs_p95: machine-relative feature — total_cpu expressed as a multiple of
    # the machine's own p95 spike threshold.  Computed here (not inside
    # _engineer_machine) so the function signature stays clean; thresholds are a
    # pipeline-level concern, not a rolling-window concern.
    # fillna(1.0) is a safe fallback: division by 1 leaves cpu_vs_p95 = total_cpu,
    # which is neutral and avoids NaN propagation for any edge-case machine.
    thresh_mapped = final["machine_id"].map(thresholds).fillna(1.0)
    final["cpu_vs_p95"] = np.where(
        thresh_mapped > 0,
        final["total_cpu"] / thresh_mapped,
        0.0,
    ).astype("float32")

    # spike_in_60m stays float64 to preserve NaN

    final = (
        final[_FEATURE_COLS]
        .sort_values(["machine_id", "bucket"])
        .reset_index(drop=True)
    )

    final.to_parquet(output_path, engine="pyarrow", compression="zstd", index=False)

    elapsed   = (time.perf_counter() - t_start) / 60
    n_labeled = int(final["spike_in_60m"].notna().sum())
    n_spikes  = int((final["spike_in_60m"] == 1).sum())
    spike_pct = 100.0 * n_spikes / max(n_labeled, 1)

    log.info("═" * 62)
    log.info("  Done")
    log.info("  Output rows  : %s", f"{len(final):,}")
    log.info(
        "  Labeled rows : %s  (last %d per machine = NaN)",
        f"{n_labeled:,}", horizon,
    )
    log.info("  Spike rate   : %.1f%%", spike_pct)
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
