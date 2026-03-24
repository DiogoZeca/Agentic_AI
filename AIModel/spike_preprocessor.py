"""Aggregate Google Cluster Traces 2011 task_usage data to node level.

Reads cluster_cpu_data.csv (278 M task-level rows, ~23 GB) in chunks and
collapses all per-task measurements into one row per physical node per
5-minute window. Writes the result as a compressed Parquet file.

Two-pass aggregation is used to handle groups that straddle chunk
boundaries:
  Pass 1  _process_chunk   aggregates each chunk independently.
  Pass 2  _merge_partials  re-aggregates all partial results to merge
                           any (machine_id, bucket) split across chunks.

cpu_rate is a *mean rate* over each task's measurement window, so a
2-second boundary task at cpu_rate=0.5 used only 2/300 of what a
full-window task at the same rate used. total_cpu is duration-weighted
to correct this:

    total_cpu = sum(cpu_rate × window_duration) / 300_000_000

Input  : data/cluster_cpu_data.csv
Output : data/cluster_agg.parquet  (~300 MB, ~24 M rows)

Usage:
    python spike_preprocessor.py
    python spike_preprocessor.py --input  data/cluster_cpu_data.csv \\
                                  --output data/cluster_agg.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

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

_BUCKET_US: int   = 300_000_000   # 5 minutes in microseconds
_CPU_MAX:   float = 64.0          # physically impossible per-task cpu_rate
                                  # (actual max in dataset ≈ 0.55; this is a
                                  #  safety net only)
_CHUNKSIZE: int   = 5_000_000     # rows per chunk — tuned for ≥ 32 GB RAM

_INPUT_COLS: list[str] = [
    "start_time",
    "end_time",
    "machine_id",
    "cpu_rate",
    "max_cpu_rate",
    "canonical_mem_usage",
    "max_mem_usage",
    "mean_disk_io_time",
    "sample_portion",
]

# Explicit dtype map passed to read_csv on every chunk.
# Without this, pandas infers types independently per chunk — a column that
# is all-integer in chunk 1 may be float in chunk 50, silently breaking
# numeric operations downstream.  machine_id is kept as float64 here because
# it arrives from the CSV as float (download script) and may contain NaN;
# it is cast to int64 in _merge_partials after null rows are dropped.
_INPUT_DTYPES: dict[str, str] = {
    "start_time":           "float64",
    "end_time":             "float64",
    "machine_id":           "float64",
    "cpu_rate":             "float64",
    "max_cpu_rate":         "float64",
    "canonical_mem_usage":  "float64",
    "max_mem_usage":        "float64",
    "mean_disk_io_time":    "float64",
    "sample_portion":       "float64",
}

# Columns produced by _process_chunk before the second aggregation pass.
# Includes the intermediate weighted_cpu_sum that is not in the final output.
_PARTIAL_COLS: list[str] = [
    "machine_id",
    "bucket",
    "weighted_cpu_sum",
    "peak_cpu",
    "total_mem",
    "peak_mem",
    "disk_io",
    "n_tasks",
]

_OUTPUT_COLS: list[str] = [
    "machine_id",
    "bucket",
    "time_us",
    "total_cpu",
    "peak_cpu",
    "total_mem",
    "peak_mem",
    "disk_io",
    "n_tasks",
]

# ── Private helpers ───────────────────────────────────────────────────────────


def _validate_schema(path: Path) -> None:
    """Check that the CSV header contains all expected columns.

    Reads only the header row (nrows=0) — no data is loaded.

    Raises
    ------
    ValueError
        If any column from _INPUT_COLS is absent.
    """
    header  = pd.read_csv(path, nrows=0)
    missing = set(_INPUT_COLS) - set(header.columns)
    if missing:
        raise ValueError(
            f"Input file {path.name!r} is missing expected columns: "
            f"{sorted(missing)}"
        )


def _process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Filter one CSV chunk and produce a partial (pass-1) aggregation.

    Groups that straddle this chunk's boundary with the next chunk will
    produce partial rows here. _merge_partials (pass 2) merges them.

    Filters applied (in order):
      1. Null machine_id  — rows without a machine cannot be attributed to a
                            node; groupby would silently create a fake NaN machine.
      2. Invalid duration — end_time <= start_time indicates clock drift or a
                            corrupt row; weighted_cpu would be zero or negative.
      3. Impossible rate  — cpu_rate > 64 is a safety net for corrupted sensors.

    Parameters
    ----------
    chunk : raw CSV chunk containing _INPUT_COLS columns.

    Returns
    -------
    Partial aggregation DataFrame with _PARTIAL_COLS columns.
    Returns an empty DataFrame (with correct columns) when no rows
    survive filtering.
    """
    # Filter 1: null machine_id
    n_null_id = int(chunk["machine_id"].isna().sum())
    if n_null_id:
        log.warning("  Dropped %d rows with null machine_id in chunk", n_null_id)
        chunk = chunk[chunk["machine_id"].notna()]

    # Filter 2: invalid duration (clock drift / corrupt entries)
    n_bad_dur = int((chunk["end_time"] <= chunk["start_time"]).sum())
    if n_bad_dur:
        log.warning("  Dropped %d rows with end_time <= start_time in chunk", n_bad_dur)
        chunk = chunk[chunk["end_time"] > chunk["start_time"]]

    # Filter 3: physically impossible cpu_rate
    chunk = chunk[chunk["cpu_rate"] <= _CPU_MAX].copy()

    if chunk.empty:
        return pd.DataFrame(columns=_PARTIAL_COLS)

    # Duration-weighted CPU contribution.
    # cpu_rate is a mean rate over the measurement window. A 2-second
    # boundary task with cpu_rate=0.5 uses the CPU for only 2 s, not 300 s.
    # Multiplying by window duration before summing prevents short-window
    # task boundary artefacts from inflating total_cpu.
    chunk["weighted_cpu"] = (
        chunk["cpu_rate"] * (chunk["end_time"] - chunk["start_time"])
    )

    chunk["bucket"] = (chunk["start_time"] // _BUCKET_US).astype("int64")

    # Only additive / associative operations so _merge_partials can safely
    # re-aggregate across chunk boundaries.
    partial = (
        chunk
        .groupby(["machine_id", "bucket"], as_index=False)
        .agg(
            weighted_cpu_sum = ("weighted_cpu",          "sum"),
            peak_cpu         = ("max_cpu_rate",          "max"),
            total_mem        = ("canonical_mem_usage",   "sum"),
            peak_mem         = ("max_mem_usage",         "max"),
            disk_io          = ("mean_disk_io_time",     "max"),
            n_tasks          = ("cpu_rate",              "count"),
        )
    )
    return partial


def _merge_partials(partials: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge boundary fragments and produce the final node-level DataFrame.

    Concatenates all partial results from _process_chunk, then
    re-aggregates to merge any (machine_id, bucket) pair split across
    chunk boundaries.

    Parameters
    ----------
    partials : non-empty list of DataFrames returned by _process_chunk.

    Returns
    -------
    Final DataFrame with _OUTPUT_COLS columns, sorted by
    (machine_id, bucket).
    """
    combined = pd.concat(partials, ignore_index=True)

    final = (
        combined
        .groupby(["machine_id", "bucket"], as_index=False)
        .agg(
            weighted_cpu_sum = ("weighted_cpu_sum", "sum"),
            peak_cpu         = ("peak_cpu",         "max"),
            total_mem        = ("total_mem",         "sum"),
            peak_mem         = ("peak_mem",          "max"),
            disk_io          = ("disk_io",           "max"),
            n_tasks          = ("n_tasks",           "sum"),
        )
    )

    # Derive total_cpu: normalise accumulated CPU-µs to a per-window average.
    final["total_cpu"] = (final["weighted_cpu_sum"] / _BUCKET_US).astype("float32")
    final = final.drop(columns=["weighted_cpu_sum"])

    # Human-readable timestamp (µs since trace epoch).
    final["time_us"] = (final["bucket"] * _BUCKET_US).astype("int64")

    # Type casting.
    # machine_id is float64 in the CSV (download script reads everything as
    # float). Max observed value is 6,301,942,525 — exceeds int32 max, must
    # use int64.
    final["machine_id"] = final["machine_id"].astype("int64")
    final["bucket"]     = final["bucket"].astype("int64")
    final["n_tasks"]    = final["n_tasks"].astype("int32")
    for col in ("peak_cpu", "total_mem", "peak_mem", "disk_io"):
        final[col] = final[col].astype("float32")

    final = (
        final
        .sort_values(["machine_id", "bucket"])
        .reset_index(drop=True)
    )
    return final[_OUTPUT_COLS]


# ── Public entry point ────────────────────────────────────────────────────────


def preprocess(
    input_path:  str | Path,
    output_path: str | Path,
    chunksize:   int = _CHUNKSIZE,
) -> pd.DataFrame:
    """Read cluster_cpu_data.csv and aggregate to node-level 5-min windows.

    Parameters
    ----------
    input_path  : path to cluster_cpu_data.csv.
    output_path : destination for the output Parquet file.
    chunksize   : rows per CSV chunk (default 5_000_000).

    Returns
    -------
    Aggregated DataFrame (also written to output_path as Parquet).

    Raises
    ------
    FileNotFoundError
        If input_path does not exist.
    ValueError
        If input_path is missing any expected column.
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    _validate_schema(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("═" * 62)
    log.info("  SPIKE PREPROCESSOR — node-level 5-min aggregation")
    log.info("  Input    : %s", input_path)
    log.info("  Output   : %s", output_path)
    log.info("  Chunk    : %s rows", f"{chunksize:,}")
    log.info("═" * 62)

    partials:         list[pd.DataFrame] = []
    total_input_rows: int                = 0
    t_start = time.perf_counter()

    # Context manager ensures the underlying C parser buffers are freed after
    # each chunk — without it, TextFileReader accumulates memory across all
    # chunks and does not release until the object is garbage-collected
    # (pandas issue #21516).
    with pd.read_csv(
        input_path,
        chunksize  = chunksize,
        dtype      = _INPUT_DTYPES,
        usecols    = _INPUT_COLS,
    ) as reader:
        for chunk_idx, chunk in enumerate(reader):
            partial           = _process_chunk(chunk)
            total_input_rows += len(chunk)

            if not partial.empty:
                partials.append(partial)

            if (chunk_idx + 1) % 10 == 0:
                elapsed = (time.perf_counter() - t_start) / 60
                log.info(
                    "  chunk %3d  |  %s rows read  |  %.1f min elapsed",
                    chunk_idx + 1,
                    f"{total_input_rows:,}",
                    elapsed,
                )

    if not partials:
        log.warning("No rows survived filtering — output is empty.")
        empty = pd.DataFrame(columns=_OUTPUT_COLS)
        empty.to_parquet(output_path, engine="pyarrow", compression="zstd", index=False)
        return empty

    log.info("Merging %d partial results …", len(partials))
    final = _merge_partials(partials)

    final.to_parquet(output_path, engine="pyarrow", compression="zstd", index=False)

    elapsed    = (time.perf_counter() - t_start) / 60
    size_mb    = output_path.stat().st_size / 1_048_576
    n_machines = final["machine_id"].nunique()
    n_buckets  = final["bucket"].nunique()

    log.info("═" * 62)
    log.info("  Done")
    log.info(
        "  Output rows  : %s  (%d machines × %d buckets)",
        f"{len(final):,}", n_machines, n_buckets,
    )
    log.info("  Output size  : %.1f MB", size_mb)
    log.info("  Elapsed      : %.1f min", elapsed)
    log.info("═" * 62)

    return final


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description = "Aggregate cluster_cpu_data.csv to node-level 5-min windows.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = (
            "examples:\n"
            "  python spike_preprocessor.py\n"
            "  python spike_preprocessor.py --input data/cluster_cpu_data.csv\n"
            "  python spike_preprocessor.py --input  data/cluster_cpu_data.csv"
            " --output data/cluster_agg.parquet\n"
        ),
    )
    p.add_argument(
        "--input",
        default  = "data/cluster_cpu_data.csv",
        metavar  = "PATH",
        help     = "Path to cluster_cpu_data.csv  (default: data/cluster_cpu_data.csv)",
    )
    p.add_argument(
        "--output",
        default  = "data/cluster_agg.parquet",
        metavar  = "PATH",
        help     = "Output Parquet path  (default: data/cluster_agg.parquet)",
    )
    p.add_argument(
        "--chunksize",
        type     = int,
        default  = _CHUNKSIZE,
        metavar  = "N",
        help     = f"Rows per CSV chunk  (default: {_CHUNKSIZE:,})",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    preprocess(
        input_path  = args.input,
        output_path = args.output,
        chunksize   = args.chunksize,
    )
