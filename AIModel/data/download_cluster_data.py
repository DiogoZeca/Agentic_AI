"""Download Google Cluster Traces 2011 — task_usage table.

Downloads the first N hours of the trace from Google Cloud Storage,
keeping only the 9 columns relevant for CPU spike prediction.

Source: https://storage.googleapis.com/clusterdata-2011-2/task_usage/
Format: 500 gzip-compressed CSV parts, no header, 20 columns each.
Docs  : https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md

Columns kept (9 of 20):
  start_time          µs since trace epoch — window start
  end_time            µs since trace epoch — window end
  machine_id          physical machine identifier
  cpu_rate            avg CPU rate in window (fraction of 1 core, 0–1+)
  max_cpu_rate        peak CPU rate in window  ← better spike signal than avg
  canonical_mem_usage avg memory (fraction of machine memory)
  max_mem_usage       peak memory in window
  mean_disk_io_time   avg disk I/O time (proxy for I/O pressure)
  sample_portion      fraction of window that was sampled (data quality flag)

Usage:
    python download_cluster_data.py                        # 160 h, default output
    python download_cluster_data.py --hours 80             # shorter window
    python download_cluster_data.py --resume               # continue interrupted run
    python download_cluster_data.py --hours 160 --out my.csv
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

# ── Column layout (0-indexed, no header in raw files) ─────────────────────────

# Map: raw column index → output name (only the 9 we keep)
_COLUMN_MAP: dict[int, str] = {
    0:  "start_time",
    1:  "end_time",
    4:  "machine_id",
    5:  "cpu_rate",
    13: "max_cpu_rate",
    6:  "canonical_mem_usage",
    10: "max_mem_usage",
    11: "mean_disk_io_time",
    17: "sample_portion",
}

# Sorted indices for pd.read_csv(usecols=...)
_USECOLS: list[int] = sorted(_COLUMN_MAP)

# Column names in the order they appear after sorting by index
_COL_NAMES_BY_IDX: list[str] = [_COLUMN_MAP[i] for i in _USECOLS]

# Final output column order (logical grouping)
_OUTPUT_ORDER: list[str] = [
    "start_time", "end_time", "machine_id",
    "cpu_rate", "max_cpu_rate",
    "canonical_mem_usage", "max_mem_usage",
    "mean_disk_io_time",
    "sample_portion",
]

# ── Constants ─────────────────────────────────────────────────────────────────

_BASE_URL       = "https://storage.googleapis.com/clusterdata-2011-2/task_usage/"
_TOTAL_PARTS    = 500
_MICROS_PER_H   = 3_600_000_000        # microseconds per hour
_DEFAULT_HOURS  = 160
_DEFAULT_OUT    = "cluster_cpu_data.csv"
_PROGRESS_EXT   = ".progress"          # companion file tracking processed parts
_MAX_RETRIES    = 3
_RETRY_BASE_S   = 5                    # initial retry wait (doubles each attempt)
_REQUEST_TIMEOUT = 180                 # seconds per HTTP request


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(message)s",
    datefmt = "%H:%M:%S",
    stream  = sys.stdout,
)
log = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description = "Download Google Cluster Traces 2011 task_usage data.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = (
            "examples:\n"
            "  python download_cluster_data.py\n"
            "  python download_cluster_data.py --hours 80\n"
            "  python download_cluster_data.py --resume\n"
            "  python download_cluster_data.py --hours 160 --out data.csv\n"
        ),
    )
    p.add_argument(
        "--hours", type=float, default=_DEFAULT_HOURS,
        help=f"Hours of trace to download  (default: {_DEFAULT_HOURS})",
    )
    p.add_argument(
        "--out", default=_DEFAULT_OUT, metavar="PATH",
        help=f"Output CSV path  (default: {_DEFAULT_OUT})",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Continue an interrupted download — appends to existing output",
    )
    return p.parse_args()


# ── Progress file ─────────────────────────────────────────────────────────────

def _progress_path(out: Path) -> Path:
    return out.with_suffix(_PROGRESS_EXT)


def _load_progress(out: Path) -> set[int]:
    """Return set of part indices already written to the output file."""
    path = _progress_path(out)
    if path.exists():
        return set(json.loads(path.read_text()).get("done", []))
    return set()


def _save_progress(out: Path, done: set[int]) -> None:
    _progress_path(out).write_text(
        json.dumps({"done": sorted(done)}, indent=2)
    )


# ── Download ──────────────────────────────────────────────────────────────────

def _download_part(part_idx: int) -> bytes | None:
    """Fetch one part file with retry + exponential backoff.

    Returns raw gzip bytes on success, None if all retries are exhausted.
    """
    url  = f"{_BASE_URL}part-{part_idx:05d}-of-00500.csv.gz"
    wait = _RETRY_BASE_S

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=_REQUEST_TIMEOUT) as resp:
                return resp.read()
        except (urllib.error.URLError, OSError) as exc:
            if attempt == _MAX_RETRIES:
                log.error(
                    "  part-%05d: failed after %d attempts — %s",
                    part_idx, _MAX_RETRIES, exc,
                )
                return None
            log.warning(
                "  part-%05d: attempt %d/%d failed (%s), retrying in %ds …",
                part_idx, attempt, _MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)
            wait *= 2

    return None


# ── Parsing ───────────────────────────────────────────────────────────────────

def _parse_part(
    raw_bytes: bytes,
    time_limit_us: int,
) -> tuple[pd.DataFrame, bool]:
    """Parse gzip bytes into a filtered DataFrame.

    Parameters
    ----------
    raw_bytes     : raw .csv.gz content
    time_limit_us : keep rows where start_time <= this value

    Returns
    -------
    (filtered_df, past_limit)
    past_limit = True when the entire file lies beyond the time window,
                 meaning we can safely stop scanning further parts.
    """
    df = pd.read_csv(
        io.BytesIO(raw_bytes),
        compression  = "gzip",
        header       = None,
        usecols      = _USECOLS,
        dtype        = float,       # float64 for all; handles large machine IDs safely
        on_bad_lines = "skip",
    )

    # Columns arrive in index order — assign human-readable names
    df.columns = _COL_NAMES_BY_IDX

    # Stopping condition: entire file is beyond the requested time window
    past_limit = bool(df["start_time"].min() > time_limit_us)

    # Keep only rows within the window, then enforce output column order
    filtered = (
        df[df["start_time"] <= time_limit_us]
        .loc[:, _OUTPUT_ORDER]
    )
    return filtered, past_limit


# ── Writing ───────────────────────────────────────────────────────────────────

def _append_to_csv(df: pd.DataFrame, out: Path, write_header: bool) -> None:
    df.to_csv(out, mode="a", index=False, header=write_header)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args          = _parse_args()
    out           = Path(args.out)
    time_limit_us = int(args.hours * _MICROS_PER_H)

    # ── Validate start state ───────────────────────────────────────────────
    done: set[int] = set()
    write_header   = True

    if args.resume:
        if not out.exists():
            log.error("--resume specified but output file %s does not exist.", out)
            sys.exit(1)
        done         = _load_progress(out)
        write_header = False
        log.info("Resuming — %d parts already processed, appending to %s", len(done), out)
    elif out.exists():
        log.error(
            "Output file %s already exists. "
            "Use --resume to continue or delete it first.",
            out,
        )
        sys.exit(1)

    # ── Banner ─────────────────────────────────────────────────────────────
    log.info("═" * 66)
    log.info("  GOOGLE CLUSTER TRACES 2011 — task_usage download")
    log.info("  Time window  : %.0f hours  (%d µs)", args.hours, time_limit_us)
    log.info("  Columns      : %s", ", ".join(_OUTPUT_ORDER))
    log.info("  Output       : %s", out)
    log.info("═" * 66)

    # ── Download loop ──────────────────────────────────────────────────────
    total_rows    = 0
    parts_written = 0
    t_start       = time.perf_counter()

    for part_idx in range(_TOTAL_PARTS):
        if part_idx in done:
            continue

        elapsed_min = (time.perf_counter() - t_start) / 60
        log.info(
            "[part %3d/~115]  downloading …  (%.1f min elapsed, %s rows so far)",
            part_idx, elapsed_min, f"{total_rows:,}",
        )

        raw = _download_part(part_idx)
        if raw is None:
            log.warning("  Skipping part-%05d (download failed, moving on)", part_idx)
            done.add(part_idx)
            _save_progress(out, done)
            continue

        try:
            df, past_limit = _parse_part(raw, time_limit_us)
        except Exception as exc:
            log.warning("  Skipping part-%05d (parse error: %s)", part_idx, exc)
            done.add(part_idx)
            _save_progress(out, done)
            continue

        if not df.empty:
            _append_to_csv(df, out, write_header=write_header)
            write_header  = False
            total_rows   += len(df)
            parts_written += 1
            size_mb        = out.stat().st_size / 1_048_576
            log.info(
                "  → %s rows written  |  total: %s rows  |  file size: %.1f MB",
                f"{len(df):,}", f"{total_rows:,}", size_mb,
            )

        done.add(part_idx)
        _save_progress(out, done)

        if past_limit:
            log.info(
                "Time limit (%.0f h) reached at part-%05d — stopping.",
                args.hours, part_idx,
            )
            break

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed    = time.perf_counter() - t_start
    final_size = out.stat().st_size / 1_048_576 if out.exists() else 0.0

    log.info("═" * 66)
    log.info("  DONE")
    log.info("  Parts processed : %d  (%d with data)", len(done), parts_written)
    log.info("  Total rows      : %s", f"{total_rows:,}")
    log.info("  Output size     : %.1f MB  (%.2f GB)", final_size, final_size / 1024)
    log.info("  Elapsed         : %.1f min", elapsed / 60)
    log.info("  Output file     : %s", out.resolve())
    log.info("═" * 66)


if __name__ == "__main__":
    main()
