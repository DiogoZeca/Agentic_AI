"""Continuous spike-prediction daemon.

Wraps predict_spike.py in a fixed-rate polling loop: every --interval seconds,
reads --input CSV, runs all loaded models, and atomically writes --output JSON.

The input CSV (cluster_agg format, last 120 min / 24 buckets per machine) is
the caller's responsibility — any monitoring system can produce it.  The daemon
is fully source-agnostic.

Input schema (cluster_agg format)
----------------------------------
  machine_id  int64   — unique machine identifier
  bucket      int64   — 5-min bucket index (monotonically increasing per machine)
  time_us     int64   — bucket * 300_000_000 (microseconds since trace epoch)
  total_cpu   float32 — duration-weighted total CPU load  (fraction of 1 core)
  peak_cpu    float32 — peak cpu_rate observed in this bucket
  total_mem   float32 — sum of canonical_mem_usage across tasks
  peak_mem    float32 — peak max_mem_usage
  disk_io     float32 — max mean_disk_io_time across tasks
  n_tasks     int32   — number of concurrent tasks in this bucket

Usage
-----
    python predict_daemon.py \\
        --input      cpu_window.csv \\
        --model-dir  models/spike/ \\
        --output     predictions.json \\
        --interval   300

    # One-shot (run once and exit — useful for cron / testing)
    python predict_daemon.py --input cpu_window.csv --model-dir models/spike/ \\
        --output predictions.json --interval 0

Timing
------
Cycles fire at a fixed rate: t=0, t=interval, t=2*interval, ...
If inference takes longer than --interval, the next cycle starts immediately
(no back-log accumulation).

Exit codes
----------
    0  clean shutdown (SIGTERM / SIGINT or --interval 0)
    1  fatal startup error (missing artefacts, bad model directory)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from predict_spike import _Artifacts, _load_artifacts, _validate_input, predict_with_artifacts

# ── Logging ───────────────────────────────────────────────────────────────────
# Mirrors predict_spike.py: all log output goes to stderr; stdout is unused.

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(message)s",
    datefmt = "%H:%M:%S",
    stream  = sys.stderr,
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_atomic(result: dict, output_path: Path) -> None:
    """Write result JSON atomically via temp-file + rename.

    Guarantees the output file is either fully written or absent — never a
    partial write, even if the process is killed mid-cycle.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2)
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(payload)
        os.replace(tmp, output_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _log_cycle(result: dict, elapsed: float) -> None:
    """Emit a one-line cycle summary: machine count, alarm list, latency."""
    preds  = result.get("predictions", [])
    alarms = [
        p["machine_id"]
        for p in preds
        if p.get("recommended_action") not in (None, "normal", "monitor")
    ]
    alarm_str = ""
    if alarms:
        shown     = alarms[:5]
        truncated = "…" if len(alarms) > 5 else ""
        alarm_str = f" [{', '.join(str(m) for m in shown)}{truncated}]"

    log.info(
        "  cycle done │ %d machines │ %d alarm(s)%s │ %.1fs",
        result.get("machines_predicted", 0),
        len(alarms),
        alarm_str,
        elapsed,
    )


def _run_cycle(
    input_path:  Path,
    output_path: Path,
    artifacts:   _Artifacts,
) -> Optional[dict]:
    """Execute one inference cycle.

    Returns the result dict on success, or None if the cycle was skipped due to
    a recoverable input error (file missing, unreadable, schema violation).

    Raises on unexpected internal errors so the daemon loop can log them with a
    full traceback without silently continuing.
    """
    if not input_path.exists():
        log.warning("  Input not found: %s  (operator pipeline not ready?) — skipping", input_path)
        return None

    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        # File exists but can't be parsed — likely being written concurrently.
        # Safe to skip: the next cycle will retry with the completed file.
        log.warning("  Could not read input (%s): %s — skipping", input_path.name, exc)
        return None

    try:
        _validate_input(df)
    except ValueError as exc:
        log.warning("  Input schema error — skipping: %s", exc)
        return None

    result = predict_with_artifacts(df, artifacts)
    _write_atomic(result, output_path)
    return result


# ── Main daemon loop ──────────────────────────────────────────────────────────


def run(
    input_path:  Path,
    model_dir:   Path,
    output_path: Path,
    interval:    int,
) -> None:
    """Load models and start the prediction loop.

    Blocks until SIGTERM/SIGINT, or returns immediately if interval=0 (one-shot).
    Exits with code 1 on fatal startup error.
    """
    log.info("═" * 62)
    log.info("  SPIKE PREDICTOR DAEMON")
    log.info("  Input     : %s", input_path.resolve())
    log.info("  Model dir : %s", model_dir.resolve())
    log.info("  Output    : %s", output_path.resolve())
    log.info("  Interval  : %s", f"{interval}s" if interval > 0 else "one-shot")
    log.info("═" * 62)

    try:
        artifacts = _load_artifacts(model_dir)
    except FileNotFoundError as exc:
        log.error("Model artefacts not found: %s", exc)
        log.error("  --model-dir should point to the 60m model directory, e.g. models/spike/")
        sys.exit(1)
    except RuntimeError as exc:
        log.error("Artefact integrity error: %s", exc)
        sys.exit(1)

    # One-shot mode: run once and exit.
    if interval == 0:
        t0     = time.monotonic()
        result = _run_cycle(input_path, output_path, artifacts)
        if result:
            _log_cycle(result, time.monotonic() - t0)
        return

    # Daemon mode: fixed-rate loop until shutdown signal.
    shutdown = threading.Event()

    def _handle_signal(sig: int, _frame) -> None:
        log.info("  Signal %d received — finishing current cycle then stopping …", sig)
        shutdown.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    log.info("  Models loaded. Entering prediction loop (Ctrl-C or SIGTERM to stop).")

    while not shutdown.is_set():
        t0 = time.monotonic()

        try:
            result = _run_cycle(input_path, output_path, artifacts)
        except Exception as exc:
            # Unexpected internal error — log full traceback but keep running.
            # Crashing the daemon over a single bad cycle would be worse.
            log.error("  Unexpected cycle error: %s", exc, exc_info=True)
            result = None

        elapsed   = time.monotonic() - t0
        if result is not None:
            _log_cycle(result, elapsed)

        # Sleep for the remainder of the interval.
        # threading.Event.wait() returns immediately when shutdown is set,
        # so SIGTERM is handled without waiting out the full sleep duration.
        sleep_for = max(0.0, interval - elapsed)
        shutdown.wait(sleep_for)

    log.info("  Daemon stopped cleanly.")


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "predict_daemon.py",
        description = (
            "Continuous spike-prediction daemon.\n"
            "Reads --input CSV every --interval seconds, runs all models,\n"
            "and writes --output JSON atomically. Source-agnostic — any\n"
            "monitoring system can produce the input CSV."
        ),
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", "-i",
        required = True,
        type     = Path,
        metavar  = "CSV",
        help     = "Per-machine aggregated data CSV (cluster_agg format, last 120 min).",
    )
    p.add_argument(
        "--model-dir", "-m",
        required = True,
        dest     = "model_dir",
        type     = Path,
        metavar  = "DIR",
        help     = "Directory containing the 60m model artefacts (spike_model.json etc.).",
    )
    p.add_argument(
        "--output", "-o",
        required = True,
        type     = Path,
        metavar  = "JSON",
        help     = "Path for atomic JSON output (overwritten each cycle).",
    )
    p.add_argument(
        "--interval", "-n",
        default = 300,
        type    = int,
        metavar = "SECONDS",
        help    = "Seconds between inference cycles (default: 300). Use 0 for one-shot.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.interval < 0:
        log.error("--interval must be >= 0 (use 0 for one-shot mode).")
        sys.exit(1)

    if not args.model_dir.is_dir():
        log.error("Model directory not found: %s", args.model_dir)
        sys.exit(1)

    run(
        input_path  = args.input,
        model_dir   = args.model_dir,
        output_path = args.output,
        interval    = args.interval,
    )


if __name__ == "__main__":
    main()
