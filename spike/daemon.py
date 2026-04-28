"""Continuous spike-prediction daemon.

Wraps predict_spike.py in a fixed-rate polling loop: every --interval seconds,
reads cluster_agg data, runs all loaded models, and atomically writes --output JSON.

Data can be supplied in two ways (mutually exclusive):

  --input CSV        Read a pre-written cluster_agg CSV file each cycle.
                     Any monitoring system can write this file; the daemon is
                     fully source-agnostic.

  --fetch-cmd CMD    Run CMD as a shell command each cycle and parse its stdout
                     as a cluster_agg CSV.  Use this to drive any data-collection
                     script directly from the daemon without an intermediate file.

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
    # File mode (operator writes cpu_window.csv every 5 min via their own script)
    python spike/daemon.py \\
        --input      cpu_window.csv \\
        --model-dir  /path/to/models/spike \\
        --output     predictions.json \\
        --interval   300

    # Fetch-command mode (operator provides a data-collection script)
    python spike/daemon.py \\
        --fetch-cmd  "python my_data_script.py" \\
        --model-dir  /path/to/models/spike \\
        --output     predictions.json \\
        --interval   300

    # One-shot (run once and exit — useful for cron / testing)
    python spike/daemon.py --input cpu_window.csv --model-dir /path/to/models/spike \\
        --output predictions.json --interval 0

Timing
------
Cycles fire at a fixed rate: t=0, t=interval, t=2*interval, ...
If inference takes longer than --interval, the next cycle starts immediately
(no back-log accumulation).

Exit codes
----------
    0  clean shutdown (SIGTERM / SIGINT or --interval 0)
    1  fatal startup error (missing artefacts, bad model directory, bad flags)
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from spike.predict import _Artifacts, _load_artifacts, _prepare_input, predict_with_artifacts

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


def _fetch_from_command(cmd: str) -> Optional[pd.DataFrame]:
    """Run *cmd* as a shell command and parse its stdout as a cluster_agg CSV.

    Returns a DataFrame on success, or None on any recoverable error (timeout,
    non-zero exit code, empty output, unparseable CSV).  The caller skips the
    cycle on None and retries at the next interval.

    shell=True is intentional: operators pass full shell commands including
    pipes, environment substitutions, and argument lists.
    """
    try:
        proc = subprocess.run(
            cmd,
            shell          = True,
            capture_output = True,
            text           = True,
            timeout        = 60,
        )
    except subprocess.TimeoutExpired:
        log.warning("  --fetch-cmd timed out after 60s — skipping cycle")
        return None

    if proc.returncode != 0:
        log.warning(
            "  --fetch-cmd exited %d — skipping cycle. stderr: %s",
            proc.returncode,
            (proc.stderr[:300].strip() or "(none)"),
        )
        return None

    if not proc.stdout.strip():
        log.warning("  --fetch-cmd produced no output — skipping cycle")
        return None

    try:
        return pd.read_csv(io.StringIO(proc.stdout))
    except Exception as exc:
        log.warning("  --fetch-cmd output could not be parsed as CSV: %s — skipping", exc)
        return None


def _run_cycle(
    output_path: Path,
    artifacts:   _Artifacts,
    *,
    input_path:  Optional[Path] = None,
    fetch_cmd:   Optional[str]  = None,
) -> Optional[dict]:
    """Execute one inference cycle.

    Exactly one of *input_path* or *fetch_cmd* must be provided (enforced by
    the CLI).  Returns the result dict on success, or None when the cycle is
    skipped due to a recoverable error (missing file, command failure, schema
    violation).  Raises on unexpected internal errors so the daemon loop can
    log a full traceback without silently continuing.
    """
    if fetch_cmd is not None:
        df = _fetch_from_command(fetch_cmd)
        if df is None:
            return None
    else:
        assert input_path is not None
        if not input_path.exists():
            log.warning(
                "  Input not found: %s  (operator pipeline not ready?) — skipping",
                input_path,
            )
            return None
        try:
            df = pd.read_csv(input_path)
        except Exception as exc:
            # File exists but can't be parsed — likely being written concurrently.
            # Safe to skip: the next cycle will retry with the completed file.
            log.warning("  Could not read input (%s): %s — skipping", input_path.name, exc)
            return None

    try:
        _prepare_input(df)
    except ValueError as exc:
        log.warning("  Input schema error — skipping: %s", exc)
        return None

    result = predict_with_artifacts(df, artifacts)
    _write_atomic(result, output_path)
    return result


# ── Main daemon loop ──────────────────────────────────────────────────────────


def run(
    model_dir:   Path,
    output_path: Path,
    interval:    int,
    *,
    input_path:  Optional[Path] = None,
    fetch_cmd:   Optional[str]  = None,
    domain_dir:  Optional[Path] = None,
) -> None:
    """Load models and start the prediction loop.

    Exactly one of *input_path* or *fetch_cmd* must be provided.
    Blocks until SIGTERM/SIGINT, or returns immediately if interval=0 (one-shot).
    Exits with code 1 on fatal startup error.
    """
    input_label = str(input_path.resolve()) if input_path else f"cmd: {fetch_cmd}"

    log.info("═" * 62)
    log.info("  SPIKE PREDICTOR DAEMON")
    log.info("  Input     : %s", input_label)
    log.info("  Model dir : %s", model_dir.resolve())
    if domain_dir is not None:
        log.info("  Domain dir: %s", domain_dir.resolve())
    log.info("  Output    : %s", output_path.resolve())
    log.info("  Interval  : %s", f"{interval}s" if interval > 0 else "one-shot")
    log.info("═" * 62)

    try:
        artifacts = _load_artifacts(model_dir, domain_dir=domain_dir)
    except FileNotFoundError as exc:
        log.error("Model artefacts not found: %s", exc)
        log.error("  --model-dir should point to the directory containing spike_model.json")
        sys.exit(1)
    except RuntimeError as exc:
        log.error("Artefact integrity error: %s", exc)
        sys.exit(1)

    cycle_kwargs: dict = dict(
        output_path = output_path,
        artifacts   = artifacts,
        input_path  = input_path,
        fetch_cmd   = fetch_cmd,
    )

    # One-shot mode: run once and exit.
    if interval == 0:
        t0     = time.monotonic()
        result = _run_cycle(**cycle_kwargs)
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
            result = _run_cycle(**cycle_kwargs)
        except Exception as exc:
            # Unexpected internal error — log full traceback but keep running.
            # Crashing the daemon over a single bad cycle would be worse.
            log.error("  Unexpected cycle error: %s", exc, exc_info=True)
            result = None

        elapsed = time.monotonic() - t0
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
        prog        = "daemon.py",
        description = (
            "Continuous spike-prediction daemon.\n"
            "Every --interval seconds, reads cluster_agg data, runs all models,\n"
            "and writes --output JSON atomically.\n\n"
            "Data source (exactly one required):\n"
            "  --input CSV       read a pre-written cluster_agg CSV file\n"
            "  --fetch-cmd CMD   run CMD and parse its stdout as cluster_agg CSV"
        ),
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )

    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input", "-i",
        type    = Path,
        metavar = "CSV",
        help    = "Per-machine aggregated data CSV (cluster_agg format, last 120 min).",
    )
    source.add_argument(
        "--fetch-cmd", "-f",
        dest    = "fetch_cmd",
        metavar = "CMD",
        help    = (
            "Shell command to run each cycle. Its stdout must be a cluster_agg CSV. "
            "Use this to drive any data-collection script directly."
        ),
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
        "--domain-dir", "-d",
        dest    = "domain_dir",
        type    = Path,
        metavar = "DIR",
        default = None,
        help    = (
            "Optional directory produced by bootstrap_thresholds.py. "
            "When provided, spike_thresholds.parquet is loaded from here "
            "instead of --model-dir, applying domain-specific CPU thresholds "
            "without retraining."
        ),
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
        model_dir   = args.model_dir,
        output_path = args.output,
        interval    = args.interval,
        input_path  = args.input,
        fetch_cmd   = args.fetch_cmd,
        domain_dir  = args.domain_dir,
    )


if __name__ == "__main__":
    main()
