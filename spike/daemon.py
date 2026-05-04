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
        --model-dir  data/full_run/spike \\
        --output     predictions.json \\
        --interval   300

    # Fetch-command mode (operator provides a data-collection script)
    python spike/daemon.py \\
        --fetch-cmd  "python my_data_script.py" \\
        --model-dir  data/full_run/spike \\
        --output     predictions.json \\
        --interval   300

    # One-shot (run once and exit — useful for cron / testing)
    python spike/daemon.py --input cpu_window.csv --model-dir data/full_run/spike \\
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
import collections
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
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


def _extract_cycle_stats(result: dict) -> dict:
    """Extract machine and alarm counts from a successful prediction result.

    Factored out so _log_cycle and _record_slo share identical alarm-counting
    logic.  An alarm is any machine whose recommended_action is not one of the
    passive states (None, "normal", "monitor").

    Returns a dict with keys: machines_total, machines_predicted,
    machines_cold_start, alarm_count, alarm_rate, cold_start_rate, alarm_list.
    """
    preds              = result.get("predictions", [])
    machines_total     = result.get("machines_total", 0)
    machines_predicted = result.get("machines_predicted", 0)
    machines_cold_start = result.get("machines_cold_start", 0)

    alarm_list = [
        p["machine_id"]
        for p in preds
        if p.get("recommended_action") not in (None, "normal", "monitor")
    ]

    # Guard against zero-denominator when all machines are cold-start.
    alarm_rate      = len(alarm_list) / machines_predicted if machines_predicted > 0 else None
    cold_start_rate = machines_cold_start / machines_total  if machines_total > 0     else None

    return {
        "machines_total":      machines_total,
        "machines_predicted":  machines_predicted,
        "machines_cold_start": machines_cold_start,
        "alarm_count":         len(alarm_list),
        "alarm_rate":          alarm_rate,
        "cold_start_rate":     cold_start_rate,
        "alarm_list":          alarm_list,
    }


def _log_cycle(result: dict, elapsed: float) -> None:
    """Emit a one-line cycle summary: machine count, alarm list, latency."""
    stats  = _extract_cycle_stats(result)
    alarms = stats["alarm_list"]

    alarm_str = ""
    if alarms:
        shown     = alarms[:5]
        truncated = "…" if len(alarms) > 5 else ""
        alarm_str = f" [{', '.join(str(m) for m in shown)}{truncated}]"

    log.info(
        "  cycle done │ %d machines │ %d alarm(s)%s │ %.1fs",
        stats["machines_predicted"],
        stats["alarm_count"],
        alarm_str,
        elapsed,
    )


def _record_slo(
    slo_deque:   collections.deque,
    cycle_index: int,
    elapsed:     float,
    result:      Optional[dict],
) -> None:
    """Append one cycle's metrics to the rolling SLO deque.

    Called after every cycle — including skipped/failed ones (result=None).
    The deque caps automatically at maxlen=300 (~25h at 5-min intervals),
    evicting the oldest entry when full.
    """
    if result is not None:
        stats = _extract_cycle_stats(result)
        # Round rates to 4 d.p. to keep the JSON file human-readable.
        alarm_rate      = round(stats["alarm_rate"],      4) if stats["alarm_rate"]      is not None else None
        cold_start_rate = round(stats["cold_start_rate"], 4) if stats["cold_start_rate"] is not None else None
    else:
        # Skipped cycle — no prediction stats available.
        stats           = {}
        alarm_rate      = None
        cold_start_rate = None

    slo_deque.append({
        "cycle_index":         cycle_index,
        "timestamp":           datetime.now(timezone.utc).isoformat(),
        "latency_s":           round(elapsed, 3),
        "success":             result is not None,
        "machines_total":      stats.get("machines_total"),
        "machines_predicted":  stats.get("machines_predicted"),
        "machines_cold_start": stats.get("machines_cold_start"),
        "alarm_count":         stats.get("alarm_count"),
        "alarm_rate":          alarm_rate,
        "cold_start_rate":     cold_start_rate,
    })


def _write_slo(
    slo_path:      Path,
    slo_deque:     collections.deque,
    *,
    model_version: Optional[dict],
) -> None:
    """Atomically write rolling SLO metrics to slo_path.

    Reuses _write_atomic (temp+rename) so readers never see a partial file.
    slo_path is typically output_path.parent / "slo_metrics.json".

    The summary block is recomputed from the full deque on every write.
    Latency percentiles cover ALL cycles (success + skip); alarm_rate and
    cold_start_rate statistics cover successful cycles only.
    """
    cycles    = list(slo_deque)
    successful = [c for c in cycles if c["success"]]

    latencies   = [c["latency_s"] for c in cycles]
    alarm_rates = [c["alarm_rate"] for c in successful if c.get("alarm_rate") is not None]
    cold_rates  = [c["cold_start_rate"] for c in successful if c.get("cold_start_rate") is not None]

    summary = {
        "window_cycles":        len(cycles),
        "window_start":         cycles[0]["timestamp"]  if cycles else None,
        "window_end":           cycles[-1]["timestamp"] if cycles else None,
        "success_rate":         round(len(successful) / len(cycles), 4) if cycles else None,
        "latency_p50_s":        round(float(np.percentile(latencies, 50)), 3) if latencies else None,
        "latency_p95_s":        round(float(np.percentile(latencies, 95)), 3) if latencies else None,
        "alarm_rate_mean":      round(float(np.mean(alarm_rates)),   4) if alarm_rates else None,
        # std requires >= 2 points; a single-cycle window makes std meaningless.
        "alarm_rate_std":       round(float(np.std(alarm_rates)),    4) if len(alarm_rates) >= 2 else None,
        "cold_start_rate_mean": round(float(np.mean(cold_rates)),    4) if cold_rates else None,
    }

    _write_atomic(
        {"model_version": model_version, "summary": summary, "cycles": cycles},
        slo_path,
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
    model_dir:       Path,
    output_path:     Path,
    interval:        int,
    *,
    input_path:      Optional[Path]  = None,
    fetch_cmd:       Optional[str]   = None,
    domain_dir:      Optional[Path]  = None,
    alarm_threshold: Optional[float] = None,
) -> None:
    """Load models and start the prediction loop.

    Exactly one of *input_path* or *fetch_cmd* must be provided.
    Blocks until SIGTERM/SIGINT, or returns immediately if interval=0 (one-shot).
    Exits with code 1 on fatal startup error.

    Parameters
    ----------
    model_dir       : directory containing the 60m model artefacts.
    output_path     : path for atomic JSON output (overwritten each cycle).
    interval        : seconds between inference cycles; 0 for one-shot mode.
    input_path      : pre-written cluster_agg CSV (file mode).
    fetch_cmd       : shell command whose stdout is parsed as cluster_agg CSV.
    domain_dir      : optional directory from bootstrap_thresholds.py with
                      domain-specific spike_thresholds.parquet.
    alarm_threshold : optional float in [0, 1] that overrides the trained alarm
                      threshold for every loaded horizon (60m, 15m, 30m, 45m,
                      severe_ovr).  Applied once after artefact loading; persists
                      for the full daemon lifetime.  When None the per-horizon
                      values from spike_config.json are used.
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
    if alarm_threshold is not None:
        log.info("  Alarm thr : %.4f (operator override — all horizons)", alarm_threshold)
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

    # Apply operator-supplied alarm threshold override once at startup.
    # predict_with_artifacts() reads thresholds from the artifacts object each cycle,
    # so mutating the dict here is sufficient — no per-cycle overhead.
    if alarm_threshold is not None:
        for h in artifacts.alarm_thresholds:
            artifacts.alarm_thresholds[h] = alarm_threshold
        log.info(
            "  Alarm threshold override applied: %.4f → horizon(s): %s",
            alarm_threshold,
            ", ".join(sorted(artifacts.alarm_thresholds)),
        )

    cycle_kwargs: dict = dict(
        output_path = output_path,
        artifacts   = artifacts,
        input_path  = input_path,
        fetch_cmd   = fetch_cmd,
    )

    # Rolling SLO tracker: a deque of per-cycle metric dicts capped at 300
    # entries (~25 h at the default 5-min interval).  The oldest entry is
    # evicted automatically when the window is full.
    # slo_metrics.json is written alongside predictions.json after every cycle.
    slo_deque: collections.deque = collections.deque(maxlen=300)
    slo_path  = output_path.parent / "slo_metrics.json"
    # version_info is set once at startup; None for legacy artefacts that
    # pre-date the versioning step.
    model_version = artifacts.version_info

    # One-shot mode: run once and exit.
    if interval == 0:
        t0      = time.monotonic()
        result  = _run_cycle(**cycle_kwargs)
        elapsed = time.monotonic() - t0
        if result:
            _log_cycle(result, elapsed)
        _record_slo(slo_deque, cycle_index=0, elapsed=elapsed, result=result)
        _write_slo(slo_path, slo_deque, model_version=model_version)
        return

    # Daemon mode: fixed-rate loop until shutdown signal.
    shutdown = threading.Event()

    def _handle_signal(sig: int, _frame) -> None:
        log.info("  Signal %d received — finishing current cycle then stopping …", sig)
        shutdown.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    log.info("  Models loaded. Entering prediction loop (Ctrl-C or SIGTERM to stop).")

    cycle_index = 0
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

        # Record this cycle and flush SLO metrics — even on skipped cycles so
        # the skip rate is visible in the rolling summary.
        _record_slo(slo_deque, cycle_index=cycle_index, elapsed=elapsed, result=result)
        _write_slo(slo_path, slo_deque, model_version=model_version)
        cycle_index += 1

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
    # --alarm-threshold: runtime operating-point override applied once at startup.
    # For long-running daemons this is the right granularity — operators set it
    # via their init script / systemd unit and it persists for the daemon lifetime.
    # To change it mid-run, restart the daemon with the new value.
    # See predict.py --alarm-threshold and _Artifacts docstring for the full
    # breakdown of which outputs are affected (binary is_spike, 60m is_severe)
    # vs. which are not (60m is_spike, which is always argmax-based).
    p.add_argument(
        "--alarm-threshold", "-t",
        type    = float,
        default = None,
        dest    = "alarm_threshold",
        metavar = "FLOAT",
        help    = (
            "Override the alarm threshold for all prediction horizons (0.0–1.0). "
            "Applied once at startup and held for the daemon lifetime. "
            "Lower values raise sensitivity (more alarms, fewer missed spikes); "
            "higher values reduce false positives. "
            "Default: per-horizon values stored in spike_config.json at training time."
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

    if args.alarm_threshold is not None and not (0.0 <= args.alarm_threshold <= 1.0):
        log.error("--alarm-threshold must be in [0.0, 1.0] (got %.4g).", args.alarm_threshold)
        sys.exit(1)

    run(
        model_dir       = args.model_dir,
        output_path     = args.output,
        interval        = args.interval,
        input_path      = args.input,
        fetch_cmd       = args.fetch_cmd,
        domain_dir      = args.domain_dir,
        alarm_threshold = args.alarm_threshold,
    )


if __name__ == "__main__":
    main()
