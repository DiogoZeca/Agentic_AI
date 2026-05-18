"""Thanos adapter — fetch OSM node metrics and feed the spike prediction API.

Queries a Thanos (Prometheus-compatible) query_range endpoint for the last
120 minutes of node-exporter metrics, builds a cluster_agg DataFrame, and
either prints it as CSV or POSTs it to the spike /summary API.

Metric pipeline on the OSM cluster
-----------------------------------
  node-exporter (port 9100)
    → OTEL daemon-collector
    → thanos-receive
    → Thanos long-term storage   ← we query here

Usage
-----
    # Port-forward Thanos first (from outside the cluster):
    kubectl port-forward svc/thanos-query 19091:9090 -n edgeharbor-controller

    # Print cluster_agg CSV to stdout (pipe or use as --fetch-cmd for daemon):
    python training/adapters/thanos.py \\
        --thanos-url http://localhost:19091 \\
        --mode csv

    # POST directly to /summary and print the JSON response:
    python training/adapters/thanos.py \\
        --thanos-url http://localhost:19091 \\
        --mode post \\
        --api-url http://spike-api.10.255.42.75.nip.io/summary

    # Inside the cluster (no port-forward needed):
    python training/adapters/thanos.py \\
        --thanos-url http://thanos-query.edgeharbor-controller.svc.cluster.local:9090 \\
        --mode post \\
        --api-url http://spike-api.spike.svc.cluster.local:8000/summary
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

_STEP = 300  # 5-minute bucket width in seconds

# ── PromQL queries ─────────────────────────────────────────────────────────────
# All sourced from node-exporter — label `nodename` identifies the host.
# total_cpu/peak_cpu are in fractions-of-core (1.0 = one full core busy),
# matching the cluster_agg schema from training/adapters/README.md.

_Q_TOTAL_CPU = (
    'sum(rate(node_cpu_seconds_total{mode!="idle"}[5m])) by (nodename)'
)
_Q_PEAK_CPU = (
    "max_over_time("
    '  sum(rate(node_cpu_seconds_total{mode!="idle"}[1m])) by (nodename)'
    "[5m:1m])"
)
_Q_TOTAL_MEM = "node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes"
_Q_DISK_IO   = "max(rate(node_disk_io_time_seconds_total[5m])) by (nodename)"
_Q_LOAD1     = "node_load1"  # 1-min load average — proxy for n_tasks


# ── HTTP helper ───────────────────────────────────────────────────────────────


def _query_range(
    base_url: str,
    promql:   str,
    start:    int,
    end:      int,
    step:     int = _STEP,
) -> pd.DataFrame:
    """Execute one Thanos query_range call.

    Returns a tidy DataFrame with columns [timestamp, nodename, value].
    Returns an empty DataFrame when the result set is empty.
    Raises RuntimeError on HTTP or API-level errors.
    """
    params = urllib.parse.urlencode({
        "query": promql,
        "start": start,
        "end":   end,
        "step":  step,
    })
    url = f"{base_url.rstrip('/')}/api/v1/query_range?{params}"

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            payload = json.loads(resp.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Thanos request failed ({url}): {exc}") from exc

    if payload.get("status") != "success":
        raise RuntimeError(
            f"Thanos error status={payload.get('status')}: {payload.get('error')}"
        )

    rows = []
    for series in payload["data"]["result"]:
        metric   = series["metric"]
        nodename = metric.get("nodename") or metric.get("node", "unknown")
        for ts_raw, val_raw in series["values"]:
            rows.append({
                "timestamp": int(float(ts_raw)),
                "nodename":  nodename,
                "value":     float(val_raw),
            })

    if not rows:
        return pd.DataFrame(columns=["timestamp", "nodename", "value"])
    return pd.DataFrame(rows)


# ── Machine-ID mapping ────────────────────────────────────────────────────────


def _machine_id_map(nodenames: list[str]) -> dict[str, int]:
    """Assign stable sequential integer IDs to node names (alphabetical order).

    Alphabetical ordering ensures IDs are consistent across calls regardless
    of the order nodes appear in Thanos results.
    """
    return {name: idx + 1 for idx, name in enumerate(sorted(set(nodenames)))}


# ── Grid reindex ─────────────────────────────────────────────────────────────


def _reindex_to_full_grid(
    merged: pd.DataFrame,
    start:  int,
    end:    int,
    step:   int = _STEP,
) -> pd.DataFrame:
    """Expand sparse Thanos data to a complete 5-min grid, filling gaps per column.

    Thanos omits buckets where rate() has insufficient data (scrape gaps, idle
    periods).  Without this step the feature engineer sees too few rows → NaN
    lag features → cold-start classification even for a running node.

    Filling strategy (research-backed for monitoring metrics):
      - total_cpu, peak_cpu, disk_io, load1:
            zero-fill — rate() = 0 when the node is idle; no activity = no value
      - total_mem, peak_mem:
            forward-fill then backward-fill — the OS always has memory allocated;
            it does not drop to 0 between scrapes
    """
    all_timestamps = list(range(start + step, end + step, step))
    nodes = merged["nodename"].unique()

    grid = pd.DataFrame(
        list(itertools.product(nodes, all_timestamps)),
        columns=["nodename", "timestamp"],
    )

    full = grid.merge(merged, on=["nodename", "timestamp"], how="left")
    full = full.sort_values(["nodename", "timestamp"]).reset_index(drop=True)

    for col in ("total_cpu", "peak_cpu", "disk_io", "load1"):
        if col in full.columns and full[col].notna().any():
            # Only zero-fill gap rows. If the column is entirely NaN the metric
            # was unavailable — leave NaN so the caller's fallback logic fires.
            full[col] = full[col].fillna(0.0)

    for col in ("total_mem", "peak_mem"):
        if col in full.columns:
            full[col] = (
                full.groupby("nodename")[col]
                    .transform(lambda s: s.ffill().bfill())
            )
            full[col] = full[col].fillna(0.0)

    filled = len(full) - len(merged)
    if filled > 0:
        log.info(
            "Grid reindex: filled %d missing bucket(s) — sparse Thanos scrape data",
            filled,
        )

    return full


# ── Core fetch ────────────────────────────────────────────────────────────────


def fetch_cluster_agg(
    thanos_url:       str,
    lookback_minutes: int            = 120,
    end_time:         Optional[int]  = None,
) -> pd.DataFrame:
    """Query Thanos and return a cluster_agg DataFrame.

    Parameters
    ----------
    thanos_url       : base URL of Thanos query API
    lookback_minutes : minutes of history to fetch (default 120 = 24 buckets)
    end_time         : Unix timestamp for window end; defaults to now
                       (aligned to the nearest 5-min boundary)

    Returns
    -------
    DataFrame with columns: machine_id, bucket, time_us,
    total_cpu, peak_cpu, total_mem, peak_mem, disk_io, n_tasks
    sorted by (machine_id, bucket).
    """
    end   = ((end_time or int(time.time())) // _STEP) * _STEP
    start = end - lookback_minutes * 60

    log.info(
        "Querying Thanos %s  window=%dm  buckets=%d",
        thanos_url, lookback_minutes, lookback_minutes // 5,
    )

    total_cpu_df = _query_range(thanos_url, _Q_TOTAL_CPU, start, end)

    if total_cpu_df.empty:
        raise RuntimeError(
            "Thanos returned no CPU data. "
            "Check that node-exporter metrics are flowing into Thanos."
        )

    # peak_cpu via sub-query — may fail on older Thanos; fall back to total_cpu
    try:
        peak_cpu_df = _query_range(thanos_url, _Q_PEAK_CPU, start, end)
    except RuntimeError:
        log.warning("peak_cpu sub-query failed — using total_cpu as peak_cpu")
        peak_cpu_df = pd.DataFrame(columns=["timestamp", "nodename", "value"])

    total_mem_df = _query_range(thanos_url, _Q_TOTAL_MEM, start, end)
    disk_io_df   = _query_range(thanos_url, _Q_DISK_IO,   start, end)

    # n_tasks: try node_load1 first, fall back to round(total_cpu)
    try:
        load1_df = _query_range(thanos_url, _Q_LOAD1, start, end)
        if load1_df.empty:
            raise ValueError("empty")
        n_tasks_source = "node_load1"
    except (RuntimeError, ValueError):
        load1_df       = None
        n_tasks_source = "total_cpu_fallback"
    log.info("n_tasks source: %s", n_tasks_source)

    # ── Pivot all metrics onto (nodename, timestamp) index ────────────────────

    def _pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
        return (
            df.rename(columns={"value": col})
              .set_index(["nodename", "timestamp"])
        )

    merged = _pivot(total_cpu_df, "total_cpu")

    for df_src, col in [
        (peak_cpu_df,  "peak_cpu"),
        (total_mem_df, "total_mem"),
        (disk_io_df,   "disk_io"),
    ]:
        if not df_src.empty:
            merged = merged.join(_pivot(df_src, col), how="left")
        else:
            merged[col] = float("nan")

    if load1_df is not None:
        merged = merged.join(_pivot(load1_df, "load1"), how="left")

    merged = merged.reset_index()

    # ── Reindex to full 5-min grid ────────────────────────────────────────────
    # Thanos only returns buckets where metrics had data. Reindexing to all
    # expected timestamps fills idle gaps so lag/EWMA features don't go NaN.
    merged = _reindex_to_full_grid(merged, start, end)

    # ── Fill missing values per README fallback rules ─────────────────────────

    # peak_cpu: use total_cpu when sub-query unavailable
    if "peak_cpu" not in merged.columns or merged["peak_cpu"].isna().all():
        merged["peak_cpu"] = merged["total_cpu"]
    else:
        merged["peak_cpu"] = merged["peak_cpu"].fillna(merged["total_cpu"])

    # peak_mem = total_mem (no sub-minute memory history available)
    merged["peak_mem"] = merged.get("total_mem", merged["total_cpu"])

    # disk_io: fill with 0.0 when unavailable
    if "disk_io" not in merged.columns:
        merged["disk_io"] = 0.0
    else:
        merged["disk_io"] = merged["disk_io"].fillna(0.0)

    # n_tasks: max(1, round(load1)) or max(1, round(total_cpu))
    n_tasks_base = (
        merged["load1"].fillna(merged["total_cpu"])
        if "load1" in merged.columns
        else merged["total_cpu"]
    )
    merged["n_tasks"] = n_tasks_base.apply(lambda x: max(1, round(x))).astype("int32")

    # ── bucket + machine_id ───────────────────────────────────────────────────

    merged["bucket"]  = (merged["timestamp"].astype("int64") // _STEP)
    merged["time_us"] = merged["bucket"] * 300_000_000

    id_map             = _machine_id_map(merged["nodename"].tolist())
    merged["machine_id"] = merged["nodename"].map(id_map).astype("int64")

    # Drop rows where mandatory fields are missing
    merged = merged.dropna(subset=["total_cpu", "total_mem"])

    result = merged[[
        "machine_id", "bucket",    "time_us",
        "total_cpu",  "peak_cpu",
        "total_mem",  "peak_mem",
        "disk_io",    "n_tasks",
    ]].copy()

    for col in ("total_cpu", "peak_cpu", "total_mem", "peak_mem", "disk_io"):
        result[col] = result[col].astype("float32")

    result = result.sort_values(["machine_id", "bucket"]).reset_index(drop=True)

    buckets_per_node = (
        result.groupby("machine_id").size().max() if len(result) > 0 else 0
    )
    log.info(
        "cluster_agg ready: %d rows  %d node(s)  %d buckets/node",
        len(result), result["machine_id"].nunique(), buckets_per_node,
    )
    return result


# ── API POST ──────────────────────────────────────────────────────────────────


def post_to_api(df: pd.DataFrame, api_url: str) -> dict:
    """POST cluster_agg rows to /summary and return the parsed response."""
    payload = json.dumps({"rows": df.to_dict(orient="records")}).encode()
    req = urllib.request.Request(
        api_url,
        data    = payload,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"API returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"API request failed: {exc}") from exc


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "thanos.py",
        description = (
            "Fetch OSM node metrics from Thanos → spike prediction API.\n\n"
            "Modes:\n"
            "  csv   Print cluster_agg CSV to stdout\n"
            "        (use as --fetch-cmd for daemon.py)\n"
            "  post  POST to /summary and print the JSON response"
        ),
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--thanos-url", required=True, metavar="URL",
        help="Base URL of Thanos query API (e.g. http://localhost:19091)",
    )
    p.add_argument(
        "--mode", choices=["csv", "post"], default="csv",
        help="Output mode (default: csv)",
    )
    p.add_argument(
        "--api-url", metavar="URL", default=None,
        help="Spike API /summary URL — required for --mode post",
    )
    p.add_argument(
        "--lookback-minutes", type=int, default=120, metavar="N",
        help="Minutes of history to fetch (default: 120 = 24 buckets)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt = "%H:%M:%S",
        stream  = sys.stderr,
    )
    args = _build_parser().parse_args(argv)

    if args.mode == "post" and not args.api_url:
        print("ERROR: --api-url is required for --mode post", file=sys.stderr)
        sys.exit(1)

    df = fetch_cluster_agg(args.thanos_url, lookback_minutes=args.lookback_minutes)

    if args.mode == "csv":
        df.to_csv(sys.stdout, index=False)
    else:
        result = post_to_api(df, args.api_url)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
