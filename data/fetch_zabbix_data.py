#!/usr/bin/env python3
"""
fetch_zabbix_data.py — Pull 89-day Zabbix history → cluster_agg schema parquet.

Fetches CPU, memory, disk I/O, and load average from all active Zabbix nodes.
Converts units to the Google Cluster Traces cluster_agg schema expected by
spike_feature_engineer.engineer().

Unit conversions
----------------
  total_cpu  = (system.cpu.util [%] / 100) * n_cores     → fraction-of-core
  peak_cpu   = total_cpu  (one sample per bucket; no intra-bucket peak available)
  total_mem  = vm.memory.utilization [%] / 100            → fraction of capacity
  peak_mem   = total_mem
  disk_io    = vfs.dev.util[*] [%] / 100                  → device utilisation [0,1]
  n_tasks    = max(1, round(system.cpu.load[all,avg1]))   → int32

Bucket alignment
----------------
  bucket   = unix_timestamp_seconds // 300
  time_us  = bucket * 300_000_000
  bucket % 288 == 0 at midnight UTC (Unix epoch is midnight UTC; 86400/300 = 288).

Outputs (written to data/zabbix_eval/)
-------
  zabbix_agg.parquet  — cluster_agg schema
  node_map.json       — {"hostname": machine_id_int, ...}  stable alphabetical order

Credentials: fill in ZABBIX_USER + ZABBIX_PASS below before running.
Never commit credentials to git.

Usage (from AIModel/):
  .venv/bin/python3 data/fetch_zabbix_data.py [--days N] [--output DIR]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import urllib3

# ══════════════════════════════════════════════════════════════════════════════
#  CREDENTIALS  —  fill in before running, do not commit to git
# ══════════════════════════════════════════════════════════════════════════════

ZABBIX_URL       = "http://atnog-mon.av.it.pt/zabbix/api_jsonrpc.php"
ZABBIX_API_TOKEN = ""      # Option A: API token (Users → API tokens)
ZABBIX_USER      = "" # Option B: username (leave empty if using token)
ZABBIX_PASS      = ""      # Option B: password
VERIFY_SSL       = False

# ══════════════════════════════════════════════════════════════════════════════

_DEFAULT_HISTORY_DAYS = 89
_DEFAULT_OUTPUT_DIR   = Path("data/zabbix_eval")

# Days per history.get chunk.  1-day windows → max 288 records × n_nodes < 5000.
_CHUNK_DAYS = 1

# Polite inter-request pause (seconds).
_RATE_LIMIT = 0.05

# Fall-back core count when system.cpu.num is unavailable.
_DEFAULT_NCORES = 4

# Anchor key used to discover active compute nodes (same as explore_zabbix.py).
_ANCHOR_KEY = "system.cpu.load"

# Item key patterns — substring match, most-specific pattern listed first.
_KEY_CPU_UTIL  = "system.cpu.util"          # overall CPU utilisation %
_KEY_MEM_UTIL  = "vm.memory.utilization"    # RAM utilisation %
_KEY_DISK_UTIL = "vfs.dev.util["            # device I/O utilisation % (any device)
_KEY_LOAD_AVG  = "system.cpu.load[all,avg1" # 1-min load average
_KEY_NCORES    = "system.cpu.num"           # logical CPU count (static)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger(__name__)

if not VERIFY_SSL:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_sess = requests.Session()
_sess.verify = VERIFY_SSL
_sess.headers["Content-Type"] = "application/json"
_token: str | None = None


# ── Auth ──────────────────────────────────────────────────────────────────────

def _api(method: str, params: dict) -> Any:
    body    = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
    headers = {}
    if _token and method != "apiinfo.version":
        headers["Authorization"] = f"Bearer {_token}"
    r = _sess.post(ZABBIX_URL, json=body, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        err = data["error"]
        raise RuntimeError(f"[{method}] {err.get('data') or err.get('message') or err}")
    return data["result"]


def _login() -> None:
    global _token
    if ZABBIX_API_TOKEN:
        _token = ZABBIX_API_TOKEN
        log.info("Auth: API token")
    else:
        _token = _api("user.login", {"username": ZABBIX_USER, "password": ZABBIX_PASS})
        log.info("Auth: session token via user.login")


def _logout() -> None:
    if ZABBIX_API_TOKEN:
        return
    try:
        _api("user.logout", {})
    except Exception:
        pass


# ── Node + item discovery ─────────────────────────────────────────────────────

def _discover_active_nodes() -> list[dict]:
    """Return active nodes (last seen < 24 h) that have the anchor CPU item."""
    anchor_items = _api("item.get", {
        "search"                : {"key_": _ANCHOR_KEY},
        "searchWildcardsEnabled": False,
        "filter"                : {"status": 0},
        "output"                : ["itemid", "hostid", "lastclock"],
    })
    if not anchor_items:
        raise RuntimeError(f"No hosts found with key_ containing '{_ANCHOR_KEY}'.")

    hostids = list({it["hostid"] for it in anchor_items})
    hosts   = _api("host.get", {
        "hostids": hostids,
        "filter" : {"status": 0},
        "output" : ["hostid", "host"],
    })

    lc_map = {it["hostid"]: int(it.get("lastclock", 0)) for it in anchor_items}
    now    = time.time()
    active = [h for h in hosts if (now - lc_map.get(h["hostid"], 0)) / 3600 < 24]
    active.sort(key=lambda h: h["host"])
    log.info("Active nodes (%d): %s", len(active), [h["host"] for h in active])
    return active


def _resolve_items(nodes: list[dict]) -> dict:
    """Fetch all active items for all nodes; return per-node item resolution.

    Returns
    -------
    {hostid: {
        "cpu":  {"itemid": str, "value_type": int} | None,
        "mem":  ...,
        "disk": ...,
        "load": ...,
        "ncores_value": int,   # from system.cpu.num lastvalue; default _DEFAULT_NCORES
    }}
    """
    all_items = _api("item.get", {
        "hostids": [h["hostid"] for h in nodes],
        "filter" : {"status": 0},
        "output" : ["itemid", "hostid", "key_", "value_type", "lastclock", "lastvalue"],
    })

    by_host: dict[str, list[dict]] = defaultdict(list)
    for it in all_items:
        by_host[it["hostid"]].append(it)

    def _pick(items: list[dict], key_pattern: str) -> dict | None:
        matches = [i for i in items if key_pattern in i["key_"]]
        if not matches:
            return None
        return max(matches, key=lambda i: int(i.get("lastclock") or 0))

    resolved: dict[str, dict] = {}
    for node in nodes:
        hid   = node["hostid"]
        items = by_host.get(hid, [])

        # For disk: there may be multiple devices; pick the most active.
        disk_item = _pick(items, _KEY_DISK_UTIL)

        # n_cores: read from lastvalue (static metric — history not needed).
        ncores_item = _pick(items, _KEY_NCORES)
        try:
            ncores = int(float(ncores_item["lastvalue"])) if ncores_item else _DEFAULT_NCORES
            ncores = max(1, ncores)
        except (ValueError, TypeError):
            ncores = _DEFAULT_NCORES

        cpu_item  = _pick(items, _KEY_CPU_UTIL)
        mem_item  = _pick(items, _KEY_MEM_UTIL)
        load_item = _pick(items, _KEY_LOAD_AVG)

        for label, item in (("cpu", cpu_item), ("load", load_item)):
            if item is None:
                log.warning("Node %s: required item '%s' not found",
                            node["host"], _KEY_CPU_UTIL if label == "cpu" else _KEY_LOAD_AVG)

        resolved[hid] = {
            "cpu"          : cpu_item,
            "mem"          : mem_item,
            "disk"         : disk_item,
            "load"         : load_item,
            "ncores_value" : ncores,
        }
        log.info(
            "  %-28s  cpu=%s  mem=%s  disk=%s  load=%s  ncores=%d",
            node["host"],
            "✓" if cpu_item  else "✗",
            "✓" if mem_item  else "✗",
            "✓" if disk_item else "✗",
            "✓" if load_item else "✗",
            ncores,
        )

    return resolved


# ── History fetch ─────────────────────────────────────────────────────────────

def _fetch_history_window(
    item_ids:   list[str],
    value_type: int,
    time_from:  int,
    time_till:  int,
) -> list[dict]:
    """Fetch all history records for a set of items within a time window.

    Uses a generous limit (50000) to avoid silent server-side truncation.
    Logs a warning if the returned count hits the limit (possible truncation).
    """
    if not item_ids:
        return []
    records = _api("history.get", {
        "itemids"  : item_ids,
        "history"  : value_type,
        "time_from": time_from,
        "time_till": time_till,
        "output"   : ["itemid", "clock", "value"],
        "sortfield": "clock",
        "sortorder": "ASC",
        "limit"    : 50000,
    })
    if len(records) >= 50000:
        log.warning(
            "history.get returned 50000 records (limit hit) for window %s–%s — "
            "consider reducing _CHUNK_DAYS",
            datetime.fromtimestamp(time_from, tz=timezone.utc).strftime("%Y-%m-%d"),
            datetime.fromtimestamp(time_till, tz=timezone.utc).strftime("%Y-%m-%d"),
        )
    return records


# ── Aggregation ───────────────────────────────────────────────────────────────

def _build_agg_rows(
    history_records: list[dict],
    itemid_to_mid:   dict[str, int],
    itemid_to_type:  dict[str, str],
    ncores_by_mid:   dict[int, int],
) -> list[dict]:
    """Aggregate raw history records into cluster_agg rows (one per machine × bucket).

    Multiple Zabbix readings that happen to fall in the same 5-min bucket
    are averaged (rare but possible after Zabbix agent restarts).

    Buckets with no CPU data are dropped (no CPU reading = no output row).
    Missing mem / disk / load for a given bucket are filled with 0 / 0 / 1.
    """
    # Accumulate: (machine_id, bucket) → {item_type: [float, ...]}
    accum: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
        lambda: {"cpu": [], "mem": [], "disk": [], "load": []}
    )

    for rec in history_records:
        iid = rec["itemid"]
        mid = itemid_to_mid.get(iid)
        if mid is None:
            continue
        itype = itemid_to_type.get(iid)
        if itype is None:
            continue
        try:
            val    = float(rec["value"])
            bucket = int(rec["clock"]) // 300
        except (ValueError, KeyError):
            continue
        accum[(mid, bucket)][itype].append(val)

    rows: list[dict] = []
    for (mid, bucket), vals in accum.items():
        if not vals["cpu"]:
            continue  # no CPU reading — skip bucket

        cpu_pct  = float(np.mean(vals["cpu"]))
        mem_pct  = float(np.mean(vals["mem"])) if vals["mem"] else 0.0
        disk_pct = float(np.mean(vals["disk"])) if vals["disk"] else 0.0
        load_avg = float(np.mean(vals["load"])) if vals["load"] else 1.0

        n_cores  = ncores_by_mid.get(mid, _DEFAULT_NCORES)
        # Unit conversions (see module docstring)
        total_cpu = np.float32(np.clip(cpu_pct / 100.0 * n_cores, 0.0, float(n_cores)))
        total_mem = np.float32(np.clip(mem_pct  / 100.0, 0.0, 1.0))
        disk_io   = np.float32(np.clip(disk_pct / 100.0, 0.0, 1.0))
        n_tasks   = int(max(1, round(load_avg)))

        rows.append({
            "machine_id": int(mid),
            "bucket"    : int(bucket),
            "time_us"   : int(bucket) * 300_000_000,
            "total_cpu" : total_cpu,
            "peak_cpu"  : total_cpu,   # no intra-bucket peak available
            "total_mem" : total_mem,
            "peak_mem"  : total_mem,
            "disk_io"   : disk_io,
            "n_tasks"   : np.int32(n_tasks),
        })

    return rows


# ── Main pipeline ─────────────────────────────────────────────────────────────

def fetch(history_days: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover nodes + items ────────────────────────────────────────────────
    log.info("═" * 62)
    log.info("  Step 1 — Discover active nodes")
    log.info("═" * 62)
    nodes    = _discover_active_nodes()
    resolved = _resolve_items(nodes)

    # Assign stable machine_ids (alphabetical by hostname)
    hostname_to_mid: dict[str, int] = {
        h["host"]: i for i, h in enumerate(nodes)
    }
    hostid_to_mid: dict[str, int] = {
        h["hostid"]: hostname_to_mid[h["host"]] for h in nodes
    }

    # Build item → machine_id and item → type maps
    itemid_to_mid:   dict[str, int] = {}
    itemid_to_type:  dict[str, str] = {}
    ncores_by_mid:   dict[int, int] = {}

    all_float_item_ids: list[str] = []

    for node in nodes:
        hid  = node["hostid"]
        mid  = hostid_to_mid[hid]
        info = resolved[hid]
        ncores_by_mid[mid] = info["ncores_value"]

        for itype, item in (
            ("cpu",  info["cpu"]),
            ("mem",  info["mem"]),
            ("disk", info["disk"]),
            ("load", info["load"]),
        ):
            if item is None:
                continue
            iid = item["itemid"]
            vtype = int(item.get("value_type", 0))
            itemid_to_mid[iid]  = mid
            itemid_to_type[iid] = itype
            if vtype == 0:  # numeric float — only type we fetch in bulk
                all_float_item_ids.append(iid)
            else:
                log.warning(
                    "Node %s: item %s has value_type=%d (expected 0=float) — skipped",
                    node["host"], item["key_"], vtype,
                )

    log.info("Total float items to fetch: %d  (%d nodes)", len(all_float_item_ids), len(nodes))

    # ── Fetch history in daily chunks ─────────────────────────────────────────
    log.info("═" * 62)
    log.info("  Step 2 — Fetch %d days of history (%d-day chunks)", history_days, _CHUNK_DAYS)
    log.info("═" * 62)

    t_now   = int(time.time())
    t_start = t_now - history_days * 86400
    chunk   = _CHUNK_DAYS * 86400

    chunk_dfs: list[pd.DataFrame] = []
    n_days_done  = 0
    n_rows_total = 0

    for t_from in range(t_start, t_now, chunk):
        t_till = min(t_from + chunk, t_now)
        day_str = datetime.fromtimestamp(t_from, tz=timezone.utc).strftime("%Y-%m-%d")

        records = _fetch_history_window(all_float_item_ids, 0, t_from, t_till)
        rows    = _build_agg_rows(records, itemid_to_mid, itemid_to_type, ncores_by_mid)
        if rows:
            chunk_dfs.append(pd.DataFrame(rows))
        n_rows_total += len(rows)

        n_days_done += 1
        if n_days_done % 10 == 0 or n_days_done == 1:
            log.info("  Day %s  |  raw records: %5d  |  agg rows: %5d  |  total rows so far: %d",
                     day_str, len(records), len(rows), n_rows_total)
        time.sleep(_RATE_LIMIT)

    log.info("History fetch complete — %d aggregated rows", n_rows_total)

    # ── Build DataFrame + validate ────────────────────────────────────────────
    log.info("═" * 62)
    log.info("  Step 3 — Build and validate DataFrame")
    log.info("═" * 62)

    if not chunk_dfs:
        log.error("No rows collected — check Zabbix connectivity and item keys.")
        sys.exit(1)

    df = pd.concat(chunk_dfs, ignore_index=True)

    n_before = len(df)
    if df.duplicated(["machine_id", "bucket"]).any():
        df = (
            df.groupby(["machine_id", "bucket"], sort=True)
            .agg({
                "time_us"  : "first",
                "total_cpu": "mean",
                "peak_cpu" : "mean",
                "total_mem": "mean",
                "peak_mem" : "mean",
                "disk_io"  : "mean",
                "n_tasks"  : "mean",
            })
            .reset_index()
        )
        log.info("  Deduplicated %d duplicate bucket rows", n_before - len(df))

    # Cast to cluster_agg schema dtypes
    df["machine_id"] = df["machine_id"].astype("int64")
    df["bucket"]     = df["bucket"].astype("int64")
    df["time_us"]    = df["time_us"].astype("int64")
    df["total_cpu"]  = df["total_cpu"].astype("float32")
    df["peak_cpu"]   = df["peak_cpu"].astype("float32")
    df["total_mem"]  = df["total_mem"].astype("float32")
    df["peak_mem"]   = df["peak_mem"].astype("float32")
    df["disk_io"]    = df["disk_io"].astype("float32")
    df["n_tasks"]    = df["n_tasks"].round().astype("int32").clip(lower=1)

    df = df.sort_values(["machine_id", "bucket"]).reset_index(drop=True)

    n_machines = df["machine_id"].nunique()
    n_buckets  = df["bucket"].nunique()
    log.info("  Rows: %d  |  Machines: %d  |  Unique buckets: %d",
             len(df), n_machines, n_buckets)
    log.info("  CPU range  : [%.4f, %.4f]  (expected ~0–n_cores)",
             df["total_cpu"].min(), df["total_cpu"].max())
    log.info("  Mem range  : [%.4f, %.4f]  (expected 0–1)",
             df["total_mem"].min(), df["total_mem"].max())
    log.info("  Disk range : [%.4f, %.4f]  (expected 0–1)",
             df["disk_io"].min(), df["disk_io"].max())
    log.info("  Tasks range: [%d, %d]",
             int(df["n_tasks"].min()), int(df["n_tasks"].max()))

    # ── Write outputs ─────────────────────────────────────────────────────────
    log.info("═" * 62)
    log.info("  Step 4 — Write outputs")
    log.info("═" * 62)

    parquet_path  = output_dir / "zabbix_agg.parquet"
    node_map_path = output_dir / "node_map.json"

    df.to_parquet(parquet_path, engine="pyarrow", compression="zstd", index=False)
    log.info("  Parquet written: %s  (%.1f MB)", parquet_path,
             parquet_path.stat().st_size / 1e6)

    node_map = {h["host"]: hostname_to_mid[h["host"]] for h in nodes}
    node_map_path.write_text(json.dumps(node_map, indent=2, sort_keys=True))
    log.info("  Node map written: %s", node_map_path)
    log.info("  Node mapping: %s", node_map)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Zabbix history → cluster_agg parquet.")
    p.add_argument(
        "--days",
        type    = int,
        default = _DEFAULT_HISTORY_DAYS,
        metavar = "N",
        help    = f"Days of history to pull (default: {_DEFAULT_HISTORY_DAYS}).",
    )
    p.add_argument(
        "--output",
        type    = Path,
        default = _DEFAULT_OUTPUT_DIR,
        metavar = "DIR",
        help    = f"Output directory (default: {_DEFAULT_OUTPUT_DIR}).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not ZABBIX_API_TOKEN and not ZABBIX_PASS:
        log.error(
            "No credentials set. Fill in ZABBIX_API_TOKEN or ZABBIX_USER+ZABBIX_PASS "
            "in the script before running."
        )
        sys.exit(1)

    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║  ZABBIX DATA FETCHER                                         ║")
    log.info("║  %s                                           ║",
             datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    log.info("╚══════════════════════════════════════════════════════════════╝")
    log.info("URL     : %s", ZABBIX_URL)
    log.info("Days    : %d", args.days)
    log.info("Output  : %s", args.output)

    _login()
    try:
        fetch(history_days=args.days, output_dir=args.output)
    finally:
        _logout()
        log.info("Done.")


if __name__ == "__main__":
    main()
