#!/usr/bin/env python3
"""
explore_zabbix.py  —  Zabbix environment diagnostic for ML pipeline planning.

Answers five questions before writing the real data extractor:
  Q1 — How many compute nodes exist, and which are active?
  Q2 — What is the actual (measured) collection interval?
  Q3 — How far back does history go per node?
  Q4 — Are key item schemas identical across all active nodes?
  Q5 — Do qemu/* VMs share the compute cluster (have Slurm items)?

Nothing is written to disk. Run time: ~1–3 minutes depending on node count.

Usage (from AIModel/):
  .venv/bin/python3 data/explore_zabbix.py

Dependencies: requests  (already in the ML venv via requirements-inference.txt)
"""

from __future__ import annotations

import statistics
import sys
import time
from datetime import datetime
from typing import Any

import requests
import urllib3

# ══════════════════════════════════════════════════════════════════════════════
#  CREDENTIALS  —  fill in before running, do not commit to git
#
#  Option A (recommended for Zabbix 7.2+): API token
#    1. Log into Zabbix web UI → Users → API tokens → Create API token
#    2. Paste the generated token below as ZABBIX_API_TOKEN
#    3. Leave ZABBIX_USER / ZABBIX_PASS empty
#
#  Option B: username + password (Zabbix < 7.2 or if API tokens unavailable)
#    1. Leave ZABBIX_API_TOKEN empty
#    2. Fill in ZABBIX_USER and ZABBIX_PASS
#
#  URL tip: if you get "Page not found", try adding or removing "/zabbix/" prefix
#    e.g.  https://YOUR_HOST/zabbix/api_jsonrpc.php
#    or    https://YOUR_HOST/api_jsonrpc.php
# ══════════════════════════════════════════════════════════════════════════════

ZABBIX_URL       = "http://atnog-mon.av.it.pt/zabbix/api_jsonrpc.php"
ZABBIX_API_TOKEN = ""          # preferred: paste API token from Users → API tokens
ZABBIX_USER      = "dsgps"          # fallback: username (leave empty if using token)
ZABBIX_PASS      = ""          # fallback: password  (leave empty if using token)
VERIFY_SSL       = False       # set True if your server has a valid certificate

# ══════════════════════════════════════════════════════════════════════════════

# ─── Discovery parameters ─────────────────────────────────────────────────────

# Substring of the item key_ used to identify compute nodes.
# Confirmed from the Problems page: janus-nodes triggered on system.cpu.load.
ANCHOR_KEY = "system.cpu.load"

# Activity thresholds
ACTIVE_HOURS = 24    # last-seen < 24 h  → active
STALE_HOURS  = 168   # last-seen < 7 d   → stale (beyond = dead)

# Interval measurement: sample this many nodes × this many consecutive records.
INTERVAL_NODES   = 3
INTERVAL_RECORDS = 60

# Metrics we expect on every compute node.
# Key: human label.  Value: substring to match against the item's key_ field.
EXPECTED_ITEMS: dict[str, str] = {
    "CPU util"    : "system.cpu.util",
    "CPU load"    : "system.cpu.load",
    "Memory"      : "vm.memory",
    "Disk util"   : "vfs.dev.util",
    "Disk I/O"    : "vfs.dev.read",
    "Slurm state" : "slurm",
    "GPU"         : "gpu",
    "Interrupts"  : "system.cpu.intr",
    "Ctx switches": "system.cpu.switches",
}

# ─── HTTP / API layer ─────────────────────────────────────────────────────────

if not VERIFY_SSL:
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_sess = requests.Session()
_sess.verify = VERIFY_SSL
_sess.headers["Content-Type"] = "application/json"

_token: str | None = None        # populated after _login() or set from ZABBIX_API_TOKEN
_using_api_token: bool = False   # True → skip user.logout (tokens are permanent)


def _api(method: str, params: dict) -> Any:
    """Send one JSON-RPC request; raise on HTTP or API-level errors.

    Zabbix 7.2+ requires the auth token in the Authorization header (Bearer),
    not in the JSON body. Both styles are handled here automatically.
    """
    body: dict = {"jsonrpc": "2.0", "method": method, "params": params, "id": 1}
    headers: dict = {}
    # apiinfo.version is a public method — Zabbix 7.x rejects an auth header on it.
    if _token and method != "apiinfo.version":
        # Zabbix 7.2+: token goes in header. Older versions accept body["auth"];
        # sending it in the header also works on 6.x, so always use header style.
        headers["Authorization"] = f"Bearer {_token}"
    r = _sess.post(ZABBIX_URL, json=body, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        err = data["error"]
        raise RuntimeError(
            f"[{method}] {err.get('data') or err.get('message') or str(err)}"
        )
    return data["result"]


def _login() -> None:
    global _token, _using_api_token
    if ZABBIX_API_TOKEN:
        # Option A: pre-generated API token — no login call needed.
        _token = ZABBIX_API_TOKEN
        _using_api_token = True
        print("  Auth: using API token (no user.login needed)")
    else:
        # Option B: username + password → obtain a session token.
        # Zabbix 5.4+ uses "username"; older versions use "user".
        _token = _api("user.login", {"username": ZABBIX_USER, "password": ZABBIX_PASS})
        _using_api_token = False
        print("  Auth: session token obtained via user.login")


def _logout() -> None:
    if _using_api_token:
        return   # API tokens are permanent; no logout call
    try:
        _api("user.logout", {})
    except Exception:
        pass   # best-effort; never crash on cleanup

# ─── Formatting helpers ───────────────────────────────────────────────────────

def _dt(clock: int | str | None) -> str:
    if not clock or int(clock) == 0:
        return "—"
    return datetime.fromtimestamp(int(clock)).strftime("%Y-%m-%d %H:%M")


def _age(clock: int | str | None) -> str:
    if not clock or int(clock) == 0:
        return "never"
    secs = time.time() - int(clock)
    if secs < 3600:
        return f"{int(secs / 60)}m ago"
    if secs < 86400:
        h = int(secs / 3600)
        m = int((secs % 3600) / 60)
        return f"{h}h {m}m ago"
    return f"{int(secs / 86400)}d ago"


def _gap(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m}m{s:02d}s"


def _hdr(title: str) -> None:
    bar = "═" * 66
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)

# ─── Q1: Compute node discovery ───────────────────────────────────────────────

def _find_compute_hosts() -> list[dict]:
    """Return all Zabbix hosts that have at least one item matching ANCHOR_KEY."""
    anchor_items = _api("item.get", {
        "search"                : {"key_": ANCHOR_KEY},
        "searchWildcardsEnabled": False,      # substring match (no wildcards needed)
        "filter"                : {"status": 0},
        "output"                : ["itemid", "hostid", "key_", "delay",
                                   "lastclock", "value_type"],
    })
    if not anchor_items:
        return []

    hostids = list({it["hostid"] for it in anchor_items})
    hosts   = _api("host.get", {
        "hostids": hostids,
        "filter" : {"status": 0},             # monitoring must be enabled
        "output" : ["hostid", "host", "name"],
    })

    # Attach the anchor item's lastclock so we can classify nodes without
    # a separate API call.
    lc_map = {it["hostid"]: int(it["lastclock"]) for it in anchor_items}
    ai_map = {it["hostid"]: it for it in anchor_items}
    for h in hosts:
        h["_lc"]    = lc_map.get(h["hostid"], 0)
        h["_anchor"] = ai_map.get(h["hostid"])

    return sorted(hosts, key=lambda x: x["host"])


def _classify(hosts: list[dict]) -> tuple[list, list, list]:
    now = time.time()
    active, stale, dead = [], [], []
    for h in hosts:
        age_h = (now - h["_lc"]) / 3600
        if   age_h <= ACTIVE_HOURS: active.append(h)
        elif age_h <= STALE_HOURS:  stale.append(h)
        else:                        dead.append(h)
    return active, stale, dead

# ─── Full schema dump (one sample node) ──────────────────────────────────────

def _dump_schema(hostid: str) -> list[dict]:
    """Print every active item on a host and return the list."""
    items = _api("item.get", {
        "hostids"  : hostid,
        "filter"   : {"status": 0},
        "output"   : ["itemid", "name", "key_", "delay",
                      "value_type", "units", "lastclock"],
        "sortfield": "key_",
    })
    print(f"\n  {'KEY_':<52} {'NAME':<35} {'DELAY':>6}  {'UNITS':>8}  LAST CHECK")
    print(f"  {'-'*52} {'-'*35} {'-'*6}  {'-'*8}  ----------")
    for it in items:
        print(
            f"  {it['key_']:<52} {it['name']:<35} "
            f"{it['delay']:>6}  {(it['units'] or '—'):>8}  {_age(it['lastclock'])}"
        )
    return items

# ─── Q4: Schema comparison across nodes ──────────────────────────────────────

def _compare_schemas(active: list[dict]) -> None:
    """Fetch all items for all active nodes in one call, then compare locally."""
    all_items = _api("item.get", {
        "hostids": [h["hostid"] for h in active],
        "filter" : {"status": 0},
        "output" : ["hostid", "key_"],
    })

    # Concatenate every key_ per host into one searchable blob (lower-cased).
    blob: dict[str, str] = {}
    for it in all_items:
        blob[it["hostid"]] = blob.get(it["hostid"], "") + " " + it["key_"].lower()

    col    = 14
    labels = list(EXPECTED_ITEMS.keys())
    header = f"  {'NODE':<22}" + "".join(f" {lb[:col - 1]:<{col}}" for lb in labels)
    print(header)
    print("  " + "-" * (22 + col * len(labels)))
    for h in active:
        keys = blob.get(h["hostid"], "")
        row  = f"  {h['host']:<22}"
        for lb in labels:
            found = EXPECTED_ITEMS[lb].lower() in keys
            row  += f" {'✓' if found else '✗':<{col}}"
        print(row)

# ─── Q3: History retention ────────────────────────────────────────────────────

def _check_retention(active: list[dict]) -> None:
    """For each active node: oldest history record + oldest trend record."""
    anchor_items = _api("item.get", {
        "hostids"  : [h["hostid"] for h in active],
        "search"   : {"key_": ANCHOR_KEY},
        "filter"   : {"status": 0},
        "output"   : ["itemid", "hostid", "value_type"],
    })
    item_map = {it["hostid"]: it for it in anchor_items}

    print(f"  {'NODE':<22}  {'OLDEST HISTORY':<20} {'DAYS':>5}  "
          f"{'OLDEST TREND':<20} TRENDS?")
    print(f"  {'-'*22}  {'-'*20} {'-'*5}  {'-'*20} -------")

    min_days = float("inf")
    for h in active:
        item = item_map.get(h["hostid"])
        if not item:
            print(f"  {h['host']:<22}  {'anchor item missing':20} {'—':>5}  {'—':20} —")
            continue

        iid = item["itemid"]
        vt  = int(item["value_type"])

        oldest_hist = _api("history.get", {
            "itemids"  : iid, "history"  : vt,
            "sortfield": "clock", "sortorder": "ASC",
            "limit"    : 1,      "output"   : ["clock"],
        })
        # trend.get can be slow without time filters; limit:1 ASC gives the oldest.
        oldest_trend = _api("trend.get", {
            "itemids"  : iid,
            "sortfield": "clock", "sortorder": "ASC",
            "limit"    : 1,      "output"   : ["clock"],
        })

        oh   = int(oldest_hist[0]["clock"])  if oldest_hist  else None
        ot   = int(oldest_trend[0]["clock"]) if oldest_trend else None
        days = int((time.time() - oh) / 86400) if oh else 0
        if days:
            min_days = min(min_days, days)

        print(
            f"  {h['host']:<22}  {_dt(oh):20} {days:>5}  "
            f"{_dt(ot):20} {'Yes' if ot else 'No'}"
        )

    if min_days < float("inf"):
        rows_est = len(active) * int(min_days) * 288  # 288 five-min buckets/day
        print(
            f"\n  Shortest history window : {int(min_days)} days"
            f"\n  Training rows estimate  : ~{rows_est:,}  "
            f"({len(active)} nodes × {int(min_days)}d × 288 buckets/day)"
            f"\n  [Compare: Google Cluster 2011 = ~24,000,000 rows]"
        )

# ─── Q2: Actual collection interval ──────────────────────────────────────────

def _measure_intervals(active: list[dict]) -> None:
    """Pull INTERVAL_RECORDS consecutive records per sample node; measure gaps."""
    sample = active[:INTERVAL_NODES]

    anchor_items = _api("item.get", {
        "hostids"  : [h["hostid"] for h in sample],
        "search"   : {"key_": ANCHOR_KEY},
        "filter"   : {"status": 0},
        "output"   : ["itemid", "hostid", "value_type"],
    })
    item_map = {it["hostid"]: it for it in anchor_items}

    print(f"  {'NODE':<22}  {'MEAN':>8} {'STD':>7} {'MIN':>7} {'MAX':>7}  GAPS>10m")
    print(f"  {'-'*22}  {'-'*8} {'-'*7} {'-'*7} {'-'*7}  --------")

    all_means: list[float] = []
    for h in sample:
        item = item_map.get(h["hostid"])
        if not item:
            print(f"  {h['host']:<22}  (anchor item not found)")
            continue

        records = _api("history.get", {
            "itemids"  : item["itemid"],
            "history"  : int(item["value_type"]),
            "sortfield": "clock", "sortorder": "DESC",
            "limit"    : INTERVAL_RECORDS,
            "output"   : ["clock"],
        })
        if len(records) < 2:
            print(f"  {h['host']:<22}  (too few records — {len(records)} found)")
            continue

        clocks    = sorted(int(r["clock"]) for r in records)
        gaps      = [clocks[i + 1] - clocks[i] for i in range(len(clocks) - 1)]
        long_gaps = sum(1 for g in gaps if g > 600)
        mg        = statistics.mean(gaps)
        all_means.append(mg)

        print(
            f"  {h['host']:<22}  {_gap(mg):>8} "
            f"{_gap(statistics.stdev(gaps)):>7} {_gap(min(gaps)):>7} "
            f"{_gap(max(gaps)):>7}  {long_gaps}"
        )

    if all_means:
        overall  = statistics.mean(all_means)
        n_per_bucket = 300 / overall   # samples per 5-min bucket
        print(f"\n  Overall mean interval    : {_gap(overall)}")
        print(f"  Samples per 5-min bucket : ~{n_per_bucket:.1f}")
        if overall > 250:
            print(
                "\n  [!] Interval > ~4 min. Ask the Zabbix admin to set item delay"
                "\n      to '300' (exactly 5 min) for CPU/memory items so bucket"
                "\n      boundaries align cleanly with collection points."
            )

# ─── Q5: qemu/* check ────────────────────────────────────────────────────────

def _check_qemu() -> None:
    """
    Two paths for qemu VMs in Zabbix:
      A — qemu/* as individual named Zabbix hosts (one host per VM).
      B — qemu LLD items under a Proxmox host (all VMs as items on one host).
    We check both.
    """
    # Path A: hosts whose name contains "qemu"
    qemu_hosts = _api("host.get", {
        "search": {"host": "qemu"},
        "output": ["hostid", "host", "name"],
        "limit" : 500,
    })
    print(f"  Path A — qemu/* as Zabbix hosts : {len(qemu_hosts)}")

    slurm_found = False
    if qemu_hosts:
        slurm_items = _api("item.get", {
            "hostids": [h["hostid"] for h in qemu_hosts[:50]],
            "search" : {"key_": "slurm"},
            "filter" : {"status": 0},
            "output" : ["itemid"],
            "limit"  : 1,
        })
        slurm_found = bool(slurm_items)
        label = "Yes → part of compute cluster" if slurm_found else "No → separate infrastructure"
        print(f"  qemu/* hosts with Slurm items   : {label}")

        # Show a sample of the qemu host names so we can see the naming convention
        if qemu_hosts:
            sample_names = [h["host"] for h in qemu_hosts[:8]]
            print(f"  Sample names                    : {', '.join(sample_names)}")

    # Path B: LLD items with "qemu" in the key_ on any host (Proxmox API template)
    lld_items = _api("item.get", {
        "search"                : {"key_": "qemu"},
        "searchWildcardsEnabled": False,
        "filter"                : {"status": 0},
        "output"                : ["hostid", "name", "key_", "lastclock"],
        "selectHosts"           : ["host"],
        "limit"                 : 20,
    })
    print(f"\n  Path B — LLD items with 'qemu' in key_ : {len(lld_items)}")
    if lld_items:
        print(f"\n  Sample LLD qemu items (first 5):")
        for it in lld_items[:5]:
            host_name = it["hosts"][0]["host"] if it.get("hosts") else "?"
            print(f"    [{host_name:<20}]  {it['key_']:<50}  {_age(it['lastclock'])}")

    # Conclusion
    if not qemu_hosts and not lld_items:
        print("\n  Conclusion: No qemu/* data found in this Zabbix instance.")
    elif slurm_found:
        print("\n  Conclusion: qemu/* VMs ARE part of the compute cluster.")
    else:
        print("\n  Conclusion: qemu/* VMs are separate infrastructure (no Slurm).")
        print("  Focus the extractor on the bare-metal compute nodes only.")

# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║          ZABBIX ENVIRONMENT DIAGNOSTIC                         ║")
    print(f"║          {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<54} ║")
    print("╚════════════════════════════════════════════════════════════════╝")

    # ── Auth ────────────────────────────────────────────────────────────────
    try:
        _login()
    except Exception as exc:
        print(f"\n[ERROR] Authentication failed: {exc}")
        print("Check ZABBIX_URL, ZABBIX_USER, ZABBIX_PASS and VERIFY_SSL.")
        sys.exit(1)

    try:
        zabbix_version = _api("apiinfo.version", {})
        print(f"\n  Connected  : {ZABBIX_URL}")
        print(f"  API version: {zabbix_version}")
        print(f"  User       : {ZABBIX_USER}")

        # ── Q1 ──────────────────────────────────────────────────────────────
        _hdr("Q1  COMPUTE NODE DISCOVERY")
        print(f"  Searching for hosts with key_ containing: '{ANCHOR_KEY}'\n")

        all_hosts = _find_compute_hosts()
        if not all_hosts:
            print(f"  [!] No hosts found. Adjust ANCHOR_KEY and rerun.")
            return

        active, stale, dead = _classify(all_hosts)
        print(f"  Total nodes found  : {len(all_hosts)}")
        print(f"  Active  (< {ACTIVE_HOURS}h)    : {len(active)}")
        print(f"  Stale   (< {STALE_HOURS // 24}d)     : {len(stale)}")
        print(f"  Dead    (> {STALE_HOURS // 24}d)     : {len(dead)}")

        print(f"\n  {'NODE':<28}  STATUS    LAST SEEN")
        print(f"  {'-'*28}  --------  ---------")
        for h in active: print(f"  {h['host']:<28}  active    {_age(h['_lc'])}")
        for h in stale:  print(f"  {h['host']:<28}  stale     {_age(h['_lc'])}")
        for h in dead:   print(f"  {h['host']:<28}  dead      {_age(h['_lc'])}")

        if not active:
            print("\n  [!] No active nodes found — cannot continue.")
            return

        # ── Full schema dump (first active node) ────────────────────────────
        sample = active[0]
        _hdr(f"FULL ITEM SCHEMA  —  {sample['host']}  (all active items, sorted by key_)")
        schema_items = _dump_schema(sample["hostid"])
        print(f"\n  Total active items on {sample['host']}: {len(schema_items)}")

        # ── Q4 ──────────────────────────────────────────────────────────────
        _hdr("Q4  METRIC PRESENCE ACROSS ALL ACTIVE NODES")
        print("  ✓ = item found   ✗ = item missing\n")
        _compare_schemas(active)

        # ── Q3 ──────────────────────────────────────────────────────────────
        _hdr(f"Q3  HISTORY RETENTION  (anchor item: {ANCHOR_KEY})")
        _check_retention(active)

        # ── Q2 ──────────────────────────────────────────────────────────────
        _hdr(
            f"Q2  ACTUAL COLLECTION INTERVAL  "
            f"({min(INTERVAL_NODES, len(active))} nodes × {INTERVAL_RECORDS} records each)"
        )
        _measure_intervals(active)

        # ── Q5 ──────────────────────────────────────────────────────────────
        _hdr("Q5  QEMU/* HOST CHECK")
        _check_qemu()

        # ── Next steps ───────────────────────────────────────────────────────
        _hdr("NEXT STEPS")
        print(
            "  1. SCHEMA  — from the full item dump above, note the exact key_\n"
            "     strings for CPU, memory, disk, and Slurm. You need these\n"
            "     in the extractor (exact match, not substring).\n"
            "\n"
            "  2. HISTORY — the shortest window from Q3 caps your training data.\n"
            "     If < 30 days, discuss with the Zabbix admin about retention policy.\n"
            "\n"
            "  3. INTERVAL — if Q2 shows mean > 4 min, ask the admin to set\n"
            "     the CPU/memory item delay to '300' (exactly 5 min) for clean\n"
            "     5-minute bucket alignment.\n"
            "\n"
            "  4. QEMU — if Q5 shows qemu/* are separate infrastructure,\n"
            "     the extractor targets bare-metal nodes only (xbox, sega, etc.).\n"
            "     If they have Slurm items, include them as additional machines.\n"
            "\n"
            "  5. ROW COUNT — if the training estimate is < 500 k rows, consider\n"
            "     reducing bucket size to 1-min or requesting longer history."
        )

    finally:
        _logout()
        print("\n  Auth token released.\n")


if __name__ == "__main__":
    main()
