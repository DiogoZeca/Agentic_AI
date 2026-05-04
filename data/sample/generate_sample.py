"""generate_sample.py — Deterministic synthetic cluster_agg sample for pipeline testing.

Produces cluster_agg_sample.csv: 20 machines × 300 five-minute buckets = 6,000 rows.

Purpose
-------
Verify that the full inference and evaluation pipeline (predict.py,
evaluate_domain.py) runs end-to-end before pointing it at real data.  The
synthetic CPU patterns are designed to produce non-degenerate spike labels
across all five model horizons (15m / 30m / 45m / 60m / OVR severe).

Machine groups (20 total)
--------------------------
  4 × quiet   — low baseline (~0.07), small spikes (~0.30 above base).
                 Most spikes are moderate; very few severe.
  8 × normal  — medium baseline (~0.22), medium spikes (~0.45 above base).
                 Mix of moderate and severe spikes.
  4 × bursty  — low baseline (~0.06), large spikes (~0.80 above base).
                 Low p95/p99 means almost every spike is classified as severe
                 → provides OVR model label coverage.
  4 × heavy   — high baseline (~0.50), medium spikes.
                 Represents continuously loaded machines.

Why 300 buckets?
----------------
300 = just above _MIN_BUCKETS_WARN (288) in evaluate_domain.py.  No data-quality
warning is fired.  The chronological 60/20/20 split gives:
  • Train:  0–179  (180 buckets) — p95/p99 bootstrapped from here
  • Val:   180–239  (60 buckets × 20 machines = 1,200 rows)
  • Test:  240–299  (60 buckets × 20 machines = 1,200 rows, minus 12-bucket
                    label horizon NaN tail = ~960 labeled test rows)

Why START_BUCKET = 5_785_632?
------------------------------
Corresponds to ~2026-01-01 00:00 UTC (1_735_689_600 seconds // 300).
time_us = bucket × 300_000_000, which is what the schema requires.

IMPORTANT: Machine IDs in this sample (1–20) do NOT exist in
spike_thresholds.parquet (which was computed from Google Cluster 2011 data).
When running predict.py or daemon.py on this sample, predict.py will warn:
  "ALL 20 machines are unknown to spike_thresholds.parquet"
This is expected — see the deployment workflow in docs/ops_runbook.md.
Run spike/bootstrap_thresholds.py on your real domain data first.

Regenerating
------------
python data/sample/generate_sample.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

# Change this seed only when intentionally changing the generated data.
_SEED = 42

N_MACHINES  = 20
N_BUCKETS   = 300

# 2026-01-01 00:00:00 UTC = 1_735_689_600 s; // 300 = 5_785_632
START_BUCKET = 5_785_632

# (count, base_cpu, spike_prob_per_bucket, spike_height, label)
# spike_prob is probability a NEW spike starts at each bucket.
# Each spike lasts 2–4 consecutive buckets.
# Effective spike fraction ≈ spike_prob × avg_duration(=3).
_GROUPS: list[tuple[int, float, float, float, str]] = [
    (4, 0.07, 0.025, 0.30, "quiet"),   # ~7.5% spike time; p95 ≈ 0.10
    (8, 0.22, 0.040, 0.45, "normal"),  # ~12%  spike time; p95 ≈ 0.30
    (4, 0.06, 0.050, 0.80, "bursty"),  # ~15%  spike time; p95 ≈ 0.09 → spikes far above p99
    (4, 0.50, 0.035, 0.35, "heavy"),   # ~10%  spike time; p95 ≈ 0.60
]


# ── CPU generation ────────────────────────────────────────────────────────────

def _generate_cpu(
    rng:        np.random.Generator,
    base:       float,
    spike_prob: float,
    spike_h:    float,
) -> np.ndarray:
    """Return total_cpu time series for one machine.

    CPU at each bucket = base + Gaussian noise + spike contribution.
    Spikes start with probability spike_prob at each bucket and last 2–4
    consecutive buckets at height spike_h × U[0.7, 1.0] above base.

    The step-through approach (incrementing b by spike duration when spiking)
    produces realistic spike clusters — consecutive buckets tend to be
    correlated, matching real workload bursts.  A pure Bernoulli draw at each
    bucket would produce independent spikes with no autocorrelation.
    """
    cpu = base + rng.normal(0.0, 0.018, N_BUCKETS)
    b   = 0
    while b < N_BUCKETS:
        if rng.random() < spike_prob:
            duration  = int(rng.integers(2, 5))           # 2, 3 or 4 buckets
            magnitude = spike_h * float(rng.uniform(0.70, 1.0))
            end       = min(b + duration, N_BUCKETS)
            cpu[b:end] += magnitude
            b = end
        else:
            b += 1
    return np.clip(cpu, 0.0, 1.0).astype("float32")


# ── Row generation ────────────────────────────────────────────────────────────

def generate(output_path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(_SEED)

    buckets  = np.arange(START_BUCKET, START_BUCKET + N_BUCKETS, dtype="int64")
    time_us  = buckets * 300_000_000

    rows: list[dict] = []
    machine_id = 1

    for count, base, spike_prob, spike_h, _ in _GROUPS:
        for _ in range(count):
            total_cpu = _generate_cpu(rng, base, spike_prob, spike_h)

            # peak_cpu: within-bucket burst, always ≥ total_cpu
            peak_cpu = np.minimum(
                total_cpu + rng.uniform(0.0, 0.08, N_BUCKETS).astype("float32"),
                1.0,
            ).astype("float32")

            # Memory: weakly correlated with CPU load
            total_mem = np.clip(
                0.30 + 0.35 * total_cpu + rng.normal(0.0, 0.04, N_BUCKETS),
                0.0, 1.0,
            ).astype("float32")
            peak_mem  = np.minimum(
                total_mem + rng.uniform(0.0, 0.04, N_BUCKETS).astype("float32"),
                1.0,
            ).astype("float32")

            # Disk IO: sparse, elevated during high-CPU periods
            disk_io = np.clip(
                rng.exponential(0.04, N_BUCKETS).astype("float32") * (total_cpu > 0.25),
                0.0, 1.0,
            ).astype("float32")

            # n_tasks: Poisson count, scales with CPU
            n_tasks = rng.poisson(2.0 + 6.0 * total_cpu).astype("int32")

            for i in range(N_BUCKETS):
                rows.append({
                    "machine_id": machine_id,
                    "bucket"    : int(buckets[i]),
                    "time_us"   : int(time_us[i]),
                    "total_cpu" : round(float(total_cpu[i]), 6),
                    "peak_cpu"  : round(float(peak_cpu[i]),  6),
                    "total_mem" : round(float(total_mem[i]), 6),
                    "peak_mem"  : round(float(peak_mem[i]),  6),
                    "disk_io"   : round(float(disk_io[i]),   6),
                    "n_tasks"   : int(n_tasks[i]),
                })

            machine_id += 1

    df = pd.DataFrame(rows)

    # Enforce dtypes so the CSV round-trips cleanly through _coerce_input()
    df["machine_id"] = df["machine_id"].astype("int64")
    df["bucket"]     = df["bucket"].astype("int64")
    df["time_us"]    = df["time_us"].astype("int64")
    df["n_tasks"]    = df["n_tasks"].astype("int32")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out = Path(__file__).parent / "cluster_agg_sample.csv"
    df  = generate(out)

    n_machines = df["machine_id"].nunique()
    n_buckets  = df["bucket"].nunique()
    size_kb    = out.stat().st_size // 1024

    print(f"Written : {out}")
    print(f"Rows    : {len(df):,}  ({n_machines} machines × {n_buckets} buckets)")
    print(f"Size    : {size_kb} KB")
    print(f"Columns : {list(df.columns)}")
    print(f"CPU range : [{df['total_cpu'].min():.3f}, {df['total_cpu'].max():.3f}]")
    print(f"n_tasks range: [{df['n_tasks'].min()}, {df['n_tasks'].max()}]")
    print()
    print("Next steps:")
    print("  1. Run bootstrap_thresholds.py on your REAL data (not this sample).")
    print("  2. python -m spike.predict --input data/sample/cluster_agg_sample.csv \\")
    print("         --model-dir data/full_run/spike --output data/sample/predictions.json")
    print("  3. python evaluation/evaluate_domain.py \\")
    print("         --input data/sample/cluster_agg_sample.csv \\")
    print("         --model-dir data/full_run/spike --output data/sample/eval")

    sys.exit(0)
