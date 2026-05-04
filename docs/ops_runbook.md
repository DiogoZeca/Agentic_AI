# Spike Predictor — Operator Runbook

This runbook covers everything a **Type B operator** needs to go from zero to running
the spike predictor on their own dataset. No machine learning background required.

**What you will have at the end:**
- A continuous prediction daemon running on your cluster data every 5 minutes
- Per-machine severity predictions (no spike / moderate / severe) for the next 60 minutes
- A deployment quality report telling you how well the model transfers to your domain

**What this runbook does NOT cover:**
- Retraining the model from scratch (requires GPU and the full training pipeline)
- Drift monitoring (post-deployment operational concern — see README)
- Docker / API deployment (CLI daemon is the recommended starting point)

**Time estimate:** 30–60 minutes end-to-end, depending on your dataset size.

---

## Table of Contents

1. [Prerequisites and Installation](#1-prerequisites-and-installation)
2. [Verify Installation — Smoke Test](#2-verify-installation--smoke-test)
3. [Prepare Your Data](#3-prepare-your-data)
4. [Bootstrap Domain Thresholds](#4-bootstrap-domain-thresholds)
5. [Evaluate on Your Data](#5-evaluate-on-your-data)
6. [Deploy — Run the Prediction Daemon](#6-deploy--run-the-prediction-daemon)
7. [Understand the Prediction Output](#7-understand-the-prediction-output)
8. [Tune the Alarm Threshold](#8-tune-the-alarm-threshold)
9. [Monitor Your Deployment](#9-monitor-your-deployment)
10. [Troubleshooting Reference](#10-troubleshooting-reference)

---

## 1. Prerequisites and Installation

**Requirements:**
- Python 3.10 or newer (`python3 --version`)
- Git
- ~500 MB disk space for model artifacts

```bash
# Clone the repository
git clone https://github.com/DiogoZeca/Agentic_AI.git
cd Agentic_AI

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# Install the spike package and all dependencies
pip install -e .
```

Verify the install completed:

```bash
python -m spike.predict --help
```

You should see the full help output with flags `--input`, `--model-dir`, `--domain-dir`,
`--output`, and `--alarm-threshold`. If you see an import error, check that your venv
is activated and that `pip install -e .` completed without errors.

---

## 2. Verify Installation — Smoke Test

Before touching your own data, confirm the full pipeline works end-to-end using the
synthetic sample included in the repository.

```bash
python -m spike.predict \
    --input      data/sample/cluster_agg_sample.csv \
    --model-dir  data/full_run/spike \
    --output     /tmp/smoke_test.json
```

**Expected output (stderr — progress logs):**

```
14:30:00  INFO     ══════════════════════════════════════════════════════════════
14:30:00  INFO       SPIKE PREDICTOR
14:30:00  INFO       Input      : data/sample/cluster_agg_sample.csv
14:30:00  INFO       Model dir  : data/full_run/spike
14:30:00  INFO     ══════════════════════════════════════════════════════════════
14:30:00  WARNING  ALL 20 machines are unknown to spike_thresholds.parquet —
                   12 threshold-relative features will use global-median fallback
                   (p95=0.330). Run spike/bootstrap_thresholds.py on domain data
                   and pass --domain-dir to fix this.
14:30:01  INFO       Predicted : 20 machines  |  cold-start skipped : 0
14:30:01  INFO       Horizon   : 60 min ahead
14:30:01  INFO       Output    : /tmp/smoke_test.json
```

> **The "unknown machines" warning is expected here.** The sample uses machine IDs
> 1–20 which do not exist in the Google 2011 training thresholds. This will be fixed
> in Step 4 (bootstrap) when you use your own data.

If the command exits with code 0 and writes `/tmp/smoke_test.json`, your installation
is working correctly. Proceed to Step 3.

---

## 3. Prepare Your Data

The model expects your telemetry in **cluster_agg format** — a CSV or Parquet file
where each row represents one machine's aggregated metrics for one 5-minute window.

### Required columns

| Column | Type | Description |
|---|---|---|
| `machine_id` | int64 | Stable integer identifier for the machine. Must not change over time. |
| `bucket` | int64 | 5-minute bucket index: `unix_timestamp_seconds // 300` |
| `time_us` | int64 | Bucket start in microseconds: `bucket × 300_000_000` |
| `total_cpu` | float32 | Average CPU load as a **fraction of one core** [0.0–1.0+] |
| `peak_cpu` | float32 | Peak CPU rate in the 5-minute window (same scale as total_cpu) |
| `total_mem` | float32 | Average memory usage as a **fraction of capacity** [0.0–1.0] |
| `peak_mem` | float32 | Peak memory usage (same scale as total_mem) |
| `disk_io` | float32 | Max mean disk I/O time, normalised [0.0–1.0] |
| `n_tasks` | int32 | Number of concurrent tasks/processes in the bucket |

### Critical pitfalls

These are the most common mistakes. The model will run without errors even if your
data is wrong — but predictions will be meaningless.

**CPU must be a fraction, not a percentage.**
If your monitoring system reports CPU as 0–100%, divide by 100 before writing to CSV.

```
WRONG:  total_cpu = 87.3   # percentage — will trigger a bounds warning
RIGHT:  total_cpu = 0.873  # fraction of one core
```

The predictor will warn you: `"Input bounds warning [total_cpu]: X% of values exceed
10.0 — expected fraction-of-core"`. If you see this warning, your CPU values are wrong.

**Memory and disk must be fractions [0.0–1.0].**
Normalise against your machine's total capacity before writing to CSV.
Warning fires at `total_mem > 1.0`.

**The bucket formula must use seconds, not milliseconds.**

```python
# If your timestamp is Unix seconds:
bucket  = unix_ts_seconds // 300
time_us = bucket * 300_000_000

# If your timestamp is Unix milliseconds:
bucket  = (unix_ts_ms // 1000) // 300
time_us = bucket * 300_000_000

# If your timestamp is Unix microseconds:
bucket  = (unix_ts_us // 1_000_000) // 300
time_us = bucket * 300_000_000
```

**machine_id must be a stable integer.**
String hostnames must be converted to a consistent integer mapping. The same physical
machine must always map to the same integer ID across all time windows.

### Data validation checklist

Before proceeding, verify your prepared CSV:

```bash
python3 - <<'EOF'
import pandas as pd, sys

df = pd.read_csv("your_data.csv")

required = ["machine_id","bucket","time_us","total_cpu","peak_cpu",
            "total_mem","peak_mem","disk_io","n_tasks"]
missing = [c for c in required if c not in df.columns]
if missing:
    print(f"FAIL — missing columns: {missing}"); sys.exit(1)

print(f"Rows      : {len(df):,}")
print(f"Machines  : {df['machine_id'].nunique()}")
print(f"Buckets   : {df['bucket'].nunique()}")
print(f"Days      : {df['bucket'].nunique() * 5 / 60 / 24:.1f}")
print(f"CPU range : [{df['total_cpu'].min():.3f}, {df['total_cpu'].max():.3f}]")
print(f"Mem range : [{df['total_mem'].min():.3f}, {df['total_mem'].max():.3f}]")

if df['total_cpu'].max() > 10:
    print("WARN — CPU values look like percentages (max > 10). Divide by 100.")
if df['total_mem'].max() > 1:
    print("WARN — Memory values exceed 1.0. Normalise against total capacity.")
if df['n_tasks'].min() < 0:
    print("WARN — Negative n_tasks values found.")
print("OK — schema looks valid")
EOF
```

**Minimum data recommended:**
- At least **7 days** of history per machine (2016 five-minute buckets)
- At least **10 machines** for meaningful cluster-level features
- For the evaluation step (Step 5): enough data to split into 60% train / 20% val / 20% test

The model can run with less data, but accuracy estimates will be less reliable and you
will see warnings about insufficient coverage.

---

## 4. Bootstrap Domain Thresholds

This step computes per-machine CPU thresholds (p95 and p99) from **your own data**.
Without it, the model falls back to Google Cluster 2011 thresholds — a completely
different hardware and workload profile. Bootstrapping is mandatory for meaningful
predictions.

```bash
python -m spike.bootstrap_thresholds \
    --input      your_data.csv \
    --domain-dir ./my_domain
```

This creates a `my_domain/` directory with two files:

```
my_domain/
├── spike_thresholds.parquet   # Per-machine p95 and p99 CPU thresholds
└── bootstrap_meta.json        # Audit record of the bootstrap run
```

**Example `bootstrap_meta.json`:**

```json
{
  "bootstrapped_at": "2026-05-04T14:00:00Z",
  "source_path": "/home/user/your_data.csv",
  "min_buckets_per_machine": 2016,
  "n_machines_total": 120,
  "n_machines_with_per_machine": 115,
  "n_machines_with_global_fallback": 5,
  "global_p95": 0.412,
  "global_p99": 0.687
}
```

`n_machines_with_global_fallback: 5` means 5 machines had less than 7 days of data
and received the domain-wide p95/p99 instead of their own. This is normal if some
machines were recently provisioned.

### The `--min-buckets` flag

The default requires **2016 buckets (7 days)** per machine before computing individual
thresholds. If your dataset is shorter, machines fall back to the global domain median.

```bash
# If you only have 3 days of data, lower the threshold:
python -m spike.bootstrap_thresholds \
    --input      your_data.csv \
    --domain-dir ./my_domain \
    --min-buckets 864              # 3 days × 288 buckets/day
```

> **Trade-off:** Lower `--min-buckets` = more machines get individual thresholds,
> but those thresholds are estimated from less data and will be noisier. If you have
> enough data, keep the default.

### Expected warnings (normal)

```
WARNING  5 machine(s) have fewer than 2016 buckets — assigning global fallback
         (p95=0.412, p99=0.687)
WARNING  3 machine(s) have p99 <= p95 — setting p99 = p95 × 1.10
```

The second warning fires for machines whose CPU is nearly constant (e.g. always
near zero) — the 10% gap enforcement prevents degenerate thresholds.

---

## 5. Evaluate on Your Data

This step backtests the model against your historical data using programmatic labels
(CPU > your domain p95 for future windows) and produces a PR-AUC score and a
deployment recommendation.

```bash
python evaluation/evaluate_domain.py \
    --input     your_data.csv \
    --model-dir data/full_run/spike \
    --output    ./eval_results
```

The script:
1. Splits your data chronologically: 60% train → 20% val → 20% test
2. Bootstraps p95/p99 thresholds from the **train split only** (no leakage)
3. Evaluates all five models against your domain labels
4. Finds the best alarm threshold for your domain (F1-max on the val split)
5. Writes `eval_results/domain_eval_report.json`

**Example `domain_eval_report.json` (annotated):**

```jsonc
{
  "run_at": "2026-05-04T14:10:00Z",
  "n_machines": 120,
  "coverage_days": 14.2,        // days of data provided
  "val_rows": 24000,
  "test_rows": 24000,
  "warnings": [],
  "models": {
    "60m": {
      "macro_pr_auc": 0.41,     // your domain score
      // Source (Google 2011) score was 0.547
      // Retention = 0.41 / 0.547 = 75% → DEPLOY
      "macro_roc_auc": 0.81,
      "alarm_threshold": 0.30,  // <-- USE THIS VALUE in Step 6
      "alarm_precision": 0.58,
      "alarm_recall": 0.49,
      "alarm_f1": 0.53,
      "alarm_rate": 0.12,       // 12% of machines flagged per cycle
      "class_rates": {
        "class_0": 0.74,        // 74% no-spike windows
        "class_1": 0.20,        // 20% moderate spike windows
        "class_2": 0.06         // 6% severe spike windows
      }
    },
    "15m": {
      "pr_auc": 0.57,
      "alarm_threshold": 0.78,  // <-- USE THIS VALUE in Step 6
      ...
    },
    "30m": { ... },
    "45m": { ... },
    "severe_ovr": { ... }
  }
}
```

### Reading the deployment recommendation

Retention % = `(your PR-AUC / source PR-AUC) × 100`

| Retention | Recommendation | What to do |
|---|---|---|
| **≥ 70%** | **Deploy** | Model transfers well. Proceed to Step 6 using the `alarm_threshold` from the report. |
| **40–70%** | **Adapt** | Model works but the alarm threshold needs tuning for your domain. See Step 8 before deploying. |
| **< 40%** | **Retrain** | The model does not transfer well to this domain. Retraining from scratch is required. |

Source PR-AUC values used for comparison:

| Model | Source PR-AUC |
|---|---|
| 60m severity | 0.547 |
| 15m binary | 0.575 |
| 30m binary | 0.564 |
| 45m binary | 0.563 |
| OVR severe | 0.543 |

### Edge cases

**"X machines have fewer than 288 buckets" warning:**
The evaluation will still run but PR-AUC estimates are less reliable. Lag and EWMA
features (which look back up to 12 buckets) will be mostly NaN for data-sparse machines.
Recommend at least 2016 buckets per machine for stable estimates.

**Some models missing from the report:**
The script continues with whatever models are found. If `spike_30m` or `spike_45m`
are absent from the report, it means those model directories were not found under
`data/full_run/`.

**`macro_pr_auc: null` for a model:**
Happens when only one class is present in your val or test split (e.g. no severe
spikes in the evaluation period). Collect more data covering diverse load conditions.

---

## 6. Deploy — Run the Prediction Daemon

The daemon reads your data every N seconds, runs all five models, and writes predictions
atomically to a JSON file your scheduler can consume.

### One-shot (test a single prediction cycle)

```bash
python -m spike.predict \
    --input          your_window.csv \
    --model-dir      data/full_run/spike \
    --domain-dir     ./my_domain \
    --alarm-threshold 0.30 \
    --output         predictions.json
```

Replace `0.30` with the `alarm_threshold` value from your `domain_eval_report.json`
(60m model). See [Step 7](#7-understand-the-prediction-output) for how to read the output.

### Continuous daemon — file mode

Your data pipeline writes `cpu_window.csv` every 5 minutes. The daemon reads it each cycle.

```bash
python -m spike.daemon \
    --input          cpu_window.csv \
    --model-dir      data/full_run/spike \
    --domain-dir     ./my_domain \
    --alarm-threshold 0.30 \
    --output         predictions.json \
    --interval       300
```

### Continuous daemon — fetch-command mode

If you have a script that queries your monitoring system, pass it directly. The daemon
runs it each cycle and parses stdout as a cluster_agg CSV.

```bash
python -m spike.daemon \
    --fetch-cmd      "python my_data_collector.py --last-24-buckets" \
    --model-dir      data/full_run/spike \
    --domain-dir     ./my_domain \
    --alarm-threshold 0.30 \
    --output         predictions.json \
    --interval       300
```

Your collector script must print a valid cluster_agg CSV to stdout, including the
header row. It must complete within 60 seconds or the cycle is skipped.

### Running as a background process

```bash
nohup python -m spike.daemon \
    --input          cpu_window.csv \
    --model-dir      data/full_run/spike \
    --domain-dir     ./my_domain \
    --alarm-threshold 0.30 \
    --output         predictions.json \
    --interval       300 \
    > daemon.log 2>&1 &

echo "Daemon PID: $!"
```

Stop cleanly with `kill -TERM <PID>` (SIGTERM). The daemon flushes the current cycle
and exits with code 0.

### Edge cases in daemon operation

**Input file not found (file mode):**
The cycle is skipped with a warning. The daemon does not exit — it waits for the
next interval. This handles the case where your data pipeline is momentarily delayed.

```
WARNING  Input not found: cpu_window.csv (operator pipeline not ready?) — skipping
```

**Fetch-command fails or times out:**
The cycle is skipped with a warning. The daemon continues running.

```
WARNING  --fetch-cmd exited 1 — skipping cycle. stderr: connection refused
WARNING  --fetch-cmd timed out after 60s — skipping cycle
```

**One-shot mode:**
Set `--interval 0` to run exactly one cycle and exit. Useful for cron jobs or testing.

```bash
python -m spike.daemon --input cpu_window.csv --model-dir data/full_run/spike \
    --domain-dir ./my_domain --output predictions.json --interval 0
```

---

## 7. Understand the Prediction Output

Every cycle, the daemon overwrites `predictions.json` with the latest predictions.
The file is written atomically — it is either complete or unchanged, never partially written.

**Full annotated example:**

```jsonc
{
  "batch_id": "a3f2c1d8-9b4e-4f1a-8c2d-7e5f6a3b1c0d",
  "predicted_at": "2026-05-04T14:35:00Z",
  "model_version": "20260101T000000Z_a1b2c3",
  "model_dir": "/home/user/Agentic_AI/data/full_run/spike",
  "horizon_minutes": 60,
  "machines_total": 3,
  "machines_predicted": 2,    // machines that received a probability estimate
  "machines_cold_start": 1,   // machines skipped due to insufficient history
  "predictions": [

    // ── Machine 42: severe spike predicted ───────────────────────────────────
    {
      "machine_id": 42,
      "status": "success",              // full 24-bucket window available
      "observations_used": 24,          // 120 minutes of history

      // 60-minute severity forecast
      "severity_class": 2,             // 0=no_spike, 1=moderate, 2=severe
      "p_no_spike":  0.08,
      "p_moderate":  0.21,
      "p_severe":    0.71,             // probability of a severe spike in the next 60 min
      "p_severe_ovr": 0.83,            // cascade Stage 2 — severe vs. moderate refinement
      "is_spike":  true,               // severity_class >= 1 (argmax-based, always set)
      "is_severe": true,               // p_severe >= alarm_threshold (threshold-gated)
      "alarm_threshold": 0.30,         // threshold used for is_severe

      // Per-machine CPU threshold (from your bootstrap)
      "threshold_used": 0.412,         // p95 CPU threshold for this machine
      "threshold_source": "learned",   // "learned" = from your bootstrap_thresholds.parquet
                                       // "global_fallback" = machine not in thresholds file

      // Recommended scheduler action
      "recommended_action": "migrate_jobs",
      // Possible values:
      //   "normal"          — no action needed
      //   "monitor"         — watch this machine
      //   "defer_batch"     — hold new batch jobs
      //   "defer_new_jobs"  — hold all new job submissions
      //   "migrate_jobs"    — move existing jobs off this machine
      //   "preempt_now"     — immediate eviction

      // Short-horizon imminence signals (15m / 30m / 45m binary models)
      "imminence": {
        "15m": { "p_spike": 0.91, "is_spike": true,  "alarm_threshold": 0.78 },
        "30m": { "p_spike": 0.85, "is_spike": true,  "alarm_threshold": 0.72 },
        "45m": { "p_spike": 0.79, "is_spike": true,  "alarm_threshold": 0.65 },
        "60m": { "severity_class": 2, "p_severe": 0.71, "is_severe": true, ... }
      },

      // Top 3 features driving this prediction
      "top_features": [
        { "feature": "cpu_vs_p95",       "contribution": 0.38 },
        { "feature": "spike_in_last_1",  "contribution": 0.21 },
        { "feature": "cpu_ewma_6",       "contribution": 0.17 }
      ],

      "data_quality": { "score": 1.0, "status": "success" }
    },

    // ── Machine 17: no spike predicted ───────────────────────────────────────
    {
      "machine_id": 17,
      "status": "success",
      "observations_used": 24,
      "severity_class": 0,
      "p_no_spike":  0.82,
      "p_moderate":  0.13,
      "p_severe":    0.05,
      "p_severe_ovr": null,            // not computed when p_spike < 0.15 (cascade gate)
      "is_spike":  false,
      "is_severe": false,
      "alarm_threshold": 0.30,
      "threshold_used": 0.351,
      "threshold_source": "learned",
      "recommended_action": "normal",
      "imminence": {
        "15m": { "p_spike": 0.07, "is_spike": false, "alarm_threshold": 0.78 },
        ...
      },
      "top_features": [...],
      "data_quality": { "score": 1.0, "status": "success" }
    },

    // ── Machine 99: cold start — not enough history ───────────────────────────
    {
      "machine_id": 99,
      "status": "cold_start",          // fewer than 12 buckets (60 min) of history
      "observations_used": 6,
      "severity_class": null,
      "p_no_spike":  null,
      "p_moderate":  null,
      "p_severe":    null,
      "p_severe_ovr": null,
      "is_spike":  null,
      "is_severe": null,
      "alarm_threshold": null,
      "threshold_used": null,
      "threshold_source": null,
      "recommended_action": null,
      "imminence": null,
      "top_features": null,
      "data_quality": { "score": 0.25, "status": "cold_start" }
    }
  ]
}
```

### Status values

| Status | Meaning | Predictions |
|---|---|---|
| `success` | ≥ 24 buckets (120 min) of history | Full probability output |
| `cold_start_degraded` | 12–23 buckets available | Predictions issued but lag/EWMA features are partial — treat with lower confidence |
| `cold_start` | < 12 buckets (< 60 min) | No prediction. Null probabilities. Normal for newly provisioned machines. |

### What `threshold_source` tells you

| Value | Meaning |
|---|---|
| `"learned"` | This machine's p95/p99 came from your `bootstrap_thresholds.py` run — thresholds are domain-specific |
| `"global_fallback"` | This machine was not in your thresholds file (not enough data at bootstrap time). Uses the domain-wide median p95. Threshold-relative features are less accurate for this machine. |

---

## 8. Tune the Alarm Threshold

The alarm threshold controls when `is_severe = true` fires for the 60m model, and
when `is_spike = true` fires for the 15m/30m/45m binary models.

**The right threshold for your domain comes from `evaluate_domain.py`** (Step 5).
The value in `domain_eval_report.json → models → 60m → alarm_threshold` was selected
by maximising F1-score on your validation split. Use it.

```bash
# Take the alarm_threshold from your eval report and pass it at daemon startup:
python -m spike.daemon \
    --alarm-threshold 0.30 \    # from domain_eval_report.json
    ...
```

### When to adjust manually

If after a few days of running you observe:

| Symptom | Cause | Fix |
|---|---|---|
| Too many alarms, most are false positives | Threshold too low | Increase `--alarm-threshold` by 0.05 increments |
| Spikes happen but `is_severe` rarely fires | Threshold too high | Decrease `--alarm-threshold` by 0.05 increments |
| Alarm rate in `slo_metrics.json` > 30% | Threshold too low for this workload | Increase threshold |
| Alarm rate in `slo_metrics.json` < 2% | Threshold too high | Decrease threshold |

The precision/recall sweep is in `domain_eval_report.json` under
`models → 60m → alarm_threshold_sweep` (if present), giving you the full trade-off
curve to pick from.

### Effect on different fields

| Field | Affected by `--alarm-threshold`? |
|---|---|
| `p_severe` | No — raw model probability, always reported |
| `is_severe` (60m) | **Yes** — `p_severe >= alarm_threshold` |
| `is_spike` (60m) | No — argmax-based (`severity_class >= 1`), threshold-independent |
| `is_spike` (15m/30m/45m) | **Yes** — `p_spike >= alarm_threshold` |
| `recommended_action` | Yes — derived from `is_severe` and `is_spike` |

---

## 9. Monitor Your Deployment

Every cycle the daemon writes a second file alongside `predictions.json`:

```
predictions.json     ← per-machine predictions (overwritten each cycle)
slo_metrics.json     ← rolling health metrics (last 300 cycles)
```

**Example `slo_metrics.json`:**

```jsonc
{
  "model_version": "20260101T000000Z_a1b2c3",
  "summary": {
    "window_cycles": 288,              // cycles tracked (288 = last 24 hours at 5-min interval)
    "window_start": "2026-05-03T14:35:00Z",
    "window_end":   "2026-05-04T14:35:00Z",
    "success_rate":      0.99,         // fraction of cycles that produced predictions
    "latency_p50_s":     0.82,         // median inference time per cycle
    "latency_p95_s":     1.14,         // 95th-percentile inference time
    "alarm_rate_mean":   0.08,         // average fraction of machines flagged per cycle
    "alarm_rate_std":    0.03,
    "cold_start_rate_mean": 0.01       // average fraction of machines in cold_start
  },
  ...
}
```

### What healthy looks like

| Metric | Healthy | Investigate |
|---|---|---|
| `success_rate` | > 0.95 | < 0.90 — cycles are being skipped; check your data pipeline |
| `latency_p95_s` | < 5 s | > 10 s — inference is slow; check CPU contention |
| `alarm_rate_mean` | 5–20% | > 30% — threshold too low or real infrastructure event |
| `alarm_rate_mean` | 5–20% | < 1% — threshold too high or no real spikes in data |
| `cold_start_rate_mean` | < 5% | > 20% — machines cycling in/out or data pipeline dropping machines |

---

## 10. Troubleshooting Reference

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Fatal error — missing file, schema violation, model artefacts not found |
| 2 | Empty input — input CSV has no machines |

### Warnings and what they mean

| Warning message | Cause | Action |
|---|---|---|
| `ALL N machines are unknown to spike_thresholds.parquet` | You have not run bootstrap or not passed `--domain-dir` | Run Step 4, then add `--domain-dir ./my_domain` |
| `N/M machines are unknown to spike_thresholds.parquet` | Some machines are new since the bootstrap run | Re-run bootstrap periodically as your fleet changes |
| `Input bounds warning [total_cpu]: X% of values exceed 10.0` | CPU fed as percentage not fraction | Divide `total_cpu` and `peak_cpu` by 100 |
| `Input bounds warning [total_mem]: X% of values exceed 1.0` | Memory not normalised | Divide memory columns by total machine RAM |
| `calibrators.pkl could not be loaded` | scikit-learn version mismatch | Ensure `scikit-learn>=1.7.0,<2.0.0` is installed; raw probabilities used as fallback |
| `NaN values in N feature column(s)` | Normal for machines with short history | Not an error; XGBoost handles NaN via learned default branches |
| `X machines have fewer than 288 buckets` | Insufficient data for reliable evaluation | Collect more data before running evaluate_domain.py |

### Common failure scenarios

**`ModuleNotFoundError: No module named 'spike'`**
The package is not installed in the active environment.
```bash
source .venv/bin/activate
pip install -e .
```

**`Input is missing required column(s): ['bucket']`**
A required column is absent from your CSV. Check your data preparation script.

**`Model directory not found: data/full_run/spike`**
The model artefacts are not present. Ensure you cloned the full repository including
the model artefacts committed to `data/full_run/`.

**`Domain directory not found: ./my_domain`**
You passed `--domain-dir` to a path that does not exist. Run bootstrap first:
```bash
python -m spike.bootstrap_thresholds --input your_data.csv --domain-dir ./my_domain
```

**Predictions are all `status: cold_start`**
Every machine has fewer than 12 buckets in your input window. Your input CSV must
contain **at least the last 60 minutes (12 buckets) per machine**. For best results,
send the last 120 minutes (24 buckets).

**`macro_pr_auc: null` in eval report**
Your val or test split contains only one class (e.g. no severe spikes). Collect more
data spanning diverse load conditions, or lower `--min-buckets` if the split boundary
is cutting your data unevenly.

**Daemon cycles keep being skipped in fetch-command mode**
Your data collector script is either failing or taking longer than 60 seconds.
Test it standalone first:
```bash
python my_data_collector.py --last-24-buckets | head -5
echo "Exit code: $?"
```
The script must exit 0 and print a valid CSV with a header row to stdout.

---

*For retraining from scratch, advanced drift monitoring, or Docker/API deployment,
see the project README.*
