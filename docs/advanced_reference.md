# Advanced Reference

This document covers the full CLI reference for every script in the spike predictor, plus in-depth explanations of drift monitoring, SLO metrics, model versioning, and the FastAPI service. It assumes the reader has already worked through `ops_runbook.md` and wants to understand the system at a deeper level.

---

## Table of Contents

1. [Complete CLI Reference](#1-complete-cli-reference)
   - 1.1 [predict.py — one-shot inference](#11-predictpy--one-shot-inference)
   - 1.2 [daemon.py — continuous polling loop](#12-daemonpy--continuous-polling-loop)
   - 1.3 [bootstrap_thresholds.py — domain threshold estimation](#13-bootstrap_thresholdspy--domain-threshold-estimation)
   - 1.4 [evaluate_domain.py — retrospective backtesting](#14-evaluate_domainpy--retrospective-backtesting)
   - 1.5 [evaluate_cross_domain.py — full cross-domain evaluation](#15-evaluate_cross_domainpy--full-cross-domain-evaluation)
   - 1.6 [drift_monitor.py — PSI drift detection](#16-drift_monitorpy--psi-drift-detection)
2. [Drift Monitoring In Depth](#2-drift-monitoring-in-depth)
3. [SLO Metrics In Depth](#3-slo-metrics-in-depth)
4. [Model Versioning](#4-model-versioning)
5. [FastAPI Service Deployment](#5-fastapi-service-deployment)
6. [Domain Adaptation Workflow (Technical Detail)](#6-domain-adaptation-workflow-technical-detail)

---

## 1. Complete CLI Reference

### 1.1 `predict.py` — one-shot inference

```
python spike/predict.py [flags]
```

Reads a cluster_agg CSV, runs all loaded models, and writes predictions to stdout (JSON). Logs go to stderr. Safe to redirect stdout to a file.

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--input` | `-i` | path | **required** | Cluster_agg CSV (cluster_agg format, last 120 min = 24 buckets per machine) |
| `--model-dir` | `-m` | path | **required** | Directory containing `spike_model.json`, `spike_config.json`, `spike_model.meta.json`, `spike_thresholds.parquet` |
| `--output` | `-o` | path | `None` | Also write predictions to this file (atomic write — safe if process is killed mid-run). JSON is always printed to stdout regardless |
| `--domain-dir` | `-d` | path | `None` | Directory produced by `bootstrap_thresholds.py`. When supplied, `spike_thresholds.parquet` is loaded from here instead of `--model-dir`. Run `bootstrap_thresholds.py` first |
| `--alarm-threshold` | `-t` | float | `None` | Override the alarm threshold for all prediction horizons (0.0–1.0). Lower = more alarms. Default: per-horizon values stored in `spike_config.json` at training time |

**Exit codes:**
- `0` — success
- `1` — missing file, schema violation, or configuration mismatch
- `2` — empty input (no machines to predict)

**Examples:**

```bash
# Minimal — output to stdout
python spike/predict.py \
    --input window.csv \
    --model-dir data/full_run/spike

# Write to file too
python spike/predict.py \
    --input window.csv \
    --model-dir data/full_run/spike \
    --output predictions.json

# With domain thresholds (after running bootstrap_thresholds.py)
python spike/predict.py \
    --input window.csv \
    --model-dir data/full_run/spike \
    --domain-dir data/my_domain/thresholds \
    --output predictions.json

# Lower the alarm threshold to catch more spikes
python spike/predict.py \
    --input window.csv \
    --model-dir data/full_run/spike \
    --alarm-threshold 0.15 \
    --output predictions.json
```

---

### 1.2 `daemon.py` — continuous polling loop

```
python spike/daemon.py [flags]
```

Runs inference on a fixed schedule. Every `--interval` seconds it reads new data, runs all models, and atomically overwrites `--output` JSON. Also writes `slo_metrics.json` alongside `--output` after every cycle.

Exactly one of `--input` or `--fetch-cmd` must be provided.

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--input` | `-i` | path | (mutually exclusive with `--fetch-cmd`) | Pre-written cluster_agg CSV file. The daemon re-reads this path every cycle — any monitoring pipeline can write it independently |
| `--fetch-cmd` | `-f` | string | (mutually exclusive with `--input`) | Shell command run each cycle. Its stdout must be a cluster_agg CSV. Supports pipes and environment variables |
| `--model-dir` | `-m` | path | **required** | Directory containing the 60m model artefacts |
| `--output` | `-o` | path | **required** | Path for atomic JSON output (overwritten each cycle) |
| `--domain-dir` | `-d` | path | `None` | Directory from `bootstrap_thresholds.py` with domain-specific `spike_thresholds.parquet` |
| `--alarm-threshold` | `-t` | float | `None` | Override alarm threshold for all horizons. Applied once at startup, held for the daemon lifetime. To change it mid-run, restart the daemon |
| `--interval` | `-n` | int (seconds) | `300` | Seconds between inference cycles. Use `0` for one-shot mode (run once and exit) |

**Timing behaviour:** Cycles fire at t=0, t=interval, t=2×interval, etc. If inference takes longer than `--interval`, the next cycle starts immediately (no backlog accumulation).

**Exit codes:**
- `0` — clean shutdown (SIGTERM / SIGINT, or `--interval 0`)
- `1` — fatal startup error (missing artefacts, bad model directory, invalid flags)

**Signal handling:** SIGTERM and SIGINT trigger a clean shutdown. The daemon finishes the current cycle before stopping — it never writes a partial prediction file.

**Examples:**

```bash
# File mode — 5-minute loop (standard production setup)
python spike/daemon.py \
    --input cpu_window.csv \
    --model-dir data/full_run/spike \
    --output predictions.json \
    --interval 300

# Fetch-command mode — daemon calls your data script directly
python spike/daemon.py \
    --fetch-cmd "python scripts/fetch_zabbix.py --host zabbix.local" \
    --model-dir data/full_run/spike \
    --output predictions.json \
    --interval 300

# One-shot (run once and exit — useful for cron jobs or testing)
python spike/daemon.py \
    --input cpu_window.csv \
    --model-dir data/full_run/spike \
    --output predictions.json \
    --interval 0

# With domain thresholds and a raised alarm threshold
python spike/daemon.py \
    --input cpu_window.csv \
    --model-dir data/full_run/spike \
    --domain-dir data/my_domain/thresholds \
    --alarm-threshold 0.30 \
    --output predictions.json \
    --interval 300

# Background via nohup
nohup python spike/daemon.py \
    --input cpu_window.csv \
    --model-dir data/full_run/spike \
    --output predictions.json \
    --interval 300 \
    > daemon.log 2>&1 &
echo "PID: $!"
```

**Note on `--fetch-cmd` timeout:** The command has a hard 60-second timeout. If it exits non-zero or produces empty output, the cycle is skipped and the previous `predictions.json` is left unchanged. The next cycle retries automatically.

---

### 1.3 `bootstrap_thresholds.py` — domain threshold estimation

```
python -m spike.bootstrap_thresholds [flags]
```

Computes per-machine p95/p99 CPU thresholds from historical cluster_agg data and writes them to `--domain-dir`. This is a one-time deployment step, not a training step — no model weights are created or modified.

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--input` | `-i` | path | **required** | Historical cluster_agg data in CSV or Parquet format. At least 2 weeks (2016 five-minute buckets) per machine is recommended for stable p99 estimation |
| `--domain-dir` | `-d` | path | **required** | Output directory for `spike_thresholds.parquet` and `bootstrap_meta.json`. Created if it does not exist |
| `--min-buckets` | | int | `2016` | Minimum distinct bucket count per machine to use per-machine percentiles. Machines below this threshold receive global-domain fallback values (overall p95/p99 across all machines). 2016 = 2 weeks × 144 five-min buckets/day |

**Outputs written to `--domain-dir`:**

| File | Description |
|------|-------------|
| `spike_thresholds.parquet` | One row per machine: `[machine_id, threshold_p95, threshold_p99]` |
| `bootstrap_meta.json` | Audit record: source path, machine counts, bucket range, global fallback values, timestamp |

**Exit codes:**
- `0` — thresholds written successfully
- `1` — fatal error (missing input, wrong schema, unreadable file)

**Examples:**

```bash
# Standard bootstrap from 2 weeks of history
python -m spike.bootstrap_thresholds \
    --input data/my_domain/history.csv \
    --domain-dir data/my_domain/thresholds

# Looser threshold: allow 1 week of history (not recommended for p99)
python -m spike.bootstrap_thresholds \
    --input data/my_domain/history.csv \
    --domain-dir data/my_domain/thresholds \
    --min-buckets 1008

# From Parquet (faster for large files)
python -m spike.bootstrap_thresholds \
    --input data/my_domain/cluster_agg.parquet \
    --domain-dir data/my_domain/thresholds
```

**Fallback behaviour:** Machines with fewer than `--min-buckets` distinct bucket indices receive the global-domain fallback (overall p95/p99 across all input rows). The count of fallback machines is logged and recorded in `bootstrap_meta.json`. If the entire input spans fewer buckets than `--min-buckets`, all machines receive the fallback and a warning is emitted.

**Degenerate machines:** If a machine's p99 ≤ p95 (nearly always-idle machines), p99 is set to p95 × 1.10 — the same minimum-gap invariant enforced during training.

---

### 1.4 `evaluate_domain.py` — retrospective backtesting

```
python evaluation/evaluate_domain.py [flags]
```

Retrospective backtesting on your own cluster data. Derives spike labels programmatically from telemetry — no manual annotation needed. Outputs a JSON report with PR-AUC, ROC-AUC, and a deployment recommendation.

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--input` | `-i` | path | **required** | Historical cluster_agg data in CSV or Parquet format |
| `--model-dir` | | path | `data/full_run/spike` | Path to the 60m model directory. Sibling models (`spike_15m`, `spike_30m`, `spike_45m`, `spike_severe_ovr`) are discovered automatically |
| `--output` | `-o` | path | `data/domain_eval` | Output directory for `domain_eval_report.json` |

**What it evaluates:** Runs the full feature engineering pipeline on your data (using domain-local p95/p99 thresholds), splits chronologically into train/val/test, and measures PR-AUC and ROC-AUC on the test split for all five models. The same production model weights are used — no retraining.

**Output:** `domain_eval_report.json` with per-model metrics and a three-tier deployment recommendation:
- `deploy` — PR-AUC ≥ 75% of source performance
- `test` — PR-AUC between 50–75% of source
- `retrain_required` — PR-AUC < 50% of source

**Example:**

```bash
python evaluation/evaluate_domain.py \
    --input data/my_domain/cluster_agg.csv \
    --model-dir data/full_run/spike \
    --output data/my_domain/eval
```

---

### 1.5 `evaluate_cross_domain.py` — full cross-domain evaluation

```
python evaluation/evaluate_cross_domain.py [flags]
```

Three-phase cross-domain evaluation. More comprehensive than `evaluate_domain.py` — adds EDA (exploratory data analysis) and PSI (feature distribution shift) phases before the model evaluation.

| Flag | Short | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--agg` | | path | **required** | Path to cluster_agg Parquet for the target domain |
| `--artifacts-dir` | | path | `data/full_run` | Artifacts directory (parent of `spike/`, `cluster_features.parquet`, etc.) |
| `--output` | | path | `data/cross_domain_eval` | Output directory for `evaluation_results.json` and `evaluation_report.md` |

**Three phases:**

- **Phase 0 — EDA:** Per-node statistics (idle fraction, burstiness, peak-to-mean ratio, spike rate). Flags potential domain-incompatibility before touching the model.
- **Phase 1 — PSI:** Population Stability Index for all 40 model features, comparing the Google Cluster 2011 training distribution to the target dataset. Requires `cluster_features.parquet` from the training VM. Skipped with a warning if the file is not found.
- **Phase 2 — Model evaluation:** Feature engineering on the target data + model evaluation on the test split. Produces PR-AUC and ROC-AUC for all five models.

**Example:**

```bash
python evaluation/evaluate_cross_domain.py \
    --agg data/my_domain/cluster_agg.parquet \
    --artifacts-dir data/full_run \
    --output data/my_domain_cross_eval
```

**Note:** Phase 1 requires `data/full_run/cluster_features.parquet` — a large file (~several GB) that lives on the training VM and is gitignored. If you only need the model evaluation (Phase 2), use `evaluate_domain.py` instead, which does not require this file.

---

### 1.6 `drift_monitor.py` — PSI drift detection

```
python -m spike.drift_monitor [flags]
```

Detects feature distribution shift between the training data and live inference data using PSI (Population Stability Index). Classifies each feature as stable / warning / retrain / emergency and fires a system alert if more than 10% of features drift simultaneously.

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--reference` | path | **required** | Path to `cluster_features.parquet` (training artifacts from the training VM). This is the reference distribution |
| `--live` | path | **required** | Path to live cluster_agg data in CSV or Parquet format. The script runs feature engineering on this internally |
| `--model-dir` | path | **required** | Path to the spike model directory (e.g., `data/full_run/spike`). Used to read the model version, which is embedded in the output report |
| `--output` | path | `drift_report.json` | Output path for the drift report (written atomically — never partially written) |
| `--ref-sample-frac` | float | `0.10` | Fraction of training split rows to use as reference. 10% of ~14M training rows ≈ 1.4M rows — sufficient for stable PSI. Increase if you need higher precision on rare features |
| `--min-bins-count` | int | `5` | Emit `min_bins_warning: true` when live rows per bin fall below this value. PSI is unreliable on sparse bins. Requires at least 48 buckets per machine (4 hours of 5-min data) to avoid this warning |
| `--n-bins` | int | `10` | Number of PSI quantile bins for continuous features |

**Exit codes:**
- `0` — report written (may or may not contain alerts)
- `1` — fatal error (reference or live data not found)

**Example:**

```bash
python -m spike.drift_monitor \
    --reference data/full_run/cluster_features.parquet \
    --live data/current_window/cluster_agg.csv \
    --model-dir data/full_run/spike \
    --output drift_report.json
```

**Dependency on `cluster_features.parquet`:** This file is large (~several GB) and lives on the training VM. It is not shipped with the model artefacts. See Section 2 for how to obtain it.

---

## 2. Drift Monitoring In Depth

### What drift monitoring detects

The model was trained on Google Cluster 2011 telemetry. When deployed on a different cluster, two things can cause performance degradation:

1. **Domain shift at deployment time:** The target cluster has different CPU load distributions, workload patterns, or machine heterogeneity compared to Google 2011.
2. **Temporal drift during operation:** The production cluster evolves — capacity additions, software updates, workload changes — and the input distributions gradually move away from what the model saw during training.

`drift_monitor.py` detects both by comparing the current live feature distributions to the training distribution using PSI.

### PSI thresholds

PSI (Population Stability Index) measures how much a feature distribution has shifted between a reference (training) and current (live) dataset.

| PSI value | Status | Meaning |
|-----------|--------|---------|
| < 0.10 | **stable** | No meaningful shift — model is operating in-distribution |
| 0.10 – 0.25 | **warning** | Moderate shift — watch the feature; likely acceptable |
| 0.25 – 0.50 | **retrain** | Major shift — this feature is behaving differently than at training time |
| > 0.50 | **emergency** | Extreme shift — feature may be meaningless; consider disabling or retraining immediately |

### System alert logic

A **system alert** fires when more than 10% of all valid features exceed PSI 0.20 simultaneously. The 10% threshold distinguishes systematic drift (a new workload pattern, a capacity event, a data pipeline change) from individual feature noise — a single binary feature fluctuating with spike rate is normal.

The alert threshold is 0.20, intentionally lower than the `retrain` boundary (0.25), so the alert fires earlier during slow, broad drift.

### Interpreting the drift report alongside SLO metrics

The real diagnostic value comes from pairing `drift_report.json` with `slo_metrics.json`:

| SLO alarm_rate | PSI system_alert | Interpretation |
|----------------|-----------------|----------------|
| Rising | False | Real infrastructure event — actual spike activity increasing |
| Rising | True | Feature drift — model is seeing out-of-distribution data; alarm rate may be unreliable |
| Stable | True | Drift present but model not yet affected — monitor closely |
| Falling | True | Possible under-prediction — drift may be suppressing alarms that should fire |

### When to run drift monitoring

- **After bootstrapping thresholds** for a new domain: run immediately after `bootstrap_thresholds.py` to verify the domain thresholds correctly align the feature distributions.
- **Periodically during operation** (weekly is sufficient for most clusters): catch slow drift before it degrades model performance.
- **After any infrastructure change:** cluster capacity event, workload migration, monitoring pipeline change.
- **When `slo_metrics.json` shows anomalies:** unexplained spike in alarm_rate, alarm_rate near zero, or rising latency.

### Obtaining `cluster_features.parquet`

This file is the training-split feature matrix (~14M rows × 40 features). It lives on the training VM under `data/full_run/cluster_features.parquet` and is gitignored (too large to ship).

To use drift monitoring, either:
- Run it directly on the training VM (the file is already there)
- Or copy it to your local machine: `scp vm-user@vm-ip:~/spike/data/full_run/cluster_features.parquet data/full_run/`

If the file is unavailable, drift monitoring is blocked. Use `evaluate_cross_domain.py` Phase 1 as an alternative for one-off domain shift analysis.

### `drift_report.json` structure

```json
{
  "generated_at": "2026-05-04T14:00:00+00:00",
  "model_version": "20260429T143000Z_7e4d2f",
  "summary": {
    "n_features_evaluated": 40,
    "n_stable":    6,
    "n_warning":   9,
    "n_retrain":   17,
    "n_emergency": 8,
    "system_alert": true,
    "system_alert_reason": "28/40 features exceed PSI 0.20 (70.0% > threshold 10.0%)",
    "min_bins_warning": false
  },
  "features": {
    "cpu_vs_p95": {
      "psi": 0.04,
      "status": "stable",
      "reference_mean": 0.82,
      "live_mean": 0.79
    },
    "total_cpu": {
      "psi": 0.61,
      "status": "emergency",
      "reference_mean": 0.18,
      "live_mean": 0.43
    }
    ...
  }
}
```

**Key fields:**
- `system_alert` — `true` when > 10% of features exceed PSI 0.20. This is the top-level actionable signal.
- `system_alert_reason` — human-readable explanation of what triggered the alert.
- `min_bins_warning` — `true` when the live window is too small for reliable PSI estimates. Requires at least 48 buckets per machine to avoid this.
- `model_version` — identifies which model artefacts were in use when the report was generated. Correlate with `slo_metrics.json → model_version`.
- Per-feature `psi` and `status` — ranked by severity; inspect the `retrain` and `emergency` features first.

**Zabbix result as reference:** On the Zabbix cross-domain evaluation, 25/40 features showed MAJOR shift (PSI > 0.25) and 8 showed emergency (PSI > 0.50). Yet the model retained >100% of its Google PR-AUC. This is because the highest-PSI features (absolute CPU scale, `n_tasks`, `disk_io`) are low-importance in the learned model, while the dominant predictors (`cpu_vs_p95`, `band_position`, `spike_in_last_*`) are machine-relative and domain-invariant. A high PSI does not automatically mean poor performance — check SLO metrics alongside it.

---

## 3. SLO Metrics In Depth

`slo_metrics.json` is written by the daemon alongside `predictions.json` after every cycle. It tracks a rolling 300-cycle window (~25 hours at the default 5-minute interval).

### Full file structure

```json
{
  "model_version": {
    "model_version": "20260429T143000Z_7e4d2f",
    "trained_at": "2026-04-29T14:30:00+00:00",
    "feature_hash": "7e4d2f",
    "xgboost_version": "2.1.3",
    "domain_bootstrapped_at": "2026-05-01T09:00:00+00:00"
  },
  "summary": {
    "window_cycles": 300,
    "window_start": "2026-05-03T13:00:00+00:00",
    "window_end":   "2026-05-04T14:00:00+00:00",
    "success_rate":         0.9967,
    "latency_p50_s":        0.84,
    "latency_p95_s":        1.12,
    "alarm_rate_mean":      0.031,
    "alarm_rate_std":       0.009,
    "cold_start_rate_mean": 0.0
  },
  "cycles": [
    {
      "cycle_index":         0,
      "timestamp":           "2026-05-03T13:00:00+00:00",
      "latency_s":           0.87,
      "success":             true,
      "machines_total":      120,
      "machines_predicted":  120,
      "machines_cold_start": 0,
      "alarm_count":         4,
      "alarm_rate":          0.0333,
      "cold_start_rate":     0.0
    }
    ...
  ]
}
```

### Field reference — `summary`

| Field | Type | What it means |
|-------|------|---------------|
| `window_cycles` | int | Number of cycles in the rolling window (max 300 = ~25h at 5min) |
| `window_start` | ISO 8601 | Timestamp of the oldest cycle in the current window |
| `window_end` | ISO 8601 | Timestamp of the most recent cycle |
| `success_rate` | float [0,1] | Fraction of cycles that produced a prediction. A skipped cycle (missing input file, `--fetch-cmd` timeout, schema error) counts as a failure |
| `latency_p50_s` | float | Median inference time in seconds. Includes feature engineering + XGBoost forward pass for all 5 models |
| `latency_p95_s` | float | 95th-percentile latency. Covers both success and skipped cycles — skipped cycles still consume time (command timeout, file-not-found check) |
| `alarm_rate_mean` | float | Mean fraction of predicted machines that triggered an alarm, computed over successful cycles only |
| `alarm_rate_std` | float | Standard deviation of alarm_rate. Needs ≥ 2 successful cycles; `null` otherwise |
| `cold_start_rate_mean` | float | Mean fraction of machines that were cold-started (insufficient history), over successful cycles |

### Field reference — per-cycle in `cycles`

| Field | Type | What it means |
|-------|------|---------------|
| `cycle_index` | int | Monotonically increasing cycle counter (starts at 0, never resets) |
| `timestamp` | ISO 8601 | When this cycle ran |
| `latency_s` | float | Wall-clock time for this cycle in seconds |
| `success` | bool | Whether a prediction was produced. `false` = cycle was skipped |
| `machines_total` | int | Total unique machines in the input (null if skipped) |
| `machines_predicted` | int | Machines that received a probability estimate |
| `machines_cold_start` | int | Machines below the minimum bucket threshold |
| `alarm_count` | int | Machines whose `recommended_action` is not `normal` or `monitor` |
| `alarm_rate` | float | `alarm_count / machines_predicted`. Null if no machines were predicted |
| `cold_start_rate` | float | `machines_cold_start / machines_total`. Null if no machines in input |

### Healthy ranges

These are operational guidelines based on the Google Cluster 2011 test results and Zabbix evaluation. Adjust for your domain.

| Metric | Healthy | Investigate |
|--------|---------|-------------|
| `success_rate` | ≥ 0.99 | < 0.95 |
| `latency_p50_s` | < 2.0 | > 5.0 |
| `latency_p95_s` | < 5.0 | > 15.0 |
| `alarm_rate_mean` | 0.01 – 0.15 | > 0.30 or < 0.001 |
| `alarm_rate_std` | < 0.05 | > 0.15 |
| `cold_start_rate_mean` | < 0.05 | > 0.20 |

**Alarm rate near zero** is as worrying as an alarm rate that is too high. It typically means the alarm threshold is set too high, the machine pool changed and most machines are cold-starting, or features are being zeroed out by a schema problem.

**Alarm rate std spike** without a corresponding alarm_rate_mean increase suggests oscillating predictions — the model is near the threshold boundary for many machines. Lower the alarm threshold or inspect those machines' feature trajectories.

### Window eviction

The `cycles` deque is capped at 300 entries. When full, the oldest entry is evicted automatically. `window_start` moves forward accordingly. The summary statistics always reflect the current window, not the full daemon lifetime.

To inspect history older than 300 cycles, aggregate `predictions.json` history from your monitoring system.

---

## 4. Model Versioning

### What the version string encodes

Every predictions.json and slo_metrics.json embed a `model_version` object. The version string has two parts:

```
20260429T143000Z_7e4d2f
└──────────────┘ └────┘
  compact_timestamp  feature_hash
```

- **`compact_timestamp`:** Compact UTC form of `trained_at` from `spike_config.json`. Example: `"2026-04-29T14:30:00+00:00"` → `"20260429T143000Z"`. Falls back to `"unknown"` when the config is absent.
- **`feature_hash`:** First 6 hex characters of SHA-256 of `"|".join(sorted(feature_cols))` from `spike_model.meta.json`. Encodes which 40-feature set was used without listing all columns. Same features always produce the same hash, regardless of weights or thresholds.

The combined string is deterministic: identical artefacts always produce the same version string.

### Full `model_version` object fields

| Field | Source | What it tells you |
|-------|--------|-------------------|
| `model_version` | computed | `"<ts>_<hash>"` — the stable identifier |
| `trained_at` | `spike_config.json → trained_at` | Raw ISO-8601 training timestamp |
| `feature_hash` | `spike_model.meta.json → feature_cols` | 6-char fingerprint of the feature set |
| `xgboost_version` | `spike_model.meta.json → xgboost_version` | XGBoost version at training time. Mismatch with inference-time XGBoost can cause calibrator pickle failures |
| `domain_bootstrapped_at` | `domain_dir/bootstrap_meta.json → bootstrapped_at` | When the domain thresholds were computed. `null` if no `--domain-dir` was given |

### How to use versioning operationally

**Correlate anomalies to model changes:** When `alarm_rate_mean` spikes or drops after a deployment, check `model_version` in `slo_metrics.json` to confirm whether the artefacts changed. Same hash + timestamp = same model, so the anomaly is real.

**Detect accidental artefact swap:** If two daemons are supposed to use the same model but show different prediction distributions, compare their `model_version` strings. Different hashes mean different feature sets were used at training time.

**Audit bootstrap timing:** `domain_bootstrapped_at` records when per-machine thresholds were estimated. If the production cluster was later restructured (machines added, workload shifted), the age of this timestamp tells you how stale the domain thresholds are.

**Legacy artefacts:** Artefacts trained before versioning was added produce `model_version: "unknown_000000"`. This is a safe fallback — inference still works normally; only the version metadata is absent.

---

## 5. FastAPI Service Deployment

### Overview

`spike/api.py` exposes four HTTP endpoints via FastAPI:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe — always 200 while the process is running |
| `/ready` | GET | Readiness probe — 503 until artefacts are loaded |
| `/predict` | POST | Full per-machine inference (probabilities, imminence, SHAP top features) |
| `/summary` | POST | Scheduler-facing summary: spike counts per horizon + per-node status |

Artefacts are loaded once at startup (FastAPI lifespan context) and held in memory. Inference calls `predict_with_artifacts()` directly — no disk reads per request.

Both `/predict` and `/summary` share the same per-machine EWMA and debounce state. Use one endpoint consistently per deployment cycle — calling both in the same cycle double-updates the smoothing and consecutive-alarm counters.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `/app/data/full_run/spike` | Path to the spike model directory inside the container. Override this to point at a different model or a volume mount |
| `EWMA_ALPHA` | `0.5` | Exponential smoothing factor for per-machine `p_spike` across requests. `1.0` = no smoothing (pass-through). `0.3` = slower response (10-cycle half-life) |
| `ALARM_MIN_CONSECUTIVE` | `2` | Minimum consecutive spiking predictions before `is_spike` fires. Set to `1` to disable debouncing. Prevents single-cycle noise from triggering scheduler actions |

### EWMA smoothing

The API maintains a per-machine exponentially weighted moving average of `p_spike = p_moderate + p_severe` across HTTP requests. This dampens single-cycle probability spikes caused by transient telemetry noise.

The smoothed score is written to `p_spike_smoothed` in the response. EWMA can only **remove** alarms, never create them — if the raw prediction says `is_spike = false`, EWMA cannot override it. Cold-start machines skip EWMA and get `p_spike_smoothed: null`.

Formula per machine: `smoothed_new = alpha × p_spike_raw + (1 − alpha) × smoothed_prev`

State is in-memory and resets on restart. A short gap after restart simply requires the machine to spike again before the alarm fires.

### Alarm debounce

The API also applies a consecutive-spike filter. A machine must predict `is_spike = true` for `ALARM_MIN_CONSECUTIVE` consecutive requests before the alarm is forwarded to the scheduler. This prevents a single noisy cycle from triggering pre-emptive scheduler actions.

`consecutive_alarms` is included in the response for every prediction — useful for dashboard display and debugging. Raw probabilities and the `imminence` detail block are always returned unchanged; only the actionable fields (`is_spike`, `is_severe`, `recommended_action`) are suppressed during debounce.

### Docker deployment

```yaml
# docker-compose.yml — inference API service
services:
  api:
    build:
      context: .
      dockerfile: spike/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./data/full_run:/app/data/full_run:ro
    environment:
      MODEL_DIR: /app/data/full_run/spike
      EWMA_ALPHA: "0.5"
      ALARM_MIN_CONSECUTIVE: "2"
```

```bash
# Start the service
docker compose up --build

# Check liveness
curl http://localhost:8000/health

# Check readiness (shows loaded horizons, machine count, feature count)
curl http://localhost:8000/ready

# Send a prediction request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "rows": [
      {"machine_id": 1, "bucket": 100, "time_us": 30000000000,
       "total_cpu": 0.42, "peak_cpu": 0.61, "total_mem": 1.2,
       "peak_mem": 2.1, "disk_io": 0.0, "n_tasks": 4}
    ]
  }'
```

### Kubernetes deployment (OSM k3s)

The API ships with five K8s manifests in `k8s/`:

| Manifest | Purpose |
|----------|---------|
| `k8s/namespace.yaml` | Creates the `spike` namespace |
| `k8s/pvc.yaml` | 500 Mi `local-path` PVC for model artifacts |
| `k8s/populate-pvc.yaml` | Temporary loader pod for `kubectl cp` of artifacts |
| `k8s/deployment.yaml` | 1-replica Deployment, `imagePullPolicy: Never`, PVC at `/app/data/full_run` |
| `k8s/service.yaml` | ClusterIP service on port 8000 |
| `k8s/ingress.yaml` | nginx Ingress — external DNS via nip.io (`ingressClassName: nginx`) |

**Image strategy:** No container registry is used. Build the image on the data VM and import it directly into k3s containerd:

```bash
# On data VM
docker build -f spike/Dockerfile -t spike-api:latest .
docker save spike-api:latest | gzip > /tmp/spike-api.tar.gz
scp /tmp/spike-api.tar.gz atnoguser@<CLUSTER_IP>:~/

# On OSM VM
sudo k3s ctr images import ~/spike-api.tar.gz
```

**Access:**

| Caller | URL |
|--------|-----|
| Pods inside the cluster | `http://spike-api.spike.svc.cluster.local:8000` |
| VIMs / external VMs | `http://spike-api.<CLUSTER_IP>.nip.io` (nginx Ingress, port 80) |

```bash
# From any VIM or external host
curl http://spike-api.<CLUSTER_IP>.nip.io/health
curl http://spike-api.<CLUSTER_IP>.nip.io/ready

curl -X POST http://spike-api.<CLUSTER_IP>.nip.io/summary \
  -H "Content-Type: application/json" \
  -d '{"rows": [{"machine_id": 1, "bucket": 100, "time_us": 30000000000,
       "total_cpu": 0.42, "peak_cpu": 0.61, "total_mem": 1.2,
       "peak_mem": 2.1, "disk_io": 0.0, "n_tasks": 4}]}'
```

See `session_anchor.md` in the repo root for the full step-by-step deployment runbook.

### `/ready` response

```json
{
  "status": "ready",
  "horizons": ["spike", "spike_15m", "spike_30m", "spike_45m", "spike_severe_ovr"],
  "known_machines": 12555,
  "feature_count": 40,
  "trained_at": "2026-04-29T14:30:00+00:00",
  "calibrated": true
}
```

`known_machines` is the number of machines in `spike_thresholds.parquet`. Machines not in this set are "unknown" — they receive global-fallback thresholds (the domain-wide p95/p99) and `threshold_source: "global_fallback"` in the response.

### `/predict` — additional API-only fields

The API response includes two fields not present in `predict.py` / `daemon.py` output:

| Field | What it means |
|-------|---------------|
| `p_spike_smoothed` | EWMA-smoothed `p_spike = p_moderate + p_severe`. `null` for cold-start machines |
| `consecutive_alarms` | Number of consecutive cycles this machine has been in alarm state. Used by debounce. `0` for cold-start machines |

### `/summary` — scheduler-facing endpoint

`POST /summary` accepts the same request body as `/predict` and runs the same inference pipeline (including EWMA smoothing and alarm debounce). It returns a compact response optimised for scheduler consumption:

```json
{
  "summary": {
    "spikes_in_15m": 2,
    "spikes_in_30m": 2,
    "spikes_in_45m": 2,
    "spikes_in_60m": 0,
    "nodes_requiring_action": 2
  },
  "per_node": [
    {
      "machine_id": 3,
      "status": "success",
      "spike_60m": false,
      "spike_imminent": true,
      "severity": "no_spike",
      "alarm_source": "binary_15m",
      "scheduler_score": 48,
      "horizon": "15m",
      "recommended_action": "preempt_now",
      "p_spike_smoothed": 0.519,
      "consecutive_alarms": 1
    }
  ],
  "predicted_at": "2026-05-15T10:28:21+00:00",
  "machines_total": 11,
  "machines_predicted": 11,
  "machines_cold_start": 0
}
```

**`summary` fields:**

| Field | Meaning |
|-------|---------|
| `spikes_in_Xm` | Machines predicted to spike within X minutes (independent per horizon, not cumulative). Cold-start machines excluded. `0` = no alarms |
| `nodes_requiring_action` | Machines with `recommended_action` of `preempt_now` or `migrate_jobs`. Cluster-level signal — no need to iterate `per_node` for a quick check |

**`per_node` fields:**

| Field | Type | Meaning |
|-------|------|---------|
| `machine_id` | int | Node identifier |
| `status` | str | `success` or `cold_start` (too few history buckets to predict) |
| `spike_60m` | bool\|null | 60m severity model verdict — sustained spike risk over the next hour. `null` for cold-start |
| `spike_imminent` | bool\|null | `true` if any short-horizon binary model (15m/30m/45m) alarmed. More sensitive than `spike_60m` for near-term bursts. `null` for cold-start |
| `severity` | str\|null | `no_spike`, `moderate`, or `severe` (from 60m model). `null` for cold-start |
| `alarm_source` | str\|null | Which model drove `recommended_action`: `none`, `binary_15m`, `binary_30m`, `binary_45m`, or `60m_severity`. Use this when `spike_60m` and `spike_imminent` disagree. `null` for cold-start |
| `scheduler_score` | int\|null | **0–100. Higher = safer.** Computed as `round(100 × (1 − p_spike_smoothed))`. Plugs directly into a Kubernetes Score plugin — no transformation required. `null` for cold-start |
| `horizon` | str\|null | Earliest alarming horizon: `15m`, `30m`, `45m`, or `60m`. The scheduler's lead time before the predicted spike. `null` if no alarm |
| `recommended_action` | str\|null | What the scheduler should do (see table below). `null` for cold-start |
| `p_spike_smoothed` | float\|null | Raw EWMA-smoothed spike probability (0.0–1.0). Useful for dashboards and custom scoring formulas. `null` for cold-start |
| `consecutive_alarms` | int | Consecutive spiking prediction cycles (debounce counter). `0` = first alarm or no alarm |

**`recommended_action` decision table:**

| Value | Meaning | Suggested scheduler behaviour |
|-------|---------|-------------------------------|
| `normal` | No spike predicted | Schedule freely |
| `monitor` | Mild risk, below alarm threshold | Continue scheduling; watch next cycle |
| `migrate_jobs` | Moderate sustained risk (60m model) | Drain long-running jobs; avoid new placements |
| `preempt_now` | Imminent spike (binary 15m/30m/45m) | Stop scheduling to this node immediately; migrate if possible |

**Key distinction:** `spike_60m` and `spike_imminent` answer different questions and can disagree:
- `spike_imminent: true` + `spike_60m: false` → short burst expected, no sustained overload. Act quickly (`preempt_now`) but expect recovery within the hour.
- `spike_60m: true` + `spike_imminent: false` → sustained overload building slowly. Drain gradually (`migrate_jobs`).
- Both true → highest urgency. Act immediately and plan for extended unavailability.

Use `alarm_source` to understand which model drove the action when the two signals disagree.

```bash
# Send a summary request
curl -X POST http://localhost:8000/summary \
  -H "Content-Type: application/json" \
  -d '{
    "rows": [
      {"machine_id": 1, "bucket": 100, "time_us": 30000000000,
       "total_cpu": 0.42, "peak_cpu": 0.61, "total_mem": 1.2,
       "peak_mem": 2.1, "disk_io": 0.0, "n_tasks": 4}
    ]
  }'
```

### Kubernetes scheduler integration

The `/summary` endpoint is designed to integrate with the [Kubernetes Scheduling Framework](https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/). The recommended pattern uses two extension points:

**PreScore** — called once per scheduling cycle with all candidate nodes. Call `/summary` here and stash the result in `CycleState`:

```go
func (pl *SpikePlugin) PreScore(ctx context.Context, state *framework.CycleState,
    pod *v1.Pod, nodes []*v1.Node) *framework.Status {

    rows := buildClusterAgg(nodes)           // fetch last 120 min of metrics
    resp, err := postSummary(spikeAPIURL, rows)
    if err != nil {
        return framework.NewStatus(framework.Success) // fail open — don't block scheduling
    }
    // Index by machine_id for O(1) per-node lookup in Score
    byNode := map[int]NodeResult{}
    for _, n := range resp.PerNode {
        byNode[n.MachineID] = n
    }
    state.Write(stateKey, &SpikeState{ByNode: byNode})
    return framework.NewStatus(framework.Success)
}
```

**Score** — called once per node, reads from `CycleState` (no additional API calls):

```go
func (pl *SpikePlugin) Score(ctx context.Context, state *framework.CycleState,
    pod *v1.Pod, nodeName string) (int64, *framework.Status) {

    s, err := state.Read(stateKey)
    if err != nil {
        return 50, framework.NewStatus(framework.Success) // neutral fallback
    }
    spikeState := s.(*SpikeState)
    nodeResult, ok := spikeState.ByNode[machineIDFor(nodeName)]
    if !ok || nodeResult.SchedulerScore == nil {
        return 50, framework.NewStatus(framework.Success) // cold-start: neutral
    }
    return int64(*nodeResult.SchedulerScore), framework.NewStatus(framework.Success)
}
```

`scheduler_score` is already in the 0–100 range the framework expects. The Score plugin returns it directly — no formula needed. Nodes with `scheduler_score: 100` are preferred; nodes with `scheduler_score: 0` are avoided.

---

## 6. Domain Adaptation Workflow (Technical Detail)

### Why domain adaptation is needed

The spike inference pipeline uses machine-relative normalisation: features like `cpu_vs_p95`, `band_position`, `spike_in_last_6`, and `peak_cpu_vs_p99` are all computed relative to per-machine p95/p99 CPU thresholds. These thresholds are estimated from training data (Google Cluster 2011) and baked into `spike_thresholds.parquet`.

When the model is deployed on a different cluster, the source-domain thresholds are wrong for the new machines — they reflect Google 2011 load patterns, not the target cluster. Bootstrapping domain-specific thresholds corrects this without retraining the XGBoost weights.

### What bootstrap changes and what it does not

**Changes (domain-specific):**
- Per-machine p95/p99 thresholds in `spike_thresholds.parquet`
- All features that depend on these thresholds: `cpu_vs_p95`, `cpu_vs_p95_delta`, `cpu_vs_p95_slope_3`, `cpu_vs_p95_slope_6`, `time_to_p95_3`, `peak_cpu_vs_p95`, `spike_now`, `spike_in_last_{1,3,6}`, `time_since_last_spike`, `cpu_spike_rate_24`, `spike_severe_now`, `cpu_vs_p99`, `peak_cpu_vs_p99`, `band_position`, `band_width`, `spike_severe_in_last_{1,3,6}`

**Does not change:**
- XGBoost model weights (the learned trees remain identical)
- Calibrators (`calibrators.pkl` — isotonic regression fitted on Google 2011 val set)
- Alarm thresholds stored in `spike_config.json` (override with `--alarm-threshold` if needed)
- 40-feature names and their expected data types

### Step-by-step domain adaptation

```
Historical cluster_agg data (≥ 2 weeks per machine)
  → [bootstrap_thresholds.py]  → domain_dir/spike_thresholds.parquet
                                  domain_dir/bootstrap_meta.json

                                         ↓

Historical cluster_agg data (as large as possible)
  → [evaluate_domain.py]       → domain_eval_report.json
                                  (PR-AUC + deployment recommendation)

                                         ↓

If recommendation = "deploy":
  → [daemon.py --domain-dir]   → predictions.json (live, every 5 min)
                                  slo_metrics.json

If recommendation = "test":
  → [daemon.py --domain-dir]   → predictions.json (pilot, monitor closely)
  → adjust --alarm-threshold
  → re-evaluate after 1 week

If recommendation = "retrain_required":
  → Collect at least 2 weeks of labelled data from your domain
  → Run training/train.py with your cluster_agg data
  → New model weights specific to your domain
```

### Threshold quality indicators

After running `bootstrap_thresholds.py`, check `bootstrap_meta.json` for:

| Field | What to look for |
|-------|-----------------|
| `n_machines_with_per_machine` | Should be > 80% of total machines. Low count means most data is too sparse |
| `n_machines_with_global_fallback` | High count means machines have < 2 weeks of history — thresholds will be less precise |
| `global_p95` / `global_p99` | Compare to your cluster's expected peak load. Values near 0 or 1 may indicate a CPU normalisation issue |

### When to retrain from scratch

Domain adaptation (threshold bootstrap) is sufficient when the target cluster follows similar CPU spike patterns — peaks above a per-machine baseline, similar burstiness, similar spike duration. It is not sufficient when:

- Your cluster is fundamentally different in spike structure (e.g., GPU-driven peaks that look different in shape from CPU-only tasks)
- `evaluate_domain.py` returns `retrain_required` (PR-AUC < 50% of source)
- `drift_monitor.py` shows emergency-level PSI (> 0.50) on the highest-importance features (`cpu_vs_p95`, `band_position`, `spike_in_last_6`)

In those cases, collect cluster_agg data from your domain, run `training/train.py`, and treat the result as a new model rather than an adapted one.

### Alarm threshold adjustment after domain adaptation

The alarm threshold stored in `spike_config.json` was selected to maximise F1 on the Google 2011 validation set. On a new domain, the optimal operating point may differ:

- **High-criticality clusters:** lower `--alarm-threshold` (e.g., 0.15–0.20) to catch more spikes at the cost of more false positives
- **Low-criticality or noisy clusters:** raise `--alarm-threshold` (e.g., 0.35–0.50) to reduce false positives
- **Reference from `evaluate_domain.py`:** the report includes an alarm threshold sweep (F1 vs threshold) for the test split. Use that to pick an operating point tuned to your domain.

The threshold affects only `is_spike` (binary horizons), `is_severe` (60m cascade output), and `recommended_action`. The raw probabilities (`p_no_spike`, `p_moderate`, `p_severe`) are always written unchanged.
