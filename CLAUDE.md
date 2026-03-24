# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

**CPU spike prediction for datacenter scheduling.** Given a rolling window of per-machine CPU telemetry, the system predicts whether a CPU spike will occur in the next 60 minutes and how severe it will be. Output feeds a workload scheduler so it can act pre-emptively — before the spike materialises.

**Production scenario:** Every 5 minutes, read the last 120 min of per-machine CPU data (24 buckets), predict severity for each machine, feed predictions to scheduler.

## Directory Layout

```
Agentic_AI/
├── CLAUDE.md                  ← this file
├── DEVELOPMENT.md             ← full decision log and phase history
├── docker-compose.yml         ← API + GPU training services
├── scripts/
│   └── vm-setup.sh            ← one-shot GPU VM provisioning (Docker + NVIDIA Container Toolkit)
└── AIModel/               ← all application code
    ├── Dockerfile              ← inference API image (CPU-only)
    ├── Dockerfile.test         ← test runner image
    ├── Dockerfile.training     ← GPU training image (nvidia/cuda:12.4.1-runtime-ubuntu24.04)
    ├── requirements-inference.txt
    ├── requirements-train.txt
    ├── spike_preprocessor.py
    ├── spike_feature_engineer.py
    ├── spike_classifier.py
    ├── train_spike_classifier.py
    ├── predict_spike.py
    ├── spike_api.py
    ├── data/
    │   ├── cluster_cpu_data.csv   ← Google Cluster Traces 2011 (278M rows, ~8GB)
    │   ├── download_cluster_data.py  ← script used to pull the dataset
    │   └── full_run/              ← all training artifacts (parquet cache + models)
    └── tests/
        ├── conftest.py
        ├── test_spike_preprocessor.py
        ├── test_spike_feature_engineer.py
        ├── test_spike_classifier.py
        ├── test_train_spike_classifier.py
        └── test_predict_spike.py
```

All commands run from `AIModel/`.

## Commands

```bash
cd AIModel/

# Run the full test suite (244 passed, 4 skipped — runs in ~17 seconds)
.venv/bin/python3 -m pytest tests/ -v

# Run the full training pipeline from scratch (CPU)
nohup .venv/bin/python3 train_spike_classifier.py \
  --data-path data/cluster_cpu_data.csv \
  --artifacts-dir data/full_run \
  --tune-hyperparams --optuna-trials 30 > data/run_log.txt 2>&1 & echo "PID: $!"

# Resume from Step 3 (training only — skips preprocessing and feature engineering)
nohup .venv/bin/python3 train_spike_classifier.py \
  --data-path data/cluster_cpu_data.csv \
  --artifacts-dir data/full_run \
  --from-step 3 \
  --tune-hyperparams --optuna-trials 30 > data/run_log.txt 2>&1 & echo "PID: $!"

# Monitor training
tail -f data/run_log.txt

# Run inference on a pre-aggregated window
.venv/bin/python3 predict_spike.py \
  --input data/window.csv \
  --model-dir data/full_run/models/spike/

# GPU training via Docker Compose (requires NVIDIA Container Toolkit on host)
cd ..   # back to Agentic_AI/
docker compose --profile train run --rm --build train

# API service (requires trained artifacts in data/full_run/)
docker compose up --build

# Run test suite in Docker
docker compose --profile test run --rm --build test

# VM setup (run once on fresh Ubuntu 22.04/24.04 GPU VM)
bash scripts/vm-setup.sh
```

## Pipeline Architecture

Three training steps, each caching its output:

```
cluster_cpu_data.csv
  → [Step 1] spike_preprocessor.py      → data/full_run/cluster_agg.parquet
  → [Step 2] spike_feature_engineer.py  → data/full_run/cluster_features.parquet
                                           data/full_run/spike_thresholds.parquet
  → [Step 3] train_spike_classifier.py  → data/full_run/models/spike/            (60m severity)
                                           data/full_run/models/spike_15m/        (15m binary)
                                           data/full_run/models/spike_severe_ovr/ (OVR severe)
```

Use `--from-step N` to resume from any step. Use `--force` to clear a step's cache.

## Model Architecture

Three models trained on the same feature set:

- **60m severity model** (`SpikeClassifier`, `multi:softprob`, 3 classes)
  Predicts: `no_spike` / `moderate` (p95 exceeded) / `severe` (p99 exceeded) in the next 60 min.
  Primary metric: macro PR-AUC.

- **15m binary model** (`BinarySpikeClassifier`, `binary:logistic`)
  Predicts: will any spike occur in the next 15 min?
  Used for imminence scoring alongside the 60m model.

- **OVR severe model** (`BinarySpikeClassifier`, `binary:logistic`)
  Label: `spike_severe_ovr` — severe (class 2) vs rest.
  Dedicated binary classifier for the rare severe class; trained alongside the multiclass model.
  Outputs `p_severe_ovr` in inference. Saved to `models/spike_severe_ovr/`.

## Data

**Dataset:** Google Cluster Traces 2011
- 278M raw rows → ~24M (machine_id × 5-min bucket) aggregates after Step 1
- 160 hours, 12,555 machines, 1,919 buckets per machine
- Split: Train 60% / Val 20% / Test 20% (chronological — no shuffle)

**Input schema (cluster_agg format):**

| Column | Type | Description |
|--------|------|-------------|
| `machine_id` | int64 | Unique machine identifier |
| `bucket` | int64 | 5-min bucket index |
| `time_us` | int64 | `bucket × 300_000_000` (microseconds) |
| `total_cpu` | float32 | Duration-weighted CPU load (fraction of 1 core) |
| `peak_cpu` | float32 | Peak CPU rate in the 5-min window |
| `total_mem` | float32 | Sum of canonical memory usage |
| `peak_mem` | float32 | Peak memory usage |
| `disk_io` | float32 | Max mean disk I/O time |
| `n_tasks` | int32 | Concurrent tasks in bucket |

## Features (37 total)

- **Raw:** `total_cpu, peak_cpu, total_mem, peak_mem, disk_io, n_tasks`
- **Lags:** `cpu_lag_{1,12,24}`
- **Trend:** `cpu_ewma_{6,24}, cpu_delta_{1,2}, cpu_rolling_std_6`
- **Load ratio:** `cpu_per_task`
- **Machine-relative (p95):** `cpu_vs_p95, cpu_vs_p95_delta, peak_cpu_vs_p95, spike_now, spike_in_last_{1,3,6}, time_since_last_spike, cpu_spike_rate_24`
- **Machine-relative (p99):** `spike_severe_now, cpu_vs_p99, peak_cpu_vs_p99, band_position, band_width, spike_severe_in_last_{1,3,6}`
- **Cluster:** `cluster_cpu_p90, machine_rank_in_cluster, task_dominance`
- **Time:** `hour_sin, hour_cos`

Per-machine p95 and p99 thresholds are computed from training data only (no leakage).
Minimum 10% gap between p99 and p95 is enforced to ensure a meaningful moderate band.

`dow_sin`/`dow_cos` were dropped in Phase 5: highest SHAP (0.356) but near-zero gain (0.011),
only 7 days of data (23 samples per label), confirmed temporal confound with the Google Cluster
2011 trace week.

## Key Design Constraints

- **Chronological splits only** — no random shuffle anywhere in the pipeline
- **Walk-forward CV** — 5 folds with a 12-bucket gap (= 1 horizon) between train and val to prevent label leakage
- **Thresholds from training data only** — p95/p99 computed on train buckets, never val/test
- **Monotone constraints** — enabled for binary models only; disabled for `multi:softprob` (undefined semantics)
- **Macro PR-AUC** — primary metric for 60m model (equal weight to all severity classes including rare severe)
- **Binary Optuna capped at 30 trials** — binary:logistic converges faster than multi:softprob
- **Optuna inner split capped at 2M rows** — OOM prevention for binary model (full inner split = 7M+ rows)
- **Early stopping** — `n_estimators=2000`, `early_stopping_rounds=150`; `n_estimators` removed from Optuna search space
- **`aucpr` not usable for multiclass** — XGBoost issue #5662; use `mlogloss` for 60m model early stopping
- **GPU training** — `device='cuda'` + `tree_method='hist'` (correct XGBoost 2.x/3.x syntax); `n_jobs=1` required when `device='cuda'`; `_resolve_device()` falls back to CPU silently if no CUDA GPU found

## Current Performance (Phase 5, 2026-03-24)

| Model | CV Macro PR-AUC | Test Macro PR-AUC |
|-------|-----------------|-------------------|
| 60m severity | **0.565 ± 0.010** | 0.522 |
| 15m binary | pending (first full run — OOM fixes applied) | — |
| OVR severe | pending (first run) | — |

Per-class on test (60m model): `no_spike`=0.965, `moderate`=0.278, `severe`=0.324.
Alarm threshold: 0.70 → Precision=0.382 / Recall=0.332 / ~12.4 alarms/day.

Note: Performance numbers are from Phase 3/4 with 39 features. Phase 5 adds `cpu_per_task`,
drops `dow_sin`/`dow_cos`, and adds the OVR severe model — results pending the VM training run.

## Training Artifacts

All written to `--artifacts-dir` (default `data/full_run/`):

| File | Description |
|------|-------------|
| `cluster_agg.parquet` | Step 1 cache |
| `cluster_features.parquet` | Step 2 cache |
| `spike_thresholds.parquet` | Per-machine p95 + p99 thresholds |
| `models/spike/spike_model.json` | 60m XGBoost model |
| `models/spike/spike_config.json` | Threshold sweep, class rates, CV summary, hyperparams |
| `models/spike/feature_importance.csv` | XGBoost gain + SHAP importances |
| `models/spike/best_params.json` | Best Optuna hyperparameters |
| `models/spike_15m/spike_model.json` | 15m binary XGBoost model |
| `models/spike_15m/spike_config.json` | Config + best params |
| `models/spike_severe_ovr/spike_model.json` | OVR severe binary XGBoost model |
| `models/spike_severe_ovr/spike_config.json` | Config + best params |
| `run_config.json` | All CLI args + timestamp (reproducibility) |
