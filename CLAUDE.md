# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

**CPU spike prediction for datacenter scheduling.** Given a rolling window of per-machine CPU telemetry, the system predicts whether a CPU spike will occur in the next 60 minutes and how severe it will be. Output feeds a workload scheduler so it can act pre-emptively — before the spike materialises.

**Production scenario:** Every 5 minutes, read the last 120 min of per-machine CPU data (24 buckets), predict severity for each machine, feed predictions to scheduler.

## Directory Layout

```
Agentic_AI/
├── CLAUDE.md                       ← this file
├── pyproject.toml                  ← makes spike/ a pip-installable package
├── docker-compose.yml              ← API + GPU training services
├── scripts/
│   └── vm-setup.sh                 ← one-shot GPU VM provisioning (Docker + NVIDIA Container Toolkit)
├── spike/                          ← inference package (pip install -e .)
│   ├── __init__.py
│   ├── feature_engineer.py
│   ├── classifier.py
│   ├── predict.py                  ← CLI inference script
│   ├── api.py                      ← FastAPI service
│   ├── daemon.py
│   ├── Dockerfile                  ← inference API image (CPU-only)
│   └── requirements-inference.txt
├── training/
│   ├── train.py                    ← CLI orchestrator (3-step pipeline)
│   ├── Dockerfile.training         ← GPU training image (nvidia/cuda:12.4.1-runtime-ubuntu22.04)
│   ├── requirements-train.txt
│   └── adapters/
│       ├── README.md               ← cluster_agg interface contract
│       ├── google_cluster.py       ← Google Cluster Traces 2011 CSV → cluster_agg
│       └── zabbix.py               ← Zabbix 7.x API → cluster_agg
├── evaluation/
│   ├── evaluate_cross_domain.py    ← 3-phase cross-domain evaluation (PSI + PR-AUC)
│   └── baselines/
│       ├── evaluate_baselines.py   ← Persistence / EWMA / ARIMA baselines
│       └── *.md / *.json           ← Baseline results
├── tests/
│   ├── conftest.py
│   ├── test_preprocessor.py
│   ├── test_feature_engineer.py
│   ├── test_classifier.py
│   ├── test_train.py
│   ├── test_predict.py
│   ├── Dockerfile.test
│   └── requirements-test.txt
├── demo/
│   ├── demo.py                     ← Streamlit dashboard (4 tabs: Overview, Threshold, Calibration, SHAP)
│   └── requirements-demo.txt
└── data/
    ├── cluster_cpu_data.csv        ← Google Cluster Traces 2011 (278M rows, ~8GB)
    ├── download_cluster_data.py
    ├── full_run/                   ← production artifacts — K=2 baseline (default)
    └── experiments/                ← ablation runs (k1/, k3/, phase8_fft/, …)
```

All commands run from the **repo root** (`Agentic_AI/`).

## Commands

```bash
# Create and activate the local venv (first time only — run from repo root)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"   # installs spike + test deps (optuna, shap, pytest, …)

# Install spike package into an existing activated venv
pip install -e .

# Run the full test suite (~30 seconds)
.venv/bin/python3.12 -m pytest tests/ -v

# Run a single test file or a specific test class/function
.venv/bin/python3.12 -m pytest tests/test_feature_engineer.py -v
.venv/bin/python3.12 -m pytest tests/test_feature_engineer.py::TestStreakFeatures -v

# Run the full training pipeline from scratch (CPU — slow)
python training/train.py \
  --data-path data/cluster_cpu_data.csv \
  --artifacts-dir data/full_run \
  --tune-hyperparams --optuna-trials 30 \
  --n-estimators 4000

# Resume from Step N (skips earlier steps if cache exists)
python training/train.py \
  --data-path data/cluster_cpu_data.csv \
  --artifacts-dir data/full_run \
  --from-step 3 \
  --tune-hyperparams --optuna-trials 30 \
  --n-estimators 4000

# Cross-domain evaluation on a new dataset
python evaluation/evaluate_cross_domain.py \
  --agg data/your_domain/cluster_agg.parquet \
  --artifacts-dir data/full_run \
  --output data/your_domain_eval

# GPU training via Docker Compose (run from ~/spike on the VM)
docker compose --profile train run --rm --build train
# Uses OPTUNA_TRIALS=150, DEVICE=cuda from docker-compose.yml

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
  → [Step 1] training/adapters/google_cluster.py  → data/full_run/cluster_agg.parquet
  → [Step 2] spike/feature_engineer.py            → data/full_run/cluster_features.parquet
                                                     data/full_run/spike_thresholds.parquet
  → [Step 3] spike/classifier.py                  → data/full_run/models/spike/            (60m severity)
                                                     data/full_run/models/spike_15m/        (15m binary)
                                                     data/full_run/models/spike_severe_ovr/ (OVR severe)
```

**Directory convention:**
- `data/full_run/` = production model (K=2 default — never run K=1 here again)
- `data/experiments/<name>/` = ablations; pass `--artifacts-dir data/experiments/<name>` at train time

Use `--from-step N` to resume from any step. Use `--force` to clear a step's cache.

**Cache invalidation rules:**
- If `_FEATURE_COLS` or `_X_COLS` changed → delete `cluster_features.parquet` (triggers Step 2 re-run)
- If `_X_COLS` changed → also delete all `optuna.db` files (hyperparams tuned on old feature set are stale)
- `cluster_agg.parquet` is safe to reuse unless the preprocessor logic changes

## Model Architecture

Four XGBoost models, all trained on the same 40-feature set:

- **60m severity model** (`SpikeClassifier`, `multi:softprob`, 3 classes)
  Predicts: `no_spike` / `moderate` (p95 exceeded) / `severe` (p99 exceeded) in next 60 min.
  Primary metric: macro PR-AUC. Calibrated with isotonic regression on the val set.

- **15m / 30m / 45m binary models** (`BinarySpikeClassifier`, `binary:logistic`)
  Predicts: will any spike occur within the next N minutes?
  Provides imminence/timing signal alongside the 60m severity model.

- **OVR severe model** (`BinarySpikeClassifier`, `binary:logistic`) — **cascade Stage 2**
  Label: `spike_severe_ovr` — severe (class 2) vs moderate (class 1).
  Trained only on spike-positive rows (moderate + severe), raising the severe fraction from
  ~2.4% to ~19% so standard scale_pos_weight handles the imbalance without custom losses.
  At inference, Stage 1 (60m model) gates Stage 2: only rows where `p(any spike) >= 0.15`
  reach the OVR model. Outputs `p_severe_ovr`. Saved to `models/spike_severe_ovr/`.

## Data

**Dataset:** Google Cluster Traces 2011
- 278M raw rows → ~24M (machine_id × 5-min bucket) aggregates after Step 1
- 160 hours, 12,555 machines, 1,919 buckets per machine
- Split: Train 60% / Val 20% / Test 20% (chronological — no shuffle)

**Input schema (cluster_agg format):**

Any domain adapter that produces this schema can feed the full pipeline unchanged.

| Column | Type | Description |
|--------|------|-------------|
| `machine_id` | int64 | Unique machine identifier |
| `bucket` | int64 | 5-min bucket index (`unix_ts // 300`) |
| `time_us` | int64 | `bucket × 300_000_000` (microseconds) |
| `total_cpu` | float32 | Duration-weighted CPU load (fraction of 1 core) |
| `peak_cpu` | float32 | Peak CPU rate in the 5-min window |
| `total_mem` | float32 | Sum of canonical memory usage |
| `peak_mem` | float32 | Peak memory usage |
| `disk_io` | float32 | Max mean disk I/O time |
| `n_tasks` | int32 | Concurrent tasks in bucket |

See `training/adapters/README.md` for adapter writing instructions.

## Features (40 total)

- **Raw:** `total_cpu, peak_cpu, total_mem, peak_mem, disk_io, n_tasks`
- **Lags:** `cpu_lag_{1,12,24}`
- **Trend:** `cpu_ewma_{6,24}, cpu_delta_{1,2}, cpu_rolling_std_6`
- **Load ratio:** `cpu_per_task`
- **Machine-relative (p95):** `cpu_vs_p95, cpu_vs_p95_delta, cpu_vs_p95_slope_3, cpu_vs_p95_slope_6, time_to_p95_3, peak_cpu_vs_p95, spike_now, spike_in_last_{1,3,6}, time_since_last_spike, cpu_spike_rate_24`
- **Machine-relative (p99):** `spike_severe_now, cpu_vs_p99, peak_cpu_vs_p99, band_position, band_width, spike_severe_in_last_{1,3,6}`
- **Cluster:** `cluster_cpu_p90, machine_rank_in_cluster, task_dominance`
- **Time:** `hour_sin, hour_cos`

Per-machine p95 and p99 thresholds are computed from training data only (no leakage).
Minimum 10% gap between p99 and p95 is enforced to ensure a meaningful moderate band.

**Dropped features (with reason):**
- `dow_sin`/`dow_cos`: highest SHAP (0.356) but near-zero gain (0.011). Only 7 days of data
  → 23 samples per label → confirmed temporal confound with the Google 2011 trace week.
- `current_spike_streak`, `max_spike_streak_24h`, `current_severe_streak`: two separate runs
  (stale + fresh Optuna) both gave 0.553 cal vs 0.574 baseline — confirmed −0.021 regression.
  Add noise/collinearity on top of the existing `spike_in_last_{1,3,6}` history features.

## Leakage Firewall

All spike-history features are derived from `exc = spike_now.shift(1)` — the previous bucket's
spike status, not the current one. This is the central leakage guard. Every feature that uses
spike history uses this shifted variable or a further roll-up of it.

Labels (`severity_in_60m`, `spike_in_15m`, etc.) look forward from the *next* bucket — the
`shift(1)` on the input side and the future window on the label side are in opposite directions
and cannot leak.

## K-of-N Label Ablation

`training/train.py` accepts `--min-future-windows K` (**default K=2**):

- **K=1:** `severity_in_60m = moderate` if *any* of the next 12 windows exceeds p95.
- **K=2 (default):** requires at least 2 of the next 12 windows to exceed p95. Production baseline.
- **K=3:** requires at least 3. More persistent spikes only.

Binary labels (`spike_in_15m`, `spike_in_30m`, `spike_in_45m`) always use K=1.

`training/Dockerfile.training` has `ENV MIN_FUTURE_WINDOWS=2` (K=2 is the production default).

## Key Design Constraints

- **Chronological splits only** — no random shuffle anywhere in the pipeline
- **Walk-forward CV** — 5 folds, expanding window, 12-bucket gap (= 1 horizon) between train and val
- **Thresholds from training data only** — p95/p99 computed on train buckets, never val/test
- **Monotone constraints** — enabled for binary models only (XGBoost `binary:logistic`); disabled for
  `multi:softprob` (undefined semantics for K-class softmax with per-feature constraints)
- **Macro PR-AUC** — primary metric for 60m model (equal weight to all classes including rare severe)
- **60m Optuna:** 150 trials via `docker-compose.yml` (`OPTUNA_TRIALS=150`)
- **Binary Optuna:** capped at `min(n_trials, 75)` — binary:logistic converges faster
- **Optuna inner split capped at 2M rows** — OOM prevention (full inner split = 7M+ rows for binary)
- **Early stopping** — `n_estimators=4000`, `early_stopping_rounds=150`; `n_estimators` not in Optuna
  search space; programmatic `run()` default is 2000 (tests use this to stay fast)
- **`aucpr` not usable for multiclass** — XGBoost issue #5662; use `mlogloss` for 60m early stopping
- **GPU training** — `device='cuda'` + `tree_method='hist'` (correct XGBoost 2.x/3.x syntax);
  `n_jobs=1` required when `device='cuda'`; `_resolve_device()` falls back to CPU if no GPU found

## Performance History

| Model | Test PR-AUC | Test ROC-AUC | Alarm P/R | Notes |
|-------|-------------|--------------|-----------|-------|
| 60m severity | **0.547** (cal) | 0.849 | 0.429/0.309 @ 0.25 | K=2, 40 features, 150-trial Optuna |
| 15m binary | **0.575** | 0.912 | 0.605/0.496 @ 0.80 | |
| 30m binary | **0.564** | 0.878 | 0.569/0.497 @ 0.70 | |
| 45m binary | **0.563** | 0.860 | 0.535/0.524 @ 0.65 | |
| OVR severe (cascade Stage 2) | **0.543** (spike-positive subset) | 0.761 | 0.425/0.718 @ 0.50 | Severe ~19% of training rows |

OVR severe note: 0.543 is measured on spike-positive test rows (base rate ~25%), not the full test
set. The baseline trained on all rows scored 0.328 PR-AUC at 2.4% positive rate — not comparable.

## Training Artifacts

All written to `--artifacts-dir` (default `data/full_run/`):

| File | Description |
|------|-------------|
| `cluster_agg.parquet` | Step 1 cache — raw CSV aggregated to 5-min buckets |
| `cluster_features.parquet` | Step 2 cache — 40 features per machine-bucket |
| `spike_thresholds.parquet` | Per-machine p95 + p99 thresholds (from training data only) |
| `models/spike/spike_model.json` | 60m XGBoost model weights |
| `models/spike/spike_model.meta.json` | Feature column list + XGBoost version (needed for correct column order on load) |
| `models/spike/spike_config.json` | Threshold sweep, class rates, CV summary, hyperparams |
| `models/spike/cv_results.csv` | Per-fold metrics from walk-forward CV |
| `models/spike/feature_importance.csv` | XGBoost gain + SHAP importances |
| `models/spike/best_params.json` | Best Optuna hyperparameters |
| `models/spike/calibrators.pkl` | Per-class isotonic calibrators (fitted on val set) |
| `models/spike_15m/spike_model.json` | 15m binary XGBoost model |
| `models/spike_15m/spike_config.json` | Config + CV summary |
| `models/spike_severe_ovr/spike_model.json` | OVR severe binary XGBoost model |
| `models/spike_severe_ovr/spike_config.json` | Config + CV summary |
| `run_config.json` | All CLI args + timestamp (reproducibility) |

## Tip
Codex will review your output once you are done.
