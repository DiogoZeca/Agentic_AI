# Development Log — CPU Spike Prediction Model

**Last updated:** 2026-03-24
**Current status:** Phase 5 complete. Code ready for GPU VM training run. Test suite: 244 passed, 4 skipped, ~17 seconds. Next: full VM run with 150 Optuna trials.

---

## Project Goal

Build a model that **predicts CPU spike severity before it happens** so a datacenter scheduler can act pre-emptively — not reactively.

**Production scenario (every 5 minutes):**
1. Read the last 120 min of per-machine CPU telemetry (24 × 5-min buckets)
2. Predict severity for the next 60 min: `no_spike` / `moderate` / `severe`
3. Feed predictions to the scheduler → it defers tasks, adds capacity, or triggers emergency failover based on severity

**Architecture decision (final):** XGBoost only. TFT/neural approaches were dropped — latency constraints in the target system make XGBoost the only viable option. Goal is to make XGBoost produce richer, more actionable outputs, not to switch frameworks.

---

## Dataset

**Google Cluster Traces 2011** (`cluster_cpu_data.csv`)
- 278M raw rows → ~24M (machine_id × 5-min bucket) aggregates after Step 1
- 160 hours, 12,555 machines, 1,919 buckets per machine
- Chronological split: **Train 60% / Val 20% / Test 20%** (no shuffle — ever)

---

## Pipeline

```
cluster_cpu_data.csv
  → [Step 1] spike_preprocessor.py      → cluster_agg.parquet
  → [Step 2] spike_feature_engineer.py  → cluster_features.parquet
                                           spike_thresholds.parquet (p95 + p99 per machine)
  → [Step 3] train_spike_classifier.py  → models/spike/            (60m severity, multi:softprob)
                                           models/spike_15m/        (15m binary, binary:logistic)
                                           models/spike_severe_ovr/ (OVR severe, binary:logistic)
  → predict_spike.py / spike_api.py     → per-machine JSON predictions
```

Use `--from-step N` to resume from any step. Use `--force` to clear a step's cache.

---

## Model Architecture (current)

### 60m Severity Model (`SpikeClassifier`, `multi:softprob`)
- **Labels:** `severity_in_60m` ∈ {0=no_spike, 1=moderate, 2=severe}
- **Thresholds:** per-machine p95 (moderate boundary) and p99 (severe boundary), computed from training data only
- **Balancing:** `compute_sample_weight('balanced', y)` — class 2 gets ~85x weight
- **Primary metric:** Macro PR-AUC (equal weight to all 3 classes including rare severe)
- **Monotone constraints:** Disabled — semantically undefined for `multi:softprob`
- **Early stopping:** `n_estimators=2000`, `early_stopping_rounds=150`, metric=`mlogloss`

### 15m Binary Model (`BinarySpikeClassifier`, `binary:logistic`)
- **Label:** `spike_in_15m` ∈ {0, 1} — any spike in next 15 min
- **Balancing:** `scale_pos_weight` computed per fold
- **Monotone constraints:** Active (cpu_vs_p95, spike history features)
- **Early stopping:** `n_estimators=2000`, `early_stopping_rounds=150`, metric=`aucpr`
- **Status:** OOM fixes applied (inner split capped at 2M rows, trials at 30). Pending first successful run.

### OVR Severe Model (`BinarySpikeClassifier`, `binary:logistic`)
- **Label:** `spike_severe_ovr` = `(severity_in_60m == 2).astype(float32)` (NaN where label is NaN)
- **Balancing:** `scale_pos_weight` ≈ 99 (count non-severe / count severe)
- **Monotone constraints:** Active (binary model can use them)
- **Optuna trials:** 30 (binary converges faster)
- **Saved to:** `models/spike_severe_ovr/`
- **Inference output:** `p_severe_ovr` field in prediction JSON

---

## Features (37 total — Phase 5)

| Category | Features |
|----------|----------|
| **Raw** | `total_cpu, peak_cpu, total_mem, peak_mem, disk_io, n_tasks` |
| **Lags** | `cpu_lag_{1,12,24}` |
| **Trend** | `cpu_ewma_{6,24}, cpu_delta_{1,2}, cpu_rolling_std_6` |
| **Load ratio** | `cpu_per_task` |
| **Machine-relative p95** | `cpu_vs_p95, cpu_vs_p95_delta, peak_cpu_vs_p95, spike_now, spike_in_last_{1,3,6}, time_since_last_spike, cpu_spike_rate_24` |
| **Machine-relative p99** | `spike_severe_now, cpu_vs_p99, peak_cpu_vs_p99, band_position, band_width, spike_severe_in_last_{1,3,6}` |
| **Cluster** | `cluster_cpu_p90, machine_rank_in_cluster, task_dominance` |
| **Time** | `hour_sin, hour_cos` |

**Dropped in Phase 5:** `dow_sin`, `dow_cos` — highest SHAP=0.356 but gain=0.011; only 7 days of data (23 samples per day-of-week label); confirmed temporal confound. `hour_sin`/`hour_cos` already capture intra-day cycles.

**Added in Phase 5:** `cpu_per_task = total_cpu / max(n_tasks, 1)` — captures per-task load; task-normalized CPU is a meaningful scheduling signal.

**Named time-interval constants (Phase 5):** `_ROLLING_STD_WINDOW=6`, `_CPU_DELTA_LAGS=(1,2)`, `_SPIKE_HISTORY_WINDOWS=(1,3,6)`, `_SEVERE_HISTORY_WINDOWS=(1,3,6)` — module-level constants in `spike_feature_engineer.py` replacing inline literals.

Per-machine p95 and p99 thresholds are computed from **training buckets only** (leakage-free). Min 10% gap between p99 and p95 enforced.

---

## Current Performance (Phase 5 — 2026-03-24)

### 60m Severity Model (Phase 3/4 numbers — Phase 5 run pending)

| Metric | CV Walk-forward (5-fold) | Test set |
|--------|--------------------------|----------|
| **Macro PR-AUC** | **0.565 ± 0.010** | **0.522** |
| Macro ROC-AUC | 0.816 ± 0.016 | 0.798 |
| Weighted F1 | 0.741 ± 0.010 | 0.629 |
| `no_spike` PR-AUC | — | 0.965 |
| `moderate` PR-AUC | — | 0.278 (skill 0.210) |
| `severe` PR-AUC | — | 0.324 (skill 0.301) ← **below 0.40 target** |
| Alarm @ threshold 0.70 | — | Precision=0.382 / Recall=0.332 / ~12.4 alarms/day |

**Best hyperparams (50 Optuna trials):**
`n_estimators=586, max_depth=3, learning_rate=0.131, min_child_weight=3, gamma=0.384`

**Top features by SHAP:** `dow_sin`=0.356 (now dropped), `cpu_vs_p95`=0.230, `cpu_vs_p99`=0.088
**Top features by gain:** `spike_severe_now`=0.119, `spike_in_last_6`=0.097, `time_since_last_spike`=0.087

**Key findings:**
- p99 features improved `moderate` PR-AUC by +25% relative (0.222 → 0.278); `severe` barely moved — bottleneck is loss function optimization pressure, not features
- `dow_sin` SHAP=0.356 (highest of all 39 features) but gain=0.011 — confirmed temporal confound; dropped in Phase 5
- CV-to-test gap (0.043) explained by test window prevalence shift: severe 5.0% → 3.3% in anomalous Google Cluster days 21–25

### 15m Binary Model
Pending first successful run. OOM fixes applied.

### OVR Severe Model
Pending first run (new in Phase 5).

---

## Key Design Decisions (standing)

| Decision | Status | Rationale |
|----------|--------|-----------|
| XGBoost only (no TFT) | ✅ Final | Latency constraints; TFT dropped |
| Chronological 60/20/20 split | ✅ Final | No random shuffle — temporal data |
| Walk-forward CV, gap=12 buckets | ✅ Final | 60-min gap = prediction horizon, prevents label leakage |
| Per-fold `compute_sample_weight` (multiclass) / `scale_pos_weight` (binary) | ✅ Final | No class-ratio leakage from future folds |
| P95 + P99 thresholds (per machine, training-only) | ✅ Final | Leakage-free; min 10% gap enforced |
| Macro PR-AUC as primary metric | ✅ Final | Correct for imbalanced multiclass; equal weight to rare severe |
| Alarm threshold tuned on val set only | ✅ Final | Never on test |
| Monotone constraints | Binary only | Disabled for `multi:softprob` — semantically undefined in softmax |
| `aucpr` early stopping | Binary only | XGBoost issue #5662: `aucpr` is binary-only; use `mlogloss` for multiclass |
| `XGBoostPruningCallback` + native early stopping | ❌ Conflict | Race condition; native early stopping preferred exclusively |
| `n_estimators` in Optuna | ❌ Removed Phase 5 | Replaced with `n_estimators=2000` + `early_stopping_rounds=150` |
| `device='cuda'` + `tree_method='hist'` | ✅ Phase 5 | Correct XGBoost 2.x/3.x GPU syntax; `gpu_hist` deprecated since 2.0 |
| `n_jobs=1` when device='cuda' | ✅ Phase 5 | XGBoost requirement; CPU threading conflicts with CUDA |

---

## OOM Prevention Fixes (applied 2026-03-24)

Applied to `train_spike_classifier.py`:
- `del clf; gc.collect()` after each Optuna trial
- Binary inner split capped at 2M rows (`_MAX_OPTUNA_ROWS_BINARY`)
- Binary Optuna trials hard-capped at 30 (`_MAX_BINARY_TRIALS`)
- RAM usage logged before binary model training loop (psutil)

---

## Phase 5 — Pre-VM Run Improvements (COMPLETE 2026-03-24)

### 5a. `dow_sin` / `dow_cos` Ablation ✅
Dropped from `_X_COLS`. Only 7 days of data (23 samples per day-of-week label); `dow_sin` had highest SHAP=0.356 but near-zero gain=0.011 — confirmed temporal confound. `hour_sin`/`hour_cos` already capture intra-day cycles.

### 5b. OVR Dedicated Severe Classifier ✅
`spike_severe_ovr` label added: `(severity_in_60m == 2).astype(float32)` (NaN-safe using `np.float32("nan")`).
`_train_binary_horizon()` reused with `label_col="spike_severe_ovr"`.
Saved to `models/spike_severe_ovr/`. Loaded optionally in `predict_spike.py`, outputs `p_severe_ovr`.

### 5c. Early Stopping Replacing Optuna `n_estimators` ✅
`n_estimators` removed from Optuna search space. Fixed at 2000 with `early_stopping_rounds=150`.
Parameters threaded through `run()` → `_train_binary_horizon()` → `_run_optuna_search()` with production defaults; tests pass `n_estimators=50, early_stopping_rounds=10` (test suite: 244 passed, ~17 seconds).

### 5d. Feature Addition: `cpu_per_task` ✅
`cpu_per_task = total_cpu / max(n_tasks, 1)` added to `_engineer_machine()` in `spike_feature_engineer.py`. Feature count: 39 → 37 (removed 2 dow features, added 1).

### 5e. Named Time-Interval Constants ✅
`_ROLLING_STD_WINDOW`, `_CPU_DELTA_LAGS`, `_SPIKE_HISTORY_WINDOWS`, `_SEVERE_HISTORY_WINDOWS` added as module-level constants in `spike_feature_engineer.py`. All inline literals replaced.

### 5f. GPU-Ready Dockerfile + Docker Compose ✅
`Dockerfile.training` rebased to `nvidia/cuda:12.4.1-runtime-ubuntu24.04`.
`docker-compose.yml` `train` service: `DEVICE=cuda` env + `deploy.resources.reservations.devices` GPU block.
`scripts/vm-setup.sh`: one-shot script — Docker Engine (APT, not Snap) + NVIDIA Container Toolkit.

### Pending for VM run
- 150 Optuna trials on 60m model (currently 30 default)
- First full run of 15m binary model (OOM fixed)
- First run of OVR severe model

---

## Future Work (Phase 6+)

- **Probability calibration** — Isotonic regression fitted on val set after final model training. Both binary models (15m, OVR severe) and the multiclass model need calibration. Note: OVR with `scale_pos_weight` inflates probabilities upward; multiclass p_severe is compressed downward. Calibrate before any ensembling.
- **Ensembling OVR + multiclass** — If OVR PR-AUC > multiclass by >10%, use `p_severe_ovr` as alarm signal. If 5–10% difference, weighted average. If <5%, keep multiclass alone.
- **Lookback window extension** — Current 120 min (2:1 to prediction horizon) is standard. 240 min possible but research shows diminishing returns rapidly; not a priority.

---

## VM Deployment

Training is containerised and GPU-ready. To run on an Ubuntu 22.04/24.04 GPU VM:

```bash
# 1. Transfer code and data to VM (run locally)
rsync -avz --partial --inplace AIModel/ user@vm-host:~/spike/AIModel/
rsync -avz --partial --inplace AIModel/data/cluster_cpu_data.csv user@vm-host:~/spike/AIModel/data/
rsync -avz --partial --inplace scripts/ docker-compose.yml user@vm-host:~/spike/

# 2. SSH into VM and run one-shot setup
ssh user@vm-host
cd ~/spike && bash scripts/vm-setup.sh

# 3. Run GPU training (artifacts written to ./AIModel/data/full_run/ on host)
docker compose --profile train run --rm --build train

# 4. Monitor
docker logs -f spike-train

# 5. Sync artifacts back locally after training
rsync -avz --partial user@vm-host:~/spike/AIModel/data/full_run/ AIModel/data/full_run/

# 6. Start API
docker compose up --build
curl http://localhost:8000/ready
```

GPU notes:
- `scripts/vm-setup.sh` installs Docker Engine (APT) + NVIDIA Container Toolkit. Docker must NOT be installed via Ubuntu Snap — Snap sandbox blocks `/dev/nvidia*`.
- `DEVICE=cuda` is set in docker-compose; `_resolve_device()` falls back to CPU silently if no GPU found.
- `capabilities: [gpu]` in docker-compose is mandatory — omitting it silently ignores the GPU.

---

## File Status

| File | Status |
|------|--------|
| `spike_preprocessor.py` | ✅ Complete |
| `spike_feature_engineer.py` | ✅ Complete — p95+p99 thresholds, severity label, 37 features, named constants |
| `spike_classifier.py` | ✅ Complete — `SpikeClassifier` (multi:softprob) + `BinarySpikeClassifier`; GPU-aware |
| `train_spike_classifier.py` | ✅ Complete — OVR severe model; early stopping; parameterized n_estimators/ESR |
| `predict_spike.py` | ✅ Complete — 3-model inference + `predict_with_artifacts()` for API |
| `spike_api.py` | ✅ Complete — FastAPI (lifespan, /health, /ready, /predict) |
| `Dockerfile` | ✅ Complete — inference API image (CPU-only) |
| `Dockerfile.test` | ✅ Complete — test runner image |
| `Dockerfile.training` | ✅ Complete — GPU training image (nvidia/cuda:12.4.1-runtime-ubuntu24.04) |
| `scripts/vm-setup.sh` | ✅ Complete — Docker Engine (APT) + NVIDIA Container Toolkit setup |
| `requirements-inference.txt` | ✅ Complete — exact pins for inference image |
| `requirements-train.txt` | ✅ Complete — extends inference + optuna/shap/psutil |
| `docker-compose.yml` | ✅ Complete — API + GPU train + test services |

---

## Training Commands (quick reference)

```bash
# All from AIModel/

# From step 3 only — fastest (skip preprocessing and feature engineering)
nohup .venv/bin/python3 train_spike_classifier.py \
  --data-path data/cluster_cpu_data.csv \
  --artifacts-dir data/full_run \
  --from-step 3 --tune-hyperparams --optuna-trials 30 > data/run_log.txt 2>&1 & echo "PID: $!"

# From step 2 — re-engineer features, then retrain (needed when features change)
nohup .venv/bin/python3 train_spike_classifier.py \
  --data-path data/cluster_cpu_data.csv \
  --artifacts-dir data/full_run \
  --from-step 2 --force --tune-hyperparams --optuna-trials 30 > data/run_log.txt 2>&1 & echo "PID: $!"

# Monitor
tail -f data/run_log.txt

# Run tests
.venv/bin/python3 -m pytest tests/ -v

# Full VM run via Docker (GPU, 150 trials)
cd ..   # Agentic_AI/
docker compose --profile train run --rm --build train
```

---

## Phase History

| Phase | What | When | CV PR-AUC |
|-------|------|------|-----------|
| Baseline | Binary `binary:logistic`, 32 features, no Optuna | 2026-03-01 | 0.634 |
| Phase 1 | Optuna 50 trials + monotone constraints | 2026-03-10 | 0.640 |
| Phase 2 | Feature engineering (+7 features: spike rates, peak_cpu_vs_p95, trend) | 2026-03-21 | 0.640 (ceiling) |
| Phase 3 | 3-class severity (`multi:softprob`) + p99 thresholds + 8 p99 features | 2026-03-24 | 0.565 macro |
| Phase 4 | Multi-horizon (15m binary sibling model) | 2026-03-24 | crashed (OOM) |
| Phase 5 | dow ablation, OVR severe model, early stopping, cpu_per_task, named constants, GPU deployment | 2026-03-24 | pending VM run |
