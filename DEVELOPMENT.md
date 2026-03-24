# Development Log — CPU Spike Prediction Model

**Last updated:** 2026-03-24
**Current status:** Phase 3 (severity tiers) + Phase 4 (multi-horizon) complete and deployed. 15m binary model crashed during Optuna (OOM) — OOM fixes applied, restart pending. Deployability complete. Next: Phase 5 pre-run improvements before VM training run.

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
  → [Step 3] train_spike_classifier.py  → models/spike/        (60m severity, multi:softprob)
                                           models/spike_15m/    (15m binary, binary:logistic)
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

### 15m Binary Model (`BinarySpikeClassifier`, `binary:logistic`)
- **Label:** `spike_in_15m` ∈ {0, 1} — any spike in next 15 min
- **Balancing:** `scale_pos_weight` computed per fold
- **Monotone constraints:** Active (cpu_vs_p95, spike history features)
- **Status:** Crashed during Optuna (OOM). OOM fixes applied. Needs restart.

---

## Features (39 total)

| Category | Features |
|----------|----------|
| **Raw** | `total_cpu, peak_cpu, total_mem, peak_mem, disk_io, n_tasks` |
| **Lags** | `cpu_lag_{1,12,24}` |
| **Trend** | `cpu_ewma_{6,24}, cpu_delta_{1,2}, cpu_rolling_std_6` |
| **Machine-relative p95** | `cpu_vs_p95, cpu_vs_p95_delta, peak_cpu_vs_p95, spike_now, spike_in_last_{1,3,6}, time_since_last_spike, cpu_spike_rate_24` |
| **Machine-relative p99** | `spike_severe_now, cpu_vs_p99, peak_cpu_vs_p99, band_position, band_width, spike_severe_in_last_{1,3,6}` |
| **Cluster** | `cluster_cpu_p90, machine_rank_in_cluster, task_dominance` |
| **Time** | `hour_sin, hour_cos, dow_sin, dow_cos` ⚠️ *dow_sin/dow_cos: ablation pending — see Phase 5* |

Per-machine p95 and p99 thresholds are computed from **training buckets only** (leakage-free). Min 10% gap between p99 and p95 enforced.

---

## Current Performance (Phase 3/4 — 2026-03-24)

### 60m Severity Model

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

**Top features by SHAP:** `dow_sin`=0.356 ⚠️, `cpu_vs_p95`=0.230, `cpu_vs_p99`=0.088
**Top features by gain:** `spike_severe_now`=0.119, `spike_in_last_6`=0.097, `time_since_last_spike`=0.087

**Key findings:**
- p99 features improved `moderate` PR-AUC by +25% relative (0.222 → 0.278); `severe` barely moved — bottleneck is loss function optimization pressure, not features
- `dow_sin` SHAP=0.356 (highest of all 39 features) but gain=0.011 — high SHAP + low gain means it's acting as a **global bias corrector through interaction effects**, not a splitting feature. With only 7 days of data (~23 samples per day-of-week label), this is a temporal confound risk. Ablation pending.
- CV-to-test gap (0.043) explained by test window prevalence shift: severe 5.0% → 3.3% in anomalous Google Cluster days 21–25

### 15m Binary Model
Crashed during Optuna (OOM — 7.16M inner split × 50 trials). OOM prevention fixes applied (see below). Status: restart pending.

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
| `aucpr` early stopping | ❌ Not usable | XGBoost issue #5662: `aucpr` is binary-only; use `mlogloss` for multiclass early stopping |
| `XGBoostPruningCallback` + native early stopping | ❌ Conflict | Race condition; use one or the other — native early stopping preferred |

---

## OOM Prevention Fixes (applied 2026-03-24)

Applied to `train_spike_classifier.py`:
- `del clf; gc.collect()` after each Optuna trial
- Binary inner split capped at 2M rows (`_MAX_OPTUNA_ROWS_BINARY`)
- Binary Optuna trials hard-capped at 30 (`_MAX_BINARY_TRIALS`)
- RAM usage logged before binary model training loop (psutil)

---

## Phase 5 — Pre-VM Run Improvements (PLANNED)

These are the changes to apply before the next full training run on the VM.

### 5a. `dow_sin` / `dow_cos` Ablation (Step 2+3, quick)

**Problem:** `dow_sin` has the highest SHAP (0.356) but near-zero gain (0.011). High SHAP + low gain means it's acting as a global bias corrector via interaction effects — not a genuine splitting signal. With only 7 days of data, 23 samples per day-of-week label is insufficient to distinguish genuine weekly periodicity from noise. Published datacenter workload research confirms day-of-week is a secondary effect dominated by intra-day cycles (which `hour_sin`/`hour_cos` already capture).

**Action:** Drop `{dow_sin, dow_cos}` from `_X_COLS`, retrain, compare CV Macro PR-AUC.
- If CV drops < 1%: features are noise → remove permanently
- If CV drops > 3%: genuine signal → keep with cautious interpretation
- If test PR-AUC improves while CV stays flat: confirmed temporal confound → remove

**Cost:** One Step 2+3 run (~10 min).

---

### 5b. OVR Dedicated Severe Classifier (new model, additive)

**Problem:** `multi:softprob`'s failure mode for 1% classes is well-documented: softmax distributes gradient signal proportionally across all 3 classes. Class 2 (1%) gets ~1% of total gradient pressure even with `compute_sample_weight('balanced')`. The 85x weight helps but fights against softmax normalization.

**Solution:** Train a dedicated `binary:logistic` OVR model alongside the existing 3-class model:
- Label: `(severity_in_60m == 2)` — severe vs rest
- `scale_pos_weight` = count(non-severe) / count(severe) ≈ 99
- Same feature set, same train/val/test split
- Monotone constraints re-enabled (binary model can use them again)
- Its own Optuna tuning (30 trials — binary converges faster)
- Outputs: `p_severe_ovr` — probability of severe event

**Where the OVR model lives:** Saved to `models/spike_severe_ovr/` (sibling of `models/spike/`). Loaded by `predict_spike.py` alongside the existing models.

**Decision rule for alarm signal (empirical — evaluated after training):**
- Measure severe-class PR-AUC of OVR model vs multiclass model on test set
- Apply isotonic regression calibration to both (scale_pos_weight inflates OVR probabilities; softmax compresses multiclass p_severe)
- If OVR PR-AUC > multiclass by **>10%**: use `p_severe_ovr` as the alarm signal
- If difference is **5–10%**: weighted average: `p_alarm = α × p_severe_ovr + (1-α) × p_severe`, where α ∝ their validation PR-AUC ratio
- If difference is **<5%**: keep multiclass alone, OVR adds complexity for negligible gain

**Expected gain:** Published results on 1% minority class show 10–25% PR-AUC improvement from OVR decomposition vs native multiclass.

---

### 5c. Early Stopping Replacing Optuna `n_estimators`

**Problem:** Currently `n_estimators` is an Optuna parameter (range 200–1200). This means each trial trains a fixed number of trees regardless of convergence — both undertrained and overtrained trials exist in the search.

**Solution:** Fix `n_estimators=2000`, use `early_stopping_rounds=150` (=2000/13, standard rule). XGBoost stops at the true convergence point. Remove `n_estimators` from Optuna search space.

**Early stopping metric:** Use `mlogloss` — NOT `aucpr` (confirmed broken for multiclass in XGBoost, issue #5662). `mlogloss` is more stable and rewards improvements in probability estimates for minority classes.

**Benefits:** 30–50% faster per trial; finds true convergence point; fewer Optuna dimensions (cleaner search).

---

### 5d. Stronger Class Weights for Severe

**Problem:** `compute_sample_weight('balanced')` gives 85x weight to class 2. Research shows this is often insufficient for 1% classes and that manual tier-based weights can outperform 'balanced' when tuned on CV.

**Proposed:** Manual tier weights as an Optuna parameter:
- `w0 = 1.0` (fixed, no_spike is the reference)
- `w1 = trial.suggest_float('w_moderate', 5.0, 30.0)`
- `w2 = trial.suggest_float('w_severe', 50.0, 200.0)`

This adds 2 Optuna dimensions but gives the search direct control over the class pressure on the severe class.

**Alternative (simpler):** Fixed multiplier on top of 'balanced': `w_severe *= 2` (i.e., 170x instead of 85x). Cheaper to implement, test this first.

---

### 5e. Feature Addition: `cpu_per_task`

**Rationale:** Current features treat `total_cpu` and `n_tasks` independently. `cpu_per_task = total_cpu / max(n_tasks, 1)` captures per-task load — if task count doubles but CPU stays flat, each task is half as loaded and spike risk drops. Published workload scheduling research identifies task-normalized CPU as a meaningful scheduling signal.

**Cost:** Computed inside `_engineer_machine()`, one line. Requires Step 2 cache clear.

---

### 5f. More Optuna Trials on VM (150 for 60m model)

Current 50 trials is underexplored. 150 trials covers the 6–7 hyperparameter search space with Bayesian TPE to near-convergence. Diminishing returns typically appear after 100–120 trials for this dimensionality.

**Hyperparameter search space to focus on (in order of leverage for rare class PR-AUC):**
1. `min_child_weight` [1–10] — highest impact; prevents extreme predictions from tiny severe-class leaf nodes
2. `gamma` [0.2–1.5] — minimum loss reduction to split; forces confident splits on minority class
3. `reg_lambda` [0.5–5.0] — L2 regularization; multiclass needs stronger L2 than binary (3 trees/round = more degrees of freedom)
4. `learning_rate` [0.05–0.15]
5. `subsample` [0.6–0.9]
6. `colsample_bytree` [0.6–1.0]
7. `max_depth` [3–6] — current 3 is defensible; try up to 6

---

### 5g. 15m Binary Model Restart

OOM fixes already applied. Just needs a restart from Step 3:
```bash
nohup .venv/bin/python3 train_spike_classifier.py \
  --data-path data/cluster_cpu_data.csv \
  --artifacts-dir data/full_run \
  --from-step 3 \
  --tune-hyperparams --optuna-trials 30 > data/rerun_log_15m.txt 2>&1 & echo "PID: $!"
```

---

### Priority Order for the VM Run

| # | Change | Files affected | Requires Step | Expected impact |
|---|--------|---------------|---------------|-----------------|
| 1 | `dow_sin`/`dow_cos` ablation | `spike_feature_engineer.py` | Step 2+3 | Reduces confound risk; cheap test |
| 2 | OVR severe model | `spike_classifier.py`, `train_spike_classifier.py`, `predict_spike.py` | Step 3 | Largest expected gain for severe PR-AUC |
| 3 | Early stopping replaces Optuna `n_estimators` | `train_spike_classifier.py` | Step 3 | Cleaner convergence, faster per trial |
| 4 | 150 Optuna trials + weight tuning | `train_spike_classifier.py` | Step 3 | More thorough hyperparameter search |
| 5 | `cpu_per_task` feature | `spike_feature_engineer.py` | Step 2+3 | Low-effort, theoretically sound |
| 6 | 15m binary restart | — | Step 3 | Completes the 2-model architecture |

Changes 1 and 5 both touch `spike_feature_engineer.py` (Step 2) so they're batched into one Step 2 run. Changes 2–4 and 6 are all Step 3.

---

## Future Work (Phase 6+)

- **Probability calibration** — Isotonic regression fitted on val set after final model training. Both binary models (15m, OVR severe) and the multiclass model need calibration. Note: OVR with `scale_pos_weight` inflates probabilities upward; multiclass p_severe is compressed downward. Calibrate before any ensembling.
- **`dow_sin` investigation** — If ablation shows genuine signal, investigate why. Is it capturing Google Cluster's specific trace week? Could be training-dataset artifact.
- **Lookback window extension** — Current 120 min (2:1 to prediction horizon) is standard. 240 min is possible but research shows diminishing returns rapidly; not a priority.

---

## VM Deployment

Intention: deploy training to an internal datacenter VM for a full 24h run with 150 Optuna trials.

```bash
# 1. Transfer code and data to VM (run locally)
rsync -avz --partial --inplace AIModel/ user@vm-host:~/spike/AIModel/
rsync -avz --partial --inplace AIModel/data/cluster_cpu_data.csv user@vm-host:~/spike/AIModel/data/

# 2. SSH into VM and run setup
ssh user@vm-host
cd ~/spike && bash setup_vm.sh

# 3. Monitor
tmux attach -t train   # Ctrl-B then D to detach safely
tail -f AIModel/data/rerun_log.txt

# 4. Sync artifacts back locally after training
rsync -avz --partial user@vm-host:~/spike/AIModel/data/full_run/ AIModel/data/full_run/

# 5. Start API
docker compose up --build
curl http://localhost:8000/ready
```

The API container (`restart: unless-stopped`) survives VM reboots as long as the Docker daemon starts on boot.

---

## File Status

| File | Status |
|------|--------|
| `spike_preprocessor.py` | ✅ Complete |
| `spike_feature_engineer.py` | ✅ Complete — p95+p99 thresholds, severity label, 39 features |
| `spike_classifier.py` | ✅ Complete — `SpikeClassifier` (multi:softprob) + `BinarySpikeClassifier` |
| `train_spike_classifier.py` | ✅ Complete — OOM fixes; binary trials capped at 30 |
| `predict_spike.py` | ✅ Complete — 2-model inference + `predict_with_artifacts()` for API |
| `spike_api.py` | ✅ Complete — FastAPI (lifespan, /health, /ready, /predict) |
| `Dockerfile` | ✅ Updated — libgomp1, requirements-inference.txt |
| `Dockerfile.test` | ✅ Updated — libgomp1, requirements-train.txt |
| `Dockerfile.training` | ✅ New — training image for VM |
| `setup_vm.sh` | ✅ New — VM setup + tmux launcher |
| `requirements-inference.txt` | ✅ New — exact pins for inference image |
| `requirements-train.txt` | ✅ New — extends inference + optuna/shap/psutil |

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

# Full VM run (150 trials, from step 2)
nohup .venv/bin/python3 train_spike_classifier.py \
  --data-path data/cluster_cpu_data.csv \
  --artifacts-dir data/full_run \
  --from-step 2 --force --tune-hyperparams --optuna-trials 150 > data/full_run_log.txt 2>&1 & echo "PID: $!"
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
| Phase 5 | Planned: dow ablation, OVR severe model, early stopping, 150 trials | pending | — |
