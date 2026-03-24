# Session Notes — CPU Spike Prediction Model

**Last updated:** 2026-03-24
**Status:** Phase 3 (severity tiers) + Phase 4 (multi-horizon) complete. 60m model trained and analyzed. 15m binary model crashed during Optuna (OOM) — crash prevention fixes applied, restart pending. Deployability work complete (spike_api.py, Dockerfile updates, setup_vm.sh).

---

## Project Purpose

Build a model that **predicts CPU spikes before they happen** so a scheduler can act on future
state rather than current state.

**Production scenario:**
- Every 5 minutes: read the last 120 min of per-machine CPU data (24 buckets)
- Predict: will a spike occur in the next 60 min on each machine?
- Feed output to a workload scheduler so it can pre-emptively act (defer tasks, add capacity,
  refuse new workloads) before the spike materialises

**Architecture decision (final):** XGBoost only. TFT/V2 was dropped due to latency constraints
in the target system. The goal is to make XGBoost produce richer, more actionable outputs
rather than switching frameworks.

---

## Current Model State (Baseline — clean, post-leakage-fix)

**Dataset:** Google Cluster Traces 2011 (`cluster_cpu_data.csv`)
- 278M raw rows → ~24M (machine_id, 5-min bucket) aggregated rows
- 160 hours, 12,555 machines, 1,919 buckets
- Train 60% / Val 20% / Test 20% (chronological, no shuffle)

**Pipeline:**
```
cluster_cpu_data.csv
  → [Step 1] spike_preprocessor.py      → cluster_agg.parquet
  → [Step 2] spike_feature_engineer.py  → cluster_features.parquet + spike_thresholds.parquet
  → [Step 3] spike_classifier.py        → spike_model.json + spike_config.json + feature_importance.csv
  → predict_spike.py                    → per-machine P(spike) JSON
```

**32 features (current):**
- Raw: `total_cpu, peak_cpu, total_mem, peak_mem, disk_io, n_tasks`
- Lags: `cpu_lag_{1,2,3,6,12,24}`
- Trend: `cpu_ewma_6, cpu_ewma_24`
- Derivatives: `cpu_delta_1, cpu_delta_2, cpu_rolling_std_6`
- Machine-relative: `task_dominance, cpu_vs_p95, cpu_vs_p95_delta`
- Spike history: `spike_now, spike_in_last_{1,3,6}, consecutive_spikes, time_since_last_spike`
- Cluster-level: `cluster_cpu_p90, machine_rank_in_cluster`
- Time: `hour_sin, hour_cos, dow_sin, dow_cos`

**Performance — Phase 2 (2026-03-21, feature engineering + Optuna 75 trials):**

| Metric | Walk-forward CV (5 folds) | Test set |
|--------|--------------------------|----------|
| PR-AUC | **0.640 ± 0.004** | **0.555** |
| ROC-AUC | 0.846 ± 0.003 | 0.832 |
| F1 (calibrated) | 0.585 ± 0.003 | 0.521 |
| Precision | 0.538 | 0.491 |
| Recall | 0.641 | 0.556 |
| Brier score | 0.153 ± 0.007 | — |

**Phase 1 (for comparison):** CV PR-AUC 0.640 / Test PR-AUC 0.548 / threshold 0.65 / ~61 alarms/day
**Baseline (pre-Phase 1):** CV PR-AUC 0.634 / Test PR-AUC 0.537 / threshold 0.70

---

## Phase 3 + 4 Results (2026-03-24)

**Model:** 3-class severity (`multi:softprob`) + 15m binary (`binary:logistic`)
**Optuna:** 50 clean trials (contaminated Phase 2 DB deleted first)
**New features added (8):** `spike_severe_now`, `cpu_vs_p99`, `peak_cpu_vs_p99`, `band_position`, `band_width`, `spike_severe_in_{last_1,3,6}`
**Architecture simplification:** 30m/45m binary models dropped (redundant signal, same features as 15m)

**60m severity model — complete:**

| Metric | CV (production estimate) | Test set |
|--------|--------------------------|----------|
| Macro PR-AUC | **0.565 ± 0.010** | **0.522** |
| Macro ROC-AUC | 0.816 ± 0.016 | 0.798 |
| Weighted F1 | 0.741 ± 0.010 | 0.629 |
| moderate PR-AUC | — | 0.278 (skill 0.210) |
| severe PR-AUC | — | 0.324 (skill 0.301) |
| Alarm @ 0.70 | — | P=0.382 / R=0.332 / ~12.4/day |

**15m binary model:** Crashed during Optuna (OOM — 7.16M inner split × 50 trials × binary:logistic). Crash prevention fixes applied (see below). Restart with `--from-step 3 --optuna-trials 30`.

**Best hyperparams (50 clean trials):**
`n_estimators=586, max_depth=3, learning_rate=0.131, min_child_weight=3, gamma=0.384`

**Top features by SHAP:** `dow_sin`=0.356 (⚠ highest — see below), `cpu_vs_p95`=0.230, `cpu_vs_p99`=0.088
**Top features by gain:** `spike_severe_now`=0.119, `spike_in_last_6`=0.097, `time_since_last_spike`=0.087

**Key findings:**
- p99 features improved moderate PR-AUC by +25% relative (0.222 → 0.278); severe barely changed (ceiling issue)
- `dow_sin` SHAP=0.356 (highest of 39 features, gain=0.011): temporal overfitting risk from 7-day dataset. Decision pending — ablation test recommended
- `band_width` SHAP=0.105 / gain=0.011: validates min-gap enforcement; rare but high-impact for narrow-band machines
- `spike_severe_now` gain=0.119 / SHAP=0.003: redundant with `cpu_vs_p95` / spike history — likely safe to drop
- CV-to-test gap (0.043) explained by test window prevalence shift (severe 5.0% → 3.3%), same anomalous Google Cluster days 21–25 as Phase 2
- Severe PR-AUC 0.324 below 0.40 target — bottleneck is loss function, not features. Asymmetric weights or OVR decomposition needed (Phase 5 scope)

**Crash prevention fixes applied (2026-03-24):**
- `del clf; gc.collect()` inside Optuna objective after each trial
- Binary inner split capped at 2M rows (`_MAX_OPTUNA_ROWS_BINARY`) — reduces memory from 7.16M × trials
- Binary Optuna capped at 30 trials (`_MAX_BINARY_TRIALS`) regardless of `--optuna-trials`
- RAM usage logged before binary model training (requires `psutil`)

**Restart command (15m model only):**
```bash
nohup .venv/bin/python3 train_spike_classifier.py \
  --data-path data/cluster_cpu_data.csv \
  --artifacts-dir data/full_run \
  --from-step 3 \
  --tune-hyperparams --optuna-trials 30 > data/rerun_log_15m.txt 2>&1 & echo "PID: $!"
```

**CV is the production number (0.640).** Flat vs Phase 1.
**Test PR-AUC +0.007** (0.548 → 0.555). Improved generalisation.
Threshold stable at 0.65. ~61 alarms/day unchanged. Precision up slightly (+0.011), Recall down slightly (-0.004).

**Phase 2 best hyperparameters (cached in best_params.json, 75 trials):**
- `n_estimators=680, max_depth=8, learning_rate=0.0194`
- `min_child_weight=8` — down from 19; model splits more freely on new features
- `reg_alpha=2.13, reg_lambda=0.193` — still L1-dominant
- `gamma=2.71` — up from 1.67; compensates for lower min_child_weight
- `subsample=0.649, colsample_bytree=0.731`
- Best trial was #65/75 — search converging but not fully settled

**Top features by gain (Phase 2 model):**
1. `spike_now` (0.268) — jumped from #3; current state is primary signal
2. `time_since_last_spike` (0.180) — recency of last exceedance
3. `cpu_vs_p95` (0.113) — ratio to machine's own p95
4. `spike_in_last_3` (0.105) — jumped from #6; 15-min window now key
5. `spike_in_last_6` (0.096) — dropped from #1 (0.292); absorbed by cpu_spike_rate_24
6. `cpu_spike_rate_24` (0.080) — NEW; top-10 entry, chronic spiker signal works
11. `peak_cpu_vs_p95` (0.010) — NEW; marginal contribution
24. `cpu_trend_slope_6` (0.004) — NEW; essentially useless, already covered by EWMAs

**Key Phase 2 insights:**
- CV ceiling confirmed at 0.640 — binary 60-min label limits discrimination ceiling
- Test set improvement (+0.007) shows new features generalise better, less overfit to training period
- `cpu_spike_rate_24` is a genuine addition (top-6); redistributed importance from spike_in_last_6
- `cpu_trend_slope_6` confirmed redundant (rank #24) — candidate for removal before Phase 3
- `min_child_weight` drop (19→8) signals new features allow more nuanced splits
- **Feature work is yielding diminishing returns — Phase 3 is the right next move**

Low-importance features (drop candidates before Phase 3):
- `cpu_trend_slope_6` — gain 0.004, rank #24; redundant with EWMAs
- `consecutive_spikes` — gain 0.003, rank #26; mostly captured by cpu_spike_rate_24

---

## Enhancement Plan (5 Phases)

### Why this order

The binary model must be optimised BEFORE changing its output format. Hyperparameter tuning
for `binary:logistic` ≠ optimal params for `multi:softprob`. Features must be stable before
multi-horizon models are trained (all 4 horizons share the same feature set). Calibration is
always last (applied to final model outputs).

---

### Phase 1 — Hyperparameter Optimisation + Monotone Constraints

**Goal:** Push the binary 60-min model toward its ceiling before any architectural changes.

**Hyperparameter search (Optuna, Bayesian, 50–100 trials):**
Optimise over walk-forward CV PR-AUC. Key parameters:
- `max_depth` (3–8) — controls tree complexity
- `min_child_weight` (1–20) — minimum samples per leaf (regularisation for rare positives)
- `subsample` (0.6–1.0) — row subsampling per tree
- `colsample_bytree` (0.4–0.9) — feature subsampling (important: 32 features, many correlated)
- `gamma` (0–5) — minimum loss reduction to split (pruning)
- `reg_alpha` / `reg_lambda` (0–10) — L1/L2 regularisation
- `learning_rate` (0.01–0.2) + `n_estimators` (200–1500) — jointly optimised
- `scale_pos_weight` stays computed per-fold (not tuned)

**Monotone constraints (add simultaneously):**
XGBoost `monotone_constraints` parameter enforces that predictions never violate physical
intuition regardless of what noisy training data might suggest:
- `cpu_vs_p95` → +1 (closer to threshold = higher spike probability, always)
- `cpu_vs_p95_delta` → 0 (can decrease if CPU dropping — leave unconstrained)
- `spike_now` → +1 (already spiking = higher future risk)
- `spike_in_last_1, spike_in_last_3, spike_in_last_6` → +1
- `consecutive_spikes` → +1
- `time_since_last_spike` → -1 (longer ago = lower risk)

Expected gain: +0.01–0.03 PR-AUC. Even if metric is neutral, constraints make the model
safer for the scheduler (no physically nonsensical predictions).

---

### Phase 2 — Feature Engineering Improvements

**Goal:** Replace low-signal features with higher-signal alternatives informed by the current
importance chart.

**Features to ADD:**

`cpu_spike_rate_24` — fraction of the last 24 buckets (120 min) where `total_cpu > threshold`.
Different from `consecutive_spikes` (only measures the *current* unbroken run) and
`spike_in_last_6` (binary presence). This captures "chronic spiker" machines — those that
spike frequently even if not right now. Likely high-importance candidate given current
feature dominance of spike history.

`cpu_trend_slope_6` — linear regression slope fitted to `total_cpu` over the last 6 buckets
(30 min). More informative than `cpu_delta_1` (single step, noisy) and complements EWMAs
(which smooth but don't give a directional rate). A machine at cpu=0.6 with slope +0.05/bucket
is far more dangerous than one at cpu=0.6 with slope −0.02/bucket. Cannot be inferred from
existing delta features alone.

`peak_cpu_vs_p95` — `peak_cpu / threshold`. Current `cpu_vs_p95` uses `total_cpu` (the bucket
average). `peak_cpu` captures the highest instantaneous load in the 5-min window. Near-miss
events (where the average looks fine but the peak grazed the threshold) are currently invisible
to the model.

**Features to potentially DROP (after Phase 1 re-evaluation):**
- `cpu_lag_2`, `cpu_lag_3`, `cpu_lag_6` — combined gain 0.010, all below `cluster_cpu_p90`.
  Likely redundant given `cpu_lag_1`, `cpu_ewma_6`, and the new trend slope feature.
  Decision: run Phase 1 first; if `colsample_bytree` tuning already sidelines them, drop
  explicitly and reduce feature count to 32 → 32 (swap 3 low for 3 high).

**Note:** Phase 2 requires re-running Step 2+3 of the pipeline. Step 2 (~2.5 min) and
Step 3 (~8 min training) must both re-run after feature changes.

---

### Phase 3 — Severity Tiers (`multi:softprob`)

**Goal:** Replace the binary spike flag with a 3-class severity signal so the scheduler can
calibrate its response (mild vs severe spike → different pre-emptive actions).

**Label definition:**
- Class 0 — `no_spike`: no window in [t+1, t+12] exceeds the machine's p95 threshold
- Class 1 — `moderate_spike`: p95 exceeded but p99 not (above-normal but manageable load)
- Class 2 — `severe_spike`: p99 exceeded (extreme load — highest priority scheduler action)

Per-machine p99 threshold computed from training data only (same leakage-prevention logic
as p95 threshold). Stored in `spike_thresholds.parquet` alongside p95.

**Architectural changes required:**
- `spike_feature_engineer.py`: compute per-machine p99 thresholds; replace `spike_in_60m`
  binary with `severity_in_60m` 3-class label {0,1,2}
- `spike_classifier.py`: change objective to `multi:softprob`, `num_class=3`; update metrics
  to per-class PR-AUC and macro-weighted F1; update threshold selection (one threshold per
  class pair)
- `train_spike_classifier.py`: update spike_config.json output format; add per-class metrics
- `predict_spike.py`: output changes from `{is_spike, spike_probability}` to
  `{severity_class, p_no_spike, p_moderate, p_severe}`
- Tests: update all label-dependent assertions

**Why severity > binary for the scheduler:** The scheduler does not need to take the same
action for a 2% above-threshold exceedance (moderate) as for a 50% above-threshold
exceedance (severe). This phase makes the model's output directly actionable at different
urgency levels.

---

### Phase 4 — Multi-Horizon Predictions (15 / 30 / 45 / 60 min)

**Goal:** Give the scheduler an imminence profile, not just a 60-min-ahead flag.

**Architecture:** 4 separate binary classifiers (one per horizon), trained on the same
feature set. The multi-output XGBoost (`multi_strategy="multi_output_tree"`) is still
experimental as of 3.2 — use separate models for production safety.

**Label windows:**
- `spike_in_15m`: any spike in [t+1, t+3] (next 15 min)
- `spike_in_30m`: any spike in [t+1, t+6] (next 30 min)
- `spike_in_45m`: any spike in [t+1, t+9] (next 45 min)
- `spike_in_60m`: any spike in [t+1, t+12] (current, unchanged)

All 4 labels computed in `spike_feature_engineer.py` simultaneously. Training loop in
`train_spike_classifier.py` iterates over horizons, produces 4 model artifacts in
`models/spike/horizon_15m/`, `horizon_30m/`, etc.

**Scheduler output:**
```json
{
  "machine_id": 1234,
  "spike_probability": {
    "15m": 0.08,
    "30m": 0.21,
    "45m": 0.49,
    "60m": 0.70
  },
  "severity": "moderate",
  "status": "success"
}
```

A monotonically increasing profile (low→high as horizon grows) means "building slowly —
act in 30 min". A flat high profile means "spike imminent — act now". A non-monotone profile
(e.g., high at 15m, lower at 60m) means "short burst, likely self-resolving".

**Note:** The severity label from Phase 3 applies to the 60-min model. The 15/30/45 models
stay binary (the severity distinction matters most at the longest horizon where you have
time to act differently).

---

### Phase 5 — Probability Calibration

**Goal:** Ensure P(spike) = 0.70 actually means ~70% of those cases spike. Currently
calibration is unknown — XGBoost ranks well (PR-AUC 0.634) but raw probabilities may not
be well-calibrated frequencies.

**Method:** Isotonic regression (non-parametric, correct choice at 14M rows) fitted on
the val set AFTER final model training. Applied as a post-processing wrapper.

**Implementation:** `sklearn.calibration.CalibratedClassifierCV(method="isotonic")` or
a manual Platt scaling fit stored in the model artifacts directory. The calibration mapping
is stored alongside `spike_model.json` so `predict_spike.py` can apply it at inference.

**Validation:** Reliability diagram (predicted probability bins vs observed frequency).
A well-calibrated model's diagram lies on the diagonal.

---

## Key Design Decisions (standing)

| Decision | Status | Rationale |
|----------|--------|-----------|
| XGBoost only (no TFT) | Confirmed | Latency constraints in target system; TFT dropped |
| Chronological 60/20/20 split | Confirmed | No random shuffle — temporal data |
| gap=12 in walk-forward CV | Confirmed | 60-min gap = prediction horizon |
| Per-fold scale_pos_weight (binary) / compute_sample_weight (multiclass) | Confirmed | No class-ratio leakage from future folds |
| P95 + P99 spike thresholds (per machine) | Confirmed | Training buckets only; leakage-free; min 10% gap enforced |
| PR-AUC as primary metric | Confirmed | Correct for imbalanced data; macro for multiclass |
| Optimal threshold on val only | Confirmed | Never on test |
| Production metric = CV Macro PR-AUC | Confirmed | 0.565; test set was anomalous period |
| Monotone constraints | Active (binary only) | Disabled for multi:softprob — semantically undefined |
| Severity tiers (3-class) | ✅ Done Phase 3 | Richer scheduler signal |
| Multi-horizon (15m + 60m) | ✅ Done Phase 4 | 30m/45m dropped as redundant |
| OVR binary decomposition for severe | Phase 5 candidate | Best path to severe skill > 0.40 |

---

## VM Deployment Intention

The goal is to deploy the training pipeline to an internal datacenter VM for a full 24h training run (enabling full Optuna search with more trials and the complete dataset).

**Setup:** `setup_vm.sh` at the project root automates venv creation and tmux session launch.

```bash
# 1. Transfer code and data to the VM (run locally)
rsync -avz --partial --inplace AIModel/ user@vm-host:~/spike/AIModel/
rsync -avz --partial --inplace AIModel/data/cluster_cpu_data.csv user@vm-host:~/spike/AIModel/data/

# 2. SSH and run setup (run on VM)
ssh user@vm-host
cd ~/spike && bash setup_vm.sh

# 3. Monitor training
tmux attach -t train      # attach
# Ctrl-B then D           # detach safely (session keeps running)
tail -f AIModel/data/rerun_log.txt
```

**After training:** Sync artifacts back and run the API:

```bash
# Sync artifacts back locally
rsync -avz user@vm-host:~/spike/AIModel/data/full_run/ AIModel/data/full_run/

# Start API
docker compose up --build
curl http://localhost:8000/ready
```

---

## File Status

| File | Status |
|------|--------|
| `spike_preprocessor.py` | ✅ Complete |
| `spike_feature_engineer.py` | ✅ Complete — Phase 3 p99 thresholds + severity label + 8 p99 features |
| `spike_classifier.py` | ✅ Complete — multi:softprob + BinarySpikeClassifier |
| `train_spike_classifier.py` | ✅ Complete — OOM fixes applied; binary trials capped at 30 |
| `predict_spike.py` | ✅ Complete — 2-model inference (60m severity + 15m binary) + `predict_with_artifacts()` |
| `spike_api.py` | ✅ Complete — FastAPI wrapper (lifespan, /health, /ready, /predict) |
| `Dockerfile` | ✅ Updated — libgomp1, requirements-inference.txt |
| `Dockerfile.test` | ✅ Updated — libgomp1, requirements-train.txt |
| `Dockerfile.training` | ✅ New — heavy training image |
| `setup_vm.sh` | ✅ New — VM setup + tmux launcher |
| `requirements-inference.txt` | ✅ New — exact pins for inference image |
| `requirements-train.txt` | ✅ New — extends inference with optuna/shap/psutil |
| `tests/` | ✅ Passing |

---

## Training Commands

```bash
# From step 3 only (fastest — skip preprocessing/features)
nohup .venv/bin/python3 train_spike_classifier.py \
  --data-path data/cluster_cpu_data.csv \
  --artifacts-dir data/full_run \
  --from-step 3 > data/rerun_log.txt 2>&1 & echo "PID: $!"

# From step 2 (re-engineer features, then retrain)
nohup .venv/bin/python3 train_spike_classifier.py \
  --data-path data/cluster_cpu_data.csv \
  --artifacts-dir data/full_run \
  --from-step 2 > data/rerun_log.txt 2>&1 & echo "PID: $!"

# Force re-run step 2 (ignore cache)
nohup .venv/bin/python3 train_spike_classifier.py \
  --data-path data/cluster_cpu_data.csv \
  --artifacts-dir data/full_run \
  --from-step 2 --force > data/rerun_log.txt 2>&1 & echo "PID: $!"

# Monitor
tail -f data/rerun_log.txt

# Run tests
.venv/bin/python3 -m pytest tests/ -v
```
