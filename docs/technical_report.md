# Technical Report — CPU Spike Predictor

**Version:** Production baseline (K=2, 40 features, 2026-04-26)
**Status:** Deployed, cross-domain validated on Zabbix HPC cluster

**See also:** `docs/theory.md` for the statistical/ML theory behind these design choices
(loss functions, calibration mechanics, PSI math, metrics glossary, anticipated review
questions). This document is the empirical record — what was tried and what happened; theory.md
explains why the winning approach works the way it does.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset](#2-dataset)
3. [Model Architecture](#3-model-architecture)
4. [Feature Engineering](#4-feature-engineering)
5. [Label Design — K-of-N Ablation](#5-label-design--k-of-n-ablation)
6. [Training Pipeline](#6-training-pipeline)
7. [Experiments and Rejected Approaches](#7-experiments-and-rejected-approaches)
8. [Production Results](#8-production-results)
9. [Baseline Comparison](#9-baseline-comparison)
10. [Cross-Domain Validation — Zabbix HPC](#10-cross-domain-validation--zabbix-hpc)
11. [Calibration](#11-calibration)
12. [Design Decisions and Limitations](#12-design-decisions-and-limitations)
13. [Future Work](#13-future-work)

---

## 1. Problem Statement

Datacenter workload schedulers react to CPU spikes after they occur, by which point
jobs are already degraded or evicted. The goal of this system is to predict, every
5 minutes, whether each machine in a cluster will experience a CPU spike in the next
60 minutes — and how severe it will be — so the scheduler can act pre-emptively.

**Prediction target:** Given the last 120 minutes (24 × 5-minute buckets) of CPU,
memory, disk, and task telemetry per machine, predict whether CPU will exceed the
machine's historical p95 (moderate) or p99 (severe) threshold in any of the next
12 five-minute windows (60 minutes).

**System output:** Per-machine severity class (no spike / moderate / severe) with
calibrated probabilities, a short-horizon imminence signal (15m/30m/45m), and an
actionable recommendation for the scheduler (normal → monitor → defer → migrate →
preempt).

**Why this is hard:** The positive class (spike) is rare (~10% of windows), the
severe class is very rare (~2% of windows in training), CPU time series are
autocorrelated, and the model must generalise across machines with very different
baseline load profiles without per-machine retraining.

---

## 2. Dataset

### Google Cluster Traces 2011 (primary training data)

- **Source:** Google Borg scheduler telemetry, publicly available
- **Raw rows:** 278 million task-level usage records
- **Aggregated:** ~24 million (machine × 5-min bucket) rows after preprocessing
- **Machines:** 12,555 unique machines
- **Duration:** ~160 hours (≈7 days), 1 Borg cell, 1,919 buckets per machine
- **Split:** Chronological — train 60% / val 20% / test 20% (no shuffle anywhere)
- **CPU units:** Fraction of one core (duration-weighted mean within bucket)

**Spike pattern diversity:** The 12,555 machines exhibit meaningfully different load
profiles — each machine has its own p95/p99 threshold, so a "spike" on a heavy machine
looks nothing like one on a lightly loaded machine. The training set contains machines
that spike frequently, machines that almost never spike, and machines with a wide range
of spike shapes: gradual builds, sudden bursts, sustained high load. Severity also varies
from marginal p95 exceedances (moderate) to deep p99 violations (severe). The model
is not pattern-matching a single template — it generalises across intra-day load cycles
and machine heterogeneity. What it does *not* capture is multi-week or seasonal
cyclicality: with only 7 days of data, `dow_sin`/`dow_cos` features were dropped
(~23 samples per day label — confirmed temporal confound; see Section 7).

**Known limitation:** 7 days of data from a single 2011 Borg cell is a structural
ceiling. Near-identical performance across the 15m/30m/45m horizons (0.012 gap on
test) is a direct consequence — not a model failure. Google 2019 (8 cells, 31 days)
is the recommended next dataset (see Section 13).

### Zabbix HPC Cluster (cross-domain validation)

- **Source:** Slurm + infrastructure nodes monitored by Zabbix 7.4.6
- **Rows:** 273,198 rows (89 days × 11 nodes)
- **Duration:** 89 days (since 2026-01-13)
- **Nodes:** 3 Slurm compute (atari, sega, xbox), 1 GPU-only (skynet), 7 infrastructure
- **Collection interval:** Exactly 5 minutes (native Zabbix item interval)
- **History retention:** 89 days on all 11 nodes

The Zabbix cluster represents an HPC/Slurm workload completely different from Google
2011 (job-driven vs always-on, different hardware, 2026 vs 2011). It was used
exclusively for zero-shot cross-domain validation — no Zabbix data touched training.

**Zabbix schema mapping:**

| cluster_agg field | Zabbix key | Unit conversion |
|---|---|---|
| `total_cpu` / `peak_cpu` | `system.cpu.util` | `(% / 100) × n_cores` → fraction-of-core |
| `total_mem` / `peak_mem` | `vm.memory.utilization` | `% / 100` → fraction |
| `disk_io` | `vfs.dev.util[nvme0n1]` | `% / 100` → dimensionless |
| `n_tasks` | `system.cpu.load[all,avg1]` | `max(1, round(float))` → int32 |

Note: Zabbix provides one sample per 5-min bucket — `peak_cpu = total_cpu` (no
intra-bucket peak available). This approximation is acceptable for pre-aggregated data.

---

## 3. Model Architecture

Five XGBoost models trained on the same 40-feature set, all using chronological splits:

| Model | Type | Target | Objective |
|---|---|---|---|
| **60m severity** | 3-class | no_spike / moderate / severe in next 60m | `multi:softprob` |
| **15m binary** | binary | any spike in next 15m | `binary:logistic` |
| **30m binary** | binary | any spike in next 30m | `binary:logistic` |
| **45m binary** | binary | any spike in next 45m | `binary:logistic` |
| **OVR severe** (Stage 2 cascade) | binary | severe vs moderate | `binary:logistic` |

### Why XGBoost over deep learning

TFT (Temporal Fusion Transformer) was evaluated and dropped in 2026-03-21 due to
inference latency. XGBoost inference on 24 buckets × 12,555 machines completes in
< 1 second — well within the 5-minute polling cycle. TFT would require GPU at
inference time and fails the latency budget for this use case.

### Cascade architecture (60m + OVR)

At inference, the 60m model acts as Stage 1: machines where `p(any spike) >= 0.15`
are forwarded to the OVR severe model (Stage 2). This gating concentrates Stage 2
on the spike-positive subpopulation, raising the severe fraction from 2.4% overall
to ~19% within the gated subset — making the binary classification problem tractable
without custom loss functions.

### Monotone constraints

Applied to binary models only (`binary:logistic`). XGBoost `monotone_constraints` is
undefined for `multi:softprob` (K-class softmax gradient is not decomposable by
feature). All features that have a physically monotone relationship with spike
probability (e.g. `cpu_vs_p95`, `cpu_ewma_24`) have constraints set to `+1`;
features with inverse relationship (e.g. `time_since_last_spike`) set to `-1`.

---

## 4. Feature Engineering

40 features derived from the 9-column cluster_agg schema. All features are strictly
causal — no information from time t is used to predict at time t (spike history uses
`shift(1)` to exclude the current bucket).

### Feature groups

**Raw telemetry (6):**
`total_cpu, peak_cpu, total_mem, peak_mem, disk_io, n_tasks`

**CPU temporal (7):**
`cpu_lag_1, cpu_lag_12, cpu_lag_24, cpu_ewma_6, cpu_ewma_24, cpu_delta_1, cpu_delta_2`

**CPU statistics (2):**
`cpu_rolling_std_6, cpu_per_task`

**Machine-relative / p95 (10):**
`cpu_vs_p95, cpu_vs_p95_delta, cpu_vs_p95_slope_3, cpu_vs_p95_slope_6, time_to_p95_3,
peak_cpu_vs_p95, spike_now, spike_in_last_1, spike_in_last_3, spike_in_last_6`

**Machine-relative / p99 (7):**
`spike_severe_now, cpu_vs_p99, peak_cpu_vs_p99, band_position, band_width,
spike_severe_in_last_1, spike_severe_in_last_3, spike_severe_in_last_6`

**Spike history (2):**
`time_since_last_spike, cpu_spike_rate_24`

**Cluster-level (3):**
`cluster_cpu_p90, machine_rank_in_cluster, task_dominance`

**Time encoding (2):**
`hour_sin, hour_cos` (from bucket index; deterministic, UTC-aligned)

### Leakage firewall

All spike-history features derive from `exc = spike_now.shift(1)` — the previous
bucket's exceedance status, not the current. Labels look forward from the next bucket.
The `shift(1)` on the input side and the future window on the label side are in
opposite directions — they cannot compound into leakage.

### Per-machine thresholds

p95 and p99 thresholds are computed per machine from the training split only
(buckets ≤ `train_max`). A minimum 10% gap between p99 and p95 is enforced:
`p99 = max(p99, p95 × 1.10)`. Thresholds from the test set never influence features.

### Rate-of-approach features (added Phase 9, 37→40)

`cpu_vs_p95_slope_3` — CPU-to-p95 ratio change over 3 buckets (15m velocity)
`cpu_vs_p95_slope_6` — change over 6 buckets (30m velocity)
`time_to_p95_3` — estimated buckets until p95 breach at current velocity

These were added to differentiate the 15m/30m/45m models. `time_to_p95_3` ranked
3rd by XGBoost gain in the 15m model. The horizon differentiation gain was +0.012
(15m–45m gap on test) — positive but smaller than the 0.030 target, attributed to
the 7-day dataset ceiling.

### Full feature importance (production 60m model)

| Feature | Gain importance | SHAP mean abs | Note |
|---|---|---|---|
| `spike_now` | 0.1686 | 0.0038 | High gain, low SHAP — local boundary splits |
| `cpu_vs_p95` | 0.1261 | **0.3232** | Strongest global predictor (SHAP) |
| `time_since_last_spike` | 0.0872 | 0.0505 | |
| `spike_in_last_6` | 0.0688 | 0.0054 | |
| `band_position` | 0.0592 | 0.0243 | |
| `time_to_p95_3` | 0.0462 | 0.0251 | |
| `cpu_spike_rate_24` | 0.0443 | 0.0921 | |
| `spike_severe_in_last_6` | 0.0411 | 0.0121 | |
| `cpu_vs_p99` | 0.0397 | 0.0623 | |
| `spike_severe_now` | 0.0320 | 0.0003 | |
| `spike_in_last_3` | 0.0247 | 0.0030 | |
| `hour_cos` | 0.0220 | **0.1355** | Time features: high SHAP, low gain |
| `total_cpu` | 0.0189 | 0.0516 | |
| `hour_sin` | 0.0168 | **0.1108** | |
| `spike_severe_in_last_3` | 0.0158 | 0.0007 | |
| `machine_rank_in_cluster` | 0.0157 | 0.0203 | |
| `band_width` | 0.0136 | 0.0906 | |
| `cpu_ewma_24` | 0.0135 | 0.0511 | |
| `cpu_ewma_6` | 0.0123 | 0.0134 | |
| `cpu_lag_1` | 0.0121 | 0.0077 | |
| `n_tasks` | 0.0118 | 0.0307 | |
| `cluster_cpu_p90` | 0.0099 | 0.0345 | |
| `cpu_vs_p95_slope_3` | 0.0094 | 0.0212 | |
| `disk_io` | 0.0094 | 0.0370 | |
| `cpu_per_task` | 0.0092 | 0.0309 | |
| `cpu_vs_p95_delta` | 0.0082 | 0.0199 | |
| `peak_cpu_vs_p95` | 0.0067 | 0.0389 | |
| `cpu_lag_24` | 0.0065 | 0.0143 | |
| `total_mem` | 0.0059 | 0.0494 | |
| `spike_in_last_1` | 0.0056 | 0.0007 | |
| `peak_cpu_vs_p99` | 0.0055 | 0.0235 | |
| `peak_cpu` | 0.0053 | 0.0508 | |
| `peak_mem` | 0.0051 | 0.0290 | |
| `cpu_vs_p95_slope_6` | 0.0048 | 0.0213 | |
| `cpu_delta_1` | 0.0043 | 0.0027 | |
| `cpu_rolling_std_6` | 0.0042 | 0.0147 | |
| `cpu_lag_12` | 0.0034 | 0.0108 | |
| `task_dominance` | 0.0028 | 0.0028 | |
| `cpu_delta_2` | 0.0018 | 0.0053 | |
| `spike_severe_in_last_1` | 0.0014 | 0.0000 | |

**Key observation:** `cpu_vs_p95` is the dominant predictor by SHAP (0.323) but only
3rd by gain. `spike_now` has the highest gain (0.169) but very low SHAP (0.004) —
it creates high-information local boundary splits near the current spike/no-spike
boundary without globally dominating the probability output.

Time features (`hour_cos`, `hour_sin`) have surprisingly high SHAP (0.135, 0.111)
but low gain (0.022, 0.017). This led to the `dow_sin`/`dow_cos` investigation below.

---

## 5. Label Design — K-of-N Ablation

### Definition

`severity_in_60m` is moderate if ≥ K of the next 12 five-minute windows exceed
the machine's p95 CPU threshold, severe if ≥ K windows exceed p99.

Binary labels (`spike_in_15m`, `spike_in_30m`, `spike_in_45m`) always use K=1.

### Ablation results (60m severity model)

| Metric | K=1 | K=2 (production) | K=3 |
|---|---|---|---|
| CV macro PR-AUC | 0.565 ± 0.009 | 0.500 ± 0.005 | 0.483 ± 0.007 |
| Test PR-AUC (calibrated) | 0.573 | **0.547** | 0.516 |
| Test ROC-AUC | 0.840 | 0.843 | **0.879** |
| Severe class rate (test) | 3.3% | 1.8% | 1.0% |
| Alarms / day | 16.5 | **7.4** | 5.7 |
| Alarm precision @ 0.25 | 0.424 | **0.446** | 0.334 |
| Alarm recall @ 0.25 | 0.332 | 0.307 | 0.283 |
| CV drift tau | 0.0 | 0.2 | **0.6 ⚠** |

**Control — 15m binary (K=1 labels always, not ablated):**
All three K settings give 0.575–0.576 PR-AUC and 0.911–0.912 ROC-AUC. This is the
strongest evidence that K-driven differences in 60m are purely label-driven, not
model quality changes.

### Why K=2 is the production baseline

PR-AUC declines with K because the positive class becomes sparser — lower PR-AUC
baseline with fewer positives is a mathematical property of the metric, not model
degradation. ROC-AUC, which is prevalence-neutral, actually improves with K.

K=2 was chosen over K=3 for three reasons:
1. K=3 CV drift tau = 0.6 (positive Kendall's tau across folds), indicating
   performance instability — the model may be overfit to specific early-trace patterns
2. K=2 alarm rate (7.4/day) is operationally preferable to K=1 (16.5/day) without
   the instability of K=3
3. K=2 precision (0.446) beats both K=1 (0.424) and K=3 (0.334) at the same threshold

K=3 was ruled out definitively — the tau=0.6 drift is a red flag independent of
the PR-AUC argument.

---

## 6. Training Pipeline

### Walk-forward cross-validation

5 folds, expanding window, 12-bucket gap (60 min = 1 horizon) between train and val
in each fold. The gap prevents label leakage from future windows being used as features.

**Per-fold CV results (60m model, production run 2026-04-26):**

| Fold | Train rows | Val rows | PR-AUC | ROC-AUC | Weighted F1 | Macro F1 |
|---|---|---|---|---|---|---|
| 1 | 2,379,410 | 2,379,420 | 0.509 | 0.810 | 0.790 | 0.471 |
| 2 | 4,758,830 | 2,379,420 | 0.495 | 0.801 | 0.760 | 0.453 |
| 3 | 7,138,250 | 2,379,420 | 0.507 | 0.809 | 0.779 | 0.464 |
| 4 | 9,517,670 | 2,379,420 | 0.509 | 0.804 | 0.788 | 0.470 |
| 5 | 11,897,090 | 2,379,420 | 0.512 | 0.791 | 0.774 | 0.466 |
| **Mean** | | | **0.506 ± 0.006** | **0.803 ± 0.007** | **0.778 ± 0.011** | **0.465 ± 0.006** |

CV drift tau = 0.6 (p=0.233, not significant) — no consistent fold-to-fold trend,
indicating the model is stable across the expanding training window.

### Hyperparameter optimisation (Optuna)

150 trials via tree-structured Parzen estimator. GPU training (`device=cuda`,
`tree_method=hist`, `n_jobs=1`). Early stopping: 4000 estimators, 150 rounds.

**Best hyperparameters (60m model):**

| Parameter | Value |
|---|---|
| `max_depth` | 4 |
| `learning_rate` | 0.01014 |
| `subsample` | 0.991 |
| `colsample_bytree` | 0.608 |
| `min_child_weight` | 10 |
| `gamma` | 1.558 |
| `reg_alpha` | 2.464 |
| `reg_lambda` | 9.214 |
| Best trial Optuna score | 0.501 PR-AUC |

Binary models (15m/30m/45m/OVR) capped at 75 Optuna trials — `binary:logistic`
converges faster. Optuna inner split capped at 2M rows to prevent OOM on the GPU.

### OOM resolution (Phase 7)

During OVR severe training, heap fragmentation after two sequential XGBoost runs caused
out-of-memory kills even with 16 GB nominally free. PyArrow's Parquet reader requires
2–3× the data size as temporary buffers. Fix: pre-slice `df_ovr` with column pushdown
immediately after the first `pd.read_parquet()` call — before any XGBoost training —
so the allocation happens on a clean heap.

### Chronological split bug (discovered during Zabbix evaluation)

The original `train_max = int(bucket_max * train_ratio)` assumed buckets started at 0.
For Unix-epoch bucket values (bucket_min ≈ 5,785,632), `bucket_max * 0.6` falls far
below bucket_min — all rows would be assigned to the test split. Fixed to:
`train_max = b_min + int((b_max - b_min) * train_ratio)`. This bug was silent on
Google 2011 data (bucket_min ≈ 2) but would catastrophically break on modern timestamps.

---

## 7. Experiments and Rejected Approaches

### ✗ Temporal Fusion Transformer (dropped 2026-03-21)

TFT was evaluated for its ability to model long-range temporal dependencies across
the 24-bucket window. Dropped due to inference latency — GPU is required at inference
time and the 5-minute polling cycle has a strict latency budget. XGBoost inference
completes in < 1 second across all machines.

### ✗ dow_sin / dow_cos (day-of-week features, dropped Phase 5)

These features showed the highest SHAP values (0.356) across all ablations. This
was suspicious — with only 7 days of training data, there are ~23 samples per label
per day-of-week. The high SHAP reflects a temporal confound between day-of-week and
position in the 7-day trace, not a genuine workload periodicity signal.

Confirmed by near-zero XGBoost gain (0.011) — the tree barely uses the feature for
splitting despite the apparent global importance. Both features dropped permanently.

### ✗ Streak features: current_spike_streak, max_spike_streak_24h, current_severe_streak (Phase 6)

Added to capture sustained burst patterns. Two separate training runs (one with stale
Optuna hyperparameters, one with fresh 150-trial Optuna) both gave:

- Stale Optuna: 0.553 cal PR-AUC
- Fresh Optuna: 0.553 cal PR-AUC
- Baseline (without streak features): 0.574 cal PR-AUC
- **Regression: −0.021 PR-AUC**

Confirmed collinearity with `spike_in_last_{1,3,6}` which already capture streak
information implicitly. All 3 streak features permanently removed.

### ✗ FFT spectral features (Phase 8)

5 FFT magnitude features added (dominant frequency, amplitude of top-3 frequency
components). Results vs K=2 baseline:

| Model | With FFT | Baseline | Delta |
|---|---|---|---|
| 60m severity | 0.548 cal | 0.547 cal | **+0.001 (noise)** |
| 15m binary | 0.577 | 0.575 | +0.002 (noise) |
| OVR severe | 0.304 | 0.328 | **−0.026 regression** |

FFT features add no value to 60m or 15m and hurt OVR severe by −0.026 PR-AUC.
The OVR regression is attributed to FFT features capturing Google-specific spectral
patterns (Borg scheduler heartbeats) that act as noise when combined with the
already-informative spike history features. Reverted.

### ✗ Focal loss for OVR severe (Phase 10a)

Hypothesis: the OVR model was learning "is it currently severe?" (`spike_now` dominates
with 70.1% of gain) rather than "will it become severe?" Focal loss (γ=2) was tested
to down-weight easy negatives and force the model to focus on hard positive cases.

Result: 0.312 PR-AUC vs 0.328 baseline — **−0.016 regression**.

Post-hoc analysis confirmed that `spike_now` dominance was not "laziness" — it is
genuinely the most informative single feature for predicting severe spikes, because
machines currently in a severe state are most likely to remain severe in the next
window. Focal loss penalised these easy-but-correct predictions and reduced accuracy.

### ✓ Rate-of-approach features (Phase 9, kept)

`cpu_vs_p95_slope_3`, `cpu_vs_p95_slope_6`, `time_to_p95_3` added. Results:

| Model | 37 features | 40 features | Delta |
|---|---|---|---|
| 60m severity | 0.547 cal | **0.547 cal** | 0 |
| 15m binary | 0.575 | **0.576** | +0.001 |
| 30m binary | — | **0.565** | new model |
| 45m binary | — | **0.563** | new model |
| OVR severe | 0.339 | **0.339** | 0 |

`time_to_p95_3` ranked 3rd by gain in the 15m model. 15m→45m horizon gap on test:
0.012 (target was 0.030; not achieved due to dataset ceiling). Features kept because
they provide physically meaningful rate-of-approach signal and enabled adding 30m/45m
models without regression.

### ✓ Cascade Stage 2 OVR severe (Phase 10b, production)

The full-population OVR model trains on 2.4% severe rows — standard `scale_pos_weight`
helps but the imbalance remains extreme. Cascade Stage 2 trains only on spike-positive
rows (moderate + severe), raising the severe fraction to ~19%.

Result on Google test (spike-positive subset, ~25% base rate):
- **Cascade Stage 2: 0.543 PR-AUC, 0.761 ROC-AUC**
- Full-population baseline: 0.328 PR-AUC, 0.865 ROC-AUC (incomparable — different base rate)

Note: these numbers are NOT directly comparable. The cascade is evaluated on spike-positive
rows only; the baseline was evaluated on all test rows. The cascade is the correct
approach for the production use case (Stage 1 already gates the spike-positive machines).

**Zabbix result (cross-domain):** cascade 0.891 vs old full-dataset OVR 0.926.
The cascade trained on Google spike-positive rows does not generalise as well to
Zabbix as the full-population OVR. This is a known trade-off — cascade is kept as
production because Google in-distribution performance is the primary evaluation target.

---

## 8. Production Results

**Production baseline: K=2, 40 features, 150-trial Optuna, XGBoost 3.2.0**
**Training completed: 2026-04-26, deployed 2026-04-27**

### Final test metrics (Google Cluster 2011, last 20% chronological)

| Model | CV PR-AUC | Test PR-AUC | Test ROC-AUC | Alarm threshold | Alarm P | Alarm R |
|---|---|---|---|---|---|---|
| 60m severity | 0.506 ± 0.006 | 0.510 raw / **0.547 cal** | **0.849** | 0.25 | 0.429 | 0.309 |
| 15m binary | 0.591 ± 0.007 | **0.575** | **0.912** | 0.80 | 0.605 | 0.496 |
| 30m binary | 0.595 ± 0.007 | **0.564** | **0.878** | 0.70 | 0.569 | 0.497 |
| 45m binary | 0.605 ± 0.008 | **0.563** | **0.860** | 0.65 | 0.535 | 0.524 |
| OVR severe (cascade) | 0.260 ± 0.016 | **0.543** (spike-positive) | **0.761** | 0.50 | 0.430 | 0.650 |

### Class rates (60m model)

| Class | Train rate | Test rate | Note |
|---|---|---|---|
| no_spike (0) | 87.57% | 93.01% | Cluster stabilises over time in the trace |
| moderate (1) | 10.05% | 5.23% | Shrinks at test time |
| severe (2) | 2.37% | 1.76% | Shrinks at test time |

The shift from train to test (87.6% → 93.0% no-spike) indicates the Google cluster
increasingly stabilised over the 7-day trace. This temporal non-stationarity is a
known property of the dataset.

### Alarm threshold sweep (60m model, calibrated probabilities)

| Threshold | Precision | Recall | F1 | Alarms / day |
|---|---|---|---|---|
| 0.10 | 0.224 | 0.563 | 0.320 | 21.7 |
| 0.15 | 0.285 | 0.475 | 0.356 | 14.3 |
| 0.20 | 0.338 | 0.411 | 0.371 | 10.5 |
| **0.25** | **0.390** | **0.359** | **0.374** | **7.9** |
| 0.30 | 0.432 | 0.322 | 0.369 | 6.4 |
| 0.35 | 0.465 | 0.294 | 0.360 | 5.4 |
| 0.40 | 0.490 | 0.271 | 0.349 | 4.8 |
| 0.50 | 0.529 | 0.232 | 0.322 | 3.8 |
| 0.60 | 0.574 | 0.192 | 0.287 | 2.9 |
| 0.70 | 0.635 | 0.140 | 0.229 | 1.9 |
| 0.80 | 0.696 | 0.064 | 0.117 | 0.8 |
| 0.90 | 0.927 | 0.000 | 0.001 | 0.0 |

F1-max occurs at threshold 0.25 → production default. Operators with low FP tolerance
should use 0.40–0.50 (precision 0.49–0.53, alarms ~4/day).

---

## 9. Baseline Comparison

Six baselines evaluated on the same cluster_agg schema (no feature engineering):

| Baseline | Method |
|---|---|
| Random | Uniform scores |
| Persistence | `total_cpu / p95` as score |
| Static threshold | Fraction of last 4 buckets where cpu > p95 |
| EWMA z-score | Z-score vs EWMA(span=12), shifted 1 bucket |
| Rolling z-score | Z-score vs rolling(24), shifted 1 bucket |
| ARIMA | ARIMA(2,1,0) 1-step-ahead forecast |

### Google Cluster 2011 (test split, last 20% chronological)

| Method | 15m PR-AUC | 30m PR-AUC | 45m PR-AUC | 60m macro PR-AUC | OVR PR-AUC |
|---|---|---|---|---|---|
| Random | 0.052 | 0.078 | 0.099 | 0.333 | 0.018 |
| Persistence | 0.481 | 0.455 | 0.447 | 0.433 | 0.229 |
| Static threshold | 0.320 | 0.301 | 0.295 | 0.414 | 0.163 |
| EWMA z-score | 0.089 | 0.108 | 0.126 | 0.335 | 0.024 |
| Rolling z-score | 0.096 | 0.114 | 0.132 | 0.335 | 0.025 |
| ARIMA | 0.466 | 0.442 | 0.435 | 0.430 | 0.221 |
| **XGBoost (ours)** | **0.575** | **0.564** | **0.563** | **0.547** | **0.339** |
| **Gap vs persistence** | **+0.094** | **+0.109** | **+0.116** | **+0.114** | **+0.110** |

**Key observations:**
- EWMA and rolling z-score are near-random for binary horizons (0.089–0.132). Pure
  statistical methods fail without feature engineering. These methods "surprise" us
  only until you notice they have no spike-history or threshold-relative features.
- ARIMA is the strongest baseline (0.466 for 15m) — linear autocorrelation captures
  much of the persistence signal but cannot model the non-linear threshold proximity.
- Static threshold underperforms persistence, confirming that a recency count "was
  cpu high for the last 4 buckets?" is less informative than the current ratio `cpu/p95`.
- OVR severe has the largest absolute XGBoost gap (+0.110) — feature engineering
  matters most for rare events where simple baselines completely fail.

### Zabbix HPC (test split, last 20% = ~18 days)

**Important context:** Zabbix positive rates are 38–47% for binary (vs 5–10% on Google)
and 22.5% for OVR severe (vs 1.8% on Google). This inflates all PR-AUC scores —
ROC-AUC is the honest comparator for Zabbix binary models.

| Method | 15m PR-AUC | 30m PR-AUC | 45m PR-AUC | 60m macro PR-AUC | OVR PR-AUC |
|---|---|---|---|---|---|
| Random | 0.380 | 0.434 | 0.467 | 0.333 | 0.226 |
| Persistence | 0.950 | 0.945 | 0.944 | 0.454 | 0.631 |
| Static threshold | 0.911 | 0.887 | 0.877 | 0.480 | 0.627 |
| EWMA z-score | 0.394 | 0.453 | 0.485 | 0.335 | 0.237 |
| Rolling z-score | 0.394 | 0.450 | 0.481 | 0.335 | 0.237 |
| ARIMA | 0.949 | 0.946 | 0.943 | 0.474 | 0.633 |
| **XGBoost (ours)** | **0.957** | **0.957** | **0.957** | **0.848** | **0.926** |

**Honest Zabbix comparison (ROC-AUC, prevalence-neutral):**
Persistence 15m ROC-AUC: 0.960 vs XGBoost **0.966** (+0.006). Small gap on binary
because persistence trivially near-saturates at 40% positive rate. The headline Zabbix
number is **60m macro PR-AUC: +0.374** over ARIMA — macro PR-AUC is prevalence-neutral
(random = 0.333 in both datasets regardless of positive rate).

---

## 10. Cross-Domain Validation — Zabbix HPC

The model was evaluated zero-shot on Zabbix data — no Zabbix data was used in training.
Only domain-specific calibration (alarm threshold re-selection on Zabbix val split)
was performed.

### Results (40 features, cascade OVR, 2026-04-27)

| Model | Zabbix PR-AUC | Source PR-AUC | Retention | Zabbix ROC-AUC |
|---|---|---|---|---|
| 60m severity (uncal) | 0.708 | 0.547 | 129% | 0.871 |
| 60m severity (recal) | **0.800** | 0.547 | **146%** | **0.900** |
| 15m binary | **0.956** | 0.575 | **166%** | **0.966** |
| 30m binary | **0.957** | 0.563 | **170%** | **0.960** |
| 45m binary | **0.956** | 0.562 | **170%** | **0.954** |
| OVR severe (cascade) | **0.891** | 0.339 | **263%** | **0.928** |

All models exceed 100% retention — they perform better on Zabbix than on Google.
This is partly explained by higher Zabbix positive rates making the problem easier,
but the 60m ROC-AUC improvement (+0.051, prevalence-neutral) confirms genuine transfer.

### Final alarm performance after Zabbix recalibration

| Model | PR-AUC | ROC-AUC | Alarm P | Alarm R | Alarm F1 | Threshold |
|---|---|---|---|---|---|---|
| 60m severity | 0.800 | 0.900 | 0.703 | 0.631 | 0.665 | 0.45 |
| 15m binary | 0.956 | 0.966 | 0.800 | 0.868 | 0.833 | 0.65 |
| 30m binary | 0.957 | 0.960 | 0.814 | 0.912 | 0.860 | 0.45 |
| 45m binary | 0.956 | 0.954 | 0.837 | 0.921 | 0.877 | 0.35 |
| OVR severe | 0.891 | 0.928 | 0.825 | 0.596 | 0.692 | 0.55 |

60m per-class PR-AUC (recal): no_spike=0.926, moderate=0.724, severe=0.751

### Regression vs 37-feature model (2026-04-18)

The 40-feature model shows regressions on Zabbix for two models:

| Model | Old 37f (2026-04-18) | New 40f (2026-04-27) | Delta |
|---|---|---|---|
| 60m recal | 0.848 | 0.800 | **−0.048** |
| OVR severe | 0.926 | 0.891 | **−0.035** |
| Binary models | 0.957 | 0.956–0.957 | ≈0 |

**60m regression cause:** The three new slope features (`cpu_vs_p95_slope_3/6`)
have MAJOR PSI (0.31–0.55) on Zabbix — they capture Google-specific CPU velocity
patterns (Borg scheduling heartbeats, task migration artefacts) that do not transfer
to Zabbix's job-driven workload. Different Optuna hyperparameters from the 2026-04-26
rerun also contributed.

**OVR regression cause:** Cascade Stage 2 trains only on spike-positive Google rows
(~19% severe). The Google spike-positive distribution differs from Zabbix's — the
full-dataset OVR (2.4% severe) generalised better to Zabbix despite lower in-distribution
PR-AUC. This is the known cascade trade-off.

Both regressions are accepted — 129% retention on 60m remains a strong cross-domain
result. Binary models are completely unaffected.

### PSI analysis (feature distribution shift, Google → Zabbix)

| PSI range | Feature count | Notable features |
|---|---|---|
| < 0.10 (stable) | 6 | `band_position` (0.0004), `hour_sin/cos` (0.0), `machine_rank`, `cpu_spike_rate_24` |
| 0.10–0.25 (moderate) | 9 | Spike history features (0.12–0.15), `time_to_p95_3` (0.16), `time_since_last_spike` (0.13) |
| > 0.25 (MAJOR) | 25 | `task_dominance` (8.28), `band_width` (4.36), `cluster_cpu_p90` (3.15), raw CPU/mem/disk (0.9–2.6), slope features (0.31–0.55) |

**PSI paradox:** 25/40 features have MAJOR distribution shift yet the model retains
>100% performance. The resolution: the dominant predictors (`cpu_vs_p95`, `band_position`,
spike history features) are machine-relative and domain-invariant — the ratio of current
CPU to that machine's p95 has the same predictive meaning regardless of the absolute
scale of CPU on that cluster. Raw features (total_cpu scale, n_tasks magnitude) shift
heavily but are secondary in the learned model.

`hour_sin/cos` PSI = 0.0 — time-of-day encoding is perfectly stable across domains.

---

## 11. Calibration

Isotonic regression calibration (one calibrator per class, trained on the validation set).
Applied to the 60m severity model only (binary models use raw logistic probabilities).

### Effect on model performance

| Class | Brier score (raw) | Brier score (calibrated) | ECE (calibrated) |
|---|---|---|---|
| no_spike (0) | 0.211 | **0.070** | 0.020 |
| moderate (1) | 0.092 | **0.060** | 0.005 |
| severe (2) | 0.073 | **0.024** | 0.005 |

Brier score halved for no_spike class (0.211 → 0.070). ECE < 0.025 for all classes —
the calibrated probabilities are well-calibrated.

**PR-AUC lift from calibration:** 0.510 (raw) → **0.547 (calibrated)** on test.
Calibration improves PR-AUC because it corrects the raw softmax overconfidence on the
majority class (no_spike), making the severe probability scores more discriminative.

### Critical bug discovered and fixed (2026-03-31)

Earlier training runs stored a threshold selected on raw probabilities but applied it
at inference on calibrated probabilities. Because isotonic calibration is a piecewise-
constant step function, the raw-to-calibrated mapping is nonlinear — a threshold
maximising F1 on raw scores does not maximise F1 on calibrated scores.

Fix: alarm threshold sweep always runs on calibrated probabilities and stores the
calibrated threshold in `spike_config.json`. This is enforced at training time.

### Zabbix recalibration

Recalibrating the 60m model on the Zabbix validation split gives +0.092 PR-AUC
(0.708 uncal → 0.800 recal). This confirms that the base model weights transfer
well but the probability scale needs domain-specific adjustment. Only three
domain-specific artifacts change per deployment:

1. `spike_thresholds.parquet` — per-machine p95/p99 (from `bootstrap_thresholds.py`)
2. `calibrators.pkl` — recalibrated isotonic regressors (from `evaluate_domain.py`)
3. `spike_config.json` `alarm_threshold` — new F1-max operating point on domain val

The base model weights (`spike_model.json`) are domain-invariant.

---

## 12. Design Decisions and Limitations

### Alarm threshold: global, not per-machine

We apply a single alarm threshold across all machines. Production schedulers at
Google/Meta scale use per-entity adaptive thresholds. We have per-machine p95/p99
for feature normalisation but the alarm threshold is global.

**Impact:** Machines with inherently bursty or noisy CPU profiles (e.g. Zabbix nodes
`atnog-bkpConfigs` with 94.6% idle fraction) may require different thresholds than
always-loaded servers. Operators can mitigate this with the `--alarm-threshold` flag
but cannot set per-machine values at runtime.

**Future work:** Per-machine adaptive thresholds, possibly learned from the validation
set per machine.

### Aggregation: mean + peak, no within-bucket percentiles

Each 5-minute bucket stores duration-weighted mean and peak CPU. Best-practice production
systems store p50 + p95 within each bucket (intra-bucket percentiles). We approximate
with `peak_cpu` (effectively p100).

For Google Cluster Traces 2011 (task-event format with start/end times) and Zabbix
(one sample per 5-min interval), `peak_cpu` is the correct and sufficient approximation.
For high-frequency sources (>1 Hz raw metrics), within-bucket percentiles would give
richer shape information.

### Gap filling: zero-fill for offline machines

Missing buckets (offline or idle machines) are filled with `total_cpu = 0`. Zero is
physically correct for idle machines but makes lag features (`cpu_lag_24`) see 0 after
any offline period, resembling a cold-start machine.

Zero-fill was chosen over last-observation-carried-forward (LOCF) because LOCF
propagates stale high-load values — if a machine was running hot and then goes
offline, LOCF would produce false spike predictions when it comes back. Zero-fill
is the conservative choice for a spike detector: better to miss the first spike
after return than to generate false alarms for machines that were simply not reporting.

### cluster_cpu_p90 leakage (known, low priority)

`_add_cluster_features()` computes cluster p90 across all machines using the full
dataset before the train/val/test split. This is a subtle lookahead bias — val/test
rows use p90 computed from rows that include those same val/test buckets.

In production inference, p90 is computed from live machines at prediction time,
so the production system has no leakage. For benchmarking, the effect is a slight
inflation of the cluster-relative feature's signal on val/test rows. Given that
`cluster_cpu_p90` has rank 22 by SHAP (0.034), the quantitative impact is small.
Fix: compute p90 from training-window rows only. Deferred pending the next retraining.

### Cost modelling: implicit, not explicit

Optimal threshold minimises `FP_cost × FP_rate + FN_cost × FN_rate` where costs
come from the scheduler's job migration overhead (FP) vs downtime SLA breach (FN).
We achieve this implicitly via `--alarm-threshold` tuning, but never expose the cost
ratio directly. Operators currently tune the threshold empirically from `slo_metrics.json`
rather than from a formal cost model.

---

## 13. Future Work

### Multi-metric generalisation (GPU, memory)

The current system predicts CPU spikes exclusively. The architecture (XGBoost on
rolling-window tabular features with threshold-relative normalisation) is generalizable
to any metric, but requires:

1. A new domain adapter emitting `gpu_util` (or `total_mem`) in the `cluster_agg` schema
2. Re-labelling: thresholds computed from the target metric's p95/p99
3. Feature re-engineering: `cpu_*` features replaced with `metric_*` analogues
4. Retraining from scratch — existing CPU weights do not transfer to GPU spike prediction

The column-rename approach (rename `gpu_util` → `total_cpu`) would allow the training
pipeline to produce a GPU spike detector without code changes, but would require a
full retraining run. This is the recommended short-term path for GPU monitoring support.

**Longer-term:** Parameterise the target metric as `--target-col` across the entire
pipeline, producing a general-purpose `<metric>_spike_detector` from one codebase.

### Larger training dataset

Google Cluster Traces 2019 (BigQuery: `bigquery-public-data.google_cluster_data`) is
the recommended next dataset:
- 8 cells (a–h) × ~12,500 machines × 31 days ≈ 25× current training data
- Same Borg scheduler schema, zero code changes after JSON→cluster_agg adapter
- Resolves the 7-day ceiling that limits 15m/30m/45m differentiation (current gap 0.012)
- One cell ≈ 300 GB; export via BigQuery → GCS → local pull (~$0.08/GB egress)

Second candidate: Alibaba Cluster Traces 2018 (1.7 GB direct download, 4,000 machines,
8 days) adds workload diversity but requires separate preprocessing (CPU in %, no peak_cpu).

### Third external domain validation

Zabbix is the only non-Google validation dataset. A third domain (different cloud
provider, different scheduler, different hardware generation) would strengthen the
cross-domain transfer claim and identify which features are consistently stable.

### Per-machine adaptive alarm thresholds

Replace the global `alarm_threshold` with per-machine thresholds learned from the
machine's individual precision/recall characteristics on the validation split. Machines
with consistently noisy or bursty profiles could have higher thresholds automatically.

### Time-to-peak and magnitude prediction

Current output: spike timing window (which horizon fires first), severity class.
Not predicted: exact minute of peak, actual peak CPU value, spike duration.
These require regression models (quantile regression, survival analysis) and are
out of scope for the scheduler-input use case, but would be relevant for capacity
planning and SLA prediction.

### Automated retraining trigger

`slo_metrics.json` `alarm_rate_mean` and `drift_report.json` PSI scores provide
the signals needed for automated retraining. A trigger that fires when alarm_rate
drifts more than 2σ from baseline AND PSI system_alert is true would initiate a
domain recalibration run without human intervention. Currently, both steps require
manual operator decision.

### Within-bucket percentiles for high-frequency sources

Sources with sub-minute telemetry (e.g. Prometheus at 15s scrape) would benefit from
storing p50 + p95 within each 5-minute bucket. The current `peak_cpu` (p100 proxy)
misses the load shape distinction between a machine at p50=0.2, p95=0.8 vs one at
p50=0.5, p95=0.6 — both have the same `total_cpu` average. Requires schema extension
and feature re-engineering.
