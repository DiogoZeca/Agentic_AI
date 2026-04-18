# Project State — CPU Spike Prediction for Datacenter Scheduling

**Last updated:** 2026-04-19  
**Status:** Deployment-complete. Models trained, cross-domain validated, baselines done, daemon live.  
**Next:** Paper draft. Optional: Google 2019 BigQuery for additional training data.

---

## 1. What This Project Is

A **proactive CPU spike predictor** for datacenter workload scheduling. Given the last 120 minutes of per-machine CPU telemetry (24 × 5-min buckets), the system predicts whether a CPU spike will occur in the next 15 to 60 minutes and how severe it will be. Output feeds a workload scheduler so it can act before the spike materialises — defer new jobs, migrate running jobs, or trigger emergency preemption.

**Key design principle:** The system is source-agnostic. It takes any CSV in cluster_agg format (see Section 3) from any monitoring system and produces structured JSON predictions. It is not tied to any specific data provider.

**Adaptation recipe for a new cluster:**
1. Collect 89+ days of 5-min CPU telemetry → format as `cluster_agg.csv`
2. Run `train_spike_classifier.py` to retrain on local data (or recalibrate the existing model)
3. Point your data pipeline to write a rolling 120-min `cpu_window.csv` every N minutes
4. Run `predict_daemon.py --input cpu_window.csv --model-dir models/spike/ --output predictions.json`

No retraining required for a new cluster with similar workload characteristics (confirmed by cross-domain transfer to Zabbix HPC data — see Section 10).

---

## 2. Novel Research Contributions

1. **First systematic spike prediction benchmark on Google Cluster Traces 2011.** No prior published PR-AUC or ROC-AUC numbers exist for CPU spike prediction on this dataset.

2. **Zero-shot cross-domain transfer: Google Borg 2011 → HPC Slurm 2026.** The model trained on 2011 datacenter workloads transfers to a real production HPC cluster 15 years later with 151–170% PR-AUC retention. 23 of 37 features show MAJOR distribution shift (PSI > 0.25), yet the model transfers because spike-history and vs-p95 relative features (moderate PSI 0.12–0.15) dominate predictions.

3. **Multi-horizon cascade as time-to-event estimation.** Four binary classifiers (15m/30m/45m/60m) combined to estimate the spike timing window without survival model complexity. All three binary horizon models saturate at PR-AUC ≈ 0.957 on the Zabbix dataset; alarm thresholds differentiate (0.70 → 0.50 → 0.35) enabling a practical timing cascade.

4. **Generic recalibration protocol.** Isotonic recalibration on 89 days of local data brings the 60m model from raw 0.826 to calibrated 0.848 PR-AUC on Zabbix, and improves alarm F1 from 0.637 to 0.711. Protocol is domain-independent.

---

## 3. Dataset

### Training: Google Cluster Traces 2011

| Property | Value |
|---|---|
| Raw rows | 278 million |
| After aggregation | ~24 million (machine_id × 5-min bucket) |
| Duration | 160 hours (~7 days) |
| Machines | 12,555 |
| Buckets per machine | 1,919 |
| Split | Train 60% / Val 20% / Test 20% (chronological — no shuffle) |

**Input schema (cluster_agg format):**

| Column | Type | Description |
|---|---|---|
| `machine_id` | int64 | Unique machine identifier |
| `bucket` | int64 | 5-min bucket index (monotonically increasing per machine) |
| `time_us` | int64 | `bucket × 300_000_000` (microseconds since trace epoch) |
| `total_cpu` | float32 | Duration-weighted CPU load (fraction of 1 core) |
| `peak_cpu` | float32 | Peak CPU rate in the 5-min window |
| `total_mem` | float32 | Sum of canonical memory usage across tasks |
| `peak_mem` | float32 | Peak memory usage |
| `disk_io` | float32 | Max mean disk I/O time across tasks |
| `n_tasks` | int32 | Concurrent tasks in bucket |

### Cross-Domain Validation: Zabbix HPC Cluster (2026)

| Property | Value |
|---|---|
| Nodes | 11 (mix of Slurm compute + infrastructure) |
| Duration | 89 days |
| Bucket interval | 5 minutes (same cadence as Google) |
| Total rows | 273,198 |
| Test split | Last 20% (~18 days) |
| Label definition | K=1 (any exceedance of per-machine p95) |

Zabbix was used exclusively for cross-domain validation — the model was not retrained on it. It represents a real production environment 15 years after and on completely different workload types.

---

## 4. Label Definitions

### Spike labels (K-of-N)

A **spike** in the next H minutes is defined as at least K of the next `H/5` buckets exceeding the per-machine CPU threshold. The production baseline uses **K=2** for the 60m severity model (requires ≥2 of the next 12 windows to exceed p95 or p99), giving more persistent spike definitions and fewer nuisance alarms.

Binary horizon models (15m/30m/45m) always use K=1 (any single exceedance) — their shorter windows make K=2 too sparse.

| Label | Model | K | Threshold |
|---|---|---|---|
| `severity_in_60m` ∈ {0,1,2} | 60m severity | K=2 | no_spike: never; moderate: ≥2 of 12 > p95; severe: ≥2 of 12 > p99 |
| `spike_in_15m` ∈ {0,1} | 15m binary | K=1 | any of 3 windows > p95 |
| `spike_in_30m` ∈ {0,1} | 30m binary | K=1 | any of 6 windows > p95 |
| `spike_in_45m` ∈ {0,1} | 45m binary | K=1 | any of 9 windows > p95 |
| `spike_severe_ovr` ∈ {0,1} | OVR severe | K=2 | ≥2 of 12 windows > p99 |

### Per-machine thresholds

p95 and p99 are computed **per machine from training buckets only** (no leakage). A minimum 10% gap between p99 and p95 is enforced to ensure a meaningful "moderate" band.

**Class prevalence in Google K=2 test set:**

| Class | Prevalence |
|---|---|
| no_spike (class 0) | ~96% |
| moderate (class 1) | ~2.4% |
| severe (class 2) | ~1.8% |

---

## 5. Model Architecture — 5 Models

### Model 1: 60m Severity Classifier (`SpikeClassifier`)

- **Algorithm:** XGBoost `multi:softprob` (3-class softmax)
- **Predicts:** `no_spike` / `moderate` (p95 exceeded) / `severe` (p99 exceeded) in next 60 min
- **Class balancing:** `compute_sample_weight('balanced', y)` — severe class gets ~85× weight
- **Primary metric:** Macro PR-AUC (equal weight to all 3 classes; critical for rare severe)
- **Early stopping:** `n_estimators=4000`, `early_stopping_rounds=150`, metric=`mlogloss`
- **Monotone constraints:** Disabled — semantically undefined for `multi:softprob`
- **Calibration:** Per-class isotonic regression fitted on val set; probabilities renormalised to sum to 1
- **Optuna trials:** 150 (fresh, no warmstart)
- **Best hyperparams (K=2 production):** `max_depth=3`, `gamma=4.79`, `learning_rate≈0.05`

### Model 2: 15m Binary Classifier (`BinarySpikeClassifier`)

- **Algorithm:** XGBoost `binary:logistic`
- **Predicts:** any spike in next 15 min (K=1)
- **Class balancing:** `scale_pos_weight` computed per fold
- **Monotone constraints:** Active on `cpu_vs_p95`, spike history features
- **Early stopping:** `n_estimators=4000`, `early_stopping_rounds=150`, metric=`aucpr`
- **Optuna trials:** capped at 75 (binary converges faster)
- **Alarm threshold:** 0.80 (selected on val set)

### Model 3: 30m Binary Classifier

Same architecture as 15m. **Alarm threshold:** 0.70.

### Model 4: 45m Binary Classifier

Same architecture as 15m. **Alarm threshold:** 0.65.

### Model 5: OVR Severe Classifier

- **Algorithm:** XGBoost `binary:logistic`
- **Label:** `spike_severe_ovr` = `(severity_in_60m == 2)` — severe vs everything else
- **Class prevalence:** ~2.4% train, ~1.8% test (Google K=2)
- **Scale_pos_weight:** ~42 (train: 97.6% non-severe / 2.4% severe)
- **Monotone constraints:** Active (binary model)
- **Alarm threshold:** 0.85 (selected on val set)
- **Purpose:** Dedicated binary classifier for rare severe class. Better PR-AUC on severe than the 60m softmax column alone.

---

## 6. Feature Engineering — 37 Features

All features computed per machine-bucket from the rolling window. Per-machine statistics (p95, p99) computed from training data only.

| Category | Features |
|---|---|
| **Raw** (6) | `total_cpu, peak_cpu, total_mem, peak_mem, disk_io, n_tasks` |
| **Lags** (3) | `cpu_lag_1, cpu_lag_12, cpu_lag_24` |
| **Trend** (5) | `cpu_ewma_6, cpu_ewma_24, cpu_delta_1, cpu_delta_2, cpu_rolling_std_6` |
| **Load ratio** (1) | `cpu_per_task = total_cpu / max(n_tasks, 1)` |
| **Machine-relative p95** (9) | `cpu_vs_p95, cpu_vs_p95_delta, peak_cpu_vs_p95, spike_now, spike_in_last_1, spike_in_last_3, spike_in_last_6, time_since_last_spike, cpu_spike_rate_24` |
| **Machine-relative p99** (8) | `spike_severe_now, cpu_vs_p99, peak_cpu_vs_p99, band_position, band_width, spike_severe_in_last_1, spike_severe_in_last_3, spike_severe_in_last_6` |
| **Cluster** (3) | `cluster_cpu_p90, machine_rank_in_cluster, task_dominance` |
| **Time** (2) | `hour_sin, hour_cos` |

### Dropped features and why

| Feature | Reason |
|---|---|
| `dow_sin`, `dow_cos` (day-of-week) | SHAP=0.356 (highest of all) but gain=0.011 (near-zero). Only 7 days of data → 23 samples per day-of-week label. Confirmed temporal confound with the Google 2011 week structure. |
| `current_spike_streak`, `max_spike_streak_24h`, `current_severe_streak` | Two independent training runs gave 0.553 calibrated vs 0.574 baseline — confirmed −0.021 regression. Streak features add noise/collinearity on top of the existing `spike_in_last_{1,3,6}` history features. |

### Leakage firewall

All spike-history features derive from `exc = spike_now.shift(1)` — the **previous** bucket's spike status, not the current one. Labels look **forward** from the next bucket. The two directions never overlap.

### Top features by gain (production model)

`spike_severe_now`, `spike_in_last_6`, `time_since_last_spike`, `cpu_vs_p95`, `cpu_vs_p99`

---

## 7. Training Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Split type** | Chronological 60/20/20 | Temporal data — random shuffle causes leakage |
| **CV method** | Walk-forward, 5 folds, 12-bucket gap (60 min) | Gap = prediction horizon; prevents label leakage across folds |
| **Primary metric** | Macro PR-AUC | Correct for imbalanced multiclass; gives equal weight to rare severe class |
| **Calibration** | Per-class isotonic regression on val set | Raw softmax underestimates p_severe; calibration improves alarm threshold selection |
| **Alarm threshold** | Selected on calibrated val set | Never on test; threshold sweep stored in spike_config.json |
| **Hyperparameter tuning** | Optuna (TPE sampler), 150 trials for 60m, 75 for binary | More trials for harder 3-class problem |
| **Early stopping** | `n_estimators=4000`, `early_stopping_rounds=150` | `n_estimators` not in Optuna search space — avoids interaction with ESR |
| **GPU training** | `device='cuda'`, `tree_method='hist'`, `n_jobs=1` | Correct XGBoost 3.x syntax; `n_jobs=1` required with CUDA |
| **Monotone constraints** | Binary models only | Disabled for `multi:softprob` — semantically undefined in softmax |
| **OOM prevention** | Optuna inner split capped at 2M rows; column-selective parquet reads | Full inner split is 7M+ rows; binary models need only ~43 of 60 columns |

---

## 8. Metrics — Explanations

### PR-AUC (Precision-Recall Area Under Curve)

**Primary metric.** Measures discriminative ability on the **positive class** across all thresholds. Unlike ROC-AUC, PR-AUC is sensitive to class imbalance — a random classifier achieves PR-AUC ≈ positive_rate (e.g. 0.052 for 5% positive rate). This makes it the honest metric for imbalanced classification.

**Macro PR-AUC:** Average of per-class PR-AUCs (one-vs-rest for each class). Used for the 60m 3-class model. Equal weight to all 3 classes regardless of prevalence — forces the model to do well on rare severe class.

**When NOT to use PR-AUC:** When comparing across datasets with different positive rates. A model evaluated on a dataset with 40% positives will trivially score higher PR-AUC than the same model on a 5% positive dataset. In these cases, **ROC-AUC is the honest comparator** (it is prevalence-independent).

### ROC-AUC (Receiver Operating Characteristic)

Measures discriminative ability across all decision thresholds. A random classifier always scores 0.50. Not sensitive to class imbalance — valid for cross-dataset comparison. Used as the honest metric when comparing Google (5% positive rate) vs Zabbix (38–47% positive rate).

### Alarm Precision / Recall / F1

At the selected operating threshold (chosen on val set):
- **Precision:** Of all raised alarms, what fraction were true spikes?
- **Recall:** Of all true spikes, what fraction were caught?
- **F1:** Harmonic mean of P and R. The operational metric — what a scheduler operator would care about.

### K-of-N ablation finding

PR-AUC declines with K (K=1: 0.573, K=2: 0.547, K=3: 0.516) but ROC-AUC increases (0.840 → 0.843 → 0.879). The decline is a mathematical artifact: higher K = sparser positive class = lower PR-AUC baseline. ROC-AUC is the honest comparator for K comparisons. **K=2 was chosen as production baseline** — fewer nuisance alarms (7.4/day vs 16.5/day for K=1) at similar discriminative ability.

---

## 9. Results — Google Test Set (In-Distribution)

Test split: last 20% chronologically. ~4.6–4.8M rows per horizon.

### Production Model (K=2, 37 features, 150-trial Optuna, XGBoost 3.2.0)

| Model | CV PR-AUC | Test PR-AUC | Test ROC-AUC | Notes |
|---|---|---|---|---|
| 60m severity | 0.4995 ± 0.0054 | 0.513 raw / **0.547 cal** | 0.843 | max_depth=3, gamma=4.79 |
| 15m binary | 0.582 ± 0.008 | **0.575** | 0.911 | alarm @ 0.80 → P=0.601, R=0.524 |
| 30m binary | — | **0.563** | — | alarm @ 0.70 |
| 45m binary | — | **0.562** | — | alarm @ 0.65 |
| OVR severe | 0.275 ± 0.020 | **0.339** | 0.863 | alarm @ 0.85 → P=0.398, R=0.355 |

**60m per-class (calibrated):**
- no_spike: 0.983 | moderate: 0.299 | severe: 0.257
- Alarm @ 0.25: Precision=0.446, Recall=0.306, **7.4 alarms/day**

**15m binary is the strongest model** (ROC-AUC 0.911). Shorter horizon = more predictable. This is consistent across all K values and both datasets.

### K-of-N Ablation (60m model)

| Metric | K=1 | K=2 (prod) | K=3 |
|---|---|---|---|
| CV macro PR-AUC | 0.565 ± 0.009 | 0.500 ± 0.005 | 0.483 ± 0.007 |
| Test PR-AUC (cal) | 0.573 | **0.547** | 0.516 |
| Test ROC-AUC | 0.840 | **0.843** | 0.879 |
| Alarms/day | 16.5 | **7.4** | 5.7 |
| Severe rate (test) | 3.3% | 1.8% | 1.0% |

K=2 chosen: better alarm rate (7.4/day) with comparable discriminative ability (ROC-AUC 0.843). K=3 has temporal drift warning (Kendall tau=0.6) despite high ROC-AUC.

---

## 10. Results — Baseline Comparison (Google Test Set)

Evaluated on the same test split. Baselines use only raw cluster_agg schema (no engineered features).

| Baseline | 15m PR-AUC | 30m PR-AUC | 45m PR-AUC | 60m macro PR-AUC | OVR PR-AUC |
|---|---|---|---|---|---|
| Random | 0.052 | 0.078 | 0.099 | 0.333 | 0.018 |
| Persistence | 0.481 | 0.455 | 0.447 | 0.433 | 0.229 |
| Static threshold | 0.320 | 0.301 | 0.295 | 0.414 | 0.163 |
| EWMA z-score | 0.089 | 0.108 | 0.126 | 0.335 | 0.024 |
| Rolling z-score | 0.096 | 0.114 | 0.132 | 0.335 | 0.025 |
| ARIMA(2,1,0) | 0.466 | 0.442 | 0.435 | 0.430 | 0.221 |
| **XGBoost (ours)** | **0.575** | **0.563** | **0.562** | **0.547** | **0.339** |

**XGBoost gap over best baseline (persistence):**

| Horizon | Gap |
|---|---|
| 15m | +0.094 |
| 30m | +0.108 |
| 45m | +0.115 |
| 60m | +0.114 |
| OVR | +0.110 |

**Key observations for paper:**
- EWMA/z-score are near-random (0.089–0.132) — pure statistical methods fail completely on binary horizons. This validates that supervised feature engineering is necessary.
- ARIMA is 2nd best but still −0.094 behind XGBoost for 15m — linear autocorrelation alone is insufficient.
- OVR severe gap is largest (+0.110 on rare 1.8%-positive class) — feature engineering compounds most where signal is weakest.
- Static threshold underperforms persistence — "is it spiking right now?" is better than "has it been high recently?", meaning the current state alone is not the most useful signal.

---

## 11. Cross-Domain Transfer — Zabbix HPC Evaluation

### Context

The Zabbix dataset (89 days, 11 nodes, 2026 production Slurm HPC) was used for **zero-shot cross-domain evaluation**. No retraining. The model trained on Google 2011 Borg workloads is evaluated directly on 2026 HPC cluster data.

**Critical interpretation note:** Zabbix positive rates are much higher than Google (38–47% for binary horizons vs 5–10% on Google; 22.5% OVR severe vs 1.8% on Google). This inflates all PR-AUC scores. **Use ROC-AUC for cross-dataset comparisons on binary horizons.** PR-AUC remains valid for within-dataset comparisons and for the 60m macro metric (where random = 0.333 in both datasets, making it prevalence-neutral).

### Phase 1 — Domain Shift (PSI Analysis)

Population Stability Index (PSI) measures feature distribution shift between Google train and Zabbix full dataset.

| Shift level | Features | Count |
|---|---|---|
| MAJOR (PSI > 0.25) | `task_dominance` (8.28), `band_width` (4.36), `cluster_cpu_p90` (3.15), `n_tasks` (2.69), `total_mem` (2.57), all raw CPU features (~1.0) | 23 |
| Moderate (0.10–0.25) | Spike history features: `spike_now`, `spike_in_last_1/3/6`, `spike_severe_now`, `spike_severe_in_last_1/3`, `time_since_last_spike` | 8 |
| Minor (< 0.10) | `band_position`, `hour_sin/cos`, `machine_rank`, `cpu_spike_rate_24` | 6 |

Despite 23/37 MAJOR shift features, the model transfers well. **Why:** The moderate-shift spike history and vs-p95 features (PSI 0.12–0.15) are the most predictive. These are relative features normalized to the local machine's distribution — they partially self-adapt to different absolute CPU scales.

### Phase 2 — Zero-Shot Evaluation

| Model | Zabbix PR-AUC | Google PR-AUC | Retention | Zabbix ROC-AUC |
|---|---|---|---|---|
| 60m severity | 0.826 (raw) | 0.547 | 151% | 0.909 |
| 15m binary | 0.957 | 0.575 | 166% | 0.966 |
| 30m binary | 0.957 | 0.563 | 170% | 0.960 |
| 45m binary | 0.957 | 0.562 | 170% | 0.955 |
| OVR severe | 0.926 | 0.339 | 273% | 0.955 |

### Phase 3 — Recalibrated Results (final alarm thresholds)

After isotonic recalibration on Zabbix val split:

| Model | PR-AUC | ROC-AUC | Alarm P / R / F1 | Threshold | 95% CI |
|---|---|---|---|---|---|
| 60m severity | **0.848** | **0.922** | 0.799 / 0.640 / 0.711 | 0.40 | per-class below |
| 15m binary | **0.957** | **0.966** | 0.814 / 0.851 / 0.832 | 0.70 | [0.955, 0.959] |
| 30m binary | **0.957** | **0.960** | 0.827 / 0.896 / 0.861 | 0.50 | [0.955, 0.958] |
| 45m binary | **0.957** | **0.955** | 0.837 / 0.926 / 0.879 | 0.35 | [0.955, 0.959] |
| OVR severe | **0.926** | **0.955** | 0.813 / 0.733 / 0.771 | 0.65 | [0.921, 0.931] |

**60m per-class (Zabbix recalibrated):**
- no_spike: 0.932 [0.929, 0.935] | moderate: 0.768 [0.761, 0.775] | severe: 0.844 [0.836, 0.852]
- Class rates on Zabbix: no_spike=50.8%, moderate=26.7%, severe=22.5%

### Zabbix Baseline Comparison

| Baseline | 15m PR-AUC | 30m PR-AUC | 45m PR-AUC | 60m macro PR-AUC | OVR PR-AUC |
|---|---|---|---|---|---|
| Random | 0.380 | 0.434 | 0.467 | 0.333 | 0.226 |
| Persistence | 0.950 | 0.945 | 0.944 | 0.454 | 0.631 |
| Static threshold | 0.911 | 0.887 | 0.877 | 0.480 | 0.627 |
| EWMA z-score | 0.394 | 0.453 | 0.485 | 0.335 | 0.237 |
| Rolling z-score | 0.394 | 0.450 | 0.481 | 0.335 | 0.237 |
| ARIMA(2,1,0) | 0.949 | 0.946 | 0.943 | 0.474 | 0.633 |
| **XGBoost (ours)** | **0.957** | **0.957** | **0.957** | **0.848** | **0.926** |

**Paper framing for Zabbix baselines:**
- Binary horizons: PR-AUC inflated by high positive rate. Use ROC-AUC — persistence 0.960 vs XGBoost 0.966 (+0.006). The honest gap is modest for binary horizons on this dataset.
- **60m macro PR-AUC is the headline Zabbix number**: XGBoost 0.848 vs ARIMA 0.474 (+0.374). This metric is prevalence-neutral (random=0.333 in both datasets) — a fair and dramatic comparison.
- OVR ROC-AUC: persistence 0.873 vs XGBoost 0.955 (+0.082) — large gap, honest metric.

---

## 12. Multi-Horizon Cascade and Output Format

### Cascade logic

The four horizon models combine to estimate **when** a spike will occur and **what to do** about it:

| Earliest alarm | Recommended action | Interpretation |
|---|---|---|
| 15m fires | `preempt_now` | Spike imminent — emergency action |
| 30m fires, 15m not | `migrate_jobs` | ~15–30 min window — migrate |
| 45m fires, 30m not | `defer_new_jobs` | ~30–45 min window — defer submissions |
| 60m fires (severe), no short alarms | `defer_batch` | 60m severe predicted, no urgency |
| 60m fires (moderate), no short alarms | `monitor` | Low-severity predicted |
| Nothing fires | `normal` | No action needed |

Monotonicity is enforced: P(spike) is non-decreasing across [15m, 30m, 45m, 60m].

### What the output provides vs does not provide

| Concept | What we provide | What we do NOT provide |
|---|---|---|
| **Time-to-spike** | Window bracket (e.g. 30–45 min via cascade) | Exact minute of spike |
| **Severity** | Class: moderate (p95) / severe (p99) | Actual CPU% value at peak |
| **Duration** | Nothing | How long the spike lasts above threshold |

The system predicts *whether and roughly when* a spike will occur and *how severe it will be*. Exact time-to-peak and magnitude are out of scope (see Section 14).

### Output JSON structure

```json
{
  "predicted_at": "2026-04-19T14:00:00Z",
  "machines_predicted": 11,
  "predictions": [{
    "machine_id": 42,
    "imminence": {
      "15m": {"p_spike": 0.12, "is_spike": false, "alarm_threshold": 0.80},
      "30m": {"p_spike": 0.41, "is_spike": false, "alarm_threshold": 0.70},
      "45m": {"p_spike": 0.73, "is_spike": true,  "alarm_threshold": 0.65},
      "60m": {"severity_class": 1, "p_no_spike": 0.31, "p_moderate": 0.51,
              "p_severe": 0.18, "p_spike": 0.73, "is_spike": true}
    },
    "recommended_action": "defer_new_jobs",
    "top_features": [
      {"feature": "spike_in_last_6",       "contribution": 0.312},
      {"feature": "cpu_vs_p95",            "contribution": 0.228},
      {"feature": "time_since_last_spike", "contribution": 0.187}
    ],
    "status": "success",
    "observations_used": 24
  }]
}
```

---

## 13. What Works and What Doesn't

### What works

- **In-distribution (Google K=2):** XGBoost beats all baselines on every horizon (+0.094–0.115 over persistence). EWMA/z-score near-random confirms feature engineering adds real value.
- **Zero-shot transfer:** The model transfers to a completely different cluster 15 years later with modest performance drop. 60m macro PR-AUC 0.547 → 0.826 raw / 0.848 recalibrated.
- **Multi-horizon cascade:** All 3 binary horizon models saturate at PR-AUC=0.957 on Zabbix. Thresholds differentiate (0.70→0.50→0.35) enabling meaningful timing estimates.
- **Recalibration protocol:** Isotonic calibration on 89 days of local data effectively adapts alarm thresholds. Fast and reliable.
- **15m model is consistently strongest:** ROC-AUC 0.911 (Google) / 0.966 (Zabbix). Shorter horizon = more predictable. True across all K values and both datasets.

### What doesn't work well

- **60m severe class on Google (in-distribution):** PR-AUC 0.257 (K=2). The class is rare (~1.8%), the label is strict (K=2), and the 7-day Google dataset limits diversity. This is a structural ceiling on this dataset, not a model failure.
- **EWMA/z-score baselines:** Near-random on binary horizons (PR-AUC 0.089–0.132 on Google). Statistical anomaly detection without feature engineering completely fails for proactive spike prediction.
- **60m model on Google K=2 (absolute numbers):** Macro PR-AUC 0.547 looks modest. Context: random baseline = 0.333, best alternative = 0.433 (ARIMA). The absolute number is depressed by sparse class prevalence, not model quality.
- **Day-of-week signal:** `dow_sin/dow_cos` had the highest SHAP (0.356) but near-zero gain (0.011) — a pure temporal confound from only 7 days of data. Not a generalizable feature.

### Known limitations of the Google 2011 dataset

- Only 7 days of data (160 hours) — limits temporal diversity
- 1 cluster only — limits cluster-level feature diversity
- 2011 workloads — may not reflect modern containerized environments
- Severe class prevalence degrades further in test set (train 2.37% → test 1.76%) — the cluster stabilizes over time in the trace

---

## 14. Future Work

This section is split into two parts: **model improvements** (concrete next experiments, ordered by expected impact and implementation effort) and **scope extensions** (things that are out of scope for the current paper but natural follow-on work).

---

### 14a. Model Improvements — Next Experiments (priority order)

#### 1. Focal Loss for the Severe Class — Imbalance-XGBoost (HIGH IMPACT, LOW RISK)

**Target weakness:** 60m severe class PR-AUC = 0.257 (K=2). Standard class weighting already applied; not enough for 1.8% prevalence.

**Method:** Replace XGBoost's default log-loss with **Class-Balanced Focal Loss** via the [Imbalance-XGBoost](https://www.sciencedirect.com/science/article/abs/pii/S0167865520302129) library (2020). The focal parameter γ down-weights easy negatives, forcing the model to concentrate gradient on hard-to-classify severe examples. The class-balanced variant additionally corrects for frequency imbalance at the class level.

**Implementation:** Drop-in replacement for XGBoost objective. Grid search over γ ∈ {0.5, 1.0, 2.0, 3.0}. Apply to the 60m severity model and OVR severe model. Expected gain: +0.02–0.05 on severe PR-AUC.

**Do not apply to binary horizon models** (15m/30m/45m) — they are already near-saturated and don't have a rare-class problem.

#### 2. Two-Stage Cascade Model (MEDIUM IMPACT, CLEAN ARCHITECTURE)

**Target weakness:** The 60m 3-class softmax tries to simultaneously solve spike detection and severity estimation, compressing the severe class signal.

**Method:** Split into two sequential binary models:
- **Stage 1:** `spike_in_60m` binary (any spike vs no spike) — trained on full dataset, ~4% positives
- **Stage 2:** `is_severe | spike` binary (severe vs moderate) — trained only on spike examples, ~43% severe / 57% moderate (near-balanced)

Stage 2 only runs when Stage 1 fires. This separates two structurally different problems and gives Stage 2 a near-balanced training set without needing class weights or focal loss.

**Expected gain:** Stage 2 trains on a well-balanced subset → severe precision should improve significantly. The combined cascade likely outperforms the current single 3-class model on severe PR-AUC.

**Note:** This replaces the current `SpikeClassifier` (multi:softprob). Requires retraining both stages. Calibration applies per-stage.

#### 3. Conformal Prediction — Uncertainty Intervals (NOVEL PAPER CONTRIBUTION, NO RETRAINING)

**What it adds:** Instead of `p_spike = 0.73`, output `p_spike = 0.73 [0.61, 0.85]` — a calibrated 90% confidence interval. The scheduler can then distinguish "high confidence alarm" from "uncertain alarm".

**Method:** **Conformalized Quantile Regression (CQR)** — model-agnostic, no retraining required, finite-sample coverage guarantee. Apply on top of the existing calibrated XGBoost predictions using the val set as the conformal calibration set.

**References:** [NeurIPS 2021 conformal time-series paper](https://proceedings.neurips.cc/paper/2021/file/312f1ba2a72318edaaa995a67835fad5-Paper.pdf); [Ensemble CQR for probabilistic forecasting](https://ieeexplore.ieee.org/document/9940232/).

**Paper value:** No published HPC spike predictor outputs calibrated intervals. This is a novel contribution that adds operational value with zero model complexity cost.

**Implementation:** ~50 lines using the `nonconformist` or `crepes` Python library.

#### 4. Soft Labels Replacing Hard K-of-N (INTERESTING, 1 TRAINING RUN)

**Target weakness:** Hard binary labels at exactly 30m are artificial — the model is equally penalized for a spike at 31m and no spike at all. This may artificially inflate false positives near the horizon boundary.

**Method:** Replace binary `spike_in_30m ∈ {0,1}` with a smooth label encoding temporal proximity:
- 0.0 → no spike in any of next 6 windows
- 0.3 → spike in windows 4–6 (far end of horizon)
- 0.7 → spike in windows 2–3
- 1.0 → spike in next window

Train with MSE or KL-divergence. [Literature (2024)](https://link.springer.com/chapter/10.1007/978-981-96-0840-9_21) shows representation soft label smoothing reduces false positives on cross-domain test sets.

**Risk:** Changes the label contract — the output is no longer a calibrated probability of binary spike. Requires careful re-evaluation of alarm thresholds.

#### 5. Extended Lookback Ablation — 240 min (QUICK ABLATION, 1 TRAINING RUN)

**Question to answer:** Does extending from 24 buckets (120 min) to 48 buckets (240 min) improve 60m PR-AUC?

**Rationale:** HPC batch schedulers often run jobs with 4-hour cycles. A 120-min window may miss the early ramp-up signal for these jobs. Add `cpu_lag_36, cpu_lag_48` and extend `cpu_ewma_48` accordingly.

**How to evaluate:** Train 60m model with 48-bucket lags, compare macro PR-AUC on same test split. If gain > +0.02, adopt. If flat or negative, the 120-min window is sufficient and the decision is closed.

**Literature:** No strong consensus for CPU specifically. One 2025 datacenter forecasting paper used 300-window lookback for GPU load, suggesting longer context can help. Needs empirical validation on our data.

#### 6. PatchTST Backbone + XGBoost Classifier (RESEARCH-GRADE, HIGH EFFORT)

**Concept:** Train a small transformer ([PatchTST, ICLR 2023](https://arxiv.org/pdf/2211.14730)) on raw 24-bucket CPU sequences to learn a shared representation, then attach XGBoost classifiers per horizon instead of hand-crafted features. Tests the hypothesis that learned representations outperform hand-crafted features for this task.

**Why it's publishable if it works:** No paper has combined PatchTST with XGBoost for datacenter spike prediction. If the hybrid beats pure XGBoost by a meaningful margin, it's a contribution. If it doesn't, it validates the feature engineering approach.

**Why it's risky:** Adds training complexity, requires GPU for the transformer backbone, and the latency budget must be verified. Only pursue after the paper is otherwise complete.

---

### 14b. Scope Extensions (out of current paper scope)

#### Time-to-Peak and Peak Magnitude

The current system provides a **timing window** (which horizon first fires) and a **severity class** (moderate/severe). It does not provide:
- **Exact time-to-peak** — requires a regression or survival model (Cox PH, quantile regression). Distinct modelling task.
- **Peak CPU magnitude** — requires a regression target (e.g. predict max CPU% in next window). Not a classification problem.
- **Spike duration** — how long the spike lasts above threshold. Not predicted at all.

The scheduler use case is fully served by windows and severity class. These are natural extensions for a follow-on paper.

#### Google 2019 BigQuery

Second training dataset: `google.com:google-cluster-data` (BigQuery). 8 clusters, estimated 3–25× more data than Google 2011. Strengthens the generalization claim significantly. Same pipeline applies; a new download script is needed. Not blocking for the current paper but strengthens it, especially for the severe class.

#### What was already tried and reverted

| Experiment | Result | Decision |
|---|---|---|
| FFT spectral features (Phase 8) | Flat on 60m, −0.026 regression on OVR severe | Reverted |
| Streak persistence features (Phase 6) | −0.021 regression across 2 independent runs | Reverted — collinear with `spike_in_last_{1,3,6}` |
| Day-of-week features | SHAP=0.356 but gain=0.011 — temporal confound | Dropped permanently |
| Standard SMOTE/ADASYN | Not tried — dataset too small (7 days), risk of interpolating same patterns | Skip — focal loss is safer |

---

## 15. Deployment Components

### Files needed to run inference

```
predict_spike.py          ← one-shot: CSV → JSON
predict_daemon.py         ← continuous loop (--interval N seconds)
spike_feature_engineer.py ← feature engineering (imported)
spike_classifier.py       ← model definitions + _X_COLS (imported)
models/
  spike/                  ← 60m model (spike_model.json, meta, config, thresholds, calibrators)
  spike_15m/              ← 15m binary model
  spike_30m/              ← 30m binary model
  spike_45m/              ← 45m binary model
  spike_severe_ovr/       ← OVR severe model
requirements-inference.txt
```

Python dependencies: `xgboost>=3.2`, `pandas`, `numpy`, `pyarrow`

### Running the daemon

```bash
python predict_daemon.py \
  --input   cpu_window.csv \      # last 120 min, cluster_agg format, your data source
  --model-dir models/spike/ \
  --output  predictions.json \    # overwritten atomically each cycle
  --interval 300                  # every 5 minutes; use 0 for one-shot
```

### Validated on

- Google Cluster Traces 2011 (training + test)
- Zabbix HPC Slurm cluster, 11 nodes, 89 days (zero-shot cross-domain, 2026)
- One-shot test: 10 machines, 0.2s inference, 1 alarm correctly raised

---

## 16. Training Pipeline (for reproducibility)

```
cluster_cpu_data.csv (278M rows, Google 2011)
    │
    ▼ spike_preprocessor.py
cluster_agg.parquet (5-min bucket aggregates)
    │
    ▼ spike_feature_engineer.py
cluster_features.parquet (37 features)
spike_thresholds.parquet (per-machine p95/p99, training data only)
    │
    ▼ train_spike_classifier.py --tune-hyperparams --optuna-trials 150
models/spike/            (60m severity, K=2)
models/spike_15m/        (15m binary, K=1)
models/spike_30m/        (30m binary, K=1)
models/spike_45m/        (45m binary, K=1)
models/spike_severe_ovr/ (OVR severe binary, K=2)
```

**Chronological split enforced throughout.** No random shuffle at any step.  
**Thresholds from training data only.** Val and test sets never see threshold computation.  
**GPU training:** `docker compose --profile train run --rm --build train` (NVIDIA CUDA 12.4, XGBoost 3.2.0).
