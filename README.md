# CPU Spike Predictor

Predicts whether a CPU spike will occur on a datacenter machine in the next **15 to 60 minutes** — and how severe it will be — so a workload scheduler can act **before** the spike happens.

## What it does

Every N minutes, the system reads the last 120 minutes of per-machine CPU telemetry (24 × 5-min buckets), runs inference across all machines, and outputs per-machine predictions across four horizons: **15m / 30m / 45m / 60m**. A fifth model adds a dedicated severe-spike signal. Together they form a timing cascade — the earliest horizon that alarms determines how urgently the scheduler should act.

The output feeds a workload scheduler that can defer batch jobs, migrate running tasks, or trigger emergency pre-emption — before the spike materialises.

**The system is source-agnostic:** it accepts any CSV in `cluster_agg` format from any monitoring system (Zabbix, Prometheus, LDMS, etc.).

**Training dataset:** [Google Cluster Traces 2011](https://github.com/google/cluster-data) — 278M rows, 12,555 machines, 160 hours.  
**Cross-domain validation:** Real production Slurm HPC cluster (11 nodes, 89 days, 2026).

---

## Results

### Google Cluster 2011 — Test Set (in-distribution)

Production baseline: K=2 label definition, 37 features, 150-trial Optuna, XGBoost 3.2.0.

| Model | CV PR-AUC | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|
| 60m severity (3-class) | 0.4995 ± 0.005 | **0.547** (calibrated) | 0.843 |
| 15m binary | 0.582 ± 0.008 | **0.575** | 0.911 |
| 30m binary | — | **0.563** | — |
| 45m binary | — | **0.562** | — |
| OVR severe (binary) | 0.275 ± 0.020 | **0.339** | 0.863 |

60m alarm at threshold 0.25: Precision 0.446 · Recall 0.306 · **7.4 alarms/day**.

### Baseline Comparison — Google Test Set

XGBoost vs the best rule-based and statistical baselines (evaluated on same raw CPU time series, no engineered features):

| Baseline | 15m PR-AUC | 30m PR-AUC | 45m PR-AUC | 60m PR-AUC | OVR PR-AUC |
|---|---|---|---|---|---|
| Random | 0.052 | 0.078 | 0.099 | 0.333 | 0.018 |
| Persistence | 0.481 | 0.455 | 0.447 | 0.433 | 0.229 |
| ARIMA(2,1,0) | 0.466 | 0.442 | 0.435 | 0.430 | 0.221 |
| EWMA z-score | 0.089 | 0.108 | 0.126 | 0.335 | 0.024 |
| **XGBoost (ours)** | **0.575** | **0.563** | **0.562** | **0.547** | **0.339** |

XGBoost beats the best baseline (persistence) by **+0.094 to +0.115** across all horizons.

### Cross-Domain Transfer — Zabbix HPC (zero-shot)

Model trained on Google 2011, evaluated zero-shot on a 2026 production HPC cluster. No retraining.

| Model | Zabbix PR-AUC | Zabbix ROC-AUC | Alarm F1 | Threshold |
|---|---|---|---|---|
| 60m severity (recalibrated) | **0.848** | **0.922** | 0.711 | 0.40 |
| 15m binary | **0.957** | **0.966** | 0.832 | 0.70 |
| 30m binary | **0.957** | **0.960** | 0.861 | 0.50 |
| 45m binary | **0.957** | **0.955** | 0.879 | 0.35 |
| OVR severe | **0.926** | **0.955** | 0.771 | 0.65 |

23 of 37 features show MAJOR distribution shift (PSI > 0.25) between the two datasets. The model transfers because spike-history and machine-relative features (moderate PSI 0.12–0.15) dominate predictions. See `project_state.md` for full PSI analysis.

---

## Understanding the metrics

### Precision and Recall — the core trade-off

**Precision** answers: *"Of all the alarms we fired, how many were real spikes?"*
- Precision = 0.45 means 45% of alarms turn out to be real spikes. The other 55% are false alarms.
- High precision = fewer false alarms.

**Recall** answers: *"Of all actual spikes that happened, how many did we catch?"*
- Recall = 0.31 means we caught 31% of all real spikes before they happened.
- High recall = fewer missed spikes.

The **alarm threshold** controls this trade-off. Raising it fires fewer but more reliable alarms (higher precision, lower recall). The full sweep is in `spike_config.json`.

---

### PR-AUC — the summary metric

**PR-AUC** summarises the precision/recall trade-off across all thresholds into a single number from 0 to 1.

- **Random baseline** = class prevalence (e.g. 0.052 for 5% positives on 15m; 0.333 for 3-class uniform random on 60m)
- **Our 15m model: 0.575** — more than 10× above random (0.052)
- **Why not accuracy:** Only ~5% of 5-min windows contain a spike. A model predicting "no spike" always would be 95% accurate but PR-AUC ≈ 0.052 — correctly useless.
- **Macro PR-AUC** averages per-class scores equally — the rare severe class counts the same as the common no_spike class.
- **PR-AUC is prevalence-dependent:** a dataset with 40% positives trivially scores higher than one with 5% positives. Use ROC-AUC when comparing across datasets with different spike rates.

---

### ROC-AUC — the rank-ordering metric

**ROC-AUC** measures whether the model ranks high-risk machines above low-risk ones, regardless of threshold.

- **0.5** = random · **1.0** = perfect · Our 15m model: **0.911**
- Intuition: pick any two machines — one that will spike, one that won't. ROC-AUC 0.911 means there is a 91.1% chance the model assigns higher probability to the one that will spike.
- **ROC-AUC is prevalence-independent** — the correct metric for cross-dataset comparisons (e.g. Zabbix vs Google).

---

### Calibration — do the probabilities mean what they say?

The model outputs probabilities (e.g., "70% chance of moderate spike in 60 minutes"). Calibration checks whether those numbers are trustworthy.

We apply **isotonic regression calibration** on the validation set for the 60m model. After calibration, the Brier score halved for all classes and ECE dropped below 0.025. The alarm threshold is always applied to calibrated probabilities.

---

## Architecture

```
cluster_cpu_data.csv  (training, Google 2011)
    │
    ▼  spike_preprocessor.py
cluster_agg.parquet   (5-min bucket aggregates)
    │
    ▼  spike_feature_engineer.py
cluster_features.parquet   (37 features per machine-bucket)
spike_thresholds.parquet   (per-machine p95 + p99, training data only)
    │
    ▼  train_spike_classifier.py  (Optuna + XGBoost, GPU)
models/spike/             (60m severity, 3-class)
models/spike_15m/         (15m binary)
models/spike_30m/         (30m binary)
models/spike_45m/         (45m binary)
models/spike_severe_ovr/  (OVR severe binary)

Inference
    predict_spike.py      one-shot: CSV → JSON
    predict_daemon.py     continuous loop (--interval N seconds)
    spike_api.py          FastAPI service (POST /predict)
```

Five XGBoost models, chronological train/val/test split (60/20/20%), walk-forward CV, isotonic calibration on the 60m model. Each pipeline step caches its output — resume from any step with `--from-step N`.

---

## Requirements

- Python 3.11+
- For GPU training: NVIDIA GPU + NVIDIA Container Toolkit (handled by Docker)
- For inference only: CPU is sufficient

---

## Step-by-step: from zero to running predictions

### 1. Clone and set up the environment

```bash
git clone https://github.com/DiogoZeca/Agentic_AI.git
cd Agentic_AI/AIModel

python3 -m venv .venv
.venv/bin/pip install -r requirements-train.txt
```

### 2. Download the dataset

```bash
.venv/bin/python3 data/download_cluster_data.py
# Downloads to AIModel/data/cluster_cpu_data.csv (~8 GB)
```

### 3. Train the models

**Local (CPU — slow, for testing only):**

```bash
.venv/bin/python3 train_spike_classifier.py \
  --data-path data/cluster_cpu_data.csv \
  --artifacts-dir data/full_run \
  --tune-hyperparams --optuna-trials 30 \
  --n-estimators 4000
```

**GPU VM via Docker (recommended for full training):**

```bash
# Run from the project root (Agentic_AI/)
docker compose --profile train run --rm --build train
# Uses OPTUNA_TRIALS=150 and DEVICE=cuda automatically
```

### 4. Run the test suite

```bash
.venv/bin/python3 -m pytest tests/ -v
```

### 5. Run one-shot inference

```bash
# Provide last 120 min of CPU telemetry as cluster_agg CSV
.venv/bin/python3 predict_spike.py \
  --input cpu_window.csv \
  --model-dir data/full_run/models/spike/ \
  --output predictions.json
```

### 6. Run as a continuous daemon

```bash
# Runs every 5 minutes, reads --input, writes --output atomically
.venv/bin/python3 predict_daemon.py \
  --input   cpu_window.csv \
  --model-dir data/full_run/models/spike/ \
  --output  predictions.json \
  --interval 300
```

The daemon is **source-agnostic**: point `--input` at any CSV your monitoring system produces in `cluster_agg` format.

### 7. Start the REST API

```bash
.venv/bin/pip install -r requirements-inference.txt
.venv/bin/python3 -m uvicorn spike_api:app --host 0.0.0.0 --port 8000
# POST /predict  →  per-machine severity predictions
```

---

## Project structure

```
Agentic_AI/
├── project_state.md            Full project documentation (paper reference)
├── CLAUDE.md                   Developer notes (AI assistant instructions)
├── docker-compose.yml          API + GPU training services
├── scripts/
│   └── vm-setup.sh             One-shot GPU VM provisioning
└── AIModel/
    ├── spike_preprocessor.py      Step 1: raw CSV → 5-min bucket aggregates
    ├── spike_feature_engineer.py  Step 2: aggregates → 37-feature dataset
    ├── spike_classifier.py        XGBoost model wrappers
    ├── train_spike_classifier.py  Step 3: full training pipeline (Optuna + CV)
    ├── predict_spike.py           One-shot inference (CSV → JSON)
    ├── predict_daemon.py          Continuous daemon (--interval N seconds)
    ├── spike_api.py               FastAPI inference service
    ├── evaluate_zabbix.py         Cross-domain evaluation script
    ├── baselines/
    │   └── evaluate_baselines.py  Baseline comparison suite (6 methods)
    ├── requirements-train.txt
    ├── requirements-inference.txt
    ├── data/
    │   ├── cluster_cpu_data.csv      Raw Google Cluster Traces 2011 (~8 GB)
    │   ├── full_run/                 Production model artifacts
    │   ├── experiments/              Ablation runs (k1/, k3/, phase8_fft/, …)
    │   └── baselines/                Baseline evaluation results
    └── tests/                        Pytest test suite
```
