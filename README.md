# CPU Spike Predictor

Predicts whether a CPU spike will occur on a datacenter machine in the next **60 minutes** — and how severe it will be — so a workload scheduler can act **before** the spike happens.

## What it does

Every 5 minutes, the system reads the last 120 minutes of per-machine CPU telemetry (24 buckets), runs inference across all machines in the cluster, and outputs a severity prediction per machine: **no spike / moderate / severe**. A second model adds a 15-minute imminence signal for urgent pre-emptions.

The output feeds a workload scheduler that can defer batch jobs, pre-empt running tasks, or simply monitor — before the spike materialises.

**Dataset used for training:** [Google Cluster Traces 2011](https://github.com/google/cluster-data) — 278M rows, 12,555 machines, 160 hours.

---

## Results

| Model | CV PR-AUC | Test PR-AUC | Test ROC-AUC |
|-------|-----------|-------------|--------------|
| 60m severity (3-class) | 0.559 ± 0.008 | **0.574** (calibrated) | 0.839 |
| 15m binary | 0.584 ± 0.008 | 0.575 | 0.911 |
| OVR severe (binary) | — | 0.362 | — |

At the default alarm threshold (0.55): Precision 0.385 · Recall 0.444 · ~18 alarms/day across 12,555 machines.

---

## Understanding the metrics

These metrics measure how well the model performs. Here is what each one means in plain terms.

### Precision and Recall — the core trade-off

These two are in permanent tension: improving one typically worsens the other.

**Precision** answers: *"Of all the alarms we fired, how many were real spikes?"*  
- Precision = 0.385 means 38.5% of our alarms turn out to be real spikes. The other 61.5% are false alarms — the scheduler acts pre-emptively but no spike was coming.  
- High precision = fewer false alarms. Low precision = the scheduler cries wolf too often.

**Recall** answers: *"Of all actual spikes that happened, how many did we catch?"*  
- Recall = 0.444 means we caught 44.4% of all real spikes before they happened. We missed 55.6%.  
- High recall = fewer missed spikes. Low recall = many spikes happen without warning.

**The alarm threshold (default 0.55)** is the knob that controls this trade-off. Raising it fires fewer but more reliable alarms (higher precision, lower recall). Lowering it catches more spikes but with more false alarms (lower precision, higher recall). The Streamlit dashboard has an interactive slider to explore this.

**False positives (false alarms)** are cases where the model predicted a spike but none came. These cause unnecessary scheduler actions (e.g., deferring a batch job that would have run fine). The alarm debounce — requiring 2 consecutive spike predictions before firing — reduces these.

---

### PR-AUC — the summary metric

**PR-AUC** (Area Under the Precision-Recall Curve) summarises the entire precision/recall trade-off across all possible alarm thresholds into a single number from 0 to 1.

- **Random baseline** ≈ the fraction of samples that are actually spikes (class prevalence). For our 3-class model, the macro average random baseline is ~0.25.  
- **Our 60m model: 0.574** — more than 2× above random. A perfect model would score 1.0.  
- **Why we use this, not accuracy**: Only ~15% of 5-minute windows contain a spike. A model that always predicts "no spike" would be 85% accurate but completely useless — PR-AUC correctly scores it near the random baseline.  
- **CV PR-AUC** is computed with walk-forward cross-validation (training on past, validating on future) — a more honest estimate of real-world performance than a single train/test split. The ± value is the standard deviation across 5 folds.  
- **Macro PR-AUC** averages the per-class scores equally — the rare "severe" class counts the same as the common "no spike" class. This prevents the model from ignoring rare but important events.

---

### ROC-AUC — the rank-ordering metric

**ROC-AUC** (Area Under the Receiver Operating Characteristic curve) measures whether the model correctly ranks high-risk machines above low-risk ones, regardless of any specific threshold.

- A score of **1.0** = perfect ranking. **0.5** = random. Our 60m model scores **0.839**.  
- Intuition: pick any two machines at random — one that will spike and one that won't. ROC-AUC 0.839 means there is an 83.9% chance the model assigns a higher probability to the one that will actually spike.  
- **Why ROC-AUC matters for cross-domain comparisons**: unlike PR-AUC, the ROC-AUC baseline is always 0.5 regardless of class imbalance. This makes it the right metric when comparing results across different datasets (e.g., when evaluating the model on a new production cluster where the spike rate might be different from the training data).

---

### Calibration — do the probabilities mean what they say?

The model outputs probabilities (e.g., "70% chance of a moderate spike in 60 minutes"). **Calibration** checks whether those probabilities are trustworthy.

- A well-calibrated model: when it says 70%, spikes happen ~70% of the time.  
- We apply **isotonic regression calibration** on the validation set. After calibration, the Brier score (a calibration quality measure) halved for all classes, and the Expected Calibration Error (ECE) dropped below 0.025.  
- Calibrated probabilities are what the alarm threshold is applied to. Using an uncalibrated threshold on calibrated probabilities (or vice versa) produces systematically wrong alarm rates.

---

## Architecture

```
cluster_cpu_data.csv
  → [Step 1] Preprocessor      → cluster_agg.parquet        (5-min bucket aggregates)
  → [Step 2] Feature Engineer  → cluster_features.parquet   (37 features per machine-bucket)
  → [Step 3] Trainer           → models/spike/              (60m severity)
                                  models/spike_15m/          (15m binary)
                                  models/spike_severe_ovr/   (OVR severe)

Inference
  → predict_spike.py           REST API payload per machine
  → spike_api.py               FastAPI service  (POST /predict)
  → demo.py                    Streamlit dashboard (model metrics + threshold explorer)
```

Three XGBoost models, chronological train/val/test split (60/20/20%), walk-forward CV, isotonic probability calibration on the 60m model.

---

## Requirements

- Python 3.11+
- For GPU training: NVIDIA GPU + NVIDIA Container Toolkit (handled by Docker)
- For local testing: CPU is fine (training will just be slow)

---

## Step-by-step: from zero to dashboard

### 1. Clone and set up the environment

```bash
git clone https://github.com/DiogoZeca/Agentic_AI.git
cd Agentic_AI/AIModel

python3 -m venv .venv
.venv/bin/pip install -r requirements-train.txt
```

### 2. Download the dataset

```bash
.venv/bin/python data/download_cluster_data.py
# Downloads to AIModel/data/cluster_cpu_data.csv (~8 GB)
```

### 3. Train the model

**Local (CPU — slow, for testing only):**

```bash
.venv/bin/python train_spike_classifier.py \
  --data-path data/cluster_cpu_data.csv \
  --artifacts-dir data/full_run \
  --tune-hyperparams --optuna-trials 30 \
  --n-estimators 4000
```

**GPU VM via Docker (recommended — uses NVIDIA GPU):**

```bash
# Run from the project root (Agentic_AI/)
docker compose --profile train run --rm --build train
```

Training runs in multiple steps. You can resume from any step with `--from-step N` if it was interrupted.

### 4. Monitor training progress

```bash
tail -f data/run_log.txt
```

### 5. Run the test suite

```bash
.venv/bin/python -m pytest tests/ -v
# Should pass ~290 tests in ~30 seconds
```

### 6. (Optional) Run a single inference

```bash
.venv/bin/python predict_spike.py \
  --input data/window.csv \
  --model-dir data/full_run/models/spike/
```

### 7. (Optional) Start the REST API

```bash
.venv/bin/pip install -r requirements-inference.txt
.venv/bin/python -m uvicorn spike_api:app --host 0.0.0.0 --port 8000
# POST /predict  →  per-machine severity predictions
```

### 8. Launch the Streamlit dashboard

```bash
.venv/bin/pip install -r requirements-demo.txt
.venv/bin/python -m streamlit run demo.py
# Opens at http://localhost:8501
```

The dashboard reads training artifacts from `data/full_run/` — no live inference needed. It shows model metrics, a threshold trade-off explorer (precision vs recall vs alarms/day), calibration quality, and feature attribution.

---

## Project structure

```
Agentic_AI/
├── CLAUDE.md                   Developer notes (AI assistant instructions)
├── DEVELOPMENT.md              Full decision log and phase history
├── docker-compose.yml          API + GPU training services
├── scripts/
│   └── vm-setup.sh             One-shot GPU VM provisioning
└── AIModel/
    ├── .venv/                  Python virtual environment
    ├── spike_preprocessor.py   Step 1: raw CSV → 5-min bucket aggregates
    ├── spike_feature_engineer.py  Step 2: aggregates → 37-feature dataset
    ├── spike_classifier.py     XGBoost model wrappers
    ├── train_spike_classifier.py  Step 3: full training pipeline
    ├── predict_spike.py        Batch inference (per-machine severity)
    ├── spike_api.py            FastAPI inference service
    ├── demo.py                 Streamlit model dashboard
    ├── requirements-train.txt
    ├── requirements-inference.txt
    ├── requirements-demo.txt
    ├── data/
    │   ├── cluster_cpu_data.csv   Raw Google Cluster Traces 2011
    │   └── full_run/              Training artifacts (models, caches, configs)
    └── tests/                  Pytest test suite (~290 tests)
```
