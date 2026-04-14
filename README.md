# CPU Spike Predictor

Predicts whether a CPU spike will occur on a datacenter machine in the next **60 minutes** — and how severe it will be — so a workload scheduler can act **before** the spike happens.

## What it does

Every 5 minutes, the system reads the last 120 minutes of per-machine CPU telemetry (24 buckets), runs inference across all machines in the cluster, and outputs a severity prediction per machine: **no spike / moderate / severe**. A second model adds a 15-minute imminence signal for urgent pre-emptions.

The output feeds a workload scheduler that can defer batch jobs, pre-empt running tasks, or simply monitor — before the spike materialises.

**Dataset used for training:** [Google Cluster Traces 2011](https://github.com/google/cluster-data) — 278M rows, 12,555 machines, 160 hours.

---

## Results

Results below are the Phase 5 reference baseline (37 features). A Phase 6 run with streak persistence features (39 features) is in progress.

| Model | CV PR-AUC | Test PR-AUC | Test ROC-AUC |
|-------|-----------|-------------|--------------|
| 60m severity (3-class) | 0.559 ± 0.008 | **0.574** (calibrated) | 0.839 |
| 15m binary | 0.584 ± 0.008 | 0.575 | 0.911 |
| OVR severe (binary) | — | 0.362 | 0.858 |

Per-class PR-AUC on test (60m model): `no_spike` = 0.970 · `moderate` = 0.367 · `severe` = 0.330.

At the default alarm threshold (0.25): Precision 0.40 · Recall 0.30.

---

## Understanding the metrics

These metrics measure how well the model performs. Here is what each one means in plain terms.

### Precision and Recall — the core trade-off

These two are in permanent tension: improving one typically worsens the other.

**Precision** answers: *"Of all the alarms we fired, how many were real spikes?"*
- Precision = 0.40 means 40% of our alarms turn out to be real spikes. The other 60% are false alarms — the scheduler acts pre-emptively but no spike was coming.
- High precision = fewer false alarms. Low precision = the scheduler cries wolf too often.

**Recall** answers: *"Of all actual spikes that happened, how many did we catch?"*
- Recall = 0.30 means we caught 30% of all real spikes before they happened. We missed 70%.
- High recall = fewer missed spikes. Low recall = many spikes happen without warning.

**The alarm threshold** is the knob that controls this trade-off. Raising it fires fewer but more reliable alarms (higher precision, lower recall). Lowering it catches more spikes but with more false alarms. The threshold sweep table in `spike_config.json` shows Precision / Recall / alarms-per-day across all thresholds.

---

### PR-AUC — the summary metric

**PR-AUC** (Area Under the Precision-Recall Curve) summarises the entire precision/recall trade-off across all possible alarm thresholds into a single number from 0 to 1.

- **Random baseline** ≈ the fraction of samples that are actually spikes (class prevalence). For our 3-class model, the macro average random baseline is roughly 0.25.
- **Our 60m model: 0.574** — more than 2× above random. A perfect model would score 1.0.
- **Why we use this, not accuracy**: Only ~15% of 5-minute windows contain a spike. A model that always predicts "no spike" would be 85% accurate but completely useless — PR-AUC correctly scores it near the random baseline.
- **CV PR-AUC** is computed with walk-forward cross-validation (training on past, validating on future) — a more honest estimate of real-world performance than a single train/test split. The ± value is the standard deviation across 5 folds.
- **Macro PR-AUC** averages the per-class scores equally — the rare "severe" class counts the same as the common "no spike" class. This prevents the model from ignoring rare but important events.

---

### ROC-AUC — the rank-ordering metric

**ROC-AUC** (Area Under the Receiver Operating Characteristic curve) measures whether the model correctly ranks high-risk machines above low-risk ones, regardless of any specific threshold.

- A score of **1.0** = perfect ranking. **0.5** = random. Our 60m model scores **0.839**.
- Intuition: pick any two machines at random — one that will spike and one that won't. ROC-AUC 0.839 means there is an 83.9% chance the model assigns a higher probability to the one that will actually spike.
- **Why ROC-AUC matters for cross-domain comparisons**: unlike PR-AUC, the ROC-AUC baseline is always 0.5 regardless of class imbalance. This makes it the right metric when comparing results across different datasets (e.g., evaluating on a new production cluster where the spike rate might differ from the training data).

---

### Calibration — do the probabilities mean what they say?

The model outputs probabilities (e.g., "70% chance of a moderate spike in 60 minutes"). **Calibration** checks whether those probabilities are trustworthy.

- A well-calibrated model: when it says 70%, spikes happen ~70% of the time.
- We apply **isotonic regression calibration** on the validation set. After calibration, the Brier score (a calibration quality measure) halved for all classes, and the Expected Calibration Error (ECE) dropped below 0.025.
- The alarm threshold is always applied to calibrated probabilities. Using an uncalibrated threshold on calibrated probabilities (or vice versa) produces systematically wrong alarm rates.

---

## Architecture

```
cluster_cpu_data.csv
  → [Step 1] Preprocessor      → cluster_agg.parquet        (5-min bucket aggregates)
  → [Step 2] Feature Engineer  → cluster_features.parquet   (39 features per machine-bucket)
  → [Step 3] Trainer           → models/spike/              (60m severity, 3-class)
                                  models/spike_15m/          (15m binary)
                                  models/spike_severe_ovr/   (OVR severe binary)

Inference
  → predict_spike.py           Batch inference per machine
  → spike_api.py               FastAPI service  (POST /predict)
```

Three XGBoost models, chronological train/val/test split (60/20/20%), walk-forward CV, isotonic probability calibration on the 60m model. Each step caches its output — resume from any step with `--from-step N`.

---

## Requirements

- Python 3.11+
- For GPU training: NVIDIA GPU + NVIDIA Container Toolkit (handled by Docker)
- For local testing: CPU is fine (training will just be slow)

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

### 3. Train the model

**Local (CPU — slow, for testing only):**

```bash
.venv/bin/python3 train_spike_classifier.py \
  --data-path data/cluster_cpu_data.csv \
  --artifacts-dir data/full_run \
  --tune-hyperparams --optuna-trials 30 \
  --n-estimators 4000
```

**GPU VM via Docker (recommended):**

```bash
# Run from the project root (Agentic_AI/)
docker compose --profile train run --rm --build train
# Uses OPTUNA_TRIALS=150 and DEVICE=cuda automatically
```

Training runs in three steps. Resume from any step with `--from-step N` if interrupted.

### 4. Monitor training

```bash
tail -f data/run_log.txt
```

### 5. Run the test suite

```bash
.venv/bin/python3 -m pytest tests/ -v
# Should pass ~258+ tests in ~30 seconds (round-trip tests require trained model artifacts)
```

### 6. Run inference

```bash
.venv/bin/python3 predict_spike.py \
  --input data/window.csv \
  --model-dir data/full_run/models/spike/
```

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
├── CLAUDE.md                   Developer notes (AI assistant instructions)
├── DEVELOPMENT.md              Full decision log and phase history
├── session_state.md            Current work anchor: what is running, next steps
├── docker-compose.yml          API + GPU training services
├── scripts/
│   └── vm-setup.sh             One-shot GPU VM provisioning (Docker + NVIDIA Container Toolkit)
└── AIModel/
    ├── spike_preprocessor.py      Step 1: raw CSV → 5-min bucket aggregates
    ├── spike_feature_engineer.py  Step 2: aggregates → 39-feature dataset
    ├── spike_classifier.py        XGBoost model wrappers (SpikeClassifier, BinarySpikeClassifier)
    ├── train_spike_classifier.py  Step 3: full training pipeline with Optuna + CV
    ├── predict_spike.py           Batch inference (per-machine severity predictions)
    ├── spike_api.py               FastAPI inference service
    ├── requirements-train.txt
    ├── requirements-inference.txt
    ├── data/
    │   ├── cluster_cpu_data.csv   Raw Google Cluster Traces 2011 (~8 GB)
    │   └── full_run/              Training artifacts (models, caches, configs)
    └── tests/                     Pytest test suite (~258+ tests)
```
