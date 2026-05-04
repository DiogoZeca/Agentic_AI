# CPU Spike Predictor

Predicts whether a CPU spike will occur on a datacenter machine in the next **15 to 60 minutes** — and how severe — so a workload scheduler can act **before** the spike materialises, not after.

Every 5 minutes the system reads the last 120 minutes of per-machine CPU telemetry (24 × 5-min buckets), runs five XGBoost models in parallel, and outputs per-machine predictions across four time horizons. The scheduler uses the earliest horizon that alarms to decide how urgently to defer batch jobs, migrate tasks, or trigger pre-emption.

**The system is monitoring-agnostic:** it accepts any CSV in `cluster_agg` format from any source (Zabbix, Prometheus, LDMS, custom scripts).

---

## Results

**Training:** [Google Cluster Traces 2011](https://github.com/google/cluster-data) — 278M rows, 12,555 machines, 160 hours.  
**Cross-domain validation:** Real production Slurm HPC cluster (Zabbix-monitored, 11 nodes, 89 days), zero-shot — no retraining.

### Google Cluster 2011 — Test Set

| Model | Test PR-AUC | Test ROC-AUC | Notes |
|-------|-------------|--------------|-------|
| 60m severity (3-class) | **0.547** | 0.849 | Isotonic calibration; 7.4 alarms/day at threshold 0.25 |
| 15m binary | **0.575** | 0.912 | |
| 30m binary | **0.563** | 0.878 | |
| 45m binary | **0.563** | 0.860 | |
| OVR severe (cascade Stage 2) | **0.543** | 0.761 | Measured on spike-positive subset |

XGBoost vs. best rule-based baseline (persistence): **+0.094 to +0.115 PR-AUC** across all horizons.

### Zero-Shot Transfer — Zabbix HPC Cluster

25 of 40 features show MAJOR distribution shift (PSI > 0.25) between the two domains. The model transfers because spike-history and machine-relative features dominate predictions and are domain-invariant.

| Model | Zabbix PR-AUC | Zabbix ROC-AUC |
|-------|---------------|----------------|
| 60m severity | **0.848** | 0.922 |
| 15m binary | **0.957** | 0.966 |
| 30m binary | **0.957** | 0.960 |
| 45m binary | **0.957** | 0.955 |
| OVR severe | **0.926** | 0.955 |

---

## Project Structure

```
Agentic_AI/
├── CLAUDE.md                       Developer reference (pipeline architecture, design decisions)
├── pyproject.toml                  Makes spike/ pip-installable
├── docker-compose.yml              API + GPU training services
│
├── docs/
│   ├── ops_runbook.md              Step-by-step guide for new-domain operators
│   ├── technical_report.md         Full experimental results, all ablations, design rationale
│   └── advanced_reference.md       Complete CLI reference, drift monitoring, SLO, API deployment
│
├── spike/                          Inference package (pip install -e .)
│   ├── predict.py                  One-shot inference: CSV → JSON
│   ├── daemon.py                   Continuous polling loop (--interval N seconds)
│   ├── api.py                      FastAPI service (POST /predict)
│   ├── bootstrap_thresholds.py     Domain threshold bootstrap (new-domain deployment)
│   ├── drift_monitor.py            PSI feature drift detection
│   ├── feature_engineer.py         40-feature engineering (shared by train and inference)
│   ├── classifier.py               XGBoost model wrappers + walk-forward CV
│   ├── psi.py                      Population Stability Index utilities
│   ├── version.py                  Model artefact versioning
│   └── Dockerfile                  Inference API image (CPU-only)
│
├── training/
│   ├── train.py                    3-step training pipeline (preprocess → features → train)
│   ├── Dockerfile.training         GPU training image (nvidia/cuda:12.4.1)
│   └── adapters/
│       ├── google_cluster.py       Google Cluster Traces 2011 → cluster_agg
│       └── zabbix.py               Zabbix 7.x API → cluster_agg
│
├── evaluation/
│   ├── evaluate_domain.py          Retrospective backtesting on your own data
│   ├── evaluate_cross_domain.py    3-phase cross-domain evaluation (EDA + PSI + metrics)
│   └── baselines/
│       └── evaluate_baselines.py   Persistence / EWMA / ARIMA baseline comparison
│
├── tests/                          Pytest suite (~430 tests, ~30s)
│
├── demo/
│   └── demo.py                     Streamlit dashboard (Overview, Threshold, Calibration, SHAP)
│
├── scripts/
│   └── vm-setup.sh                 One-shot GPU VM provisioning (Docker + NVIDIA Container Toolkit)
│
└── data/
    ├── cluster_cpu_data.csv        Google Cluster Traces 2011 (278M rows, ~8 GB)
    ├── full_run/                   Production artefacts (K=2 baseline)
    │   ├── cluster_agg.parquet     Step 1 cache
    │   ├── cluster_features.parquet Step 2 cache (stays on training VM)
    │   ├── spike_thresholds.parquet Per-machine p95/p99 thresholds
    │   └── spike/                  60m model weights, config, calibrators
    │       ├── spike_15m/          15m binary model
    │       ├── spike_30m/          30m binary model
    │       ├── spike_45m/          45m binary model
    │       └── spike_severe_ovr/   OVR severe cascade model
    └── experiments/                Ablation runs (k1/, k3/, phase8_fft/, …)
```

---

## Quick Start

```bash
# Set up environment (run once from repo root)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"

# Run the test suite
.venv/bin/python3.12 -m pytest tests/ -v

# One-shot inference (provide last 120 min of CPU telemetry as cluster_agg CSV)
python spike/predict.py \
    --input cpu_window.csv \
    --model-dir data/full_run/spike \
    --output predictions.json

# Continuous daemon (re-runs every 5 minutes)
python spike/daemon.py \
    --input cpu_window.csv \
    --model-dir data/full_run/spike \
    --output predictions.json \
    --interval 300

# GPU training (via Docker Compose on a CUDA VM)
docker compose --profile train run --rm --build train
```

---

## Deploying on a New Domain

If you have your own cluster data and want to test the model without retraining:

1. **Prepare your data** — produce a `cluster_agg` CSV (schema in `docs/ops_runbook.md`)
2. **Bootstrap thresholds** — estimate per-machine p95/p99 from your historical data
3. **Evaluate** — run `evaluate_domain.py` to get a PR-AUC and deployment recommendation
4. **Deploy** — point `daemon.py` at your data source and domain thresholds

Full walkthrough: `docs/ops_runbook.md`  
CLI flag reference and deeper explanations: `docs/advanced_reference.md`

---

## Documentation

| File | Audience | Contents |
|------|----------|----------|
| `docs/ops_runbook.md` | New-domain operators | Install → smoke test → bootstrap → evaluate → deploy → monitor |
| `docs/technical_report.md` | Researchers / reviewers | All experimental results, ablations, design decisions, rejected approaches |
| `docs/advanced_reference.md` | System integrators | Complete CLI reference, drift monitoring, SLO metrics, FastAPI deployment |
| `CLAUDE.md` | Developers | Pipeline architecture, feature list, leakage guards, training constraints |
