# Energy Infrastructure Intelligence — PoC for EVIDEN

---

## Who is EVIDEN

EVIDEN (formerly Atos) is one of Europe's largest IT infrastructure providers — €5B revenue, 47 countries, positions #1 #2 and #6 on the Green500 list of most energy-efficient supercomputers. They build AI infrastructure (BullSequana servers) and sell sustainability services to enterprise clients.

---

## What EVIDEN Actually Said

> *"Our idea is to enable anyone to send forecasts to the observability framework about any of the existing metrics (performance & eco-efficiency). Then we can enable scheduling policies based on the future state of the nodes instead of what is happening right now."*

**What this tells us:**
- They have an existing observability framework — we do not know which one
- They want forecasts pushed into it — we do not know the format
- The metrics are performance + eco-efficiency — we do not know which ones exactly
- They want scheduling based on predicted future state, not current state
- "Nodes" is their unit — we do not know their infrastructure setup

**What we still need to ask them:**
1. What is the observability framework?
2. What metrics do they already collect (names, units, frequency)?
3. How should forecasts be sent in — what format, what protocol?
4. At what granularity — per node, per service, per namespace?

Until those questions are answered, the integration layer cannot be built.

---

## What is Built and Working

A forecasting pipeline for energy and carbon metrics — 183 tests passing.

```
data_generator.py     →  synthetic data (replaces real data for now)
data_loader.py        →  reads CSV, validates schema, auto-detects columns
prophet_model.py      →  fits Prophet models per metric
timesfm_model.py      →  zero-shot forecasting (no training needed)
ensemble_model.py     →  Prophet + TimesFM residual learning ensemble
api.py                →  REST API exposing forecasts + anomaly detection
anomaly_detector.py   →  3-stage anomaly engine
carbon_analysis.py    →  8-chart analysis pipeline
ensemble_analysis.py  →  3-way model comparison
```

The API runs in Docker. It fits 7 Prophet models at startup and exposes 7 endpoints.

---

## Current Architecture

```
Layer 1 — Data
  data_generator.py        Synthetic telemetry (8,760 rows × 15 cols)
  data_loader.py           Schema validation, graceful degradation on missing cols

Layer 2 — Forecasting
  prophet_model.py         EnergyProphet: multiplicative seasonality + regressors
  timesfm_model.py         EnergyTimesFM: zero-shot, no training
  ensemble_model.py        Prophet baseline + TimesFM residual correction

Layer 3 — REST API + Analysis
  api.py                   FastAPI: 7 Prophet models, 7 endpoints
  carbon_analysis.py       8 PNG charts
  ensemble_analysis.py     Model comparison charts + accuracy table

Layer 4 — Anomaly Detection
  anomaly_detector.py      point anomalies → recurring patterns → investigation leads
  api.py /anomalies        dual-model confidence scoring (Prophet + TimesFM)
  api.py /investigation-leads  ranked patterns with carbon/cost savings
```

**What is NOT built:** any connection to EVIDEN's infrastructure. The system currently reads a local CSV and outputs JSON. Nothing talks to any external system.

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Service readiness, model list |
| `GET /forecast/all?horizon=N` | All 7 metrics + SCI |
| `GET /forecast/{metric}?horizon=N` | Single metric forecast |
| `GET /forecast/sci?horizon=N` | SCI with propagated uncertainty |
| `GET /optimal-window?horizon_days=N` | Lowest-carbon scheduling window per day |
| `GET /anomalies?metric=X&lookback_days=N` | Point anomalies vs Prophet confidence interval |
| `GET /investigation-leads?top_n=N` | Ranked recurring anomaly patterns |

---

## Data Schema (15 columns)

| Column | Unit | Description |
|--------|------|-------------|
| `ds` | datetime | Hourly timestamp |
| `consumption` | kWh/h | Energy drawn — **primary forecast target** |
| `carbonEmissions` | kgCO2e/h | Operational + embodied — **secondary target** |
| `carbonIntensityFactor` | kgCO2/kWh | Grid carbon intensity |
| `greenConsumptionPercentage` | % | Renewable energy fraction |
| `functionalUnit` | req/h | Request rate — SCI denominator + Prophet regressor |
| `cpuUtilization` | fraction | CPU load [0.05, 0.95] |
| `cost` | EUR/h | Electricity cost |
| `softwareCarbonIntensity` | kgCO2e/req | SCI — always derived, never fit directly |
| `operationalEmissions` | kgCO2e/h | `consumption × carbonIntensityFactor` |
| `embodiedEmissions` | kgCO2e/h | Hardware amortisation (constant 0.002) |
| `timeWindow` | s | Measurement window (constant 3600) |
| `measurementSource` | str | "RAPL" or "TDP" |
| `totalConsumption` | kWh | Cumulative |
| `totalCost` | EUR | Cumulative |

**Minimum required for the pipeline:** `ds` + `consumption`. Everything else is optional — the pipeline detects what is present and degrades gracefully.

---

## SCI Formula

```
SCI = (E × I + M) / R
  E = consumption (kWh/h)
  I = carbonIntensityFactor (kgCO2/kWh)
  M = embodiedEmissions (0.002 kgCO2e/h)
  R = functionalUnit (req/h)
```

SCI is always **derived** from component forecasts, never fit directly — fitting a ratio causes instability at low request rates.

---

## Model Results (80/20 train/test, 1 year hourly data)

### Energy — `consumption`
| Model | sMAPE | MAPE |
|-------|-------|------|
| **Prophet** | **5.0%** | **5.2%** |
| TimesFM | 6.4% | 6.1% |
| Ensemble | 5.0% | 5.2% |

Prophet wins. Energy follows clean daily/weekly seasonality + load regressor.

### Carbon — `carbonEmissions`
| Model | sMAPE | MAPE |
|-------|-------|------|
| Prophet | 13.9% | 15.1% |
| **TimesFM** | **7.3%** | **7.0%** |
| Ensemble | 13.0% | 14.0% |

TimesFM wins. Carbon = `consumption × carbonIntensityFactor` — a nonlinear product that Prophet's decomposition cannot capture cleanly.

---

## Docker Commands

```bash
# Run tests
docker compose --profile test run --rm test

# Start the REST API
docker compose up --build

# Run analysis pipeline (generates charts)
docker compose --profile analysis up --build
```

---

## Design Decisions

- **Python 3.11 only** — TimesFM does not support 3.12+
- **Multiplicative seasonality** — energy patterns scale with load
- **Chained forecasting** — `functionalUnit` is forecast first, then injected into `consumption` model. Without this, future regressor values default to 0 (service appears idle during business hours)
- **SCI never fit directly** — always derived from carbon and request rate forecasts
- **Graceful degradation** — missing columns skip charts and metrics, not crash
- **Dual-model anomaly confidence** — "high" = both Prophet AND TimesFM flag the same timestamp; "prophet-only" = weaker signal, could be baseline drift
