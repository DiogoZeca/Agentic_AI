# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

**Energy and carbon metrics forecasting pipeline for EVIDEN.** The system forecasts energy consumption and carbon emissions for infrastructure services, aligned with the Green Software Foundation's SCI framework.

EVIDEN's requirement (their exact words): *"enable anyone to send forecasts to the observability framework about any of the existing metrics (performance & eco-efficiency). Then we can enable scheduling policies based on the future state of the nodes instead of what is happening right now."*

**What is known:** EVIDEN wants forecasts pushed into their observability framework to enable predictive scheduling.
**What is unknown:** which observability framework, what format, what metrics they collect, at what granularity.

The pipeline currently reads synthetic data from a local CSV and exposes forecasts via REST API. No connection to any external system exists yet.

## Directory Layout

```
Agentic_AI/
├── CLAUDE.md              ← this file (project-wide guidance)
├── session-notes.md       ← session history and decision log
├── docker-compose.yml     ← compose file (must be run from HERE, not from AIModel/)
└── AIModel/               ← all application code lives here
    ├── Dockerfile          ← API service image (Prophet + TimesFM)
    ├── Dockerfile.test     ← test image (Prophet only, no TimesFM/PyTorch)
    ├── Dockerfile.analysis ← analysis image (uses COPY *.py ./)
    ├── requirements-service.txt
    ├── requirements-test.txt
    ├── data/               ← CSV data files (mounted as Docker volume)
    ├── analysis_output/    ← PNG charts written by carbon_analysis.py / ensemble_analysis.py
    └── tests/
        ├── test_data.py
        ├── test_forecasting.py
        ├── test_api.py
        ├── test_anomaly.py
        ├── test_physics.py
        └── test_ensemble.py
```

**Local Python commands** (venv, analysis scripts): run from `AIModel/`.
**Docker Compose commands**: run from the project root (`Agentic_AI/`) — that is where `docker-compose.yml` lives.

## Commands

```bash
# Enter the application directory first
cd AIModel/

# Activate the venv (Python 3.11 required — TimesFM limitation)
source .venv/bin/activate          # or: .venv/bin/python3 directly

# Generate synthetic service data (writes to data/sample_energy_data.csv)
.venv/bin/python3 data_generator.py

# Validate loader + pipeline config
.venv/bin/python3 -c "from data_loader import load_and_validate, build_pipeline_config; df=load_and_validate('data/sample_energy_data.csv'); cfg=build_pipeline_config(df); print('metrics:', cfg.metrics, 'has_sci:', cfg.has_sci)"

# Prophet-only analysis — 8 charts in analysis_output/
.venv/bin/python3 carbon_analysis.py

# Ensemble comparison (Prophet vs TimesFM vs Ensemble) — 2 charts + MAPE table
.venv/bin/python3 ensemble_analysis.py
```

Run the test suite (Docker — recommended, run from project root `Agentic_AI/`):

```bash
# All 260 tests (5 skipped — TimesFM integration tests, require full image)
docker compose --profile test run --rm test

# Rebuild image first (after code changes)
docker compose --profile test run --rm --build test

# Single test file
docker compose --profile test run --rm test pytest tests/test_data.py -v
docker compose --profile test run --rm test pytest tests/test_forecasting.py -v
docker compose --profile test run --rm test pytest tests/test_api.py -v
docker compose --profile test run --rm test pytest tests/test_anomaly.py -v
docker compose --profile test run --rm test pytest tests/test_physics.py -v
docker compose --profile test run --rm test pytest tests/test_ensemble.py -v
```

Run the analysis pipeline (Docker — run from project root `Agentic_AI/`):

```bash
# Run all 3 services: generate data + carbon_analysis.py + ensemble_analysis.py
docker compose --profile analysis up --build

# Run only carbon_analysis.py (8 charts)
docker compose --profile analysis up --build generate analyse

# Run only ensemble_analysis.py (2 charts + MAPE table, needs TimesFM)
docker compose --profile analysis up --build generate ensemble

# Charts land in AIModel/analysis_output/ on the host (volume-mounted)
```

Run the API service (from project root `Agentic_AI/`):

```bash
# Starts FastAPI on http://localhost:8000
docker compose up --build

# Health check
curl http://localhost:8000/health
```

Six test groups:
- `test_data.py`        — data contract (schema, value ranges, SCI identity)
- `test_forecasting.py` — chained forecasting validation (Prophet regressor chain)
- `test_api.py`         — API contract (endpoints, response shapes, error codes)
- `test_anomaly.py`     — anomaly detection engine (point anomalies, recurring patterns, investigation leads)
- `test_physics.py`     — physics formula contract (non-negativity, interval ordering, sMAPE validity)
- `test_ensemble.py`    — ensemble pure functions (21 tests) + integration contract (5 skipped without TimesFM)

## Architecture

Eleven-module pipeline:

- **data_generator.py** — Generates 8,760 hourly rows simulating a CPU/memory-based web service. Vectorised NumPy. No Python loops.

- **data_loader.py** — `load_and_validate()` enforces minimum schema (`ds`, `consumption`); raises `FileNotFoundError` / `ValueError` on failure (not `sys.exit`). `build_pipeline_config()` auto-detects metrics/regressors and returns a `PipelineConfig` dataclass. Pipeline degrades gracefully on missing columns.

- **model_base.py** — `ForecasterBase` ABC: formal interface contract for all single-metric forecasting models. Declares `fit()`, `predict()`, `evaluate()`. `EnsembleForecaster` is intentionally excluded (returns a nested dict, different contract).

- **prophet_model.py** — `EnergyProphet(ForecasterBase)` wraps Meta's Prophet. Uses **multiplicative** seasonality. Workflow: `fit()` → `predict()` → `evaluate()`. Accepts `functionalUnit` as a regressor for `consumption` forecasting.

- **timesfm_model.py** — `EnergyTimesFM(ForecasterBase)` wraps Google's TimesFM (200M params, zero-shot). Lazy-loads from HuggingFace (~925 MB first run). Same `fit/predict/evaluate` interface. `evaluate()` returns all 7 keys incl. `sMAPE`.

- **physics_constraint.py** — Pure-function module implementing `carbonEmissions = E × CIF + 0.002`. Key functions: `derive_carbon_emissions(consumption_fc, cif_fc)`, `evaluate_formula_accuracy(df)`, `check_physical_consistency(forecast_df)`. `EMBODIED_EMISSIONS_KGC02E_H = 0.002` constant. No learnable parameters.

- **model_registry.py** — `MODEL_ROUTING` dict + `get_best_model_key(metric, available_keys)`. Replaces the old `_TIMESFM_PREFERRED` frozenset. For `carbonEmissions`: routing order is `["formula", "timesfm", "prophet"]`. For `consumption`: `["prophet"]`. Change routing here without touching API logic.

- **ensemble_model.py** — `EnsembleForecaster`: Prophet baseline → residual = actual − Prophet → TimesFM learns residual patterns → `ensemble = Prophet + alpha × clamp(TimesFM_residual, ±50%)`. Alpha learned by grid-search on a 10% validation split. `get_test_predictions()` exposes raw test arrays for analysis scripts.

- **anomaly_detector.py** — Pure analysis engine (no Prophet/FastAPI imports). Three-stage pipeline: `detect_point_anomalies()` → `find_recurring_patterns()` → `build_investigation_leads()`. Returns typed dataclasses; the API layer owns Prophet interaction.

- **carbon_analysis.py** — Prophet + TimesFM + formula pipeline; produces 8 PNG charts + scheduling insights. Prints a **forecast method comparison table** (Formula vs TimesFM/Prophet) with sMAPE for each. TimesFM eval is now correctly re-computed after override (was a bug).

- **ensemble_analysis.py** — Three-way evaluation (Prophet / TimesFM / Ensemble); produces 2 PNG charts + sMAPE summary table.

## Data Schema (15 columns, 8,760 rows)

Prophet requires a `ds` (datetime) column. The target column is renamed to `y` internally by `prepare_data()`.

| Column | Unit | Description |
|--------|------|-------------|
| `ds` | datetime | Hourly timestamp |
| `timeWindow` | s | Measurement window (constant 3600) |
| `consumption` | kWh/h | Energy drawn by the service (PRIMARY TARGET) |
| `totalConsumption` | kWh | Cumulative energy |
| `measurementSource` | str | "RAPL" if cpuUtil > 0.6, else "TDP" |
| `cost` | EUR/h | Electricity cost (peak/off-peak pricing) |
| `totalCost` | EUR | Cumulative electricity cost |
| `carbonEmissions` | kgCO2e/h | Operational + embodied (SECONDARY TARGET) |
| `operationalEmissions` | kgCO2e/h | `consumption × carbonIntensityFactor` |
| `embodiedEmissions` | kgCO2e/h | Hardware amortization (constant 0.002) |
| `carbonIntensityFactor` | kgCO2/kWh | Grid carbon intensity (regressor, not target) |
| `greenConsumptionPercentage` | % | Renewable energy fraction |
| `softwareCarbonIntensity` | kgCO2e/req | SCI = carbonEmissions / functionalUnit (DERIVED) |
| `functionalUnit` | req/h | Request rate — the SCI denominator + Prophet regressor |
| `cpuUtilization` | fraction | CPU load fraction [0.05, 0.95] |

**Forecast targets:** `consumption`, `carbonEmissions`. `softwareCarbonIntensity` is always derived as `carbon / requests`, never fit directly.

**Regressor:** `functionalUnit` is a Prophet regressor for `consumption` only (direct causal: more requests → more CPU → more energy). For future rows where `functionalUnit` is unknown, the model fills `0` (unconditional extrapolation).

## SCI Formula

```
SCI = (E × I + M) / R
    E = consumption (kWh/h)
    I = carbonIntensityFactor (kgCO2/kWh)
    M = embodiedEmissions (kgCO2e/h = 0.002)
    R = functionalUnit (req/h)
```

## Typical Usage Pattern

```python
from data_loader import load_and_validate, build_pipeline_config
from prophet_model import EnergyProphet
from ensemble_model import EnsembleForecaster

df = load_and_validate("data/sample_energy_data.csv")
config = build_pipeline_config(df)

# Prophet forecast (energy with request-rate regressor)
model = EnergyProphet(regressors=["functionalUnit"])
model.fit(df, "consumption")
forecast = model.predict(periods=168, future_df=df)   # 7-day horizon

# Ensemble three-way evaluation
ens = EnsembleForecaster(regressors=["functionalUnit"])
results = ens.evaluate(df, "consumption")
# results = {"prophet": {sMAPE, MAPE, RMSE, ...}, "timesfm": {...}, "ensemble": {...}}
```

## Key Design Constraints

- **Python 3.11 only** — TimesFM does not support 3.12+
- **SCI never fit directly** — always derived from component forecasts (ratio stability)
- **Multiplicative seasonality** — energy patterns scale with load (not additive)
- **293-test suite** — six groups: data contract, chained forecasting, API contract (incl. POST /data, /forecast/peak, L.1801 metadata), anomaly detection engine, physics formula contract, ensemble pure functions
- **No external integration yet** — reads local CSV, outputs JSON REST API. Integration with EVIDEN's stack depends on answers they have not yet provided.
- **Formula first for carbonEmissions** — `MODEL_ROUTING` prioritises the physics formula over TimesFM; falls back gracefully if component models are unavailable
- **evaluate() must return 7 keys** — MAE, MSE, RMSE, MAPE, sMAPE, train_size, test_size. Missing sMAPE causes silent crashes in comparison tables.
- **`node` query param** — the API accepts `?node=<name>` on all forecast/anomaly endpoints. Per-node state isolation is implemented; nodes bootstrap from the default model until their own background refit completes.
- **`analysis_output/` is gitignored** — PNG charts are written there at runtime; directory is created by Docker at build time and mounted as a volume.

## API Endpoints (10 total)

| Method | Path | Key query params | Description |
|--------|------|-----------------|-------------|
| GET | `/health` | — | Liveness + model state + ITU-T L.1801 partial compliance declaration |
| POST | `/data` | — | Ingest new rows (JSON), triggers async Prophet refit; returns 202 immediately |
| GET | `/forecast/all` | `horizon` (1–168, default 1), `node` | All 7 metrics + SCI in one response |
| GET | `/forecast/sci` | `horizon` (1–168, default 1), `include_breakdown`, `node` | Derived SCI with propagated uncertainty intervals + functional unit declaration |
| GET | `/forecast/peak` | `metric` (default "consumption"), `horizon` (1–168, default 1), `node` | Peak provisioning value (yhat_upper) for KEDA/HPA auto-scaling |
| GET | `/forecast/{metric}` | `horizon` (1–168, default 1), `include_breakdown`, `node` | Best-model forecast for one metric (formula→TimesFM→Prophet) |
| GET | `/optimal-window` | `horizon_days` (1–14, default 7), `window_hours` (1–12, default 6), `node` | Lowest-carbon scheduling window per day |
| GET | `/anomalies` | `metric` (required), `lookback_days` (1–365, default 30), `direction` (excess/deficit/both, default excess), `node` | Point anomalies with dual-model confidence scoring |
| GET | `/investigation-leads` | `lookback_days` (1–365, default 30), `top_n` (1–20, default 5), `min_occurrences` (default 3), `node` | Ranked recurring anomaly patterns with carbon/cost saving estimates |
| GET | `/metrics/prometheus` | `horizon` (1–168, default 1), `node` | Consumption + carbon forecasts in Prometheus text format — yhat, yhat_upper, yhat_lower as separate GAUGE families with `horizon` label |

`node` is accepted on all endpoints. Per-node isolation is implemented; nodes bootstrap from the default model until their own background refit completes.

## Physics Formula

```
carbonEmissions = consumption × carbonIntensityFactor + EMBODIED_EMISSIONS_KGC02E_H
EMBODIED_EMISSIONS_KGC02E_H = 0.002  (kgCO2e/h hardware amortization)
```

Implemented in `physics_constraint.py`. Interval bounds propagated element-wise (lower×lower, upper×upper). All outputs clipped to >= 0.
