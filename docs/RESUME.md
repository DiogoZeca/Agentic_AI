# Energy Infrastructure Intelligence — PoC for EVIDEN

---

## Project Context

This is a **Proof of Concept** built for **EVIDEN** (formerly Atos, €5B revenue, 47 countries), one of Europe's largest IT infrastructure and HPC providers. EVIDEN holds positions #1, #2, and #6 on the Green500 list of the world's most energy-efficient supercomputers, builds AI infrastructure (BullSequana servers), and sells sustainability and decarbonisation services to enterprise clients.

**What EVIDEN asked for:** A system that identifies energy inefficiencies in infrastructure and quantifies the carbon and cost impact — automatically, without needing to know in advance what is running on that infrastructure.

**Why this matters to them:** They manage infrastructure at scale. Inefficiencies that are invisible to operations teams (background workloads running at peak carbon hours, services consuming more energy than their load pattern justifies) represent both unnecessary cost and carbon liability. An AI system that surfaces these automatically — as investigation leads — would be deployable on any client's telemetry data without a lengthy onboarding process.

---

## What This System Proves

**The research gap:** Every existing carbon-aware scheduling system (GREEN, CASPER, EXIGENCE) operates reactively — they respond to current carbon intensity. None forecast where inefficiencies will occur before they happen.

**This system is predictive.** It learns what "normal" looks like for a service (the expected energy and carbon pattern given its load profile), then identifies where and when actual behaviour deviates from that expectation. Those deviations are investigation leads.

**The core value proposition:**

> *"There is a service in this infrastructure consuming 40% more energy than expected every weekday at 15:00. This occurs during the highest-carbon window of the day. Estimated impact if the underlying workload were shifted to the overnight low-carbon window: −29% carbon, −22% cost per occurrence. Investigate what triggers at 15:00."*

The system does not need to know what the service is. It finds the anomaly. The operations team identifies the cause. This works on any infrastructure with hourly telemetry.

**Dual-objective:** Unlike all existing approaches, this system optimises for **both carbon and cost simultaneously**. In most grids these align (overnight is cheaper AND greener), but when they diverge the system surfaces the tradeoff explicitly rather than hiding it.

---

## How It Works — The Core Loop

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1 — Learn the baseline                                     │
│  Prophet fits 7 models on historical metrics.                    │
│  Learns: daily pattern, weekly cycle, annual trend, load         │
│  relationships. This is the "expected" behaviour fingerprint.    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2 — Forecast the next window                               │
│  7-day forecasts for all 7 metrics + SCI.                        │
│  Includes calibrated uncertainty intervals (yhat_lower/upper).   │
│  Identifies the optimal low-carbon scheduling window per day.    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3 — Detect anomalies (Next — Agent Layer)                  │
│  Compare actual consumption against the forecast baseline.       │
│  Flag windows where actual > yhat_upper (unexpected excess).     │
│  Characterise: when, how much above baseline, how often it       │
│  recurs, where it sits on the carbon/cost curve.                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4 — Generate insights (Next — Agent Layer)                 │
│  Claude reasons over the anomalies and optimal windows.          │
│  Produces structured investigation leads with estimated          │
│  savings. No scheduling actions. No prior knowledge of jobs.     │
│  Output: "Investigate X at time T — estimated saving Y%."        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture Overview

```
Layer 1 — Data
  data_generator.py        Synthetic telemetry (8,760 rows × 15 cols, SCI schema)
  data_loader.py           Schema validation + pipeline config (graceful degradation)

Layer 2 — Forecasting Service  ✅ COMPLETE
  prophet_model.py         EnergyProphet (baseline + seasonality + regressors)
  timesfm_model.py         EnergyTimesFM (zero-shot foundation model)
  ensemble_model.py        Residual learning ensemble (Prophet + TimesFM)
  api.py                   FastAPI REST service — 7 Prophet models, 5 endpoints

Layer 3 — Analysis Pipeline  ✅ COMPLETE
  carbon_analysis.py       8-chart Prophet analysis pipeline
  ensemble_analysis.py     3-way model comparison (Prophet / TimesFM / Ensemble)

Layer 4 — Agent Layer  ← NEXT
  anomaly_detector.py      Actual vs forecast gap detection (to be built)
  insight_agent.py         Claude-powered insight generation (to be built)
  mcp_server.py            MCP tools wrapping the forecast API (to be built)
```

---

## Forecasting Layer

### Three Complementary Models

#### Prophet — The Baseline (interpretable)

Meta's statistical decomposition model. Breaks the time series into components:

```
forecast = trend × daily_cycle × weekly_cycle × yearly_cycle
```

- **Trend** — Is baseline energy drifting up or down over months?
- **Daily cycle** — Which hours draw most power?
- **Weekly cycle** — Weekday vs weekend load difference
- **Regressor** — `functionalUnit` (req/h) as an external covariate for energy: more requests → higher CPU → more kWh

Prophet is **fast, interpretable, and explainable**. Its forecast defines what "normal" looks like — deviations from it are anomalies.

#### TimesFM — The Deep Pattern Finder (accurate on carbon)

Google's 200M-parameter foundation model pre-trained on billions of real-world time series. **Zero-shot** — no training needed.

Captures nonlinear patterns, especially the `consumption × carbonIntensityFactor` interaction that Prophet's decomposition misses. Wins on carbon forecast accuracy (7.3% sMAPE vs Prophet's 13.9%).

Weakness: black box, adds noise on well-structured signals.

#### Ensemble — Residual Learning

Two-stage pipeline, not a simple average:

```
Step 1  Prophet predicts energy              →  0.112 kWh/h
Step 2  Actual value                         →  0.149 kWh/h  (burst!)
Step 3  Residual = actual − prophet          →  +0.037
Step 4  TimesFM learns residual patterns     →  predicts +0.033 next hour
Step 5  Final = Prophet + α × TimesFM        →  0.145 ✓
```

- **Adaptive α** — grid-searched on last 10% of training data (minimises MAE). α=0 means Prophet only.
- **Correction clamping** — TimesFM residual clipped to ±50% of Prophet's forecast.
- **Per-metric regressors** — `functionalUnit` regressor only on `consumption` (direct causal link). Omitted for `carbonEmissions` (indirect path through grid intensity).

### Chained Forecasting

Prophet needs future regressor values for every forecast horizon. The naive fallback is `0`, meaning "the service has zero traffic in the future" — which collapses energy forecasts to idle baseline, invalidating every business-hours prediction.

**Fix:** `functionalUnit` has its own Prophet model with strong daily/weekly seasonality. When `consumption` needs a 24-hour forecast, the pipeline first runs the `functionalUnit` model, then injects those predictions as future regressor values.

```
GET /forecast/consumption?horizon=24
  │
  ├─ regressors = ["functionalUnit"]            non-empty → chained path
  │
  ├─ _build_regressor_future_df(24, ["functionalUnit"])
  │    ├─ historical df["functionalUnit"]        real values, 8,760 rows
  │    └─ _run_forecast("functionalUnit", 24)    no regressors → simple path, no recursion
  │         returns realistic business-hours curve (not zeros)
  │
  └─ consumption model.predict(future_df = historical + predicted functionalUnit)
       historical rows → actual functionalUnit   ✓
       future rows     → predicted functionalUnit ✓  (was zero before)
       result: energy forecast reflects morning ramp-up, midday peak, overnight trough
```

No circular recursion: `functionalUnit` has no regressors, so it always takes the simple path.

---

## SCI Framework

All metrics align with the Green Software Foundation's **Software Carbon Intensity** specification (ISO/IEC 21031:2024):

```
SCI = (E × I + M) / R

  E  consumption           kWh/h         energy drawn by the service
  I  carbonIntensityFactor kgCO2/kWh     real-time grid carbon intensity
  M  embodiedEmissions     kgCO2e/h      hardware amortisation (constant 0.002)
  R  functionalUnit        req/h         functional unit — SCI denominator
```

`softwareCarbonIntensity` is always **derived** from component forecasts, never fit directly:

```python
SCI_forecast = carbon_forecast / max(requests_forecast, 1)
```

Fitting SCI as a ratio causes numerical instability at low request rates (near-zero denominator). Deriving it from components propagates uncertainty correctly:

```
yhat_lower = carbon_lower / max(request_upper, 1)   # best-case SCI
yhat_upper = carbon_upper / max(request_lower, 1)   # worst-case SCI
```

---

## Agent Layer — Design (Next Step)

### What the Agent Detects

The forecasting service already defines what "normal" looks like (the Prophet baseline with calibrated uncertainty bands). The agent layer compares actual telemetry against this baseline and flags:

- **Excess consumption anomalies** — actual > `yhat_upper` at a specific hour. Something is consuming more than the load pattern justifies.
- **Recurring patterns** — the same excess appearing at the same time window across multiple days/weeks. Suggests a scheduled background workload.
- **Carbon-cost window misalignment** — recurring compute bursts that correlate with high `carbonIntensityFactor` or high `cost` periods. The work is happening at the most expensive and dirtiest time.

### What the Agent Does Not Need to Know

- What specific service or job is causing the anomaly
- What the job's deadline is
- Whether the job is deferrable
- Anything about the infrastructure topology

The agent identifies the signal. The operations team identifies the cause. This is intentional — it makes the system deployable on any client's data without prior configuration.

### Output Format — Investigation Leads

```json
{
  "anomaly_id": "weekday-15h-cpu-burst",
  "detected_pattern": "CPU utilization 38% above baseline every weekday 15:00–16:00",
  "correlation": "Does not follow request rate pattern (traffic peaks at 13:00)",
  "hypothesis": "Scheduled background workload independent of user demand",
  "current_window_carbon": 0.047,
  "current_window_cost": 0.18,
  "optimal_window": "02:00–05:00",
  "optimal_window_carbon": 0.031,
  "estimated_carbon_saving_pct": 34,
  "estimated_cost_saving_pct": 22,
  "recommendation": "Investigate what service or job triggers at 15:00 on weekdays"
}
```

### Target Architecture — 4 Specialized Agents

Based on the AgenticAI.md research and the Supervisor+DAG pattern:

```
Supervisor Agent
  │
  ├── Monitor Agent
  │     Reads telemetry, identifies candidate anomalies,
  │     computes actual vs forecast gaps per metric
  │
  ├── Forecast Agent
  │     Calls MCP tools (/forecast/all, /optimal-window),
  │     interprets carbon/cost outlook, identifies green windows
  │
  ├── Carbon Agent
  │     Computes dual-objective scores (carbon saving % + cost saving %),
  │     applies policy rules, flags cost/carbon tradeoffs
  │
  └── Scheduler Agent
        Generates ranked investigation leads,
        produces structured output + human-readable summary
        (future: patches K8s CronJob schedules)
```

MCP protocol is the integration layer between the forecast REST API and the agent tools — each endpoint becomes a semantically-described tool that Claude calls natively.

### Trust Curve for Production

```
PoC     → Observe and report (no actions, no prior job knowledge)
           Agent identifies anomalies and investigation leads
           EVIDEN team validates manually

Phase 2 → Semi-automatic (human approval via Slack)
           For clients with known workloads: "We detected X at 15:00.
           Is this the Y job? Should we suggest deferral? [Yes] [No]"

Phase 3 → Automatic for low-risk patterns
           Confirmed recurring patterns → automatic deferral suggestions
           High-impact jobs → still require human approval
           Full audit log for every action

Phase 4 → K8s integration
           Agent patches CronJob schedules, writes carbon annotations,
           KEDA ScaledObject reads forecast signals
```

---

## PoC Roadmap

### Stage 1 — Forecasting Service ✅ COMPLETE

FastAPI service running in Docker. Fits 7 Prophet models at startup. Chained forecasting for regressors.

| Endpoint | Status | Description |
|----------|--------|-------------|
| `GET /health` | ✅ | Model readiness + 7 model names |
| `GET /forecast/all?horizon=N` | ✅ | All 7 metrics + SCI in one response |
| `GET /forecast/{metric}?horizon=N` | ✅ | Single metric forecast |
| `GET /forecast/sci?horizon=N` | ✅ | SCI with propagated uncertainty |
| `GET /optimal-window?horizon_days=N` | ✅ | Best low-carbon window per day |

### Stage 2 — Anomaly Detection Layer

Compare Prophet forecast baseline against actual historical data. Flag windows where actual consumption exceeds `yhat_upper` with statistical significance. Characterise by frequency, magnitude, and position on the carbon/cost curve.

### Stage 3 — Agent Insight Generator

Claude agent consuming anomaly reports + optimal window data via MCP tools. Produces structured investigation leads. No job registry required — the system works on raw telemetry.

### Stage 4 — PoC Demo Package

End-to-end demonstration: Docker stack up → synthetic data → analysis charts → agent runs → insight report. Story: "Give us your infrastructure telemetry. This system identifies your energy inefficiencies automatically."

### Post-PoC (Production Path)

- Real data connector: Prometheus/Kepler → same 15-column schema
- Workload registry: for clients who want automated scheduling suggestions
- K8s CronJob patching: agent acts, not just recommends
- KEDA ScaledObject: scales workloads based on carbon forecast signals
- Feedback loop: actual vs forecast savings measurement over time

---

## Project Structure

```
Agentic_AI/
├── AIModel/
│   ├── api.py                    REST forecasting service (FastAPI, 7 Prophet models)
│   ├── data_generator.py         Synthetic telemetry generator (8,760 rows × 15 cols)
│   ├── data_loader.py            Schema validation + PipelineConfig
│   ├── prophet_model.py          EnergyProphet wrapper (fit/predict/evaluate)
│   ├── timesfm_model.py          EnergyTimesFM wrapper (zero-shot)
│   ├── ensemble_model.py         Prophet + TimesFM residual ensemble
│   ├── carbon_analysis.py        8-chart Prophet analysis pipeline
│   ├── ensemble_analysis.py      3-way model comparison (2 charts + accuracy table)
│   ├── visualizations.py         Reusable Matplotlib plotting utilities
│   ├── Dockerfile                Service image (Prophet + FastAPI, lightweight)
│   ├── Dockerfile.analysis       Analysis image (full stack + TimesFM + matplotlib)
│   ├── requirements.txt          Full deps (analysis + TimesFM)
│   └── requirements-service.txt  Lightweight deps (service only)
├── docker-compose.yml            Two modes: analysis profile + API service
├── docs/
│   ├── RESUME.md                 This file — full project reference
│   ├── AgenticAI.md              Research notes: agentic AI, multi-agent patterns, carbon tools
│   └── [research PDFs]           GREEN, CASPER, EXIGENCE, TimesFM, Mixture-of-Agents, etc.
└── START.md                      Docker-first quick start guide
```

---

## What Each File Does

### `data_generator.py` — Synthetic Telemetry

Generates 1 year of hourly data (8,760 rows) simulating a CPU/memory-based web service. All computation vectorised with NumPy (no Python loops).

| Signal | Generation logic | Range |
|--------|-----------------|-------|
| `functionalUnit` | Business-hours curve (50→500 req/h), weekend ×0.4, +15% annual trend, ±8% noise | 5–600 req/h |
| `cpuUtilization` | `0.10 + 0.75 × (load^0.8)`, clipped [0.05, 0.95] | 0.14–0.85 |
| `consumption` | `(20 + 130 × cpuUtil) × PUE_1.3 / 1000` kWh/h | 0.049–0.170 kWh/h |
| `carbonIntensityFactor` | 0.4 base − midday solar dip (Gaussian σ=2 at noon) − summer offset, ±0.015 noise | 0.30–0.50 kgCO2/kWh |
| `greenConsumptionPercentage` | Gaussian peak at 13:00, summer boost | 20–70% |
| `carbonEmissions` | `consumption × carbonIntensityFactor + 0.002` | ~0.016–0.086 kgCO2e/h |
| `cost` | `consumption × 0.12 × peak_factor` (peak 1.5× during 09:00–18:00) | ~0.006–0.030 EUR/h |
| `softwareCarbonIntensity` | `carbonEmissions / max(functionalUnit, 1)` | ~0.00002–0.0018 kgCO2e/req |
| `measurementSource` | "RAPL" if cpuUtil > 0.6, else "TDP" | categorical |

### `data_loader.py` — Schema Validation + Config

- `load_and_validate(path)` — enforces `["ds", "consumption"]` as minimum, prints schema report, returns DataFrame
- `build_pipeline_config(df)` — auto-detects available columns, returns `PipelineConfig`:
  - `fit_metrics`: all 7 directly-fit metrics present in data: `consumption`, `carbonEmissions`, `carbonIntensityFactor`, `greenConsumptionPercentage`, `cpuUtilization`, `functionalUnit`, `cost`
  - `regressor_map`: `{"consumption": ["functionalUnit"], ...}` — only `consumption` uses a regressor
  - `has_sci`: True when both `carbonEmissions` and `functionalUnit` present
  - `available_charts`: chart name → bool (graceful degradation when columns missing)

### `prophet_model.py` — EnergyProphet

Multiplicative seasonality — energy patterns *scale* with load (doubling requests roughly doubles power).

```python
EnergyProphet(
    seasonality_mode="multiplicative",
    changepoint_prior_scale=0.05,
    regressors=["functionalUnit"],
)
```

`predict(periods, freq, future_df)`: merges `future_df` on `ds` timestamp to fill regressor values. Any unmatched future timestamps fill with `0` — avoid this by using chained forecasting in the service layer.

### `timesfm_model.py` — EnergyTimesFM

Google TimesFM 200M parameters (PyTorch). Lazy-loads from HuggingFace (~925 MB, cached in `hf-cache` Docker volume after first run).

- `fit()` — stores history (zero-shot, no training)
- `predict()` — uses last 512 hours as context, forecasts in 128-hour chunks, returns 10th/90th percentile uncertainty
- `forecast_batch()` — efficient multi-metric inference in one model call

### `ensemble_model.py` — EnsembleForecaster

1. Prophet fits baseline → compute residuals
2. TimesFM learns residual patterns
3. α learned via grid-search on last 10% of training (minimises MAE)
4. `ensemble = prophet + α × clamp(timesfm_residual, ±50%)`

`get_residual_stats()` — diagnostic: high `autocorr_lag1` means TimesFM corrections are most useful.

### `carbon_analysis.py` — Prophet Analysis Pipeline

Fits Prophet on all metrics, derives SCI, generates 8 charts:

| # | Chart | What it reveals for anomaly investigation |
|---|-------|------------------------------------------|
| 01 | `01_service_profile.png` | Energy + request load shape — confirms daily pattern baseline |
| 02 | `02_energy_forecast.png` | 7-day energy forecast with CI — defines normal bounds |
| 03 | `03_energy_decomposition.png` | Trend / weekly / daily / yearly breakdown |
| 04 | `04_daily_pattern.png` | Mean energy by hour — hourly baseline fingerprint |
| 05 | `05_weekly_pattern.png` | Weekday vs weekend delta — weekly baseline fingerprint |
| 06 | `06_carbon_intensity.png` | Grid intensity + green % + optimal window |
| 07 | `07_sci_by_hour.png` | SCI (kgCO2e/req) by hour — efficiency profile |
| 08 | `08_carbon_forecast.png` | 7-day carbon forecast — carbon outlook |

### `ensemble_analysis.py` — Model Comparison

Runs all three models per metric, produces:
- `07_forecast_comparison.png` — actual vs Prophet/TimesFM/Ensemble on test week
- `08_model_accuracy.png` — sMAPE grouped bars per metric, ★ winner highlighted

### `api.py` — Forecasting REST Service

FastAPI service. Fits 7 Prophet models at startup (~2–3 min). All models persist in-process; single worker by design (model state must not be shared across forks).

Chained forecasting implemented in `_build_regressor_future_df()` and `_run_forecast()` — no zero-fill for future regressors.

---

## Data Schema

```
Column                      Unit          Description
─────────────────────────────────────────────────────────────────────
ds                          datetime      Hourly timestamp (Prophet required)

── Service Load ──
functionalUnit              req/h         Request rate — SCI denominator + Prophet regressor
cpuUtilization              fraction      CPU load [0.05, 0.95]
timeWindow                  s             Measurement window (constant 3600)
measurementSource           str           "RAPL" (hw counter) or "TDP" (estimate)

── Energy ──
consumption                 kWh/h         Energy drawn  ← PRIMARY FORECAST TARGET
totalConsumption            kWh           Cumulative energy
cost                        EUR/h         Electricity cost (peak/off-peak pricing)
totalCost                   EUR           Cumulative electricity cost

── Carbon (SCI components) ──
carbonIntensityFactor       kgCO2/kWh     Real-time grid carbon intensity
greenConsumptionPercentage  %             Renewable energy fraction
operationalEmissions        kgCO2e/h      consumption × carbonIntensityFactor
embodiedEmissions           kgCO2e/h      Hardware amortisation (constant 0.002)
carbonEmissions             kgCO2e/h      Operational + embodied ← SECONDARY FORECAST TARGET

── SCI (derived) ──
softwareCarbonIntensity     kgCO2e/req    SCI = carbonEmissions / functionalUnit ← NEVER FIT DIRECTLY
```

**Minimum required for the pipeline:** `ds` + `consumption`.
All other columns are optional — the pipeline auto-detects and degrades gracefully.

---

## Docker Usage

Everything runs in Docker. No local Python required.

```bash
# Analysis pipeline — generate data + 10 charts
docker compose --profile analysis up --build

# Forecasting API — starts on http://localhost:8000
docker compose up --build

# API is ready when logs show:
# INFO  Service ready — 8760 rows, models: ['consumption', 'carbonEmissions', ...]
```

Charts appear in `AIModel/analysis_output/` on the host (bind-mounted).
Data is persisted in `AIModel/data/` (bind-mounted).
TimesFM model weights cached in `hf-cache` Docker named volume (~925 MB, downloaded once).

Full instructions → `START.md`

---

## Model Results

Evaluation on 1 year of hourly synthetic data (80/20 train/test split):

### Energy Consumption — `consumption` (with `functionalUnit` regressor)

| Model | sMAPE | MAPE | RMSE |
|-------|-------|------|------|
| **Prophet** | **5.0%** | **5.2%** | **0.0052** |
| TimesFM | 6.4% | 6.1% | 0.0078 |
| Ensemble | 5.0% | 5.2% | 0.0052 |

Prophet dominates. Energy follows clean daily/weekly patterns; `functionalUnit` regressor explains load scaling precisely. Chained forecasting ensures future regressor values are realistic.

### Carbon Emissions — `carbonEmissions` (no regressor)

| Model | sMAPE | MAPE | RMSE |
|-------|-------|------|------|
| Prophet | 13.9% | 15.1% | 0.0061 |
| **TimesFM** | **7.3%** | **7.0%** | **0.0036** |
| Ensemble | 13.0% | 14.0% | 0.0058 |

TimesFM dominates. Carbon = `consumption × carbonIntensityFactor` — a nonlinear product that Prophet's decomposition cannot capture cleanly.

### Software Carbon Intensity — `softwareCarbonIntensity` (derived)

| Model | sMAPE | MAPE | RMSE |
|-------|-------|------|------|
| Prophet | 31.6% | 189.5% | 0.00458 |
| **TimesFM** | **15.2%** | **17.0%** | **0.00011** |
| Ensemble | 30.8% | 194.8% | 0.00458 |

MAPE inflated by near-zero denominators at off-peak. **sMAPE is the reliable metric.** TimesFM wins 2×.

### Signal Selection by Use Case

| Use case | Best model | Reason |
|----------|-----------|--------|
| Anomaly baseline (expected energy) | Prophet | Interpretable seasonality + regressor |
| Carbon forecast for anomaly scoring | TimesFM | Nonlinear grid intensity interactions |
| SCI hourly efficiency profile | TimesFM | Best ratio estimation via components |
| When does TimesFM help most? | `get_residual_stats()` | High `autocorr_lag1` → corrections useful |

---

## Design Principles

1. **Forecast targets are absolute, not ratios** — SCI is always derived post-forecast to avoid ratio instability at low denominators
2. **Model selection is per-metric** — Prophet for energy (structured seasonality), TimesFM for carbon and SCI (nonlinear interactions)
3. **Graceful degradation** — missing columns cause charts/metrics to skip, not crash; minimum schema is `ds` + `consumption`
4. **Fair evaluation** — identical 80/20 train/test splits across all three models; no data leakage
5. **Interpretability preserved** — Prophet decomposition charts are the primary anomaly investigation input; the baseline must be explainable to operations teams
6. **Adaptive residual weighting** — α=0 means "trust Prophet only"; the system learns when TimesFM corrections are beneficial per metric
7. **Measurement source tracking** — `measurementSource` distinguishes RAPL hardware counters from TDP estimates; data quality signals matter for anomaly confidence
8. **Chained forecasting for regressors** — when metric A uses metric B as a regressor, B is always forecast first; zero-filling future regressors produces idle-state predictions during business hours, invalidating all peak-hour forecasts
9. **Dual-objective optimisation** — every insight includes both carbon saving and cost saving; when they conflict the tradeoff is surfaced explicitly, never hidden
10. **No prior knowledge required** — the anomaly detection layer works from raw telemetry alone; no workload catalogue, no job schedules, no infrastructure topology needed for the PoC
