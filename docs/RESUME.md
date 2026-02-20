# Energy-Aware Kubernetes Scheduling via AI Forecasting

## System Purpose

Traditional Kubernetes schedulers place pods based on **current** CPU/memory requests and cluster capacity. They are blind to two critical dimensions:

1. **Energy cost** — how much power a placement decision will actually draw
2. **Carbon cost** — when the grid is dirty vs clean, making the same workload more or less harmful

This system closes that gap. It uses a Prophet + TimesFM ensemble to **forecast** energy consumption and carbon emissions for a CPU/memory-based web service, then surfaces actionable scheduling signals aligned with the Green Software Foundation's **Software Carbon Intensity (SCI)** framework.

The forecasting engine feeds two Kubernetes layers:

```
┌──────────────────────────────────────────────────────────────────┐
│            AI FORECASTING ENGINE  (this repo)                    │
│  Input: hourly service metrics (CPU util, req/h, kWh, kgCO2e)  │
│  Output: 7-day energy + carbon forecasts + SCI efficiency map   │
└────────────────────┬─────────────────────────────────────────────┘
                     │  forecast signals
          ┌──────────┴──────────┐
          ▼                     ▼
┌─────────────────────┐  ┌────────────────────────────────────────┐
│  UPPER LAYER        │  │  LOWER LAYER                           │
│  Orchestration      │  │  Scheduler                             │
├─────────────────────┤  ├────────────────────────────────────────┤
│ • Defer batch jobs  │  │ • Weight nodes by predicted kWh/req    │
│   to low-carbon     │  │ • Bin-pack pods on most efficient node │
│   windows           │  │   given current CPU util forecast      │
│ • Namespace-level   │  │ • Avoid overcommit on nodes predicted  │
│   workload shifting │  │   to breach energy budget in <2h       │
│ • HPA target        │  │ • Node scoring plugin reads SCI map    │
│   adjustments when  │  │   (kgCO2e/req by hour) to prefer      │
│   SCI is trending   │  │   placements during green windows      │
│   up                │  │                                        │
└─────────────────────┘  └────────────────────────────────────────┘
```

> **Current state:** The AI pipeline is fully operational and produces forecasts + charts. Kubernetes API integration (Custom Scheduler Plugin / Operator) is the next development milestone.

---

## Why Two Layers?

Kubernetes scheduling operates at two distinct granularities that require different signal types:

| Layer | Kubernetes Component | Decision horizon | Signal needed |
|-------|---------------------|-----------------|---------------|
| **Upper / Orchestration** | Deployment controller, HPA, CronJob, cluster autoscaler | Hours to days | 7-day Prophet forecast — trend + seasonal patterns |
| **Lower / Scheduler** | `kube-scheduler` (scoring phase) | Seconds to minutes | Rolling 2-hour ensemble forecast — fine-grained accuracy |

Prophet's interpretable trend/seasonality output is ideal for orchestration (predictable daily patterns). TimesFM's superior short-term accuracy (sMAPE 7.3% vs Prophet's 13.9% on carbon) is the right signal for the scheduler's near-term node scoring.

---

## How the Forecasting Pipeline Works

### Two Complementary Models

#### Prophet (The Explainer)

Meta's statistical decomposition model breaks the time series into interpretable pieces:

```
forecast = trend × daily_cycle × weekly_cycle
```

- **Trend** — Is baseline energy going up or down over weeks?
- **Daily cycle** — Which hours of the day draw most power?
- **Weekly cycle** — How do weekdays compare to weekends (business-hours traffic)?
- **Regressor** — `functionalUnit` (req/h) as an external covariate for energy forecasting: more requests → higher CPU utilisation → more kWh.

Prophet is **fast, interpretable, and explainable** — but misses complex nonlinear interactions.

#### TimesFM (The Deep Pattern Finder)

Google's 200M-parameter foundation model pre-trained on billions of real-world time series. **Zero-shot** — no training needed, just run inference.

Strengths: captures nonlinear patterns, load-to-carbon intensity interactions, and subtle residual structure that Prophet leaves behind.

Weakness: black box, and can add noise when the signal is well-explained by seasonality alone.

#### Ensemble (Residual Learning)

The two models form a pipeline, not a simple average:

```
Step 1  Prophet predicts energy consumption     →  0.112 kWh/h
Step 2  Actual value                            →  0.149 kWh/h  (burst!)
Step 3  Residual = actual − prophet             →  +0.037       (missed pattern)
Step 4  TimesFM learns patterns in residuals    →  predicts +0.033 next hour
Step 5  Final = Prophet + alpha × TimesFM       →  0.112 + 1.0×0.033 = 0.145 ✓
```

**Design choices:**
- **Adaptive alpha** — grid-searched on the last 10% of training data (minimises MAE). `alpha=0` means TimesFM corrections are ignored entirely.
- **Correction clamping** — TimesFM residual corrections are clipped to ±50% of Prophet's forecast to prevent wild swings.
- **Per-metric regressors** — `functionalUnit` is a regressor only for `consumption` (direct causal: requests → CPU → energy). It is omitted for `carbonEmissions` (indirect path through grid intensity).

---

## SCI Framework

All metrics are aligned with the Green Software Foundation's **Software Carbon Intensity** specification:

```
SCI = (E × I + M) / R

  E  consumption           kWh/h         energy drawn by the service
  I  carbonIntensityFactor kgCO2/kWh     real-time grid carbon intensity
  M  embodiedEmissions     kgCO2e/h      hardware amortisation (constant 0.002)
  R  functionalUnit        req/h         functional unit — the SCI denominator
```

`softwareCarbonIntensity` (kgCO2e/req) is always **derived** from component forecasts, never fit directly as a ratio:

```python
SCI_forecast = carbon_forecast / max(requests_forecast, 1)
```

This avoids the numerical instability that comes from fitting ratio metrics with near-zero denominators.

---

## Project Structure

```
AIModel/
├── requirements.txt        # Dependencies
├── data_generator.py       # Synthetic web service data generator (15 cols, 8,760 rows)
├── data_loader.py          # CSV validation + PipelineConfig auto-detection
├── prophet_model.py        # EnergyProphet — Prophet wrapper (fit/predict/evaluate)
├── timesfm_model.py        # EnergyTimesFM — TimesFM wrapper (zero-shot)
├── ensemble_model.py       # EnsembleForecaster (Prophet + TimesFM residuals)
├── visualizations.py       # 5 reusable Matplotlib plotting functions
├── carbon_analysis.py      # Prophet-only analysis → 8 charts + scheduling insights
├── ensemble_analysis.py    # Three-way model comparison → 2 charts + MAPE table
├── data/                   # CSV data storage (sample_energy_data.csv)
├── analysis_output/        # All generated charts (10 PNGs)
├── CLAUDE.md               # Claude Code guidance (concise)
└── RESUME.md               # This file — full developer reference
```

---

## What Each File Does

### `data_generator.py` — Web Service Synthetic Data

Generates 1 year of hourly data (8,760 rows) simulating a **CPU/memory-based web service** using vectorised NumPy (no Python loops):

| Signal | Generation logic | Range |
|--------|-----------------|-------|
| `functionalUnit` | Business-hours curve (50→500 req/h), weekend ×0.4, +15% annual growth, ±8% noise | 5 – ~600 req/h |
| `cpuUtilization` | `0.10 + 0.75 × (load_fraction^0.8)`, clipped [0.05, 0.95] | 0.14 – 0.85 |
| `consumption` | `(20 + 130 × cpuUtil) × 1.3 / 1000` kWh/h | 0.049 – 0.170 kWh/h |
| `carbonIntensityFactor` | 0.4 base − midday solar dip (Gaussian σ=2 at hour 12) − summer −0.05, noise ±0.015 | 0.30 – 0.50 kgCO2/kWh |
| `greenConsumptionPercentage` | Gaussian peak at 13:00, summer boost | 20 – 70% |
| `carbonEmissions` | operational + embodied | ~0.016 – 0.086 kgCO2e/h |
| `softwareCarbonIntensity` | `carbonEmissions / functionalUnit` | ~0.00002 – 0.0018 kgCO2e/req |
| `measurementSource` | "RAPL" if cpuUtil > 0.6, "TDP" otherwise | categorical |

### `data_loader.py` — Schema Validation + Config

- `load_and_validate(path)` — enforces `["ds", "consumption"]` as minimum required columns, prints a schema report, returns DataFrame
- `build_pipeline_config(df)` — returns `PipelineConfig` dataclass:
  - `metrics`: `["consumption", "carbonEmissions", "softwareCarbonIntensity"]`
  - `fit_metrics`: `["consumption", "carbonEmissions", "functionalUnit"]`
  - `regressor_map`: `{"consumption": ["functionalUnit"], "carbonEmissions": [], "functionalUnit": []}`
  - `has_sci`: True when both `carbonEmissions` and `functionalUnit` are present
  - `available_charts`: dict of chart name → bool (graceful degradation)

### `prophet_model.py` — EnergyProphet

Wraps Prophet into a clean workflow. Key design: **multiplicative** seasonality — energy patterns *scale* with load (doubling requests roughly doubles power, not adds a constant).

```python
EnergyProphet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    seasonality_mode="multiplicative",
    changepoint_prior_scale=0.05,
    regressors=["functionalUnit"],   # optional
)
```

Methods: `fit(df, target_column)` → `predict(periods, freq, future_df)` → `evaluate(df, target_column)` → `get_seasonality_components()`

`quick_forecast(csv_path, target_column, forecast_periods)` — convenience end-to-end function.

### `timesfm_model.py` — EnergyTimesFM

Wraps Google's TimesFM (200M parameters, PyTorch backend). Lazy-loads from HuggingFace (~925 MB on first run).

- `fit()` stores historical data (zero-shot: no actual training)
- `predict()` runs foundation model inference; returns uncertainty intervals via 10th/90th percentile quantiles
- `forecast_batch()` processes multiple metrics in one efficient model call
- `evaluate()` uses identical 80/20 train/test split for fair comparison with Prophet

### `ensemble_model.py` — EnsembleForecaster

Two-stage residual learning pipeline:

1. Prophet fits baseline + regressors → compute residuals
2. TimesFM learns patterns in residuals
3. Alpha learned via grid-search on last 10% of training (minimises MAE)
4. `ensemble = prophet + alpha × clamp(timesfm_residual, ±50% of prophet_forecast)`

`evaluate()` returns a three-way dict `{prophet: {...}, timesfm: {...}, ensemble: {...}}` using **identical** test sets for fair comparison. Stores `_test_actuals`, `_test_prophet`, `_test_timesfm`, `_test_ensemble` arrays for chart generation.

`get_residual_stats()` returns `{mean, std, min, max, autocorr_lag1, autocorr_lag24, alpha}` — a diagnostic signal: high `autocorr_lag1` means TimesFM corrections will be most useful.

### `carbon_analysis.py` — Prophet Analysis

Fits Prophet on all metrics, derives SCI, generates 8 charts:

| # | Chart | Scheduling signal it provides |
|---|-------|-------------------------------|
| 01 | `01_service_profile.png` | Energy + request load shape (2 weeks) — confirms daily pattern |
| 02 | `02_energy_forecast.png` | 7-day energy forecast with CI — orchestration input |
| 03 | `03_energy_decomposition.png` | Trend / weekly / daily / yearly breakdown |
| 04 | `04_daily_pattern.png` | Which hours consume most energy — HPA schedule hints |
| 05 | `05_weekly_pattern.png` | Weekday vs weekend load delta — CronJob placement hints |
| 06 | `06_carbon_intensity.png` | Grid intensity by hour + optimal low-carbon window — batch job deferral window |
| 07 | `07_sci_by_hour.png` | SCI (kgCO2e/req) by hour — node scoring weight by time of day |
| 08 | `08_carbon_forecast.png` | 7-day carbon forecast — carbon budget enforcement |

`print_insights()` prints:
- SCI efficiency ratio (best hour vs worst hour) — currently 13:00 is **6.1×** more efficient than 06:00
- Optimal scheduling window (10:00–15:00 → 16% lower carbon intensity)

### `ensemble_analysis.py` — Model Comparison

Runs all three models per metric and generates:

- `07_forecast_comparison.png` — actual vs all 3 models on test week (visual accuracy check)
- `08_model_accuracy.png` — sMAPE grouped bars + ★ winner per metric

Prints improvement-over-Prophet table.

### `visualizations.py` — Plotting Utilities

5 reusable functions accepting DataFrames and optional `save_path` for file export at 150 DPI: `plot_forecast`, `plot_components`, `plot_metric_comparison`, `plot_daily_pattern`, `plot_weekly_pattern`.

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
cost                        EUR/h         Electricity cost (peak/off-peak)
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

---

## Step-by-Step Usage

### Environment Setup

Requires **Python 3.11** (TimesFM constraint).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Generate Data

```bash
.venv/bin/python3 data_generator.py
# → data/sample_energy_data.csv (8,760 rows × 15 columns)
```

### Validate Pipeline Config

```bash
.venv/bin/python3 -c "
from data_loader import load_and_validate, build_pipeline_config
df = load_and_validate('data/sample_energy_data.csv')
cfg = build_pipeline_config(df)
print('metrics:', cfg.metrics)
print('has_sci:', cfg.has_sci)
print('regressor_map:', cfg.regressor_map)
"
# → metrics: ['consumption', 'carbonEmissions', 'softwareCarbonIntensity']
# → has_sci: True
# → regressor_map: {'consumption': ['functionalUnit'], 'carbonEmissions': [], 'functionalUnit': []}
```

### Run Prophet Analysis (8 charts)

```bash
.venv/bin/python3 carbon_analysis.py
# → analysis_output/01_service_profile.png  through  08_carbon_forecast.png
```

### Run Ensemble Comparison

```bash
.venv/bin/python3 ensemble_analysis.py
# → analysis_output/07_forecast_comparison.png  08_model_accuracy.png
# → prints Prophet vs TimesFM vs Ensemble sMAPE table
```

### Python API

```python
from data_loader import load_and_validate, build_pipeline_config
from prophet_model import EnergyProphet
from timesfm_model import EnergyTimesFM
from ensemble_model import EnsembleForecaster

df = load_and_validate("data/sample_energy_data.csv")
config = build_pipeline_config(df)

# --- Prophet (interpretable, orchestration-layer signal) ---
prophet = EnergyProphet(regressors=["functionalUnit"])
prophet.fit(df, "consumption")
forecast_7d = prophet.predict(periods=168, future_df=df)   # 7 days hourly
# forecast_7d: DataFrame [ds, yhat, yhat_lower, yhat_upper, trend, daily, weekly, yearly]

# --- TimesFM (accurate, scheduler-layer signal) ---
tfm = EnergyTimesFM()
tfm.fit(df, "carbonEmissions")
forecast_2h = tfm.predict(periods=2)

# --- Ensemble (three-way evaluation) ---
ens = EnsembleForecaster(regressors=["functionalUnit"])
results = ens.evaluate(df, "consumption")
# results = {
#   "prophet":  {"sMAPE": 5.0, "MAPE": 5.2, "RMSE": 0.0052, ...},
#   "timesfm":  {"sMAPE": 6.4, "MAPE": 6.1, "RMSE": 0.0078, ...},
#   "ensemble": {"sMAPE": 5.0, "MAPE": 5.2, "RMSE": 0.0052, ...},
# }

# SCI is always derived, not fit directly
import numpy as np
ens_c = EnsembleForecaster()
ens_c.evaluate(df, "carbonEmissions")
ens_r = EnsembleForecaster()
ens_r.evaluate(df, "functionalUnit")
sci = ens_c._test_ensemble / np.maximum(ens_r._test_ensemble, 1)   # kgCO2e/req
```

### Swap to Real Data

Replace `data/sample_energy_data.csv` with actual measurements. Minimum required columns:

```
ds            datetime   hourly timestamps
consumption   float      kWh per hour
```

All other columns are optional — the pipeline auto-detects what's available and skips charts/metrics that depend on missing columns. Re-run both scripts; expect better ensemble improvement with real-world complexity.

---

## Results — Three-Way Model Comparison

Evaluation on 1 year of hourly synthetic service data (80/20 train/test split):

### Energy Consumption — `consumption` (with `functionalUnit` regressor)

| Model | sMAPE | MAPE | RMSE |
|-------|-------|------|------|
| Prophet | **5.0%** | 5.2% | 0.0052 |
| TimesFM | 6.4% | 6.1% | 0.0078 |
| **Ensemble** | **5.0%** | **5.2%** | **0.0052** |

Prophet + regressor dominates. Service energy follows clean daily/weekly patterns; `functionalUnit` explains the load scaling precisely. TimesFM adds marginal noise on a well-structured signal. Ensemble matches Prophet.

**Orchestration use:** 7-day Prophet forecast is the primary input for HPA schedule adjustments and batch job deferral windows.

### Carbon Emissions — `carbonEmissions` (no regressor)

| Model | sMAPE | MAPE | RMSE |
|-------|-------|------|------|
| Prophet | 13.9% | 15.1% | 0.0061 |
| **TimesFM** | **7.3%** | **7.0%** | **0.0036** |
| Ensemble | 13.0% | 14.0% | 0.0058 |

TimesFM dominates. Carbon involves the product `consumption × carbonIntensityFactor` — a nonlinear interaction that Prophet's additive/multiplicative decomposition can't capture cleanly. No regressor applied: the path from requests to carbon runs through two intermediate variables, making regressors counterproductive.

**Scheduler use:** 2-hour rolling TimesFM forecast is the best signal for real-time node scoring by carbon cost.

### Software Carbon Intensity — `softwareCarbonIntensity` (derived)

SCI = `carbonEmissions` / `functionalUnit` per hour — forecast via component models, never directly.

| Model | sMAPE | MAPE | RMSE |
|-------|-------|------|------|
| Prophet | 31.6% | 189.5% | 0.00458 |
| **TimesFM** | **15.2%** | **17.0%** | **0.00011** |
| Ensemble | 30.8% | 194.8% | 0.00458 |

MAPE is inflated by near-zero denominators at off-peak hours; **sMAPE is the reliable metric**. TimesFM wins by 2× on sMAPE. **Use SCI hourly maps for node scoring weights, not MAPE-based model selection.**

### Key Takeaways

| Signal needed | Best model | Why |
|--------------|-----------|-----|
| 7-day energy trend for orchestration | Prophet | Interpretable seasonality + regressor |
| 2-hour carbon forecast for scheduler | TimesFM | Nonlinear grid-intensity interactions |
| SCI hourly map (kgCO2e/req by hour) | TimesFM | Best ratio estimation via components |
| Residual autocorrelation diagnostic | `ens.get_residual_stats()` | High lag-1 → TimesFM corrections help |

---

## Kubernetes Integration Roadmap

### Phase 1 — Forecast Signal API (next milestone)
- Wrap `carbon_analysis.py` + `ensemble_analysis.py` in a REST/gRPC service
- Expose endpoints:
  - `GET /forecast/energy?horizon=168` → 7-day kWh/h forecast
  - `GET /forecast/carbon?horizon=2` → 2-hour kgCO2e/h forecast
  - `GET /sci/hourly` → 24-hour SCI map (kgCO2e/req by hour)
  - `GET /schedule/window` → optimal low-carbon scheduling window

### Phase 2 — Upper Layer (Orchestration)
- **Kubernetes Operator** subscribing to forecast API
- Adjusts `HorizontalPodAutoscaler` targets when energy trend is rising
- Patches `CronJob` schedules to align batch work with optimal scheduling window
- Emits `Events` for audit trail (`reason: CarbonAwareReschedule`)

### Phase 3 — Lower Layer (Custom Scheduler Plugin)
- **Scoring plugin** for `kube-scheduler` (framework extension point `Score`)
- Plugin queries the 2-hour rolling forecast per node
- Adds a `CarbonScore` to the default scoring stack:
  ```
  node_score = default_score × (1 − λ × predicted_sci_ratio)
  ```
- `λ` (carbon weight) tunable per namespace via annotation
- Favours nodes predicted to operate in their efficient CPU range (CPU util 60–80%)

### Phase 4 — Feedback Loop
- Actual energy measurements from RAPL/IPMI flow back as training data
- Continuous model re-evaluation (weekly retrain of Prophet, zero-shot TimesFM needs no retraining)
- Drift detection: if sMAPE on rolling 7-day window exceeds threshold → alert + fallback to Prophet-only

---

## Design Principles

1. **Forecast targets are absolute, not ratios** — SCI is always derived post-forecast to avoid numerical instability
2. **Model selection is per-metric** — Prophet for energy (structured), TimesFM for carbon/SCI (nonlinear)
3. **Graceful degradation** — missing columns → charts/metrics skip, not crash
4. **Fair evaluation** — identical 80/20 train/test splits across all three models
5. **Interpretability preserved** — Prophet decomposition charts remain the primary orchestration input
6. **Adaptive residual weighting** — alpha=0 means "trust Prophet only"; system learns when TimesFM helps
7. **Measurement source tracking** — `measurementSource` distinguishes RAPL hardware counters from TDP estimates, enabling data quality signals to the scheduler
