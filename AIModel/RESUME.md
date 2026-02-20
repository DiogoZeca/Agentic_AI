# AI Infrastructure Carbon Forecasting — AI Model Project

## The Goal

Use AI to discover carbon emission patterns in AI infrastructure (inference serving + training runs) and identify actionable opportunities to reduce power and emissions. Predict future consumption, find optimal scheduling windows for training, and detect complex interactions that simple models miss.

---

## How It Works — The Two-Model Approach

This project uses **two complementary AI models** working together:

### Model 1: Prophet (The Explainer)

Prophet is a **statistical decomposition model** by Meta. It breaks any time series into understandable pieces:

```
forecast = trend x daily_cycle x weekly_cycle
```

- **Trend** — Is the baseline going up or down over months?
- **Daily cycle** — Which hours of the day are highest/lowest?
- **Weekly cycle** — How do weekdays compare to weekends?

Prophet's strength is **interpretability**. When it says "emissions peak at 14:00", you can see exactly why — the daily seasonality component shows the curve.

Prophet's weakness is **simplicity**. It can't understand complex interactions between variables (like training spikes × PUE × carbon intensity). These end up as prediction errors.

### Model 2: TimesFM (The Deep Pattern Finder)

TimesFM is a **foundation model** by Google — a 200M parameter neural network pre-trained on billions of real-world time series. Think of it like ChatGPT but for time series: it learned temporal patterns from massive data and can forecast your data **without any training** (zero-shot).

TimesFM's strength is **capturing complex, nonlinear patterns** that Prophet misses.

TimesFM's weakness is that it's a **black box** — predictions without explanations.

### The Residual Approach (Ensemble)

Instead of choosing one model, we use both in a pipeline:

```
Step 1:  Prophet predicts energy consumption    → 1.85 kWh
Step 2:  Actual value was                       → 2.30 kWh  (training spike!)
Step 3:  Residual = actual - predicted          → +0.45  (what Prophet couldn't explain)
Step 4:  TimesFM learns patterns in residuals   → predicts +0.40 for next hour
Step 5:  Final forecast = Prophet + alpha*TimesFM → 1.85 + 1.0*0.40 = 2.25 (closer!)
```

**Residuals** are Prophet's prediction errors. The AI infrastructure data creates **structured residuals** through:
- **Training spikes** — irregular 4-8h bursts of 1400W that Prophet can't predict
- **Batch efficiency** — nonlinear energy/request curves at different load levels
- **PUE × temperature** — seasonal cooling overhead that interacts with all power draws

**Key design choices:**
- **Adaptive alpha** — learned from a validation split (last 10% of training data). Tests alpha 0.0→1.0 step 0.05, picks the value that minimizes MAE. alpha=0 means "ignore TimesFM corrections entirely".
- **Correction clamping** — TimesFM residual corrections are clipped to ±50% of Prophet's prediction, preventing wild swings.
- **Per-metric regressors** — `trainingActive` as a Prophet regressor only for `totalEnergyConsumption` (direct causal link). Omitted for carbon/SCI where the relationship is indirect.

If residuals contain hidden patterns, TimesFM can find and predict them.

---

## Project Structure

```
AIModel/
├── requirements.txt        # Dependencies (Prophet + TimesFM + plotting)
├── data_generator.py       # AI infrastructure synthetic data generator
├── prophet_model.py        # EnergyProphet class (Prophet wrapper)
├── timesfm_model.py        # EnergyTimesFM class (TimesFM wrapper)
├── ensemble_model.py       # EnsembleForecaster (Prophet + TimesFM residuals)
├── visualizations.py       # Reusable plotting functions
├── carbon_analysis.py      # Prophet-only AI analysis (10 charts)
├── ensemble_analysis.py    # Three-way model comparison (6 charts)
├── data/                   # CSV data storage
├── analysis_output/        # All generated charts (16 PNGs)
└── RESUME.md               # This file
```

---

## What Each File Does

### `data_generator.py` — AI Infrastructure Synthetic Data

Generates 1 year of hourly data (8,760 rows) simulating a 4-GPU AI cluster with inference serving and periodic training runs. Four interconnected angles:

| Angle | Metrics | Pattern |
|-------|---------|---------|
| **Inference** | `inferenceRequests`, `avgBatchSize`, `gpuUtilInference`, `inferencePowerDraw`, `energyPerInference` | User demand curve: ~50 req/h at night, ~500/h peak business hours, weekend 40% of weekday. Auto-batching improves efficiency at high load. |
| **Training** | `trainingActive`, `trainingPowerDraw`, `trainingGpuUtil` | 2-3 runs/week, 4-8h each, 70% scheduled at night/weekends. 4 GPUs × 350W = 1400W spikes. |
| **Infrastructure** | `totalGpuPower`, `cpuAndMemoryPower`, `pue`, `totalInfrastructurePower`, `totalEnergyConsumption` | PUE base 1.2 + summer seasonal +0.15. CPU/memory 15% of GPU + 50W base. |
| **Carbon & SCI** | `carbonIntensityFactor`, `greenConsumptionPercentage`, `operationalEmissions`, `embodiedEmissions`, `carbonEmissions`, `sciPerInference` | Grid carbon intensity lower midday (solar). SCI = gCO2e per inference request. |

### `prophet_model.py` — EnergyProphet Class

Wraps Prophet into a clean workflow: `fit()` → `predict()` → `evaluate()`.
Uses **multiplicative** seasonality — energy patterns scale with load, not additive.

### `timesfm_model.py` — EnergyTimesFM Class

Wraps Google's TimesFM with the same interface: `fit()` → `predict()` → `evaluate()`.
- `fit()` stores data (no training — TimesFM works zero-shot)
- `predict()` runs inference using pre-trained weights
- `evaluate()` uses identical 80/20 train/test split for fair comparison with Prophet
- `forecast_batch()` processes multiple metrics in a single efficient call
- Supports uncertainty intervals via quantile forecasts (10th/90th percentile)

### `ensemble_model.py` — EnsembleForecaster

The two-stage pipeline:
1. Prophet fits and predicts (with optional regressors) → compute residuals
2. TimesFM forecasts residual patterns
3. Learn optimal alpha from validation split (last 10% of training)
4. Final = Prophet + alpha × clamp(TimesFM correction, ±50%)

Provides `evaluate()` that returns a three-way comparison (Prophet vs TimesFM vs Ensemble) using identical test sets. Also includes `get_residual_stats()` to analyze residual structure (autocorrelation, alpha weight).

### `visualizations.py` — Plotting Utilities

5 reusable functions: `plot_forecast()`, `plot_components()`, `plot_metric_comparison()`, `plot_daily_pattern()`, `plot_weekly_pattern()`.

### `carbon_analysis.py` — Prophet Analysis (Charts 1-10)

| # | Chart | What it tells you |
|---|-------|------------------|
| 1 | AI power overview | 3 panels: inference power, training spikes, total infrastructure (first 2 weeks) |
| 2 | Total energy forecast | Prophet forecast + uncertainty bands for kWh consumption |
| 3 | Energy decomposition | Trend, daily, weekly breakdown of energy consumption |
| 4 | Daily power pattern | Stacked bar: inference + training power by hour |
| 5 | Weekly power pattern | Avg energy consumption by day of week |
| 6 | Intensity vs green energy | **Most actionable** — optimal scheduling window shaded |
| 7 | Training spikes timeline | Training power bursts over first month with total infrastructure overlay |
| 8 | SCI per inference | gCO2e per inference request by hour (color-coded) — shows when inference is "greenest" |
| 9 | Load vs efficiency | Scatter: inference requests vs energy/inference — reveals batch efficiency curve |
| 10 | Carbon emissions forecast | Prophet forecast for total carbon emissions |

### `ensemble_analysis.py` — Model Comparison (Charts 11-16)

| # | Chart | What it tells you |
|---|-------|------------------|
| 11 | Residual time series | Do Prophet's errors have visible patterns (training spikes) or look random? |
| 12 | Residual autocorrelation | Statistically: are consecutive errors correlated? Bars above the red line = significant |
| 13 | Residual distribution | Are errors symmetric (unbiased) or skewed? |
| 14 | Model comparison | All 3 models overlaid on test data — visual accuracy check |
| 15 | Error distributions | Histogram of absolute errors per model — which has tighter errors? |
| 16 | MAPE comparison | Grouped bars — MAPE for each model across all 3 AI metrics |

---

## Data Schema

```
Column                      Unit          Description
──────────────────────────────────────────────────────────────────
ds                          datetime      Timestamp

── Inference ──
inferenceRequests           count/h       User demand for AI inference
avgBatchSize                requests      Auto-batching size (higher at peak load)
gpuUtilInference            %             GPU utilization for inference (capped ~85%)
inferencePowerDraw          W             Total inference GPU power (4 GPUs)
energyPerInference          Wh            Power / requests — drops at high load

── Training ──
trainingActive              0/1           Training run in progress
trainingPowerDraw           W             4 GPUs × 350W when active, 0 otherwise
trainingGpuUtil             %             0% or 85-95% during training

── Infrastructure ──
totalGpuPower               W             Inference + training GPU power
cpuAndMemoryPower           W             15% of GPU + 50W base
pue                         ratio         Power Usage Effectiveness (1.2 base + seasonal)
totalInfrastructurePower    W             (GPU + CPU) × PUE
totalEnergyConsumption      kWh           Infrastructure power / 1000

── Carbon & SCI ──
carbonIntensityFactor       kgCO2/kWh     Grid carbon intensity (lower midday)
greenConsumptionPercentage  %             Renewable energy percentage
operationalEmissions        kgCO2e        Energy × carbon intensity
embodiedEmissions           kgCO2e        Hardware amortization (constant)
carbonEmissions             kgCO2e        Operational + embodied
sciPerInference             gCO2e         Carbon per inference request × 1000

── Legacy (derived from above) ──
consumption                 kWh           = totalEnergyConsumption
method                      str           Energy measurement method (RAPL/TDP)
cost                        EUR           Consumption × peak/off-peak pricing
totalCost                   EUR           Cumulative electricity cost
softwareCarbonIntensity     gCO2e         = sciPerInference (SCI metric)
functionalUnitCount         count         = inferenceRequests
```

---

## Step-by-Step Usage Guide

### Step 1 — Set up the environment

**Important:** Requires **Python 3.11** (TimesFM doesn't support 3.12+).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Step 2 — Generate synthetic data

```bash
python data_generator.py
```

Creates `data/sample_energy_data.csv` with 1 year of hourly AI infrastructure data. You'll see training hours, total inference requests, and average energy per inference.

### Step 3 — Run Prophet-only analysis

```bash
python carbon_analysis.py
```

Fits Prophet on energy, carbon, and inference requests. SCI is derived from component forecasts (carbon / requests). Generates charts 1-10, prints AI workload insights (optimal training windows, SCI by hour, batch efficiency).

### Step 4 — Run the ensemble comparison

```bash
python ensemble_analysis.py
```

Runs the three-way evaluation (Prophet vs TimesFM vs Ensemble), generates charts 11-16, prints MAPE comparison table and residual analysis.

### Step 5 — Review outputs

All 16 charts are saved in `analysis_output/`:
- Charts 01-10: AI infrastructure analysis and carbon patterns
- Charts 11-16: Model comparison and residual diagnostics

### Step 6 — Swap to real data

Replace `data/sample_energy_data.csv` with your actual data. Requirements:
- `ds` column with datetime values
- Metric columns (`totalEnergyConsumption`, `carbonEmissions`, `sciPerInference`, etc.)
- Hourly frequency works best

Re-run both scripts to see how the models perform on real-world complexity.

### Using the API

```python
# Prophet with regressor (interpretable forecasting)
from prophet_model import EnergyProphet
model = EnergyProphet(regressors=["trainingActive"])
model.fit(df, "totalEnergyConsumption")
forecast = model.predict(periods=168, future_df=df)  # pass df for regressor values

# TimesFM (zero-shot foundation model)
from timesfm_model import EnergyTimesFM
model = EnergyTimesFM()
model.fit(df, "totalEnergyConsumption")
forecast = model.predict(periods=168)

# Ensemble (three-way comparison with regressor)
from ensemble_model import EnsembleForecaster
ensemble = EnsembleForecaster(regressors=["trainingActive"])
results = ensemble.evaluate(df, "totalEnergyConsumption")
# results = {"prophet": {...}, "timesfm": {...}, "ensemble": {...}}
# Each dict has: MAE, MSE, RMSE, MAPE, sMAPE, train_size, test_size
```

---

## Results — Three-Way Model Comparison

Evaluation on 1 year of hourly synthetic AI infrastructure data (80/20 train/test split):

### Total Energy Consumption (with `trainingActive` regressor)

| Model | MAE | RMSE | MAPE | sMAPE |
|-------|-----|------|------|-------|
| Prophet | 0.1513 | 0.1840 | 15.10% | 14.09% |
| TimesFM | 0.2337 | 0.5451 | 13.90% | 15.69% |
| **Ensemble** | **0.1466** | **0.1803** | 14.53% | **13.64%** |

Ensemble wins by sMAPE and has the lowest MAE/RMSE. The `trainingActive` regressor lets Prophet explain training power spikes, and TimesFM captures remaining patterns (residual autocorrelation: 0.577, alpha: 1.00).

### Carbon Emissions (no regressor)

| Model | MAE | RMSE | MAPE | sMAPE |
|-------|-----|------|------|-------|
| Prophet | 0.1018 | 0.1884 | 22.22% | 22.42% |
| **TimesFM** | **0.0897** | **0.1872** | **17.75%** | **18.78%** |
| Ensemble | 0.0968 | 0.2023 | 18.40% | 22.19% |

TimesFM dominates. Carbon emissions involve a complex interaction (energy × grid carbon intensity) where TimesFM's zero-shot pattern recognition outperforms the decomposition approach. No regressor applied — `trainingActive` has no direct causal link to carbon intensity.

### SCI Per Inference (derived from components)

SCI is forecast by predicting `carbonEmissions` and `inferenceRequests` separately, then computing `SCI = carbon / requests * 1000`. This avoids Prophet's breakdown on ratio metrics.

| Model | MAE | RMSE | MAPE | sMAPE |
|-------|-----|------|------|-------|
| Prophet | 24.54 | 87.98 | 332.06% | 46.08% |
| **TimesFM** | **1.20** | **2.52** | **32.60%** | **28.57%** |
| Ensemble | 18.90 | 66.34 | 263.68% | 52.10% |

TimesFM dominates. SCI is a ratio metric (gCO2e/request) with extreme variance (0.15 to 56.6). MAPE is inflated by near-zero denominators; **sMAPE is the primary accuracy metric** for SCI. The component-based approach improved sMAPE across all models vs direct forecasting (Prophet: 83% → 46%, TimesFM: 35% → **28.6%**, Ensemble: 78% → 52%).

### Key Takeaways

- **Ensemble proves its value for energy** — where Prophet has a structural advantage (regressor knowledge) and TimesFM fills in the remaining gaps
- **TimesFM standalone excels for complex, nonlinear metrics** (carbon, SCI) where Prophet's decomposition approach is too rigid
- **Component-based ratio forecasting** — forecasting numerator and denominator separately then combining dramatically improves SCI predictions
- **sMAPE is essential** alongside MAPE for metrics with extreme values (SCI)
- **Per-metric regressor selection matters** — applying `trainingActive` to carbon actually made predictions worse (28.93% vs 22.22%) because the causal link is indirect

---

## Next Steps

- **Real data** — Replace synthetic CSV with actual infrastructure measurements; the ensemble should improve significantly with complex real-world patterns
- **Fine-tuning TimesFM** — Adapt the foundation model specifically to energy/carbon data
- **More regressors** — Feed weather data, grid carbon forecasts, or production schedules as Prophet covariates for carbon/SCI metrics
- **Anomaly detection** — Flag data points where Prophet and TimesFM disagree strongly
- **Agentic AI orchestration** — Evolve into a carbon-aware agent that automatically schedules training runs, adjusts batch sizes, and selects green energy windows (see `../AgenticAI.md`)
