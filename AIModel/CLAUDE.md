# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Prophet + TimesFM ensemble forecasting for AI infrastructure power consumption and carbon emissions. Simulates a 4-GPU cluster with inference serving and periodic training runs. Uses Meta's Prophet wrapped in `EnergyProphet` and Google's TimesFM in `EnergyTimesFM`, combined via `EnsembleForecaster`. Currently operates with synthetic data; designed to be swapped to real data when available.

## Commands

```bash
# Install dependencies
python -m pip install -r requirements.txt

# Generate synthetic data (writes to data/sample_energy_data.csv)
python data_generator.py

# Run Prophet-only carbon analysis (10 charts)
python carbon_analysis.py

# Run ensemble comparison: Prophet vs TimesFM vs Ensemble (6 charts)
python ensemble_analysis.py
```

There is no test suite, linter, or build system configured.

## Architecture

Six-module pipeline with clear separation of concerns:

- **data_generator.py** — Generates synthetic hourly AI infrastructure data with 4 interconnected angles: inference demand (user load curves, auto-batching), training runs (irregular GPU-intensive spikes), total infrastructure (PUE, cooling), and carbon/SCI metrics. All data uses Prophet's required `ds` column for timestamps.

- **prophet_model.py** — `EnergyProphet` class wrapping Prophet. Key design: uses **multiplicative** seasonality (energy patterns scale with load, not additive). Provides `fit()` → `predict()` → `evaluate()` workflow. `quick_forecast()` is a convenience function for end-to-end CSV-to-forecast. Evaluation uses train/test split with MAE, MSE, RMSE, MAPE metrics.

- **timesfm_model.py** — `EnergyTimesFM` class wrapping Google's TimesFM foundation model. Zero-shot forecasting with same interface as EnergyProphet.

- **ensemble_model.py** — `EnsembleForecaster` combining Prophet + TimesFM on residuals. Prophet captures trend/seasonality, TimesFM captures structured residuals (training spikes, nonlinear interactions).

- **carbon_analysis.py** — Prophet-only analysis generating 10 charts and AI workload insights.

- **ensemble_analysis.py** — Three-way model comparison generating 6 charts.

- **visualizations.py** — Matplotlib-based plotting utilities. Functions accept trained Prophet models/DataFrames and optional `save_path` for file export.

## Data Schema

Prophet requires a `ds` (datetime) column. The target column (e.g., `totalEnergyConsumption`, `carbonEmissions`) is renamed to `y` internally by `prepare_data()`. Key metric groups:

- **Inference**: `inferenceRequests`, `avgBatchSize`, `gpuUtilInference`, `inferencePowerDraw`, `energyPerInference`
- **Training**: `trainingActive`, `trainingPowerDraw`, `trainingGpuUtil`
- **Infrastructure**: `totalGpuPower`, `cpuAndMemoryPower`, `pue`, `totalInfrastructurePower`, `totalEnergyConsumption`
- **Carbon & SCI**: `carbonIntensityFactor`, `greenConsumptionPercentage`, `operationalEmissions`, `embodiedEmissions`, `carbonEmissions`, `sciPerInference`
- **Legacy (derived)**: `consumption` (=totalEnergyConsumption), `cost`, `totalCost`, `method`, `softwareCarbonIntensity` (=sciPerInference), `functionalUnitCount` (=inferenceRequests)

## Typical Usage Pattern

```python
from data_generator import generate_energy_carbon_data
from prophet_model import EnergyProphet

df = generate_energy_carbon_data()
model = EnergyProphet()
model.fit(df, target_column="totalEnergyConsumption")
forecast = model.predict(periods=168)  # 1 week of hourly predictions
metrics = model.evaluate(df, "totalEnergyConsumption")
```
