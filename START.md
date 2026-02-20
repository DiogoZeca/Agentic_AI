# Getting Started

Energy & Carbon Forecasting System — Docker only.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Docker Engine | ≥ 25 |
| Docker Compose | ≥ 2.22 |

No Python. No pip. No virtual environments.

---

## Two Modes

### Mode 1 — Analysis Pipeline

Generates synthetic data and produces all forecast charts as PNG files.

```bash
docker compose --profile analysis up --build
```

**What runs (in order):**

1. **generate** — creates `AIModel/data/sample_energy_data.csv` (8,760 hourly rows, 15 columns)
2. **analyse** — fits Prophet on all metrics, writes 8 charts to `AIModel/analysis_output/`
3. **ensemble** — runs Prophet vs TimesFM vs Ensemble comparison, writes 2 more charts

> **First run:** downloads TimesFM model weights (~925 MB) into a Docker named volume.
> **Subsequent runs:** weights are cached — starts in seconds.

Charts appear in `AIModel/analysis_output/` on your machine:

| File | Content |
|------|---------|
| `01_service_profile.png` | Energy + request load (first 2 weeks) |
| `02_energy_forecast.png` | 7-day Prophet forecast with confidence bands |
| `03_energy_decomposition.png` | Trend + weekly + daily seasonality |
| `04_daily_pattern.png` | Average energy by hour of day |
| `05_weekly_pattern.png` | Average energy by day of week |
| `06_carbon_intensity.png` | Grid intensity + green % + optimal window |
| `07_sci_by_hour.png` | SCI (kgCO2e/req) by hour |
| `08_carbon_forecast.png` | 7-day carbon emissions forecast |
| `07_forecast_comparison.png` | Prophet vs TimesFM vs Ensemble |
| `08_model_accuracy.png` | sMAPE accuracy comparison |

---

### Mode 2 — Forecasting API

Starts a long-running REST service. Fits 7 Prophet models on startup (~2–3 min).

```bash
docker compose up --build
```

Service is ready when you see:

```
INFO  Service ready — 8760 rows, models: ['consumption', 'carbonEmissions', ...]
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## API Reference

Interactive docs (Swagger UI): **http://localhost:8000/docs**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Model status and readiness |
| GET | `/forecast/all?horizon=24` | All metrics in one response |
| GET | `/forecast/{metric}?horizon=24` | Single metric forecast |
| GET | `/forecast/sci?horizon=24` | SCI — derived, kgCO2e/req |
| GET | `/optimal-window?horizon_days=7` | Best low-carbon scheduling windows |

**Available metrics for `/forecast/{metric}`:**

| Metric | Unit | Description |
|--------|------|-------------|
| `consumption` | kWh/h | Energy consumption |
| `carbonEmissions` | kgCO2e/h | Total carbon emissions |
| `carbonIntensityFactor` | kgCO2/kWh | Grid carbon intensity |
| `greenConsumptionPercentage` | % | Renewable energy fraction |
| `cpuUtilization` | fraction | CPU load |
| `functionalUnit` | req/h | Request rate (SCI denominator) |
| `cost` | EUR/h | Electricity cost |
| `softwareCarbonIntensity` | kgCO2e/req | SCI — use `/forecast/sci` |

### Example Calls

```bash
# Health check
curl http://localhost:8000/health

# All metrics at once — 24h forecast
curl "http://localhost:8000/forecast/all?horizon=24"

# Carbon intensity forecast — next 48h
curl "http://localhost:8000/forecast/carbonIntensityFactor?horizon=48"

# SCI forecast — next 6h
curl "http://localhost:8000/forecast/sci?horizon=6"

# Best scheduling window for the next 3 days (4-hour blocks)
curl "http://localhost:8000/optimal-window?horizon_days=3&window_hours=4"
```

### Example Response — `/optimal-window`

```json
{
  "horizon_days": 3,
  "window_hours": 4,
  "windows": [
    {
      "date": "2024-12-31",
      "start_hour": 2,
      "end_hour": 6,
      "avg_predicted_carbon": 0.030381,
      "vs_daily_mean_pct": -27.2,
      "unit": "kgCO2e/h"
    }
  ]
}
```

`vs_daily_mean_pct: -27.2` — this window is 27% greener than the day's average.
Schedule batch jobs here.

---

## Stop

```bash
docker compose down                        # stop the API
docker compose --profile analysis down     # clean up after analysis
```

---

## Project Structure

```
Agentic_AI/
├── AIModel/
│   ├── api.py                    ← Forecasting REST service (FastAPI)
│   ├── data_generator.py         ← Synthetic data (8,760 rows, 15 columns)
│   ├── data_loader.py            ← Schema validation + pipeline config
│   ├── prophet_model.py          ← Prophet wrapper (EnergyProphet)
│   ├── timesfm_model.py          ← TimesFM wrapper (zero-shot)
│   ├── ensemble_model.py         ← Prophet + TimesFM ensemble
│   ├── carbon_analysis.py        ← 8-chart Prophet analysis pipeline
│   ├── ensemble_analysis.py      ← Three-way model comparison
│   ├── visualizations.py         ← Reusable Matplotlib functions
│   ├── Dockerfile                ← Service image (Prophet + FastAPI, lightweight)
│   ├── Dockerfile.analysis       ← Analysis image (full stack + TimesFM)
│   ├── requirements.txt          ← Full deps (analysis + TimesFM)
│   ├── requirements-service.txt  ← Lightweight deps (service only)
│   └── data/                     ← Generated CSV (gitignored, bind-mounted)
│       └── sample_energy_data.csv
├── docker-compose.yml            ← Two modes: analysis profile + API service
├── docs/                         ← Research papers and reference images
└── START.md                      ← This file
```

---

## Troubleshooting

**`docker compose` not found**
Use `docker-compose` (with a hyphen) for older Docker Desktop installations.

**Port 8000 already in use**
Edit `docker-compose.yml` and change `"8000:8000"` to `"8001:8000"`, then access on port 8001.

**TimesFM download hangs**
The model weights are ~925 MB. Ensure a stable connection.
They cache in the `hf-cache` Docker named volume after first download.

**Charts not appearing**
Run the analysis pipeline before the API:
```bash
docker compose --profile analysis up --build
```

**API slow to start**
Normal. 7 Prophet models × ~15s each on 8,760 rows = ~2 minutes.
The health check has a 180s grace period built in.

**Analysis image build is slow**
Normal — PyTorch (~2 GB) is installed once. Subsequent builds use the layer cache.
