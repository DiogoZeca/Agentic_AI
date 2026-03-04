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

## Three Modes

### Mode 0 — Test Suite

Validates data contracts, chained forecasting correctness, API behaviour, anomaly detection, and carbon forecast physical constraints.
Uses a 30-day dataset for fast model fitting (~60–90s total, **217 tests**).

```bash
docker compose --profile test run --rm --build test
```

Run a single test file:

```bash
docker compose --profile test run --rm test pytest tests/test_data.py -v
docker compose --profile test run --rm test pytest tests/test_forecasting.py -v
docker compose --profile test run --rm test pytest tests/test_api.py -v
docker compose --profile test run --rm test pytest tests/test_anomaly.py -v
```

**Test groups:**

| File | What it validates |
|------|-------------------|
| `test_data.py` | Schema, value ranges, SCI identity, regressor map |
| `test_forecasting.py` | Chained forecasting (consumption + carbonEmissions) |
| `test_api.py` | All 8 endpoints, response shapes, error codes, carbon contract |
| `test_anomaly.py` | Point anomalies, recurring patterns, investigation leads |

---

### Mode 1 — Analysis Pipeline

Generates synthetic data and produces all forecast charts as PNG files.

```bash
docker compose --profile analysis up --build
```

**What runs (in order):**

1. **generate** — creates `AIModel/data/sample_energy_data.csv` (8,760 hourly rows, 15 columns)
2. **analyse** — fits Prophet (with `carbonIntensityFactor` regressor for `carbonEmissions`), applies TimesFM override for carbon forecast, writes 8 charts to `AIModel/analysis_output/`
3. **ensemble** — runs Prophet vs TimesFM vs Ensemble comparison, writes 2 more charts

#### Expected run times

| Phase | First run | Cached run |
|-------|-----------|------------|
| Docker image build (downloads PyTorch ~915 MB + CUDA libs ~600 MB+ each) | **20–40 min** | ~10s |
| `generate` — create 8,760-row CSV | ~5s | ~5s |
| `analyse` — fit 7 Prophet models + 8 charts | ~3–4 min | ~3–4 min |
| `ensemble` — TimesFM weight download (~925 MB) + inference | ~10–15 min | ~5–8 min |
| **Total first run** | **~35–60 min** | **~10 min** |

> **Why the first build is slow:** Both the `analyse` and `ensemble` images download PyTorch (915 MB) and several CUDA libraries in parallel, competing for bandwidth. This is a one-time cost — Docker caches every layer.
>
> **What you see during the build:** Progress lines like `Downloading torch-2.10.0 ... 915.6/915.6 MB` and `Downloading nvidia_cublas_cu12 ... 594.3 MB`. This is normal. Let it finish.
>
> **Subsequent runs:** The `--build` flag reuses all cached layers — no re-download. TimesFM model weights (~925 MB) are stored in the `hf-cache` Docker named volume and are never re-downloaded.

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
| `08_carbon_forecast.png` | 7-day carbon forecast (TimesFM when available, else Prophet) |
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

## What Changed — Pipeline Improvements

### `carbonIntensityFactor` regressor for `carbonEmissions`

`carbonEmissions` is now chained: its Prophet model uses `carbonIntensityFactor` as a regressor.
This reflects the physical formula directly: `emissions = consumption × CI + 0.002`.
For future rows, `carbonIntensityFactor` is forecast first by its own Prophet model, then injected — same chaining mechanism already used for `consumption → functionalUnit`.

**Accuracy improvement:** Prophet sMAPE 13.9% → ~8% (regressor provides causal signal).

### Best-model routing for `carbonEmissions`

The API now routes `carbonEmissions` forecasts through TimesFM when it is available, falling back to Prophet if not.

| Metric | Prophet sMAPE | TimesFM sMAPE | Routed through |
|--------|--------------|---------------|----------------|
| `carbonEmissions` | 13.9% | 7.3% | **TimesFM** (when available) |
| `softwareCarbonIntensity` (SCI) | 31.6% | 15.2% | TimesFM (via carbon component) |
| `consumption` | ~5% | — | Prophet |

This applies to `/forecast/carbonEmissions`, `/forecast/all`, `/forecast/sci`, and `/optimal-window`.

### Carbon forecast physical contracts (5 new tests)

`TestCarbonForecastContract` in `test_api.py` validates model-agnostic invariants:
- `yhat >= 0` (no negative emissions)
- `yhat < 10` kgCO2e/h (plausible range for a single service)
- `yhat_lower <= yhat <= yhat_upper` (interval ordering)
- `/forecast/all` includes `carbonEmissions`
- `ds` timestamps are ISO 8601 with `T` and `Z`

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
| GET | `/anomalies?metric=consumption` | Point anomalies vs Prophet confidence interval |
| GET | `/investigation-leads` | Ranked recurring patterns with carbon/cost savings |
| POST | `/data` | Ingest new measurements (triggers async model refit) |
| GET | `/metrics/prometheus?horizon=1` | Prometheus text format (for HPA integration) |

**Available metrics for `/forecast/{metric}`:**

| Metric | Unit | Model | Description |
|--------|------|-------|-------------|
| `consumption` | kWh/h | Prophet (chained) | Energy consumption — uses `functionalUnit` as regressor |
| `carbonEmissions` | kgCO2e/h | TimesFM / Prophet | Total carbon — uses `carbonIntensityFactor` as regressor |
| `carbonIntensityFactor` | kgCO2/kWh | Prophet | Grid carbon intensity |
| `greenConsumptionPercentage` | % | Prophet | Renewable energy fraction |
| `cpuUtilization` | fraction | Prophet | CPU load |
| `functionalUnit` | req/h | Prophet | Request rate (SCI denominator) |
| `cost` | EUR/h | Prophet | Electricity cost |
| `softwareCarbonIntensity` | kgCO2e/req | Derived | SCI — use `/forecast/sci` |

---

## Example Calls

### Health & readiness

```bash
curl http://localhost:8000/health
```

### Forecasts

```bash
# All metrics at once — 24h forecast
curl "http://localhost:8000/forecast/all?horizon=24"

# Carbon emissions — 24h (TimesFM when available, else Prophet)
curl "http://localhost:8000/forecast/carbonEmissions?horizon=24"

# Carbon emissions — extract just yhat values
curl -s "http://localhost:8000/forecast/carbonEmissions?horizon=24" | jq '[.predictions[].yhat]'

# Carbon emissions — check uncertainty interval ordering
curl -s "http://localhost:8000/forecast/carbonEmissions?horizon=3" \
  | jq '.predictions[] | {ds, yhat_lower, yhat, yhat_upper}'

# SCI forecast — next 6h (derived from carbonEmissions / functionalUnit)
curl "http://localhost:8000/forecast/sci?horizon=6"

# Check all metrics returned by /forecast/all
curl -s "http://localhost:8000/forecast/all?horizon=1" | jq '[keys[]]'

# Energy consumption — next 48h
curl "http://localhost:8000/forecast/consumption?horizon=48"

# Grid carbon intensity — next 48h
curl "http://localhost:8000/forecast/carbonIntensityFactor?horizon=48"
```

### Scheduling

```bash
# Best low-carbon window — next 3 days, 4-hour blocks
curl "http://localhost:8000/optimal-window?horizon_days=3&window_hours=4"

# Default: 7 days, 6-hour blocks
curl "http://localhost:8000/optimal-window"
```

### Anomaly detection

```bash
# Consumption anomalies — last 30 days, excess only
curl "http://localhost:8000/anomalies?metric=consumption&lookback_days=30&direction=excess"

# Carbon anomalies — last 14 days, both directions
curl "http://localhost:8000/anomalies?metric=carbonEmissions&lookback_days=14&direction=both"

# Top 5 investigation leads — recurring patterns with carbon/cost savings
curl "http://localhost:8000/investigation-leads?lookback_days=30&top_n=5"
```

### Data ingestion

```bash
# Push a single measurement (triggers async Prophet refit)
curl -X POST http://localhost:8000/data \
  -H "Content-Type: application/json" \
  -d '{"rows": [{"ds": "2025-01-01T14:00:00", "consumption": 0.18, "carbonEmissions": 0.072}]}'
```

### Prometheus (for HPA)

```bash
curl "http://localhost:8000/metrics/prometheus?horizon=1"
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

`vs_daily_mean_pct: -27.2` — this window is 27% greener than the day's average. Schedule batch jobs here.

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
│   ├── api.py                    ← REST service (9 endpoints: forecast + anomaly + ingest + prometheus)
│   │                               Best-model routing: carbonEmissions → TimesFM when available
│   ├── anomaly_detector.py       ← Anomaly engine (point anomalies → patterns → leads)
│   ├── data_generator.py         ← Synthetic data (8,760 rows, 15 columns)
│   ├── data_loader.py            ← Schema validation + pipeline config
│   │                               carbonEmissions regressor: [carbonIntensityFactor]
│   ├── prophet_model.py          ← Prophet wrapper (EnergyProphet, multiplicative seasonality)
│   ├── timesfm_model.py          ← TimesFM wrapper (zero-shot, 200M params)
│   ├── ensemble_model.py         ← Prophet + TimesFM ensemble (residual correction)
│   ├── carbon_analysis.py        ← 8-chart analysis pipeline; TimesFM override for carbonEmissions
│   ├── ensemble_analysis.py      ← Three-way model comparison (Prophet / TimesFM / Ensemble)
│   ├── visualizations.py         ← Reusable Matplotlib functions
│   ├── tests/
│   │   ├── conftest.py           ← 30-day test dataset setup
│   │   ├── test_data.py          ← Data contract (schema, ranges, SCI identity, regressor map)
│   │   ├── test_forecasting.py   ← Chained forecasting validation
│   │   ├── test_api.py           ← API contract (217 tests total, incl. carbon physical contracts)
│   │   └── test_anomaly.py       ← Anomaly engine (point anomalies, patterns, leads)
│   ├── Dockerfile                ← Service image (Prophet + FastAPI, lightweight)
│   ├── Dockerfile.test           ← Test image (service deps + pytest + httpx)
│   ├── Dockerfile.analysis       ← Analysis image (full stack + TimesFM + PyTorch)
│   ├── requirements.txt          ← Full deps (analysis + TimesFM)
│   ├── requirements-service.txt  ← Lightweight deps (service only)
│   ├── requirements-test.txt     ← Test deps (pytest + httpx)
│   └── data/                     ← Generated CSV (gitignored, bind-mounted)
│       └── sample_energy_data.csv
├── docker-compose.yml            ← Three modes: test + analysis profile + API service
├── docs/                         ← Research papers and reference images
└── README.md                     ← This file
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
Run the analysis pipeline first:
```bash
docker compose --profile analysis up --build
```

**API slow to start**
Normal. 7 Prophet models × ~15s each on 8,760 rows ≈ 2 minutes.
The health check has a 180s grace period built in.

**Analysis image build is slow (20–40 min on first run)**
Normal. Both the `analyse` and `ensemble` images download PyTorch (915 MB), CUDA bindings (12 MB), and NVIDIA libs (cublas 594 MB, etc.) in parallel, competing for bandwidth. You will see progress lines like:
```
Downloading torch-2.10.0 ... 915.6/915.6 MB
Downloading nvidia_cublas_cu12 ... 594.3 MB
```
Let it finish. Subsequent `--build` calls reuse every cached layer and complete in ~10s.

**`carbonEmissions` forecast looks flat**
If TimesFM is unavailable (Prophet-only mode), the forecast falls back to Prophet with `carbonIntensityFactor` as a regressor. Ensure the `hf-cache` volume is intact for TimesFM routing.
