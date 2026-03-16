# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

**CPU power spike prediction for EVIDEN.** Given a CPU type and utilization percentage, the system predicts power consumption (Watts) and flags whether it constitutes a spike.

EVIDEN's requirement: *"enable anyone to send forecasts to the observability framework about any of the existing metrics (performance & eco-efficiency). Then we can enable scheduling policies based on the future state of the nodes instead of what is happening right now."*

**Current scope:** CPU% → Power(W) spike prediction, trained on `cpu_data.dat` (real SPECpower-style measurements, ~15M samples across 11 CPU types).
**Next steps (not yet built):** time-series CPU% forecasting, carbon emissions pipeline.

## Directory Layout

```
Agentic_AI/
├── CLAUDE.md              ← this file
├── session-notes.md       ← session history and decision log
├── docker-compose.yml     ← compose file (run from HERE, not from AIModel/)
└── AIModel/               ← all application code
    ├── Dockerfile          ← API image (python:3.12-slim, sklearn + FastAPI)
    ├── Dockerfile.test     ← test image (same deps + pytest + httpx)
    ├── requirements-service.txt
    ├── requirements-test.txt
    ├── data/
    │   └── cpu_data.dat   ← real CPU power measurements (bind-mounted)
    └── tests/
        ├── conftest.py
        └── test_api.py
```

**Local Python commands**: run from `AIModel/`.
**Docker Compose commands**: run from the project root (`Agentic_AI/`).

## Commands

```bash
cd AIModel/

# Validate the data file loads correctly
.venv/bin/python3 -c "from data_loader import load_cpu_power_data; df = load_cpu_power_data('data/cpu_data.dat'); print(df.shape, df['CPUTYPE'].unique())"

# Quick smoke test of the model
.venv/bin/python3 -c "
from data_loader import load_cpu_power_data
from cpu_power_model import CpuPowerModel
model = CpuPowerModel().fit(load_cpu_power_data('data/cpu_data.dat'))
print(model.predict(85, 'intel-xeon-e5420'))
"

# Run tests locally
.venv/bin/python3 -m pytest tests/ -v
```

Run the test suite (Docker — run from `Agentic_AI/`):

```bash
docker compose --profile test run --rm --build test
```

Run the API service:

```bash
docker compose up --build
curl http://localhost:8000/health
```

## Architecture

Three-module pipeline:

- **data_loader.py** — `load_cpu_power_data(filepath)` reads `cpu_data.dat`, validates schema and value ranges, returns a typed DataFrame.

- **cpu_power_model.py** — `CpuPowerModel`: fits a degree-2 polynomial regression (sklearn Pipeline) per CPU type, weighted by `NPTS` (sample count). Recovers per-bucket std dev from `SUM2`. Spike threshold = idle + 75% of dynamic range. Falls back to `unknown` type for unrecognised CPUs.

- **api.py** — FastAPI service. Loads `cpu_data.dat` on startup via lifespan context. 4 endpoints (see below).

## Data Schema (`cpu_data.dat`)

| Column | Type | Description |
|--------|------|-------------|
| `CPUTYPE` | str | CPU model name (11 types incl. `unknown`) |
| `CPUPCT` | int | CPU utilization bucket, 0–100 |
| `NPTS` | int | Number of real measurements in this bucket |
| `SUM` | float | Sum of power readings (W) |
| `SUM2` | float | Sum of squared power readings (used to compute std dev) |
| `AVGPOWER` | float | Mean power at this CPU% (W) |

1,002 rows, ~15M total measurements. Power range: ~114W idle → ~370W full load.

## Key Design Constraints

- **Spike threshold** = `idle_w + 0.75 × (full_w - idle_w)` — top 25% of a CPU's dynamic range
- **Uncertainty bounds** = mean ± 2× weighted std dev (95% CI approximation)
- **Unknown CPU fallback** — if `cpu_type` not in dataset, silently uses `unknown` model
- **22-test suite** — `test_api.py`: health, predict, batch, models endpoints + spike flag logic + monotonicity
- **No time-series yet** — `cpu_data.dat` is aggregate data (no timestamps). Time-series forecasting will be added once EVIDEN provides real monitoring data.

## API Endpoints (4 total)

| Method | Path | Params | Description |
|--------|------|--------|-------------|
| GET | `/health` | — | Status + list of loaded CPU types |
| GET | `/predict` | `cpu_pct` (0–100, required), `cpu_type` (default `unknown`) | Single power prediction + spike flag |
| POST | `/predict/batch` | JSON body: `{predictions: [{cpu_pct, cpu_type}]}` | Batch predictions |
| GET | `/models` | — | All CPU types with idle/full/threshold/std stats |

## Typical Usage

```python
from data_loader import load_cpu_power_data
from cpu_power_model import CpuPowerModel

model = CpuPowerModel().fit(load_cpu_power_data("data/cpu_data.dat"))

result = model.predict(85, "intel-xeon-e5420")
# PowerPrediction(cpu_type='intel-xeon-e5420', cpu_pct=85, power_w=222.6,
#   power_lower_w=189.5, power_upper_w=255.6, is_spike=True, spike_threshold_w=204.9)
```
