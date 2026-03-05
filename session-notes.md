# Session Notes — Energy & Carbon Forecasting Pipeline

**Last updated:** 2026-03-05
**Status:** COMPLETE — 260 passed, 5 skipped (TimesFM-only tests, skipped in lightweight image)

---

## Cumulative Work Log

### Pass 1 — Structural overhaul (222 → 222 tests)

| # | Problem | Fix |
|---|---------|-----|
| 1 | `EnergyTimesFM.evaluate()` missing `sMAPE` key — caused `KeyError` in comparison tables | Added sMAPE + `float()` casts to all returned values |
| 2 | `_TIMESFM_PREFERRED: frozenset` magic constant — hard to extend | Replaced with `MODEL_ROUTING` dict + `get_best_model_key()` in `model_registry.py` |
| 3 | Physics formula `E × CIF + 0.002` documented but never used as a forecast strategy | Implemented in `physics_constraint.py`, wired into `api.py` as `_run_formula_forecast()` |
| 4 | `carbon_analysis.py` showing Prophet's sMAPE even after TimesFM override | Fixed by calling `tfm.evaluate()` explicitly after the forecast override |

### Pass 2 — Metric reliability (222 → 225 tests)

| # | Problem | Fix |
|---|---------|-----|
| 5 | TimesFM sMAPE epsilon `1e-15` vs Prophet `1e-10` — inconsistent scores | Standardised to `1e-10` in `timesfm_model.py` |
| 6 | MAPE zero-division when actuals contain 0 — silent `inf` propagation | Added `if np.any(actuals == 0): mape = float("nan")` in Prophet + TimesFM |
| 7 | Formula sMAPE vs TimesFM 7.3% unvalidated | Ran analysis pipeline: formula sMAPE = **0.0%** → routing confirmed correct |

### Pass 3 — Audit fixes + model_used field (225 → 230 tests)

| # | Problem | Fix |
|---|---------|-----|
| 8 | `/optimal-window` and `/metrics/prometheus` calling `_run_forecast` (Prophet-only), bypassing formula routing | Fixed both to use `_run_best_forecast` |
| 9 | No `model_used` field on forecast responses — callers couldn't tell which model served | Added `_ForecastResult(NamedTuple)`, `_ModelKey = Literal[...]`, `model_used` on `ForecastResponse` + `OptimalWindowResponse` + Prometheus labels |
| 10 | `sMAPE` epsilon `1e-15` in `carbon_analysis.py:_derive_sci()` (missed in pass 2) | Fixed to `1e-10` |
| 11 | CLAUDE.md endpoint table had 4 non-existent endpoints, missing 4 real ones | Rewrote endpoint table to match actual code |

### Pass 4 — Ensemble fixes + test_ensemble.py (230 → 252 tests, +5 skipped)

| # | Problem | Fix |
|---|---------|-----|
| 12 | `ensemble_model._compute_metrics()` returned numpy scalars (no `float()` casts) | Added `float()` casts to all 5 metric values |
| 13 | `ensemble_model._compute_metrics()` MAPE zero-division → `inf` not `nan` | Added zero-guard consistent with all other modules |
| 14 | `ensemble_model.__main__` block and class docstring referenced `"totalEnergyConsumption"` and `"trainingActive"` (wrong schema) | Updated to `"consumption"` and `"functionalUnit"` |
| 15 | No test coverage for `EnsembleForecaster` | Created `tests/test_ensemble.py` — 21 pure-function tests (no real model fitting), 5 integration tests skipped without TimesFM |

`tests/test_ensemble.py` uses `sys.modules` patching (`torch`, `timesfm`, `timesfm_model`) to run pure-function tests in the lightweight Docker image. The `TestEvaluateContract` class is marked `skipif(not _TIMESFM_INSTALLED)` and runs locally with the full environment.

### Pass 5 — Per-node isolation (252 → 260 tests)

| # | Problem | Fix |
|---|---------|-----|
| 16 | `?node=<name>` accepted by all endpoints but silently ignored — all requests hit the same models | Refactored `_state` from a flat dict to `dict[str, dict]` keyed by node name |
| 17 | New nodes had no state until EVIDEN integration is done | `POST /data?node=<name>` bootstraps new nodes from default state; node-specific Prophet refit queued async |
| 18 | `GET /health` showed no information about known nodes | Added `nodes: list[str]` field to `HealthResponse` |

**Per-node design:**
- `_DEFAULT_NODE = "_default"` — startup models always live here
- `_require_state(node)` — returns node state or falls back to `_DEFAULT_NODE` (unknown nodes never get 503)
- `POST /data?node=<name>` — first call initialises node with copy of default df + models, then schedules node-specific refit
- Atomic swap: refit builds new models locally, replaces node's state dict in one assignment
- All internal functions (`_run_forecast`, `_run_best_forecast`, etc.) thread `node: Optional[str]` consistently

### Pass 6 — Bug audit + sign convention fix (260 → 260 tests, 2 assertions corrected)

| # | Problem | Fix |
|---|---------|-----|
| 19 | `ensemble_analysis.py:_derive_sci_ensemble()` MAPE divides by `sci_actuals` directly — no zero-guard | Added standard `if np.any == 0: nan else: float(...)` guard + `float()` casts to all 5 metrics |
| 20 | `ensemble_analysis.py:_chart_forecast_comparison()` — 3 inline MAPE calculations with no zero-guard | Extracted `_safe_mape(actuals, preds)` helper; 3 divisions replaced with one-liner calls |
| 21 | `anomaly_detector.py` `estimated_carbon_saving_pct` / `estimated_cost_saving_pct` sign inverted — negative meant saving | Fixed formula to `(baseline - optimal) / baseline × 100`: **positive = saving** (matches GHG Protocol and AWS Compute Optimizer conventions) |
| 22 | `tests/test_anomaly.py` — 2 tests encoded the inverted sign as correct | Assertions flipped `< 0` → `> 0`; test names + docstrings updated to document correct semantics |
| 23 | `carbon_analysis.py:_derive_sci()` MAPE used `np.maximum(sci_actuals, 1e-12)` — inconsistent with standard zero-guard pattern | Replaced with standard guard; `float()` casts added to all 5 metrics |
| 24 | `carbon_analysis.py:_chart_forecast()` labelled all model fits as "Prophet fit" — wrong when TimesFM is active for carbonEmissions | Added `model_label: str = "Prophet"` param; `generate_charts()` passes `"TimesFM"` or `"Prophet"` based on `_tfm_used` |

---

## Key Measurements

| Metric | Method | sMAPE |
|--------|--------|-------|
| carbonEmissions | Formula (E×CIF+0.002) | **0.0%** |
| carbonEmissions | TimesFM | 7.3% |
| carbonEmissions | Prophet | ~15% |

Formula first for `carbonEmissions` is validated. Routing `["formula", "timesfm", "prophet"]` in `model_registry.py` is correct.

---

## Open Questions (requires EVIDEN answers)

- Which observability framework? (Prometheus? Grafana? Custom?)
- Push format/protocol? (HTTP POST? Prometheus remote-write? OTLP?)
- What metrics and granularity are collected at their end?
- Real data connector: currently reads synthetic CSV — no live telemetry ingestion path

---

## Architecture Snapshot (current)

```
Modules (13):
  data_generator.py        synthetic CSV (8,760 rows, 15 columns)
  data_loader.py           schema validation + PipelineConfig  [NOTE: uses sys.exit() — see Known Issues]
  model_base.py            ForecasterBase ABC
  prophet_model.py         EnergyProphet(ForecasterBase)
  timesfm_model.py         EnergyTimesFM(ForecasterBase)
  physics_constraint.py    derive_carbon_emissions(), evaluate_formula_accuracy()
  model_registry.py        MODEL_ROUTING, get_best_model_key()
  benchmark.py             ModelBenchmark — DEAD CODE (never imported anywhere)
  ensemble_model.py        EnsembleForecaster (Prophet + TimesFM residual)
  anomaly_detector.py      detect_point_anomalies(), find_recurring_patterns()
  api.py                   FastAPI, 9 endpoints, per-node state isolation
  carbon_analysis.py       analysis script + formula vs TimesFM comparison table
  ensemble_analysis.py     3-way comparison script (Prophet / TimesFM / Ensemble)
                           [NOTE: accesses private _test_* attrs on EnsembleForecaster]

Tests (260 passed, 5 skipped):
  test_data.py             data contract (schema, value ranges, SCI identity)
  test_forecasting.py      chained Prophet regressor (zero-fill vs chained)
  test_api.py              API contract + model_used field + per-node isolation
  test_anomaly.py          anomaly engine (Prophet-free, synthetic forecasts)
  test_physics.py          physics formula contract (8 tests)
  test_ensemble.py         ensemble pure functions (21 tests) + contract (5 skipped)

Requirements:
  requirements-service.txt  prophet, pandas, numpy, fastapi, uvicorn
  requirements-test.txt     pytest, httpx
  requirements-analysis.txt prophet + timesfm + matplotlib + plotly (analysis image only)

Docker images:
  Dockerfile               API service (Prophet + TimesFM + all 13 modules)
  Dockerfile.test          Lightweight test image (no TimesFM/PyTorch)
  Dockerfile.analysis      Analysis image (COPY *.py ./), uses requirements-analysis.txt
```

## API Endpoints (9 total)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness + model state + known nodes |
| POST | `/data` | Ingest rows, schedule refit (`?node=` scopes to node) |
| GET | `/forecast/all` | All metrics in one response |
| GET | `/forecast/sci` | SCI derived (kgCO2e/req) |
| GET | `/forecast/{metric}` | Single metric, best model |
| GET | `/optimal-window` | Best low-carbon scheduling windows |
| GET | `/anomalies` | Point anomalies + dual-model confidence |
| GET | `/investigation-leads` | Ranked recurring anomaly patterns |
| GET | `/metrics/prometheus` | Prometheus text format for HPA |

All forecast/anomaly endpoints accept `?node=<name>` for per-node isolation.

---

## Known Issues (not yet fixed)

| # | File | Issue | Severity |
|---|------|-------|----------|
| A | `data_loader.py:113,122` | Uses `sys.exit()` — anti-pattern in library code; should raise `FileNotFoundError` / `ValueError` so callers can catch | Medium |
| B | `benchmark.py` | Never imported anywhere — dead code with no tests | Low |
| C | `ensemble_analysis.py:134-135` | Accesses private `_test_prophet`, `_test_timesfm`, `_test_ensemble`, `_test_actuals` attrs directly on `EnsembleForecaster` — fragile coupling | Low |
| D | Analysis print statements | `f"{e['MAPE']:.1f}%"` prints `"nan%"` when actuals contain zeros — cosmetically broken | Low |
| E | CLAUDE.md | Lists `visualizations.py` in module description but the file does not exist | Low |
