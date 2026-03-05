# Session Notes — Pipeline Structural Improvements + Physics Formula Forecaster

**Date:** 2026-03-04
**Status:** COMPLETE — 225/225 tests passing

---

## What Was Done

Two-pass implementation of a structural overhaul + metric reliability fixes.

### Pass 1 — 11-file plan (222 tests)

| # | Problem | Fix |
|---|---------|-----|
| 1 | `EnergyTimesFM.evaluate()` missing `sMAPE` key — caused `KeyError` in comparison tables | Added sMAPE computation + `float()` casts to all returned values |
| 2 | `_TIMESFM_PREFERRED: frozenset` magic constant — hard to extend, not self-documenting | Replaced with `MODEL_ROUTING` dict + `get_best_model_key()` in `model_registry.py` |
| 3 | Physics formula `E × CIF + 0.002` documented but never used as a forecast strategy | Implemented in `physics_constraint.py`, wired into `api.py` as `_run_formula_forecast()` |
| 4 | `carbon_analysis.py` showing Prophet's sMAPE even after TimesFM override | Fixed by calling `tfm.evaluate()` explicitly after the forecast override |

### Pass 2 — metric reliability fixes (225 tests)

| # | Problem | Fix |
|---|---------|-----|
| 5 | TimesFM sMAPE epsilon `1e-15` vs Prophet `1e-10` — different scores on identical data | Standardised to `1e-10` in `timesfm_model.py` |
| 6 | MAPE divides by zero if any actual = 0 — `NaN` propagates silently | Added `if np.any(actuals == 0): mape = float("nan")` guard in both Prophet and TimesFM |
| 7 | Formula sMAPE vs TimesFM 7.3% unknown — `MODEL_ROUTING` unvalidated | Ran analysis pipeline: formula sMAPE = **0.0%** → routing confirmed correct |

---

## Files Created

### `model_base.py`
Abstract base class (ABC) for all single-metric forecasting models.
```python
class ForecasterBase(ABC):
    def fit(self, df, target_column) -> "ForecasterBase": ...
    def predict(self, periods, **kwargs) -> pd.DataFrame: ...   # [ds, yhat, yhat_lower, yhat_upper]
    def evaluate(self, df, target_column, train_ratio=0.8) -> Dict[str, float]: ...  # 7 keys
```
`EnsembleForecaster` is NOT a subclass — it returns a nested dict, different contract.

### `physics_constraint.py`
Pure-function module. No learnable parameters.
```python
EMBODIED_EMISSIONS_KGC02E_H: float = 0.002

def derive_carbon_emissions(consumption_fc, cif_fc) -> pd.DataFrame:
    # yhat       = consumption.yhat       * cif.yhat       + 0.002, clipped >= 0
    # yhat_lower = consumption.yhat_lower * cif.yhat_lower + 0.002
    # yhat_upper = consumption.yhat_upper * cif.yhat_upper + 0.002

def evaluate_formula_accuracy(df, train_ratio=0.8) -> Dict:
    # Returns: MAE, MSE, RMSE, MAPE, sMAPE, train_size, test_size
    # 80/20 split — same methodology as Prophet.evaluate()

def check_physical_consistency(forecast_df) -> Dict:
    # Returns: is_non_negative, negative_count, max_value, min_value, interval_ordered
```

### `model_registry.py`
Routing registry — replaces magic frozenset in api.py.
```python
MODEL_ROUTING: dict[str, list[str]] = {
    "carbonEmissions": ["formula", "timesfm", "prophet"],
    "consumption":     ["prophet"],
}
_DEFAULT_ROUTING: list[str] = ["timesfm", "prophet"]

def get_best_model_key(metric: str, available_keys: set[str]) -> str:
    # Returns first key from routing list present in available_keys
    # Unconditional fallback: "prophet"
```

### `benchmark.py`
Unified evaluation runner used by analysis scripts.
```python
class ModelBenchmark:
    def run(self, models: dict, df, metric, train_ratio=0.8) -> pd.DataFrame: ...
    def print_table(self, results: pd.DataFrame) -> None: ...
    def winner(self, results: pd.DataFrame) -> str: ...   # model with lowest sMAPE
```
Skips failing models with a printed warning — never crashes the pipeline.

### `tests/test_physics.py`
8 tests in `TestPhysicsFormula`:
- `test_formula_output_non_negative` — yhat/lower/upper all >= 0
- `test_formula_uncertainty_ordered` — yhat_lower <= yhat <= yhat_upper each row
- `test_formula_evaluate_returns_required_keys` — all 7 keys present
- `test_formula_smape_is_finite` — sMAPE is finite and > 0
- `test_physical_consistency_check` — check_physical_consistency() flags are correct
- `test_evaluate_returns_python_floats` — all numeric keys are Python `float`, not numpy scalars
- `test_evaluate_zero_actual_smape_stays_finite` — sMAPE finite when actuals = 0
- `test_smape_uses_consistent_epsilon` — sMAPE > 0 and < 100 on normal data

---

## Files Modified

### `prophet_model.py`
- Inherit `ForecasterBase`
- Zero-actual MAPE guard: `if np.any(actuals == 0): mape = float("nan")`

### `timesfm_model.py`
- Inherit `ForecasterBase`
- Added missing `sMAPE` key to `evaluate()` return dict
- `float()` casts on all returned values
- sMAPE epsilon `1e-15` → `1e-10` (matches Prophet)
- Zero-actual MAPE guard: `if np.any(actuals == 0): mape = float("nan")`

### `api.py`
- Removed `_TIMESFM_PREFERRED: frozenset`
- Added `from model_registry import get_best_model_key`
- Added `from physics_constraint import derive_carbon_emissions`
- Added `_run_formula_forecast(horizon)` — runs consumption + CIF forecasts, applies formula
- Rewrote `_run_best_forecast(metric, horizon)` — builds `available` set, calls `get_best_model_key()`, falls through formula → timesfm → prophet

### `carbon_analysis.py`
- Added formula evaluation block — stores as `results["carbonEmissions_formula"]`
- Fixed TimesFM eval bug: calls `tfm.evaluate()` after forecast override (was keeping Prophet's eval)
- Added `results["_tfm_used"]` bool for comparison table label
- Added comparison table in `print_insights()`:
  ```
  carbonEmissions FORECAST METHOD COMPARISON:
  Method                       sMAPE
  ------------------------------------
  Formula (E×CIF+0.002)         0.0%
  TimesFM                        7.3%
  ```

### `Dockerfile`
Added `model_base.py model_registry.py physics_constraint.py` to COPY list.

### `Dockerfile.test`
Added `model_base.py model_registry.py physics_constraint.py` to COPY list.
(`benchmark.py` excluded — not imported by test files or api.py.)
`Dockerfile.analysis` uses `COPY *.py ./` — picks up all new files automatically.

---

## Verification

```bash
docker compose --profile test run --rm --build test
# 222 passed  (after pass 1)
# 225 passed  (after pass 2)

docker compose --profile analysis up --build generate analyse
# Formula sMAPE = 0.0%  |  TimesFM sMAPE = 7.3%
# Routing ["formula", "timesfm", "prophet"] confirmed correct — no change needed
```

---

## Pending / Open Questions

### `node` query param — stub, not implemented
The API accepts `?node=<name>` on `/forecast/{metric}`, `/anomalies/{metric}`, and `/optimal-window` endpoints. Currently ignored — all nodes share a single trained model. To implement per-node isolation, `_app_state` would need to key models by node name and training would need to partition the CSV by node.

### Still missing (requires EVIDEN answers)
- Which observability framework do they use? (Prometheus? Grafana? Custom?)
- What push format/protocol? (HTTP POST? Prometheus remote-write? OTLP?)
- What metrics and granularity are collected at their end?
- Real data connector: currently reads a local synthetic CSV — no live telemetry ingestion path exists yet

---

## Architecture Snapshot (current)

```
data_generator.py        → synthetic CSV (8,760 rows)
data_loader.py           → validation + PipelineConfig
model_base.py            → ForecasterBase ABC
prophet_model.py         → EnergyProphet(ForecasterBase)
timesfm_model.py         → EnergyTimesFM(ForecasterBase)
physics_constraint.py    → derive_carbon_emissions(), evaluate_formula_accuracy()
model_registry.py        → MODEL_ROUTING, get_best_model_key()
benchmark.py             → ModelBenchmark (evaluation runner)
ensemble_model.py        → EnsembleForecaster (Prophet + TimesFM residual)
anomaly_detector.py      → detect_point_anomalies(), find_recurring_patterns()
api.py                   → FastAPI, 8 endpoints
carbon_analysis.py       → analysis + comparison table output
ensemble_analysis.py     → 3-way comparison output
visualizations.py        → 5 Matplotlib helpers
tests/
  test_data.py           → data contract
  test_forecasting.py    → chained Prophet forecasting
  test_api.py            → API contract
  test_anomaly.py        → anomaly engine
  test_physics.py        → physics formula contract
```
