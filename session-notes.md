# Session Notes — Energy & Carbon Forecasting Pipeline

**Last updated:** 2026-03-06
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

### Pass 6 — Bug audit + sign convention fix (260 → 260 tests, 2 assertions corrected + 2 renamed)

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

### Pass 7 — Enhancement A+B: library exceptions + public ensemble API (260 → 260 tests)

| # | Problem | Fix |
|---|---------|-----|
| 25 | `data_loader.load_and_validate()` called `sys.exit()` — kills API server silently on bad CSV schema | Replaced with `raise FileNotFoundError` / `raise ValueError`; `api.py` lifespan catches and falls back to synthetic data; analysis scripts catch and re-raise `SystemExit(1)` at CLI entry point |
| 26 | `test_data.py` asserted `pytest.raises(SystemExit)` — tested the bug, not the spec | Updated to `raises(FileNotFoundError, match=...)` / `raises(ValueError, match=...)`; method names corrected |
| 27 | `ensemble_analysis.py` accessed `ens._test_prophet` etc. directly — fragile private coupling | Added `get_test_predictions() -> dict` public method to `EnsembleForecaster`; all 6 `ens._test_*` access sites replaced with `preds = ens.get_test_predictions()` |
| 28 | `TestEvaluateContract` test checked for private `_test_*` attrs — tested internals not contract | Updated to test `get_test_predictions()` keys and array types |

### Pass 8 — Enhancement C+D+E: dead code + NaN display + doc accuracy (260 → 260 tests)

| # | Problem | Fix |
|---|---------|-----|
| 29 | `benchmark.py` never imported anywhere — dead code | Deleted file entirely |
| 30 | `visualizations.py` listed in CLAUDE.md but never existed | Removed from architecture section |
| 31 | CLAUDE.md said "Thirteen-module pipeline" — counted two non-existent modules | Corrected to "Eleven-module pipeline"; `data_loader.py` docstring updated; `ensemble_model.py` entry updated with `get_test_predictions()` mention |
| 32 | CLAUDE.md said "225-test suite" — stale count | Corrected to "260-test suite" with six groups listed |
| 33 | `f"{e['MAPE']:.1f}%"` prints `"nan%"` when actuals contain zeros — cosmetically broken | Added `_fmt_pct(v)` module-level helper to `carbon_analysis.py` and `ensemble_analysis.py`; fixed 5 call sites in each; inline guard in `ensemble_model.py` and `prophet_model.py` demo blocks |

**`_fmt_pct` design:** 3-line helper using `math.isnan()` — defined in each analysis script independently (they are standalone scripts; a shared `utils.py` for one function would over-engineer). Demo blocks in model files use inline `math.isnan()` guard to avoid adding module-level symbols used only in `__main__`.

---

## Open Questions (requires EVIDEN answers)

- Which observability framework? (Prometheus? Grafana? Custom?)
- Push format/protocol? (HTTP POST? Prometheus remote-write? OTLP?)
- What metrics and granularity are collected at their end?
- Real data connector: currently reads synthetic CSV — no live telemetry ingestion path

---

## Architecture Snapshot (current)

```
Modules (11):
  data_generator.py        synthetic CSV (8,760 rows, 15 columns)
  data_loader.py           schema validation + PipelineConfig
                           raises FileNotFoundError / ValueError (not sys.exit)
  model_base.py            ForecasterBase ABC
  prophet_model.py         EnergyProphet(ForecasterBase)
  timesfm_model.py         EnergyTimesFM(ForecasterBase)
  physics_constraint.py    derive_carbon_emissions(), evaluate_formula_accuracy()
  model_registry.py        MODEL_ROUTING, get_best_model_key()
  ensemble_model.py        EnsembleForecaster (Prophet + TimesFM residual)
                           get_test_predictions() → public API for analysis scripts
  anomaly_detector.py      detect_point_anomalies(), find_recurring_patterns()
  api.py                   FastAPI, 9 endpoints, per-node state isolation
  carbon_analysis.py       analysis script + formula vs TimesFM comparison table
                           _fmt_pct() helper for NaN-safe % display
  ensemble_analysis.py     3-way comparison script (Prophet / TimesFM / Ensemble)
                           _fmt_pct() helper; uses get_test_predictions() public API

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
  Dockerfile               API service (Prophet + TimesFM + all 11 modules)
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

None outstanding. All previously identified issues resolved across Passes 1–8.

---

## ITU-T L.1801 — Standard Analysis & Implementation Roadmap

### What is L.1801?

**ITU-T L.1801** — "Guidelines for assessing the environmental impact of artificial intelligence systems"
— is the world's first standardised methodology for measuring an AI system's complete environmental
footprint across its entire life cycle. Approved **February 6, 2026**. Co-published simultaneously by:

- **ITU-T Study Group 5** (as Recommendation ITU-T L.1801)
- **ETSI TC EE** (as ETSI ES 204 135)

**Normative references:**
- ITU-T L.1410 — Life Cycle Assessment (LCA) methodology (the methodological foundation)
- ITU-T L.1480 — Enabling effects method (second- and higher-order consequences)
- ISO/IEC 22989 — AI system definitions
- ISO 14040 — LCA framework

**Status:** Voluntary Recommendation today. Designed to become mandatory under the EU AI Act
(stated intent in the standard's own preamble). EVIDEN as an EU company may face mandatory
compliance within 2–3 years.

**Important note:** No ITU-T L.1808 exists. The user referred to L.1808 but meant L.1801
(the only L.18xx standard published to date).

---

### L.1801 Framework Summary

#### Four Life Cycle Stages (LCA)

| Stage | Scope |
|-------|-------|
| 1. Raw Material & Manufacturing | Mining, chip fabrication, hardware assembly, shipping |
| 2. Design, Development & Training | Algorithm design, data prep, training runs, CI/CD compute |
| 3. Deployment & Inference | Live queries, continuous learning, data transmission, cooling |
| 4. Retirement & End-of-Life | Decommissioning, data deletion, hardware recycling |

Training and inference **must always be reported separately** — blending them is non-compliant.

#### AI Classification System (Section 6.3)

| Type | Category | Hardware | Energy profile |
|------|----------|----------|---------------|
| 01 | Expert Systems | CPU only | Low |
| 02 | Machine Learning | CPU + GPU | Medium |
| 03 | Deep Learning | CPU + GPU | Medium-High |
| 04 | Generative AI | GPUs/TPUs | Comparatively high |

TimesFM (200M-parameter transformer) falls under **Type 03 (Deep Learning)**.
Prophet falls under **Type 02 (Machine Learning)**.

#### Mandatory vs. Recommended Impact Categories

| Category | Status | Currently in pipeline? |
|----------|--------|----------------------|
| GHG / Climate change (all gases, full LCA) | **Mandatory** | Partial — inference only |
| Water use (WUE — direct cooling + power generation) | Recommended | No |
| Minerals & metals (rare earths in hardware) | Recommended | No |
| Fossil fuel composition | Recommended | No |
| Biodiversity | Recommended (future) | No |

#### Functional Unit Requirement

Every carbon figure must be expressed relative to a **declared functional unit** — a reference
measure of the service delivered that also captures quality. Comparative claims across
configurations require identical functional units.

For network management / predictive scheduling AI, L.1801 explicitly acknowledges:
"difficult to set numeric value" — this is an open problem in the standard itself.

#### Compliance Model

Partial compliance is permitted. Omissions **must be explicitly declared** — they cannot be
silently omitted. This prevents cherry-picking favourable metrics. No enforcement today;
enforcement depends on regulatory adoption (EU AI Act).

---

### Current Pipeline Alignment with L.1801

What the pipeline already does correctly, mapped to L.1801 requirements:

| L.1801 Requirement | Pipeline implementation | Location |
|-------------------|------------------------|----------|
| GHG emissions per functional unit | `SCI = (E × I + M) / R` | `physics_constraint.py` |
| Separate operational + embodied emissions | `operationalEmissions` + `embodiedEmissions = 0.002` | `data_generator.py`, data schema |
| Carbon Intensity Factor (grid mix) | `carbonIntensityFactor` column + regressor | all models |
| Hourly granularity measurement | 8,760 hourly rows | `data_generator.py` |
| Optimal scheduling to lower-CI windows | `/optimal-window` endpoint | `api.py`, `anomaly_detector.py` |
| Excess carbon anomaly detection | `/anomalies`, `/investigation-leads` | `anomaly_detector.py` |
| Model routing by accuracy | `MODEL_ROUTING` — formula → TimesFM → Prophet | `model_registry.py` |

The pipeline is a strong implementation of **Stage 3 (Deployment & Inference)** of the L.1801
lifecycle. Stages 1, 2, and 4 are entirely absent.

---

### Compliance Gaps — Detail

#### Gap 1: Training Phase Emissions (Stage 2) — MANDATORY

**What L.1801 requires:**
Every AI system report must include Stage 2 emissions. For third-party foundation models, the
standard requires allocating a fraction of the model's total training carbon cost to each inference:

```
Training CO₂ per inference = Total training CO₂ ÷ Total estimated lifetime inferences
```

**Current pipeline state:** Zero. The pipeline has no concept of training carbon. TimesFM's
~925 MB model was produced by Google using large GPU clusters. That carbon cost is completely
invisible. Prophet's training carbon is also untracked (negligible per inference, but still
untracked).

**The data problem:** Google has not published TimesFM's training energy or carbon figures.
This is the standard's acknowledged "Data Transparency Problem" — unsolvable from EVIDEN's
side without disclosure from the model provider.

**Implementation path:**
1. Add `training_carbon_kgco2e_per_inference: Optional[float]` field to `ForecastResponse`
2. For TimesFM: use a conservative published estimate for comparable 200M-param models
   (reference: CodeCarbon / ML CO₂ Impact calculator for similar-scale transformers)
3. For Prophet: estimate from training time × hardware power × CI at training location
4. Document source and assumptions in the API response and in L.1801 compliance statement
5. If no data available: set to `null` and emit `"training_carbon": "excluded — model provider
   data not disclosed"` in a compliance metadata field

**Effort:** Medium. Primarily research + API metadata; no new model code needed.

#### Gap 2: Hardware Manufacturing Carbon — the real `M` (Stage 1) — MANDATORY

**What L.1801 requires:**
Full Stage 1 LCA: mining, chip fabrication, hardware assembly, shipping — allocated across
the hardware's useful life, then by the service's resource fraction.

Correct formula for M:
```
M = (Server manufacturing CO₂ × Service CPU/RAM allocation fraction)
    ───────────────────────────────────────────────────────────────────
    Server useful life in hours (typically ~35,040 h = 4 years)
```

**Current pipeline state:**
```python
EMBODIED_EMISSIONS_KGC02E_H = 0.002  # kgCO2e/h — hardcoded constant
```
This constant is a reasonable approximation (typical hyperscale server manufacturing
CO₂ ~300–600 kg, over 4 years at 10% allocation → M ≈ 0.001–0.002 kgCO2e/h) but it is
**not derived from actual hardware** and cannot be claimed L.1801-compliant.

Critical operational risk: if EVIDEN migrates to new hardware (GPU node for inference,
cloud migration, server refresh), the `0.002` becomes silently wrong.

**Implementation path:**
1. Identify the actual server model(s) running the service (EVIDEN must provide this)
2. Look up manufacturing carbon via **Boavizta API** (`https://api.boavizta.org`) or
   vendor product carbon footprint databases (Dell, HP, HPE all publish these)
3. Define service allocation fraction (% CPU, % RAM the service uses on each server)
4. Replace hardcoded constant with a config-driven value in `physics_constraint.py`:
   ```python
   EMBODIED_EMISSIONS_KGC02E_H: float  # loaded from config, not hardcoded
   ```
5. Add `hardware_lca_source: str` field to API responses for traceability
6. Re-run `test_physics.py` — the constant change will require updating test fixtures

**Key external resource:** Boavizta is an open-source API that returns CO₂ equivalent for
hardware manufacturing given a server model name, usage duration, and location. Free to use.

**Effort:** Medium. Requires EVIDEN hardware info + one Boavizta API call + config refactor.

#### Gap 3: Water Use / WUE (Stage 3) — RECOMMENDED

**What L.1801 requires (recommended, not mandatory):**
Water Usage Effectiveness (WUE) = Total water consumed (litres) / IT energy (kWh).
Covers direct cooling water + indirect water from power generation.

Typical ranges:
- Air-cooled data center: 1.8–2.5 L/kWh
- Evaporatively cooled: 4–10 L/kWh
- For a 0.1 kWh/h service: 0.2–1 litre/h invisible today

**Current pipeline state:** No water column in schema. No WUE in API responses. Complete absence.

**Implementation path:**
1. EVIDEN must obtain WUE from their infrastructure/cloud provider
2. Add `wue_l_per_kwh: Optional[float]` to the data schema and API config
3. Derive `water_consumption_l_h = consumption × wue_l_per_kwh` as a new metric
4. Expose in `/forecast/all` and in `/metrics/prometheus` for HPA consumption
5. If WUE unavailable: declare exclusion in L.1801 compliance metadata

**Effort:** Low-Medium. Requires WUE data from infrastructure provider; pipeline changes minimal.

#### Gap 4: Second-Order Effects — Scheduling Benefits (Enabling Effects) — MANDATORY

**What L.1801 requires:**
Built on ITU-T L.1480 (enabling effects method), L.1801 requires mapping consequences beyond
the AI system's direct power consumption. For EVIDEN's predictive scheduling use case, the
consequence tree is:

```
First-order effects (already measured):
  - Energy/carbon of API calls to generate forecasts
  - Network transmission to observability framework

Second-order effects (NOT measured — the entire justification for the pipeline):
  - Workloads rescheduled to lower-CI windows → reduced carbon from the scheduled work
  - Rebound effect: scheduling efficiency → increased total workload → partial carbon offset
  - False positive investigation leads → wasted operator time (human + terminal energy)

Higher-order effects (out of scope today):
  - If carbon-aware scheduling becomes standard, does total ICT demand increase?
```

**Critical implication:** Without measuring second-order effects, EVIDEN cannot claim any
**net carbon reduction benefit** from the pipeline — only that the pipeline itself has a
certain footprint. The business case for the whole system is in the second-order measurement.

**Complete L.1801 net impact formula:**
```
Net environmental impact =
    Carbon cost of running the forecasting pipeline
    MINUS carbon saved by workloads rescheduled to lower-CI windows
    PLUS/MINUS rebound effect from utilisation changes
```

**Implementation path:**
1. Add a `scheduling_outcome` feedback endpoint (e.g., `POST /scheduling-outcome`) where
   EVIDEN's orchestrator reports: which workload was rescheduled, from which hour, to which
   hour, and the workload's energy profile
2. `anomaly_detector.py` already computes `estimated_carbon_saving_pct` per investigation
   lead — this becomes the *prediction*; the feedback endpoint captures the *actual* saving
3. Compute realised saving: `(CI_original_hour - CI_rescheduled_hour) × workload_energy`
4. Accumulate savings in a `scheduling_outcomes` table alongside the service data
5. Expose cumulative net carbon saving in `/health` and `/metrics/prometheus`

This is the highest-value gap to close and the most scientifically important one for L.1801.

**Effort:** High. Requires integration with EVIDEN's orchestrator/scheduler (currently unknown).
Depends on the observability framework integration question being answered first.

#### Gap 5: Inference vs. Training Separation in API Responses — MANDATORY

**What L.1801 requires:**
Every carbon figure in an L.1801 report must be tagged to its LCA stage. Mandatory breakdown:
- Manufacturing carbon (Stage 1)
- Training carbon (Stage 2)
- Operational inference carbon per request (Stage 3)
- End-of-life carbon (Stage 4)

**Current pipeline state:**
`ForecastResponse.value` returns a single carbon figure with no stage tagging. The `model_used`
field identifies which model served the forecast but says nothing about carbon stage attribution.
`carbonEmissions` in forecast responses is the **carbon of the monitored service**, not the
carbon of the forecasting pipeline itself — these are two different things and the L.1801
system boundary must be declared.

**Implementation path:**
1. Add a `carbon_breakdown` object to `ForecastResponse` (optional, enabled by query param):
   ```json
   "carbon_breakdown": {
     "operational_inference_kgco2e_h": 0.041,
     "embodied_manufacturing_kgco2e_h": 0.002,
     "training_allocated_kgco2e_per_call": null,
     "training_note": "excluded — TimesFM training data not disclosed by Google"
   }
   ```
2. Add a `l1801_compliance` metadata block to `/health` response declaring:
   - System boundary (Stage 3 only, partial)
   - Excluded categories and reasons
   - Functional unit declaration
   - Data sources for each value

**Effort:** Low. Pure API metadata addition; no new data collection or model changes.

#### Gap 6: Functional Unit — Not Formally Declared — MANDATORY

**What L.1801 requires:**
Every carbon figure must be expressed relative to a declared functional unit. The functional
unit must be explicitly stated in all reports. Comparative claims require identical units.
Quality must be captured, not just quantity.

**Current pipeline state:**
- `functionalUnit` (req/h) exists as a column, regressor, and SCI denominator
- Never returned in API responses as a declared reference unit
- SCI response does not document what "1 request" means in quality/accuracy terms
- When `functionalUnit` is unknown for future periods, pipeline fills `0` → SCI undefined
- No statement of what forecast quality level R represents (horizon? accuracy band?)

**Implementation path:**
1. Add `functional_unit` field to `SciResponse`:
   ```json
   "functional_unit": {
     "value": 1,
     "unit": "request/hour",
     "description": "One HTTP request served by the monitored infrastructure service",
     "quality_note": "Request rate only; response quality/latency not captured (L.1801 open problem for network management AI)"
   }
   ```
2. When `functionalUnit` column is zero/null for future periods: emit a warning in the
   response rather than silently computing SCI with a zero denominator
3. Add to `/health` response: the declared functional unit for the current node/dataset

**Effort:** Low. Entirely metadata and documentation; no model or data changes.

---

### L.1801 Compliance Tier Summary

| Gap | L.1801 status | Fixable unilaterally? | Effort | Priority |
|-----|--------------|----------------------|--------|----------|
| Gap 5: Inference/training stage tagging in API | Mandatory | Yes | Low | 1 — do first |
| Gap 6: Functional unit declaration | Mandatory | Yes | Low | 2 — do with Gap 5 |
| Gap 1: Training phase emissions | Mandatory | Partial (data problem) | Medium | 3 |
| Gap 2: Hardware manufacturing M | Mandatory | Yes (needs HW info) | Medium | 4 |
| Gap 4: Second-order scheduling effects | Mandatory | Requires EVIDEN integration | High | 5 — after integration |
| Gap 3: Water use / WUE | Recommended | Requires infra provider data | Low-Med | 6 |

**Recommended implementation sequence:**
- **Pass 9** — Gaps 5 + 6: L.1801 compliance metadata in API responses (pure additions,
  no breaking changes, no new data needed, enables partial compliance claim immediately)
- **Pass 10** — Gap 2: Replace hardcoded `0.002` with config-driven hardware LCA value
  (requires EVIDEN to identify server hardware; Boavizta API lookup)
- **Pass 11** — Gap 1: Training carbon attribution (research phase first; add to API once
  estimates are validated)
- **Pass 12** — Gap 4: Scheduling outcome feedback loop (blocked on EVIDEN integration;
  design the endpoint now, implement when orchestrator is known)
- **Pass 13** — Gap 3: Water use / WUE (blocked on infrastructure provider data)

---

### New Open Questions (requires EVIDEN answers for L.1801)

- What server hardware runs the pipeline? (Make, model, generation — needed for Boavizta LCA)
- What is the service's CPU/RAM allocation fraction on those servers?
- What is the data center's WUE (Water Usage Effectiveness)?
- What scheduling orchestrator will consume the forecast API? (Kubernetes HPA? Custom?)
- Will EVIDEN's orchestrator be able to report actual scheduling outcomes back to the pipeline?
- What is EVIDEN's target L.1801 compliance tier? (Partial Stage 3 only, or full 4-stage?)
- Is there a regulatory deadline driving the compliance requirement (EU AI Act timeline)?

---

### External Resources for Implementation

| Resource | URL | Use |
|----------|-----|-----|
| Boavizta API | `https://api.boavizta.org` | Hardware manufacturing LCA by server model |
| CodeCarbon | `https://codecarbon.io` | Training carbon estimation for ML models |
| ML CO₂ Impact | `https://mlco2.impact.nextml.org` | Quick training carbon calculator |
| ITU-T L.1801 text | `https://www.itu.int/rec/T-REC-L.1801/en` | Normative standard text |
| L.1801 framework site | `https://l1801framework.netlify.app/` | Interactive framework overview |
| ISO/IEC 21031:2024 | `https://www.iso.org/standard/86612.html` | SCI specification (aligns with our formula) |
| Green Software Foundation SCI | `https://sci.greensoftware.foundation/` | SCI spec + pattern library |
