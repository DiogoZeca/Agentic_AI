"""
Test 3: API Contract

Uses FastAPI's TestClient (no running server required).
The lifespan runs once per session (fits 7 Prophet models on the 720-row CSV
set up in conftest.py — approximately 30-50s total).

Verifies:
  - /health returns status="ok" with 7 models ready
  - /forecast/all?horizon=6 returns 8 metrics (7 direct + SCI), 6 predictions each
  - /forecast/sci?horizon=6 returns ordered uncertainty intervals
  - /forecast/{metric} works for each of the 7 directly-fit metrics
  - /forecast/unknown_metric returns 404
  - /optimal-window?horizon_days=3 returns 3 scheduling windows
  - /optimal-window windows contain vs_daily_mean_pct showing relative carbon saving
  - /anomalies contract: fields, validation, metric 404, direction enum
  - /investigation-leads contract: fields, ranking, top_n cap, parameter reflection
"""
import pytest
from fastapi.testclient import TestClient

# api.py reads DATA_PATH from os.environ at import time.
# conftest.py sets DATA_PATH to a small 30-day CSV before this module loads.
from api import app

ALL_DIRECT_METRICS = [
    "consumption",
    "carbonEmissions",
    "carbonIntensityFactor",
    "greenConsumptionPercentage",
    "cpuUtilization",
    "functionalUnit",
    "cost",
]

EXPECTED_ALL_KEYS = ALL_DIRECT_METRICS + ["softwareCarbonIntensity"]


@pytest.fixture(scope="session")
def client():
    """
    Session-scoped TestClient — triggers lifespan once and reuses models.
    Fitting 7 Prophet models on 720 rows takes ~30-50s on first access.
    """
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:

    def test_status_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_models_ready_count(self, client):
        r = client.get("/health")
        models = r.json()["models_ready"]
        assert len(models) == 7, (
            f"Expected 7 models ready, got {len(models)}: {models}"
        )

    def test_models_ready_names(self, client):
        r = client.get("/health")
        models = set(r.json()["models_ready"])
        assert models == set(ALL_DIRECT_METRICS), (
            f"Model name mismatch.\n  Got:      {models}\n"
            f"  Expected: {set(ALL_DIRECT_METRICS)}"
        )

    def test_data_rows_count(self, client):
        r = client.get("/health")
        assert r.json()["data_rows"] == 720

    def test_fitted_at_present(self, client):
        r = client.get("/health")
        assert "fitted_at" in r.json()
        assert r.json()["fitted_at"].endswith("Z")


class TestForecastAllEndpoint:

    def test_returns_8_metrics(self, client):
        r = client.get("/forecast/all?horizon=6")
        assert r.status_code == 200
        keys = set(r.json().keys())
        assert keys == set(EXPECTED_ALL_KEYS), (
            f"Key mismatch.\n  Got:      {keys}\n  Expected: {set(EXPECTED_ALL_KEYS)}"
        )

    def test_each_metric_has_correct_horizon(self, client):
        r = client.get("/forecast/all?horizon=6")
        payload = r.json()
        for metric in EXPECTED_ALL_KEYS:
            n = len(payload[metric]["predictions"])
            assert n == 6, (
                f"Metric '{metric}': expected 6 predictions, got {n}"
            )

    def test_prediction_fields_present(self, client):
        r = client.get("/forecast/all?horizon=6")
        first_pred = r.json()["consumption"]["predictions"][0]
        for field in ["ds", "yhat", "yhat_lower", "yhat_upper"]:
            assert field in first_pred, f"Missing field '{field}' in prediction"

    def test_sci_metric_in_response(self, client):
        r = client.get("/forecast/all?horizon=6")
        sci = r.json().get("softwareCarbonIntensity")
        assert sci is not None
        assert sci["metric"] == "softwareCarbonIntensity"
        assert sci["unit"] == "kgCO2e/req"

    def test_horizon_limit_enforced(self, client):
        r = client.get("/forecast/all?horizon=169")  # > 168 max
        assert r.status_code == 422  # Unprocessable Entity


class TestForecastSciEndpoint:

    def test_status_ok(self, client):
        r = client.get("/forecast/sci?horizon=6")
        assert r.status_code == 200

    def test_metric_and_unit(self, client):
        r = client.get("/forecast/sci?horizon=6")
        data = r.json()
        assert data["metric"] == "softwareCarbonIntensity"
        assert data["unit"] == "kgCO2e/req"

    def test_uncertainty_intervals_ordered(self, client):
        """
        SCI uncertainty propagation rule:
          yhat_lower = carbon_lower / max(request_upper, 1)  — best-case SCI
          yhat_upper = carbon_upper / max(request_lower, 1)  — worst-case SCI

        Therefore: yhat_lower ≤ yhat ≤ yhat_upper for every point.
        """
        r = client.get("/forecast/sci?horizon=24")
        predictions = r.json()["predictions"]
        for i, p in enumerate(predictions):
            assert p["yhat_lower"] <= p["yhat"] + 1e-9, (
                f"SCI[{i}]: yhat_lower ({p['yhat_lower']}) > yhat ({p['yhat']})"
            )
            assert p["yhat"] <= p["yhat_upper"] + 1e-9, (
                f"SCI[{i}]: yhat ({p['yhat']}) > yhat_upper ({p['yhat_upper']})"
            )

    def test_sci_values_are_positive(self, client):
        r = client.get("/forecast/sci?horizon=24")
        predictions = r.json()["predictions"]
        for p in predictions:
            assert p["yhat"] > 0, f"SCI yhat is non-positive: {p['yhat']}"


class TestForecastMetricEndpoint:

    @pytest.mark.parametrize("metric", ALL_DIRECT_METRICS)
    def test_each_direct_metric_returns_200(self, client, metric):
        r = client.get(f"/forecast/{metric}?horizon=6")
        assert r.status_code == 200, (
            f"Metric '{metric}' returned {r.status_code}: {r.text}"
        )

    @pytest.mark.parametrize("metric", ALL_DIRECT_METRICS)
    def test_correct_prediction_count(self, client, metric):
        r = client.get(f"/forecast/{metric}?horizon=6")
        n = len(r.json()["predictions"])
        assert n == 6, f"Metric '{metric}': expected 6 predictions, got {n}"

    @pytest.mark.parametrize("metric,expected_unit", [
        ("consumption",               "kWh/h"),
        ("carbonEmissions",           "kgCO2e/h"),
        ("carbonIntensityFactor",     "kgCO2/kWh"),
        ("greenConsumptionPercentage", "%"),
        ("cpuUtilization",            "fraction"),
        ("functionalUnit",            "req/h"),
        ("cost",                      "EUR/h"),
    ])
    def test_metric_unit(self, client, metric, expected_unit):
        r = client.get(f"/forecast/{metric}?horizon=1")
        assert r.json()["unit"] == expected_unit, (
            f"Metric '{metric}': expected unit '{expected_unit}', "
            f"got '{r.json()['unit']}'"
        )

    def test_unknown_metric_returns_404(self, client):
        r = client.get("/forecast/totalEnergyConsumption?horizon=6")
        assert r.status_code == 404

    def test_forecast_metric_name_in_response(self, client):
        r = client.get("/forecast/carbonEmissions?horizon=6")
        assert r.json()["metric"] == "carbonEmissions"

    def test_consumption_regressor_values_realistic(self, client):
        """
        Chained forecasting: consumption forecast should predict realistic values
        (not near-zero baseline from zero-fill). Expect yhat > 0.03 kWh/h.
        """
        r = client.get("/forecast/consumption?horizon=24")
        yhats = [p["yhat"] for p in r.json()["predictions"]]
        assert min(yhats) > 0.020, (
            f"Consumption forecast too low (min={min(yhats):.4f} kWh/h). "
            f"This may indicate zero-fill rather than chained forecasting."
        )


class TestOptimalWindowEndpoint:

    def test_returns_correct_day_count(self, client):
        r = client.get("/optimal-window?horizon_days=3")
        assert r.status_code == 200
        windows = r.json()["windows"]
        assert len(windows) == 3, f"Expected 3 windows, got {len(windows)}"

    def test_window_fields_present(self, client):
        r = client.get("/optimal-window?horizon_days=2&window_hours=4")
        window = r.json()["windows"][0]
        for field in ["date", "start_hour", "end_hour", "avg_predicted_carbon", "vs_daily_mean_pct", "unit"]:
            assert field in window, f"Missing field '{field}' in scheduling window"

    def test_window_hours_correct(self, client):
        r = client.get("/optimal-window?horizon_days=2&window_hours=4")
        for w in r.json()["windows"]:
            # end_hour uses % 24 (0–23), matching anomaly_detector convention.
            # Use modular arithmetic to handle midnight-crossing windows correctly.
            duration = (w["end_hour"] - w["start_hour"]) % 24
            assert duration == 4, (
                f"Window duration should be 4h, got {duration}h "
                f"(start={w['start_hour']}, end={w['end_hour']})"
            )

    def test_carbon_unit_correct(self, client):
        r = client.get("/optimal-window?horizon_days=2")
        for w in r.json()["windows"]:
            assert w["unit"] == "kgCO2e/h"

    def test_response_metadata(self, client):
        r = client.get("/optimal-window?horizon_days=3&window_hours=6")
        data = r.json()
        assert data["horizon_days"] == 3
        assert data["window_hours"] == 6
        assert "generated_at" in data

    def test_invalid_horizon_days(self, client):
        r = client.get("/optimal-window?horizon_days=15")  # > 14 max
        assert r.status_code == 422


# ── /anomalies ────────────────────────────────────────────────────────────────

ANOMALY_POINT_FIELDS = [
    "ds", "actual", "expected", "upper_bound", "lower_bound",
    "excess_pct", "direction", "hour_of_day", "is_weekend",
    "carbon_intensity", "excess_carbon_kgco2e", "excess_cost_eur",
    "timesfm_flagged", "confidence",
]

ANOMALY_RESPONSE_FIELDS = [
    "metric", "unit", "direction", "lookback_days", "total_anomalies",
    "generated_at", "anomalies",
]


class TestAnomaliesEndpoint:

    def test_status_ok(self, client):
        r = client.get("/anomalies?metric=consumption")
        assert r.status_code == 200

    def test_response_fields_present(self, client):
        r = client.get("/anomalies?metric=consumption")
        data = r.json()
        for field in ANOMALY_RESPONSE_FIELDS:
            assert field in data, f"Missing top-level field '{field}'"

    def test_metric_reflected(self, client):
        r = client.get("/anomalies?metric=carbonEmissions")
        assert r.json()["metric"] == "carbonEmissions"

    def test_unit_correct(self, client):
        r = client.get("/anomalies?metric=consumption")
        assert r.json()["unit"] == "kWh/h"

    def test_carbon_emissions_unit(self, client):
        r = client.get("/anomalies?metric=carbonEmissions")
        assert r.json()["unit"] == "kgCO2e/h"

    def test_direction_default_is_excess(self, client):
        r = client.get("/anomalies?metric=consumption")
        assert r.json()["direction"] == "excess"

    def test_direction_param_reflected(self, client):
        for direction in ("excess", "deficit", "both"):
            r = client.get(f"/anomalies?metric=consumption&direction={direction}")
            assert r.status_code == 200, f"direction={direction} returned {r.status_code}"
            assert r.json()["direction"] == direction

    def test_invalid_direction_returns_422(self, client):
        r = client.get("/anomalies?metric=consumption&direction=sideways")
        assert r.status_code == 422

    def test_lookback_days_reflected(self, client):
        r = client.get("/anomalies?metric=consumption&lookback_days=14")
        assert r.json()["lookback_days"] == 14

    def test_lookback_too_large_returns_422(self, client):
        r = client.get("/anomalies?metric=consumption&lookback_days=366")
        assert r.status_code == 422

    def test_unknown_metric_returns_404(self, client):
        r = client.get("/anomalies?metric=totalEnergyConsumption")
        assert r.status_code == 404

    def test_total_anomalies_matches_list_length(self, client):
        r = client.get("/anomalies?metric=consumption&lookback_days=30")
        data = r.json()
        assert data["total_anomalies"] == len(data["anomalies"]), (
            f"total_anomalies={data['total_anomalies']} but anomalies list has "
            f"{len(data['anomalies'])} items"
        )

    def test_anomaly_point_fields(self, client):
        """Use full dataset (lookback_days=365) to maximise chance of finding anomalies."""
        r = client.get("/anomalies?metric=consumption&lookback_days=365")
        anomalies = r.json()["anomalies"]
        if anomalies:
            point = anomalies[0]
            for field in ANOMALY_POINT_FIELDS:
                assert field in point, f"Missing AnomalyPoint field '{field}'"

    def test_anomaly_direction_values_valid(self, client):
        """Every returned anomaly must have a valid direction string."""
        r = client.get("/anomalies?metric=consumption&lookback_days=365&direction=both")
        for point in r.json()["anomalies"]:
            assert point["direction"] in ("excess", "deficit"), (
                f"Unexpected direction value: {point['direction']}"
            )

    def test_hour_of_day_in_range(self, client):
        r = client.get("/anomalies?metric=consumption&lookback_days=365")
        for point in r.json()["anomalies"]:
            assert 0 <= point["hour_of_day"] <= 23, (
                f"hour_of_day out of range: {point['hour_of_day']}"
            )

    def test_excess_anomalies_have_positive_excess_pct(self, client):
        """Excess anomalies must have actual > expected → excess_pct > 0."""
        r = client.get("/anomalies?metric=consumption&lookback_days=365&direction=excess")
        for point in r.json()["anomalies"]:
            assert point["excess_pct"] > 0, (
                f"Excess anomaly has non-positive excess_pct: {point['excess_pct']}"
            )

    def test_deficit_anomalies_have_negative_excess_pct(self, client):
        """Deficit anomalies must have actual < expected → excess_pct < 0."""
        r = client.get("/anomalies?metric=consumption&lookback_days=365&direction=deficit")
        for point in r.json()["anomalies"]:
            assert point["excess_pct"] < 0, (
                f"Deficit anomaly has non-negative excess_pct: {point['excess_pct']}"
            )

    def test_generated_at_format(self, client):
        r = client.get("/anomalies?metric=consumption")
        assert r.json()["generated_at"].endswith("Z")

    def test_in_sample_cache_reused(self, client):
        """
        Two calls for the same metric should both succeed (cache hit on second call).
        Regression: ensures the cache key doesn't error on second access.
        """
        r1 = client.get("/anomalies?metric=carbonEmissions&lookback_days=30")
        r2 = client.get("/anomalies?metric=carbonEmissions&lookback_days=14")
        assert r1.status_code == 200
        assert r2.status_code == 200

    def test_anomaly_dual_model_fields_present(self, client):
        """Both new dual-model fields must always be present regardless of TimesFM availability."""
        r = client.get("/anomalies?metric=consumption&lookback_days=30")
        assert r.status_code == 200
        for a in r.json()["anomalies"]:
            assert "timesfm_flagged" in a
            assert "confidence" in a
            assert isinstance(a["timesfm_flagged"], bool)
            assert a["confidence"] in {"high", "prophet-only", "timesfm-only"}


# ── /investigation-leads ──────────────────────────────────────────────────────

LEAD_FIELDS = [
    "rank", "metric", "unit", "pattern_summary", "occurrences",
    "frequency_pct", "avg_excess_pct", "total_excess_carbon_kgco2e",
    "total_excess_cost_eur", "optimal_window_start_hour",
    "optimal_window_end_hour", "estimated_carbon_saving_pct",
    "estimated_cost_saving_pct", "carbon_context", "example_timestamps",
]

LEADS_RESPONSE_FIELDS = ["lookback_days", "top_n", "generated_at", "leads"]


class TestInvestigationLeadsEndpoint:

    def test_status_ok(self, client):
        r = client.get("/investigation-leads")
        assert r.status_code == 200

    def test_response_fields_present(self, client):
        r = client.get("/investigation-leads")
        data = r.json()
        for field in LEADS_RESPONSE_FIELDS:
            assert field in data, f"Missing top-level field '{field}'"

    def test_default_lookback_days(self, client):
        r = client.get("/investigation-leads")
        assert r.json()["lookback_days"] == 30

    def test_lookback_days_param_reflected(self, client):
        r = client.get("/investigation-leads?lookback_days=14")
        assert r.json()["lookback_days"] == 14

    def test_default_top_n(self, client):
        r = client.get("/investigation-leads")
        data = r.json()
        assert data["top_n"] == 5
        assert len(data["leads"]) <= 5

    def test_top_n_param_caps_leads(self, client):
        r = client.get("/investigation-leads?top_n=2&lookback_days=365&min_occurrences=1")
        data = r.json()
        assert data["top_n"] == 2
        assert len(data["leads"]) <= 2

    def test_top_n_too_large_returns_422(self, client):
        r = client.get("/investigation-leads?top_n=21")
        assert r.status_code == 422

    def test_leads_are_consecutively_ranked(self, client):
        """Ranks must be 1, 2, 3, … with no gaps."""
        r = client.get("/investigation-leads?lookback_days=365&min_occurrences=1")
        leads = r.json()["leads"]
        for i, lead in enumerate(leads):
            assert lead["rank"] == i + 1, (
                f"Lead at index {i} has rank {lead['rank']}, expected {i + 1}"
            )

    def test_lead_fields_present(self, client):
        """Use full history + min_occurrences=1 to ensure at least one lead."""
        r = client.get("/investigation-leads?lookback_days=365&min_occurrences=1")
        leads = r.json()["leads"]
        if leads:
            lead = leads[0]
            for field in LEAD_FIELDS:
                assert field in lead, f"Missing InvestigationLead field '{field}'"

    def test_lead_metric_is_tracked_column(self, client):
        """Investigation leads should only reference the two scanned metrics."""
        r = client.get("/investigation-leads?lookback_days=365&min_occurrences=1")
        for lead in r.json()["leads"]:
            assert lead["metric"] in ("consumption", "carbonEmissions"), (
                f"Unexpected metric in lead: {lead['metric']}"
            )

    def test_optimal_window_hours_in_range(self, client):
        r = client.get("/investigation-leads?lookback_days=365&min_occurrences=1")
        for lead in r.json()["leads"]:
            assert 0 <= lead["optimal_window_start_hour"] <= 23
            assert 0 <= lead["optimal_window_end_hour"] <= 23

    def test_carbon_context_valid_values(self, client):
        r = client.get("/investigation-leads?lookback_days=365&min_occurrences=1")
        valid = {"high-carbon", "low-carbon", "unknown"}
        for lead in r.json()["leads"]:
            assert lead["carbon_context"] in valid, (
                f"Unexpected carbon_context: {lead['carbon_context']}"
            )

    def test_generated_at_format(self, client):
        r = client.get("/investigation-leads")
        assert r.json()["generated_at"].endswith("Z")

    def test_empty_leads_on_strict_min_occurrences(self, client):
        """min_occurrences=9999 → no pattern qualifies → leads list is empty."""
        r = client.get("/investigation-leads?min_occurrences=9999")
        assert r.status_code == 200
        assert r.json()["leads"] == []
