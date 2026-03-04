"""
Test 1: Data Contract

Verifies:
  - generate_energy_carbon_data() output shape, column names, and value ranges
  - load_and_validate() accepts the generated schema without errors
  - build_pipeline_config() detects 7 fit_metrics and has_sci=True
  - SCI identity: carbonEmissions / max(functionalUnit, 1) ≈ softwareCarbonIntensity
  - Cumulative columns are monotonically non-decreasing
  - Missing required column causes a clean SystemExit (not a crash)
"""
import pandas as pd
import pytest

from data_generator import generate_energy_carbon_data
from data_loader import CANDIDATE_FIT_METRICS, build_pipeline_config, load_and_validate

# All 15 columns defined by the SCI schema
EXPECTED_COLUMNS = [
    "ds",
    "timeWindow",
    "consumption",
    "totalConsumption",
    "measurementSource",
    "cost",
    "totalCost",
    "carbonEmissions",
    "operationalEmissions",
    "embodiedEmissions",
    "carbonIntensityFactor",
    "greenConsumptionPercentage",
    "softwareCarbonIntensity",
    "functionalUnit",
    "cpuUtilization",
]

# All 7 metrics that should be directly fit by Prophet
EXPECTED_FIT_METRICS = {
    "consumption",
    "carbonEmissions",
    "carbonIntensityFactor",
    "greenConsumptionPercentage",
    "cpuUtilization",
    "functionalUnit",
    "cost",
}


class TestDataGenerator:
    """Validate that the synthetic data generator produces the correct schema."""

    def test_default_shape(self):
        """Full year = 8,760 hourly rows × 15 columns."""
        df = generate_energy_carbon_data()
        assert df.shape == (8760, 15), f"Expected (8760, 15), got {df.shape}"

    def test_all_columns_present(self, small_df):
        missing = set(EXPECTED_COLUMNS) - set(small_df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_no_extra_columns(self, small_df):
        extra = set(small_df.columns) - set(EXPECTED_COLUMNS)
        assert not extra, f"Unexpected extra columns: {extra}"

    def test_consumption_range(self, small_df):
        """Server model: (20W + 130W × cpu) × PUE 1.3 → 0.034–0.187 kWh/h."""
        assert small_df["consumption"].min() >= 0.030, "Consumption below expected minimum"
        assert small_df["consumption"].max() <= 0.200, "Consumption above expected maximum"

    def test_cpu_utilization_range(self, small_df):
        """clipped to [0.05, 0.95]"""
        assert small_df["cpuUtilization"].min() >= 0.05
        assert small_df["cpuUtilization"].max() <= 0.95

    def test_carbon_intensity_range(self, small_df):
        """clipped to [0.30, 0.50] kgCO2/kWh (allow tiny float rounding)"""
        assert small_df["carbonIntensityFactor"].min() >= 0.29
        assert small_df["carbonIntensityFactor"].max() <= 0.51

    def test_green_percentage_range(self, small_df):
        """clipped to [20%, 70%] (allow tiny float rounding)"""
        assert small_df["greenConsumptionPercentage"].min() >= 19.0
        assert small_df["greenConsumptionPercentage"].max() <= 71.0

    def test_functional_unit_positive(self, small_df):
        """Requests/hour must always be positive (minimum clip is 5.0)."""
        assert (small_df["functionalUnit"] > 0).all()
        assert small_df["functionalUnit"].min() >= 4.9  # allow rounding

    def test_no_nan_in_required_columns(self, small_df):
        for col in ["ds", "consumption", "carbonEmissions", "functionalUnit"]:
            n_nan = small_df[col].isna().sum()
            assert n_nan == 0, f"Found {n_nan} NaN values in '{col}'"

    def test_sci_identity(self, small_df):
        """
        SCI stored in column = carbonEmissions / max(functionalUnit, 1.0).

        Components are rounded before storage, so the identity holds within
        the rounding error introduced by `np.round(..., 6)` on inputs.
        We accept a generous tolerance of 5e-5 kgCO2e/req.
        """
        recomputed = small_df["carbonEmissions"] / small_df["functionalUnit"].clip(lower=1.0)
        max_err = (recomputed - small_df["softwareCarbonIntensity"]).abs().max()
        assert max_err < 5e-5, (
            f"SCI identity violated — max absolute error: {max_err:.2e} kgCO2e/req"
        )

    def test_emissions_components_add_up(self, small_df):
        """carbonEmissions = operationalEmissions + embodiedEmissions (within rounding)."""
        expected = small_df["operationalEmissions"] + small_df["embodiedEmissions"]
        max_err = (expected - small_df["carbonEmissions"]).abs().max()
        assert max_err < 1e-5, (
            f"Emissions identity violated — max error: {max_err:.2e} kgCO2e/h"
        )

    def test_cumulative_total_consumption_monotonic(self, small_df):
        diffs = small_df["totalConsumption"].diff().dropna()
        assert (diffs >= -1e-9).all(), "totalConsumption is not monotonically non-decreasing"

    def test_cumulative_total_cost_monotonic(self, small_df):
        diffs = small_df["totalCost"].diff().dropna()
        assert (diffs >= -1e-9).all(), "totalCost is not monotonically non-decreasing"

    def test_measurement_source_valid_values(self, small_df):
        valid = {"RAPL", "TDP"}
        found = set(small_df["measurementSource"].unique())
        assert found.issubset(valid), f"Unexpected measurementSource values: {found - valid}"

    def test_time_window_constant(self, small_df):
        """timeWindow = 3600 seconds (1 hour) for every row."""
        assert (small_df["timeWindow"] == 3600).all()

    def test_ds_is_hourly(self, small_df):
        """Timestamps must be exactly 1 hour apart."""
        diffs = small_df["ds"].diff().dropna()
        assert (diffs == pd.Timedelta("1h")).all(), "Non-hourly gaps found in 'ds'"


class TestDataLoader:
    """Validate schema enforcement and pipeline config auto-detection."""

    def test_load_and_validate_succeeds(self, test_csv_path):
        df = load_and_validate(test_csv_path)
        assert len(df) == 720, f"Expected 720 rows, got {len(df)}"
        assert "ds" in df.columns
        assert "consumption" in df.columns

    def test_pipeline_config_fit_metrics_count(self, test_csv_path):
        df = load_and_validate(test_csv_path)
        config = build_pipeline_config(df)
        assert len(config.fit_metrics) == 7, (
            f"Expected 7 fit metrics, got {len(config.fit_metrics)}: {config.fit_metrics}"
        )

    def test_pipeline_config_fit_metrics_names(self, test_csv_path):
        df = load_and_validate(test_csv_path)
        config = build_pipeline_config(df)
        assert set(config.fit_metrics) == EXPECTED_FIT_METRICS, (
            f"Fit metric mismatch.\n  Got:      {set(config.fit_metrics)}\n"
            f"  Expected: {EXPECTED_FIT_METRICS}"
        )

    def test_pipeline_config_has_sci(self, test_csv_path):
        df = load_and_validate(test_csv_path)
        config = build_pipeline_config(df)
        assert config.has_sci is True, (
            "has_sci should be True when both carbonEmissions and functionalUnit are present"
        )

    def test_pipeline_config_regressor_map(self, test_csv_path):
        df = load_and_validate(test_csv_path)
        config = build_pipeline_config(df)
        # consumption uses functionalUnit as a regressor
        assert config.regressor_map.get("consumption") == ["functionalUnit"], (
            f"Expected consumption regressor=['functionalUnit'], "
            f"got {config.regressor_map.get('consumption')}"
        )
        # carbonEmissions uses carbonIntensityFactor as a regressor (direct physical driver)
        assert config.regressor_map.get("carbonEmissions") == ["carbonIntensityFactor"], (
            f"Expected carbonEmissions regressor=['carbonIntensityFactor'], "
            f"got {config.regressor_map.get('carbonEmissions')}"
        )

    def test_pipeline_config_candidate_fit_metrics_complete(self):
        """CANDIDATE_FIT_METRICS must list exactly the 7 expected metrics."""
        assert set(CANDIDATE_FIT_METRICS) == EXPECTED_FIT_METRICS

    def test_missing_required_column_raises_system_exit(self, tmp_path):
        """load_and_validate must exit cleanly if 'consumption' is absent."""
        bad_df = pd.DataFrame({
            "ds":   pd.date_range("2024-01-01", periods=10, freq="h"),
            "energy": [1.0] * 10,  # wrong column name
        })
        bad_csv = str(tmp_path / "bad.csv")
        bad_df.to_csv(bad_csv, index=False)
        with pytest.raises(SystemExit):
            load_and_validate(bad_csv)

    def test_missing_file_raises_system_exit(self):
        with pytest.raises(SystemExit):
            load_and_validate("/nonexistent/path/data.csv")
