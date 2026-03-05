"""Test 5: Physics Constraint Contract"""
import numpy as np
import pandas as pd
import pytest
from data_generator import generate_energy_carbon_data
from physics_constraint import (
    EMBODIED_EMISSIONS_KGC02E_H,
    check_physical_consistency,
    derive_carbon_emissions,
    evaluate_formula_accuracy,
)


@pytest.fixture(scope="module")
def sample_df():
    return generate_energy_carbon_data(start_date="2024-01-01", periods=24 * 30)


@pytest.fixture(scope="module")
def sample_component_forecasts():
    n = 24
    ds = pd.date_range("2024-02-01", periods=n, freq="h")
    consumption_fc = pd.DataFrame({
        "ds":         ds,
        "yhat":       np.full(n, 0.5),
        "yhat_lower": np.full(n, 0.4),
        "yhat_upper": np.full(n, 0.6),
    })
    cif_fc = pd.DataFrame({
        "ds":         ds,
        "yhat":       np.full(n, 0.4),
        "yhat_lower": np.full(n, 0.35),
        "yhat_upper": np.full(n, 0.45),
    })
    return consumption_fc, cif_fc


class TestPhysicsFormula:

    def test_formula_output_non_negative(self, sample_component_forecasts):
        c_fc, cif_fc = sample_component_forecasts
        result = derive_carbon_emissions(c_fc, cif_fc)
        assert (result["yhat"] >= 0).all()
        assert (result["yhat_lower"] >= 0).all()
        assert (result["yhat_upper"] >= 0).all()

    def test_formula_uncertainty_ordered(self, sample_component_forecasts):
        c_fc, cif_fc = sample_component_forecasts
        result = derive_carbon_emissions(c_fc, cif_fc)
        for _, row in result.iterrows():
            assert row["yhat_lower"] <= row["yhat"] <= row["yhat_upper"]

    def test_formula_evaluate_returns_required_keys(self, sample_df):
        required = {"MAE", "MSE", "RMSE", "MAPE", "sMAPE", "train_size", "test_size"}
        result = evaluate_formula_accuracy(sample_df)
        assert required == set(result.keys())

    def test_formula_smape_is_finite(self, sample_df):
        result = evaluate_formula_accuracy(sample_df)
        assert np.isfinite(result["sMAPE"])
        assert result["sMAPE"] > 0

    def test_physical_consistency_check(self, sample_component_forecasts):
        c_fc, cif_fc = sample_component_forecasts
        result = derive_carbon_emissions(c_fc, cif_fc)
        info = check_physical_consistency(result)
        assert info["is_non_negative"] is True
        assert info["negative_count"] == 0
        assert info["interval_ordered"] is True

    def test_evaluate_returns_python_floats(self, sample_df):
        """All numeric values in evaluate() must be Python floats, not numpy scalars."""
        result = evaluate_formula_accuracy(sample_df)
        for key, val in result.items():
            if key not in ("train_size", "test_size"):
                assert isinstance(val, float), f"{key} returned {type(val)}, expected float"

    def test_evaluate_zero_actual_smape_stays_finite(self):
        """sMAPE must stay finite when actuals contain zeros (denominator is safe)."""
        n = 24 * 30
        ds = pd.date_range("2024-01-01", periods=n, freq="h")
        df_zeros = pd.DataFrame({
            "ds": ds,
            "consumption": np.zeros(n),
            "carbonEmissions": np.zeros(n),
            "carbonIntensityFactor": np.full(n, 0.3),
        })
        result = evaluate_formula_accuracy(df_zeros)
        assert np.isfinite(result["sMAPE"])
        assert not np.isinf(result.get("MAPE", 0.0) or 0.0)  # MAPE may be nan, never inf

    def test_smape_uses_consistent_epsilon(self, sample_df):
        """evaluate_formula_accuracy sMAPE must be finite and positive on normal data."""
        result = evaluate_formula_accuracy(sample_df)
        assert result["sMAPE"] > 0
        assert result["sMAPE"] < 100   # sanity upper bound for synthetic data
