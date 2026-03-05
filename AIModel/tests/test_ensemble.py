"""
Test 6: Ensemble Model Contract

Covers the pure-function components of EnsembleForecaster without fitting any
real models. These tests run in milliseconds and execute in the lightweight
test image (no TimesFM / PyTorch required):

  TestComputeMetrics    — _compute_metrics() correctness, type safety, edge cases
  TestApplyCorrection   — ±50% clamp behaviour and alpha weighting
  TestLearnAlpha        — grid-search selects correct alpha in clear-cut cases
  TestResidualColName   — naming convention for residual columns

Integration tests that require TimesFM (skipped in the lightweight image):

  TestEvaluateContract  — full evaluate() output structure and value sanity

Design: EnsembleForecaster.__init__ instantiates EnergyTimesFM, which imports
torch and timesfm at module level. In environments without those packages, we
stub the import chain in sys.modules before loading ensemble_model — the same
technique pandas uses for optional backends such as pyarrow.

See: pandas/tests/io/test_parquet.py (importorskip pattern) and
     Prophet's conftest.py (sys.modules patching for optional Stan backends).
"""
from __future__ import annotations

import importlib.util
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# ── Check real TimesFM availability BEFORE any sys.modules patching ───────────
# importlib.util.find_spec does not import the package — safe to call first.
_TIMESFM_INSTALLED = importlib.util.find_spec("timesfm") is not None

# ── Stub heavy optional dependencies ──────────────────────────────────────────
# torch and timesfm are top-level imports in timesfm_model.py (lines 11–12).
# Without this patch, `import ensemble_model` fails in the lightweight image
# with ModuleNotFoundError even though ensemble_model itself has no heavy deps.
# We also stub timesfm_model itself so ensemble_model's module-level import
# `from timesfm_model import EnergyTimesFM` resolves to a MagicMock, making
# EnsembleForecaster.__init__ safe to call without a real GPU backend.
for _stub in ("torch", "timesfm", "timesfm_model"):
    sys.modules.setdefault(_stub, MagicMock())

from ensemble_model import EnsembleForecaster  # noqa: E402


# ── Factories ─────────────────────────────────────────────────────────────────

def _make_arrays(n: int = 100, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return (actuals, predictions) with small realistic noise."""
    rng = np.random.default_rng(seed)
    actuals     = rng.uniform(0.05, 0.20, n)
    predictions = actuals + rng.normal(0, 0.005, n)
    return actuals, predictions


def _make_forecaster(alpha: float = 1.0) -> EnsembleForecaster:
    """
    Return an EnsembleForecaster with a known alpha, bypassing real model loads.
    __init__ runs normally; EnergyTimesFM is a MagicMock so no GPU is needed.
    """
    ens = EnsembleForecaster()
    ens._alpha = alpha
    return ens


# ── TestComputeMetrics ────────────────────────────────────────────────────────

class TestComputeMetrics:
    """
    _compute_metrics is a @staticmethod — call it directly on the class.
    Tests mirror the contract enforced across prophet_model, timesfm_model,
    and physics_constraint: Python floats, zero-guard on MAPE, finite sMAPE.
    """

    _REQUIRED_KEYS = {"MAE", "MSE", "RMSE", "MAPE", "sMAPE", "train_size", "test_size"}

    def _call(
        self,
        actuals: np.ndarray,
        predictions: np.ndarray,
        train_size: int = 80,
        test_size: int = 20,
    ) -> dict:
        return EnsembleForecaster._compute_metrics(actuals, predictions, train_size, test_size)

    def test_returns_all_required_keys(self):
        actuals, preds = _make_arrays()
        result = self._call(actuals, preds)
        assert set(result.keys()) == self._REQUIRED_KEYS

    def test_all_numeric_values_are_python_floats(self):
        """No numpy scalars — JSON serialisation safety."""
        actuals, preds = _make_arrays()
        result = self._call(actuals, preds)
        for key, val in result.items():
            if key not in ("train_size", "test_size"):
                assert isinstance(val, float), (
                    f"{key} returned {type(val).__name__}, expected float"
                )

    def test_mape_is_nan_when_actuals_contain_zero(self):
        """Zero-guard: division by zero → nan, not inf or crash."""
        actuals = np.array([0.1, 0.0, 0.2])
        preds   = np.array([0.1, 0.1, 0.2])
        result  = self._call(actuals, preds)
        assert np.isnan(result["MAPE"]), "MAPE must be nan when actuals contain 0"

    def test_mape_is_not_inf_when_actuals_contain_zero(self):
        """Zero-guard must produce nan, never inf."""
        actuals = np.array([0.0, 0.0, 0.0])
        preds   = np.array([0.1, 0.1, 0.1])
        result  = self._call(actuals, preds)
        assert not np.isinf(result.get("MAPE", 0.0) or 0.0)

    def test_mape_is_finite_on_normal_data(self):
        actuals, preds = _make_arrays()
        result = self._call(actuals, preds)
        assert np.isfinite(result["MAPE"])

    def test_smape_is_finite_and_positive(self):
        actuals, preds = _make_arrays()
        result = self._call(actuals, preds)
        assert np.isfinite(result["sMAPE"])
        assert result["sMAPE"] > 0

    def test_smape_is_symmetric(self):
        """
        sMAPE is symmetric by definition: swapping actuals ↔ predictions
        must return the same value. This is the core distinction from MAPE.
        """
        actuals, preds = _make_arrays()
        forward  = self._call(actuals, preds)["sMAPE"]
        backward = self._call(preds, actuals)["sMAPE"]
        assert forward == pytest.approx(backward, rel=1e-9)

    def test_perfect_predictions_give_zero_errors(self):
        actuals = np.array([0.10, 0.20, 0.15, 0.18])
        preds   = actuals.copy()
        result  = self._call(actuals, preds)
        for key in ("MAE", "MSE", "RMSE", "MAPE", "sMAPE"):
            assert result[key] == pytest.approx(0.0, abs=1e-12), (
                f"{key} should be 0 for perfect predictions, got {result[key]}"
            )

    def test_train_test_size_passthrough(self):
        """train_size and test_size are passed through, not recomputed."""
        actuals, preds = _make_arrays()
        result = self._call(actuals, preds, train_size=576, test_size=144)
        assert result["train_size"] == 576
        assert result["test_size"]  == 144

    def test_mae_formula(self):
        """Spot-check MAE against manual calculation."""
        actuals = np.array([1.0, 2.0, 3.0])
        preds   = np.array([1.5, 1.5, 3.5])
        # |1.0-1.5| + |2.0-1.5| + |3.0-3.5| = 0.5 + 0.5 + 0.5 → mean = 0.5
        result = self._call(actuals, preds)
        assert result["MAE"] == pytest.approx(0.5, rel=1e-9)

    def test_rmse_is_sqrt_of_mse(self):
        actuals, preds = _make_arrays()
        result = self._call(actuals, preds)
        assert result["RMSE"] == pytest.approx(result["MSE"] ** 0.5, rel=1e-9)


# ── TestApplyCorrection ───────────────────────────────────────────────────────

class TestApplyCorrection:
    """
    _apply_correction implements: clip(alpha * residual, ±50% * |prophet_yhat|)
    Tests cover the clamp boundaries and alpha scaling.
    """

    def test_alpha_zero_suppresses_all_corrections(self):
        ens      = _make_forecaster(alpha=0.0)
        prophet  = np.array([0.10, 0.20, 0.15])
        residual = np.array([0.05, -0.08, 0.12])
        correction = ens._apply_correction(prophet, residual)
        np.testing.assert_array_almost_equal(correction, np.zeros(3))

    def test_small_correction_passes_through_unchanged(self):
        """Correction within ±50% of prophet_yhat is not clamped."""
        ens      = _make_forecaster(alpha=1.0)
        prophet  = np.array([0.10])
        residual = np.array([0.01])  # 10% of 0.10 — well within the 50% bound
        correction = ens._apply_correction(prophet, residual)
        assert correction[0] == pytest.approx(0.01, rel=1e-9)

    def test_large_positive_correction_clamped_at_50pct(self):
        """Correction > 50% of |prophet_yhat| must be capped."""
        ens      = _make_forecaster(alpha=1.0)
        prophet  = np.array([0.10])
        residual = np.array([0.20])   # 200% of prophet → clamped to 0.05
        correction = ens._apply_correction(prophet, residual)
        assert correction[0] == pytest.approx(0.05, rel=1e-9)

    def test_large_negative_correction_clamped_at_minus_50pct(self):
        ens      = _make_forecaster(alpha=1.0)
        prophet  = np.array([0.10])
        residual = np.array([-0.20])  # -200% of prophet → clamped to -0.05
        correction = ens._apply_correction(prophet, residual)
        assert correction[0] == pytest.approx(-0.05, rel=1e-9)

    def test_zero_prophet_gives_zero_correction(self):
        """max_correction = 0.5 * |0| = 0 → correction is always 0."""
        ens      = _make_forecaster(alpha=1.0)
        prophet  = np.array([0.0, 0.0])
        residual = np.array([0.5, -0.3])
        correction = ens._apply_correction(prophet, residual)
        np.testing.assert_array_almost_equal(correction, np.zeros(2))

    def test_alpha_halves_the_correction(self):
        """alpha=0.5 halves a correction that is within the clamp bounds."""
        prophet  = np.array([1.0])
        residual = np.array([0.2])   # 20% of 1.0 — within the 50% bound
        full = _make_forecaster(alpha=1.0)._apply_correction(prophet, residual)
        half = _make_forecaster(alpha=0.5)._apply_correction(prophet, residual)
        assert half[0] == pytest.approx(full[0] * 0.5, rel=1e-9)


# ── TestLearnAlpha ────────────────────────────────────────────────────────────

class TestLearnAlpha:
    """
    _learn_alpha grid-searches alpha in [0, 1] (step 0.01) to minimise MAE
    on a validation set. Two extreme cases have deterministic outcomes:
      - perfect residuals → alpha > 0 (correction genuinely helps)
      - random noise     → alpha = 0  (correction only hurts)
    """

    def _ens(self) -> EnsembleForecaster:
        return _make_forecaster()

    def test_perfect_residuals_select_nonzero_alpha(self):
        """
        When the residual model perfectly predicts Prophet's error,
        applying the correction (alpha > 0) must minimise MAE.
        """
        n              = 100
        prophet_preds  = np.full(n, 0.10)
        true_error     = np.full(n, 0.02)    # prophet consistently 0.02 too low
        actuals        = prophet_preds + true_error
        residual_preds = true_error           # residual model is perfect
        alpha = self._ens()._learn_alpha(prophet_preds, residual_preds, actuals)
        assert alpha > 0.0, f"Perfect residuals should yield alpha > 0, got {alpha}"

    def test_pure_noise_residuals_select_alpha_zero(self):
        """
        When Prophet is already perfect and residuals are noise,
        any correction worsens the forecast — alpha=0 is optimal.
        """
        rng            = np.random.default_rng(0)
        n              = 200
        actuals        = np.full(n, 0.10)
        prophet_preds  = np.full(n, 0.10)          # prophet is perfect
        residual_preds = rng.uniform(-1.0, 1.0, n)  # uncorrelated noise
        alpha = self._ens()._learn_alpha(prophet_preds, residual_preds, actuals)
        assert alpha == pytest.approx(0.0, abs=1e-9), (
            f"Noisy residuals should yield alpha=0, got {alpha}"
        )

    def test_alpha_always_in_valid_range(self):
        """alpha must be in [0, 1] regardless of input distributions."""
        rng = np.random.default_rng(7)
        n   = 50
        for _ in range(10):
            prophet  = rng.uniform(0.05, 0.20, n)
            residual = rng.uniform(-0.10, 0.10, n)
            actuals  = rng.uniform(0.05, 0.20, n)
            alpha = self._ens()._learn_alpha(prophet, residual, actuals)
            assert 0.0 <= alpha <= 1.0, f"alpha out of range: {alpha}"

    def test_alpha_rounded_to_two_decimal_places(self):
        """_learn_alpha returns round(alpha, 2) — verify the rounding contract."""
        rng     = np.random.default_rng(99)
        n       = 80
        prophet  = rng.uniform(0.05, 0.20, n)
        residual = rng.uniform(-0.05, 0.05, n)
        actuals  = rng.uniform(0.05, 0.20, n)
        alpha = self._ens()._learn_alpha(prophet, residual, actuals)
        assert alpha == round(alpha, 2), (
            f"alpha has more than 2 decimal places: {alpha}"
        )


# ── TestResidualColName ───────────────────────────────────────────────────────

class TestResidualColName:

    def test_col_name_follows_convention(self):
        assert EnsembleForecaster._residual_col_name("consumption") == "consumption_residual"
        assert EnsembleForecaster._residual_col_name("carbonEmissions") == "carbonEmissions_residual"


# ── TestEvaluateContract ──────────────────────────────────────────────────────
# Skipped in the lightweight test image — TimesFM / PyTorch not installed.
# Runs locally or in the full service image where timesfm[torch] is available.

@pytest.mark.skipif(not _TIMESFM_INSTALLED, reason="timesfm not installed — skipped in lightweight test image")
class TestEvaluateContract:
    """
    Full evaluate() integration test. Requires a real TimesFM installation.
    Uses the session-scoped `small_df` fixture (30-day, 720 rows) from conftest.
    """

    _REQUIRED_KEYS     = {"MAE", "MSE", "RMSE", "MAPE", "sMAPE", "train_size", "test_size"}
    _REQUIRED_MODELS   = {"prophet", "timesfm", "ensemble"}

    @pytest.fixture(scope="class")
    def ensemble_results(self, small_df):
        ens = EnsembleForecaster()
        return ens, ens.evaluate(small_df, "consumption")

    def test_returns_three_model_keys(self, ensemble_results):
        _, results = ensemble_results
        assert set(results.keys()) == self._REQUIRED_MODELS

    def test_each_model_has_seven_metric_keys(self, ensemble_results):
        _, results = ensemble_results
        for model_name in self._REQUIRED_MODELS:
            assert set(results[model_name].keys()) == self._REQUIRED_KEYS, (
                f"'{model_name}' is missing metric keys"
            )

    def test_all_metric_values_are_python_floats(self, ensemble_results):
        _, results = ensemble_results
        for model_name, metrics in results.items():
            for key, val in metrics.items():
                if key not in ("train_size", "test_size"):
                    assert isinstance(val, float), (
                        f"{model_name}.{key} returned {type(val).__name__}, expected float"
                    )

    def test_smape_in_reasonable_range(self, ensemble_results):
        """sMAPE must be between 0% and 100% for all three models on real data."""
        _, results = ensemble_results
        for model_name in self._REQUIRED_MODELS:
            smape = results[model_name]["sMAPE"]
            assert 0 < smape < 100, (
                f"{model_name}.sMAPE = {smape:.2f}% is outside the expected range"
            )

    def test_evaluate_stores_test_attributes(self, ensemble_results):
        """evaluate() must expose _test_actuals, _test_prophet, etc. for analysis scripts."""
        ens, _ = ensemble_results
        for attr in ("_test_actuals", "_test_prophet", "_test_timesfm", "_test_ensemble"):
            assert hasattr(ens, attr), f"EnsembleForecaster missing attribute '{attr}'"
            assert isinstance(getattr(ens, attr), np.ndarray), (
                f"'{attr}' should be a numpy array"
            )
