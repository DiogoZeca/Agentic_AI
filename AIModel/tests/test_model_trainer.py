"""Tests for model_trainer.py — XGBoost and MLP trainer contracts."""

import numpy as np
import pytest

from feature_engineering import load_and_build
from model_trainer import (
    SPARSE_TYPES,
    MLPPowerTrainer,
    XGBoostPowerTrainer,
    cv_interpolation,
    cv_loco,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def features():
    return load_and_build("data/cpu_data.dat")


@pytest.fixture(scope="module")
def X(features):
    return features[0]


@pytest.fixture(scope="module")
def y_reg(features):
    return features[1]


@pytest.fixture(scope="module")
def weights(features):
    return features[3]


@pytest.fixture(scope="module")
def metadata(features):
    return features[4]


@pytest.fixture(scope="module")
def xgb_trainer(features):
    """XGBoost fitted on the full dataset (fast n_estimators for CI)."""
    X, y_reg, _, weights, _ = features
    return XGBoostPowerTrainer(n_estimators=50).fit(X, y_reg, weights)


@pytest.fixture(scope="module")
def mlp_trainer(features):
    """MLP fitted on the full dataset.

    300 epochs is sufficient for the model to learn the correct power scale
    (~114–368 W) from the StandardScaler-normalised input.  Full-batch on
    1 002 rows is ~microseconds per epoch, so 300 epochs takes < 1 s on CPU.
    """
    X, y_reg, _, weights, _ = features
    return MLPPowerTrainer(max_epochs=300, patience=30).fit(X, y_reg, weights)


# ── XGBoostPowerTrainer ────────────────────────────────────────────────────────


class TestXGBoostPowerTrainer:
    def test_fit_returns_self(self, X, y_reg, weights):
        t = XGBoostPowerTrainer(n_estimators=10)
        assert t.fit(X, y_reg, weights) is t

    def test_predict_power_shape(self, xgb_trainer, X):
        preds = xgb_trainer.predict_power(X)
        assert preds.shape == (len(X),)

    def test_predict_power_is_numpy(self, xgb_trainer, X):
        assert isinstance(xgb_trainer.predict_power(X), np.ndarray)

    def test_predict_power_all_positive(self, xgb_trainer, X):
        preds = xgb_trainer.predict_power(X)
        assert (preds > 0).all(), "Some predicted power values are non-positive"

    def test_predict_power_in_plausible_range(self, xgb_trainer, X):
        preds = xgb_trainer.predict_power(X)
        assert preds.min() > 50, "Predictions unexpectedly low (< 50 W)"
        assert preds.max() < 500, "Predictions unexpectedly high (> 500 W)"

    def test_predict_power_monotone_per_cpu_type(self, xgb_trainer, X):
        """Isotonic smoothing in feature engineering guarantees smooth targets;
        the trained model should approximate monotonicity at 0% and 100%."""
        for cpu_type in X["CPUTYPE"].unique():
            mask = X["CPUTYPE"] == cpu_type
            group = X[mask].copy().sort_values("cpu_pct")
            preds = xgb_trainer.predict_power(group)
            # Allow small tolerance for noise in the learned model
            p0 = preds[group["cpu_pct"].values == 0].mean()
            p100 = preds[group["cpu_pct"].values == 100].mean()
            assert p100 > p0, (
                f"{cpu_type}: predicted power at 100% ({p100:.1f}W) "
                f"<= power at 0% ({p0:.1f}W)"
            )

    def test_unfitted_raises(self, X):
        with pytest.raises(RuntimeError, match="fit()"):
            XGBoostPowerTrainer().predict_power(X)

    def test_unknown_cpu_type_does_not_crash(self, xgb_trainer, X):
        """Unseen CPU types are handled gracefully (NaN category → numeric fallback)."""
        X_unknown = X.copy()
        X_unknown["CPUTYPE"] = "never-seen-cpu-9000"
        preds = xgb_trainer.predict_power(X_unknown)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()

    def test_save_load_predictions_match(self, xgb_trainer, X, tmp_path):
        xgb_trainer.save(tmp_path / "xgb")
        loaded = XGBoostPowerTrainer.load(tmp_path / "xgb")
        np.testing.assert_allclose(
            xgb_trainer.predict_power(X),
            loaded.predict_power(X),
            rtol=1e-5,
            err_msg="save/load roundtrip changed predictions",
        )

    def test_save_creates_expected_files(self, xgb_trainer, tmp_path):
        xgb_trainer.save(tmp_path / "xgb")
        assert (tmp_path / "xgb" / "model.ubj").exists()
        assert (tmp_path / "xgb" / "categories.json").exists()


# ── MLPPowerTrainer ────────────────────────────────────────────────────────────


class TestMLPPowerTrainer:
    def test_fit_returns_self(self, X, y_reg, weights):
        t = MLPPowerTrainer(max_epochs=5, patience=3)
        assert t.fit(X, y_reg, weights) is t

    def test_predict_power_shape(self, mlp_trainer, X):
        preds = mlp_trainer.predict_power(X)
        assert preds.shape == (len(X),)

    def test_predict_power_is_numpy(self, mlp_trainer, X):
        assert isinstance(mlp_trainer.predict_power(X), np.ndarray)

    def test_predict_power_non_negative(self, mlp_trainer, X):
        # predict_power clips to >= 0 (physical constraint: power cannot be negative)
        preds = mlp_trainer.predict_power(X)
        assert (preds >= 0).all(), "predict_power must not return negative values"

    def test_predict_power_in_plausible_range(self, mlp_trainer, X):
        preds = mlp_trainer.predict_power(X)
        assert preds.min() > 50, f"MLP min prediction {preds.min():.1f} W is unexpectedly low"
        assert preds.max() < 500, f"MLP max prediction {preds.max():.1f} W is unexpectedly high"

    def test_unfitted_raises(self, X):
        with pytest.raises(RuntimeError, match="fit()"):
            MLPPowerTrainer().predict_power(X)

    def test_unknown_cpu_type_zeros_one_hot(self, mlp_trainer, X):
        """Unseen CPU type → all-zero one-hot → model uses only numeric features."""
        X_unknown = X.copy()
        X_unknown["CPUTYPE"] = "never-seen-cpu-9000"
        preds = mlp_trainer.predict_power(X_unknown)
        assert preds.shape == (len(X),)
        assert np.isfinite(preds).all()

    def test_save_load_predictions_match(self, mlp_trainer, X, tmp_path):
        mlp_trainer.save(tmp_path / "mlp")
        loaded = MLPPowerTrainer.load(tmp_path / "mlp")
        np.testing.assert_allclose(
            mlp_trainer.predict_power(X),
            loaded.predict_power(X),
            rtol=1e-5,
            err_msg="save/load roundtrip changed predictions",
        )

    def test_save_creates_expected_files(self, mlp_trainer, tmp_path):
        mlp_trainer.save(tmp_path / "mlp")
        for fname in ("model.pt", "scaler.pkl", "encoder.pkl", "config.json"):
            assert (tmp_path / "mlp" / fname).exists(), f"Missing {fname}"

    def test_load_config_matches(self, mlp_trainer, tmp_path):
        import json
        mlp_trainer.save(tmp_path / "mlp")
        with open(tmp_path / "mlp" / "config.json") as f:
            config = json.load(f)
        assert config["hidden_sizes"] == [128, 64, 32]
        assert config["dropout"] == 0.1
        assert config["n_input"] == 18


# ── cv_interpolation ───────────────────────────────────────────────────────────


class TestCvInterpolation:
    @pytest.fixture(scope="class")
    def interp_result(self, X, y_reg, weights, metadata):
        return cv_interpolation(
            XGBoostPowerTrainer,
            X, y_reg, weights, metadata,
        )

    def test_returns_three_keys(self, interp_result):
        assert set(interp_result.keys()) == {"rmse", "mae", "spike_f1"}

    def test_rmse_is_positive_finite(self, interp_result):
        assert np.isfinite(interp_result["rmse"])
        assert interp_result["rmse"] > 0

    def test_mae_is_positive_finite(self, interp_result):
        assert np.isfinite(interp_result["mae"])
        assert interp_result["mae"] > 0

    def test_mae_lte_rmse(self, interp_result):
        # MAE <= RMSE always (RMSE penalises large errors more)
        assert interp_result["mae"] <= interp_result["rmse"] + 1e-6

    def test_spike_f1_in_unit_interval(self, interp_result):
        assert 0.0 <= interp_result["spike_f1"] <= 1.0

    def test_rmse_better_than_mean_baseline(self, interp_result, y_reg):
        """Model RMSE must beat the naive 'always predict the global mean' baseline."""
        naive_rmse = float(np.sqrt(np.mean((y_reg.values - y_reg.mean()) ** 2)))
        assert interp_result["rmse"] < naive_rmse, (
            f"Model RMSE {interp_result['rmse']:.2f}W >= naive baseline {naive_rmse:.2f}W"
        )

    def test_rmse_plausible_magnitude(self, interp_result):
        # For a well-fitted power model, RMSE should be < 30 W
        assert interp_result["rmse"] < 30, (
            f"Interpolation RMSE {interp_result['rmse']:.2f}W seems too high"
        )


# ── cv_loco ────────────────────────────────────────────────────────────────────


class TestCvLoco:
    @pytest.fixture(scope="class")
    def loco_result(self, X, y_reg, weights, metadata):
        return cv_loco(
            XGBoostPowerTrainer,
            X, y_reg, weights, metadata,
        )

    def test_has_full_and_sparse_keys(self, loco_result):
        assert set(loco_result.keys()) == {"full_types", "sparse_types"}

    def test_full_types_has_8_cpu_types_plus_mean(self, loco_result):
        keys = set(loco_result["full_types"].keys())
        assert "mean" in keys
        assert len(keys) == 9  # 8 CPU types + "mean"

    def test_sparse_types_has_3_cpu_types_plus_mean(self, loco_result):
        keys = set(loco_result["sparse_types"].keys())
        assert "mean" in keys
        assert len(keys) == 4  # 3 CPU types + "mean"

    def test_sparse_types_are_correct(self, loco_result):
        reported = set(loco_result["sparse_types"].keys()) - {"mean"}
        assert reported == SPARSE_TYPES

    def test_each_entry_has_three_metrics(self, loco_result):
        for group in ("full_types", "sparse_types"):
            for key, metrics in loco_result[group].items():
                assert set(metrics.keys()) == {"rmse", "mae", "spike_f1"}, (
                    f"{group}/{key} missing metric keys"
                )

    def test_all_rmse_positive_finite(self, loco_result):
        for group in ("full_types", "sparse_types"):
            for key, metrics in loco_result[group].items():
                assert np.isfinite(metrics["rmse"]) and metrics["rmse"] > 0, (
                    f"{group}/{key} RMSE is invalid: {metrics['rmse']}"
                )

    def test_all_spike_f1_in_unit_interval(self, loco_result):
        for group in ("full_types", "sparse_types"):
            for key, metrics in loco_result[group].items():
                assert 0.0 <= metrics["spike_f1"] <= 1.0, (
                    f"{group}/{key} spike_f1 out of range: {metrics['spike_f1']}"
                )

    def test_full_loco_rmse_better_than_mean_baseline(self, loco_result, y_reg):
        naive_rmse = float(np.sqrt(np.mean((y_reg.values - y_reg.mean()) ** 2)))
        loco_rmse = loco_result["full_types"]["mean"]["rmse"]
        assert loco_rmse < naive_rmse, (
            f"LOCO RMSE {loco_rmse:.2f}W >= naive baseline {naive_rmse:.2f}W"
        )
