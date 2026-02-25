"""
Ensemble Model: Prophet + TimesFM Residual Learning

Two-stage pipeline:
  1. Prophet captures interpretable patterns (trend, daily, weekly seasonality)
     plus known regressors (e.g., training schedule)
  2. TimesFM learns complex patterns in Prophet's residuals (prediction errors)
  3. Final forecast = Prophet forecast + alpha * TimesFM residual correction

The weight alpha is learned from a validation split to prevent the residual
correction from making things worse. Corrections are clamped for stability.
"""
import logging

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from prophet_model import EnergyProphet
from timesfm_model import EnergyTimesFM

logger = logging.getLogger(__name__)


class EnsembleForecaster:
    """
    Ensemble forecaster combining Prophet and TimesFM.

    Prophet handles the "expected" (trend + seasonality + known regressors).
    TimesFM handles the "unexpected" (complex patterns in residuals).

    The residual correction is weighted by an alpha learned from a validation
    split within the training data, preventing TimesFM from adding noise.

    Usage:
        ensemble = EnsembleForecaster(regressors=["trainingActive"])
        results = ensemble.evaluate(df, "totalEnergyConsumption")
    """

    def __init__(
        self,
        prophet_kwargs: Optional[dict] = None,
        timesfm_kwargs: Optional[dict] = None,
        regressors: Optional[List[str]] = None,
    ):
        self.prophet_kwargs = prophet_kwargs or {}
        self.timesfm_kwargs = timesfm_kwargs or {}
        self.regressors = regressors or []

        # Inject regressors into Prophet kwargs
        prophet_kw = {**self.prophet_kwargs, "regressors": self.regressors}
        self.prophet = EnergyProphet(**prophet_kw)
        self.timesfm_standalone = EnergyTimesFM(**self.timesfm_kwargs)
        self.timesfm_residual = EnergyTimesFM(**self.timesfm_kwargs)

        self.target_column: Optional[str] = None
        self._residuals: Optional[np.ndarray] = None
        self._alpha: float = 1.0  # residual correction weight

    def fit(self, df: pd.DataFrame, target_column: str) -> "EnsembleForecaster":
        """
        Fit the ensemble: train Prophet, then prepare residuals for TimesFM.

        Args:
            df: Input DataFrame with 'ds' column
            target_column: Column to forecast

        Returns:
            self for method chaining
        """
        self.target_column = target_column

        # Stage 1: Fit Prophet (with regressors if configured)
        self.prophet.fit(df, target_column)

        # Get Prophet's fitted values on training data
        prophet_fitted = self.prophet.predict(periods=0, future_df=df)
        prophet_values = prophet_fitted["yhat"].values
        actual_values = df[target_column].values[:len(prophet_values)]

        # Stage 2: Compute residuals
        self._residuals = actual_values - prophet_values

        # Stage 3: Prepare TimesFM on residuals
        residual_df = pd.DataFrame({
            "ds": df["ds"].iloc[:len(self._residuals)],
            self._residual_col: self._residuals,
        })
        self.timesfm_residual.fit(residual_df, self._residual_col)

        # Also fit standalone TimesFM on raw data for comparison
        self.timesfm_standalone.fit(df, target_column)

        return self

    @property
    def _residual_col(self) -> str:
        return self._residual_col_name(self.target_column)

    def predict(self, periods: int, freq: str = "h", future_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate ensemble forecast: Prophet + alpha * TimesFM residual correction.

        Args:
            periods: Number of periods to forecast
            freq: Frequency
            future_df: Optional DataFrame with regressor values for future period

        Returns:
            DataFrame with ds, yhat (ensemble), prophet_yhat, timesfm_residual,
            yhat_lower, yhat_upper
        """
        # Prophet forecast
        prophet_forecast = self.prophet.predict(periods, freq, future_df=future_df)
        prophet_future = prophet_forecast.tail(periods)

        # TimesFM residual forecast
        residual_forecast = self.timesfm_residual.predict(periods, freq)
        residual_future = residual_forecast.tail(periods)

        # Apply weighted + clamped residual correction
        prophet_yhat = prophet_future["yhat"].values
        raw_correction = residual_future["yhat"].values
        correction = self._apply_correction(prophet_yhat, raw_correction)

        ensemble_yhat = prophet_yhat + correction

        return pd.DataFrame({
            "ds": prophet_future["ds"].values,
            "yhat": ensemble_yhat,
            "prophet_yhat": prophet_yhat,
            "timesfm_residual": correction,
            "yhat_lower": prophet_future["yhat_lower"].values + residual_future["yhat_lower"].values * self._alpha,
            "yhat_upper": prophet_future["yhat_upper"].values + residual_future["yhat_upper"].values * self._alpha,
        })

    def _apply_correction(self, prophet_yhat: np.ndarray, raw_correction: np.ndarray) -> np.ndarray:
        """Apply alpha weighting and clamp corrections to ±50% of Prophet's prediction."""
        weighted = self._alpha * raw_correction
        max_correction = 0.5 * np.abs(prophet_yhat)
        return np.clip(weighted, -max_correction, max_correction)

    def _learn_alpha(
        self,
        prophet_preds: np.ndarray,
        residual_preds: np.ndarray,
        actuals: np.ndarray,
    ) -> float:
        """
        Find the optimal alpha that minimizes MAE on validation data.
        Tests alpha in [0.0, 0.01, 0.02, ..., 1.0] and picks the best.
        alpha=0 means "ignore TimesFM corrections entirely".
        """
        best_alpha = 0.0
        best_mae = float("inf")

        for alpha in np.arange(0.0, 1.01, 0.01):
            correction = alpha * residual_preds
            max_corr = 0.5 * np.abs(prophet_preds)
            clamped = np.clip(correction, -max_corr, max_corr)
            ensemble = prophet_preds + clamped
            mae = np.mean(np.abs(actuals - ensemble))
            if mae < best_mae:
                best_mae = mae
                best_alpha = alpha

        return round(best_alpha, 2)

    def evaluate(
        self,
        df: pd.DataFrame,
        target_column: str,
        train_ratio: float = 0.8,
    ) -> Dict[str, Dict[str, float]]:
        """
        Three-way evaluation: Prophet vs TimesFM vs Ensemble.

        Uses the same train/test split for fair comparison.
        Learns optimal alpha from a validation portion within training data.

        Args:
            df: Full dataset
            target_column: Column to forecast
            train_ratio: Proportion of data for training

        Returns:
            Dict with keys 'prophet', 'timesfm', 'ensemble',
            each containing MAE, MSE, RMSE, MAPE, sMAPE metrics
        """
        self.target_column = target_column
        n = len(df)
        train_size = int(n * train_ratio)
        test_size = n - train_size
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        actuals = test_df[target_column].values

        # 1. Prophet: fit on training data, predict test period
        #    (regressors are injected via __init__)
        self.prophet.fit(train_df, target_column)
        prophet_forecast = self.prophet.predict(periods=test_size, future_df=df)
        prophet_preds = prophet_forecast.iloc[train_size:]["yhat"].values

        # 2. Compute residuals on training data
        prophet_train_fitted = self.prophet.predict(periods=0, future_df=train_df)
        prophet_train_values = prophet_train_fitted["yhat"].values
        actual_train = train_df[target_column].values[:len(prophet_train_values)]
        residuals = actual_train - prophet_train_values

        # 3. TimesFM on residuals: forecast test-period residuals
        residual_col = self._residual_col_name(target_column)
        residual_df = pd.DataFrame({
            "ds": train_df["ds"].iloc[:len(residuals)],
            residual_col: residuals,
        })
        self.timesfm_residual.fit(residual_df, residual_col)
        residual_forecast = self.timesfm_residual.predict(periods=test_size)
        residual_preds = residual_forecast.tail(test_size)["yhat"].values

        # 4. Learn optimal alpha using validation split within training
        # Use last 10% of training as validation for alpha selection
        val_ratio = 0.1
        val_size = int(n * val_ratio)
        inner_train_size = train_size - val_size

        if val_size > 48:  # need enough validation data
            inner_train_df = df.iloc[:inner_train_size]
            val_df = df.iloc[inner_train_size:train_size]
            val_actuals = val_df[target_column].values

            # Fit Prophet on inner training, predict validation period
            prophet_val = EnergyProphet(**{**self.prophet_kwargs, "regressors": self.regressors})
            prophet_val.fit(inner_train_df, target_column)
            prophet_val_forecast = prophet_val.predict(periods=val_size, future_df=df)
            prophet_val_preds = prophet_val_forecast.iloc[inner_train_size:train_size]["yhat"].values

            # Compute inner residuals and forecast them
            prophet_inner_fitted = prophet_val.predict(periods=0, future_df=inner_train_df)
            inner_residuals = inner_train_df[target_column].values[:len(prophet_inner_fitted)] - prophet_inner_fitted["yhat"].values
            inner_res_df = pd.DataFrame({
                "ds": inner_train_df["ds"].iloc[:len(inner_residuals)],
                residual_col: inner_residuals,
            })
            tfm_val = EnergyTimesFM(**self.timesfm_kwargs)
            tfm_val.fit(inner_res_df, residual_col)
            val_res_forecast = tfm_val.predict(periods=val_size)
            val_res_preds = val_res_forecast.tail(val_size)["yhat"].values

            self._alpha = self._learn_alpha(prophet_val_preds, val_res_preds, val_actuals)
        else:
            self._alpha = 0.5  # conservative default

        # 5. Ensemble predictions with learned alpha + clamping
        ensemble_preds = prophet_preds + self._apply_correction(prophet_preds, residual_preds)

        # 6. TimesFM standalone for comparison
        self.timesfm_standalone.fit(train_df, target_column)
        tfm_forecast = self.timesfm_standalone.predict(periods=test_size)
        tfm_preds = tfm_forecast.tail(test_size)["yhat"].values

        # Store for analysis
        self._residuals = residuals
        self._test_actuals = actuals
        self._test_prophet = prophet_preds
        self._test_timesfm = tfm_preds
        self._test_ensemble = ensemble_preds

        return {
            "prophet": self._compute_metrics(actuals, prophet_preds, train_size, test_size),
            "timesfm": self._compute_metrics(actuals, tfm_preds, train_size, test_size),
            "ensemble": self._compute_metrics(actuals, ensemble_preds, train_size, test_size),
        }

    @staticmethod
    def _residual_col_name(target_column: str) -> str:
        return f"{target_column}_residual"

    @staticmethod
    def _compute_metrics(
        actuals: np.ndarray,
        predictions: np.ndarray,
        train_size: int,
        test_size: int,
    ) -> Dict[str, float]:
        mae = np.mean(np.abs(predictions - actuals))
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        smape = np.mean(
            2 * np.abs(actuals - predictions)
            / (np.abs(actuals) + np.abs(predictions) + 1e-10)
        ) * 100
        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "sMAPE": smape,
            "train_size": train_size,
            "test_size": test_size,
        }

    def get_residual_stats(self) -> Dict[str, float]:
        """
        Analyze Prophet's residuals to check if they contain patterns.

        Returns:
            Dict with residual statistics, autocorrelation, and learned alpha
        """
        if self._residuals is None:
            raise ValueError("No residuals computed. Call evaluate() first.")

        r = self._residuals
        return {
            "mean": float(np.mean(r)),
            "std": float(np.std(r)),
            "min": float(np.min(r)),
            "max": float(np.max(r)),
            "autocorr_lag1": float(np.corrcoef(r[:-1], r[1:])[0, 1]),
            "autocorr_lag24": float(
                np.corrcoef(r[:-24], r[24:])[0, 1]
            ) if len(r) > 24 else 0.0,
            "alpha": self._alpha,
        }


if __name__ == "__main__":
    import logging
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").addHandler(logging.NullHandler())
    logging.getLogger("cmdstanpy").propagate = False

    from data_generator import generate_energy_carbon_data

    print("Generating sample data...")
    df = generate_energy_carbon_data(periods=24 * 30)

    print("\nRunning three-way evaluation on 'totalEnergyConsumption'...")
    ensemble = EnsembleForecaster(regressors=["trainingActive"])
    results = ensemble.evaluate(df, "totalEnergyConsumption")

    print("\n" + "=" * 60)
    print(f"  {'Model':<12} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'sMAPE':>8}")
    print("=" * 60)
    for model_name in ["prophet", "timesfm", "ensemble"]:
        m = results[model_name]
        print(f"  {model_name:<12} {m['MAE']:>8.4f} {m['RMSE']:>8.4f} {m['MAPE']:>7.2f}% {m['sMAPE']:>7.2f}%")
    print("=" * 60)

    stats = ensemble.get_residual_stats()
    print(f"\nResidual stats (after regressor):")
    print(f"  autocorr lag 1: {stats['autocorr_lag1']:.3f}")
    print(f"  learned alpha:  {stats['alpha']:.2f}")
