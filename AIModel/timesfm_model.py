"""
TimesFM Foundation Model for Energy & Carbon Metrics Forecasting

Zero-shot time series forecasting using Google's pre-trained TimesFM.
Mirrors EnergyProphet's interface (fit/predict/evaluate) for easy comparison.
"""
import logging

import pandas as pd
import numpy as np
import torch
import timesfm
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Model checkpoint — 200M parameter version (lighter, ~925 MB download)
_DEFAULT_REPO_ID = "google/timesfm-1.0-200m-pytorch"


class EnergyTimesFM:
    """
    TimesFM wrapper for energy and carbon metrics forecasting.

    Unlike Prophet, TimesFM is a foundation model — it requires no training.
    It uses patterns learned from billions of time series during pre-training
    to forecast your data zero-shot.

    Interface mirrors EnergyProphet for fair comparison:
        model.fit(df, target_column)   # stores data (no actual training)
        model.predict(periods)         # zero-shot forecast
        model.evaluate(df, column)     # train/test split evaluation
    """

    def __init__(
        self,
        context_len: int = 512,
        horizon_len: int = 128,
        backend: str = "cpu",
        repo_id: str = _DEFAULT_REPO_ID,
    ):
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.backend = backend
        self.repo_id = repo_id

        self.target_column: Optional[str] = None
        self._history: Optional[np.ndarray] = None
        self._history_dates: Optional[pd.Series] = None
        self._freq: Optional[str] = None
        self._model: Optional[timesfm.TimesFm] = None

    def _ensure_model(self):
        """Load the TimesFM model on first use (lazy loading)."""
        if self._model is not None:
            return

        torch.set_float32_matmul_precision("high")

        checkpoint = timesfm.TimesFmCheckpoint(
            huggingface_repo_id=self.repo_id,
        )
        self._model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                context_len=self.context_len,
                horizon_len=self.horizon_len,
                per_core_batch_size=32,
                backend=self.backend,
            ),
            checkpoint=checkpoint,
        )
        self._model.load_from_checkpoint(checkpoint)

    def fit(self, df: pd.DataFrame, target_column: str, freq: str = "h") -> "EnergyTimesFM":
        """
        Store data for forecasting. No actual training happens —
        TimesFM works zero-shot from its pre-trained weights.

        Args:
            df: Input DataFrame with 'ds' column and target metric
            target_column: Column to forecast
            freq: Data frequency ('h' for hourly, 'D' for daily)

        Returns:
            self for method chaining
        """
        self.target_column = target_column
        self._history = df[target_column].astype(float).values
        self._history_dates = pd.to_datetime(df["ds"])
        self._freq = freq
        return self

    def predict(self, periods: int, freq: str = "h") -> pd.DataFrame:
        """
        Generate future predictions using zero-shot inference.

        Args:
            periods: Number of periods to forecast
            freq: Frequency for generating future dates

        Returns:
            DataFrame with columns: ds, yhat, yhat_lower, yhat_upper
            (matching Prophet's output format for compatibility)
        """
        if self._history is None:
            raise ValueError("No data loaded. Call fit() first.")

        self._ensure_model()

        # TimesFM forecasts in chunks of horizon_len, so we may need
        # multiple passes for long horizons. Use the last context_len points.
        context = self._history[-self.context_len:]

        # Forecast — may produce more points than needed, we'll trim
        all_predictions = []
        all_lower = []
        all_upper = []
        remaining = periods
        current_context = context.copy()

        while remaining > 0:
            point, quantile = self._model.forecast(
                [current_context], freq=[0]
            )
            chunk_point = point[0, :remaining]
            # quantile shape: (1, horizon, 10) — index 0=10th pct, 8=90th pct
            chunk_lower = quantile[0, :remaining, 0]  # 10th percentile
            chunk_upper = quantile[0, :remaining, 8]  # 90th percentile

            all_predictions.append(chunk_point)
            all_lower.append(chunk_lower)
            all_upper.append(chunk_upper)

            remaining -= len(chunk_point)
            if remaining > 0:
                # Extend context with predictions for next chunk
                current_context = np.concatenate([
                    current_context, chunk_point
                ])[-self.context_len:]

        predictions = np.concatenate(all_predictions)[:periods]
        lower = np.concatenate(all_lower)[:periods]
        upper = np.concatenate(all_upper)[:periods]

        # Build future dates
        last_date = self._history_dates.iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(hours=1),
            periods=periods,
            freq=freq,
        )

        # Build result matching Prophet's output column names
        history_df = pd.DataFrame({
            "ds": self._history_dates,
            "yhat": self._history,
            "yhat_lower": self._history,
            "yhat_upper": self._history,
        })
        forecast_df = pd.DataFrame({
            "ds": future_dates,
            "yhat": predictions,
            "yhat_lower": lower,
            "yhat_upper": upper,
        })

        return pd.concat([history_df, forecast_df], ignore_index=True)

    def evaluate(
        self,
        df: pd.DataFrame,
        target_column: str,
        train_ratio: float = 0.8,
    ) -> Dict[str, float]:
        """
        Train/test split evaluation — same methodology as EnergyProphet.

        Args:
            df: Full dataset
            target_column: Column to forecast
            train_ratio: Proportion of data for context

        Returns:
            Dictionary with MAE, MSE, RMSE, MAPE metrics
        """
        n = len(df)
        train_size = int(n * train_ratio)
        test_size = n - train_size

        # Use training data as context
        self.fit(df.iloc[:train_size], target_column)

        # Forecast the test period
        forecast = self.predict(periods=test_size)
        predictions = forecast.iloc[train_size:]["yhat"].values
        actuals = df.iloc[train_size:][target_column].values

        mae = np.mean(np.abs(predictions - actuals))
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "train_size": train_size,
            "test_size": test_size,
        }

    def forecast_batch(
        self, df: pd.DataFrame, columns: list, periods: int
    ) -> Dict[str, np.ndarray]:
        """
        Forecast multiple metrics in a single efficient batch.

        Args:
            df: Input DataFrame
            columns: List of column names to forecast
            periods: Number of periods to forecast per metric

        Returns:
            Dict mapping column name → predicted values array
        """
        self._ensure_model()

        inputs = [
            df[col].astype(float).values[-self.context_len:]
            for col in columns
        ]
        freq = [0] * len(columns)

        point, _ = self._model.forecast(inputs, freq=freq)

        return {
            col: point[i, :periods]
            for i, col in enumerate(columns)
        }


if __name__ == "__main__":
    from data_generator import generate_energy_carbon_data

    print("Generating sample data...")
    df = generate_energy_carbon_data(periods=24 * 30)

    print("\nLoading TimesFM model (first run downloads ~925 MB)...")
    model = EnergyTimesFM()

    print("\nEvaluating on 'totalEnergyConsumption'...")
    metrics = model.evaluate(df, "totalEnergyConsumption")
    print("\nEvaluation Metrics:")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")

    print("\nGenerating 24-hour forecast...")
    model.fit(df, "totalEnergyConsumption")
    forecast = model.predict(periods=24)
    print("\nForecast (next 24 hours):")
    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(24))
