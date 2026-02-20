"""
Prophet Model for Energy & Carbon Metrics Forecasting
"""
import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Optional, List, Tuple, Dict, Any


class EnergyProphet:
    """
    Prophet wrapper for energy and carbon metrics forecasting.

    Prophet requires data with two columns:
    - ds: datetime column
    - y: the value to forecast

    Supports external regressors (e.g., trainingActive) that Prophet uses
    as additional input features to explain spikes and irregular patterns.
    """

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        seasonality_mode: str = "multiplicative",
        changepoint_prior_scale: float = 0.05,
        regressors: Optional[List[str]] = None,
    ):
        """
        Initialize the Prophet model.

        Args:
            yearly_seasonality: Include yearly patterns
            weekly_seasonality: Include weekly patterns (workdays vs weekends)
            daily_seasonality: Include daily patterns (peak hours)
            seasonality_mode: 'additive' or 'multiplicative'
            changepoint_prior_scale: Flexibility of trend (higher = more flexible)
            regressors: List of column names to add as external regressors
                        (e.g., ["trainingActive"]). These columns must exist
                        in the DataFrame passed to fit() and evaluate().
        """
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.regressors = regressors or []
        self.model: Optional[Prophet] = None
        self.target_column: Optional[str] = None

    def prepare_data(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Prepare data for Prophet (requires 'ds' and 'y' columns,
        plus any configured regressor columns).

        Args:
            df: Input DataFrame with 'ds' column and target metric
            target_column: Name of the column to forecast

        Returns:
            DataFrame with 'ds', 'y', and regressor columns
        """
        self.target_column = target_column

        prophet_df = pd.DataFrame({
            "ds": pd.to_datetime(df["ds"]),
            "y": df[target_column].astype(float)
        })

        # Add regressor columns
        for reg in self.regressors:
            if reg in df.columns:
                prophet_df[reg] = df[reg].astype(float).values

        prophet_df = prophet_df.dropna()
        return prophet_df

    def fit(self, df: pd.DataFrame, target_column: str) -> "EnergyProphet":
        """
        Train the Prophet model.

        Args:
            df: Input DataFrame
            target_column: Column to forecast

        Returns:
            self for method chaining
        """
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
        )

        # Register regressors before fitting
        for reg in self.regressors:
            if reg in df.columns:
                self.model.add_regressor(reg)

        train_data = self.prepare_data(df, target_column)
        self.model.fit(train_data)
        return self

    def predict(
        self,
        periods: int,
        freq: str = "h",
        future_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate future predictions.

        Args:
            periods: Number of periods to forecast
            freq: Frequency ('h' for hourly, 'D' for daily)
            future_df: Optional DataFrame with 'ds' and regressor columns
                       for the future period. If None, regressors default to 0.

        Returns:
            DataFrame with predictions and uncertainty intervals
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        future = self.model.make_future_dataframe(periods=periods, freq=freq)

        # Fill regressor values
        if future_df is not None and len(self.regressors) > 0:
            # Merge regressor values from the provided future_df
            future_regs = future_df[["ds"] + self.regressors].copy()
            future_regs["ds"] = pd.to_datetime(future_regs["ds"])
            future = future.merge(future_regs, on="ds", how="left")
            for reg in self.regressors:
                future[reg] = future[reg].fillna(0)
        else:
            # Default: regressors = 0 (e.g., no training scheduled)
            for reg in self.regressors:
                if reg not in future.columns:
                    future[reg] = 0

        forecast = self.model.predict(future)
        return forecast

    def evaluate(
        self,
        df: pd.DataFrame,
        target_column: str,
        train_ratio: float = 0.8
    ) -> Dict[str, float]:
        """
        Train/test split evaluation. When regressors are configured,
        uses known regressor values from the test set (simulating a
        scenario where training schedules are known in advance).

        Args:
            df: Full dataset
            target_column: Column to forecast
            train_ratio: Proportion of data for training

        Returns:
            Dictionary with evaluation metrics
        """
        n = len(df)
        train_size = int(n * train_ratio)

        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        self.fit(train_df, target_column)

        # Build future with known regressor values from the full dataset
        forecast = self.predict(periods=len(test_df), future_df=df)

        predictions = forecast.iloc[train_size:]["yhat"].values
        actuals = test_df[target_column].values

        mae = np.mean(np.abs(predictions - actuals))
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        # sMAPE: symmetric MAPE, robust to extreme values
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
            "test_size": len(test_df),
        }

    def get_seasonality_components(self) -> pd.DataFrame:
        """
        Extract seasonality components for analysis.

        Returns:
            DataFrame with trend and seasonality breakdowns
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        future = self.model.make_future_dataframe(periods=1, freq="h")
        for reg in self.regressors:
            if reg not in future.columns:
                future[reg] = 0
        forecast = self.model.predict(future)

        return forecast[["ds", "trend", "weekly", "daily", "yearly"]].dropna()


def quick_forecast(
    csv_path: str,
    target_column: str = "totalEnergyConsumption",
    forecast_periods: int = 168,  # 1 week of hours
    freq: str = "h",
    regressors: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Quick function to load data, train, and forecast.

    Args:
        csv_path: Path to CSV file with 'ds' column
        target_column: Column to forecast
        forecast_periods: Number of periods to predict
        freq: Frequency of data
        regressors: Optional list of regressor column names

    Returns:
        Tuple of (forecast DataFrame, evaluation metrics)
    """
    df = pd.read_csv(csv_path, parse_dates=["ds"])

    model = EnergyProphet(regressors=regressors)
    metrics = model.evaluate(df, target_column)

    # Retrain on full data for final forecast
    model.fit(df, target_column)
    forecast = model.predict(periods=forecast_periods, freq=freq, future_df=df)

    return forecast, metrics


if __name__ == "__main__":
    from data_generator import generate_energy_carbon_data

    print("Generating sample data...")
    df = generate_energy_carbon_data(periods=24 * 30)  # 1 month hourly

    # Without regressors (baseline)
    print("\n--- Prophet WITHOUT regressors ---")
    model_base = EnergyProphet()
    metrics_base = model_base.evaluate(df, "totalEnergyConsumption")
    print(f"  MAPE: {metrics_base['MAPE']:.2f}%  |  sMAPE: {metrics_base['sMAPE']:.2f}%")

    # With trainingActive regressor
    print("\n--- Prophet WITH trainingActive regressor ---")
    model_reg = EnergyProphet(regressors=["trainingActive"])
    metrics_reg = model_reg.evaluate(df, "totalEnergyConsumption")
    print(f"  MAPE: {metrics_reg['MAPE']:.2f}%  |  sMAPE: {metrics_reg['sMAPE']:.2f}%")

    improvement = metrics_base["MAPE"] - metrics_reg["MAPE"]
    print(f"\n  Regressor improvement: {improvement:+.2f}% MAPE")
