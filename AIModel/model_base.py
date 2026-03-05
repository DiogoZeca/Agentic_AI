"""
ForecasterBase — abstract interface contract for all single-metric forecasting models.

EnsembleForecaster intentionally excluded — its evaluate() returns a nested dict
(prophet/timesfm/ensemble) and serves a different purpose.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd


class ForecasterBase(ABC):

    @abstractmethod
    def fit(self, df: pd.DataFrame, target_column: str) -> "ForecasterBase":
        """Fit or configure the model on historical data. Returns self."""

    @abstractmethod
    def predict(self, periods: int, **kwargs) -> pd.DataFrame:
        """Return forecast DataFrame: [ds, yhat, yhat_lower, yhat_upper].
        Must return exactly `periods` future rows (not history + future).
        """

    @abstractmethod
    def evaluate(
        self, df: pd.DataFrame, target_column: str, train_ratio: float = 0.8
    ) -> Dict[str, float]:
        """Evaluate on held-out test set. Must return all 7 keys:
        MAE, MSE, RMSE, MAPE, sMAPE, train_size, test_size
        """
