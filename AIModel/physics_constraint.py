"""
Physics Constraint Module — formula-derived carbon emissions forecasting.

The physical formula from the SCI framework:
    carbonEmissions = consumption × carbonIntensityFactor + embodiedEmissions
                    = E × I + M
where M = 0.002 kgCO2e/h (hardware amortization constant).

Pure functions — no learnable parameters, no Prophet/TimesFM dependencies.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict

# Hardware amortization constant (embodied emissions, kgCO2e/h)
EMBODIED_EMISSIONS_KGC02E_H: float = 0.002


def derive_carbon_emissions(
    consumption_fc: pd.DataFrame,
    cif_fc: pd.DataFrame,
) -> pd.DataFrame:
    """Derive carbon emissions forecast analytically from component forecasts.

    Formula:
        yhat       = consumption.yhat       × cif.yhat       + 0.002
        yhat_lower = consumption.yhat_lower × cif.yhat_lower + 0.002
        yhat_upper = consumption.yhat_upper × cif.yhat_upper + 0.002

    Both DataFrames must have columns: ds, yhat, yhat_lower, yhat_upper.
    Interval ordering is guaranteed when inputs are non-negative.
    All outputs are clipped to >= 0.

    Args:
        consumption_fc: Forecast DataFrame for energy consumption (kWh/h)
        cif_fc:         Forecast DataFrame for carbon intensity factor (kgCO2/kWh)

    Returns:
        DataFrame with columns: ds, yhat, yhat_lower, yhat_upper (kgCO2e/h)
    """
    c = consumption_fc.reset_index(drop=True)
    i = cif_fc.reset_index(drop=True)

    result = pd.DataFrame({
        "ds":         c["ds"],
        "yhat":       np.maximum(c["yhat"]       * i["yhat"]       + EMBODIED_EMISSIONS_KGC02E_H, 0.0),
        "yhat_lower": np.maximum(c["yhat_lower"] * i["yhat_lower"] + EMBODIED_EMISSIONS_KGC02E_H, 0.0),
        "yhat_upper": np.maximum(c["yhat_upper"] * i["yhat_upper"] + EMBODIED_EMISSIONS_KGC02E_H, 0.0),
    })
    return result.reset_index(drop=True)


def evaluate_formula_accuracy(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> Dict[str, float]:
    """Evaluate formula accuracy on a held-out 20% test set.

    Uses the same 80/20 train/test split as Prophet.evaluate() for fair comparison.
    Applies the formula: carbonEmissions ≈ consumption × carbonIntensityFactor + 0.002

    Args:
        df:          Full dataset with columns: consumption, carbonIntensityFactor,
                     carbonEmissions, ds
        train_ratio: Fraction of data to use as training context (unused by formula,
                     but determines the test window for fair evaluation)

    Returns:
        Dict with keys: MAE, MSE, RMSE, MAPE, sMAPE, train_size, test_size
    """
    n = len(df)
    train_size = int(n * train_ratio)
    test_df = df.iloc[train_size:].copy()

    predictions = (
        test_df["consumption"].values * test_df["carbonIntensityFactor"].values
        + EMBODIED_EMISSIONS_KGC02E_H
    )
    actuals = test_df["carbonEmissions"].values

    mae   = float(np.mean(np.abs(predictions - actuals)))
    mse   = float(np.mean((predictions - actuals) ** 2))
    rmse  = float(np.sqrt(mse))
    if np.any(actuals == 0):
        mape = float("nan")
    else:
        mape = float(np.mean(np.abs((actuals - predictions) / actuals)) * 100)
    smape = float(np.mean(
        2 * np.abs(actuals - predictions)
        / (np.abs(actuals) + np.abs(predictions) + 1e-10)
    ) * 100)

    return {
        "MAE":        mae,
        "MSE":        mse,
        "RMSE":       rmse,
        "MAPE":       mape,
        "sMAPE":      smape,
        "train_size": train_size,
        "test_size":  len(test_df),
    }


def check_physical_consistency(forecast_df: pd.DataFrame) -> Dict:
    """Check that a carbon emissions forecast satisfies physical constraints.

    Args:
        forecast_df: DataFrame with columns: yhat, yhat_lower, yhat_upper

    Returns:
        Dict with keys:
            is_non_negative  — True if all yhat/lower/upper >= 0
            negative_count   — number of rows with any negative value
            max_value        — maximum yhat value
            min_value        — minimum yhat value
            interval_ordered — True if yhat_lower <= yhat <= yhat_upper for all rows
    """
    yhat       = forecast_df["yhat"].values
    yhat_lower = forecast_df["yhat_lower"].values
    yhat_upper = forecast_df["yhat_upper"].values

    all_non_neg = bool(
        (yhat >= 0).all() and (yhat_lower >= 0).all() and (yhat_upper >= 0).all()
    )
    neg_count = int(
        ((yhat < 0) | (yhat_lower < 0) | (yhat_upper < 0)).sum()
    )
    interval_ordered = bool(
        ((yhat_lower <= yhat) & (yhat <= yhat_upper)).all()
    )

    return {
        "is_non_negative":  all_non_neg,
        "negative_count":   neg_count,
        "max_value":        float(yhat.max()),
        "min_value":        float(yhat.min()),
        "interval_ordered": interval_ordered,
    }
