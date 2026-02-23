"""
Test 2: Chained Forecasting Validation

The core PoC correctness test.

Background
----------
`consumption` uses `functionalUnit` (requests/hour) as a Prophet regressor.
For future timestamps, the regressor value is unknown. Two strategies exist:

  Zero-fill (naive):   future functionalUnit = 0
                       → model assumes the service is idle
                       → business-hours consumption is UNDER-predicted

  Chained (correct):   forecast functionalUnit first using its own Prophet model,
                       then inject those predictions as future_df for consumption
                       → model sees realistic ~400 req/h during business hours
                       → business-hours consumption is correctly HIGHER

What this test verifies
-----------------------
1. Chained predictions are not identical to zero-fill predictions.
2. During business hours (09:00–17:00), the chained forecast predicts
   meaningfully higher consumption than the zero-fill forecast.
   This validates that the regressor chain actually propagates workload
   information into the energy forecast.

Speed note
----------
Uses a 30-day dataset (720 rows) for fast Prophet fitting (~5-10s per model).
Two models are fit: EnergyProphet(regressors=["functionalUnit"]) for consumption,
and EnergyProphet() for functionalUnit (to produce chained future values).
Total runtime: ~20-40s for this test.
"""
import pandas as pd
import pytest

from prophet_model import EnergyProphet

HORIZON = 48  # 2 days ahead — enough to cover business hours across two days


@pytest.fixture(scope="module")
def zero_fill_and_chained_forecasts(small_df):
    """
    Fit models and produce both zero-fill and chained forecasts.
    Module-scoped to avoid re-fitting for every test function.

    Returns (zero_future, chained_future) — each a DataFrame with columns:
      ds, yhat, yhat_lower, yhat_upper
    Limited to the last HORIZON rows (future only).
    """
    df = small_df.copy()

    # ── Model 1: consumption with functionalUnit regressor ────────────────────
    consumption_model = EnergyProphet(regressors=["functionalUnit"])
    consumption_model.fit(df, "consumption")

    # Zero-fill forecast: pass historical df only.
    # Future timestamps (beyond df's end) will get functionalUnit=NaN → 0.
    zero_forecast = consumption_model.predict(HORIZON, future_df=df)
    zero_future = zero_forecast.tail(HORIZON).reset_index(drop=True)

    # ── Model 2: functionalUnit (no regressors) ───────────────────────────────
    fu_model = EnergyProphet()
    fu_model.fit(df, "functionalUnit")
    fu_forecast = fu_model.predict(HORIZON, future_df=df)

    # Build combined df: historical functionalUnit + forecasted future values
    fu_future_rows = (
        fu_forecast.tail(HORIZON)[["ds", "yhat"]]
        .rename(columns={"yhat": "functionalUnit"})
        .assign(ds=lambda x: pd.to_datetime(x["ds"]))
    )
    hist_fu = df[["ds", "functionalUnit"]].assign(ds=lambda x: pd.to_datetime(x["ds"]))
    combined = pd.concat([hist_fu, fu_future_rows], ignore_index=True)

    # Chained forecast: future timestamps now have realistic functionalUnit values
    chained_forecast = consumption_model.predict(HORIZON, future_df=combined)
    chained_future = chained_forecast.tail(HORIZON).reset_index(drop=True)

    return zero_future, chained_future


class TestChainedForecasting:

    def test_forecasts_differ(self, zero_fill_and_chained_forecasts):
        """
        Zero-fill and chained must produce different predictions.
        If they're identical, chaining is a no-op (regressor has no effect).
        """
        zero_future, chained_future = zero_fill_and_chained_forecasts
        max_abs_diff = (chained_future["yhat"] - zero_future["yhat"]).abs().max()
        assert max_abs_diff > 1e-6, (
            "Zero-fill and chained forecasts are identical — "
            "functionalUnit regressor has no effect on consumption predictions."
        )

    def test_chained_higher_during_business_hours(self, zero_fill_and_chained_forecasts):
        """
        During business hours (09–17), chained forecast must predict HIGHER
        consumption than zero-fill.

        Physical reasoning:
          Zero-fill sets future functionalUnit=0 → model sees idle service.
          Chained sets future functionalUnit~400 req/h → model sees normal workload.
          More requests → more CPU → more energy.
        """
        zero_future, chained_future = zero_fill_and_chained_forecasts

        # Identify business-hour rows in the future window
        hours = pd.to_datetime(chained_future["ds"]).dt.hour
        biz_mask = (hours >= 9) & (hours <= 17)

        assert biz_mask.sum() > 0, (
            "No business-hour rows in the 48h forecast window — "
            "check the start_date / horizon configuration."
        )

        zero_biz_mean = float(zero_future.loc[biz_mask, "yhat"].mean())
        chained_biz_mean = float(chained_future.loc[biz_mask, "yhat"].mean())

        assert chained_biz_mean > zero_biz_mean, (
            f"Chained forecast ({chained_biz_mean:.4f} kWh/h) should exceed "
            f"zero-fill ({zero_biz_mean:.4f} kWh/h) during business hours.\n"
            f"  Business hours sampled: {biz_mask.sum()}"
        )

    def test_chained_business_hours_difference_is_meaningful(
        self, zero_fill_and_chained_forecasts
    ):
        """
        The gap between chained and zero-fill during business hours must be
        at least 0.002 kWh/h (~1% of max consumption).

        This guards against a case where the regressor has a coefficient so
        small the improvement is negligible (model fitting issue, not a code bug).
        """
        zero_future, chained_future = zero_fill_and_chained_forecasts

        hours = pd.to_datetime(chained_future["ds"]).dt.hour
        biz_mask = (hours >= 9) & (hours <= 17)

        zero_biz_mean = float(zero_future.loc[biz_mask, "yhat"].mean())
        chained_biz_mean = float(chained_future.loc[biz_mask, "yhat"].mean())
        diff = chained_biz_mean - zero_biz_mean

        assert diff >= 0.002, (
            f"Chained vs zero-fill difference during business hours is only "
            f"{diff:.4f} kWh/h — expected ≥0.002 kWh/h.\n"
            f"  This may indicate the functionalUnit regressor coefficient is "
            f"too small to be meaningful."
        )

    def test_future_functionalunit_values_are_realistic(self, small_df):
        """
        The functionalUnit model (used for chaining) should forecast values
        in the expected business-day range (roughly 50–600 req/h).
        """
        df = small_df.copy()
        fu_model = EnergyProphet()
        fu_model.fit(df, "functionalUnit")
        fu_forecast = fu_model.predict(HORIZON, future_df=df)
        future_fu = fu_forecast.tail(HORIZON)["yhat"]

        assert future_fu.min() > -50, (
            f"Forecasted functionalUnit went too negative: min={future_fu.min():.1f}"
        )
        assert future_fu.max() < 800, (
            f"Forecasted functionalUnit is unrealistically high: max={future_fu.max():.1f}"
        )

    def test_uncertainty_intervals_ordered(self, zero_fill_and_chained_forecasts):
        """
        Prophet uncertainty intervals must be ordered: yhat_lower ≤ yhat ≤ yhat_upper.
        Tests both forecasts.
        """
        for name, fc in zip(
            ["zero_fill", "chained"], zero_fill_and_chained_forecasts
        ):
            violations_lower = (fc["yhat"] < fc["yhat_lower"] - 1e-9).sum()
            violations_upper = (fc["yhat"] > fc["yhat_upper"] + 1e-9).sum()
            assert violations_lower == 0, (
                f"{name}: yhat < yhat_lower in {violations_lower} rows"
            )
            assert violations_upper == 0, (
                f"{name}: yhat > yhat_upper in {violations_upper} rows"
            )
