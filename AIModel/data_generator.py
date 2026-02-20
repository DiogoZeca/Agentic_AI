"""
Web Service Energy & Carbon Data Generator (SCI Framework)

Generates synthetic hourly data simulating a CPU/memory-based web service.
Aligned with the Green Software Foundation's SCI (Software Carbon Intensity) framework.

Schema (15 columns):
  ds, timeWindow, consumption, totalConsumption, measurementSource,
  cost, totalCost,
  carbonEmissions, operationalEmissions, embodiedEmissions,
  carbonIntensityFactor, greenConsumptionPercentage,
  softwareCarbonIntensity, functionalUnit,
  cpuUtilization
"""
import pandas as pd
import numpy as np
from datetime import datetime


def generate_energy_carbon_data(
    start_date: str = "2024-01-01",
    periods: int = 365 * 24,  # 1 year of hourly data
    freq: str = "h",
) -> pd.DataFrame:
    """
    Generate synthetic web service energy & carbon data aligned with SCI framework.

    Args:
        start_date: Starting date for the time series
        periods: Number of time periods to generate
        freq: Frequency ('h' for hourly)

    Returns:
        DataFrame with 15 columns matching the service-based SCI schema
    """
    np.random.seed(42)

    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    hour = dates.hour.values.astype(float)
    dow = dates.dayofweek.values          # 0=Mon, 6=Sun
    day_of_year = dates.dayofyear.values.astype(float)
    t = np.arange(periods, dtype=float)

    # ── Functional Unit: requests/hour ───────────────────────────────────────

    # Business-hours demand curve (vectorised)
    demand_profile = np.where(
        hour < 6,   50 + 20 * np.sin(np.pi * hour / 6),
        np.where(
            hour < 10, 50 + (hour - 6) * 112.5,
            np.where(
                hour < 16, 500 - 30 * np.abs(hour - 13),
                np.where(
                    hour < 20, 500 - (hour - 16) * 50,
                    300 - (hour - 20) * 62.5
                )
            )
        )
    )

    is_weekend = dow >= 5
    weekend_factor = np.where(is_weekend, 0.4, 1.0)
    trend = 1.0 + 0.15 * t / periods                  # +15% annual growth
    noise = np.random.normal(0, 0.08, periods)

    functional_unit = demand_profile * weekend_factor * trend * (1 + noise)
    functional_unit = np.maximum(functional_unit, 5.0)

    # ── CPU Utilization ───────────────────────────────────────────────────────

    load_fraction = functional_unit / functional_unit.max()
    cpu_utilization = 0.10 + 0.75 * (load_fraction ** 0.8)
    cpu_utilization = np.clip(cpu_utilization, 0.05, 0.95)

    # ── Consumption (kWh/h) ───────────────────────────────────────────────────
    # Server power model: (20W idle + 130W max * cpu_util) * PUE 1.3 / 1000
    consumption = (20.0 + 130.0 * cpu_utilization) * 1.3 / 1000.0
    # Range: (20+130*0.05)*1.3/1000 ≈ 0.034  to  (20+130*0.95)*1.3/1000 ≈ 0.187 kWh/h

    total_consumption = np.cumsum(consumption)

    # ── Carbon Intensity Factor (kgCO2/kWh) ──────────────────────────────────

    # Midday solar dip (Gaussian centred at hour 12)
    solar_dip = 0.10 * np.exp(-((hour - 12) ** 2) / 8.0)
    # Seasonal: slightly greener grid in summer
    summer_peak = np.clip(np.sin(2 * np.pi * (day_of_year - 80) / 365), 0, 1)
    seasonal_green = 0.05 * summer_peak

    carbon_intensity_factor = (
        0.4 - solar_dip - seasonal_green
        + np.random.normal(0, 0.015, periods)
    )
    carbon_intensity_factor = np.clip(carbon_intensity_factor, 0.3, 0.5)

    # ── Green Consumption Percentage (%) ─────────────────────────────────────

    green_consumption_percentage = (
        20.0
        + 50.0 * np.exp(-((hour - 13) ** 2) / 15.0)
        + 10.0 * summer_peak
        + np.random.normal(0, 3, periods)
    )
    green_consumption_percentage = np.clip(green_consumption_percentage, 20.0, 70.0)

    # ── Emissions ─────────────────────────────────────────────────────────────

    operational_emissions = consumption * carbon_intensity_factor
    embodied_emissions = np.full(periods, 0.002)            # constant 0.002 kgCO2e/h
    carbon_emissions = operational_emissions + embodied_emissions

    # ── Cost (EUR) ────────────────────────────────────────────────────────────

    is_peak = (hour >= 9) & (hour <= 18)
    cost_multiplier = np.where(is_peak, 1.5, 1.0)
    cost = consumption * 0.12 * cost_multiplier
    total_cost = np.cumsum(cost)

    # ── Software Carbon Intensity (kgCO2e/req) ───────────────────────────────

    software_carbon_intensity = carbon_emissions / np.maximum(functional_unit, 1.0)
    # Range: ~0.00002–0.0018 kgCO2e/req

    # ── Measurement Source ────────────────────────────────────────────────────

    measurement_source = np.where(cpu_utilization > 0.6, "RAPL", "TDP")

    # ── Build DataFrame ───────────────────────────────────────────────────────

    df = pd.DataFrame({
        "ds":                       dates,
        "timeWindow":               3600,                                   # seconds
        "consumption":              np.round(consumption, 6),               # kWh/h
        "totalConsumption":         np.round(total_consumption, 4),         # kWh cumulative
        "measurementSource":        measurement_source,
        "cost":                     np.round(cost, 6),                      # EUR/h
        "totalCost":                np.round(total_cost, 4),                # EUR cumulative
        "carbonEmissions":          np.round(carbon_emissions, 6),          # kgCO2e/h
        "operationalEmissions":     np.round(operational_emissions, 6),     # kgCO2e/h
        "embodiedEmissions":        np.round(embodied_emissions, 6),        # kgCO2e/h
        "carbonIntensityFactor":    np.round(carbon_intensity_factor, 4),   # kgCO2/kWh
        "greenConsumptionPercentage": np.round(green_consumption_percentage, 1),  # %
        "softwareCarbonIntensity":  np.round(software_carbon_intensity, 8), # kgCO2e/req
        "functionalUnit":           np.round(functional_unit, 1),           # req/h
        "cpuUtilization":           np.round(cpu_utilization, 4),           # fraction
    })

    return df


def save_sample_data(filepath: str = "data/sample_energy_data.csv"):
    """Generate and save sample data to CSV."""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    df = generate_energy_carbon_data()
    df.to_csv(filepath, index=False)
    print(f"Sample data saved to {filepath}")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    print(f"\nTotal requests served:  {df['functionalUnit'].sum():,.0f} req")
    print(f"Total energy consumed:  {df['consumption'].sum():,.1f} kWh")
    print(f"Total carbon emitted:   {df['carbonEmissions'].sum():,.2f} kgCO2e")
    print(f"Avg SCI:                {df['softwareCarbonIntensity'].mean():.6f} kgCO2e/req")
    print(f"CPU util range:         {df['cpuUtilization'].min():.2f} – {df['cpuUtilization'].max():.2f}")
    print(f"Consumption range:      {df['consumption'].min():.4f} – {df['consumption'].max():.4f} kWh/h")
    return df


if __name__ == "__main__":
    df = save_sample_data()
    print("\nFirst few rows:")
    print(df.head())
    print("\nColumn types:")
    print(df.dtypes)
