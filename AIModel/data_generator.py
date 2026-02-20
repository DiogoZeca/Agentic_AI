"""
AI Infrastructure Power Consumption Data Generator

Generates synthetic hourly data simulating an AI inference/training cluster.
Four interconnected angles create complex nonlinear interactions:
  1. Inference — user-driven demand with auto-batching efficiency
  2. Training — irregular GPU-intensive runs (2-3 per week)
  3. Total Infrastructure — PUE, cooling, auxiliary power
  4. Carbon & SCI — grid intensity, embodied emissions, per-inference carbon
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_energy_carbon_data(
    start_date: str = "2024-01-01",
    periods: int = 365 * 24,  # 1 year of hourly data
    freq: str = "h",
    method: str = "RAPL"
) -> pd.DataFrame:
    """
    Generate synthetic AI infrastructure power consumption data.

    Args:
        start_date: Starting date for the time series
        periods: Number of time periods to generate
        freq: Frequency ('h' for hourly)
        method: Energy measurement method (RAPL, TDP)

    Returns:
        DataFrame with all metrics matching the AI infrastructure schema
    """
    np.random.seed(42)

    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    hour = dates.hour
    dow = dates.dayofweek  # 0=Mon, 6=Sun
    day_of_year = dates.dayofyear

    # ── Angle 1: Inference ──────────────────────────────────────────────────

    # Base demand curve: low at night, ramp morning, peak business hours,
    # secondary evening peak, weekend 40% of weekday
    demand_profile = np.zeros(periods)
    for i in range(periods):
        h = hour[i]
        if h < 6:
            base = 50 + 20 * np.sin(np.pi * h / 6)  # ~50-70 at night
        elif h < 10:
            base = 50 + (h - 6) * 112.5  # ramp 50→500
        elif h < 16:
            base = 500 - 30 * abs(h - 13)  # plateau ~410-500
        elif h < 20:
            base = 500 - (h - 16) * 50  # decline to ~300
        else:
            base = 300 - (h - 20) * 62.5  # evening decline to ~50
        demand_profile[i] = base

    # Weekend reduction (40% of weekday)
    is_weekend = dow >= 5
    weekend_factor = np.where(is_weekend, 0.4, 1.0)

    # Slow growth trend over the year (~20% increase)
    trend = 1 + 0.2 * np.arange(periods) / periods

    # Add noise (10% of signal)
    noise = np.random.normal(0, 0.10, periods)

    inference_requests = demand_profile * weekend_factor * trend * (1 + noise)
    inference_requests = np.maximum(inference_requests, 5).astype(float)

    # Auto-batching: larger batches at high load
    load_fraction = inference_requests / inference_requests.max()
    avg_batch_size = 1 + 31 * load_fraction**1.5  # ~1-4 at low, ~16-32 at peak
    avg_batch_size += np.random.normal(0, 0.5, periods)
    avg_batch_size = np.clip(avg_batch_size, 1, 64)

    # GPU utilization for inference — derived from load, capped at 85%
    gpu_util_inference = 10 + 75 * load_fraction  # 10-85%
    gpu_util_inference += np.random.normal(0, 2, periods)
    gpu_util_inference = np.clip(gpu_util_inference, 5, 85)

    # Inference power draw: 4 GPUs, idle 80W each, max 350W each
    n_gpu = 4
    idle_power = 80  # W per GPU
    max_power = 350  # W per GPU
    per_gpu_inference_power = idle_power + (max_power - idle_power) * (gpu_util_inference / 100)
    inference_power_draw = n_gpu * per_gpu_inference_power  # total W

    # Energy per inference: drops at high load (batch efficiency)
    # power (W) / requests per hour → Wh per request
    energy_per_inference = inference_power_draw / np.maximum(inference_requests, 1)

    # ── Angle 2: Training ───────────────────────────────────────────────────

    # Schedule 2-3 training runs per week, each 4-8 hours
    # 70% start at night/weekends, 30% during business hours
    training_active = np.zeros(periods, dtype=float)
    training_starts = []

    i = 0
    while i < periods:
        # Decide how many runs this week (2 or 3)
        runs_this_week = np.random.choice([2, 3])
        week_start = i
        week_end = min(i + 168, periods)

        for _ in range(runs_this_week):
            duration = np.random.randint(4, 9)  # 4-8 hours

            # 70% chance night/weekend start
            if np.random.random() < 0.7:
                # Night hours (22-06) or weekend
                night_slots = []
                for t in range(week_start, week_end):
                    if t < periods:
                        h_t = hour[t]
                        d_t = dow[t]
                        if h_t >= 22 or h_t <= 6 or d_t >= 5:
                            night_slots.append(t)
                if night_slots:
                    start_idx = np.random.choice(night_slots)
                else:
                    start_idx = np.random.randint(week_start, max(week_start + 1, week_end - duration))
            else:
                # Business hours (9-17)
                biz_slots = []
                for t in range(week_start, week_end):
                    if t < periods:
                        h_t = hour[t]
                        d_t = dow[t]
                        if 9 <= h_t <= 17 and d_t < 5:
                            biz_slots.append(t)
                if biz_slots:
                    start_idx = np.random.choice(biz_slots)
                else:
                    start_idx = np.random.randint(week_start, max(week_start + 1, week_end - duration))

            end_idx = min(start_idx + duration, periods)
            training_active[start_idx:end_idx] = 1.0
            training_starts.append(start_idx)

        i += 168  # advance one week

    # Training power: 4 GPUs at ~350W each when active
    training_gpu_util = training_active * (85 + np.random.normal(0, 3, periods))
    training_gpu_util = np.clip(training_gpu_util, 0, 98)
    training_power_draw = training_active * n_gpu * max_power  # ~1400W when active

    # ── Angle 3: Total Infrastructure ───────────────────────────────────────

    total_gpu_power = inference_power_draw + training_power_draw

    # CPU and memory: 15% of GPU power + 50W base
    cpu_and_memory_power = 0.15 * total_gpu_power + 50

    # PUE: base 1.2, seasonal variation (higher in summer), daily noise
    summer_peak = np.sin(2 * np.pi * (day_of_year - 80) / 365)  # peaks ~day 172 (June)
    summer_peak = np.clip(summer_peak, 0, 1)
    pue = 1.20 + 0.15 * summer_peak + np.random.normal(0, 0.02, periods)
    pue = np.clip(pue, 1.05, 1.60)

    total_infrastructure_power = (total_gpu_power + cpu_and_memory_power) * pue  # W
    total_energy_consumption = total_infrastructure_power / 1000  # kWh (per hour)

    # ── Angle 4: Carbon & SCI ───────────────────────────────────────────────

    # Carbon intensity: lower midday (solar), higher at night
    base_intensity = 0.4  # kgCO2/kWh
    solar_reduction = 0.25 * np.exp(-((hour - 12) ** 2) / 10)
    seasonal_green = 0.05 * summer_peak  # slightly greener grid in summer
    carbon_intensity_factor = (
        base_intensity - solar_reduction - seasonal_green
        + np.random.normal(0, 0.02, periods)
    )
    carbon_intensity_factor = np.clip(carbon_intensity_factor, 0.05, 0.60)

    # Green consumption percentage (inverse of intensity)
    green_consumption_percentage = (
        20 + 50 * np.exp(-((hour - 13) ** 2) / 15)
        + 10 * summer_peak
        + np.random.normal(0, 3, periods)
    )
    green_consumption_percentage = np.clip(green_consumption_percentage, 0, 100)

    # Operational emissions
    operational_emissions = total_energy_consumption * carbon_intensity_factor

    # Embodied emissions: constant hardware amortization rate
    embodied_emissions_rate = 0.005  # kgCO2e per hour (server hardware)
    embodied_emissions = np.full(periods, embodied_emissions_rate)

    # Total carbon
    carbon_emissions = operational_emissions + embodied_emissions

    # SCI per inference: gCO2e per inference request
    sci_per_inference = (carbon_emissions / np.maximum(inference_requests, 1)) * 1000

    # ── Legacy metrics (kept from original schema) ──────────────────────────

    # consumption = totalEnergyConsumption (kWh per hour)
    consumption = total_energy_consumption

    # Financial: electricity cost with peak/off-peak pricing
    base_cost_per_kwh = 0.12  # EUR/kWh
    peak_hours = (hour >= 9) & (hour <= 21)
    cost_multiplier = np.where(peak_hours, 1.5, 0.8)
    cost = consumption * base_cost_per_kwh * cost_multiplier
    total_cost = np.cumsum(cost)

    # SCI: functionalUnitCount = inferenceRequests (natural functional unit)
    functional_unit_count = inference_requests
    software_carbon_intensity = (carbon_emissions / np.maximum(functional_unit_count, 1)) * 1000

    # ── Build DataFrame ─────────────────────────────────────────────────────

    df = pd.DataFrame({
        # Time
        "ds": dates,

        # Angle 1: Inference
        "inferenceRequests": np.round(inference_requests, 1),
        "avgBatchSize": np.round(avg_batch_size, 1),
        "gpuUtilInference": np.round(gpu_util_inference, 1),
        "inferencePowerDraw": np.round(inference_power_draw, 1),
        "energyPerInference": np.round(energy_per_inference, 4),

        # Angle 2: Training
        "trainingActive": training_active.astype(int),
        "trainingPowerDraw": np.round(training_power_draw, 1),
        "trainingGpuUtil": np.round(training_gpu_util, 1),

        # Angle 3: Total Infrastructure
        "totalGpuPower": np.round(total_gpu_power, 1),
        "cpuAndMemoryPower": np.round(cpu_and_memory_power, 1),
        "pue": np.round(pue, 3),
        "totalInfrastructurePower": np.round(total_infrastructure_power, 1),
        "totalEnergyConsumption": np.round(total_energy_consumption, 4),

        # Angle 4: Carbon & SCI
        "carbonIntensityFactor": np.round(carbon_intensity_factor, 4),
        "greenConsumptionPercentage": np.round(green_consumption_percentage, 1),
        "operationalEmissions": np.round(operational_emissions, 4),
        "embodiedEmissions": np.round(embodied_emissions, 4),
        "carbonEmissions": np.round(carbon_emissions, 4),
        "sciPerInference": np.round(sci_per_inference, 4),

        # Legacy metrics (kept from original schema)
        "consumption": np.round(consumption, 4),
        "method": method,
        "cost": np.round(cost, 4),
        "totalCost": np.round(total_cost, 4),
        "softwareCarbonIntensity": np.round(software_carbon_intensity, 4),
        "functionalUnitCount": np.round(functional_unit_count, 1),
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
    print(f"\nTraining hours: {df['trainingActive'].sum():.0f}h "
          f"({df['trainingActive'].sum() / len(df) * 100:.1f}% of time)")
    print(f"Total inference requests: {df['inferenceRequests'].sum():,.0f}")
    print(f"Avg energy per inference: {df['energyPerInference'].mean():.4f} Wh")
    return df


if __name__ == "__main__":
    df = save_sample_data()
    print("\nFirst few rows:")
    print(df.head())
    print("\nColumn types:")
    print(df.dtypes)
