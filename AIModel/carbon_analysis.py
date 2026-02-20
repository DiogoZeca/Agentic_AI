"""
AI Infrastructure Carbon Analysis

Generates 6 focused charts answering key questions about your AI cluster's
energy and carbon footprint. Uses Prophet for time-series forecasting.

Charts produced:
  01_power_profile.png    — When do training spikes hit the cluster?
  02_daily_pattern.png    — Which hours of the day consume most energy?
  03_training_window.png  — When should training run (lowest grid carbon)?
  04_energy_forecast.png  — 7-day energy forecast with accuracy metrics
  05_carbon_forecast.png  — 7-day carbon emissions forecast
  06_sci_by_hour.png      — CO2 cost per inference request by hour of day

Usage:
    python carbon_analysis.py
    DATA_PATH=data/my_data.csv python carbon_analysis.py
"""
import os
import logging
import warnings

logging.getLogger("prophet").setLevel(logging.WARNING)
_cmdstanpy_logger = logging.getLogger("cmdstanpy")
_cmdstanpy_logger.setLevel(logging.ERROR)
_cmdstanpy_logger.addHandler(logging.NullHandler())
_cmdstanpy_logger.propagate = False
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from prophet_model import EnergyProphet
from data_loader import load_and_validate, build_pipeline_config, PipelineConfig

OUTPUT_DIR = "analysis_output"
DATA_PATH = os.environ.get("DATA_PATH", "data/sample_energy_data.csv")
FORECAST_PERIODS = 168   # 7 days × 24 hours
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# ── Consistent colour palette ──────────────────────────────────────────────────
C = {
    "energy":     "#1565C0",   # deep blue
    "training":   "#BF360C",   # deep orange-red
    "inference":  "#1565C0",   # blue
    "carbon":     "#B71C1C",   # deep red
    "forecast":   "#6A1B9A",   # purple
    "green":      "#2E7D32",   # dark green
    "actual":     "#212121",   # near-black
    "history":    "#9E9E9E",   # grey
    "ci":         "#CE93D8",   # light purple
    "overhead":   "#546E7A",   # blue-grey
}


def _apply_style():
    """Apply consistent chart style."""
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "#FAFAFA",
        "axes.grid":         True,
        "grid.alpha":        0.25,
        "grid.linestyle":    "--",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.size":         10,
        "axes.titlesize":    11,
        "axes.labelsize":    10,
    })


# ── Section 1: Load & Summarise ────────────────────────────────────────────────

def load_and_summarise(path: str) -> tuple[pd.DataFrame, PipelineConfig]:
    """Load CSV, validate schema, print summary."""
    df = load_and_validate(path)
    config = build_pipeline_config(df)

    print("=" * 60)
    print("  AI INFRASTRUCTURE CARBON ANALYSIS")
    print("=" * 60)
    print(f"  Records : {len(df)}")
    print(f"  Period  : {df['ds'].min().date()} → {df['ds'].max().date()}")
    print(f"\n  Energy  : {df['totalEnergyConsumption'].sum():,.1f} kWh total")
    if "carbonEmissions" in df.columns:
        print(f"  Carbon  : {df['carbonEmissions'].sum():,.1f} kgCO2e total")
    if "inferenceRequests" in df.columns:
        print(f"  Requests: {df['inferenceRequests'].sum():,.0f} total inference requests")
    if "trainingActive" in df.columns:
        th = df["trainingActive"].sum()
        print(f"  Training: {th:.0f} h ({th / len(df) * 100:.1f}% of time)")
    if "sciPerInference" in df.columns:
        print(f"  Avg SCI : {df['sciPerInference'].mean():.2f} gCO2e / request")
    if config.missing_recommended:
        print(f"\n  [Note] Missing optional columns: {', '.join(config.missing_recommended)}")
    return df, config


# ── Section 2: Fit Models ─────────────────────────────────────────────────────

def fit_models(df: pd.DataFrame, config: PipelineConfig) -> dict:
    """Fit Prophet on each available metric; derive SCI from components."""
    results = {}
    for metric in config.fit_metrics:
        regs = config.regressor_map.get(metric, [])
        label = f" (+ {', '.join(regs)})" if regs else ""
        print(f"\n  Fitting Prophet on '{metric}'{label} …")
        model = EnergyProphet(regressors=regs)
        eval_m = model.evaluate(df, metric)
        print(f"    sMAPE {eval_m['sMAPE']:.1f}%  |  MAPE {eval_m['MAPE']:.1f}%  |  RMSE {eval_m['RMSE']:.4f}")
        model.fit(df, metric)
        forecast = model.predict(FORECAST_PERIODS, future_df=df)
        results[metric] = {"model": model, "forecast": forecast, "eval": eval_m}

    if config.has_sci and "carbonEmissions" in results and "inferenceRequests" in results:
        print("\n  Deriving SCI from carbonEmissions / inferenceRequests …")
        results["sciPerInference"] = _derive_sci(df, results)
        e = results["sciPerInference"]["eval"]
        print(f"    sMAPE {e['sMAPE']:.1f}%  |  MAPE {e['MAPE']:.1f}%  |  RMSE {e['RMSE']:.4f}")

    return results


def _derive_sci(df: pd.DataFrame, results: dict) -> dict:
    """Derive SCI evaluation and forecast from component models."""
    n = len(df)
    train_size = int(n * 0.8)
    test_size = n - train_size
    train_df = df.iloc[:train_size]

    cm = EnergyProphet()
    cm.fit(train_df, "carbonEmissions")
    c_fc = cm.predict(test_size, future_df=df)
    c_preds = c_fc.iloc[train_size:]["yhat"].values

    rm = EnergyProphet()
    rm.fit(train_df, "inferenceRequests")
    r_fc = rm.predict(test_size, future_df=df)
    r_preds = np.maximum(r_fc.iloc[train_size:]["yhat"].values, 1)

    sci_preds = c_preds / r_preds * 1000

    if "sciPerInference" in df.columns:
        sci_actuals = df.iloc[train_size:]["sciPerInference"].values
        sci_p99 = np.percentile(df["sciPerInference"].values, 99)
    else:
        s = df["carbonEmissions"] / df["inferenceRequests"].clip(lower=1) * 1000
        sci_actuals = s.iloc[train_size:].values
        sci_p99 = np.percentile(s.values, 99)

    mae  = np.mean(np.abs(sci_preds - sci_actuals))
    mse  = np.mean((sci_preds - sci_actuals) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((sci_actuals - sci_preds) / sci_actuals)) * 100
    smape = np.mean(
        2 * np.abs(sci_actuals - sci_preds)
        / (np.abs(sci_actuals) + np.abs(sci_preds) + 1e-10)
    ) * 100

    # Forecast from full-data components
    cf = results["carbonEmissions"]["forecast"][["ds", "yhat"]].copy()
    rf = results["inferenceRequests"]["forecast"][["ds", "yhat"]].copy()
    mg = cf.merge(rf, on="ds", suffixes=("_c", "_r"))
    mg["yhat"] = mg["yhat_c"] / np.maximum(mg["yhat_r"], 1) * 1000
    mg["yhat"] = np.clip(mg["yhat"], 0, sci_p99 * 2)

    return {
        "model": None,
        "forecast": mg[["ds", "yhat"]],
        "eval": {"MAE": mae, "MSE": mse, "RMSE": rmse,
                 "MAPE": mape, "sMAPE": smape,
                 "train_size": train_size, "test_size": test_size},
    }


# ── Section 3: Charts ─────────────────────────────────────────────────────────

def generate_charts(df: pd.DataFrame, results: dict, config: PipelineConfig):
    """Generate and save all available charts."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _apply_style()
    charts = config.available_charts
    print(f"\n  Saving charts to '{OUTPUT_DIR}/' …")

    # Chart 1 — Power profile (requires power breakdown columns)
    if charts.get("power_overview"):
        _chart_power_profile(df)
    else:
        print("  [skip] 01_power_profile — missing inferencePowerDraw / trainingPowerDraw")

    # Chart 2 — Daily energy + carbon pattern (always available)
    _chart_daily_pattern(df, config)

    # Chart 3 — Grid carbon intensity + optimal training window
    if "carbonIntensityFactor" in df.columns:
        _chart_training_window(df)
    else:
        print("  [skip] 03_training_window — missing carbonIntensityFactor")

    # Chart 4 — Energy forecast (always available)
    if "totalEnergyConsumption" in results:
        _chart_forecast(
            df, results["totalEnergyConsumption"],
            metric="totalEnergyConsumption",
            ylabel="Energy Consumption (kWh/h)",
            title_prefix="Total Energy",
            color=C["energy"],
            filename="04_energy_forecast.png",
        )

    # Chart 5 — Carbon forecast
    if "carbonEmissions" in results:
        _chart_forecast(
            df, results["carbonEmissions"],
            metric="carbonEmissions",
            ylabel="Carbon Emissions (kgCO2e/h)",
            title_prefix="Carbon Emissions",
            color=C["carbon"],
            filename="05_carbon_forecast.png",
        )
    else:
        print("  [skip] 05_carbon_forecast — missing carbonEmissions")

    # Chart 6 — SCI by hour
    if charts.get("sci_by_hour") or ("sciPerInference" in results and "sciPerInference" in df.columns):
        _chart_sci_by_hour(df)
    else:
        print("  [skip] 06_sci_by_hour — missing sciPerInference")

    print(f"  Done.")


def _chart_power_profile(df: pd.DataFrame):
    """Chart 1: Two-week power breakdown — inference base + training spikes + PUE overhead."""
    show = df.head(336).copy()   # first 2 weeks

    fig, ax = plt.subplots(figsize=(14, 5))

    # Stacked area: inference at the bottom, training on top
    ax.fill_between(
        show["ds"], 0, show["inferencePowerDraw"] / 1000,
        color=C["inference"], alpha=0.75, label="Inference GPU power",
    )
    ax.fill_between(
        show["ds"],
        show["inferencePowerDraw"] / 1000,
        (show["inferencePowerDraw"] + show["trainingPowerDraw"]) / 1000,
        color=C["training"], alpha=0.85, label="Training GPU power (spikes)",
    )
    # Total infra line includes cooling/PUE overhead
    ax.plot(
        show["ds"], show["totalInfrastructurePower"] / 1000,
        color=C["overhead"], linewidth=0.9, alpha=0.7,
        label="Total infrastructure (GPU + CPU + cooling/PUE)",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Power (kW)")
    ax.set_title(
        "AI Cluster Power Consumption — First 2 Weeks\n"
        "Orange spikes = training runs. Gap above stacked area = PUE cooling overhead."
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "01_power_profile.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 01_power_profile.png")


def _chart_daily_pattern(df: pd.DataFrame, config: PipelineConfig):
    """Chart 2: Average energy (bar) + carbon (line) by hour of day."""
    df = df.copy()
    df["hour"] = df["ds"].dt.hour

    hourly_energy = df.groupby("hour")["totalEnergyConsumption"].mean()

    # Colour each bar by relative intensity (green=low, red=high)
    norm   = plt.Normalize(hourly_energy.min(), hourly_energy.max())
    cmap   = plt.cm.RdYlGn_r
    colors = [cmap(norm(v)) for v in hourly_energy.values]

    has_carbon = "carbonEmissions" in config.fit_metrics

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.bar(
        hourly_energy.index, hourly_energy.values,
        color=colors, edgecolor="white", linewidth=0.4, alpha=0.9,
        label="Avg energy (kWh)",
    )
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Avg Energy Consumption (kWh/h)")
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])

    if has_carbon:
        hourly_carbon = df.groupby("hour")["carbonEmissions"].mean()
        ax2 = ax1.twinx()
        ax2.plot(
            hourly_carbon.index, hourly_carbon.values,
            color=C["carbon"], linewidth=2.5, marker="o", markersize=5,
            label="Avg carbon (kgCO2e)",
        )
        ax2.set_ylabel("Avg Carbon Emissions (kgCO2e/h)", color=C["carbon"])
        ax2.tick_params(axis="y", labelcolor=C["carbon"])
        ax2.spines["right"].set_visible(True)

    subtitle = "Red/orange bars = high-consumption hours. Carbon line shows when grid intensity amplifies emissions."
    ax1.set_title(f"When Does the Cluster Consume Most Energy?\n{subtitle}")

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    if has_carbon:
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    else:
        ax1.legend(h1, l1, loc="upper left", fontsize=9)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_daily_pattern.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 02_daily_pattern.png")


def _chart_training_window(df: pd.DataFrame):
    """Chart 3: Grid carbon intensity by hour + recommended training window."""
    df = df.copy()
    df["hour"] = df["ds"].dt.hour

    hourly_intensity = df.groupby("hour")["carbonIntensityFactor"].mean()
    greenest_hours   = sorted(hourly_intensity.nsmallest(6).index.tolist())
    ws, we           = greenest_hours[0], greenest_hours[-1]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Shade the recommended training window
    ax1.axvspan(
        ws - 0.5, we + 0.5,
        color=C["green"], alpha=0.12,
        label=f"Recommended training window ({ws:02d}:00–{we:02d}:00)",
    )

    ax1.plot(
        hourly_intensity.index, hourly_intensity.values,
        color=C["carbon"], linewidth=2.5, marker="o", markersize=5,
        label="Grid carbon intensity (kgCO2/kWh)",
    )
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Grid Carbon Intensity (kgCO2/kWh)", color=C["carbon"])
    ax1.tick_params(axis="y", labelcolor=C["carbon"])
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])

    # Annotate the recommended window
    mid = (ws + we) / 2
    ax1.annotate(
        f"Best window\n{ws:02d}:00–{we:02d}:00",
        xy=(mid, hourly_intensity[greenest_hours].mean()),
        xytext=(mid, hourly_intensity.max() * 0.85),
        fontsize=9, ha="center", color=C["green"],
        arrowprops=dict(arrowstyle="->", color=C["green"], lw=1.5),
    )

    # Overlay training frequency if available
    if "trainingActive" in df.columns:
        training_freq = df.groupby("hour")["trainingActive"].mean() * 100
        ax2 = ax1.twinx()
        ax2.bar(
            training_freq.index, training_freq.values,
            color=C["training"], alpha=0.25, width=0.8,
            label="Current training frequency (%)",
        )
        ax2.set_ylabel("Training Active (%)", color=C["training"])
        ax2.tick_params(axis="y", labelcolor=C["training"])
        ax2.spines["right"].set_visible(True)
        h2, l2 = ax2.get_legend_handles_labels()
    else:
        h2, l2 = [], []

    ax1.set_title(
        "When Is the Grid Cleanest? → Optimal Training Window\n"
        "Schedule GPU training during the green zone to minimise carbon per training run."
    )

    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_training_window.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 03_training_window.png")


def _chart_forecast(
    df: pd.DataFrame,
    result: dict,
    metric: str,
    ylabel: str,
    title_prefix: str,
    color: str,
    filename: str,
):
    """Charts 4 & 5: Historical context + model fit + 7-day forecast with CI."""
    forecast = result["forecast"]
    eval_m   = result["eval"]
    n        = len(df)

    # Split forecast into historical fitted and future
    hist_fc   = forecast.iloc[:n]
    future_fc = forecast.iloc[n:]

    # Show last 2 weeks of history as context
    context_n       = min(336, n)
    context_actual  = df.iloc[-context_n:]
    context_fitted  = hist_fc.iloc[-context_n:]

    fig, ax = plt.subplots(figsize=(14, 5))

    # Historical actual (grey background context)
    ax.plot(
        context_actual["ds"], context_actual[metric],
        color=C["actual"], linewidth=1.0, alpha=0.85,
        label="Actual (last 2 weeks)",
    )

    # Prophet fit on the same window
    ax.plot(
        context_fitted["ds"], context_fitted["yhat"],
        color=color, linewidth=1.2, linestyle="--", alpha=0.65,
        label=f"Prophet fit  (sMAPE {eval_m['sMAPE']:.1f}%)",
    )

    # Future forecast
    ax.plot(
        future_fc["ds"], future_fc["yhat"],
        color=C["forecast"], linewidth=2.2,
        label="7-day forecast",
    )

    # Confidence interval
    if "yhat_lower" in future_fc.columns:
        ax.fill_between(
            future_fc["ds"],
            future_fc["yhat_lower"], future_fc["yhat_upper"],
            color=C["forecast"], alpha=0.15,
            label="95% confidence interval",
        )

    # Vertical line at forecast start
    forecast_start = df["ds"].iloc[-1]
    ax.axvline(x=forecast_start, color="#FF5722", linestyle=":", linewidth=1.5, alpha=0.8)
    ylim = ax.get_ylim()
    ax.text(
        forecast_start, ylim[1],
        "  → Forecast", color="#FF5722", fontsize=9, va="top",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{title_prefix} — 7-Day Forecast\n"
        f"Model accuracy on held-out test set: sMAPE {eval_m['sMAPE']:.1f}%  |  "
        f"MAPE {eval_m['MAPE']:.1f}%  |  RMSE {eval_m['RMSE']:.4f}"
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def _chart_sci_by_hour(df: pd.DataFrame):
    """Chart 6: Software Carbon Intensity per inference request by hour of day."""
    df = df.copy()
    df["hour"] = df["ds"].dt.hour
    hourly_sci = df.groupby("hour")["sciPerInference"].mean()
    daily_avg  = hourly_sci.mean()

    norm   = plt.Normalize(hourly_sci.min(), hourly_sci.max())
    cmap   = plt.cm.RdYlGn_r
    colors = [cmap(norm(v)) for v in hourly_sci.values]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(
        hourly_sci.index, hourly_sci.values,
        color=colors, edgecolor="white", linewidth=0.4,
    )

    # Daily average reference line
    ax.axhline(
        daily_avg, color="#424242", linestyle="--", linewidth=1.2,
        label=f"Daily average: {daily_avg:.2f} gCO2e/req",
    )

    # Annotate best and worst hours
    best_h  = int(hourly_sci.idxmin())
    worst_h = int(hourly_sci.idxmax())
    ax.annotate(
        f"Best\n{best_h:02d}:00\n{hourly_sci[best_h]:.2f}",
        xy=(best_h, hourly_sci[best_h]),
        xytext=(best_h, hourly_sci[best_h] + (hourly_sci.max() - hourly_sci.min()) * 0.12),
        ha="center", fontsize=8, color=C["green"],
        arrowprops=dict(arrowstyle="->", color=C["green"], lw=1.2),
    )
    ax.annotate(
        f"Worst\n{worst_h:02d}:00\n{hourly_sci[worst_h]:.2f}",
        xy=(worst_h, hourly_sci[worst_h]),
        xytext=(worst_h, hourly_sci[worst_h] + (hourly_sci.max() - hourly_sci.min()) * 0.08),
        ha="center", fontsize=8, color=C["carbon"],
        arrowprops=dict(arrowstyle="->", color=C["carbon"], lw=1.2),
    )

    # Colourbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="gCO2e per request (green = efficient)")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("SCI (gCO2e per inference request)")
    ax.set_title(
        "Carbon Cost Per Inference Request — by Hour of Day (SCI Metric)\n"
        "Lower = more efficient. Peak hours batch better → lower SCI. "
        "Night hours: low load but high grid intensity."
    )
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.legend(fontsize=9)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "06_sci_by_hour.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 06_sci_by_hour.png")


# ── Section 4: Print Insights ──────────────────────────────────────────────────

def print_insights(df: pd.DataFrame, results: dict, config: PipelineConfig):
    """Print key findings and scheduling recommendations."""
    print("\n" + "=" * 60)
    print("  KEY FINDINGS & RECOMMENDATIONS")
    print("=" * 60)

    # Model accuracy
    print("\n  FORECAST ACCURACY (held-out 20% test set):")
    print(f"  {'Metric':<28} {'MAPE':>7} {'sMAPE':>8} {'RMSE':>10}")
    print(f"  {'-'*55}")
    for metric in config.metrics:
        if metric not in results:
            continue
        e = results[metric]["eval"]
        print(f"  {metric:<28} {e['MAPE']:>6.1f}%  {e['sMAPE']:>6.1f}%  {e['RMSE']:>10.4f}")

    # Scheduling insight
    if "carbonIntensityFactor" in df.columns:
        df2 = df.copy()
        df2["hour"] = df2["ds"].dt.hour
        hi = df2.groupby("hour")["carbonIntensityFactor"].mean()
        greenest = sorted(hi.nsmallest(6).index.tolist())
        reduction = (hi.drop(greenest).mean() - hi[greenest].mean()) / hi.drop(greenest).mean() * 100
        print(f"\n  SCHEDULING OPPORTUNITY:")
        print(f"  Training during {greenest[0]:02d}:00–{greenest[-1]:02d}:00 saves"
              f" ~{reduction:.0f}% carbon intensity vs other hours.")

    # SCI range
    if "sciPerInference" in df.columns:
        df2 = df.copy()
        df2["hour"] = df2["ds"].dt.hour
        sci_h = df2.groupby("hour")["sciPerInference"].mean()
        best = int(sci_h.idxmin())
        worst = int(sci_h.idxmax())
        ratio = sci_h[worst] / sci_h[best]
        print(f"\n  SCI EFFICIENCY:")
        print(f"  {best:02d}:00 requests are {ratio:.1f}x more carbon-efficient than {worst:02d}:00 requests.")

    print("\n" + "=" * 60)
    print(f"  Charts saved to '{OUTPUT_DIR}/'")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df, config = load_and_summarise(DATA_PATH)

    print("\n  Fitting Prophet models …")
    results = fit_models(df, config)

    generate_charts(df, results, config)
    print_insights(df, results, config)


if __name__ == "__main__":
    main()
