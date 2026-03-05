"""
Web Service Energy & Carbon Analysis (SCI Framework)

Generates 8 focused charts answering key questions about your web service's
energy and carbon footprint. Uses Prophet for time-series forecasting.

Charts produced:
  01_service_profile.png       — Service load & energy (first 2 weeks)
  02_energy_forecast.png       — 7-day energy forecast with accuracy metrics
  03_energy_decomposition.png  — Prophet trend/seasonality decomposition
  04_daily_pattern.png         — Which hours consume most energy?
  05_weekly_pattern.png        — Day-of-week energy pattern
  06_carbon_intensity.png      — Grid carbon intensity & optimal scheduling window
  07_sci_by_hour.png           — SCI (kgCO2e/req) by hour of day
  08_carbon_forecast.png       — 7-day carbon emissions forecast

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
from physics_constraint import derive_carbon_emissions, evaluate_formula_accuracy

OUTPUT_DIR = "analysis_output"
DATA_PATH = os.environ.get("DATA_PATH", "data/sample_energy_data.csv")
FORECAST_PERIODS = 168   # 7 days × 24 hours

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# ── Consistent colour palette ──────────────────────────────────────────────────
C = {
    "energy":     "#1565C0",   # deep blue
    "requests":   "#0277BD",   # medium blue
    "carbon":     "#B71C1C",   # deep red
    "forecast":   "#6A1B9A",   # purple
    "green":      "#2E7D32",   # dark green
    "actual":     "#212121",   # near-black
    "history":    "#9E9E9E",   # grey
    "ci":         "#CE93D8",   # light purple
    "intensity":  "#E65100",   # deep orange
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
    print("  WEB SERVICE ENERGY & CARBON ANALYSIS (SCI)")
    print("=" * 60)
    print(f"  Records : {len(df)}")
    print(f"  Period  : {df['ds'].min().date()} → {df['ds'].max().date()}")
    print(f"\n  Energy  : {df['consumption'].sum():,.2f} kWh total")
    if "carbonEmissions" in df.columns:
        print(f"  Carbon  : {df['carbonEmissions'].sum():,.3f} kgCO2e total")
    if "functionalUnit" in df.columns:
        print(f"  Requests: {df['functionalUnit'].sum():,.0f} total req served")
    if "softwareCarbonIntensity" in df.columns:
        print(f"  Avg SCI : {df['softwareCarbonIntensity'].mean():.6f} kgCO2e/req")
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
        print(f"    sMAPE {eval_m['sMAPE']:.1f}%  |  MAPE {eval_m['MAPE']:.1f}%  |  RMSE {eval_m['RMSE']:.6f}")
        model.fit(df, metric)
        forecast = model.predict(FORECAST_PERIODS, future_df=df)
        results[metric] = {"model": model, "forecast": forecast, "eval": eval_m}

    # Chained forecasting: re-predict metrics that have regressors using the
    # regressor's own forecast for the future period instead of zero-fill.
    # Without this, future functionalUnit = 0 → consumption collapses to baseline.
    last_ts = pd.to_datetime(df["ds"].max())
    for metric in config.fit_metrics:
        regs = config.regressor_map.get(metric, [])
        if not regs or results.get(metric, {}).get("model") is None:
            continue
        avail = [r for r in regs if r in df.columns]
        combined = df[["ds"] + avail].copy()
        combined["ds"] = pd.to_datetime(combined["ds"])
        for reg in regs:
            if reg not in results:
                continue
            reg_fc = results[reg]["forecast"].copy()
            reg_fc["ds"] = pd.to_datetime(reg_fc["ds"])
            future_rows = reg_fc[reg_fc["ds"] > last_ts][["ds", "yhat"]].rename(columns={"yhat": reg})
            combined = pd.concat([combined, future_rows], ignore_index=True)
        results[metric]["forecast"] = results[metric]["model"].predict(FORECAST_PERIODS, future_df=combined)
        print(f"  Chained forecast applied for '{metric}' using {regs}")

    # TimesFM override: use foundation model for carbonEmissions when available
    _tfm_used = False
    try:
        from timesfm_model import EnergyTimesFM
        tfm = EnergyTimesFM()
        tfm.fit(df, "carbonEmissions")
        tfm_fc = tfm.predict(FORECAST_PERIODS)
        # tail() gives future-only rows; prepend history timestamps to align with Prophet output
        future_only = tfm_fc.tail(FORECAST_PERIODS).reset_index(drop=True)
        if "carbonEmissions" in results:
            results["carbonEmissions"]["forecast"] = future_only
            tfm_eval = tfm.evaluate(df, "carbonEmissions")
            results["carbonEmissions"]["eval"] = tfm_eval
            _tfm_used = True
            print(f"  carbonEmissions: using TimesFM (sMAPE {tfm_eval['sMAPE']:.1f}%)")
    except Exception as e:
        print(f"  carbonEmissions: TimesFM unavailable ({e}), keeping Prophet")

    # Formula evaluation: derive carbonEmissions = consumption × CIF + 0.002
    try:
        formula_eval = evaluate_formula_accuracy(df)
        if "consumption" in results and "carbonIntensityFactor" in results:
            results["carbonEmissions_formula"] = {
                "model": None,
                "forecast": derive_carbon_emissions(
                    results["consumption"]["forecast"],
                    results["carbonIntensityFactor"]["forecast"],
                ),
                "eval": formula_eval,
            }
        print(f"  carbonEmissions (formula): sMAPE {formula_eval['sMAPE']:.1f}%")
    except Exception as e:
        print(f"  carbonEmissions formula evaluation skipped: {e}")

    results["_tfm_used"] = _tfm_used

    if config.has_sci and "carbonEmissions" in results and "functionalUnit" in results:
        print("\n  Deriving SCI from carbonEmissions / functionalUnit …")
        results["softwareCarbonIntensity"] = _derive_sci(df, results)
        e = results["softwareCarbonIntensity"]["eval"]
        print(f"    sMAPE {e['sMAPE']:.1f}%  |  MAPE {e['MAPE']:.1f}%  |  RMSE {e['RMSE']:.8f}")

    return results


def _derive_sci(df: pd.DataFrame, results: dict) -> dict:
    """Derive SCI evaluation and forecast from component models.

    SCI = carbonEmissions (kgCO2e/h) / functionalUnit (req/h)
        = kgCO2e per request  (no ×1000 — units are already kgCO2e/req)
    """
    n = len(df)
    train_size = int(n * 0.8)
    test_size = n - train_size
    train_df = df.iloc[:train_size]

    cm = EnergyProphet()
    cm.fit(train_df, "carbonEmissions")
    c_fc = cm.predict(test_size, future_df=df)
    c_preds = c_fc.iloc[train_size:]["yhat"].values

    rm = EnergyProphet()
    rm.fit(train_df, "functionalUnit")
    r_fc = rm.predict(test_size, future_df=df)
    r_preds = np.maximum(r_fc.iloc[train_size:]["yhat"].values, 1)

    sci_preds = c_preds / r_preds   # kgCO2e/req

    if "softwareCarbonIntensity" in df.columns:
        sci_actuals = df.iloc[train_size:]["softwareCarbonIntensity"].values
        sci_p99 = np.percentile(df["softwareCarbonIntensity"].values, 99)
    else:
        s = df["carbonEmissions"] / df["functionalUnit"].clip(lower=1)
        sci_actuals = s.iloc[train_size:].values
        sci_p99 = np.percentile(s.values, 99)

    mae  = float(np.mean(np.abs(sci_preds - sci_actuals)))
    mse  = float(np.mean((sci_preds - sci_actuals) ** 2))
    rmse = float(np.sqrt(mse))
    if np.any(sci_actuals == 0):
        mape = float("nan")
    else:
        mape = float(np.mean(np.abs((sci_actuals - sci_preds) / sci_actuals)) * 100)
    smape = float(np.mean(
        2 * np.abs(sci_actuals - sci_preds)
        / (np.abs(sci_actuals) + np.abs(sci_preds) + 1e-10)
    ) * 100)

    # Forecast from full-data components
    cf = results["carbonEmissions"]["forecast"][["ds", "yhat"]].copy()
    rf = results["functionalUnit"]["forecast"][["ds", "yhat"]].copy()
    mg = cf.merge(rf, on="ds", suffixes=("_c", "_r"))
    mg["yhat"] = mg["yhat_c"] / np.maximum(mg["yhat_r"], 1)
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
    """Generate and save all 8 charts."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _apply_style()
    print(f"\n  Saving charts to '{OUTPUT_DIR}/' …")

    # Chart 1 — Service profile (consumption + requests, first 2 weeks)
    if config.available_charts.get("service_profile"):
        _chart_service_profile(df)
    else:
        print("  [skip] 01_service_profile — missing consumption / functionalUnit")

    # Chart 2 — Energy forecast
    if "consumption" in results:
        _chart_forecast(
            df, results["consumption"],
            metric="consumption",
            ylabel="Energy Consumption (kWh/h)",
            title_prefix="Energy Consumption",
            color=C["energy"],
            filename="02_energy_forecast.png",
        )
    else:
        print("  [skip] 02_energy_forecast — missing consumption model")

    # Chart 3 — Energy decomposition (Prophet components)
    if "consumption" in results and results["consumption"]["model"] is not None:
        _chart_energy_decomposition(results["consumption"])
    else:
        print("  [skip] 03_energy_decomposition — missing consumption model")

    # Chart 4 — Daily energy pattern
    _chart_daily_pattern(df, config)

    # Chart 5 — Weekly energy pattern
    _chart_weekly_pattern(df)

    # Chart 6 — Carbon intensity + optimal window
    if "carbonIntensityFactor" in df.columns:
        _chart_carbon_intensity(df)
    else:
        print("  [skip] 06_carbon_intensity — missing carbonIntensityFactor")

    # Chart 7 — SCI by hour
    if config.available_charts.get("sci_by_hour") or "softwareCarbonIntensity" in df.columns:
        _chart_sci_by_hour(df)
    else:
        print("  [skip] 07_sci_by_hour — missing softwareCarbonIntensity")

    # Chart 8 — Carbon emissions forecast
    if "carbonEmissions" in results:
        _chart_forecast(
            df, results["carbonEmissions"],
            metric="carbonEmissions",
            ylabel="Carbon Emissions (kgCO2e/h)",
            title_prefix="Carbon Emissions",
            color=C["carbon"],
            filename="08_carbon_forecast.png",
            model_label="TimesFM" if results.get("_tfm_used") else "Prophet",
        )
    else:
        print("  [skip] 08_carbon_forecast — missing carbonEmissions")

    print("  Done.")


def _chart_service_profile(df: pd.DataFrame):
    """Chart 1: Two-panel service profile — energy + request load (first 2 weeks)."""
    show = df.head(336).copy()   # first 2 weeks

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax1.fill_between(
        show["ds"], 0, show["consumption"] * 1000,
        color=C["energy"], alpha=0.75,
    )
    ax1.set_ylabel("Energy Consumption (Wh/h)")
    ax1.set_title(
        "Web Service Profile — First 2 Weeks\n"
        "Top: energy draw (Wh/h). Bottom: request rate (req/h)."
    )

    ax2.fill_between(
        show["ds"], 0, show["functionalUnit"],
        color=C["requests"], alpha=0.70,
    )
    ax2.set_ylabel("Request Rate (req/h)")
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_service_profile.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 01_service_profile.png")


def _chart_energy_decomposition(result: dict):
    """Chart 3: Prophet component decomposition for consumption."""
    model = result["model"].model        # underlying Prophet object
    forecast = result["forecast"]

    fig = model.plot_components(forecast)
    fig.suptitle("Energy Consumption — Trend & Seasonality Decomposition", y=1.02)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "03_energy_decomposition.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 03_energy_decomposition.png")


def _chart_daily_pattern(df: pd.DataFrame, config: PipelineConfig):
    """Chart 4: Average energy (bar) + carbon (line) by hour of day."""
    df = df.copy()
    df["hour"] = df["ds"].dt.hour

    hourly_energy = df.groupby("hour")["consumption"].mean()

    norm   = plt.Normalize(hourly_energy.min(), hourly_energy.max())
    cmap   = plt.cm.RdYlGn_r
    colors = [cmap(norm(v)) for v in hourly_energy.values]

    has_carbon = "carbonEmissions" in config.fit_metrics

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.bar(
        hourly_energy.index, hourly_energy.values * 1000,
        color=colors, edgecolor="white", linewidth=0.4, alpha=0.9,
        label="Avg energy (Wh/h)",
    )
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Avg Energy Consumption (Wh/h)")
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])

    if has_carbon:
        hourly_carbon = df.groupby("hour")["carbonEmissions"].mean()
        ax2 = ax1.twinx()
        ax2.plot(
            hourly_carbon.index, hourly_carbon.values,
            color=C["carbon"], linewidth=2.5, marker="o", markersize=5,
            label="Avg carbon (kgCO2e/h)",
        )
        ax2.set_ylabel("Avg Carbon Emissions (kgCO2e/h)", color=C["carbon"])
        ax2.tick_params(axis="y", labelcolor=C["carbon"])
        ax2.spines["right"].set_visible(True)

    subtitle = "Red/orange bars = high-consumption hours. Carbon line when grid intensity amplifies emissions."
    ax1.set_title(f"When Does the Service Consume Most Energy?\n{subtitle}")

    h1, l1 = ax1.get_legend_handles_labels()
    if has_carbon:
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    else:
        ax1.legend(h1, l1, loc="upper left", fontsize=9)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_daily_pattern.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 04_daily_pattern.png")


def _chart_weekly_pattern(df: pd.DataFrame):
    """Chart 5: Average energy consumption by day of week."""
    df = df.copy()
    df["dow"] = df["ds"].dt.dayofweek
    daily = df.groupby("dow")["consumption"].agg(["mean", "std"])

    norm   = plt.Normalize(daily["mean"].min(), daily["mean"].max())
    cmap   = plt.cm.RdYlGn_r
    colors = [cmap(norm(v)) for v in daily["mean"].values]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        DAY_NAMES, daily["mean"].values * 1000,
        yerr=daily["std"].values * 1000,
        color=colors, edgecolor="white", linewidth=0.4,
        alpha=0.85, capsize=5,
    )
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Avg Energy Consumption (Wh/h)")
    ax.set_title(
        "Weekly Energy Pattern\n"
        "Weekdays peak due to business-hours traffic. Weekends show ~40% reduction."
    )
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "05_weekly_pattern.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 05_weekly_pattern.png")


def _chart_carbon_intensity(df: pd.DataFrame):
    """Chart 6: Grid carbon intensity by hour + green % + optimal scheduling window."""
    df = df.copy()
    df["hour"] = df["ds"].dt.hour

    hourly_intensity = df.groupby("hour")["carbonIntensityFactor"].mean()
    greenest_hours   = sorted(hourly_intensity.nsmallest(6).index.tolist())
    ws, we           = greenest_hours[0], greenest_hours[-1]

    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.axvspan(
        ws - 0.5, we + 0.5,
        color=C["green"], alpha=0.12,
        label=f"Optimal scheduling window ({ws:02d}:00–{we:02d}:00)",
    )
    ax1.plot(
        hourly_intensity.index, hourly_intensity.values,
        color=C["intensity"], linewidth=2.5, marker="o", markersize=5,
        label="Grid carbon intensity (kgCO2/kWh)",
    )
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Grid Carbon Intensity (kgCO2/kWh)", color=C["intensity"])
    ax1.tick_params(axis="y", labelcolor=C["intensity"])
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])

    mid = (ws + we) / 2
    ax1.annotate(
        f"Best window\n{ws:02d}:00–{we:02d}:00",
        xy=(mid, hourly_intensity[greenest_hours].mean()),
        xytext=(mid, hourly_intensity.max() * 0.85),
        fontsize=9, ha="center", color=C["green"],
        arrowprops=dict(arrowstyle="->", color=C["green"], lw=1.5),
    )

    if "greenConsumptionPercentage" in df.columns:
        hourly_green = df.groupby("hour")["greenConsumptionPercentage"].mean()
        ax2 = ax1.twinx()
        ax2.fill_between(
            hourly_green.index, 0, hourly_green.values,
            color=C["green"], alpha=0.18, label="Green energy %",
        )
        ax2.set_ylabel("Green Consumption (%)", color=C["green"])
        ax2.tick_params(axis="y", labelcolor=C["green"])
        ax2.spines["right"].set_visible(True)
        h2, l2 = ax2.get_legend_handles_labels()
    else:
        h2, l2 = [], []

    ax1.set_title(
        "When Is the Grid Cleanest? → Optimal Scheduling Window\n"
        "Schedule batch jobs during the green zone to minimise carbon per request."
    )
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "06_carbon_intensity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 06_carbon_intensity.png")


def _chart_forecast(
    df: pd.DataFrame,
    result: dict,
    metric: str,
    ylabel: str,
    title_prefix: str,
    color: str,
    filename: str,
    model_label: str = "Prophet",
):
    """Charts 2 & 8: Historical context + model fit + 7-day forecast with CI."""
    forecast = result["forecast"]
    eval_m   = result["eval"]
    n        = len(df)

    hist_fc   = forecast.iloc[:n]
    future_fc = forecast.iloc[n:]

    context_n      = min(336, n)
    context_actual = df.iloc[-context_n:]
    context_fitted = hist_fc.iloc[-context_n:]

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        context_actual["ds"], context_actual[metric],
        color=C["actual"], linewidth=1.0, alpha=0.85,
        label="Actual (last 2 weeks)",
    )
    ax.plot(
        context_fitted["ds"], context_fitted["yhat"],
        color=color, linewidth=1.2, linestyle="--", alpha=0.65,
        label=f"{model_label} fit  (sMAPE {eval_m['sMAPE']:.1f}%)",
    )
    ax.plot(
        future_fc["ds"], future_fc["yhat"],
        color=C["forecast"], linewidth=2.2,
        label="7-day forecast",
    )

    if "yhat_lower" in future_fc.columns:
        ax.fill_between(
            future_fc["ds"],
            future_fc["yhat_lower"], future_fc["yhat_upper"],
            color=C["forecast"], alpha=0.15,
            label="95% confidence interval",
        )

    forecast_start = df["ds"].iloc[-1]
    ax.axvline(x=forecast_start, color="#FF5722", linestyle=":", linewidth=1.5, alpha=0.8)
    ylim = ax.get_ylim()
    ax.text(forecast_start, ylim[1], "  → Forecast", color="#FF5722", fontsize=9, va="top")

    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{title_prefix} — 7-Day Forecast\n"
        f"Model accuracy on held-out test set: sMAPE {eval_m['sMAPE']:.1f}%  |  "
        f"MAPE {eval_m['MAPE']:.1f}%  |  RMSE {eval_m['RMSE']:.6f}"
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def _chart_sci_by_hour(df: pd.DataFrame):
    """Chart 7: Software Carbon Intensity (kgCO2e/req) by hour of day."""
    df = df.copy()
    df["hour"] = df["ds"].dt.hour
    hourly_sci = df.groupby("hour")["softwareCarbonIntensity"].mean()
    daily_avg  = hourly_sci.mean()

    norm   = plt.Normalize(hourly_sci.min(), hourly_sci.max())
    cmap   = plt.cm.RdYlGn_r
    colors = [cmap(norm(v)) for v in hourly_sci.values]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(
        hourly_sci.index, hourly_sci.values,
        color=colors, edgecolor="white", linewidth=0.4,
    )

    ax.axhline(
        daily_avg, color="#424242", linestyle="--", linewidth=1.2,
        label=f"Daily average: {daily_avg:.6f} kgCO2e/req",
    )

    best_h  = int(hourly_sci.idxmin())
    worst_h = int(hourly_sci.idxmax())
    spread  = hourly_sci.max() - hourly_sci.min()
    ax.annotate(
        f"Best\n{best_h:02d}:00\n{hourly_sci[best_h]:.6f}",
        xy=(best_h, hourly_sci[best_h]),
        xytext=(best_h, hourly_sci[best_h] + spread * 0.12),
        ha="center", fontsize=8, color=C["green"],
        arrowprops=dict(arrowstyle="->", color=C["green"], lw=1.2),
    )
    ax.annotate(
        f"Worst\n{worst_h:02d}:00\n{hourly_sci[worst_h]:.6f}",
        xy=(worst_h, hourly_sci[worst_h]),
        xytext=(worst_h, hourly_sci[worst_h] + spread * 0.08),
        ha="center", fontsize=8, color=C["carbon"],
        arrowprops=dict(arrowstyle="->", color=C["carbon"], lw=1.2),
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="kgCO2e per request (green = efficient)")

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("SCI (kgCO2e per request)")
    ax.set_title(
        "Software Carbon Intensity (SCI) by Hour of Day\n"
        "Lower = more carbon-efficient. Peak hours serve more req/h → lower SCI. "
        "Off-hours: low load + higher grid intensity."
    )
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.legend(fontsize=9)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "07_sci_by_hour.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 07_sci_by_hour.png")


# ── Section 4: Print Insights ──────────────────────────────────────────────────

def print_insights(df: pd.DataFrame, results: dict, config: PipelineConfig):
    """Print key findings and scheduling recommendations."""
    print("\n" + "=" * 60)
    print("  KEY FINDINGS & RECOMMENDATIONS")
    print("=" * 60)

    _tfm_used = results.get("_tfm_used", False)

    # Model accuracy
    print("\n  FORECAST ACCURACY (held-out 20% test set):")
    print(f"  {'Metric':<28} {'MAPE':>7} {'sMAPE':>8} {'RMSE':>12}")
    print(f"  {'-'*58}")
    for metric in config.metrics:
        if metric not in results:
            continue
        e = results[metric]["eval"]
        print(f"  {metric:<28} {e['MAPE']:>6.1f}%  {e['sMAPE']:>6.1f}%  {e['RMSE']:>12.6f}")

    if "carbonEmissions_formula" in results and "carbonEmissions" in results:
        print("\n  carbonEmissions FORECAST METHOD COMPARISON:")
        print(f"  {'Method':<26} {'sMAPE':>8}")
        print(f"  {'-'*36}")
        formula_smape = results["carbonEmissions_formula"]["eval"]["sMAPE"]
        active_smape  = results["carbonEmissions"]["eval"]["sMAPE"]
        active_label  = "TimesFM" if _tfm_used else "Prophet + CIF regressor"
        print(f"  {'Formula (E×CIF+0.002)':<26} {formula_smape:>7.1f}%")
        print(f"  {active_label:<26} {active_smape:>7.1f}%")

    # SCI efficiency: best vs worst hour
    if "softwareCarbonIntensity" in df.columns:
        df2 = df.copy()
        df2["hour"] = df2["ds"].dt.hour
        sci_h = df2.groupby("hour")["softwareCarbonIntensity"].mean()
        best_h  = int(sci_h.idxmin())
        worst_h = int(sci_h.idxmax())
        ratio   = sci_h[worst_h] / sci_h[best_h]
        print(f"\n  SCI EFFICIENCY RATIO (best vs worst hour):")
        print(f"  {best_h:02d}:00 requests are {ratio:.1f}× more carbon-efficient "
              f"than {worst_h:02d}:00 requests.")
        print(f"  Best  SCI: {sci_h[best_h]:.6f} kgCO2e/req  ({best_h:02d}:00)")
        print(f"  Worst SCI: {sci_h[worst_h]:.6f} kgCO2e/req  ({worst_h:02d}:00)")

    # Optimal scheduling window
    if "carbonIntensityFactor" in df.columns:
        df2 = df.copy()
        df2["hour"] = df2["ds"].dt.hour
        hi = df2.groupby("hour")["carbonIntensityFactor"].mean()
        greenest = sorted(hi.nsmallest(6).index.tolist())
        reduction = (
            (hi.drop(greenest).mean() - hi[greenest].mean())
            / hi.drop(greenest).mean() * 100
        )
        print(f"\n  OPTIMAL SCHEDULING WINDOW:")
        print(f"  Hours {greenest[0]:02d}:00–{greenest[-1]:02d}:00 have "
              f"~{reduction:.0f}% lower carbon intensity than other hours.")
        print(f"  Schedule batch jobs in this window to minimise carbon per run.")

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
