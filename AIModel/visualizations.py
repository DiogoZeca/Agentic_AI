"""
Visualization utilities for Energy & Carbon Prophet analysis.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional
from prophet import Prophet


def plot_forecast(
    model: Prophet,
    forecast: pd.DataFrame,
    title: str = "Energy Consumption Forecast",
    xlabel: str = "Date",
    ylabel: str = "Value",
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot forecast with uncertainty intervals.

    Args:
        model: Trained Prophet model
        forecast: Forecast DataFrame from model.predict()
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot actual data points
    ax.plot(
        model.history["ds"],
        model.history["y"],
        "k.",
        label="Actual",
        alpha=0.5
    )

    # Plot forecast line
    ax.plot(
        forecast["ds"],
        forecast["yhat"],
        color="blue",
        label="Forecast"
    )

    # Plot uncertainty interval
    ax.fill_between(
        forecast["ds"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        color="blue",
        alpha=0.2,
        label="Uncertainty"
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig


def plot_components(
    model: Prophet,
    forecast: pd.DataFrame,
    figsize: tuple = (12, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot trend and seasonality components.

    Args:
        model: Trained Prophet model
        forecast: Forecast DataFrame
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    # Use Prophet's built-in component plot
    fig = model.plot_components(forecast, figsize=figsize)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Components figure saved to {save_path}")

    return fig


def plot_metric_comparison(
    df: pd.DataFrame,
    metrics: list,
    figsize: tuple = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot multiple metrics on the same time axis for comparison.

    Args:
        df: DataFrame with 'ds' column and metric columns
        metrics: List of column names to plot
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)

    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors

    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        ax.plot(df["ds"], df[metric], color=colors[i % 10], linewidth=0.8)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{metric} over time")

    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison figure saved to {save_path}")

    return fig


def plot_daily_pattern(
    df: pd.DataFrame,
    metric: str,
    figsize: tuple = (10, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Show average pattern by hour of day.

    Args:
        df: DataFrame with 'ds' column
        metric: Column to analyze
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    df = df.copy()
    df["hour"] = pd.to_datetime(df["ds"]).dt.hour

    hourly_avg = df.groupby("hour")[metric].agg(["mean", "std"])

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(hourly_avg.index, hourly_avg["mean"], "b-", linewidth=2, label="Mean")
    ax.fill_between(
        hourly_avg.index,
        hourly_avg["mean"] - hourly_avg["std"],
        hourly_avg["mean"] + hourly_avg["std"],
        alpha=0.3,
        label="±1 Std Dev"
    )

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel(metric)
    ax.set_title(f"Daily Pattern: {metric}")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_weekly_pattern(
    df: pd.DataFrame,
    metric: str,
    figsize: tuple = (10, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Show average pattern by day of week.

    Args:
        df: DataFrame with 'ds' column
        metric: Column to analyze
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure
    """
    df = df.copy()
    df["dayofweek"] = pd.to_datetime(df["ds"]).dt.dayofweek

    daily_avg = df.groupby("dayofweek")[metric].agg(["mean", "std"])

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(days, daily_avg["mean"], yerr=daily_avg["std"], capsize=5, alpha=0.7)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel(metric)
    ax.set_title(f"Weekly Pattern: {metric}")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


if __name__ == "__main__":
    # Example usage
    from data_generator import generate_energy_carbon_data
    from prophet_model import EnergyProphet

    print("Generating sample data...")
    df = generate_energy_carbon_data(periods=24 * 7)  # 1 week

    print("Plotting daily pattern...")
    fig1 = plot_daily_pattern(df, "consumption")
    plt.show()

    print("Plotting weekly pattern...")
    fig2 = plot_weekly_pattern(df, "consumption")
    plt.show()

    print("Plotting metric comparison...")
    fig3 = plot_metric_comparison(
        df,
        ["consumption", "carbonEmissions", "greenConsumptionPercentage"]
    )
    plt.show()
