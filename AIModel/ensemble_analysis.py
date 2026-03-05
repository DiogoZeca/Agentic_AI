"""
Ensemble Model Comparison: Prophet vs TimesFM vs Ensemble

Generates 2 focused charts that answer: which forecast model should you trust?

Charts produced:
  07_forecast_comparison.png  — Actual vs all 3 models on a sample test week
  08_model_accuracy.png       — sMAPE comparison across all metrics

Usage:
    python ensemble_analysis.py
    DATA_PATH=data/my_data.csv python ensemble_analysis.py
"""
import math
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

from ensemble_model import EnsembleForecaster
from data_loader import load_and_validate, build_pipeline_config, PipelineConfig

OUTPUT_DIR = "analysis_output"
DATA_PATH = os.environ.get("DATA_PATH", "data/sample_energy_data.csv")

# Consistent colour palette (matches carbon_analysis.py)
C = {
    "prophet":  "#1565C0",   # deep blue
    "timesfm":  "#2E7D32",   # dark green
    "ensemble": "#E53935",   # red
    "actual":   "#212121",   # near-black
}


def _fmt_pct(v: float) -> str:
    """Format a percentage for display; returns 'N/A' for nan (e.g. MAPE when actuals are zero)."""
    return "N/A" if math.isnan(v) else f"{v:.1f}%"


def _safe_mape(actuals: np.ndarray, preds: np.ndarray) -> float:
    """MAPE with zero-actual guard — returns nan instead of inf when actuals contain 0."""
    if np.any(actuals == 0):
        return float("nan")
    return float(np.mean(np.abs((actuals - preds) / actuals)) * 100)


def _apply_style():
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


# ── Section 1: Load Data ──────────────────────────────────────────────────────

def load_data(path: str) -> tuple[pd.DataFrame, PipelineConfig]:
    """Load CSV and build pipeline config.

    Raises SystemExit with a human-readable message if the file is missing
    or the schema is invalid — appropriate for a CLI script entry point.
    """
    try:
        df = load_and_validate(path)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, flush=True)
        raise SystemExit(1) from exc
    config = build_pipeline_config(df)

    print("=" * 65)
    print("  ENSEMBLE COMPARISON: Prophet vs TimesFM vs Ensemble")
    print("=" * 65)
    print(f"  Records : {len(df)}")
    print(f"  Period  : {df['ds'].min().date()} → {df['ds'].max().date()}")
    print(f"  Metrics : {', '.join(config.metrics)}")
    return df, config


# ── Section 2: Three-Way Evaluation ──────────────────────────────────────────

def run_evaluation(df: pd.DataFrame, config: PipelineConfig) -> dict:
    """Run Prophet / TimesFM / Ensemble evaluation for each metric."""
    all_results = {}

    for metric in config.fit_metrics:
        regs = config.regressor_map.get(metric, [])
        label = f" (+ {', '.join(regs)})" if regs else ""
        print(f"\n  Evaluating '{metric}'{label} …")

        ens = EnsembleForecaster(regressors=regs)
        res = ens.evaluate(df, metric)

        print(f"  {'Model':<12} {'sMAPE':>8} {'MAPE':>8} {'RMSE':>10}")
        print(f"  {'-'*42}")
        for name in ["prophet", "timesfm", "ensemble"]:
            m = res[name]
            print(f"  {name:<12} {_fmt_pct(m['sMAPE']):>8}  {_fmt_pct(m['MAPE']):>8}  {m['RMSE']:>10.4f}")

        stats = ens.get_residual_stats()
        print(f"  Residual autocorr lag-1: {stats['autocorr_lag1']:.3f}  |  alpha: {stats['alpha']:.2f}")

        all_results[metric] = {"ensemble": ens, "results": res, "residual_stats": stats}

    # Derive SCI if both components were evaluated
    if config.has_sci and "carbonEmissions" in all_results and "functionalUnit" in all_results:
        print("\n  Deriving SCI from carbonEmissions / functionalUnit …")
        all_results["softwareCarbonIntensity"] = _derive_sci_ensemble(df, all_results)
        sci_res = all_results["softwareCarbonIntensity"]["results"]
        print(f"  {'Model':<12} {'sMAPE':>8} {'MAPE':>8} {'RMSE':>10}")
        print(f"  {'-'*42}")
        for name in ["prophet", "timesfm", "ensemble"]:
            m = sci_res[name]
            print(f"  {name:<12} {_fmt_pct(m['sMAPE']):>8}  {_fmt_pct(m['MAPE']):>8}  {m['RMSE']:>10.4f}")

    return all_results


def _derive_sci_ensemble(df: pd.DataFrame, all_results: dict) -> dict:
    """Derive SCI three-way comparison from component model predictions.

    SCI = carbonEmissions / functionalUnit  (kgCO2e/req — no ×1000)
    """
    ens_c = all_results["carbonEmissions"]["ensemble"]
    ens_r = all_results["functionalUnit"]["ensemble"]

    n          = len(df)
    train_size = int(n * 0.8)
    test_size  = n - train_size

    if "softwareCarbonIntensity" in df.columns:
        sci_actuals = df.iloc[train_size:]["softwareCarbonIntensity"].values
    else:
        s = df["carbonEmissions"] / df["functionalUnit"].clip(lower=1)
        sci_actuals = s.iloc[train_size:].values

    c_test = ens_c.get_test_predictions()
    r_test = ens_r.get_test_predictions()

    sci_results = {}
    for name in ("prophet", "timesfm", "ensemble"):
        c_preds = c_test[name]
        r_preds = np.maximum(r_test[name], 1)
        sci_preds = c_preds / r_preds   # kgCO2e/req

        mae   = float(np.mean(np.abs(sci_preds - sci_actuals)))
        mse   = float(np.mean((sci_preds - sci_actuals) ** 2))
        rmse  = float(np.sqrt(mse))
        if np.any(sci_actuals == 0):
            mape = float("nan")
        else:
            mape = float(np.mean(np.abs((sci_actuals - sci_preds) / sci_actuals)) * 100)
        smape = float(np.mean(
            2 * np.abs(sci_actuals - sci_preds)
            / (np.abs(sci_actuals) + np.abs(sci_preds) + 1e-10)
        ) * 100)
        sci_results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse,
                             "MAPE": mape, "sMAPE": smape,
                             "train_size": train_size, "test_size": test_size}

    return {
        "ensemble": ens_c,
        "results": sci_results,
        "residual_stats": all_results["carbonEmissions"]["residual_stats"],
    }


# ── Section 3: Charts ─────────────────────────────────────────────────────────

def generate_charts(df: pd.DataFrame, all_results: dict, config: PipelineConfig):
    """Generate and save comparison charts."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _apply_style()
    print(f"\n  Saving charts to '{OUTPUT_DIR}/' …")

    # Chart 7 — Actual vs 3 models on test week
    primary = "carbonEmissions" if "carbonEmissions" in all_results else list(all_results.keys())[0]
    _chart_forecast_comparison(df, all_results[primary]["ensemble"], primary)

    # Chart 8 — sMAPE comparison across all metrics
    _chart_model_accuracy(all_results, config)

    print("  Done.")


def _chart_forecast_comparison(df: pd.DataFrame, ens: EnsembleForecaster, metric: str):
    """Chart 7: One representative test week — actual vs all 3 model predictions."""
    preds   = ens.get_test_predictions()
    actuals = preds["actuals"]
    n_test  = len(actuals)

    # Show a representative week (first 168 h of the test set)
    show_n  = min(168, n_test)
    idx     = range(show_n)

    prophet_mape  = _safe_mape(actuals[:show_n], preds["prophet"][:show_n])
    timesfm_mape  = _safe_mape(actuals[:show_n], preds["timesfm"][:show_n])
    ensemble_mape = _safe_mape(actuals[:show_n], preds["ensemble"][:show_n])

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(idx, actuals[:show_n],
            color=C["actual"], linewidth=1.8, label="Actual", zorder=4)
    ax.plot(idx, preds["prophet"][:show_n],
            color=C["prophet"], linewidth=1.2, linestyle="--",
            label=f"Prophet  (MAPE {_fmt_pct(prophet_mape)})", alpha=0.8, zorder=3)
    ax.plot(idx, preds["timesfm"][:show_n],
            color=C["timesfm"], linewidth=1.2, linestyle=":",
            label=f"TimesFM  (MAPE {_fmt_pct(timesfm_mape)})", alpha=0.8, zorder=3)
    ax.plot(idx, preds["ensemble"][:show_n],
            color=C["ensemble"], linewidth=1.6,
            label=f"Ensemble (MAPE {_fmt_pct(ensemble_mape)})", alpha=0.9, zorder=4)

    # Highlight worst gap for Prophet to make differences visible
    errors = np.abs(actuals[:show_n] - preds["prophet"][:show_n])
    worst_idx = int(np.argmax(errors))
    ax.annotate(
        f"Largest Prophet error\n({errors[worst_idx]:.3f})",
        xy=(worst_idx, actuals[worst_idx]),
        xytext=(worst_idx + 8, actuals[worst_idx] * 1.05),
        fontsize=8, color=C["prophet"],
        arrowprops=dict(arrowstyle="->", color=C["prophet"], lw=1.2),
    )

    ax.set_xlabel("Hours into test period")
    ax.set_ylabel(metric)
    ax.set_title(
        f"Do the Models Actually Differ? — {metric} (First Test Week)\n"
        "Solid black = ground truth. Models with lower MAPE track more closely."
    )
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "07_forecast_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 07_forecast_comparison.png")


def _chart_model_accuracy(all_results: dict, config: PipelineConfig):
    """Chart 8: sMAPE grouped bars — which model wins for each metric?"""
    display_metrics = [m for m in config.metrics if m in all_results]
    model_names     = ["prophet", "timesfm", "ensemble"]
    x     = np.arange(len(display_metrics))
    width = 0.24

    fig, ax = plt.subplots(figsize=(max(10, len(display_metrics) * 3), 5))

    for i, name in enumerate(model_names):
        smapes = [all_results[m]["results"][name]["sMAPE"] for m in display_metrics]
        bars   = ax.bar(
            x + i * width, smapes, width,
            label=name.capitalize(),
            color=C[name], alpha=0.85,
        )
        for bar, val in zip(bars, smapes):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
            )

    # Star the winner for each metric
    for j, metric in enumerate(display_metrics):
        best_name  = min(model_names, key=lambda k: all_results[metric]["results"][k]["sMAPE"])
        best_val   = all_results[metric]["results"][best_name]["sMAPE"]
        best_i     = model_names.index(best_name)
        bar_x      = x[j] + best_i * width + width / 2
        ax.text(bar_x, best_val + 2.5, "★", ha="center", fontsize=12, color="gold")

    label_map = {
        "consumption":              "Energy\n(kWh/h)",
        "carbonEmissions":          "Carbon\n(kgCO2e/h)",
        "softwareCarbonIntensity":  "SCI\n(kgCO2e/req)",
        "functionalUnit":           "Requests\n(req/h)",
    }
    short_labels = [label_map.get(m, m[:12]) for m in display_metrics]

    ax.set_xlabel("Metric")
    ax.set_ylabel("sMAPE (%) — lower is better")
    ax.set_title(
        "Which Model Is Most Accurate? (sMAPE on held-out test set)\n"
        "★ = best model per metric. sMAPE used instead of MAPE (robust to near-zero values)."
    )
    ax.set_xticks(x + width)
    ax.set_xticklabels(short_labels, fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "08_model_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 08_model_accuracy.png")


# ── Section 4: Summary ────────────────────────────────────────────────────────

def print_summary(all_results: dict, config: PipelineConfig):
    """Print the three-way sMAPE comparison table."""
    print("\n" + "=" * 65)
    print("  MODEL ACCURACY SUMMARY (sMAPE — lower is better)")
    print("=" * 65)

    display_metrics = [m for m in config.metrics if m in all_results]

    print(f"\n  {'Metric':<28} {'Prophet':>9} {'TimesFM':>9} {'Ensemble':>9}  Best")
    print(f"  {'-'*60}")
    for metric in display_metrics:
        res  = all_results[metric]["results"]
        vals = {n: res[n]["sMAPE"] for n in ["prophet", "timesfm", "ensemble"]}
        best = min(vals, key=vals.get)
        row  = f"  {metric:<28}"
        for name in ["prophet", "timesfm", "ensemble"]:
            marker = "*" if name == best else " "
            row += f"  {vals[name]:>6.1f}%{marker}"
        row += f"  {best}"
        print(row)

    print(f"\n  (* = best for that metric)")

    print(f"\n  IMPROVEMENT OVER PROPHET BASELINE:")
    for metric in display_metrics:
        res          = all_results[metric]["results"]
        prophet_smape = res["prophet"]["sMAPE"]
        best_name    = min(res, key=lambda k: res[k]["sMAPE"])
        improvement  = prophet_smape - res[best_name]["sMAPE"]
        print(f"    {metric}: {improvement:+.1f}% sMAPE gain  (winner: {best_name})")

    print("\n" + "=" * 65)
    print(f"  Charts saved to '{OUTPUT_DIR}/'")
    print("=" * 65)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df, config = load_data(DATA_PATH)

    print("\n  Running three-way evaluation …")
    all_results = run_evaluation(df, config)

    generate_charts(df, all_results, config)
    print_summary(all_results, config)


if __name__ == "__main__":
    main()
