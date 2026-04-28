"""CPU Spike Predictor — Streamlit Model Dashboard.

Reads training artifacts from data/full_run/ (or ARTIFACTS_DIR env var) and
renders four interactive tabs:

  1. Overview          — model metrics, per-class PR-AUC, CV fold stability
  2. Threshold Explorer — alarm threshold trade-off explorer per model
  3. Calibration        — Brier score before/after isotonic calibration (60m)
  4. Feature Attribution — SHAP / gain importance bar chart

No live inference is performed — all data comes from the artifacts written
during training (spike_config.json, feature_importance.csv, cv_results.csv).

Usage
-----
    cd AIModel
    pip install -r requirements-demo.txt
    streamlit run demo.py

    # Custom artifacts path:
    ARTIFACTS_DIR=path/to/full_run streamlit run demo.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────

_DEFAULT_ARTIFACTS_DIR = Path(__file__).parent / "data" / "full_run"

_MODEL_KEYS: list[str] = ["60m", "15m", "ovr"]

_MODEL_LABELS: dict[str, str] = {
    "60m": "60m Severity (multiclass)",
    "15m": "15m Binary",
    "ovr": "Severe OVR (binary)",
}

_MODEL_SUBDIRS: dict[str, str] = {
    "60m": "models/spike",
    "15m": "models/spike_15m",
    "ovr": "models/spike_severe_ovr",
}

# Explicit field mapping per model type.
# Multiclass (60m) and binary (15m, ovr) configs use different key names for
# the same conceptual metrics.  Accessing via this map prevents KeyError.
_METRICS_60M: dict[str, str | None] = {
    "pr_auc":     "macro_pr_auc",
    "pr_auc_cal": "macro_pr_auc_calibrated",   # isotonic calibration applied
    "roc_auc":    "macro_roc_auc",
    "precision":  "alarm_precision",
    "recall":     "alarm_recall",
    "f1":         None,   # not stored at top level for multiclass
}
_METRICS_BIN: dict[str, str | None] = {
    "pr_auc":     "pr_auc",
    "pr_auc_cal": None,   # binary models have no isotonic calibration
    "roc_auc":    "roc_auc",
    "precision":  "precision",
    "recall":     "recall",
    "f1":         "f1",
}
_MODEL_METRICS: dict[str, dict[str, str | None]] = {
    "60m": _METRICS_60M,
    "15m": _METRICS_BIN,
    "ovr": _METRICS_BIN,
}

_CLASS_LABELS: dict[int, str] = {0: "No Spike", 1: "Moderate", 2: "Severe"}
_CLASS_COLORS: dict[int, str] = {0: "#2ecc71",  1: "#f39c12",  2: "#e74c3c"}

# Color for each CV fold bar chart series
_CV_BAR_COLOR = "#3498db"
_CV_MEAN_COLOR = "#e67e22"

# Feature group taxonomy — used for color-coding the attribution chart.
_FEATURE_GROUPS: dict[str, str] = {
    "total_cpu":              "Raw Signal",
    "peak_cpu":               "Raw Signal",
    "total_mem":              "Raw Signal",
    "peak_mem":               "Raw Signal",
    "disk_io":                "Raw Signal",
    "n_tasks":                "Raw Signal",
    "cpu_per_task":           "Raw Signal",
    "cpu_lag_1":              "Lag",
    "cpu_lag_12":             "Lag",
    "cpu_lag_24":             "Lag",
    "cpu_ewma_6":             "Trend",
    "cpu_ewma_24":            "Trend",
    "cpu_delta_1":            "Trend",
    "cpu_delta_2":            "Trend",
    "cpu_rolling_std_6":      "Trend",
    "task_dominance":         "Machine-Relative",
    "cpu_vs_p95":             "Machine-Relative",
    "cpu_vs_p95_delta":       "Machine-Relative",
    "peak_cpu_vs_p95":        "Machine-Relative",
    "cpu_vs_p99":             "Machine-Relative",
    "peak_cpu_vs_p99":        "Machine-Relative",
    "band_position":          "Machine-Relative",
    "band_width":             "Machine-Relative",
    "spike_now":              "Spike History",
    "spike_in_last_1":        "Spike History",
    "spike_in_last_3":        "Spike History",
    "spike_in_last_6":        "Spike History",
    "time_since_last_spike":  "Spike History",
    "cpu_spike_rate_24":      "Spike History",
    "spike_severe_now":       "Spike History (Severe)",
    "spike_severe_in_last_1": "Spike History (Severe)",
    "spike_severe_in_last_3": "Spike History (Severe)",
    "spike_severe_in_last_6": "Spike History (Severe)",
    "cluster_cpu_p90":        "Cluster",
    "machine_rank_in_cluster":"Cluster",
    "hour_sin":               "Time",
    "hour_cos":               "Time",
}

_GROUP_COLORS: dict[str, str] = {
    "Raw Signal":             "#3498db",
    "Lag":                    "#9b59b6",
    "Trend":                  "#1abc9c",
    "Machine-Relative":       "#e67e22",
    "Spike History":          "#e74c3c",
    "Spike History (Severe)": "#c0392b",
    "Cluster":                "#f39c12",
    "Time":                   "#95a5a6",
    "Other":                  "#bdc3c7",
}


# ── Data loading ───────────────────────────────────────────────────────────────
#
# _load_all is the single @st.cache_data entry point.  Keeping individual
# helpers as plain functions avoids nested-cache behaviour and makes them
# testable without a Streamlit context.


def _read_config(path: Path) -> dict | None:
    """Read a spike_config.json; return None if the file does not exist."""
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _read_feature_importance(path: Path) -> pd.DataFrame | None:
    """Read feature_importance.csv; fill any null shap_mean_abs with 0."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["shap_mean_abs"] = df["shap_mean_abs"].fillna(0.0)
    return df


def _read_cv_results(path: Path) -> pd.DataFrame | None:
    """Read cv_results.csv; return None if the file does not exist."""
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def _load_all(artifacts_dir: Path) -> dict[str, Any]:
    """Load all model artifacts once and cache the result.

    Returns a dict with keys "artifacts_dir" (Path) and one entry per model
    key ("60m", "15m", "ovr"), each containing:
        config             : dict from spike_config.json, or None if absent
        feature_importance : DataFrame from feature_importance.csv, or None
        cv_results         : DataFrame from cv_results.csv, or None
    """
    data: dict[str, Any] = {"artifacts_dir": artifacts_dir}
    for key in _MODEL_KEYS:
        model_dir = artifacts_dir / _MODEL_SUBDIRS[key]
        data[key] = {
            "config":             _read_config(model_dir / "spike_config.json"),
            "feature_importance": _read_feature_importance(model_dir / "feature_importance.csv"),
            "cv_results":         _read_cv_results(model_dir / "cv_results.csv"),
        }
    return data


def _resolve_artifacts_dir() -> Path:
    env = os.environ.get("ARTIFACTS_DIR")
    return Path(env) if env else _DEFAULT_ARTIFACTS_DIR


# ── Metric helpers ─────────────────────────────────────────────────────────────


def _get_metric(config: dict, model_key: str, metric_key: str) -> float | None:
    """Safely extract a final_metrics value using the per-model-type field map."""
    field = _MODEL_METRICS[model_key].get(metric_key)
    if field is None:
        return None
    return config.get("final_metrics", {}).get(field)


def _fmt(value: float | None, precision: int = 3) -> str:
    """Format a float for table display; return '—' for None."""
    if value is None:
        return "—"
    return f"{value:.{precision}f}"


def _random_baseline_pr_auc(config: dict, model_key: str) -> float | None:
    """Compute the random-classifier PR-AUC baseline from test-set class rates.

    For a random classifier, PR-AUC per class ≈ positive rate for that class.
    Macro PR-AUC baseline = mean of all class positive rates.
    For binary models: baseline = spike_rate_test (single positive rate).
    """
    if model_key == "60m":
        rates = config.get("data", {}).get("class_rates_test")
        if rates:
            return float(np.mean(list(rates.values())))
        return None
    else:
        rate = config.get("spike_rate_test")
        return float(rate) if rate is not None else None


# ── Tab 1: Overview ────────────────────────────────────────────────────────────


def _render_overview(data: dict) -> None:
    config_60m = data["60m"]["config"]
    if config_60m is None:
        st.error("60m model config not found — run the full training pipeline first.")
        return

    # ── Top-level KPIs ────────────────────────────────────────────────────────
    trained_at       = config_60m.get("trained_at", "unknown")[:10]
    training_rows    = config_60m.get("data", {}).get("training_rows", 0)
    training_machines = config_60m.get("data", {}).get("training_machines", 0)

    pr_auc_raw = _get_metric(config_60m, "60m", "pr_auc")
    pr_auc_cal = _get_metric(config_60m, "60m", "pr_auc_cal")
    roc_auc    = _get_metric(config_60m, "60m", "roc_auc")
    cv_mean    = config_60m.get("cv_summary", {}).get("cv_macro_pr_auc_mean")
    cv_std     = config_60m.get("cv_summary", {}).get("cv_macro_pr_auc_std")

    cal_delta = (
        f"+{pr_auc_cal - pr_auc_raw:.3f} vs raw"
        if pr_auc_cal is not None and pr_auc_raw is not None
        else None
    )
    cv_label = (
        f"{cv_mean:.3f} ± {cv_std:.3f}"
        if cv_mean is not None and cv_std is not None
        else "—"
    )

    baseline = _random_baseline_pr_auc(config_60m, "60m")
    vs_baseline = (
        f"+{pr_auc_cal - baseline:.3f} vs random"
        if pr_auc_cal is not None and baseline is not None
        else None
    )
    roc_vs_random = (
        f"+{roc_auc - 0.5:.3f} vs random (0.5)"
        if roc_auc is not None
        else None
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "PR-AUC (calibrated, 60m)", _fmt(pr_auc_cal), delta=vs_baseline,
        help=(
            "Precision-Recall AUC — the main metric for imbalanced problems like spike detection.\n\n"
            "Measures how well the model ranks machines by spike risk across *all possible thresholds*. "
            "1.0 = perfect. Random classifier ≈ mean class prevalence (shown below). "
            "After isotonic calibration, predicted probabilities better match actual spike rates."
        ),
    )
    col2.metric(
        "ROC-AUC (60m)", _fmt(roc_auc), delta=roc_vs_random,
        help=(
            "ROC-AUC — probability that a randomly chosen spiking machine scores higher than "
            "a randomly chosen non-spiking machine.\n\n"
            "0.5 = random, 1.0 = perfect. Less informative than PR-AUC when classes are imbalanced, "
            "but useful as a secondary check."
        ),
    )
    col3.metric(
        "CV PR-AUC (5-fold)", cv_label,
        delta="gap < 0.01 = stable" if cv_mean and pr_auc_raw and abs(cv_mean - pr_auc_raw) < 0.01 else None,
        help=(
            "Cross-validation PR-AUC on 5 chronological folds (mean ± std).\n\n"
            "Shows whether the model generalises consistently over time. "
            "Low std (< 0.01) means the model isn't just lucky on one time window. "
            "A small gap between CV mean and final test PR-AUC means no overfitting."
        ),
    )
    col4.metric(
        "Trained on", trained_at,
        delta=f"{training_machines:,} machines · {training_rows:,} rows",
        help="Date the model artifacts were written. Training data size shown as delta.",
    )

    if baseline is not None:
        st.caption(
            f"**Random classifier baseline** (macro PR-AUC ≈ mean class prevalence): **{baseline:.3f}**  —  "
            f"anything above this means the model is learning a real signal."
        )

    st.divider()

    # ── Per-class PR-AUC bar chart (60m model) ────────────────────────────────
    st.subheader("60m Model — Per-Class PR-AUC")

    final_metrics = config_60m.get("final_metrics", {})
    per_class_values = {
        label: final_metrics.get(f"pr_auc_class_{k}")
        for k, label in _CLASS_LABELS.items()
    }

    if all(v is not None for v in per_class_values.values()):
        # Per-class random baseline = class prevalence in test set
        class_rates_test = config_60m.get("data", {}).get("class_rates_test", {})
        baseline_values = {
            label: class_rates_test.get(str(k)) or class_rates_test.get(k)
            for k, label in _CLASS_LABELS.items()
        }

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Model PR-AUC",
            x=list(per_class_values.keys()),
            y=list(per_class_values.values()),
            marker_color=[_CLASS_COLORS[k] for k in _CLASS_LABELS],
            text=[f"{v:.3f}" for v in per_class_values.values()],
            textposition="outside",
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        ))
        if all(v is not None for v in baseline_values.values()):
            fig.add_trace(go.Bar(
                name="Random baseline",
                x=list(baseline_values.keys()),
                y=list(baseline_values.values()),
                marker_color="rgba(150,150,150,0.45)",
                text=[f"{v:.3f}" for v in baseline_values.values()],
                textposition="outside",
                hovertemplate="Random baseline %{x}: %{y:.4f}<extra></extra>",
            ))
        fig.update_layout(
            barmode="group",
            yaxis=dict(title="PR-AUC", range=[0, 1.12]),
            xaxis_title="Class",
            height=330,
            margin=dict(t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "**Random baseline** = class prevalence in the test set. "
            "A model PR-AUC well above baseline means the model is genuinely useful for that class. "
            "Rare classes (Moderate, Severe) have low baselines — even modest PR-AUC is meaningful."
        )

    st.divider()

    # ── All-models comparison table ───────────────────────────────────────────
    st.subheader("All Models — Metrics Comparison")

    rows = []
    for key in _MODEL_KEYS:
        cfg = data[key]["config"]
        bl = _random_baseline_pr_auc(cfg, key) if cfg else None
        bl_str = f"{bl:.3f}" if bl is not None else "—"
        if cfg is None:
            rows.append({
                "Model":           _MODEL_LABELS[key],
                "PR-AUC":          "—",
                "PR-AUC (cal)":    "—",
                "Random Baseline": bl_str,
                "ROC-AUC":         "—",
                "Precision":       "—",
                "Recall":          "—",
                "Alarm Threshold": "—",
            })
        else:
            rows.append({
                "Model":           _MODEL_LABELS[key],
                "PR-AUC":          _fmt(_get_metric(cfg, key, "pr_auc")),
                "PR-AUC (cal)":    _fmt(_get_metric(cfg, key, "pr_auc_cal")),
                "Random Baseline": bl_str,
                "ROC-AUC":         _fmt(_get_metric(cfg, key, "roc_auc")),
                "Precision":       _fmt(_get_metric(cfg, key, "precision")),
                "Recall":          _fmt(_get_metric(cfg, key, "recall")),
                "Alarm Threshold": _fmt(cfg.get("alarm_threshold")),
            })

    st.dataframe(
        pd.DataFrame(rows).set_index("Model"),
        use_container_width=True,
    )

    st.caption(
        "**Random Baseline**: PR-AUC a random classifier would achieve (= mean class prevalence). "
        "Model PR-AUC should be well above this to be useful.  \n"
        "**ROC-AUC**: 0.5 = random, 1.0 = perfect — less sensitive to class imbalance than PR-AUC.  \n"
        "**PR-AUC (cal)**: after isotonic calibration (60m model only).  \n"
        "**Alarm Threshold**: chosen on the validation set to maximise F1."
    )

    st.divider()

    # ── CV fold stability ─────────────────────────────────────────────────────
    st.subheader("Cross-Validation Stability (5 Folds, Chronological)")

    cols = st.columns(3)
    for i, key in enumerate(_MODEL_KEYS):
        cv_df = data[key]["cv_results"]
        with cols[i]:
            st.caption(_MODEL_LABELS[key])
            if cv_df is None:
                st.warning("cv_results.csv not found.")
                continue

            mean_auc = float(cv_df["macro_pr_auc"].mean())
            fig = go.Figure()
            fig.add_bar(
                x=[f"Fold {int(f)}" for f in cv_df["fold"]],
                y=cv_df["macro_pr_auc"],
                marker_color=_CV_BAR_COLOR,
                text=[f"{v:.3f}" for v in cv_df["macro_pr_auc"]],
                textposition="outside",
                hovertemplate="Fold %{x}: %{y:.4f}<extra></extra>",
            )
            fig.add_hline(
                y=mean_auc,
                line_dash="dash",
                line_color=_CV_MEAN_COLOR,
                annotation_text=f"Mean {mean_auc:.3f}",
                annotation_position="top right",
            )
            fig.update_layout(
                yaxis=dict(title="PR-AUC", range=[0, 1]),
                height=270,
                margin=dict(t=30, b=10, l=10, r=10),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)


# ── Tab 2: Threshold Explorer ──────────────────────────────────────────────────


def _render_threshold_explorer(data: dict) -> None:
    model_key: str = st.selectbox(  # type: ignore[assignment]
        "Model",
        options=_MODEL_KEYS,
        format_func=lambda k: _MODEL_LABELS[k],
        key="thresh_model",
    )

    cfg = data[model_key]["config"]
    if cfg is None:
        st.warning(f"{_MODEL_LABELS[model_key]} model not found.")
        return

    sweep = cfg.get("alarm_threshold_sweep", [])
    if not sweep:
        st.warning("No threshold sweep data in this model's config.")
        return

    # Round sweep thresholds to remove floating-point noise (e.g. 0.75000001)
    thresholds = [round(row["threshold"], 4) for row in sweep]
    sweep_map  = {round(row["threshold"], 4): row for row in sweep}

    current = cfg.get("alarm_threshold", thresholds[0])
    nearest = min(thresholds, key=lambda t: abs(t - current))

    st.info(
        f"Current trained threshold: **{current:.2f}**  "
        f"({'p_moderate + p_severe' if model_key == '60m' else 'binary p_spike'})"
    )

    selected: float = st.select_slider(  # type: ignore[assignment]
        "Alarm Threshold",
        options=thresholds,
        value=nearest,
        format_func=lambda v: f"{v:.2f}",
        help=(
            "60m model: threshold on p_spike = p_moderate + p_severe.  "
            "15m / OVR: threshold on binary p_spike directly."
        ),
    )

    row = sweep_map[selected]

    false_alarms_per_day = row["alarms_per_day"] * (1.0 - row["precision"]) if row["precision"] else 0.0
    true_alarms_per_day  = row["alarms_per_day"] * row["precision"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Precision", f"{row['precision']:.1%}",
        help=(
            "Of every alarm fired, what fraction was a real spike?\n\n"
            f"At this threshold: {true_alarms_per_day:.1f} real-spike alarms/day, "
            f"{false_alarms_per_day:.1f} false alarms/day.\n\n"
            "High precision = fewer wasted scheduler interventions."
        ),
    )
    c2.metric(
        "Recall", f"{row['recall']:.1%}",
        help=(
            "Of every real spike that happened, what fraction did the model catch?\n\n"
            "Low recall = spikes that hit without warning. "
            "High recall = fewer surprises, but usually at the cost of more false alarms."
        ),
    )
    c3.metric(
        "F1", f"{row['f1']:.3f}",
        help=(
            "Harmonic mean of Precision and Recall — a single balanced score.\n\n"
            "Useful when you care equally about catching spikes and avoiding false alarms. "
            "The trained threshold was chosen to maximise this on the validation set."
        ),
    )
    c4.metric(
        "Alarms / day", f"{row['alarms_per_day']:.1f}",
        help="Total alarms fired per day across all machines in the cluster (real + false).",
    )
    c5.metric(
        "False alarms / day", f"{false_alarms_per_day:.1f}",
        delta=f"{false_alarms_per_day / row['alarms_per_day']:.0%} of all alarms" if row['alarms_per_day'] else None,
        delta_color="off",
        help=(
            "Alarms where the model predicted a spike but no spike occurred.\n\n"
            "These cost the scheduler unnecessary interventions. "
            "Move the threshold right (higher) to reduce false alarms, "
            "but this will also reduce recall (more missed spikes)."
        ),
    )

    st.divider()

    df_sweep = pd.DataFrame([
        {
            "threshold":           round(r["threshold"], 4),
            "precision":           r["precision"],
            "recall":              r["recall"],
            "f1":                  r["f1"],
            "alarms_per_day":      r["alarms_per_day"],
            "false_alarms_per_day": r["alarms_per_day"] * (1.0 - r["precision"]),
        }
        for r in sweep
    ])

    # Precision-Recall curve
    col_pr, col_alarms = st.columns([3, 2])

    with col_pr:
        st.subheader("Precision-Recall Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_sweep["recall"],
            y=df_sweep["precision"],
            mode="lines+markers",
            marker=dict(size=7, color=_CV_BAR_COLOR),
            line=dict(color=_CV_BAR_COLOR, width=2),
            name="All thresholds",
            customdata=df_sweep["threshold"],
            hovertemplate=(
                "Threshold: %{customdata:.2f}<br>"
                "Precision: %{y:.3f}<br>"
                "Recall: %{x:.3f}<extra></extra>"
            ),
        ))
        fig.add_trace(go.Scatter(
            x=[row["recall"]],
            y=[row["precision"]],
            mode="markers",
            marker=dict(size=15, color="#e74c3c", symbol="star"),
            name=f"Selected ({selected:.2f})",
            hovertemplate=(
                f"Selected threshold {selected:.2f}<br>"
                f"Precision: {row['precision']:.3f}<br>"
                f"Recall: {row['recall']:.3f}<extra></extra>"
            ),
        ))
        fig.update_layout(
            xaxis=dict(title="Recall",    range=[0, 1.05]),
            yaxis=dict(title="Precision", range=[0, 1.05]),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=40, b=30),
            hovermode="closest",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_alarms:
        st.subheader("Alarms / Day vs Threshold")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_sweep["threshold"],
            y=df_sweep["alarms_per_day"],
            mode="lines+markers",
            name="Total alarms",
            marker=dict(size=7, color="#9b59b6"),
            line=dict(color="#9b59b6", width=2),
            hovertemplate="Threshold: %{x:.2f}<br>Total alarms/day: %{y:.1f}<extra></extra>",
        ))
        fig2.add_trace(go.Scatter(
            x=df_sweep["threshold"],
            y=df_sweep["false_alarms_per_day"],
            mode="lines+markers",
            name="False alarms",
            marker=dict(size=7, color="#e74c3c"),
            line=dict(color="#e74c3c", width=2, dash="dot"),
            hovertemplate="Threshold: %{x:.2f}<br>False alarms/day: %{y:.1f}<extra></extra>",
        ))
        fig2.add_vline(
            x=selected,
            line_dash="dash",
            line_color="#2c3e50",
            annotation_text=f"selected: {selected:.2f}",
            annotation_position="top right",
        )
        fig2.update_layout(
            xaxis_title="Threshold",
            yaxis_title="Alarms / day",
            height=400,
            margin=dict(t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "**False alarms** (red dashed) = alarms that fired but no spike occurred — "
            "these cost the scheduler unnecessary interventions. "
            "The gap between total and false alarms = real catches."
        )

    with st.expander("Full sweep table"):
        st.dataframe(
            df_sweep.rename(columns={
                "threshold":            "Threshold",
                "precision":            "Precision",
                "recall":               "Recall",
                "f1":                   "F1",
                "alarms_per_day":       "Total alarms/day",
                "false_alarms_per_day": "False alarms/day",
            }),
            use_container_width=True,
            hide_index=True,
        )


# ── Tab 3: Calibration ─────────────────────────────────────────────────────────


def _render_calibration(data: dict) -> None:
    config_60m = data["60m"]["config"]
    if config_60m is None:
        st.error("60m model config not found.")
        return

    cal = config_60m.get("calibration")
    if cal is None:
        st.info("No calibration section found in the 60m config.")
        return

    st.info(
        f"Method: **{cal.get('method', '?')}** — "
        f"fit on **{cal.get('fit_on', '?')}** — "
        f"sklearn {cal.get('sklearn_version', '?')}"
    )

    # Build a per-class summary DataFrame
    per_class_raw = cal.get("per_class", {})
    rows = []
    for k, label in _CLASS_LABELS.items():
        c = per_class_raw.get(f"class_{k}", {})
        brier_raw = c.get("brier_raw")
        brier_cal = c.get("brier_cal")
        ece_cal   = c.get("ece_cal")
        improvement = (
            (brier_raw - brier_cal) / brier_raw * 100
            if brier_raw and brier_cal and brier_raw > 0
            else None
        )
        rows.append({
            "class_key":    k,
            "Class":        label,
            "Brier (raw)":  brier_raw,
            "Brier (cal)":  brier_cal,
            "Improvement":  improvement,
            "ECE (cal)":    ece_cal,
        })
    cal_df = pd.DataFrame(rows)

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        st.subheader("Brier Score Before vs After Calibration")
        st.caption(
            "**Lower is better.** Brier score = mean squared error of predicted probabilities.  \n"
            "Random classifier Brier ≈ p·(1−p) per class (e.g. ~0.05 for a 5% positive rate).  \n"
            "A well-calibrated model's Brier should be clearly below that. "
            "Green bars (calibrated) should be shorter than red (raw)."
        )

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Raw",
            x=cal_df["Class"],
            y=cal_df["Brier (raw)"],
            marker_color="#e74c3c",
            text=[f"{v:.4f}" if v else "—" for v in cal_df["Brier (raw)"]],
            textposition="outside",
        ))
        fig.add_trace(go.Bar(
            name="Calibrated",
            x=cal_df["Class"],
            y=cal_df["Brier (cal)"],
            marker_color="#2ecc71",
            text=[f"{v:.4f}" if v else "—" for v in cal_df["Brier (cal)"]],
            textposition="outside",
        ))
        y_max = cal_df["Brier (raw)"].dropna().max()
        fig.update_layout(
            barmode="group",
            yaxis=dict(title="Brier Score", range=[0, y_max * 1.35 if y_max else 0.3]),
            height=360,
            margin=dict(t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.subheader("Summary")
        display = cal_df[["Class", "Brier (raw)", "Brier (cal)", "Improvement", "ECE (cal)"]].copy()
        display["Brier (raw)"]  = display["Brier (raw)"].apply(_fmt)
        display["Brier (cal)"]  = display["Brier (cal)"].apply(_fmt)
        display["Improvement"]  = display["Improvement"].apply(
            lambda v: f"+{v:.1f}%" if v is not None else "—"
        )
        display["ECE (cal)"]    = display["ECE (cal)"].apply(lambda v: _fmt(v, 4))
        st.dataframe(display.set_index("Class"), use_container_width=True)

        st.caption(
            "**Improvement**: % reduction in Brier score from calibration — positive means calibration helped.  \n"
            "**ECE**: Expected Calibration Error — mean gap between predicted probability and actual positive rate.  \n"
            "ECE < 0.05 is considered well-calibrated; ECE < 0.02 is excellent."
        )


# ── Tab 4: Feature Attribution ─────────────────────────────────────────────────


def _render_feature_attribution(data: dict) -> None:
    fi_df = data["60m"]["feature_importance"]
    if fi_df is None:
        st.warning("Feature importance file not found for the 60m model.")
        return

    col_sort, col_n = st.columns([2, 1])
    with col_sort:
        sort_by: str = st.radio(  # type: ignore[assignment]
            "Sort by",
            options=["SHAP Mean Abs", "Gain Importance"],
            horizontal=True,
            key="feat_sort",
        )
    with col_n:
        top_n: int = st.slider(  # type: ignore[assignment]
            "Features to display",
            min_value=5,
            max_value=len(fi_df),
            value=min(20, len(fi_df)),
            step=5,
            key="feat_top_n",
        )

    sort_col = "shap_mean_abs" if sort_by == "SHAP Mean Abs" else "gain_importance"

    plot_df = (
        fi_df
        .sort_values(sort_col, ascending=False)
        .head(top_n)
        .copy()
    )
    plot_df["group"] = plot_df["feature"].map(_FEATURE_GROUPS).fillna("Other")
    # Sort ascending so the highest bar ends up at the top of the horizontal chart
    plot_df = plot_df.sort_values(sort_col, ascending=True)

    fig = px.bar(
        plot_df,
        x=sort_col,
        y="feature",
        color="group",
        color_discrete_map=_GROUP_COLORS,
        orientation="h",
        labels={
            sort_col:          sort_by,
            "feature":         "Feature",
            "group":           "Group",
            "shap_mean_abs":   "SHAP Mean Abs",
            "gain_importance": "Gain Importance",
        },
        hover_data={
            "gain_importance": ":.4f",
            "shap_mean_abs":   ":.4f",
            "group":           True,
        },
    )
    fig.update_layout(
        height=max(400, top_n * 24),
        margin=dict(t=20, b=20, l=10, r=20),
        legend=dict(title="Feature Group"),
        yaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full feature importance table"):
        full_df = (
            fi_df
            .sort_values(sort_col, ascending=False)
            .reset_index(drop=True)
        )
        full_df.index += 1
        st.dataframe(full_df, use_container_width=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────


def _build_sidebar(data: dict) -> None:
    with st.sidebar:
        st.header("Model Info")

        config_60m = data["60m"]["config"]
        if config_60m:
            trained_at = config_60m.get("trained_at", "unknown")
            st.metric("Trained at", trained_at[:10])

            inf = config_60m.get("inference", {})
            data_sec = config_60m.get("data", {})
            st.caption(
                f"Horizon: {inf.get('horizon_minutes', '?')} min  \n"
                f"Lookback: {inf.get('lookback_minutes', '?')} min  \n"
                f"Machines: {data_sec.get('training_machines', '?'):,}  \n"
                f"Training rows: {data_sec.get('training_rows', '?'):,}"
            )

            env_section = config_60m.get("environment", {})
            st.caption(
                f"XGBoost {env_section.get('xgboost_version', '?')}  \n"
                f"Python {env_section.get('python_version', '?')}"
            )
        else:
            st.warning("60m model config not found.")

        st.divider()
        st.caption(
            f"Artifacts dir:  \n`{data['artifacts_dir']}`  \n\n"
            "Override:  \n"
            "```\nARTIFACTS_DIR=path \\\n  streamlit run demo.py\n```"
        )


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(
        page_title="Spike Predictor — Dashboard",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    artifacts_dir = _resolve_artifacts_dir()

    # Fail fast with a clear message if the 60m model is missing
    required_config = artifacts_dir / _MODEL_SUBDIRS["60m"] / "spike_config.json"
    if not required_config.exists():
        st.error(
            f"Required 60m model not found at `{required_config}`.  \n"
            "Run the full training pipeline first, then relaunch the dashboard."
        )
        st.stop()

    data = _load_all(artifacts_dir)
    _build_sidebar(data)

    st.title("⚡ CPU Spike Predictor — Model Dashboard")
    st.caption(
        "Offline dashboard — reads training artifacts from disk. No live inference required."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🎚 Threshold Explorer",
        "🎯 Calibration",
        "🔍 Feature Attribution",
    ])

    with tab1:
        _render_overview(data)
    with tab2:
        _render_threshold_explorer(data)
    with tab3:
        _render_calibration(data)
    with tab4:
        _render_feature_attribution(data)


if __name__ == "__main__":
    main()
