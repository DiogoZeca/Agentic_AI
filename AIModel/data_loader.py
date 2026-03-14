"""
Data Loader — ingestion layer for web service energy & carbon metrics.

Requires 'ds' (datetime) and 'consumption' columns; all others are optional.
Auto-detects available metrics and regressors; pipeline degrades gracefully on
missing columns. See CLAUDE.md for the full 15-column schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


# ── Column definitions ────────────────────────────────────────────────────────

REQUIRED_COLUMNS: list[str] = ["ds", "consumption"]

# Metrics the models fit directly (SCI is always derived from components)
CANDIDATE_FIT_METRICS: list[str] = [
    "consumption",
    "carbonEmissions",
    "carbonIntensityFactor",
    "greenConsumptionPercentage",
    "cpuUtilization",
    "functionalUnit",
    "cost",
]

# Possible regressors per metric — only used if column exists in data
POSSIBLE_REGRESSORS: dict[str, list[str]] = {
    "consumption":                ["functionalUnit"],
    "carbonEmissions":            ["carbonIntensityFactor"],
    "carbonIntensityFactor":      [],
    "greenConsumptionPercentage": [],
    "cpuUtilization":             [],
    "functionalUnit":             [],
    "cost":                       [],
}

# Columns needed for each optional chart
CHART_COLUMN_DEPS: dict[str, list[str]] = {
    "service_profile":   ["consumption", "functionalUnit"],
    "decomposition":     ["consumption"],
    "sci_by_hour":       ["softwareCarbonIntensity"],
    "intensity_vs_green": ["carbonIntensityFactor", "greenConsumptionPercentage"],
}


# ── Pipeline config dataclass ─────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Holds auto-detected column availability for analysis scripts."""
    metrics: list[str]
    fit_metrics: list[str]
    regressor_map: dict[str, list[str]]
    has_sci: bool                          # can SCI be derived?
    available_charts: dict[str, bool]      # which optional charts can be rendered
    missing_recommended: list[str]         # columns that were absent


# ── Core functions ────────────────────────────────────────────────────────────

def load_and_validate(path: str) -> pd.DataFrame:
    """Load CSV and enforce minimum schema.

    Raises:
        FileNotFoundError: if the CSV file does not exist at ``path``.
        ValueError: if required columns (``ds``, ``consumption``) are absent.

    Prints a schema report showing what was found.
    """
    try:
        df = pd.read_csv(path, parse_dates=["ds"])
    except FileNotFoundError:
        raise FileNotFoundError(
            f"\n[data_loader] File not found: {path}\n"
            "  Generate synthetic data first:  python data_generator.py\n"
            "  Or point DATA_PATH to your CSV.\n"
        )

    # Check required columns
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"\n[data_loader] Missing required columns: {missing_required}\n"
            f"  Found columns: {list(df.columns)}\n"
            f"  Required: {REQUIRED_COLUMNS}\n"
        )

    # Ensure ds is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["ds"]):
        df["ds"] = pd.to_datetime(df["ds"])

    # Drop fully-NaN rows on key columns
    before = len(df)
    df = df.dropna(subset=["ds", "consumption"])
    dropped = before - len(df)
    if dropped:
        print(f"[data_loader] Dropped {dropped} rows with NaN in required columns.")

    _print_schema_report(df)
    return df


def build_pipeline_config(df: pd.DataFrame) -> PipelineConfig:
    """Build the full pipeline configuration from available columns."""
    cols = set(df.columns)

    # Which fit metrics are available
    fit_metrics = [m for m in CANDIDATE_FIT_METRICS if m in cols]

    # SCI can be derived only if both components exist
    has_sci = ("carbonEmissions" in cols) and ("functionalUnit" in cols)

    # Final display metrics (SCI added if derivable)
    metrics = [m for m in ["consumption", "carbonEmissions"] if m in cols]
    if has_sci:
        metrics.append("softwareCarbonIntensity")

    # Regressor map — only include regressors that exist in data
    regressor_map = {
        metric: [r for r in regs if r in cols]
        for metric, regs in POSSIBLE_REGRESSORS.items()
        if metric in fit_metrics
    }

    # Which optional charts can be rendered
    available_charts = {
        name: all(c in cols for c in deps)
        for name, deps in CHART_COLUMN_DEPS.items()
    }

    # Recommended columns that are absent
    recommended = [
        "carbonEmissions", "functionalUnit", "carbonIntensityFactor",
        "softwareCarbonIntensity",
    ]
    missing_recommended = [c for c in recommended if c not in cols]

    return PipelineConfig(
        metrics=metrics,
        fit_metrics=fit_metrics,
        regressor_map=regressor_map,
        has_sci=has_sci,
        available_charts=available_charts,
        missing_recommended=missing_recommended,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _print_schema_report(df: pd.DataFrame) -> None:
    """Print a clear report of what columns were found and what's missing."""
    cols = set(df.columns)
    all_known = [
        # Required
        "ds", "consumption",
        # Fit metrics
        "carbonEmissions", "functionalUnit",
        # Derived
        "softwareCarbonIntensity",
        # Supporting columns
        "timeWindow", "totalConsumption", "measurementSource",
        "cost", "totalCost",
        "operationalEmissions", "embodiedEmissions",
        "carbonIntensityFactor", "greenConsumptionPercentage",
        "cpuUtilization",
    ]

    found   = [c for c in all_known if c in cols]
    missing = [c for c in all_known if c not in cols]
    extra   = [c for c in cols if c not in all_known]

    print(f"\n[data_loader] Schema report — {len(df)} rows")
    print(f"  Date range:  {df['ds'].min().date()} → {df['ds'].max().date()}")
    print(f"  Found ({len(found)}):   {', '.join(found)}")
    if missing:
        print(f"  Missing ({len(missing)}): {', '.join(missing)}")
    if extra:
        print(f"  Extra ({len(extra)}):  {', '.join(extra)}")
