"""Threshold bootstrap utility for new deployment domains.

Given historical cluster_agg data from a target domain, computes per-machine
p95/p99 CPU thresholds and writes spike_thresholds.parquet to --domain-dir.

This is a deployment-time tool — not a training tool.  No model retraining is
needed; only threshold percentiles are re-estimated for the new domain.

Why this is necessary
---------------------
The spike inference pipeline normalises load signals with per-machine p95/p99
CPU thresholds (cpu_vs_p95, band_position, etc.).  Thresholds baked in during
training reflect the source domain's load distribution.  When the model is
deployed on a different cluster, those thresholds are mismatched — bootstrapping
domain-specific thresholds from observed history corrects this without retraining.

Minimum history recommendation
-------------------------------
At least 2 weeks (2016 five-minute buckets) per machine for stable p99
estimation — consistent with the AWS CloudWatch anomaly detection warm-up
guideline and with academic sample-size recommendations for the 99th percentile
(210+ observations for 95% CI).  Machines below --min-buckets receive the
global-domain fallback (overall p95/p99 across all machines in the window).

Outputs written to --domain-dir
---------------------------------
  spike_thresholds.parquet  — one row per machine:
                              [machine_id, threshold_p95, threshold_p99]
  bootstrap_meta.json       — audit record: source, n_machines, bucket range,
                              quantile values, fallback statistics

Usage
-----
    python -m spike.bootstrap_thresholds \\
        --input   /path/to/history/cluster_agg.csv \\
        --domain-dir /path/to/domain/config

    python -m spike.bootstrap_thresholds \\
        --input   cluster_agg.parquet \\
        --domain-dir /path/to/domain/config \\
        --min-buckets 2016

Exit codes
----------
    0  thresholds written successfully
    1  fatal error (missing input, wrong schema, unreadable file)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(message)s",
    datefmt = "%H:%M:%S",
    stream  = sys.stderr,
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_QUANTILE_P95: float = 0.95
_QUANTILE_P99: float = 0.99

# Minimum gap between p99 and p95 — mirrors the invariant enforced during
# training (feature_engineer.py) so inference features are consistent.
_MIN_GAP_FACTOR: float = 1.10

# Default minimum buckets a machine must contribute for per-machine percentiles.
# Below this, the machine receives global-domain fallback values.
# 2016 = 2 weeks × 144 five-min buckets/day (industry warm-up guideline).
_DEFAULT_MIN_BUCKETS: int = 2016

_REQUIRED_COLS: tuple[str, ...] = ("machine_id", "bucket", "total_cpu")
_THRESHOLD_COLS: list[str]      = ["machine_id", "threshold_p95", "threshold_p99"]


# ── Input loading ─────────────────────────────────────────────────────────────


def _load_cluster_agg(input_path: Path) -> pd.DataFrame:
    """Load cluster_agg data from CSV or Parquet.

    Raises
    ------
    SystemExit(1)  on any read error or missing required columns.
    """
    suffix = input_path.suffix.lower()
    try:
        if suffix == ".parquet":
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path)
    except Exception as exc:
        log.error("Could not read input file %s: %s", input_path, exc)
        sys.exit(1)

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        log.error(
            "Input is missing required column(s): %s — got %s",
            missing, list(df.columns),
        )
        sys.exit(1)

    if df.empty:
        log.error("Input file is empty — nothing to compute thresholds from.")
        sys.exit(1)

    return df


# ── Threshold computation ─────────────────────────────────────────────────────


def _compute_bootstrap_thresholds(
    df: pd.DataFrame,
    min_buckets_per_machine: int,
) -> tuple[pd.DataFrame, dict]:
    """Compute per-machine p95/p99 from all rows in *df*.

    Machines with fewer than *min_buckets_per_machine* distinct bucket indices
    are excluded from per-machine estimation and receive the global-domain
    fallback (overall p95/p99 across all machines in the input window).

    Returns
    -------
    (thresh_df, stats)
        thresh_df : DataFrame with columns [machine_id, threshold_p95, threshold_p99]
        stats     : dict of scalar audit values for bootstrap_meta.json
    """
    global_p95 = float(df["total_cpu"].quantile(_QUANTILE_P95))
    global_p99 = float(df["total_cpu"].quantile(_QUANTILE_P99))

    bucket_counts = df.groupby("machine_id")["bucket"].nunique()
    rich_mask  = bucket_counts >= min_buckets_per_machine
    rich_ids   = set(bucket_counts[rich_mask].index)
    sparse_ids = set(bucket_counts[~rich_mask].index)

    n_sparse = len(sparse_ids)
    if n_sparse:
        log.warning(
            "  %d machine(s) have fewer than %d buckets — "
            "assigning global fallback (p95=%.4f, p99=%.4f)",
            n_sparse, min_buckets_per_machine, global_p95, global_p99,
        )

    if rich_ids:
        df_rich    = df[df["machine_id"].isin(rich_ids)]
        p95_series = df_rich.groupby("machine_id")["total_cpu"].quantile(_QUANTILE_P95)
        p99_series = df_rich.groupby("machine_id")["total_cpu"].quantile(_QUANTILE_P99)
    else:
        p95_series = pd.Series(dtype="float64", name="total_cpu")
        p99_series = pd.Series(dtype="float64", name="total_cpu")

    if sparse_ids:
        sparse_idx = pd.Index(sorted(sparse_ids), name="machine_id")
        p95_series = pd.concat(
            [p95_series, pd.Series(global_p95, index=sparse_idx, dtype="float64")]
        ).sort_index()
        p99_series = pd.concat(
            [p99_series, pd.Series(global_p99, index=sparse_idx, dtype="float64")]
        ).sort_index()

    # Guard: p99 must be strictly above p95 (degenerate on nearly-always-idle machines).
    degenerate = p99_series <= p95_series
    if degenerate.any():
        n_deg = int(degenerate.sum())
        log.warning(
            "  %d machine(s) have p99 <= p95 — setting p99 = p95 × %.2f",
            n_deg, _MIN_GAP_FACTOR,
        )
        p99_series = p99_series.where(~degenerate, p95_series * _MIN_GAP_FACTOR)

    thresh_df = pd.DataFrame({
        "machine_id":    p95_series.index.astype("int64"),
        "threshold_p95": p95_series.values.astype("float64"),
        "threshold_p99": p99_series.values.astype("float64"),
    })

    # Enforce minimum 10 % gap — mirrors feature_engineer.py invariant.
    gap_mask   = thresh_df["threshold_p99"] < thresh_df["threshold_p95"] * _MIN_GAP_FACTOR
    n_adjusted = int(gap_mask.sum())
    if n_adjusted:
        thresh_df.loc[gap_mask, "threshold_p99"] = (
            thresh_df.loc[gap_mask, "threshold_p95"] * _MIN_GAP_FACTOR
        )
        log.info(
            "  Enforced min %.0f%% p99/p95 gap for %d machine(s)",
            (_MIN_GAP_FACTOR - 1) * 100,
            n_adjusted,
        )

    stats = {
        "n_machines_total":                len(bucket_counts),
        "n_machines_with_per_machine":     len(rich_ids),
        "n_machines_with_global_fallback": n_sparse,
        "global_p95":                      global_p95,
        "global_p99":                      global_p99,
    }
    return thresh_df, stats


# ── Main entry point ──────────────────────────────────────────────────────────


def run(
    input_path:              Path,
    domain_dir:              Path,
    min_buckets_per_machine: int = _DEFAULT_MIN_BUCKETS,
) -> None:
    """Bootstrap domain thresholds from *input_path* and write to *domain_dir*.

    Parameters
    ----------
    input_path              : cluster_agg CSV or Parquet (historical window).
    domain_dir              : output directory for spike_thresholds.parquet and
                              bootstrap_meta.json.
    min_buckets_per_machine : machines below this distinct-bucket count receive
                              global fallback instead of per-machine percentiles.
    """
    log.info("═" * 62)
    log.info("  THRESHOLD BOOTSTRAP")
    log.info("  Input       : %s", input_path.resolve())
    log.info("  Domain dir  : %s", domain_dir.resolve())
    log.info("  Min buckets : %d (%.1f days)", min_buckets_per_machine, min_buckets_per_machine / 288)
    log.info("═" * 62)

    df = _load_cluster_agg(input_path)

    n_machines   = df["machine_id"].nunique()
    n_buckets    = len(df)
    bucket_min   = int(df["bucket"].min())
    bucket_max   = int(df["bucket"].max())
    bucket_range = bucket_max - bucket_min + 1

    log.info(
        "  Loaded %d rows | %d machines | %d total buckets | range [%d, %d]",
        n_buckets, n_machines, bucket_range, bucket_min, bucket_max,
    )

    if bucket_range < min_buckets_per_machine:
        log.warning(
            "  Input spans only %d buckets (%.1f days) — "
            "fewer than --min-buckets %d. "
            "All machines will receive global fallback thresholds.",
            bucket_range, bucket_range / 288, min_buckets_per_machine,
        )

    thresh_df, stats = _compute_bootstrap_thresholds(df, min_buckets_per_machine)

    domain_dir.mkdir(parents=True, exist_ok=True)
    thresholds_path = domain_dir / "spike_thresholds.parquet"
    thresh_df.to_parquet(thresholds_path, index=False)

    meta = {
        "bootstrapped_at":    datetime.now(timezone.utc).isoformat(),
        "source_path":        str(input_path.resolve()),
        "min_buckets_per_machine": min_buckets_per_machine,
        "quantile_p95":       _QUANTILE_P95,
        "quantile_p99":       _QUANTILE_P99,
        "n_rows_input":       n_buckets,
        "bucket_range":       [bucket_min, bucket_max],
        **stats,
    }
    meta_path = domain_dir / "bootstrap_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    log.info("  Wrote %s (%d machines)", thresholds_path.name, len(thresh_df))
    log.info("  Wrote %s", meta_path.name)
    log.info(
        "  Global fallback: p95=%.4f  p99=%.4f  "
        "(%d/%d machines used per-machine percentiles)",
        stats["global_p95"],
        stats["global_p99"],
        stats["n_machines_with_per_machine"],
        stats["n_machines_total"],
    )
    log.info("  Done. Pass --domain-dir %s to daemon.py or predict.py.", domain_dir)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "bootstrap_thresholds.py",
        description = (
            "Bootstrap per-machine p95/p99 CPU thresholds for a new deployment domain.\n\n"
            "Reads historical cluster_agg data (CSV or Parquet), computes per-machine\n"
            "percentiles, and writes spike_thresholds.parquet + bootstrap_meta.json to\n"
            "--domain-dir.  Pass that directory to daemon.py via --domain-dir."
        ),
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", "-i",
        required = True,
        type     = Path,
        metavar  = "FILE",
        help     = "Historical cluster_agg CSV or Parquet (at least 2 weeks recommended).",
    )
    p.add_argument(
        "--domain-dir", "-d",
        required = True,
        dest     = "domain_dir",
        type     = Path,
        metavar  = "DIR",
        help     = "Output directory for spike_thresholds.parquet and bootstrap_meta.json.",
    )
    p.add_argument(
        "--min-buckets",
        default = _DEFAULT_MIN_BUCKETS,
        type    = int,
        metavar = "N",
        dest    = "min_buckets",
        help    = (
            f"Minimum distinct bucket count per machine for per-machine percentiles "
            f"(default: {_DEFAULT_MIN_BUCKETS} = 2 weeks). "
            "Machines below this use global-domain fallback."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if not args.input.exists():
        log.error("Input not found: %s", args.input)
        sys.exit(1)

    if args.min_buckets < 1:
        log.error("--min-buckets must be >= 1 (got %d).", args.min_buckets)
        sys.exit(1)

    run(
        input_path              = args.input,
        domain_dir              = args.domain_dir,
        min_buckets_per_machine = args.min_buckets,
    )


if __name__ == "__main__":
    main()
