"""Train CPU power models, compare with cross-validation, save the best artefact.

Entry point for the ML training pipeline.  Reads cpu_data.dat, trains both an
XGBoost and a PyTorch MLP regressor, evaluates with cross-validation, and
writes the winning model artefacts to an output directory.

After running, commit the output directory to git so the API image can COPY it:

    python train.py                           # full run with LOCO CV
    python train.py --fast                    # skip LOCO CV (~3× faster)
    python train.py --data data/cpu_data.dat --out models/

Option A deployment workflow
-----------------------------
1. python train.py            → produces models/
2. Review the comparison table
3. git add models/ && git commit -m "chore: retrain power models"
4. docker compose up --build  → image COPYs models/ in, API loads at startup
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import xgboost as xgb

from data_loader import load_cpu_power_data
from feature_engineering import build_features, save_metadata
from model_trainer import MLPPowerTrainer, XGBoostPowerTrainer, cv_interpolation, cv_loco

# ── Defaults ───────────────────────────────────────────────────────────────────

_DEFAULT_DATA = "data/cpu_data.dat"
_DEFAULT_OUT  = "models"

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# Suppress verbose output from training libraries — their noise drowns the table
logging.getLogger("xgboost").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost and MLP power models; save the winner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python train.py\n"
            "  python train.py --fast\n"
            "  python train.py --data data/cpu_data.dat --out models/\n"
        ),
    )
    parser.add_argument(
        "--data",
        default=_DEFAULT_DATA,
        metavar="PATH",
        help=f"Path to cpu_data.dat  (default: {_DEFAULT_DATA})",
    )
    parser.add_argument(
        "--out",
        default=_DEFAULT_OUT,
        metavar="DIR",
        help=f"Output directory for artefacts  (default: {_DEFAULT_OUT})",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip LOCO cross-validation (use interpolation CV only)",
    )

    args = parser.parse_args()

    # Fail-fast validation before any slow work starts
    if not Path(args.data).exists():
        parser.error(f"data file not found: {args.data}")

    return args


# ── Helpers ────────────────────────────────────────────────────────────────────

def _divider(char: str = "─", width: int = 66) -> None:
    log.info(char * width)


def _file_md5(path: str) -> str:
    """Return the MD5 hex digest of a file (for manifest reproducibility tracking)."""
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(data: dict, dest: Path) -> None:
    """Write a JSON file atomically: temp file on same filesystem → os.replace().

    Prevents a partial manifest from being visible to the API if the process
    is interrupted between the open() and close() of a direct write.
    """
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _library_versions() -> dict[str, str]:
    return {
        "xgboost": xgb.__version__,
        "torch":   torch.__version__,
    }


# ── Training + evaluation ──────────────────────────────────────────────────────

def _run(
    name: str,
    trainer_cls: type,
    X,
    y_reg,
    weights,
    metadata: dict,
    fast: bool,
) -> dict:
    """Train one model on the full dataset, run CV, return results dict."""
    log.info("Training %s ...", name)
    t0 = time.perf_counter()
    trainer = trainer_cls()
    trainer.fit(X, y_reg, weights)
    log.info("  ✓ trained  (%.1fs)", time.perf_counter() - t0)

    log.info("Evaluating %s — interpolation CV (5-fold) ...", name)
    t0 = time.perf_counter()
    interp = cv_interpolation(trainer_cls, X, y_reg, weights, metadata)
    log.info("  ✓ done  (%.1fs)", time.perf_counter() - t0)

    loco: dict | None = None
    if not fast:
        log.info("Evaluating %s — LOCO CV (11 folds) ...", name)
        t0 = time.perf_counter()
        loco = cv_loco(trainer_cls, X, y_reg, weights, metadata)
        log.info("  ✓ done  (%.1fs)", time.perf_counter() - t0)

    return {"name": name, "trainer": trainer, "interp": interp, "loco": loco}


# ── Comparison table ───────────────────────────────────────────────────────────

def _print_table(xgb_r: dict, mlp_r: dict, winner_name: str, fast: bool) -> None:
    """Log a formatted comparison table to stdout."""
    _divider()
    log.info("  MODEL COMPARISON")
    _divider()

    W = 12  # column width

    def row(label: str, xval: str, mval: str, flag: str = "") -> None:
        log.info("  %-38s %*s  %*s  %s", label, W, xval, W, mval, flag)

    row("Metric", "XGBoost", "MLP")
    log.info("  " + "─" * 62)

    xi, mi = xgb_r["interp"], mlp_r["interp"]
    xwins = winner_name == "xgboost"

    row(
        "Interpolation RMSE (W)  ← decides winner",
        f"{xi['rmse']:.2f} W",
        f"{mi['rmse']:.2f} W",
        "★" if xwins else "  ★",
    )
    row("Interpolation MAE  (W)", f"{xi['mae']:.2f} W",    f"{mi['mae']:.2f} W")
    row("Interpolation Spike F1", f"{xi['spike_f1']:.3f}", f"{mi['spike_f1']:.3f}")

    if not fast:
        log.info("  " + "─" * 62)
        xf = xgb_r["loco"]["full_types"]["mean"]
        mf = mlp_r["loco"]["full_types"]["mean"]
        xs = xgb_r["loco"]["sparse_types"]["mean"]
        ms = mlp_r["loco"]["sparse_types"]["mean"]

        row("LOCO RMSE  — 8 full types  (W)", f"{xf['rmse']:.2f} W", f"{mf['rmse']:.2f} W")
        row("LOCO MAE   — 8 full types  (W)", f"{xf['mae']:.2f} W",  f"{mf['mae']:.2f} W")
        row("LOCO Spike F1 — 8 full types",   f"{xf['spike_f1']:.3f}", f"{mf['spike_f1']:.3f}")
        row("LOCO RMSE  — 3 sparse types (W)", f"{xs['rmse']:.2f} W", f"{ms['rmse']:.2f} W")

    _divider()
    margin = abs(xi["rmse"] - mi["rmse"])
    log.info(
        "  ★ Winner: %-10s  (margin: %.2f W RMSE on interpolation CV)",
        winner_name.upper(),
        margin,
    )
    _divider()


# ── Artefact persistence ───────────────────────────────────────────────────────

def _save(
    xgb_r: dict,
    mlp_r: dict,
    winner_name: str,
    metadata: dict,
    out_dir: Path,
    data_path: str,
    run_ts: str,
    t_start: float,
    fast: bool,
) -> None:
    """Persist both model artefacts, metadata, and winner manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Saving XGBoost artefact ...")
    xgb_r["trainer"].save(out_dir / "xgb")

    log.info("Saving MLP artefact ...")
    mlp_r["trainer"].save(out_dir / "mlp")

    log.info("Saving feature metadata ...")
    save_metadata(metadata, out_dir / "metadata.json")

    # Winner manifest — read by cpu_power_model.py at API startup
    winner_r = xgb_r if winner_name == "xgboost" else mlp_r
    # Store only the subdirectory name — never a full path.
    # cpu_power_model.py resolves the full path as: parent(winner.json) / dir.
    # This keeps the manifest portable between local dev and Docker.
    winner_dir = "xgb" if winner_name == "xgboost" else "mlp"

    manifest: dict = {
        "model":       winner_name,
        "dir":         winner_dir,
        "run_ts":      run_ts,
        "trained_at":  datetime.now(timezone.utc).isoformat(),
        "duration_s":  round(time.perf_counter() - t_start, 1),
        "data_path":   str(Path(data_path).resolve()),
        "data_md5":    _file_md5(data_path),
        "fast_mode":   fast,
        "metrics": {
            "interpolation_rmse_w":   round(winner_r["interp"]["rmse"],     3),
            "interpolation_mae_w":    round(winner_r["interp"]["mae"],      3),
            "interpolation_spike_f1": round(winner_r["interp"]["spike_f1"], 3),
        },
        "all_models": {
            "xgboost": {
                "interpolation_rmse_w":   round(xgb_r["interp"]["rmse"],     3),
                "interpolation_mae_w":    round(xgb_r["interp"]["mae"],      3),
                "interpolation_spike_f1": round(xgb_r["interp"]["spike_f1"], 3),
            },
            "mlp": {
                "interpolation_rmse_w":   round(mlp_r["interp"]["rmse"],     3),
                "interpolation_mae_w":    round(mlp_r["interp"]["mae"],      3),
                "interpolation_spike_f1": round(mlp_r["interp"]["spike_f1"], 3),
            },
        },
        "library_versions": _library_versions(),
    }

    if not fast:
        for model_name, result in (("xgboost", xgb_r), ("mlp", mlp_r)):
            manifest["all_models"][model_name]["loco"] = {
                "full_mean_rmse_w":    round(result["loco"]["full_types"]["mean"]["rmse"],     3),
                "full_mean_mae_w":     round(result["loco"]["full_types"]["mean"]["mae"],      3),
                "full_mean_spike_f1":  round(result["loco"]["full_types"]["mean"]["spike_f1"], 3),
                "sparse_mean_rmse_w":  round(result["loco"]["sparse_types"]["mean"]["rmse"],   3),
            }

    log.info("Writing winner manifest ...")
    _atomic_write_json(manifest, out_dir / "winner.json")

    log.info("  ✓ All artefacts saved to %s/", out_dir)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args    = _parse_args()
    run_ts  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    t_start = time.perf_counter()
    out_dir = Path(args.out)

    _divider("═")
    log.info("  CPU POWER SPIKE MODEL — TRAINING PIPELINE  [%s]", run_ts)
    _divider("═")
    if args.fast:
        log.info("  Mode: fast  (LOCO cross-validation skipped)")

    # ── Load data + build features ─────────────────────────────────────────
    log.info("Loading data from %s ...", args.data)
    t0 = time.perf_counter()
    try:
        df = load_cpu_power_data(args.data)
    except (FileNotFoundError, ValueError) as exc:
        log.error("Data loading failed: %s", exc)
        sys.exit(1)

    X, y_reg, _, weights, metadata = build_features(df)
    log.info(
        "  ✓ Features ready  (%d rows · %d CPU types · %.1fs)",
        len(X),
        X["CPUTYPE"].nunique(),
        time.perf_counter() - t0,
    )

    # ── Train + evaluate both models ───────────────────────────────────────
    _divider()
    xgb_r = _run("XGBoost", XGBoostPowerTrainer, X, y_reg, weights, metadata, args.fast)
    _divider()
    mlp_r = _run("MLP",     MLPPowerTrainer,      X, y_reg, weights, metadata, args.fast)

    # ── Select winner (lower interpolation RMSE) ───────────────────────────
    winner_name = (
        "xgboost"
        if xgb_r["interp"]["rmse"] <= mlp_r["interp"]["rmse"]
        else "mlp"
    )

    # ── Print comparison table ─────────────────────────────────────────────
    _print_table(xgb_r, mlp_r, winner_name, args.fast)

    # ── Save artefacts ─────────────────────────────────────────────────────
    _divider()
    _save(xgb_r, mlp_r, winner_name, metadata, out_dir, args.data, run_ts, t_start, args.fast)

    _divider("═")
    log.info("  DONE  (total: %.1fs)", time.perf_counter() - t_start)
    log.info("  Artefacts : %s/", out_dir)
    log.info("  Next step : git add %s/ && git commit", out_dir)
    _divider("═")


if __name__ == "__main__":
    main()
