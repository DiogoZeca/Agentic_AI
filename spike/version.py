"""Model artefact versioning utilities.

Reads existing artefact files and produces a deterministic version string +
metadata dict.  No files are written — this is a pure read-only utility.

Version string format
---------------------
"<compact_timestamp>_<feature_hash>"

  compact_timestamp   compact ISO 8601 from spike_config.json → trained_at
                      (e.g. "20260429T143000Z").  Falls back to "unknown"
                      when trained_at is absent or unparseable.

  feature_hash        first 6 hex chars of SHA-256 of "|".join(sorted(feature_cols)).
                      Encodes which feature set was used without listing all columns.
                      Same features → same hash, regardless of weights or thresholds.

Example: "20260429T143000Z_7e4d2f"

The version string is deterministic: identical artefacts always produce the same
string, making it safe to use as a deployment tag or SLO metric label.

Files read (all optional — missing fields degrade gracefully)
------------------------------------------------------------
  model_dir/spike_config.json        → trained_at
  model_dir/spike_model.meta.json    → feature_cols, xgboost_version
  domain_dir/bootstrap_meta.json     → bootstrapped_at (domain threshold step)
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def _compact_timestamp(iso_str: str) -> str:
    """Convert an ISO 8601 string to a compact UTC form.

    "2026-04-29T14:30:00+00:00" → "20260429T143000Z"

    Falls back to a sanitised substring when parsing fails so the version
    string is always a safe filesystem/label-friendly value.
    """
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    except (ValueError, TypeError):
        # Unparseable — strip special chars and truncate so it stays usable.
        safe = iso_str[:20].replace(":", "").replace("-", "").replace(" ", "")
        return safe or "unknown"


def _feature_hash(feature_cols: list[str]) -> str:
    """Short deterministic hash of the feature column list.

    SHA-256 of "|".join(sorted(feature_cols)) → first 6 hex chars.
    Sorting makes the hash independent of column ordering in the meta file,
    so two artefacts trained on the same features always match.
    """
    key = "|".join(sorted(feature_cols))
    return hashlib.sha256(key.encode()).hexdigest()[:6]


def get_model_version(
    model_dir: Path,
    *,
    domain_dir: Optional[Path] = None,
) -> dict:
    """Read artefact files and return a version metadata dict.

    All file reads are defensive: missing files and missing JSON keys produce
    None values or graceful fallbacks, never exceptions.  The daemon and
    inference script call this once at startup; the result is stored in
    _Artifacts.version_info and embedded in every predictions.json and
    slo_metrics.json write.

    Parameters
    ----------
    model_dir  : directory containing spike_config.json and spike_model.meta.json.
    domain_dir : optional directory produced by bootstrap_thresholds.py.
                 When provided, bootstrap_meta.json is read for domain version info.

    Returns
    -------
    dict with keys:
      model_version          str        — "<compact_ts>_<feature_hash>" (always set)
      trained_at             str | None — raw ISO-8601 from spike_config.json
      feature_hash           str | None — 6-char hex from spike_model.meta.json
      xgboost_version        str | None — XGBoost version string from meta.json
      domain_bootstrapped_at str | None — timestamp from domain_dir/bootstrap_meta.json
    """
    trained_at:      Optional[str]  = None
    feature_cols:    Optional[list] = None
    xgboost_version: Optional[str]  = None
    domain_ts:       Optional[str]  = None

    # ── spike_config.json → trained_at ───────────────────────────────────────
    config_path = model_dir / "spike_config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            trained_at = cfg.get("trained_at")
        except Exception as exc:
            log.debug("version: could not read spike_config.json: %s", exc)

    # ── spike_model.meta.json → feature_cols, xgboost_version ───────────────
    meta_path = model_dir / "spike_model.meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            feature_cols    = meta.get("feature_cols")
            xgboost_version = meta.get("xgboost_version")
        except Exception as exc:
            log.debug("version: could not read spike_model.meta.json: %s", exc)

    # ── domain_dir/bootstrap_meta.json → bootstrapped_at ─────────────────────
    if domain_dir is not None:
        bm_path = domain_dir / "bootstrap_meta.json"
        if bm_path.exists():
            try:
                bm = json.loads(bm_path.read_text())
                domain_ts = bm.get("bootstrapped_at")
            except Exception as exc:
                log.debug("version: could not read bootstrap_meta.json: %s", exc)

    # ── Compose version string ────────────────────────────────────────────────
    # Timestamp part: compact trained_at or "unknown" sentinel.
    ts_part = _compact_timestamp(trained_at) if trained_at else "unknown"
    # Hash part: 6-char feature fingerprint; "000000" when meta.json is absent.
    h_part  = _feature_hash(feature_cols) if feature_cols else "000000"

    return {
        "model_version":          f"{ts_part}_{h_part}",
        "trained_at":             trained_at,
        "feature_hash":           h_part if feature_cols else None,
        "xgboost_version":        xgboost_version,
        "domain_bootstrapped_at": domain_ts,
    }
