"""Tests for spike/version.py — deterministic model artefact versioning."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike.version import _compact_timestamp, _feature_hash, get_model_version


# ── Minimal artefact writer ───────────────────────────────────────────────────


def _write_model_dir(
    tmp_path:     Path,
    trained_at:   str | None = "2026-01-01T00:00:00+00:00",
    feature_cols: list | None = None,
    xgb_version:  str        = "2.1.0",
) -> Path:
    """Write just the JSON files that get_model_version reads (no XGBoost weights)."""
    model_dir = tmp_path / "models" / "spike"
    model_dir.mkdir(parents=True, exist_ok=True)

    cfg: dict = {}
    if trained_at is not None:
        cfg["trained_at"] = trained_at
    (model_dir / "spike_config.json").write_text(json.dumps(cfg))

    if feature_cols is not None:
        (model_dir / "spike_model.meta.json").write_text(json.dumps({
            "feature_cols":    feature_cols,
            "xgboost_version": xgb_version,
        }))

    return model_dir


# ── _compact_timestamp ────────────────────────────────────────────────────────


class TestCompactTimestamp:
    def test_standard_utc_offset(self):
        assert _compact_timestamp("2026-04-29T14:30:00+00:00") == "20260429T143000Z"

    def test_non_utc_offset_normalised_to_utc(self):
        # UTC+1 14:30 → UTC 13:30
        result = _compact_timestamp("2026-04-29T14:30:00+01:00")
        assert result == "20260429T133000Z"

    def test_garbage_returns_safe_string(self):
        result = _compact_timestamp("not-a-date")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_string_returns_safe_string(self):
        result = _compact_timestamp("")
        assert isinstance(result, str)

    def test_output_has_no_colons_or_spaces(self):
        """Version strings must be filesystem/label-safe."""
        result = _compact_timestamp("2026-01-01T00:00:00+00:00")
        assert ":" not in result
        assert " " not in result


# ── _feature_hash ─────────────────────────────────────────────────────────────


class TestFeatureHash:
    def test_exactly_6_hex_chars(self):
        h = _feature_hash(["a", "b", "c"])
        assert len(h) == 6
        assert all(c in "0123456789abcdef" for c in h)

    def test_order_independent(self):
        """Column order must not affect the hash."""
        assert _feature_hash(["a", "b", "c"]) == _feature_hash(["c", "a", "b"])

    def test_different_columns_differ(self):
        assert _feature_hash(["total_cpu"]) != _feature_hash(["peak_cpu"])

    def test_deterministic_across_calls(self):
        cols = ["total_cpu", "peak_cpu", "disk_io"]
        assert _feature_hash(cols) == _feature_hash(cols)


# ── get_model_version ─────────────────────────────────────────────────────────


class TestGetModelVersion:
    def test_returns_all_expected_keys(self, tmp_path):
        model_dir = _write_model_dir(tmp_path, feature_cols=["total_cpu", "peak_cpu"])
        result    = get_model_version(model_dir)

        for key in ("model_version", "trained_at", "feature_hash", "xgboost_version",
                    "domain_bootstrapped_at"):
            assert key in result, f"missing key: {key}"

    def test_version_string_format(self, tmp_path):
        """model_version must be '<compact_ts>_<6-char-hex>'."""
        model_dir = _write_model_dir(tmp_path, feature_cols=["total_cpu"])
        version   = get_model_version(model_dir)["model_version"]

        parts = version.split("_")
        assert len(parts) == 2, f"expected exactly one '_', got: {version!r}"
        _, h_part = parts
        assert len(h_part) == 6, f"feature hash must be 6 chars, got: {h_part!r}"
        assert all(c in "0123456789abcdef" for c in h_part)

    def test_trained_at_populated(self, tmp_path):
        model_dir = _write_model_dir(tmp_path, feature_cols=["total_cpu"])
        result    = get_model_version(model_dir)
        assert result["trained_at"] == "2026-01-01T00:00:00+00:00"

    def test_missing_trained_at_uses_unknown_prefix(self, tmp_path):
        """spike_config.json exists but has no trained_at → 'unknown_<hash>'."""
        model_dir = _write_model_dir(tmp_path, trained_at=None, feature_cols=["total_cpu"])
        version   = get_model_version(model_dir)["model_version"]
        assert version.startswith("unknown_")

    def test_missing_meta_json_uses_sentinel_hash(self, tmp_path):
        """No spike_model.meta.json → feature_hash is None, version ends in '_000000'."""
        model_dir = _write_model_dir(tmp_path)  # no feature_cols → no meta.json
        result    = get_model_version(model_dir)

        assert result["feature_hash"] is None
        assert result["model_version"].endswith("_000000")

    def test_xgboost_version_populated(self, tmp_path):
        model_dir = _write_model_dir(tmp_path, feature_cols=["total_cpu"], xgb_version="3.0.0")
        assert get_model_version(model_dir)["xgboost_version"] == "3.0.0"

    def test_domain_dir_adds_bootstrapped_at(self, tmp_path):
        model_dir  = _write_model_dir(tmp_path, feature_cols=["total_cpu"])
        domain_dir = tmp_path / "domain"
        domain_dir.mkdir()
        (domain_dir / "bootstrap_meta.json").write_text(json.dumps({
            "bootstrapped_at": "2026-04-28T10:00:00+00:00",
        }))

        result = get_model_version(model_dir, domain_dir=domain_dir)
        assert result["domain_bootstrapped_at"] == "2026-04-28T10:00:00+00:00"

    def test_domain_dir_without_meta_json_is_safe(self, tmp_path):
        """domain_dir exists but has no bootstrap_meta.json — must not raise."""
        model_dir  = _write_model_dir(tmp_path, feature_cols=["total_cpu"])
        domain_dir = tmp_path / "domain"
        domain_dir.mkdir()

        result = get_model_version(model_dir, domain_dir=domain_dir)
        assert result["domain_bootstrapped_at"] is None

    def test_missing_config_json_is_safe(self, tmp_path):
        """Neither file exists → no crash, model_version still returned."""
        model_dir = tmp_path / "models" / "spike"
        model_dir.mkdir(parents=True)

        result = get_model_version(model_dir)
        assert "model_version" in result
        assert isinstance(result["model_version"], str)

    def test_deterministic_same_input(self, tmp_path):
        """Calling twice with identical artefacts must return the same version string."""
        model_dir = _write_model_dir(tmp_path, feature_cols=["total_cpu", "peak_cpu"])
        v1 = get_model_version(model_dir)["model_version"]
        v2 = get_model_version(model_dir)["model_version"]
        assert v1 == v2

    def test_no_run_config_does_not_crash(self, tmp_path):
        """run_config.json is absent in test fixtures — version must still work."""
        # _write_model_dir never writes run_config.json; this test confirms
        # get_model_version does not depend on it.
        model_dir = _write_model_dir(tmp_path, feature_cols=["a", "b"])
        result    = get_model_version(model_dir)
        assert result["model_version"] is not None
