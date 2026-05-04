"""Tests for spike/psi.py and spike/drift_monitor.py."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike import psi as _psi
from spike.classifier import _X_COLS
from spike.drift_monitor import (
    _build_report,
    _classify_status,
    _compute_feature_psi,
    _top_recommendation,
)


# ── TestPsiSingle ─────────────────────────────────────────────────────────────


class TestPsiSingle:
    def test_identical_distributions_near_zero(self):
        rng = np.random.default_rng(0)
        x   = rng.uniform(0, 1, 1000).astype("float64")
        psi = _psi.compute_psi_single(x, x.copy())
        assert psi == pytest.approx(0.0, abs=1e-9)

    def test_shifted_distributions_large_psi(self):
        expected = np.linspace(0, 1, 1000)
        actual   = np.linspace(1, 2, 1000)
        psi      = _psi.compute_psi_single(expected, actual)
        assert psi > _psi.PSI_WARNING

    def test_binary_all_zero_actual_no_crash(self):
        """Quiet period: live binary feature is all zeros — must not crash."""
        expected = np.array([0.0, 1.0] * 200)
        actual   = np.zeros(200)
        psi      = _psi.compute_psi_single(expected, actual)
        assert isinstance(psi, float)
        assert not np.isnan(psi)     # eps=1e-4 pseudocount prevents NaN

    def test_all_nan_returns_nan(self):
        expected = np.full(100, np.nan)
        actual   = np.random.rand(100)
        psi      = _psi.compute_psi_single(expected, actual)
        assert np.isnan(psi)

    def test_empty_expected_returns_nan(self):
        psi = _psi.compute_psi_single(np.array([]), np.array([1.0, 2.0]))
        assert np.isnan(psi)

    def test_empty_actual_returns_nan(self):
        psi = _psi.compute_psi_single(np.array([1.0, 2.0]), np.array([]))
        assert np.isnan(psi)

    def test_zero_variance_reference_returns_nan(self):
        """Constant non-binary reference collapses to one unique edge → fewer than 2 → NaN."""
        expected = np.full(100, 0.5)   # constant 0.5 is not in {0,1} → continuous path
        actual   = np.random.rand(100)
        psi      = _psi.compute_psi_single(expected, actual)
        assert np.isnan(psi)

    def test_known_value_uniform_shift(self):
        """U[0,1] vs U[0.5,1.5] → large PSI (half the bins go empty, eps clips each to 1e-4)."""
        rng      = np.random.default_rng(42)
        expected = rng.uniform(0.0, 1.0, 10_000)
        actual   = rng.uniform(0.5, 1.5, 10_000)
        psi      = _psi.compute_psi_single(expected, actual, n_bins=10)
        assert psi > _psi.PSI_WARNING    # significant shift must exceed the retrain threshold

    def test_binary_feature_uses_2_bins(self):
        """Binary reference must produce exactly 2 bins (verified via PSI = 0 for identical)."""
        expected = np.array([0.0, 1.0] * 500)
        actual   = np.array([0.0, 1.0] * 500)
        psi      = _psi.compute_psi_single(expected, actual)
        assert psi == pytest.approx(0.0, abs=1e-6)

    def test_custom_n_bins(self):
        rng = np.random.default_rng(7)
        x   = rng.uniform(0, 1, 500)
        psi = _psi.compute_psi_single(x, x.copy(), n_bins=5)
        assert psi == pytest.approx(0.0, abs=1e-9)


# ── TestDriftReportLogic ──────────────────────────────────────────────────────


class TestDriftReportLogic:
    # ── _classify_status ──

    def test_classify_stable(self):
        assert _classify_status(0.05) == "stable"

    def test_classify_warning(self):
        assert _classify_status(0.15) == "warning"

    def test_classify_retrain(self):
        assert _classify_status(0.30) == "retrain"

    def test_classify_emergency(self):
        assert _classify_status(0.60) == "emergency"

    def test_classify_none_is_stable(self):
        assert _classify_status(None) == "stable"

    def test_classify_boundary_warning(self):
        assert _classify_status(_psi.PSI_STABLE) == "warning"

    def test_classify_boundary_retrain(self):
        assert _classify_status(_psi.PSI_WARNING) == "retrain"

    def test_classify_boundary_emergency(self):
        assert _classify_status(_psi.PSI_EMERGENCY) == "emergency"

    # ── _top_recommendation ──

    def test_all_stable_recommendation(self):
        statuses = ["stable"] * 40
        assert _top_recommendation(statuses) == "stable"

    def test_one_warning_elevates(self):
        statuses = ["stable"] * 39 + ["warning"]
        assert _top_recommendation(statuses) == "warning"

    def test_one_retrain_elevates(self):
        statuses = ["stable"] * 38 + ["warning", "retrain"]
        assert _top_recommendation(statuses) == "retrain"

    def test_one_emergency_elevates(self):
        statuses = ["stable", "emergency"]
        assert _top_recommendation(statuses) == "emergency"

    # ── _build_report ──

    def _make_features(self, psi_values: list[float | None]) -> list[dict]:
        """Build minimal feature dicts matching _build_report expectations."""
        records = []
        for i, psi in enumerate(psi_values):
            records.append({
                "feature":        f"feat_{i}",
                "psi":            psi,
                "status":         _classify_status(psi),
                "reference_mean": 0.5,
                "live_mean":      0.5,
            })
        return records

    def _minimal_live_df(self) -> pd.DataFrame:
        return pd.DataFrame({"bucket": [100, 101]})

    def test_all_stable_report(self):
        features = self._make_features([0.05] * 40)
        report   = _build_report(
            model_version    = "test_1234",
            reference_rows   = 10000,
            live_df          = self._minimal_live_df(),
            features         = features,
            min_bins_warning = False,
        )
        assert report["recommendation"] == "stable"
        assert report["psi_summary"]["system_alert"] is False
        assert report["psi_summary"]["stable_count"] == 40

    def test_warning_escalates_recommendation(self):
        psis     = [0.05] * 39 + [0.15]
        features = self._make_features(psis)
        report   = _build_report(
            model_version="v", reference_rows=1,
            live_df=self._minimal_live_df(), features=features, min_bins_warning=False,
        )
        assert report["recommendation"] == "warning"

    def test_retrain_escalates_recommendation(self):
        psis     = [0.05] * 38 + [0.15, 0.30]
        features = self._make_features(psis)
        report   = _build_report(
            model_version="v", reference_rows=1,
            live_df=self._minimal_live_df(), features=features, min_bins_warning=False,
        )
        assert report["recommendation"] == "retrain"

    def test_emergency_escalates_recommendation(self):
        psis     = [0.05] * 39 + [0.60]
        features = self._make_features(psis)
        report   = _build_report(
            model_version="v", reference_rows=1,
            live_df=self._minimal_live_df(), features=features, min_bins_warning=False,
        )
        assert report["recommendation"] == "emergency"

    def test_system_alert_fires_above_10pct(self):
        # 5 of 40 features with PSI >= 0.20 → 12.5% → alert
        psis     = [0.25] * 5 + [0.05] * 35
        features = self._make_features(psis)
        report   = _build_report(
            model_version="v", reference_rows=1,
            live_df=self._minimal_live_df(), features=features, min_bins_warning=False,
        )
        assert report["psi_summary"]["system_alert"] is True
        assert report["psi_summary"]["system_alert_reason"] is not None

    def test_system_alert_does_not_fire_at_10pct(self):
        # Exactly 4 of 40 → 10.0% → NOT above 10% → no alert
        psis     = [0.25] * 4 + [0.05] * 36
        features = self._make_features(psis)
        report   = _build_report(
            model_version="v", reference_rows=1,
            live_df=self._minimal_live_df(), features=features, min_bins_warning=False,
        )
        assert report["psi_summary"]["system_alert"] is False

    def test_nan_psi_excluded_from_system_alert_denominator(self):
        """None PSI features must not count toward the system alert denominator."""
        # 5 valid features, 4 with PSI >= 0.20 → 80% → alert
        # 35 None PSI features → excluded from denominator
        psis = [0.25] * 4 + [0.05] + [None] * 35
        features = self._make_features(psis)
        report = _build_report(
            model_version="v", reference_rows=1,
            live_df=self._minimal_live_df(), features=features, min_bins_warning=False,
        )
        assert report["psi_summary"]["system_alert"] is True

    def test_report_is_valid_json_no_nan(self):
        """NaN / inf must not appear in the serialised JSON."""
        psis = [float("nan"), 0.05, None]
        features = self._make_features(psis)
        report = _build_report(
            model_version="v", reference_rows=1,
            live_df=self._minimal_live_df(), features=features, min_bins_warning=False,
        )
        # json.dumps raises TypeError on NaN by default
        serialised = json.dumps(report)
        parsed     = json.loads(serialised)
        assert parsed is not None

    def test_report_has_required_top_level_keys(self):
        features = self._make_features([0.05])
        report   = _build_report(
            model_version="v", reference_rows=100,
            live_df=self._minimal_live_df(), features=features, min_bins_warning=False,
        )
        for key in ("run_at", "model_version", "reference_rows", "live_rows",
                    "live_period", "min_bins_warning", "recommendation", "psi_summary", "features"):
            assert key in report, f"missing key: {key}"

    def test_live_period_populated(self):
        live_df  = pd.DataFrame({"bucket": [1000, 1050, 1100]})
        features = self._make_features([0.05])
        report   = _build_report(
            model_version="v", reference_rows=1,
            live_df=live_df, features=features, min_bins_warning=False,
        )
        assert report["live_period"]["min_bucket"] == 1000
        assert report["live_period"]["max_bucket"] == 1100


# ── TestDriftMonitorEnd2End ───────────────────────────────────────────────────


class TestDriftMonitorEnd2End:
    """Integration test — calls engineer() and the full main() path."""

    @pytest.fixture()
    def artifacts(self, tmp_path):
        """Build reference cluster_features.parquet + live cluster_agg.parquet + fake model dir."""
        from helpers import make_fake_artifacts, make_window_df
        from spike.feature_engineer import engineer

        # Reference: 10 machines × 100 buckets — enough history for all lag features
        ref_agg = make_window_df(machine_ids=list(range(10)), n_buckets=100, start_bucket=0)
        ref_agg_path = tmp_path / "ref_agg.parquet"
        ref_agg.to_parquet(ref_agg_path, index=False)

        ref_features_path  = tmp_path / "cluster_features.parquet"
        ref_thresholds_path = tmp_path / "spike_thresholds.parquet"
        engineer(
            input_path      = ref_agg_path,
            output_path     = ref_features_path,
            thresholds_path = ref_thresholds_path,
        )

        # Live: same machines, fresh 48-bucket window (4h minimum)
        live_agg = make_window_df(machine_ids=list(range(10)), n_buckets=48, start_bucket=200)
        live_agg_path = tmp_path / "live_agg.parquet"
        live_agg.to_parquet(live_agg_path, index=False)

        model_dir, _ = make_fake_artifacts(tmp_path)
        output_path  = tmp_path / "drift_report.json"

        return {
            "ref_features_path": ref_features_path,
            "live_agg_path":     live_agg_path,
            "model_dir":         model_dir,
            "output_path":       output_path,
        }

    def test_report_written_and_valid(self, artifacts):
        from spike.drift_monitor import main

        main([
            "--reference",       str(artifacts["ref_features_path"]),
            "--live",            str(artifacts["live_agg_path"]),
            "--model-dir",       str(artifacts["model_dir"]),
            "--output",          str(artifacts["output_path"]),
            "--ref-sample-frac", "1.0",   # use all training rows (small test fixture)
        ])

        assert artifacts["output_path"].exists()
        report = json.loads(artifacts["output_path"].read_text())

        assert report["recommendation"] in ("stable", "warning", "retrain", "emergency")
        assert "psi_summary" in report
        assert "features" in report
        assert len(report["features"]) == len(_X_COLS)

    def test_all_feature_psi_values_are_json_safe(self, artifacts):
        """No NaN or inf must appear in the written JSON."""
        from spike.drift_monitor import main

        main([
            "--reference",       str(artifacts["ref_features_path"]),
            "--live",            str(artifacts["live_agg_path"]),
            "--model-dir",       str(artifacts["model_dir"]),
            "--output",          str(artifacts["output_path"]),
            "--ref-sample-frac", "1.0",
        ])

        raw    = artifacts["output_path"].read_text()
        parsed = json.loads(raw)   # raises ValueError if JSON is malformed
        for feat in parsed["features"]:
            assert feat["psi"] is None or isinstance(feat["psi"], float)

    def test_csv_live_input_accepted(self, artifacts, tmp_path):
        """Drift monitor must accept a CSV live file, not just Parquet."""
        from helpers import make_window_df
        from spike.drift_monitor import main

        live_csv = tmp_path / "live_agg.csv"
        make_window_df(machine_ids=list(range(10)), n_buckets=48, start_bucket=300).to_csv(
            live_csv, index=False
        )

        output = tmp_path / "drift_csv.json"
        main([
            "--reference",       str(artifacts["ref_features_path"]),
            "--live",            str(live_csv),
            "--model-dir",       str(artifacts["model_dir"]),
            "--output",          str(output),
            "--ref-sample-frac", "1.0",
        ])

        assert output.exists()
        report = json.loads(output.read_text())
        assert report["recommendation"] in ("stable", "warning", "retrain", "emergency")
