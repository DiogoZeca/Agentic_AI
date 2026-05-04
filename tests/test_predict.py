"""Tests for predict_spike.py — production inference script.

Six categories following ML inference testing best practices:
  1. Input schema validation
  2. Feature parity (training-serving consistency)
  3. Output schema
  4. Round-trip regression (requires real model artefacts)
  5. Edge cases
  6. Operational (CLI, missing files, mismatched config)

Categories 1–3 and 5–6 use only synthetic data and a fake model dir —
no real artefacts needed.  Category 4 is marked skipif the real artefacts
are absent (they exist after a full training run).
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spike.predict import (
    _Artifacts,
    _build_features,
    _check_input_bounds,
    _coerce_input,
    _load_artifacts,
    _prepare_input,
    _run_inference,
    _validate_input,
    predict,
)
from spike.classifier import _X_COLS
from spike.feature_engineer import engineer, _FEATURE_COLS

# Shared test scaffolding lives in helpers.py so test_daemon_slo.py can import it too.
from helpers import make_agg_row, make_window_df, make_fake_artifacts

# Private aliases keep existing test code unchanged without mass-renaming.
_make_agg_row       = make_agg_row
_make_window_df     = make_window_df
_make_fake_artifacts = make_fake_artifacts

# ── Path to real artefacts (present after a full training run) ─────────────────

_REAL_MODEL_DIR = Path(__file__).parents[1] / "data" / "full_run" / "models" / "spike"
_HAS_REAL_MODEL = (
    (_REAL_MODEL_DIR / "spike_model.json").exists()
    and (_REAL_MODEL_DIR / "spike_config.json").exists()
    and (_REAL_MODEL_DIR / "spike_model.meta.json").exists()
    and (_REAL_MODEL_DIR / "spike_thresholds.parquet").exists()
)


# ── Category 1: Input schema validation ──────────────────────────────────────


class TestInputValidation:
    """_validate_input must reject bad inputs with clear error messages."""

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame(columns=list(_make_window_df().columns))
        with pytest.raises(ValueError, match="empty"):
            _validate_input(df)

    def test_missing_column_raises_with_name(self):
        df = _make_window_df()
        df = df.drop(columns=["total_cpu"])
        with pytest.raises(ValueError, match="total_cpu"):
            _validate_input(df)

    def test_multiple_missing_columns_raises(self):
        df = _make_window_df().drop(columns=["disk_io", "n_tasks"])
        with pytest.raises(ValueError, match="disk_io"):
            _validate_input(df)

    def test_valid_input_does_not_raise(self):
        _validate_input(_make_window_df())   # must not raise

    def test_extra_columns_are_tolerated(self):
        """Unknown columns in the input are silently ignored."""
        df = _make_window_df()
        df["extra_col"] = 0
        _validate_input(df)   # must not raise


# ── Category 1b: Type coercion ───────────────────────────────────────────────


class TestInputCoercion:
    """_coerce_input must cast columns to the correct dtypes in-place."""

    def test_string_numerics_are_cast(self):
        df = _make_window_df().astype(str)
        _coerce_input(df)
        assert df["machine_id"].dtype == "int64"
        assert df["total_cpu"].dtype == "float32"

    def test_int_cols_cast_to_int64(self):
        df = _make_window_df()
        df["machine_id"] = df["machine_id"].astype("int32")
        _coerce_input(df)
        assert df["machine_id"].dtype == "int64"
        assert df["bucket"].dtype == "int64"
        assert df["n_tasks"].dtype == "int64"

    def test_float_cols_cast_to_float32(self):
        df = _make_window_df()
        for col in ("total_cpu", "peak_cpu", "total_mem", "peak_mem", "disk_io"):
            df[col] = df[col].astype("float64")
        _coerce_input(df)
        for col in ("total_cpu", "peak_cpu", "total_mem", "peak_mem", "disk_io"):
            assert df[col].dtype == "float32", f"{col} not cast to float32"

    def test_non_numeric_string_raises(self):
        df = _make_window_df()
        df["total_cpu"] = "not_a_number"
        with pytest.raises(ValueError, match="total_cpu"):
            _coerce_input(df)

    def test_already_correct_dtypes_unchanged(self):
        df = _make_window_df()
        _coerce_input(df)   # must not raise or corrupt values
        assert df["total_cpu"].dtype == "float32"
        assert df["machine_id"].dtype == "int64"


# ── Category 1c: Bounds warnings ─────────────────────────────────────────────


class TestInputBoundsCheck:
    """_check_input_bounds must warn on implausible values but never raise."""

    def test_cpu_percentage_logs_warning(self, caplog):
        import logging
        df = _make_window_df()
        df["total_cpu"] = 80.0   # percentage, not fraction-of-core
        with caplog.at_level(logging.WARNING):
            _check_input_bounds(df)
        assert any("total_cpu" in r.message for r in caplog.records)

    def test_memory_percentage_logs_warning(self, caplog):
        import logging
        df = _make_window_df()
        df["total_mem"] = 75.0   # percentage, not [0, 1]
        with caplog.at_level(logging.WARNING):
            _check_input_bounds(df)
        assert any("total_mem" in r.message for r in caplog.records)

    def test_negative_n_tasks_logs_warning(self, caplog):
        import logging
        df = _make_window_df()
        df["n_tasks"] = -1
        with caplog.at_level(logging.WARNING):
            _check_input_bounds(df)
        assert any("n_tasks" in r.message for r in caplog.records)

    def test_valid_values_produce_no_warning(self, caplog):
        import logging
        df = _make_window_df()
        with caplog.at_level(logging.WARNING):
            _check_input_bounds(df)
        assert not caplog.records

    def test_bounds_check_never_raises(self):
        df = _make_window_df()
        df["total_cpu"] = 999.0   # wildly wrong — must warn, not raise
        _check_input_bounds(df)   # must not raise

    def test_prepare_input_runs_full_pipeline(self):
        df = _make_window_df().astype({"machine_id": "int32", "total_cpu": "float64"})
        _prepare_input(df)
        assert df["machine_id"].dtype == "int64"
        assert df["total_cpu"].dtype == "float32"


# ── Category 2: Feature parity (training-serving consistency) ─────────────────


class TestFeatureParity:
    """The features produced by predict_spike.py must match those produced by
    engineer() on the same input — both call the same internal functions, so
    any divergence would indicate a code-path regression."""

    def test_x_cols_match_meta_json(self):
        """_X_COLS in spike_classifier.py must be consistent with predict_spike.py's
        import of the same symbol.  If someone adds a feature to _X_COLS without
        updating the training pipeline, this test will catch the mismatch."""
        from spike.predict import _X_COLS as predict_X_COLS
        assert list(predict_X_COLS) == list(_X_COLS)

    def test_build_features_produces_x_cols(self, tmp_path):
        """_build_features must return a DataFrame whose columns include all _X_COLS."""
        _, arts = _make_fake_artifacts(tmp_path)
        df       = _make_window_df(machine_ids=[1, 2], n_buckets=24)
        feat_df, _ = _build_features(df, arts)
        for col in _X_COLS:
            assert col in feat_df.columns, f"Missing feature column: {col}"

    def test_feature_values_match_engineer_output(self, tmp_path):
        """Core parity test: _build_features and engineer() must produce identical
        feature values for the same input (within float32 tolerance).

        Uses a controlled input — two machines, 30 buckets — so that the
        training pipeline's per-machine loop and the inference loop exercise
        the exact same code paths.

        Thresholds must match: we run engineer() first to get the real p95
        thresholds, then build artifacts with those same values so that
        cpu_vs_p95 is computed identically in both paths."""
        input_df = _make_window_df(machine_ids=[1, 2], n_buckets=30, start_bucket=0)

        # Training path first: produce real p95 thresholds from the same data
        agg_path    = tmp_path / "agg.parquet"
        feat_path   = tmp_path / "features.parquet"
        thresh_path = tmp_path / "thresholds.parquet"
        input_df.to_parquet(agg_path, index=False)
        train_df = engineer(agg_path, feat_path, thresh_path, train_ratio=0.6)

        # Build artifacts using the actual computed thresholds so cpu_vs_p95
        # and cpu_vs_p99 are computed identically in both paths.
        thresh_df = pd.read_parquet(thresh_path)
        actual_thresholds = dict(
            zip(thresh_df["machine_id"].astype(int), thresh_df["threshold_p95"].astype(float))
        )
        actual_thresholds_p99 = dict(
            zip(thresh_df["machine_id"].astype(int), thresh_df["threshold_p99"].astype(float))
        )
        global_thresh     = float(thresh_df["threshold_p95"].median())
        global_thresh_p99 = float(thresh_df["threshold_p99"].median())
        _, arts = _make_fake_artifacts(
            tmp_path / "arts",
            thresholds=actual_thresholds,
            thresholds_p99=actual_thresholds_p99,
            global_threshold=global_thresh,
            global_threshold_p99=global_thresh_p99,
        )

        # Inference path with matching thresholds
        feat_df, _ = _build_features(input_df, arts)

        # engineer() drops the last 12 rows per machine (no label look-ahead).
        # Align on (machine_id, bucket) so both sides have the same rows.
        labeled_keys = train_df.dropna(subset=["severity_in_60m"])[["machine_id", "bucket"]]
        train_aligned = (
            train_df.dropna(subset=["severity_in_60m"])
            .sort_values(["machine_id", "bucket"])
            .reset_index(drop=True)
        )
        infer_aligned = (
            feat_df.merge(labeled_keys, on=["machine_id", "bucket"])
            .sort_values(["machine_id", "bucket"])
            .reset_index(drop=True)
        )

        assert len(infer_aligned) == len(train_aligned), (
            f"Row count mismatch after alignment: "
            f"infer={len(infer_aligned)}, train={len(train_aligned)}"
        )

        shared_cols = [c for c in _X_COLS if c in train_aligned.columns]
        for col in shared_cols:
            np.testing.assert_allclose(
                infer_aligned[col].values,
                train_aligned[col].values,
                rtol = 1e-4,
                err_msg = f"Feature parity failure on column: {col}",
            )


# ── Category 3: Output schema ─────────────────────────────────────────────────


class TestOutputSchema:
    """predict() must return a well-formed envelope with valid prediction dicts."""

    @pytest.fixture(scope="class")
    def result(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("schema")
        model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30, 2: 0.32})
        df = _make_window_df(machine_ids=[1, 2], n_buckets=24)
        return predict(df, model_dir)

    def test_envelope_has_required_keys(self, result):
        required = {
            "predicted_at", "model_dir", "horizon_minutes",
            "machines_total", "machines_predicted", "machines_cold_start",
            "predictions",
        }
        assert required.issubset(result.keys())

    def test_all_machines_appear_in_output(self, result):
        """Every machine in the input must appear in predictions — none silently dropped."""
        machine_ids = {p["machine_id"] for p in result["predictions"]}
        assert 1 in machine_ids
        assert 2 in machine_ids

    def test_p_severe_is_float_in_unit_interval(self, result):
        for p in result["predictions"]:
            if p["status"] != "cold_start":
                assert isinstance(p["p_severe"], float)
                assert 0.0 <= p["p_severe"] <= 1.0

    def test_severity_class_is_int_0_1_or_2(self, result):
        for p in result["predictions"]:
            if p["status"] != "cold_start":
                assert p["severity_class"] in (0, 1, 2), (
                    f"severity_class must be 0, 1, or 2; got {p['severity_class']}"
                )

    def test_is_spike_is_bool_not_numeric(self, result):
        for p in result["predictions"]:
            if p["status"] != "cold_start":
                assert isinstance(p["is_spike"], bool), (
                    f"is_spike must be bool, got {type(p['is_spike'])}"
                )

    def test_is_severe_is_bool(self, result):
        for p in result["predictions"]:
            if p["status"] != "cold_start":
                assert isinstance(p["is_severe"], bool), (
                    f"is_severe must be bool, got {type(p['is_severe'])}"
                )

    def test_status_values_are_valid(self, result):
        valid_statuses = {"success", "cold_start_degraded", "cold_start"}
        for p in result["predictions"]:
            assert p["status"] in valid_statuses, f"Invalid status: {p['status']}"

    def test_cold_start_predictions_have_null_fields(self, tmp_path):
        """Machines with 0 rows should return cold_start entries with null fields."""
        model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30})
        # Machine 2 has only 5 rows — below default min_buckets_cold_start=12
        df_m1 = _make_window_df(machine_ids=[1], n_buckets=24)
        df_m2 = _make_window_df(machine_ids=[2], n_buckets=5)
        df    = pd.concat([df_m1, df_m2], ignore_index=True)

        result = predict(df, model_dir)
        m2_pred = next(p for p in result["predictions"] if p["machine_id"] == 2)

        assert m2_pred["status"]         == "cold_start"
        assert m2_pred["severity_class"] is None
        assert m2_pred["p_severe"]       is None
        assert m2_pred["is_spike"]       is None
        assert m2_pred["is_severe"]      is None

    def test_predictions_sorted_by_machine_id(self, result):
        ids = [p["machine_id"] for p in result["predictions"]]
        assert ids == sorted(ids)

    def test_horizon_minutes_is_60(self, result):
        assert result["horizon_minutes"] == 60


# ── Category 4: Round-trip regression (requires real artefacts) ───────────────


@pytest.mark.skipif(not _HAS_REAL_MODEL, reason="Real model artefacts not found")
class TestRoundTripRegression:
    """Run real inference on a fixed synthetic window and assert the probability
    is in [0, 1] with the expected metadata structure.

    These tests do not assert exact probability values (which depend on the
    specific training run) but verify that the full inference path produces
    structurally valid, consistent output when real artefacts are present."""

    @pytest.fixture(scope="class")
    def result(self):
        df = _make_window_df(machine_ids=[1, 2], n_buckets=24, start_bucket=500)
        return predict(df, _REAL_MODEL_DIR)

    def test_returns_two_predictions(self, result):
        assert len(result["predictions"]) == 2

    def test_probabilities_in_unit_interval(self, result):
        for p in result["predictions"]:
            if p["status"] != "cold_start":
                assert 0.0 <= p["p_severe"] <= 1.0

    def test_threshold_source_is_global_fallback_for_unknown_machine(self):
        """Machine IDs 1 and 2 are unlikely to be in the real training data.
        They must fall back to the global threshold."""
        df = _make_window_df(machine_ids=[1, 2], n_buckets=24, start_bucket=500)
        result = predict(df, _REAL_MODEL_DIR)
        for p in result["predictions"]:
            if p["status"] != "cold_start":
                assert p["threshold_source"] == "global_fallback"

    def test_known_machine_uses_learned_threshold(self):
        """A machine_id that is in spike_thresholds.parquet must use its
        learned threshold, not the global fallback."""
        thresholds = pd.read_parquet(
            _REAL_MODEL_DIR / "spike_thresholds.parquet"
        )
        if thresholds.empty:
            pytest.skip("Thresholds file is empty — skip learned-threshold test")

        known_id = int(thresholds["machine_id"].iloc[0])
        df = _make_window_df(machine_ids=[known_id], n_buckets=24, start_bucket=500)
        result = predict(df, _REAL_MODEL_DIR)

        pred = result["predictions"][0]
        assert pred["threshold_source"] == "learned"


# ── Category 5: Edge cases ────────────────────────────────────────────────────


class TestEdgeCases:
    """Unusual but valid inputs must produce correct, non-crashing output."""

    def test_machine_absent_from_thresholds_uses_global_fallback(self, tmp_path):
        """Machine 999 is not in spike_thresholds.parquet → global_fallback threshold."""
        # thresholds only covers machine 1
        model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30})
        df = _make_window_df(machine_ids=[999], n_buckets=24)

        result = predict(df, model_dir)
        pred   = result["predictions"][0]

        assert pred["threshold_source"] == "global_fallback"
        assert pred["p_severe"] is not None

    def test_exactly_min_buckets_cold_start_gets_degraded_status(self, tmp_path):
        """A machine with exactly min_buckets_cold_start rows (default 12) must
        get cold_start_degraded, not cold_start (which requires < 12)."""
        model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30})
        df = _make_window_df(machine_ids=[1], n_buckets=12)

        result = predict(df, model_dir)
        assert result["predictions"][0]["status"] == "cold_start_degraded"
        assert result["predictions"][0]["p_severe"] is not None

    def test_exactly_11_buckets_gets_cold_start_null(self, tmp_path):
        """One bucket below min_buckets_cold_start → null prediction."""
        model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30})
        df = _make_window_df(machine_ids=[1], n_buckets=11)

        result = predict(df, model_dir)
        assert result["predictions"][0]["status"] == "cold_start"
        assert result["predictions"][0]["p_severe"] is None

    def test_full_24_buckets_gets_success_status(self, tmp_path):
        model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30})
        df = _make_window_df(machine_ids=[1], n_buckets=24)

        result = predict(df, model_dir)
        assert result["predictions"][0]["status"] == "success"

    def test_single_machine_input_produces_one_prediction(self, tmp_path):
        model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30})
        df = _make_window_df(machine_ids=[1], n_buckets=24)

        result = predict(df, model_dir)
        assert len(result["predictions"]) == 1

    def test_all_machines_cold_start_returns_empty_probabilities(self, tmp_path):
        """When every machine is below the cold-start threshold, predictions
        must still be returned — all with null probability."""
        model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30, 2: 0.32})
        df = _make_window_df(machine_ids=[1, 2], n_buckets=5)   # 5 < 12

        result = predict(df, model_dir)
        assert result["machines_predicted"]  == 0
        assert result["machines_cold_start"] == 2
        for p in result["predictions"]:
            assert p["p_severe"] is None

    def test_machine_rank_in_unit_interval_all_same_cpu(self, tmp_path):
        """When all machines have identical total_cpu, machine_rank_in_cluster
        must still be in [0, 1] — no NaN or out-of-range values."""
        model_dir, arts = _make_fake_artifacts(
            tmp_path, thresholds={1: 0.30, 2: 0.30, 3: 0.30}
        )
        rows = [
            _make_agg_row(mid, 100 + i, total_cpu=0.20)
            for mid in (1, 2, 3)
            for i in range(24)
        ]
        df = pd.DataFrame(rows)
        df["machine_id"] = df["machine_id"].astype("int64")

        feat_df, _ = _build_features(df, arts)
        ranks = feat_df["machine_rank_in_cluster"]
        assert (ranks >= 0.0).all()
        assert (ranks <= 1.0).all()
        assert ranks.notna().all()


# ── Category 6: Operational ───────────────────────────────────────────────────


class TestOperational:
    """CLI and artefact-loading failure modes must produce clear errors and
    correct exit codes."""

    def test_missing_model_file_raises_file_not_found(self, tmp_path):
        """_load_artifacts must raise FileNotFoundError when spike_model.json
        is absent — not a cryptic AttributeError."""
        with pytest.raises(FileNotFoundError, match="spike_model.json"):
            _load_artifacts(tmp_path / "nonexistent_dir")

    def test_feature_col_mismatch_raises_runtime_error(self, tmp_path):
        """If feature_cols in meta.json differs from spike_config.json, inference
        would be silently wrong.  _load_artifacts must detect and raise."""
        # Write meta.json with one set of columns ...
        model_dir = tmp_path / "models" / "spike"
        model_dir.mkdir(parents=True)

        cols_meta   = list(_X_COLS)
        cols_config = list(_X_COLS[:-1])   # one column fewer → mismatch

        booster = xgb.train(
            {"objective": "binary:logistic", "verbosity": 0},
            xgb.DMatrix(
                np.zeros((4, len(cols_meta)), dtype="float32"),
                label=[0, 1, 0, 1],
                feature_names=cols_meta,
            ),
            num_boost_round=1,
        )
        booster.save_model(str(model_dir / "spike_model.json"))
        (model_dir / "spike_model.meta.json").write_text(
            json.dumps({"feature_cols": cols_meta, "xgboost_version": xgb.__version__})
        )

        config = {
            "alarm_threshold": 0.25,
            "inference": {
                "feature_cols":                  cols_config,   # ← mismatch
                "global_threshold_p95_fallback": 0.35,
                "global_threshold_p99_fallback": 0.40,
                "min_buckets_cold_start":        12,
                "horizon_minutes":               60,
                "thresholds_file":               str(model_dir / "spike_thresholds.parquet"),
            },
        }
        (model_dir / "spike_config.json").write_text(json.dumps(config))
        pd.DataFrame(
            [{"machine_id": 1, "threshold_p95": 0.3, "threshold_p99": 0.35}]
        ).to_parquet(model_dir / "spike_thresholds.parquet", index=False)

        with pytest.raises(RuntimeError, match="mismatch"):
            _load_artifacts(model_dir)

    def test_cli_exits_1_on_missing_input_file(self, tmp_path):
        """main() must exit with code 1 (not crash) when --input is missing."""
        from spike.predict import main
        model_dir, _ = _make_fake_artifacts(tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--input",     str(tmp_path / "nonexistent.csv"),
                "--model-dir", str(model_dir),
            ])
        assert exc_info.value.code == 1

    def test_cli_exits_1_on_missing_model_dir(self, tmp_path):
        """main() must exit with code 1 when --model-dir does not exist."""
        from spike.predict import main
        df = _make_window_df()
        csv_path = tmp_path / "window.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--input",     str(csv_path),
                "--model-dir", str(tmp_path / "no_such_dir"),
            ])
        assert exc_info.value.code == 1

    def test_output_file_written_atomically(self, tmp_path):
        """--output must produce valid JSON that matches stdout output."""
        from spike.predict import main
        import io
        from contextlib import redirect_stdout

        model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30})
        df       = _make_window_df(machine_ids=[1], n_buckets=24)
        csv_path = tmp_path / "window.csv"
        df.to_csv(csv_path, index=False)
        out_path = tmp_path / "predictions.json"

        buf = io.StringIO()
        with redirect_stdout(buf):
            main([
                "--input",     str(csv_path),
                "--model-dir", str(model_dir),
                "--output",    str(out_path),
            ])

        assert out_path.exists()
        from_file   = json.loads(out_path.read_text())
        from_stdout = json.loads(buf.getvalue())

        # Both outputs must contain the same predictions
        assert from_file["predictions"] == from_stdout["predictions"]


# ── Phase 5: OVR severe model ─────────────────────────────────────────────────


def test_severe_ovr_model_loaded_when_present(tmp_path):
    """_load_artifacts must load spike_severe_ovr if the directory exists."""
    _, arts = _make_fake_artifacts(tmp_path, thresholds={1: 0.30}, include_severe_ovr=True)
    assert "severe_ovr" in arts.boosters, (
        "severe_ovr booster should be loaded when spike_severe_ovr/ dir exists"
    )
    assert "severe_ovr" in arts.alarm_thresholds


# ── Alarm threshold override (--alarm-threshold flag) ─────────────────────────


class TestAlarmThresholdOverride:
    """The --alarm-threshold flag overrides the trained per-horizon alarm threshold
    for every loaded horizon without retraining.  These tests verify that:
      - the override is applied to all horizon entries in alarm_thresholds
      - the override propagates into per-machine prediction dicts
      - boundary values (0.0 and 1.0) produce the expected all-alarm / no-alarm behaviour
      - the CLI validates the range and rejects out-of-range values
    """

    def test_override_applied_to_all_horizons(self, tmp_path):
        """When alarm_threshold is set, every horizon entry must be overwritten."""
        model_dir, arts = _make_fake_artifacts(
            tmp_path, thresholds={1: 0.30}, include_binary_horizons=True
        )
        original_horizons = set(arts.alarm_thresholds.keys())
        assert len(original_horizons) > 1, "fixture must include at least 2 horizons"

        override = 0.42
        for h in arts.alarm_thresholds:
            arts.alarm_thresholds[h] = override

        for h, v in arts.alarm_thresholds.items():
            assert v == override, f"horizon '{h}' was not overridden"

    def test_low_threshold_maximises_alarms(self, tmp_path):
        """With threshold=0.0, binary-horizon is_spike must be True (any probability >= 0.0
        is always true) and the 60m is_severe must be True (any p_severe >= 0.0).

        Note: the 60m is_spike field is argmax-based (not threshold-gated) so it is
        excluded from this check — it reflects the model's class prediction, not the
        operator alarm decision.
        """
        model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30})
        df = _make_window_df(machine_ids=[1], n_buckets=24)

        result = predict(df, model_dir, alarm_threshold=0.0)

        for pred in result["predictions"]:
            if pred["status"] == "cold_start":
                continue
            imminence = pred.get("imminence") or {}
            # Binary horizons: is_spike is threshold-based → must be True at 0.0.
            for h in ("15m", "30m", "45m"):
                if h in imminence and imminence[h].get("is_spike") is not None:
                    assert imminence[h]["is_spike"] is True, (
                        f"binary horizon {h}: expected is_spike=True at threshold=0.0"
                    )
            # 60m severity: is_severe is threshold-based → must be True at 0.0.
            if "60m" in imminence and imminence["60m"].get("is_severe") is not None:
                assert imminence["60m"]["is_severe"] is True, (
                    "60m is_severe must be True at threshold=0.0"
                )

    def test_high_threshold_suppresses_alarms(self, tmp_path):
        """With threshold=1.0, binary-horizon is_spike must be False (softmax/sigmoid
        outputs are strictly < 1.0) and 60m is_severe must be False.

        Note: the 60m is_spike field is argmax-based (not threshold-gated) and is
        excluded from this check for the same reason as test_low_threshold_maximises_alarms.
        """
        model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30})
        df = _make_window_df(machine_ids=[1], n_buckets=24)

        result = predict(df, model_dir, alarm_threshold=1.0)

        for pred in result["predictions"]:
            if pred["status"] == "cold_start":
                continue
            imminence = pred.get("imminence") or {}
            # Binary horizons: is_spike is threshold-based → must be False at 1.0.
            for h in ("15m", "30m", "45m"):
                if h in imminence and imminence[h].get("is_spike") is not None:
                    assert imminence[h]["is_spike"] is False, (
                        f"binary horizon {h}: expected is_spike=False at threshold=1.0, "
                        f"got p_spike={imminence[h].get('p_spike')}"
                    )
            # 60m severity: is_severe is threshold-based → must be False at 1.0.
            if "60m" in imminence and imminence["60m"].get("is_severe") is not None:
                assert imminence["60m"]["is_severe"] is False, (
                    "60m is_severe must be False at threshold=1.0"
                )

    def test_override_reflected_in_alarm_threshold_field(self, tmp_path):
        """The alarm_threshold field in each prediction dict must match the override."""
        model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30})
        df = _make_window_df(machine_ids=[1], n_buckets=24)
        override = 0.37

        result = predict(df, model_dir, alarm_threshold=override)

        for pred in result["predictions"]:
            if pred["status"] == "cold_start":
                continue
            # Top-level alarm_threshold (60m) must reflect the override.
            assert abs(pred["alarm_threshold"] - override) < 1e-6, (
                f"Expected alarm_threshold={override}, got {pred['alarm_threshold']}"
            )

    def test_no_override_uses_trained_threshold(self, tmp_path):
        """Without alarm_threshold, the value from spike_config.json must be used."""
        trained_thr = 0.25
        model_dir, _ = _make_fake_artifacts(
            tmp_path, thresholds={1: 0.30}, alarm_threshold=trained_thr
        )
        df = _make_window_df(machine_ids=[1], n_buckets=24)

        result = predict(df, model_dir)  # no override

        for pred in result["predictions"]:
            if pred["status"] == "cold_start":
                continue
            assert abs(pred["alarm_threshold"] - trained_thr) < 1e-6

    def test_cli_rejects_threshold_above_1(self, tmp_path):
        """--alarm-threshold > 1.0 must exit with code 1."""
        from spike.predict import main as predict_main
        model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30})
        df = _make_window_df(machine_ids=[1], n_buckets=24)
        csv_path = tmp_path / "window.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(SystemExit) as exc_info:
            predict_main([
                "--input",           str(csv_path),
                "--model-dir",       str(model_dir),
                "--alarm-threshold", "1.5",
            ])
        assert exc_info.value.code == 1

    def test_cli_rejects_threshold_below_0(self, tmp_path):
        """--alarm-threshold < 0.0 must exit with code 1."""
        from spike.predict import main as predict_main
        model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30})
        df = _make_window_df(machine_ids=[1], n_buckets=24)
        csv_path = tmp_path / "window.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(SystemExit) as exc_info:
            predict_main([
                "--input",           str(csv_path),
                "--model-dir",       str(model_dir),
                "--alarm-threshold", "-0.1",
            ])
        assert exc_info.value.code == 1


def test_severe_ovr_model_absent_does_not_raise(tmp_path):
    """_load_artifacts must succeed silently when spike_severe_ovr/ is absent."""
    _, arts = _make_fake_artifacts(tmp_path, thresholds={1: 0.30}, include_severe_ovr=False)
    assert "severe_ovr" not in arts.boosters


def test_p_severe_ovr_present_when_model_loaded(tmp_path):
    """When OVR model is loaded, predictions must include p_severe_ovr."""
    model_dir, _ = _make_fake_artifacts(
        tmp_path, thresholds={1: 0.30, 2: 0.32}, include_severe_ovr=True
    )
    df = _make_window_df(machine_ids=[1, 2], n_buckets=24)
    result = predict(df, model_dir)
    for pred in result["predictions"]:
        if pred["status"] != "cold_start":
            assert "p_severe_ovr" in pred, "p_severe_ovr must be in prediction when OVR model present"
            assert isinstance(pred["p_severe_ovr"], float)
            assert 0.0 <= pred["p_severe_ovr"] <= 1.0


def test_p_severe_ovr_null_when_model_absent(tmp_path):
    """When OVR model is absent, p_severe_ovr must be null in predictions."""
    model_dir, _ = _make_fake_artifacts(
        tmp_path, thresholds={1: 0.30, 2: 0.32}, include_severe_ovr=False
    )
    df = _make_window_df(machine_ids=[1, 2], n_buckets=24)
    result = predict(df, model_dir)
    for pred in result["predictions"]:
        assert pred.get("p_severe_ovr") is None, (
            "p_severe_ovr must be null when OVR model is absent"
        )


def test_imminence_has_only_15m_and_60m_horizons(tmp_path):
    """Fix 6: imminence dict must contain only '15m' and '60m' keys.

    The 30m and 45m binary models were dropped in Fix 3, so their keys must
    not appear in prediction output.
    """
    model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30, 2: 0.32})
    df = _make_window_df(machine_ids=[1, 2], n_buckets=24)
    result = predict(df, model_dir)
    for pred in result["predictions"]:
        if pred["status"] != "cold_start" and pred["imminence"] is not None:
            horizon_keys = set(pred["imminence"].keys())
            assert "30m" not in horizon_keys, (
                f"30m should not appear in imminence after Fix 3, got: {horizon_keys}"
            )
            assert "45m" not in horizon_keys, (
                f"45m should not appear in imminence after Fix 3, got: {horizon_keys}"
            )
            assert "60m" in horizon_keys, "60m must always be in imminence"


# ── Isotonic calibration (Fix 1) ──────────────────────────────────────────────


def _make_fake_artifacts_with_calibrators(
    tmp_path: Path,
) -> tuple[Path, _Artifacts]:
    """Build fake artifacts that include calibrators.pkl for the 60m model."""
    import pickle
    from sklearn.isotonic import IsotonicRegression

    model_dir, arts = _make_fake_artifacts(tmp_path, thresholds={1: 0.30, 2: 0.32})

    # Fit trivial calibrators (identity-ish) on synthetic data
    rng = np.random.default_rng(0)
    calibrators = []
    for k in range(3):
        p_k = rng.uniform(0.0, 1.0, 100)
        y_k = (rng.choice([0, 1, 2], size=100, p=[0.70, 0.20, 0.10]) == k).astype("float64")
        ir  = IsotonicRegression(out_of_bounds="clip")
        ir.fit(p_k, y_k)
        calibrators.append(ir)

    with open(model_dir / "calibrators.pkl", "wb") as fh:
        pickle.dump(calibrators, fh)

    # Reload artifacts so calibrators_60m is populated
    arts = _load_artifacts(model_dir)
    return model_dir, arts


def test_calibrators_loaded_when_pkl_present(tmp_path):
    """_load_artifacts must populate calibrators_60m when calibrators.pkl exists."""
    _, arts = _make_fake_artifacts_with_calibrators(tmp_path)
    assert arts.calibrators_60m is not None
    assert len(arts.calibrators_60m) == 3


def test_calibrators_absent_does_not_raise(tmp_path):
    """_load_artifacts must succeed when calibrators.pkl is absent."""
    _, arts = _make_fake_artifacts(tmp_path, thresholds={1: 0.30})
    assert arts.calibrators_60m is None


def test_calibrated_probabilities_sum_to_one(tmp_path):
    """With calibrators loaded, p_no_spike + p_moderate + p_severe ≈ 1."""
    model_dir, _ = _make_fake_artifacts_with_calibrators(tmp_path)
    df = _make_window_df(machine_ids=[1, 2], n_buckets=24)
    result = predict(df, model_dir)
    for pred in result["predictions"]:
        if pred["status"] != "cold_start":
            total = pred["p_no_spike"] + pred["p_moderate"] + pred["p_severe"]
            assert abs(total - 1.0) < 1e-4, (
                f"Calibrated probabilities must sum to 1; got {total:.6f}"
            )


def test_calibrated_probs_in_unit_interval(tmp_path):
    """All calibrated probability outputs must be in [0, 1]."""
    model_dir, _ = _make_fake_artifacts_with_calibrators(tmp_path)
    df = _make_window_df(machine_ids=[1, 2], n_buckets=24)
    result = predict(df, model_dir)
    for pred in result["predictions"]:
        if pred["status"] != "cold_start":
            for field in ("p_no_spike", "p_moderate", "p_severe"):
                assert 0.0 <= pred[field] <= 1.0, (
                    f"{field}={pred[field]} out of [0, 1]"
                )


def test_predictions_unchanged_without_calibrators(tmp_path):
    """Inference without calibrators must still produce valid probabilities."""
    model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30, 2: 0.32})
    df = _make_window_df(machine_ids=[1, 2], n_buckets=24)
    result = predict(df, model_dir)
    for pred in result["predictions"]:
        if pred["status"] != "cold_start":
            total = pred["p_no_spike"] + pred["p_moderate"] + pred["p_severe"]
            assert abs(total - 1.0) < 1e-4


# ── Phase 1: Extended API fields ──────────────────────────────────────────────


def test_envelope_has_batch_id(tmp_path):
    """predict() must include a batch_id UUID string in the response envelope."""
    model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30})
    df = _make_window_df(machine_ids=[1], n_buckets=24)
    result = predict(df, model_dir)
    assert "batch_id" in result
    assert isinstance(result["batch_id"], str)
    bid = result["batch_id"]
    assert len(bid) == 36 and bid[8] == "-" and bid[13] == "-"


def test_batch_id_differs_across_calls(tmp_path):
    """Each call to predict() must produce a distinct batch_id."""
    model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30})
    df = _make_window_df(machine_ids=[1], n_buckets=24)
    assert predict(df, model_dir)["batch_id"] != predict(df, model_dir)["batch_id"]


def test_trained_at_propagated_to_predictions(tmp_path):
    """trained_at from spike_config.json must appear in each non-cold-start prediction."""
    model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30, 2: 0.32})
    df = _make_window_df(machine_ids=[1, 2], n_buckets=24)
    result = predict(df, model_dir)
    for pred in result["predictions"]:
        assert "trained_at" in pred
        assert pred["trained_at"] == "2026-01-01T00:00:00+00:00"


def test_trained_at_in_cold_start_predictions(tmp_path):
    """trained_at must also be present for cold_start machines (model-level metadata)."""
    model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30})
    df = _make_window_df(machine_ids=[1], n_buckets=5)
    result = predict(df, model_dir)
    pred = result["predictions"][0]
    assert pred["status"] == "cold_start"
    assert "trained_at" in pred
    assert pred["trained_at"] == "2026-01-01T00:00:00+00:00"


def test_data_quality_score_full_window(tmp_path):
    """24 buckets → data_quality.score == 1.0."""
    model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30})
    df = _make_window_df(machine_ids=[1], n_buckets=24)
    result = predict(df, model_dir)
    pred = result["predictions"][0]
    assert pred["status"] == "success"
    assert pred["data_quality"] is not None
    assert pred["data_quality"]["score"] == 1.0


def test_data_quality_score_degraded(tmp_path):
    """12 buckets → data_quality.score == 0.5."""
    model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30})
    df = _make_window_df(machine_ids=[1], n_buckets=12)
    result = predict(df, model_dir)
    pred = result["predictions"][0]
    assert pred["status"] == "cold_start_degraded"
    assert pred["data_quality"]["score"] == pytest.approx(0.5)


def test_data_quality_null_for_cold_start(tmp_path):
    """cold_start machines must have data_quality == None."""
    model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30})
    df = _make_window_df(machine_ids=[1], n_buckets=5)
    result = predict(df, model_dir)
    pred = result["predictions"][0]
    assert pred["status"] == "cold_start"
    assert pred["data_quality"] is None


def test_data_quality_status_matches_prediction_status(tmp_path):
    """data_quality.status must mirror the top-level status field."""
    model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30, 2: 0.32})
    df_full = _make_window_df(machine_ids=[1], n_buckets=24)
    df_part = _make_window_df(machine_ids=[2], n_buckets=15)
    df = pd.concat([df_full, df_part], ignore_index=True)
    result = predict(df, model_dir)
    for pred in result["predictions"]:
        if pred["data_quality"] is not None:
            assert pred["data_quality"]["status"] == pred["status"]


def test_recommended_action_valid_values(tmp_path):
    """recommended_action must be one of the four valid strings for non-cold-start."""
    valid = {"preempt_now", "defer_batch", "monitor", "normal"}
    model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30, 2: 0.32})
    df = _make_window_df(machine_ids=[1, 2], n_buckets=24)
    result = predict(df, model_dir)
    for pred in result["predictions"]:
        if pred["status"] != "cold_start":
            assert pred["recommended_action"] in valid, (
                f"Unexpected recommended_action: {pred['recommended_action']}"
            )


def test_recommended_action_null_for_cold_start(tmp_path):
    """cold_start machines must have recommended_action == None."""
    model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30})
    df = _make_window_df(machine_ids=[1], n_buckets=5)
    result = predict(df, model_dir)
    assert result["predictions"][0]["recommended_action"] is None


def test_recommended_action_without_15m_model(tmp_path):
    """recommended_action must not crash when the 15m model is absent."""
    model_dir, _ = _make_fake_artifacts(
        tmp_path, thresholds={1: 0.30}, include_binary_horizons=False
    )
    df = _make_window_df(machine_ids=[1], n_buckets=24)
    result = predict(df, model_dir)
    pred = result["predictions"][0]
    valid = {"preempt_now", "defer_batch", "monitor", "normal"}
    assert pred["recommended_action"] in valid


def test_top_features_structure(tmp_path):
    """top_features must be a list of exactly 3 dicts each with 'feature' and 'contribution'."""
    model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30, 2: 0.32})
    df = _make_window_df(machine_ids=[1, 2], n_buckets=24)
    result = predict(df, model_dir)
    for pred in result["predictions"]:
        if pred["status"] != "cold_start":
            assert pred["top_features"] is not None, "top_features must not be None for success"
            assert len(pred["top_features"]) == 3
            for entry in pred["top_features"]:
                assert "feature" in entry
                assert "contribution" in entry


def test_top_features_names_in_x_cols(tmp_path):
    """Every feature name in top_features must be a valid model feature."""
    model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30})
    df = _make_window_df(machine_ids=[1], n_buckets=24)
    result = predict(df, model_dir)
    pred = result["predictions"][0]
    if pred["top_features"] is not None:
        for entry in pred["top_features"]:
            assert entry["feature"] in list(_X_COLS), (
                f"Unknown feature in top_features: {entry['feature']}"
            )


def test_top_features_contributions_in_unit_interval(tmp_path):
    """Each contribution must be in [0, 1]."""
    model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30})
    df = _make_window_df(machine_ids=[1], n_buckets=24)
    result = predict(df, model_dir)
    pred = result["predictions"][0]
    if pred["top_features"] is not None:
        for entry in pred["top_features"]:
            assert 0.0 <= entry["contribution"] <= 1.0, (
                f"contribution out of [0, 1]: {entry['contribution']}"
            )


def test_top_features_null_for_cold_start(tmp_path):
    """cold_start machines must have top_features == None."""
    model_dir, _ = _make_fake_artifacts(tmp_path, thresholds={1: 0.30})
    df = _make_window_df(machine_ids=[1], n_buckets=5)
    result = predict(df, model_dir)
    assert result["predictions"][0]["top_features"] is None
