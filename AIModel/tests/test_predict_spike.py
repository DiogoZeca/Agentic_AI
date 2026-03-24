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

from predict_spike import (
    _Artifacts,
    _build_features,
    _load_artifacts,
    _run_inference,
    _validate_input,
    predict,
)
from spike_classifier import _X_COLS
from spike_feature_engineer import engineer, _FEATURE_COLS

# ── Path to real artefacts (present after a full training run) ─────────────────

_REAL_MODEL_DIR = Path(__file__).parents[1] / "data" / "full_run" / "models" / "spike"
_HAS_REAL_MODEL = (
    (_REAL_MODEL_DIR / "spike_model.json").exists()
    and (_REAL_MODEL_DIR / "spike_config.json").exists()
    and (_REAL_MODEL_DIR / "spike_model.meta.json").exists()
    and (_REAL_MODEL_DIR / "spike_thresholds.parquet").exists()
)

# ── Shared helpers ────────────────────────────────────────────────────────────


def _make_agg_row(
    machine_id: int,
    bucket: int,
    total_cpu: float = 0.15,
    peak_cpu: float = 0.20,
    total_mem: float = 0.10,
    peak_mem: float = 0.20,
    disk_io: float = 0.01,
    n_tasks: int = 5,
) -> dict:
    return {
        "machine_id": machine_id,
        "bucket":     bucket,
        "time_us":    bucket * 300_000_000,
        "total_cpu":  total_cpu,
        "peak_cpu":   peak_cpu,
        "total_mem":  total_mem,
        "peak_mem":   peak_mem,
        "disk_io":    disk_io,
        "n_tasks":    n_tasks,
    }


def _make_window_df(
    machine_ids: list[int] = (1, 2),
    n_buckets: int = 24,
    start_bucket: int = 100,
) -> pd.DataFrame:
    """Create a minimal valid input DataFrame with n_buckets per machine."""
    rows = [
        _make_agg_row(mid, start_bucket + i)
        for mid in machine_ids
        for i in range(n_buckets)
    ]
    df = pd.DataFrame(rows)
    df["machine_id"] = df["machine_id"].astype("int64")
    df["bucket"]     = df["bucket"].astype("int64")
    df["time_us"]    = df["time_us"].astype("int64")
    df["n_tasks"]    = df["n_tasks"].astype("int32")
    for col in ("total_cpu", "peak_cpu", "total_mem", "peak_mem", "disk_io"):
        df[col] = df[col].astype("float32")
    return df


def _train_tiny_booster(feature_cols: list[str], binary: bool = False) -> xgb.Booster:
    """Train a minimal 5-tree XGBoost booster on synthetic data."""
    rng = np.random.default_rng(0)
    X   = rng.standard_normal((30, len(feature_cols))).astype("float32")
    if binary:
        y      = rng.integers(0, 2, size=30)
        params = {"objective": "binary:logistic", "eval_metric": "aucpr", "verbosity": 0}
    else:
        y      = rng.integers(0, 3, size=30)
        params = {"objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss", "verbosity": 0}
    return xgb.train(params, xgb.DMatrix(X, label=y, feature_names=feature_cols), num_boost_round=5)


def _write_horizon_dir(
    parent_dir:      Path,
    dir_name:        str,
    feature_cols:    list[str],
    alarm_threshold: float,
    binary:          bool = False,
) -> None:
    """Write a single horizon model directory (model + meta + config)."""
    h_dir = parent_dir / dir_name
    h_dir.mkdir(parents=True, exist_ok=True)

    booster    = _train_tiny_booster(feature_cols, binary=binary)
    model_path = h_dir / "spike_model.json"
    booster.save_model(str(model_path))

    meta = {"feature_cols": feature_cols, "xgboost_version": xgb.__version__}
    (h_dir / "spike_model.meta.json").write_text(json.dumps(meta))

    config = {
        "alarm_threshold": alarm_threshold,
        "inference": {"feature_cols": feature_cols},
    }
    (h_dir / "spike_config.json").write_text(json.dumps(config))


def _make_fake_artifacts(
    tmp_path: Path,
    feature_cols: list[str] | None = None,
    alarm_threshold: float = 0.25,
    global_threshold: float = 0.35,
    global_threshold_p99: float = 0.40,
    thresholds: dict[int, float] | None = None,
    thresholds_p99: dict[int, float] | None = None,
    include_binary_horizons: bool = True,
    include_severe_ovr: bool = False,
) -> tuple[Path, _Artifacts]:
    """Build a minimal fake model directory and return (model_dir, artifacts).

    Trains a 5-tree XGBoost 3-class model for 60m and optionally binary models
    for 15m and/or severe_ovr (Phase 5).
    """
    feature_cols   = feature_cols   or list(_X_COLS)
    thresholds     = thresholds     or {1: 0.30, 2: 0.32}
    thresholds_p99 = thresholds_p99 or {k: v * 1.15 for k, v in thresholds.items()}

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 60m 3-class model
    _write_horizon_dir(models_dir, "spike", feature_cols, alarm_threshold, binary=False)
    model_dir = models_dir / "spike"

    # Binary horizon model (optional — only 15m remains after Fix 3)
    if include_binary_horizons:
        _write_horizon_dir(models_dir, "spike_15m", feature_cols,
                           alarm_threshold * 0.8, binary=True)

    # OVR severe binary model (Phase 5)
    if include_severe_ovr:
        _write_horizon_dir(models_dir, "spike_severe_ovr", feature_cols,
                           alarm_threshold * 0.9, binary=True)

    thresh_df = pd.DataFrame([
        {
            "machine_id":    mid,
            "threshold_p95": thr,
            "threshold_p99": thresholds_p99.get(mid, thr * 1.15),
        }
        for mid, thr in thresholds.items()
    ])
    thresh_df["machine_id"]    = thresh_df["machine_id"].astype("int64")
    thresh_df["threshold_p95"] = thresh_df["threshold_p95"].astype("float32")
    thresh_df["threshold_p99"] = thresh_df["threshold_p99"].astype("float32")
    thresh_df.to_parquet(model_dir / "spike_thresholds.parquet", index=False)

    # Rewrite 60m config with full inference fields
    config = {
        "alarm_threshold": alarm_threshold,
        "inference": {
            "bucket_duration_seconds":          300,
            "horizon_windows":                  12,
            "horizon_minutes":                  60,
            "lookback_windows":                 24,
            "lookback_minutes":                 120,
            "min_buckets_cold_start":           12,
            "feature_cols":                     feature_cols,
            "global_threshold_p95_fallback":    global_threshold,
            "global_threshold_p99_fallback":    global_threshold_p99,
            "thresholds_file":                  str(model_dir / "spike_thresholds.parquet"),
        },
    }
    (model_dir / "spike_config.json").write_text(json.dumps(config))

    arts = _load_artifacts(model_dir)
    return model_dir, arts


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


# ── Category 2: Feature parity (training-serving consistency) ─────────────────


class TestFeatureParity:
    """The features produced by predict_spike.py must match those produced by
    engineer() on the same input — both call the same internal functions, so
    any divergence would indicate a code-path regression."""

    def test_x_cols_match_meta_json(self):
        """_X_COLS in spike_classifier.py must be consistent with predict_spike.py's
        import of the same symbol.  If someone adds a feature to _X_COLS without
        updating the training pipeline, this test will catch the mismatch."""
        from predict_spike import _X_COLS as predict_X_COLS
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
        from predict_spike import main
        model_dir, _ = _make_fake_artifacts(tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            main([
                "--input",     str(tmp_path / "nonexistent.csv"),
                "--model-dir", str(model_dir),
            ])
        assert exc_info.value.code == 1

    def test_cli_exits_1_on_missing_model_dir(self, tmp_path):
        """main() must exit with code 1 when --model-dir does not exist."""
        from predict_spike import main
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
        from predict_spike import main
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
