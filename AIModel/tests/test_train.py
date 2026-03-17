"""Smoke tests for train.py — verifies artefact structure, not numerical values."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def trained_models(tmp_path_factory):
    """Run train.py --fast and return the output directory."""
    out = tmp_path_factory.mktemp("models")
    result = subprocess.run(
        [sys.executable, "train.py", "--data", "data/cpu_data.dat", "--out", str(out), "--fast"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"train.py exited with code {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return out


class TestArtefactStructure:
    def test_winner_json_exists(self, trained_models):
        assert (trained_models / "winner.json").exists()

    def test_metadata_json_exists(self, trained_models):
        assert (trained_models / "metadata.json").exists()

    def test_xgb_artefacts_exist(self, trained_models):
        assert (trained_models / "xgb" / "model.ubj").exists()
        assert (trained_models / "xgb" / "categories.json").exists()

    def test_mlp_artefacts_exist(self, trained_models):
        assert (trained_models / "mlp" / "model.pt").exists()
        assert (trained_models / "mlp" / "scaler.pkl").exists()
        assert (trained_models / "mlp" / "encoder.pkl").exists()
        assert (trained_models / "mlp" / "config.json").exists()


class TestWinnerManifest:
    @pytest.fixture(scope="class")
    def manifest(self, trained_models):
        with open(trained_models / "winner.json") as f:
            return json.load(f)

    def test_required_keys(self, manifest):
        required = {
            "model", "dir", "run_ts", "trained_at", "duration_s",
            "data_path", "data_md5", "fast_mode", "metrics",
            "all_models", "library_versions",
        }
        assert required == set(manifest.keys())

    def test_model_is_valid(self, manifest):
        assert manifest["model"] in ("xgboost", "mlp")

    def test_fast_mode_recorded(self, manifest):
        assert manifest["fast_mode"] is True

    def test_metrics_keys(self, manifest):
        assert set(manifest["metrics"].keys()) == {
            "interpolation_rmse_w",
            "interpolation_mae_w",
            "interpolation_spike_f1",
        }

    def test_rmse_is_positive(self, manifest):
        assert manifest["metrics"]["interpolation_rmse_w"] > 0

    def test_spike_f1_in_unit_interval(self, manifest):
        assert 0.0 <= manifest["metrics"]["interpolation_spike_f1"] <= 1.0

    def test_all_models_has_both(self, manifest):
        assert set(manifest["all_models"].keys()) == {"xgboost", "mlp"}

    def test_library_versions_present(self, manifest):
        assert "xgboost" in manifest["library_versions"]
        assert "torch" in manifest["library_versions"]

    def test_dir_is_relative(self, manifest):
        # Stored path must not be absolute — would break inside Docker
        assert not Path(manifest["dir"]).is_absolute(), (
            f"'dir' must be a relative path, got: {manifest['dir']}"
        )

    def test_winner_dir_matches_model(self, manifest):
        expected_subdir = "xgb" if manifest["model"] == "xgboost" else "mlp"
        assert manifest["dir"].endswith(expected_subdir), (
            f"'dir' should end with '{expected_subdir}', got: {manifest['dir']}"
        )


class TestMetadata:
    @pytest.fixture(scope="class")
    def metadata(self, trained_models):
        with open(trained_models / "metadata.json") as f:
            return json.load(f)

    def test_has_11_cpu_types(self, metadata):
        assert len(metadata) == 11

    def test_each_type_has_required_keys(self, metadata):
        required = {"idle_w", "full_w", "dynamic_range_w", "spike_threshold_w", "mean_std_w"}
        for cpu_type, meta in metadata.items():
            assert set(meta.keys()) == required, f"{cpu_type} missing keys"

    def test_full_w_gt_idle_w(self, metadata):
        for cpu_type, meta in metadata.items():
            assert meta["full_w"] > meta["idle_w"], f"{cpu_type}: full_w <= idle_w"


class TestCli:
    def test_missing_data_exits_nonzero(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "train.py", "--data", "nonexistent.dat", "--out", str(tmp_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "train.py", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--fast" in result.stdout
