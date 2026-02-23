"""
Shared fixtures and pre-test setup for the Energy & Carbon PoC test suite.

Execution order in pytest:
  1. This conftest.py is imported (module-level code runs).
  2. DATA_PATH env var is set to a temp CSV BEFORE api.py is ever imported.
  3. test_*.py files are imported and their tests collected.

This ordering guarantees that `api.py`'s module-level DATA_PATH assignment
picks up our small 30-day CSV instead of the full 8,760-row production file.

Run all tests inside Docker:
    docker compose --profile test run --rm test

Run a specific test file:
    docker compose --profile test run --rm test pytest tests/test_data.py -v
"""
import os
import sys
import tempfile

import pandas as pd
import pytest

# ── Path setup ─────────────────────────────────────────────────────────────────
# Allow imports from /app (where all modules live in Docker).
# When running locally from AIModel/, the cwd is already there.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Pre-generate small CSV before any module imports DATA_PATH ─────────────────
# api.py reads `DATA_PATH = os.environ.get("DATA_PATH", "...")` at import time.
# We must set the env var HERE, at conftest load time, before test_api.py
# imports `app` from api.py.
from data_generator import generate_energy_carbon_data  # noqa: E402

_tmp_dir = tempfile.mkdtemp()
_TEST_CSV = os.path.join(_tmp_dir, "test_energy_data.csv")

# 30-day dataset (720 rows) — fast enough for Prophet fitting (~30-50s total)
_base_df = generate_energy_carbon_data(start_date="2024-01-01", periods=24 * 30)
_base_df.to_csv(_TEST_CSV, index=False)

# Point the API at our small CSV
os.environ["DATA_PATH"] = _TEST_CSV


# ── Session-scoped fixtures ────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def small_df() -> pd.DataFrame:
    """30-day hourly dataset (720 rows). Re-used across all tests for speed."""
    return _base_df.copy()


@pytest.fixture(scope="session")
def test_csv_path() -> str:
    """Path to the small CSV written at test startup."""
    return _TEST_CSV
