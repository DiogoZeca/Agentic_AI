import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["DATA_PATH"] = "data/cpu_data.dat"

from api import app  # noqa: E402 — must import after env var is set


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c
