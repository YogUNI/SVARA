"""API integration tests using FastAPI TestClient (Task P8-04).

Reference: docs/05 §8
"""

import io
import json
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from svara.serve.app import app
import svara.serve.app as serve_app
from svara.serve.device_state import get_initial_home_state


client = TestClient(app)


def test_api_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_api_devices_get_and_reset():
    # GET initial state
    res = client.get("/api/devices")
    assert res.status_code == 200
    assert "rooms" in res.json()

    # Reset
    res_reset = client.post("/api/devices/reset")
    assert res_reset.status_code == 200
    assert res_reset.json()["status"] == "reset"


def test_api_results_endpoint():
    res = client.get("/api/results")
    assert res.status_code == 200
    # Either valid data dict or empty-state message dict


def test_api_predict_rejection_path_mock():
    """Test /api/predict rejects unreadable or malformed audio correctly."""
    # 1. Non-audio empty file
    res = client.post(
        "/api/predict",
        files={"audio": ("empty.txt", b"", "text/plain")},
    )
    # Should return 422 Unprocessable Entity
    assert res.status_code in (422, 503)
