"""Integration tests for analysis endpoints."""

import pytest
from fastapi.testclient import TestClient
import tempfile
from pathlib import Path

from obsidian.api.app import app
from obsidian.orchestration import SessionManager


@pytest.fixture(autouse=True)
def reset_session_manager():
    """Reset SessionManager before each test."""
    SessionManager.reset_instance()
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = SessionManager(storage_dir=Path(tmpdir) / "sessions")
        yield manager
    SessionManager.reset_instance()


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def fitted_session(client):
    """Create a fitted session with data for analysis."""
    # Create session
    session_config = {
        "name": "Analysis Test",
        "parameters": [
            {"type": "continuous", "name": "Temperature", "min": 0, "max": 100},
            {"type": "continuous", "name": "Pressure", "min": 1, "max": 10},
        ],
        "targets": [{"name": "Yield", "aim": "max"}],
        "seed": 42,
    }

    create_response = client.post("/api/v1/sessions", json=session_config)
    assert create_response.status_code == 201
    session_id = create_response.json()["session_id"]

    # Initialize
    init_response = client.post(
        f"/api/v1/sessions/{session_id}/initialize",
        json={"m_initial": 10, "method": "LHS"}
    )
    assert init_response.status_code == 200

    # Add data
    data_points = [
        {"Temperature": 10.0 + i * 10, "Pressure": 2.0 + i * 0.5, "Yield": 50.0 + i * 5.0}
        for i in range(10)
    ]
    data_response = client.post(
        f"/api/v1/sessions/{session_id}/data",
        json={"data": data_points}
    )
    assert data_response.status_code == 200

    # Fit model
    fit_response = client.post(
        f"/api/v1/sessions/{session_id}/fit",
        json={"verbose": 0}
    )
    assert fit_response.status_code == 200

    return session_id


def test_shap_analysis_basic(client, fitted_session):
    """Test basic SHAP analysis."""
    session_id = fitted_session

    # Request SHAP analysis
    response = client.post(
        f"/api/v1/sessions/{session_id}/analysis/shap",
        json={
            "target_name": "Yield",
            "n_samples": 50,
            "seed": 42
        }
    )

    assert response.status_code == 200
    data = response.json()

    # Validate response structure
    assert data["target_name"] == "Yield"
    assert "reference_point" in data
    assert "Temperature" in data["reference_point"]
    assert "Pressure" in data["reference_point"]

    assert "predicted_value" in data
    assert "expected_value" in data
    assert isinstance(data["predicted_value"], float)
    assert isinstance(data["expected_value"], float)

    # Check SHAP values
    assert "shap_values_mean" in data
    assert "Temperature" in data["shap_values_mean"]
    assert "Pressure" in data["shap_values_mean"]
    assert isinstance(data["shap_values_mean"]["Temperature"], float)

    # Check feature importance
    assert "feature_importance" in data
    assert len(data["feature_importance"]) == 2
    assert data["feature_importance"][0]["rank"] == 1
    assert data["feature_importance"][1]["rank"] == 2

    # Check samples
    assert "samples" in data
    assert len(data["samples"]) > 0
    assert len(data["samples"]) <= 50  # Limited to 50 or n_samples

    assert data["n_samples"] == 50


def test_shap_analysis_with_reference_point(client, fitted_session):
    """Test SHAP analysis with custom reference point."""
    session_id = fitted_session

    response = client.post(
        f"/api/v1/sessions/{session_id}/analysis/shap",
        json={
            "target_name": "Yield",
            "n_samples": 30,
            "reference_point": {"Temperature": 50.0, "Pressure": 5.0},
            "seed": 42
        }
    )

    assert response.status_code == 200
    data = response.json()

    # Verify reference point was used
    assert data["reference_point"]["Temperature"] == 50.0
    assert data["reference_point"]["Pressure"] == 5.0


def test_shap_analysis_invalid_target(client, fitted_session):
    """Test SHAP analysis with invalid target name."""
    session_id = fitted_session

    response = client.post(
        f"/api/v1/sessions/{session_id}/analysis/shap",
        json={
            "target_name": "NonexistentTarget",
            "n_samples": 50
        }
    )

    assert response.status_code == 400
    assert "not found" in response.json()["detail"].lower()


def test_shap_analysis_unfitted_model(client):
    """Test SHAP analysis on unfitted model."""
    # Create session without fitting
    session_config = {
        "parameters": [
            {"type": "continuous", "name": "Temperature", "min": 0, "max": 100},
        ],
        "targets": [{"name": "Yield", "aim": "max"}],
        "seed": 42,
    }

    create_response = client.post("/api/v1/sessions", json=session_config)
    session_id = create_response.json()["session_id"]

    # Try SHAP analysis
    response = client.post(
        f"/api/v1/sessions/{session_id}/analysis/shap",
        json={"target_name": "Yield"}
    )

    assert response.status_code == 400
    assert "fitted" in response.json()["detail"].lower()


def test_shap_analysis_session_not_found(client):
    """Test SHAP analysis with nonexistent session."""
    response = client.post(
        "/api/v1/sessions/nonexistent-id/analysis/shap",
        json={"target_name": "Yield"}
    )

    assert response.status_code == 404


def test_sensitivity_analysis_basic(client, fitted_session):
    """Test basic sensitivity analysis."""
    session_id = fitted_session

    response = client.post(
        f"/api/v1/sessions/{session_id}/analysis/sensitivity",
        json={}
    )

    assert response.status_code == 200
    data = response.json()

    # Validate response structure
    assert "reference_point" in data
    assert "Temperature" in data["reference_point"]
    assert "Pressure" in data["reference_point"]

    # Check predictions at reference
    assert "predictions_at_reference" in data
    assert "Yield" in data["predictions_at_reference"]
    assert isinstance(data["predictions_at_reference"]["Yield"], float)

    # Check sensitivity values
    assert "sensitivity" in data
    assert "Yield" in data["sensitivity"]
    assert "Temperature" in data["sensitivity"]["Yield"]
    assert "Pressure" in data["sensitivity"]["Yield"]
    assert isinstance(data["sensitivity"]["Yield"]["Temperature"], float)

    # Check normalized sensitivity
    assert "sensitivity_normalized" in data
    assert "Yield" in data["sensitivity_normalized"]
    assert 0 <= data["sensitivity_normalized"]["Yield"]["Temperature"] <= 1
    assert 0 <= data["sensitivity_normalized"]["Yield"]["Pressure"] <= 1

    # Check perturbation
    assert "perturbation" in data
    assert data["perturbation"] == 1e-6  # Default


def test_sensitivity_analysis_with_reference_point(client, fitted_session):
    """Test sensitivity analysis with custom reference point."""
    session_id = fitted_session

    response = client.post(
        f"/api/v1/sessions/{session_id}/analysis/sensitivity",
        json={
            "reference_point": {"Temperature": 60.0, "Pressure": 6.0},
            "perturbation": 1e-5
        }
    )

    assert response.status_code == 200
    data = response.json()

    # Verify reference point was used
    assert data["reference_point"]["Temperature"] == 60.0
    assert data["reference_point"]["Pressure"] == 6.0
    assert data["perturbation"] == 1e-5


def test_sensitivity_analysis_unfitted_model(client):
    """Test sensitivity analysis on unfitted model."""
    # Create session without fitting
    session_config = {
        "parameters": [
            {"type": "continuous", "name": "Temperature", "min": 0, "max": 100},
        ],
        "targets": [{"name": "Yield", "aim": "max"}],
        "seed": 42,
    }

    create_response = client.post("/api/v1/sessions", json=session_config)
    session_id = create_response.json()["session_id"]

    # Try sensitivity analysis
    response = client.post(
        f"/api/v1/sessions/{session_id}/analysis/sensitivity",
        json={}
    )

    assert response.status_code == 400
    assert "fitted" in response.json()["detail"].lower()


def test_sensitivity_analysis_session_not_found(client):
    """Test sensitivity analysis with nonexistent session."""
    response = client.post(
        "/api/v1/sessions/nonexistent-id/analysis/sensitivity",
        json={}
    )

    assert response.status_code == 404


def test_shap_and_sensitivity_consistency(client, fitted_session):
    """Test that SHAP and sensitivity provide consistent importance rankings."""
    session_id = fitted_session

    # Get SHAP analysis
    shap_response = client.post(
        f"/api/v1/sessions/{session_id}/analysis/shap",
        json={"target_name": "Yield", "n_samples": 50, "seed": 42}
    )
    assert shap_response.status_code == 200
    shap_data = shap_response.json()

    # Get sensitivity analysis
    sens_response = client.post(
        f"/api/v1/sessions/{session_id}/analysis/sensitivity",
        json={}
    )
    assert sens_response.status_code == 200
    sens_data = sens_response.json()

    # Both should identify same parameters
    shap_features = set(shap_data["shap_values_mean"].keys())
    sens_features = set(sens_data["sensitivity"]["Yield"].keys())
    assert shap_features == sens_features

    # Both should provide numeric values for all parameters
    for param in shap_features:
        assert isinstance(shap_data["shap_values_mean"][param], float)
        assert isinstance(sens_data["sensitivity"]["Yield"][param], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
