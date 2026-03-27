"""Integration tests for the REST API."""

import pytest
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient

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
def sample_session_config():
    """Sample session configuration."""
    return {
        "name": "Test Optimization",
        "parameters": [
            {"type": "continuous", "name": "Temperature", "min": -10, "max": 30},
            {"type": "continuous", "name": "Concentration", "min": 10, "max": 150},
            {"type": "categorical", "name": "Variant", "categories": ["A", "B", "C"]},
        ],
        "targets": [{"name": "Yield", "aim": "max"}],
        "seed": 42,
    }


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_create_session(client, sample_session_config):
    """Test session creation."""
    response = client.post("/api/v1/sessions", json=sample_session_config)

    assert response.status_code == 201
    data = response.json()
    assert "session_id" in data
    assert data["name"] == "Test Optimization"
    assert data["status"] == "configured"
    assert data["n_parameters"] == 3
    assert data["n_targets"] == 1


def test_create_session_invalid_parameters(client):
    """Test session creation with invalid parameters."""
    invalid_config = {
        "name": "Invalid",
        "parameters": [],  # Empty parameters
        "targets": [{"name": "Yield", "aim": "max"}],
    }

    response = client.post("/api/v1/sessions", json=invalid_config)
    assert response.status_code == 422  # Validation error


def test_list_sessions(client, sample_session_config):
    """Test listing sessions."""
    # Create a session first
    create_response = client.post("/api/v1/sessions", json=sample_session_config)
    assert create_response.status_code == 201

    # List sessions
    response = client.get("/api/v1/sessions")
    assert response.status_code == 200
    sessions = response.json()
    assert isinstance(sessions, list)
    assert len(sessions) >= 1


def test_get_session(client, sample_session_config):
    """Test getting session details."""
    # Create session
    create_response = client.post("/api/v1/sessions", json=sample_session_config)
    session_id = create_response.json()["session_id"]

    # Get session
    response = client.get(f"/api/v1/sessions/{session_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert "parameter_names" in data
    assert "target_names" in data
    assert len(data["parameter_names"]) == 3
    assert data["target_names"] == ["Yield"]


def test_get_nonexistent_session(client):
    """Test getting non-existent session."""
    response = client.get("/api/v1/sessions/nonexistent-id")
    assert response.status_code == 404


def test_delete_session(client, sample_session_config):
    """Test session deletion."""
    # Create session
    create_response = client.post("/api/v1/sessions", json=sample_session_config)
    session_id = create_response.json()["session_id"]

    # Delete session
    response = client.delete(f"/api/v1/sessions/{session_id}")
    assert response.status_code == 204

    # Verify deletion
    get_response = client.get(f"/api/v1/sessions/{session_id}")
    assert get_response.status_code == 404


def test_initialize_session(client, sample_session_config):
    """Test initializing experiment design."""
    # Create session
    create_response = client.post("/api/v1/sessions", json=sample_session_config)
    session_id = create_response.json()["session_id"]

    # Initialize
    init_request = {"m_initial": 10, "method": "LHS"}
    response = client.post(f"/api/v1/sessions/{session_id}/initialize", json=init_request)

    assert response.status_code == 200
    data = response.json()
    assert data["n_experiments"] == 10
    assert data["method"] == "LHS"
    assert len(data["suggestions"]) == 10
    # Check that all parameters are present
    assert "Temperature" in data["suggestions"][0]
    assert "Concentration" in data["suggestions"][0]
    assert "Variant" in data["suggestions"][0]


def test_add_data(client, sample_session_config):
    """Test adding experimental data."""
    # Create and initialize session
    create_response = client.post("/api/v1/sessions", json=sample_session_config)
    session_id = create_response.json()["session_id"]

    init_response = client.post(f"/api/v1/sessions/{session_id}/initialize", json={"m_initial": 5, "method": "LHS"})
    suggestions = init_response.json()["suggestions"]

    # Add Yield values
    for s in suggestions:
        s["Yield"] = 85.0  # Fake yield

    # Add data
    response = client.post(f"/api/v1/sessions/{session_id}/data", json={"data": suggestions})

    assert response.status_code == 200
    data = response.json()
    assert data["rows_added"] == 5
    assert data["total_rows"] == 5


def test_fit_model(client, sample_session_config):
    """Test fitting surrogate model."""
    # Setup: create, initialize, add data
    create_response = client.post("/api/v1/sessions", json=sample_session_config)
    session_id = create_response.json()["session_id"]

    init_response = client.post(f"/api/v1/sessions/{session_id}/initialize", json={"m_initial": 5, "method": "LHS"})
    suggestions = init_response.json()["suggestions"]

    # Add fake yields
    for s in suggestions:
        s["Yield"] = 85.0

    client.post(f"/api/v1/sessions/{session_id}/data", json={"data": suggestions})

    # Fit
    response = client.post(f"/api/v1/sessions/{session_id}/fit")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "fitted"
    assert data["message"] == "Model fitted successfully"


def test_suggest_experiments(client, sample_session_config):
    """Test suggesting next experiments."""
    # Full setup: create, initialize, add data, fit
    create_response = client.post("/api/v1/sessions", json=sample_session_config)
    session_id = create_response.json()["session_id"]

    init_response = client.post(f"/api/v1/sessions/{session_id}/initialize", json={"m_initial": 5, "method": "LHS"})
    suggestions = init_response.json()["suggestions"]

    for s in suggestions:
        s["Yield"] = 85.0

    client.post(f"/api/v1/sessions/{session_id}/data", json={"data": suggestions})
    client.post(f"/api/v1/sessions/{session_id}/fit")

    # Suggest
    suggest_request = {"m_batch": 3, "acquisition": ["NEI"]}
    response = client.post(f"/api/v1/sessions/{session_id}/suggest", json=suggest_request)

    assert response.status_code == 200
    data = response.json()
    assert data["n_suggestions"] == 3
    assert len(data["suggestions"]) == 3
    # Check parameters are present
    assert "Temperature" in data["suggestions"][0]


def test_evaluate_points(client, sample_session_config):
    """Test evaluating arbitrary points."""
    # Full setup
    create_response = client.post("/api/v1/sessions", json=sample_session_config)
    session_id = create_response.json()["session_id"]

    init_response = client.post(f"/api/v1/sessions/{session_id}/initialize", json={"m_initial": 5, "method": "LHS"})
    suggestions = init_response.json()["suggestions"]

    for s in suggestions:
        s["Yield"] = 85.0

    client.post(f"/api/v1/sessions/{session_id}/data", json={"data": suggestions})
    client.post(f"/api/v1/sessions/{session_id}/fit")

    # Evaluate
    eval_request = {
        "X": [
            {"Temperature": 20, "Concentration": 100, "Variant": "A"},
            {"Temperature": 15, "Concentration": 120, "Variant": "B"},
        ]
    }
    response = client.post(f"/api/v1/sessions/{session_id}/evaluate", json=eval_request)

    assert response.status_code == 200
    data = response.json()
    assert data["n_points"] == 2
    assert len(data["predictions"]) == 2


def test_get_best_results(client, sample_session_config):
    """Test getting best results."""
    # Setup with data
    create_response = client.post("/api/v1/sessions", json=sample_session_config)
    session_id = create_response.json()["session_id"]

    # Before any data
    response = client.get(f"/api/v1/sessions/{session_id}/best")
    assert response.status_code == 200
    data = response.json()
    assert data["X_best"] is None

    # After adding data
    init_response = client.post(f"/api/v1/sessions/{session_id}/initialize", json={"m_initial": 5, "method": "LHS"})
    suggestions = init_response.json()["suggestions"]

    for i, s in enumerate(suggestions):
        s["Yield"] = 80.0 + i  # Varying yields

    client.post(f"/api/v1/sessions/{session_id}/data", json={"data": suggestions})

    response = client.get(f"/api/v1/sessions/{session_id}/best")
    assert response.status_code == 200
    data = response.json()
    assert data["X_best"] is not None
    assert data["n_experiments"] == 5

    # response_max is a dict for single or multiple targets
    assert isinstance(data["response_max"], dict)
    assert "Yield" in data["response_max"]
    assert data["response_max"]["Yield"] == 84.0  # Max of 80, 81, 82, 83, 84


def test_full_workflow(client, sample_session_config):
    """Test complete optimization workflow."""
    # 1. Create session
    create_response = client.post("/api/v1/sessions", json=sample_session_config)
    assert create_response.status_code == 201
    session_id = create_response.json()["session_id"]

    # 2. Initialize
    init_response = client.post(f"/api/v1/sessions/{session_id}/initialize", json={"m_initial": 8, "method": "LHS"})
    assert init_response.status_code == 200
    suggestions = init_response.json()["suggestions"]

    # 3. Add data
    for s in suggestions:
        s["Yield"] = 85.0

    data_response = client.post(f"/api/v1/sessions/{session_id}/data", json={"data": suggestions})
    assert data_response.status_code == 200

    # 4. Fit
    fit_response = client.post(f"/api/v1/sessions/{session_id}/fit")
    assert fit_response.status_code == 200

    # 5. Suggest
    suggest_response = client.post(
        f"/api/v1/sessions/{session_id}/suggest", json={"m_batch": 2, "acquisition": ["NEI"]}
    )
    assert suggest_response.status_code == 200
    new_suggestions = suggest_response.json()["suggestions"]

    # 6. Add second batch
    for s in new_suggestions:
        s["Yield"] = 90.0

    data_response2 = client.post(f"/api/v1/sessions/{session_id}/data", json={"data": new_suggestions})
    assert data_response2.status_code == 200
    assert data_response2.json()["total_rows"] == 10

    # 7. Refit
    fit_response2 = client.post(f"/api/v1/sessions/{session_id}/fit")
    assert fit_response2.status_code == 200

    # 8. Get best
    best_response = client.get(f"/api/v1/sessions/{session_id}/best")
    assert best_response.status_code == 200
    best = best_response.json()
    assert best["n_experiments"] == 10

    # 9. Delete
    delete_response = client.delete(f"/api/v1/sessions/{session_id}")
    assert delete_response.status_code == 204


# ============================================================================
# New Endpoint Tests: Data, Diagnostics, History, Evaluate with Std
# ============================================================================


def test_get_campaign_data_empty(client, sample_session_config):
    """Test getting data from session with no experiments."""
    # Create session
    response = client.post("/api/v1/sessions", json=sample_session_config)
    session_id = response.json()["session_id"]

    # Get data (should be empty)
    response = client.get(f"/api/v1/sessions/{session_id}/data")
    assert response.status_code == 200
    data = response.json()
    assert data["n_rows"] == 0
    assert data["data"] == []
    assert "metadata" in data
    assert data["metadata"]["n_parameters"] == 3
    assert data["metadata"]["n_targets"] == 1


def test_get_campaign_data_with_experiments(client, sample_session_config):
    """Test getting data after adding experiments."""
    # Setup: create, initialize, add data
    response = client.post("/api/v1/sessions", json=sample_session_config)
    session_id = response.json()["session_id"]

    init_response = client.post(f"/api/v1/sessions/{session_id}/initialize", json={"m_initial": 5})
    suggestions = init_response.json()["suggestions"]
    for s in suggestions:
        s["Yield"] = 85.0

    client.post(f"/api/v1/sessions/{session_id}/data", json={"data": suggestions})

    # Get data
    response = client.get(f"/api/v1/sessions/{session_id}/data")
    assert response.status_code == 200
    data = response.json()
    assert data["n_rows"] == 5
    assert len(data["data"]) == 5
    assert "Temperature" in data["data"][0]
    assert "Yield" in data["data"][0]
    assert "Iteration" in data["columns"]
    assert 0 in data["iterations"]


def test_diagnostics_unfitted(client, sample_session_config):
    """Test diagnostics before fitting model."""
    response = client.post("/api/v1/sessions", json=sample_session_config)
    session_id = response.json()["session_id"]

    response = client.get(f"/api/v1/sessions/{session_id}/diagnostics")
    assert response.status_code == 200
    data = response.json()
    assert data["is_fitted"] is False
    assert data["n_experiments"] == 0
    assert len(data["surrogates"]) == 0
    assert data["is_multi_objective"] is False


def test_diagnostics_fitted(client, sample_session_config):
    """Test diagnostics after fitting model."""
    # Setup: create, init, add data, fit
    response = client.post("/api/v1/sessions", json=sample_session_config)
    session_id = response.json()["session_id"]

    init_response = client.post(f"/api/v1/sessions/{session_id}/initialize", json={"m_initial": 10})
    suggestions = init_response.json()["suggestions"]
    # Use varying yields to get valid R² score
    for i, s in enumerate(suggestions):
        s["Yield"] = 80.0 + i

    client.post(f"/api/v1/sessions/{session_id}/data", json={"data": suggestions})
    client.post(f"/api/v1/sessions/{session_id}/fit")

    # Get diagnostics
    response = client.get(f"/api/v1/sessions/{session_id}/diagnostics")
    assert response.status_code == 200
    data = response.json()

    assert data["is_fitted"] is True
    assert data["n_experiments"] == 10
    assert data["n_training_points"] == 10
    assert len(data["surrogates"]) == 1  # Single target

    surr = data["surrogates"][0]
    assert surr["response_name"] == "Yield"
    assert surr["model_type"] in ["GP", "MixedGP"]
    # r2_score and loss may be None if data is constant or NaN, but should exist
    assert "r2_score" in surr
    assert "loss" in surr
    assert surr["is_fit"] is True

    assert data["current_iteration"] == 0
    assert data["n_iterations"] == 1


def test_history_empty(client, sample_session_config):
    """Test history endpoint with no experiments."""
    response = client.post("/api/v1/sessions", json=sample_session_config)
    session_id = response.json()["session_id"]

    response = client.get(f"/api/v1/sessions/{session_id}/history")
    assert response.status_code == 200
    data = response.json()
    assert data["n_iterations"] == 0
    assert data["iterations"] == []
    assert data["total_experiments"] == 0


def test_history_single_iteration(client, sample_session_config):
    """Test history with single iteration."""
    response = client.post("/api/v1/sessions", json=sample_session_config)
    session_id = response.json()["session_id"]

    # Initialize and add data
    init_response = client.post(f"/api/v1/sessions/{session_id}/initialize", json={"m_initial": 5})
    suggestions = init_response.json()["suggestions"]
    for i, s in enumerate(suggestions):
        s["Yield"] = 80.0 + i  # Varying yields: 80, 81, 82, 83, 84

    client.post(f"/api/v1/sessions/{session_id}/data", json={"data": suggestions})

    # Get history
    response = client.get(f"/api/v1/sessions/{session_id}/history")
    assert response.status_code == 200
    data = response.json()

    assert data["n_iterations"] == 1
    assert len(data["iterations"]) == 1
    assert data["total_experiments"] == 5

    iter0 = data["iterations"][0]
    assert iter0["iteration"] == 0
    assert iter0["n_experiments"] == 5
    assert "Yield" in iter0["best_response"]
    assert iter0["best_response"]["Yield"] == 84.0  # Max of 80-84
    assert "Yield" in iter0["mean_response"]
    assert iter0["mean_response"]["Yield"] == 82.0  # Mean of 80-84


def test_history_multiple_iterations(client, sample_session_config):
    """Test history with multiple optimization iterations."""
    response = client.post("/api/v1/sessions", json=sample_session_config)
    session_id = response.json()["session_id"]

    # Iteration 0: Initial design
    init_response = client.post(f"/api/v1/sessions/{session_id}/initialize", json={"m_initial": 5})
    suggestions = init_response.json()["suggestions"]
    for s in suggestions:
        s["Yield"] = 80.0
    client.post(f"/api/v1/sessions/{session_id}/data", json={"data": suggestions})

    # Fit and suggest iteration 1
    client.post(f"/api/v1/sessions/{session_id}/fit")
    suggest_response = client.post(f"/api/v1/sessions/{session_id}/suggest", json={"m_batch": 3})
    new_suggestions = suggest_response.json()["suggestions"]
    for s in new_suggestions:
        s["Yield"] = 90.0  # Better yield
    client.post(f"/api/v1/sessions/{session_id}/data", json={"data": new_suggestions})

    # Get history
    response = client.get(f"/api/v1/sessions/{session_id}/history")
    assert response.status_code == 200
    data = response.json()

    assert data["n_iterations"] == 2
    assert len(data["iterations"]) == 2
    assert data["total_experiments"] == 8

    # Check iteration 0
    iter0 = data["iterations"][0]
    assert iter0["iteration"] == 0
    assert iter0["n_experiments"] == 5
    assert iter0["best_response"]["Yield"] == 80.0

    # Check iteration 1 - should show improvement
    iter1 = data["iterations"][1]
    assert iter1["iteration"] == 1
    assert iter1["n_experiments"] == 3
    assert iter1["best_response"]["Yield"] == 90.0  # Cumulative best
    assert iter1["mean_response"]["Yield"] == 90.0  # Mean of iteration 1


def test_evaluate_with_std(client, sample_session_config):
    """Test evaluating with standard deviation."""
    # Full setup
    response = client.post("/api/v1/sessions", json=sample_session_config)
    session_id = response.json()["session_id"]

    # Initialize, add data, fit
    init_response = client.post(f"/api/v1/sessions/{session_id}/initialize", json={"m_initial": 8})
    suggestions = init_response.json()["suggestions"]
    for s in suggestions:
        s["Yield"] = 85.0

    client.post(f"/api/v1/sessions/{session_id}/data", json={"data": suggestions})
    client.post(f"/api/v1/sessions/{session_id}/fit")

    # Evaluate with std
    eval_request = {"X": [{"Temperature": 20, "Concentration": 100, "Variant": "A"}], "return_std": True}
    response = client.post(f"/api/v1/sessions/{session_id}/evaluate", json=eval_request)

    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 1
    assert "Yield (pred)" in data["predictions"][0]
    assert "Yield (std)" in data["predictions"][0]
    # Should NOT have lb/ub
    assert "Yield lb" not in data["predictions"][0]


def test_evaluate_mean_only(client, sample_session_config):
    """Test that default behavior returns mean only."""
    # Setup
    response = client.post("/api/v1/sessions", json=sample_session_config)
    session_id = response.json()["session_id"]

    init_response = client.post(f"/api/v1/sessions/{session_id}/initialize", json={"m_initial": 8})
    suggestions = init_response.json()["suggestions"]
    for s in suggestions:
        s["Yield"] = 85.0

    client.post(f"/api/v1/sessions/{session_id}/data", json={"data": suggestions})
    client.post(f"/api/v1/sessions/{session_id}/fit")

    # Evaluate WITHOUT return_std (should default to False - mean only)
    eval_request = {
        "X": [{"Temperature": 20, "Concentration": 100, "Variant": "A"}]
        # return_std defaults to False
    }
    response = client.post(f"/api/v1/sessions/{session_id}/evaluate", json=eval_request)

    assert response.status_code == 200
    data = response.json()
    # Should have ONLY pred (mean), no intervals or std
    assert "Yield (pred)" in data["predictions"][0]
    assert "Yield lb" not in data["predictions"][0]
    assert "Yield ub" not in data["predictions"][0]
    assert "Yield (std)" not in data["predictions"][0]


def test_get_state_dict_campaign(client, sample_session_config):
    """Test getting campaign state dictionary."""
    response = client.post("/api/v1/sessions", json=sample_session_config)
    session_id = response.json()["session_id"]

    # Initialize and add some data
    init_response = client.post(f"/api/v1/sessions/{session_id}/initialize", json={"m_initial": 5})
    suggestions = init_response.json()["suggestions"]
    for s in suggestions:
        s["Yield"] = 85.0
    client.post(f"/api/v1/sessions/{session_id}/data", json={"data": suggestions})

    # Get campaign state dict
    response = client.get(f"/api/v1/sessions/{session_id}/state_dict?object=campaign")
    assert response.status_code == 200
    data = response.json()
    assert data["object_type"] == "campaign"
    assert "state" in data
    # Campaign state should include optimizer, RNG, data, etc.
    assert "optimizer" in data["state"]
    assert "X_space" in data["state"]
    assert "data" in data["state"]


def test_get_state_dict_optimizer(client, sample_session_config):
    """Test getting optimizer state dictionary."""
    response = client.post("/api/v1/sessions", json=sample_session_config)
    session_id = response.json()["session_id"]

    # Initialize, add data, and fit
    init_response = client.post(f"/api/v1/sessions/{session_id}/initialize", json={"m_initial": 5})
    suggestions = init_response.json()["suggestions"]
    for i, s in enumerate(suggestions):
        s["Yield"] = 80.0 + i
    client.post(f"/api/v1/sessions/{session_id}/data", json={"data": suggestions})
    client.post(f"/api/v1/sessions/{session_id}/fit")

    # Get optimizer state dict
    response = client.get(f"/api/v1/sessions/{session_id}/state_dict?object=optimizer")
    assert response.status_code == 200
    data = response.json()
    assert data["object_type"] == "optimizer"
    assert "state" in data
    # Optimizer state should include model states, fit status, X_space
    assert "model_states" in data["state"]
    assert "is_fitted" in data["state"]
    assert data["state"]["is_fitted"] is True


def test_get_state_dict_invalid_object(client, sample_session_config):
    """Test getting state dict with invalid object type."""
    response = client.post("/api/v1/sessions", json=sample_session_config)
    session_id = response.json()["session_id"]

    # Try invalid object type
    response = client.get(f"/api/v1/sessions/{session_id}/state_dict?object=invalid")
    assert response.status_code == 400


def test_full_llm_workflow(client, sample_session_config):
    """Test complete LLM exploration workflow using all new endpoints."""
    # 1. Create session
    response = client.post("/api/v1/sessions", json=sample_session_config)
    assert response.status_code == 201
    session_id = response.json()["session_id"]

    # 2. Initialize
    init_response = client.post(f"/api/v1/sessions/{session_id}/initialize", json={"m_initial": 8})
    assert init_response.status_code == 200
    suggestions = init_response.json()["suggestions"]

    # 3. Add data
    for i, s in enumerate(suggestions):
        s["Yield"] = 80.0 + i

    data_response = client.post(f"/api/v1/sessions/{session_id}/data", json={"data": suggestions})
    assert data_response.status_code == 200

    # 4. Fit
    fit_response = client.post(f"/api/v1/sessions/{session_id}/fit")
    assert fit_response.status_code == 200

    # 5. GET /diagnostics → Check R² scores, verify model quality
    diag_response = client.get(f"/api/v1/sessions/{session_id}/diagnostics")
    assert diag_response.status_code == 200
    diag = diag_response.json()
    assert diag["is_fitted"] is True
    assert diag["n_experiments"] == 8
    assert len(diag["surrogates"]) == 1
    assert diag["surrogates"][0]["r2_score"] is not None

    # 6. GET /data → Analyze all experiments
    data_response = client.get(f"/api/v1/sessions/{session_id}/data")
    assert data_response.status_code == 200
    exp_data = data_response.json()
    assert exp_data["n_rows"] == 8
    assert len(exp_data["data"]) == 8

    # 7. POST /evaluate (with return_std=False) → Get mean predictions
    eval_response = client.post(
        f"/api/v1/sessions/{session_id}/evaluate",
        json={"X": [{"Temperature": 20, "Concentration": 100, "Variant": "A"}], "return_std": False},
    )
    assert eval_response.status_code == 200
    eval_data = eval_response.json()
    assert "Yield (pred)" in eval_data["predictions"][0]
    assert "Yield (std)" not in eval_data["predictions"][0]

    # 7b. POST /evaluate (with return_std=True) → Explore uncertainty
    eval_response_std = client.post(
        f"/api/v1/sessions/{session_id}/evaluate",
        json={"X": [{"Temperature": 20, "Concentration": 100, "Variant": "A"}], "return_std": True},
    )
    assert eval_response_std.status_code == 200
    eval_data_std = eval_response_std.json()
    assert "Yield (pred)" in eval_data_std["predictions"][0]
    assert "Yield (std)" in eval_data_std["predictions"][0]

    # 8. POST /suggest → Get next experiments
    suggest_response = client.post(f"/api/v1/sessions/{session_id}/suggest", json={"m_batch": 2})
    assert suggest_response.status_code == 200
    new_suggestions = suggest_response.json()["suggestions"]

    # 9. Add second batch
    for s in new_suggestions:
        s["Yield"] = 95.0  # Even better

    data_response2 = client.post(f"/api/v1/sessions/{session_id}/data", json={"data": new_suggestions})
    assert data_response2.status_code == 200
    assert data_response2.json()["total_rows"] == 10

    # 10. Refit
    fit_response2 = client.post(f"/api/v1/sessions/{session_id}/fit")
    assert fit_response2.status_code == 200

    # 11. GET /history → Analyze iteration-by-iteration improvement
    hist_response = client.get(f"/api/v1/sessions/{session_id}/history")
    assert hist_response.status_code == 200
    hist_data = hist_response.json()
    assert hist_data["n_iterations"] == 2
    assert hist_data["iterations"][0]["best_response"]["Yield"] == 87.0  # Max from iteration 0
    assert hist_data["iterations"][1]["best_response"]["Yield"] == 95.0  # Max overall

    # 12. GET /diagnostics → Check final model quality
    final_diag_response = client.get(f"/api/v1/sessions/{session_id}/diagnostics")
    assert final_diag_response.status_code == 200
    final_diag = final_diag_response.json()
    assert final_diag["n_experiments"] == 10
    assert final_diag["current_iteration"] == 1


def test_suggest_with_different_manual_seeds(client, sample_session_config):
    """Test that different manual_seed values produce different suggestions."""
    # Setup: create session, initialize, add data, fit model
    create_response = client.post("/api/v1/sessions", json=sample_session_config)
    session_id = create_response.json()["session_id"]

    # Initialize
    init_response = client.post(f"/api/v1/sessions/{session_id}/initialize", json={"m_initial": 8, "method": "LHS"})
    suggestions = init_response.json()["suggestions"]

    # Add data with varying yields
    for i, s in enumerate(suggestions):
        s["Yield"] = 75.0 + i * 2.5

    client.post(f"/api/v1/sessions/{session_id}/data", json={"data": suggestions})

    # Fit model
    fit_response = client.post(f"/api/v1/sessions/{session_id}/fit")
    assert fit_response.status_code == 200

    # Suggest with manual_seed=42
    suggest_request_1 = {"m_batch": 3, "acquisition": ["NEI"], "manual_seed": 42}
    response_1 = client.post(f"/api/v1/sessions/{session_id}/suggest", json=suggest_request_1)
    assert response_1.status_code == 200
    suggestions_1 = response_1.json()["suggestions"]
    assert len(suggestions_1) == 3

    # Suggest with manual_seed=123 (different seed)
    suggest_request_2 = {"m_batch": 3, "acquisition": ["NEI"], "manual_seed": 123}
    response_2 = client.post(f"/api/v1/sessions/{session_id}/suggest", json=suggest_request_2)
    assert response_2.status_code == 200
    suggestions_2 = response_2.json()["suggestions"]
    assert len(suggestions_2) == 3

    # Verify suggestions are different
    # Check that at least one suggestion differs between the two calls
    differences_found = False
    for s1, s2 in zip(suggestions_1, suggestions_2):
        # Compare parameter values
        if (
            abs(s1.get("Temperature", 0) - s2.get("Temperature", 0)) > 0.01
            or abs(s1.get("Concentration", 0) - s2.get("Concentration", 0)) > 0.01
            or s1.get("Variant") != s2.get("Variant")
        ):
            differences_found = True
            break

    assert differences_found, "Different manual_seed values should produce different suggestions"

    # Verify same seed produces same suggestions (reproducibility)
    suggest_request_3 = {"m_batch": 3, "acquisition": ["NEI"], "manual_seed": 42}  # Same as first call
    response_3 = client.post(f"/api/v1/sessions/{session_id}/suggest", json=suggest_request_3)
    assert response_3.status_code == 200
    suggestions_3 = response_3.json()["suggestions"]

    # Verify suggestions_1 and suggestions_3 are identical (same seed)
    for s1, s3 in zip(suggestions_1, suggestions_3):
        assert abs(s1.get("Temperature", 0) - s3.get("Temperature", 0)) < 0.01
        assert abs(s1.get("Concentration", 0) - s3.get("Concentration", 0)) < 0.01
        assert s1.get("Variant") == s3.get("Variant")
