"""
Integration tests for OpenAI function calling with Obsidian API.

These tests require a running API server. They test:
1. Tool definitions load correctly
2. Tool executor can connect to API
3. Basic workflow functions work (create session, initialize)

Run with:
    # Terminal 1: Start API server
    uvicorn obsidian.api.app:app --reload

    # Terminal 2: Run integration tests
    pytest obsidian/api/llm/tests/test_integration.py -v
"""

import json
import pytest
import requests
from pathlib import Path

from obsidian.api.llm import ObsidianToolExecutor


def _check_api_server(base_url="http://localhost:8000"):
    """Check if API server is running."""
    try:
        response = requests.get(f"{base_url}/health", timeout=2)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


@pytest.fixture(scope="module")
def api_server():
    """
    Fixture that checks API server is running.

    Skips tests if server is not available.
    To run these tests, start the server first:
        uvicorn obsidian.api.app:app --reload
    """
    base_url = "http://localhost:8000"

    if not _check_api_server(base_url):
        pytest.skip("API server not running. Start with: uvicorn obsidian.api.app:app --reload")

    yield base_url

    # Cleanup could go here if needed


@pytest.fixture
def executor(api_server):
    """Create tool executor connected to API server."""
    return ObsidianToolExecutor(base_url=f"{api_server}/api/v1")


@pytest.fixture
def test_session(executor):
    """Create a test session and clean it up after test."""
    # Create session
    result = executor.execute_tool_call(
        "create_optimization_session",
        {
            "name": "LLM Integration Test",
            "parameters": [
                {"type": "continuous", "name": "Temperature", "min": 0, "max": 100},
                {"type": "continuous", "name": "Pressure", "min": 1, "max": 10},
            ],
            "targets": [{"name": "Yield", "aim": "max"}],
            "seed": 42,
        },
    )

    assert "session_id" in result, f"Failed to create session: {result}"
    session_id = result["session_id"]

    yield session_id

    # Cleanup
    executor.execute_tool_call("delete_optimization_session", {"session_id": session_id})


@pytest.mark.integration
def test_tool_definitions_load():
    """Test that tool definitions load correctly."""
    tools_path = Path(__file__).parent.parent / "openai_tools.json"

    with open(tools_path) as f:
        tools = json.load(f)

    assert len(tools) == 14, "Should have 14 tool definitions"

    # Verify structure
    assert all(t["type"] == "function" for t in tools)
    assert all("function" in t for t in tools)


@pytest.mark.integration
def test_executor_initialization(api_server):
    """Test executor initializes and can connect to API."""
    executor = ObsidianToolExecutor(base_url=f"{api_server}/api/v1")

    assert executor.base_url == f"{api_server}/api/v1"

    # Test connection with list_sessions
    result = executor.execute_tool_call("list_optimization_sessions", {})

    # Should get a list (not an error)
    assert isinstance(result, list), f"Expected list, got {type(result)}: {result}"


@pytest.mark.integration
def test_list_optimization_sessions(executor):
    """Test listing optimization sessions."""
    result = executor.execute_tool_call("list_optimization_sessions", {})

    # Should return a list
    assert isinstance(result, list)

    # Each session should have expected fields
    for session in result:
        assert "session_id" in session
        assert "name" in session
        assert "status" in session


@pytest.mark.integration
def test_create_optimization_session(executor):
    """Test creating an optimization session."""
    result = executor.execute_tool_call(
        "create_optimization_session",
        {
            "name": "Test Session",
            "parameters": [
                {"type": "continuous", "name": "Temperature", "min": 0, "max": 100},
            ],
            "targets": [{"name": "Yield", "aim": "max"}],
            "seed": 42,
        },
    )

    assert "session_id" in result
    session_id = result["session_id"]

    # Cleanup
    executor.execute_tool_call("delete_optimization_session", {"session_id": session_id})


@pytest.mark.integration
def test_initialize_experiments(executor, test_session):
    """Test initializing experiments."""
    result = executor.execute_tool_call(
        "initialize_experiments", {"session_id": test_session, "m_initial": 5, "method": "LHS"}
    )

    assert "suggestions" in result
    assert len(result["suggestions"]) == 5

    # Each suggestion should have parameters
    for suggestion in result["suggestions"]:
        assert "Temperature" in suggestion
        assert "Pressure" in suggestion


@pytest.mark.integration
def test_get_session_details(executor, test_session):
    """Test getting session details."""
    result = executor.execute_tool_call("get_session_details", {"session_id": test_session})

    assert "parameter_names" in result
    assert "target_names" in result
    assert set(result["parameter_names"]) == {"Temperature", "Pressure"}
    assert result["target_names"] == ["Yield"]


@pytest.mark.integration
def test_full_workflow(executor):
    """Test complete workflow: create -> initialize -> add data -> fit -> suggest -> delete."""
    # 1. Create session
    create_result = executor.execute_tool_call(
        "create_optimization_session",
        {
            "name": "Full Workflow Test",
            "parameters": [
                {"type": "continuous", "name": "x", "min": 0, "max": 10},
                {"type": "continuous", "name": "y", "min": 0, "max": 10},
            ],
            "targets": [{"name": "z", "aim": "max"}],
            "seed": 123,
        },
    )

    session_id = create_result["session_id"]

    try:
        # 2. Initialize
        init_result = executor.execute_tool_call(
            "initialize_experiments", {"session_id": session_id, "m_initial": 5, "method": "LHS"}
        )

        suggestions = init_result["suggestions"]
        assert len(suggestions) == 5

        # 3. Add data
        for i, s in enumerate(suggestions):
            s["z"] = 50.0 + i * 5.0

        add_result = executor.execute_tool_call(
            "add_experimental_data", {"session_id": session_id, "data": suggestions}
        )

        assert add_result["rows_added"] == 5

        # 4. Fit model
        fit_result = executor.execute_tool_call("fit_surrogate_model", {"session_id": session_id})

        assert "status" in fit_result
        assert fit_result["status"] == "fitted"

        # 5. Suggest next experiments
        suggest_result = executor.execute_tool_call(
            "suggest_next_experiments", {"session_id": session_id, "m_batch": 3, "acquisition": ["NEI"]}
        )

        assert "suggestions" in suggest_result
        assert len(suggest_result["suggestions"]) == 3

        # 6. Get best results
        best_result = executor.execute_tool_call("get_best_results", {"session_id": session_id})

        assert "X_best" in best_result
        assert "response_max" in best_result
        assert best_result["n_experiments"] == 5

    finally:
        # Cleanup
        executor.execute_tool_call("delete_optimization_session", {"session_id": session_id})


@pytest.mark.integration
def test_error_handling(executor):
    """Test that errors are handled gracefully."""
    # Try to get nonexistent session
    result = executor.execute_tool_call("get_session_details", {"session_id": "nonexistent-session-id"})

    # Should get error response
    assert "success" in result
    assert result["success"] is False
    assert "error" in result
    assert result["error"]["type"] == "HTTPError"
    assert result["error"]["http_status"] == 404
