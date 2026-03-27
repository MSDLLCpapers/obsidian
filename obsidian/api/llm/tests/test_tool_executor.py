"""
Unit tests for ObsidianToolExecutor.

Tests the HTTP client wrapper for executing function calls.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
import requests

from obsidian.api.llm.tool_executor import ObsidianToolExecutor


def test_executor_init_default_url():
    """Test executor initializes with default URL."""
    executor = ObsidianToolExecutor()
    assert executor.base_url == "http://localhost:8000/api/v1"


def test_executor_init_custom_url():
    """Test executor initializes with custom URL."""
    custom_url = "http://custom-api:9000/api/v1"
    executor = ObsidianToolExecutor(base_url=custom_url)
    assert executor.base_url == custom_url


def test_executor_init_env_var():
    """Test executor uses OBSIDIAN_API_URL environment variable."""
    env_url = "http://env-api:8080/api/v1"

    with patch.dict(os.environ, {"OBSIDIAN_API_URL": env_url}):
        executor = ObsidianToolExecutor()
        assert executor.base_url == env_url


def test_executor_init_custom_overrides_env():
    """Test custom URL overrides environment variable."""
    env_url = "http://env-api:8080/api/v1"
    custom_url = "http://custom-api:9000/api/v1"

    with patch.dict(os.environ, {"OBSIDIAN_API_URL": env_url}):
        executor = ObsidianToolExecutor(base_url=custom_url)
        assert executor.base_url == custom_url


def test_unknown_function():
    """Test executor returns error for unknown function."""
    executor = ObsidianToolExecutor()

    result = executor.execute_tool_call("unknown_function", {})

    assert result["success"] is False
    assert "error" in result
    assert result["error"]["type"] == "UnknownFunction"


def test_endpoint_mapping():
    """Test all 14 functions have endpoint mappings."""
    executor = ObsidianToolExecutor()

    # Access the endpoint_map from execute_tool_call (it's defined inline)
    # We'll just verify functions don't return UnknownFunction error
    expected_functions = [
        "create_optimization_session",
        "list_optimization_sessions",
        "get_session_details",
        "delete_optimization_session",
        "initialize_experiments",
        "add_experimental_data",
        "fit_surrogate_model",
        "suggest_next_experiments",
        "evaluate_predictions",
        "get_best_results",
        "get_campaign_data",
        "get_model_diagnostics",
        "get_optimization_history",
        "export_state_dictionary",
    ]

    with patch.object(executor.session, "get") as mock_get, patch.object(
        executor.session, "post"
    ) as mock_post, patch.object(executor.session, "delete") as mock_delete:

        # Mock successful responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_get.return_value = mock_response
        mock_post.return_value = mock_response
        mock_delete.return_value = mock_response

        for func_name in expected_functions:
            result = executor.execute_tool_call(func_name, {})
            # Should not get UnknownFunction error
            if "error" in result:
                assert result["error"]["type"] != "UnknownFunction", f"{func_name} not mapped"


@patch("obsidian.api.llm.tool_executor.requests.Session")
def test_execute_post_request(mock_session_class):
    """Test POST request execution."""
    mock_session = MagicMock()
    mock_response = Mock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"session_id": "test123"}
    mock_session.post.return_value = mock_response
    mock_session_class.return_value = mock_session

    executor = ObsidianToolExecutor()

    result = executor.execute_tool_call(
        "create_optimization_session",
        {
            "parameters": [{"type": "continuous", "name": "temp", "min": 0, "max": 100}],
            "targets": [{"name": "yield", "aim": "max"}],
        },
    )

    # Verify POST was called
    assert mock_session.post.called
    assert result["session_id"] == "test123"


@patch("obsidian.api.llm.tool_executor.requests.Session")
def test_execute_get_request(mock_session_class):
    """Test GET request execution."""
    mock_session = MagicMock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"session_id": "test123"}]
    mock_session.get.return_value = mock_response
    mock_session_class.return_value = mock_session

    executor = ObsidianToolExecutor()

    result = executor.execute_tool_call("list_optimization_sessions", {})

    # Verify GET was called
    assert mock_session.get.called
    assert isinstance(result, list)


@patch("obsidian.api.llm.tool_executor.requests.Session")
def test_execute_delete_request(mock_session_class):
    """Test DELETE request execution."""
    mock_session = MagicMock()
    mock_response = Mock()
    mock_response.status_code = 204  # No content
    mock_session.delete.return_value = mock_response
    mock_session_class.return_value = mock_session

    executor = ObsidianToolExecutor()

    result = executor.execute_tool_call("delete_optimization_session", {"session_id": "test123"})

    # Verify DELETE was called
    assert mock_session.delete.called
    assert result["success"] is True


@patch("obsidian.api.llm.tool_executor.requests.Session")
def test_session_id_formatting(mock_session_class):
    """Test that session_id is properly inserted into endpoint URL."""
    mock_session = MagicMock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"X_best": {}}
    mock_session.get.return_value = mock_response
    mock_session_class.return_value = mock_session

    executor = ObsidianToolExecutor()

    executor.execute_tool_call("get_best_results", {"session_id": "abc123"})

    # Verify URL contains session_id
    call_args = mock_session.get.call_args
    assert "abc123" in call_args[0][0]


@patch("obsidian.api.llm.tool_executor.requests.Session")
def test_http_error_handling(mock_session_class):
    """Test HTTP error handling (404, 500, etc.)."""
    mock_session = MagicMock()
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.json.return_value = {"detail": "Session not found"}

    http_error = requests.exceptions.HTTPError(response=mock_response)
    mock_response.raise_for_status.side_effect = http_error

    mock_session.get.return_value = mock_response
    mock_session_class.return_value = mock_session

    executor = ObsidianToolExecutor()

    result = executor.execute_tool_call("get_session_details", {"session_id": "nonexistent"})

    assert result["success"] is False
    assert result["error"]["type"] == "HTTPError"
    assert result["error"]["http_status"] == 404
    assert "not found" in result["error"]["message"].lower()


@patch("obsidian.api.llm.tool_executor.requests.Session")
def test_network_error_handling(mock_session_class):
    """Test network error handling (timeout, connection error)."""
    mock_session = MagicMock()
    mock_session.get.side_effect = requests.exceptions.ConnectionError("Connection refused")
    mock_session_class.return_value = mock_session

    executor = ObsidianToolExecutor()

    result = executor.execute_tool_call("list_optimization_sessions", {})

    assert result["success"] is False
    assert result["error"]["type"] == "ConnectionError"
    assert result["error"]["http_status"] is None


def test_context_manager():
    """Test executor works as context manager."""
    with ObsidianToolExecutor() as executor:
        assert executor.session is not None

    # Session should be closed after context exit
    # (We can't directly test this without mocking, but verify no error)


def test_close_method():
    """Test executor close method."""
    executor = ObsidianToolExecutor()
    session_mock = Mock()
    executor.session = session_mock

    executor.close()

    session_mock.close.assert_called_once()


@patch("obsidian.api.llm.tool_executor.requests.Session")
def test_arguments_not_modified(mock_session_class):
    """Test that original arguments dict is not modified."""
    mock_session = MagicMock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"X_best": {}}
    mock_session.get.return_value = mock_response
    mock_session_class.return_value = mock_session

    executor = ObsidianToolExecutor()

    original_args = {"session_id": "test123", "other_param": "value"}
    args_copy = original_args.copy()

    executor.execute_tool_call("get_best_results", original_args)

    # Original arguments should not be modified
    assert original_args == args_copy


@patch("obsidian.api.llm.tool_executor.requests.Session")
def test_query_parameters_for_get(mock_session_class):
    """Test that non-session_id parameters are passed as query params for GET."""
    mock_session = MagicMock()
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = []
    mock_session.get.return_value = mock_response
    mock_session_class.return_value = mock_session

    executor = ObsidianToolExecutor()

    executor.execute_tool_call("list_optimization_sessions", {"status_filter": "fitted"})

    # Verify GET was called with params
    call_args = mock_session.get.call_args
    assert "params" in call_args[1]
    assert call_args[1]["params"]["status_filter"] == "fitted"


@patch("obsidian.api.llm.tool_executor.requests.Session")
def test_json_body_for_post(mock_session_class):
    """Test that parameters are passed as JSON body for POST."""
    mock_session = MagicMock()
    mock_response = Mock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"rows_added": 5}
    mock_session.post.return_value = mock_response
    mock_session_class.return_value = mock_session

    executor = ObsidianToolExecutor()

    data = [{"Temperature": 25, "Yield": 85}]
    executor.execute_tool_call("add_experimental_data", {"session_id": "test123", "data": data})

    # Verify POST was called with json
    call_args = mock_session.post.call_args
    assert "json" in call_args[1]
    assert call_args[1]["json"]["data"] == data
