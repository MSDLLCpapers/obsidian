"""
Unit tests for OpenAI tool definitions.

Tests that all function definitions are valid OpenAI format.
"""

import json
from pathlib import Path
import pytest

from obsidian.api.llm.tool_definitions import TOOL_DEFINITIONS
from obsidian.api.llm import get_tools


def test_tool_definitions_count():
    """Test that we have all 14 function definitions."""
    assert len(TOOL_DEFINITIONS) == 14


def test_tool_definitions_structure():
    """Test that all definitions have required OpenAI format fields."""
    for func_name, definition in TOOL_DEFINITIONS.items():
        # Check top-level fields
        assert "name" in definition, f"{func_name} missing 'name'"
        assert "description" in definition, f"{func_name} missing 'description'"
        assert "parameters" in definition, f"{func_name} missing 'parameters'"

        # Check parameters structure
        params = definition["parameters"]
        assert params["type"] == "object", f"{func_name} parameters must be object type"
        assert "properties" in params, f"{func_name} missing properties"

        # Check required fields exist (if specified)
        if "required" in params:
            assert isinstance(params["required"], list), f"{func_name} required must be a list"
            for req_field in params["required"]:
                assert req_field in params["properties"], f"{func_name} required field '{req_field}' not in properties"


def test_function_names_match():
    """Test that definition names match dictionary keys."""
    for func_name, definition in TOOL_DEFINITIONS.items():
        assert definition["name"] == func_name, f"Name mismatch: {func_name} != {definition['name']}"


def test_session_management_functions():
    """Test that session management functions are present."""
    expected = [
        "create_optimization_session",
        "list_optimization_sessions",
        "get_session_details",
        "delete_optimization_session",
    ]
    for func_name in expected:
        assert func_name in TOOL_DEFINITIONS, f"Missing: {func_name}"


def test_workflow_functions():
    """Test that workflow functions are present."""
    expected = [
        "initialize_experiments",
        "add_experimental_data",
        "fit_surrogate_model",
        "suggest_next_experiments",
        "evaluate_predictions",
    ]
    for func_name in expected:
        assert func_name in TOOL_DEFINITIONS, f"Missing: {func_name}"


def test_analysis_functions():
    """Test that analysis functions are present."""
    expected = [
        "get_best_results",
        "get_campaign_data",
        "get_model_diagnostics",
        "get_optimization_history",
        "export_state_dictionary",
    ]
    for func_name in expected:
        assert func_name in TOOL_DEFINITIONS, f"Missing: {func_name}"


def test_session_id_required():
    """Test that functions requiring session_id have it marked as required."""
    requires_session = [
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

    for func_name in requires_session:
        definition = TOOL_DEFINITIONS[func_name]
        params = definition["parameters"]
        assert "session_id" in params.get("required", []), f"{func_name} should require session_id"


def test_create_session_required_fields():
    """Test create_optimization_session has correct required fields."""
    definition = TOOL_DEFINITIONS["create_optimization_session"]
    required = definition["parameters"].get("required", [])

    assert "parameters" in required
    assert "targets" in required
    assert "name" not in required  # Optional
    assert "seed" not in required  # Optional


def test_parameter_types():
    """Test that parameter types are valid JSON Schema types."""
    valid_types = ["string", "number", "integer", "boolean", "object", "array"]

    for func_name, definition in TOOL_DEFINITIONS.items():
        params = definition["parameters"]
        for prop_name, prop_def in params.get("properties", {}).items():
            assert "type" in prop_def, f"{func_name}.{prop_name} missing type"
            assert prop_def["type"] in valid_types, f"Invalid type in {func_name}.{prop_name}: {prop_def['type']}"


def test_generated_json_exists():
    """Test that openai_tools.json was generated."""
    json_path = Path(__file__).parent.parent / "openai_tools.json"
    assert json_path.exists(), "openai_tools.json not found - run generator.py"


def test_generated_json_valid():
    """Test that generated JSON is valid OpenAI format."""
    json_path = Path(__file__).parent.parent / "openai_tools.json"

    with open(json_path) as f:
        tools = json.load(f)

    # Check we have all 14 tools
    assert len(tools) == 14

    # Check each tool has correct structure
    for tool in tools:
        assert tool["type"] == "function"
        assert "function" in tool

        func_def = tool["function"]
        assert "name" in func_def
        assert "description" in func_def
        assert "parameters" in func_def


def test_generated_json_matches_definitions():
    """Test that JSON matches Python definitions exactly."""
    json_path = Path(__file__).parent.parent / "openai_tools.json"

    with open(json_path) as f:
        tools = json.load(f)

    # Extract function definitions from JSON
    json_funcs = {tool["function"]["name"]: tool["function"] for tool in tools}

    # Compare with Python definitions
    for func_name, py_def in TOOL_DEFINITIONS.items():
        assert func_name in json_funcs, f"{func_name} not in generated JSON"

        json_def = json_funcs[func_name]

        assert json_def["name"] == py_def["name"]
        assert json_def["description"] == py_def["description"]
        assert json_def["parameters"] == py_def["parameters"]


def test_descriptions_are_helpful():
    """Test that descriptions are meaningful (not just function names)."""
    for func_name, definition in TOOL_DEFINITIONS.items():
        desc = definition["description"]

        # Should be longer than just the name
        assert len(desc) > len(func_name) * 2, f"{func_name} description too short"

        # Should contain useful keywords
        assert any(
            word in desc.lower() for word in ["use", "return", "when", "optimization"]
        ), f"{func_name} description not helpful"


def test_array_items_defined():
    """Test that array parameters have items defined."""
    for func_name, definition in TOOL_DEFINITIONS.items():
        params = definition["parameters"]

        for prop_name, prop_def in params.get("properties", {}).items():
            if prop_def.get("type") == "array":
                assert "items" in prop_def, f"{func_name}.{prop_name} array missing items"


def test_default_values():
    """Test that default values have correct types."""
    for func_name, definition in TOOL_DEFINITIONS.items():
        params = definition["parameters"]

        for prop_name, prop_def in params.get("properties", {}).items():
            if "default" in prop_def:
                default = prop_def["default"]
                param_type = prop_def["type"]

                # Check type consistency
                if param_type == "boolean":
                    assert isinstance(default, bool), f"{func_name}.{prop_name} default should be bool"
                elif param_type == "integer":
                    assert isinstance(default, int) and not isinstance(
                        default, bool
                    ), f"{func_name}.{prop_name} default should be int"
                elif param_type == "number":
                    assert isinstance(default, (int, float)) and not isinstance(
                        default, bool
                    ), f"{func_name}.{prop_name} default should be number"
                elif param_type == "string":
                    assert isinstance(default, str), f"{func_name}.{prop_name} default should be string"
                elif param_type == "array":
                    assert isinstance(default, list), f"{func_name}.{prop_name} default should be list"


def test_get_tools_helper():
    """Test get_tools() helper function returns correct format."""
    tools = get_tools()

    # Should return list of 14 tools
    assert isinstance(tools, list)
    assert len(tools) == 14

    # Each tool should have correct OpenAI format
    for tool in tools:
        assert tool["type"] == "function"
        assert "function" in tool

        func_def = tool["function"]
        assert "name" in func_def
        assert "description" in func_def
        assert "parameters" in func_def

    # Verify specific tool names exist
    tool_names = [t["function"]["name"] for t in tools]
    assert "create_optimization_session" in tool_names
    assert "suggest_next_experiments" in tool_names
    assert "get_best_results" in tool_names


def test_get_tools_matches_tool_definitions():
    """Test that get_tools() output matches TOOL_DEFINITIONS."""
    tools = get_tools()

    # Extract function definitions from tools
    tool_funcs = {t["function"]["name"]: t["function"] for t in tools}

    # Should have same tools as TOOL_DEFINITIONS
    assert set(tool_funcs.keys()) == set(TOOL_DEFINITIONS.keys())

    # Each tool should match exactly
    for name, func_def in tool_funcs.items():
        assert func_def == TOOL_DEFINITIONS[name]
