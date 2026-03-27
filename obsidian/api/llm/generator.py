"""
Generate OpenAI-compatible JSON tool definitions from Python definitions.

This script converts the Python TOOL_DEFINITIONS dict to OpenAI function calling
JSON format and writes it to openai_tools.json.

Usage:
    python obsidian/api/llm/generator.py
"""

import json
from pathlib import Path
from obsidian.api.llm.tool_definitions import TOOL_DEFINITIONS


def generate_openai_tools_json():
    """Generate openai_tools.json from Python tool definitions."""
    tools = []

    for func_name, func_def in TOOL_DEFINITIONS.items():
        # Wrap each function definition in OpenAI format
        tools.append({"type": "function", "function": func_def})

    # Write to openai_tools.json in the same directory
    output_path = Path(__file__).parent / "openai_tools.json"
    with open(output_path, "w") as f:
        json.dump(tools, f, indent=2)

    print(f"✓ Generated {len(tools)} tool definitions")
    print(f"✓ Output: {output_path}")

    # Validate JSON can be reloaded
    with open(output_path, "r") as f:
        loaded = json.load(f)
        assert len(loaded) == len(tools), "JSON validation failed"

    print(f"✓ Validation passed")

    return output_path


if __name__ == "__main__":
    output_path = generate_openai_tools_json()
    print(f"\nSuccess! Tool definitions ready at:\n{output_path}")
