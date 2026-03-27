"""
OpenAI function calling integration for Obsidian REST API.

This module provides OpenAI-compatible tool definitions for all Obsidian API endpoints,
enabling LLM agents to interact with the optimization API programmatically.
"""

from obsidian.api.llm.tool_definitions import TOOL_DEFINITIONS
from obsidian.api.llm.tool_executor import ObsidianToolExecutor


def get_tools():
    """
    Get OpenAI-compatible tool definitions.

    Returns tool definitions in OpenAI function calling format, ready to use
    with OpenAI client or any LLM provider that supports function calling.

    Returns:
        list[dict]: List of 14 tool definitions in OpenAI format.

    Example:
        >>> from obsidian.api.llm import get_tools
        >>> from openai import OpenAI
        >>>
        >>> tools = get_tools()
        >>> client = OpenAI()
        >>> response = client.chat.completions.create(
        ...     model="gpt-4",
        ...     messages=[{"role": "user", "content": "Create an optimization session"}],
        ...     tools=tools
        ... )
    """
    return [
        {"type": "function", "function": func_def}
        for func_def in TOOL_DEFINITIONS.values()
    ]


__all__ = ["TOOL_DEFINITIONS", "ObsidianToolExecutor", "get_tools"]
