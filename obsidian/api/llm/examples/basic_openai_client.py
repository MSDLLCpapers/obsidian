"""
Basic example: OpenAI function calling with Obsidian API.

This example demonstrates how to use OpenAI's function calling feature
to interact with the Obsidian REST API for Bayesian optimization.

Prerequisites:
    1. Start Obsidian API server: uvicorn obsidian.api.app:app --reload
    2. Set OPENAI_API_KEY environment variable
    3. Install: pip install openai
"""

import json
import os
from openai import OpenAI

# Import tool definitions and executor
from obsidian.api.llm import get_tools, ObsidianToolExecutor


def main():
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Get tool definitions
    tools = get_tools()

    print(f"Loaded {len(tools)} tool definitions")

    # Initialize tool executor
    executor = ObsidianToolExecutor()
    print(f"API Base URL: {executor.base_url}\n")

    # Create a conversation with the LLM
    messages = [
        {
            "role": "system",
            "content": "You are an optimization assistant. Help users run Bayesian optimization campaigns.",
        },
        {
            "role": "user",
            "content": (
                """Create an optimization session to maximize yield by optimizing:
            - Temperature (0-100°C)
            - Pressure (1-10 bar)
            - Catalyst (options: A, B, C)

            Then generate 10 initial experiments using LHS."""
            ),
        },
    ]

    # Run conversation loop (up to 5 turns)
    for turn in range(5):
        print(f"--- Turn {turn + 1} ---")

        # Call LLM with tools
        response = client.chat.completions.create(model="gpt-4", messages=messages, tools=tools, tool_choice="auto")

        response_message = response.choices[0].message
        messages.append(response_message)

        # Check if LLM wants to call tools
        if response_message.tool_calls:
            print(f"LLM is calling {len(response_message.tool_calls)} tool(s)")

            # Execute each tool call
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                print(f"\nCalling: {function_name}")
                print(f"Arguments: {json.dumps(arguments, indent=2)}")

                # Execute the tool call
                result = executor.execute_tool_call(function_name, arguments)

                print(f"Result: {json.dumps(result, indent=2)[:200]}...")

                # Add tool result to conversation
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "name": function_name, "content": json.dumps(result)}
                )

        else:
            # LLM provided a text response (no tool calls)
            print(f"\nLLM Response: {response_message.content}")
            break

    print("\n✓ Conversation complete!")


if __name__ == "__main__":
    main()
