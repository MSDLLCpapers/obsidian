"""
Databricks example: Claude model with Obsidian API.

This example shows how to use Claude models hosted on Databricks
with OpenAI-compatible endpoints to control Obsidian optimization campaigns.

Prerequisites:
    1. Databricks workspace with Claude model endpoint
    2. Set DATABRICKS_TOKEN environment variable
    3. Start Obsidian API server accessible from Databricks
    4. Set OBSIDIAN_API_URL to point to your API server
"""

import json
import os
from openai import OpenAI
from obsidian.api.llm import get_tools


def main():
    # Configure for Databricks OpenAI-compatible endpoint
    databricks_token = os.getenv("DATABRICKS_TOKEN")
    if not databricks_token:
        raise ValueError("DATABRICKS_TOKEN environment variable not set")

    # Update this to your Databricks workspace and endpoint
    databricks_url = os.getenv("DATABRICKS_ENDPOINT", "https://your-workspace.databricks.com/serving-endpoints")

    client = OpenAI(api_key=databricks_token, base_url=databricks_url)

    print(f"Databricks endpoint: {databricks_url}")

    # Set API base URL for internal network
    # This should point to your Obsidian API server
    obsidian_api_url = os.getenv("OBSIDIAN_API_URL", "http://internal-api:8000/api/v1")  # Update this!
    os.environ["OBSIDIAN_API_URL"] = obsidian_api_url
    print(f"Obsidian API: {obsidian_api_url}\n")

    # Get tool definitions
    tools = get_tools()

    # Create optimization task
    messages = [
        {
            "role": "user",
            "content": (
                """Run a complete Bayesian optimization campaign:

        1. Create a session to optimize temperature (20-100°C) and concentration (10-150 mg/mL)
           to maximize yield
        2. Generate 12 initial experiments with LHS
        3. I'll provide the results, then you fit the model
        4. Suggest 5 new experiments using NEI acquisition
        5. Show me the diagnostics and best results

        For step 2, just create the session and initialize. Wait for me to provide data."""
            ),
        }
    ]

    print("Prompt:", messages[0]["content"][:100], "...\n")

    # Call Claude model (adjust model name to your Databricks endpoint)
    response = client.chat.completions.create(
        model="claude-3-5-sonnet-20241022", messages=messages, tools=tools, max_tokens=4096  # Update to your model name
    )

    # Display the response
    response_message = response.choices[0].message

    if response_message.content:
        print("Claude response:")
        print(response_message.content)

    if response_message.tool_calls:
        print(f"\nClaude wants to call {len(response_message.tool_calls)} tool(s):")
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"\n  Function: {function_name}")
            print(f"  Arguments: {json.dumps(arguments, indent=4)}")

    print("\n✓ Successfully called Claude via Databricks!")
    print("\nNext steps:")
    print("1. Execute the tool calls using ObsidianToolExecutor")
    print("2. Add tool results back to messages")
    print("3. Continue the conversation loop")


if __name__ == "__main__":
    main()
