"""
Autonomous optimization example: LLM-driven Bayesian optimization loop.

This example demonstrates a fully autonomous optimization workflow where the LLM:
1. Creates and configures an optimization session
2. Generates initial experiments
3. Simulates experimental results
4. Fits models and analyzes diagnostics
5. Suggests new experiments iteratively
6. Analyzes convergence and determines when to stop

This showcases the power of LLM function calling for autonomous campaign management.

Prerequisites:
    1. Start Obsidian API server: uvicorn obsidian.api.app:app --reload
    2. Set OPENAI_API_KEY environment variable
    3. Install: pip install openai numpy
"""

import json
import os
from openai import OpenAI
import numpy as np

# Import tool definitions and executor
from obsidian.api.llm import get_tools, ObsidianToolExecutor


def simulate_experiment(temperature: float, pressure: float) -> float:
    """
    Simulate experimental yield as a function of temperature and pressure.

    This is a mock objective function for demonstration purposes.
    In real use, this would be replaced with actual experiments.
    """
    # Branin function modified for yield optimization
    t_norm = (temperature - 50) / 50  # Normalize to [-1, 1]
    p_norm = (pressure - 5.5) / 4.5  # Normalize to [-1, 1]

    yield_val = (
        -((p_norm - 5.1 * t_norm**2 / (4 * np.pi**2) + 5 * t_norm / np.pi - 6) ** 2)
        - 10 * (1 - 1 / (8 * np.pi)) * np.cos(t_norm)
        + 10
    )

    # Add noise
    yield_val += np.random.normal(0, 0.5)

    # Scale to realistic yield range (60-95%)
    yield_val = 60 + (yield_val + 5) * 5

    return float(np.clip(yield_val, 60, 95))


def main():
    # Initialize clients
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    executor = ObsidianToolExecutor()

    # Load tool definitions
    tools = get_tools()

    print("=" * 80)
    print("AUTONOMOUS BAYESIAN OPTIMIZATION WITH LLM FUNCTION CALLING")
    print("=" * 80)
    print(f"\nAPI Base URL: {executor.base_url}")
    print(f"Loaded {len(tools)} tool definitions\n")

    # Set up the autonomous optimization task
    messages = [
        {
            "role": "system",
            "content": (
                """You are an expert Bayesian optimization assistant. You autonomously manage
optimization campaigns by calling API functions. You:
- Create sessions and configure parameters
- Generate and analyze experimental designs
- Check model quality and convergence
- Make data-driven decisions about when to stop
- Provide clear summaries of results"""
            ),
        },
        {
            "role": "user",
            "content": (
                """Run an autonomous Bayesian optimization campaign to maximize yield:

Parameters:
- Temperature: 0-100°C
- Pressure: 1-10 bar

Instructions:
1. Create a session named "Autonomous Demo"
2. Generate 10 initial LHS experiments
3. I will provide simulated results for each batch of experiments
4. Fit the model and check diagnostics (R²)
5. If R² > 0.7, suggest 3 new experiments with NEI
6. Repeat steps 4-5 for 3 optimization iterations
7. Analyze final results: best parameters, convergence, history

After each experiment suggestion, I'll reply with "results: [data]". Continue autonomously."""
            ),
        },
    ]

    # Track session
    session_id = None
    iteration = 0
    max_iterations = 4  # 1 initial + 3 optimization iterations

    # Autonomous loop
    while iteration < max_iterations:
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}")
        print("=" * 80)

        # Call LLM
        response = client.chat.completions.create(model="gpt-4", messages=messages, tools=tools, tool_choice="auto")

        response_message = response.choices[0].message
        messages.append(response_message)

        # Handle tool calls
        if response_message.tool_calls:
            print(f"\nLLM is calling {len(response_message.tool_calls)} tool(s):\n")

            tool_results = []

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                print(f"→ {function_name}")

                # Execute tool call
                result = executor.execute_tool_call(function_name, arguments)

                # Extract session_id from creation
                if function_name == "create_optimization_session" and "session_id" in result:
                    session_id = result["session_id"]
                    print(f"  Created session: {session_id}")

                # Check if this is experiment suggestions
                if function_name in ["initialize_experiments", "suggest_next_experiments"]:
                    if "suggestions" in result:
                        n_suggestions = len(result["suggestions"])
                        print(f"  Generated {n_suggestions} experiment(s)")

                        # Simulate experiments
                        simulated_data = []
                        print("\n  Simulating experiments...")
                        for exp in result["suggestions"]:
                            temp = exp.get("Temperature", 50)
                            pres = exp.get("Pressure", 5.5)
                            yield_val = simulate_experiment(temp, pres)

                            simulated_data.append({"Temperature": temp, "Pressure": pres, "Yield": yield_val})
                            print(f"    T={temp:.1f}, P={pres:.1f} → Yield={yield_val:.2f}%")

                        # Add simulated results to conversation
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": function_name,
                                "content": json.dumps(result),
                            }
                        )

                        messages.append({"role": "user", "content": f"results: {json.dumps(simulated_data)}"})

                        iteration += 1
                        continue

                # Add tool result to conversation
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "name": function_name, "content": json.dumps(result)}
                )

                tool_results.append(result)

        else:
            # LLM provided text response
            if response_message.content:
                print(f"\nLLM: {response_message.content}\n")

            # Check if done
            if iteration >= max_iterations - 1:
                break

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)

    # Get final results
    if session_id:
        print(f"\nFetching final results for session {session_id}...")

        best = executor.execute_tool_call("get_best_results", {"session_id": session_id})
        if "X_best" in best:
            print(f"\n✓ Best Parameters:")
            for param, value in best["X_best"].items():
                print(f"    {param}: {value}")
            print(f"\n✓ Best Yield: {best['response_max']:.2f}%")
            print(f"✓ Total Experiments: {best['n_experiments']}")

    print("\n✓ Autonomous optimization complete!")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()
