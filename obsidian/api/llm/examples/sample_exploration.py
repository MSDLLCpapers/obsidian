"""
Example: Using sample_parameter_space for exploration.

This example demonstrates how to use the sample_parameter_space endpoint to
explore the parameter space and fitted surrogate model WITHOUT initializing
the session or affecting its state.

Use cases:
- Explore fitted surrogate surfaces before suggesting next experiments
- Visualize parameter space coverage
- Understand model predictions across the design space
- Agent exploration without committing to initialization

Usage:
    # Terminal 1: Start API server
    uvicorn obsidian.api.app:app --reload

    # Terminal 2: Run this example
    python obsidian/api/llm/examples/sample_exploration.py
"""

from obsidian.api.llm import ObsidianToolExecutor
import random


def main():
    executor = ObsidianToolExecutor(verbose=True)

    print("=" * 80)
    print("Sample Parameter Space - Exploration Example")
    print("=" * 80)

    # Create a session
    print("\n[Step 1: Create session]")
    result = executor.execute_tool_call(
        "create_optimization_session",
        {
            "name": "Exploration Demo",
            "parameters": [
                {"type": "continuous", "name": "temperature", "min": 0, "max": 100},
                {"type": "continuous", "name": "pressure", "min": 1, "max": 10},
            ],
            "targets": [{"name": "yield", "aim": "max"}],
            "seed": 42,
        },
    )

    session_id = result["session_id"]
    print(f"Session: {session_id}")
    print(f"Status: {result['status']}")

    # Sample some points BEFORE initialization
    print("\n[Step 2: Sample points BEFORE initialization]")
    print("This explores the parameter space without committing to initialization")

    sample_result = executor.execute_tool_call(
        "sample_parameter_space", {"session_id": session_id, "n_points": 10, "method": "LHS", "seed": 100}
    )

    print(f"Generated {sample_result['n_points']} sample points")
    print(f"First 3 samples:")
    for i, sample in enumerate(sample_result["samples"][:3], 1):
        print(f"  {i}. temp={sample['temperature']:.1f}, pressure={sample['pressure']:.1f}")

    # Verify status unchanged
    details = executor.execute_tool_call("get_session_details", {"session_id": session_id})
    print(f"\nSession status after sampling: {details['status']} (still 'configured')")

    # Now actually initialize with different seed/points
    print("\n[Step 3: Initialize session (different from samples)]")
    init_result = executor.execute_tool_call(
        "initialize_experiments", {"session_id": session_id, "m_initial": 5, "method": "LHS", "seed": 200}
    )

    print(f"Initialized with {init_result['n_experiments']} experiments")

    # Add synthetic data
    print("\n[Step 4: Add data and fit model]")
    random.seed(42)
    data = []
    for exp in init_result["suggestions"]:
        temp = exp["temperature"]
        pressure = exp["pressure"]
        # Synthetic function: yield = 100 - (temp-50)^2/10 - (pressure-5)^2
        yield_val = 100 - ((temp - 50) ** 2 / 10) - ((pressure - 5) ** 2)
        data.append({"temperature": temp, "pressure": pressure, "yield": yield_val})

    executor.execute_tool_call("add_experimental_data", {"session_id": session_id, "data": data})
    executor.execute_tool_call("fit_surrogate_model", {"session_id": session_id})

    print("Model fitted successfully")

    # Now use sampling to explore the fitted model
    print("\n[Step 5: Sample points to explore fitted model]")
    print("Generate 100 points and evaluate predictions")

    sample_result = executor.execute_tool_call(
        "sample_parameter_space", {"session_id": session_id, "n_points": 100, "method": "LHS", "seed": 300}
    )

    # Evaluate predictions on sampled points
    eval_result = executor.execute_tool_call(
        "evaluate_predictions",
        {"session_id": session_id, "X": sample_result["samples"], "return_std": True},
    )

    # Find best predicted point
    predictions = eval_result["predictions"]
    best_pred = max(predictions, key=lambda p: p["yield_mean"])

    print(f"\nExplored {len(predictions)} points in parameter space")
    print(f"Best predicted point:")
    print(f"  temperature={best_pred['temperature']:.1f}, pressure={best_pred['pressure']:.1f}")
    print(f"  predicted yield={best_pred['yield_mean']:.1f} ± {best_pred['yield_std']:.1f}")

    # Compare with actual best from data
    best_result = executor.execute_tool_call("get_best_results", {"session_id": session_id})
    print(f"\nBest observed point:")
    print(f"  temperature={best_result['X_best']['temperature']:.1f}, pressure={best_result['X_best']['pressure']:.1f}")
    print(f"  actual yield={best_result['response_max']['yield']:.1f}")

    # Clean up
    executor.execute_tool_call("delete_optimization_session", {"session_id": session_id})

    print("\n" + "=" * 80)
    print("Key Benefits of sample_parameter_space")
    print("=" * 80)
    print(
        """
✓ Stateless - no side effects on session
✓ Repeatable - can sample multiple times with different methods/seeds
✓ Flexible - explore before committing to initialization
✓ Powerful - combine with evaluate_predictions to explore surrogate

Common workflow:
1. sample_parameter_space → generate test points
2. evaluate_predictions → get model predictions on test points
3. analyze predictions → understand model behavior
4. suggest_next_experiments → make informed decisions
"""
    )

    executor.close()


if __name__ == "__main__":
    main()
