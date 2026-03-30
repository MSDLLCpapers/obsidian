"""
Demonstration of verbosity control for LLM agents.

This example shows how the ObsidianToolExecutor provides automatic verbosity
control for both the executor itself and the underlying optimizer.

Features:
- Executor verbose=True prints [Obsidian] messages for function calls
- Optimizer verbose=3 (default) prints detailed optimization progress
- Auto-injection of verbose parameter for fit/suggest/evaluate operations
- Temporary override preserves original optimizer settings

Usage:
    # Terminal 1: Start API server
    uvicorn obsidian.api.app:app --reload

    # Terminal 2: Run this example
    python obsidian/api/llm/examples/verbosity_example.py
"""

from obsidian.api.llm import ObsidianToolExecutor

def main():
    # Initialize executor with maximum verbosity (default for LLM agents)
    # - verbose=True: Prints [Obsidian] messages for all function calls
    # - optimizer_verbose=3: Prints detailed optimizer output (debug level)
    print("=" * 80)
    print("Example 1: Maximum Verbosity (Default for LLM Agents)")
    print("=" * 80)

    executor = ObsidianToolExecutor(verbose=True, optimizer_verbose=3)

    # Create a session
    print("\n[Creating session...]")
    result = executor.execute_tool_call("create_optimization_session", {
        "name": "Verbosity Demo",
        "parameters": [
            {"type": "continuous", "name": "temperature", "min": 0, "max": 100},
            {"type": "continuous", "name": "pressure", "min": 1, "max": 10}
        ],
        "targets": [
            {"name": "yield", "aim": "max"}
        ],
        "seed": 42
    })

    if "session_id" not in result:
        print(f"Error creating session: {result}")
        return

    session_id = result["session_id"]
    print(f"\nSession created: {session_id}")

    # Initialize experiments
    print("\n[Initializing experiments...]")
    result = executor.execute_tool_call("initialize_experiments", {
        "session_id": session_id,
        "m_initial": 5,
        "method": "LHS",
        "seed": 42
    })

    # Add some synthetic data
    print("\n[Adding experimental data...]")
    import random
    random.seed(42)

    data = []
    for i in range(5):
        temp = random.uniform(0, 100)
        pressure = random.uniform(1, 10)
        # Synthetic yield function
        yield_val = 100 - ((temp - 50)**2 / 10) - ((pressure - 5)**2)
        data.append({"temperature": temp, "pressure": pressure, "yield": yield_val})

    result = executor.execute_tool_call("add_experimental_data", {
        "session_id": session_id,
        "data": data
    })

    # Fit model (this will show optimizer verbosity)
    print("\n[Fitting surrogate model...]")
    print("Watch for optimizer output below:")
    print("-" * 80)
    result = executor.execute_tool_call("fit_surrogate_model", {
        "session_id": session_id
    })
    print("-" * 80)

    # Suggest next experiments (this will also show optimizer verbosity)
    print("\n[Suggesting next experiments...]")
    print("Watch for optimizer output below:")
    print("-" * 80)
    result = executor.execute_tool_call("suggest_next_experiments", {
        "session_id": session_id,
        "m_batch": 2,
        "acquisition": ["NEI"]
    })
    print("-" * 80)

    # Clean up
    print("\n[Cleaning up...]")
    executor.execute_tool_call("delete_optimization_session", {
        "session_id": session_id
    })

    print("\n" + "=" * 80)
    print("Example 2: Quiet Mode (Not Typical for LLM Agents)")
    print("=" * 80)

    # Initialize executor with minimal verbosity
    executor_quiet = ObsidianToolExecutor(verbose=False, optimizer_verbose=0)

    print("\n[Creating and running a session quietly...]")
    result = executor_quiet.execute_tool_call("create_optimization_session", {
        "name": "Quiet Demo",
        "parameters": [
            {"type": "continuous", "name": "x", "min": 0, "max": 1}
        ],
        "targets": [
            {"name": "y", "aim": "max"}
        ]
    })

    if "session_id" in result:
        session_id_quiet = result["session_id"]
        # Initialize, add data, fit, suggest - all quietly
        executor_quiet.execute_tool_call("initialize_experiments", {
            "session_id": session_id_quiet,
            "m_initial": 3
        })
        executor_quiet.execute_tool_call("add_experimental_data", {
            "session_id": session_id_quiet,
            "data": [{"x": 0.1, "y": 0.5}, {"x": 0.5, "y": 0.8}, {"x": 0.9, "y": 0.3}]
        })
        executor_quiet.execute_tool_call("fit_surrogate_model", {
            "session_id": session_id_quiet
        })
        executor_quiet.execute_tool_call("suggest_next_experiments", {
            "session_id": session_id_quiet,
            "m_batch": 1
        })
        executor_quiet.execute_tool_call("delete_optimization_session", {
            "session_id": session_id_quiet
        })
        print("Session completed with no output (verbose=False, optimizer_verbose=0)")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("""
For LLM agents, we recommend:
- verbose=True (executor prints [Obsidian] messages)
- optimizer_verbose=3 (optimizer prints debug-level details)

This provides maximum transparency for:
- Debugging function calls
- Understanding optimization progress
- Monitoring convergence
- Diagnosing issues

The verbose parameter is automatically injected into fit/suggest/evaluate
operations and temporarily overrides the optimizer's verbosity setting.
""")

    executor.close()
    executor_quiet.close()

if __name__ == "__main__":
    main()
