"""
Demonstration of acquisition function discovery for LLM agents.

This example shows how LLM agents can dynamically discover valid acquisition
functions and their metadata, rather than having them hard-coded.

This is particularly useful when:
- Agent needs to choose appropriate acquisition function for problem type
- User asks "what acquisition functions are available?"
- Agent wants to understand hyperparameter options
- Selecting between single-objective vs multi-objective functions

Usage:
    # Terminal 1: Start API server
    uvicorn obsidian.api.app:app --reload

    # Terminal 2: Run this example
    python obsidian/api/llm/examples/acquisition_discovery.py
"""

from obsidian.api.llm import ObsidianToolExecutor


def main():
    # Initialize executor
    executor = ObsidianToolExecutor(verbose=True)

    print("=" * 80)
    print("Acquisition Function Discovery")
    print("=" * 80)

    # List all acquisition functions
    print("\n[Discovering acquisition functions...]")
    result = executor.execute_tool_call("list_acquisition_functions", {})

    if "count" not in result:
        print(f"Error: {result}")
        return

    print(f"\n✓ Found {result['count']} acquisition functions\n")

    # Display categorized lists
    print("Single-Objective Functions:")
    print(f"  {', '.join(result['single_objective'])}")

    print("\nMulti-Objective Functions:")
    print(f"  {', '.join(result['multi_objective'])}")

    print("\nUniversal Functions (work for both):")
    print(f"  {', '.join(result['universal'])}")

    # Show details for recommended functions
    print("\n" + "=" * 80)
    print("Recommended Functions (with details)")
    print("=" * 80)

    recommended = ["NEI", "NEHVI", "UCB", "RS"]
    for func in result["functions"]:
        if func["name"] in recommended:
            print(f"\n{func['name']}: {func['description']}")
            print(f"  Modalities: {', '.join(func['modalities'])}")
            print(f"  Task types: {', '.join(func['task_types'])}")
            if func["hyperparameters"]:
                print(f"  Hyperparameters:")
                for param_name, param_info in func["hyperparameters"].items():
                    default = param_info["default"]
                    ptype = param_info["type"]
                    optional = " (optional)" if param_info["optional"] else ""
                    print(f"    - {param_name}: {ptype} = {default}{optional}")
            else:
                print(f"  Hyperparameters: None")

    # Example: Agent choosing acquisition function based on problem
    print("\n" + "=" * 80)
    print("Example: Agent Decision Logic")
    print("=" * 80)

    print("\nScenario 1: Single-objective optimization")
    print("→ Agent chooses: NEI (Noisy Expected Improvement)")
    print("  Reason: Robust to noise, well-suited for most cases")

    print("\nScenario 2: Multi-objective optimization")
    print("→ Agent chooses: NEHVI (Noisy Expected Hypervolume Improvement)")
    print("  Reason: Recommended for multi-objective with noisy observations")

    print("\nScenario 3: Pure exploration (characterization)")
    print("→ Agent chooses: SF (Space Filling)")
    print("  Reason: Maximizes coverage of parameter space")

    print("\nScenario 4: User asks for 'most conservative' option")
    print("→ Agent chooses: UCB with high beta")
    print("  Reason: UCB with beta>1 favors exploration over exploitation")

    print("\n" + "=" * 80)
    print("Benefits for LLM Agents")
    print("=" * 80)
    print("""
✓ Dynamic discovery - no hard-coded function names
✓ Self-documenting - descriptions explain when to use each function
✓ Type-aware - can filter by single/multi-objective
✓ Hyperparameter discovery - learn valid options dynamically
✓ Robust to updates - new functions appear automatically

This enables agents to make informed decisions about acquisition functions
based on the optimization problem characteristics and user requirements.
""")

    executor.close()


if __name__ == "__main__":
    main()
