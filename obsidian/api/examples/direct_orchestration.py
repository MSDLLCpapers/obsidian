"""
Example: Using the orchestration layer directly (without HTTP API).

This demonstrates how to use SessionManager and CampaignSession directly,
which is useful for:
- Dash application callbacks
- CLI tools
- Batch processing scripts
- Any Python application that doesn't need HTTP

No web server required!
"""

import pandas as pd
import numpy as np
from pathlib import Path

from obsidian.orchestration import SessionManager, SessionStatus
from obsidian.parameters import ParamSpace, Param_Continuous, Param_Categorical, Target


def simulate_experiment(params: dict) -> float:
    """Simple simulator for demonstration."""
    temp = params["Temperature"]
    conc = params["Concentration"]
    variant_bonus = {"A": 0, "B": 5, "C": -3}
    bonus = variant_bonus.get(params.get("Variant", "A"), 0)

    yield_val = 100 - (temp - 20) ** 2 - (conc - 100) ** 2 / 100 + bonus
    yield_val += np.random.normal(0, 2)
    return max(0, yield_val)


def main():
    print("=" * 70)
    print("Direct Orchestration Layer Usage Example")
    print("=" * 70)

    # Step 1: Create parameter space and targets
    print("\n1. Setting up parameter space...")
    params = [
        Param_Continuous("Temperature", -10, 30),
        Param_Continuous("Concentration", 10, 150),
        Param_Categorical("Variant", ["A", "B", "C"]),
    ]
    X_space = ParamSpace(params)
    target = Target("Yield", aim="max")
    print("✓ Parameter space created")

    # Step 2: Get SessionManager (singleton)
    print("\n2. Getting SessionManager...")
    manager = SessionManager.get_instance()
    print(f"✓ SessionManager: {manager}")

    # Step 3: Create session
    print("\n3. Creating optimization session...")
    session = manager.create_session(X_space=X_space, target=target, name="Direct Orchestration Demo", seed=42)
    print(f"✓ Session created: {session.session_id}")
    print(f"  Status: {session.status}")

    # Step 4: Initialize
    print("\n4. Initializing experiment design...")
    X0 = session.initialize(m_initial=10, method="LHS")
    print(f"✓ Generated {len(X0)} initial experiments")
    print(X0.head(3))

    # Step 5: Simulate and add data
    print("\n5. Running simulated experiments...")
    results = []
    for _, row in X0.iterrows():
        params_dict = row.to_dict()
        yield_val = simulate_experiment(params_dict)
        results.append({**params_dict, "Yield": yield_val})

    results_df = pd.DataFrame(results)
    rows_added = session.add_data(results_df)
    print(f"✓ Added {rows_added} experiments")
    print(f"  Yield range: {results_df['Yield'].min():.2f} - {results_df['Yield'].max():.2f}")

    # Step 6: Fit model
    print("\n6. Fitting surrogate model...")
    session.fit()
    print(f"✓ Model fitted")
    print(f"  Status: {session.status}")

    # Step 7: Suggest next experiments
    print("\n7. Generating suggestions...")
    X_suggest, eval_suggest = session.suggest(m_batch=3, acquisition=["NEI"])
    print(f"✓ Generated {len(X_suggest)} suggestions")
    print(X_suggest)

    # Step 8: Get best results
    print("\n8. Getting best results...")
    best = session.get_best()
    print(f"✓ Best from {best['n_experiments']} experiments:")
    print(f"  Parameters: {best['X_best']}")
    print(f"  Yield: {best['response_max']:.2f}")

    # Step 9: Save state
    print("\n9. Saving session state...")
    save_dir = Path.home() / "obsidian_demo" / session.session_id
    session.save_state(directory=save_dir)
    print(f"✓ Session saved to {save_dir}")

    # Step 10: List sessions
    print("\n10. Listing all sessions...")
    all_sessions = manager.list_sessions()
    print(f"✓ Found {len(all_sessions)} session(s):")
    for s in all_sessions[:5]:
        print(f"  {s['session_id'][:8]}... | {s['name'][:30]:<30} | {s['status']}")

    # Step 11: Load session (demonstrate persistence)
    print("\n11. Demonstrating session reload...")
    # Clear from cache
    manager.sessions.pop(session.session_id, None)
    # Reload from disk
    reloaded_session = manager.get_session(session.session_id)
    print(f"✓ Session reloaded: {reloaded_session.session_id}")
    print(f"  Experiments: {reloaded_session.campaign.m_exp}")
    print(f"  Status: {reloaded_session.status}")

    print("\n" + "=" * 70)
    print("Direct orchestration workflow completed successfully!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("- No HTTP server needed")
    print("- Direct Python API")
    print("- Perfect for Dash callbacks, CLI tools, or scripts")
    print("- Same SessionManager can be used across different interfaces")


if __name__ == "__main__":
    main()
