"""
Example: Basic optimization workflow using the REST API.

This script demonstrates how to use the Obsidian REST API to:
1. Create an optimization session
2. Initialize with an experiment design
3. Simulate experiments
4. Add results to the session
5. Fit a surrogate model
6. Get next experiment suggestions
7. Retrieve best results

Requirements:
    - API server running: uvicorn obsidian.api.app:app --reload
    - requests library: pip install requests
"""

import requests
import pandas as pd
import numpy as np

# API base URL
BASE_URL = "http://localhost:8000/api/v1"


def simulate_experiment(params: dict) -> float:
    """
    Simple simulator for demonstration.
    Simulates: Yield = 100 - (Temperature - 20)^2 - (Concentration - 100)^2/100
    """
    temp = params["Temperature"]
    conc = params["Concentration"]
    variant_bonus = {"A": 0, "B": 5, "C": -3}
    bonus = variant_bonus.get(params.get("Variant", "A"), 0)

    yield_val = 100 - (temp - 20) ** 2 - (conc - 100) ** 2 / 100 + bonus
    # Add noise
    yield_val += np.random.normal(0, 2)
    return max(0, yield_val)


def main():
    print("=" * 70)
    print("Obsidian API - Basic Workflow Example")
    print("=" * 70)

    # Step 1: Create session
    print("\n1. Creating optimization session...")
    session_config = {
        "name": "Enzyme Optimization Demo",
        "parameters": [
            {"type": "continuous", "name": "Temperature", "min": -10, "max": 30},
            {"type": "continuous", "name": "Concentration", "min": 10, "max": 150},
            {"type": "categorical", "name": "Variant", "categories": ["A", "B", "C"]},
        ],
        "targets": [{"name": "Yield", "aim": "max"}],
        "seed": 42,
    }

    response = requests.post(f"{BASE_URL}/sessions", json=session_config)
    if response.status_code != 201:
        print(f"Error: {response.json()}")
        return

    session_data = response.json()
    session_id = session_data["session_id"]
    print(f"✓ Session created: {session_id}")
    print(f"  Name: {session_data['name']}")
    print(f"  Status: {session_data['status']}")

    # Step 2: Initialize experiment design
    print("\n2. Initializing experiment design...")
    init_config = {"m_initial": 10, "method": "LHS"}

    response = requests.post(f"{BASE_URL}/sessions/{session_id}/initialize", json=init_config)
    if response.status_code != 200:
        print(f"Error: {response.json()}")
        return

    init_data = response.json()
    X0 = pd.DataFrame(init_data["suggestions"])
    print(f"✓ Generated {len(X0)} initial experiments using {init_data['method']}")
    print(f"\nFirst 3 experiments:")
    print(X0.head(3))

    # Step 3: Simulate experiments
    print("\n3. Running simulated experiments...")
    results = []
    for _, row in X0.iterrows():
        params = row.to_dict()
        yield_val = simulate_experiment(params)
        results.append({**params, "Yield": yield_val})

    results_df = pd.DataFrame(results)
    print(f"✓ Completed {len(results)} experiments")
    print(f"\nResults summary:")
    print(f"  Yield: {results_df['Yield'].min():.2f} - {results_df['Yield'].max():.2f}")
    print(f"  Mean: {results_df['Yield'].mean():.2f}")

    # Step 4: Add data to session
    print("\n4. Adding results to session...")
    response = requests.post(f"{BASE_URL}/sessions/{session_id}/data", json={"data": results})
    if response.status_code != 200:
        print(f"Error: {response.json()}")
        return

    data_response = response.json()
    print(f"✓ Added {data_response['rows_added']} rows")
    print(f"  Total experiments: {data_response['total_rows']}")

    # Step 5: Fit surrogate model
    print("\n5. Fitting surrogate model...")
    response = requests.post(f"{BASE_URL}/sessions/{session_id}/fit")
    if response.status_code != 200:
        print(f"Error: {response.json()}")
        return

    fit_response = response.json()
    print(f"✓ Model fitted successfully")
    print(f"  Status: {fit_response['status']}")

    # Step 6: Get next suggestions
    print("\n6. Getting next experiment suggestions...")
    suggest_config = {"m_batch": 3, "acquisition": ["NEI"]}

    response = requests.post(f"{BASE_URL}/sessions/{session_id}/suggest", json=suggest_config)
    if response.status_code != 200:
        print(f"Error: {response.json()}")
        return

    suggest_response = response.json()
    X_suggest = pd.DataFrame(suggest_response["suggestions"])
    print(f"✓ Generated {suggest_response['n_suggestions']} suggestions")
    print(f"\nSuggested experiments:")
    print(X_suggest)

    # Step 7: Get best results
    print("\n7. Getting best results so far...")
    response = requests.get(f"{BASE_URL}/sessions/{session_id}/best")
    if response.status_code != 200:
        print(f"Error: {response.json()}")
        return

    best_response = response.json()
    print(f"✓ Best results from {best_response['n_experiments']} experiments:")
    print(f"\nBest parameters:")
    for param, value in best_response["X_best"].items():
        print(f"  {param}: {value}")
    print(f"\nBest Yield: {best_response['response_max']:.2f}")

    # Step 8: List all sessions
    print("\n8. Listing all sessions...")
    response = requests.get(f"{BASE_URL}/sessions")
    if response.status_code != 200:
        print(f"Error: {response.json()}")
        return

    sessions = response.json()
    print(f"✓ Found {len(sessions)} session(s):")
    for s in sessions[:5]:  # Show first 5
        print(f"  {s['session_id'][:8]}... | {s['name'][:30]:<30} | {s['status']:<10} | {s['n_experiments']} exps")

    print("\n" + "=" * 70)
    print("Workflow completed successfully!")
    print(f"Session ID: {session_id}")
    print("=" * 70)


if __name__ == "__main__":
    main()
