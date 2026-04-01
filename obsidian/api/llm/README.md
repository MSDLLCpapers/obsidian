# LLM Integration for Obsidian API

OpenAI function calling integration for autonomous Bayesian optimization with LLM agents.

## Overview

This module enables Large Language Models (GPT-4, Claude, etc.) to autonomously interact with the Obsidian REST API using OpenAI's function calling / tool use feature. LLMs can:

- Create and manage optimization sessions
- Generate experiment designs
- Analyze results and suggest next experiments
- Run complete optimization workflows autonomously
- Make data-driven decisions based on model diagnostics

**Use Cases:**

- **Autonomous optimization agents** - LLM runs full campaigns without human intervention
- **Interactive guidance** - LLM suggests next steps based on current results
- **Data analysis** - LLM interprets diagnostics and optimization history
- **Workflow automation** - Natural language → API calls → optimization

**Compatibility:**

- OpenAI models (GPT-4, GPT-4 Turbo)
- Databricks serving endpoints (Claude, Llama via OpenAI-compatible API)
- Any LLM provider supporting OpenAI function calling format

## Quick Start

### Installation

```bash
# Install with LLM extras (includes openai SDK)
poetry install -E llm

# Or just the API dependencies
poetry install -E api
pip install openai
```

### Basic Example

```python
from openai import OpenAI
import json
from obsidian.api.llm import get_tools, ObsidianToolExecutor

# Get tool definitions (no file path needed!)
tools = get_tools()

# Initialize OpenAI client and tool executor
client = OpenAI()
executor = ObsidianToolExecutor()

# Ask LLM to create optimization session
messages = [{
    "role": "user",
    "content": "Create an optimization session for temperature (0-100°C) and pressure (1-10 bar) to maximize yield."
}]

response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

# Handle tool calls
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        # Execute function via tool executor
        result = executor.execute_tool_call(function_name, arguments)
        print(f"Called {function_name}: {result}")
```

See [examples/basic_openai_client.py](examples/basic_openai_client.py) for a complete working example with conversation loop.

## Architecture

### Components

```
┌─────────────────────────────────────────────────┐
│  LLM (GPT-4, Claude, etc.)                      │
│  - Receives tool definitions                    │
│  - Decides which tools to call                  │
│  - Generates function arguments                 │
└─────────────┬───────────────────────────────────┘
              │
              ├──> tool_definitions.py (16 function schemas)
              │
              ├──> openai_tools.json (generated JSON)
              │
              └──> tool_executor.py (HTTP client)
                   │
                   └──> Obsidian REST API
                        │
                        └──> Orchestration Layer → Core Engine
```

### Files

- **`tool_definitions.py`** - Python dictionary with 16 function definitions
- **`openai_tools.json`** - Generated JSON in OpenAI format (loaded by LLM clients)
- **`tool_executor.py`** - HTTP client that maps function calls to API endpoints
- **`generator.py`** - Script to regenerate JSON from Python definitions
- **`examples/`** - Working examples for OpenAI, Databricks, and autonomous agents

## Function Catalog

### Session Management (4 functions)

| Function | Purpose | Key Parameters |
|----------|---------|----------------|
| `create_optimization_session` | Start new campaign | parameters, targets, name, seed, **surrogate** |
| `list_optimization_sessions` | Discover existing sessions | status_filter (optional) |
| `get_session_details` | Inspect session configuration | session_id |
| `delete_optimization_session` | Clean up session | session_id |

**New in surrogate selection:** Choose between GP, DNN, DKL, and other surrogate models at session creation. Pass as string (`"GP"`), dict with hyperparameters (`{"DNN": {"p_dropout": 0.3}}`), or list for multi-output (`["GP", {"DNN": {...}}]`).

### Workflow Operations (6 functions)

| Function | Purpose | Key Parameters |
|----------|---------|----------------|
| `initialize_experiments` | Generate initial design | session_id, m_initial, method |
| `sample_parameter_space` | Sample points without initializing | session_id, n_points, method |
| `add_experimental_data` | Upload results | session_id, data |
| `fit_surrogate_model` | Train model | session_id |
| `suggest_next_experiments` | Bayesian optimization | session_id, m_batch, acquisition, **fixed_var**, **optim_sequential**, **X_pending** |
| `evaluate_predictions` | Predict at arbitrary points | session_id, X, return_std |

**New in advanced suggest options:**

- **Acquisition hyperparameters**: Pass acquisition as dict to customize behavior (e.g., `[{"UCB": {"beta": 2.0}}]` for more exploration, `[{"NEHVI": {"ref_point": [50, 100]}}]` for multi-objective reference point)
- **fixed_var**: Lock specific parameters at constant values for sensitivity analysis (e.g., `{"Temperature": 25.0}`)
- **optim_sequential**: Toggle between sequential (faster) and joint (better batch quality) optimization
- **X_pending**: Account for experiments already queued/running to avoid redundant suggestions

**Note on sampling:** `sample_parameter_space` is stateless and does NOT change session status or store points. Use it for exploration, visualization, or custom designs. `initialize_experiments` commits to initialization and changes session status.

### Analysis & Results (7 functions)

| Function | Purpose | Key Parameters |
|----------|---------|----------------|
| `get_best_results` | Current best result | session_id |
| `get_campaign_data` | Export all data | session_id |
| `get_model_diagnostics` | Model quality metrics | session_id |
| `get_optimization_history` | Iteration-by-iteration progress | session_id |
| `export_state_dictionary` | Full state dump | session_id, object |
| `analyze_shap` | **NEW:** SHAP feature importance | session_id, target_name, n_samples, reference_point, seed |
| `analyze_sensitivity` | **NEW:** Gradient-based sensitivity | session_id, reference_point, perturbation |

### Informational (1 function)

| Function                        | Purpose                                  | Key Parameters |
|---------------------------------|------------------------------------------|----------------|
| `list_acquisition_functions`    | Discover valid acquisition functions     | None           |

This endpoint returns metadata about all built-in acquisition functions including:

- Name (e.g., "NEI", "EHVI", "UCB")
- Modalities (single-objective, multi-objective, or universal)
- Task types (optimization, characterization)
- Hyperparameters with defaults and types
- Human-readable descriptions

**Use cases:**

- Agent needs to choose appropriate acquisition function dynamically
- User asks "what acquisition functions are available?"
- Understanding hyperparameter options
- Filtering by single vs multi-objective

**Example response:**

```json
{
  "count": 12,
  "single_objective": ["NEI", "EI", "UCB", "PI", "SR", ...],
  "multi_objective": ["NEHVI", "EHVI", "NParEGO", ...],
  "universal": ["Mean", "RS", "SF"],
  "functions": [
    {
      "name": "NEI",
      "description": "Noisy Expected Improvement - EI variant for noisy observations",
      "modalities": ["single"],
      "task_types": ["optimization"],
      "hyperparameters": {}
    },
    ...
  ]
}
```

## Loading Tool Definitions

### Recommended: Use `get_tools()` Helper

The easiest way to load tool definitions is using the `get_tools()` helper function:

```python
from obsidian.api.llm import get_tools

# Get all 16 tool definitions in OpenAI format
tools = get_tools()

# Use with any OpenAI-compatible client
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools
)
```

**Benefits:**

- ✅ Works when installed as a package
- ✅ No file path dependencies
- ✅ Always in sync with code
- ✅ Clean import

### Alternative: Import Python Dictionary

For advanced use cases (custom filtering, modification):

```python
from obsidian.api.llm import TOOL_DEFINITIONS

# Access individual tool definitions
create_session_def = TOOL_DEFINITIONS["create_optimization_session"]

# Convert to OpenAI format manually
tools = [
    {"type": "function", "function": func_def}
    for func_def in TOOL_DEFINITIONS.values()
]

# Or filter to specific tools
workflow_tools = [
    {"type": "function", "function": TOOL_DEFINITIONS[name]}
    for name in ["initialize_experiments", "add_experimental_data", "fit_surrogate_model"]
]
```

### Legacy: Load from JSON File (Not Recommended)

The `openai_tools.json` file exists for reference but shouldn't be loaded directly:

```python
# ❌ Don't do this - breaks when installed as package
import json
with open("obsidian/api/llm/openai_tools.json") as f:
    tools = json.load(f)

# ✅ Do this instead
from obsidian.api.llm import get_tools
tools = get_tools()
```

## Tool Executor

### ObsidianToolExecutor Class

HTTP client wrapper that translates function calls to REST API requests.

**Features:**

- Automatic endpoint mapping (function name → HTTP method + URL)
- Session ID extraction and URL formatting
- Structured error handling
- Environment-based configuration
- Context manager support

**Usage:**

```python
from obsidian.api.llm.tool_executor import ObsidianToolExecutor

# Initialize (uses OBSIDIAN_API_URL env var or defaults to localhost)
executor = ObsidianToolExecutor()

# Or specify base URL
executor = ObsidianToolExecutor(base_url="http://api.example.com/api/v1")

# Execute function call
result = executor.execute_tool_call("create_optimization_session", {
    "name": "My Optimization",
    "parameters": [
        {"type": "continuous", "name": "x", "min": 0, "max": 10}
    ],
    "targets": [
        {"name": "y", "aim": "max"}
    ]
})

# Result is a dictionary (JSON response from API)
session_id = result["session_id"]

# Context manager usage
with ObsidianToolExecutor() as executor:
    result = executor.execute_tool_call("list_optimization_sessions", {})
```

**Error Handling:**

```python
result = executor.execute_tool_call("get_session_details", {
    "session_id": "nonexistent-id"
})

# Errors returned as structured dictionaries
if result.get("success") == False:
    print(f"Error: {result['error']['message']}")
    print(f"HTTP Status: {result['error']['http_status']}")
    print(f"Type: {result['error']['type']}")
```

## Configuration

### Environment Variables

```bash
# API base URL (default: http://localhost:8000/api/v1)
export OBSIDIAN_API_URL="http://your-api-server:8000/api/v1"

# OpenAI API key (for OpenAI models)
export OPENAI_API_KEY="sk-..."

# Databricks token (for Databricks models)
export DATABRICKS_TOKEN="dapi..."
```

### Databricks Configuration

For Claude or other models hosted on Databricks:

```python
from openai import OpenAI
import os

# Databricks uses OpenAI-compatible client
client = OpenAI(
    api_key=os.getenv("DATABRICKS_TOKEN"),
    base_url="https://your-workspace.databricks.com/serving-endpoints"
)

# Set Obsidian API URL (may be internal network)
os.environ["OBSIDIAN_API_URL"] = "http://internal-api.company.com:8000/api/v1"

# Load tools and use normally
response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",  # Databricks model name
    messages=messages,
    tools=tools
)
```

See [examples/databricks_example.py](examples/databricks_example.py) for complete example.

## Advanced Features (New)

### Surrogate Model Selection

Choose between different surrogate models when creating sessions:

```python
# Gaussian Process (default)
session = executor.execute_tool_call("create_optimization_session", {
    "parameters": [...],
    "targets": [...],
    "surrogate": "GP"
})

# Deep Neural Network with custom hyperparameters
session = executor.execute_tool_call("create_optimization_session", {
    "parameters": [...],
    "targets": [...],
    "surrogate": {"DNN": {"p_dropout": 0.3, "h_width": 32, "h_layers": 3}}
})

# Deep Kernel Learning
session = executor.execute_tool_call("create_optimization_session", {
    "parameters": [...],
    "targets": [...],
    "surrogate": "DKL"
})

# Multi-output with mixed surrogates
session = executor.execute_tool_call("create_optimization_session", {
    "parameters": [...],
    "targets": [{"name": "Yield", "aim": "max"}, {"name": "Purity", "aim": "max"}],
    "surrogate": ["GP", {"DNN": {"p_dropout": 0.2}}]
})
```

**Available surrogates**: `GP` (Gaussian Process), `DNN` (Deep Neural Network), `DKL` (Deep Kernel Learning), `MixedGP` (for mixed parameter types), `GPflat`, `GPprior`, `MTGP` (Multi-task).

### Acquisition Function Hyperparameters

Customize acquisition function behavior:

```python
# UCB with high exploration (beta controls exploration-exploitation tradeoff)
suggestions = executor.execute_tool_call("suggest_next_experiments", {
    "session_id": session_id,
    "m_batch": 3,
    "acquisition": [{"UCB": {"beta": 2.0}}]  # >1 = more exploration
})

# Multi-objective with reference point
suggestions = executor.execute_tool_call("suggest_next_experiments", {
    "session_id": session_id,
    "m_batch": 5,
    "acquisition": [{"NEHVI": {"ref_point": [-100, -50]}}]  # For hypervolume calculation
})

# Mixed acquisition functions (returns 6 suggestions: 2 per function)
suggestions = executor.execute_tool_call("suggest_next_experiments", {
    "session_id": session_id,
    "m_batch": 2,
    "acquisition": ["NEI", {"UCB": {"beta": 1.5}}, "SF"]  # 3 functions × 2 batch = 6
})
```

**Common hyperparameters**:

- `UCB`: `beta` (float, exploration weight, default: 1.0)
- `EHVI/NEHVI`: `ref_point` (list[float], reference point for hypervolume)
- `NParEGO`: `scalarization_weights` (list[float])
- `EI/PI`: `inflate` (float, inflation factor, default: 0)

### Advanced Suggest Options

#### Fixed Variables (Sensitivity Analysis)

Lock specific parameters at constant values:

```python
# Fix Temperature at 25°C, only vary other parameters
suggestions = executor.execute_tool_call("suggest_next_experiments", {
    "session_id": session_id,
    "m_batch": 5,
    "fixed_var": {"Temperature": 25.0}
})

# Fix multiple parameters (continuous and categorical)
suggestions = executor.execute_tool_call("suggest_next_experiments", {
    "session_id": session_id,
    "m_batch": 3,
    "fixed_var": {"Temperature": 25.0, "Catalyst": "TypeA"}
})
```

**Use cases**: Sensitivity analysis, constrained optimization, parameter sweeps at fixed conditions.

#### Batch Optimization Strategy

Control sequential vs joint optimization:

```python
# Sequential optimization (default, faster)
suggestions = executor.execute_tool_call("suggest_next_experiments", {
    "session_id": session_id,
    "m_batch": 5,
    "optim_sequential": True  # Optimize one at a time
})

# Joint optimization (slower but better batch designs)
suggestions = executor.execute_tool_call("suggest_next_experiments", {
    "session_id": session_id,
    "m_batch": 5,
    "optim_sequential": False  # Jointly optimize all 5 points
})
```

**Trade-off**: Sequential is ~5x faster but joint produces better space-filling batch designs.

#### Pending Experiments

Account for experiments already queued/running:

```python
# Specify experiments that are pending
pending = [
    {"Temperature": 50, "Pressure": 5, "Catalyst": "A"},
    {"Temperature": 75, "Pressure": 8, "Catalyst": "B"}
]

suggestions = executor.execute_tool_call("suggest_next_experiments", {
    "session_id": session_id,
    "m_batch": 3,
    "X_pending": pending
})

# Returns 5 total suggestions: 2 pending + 3 new
# Acquisition function accounts for pending points to avoid redundancy
```

**Use case**: Parallel experimentation where some experiments are already running.

#### Combined Advanced Options

Use all options together:

```python
suggestions = executor.execute_tool_call("suggest_next_experiments", {
    "session_id": session_id,
    "m_batch": 3,
    "acquisition": [{"UCB": {"beta": 1.8}}],
    "fixed_var": {"Catalyst": "TypeB"},
    "optim_sequential": False,
    "X_pending": [{"Temperature": 50, "Pressure": 5, "Catalyst": "TypeB"}]
})
```

### Model Interpretation & Analysis (New)

#### SHAP Analysis

Understand which parameters most influence the optimization target using SHAP (SHapley Additive exPlanations):

```python
# Compute SHAP values for a target
shap_result = executor.execute_tool_call("analyze_shap", {
    "session_id": session_id,
    "target_name": "Yield",
    "n_samples": 100,  # Number of samples for SHAP computation
    "seed": 42
})

# Returns feature importance ranking
for feature in shap_result["feature_importance"]:
    print(f"{feature['rank']}. {feature['parameter']}: {feature['importance']:.3f}")

# Mean SHAP values (positive = increases prediction)
print("SHAP values:", shap_result["shap_values_mean"])
# Example: {"Temperature": 5.2, "Pressure": -2.1, "Catalyst": 0.8}

# Model prediction at reference point
print(f"Predicted: {shap_result['predicted_value']}")
print(f"Expected (baseline): {shap_result['expected_value']}")
```

**Interpretation:**
- Positive SHAP value → parameter increases the predicted response
- Negative SHAP value → parameter decreases the predicted response  
- Larger absolute value → stronger effect on prediction
- Feature importance ranks parameters by mean absolute SHAP value

**Custom reference point:**

```python
# Analyze at specific operating conditions
shap_result = executor.execute_tool_call("analyze_shap", {
    "session_id": session_id,
    "target_name": "Yield",
    "reference_point": {"Temperature": 50, "Pressure": 5, "Catalyst": "A"},
    "n_samples": 200
})
```

**Use cases:**
- Understanding which parameters drive optimization
- Deciding which parameters to fix or vary
- Explaining model predictions to stakeholders
- Identifying key optimization levers

#### Sensitivity Analysis

Fast gradient-based sensitivity for local parameter effects:

```python
# Compute dy/dx at best point (default)
sens_result = executor.execute_tool_call("analyze_sensitivity", {
    "session_id": session_id
})

# Access raw sensitivities (dy/dx)
for target, param_sens in sens_result["sensitivity"].items():
    print(f"\n{target}:")
    for param, grad in param_sens.items():
        print(f"  d{target}/d{param} = {grad:.4f}")

# Access normalized sensitivities (0-1 scale for comparison)
normalized = sens_result["sensitivity_normalized"]["Yield"]
print("Normalized:", normalized)
# Example: {"Temperature": 1.0, "Pressure": 0.42, "Catalyst": 0.18}
```

**Interpretation:**
- Positive gradient → increasing parameter increases response
- Negative gradient → increasing parameter decreases response
- Normalized values → relative importance (1.0 = most sensitive parameter)

**Custom reference point:**

```python
# Analyze at specific conditions
sens_result = executor.execute_tool_call("analyze_sensitivity", {
    "session_id": session_id,
    "reference_point": {"Temperature": 75, "Pressure": 8},
    "perturbation": 1e-5  # Smaller perturbation for more accurate gradients
})
```

**SHAP vs Sensitivity comparison:**

| Feature | SHAP | Sensitivity |
|---------|------|-------------|
| **Speed** | Slower (~100 samples) | Fast (single gradient) |
| **Scope** | Global (averaged over space) | Local (one point) |
| **Output** | All targets separately | All targets together |
| **Best for** | Understanding overall importance | Quick local checks |

**Combined workflow:**

```python
# 1. Quick sensitivity check
sens = executor.execute_tool_call("analyze_sensitivity", {"session_id": sid})
top_param = max(sens["sensitivity_normalized"]["Yield"], key=lambda x: x[1])

# 2. Deep dive with SHAP on important parameter
shap = executor.execute_tool_call("analyze_shap", {
    "session_id": sid,
    "target_name": "Yield",
    "n_samples": 200
})

# 3. Decide whether to fix less important parameters
if shap["feature_importance"][-1]["importance"] < 0.1:
    # Fix least important parameter, focus on others
    fixed_param = shap["feature_importance"][-1]["parameter"]
```

## Usage Patterns

### 1. Autonomous Optimization Agent

LLM runs complete optimization workflow without human intervention:

```python
messages = [{
    "role": "user",
    "content": """
    Run a Bayesian optimization campaign:
    1. Create session for temperature (0-100) and pressure (1-10) to maximize yield
    2. Generate 10 initial experiments
    3. Simulate experiments (yield = 100 - (temp-50)^2/10 - (pressure-5)^2)
    4. Fit model
    5. Suggest 5 more experiments
    6. Report best result and convergence analysis
    """
}]

# LLM autonomously calls:
# - create_optimization_session
# - initialize_experiments
# - add_experimental_data
# - fit_surrogate_model
# - suggest_next_experiments
# - get_best_results
# - get_model_diagnostics
```

### 2. Interactive Guidance

LLM suggests next steps based on current state:

```python
messages = [{
    "role": "user",
    "content": "I have session abc-123 with 15 experiments. What should I do next?"
}]

# LLM calls:
# - get_session_details (understand configuration)
# - get_campaign_data (check data quality)
# - get_model_diagnostics (assess model fit)
# Then suggests: "Model R² is 0.95, good fit. Suggest running 5 more experiments..."
```

### 3. Data Analysis

LLM interprets optimization results:

```python
messages = [{
    "role": "user",
    "content": "Analyze the convergence of session xyz-789. Is it improving?"
}]

# LLM calls:
# - get_optimization_history (iteration-by-iteration progress)
# - get_model_diagnostics (R², hypervolume)
# Then provides analysis: "Campaign shows strong convergence. Best yield improved
# from 75 (iteration 0) to 92 (iteration 3). Model R² is 0.93. Pareto front has
# 8 non-dominated solutions..."
```

### 4. Multi-Session Management

LLM compares or manages multiple campaigns:

```python
messages = [{
    "role": "user",
    "content": "List all my sessions and tell me which one has the best results."
}]

# LLM calls:
# - list_optimization_sessions
# - get_best_results (for each session)
# Then reports: "You have 5 sessions. Session 'High Temp' achieved best yield
# of 94.2, significantly better than others..."
```

## Examples

### Basic OpenAI Client

[examples/basic_openai_client.py](examples/basic_openai_client.py)

Complete conversation loop with GPT-4:

- Loads tool definitions
- Creates optimization session
- Generates initial experiments
- Multi-turn conversation with tool calls

**Run:**

```bash
export OPENAI_API_KEY="sk-..."
uvicorn obsidian.api.app:app --reload  # Terminal 1
python obsidian/api/llm/examples/basic_openai_client.py  # Terminal 2
```

### Databricks Integration

[examples/databricks_example.py](examples/databricks_example.py)

Uses Claude via Databricks OpenAI-compatible endpoint:

- Configures Databricks client
- Sets internal API URL
- Runs optimization with Claude model

**Run:**

```bash
export DATABRICKS_TOKEN="dapi..."
export OBSIDIAN_API_URL="http://internal-api:8000/api/v1"
python obsidian/api/llm/examples/databricks_example.py
```

### Autonomous Optimization

[examples/autonomous_optimization.py](examples/autonomous_optimization.py)

LLM autonomously runs 3 full optimization iterations:

- Creates session
- Generates initial design
- Iterates: suggest → simulate → add data → fit
- Analyzes convergence and reports results

## Testing

### Unit Tests

Test tool definitions and executor without running API server:

```bash
# Test tool definitions structure
pytest obsidian/api/llm/tests/test_tool_definitions.py -v

# Test executor logic (mocked HTTP)
pytest obsidian/api/llm/tests/test_tool_executor.py -v
```

**Coverage:** 32 tests covering:

- Tool definition format validation
- OpenAI schema compliance
- Executor endpoint mapping
- Error handling

### Integration Tests

Test actual LLM integration with running API server:

```bash
# Terminal 1: Start API
uvicorn obsidian.api.app:app --reload

# Terminal 2: Run integration tests
pytest obsidian/api/llm/tests/test_integration.py -v
```

**Coverage:** 8 integration tests covering:

- Tool executor connecting to real API
- Session creation and workflow operations
- Error handling (404s, invalid calls)
- Full optimization workflow end-to-end

## Regenerating Tool Definitions

If you modify API endpoints or add new ones, regenerate the JSON:

```bash
python obsidian/api/llm/generator.py
```

This updates `openai_tools.json` from `tool_definitions.py`.

**When to regenerate:**

- Added new API endpoint
- Changed endpoint parameters
- Updated function descriptions
- Modified parameter schemas

## Best Practices

### 1. LLM-Friendly Descriptions

Tool definitions include detailed descriptions for LLM understanding:

```python
"description": """Generate next experiment suggestions using Bayesian optimization.

Uses the fitted surrogate model to suggest promising experiments by optimizing the
acquisition function (e.g., Expected Improvement). Only call this after the model
has been fitted with fit_surrogate_model.

Use when:
- Model is fitted (check with get_model_diagnostics)
- You want to generate the next batch of experiments
- Continuing an optimization loop

Limitations:
- Requires at least one model fit
- Acquisition function must be compatible with problem type
"""
```

### 2. Explicit Session IDs

Always pass `session_id` explicitly (no global state):

```python
# Good: Explicit session ID
result = executor.execute_tool_call("suggest_next_experiments", {
    "session_id": "abc-123",
    "m_batch": 5
})

# Bad: Implicit session (not supported)
executor.set_session("abc-123")  # ❌ Doesn't exist
result = executor.execute_tool_call("suggest_next_experiments", {
    "m_batch": 5
})
```

### 3. Error Handling

Always check for errors before proceeding:

```python
# After every tool call
result = executor.execute_tool_call("fit_surrogate_model", {"session_id": sid})

if result.get("success") == False:
    print(f"Error fitting model: {result['error']['message']}")
    # Handle error...
else:
    print(f"Model fitted successfully: {result['status']}")
```

### 4. Model Diagnostics

Check model quality before trusting suggestions:

```python
# Get diagnostics
diag = executor.execute_tool_call("get_model_diagnostics", {"session_id": sid})

# Check R² score
r2 = diag["surrogates"]["Yield"]["r2_score"]
if r2 < 0.7:
    print("Warning: Low R² score, model fit may be poor")
    # Maybe add more data before suggesting...
```

## Troubleshooting

### Issue: Tool calls not working

**Symptoms:** LLM doesn't call tools, just responds with text

**Solutions:**

- Verify tool definitions loaded correctly: `len(tools) == 14`
- Check OpenAI API version: `pip install --upgrade openai`
- Use `tool_choice="auto"` or `tool_choice="required"` in API call
- Check model supports function calling (GPT-4, not GPT-3.5-turbo-instruct)

### Issue: API connection errors

**Symptoms:** `ConnectionError`, `ConnectionRefused`

**Solutions:**

- Verify API server running: `curl http://localhost:8000/health`
- Check `OBSIDIAN_API_URL` environment variable
- Verify network connectivity (firewall, VPN, etc.)

### Issue: Invalid function arguments

**Symptoms:** LLM calls function with wrong parameter types

**Solutions:**

- Check tool definitions have correct JSON schema types
- Regenerate JSON: `python obsidian/api/llm/generator.py`
- Add more detailed descriptions and examples in definitions
- Use stricter type hints (`integer` not `number` for counts)

### Issue: Session not found (404)

**Symptoms:** `get_session_details` returns 404 error

**Solutions:**

- Call `list_optimization_sessions` first to get valid session IDs
- Check session wasn't deleted
- Verify session ID format (UUID string)

## Performance Considerations

### Token Usage

Each function call consumes tokens:

- Tool definitions in prompt: ~3-5K tokens (all 16 functions)
- Function call response: ~100-500 tokens
- Full conversation (10 turns): ~10-20K tokens

**Optimization:**

- Only load relevant tools if possible (not all 14)
- Use shorter function descriptions (keep key info)
- Batch tool calls in single turn when possible

### API Calls

Each LLM tool call = 1 HTTP request to Obsidian API:

- Typical latency: 10-100ms (local), 100-500ms (remote)
- Suggest using batch operations (`m_batch=5` vs 5 separate calls)
- Consider caching diagnostics/data for repeated queries

## Related Documentation

- [REST API Documentation](../README.md) - Full API endpoint reference
- [Orchestration Layer](../../orchestration/README.md) - Session management internals
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling) - Official OpenAI docs
- [Databricks OpenAI](https://docs.databricks.com/en/generative-ai/external-models/openai.html) - Databricks integration guide

## License

GNU General Public License v3 (GPLv3)
