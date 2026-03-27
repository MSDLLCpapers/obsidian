# Obsidian REST API

HTTP REST API for Bayesian optimization and experimental design with Obsidian.

## Overview

The Obsidian API provides HTTP endpoints for managing optimization campaigns, generating experiment designs, and analyzing results. It exposes Obsidian's Bayesian optimization engine through a RESTful interface that can be accessed from any programming language or HTTP client.

**Key Features:**

- **14 REST endpoints** covering the full optimization workflow
- **OpenAPI documentation** with interactive Swagger UI
- **LLM integration** via OpenAI function calling (for autonomous optimization agents)
- **Session management** with persistent storage
- **Multi-objective optimization** with Pareto frontier analysis
- **Real-time diagnostics** including R², hypervolume, and model quality metrics

**Architecture:** The API uses a thin adapter pattern - HTTP endpoints delegate business logic to the [orchestration layer](../orchestration/README.md), which manages sessions and campaign lifecycles.

## Quick Start

### Installation

Install Obsidian with API extras:

```bash
# Using Poetry
poetry install -E api

# Or with pip (after building)
pip install obsidian-apo[api]
```

This installs:

- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- `requests` - HTTP client

### Start the API Server

```bash
uvicorn obsidian.api.app:app --reload
```

The server starts on `http://localhost:8000` with:

- **Swagger UI**: <http://localhost:8000/docs>
- **ReDoc**: <http://localhost:8000/redoc>
- **OpenAPI JSON**: <http://localhost:8000/openapi.json>

### Basic Workflow

```python
import requests

BASE_URL = "http://localhost:8000/api/v1"

# 1. Create optimization session
session_config = {
    "name": "Temperature Optimization",
    "parameters": [
        {"type": "continuous", "name": "Temperature", "min": 0, "max": 100},
        {"type": "continuous", "name": "Pressure", "min": 1, "max": 10}
    ],
    "targets": [
        {"name": "Yield", "aim": "max"}
    ],
    "seed": 42
}

response = requests.post(f"{BASE_URL}/sessions", json=session_config)
session_id = response.json()["session_id"]
print(f"Session created: {session_id}")

# 2. Initialize with experiment design
init_response = requests.post(
    f"{BASE_URL}/sessions/{session_id}/initialize",
    json={"m_initial": 10, "method": "LHS"}
)
suggestions = init_response.json()["suggestions"]
print(f"Generated {len(suggestions)} initial experiments")

# 3. Run experiments (simulated here)
for exp in suggestions:
    exp["Yield"] = 75.0 + (exp["Temperature"] - 50) * 0.2

# 4. Add experimental data
data_response = requests.post(
    f"{BASE_URL}/sessions/{session_id}/data",
    json={"data": suggestions}
)
print(f"Added {data_response.json()['rows_added']} experiments")

# 5. Fit surrogate model
fit_response = requests.post(f"{BASE_URL}/sessions/{session_id}/fit")
print(f"Model status: {fit_response.json()['status']}")

# 6. Get next experiment suggestions
suggest_response = requests.post(
    f"{BASE_URL}/sessions/{session_id}/suggest",
    json={"m_batch": 5, "acquisition": ["NEI"]}
)
next_experiments = suggest_response.json()["suggestions"]
print(f"Next experiments to run: {next_experiments}")

# 7. Get best results
best_response = requests.get(f"{BASE_URL}/sessions/{session_id}/best")
best = best_response.json()
print(f"Best result: {best['X_best']} → {best['response_max']}")
```

See [examples/basic_workflow.py](examples/basic_workflow.py) for a complete working example.

## API Endpoints

### Session Management (4 endpoints)

#### POST `/api/v1/sessions`

Create a new optimization session.

**Request Body:**

```json
{
  "name": "Optional session name",
  "parameters": [
    {
      "type": "continuous",
      "name": "Temperature",
      "min": 0,
      "max": 100
    },
    {
      "type": "categorical",
      "name": "Catalyst",
      "categories": ["A", "B", "C"]
    }
  ],
  "targets": [
    {
      "name": "Yield",
      "aim": "max",
      "f_transform": "none"
    }
  ],
  "seed": 42
}
```

**Response:** `201 Created`

```json
{
  "session_id": "a3f2c1b5-4d6e-7f8a-9b0c-1d2e3f4a5b6c",
  "name": "Optional session name",
  "status": "configured",
  "created_at": "2024-03-20T10:30:15Z"
}
```

#### GET `/api/v1/sessions`

List all sessions with optional status filter.

**Query Parameters:**

- `status_filter` (optional): Filter by status (`configured`, `initialized`, `fitted`, etc.)

**Response:** `200 OK`

```json
[
  {
    "session_id": "a3f2c1b5-...",
    "name": "Session 1",
    "status": "fitted",
    "created_at": "2024-03-20T10:30:15Z",
    "updated_at": "2024-03-20T11:45:22Z"
  }
]
```

#### GET `/api/v1/sessions/{session_id}`

Get detailed session information.

**Response:** `200 OK`

```json
{
  "session_id": "a3f2c1b5-...",
  "name": "Temperature Optimization",
  "status": "fitted",
  "parameter_names": ["Temperature", "Pressure"],
  "target_names": ["Yield"],
  "n_experiments": 15,
  "created_at": "2024-03-20T10:30:15Z",
  "updated_at": "2024-03-20T11:45:22Z"
}
```

#### DELETE `/api/v1/sessions/{session_id}`

Delete a session and all associated data.

**Response:** `204 No Content`

---

### Workflow Operations (5 endpoints)

#### POST `/api/v1/sessions/{session_id}/initialize`

Generate initial experiment design using Latin Hypercube Sampling (LHS) or Random sampling.

**Request Body:**

```json
{
  "m_initial": 10,
  "method": "LHS",
  "seed": 42
}
```

**Response:** `200 OK`

```json
{
  "suggestions": [
    {"Temperature": 15.2, "Pressure": 3.8},
    {"Temperature": 67.4, "Pressure": 8.1}
  ],
  "n_suggestions": 10,
  "method": "LHS"
}
```

#### POST `/api/v1/sessions/{session_id}/data`

Add experimental results to the session.

**Request Body:**

```json
{
  "data": [
    {"Temperature": 15.2, "Pressure": 3.8, "Yield": 72.5},
    {"Temperature": 67.4, "Pressure": 8.1, "Yield": 85.3}
  ]
}
```

**Response:** `200 OK`

```json
{
  "rows_added": 2,
  "total_rows": 12
}
```

#### POST `/api/v1/sessions/{session_id}/fit`

Fit surrogate model to experimental data.

**Request Body (optional):**

```json
{
  "fit_options": {}
}
```

**Response:** `200 OK`

```json
{
  "status": "fitted",
  "message": "Model fitted successfully"
}
```

#### POST `/api/v1/sessions/{session_id}/suggest`

Suggest next experiments using Bayesian optimization.

**Request Body:**

```json
{
  "m_batch": 3,
  "acquisition": ["NEI"],
  "optim_samples": 512,
  "optim_restarts": 10,
  "manual_seed": 42
}
```

**Response:** `200 OK`

```json
{
  "suggestions": [
    {"Temperature": 45.2, "Pressure": 6.3},
    {"Temperature": 78.9, "Pressure": 4.1},
    {"Temperature": 23.5, "Pressure": 7.8}
  ],
  "evaluation": {
    "acq_values": [2.34, 1.89, 1.67]
  },
  "n_suggestions": 3
}
```

**Acquisition Functions:**

- `NEI` - Noisy Expected Improvement (default, handles noise)
- `EI` - Expected Improvement
- `UCB` - Upper Confidence Bound
- `Mean` - Posterior mean (exploitation only)
- `qNEI` - Batch Noisy Expected Improvement
- `qEI` - Batch Expected Improvement

#### POST `/api/v1/sessions/{session_id}/evaluate`

Evaluate model predictions at arbitrary points (with optional uncertainty quantification).

**Request Body:**

```json
{
  "X": [
    {"Temperature": 50.0, "Pressure": 5.0},
    {"Temperature": 75.0, "Pressure": 7.5}
  ],
  "return_std": true
}
```

**Response:** `200 OK`

```json
{
  "predictions": [
    {
      "Temperature": 50.0,
      "Pressure": 5.0,
      "Yield_pred": 82.3,
      "Yield_std": 2.1
    },
    {
      "Temperature": 75.0,
      "Pressure": 7.5,
      "Yield_pred": 88.7,
      "Yield_std": 3.4
    }
  ],
  "n_points": 2
}
```

---

### Analysis & Results (5 endpoints)

#### GET `/api/v1/sessions/{session_id}/best`

Get best observed results so far.

**Response:** `200 OK`

```json
{
  "X_best": {
    "Temperature": 78.9,
    "Pressure": 6.3
  },
  "response_max": {
    "Yield": 91.2
  },
  "n_experiments": 25,
  "message": "Best result from 25 experiments"
}
```

#### GET `/api/v1/sessions/{session_id}/data`

Export all experimental data.

**Response:** `200 OK`

```json
{
  "data": [
    {"Temperature": 15.2, "Pressure": 3.8, "Yield": 72.5, "iteration": 0},
    {"Temperature": 67.4, "Pressure": 8.1, "Yield": 85.3, "iteration": 0}
  ],
  "n_rows": 25,
  "columns": {
    "parameters": ["Temperature", "Pressure"],
    "targets": ["Yield"]
  },
  "iterations": {
    "current": 2,
    "experiments_per_iteration": [10, 10, 5]
  }
}
```

#### GET `/api/v1/sessions/{session_id}/diagnostics`

Get model quality metrics and diagnostics.

**Response:** `200 OK`

```json
{
  "surrogates": {
    "Yield": {
      "r2_score": 0.92,
      "loss": 12.34,
      "model_type": "GP"
    }
  },
  "hypervolume": 1456.78,
  "n_pareto_points": 12,
  "n_experiments": 25,
  "current_iteration": 2,
  "status": "fitted"
}
```

**Metrics:**

- **R² score**: Goodness of fit (closer to 1.0 = better)
- **Loss**: Training loss (lower = better)
- **Hypervolume**: Multi-objective quality indicator (higher = better Pareto front)
- **Pareto points**: Number of non-dominated solutions

#### GET `/api/v1/sessions/{session_id}/history`

Get iteration-by-iteration optimization history.

**Response:** `200 OK`

```json
{
  "iterations": [
    {
      "iteration": 0,
      "n_experiments": 10,
      "best_response": {"Yield": 85.3},
      "mean_response": {"Yield": 78.2}
    },
    {
      "iteration": 1,
      "n_experiments": 10,
      "best_response": {"Yield": 91.2},
      "mean_response": {"Yield": 84.5}
    }
  ],
  "n_iterations": 2,
  "parameter_names": ["Temperature", "Pressure"],
  "target_names": ["Yield"]
}
```

#### GET `/api/v1/sessions/{session_id}/state_dict`

Export internal state dictionary for checkpointing or debugging.

**Query Parameters:**

- `object` (optional): `"campaign"` (default) or `"optimizer"`

**Response:** `200 OK`

```json
{
  "state_dict": {
    "param_space": {...},
    "targets": [...],
    "data": {...}
  },
  "object_type": "campaign"
}
```

---

## Configuration

### Environment Variables

- `OBSIDIAN_API_URL` - Base URL for API client (default: `http://localhost:8000/api/v1`)
- `OBSIDIAN_STORAGE_DIR` - Session storage directory (default: `~/.obsidian/sessions`)

### CORS Settings

CORS is enabled for all origins by default. For production, configure in `obsidian/api/app.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],  # Specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Server Configuration

```bash
# Custom host and port
uvicorn obsidian.api.app:app --host 0.0.0.0 --port 8080

# Production deployment
uvicorn obsidian.api.app:app --workers 4 --host 0.0.0.0 --port 8000

# With SSL
uvicorn obsidian.api.app:app --ssl-keyfile key.pem --ssl-certfile cert.pem
```

## Error Handling

### HTTP Status Codes

- `200 OK` - Successful operation
- `201 Created` - Session created
- `204 No Content` - Session deleted
- `400 Bad Request` - Invalid input (validation error)
- `404 Not Found` - Session not found
- `500 Internal Server Error` - Server error

### Error Response Format

```json
{
  "detail": "Session not found: nonexistent-id"
}
```

### Common Issues

**1. Session not found (404)**

```python
# Check session exists before accessing
response = requests.get(f"{BASE_URL}/sessions")
session_ids = [s["session_id"] for s in response.json()]
```

**2. Model not fitted (400)**

```python
# Always fit before suggesting
requests.post(f"{BASE_URL}/sessions/{session_id}/fit")
response = requests.post(
    f"{BASE_URL}/sessions/{session_id}/suggest",
    json={"m_batch": 5}
)
```

**3. Invalid parameter values (400)**

```python
# Check parameter bounds
# Temperature: [0, 100] ✓
# Temperature: 150 ✗ (out of bounds)
```

## LLM Integration

The API includes OpenAI function calling support for LLM agents (GPT-4, Claude via Databricks, etc.) to autonomously run optimization campaigns.

**Features:**

- 14 tool definitions (one per API endpoint)
- ObsidianToolExecutor HTTP client wrapper
- Examples for OpenAI and Databricks
- Autonomous optimization agents

**Quick Start:**

```python
from openai import OpenAI
import json

# Load tool definitions
with open("obsidian/api/llm/openai_tools.json") as f:
    tools = json.load(f)

# Create OpenAI client
client = OpenAI()

# Ask LLM to optimize
messages = [{
    "role": "user",
    "content": "Create an optimization session for temperature (0-100) to maximize yield, run 10 initial experiments, and suggest 5 more."
}]

response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools
)

# LLM autonomously calls API functions
# Tool executor translates to HTTP requests
```

See [llm/README.md](llm/README.md) for comprehensive LLM integration guide.

## Examples

### Python Requests

- [basic_workflow.py](examples/basic_workflow.py) - Complete optimization workflow
- [direct_orchestration.py](examples/direct_orchestration.py) - Direct use of orchestration layer

### LLM Integration

- [llm/examples/basic_openai_client.py](llm/examples/basic_openai_client.py) - OpenAI function calling
- [llm/examples/databricks_example.py](llm/examples/databricks_example.py) - Databricks + Claude
- [llm/examples/autonomous_optimization.py](llm/examples/autonomous_optimization.py) - Autonomous agent

### cURL Examples

```bash
# Create session
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": [{"type": "continuous", "name": "x", "min": 0, "max": 10}],
    "targets": [{"name": "y", "aim": "max"}]
  }'

# List sessions
curl http://localhost:8000/api/v1/sessions

# Get session details
curl http://localhost:8000/api/v1/sessions/{session_id}
```

## Testing

### Unit Tests

```bash
# Run all API tests
pytest obsidian/api/tests/ -v

# Specific test
pytest obsidian/api/tests/test_api.py::test_full_workflow -v
```

### Integration Tests (LLM)

```bash
# Start API server first
uvicorn obsidian.api.app:app --reload

# Run integration tests
pytest obsidian/api/llm/tests/test_integration.py -v
```

**Test Coverage:**

- 28 API endpoint tests
- 32 LLM tool definition tests
- 8 integration tests
- All passing ✓

## Architecture

### Thin Adapter Pattern

```
HTTP Request
    ↓
FastAPI Router (obsidian/api/routers/sessions.py)
    ↓
Orchestration Layer (obsidian/orchestration/session_manager.py)
    ↓
Core Engine (obsidian/campaign/campaign.py)
    ↓
HTTP Response
```

**Benefits:**

- Clear separation of concerns
- Testable business logic independent of HTTP layer
- Reusable orchestration layer (works with Dash, CLI, etc.)
- Thin HTTP layer (just request/response translation)

### Dependency Injection

FastAPI dependency injection provides SessionManager instance:

```python
from fastapi import Depends

def get_session_manager():
    return SessionManager.get_instance()

@app.post("/sessions")
def create_session(
    config: dict,
    manager: SessionManager = Depends(get_session_manager)
):
    ...
```

### Pydantic Models

All request/response data validated with Pydantic models (see [models.py](models.py)):

- Type checking
- Automatic validation
- OpenAPI schema generation
- Clear error messages

## Related Documentation

- [Orchestration Layer](../orchestration/README.md) - Session management and persistence
- [LLM Integration](llm/README.md) - OpenAI function calling for LLM agents
- [Core Campaign](../campaign/) - Underlying optimization engine
- [Parameter Definitions](../parameters/) - Parameter types and spaces

## License

GNU General Public License v3 (GPLv3)
