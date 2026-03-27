# Obsidian Orchestration Layer

Framework-agnostic session management for multi-campaign Bayesian optimization.

## Overview

The orchestration layer provides a bridge between Obsidian's core optimization engine (`Campaign`, `Optimizer`, `Surrogate`) and user-facing applications (REST APIs, Dash UIs, CLI tools). It adds:

- **Multi-session management** - Run multiple optimization campaigns concurrently
- **Persistent storage** - Sessions saved to disk with lazy loading
- **Lifecycle tracking** - Monitor campaign status through well-defined states
- **Operation history** - Audit trail of all operations with timestamps
- **Model diagnostics** - Track R², loss, hypervolume, and Pareto metrics
- **Framework independence** - No web/HTTP dependencies; works with any Python application

**Design Principle**: The orchestration layer is framework-agnostic. It doesn't depend on FastAPI, Dash, Flask, or any web framework. This allows it to be used in REST APIs, desktop apps, notebooks, or CLI tools.

## Architecture

### Core Components

```
SessionManager (Singleton)
├── Manages multiple CampaignSession objects
├── In-memory cache + disk persistence
├── Session discovery and lazy loading
└── Cleanup and lifecycle management

CampaignSession (Wrapper)
├── Wraps core Campaign class
├── Adds metadata (ID, name, timestamps)
├── Tracks status (SessionStatus enum)
├── Logs operation history
└── Provides convenience methods

SessionStatus (Enum)
├── CONFIGURED    → Session created
├── INITIALIZED   → Initial experiments generated
├── FITTED        → Surrogate model trained
├── SUGGESTING    → Generating suggestions
├── EVALUATING    → Evaluating predictions
├── COMPLETED     → Optimization finished
└── ERROR         → Error occurred
```

### Storage Structure

Sessions are stored in `~/.obsidian/sessions/{session_id}/`:

```
{session_id}/
├── metadata.json           # Session ID, name, status, timestamps
├── campaign_state.json     # Full Campaign state (parameters, targets, etc.)
├── history.jsonl          # Operation history (one JSON object per line)
└── data.csv               # Experimental data (parameters + responses)
```

## Quick Start

### Basic Usage

```python
from obsidian.orchestration import SessionManager
from obsidian.parameters import ParamSpace, Param_Continuous, Target

# Get singleton instance
manager = SessionManager.get_instance()

# Define optimization problem
param_space = ParamSpace([
    Param_Continuous("Temperature", 0, 100),
    Param_Continuous("Pressure", 1, 10)
])
targets = [Target("Yield", aim="max")]

# Create session
session_id = manager.create_session(
    param_space=param_space,
    targets=targets,
    name="Temperature-Pressure Optimization"
)

# Get session
session = manager.get_session(session_id)

# Initialize with Latin Hypercube Sampling
suggestions = session.initialize(m_initial=10, method="LHS")
print(f"Initial experiments: {suggestions}")

# Add experimental data (simulate results)
import pandas as pd
data = pd.DataFrame(suggestions)
data["Yield"] = [75.0, 82.0, 78.0, 85.0, 80.0, 83.0, 79.0, 84.0, 81.0, 86.0]
session.add_data(data)

# Fit surrogate model
session.fit()

# Suggest next experiments
next_experiments = session.suggest(m_batch=5, acquisition=["NEI"])
print(f"Next experiments: {next_experiments}")

# Get best results
best = session.get_best()
print(f"Best so far: {best}")

# Save session to disk
manager.save_session(session_id)
```

### Loading Existing Sessions

```python
# List all sessions
sessions = manager.list_sessions()
for s in sessions:
    print(f"{s['session_id']}: {s['name']} ({s['status']})")

# Filter by status
fitted_sessions = manager.list_sessions(status=SessionStatus.FITTED)

# Load specific session
session = manager.get_session(session_id)
print(f"Status: {session.status}")
print(f"Experiments: {len(session.campaign.data)}")
```

### Multi-Session Management

```python
# Create multiple sessions
session_ids = []
for temp_range in [(0, 100), (50, 150), (100, 200)]:
    param_space = ParamSpace([
        Param_Continuous("Temperature", temp_range[0], temp_range[1])
    ])
    sid = manager.create_session(param_space, targets, name=f"Temp {temp_range}")
    session_ids.append(sid)

# Work with each session
for sid in session_ids:
    session = manager.get_session(sid)
    session.initialize(m_initial=5)
    # ... run optimization
    manager.save_session(sid)

# Cleanup old sessions
manager.cleanup_old_sessions(days=30)
```

## API Reference

### SessionManager

**Singleton Access:**
```python
manager = SessionManager.get_instance()
manager = SessionManager(storage_dir="custom/path")  # Custom storage location
SessionManager.reset_instance()  # Reset singleton (useful for testing)
```

**Session Management:**
```python
# Create new session
session_id = manager.create_session(
    param_space: ParamSpace,
    targets: List[Target],
    name: str = None,
    seed: int = None
) -> str

# Retrieve session (loads from disk if needed)
session = manager.get_session(session_id: str) -> CampaignSession

# List sessions
sessions = manager.list_sessions(status: SessionStatus = None) -> List[dict]

# Check existence
exists = manager.session_exists(session_id: str) -> bool

# Delete session
manager.delete_session(session_id: str)

# Save session to disk
manager.save_session(session_id: str)

# Cleanup old sessions
manager.cleanup_old_sessions(days: int = 30) -> int
```

### CampaignSession

**Workflow Methods:**
```python
# Initialize with design of experiments
suggestions = session.initialize(
    m_initial: int = 10,
    method: str = "LHS",  # "LHS" or "Random"
    seed: int = None
) -> List[dict]

# Add experimental data
session.add_data(data: pd.DataFrame) -> int  # Returns rows added

# Fit surrogate model
session.fit(fit_options: dict = None)

# Suggest next experiments
suggestions = session.suggest(
    m_batch: int = 1,
    acquisition: List[str] = ["NEI"],
    optim_samples: int = None,
    optim_restarts: int = None,
    seed: int = None
) -> dict  # {suggestions: [...], evaluation: {...}}

# Evaluate predictions at arbitrary points
predictions = session.evaluate(
    X: List[dict],
    return_std: bool = False
) -> List[dict]

# Get best results
best = session.get_best() -> dict  # {X_best: {...}, response_max: {...}, ...}
```

**State & Diagnostics:**
```python
# Access underlying campaign
campaign = session.campaign

# Get session status
status = session.status  # SessionStatus enum

# Get operation history
history = session.history  # List[dict]

# Get diagnostics (R², loss, hypervolume, Pareto)
diagnostics = session.get_diagnostics() -> dict

# Get state dictionary
state = session.get_state() -> dict

# Export/import state
state_dict = session.save_state()
new_session = CampaignSession.load_state(state_dict)
```

### SessionStatus Enum

```python
from obsidian.orchestration import SessionStatus

SessionStatus.CONFIGURED    # Session created, campaign configured
SessionStatus.INITIALIZED   # Initial experiments generated
SessionStatus.FITTED        # Surrogate model fitted
SessionStatus.SUGGESTING    # Currently generating suggestions
SessionStatus.EVALUATING    # Currently evaluating predictions
SessionStatus.COMPLETED     # Optimization completed
SessionStatus.ERROR         # Error occurred
```

## Key Features

### 1. Persistent Storage with Lazy Loading

Sessions are saved to disk but only loaded into memory when accessed:

```python
# Create and save 100 sessions
for i in range(100):
    sid = manager.create_session(param_space, targets)
    # Session automatically saved on creation

# List sessions (fast - only reads metadata)
sessions = manager.list_sessions()  # Doesn't load full Campaign objects

# Access specific session (lazy load)
session = manager.get_session(session_ids[42])  # Only loads this session
```

**Benefits:**
- Fast session discovery (metadata only)
- Memory efficient (load on demand)
- Handles large numbers of sessions

### 2. Operation History & Audit Trail

Every operation is logged with timestamp and parameters:

```python
session.initialize(m_initial=10)
session.add_data(data)
session.fit()

# View history
for op in session.history:
    print(f"{op['timestamp']}: {op['operation']} - {op['status']}")
    # Example:
    # 2024-03-20T10:30:15: initialize - success (m_initial=10, method=LHS)
    # 2024-03-20T10:35:22: add_data - success (rows_added=10)
    # 2024-03-20T10:36:45: fit - success
```

**Use Cases:**
- Debugging (what operations ran and when?)
- Auditing (who did what?)
- Reproducibility (replay operation sequence)

### 3. Model Diagnostics

Track model quality metrics automatically:

```python
diagnostics = session.get_diagnostics()

# Single-objective diagnostics
print(f"R² score: {diagnostics['surrogates']['Yield']['r2_score']}")
print(f"Loss: {diagnostics['surrogates']['Yield']['loss']}")

# Multi-objective diagnostics
print(f"Hypervolume: {diagnostics['hypervolume']}")
print(f"Pareto points: {diagnostics['n_pareto_points']}")
```

**Available Metrics:**
- R² score (goodness of fit)
- Loss value (model training loss)
- Hypervolume (multi-objective quality)
- Pareto frontier size
- Number of experiments

### 4. Error Handling

Errors are captured and logged without crashing:

```python
try:
    session.fit()
except Exception as e:
    # Session status → ERROR
    # Error logged in history
    pass

# Check status
if session.status == SessionStatus.ERROR:
    # View last error
    last_op = session.history[-1]
    print(f"Error: {last_op['error_message']}")
```

### 5. Session Cleanup

Remove old sessions automatically:

```python
# Delete sessions older than 30 days
deleted_count = manager.cleanup_old_sessions(days=30)
print(f"Cleaned up {deleted_count} old sessions")
```

## Integration Examples

### REST API Integration

```python
# FastAPI example
from fastapi import FastAPI, Depends
from obsidian.orchestration import SessionManager

app = FastAPI()

def get_session_manager():
    return SessionManager.get_instance()

@app.post("/sessions")
def create_session(
    config: dict,
    manager: SessionManager = Depends(get_session_manager)
):
    session_id = manager.create_session(...)
    return {"session_id": session_id}

@app.get("/sessions/{session_id}/suggest")
def suggest_experiments(
    session_id: str,
    m_batch: int = 1,
    manager: SessionManager = Depends(get_session_manager)
):
    session = manager.get_session(session_id)
    suggestions = session.suggest(m_batch=m_batch)
    manager.save_session(session_id)
    return suggestions
```

See [obsidian/api/](../api/) for full REST API implementation.

### Dash UI Integration

```python
# Dash callback example
import dash
from obsidian.orchestration import SessionManager

app = dash.Dash(__name__)
manager = SessionManager.get_instance()

@app.callback(
    Output("session-list", "data"),
    Input("refresh-button", "n_clicks")
)
def update_session_list(n_clicks):
    sessions = manager.list_sessions()
    return sessions

@app.callback(
    Output("suggestions", "data"),
    Input("suggest-button", "n_clicks"),
    State("session-dropdown", "value")
)
def get_suggestions(n_clicks, session_id):
    session = manager.get_session(session_id)
    suggestions = session.suggest(m_batch=3)
    return suggestions
```

### CLI Integration

```python
# Command-line tool example
import click
from obsidian.orchestration import SessionManager

@click.group()
def cli():
    pass

@cli.command()
@click.option("--name", required=True)
def create(name):
    """Create new optimization session."""
    manager = SessionManager.get_instance()
    session_id = manager.create_session(param_space, targets, name=name)
    click.echo(f"Created session: {session_id}")

@cli.command()
def list():
    """List all sessions."""
    manager = SessionManager.get_instance()
    sessions = manager.list_sessions()
    for s in sessions:
        click.echo(f"{s['session_id']}: {s['name']} ({s['status']})")

if __name__ == "__main__":
    cli()
```

## Configuration

### Storage Location

Default: `~/.obsidian/sessions/`

Custom location:
```python
manager = SessionManager(storage_dir="/custom/path/sessions")
```

### Session ID Format

UUIDs (e.g., `a3f2c1b5-4d6e-7f8a-9b0c-1d2e3f4a5b6c`)

### File Formats

- **metadata.json**: Session metadata (ID, name, status, timestamps)
- **campaign_state.json**: Full Campaign state (parameters, targets, optimizer config)
- **history.jsonl**: Operation history (JSON Lines format - one object per line)
- **data.csv**: Experimental data (CSV with parameter and target columns)

## Testing

Run orchestration tests:
```bash
pytest obsidian/orchestration/tests/ -v
```

Test coverage:
- Session creation and retrieval (11 tests)
- Campaign operations (12 tests)
- Persistence and lazy loading
- Multi-session management
- Error handling
- Status transitions
- Cleanup functionality

## See Also

- [REST API Documentation](../api/README.md) - HTTP endpoints using orchestration
- [Core Campaign Documentation](../campaign/) - Underlying optimization engine
- [Examples](../api/examples/) - Usage examples and patterns

## License

GNU General Public License v3 (GPLv3)
