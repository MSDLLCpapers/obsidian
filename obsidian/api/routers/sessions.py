"""
Session management endpoints.

This router provides all HTTP endpoints for managing optimization sessions.
It's a thin adapter layer that delegates to the orchestration layer.
"""

from typing import List, Optional
import pandas as pd
from fastapi import APIRouter, HTTPException, status, Depends

from obsidian.orchestration import SessionManager, SessionStatus
from obsidian.api.models import (
    SessionCreate,
    SessionMetadata,
    SessionDetail,
    InitializeRequest,
    InitializeResponse,
    SampleRequest,
    SampleResponse,
    DataRequest,
    DataResponse,
    FitRequest,
    FitResponse,
    SuggestRequest,
    SuggestResponse,
    EvaluateRequest,
    EvaluateResponse,
    BestResult,
    DataExportResponse,
    DiagnosticsResponse,
    HistoryResponse,
    StateExportResponse,
    build_param_space,
    build_targets,
)

router = APIRouter()


# Dependency to get SessionManager
def get_session_manager() -> SessionManager:
    """Dependency that returns the SessionManager singleton."""
    return SessionManager.get_instance()


# ============================================================================
# Session CRUD Endpoints
# ============================================================================


@router.post("/sessions", response_model=SessionMetadata, status_code=status.HTTP_201_CREATED)
def create_session(session_data: SessionCreate, manager: SessionManager = Depends(get_session_manager)):
    """
    Create a new optimization session.

    Args:
        session_data: Session configuration (parameters, targets, name, seed)

    Returns:
        SessionMetadata with session ID and initial status
    """
    try:
        # Build ParamSpace and Targets
        X_space = build_param_space(session_data.parameters)
        targets = build_targets(session_data.targets)

        # Create optimizer with surrogate selection
        from obsidian.optimizer import BayesianOptimizer
        optimizer = BayesianOptimizer(
            X_space=X_space,
            surrogate=session_data.surrogate,
            seed=session_data.seed,
        )

        # Create session
        session = manager.create_session(
            X_space=X_space,
            target=targets if len(targets) > 1 else targets[0],
            name=session_data.name,
            seed=session_data.seed,
            optimizer=optimizer,
        )

        return SessionMetadata(**session.to_dict())

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to create session: {str(e)}")


@router.get("/sessions", response_model=list[SessionMetadata])
def list_sessions(
    status_filter: Optional[str] = None,
    manager: SessionManager = Depends(get_session_manager),
):
    """
    List all sessions.

    Args:
        status_filter: Optional filter by status (e.g., 'fitted', 'configured')

    Returns:
        List of session metadata
    """
    try:
        # Parse status filter
        status_enum = SessionStatus(status_filter) if status_filter else None

        # Get sessions
        sessions = manager.list_sessions(status=status_enum)

        return [SessionMetadata(**s) for s in sessions]

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid status filter: {str(e)}")


@router.get("/sessions/{session_id}", response_model=SessionDetail)
def get_session(session_id: str, manager: SessionManager = Depends(get_session_manager)):
    """
    Get detailed information about a session.

    Args:
        session_id: Session ID

    Returns:
        Detailed session information
    """
    try:
        session = manager.get_session(session_id)

        detail = session.to_dict()
        detail["parameter_names"] = list(session.campaign.X_space.X_names)
        detail["target_names"] = session.campaign.y_names

        return SessionDetail(**detail)

    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found")


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: str, manager: SessionManager = Depends(get_session_manager)):
    """
    Delete a session.

    Args:
        session_id: Session ID to delete
    """
    try:
        manager.delete_session(session_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found")


# ============================================================================
# Workflow Endpoints
# ============================================================================


@router.post("/sessions/{session_id}/initialize", response_model=InitializeResponse)
def initialize_session(
    session_id: str, request: InitializeRequest, manager: SessionManager = Depends(get_session_manager)
):
    """
    Generate initial experiment design.

    Args:
        session_id: Session ID
        request: Initialization parameters

    Returns:
        Initial experiment design
    """
    try:
        session = manager.get_session(session_id)

        X0 = session.initialize(m_initial=request.m_initial, method=request.method, seed=request.seed)

        # Save session state
        manager.save_session(session_id)

        return InitializeResponse(
            suggestions=X0.to_dict(orient="records"), n_experiments=len(X0), method=request.method
        )

    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to initialize: {str(e)}")


@router.post("/sessions/{session_id}/sample", response_model=SampleResponse)
def sample_parameter_space(
    session_id: str, request: SampleRequest, manager: SessionManager = Depends(get_session_manager)
):
    """
    Sample points from the parameter space without initializing the session.

    This is a stateless operation that generates random points according to the
    specified sampling method (LHS, Random, Sobol, etc.) without changing the
    session status. Useful for:
    - Exploring the parameter space
    - Generating test points for visualization
    - Understanding parameter bounds
    - Creating custom experimental designs

    Unlike /initialize, this does NOT:
    - Change session status to "initialized"
    - Store points in the session
    - Affect subsequent workflow operations

    Args:
        session_id: Session ID
        request: Sampling parameters (n_points, method, seed)

    Returns:
        Sampled points from parameter space
    """
    try:
        session = manager.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found")

    try:
        # Use the designer directly without changing session state
        # Temporarily override seed if provided
        original_seed = session.campaign.designer.seed
        if request.seed is not None:
            session.campaign.designer.seed = request.seed

        try:
            # Generate samples using the designer
            samples = session.campaign.designer.initialize(m_initial=request.n_points, method=request.method)

            return SampleResponse(samples=samples.to_dict(orient="records"), n_points=len(samples), method=request.method)
        finally:
            # Restore original seed
            session.campaign.designer.seed = original_seed

    except KeyError as e:
        # KeyError from designer (invalid method)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid sampling method: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to sample: {str(e)}")


@router.post("/sessions/{session_id}/data", response_model=DataResponse)
def add_data(session_id: str, request: DataRequest, manager: SessionManager = Depends(get_session_manager)):
    """
    Add experimental results to the session.

    Args:
        session_id: Session ID
        request: Experimental data

    Returns:
        Number of rows added
    """
    try:
        session = manager.get_session(session_id)

        # Convert to DataFrame
        df = pd.DataFrame(request.data)

        rows_added = session.add_data(df)

        # Save session state
        manager.save_session(session_id)

        return DataResponse(rows_added=rows_added, total_rows=session.campaign.m_exp)

    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to add data: {str(e)}")


@router.post("/sessions/{session_id}/fit", response_model=FitResponse)
def fit_model(
    session_id: str, request: FitRequest = FitRequest(), manager: SessionManager = Depends(get_session_manager)
):
    """
    Fit surrogate model to data.

    Args:
        session_id: Session ID
        request: Fitting options

    Returns:
        Fit status
    """
    try:
        session = manager.get_session(session_id)

        # Temporarily override optimizer verbosity if requested (for LLM agents)
        original_verbose = session.campaign.optimizer.verbose
        if request.verbose is not None:
            session.campaign.optimizer.verbose = request.verbose

        try:
            session.fit(fit_options=request.fit_options)

            # Save session state
            manager.save_session(session_id)

            return FitResponse(status=str(session.status), message="Model fitted successfully")
        finally:
            # Always restore original verbosity
            session.campaign.optimizer.verbose = original_verbose

    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to fit: {str(e)}")


@router.post("/sessions/{session_id}/suggest", response_model=SuggestResponse)
def suggest_experiments(
    session_id: str, request: SuggestRequest, manager: SessionManager = Depends(get_session_manager)
):
    """
    Generate next experiment suggestions.

    Args:
        session_id: Session ID
        request: Suggestion parameters

    Returns:
        Suggested experiments and acquisition values
    """
    try:
        session = manager.get_session(session_id)

        # Temporarily override optimizer verbosity if requested (for LLM agents)
        original_verbose = session.campaign.optimizer.verbose
        if request.verbose is not None:
            session.campaign.optimizer.verbose = request.verbose

        try:
            # Build kwargs
            optim_kwargs = {}
            if request.optim_samples:
                optim_kwargs["optim_samples"] = request.optim_samples
            if request.optim_restarts:
                optim_kwargs["optim_restarts"] = request.optim_restarts
            if request.manual_seed is not None:
                optim_kwargs["manual_seed"] = request.manual_seed

            # Add new parameters
            if request.fixed_var is not None:
                optim_kwargs["fixed_var"] = request.fixed_var
            optim_kwargs["optim_sequential"] = request.optim_sequential

            # Convert X_pending to DataFrame if provided
            if request.X_pending is not None:
                import pandas as pd
                optim_kwargs["X_pending"] = pd.DataFrame(request.X_pending)

            X_suggest, eval_suggest = session.suggest(
                m_batch=request.m_batch, acquisition=request.acquisition, **optim_kwargs
            )

            # Save session state
            manager.save_session(session_id)

            return SuggestResponse(
                suggestions=X_suggest.to_dict(orient="records"),
                evaluation=eval_suggest.to_dict(orient="records") if eval_suggest is not None else None,
                n_suggestions=len(X_suggest),
            )
        finally:
            # Always restore original verbosity
            session.campaign.optimizer.verbose = original_verbose

    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to suggest: {str(e)}")


@router.post("/sessions/{session_id}/evaluate", response_model=EvaluateResponse)
def evaluate_points(session_id: str, request: EvaluateRequest, manager: SessionManager = Depends(get_session_manager)):
    """
    Evaluate (predict) on arbitrary points.

    Args:
        session_id: Session ID
        request: Points to evaluate with optional return_std flag

    Returns:
        Predictions:
        - If return_std=False (default): Mean predictions only
        - If return_std=True: Mean predictions + standard deviation
    """
    try:
        session = manager.get_session(session_id)

        # Temporarily override optimizer verbosity if requested (for LLM agents)
        original_verbose = session.campaign.optimizer.verbose
        if request.verbose is not None:
            session.campaign.optimizer.verbose = request.verbose

        try:
            # Convert to DataFrame
            X = pd.DataFrame(request.X)

            predictions = session.evaluate(X, return_std=request.return_std)

            return EvaluateResponse(predictions=predictions.to_dict(orient="records"), n_points=len(predictions))
        finally:
            # Always restore original verbosity
            session.campaign.optimizer.verbose = original_verbose

    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Model not fitted: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to evaluate: {str(e)}")


# ============================================================================
# Results Endpoints
# ============================================================================


@router.get("/sessions/{session_id}/best", response_model=BestResult)
def get_best_results(session_id: str, manager: SessionManager = Depends(get_session_manager)):
    """
    Get best results from the session.

    Args:
        session_id: Session ID

    Returns:
        Best parameters and response values
    """
    try:
        session = manager.get_session(session_id)

        best = session.get_best()

        return BestResult(**best)

    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to get best results: {str(e)}")


@router.get("/sessions/{session_id}/data", response_model=DataExportResponse)
def get_campaign_data(session_id: str, manager: SessionManager = Depends(get_session_manager)):
    """
    Get full campaign data for analysis.

    Args:
        session_id: Session ID

    Returns:
        Complete campaign dataset with metadata
    """
    try:
        session = manager.get_session(session_id)
        data_export = session.get_data()
        return DataExportResponse(**data_export)

    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve data: {str(e)}"
        )


@router.get("/sessions/{session_id}/diagnostics", response_model=DiagnosticsResponse)
def get_model_diagnostics(session_id: str, manager: SessionManager = Depends(get_session_manager)):
    """
    Get model diagnostics and quality metrics.

    Args:
        session_id: Session ID

    Returns:
        Model quality metrics, training info, and multi-objective metrics
    """
    try:
        session = manager.get_session(session_id)
        diagnostics = session.get_diagnostics()
        return DiagnosticsResponse(**diagnostics)

    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve diagnostics: {str(e)}"
        )


@router.get("/sessions/{session_id}/history", response_model=HistoryResponse)
def get_optimization_history(session_id: str, manager: SessionManager = Depends(get_session_manager)):
    """
    Get iteration-by-iteration optimization history.

    Args:
        session_id: Session ID

    Returns:
        Per-iteration summary of optimization progress
    """
    try:
        session = manager.get_session(session_id)
        history = session.get_history()
        return HistoryResponse(**history)

    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve history: {str(e)}"
        )


@router.get("/sessions/{session_id}/state_dict", response_model=StateExportResponse)
def get_state_dictionary(
    session_id: str, object: str = "campaign", manager: SessionManager = Depends(get_session_manager)
):
    """
    Get state dictionary from campaign or optimizer.

    Args:
        session_id: Session ID
        object: Object type to export - "campaign" (default) or "optimizer"

    Returns:
        State dictionary from object.save_state()
    """
    try:
        session = manager.get_session(session_id)
        state_dict = session.get_state_dict(object_type=object)
        return StateExportResponse(**state_dict)

    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to retrieve state dictionary: {str(e)}"
        )
