"""
Analysis endpoints for model interpretation and explainability.

This router provides endpoints for post-hoc analysis of fitted surrogate models:
- SHAP analysis for feature importance
- Sensitivity analysis for gradient-based parameter effects
- Parameter importance rankings

These endpoints return structured data (JSON), not plots, so consumers can
visualize results however they prefer.
"""

from typing import Optional
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Depends

from obsidian.campaign.explainer import Explainer
from obsidian.orchestration import SessionManager
from obsidian.api.models import (
    ShapAnalysisRequest,
    ShapAnalysisResponse,
    SensitivityAnalysisRequest,
    SensitivityAnalysisResponse,
)

router = APIRouter()


# Dependency to get SessionManager
def get_session_manager() -> SessionManager:
    """Dependency that returns the SessionManager singleton."""
    return SessionManager.get_instance()


@router.post("/sessions/{session_id}/analysis/shap", response_model=ShapAnalysisResponse)
def compute_shap_analysis(
    session_id: str,
    request: ShapAnalysisRequest,
    manager: SessionManager = Depends(get_session_manager)
):
    """
    Compute SHAP values for feature importance analysis.

    Uses Kernel SHAP to explain which parameters have the strongest effect on
    the predicted response. Returns structured data including SHAP values,
    feature importance rankings, and sample points.

    Args:
        session_id: Session ID
        request: SHAP analysis configuration

    Returns:
        Structured SHAP analysis results (not plots)

    Raises:
        404: Session not found
        400: Model not fitted or invalid target name
    """
    try:
        session = manager.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    # Check model is fitted
    if not session.campaign.optimizer.is_fit:
        raise HTTPException(
            status_code=400,
            detail="Model must be fitted before running SHAP analysis. Call /fit first."
        )

    # Validate target name
    if request.target_name not in session.campaign.y_names:
        raise HTTPException(
            status_code=400,
            detail=f"Target '{request.target_name}' not found. Available: {session.campaign.y_names}"
        )

    try:
        # Create explainer
        explainer = Explainer(session.campaign.optimizer)

        # Get target index
        target_idx = session.campaign.y_names.index(request.target_name)

        # Prepare reference point
        X_ref = None
        if request.reference_point:
            X_ref = pd.DataFrame([request.reference_point])

        # Compute SHAP values
        explainer.shap_explain(
            responseid=target_idx,
            n=request.n_samples,
            X_ref=X_ref,
            seed=request.seed
        )

        # Extract data from explainer
        shap_values = explainer.shap['values']  # Shape: (n_samples, n_features)
        X_sample = explainer.shap['X_sample']  # DataFrame
        X_ref_used = explainer.shap['X_ref']  # Reference point used
        expected_value = explainer.shap['explainer'].expected_value

        # Compute mean absolute SHAP values (feature importance)
        shap_values_abs_mean = np.abs(shap_values).mean(axis=0)

        # Get reference point prediction
        pred_at_ref = explainer.optimizer.predict(X_ref_used, return_f_inv=True)
        predicted_value = float(pred_at_ref[f"{request.target_name} (pred)"].values[0])

        # Build feature importance ranking
        feature_names = list(X_sample.columns)
        importance_list = [
            {
                "parameter": feature_names[i],
                "importance": float(shap_values_abs_mean[i]),
                "rank": int(rank + 1)
            }
            for rank, i in enumerate(np.argsort(shap_values_abs_mean)[::-1])
        ]

        # Build SHAP values dict (mean SHAP per feature)
        shap_values_mean = shap_values.mean(axis=0)
        shap_values_dict = {
            feature_names[i]: float(shap_values_mean[i])
            for i in range(len(feature_names))
        }

        # Build response
        response = ShapAnalysisResponse(
            target_name=request.target_name,
            reference_point=X_ref_used.to_dict(orient="records")[0],
            predicted_value=predicted_value,
            expected_value=float(expected_value),
            shap_values_mean=shap_values_dict,
            feature_importance=importance_list,
            n_samples=request.n_samples,
            samples=X_sample.head(min(50, request.n_samples)).to_dict(orient="records"),  # Limit to 50 for response size
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"SHAP analysis failed: {str(e)}"
        )


@router.post("/sessions/{session_id}/analysis/sensitivity", response_model=SensitivityAnalysisResponse)
def compute_sensitivity_analysis(
    session_id: str,
    request: SensitivityAnalysisRequest,
    manager: SessionManager = Depends(get_session_manager)
):
    """
    Compute gradient-based sensitivity analysis.

    Calculates dy/dx (local sensitivity) for each parameter at a reference point.
    This is faster than SHAP but only provides local information.

    Args:
        session_id: Session ID
        request: Sensitivity analysis configuration

    Returns:
        Structured sensitivity analysis results

    Raises:
        404: Session not found
        400: Model not fitted
    """
    try:
        session = manager.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    # Check model is fitted
    if not session.campaign.optimizer.is_fit:
        raise HTTPException(
            status_code=400,
            detail="Model must be fitted before running sensitivity analysis. Call /fit first."
        )

    try:
        # Create explainer
        explainer = Explainer(session.campaign.optimizer)

        # Prepare reference point
        X_ref = None
        if request.reference_point:
            X_ref = pd.DataFrame([request.reference_point])

        # Compute sensitivity
        sensitivity_df = explainer.sensitivity(dx=request.perturbation, X_ref=X_ref)

        # Get reference point used
        X_ref_used = X_ref if X_ref is not None else explainer.optimizer.X_best_f

        # Get predictions at reference
        pred_at_ref = explainer.optimizer.predict(X_ref_used, return_f_inv=True)
        predictions = {
            target_name: float(pred_at_ref[f"{target_name} (pred)"].values[0])
            for target_name in session.campaign.y_names
        }

        # Convert sensitivity DataFrame to nested dict
        # Format: {target: {parameter: sensitivity_value}}
        sensitivity = {}
        for target_name in session.campaign.y_names:
            col_name = f"{target_name} (pred)"
            if col_name in sensitivity_df.columns:
                sensitivity[target_name] = {
                    param: float(sensitivity_df.loc[param, col_name])
                    for param in sensitivity_df.index
                }

        # Compute normalized sensitivity (absolute values normalized to [0,1])
        sensitivity_normalized = {}
        for target_name, param_sens in sensitivity.items():
            abs_values = np.abs(list(param_sens.values()))
            max_abs = abs_values.max() if abs_values.max() > 0 else 1.0
            sensitivity_normalized[target_name] = {
                param: float(abs(sens) / max_abs)
                for param, sens in param_sens.items()
            }

        response = SensitivityAnalysisResponse(
            reference_point=X_ref_used.to_dict(orient="records")[0],
            predictions_at_reference=predictions,
            sensitivity=sensitivity,
            sensitivity_normalized=sensitivity_normalized,
            perturbation=request.perturbation,
        )

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Sensitivity analysis failed: {str(e)}"
        )
