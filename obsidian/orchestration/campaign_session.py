"""
Campaign Session wrapper with lifecycle management and metadata.

This module provides a stateful wrapper around the Campaign class that adds:
- Session metadata (ID, name, timestamps)
- Lifecycle status tracking
- Operation history logging
- Simplified state persistence

Design: Framework-agnostic, no web/HTTP dependencies. Can be used by REST APIs,
Dash apps, CLI tools, or any Python application.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from obsidian.campaign import Campaign
from obsidian.orchestration.enums import SessionStatus


class CampaignSession:
    """
    Wrapper around Campaign with lifecycle management and metadata tracking.

    This class provides a stateful interface to Campaign objects, adding:
    - Unique session ID
    - Human-readable name
    - Creation and update timestamps
    - Lifecycle status tracking
    - Operation history logging
    - Simplified persistence

    Attributes:
        session_id: Unique identifier for this session
        name: Human-readable session name
        created_at: Session creation timestamp
        updated_at: Last modification timestamp
        status: Current lifecycle status
        campaign: Underlying Campaign object
        history: List of operation records
    """

    def __init__(
        self,
        campaign: Campaign,
        session_id: str | None = None,
        name: str | None = None,
        status: SessionStatus = SessionStatus.CONFIGURED,
    ):
        """
        Initialize a Campaign Session.

        Args:
            campaign: The Campaign object to wrap
            session_id: Unique ID (generated if not provided)
            name: Human-readable name (auto-generated if not provided)
            status: Initial status (default: CONFIGURED)
        """
        self.session_id = session_id or str(uuid4())
        self.name = name or f"Campaign-{self.session_id[:8]}"
        self.created_at = datetime.utcnow()
        self.updated_at = self.created_at
        self.status = status
        self.campaign = campaign
        self.optimizer = campaign.optimizer  # Alias for easy access
        self.history: list[dict[str, Any]] = []

    def _log_operation(self, operation: str, params: dict[str, Any] | None = None):
        """
        Log an operation to the history.

        Args:
            operation: Operation name
            params: Operation parameters (optional)
        """
        entry = {"timestamp": datetime.utcnow().isoformat(), "operation": operation, "status": str(self.status)}
        if params:
            entry["params"] = params
        self.history.append(entry)
        self.updated_at = datetime.utcnow()

    def _update_status(self, new_status: SessionStatus):
        """Update session status and timestamp."""
        self.status = new_status
        self.updated_at = datetime.utcnow()

    # Campaign workflow methods
    def initialize(self, m_initial: int = 10, method: str = "LHS", seed: int | None = None) -> pd.DataFrame:
        """
        Generate initial experiment design.

        Args:
            m_initial: Number of initial experiments
            method: Design method ('LHS', 'Random', 'Factorial', etc.)
            seed: Random seed (optional)

        Returns:
            DataFrame with initial experiment design
        """
        try:
            self._log_operation("initialize", {"m_initial": m_initial, "method": method, "seed": seed})

            X0 = self.campaign.initialize(m_initial=m_initial, method=method)
            self._update_status(SessionStatus.INITIALIZED)

            return X0
        except Exception as e:
            self._update_status(SessionStatus.ERROR)
            self._log_operation("initialize_error", {"error": str(e)})
            raise

    def add_data(self, data: pd.DataFrame) -> int:
        """
        Add experimental results to the campaign.

        Args:
            data: DataFrame with parameter columns and response column(s)

        Returns:
            Number of rows added
        """
        try:
            rows_before = len(self.campaign.data)
            self.campaign.add_data(data)
            rows_added = len(self.campaign.data) - rows_before

            self._log_operation("add_data", {"rows_added": rows_added})

            return rows_added
        except Exception as e:
            self._update_status(SessionStatus.ERROR)
            self._log_operation("add_data_error", {"error": str(e)})
            raise

    def fit(self, fit_options: dict[str, Any] | None = None):
        """
        Fit surrogate model to data.

        Args:
            fit_options: Optional fitting options
        """
        try:
            fit_options = fit_options or {}
            self._log_operation("fit", {"fit_options": fit_options})

            self.campaign.fit(fit_options=fit_options)
            self._update_status(SessionStatus.FITTED)
        except Exception as e:
            self._update_status(SessionStatus.ERROR)
            self._log_operation("fit_error", {"error": str(e)})
            raise

    def suggest(
        self, m_batch: int = 1, acquisition: list[str] | None = None, **optim_kwargs
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """
        Generate next experiment suggestions.

        Args:
            m_batch: Number of experiments to suggest
            acquisition: Acquisition function(s) to use
            **optim_kwargs: Additional optimizer kwargs

        Returns:
            Tuple of (suggestions DataFrame, evaluation DataFrame)
        """
        try:
            self._update_status(SessionStatus.SUGGESTING)
            acquisition = acquisition or ["NEI"]

            self._log_operation("suggest", {"m_batch": m_batch, "acquisition": acquisition})

            X_suggest, eval_suggest = self.campaign.suggest(m_batch=m_batch, acquisition=acquisition, **optim_kwargs)

            self._update_status(SessionStatus.FITTED)  # Return to fitted state
            return X_suggest, eval_suggest
        except Exception as e:
            self._update_status(SessionStatus.ERROR)
            self._log_operation("suggest_error", {"error": str(e)})
            raise

    def evaluate(self, X: pd.DataFrame, return_std: bool = False, **kwargs) -> pd.DataFrame:
        """
        Evaluate (predict) on arbitrary points.

        Args:
            X: Points to evaluate
            return_std: If True, return mean + standard deviation; if False, return mean only
            **kwargs: Additional kwargs (ignored, kept for compatibility)

        Returns:
            Predictions DataFrame:
            - If return_std=False: Only {response} (pred) columns
            - If return_std=True: {response} (pred) and {response} (std) columns
        """
        try:
            self._update_status(SessionStatus.EVALUATING)
            self._log_operation("evaluate", {"n_points": len(X), "return_std": return_std})

            if not self.campaign.optimizer.is_fit:
                raise ValueError("Model must be fitted before evaluation")

            # Get predictions with 1-sigma interval to compute std
            # For normal distribution, 68.27% interval corresponds to ±1σ
            predictions = self.campaign.optimizer.predict(X, return_f_inv=True, PI_range=0.6827)

            # Extract mean predictions
            result = pd.DataFrame()
            for target_name in self.campaign.y_names:
                mean_col = f"{target_name} (pred)"
                result[mean_col] = predictions[mean_col]

                if return_std:
                    # Compute std from interval width: std = (ub - lb) / 2
                    lb_col = f"{target_name} lb"
                    ub_col = f"{target_name} ub"
                    std = (predictions[ub_col] - predictions[lb_col]) / 2
                    result[f"{target_name} (std)"] = std

            self._update_status(SessionStatus.FITTED)  # Return to fitted state

            return result
        except Exception as e:
            self._update_status(SessionStatus.ERROR)
            self._log_operation("evaluate_error", {"error": str(e)})
            raise

    def get_best(self) -> dict[str, Any]:
        """
        Get best results from campaign.

        Returns:
            Dictionary with best parameters and responses
        """
        try:
            if self.campaign.m_exp == 0:
                return {"X_best": None, "response_max": None, "n_experiments": 0, "message": "No data yet"}

            X_best = self.campaign.X_best
            response_max = self.campaign.response_max

            return {
                "X_best": X_best.to_dict(orient="records")[0] if len(X_best) > 0 else None,
                "response_max": response_max.to_dict() if isinstance(response_max, pd.Series) else float(response_max),
                "n_experiments": self.campaign.m_exp,
            }
        except Exception as e:
            self._log_operation("get_best_error", {"error": str(e)})
            raise

    def get_data(self) -> dict[str, Any]:
        """
        Export campaign data with metadata for analysis.

        Returns:
            Dictionary containing:
            - data: list of row dictionaries
            - n_rows: total number of experiments
            - n_columns: number of columns
            - columns: column names
            - iterations: unique iteration numbers
            - metadata: parameter and target names, fit status
        """
        if self.campaign.data.empty:
            return {
                "data": [],
                "n_rows": 0,
                "n_columns": 0,
                "columns": [],
                "iterations": [],
                "metadata": {
                    "parameter_names": list(self.campaign.X_space.X_names),
                    "target_names": self.campaign.y_names,
                    "is_fitted": False,
                    "n_parameters": len(self.campaign.X_space.X_names),
                    "n_targets": len(self.campaign.y_names),
                },
            }

        df = self.campaign.data

        return {
            "data": df.to_dict(orient="records"),
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": df.columns.tolist(),
            "iterations": sorted(df["Iteration"].unique().tolist()) if "Iteration" in df.columns else [],
            "metadata": {
                "parameter_names": list(self.campaign.X_space.X_names),
                "target_names": self.campaign.y_names,
                "is_fitted": self.campaign.optimizer.is_fit,
                "n_parameters": len(self.campaign.X_space.X_names),
                "n_targets": len(self.campaign.y_names),
            },
        }

    def get_diagnostics(self) -> dict[str, Any]:
        """
        Get model diagnostics and quality metrics.

        Returns:
            Dictionary containing:
            - session_id: Session identifier
            - is_fitted: Whether model is fitted
            - n_experiments: Total experiments in campaign
            - n_training_points: Training points (after NaN filtering)
            - n_parameters: Number of parameters
            - n_targets: Number of targets
            - surrogates: List of per-model diagnostics
            - is_multi_objective: Whether multi-objective optimization
            - hypervolume: Current hypervolume (if multi-objective)
            - n_pareto_points: Number of Pareto optimal points
            - current_iteration: Latest iteration
            - n_iterations: Total number of iterations
        """
        diagnostics = {
            "session_id": self.session_id,
            "is_fitted": False,
            "n_experiments": self.campaign.m_exp,
            "n_training_points": 0,
            "n_parameters": len(self.campaign.X_space.X_names),
            "n_targets": len(self.campaign.y_names),
            "surrogates": [],
            "is_multi_objective": self.campaign._is_mo,
            "hypervolume": None,
            "n_pareto_points": None,
            "current_iteration": None,
            "n_iterations": 0,
        }

        # Iteration info
        if not self.campaign.data.empty and "Iteration" in self.campaign.data.columns:
            iterations = self.campaign.data["Iteration"].dropna()
            if len(iterations) > 0:
                diagnostics["current_iteration"] = int(iterations.max())
                diagnostics["n_iterations"] = int(iterations.nunique())

        # Check if fitted
        if not self.campaign.optimizer.is_fit:
            return diagnostics

        diagnostics["is_fitted"] = True
        diagnostics["n_training_points"] = self.campaign.optimizer.X_train.shape[0]

        # Per-surrogate diagnostics
        for i in range(self.campaign.optimizer.n_response):
            surrogate_i = self.campaign.optimizer.surrogate[i]

            # Handle r2_score - convert NaN to None for JSON serialization
            r2_score = None
            if hasattr(surrogate_i, "r2_score"):
                r2_val = float(surrogate_i.r2_score)
                # Check if it's NaN or infinite
                import math

                if not (math.isnan(r2_val) or math.isinf(r2_val)):
                    r2_score = r2_val

            # Handle loss - convert NaN/inf to None
            loss = None
            if hasattr(surrogate_i, "loss"):
                loss_val = float(surrogate_i.loss)
                import math

                if not (math.isnan(loss_val) or math.isinf(loss_val)):
                    loss = loss_val

            surr_diag = {
                "response_name": self.campaign.y_names[i],
                "model_type": self.campaign.optimizer.surrogate_type[i],
                "r2_score": r2_score,
                "loss": loss,
                "n_training": (
                    surrogate_i.train_X.shape[0]
                    if hasattr(surrogate_i, "train_X")
                    else diagnostics["n_training_points"]
                ),
                "is_fit": surrogate_i.is_fit,
            }
            diagnostics["surrogates"].append(surr_diag)

        # Multi-objective metrics
        if diagnostics["is_multi_objective"] and self.campaign.m_exp > 0:
            try:
                import torch

                out_tensor = torch.tensor(self.campaign.out.values)
                if out_tensor.shape[1] > 1:
                    hv = self.campaign.optimizer.hypervolume(out_tensor)
                    diagnostics["hypervolume"] = float(hv)

                    pf = self.campaign.optimizer.pareto(out_tensor)
                    diagnostics["n_pareto_points"] = int(sum(pf))
            except Exception as e:
                # If hypervolume calculation fails, leave as None
                self._log_operation("diagnostics_warning", {"warning": f"Could not compute hypervolume: {str(e)}"})

        return diagnostics

    def get_history(self) -> dict[str, Any]:
        """
        Get iteration-by-iteration optimization history.

        Returns:
            Dictionary containing:
            - session_id: Session identifier
            - n_iterations: Number of iterations
            - iterations: List of per-iteration summaries
            - parameter_names: Parameter names
            - target_names: Target names
            - is_multi_objective: Whether multi-objective
            - total_experiments: Total experiments across all iterations
        """
        history_data = {
            "session_id": self.session_id,
            "n_iterations": 0,
            "iterations": [],
            "parameter_names": list(self.campaign.X_space.X_names),
            "target_names": self.campaign.y_names,
            "is_multi_objective": self.campaign._is_mo,
            "total_experiments": self.campaign.m_exp,
        }

        if self.campaign.data.empty:
            return history_data

        if "Iteration" not in self.campaign.data.columns:
            # No iteration tracking - return empty
            return history_data

        # Group by iteration
        df = self.campaign.data
        iterations = sorted(df["Iteration"].unique())
        history_data["n_iterations"] = len(iterations)

        import torch

        for iter_num in iterations:
            # Data up to and including this iteration
            df_cumulative = df[df["Iteration"] <= iter_num]
            df_current = df[df["Iteration"] == iter_num]

            iter_summary = {
                "iteration": int(iter_num),
                "n_experiments": len(df_current),
                "best_response": {},
                "mean_response": {},
                "hypervolume": None,
                "n_pareto_points": None,
            }

            # Best response values (cumulative)
            for target_name in self.campaign.y_names:
                if target_name in df_cumulative.columns:
                    best_val = df_cumulative[target_name].max()
                    iter_summary["best_response"][target_name] = float(best_val)

                    # Mean for current iteration
                    mean_val = df_current[target_name].mean()
                    iter_summary["mean_response"][target_name] = float(mean_val)

            # Multi-objective metrics (cumulative)
            if self.campaign._is_mo and self.campaign.optimizer.is_fit:
                try:
                    # Use pre-computed columns if available
                    if "Hypervolume (iter)" in df_cumulative.columns:
                        hv_col = df_cumulative[df_cumulative["Iteration"] == iter_num]["Hypervolume (iter)"]
                        if len(hv_col) > 0:
                            iter_summary["hypervolume"] = float(hv_col.iloc[0])

                    # Count Pareto points up to this iteration
                    out_cumulative = self.campaign.out.loc[df_cumulative.index]
                    if len(out_cumulative) > 0:
                        out_tensor = torch.tensor(out_cumulative.values)
                        pf = self.campaign.optimizer.pareto(out_tensor)
                        iter_summary["n_pareto_points"] = int(sum(pf))
                except Exception:
                    # If multi-objective calculations fail, leave as None
                    pass

            history_data["iterations"].append(iter_summary)

        return history_data

    def get_state_dict(self, object_type: str = "campaign") -> dict[str, Any]:
        """
        Get state dictionary from campaign or optimizer.

        Args:
            object_type: Type of object to export ("campaign" or "optimizer")

        Returns:
            Dictionary containing:
            - state: State dictionary from object.save_state()
            - object_type: Type of object exported

        Raises:
            ValueError: If object_type is not "campaign" or "optimizer"
        """
        if object_type not in ["campaign", "optimizer"]:
            raise ValueError(f"object_type must be 'campaign' or 'optimizer', got '{object_type}'")

        if object_type == "campaign":
            state = self.campaign.save_state()
        else:  # optimizer
            state = self.campaign.optimizer.save_state()

        return {"state": state, "object_type": object_type}

    # State management

    def save_state(self, directory: Path | None = None) -> dict[str, Any]:
        """
        Save session state to dictionary (and optionally to disk).

        Args:
            directory: Optional directory to save files (metadata, campaign state, history)

        Returns:
            Dictionary containing full session state
        """
        state = {
            "session_id": self.session_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": str(self.status),
            "campaign_state": self.campaign.save_state(),
            "history": self.history,
        }

        if directory:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)

            # Save metadata
            with open(directory / "metadata.json", "w") as f:
                metadata = {k: v for k, v in state.items() if k not in ["campaign_state", "history"]}
                json.dump(metadata, f, indent=2)

            # Save campaign state
            with open(directory / "campaign_state.json", "w") as f:
                json.dump(state["campaign_state"], f, indent=2)

            # Save history (JSON Lines format for append-only)
            with open(directory / "history.jsonl", "w") as f:
                for entry in self.history:
                    f.write(json.dumps(entry) + "\n")

            # Save data backup
            if not self.campaign.data.empty:
                self.campaign.data.to_csv(directory / "data.csv", index=False)

        return state

    @classmethod
    def load_state(cls, state: dict[str, Any]) -> "CampaignSession":
        """
        Load session from state dictionary.

        Args:
            state: State dictionary from save_state()

        Returns:
            Restored CampaignSession
        """
        # Restore Campaign
        campaign = Campaign.load_state(state["campaign_state"])

        # Create session
        session = cls(
            campaign=campaign, session_id=state["session_id"], name=state["name"], status=SessionStatus(state["status"])
        )

        # Restore timestamps
        session.created_at = datetime.fromisoformat(state["created_at"])
        session.updated_at = datetime.fromisoformat(state["updated_at"])
        session.history = state.get("history", [])

        return session

    @classmethod
    def load_from_directory(cls, directory: Path) -> "CampaignSession":
        """
        Load session from directory structure.

        Args:
            directory: Directory containing metadata.json, campaign_state.json, history.jsonl

        Returns:
            Restored CampaignSession
        """
        directory = Path(directory)

        # Load metadata
        with open(directory / "metadata.json") as f:
            metadata = json.load(f)

        # Load campaign state
        with open(directory / "campaign_state.json") as f:
            campaign_state = json.load(f)

        # Load history
        history = []
        history_file = directory / "history.jsonl"
        if history_file.exists():
            with open(history_file) as f:
                for line in f:
                    history.append(json.loads(line.strip()))

        # Combine into full state
        state = {**metadata, "campaign_state": campaign_state, "history": history}

        return cls.load_state(state)

    def to_dict(self) -> dict[str, Any]:
        """Get session metadata as dictionary (without full campaign state)."""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": str(self.status),
            "n_experiments": self.campaign.m_exp,
            "n_parameters": len(self.campaign.X_space.X_names),
            "n_targets": len(self.campaign.y_names) if hasattr(self.campaign, "y_names") else 0,
        }

    def __repr__(self) -> str:
        return (
            f"CampaignSession(id={self.session_id[:8]}, name={self.name}, status={self.status},"
            f" n_exp={self.campaign.m_exp})"
        )
