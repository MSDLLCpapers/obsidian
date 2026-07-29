"""Characterization evaluator for level set estimation"""

import math
import numbers

import numpy as np
import pandas as pd
from scipy.stats import norm

from obsidian.parameters.targets import resolve_target_names

from .campaign import Campaign

# Standard confidence ladder used by classify_confidence_levels and
# summarize_confidence. ``k=0`` recovers the mean test; the others are the
# one-sided multipliers for 70% and 95% prediction intervals.
_CONFIDENCE_K_VALUES = {
    "mean": 0.0,
    "70": float(norm.ppf(0.85)),  # ≈ 1.04
    "95": float(norm.ppf(0.975)),  # ≈ 1.96
}


class CharacterizationEvaluator:
    """Evaluate classification performance for characterization campaigns.

    Classifies regions of parameter space based on GP posterior predictions and thresholds.
    Optionally computes confusion matrix and Jaccard when ground truth is available.

    Args:
        campaign: Fitted campaign with thresholds set on targets
        seed: Random seed for Sobol sampling
    """

    def __init__(self, campaign: Campaign, seed: int | None = None):
        self.campaign = campaign
        self.optimizer = campaign.optimizer
        if seed is None:
            seed = campaign.seed
        self.seed = seed

        if not self.optimizer.is_fit:
            raise ValueError("Campaign optimizer must be fitted before evaluation")

        # Validate at construction time against current state
        self._validate_targets()

    @property
    def targets(self):
        """Active (non-tracking-only) targets with thresholds. Re-evaluated on each access."""
        return [t for t in self.campaign.target if not t.tracking_only and t.threshold is not None]

    def _validate_targets(self):
        active = self.targets
        if not active:
            raise ValueError("At least one target must have a threshold set for characterization")
        # tracking_only targets without thresholds are silently ignored — that is expected

    def _resolve_targets(self, target_names: list[str] | None):
        """Resolve a target_names filter against the active targets.

        ``None`` means "all active targets". Unknown names raise so a typo
        doesn't silently produce an empty result. Returns the targets in the
        order they appear in ``self.targets`` so per-target indexing into
        ``_get_posterior`` outputs stays aligned.
        """
        return resolve_target_names(
            self.targets,
            target_names,
            require_thresholds=False,  # self.targets already filtered for thresholds
            drop_tracking_only=False,  # ditto for tracking_only
        )

    @staticmethod
    def _validate_PI_range(PI_range: float) -> None:
        """Range-check PI_range so norm.ppf produces a finite confidence multiplier."""
        if not (isinstance(PI_range, numbers.Number) and 0.0 <= float(PI_range) < 1.0):
            raise ValueError(f"PI_range must be a number in [0, 1), got {PI_range!r}")

    def _normalize_dim_weights(self, dim_weights: dict | None) -> dict[str, float]:
        """Validate user-provided dim_weights and fill defaults for missing dims.

        Returns a dict keyed by every continuous parameter name in X_space.
        Missing keys default to 1.0; unknown keys (typos or discrete params)
        raise. Zero/negative/non-finite/non-numeric weights also raise — the
        binary search would otherwise divide by zero or invert volume signs.
        """
        cont_names = [p.name for p in self.campaign.designer.X_space.X_cont]
        if dim_weights is None:
            return {p: 1.0 for p in cont_names}

        unknown = set(dim_weights) - set(cont_names)
        if unknown:
            raise ValueError(
                f"dim_weights has unknown parameters {sorted(unknown)}; " f"valid continuous parameters: {cont_names}"
            )
        bad_weights = {
            p: w
            for p, w in dim_weights.items()
            if not (isinstance(w, numbers.Number) and math.isfinite(float(w)) and float(w) > 0)
        }
        if bad_weights:
            raise ValueError(f"dim_weights must be finite positive numbers, got {bad_weights}")

        return {p: float(dim_weights.get(p, 1.0)) for p in cont_names}

    def _classify_at_k_values(
        self,
        X: pd.DataFrame | int,
        k_values: dict[str, float],
        target_names: list[str] | None = None,
    ) -> dict:
        """Per-target pass / fail boolean masks at a set of k-σ thresholds.

        The single shared primitive behind ``classify_points``,
        ``classify_confidence_levels`` and ``summarize_confidence``. Each public
        method calls this once and reshapes the masks into its own return form.

        For each selected target and each ``(label, k)`` in ``k_values``:
            pass[label]  =  (mean_signed - k·σ) >= τ_signed
            fail[label]  =  (mean_signed + k·σ) <  τ_signed

        Args:
            X: Points to classify (DataFrame) or number of Sobol samples (int).
            k_values: Map of label → k. ``k=0`` recovers the mean test.
            target_names: Restrict to a subset of active targets.

        Returns:
            ``{"X": <DataFrame>, "selected": [Target, ...],
               "per_target": {name: {"pass": {label: mask}, "fail": {label: mask}}}}``
        """
        if isinstance(X, int):
            X = self._generate_sobol_samples(X)

        means_unsigned, stds = self._get_posterior(X)
        active = self.targets
        selected = self._resolve_targets(target_names)
        active_index = {t.name: i for i, t in enumerate(active)}

        per_target: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for target in selected:
            i = active_index[target.name]
            mean_signed, signed_threshold, std = self._prepare_signed_data(target, means_unsigned[:, i], stds[:, i])
            pass_masks = {label: (mean_signed - k * std) >= signed_threshold for label, k in k_values.items()}
            fail_masks = {label: (mean_signed + k * std) < signed_threshold for label, k in k_values.items()}
            per_target[target.name] = {"pass": pass_masks, "fail": fail_masks}

        return {"X": X, "selected": selected, "per_target": per_target}

    def classify_points(
        self,
        X: pd.DataFrame | int,
        PI_range: float = 0.7,
        return_samples: bool = False,
        target_names: list[str] | None = None,
    ) -> dict:
        """Classify points based on GP predictions and thresholds.

        Args:
            X: Points to classify (pd.DataFrame) OR number of Sobol samples (int)
            PI_range: Prediction interval coverage (0.7 or 0.95)
            return_samples: If True, include 'X_samples' and 'X_feasible' in result.
                Use this when planning to compute multiple hypercubes to avoid
                redundant GP predictions.
            target_names: Restrict classification (and the Joint key) to this
                subset of active targets. Defaults to all active targets.

        Returns:
            dict keyed by target names, plus 'Joint' key if multiple targets.
            Each value is a dict containing pos_frac, neg_frac, classified_frac, pred_mask.
            Optional keys: 'X_samples', 'X_feasible' (if return_samples=True)
        """
        self._validate_PI_range(PI_range)
        k_sigma = float(norm.ppf((1 + PI_range) / 2))
        primitive = self._classify_at_k_values(X, {"k": k_sigma}, target_names)
        X_eval = primitive["X"]
        selected = primitive["selected"]

        target_results = []
        result = {}
        for target in selected:
            masks = primitive["per_target"][target.name]
            classification = self._classification_metrics(masks["pass"]["k"], masks["fail"]["k"])
            target_results.append(classification)
            result[target.name] = classification

        if len(target_results) > 1:
            result["Joint"] = self._get_joint_classification_metrics(target_results)

        if return_samples:
            pred_mask = self._get_joint_mask(result, selected)
            result["X_samples"] = X_eval
            result["X_feasible"] = X_eval[pred_mask]

        return result

    def classify_confidence_levels(
        self,
        X: pd.DataFrame | int,
        target_names: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Classify points into 4 confidence levels per active target.

        Levels (per cell, per target):
            0 — Fail:            mean fails the threshold
            1 — Uncertain Pass:  mean passes; 70% PI lower bound fails
            2 — Likely Pass:     70% PI lower bound passes; 95% fails
            3 — Confident Pass:  95% PI lower bound passes

        Consistent with ``classify_points`` by construction:
        ``pred_mask`` of ``classify_points(PI_range=0.7)`` equals ``levels >= 2``,
        and at ``PI_range=0.95`` it equals ``levels >= 3``.

        Args:
            X: Points to classify (pd.DataFrame) OR number of Sobol samples (int).
            target_names: Restrict to this subset of active targets. The Joint
                key (when present) is the elementwise min across the subset.
                Defaults to all active targets.

        Returns:
            dict keyed by target name. Each value is a length-N int array of
            levels in {0, 1, 2, 3}. Includes a ``"Joint"`` key (elementwise
            min across targets) when more than one active target exists.
        """
        primitive = self._classify_at_k_values(X, _CONFIDENCE_K_VALUES, target_names)
        selected = primitive["selected"]

        out: dict[str, np.ndarray] = {}
        for target in selected:
            passes = primitive["per_target"][target.name]["pass"]
            out[target.name] = passes["mean"].astype(int) + passes["70"].astype(int) + passes["95"].astype(int)

        if len(selected) > 1:
            stacked = np.stack([out[t.name] for t in selected], axis=0)
            out["Joint"] = stacked.min(axis=0)
        return out

    def summarize_confidence(self, X: pd.DataFrame | int) -> dict[str, dict[str, float]]:
        """Pass / fail / classified fractions across the standard CI ladder.

        Returns:
            ``{target_name: {"pass_mean", "pass_70", "pass_95",
                             "fail_70", "fail_95",
                             "classified_70", "classified_95"}}``
            with values in [0, 1]. Includes a ``"Joint"`` entry when there
            is more than one active target.
        """
        primitive = self._classify_at_k_values(X, _CONFIDENCE_K_VALUES)
        selected = primitive["selected"]
        per_target = primitive["per_target"]

        def _row(passes: dict[str, np.ndarray], fails: dict[str, np.ndarray]) -> dict[str, float]:
            return {
                "pass_mean": float(passes["mean"].mean()),
                "pass_70": float(passes["70"].mean()),
                "pass_95": float(passes["95"].mean()),
                "fail_70": float(fails["70"].mean()),
                "fail_95": float(fails["95"].mean()),
                "classified_70": float((passes["70"] | fails["70"]).mean()),
                "classified_95": float((passes["95"] | fails["95"]).mean()),
            }

        result: dict[str, dict[str, float]] = {
            t.name: _row(per_target[t.name]["pass"], per_target[t.name]["fail"]) for t in selected
        }

        if len(selected) > 1:
            joint_pass = {
                label: np.all([per_target[t.name]["pass"][label] for t in selected], axis=0)
                for label in ("mean", "70", "95")
            }
            joint_fail = {
                label: np.any([per_target[t.name]["fail"][label] for t in selected], axis=0) for label in ("70", "95")
            }
            result["Joint"] = _row(joint_pass, joint_fail)

        return result

    def evaluate_with_ground_truth(self, X: pd.DataFrame, y_true: np.ndarray, PI_range: float = 0.7) -> dict:
        """Compute confusion matrix and Jaccard index for benchmarking.

        Args:
            X: Points to evaluate
            y_true: Ground truth values, shape (n_points,) or (n_points, n_active_targets)
            PI_range: Prediction interval coverage

        Returns:
            dict with per-target and joint Jaccard scores and confusion matrices
        """
        self._validate_PI_range(PI_range)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        expected_shape = (len(X), len(self.targets))
        if y_true.shape != expected_shape:
            raise ValueError(
                f"y_true has shape {y_true.shape}, expected {expected_shape} (n_points, n_active_targets)."
            )

        k_sigma = float(norm.ppf((1 + PI_range) / 2))
        means_unsigned, stds = self._get_posterior(X)

        target_results = []
        result = {}
        for i, target in enumerate(self.targets):
            mean_signed, signed_threshold, std = self._prepare_signed_data(target, means_unsigned[:, i], stds[:, i])
            classification = self._classifier(mean_signed, std, signed_threshold, k_sigma)
            pred_mask = classification["pred_mask"]

            # Use >= 0 so the boundary is assigned to Pass, matching the convention in the code
            true_mask = (y_true[:, i] - target.threshold) * target.multiplier >= 0
            confusion = self._get_confusion_matrix(pred_mask, true_mask)
            target_result = {
                "jaccard": confusion["jaccard"],
                "confusion_matrix": {k: confusion[k] for k in ("TP", "TN", "FP", "FN")},
                "pred_mask": pred_mask,
                "true_mask": true_mask,
            }
            target_results.append(target_result)
            result[target.name] = target_result

        if len(target_results) > 1:
            joint_pred_mask = np.all([r["pred_mask"] for r in target_results], axis=0)
            joint_true_mask = np.all([r["true_mask"] for r in target_results], axis=0)
            joint_confusion = self._get_confusion_matrix(joint_pred_mask, joint_true_mask)
            result["Joint"] = {
                "jaccard": joint_confusion["jaccard"],
                "confusion_matrix": {k: joint_confusion[k] for k in ("TP", "TN", "FP", "FN")},
            }

        return result

    def plan_sample_size(
        self,
        pilot_ratio: float = 0.1,
        epsilon: float = 0.01,
        z: float = 1.96,
        max_samples: int = 100_000,
    ) -> int:
        """Plan appropriate sample size using Bernoulli variance on pilot samples.

        Args:
            pilot_ratio: Fraction of max_samples to use as pilot
            epsilon: Desired precision for fraction estimates
            z: Z-score for confidence level (1.96 for 95%)
            max_samples: Maximum allowed samples

        Returns:
            Recommended sample size (at least n_pilot, at most max_samples)
        """
        n_pilot = max(100, int(max_samples * pilot_ratio))
        X_pilot = self._generate_sobol_samples(n_pilot)
        means_unsigned, stds = self._get_posterior(X_pilot)

        max_v_bar = 0.0
        tiny = 1e-12

        for i, target in enumerate(self.targets):
            mean_signed, signed_threshold, std = self._prepare_signed_data(target, means_unsigned[:, i], stds[:, i])
            p_pilot = np.where(
                std <= tiny,
                (mean_signed > signed_threshold).astype(float),
                norm.cdf((mean_signed - signed_threshold) / std),
            )
            v_bar = float(np.mean(p_pilot * (1.0 - p_pilot)))
            max_v_bar = max(max_v_bar, v_bar)

        if max_v_bar > 0:
            N_req = math.ceil((z * math.sqrt(max_v_bar) / epsilon) ** 2)
            N_req = min(N_req, max_samples)
        else:
            N_req = n_pilot

        return max(n_pilot, N_req)

    def compute_largest_hypercube(
        self,
        X: pd.DataFrame | int,
        PI_range: float = 0.7,
        center: dict[str, float] | pd.Series | None = None,
        dim_weights: dict[str, float] | None = None,
        fixed_dims: dict[str, float] | None = None,
        tolerance: float = 1e-3,
        max_iter: int = 100,
    ) -> dict:
        """Compute largest inscribed hypercube within feasible region.

        Finds the largest axis-aligned hypercube that fits entirely within the
        Pass region. For continuous parameters only, finds single hypercube.
        For problems with discrete parameters, enumerates all discrete combinations
        and returns the one yielding the largest hypercube.

        Args:
            X: Evaluation points (DataFrame) or number of Sobol samples (int).
                Recommend 1000*d samples for accuracy.
            PI_range: Prediction interval for classification (0.7 or 0.95).
            center: Optional fixed center. If None, uses centroid of feasible region.
                Dict format: {param_name: value} or pd.Series
            dim_weights: Relative weights for each dimension. If None, all dimensions
                weighted equally. Dict format: {param_name: weight}.
                Example: {'x': 2, 'y': 1} means x can be twice as wide as y.
            fixed_dims: Parameters to fix at specific values.
                Dict format: {param_name: value}
            tolerance: Convergence tolerance for binary search.
            max_iter: Max iterations for optimization.

        Returns:
            dict with:
                - 'center': Center point (pd.Series with param names)
                - 'volume': Hypercube volume in parameter space
                - 'bounds': Dict with 'lower' and 'upper' (pd.Series)
                - 'fixed_dims': Dict of user-specified fixed continuous dimensions
                    (empty if none were fixed).
                - 'categorical_values': Dict of categorical parameter assignments,
                    user-specified or chosen by enumeration (empty if the campaign
                    has no categorical parameters).
                - 'n_corners_feasible': Number of feasible corners checked
                - 'convergence_iters': Number of iterations to converge

        Raises:
            ValueError: If no feasible hypercube exists or conflicting constraints.
        """
        self._validate_PI_range(PI_range)
        dim_weights = self._normalize_dim_weights(dim_weights)

        # 1. Generate/validate samples and classify
        if isinstance(X, int):
            X_eval = self._generate_sobol_samples(X)
        else:
            X_eval = X

        classification = self.classify_points(X_eval, PI_range)
        pred_mask = self._get_joint_mask(classification)
        X_feasible = X_eval[pred_mask]
        X_infeasible = X_eval[~pred_mask]

        if len(X_feasible) < 10:
            raise ValueError(
                f"Too few feasible points ({len(X_feasible)}) for hypercube computation. "
                "Try increasing sample size, lowering PI_range, or adjusting thresholds."
            )

        # 2. Setup parameter space
        param_space = self.campaign.designer.X_space
        discrete_params = param_space.X_discrete

        if fixed_dims is None:
            fixed_dims = {}

        # Apply fixed-discrete filtering up front so neither path sees samples
        # from non-pinned categories.
        discrete_names = {p.name for p in discrete_params}
        fixed_discrete = {k: v for k, v in fixed_dims.items() if k in discrete_names}
        if fixed_discrete:
            mask_feas = pd.Series(True, index=X_feasible.index)
            mask_infeas = pd.Series(True, index=X_infeasible.index)
            for name, value in fixed_discrete.items():
                mask_feas &= X_feasible[name] == value
                mask_infeas &= X_infeasible[name] == value
            X_feasible = X_feasible[mask_feas]
            X_infeasible = X_infeasible[mask_infeas]

            if len(X_feasible) < 10:
                raise ValueError(
                    f"Too few feasible points ({len(X_feasible)}) after applying fixed_dims "
                    f"{fixed_discrete} for hypercube computation. "
                    "Try increasing sample size, lowering PI_range, or adjusting thresholds."
                )

        # Check if any discrete params are free (not fixed)
        free_discrete = [p for p in discrete_params if p.name not in fixed_dims]

        # Handle discrete parameters by enumerating combinations
        if len(free_discrete) > 0:
            return self._compute_hypercube_with_discrete(
                X_feasible=X_feasible,
                X_infeasible=X_infeasible,
                discrete_params=free_discrete,
                center=center,
                dim_weights=dim_weights,
                fixed_dims=fixed_dims,
                tolerance=tolerance,
                max_iter=max_iter,
            )

        # No free discrete params - single call
        return self._compute_hypercube_continuous(
            X_feasible=X_feasible,
            X_infeasible=X_infeasible,
            center=center,
            dim_weights=dim_weights,
            fixed_dims=fixed_dims,
            tolerance=tolerance,
            max_iter=max_iter,
        )

    def _compute_hypercube_continuous(
        self,
        X_feasible: pd.DataFrame,
        X_infeasible: pd.DataFrame,
        center: dict[str, float] | pd.Series | None,
        dim_weights: dict[str, float],
        fixed_dims: dict[str, float] | None,
        tolerance: float,
        max_iter: int,
    ) -> dict:
        """Compute hypercube for continuous parameters only.

        `fixed_dims` may carry both continuous and categorical pinnings (the
        discrete path merges its chosen category into the same dict). We split
        them here so the result distinguishes the two concepts.
        """
        param_space = self.campaign.designer.X_space
        # Only continuous parameter names
        param_names = [p.name for p in param_space.X_cont]
        param_bounds = {p.name: (float(p.min), float(p.max)) for p in param_space.X_cont}

        # Resolve free parameters (exclude fixed_dims)
        if fixed_dims is None:
            fixed_dims = {}
        free_params = [p for p in param_names if p not in fixed_dims]

        if len(free_params) == 0:
            raise ValueError("All parameters are fixed. No hypercube to compute.")

        # Split incoming fixed_dims into continuous vs categorical by checking
        # against the continuous-parameter set.
        continuous_fixed_dims = {k: v for k, v in fixed_dims.items() if k in param_names}
        categorical_values = {k: v for k, v in fixed_dims.items() if k not in param_names}

        if center is None:
            center_series = self._compute_feasible_centroid(X_feasible, param_names, continuous_fixed_dims)
        else:
            center_series = self._validate_center(center, param_names, continuous_fixed_dims)

        # dim_weights was normalized upstream to cover every continuous param;
        # restrict to the free ones so the binary search ignores fixed dims.
        dim_weights = {p: dim_weights[p] for p in free_params}

        result = self._binary_search_hypercube(
            center=center_series,
            X_infeasible=X_infeasible,
            free_params=free_params,
            fixed_dims=continuous_fixed_dims,
            categorical_values=categorical_values,
            param_bounds=param_bounds,
            dim_weights=dim_weights,
            tolerance=tolerance,
            max_iter=max_iter,
        )

        return result

    def _compute_hypercube_with_discrete(
        self,
        X_feasible: pd.DataFrame,
        X_infeasible: pd.DataFrame,
        discrete_params: list,
        center: dict[str, float] | pd.Series | None,
        dim_weights: dict[str, float],
        fixed_dims: dict[str, float] | None,
        tolerance: float,
        max_iter: int,
    ) -> dict:
        """Enumerate discrete combinations and find the one with largest hypercube.

        Assumes discrete_params only contains free (unfixed) discrete parameters.
        """
        import itertools

        if fixed_dims is None:
            fixed_dims = {}

        # Get all discrete parameter values
        discrete_combos = []
        discrete_names = []
        for param in discrete_params:
            discrete_names.append(param.name)
            discrete_combos.append(param.categories)

        # Enumerate all combinations
        best_result = None
        best_volume = -1

        for combo in itertools.product(*discrete_combos):
            discrete_dict = dict(zip(discrete_names, combo))

            # Filter X_feasible to only points matching this discrete combination
            mask_feas = pd.Series(True, index=X_feasible.index)
            mask_infeas = pd.Series(True, index=X_infeasible.index)
            for param_name, param_value in discrete_dict.items():
                mask_feas &= X_feasible[param_name] == param_value
                mask_infeas &= X_infeasible[param_name] == param_value

            X_feasible_combo = X_feasible[mask_feas]
            X_infeasible_combo = X_infeasible[mask_infeas]

            # Skip if too few feasible points for this combination
            if len(X_feasible_combo) < 10:
                continue

            # Merge discrete values into fixed_dims for this run before calling
            # _compute_hypercube_continuous(). In the returned result, any
            # non-continuous fixed values are exposed under 'categorical_values'
            # rather than the result's 'fixed_dims' field.
            fixed_dims_combo = {**fixed_dims, **discrete_dict}

            try:
                result = self._compute_hypercube_continuous(
                    X_feasible=X_feasible_combo,
                    X_infeasible=X_infeasible_combo,
                    center=center,
                    dim_weights=dim_weights,
                    fixed_dims=fixed_dims_combo,
                    tolerance=tolerance,
                    max_iter=max_iter,
                )

                if result["volume"] > best_volume:
                    best_volume = result["volume"]
                    best_result = result
            except ValueError:
                # This combination didn't work, skip it
                continue

        if best_result is None:
            raise ValueError(
                "No feasible hypercube found for any discrete parameter combination. "
                "Try increasing sample size, lowering PI_range, or adjusting thresholds."
            )

        return best_result

    # Internal helpers
    def _generate_sobol_samples(self, n_samples: int) -> pd.DataFrame:
        """Generate n_samples Sobol points decoded into the parameter space."""
        return self.campaign.designer.initialize(m_initial=n_samples, method="Sobol")

    def _get_posterior(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Get mean predictions and std for all active targets.

        Uses PI_range=0.95 as reference to back out the GP posterior std.

        Returns:
            means_unsigned: (n_points, n_active_targets) in original space
            stds: (n_points, n_active_targets), always positive
        """
        z_ref = norm.ppf(0.975)  # 1.96 for 95% PI
        preds: pd.DataFrame = self.optimizer.predict(X, return_f_inv=True, PI_range=0.95) # type: ignore

        active = self.targets
        n_points = len(X)
        means = np.zeros((n_points, len(active)))
        stds = np.zeros((n_points, len(active)))

        for i, target in enumerate(active):
            name = target.name
            means[:, i] = preds[f"{name} (pred)"].values
            lb = preds[f"{name} lb"].values
            ub = preds[f"{name} ub"].values
            stds[:, i] = np.abs(ub - lb) / (2.0 * z_ref)

        return means, stds

    @staticmethod
    def _prepare_signed_data(target, mean_unsigned: np.ndarray, std: np.ndarray):
        """Apply sign convention so classification is always mean_signed >= signed_threshold."""
        sign = target.multiplier  # 1 for max, -1 for min
        return mean_unsigned * sign, target.threshold * sign, std

    @staticmethod
    def _classification_metrics(pos_def: np.ndarray, neg_def: np.ndarray) -> dict:
        """Bundle pre-computed pass / fail masks into the classify_points return shape.

        ``pos_def`` and ``neg_def`` are mutually exclusive by construction (the
        threshold uses a half-open / half-closed partition); the boundary is
        assigned to "Pass" by convention.
        """
        N = pos_def.size
        pos_frac = float(np.sum(pos_def)) / N
        neg_frac = float(np.sum(neg_def)) / N
        return {
            "pred_mask": pos_def,
            "neg_def_mask": neg_def,
            "pos_frac": pos_frac,
            "neg_frac": neg_frac,
            "classified_frac": pos_frac + neg_frac,
            "pos_count": int(np.sum(pos_def)),
            "neg_count": int(np.sum(neg_def)),
        }

    @classmethod
    def _classifier(
        cls,
        mean_signed: np.ndarray,
        std: np.ndarray,
        signed_threshold: float,
        k_sigma: float,
    ) -> dict:
        """Classify points at k_sigma confidence level (used by ground-truth eval)."""
        pos_def = (mean_signed - k_sigma * std) >= signed_threshold
        neg_def = (mean_signed + k_sigma * std) < signed_threshold
        return cls._classification_metrics(pos_def, neg_def)

    @staticmethod
    def _get_confusion_matrix(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict:
        TP = int(np.sum(pred_mask & true_mask))
        TN = int(np.sum(~pred_mask & ~true_mask))
        FP = int(np.sum(pred_mask & ~true_mask))
        FN = int(np.sum(~pred_mask & true_mask))
        union = TP + FP + FN
        return {"jaccard": TP / union if union > 0 else float("nan"), "TP": TP, "TN": TN, "FP": FP, "FN": FN}

    @staticmethod
    def _get_joint_classification_metrics(target_classifications: list[dict]) -> dict:
        """Joint classification: pass = ALL targets pass, fail = ANY target fails."""
        pass_masks = [c["pred_mask"] for c in target_classifications]
        fail_masks = [c["neg_def_mask"] for c in target_classifications]

        joint_pass = np.all(pass_masks, axis=0)
        joint_fail = np.any(fail_masks, axis=0)
        joint_classified = joint_pass | joint_fail

        N = joint_pass.size
        return {
            "pred_mask": joint_pass,
            "pos_frac": float(np.sum(joint_pass)) / N,
            "neg_frac": float(np.sum(joint_fail)) / N,
            "classified_frac": float(np.sum(joint_classified)) / N,
            "pos_count": int(np.sum(joint_pass)),
            "neg_count": int(np.sum(joint_fail)),
            "classified_count": int(np.sum(joint_classified)),
        }

    # Hypercube computation helpers
    def _get_joint_mask(self, classification: dict, selected: list | None = None) -> np.ndarray:
        """Extract joint pass mask for the targets actually classified.

        ``selected`` is the list of targets whose results live in
        ``classification``. Defaults to ``self.targets`` for callers that
        classified the full active set.
        """
        if "Joint" in classification:
            return classification["Joint"]["pred_mask"]
        targets = selected if selected is not None else self.targets
        return classification[targets[0].name]["pred_mask"]

    def _compute_feasible_centroid(
        self, X_feasible: pd.DataFrame, param_names: list[str], fixed_dims: dict[str, float]
    ) -> pd.Series:
        """Compute centroid of feasible region as default center."""
        center_series = pd.Series({param: X_feasible[param].mean() for param in param_names})

        # Apply fixed_dims
        for param, value in fixed_dims.items():
            center_series[param] = value

        return center_series

    def _validate_center(
        self, center: dict[str, float] | pd.Series, param_names: list[str], fixed_dims: dict[str, float]
    ) -> pd.Series:
        """Validate and convert center to pd.Series."""
        if isinstance(center, dict):
            center_series = pd.Series(center)
        else:
            center_series = center.copy()

        # Check all parameters are present
        missing = set(param_names) - set(center_series.index)
        if missing:
            raise ValueError(f"Center missing parameters: {missing}")

        # Apply fixed_dims
        for param, value in fixed_dims.items():
            center_series[param] = value

        return center_series

    def _binary_search_hypercube(
        self,
        center: pd.Series,
        X_infeasible: pd.DataFrame,
        free_params: list[str],
        fixed_dims: dict[str, float],
        categorical_values: dict,
        param_bounds: dict[str, tuple[float, float]],
        dim_weights: dict[str, float],
        tolerance: float,
        max_iter: int,
    ) -> dict:
        """Binary search to find largest feasible hypercube.

        Uses conservative check: a hypercube is feasible iff no infeasible sample
        falls inside it. This guarantees the returned hypercube has no known
        infeasible points within its bounds.

        Args:
            fixed_dims: User-fixed continuous dimensions. Reported in result; also
                needs tolerance-based filtering against Sobol samples since Sobol
                points won't match the fixed value exactly.
            categorical_values: Categorical parameter assignments (user-specified
                or chosen by enumeration). Reported in result; filtering was
                already applied exactly upstream.
        """
        # Compute max possible half-widths (to parameter space boundaries)
        r_max_per_dim = {}
        for param in free_params:
            r_max_per_dim[param] = min(
                center[param] - param_bounds[param][0],
                param_bounds[param][1] - center[param],
            )

        # Pre-filter infeasible points to those near the continuous fixed-dim slice.
        # A zero-width fixed dim only "contains" points at exactly fval, but Sobol
        # samples are continuous and won't match exactly. Use a tolerance of
        # ~2x typical Sobol spacing per axis (N^(-1/d) in a d-dim unit cube).
        X_infeas_filtered = X_infeasible
        if fixed_dims:
            d_full = len(param_bounds)
            n_total = max(len(X_infeasible), 1)
            for fname, fval in fixed_dims.items():
                if len(X_infeas_filtered) == 0 or fname not in X_infeas_filtered.columns:
                    continue
                lo, hi = param_bounds[fname]
                fixed_tol = 2.0 * (hi - lo) * n_total ** (-1.0 / d_full)
                X_infeas_filtered = X_infeas_filtered[(X_infeas_filtered[fname] - fval).abs() <= fixed_tol]

        X_infeas_arr = (
            X_infeas_filtered[free_params].values if len(X_infeas_filtered) > 0 else np.empty((0, len(free_params)))
        )
        center_arr = np.array([center[p] for p in free_params])
        weights_arr = np.array([dim_weights[p] for p in free_params])

        # Binary search on scale factor: half_widths = scale * dim_weights
        scale_lower = 0.0
        scale_upper = min(r_max_per_dim[p] / dim_weights[p] for p in free_params)

        iter_count = 0
        for iter_count in range(max_iter):
            scale_mid = (scale_lower + scale_upper) / 2.0

            if self._check_hypercube_feasible(center_arr, scale_mid, weights_arr, X_infeas_arr):
                scale_lower = scale_mid
            else:
                scale_upper = scale_mid

            if scale_upper - scale_lower < tolerance:
                break

        final_scale = scale_lower
        final_half_widths = {p: final_scale * dim_weights[p] for p in free_params}

        # Build result dictionary
        half_widths_series = pd.Series(final_half_widths)
        all_params = center.index.tolist()
        full_half_widths = pd.Series({p: half_widths_series.get(p, 0.0) for p in all_params})

        lower_bounds = center - full_half_widths
        upper_bounds = center + full_half_widths

        volume = float(np.prod([final_half_widths[p] * 2 for p in free_params]))

        # Count corners
        d = len(free_params)
        n_corners = min(2**d, 1000) if d > 15 else 2**d

        return {
            "center": center,
            "volume": volume,
            "bounds": {"lower": lower_bounds, "upper": upper_bounds},
            "fixed_dims": dict(fixed_dims),
            "categorical_values": dict(categorical_values),
            "n_corners_feasible": n_corners,
            "convergence_iters": iter_count + 1,
        }

    @staticmethod
    def _check_hypercube_feasible(
        center: np.ndarray,
        scale: float,
        weights: np.ndarray,
        X_infeasible: np.ndarray,
    ) -> bool:
        """Check if hypercube contains no known infeasible sample.

        Conservative check: hypercube centered at `center` with per-dimension
        half-widths `scale * weights` is feasible iff no point in `X_infeasible`
        lies inside it. Boundary points (axis-distance exactly equal to a
        half-width) count as inside, so a hypercube that merely touches an
        infeasible point is rejected.
        """
        if X_infeasible.shape[0] == 0:
            return True

        half_widths = scale * weights  # shape (d,)
        # Distance along each axis from center
        dists = np.abs(X_infeasible - center)  # shape (n, d)
        # Point is inside hypercube iff all axis-distances <= half_widths
        inside = np.all(dists <= half_widths, axis=1)
        return not inside.any()
