import math
from typing import Any
from warnings import warn

import numpy as np
import torch
from botorch.acquisition import MCAcquisitionFunction
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.utils.transforms import t_batch_mode_transform

from ..config import DEFAULT_DEVICE, TORCH_DTYPE
from ..parameters.targets import Target
from ..rng import derive_seed, get_new_seed
from .utils import ParserContext, default_sampler

_SYNC_WARNING = (
    "Note: Synchronized beta across objectives is enabled. Empirically, synced beta performs worse "
    "than independent per-objective beta sampling. If you are willing to provide a datapoint "
    "to support or dispute this observation, please contact the developers."
)


def _parse_thresholds(
    threshold_spec: float | list[float] | dict[str, float] | None, targets: list[Target]
) -> list[float]:
    """
    Resolve threshold specification into a list of transformed threshold values.

    Supported formats:
    1. number - for single-objective (len(targets) == 1)
    2. list - for multi-objective, must have len(list) == len(targets)
    3. dict {target_name: value} - keys must exactly match all target names
       (warns if single-objective since redundant)
    4. None - use target.threshold from all Target objects (encouraged)

    Args:
        threshold_spec: Threshold value(s) from hyperparameters, or None
        targets: List of Target objects from context

    Returns:
        List of transformed threshold values, one per target, in same order as targets

    Raises:
        ValueError: If threshold specification is invalid or incompatible with targets
    """
    # None
    if threshold_spec is None:
        result = []
        for target in targets:
            threshold = target.get_threshold(transformed=True)
            if threshold is None:
                raise ValueError(
                    f"No threshold specified for target '{target.name}' and target.threshold is not set. "
                    "Either pass threshold in hyperparameters or set threshold on Target object."
                )
            result.append(threshold)
        return result

    warn(
        "Explicit threshold specification in hyperparameters is not recommended. For better modularity and "
        "encapsulation, please set threshold values directly on Target objects. This allows the acquisition function "
        "to automatically use the correct thresholds without needing to pass them through hyperparameters, and keeps "
        "the acquisition function decoupled from specific target names or ordering.",
        UserWarning,
    )

    # Dict {target_name: threshold_value}
    if isinstance(threshold_spec, dict):
        # Warn if single-objective (redundant)
        if len(targets) == 1:
            warn(
                "Using dict for single-objective threshold is redundant. Use a number instead.",
                UserWarning,
                stacklevel=2,
            )

        # Validate: keys must exactly match all target names
        target_names = {t.name for t in targets}
        spec_names = set(threshold_spec.keys())

        if spec_names != target_names:
            raise ValueError(
                f"Threshold dict keys {spec_names} do not exactly match target names {target_names}. "
                "All targets must have thresholds specified."
            )

        # Return in same order as targets list, transformed
        return [t.transform_f(threshold_spec[t.name]).iloc[0] for t in targets]

    # List of numbers
    if isinstance(threshold_spec, list):
        if len(threshold_spec) != len(targets):
            raise ValueError(
                f"Threshold list has {len(threshold_spec)} values but {len(targets)} targets. "
                "Must provide exactly one threshold per target."
            )
        # Transform all thresholds
        return [target.transform_f(t_val).iloc[0] for t_val, target in zip(threshold_spec, targets)]

    # Single number (single-objective)
    if isinstance(threshold_spec, (int, float)):
        if len(targets) != 1:
            raise ValueError(
                f"Scalar threshold {threshold_spec} is only valid for single-target characterization, "
                f"but got {len(targets)} targets. Provide a list or dict with one threshold per target."
            )
        # Transform the threshold
        return [targets[0].transform_f(threshold_spec).iloc[0]]

    raise ValueError(f"Invalid threshold type: {type(threshold_spec).__name__}. Must be number, list, dict, or None.")


def chi2_sampler(
    n: int = 1,
    generator: torch.Generator | None = None,
    device: torch.device | str = DEFAULT_DEVICE,
    dtype=TORCH_DTYPE,
) -> torch.Tensor:
    """
    Sample from chi-squared distribution with 2 degrees of freedom.

    Args:
        n: Number of samples to draw
        generator: Random number generator for reproducibility
        device: Device to create tensor on
        dtype: Data type of output tensor

    Returns:
        Tensor of shape (n,) if n > 1, scalar if n == 1
    """
    shape = (n,) if n > 1 else ()
    U = torch.rand(shape, generator=generator, device=device, dtype=dtype).clamp_min(1e-12)
    return -2.0 * torch.log(U)


def _sample_beta(
    m_batch: int,
    num_targets: int,
    sync_beta: bool,
    generator: torch.Generator | None = None,
    device: torch.device | str = DEFAULT_DEVICE,
) -> torch.Tensor:
    """Sample beta values for single-objective (using sync_beta) or multi-objective randomized straddle.

    Returns β ~ χ²₂ directly, following the GP-UCB convention.
    The σ-multiplier used by the kernel is √β.
    """
    if sync_beta:
        # (m_batch,) shape for both single-objective and multi-objective cases
        beta = chi2_sampler(n=m_batch, generator=generator, device=device)
        if num_targets > 1 and m_batch > 1:
            beta = beta.view(m_batch, 1)
    else:
        # Full independence: (m_batch, num_targets) shape
        beta = chi2_sampler(n=m_batch * num_targets, generator=generator, device=device)
        if m_batch > 1:
            beta = beta.view(m_batch, num_targets)
    return beta


def _normalize_beta(
    beta_in,
    m_batch: int,
    num_targets: int,
    generator: torch.Generator,
    device: torch.device | str = DEFAULT_DEVICE,
    sync_beta: bool = False,
) -> torch.Tensor:
    """Normalize user-provided beta values or sample automatically if None.
    ensuring correct shape and broadcasting for both qRandomizedStraddle and qMultiRandomizedStraddle.
    """
    # Single-objective is inherently synced, normalize the flag
    if num_targets == 1:
        sync_beta = True
    elif sync_beta:
        warn(_SYNC_WARNING, UserWarning, stacklevel=2)

    if beta_in is not None:
        warn(
            "Note: User-provided beta values. Users are solely responsible for this choice.",
            UserWarning,
            stacklevel=2,
        )
        # User provided explicit beta value(s)
        beta_tensor = torch.as_tensor(beta_in, device=device, dtype=TORCH_DTYPE).flatten()
        n_values = beta_tensor.numel()

        # for multi-objective case,
        # any 0D, (num_targets), (q, 1), (q, num_targets) shape can work with broadcasting rules
        # For single-objective case
        # any 0D, (q) shape can work with broadcasting rules
        if n_values == 1:
            # Scalar beta - use as-is
            return beta_tensor

        if m_batch > 1:
            # Multi-objective, batched
            if num_targets > 1:
                # Full (q, num_targets) specification
                if n_values == m_batch * num_targets:
                    # return as (m_batch, num_targets) for broadcasting in forward pass
                    return beta_tensor.view(m_batch, num_targets)
                elif n_values == m_batch:
                    if not sync_beta:
                        warn(
                            "Using the same beta for all targets, but m_batch > 1. If you intended to have different"
                            " betas per target, please provide num_targets values or set sync_objective_beta=True."
                        )
                    return beta_tensor.view(m_batch, 1)
            # Single-objective, batched
            elif n_values == m_batch:
                return beta_tensor
        elif n_values == num_targets:
            # Multi-objective, non-batched
            return beta_tensor

        raise ValueError(
            f"beta must be scalar, shape ({m_batch},), ({num_targets},), or ({m_batch}, {num_targets}), got"
            f" {n_values} values"
        )
    else:
        return _sample_beta(m_batch, num_targets, sync_beta, generator, device)


def _base_parser(aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext):
    aq_kwargs["beta"] = hps["beta"]
    # Only forward device when context actually provides one
    if context:
        device = context.get("device")
        if device is not None:
            aq_kwargs["device"] = device
    # Prefer generator from hps if provided, otherwise don't set (stays None)
    # This allows fresh generator to inherit global RNG seed from context manager.
    # `generator` is a declared hp only for the randomized straddle variants, so this
    # block is reached only for them; forward the data-state fingerprint (n_obs) so the
    # beta draw varies across iterations while staying idempotent within a suggest().
    if "generator" in hps:
        aq_kwargs["generator"] = hps["generator"]
        if context is not None:
            aq_kwargs["n_obs"] = context.get("n_obs")
    return aq_kwargs


def _base_single_parser(aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext):
    if not context:
        raise ValueError("Context is required for parsing characterization functions.")

    # Validate single-objective
    if len(context["target"]) != 1:
        raise ValueError(
            f"Single-objective characterization requires exactly one target, got {len(context['target'])}."
        )

    # Parse and transform threshold (returns list of transformed values)
    thresholds = _parse_thresholds(hps.get("threshold"), context["target"])

    # Assign single transformed threshold value
    aq_kwargs["threshold"] = thresholds[0]

    return _base_parser(aq_kwargs, hps, context)


def _base_multi_parser(aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext):
    if not context:
        raise ValueError("Context is required for parsing characterization functions.")

    # Parse and transform thresholds (returns list of transformed values)
    thresholds_transformed = _parse_thresholds(hps.get("threshold"), context["target"])

    aq_kwargs["threshold"] = thresholds_transformed
    aq_kwargs["weights"] = hps.get("weights", None)
    aq_kwargs["k_decay"] = hps.get("k_decay", 2.0)
    aq_kwargs["tau"] = hps.get("tau", 0.5)
    aq_kwargs["kernel_reduction"] = hps.get("kernel_reduction", "softmin")
    return _base_parser(aq_kwargs, hps, context)


def _compute_straddle_kernel(
    obj: torch.Tensor,
    k: torch.Tensor,
    h: torch.Tensor,
    clamp_zero: bool = True,
) -> torch.Tensor:
    """
    Compute the straddle kernel component: max(0, k * σ - |μ - h|)

    This is the "boundary-seeking" part of the straddle acquisition function that
    seeks high uncertainty near the threshold boundary.

    Args:
        obj: Objective samples with shape (num_samples, batch_shape, q, [num_targets])
        k: Confidence parameter. Can be:
            - Scalar: same k for all points
            - Shape (num_targets,): per-target k
            - Shape (1, 1, q): per-q-point k (for single-target randomized straddle)
            - Shape (1, 1, q, 1): per-q-point k broadcasted to all targets
            - Shape (1, 1, 1, num_targets): per-target k
            - Shape (1, 1, q, num_targets): full per-point-per-target k
        h: Threshold values, scalar or shape (num_targets,)
        clamp_zero: Whether to clamp negative values to zero

    Returns:
        Kernel values with same shape as obj
    """
    # Compute mean and deviation
    mean = obj.mean(dim=0, keepdim=True)  # 1 x batch_shape x q x [num_targets]
    dev = (obj - mean).abs()  # num_samples x batch_shape x q x [num_targets]
    dist_to_t = (mean - h).abs()  # 1 x batch_shape x q x [num_targets]

    # Straddle kernel: exploration bonus minus distance to threshold
    kernel = k * dev - dist_to_t  # num_samples x batch_shape x q x [num_targets]

    if clamp_zero:
        kernel = torch.clamp(kernel, min=0.0)

    return kernel


class qStraddle(MCAcquisitionFunction):
    """
    qStraddle implements the classic "straddle" acquisition function for level set estimation (LSE)
    with Monte Carlo sampling in BoTorch style.

    References
    Bryan, B., Nichol, R. C., Genovese, C. R., Schneider, J., Miller, C. J., & Wasserman, L. (2005).
    "Active Learning For Identifying Function Threshold Boundaries."
    """

    def __init__(
        self,
        model,
        threshold: float,
        beta: torch.Tensor | float | None = None,
        sampler: MCSampler = default_sampler,
        objective=None,
        posterior_transform=None,
        X_pending=None,
        device: torch.device | str = DEFAULT_DEVICE,
    ):
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        dtype = TORCH_DTYPE
        self.h = torch.as_tensor(threshold, dtype=dtype, device=device)
        # β is the squared confidence parameter (GP-UCB convention)
        if beta is None:
            beta = 1.96 ** 2
        # Any no more than 3D shape that can be broadcasted with (a, b, q) is valid for beta,
        # Example: scalar, (q,), (1, q)
        # However, for non-randomized flavor, we do not anticipate users to use anything than a scalar
        # As for randomized cases, `_normalize_beta` will ensure correct shape
        self.beta = torch.as_tensor(beta, dtype=dtype, device=device)
        self.k = torch.sqrt(self.beta * math.pi / 2.0)
        self.clamp_zero = False

    parser = staticmethod(_base_single_parser)

    @t_batch_mode_transform()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        samples, obj = self._get_samples_and_objectives(X)
        # obj: num_samples x batch_shape x q

        # Compute straddle kernel
        kernel = _compute_straddle_kernel(obj, self.k, self.h, self.clamp_zero)

        # Average over samples, then max over q
        acq = kernel.mean(dim=0).max(dim=-1).values

        return acq


class qRandomizedStraddle(qStraddle):
    """
    qRandomizedStraddle implements the randomized straddle acquisition for level set estimation (LSE),
    replacing the fixed confidence parameter with a chi-squared random draw and clipping the score at zero.

    Following the UCB/LSE convention, ``beta`` is the **squared** confidence
    parameter; the multiplier used by the kernel is √β.

    Args:
        model: A fitted single-output GP model.
        threshold: The target threshold value for level set estimation.
        beta: Optional squared confidence parameter(s). Can be:
            - None: Sample β ~ χ²₂ (default; the multiplier √β is then ~ √χ²₂)
            - Scalar: Use single β for all q points
            - 1D tensor of shape (q,): Per-q-point β for batch optimization
              (typically pre-sampled by the parser when m_batch > 1)
        generator: Random number generator for reproducibility when sampling beta.
        device: Device to create tensors on.

    References:
        Inatsu, R., Abe, K., & Nishikawa, M. (2024).
        "Active Learning for Level Set Estimation Using Randomized Straddle Algorithms."
    """

    def __init__(
        self,
        model,
        threshold: float,
        batch_size: int = 1,
        sampler: MCSampler = default_sampler,
        objective=None,
        posterior_transform=None,
        X_pending=None,
        generator: torch.Generator | None = None,
        beta: torch.Tensor | float | None = None,
        n_obs: int | None = None,
        device: torch.device | str = DEFAULT_DEVICE,
    ):
        if generator is None:
            # Mix the effective RNG seed with the data-state fingerprint (n_obs) so the
            # beta draw varies across iterations but is idempotent within a suggest().
            base: int = get_new_seed(1)  # type: ignore
            seed = derive_seed(base, n_obs)
            gen = torch.Generator(device=torch.device(device)).manual_seed(seed)
        else:
            gen = generator

        beta = _normalize_beta(beta, batch_size, 1, gen, device)

        super().__init__(
            model=model,
            threshold=threshold,
            beta=beta,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            device=device,
        )
        self.clamp_zero = True

    @staticmethod
    def parser(aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext):
        aq_kwargs["batch_size"] = context.get("m_batch", 1)
        return _base_single_parser(aq_kwargs, hps, context)


class qMultiStraddle(MCAcquisitionFunction):
    """
    Multi-target straddle acquisition using smooth masking for level set estimation.

    Args:
        threshold: 1D tensor or list of thresholds, one per target
        beta: Confidence parameter(s) for UCB computation
        k_decay: Decay rate for the smooth plateau (default: 2.0)
        kernel_reduction: How to combine kernel values across targets ("sum", "max", or "softmin")
        tau: Temperature for softmin reduction (scalar, must be > 0). Lower tau enforces
            balanced criticality across targets (soft-AND); higher tau tolerates candidates
            critical on only one target (approaches the mean). Default 0.5.
        weights: Per-target weights. For "sum" reduction, these are direct multipliers.
            For "softmin" reduction, these act as focus multipliers (higher = more attention
            to that target). Default is uniform weights.
    """

    def __init__(
        self,
        model: Model,
        threshold: list[float] | np.ndarray | torch.Tensor,
        weights: list[float] | np.ndarray | torch.Tensor | None = None,
        beta: torch.Tensor | float | None = None,
        k_decay: float = 2.0,
        tau: float = 0.5,
        kernel_reduction: str = "softmin",
        sampler: MCSampler = default_sampler,
        objective=None,
        posterior_transform=None,
        X_pending=None,
        device: torch.device | str = DEFAULT_DEVICE,
    ):
        # An objective is required if this aq is used on a multi-output model
        if objective is None:
            if model.num_outputs > 1:
                objective = IdentityMCMultiOutputObjective()
            else:
                raise ValueError("For single-objective characterization, please use STR or RANDSTR instead.")
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )

        # Validate and setup threshold
        dtype = TORCH_DTYPE
        num_targets = len(threshold)
        if num_targets < 1:
            raise ValueError("No thresholds provided.")
        elif num_targets == 1:
            raise ValueError("For single-objective characterization, please use STR or RANDSTR instead.")
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.tensor(threshold, dtype=dtype, device=device)
        else:
            threshold = threshold.to(dtype=dtype, device=device)

        if threshold.ndim != 1:
            raise ValueError(
                f"threshold must be a 1D tensor or list for multi-target straddle, got shape {threshold.shape}"
            )

        self.h = threshold
        self.num_targets = num_targets
        if weights is None:
            weights = torch.ones_like(threshold)
        else:
            if isinstance(weights, np.ndarray):
                weights = torch.from_numpy(weights)
            elif isinstance(weights, list):
                weights = torch.tensor(weights, dtype=dtype)
            if weights.ndim != 1 or len(weights) != self.num_targets:
                raise ValueError(
                    f"weights must be a 1D tensor or list of length {self.num_targets}, got shape {weights.shape}"
                )
            weights = weights.to(dtype=dtype, device=device)
        self.weights = weights.unsqueeze(0)

        # Setup beta (squared confidence parameter for UCB and kernel)
        if beta is None:
            beta = 1.96 ** 2
        # Any no more than 3D shape that can be broadcasted with (1, q, num_targets) is valid for beta,
        # Example: scalar, (num_targets,), (q, 1), (q, num_targets)
        # However, for non-randomized flavor, we do not anticipate users to use anything than a scalar
        # As for randomized cases, `_normalize_beta` will ensure correct shape
        self.beta = torch.as_tensor(beta, dtype=dtype, device=device)

        # k for the straddle kernel
        self.k = torch.sqrt(self.beta * math.pi / 2.0)

        # k_decay for the smooth plateau mask
        self.k_decay = torch.as_tensor(k_decay, dtype=dtype, device=device)
        if tau <= 0:
            raise ValueError(f"tau must be strictly positive, got {tau}")
        self.tau = torch.as_tensor(tau, dtype=dtype, device=device)

        # Kernel reduction method
        if kernel_reduction not in ["sum", "max", "softmin"]:
            raise ValueError(f"kernel_reduction must be 'sum', 'max' or 'softmin', got {kernel_reduction}")
        self.kernel_reduction = kernel_reduction

    parser = staticmethod(_base_multi_parser)

    def _softmin_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Soft-AND across per-target kernels.

        Smoothly interpolates between the mean (tau -> inf) and the minimum (tau -> 0+).
        Smaller tau enforces balance, so candidates critical on only one target are suppressed;

        Args:
            x: kernel values, shape (batch x q x num_targets), non-negative

        Returns:
            weights: shape (batch x q x num_targets), sum to 1 along last dim
        """
        # Negate so softmax over scores acts as softmin over x
        scores = -x  # batch x q x num_targets, non-positive

        # Apply weights as focus multiplier. Scores are non-positive, so we divide by weights
        # self.weights has shape (1, num_targets), broadcast to (batch, q, num_targets)
        scores = scores / self.weights.squeeze(0)

        # Apply temperature
        w = torch.softmax(scores / self.tau, dim=-1)  # batch x q x num_targets

        return w

    @t_batch_mode_transform()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        samples, obj = self._get_samples_and_objectives(X)
        # obj: num_samples x batch_shape x q x num_targets

        # Compute mean and std for UCB
        mean = obj.mean(dim=0)  # batch_shape x q x num_targets
        std = obj.std(dim=0)  # batch_shape x q x num_targets

        # Compute UCB per target: μ + √β · σ
        ucb = mean + torch.sqrt(self.beta) * std  # batch_shape x q x num_targets

        # Distance to passing for each target: UCB - threshold
        d_per_target = ucb - self.h  # batch_shape x q x num_targets

        # Worst-case distance, min across targets
        d = d_per_target.min(dim=-1).values  # batch_shape x q

        # Soft smooth mask: 1 if d >= 0, else sech(k_decay * d)
        mask = torch.ones_like(d)
        negative_mask = d < 0
        if negative_mask.any():
            mask[negative_mask] = 1.0 / torch.cosh(self.k_decay * d[negative_mask])

        # Compute straddle kernel per target
        kernel_per_target = _compute_straddle_kernel(
            obj, self.k, self.h, clamp_zero=True
        )  # num_samples x batch_shape x q x num_targets

        # Average over samples
        kernel_per_target = kernel_per_target.mean(dim=0)  # batch_shape x q x num_targets

        # Reduce across targets
        if self.kernel_reduction == "max":
            kernel = kernel_per_target.max(dim=-1).values  # batch_shape x q
        else:
            if self.kernel_reduction == "softmin":
                weights = self._softmin_weights(kernel_per_target)
            else:
                weights = self.weights
            # TODO: use einsum or tensor product
            kernel = (kernel_per_target * weights).sum(dim=-1)  # batch_shape x q

        # Apply mask to kernel
        acq = mask * kernel  # batch_shape x q

        # Max over q
        acq = acq.max(dim=-1).values  # batch_shape

        return acq


class qMultiRandomizedStraddle(qMultiStraddle):
    """
    Multi-target randomized straddle with smooth plateau masking.

    Extends qMultiStraddle by sampling beta independently per target from a
    chi-squared distribution, following the randomized straddle algorithm.
    Following GP-UCB convention, ``beta`` is the **squared** confidence parameter;
    the σ-multiplier used by the kernel is √β.

    Args:
        threshold: 1D tensor or list of thresholds, one per target.
        beta: Squared confidence parameter(s). When sampled automatically, β ~ χ²₂.
            Can be:
            - None: For m_batch=1, samples independently per-target (legacy behavior).
            - Scalar: Use single β for all q points and all targets.
            - 1D tensor of shape (q,): Per-q-point β, same across all targets.
              This is the default when m_batch > 1 (sampled by parser).
            - 1D tensor of shape (num_targets,): Per-target β (legacy mode).
            - 2D tensor of shape (q, num_targets): Full specification.
        generator: Random generator for reproducible sampling.
        tau: Temperature for softmin reduction (scalar, must be > 0). Lower tau enforces
            balanced criticality across targets (soft-AND); higher tau tolerates candidates
            critical on only one target (approaches the mean). Default 0.5.
        weights: Per-target weights. For "softmin" reduction, acts as focus multiplier.
    """

    def __init__(
        self,
        model: Model,
        threshold: torch.Tensor | list[float],
        weights: list[float] | np.ndarray | torch.Tensor | None = None,
        beta: torch.Tensor | list[float] | None = None,
        batch_size: int = 1,
        sync_objective_beta: bool = False,
        k_decay: float = 2.0,
        tau: float = 0.5,
        kernel_reduction: str = "softmin",
        sampler: MCSampler = default_sampler,
        objective=None,
        posterior_transform=None,
        X_pending=None,
        generator: torch.Generator | None = None,
        n_obs: int | None = None,
        device: torch.device | str = DEFAULT_DEVICE,
    ):
        # Determine number of targets from threshold
        if not isinstance(threshold, torch.Tensor):
            dtype = TORCH_DTYPE
            threshold = torch.tensor(threshold, dtype=dtype, device=device)

        num_targets = len(threshold)

        if generator is None:
            # Mix the effective RNG seed with the data-state fingerprint (n_obs) so the
            # beta draw varies across iterations but is idempotent within a suggest().
            base: int = get_new_seed(1)  # type: ignore
            seed = derive_seed(base, n_obs)
            gen = torch.Generator(device=torch.device(device)).manual_seed(seed)
        else:
            gen = generator

        beta = _normalize_beta(beta, batch_size, num_targets, gen, device, sync_objective_beta)

        super().__init__(
            model=model,
            threshold=threshold,
            weights=weights,
            beta=beta,
            k_decay=k_decay,
            tau=tau,
            kernel_reduction=kernel_reduction,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
            device=device,
        )

    @staticmethod
    def parser(aq_kwargs: dict[str, Any], hps: dict[str, Any], context: ParserContext):
        aq_kwargs["batch_size"] = context.get("m_batch", 1)
        aq_kwargs["sync_objective_beta"] = hps.get("sync_objective_beta", False)
        return _base_multi_parser(aq_kwargs, hps, context)
