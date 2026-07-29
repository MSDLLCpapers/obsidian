import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import warnings
from scipy.stats import qmc
from scipy import linalg
from scipy.stats import chi2_contingency
from scipy.spatial.distance import pdist
import concurrent.futures
from decimal import Decimal
from obsidian.experiment.design import ExpDesigner
from obsidian.rng import derive_seed

__all__ = ["AdvExpDesigner"]

# ---------------------------------------------------------------------------
# Module-level defaults shared by all optimization paths
# ---------------------------------------------------------------------------

DEFAULT_METRICS = [
    "D-optimality",
    "A-optimality",
    "Condition Number",
    "Pairwise Distance CV",
    "Max Continuous Corr",
    "Max Categorical Corr",
    "Max Mixed Corr",
]


_MAXIMIZE_BY_METRIC = {
    "D-optimality": True,
    "A-optimality": False,
    "Condition Number": False,
    "Pairwise Distance CV": False,
    "Max Continuous Corr": False,
    "Max Categorical Corr": False,
    "Max Mixed Corr": False,
}


def _default_maximize_metrics(metrics):
    """Return a maximize-flags list for ``metrics`` derived by metric name.

    Raises ValueError for any metric not in the known set so that callers with
    custom metric lists are forced to supply explicit ``maximize_metrics``.
    """
    unknown = [m for m in metrics if m not in _MAXIMIZE_BY_METRIC]
    if unknown:
        raise ValueError(
            f"Unknown metric(s) {unknown}. Provide an explicit maximize_metrics "
            f"list, or use one of the standard metrics: {list(_MAXIMIZE_BY_METRIC)}."
        )
    return [_MAXIMIZE_BY_METRIC[m] for m in metrics]


class AdvExpDesigner(ExpDesigner):
    """
    An advanced experimental designer that extends ExpDesigner with support for
    biased/constrained sampling, categorical subparameters, and design quality metrics.

    Extends ExpDesigner so it can be passed directly to Campaign as the ``designer``
    argument. When ``X_space`` is provided, ``campaign.initialize()`` will call
    ``generate_design()`` and honor all biases and constraints defined in
    ``continuous_params`` and ``conditional_subparameters``.
    """

    def __init__(
        self,
        continuous_params: dict | None = None,
        conditional_subparameters: dict | None = None,
        subparam_mapping: dict | None = None,
        design_df: pd.DataFrame | None = None,
        X_space=None,
        seed: int | None = None,
        n_category_trials: int = 100,
        corr_threshold: float = 0.01,
    ):
        """
        Initializes the AdvExpDesigner with experimental parameters and optional
        subparameter mappings.

        Args:
            continuous_params: A dictionary containing the continuous parameters for the
                design. Each parameter can be specified as:

                - ``(low, high, step)``: Linear spacing with a fixed step size.
                - ``(low, high, "geometric")``: Geometric spacing (doubling).
                - ``(low, high, "logarithmic")``: Logarithmic spacing (powers of 10).
                - ``[value1, value2, ...]``: Custom list of specific levels to sample from.
                - ``{'levels': [...], 'biases': [...]}``: Custom levels with optional
                  bias weights for non-uniform sampling. Biases are normalized to sum to
                  1.0 if they do not already.

            conditional_subparameters: A dictionary containing the conditional
                subparameters for the design.
            subparam_mapping: A dictionary for mapping; inferred automatically if not
                provided.
            design_df: A pandas DataFrame of an existing experimental design. Defaults
                to None.
            X_space: obsidian ParamSpace for Campaign integration. When provided, the
                designer can be passed as ``designer=`` to a Campaign and
                ``campaign.initialize()`` will use the biased/constrained sampling
                defined here. The keys in ``continuous_params`` must match the parameter
                names in X_space.
            seed: Random seed for reproducibility (used by ``initialize()``).
            n_category_trials: Number of random category assignments evaluated when
                ``optimize_categories=True``. Higher values reduce inter-category
                correlation at the cost of runtime. Defaults to 100.
            corr_threshold: Early-exit correlation threshold for category optimization.
                When the best assignment reaches a max correlation below this value the
                search terminates early. Defaults to 0.01.
        """
        if X_space is not None:
            super().__init__(X_space, seed=seed)
        else:
            self.X_space = None
            self.seed = seed

        self.continuous_params = continuous_params if continuous_params else {}
        self.conditional_subparameters = conditional_subparameters if conditional_subparameters else {}
        self.n_category_trials = n_category_trials
        self.corr_threshold = corr_threshold

        if design_df is not None and not design_df.empty:
            self.design = design_df
            categorical, numerical = infer_column_types(design_df)
            self.categorical_keys = categorical
            if continuous_params:
                self.continuous_keys = list(self.continuous_params.keys())
            else:
                self.continuous_keys = numerical
        else:
            self.continuous_keys = list(self.continuous_params.keys()) if continuous_params else []
            self.categorical_keys = list(self.conditional_subparameters.keys()) if conditional_subparameters else []

        self.subparam_mapping = subparam_mapping or infer_subparam_mapping(self.conditional_subparameters)
        self.subparam_keys = list(self.subparam_mapping.values()) if self.subparam_mapping else []

    def __repr__(self):
        n_cont = len(self.continuous_params)
        n_cat = len(self.conditional_subparameters)
        return f"AdvExpDesigner(continuous_params={n_cont}, conditional_subparameters={n_cat}, X_space={self.X_space})"

    def generate_design(self, seed, n_samples, optimize_categories=True):
        """
        Generates a design by sampling from the given parameter space.

        Args:
            seed: Random seed for reproducibility.
            n_samples: Number of samples to generate.
            optimize_categories: Whether to optimize categorical assignments to reduce
                inter-category correlation. Defaults to True.

        Returns:
            pd.DataFrame: The generated sample design.

        Note:
            When ``optimize_categories=True``, only the first subparam-mapped category
            (as determined by ``subparam_mapping``) is optimized. Additional categorical
            variables are assigned with a single random draw.
        """
        return sample_design(
            seed=seed,
            n_samples=n_samples,
            continuous_params=self.continuous_params,
            conditional_subparameters=self.conditional_subparameters,
            subparam_mapping=self.subparam_mapping,
            optimize_categories=optimize_categories,
            n_category_trials=self.n_category_trials,
            corr_threshold=self.corr_threshold,
        )

    def initialize(self, m_initial=None, method="LHS", sample_custom=None, optimize_categories=False):
        """
        Generates an initial experimental design honoring all biases and constraints
        defined in ``continuous_params`` and ``conditional_subparameters``.

        Overrides ``ExpDesigner.initialize()`` so that a Campaign whose ``designer``
        is an AdvExpDesigner will automatically use biased/constrained sampling.

        Args:
            m_initial: Number of initial experiments. Defaults to ``2 * n_dim`` when
                X_space is provided, or raises if neither is available.
            method: Sampling strategy.

                - ``'LHS'`` (default): calls ``generate_design()`` with LHS + biases.
                - ``'Optimized'``: calls ``optimize_design()`` to maximize D-optimality
                  across multiple trials (slower but higher-quality).

            sample_custom: Ignored; retained for API compatibility with ExpDesigner.
            optimize_categories: Whether to optimize categorical assignments to minimize
                correlation (passed to ``generate_design()``). Defaults to False.

        Returns:
            pd.DataFrame: The generated design.

        Raises:
            ValueError: If m_initial cannot be inferred (no X_space and no m_initial
                given).
        """
        if m_initial is None:
            if self.X_space is not None:
                m_initial = int(self.X_space.n_dim * 2)
            else:
                raise ValueError("m_initial must be specified when X_space is not provided.")

        seed = self.seed if self.seed is not None else int(np.random.default_rng().integers(0, 2**31))

        _SUPPORTED_METHODS = {"LHS", "Optimized"}
        if method not in _SUPPORTED_METHODS:
            raise ValueError(f"Unknown method {method!r}. Supported methods are: {sorted(_SUPPORTED_METHODS)}.")

        if method == "Optimized" and optimize_categories:
            raise ValueError(
                "optimize_categories=True is not supported with method='Optimized'. "
                "Use method='LHS' to honor optimize_categories, or call "
                "generate_design() / sample_design() directly."
            )

        if method == "Optimized":
            design, _ = self.optimize_design(n_trials=10, n_samples=m_initial, seed_start=seed)
        else:
            design = self.generate_design(
                seed=seed,
                n_samples=m_initial,
                optimize_categories=optimize_categories,
            )

        if self.X_space is not None:
            missing = set(self.X_space.X_names) - set(design.columns)
            if missing:
                warnings.warn(
                    f"AdvExpDesigner design is missing columns present in X_space: {missing}. "
                    "Ensure continuous_params keys and categorical keys match the X_space "
                    "parameter names."
                )

        return design

    def evaluate_design(self, design, metrics_to_optimize=None):
        """
        Evaluates the quality of the given design based on specified metrics.

        Args:
            design: The design DataFrame to evaluate.
            metrics_to_optimize: List of metric names to evaluate. Defaults to all
                metrics in ``DEFAULT_METRICS``.

        Returns:
            dict: Computed metric values keyed by metric name.
        """
        if metrics_to_optimize is None:
            metrics_to_optimize = list(DEFAULT_METRICS)

        return evaluate_design(
            design=design,
            continuous_keys=self.continuous_keys,
            categorical_keys=self.categorical_keys,
            subparam_mapping=self.subparam_mapping,
            metrics_to_optimize=metrics_to_optimize,
        )

    def optimize_design(
        self,
        n_trials,
        n_samples,
        metrics_to_optimize=None,
        maximize_metrics=None,
        seed_start=0,
        max_workers=None,
    ):
        """
        Optimizes the design by generating multiple candidates and selecting the best
        according to a composite score over the specified metrics.

        Args:
            n_trials: Number of candidate designs to generate and evaluate.
            n_samples: Number of experiments in each candidate design.
            metrics_to_optimize: List of metric names to include in the composite
                score. Defaults to all seven standard metrics.
            maximize_metrics: List of booleans, one per metric, indicating whether
                each metric should be maximized (True) or minimized (False). Defaults
                to ``[True, False, False, ...]`` — maximize D-optimality only.
            seed_start: Starting random seed for candidate generation. Defaults to 0.
            max_workers: Maximum number of parallel worker processes. Defaults to
                ``None`` (uses all available CPUs).

        Returns:
            tuple: ``(best_design, metrics_df)`` where ``best_design`` is the
                highest-scoring pd.DataFrame and ``metrics_df`` is a pd.DataFrame
                summarizing all candidates.

        """
        if metrics_to_optimize is None:
            metrics_to_optimize = list(DEFAULT_METRICS)
        if maximize_metrics is None:
            maximize_metrics = _default_maximize_metrics(metrics_to_optimize)

        return find_best_design_parallel(
            n=n_trials,
            n_samples=n_samples,
            continuous_params=self.continuous_params,
            conditional_subparameters=self.conditional_subparameters,
            subparam_mapping=self.subparam_mapping,
            metrics_to_optimize=metrics_to_optimize,
            maximize_metrics=maximize_metrics,
            seed_start=seed_start,
            max_workers=max_workers,
            n_category_trials=self.n_category_trials,
            corr_threshold=self.corr_threshold,
        )

    def extend_design(
        self,
        existing_design,
        n,
        seed=None,
        n_trials=10,
        metrics_to_optimize=None,
        maximize_metrics=None,
        max_workers=None,
    ):
        """
        Extends an existing design by appending the best-scoring set of new samples
        chosen from multiple candidates.

        Args:
            existing_design: The existing design DataFrame to extend.
            n: Number of new samples to add.
            seed: Optional random seed for reproducibility.
            n_trials: Number of candidate extensions to evaluate. Defaults to 10.
            metrics_to_optimize: List of metric names to include in scoring. Defaults
                to all seven standard metrics.
            maximize_metrics: List of booleans indicating whether to maximize each
                metric. Defaults to ``[True, False, False, ...]``.
            max_workers: Number of parallel worker processes.

        Returns:
            tuple: ``(extended_design, metrics_summary)`` where ``extended_design``
                contains all original rows plus the best new rows, and
                ``metrics_summary`` is a pd.DataFrame of candidate scores.

        """
        seed_start = seed if seed is not None else 1000

        return extend_design(
            existing_design=existing_design,
            n=n,
            continuous_params=self.continuous_params,
            conditional_subparameters=self.conditional_subparameters,
            subparam_mapping=self.subparam_mapping,
            metrics_to_optimize=metrics_to_optimize,
            maximize_metrics=maximize_metrics,
            n_trials=n_trials,
            seed_start=seed_start,
            max_workers=max_workers,
            n_category_trials=self.n_category_trials,
            corr_threshold=self.corr_threshold,
        )

    def plot_quality_evolution(self, metrics_df):
        """
        Plots per-metric bar charts over trial seeds to visualize design quality
        evolution.

        Args:
            metrics_df: DataFrame containing trial metrics (must include a 'seed'
                column).
        """
        plot_design_quality_evolution(metrics_df)

    def plot_histograms(self, design):
        """
        Plots histograms (continuous) and bar charts (categorical) for each parameter
        in the design.

        Args:
            design: The design DataFrame to visualize.
        """
        plot_design_histograms(
            design=design,
            continuous_keys=self.continuous_keys,
            categorical_keys=self.categorical_keys,
            subparam_mapping=self.subparam_mapping,
        )

    def plot_correlation(self, design):
        """
        Plots a mixed correlation matrix heatmap for the design's parameters.

        Args:
            design: The design DataFrame to visualize.
        """
        plot_correlation_matrix(design, self.categorical_keys)

    def plot_pca(self, design, hue=None):
        """
        Performs PCA on the continuous parameters and plots the first two components.

        Args:
            design: The design DataFrame to analyze.
            hue: Name of a categorical column to use for color-coding points.
        """
        plot_pca(design, self.continuous_keys, self.subparam_mapping, hue)

    def plot_mds(self, design, hue=None):
        """
        Performs Multidimensional Scaling (MDS) on the continuous parameters and
        plots the two-dimensional embedding.

        Args:
            design: The design DataFrame to analyze.
            hue: Name of a categorical column to use for color-coding points.
        """
        plot_mds(design, self.continuous_keys, self.subparam_mapping, hue)

    def plot_umap(self, design, hue=None, verbose=False):
        """
        Performs UMAP dimensionality reduction on the continuous parameters and
        plots the two-dimensional embedding.

        Args:
            design: The design DataFrame to analyze.
            hue: Name of a categorical column to use for color-coding points.
            verbose: Whether to show UMAP's internal progress log. Defaults to
                False.
        """
        plot_umap(design, self.continuous_keys, self.subparam_mapping, hue, verbose=verbose)

    def compare_frequencies(self, design, verbose=True):
        """
        Compares the empirical frequencies of categorical variables in the design
        with the expected frequencies defined in ``conditional_subparameters``.

        Args:
            design: The design DataFrame to analyze.
            verbose: If True, print the frequency table to stdout. Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame with columns
                ``['categorical_var', 'level', 'expected', 'empirical']`` containing
                one row per level of each categorical variable.
        """
        if not self.conditional_subparameters:
            raise ValueError("compare_frequencies requires conditional_subparameters to be defined.")
        missing = [k for k in self.categorical_keys if k not in self.conditional_subparameters]
        if missing:
            raise ValueError(
                f"compare_frequencies: no conditional_subparameters entry for categorical variable(s): {missing}"
            )
        rows = []
        for cat_var in self.categorical_keys:
            level_info = self.conditional_subparameters[cat_var]
            levels = list(level_info.keys())
            expected_freq = np.array([level_info[lvl].get("freq", 1 / len(levels)) for lvl in levels])
            expected_freq /= expected_freq.sum()

            counts = design[cat_var].value_counts(normalize=True).reindex(levels).fillna(0).values

            for lvl, exp_f, emp_f in zip(levels, expected_freq, counts):
                rows.append(
                    {
                        "categorical_var": cat_var,
                        "level": lvl,
                        "expected": round(float(exp_f), 4),
                        "empirical": round(float(emp_f), 4),
                    }
                )

        result = pd.DataFrame(rows, columns=["categorical_var", "level", "expected", "empirical"])

        if verbose:
            print(result.to_string(index=False))

        return result

    def save_state(self) -> dict:
        """Save the designer state to a JSON-serializable dictionary.

        Returns:
            dict: JSON-safe payload containing every constructor argument plus
                the class name for polymorphic dispatch.
        """
        state = {
            "name": self.__class__.__name__,
            "X_space": self.X_space.save_state() if self.X_space is not None else None,
            "seed": self.seed,
            "continuous_params": _serialize_continuous_params(self.continuous_params),
            "conditional_subparameters": _serialize_cond_subparam(self.conditional_subparameters),
            "subparam_mapping": dict(self.subparam_mapping) if self.subparam_mapping else {},
            "n_category_trials": int(self.n_category_trials),
            "corr_threshold": float(self.corr_threshold),
        }
        design = getattr(self, "design", None)
        if design is not None and isinstance(design, pd.DataFrame) and not design.empty:
            state["design_df"] = design.to_dict(orient="list")
        else:
            state["design_df"] = None
        return state

    @classmethod
    def load_state(cls, obj_dict: dict, X_space=None, seed: int | None = None) -> "AdvExpDesigner":
        """Reconstruct an :class:`AdvExpDesigner` from a saved state dictionary.

        Args:
            obj_dict (dict): Output of :meth:`save_state`.
            X_space: Override for the parameter space. When provided
                (typically by :meth:`Campaign.load_state`), the X_space payload
                in ``obj_dict`` is ignored. Defaults to ``None``.
            seed (int | None, optional): Override for the seed. When provided,
                ``obj_dict['seed']`` is ignored. Defaults to ``None``.

        Returns:
            AdvExpDesigner: A new designer instance equivalent to the saved one.
        """
        from obsidian.parameters import ParamSpace

        if X_space is None and obj_dict.get("X_space") is not None:
            X_space = ParamSpace.load_state(obj_dict["X_space"])

        if seed is None:
            seed = obj_dict.get("seed")

        design_payload = obj_dict.get("design_df")
        if design_payload is not None:
            design_df = pd.DataFrame(design_payload)
        else:
            design_df = None

        subparam_mapping = obj_dict.get("subparam_mapping") or None

        return cls(
            continuous_params=_deserialize_continuous_params(obj_dict.get("continuous_params") or {}),
            conditional_subparameters=_deserialize_cond_subparam(
                obj_dict.get("conditional_subparameters") or {}
            ),
            subparam_mapping=subparam_mapping,
            design_df=design_df,
            X_space=X_space,
            seed=seed,
            n_category_trials=int(obj_dict.get("n_category_trials", 100)),
            corr_threshold=float(obj_dict.get("corr_threshold", 0.01)),
        )


# Every serialized ``continuous_params`` entry is a dict. Two shapes are distinguished by their keys
#
#   - ``(low, high, step)``  →  {"low": ..., "high": ..., "step": ...}
#       ``step`` is a number or one of the strings ``'geometric'`` /
#       ``'logarithmic'``.
#   - Explicit level list / dict + biases  →
#       {"levels": [...], "biases": None | [...]}
def _serialize_continuous_param(spec):
    """Return a JSON-safe dict representation of a single ``continuous_params`` spec.

    Accepts any of the four in-memory shapes produced by users:

    - ``(low, high, step)`` 3-tuple → ``{"low": ..., "high": ..., "step": ...}``
    - Sequence of explicit levels (list / long tuple) → ``{"levels": [...], "biases": None}``
    - ``{'levels': [...], 'biases': [...]}`` dict → same dict with values JSONified

    numpy scalars / arrays are coerced to native Python types.
    """
    if isinstance(spec, dict):
        levels = [float(x) for x in spec["levels"]]
        biases = spec.get("biases")
        if biases is not None:
            biases = [float(x) for x in biases]
        return {"levels": levels, "biases": biases}

    if isinstance(spec, (list, tuple)) and len(spec) == 3 and not isinstance(spec[0], (list, tuple)):
        low, high, step = spec
        step_out = step if isinstance(step, str) else float(step)
        return {"low": float(low), "high": float(high), "step": step_out}

    # Long tuple / list of explicit levels → normalize to the same dict shape
    # as the explicit-dict form, with biases=None.
    return {
        "levels": [x if isinstance(x, str) else float(x) for x in spec],
        "biases": None,
    }


def _deserialize_continuous_param(payload):
    """Inverse of :func:`_serialize_continuous_param`.

    Dispatches on which keys the payload has: ``"low"``/``"high"``/``"step"``
    → a 3-tuple; ``"levels"`` → the dict form (optionally with ``"biases"``).
    """
    if "levels" in payload:
        result = {"levels": list(payload["levels"])}
        if payload.get("biases") is not None:
            result["biases"] = list(payload["biases"])
        return result

    if "low" in payload and "high" in payload and "step" in payload:
        return (payload["low"], payload["high"], payload["step"])

    raise ValueError(
        f"Unknown serialized continuous_param payload: {payload!r}. "
        "Expected keys {'low', 'high', 'step'} or {'levels', 'biases'}."
    )



def _serialize_continuous_params(continuous_params):
    """Serialize a full ``continuous_params`` dict."""
    if not continuous_params:
        return {}
    return {name: _serialize_continuous_param(spec) for name, spec in continuous_params.items()}


def _deserialize_continuous_params(payload):
    """Inverse of :func:`_serialize_continuous_params`."""
    if not payload:
        return {}
    return {name: _deserialize_continuous_param(spec) for name, spec in payload.items()}


def _walk_cond_subparam_dict(conditional_subparameters, subparam_transform):
    """Walk a ``conditional_subparameters`` dict and transform each sub-parameter leaf."""
    if not conditional_subparameters:
        return {}

    out = {}
    for cat_var, levels in conditional_subparameters.items():
        out_levels = {}
        for level_name, level_info in levels.items():
            out_info = {}
            for key, value in level_info.items():
                if key == "freq":
                    out_info["freq"] = float(value)
                else:
                    out_info[key] = subparam_transform(value)
            out_levels[level_name] = out_info
        out[cat_var] = out_levels
    return out


def _serialize_cond_subparam(conditional_subparameters):
    """Serialize the nested ``conditional_subparameters`` dict.

    The ``(values, weights)`` 2-tuples are converted to dicts with ``values`` and ``weights`` keys.
    """
    def _serialize_subparam(value):
        values, weights = value
        return {
            "values": [float(v) if not isinstance(v, str) else v for v in values],
            "weights": [float(w) for w in weights],
        }

    return _walk_cond_subparam_dict(conditional_subparameters, _serialize_subparam)


def _deserialize_cond_subparam(payload):
    """Inverse of :func:`_serialize_cond_subparam`."""
    def _deserialize_subparam(value):
        # Stored as {'values': [...], 'weights': [...]}
        return (list(value["values"]), list(value["weights"]))

    return _walk_cond_subparam_dict(payload, _deserialize_subparam)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def _get_param_levels(key, spec):
    """
    Parse a single ``continuous_params`` spec entry and return its discrete levels
    together with optional sampling bias weights.

    Args:
        key: Parameter name (used only in error messages).
        spec: One of:

            - ``(low, high, step)``: 3-tuple; step may be a number, ``'geometric'``,
              or ``'logarithmic'``.
            - ``[v1, v2, ...]`` or ``(v1, v2, v3, v4, ...)``: Sequence of explicit
              levels (length ≠ 3 or first element is itself a sequence).
            - ``{'levels': [...], 'biases': [...]}``: Dict with explicit levels and
              optional normalized bias weights.

    Returns:
        tuple: ``(levels, biases)`` where ``levels`` is a ``np.ndarray`` of the
            discrete values and ``biases`` is a normalized ``np.ndarray`` of the same
            length or ``None`` if uniform sampling is desired.

    Raises:
        ValueError: If a dict spec is missing ``'levels'``, if bias length does not
            match levels length, or if logarithmic bounds are non-positive.

    Note:
        A 3-element sequence whose first element is a scalar (e.g. ``[10, 50, 100]``)
        is indistinguishable from a ``(low, high, step)`` triple and will be interpreted
        as the latter. To pass exactly three custom levels use the dict form instead:
        ``{'levels': [10, 50, 100]}``.
    """
    if isinstance(spec, dict):
        levels = spec.get("levels")
        if levels is None:
            raise ValueError(f"Dictionary specification for parameter '{key}' must include 'levels' key")
        levels = np.array(levels)
        biases = spec.get("biases", None)
        if biases is not None:
            biases = np.array(biases, dtype=float)
            if len(biases) != len(levels):
                raise ValueError(
                    f"Bias length ({len(biases)}) must match levels length ({len(levels)}) for parameter '{key}'"
                )
            if biases.sum() == 0:
                raise ValueError(
                    f"Biases for parameter '{key}' sum to zero — at least one bias value must be positive."
                )
            if not np.isclose(biases.sum(), 1.0):
                biases = biases / biases.sum()
        return levels, biases

    if isinstance(spec, (list, tuple)) and not (len(spec) == 3 and not isinstance(spec[0], (list, tuple))):
        return np.array(spec), None

    # 3-tuple: (low, high, step)
    low, high, step = spec
    if step == "geometric":
        possible = []
        value = low
        while value <= high:
            possible.append(value)
            value *= 2
        return np.array(possible), None
    if step == "logarithmic":
        if low <= 0 or high <= 0:
            raise ValueError(f"Logarithmic step requires positive low and high for parameter '{key}'")
        exp_low = int(np.floor(np.log10(low)))
        exp_high = int(np.floor(np.log10(high)))
        return 10.0 ** np.arange(exp_low, exp_high + 1), None
    # Numeric step
    num_steps = int(round((high - low) / step)) + 1
    return np.linspace(low, high, num_steps), None


# ---------------------------------------------------------------------------
# Sampling functions
# ---------------------------------------------------------------------------


def sample_continuous_lhs(continuous_params, n_samples, seed):
    """
    Draw LHS samples for all continuous parameters, mapping [0, 1) uniform samples
    to the discrete level sets defined in ``continuous_params``.

    Args:
        continuous_params: Dict mapping parameter name → spec (see
            ``_get_param_levels`` for accepted formats).
        n_samples: Number of rows to sample.
        seed: Random seed passed to the LHS sampler.

    Returns:
        dict: Mapping parameter name → list of sampled values.
    """
    if not continuous_params:
        return {}
    sampler = qmc.LatinHypercube(
        d=len(continuous_params),
        seed=seed,
        scramble=True,
        strength=1,
        optimization="random-cd",
    )
    sample_cont = sampler.random(n=n_samples)
    cont_samples = {}
    for idx, key in enumerate(continuous_params):
        possible, biases = _get_param_levels(key, continuous_params[key])
        uniform_samples = sample_cont[:, idx]

        if biases is not None:
            cdf = np.cumsum(biases)
            indices = np.searchsorted(cdf, uniform_samples)
        else:
            indices = np.floor(uniform_samples * len(possible)).astype(int)

        indices = np.clip(indices, 0, len(possible) - 1)
        cont_samples[key] = possible[indices]

    return cont_samples


def non_uniform_lhs_categorical(level_dict, n_samples, seed=None, scramble=True):
    """
    Draw LHS-stratified categorical samples using inverse-transform sampling with
    per-level frequency weights.

    Args:
        level_dict: Dict mapping level name → ``{'freq': float, subparam: ([values],
            [weights]), ...}``.
        n_samples: Number of samples to draw.
        seed: Random seed. Defaults to None.
        scramble: Whether to scramble the LHS sampler. Defaults to True.

    Returns:
        list[dict]: One dict per sample containing ``'level'`` and any sampled
            subparameter values.
    """
    levels = list(level_dict.keys())
    probabilities = [level_dict[level]["freq"] for level in levels]
    if not np.isclose(sum(probabilities), 1.0):
        probabilities = np.array(probabilities) / np.sum(probabilities)
    sampler = qmc.LatinHypercube(
        d=1,
        seed=20 + 3 * seed if seed is not None else None,
        scramble=scramble,
        strength=1,
        optimization="random-cd",
    )
    uniform_samples = sampler.random(n=n_samples).flatten()
    cdf = np.cumsum(probabilities)
    results = []
    for i, sample in enumerate(uniform_samples):
        index = min(np.searchsorted(cdf, sample), len(levels) - 1)
        level = levels[index]
        entry = {"level": level}
        for subparam, value in level_dict[level].items():
            if subparam == "freq":
                continue
            values, weights = value
            weights = np.array(weights, dtype=float)
            weights /= weights.sum()
            sub_sampler = qmc.LatinHypercube(
                d=1,
                seed=seed + i if seed is not None else None,
                scramble=scramble,
                strength=1,
                optimization="random-cd",
            )
            sub_sample = sub_sampler.random(n=1).flatten()[0]
            sub_cdf = np.cumsum(weights)
            sub_index = min(np.searchsorted(sub_cdf, sub_sample), len(values) - 1)
            entry[subparam] = values[sub_index]
        results.append(entry)
    return results


def optimize_category_assignment_parallel(
    cat_samples,
    conditional_subparameters,
    subparam_mapping,
    n_samples,
    seed,
    max_workers=4,
    n_category_trials=100,
    corr_threshold=0.01,
):
    """
    Search for the category assignment (for the variable that has a subparam mapping)
    that minimizes the maximum inter-category correlation.

    Args:
        cat_samples: Dict of already-sampled categorical columns (excluding the
            variable being optimized).
        conditional_subparameters: Full conditional subparameter specification.
        subparam_mapping: Dict mapping each categorical variable to its subparameter.
        n_samples: Number of rows in the design.
        seed: Base random seed.
        max_workers: Maximum number of parallel threads. Defaults to 4.
        n_category_trials: Number of random assignments to evaluate. Defaults to 100.
        corr_threshold: Stop early if max correlation drops below this value.
            Defaults to 0.01.

    Returns:
        list: Best-found list of category values for the optimized variable.
    """
    category_key = next((k for k in subparam_mapping if k in conditional_subparameters), None)
    if category_key is None:
        raise ValueError(
            "Could not infer a categorical key: no key in subparam_mapping is present "
            "in conditional_subparameters. Ensure subparam_mapping keys match the "
            "conditional_subparameters keys."
        )
    other_cat_keys = [k for k in conditional_subparameters.keys() if k != category_key]

    def evaluate_assignment(j):
        sample_cat_entries = non_uniform_lhs_categorical(
            conditional_subparameters[category_key], n_samples, seed=3 * seed + 220 + j
        )
        sample_cat = [entry["level"] for entry in sample_cat_entries]
        temp_cat_samples = cat_samples.copy()
        temp_cat_samples[category_key] = sample_cat

        corr_matrix = calculate_mixed_correlation_matrix(
            pd.DataFrame(temp_cat_samples),
            categorical_vars=[category_key] + other_cat_keys,
        )

        if other_cat_keys:
            max_corr = max(abs(corr_matrix.loc[category_key, other_key]) for other_key in other_cat_keys)
        else:
            max_corr = 0.0

        return max_corr, sample_cat

    best_category_assignment = None
    min_max_correlation = float("inf")
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = [executor.submit(evaluate_assignment, j) for j in range(n_category_trials)]
        for future in concurrent.futures.as_completed(futures):
            max_corr, sample_cat = future.result()
            if max_corr < min_max_correlation:
                min_max_correlation = max_corr
                best_category_assignment = sample_cat
                if min_max_correlation < corr_threshold:
                    break
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return best_category_assignment


def assign_conditional_subparameter(cat_samples, conditional_subparameters, parent_key, subparam_key, n_samples, seed):
    """
    Sample subparameter values conditioned on the parent category assignment.

    Args:
        cat_samples: Dict containing the already-sampled parent category column.
        conditional_subparameters: Full conditional subparameter specification.
        parent_key: Name of the parent categorical variable.
        subparam_key: Name of the subparameter to sample.
        n_samples: Total number of rows in the design.
        seed: Base random seed.

    Returns:
        list: Sampled subparameter values, one per row.

    Raises:
        ValueError: If a level is missing from ``conditional_subparameters`` or
            lacks the expected subparameter entry.
    """
    subparam_samples = [None] * n_samples
    level_indices = {}
    for i, level in enumerate(cat_samples[parent_key]):
        level_indices.setdefault(level, []).append(i)
    for level, indices in level_indices.items():
        level_info = conditional_subparameters[parent_key].get(level)
        if level_info is None or subparam_key not in level_info:
            raise ValueError(f"Level '{level}' missing or lacks '{subparam_key}' in conditional_subparameters")
        values, weights = level_info[subparam_key]
        weights = np.array(weights) / np.sum(weights)
        n_level_samples = len(indices)
        level_seed = derive_seed(seed, level)
        sampled_values = non_uniform_lhs_categorical(
            {str(v): {"freq": w} for v, w in zip(values, weights)},
            n_level_samples,
            seed=level_seed,
        )
        sampled_values = [float(d["level"]) for d in sampled_values]
        for i, idx in enumerate(indices):
            subparam_samples[idx] = sampled_values[i]
    return subparam_samples


def infer_subparam_mapping(conditional_subparameters):
    """
    Infer the subparameter mapping from ``conditional_subparameters`` by finding
    categorical variables that have exactly one subparameter across all their levels.

    Args:
        conditional_subparameters: Full conditional subparameter specification.

    Returns:
        dict: Mapping ``{categorical_param: subparam_name}`` for each categorical
            variable that has a uniquely named subparameter.
    """
    mapping = {}
    for cat_param, levels in conditional_subparameters.items():
        subparam_candidates = set()
        for level_info in levels.values():
            subparams = [k for k in level_info if k != "freq"]
            subparam_candidates.update(subparams)
        if len(subparam_candidates) == 1:
            mapping[cat_param] = subparam_candidates.pop()
        elif len(subparam_candidates) > 1:
            warnings.warn(
                f"Category '{cat_param}' has multiple subparameters {subparam_candidates} "
                "and will be excluded from subparam_mapping. Pass subparam_mapping "
                "explicitly if you need to map more than one subparameter per category.",
                UserWarning,
                stacklevel=2,
            )
    return mapping


def _round_continuous_columns(design, continuous_params):
    """Round each continuous column to the precision implied by its spec (see sample_design)."""
    for key in continuous_params.keys():
        spec = continuous_params[key]
        possible, _ = _get_param_levels(key, spec)
        is_3tuple = (
            isinstance(spec, (list, tuple)) and len(spec) == 3
            and not isinstance(spec[0], (list, tuple))
        )
        if is_3tuple:
            step = spec[2]
            if step == 'geometric':
                design[key] = design[key].round(3)
            elif step == 'logarithmic':
                design[key] = design[key].round(5)
            else:
                decimals = max(0, -Decimal(str(step)).normalize().as_tuple().exponent)
                design[key] = design[key].round(decimals)
                if isinstance(step, int) or (isinstance(step, float) and step.is_integer()):
                    design[key] = design[key].astype(int)
        else:
            if np.all(np.equal(np.mod(possible, 1), 0)):
                design[key] = design[key].astype(int)
            else:
                max_decimals = 0
                for val in possible:
                    if not np.isnan(val):
                        d = max(0, -Decimal(str(val)).normalize().as_tuple().exponent)
                        max_decimals = max(max_decimals, d)
                design[key] = design[key].round(max_decimals)
    return design


def sample_design(
    seed,
    n_samples,
    continuous_params,
    conditional_subparameters,
    subparam_mapping=None,
    optimize_categories=False,
    n_category_trials=100,
    corr_threshold=0.01,
):
    """
    Generate a complete experimental design by LHS-sampling all continuous and
    categorical parameters.

    Args:
        seed: Random seed for reproducibility.
        n_samples: Number of rows to generate.
        continuous_params: Continuous parameter specifications (see
            ``_get_param_levels``).
        conditional_subparameters: Conditional categorical/subparameter
            specifications.
        subparam_mapping: Dict mapping categorical variables to their subparameters.
            Inferred automatically if not provided.
        optimize_categories: Whether to optimize the category assignment for the
            variable that has a subparam mapping to reduce inter-category correlation.
            Defaults to False.
        n_category_trials: Number of random assignments evaluated during category
            optimization. Defaults to 100.
        corr_threshold: Early-exit correlation threshold for category optimization.
            Defaults to 0.01.

    Returns:
        pd.DataFrame: The generated design with appropriately rounded values.

    Note:
        When ``optimize_categories=True``, only the first subparam-mapped category
        (as determined by ``subparam_mapping``) is optimized. Additional categorical
        variables are assigned with a single random draw.
    """
    if subparam_mapping is None:
        subparam_mapping = infer_subparam_mapping(conditional_subparameters)

    cont_samples = sample_continuous_lhs(continuous_params, n_samples, seed)
    cat_samples = {}
    subparam_samples = {}

    category_to_optimize = (
        next((k for k in subparam_mapping if k in conditional_subparameters), None) if optimize_categories else None
    )

    for cat_key, level_dict in conditional_subparameters.items():
        if optimize_categories and cat_key == category_to_optimize:
            continue
        samples = non_uniform_lhs_categorical(
            level_dict=level_dict,
            n_samples=n_samples,
            seed=derive_seed(seed, cat_key),
        )
        cat_samples[cat_key] = [entry["level"] for entry in samples]

    if optimize_categories and category_to_optimize:
        optimized_assignment = optimize_category_assignment_parallel(
            cat_samples=cat_samples,
            conditional_subparameters=conditional_subparameters,
            subparam_mapping=subparam_mapping,
            n_samples=n_samples,
            seed=seed,
            n_category_trials=n_category_trials,
            corr_threshold=corr_threshold,
        )
        cat_samples[category_to_optimize] = optimized_assignment

    for parent_key, subparam_key in subparam_mapping.items():
        if parent_key in cat_samples:
            subparam_samples[subparam_key] = assign_conditional_subparameter(
                cat_samples,
                conditional_subparameters,
                parent_key,
                subparam_key,
                n_samples,
                seed,
            )

    design = pd.DataFrame({**cont_samples, **cat_samples, **subparam_samples})
    return _round_continuous_columns(design, continuous_params)


# ---------------------------------------------------------------------------
# Mixed correlation matrix
# ---------------------------------------------------------------------------


def cramers_v_np(contingency):
    """
    Compute Cramér's V association statistic for a contingency table.

    Args:
        contingency: 2-D integer array (contingency table).

    Returns:
        float: Cramér's V in [0, 1].
    """
    chi2, _, _, _ = chi2_contingency(contingency, correction=False)
    n = contingency.sum()
    min_dim = min(contingency.shape) - 1
    if min_dim == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))


def eta_squared_np(cat_codes, num_values):
    """
    Compute the eta-squared effect size between a categorical and a numeric variable.

    Args:
        cat_codes: Integer-coded category labels (1-D array).
        num_values: Numeric values (1-D array, same length as ``cat_codes``).

    Returns:
        float: Eta-squared in [0, 1].
    """
    overall_mean = np.mean(num_values)
    unique_cats, inverse = np.unique(cat_codes, return_inverse=True)
    counts = np.bincount(inverse)
    means = np.bincount(inverse, weights=num_values) / counts
    ss_between = np.sum(counts * (means - overall_mean) ** 2)
    ss_total = np.sum((num_values - overall_mean) ** 2)
    if ss_total == 0:
        return 0.0
    return ss_between / ss_total


def infer_column_types(df):
    """Canonical column-type rule: object/category dtype -> categorical, else numerical.

    Single source of truth for type inference (used by the correlation matrix and AdvExpDesigner).
    """
    categorical = [c for c in df.columns if df[c].dtype.name in ("object", "category")]
    numerical = [c for c in df.columns if c not in categorical]
    return categorical, numerical


def calculate_mixed_correlation_matrix(df, categorical_vars=None):
    """
    Compute a pairwise correlation matrix that handles mixed variable types:

    - numeric ↔ numeric: Pearson r
    - categorical ↔ categorical: Cramér's V
    - categorical ↔ numeric: sqrt(eta-squared)

    Args:
        df: Input DataFrame.
        categorical_vars: List of column names to treat as categorical. If None,
            columns with object or category dtype are used.

    Returns:
        pd.DataFrame: Square symmetric correlation matrix with 1.0 on the diagonal.
    """
    columns = df.columns
    n_vars = len(columns)
    corr_matrix = np.eye(n_vars)

    if categorical_vars is None:
        categorical_vars, _ = infer_column_types(df)
    categorical_set = set(categorical_vars)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            var1, var2 = columns[i], columns[j]
            # Drop rows where either column is missing, once, for every pair type.
            valid = (df[var1].notna() & df[var2].notna()).to_numpy()
            sub = df.loc[valid, [var1, var2]]

            v1_cat = var1 in categorical_set
            v2_cat = var2 in categorical_set

            if not v1_cat and not v2_cat:
                x = sub[var1].to_numpy(dtype=float)
                y = sub[var2].to_numpy(dtype=float)
                if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
                    corr = 0.0
                else:
                    corr = np.corrcoef(x, y)[0, 1]
            elif v1_cat and v2_cat:
                x = sub[var1].astype("category").cat.codes.to_numpy()
                y = sub[var2].astype("category").cat.codes.to_numpy()
                if len(x) < 1 or x.max() < 0 or y.max() < 0:
                    corr = 0.0
                else:
                    contingency = np.zeros((x.max() + 1, y.max() + 1), dtype=int)
                    np.add.at(contingency, (x, y), 1)
                    corr = cramers_v_np(contingency)
            else:
                if v1_cat:
                    cat = sub[var1].astype("category").cat.codes.to_numpy()
                    num = sub[var2].to_numpy(dtype=float)
                else:
                    cat = sub[var2].astype("category").cat.codes.to_numpy()
                    num = sub[var1].to_numpy(dtype=float)
                if len(num) < 1:
                    corr = 0.0
                else:
                    corr = np.sqrt(eta_squared_np(cat, num))

            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    return pd.DataFrame(corr_matrix, index=columns, columns=columns)


# ---------------------------------------------------------------------------
# Design quality metrics
# ---------------------------------------------------------------------------


def _standardize(X):
    """Column-standardize X; zero-variance columns yield 0 (std replaced by 1.0)."""
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    return (X - np.mean(X, axis=0)) / std


def _information_matrix(X_std):
    """Prepend an intercept column and return the information matrix X_model^T X_model."""
    X_model = np.column_stack((np.ones(X_std.shape[0]), X_std))
    return X_model.T @ X_model


def _continuous_columns(continuous_keys, subparam_mapping):
    """Combine continuous parameter keys with subparameter columns (from the mapping)."""
    subparam_keys = list(subparam_mapping.values()) if subparam_mapping else []
    return continuous_keys + subparam_keys


def _prepare_design_matrix(design, continuous_params_keys, subparam_keys):
    """
    Normalize ``subparam_keys``, combine with ``continuous_params_keys``, and return a
    standardized design matrix together with the combined key list.

    Shared setup used by all five scalar metric functions to avoid boilerplate duplication.

    Args:
        design: Design DataFrame.
        continuous_params_keys: List of continuous parameter column names.
        subparam_keys: List of subparameter column names, a single string, or None.

    Returns:
        tuple: ``(X_std, continuous_keys)`` where ``X_std`` is an ``(n, p)`` ndarray
            of column-standardized values and ``continuous_keys`` is the combined list
            of column names used.

    Note:
        Constant columns (zero variance) have their std replaced with 1.0 before
        dividing, so X_std contains 0 for those columns rather than NaN.
    """
    if subparam_keys is None:
        subparam_keys = []
    elif isinstance(subparam_keys, str):
        subparam_keys = [subparam_keys]
    continuous_keys = continuous_params_keys + subparam_keys
    X = design[continuous_keys].values
    X_std = _standardize(X)
    return X_std, continuous_keys


def calculate_d_optimality(design, continuous_params_keys, subparam_keys):
    """
    Calculate D-optimality (determinant of the information matrix) for the
    continuous and subparameter columns.

    Args:
        design: Design DataFrame.
        continuous_params_keys: List of continuous parameter column names.
        subparam_keys: List of subparameter column names, or a single string.

    Returns:
        float: Determinant of X^T X (higher is better).
    """
    X_std, _ = _prepare_design_matrix(design, continuous_params_keys, subparam_keys)
    XtX = _information_matrix(X_std)
    return linalg.det(XtX)


def calculate_a_optimality(design, continuous_params_keys, subparam_keys):
    """
    Calculate A-optimality (trace of the inverse information matrix) for the
    continuous and subparameter columns.

    Args:
        design: Design DataFrame.
        continuous_params_keys: List of continuous parameter column names.
        subparam_keys: List of subparameter column names, or a single string.

    Returns:
        float: Trace of (X^T X)^{-1} (lower is better).
    """
    X_std, _ = _prepare_design_matrix(design, continuous_params_keys, subparam_keys)
    XtX = _information_matrix(X_std)

    try:
        XtX_inv = np.linalg.inv(XtX)
        return np.trace(XtX_inv)
    except np.linalg.LinAlgError:
        return np.inf


def calculate_condition_number(design, continuous_params_keys, subparam_keys):
    """
    Calculate the condition number of the information matrix for the continuous and
    subparameter columns.

    Args:
        design: Design DataFrame.
        continuous_params_keys: List of continuous parameter column names.
        subparam_keys: List of subparameter column names, or a single string.

    Returns:
        float: Condition number of X^T X (lower is better).
    """
    X_std, _ = _prepare_design_matrix(design, continuous_params_keys, subparam_keys)
    XtX = _information_matrix(X_std)
    return np.linalg.cond(XtX)


def calculate_pairwise_distance_uniformity(design, continuous_params_keys, subparam_keys):
    """
    Calculate the coefficient of variation (CV) of pairwise Euclidean distances
    between design points in the continuous/subparameter space. Lower CV means more
    uniformly spread points (better space filling).

    Args:
        design: Design DataFrame.
        continuous_params_keys: List of continuous parameter column names.
        subparam_keys: List of subparameter column names, or a single string.

    Returns:
        float: CV of pairwise distances (lower is better).
    """
    X_std, _ = _prepare_design_matrix(design, continuous_params_keys, subparam_keys)
    distances = pdist(X_std, metric="euclidean")
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    return std_dist / mean_dist if mean_dist != 0 else np.inf


def calculate_max_continuous_correlation(design, continuous_params_keys, subparam_keys):
    """
    Calculate the maximum absolute Pearson correlation among all pairs of continuous
    and subparameter columns.

    Args:
        design: Design DataFrame.
        continuous_params_keys: List of continuous parameter column names.
        subparam_keys: List of subparameter column names, or a single string.

    Returns:
        float: Maximum off-diagonal absolute correlation (lower is better).
    """
    _, continuous_keys = _prepare_design_matrix(design, continuous_params_keys, subparam_keys)
    if not continuous_keys:
        return 0.0
    corr = design[continuous_keys].corr().abs().fillna(0).to_numpy()
    upper = corr[np.triu_indices_from(corr, k=1)]
    return upper.max() if upper.size else 0.0


def calculate_max_categorical_correlation(design, categorical_keys):
    """
    Calculate the maximum Cramér's V association among all pairs of categorical
    columns.

    Args:
        design: Design DataFrame.
        categorical_keys: List of categorical column names.

    Returns:
        float: Maximum off-diagonal Cramér's V (lower is better).
    """
    if not categorical_keys:
        return 0.0
    corr = calculate_mixed_correlation_matrix(
        design[categorical_keys], categorical_vars=categorical_keys
    ).to_numpy()
    upper = np.abs(corr)[np.triu_indices_from(corr, k=1)]
    return upper.max() if upper.size else 0.0


def calculate_max_mixed_correlation(design, continuous_keys, categorical_keys, subparam_mapping):
    """
    Calculate the maximum association between categorical and continuous/subparameter
    columns using eta-squared (sqrt). Structural parent–subparameter pairs (e.g.
    buffer_type ↔ pH) are excluded because their correlation is by design.

    Args:
        design: Design DataFrame.
        continuous_keys: List of continuous parameter column names.
        categorical_keys: List of categorical column names.
        subparam_mapping: Dict mapping each categorical variable to its subparameter.

    Returns:
        float: Maximum off-diagonal mixed association (lower is better).
    """
    all_continuous = _continuous_columns(continuous_keys, subparam_mapping)
    full_corr = calculate_mixed_correlation_matrix(design, categorical_vars=categorical_keys)

    max_mixed = 0.0
    for cat in categorical_keys:
        for cont in all_continuous:
            if subparam_mapping and cat in subparam_mapping and subparam_mapping[cat] == cont:
                continue
            if cat in full_corr.index and cont in full_corr.columns:
                corr_val = abs(full_corr.loc[cat, cont])
                max_mixed = max(max_mixed, corr_val)

    return max_mixed


# ---------------------------------------------------------------------------
# Metric registry — one source of truth for metric metadata and compute fns
# ---------------------------------------------------------------------------

_METRICS = {
    "D-optimality":       {"maximize": True,
        "compute": lambda d, ck, kk, sk, sm: calculate_d_optimality(d, ck, sk)},
    "A-optimality":       {"maximize": False,
        "compute": lambda d, ck, kk, sk, sm: calculate_a_optimality(d, ck, sk)},
    "Condition Number":   {"maximize": False,
        "compute": lambda d, ck, kk, sk, sm: calculate_condition_number(d, ck, sk)},
    "Pairwise Distance CV": {"maximize": False,
        "compute": lambda d, ck, kk, sk, sm: calculate_pairwise_distance_uniformity(d, ck, sk)},
    "Max Continuous Corr": {"maximize": False,
        "compute": lambda d, ck, kk, sk, sm: calculate_max_continuous_correlation(d, ck, sk)},
    "Max Categorical Corr": {"maximize": False,
        "compute": lambda d, ck, kk, sk, sm: calculate_max_categorical_correlation(d, kk)},
    "Max Mixed Corr":     {"maximize": False,
        "compute": lambda d, ck, kk, sk, sm: calculate_max_mixed_correlation(d, ck, kk, sm)},
}

assert list(_METRICS) == DEFAULT_METRICS
assert {k: v["maximize"] for k, v in _METRICS.items()} == _MAXIMIZE_BY_METRIC


# ---------------------------------------------------------------------------
# Dimensionality reduction plots
# ---------------------------------------------------------------------------


def _plot_embedding(design, continuous_params_keys, subparam_mapping, *, fit_2d,
                    title, xlabel, ylabel, default_color, empty_msg=None, hue=None):
    """Standardize continuous columns, project to 2D via fit_2d, and scatter (optional hue split)."""
    continuous_keys = _continuous_columns(continuous_params_keys, subparam_mapping)
    if not continuous_keys:
        msg = empty_msg if empty_msg is not None else (
            f"{title} requires at least one continuous or subparameter column."
        )
        raise ValueError(msg)
    X_std = _standardize(design[continuous_keys].values)
    coords = fit_2d(X_std)
    plt.figure(figsize=(8, 6))
    if hue and hue in design.columns:
        categories = design[hue].astype(str)
        for cat in categories.unique():
            mask = categories == cat
            plt.scatter(coords[mask, 0], coords[mask, 1], label=cat, alpha=0.7)
        plt.legend(title=hue)
    else:
        plt.scatter(coords[:, 0], coords[:, 1], c=default_color, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_pca(design, continuous_params_keys, subparam_mapping, hue=None):
    """
    Plot a PCA projection of the continuous and subparameter columns.

    Args:
        design: Design DataFrame.
        continuous_params_keys: List of continuous parameter column names.
        subparam_mapping: Dict mapping categorical variables to their subparameters.
        hue: Column name to use for color-coding. Defaults to None.
    """
    # empty-column check fires BEFORE sklearn import (preserves original error order)
    def fit_2d(X):
        try:
            from sklearn.decomposition import PCA
        except ImportError as exc:
            raise ImportError(
                "plot_pca requires scikit-learn. Install it with: pip install scikit-learn"
            ) from exc
        return PCA(n_components=2).fit_transform(X)

    _plot_embedding(
        design, continuous_params_keys, subparam_mapping,
        fit_2d=fit_2d,
        title="PCA of Continuous Design Variables",
        xlabel="PC1",
        ylabel="PC2",
        default_color="blue",
        empty_msg="plot_pca requires at least one continuous or subparameter column.",
        hue=hue,
    )


def plot_mds(design, continuous_params_keys, subparam_mapping, hue=None, metric="euclidean"):
    """
    Plot an MDS projection of the continuous and subparameter columns.

    Args:
        design: Design DataFrame.
        continuous_params_keys: List of continuous parameter column names.
        subparam_mapping: Dict mapping categorical variables to their subparameters.
        hue: Column name to use for color-coding. Defaults to None.
        metric: Distance metric passed to MDS. Defaults to ``'euclidean'``.
    """
    # empty-column check fires BEFORE sklearn import (preserves original error order)
    def fit_2d(X):
        try:
            from sklearn.manifold import MDS
        except ImportError as exc:
            raise ImportError(
                "plot_mds requires scikit-learn. Install it with: pip install scikit-learn"
            ) from exc
        return MDS(n_components=2, dissimilarity=metric, random_state=42, n_init=4).fit_transform(X)

    _plot_embedding(
        design, continuous_params_keys, subparam_mapping,
        fit_2d=fit_2d,
        title="MDS of Continuous Design Variables",
        xlabel="MDS Dimension 1",
        ylabel="MDS Dimension 2",
        default_color="green",
        empty_msg="plot_mds requires at least one continuous or subparameter column.",
        hue=hue,
    )


def plot_umap(
    design,
    continuous_params_keys,
    subparam_mapping,
    hue=None,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    verbose=False,
):
    """
    Plot a UMAP projection of the continuous and subparameter columns.

    Requires the ``umap-learn`` package (``pip install umap-learn``).

    Args:
        design: Design DataFrame.
        continuous_params_keys: List of continuous parameter column names.
        subparam_mapping: Dict mapping categorical variables to their subparameters.
        hue: Column name to use for color-coding. Defaults to None.
        n_neighbors: UMAP neighborhood size. Defaults to 15.
        min_dist: UMAP minimum distance parameter. Defaults to 0.1.
        metric: Distance metric for UMAP. Defaults to ``'euclidean'``.
        verbose: Whether to show UMAP's internal progress log. Defaults to False.

    Raises:
        ImportError: If ``umap-learn`` is not installed.
    """
    # umap import is eager (BEFORE empty-column check) — preserves original error order
    try:
        import umap.umap_ as umap
    except ImportError as exc:
        raise ImportError(
            "plot_umap requires umap-learn. Install it with: pip install umap-learn"
        ) from exc

    def fit_2d(X):
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42,
            n_jobs=1,
            verbose=verbose,
        )
        return reducer.fit_transform(X)

    _plot_embedding(
        design, continuous_params_keys, subparam_mapping,
        fit_2d=fit_2d,
        title="UMAP of Continuous Design Variables",
        xlabel="UMAP Dimension 1",
        ylabel="UMAP Dimension 2",
        default_color="red",
        empty_msg="plot_umap requires at least one continuous or subparameter column.",
        hue=hue,
    )


# ---------------------------------------------------------------------------
# Design evaluation
# ---------------------------------------------------------------------------


def evaluate_design(design, continuous_keys, categorical_keys, subparam_mapping, metrics_to_optimize):
    """
    Evaluate a design using the specified quality metrics.

    Args:
        design: Design DataFrame to evaluate.
        continuous_keys: List of continuous parameter column names.
        categorical_keys: List of categorical column names.
        subparam_mapping: Dict mapping categorical variables to their subparameters
            (e.g. ``{'buffer': 'pH', 'sugar': '[Sugar] (%)'}``).
        metrics_to_optimize: List of metric names to compute. Valid values are the
            entries of ``DEFAULT_METRICS`` (``'D-optimality'``, ``'A-optimality'``,
            ``'Condition Number'``, ``'Pairwise Distance CV'``, ``'Max Continuous Corr'``,
            ``'Max Categorical Corr'``, ``'Max Mixed Corr'``).

    Returns:
        dict: Computed metric values for the requested metrics only.
    """
    subparam_keys = list(subparam_mapping.values()) if subparam_mapping else []
    requested = set(metrics_to_optimize)
    return {
        name: _METRICS[name]["compute"](
            design, continuous_keys, categorical_keys, subparam_keys, subparam_mapping)
        for name in DEFAULT_METRICS
        if name in requested
    }


# ---------------------------------------------------------------------------
# Parallel design optimization helpers
# ---------------------------------------------------------------------------


def generate_and_evaluate(
    seed,
    n_samples,
    continuous_params,
    conditional_subparameters,
    subparam_mapping,
    continuous_keys,
    categorical_keys,
    metrics_to_optimize,
    n_category_trials=100,
    corr_threshold=0.01,
):
    """
    Generate a single candidate design and compute its quality metrics.

    This function is a top-level callable so it can be pickled by
    ``ProcessPoolExecutor``.

    Args:
        seed: Random seed for this candidate.
        n_samples: Number of rows in the candidate design.
        continuous_params: Continuous parameter specifications.
        conditional_subparameters: Conditional subparameter specifications.
        subparam_mapping: Subparameter mapping dict.
        continuous_keys: List of continuous parameter column names.
        categorical_keys: List of categorical column names.
        metrics_to_optimize: List of metric names to compute.
        n_category_trials: Passed to ``sample_design`` for category optimization.
            Defaults to 100.
        corr_threshold: Passed to ``sample_design`` for category optimization.
            Defaults to 0.01.

    Returns:
        dict: Contains ``'seed'``, ``'design'``, one key per metric, and
            ``'metric_values'`` (list in the same order as
            ``metrics_to_optimize``).
    """
    design = sample_design(
        seed=seed,
        n_samples=n_samples,
        continuous_params=continuous_params,
        conditional_subparameters=conditional_subparameters,
        subparam_mapping=subparam_mapping,
        n_category_trials=n_category_trials,
        corr_threshold=corr_threshold,
    )
    metrics = evaluate_design(
        design=design,
        continuous_keys=continuous_keys,
        categorical_keys=categorical_keys,
        subparam_mapping=subparam_mapping,
        metrics_to_optimize=metrics_to_optimize,
    )
    metric_values = [metrics[m] for m in metrics_to_optimize]
    return {"seed": seed, "design": design, **metrics, "metric_values": metric_values}


def _composite_scores(metric_array, maximize_metrics):
    """Min-max normalize each metric column, invert minimize-metrics, and sum to one score per candidate.

    metric_array: (n_candidates, n_metrics). Constant columns contribute 0; NaN normalizes to 0.
    Returns a (n_candidates,) score array (higher is better).
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.array([
            (vals - vals.min()) / (vals.max() - vals.min())
            if vals.max() != vals.min() else np.zeros_like(vals)
            for vals in metric_array.T
        ])
    norm = np.nan_to_num(norm, nan=0.0)
    for idx, maximize in enumerate(maximize_metrics):
        if not maximize:
            norm[idx] = 1 - norm[idx]
    return norm.sum(axis=0)


def _run_parallel_search(candidate_fn, submit_arg_lists, *, maximize_metrics,
                         max_workers, use_tqdm, tqdm_desc="Optimizing Designs"):
    """Fan candidate_fn over submit_arg_lists in a ProcessPoolExecutor, score, return (records, scores, best_idx)."""
    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(candidate_fn, *args) for args in submit_arg_lists]
        completed = concurrent.futures.as_completed(futures)
        if use_tqdm:
            try:
                from tqdm import tqdm as _tqdm
                completed = _tqdm(completed, total=len(futures), desc=tqdm_desc)
            except ImportError:
                pass
        for future in completed:
            records.append(future.result())
    metric_array = np.array([r["metric_values"] for r in records])
    scores = _composite_scores(metric_array, maximize_metrics)
    best_idx = int(np.argmax(scores))
    return records, scores, best_idx


def find_best_design_parallel(
    n,
    n_samples,
    continuous_params,
    conditional_subparameters,
    subparam_mapping=None,
    metrics_to_optimize=None,
    maximize_metrics=None,
    seed_start=0,
    max_workers=None,
    n_category_trials=100,
    corr_threshold=0.01,
):
    """
    Generate ``n`` candidate designs in parallel and return the one with the
    highest composite score.

    Args:
        n: Number of candidate designs to evaluate.
        n_samples: Number of rows per candidate design.
        continuous_params: Continuous parameter specifications.
        conditional_subparameters: Conditional subparameter specifications.
        subparam_mapping: Subparameter mapping dict. Inferred if not provided.
        metrics_to_optimize: List of metric names. Defaults to all seven standard
            metrics.
        maximize_metrics: List of booleans indicating whether to maximize (True) or
            minimize (False) each metric. Defaults to
            ``[True, False, False, ...]`` — maximize D-optimality only.
        seed_start: First seed used; subsequent candidates use
            ``seed_start + i``. Defaults to 0.
        max_workers: Maximum worker processes. Defaults to None (all CPUs).
        n_category_trials: Forwarded to each worker's ``sample_design`` call.
            Defaults to 100.
        corr_threshold: Forwarded to each worker's ``sample_design`` call.
            Defaults to 0.01.

    Returns:
        tuple: ``(best_design, metrics_df)`` where ``best_design`` is the
            highest-scoring pd.DataFrame and ``metrics_df`` summarizes all
            candidates.
    """
    if subparam_mapping is None:
        subparam_mapping = infer_subparam_mapping(conditional_subparameters)
    if metrics_to_optimize is None:
        metrics_to_optimize = list(DEFAULT_METRICS)
    if maximize_metrics is None:
        maximize_metrics = _default_maximize_metrics(metrics_to_optimize)
    if len(maximize_metrics) != len(metrics_to_optimize):
        raise ValueError(
            f"maximize_metrics has {len(maximize_metrics)} entries but "
            f"metrics_to_optimize has {len(metrics_to_optimize)}. "
            "They must have the same length."
        )

    continuous_keys = list(continuous_params.keys())
    categorical_keys = list(conditional_subparameters.keys())

    submit_args = [
        (seed_start + i, n_samples, continuous_params, conditional_subparameters,
         subparam_mapping, continuous_keys, categorical_keys, metrics_to_optimize,
         n_category_trials, corr_threshold)
        for i in range(n)
    ]
    records, scores, best_idx = _run_parallel_search(
        generate_and_evaluate, submit_args,
        maximize_metrics=maximize_metrics, max_workers=max_workers, use_tqdm=True)
    for i, r in enumerate(records):
        r["score"] = scores[i]
    best_design = records[best_idx]["design"]
    metrics_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k not in ["design", "metric_values"]} for r in records])
    return best_design, metrics_df


def plot_design_quality_evolution(metrics_df):
    """
    Plot per-metric bar charts over trial seeds to visualize design quality evolution.

    Args:
        metrics_df: DataFrame with a ``'seed'`` column and one column per metric.
    """
    metrics_df = metrics_df.sort_values("seed")
    metrics = [c for c in metrics_df.columns if c not in ("seed", "score")]
    n_metrics = len(metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = np.array(axes).flatten()

    seed_labels = metrics_df["seed"].astype(str).tolist()
    n_bars = len(seed_labels)
    tick_every = max(1, n_bars // 10)
    tick_positions = list(range(0, n_bars, tick_every))

    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.bar(range(n_bars), metrics_df[metric])
        ax.set_title(f"{metric} vs Seed")
        ax.set_xlabel("Seed")
        ax.set_ylabel(metric)
        ax.xaxis.set_major_locator(plt.FixedLocator(tick_positions))
        ax.xaxis.set_major_formatter(plt.FixedFormatter([seed_labels[j] for j in tick_positions]))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y")

    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(design, categorical_vars):
    """
    Plot a heatmap of the mixed correlation matrix.

    Args:
        design: Design DataFrame.
        categorical_vars: List of categorical column names (used to select the
            appropriate correlation measure).
    """
    try:
        import seaborn as sns
    except ImportError as exc:
        raise ImportError("plot_correlation_matrix requires seaborn. Install it with: pip install seaborn") from exc
    corr_df = calculate_mixed_correlation_matrix(design, categorical_vars)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Mixed Correlation Matrix")
    plt.tight_layout()
    plt.show()


def plot_design_histograms(
    design,
    continuous_keys,
    categorical_keys,
    subparam_mapping=None,
    bins=50,
    figsize=(18, 10),
):
    """
    Plot histograms for continuous parameters and bar charts for categorical
    variables, with stacked histograms for subparameters colored by parent category.

    Args:
        design: Design DataFrame.
        continuous_keys: List of continuous parameter column names.
        categorical_keys: List of categorical column names.
        subparam_mapping: Dict mapping categorical variables to their subparameters.
            Defaults to None.
        bins: Number of bins for histograms. Defaults to 50.
        figsize: Figure size tuple. Defaults to (18, 10).
    """
    try:
        import seaborn as sns
    except ImportError as exc:
        raise ImportError("plot_design_histograms requires seaborn. Install it with: pip install seaborn") from exc
    total_plots = len(continuous_keys) + len(categorical_keys) + (len(subparam_mapping) if subparam_mapping else 0)

    ncols = 3
    nrows = math.ceil(total_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, key in enumerate(continuous_keys):
        ax = axes[i]
        ax.hist(design[key].dropna(), bins=bins, color="skyblue", edgecolor="black")
        ax.set_title(f"Histogram of {key}")
        ax.set_xlabel(key)
        ax.set_ylabel("Frequency")

    offset = len(continuous_keys)
    for j, key in enumerate(categorical_keys):
        ax = axes[offset + j]
        counts = design[key].dropna().value_counts()
        ax.bar(
            counts.index.astype(str),
            counts.values,
            color="lightgreen",
            edgecolor="black",
        )
        ax.set_title(f"Bar plot of {key}")
        ax.set_xlabel(key)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

    if subparam_mapping:
        offset = len(continuous_keys) + len(categorical_keys)
        for k, (cat_key, sub_key) in enumerate(subparam_mapping.items()):
            ax = axes[offset + k]
            sns.histplot(
                data=design,
                x=sub_key,
                hue=cat_key,
                bins=bins,
                multiple="stack",
                palette="Set2",
                edgecolor="black",
                ax=ax,
            )
            ax.set_title(f"Histogram of {sub_key} by {cat_key}")
            ax.set_xlabel(sub_key)
            ax.set_ylabel("Count")

    for k in range(total_plots, len(axes)):
        axes[k].axis("off")

    plt.tight_layout()
    plt.show()


def evaluate_candidate(
    i,
    seed_start,
    n,
    continuous_params,
    conditional_subparameters,
    subparam_mapping,
    existing_design,
    continuous_keys,
    categorical_keys,
    metrics_to_optimize,
    n_category_trials=100,
    corr_threshold=0.01,
):
    """
    Generate ``n`` new samples, append them to ``existing_design``, and compute
    quality metrics for the combined design.

    This function is a top-level callable so it can be pickled by
    ``ProcessPoolExecutor``.

    Args:
        i: Candidate index (added to ``seed_start`` to form the random seed).
        seed_start: Base seed value.
        n: Number of new samples to generate.
        continuous_params: Continuous parameter specifications.
        conditional_subparameters: Conditional subparameter specifications.
        subparam_mapping: Subparameter mapping dict.
        existing_design: DataFrame of existing design rows.
        continuous_keys: List of continuous parameter column names.
        categorical_keys: List of categorical column names.
        metrics_to_optimize: List of metric names to compute.
        n_category_trials: Forwarded to ``sample_design``. Defaults to 100.
        corr_threshold: Forwarded to ``sample_design``. Defaults to 0.01.

    Returns:
        dict: Contains ``'seed'``, ``'metrics'`` (dict), ``'metric_values'``
            (list), and ``'new_samples'`` (DataFrame).
    """
    if subparam_mapping is None:
        subparam_mapping = infer_subparam_mapping(conditional_subparameters)

    seed = seed_start + i
    new_samples = sample_design(
        seed,
        n,
        continuous_params,
        conditional_subparameters,
        subparam_mapping,
        n_category_trials=n_category_trials,
        corr_threshold=corr_threshold,
    )
    combined_design = pd.concat([existing_design, new_samples], ignore_index=True)

    # Compute all standard metrics for the summary; filter to metrics_to_optimize for scoring
    all_metrics = evaluate_design(
        design=combined_design,
        continuous_keys=continuous_keys,
        categorical_keys=categorical_keys,
        subparam_mapping=subparam_mapping,
        metrics_to_optimize=list(DEFAULT_METRICS),
    )

    return {
        "seed": seed,
        "metrics": all_metrics,
        "metric_values": [all_metrics[m] for m in metrics_to_optimize],
        "new_samples": new_samples,
    }


def extend_design(
    existing_design,
    n,
    continuous_params,
    conditional_subparameters,
    subparam_mapping=None,
    metrics_to_optimize=None,
    maximize_metrics=None,
    n_trials=10,
    seed_start=1000,
    max_workers=None,
    n_category_trials=100,
    corr_threshold=0.01,
):
    """
    Extend an existing design by finding the best set of ``n`` new samples from
    among multiple candidates evaluated in parallel.

    Args:
        existing_design: DataFrame of the existing design.
        n: Number of new samples to add.
        continuous_params: Continuous parameter specifications.
        conditional_subparameters: Conditional subparameter specifications.
        subparam_mapping: Subparameter mapping dict. Inferred if not provided.
        metrics_to_optimize: List of metric names. Defaults to all seven standard
            metrics.
        maximize_metrics: List of booleans indicating direction for each metric.
            Defaults to ``[True, False, False, ...]``.
        n_trials: Number of candidate extensions to evaluate. Defaults to 10.
        seed_start: Starting seed for candidate generation. Defaults to 1000.
        max_workers: Maximum worker processes. Defaults to None (all CPUs).
        n_category_trials: Forwarded to each worker's ``sample_design`` call.
            Defaults to 100.
        corr_threshold: Forwarded to each worker's ``sample_design`` call.
            Defaults to 0.01.

    Returns:
        tuple: ``(extended_design, metrics_summary)`` where ``extended_design``
            contains all original rows plus the best new rows, and
            ``metrics_summary`` is a pd.DataFrame of candidate scores.
    """
    if subparam_mapping is None:
        subparam_mapping = infer_subparam_mapping(conditional_subparameters)
    if metrics_to_optimize is None:
        metrics_to_optimize = list(DEFAULT_METRICS)
    if maximize_metrics is None:
        maximize_metrics = _default_maximize_metrics(metrics_to_optimize)
    if len(maximize_metrics) != len(metrics_to_optimize):
        raise ValueError(
            f"maximize_metrics has {len(maximize_metrics)} entries but "
            f"metrics_to_optimize has {len(metrics_to_optimize)}. "
            "They must have the same length."
        )

    continuous_keys = list(continuous_params.keys())
    categorical_keys = list(conditional_subparameters.keys())

    submit_args = [
        (i, seed_start, n, continuous_params, conditional_subparameters,
         subparam_mapping, existing_design, continuous_keys, categorical_keys,
         metrics_to_optimize, n_category_trials, corr_threshold)
        for i in range(n_trials)
    ]
    records, scores, best_idx = _run_parallel_search(
        evaluate_candidate, submit_args,
        maximize_metrics=maximize_metrics, max_workers=max_workers, use_tqdm=False)
    best_extension = records[best_idx]["new_samples"]
    extended_design = pd.concat([existing_design, best_extension], ignore_index=True)
    metrics_summary = pd.DataFrame(
        [{**{"seed": r["seed"], "score": s}, **r["metrics"]} for r, s in zip(records, scores)])
    return extended_design, metrics_summary
