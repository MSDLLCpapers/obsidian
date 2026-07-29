"""Matplotlib figure-generating functions"""

import numbers
from itertools import product
from math import prod

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from obsidian.campaign import Campaign
from obsidian.optimizer import Optimizer
from obsidian.parameters.targets import resolve_target_names
from obsidian.plotting.branding import magenta, obsidian_colors


def plot_ofat_ranges(optimizer: Optimizer, ofat_ranges: pd.DataFrame) -> Figure:
    """
    Plots each parameter's 1D OFAT acceptable range

    Args:
        optimizer (Optimizer): The optimizer object which contains a surrogate
            that has been fit to data and can be used to make predictions.
        ofat_ranges (pd.DataFrame): A DataFrame containing the acceptable range
            values for each parameter, at the low bound, average, and high bound.

    Returns:
        Figure: The parameter OFAT acceptable-range plot
    """

    fig = plt.figure(figsize=(2 * len(ofat_ranges), 4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Iterate over the parameters
    for i, (p_name, row) in enumerate(ofat_ranges.iterrows()):
        color = colors[i]

        # Plot as a bar chart; x-axis is the parameter name, y-axis is the scaled value
        plt.plot(
            [p_name, p_name],
            [row["Min_LB"], row["Max_LB"]],
            linewidth=6,
            linestyle="solid",
            color=color,
            label="High Confidence" if i == 0 else None,
        )

        # If the edges of LB are too close to mean, only annotate LB (higher conf)
        if row["Min_LB"] > row["Min_Mu"]:
            plt.annotate(
                f'{(optimizer.X_space[i].unit_demap(row["Min_LB"])):.2f}',
                xy=(i, row["Min_LB"]),
                xytext=(i + 0.25, row["Min_LB"]),
                fontsize=8,
                ha="left",
                va="center",
                rotation=0,
                arrowprops=dict(arrowstyle="-", color=color, lw=1),
            )
        if row["Max_LB"] < row["Max_Mu"]:
            plt.annotate(
                f'{(optimizer.X_space[i].unit_demap(row["Max_LB"])):.2f}',
                xy=(i, row["Max_LB"]),
                xytext=(i + 0.25, row["Max_LB"]),
                fontsize=8,
                ha="left",
                va="center",
                rotation=0,
                arrowprops=dict(arrowstyle="-", color=color, lw=1),
            )

        plt.plot(
            [p_name, p_name],
            [row["Min_Mu"], row["Max_Mu"]],
            linewidth=3,
            linestyle="solid",
            color=color,
            label="Average" if i == 0 else None,
        )

        # If the edges of the mean are too close to the UB, only annotate mean (higher conf)
        plt.annotate(
            f'{(optimizer.X_space[i].unit_demap(row["Min_Mu"])):.2f}',
            xy=(i, row["Min_Mu"]),
            xytext=(i + 0.25, row["Min_Mu"]),
            fontsize=8,
            ha="left",
            va="center",
            rotation=0,
            arrowprops=dict(arrowstyle="-", color=color, lw=1),
        )
        plt.annotate(
            f'{(optimizer.X_space[i].unit_demap(row["Max_Mu"])):.2f}',
            xy=(i, row["Max_Mu"]),
            xytext=(i + 0.25, row["Max_Mu"]),
            fontsize=8,
            ha="left",
            va="center",
            rotation=0,
            arrowprops=dict(arrowstyle="-", color=color, lw=1),
        )

        # Only plot UB if it isn't already encompassed by higher-confidence ranges
        if row["Min_UB"] < row["Min_Mu"]:
            plt.plot([p_name, p_name], [row["Min_UB"], row["Min_Mu"]], linewidth=1, linestyle=":", color=color)
        if row["Max_UB"] > row["Max_Mu"]:
            plt.plot([p_name, p_name], [row["Max_UB"], row["Max_Mu"]], linewidth=1, linestyle=":", color=color)
        plt.plot([0], [0], linewidth=1, linestyle=":", color=color, label="Low Confidence" if i == 0 else None)

        # Never annotate UB (low confidence)

    alpha = ofat_ranges["PI Range"].mode().iloc[0]
    LCL = (1 - alpha) / 2
    UCL = 1 - LCL
    comparator = ">" if row["Aim"] == "max" else "<"

    plt.xticks(rotation=90)
    plt.ylabel("Parameter Value (Scaled)")
    plt.ylim([-0.15, 1.15])
    plt.xlim([-1, len(ofat_ranges)])
    plt.title(
        "Univariate Range (OFAT) Estimates from APO Model \n"
        + f'Ranges Satisfying {row["Response"]} '
        + comparator
        + f' {row["Threshold"]} \n'
        + f"Confidence Range: {LCL*100:.1f} - {UCL*100:.1f}%",
        fontsize=10,
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.close(fig)

    return fig


def plot_interactions(optimizer: Optimizer, cor: np.ndarray, clamp: bool = False):
    """
    Plots the parameter interaction matrix

    Args:
        optimizer (ptimizer): The optimizer object which contains a surrogate
            that has been fit to data and can be used to make predictions.
        cor (np.ndarray): The correlation matrix representing the parameter interactions.
        clamp (bool, optional): Whether to clamp the colorbar range to (0, 1).
            Defaults to ``False``.

    Returns:
        Figure: The parameter interaction plot
    """

    fig = plt.figure(figsize=(4, 4))
    ax = fig.gca()

    # Use matrix imshow to plot correlation matrix
    cax = ax.matshow(cor)
    if clamp:
        cax.set_clim(0, 1)

    # Set axis labels and ticks
    axis = np.arange(len(optimizer.X_space.X_names))
    names = optimizer.X_space.X_names
    ax.set_xticks(axis)
    ax.set_xticklabels(names, rotation=90)
    ax.set_yticks(axis)
    ax.set_yticklabels(names, rotation=0)
    cbar = fig.colorbar(cax)
    ax.set_title("Parameter Interactions")
    cbar.ax.set_ylabel("Range Shrinkage")

    # Add text annotations if correlation is greater than 0.05
    for (i, j), z in np.ndenumerate(cor):
        if z > 0.05:
            ax.text(j, i, "{:0.2f}".format(z), ha="center", va="center", fontsize=8)
    plt.close(fig)

    return fig


_CONFIDENCE_DEFAULT_ALPHAS = [0.55, 0.55, 0.85, 1.0]
_CONFIDENCE_LABELS = ["Fail", "Pass (mean)", "Pass (70% CI)", "Pass (95% CI)"]

_PASSFAIL_FAIL_COLOR = magenta
_PASSFAIL_PASS_COLOR = obsidian_colors.primary.teal

# 4-level confidence palette, ordered fail → confident pass. The two pass-tier
# teals share a hue but differ in saturation/value so they read as a confidence
# ramp rather than two indistinguishable greens.
_CONFIDENCE_COLORS = [
    magenta,
    obsidian_colors.accent.lemon,
    obsidian_colors.secondary.light_teal,
    obsidian_colors.primary.teal,
]


def _format_value(v) -> str:
    """Format a row/col level value for tick labels."""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{v:.4g}"
    return str(v)


def _validate_response_map_params(
    campaign: Campaign,
    row_params: dict,
    col_params: dict,
    fixed_params: dict | None,
    target_names: list[str] | None,
    param_display_names: dict | None,
    target_display_names: dict | None,
    mode: str,
    include_joint: bool,
) -> dict:
    """Validate inputs and resolve smart defaults for ``plot_2d_response_map``.

    Enforces:
      - Campaign is fitted.
      - row/col/fixed names exist in the campaign and don't overlap each other.
      - Every parameter in ``X_space`` is covered by exactly one of the three
        categories (no silent midpoint defaults).
      - row/col values are non-empty lists.
      - Targets exist; thresholds are present where required by the mode.
    """
    if not campaign.optimizer.is_fit:
        raise ValueError("Campaign optimizer must be fitted before plotting")

    if mode not in ("passfail", "confidence", "continuous"):
        raise ValueError(f"mode must be one of 'passfail', 'confidence', 'continuous'; got {mode!r}")

    require_thresholds = mode in ("passfail", "confidence")

    selected = resolve_target_names(
        campaign.target,
        target_names,
        require_thresholds=require_thresholds,
        drop_tracking_only=True,
    )
    target_names = [t.name for t in selected]
    thresholds = {t.name: t.threshold for t in selected} if require_thresholds else None
    # Continuous mode: one panel per cell-grid, so a single target only.
    # Threshold is irrelevant — we plot the mean (or std).
    if mode == "continuous" and len(target_names) != 1:
        raise ValueError(
            f"mode='continuous' requires exactly one target (got {len(target_names)}: {target_names}). "
            "Pass target_names=['<one>'] to select."
        )

    # Validate row/col/fixed dicts
    if not isinstance(row_params, dict) or not row_params:
        raise ValueError("row_params must be a non-empty dict {name: [values, ...]}")
    if not isinstance(col_params, dict) or not col_params:
        raise ValueError("col_params must be a non-empty dict {name: [values, ...]}")
    fixed_params = dict(fixed_params) if fixed_params else {}

    all_names = list(campaign.X_space.X_names)
    all_names_set = set(all_names)

    def _check_dict(d, label, allow_scalar=False):
        for name, values in d.items():
            if name not in all_names_set:
                raise ValueError(f"{label}[{name!r}]: parameter not in campaign. Valid: {all_names}")
            if allow_scalar:
                # fixed_params expects a single value per parameter (numeric or
                # categorical string). Lists/arrays would silently broadcast
                # into _build_response_map_eval_df and corrupt the eval grid.
                if not isinstance(values, (numbers.Number, str)):
                    raise ValueError(
                        f"{label}[{name!r}] must be a scalar (number or string), got {type(values).__name__}"
                    )
                continue
            if not isinstance(values, (list, tuple, np.ndarray)) or isinstance(values, str):
                raise ValueError(f"{label}[{name!r}] must be a list of values, got {type(values).__name__}")
            if len(values) < 1:
                raise ValueError(f"{label}[{name!r}] must contain at least one value")

    _check_dict(row_params, "row_params")
    _check_dict(col_params, "col_params")
    _check_dict(fixed_params, "fixed_params", allow_scalar=True)

    used = []
    for d in (row_params, col_params, fixed_params):
        used.extend(d.keys())
    dups = {n for n in used if used.count(n) > 1}
    if dups:
        raise ValueError(f"Parameter(s) appear in more than one of row/col/fixed: {sorted(dups)}")

    missing_cov = [n for n in all_names if n not in used]
    if missing_cov:
        raise ValueError(
            "Every parameter must appear in exactly one of row_params, col_params, or fixed_params. "
            f"Missing: {missing_cov}"
        )

    return {
        "target_names": target_names,
        "thresholds": thresholds,
        "fixed_params": fixed_params,
        "param_display_names": param_display_names,
        "target_display_names": target_display_names,
        "all_param_names": all_names,
        "include_joint": include_joint and require_thresholds and len(target_names) > 1,
    }


def _compute_response_map_layout(row_params: dict, col_params: dict) -> dict:
    """Compute the row × column Cartesian product for the response map.

    Convention: the **first key** in each dict is the *innermost* level (closest
    to the cells); the last key is outermost. The Cartesian product is built
    outermost-first so that row/col index 0 corresponds to the visual top-left,
    matching the reference plot.
    """
    row_levels = list(reversed(list(row_params.items())))  # outermost → innermost
    col_levels = list(reversed(list(col_params.items())))

    row_combos = list(product(*[v for _, v in row_levels])) if row_levels else [()]
    col_combos = list(product(*[v for _, v in col_levels])) if col_levels else [()]

    R = len(row_combos)
    C = len(col_combos)

    return {
        "R": R,
        "C": C,
        "row_levels": row_levels,
        "col_levels": col_levels,
        "row_param_names": [n for n, _ in row_levels],
        "col_param_names": [n for n, _ in col_levels],
        "row_combos": row_combos,
        "col_combos": col_combos,
    }


def _build_response_map_eval_df(
    layout: dict,
    fixed_params: dict,
    all_param_names: list[str],
) -> pd.DataFrame:
    """Build a single (R*C, n_params) DataFrame for one batched predict() call.

    Row order is row-major over (row_combo, col_combo): cell (r, c) is at index
    ``r * C + c``. Reshape predictions back with ``.reshape(R, C)``.
    """
    R, C = layout["R"], layout["C"]
    row_combos = layout["row_combos"]
    col_combos = layout["col_combos"]
    row_names = layout["row_param_names"]
    col_names = layout["col_param_names"]

    # Build per-column arrays of length R*C
    columns: dict[str, np.ndarray] = {}

    if row_names:
        row_block = np.array(row_combos, dtype=object)  # (R, len(row_names))
        row_repeat = np.repeat(row_block, C, axis=0)  # (R*C, len(row_names))
        for k, name in enumerate(row_names):
            columns[name] = row_repeat[:, k]
    if col_names:
        col_block = np.array(col_combos, dtype=object)  # (C, len(col_names))
        col_tile = np.tile(col_block, (R, 1))  # (R*C, len(col_names))
        for k, name in enumerate(col_names):
            columns[name] = col_tile[:, k]
    for name, value in fixed_params.items():
        columns[name] = np.full(R * C, value, dtype=object)

    df = pd.DataFrame({name: columns[name] for name in all_param_names})

    # Promote numeric-looking object columns to numeric so downstream encoders
    # treat continuous params as floats (categorical params remain as strings).
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col])
            except (TypeError, ValueError):
                pass
    return df


def _posterior_std(campaign: Campaign, df: pd.DataFrame, target_name: str) -> np.ndarray:
    """Return the GP posterior std for ``target_name`` over the rows of ``df``.

    Recovers std from a 95% prediction interval using ``(ub - lb) / (2 * 1.96)``.
    """
    from scipy.stats import norm

    z_ref = float(norm.ppf(0.975))
    preds = campaign.optimizer.predict(df, return_f_inv=True, PI_range=0.95)
    lb = preds[f"{target_name} lb"].values
    ub = preds[f"{target_name} ub"].values
    return np.abs(ub - lb) / (2.0 * z_ref)


def _resolve_passfail_PI_range(PI_range: float | None, confidence_level: str | None) -> tuple[float, str]:
    """Apply the legacy passfail mode/PI_range resolution semantics.

    Returns the effective (PI_range, mode_description) pair. ``confidence_level``
    overrides ``PI_range`` when set; the description is used in the legend.
    """
    import warnings

    if confidence_level not in (None, "mean", "70%", "95%"):
        raise ValueError(f"confidence_level must be None, 'mean', '70%', or '95%', got {confidence_level!r}")

    if confidence_level == "mean":
        if PI_range is not None:
            warnings.warn(
                "PI_range is ignored when confidence_level='mean'; pass/fail evaluated on the mean.",
                UserWarning,
            )
        return 0.0, "mean prediction"
    if confidence_level == "70%":
        if PI_range is not None and PI_range != 0.7:
            warnings.warn("PI_range overridden by confidence_level='70%'.", UserWarning)
        return 0.7, "70% CI"
    if confidence_level == "95%":
        if PI_range is not None and PI_range != 0.95:
            warnings.warn("PI_range overridden by confidence_level='95%'.", UserWarning)
        return 0.95, "95% CI"

    # Legacy: caller-supplied PI_range
    eff = 0.7 if PI_range is None else PI_range
    if eff not in (0.7, 0.95):
        raise ValueError(f"PI_range must be 0.7 or 0.95, got {eff}")
    return eff, f"{int(round(eff * 100))}% CI"


def _setup_response_map_axes(
    n_panels: int,
    R: int,
    C: int,
    fig_width: float,
    has_top_colorbar: bool,
) -> dict:
    """Create the figure + side-by-side panel axes (one per target / joint).

    Reserves left/bottom gutters proportional to the row/col-hierarchy depth
    so the hierarchical labels have somewhere to live. Returns the figure,
    the axes array, and the gridspec margins for downstream label placement.
    """
    # Per-panel margins (fractions of the total figure)
    gs_left = 0.14
    gs_right = 0.98
    gs_top = 0.80 if has_top_colorbar else 0.88
    gs_bottom = 0.22

    plot_width = (gs_right - gs_left) * fig_width
    panel_width = plot_width / n_panels
    cell_width = panel_width / C
    plot_height = cell_width * R
    fig_height = plot_height / (gs_top - gs_bottom)
    fig_height = max(fig_height, 2.5)

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        1,
        n_panels,
        wspace=0.06,
        left=gs_left,
        right=gs_right,
        top=gs_top,
        bottom=gs_bottom,
    )

    axes = np.empty((1, n_panels), dtype=object)
    for j in range(n_panels):
        axes[0, j] = fig.add_subplot(gs[0, j])

    return {
        "fig": fig,
        "axes": axes,
        "gs_left": gs_left,
        "gs_right": gs_right,
        "gs_top": gs_top,
        "gs_bottom": gs_bottom,
    }


def _style_response_map_panel(ax, R: int, C: int) -> None:
    """Square cells, white minor grid, no major ticks (labels go in the gutter).

    Y limits are inverted (R-0.5 → -0.5) so row index 0 sits at the visual top,
    matching ``origin="upper"`` in ``imshow`` and the row-label placement in
    ``_add_hierarchical_labels``.
    """
    ax.set_xlim(-0.5, C - 0.5)
    ax.set_ylim(R - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(C + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(R + 1) - 0.5, minor=True)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="minor", color="white", linewidth=1.0, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _add_hierarchical_labels(
    fig: Figure,
    axes: np.ndarray,
    layout: dict,
    panel_labels: list[str],
    param_display_names: dict[str, str],
    margins: dict,
    base_fontsize: float = 8.0,
) -> None:
    """Render nested row labels on the left and nested col labels on the bottom.

    For each panel:
      - column labels stack below the panel, innermost level closest to the cells
      - row labels stack to the LEFT of the leftmost panel, innermost closest
      - thin gray separator strokes span between outermost-level groups
      - per-panel target name centered above each panel

    ``base_fontsize`` sets the value-label size; param names render at
    +1pt and panel titles at +3pt so the three tiers stay visually ordered.
    """
    R, C = layout["R"], layout["C"]
    row_levels = layout["row_levels"]  # outermost → innermost
    col_levels = layout["col_levels"]
    n_panels = axes.shape[1]

    fs_value = base_fontsize
    fs_param = base_fontsize + 1.0
    fs_title = base_fontsize + 3.0

    for j in range(n_panels):
        ax = axes[0, j]
        bbox = ax.get_position()
        panel_left, panel_right = bbox.x0, bbox.x1
        panel_top, panel_bottom = bbox.y1, bbox.y0
        cell_w = (panel_right - panel_left) / C
        cell_h = (panel_top - panel_bottom) / R

        # Title above each panel
        fig.text(
            (panel_left + panel_right) / 2,
            panel_top + 0.015,
            panel_labels[j],
            ha="center",
            va="bottom",
            fontsize=fs_title,
            fontweight="bold",
            transform=fig.transFigure,
        )

        # Column labels: walk from innermost (closest to plot) to outermost.
        # The innermost level sits in band 0 directly below the panel; deeper
        # levels stack outwards.
        col_label_band = 0.022
        col_levels_inner_first = list(reversed(col_levels))  # innermost first
        n_col_levels = len(col_levels_inner_first)
        for depth, (pname, values) in enumerate(col_levels_inner_first):
            block_size = prod(len(v) for _, v in col_levels_inner_first[:depth])
            block_size = max(1, block_size)
            n_outer_repeats = C // (block_size * len(values))
            y_band = panel_bottom - (depth + 0.5) * col_label_band - 0.01
            for rep in range(n_outer_repeats):
                for k, v in enumerate(values):
                    block_start = (rep * len(values) + k) * block_size
                    x_center = panel_left + (block_start + block_size / 2) * cell_w
                    fig.text(
                        x_center,
                        y_band,
                        _format_value(v),
                        ha="center",
                        va="center",
                        fontsize=fs_value,
                        transform=fig.transFigure,
                    )
            # Param name in the left gutter (only for the leftmost panel)
            if j == 0:
                disp = param_display_names.get(pname, pname)
                fig.text(
                    margins["gs_left"] - 0.012,
                    y_band,
                    disp,
                    ha="right",
                    va="center",
                    fontsize=fs_param,
                    fontweight="bold",
                    transform=fig.transFigure,
                )

        # Hierarchical column separators: every non-innermost level draws a
        # vertical stroke at its block boundaries, with linewidth scaling so
        # outer levels read as wider gaps. Innermost gets no extra stroke
        # (the white minor grid already separates individual cells).
        for depth in range(1, n_col_levels):
            block_size = prod(len(v) for _, v in col_levels_inner_first[:depth])
            block_size = max(1, block_size)
            lw = 0.6 + 1.2 * depth
            for k in range(1, C // block_size):
                x = panel_left + k * block_size * cell_w
                _draw_fig_line(fig, x, x, panel_bottom, panel_top, color="white", lw=lw)

        # Row labels (only on the leftmost panel)
        if j == 0:
            row_label_band = 0.022
            row_levels_inner_first = list(reversed(row_levels))
            n_row_levels = len(row_levels_inner_first)
            for depth, (pname, values) in enumerate(row_levels_inner_first):
                block_size = prod(len(v) for _, v in row_levels_inner_first[:depth])
                block_size = max(1, block_size)
                n_outer_repeats = R // (block_size * len(values))
                x_band = panel_left - (depth + 0.5) * row_label_band - 0.005
                # values rotated 90° to read along the y-axis like normal
                # tick labels
                for rep in range(n_outer_repeats):
                    for k, v in enumerate(values):
                        block_start = (rep * len(values) + k) * block_size
                        y_center = panel_top - (block_start + block_size / 2) * cell_h
                        fig.text(
                            x_band,
                            y_center,
                            _format_value(v),
                            ha="center",
                            va="center",
                            fontsize=fs_value,
                            rotation=90,
                            transform=fig.transFigure,
                        )
                # Param name above the row-label band, rotated 90° (matches
                # the rotated value labels below it). Each level's name sits
                # directly above its column of values.
                disp = param_display_names.get(pname, pname)
                fig.text(
                    x_band,
                    panel_top + 0.005,
                    disp,
                    ha="center",
                    va="bottom",
                    fontsize=fs_param,
                    fontweight="bold",
                    rotation=90,
                    transform=fig.transFigure,
                )

            # Hierarchical row separators (white strokes at each block
            # boundary, wider for outer levels). Span all panels horizontally.
            last_panel_right = axes[0, -1].get_position().x1
            for depth in range(1, n_row_levels):
                block_size = prod(len(v) for _, v in row_levels_inner_first[:depth])
                block_size = max(1, block_size)
                lw = 0.6 + 1.2 * depth
                for k in range(1, R // block_size):
                    y = panel_top - k * block_size * cell_h
                    _draw_fig_line(fig, panel_left, last_panel_right, y, y, color="white", lw=lw)


def _draw_fig_line(fig: Figure, x0: float, x1: float, y0: float, y1: float, color: str, lw: float) -> None:
    """Draw a figure-level line in figure-fraction coordinates."""
    from matplotlib.lines import Line2D

    line = Line2D([x0, x1], [y0, y1], color=color, linewidth=lw, transform=fig.transFigure)
    line.set_clip_on(False)
    fig.add_artist(line)


def plot_2d_response_map(
    campaign: Campaign,
    row_params: dict[str, list],
    col_params: dict[str, list],
    *,
    fixed_params: dict[str, float | str] | None = None,
    mode: str = "passfail",
    value: str = "mean",
    target_names: list[str] | None = None,
    target_display_names: dict[str, str] | None = None,
    param_display_names: dict[str, str] | None = None,
    PI_range: float | None = None,
    confidence_level: str | None = None,
    include_joint: bool = True,
    cmap=None,
    fail_color: str | None = None,
    pass_color: str | None = None,
    fail_alpha: float = 1.0,
    pass_alpha: float = 1.0,
    confidence_cmap: list[str] | None = None,
    fig_width: float = 5.0,
) -> tuple[Figure, np.ndarray]:
    """
    Plot a hierarchical 2D response map for a fitted campaign.

    Each cell is one GP prediction at a single point in parameter
    space. Row and column hierarchies (each potentially several nested levels
    deep) are constructed from the Cartesian product of ``row_params`` and
    ``col_params``. Every parameter in the campaign must appear in exactly one
    of ``row_params``, ``col_params``, or ``fixed_params`` (parameters held
    constant don't appear as labels). Works for both characterization and
    plain optimization campaigns.

    Args:
        campaign: Fitted campaign (any task).
        row_params: ``{name: [v1, v2, ...]}``. **First key is innermost** (closest
            to the cells); last key is outermost.
        col_params: ``{name: [v1, v2, ...]}``. Same convention as ``row_params``.
        fixed_params: Parameters held constant. Numeric or categorical values OK.
            Required for any campaign parameter not in ``row_params`` / ``col_params``.
        mode: One of:

            * ``"passfail"`` – binary pass/fail per target (and optional joint
              column). Uses ``confidence_level`` / ``PI_range`` to control how
              strict the pass/fail decision is. Requires thresholds.
            * ``"confidence"`` – 4-level confidence per target (and optional
              joint = elementwise min). Levels: 0 fail, 1 uncertain pass,
              2 likely pass, 3 confident pass. Requires thresholds.
            * ``"continuous"`` – continuous colormap of the mean prediction
              (or std, if ``value="std"``). Requires exactly one target;
              thresholds are not used.
        value: Continuous-mode quantity to plot. ``"mean"`` (default) plots
            the predictive mean; ``"std"`` plots the predictive standard
            deviation. Ignored for non-continuous modes.
        target_names: Subset of campaign targets to plot. Default: all relevant.
        target_display_names: Pretty names for panel titles.
        param_display_names: Pretty names for parameters in the row/col gutters.
        PI_range: Prediction-interval coverage (passfail mode only; 0.7 or 0.95).
        confidence_level: Passfail-mode override for PI: ``None``, ``"mean"``,
            ``"70%"``, or ``"95%"``.
        include_joint: Add a joint panel (passfail/confidence multi-target).
        cmap: Colormap for ``mode="continuous"``. Default ``obsidian_viridis``.
        fail_color, pass_color: Passfail-mode colors. Defaults: Obsidian
            branding magenta (fail) / teal (pass).
        fail_alpha, pass_alpha: Passfail-mode alphas.
        confidence_cmap: 4 hex colors for confidence mode.
        fig_width: Figure width per panel band, in inches.

    Returns:
        ``(fig, axes)``. ``axes`` has shape ``(1, n_panels)``.
    """
    if value not in ("mean", "std"):
        raise ValueError(f"value must be 'mean' or 'std', got {value!r}")
    if value == "std" and mode != "continuous":
        raise ValueError("value='std' is only supported for mode='continuous'")
    cfg = _validate_response_map_params(
        campaign,
        row_params,
        col_params,
        fixed_params,
        target_names,
        param_display_names,
        target_display_names,
        mode,
        include_joint,
    )
    
    # Normalize these variables to avoid linter complaints about None
    target_names = cfg["target_names"] or []
    fixed_params = cfg["fixed_params"]
    param_display_names = cfg["param_display_names"] or {}
    target_display_names = cfg["target_display_names"] or {}
    all_param_names = cfg["all_param_names"]
    include_joint = cfg["include_joint"]

    layout = _compute_response_map_layout(row_params, col_params)
    R, C = layout["R"], layout["C"]

    df = _build_response_map_eval_df(layout, fixed_params, all_param_names)

    panel_grids: list[tuple[str, np.ndarray]] = []  # (display_label, R×C array)
    legend_text: str | None = None

    if mode == "passfail":
        eff_PI, mode_desc = _resolve_passfail_PI_range(PI_range, confidence_level)
        legend_text = mode_desc
        from obsidian.campaign.characterization import CharacterizationEvaluator

        evaluator = CharacterizationEvaluator(campaign)
        results = evaluator.classify_points(df, PI_range=eff_PI, target_names=target_names)
        for tname in target_names:
            grid = results[tname]["pred_mask"].reshape(R, C).astype(float)
            panel_grids.append((target_display_names.get(tname, tname), grid))
        if include_joint:
            joint = results["Joint"]["pred_mask"].reshape(R, C).astype(float)
            panel_grids.append(("Joint", joint))

    elif mode == "confidence":
        from obsidian.campaign.characterization import CharacterizationEvaluator

        evaluator = CharacterizationEvaluator(campaign)
        results = evaluator.classify_confidence_levels(df, target_names=target_names)
        for tname in target_names:
            grid = results[tname].reshape(R, C).astype(float)
            panel_grids.append((target_display_names.get(tname, tname), grid))
        if include_joint:
            joint = results["Joint"].reshape(R, C).astype(float)
            panel_grids.append(("Joint", joint))

    else:  # continuous
        tname = target_names[0]
        if value == "std":
            arr = _posterior_std(campaign, df, tname)
            label_suffix = " (std)"
        else:
            eff_PI = 0.7 if PI_range is None else PI_range
            if eff_PI not in (0.7, 0.95):
                raise ValueError(f"PI_range must be 0.7 or 0.95 for continuous mode, got {eff_PI}")
            pred_df = campaign.optimizer.predict(df, return_f_inv=True, PI_range=eff_PI)
            arr = pred_df[f"{tname} (pred)"].values.astype(float) # type: ignore
            label_suffix = ""
        grid = arr.reshape(R, C)
        panel_grids.append((target_display_names.get(tname, tname) + label_suffix, grid))

    n_panels = len(panel_grids)

    plot_state = _setup_response_map_axes(
        n_panels,
        R,
        C,
        fig_width=fig_width * n_panels,
        has_top_colorbar=(mode == "continuous"),
    )
    fig = plot_state["fig"]
    axes = plot_state["axes"]
    margins = {k: plot_state[k] for k in ("gs_left", "gs_right", "gs_top", "gs_bottom")}

    # Scale label fonts with the per-panel width so labels stay legible on
    # small figures and don't bloat on large ones. 7" per panel is the default;
    # calibrate the base font to 8pt at that size, clamp to [6, 16].
    fig_w, _ = fig.get_size_inches()
    per_panel_w = fig_w / max(n_panels, 1)
    base_fontsize = float(np.clip(per_panel_w * (8.0 / 7.0), 6.0, 16.0))

    # Anchor the legend just under the col-label stack instead of at the
    # bottom of the figure (which leaves a large empty band when there are
    # several col levels). _add_hierarchical_labels uses col_label_band=0.022
    # and reserves an extra ~0.01 below the band for the bottommost label.
    n_col_levels = max(len(col_params), 1)
    col_label_band = 0.022
    legend_y = max(margins["gs_bottom"] - n_col_levels * col_label_band - 0.045, 0.005)

    # ---- Render
    if mode == "passfail":
        if fail_color is None:
            fail_color = _PASSFAIL_FAIL_COLOR
        if pass_color is None:
            pass_color = _PASSFAIL_PASS_COLOR
        cmap_pf = ListedColormap(
            [
                (*mcolors.to_rgb(fail_color), fail_alpha),
                (*mcolors.to_rgb(pass_color), pass_alpha),
            ]
        )
        for j, (_, grid) in enumerate(panel_grids):
            axes[0, j].imshow(
                grid,
                origin="upper",
                aspect="equal",
                interpolation="nearest",
                cmap=cmap_pf,
                vmin=0,
                vmax=1,
            )
            _style_response_map_panel(axes[0, j], R, C)

        from matplotlib.patches import Patch

        suffix = f" ({legend_text})" if legend_text else ""
        legend_patches = [
            Patch(facecolor=(*mcolors.to_rgb(fail_color), fail_alpha), edgecolor="gray", label=f"Fail{suffix}"),
            Patch(facecolor=(*mcolors.to_rgb(pass_color), pass_alpha), edgecolor="gray", label=f"Pass{suffix}"),
        ]
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=2,
            frameon=False,
            fontsize=base_fontsize + 1.0,
            bbox_to_anchor=(0.5, legend_y),
        )

    elif mode == "confidence":
        colors = list(confidence_cmap or _CONFIDENCE_COLORS)
        if len(colors) != 4:
            raise ValueError(f"confidence_cmap must have 4 colors, got {len(colors)}")
        cmap_conf = ListedColormap([(*mcolors.to_rgb(c), a) for c, a in zip(colors, _CONFIDENCE_DEFAULT_ALPHAS)])
        for j, (_, grid) in enumerate(panel_grids):
            axes[0, j].imshow(
                grid,
                origin="upper",
                aspect="equal",
                interpolation="nearest",
                cmap=cmap_conf,
                vmin=-0.5,
                vmax=3.5,
            )
            _style_response_map_panel(axes[0, j], R, C)

        from matplotlib.patches import Patch

        legend_patches = [
            Patch(facecolor=(*mcolors.to_rgb(c), a), edgecolor="gray", label=lbl)
            for c, a, lbl in zip(colors, _CONFIDENCE_DEFAULT_ALPHAS, _CONFIDENCE_LABELS)
        ]
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=4,
            frameon=False,
            fontsize=base_fontsize + 1.0,
            bbox_to_anchor=(0.5, legend_y),
        )

    else:  # continuous
        if cmap is None:
            cmap = obsidian_colors.cm.obsidian_viridis
        for j, (_, grid) in enumerate(panel_grids):
            im = axes[0, j].imshow(
                grid,
                origin="upper",
                aspect="equal",
                interpolation="nearest",
                cmap=cmap,
            )
            _style_response_map_panel(axes[0, j], R, C)
            # Per-panel colorbar above the panel
            bbox = axes[0, j].get_position()
            cbar_ax = fig.add_axes([bbox.x0, bbox.y1 + 0.085, bbox.width, 0.018])
            cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
            cbar.ax.tick_params(labelsize=base_fontsize)
            cbar.ax.xaxis.set_ticks_position("top")
            cbar.ax.xaxis.set_label_position("top")
            cbar.set_label(panel_grids[j][0], fontsize=base_fontsize + 2.0, fontweight="bold", labelpad=6)

    panel_labels = [label for label, _ in panel_grids]
    # In continuous mode the panel title is already on the colorbar, so don't double-print.
    if mode == "continuous":
        panel_labels = [""] * n_panels
    _add_hierarchical_labels(
        fig,
        axes,
        layout,
        panel_labels,
        param_display_names,
        margins,
        base_fontsize=base_fontsize,
    )

    return fig, axes
