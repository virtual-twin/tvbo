"""Plot bifurcation diagrams from continuation results.

Provides helpers to draw equilibrium branches (coloured by stability with special bifurcation points marked) and periodic-orbit envelopes onto a
Matplotlib axis, deriving the plotted variable of interest from continuation
DataFrames.
"""

import matplotlib.pyplot as plt
from sympy import parse_expr, pycode, symbols
from tvbo.classes import equation as equations


def compute_voi(df, VOI, prefix="", state_var_index=None):
    """Compute variable of interest from DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing bifurcation data
    VOI : str
        Variable of interest expression
    prefix : str
        Prefix for variable names
    state_var_index : dict, optional
        Mapping from state variable names to indices in 'x' column

    Returns
    -------
    pd.Series
        Computed VOI values
    """
    # First, check if VOI exists directly as a column
    if VOI in df.columns:
        return df[VOI]

    # Parse the VOI expression
    exp = parse_expr(VOI, equations._clash1)
    variables = list(exp.free_symbols)

    # For single variable that doesn't exist as column, check if it's in 'x'
    if len(variables) == 1 and "x" in df.columns and state_var_index is not None:
        var_name = str(variables[0])
        if var_name in state_var_index:
            # Extract the specific state variable from 'x' column
            idx = state_var_index[var_name]
            return df["x"].apply(lambda x_val: x_val[idx] if hasattr(x_val, "__getitem__") else x_val)

    # Otherwise, try standard evaluation with prefix substitution
    exp = exp.subs({v: symbols(f"{prefix}{v}") for v in variables})
    return df.eval(pycode(exp, fully_qualified_modules=False))


def plot_equilibrium_branch(df, ax, ICS=None, VOI=None, **kwargs):
    """Plot an equilibrium branch coloured by stability with special points marked.

    The branch is split into contiguous segments wherever the `stable` flag changes, drawing stable segments as solid lines and unstable ones as dashed, then overlaying scatter markers for each special bifurcation point (excluding `endpoint`) coloured after the Julia BifurcationKit palette.

    Args:
        df: Continuation DataFrame with `param`, `stable`, and `specialpoint`
            columns (plus the variable(s) referenced by `VOI`).
        ax: Matplotlib axis to draw on.
        ICS: Unused; accepted for interface compatibility.
        VOI: Variable-of-interest expression to plot on the y-axis; defaults to
            the first column of `df` when `None`.
        **kwargs: Ignored; accepted for interface compatibility.
    """
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    stable_color = color_cycle[0]  # First color for stable
    unstable_color = color_cycle[0]  # Second color for unstable
    df.specialpoint.dropna().unique()
    # Julia BifurcationKit color mapping replicated (subset)
    julia_colorbif = {
        "fold": "black",
        "hopf": "red",
        "bp": "blue",
        "nd": "magenta",
        "none": "yellow",
        "ns": "orange",
        "pd": "green",
        "bt": "red",
        "cusp": "sienna",
        "gh": "brown",
        "zh": "burlywood",
        "hh": "green",
        "R": "chartreuse",
        "R1": "chartreuse",
        "R2": "chartreuse",
        "R3": "chartreuse",
        "R4": "blue",
        "foldFlip": "navy",
        "ch": "darkred",
        "foldNS": "cyan",
        "flipNS": "goldenrod",
        "pdNS": "maroon",
        "nsns": "purple",
        "gpd": "salmon",
        "user": "goldenrod",
    }
    colormap = julia_colorbif

    if VOI is None:
        VOI = df.columns[0]
    # Create a new column for segment labeling based on stability
    df["segment"] = (df.stable != df.stable.shift()).cumsum()

    # Iterate over the unique segments and plot each with its corresponding style and label
    for segment_id, segment_data in df.groupby("segment"):
        # Determine the stability of the segment
        is_stable = segment_data.iloc[0].stable
        label = "Stable" if is_stable else "Unstable"

        # Add label only if it hasn't been added before
        current_labels = ax.get_legend_handles_labels()[1]
        ax.plot(
            segment_data["param"],
            compute_voi(segment_data, VOI),
            "-" if is_stable else "--",
            zorder=1,
            label=label if label not in current_labels else None,
            color=stable_color if is_stable else unstable_color,
        )

    # Scatter special points, excluding 'endpoint'
    for i, r in df[df["specialpoint"].notna()].iterrows():
        if r.specialpoint != "endpoint":
            # Add label only if it hasn't been added before
            current_labels = ax.get_legend_handles_labels()[1]
            color = colormap.get(r.specialpoint, "black")
            ax.scatter(
                r.param,
                compute_voi(df, VOI).loc[i],
                zorder=2,
                label=r.specialpoint if r.specialpoint not in current_labels else None,
                color=color,
            )


def plot_periodic_orbit(df_po, VOI, ax, color_cycle_index=1, **kwargs):
    """Plot the min/max envelope of a periodic-orbit branch.

    Draws two lines in a shared colour tracing the minimum and maximum of the variable of interest along the branch (using the `min_`/`max_` column prefixes), labelling only the first so the legend gains a single `Periodic orbit` entry.

    Args:
        df_po: Periodic-orbit DataFrame with a `param` column and `min_`/`max_`
            prefixed columns for the variable of interest.
        VOI: Variable-of-interest name; when `None`, inferred from the first
            column of `df_po` with any `min_`/`max_` prefix stripped.
        ax: Matplotlib axis to draw on.
        color_cycle_index: Index into the current axis colour cycle used for
            both envelope lines.
        **kwargs: Ignored; accepted for interface compatibility.
    """
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    periodic_color = color_cycle[color_cycle_index]  # Third color for periodic orbit

    if VOI is None:
        VOI = df_po.columns[0].replace("min_", "").replace("max_", "")

    # Add periodic orbit label only once
    current_labels = ax.get_legend_handles_labels()[1]
    label = "Periodic orbit" if "Periodic orbit" not in current_labels else None

    ax.plot(
        df_po["param"],
        compute_voi(df_po, VOI, prefix="min_"),
        # df_po[f"min_{VOI}"],
        zorder=1,
        label=label,
        color=periodic_color,
    )
    ax.plot(
        df_po["param"],
        compute_voi(df_po, VOI, prefix="max_"),
        # df_po[f"max_{VOI}"],
        zorder=1,
        color=periodic_color,
    )
