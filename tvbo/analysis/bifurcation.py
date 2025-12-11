"""Bifurcation analysis result objects and helpers.

Contains the BifurcationResult class whose instances are returned by
`model.run(format="bifurcation-julia", ...)`.

Key attributes
--------------
df : pandas.DataFrame
    Continuation branch points with columns (x, param, itnewton, itlinear, ds, n_unstable, n_imag, stable, step, specialpoint, ...).
hopf_indices / bp_indices : list[int]
    Row indices in `df` where Hopf / Branch (bp) special points occur.
hopf_steps / bp_steps : list[int]
    Corresponding continuation step values.
periodic_orbits : list[BifurcationResult | Any]
    If periodic orbits were computed in Julia (`po_results`), each periodic orbit branch is wrapped as a child
    BifurcationResult when possible; otherwise the raw Julia object is stored.
"""

from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
from sympy import parse_expr, pycode, symbols

from tvbo.knowledge.simulation import equations  # for VOI parsing consistency


def continuation_kind(obj):
    from julia import Base, Main

    T = Main.typeof(obj)
    params = getattr(T, "parameters", None)
    return Base.first(params).name.name


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
    if len(variables) == 1 and 'x' in df.columns and state_var_index is not None:
        var_name = str(variables[0])
        if var_name in state_var_index:
            # Extract the specific state variable from 'x' column
            idx = state_var_index[var_name]
            return df['x'].apply(lambda x_val: x_val[idx] if hasattr(x_val, '__getitem__') else x_val)

    # Otherwise, try standard evaluation with prefix substitution
    exp = exp.subs({v: symbols(f"{prefix}{v}") for v in variables})
    return df.eval(pycode(exp, fully_qualified_modules=False))


class BifurcationResult:
    def __init__(self, br, **kwargs):
        self.br = br
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Extract state variable names and create index mapping
        self.state_var_index = {}
        if hasattr(self, 'ICS') and hasattr(self, 'model'):
            # Get state variables from model if available
            if hasattr(self.model, 'state_variables'):
                self.state_var_index = {name: idx for idx, name in enumerate(self.model.state_variables.keys())}

        # Allow explicit state_var_index to be passed
        if 'state_var_index' in kwargs:
            self.state_var_index = kwargs['state_var_index']

        if continuation_kind(br) == "EquilibriumCont":
            self.df = pd.DataFrame(
                self.br.branch,
                columns=[
                    "x",
                    "param",
                    "itnewton",
                    "itlinear",
                    "ds",
                    "n_unstable",
                    "n_imag",
                    "stable",
                    "step",
                ],
            )
            sp = getattr(self.br, "specialpoint", None)

        elif continuation_kind(br) == "PeriodicOrbitCont":
            # Get the actual data shape first to determine columns dynamically
            branch_data = br.γ.branch

            # Convert to numpy array to get actual dimensions
            try:
                arr = np.array(branch_data)
                if arr.ndim == 1:
                    # Single row case
                    n_cols = len(arr)
                    n_rows = 1
                else:
                    n_rows, n_cols = arr.shape
            except Exception as e:
                # Fallback: try to get length from first element
                try:
                    n_rows = len(branch_data)
                    n_cols = len(branch_data[0]) if n_rows > 0 else 13
                except:
                    n_rows = 0
                    n_cols = 13

            # Build column names based on actual data size
            # Common columns that should appear in order
            base_cols = ["max_V", "min_V", "max_W", "min_W", "period", "param",
                        "itnewton", "itlinear", "ds", "n_unstable", "n_imag", "stable", "step"]

            if n_cols <= len(base_cols):
                expected_cols = base_cols[:n_cols]
            else:
                # More columns than expected - add generic names
                expected_cols = base_cols + [f"col_{i}" for i in range(len(base_cols), n_cols)]

            # Create DataFrame without specifying columns if there's a size mismatch risk
            try:
                self.df = pd.DataFrame(branch_data, columns=expected_cols)
            except (ValueError, AssertionError) as e:
                # Fallback: let pandas infer columns
                print(f"Warning: Column mismatch ({len(expected_cols)} expected, {n_cols} actual). Using auto-generated column names.")
                self.df = pd.DataFrame(branch_data)
                # Rename columns to match expected names where possible
                for i, col in enumerate(expected_cols[:len(self.df.columns)]):
                    self.df.rename(columns={self.df.columns[i]: col}, inplace=True)

            sp = getattr(self.br.γ, "specialpoint", None)

        # Annotate special points (fold, hopf, bp, endpoint, etc.) if available

        if sp is not None:
            if not isinstance(sp, (list, tuple)):
                sp = [sp]
            if "specialpoint" not in self.df.columns:
                self.df["specialpoint"] = None
            if "sp_norm" not in self.df.columns:
                self.df["sp_norm"] = np.nan
            if "sp_idx" not in self.df.columns:
                self.df["sp_idx"] = np.nan
            for point in sp:
                step = int(getattr(point, "step", getattr(point, "idx", -1)))
                typ = str(getattr(point, "type", ""))
                norm = float(getattr(point, "norm", np.nan))
                idx_val = int(getattr(point, "idx", step))
                if "step" in self.df.columns and step in self.df.step.values:
                    rows = self.df.index[self.df.step == step].tolist()
                else:
                    pval = float(getattr(point, "param", np.nan))
                    if np.isfinite(pval):
                        rows = [int(np.abs(self.df.param - pval).argmin())]
                    else:
                        rows = []
                for rix in rows:
                    existing = self.df.at[rix, "specialpoint"]
                    if existing is None or existing == "":
                        self.df.at[rix, "specialpoint"] = typ
                    elif typ not in str(existing).split(","):
                        self.df.at[rix, "specialpoint"] = f"{existing},{typ}"
                    self.df.at[rix, "sp_norm"] = norm
                    self.df.at[rix, "sp_idx"] = idx_val
        # Store hopf and bp indices (row indices in the DataFrame) and corresponding step values
        # A row might contain multiple specialpoint labels separated by commas.
        if "specialpoint" in self.df.columns:
            sp_series = self.df["specialpoint"].astype(str)
            hopf_mask = sp_series.str.contains("hopf", case=False, na=False)
            bp_mask = sp_series.str.contains("bp", case=False, na=False)
            self.hopf_indices = self.df.index[hopf_mask].tolist()
            self.bp_indices = self.df.index[bp_mask].tolist()
            self.hopf_steps = self.df.loc[hopf_mask, "step"].tolist()
            self.bp_steps = self.df.loc[bp_mask, "step"].tolist()

    def plot_special_points(self, VOI, ax=None, **kwargs):
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

        # Scatter special points, excluding 'endpoint'
        for i, r in self.df[self.df["specialpoint"].notna()].iterrows():
            if r.specialpoint != "endpoint":
                # Add label only if it hasn't been added before
                current_labels = ax.get_legend_handles_labels()[1]
                color = julia_colorbif.get(r.specialpoint, "black")
                ax.scatter(
                    r.param,
                    compute_voi(self.df, VOI, state_var_index=self.state_var_index).loc[i],
                    zorder=2,
                    label=(
                        r.specialpoint if r.specialpoint not in current_labels else None
                    ),
                    color=color,
                )

    def plot_branch(self, ax, ICS=None, VOI=None, **kwargs):
        if VOI is None:
            VOI = self.df.columns[0]
        # Create a new column for segment labeling based on stability
        self.df["segment"] = (self.df.stable != self.df.stable.shift()).cumsum()

        for segment_id, segment_data in self.df.groupby("segment"):
            # Determine the stability of the segment
            is_stable = segment_data.iloc[0].stable
            label = "Stable" if is_stable else "Unstable"

            # Add label only if it hasn't been added before
            current_labels = ax.get_legend_handles_labels()[1]
            ax.plot(
                segment_data["param"],
                compute_voi(segment_data, VOI, state_var_index=self.state_var_index),
                "-" if is_stable else "--",
                zorder=1,
                label=label if label not in current_labels else None,
                **kwargs,
            )

    def plot_equilibrium_branch(self, ax, ICS=None, VOI=None, **kwargs):
        self.plot_branch(ax, ICS=ICS, VOI=VOI, **kwargs)
        self.plot_special_points(VOI="x", ax=ax, **kwargs)

    def plot(self, ax=None, ICS=None, VOI=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        self.plot_branch(ax, ICS=ICS, VOI=VOI, **kwargs)
        self.plot_special_points(VOI=VOI, ax=ax, **kwargs)
        return ax


__all__ = ["BifurcationResult"]
