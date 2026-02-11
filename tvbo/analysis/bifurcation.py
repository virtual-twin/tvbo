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
from sympy import parse_expr, pycode, symbols

from tvbo.knowledge.simulation import equations  # for VOI parsing consistency


def continuation_kind(obj):
    """Determine the continuation kind from a BifurcationKit result object.

    Works with juliacall (PythonCall.jl) by inspecting the type string.
    """
    from juliacall import Main
    # Store the object in Julia's Main and inspect its type
    Main._bif_kind_obj = obj
    type_str = str(Main.seval("string(typeof(_bif_kind_obj))"))

    if "EquilibriumCont" in type_str:
        return "EquilibriumCont"
    elif "PeriodicOrbitCont" in type_str:
        return "PeriodicOrbitCont"
    else:
        return type_str


def _extract_equilibrium_df(br):
    """Extract equilibrium continuation branch as a pandas DataFrame.

    Converts Julia StructArrays to Python via JSON-like column extraction,
    which is robust across juliacall versions.
    """
    from juliacall import Main

    Main._br_extract = br
    # Use Julia to extract each column as a plain array
    n = int(Main.seval("length(_br_extract.branch)"))
    if n == 0:
        return pd.DataFrame()

    # Get column names from the Julia StructArray
    col_names_jl = Main.seval("string.(fieldnames(eltype(_br_extract.branch)))")
    col_names = [str(c) for c in col_names_jl]

    data = {}
    for col in col_names:
        try:
            vals = Main.seval(f"collect(getproperty(_br_extract.branch, Symbol(\"{col}\")))")
            arr = np.array(vals)
            if arr.dtype == object:
                # Vector-valued column (e.g. legacy 'x' without record_from_solution)
                data[col] = [np.array(v) for v in vals]
            else:
                data[col] = arr
        except Exception:
            pass

    return pd.DataFrame(data)


def _extract_special_points(br):
    """Extract special points from a BifurcationKit branch result."""
    from juliacall import Main

    Main._br_sp = br
    n_sp = int(Main.seval("length(_br_sp.specialpoint)"))
    if n_sp == 0:
        return []

    points = []
    for i in range(1, n_sp + 1):
        sp = Main.seval(f"_br_sp.specialpoint[{i}]")
        point = {
            'type': str(Main.seval(f"string(_br_sp.specialpoint[{i}].type)")),
            'step': int(Main.seval(f"_br_sp.specialpoint[{i}].step")),
            'param': float(Main.seval(f"_br_sp.specialpoint[{i}].param")),
            'idx': int(Main.seval(f"_br_sp.specialpoint[{i}].idx")),
        }
        try:
            point['norm'] = float(Main.seval(f"_br_sp.specialpoint[{i}].norm"))
        except Exception:
            point['norm'] = np.nan
        points.append(point)

    return points


def compute_voi(df, VOI, prefix="", state_var_index=None):
    """Compute variable of interest from DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing bifurcation data
    VOI : str
        Variable of interest expression (column name or sympy expression)
    prefix : str
        Prefix for variable names
    state_var_index : dict, optional
        Mapping from state variable names to indices in 'x' column (legacy)

    Returns
    -------
    pd.Series
        Computed VOI values
    """
    if VOI is None:
        raise ValueError("VOI must not be None; use _resolve_voi() first")

    # Direct column access (common case with record_from_solution)
    if VOI in df.columns:
        return df[VOI]

    # Parse the VOI expression
    exp = parse_expr(VOI, equations._clash1)
    variables = list(exp.free_symbols)

    # Legacy: single variable in a vector 'x' column
    if len(variables) == 1 and 'x' in df.columns and state_var_index is not None:
        var_name = str(variables[0])
        if var_name in state_var_index:
            idx = state_var_index[var_name]
            return df['x'].apply(lambda x_val: x_val[idx] if hasattr(x_val, '__getitem__') else x_val)

    # Symbolic expression evaluation
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

        sp_list = None  # list of dicts from _extract_special_points

        kind = continuation_kind(br)

        if kind == "EquilibriumCont":
            self.df = _extract_equilibrium_df(br)
            sp_list = _extract_special_points(br)

        elif kind == "PeriodicOrbitCont":
            # PO ContResult has the same .branch StructArray as equilibrium
            self.df = _extract_equilibrium_df(br)
            sp_list = _extract_special_points(br)

        # Annotate special points (fold, hopf, bp, endpoint, etc.)
        if sp_list:
            if "specialpoint" not in self.df.columns:
                self.df["specialpoint"] = None
            if "sp_norm" not in self.df.columns:
                self.df["sp_norm"] = np.nan
            if "sp_idx" not in self.df.columns:
                self.df["sp_idx"] = np.nan
            for point in sp_list:
                step = point.get("step", point.get("idx", -1))
                typ = point.get("type", "")
                norm = point.get("norm", np.nan)
                idx_val = point.get("idx", step)
                if "step" in self.df.columns and step in self.df.step.values:
                    rows = self.df.index[self.df.step == step].tolist()
                else:
                    pval = point.get("param", np.nan)
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
        self.hopf_indices = []
        self.bp_indices = []
        self.hopf_steps = []
        self.bp_steps = []
        if "specialpoint" in self.df.columns:
            sp_series = self.df["specialpoint"].astype(str)
            hopf_mask = sp_series.str.contains("hopf", case=False, na=False)
            bp_mask = sp_series.str.contains("bp", case=False, na=False)
            self.hopf_indices = self.df.index[hopf_mask].tolist()
            self.bp_indices = self.df.index[bp_mask].tolist()
            if "step" in self.df.columns:
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
                    s=80,
                    label=(
                        r.specialpoint if r.specialpoint not in current_labels else None
                    ),
                    color=color,
                )

    def plot_branch(self, ax, ICS=None, VOI=None, **kwargs):
        VOI = self._resolve_voi(VOI)
        # BifurcationKit convention: stable = thick solid, unstable = thin solid
        lw = kwargs.pop("linewidth", kwargs.pop("lw", None))
        # Filter out non-matplotlib kwargs that may leak through
        _plot_ignore = {"periodic_orbits", "verbose", "max_steps", "ds",
                        "dsmin", "dsmax", "p_min", "p_max", "quiet",
                        "detect_bifurcation", "nev", "n_inversion",
                        "max_bisection_steps", "tol_stability", "bothside",
                        "bifurcation_points", "n_runs", "model",
                        "state_var_index", "ICS"}
        plot_kwargs = {k: v for k, v in kwargs.items() if k not in _plot_ignore}
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
                linewidth=lw if lw is not None else (2.0 if is_stable else 1.0),
                zorder=1,
                label=label if label not in current_labels else None,
                **plot_kwargs,
            )

    def plot_equilibrium_branch(self, ax, ICS=None, VOI=None, **kwargs):
        VOI = self._resolve_voi(VOI)
        self.plot_branch(ax, ICS=ICS, VOI=VOI, **kwargs)
        self.plot_special_points(VOI=VOI, ax=ax, **kwargs)

    def _resolve_voi(self, VOI):
        """Resolve VOI to a concrete column name / expression string."""
        if VOI is not None:
            return VOI
        # Default: first state variable column (user-defined via record_from_solution)
        if self.state_var_index:
            return next(iter(self.state_var_index))
        # Fallback: first non-metadata column
        meta = {"param", "itnewton", "itlinear", "ds", "n_unstable",
                "n_imag", "stable", "step", "specialpoint", "sp_norm",
                "sp_idx", "segment"}
        for c in self.df.columns:
            if c not in meta:
                return c
        return self.df.columns[0]

    def plot(self, ax=None, ICS=None, VOI=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        VOI = self._resolve_voi(VOI)
        self.plot_branch(ax, ICS=ICS, VOI=VOI, **kwargs)
        self.plot_special_points(VOI=VOI, ax=ax, **kwargs)

        # Plot periodic orbit envelopes (max/min) if available
        po_list = getattr(self, 'periodic_orbits', None)
        if isinstance(po_list, list) and po_list:
            for po_br in po_list:
                # PO record_from_solution produces max_<sv>, min_<sv> columns
                max_col = f"max_{VOI}"
                min_col = f"min_{VOI}"
                if max_col in po_br.df.columns:
                    po_br.plot_branch(ax, ICS=ICS, VOI=max_col, **kwargs)
                    po_br.plot_branch(ax, ICS=ICS, VOI=min_col, **kwargs)
                    po_br.plot_special_points(VOI=max_col, ax=ax, **kwargs)
                else:
                    # Fallback: plot first non-metadata column
                    po_voi = po_br._resolve_voi(None)
                    po_br.plot_branch(ax, ICS=ICS, VOI=po_voi, **kwargs)
                    po_br.plot_special_points(VOI=po_voi, ax=ax, **kwargs)

        # Axis labels
        ics_label = ICS if ICS else getattr(self, 'ICS', 'param')
        ax.set_xlabel(ics_label)
        ax.set_ylabel(VOI)
        ax.legend()
        return ax


__all__ = ["BifurcationResult"]
