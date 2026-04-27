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

from tvbo.classes import equation as equations  # for VOI parsing consistency


# ── Publication-quality color palette (tvbo viridis cycle) ──────────
# Semantic mapping for bifurcation diagram elements
_C = {
    "stable": "#000000",  # black — stable equilibrium
    "unstable": "#888888",  # medium grey — unstable equilibrium
    "hopf": "#440154",  # dark violet — Hopf bifurcation points
    "fold": "#414487",  # indigo — fold/saddle-node points
    "po_surface": "#22a884",  # teal-green — periodic orbit surface
    "po_line": "#2a788e",  # darker teal — orbit wireframes
    "c2_hopf": "#440154",  # violet — Hopf continuation curve
    "c2_fold": "#414487",  # indigo — fold continuation curve
    "bt": "#2a788e",  # teal — Bogdanov-Takens
    "gh": "#7ad151",  # lime — Generalized Hopf
    "cusp": "#fde725",  # yellow — Cusp
    "zh": "#22a884",  # teal-green — Zero-Hopf
    "bp": "#414487",  # indigo — branch point
    "ns": "#fde725",  # yellow — Neimark-Sacker
    "pd": "#22a884",  # teal-green — period doubling
    "po_env": "#22a884",  # teal-green — PO envelope fill
}


def _apply_style():
    """Activate tvbo publication style if bsplot is available."""
    try:
        from bsplot import style

        style.use("tvbo")
    except ImportError:
        pass


def _format_fig(fig):
    """Apply bsplot figure formatting if available."""
    try:
        from bsplot import style

        # Only format 2D axes (format_fig doesn't handle Axes3D)
        for ax in fig.axes:
            if not hasattr(ax, "set_zlabel"):
                style.format_ax(ax, add_panel_numbers=False)
    except (ImportError, Exception):
        pass


def _format_3d_axes(ax):
    """Style a 3D axes for publication: clean panes, no grid, thin edges."""
    # White panes, no grid
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")
    pane_color = (0.97, 0.97, 0.97, 0.4)
    ax.xaxis.pane.set_facecolor(pane_color)
    ax.yaxis.pane.set_facecolor(pane_color)
    ax.zaxis.pane.set_facecolor(pane_color)
    ax.grid(False)

    # Thin axis lines
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.line.set_linewidth(0.6)
        axis.line.set_color("#333333")

    # Tick styling
    ax.tick_params(axis="both", which="major", labelsize=7, length=3, width=0.6, pad=2, colors="#333333")

    # Label styling
    for setter in [ax.set_xlabel, ax.set_ylabel, ax.set_zlabel]:
        pass  # labels set by caller; just ensure consistency


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
    elif "HopfCont" in type_str:
        return "HopfCont"
    elif "FoldCont" in type_str:
        return "FoldCont"
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
            vals = Main.seval(f'collect(getproperty(_br_extract.branch, Symbol("{col}")))')
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
        Main.seval(f"_br_sp.specialpoint[{i}]")
        point = {
            "type": str(Main.seval(f"string(_br_sp.specialpoint[{i}].type)")),
            "step": int(Main.seval(f"_br_sp.specialpoint[{i}].step")),
            "param": float(Main.seval(f"_br_sp.specialpoint[{i}].param")),
            "idx": int(Main.seval(f"_br_sp.specialpoint[{i}].idx")),
        }
        try:
            point["norm"] = float(Main.seval(f"_br_sp.specialpoint[{i}].norm"))
        except Exception:
            point["norm"] = np.nan
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
    if len(variables) == 1 and "x" in df.columns and state_var_index is not None:
        var_name = str(variables[0])
        if var_name in state_var_index:
            idx = state_var_index[var_name]
            return df["x"].apply(lambda x_val: x_val[idx] if hasattr(x_val, "__getitem__") else x_val)

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
        if hasattr(self, "ICS") and hasattr(self, "model"):
            # Get state variables from model if available
            if hasattr(self.model, "state_variables"):
                self.state_var_index = {name: idx for idx, name in enumerate(self.model.state_variables.keys())}

        # Allow explicit state_var_index to be passed
        if "state_var_index" in kwargs:
            self.state_var_index = kwargs["state_var_index"]

        sp_list = None  # list of dicts from _extract_special_points

        kind = continuation_kind(br)

        if kind == "EquilibriumCont":
            self.df = _extract_equilibrium_df(br)
            sp_list = _extract_special_points(br)

        elif kind == "PeriodicOrbitCont":
            # PO ContResult has the same .branch StructArray as equilibrium
            self.df = _extract_equilibrium_df(br)
            sp_list = _extract_special_points(br)

        elif kind in ("HopfCont", "FoldCont"):
            # Codim-2 ContResult: same .branch StructArray layout
            self.df = _extract_equilibrium_df(br)
            sp_list = _extract_special_points(br)

        # Fallback: ensure df always exists
        if not hasattr(self, "df"):
            self.df = pd.DataFrame()

        # Initialize codim-2 curves list (populated by adapters after init)
        if not hasattr(self, "codim2_curves"):
            self.codim2_curves = []

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
        _sp_colors = {
            "fold": _C["fold"],
            "hopf": _C["hopf"],
            "bp": _C["bp"],
            "nd": "#888888",
            "none": "#888888",
            "ns": _C["ns"],
            "pd": _C["pd"],
            "bt": _C["bt"],
            "cusp": _C["cusp"],
            "gh": _C["gh"],
            "zh": _C["zh"],
            "hh": _C["gh"],
        }
        _sp_markers = {
            "fold": "s",
            "hopf": "o",
            "bp": "D",
            "nd": "v",
            "ns": "^",
            "pd": "p",
            "bt": "s",
            "cusp": "D",
            "gh": "^",
            "zh": "v",
            "hh": "p",
        }

        for i, r in self.df[self.df["specialpoint"].notna()].iterrows():
            sp = str(r.specialpoint).lower().split(",")[0].strip()
            if sp in ("endpoint", "none", "nan", ""):
                continue
            current_labels = ax.get_legend_handles_labels()[1]
            color = _sp_colors.get(sp, "#333333")
            marker = _sp_markers.get(sp, "o")
            ax.scatter(
                r.param,
                compute_voi(self.df, VOI, state_var_index=self.state_var_index).loc[i],
                zorder=5,
                s=40,
                marker=marker,
                facecolors=color,
                edgecolors="white",
                linewidths=0.8,
                label=(sp.upper() if sp.upper() not in current_labels else None),
            )

    def plot_branch(self, ax, ICS=None, VOI=None, **kwargs):
        VOI = self._resolve_voi(VOI)
        if self.df.empty:
            return
        lw = kwargs.pop("linewidth", kwargs.pop("lw", None))
        _plot_ignore = {
            "periodic_orbits",
            "verbose",
            "max_steps",
            "ds",
            "dsmin",
            "dsmax",
            "p_min",
            "p_max",
            "quiet",
            "detect_bifurcation",
            "nev",
            "n_inversion",
            "max_bisection_steps",
            "tol_stability",
            "bothside",
            "bifurcation_points",
            "n_runs",
            "model",
            "state_var_index",
            "ICS",
        }
        plot_kwargs = {k: v for k, v in kwargs.items() if k not in _plot_ignore}
        if "stable" not in self.df.columns:
            self.df["stable"] = True
        self.df["segment"] = (self.df.stable != self.df.stable.shift()).cumsum()

        for segment_id, segment_data in self.df.groupby("segment"):
            is_stable = segment_data.iloc[0].stable
            label = "Stable" if is_stable else "Unstable"
            current_labels = ax.get_legend_handles_labels()[1]
            ax.plot(
                segment_data["param"],
                compute_voi(segment_data, VOI, state_var_index=self.state_var_index),
                "-" if is_stable else "--",
                color=(_C["stable"] if is_stable else _C["unstable"]),
                linewidth=lw if lw is not None else (1.5 if is_stable else 1.0),
                zorder=2 if is_stable else 1,
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
        meta = {
            "param",
            "itnewton",
            "itlinear",
            "ds",
            "n_unstable",
            "n_imag",
            "stable",
            "step",
            "specialpoint",
            "sp_norm",
            "sp_idx",
            "segment",
        }
        for c in self.df.columns:
            if c not in meta:
                return c
        return self.df.columns[0]

    def extract_orbit_meshes(self, n_samples=40):
        """Extract full periodic orbit solution meshes from a PO branch.

        Works with BifurcationKit.jl ContResult objects that store
        ``.sol`` (vector of orbit solutions at each continuation step).

        Parameters
        ----------
        n_samples : int
            Number of orbits to sample evenly across the branch.

        Returns
        -------
        list[dict]
            Each dict has keys: ``param`` (float), state variable names
            (1D arrays of the orbit trace), and ``t`` (mesh times).
            Returns empty list if orbit data is unavailable.
        """
        if self.br is None:
            return []

        try:
            from juliacall import Main as jl

            sv_names = list(self.state_var_index.keys()) if self.state_var_index else ["V", "W"]

            # Store br in Julia scope
            jl._tvbo_po_br = self.br

            # Check .sol availability
            n_sol = int(jl.seval("length(_tvbo_po_br.sol)"))
            if n_sol == 0:
                return []

            # Compute sample indices
            n_samples = min(n_samples, n_sol)
            indices = np.unique(np.round(np.linspace(1, n_sol, n_samples)).astype(int))

            result = []
            for idx in indices:
                try:
                    jl._tvbo_idx = int(idx)
                    jl.seval("""
                    _tvbo_s = _tvbo_po_br.sol[_tvbo_idx]
                    _tvbo_xtt = get_periodic_orbit(
                        _tvbo_po_br.prob.prob, _tvbo_s.x, _tvbo_s.p
                    )
                    """)
                    d = {
                        "param": float(jl.seval("Float64(_tvbo_s.p)")),
                        "t": np.array(jl.seval("collect(Float64, _tvbo_xtt.t)")),
                    }
                    for i, sv in enumerate(sv_names):
                        d[sv] = np.array(jl.seval(f"collect(Float64, _tvbo_xtt[{i + 1},:])"))
                    result.append(d)
                except Exception:
                    continue  # skip this orbit on failure

            return result

        except Exception:
            return []

    def plot(self, ax=None, ICS=None, VOI=None, save=None, **kwargs):
        _apply_style()
        if ax is None:
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
        else:
            fig = ax.get_figure()
        VOI = self._resolve_voi(VOI)
        self.plot_branch(ax, ICS=ICS, VOI=VOI, **kwargs)
        self.plot_special_points(VOI=VOI, ax=ax, **kwargs)

        # Periodic orbit envelopes as filled region
        po_list = getattr(self, "periodic_orbits", None)
        if isinstance(po_list, list) and po_list:
            for po_br in po_list:
                if po_br.df.empty:
                    continue
                max_col = f"max_{VOI}"
                min_col = f"min_{VOI}"
                if max_col in po_br.df.columns:
                    params = po_br.df["param"].values
                    v_max = po_br.df[max_col].values
                    v_min = po_br.df[min_col].values
                    clabels = ax.get_legend_handles_labels()[1]
                    ax.fill_between(
                        params,
                        v_min,
                        v_max,
                        color=_C["po_env"],
                        alpha=0.15,
                        label=("PO envelope" if "PO envelope" not in clabels else None),
                        zorder=0,
                    )
                    ax.plot(params, v_max, "-", color=_C["po_line"], linewidth=0.8, alpha=0.6)
                    ax.plot(params, v_min, "-", color=_C["po_line"], linewidth=0.8, alpha=0.6)
                    po_br.plot_special_points(VOI=max_col, ax=ax, **kwargs)
                elif VOI in po_br.df.columns:
                    po_br.plot_branch(ax, ICS=ICS, VOI=VOI, **kwargs)
                    po_br.plot_special_points(VOI=VOI, ax=ax, **kwargs)

        ics_label = ICS if ICS else getattr(self, "ICS", "param")
        ax.set_xlabel(ics_label)
        ax.set_ylabel(VOI)
        ax.legend(framealpha=0.9, edgecolor="none", fontsize=7)
        _format_fig(fig)
        if save:
            fig.savefig(save, dpi=500, bbox_inches="tight")
        return ax

    def plot_codim2(self, ax=None, ICS=None, ICS2=None, save=None, **kwargs):
        """Plot codim-2 bifurcation curves in (param1, param2) space.

        Axis convention:
            x = primary free parameter (codim-1, e.g. I)  = param2 in c2 data
            y = secondary parameter (codim-2, e.g. b)     = param in c2 data
        """
        c2_list = getattr(self, "codim2_curves", None)
        if not c2_list:
            return ax

        _apply_style()
        if ax is None:
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
        else:
            fig = ax.get_figure()

        ics_label = ICS if ICS else getattr(self, "ICS", "param")

        _c2_colors = {
            "fold": _C["c2_fold"],
            "hopf": _C["c2_hopf"],
            "bp": _C["bp"],
        }
        _c2_sp_colors = {
            "bt": _C["bt"],
            "cusp": _C["cusp"],
            "gh": _C["gh"],
            "zh": _C["zh"],
            "hh": _C["gh"],
        }
        _c2_sp_markers = {
            "bt": "s",
            "cusp": "D",
            "gh": "^",
            "zh": "v",
            "hh": "p",
        }

        for c2 in c2_list:
            if c2.df.empty:
                continue

            src_type = getattr(c2, "_source_type", "fold")
            color = _c2_colors.get(src_type, "#555555")
            label = f"{src_type.capitalize()} curve"

            param2_col = "param2"
            if param2_col not in c2.df.columns:
                fp2 = getattr(c2, "_fp2_name", None)
                if fp2 and fp2 in c2.df.columns:
                    param2_col = fp2
                else:
                    continue

            current_labels = ax.get_legend_handles_labels()[1]
            ax.plot(
                c2.df[param2_col],
                c2.df["param"],
                "-",
                color=color,
                linewidth=1.5,
                label=label if label not in current_labels else None,
            )

            if "specialpoint" in c2.df.columns:
                for i, r in c2.df[c2.df["specialpoint"].notna()].iterrows():
                    sp_raw = str(r.specialpoint).lower()
                    for sp in sp_raw.split(","):
                        sp = sp.strip()
                        if sp in ("endpoint", "none", "nan", ""):
                            continue
                        sp_color = _c2_sp_colors.get(sp, "#333333")
                        sp_marker = _c2_sp_markers.get(sp, "o")
                        sp_label = sp.upper()
                        current_labels = ax.get_legend_handles_labels()[1]
                        ax.scatter(
                            r[param2_col],
                            r["param"],
                            s=45,
                            zorder=5,
                            marker=sp_marker,
                            facecolors=sp_color,
                            edgecolors="white",
                            linewidths=0.8,
                            label=(sp_label if sp_label not in current_labels else None),
                        )

        c2_ics = None
        if c2_list:
            c2_ics = getattr(c2_list[0], "_ics_name", None)
        ics2_label = ICS2 or c2_ics or "param2"

        ax.set_xlabel(ics_label)
        ax.set_ylabel(ics2_label)
        ax.legend(framealpha=0.9, edgecolor="none", fontsize=7)
        _format_fig(fig)
        if save:
            fig.savefig(save, dpi=500, bbox_inches="tight")
        return ax

    def plot_3d(self, ax=None, ICS=None, ICS2=None, VOI=None, save=None, n_orbit_samples=40, **kwargs):
        """Plot 3D bifurcation diagram with periodic orbit surfaces.

        Shows:
        - Codim-1 equilibrium backbone (stable=solid, unstable=dashed)
        - Periodic orbit tube surface with W displacement cross-sections
        - Codim-2 curves (Hopf/fold loci)
        - Special codim-2 points (BT, GH, cusp, ZH)

        Axis convention:
            x = primary free parameter (codim-1, e.g. I)
            y = secondary parameter (codim-2, e.g. b)
            z = state variable (VOI, e.g. V)
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        _apply_style()
        VOI = self._resolve_voi(VOI)
        ics_label = ICS if ICS else getattr(self, "ICS", "param")

        sv_names = list(self.state_var_index.keys()) if self.state_var_index else []
        sv2 = None
        for sv in sv_names:
            if sv != VOI:
                sv2 = sv
                break

        c2_list = getattr(self, "codim2_curves", None) or []

        if ax is None:
            fig = plt.figure(figsize=(7.2, 5.5))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.get_figure()

        _format_3d_axes(ax)

        # Resolve codim-2 parameter names
        c2_ics = None
        if c2_list:
            c2_ics = getattr(c2_list[0], "_ics_name", None)

        # Default value of codim-2 parameter (backbone position)
        c2_default = None
        if c2_ics and hasattr(self, "model") and self.model:
            params = getattr(self.model, "parameters", {})
            if c2_ics in params:
                p = params[c2_ics]
                c2_default = float(getattr(p, "value", None) or 0)
        if c2_default is None:
            all_y = []
            for c2 in c2_list:
                if "param" in c2.df.columns:
                    all_y.extend(c2.df["param"].values)
            c2_default = np.median(all_y) if all_y else 0

        # ── 1. Codim-1 backbone ──
        if not self.df.empty and "param" in self.df.columns:
            voi_eq = compute_voi(self.df, VOI, state_var_index=self.state_var_index)
            if "stable" not in self.df.columns:
                self.df["stable"] = True
            if "segment" not in self.df.columns:
                self.df["segment"] = (self.df.stable != self.df.stable.shift()).cumsum()

            for _, seg in self.df.groupby("segment"):
                is_stable = seg.iloc[0].stable
                seg_voi = compute_voi(seg, VOI, state_var_index=self.state_var_index)
                y_bb = np.full(len(seg), c2_default)
                style = "-" if is_stable else "--"
                lw = 1.5 if is_stable else 0.9
                lbl = "Stable eq." if is_stable else "Unstable eq."
                clabels = ax.get_legend_handles_labels()[1]
                ax.plot(
                    seg["param"].values,
                    y_bb,
                    seg_voi.values,
                    style,
                    color=(_C["stable"] if is_stable else _C["unstable"]),
                    linewidth=lw,
                    alpha=0.9,
                    label=lbl if lbl not in clabels else None,
                    zorder=5,
                )

            # Hopf markers on backbone
            for hi in self.hopf_indices:
                clabels = ax.get_legend_handles_labels()[1]
                ax.scatter(
                    [self.df.at[hi, "param"]],
                    [c2_default],
                    [voi_eq.iloc[hi]],
                    s=40,
                    marker="o",
                    zorder=10,
                    facecolors=_C["hopf"],
                    edgecolors="white",
                    linewidths=0.8,
                    label="Hopf" if "Hopf" not in clabels else None,
                )

        # ── 2. Periodic orbit surfaces ──
        po_list = getattr(self, "periodic_orbits", None) or []
        has_orbits = False

        for po_br in po_list:
            if po_br.df.empty:
                continue
            orbits = po_br.extract_orbit_meshes(n_samples=n_orbit_samples)
            if not orbits or not sv2:
                max_col = f"max_{VOI}"
                min_col = f"min_{VOI}"
                if max_col in po_br.df.columns:
                    y_po = np.full(len(po_br.df), c2_default)
                    for col in [max_col, min_col]:
                        clabels = ax.get_legend_handles_labels()[1]
                        ax.plot(
                            po_br.df["param"].values,
                            y_po,
                            po_br.df[col].values,
                            "-",
                            color=_C["po_line"],
                            linewidth=0.9,
                            alpha=0.6,
                            label=("PO envelope" if "PO envelope" not in clabels else None),
                        )
                continue

            has_orbits = True
            n_theta = 80
            n_orb = len(orbits)
            I_vals = np.array([o["param"] for o in orbits])
            V_mesh = np.zeros((n_orb, n_theta))
            W_mesh = np.zeros((n_orb, n_theta))

            for j, orb in enumerate(orbits):
                t = orb["t"]
                t_norm = (t - t[0]) / (t[-1] - t[0])
                t_uni = np.linspace(0, 1, n_theta, endpoint=False)
                V_mesh[j, :] = np.interp(t_uni, t_norm, orb[VOI])
                W_mesh[j, :] = np.interp(t_uni, t_norm, orb[sv2])

            # Scale W displacement for tube cross-section
            y_range = 0
            if c2_list:
                y_vals = []
                for c2 in c2_list:
                    if "param" in c2.df.columns:
                        y_vals.extend(c2.df["param"].values)
                if y_vals:
                    y_range = max(y_vals) - min(y_vals)
            if y_range == 0:
                y_range = abs(c2_default) * 0.5 or 5.0

            W_range = W_mesh.max() - W_mesh.min()
            if W_range > 0:
                W_scaled = (W_mesh - W_mesh.mean()) / W_range * y_range * 0.15
            else:
                W_scaled = np.zeros_like(W_mesh)

            X = np.repeat(I_vals[:, None], n_theta, axis=1)
            Y = c2_default + W_scaled
            Z = V_mesh

            ax.plot_surface(
                X,
                Y,
                Z,
                alpha=0.2,
                color=_C["po_surface"],
                edgecolor="none",
                shade=True,
                zorder=3,
            )

            # Wireframe orbit loops (every ~5th)
            step = max(1, n_orb // 8)
            for j in range(0, n_orb, step):
                xx = np.append(X[j, :], X[j, 0])
                yy = np.append(Y[j, :], Y[j, 0])
                zz = np.append(Z[j, :], Z[j, 0])
                clabels = ax.get_legend_handles_labels()[1]
                ax.plot(
                    xx,
                    yy,
                    zz,
                    color=_C["po_line"],
                    linewidth=0.5,
                    alpha=0.4,
                    zorder=4,
                    label=("Periodic orbit" if "Periodic orbit" not in clabels else None),
                )

        # Envelope fallback
        if not has_orbits:
            for po_br in po_list:
                if po_br.df.empty:
                    continue
                max_col = f"max_{VOI}"
                min_col = f"min_{VOI}"
                if max_col in po_br.df.columns:
                    y_po = np.full(len(po_br.df), c2_default)
                    for col in [max_col, min_col]:
                        clabels = ax.get_legend_handles_labels()[1]
                        ax.plot(
                            po_br.df["param"].values,
                            y_po,
                            po_br.df[col].values,
                            "-",
                            color=_C["po_line"],
                            linewidth=0.9,
                            alpha=0.6,
                            label=("PO envelope" if "PO envelope" not in clabels else None),
                        )

        # ── 3. Codim-2 curves ──
        _c2_colors = {
            "fold": _C["c2_fold"],
            "hopf": _C["c2_hopf"],
            "bp": _C["bp"],
        }

        for c2 in c2_list:
            if c2.df.empty:
                continue
            src_type = getattr(c2, "_source_type", "fold")
            color = _c2_colors.get(src_type, "#555555")

            param2_col = "param2"
            if param2_col not in c2.df.columns:
                fp2 = getattr(c2, "_fp2_name", None)
                if fp2 and fp2 in c2.df.columns:
                    param2_col = fp2
                else:
                    continue

            voi_vals = (
                compute_voi(c2.df, VOI, state_var_index=self.state_var_index) if VOI in c2.df.columns else c2.df.get(VOI)
            )
            if voi_vals is None:
                voi_vals = np.full(len(c2.df), 0)

            label = f"{src_type.capitalize()} curve"
            clabels = ax.get_legend_handles_labels()[1]
            ax.plot(
                c2.df[param2_col],
                c2.df["param"],
                voi_vals,
                "-",
                color=color,
                linewidth=1.8,
                label=label if label not in clabels else None,
                zorder=8,
            )

            # Special points
            _sp_map = {
                "bt": ("s", _C["bt"], "BT"),
                "gh": ("^", _C["gh"], "GH"),
                "cusp": ("D", _C["cusp"], "Cusp"),
                "zh": ("v", _C["zh"], "ZH"),
                "hh": ("p", _C["gh"], "HH"),
            }
            if "specialpoint" in c2.df.columns:
                for i, r in c2.df[c2.df["specialpoint"].notna()].iterrows():
                    sp_raw = str(r.specialpoint).lower()
                    for sp in sp_raw.split(","):
                        sp = sp.strip()
                        if sp in ("endpoint", "none", "nan", ""):
                            continue
                        if sp not in _sp_map:
                            continue
                        z = voi_vals.loc[i] if hasattr(voi_vals, "loc") else voi_vals[i]
                        mk, clr, lbl = _sp_map[sp]
                        clabels = ax.get_legend_handles_labels()[1]
                        ax.scatter(
                            [r[param2_col]],
                            [r["param"]],
                            [z],
                            s=45,
                            zorder=12,
                            marker=mk,
                            facecolors=clr,
                            edgecolors="white",
                            linewidths=0.8,
                            label=lbl if lbl not in clabels else None,
                        )

        ics2_label = ICS2 or c2_ics or "param2"
        ax.set_xlabel(ics_label, fontsize=8, labelpad=6)
        ax.set_ylabel(ics2_label, fontsize=8, labelpad=6)
        ax.set_zlabel(VOI, fontsize=8, labelpad=6)
        ax.view_init(elev=25, azim=-55)
        ax.legend(framealpha=0, edgecolor="none", fontsize=7, loc="upper left", borderpad=0.5)
        if save:
            fig.savefig(save, dpi=500, bbox_inches="tight")
        return ax


class PyRatesBifurcationResult(BifurcationResult):
    """Bifurcation result from PyRates/PyCoBi (AUTO-07p) backend.

    Provides the same interface as ``BifurcationResult`` but extracts
    data from PyCoBi's ``ODESystem`` instead of juliacall objects.
    """

    def __init__(
        self,
        ode,
        cont_name,
        model=None,
        state_var_names=None,
        icp=1,
        fp_name="param",
        periodic_orbit_results=None,
        codim2_results=None,
        **kwargs,
    ):
        # Skip parent __init__ (it uses juliacall)
        self.br = None
        self.ode = ode
        self.model = model
        self.ICS = fp_name
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.state_var_index = {}
        if state_var_names:
            self.state_var_index = {name: idx for idx, name in enumerate(state_var_names)}

        # Extract continuation data from PyCoBi's summary DataFrame
        self.df = self._extract_pycobi_df(ode, cont_name, state_var_names, icp)

        # Build hopf/bp index lists
        self.hopf_indices = []
        self.bp_indices = []
        self.hopf_steps = []
        self.bp_steps = []
        if "specialpoint" in self.df.columns:
            sp_series = self.df["specialpoint"].astype(str)
            hopf_mask = sp_series.str.contains("hopf|HB", case=False, na=False)
            bp_mask = sp_series.str.contains("bp|BP", case=False, na=False)
            self.hopf_indices = self.df.index[hopf_mask].tolist()
            self.bp_indices = self.df.index[bp_mask].tolist()
            if "step" in self.df.columns:
                self.hopf_steps = self.df.loc[hopf_mask, "step"].tolist()
                self.bp_steps = self.df.loc[bp_mask, "step"].tolist()

        # Handle periodic orbits
        self.periodic_orbits = []
        if periodic_orbit_results:
            for po_name, po_cont in periodic_orbit_results:
                po_res = PyRatesBifurcationResult(
                    ode=ode,
                    cont_name=po_name,
                    model=model,
                    state_var_names=state_var_names,
                    icp=icp,
                    fp_name=fp_name,
                )
                self.periodic_orbits.append(po_res)

        # Handle codim-2 curves
        self.codim2_curves = []
        if codim2_results:
            for c2_res in codim2_results:
                # Extract the 2nd parameter column into df['param2']
                icp2 = getattr(c2_res, "_icp2", None)
                if icp2 is not None and not c2_res.df.empty:
                    c2_res.df = self._add_param2_column(
                        ode,
                        c2_res.df,
                        getattr(c2_res, "_cont_name", None) or cont_name,
                        state_var_names,
                        icp2,
                    )
                self.codim2_curves.append(c2_res)

            # Store _ICS2 for labeling
            if codim2_results:
                self._ICS2 = getattr(codim2_results[0], "_fp2_name", "param2")

    # Standard columns expected by plot_branch / plot_special_points
    _STANDARD_COLS = [
        "param",
        "stable",
        "step",
        "specialpoint",
        "n_unstable",
        "n_imag",
    ]

    @staticmethod
    def _add_param2_column(ode, df, cont_name, state_var_names, icp2):
        """Add 'param2' column to a codim-2 DataFrame.

        Extracts the second free parameter from raw branch data.
        """
        par2_col = f"PAR({icp2})"
        try:
            cont_key = ode._results_map[cont_name]
            sol = ode.auto_solutions[cont_key]
            branch = sol.data[0]
            branch_data = branch.todict()
            if par2_col in branch_data:
                p2_vals = [float(v) for v in branch_data[par2_col]]
                # Align with df length
                if len(p2_vals) >= len(df):
                    df["param2"] = p2_vals[: len(df)]
                else:
                    # Interpolate or pad
                    df["param2"] = np.interp(
                        np.linspace(0, 1, len(df)),
                        np.linspace(0, 1, len(p2_vals)),
                        p2_vals,
                    )
        except (KeyError, IndexError, AttributeError):
            # Fallback: try from summary
            try:
                summary = ode.get_summary(cont_name)
                if summary is not None and len(summary) > 0:
                    cols = summary.columns
                    translated = ode._var_map_inv.get(par2_col, ode._var_map_inv.get(icp2, None))
                    for cand in [translated, par2_col]:
                        if cand and (cand, "") in cols:
                            p2_vals = [float(summary.iloc[i][(cand, "")]) for i in range(len(summary))]
                            if len(p2_vals) >= len(df):
                                df["param2"] = p2_vals[: len(df)]
                            break
            except Exception:
                pass
        return df

    @staticmethod
    def _extract_pycobi_df(ode, cont_name, state_var_names, icp):
        """Extract continuation branch as a pandas DataFrame from PyCoBi.

        Uses the full branch data from AUTO's raw solution (every
        continuation step) for a smooth curve, and overlays true
        bifurcation point labels from ``get_summary``.

        For periodic-orbit continuations the raw fort.7 data has mesh
        discretisation columns, so we fall back to ``get_summary``
        which provides proper min/max envelopes.
        """
        # True bifurcation types only (no UZ, EP, RG, MX)
        auto_to_bif = {
            "LP": "fold",
            "HB": "hopf",
            "BP": "bp",
            "PD": "pd",
            "TR": "ns",
            "BT": "bt",
            "CP": "cusp",
            "GH": "gh",
            "ZH": "zh",
        }

        def _empty_df():
            cols = list(state_var_names or []) + PyRatesBifurcationResult._STANDARD_COLS
            return pd.DataFrame(columns=cols)

        def _parse_stability(sv):
            if isinstance(sv, (bool, np.bool_)):
                return bool(sv)
            if isinstance(sv, (int, np.integer, float, np.floating)):
                return sv > 0
            if isinstance(sv, str):
                return sv.strip().upper() in ("S", "TRUE", "1")
            try:
                return bool(sv.item())
            except (AttributeError, ValueError):
                return True

        # --- Get summary (always needed for stability + special pts) ---
        try:
            summary = ode.get_summary(cont_name)
        except (KeyError, IndexError, AttributeError):
            summary = None

        # --- Full curve from raw AUTO branch data ---
        branch_data = None
        n_steps = 0
        try:
            cont_key = ode._results_map[cont_name]
            sol = ode.auto_solutions[cont_key]
            branch = sol.data[0]
            branch_data = branch.todict()
            n_steps = len(branch)
        except (KeyError, IndexError, AttributeError):
            pass

        # Detect periodic-orbit continuation: if get_summary has
        # MultiIndex tuple columns for state variables (e.g. ('V', 0)
        # and ('V', 1) for min/max envelope), use summary-based path.
        is_po = False
        if summary is not None and len(summary) > 0 and state_var_names:
            sv0 = state_var_names[0]
            auto0 = "U(1)"
            s_cols = summary.columns
            is_po = ((sv0, 0) in s_cols and (sv0, 1) in s_cols) or ((auto0, 0) in s_cols and (auto0, 1) in s_cols)

        par_col = f"PAR({icp})"

        def _col(name, cols):
            """Resolve a column name in a possibly-MultiIndex columns.

            PyCoBi's get_summary returns a MultiIndex where ALL
            columns are tuples: state vars as ``('V', 0)``,
            ``('V', 1)`` and metadata as ``('stability', '')``, etc.

            When columns are a MultiIndex, ``'stability' in cols``
            matches level-0, but ``rd['stability']`` returns a
            sub-Series instead of a scalar — so we must return the
            full tuple key.
            """
            # Check tuple forms first (safe for both Index and MultiIndex)
            if (name, "") in cols:
                return (name, "")
            if (name, 0) in cols:
                return (name, 0)
            # Plain string — only valid for a flat Index
            if isinstance(cols, pd.MultiIndex):
                return None
            if name in cols:
                return name
            return None

        # ── Periodic-orbit path: use get_summary with min/max ──
        if is_po or (n_steps == 0 and summary is not None):
            if summary is None or len(summary) == 0:
                return _empty_df()
            cols = summary.columns

            # Resolve parameter column (translated or raw)
            s_par_col = None
            translated = ode._var_map_inv.get(par_col, ode._var_map_inv.get(icp, None))
            for cand in [translated, par_col]:
                if cand:
                    resolved = _col(cand, cols)
                    if resolved is not None:
                        s_par_col = resolved
                        break

            # Resolve state-variable columns (MultiIndex tuples)
            sv_col_map = {}
            if state_var_names:
                for i, sv_name in enumerate(state_var_names):
                    auto_name = f"U({i + 1})"
                    val_col = max_col = None
                    for cand in [sv_name, auto_name]:
                        if (cand, 0) in cols:
                            val_col = (cand, 0)
                            max_col = (cand, 1) if (cand, 1) in cols else None
                            break
                        if cand in cols:
                            val_col = cand
                            break
                    if val_col is not None:
                        sv_col_map[sv_name] = {"val": val_col, "max": max_col}

            stab_col = _col("stability", cols)
            bif_col = _col("bifurcation", cols)

            rows = []
            for row_idx in range(len(summary)):
                rd = summary.iloc[row_idx]
                row = {}
                for sv_name, ci in sv_col_map.items():
                    v = rd[ci["val"]]
                    if ci["max"] is not None:
                        mx = rd[ci["max"]]
                        row[sv_name] = float(np.max(mx))
                        row[f"max_{sv_name}"] = float(np.max(mx))
                        row[f"min_{sv_name}"] = float(np.min(v))
                    elif hasattr(v, "__len__") and not isinstance(v, str):
                        row[sv_name] = float(np.max(v))
                        row[f"max_{sv_name}"] = float(np.max(v))
                        row[f"min_{sv_name}"] = float(np.min(v))
                    else:
                        row[sv_name] = float(v)
                row["param"] = float(rd[s_par_col]) if s_par_col else np.nan
                stab = True
                if stab_col is not None:
                    stab = _parse_stability(rd[stab_col])
                row["stable"] = stab
                row["step"] = row_idx
                bif_type = None
                if bif_col is not None:
                    bv = str(rd[bif_col]).strip()
                    bif_type = auto_to_bif.get(bv)
                row["specialpoint"] = bif_type
                row["n_unstable"] = 0
                row["n_imag"] = 0
                rows.append(row)

            if not rows:
                return _empty_df()
            df = pd.DataFrame(rows)
            df["stable"] = df["stable"].astype(bool)
            return df

        # ── Equilibrium path: full curve from raw branch data ──
        if branch_data is None or n_steps == 0:
            return _empty_df()

        rows = []
        for step in range(n_steps):
            row = {}
            if state_var_names:
                for i, sv_name in enumerate(state_var_names):
                    auto_col = f"U({i + 1})"
                    if auto_col in branch_data:
                        row[sv_name] = float(branch_data[auto_col][step])
            if par_col in branch_data:
                row["param"] = float(branch_data[par_col][step])
            else:
                row["param"] = np.nan
            row["stable"] = True
            row["step"] = step
            row["specialpoint"] = None
            row["n_unstable"] = 0
            row["n_imag"] = 0
            rows.append(row)

        if not rows:
            return _empty_df()

        df = pd.DataFrame(rows)

        # Overlay stability + special points from summary
        if summary is not None and len(summary) > 0:
            cols = summary.columns
            s_par_col = None
            translated = ode._var_map_inv.get(par_col, ode._var_map_inv.get(icp, None))
            for cand in [translated, par_col]:
                if cand:
                    resolved = _col(cand, cols)
                    if resolved is not None:
                        s_par_col = resolved
                        break

            stab_col = _col("stability", cols)
            bif_col = _col("bifurcation", cols)

            stab_data = []
            for row_idx in range(len(summary)):
                rd = summary.iloc[row_idx]
                p_val = float(rd[s_par_col]) if s_par_col else np.nan
                stab = _parse_stability(rd[stab_col]) if stab_col is not None else True
                bif_type = None
                if bif_col is not None:
                    bv = str(rd[bif_col]).strip()
                    bif_type = auto_to_bif.get(bv)
                stab_data.append((p_val, stab, bif_type))

            if stab_data and s_par_col:
                lp = np.array([s[0] for s in stab_data])
                ls = np.array([s[1] for s in stab_data])
                bp = df["param"].values
                nearest = np.abs(bp[:, None] - lp[None, :]).argmin(axis=1)
                df["stable"] = ls[nearest]
                for _, (pv, __, bt) in enumerate(stab_data):
                    if bt is None:
                        continue
                    si = int(np.abs(bp - pv).argmin())
                    df.loc[si, "specialpoint"] = bt

        df["stable"] = df["stable"].astype(bool)
        return df


__all__ = ["BifurcationResult", "PyRatesBifurcationResult"]
