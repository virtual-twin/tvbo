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


# ── Unified style registry for bifurcation diagram elements ──────────
# Single source of truth for branch lines (SFP/UFP/SLC/ULC) and special
# points (LP/HB/BP/PD/TR/CP/GH/ZH/BT/HH). Used by plot_branch,
# plot_special_points, plot_3d, and the BifLegend helper.

_LINE_BASE = dict(marker="", lw=1.5, picker=True, pickradius=8, zorder=5)
_MARK_BASE = dict(lw=0, linestyle="none", fillstyle="full",
                  markeredgecolor="white", markeredgewidth=0.8,
                  markersize=8, alpha=1, zorder=50)

BIF_STYLES = {
    # Branch lines
    "SFP": dict(color=_C["stable"],   ls="-", lw=1.5, label="Stable FP",   zorder=5),
    "UFP": dict(color=_C["stable"],   ls=":", lw=1.5, label="Unstable FP", zorder=5),
    "SLC": dict(color=_C["po_line"],  ls="-", lw=1.5, label="Stable LC",   zorder=5),
    "ULC": dict(color=_C["po_line"],  ls=":", lw=1.5, label="Unstable LC", zorder=5),
    # Codim-1 special points
    "LP": dict(marker="o", color=_C["fold"], label="LP"),     # fold / saddle-node
    "HB": dict(marker="o", color=_C["hopf"], label="Hopf"),   # Hopf
    "BP": dict(marker="s", color=_C["bp"],   label="BP"),     # branch point
    "PD": dict(marker="^", color=_C["pd"],   label="PD"),     # period doubling
    "TR": dict(marker="*", color=_C["ns"],   label="Torus"),  # torus / Neimark-Sacker
    # Codim-2 special points
    "CP": dict(marker="D", color=_C["cusp"], label="Cusp"),
    "GH": dict(marker="d", color=_C["gh"],   label="GH"),
    "BT": dict(marker="*", color=_C["bt"],   label="BT"),
    "ZH": dict(marker="o", color=_C["zh"],   label="ZH"),
    "HH": dict(marker="p", color=_C["gh"],   label="HH"),
    # PO-on-LC special points (prefix LC_)
    "LC_LP": dict(marker="o", color="purple",     label="LC LP"),
    "LC_PD": dict(marker="^", color="cyan",       label="LC PD"),
    "LC_TR": dict(marker="*", color="yellowgreen", label="LC TR"),
    "LC_BP": dict(marker="s", color=_C["bp"],     label="LC BP"),
    "LC_EP": dict(marker="none", color=_C["po_line"], label="LC EP"),
}

# Canonical TY normalisation (lower-case backend strings → upper TY key)
_TY_ALIASES = {
    # equilibria
    "fold": "LP", "saddle-node": "LP", "sn": "LP", "lp": "LP",
    "hopf": "HB", "hb": "HB",
    "bp": "BP", "branchpoint": "BP", "branch-point": "BP",
    # PO codim-1
    "pd": "PD", "period-doubling": "PD",
    "ns": "TR", "tr": "TR", "neimark-sacker": "TR", "torus": "TR",
    # codim-2
    "cusp": "CP", "cp": "CP",
    "gh": "GH", "bautin": "GH",
    "bt": "BT", "bogdanov-takens": "BT",
    "zh": "ZH", "zero-hopf": "ZH",
    "hh": "HH", "hopf-hopf": "HH",
}


def canonical_ty(ty):
    """Normalise any backend label to a key in ``BIF_STYLES``."""
    if ty is None:
        return None
    s = str(ty).strip()
    if not s:
        return None
    # Already canonical (incl. LC_ prefix)
    if s in BIF_STYLES:
        return s
    su = s.upper()
    if su in BIF_STYLES:
        return su
    return _TY_ALIASES.get(s.lower())


def get_bif_style(ty, base=None):
    """Look up the merged style dict for a TY (e.g. 'LP', 'fold', 'SLC').

    ``base`` defaults to the line/marker base style depending on whether
    the entry is a branch (no marker) or a point (marker is present).
    """
    key = canonical_ty(ty) or ty
    style = BIF_STYLES.get(key)
    if style is None:
        return None
    if base is None:
        base = _MARK_BASE if "marker" in style and style.get("marker") not in (None, "", "none") else _LINE_BASE
    out = dict(base)
    out.update(style)
    return out


# ── Coord DSL & PO orbit reductions ──────────────────────────────────
# Reduce an orbit mesh (n_orbit, n_time) → (n_orbit,) per-step value.

def _reduce_minmax(arr):
    return np.array([arr.min(axis=-1), arr.max(axis=-1)])


def _reduce_avg(arr, t=None):
    if t is None:
        return arr.mean(axis=-1)
    return np.trapz(arr, t, axis=-1) / (t[-1] - t[0])


def _reduce_min(arr):
    return arr.min(axis=-1)


def _reduce_max(arr):
    return arr.max(axis=-1)


def _reduce_norm(arr):
    return np.sqrt((arr ** 2).mean(axis=-1))


PO_REDUCTIONS = {
    "minmax": _reduce_minmax,
    "avg": _reduce_avg,
    "min": _reduce_min,
    "max": _reduce_max,
    "norm": _reduce_norm,
}


def resolve_coord(df, expr, state_var_index=None, po_orbits=None):
    """Evaluate one coord expression against ``df`` (and optionally PO meshes).

    Accepts:
      * a column name in ``df`` (e.g. ``'x'``, ``'param'``)
      * a sympy-parseable expression of column names (e.g. ``'V**2 + W'``)
      * a reduction call ``'minmax(V)'`` / ``'avg(V)'`` / ``'norm(V)'``
        which uses the PO orbit meshes if available, else falls back to
        the column itself.

    Returns a pandas Series (or ndarray for minmax → shape (2, N)).
    """
    if expr is None:
        return None
    s = str(expr).strip()
    # Reduction call
    for name, fn in PO_REDUCTIONS.items():
        prefix = f"{name}("
        if s.startswith(prefix) and s.endswith(")"):
            sv = s[len(prefix):-1].strip()
            if po_orbits:
                arr = np.array([np.asarray(o[sv]) for o in po_orbits])  # (n_orb, n_t)
                if name == "avg":
                    return fn(arr, t=np.asarray(po_orbits[0]["t"]))
                return fn(arr)
            # No PO meshes: fall through to plain column eval
            return compute_voi(df, sv, state_var_index=state_var_index)
    # Plain column or expression
    return compute_voi(df, s, state_var_index=state_var_index)


def resolve_coords(coords, df, state_var_index=None, po_orbits=None,
                   default_voi=None, default_param="param"):
    """Normalise the user's ``coords`` argument to a tuple of arrays.

    ``coords`` accepts:
      * ``None``               → ``(param, default_voi)``
      * ``'V'``                → ``(param, V)``
      * ``'minmax(V)'``        → ``(param, [Vmin, Vmax])``
      * ``(p, V)``             → 2D
      * ``(p, V, W)``          → 3D
    Each element may itself be a string or already-resolved array.
    """
    if coords is None:
        coords = (default_param, default_voi)
    elif isinstance(coords, str):
        coords = (default_param, coords)
    elif not isinstance(coords, (list, tuple)):
        coords = (default_param, coords)

    out = []
    for c in coords:
        if isinstance(c, str):
            out.append(resolve_coord(df, c, state_var_index=state_var_index, po_orbits=po_orbits))
        else:
            out.append(c)
    return tuple(out)


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
    """Backend-agnostic bifurcation result.

    A single ``BifurcationResult`` represents one continuation branch
    (equilibrium, periodic orbit, or codim-2 curve) regardless of the
    backend that produced it (BifurcationKit.jl, PyRates/PyCoBi,
    AUTO-07p/numcont). Once the data lives in ``self.df`` and the
    nested ``periodic_orbits`` / ``codim2_curves`` lists, *all* plotting,
    legend, and export methods (``plot``, ``plot_3d``, ``bif_legend``,
    ``enable_picker``, ...) work uniformly across backends.

    There are *no* backend-specific result subclasses. Each adapter
    extracts a unified DataFrame and either calls the constructor
    directly with ``df=...`` or one of the factory shortcuts:

    * ``BifurcationResult.from_bifkit(br, ...)`` — BifurcationKit.jl
      ``ContResult`` (juliacall).
    * ``BifurcationResult.from_pycobi(ode, cont_name, ...)`` —
      PyRates / PyCoBi ``ODESystem`` continuations.
    * ``BifurcationResult.from_auto(bd, ...)`` — in-tree AUTO-07p
      ``bifDiag`` (numcont backend).

    All visual differences between backends are encoded in the unified
    :data:`BIF_STYLES` registry, not in subclasses.
    """

    def __init__(self, br=None, *, df=None, **kwargs):
        self.br = br
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Extract state variable names and create index mapping
        self.state_var_index = kwargs.get("state_var_index", {})
        if not self.state_var_index and getattr(self, "model", None) is not None:
            if hasattr(self.model, "state_variables"):
                self.state_var_index = {n: i for i, n in enumerate(self.model.state_variables.keys())}

        # Path A — pre-extracted DataFrame (from any adapter):
        # base class just stores it and finalises bookkeeping.
        if df is not None:
            self.df = df
            if not hasattr(self, "codim2_curves"):
                self.codim2_curves = []
            if not hasattr(self, "periodic_orbits"):
                self.periodic_orbits = []
            self._finalize()
            return

        # Path B — raw juliacall BifurcationKit ``ContResult``:
        # extract a DataFrame the same way the BK adapter would.
        sp_list = None
        kind = continuation_kind(br) if br is not None else None
        if kind in ("EquilibriumCont", "PeriodicOrbitCont", "HopfCont", "FoldCont"):
            self.df = _extract_equilibrium_df(br)
            sp_list = _extract_special_points(br)
        if not hasattr(self, "df"):
            self.df = pd.DataFrame()
        if not hasattr(self, "codim2_curves"):
            self.codim2_curves = []
        if not hasattr(self, "periodic_orbits"):
            self.periodic_orbits = []

        # Annotate special points (BK path only — PyRates/NumCont
        # already populate ``df['specialpoint']`` in their adapters)
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
                    rows = [int(np.abs(self.df.param - pval).argmin())] if np.isfinite(pval) else []
                for rix in rows:
                    existing = self.df.at[rix, "specialpoint"]
                    if existing is None or existing == "":
                        self.df.at[rix, "specialpoint"] = typ
                    elif typ not in str(existing).split(","):
                        self.df.at[rix, "specialpoint"] = f"{existing},{typ}"
                    self.df.at[rix, "sp_norm"] = norm
                    self.df.at[rix, "sp_idx"] = idx_val

        self._finalize()

    # ── Shared post-extraction bookkeeping ──────────────────────────────
    def _finalize(self):
        """Populate ``hopf_indices`` / ``bp_indices`` (+ matching steps).

        Called once by ``__init__`` regardless of which adapter built
        the DataFrame.
        """
        self.hopf_indices = []
        self.bp_indices = []
        self.hopf_steps = []
        self.bp_steps = []
        if self.df is None or self.df.empty or "specialpoint" not in self.df.columns:
            return
        sp = self.df["specialpoint"].astype(str)
        hopf_mask = sp.str.contains("hopf|HB", case=False, na=False, regex=True)
        bp_mask = sp.str.contains(r"\bbp\b|BP", case=False, na=False, regex=True)
        self.hopf_indices = self.df.index[hopf_mask].tolist()
        self.bp_indices = self.df.index[bp_mask].tolist()
        if "step" in self.df.columns:
            self.hopf_steps = self.df.loc[hopf_mask, "step"].tolist()
            self.bp_steps = self.df.loc[bp_mask, "step"].tolist()

    # ── Backend factory shortcuts ───────────────────────────────────────
    @classmethod
    def from_bifkit(cls, br, **kwargs):
        """Wrap a BifurcationKit.jl ``ContResult`` (juliacall object)."""
        return cls(br=br, **kwargs)

    @classmethod
    def from_pycobi(cls, ode, cont_name, *, model=None, state_var_names=None,
                    icp=1, fp_name="param", periodic_orbit_results=None,
                    codim2_results=None, **kwargs):
        """Wrap a PyRates / PyCoBi continuation by name.

        All visualisation/export logic lives on this class -- the
        adapter just hands the extracted DataFrame straight to
        ``__init__``.
        """
        sv_names = list(state_var_names or [])
        df = _extract_pycobi_df(ode, cont_name, sv_names, icp)

        # Recursively wrap nested PO continuations
        periodic_orbits = []
        for po_name, _po_cont in periodic_orbit_results or []:
            periodic_orbits.append(
                cls.from_pycobi(
                    ode, po_name,
                    model=model, state_var_names=sv_names,
                    icp=icp, fp_name=fp_name,
                )
            )

        # Codim-2 curves: existing BifurcationResult instances; just
        # augment with their second-parameter trajectory.
        codim2_curves = []
        ICS2 = None
        for c2_res in codim2_results or []:
            icp2 = getattr(c2_res, "_icp2", None)
            if icp2 is not None and not c2_res.df.empty:
                c2_res.df = _add_pycobi_param2(
                    ode, c2_res.df,
                    getattr(c2_res, "_cont_name", None) or cont_name,
                    sv_names, icp2,
                )
            codim2_curves.append(c2_res)
        if codim2_results:
            ICS2 = getattr(codim2_results[0], "_fp2_name", "param2")

        result = cls(
            br=None,
            df=df,
            model=model,
            ICS=fp_name,
            ode=ode,
            state_var_index={n: i for i, n in enumerate(sv_names)},
            periodic_orbits=periodic_orbits,
            codim2_curves=codim2_curves,
            **kwargs,
        )
        if ICS2 is not None:
            result._ICS2 = ICS2
        return result

    @classmethod
    def from_auto(cls, bd, *, cont_name=None, model=None, continuation=None,
                  ICS=None, periodic_orbits_raw=None, codim2_raw=None,
                  workdir=None, **kwargs):
        """Wrap an AUTO-07p ``bifDiag`` (numcont backend).

        Parameters
        ----------
        bd : auto.bifDiag
            The codim-1 equilibrium continuation result.
        codim2_raw : list, optional
            ``[(name, source_type, fp1_name, fp2_name, R_c2), …]`` —
            codim-2 fold/Hopf/BP curves produced by
            ``NumContAdapter._run_codim2_branches``. Each entry is wrapped
            as a child ``BifurcationResult`` and attached to
            ``self.codim2_curves`` with metadata (``_source_type``,
            ``_fp2_name``) that ``_plot_codim2`` consumes.
        """
        sv_names = list(model.state_variables.keys()) if model else []
        df = _extract_auto_df(bd, sv_names, ICS)

        periodic_orbits = []
        for po_name, po_bd in periodic_orbits_raw or []:
            periodic_orbits.append(
                cls.from_auto(
                    po_bd, cont_name=po_name,
                    model=model, continuation=continuation,
                    ICS=ICS, workdir=workdir,
                )
            )

        codim2_curves = []
        for c2_name, c2_source_type, c2_fp1, c2_fp2, c2_bd in codim2_raw or []:
            # The codim-2 continuation tracks (fp1, fp2). ICS=fp1 makes the
            # 'param' column carry fp1; the 'param2' column is fp2 (already
            # resolved by _auto_branch_to_df from PAR(.) → parnames mapping).
            c2_df = _extract_auto_df(c2_bd, sv_names, c2_fp1)
            # Promote second-parameter column for _plot_codim2 ergonomics
            if c2_fp2 in c2_df.columns and "param2" not in c2_df.columns:
                c2_df = c2_df.copy()
                c2_df["param2"] = c2_df[c2_fp2]
            c2_result = cls(
                br=c2_bd,
                df=c2_df,
                model=model,
                ICS=c2_fp1,
                state_var_index={n: i for i, n in enumerate(sv_names)},
                cont_name=c2_name,
                continuation=continuation,
                workdir=workdir,
            )
            # Metadata consumed by _plot_codim2
            c2_result._source_type = c2_source_type
            c2_result._fp2_name = c2_fp2
            c2_result._ICS2 = c2_fp2
            codim2_curves.append(c2_result)

        return cls(
            br=bd,
            df=df,
            model=model,
            ICS=ICS,
            state_var_index={n: i for i, n in enumerate(sv_names)},
            periodic_orbits=periodic_orbits,
            codim2_curves=codim2_curves,
            cont_name=cont_name,
            continuation=continuation,
            workdir=workdir,
            **kwargs,
        )



    def plot_special_points(self, VOI, ax=None, types=None, **kwargs):
        """Mark codim-1 special points (LP/HB/BP/PD/TR/...) on ``ax``.

        Parameters
        ----------
        types : iterable[str], optional
            Restrict markers to these canonical TYs (e.g. ``['LP','HB']``).
            By default every TY found in ``df.specialpoint`` is plotted
            (except ``endpoint``).
        """
        if ax is None or "specialpoint" not in self.df.columns:
            return
        is_po = bool(getattr(self, "is_po", False))
        type_filter = {canonical_ty(t) for t in types} if types else None
        clabels = ax.get_legend_handles_labels()[1]

        for i, r in self.df[self.df["specialpoint"].notna()].iterrows():
            sp_raw = str(r.specialpoint)
            for sp in sp_raw.split(","):
                sp = sp.strip()
                if sp.lower() in ("endpoint", "none", "nan", ""):
                    continue
                key = canonical_ty(sp)
                if key is None:
                    continue
                if is_po and not key.startswith("LC_"):
                    key = f"LC_{key}"
                if type_filter and key not in type_filter:
                    continue
                style = get_bif_style(key)
                if style is None:
                    continue
                # scatter-style draw via plot for legend consistency
                voi_val = compute_voi(self.df, VOI, state_var_index=self.state_var_index).loc[i]
                lab = style.get("label")
                ax.plot(
                    [r.param], [voi_val],
                    marker=style.get("marker", "o"),
                    color=style.get("color", "#333"),
                    markersize=style.get("markersize", 8),
                    markeredgecolor=style.get("markeredgecolor", "white"),
                    markeredgewidth=style.get("markeredgewidth", 0.8),
                    linestyle="none",
                    zorder=style.get("zorder", 50),
                    label=(lab if lab and lab not in clabels else None),
                )
                if lab and lab not in clabels:
                    clabels.append(lab)

    def plot_branch(self, ax, ICS=None, VOI=None, **kwargs):
        """Draw the continuation branch as stability-coded line segments.

        Splits the branch into contiguous stable/unstable runs (and, when a
        `branch_id` column is present, per branch) and plots each segment with
        the style from the central registry — `SFP`/`UFP` for equilibria or
        `SLC`/`ULC` for periodic orbits. Does nothing when the branch is empty.

        Args:
            ax: Matplotlib axes to draw on.
            ICS: Free (continuation) parameter name; accepted for parity with
                the other plot methods and not used here.
            VOI: Variable of interest plotted on the y-axis; resolved to a
                default branch column when `None`.
            **kwargs: Line-style overrides forwarded to `matplotlib` (e.g.
                `linewidth`/`lw`, `color`, `linestyle`); continuation-config
                keys are filtered out before plotting.
        """
        VOI = self._resolve_voi(VOI)
        if self.df.empty:
            return
        is_po = bool(getattr(self, "is_po", False))
        lw = kwargs.pop("linewidth", kwargs.pop("lw", None))
        _plot_ignore = {
            "periodic_orbits", "verbose", "max_steps", "ds", "dsmin", "dsmax",
            "p_min", "p_max", "quiet", "detect_bifurcation", "nev", "n_inversion",
            "max_bisection_steps", "tol_stability", "bothside", "bifurcation_points",
            "n_runs", "model", "state_var_index", "ICS", "types", "coords",
            "po_orbits", "branch_id",
        }
        plot_kwargs = {k: v for k, v in kwargs.items() if k not in _plot_ignore}
        if "stable" not in self.df.columns:
            self.df["stable"] = True
        # Group per (optional branch_id, contiguous-stability) segment
        if "branch_id" in self.df.columns:
            group_cols = ["branch_id"]
        else:
            group_cols = []
        df_plot = self.df.copy()
        df_plot["__seg"] = (df_plot.stable != df_plot.stable.shift()).cumsum()
        clabels = ax.get_legend_handles_labels()[1]
        for _, segment_data in df_plot.groupby(group_cols + ["__seg"]):
            is_stable = bool(segment_data.iloc[0].stable)
            ty = ("SLC" if is_stable else "ULC") if is_po else ("SFP" if is_stable else "UFP")
            style = get_bif_style(ty)
            kw = dict(style)
            if lw is not None:
                kw["lw"] = lw
            kw.update(plot_kwargs)
            lab = kw.pop("label", None)
            ax.plot(
                segment_data["param"],
                compute_voi(segment_data, VOI, state_var_index=self.state_var_index),
                color=kw.pop("color"),
                linestyle=kw.pop("ls", kw.pop("linestyle", "-")),
                linewidth=kw.pop("lw", kw.pop("linewidth", 1.5)),
                zorder=kw.pop("zorder", 5),
                label=(lab if lab and lab not in clabels else None),
                **{k: v for k, v in kw.items() if k not in ("marker", "picker", "pickradius", "fillstyle", "alpha")},
            )
            if lab and lab not in clabels:
                clabels.append(lab)

    def bif_legend(self, ax, tys, labels=None, **lgd_kwargs):
        """Add a curated legend listing the selected TYs.

        Mirrors ``ContinuationPlot.BifLegend``: draws an off-screen artist
        per TY using the central style registry and feeds them to a single
        ``ax.legend`` call so the user can pin exactly which entries appear.
        """
        handles = []
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        for k, ty in enumerate(tys):
            style = get_bif_style(ty)
            if style is None:
                continue
            kw = dict(style)
            if labels and k < len(labels):
                kw["label"] = labels[k]
            h, = ax.plot([xlim[0]], [ylim[0]], **{
                k_: v for k_, v in kw.items()
                if k_ not in ("picker", "pickradius", "fillstyle", "zorder")
            })
            h.remove()
            handles.append(h)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        return ax.legend(handles=handles, **lgd_kwargs)

    def plot_equilibrium_branch(self, ax, ICS=None, VOI=None, **kwargs):
        """Draw the equilibrium branch and overlay its special points.

        Convenience wrapper that calls `plot_branch` and then
        `plot_special_points` on the same axes.

        Args:
            ax: Matplotlib axes to draw on.
            ICS: Free (continuation) parameter name, passed through to the
                underlying calls.
            VOI: Variable of interest plotted on the y-axis; resolved to a
                default branch column when `None`.
            **kwargs: Style overrides forwarded to `plot_branch` and
                `plot_special_points`.
        """
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
            pass

        # AUTO-07p backend: ``self.br`` is a ``bifDiag`` whose call returns a
        # ``parseS`` of ``AUTOSolution`` objects. Each holds the full orbit
        # mesh in ``coordarray`` (shape ``(ndim, ntst*ncol+1)``), ``indepvararray``
        # (mesh times in [0, 1] times PERIOD) and ``PAR`` (parameter dict).
        try:
            sols = self.br()  # parseS
            n_sol = len(sols)
            if n_sol == 0:
                return []
            sv_names = list(self.state_var_index.keys()) if self.state_var_index else []
            if not sv_names:
                sv_names = list(sols[0].coordnames)
            par_name = None
            if not self.df.empty:
                # Param-name guess: first ICP from continuation, else 'a'
                par_name = sv_names[0]  # placeholder; overridden below
            # Pick the parameter that varies in PO branch (matches df.param)
            try:
                p0 = float(sols[0].PAR[list(sols[0].PAR.coordnames)[0]])
                pN = float(sols[-1].PAR[list(sols[-1].PAR.coordnames)[0]])
                if abs(pN - p0) > 1e-9:
                    par_name = list(sols[0].PAR.coordnames)[0]
            except Exception:
                pass
            # Better: use df['param'] alignment via solution index
            indices = np.unique(np.round(np.linspace(0, n_sol - 1, min(n_samples, n_sol))).astype(int))
            result = []
            for idx in indices:
                s = sols[int(idx)]
                arr = np.array(s.coordarray)
                t = np.array(s.indepvararray)
                # Find param: look for any matching entry in df by step
                try:
                    p_val = None
                    for pname in s.PAR.coordnames:
                        v = float(s.PAR[pname])
                        if not self.df.empty and "param" in self.df.columns:
                            if np.any(np.abs(self.df["param"].values - v) < 1e-6):
                                p_val = v
                                break
                    if p_val is None:
                        p_val = float(s.PAR[list(s.PAR.coordnames)[0]])
                except Exception:
                    p_val = float("nan")
                d = {"param": p_val, "t": t}
                for i, sv in enumerate(sv_names):
                    if i < arr.shape[0]:
                        d[sv] = arr[i]
                result.append(d)
            return result
        except Exception:
            return []

    def plot(self, ax=None, ICS=None, VOI=None, save=None, **kwargs):
        """Render the full bifurcation diagram for this branch.

        Draws the equilibrium branch, its special points and any periodic-orbit
        envelopes (a filled min/max region, or a `max` line when only maxima are
        available), labels the axes, adds a legend and applies publication
        styling. When nested continuation produced codim-2 curves, dispatches to
        the codim-2 renderer instead.

        Args:
            ax: Existing axes to draw on; a new figure is created when `None`.
            ICS: Label for the x-axis (the free/continuation parameter);
                defaults to `self.ICS` or `"param"`.
            VOI: Variable of interest plotted on the y-axis; resolved to a
                default branch column when `None`.
            save: File path to write the figure to (at 500 dpi); skipped when
                `None`.
            **kwargs: Style overrides forwarded to `plot_branch` and
                `plot_special_points`.

        Returns:
            The matplotlib axes the diagram was drawn on.
        """
        # Auto-dispatch to codim-2 rendering when nested continuation produced codim-2 curves.
        if getattr(self, "codim2_curves", None):
            return self._plot_codim2(ax=ax, ICS=ICS, save=save, **kwargs)
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
                    v_min = (po_br.df[min_col].values
                             if min_col in po_br.df.columns else None)
                    clabels = ax.get_legend_handles_labels()[1]
                    if v_min is not None:
                        ax.fill_between(
                            params,
                            v_min,
                            v_max,
                            color=_C["po_env"],
                            alpha=0.15,
                            label=("PO envelope" if "PO envelope" not in clabels else None),
                            zorder=0,
                        )
                        ax.plot(params, v_min, "-", color=_C["po_line"], linewidth=0.8, alpha=0.6)
                    else:
                        ax.plot(
                            params, v_max, "-",
                            color=_C["po_line"], linewidth=1.2,
                            label=("PO max" if "PO max" not in clabels else None),
                            zorder=0,
                        )
                        continue
                    ax.plot(params, v_max, "-", color=_C["po_line"], linewidth=0.8, alpha=0.6)
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

    def _plot_codim2(self, ax=None, ICS=None, ICS2=None, save=None, **kwargs):
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
                    cols_avail = [c for c in (max_col, min_col)
                                  if c in po_br.df.columns]
                    for col in cols_avail:
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

            # Per-orbit stability → split surface into stable/unstable
            # contiguous segments (stable → plot_surface, unstable →
            # plot_wireframe), mirroring ContinuationPlot.PlotBifCurve.
            if "stable" in po_br.df.columns and len(po_br.df) >= n_orb:
                # Sample stability at the same indices used for orbits
                sample_idx = np.unique(np.round(np.linspace(0, len(po_br.df) - 1, n_orb)).astype(int))
                stab_arr = po_br.df["stable"].values[sample_idx[:n_orb]].astype(bool)
            else:
                stab_arr = np.ones(n_orb, dtype=bool)

            # Contiguous stability segments
            seg_breaks = np.concatenate([[0], np.where(np.diff(stab_arr.astype(int)) != 0)[0] + 1, [n_orb]])
            for s_start, s_end in zip(seg_breaks[:-1], seg_breaks[1:]):
                if s_end - s_start < 2:
                    continue
                seg_stable = bool(stab_arr[s_start])
                Xs, Ys, Zs = X[s_start:s_end], Y[s_start:s_end], Z[s_start:s_end]
                slc_style = get_bif_style("SLC" if seg_stable else "ULC")
                color = slc_style["color"] if slc_style else _C["po_line"]
                if seg_stable:
                    ax.plot_surface(
                        Xs, Ys, Zs,
                        alpha=0.25, color=color, edgecolor="none",
                        shade=True, zorder=3, rasterized=True,
                    )
                else:
                    ax.plot_wireframe(
                        Xs, Ys, Zs,
                        alpha=0.5, color=color, linewidth=0.6,
                        rstride=max(1, (s_end - s_start) // 6), cstride=10,
                        zorder=3, rasterized=True,
                    )

            # Wireframe orbit loops (every ~5th) for visual cue
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
                    cols_avail = [c for c in (max_col, min_col)
                                  if c in po_br.df.columns]
                    for col in cols_avail:
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

        ics2_label = ICS2 or c2_ics or sv2 or "param2"
        ax.set_xlabel(ics_label, fontsize=8, labelpad=6)
        ax.set_ylabel(ics2_label, fontsize=8, labelpad=6)
        ax.set_zlabel(VOI, fontsize=8, labelpad=6)
        ax.view_init(elev=25, azim=-55)
        ax.legend(framealpha=0, edgecolor="none", fontsize=7, loc="upper left", borderpad=0.5)
        if save:
            fig.savefig(save, dpi=500, bbox_inches="tight")
        return ax

    def _lc_ring_at_param(self, val, VOI, sv2, y_center, n_theta=80):
        """Sample a closed periodic-orbit loop at ``param=val`` for 3D overlay.

        Returns ``(X, Y, Z)`` arrays (length ``n_theta + 1``, last point
        repeats the first) tracing the limit cycle in the same coordinates
        used by :meth:`plot_3d` (``x = param``, ``y = y_center + sv2_disp``,
        ``z = VOI``), or ``None`` if no PO branch covers ``val``.
        """
        po_list = getattr(self, "periodic_orbits", None) or []
        if not po_list:
            return None

        for po_br in po_list:
            df = po_br.df
            if df.empty or "param" not in df.columns:
                continue
            params = df["param"].values
            if val < params.min() or val > params.max():
                continue

            # Try full orbit shape first (Julia BifurcationKit / AUTO bd()).
            # Reuse the EXACT same Y-scaling that plot_3d applies to the
            # PO tube so the ring sits on the tube's surface, not next to it.
            orbits = po_br.extract_orbit_meshes(n_samples=max(8, n_theta // 4))
            if orbits and sv2 and VOI in orbits[0] and sv2 in orbits[0]:
                o_params = np.array([o["param"] for o in orbits])
                j = int(np.abs(o_params - val).argmin())
                orb = orbits[j]
                t = orb["t"]
                t_norm = (t - t[0]) / (t[-1] - t[0])
                t_uni = np.linspace(0, 1, n_theta, endpoint=False)
                z = np.interp(t_uni, t_norm, orb[VOI])
                w = np.interp(t_uni, t_norm, orb[sv2])

                # Replicate plot_3d's W-scaling: collect all sv2 samples to
                # get the global W_range and use abs(y_center)*0.5 (or 5.0)
                # as the y_range proxy when no codim-2 curves exist.
                all_w = []
                for o in orbits:
                    if sv2 in o:
                        all_w.append(np.asarray(o[sv2]))
                W_all = np.concatenate(all_w) if all_w else w
                W_range_all = float(W_all.max() - W_all.min())
                y_range = abs(y_center) * 0.5 or 5.0
                if W_range_all > 0:
                    w_disp = (w - W_all.mean()) / W_range_all * y_range * 0.15
                else:
                    w_disp = np.zeros_like(w)

                X = np.full(n_theta + 1, val)
                Y = np.append(y_center + w_disp, y_center + w_disp[0])
                Z = np.append(z, z[0])
                return X, Y, Z

            # Fallback: AUTO-style envelope. Reconstruct an ellipse from
            # max_<sv> radii in the (sv2, VOI) plane around the equilibrium.
            max_voi = f"max_{VOI}"
            if max_voi not in df.columns:
                continue
            r_voi = float(np.interp(val, params, df[max_voi].values))
            r_sv2 = 0.0
            if sv2 and f"max_{sv2}" in df.columns:
                r_sv2 = float(np.interp(val, params, df[f"max_{sv2}"].values))

            # Center on equilibrium at this param (z) and y_center (y).
            z_eq = 0.0
            if not self.df.empty and "param" in self.df.columns:
                eq_params = self.df["param"].values
                voi_eq = compute_voi(self.df, VOI,
                                     state_var_index=self.state_var_index).values
                z_eq = float(np.interp(val, eq_params, voi_eq))

            theta = np.linspace(0, 2 * np.pi, n_theta + 1)
            X = np.full_like(theta, val)
            Y = y_center + r_sv2 * np.sin(theta)
            Z = z_eq + r_voi * np.cos(theta)
            return X, Y, Z

        return None

    def _lc_orbit_at_param(self, val, x_var, y_var, n_theta=240):
        """Sample a periodic orbit in phase-plane coordinates."""
        po_list = getattr(self, "periodic_orbits", None) or []
        if not po_list:
            return None

        for po_br in po_list:
            df = po_br.df
            if df.empty or "param" not in df.columns:
                continue
            params = df["param"].values
            if val < params.min() or val > params.max():
                continue

            orbits = po_br.extract_orbit_meshes(n_samples=max(8, n_theta // 4))
            if orbits and x_var in orbits[0] and y_var in orbits[0]:
                o_params = np.array([o["param"] for o in orbits])
                j = int(np.abs(o_params - val).argmin())
                orb = orbits[j]
                t = np.asarray(orb["t"])
                if len(t) < 2:
                    continue
                t_norm = (t - t[0]) / (t[-1] - t[0])
                t_uni = np.linspace(0, 1, n_theta, endpoint=False)
                x = np.interp(t_uni, t_norm, orb[x_var])
                y = np.interp(t_uni, t_norm, orb[y_var])
                return np.r_[x, x[0]], np.r_[y, y[0]]

            max_x = f"max_{x_var}"
            if max_x not in df.columns:
                continue
            max_y = f"max_{y_var}"
            r_x = float(np.interp(val, params, df[max_x].values))
            r_y = float(np.interp(val, params, df[max_y].values)) if max_y in df.columns else r_x

            x_eq = 0.0
            y_eq = 0.0
            if not self.df.empty and "param" in self.df.columns:
                eq_params = self.df["param"].values
                x_eq_vals = compute_voi(self.df, x_var, state_var_index=self.state_var_index).values
                y_eq_vals = compute_voi(self.df, y_var, state_var_index=self.state_var_index).values
                x_eq = float(np.interp(val, eq_params, x_eq_vals))
                y_eq = float(np.interp(val, eq_params, y_eq_vals))

            theta = np.linspace(0, 2 * np.pi, n_theta + 1)
            return x_eq + r_x * np.cos(theta), y_eq + r_y * np.sin(theta)

        return None

    def animate(self, dynamics, parameter, values, *dims,
                kind="phaseplane", VOI=None, interval=80,
                figsize=(11, 4.8), title_fmt="{name} = {value:+.2f}",
                marker_kwargs=None, simulation=False,
                simulation_duration=200.0, simulation_dt=0.01,
                simulation_backend="tvboptim",
                simulation_initial_values=None,
                trajectory_kwargs=None, show_periodic_orbit=True,
                orbit_kwargs=None, **plot_kwargs):
        """Animate ``dynamics`` alongside this 3D bifurcation diagram.

        For each value of ``parameter`` a left panel re-renders a
        ``Dynamics`` plot (``kind`` forwarded to :func:`plot_dynamics`,
        defaults to ``"phaseplane"``), while a right panel shows
        :meth:`plot_3d` once with a moving marker that tracks the
        current parameter value on the equilibrium backbone.

        Parameters
        ----------
        dynamics : Dynamics
            Model whose parameter is being swept.
        parameter : str
            Parameter name (must exist in ``dynamics.parameters`` and
            match this result's continuation parameter).
        values : sequence of float
            Parameter values, one per frame.
        *dims, **plot_kwargs
            Forwarded to :func:`tvbo.plot.dynamics.plot_dynamics`.
        kind : str
            Plot kind for the left panel (default ``"phaseplane"``).
        VOI : str, optional
            State variable plotted on the z-axis of the 3D diagram.
        interval : int
            Delay between frames in ms.
        figsize : (float, float)
        title_fmt : str
            Title format with ``{name}`` and ``{value}`` placeholders.
        marker_kwargs : dict, optional
            Style overrides for the moving marker.
        simulation : bool
            If true, overlay a trajectory computed with
            :class:`tvbo.classes.experiment.SimulationExperiment` for each
            frame. This keeps animated trajectories on the same backend path
            as full experiments instead of using ``Dynamics.run``.
        simulation_duration, simulation_dt : float
            Integration settings used when ``simulation`` is true.
        simulation_backend : str
            Backend passed to ``SimulationExperiment.run``.
        simulation_initial_values : dict or callable, optional
            State-variable initial values used for each simulated frame. If a
            callable is supplied, it receives the current parameter value and
            returns a mapping for that frame. Returning ``None`` skips the
            simulated trajectory for that frame.
        trajectory_kwargs : dict, optional
            Style overrides for simulated trajectory overlays.
        show_periodic_orbit : bool
            Draw the current periodic orbit in phase-plane coordinates when a
            periodic-orbit ring is also available in the bifurcation panel.
        orbit_kwargs : dict, optional
            Style overrides for the phase-plane periodic-orbit circle.

        Returns
        -------
        matplotlib.animation.FuncAnimation
        """
        import copy
        from matplotlib.animation import FuncAnimation
        from tvbo.plot.dynamics import plot_dynamics

        values = list(values)

        if parameter not in dynamics.parameters:
            raise ValueError(
                f"parameter {parameter!r} not in dynamics "
                f"(available: {list(dynamics.parameters)})"
            )

        dyn = copy.deepcopy(dynamics)
        VOI = self._resolve_voi(VOI)
        phase_dims = [str(d) for d in dims[:2]]
        if not phase_dims and len(dynamics.state_variables) >= 2:
            phase_dims = list(dynamics.state_variables)[:2]

        trajectory_data = []
        if simulation:
            from tvbo.classes.experiment import SimulationExperiment

            for val in values:
                frame_initial_values = (
                    simulation_initial_values(float(val))
                    if callable(simulation_initial_values)
                    else simulation_initial_values
                )
                if frame_initial_values is None:
                    trajectory_data.append(None)
                    continue

                sim_dyn = copy.deepcopy(dynamics)
                sim_dyn.parameters[parameter].value = float(val)
                if frame_initial_values:
                    for name, value in frame_initial_values.items():
                        sim_dyn.state_variables[name].initial_value = float(value)
                exp = SimulationExperiment(dynamics=sim_dyn)
                exp.integration.duration = simulation_duration
                exp.integration.step_size = simulation_dt
                res = exp.run(simulation_backend).integration
                trajectory_data.append({
                    name: np.asarray(res.data.sel(variable=name)).squeeze()
                    for name in phase_dims
                })

        fig = plt.figure(figsize=figsize)
        ax_left = fig.add_subplot(1, 2, 1)
        ax_right = fig.add_subplot(1, 2, 2, projection="3d")
        self.plot_3d(ax=ax_right, VOI=VOI)

        # Backbone position (param, c2_default, voi_eq) for marker lookup
        params = self.df["param"].values if not self.df.empty else np.array([])
        voi_eq = (compute_voi(self.df, VOI, state_var_index=self.state_var_index).values
                  if not self.df.empty else np.array([]))
        # y-coordinate matches plot_3d backbone (median of codim-2 if any, else 0)
        y_bb = 0.0
        if hasattr(ax_right, "lines") and ax_right.lines:
            ydata = ax_right.lines[0].get_ydata()
            if len(ydata):
                y_bb = float(ydata[0])

        # Resolve "other" state variable name (sv2) for LC ring rendering
        sv_names = list(self.state_var_index.keys()) if self.state_var_index else []
        sv2 = next((s for s in sv_names if s != VOI), None)

        mk = dict(marker="o", s=80, color="red", edgecolors="white",
                  linewidths=1.0, zorder=20)
        if marker_kwargs:
            mk.update(marker_kwargs)
        marker = ax_right.scatter([params[0] if len(params) else 0.0],
                                  [y_bb],
                                  [voi_eq[0] if len(voi_eq) else 0.0], **mk)
        ring_artist = [None]  # boxed so the closure can rebind it

        traj_style = dict(color="red", lw=1.2, alpha=0.9, zorder=9)
        if trajectory_kwargs:
            traj_style.update(trajectory_kwargs)
        orbit_style = dict(color="red", lw=2.0, alpha=0.95, zorder=11)
        if orbit_kwargs:
            orbit_style.update(orbit_kwargs)

        def _update(frame_idx):
            ax_left.clear()
            val = float(values[frame_idx])
            dyn.parameters[parameter].value = val
            plot_dynamics(dyn, *dims, kind=kind, ax=ax_left, **plot_kwargs)
            if simulation and len(phase_dims) >= 2:
                traj = trajectory_data[frame_idx]
                if traj is not None:
                    ax_left.plot(traj[phase_dims[0]], traj[phase_dims[1]], **traj_style)
            ax_left.set_title(title_fmt.format(name=parameter, value=val),
                              color="red")
            if len(params):
                i = int(np.abs(params - val).argmin())
                z_val = float(voi_eq[i]) if len(voi_eq) else 0.0
                marker._offsets3d = ([val], [y_bb], [z_val])

            # LC ring: clear previous, draw current if val sits on a PO branch
            if ring_artist[0] is not None:
                try:
                    ring_artist[0].remove()
                except Exception:
                    pass
                ring_artist[0] = None
            ring = self._lc_ring_at_param(val, VOI, sv2, y_bb)
            if ring is not None:
                X, Y, Z = ring
                ring_artist[0], = ax_right.plot(
                    X, Y, Z, "-", color="red", linewidth=2.0,
                    zorder=21, alpha=0.95,
                )
                if show_periodic_orbit and len(phase_dims) >= 2:
                    orbit = self._lc_orbit_at_param(val, phase_dims[0], phase_dims[1])
                    if orbit is not None:
                        ax_left.plot(*orbit, **orbit_style)
            return [ax_left, marker]

        anim = FuncAnimation(fig, _update, frames=len(values),
                             interval=interval, blit=False)
        plt.close(fig)
        return anim


# ── Backend extractors (formerly PyRates/NumCont subclass methods) ──
#
# These functions convert backend-native objects (PyCoBi ``ODESystem``,
# AUTO-07p ``bifDiag``) into the unified DataFrame schema consumed by
# :class:`BifurcationResult`. Keeping them at module level (instead of
# nested inside subclasses) makes the data flow explicit:
#
#     backend_object → _extract_<backend>_df(...) → BifurcationResult(df=...)
#
# and lets the same plotting/legend/export code apply to *every*
# backend without inheritance.

# Standard columns expected by all extractors
_BIF_STANDARD_COLS = ["param", "stable", "step", "specialpoint", "n_unstable", "n_imag"]

# AUTO label → canonical specialpoint name used by plotting layer
_AUTO_LABEL_MAP = {
    "HB": "hopf", "LP": "fold", "BP": "bp", "PD": "pd", "TR": "ns",
    "EP": "endpoint", "MX": "mx", "UZ": "uz",
    "BT": "bt", "CP": "cusp", "GH": "gh", "ZH": "zh",
}


# ── PyRates / PyCoBi extractor ──────────────────────────────────────────

def _add_pycobi_param2(ode, df, cont_name, state_var_names, icp2):
    """Append a ``param2`` column to a codim-2 DataFrame.

    Pulls the second free-parameter trajectory from PyCoBi's raw AUTO
    branch (or, on failure, its summary).
    """
    par2_col = f"PAR({icp2})"
    try:
        cont_key = ode._results_map[cont_name]
        sol = ode.auto_solutions[cont_key]
        branch = sol.data[0]
        branch_data = branch.todict()
        if par2_col in branch_data:
            p2_vals = [float(v) for v in branch_data[par2_col]]
            if len(p2_vals) >= len(df):
                df["param2"] = p2_vals[: len(df)]
            else:
                df["param2"] = np.interp(
                    np.linspace(0, 1, len(df)),
                    np.linspace(0, 1, len(p2_vals)),
                    p2_vals,
                )
    except (KeyError, IndexError, AttributeError):
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


def _extract_pycobi_df(ode, cont_name, state_var_names, icp):
    """Convert a PyCoBi continuation result into the unified DataFrame.

    Uses the full raw AUTO branch for equilibria (every continuation
    step) and falls back to ``get_summary`` -- which provides min/max
    envelopes -- for periodic-orbit branches.
    """
    auto_to_bif = {k: v for k, v in _AUTO_LABEL_MAP.items()
                   if k in {"LP", "HB", "BP", "PD", "TR", "BT", "CP", "GH", "ZH"}}

    def _empty_df():
        return pd.DataFrame(columns=list(state_var_names or []) + _BIF_STANDARD_COLS)

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

    try:
        summary = ode.get_summary(cont_name)
    except (KeyError, IndexError, AttributeError):
        summary = None

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

    is_po = False
    if summary is not None and len(summary) > 0 and state_var_names:
        sv0 = state_var_names[0]
        s_cols = summary.columns
        is_po = ((sv0, 0) in s_cols and (sv0, 1) in s_cols) or (("U(1)", 0) in s_cols and ("U(1)", 1) in s_cols)

    par_col = f"PAR({icp})"

    def _col(name, cols):
        if (name, "") in cols:
            return (name, "")
        if (name, 0) in cols:
            return (name, 0)
        if isinstance(cols, pd.MultiIndex):
            return None
        if name in cols:
            return name
        return None

    # ── Periodic-orbit path: summary-based with min/max ──
    if is_po or (n_steps == 0 and summary is not None):
        if summary is None or len(summary) == 0:
            return _empty_df()
        cols = summary.columns
        s_par_col = None
        translated = ode._var_map_inv.get(par_col, ode._var_map_inv.get(icp, None))
        for cand in [translated, par_col]:
            if cand:
                resolved = _col(cand, cols)
                if resolved is not None:
                    s_par_col = resolved
                    break

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
            row["stable"] = _parse_stability(rd[stab_col]) if stab_col is not None else True
            row["step"] = row_idx
            bif_type = None
            if bif_col is not None:
                bif_type = auto_to_bif.get(str(rd[bif_col]).strip())
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

    # PyRates-generated branches key columns by the variable NAME (V, W, I_);
    # hand-written fortran keys them by U(i)/PAR(i). Accept whichever is present.
    par_name = ode._var_map_inv.get(par_col)
    par_key = next((k for k in (par_name, par_col) if k and k in branch_data), None)
    sv_keys = {}
    if state_var_names:
        for i, sv_name in enumerate(state_var_names):
            sv_keys[sv_name] = next(
                (k for k in (sv_name, f"U({i + 1})") if k in branch_data), None
            )

    rows = []
    for step in range(n_steps):
        row = {}
        for sv_name, key in sv_keys.items():
            if key is not None:
                row[sv_name] = float(branch_data[key][step])
        row["param"] = float(branch_data[par_key][step]) if par_key else np.nan
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
                bif_type = auto_to_bif.get(str(rd[bif_col]).strip())
            stab_data.append((p_val, stab, bif_type))
        if stab_data and s_par_col:
            lp = np.array([s[0] for s in stab_data])
            ls = np.array([s[1] for s in stab_data])
            bp = df["param"].values
            nearest = np.abs(bp[:, None] - lp[None, :]).argmin(axis=1)
            df["stable"] = ls[nearest]
            for pv, _, bt in stab_data:
                if bt is None:
                    continue
                si = int(np.abs(bp - pv).argmin())
                df.loc[si, "specialpoint"] = bt
    df["stable"] = df["stable"].astype(bool)
    return df


# ── AUTO-07p (numcont) extractor ────────────────────────────────────────

def _auto_branch_to_df(branch, sv_names, fp_name):
    """Convert one AUTO ``branch`` into the unified DataFrame."""
    coordnames = list(branch.coordnames)
    coordarray = np.asarray(branch.coordarray)
    if coordarray.size == 0 or coordarray.ndim != 2:
        return pd.DataFrame()
    n_rows = coordarray.shape[1]

    df = pd.DataFrame({cn: coordarray[i] for i, cn in enumerate(coordnames)})
    df["step"] = np.arange(n_rows)

    # Resolve PAR(i) → name via parsed constants header (branch.c)
    c = getattr(branch, "c", {}) or {}
    parnames_raw = c.get("parnames", {}) if isinstance(c, dict) else {}
    pname_by_idx = {}
    items_iter = (parnames_raw.items() if isinstance(parnames_raw, dict)
                  else parnames_raw if isinstance(parnames_raw, (list, tuple))
                  else [])
    for k, v in items_iter:
        try:
            pname_by_idx[int(k)] = str(v)
        except (TypeError, ValueError):
            pass

    for col in list(df.columns):
        if col.startswith("PAR(") and col.endswith(")"):
            try:
                idx = int(col[4:-1])
            except ValueError:
                continue
            pname = pname_by_idx.get(idx)
            if pname:
                df[pname] = df[col]
                if pname == fp_name:
                    df["param"] = df[col]
    if "param" not in df.columns:
        if "PAR(1)" in df.columns:
            df["param"] = df["PAR(1)"]
        elif fp_name and fp_name in df.columns:
            df["param"] = df[fp_name]

    for i, sv in enumerate(sv_names, start=1):
        ucol = f"U({i})"
        if ucol in df.columns:
            df[sv] = df[ucol]

    # Periodic-orbit branches: AUTO writes ``MAX <sv>`` / ``MIN <sv>``
    # (or ``MAX U(i)``) into the b-file. Map those to the canonical
    # ``max_<sv>`` / ``min_<sv>`` columns the plot layer expects, and
    # fall back to ``MAX`` as the SV column when no U(i) was emitted.
    for i, sv in enumerate(sv_names, start=1):
        for prefix, target in (("MAX", "max"), ("MIN", "min")):
            for src in (f"{prefix} {sv}", f"{prefix} U({i})"):
                if src in df.columns:
                    df[f"{target}_{sv}"] = df[src]
                    if sv not in df.columns:
                        df[sv] = df[src]
                    break

    # Stability — branch.stability() returns signed segment endpoints
    stable = np.ones(n_rows, dtype=bool)
    try:
        prev = 0
        for s in branch.stability():
            end = abs(int(s))
            stable[prev:end] = int(s) < 0  # negative => stable segment
            prev = end
    except Exception:
        pass
    df["stable"] = stable

    # Special points from branch.labels
    df["specialpoint"] = ""
    labels = getattr(branch, "labels", None)
    if labels is not None:
        by_label = getattr(labels, "by_label", None)
        label_iter = (by_label.items() if hasattr(by_label, "items")
                      else labels.items() if hasattr(labels, "items") else [])
        for label, points in label_iter:
            canon = _AUTO_LABEL_MAP.get(str(label).upper())
            if not canon:
                continue
            point_items = points.items() if isinstance(points, dict) else points
            for idx, _payload in point_items:
                try:
                    idx = int(idx)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx < n_rows:
                    cur = df.at[idx, "specialpoint"]
                    df.at[idx, "specialpoint"] = canon if not cur else f"{cur},{canon}"
    return df


def _extract_auto_df(bd, sv_names, fp_name):
    """Concatenate every branch in an AUTO ``bifDiag`` into one DataFrame.

    Adds a ``branch_id`` column so plotting can keep sub-branches distinct.
    For periodic-orbit branches, augments missing ``min_<sv>`` columns by
    scanning the orbit solutions in ``bd()`` (AUTO does not write MIN to
    the b-file).
    """
    frames = []
    for bid, br in enumerate(bd):
        df_br = _auto_branch_to_df(br, sv_names, fp_name)
        if not df_br.empty:
            df_br["branch_id"] = bid
            frames.append(df_br)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    # Fill missing min_<sv> by sampling orbit solutions when this is a PO bd.
    needs_min = [sv for sv in sv_names
                 if f"max_{sv}" in df.columns and f"min_{sv}" not in df.columns]
    if needs_min:
        try:
            sols = bd()
            n = len(sols)
            if n:
                # Map solution index → row in df by matching param values
                params = df["param"].values if "param" in df.columns else None
                for sv in needs_min:
                    df[f"min_{sv}"] = np.full(len(df), np.nan)
                for sidx in range(n):
                    s = sols[sidx]
                    arr = np.array(s.coordarray)
                    try:
                        p_val = float(s.PAR[fp_name]) if fp_name else float(s.PAR[
                            list(s.PAR.coordnames)[0]])
                    except Exception:
                        continue
                    if params is None or len(params) == 0:
                        continue
                    j = int(np.abs(params - p_val).argmin())
                    for i, sv in enumerate(sv_names):
                        if i < arr.shape[0] and f"min_{sv}" in df.columns:
                            df.at[j, f"min_{sv}"] = float(arr[i].min())
                # Backfill with -max as a last resort for symmetric orbits
                for sv in needs_min:
                    mask = df[f"min_{sv}"].isna()
                    if mask.any() and f"max_{sv}" in df.columns:
                        df.loc[mask, f"min_{sv}"] = -df.loc[mask, f"max_{sv}"]
        except Exception:
            pass
    return df


# Stub kept for the deprecated dummy class definition below.

__all__ = [
    "BifurcationResult",
    "BIF_STYLES",
    "canonical_ty",
    "get_bif_style",
    "resolve_coord",
    "resolve_coords",
    "PO_REDUCTIONS",
    "CurvePicker",
]


# ── Interactive curve picker (port of ContinuationPlot.CurvePicker) ──

class CurvePicker:
    """Click any branch line to inspect the underlying point.

    Activated via ``BifurcationResult.enable_picker(ax, callback=...)``.
    Each branch line drawn by ``plot_branch`` carries ``picker=True`` so
    matplotlib raises a ``pick_event`` on click; the picker resolves the
    nearest df row and forwards it to ``callback(result, row_index)``.
    """

    def __init__(self, fig, result, callback=None):
        self.fig = fig
        self.result = result
        self.callback = callback or self._default_print
        self.cid = fig.canvas.mpl_connect("pick_event", self._on_pick)
        self._marker = None

    def _default_print(self, result, idx):
        row = result.df.iloc[idx]
        sv_cols = [c for c in result.state_var_index] if result.state_var_index else []
        info = ", ".join(f"{c}={row[c]:.4g}" for c in (["param"] + sv_cols) if c in row)
        sp = row.get("specialpoint", "")
        print(f"[idx={idx}] {info}{(' [' + sp + ']') if sp else ''}")

    def _on_pick(self, event):
        if not hasattr(event, "mouseevent") or event.mouseevent.button != 1:
            return
        line = event.artist
        ind = event.ind[0] if len(event.ind) else 0
        x, y = line.get_data()
        ax = line.axes
        if self._marker is not None:
            try:
                self._marker.remove()
            except Exception:
                pass
        self._marker, = ax.plot([x[ind]], [y[ind]], "o",
                                color="C0", zorder=100, ms=8,
                                markeredgecolor="white", markeredgewidth=1)
        self.fig.canvas.draw_idle()
        # Resolve nearest df row by param value
        try:
            df = self.result.df
            row_idx = int(np.abs(df["param"].values - x[ind]).argmin())
            self.callback(self.result, row_idx)
        except Exception as exc:
            print(f"[CurvePicker] callback failed: {exc}")

    def disconnect(self):
        """Detach the pick-event handler so clicks are no longer captured.

        Safe to call more than once; the connection id is cleared after the
        first disconnect.
        """
        if self.cid is not None:
            self.fig.canvas.mpl_disconnect(self.cid)
            self.cid = None


def _enable_picker(self, ax=None, callback=None):
    """Attach a ``CurvePicker`` to ``ax``'s figure (or current figure)."""
    fig = ax.get_figure() if ax is not None else plt.gcf()
    return CurvePicker(fig, self, callback=callback)


BifurcationResult.enable_picker = _enable_picker
