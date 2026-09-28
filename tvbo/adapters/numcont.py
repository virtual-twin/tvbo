"""Self-contained AUTO-07p (numcont) backend adapter for SimulationExperiment.

This adapter does NOT depend on any external `numcont` package. It uses the `auto-07p` Python bindings directly (`auto.run`, `auto.sv`, `auto.loadbd`, `auto.merge`) and the Mako template at ``tvbo/templates/numcont/tvbo-auto7p.py.mako`` to emit the model `.f90` file consumed by AUTO.

Requires the ``AUTO_DIR`` environment variable to point at an installed auto-07p tree (validated via :func:`tvbo.utils.auto.check_auto_dir`).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import TYPE_CHECKING

import numpy as np

from tvbo.adapters.base import ContinuationAdapter

if TYPE_CHECKING:
    pass


# AUTO reserves PAR(11)=PERIOD and PAR(12)=ANGLE; user params take 1..10 then 13..NPAR.
_RESERVED_LO = 11
_RESERVED_HI = 12


def _auto_par_index(i: int) -> int:
    """Map zero-based user-parameter index to AUTO PAR(.) index."""
    return i + 1 if i + 1 <= 10 else i + 3


def _build_parnames(model) -> dict[int, str]:
    """Build AUTO parnames dict, matching the f90 template's PAR layout."""
    pn = {}
    for i, p in enumerate(model.parameters.values()):
        pn[_auto_par_index(i)] = p.name
    pn[_RESERVED_LO] = "PERIOD"
    pn[_RESERVED_HI] = "ANGLE"
    return pn


def _build_unames(model) -> dict[int, str]:
    return {i + 1: sv.name for i, sv in enumerate(model.state_variables.values())}


def _schema_initial_values(model) -> np.ndarray:
    """Initial state from model state-variable defaults.

    Reads ``StateVariable.initial_value`` (the canonical schema field, per ``schema/tvbo_datamodel.yaml:1346-1348``) with ``StateVariable.value`` as a legacy fallback for older models. Defaults to 0.0 when neither is set.
    """
    n = len(model.state_variables)
    x0 = np.zeros(n)
    for i, sv in enumerate(model.state_variables.values()):
        v = getattr(sv, "initial_value", None)
        if v is None:
            v = getattr(sv, "value", None)
        if v is not None:
            x0[i] = float(v)
    return x0


def _warmup_to_steady_state(model, cont) -> np.ndarray:
    """Time-integrate the dynamics to produce a starting state for AUTO.

    Honors ``cont.initial_state.method`` (default ``time_integration``):

    - ``time_integration`` (default): run ``Dynamics.run(format='python')`` for ``initial_state.duration`` time units and return the final state.
    - ``given``: return the schema defaults verbatim (no integration).
    - other methods: fall back to schema defaults (not yet implemented).
    """
    x0 = _schema_initial_values(model)

    iss = getattr(cont, "initial_state", None) if cont else None
    method = getattr(iss, "method", None)
    method_str = str(method) if method is not None else "time_integration"

    if method_str == "given":
        return x0

    duration = float(getattr(iss, "duration", None) or 2000.0)
    ts = model.run(format="python", u_0=x0, duration=duration, save=False, verbose=0)
    # TimeSeries data is (T, n_sv, 1, 1)
    return np.asarray(ts.data[-1, :, 0, 0], dtype=float)


def _initial_state_dict(model, cont) -> dict[int, float]:
    """Build the U={1: x0, ...} initial-condition dict for AUTO."""
    x0 = _warmup_to_steady_state(model, cont)
    return {i + 1: float(x0[i]) for i in range(len(x0))}


def _param_values(model) -> dict[str, float]:
    """Default parameter values keyed by name (skips None)."""
    from tvbo.utils import is_array_valued

    out = {}
    for p in model.parameters.values():
        if p.value is not None and not is_array_valued(p.value):
            out[p.name] = float(p.value)
    return out


def _free_parameter(cont, model):
    """Return (name, p_min, p_max) for the primary free parameter."""
    fp_dict = getattr(cont, "free_parameters", None) if cont else None
    if fp_dict:
        fp_first = next(iter(fp_dict.values())) if isinstance(fp_dict, dict) else fp_dict[0]
        name = str(fp_first.name)
        dom = fp_first.domain or model.parameters[name].domain
    else:
        name = next(iter(model.parameters.keys()))
        dom = model.parameters[name].domain
    p_min = float(dom.lo) if dom and dom.lo is not None else -10.0
    p_max = float(dom.hi) if dom and dom.hi is not None else 10.0
    return name, p_min, p_max


# The declared continuation fields each AUTO constant is filled from when `cont.parameters` names no override.
_SCHEMA_CONSTANTS = {"NMX": "max_steps", "DS": "ds", "DSMAX": "ds_max", "DSMIN": "ds_min"}


def _cont_par(cont, key, default=None):
    """The AUTO constant *key*: an explicit ``cont.parameters`` entry first, then the schema field it corresponds to, then *default*."""
    if cont is None:
        return default
    params = getattr(cont, "parameters", None)
    if params and isinstance(params, dict):
        p = params.get(key)
        if p and getattr(p, "value", None) is not None:
            return p.value
    declared = getattr(cont, _SCHEMA_CONSTANTS.get(key, ""), None)
    return default if declared is None else declared


def _po_par(cont, po_cont, key, default):
    """One AUTO constant for the periodic-orbit branch: the parent's ``<KEY>_PO`` override, else the nested Hopf branch's own declaration, else *default*.

    Presence decides, not truthiness: a declared ``DS_PO: 0`` is a bad step size the caller should see rejected, not a silent fall-through to the branch below it.
    """
    override = _cont_par(cont, f"{key}_PO", None)
    return override if override is not None else _cont_par(po_cont, key, default)


def _branches(cont) -> dict:
    """*cont*'s branch switches keyed by name, however the declaration held them."""
    branches = getattr(cont, "branches", None) or {}
    if isinstance(branches, dict):
        return branches
    return {getattr(bs, "name", f"branch_{i}"): bs for i, bs in enumerate(branches)}


def _declared_sources(cont) -> list:
    """``(name, branch, kind, index)`` for every branch of *cont* that declares a source point, read by `ContinuationAdapter.source_point`.

    A branch declaring none is started by nothing here: AUTO continues periodic orbits from every Hopf point unless a periodic-orbit branch selects some.
    """
    return [
        (name, branch, *ContinuationAdapter.source_point(branch, ""))
        for name, branch in _branches(cont).items()
        if getattr(branch, "source_point", None)
    ]


def _po_branch(cont):
    """The periodic-orbit :class:`BranchSwitch` *cont* declares — its first one-parameter branch from a Hopf point — and the Hopf index it selects, or ``(None, None)``."""
    return next(
        (
            (branch, index)
            for _, branch, kind, index in _declared_sources(cont)
            if kind == "HB" and not ContinuationAdapter.is_codim2(branch, cont)
        ),
        (None, None),
    )


def _n_labelled(bundle, label: str) -> int:
    """How many special points labelled *label* the AUTO *bundle* carries across all its branches, which is the range AUTO's 1-based ordinals (``HB2``) address."""
    return sum(len(branch.labels.by_label.get(label, {}) or {}) for branch in bundle)


def _orbit_profiles(solutions, sv_names, n_phase=101):
    """Every periodic solution resampled onto one shared phase grid, as ``[n_steps, n_phase, n_vars]``.

    AUTO meshes each orbit adaptively, so the raw profiles have different lengths and cannot be stacked; interpolating each onto the same normalised phase makes them one array the result container can store.
    """
    import numpy as np

    grid = np.linspace(0.0, 1.0, n_phase)
    profiles = []
    for solution in solutions:
        values = np.asarray(solution.coordarray, dtype=float)
        names = list(solution.coordnames)
        if values.ndim != 2 or not all(sv in names for sv in sv_names):
            return None
        phase = np.asarray(solution.indepvararray, dtype=float)
        if phase.size == 0 or phase.size != values.shape[1]:
            return None
        span = phase[-1] - phase[0]
        phase = (phase - phase[0]) / span if span else np.linspace(0.0, 1.0, phase.size)
        profiles.append(np.column_stack([np.interp(grid, phase, values[names.index(sv)]) for sv in sv_names]))
    return np.asarray(profiles) if profiles else None


class NumContAdapter(ContinuationAdapter):
    """Adapter for bifurcation analysis via AUTO-07p (no external deps).

    Renders the AUTO-07p Fortran (.f90) model source from `TEMPLATE`, with the shared `(model, continuation)` context.
    """

    TEMPLATE = "tvbo-auto7p.py.mako"

    FUNC_ARGUMENTS = ("ndim", "u", "icp", "par", "ijac", "f", "dfdu", "dfdp")
    """The arguments of AUTO's ``FUNC`` subroutine, in scope wherever the emitted model's symbols are, so a symbol spelled like one is renamed (`ContinuationAdapter.fortran_names`)."""

    @staticmethod
    def _prepare_context(model, continuation, **kwargs) -> dict:
        """The template context for *continuation* on *model*: the shared pair, the Fortran name of every symbol the template declares (``replace``), and the coupling inputs a single node zeroes (``coupling_zero``)."""
        context = ContinuationAdapter._prepare_context(model, continuation, **kwargs)
        emitted = [
            *model.state_variables,
            *model.parameters,
            *(model.in_dependency_order("derived_variables") if model.derived_variables else ()),
            *(model.in_dependency_order("derived_parameters") if model.derived_parameters else ()),
        ]
        context["replace"] = ContinuationAdapter.fortran_names(emitted, reserved=NumContAdapter.FUNC_ARGUMENTS)
        context["coupling_zero"] = list(model.coupling_inputs or ())
        return context

    # ── Public API ───────────────────────────────────────────────────────

    def run_one(self, model, cont, cont_name, **kwargs):
        """Continue *cont* on *model* in AUTO-07p, from its equilibrium branch through its Hopf and codim-2 branches, saving each under *cont_name*."""
        import contextlib
        import io

        from tvbo.utils.auto import check_auto_dir

        check_auto_dir()

        # AUTO-07p prints a Tkinter import warning to stdout even when plotting is unused. Suppress it.
        _buf = io.StringIO()
        with contextlib.redirect_stdout(_buf):
            import auto

        from tvbo.analysis.bifurcation import BifurcationResult

        # 1. Render f90 to a temporary working dir
        workdir = tempfile.mkdtemp(prefix=f"tvbo_numcont_{model.name}_")
        f90_path = os.path.join(workdir, "model.f90")
        with open(f90_path, "w") as fh:
            fh.write(self.render_continuation(model, cont))

        # 2. Build common AUTO arguments
        parnames = _build_parnames(model)
        unames = _build_unames(model)
        ndim = len(model.state_variables)
        npar = max(_auto_par_index(len(model.parameters) - 1), 12) if model.parameters else 12
        par_vals = _param_values(model)
        u0 = _initial_state_dict(model, cont)
        fp_name, p_min, p_max = _free_parameter(cont, model)

        # AUTO needs the source file path WITHOUT the .f90 extension
        source_stub = os.path.join(workdir, "model")

        # 3. Equilibrium (1-parameter) continuation
        kwargs_eq = dict(
            e=source_stub,
            parnames=parnames,
            unames=unames,
            NDIM=ndim,
            NPAR=npar,
            PAR=par_vals,
            U=u0,
            ICP=[fp_name],
            IPS=1,
            ISP=2,
            ISW=1,
            ILP=1,
            IADS=int(_cont_par(cont, "IADS", 1)),
            IAD=1,
            IID=0,
            EPSL=float(_cont_par(cont, "EPSL", 1e-7)),
            EPSU=float(_cont_par(cont, "EPSU", 1e-7)),
            EPSS=float(_cont_par(cont, "EPSS", 1e-5)),
            RL0=p_min,
            RL1=p_max,
            NMX=int(_cont_par(cont, "NMX", 1000)),
            NPR=int(_cont_par(cont, "NPR", 10)),
            DS=float(_cont_par(cont, "DS", 0.005)),
            DSMAX=float(_cont_par(cont, "DSMAX", 0.05)),
            DSMIN=float(_cont_par(cont, "DSMIN", 1e-6)),
            MXBF=int(_cont_par(cont, "MXBF", 50)),
        )

        cwd0 = os.getcwd()
        os.chdir(workdir)
        try:
            R_eq = auto.run(**kwargs_eq)
            # Bidirectional sweep: also continue with DS<0 and merge
            if bool(getattr(cont, "bothside", False)):
                kwargs_eq_neg = dict(kwargs_eq, DS=-kwargs_eq["DS"])
                R_eq_neg = auto.run(**kwargs_eq_neg)
                R_eq = auto.merge(R_eq + R_eq_neg)
            auto.sv(R_eq, cont_name)

            # 4. Periodic-orbit continuation from the Hopf points the periodic-orbit branch selects, every one where the continuation declares none
            po_branch, hopf_index = _po_branch(cont)
            po_cont = getattr(po_branch, "continuation", None)
            po_results = []
            po_profiles = []
            for ordinal in self.select_points(range(1, _n_labelled(R_eq, "HB") + 1), hopf_index):
                kwargs_po = dict(
                    data=R_eq(f"HB{ordinal}"),
                    EPSL=kwargs_eq["EPSL"],
                    EPSU=kwargs_eq["EPSU"],
                    EPSS=kwargs_eq["EPSS"],
                    IPS=2,
                    ISP=2,
                    ISW=1,
                    ICP=[fp_name, "PERIOD"],
                    RL0=p_min,
                    RL1=p_max,
                    NMX=int(_po_par(cont, po_cont, "NMX", 400)),
                    NPR=1,
                    DS=float(_po_par(cont, po_cont, "DS", 0.01)),
                    DSMAX=float(_po_par(cont, po_cont, "DSMAX", 0.1)),
                    DSMIN=float(_po_par(cont, po_cont, "DSMIN", 1e-6)),
                    IADS=1,
                    MXBF=int(_cont_par(cont, "MXBF", 50)),
                    IID=0,
                )
                R_po = auto.run(**kwargs_po)
                po_name = f"{cont_name}_HB{ordinal - 1}"
                auto.sv(R_po, po_name)
                po_results.append((po_name, R_po))
                po_profiles.append(_orbit_profiles(R_po(), list(model.state_variables)))

            # 5. Codim-2 fold (and Hopf, BP) continuation from BranchSwitch specs
            codim2_results = self._run_codim2_branches(
                auto=auto,
                R_eq=R_eq,
                cont=cont,
                fp_name=fp_name,
                kwargs_eq=kwargs_eq,
            )
        finally:
            os.chdir(cwd0)

        result = BifurcationResult.from_auto(
            R_eq,
            cont_name=cont_name,
            model=model,
            continuation=cont,
            ICS=fp_name,
            periodic_orbits_raw=po_results,
            codim2_raw=codim2_results,
            workdir=workdir,
        )

        # AUTO returns each periodic solution as a full profile over one period, but the branch table keeps only per-variable extrema. Carrying the profiles lets a consumer take the exact envelope of any observable, including an expression such as `y1 - y2` that no single column bounds.
        orbits = getattr(result, "periodic_orbits", None) or []
        if len(orbits) == len(po_profiles):
            for orbit, profiles in zip(orbits, po_profiles, strict=True):
                if profiles is not None and len(profiles) == len(orbit.df):
                    orbit.orbit_profiles = profiles
                    if getattr(orbit, "model", None) is None:
                        orbit.model = model
        return result

    # ── Codim-2 continuation ──────────────────────────────────────────────

    def _run_codim2_branches(self, *, auto, R_eq, cont, fp_name, kwargs_eq):
        """Run codim-2 fold/Hopf/BP continuations declared via ``cont.branches``.

        Each :class:`~tvbo.classes.continuation.BranchSwitch` that `ContinuationAdapter.is_codim2` reads as a two-parameter continuation triggers a separate AUTO restart, with ``ISW=2`` (fold/Hopf continuation) and ``ICP`` the primary *fp_name* and the branch's second parameter (`ContinuationAdapter.codim2_parameter`), from every special point its ``source_point`` selects — ``'fold:1'``, ``'hopf:all'``, ``'bp:-1'``, read by `ContinuationAdapter.source_point` — and in both directions where the branch declares ``bothside``. AUTO bounds the principal parameter by ``RL0``/``RL1``, so the primary keeps the equilibrium continuation's bounds (*kwargs_eq*) and the second parameter's domain stops the curve through ``UZSTOP``. The step is the nested continuation's where it declares one, else the equilibrium continuation's.

        Returns a list of ``(name, source_type, fp1_name, fp2_name, R_c2)`` tuples consumed by :meth:`BifurcationResult.from_auto`.
        """
        out = []
        for bname, bswitch, kind, index in _declared_sources(cont):
            fp2 = self.codim2_parameter(bswitch, cont)
            if fp2 is None:
                continue
            # AUTO labels a special point by its canonical code (LP, HB, BP) and restarts from it with ISW=2.
            label_prefix = kind
            source_type = self.SOURCE_KINDS[kind]
            sub_cont = bswitch.continuation
            fp1_name = fp_name
            fp2_name = str(fp2.name)
            dom = getattr(fp2, "domain", None)
            fp2_lo = float(dom.lo) if dom and dom.lo is not None else -10.0
            fp2_hi = float(dom.hi) if dom and dom.hi is not None else 10.0
            ordinals = self.select_points(range(1, _n_labelled(R_eq, label_prefix) + 1), index)

            for ordinal in ordinals:
                lab = ordinal  # AUTO ordinal label
                try:
                    kwargs_c2 = dict(
                        data=R_eq(f"{label_prefix}{lab}"),
                        EPSL=kwargs_eq["EPSL"],
                        EPSU=kwargs_eq["EPSU"],
                        EPSS=kwargs_eq["EPSS"],
                        IPS=1,
                        ISP=2,
                        ISW=2,
                        ILP=0,
                        ICP=[fp1_name, fp2_name],
                        RL0=kwargs_eq["RL0"],
                        RL1=kwargs_eq["RL1"],
                        UZSTOP={fp2_name: [fp2_lo, fp2_hi]},
                        NMX=int(_cont_par(sub_cont, "NMX", 400)),
                        NPR=1,
                        DS=float(_cont_par(sub_cont, "DS", 0.01)),
                        DSMAX=float(_cont_par(sub_cont, "DSMAX", kwargs_eq["DSMAX"])),
                        DSMIN=float(_cont_par(sub_cont, "DSMIN", kwargs_eq["DSMIN"])),
                        IADS=int(_cont_par(sub_cont, "IADS", kwargs_eq["IADS"])),
                        MXBF=int(_cont_par(sub_cont, "MXBF", 50)),
                        IID=0,
                    )
                    R_c2 = auto.run(**kwargs_c2)
                    if bool(getattr(bswitch, "bothside", False)):
                        R_c2_neg = auto.run(**dict(kwargs_c2, DS=-kwargs_c2["DS"]))
                        R_c2 = auto.merge(R_c2 + R_c2_neg)
                    c2_name = f"{bname}_{label_prefix}{lab}"
                    auto.sv(R_c2, c2_name)
                    out.append((c2_name, source_type, fp1_name, fp2_name, R_c2))
                except Exception as e:
                    import warnings

                    warnings.warn(
                        f"Codim-2 continuation '{bname}' from {label_prefix}{lab} failed: {type(e).__name__}: {e}",
                        stacklevel=2,
                    )
        return out

    # ── Cleanup ──────────────────────────────────────────────────────────

    @staticmethod
    def cleanup(result):
        """Remove the temporary working directory of a NumCont result."""
        wd = getattr(result, "workdir", None)
        if wd and os.path.isdir(wd):
            shutil.rmtree(wd, ignore_errors=True)
