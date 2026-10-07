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


SCIPY_SOLVERS = {"Dopri5": "RK45", "Dopri853": "DOP853"}
"""Each canonical integration method SciPy's ``solve_ivp`` implements → its name there, the Dormand–Prince pairs; ``solve_ivp``'s own method names (``SOLVE_IVP_METHODS``) are taken as they are."""

SOLVE_IVP_METHODS = ("RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA")


def _warmup_to_steady_state(model, cont) -> np.ndarray:
    """The state AUTO starts *cont* from, by the ``initial_state`` `ContinuationAdapter.initial_state` resolves.

    - ``given`` and ``newton``: *model*'s declared initial state (`ContinuationAdapter.declared_state`), which `ContinuationAdapter.start_dynamics` set to the Newton equilibrium for ``newton``, with no integration.
    - ``time_integration``, the default: integrate the model's vector field for ``initial_state.duration`` time units and return the final state, by the ``initial_state.solver`` declared. An adaptive solver SciPy implements (`SCIPY_SOLVERS`, `SOLVE_IVP_METHODS`) runs at the ``initial_state``'s ``abs_tol`` and ``rel_tol``, the pair BifurcationKit reads; a fixed-step method (Euler, Heun, RungeKutta4thOrder) steps by its update expression (`compgraph.integration_step`) at the solver's ``step_size``. Without a solver the settle is SciPy's LSODA, tolerance-controlled like BifurcationKit's rather than stepped by a fixed-step method, which a stiff or fast model would take outside its stability region at a step nobody declared.

    Raises:
        ValueError: The solver is one SciPy does not implement (``Tsit5``, `Solver.method`'s default, among them), a fixed-step method without a ``step_size``, or an adaptive one with one.
        RuntimeError: The adaptive integration fails.
    """
    from scipy.integrate import solve_ivp

    from tvbo.datamodel.schema import InitialState, Integrator
    from tvbo.run.compgraph import integration_step
    from tvbo.utils import integration_method

    x0 = ContinuationAdapter.declared_state(model)
    if not ContinuationAdapter.integrates_to_start(cont):
        return x0

    declared, default = ContinuationAdapter.initial_state(cont), InitialState()
    duration, atol, rtol = (
        float(getattr(default, slot) if getattr(declared, slot, None) is None else getattr(declared, slot))
        for slot in ("duration", "abs_tol", "rel_tol")
    )
    tolerances = {"atol": atol, "rtol": rtol}
    solver = getattr(declared, "solver", None)
    name = str(solver.method) if solver is not None else "LSODA"
    step_size = getattr(solver, "step_size", None)
    canonical = integration_method(name, strict=False)
    dfun = model.execute(format="python")

    adaptive = SCIPY_SOLVERS.get(canonical, name if name in SOLVE_IVP_METHODS else None)
    if adaptive is not None:
        if step_size:
            raise ValueError(
                f"initial_state.solver declares step_size {step_size} for {name}, an adaptive solver, which chooses its own step from abs_tol and rel_tol"
            )
        solution = solve_ivp(lambda t, u: dfun(u, t), (0.0, duration), x0, method=adaptive, **tolerances)
        if not solution.success:
            raise RuntimeError(f"the time-integration warm-up of {model.name} failed: {solution.message}")
        return np.asarray(solution.y[:, -1], dtype=float)

    try:
        step = integration_step(Integrator(method=canonical or name))
    except ValueError as err:
        raise ValueError(
            f"initial_state.solver.method {name!r} is not a solver the numcont warm-up can integrate by: it integrates with SciPy, by one of "
            f"{', '.join(SOLVE_IVP_METHODS)}, Dopri5 or Dopri853, or by a fixed-step method (Euler, Heun, RungeKutta4thOrder) at a step_size. "
            f"Declare one of them, or continue on bifurcationkit, which hands {name!r} to DifferentialEquations.jl."
        ) from err
    if not step_size:
        raise ValueError(
            f"initial_state.solver.method {name!r} is a fixed-step method, and initial_state.solver declares no step_size"
        )
    state = x0
    for k in range(int(round(duration / step_size))):
        state = step(lambda u, t=k * step_size: dfun(u, t), state, step_size)
    return np.asarray(state, dtype=float)


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
    """``(name, p_min, p_max)`` of *cont*'s free parameter, bounded as `ContinuationAdapter.parameter_bounds` resolves it."""
    from tvbo.utils import as_list

    first = as_list(cont.free_parameters)[0]
    return (
        str(first.name),
        *ContinuationAdapter.parameter_bounds(first, model, f"continuation {getattr(cont, 'name', None)!r}"),
    )


def _state_jacobian(model, continuation) -> dict | None:
    """The nonzero entries of the Jacobian ``DFDU`` of *model*'s single node, keyed by 0-based ``(equation, state variable)``, where *continuation* declares a two-parameter branch; else ``None``.

    AUTO-07p differences ``FUNC`` wherever the source supplies no ``DFDU``, and on a two-parameter curve it differences that Jacobian again, which leaves its Newton corrector too little accuracy to meet the ``newton_tol`` an equilibrium meets, so the curve stops short. So the source carries the analytic Jacobian there, with every model function inlined and every coupling input zeroed as ``FUNC`` zeroes it, and AUTO runs with ``JAC=-1``, differencing only in the parameters. A continuation without a two-parameter branch keeps AUTO's differences, which meet its tolerances, rather than paying for a symbolic Jacobian whose size grows with the model's. Where an entry has no closed form (a derivative of ``abs``, ``Mod`` or ``Heaviside``, or of a function SymPy cannot differentiate), this is ``None`` too, and AUTO differences.
    """
    import sympy as sp
    from sympy.core.function import AppliedUndef

    from tvbo.analysis.linear_response import jacobian_terms

    branches = ContinuationAdapter.branches_of(continuation).values() if continuation is not None else ()
    if not any(ContinuationAdapter.is_codim2(branch, continuation) for branch in branches):
        return None
    zero = {sp.Symbol(name): sp.S.Zero for name in model.coupling_inputs or ()}
    jacobian = jacobian_terms(model, inline_functions=True)["Jloc"].xreplace(zero)
    entries = {(i, j): jacobian[i, j] for i in range(jacobian.rows) for j in range(jacobian.cols) if jacobian[i, j] != 0}
    if any(entry.has(sp.Derivative, sp.Subs, sp.DiracDelta) or entry.atoms(AppliedUndef) for entry in entries.values()):
        return None
    return entries


_NEWTON_CONSTANTS = ContinuationAdapter.AUTO_NEWTON_CONSTANTS

_SCHEMA_CONSTANTS = {
    "NMX": "max_steps",
    "DS": "ds",
    "DSMAX": "ds_max",
    "DSMIN": "ds_min",
    **{key: slot for key, (slot, _, _) in _NEWTON_CONSTANTS.items()},
}
"""The continuation slot each AUTO constant a continuation can declare is read from; every other constant is this backend's own."""


def _cont_par(cont, key, default=None):
    """The AUTO constant *key* as *cont* declares it through its slot (`_SCHEMA_CONSTANTS`), else *default*."""
    declared = getattr(cont, _SCHEMA_CONSTANTS[key], None) if cont is not None else None
    return default if declared is None else declared


def _branch_par(cont, nested, key, default):
    """One AUTO constant for a branch run of *cont*: its *nested* continuation's declaration, else *cont*'s (`ContinuationAdapter.setting`), else *default*, this backend's own for that kind of run."""
    value = ContinuationAdapter.setting(key, nested, cont, read=_cont_par)
    return default if value is None else value


def _po_branches(cont) -> dict:
    """*cont*'s periodic-orbit branches keyed by name: every branch `ContinuationAdapter.is_codim2` does not read as two-parameter."""
    return {
        name: branch
        for name, branch in ContinuationAdapter.branches_of(cont).items()
        if not ContinuationAdapter.is_codim2(branch, cont)
    }


def _fresh_start() -> dict:
    """The keywords that start an AUTO-07p run from no previous run: AUTO's default constants (``c``) and no solution (``s``).

    AUTO keeps its constants and solution in one module-level runner, and a run given no starting solution reuses both, so the equilibrium run of a continuation would otherwise inherit whatever the process last ran: its ``UZSTOP``, ``NTST`` or ``STOP``, and a solution of another model's dimension.
    """
    from auto import parseC

    return {"c": parseC.parseC(), "s": None}


def _run_both_ways(auto, kwargs, bothside):
    """``auto.run(**kwargs)``, merged with the same run at ``-DS`` where *bothside*."""
    result = auto.run(**kwargs)
    if bothside:
        result = auto.merge(result + auto.run(**dict(kwargs, DS=-kwargs["DS"])))
    return result


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
        """The template context for *continuation* on *model*: the shared pair, the Fortran name of every symbol the template declares (``replace``), the coupling inputs a single node zeroes (``coupling_zero``), and the analytic Jacobian the source carries (``jacobian``, `_state_jacobian`)."""
        context = ContinuationAdapter._prepare_context(model, continuation, **kwargs)
        emitted = [
            *model.state_variables,
            *model.parameters,
            *(model.in_dependency_order("derived_variables") if model.derived_variables else ()),
            *(model.in_dependency_order("derived_parameters") if model.derived_parameters else ()),
        ]
        context["replace"] = ContinuationAdapter.fortran_names(emitted, reserved=NumContAdapter.FUNC_ARGUMENTS)
        context["coupling_zero"] = list(model.coupling_inputs or ())
        context["jacobian"] = _state_jacobian(model, continuation)
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
        context = self._prepare_context(model, cont)
        with open(f90_path, "w") as fh:
            fh.write(self.render_template(context))

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
            **_fresh_start(),
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
            IADS=1,
            IAD=1,
            IID=0,
            EPSS=1e-5,
            **{key: cast(_cont_par(cont, key, default)) for key, (_, cast, default) in _NEWTON_CONSTANTS.items()},
            RL0=p_min,
            RL1=p_max,
            NMX=int(_cont_par(cont, "NMX", 1000)),
            NPR=10,
            DS=float(_cont_par(cont, "DS", 0.005)),
            DSMAX=float(_cont_par(cont, "DSMAX", 0.05)),
            DSMIN=float(_cont_par(cont, "DSMIN", 1e-6)),
            MXBF=50,
            # Every run restarted from this one inherits JAC: -1 where the source carries DFDU, else AUTO differences FUNC.
            JAC=-1 if context["jacobian"] else 0,
        )

        cwd0 = os.getcwd()
        os.chdir(workdir)
        try:
            R_eq = _run_both_ways(auto, kwargs_eq, self.bothside(cont))
            auto.sv(R_eq, cont_name)

            # 4. Periodic-orbit continuation, per periodic-orbit branch, from the Hopf points it selects
            po_results = []
            po_profiles = []
            for po_name, po_branch in _po_branches(cont).items():
                po_cont = getattr(po_branch, "continuation", None)
                hopf_index = self.periodic_orbit_source(po_branch)
                for ordinal in self.select_points(range(1, _n_labelled(R_eq, "HB") + 1), hopf_index):
                    kwargs_po = dict(
                        data=R_eq(f"HB{ordinal}"),
                        EPSS=1e-5,
                        IPS=2,
                        ISP=2,
                        ISW=1,
                        ICP=[fp_name, "PERIOD"],
                        RL0=p_min,
                        RL1=p_max,
                        NMX=int(_branch_par(cont, po_cont, "NMX", 400)),
                        NPR=1,
                        DS=float(_branch_par(cont, po_cont, "DS", 0.01)),
                        DSMAX=float(_branch_par(cont, po_cont, "DSMAX", 0.1)),
                        DSMIN=float(_branch_par(cont, po_cont, "DSMIN", 1e-6)),
                        IADS=1,
                        MXBF=50,
                        IID=0,
                        **{
                            key: cast(_branch_par(cont, po_cont, key, default))
                            for key, (_, cast, default) in _NEWTON_CONSTANTS.items()
                        },
                    )
                    R_po = _run_both_ways(auto, kwargs_po, self.bothside(po_branch))
                    run_name = f"{po_name}_HB{ordinal}"
                    auto.sv(R_po, run_name)
                    po_results.append((run_name, R_po))
                    po_profiles.append(_orbit_profiles(R_po(), list(model.state_variables)))

            # 5. Codim-2 fold (and Hopf, BP) continuation from BranchSwitch specs
            codim2_results = self._run_codim2_branches(
                auto=auto,
                R_eq=R_eq,
                cont=cont,
                fp_name=fp_name,
                kwargs_eq=kwargs_eq,
                model=model,
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

    def _run_codim2_branches(self, *, auto, R_eq, cont, fp_name, kwargs_eq, model):
        """Run codim-2 fold/Hopf/BP continuations declared via ``cont.branches``.

        Each :class:`~tvbo.classes.continuation.BranchSwitch` that `ContinuationAdapter.is_codim2` reads as a two-parameter continuation triggers a separate AUTO restart, with ``ISW=2`` (fold/Hopf continuation) and ``ICP`` the primary *fp_name* and the branch's second parameter (`ContinuationAdapter.codim2_parameter`), from every special point its ``source_point`` selects — ``'fold:1'``, ``'hopf:all'``, ``'bp:-1'``, read by `ContinuationAdapter.codim2_source` — and in both directions where `ContinuationAdapter.bothside` reads the branch so. AUTO bounds the principal parameter by ``RL0``/``RL1``, so the primary keeps the equilibrium continuation's bounds (*kwargs_eq*) and the second parameter's bounds on *model* (`ContinuationAdapter.parameter_bounds`) stop the curve through ``UZSTOP``. Every AUTO constant is the nested continuation's where it declares one, else the equilibrium continuation's, else this backend's default for a two-parameter run (`_branch_par`).

        Returns a list of ``(name, source_type, fp1_name, fp2_name, R_c2)`` tuples consumed by :meth:`BifurcationResult.from_auto`.
        """
        out = []
        for bname, bswitch in self.branches_of(cont).items():
            fp2 = self.codim2_parameter(bswitch, cont)
            if fp2 is None:
                continue
            # AUTO labels a special point by its canonical code (LP, HB, BP) and restarts from it with ISW=2.
            kind, index = self.codim2_source(bswitch)
            label_prefix = kind
            source_type = self.SOURCE_KINDS[kind]
            sub_cont = bswitch.continuation
            fp1_name = fp_name
            fp2_name = str(fp2.name)
            fp2_lo, fp2_hi = self.parameter_bounds(fp2, model, f"branch {bname!r}")
            ordinals = self.select_points(range(1, _n_labelled(R_eq, label_prefix) + 1), index)

            for ordinal in ordinals:
                lab = ordinal  # AUTO ordinal label
                try:
                    kwargs_c2 = dict(
                        data=R_eq(f"{label_prefix}{lab}"),
                        EPSS=1e-5,
                        IPS=1,
                        ISP=2,
                        ISW=2,
                        ILP=0,
                        ICP=[fp1_name, fp2_name],
                        RL0=kwargs_eq["RL0"],
                        RL1=kwargs_eq["RL1"],
                        UZSTOP={fp2_name: [fp2_lo, fp2_hi]},
                        NMX=int(_branch_par(cont, sub_cont, "NMX", 400)),
                        NPR=1,
                        DS=float(_branch_par(cont, sub_cont, "DS", 0.01)),
                        DSMAX=float(_branch_par(cont, sub_cont, "DSMAX", 0.05)),
                        DSMIN=float(_branch_par(cont, sub_cont, "DSMIN", 1e-6)),
                        IADS=1,
                        MXBF=50,
                        IID=0,
                        **{
                            key: cast(_branch_par(cont, sub_cont, key, default))
                            for key, (_, cast, default) in _NEWTON_CONSTANTS.items()
                        },
                    )
                    R_c2 = _run_both_ways(auto, kwargs_c2, self.bothside(bswitch))
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
