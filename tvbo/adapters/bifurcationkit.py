"""BifurcationKit.jl backend adapter for SimulationExperiment.

Uses juliacall to execute generated BifurcationKit Julia code and return BifurcationResult objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tvbo.adapters.base import ContinuationAdapter
from tvbo.adapters.julia_model import build_model_context, julia_solver

if TYPE_CHECKING:
    from tvbo.analysis.bifurcation import BifurcationResult

# Schema attr → Julia kwarg name mapping for ContinuationPar
_CONT_FIELDS = [
    ("ds", "ds"),
    ("ds_min", "dsmin"),
    ("ds_max", "dsmax"),
    ("max_steps", "max_steps"),
    ("tol_stability", "tol_stability"),
    ("nev", "nev"),
    ("n_inversion", "n_inversion"),
    ("max_bisection_steps", "max_bisection_steps"),
    ("detect_bifurcation", "detect_bifurcation"),
]


def _str(val):
    """Convert LinkML PermissibleValue enums or plain values to string."""
    if val is None:
        return None
    return val.text if hasattr(val, "text") else str(val)


def _get(obj, path, default=None):
    """Read a dotted attribute path, e.g. ``'initial_state.duration'``, returning *default* at the first missing link."""
    cur = obj
    for p in path.split("."):
        if cur is None:
            return default
        cur = getattr(cur, p, None)
    return cur if cur is not None else default


def _get_param(obj, name):
    """Look up named parameter value in obj.parameters."""
    if obj is None:
        return None
    params = getattr(obj, "parameters", None)
    if params is None:
        return None
    if isinstance(params, dict):
        p = params.get(name)
        return p.value if p and p.value is not None else None
    if isinstance(params, (list, tuple)):
        for p in params:
            if getattr(p, "name", None) == name:
                return p.value if p.value is not None else None
    return None


def _get_option(obj, name):
    """Look up named option value in obj.options."""
    if obj is None:
        return None
    opts = getattr(obj, "options", None)
    if opts is None:
        return None
    if isinstance(opts, dict):
        o = opts.get(name)
        return o.value if o and o.value is not None else None
    if isinstance(opts, (list, tuple)):
        for o in opts:
            if getattr(o, "name", None) == name:
                return o.value if o.value is not None else None
    return None


def _julia_solver(method):
    """The DifferentialEquations.jl solver a declared `Solver.method` names (`julia_solver`), or ``None`` where none is declared."""
    method = _str(method)
    return julia_solver(method) if method else None


def _julia_ordinal(index):
    """A `source_point` index as a Julia index into the special points of its kind — ``3``, ``end``, ``end-1`` — or ``None`` for every point."""
    if index is None:
        return None
    return str(index) if index > 0 else "end" if index == -1 else f"end{index + 1}"


def _cont_fields(continuation, parent=None) -> dict:
    """The `ContinuationPar` fields of *continuation*, keyed by their Julia keyword, each one it leaves unset taken from *parent* (`ContinuationAdapter.setting`); a field neither declares is left out, to BifurcationKit's default."""
    fields = {}
    for slot, keyword in _CONT_FIELDS:
        value = ContinuationAdapter.setting(slot, continuation, parent)
        if value is not None:
            fields[keyword] = value
    return fields


def _cont_kwargs(continuation, parent=None):
    """`_cont_fields` as ``keyword = value`` strings."""
    return [f"{keyword} = {value}" for keyword, value in _cont_fields(continuation, parent).items()]


def _branch_alg(branch_continuation):
    """The ``alg`` keyword of a branch run whose nested continuation declares a ``tangent`` option, else ``None``: BifurcationKit continues a branch by its parent's algorithm unless it is given one."""
    tangent = _str(_get_option(branch_continuation, "tangent"))
    return f"alg = PALC(tangent = {tangent}())" if tangent else None


def _newton_kwargs(continuation, parent=None):
    """The `NewtonPar` keywords of *continuation*, each Newton setting it leaves unset taken from *parent* (`ContinuationAdapter.setting`): ``newton_tol`` as the residual tolerance ``tol``, ``newton_max_iterations`` as ``max_iterations``."""
    args = []
    for slot, keyword in (("newton_tol", "tol"), ("newton_max_iterations", "max_iterations")):
        value = ContinuationAdapter.setting(slot, continuation, parent)
        if value is not None:
            args.append(f"{keyword} = {value}")
    return args


class BifurcationKitAdapter(ContinuationAdapter):
    """Adapter for running bifurcation analysis via BifurcationKit.jl.

    A continuation backend: it renders one ``(Dynamics, Continuation)`` pair at a time rather than a whole network context, so it takes its pair resolution from [`ContinuationAdapter`](#tvbo.adapters.base.ContinuationAdapter).
    """

    TEMPLATE = "tvbo-julia-BifurcationKit.jl.mako"

    # ── Context preparation ──────────────────────────────────────────────

    @staticmethod
    def _prepare_context(model, cont, **kwargs):
        """Pre-compute every value the template needs.

        Returns a flat dict of simple strings/numbers/booleans so the template does zero processing.
        """
        ctx = dict(model=model, continuation=cont)

        # Network context (None or 1-node ⇒ single-node RHS; >1 node ⇒ coupled).
        network = kwargs.get("network")
        n_nodes = int(getattr(network, "number_of_nodes", 0) or 0) if network is not None else 0
        ctx["network"] = network if n_nodes > 1 else None
        ctx["n_nodes"] = n_nodes if n_nodes > 1 else 1

        # Constraint-defined free parameters (FIC J_i): promoted to unknown state blocks whose defining equation is the TuningObjective residual (D2).
        ctx["constraints"] = kwargs.get("constraints") or []

        # -- Free parameter --
        fp_dict = cont.free_parameters if cont else None
        if fp_dict:
            fp_first = next(iter(fp_dict.values())) if isinstance(fp_dict, dict) else fp_dict[0]
            ICS = str(fp_first.name)
            p_min, p_max = BifurcationKitAdapter.parameter_bounds(
                fp_first, model, f"continuation {getattr(cont, 'name', None)!r}"
            )
        else:
            ICS = kwargs.get("ICS")
            dom = model.parameters[ICS].domain if ICS else None
            p_min = float(kwargs.get("p_min", dom.lo if dom else None))
            p_max = float(kwargs.get("p_max", dom.hi if dom else None))

        p_default = float(model.parameters[ICS].value) if model.parameters[ICS].value is not None else None
        ctx["ICS"] = ICS
        ctx["p_min"] = p_min
        ctx["p_max"] = p_max

        # -- Model function, and what each continuation step records --
        mc = build_model_context(model, ctx["network"], constraints=ctx["constraints"])
        ctx["mc"] = mc
        # The record closure takes the continuation parameter as its argument, so it destructures every other one from `p`.
        ctx["record_destructure"] = ", ".join(name for name in mc["destructure_names"] if name != ICS)
        if ctx["network"] is None:
            ctx["record_fields"] = ", ".join(f"{name} = x[{i + 1}]" for i, name in enumerate(model.state_variables))
        else:
            ctx["record_fields"] = ", ".join(
                f"{name}_max = _{name}_max, {name}_mean = _{name}_sum / N" for name in mc["record_obs"]
            )

        # -- Initial state: integrated to, or the model's own (`ContinuationAdapter.integrates_to_start`) --
        iss = BifurcationKitAdapter.initial_state(cont)
        ctx["integrate_to_start"] = BifurcationKitAdapter.integrates_to_start(cont)
        ctx["iss_duration"] = _get(iss, "duration")
        ctx["iss_solver"] = _julia_solver(_get(iss, "solver.method"))
        ctx["iss_atol"] = _get(iss, "abs_tol")
        ctx["iss_rtol"] = _get(iss, "rel_tol")

        # -- Algorithm string (schema default: PALC) --
        alg = _str(cont.algorithm) if cont else None
        tangent = _str(_get_option(cont, "tangent"))
        ctx["alg_str"] = f"{alg}(tangent = {tangent}())" if alg and tangent else f"{alg}()" if alg else None

        # -- ContinuationPar args string --
        cp_args = [f"p_min = {float(p_min)}", f"p_max = {float(p_max)}"]
        cp_args.extend(_cont_kwargs(cont))
        nargs = _newton_kwargs(cont)
        if nargs:
            cp_args.append(f"newton_options = NewtonPar({', '.join(nargs)})")
        ctx["cp_args_str"] = ",\n    ".join(cp_args)

        # -- continuation() kwargs string --
        cont_kw = ["normC = norminf"]
        if BifurcationKitAdapter.bothside(cont):
            cont_kw.append("bothside = true")
        ctx["cont_call_kwargs_str"] = ", ".join(cont_kw)

        # -- Quiet --
        ctx["quiet"] = kwargs.get("quiet", True)

        # -- Branches --
        po_branches = []
        codim2_branches = []
        for b in BifurcationKitAdapter.branches_of(cont).values():
            if BifurcationKitAdapter.is_codim2(b, cont):
                codim2_branches.append(BifurcationKitAdapter._prepare_codim2_branch(b, cont, model, (p_min, p_max)))
            else:
                po_branches.append(BifurcationKitAdapter._prepare_branch(b, cont))
        ctx["branches"] = po_branches
        ctx["codim2_branches"] = codim2_branches
        # BifurcationKit continues a parameter as a Float64, so each one a branch continues in starts as one, whatever literal the model declares.
        start = {ICS: p_default, **{c2["ICS2"]: model.parameters[c2["ICS2"]].value for c2 in codim2_branches}}
        ctx["start_params"] = ", ".join(f"{name} = {float(value)}" for name, value in start.items())

        return ctx

    @staticmethod
    def _prepare_branch(br, parent=None):
        """Pre-compute all values for one periodic-orbit branch of the continuation *parent*, whose Newton settings it inherits where it leaves them unset."""
        bc = br.continuation  # sub-Continuation or None

        # The branch's ContinuationPar: the parent's (`opts_br`), with every field the branch or its parent declares.
        po_cp = _cont_kwargs(bc, parent)
        po_n = _newton_kwargs(bc, parent)
        if po_n:
            po_cp.append(f"newton_options = NewtonPar({', '.join(po_n)})")
        # Defaults to 0, which leaves `br.sol` empty and every orbit profile `nothing`.
        po_cp.append("save_sol_every_step = 1")
        po_cp_str = ", ".join(po_cp)

        hopf_idx = BifurcationKitAdapter.periodic_orbit_source(br)

        # Discretization (use schema defaults when unspecified)
        from tvbo.datamodel.schema import Discretization

        disc = br.discretization or Discretization()
        method = _str(_get(disc, "method"))

        # Collocation/trapezoid params (direct attributes with schema defaults)
        mesh_intervals = _get(disc, "mesh_intervals")
        degree = _get(disc, "degree")
        meshadapt = _get_param(disc, "mesh_adaptation")
        jacobian = _str(_get_option(disc, "jacobian"))

        # Shooting params (direct attribute with schema default)
        n_sections = _get(disc, "n_sections")
        parallel = _get_param(disc, "parallel")

        # ODE solver for flow-based methods (shooting, poincaré)
        ode_solver = _julia_solver(_get(disc, "ode_solver.method"))
        ode_abstol = _get(disc, "ode_solver.abs_tol")
        ode_reltol = _get(disc, "ode_solver.rel_tol")
        ode_time_span = float(_get_param(disc, "ode_time_span") or 1000.0)

        # Linear solver — Solver object on Discretization
        linear_solver = _str(_get(disc, "linear_solver.method"))

        # PO continuation() kwargs string
        po_kw = ["plot = false", "args_po..."]
        if br.delta_p is not None:
            po_kw.append(f"\u03b4p = {br.delta_p}")
        po_alg = _branch_alg(bc)
        if po_alg:
            po_kw.append(po_alg)
        if linear_solver:
            po_kw.append(f"linear_algo = {linear_solver}()")
        po_kw.append("verbosity = 0")
        if BifurcationKitAdapter.bothside(br):
            po_kw.append("bothside = true")
        max_norm = _get_param(br, "max_norm_bound")
        if max_norm is not None:
            po_kw.append(f"callback_newton = BifurcationKit.cbMaxNorm({float(max_norm)})")
        po_kwargs_str = ",\n            ".join(po_kw)

        return dict(
            name=str(getattr(br, "name", None) or "?"),
            po_cp_args_str=po_cp_str,
            hopf_idx_jl=_julia_ordinal(hopf_idx),
            method=method,
            mesh_intervals=mesh_intervals,
            degree=degree,
            meshadapt=meshadapt,
            jacobian=jacobian,
            n_sections=n_sections,
            parallel=parallel,
            ode_solver=ode_solver,
            ode_abstol=ode_abstol,
            ode_reltol=ode_reltol,
            ode_time_span=ode_time_span,
            po_kwargs_str=po_kwargs_str,
        )

    # ── Public API ───────────────────────────────────────────────────────

    def render_continuation(self, model, continuation, **kwargs) -> str:
        """BifurcationKit Julia for *continuation* on *model*, continuing the coupled system where the experiment declares a multi-node network; *kwargs* is extra context for `_prepare_context`."""
        network = getattr(self.experiment, "network", None)
        return super().render_continuation(
            model, continuation, network=network, constraints=self._derive_constraints(model), **kwargs
        )

    def _derive_constraints(self, model):
        """Derive constraint-defined free parameters for the continuation.

        Reuses the *existing* declarations (no new schema): a parameter marked ``free: true`` on the model, together with an activity-target ``TuningObjective`` on one of the experiment's algorithms, defines a constraint ``target_variable = target_value``. Each such free parameter (e.g. the FIC ``J_i``) is promoted by the emitter to an unknown state block whose defining equation is that residual (see ``_build_network_context``).

        Returns a list of ``{"parameter", "target_variable", "target_value"}``; empty when no parameter is free (E-E / FFI variants ⇒ plain continuation).
        """
        from tvbo.utils import as_list

        free = [p.name for p in model.parameters.values() if getattr(p, "free", False)]
        if not free:
            return []
        algos = as_list(getattr(self.experiment, "algorithms", None))

        def _activity_objective(a):
            """The algorithm's objective iff it is an activity target (has a target_variable + target_value); else None."""
            o = getattr(a, "objective", None)
            tv = getattr(o, "target_variable", None) if o is not None else None
            return o if (tv is not None and getattr(o, "target_value", None) is not None) else None

        def _tuned_params(a):
            """Parameter names this algorithm's update rules tune (target_parameter)."""
            names = set()
            for r in as_list(getattr(a, "update_rules", None)):
                tp = getattr(r, "target_parameter", None)
                n = getattr(tp, "name", None) or (str(tp) if tp is not None else None)
                if n:
                    names.add(n)
            return names

        constraints = []
        for fp in free:
            # Prefer the algorithm that explicitly tunes THIS parameter (so multiple free params each get their own target); fall back to a lone activity objective only when this is the sole free param.
            obj = next(
                (_activity_objective(a) for a in algos if fp in _tuned_params(a) and _activity_objective(a)),
                None,
            )
            if obj is None and len(free) == 1:
                obj = next((_activity_objective(a) for a in algos if _activity_objective(a)), None)
            if obj is None:
                continue
            tv = getattr(obj.target_variable, "name", None) or str(obj.target_variable)
            constraints.append({"parameter": str(fp), "target_variable": str(tv), "target_value": float(obj.target_value)})
        return constraints

    @staticmethod
    def _get_ics(cont):
        """Extract the free parameter name from a Continuation spec."""
        fp = getattr(cont, "free_parameters", None)
        if not fp:
            return None
        if isinstance(fp, dict) and fp:
            return str(next(iter(fp.values())).name)
        if isinstance(fp, list) and fp:
            return str(fp[0].name)
        return None

    def run_one(self, model, cont, name, **kwargs) -> BifurcationResult:
        """Render *cont* on *model* as BifurcationKit Julia, execute it, and wrap the branch in a `BifurcationResult`, with its periodic-orbit and codim-2 branches where it declares any."""
        from tvbo.analysis import BifurcationResult
        from tvbo.run.julia import extract_bifurcation_result, run_julia_code

        run_julia_code(self.render_continuation(model, cont, **kwargs))

        ICS = self._get_ics(cont)
        bif_res = BifurcationResult(br=extract_bifurcation_result(), model=model, ICS=ICS, **kwargs)
        branches = self.branches_of(cont).values()
        if any(not self.is_codim2(branch, cont) for branch in branches):
            bif_res.periodic_orbits = self._extract_periodic_orbits(model, ICS=ICS, **kwargs)
        if any(self.is_codim2(branch, cont) for branch in branches):
            bif_res.codim2_curves = self._extract_codim2_results(model, ICS=ICS, **kwargs)
        return bif_res

    # ── Private helpers ──────────────────────────────────────────────────

    def _extract_periodic_orbits(self, model, **kwargs) -> list:
        """Extract periodic orbit branches from Julia Main after execution.

        Also attaches each branch's orbit waveforms (``orbit_profiles``, a ``[n_steps, NPROF, n_vars]`` array phase-resampled over one period) when the Julia run produced them (``po_results.profiles``); the actual periodic-orbit profile is otherwise not recorded by BifurcationKit.
        """
        import numpy as np

        from tvbo.adapters.julia import eval_with_auto_install
        from tvbo.analysis import BifurcationResult

        po = eval_with_auto_install("po_results")
        try:
            prof_list = list(getattr(po, "profiles", None) or [])
        except Exception:
            prof_list = []
        out = []
        for i, p in enumerate(po.branches):
            res = BifurcationResult(br=p, model=model, **kwargs)
            if i < len(prof_list) and prof_list[i] is not None:
                try:
                    res.orbit_profiles = np.asarray(prof_list[i], dtype=float)
                except Exception:
                    pass
            out.append(res)
        return out

    def _extract_codim2_results(self, model, ICS, **kwargs) -> list:
        """Every codim-2 curve the Julia run left in ``codim2_results``, each carrying the primary *ICS* in ``param`` and the second parameter in ``param2``, as every continuation backend does.

        BifurcationKit continues a codim-2 curve in the second parameter, so its own ``param`` column holds that one; both parameters are also recorded under their names, which is where the two columns are read from.

        Raises:
            KeyError: If a curve records the primary or no second parameter under its name.
        """
        from tvbo.adapters.julia import eval_with_auto_install
        from tvbo.analysis import BifurcationResult
        from tvbo.analysis.bifurcation import continuation_kind

        kinds = {"HopfCont": "hopf", "FoldCont": "fold"}
        results = []
        for br_obj in eval_with_auto_install("codim2_results"):
            res = BifurcationResult(br=br_obj, model=model, ICS=ICS, **kwargs)
            res._is_codim2 = True
            res._source_type = kinds.get(continuation_kind(br_obj), "fold")
            if not res.df.empty:
                second = [c for c in res.df.columns if c in model.parameters and c != ICS]
                if ICS not in res.df.columns or len(second) != 1:
                    raise KeyError(
                        f"a BifurcationKit codim-2 curve records the parameters {[c for c in res.df.columns if c in model.parameters]}, "
                        f"where the primary {ICS!r} and one second parameter were expected."
                    )
                res.df["param"], res.df["param2"] = res.df[ICS], res.df[second[0]]
                res._ics_name, res._fp2_name = ICS, second[0]
            results.append(res)
        return results

    @staticmethod
    def _prepare_codim2_branch(br, parent_cont, model, primary_bounds):
        """Pre-compute context for a codim-2 branch of *parent_cont* on *model*, continued in the parent's primary parameter and the second `ContinuationAdapter.codim2_parameter` names.

        BifurcationKit continues the curve in the second parameter, bounded as `ContinuationAdapter.parameter_bounds` resolves it, and carries the primary among its unknowns, so the primary's *primary_bounds* stop the curve through a finaliser, as AUTO-07p's ``RL0``/``RL1`` do.

        Raises:
            ValueError: If *br* is not a two-parameter continuation (`ContinuationAdapter.is_codim2`), or declares no source point (`ContinuationAdapter.codim2_source`).
        """
        bc = br.continuation
        fp2 = BifurcationKitAdapter.codim2_parameter(br, parent_cont)
        if fp2 is None:
            raise ValueError(
                f"codim-2 branch {getattr(br, 'name', '?')!r} frees no parameter besides the primary it inherits; "
                "declare the second parameter in its continuation's free_parameters."
            )

        ICS2 = str(fp2.name)
        p2_min, p2_max = BifurcationKitAdapter.parameter_bounds(fp2, model, f"branch {getattr(br, 'name', None)!r}")

        kind, index = BifurcationKitAdapter.codim2_source(br)

        # Codim-2 ContinuationPar args
        fields = {"p_min": p2_min, "p_max": p2_max, "ds": 0.01, "dsmax": 0.1, "max_steps": 300}
        fields.update(_cont_fields(bc, parent_cont))
        cp_args = [f"{keyword} = {value}" for keyword, value in fields.items()]
        newton = _newton_kwargs(bc, parent_cont)
        if newton:
            cp_args.append(f"newton_options = NewtonPar({', '.join(newton)})")
        codim2_cp_str = ", ".join(cp_args)

        # Codim-2 continuation kwargs
        codim2_kw = [
            "normC = norminf",
            "detect_codim2_bifurcation = 2",
            "update_minaug_every_step = 1",
            "start_with_eigen = true",
            "verbosity = 0",
            f"finalise_solution = (z, tau, step, contResult; prob, k...) -> {float(primary_bounds[0])} <= first(BifurcationKit.getp(z.u, prob)) <= {float(primary_bounds[1])}",
        ]
        codim2_alg = _branch_alg(bc)
        if codim2_alg:
            codim2_kw.append(codim2_alg)
        if BifurcationKitAdapter.bothside(br):
            codim2_kw.append("bothside = true")
        codim2_kwargs_str = ",\n            ".join(codim2_kw)

        return {
            "name": str(br.name),
            "ICS2": ICS2,
            "p2_min": p2_min,
            "p2_max": p2_max,
            "source_type": BifurcationKitAdapter.SOURCE_KINDS[kind],
            # BifurcationKit labels a fold on an equilibrium branch `:bp` as readily as `:fold`, so a fold or branch-point source selects both.
            "is_fold": kind in ("LP", "BP"),
            "source_idx_jl": _julia_ordinal(index),
            "codim2_cp_str": codim2_cp_str,
            "codim2_kwargs_str": codim2_kwargs_str,
        }
