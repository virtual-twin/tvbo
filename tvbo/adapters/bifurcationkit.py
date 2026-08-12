# -*- coding: utf-8 -*-
"""BifurcationKit.jl backend adapter for SimulationExperiment.

Uses juliacall to execute generated BifurcationKit Julia code
and return BifurcationResult objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tvbo.analysis.bifurcation import BifurcationResult
    from tvbo.classes.experiment import SimulationExperiment

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
    """Nested attribute access: _get(cont, 'initial_state.duration')."""
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


def _cont_kwargs(c):
    """Build list of 'jl_key = value' strings from a Continuation object."""
    args = []
    if c is None:
        return args
    for schema_attr, jl_key in _CONT_FIELDS:
        v = getattr(c, schema_attr, None)
        if v is not None:
            args.append(f"{jl_key} = {v}")
    return args


def _newton_kwargs(c):
    """Build NewtonPar kwargs from a Continuation object."""
    args = []
    if c is None:
        return args
    if c.newton_tol is not None:
        args.append(f"tol = {c.newton_tol}")
    if c.newton_max_iterations is not None:
        args.append(f"max_iterations = {c.newton_max_iterations}")
    return args


class BifurcationKitAdapter:
    """Adapter for running bifurcation analysis via BifurcationKit.jl.

    Unlike NetworkDynamics/MTK adapters, this does not inherit from
    BaseAdapter — bifurcation analysis operates on individual
    (Dynamics, Continuation) pairs rather than a full network context.
    """

    def __init__(self, experiment: "SimulationExperiment"):
        self.experiment = experiment

    # ── Context preparation ──────────────────────────────────────────────

    @staticmethod
    def _prepare_context(model, cont, **kwargs):
        """Pre-compute every value the template needs.

        Returns a flat dict of simple strings/numbers/booleans
        so the template does zero processing.
        """
        ctx = dict(model=model, continuation=cont)

        # Network context (None or 1-node ⇒ single-node RHS; >1 node ⇒ coupled).
        network = kwargs.get("network")
        n_nodes = int(getattr(network, "number_of_nodes", 0) or 0) if network is not None else 0
        ctx["network"] = network if n_nodes > 1 else None
        ctx["n_nodes"] = n_nodes if n_nodes > 1 else 1

        # Constraint-defined free parameters (FIC J_i): promoted to unknown state
        # blocks whose defining equation is the TuningObjective residual (D2).
        ctx["constraints"] = kwargs.get("constraints") or []

        # -- Free parameter --
        fp_dict = cont.free_parameters if cont else None
        if fp_dict:
            fp_first = next(iter(fp_dict.values())) if isinstance(fp_dict, dict) else fp_dict[0]
            ICS = str(fp_first.name)
            if fp_first.domain:
                p_min = float(fp_first.domain.lo)
                p_max = float(fp_first.domain.hi)
            elif model.parameters[ICS].domain:
                p_min = float(model.parameters[ICS].domain.lo)
                p_max = float(model.parameters[ICS].domain.hi)
            else:
                p_min, p_max = None, None
        else:
            ICS = kwargs.get("ICS")
            dom = model.parameters[ICS].domain if ICS else None
            p_min = float(kwargs.get("p_min", dom.lo if dom else None))
            p_max = float(kwargs.get("p_max", dom.hi if dom else None))

        p_default = float(model.parameters[ICS].value) if model.parameters[ICS].value is not None else None
        ctx["ICS"] = ICS
        ctx["p_min"] = p_min
        ctx["p_max"] = p_max
        ctx["p_start"] = p_default

        # -- Initial state (use schema defaults when unspecified) --
        from tvbo.datamodel.schema import InitialState

        iss = (cont.initial_state if cont else None) or InitialState()
        ctx["iss_duration"] = _get(iss, "duration")
        ctx["iss_solver"] = _str(_get(iss, "solver.method"))
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
        if cont and cont.bothside:
            cont_kw.append("bothside = true")
        ctx["cont_call_kwargs_str"] = ", ".join(cont_kw)

        # -- Quiet --
        ctx["quiet"] = kwargs.get("quiet", True)

        # -- Branches --
        branches_raw = []
        if cont and cont.branches:
            br_raw = cont.branches
            branches_raw = list(br_raw.values()) if isinstance(br_raw, dict) else list(br_raw)

        # Separate PO branches from codim-2 branches
        po_branches = []
        codim2_branches = []
        for b in branches_raw:
            bc = getattr(b, "continuation", None)
            has_fp2 = bc and getattr(bc, "free_parameters", None)
            if has_fp2:
                codim2_branches.append(BifurcationKitAdapter._prepare_codim2_branch(b, cont))
            else:
                po_branches.append(BifurcationKitAdapter._prepare_branch(b))
        ctx["branches"] = po_branches
        ctx["codim2_branches"] = codim2_branches

        return ctx

    @staticmethod
    def _prepare_branch(br):
        """Pre-compute all values for one branch (periodic orbit)."""
        bc = br.continuation  # sub-Continuation or None

        # PO ContinuationPar override args
        po_cp = _cont_kwargs(bc)
        po_n = _newton_kwargs(bc)
        if po_n:
            po_cp.append(f"newton_options = NewtonPar({', '.join(po_n)})")
        # BifurcationKit defaults `save_sol_every_step` to 0, leaving `br.sol` empty —
        # and the orbit waveforms are reconstructed from exactly those saved solutions.
        # Always request them, so `po_results.profiles` is populated rather than a
        # branch-long run of `nothing` behind a buried @warn.
        po_cp.append("save_sol_every_step = 1")
        po_cp_str = ", ".join(po_cp)

        # Source point
        source = br.source_point
        all_hopf = source == "hopf:all" if source else False
        if source and ":" in source:
            hopf_idx_str = source.split(":")[1]
            hopf_idx = int(hopf_idx_str) if hopf_idx_str.lstrip("-").isdigit() else None
        else:
            hopf_idx = None

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
        ode_solver = _str(_get(disc, "ode_solver.method"))
        ode_abstol = _get(disc, "ode_solver.abs_tol")
        ode_reltol = _get(disc, "ode_solver.rel_tol")
        ode_time_span = float(_get_param(disc, "ode_time_span") or 1000.0)

        # Linear solver — Solver object on Discretization
        linear_solver = _str(_get(disc, "linear_solver.method"))

        # PO continuation() kwargs string
        po_kw = ["plot = false", "args_po..."]
        if br.delta_p is not None:
            po_kw.append(f"\u03b4p = {br.delta_p}")
        po_tangent = _str(_get_option(bc, "tangent"))
        if po_tangent:
            po_kw.append(f"alg = PALC(tangent = {po_tangent}())")
        if linear_solver:
            po_kw.append(f"linear_algo = {linear_solver}()")
        po_kw.append("verbosity = 0")
        if br.bothside:
            po_kw.append("bothside = true")
        max_norm = _get_param(br, "max_norm_bound")
        if max_norm is not None:
            po_kw.append(f"callback_newton = BifurcationKit.cbMaxNorm({float(max_norm)})")
        po_kwargs_str = ",\n            ".join(po_kw)

        return dict(
            po_cp_args_str=po_cp_str,
            source=source,
            all_hopf=all_hopf,
            hopf_idx=hopf_idx,
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

    def render_code(self, model=None, continuation=None, **kwargs) -> str:
        """Render BifurcationKit Julia code for a single continuation.

        Parameters
        ----------
        model : Dynamics, optional
            The dynamics model. Defaults to ``experiment.dynamics``.
        continuation : Continuation, optional
            The continuation spec. Defaults to first in experiment.
        **kwargs
            Extra context passed to the Mako template.
        """
        from tvbo import templates

        model = model or self.experiment.dynamics
        if continuation is None:
            conts = getattr(self.experiment, "continuations", None) or {}
            if conts:
                continuation = next(iter(conts.values()))

        # A multi-node network on the experiment ⇒ continue the coupled system.
        network = getattr(self.experiment, "network", None)
        constraints = self._derive_constraints(model)
        ctx = self._prepare_context(model, continuation, network=network, constraints=constraints, **kwargs)

        template = templates.lookup.get_template("tvbo-julia-BifurcationKit.jl.mako")
        return template.render(**ctx)

    def _derive_constraints(self, model):
        """Derive constraint-defined free parameters for the continuation.

        Reuses the *existing* declarations (no new schema): a parameter marked
        ``free: true`` on the model, together with an activity-target
        ``TuningObjective`` on one of the experiment's algorithms, defines a
        constraint ``target_variable = target_value``. Each such free parameter
        (e.g. the FIC ``J_i``) is promoted by the emitter to an unknown state
        block whose defining equation is that residual (see ``_build_network_context``).

        Returns a list of ``{"parameter", "target_variable", "target_value"}``;
        empty when no parameter is free (E-E / FFI variants ⇒ plain continuation).
        """
        from tvbo.utils import as_list

        free = [p.name for p in model.parameters.values() if getattr(p, "free", False)]
        if not free:
            return []
        algos = as_list(getattr(self.experiment, "algorithms", None))

        def _activity_objective(a):
            """The algorithm's objective iff it is an activity target (has a
            target_variable + target_value); else None."""
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
            # Prefer the algorithm that explicitly tunes THIS parameter (so multiple
            # free params each get their own target); fall back to a lone activity
            # objective only when this is the sole free param.
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

    def run(self, **kwargs) -> "BifurcationResult | dict[str, BifurcationResult]":
        """Run bifurcation analysis for each continuation in the experiment.

        Iterates over ``experiment.continuations``, resolves the dynamics
        model for each, renders BifurcationKit Julia code, executes it,
        and wraps the result in ``BifurcationResult`` objects.

        Returns
        -------
        BifurcationResult or dict[str, BifurcationResult]
            Single result if one continuation, dict if multiple.
        """
        from tvbo.analysis import BifurcationResult
        from tvbo.run.julia import extract_bifurcation_result, run_julia_code

        exp = self.experiment
        conts = getattr(exp, "continuations", None) or {}
        if not conts:
            raise ValueError(
                "No continuations defined. Add continuation specs via exp.continuations or load from a bifurcation YAML."
            )

        results = {}
        for name, cont in conts.items():
            model = self._resolve_dynamics(cont)
            code = self.render_code(model=model, continuation=cont, **kwargs)

            run_julia_code(code)

            br_obj = extract_bifurcation_result()
            ICS = self._get_ics(cont)
            bif_res = BifurcationResult(br=br_obj, model=model, ICS=ICS, **kwargs)

            if getattr(cont, "branches", None):
                bif_res.periodic_orbits = self._extract_periodic_orbits(model, ICS=ICS, **kwargs)
                bif_res.codim2_curves = self._extract_codim2_results(model, ICS=ICS, **kwargs)

            results[name] = bif_res

        if len(results) == 1:
            return next(iter(results.values()))
        return results

    # ── Private helpers ──────────────────────────────────────────────────

    def _resolve_dynamics(self, cont):
        """Resolve the Dynamics model for a continuation spec."""
        exp = self.experiment
        dyn_ref = getattr(cont, "dynamics", None)
        if dyn_ref:
            dyn_name = str(dyn_ref)
            # Check primary dynamics
            if exp.dynamics and getattr(exp.dynamics, "name", None) == dyn_name:
                return exp.dynamics
            # Check network dynamics dict
            net_dyn = getattr(exp.network, "dynamics", None) if exp.network else None
            if isinstance(net_dyn, dict) and dyn_name in net_dyn:
                return net_dyn[dyn_name]
        if exp.dynamics is not None:
            return exp.dynamics
        raise ValueError(f"Cannot resolve dynamics for continuation. dynamics='{dyn_ref}' not found.")

    def _extract_periodic_orbits(self, model, **kwargs) -> list:
        """Extract periodic orbit branches from Julia Main after execution.

        Also attaches each branch's orbit waveforms (``orbit_profiles``, a
        ``[n_steps, NPROF, n_vars]`` array phase-resampled over one period) when the
        Julia run produced them (``po_results.profiles``); the actual periodic-orbit
        profile is otherwise not recorded by BifurcationKit.
        """
        import numpy as np
        from tvbo.adapters.julia import eval_with_auto_install
        from tvbo.analysis import BifurcationResult

        try:
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
        except Exception:
            return []

    def _extract_codim2_results(self, model, **kwargs) -> list:
        """Extract codim-2 continuation curves from Julia Main."""
        from tvbo.adapters.julia import eval_with_auto_install
        from tvbo.analysis import BifurcationResult

        try:
            c2 = eval_with_auto_install("codim2_results")
            results = []
            for entry in c2:
                br_obj = entry
                res = BifurcationResult(br=br_obj, model=model, **kwargs)
                res._is_codim2 = True

                # Infer source type from continuation kind
                from tvbo.analysis.bifurcation import continuation_kind

                kind = continuation_kind(br_obj)
                if kind == "HopfCont":
                    res._source_type = "hopf"
                elif kind == "FoldCont":
                    res._source_type = "fold"
                else:
                    res._source_type = "fold"

                # BifurcationKit codim-2 branches store both parameters
                # as named columns plus the 'param' column (= continuation
                # parameter). Identify both by matching model parameters.
                if not res.df.empty and model:
                    import numpy as np

                    model_params = set(model.parameters.keys()) if hasattr(model, "parameters") else set()
                    param_cols = [c for c in res.df.columns if c in model_params]
                    # The column whose values match 'param' is the
                    # codim-2 continuation parameter; the other is the
                    # co-parameter (param2).
                    res._ics_name = None
                    res._fp2_name = None
                    for col in param_cols:
                        if np.allclose(res.df[col].values, res.df["param"].values, equal_nan=True, rtol=1e-10):
                            res._ics_name = col
                        else:
                            res._fp2_name = col
                            res.df["param2"] = res.df[col]
                    # Fallback: if we couldn't match, use first param col
                    if res._fp2_name is None and len(param_cols) >= 2:
                        for col in param_cols:
                            if col != (res._ics_name or ""):
                                res._fp2_name = col
                                res.df["param2"] = res.df[col]
                                break

                results.append(res)
            return results
        except Exception:
            return []

    @staticmethod
    def _prepare_codim2_branch(br, parent_cont):
        """Pre-compute context for a codim-2 branch."""
        bc = br.continuation
        fp2 = getattr(bc, "free_parameters", None) or {}
        if isinstance(fp2, dict) and fp2:
            fp2_first = next(iter(fp2.values()))
        elif isinstance(fp2, list) and fp2:
            fp2_first = fp2[0]
        else:
            raise ValueError("Codim-2 branch requires free_parameters.")

        ICS2 = str(fp2_first.name)
        p2_min = float(fp2_first.domain.lo) if fp2_first.domain else -20
        p2_max = float(fp2_first.domain.hi) if fp2_first.domain else 20

        source = getattr(br, "source_point", None) or "hopf:all"
        source_type = source.split(":")[0]  # 'hopf' or 'fold'
        all_source = ":all" in source

        # Source index (Julia 1-based)
        source_idx_jl = None
        if not all_source and ":" in source:
            idx_str = source.split(":")[1]
            if idx_str.lstrip("-").isdigit():
                idx = int(idx_str)
                source_idx_jl = idx if idx >= 0 else f"end{idx + 1}" if idx != -1 else "end"

        # Codim-2 ContinuationPar args
        cp_args = [f"p_min = {p2_min}", f"p_max = {p2_max}"]
        cp_args.extend(_cont_kwargs(bc))
        if not any("ds =" in a for a in cp_args):
            cp_args.append("ds = 0.01")
        if not any("dsmax" in a.lower() for a in cp_args):
            cp_args.append("dsmax = 0.1")
        if not any("max_steps" in a for a in cp_args):
            cp_args.append("max_steps = 300")
        codim2_cp_str = ", ".join(cp_args)

        # Codim-2 continuation kwargs
        codim2_kw = [
            "normC = norminf",
            "detect_codim2_bifurcation = 2",
            "update_minaug_every_step = 1",
            "start_with_eigen = true",
            "verbosity = 0",
        ]
        if br.bothside:
            codim2_kw.append("bothside = true")
        codim2_kwargs_str = ",\n            ".join(codim2_kw)

        return {
            "name": str(br.name),
            "ICS2": ICS2,
            "p2_min": p2_min,
            "p2_max": p2_max,
            "source_type": source_type,
            "all_source": all_source,
            "source_idx_jl": source_idx_jl,
            "codim2_cp_str": codim2_cp_str,
            "codim2_kwargs_str": codim2_kwargs_str,
        }
