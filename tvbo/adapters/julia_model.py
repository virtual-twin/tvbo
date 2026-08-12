# -*- coding: utf-8 -*-
"""Prepared codegen context for the Julia model templates.

The Julia backends (DifferentialEquations.jl, NetworkDynamics.jl, ModelingToolkit.jl)
share the same *metadata → Julia* translation logic: which state variables/parameters
become symbols, which optional packages are needed, how conditional derived variables
fold into ``ifelse``, and how multi-mode models lay their state out along a mode axis.

This module owns that logic so the Mako templates stay slim — they only emit syntax
from the dict returned by :func:`build_model_context` (and the small helpers here).
Mirrors the "resolve in Python, not Mako" convention used by the other adapters.
"""

from __future__ import annotations

import re

from tvbo.codegen import render_expression
from tvbo.parse.expression import parse_eq, states_an_expression
from tvbo.utils import initial_value

# Solver name → the minimal OrdinaryDiffEq sub-package that provides it. Splitting
# out the umbrella package keeps Julia precompilation cheap / avoids Bus errors.
JULIA_SOLVER_PACKAGES = {
    "Tsit5": "OrdinaryDiffEqTsit5",
    "AutoTsit5": "OrdinaryDiffEqTsit5",
    "DP5": "OrdinaryDiffEqTsit5",
    "Heun": "OrdinaryDiffEqLowOrderRK",
    "Euler": "OrdinaryDiffEqLowOrderRK",
    "Midpoint": "OrdinaryDiffEqLowOrderRK",
    "RK4": "OrdinaryDiffEqLowOrderRK",
    "BS3": "OrdinaryDiffEqLowOrderRK",
    "Vern7": "OrdinaryDiffEqVerner",
    "Rodas5": "OrdinaryDiffEqRosenbrock",
    "TRBDF2": "OrdinaryDiffEqSDIRK",
}

# Elementary functions that require ``using SpecialFunctions`` in Julia.
JULIA_SPECIAL_FUNCTIONS = (
    "erf", "erfc", "erfi", "erfcx", "lgamma", "digamma",
    "beta", "lbeta", "besselj", "bessely", "besseli", "gamma",
)


def julia_ode_package(solver_method) -> str:
    """OrdinaryDiffEq sub-package providing ``solver_method`` (default: umbrella)."""
    return JULIA_SOLVER_PACKAGES.get(str(solver_method), "OrdinaryDiffEq")


def symbol_names(model):
    """Return ``(sv, params, coupling, derived_vars, derived_params)`` name lists.

    These are exactly the names the Julia expression printer must treat as bare
    symbols rather than trying to resolve.
    """
    sv = list(model.state_variables.keys())
    params = list((model.parameters or {}).keys())
    coupling = list((model.coupling_terms or {}).keys()) if model.coupling_terms else []
    derived_vars = list((model.derived_variables).keys())
    derived_params = list((model.derived_parameters).keys())
    return sv, params, coupling, derived_vars, derived_params


def parse_namespace(model) -> dict:
    """How this model's equations parse, as keyword arguments for `parse_eq`.

    Both flavours of model reach this adapter — the runtime `Dynamics` in `tvbo.classes`
    and the generated one a heterogeneous node's inline dynamics is — and both carry the
    symbolic layer, so both are parsed against the table that model's own equations were.
    """
    return {"local_dict": model.get_symbolic_elements()}


def equation_rhs_text(model) -> str:
    """Concatenated text of every SV / derived-variable / derived-parameter equation.

    Used to sniff which optional Julia packages the emitted model needs. Resolved through
    the parser rather than read off `.rhs`, because an equation stated purely as
    conditional branches has no `rhs` to read: the sniff would see `"None"`, miss the
    `Piecewise` it contains, and emit a model that uses NaNMath without importing it.
    """
    namespace = parse_namespace(model)
    collections = (model.state_variables, model.derived_variables, model.derived_parameters)
    return " ".join(
        str(parse_eq(element.equation, **namespace))
        for collection in collections
        for element in collection.values()
        if states_an_expression(element.equation)
    )


def needs_special_functions(model) -> bool:
    """True if any equation calls a SpecialFunctions.jl function (erf, gamma, …)."""
    rhs = equation_rhs_text(model)
    return any(re.search(rf"\b{fn}\s*\(", rhs) for fn in JULIA_SPECIAL_FUNCTIONS)


def needs_nanmath(model) -> bool:
    """True if any equation contains a ``Piecewise``.

    The Julia printer routes domain-restricted powers inside Piecewise branches
    through NaNMath (NaN instead of DomainError, matching numpy/JAX), so those
    models must ``import NaNMath``.
    """
    return "Piecewise" in equation_rhs_text(model)


def make_renderer(model, fmt="julia"):
    """Return a renderer bound to this model's symbol table.

    Takes an `Equation` as readily as an expression or a string, resolving it through the
    one parser, so a caller never has to know whether the equation states itself as a
    right-hand side or as conditional branches. Reaching for `.rhs` at the call site works
    only for the first kind and yields `None` for the second.
    """
    sv, params, coupling, dvars, dparams = symbol_names(model)
    all_symbols = sv + params + coupling + dvars + dparams
    func_names = {str(f): str(f) for f in (model.functions)}
    namespace = parse_namespace(model)

    def render(equation):
        return render_expression(
            parse_eq(equation, **namespace),
            format=fmt, parameters=all_symbols, user_functions=func_names,
        )

    return render


def _build_network_context(model, network, n_nodes, constraints=None) -> dict:
    """Network-coupled variant of :func:`build_model_context` (loop-based RHS).

    State is ``n_nodes`` blocks per state variable (block ``k`` spans indices
    ``k*N+1 .. (k+1)*N``). Each long-range coupling term becomes a connectivity
    matvec evaluated once per step (``_coup_<c> = W_NET · s_view``); the per-node
    scalar RHS (the same expressions the single-node emitter produces) runs inside
    a ``for i in 1:N`` loop. ``local`` coupling inputs are zero in a region sim.

    ``constraints`` promotes constraint-defined free parameters (e.g. the FIC
    ``J_i``) from parameters to extra unknown STATE blocks, appended after the real
    state. Each block's defining equation is the ``TuningObjective`` residual
    (``target_variable − target_value``), not an ODE: at equilibrium the residual
    is zero so the constraint holds, and during the initial-state warm-up the same
    residual is stabilising negative feedback (``target_variable`` ↑ ⇒ free param ↑
    ⇒ inhibition ↑ ⇒ ``target_variable`` ↓), so no separate solver is needed. This
    is the FIC branch of Deco 2014 Fig 2c. See ``DEV_PLAN_recipe_native.md`` (D2).
    Each constraint is ``{"parameter", "target_variable", "target_value"}``.
    """
    import numpy as np

    sv, params, coupling, _dvars, _dparams = symbol_names(model)
    jl = make_renderer(model, "julia")
    arg_x = "_x" if "x" in sv else "x"

    constraints = list(constraints or [])
    free_names = {c["parameter"] for c in constraints}

    # Connectivity matrix as a Julia literal (rows ';'-separated).
    W = np.asarray(network.weights_matrix, dtype=float)
    if W.shape != (n_nodes, n_nodes):
        raise ValueError(f"weights_matrix shape {W.shape} != ({n_nodes}, {n_nodes})")
    w_const = "[" + ";\n ".join(" ".join(repr(float(v)) for v in row) for row in W) + "]"

    # Coupling source = the state variable flagged coupling_variable (fallback: sv[0]).
    csv_k = 0
    for k, s in enumerate(model.state_variables.values()):
        if getattr(s, "coupling_variable", False):
            csv_k = k
            break
    csv_lo, csv_hi = csv_k * n_nodes + 1, (csv_k + 1) * n_nodes

    # Per coupling term: local → 0.0 per node; long-range → W·s matvec once + gather.
    cinputs = model.coupling_inputs
    coupling_pre, coupling_body = [], []
    for c in coupling:
        ci = cinputs.get(c) if hasattr(cinputs, "get") else None
        if ci is not None and bool(getattr(ci, "local", False)):
            coupling_body.append(f"{c} = 0.0")
        else:
            coupling_pre.append(f"_coup_{c} = W_NET * (@view {arg_x}[{csv_lo}:{csv_hi}])")
            coupling_body.append(f"{c} = _coup_{c}[i]")

    # Per-node state unpack and derivative LHS (block index k*N + i; k=0 ⇒ i).
    unpack, dfun = [], []
    for k, (name, s) in enumerate(model.state_variables.items()):
        idx = "i" if k == 0 else f"{k * n_nodes} + i"
        unpack.append(f"{name} = {arg_x}[{idx}]")
        dfun.append((f"dx[{idx}] =", jl(s.equation)))

    functions = []
    for fname, fdef in (model.functions).items():
        functions.append((str(fname), [str(a) for a in fdef.arguments], jl(fdef.equation)))
    derived_params = [
        (dp.name, jl(dp.equation))
        for dp in (model.derived_parameters).values()
    ]
    derived_vars = [(dv.name, jl(dv.equation)) for dv in (model.derived_variables).values()]

    # Parameters. A heterogeneous per-node parameter (value is a length-n_nodes
    # array, e.g. the FIC-tuned J_i) is emitted as a Julia vector ``<name>_vec``
    # and gathered per node (``<name> = <name>_vec[i]``) at the top of the loop;
    # scalar parameters (incl. a scalar default on a ``(n_nodes,)`` slot) stay scalar.
    pval_parts, destructure_names, pernode_gather = [], [], []
    for p in model.parameters.values():
        if p.name in free_names:
            continue  # promoted to an unknown state block below (constraint-defined)
        v = p.value
        is_pernode = hasattr(v, "__len__") and not isinstance(v, str) and len(v) == n_nodes
        if is_pernode:
            vec = "[" + ", ".join(repr(float(x)) for x in v) + "]"
            pval_parts.append(f"{p.name}_vec = {vec}")
            destructure_names.append(f"{p.name}_vec")
            pernode_gather.append(f"{p.name} = {p.name}_vec[i]")
        else:
            pval_parts.append(f"{p.name} = {v}")
            destructure_names.append(p.name)
    param_values = ", ".join(pval_parts) + ("," if len(pval_parts) == 1 else "")
    destructure = ", ".join(destructure_names) + ("," if len(destructure_names) == 1 else "")

    u0 = []
    for s in model.state_variables.values():
        u0.extend([initial_value(s)] * n_nodes)

    # Observables to record along the branch: every state variable plus any
    # derived variable listed in the model's ``output`` (e.g. the firing rate
    # H_e). Each is reduced across nodes to a max and a mean (see the record hook
    # in the BifurcationKit template), so the branch plots e.g. max r_E vs G.
    dv_names = {name for name, _ in derived_vars}
    record_obs = list(sv)
    for o in [str(o) for o in (model.output)]:
        if o in dv_names and o not in record_obs:
            record_obs.append(o)

    # ── Constraint-defined free-parameter blocks (FIC J_i etc.) ──
    # Appended after the real state blocks: unpacked from the state vector, given a
    # residual "dfun" (target_variable − target_value), seeded in u0 from the
    # declared value, and recorded along the branch (so J_i(G) is available).
    n_sv = len(model.state_variables)
    for j, c in enumerate(constraints):
        pname, tv, tval = c["parameter"], str(c["target_variable"]), float(c["target_value"])
        idx = f"{(n_sv + j) * n_nodes} + i"  # block after the real state (always ≥ 1)
        unpack.append(f"{pname} = {arg_x}[{idx}]")
        # Defining equation: residual → 0. Rendered through the sympy printer (like
        # every other dfun rhs) rather than string-formatted, for consistent emission.
        dfun.append((f"dx[{idx}] =", jl(f"{tv} - {tval}")))
        pv = model.parameters[pname].value
        if hasattr(pv, "__len__") and not isinstance(pv, str) and len(pv) == n_nodes:
            u0.extend([float(x) for x in pv])
        else:
            u0.extend([(float(pv) if pv is not None and not hasattr(pv, "__len__") else 1.0)] * n_nodes)
        for name in (pname, tv):
            if name not in record_obs:
                record_obs.append(name)

    return {
        "func_name": model.name,
        "arg_x": arg_x,
        "destructure": destructure,
        "needs_special": needs_special_functions(model),
        "needs_nanmath": needs_nanmath(model),
        "functions": functions,
        "derived_params": derived_params,
        "derived_vars": derived_vars,
        "unpack": unpack,
        "dfun": dfun,
        "param_values": param_values,
        "u0": u0,
        "n_modes": 1,
        "network_mode": True,
        "n_nodes": n_nodes,
        "w_const": w_const,
        "coupling_pre": coupling_pre,
        "coupling_body": coupling_body,
        "pernode_gather": pernode_gather,
        "record_obs": record_obs,
    }


def build_model_context(model, network=None, constraints=None) -> dict:
    """Build the full DifferentialEquations.jl model-function context.

    Everything the ``tvbo-julia-model.jl.mako`` / ``tvbo-julia-ODEProblem.jl.mako``
    templates need is pre-rendered here so those templates only emit syntax.

    Multi-mode models (``number_of_modes > 1``) lay each state variable out as a
    contiguous length-n_modes block, so the dfun operates on per-mode vectors and
    writes vector slices (``dx[lo:hi] .= …``); scalar models keep the flat layout.

    When ``network`` is supplied (a multi-node ``Network``), the model is emitted
    as a coupled network: the state is laid out as ``n_nodes`` blocks per state
    variable, each long-range coupling term becomes a connectivity matvec
    (``c = W · s_coupling``) evaluated once per step, and the per-node scalar RHS
    runs inside a ``for i in 1:N`` loop — reusing the single-node equation emission
    verbatim (no vectorised broadcasting). This is the vector field a whole-brain
    equilibrium/periodic-orbit continuation (e.g. Deco 2014 Fig 2c) continues in G.
    """
    n_nodes = int(getattr(network, "number_of_nodes", 0) or 0) if network is not None else 0
    if n_nodes > 1:
        return _build_network_context(model, network, n_nodes, constraints=constraints)

    if constraints:
        # Constraint-defined free params (FIC J_i) are only implemented for the
        # coupled network path; silently ignoring them here would emit an untuned
        # (wrong) branch. Fail loudly instead.
        raise NotImplementedError(
            "Constraint-defined free parameters (e.g. FIC J_i) require a multi-node "
            "network continuation; single-node constraint continuation is not implemented."
        )

    sv, params, coupling, _dvars, _dparams = symbol_names(model)
    jl = make_renderer(model, "julia")

    n_sv = len(sv)
    n_modes = getattr(model, "number_of_modes", 1) or 1
    # Rename the function's state argument if a state variable is literally named 'x'.
    arg_x = "_x" if "x" in sv else "x"

    # State unpacking + derivative LHS/RHS.
    unpack = []
    dfun = []
    for i, (name, s) in enumerate(model.state_variables.items()):
        rhs = jl(s.equation)
        if n_modes > 1:
            lo, hi = i * n_modes + 1, (i + 1) * n_modes
            unpack.append(f"{name} = @view {arg_x}[{lo}:{hi}]")
            dfun.append((f"dx[{lo}:{hi}] .=", rhs))
        else:
            dfun.append((f"dx[{i + 1}] =", rhs))
    if n_modes == 1:
        if n_sv == 1:
            unpack = [f"{sv[0]} = {arg_x}[1]"]
        else:
            unpack = [f"{', '.join(sv)} = {arg_x}"]

    # Custom functions (e.g. Sigm): (name, [args], body).
    functions = []
    for fname, fdef in (model.functions).items():
        fargs = [str(name) for name in fdef.arguments]
        functions.append((str(fname), fargs, jl(fdef.equation)))

    # Derived parameters and derived variables (conditional ones folded to ifelse).
    derived_params = [
        (dp.name, jl(dp.equation))
        for dp in (model.derived_parameters).values()
    ]
    derived_vars = [(dv.name, jl(dv.equation)) for dv in (model.derived_variables).values()]

    # `p = (...)` parameter tuple (coupling terms default to 0.0 for single-node).
    pval_parts = [f"{p.name} = {p.value}" for p in model.parameters.values()]
    pval_parts += [f"{c} = 0.0" for c in coupling]
    param_values = ", ".join(pval_parts) + ("," if len(pval_parts) == 1 else "")

    # NamedTuple destructuring on the parameter struct.
    destructure_names = params + coupling
    destructure = ", ".join(destructure_names) + ("," if len(destructure_names) == 1 else "")

    # Initial conditions, mode-expanded (each SV repeated n_modes times).
    u0 = []
    for s in model.state_variables.values():
        for _ in range(n_modes if n_modes > 1 else 1):
            u0.append(initial_value(s))

    return {
        "func_name": model.name,
        "arg_x": arg_x,
        "destructure": destructure,
        "needs_special": needs_special_functions(model),
        "needs_nanmath": needs_nanmath(model),
        "functions": functions,
        "derived_params": derived_params,
        "derived_vars": derived_vars,
        "unpack": unpack,
        "dfun": dfun,
        "param_values": param_values,
        "u0": u0,
        "n_modes": n_modes,
    }
