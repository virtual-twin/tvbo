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
    derived_vars = list((getattr(model, "derived_variables", None) or {}).keys())
    derived_params = list((getattr(model, "derived_parameters", None) or {}).keys())
    return sv, params, coupling, derived_vars, derived_params


def equation_rhs_text(model) -> str:
    """Concatenated RHS text of all SV / derived-variable / derived-parameter equations.

    Used to sniff which optional Julia packages the emitted model needs.
    """
    parts = []
    for sv in model.state_variables.values():
        parts.append(str(sv.equation.rhs))
    for dv in (getattr(model, "derived_variables", None) or {}).values():
        parts.append(str(dv.equation.rhs))
    for dp in (getattr(model, "derived_parameters", None) or {}).values():
        parts.append(str(dp.equation.rhs))
    return " ".join(parts)


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


def build_ifelse(cases, render) -> str:
    """Fold a list of conditional ``cases`` into nested Julia ``ifelse(...)`` calls.

    ``render`` turns an equation RHS into a Julia expression string. A case whose
    condition is ``true`` (or the final case) becomes the else branch.
    """
    if len(cases) == 1:
        return render(cases[0].equation.rhs)
    first = cases[0]
    cond = str(first.condition).strip()
    if cond.lower() == "true":
        return render(first.equation.rhs)
    return f"ifelse({cond}, {render(first.equation.rhs)}, {build_ifelse(cases[1:], render)})"


def make_renderer(model, fmt="julia"):
    """Return an ``expr -> str`` renderer bound to this model's symbol table."""
    sv, params, coupling, dvars, dparams = symbol_names(model)
    all_symbols = sv + params + coupling + dvars + dparams
    func_names = {str(f): str(f) for f in (getattr(model, "functions", None) or {})}
    return lambda expr: render_expression(
        expr, format=fmt, parameters=all_symbols, user_functions=func_names
    )


def build_model_context(model) -> dict:
    """Build the full DifferentialEquations.jl model-function context.

    Everything the ``tvbo-julia-model.jl.mako`` / ``tvbo-julia-ODEProblem.jl.mako``
    templates need is pre-rendered here so those templates only emit syntax.

    Multi-mode models (``number_of_modes > 1``) lay each state variable out as a
    contiguous length-n_modes block, so the dfun operates on per-mode vectors and
    writes vector slices (``dx[lo:hi] .= …``); scalar models keep the flat layout.
    """
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
        rhs = jl(s.equation.rhs)
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
    for fname, fdef in (getattr(model, "functions", None) or {}).items():
        fargs = [str(name) for name in fdef.arguments]
        functions.append((str(fname), fargs, jl(fdef.equation.rhs)))

    # Derived parameters and derived variables (conditional ones folded to ifelse).
    derived_params = [
        (dp.name, jl(dp.equation.rhs))
        for dp in (getattr(model, "derived_parameters", None) or {}).values()
    ]
    derived_vars = []
    for dv in (getattr(model, "derived_variables", None) or {}).values():
        if getattr(dv, "conditional", False) and getattr(dv, "cases", None):
            derived_vars.append((dv.name, build_ifelse(list(dv.cases), jl)))
        else:
            derived_vars.append((dv.name, jl(dv.equation.rhs)))

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
            u0.append(s.initial_value)

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
