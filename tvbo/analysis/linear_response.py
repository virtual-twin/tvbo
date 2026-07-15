#
# Module: linear_response.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#

"""
Linear response
================
Symbolic linear-response machinery derived entirely from a model's declarative
metadata — the network Jacobian ``A`` at an operating point, and the noise input
matrix ``Q`` — from which fixed-point observables follow (stationary covariance
via the Lyapunov equation, power spectra, Fisher information; Deco 2014 Figs 5/6).

Everything model-specific is **symbolic** and backend-independent: :func:`jacobian_terms`
differentiates the dfun metadata with ``sympy`` (derived-variable chain unfolded) and
returns the symbolic per-node Jacobians, which the **code generator renders to any
backend** through ``render_expression``. The network assembly (block-diagonal local
Jacobian + connectome-scattered coupling Jacobian) is likewise **emitted by codegen
per backend** — a ``vmap``/scatter on JAX, a loop on Julia — exactly as the network RHS
is emitted (one metadata source, every backend); ideally through the backend-abstracted
``arrayops`` structural primitives so the assembly, too, is one handler.

:func:`network_jacobian` below is a **NumPy reference oracle only** — it assembles ``A``
numerically so the symbolic terms can be verified against a finite-difference Jacobian in
tests. It is NOT the runtime path (the runtime path is the codegen described above); do
not call it from generated code.

The full network Jacobian is

    A[(k,i),(l,j)] = δ_ij · ∂f_k/∂x_l                       (local block, per node)
                   + Σ_c (∂f_k/∂c) · (∂c_i/∂x_{l,j})         (coupling block)

where for an instantaneous coupling input ``c`` whose source state variable is
``s`` (``c_i = Σ_j W_ij s_j``), ``∂c_i/∂x_{l,j} = W_ij`` when ``l == s`` else 0.
Both ``∂f_k/∂x_l`` (``Jloc``) and ``∂f_k/∂c`` (``Jcpl``) are symbolic per-node
Jacobians of the metadata dfun.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp


def _dfun_symbols(model):
    """Return (state_syms, net_coupling_names, source_var, per-node f expressions).

    ``f`` are the state-variable RHS with the derived-variable chain fully
    unfolded and local (non-network) coupling inputs zeroed — so each ``f_k`` is
    expressed in state variables, network-coupling inputs, and parameters only.
    """
    svs = list(model.state_variables)
    cpl_inputs = dict(getattr(model, "coupling_inputs", {}) or {})
    net_cpls = [c for c, ci in cpl_inputs.items() if not getattr(ci, "local", False)]
    local_cpls = [c for c, ci in cpl_inputs.items() if getattr(ci, "local", False)]
    source_var = next(
        (n for n, sv in model.state_variables.items() if getattr(sv, "coupling_variable", False)),
        svs[0],
    )

    from tvbo.classes.equation import substitute_function_in_state_equations

    dvars = {n: sp.sympify(dv.equation.rhs)
             for n, dv in (getattr(model, "derived_variables", {}) or {}).items()}
    zero_local = {sp.Symbol(c): 0 for c in local_cpls}

    # Inline the derived-variable chain into the state equations with the codebase's
    # canonical inliner, iterated to unfold nested references (a derived var may
    # reference another), then zero local (non-network) coupling inputs.
    sv_eqs = {v: sp.sympify(model.state_variables[v].equation.rhs) for v in svs}
    for _ in range(len(dvars) + 1):
        substitute_function_in_state_equations(sv_eqs, dvars)
    state_syms = [sp.Symbol(v) for v in svs]
    f = [sv_eqs[v].subs(zero_local) for v in svs]
    return svs, state_syms, net_cpls, source_var, f


def jacobian_terms(model):
    """Symbolic per-node Jacobian terms of the metadata dfun.

    Returns a dict with the symbolic ``Jloc`` (∂f/∂state, ``n_sv × n_sv``) and
    ``Jcpl`` (∂f/∂net-coupling, ``n_sv × n_cpl``) sympy matrices plus the symbol
    ordering needed to lower them (state vars, network coupling names, the coupling
    source variable). Backend-independent — a printer turns these into code.
    """
    svs, state_syms, net_cpls, source_var, f = _dfun_symbols(model)
    cpl_syms = [sp.Symbol(c) for c in net_cpls]
    fmat = sp.Matrix(f)
    return {
        "state_vars": svs,
        "state_syms": state_syms,
        "net_couplings": net_cpls,
        "coupling_syms": cpl_syms,
        "source_var": source_var,
        "Jloc": fmat.jacobian(state_syms),
        "Jcpl": fmat.jacobian(cpl_syms) if cpl_syms else sp.zeros(len(svs), 0),
    }


def linear_response_context(model):
    """Resolution for the linear-response codegen: symbolic terms + layout, NO code.

    Keeps *resolution* in Python and *code structure* in the template, per the codegen
    convention. The Mako ``<%def>`` partials in ``_linear_response.py.mako`` consume this
    and emit the vector-field / Jacobian / covariance structure, rendering each symbolic
    entry with ``render_expression`` (so any backend prints it) — no Python string-emit.

    Returns the state / coupling / parameter layout — including which parameters are
    per-node (``pernode``: heterogeneous, gathered by node index) and the symbol set the
    printer must treat as plain symbols (``syms``) — plus the symbolic per-node RHS
    (``rhs``), local Jacobian (``Jloc``) and coupling Jacobian (``Jcpl``).
    """
    t = jacobian_terms(model)
    svs, net_cpls, src = t["state_vars"], t["net_couplings"], t["source_var"]
    pnames = [p.name for p in model.parameters.values()]
    pernode = {p.name for p in model.parameters.values() if getattr(p, "heterogeneous", False)}
    return {
        "svs": svs,
        "net_cpls": net_cpls,
        "src_k": svs.index(src),
        "n_sv": len(svs),
        "n_cpl": len(net_cpls),
        "pnames": pnames,
        "pernode": pernode,
        "syms": svs + net_cpls + pnames,
        "rhs": _dfun_symbols(model)[4],
        "Jloc": t["Jloc"],
        "Jcpl": t["Jcpl"],
    }


def network_jacobian(model, weights: Any, state: Any, params: dict) -> np.ndarray:
    """NumPy **reference oracle** — assemble ``A`` numerically for verification only.

    Used by tests to check the symbolic :func:`jacobian_terms` against a
    finite-difference Jacobian. The runtime path renders those symbolic terms to the
    target backend and assembles ``A`` in codegen (``vmap``/scatter on JAX, loop on
    Julia); this function is deliberately NumPy and must not be called from generated
    code.

    Parameters
    ----------
    model : Dynamics
        The model (source of the symbolic dfun).
    weights : array (n_nodes, n_nodes)
        Connectome ``W`` (``c_i = Σ_j W_ij s_j`` for the coupling source ``s``).
    state : array (n_sv, n_nodes)
        The operating point (e.g. the deterministic fixed point), per state
        variable and node.
    params : dict
        Scalar parameter values by name.

    Returns
    -------
    A : array (n_sv·n_nodes, n_sv·n_nodes)
        The Jacobian in block layout (state-variable block ``k`` spans rows/cols
        ``k·N .. (k+1)·N``), matching the network state layout used elsewhere.
    """
    t = jacobian_terms(model)
    svs, net_cpls, src = t["state_vars"], t["net_couplings"], t["source_var"]
    n_sv = len(svs)
    W = np.asarray(weights, float)
    Y = np.asarray(state, float).reshape(n_sv, W.shape[0])
    N = W.shape[0]
    src_k = svs.index(src)

    arg_syms = t["state_syms"] + t["coupling_syms"] + [sp.Symbol(p) for p in params]
    Jloc = sp.lambdify(arg_syms, t["Jloc"], "numpy")
    Jcpl = sp.lambdify(arg_syms, t["Jcpl"], "numpy")

    # coupling per network input per node: c_i = Σ_j W_ij s_src,j
    C = {c: W @ Y[src_k] for c in net_cpls}

    A = np.zeros((n_sv * N, n_sv * N))
    for i in range(N):
        ci = [C[c][i] for c in net_cpls]
        # per-node (array-valued) params are indexed by node; scalars pass through
        pvals_i = [np.asarray(v)[i] if np.ndim(v) > 0 else v for v in params.values()]
        jl = np.asarray(Jloc(*Y[:, i], *ci, *pvals_i), float).reshape(n_sv, n_sv)
        jc = np.asarray(Jcpl(*Y[:, i], *ci, *pvals_i), float).reshape(n_sv, len(net_cpls))
        for k in range(n_sv):
            for l in range(n_sv):
                A[k * N + i, l * N + i] += jl[k, l]  # local block (node-diagonal)
            for cix in range(len(net_cpls)):
                # ∂f_k/∂c · ∂c_i/∂s_src,j = jc[k]·W_ij  (source variable's column block)
                A[k * N + i, src_k * N : src_k * N + N] += jc[k, cix] * W[i, :]
    return A
