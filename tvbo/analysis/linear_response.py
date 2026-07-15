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

Everything model-specific is **symbolic** (``sympy`` differentiation of the dfun,
with the derived-variable chain unfolded), so the same derivation renders to any
backend through the code generator. Only the generic assembly (block-diagonal
local Jacobian + connectome-scattered coupling Jacobian) and the linear-algebra
solves are numeric. This mirrors, on the JAX/tvboptim side, exactly what the
Julia network-continuation emitter builds — one metadata source, two backends.

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

    dvars = [
        (n, sp.sympify(dv.equation.rhs))
        for n, dv in (getattr(model, "derived_variables", {}) or {}).items()
    ]
    zero_local = {sp.Symbol(c): 0 for c in local_cpls}

    def resolve(rhs):
        expr = sp.sympify(rhs)
        for _ in range(len(dvars) + 2):  # unfold the (topologically ordered) chain
            expr = expr.subs({sp.Symbol(n): d for n, d in dvars})
        return expr.subs(zero_local)  # zero local coupling AFTER unfolding

    state_syms = [sp.Symbol(v) for v in svs]
    f = [resolve(model.state_variables[v].equation.rhs) for v in svs]
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


def network_jacobian(model, weights: Any, state: Any, params: dict) -> np.ndarray:
    """Assemble the full network Jacobian ``A`` at an operating point (numeric).

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
    pvals = list(params.values())

    # coupling per network input per node: c_i = Σ_j W_ij s_src,j
    C = {c: W @ Y[src_k] for c in net_cpls}

    A = np.zeros((n_sv * N, n_sv * N))
    for i in range(N):
        ci = [C[c][i] for c in net_cpls]
        jl = np.asarray(Jloc(*Y[:, i], *ci, *pvals), float).reshape(n_sv, n_sv)
        jc = np.asarray(Jcpl(*Y[:, i], *ci, *pvals), float).reshape(n_sv, len(net_cpls))
        for k in range(n_sv):
            for l in range(n_sv):
                A[k * N + i, l * N + i] += jl[k, l]  # local block (node-diagonal)
            for cix in range(len(net_cpls)):
                # ∂f_k/∂c · ∂c_i/∂s_src,j = jc[k]·W_ij  (source variable's column block)
                A[k * N + i, src_k * N : src_k * N + N] += jc[k, cix] * W[i, :]
    return A
