# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2


"""Linear response.

Symbolic linear-response machinery derived entirely from a model's declarative metadata — the network Jacobian ``A`` at an operating point, and the noise input matrix ``Q`` — from which fixed-point observables follow (stationary covariance via the Lyapunov equation, power spectra, Fisher information; Deco 2014 Figs 5/6).

Everything model-specific is **symbolic** and backend-independent: :func:`jacobian_terms` differentiates the dfun metadata with ``sympy`` (derived-variable chain unfolded) and returns the symbolic per-node Jacobians, which the **code generator renders to any backend** through ``render_expression``. The network assembly (block-diagonal local Jacobian + connectome-scattered coupling Jacobian) is likewise **emitted by codegen per backend** — a ``vmap``/scatter on JAX, a loop on Julia — exactly as the network RHS is emitted (one metadata source, every backend); ideally through the backend-abstracted ``arrayops`` structural primitives so the assembly, too, is one handler.

:func:`network_jacobian` below is a **NumPy reference oracle only** — it assembles ``A`` numerically so the symbolic terms can be verified against a finite-difference Jacobian in tests. It is NOT the runtime path (the runtime path is the codegen described above); do not call it from generated code.

The full network Jacobian is

    A[(k,i),(l,j)] = δ_ij · ∂f_k/∂x_l                       (local block, per node)
                   + Σ_c (∂f_k/∂c) · (∂c_i/∂x_{l,j})         (coupling block)

where for an instantaneous coupling input ``c`` whose source state variable is ``s`` (``c_i = Σ_j W_ij s_j``), ``∂c_i/∂x_{l,j} = W_ij`` when ``l == s`` else 0.
Both ``∂f_k/∂x_l`` (``Jloc``) and ``∂f_k/∂c`` (``Jcpl``) are symbolic per-node Jacobians of the metadata dfun.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sympy as sp


def _coupling_split(model):
    """``(network, local)``: the model's coupling-input names, split by whether the input is local (non-network)."""
    cpl_inputs = dict(getattr(model, "coupling_inputs", {}) or {})
    local = [c for c, ci in cpl_inputs.items() if getattr(ci, "local", False)]
    return [c for c in cpl_inputs if c not in local], local


def _parsed(model, group):
    """``{name: rhs}`` of one equation group of the model's cached symbolic form, parsed once against the model's own symbol scope (so a name like ``gamma`` is a Symbol, not a function)."""
    return {name: eq.rhs for name, eq in model.symbolic_system.form(notation="symbol")[group].items()}


def _unfolded(model, expressions):
    """*expressions* (``{name: expr}``) with the derived-variable chain inlined and local (non-network) coupling inputs zeroed.

    Each result is in state variables, network-coupling inputs and parameters only. The chain is inlined with the codebase's canonical inliner, once per derived variable plus one, so a derived variable that reads another unfolds completely.
    """
    from tvbo.classes.equation import substitute_function_in_state_equations

    dvars = _parsed(model, "derived-variables")
    out = dict(expressions)
    for _ in range(len(dvars) + 1):
        substitute_function_in_state_equations(out, dvars)
    zero_local = {sp.Symbol(c): 0 for c in _coupling_split(model)[1]}
    return {name: expr.subs(zero_local) for name, expr in out.items()}


def _dfun_symbols(model):
    """Return (state_vars, state_syms, net_coupling_names, source_var, per-node f expressions).

    ``f`` are the state-variable RHS through :func:`_unfolded` — so each ``f_k`` is expressed in state variables, network-coupling inputs, and parameters only.
    """
    svs = list(model.state_variables)
    net_cpls, _ = _coupling_split(model)
    source_var = next(
        (n for n, sv in model.state_variables.items() if getattr(sv, "coupling_variable", False)),
        svs[0],
    )
    rhs = _parsed(model, "state-equations")
    f = _unfolded(model, {v: rhs[v] for v in svs})
    return svs, [sp.Symbol(v) for v in svs], net_cpls, source_var, [f[v] for v in svs]


def jacobian_terms(model):
    """Symbolic per-node Jacobian terms of the metadata dfun.

    Returns a dict with the symbolic ``Jloc`` (∂f/∂state, ``n_sv × n_sv``) and ``Jcpl`` (∂f/∂net-coupling, ``n_sv × n_cpl``) sympy matrices plus the symbol ordering needed to lower them (state vars, network coupling names, the coupling source variable). Backend-independent — a printer turns these into code.
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
        "rhs": f,  # per-node RHS (derived-var chain unfolded, local coupling zeroed)
        "Jloc": fmat.jacobian(state_syms),
        "Jcpl": fmat.jacobian(cpl_syms) if cpl_syms else sp.zeros(len(svs), 0),
    }


def constraint_expr(model, var_name):
    """Unfolded symbolic expression of a derived variable (e.g. the FIC constraint variable ``I_E``), in state variables, network-coupling inputs and parameters — same unfolding as :func:`_dfun_symbols` uses for the RHS (derived-variable chain inlined, local coupling zeroed), so it prints against the same symbol set (``ctx['syms']``). Used to emit the constraint residual of a constraint-defined operating point (Deco FIC: ``I_E = target``, with ``J_i`` the free parameter), solved deterministically alongside the fixed point."""
    dvars = _parsed(model, "derived-variables")
    if var_name not in dvars:
        raise KeyError(f"constraint variable '{var_name}' is not a derived variable of the model")
    return _unfolded(model, {var_name: dvars[var_name]})[var_name]


def observable_terms(model, name):
    """Symbolic observation row ``∂y/∂x`` of a declared observable ``y``.

    ``y`` is either a state variable (the row is a selector) or a derived variable — a BOLD signal, a firing rate, any declared readout — unfolded through the same derived-variable chain the RHS uses, so the linear response can be carried through whatever cascade the model declares rather than stopping at the state vector.

    Returns the per-node Jacobian of ``y`` with respect to the state variables (``Hloc``, ``1 × n_sv``) and with respect to the network coupling inputs (``Hcpl``, ``1 × n_cpl``); the latter scatters through the connectome exactly as ``Jcpl`` does, so an observable reading a coupling term stays correct.
    """
    svs = list(model.state_variables)
    state_syms = [sp.Symbol(v) for v in svs]
    if name in svs:
        expr = sp.Symbol(name)
    else:
        expr = constraint_expr(model, name)
    row = sp.Matrix([expr])
    cpl_syms = [sp.Symbol(c) for c in _coupling_split(model)[0]]
    return {
        "Hloc": row.jacobian(state_syms),
        "Hcpl": row.jacobian(cpl_syms) if cpl_syms else sp.zeros(1, 0),
    }


def noise_terms(model):
    """Per-state-variable noise standard deviations declared on the model, or ``None``.

    The Lyapunov equation's input matrix ``Q`` is ``diag(σ_k²)`` over the state blocks, which is only ``σ² I`` when every state variable is driven. A model whose noise enters two of six equations — two synaptic gating variables and a four-state haemodynamic cascade that is driven, not forced — needs the declared per-state amplitudes, and a uniform ``Q`` would put noise into the haemodynamics.

    Returns ``None`` when no state variable declares noise at all, which is the signal to fall back to a uniform amplitude supplied by the analysis observation.
    """
    from tvbo.utils import noise_sigma

    sigmas = [noise_sigma(getattr(sv, "noise", None)) for sv in model.state_variables.values()]
    if all(s is None for s in sigmas):
        return None
    return [0.0 if s is None else float(s) for s in sigmas]


def linear_response_context(model):
    """Resolution for the linear-response codegen: symbolic terms + layout, NO code.

    Keeps *resolution* in Python and *code structure* in the template, per the codegen convention. The Mako ``<%def>`` partials in ``_linear_response.py.mako`` consume this and emit the vector-field / Jacobian / covariance structure, rendering each symbolic entry with ``render_expression`` (so any backend prints it) — no Python string-emit.

    Returns the state / coupling / parameter layout — including which parameters are per-node (``pernode``: heterogeneous, gathered by node index) and the symbol set the printer must treat as plain symbols (``syms``) — plus the symbolic per-node RHS (``rhs``), local Jacobian (``Jloc``), coupling Jacobian (``Jcpl``) and the declared per-state noise amplitudes (``noise``, or ``None`` when the model declares none).
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
        "rhs": t["rhs"],
        "Jloc": t["Jloc"],
        "Jcpl": t["Jcpl"],
        "noise": noise_terms(model),
    }


def network_jacobian(model, weights: Any, state: Any, params: dict) -> np.ndarray:
    """NumPy **reference oracle** — assemble ``A`` numerically for verification only.

    Used by tests to check the symbolic :func:`jacobian_terms` against a finite-difference Jacobian. The runtime path renders those symbolic terms to the target backend and assembles ``A`` in codegen (``vmap``/scatter on JAX, loop on Julia); this function is deliberately NumPy and must not be called from generated code.

    Parameters
    ----------
    model : Dynamics
        The model (source of the symbolic dfun).
    weights : array (n_nodes, n_nodes)
        connectome ``W`` (``c_i = Σ_j W_ij s_j`` for the coupling source ``s``).
    state : array (n_sv, n_nodes)
        The operating point (e.g. the deterministic fixed point), per state
        variable and node.
    params : dict
        Scalar parameter values by name.

    Returns:
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
            for m in range(n_sv):
                A[k * N + i, m * N + i] += jl[k, m]  # local block (node-diagonal)
            for cix in range(len(net_cpls)):
                # ∂f_k/∂c · ∂c_i/∂s_src,j = jc[k]·W_ij  (source variable's column block)
                A[k * N + i, src_k * N : src_k * N + N] += jc[k, cix] * W[i, :]
    return A
