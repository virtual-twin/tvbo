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


def _emit_context(model, fmt):
    """Shared emission context for the vector-field and Jacobian codegen.

    Returns the symbol layout plus ready-made line builders — an ``entry`` renderer
    (one symbolic expression → backend code), the per-node function ``header``/unpack
    (state from ``s``, coupling from ``c``, params from ``p_`` with per-node gather),
    and the ``coupling`` setup (``_W``/``_X``/``_N``/``_C``) — so the two emitters below
    share one definition of how metadata symbols become code.
    """
    from tvbo.codegen import render_expression

    if fmt != "jax":
        raise NotImplementedError(f"linear_response: backend {fmt!r} not yet emitted (jax only).")
    t = jacobian_terms(model)
    svs, net_cpls, src = t["state_vars"], t["net_couplings"], t["source_var"]
    src_k = svs.index(src)
    n_sv, n_cpl = len(svs), len(net_cpls)
    pnames = [p.name for p in model.parameters.values()]
    # Heterogeneous (per-node) params — e.g. the FIC-tuned J_i, shape (n_nodes,) — are
    # gathered by the node index inside the per-node function; scalar params are not.
    pernode = {p.name for p in model.parameters.values() if getattr(p, "heterogeneous", False)}
    syms = svs + net_cpls + pnames

    def entry(expr):
        return render_expression(expr, format="jax", parameters=syms)

    def node_unpack():
        lines = [f"    {v} = s[{i}]" for i, v in enumerate(svs)]
        lines += [f"    {c} = c[{i}]" for i, c in enumerate(net_cpls)]
        lines += [
            (f"    {p} = jnp.asarray(getattr(p_, '{p}'))[_i]" if p in pernode
             else f"    {p} = getattr(p_, '{p}')")
            for p in pnames
        ]
        return lines

    def coupling_setup():
        return [
            "    _W = jnp.asarray(weights); _X = jnp.asarray(x); _N = _W.shape[0]",
            (f"    _C = jnp.stack([_W @ _X[{src_k}] for _ in range({n_cpl})])" if n_cpl
             else "    _C = jnp.zeros((0, _N))"),
        ]

    return dict(terms=t, svs=svs, net_cpls=net_cpls, src_k=src_k, n_sv=n_sv, n_cpl=n_cpl,
                entry=entry, node_unpack=node_unpack, coupling_setup=coupling_setup)


def render_vf_code(model, func_name: str = "_lr_vf", fmt: str = "jax") -> str:
    """Emit backend code for the deterministic network vector field ``dy/dt = f(y)``.

    The per-node RHS (dfun with the derived-variable chain unfolded and local coupling
    zeroed, from :func:`jacobian_terms`) is rendered via ``render_expression`` and
    ``vmap``-ed over nodes; long-range coupling is the connectome matvec. Emits

        ``<func_name>(x, weights, p) -> dy/dt``   (both [n_sv, N])

    — noise-free by construction, so settling it (or a Newton step) gives the
    deterministic operating point the linear-response observables are evaluated at.
    """
    c = _emit_context(model, fmt)
    rhs = [c["entry"](e) for e in _dfun_symbols(model)[4]]  # per-node RHS expressions f(state, coupling, params)
    lines = [f"def {func_name}_node(s, c, p_, _i):", *c["node_unpack"](),
             f"    return jnp.array([{', '.join(rhs)}])", ""]
    lines += [f"def {func_name}(x, weights, p):", *c["coupling_setup"](),
              f"    _F = jax.vmap(lambda i: {func_name}_node(_X[:, i], _C[:, i], p, i))(jnp.arange(_N))",
              "    return _F.T"]  # vmap stacks nodes on axis 0 → transpose to [n_sv, N]
    return "\n".join(lines)


def render_jacobian_code(model, func_name: str = "_lr_jacobian", fmt: str = "jax") -> str:
    """Emit backend code that builds the network Jacobian ``A`` at an operating point.

    Renders the symbolic per-node Jacobians (:func:`jacobian_terms`) to ``fmt`` via
    ``render_expression``, and emits the assembly — ``vmap`` over nodes for the local
    block-diagonal plus a connectome scatter for the coupling block. The result is a
    self-contained function

        ``<func_name>(x, weights, p) -> A``   (``x``: [n_sv, N], ``A``: [n_sv·N, n_sv·N])

    that generated analysis code calls with the fixed point, the connectome, and the
    resolved parameter bunch ``p``. Backend-independent by construction: only the entry
    expressions (via the printer) and the array vocabulary differ per backend; ``fmt='jax'``
    emits ``jnp``. No numpy in the emitted code.
    """
    c = _emit_context(model, fmt)
    svs, net_cpls, src_k, n_sv, n_cpl = c["svs"], c["net_cpls"], c["src_k"], c["n_sv"], c["n_cpl"]
    t = c["terms"]

    def _matrix_fn(name, M, ncol):
        rows = ", ".join(
            "[" + ", ".join(c["entry"](M[k, l]) for l in range(ncol)) + "]" for k in range(n_sv)
        )
        return [f"def {name}(s, c, p_, _i):", *c["node_unpack"](), f"    return jnp.array([{rows}])", ""]

    lines: list[str] = []
    lines += _matrix_fn(f"{func_name}_jloc", t["Jloc"], n_sv)
    lines += _matrix_fn(f"{func_name}_jcpl", t["Jcpl"], n_cpl) if n_cpl else []
    lines += [f"def {func_name}(x, weights, p):", *c["coupling_setup"](),
              f"    _Jl = jax.vmap(lambda i: {func_name}_jloc(_X[:, i], _C[:, i], p, i))(jnp.arange(_N))"]
    if n_cpl:
        lines.append(f"    _Jc = jax.vmap(lambda i: {func_name}_jcpl(_X[:, i], _C[:, i], p, i))(jnp.arange(_N))")
    lines.append(f"    _A = jnp.zeros(({n_sv} * _N, {n_sv} * _N))")
    for k in range(n_sv):
        for l in range(n_sv):
            lines.append(
                f"    _A = _A.at[{k}*_N:{k+1}*_N, {l}*_N:{l+1}*_N].add(jnp.diag(_Jl[:, {k}, {l}]))"
            )
        for cix in range(n_cpl):
            lines.append(
                f"    _A = _A.at[{k}*_N:{k+1}*_N, {src_k}*_N:{src_k+1}*_N].add(_Jc[:, {k}, {cix}][:, None] * _W)"
            )
    lines.append("    return _A")
    return "\n".join(lines)


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
