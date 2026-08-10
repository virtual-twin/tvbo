# -*- coding: utf-8 -*-
#
# Module: utils.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""
TVB-Optim Template Utilities
============================

Reusable Python functions for tvboptim Mako templates.
Import these in template blocks to avoid code duplication.

Usage in templates:
    <%
    from tvbo.templates.tvboptim.utils import (
        safe_name, as_list, is_network_observation,
        parse_loss_function, get_observation_refs
    )
    %>
"""

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from tvbo.utils import as_list


# =============================================================================
# Basic Helpers
# =============================================================================


def safe_name(name: str) -> str:
    """Convert name to valid Python identifier (preserves case).

    Python identifiers are case-sensitive, and result keys must match the
    user's YAML keys verbatim so that ``res.explorations.C_sweep_fig3``
    works for a YAML entry named ``C_sweep_fig3``. Only characters that
    are invalid in identifiers (spaces, hyphens) are replaced.
    """
    return str(name).replace(" ", "_").replace("-", "_")


def get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Safe attribute access."""
    return getattr(obj, name, default) if obj else default


def get_param_info(parameters: dict) -> Tuple[List[str], Dict[str, float], Dict[str, str]]:
    """Extract parameter names, defaults, and shapes from a parameters collection.

    Works for both model.parameters and coupling.parameters.

    Args:
        parameters: dict-like of Parameter objects

    Returns:
        tuple: (param_names, param_defaults, param_shapes)
            - param_names: list of parameter names
            - param_defaults: dict of name -> scalar value (for DEFAULT_PARAMS)
            - param_shapes: dict of name -> shape string (only for params with shape attribute)
    """
    if not parameters:
        return [], {}, {}

    params = list(parameters.values()) if hasattr(parameters, "values") else list(parameters)
    # Consistent, deterministic parameter ordering (by name). Every emission site
    # binds params by name (DEFAULT_PARAMS Bunch, params.<name>, override dicts) so
    # order is purely cosmetic — sorting makes it stable across models and codegen
    # paths, independent of how the source parameter collection was built.
    params = sorted(params, key=lambda p: str(p.name))
    param_names = [p.name for p in params]
    param_defaults = {}
    param_shapes = {}

    for p in params:
        if isinstance(p.value, (list, tuple)):
            # Array-valued constant (e.g. a mode-coupling matrix): keep the nested
            # list; the codegen wraps it as a jnp.array, not a scalar default.
            param_defaults[p.name] = list(p.value)
        elif p.value is not None:
            param_defaults[p.name] = float(p.value)
        # A parameter with no literal `value` is simply ABSENT from the defaults: it
        # may be free, or sourced/produced (resolved from storage, never inlined).
        # Emission sites supply their own fallback via `.get(name, 1.0)`; inventing one
        # here instead would erase the difference between "undeclared" and "1.0" and let
        # a resolved operator silently emit as a scalar.
        shape = getattr(p, "shape", None)
        if shape:
            param_shapes[p.name] = str(shape)

    return param_names, param_defaults, param_shapes


def get_mode_layout(model: Any) -> Tuple[int, List[str], Dict[str, List[int]]]:
    """Compute the folded scalar-state layout for a (possibly multi-mode) model.

    tvboptim's solver carries a 2-D state ``(n_states, n_nodes)`` and its coupling
    contracts the node axis with a plain matmul, so it has no place for a third
    per-node mode axis. A model with ``number_of_modes > 1`` (the Stefanescu-Jirsa
    ReducedSet models) folds that mode axis into the state axis: each state
    variable ``v`` occupies ``n_modes`` contiguous scalar slots
    ``v__mode0 .. v__mode{M-1}``. The dfun reconstructs the ``(n_nodes, n_modes)``
    mode-vector for each variable from its slots, evaluates the mode-aware
    equations (``mode_dot``/``mode_sum``), and scatters the per-mode derivatives
    back into those slots; per-mode coupling falls out of the existing 2-D matmul
    because each ``(var, mode)`` slot couples to the same slot across nodes.

    For a single-mode model this is the identity (one slot per variable), so the
    generated code is byte-for-byte unchanged.

    Returns ``(n_modes, slot_names, var_slots)`` where ``slot_names`` is the flat
    solver state ordering (grouped by variable, then mode) and ``var_slots`` maps
    each variable name to the slot indices of its modes.
    """
    var_names = list(model.state_variables.keys()) if model and model.state_variables else []
    n_modes = int(getattr(model, "number_of_modes", None) or 1)
    if n_modes <= 1:
        return 1, list(var_names), {v: [i] for i, v in enumerate(var_names)}
    slot_names: List[str] = []
    var_slots: Dict[str, List[int]] = {}
    for v in var_names:
        var_slots[v] = []
        for m in range(n_modes):
            var_slots[v].append(len(slot_names))
            slot_names.append(f"{v}__mode{m}")
    return n_modes, slot_names, var_slots


def render_jax_default(value: Any) -> str:
    """Render a parameter default as a JAX-ready source literal.

    Array-valued constants (mode-coupling matrices, Gaussian-quadrature vectors)
    must be wrapped in ``jnp.array(...)`` so the generated dfun's arithmetic
    broadcasts; emitting the bare Python list would make ``scalar * list`` raise
    ``TypeError`` at runtime. Scalars render as their full-precision ``repr``
    literal (``str``/``repr`` of a float are equivalent in Python 3, so no
    precision is lost).
    """
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return f"jnp.array({list(value)!r})"
    return repr(value)


def get_recorded_variable_names(model: Any, experiment: Any = None) -> Tuple[List[str], List[str], List[str]]:
    """Compute the variable layout recorded by tvboptim's solver.

    The generated dynamics class declares ``VARIABLES_OF_INTEREST = state_names + recorded_aux``
    where ``recorded_aux`` is the union of:
      * derived variables with ``record: true``,
      * model.output entries that are derived (auxiliary) variables, and
      * derived variables referenced as the ``source`` of any experiment observation
        (so observations of auxiliaries work without requiring users to also list them
        in ``model.output``).

    Returns ``(state_names, recorded_aux, all_var_names)`` where ``all_var_names`` is
    the runtime ordering on axis 1 of ``solution.ys`` / ``result.data`` and matches
    ``solution.variable_names`` produced by tvboptim >= 0.2.7.

    Args:
        model: Dynamics object (with state_variables and derived_variables).
        experiment: Optional SimulationExperiment; observations are scanned when present.
    """
    # Recorded state channels follow the solver's (possibly mode-folded) layout:
    # for number_of_modes>1 each variable contributes n_modes scalar slots.
    _, state_names, _ = get_mode_layout(model) if model and model.state_variables else (1, [], {})
    aux_names = list(model.derived_variables.keys()) if model and getattr(model, "derived_variables", None) else []

    output_vars = getattr(model, "output", None) or []
    if isinstance(output_vars, str):
        output_vars = [output_vars]
    requested_aux = [v for v in output_vars if v in aux_names]

    # Include derived variables marked with record: true
    if model and getattr(model, "derived_variables", None):
        for dv_name, dv in model.derived_variables.items():
            if getattr(dv, "record", False) and dv_name not in requested_aux:
                requested_aux.append(dv_name)

    if experiment is not None and getattr(experiment, "observations", None):
        for obs in experiment.observations.values():
            # `source` is multivalued (a list) and its entries may be objects with `.name`
            # or plain strings; unwrap each so an auxiliary read by an observation is recorded
            # even when it is record: false (its trajectory is streamed, not a user output) and
            # not listed in model.output. A raw str(src) would stringify the whole list and
            # never match a derived-variable name.
            for src in as_list(getattr(obs, "source", None)):
                src_name = str(getattr(src, "name", src))
                if src_name in aux_names and src_name not in requested_aux:
                    requested_aux.append(src_name)

    all_var_names = state_names + requested_aux
    return state_names, requested_aux, all_var_names


def _state_recomputable_derived(model: Any) -> Set[str]:
    """Names of derived variables recomputable from the recorded state alone.

    Each derived variable's expression is fully expanded — via sympy substitution,
    in declaration order (which is topological: a derived variable may only
    reference earlier ones, as the dfun requires) — down to its primitive symbols.
    A variable is state-recomputable iff every remaining symbol is one the
    post-solve realignment binds from the recorded state: a state variable, a
    non-stochastic parameter, a derived parameter, or ``t``.

    A whitelist, not a coupling-name blacklist: coupling inputs surface as their
    per-key symbols (an ``EIBLinearCoupling`` unpacks to ``c_lre`` / ``c_ffi``), not
    under the coupling's declared name, so a blacklist would miss them. Time-varying
    (stochastic) parameters are excluded too — the realignment cannot resolve them
    host-side. Conservative: an unparseable expression, a forward reference, or a
    cyclic one leaves an unexpanded derived-variable symbol behind and so falls out
    as *not* recomputable, without a hand-rolled traversal or cycle guard.

    Shared by :func:`state_only_recorded_aux` and :func:`state_only_derived_var_names`
    so the two never disagree on which auxiliaries are recomputable from the state.
    """
    import sympy as sp

    from tvbo.classes.equation import sympify as _sympify

    dvars = model.derived_variables or {}
    if not dvars:
        return set()
    dparams = getattr(model, "derived_parameters", None) or {}
    params = getattr(model, "parameters", None) or {}
    state_vars = model.state_variables or {}

    def _is_time_stochastic(p):
        dist = getattr(p, "distribution", None)
        return dist is not None and "time" in str(getattr(dist, "axis", "space"))

    safe = set(state_vars.keys())
    safe |= {name for name, p in params.items() if not _is_time_stochastic(p)}
    safe |= set(dparams.keys())
    safe.add("t")

    expanded: Dict[str, Any] = {}  # name -> fully-expanded expr, or None if unparseable
    for name, dv in dvars.items():
        eq = getattr(dv, "equation", None)
        rhs = getattr(eq, "rhs", None) if eq is not None else None
        try:
            expr = _sympify(str(rhs)) if rhs is not None else None
        except Exception:
            expr = None
        if expr is not None:
            subs = {sp.Symbol(n): x for n, x in expanded.items() if x is not None and sp.Symbol(n) in expr.free_symbols}
            if subs:
                expr = expr.subs(subs)
        expanded[name] = expr

    return {name for name, expr in expanded.items() if expr is not None and {str(s) for s in expr.free_symbols} <= safe}


def state_only_derived_var_names(model: Any) -> List[str]:
    """Derived-variable names that are provably functions of the state alone.

    Returned in the model's declaration order (dependency order — a derived
    variable may only reference earlier ones, as the dfun requires). The post-solve
    realignment binds these as locals so a recorded auxiliary can reach the
    *intermediate* derived variables it depends on (e.g. a firing rate that is a
    function of a synaptic-current derived variable) without a ``NameError``.
    """
    recomputable = _state_recomputable_derived(model)
    return [name for name in (model.derived_variables or {}) if name in recomputable]


def state_only_recorded_aux(model: Any, experiment: Any = None) -> List[Tuple[str, int]]:
    """Recorded derived variables that are provably functions of the state alone.

    Returns ``[(name, aux_offset), ...]`` for each recorded derived variable that is
    state-recomputable (see :func:`_state_recomputable_derived`); ``aux_offset`` is
    the variable's index within the recorded-auxiliary block (trajectory channel
    ``len(state_names) + aux_offset``). These can be recomputed from the recorded
    post-step state to undo the solver's one-step auxiliary lag; coupling-dependent
    ones cannot (they need the in-scan coupling) and are omitted.
    """
    _, requested_aux, _ = get_recorded_variable_names(model, experiment)
    if not requested_aux or not (model.derived_variables or {}):
        return []

    recomputable = _state_recomputable_derived(model)
    return [(name, offset) for offset, name in enumerate(requested_aux) if name in recomputable]


def get_output_channels(model: Any, experiment: Any = None) -> Tuple[List[int], List[str], bool]:
    """Resolve the ``sv.record``-honoring output channels for the presented result.

    tvboptim's solver records ALL states (``VARIABLES_OF_INTEREST`` = states +
    recorded aux) because the full trajectory is needed for observations and the
    algorithm warmup. The user-facing ``SimulationResult`` should instead present
    only ``record=True`` state channels (+ recorded auxiliaries), matching the tvb
    backend's ``variables_of_interest``.

    Returns ``(output_indices, output_names, is_subset)`` — the indices/names of the
    kept channels within the full recorded ordering (:func:`get_recorded_variable_names`),
    and whether that is a strict subset. For the common all-``record`` model this is
    the identity (``is_subset`` False), so the template emits the result unsliced.
    Modes are honored: each ``v__mode{m}`` slot inherits ``v``'s record flag.
    """
    _, _requested_aux, all_var_names = get_recorded_variable_names(model, experiment)
    n_modes, _, _ = get_mode_layout(model)
    record_flag: Dict[str, bool] = {}
    for var_name, sv in (model.state_variables or {}).items():
        rec = bool(getattr(sv, "record", True))
        slots = [f"{var_name}__mode{m}" for m in range(n_modes)] if n_modes > 1 else [var_name]
        for slot in slots:
            record_flag[slot] = rec
    # Non-state channels (auxiliaries) are always kept — they are recorded only
    # when explicitly requested, so record_flag.get(nm, True) leaves them in.
    output_indices = [i for i, nm in enumerate(all_var_names) if record_flag.get(nm, True)]
    output_names = [all_var_names[i] for i in output_indices]
    is_subset = output_indices != list(range(len(all_var_names)))
    return output_indices, output_names, is_subset


def resolve_model_output_indices(model: Any, experiment: Any = None) -> Tuple[List[int], List[str]]:
    """Resolve ``model.output`` entries to channel indices in the recorded ordering.

    ``model.output`` may name state variables, auxiliary (derived) variables, or a
    mix of the two. The recorded ordering is the state channels followed by the
    auxiliaries that were actually requested (:func:`get_recorded_variable_names`),
    so an output's position cannot be inferred from its kind — a state output sits
    *before* the auxiliaries, not after them. Multi-mode state variables expand to
    all of their ``v__mode{m}`` slots.

    Returns ``(indices, names)`` in the declared output order, so the emitted
    channel order and the reported ``output_names`` always agree.

    Args:
        model: Dynamics object (with state_variables and derived_variables).
        experiment: Optional SimulationExperiment, forwarded to the layout resolver.

    Raises:
        ValueError: if an output names neither a recorded state nor a recorded
            auxiliary. Model construction already rejects unknown output names and
            every listed auxiliary is recorded by construction, so this guards the
            invariant against future changes to the recorded layout rather than a
            reachable input.
    """
    _, _, all_var_names = get_recorded_variable_names(model, experiment)
    n_modes, _, _ = get_mode_layout(model)
    state_vars = (model.state_variables or {}) if model else {}

    output_vars = getattr(model, "output", None) or []
    if isinstance(output_vars, str):
        output_vars = [output_vars]

    indices: List[int] = []
    names: List[str] = []
    for var in output_vars:
        if n_modes > 1 and var in state_vars:
            slots = [f"{var}__mode{m}" for m in range(n_modes)]
        else:
            slots = [var]
        for slot in slots:
            if slot not in all_var_names:
                raise ValueError(
                    f"output {var!r} is neither a state variable nor a recorded "
                    f"derived variable; recorded channels are {all_var_names}"
                )
            indices.append(all_var_names.index(slot))
            names.append(slot)
    return indices, names


def format_channel_index(indices: List[int], n_channels: int) -> str:
    """Render axis-1 channel indices as the narrowest correct index expression.

    A single channel yields a scalar index (dropping the variable dimension);
    a contiguous run yields a slice, so the common all-auxiliaries case emits the
    same ``n_states:`` slice as before; anything else yields an explicit index
    list, which preserves the declared output order under advanced indexing. An
    empty selection yields ``:`` (all channels) rather than raising.
    """
    if not indices:
        return ":"
    if len(indices) == 1:
        return str(indices[0])
    lo, hi = indices[0], indices[-1] + 1
    if indices == list(range(lo, hi)):
        return f"{lo}:" if hi == n_channels else f"{lo}:{hi}"
    return "[" + ", ".join(str(i) for i in indices) + "]"


def parse_list_elements(rhs_str: str) -> List[str]:
    """Split a ``[a, b, c]`` list-literal string into top-level element strings,
    respecting nested brackets/parens (so ``[f(x, y), g(z)]`` yields two elements)."""
    inner = rhs_str[1:-1]  # strip [ ]
    elements: List[str] = []
    depth = 0
    current: List[str] = []
    for c in inner:
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            elements.append("".join(current).strip())
            current = []
            continue
        current.append(c)
    if current:
        elements.append("".join(current).strip())
    return elements


def _shape_ndim(shape_str) -> int:
    """Rank of a declared parameter shape string, e.g. ``"(n_nodes, n_nodes)"`` → 2.

    Counts the comma-separated dimensions inside the parentheses so a graph-shaped
    (2-D, per-edge) parameter can be told apart from a per-node (1-D) or scalar one.
    Returns 0 for an absent/empty shape.
    """
    if not shape_str:
        return 0
    inner = str(shape_str).strip().strip("()").strip()
    if not inner:
        return 0
    return len([p for p in inner.split(",") if p.strip()])


def graph_selection(network, has_delay: bool) -> Tuple[bool, bool]:
    """Pick the tvboptim graph type for a (possibly delayed) network.

    Returns ``(use_length_graph, use_delay_graph)``:

    - **tract lengths present** → ``DenseLengthGraph`` (delays = lengths /
      conduction_speed, so the conduction speed is a live, sweepable and
      differentiable graph leaf) → ``(True, False)``.
    - **only explicit per-edge ``delay`` attributes** (no lengths) →
      ``DenseDelayGraph`` over those delays → ``(False, True)``.
    - **no delays** (``has_delay`` False) → ``DenseGraph`` → ``(False, False)``.

    Lengths win over edge delays: a network that measures tract lengths derives
    its delays from the swept/optimised conduction speed.
    """
    if not has_delay:
        return False, False
    has_lengths = False
    try:
        lengths = network.lengths_matrix
        if lengths is not None:
            has_lengths = bool(float(lengths.max()) > 0)
    except Exception:
        has_lengths = False
    try:
        has_edge_delays = network._delays_from_edges() is not None
    except Exception:
        has_edge_delays = False
    use_delay_graph = has_edge_delays and not has_lengths
    return (not use_delay_graph), use_delay_graph


def resolve_coupling_spec(coupling, coupling_key, model, coupling_inputs_info, func_to_ci, n_modes=1) -> Dict[str, Any]:
    """Resolve every derived field a tvboptim coupling class needs from a Coupling.

    Keeps the cfun mako template emission-only (resolution lives here, per the
    resolve-in-Python-not-mako convention). Covers: output dimension, incoming/local
    states (explicit, inferred from the pre-expression, or the model's
    coupling_variable states), the mode fold (a multi-mode cvar → its per-node mode
    slots, one output per mode), pre-expression term parsing (list decomposition +
    n_pre), vectorized-vs-per-edge selection, class/base names, the differentiable-
    delay kwarg, state-subscript aliases (``{state}_j``/``_i``) and post-recombination
    symbols, plus the symbol list for the JAX expression printer. Expression rendering
    (``jaxcode``) stays in the template.
    """
    ci_key = func_to_ci.get(coupling_key, coupling_key)
    ci_info = coupling_inputs_info.get(ci_key, {"dimension": 1, "keys": None})
    n_output = ci_info["dimension"]
    has_delay = getattr(coupling, "delayed", False)
    param_names, param_defaults, param_shapes = get_param_info(getattr(coupling, "parameters", None))

    incoming_states = getattr(coupling, "incoming_states", None) or []
    if isinstance(incoming_states, str):
        incoming_states = [incoming_states]
    incoming_states = list(incoming_states) if incoming_states else []
    local_states = getattr(coupling, "local_states", None) or []
    if isinstance(local_states, str):
        local_states = [local_states]
    local_states = list(local_states) if local_states else []

    pre_expr = getattr(coupling, "pre_expression", None) or None
    post_expr = getattr(coupling, "post_expression", None) or None

    # Infer incoming_states from the pre-expression when not explicit: match model
    # state-variable names in the RHS, else fall back to the coupling_variable states.
    if not incoming_states and not local_states and pre_expr:
        svar_names = set()
        if model and getattr(model, "state_variables", None):
            svar_names = {
                sv if isinstance(sv, str) else getattr(sv, "name", str(sv))
                for sv in (model.state_variables.keys() if hasattr(model.state_variables, "keys") else model.state_variables)
            }
        pre_rhs = str(pre_expr.rhs) if pre_expr else ""
        for sv in svar_names:
            if sv in pre_rhs:
                incoming_states.append(sv)
        if not incoming_states:
            cvars = []
            if model and getattr(model, "state_variables", None):
                for sv_name, sv_obj in model.state_variables.items():
                    if getattr(sv_obj, "coupling_variable", False):
                        cvars.append(sv_name)
            incoming_states = cvars if cvars else [pre_rhs.strip()]

    # Mode fold (number_of_modes>1): a coupling reading a multi-mode cvar emits one
    # coupled output per mode; resolve the source cvar to its n_modes per-node slots.
    mode_coupling = bool(n_modes > 1 and incoming_states and not local_states)
    if mode_coupling:
        src_cvar = incoming_states[0]
        incoming_states = [f"{src_cvar}__mode{m}" for m in range(n_modes)]
        n_output = n_modes

    _pre_rhs0 = str(pre_expr.rhs).strip() if pre_expr else ""
    pre_is_list = _pre_rhs0.startswith("[") and _pre_rhs0.endswith("]")
    pre_terms = parse_list_elements(_pre_rhs0) if pre_is_list else ([_pre_rhs0] if _pre_rhs0 else [])
    n_pre = len(pre_terms)

    # Vectorized (matmul) vs per-edge reduction. The identity-sentinel pre() — no
    # pre_expr, or the bare `local_states`/`incoming_states` keyword meaning "sources
    # unchanged" — is a pure linear incoming sum (gx = W @ source): it takes the
    # vectorized incoming-identity path however its states are declared — local-only
    # (FastLinearCoupling), incoming-only, or both (e.g. c_glob: incoming+local same
    # cvar). Source-only phase couplings with a real pre-expression stay per-edge (exact
    # for any connectome) unless the coupling opts in with `vectorized: true`.
    vectorized = getattr(coupling, "vectorized", False)
    vec_identity_sentinel = (not pre_expr) or (_pre_rhs0 in ("local_states", "incoming_states"))
    if (
        not vectorized
        and (incoming_states or local_states)
        and (vec_identity_sentinel or (local_states and not incoming_states))
    ):
        vectorized = True
    vec_states = list(dict.fromkeys(incoming_states + local_states))
    # Identity source-only pre(): the sentinel resolves to "return incoming_states".
    vec_identity = vectorized and vec_identity_sentinel
    # Graph-shaped (n_nodes, n_nodes) params are per-edge; tvboptim requires them
    # declared in EDGE_PARAMS. Read from the declared shape so it holds for optimised weights.
    edge_params = tuple(sorted(n for n in param_names if _shape_ndim(param_shapes.get(n)) == 2))

    class_name = coupling_key.replace(" ", "").replace("-", "")
    base_class = "DelayedCoupling" if has_delay else "InstantaneousCoupling"
    interp_kw = (
        "history_interpolation='linear', " if (has_delay and bool(getattr(coupling, "interpolate_delays", False))) else ""
    )

    # Target-state aliases (theta_i) referenced in post_expression.
    post_aliases_i = []
    post_is_list = False
    if post_expr:
        _post_rhs_str = str(post_expr.rhs).strip()
        post_is_list = _post_rhs_str.startswith("[") and _post_rhs_str.endswith("]")
        _post_state_list = vec_states if vectorized else local_states
        for idx, s in enumerate(_post_state_list):
            if f"{s}_i" in _post_rhs_str:
                post_aliases_i.append((f"{s}_i", idx))

    # Source/target subscript aliases ({state}_j / {state}_i) in the pre-expression.
    state_aliases_j = []
    state_aliases_i = []
    if pre_expr:
        _pre_rhs_str = str(pre_expr.rhs)
        for idx, s in enumerate(incoming_states):
            if f"{s}_j" in _pre_rhs_str:
                state_aliases_j.append((f"{s}_j", idx))
        for idx, s in enumerate(local_states):
            if f"{s}_i" in _pre_rhs_str:
                state_aliases_i.append((f"{s}_i", idx))
    alias_symbols = [a[0] for a in state_aliases_j] + [a[0] for a in state_aliases_i]
    # pre() reads a target-local state: a <state>_i alias, or x_i / local_states as whole tokens.
    uses_local = (
        bool(state_aliases_i) or bool(re.search(r"\bx_i\b", _pre_rhs0)) or bool(re.search(r"\blocal_states\b", _pre_rhs0))
    )
    gx_symbols = ["gx_%d" % k for k in range(n_pre)] if n_pre > 1 else []
    post_alias_symbols = [a[0] for a in post_aliases_i]

    all_symbols = (
        param_names
        + incoming_states
        + local_states
        + ["gx", "G", "x_i", "x_j", "incoming_states", "local_states"]
        + alias_symbols
        + gx_symbols
        + post_alias_symbols
    )
    description = getattr(coupling, "description", None) or "Auto-generated coupling function."

    return {
        "n_output": n_output,
        "has_delay": has_delay,
        "param_names": param_names,
        "param_defaults": param_defaults,
        "param_shapes": param_shapes,
        "incoming_states": incoming_states,
        "local_states": local_states,
        "pre_expr": pre_expr,
        "post_expr": post_expr,
        "mode_coupling": mode_coupling,
        "pre_is_list": pre_is_list,
        "pre_terms": pre_terms,
        "n_pre": n_pre,
        "vectorized": vectorized,
        "vec_states": vec_states,
        "vec_identity": vec_identity,
        "edge_params": edge_params,
        "class_name": class_name,
        "base_class": base_class,
        "interp_kw": interp_kw,
        "post_aliases_i": post_aliases_i,
        "post_is_list": post_is_list,
        "state_aliases_j": state_aliases_j,
        "state_aliases_i": state_aliases_i,
        "uses_local": uses_local,
        "gx_symbols": gx_symbols,
        "all_symbols": all_symbols,
        "description": description,
    }


def resolve_coupling_input_map(model, all_couplings, coupling_inputs_dict):
    """Map coupling-input names to coupling functions for the tvboptim network dict.

    tvboptim keys coupling by coupling-input name; the schema keys by function name.
    Resolution order: (1) explicit ``CouplingInput.source``, (2) same name,
    (3) a single unmapped function broadcasts to all remaining inputs, (4) equal
    counts zip positionally. LOCAL inputs (``CouplingInput.local=True``, e.g.
    ``local_coupling``) are then dropped from the network mapping — a local term is
    TVB's surface/local coupling, zero for the region-based simulations tvboptim
    supports, so it must not be wired to the long-range connectome (the dfun binds
    it to 0 via its fallback).

    Returns ``(ci_coupling_map, func_to_first_ci)`` where ``ci_coupling_map`` maps
    ci_name -> (func_name, coupling_obj) and ``func_to_first_ci`` maps func_name to
    the first ci_name using it (for state-access translation).
    """
    ci_coupling_map = {}
    func_to_first_ci = {}
    if coupling_inputs_dict and all_couplings:
        funcs = list(all_couplings.items())
        ci_names = list(coupling_inputs_dict.keys())

        # 1. Explicit source attribute
        for ci_name in ci_names:
            ci_obj = coupling_inputs_dict[ci_name]
            src = getattr(ci_obj, "source", None)
            if src and src in all_couplings:
                ci_coupling_map[ci_name] = (src, all_couplings[src])
                func_to_first_ci.setdefault(src, ci_name)

        # 2. Same-name match
        for ci_name in ci_names:
            if ci_name not in ci_coupling_map and ci_name in all_couplings:
                ci_coupling_map[ci_name] = (ci_name, all_couplings[ci_name])
                func_to_first_ci.setdefault(ci_name, ci_name)

        # 3/4. Fallback for remaining unmapped
        unmapped_cis = [c for c in ci_names if c not in ci_coupling_map]
        unmapped_funcs = [(n, o) for n, o in funcs if n not in func_to_first_ci]
        if len(unmapped_funcs) == 1 and unmapped_cis:
            for ci_name in unmapped_cis:
                ci_coupling_map[ci_name] = unmapped_funcs[0]
            func_to_first_ci.setdefault(unmapped_funcs[0][0], unmapped_cis[0])
        elif len(unmapped_funcs) == len(unmapped_cis):
            for ci_name, (fn, co) in zip(unmapped_cis, unmapped_funcs):
                ci_coupling_map[ci_name] = (fn, co)
                func_to_first_ci.setdefault(fn, ci_name)

    # Drop LOCAL coupling terms from the network wiring (they stay in
    # coupling_inputs_dict so the dfun still binds the symbol to its 0.0 fallback).
    if model is not None and getattr(model, "coupling_inputs", None):
        for ci_name, ci in model.coupling_inputs.items():
            if getattr(ci, "local", False):
                ci_coupling_map.pop(ci_name, None)

    return ci_coupling_map, func_to_first_ci


def get_node_state_overrides(
    network: Any, n_nodes: int, state_names: List[str], default_initial_state: List[float]
) -> Dict[str, List[float]]:
    """Scan network.nodes for per-node initial state overrides.

    When nodes define ``state: {theta: {value: 0.8}}`` in the YAML, build
    per-node arrays for state variables that differ across nodes.

    Args:
        network: Network object with .nodes list
        n_nodes: number of nodes
        state_names: ordered list of state variable names
        default_initial_state: default initial value per state variable

    Returns:
        dict of sv_name -> list of per-node values (length n_nodes)
    """
    if not network or not getattr(network, "nodes", None):
        return {}

    nodes = list(network.nodes) if not isinstance(network.nodes, list) else network.nodes
    if len(nodes) != n_nodes:
        return {}

    overrides = {}
    for i, sv_name in enumerate(state_names):
        default = default_initial_state[i]
        arr = [default] * n_nodes
        has_override = False
        for node in nodes:
            node_state = getattr(node, "state", None)
            if node_state and sv_name in node_state:
                sv_obj = node_state[sv_name]
                val = float(sv_obj.value) if hasattr(sv_obj, "value") else float(sv_obj)
                arr[int(node.id)] = val
                has_override = True
        if has_override:
            overrides[sv_name] = arr

    return overrides


def get_node_param_overrides(network: Any, n_nodes: int, dyn_param_defaults: Dict[str, float]) -> Dict[str, List[float]]:
    """Scan network.nodes for per-node parameter overrides.

    When nodes define parameters that differ from the dynamics defaults,
    build per-node arrays. Only parameters that differ on at least one
    node are returned.

    Args:
        network: Network object with .nodes list
        n_nodes: number of nodes
        dyn_param_defaults: dict of param_name -> scalar default from dynamics

    Returns:
        dict of param_name -> list of per-node values (length n_nodes)
    """
    if not network or not getattr(network, "nodes", None):
        return {}

    nodes = list(network.nodes) if not isinstance(network.nodes, list) else network.nodes
    if len(nodes) != n_nodes:
        return {}

    # Collect per-node values for all parameters defined on any node
    node_params = {}  # param_name -> {node_id: value}
    for node in nodes:
        node_id = int(node.id)
        if not getattr(node, "parameters", None):
            continue
        params = node.parameters
        if hasattr(params, "values"):
            params = params.values()
        for p in params:
            pname = str(p.name)
            val = float(p.value) if p.value is not None else None
            if val is not None:
                node_params.setdefault(pname, {})[node_id] = val

    # Build per-node arrays using dynamics defaults as base
    overrides = {}
    for pname, node_vals in node_params.items():
        base = dyn_param_defaults.get(pname, 1.0)
        arr = [base] * n_nodes
        for node_id, val in node_vals.items():
            if 0 <= node_id < n_nodes:
                arr[node_id] = val
        # Only include if at least one node differs from default
        if any(v != base for v in arr):
            overrides[pname] = arr

    return overrides


def weight_transform_codegen(network) -> Tuple[List[Tuple[str, List[str]]], List[str]]:
    """Render `transforms:` targeting weight to JAX for inlining in the generated script.

    A transform is a `Function`, so both of its forms are lowered: an `equation:` renders
    through the same expression resolver and primitive table the runtime evaluates
    (`Network.transform_expression`, `tvbo.codegen.transforms`), and a `callable:` renders
    as an import of that callable plus a call. The kit therefore applies the declared
    transform exactly as the runtime does, but visibly, on the RAW weights the generated
    `create_network` is handed — raw SC stays in the network file, the transform stays
    declared in the spec, and the operation is in the script rather than hidden in the tvbo
    runtime. This helper runs only at render time, inside the tvbo environment.

    Args:
        network: The `Network` whose transforms are being lowered.

    Returns:
        A `(transforms, const_env)` pair. `transforms` is a list of
        `(jax_expr, matrix_env)`; `matrix_env` rebinds matrix-derived primitives from the
        current `weights` so a chain of transforms composes. `const_env` binds
        data-derived symbols — `L`, per-node parameter vectors, imported callables — once.

    Raises:
        ValueError: If a transform equation names a symbol that is neither a primitive,
            a declared per-node parameter, nor a substituted argument. Failing here beats
            emitting an undefined name into a kit that only dies once it reaches a cluster.
            Symbols are checked by base identifier, so a masked expression such as
            `mean(W[W > 0])` is validated — and bound — as `W`.
    """
    import numpy as np

    from tvbo.codegen.code import render_expression
    from tvbo.codegen.transforms import PRIMITIVES, emit_env

    node_vectors = getattr(network, "node_parameter_vectors", {}) or {}
    transforms: List[Tuple[str, List[str]]] = []
    const_env: List[str] = []
    const_seen: Set[str] = set()

    def _add_const(name: str, line: str) -> None:
        if name not in const_seen:
            const_env.append(line)
            const_seen.add(name)

    for t in network.transforms_for("weight"):
        c = getattr(t, "callable", None)
        if c is not None:
            alias = f"_tf_{c.name}"
            _add_const(alias, f"from {c.module} import {c.name} as {alias}")
            args = "".join(
                f", {n}={getattr(a, 'value', None)!r}"
                for n, a in (getattr(t, "arguments", {}) or {}).items()
            )
            transforms.append((f"{alias}(weights{args})", []))
            continue

        exp = network.transform_expression(t)
        if exp is None:
            continue
        expr = render_expression(exp, format="jax")
        if "jsp." in expr:
            _add_const("jsp", "import jax.scipy as jsp")
        symbols = sorted({str(sym).split("[", 1)[0] for sym in exp.free_symbols})
        matrix_env, data_env = emit_env(symbols, "weights", "distances")
        for line in data_env:
            _add_const(line.split(" = ", 1)[0], line)
        for s in symbols:
            if s in node_vectors:
                vals = ", ".join(repr(float(v)) for v in np.asarray(node_vectors[s]).ravel())
                _add_const(s, f"{s} = jnp.asarray([{vals}]).reshape(-1, 1)")
            elif s not in PRIMITIVES:
                raise ValueError(
                    f"weight transform {getattr(t, 'name', '?')!r} references {s!r}, which is "
                    f"neither a transform primitive nor a declared per-node parameter of this "
                    f"network. Declare it as a Function argument, or add it to the network."
                )
        transforms.append((expr, matrix_env))
    return transforms, const_env


def to_numeric(val: Any) -> Union[int, float, Any]:
    """Convert string to numeric if possible."""
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        try:
            return int(val) if "." not in val else float(val)
        except ValueError:
            return val
    return val


def normalize_coupling_aliases(all_couplings: Dict[str, Any], model: Any = None) -> Dict[str, Any]:
    """Collapse duplicate keys that point to the same coupling object.

    tvbo can expose one coupling under multiple names such as the coupling
    function name, a coupling-input key, or an explicit CouplingInput.source.
    tvboptim only needs one key per distinct object, so prefer stable,
    user-meaningful aliases before attempting input-to-coupling mapping.
    """
    if not all_couplings:
        return {}

    ci_names = set()
    explicit_sources = set()
    if model is not None and getattr(model, "coupling_inputs", None):
        for ci_name, ci in model.coupling_inputs.items():
            ci_names.add(str(ci_name))
            source = getattr(ci, "source", None)
            if source:
                explicit_sources.add(str(source))

    def rank(key: str, coupling: Any) -> tuple[int, int, str]:
        """Rank an alias `key` for `coupling` so the preferred name sorts first.

        Builds a sort key ordering candidate aliases by preference: an explicit
        `CouplingInput.source` (0), a coupling-input key (1), the coupling's own
        `name` (2), then anything else (3). Ties break by key length and then the
        key string, so shorter, stable names win.

        Args:
            key: Candidate alias under which the coupling is exposed.
            coupling: The coupling object the alias points to.

        Returns:
            A `(priority, key_length, key)` tuple; lower compares as preferred.
        """
        key = str(key)
        coupling_name = str(getattr(coupling, "name", "") or "")
        if key in explicit_sources:
            return (0, len(key), key)
        if key in ci_names:
            return (1, len(key), key)
        if coupling_name and key == coupling_name:
            return (2, len(key), key)
        return (3, len(key), key)

    deduped = {}
    chosen_keys = {}
    for key, coupling in all_couplings.items():
        key = str(key)
        if coupling is None:
            deduped[key] = coupling
            continue

        coupling_id = id(coupling)
        current_key = chosen_keys.get(coupling_id)
        if current_key is None:
            chosen_keys[coupling_id] = key
            deduped[key] = coupling
            continue

        if rank(key, coupling) < rank(current_key, coupling):
            deduped.pop(current_key, None)
            deduped[key] = coupling
            chosen_keys[coupling_id] = key

    return deduped


def iter_parameter_values(parameters: Any):
    """Yield ``(name, value)`` pairs from schema Parameter collections."""
    if not parameters:
        return

    if hasattr(parameters, "items"):
        parameter_items = parameters.items()
    elif isinstance(parameters, (list, tuple)):
        parameter_items = ((getattr(parameter, "name", None), parameter) for parameter in parameters)
    else:
        parameter_items = []

    for parameter_key, parameter in parameter_items:
        parameter_name = getattr(parameter, "name", None) or parameter_key
        parameter_value = getattr(parameter, "value", parameter)
        if parameter_name is not None and parameter_value is not None:
            yield str(parameter_name), to_numeric(parameter_value)


def parameter_value(parameters: Any, name: str, default: Any = None) -> Any:
    """Return a named Parameter value from a schema collection."""
    for parameter_name, value in iter_parameter_values(parameters):
        if parameter_name == name:
            return value
    return default


def pipeline_equation_parameters(pipeline: Any) -> Dict[str, Any]:
    """Collect equation parameters from all observation pipeline steps."""
    values = {}
    for step in pipeline or []:
        equation = getattr(step, "equation", None)
        if equation is None:
            continue
        for parameter_name, value in iter_parameter_values(getattr(equation, "parameters", None)):
            values[parameter_name] = value
    return values


def pipeline_argument(pipeline: Any, name: str) -> Any:
    """Return the named pipeline argument object (arguments are keyed by name)."""
    for step in pipeline or []:
        args = getattr(step, "arguments", None) or {}
        if name in args:
            return args[name]
    return None


def time_argument_ms(argument: Any, default: float) -> float:
    """Resolve a time-valued schema argument to milliseconds."""
    if argument is None or getattr(argument, "value", None) is None:
        return default
    value = float(to_numeric(argument.value))
    unit = str(getattr(argument, "unit", "") or "").lower()
    if unit in {"s", "sec", "second", "seconds"}:
        return value * 1000.0
    return value


def _integration_dt(experiment: Any) -> Optional[float]:
    """The experiment's integration step in ms, or ``None`` when there is no grid.

    Every reducer that turns a declared period into a number of steps needs this, so it
    is read in one place rather than re-walking ``experiment.integration.step_size`` at
    each call site.
    """
    dt = get_attr(get_attr(experiment, "integration"), "step_size") if experiment is not None else None
    return float(to_numeric(dt)) if dt else None


def _reduction_init_value(sv: Any) -> float:
    """Initial scalar for an observer state: its declared ``initial_value``, else 0.0.

    An accumulator starts at its reduction identity (0.0 for a sum) and a memory state's
    init is irrelevant (it is overwritten on the first step), so 0.0 is the fallback —
    which is why this passes its own default rather than the model-state one. An observer
    that is a genuine ODE (the Balloon-Windkessel hemodynamics, whose blood inflow, volume
    and deoxyhaemoglobin rest at 1.0) declares ``initial_value`` and gets it.
    """
    from tvbo.utils import initial_value

    return initial_value(sv, default=0.0)


def _resolve_bold_stream(obs: Any, experiment: Any = None) -> Dict[str, Any]:
    """Lift a streaming ``(init, update, finalize)`` BOLD reducer from an HRF-Volterra
    pipeline observation marked ``reduce: streaming``.

    The materialised ``bold`` pipeline (HRF kernel -> stride decimation -> prepend the
    downsampled transient -> ``'valid'`` HRF convolution -> Volterra scaling -> subsample
    at the TR) is recast as a block reducer: a downsampled-history ring buffer folds each
    integration block, ``strided_convolve`` evaluates the HRF convolution ONLY at the TR
    boundaries, and the Volterra-scaled samples are written into a preallocated BOLD
    buffer — so the full trajectory is never held. Byte-identical to the pipeline value
    to f64 rounding (``strided_convolve`` is ~1e-12 vs the FFT ``fftconvolve``), with no
    FFT buffer. Streamable only when the decimation is a pure ``subsample`` stride;
    ``temporal_average`` is an averaging window (even ``period_samples == 1`` shifts by
    one sample — it reproduces tvboptim TemporalAverage, not identity) and raises.

    Returns a reduction dict tagged ``kind: 'convolution'`` carrying the lifted constants
    (source, HRF-kernel call, decimation stride, TR stride, Volterra ``k_1``/``V_0``); the
    convolution branch of ``render_reduction`` emits the reducer from it. Called bare (no
    ``experiment``) only as the ``is this streaming?`` predicate — that path stays
    side-effect-free and returns the tag without resolving function defaults.
    """
    name = str(get_attr(obs, "name", "observation"))
    src = as_list(get_attr(obs, "source"))
    source = str(src[0]) if src else None
    red: Dict[str, Any] = {"kind": "convolution", "source": source}
    if experiment is None:
        # Predicate call ("is this a streaming reduction?"): stay side-effect-free and
        # skip the function-default lookups that need the experiment. Non-None is enough.
        return red

    pipeline = as_list(get_attr(obs, "pipeline"))
    if not pipeline:
        raise ValueError(
            f"Observation {name!r} declares reduce: streaming but has no pipeline; "
            "streaming is currently supported for the HRF-Volterra BOLD pipeline only."
        )
    exp_funcs = get_attr(experiment, "functions")

    def _func(fname):
        if exp_funcs is None:
            return None
        if hasattr(exp_funcs, "get"):
            return exp_funcs.get(fname)
        for f in exp_funcs:
            if str(get_attr(f, "name", "")) == fname:
                return f
        return None

    def _fname_of(step):
        fn = get_attr(step, "function")
        return str(get_attr(fn, "name", fn)) if fn is not None else None

    def _step_by_func(fname):
        return next((st for st in pipeline if _fname_of(st) == fname), None)

    def _arg_val(argcoll, key):
        if not argcoll:
            return None
        items = argcoll.items() if hasattr(argcoll, "items") else ((str(get_attr(a, "name", "")), a) for a in argcoll)
        for k, a in items:
            if str(k) == key:
                return get_attr(a, "value", a)
        return None

    def _step_or_func_arg(fname, key, default=None):
        st = _step_by_func(fname)
        v = _arg_val(get_attr(st, "arguments"), key) if st is not None else None
        if v is None:
            v = _arg_val(get_attr(_func(fname), "arguments"), key)
        return default if v is None else v

    # HRF kernel: the kernel-generator step (its Function carries a `time_range`). Reuse
    # the emitted kernel function so the streaming kernel is byte-identical to the
    # pipeline's; forward the step's literal arguments, else the function's own defaults.
    kernel_step = next(
        (st for st in pipeline if _func(_fname_of(st)) is not None and get_attr(_func(_fname_of(st)), "time_range")),
        None,
    )
    if kernel_step is None:
        raise ValueError(
            f"Observation {name!r} declares reduce: streaming but its pipeline has no HRF "
            "kernel generator (a Function with a time_range); cannot lift a streaming BOLD reducer."
        )
    kernel_fname = _fname_of(kernel_step)
    _kw = []
    for _k, _a in (
        (get_attr(kernel_step, "arguments") or {}).items()
        if hasattr(get_attr(kernel_step, "arguments") or {}, "items")
        else []
    ):
        _v = get_attr(_a, "value", None)
        if _v is not None and not isinstance(_v, str):
            _kw.append(f"{_k}={_v}")
    red["kernel_call"] = f"{kernel_fname}({', '.join(_kw)})"

    # The decimation MUST be a pure subsample stride; temporal_average is an averaging
    # window (even period_samples=1 shifts by one sample) and is not block-additive, so it
    # cannot be streamed faithfully.
    if _step_by_func("temporal_average") is not None:
        raise ValueError(
            f"Observation {name!r}: reduce: streaming requires a subsample decimation, but the "
            "pipeline decimates with temporal_average (an averaging window; even period_samples=1 "
            "shifts by one sample). Replace it with a subsample step, e.g. a function "
            "`subsample: {equation: {rhs: 'subsample(data, step - 1, step)'}}` and `step: <stride>`."
        )

    # Decimation stride (ds_steps, raw integration steps per downsampled sample) and TR
    # stride (downsampled samples per BOLD repetition time) are GRID-DERIVED: the HRF kernel
    # is sampled on the model's `downsample_period` grid, so on an integration grid with a
    # different `dt` the neural signal must be decimated to that same grid before convolving,
    # and the TR subsample counts downsampled samples. A pipeline's hardcoded subsample
    # `step` / strided `stride` are grid-specific (they assume one dt) and wrong on another
    # grid — so derive from `downsample_period`, the observation's TR (`period`) and `dt`
    # when available, falling back to the declared strides only when they are not.
    _dt = _integration_dt(experiment)
    _dsp = get_attr(obs, "downsample_period")
    _dsp = float(to_numeric(_dsp)) if _dsp else None
    _period = get_attr(obs, "period")
    _period = float(to_numeric(_period)) if _period is not None else None

    if _dsp is not None and _dt:
        ds_steps = int(round(_dsp / _dt))
    else:
        ds_steps = int(to_numeric(_step_or_func_arg("subsample", "step", 1) or 1))
    red["ds_steps"] = max(1, ds_steps)

    if _period is not None and _dt:
        red["tr_stride"] = int(round(_period / (red["ds_steps"] * _dt)))
    else:
        # Un-fused pipeline subsamples at the TR with a `subsample_bold` step (arg `s`); the
        # fused strided pipeline folds it into the strided convolution `stride`.
        _tr = (
            _step_or_func_arg("subsample_bold", "s", None)
            or _step_or_func_arg("strided_hrf", "stride", None)
            or _step_or_func_arg("strided_convolve", "stride", None)
        )
        if _tr is None:
            raise ValueError(
                f"Observation {name!r}: reduce: streaming could not determine the TR stride "
                "(no `period`+`dt`, nor a subsample_bold `s` / strided `stride`, available)."
            )
        red["tr_stride"] = int(to_numeric(_tr))

    # Volterra BOLD scaling k_1 * V_0 * (x - 1). The scaling constants live on a
    # `volterra_transform` step — either a referenced Function's equation parameters, or an
    # inline `equation` on the step itself (the fused strided pipeline uses the latter).
    _vt = _func("volterra_transform")
    _vpar = get_attr(get_attr(_vt, "equation"), "parameters") if _vt is not None else None
    if _vpar is None:
        _vstep = next(
            (
                st
                for st in pipeline
                if str(get_attr(st, "name", "")) == "volterra_transform" or _fname_of(st) == "volterra_transform"
            ),
            None,
        )
        _vpar = get_attr(get_attr(_vstep, "equation"), "parameters") if _vstep is not None else None
    red["k_1"] = float(to_numeric(parameter_value(_vpar, "k_1", 5.6)))
    red["V_0"] = float(to_numeric(parameter_value(_vpar, "V_0", 0.02)))
    return red


def _resolve_stat_stream(obs: Any) -> Dict[str, Any]:
    """Synthesize a cumulative streaming mean/std/variance reducer from ``aggregation``.

    An observation marked ``reduce: streaming`` whose ``aggregation`` is ``mean``, ``std``
    or ``variance`` — and which has no HRF/BOLD ``pipeline`` — folds into the integrator
    carry as a running-moment accumulator instead of materialising the source trajectory.
    ``mean`` carries one accumulator (a running sum, divided by the sample count at
    finalize); ``std``/``variance`` add a sum-of-squares and read ``sqrt(E[x^2] - E[x]^2)``
    (variance without the ``sqrt``) — the ddof=0 form that matches the host ``jnp.std`` /
    ``jnp.var``. The returned dict is shaped exactly as the recurrence resolver's, with no
    ``kind`` tag, so a stat stream reuses :func:`render_recurrence_reduction` unchanged.

    ``skip_inclusive`` marks the reduction a pure accumulator with no per-sample memory
    dependency, so the emitter folds the sample AT ``skip`` (``_gstep >= skip``) rather than
    the step after it — a running mean must not silently drop its first sample, unlike a
    phase-difference observer whose first step has no predecessor. Every RHS is parsed to a
    sympy ``Expr`` against {source, the accumulators, ``count``, ``dt``}, exactly as the
    recurrence path resolves its updates. Takes no ``experiment`` — the full dict resolves
    unconditionally, so the bare ``resolve_reduction(obs)`` streaming predicate stays truthy.
    """
    import sympy as sp

    name = str(get_attr(obs, "name", "observation"))
    src = as_list(get_attr(obs, "source"))
    source = str(src[0]) if src else None
    if source is None:
        raise ValueError(f"Observation {name!r} declares aggregation + reduce: streaming but has no source to reduce over.")
    agg = get_attr(obs, "aggregation", None)
    agg = str(getattr(agg, "value", agg) or "mean").lower()

    allowed = {source, "_s_sum", "_s_sq", "count", "dt"}
    loc = {n: sp.Symbol(n) for n in allowed}

    def _parse(rhs: str) -> Any:
        expr = sp.sympify(rhs, locals=loc)
        unknown = {str(s) for s in expr.free_symbols} - allowed
        if unknown:
            raise ValueError(
                f"Observation {name!r} stat-stream reduction: unknown symbol(s) "
                f"{sorted(unknown)}; available are the source {source!r}, the accumulators "
                "_s_sum/_s_sq, and count/dt."
            )
        return expr

    if agg == "mean":
        states = [
            {
                "name": "_s_sum",
                "init": 0.0,
                "update": _parse(f"_s_sum + {source}"),
                "evict": None,
                "is_accumulator": True,
            }
        ]
        output = _parse("_s_sum / count")
    else:
        states = [
            {"name": "_s_sum", "init": 0.0, "update": _parse(f"_s_sum + {source}"), "evict": None, "is_accumulator": True},
            {"name": "_s_sq", "init": 0.0, "update": _parse(f"_s_sq + ({source})**2"), "evict": None, "is_accumulator": True},
        ]
        # ddof=0. Guard the one-pass co-moment with Abs: catastrophic f64 cancellation on a
        # near-constant source can make it slightly negative, which two-pass jnp.var/jnp.std
        # never return (and would NaN under the sqrt). Abs (unary → jnp.abs) is a no-op for real
        # variance, so it stays byte-identical there while giving ~0 (not NaN) at the degenerate end.
        variance = "Abs(_s_sq / count - (_s_sum / count)**2)"
        output = _parse(f"sqrt({variance})" if agg == "std" else variance)

    return {
        "source": source,
        "states": states,
        "derived": [],
        "surrogates": [],
        "output_name": None,
        "parameters": {},
        "output": output,
        "functions": {},
        "statistic": "mean",
        "histogram": None,
        "windowed": False,
        "skip_inclusive": True,  # pure accumulator: fold the sample AT skip, not the next
    }


def _resolve_fc_stream(obs: Any) -> Optional[Dict[str, Any]]:
    """Lift a cumulative co-moment FC reducer from a ``compute_fc`` pipeline marked
    ``reduce: streaming``.

    A ``pipeline: [compute_fc]`` observation — a node-node Pearson-correlation FC over a
    recorded source (e.g. inp_corr over the excitatory input current ``x_e_pre``) — folds
    into the integrator carry as a Welford co-moment accumulator: the existing
    ``windowed_fc`` reducer recipe's ``add`` update and zero-diagonal Pearson ``emit``,
    WITHOUT its sliding-window ``evict`` (this reduction is *cumulative* over the whole
    run, not a moving window). The ``compute_fc`` ``skip_t`` (default 0) drops the first
    samples exactly as the materialised pipeline does. Byte-identical to
    ``compute_fc(source, skip_t=...)`` to f64 summation order, holding no trajectory
    (``O(n^2)`` co-moment state vs the ``O(n_time * n)`` trajectory).

    The reducer's assignments are lowered to backend source here (via the shared
    :func:`tvbo.codegen.reducers.resolve_streaming_reducer`) so the render partial emits
    from clean context; the dict is tagged ``kind: 'comoment'`` for the matrix-state
    render branch. Returns ``None`` when the pipeline's first reducer has no registered
    ``window`` streaming form (so the caller falls through to the BOLD path). Takes no
    ``experiment`` — side-effect-free, so the bare ``resolve_reduction(obs)`` streaming
    predicate stays truthy.
    """
    from tvbo.codegen.streaming_reducers import lookup_streaming_reducer
    from tvbo.codegen.reducers import resolve_streaming_reducer

    name = str(get_attr(obs, "name", "observation"))
    src = as_list(get_attr(obs, "source"))
    source = str(src[0]) if src else None
    if source is None:
        raise ValueError(
            f"Observation {name!r} declares a compute_fc pipeline + reduce: streaming but has no source to reduce over."
        )
    pipeline = as_list(get_attr(obs, "pipeline"))
    step = pipeline[0]
    call = get_attr(step, "callable")
    mod = str(get_attr(call, "module", "") or "")
    fname = str(get_attr(call, "name", "") or "")
    spec = lookup_streaming_reducer("tvboptim", mod, fname)
    if spec is None or spec.emit_kind != "window":
        return None

    # compute_fc `skip_t` (initial samples dropped before correlating), read off the step
    # arguments (dict-keyed or a legacy list of named args).
    skip_t = 0
    args = get_attr(step, "arguments", None) or {}
    items = args.items() if hasattr(args, "items") else ((str(get_attr(a, "name", "")), a) for a in args)
    for aname, arg in items:
        if str(aname) == "skip_t":
            # a named-arg object carries `.value`; a plain keyed-dict entry IS the scalar.
            _v = get_attr(arg, "value", arg)
            if _v is not None:
                skip_t = int(to_numeric(_v))

    lowered = resolve_streaming_reducer(spec, "jax")
    return {
        "kind": "comoment",
        "source": source,
        "states": list(spec.state),  # e.g. ['count', 'mean', 'comoment']
        "add": lowered["add"],  # [(lhs, jax_rhs_src), ...] — sequential Welford update
        "emit": lowered["emit"],  # zero-diagonal Pearson correlation over the co-moment
        "skip_t": int(skip_t),
        "statistic": "mean",
        "windowed": False,
    }


def _step_reducer_name(step: Any) -> str:
    """Name of a pipeline step's operation.

    A step names its operation in one of three ways and all three occur in curated
    observation models: a ``callable`` reference, a ``function`` reference, or — for a step
    that carries its own ``equation`` — the step's own ``name``.
    """
    for slot in ("callable", "function"):
        ref = get_attr(step, slot)
        if ref is not None:
            return str(get_attr(ref, "name", ref))
    return str(get_attr(step, "name", "") or "")


_STRIDE_REDUCERS = {"subsampling", "subsample", "sub_sample"}


def _resolve_subsample_stream(obs: Any, experiment: Any = None) -> Optional[Dict[str, Any]]:
    """Lift a stride reducer from a pure-decimation pipeline marked ``reduce: streaming``.

    The simplest streamable pipeline is a *stride*: keep every ``k``-th sample of the
    source and drop the rest (``SubSampling(period=TR)``). Materialised it costs a full
    trajectory to produce a ``1/k`` slice of it; folded into the carry it costs only the
    slice. Every sample the reducer keeps is a sample the pipeline would have kept, so the
    streamed value is bit-identical rather than merely close.

    This is deliberately NOT the ``temporal_average`` case: averaging a window is a
    different operation (and tvboptim's emitted form shifts by one sample even at
    ``period_samples == 1``), so it is left on the materialise path.

    Returns a reduction dict tagged ``kind: 'stride'`` carrying the decimation in
    integration steps, or ``None`` when the pipeline is not a pure stride — in which case
    the caller falls through to the BOLD path, whose error message names what it needs.
    """
    pipeline = as_list(get_attr(obs, "pipeline"))
    if not pipeline or any(_step_reducer_name(st).lower() not in _STRIDE_REDUCERS for st in pipeline):
        return None

    name = str(get_attr(obs, "name", "observation"))
    src = as_list(get_attr(obs, "source"))
    source = str(src[0]) if src else None
    red: Dict[str, Any] = {"kind": "stride", "source": source, "windowed": False}
    if experiment is None:
        return red  # predicate call ("is this streaming?"): stay side-effect-free

    def _arg(step, key):
        args = get_attr(step, "arguments", None) or {}
        items = args.items() if hasattr(args, "items") else ((str(get_attr(a, "name", "")), a) for a in args)
        for k, a in items:
            if str(k) == key:
                return get_attr(a, "value", a)
        return None

    dt = _integration_dt(experiment)
    period = next((_arg(st, "period") for st in pipeline if _arg(st, "period") is not None), get_attr(obs, "period"))
    if period is not None and dt:
        ds_steps = int(round(float(to_numeric(period)) / dt))
    else:
        step_arg = next((_arg(st, "step") for st in pipeline if _arg(st, "step") is not None), None)
        if step_arg is None:
            raise ValueError(
                f"Observation {name!r}: reduce: streaming could not determine the decimation "
                "stride (no `period` on the step or the observation with an integration "
                "`step_size`, and no explicit `step`)."
            )
        ds_steps = int(to_numeric(step_arg))
    red["ds_steps"] = max(1, ds_steps)
    return red


# The canonical symbol an observer's expressions use for the signal it observes, bound to
# whatever the observation's `source` names. It is what lets an observation model be curated
# once and pointed at any model's state variable, the same role `x_j` plays in a curated
# coupling's pre_expression.
OBSERVER_INPUT = "x"

_REDUCTION_DIMS = {
    "convolution": ("time", "node"),  # HRF-Volterra BOLD: one kept sample per TR, per node
    "stride": ("time", "node"),  # decimation: every k-th sample, per node
    "comoment": ("node", "node_j"),  # cumulative co-moment FC: a node-by-node matrix
    "recurrence": ("node",),  # a folded statistic: one value per node
    "monitor": ("time", "node"),  # a co-integrated observer emitting at its period
    "wave": ("group", "metric"),  # grouped wave detector: per-group scalar metrics
}


def reduction_dims(red: Optional[Dict[str, Any]]) -> tuple:
    """The axis names a reduction's output carries.

    A reduction's output shape is fixed by its kind, so its axes are named here — where
    the reducer is chosen — and travel with it to the result container. Naming them at
    the point of production is what keeps a reduced observation keyed like every other
    tvbo array without anyone re-deriving the axes from shape afterwards, which cannot
    distinguish (say) a frequency-by-node spectrum from a node-by-node matrix.
    """
    if not red:
        return ()
    return _REDUCTION_DIMS.get(str(red.get("kind")), ())


def _partition_group_count(pdef: Dict[str, Any], gather: str) -> int:
    """Group count for a `partition` = the leading axis of its (n_groups, n) gather table.
    Fixed at codegen: from the declared shape's literal leading dim, the literal value's outer
    length, or the leading dim of the materialised (produced/sourced) gather artifact."""
    shape = pdef.get("shape")
    if shape is not None:
        head = shape[0] if isinstance(shape, (list, tuple)) else str(shape).strip().lstrip("(").split(",")[0].strip()
        try:
            return int(head)
        except (TypeError, ValueError):
            pass
    val = pdef.get("value")
    if val is not None:
        try:
            return len(val)
        except TypeError:
            pass
    lazy = pdef.get("lazy")
    if lazy:
        import numpy as np
        from tvbo.data import param_io

        path, key = lazy
        return int(np.asarray(param_io.read_artifact(path, key)).shape[0])
    raise ValueError(
        f"partition gather {gather!r}: cannot determine the group count from its shape ({shape!r}), "
        "value, or materialised artifact; declare a literal (n_groups, n) leading dimension."
    )


def _toposort_derived(derived: List[Dict[str, Any]], dv_names: set) -> List[Dict[str, Any]]:
    """Order observer derived variables so each follows the DVs its expression reads — a
    STABLE topological sort of the DV dependency DAG. The observer's derived variables form a
    DAG (each RHS reads earlier DVs, states, parameters, or the source), but the keyed
    collection does not preserve authoring order, so a chain like ``U = f(pgn)`` declared
    before ``pgn = g(pg)`` would otherwise emit ``U`` above its input and reference an unbound
    name. Independent variables keep their incoming order (an already-valid or single-DV chain
    is unchanged, so existing observers emit byte-identically); raises on a dependency cycle.
    """

    def _deps(entry: Dict[str, Any]) -> set:
        if "surrogate" in entry:
            s = entry["surrogate"]
            d = {n for n in (str(x) for x in s["expr"].free_symbols) if n in dv_names}
            d.add(s["statistic"])  # the observed value reuses the statistic DV
            return d & dv_names
        return {n for n in (str(x) for x in entry["expr"].free_symbols) if n in dv_names}

    idx = {d["name"]: i for i, d in enumerate(derived)}
    deps = {d["name"]: _deps(d) for d in derived}
    placed: set = set()
    ordered: List[Dict[str, Any]] = []
    remaining = list(derived)
    while remaining:
        ready = [d for d in remaining if deps[d["name"]] <= placed]
        if not ready:
            raise ValueError(
                f"Observation reduction has a cyclic derived-variable dependency among {sorted(d['name'] for d in remaining)}."
            )
        nxt = min(ready, key=lambda d: idx[d["name"]])
        ordered.append(nxt)
        placed.add(nxt["name"])
        remaining.remove(nxt)
    return ordered


def resolve_reduction(obs: Any, experiment: Any = None) -> Optional[Dict[str, Any]]:
    """Lift an observation's auxiliary ``dynamics`` into a backend-agnostic reduction.

    An observation may declare a co-integrated auxiliary ``Dynamics`` (the observer)
    that computes it online as a time recurrence, instead of a post-scan ``pipeline``.
    An observation may instead opt a post-scan ``pipeline`` into streaming via
    ``reduce: streaming`` (currently the HRF-Volterra BOLD pipeline), in which case the
    reducer is lifted by :func:`_resolve_bold_stream` (tagged ``kind: 'convolution'``).
    This resolves that Dynamics into clean context for the reduction partial: the
    source state variable read, and for each observer state its ``init`` value, its
    discrete update RHS (``equation.rhs`` with ``equation_type: recurrence``), and
    whether it is an *accumulator* — its update references its own symbol, so its
    commit is gated on the first step while its memory input is still unset (a memory
    state, which does not reference itself, updates every step). The readout is the
    observer ``output`` (a derived variable's RHS, or a bare final state), and any
    user ``functions`` (e.g. ``wrap``) are surfaced for the printer. Returns ``None``
    when the observation declares no ``dynamics`` (the post-scan path runs).

    Every RHS is parsed to a **sympy** expression against the observer's symbolic
    vocabulary (its states, its ``parameters``, the source, and the framework scalars
    ``dt``/``count``; user functions become undefined ``Function``s). An observer
    parameter is a named constant exactly as a model Dynamics' parameters are, scalar
    or array-valued (a mesh operator, a template matrix), and binds by name in the
    rendered update. That makes the analysis symbolic,
    not string-based: accumulator classification is ``state_symbol in expr.free_symbols``,
    and an unknown symbol (a typo) is caught here rather than surfacing as a codegen
    error. The context carries sympy ``Expr`` objects; the partial renders them per
    backend via ``render_expression`` (which accepts sympy directly). Returns ``None``
    when the observation declares no ``dynamics`` (the post-scan path runs).

    Backend array primitives (``take``, ``sum_axis``, ``clip``, ``matmul``, …) join that
    vocabulary as undefined ``Function``s the printer lowers later, exactly as ``parse_eq``
    registers them — without it a name colliding with a SymPy builtin (``take`` is
    ``sympy.utilities.iterables.take``) is evaluated at parse time and blows up.

    A derived variable's ``surrogate`` reuses the already-computed statistic DV as its
    observed value, so the statistic has to be declared BEFORE it; the reverse order would
    reference a name assigned further down the emitted function, a runtime ``NameError``
    with nothing failing at codegen. The family-wise (Westfall–Young) extremum follows the
    test sidedness — max-T for a ``>=`` test, min-T for ``<=`` — and is derived here rather
    than author-set, so an incoherent extremum/direction pairing cannot be expressed. Each
    surrogate's p-value DV is interleaved back into the chain at its declaration position so
    a DV consuming it is emitted after it; a surrogate entry carries no ``expr``, because the
    renderer emits the vmap-fold over the permutation table instead.

    A ``partition`` lifts the observer into a GROUPED reduction (``kind: 'wave'``): the
    per-step chain is written once for a single group, vmapped over the partition axis, and
    folded to per-group scalar metrics.
    """
    # Opt-in streaming of a post-scan observation: lifted to a block reducer instead of the
    # dynamics-observer recurrence path below. A cumulative mean/std/variance over a source
    # (no HRF/BOLD pipeline) synthesizes a running-moment accumulator; an HRF-Volterra BOLD
    # pipeline lifts the convolution reducer.
    # An observation that declares its own `dynamics` is already a reducer; `reduce:
    # streaming` on it opts that reducer into the post-tuning carry (streaming_post_eval_plan)
    # rather than selecting a pipeline-lifted one, so the observer path below owns it.
    _rm = get_attr(obs, "reduce")
    if _rm is not None and str(getattr(_rm, "value", _rm)) == "streaming" and get_attr(obs, "dynamics") is None:
        _agg = get_attr(obs, "aggregation", None)
        _agg = str(getattr(_agg, "value", _agg) or "").lower()
        _pipe = as_list(get_attr(obs, "pipeline"))
        if not _pipe and _agg in {"mean", "std", "variance"}:
            return _resolve_stat_stream(obs)
        # A compute_fc (windowed-correlation) pipeline streams as a cumulative co-moment
        # FC reducer (add-only Welford), NOT the HRF/BOLD convolution reducer; fall through
        # to the BOLD path only when the pipeline is not a registered windowed reducer.
        if _pipe:
            _fc = _resolve_fc_stream(obs)
            if _fc is not None:
                return _fc
            _sub = _resolve_subsample_stream(obs, experiment)
            if _sub is not None:
                return _sub
        return _resolve_bold_stream(obs, experiment)

    import sympy as sp

    dyn = get_attr(obs, "dynamics")
    if dyn is None:
        return None
    src = as_list(get_attr(obs, "source"))
    source = str(src[0]) if src else None

    svs = get_attr(dyn, "state_variables")
    sv_pairs = list(svs.items()) if hasattr(svs, "items") else []
    sv_names = [str(n) for n, _ in sv_pairs]

    # Derived variables (per-step intermediates). Their names join the symbolic vocabulary
    # so a richer observer's DV can reference an earlier DV (a detector chain, e.g.
    # ``pgn = pg / nrm``); a simple observer has none and this is inert.
    dvs = get_attr(dyn, "derived_variables")
    dv_map = dict(dvs.items()) if hasattr(dvs, "items") else {}
    dv_names = [str(n) for n in dv_map]

    # Derived PARAMETERS are constants (see below); their names bind in every expression
    # exactly as a parameter's does.
    dps = get_attr(dyn, "derived_parameters")
    dp_map = dict(dps.items()) if hasattr(dps, "items") else {}
    dp_names = [str(n) for n in dp_map]

    # Named constants the observer's expressions bind by name — resolved exactly as a
    # model Dynamics' parameters are (same helper), so a scalar and an array-valued
    # constant (e.g. a mesh operator) declare identically and need no observer-specific
    # provenance path.
    from tvbo.data import param_io

    param_map = get_attr(dyn, "parameters") or {}
    param_objs = dict(param_map.items()) if hasattr(param_map, "items") else {str(get_attr(p, "name")): p for p in param_map}
    param_names, param_values, param_shapes = get_param_info(param_map)

    # Resolve each constant to what the template emits: a literal binds inline; a
    # sourced/produced one is materialised (codegen-time) to a content-addressed artifact
    # and emitted as a lazy (file, key) the backend reads at run time, so a large operator
    # never enters the generated source. The spec dir grounds a relative source path
    # exactly as bids_dir is grounded.
    #
    # Materialisation runs the producer and writes to disk, so it happens ONLY when an
    # experiment is supplied — i.e. at genuine emission. resolve_reduction is also called
    # bare (no experiment) as a cheap "is this a streaming reducer?" predicate; that path
    # must stay side-effect-free, so a lazy constant is left deferred (never materialised
    # just to answer a boolean, and never triggering an igl precompute as a side effect).
    src_file = get_attr(experiment, "_source_file", None) if experiment is not None else None
    src_dir = Path(src_file).parent if src_file else None
    parameters: Dict[str, Any] = {}
    for n in param_names:
        if n in param_values:
            parameters[n] = {"value": param_values[n], "shape": param_shapes.get(n)}
        elif experiment is None:
            parameters[n] = {"lazy": None, "shape": param_shapes.get(n)}  # deferred
        else:
            path, key = param_io.materialise(param_objs[n], source_dir=src_dir, context=experiment)
            parameters[n] = {"lazy": (str(path), key), "shape": param_shapes.get(n)}
    # One symbol cannot denote both a per-step state and a fixed constant: the update
    # would read whichever the renderer bound last, silently.
    shadowed = sorted(set(param_names) & set(sv_names))
    if shadowed:
        raise ValueError(
            f"Observation reduction observer declares {shadowed} as both a state "
            f"variable and a parameter; a symbol must denote one or the other."
        )

    # A curated observer cannot know which model state a study will point it at, so it
    # writes its input as the canonical OBSERVER_INPUT — the device a curated Coupling
    # already uses for its presynaptic state (`x_j`) and summed input (`gx`). It is bound
    # here to whatever `source` names, so every rendered expression carries the real symbol
    # and no backend template needs to know about the alias. An observer that declares its
    # own symbol of that name means that symbol, and no alias applies.
    own = set(sv_names) | set(param_names) | set(dv_names) | set(dp_names)
    alias = OBSERVER_INPUT if source and OBSERVER_INPUT not in own else None

    # Symbolic vocabulary: observer states + parameters + source + framework scalars are
    # Symbols; the observer's user functions are undefined Functions. Parse every RHS
    # against it.
    allowed = own | ({source} if source else set()) | {"dt", "count"} | ({alias} if alias else set())
    loc: Dict[str, Any] = {n: sp.Symbol(n) for n in allowed}
    loc["pi"] = sp.pi
    # setdefault, so a declared symbol of the same name still wins over the primitive.
    from tvbo.parse.expression import ARRAY_FUNCTIONS

    for _afn in ARRAY_FUNCTIONS:
        loc.setdefault(str(_afn), sp.Function(str(_afn)))
    funcs = get_attr(dyn, "functions")
    functions: Dict[str, Any] = {}
    for fname, fn in funcs.items() if hasattr(funcs, "items") else []:
        feq = get_attr(fn, "equation")
        fargs = [str(get_attr(a, "name", a)) for a in as_list(get_attr(fn, "arguments"))]
        frhs = get_attr(feq, "rhs") if feq is not None else None
        loc[str(fname)] = sp.Function(str(fname))
        fexpr = sp.sympify(str(frhs), locals={**{a: sp.Symbol(a) for a in fargs}, "pi": sp.pi}) if frhs is not None else None
        functions[str(fname)] = {"args": fargs, "expr": fexpr}

    def _parse(rhs_str: str, where: str):
        expr = sp.sympify(rhs_str, locals=loc)
        unknown = {str(s) for s in expr.free_symbols} - allowed
        if unknown:
            raise ValueError(
                f"Observation reduction {where}: unknown symbol(s) {sorted(unknown)}; "
                f"available are the observer states {sv_names}, its parameters "
                f"{param_names}, the source {source!r}"
                f"{f' (also as {alias!r})' if alias else ''}, and dt/count."
            )
        return expr.subs(sp.Symbol(alias), sp.Symbol(source)) if alias else expr

    states: List[Dict[str, Any]] = []
    windowed = False
    for name, sv in sv_pairs:
        name = str(name)
        eq = get_attr(sv, "equation")
        rhs = get_attr(eq, "rhs") if eq is not None else None
        if rhs is None:
            continue
        expr = _parse(str(rhs), f"state {name!r}")
        # Optional reverse recurrence: the per-step downdate removing the sample leaving a
        # sliding window (the inverse of ``update``, which folds an arriving one in). Parsed
        # against the same vocabulary — ``source`` denotes the sample being folded/removed.
        # Its presence marks the observer a *windowed* reduction (add + evict + resync)
        # rather than a cumulative accumulator; absent -> cumulative, unchanged.
        eeq = get_attr(sv, "evict_equation")
        erhs = get_attr(eeq, "rhs") if eeq is not None else None
        evict = _parse(str(erhs), f"evict of state {name!r}") if erhs is not None else None
        windowed = windowed or evict is not None
        states.append(
            {
                "name": name,
                "init": _reduction_init_value(sv),
                "update": expr,  # sympy Expr — the printer renders it
                "evict": evict,  # sympy Expr (sliding-window downdate) or None
                "is_accumulator": sp.Symbol(name) in expr.free_symbols,
            }
        )

    # The derived-variable chain, resolved as sympy exprs in declaration order (a DV may
    # reference earlier DVs, the source, states, and parameters). A simple observer has an
    # empty chain and reads its value straight from ``output`` below; a detector observer
    # computes this chain per step and the renderer emits it as sequential assignments.
    derived = []
    for dname, dvobj in dv_map.items():
        deq = get_attr(dvobj, "equation")
        drhs = get_attr(deq, "rhs") if deq is not None else None
        if drhs is None:
            continue
        derived.append({"name": str(dname), "expr": _parse(str(drhs), f"derived variable {dname!r}")})

    # `derived_parameters` are constants of the observer — functions of its parameters and
    # `dt`, never of a state or the observed signal — so they are bound ONCE in the reducer's
    # preamble rather than recomputed per step. This is the slot every other backend already
    # uses for exactly this (model Dynamics do the same), and it is what lets a readout built
    # only from states and constants be evaluated per emitted sample instead of per step.
    derived_constants = []
    for pname, pobj in dp_map.items() if hasattr(dp_map, "items") else []:
        peq = get_attr(pobj, "equation")
        prhs = get_attr(peq, "rhs") if peq is not None else None
        if prhs is None:
            continue
        expr = _parse(str(prhs), f"derived parameter {pname!r}")
        bad = {str(s) for s in expr.free_symbols} & (set(sv_names) | set(dv_names) | ({source} if source else set()))
        if bad:
            raise ValueError(
                f"Observation reduction derived parameter {str(pname)!r} references {sorted(bad)}, "
                "which vary per step; a derived parameter is a constant. Declare it as a "
                "derived variable instead."
            )
        derived_constants.append({"name": str(pname), "expr": expr})

    # Permutation-significance surrogates: a derived variable may declare a `surrogate`
    # (re-evaluate a named statistic DV under permutations of a field, report the exceedance
    # p-value). Resolve each to the statistic EXPANDED for its permute symbol — DVs that
    # depend on the permuted symbol are inlined so the statistic becomes a self-contained
    # function of it, while DVs that don't (a captured observed value, e.g. U w.r.t. an
    # in-strength permutation) stay as symbols the renderer binds in-step. General; the wave
    # detector is one consumer.
    dv_expr = {d["name"]: d["expr"] for d in derived}

    def _dv_deps(expr, target, seen=frozenset()):
        for s in (str(x) for x in expr.free_symbols):
            if s == target:
                return True
            if s in dv_expr and s not in seen and _dv_deps(dv_expr[s], target, seen | {s}):
                return True
        return False

    def _expand_stat(stat_name, permute):
        expr = dv_expr[stat_name]
        for _ in range(100):
            pend = {n for n in ({str(s) for s in expr.free_symbols} & set(dv_expr)) if _dv_deps(dv_expr[n], permute)}
            if not pend:
                return expr
            expr = expr.subs({sp.Symbol(n): dv_expr[n] for n in pend})
        raise ValueError(f"surrogate statistic {stat_name!r} expansion did not converge (cyclic derived variables?)")

    surrogates = []
    _dv_order = [str(k) for k in dv_map]  # declaration order = the emitted per-step chain order
    for dname, dvobj in dv_map.items():
        surr = get_attr(dvobj, "surrogate")
        if surr is None:
            continue
        stat_name = str(get_attr(surr, "statistic"))
        if stat_name not in dv_expr:
            raise ValueError(
                f"surrogate on derived variable {str(dname)!r} names statistic {stat_name!r}, "
                f"which is not a derived variable with an equation ({sorted(dv_expr)})."
            )
        if _dv_order.index(stat_name) >= _dv_order.index(str(dname)):
            raise ValueError(
                f"surrogate on derived variable {str(dname)!r} names statistic {stat_name!r} that is "
                "declared at or after it; the statistic must be computed before its surrogate — "
                "declare it earlier in the observer's derived variables."
            )
        permute = str(get_attr(surr, "permute"))
        if permute not in allowed:
            raise ValueError(
                f"surrogate on derived variable {str(dname)!r} permutes {permute!r}, which is "
                f"not an observer symbol (states {sv_names}, parameters {param_names}, source "
                f"{source!r})."
            )
        perms = str(get_attr(surr, "permutations"))
        if perms not in param_names:
            raise ValueError(
                f"surrogate on derived variable {str(dname)!r} names permutation table {perms!r}, "
                f"which is not an observer parameter ({param_names}); declare the fixed "
                "(n_perm, n) index array as a produced/sourced constant."
            )
        direction = str(get_attr(surr, "direction") or "greater_equal")
        if direction not in ("greater_equal", "less_equal"):
            raise ValueError(
                f"surrogate on derived variable {str(dname)!r} sets direction {direction!r}; "
                "expected 'greater_equal' (positive/max-T) or 'less_equal' (negative/min-T). A "
                "silent fallthrough would flip the test sidedness and the family-wise extremum."
            )
        family_reduce = (
            ("nanmax" if direction == "greater_equal" else "nanmin") if bool(get_attr(surr, "family_wise")) else None
        )
        surrogates.append(
            {
                "name": str(dname),
                "expr": _expand_stat(stat_name, permute),  # statistic as a function of `permute`
                "statistic": stat_name,  # observed value reuses the already-computed statistic DV
                "permute": permute,
                "perms": perms,
                "compare": ">=" if direction == "greater_equal" else "<=",
                "family_reduce": family_reduce,  # nan-aware FWE extremum, or None → per-element
            }
        )

    if surrogates:
        _surr_by_name = {s["name"]: s for s in surrogates}
        _eqn_by_name = {d["name"]: d for d in derived}
        derived = [
            {"name": str(n), "surrogate": _surr_by_name[str(n)]} if str(n) in _surr_by_name else _eqn_by_name[str(n)]
            for n in dv_map
            if str(n) in _surr_by_name or str(n) in _eqn_by_name
        ]

    # Emit each derived variable AFTER the DVs its expression reads. The keyed collection does
    # not preserve authoring order, so without this a valid DAG can emit an assignment above
    # its input (an unbound-name crash); the stable sort leaves an already-ordered chain as is.
    derived = _toposort_derived(derived, {d["name"] for d in derived})

    # The readout: the (deprecated) `output` list, else the derived variable marked
    # `record: true` — the current way a Dynamics names what it reports. Exactly one
    # recorded variable can be the observation's value; several is ambiguous, not a default.
    outs = as_list(get_attr(dyn, "output"))
    if not outs:
        outs = [str(n) for n, d in dv_map.items() if get_attr(d, "record")]
        if len(outs) > 1:
            raise ValueError(
                f"Observation reduction observer marks {sorted(outs)} as `record: true`; "
                "an observation has one value, so exactly one derived variable may be recorded."
            )
    output = None
    output_name = str(outs[0]) if outs else None
    if outs:
        out_name = str(outs[0])
        dv = dv_map.get(out_name)
        oeq = get_attr(dv, "equation") if dv is not None else None
        output = _parse(str(get_attr(oeq, "rhs")), f"output {out_name!r}") if oeq is not None else sp.Symbol(out_name)

    # Time-reduction statistic (Observation.aggregation). Default (anything but 'median')
    # folds a running sum (the accumulator state above) and divides by count. 'median' folds
    # a per-node HISTOGRAM of the per-step output (bins from Observation.histogram) into the
    # carry (O(bins) memory, no trajectory) and reads the 0.5 quantile at finalize — the
    # streaming-safe way to get the robust median instantaneous frequency Koller uses, which
    # a running sum cannot compute.
    statistic = "median" if str(get_attr(obs, "aggregation", None) or "mean").lower() == "median" else "mean"
    hist = get_attr(obs, "histogram", None)
    histogram = None
    if hist is not None:
        _bins = get_attr(hist, "n", None)
        histogram = {
            "lo": float(get_attr(hist, "lo", -5.0)),
            "hi": float(get_attr(hist, "hi", 55.0)),
            "bins": int(_bins) if _bins is not None else 512,
        }
    # A streaming median needs explicit bins spanning the reduced quantity's range;
    # silently assuming a default window would clip out-of-range samples and pin the
    # result to an edge. Require the histogram slot rather than guess.
    if statistic == "median" and histogram is None:
        raise ValueError(
            f"Observation {get_attr(obs, 'name', None)!r} sets aggregation: median but declares no "
            "`histogram` slot; a streaming median needs explicit bins (lo/hi/n) spanning the "
            "reduced quantity's range."
        )

    # `period` makes the observer a MONITOR: its output is emitted every period into a time
    # series, instead of collapsing time into one value per node at the final step. That is
    # the declared contract of Observation.dynamics ("read at the final step for a reduction
    # ... or, when `period` is set, sub-sampled over time for a monitor"). The period is
    # resolved against the integration grid, so it needs the experiment; the bare predicate
    # call ("is this a reduction?") only asks whether the observer exists.
    _period = get_attr(obs, "period")
    _period = float(to_numeric(_period)) if _period is not None else None
    _dt = _integration_dt(experiment)
    period_steps = None
    if _period and experiment is not None:
        if not _dt:
            raise ValueError(
                f"Observation {get_attr(obs, 'name', None)!r} declares an observer `dynamics` "
                f"and period {_period} ms, but the experiment has no integration step_size to "
                "resolve it against; the observer cannot know how often to emit."
            )
        period_steps = int(round(_period / _dt))
        if period_steps < 1:
            raise ValueError(
                f"Observation {get_attr(obs, 'name', None)!r}: period {_period} ms is shorter "
                f"than the integration step {_dt} ms; an observer cannot emit faster than it advances."
            )

    # A readout built only from states and constants is the same value whether it is
    # evaluated every step or once per emitted sample, so a monitor evaluates it per sample.
    # One that reads the observed signal or a per-step derived variable cannot be deferred.
    _step_ctx = ({source} if source else set()) | {d["name"] for d in derived if d["name"] != output_name}
    output_per_step = bool(output is not None and {str(s) for s in output.free_symbols} & _step_ctx)

    red = {
        # 'monitor' emits every period_steps into a time series; 'recurrence' folds the whole
        # run into one value per node. Both render from the same observer context.
        "kind": "monitor" if period_steps else "recurrence",
        "period_steps": period_steps,
        "derived_constants": derived_constants,  # bound once in the reducer preamble
        "output_per_step": output_per_step,
        "source": source,
        "states": states,
        "derived": derived,  # per-step DV chain (sympy Exprs, declaration order); [] for a simple observer
        "surrogates": surrogates,  # permutation-significance tests {name, expr(function of permute), permute, perms, compare}; [] if none
        "output_name": output_name,  # the output DV's name (stays inlined at finalize; excluded from the per-step chain)
        "parameters": parameters,  # {name: {value (scalar|nested list), shape}} — named constants
        "output": output,
        "functions": functions,
        "statistic": statistic,  # 'mean' | 'median'
        "histogram": histogram,
        "windowed": windowed,  # True if any state declares an evict_equation (sliding window)
    }

    partition = get_attr(obs, "partition", None)
    if partition is not None:
        if _period is None:
            raise ValueError(
                f"Observation {get_attr(obs, 'name', None)!r} declares a `partition` (grouped wave "
                "reduction) but no `period`; a wave detector samples the phase on a downsample "
                "period, so `period` is required."
            )
        _roles = {r: get_attr(partition, r, None) for r in ("waves", "directed", "correlation")}
        _unset = [r for r, v in _roles.items() if v is None]
        if _unset:
            raise ValueError(
                f"Observation {get_attr(obs, 'name', None)!r}: its `partition` names no derived "
                f"variable for {_unset}. The wave reduction emits proportion-of-waves, "
                "proportion-directed and rho together, so all three roles must be named."
            )
        gather = str(get_attr(partition, "gather"))
        if gather not in parameters:
            raise ValueError(
                f"Observation {get_attr(obs, 'name', None)!r} partition gathers on {gather!r}, which "
                f"is not an observer parameter ({sorted(parameters)}); it must be the produced "
                "(n_groups, n) node-index table."
            )
        over = [str(x) for x in as_list(get_attr(partition, "over"))]
        _missing = [p for p in over if p not in parameters]
        if _missing:
            raise ValueError(
                f"Observation {get_attr(obs, 'name', None)!r} partition lists group-indexed operator(s) "
                f"{_missing} in `over` that are not observer parameters ({sorted(parameters)})."
            )
        red.update(
            {
                "kind": "wave",
                # n_groups reads the produced gather table, so it needs the materialised artifact
                # (experiment supplied). The bare predicate call (experiment=None) only asks IS this a
                # streaming reducer, so leave it deferred there rather than force an igl precompute.
                "n_groups": _partition_group_count(parameters[gather], gather) if experiment is not None else None,
                "group_vmap": {"gather": gather, "over": over},
                "wave_present": str(get_attr(partition, "waves")),
                "sig_corr": str(get_attr(partition, "directed")),
                "corr": str(get_attr(partition, "correlation")),
            }
        )
    return red


def streaming_post_eval_plan(experiment: Any) -> Dict[str, Any]:
    """Plan a streaming post-tuning evaluation for a fitting experiment.

    A ``reduce: streaming`` observation (currently HRF-Volterra BOLD) is folded into the
    integrator carry via ``prepare(reduce=...)`` in the algorithm post-tuning evaluation,
    so the full-length fit trajectory is never materialised (the memory bomb an FC group
    fit hits at the paper's real per-stage duration). This resolves, once for the whole
    experiment:

    - ``names``: the streaming observations to fold (empty => no streaming post-eval; the
      materialise path is unchanged, so every non-opted-in experiment is untouched);
    - ``deliverables``: the derived observations computable from the streamed values plus
      the static network observations ALONE — i.e. WITHOUT the raw trajectory (the FC
      family: ``fc`` from streamed ``bold``, then ``fc_corr``/``fc_rmse`` from ``fc`` and
      the empirical target). Observations that need the raw trajectory (e.g. a post-scan
      ``mean`` over a state variable) are intentionally absent — at fit scale they cannot
      be materialised anyway;
    - ``period_in_steps``: the block-size unit — a multiple of every reducer's
      ``ds_steps * tr_stride`` — so BOLD TR boundaries align to integrator block
      boundaries (a partial-TR block would misalign the reducer's slot writing);
    - ``dims``: each streamed observation's axis names, from the reduction that produces
      it (:func:`reduction_dims`), so the result container binds declared labels rather
      than inferring them from the array's shape.

    Both the experiment template (which builds the streaming ``post_model_fn``) and the
    algorithm template (which consumes it) call this, so the two sides cannot drift.
    """
    obs = get_attr(experiment, "observations") or {}
    obs_by_name = dict(obs.items()) if hasattr(obs, "items") else {str(get_attr(o, "name")): o for o in obs}

    def _is_streaming(o: Any) -> bool:
        _rm = get_attr(o, "reduce")
        return _rm is not None and str(getattr(_rm, "value", _rm)) == "streaming"

    streaming = {}
    for n, o in obs_by_name.items():
        r = resolve_reduction(o, experiment)
        # Fold in-carry reducers: a reduce: streaming observation (BOLD / stat stream) OR a
        # `partition` grouped reduction (kind 'wave'), which writes per-group metrics into a
        # sample-indexed carry the same way — so the base/post-eval run never materialises the
        # trajectory to vmap every frame at once. A plain dynamics observer stays materialised.
        if r is not None and not r.get("windowed") and (_is_streaming(o) or r.get("kind") == "wave"):
            streaming[str(n)] = r
    if not streaming:
        return {"names": [], "deliverables": [], "period_in_steps": None, "dims": {}}

    def _is_static_source(o):
        s = as_list(get_attr(o, "source"))
        s0 = str(get_attr(s[0], "name", s[0])) if s else ""
        return s0.startswith("network.observations.") or s0.startswith("dataset.subject")

    def _obs_sources(o):
        return [str(get_attr(s, "name", s)) for s in as_list(get_attr(o, "source"))]

    # Transitive closure of derived observations reachable from the streamed values and the
    # static (trajectory-free) network observations: a derived obs is computable iff every
    # observation it sources is already available.
    available = set(streaming) | {str(n) for n, o in obs_by_name.items() if _is_static_source(o)}
    deliverables: List[str] = []
    changed = True
    while changed:
        changed = False
        for n, o in obs_by_name.items():
            n = str(n)
            if n in available:
                continue
            srcs = _obs_sources(o)
            obs_srcs = [s for s in srcs if s in obs_by_name]
            # Trajectory-free only when EVERY source is an available observation or a
            # static network/dataset ref; a bare state/column source needs the trajectory.
            if obs_srcs and all(s in available or s.startswith("network.") or s.startswith("dataset.") for s in srcs):
                available.add(n)
                deliverables.append(n)
                changed = True

    import math

    # A reducer that writes into a sample-indexed buffer carries a block to align to, so
    # that a slot boundary never falls inside a block: ds_steps * tr_stride for a
    # convolution (BOLD) reducer, ds_steps for a plain stride, period_steps for a monitor
    # observer. A scalar stat stream has none, so default the block to a plain 1000 steps.
    slotted = [r for r in streaming.values() if r.get("period_steps") or r.get("ds_steps")]
    pis = 1
    for r in slotted:
        step = int(r["period_steps"] if r.get("period_steps") else int(r["ds_steps"]) * int(r.get("tr_stride", 1)))
        pis = pis * step // math.gcd(pis, step)
    period = pis if slotted else 1000
    return {
        "names": sorted(streaming),
        "deliverables": deliverables,
        "period_in_steps": period,
        "dims": {n: reduction_dims(r) for n, r in streaming.items() if reduction_dims(r)},
    }


def _literal_code(value: Any) -> str:
    """Render a literal constructor value as generated Python code."""
    return repr(value)


def _set_literal_arg(class_info: Dict[str, Any], name: str, value: Any) -> None:
    class_info["constructor_args"][name] = value
    class_info["constructor_arg_codes"][name] = _literal_code(value)


def _set_code_arg(class_info: Dict[str, Any], name: str, code: str) -> None:
    class_info["constructor_arg_codes"][name] = code


def _base_class_info(module: str, name: str, source_info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": name,
        "module": module,
        "constructor_args": {},
        "constructor_arg_codes": {},
        "call_args": source_info.get("call_args", {}),
        "warmup_source": source_info.get("warmup_source"),
        "accepts_voi": False,
        "accepts_history": bool(source_info.get("warmup_source")),
        "extra_imports": [],
    }


def adapt_class_reference_for_tvboptim(class_info: Dict[str, Any], obs: Any, dt: float) -> Optional[Dict[str, Any]]:
    """Translate schema class references to native tvboptim monitor classes.

    Database observation metadata may point at TVB monitor classes because the
    same schema object is used by the TVB backend. The tvboptim backend should
    consume the equivalent tvboptim monitor API when one exists.
    """
    class_info.setdefault(
        "constructor_arg_codes",
        {name: _literal_code(value) for name, value in class_info.get("constructor_args", {}).items() if value is not None},
    )
    class_info.setdefault("accepts_voi", False)
    class_info.setdefault("accepts_history", bool(class_info.get("warmup_source")))
    class_info.setdefault("extra_imports", [])

    module = str(class_info.get("module") or "")
    class_name = str(class_info.get("name") or "")
    if module != "tvb.simulator.monitors":
        return class_info

    if class_name == "Bold":
        return _adapt_tvb_bold_reference(class_info, obs, dt)
    if class_name == "TemporalAverage":
        return _adapt_tvb_downsampling_reference(class_info, obs, "TemporalAverage")
    if class_name == "SubSample":
        return _adapt_tvb_downsampling_reference(class_info, obs, "SubSampling")
    return class_info


def _adapt_tvb_downsampling_reference(class_info: Dict[str, Any], obs: Any, tvboptim_class_name: str) -> Dict[str, Any]:
    if tvboptim_class_name == "TemporalAverage":
        tvboptim_class_name = "TVBTemporalAverage"
        module = "tvbo.templates.tvboptim.observations"
    else:
        module = "tvboptim.observations.tvb_monitors.downsampling"
    adapted = _base_class_info(
        module,
        tvboptim_class_name,
        class_info,
    )
    period = class_info.get("constructor_args", {}).get("period", getattr(obs, "period", None))
    if period is not None:
        _set_literal_arg(adapted, "period", to_numeric(period))
    adapted["accepts_voi"] = True
    return adapted


def _adapt_tvb_bold_reference(class_info: Dict[str, Any], obs: Any, dt: float) -> Optional[Dict[str, Any]]:
    constructor_args = class_info.get("constructor_args", {})
    hrf_kernel = constructor_args.get("hrf_kernel", "FirstOrderVolterra")
    if str(hrf_kernel) != "FirstOrderVolterra":
        return None

    adapted = _base_class_info(
        "tvbo.templates.tvboptim.observations",
        "TVBBold",
        class_info,
    )
    adapted["accepts_voi"] = True
    adapted["accepts_history"] = True

    equation_parameters = pipeline_equation_parameters(getattr(obs, "pipeline", None))
    period = constructor_args.get("period", getattr(obs, "period", None))
    if period is None:
        period = parameter_value(getattr(obs, "parameters", None), "TR")
    if period is not None:
        _set_literal_arg(adapted, "period", to_numeric(period))
    stock_dt = pipeline_argument(getattr(obs, "pipeline", None), "stock_dt")
    _set_literal_arg(adapted, "downsample_period", time_argument_ms(stock_dt, 4.0))

    for argument_name in ("k_1", "V_0"):
        if argument_name in equation_parameters:
            _set_literal_arg(adapted, argument_name, equation_parameters[argument_name])

    _set_literal_arg(adapted, "hrf_length", to_numeric(constructor_args.get("hrf_length", 20000.0)))
    for argument_name in ("tau_s", "tau_f", "scaling"):
        if argument_name in equation_parameters:
            _set_literal_arg(adapted, argument_name, equation_parameters[argument_name])
    return adapted


# =============================================================================
# Solver / differentiation kwargs
# =============================================================================


def resolve_solver_kwargs(integration: Any, dt: float, is_diffrax: bool = False) -> str:
    """Map the backend-neutral ``integration.differentiation`` strategy onto
    native-solver kwargs, returned as a ready-to-emit string (e.g.
    ``"grad_horizon=100, block_size=50"``).

    ``truncation_window`` / ``checkpoint_interval`` are in ms of simulated time;
    the native JAX solver counts integration steps, so they are converted with
    ``dt``. Diffrax has no such knobs, so ``is_diffrax=True`` yields ``""``.
    Shared by the experiment and solver templates so the mapping lives in one
    place rather than being duplicated in both mako blocks.
    """
    if integration is None or is_diffrax:
        return ""
    kwargs = []
    # Coupling evaluation across integrator stages (backend-neutral -> native flag):
    # per_stage recomputes the coupling every solver stage (accurate); per_step (the
    # native default) holds it constant across stages. Only emit the non-default.
    ce = getattr(integration, "coupling_evaluation", None)
    if ce is not None and str(ce) == "per_stage":
        kwargs.append("recompute_coupling_per_stage=True")
    diff = getattr(integration, "differentiation", None)
    if diff is not None:
        tw = getattr(diff, "truncation_window", None)
        if tw is not None:
            kwargs.append(f"grad_horizon={int(round(float(tw) / dt))}")
        ci = getattr(diff, "checkpoint_interval", None)
        if ci is not None:
            kwargs.append(f"block_size={int(round(float(ci) / dt))}")
    return ", ".join(kwargs)


def _analysis_solver_kwargs(solver_kwargs: str) -> str:
    """Drop the differentiation-truncation kwargs from a solver-kwargs string.

    ``grad_horizon`` / ``block_size`` are truncated-BPTT knobs for the optimization
    forward/backward pass; they are not part of an analysis diagnostic and break a
    tangent-space (JVP) Lyapunov spectrum — the truncation ``stop_gradient``s the
    early segment, so the initial-state perturbation never reaches the segment end
    and the leading exponent collapses to ``log(0) = -inf``. The reference analysis
    solves build a plain solver for the same reason. Coupling-evaluation config
    (``recompute_coupling_per_stage``) is kept so the diagnostic characterises the
    same trajectory as the main sim.
    """
    kept = [
        tok
        for tok in (t.strip() for t in solver_kwargs.split(","))
        if tok and not tok.startswith(("grad_horizon=", "block_size="))
    ]
    return ", ".join(kept)


def resolve_optimizer_mode(integration: Any) -> str:
    """Map the backend-neutral ``integration.differentiation.mode`` onto the native
    optimizer differentiation mode.

    ``reverse`` -> ``"rev"`` (reverse-mode BPTT; pairs with a ``grad_horizon`` window
    for truncated BPTT); ``forward`` -> ``"fwd"`` (forward-mode AD, the exact
    untruncated gradient for a scalar parameter). Defaults to ``"rev"`` when no
    differentiation strategy is declared.
    """
    diff = getattr(integration, "differentiation", None) if integration else None
    mode = getattr(diff, "mode", None) if diff is not None else None
    if mode is None:
        return "rev"
    return {"forward": "fwd", "reverse": "rev"}.get(str(mode), str(mode))


def resolve_config_access(dotted: str, coupling_keys: Set[str], external_keys: Set[str] = frozenset()) -> Optional[str]:
    """Dotted state-config path for a `<scope>.<param>` parameter reference.

    One addressing grammar, shared by optimization ``free_parameters``, analysis
    ``wrt``, and inference ``priors`` — so "which knob" reads the same everywhere:

    - ``<coupling_key>.<param>``  -> ``coupling.<key>.<param>``  (prefix ∈ coupling_keys)
    - ``<event_name>.<param>``    -> ``external.<name>.<param>`` (prefix ∈ external_keys)
    - ``<Dynamics>.<param>`` or bare ``<param>`` -> ``dynamics.<param>`` (default scope)

    ``external_keys`` are stimulus/external-input event names (the keys of the
    network's ``external_input`` dict, e.g. ``stimulus``).
    """
    wp = dotted.split(".") if dotted else []
    if len(wp) == 2 and wp[0] in coupling_keys:
        return f"coupling.{wp[0]}.{wp[1]}"
    if len(wp) == 2 and wp[0] in external_keys:
        return f"external.{wp[0]}.{wp[1]}"
    if wp:
        return f"dynamics.{wp[-1]}"
    return None


def _analysis_wrt_access(wrt: List[str], coupling_keys: Set[str]) -> Optional[str]:
    """Resolve an analysis ``wrt`` reference to a config path (coupling/dynamics)."""
    return resolve_config_access(wrt[0], coupling_keys) if wrt else None


def _lr_analysis_spec(lr_obs, model, events, op_constraint, time_si_factor, dt):
    """Resolve the linear-response analysis observations (covariance / psd / fisher) into the spec
    the ``lr_analysis_block`` Mako orchestrator emits — resolution only, no code strings. Provides
    the operating-point symbol layout (:func:`linear_response_context`), each observable's
    parameters, the Fisher stimulus (event → per-node target ``nodes`` + a heterogeneous-variable
    linearisation context), and — for a constraint-defined (Deco FIC) operating point — the
    unfolded constraint expression. The template owns the structure and orchestration.

    The operating-point settle steps at ``settle_dt``, capped by the experiment's own
    integration step: a settle step is in MODEL time units, so a fixed one is right for only
    one time unit — 0.1 is a tenth of a millisecond for a millisecond model and a hundred
    milliseconds for a second-based one, well past the stability boundary of a 10 ms
    inhibitory time constant. The recipe already states a step that integrates this model
    stably, so never exceeding it is the metadata-driven bound.
    """
    from tvbo.analysis.linear_response import (
        constraint_expr,
        linear_response_context,
        observable_terms,
    )

    ctx = linear_response_context(model)
    settle_dt = min(0.1, float(dt)) if dt else 0.1
    obs_specs: List[Dict[str, Any]] = []
    for name, aobs in lr_obs.items():
        an = aobs.analysis
        atype = str(an.type)
        p = {str(k): (v.value if hasattr(v, "value") else v) for k, v in (getattr(an, "parameters", None) or {}).items()}
        if atype == "covariance":
            # `sigma` states a UNIFORM noise amplitude; omitting it takes the per-state
            # amplitudes the model declares (ctx['noise']). `observable` names the declared
            # quantity whose covariance is wanted — a state variable or any derived variable,
            # so the response can be read out through a declared cascade (BOLD) rather than
            # off the state vector. Omitting it keeps the first state block.
            observable = p.get("observable")
            obs_specs.append(
                {
                    "type": "covariance",
                    "name": name,
                    "sigma": None if p.get("sigma") is None else float(p["sigma"]),
                    "observable": None if observable is None else str(observable),
                    "observable_terms": None if observable is None else observable_terms(model, str(observable)),
                    "return": str(p.get("return", "covariance")),
                }
            )
        elif atype == "psd":
            obs_specs.append(
                {
                    "type": "psd",
                    "name": name,
                    "sigma": float(p.get("sigma", 0.01)),
                    "f_lo": float(p.get("f_lo", 0.1)),
                    "f_hi": float(p.get("f_hi", 50.0)),
                    "n_freq": int(p.get("n_freq", 128)),
                }
            )
        elif atype == "fisher":
            _target = str(getattr(an, "target", "") or "")
            ev = (events or {}).get(_target)
            if ev is None:
                raise ValueError(
                    f"fisher analysis observation '{name}' targets stimulus event '{_target}', "
                    f"which is not defined on this experiment (available: {sorted((events or {}).keys())}). "
                    f"Declare a stimulus event with that key (event_type: stimulus, target_variable, "
                    f"target_regions)."
                )
            stim_var = str(getattr(ev, "name", None) or getattr(ev, "target_variable", None))
            obs_specs.append(
                {
                    "type": "fisher",
                    "name": name,
                    "fisher_ctx": {**ctx, "pernode": set(ctx["pernode"]) | {stim_var}},
                    "stim_var": stim_var,
                    "nodes": [int(x) for x in (getattr(ev, "nodes", None) or [])],
                    "sigma": float(p.get("sigma", 0.01)),
                    "dI_lo": float(p.get("dI_lo", 0.02)),
                    "dI_hi": float(p.get("dI_hi", 0.1)),
                    "dI_step": float(p.get("dI_step", 0.006)),
                }
            )
    cexpr = constraint_expr(model, str(op_constraint["constraint_variable"])) if op_constraint else None
    return {
        "ctx": ctx,
        "op_constraint": op_constraint,
        "constraint_expr": cexpr,
        "time_scale": time_si_factor,
        "settle_dt": settle_dt,
        "obs": obs_specs,
    }


def render_analysis_observations(
    analysis_obs: Dict[str, Any],
    coupling_keys: Set[str],
    solver_class: str,
    transient_time: float,
    t1_default: float,
    dt: float,
    solver_kwargs: str = "",
    model: Any = None,
    time_si_factor: float = 1.0e-3,
    events: Optional[Dict[str, Any]] = None,
    op_constraint: Optional[Dict[str, Any]] = None,
) -> str:
    """Render the body of the generated ``compute_analysis_observations()`` function.

    Analysis observations ANALYZE the solve/loss (Lyapunov spectrum, autodiff and
    finite-difference gradients) rather than transforming ``result.data``. Each is
    emitted from its declarative ``analysis`` metadata (type + target + wrt +
    parameters). This lives in the adapter/Python layer — NOT the mako template —
    so the per-type branching can be deduped/harmonized and reused across backends;
    the template only interpolates the returned block. Analysis solves drop the
    differentiation-truncation window (an optimization knob, not part of these
    diagnostics — see :func:`_analysis_solver_kwargs`) while keeping the coupling-
    evaluation config. ``time_si_factor`` is seconds per model time unit (ms -> 1e-3); the
    linear-response operating point rescales the Jacobian A to per-second with it, so every
    downstream quantity (covariance, PSD in Hz, Fisher) is physical. Returns a string whose
    lines are indented for a function body (4 spaces), empty string if there are no analysis
    observations.
    """
    window = f"t0=0.0 + {transient_time}, t1={transient_time} + {t1_default}, dt={dt}"
    solver_kwargs = _analysis_solver_kwargs(solver_kwargs)
    lines: List[str] = []

    # Linear-response analysis (covariance / psd / fisher): all share ONE deterministic operating
    # point, so the whole block — vector field, Jacobian, operating point (a noise-off settle, or a
    # constraint solve for a Deco-FIC-style tuned parameter), and each observable's linear algebra —
    # is emitted by ONE Mako orchestrator (``lr_analysis_block``): the template owns the structure
    # AND its orchestration (partial choice, ordering, per-observable loop). Python only RESOLVES
    # the spec (``_lr_analysis_spec``) — no Python string-emit of code bodies.
    _LR_TYPES = {"covariance", "psd", "fisher"}
    _lr_obs = {n: a for n, a in analysis_obs.items() if str(getattr(a.analysis, "type", "") or "") in _LR_TYPES}
    if _lr_obs and model is None:
        lines += [f"# {n}: {str(a.analysis.type)} analysis needs the model threaded — skipped." for n, a in _lr_obs.items()]
    elif _lr_obs:
        from tvbo import templates as _templates

        _lr_tpl = _templates.lookup.get_template("_linear_response.py.mako")
        _spec = _lr_analysis_spec(_lr_obs, model, events, op_constraint, time_si_factor, dt)
        lines += _lr_tpl.get_def("lr_analysis_block").render(spec=_spec).strip("\n").split("\n")

    # Finite-difference observations that share the exact same per-seed computation
    # (same target, wrt, delta, seeds, seed_base) reuse ONE ``jax.lax.map`` — matching the
    # reference, which derives fd_mean and fd_sem from a single map — rather than
    # recomputing the seeds once per reduction. Assign a shared group id per signature.
    def _fd_signature(aobs):
        an = aobs.analysis
        p = {str(k): (v.value if hasattr(v, "value") else v) for k, v in (getattr(an, "parameters", None) or {}).items()}
        wrt = [str(w) for w in (getattr(an, "wrt", None) or [])]
        return (
            str(getattr(an, "target", None) or "loss"),
            _analysis_wrt_access(wrt, coupling_keys),
            float(p.get("delta", 0.3)),
            int(p.get("seeds", 8)),
            int(p.get("seed_base", 0)),
        )

    fd_group = {}  # signature -> group id
    for aobs in analysis_obs.values():
        if str(getattr(aobs.analysis, "type", "") or "") == "finite_difference":
            fd_group.setdefault(_fd_signature(aobs), f"fdgrp{len(fd_group)}")
    fd_emitted: Set[tuple] = set()

    for name, aobs in analysis_obs.items():
        an = aobs.analysis
        atype = str(getattr(an, "type", "") or "")
        if atype in _LR_TYPES:
            continue  # linear-response types are emitted together by lr_analysis_block (above)
        params = {str(k): (v.value if hasattr(v, "value") else v) for k, v in (getattr(an, "parameters", None) or {}).items()}
        target = str(getattr(an, "target", None) or "loss")
        wrt = [str(w) for w in (getattr(an, "wrt", None) or [])]
        access = _analysis_wrt_access(wrt, coupling_keys)
        if atype == "lyapunov":
            seg = float(params["segment_time"])
            # Accept the declarative names (n_steps / n_exponents) and the short
            # aliases (n / k). n_exponents defaults to 1 (the leading exponent).
            n = int(params.get("n_steps", params.get("n", 10)))
            k = int(params.get("n_exponents", params.get("k", 1)))
            lines += [
                f"# {name}: Benettin QR spectrum + leading Lyapunov vector on a segment solve.",
                "# Same integrator config as the main sim (coupling_evaluation) so lambda_1",
                "# matches the trajectory it characterises. Emits the exponents and the",
                f"# per-node leading-vector profile ({name}_xi, paper's xi_i).",
                f"_le_solve, _le_cfg = prepare(network, {solver_class}({solver_kwargs}), t0=0.0, t1={seg}, dt={dt})",
            ]
            # Evaluate at the current operating point: sync `wrt` (e.g. the swept coupling)
            # into the segment config so a K-sweep reports lambda_1(K), not at the network's
            # fixed parameters. Optional — omit `wrt` for a static Lyapunov.
            if access:
                lines.append(f"_le_cfg = eqx.tree_at(lambda _c: _c.{access}, _le_cfg, state.{access})")
            lines.append(
                f"obs.{name}, obs.{name}_xi = benettin_spectrum_and_vectors(_le_solve, _le_cfg, t={seg}, n={n}, k={k})"
            )
        elif atype == "gradient":
            mode = params.get("mode", "reverse")
            lines += [
                f"# {name}: full (untruncated) {mode}-mode gradient of '{target}' wrt {wrt[0]}",
                f"_asolve_{name}, _ = prepare(network, {solver_class}({solver_kwargs}), {window})",
                f"def _grad_of_{name}(_p):",
                f"    _gs = eqx.tree_at(lambda _s: _s.{access}, state, _p)",
                f"    return compute_all_observations(_asolve_{name}(_gs), _gs, result_transient).{target}",
                f"_, obs.{name} = jax.value_and_grad(_grad_of_{name})(state.{access})",
            ]
        elif atype == "finite_difference":
            delta = float(params.get("delta", 0.3))
            seeds = int(params.get("seeds", 8))
            seed_base = int(params.get("seed_base", 0))
            # `stat` selects the reduction over the per-seed central differences: 'mean'
            # (default) = the seed-averaged gradient estimate; 'sem' = its standard error
            # (std / sqrt(seeds)). A mean/sem pair on the same settings shares the single
            # per-seed map below, so the seeds are computed once (as in the reference).
            stat = str(params.get("stat", "mean"))
            sig = _fd_signature(aobs)
            gid = fd_group[sig]
            arr = f"_fds_{gid}"
            if sig not in fd_emitted:
                lines += [
                    f"# per-seed central differences of '{target}' wrt {wrt[0]} (shared across its reductions)",
                    f"_asolve_{gid}, _ = prepare(network, {solver_class}({solver_kwargs}), {window})",
                    f"_delta_{gid} = {delta}",
                    f"_keys_{gid} = jax.random.split(jax.random.key({seed_base}), {seeds})",
                    f"_g0_{gid} = state.{access}",
                    f"def _fd_{gid}(_key):",
                    "    _cs = eqx.tree_at(lambda _s: _s.noise.key, state, _key)",
                    f"    _loss_at = lambda _g: compute_all_observations(_asolve_{gid}(eqx.tree_at(lambda _s: _s.{access}, _cs, _g)), _cs, result_transient).{target}",
                    f"    return (_loss_at(_g0_{gid} + _delta_{gid}) - _loss_at(_g0_{gid} - _delta_{gid})) / (2.0 * _delta_{gid})",
                    f"{arr} = jax.lax.map(_fd_{gid}, _keys_{gid})",
                ]
                fd_emitted.add(sig)
            reduce = f"jnp.std({arr}) / jnp.sqrt({seeds})" if stat == "sem" else f"jnp.mean({arr})"
            lines += [
                f"# {name}: seed-averaged central finite-difference {stat} of '{target}' wrt {wrt[0]}",
                f"obs.{name} = {reduce}",
            ]
        else:
            lines.append(f"# {name}: analysis type '{atype}' not yet lowered for this backend — skipped.")
    return "\n".join(f"    {ln}" for ln in lines)


def render_adiabatic_signal(signal_expr: str, var_names: List[str]) -> str:
    """Render an envelope-signal expression over recorded variables as an ``observe`` body.

    Each recorded-variable name in ``signal_expr`` (e.g. ``"y1 - y2"``) is replaced by its
    slice ``_r.ys[:, <index>, :]`` (a ``[n_time, n_nodes]`` view). ``<index>`` is the
    variable's position in the solver's recorded ordering (:func:`get_recorded_variable_names`).
    The replacement is a single alternation pass (longest names first, so a name is not
    matched where it is a prefix of another like ``y1`` in ``y12``); a single pass — not
    iterated ``re.sub`` per name — also guarantees the emitted slice text (which itself
    contains ``ys``/``r``) is never re-scanned and re-substituted. Lets the adiabatic-scan
    exploration observe an arbitrary state/derived signal declaratively, without a driver.
    """
    import re

    if not var_names:
        return str(signal_expr)
    index = {nm: i for i, nm in enumerate(var_names)}
    alt = "|".join(re.escape(nm) for nm in sorted(var_names, key=len, reverse=True))
    pattern = re.compile(rf"\b({alt})\b")
    return pattern.sub(lambda m: f"_r.ys[:, {index[m.group(0)]}, :]", str(signal_expr))


def render_recorded_observable(
    record_names: List[str],
    derived_names: List[str],
    network_obs_names: List[str],
    analysis_names: List[str],
    only_obs: Optional[List[str]] = None,
    recorded_var_names: Optional[List[str]] = None,
) -> str:
    """Render the body of an exploration ``observable_fn`` that records a `record:` list.

    Each recorded name resolves to one of three sources: an ``analysis`` diagnostic
    (Lyapunov, gradients) from ``compute_analysis_observations``; a raw model channel —
    a state variable or a recorded auxiliary/derived variable in ``recorded_var_names`` —
    read from ``result.data`` at its channel index; or a declared observation from
    ``compute_all_observations``. The observable returns a ``Bunch`` of the named values,
    which the exploration stacks over the grid into one array per name. Kept in the
    adapter (not the template) so the same routing serves any backend. Returns the
    function-body string (8-space indented for ``def observable_fn(s):`` inside the
    exploration function).
    """
    analysis_set = set(analysis_names)
    channel = {n: i for i, n in enumerate(recorded_var_names or [])}
    # A declared observation is a recorded name that is neither an analysis diagnostic
    # nor a raw model channel; only those need compute_all_observations.
    obs_names = [n for n in record_names if n not in analysis_set and n not in channel]
    lines = ["result = _expl_model_fn(s)"]
    if obs_names:
        # Restrict the per-cell computation to the recorded observations and their
        # closure (passed by the caller), so non-recorded — possibly non-jittable —
        # observations never execute inside this jitted observable.
        if only_obs is not None:
            _only = [n for n in only_obs if n not in channel]
            # An empty closure must emit an empty *set* literal — "{}" is an empty
            # dict, which would change compute_all_observations' `only=` semantics.
            _only_lit = ("{%s}" % ", ".join(repr(n) for n in sorted(_only))) if _only else "set()"
            lines.append(f"_all_obs = compute_all_observations(result, s, result_transient, only={_only_lit})")
        else:
            lines.append("_all_obs = compute_all_observations(result, s, result_transient)")
    if any(n in analysis_set for n in record_names):
        lines.append("_an_obs = compute_analysis_observations(s, _network, result_transient)")
    entries = []
    for n in record_names:
        if n in analysis_set:
            entries.append(f"{n}=_an_obs.{n}")
        elif n in channel:
            entries.append(f"{n}=result.data[:, {channel[n]}, ...]")
        else:
            entries.append(
                f"{n}=getattr(_all_obs, '{n}').data if hasattr(getattr(_all_obs, '{n}', None), 'data') else getattr(_all_obs, '{n}')"
            )
    lines.append(f"return Bunch({', '.join(entries)})")
    return "\n".join(f"        {ln}" for ln in lines)


def _dist_params(dist_obj: Any) -> Dict[str, float]:
    """Numeric parameters of a Distribution, tolerant of the value/label gotcha.

    A bare scalar in a keyed ``Parameter`` collection lands in ``.label`` (str),
    not ``.value``; accept either so YAML can write ``std: 2.0`` or
    ``std: {value: 2.0}``.
    """
    out: Dict[str, float] = {}
    for k, p in (getattr(dist_obj, "parameters", None) or {}).items():
        v = getattr(p, "value", None)
        if v is None:
            v = getattr(p, "label", None)
        if v is not None:
            out[str(k)] = float(v)
    return out


def dist_expr(dist_obj: Any) -> str:
    """Render a Distribution as a numpyro ``dist.*`` constructor string.

    ``Normal`` -> ``dist.Normal(mean, std)``; ``Uniform`` -> ``dist.Uniform(lo, hi)``
    (from ``parameters`` or ``domain``). Reuses the standard Distribution vocabulary
    (name + parameters/domain) as both prior and likelihood-noise family.
    """
    name = str(getattr(dist_obj, "name", None) or "Normal")
    p = _dist_params(dist_obj)
    dom = getattr(dist_obj, "domain", None)
    if name in ("Normal", "Gaussian"):
        return f"dist.Normal({p.get('mean', 0.0)}, {p.get('std', p.get('sigma', 1.0))})"
    if name == "Uniform":
        lo = p.get("lo", float(getattr(dom, "lo", 0.0)) if dom else 0.0)
        hi = p.get("hi", float(getattr(dom, "hi", 1.0)) if dom else 1.0)
        return f"dist.Uniform({lo}, {hi})"
    if name in ("LogNormal", "HalfNormal"):
        args = ", ".join(str(v) for v in p.values())
        return f"dist.{name}({args})"
    # Fallback: pass parameters positionally.
    return f"dist.{name}({', '.join(str(v) for v in p.values())})"


def render_inference(
    inf: Any, coupling_keys: Set[str], external_keys: Set[str], derived_names: Set[str], network_obs_names: Set[str]
) -> str:
    """Render the body of one Bayesian inference (numpyro NUTS/MCMC), 8-space indented.

    Mirrors the tvboptim workflow's ``make_model`` + ``MCMC(NUTS(...)).run``: sample
    each prior, inject it into the forward config at its resolved path, run the SAME
    differentiable ``model_fn``, score the observed observable under the likelihood.
    Config injection uses ``eqx.tree_at`` (functionally identical to the reference's
    in-place mutation). The observed data comes from the ``likelihood.source``
    observation — a runtime binding or a loaded network measure — so synthetic
    ground-truth generation stays out of the schema.
    """
    name = str(getattr(inf, "name", "inference"))
    lik = getattr(inf, "likelihood", None)
    source = str((getattr(lik, "source", None) or ["recorded_ts"])[0])
    sigma = getattr(lik, "sigma", None)
    noise_family = str(getattr(lik, "name", None) or "Normal")
    sampler = str(getattr(inf, "sampler", None) or "nuts").lower()
    if sampler != "nuts":
        raise ValueError(
            f"inference {name!r} declares sampler={sampler!r}; the tvboptim backend "
            f"emits numpyro NUTS only. Declare `sampler: nuts`, or add the mapping."
        )
    n_warmup = int(getattr(inf, "num_warmup", None) or 1000)
    n_samples = int(getattr(inf, "num_samples", None) or 1000)
    n_chains = int(getattr(inf, "num_chains", None) or 1)
    seed = int(getattr(inf, "seed", None) or 0)

    def _var(dotted: str) -> str:
        return "_p_" + "".join(c if c.isalnum() else "_" for c in dotted)

    def _pred_stmts(cfg: str, target: str, indent: str) -> List[str]:
        """Statements computing the observed/predicted observable into ``target``,
        hoisting the compute_all_observations call to a single temp."""
        if source in network_obs_names:
            return [f"{indent}{target} = {source}"]
        tmp = f"_oa{target}"
        acc = f"compute_all_observations(model_fn({cfg}), {cfg}, transient).{source}"
        if source in derived_names:
            return [f"{indent}{target} = {acc}"]
        return [f"{indent}{tmp} = {acc}", f"{indent}{target} = {tmp}.data if hasattr({tmp}, 'data') else {tmp}"]

    model = [f"def _bayes_model_{name}(v_obs):", "    _cfg = state"]
    for key, prior in (getattr(inf, "priors", None) or {}).items():
        access = resolve_config_access(str(key), coupling_keys, external_keys)
        model += [
            f'    {_var(str(key))} = numpyro.sample("{key}", {dist_expr(getattr(prior, "distribution", None))})',
            f"    _cfg = eqx.tree_at(lambda _s: _s.{access}, _cfg, {_var(str(key))})",
        ]
    model += _pred_stmts("_cfg", "_pred", "    ")
    model += [f'    numpyro.sample("obs", dist.{noise_family}(_pred, {sigma}), obs=v_obs)']

    runner = [
        f"# Observed data for '{name}': the likelihood.source observation (runtime-bound or loaded).",
        *_pred_stmts("state", f"_v_obs_{name}", ""),
        f"_v_obs_{name} = kwargs.get('{source}', _v_obs_{name})",
        f"_nuts_{name} = numpyro.infer.NUTS(_bayes_model_{name}, dense_mass=True)",
        f"_mcmc_{name} = numpyro.infer.MCMC(_nuts_{name}, num_warmup={n_warmup}, num_samples={n_samples}, num_chains={n_chains}, progress_bar=False)",
        f"_mcmc_{name}.run(jax.random.key({seed}), _v_obs_{name})",
        f"_post_{name} = _mcmc_{name}.get_samples()",
        f"results.setdefault('inferences', Bunch())['{name}'] = InferenceResult("
        f"name='{name}', posterior=_post_{name}, "
        f"diagnostics=numpyro.diagnostics.summary(_mcmc_{name}.get_samples(group_by_chain=True)))",
    ]
    return "\n".join(f"        {ln}" for ln in (model + [""] + runner))


# =============================================================================
# State Variable Bounds
# =============================================================================


def materialise_lazy_params(parameters: Any, experiment: Any = None) -> Dict[str, tuple]:
    """Resolve every sourced/produced parameter to a content-addressed artifact.

    A `Parameter` carrying `producer:`, `source:` or `used:` has no literal `value`, so
    the emission sites would otherwise fall back to a scalar default and silently drop
    the array — a per-node operator would become ``jnp.full(shape, 1.0)`` and the model
    would run with the geometry erased. Materialising here writes the array once at
    codegen time and lets the generated module read it back, so an operator of any size
    costs nothing in the generated source.

    Returns ``{param_name: (path, key)}`` for the lazy parameters only; literals and
    free parameters are absent. Empty when *experiment* is None, because materialising
    runs the producer and this must stay side-effect-free when called as a predicate.
    """
    from pathlib import Path

    from tvbo.data import param_io

    if not parameters or experiment is None:
        return {}
    src_file = getattr(experiment, "_source_file", None)
    src_dir = Path(src_file).parent if src_file else None

    params = list(parameters.values()) if hasattr(parameters, "values") else list(parameters)
    out: Dict[str, tuple] = {}
    for p in params:
        if not param_io.is_lazy(p):
            continue
        path, key = param_io.materialise(p, source_dir=src_dir, context=experiment)
        out[str(p.name)] = (str(path), key)
    return out


def get_noise_covariance(model: Any, experiment: Any = None) -> Optional[Dict[str, Any]]:
    """The noise covariance a model declares, ready for the solver template to emit.

    Mirrors the provenance rule every other array-valued quantity follows: a literal
    binds inline, while a sourced or produced matrix (the usual case — a projection
    operator, a connectome-derived structure) is materialised codegen-time to a
    content-addressed artifact and emitted as a lazy ``(file, key)`` the generated
    module reads at run time, so a large operator never enters the generated source.

    Every noisy state variable must agree: one Wiener increment is drawn per step for
    all of them, so two different covariances cannot both be imposed on it.

    Args:
        model: Dynamics instance with ``.state_variables``.
        experiment: The declaring experiment, used to ground a relative source path and
            to resolve a producer's arguments. Without it a lazy covariance stays
            deferred, so calling this as a cheap predicate has no side effects.

    Returns:
        ``{"axis": str, "value": [[...]] | None, "lazy": (path, key) | None}``, or None
        when no covariance is declared.
    """
    from pathlib import Path

    import numpy as np

    from tvbo.data import param_io

    svs = getattr(model, "state_variables", None) if model else None
    if not svs:
        return None

    src_file = getattr(experiment, "_source_file", None) if experiment is not None else None
    src_dir = Path(src_file).parent if src_file else None

    found = None
    for sv_name, sv in svs.items():
        noise = getattr(sv, "noise", None)
        cov = getattr(noise, "covariance", None) if noise is not None else None
        if cov is None:
            continue
        axis = getattr(noise, "correlated_over", None)
        if axis is None:
            raise ValueError(
                f"state variable {sv_name!r}: `noise.covariance` is set but "
                f"`noise.correlated_over` is not. A covariance without a named axis "
                f"does not identify a process; set it to `node` (or `state`)."
            )
        entry = {"axis": str(axis), "value": None, "lazy": None}
        if param_io.is_lazy(cov):
            if experiment is not None:
                path, key = param_io.materialise(cov, source_dir=src_dir, context=experiment)
                entry["lazy"] = (str(path), key)
        else:
            entry["value"] = np.asarray(param_io.resolve(cov, source_dir=src_dir, context=experiment), dtype=float).tolist()
        if found is None:
            found = (entry, sv_name)
        elif found[0] != entry:
            raise ValueError(
                f"state variables {found[1]!r} and {sv_name!r} declare different "
                f"correlated noise; one Wiener increment is shared across all noisy "
                f"states, so they must declare the same covariance and axis."
            )
    return found[0] if found else None


def get_state_bounds(model: Any) -> Tuple[List, List, bool]:
    """Extract state variable bounds as SymPy expressions.

    Uses ``sympy.oo`` for unbounded dimensions so that code printers
    automatically render the correct backend literal (``jnp.inf``,
    ``np.inf``, ``Inf``, etc.).

    Args:
        model: Dynamics instance with ``.state_variables``

    Returns:
        tuple: (bounds_lo, bounds_hi, has_finite_bounds)
            - bounds_lo: list of sympy expressions (Float or -oo)
            - bounds_hi: list of sympy expressions (Float or oo)
            - has_finite_bounds: True if any bound is finite
    """
    from sympy import oo, Float

    bounds_lo: list = []
    bounds_hi: list = []

    if not model or not getattr(model, "state_variables", None):
        return bounds_lo, bounds_hi, False

    import math
    from tvbo.utils import domain_enforcement

    def _finite(b):
        """Whether a clamp bound is a finite real number (not None / ±inf).

        Range.lo/hi may also be an argument-name string or a sympy symbol
        (the schema permits both); those are treated as unbounded.
        """
        try:
            return b is not None and math.isfinite(float(b))
        except (TypeError, ValueError):
            return False

    # A ``domain`` only constrains integration when its ``enforce`` attribute opts
    # in. ``enforce: clamp`` hard-clips to [lo, hi]; ``none`` (default) treats the
    # domain as descriptive metadata (expected/plot range, optimisation hints,
    # IC-sampling support) and never alters the dynamics — so e.g. a phase θ with
    # domain [0, 2π] and no enforcement is left unclamped. The legacy ``boundaries``
    # slot is folded into ``domain`` with ``enforce: clamp`` by the Dynamics loader.
    for _sv_name, sv in model.state_variables.items():
        lo, hi = None, None
        dom = getattr(sv, "domain", None)
        enforce = domain_enforcement(dom)
        if enforce == "wrap":
            raise NotImplementedError(
                f"State variable '{_sv_name}' uses domain enforce='wrap', which the "
                f"tvboptim backend does not yet support. Use 'clamp' or 'none'."
            )
        if dom is not None and enforce == "clamp":
            lo = getattr(dom, "lo", None)
            hi = getattr(dom, "hi", None)
        bounds_lo.append(Float(lo) if _finite(lo) else -oo)
        bounds_hi.append(Float(hi) if _finite(hi) else oo)

    has_finite = any(v != -oo for v in bounds_lo) or any(v != oo for v in bounds_hi)
    return bounds_lo, bounds_hi, has_finite


def format_bounds_array(bounds: List, format: str = "jax") -> str:
    """Render a list of SymPy bound values as a code-level list literal.

    Uses the appropriate TVBO code printer so infinity is rendered
    correctly for any backend (``jnp.inf``, ``np.inf``, ``Inf``, …).

    Args:
        bounds: list of sympy expressions (Float / oo / -oo)
        format: target backend (``'jax'``, ``'numpy'``, ``'julia'``, …)

    Returns:
        String like ``[-10.0, -jnp.inf]`` ready for code generation.
    """
    from tvbo.codegen.code import get_printer

    printer = get_printer(format)
    parts = [printer.doprint(v) for v in bounds]
    return "[" + ", ".join(parts) + "]"


# =============================================================================
# Observation Helpers
# =============================================================================


def is_network_observation(obs: Any) -> bool:
    """Check if observation is bound from data rather than the simulation state.

    True when the source starts with ``network.observations``, ``network.edges``
    (data carried by the model network), or ``dataset.subject`` (a per-subject
    empirical target resolved from the dataset). All three are materialized into
    a module-level constant and bound at ``run_experiment`` time via
    ``_bind_network_observations``, not recorded from the solver. The slot is
    multivalued; accept both scalar and list forms.
    """
    if not obs:
        return False
    source = getattr(obs, "source", None)
    if not source:
        return False
    if isinstance(source, (list, tuple)):
        items = source
    else:
        items = [source]
    for item in items:
        name = item.name if hasattr(item, "name") else item
        s = str(name)
        if s.startswith("network.observations") or s.startswith("network.edges") or s.startswith("dataset.subject"):
            return True
    return False


def is_external_observation(obs: Any) -> bool:
    """Check if observation is external (has data_source or network.observations source)."""
    if not obs:
        return False
    # Explicit data_source
    if getattr(obs, "data_source", None):
        return True
    # Source pointing to network.observations.*
    return is_network_observation(obs)


def obs_has_all_args(obs: Any) -> bool:
    """Check if observation has all required arguments satisfied.

    Returns True if all pipeline step arguments either have values
    or are implicitly satisfied by source.
    """
    if getattr(obs, "class_reference", None):
        return True

    pipeline = getattr(obs, "pipeline", None) or []
    has_source = getattr(obs, "source", None) or getattr(obs, "source_observation", None)

    for step_idx, func in enumerate(pipeline):
        is_first_step = step_idx == 0
        # Function arguments are keyed by name (dict); tolerate a legacy list too.
        args = getattr(func, "arguments", None) or {}
        arg_items = args.items() if hasattr(args, "items") else [(getattr(a, "name", None), a) for a in args]
        for arg_name, arg in arg_items:
            arg_name = arg_name or getattr(arg, "name", None)
            if arg_name and getattr(arg, "value", None) is None:
                # First step's data-like args are satisfied by source
                if is_first_step and has_source and arg_name in ("data", "X", "x", "input", "timeseries", "a"):
                    continue
                return False
    return True


def get_observation_refs(observations_dict: Dict[str, Any]) -> Tuple[Set[str], List[str]]:
    """Categorize observations into network vs simulation-derived.

    Returns:
        (network_observation_names, observation_names_with_all_args)
    """
    network_obs = set()
    valid_obs = []

    for name, obs in observations_dict.items():
        if is_network_observation(obs):
            network_obs.add(name)
        if obs_has_all_args(obs):
            valid_obs.append(name)

    return network_obs, valid_obs


def get_observation_dependencies(obs_name: str, derived_obs_dict: Dict[str, Any], all_observations: Any) -> Set[str]:
    """Observations that ``obs_name`` derives from — its ``source`` entries that are
    themselves observations (edges in the observation dependency graph).

    ``all_observations`` is the full observation collection (its membership test
    filters sources down to observation references, ignoring result/state sources).
    """
    deps: Set[str] = set()
    dobs_def = derived_obs_dict.get(obs_name)
    if dobs_def:
        for src in dobs_def.source or []:
            key = getattr(src, "name", None) or src
            if key in all_observations:
                deps.add(str(src.name) if hasattr(src, "name") else str(src))
    return deps


def toposort_observations(obs_names: List[str], derived_obs_dict: Dict[str, Any], all_observations: Any) -> List[str]:
    """Dependency-order observations so any that lists another as a ``source`` is
    emitted AFTER that source — the same dependency-graph principle used for derived
    variables/parameters (see ``tvbo.classes.equation``). Independent observations
    keep their input order (stable / deterministic). Lives in the tvboptim adapter so
    the mako templates only call it rather than redefining the sort inline.
    """
    sorted_obs: List[str] = []
    visited: Set[str] = set()
    obs_set = set(obs_names)

    def visit(name):
        """Depth-first visit `name`, emitting its in-scope dependencies first.

        Recurses into each dependency that is itself part of `obs_names`, then
        appends `name` to `sorted_obs`, so every source lands before the
        observation that derives from it.

        Args:
            name: Observation to place after every observation it derives from.
        """
        if name in visited:
            return
        visited.add(name)
        for dep in get_observation_dependencies(name, derived_obs_dict, all_observations):
            if dep in obs_set:
                visit(dep)
        sorted_obs.append(name)

    for name in obs_names:
        visit(name)
    return sorted_obs


# =============================================================================
# Loss Function Parsing
# =============================================================================


def parse_loss_arguments(loss_call: Any) -> Tuple[List[Dict], Set[str]]:
    """Parse loss function call arguments.

    Returns:
        (parsed_args, obs_refs) where:
        - parsed_args: list of dicts with 'name', 'type', and type-specific keys
        - obs_refs: set of observation names referenced
    """
    loss_args = getattr(loss_call, "arguments", None) or {}  # keyed by name
    parsed_args = []
    obs_refs = set()

    for arg_name, arg in loss_args.items():
        arg_value = getattr(arg, "value", None)

        if not arg_name:
            continue

        if arg_value is not None:
            val_str = str(arg_value)

            # Check if numeric constant
            try:
                float(arg_value)
                parsed_args.append(
                    {
                        "name": arg_name,
                        "type": "constant",
                        "value": arg_value,
                    }
                )
                continue
            except (ValueError, TypeError):
                pass

            # Parse observation references
            if val_str.startswith("observations."):
                parts = val_str.split(".", 2)
                obs_name = parts[1] if len(parts) > 1 else None
                output_key = parts[2] if len(parts) > 2 else None
                if obs_name:
                    obs_refs.add(obs_name)
                    parsed_args.append(
                        {
                            "name": arg_name,
                            "type": "observation",
                            "obs_name": obs_name,
                            "output_key": output_key,
                        }
                    )
            elif "." in val_str:
                # Old-style obs_name.key
                obs_name, output_key = val_str.split(".", 1)
                obs_refs.add(obs_name)
                parsed_args.append(
                    {
                        "name": arg_name,
                        "type": "observation",
                        "obs_name": obs_name,
                        "output_key": output_key,
                    }
                )
            else:
                # Just observation name
                obs_refs.add(val_str)
                parsed_args.append(
                    {
                        "name": arg_name,
                        "type": "observation",
                        "obs_name": val_str,
                        "output_key": None,
                    }
                )
        else:
            # No value = runtime input
            parsed_args.append(
                {
                    "name": arg_name,
                    "type": "runtime",
                    "kwarg_name": arg_name,
                }
            )

    return parsed_args, obs_refs


def parse_loss_function(opt: Any) -> Optional[Dict]:
    """Parse optimization loss function specification.

    Returns dict with: opt_name, func_name, args, obs_refs, agg_over, agg_type
    or None if no loss defined.
    """
    loss_call = getattr(opt, "loss", None)
    if not loss_call:
        return None

    # Determine function name
    func_ref = getattr(loss_call, "function", None)
    callable_ref = getattr(loss_call, "callable", None)

    if func_ref:
        func_name = str(func_ref) if isinstance(func_ref, str) else (getattr(func_ref, "name", None) or str(func_ref))
    elif callable_ref:
        func_name = getattr(callable_ref, "name", None) or getattr(callable_ref, "qualname", None) or "loss"
    else:
        func_name = "loss"

    # Parse aggregate specification
    aggregate = getattr(loss_call, "aggregate", None)
    agg_over = None
    agg_type = None
    if aggregate:
        agg_over = str(getattr(aggregate, "over", "")).split(".")[-1] or None
        agg_type = str(getattr(aggregate, "type", "mean")).split(".")[-1]

    # Parse arguments
    parsed_args, obs_refs = parse_loss_arguments(loss_call)

    return {
        "opt_name": getattr(opt, "name", None) or "loss",
        "func_name": func_name,
        "args": parsed_args,
        "obs_refs": obs_refs,
        "agg_over": agg_over,
        "agg_type": agg_type,
    }


# =============================================================================
# Parameter Parsing
# =============================================================================


def get_domain_bounds(param_name: str, model: Any, all_couplings: Dict) -> Tuple[Optional[float], Optional[float]]:
    """Lookup domain bounds from model.parameters or coupling.parameters.

    Returns (lo, hi) tuple, where None means unbounded.
    """

    def extract_bounds(param):
        """Read `(lo, hi)` domain bounds from a parameter as floats.

        Args:
            param: A schema parameter whose optional `domain` carries `lo`/`hi`.

        Returns:
            A `(lo, hi)` tuple of floats; either is `None` when unset or when the
            bound is not numeric.
        """
        domain = getattr(param, "domain", None)
        if not domain:
            return (None, None)
        bounds = []
        for attr in ("lo", "hi"):
            value = getattr(domain, attr, None)
            try:
                bounds.append(float(value) if value is not None else None)
            except (TypeError, ValueError):
                bounds.append(None)
        return (bounds[0], bounds[1])

    # Check dynamics parameters
    if model and hasattr(model, "parameters") and param_name in model.parameters:
        lo, hi = extract_bounds(model.parameters[param_name])
        if lo is not None or hi is not None:
            return (lo, hi)

    # Check coupling parameters
    for cobj in all_couplings.values():
        if hasattr(cobj, "parameters") and cobj.parameters and param_name in cobj.parameters:
            return extract_bounds(cobj.parameters[param_name])

    return (None, None)


def parse_free_param(fp: Any, coupling_keys: Set[str], model: Any = None, all_couplings: Dict = None) -> Optional[Dict]:
    """Parse a free_parameter entry.

    Handles: str, dotted notation, stringified dict, dict, and Parameter objects.

    Returns dict with: name, heterogeneous, shape, coupling_key, dynamics_key,
                       lower_bound, upper_bound
    """
    all_couplings = all_couplings or {}
    result = None
    source_key = None
    is_coupling = False

    # FreeParameter wrapper object: has .parameter (dotted ref), .heterogeneous, .shape, .domain
    if hasattr(fp, "parameter") and getattr(fp, "parameter", None):
        ref = str(fp.parameter)
        if "." in ref:
            prefix, param_name = ref.rsplit(".", 1)
            is_coupling = prefix in coupling_keys
            source_key = prefix
        else:
            param_name = ref
        result = {
            "name": param_name,
            "heterogeneous": bool(getattr(fp, "heterogeneous", False)),
            "shape": str(fp.shape) if getattr(fp, "shape", None) else None,
            "coupling_key": source_key if is_coupling else None,
            "dynamics_key": source_key if not is_coupling and source_key else None,
        }
        domain = getattr(fp, "domain", None)
        if domain:
            lo = getattr(domain, "lo", None)
            hi = getattr(domain, "hi", None)
            if lo is not None:
                try:
                    result["lower_bound"] = float(lo)
                except (TypeError, ValueError):
                    pass
            if hi is not None:
                try:
                    result["upper_bound"] = float(hi)
                except (TypeError, ValueError):
                    pass
        # Optional optimizer start value (overrides the referenced Parameter's value),
        # so the descent can begin from a declared point without mutating the base config.
        iv = getattr(fp, "initial_value", None)
        if iv is not None:
            iv = getattr(iv, "value", iv)
            try:
                result["initial_value"] = float(iv)
            except (TypeError, ValueError):
                pass

    elif isinstance(fp, str):
        stripped = fp.strip()

        # Check for stringified dict
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, dict) and "name" in parsed:
                    param_name = str(parsed["name"])
                    if "." in param_name:
                        prefix, param_name = param_name.rsplit(".", 1)
                        is_coupling = prefix in coupling_keys
                        source_key = prefix
                    result = {
                        "name": param_name,
                        "heterogeneous": bool(parsed.get("heterogeneous", False)),
                        "shape": parsed.get("shape"),
                        "coupling_key": source_key if is_coupling else None,
                        "dynamics_key": source_key if not is_coupling and source_key else None,
                    }
            except (ValueError, SyntaxError):
                pass

        if result is None:
            # Check for dotted notation
            if "." in stripped:
                prefix, param_name = stripped.rsplit(".", 1)
                is_coupling = prefix in coupling_keys
                source_key = prefix
                result = {
                    "name": param_name,
                    "heterogeneous": False,
                    "shape": None,
                    "coupling_key": source_key if is_coupling else None,
                    "dynamics_key": source_key if not is_coupling else None,
                }
            else:
                result = {
                    "name": fp,
                    "heterogeneous": False,
                    "shape": None,
                    "coupling_key": None,
                    "dynamics_key": None,
                }

    elif isinstance(fp, dict) and "name" in fp:
        param_name = str(fp["name"])
        if "." in param_name:
            prefix, param_name = param_name.rsplit(".", 1)
            is_coupling = prefix in coupling_keys
            source_key = prefix
        result = {
            "name": param_name,
            "heterogeneous": bool(fp.get("heterogeneous", False)),
            "shape": str(fp["shape"]) if fp.get("shape") else None,
            "coupling_key": source_key if is_coupling else None,
            "dynamics_key": source_key if not is_coupling and source_key else None,
        }
        # Check for domain in dict
        domain = fp.get("domain", {})
        if isinstance(domain, dict):
            if "lo" in domain:
                try:
                    result["lower_bound"] = float(domain["lo"])
                except (TypeError, ValueError):
                    pass
            if "hi" in domain:
                try:
                    result["upper_bound"] = float(domain["hi"])
                except (TypeError, ValueError):
                    pass

    elif not isinstance(fp, (str, dict)):
        # Parameter object
        param_name = str(getattr(fp, "name", ""))
        if "." in param_name:
            prefix, param_name = param_name.rsplit(".", 1)
            is_coupling = prefix in coupling_keys
            source_key = prefix
        result = {
            "name": param_name,
            "heterogeneous": bool(getattr(fp, "heterogeneous", False)),
            "shape": str(fp.shape) if getattr(fp, "shape", None) else None,
            "coupling_key": source_key if is_coupling else None,
            "dynamics_key": source_key if not is_coupling and source_key else None,
        }
        # Check domain on Parameter object
        domain = getattr(fp, "domain", None)
        if domain:
            lo = getattr(domain, "lo", None)
            hi = getattr(domain, "hi", None)
            if lo is not None:
                try:
                    result["lower_bound"] = float(lo)
                except (TypeError, ValueError):
                    pass
            if hi is not None:
                try:
                    result["upper_bound"] = float(hi)
                except (TypeError, ValueError):
                    pass

    if result is None:
        return None

    # Set defaults
    result.setdefault("coupling_key", None)
    result.setdefault("dynamics_key", None)

    # Lookup bounds from model if not specified
    if "lower_bound" not in result or "upper_bound" not in result:
        if model or all_couplings:
            model_lo, model_hi = get_domain_bounds(result["name"], model, all_couplings)
            if "lower_bound" not in result and model_lo is not None:
                result["lower_bound"] = model_lo
            if "upper_bound" not in result and model_hi is not None:
                result["upper_bound"] = model_hi

    result.setdefault("lower_bound", None)
    result.setdefault("upper_bound", None)
    result.setdefault("shape", None)

    # Auto-detect coupling parameters
    if result.get("coupling_key") is None and model and all_couplings:
        param_name = result["name"]
        is_dynamics = hasattr(model, "parameters") and param_name in model.parameters
        if not is_dynamics:
            for ck, cobj in all_couplings.items():
                if hasattr(cobj, "parameters") and cobj.parameters and param_name in cobj.parameters:
                    result["coupling_key"] = ck
                    break

    return result


# =============================================================================
# Exploration Parsing
# =============================================================================


def normalize_n_parallel(expl: Any):
    """Normalise an ``Exploration.n_parallel`` to an int chunk size or ``"auto"``.

    ``"auto"`` (the schema default) defers the vmap chunk width to runtime, where the
    grid size is known — the generated script resolves it via
    :func:`tvbo.templates.tvboptim.callbacks.resolve_n_vmap`. An explicit integer is
    passed through unchanged (``1`` = fully sequential).

    Raises:
        ValueError: if ``n_parallel`` is a non-numeric string other than ``"auto"``
            (the schema allows any string, so a typo like ``"sequential"`` reaches
            here — fail with a clear message rather than a bare ``int()`` error).
    """
    v = getattr(expl, "n_parallel", None)
    if v is None:
        return "auto"
    if isinstance(v, str):
        s = v.strip()
        if s.lower() == "auto":
            return "auto"
        try:
            return int(s)  # numeric string (e.g. schema.py coerces an explicit int → "4")
        except ValueError:
            raise ValueError(f"Exploration.n_parallel must be a positive integer or 'auto', got {v!r}.") from None
    return int(v)


def jax_platform(accelerator: Any) -> Optional[str]:
    """Map an ``ExecutionConfig.accelerator`` to the JAX platform name it pins.

    ``'auto'`` (the schema default) means "let JAX detect the machine", which is
    expressed by leaving ``JAX_PLATFORMS`` unset — hence ``None`` rather than a
    string. ``'gpu'`` is JAX's ``'cuda'``; any other tier passes through lowercased,
    so a platform JAX gains needs no change here.
    """
    accel = str(accelerator or "auto").strip().lower()
    return None if accel == "auto" else {"gpu": "cuda"}.get(accel, accel)


def parse_exploration(expl: Any, all_couplings: Dict, get_pipeline_output_key_fn=None) -> Dict:
    """Parse exploration specification from YAML.

    Returns dict with: name, label, mode, n_parallel, axes, observable_*
    """
    exp_info = {
        "name": getattr(expl, "name", ""),
        "label": getattr(expl, "label", "") or "",
        "mode": getattr(expl, "mode", None) or "product",
        "n_parallel": normalize_n_parallel(expl),
        "axes": [],
        # Named observations to compute + stack per grid point (e.g. `loss`, or the
        # `analysis` diagnostics). Declarative alternative to a single scalar observable.
        "record": [str(r) for r in (getattr(expl, "record", None) or [])],
    }

    # Parse exploration axes (schema: `space` is keyed by parameter)
    axes_list = as_list(getattr(expl, "space", None))

    for axis in axes_list:
        domain = getattr(axis, "domain", None)
        if not domain:
            continue

        pname = str(getattr(axis, "parameter", ""))
        source_key = None
        is_coupling_param = False

        if "." in pname:
            prefix, pname = pname.rsplit(".", 1)
            is_coupling_param = prefix in all_couplings
            source_key = prefix

        # ExplorationAxis.reduce: collapse this axis by a statistic in the result
        # container instead of keeping it as a grid dim (statistic defaults to mean).
        _reduce = getattr(axis, "reduce", None)
        _reduce_stat = str(getattr(_reduce, "statistic", None) or "mean") if _reduce is not None else None

        exp_info["axes"].append(
            {
                "name": pname,
                "lo": float(getattr(domain, "lo", 0)),
                "hi": float(getattr(domain, "hi", 1)),
                "n": int(getattr(domain, "n", 10)),
                "is_coupling": is_coupling_param,
                "coupling_key": source_key if is_coupling_param else None,
                "dynamics_key": source_key if not is_coupling_param and source_key else None,
                "reduce": _reduce_stat,
            }
        )

    # Parse observable
    observable = getattr(expl, "observable", None)
    if observable:
        func = getattr(observable, "function", None)
        func_name = getattr(func, "name", None) if hasattr(func, "name") else str(func) if func else None
        # FunctionCall arguments are keyed by name (dict); tolerate a legacy list too.
        args = getattr(observable, "arguments", None) or {}

        if args:
            exp_info["observable_type"] = "function_call"
            exp_info["observable_func"] = func_name
            exp_info["observable_args"] = []
            arg_items = args.items() if hasattr(args, "items") else [(getattr(a, "name", None), a) for a in args]
            for arg_name, arg in arg_items:
                arg_name = arg_name or getattr(arg, "name", None) or str(arg)
                arg_value = getattr(arg, "value", None)
                if arg_value:
                    val_str = str(arg_value)
                    if "." in val_str:
                        obs_ref, output_key = val_str.split(".", 1)
                        exp_info["observable_args"].append({"name": arg_name, "obs": obs_ref, "key": output_key})
                    else:
                        exp_info["observable_args"].append({"name": arg_name, "obs": val_str, "key": "data"})
                else:
                    exp_info["observable_args"].append({"name": arg_name, "obs": None, "key": None})
        else:
            exp_info["observable_type"] = "observation"
            exp_info["observable"] = func_name
            if get_pipeline_output_key_fn and func_name:
                exp_info["output_key"] = get_pipeline_output_key_fn(func_name)
            else:
                exp_info["output_key"] = None

    return exp_info


# =============================================================================
# Algorithm Helpers
# =============================================================================


def get_include_info(inc: Any) -> Tuple[str, Dict]:
    """Extract algorithm name and argument overrides from AlgorithmInclude.

    Returns (algo_name, {param_name: value}) tuple.
    """
    if hasattr(inc, "algorithm"):
        algo = getattr(inc, "algorithm", None)
        algo_name = getattr(algo, "name", None) if hasattr(algo, "name") else str(algo)
        args = {}
        for arg in as_list(getattr(inc, "arguments", None)):
            name = getattr(arg, "name", None)
            if name:
                args[str(name)] = getattr(arg, "value", None)
        return algo_name, args
    return str(inc), {}


def _include_is_nested(inc: Any) -> bool:
    """True if an AlgorithmInclude uses nested (inner-loop) composition.

    Nested includes run the included algorithm's own run_<inner>() as a
    converging inner loop per outer iteration, so their rules/observations/
    hyperparameters belong to the inner call, NOT the flattened outer loop.
    """
    return str(getattr(inc, "mode", "combined") or "combined") == "nested"


def get_all_observations_from_algo(algo: Any, algorithms_dict: Dict) -> List[str]:
    """Get all observation names including from COMBINED included algorithms.

    Nested includes are skipped — their observations are computed inside the
    inner algorithm's own loop, not the outer one.
    """
    obs = []
    seen = set()

    # From included algorithms (combined-mode only)
    for inc in as_list(getattr(algo, "includes", None)):
        if _include_is_nested(inc):
            continue
        inc_name, _ = get_include_info(inc)
        inc_algo = algorithms_dict.get(inc_name)
        if inc_algo:
            for o in as_list(getattr(inc_algo, "observations", None)):
                o_str = str(o)
                if o_str not in seen:
                    obs.append(o_str)
                    seen.add(o_str)

    # This algorithm's observations
    for o in as_list(getattr(algo, "observations", None)):
        o_str = str(o)
        if o_str not in seen:
            obs.append(o_str)
            seen.add(o_str)

    return obs


def get_all_hyperparams(algo: Any, algorithms_dict: Dict) -> Dict:
    """Get all hyperparameters including from COMBINED included algorithms.

    Nested includes are skipped — their hyperparameters are passed directly to
    the inner algorithm's run_<inner>() call, not exposed on the outer signature.
    """
    all_hp = {}

    for inc in as_list(getattr(algo, "includes", None)):
        if _include_is_nested(inc):
            continue
        inc_name, arg_overrides = get_include_info(inc)
        inc_algo = algorithms_dict.get(inc_name)
        if inc_algo:
            for hp in as_list(getattr(inc_algo, "hyperparameters", None)):
                hp_name = str(getattr(hp, "name", ""))
                if hp_name in arg_overrides:
                    all_hp[hp_name] = arg_overrides[hp_name]
                else:
                    all_hp[hp_name] = getattr(hp, "value", None)

    for hp in as_list(getattr(algo, "hyperparameters", None)):
        all_hp[str(getattr(hp, "name", ""))] = getattr(hp, "value", None)

    return all_hp


# ---------------------------------------------------------------------------
# Network edge references (network.weight(s)/length(s) → connectome matrices)
# ---------------------------------------------------------------------------
# `weight`/`weights`/`length`/`lengths` are ergonomic shortcuts for the canonical
# `network.edges.<label>`; both resolve to a connectome matrix via Network.matrix().
_NETWORK_EDGE_ALIASES = {"weight": "weight", "weights": "weight", "length": "length", "lengths": "length"}


def edge_label(ref: Any) -> Optional[str]:
    """Canonical ``Network.matrix()`` label for a network reference, else None.

    Accepts the fully-qualified form (``network.weight``, ``network.edges.length``),
    the explicit ``edges.<label>`` form (any label), and the bare
    ``weight(s)``/``length(s)`` shortcut. Returns None for anything that is not a
    connectome-matrix reference (state variables, ``network.observations.*``, ...),
    which callers route through their normal path.
    """
    if not isinstance(ref, str):
        return None
    r = ref[len("network.") :] if ref.startswith("network.") else ref
    if r.startswith("edges."):
        return r.split("edges.", 1)[1] or None
    return _NETWORK_EDGE_ALIASES.get(r)


def edge_const(label: str) -> str:
    """Module-constant identifier holding the embedded matrix for ``label``."""
    import re

    return "_network_edge_" + re.sub(r"\W", "_", label)


# ---------------------------------------------------------------------------
# `network.`-scoped exploration axes sweep a live leaf of the backend graph, so
# every cell sees its own value without a Network or graph rebuild. Each entry
# below names the graph leaf carrying an attribute; adding a sweepable edge
# attribute is one entry here plus a graph leaf that holds it.
_NETWORK_EDGE_GRAPH_LEAVES = {"weight": "weights", "length": "lengths"}
_NETWORK_SCALAR_GRAPH_LEAVES = {"conduction_speed": "speed"}


def network_axis_leaf(ref: Any) -> Optional[str]:
    """Graph leaf swept by a ``network.``-scoped exploration axis, else None.

    Accepts the canonical ``network.edges.<label>`` form and the
    ``network.weight(s)``/``network.length(s)`` shortcuts (both via
    ``edge_label``, so axes and observations resolve a matrix identically), plus
    the network's own scalars (``network.conduction_speed``). Returns None for a
    reference outside the ``network.`` scope, which callers route through the
    dynamics/coupling path.

    Raises:
        ValueError: the reference is ``network.``-scoped but names an attribute
            with no live graph leaf to sweep — failing at codegen rather than
            silently writing the axis into the wrong scope.
    """
    if not isinstance(ref, str) or not ref.startswith("network."):
        return None
    attr = ref[len("network.") :]
    label = edge_label(ref)
    if label is not None:
        leaf = _NETWORK_EDGE_GRAPH_LEAVES.get(label)
        if leaf is None:
            raise ValueError(
                f"exploration axis '{ref}': edge attribute '{label}' has no graph "
                f"leaf to sweep (sweepable: "
                f"{', '.join('network.edges.' + k for k in sorted(_NETWORK_EDGE_GRAPH_LEAVES))}). "
                f"An edge attribute without a leaf can be referenced by an "
                f"observation, but cannot be swept."
            )
        return leaf
    leaf = _NETWORK_SCALAR_GRAPH_LEAVES.get(attr)
    if leaf is None:
        raise ValueError(
            f"exploration axis '{ref}': unknown network attribute '{attr}' "
            f"(sweepable: "
            f"{', '.join('network.' + k for k in sorted(_NETWORK_SCALAR_GRAPH_LEAVES))}, "
            f"{', '.join('network.edges.' + k for k in sorted(_NETWORK_EDGE_GRAPH_LEAVES))})."
        )
    return leaf


_INITIAL_CONDITIONS_SCOPE = "initial_conditions."


def initial_conditions_axis_sv(ref: Any) -> Optional[str]:
    """State variable swept by an ``initial_conditions.``-scoped axis, else None.

    ``initial_conditions.<state_var>`` sweeps the *initial value* of one state
    variable across grid cells — a deterministic initial-condition ensemble (one
    trajectory per swept value), as opposed to the stochastic ``n_trials`` +
    ``StateVariable.distribution`` ensemble. Returns the bare ``<state_var>``
    name; None for a reference outside the ``initial_conditions.`` scope, which
    callers route through the dynamics/coupling path. The name is validated
    against the model's state variables at codegen, where they are known.

    Raises:
        ValueError: the reference is ``initial_conditions.``-scoped but does not
            name a single state variable (empty, or a further-dotted path) —
            failing at codegen rather than sweeping nothing.
    """
    if not isinstance(ref, str) or not ref.startswith(_INITIAL_CONDITIONS_SCOPE):
        return None
    sv = ref[len(_INITIAL_CONDITIONS_SCOPE) :]
    if not sv or "." in sv:
        raise ValueError(
            f"exploration axis '{ref}': the initial_conditions scope takes a single "
            f"state-variable name (e.g. 'initial_conditions.E')."
        )
    return sv


def collect_network_edge_arrays(experiment: Any) -> Dict[str, list]:
    """Embed connectome matrices referenced by observations as ``{label: nested list}``.

    Scans every observation's ``source`` and every pipeline-step argument for a
    fully-qualified ``network.weight(s)``/``length(s)`` shortcut or explicit
    ``network.edges.<label>`` reference, resolving each to a dense matrix via
    ``Network.matrix()``. Covers derived and non-derived observations alike so the
    emitted constant serves both the observation-module source path and the
    experiment-module derived resolver. Raises if a referenced matrix is absent.
    """
    import numpy as np

    net = get_attr(experiment, "network", None)
    obs_map = get_attr(experiment, "observations", None) or {}
    obs_iter = obs_map.values() if hasattr(obs_map, "values") else obs_map
    arrays: Dict[str, list] = {}

    def add(val: Any) -> None:
        name = val if isinstance(val, str) else get_attr(val, "name", None)
        # Only fully-qualified `network.*` references embed a matrix; a bare `weight`
        # would be a state variable, not the connectome.
        if not isinstance(name, str) or not name.startswith("network."):
            return
        lab = edge_label(name)
        if not lab or lab in arrays:
            return
        mat = net.matrix(lab) if net is not None else None
        if mat is None:
            raise ValueError(
                f"An observation references {name} but the network has no {lab!r} "
                f"matrix to embed (Network.matrix({lab!r}) is None)."
            )
        arrays[lab] = np.asarray(mat, dtype=float).tolist()

    for obs in obs_iter:
        for src in get_attr(obs, "source", None) or []:
            add(src)
        for stage in get_attr(obs, "pipeline", None) or []:
            stage_args = get_attr(stage, "arguments", None) or {}
            for arg in stage_args.values() if hasattr(stage_args, "values") else stage_args:
                add(get_attr(arg, "value", None))
    return arrays


# ---------------------------------------------------------------------------
# Network node references (network.positions/instrength → per-node vectors)
# ---------------------------------------------------------------------------
# Node-level analogue of the edge-matrix refs above: a callable argument or an
# observer (Observation.dynamics) parameter may reference a per-node vector derived
# from the network — its region centroids (for mesh building) or its in-strength
# (weighted in-degree). Both embed once as a module constant, exactly like a
# connectome matrix, so any backend consumes them without a live Network object.
_NETWORK_NODE_MEASURES = ("positions", "instrength")


def node_label(ref: Any) -> Optional[str]:
    """Canonical per-node vector name for a ``network.<measure>`` reference, else None.

    Recognises ``network.positions`` (region centroids, ``(n_nodes, 3)``) and
    ``network.instrength`` (weighted in-degree, ``(n_nodes,)``). Accepts BOTH the
    fully-qualified ``network.positions`` form (observation source / collect scan)
    and the bare ``positions`` key that ``parse_reference`` hands ``ref_to_code``
    (it splits ``network.X`` into ``('network', 'X')``) — mirroring ``edge_label``,
    so the emitted constant name and the resolved reference cannot disagree.
    Returns None for everything else (edge matrices, state variables,
    ``network.observations.*``), which callers route through their normal path.
    """
    if not isinstance(ref, str):
        return None
    measure = ref[len("network.") :] if ref.startswith("network.") else ref
    return measure if measure in _NETWORK_NODE_MEASURES else None


def node_const(label: str) -> str:
    """Module-constant identifier holding the embedded per-node vector for ``label``."""
    import re

    return "_network_node_" + re.sub(r"\W", "_", label)


def collect_network_node_arrays(experiment: Any) -> Dict[str, list]:
    """Embed per-node vectors referenced by observations as ``{measure: nested list}``.

    Scans every observation's ``source``, its pipeline-step arguments, AND its
    observer (``dynamics``) parameters for a ``network.positions`` /
    ``network.instrength`` reference, resolving each once against the network:
    ``positions`` → ``Network.node_positions()``; ``instrength`` → the weighted
    in-degree ``matrix('weight').sum(axis=1)`` (row sum = incoming, the TVB/Koller
    convention, ``koller2024_networks.instrength_normalize``). Raises if a
    referenced vector cannot be built.
    """
    from tvbo.data import param_io

    net = get_attr(experiment, "network", None)
    obs_map = get_attr(experiment, "observations", None) or {}
    obs_iter = obs_map.values() if hasattr(obs_map, "values") else obs_map
    arrays: Dict[str, list] = {}

    def _resolve(measure: str):
        return None if net is None else param_io.resolve_network_node(net, measure)

    def add(val: Any) -> None:
        name = val if isinstance(val, str) else get_attr(val, "name", None)
        lab = node_label(name)
        if not lab or lab in arrays:
            return
        vec = _resolve(lab)
        if vec is None:
            raise ValueError(
                f"An observation references network.{lab} but it cannot be built from "
                f"the network (node positions absent, or Network.matrix('weight') is None)."
            )
        arrays[lab] = vec.tolist()

    for obs in obs_iter:
        for src in get_attr(obs, "source", None) or []:
            add(src)
        for stage in get_attr(obs, "pipeline", None) or []:
            stage_args = get_attr(stage, "arguments", None) or {}
            for arg in stage_args.values() if hasattr(stage_args, "values") else stage_args:
                add(get_attr(arg, "value", None))
        dyn = get_attr(obs, "dynamics", None)
        pmap = (get_attr(dyn, "parameters", None) or {}) if dyn is not None else {}
        for p in pmap.values() if hasattr(pmap, "values") else pmap:
            add(get_attr(p, "source", None))
    return arrays
