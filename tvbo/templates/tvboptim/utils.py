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
            val = list(p.value)
        elif p.value is not None:
            val = float(p.value)
        else:
            val = 1.0
        param_defaults[p.name] = val
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
            src = getattr(obs, "source", None)
            if not src:
                continue
            src_name = str(src)
            if src_name in aux_names and src_name not in requested_aux:
                requested_aux.append(src_name)

    all_var_names = state_names + requested_aux
    return state_names, requested_aux, all_var_names


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

    # Vectorized (matmul) vs per-edge reduction. Only the legacy local-states-only
    # identity case is auto-vectorized; source-only phase couplings stay per-edge
    # (exact for any connectome) unless the coupling opts in with vectorized: true.
    vectorized = getattr(coupling, "vectorized", False)
    if not vectorized and local_states and not incoming_states:
        vectorized = True
    vec_states = list(dict.fromkeys(incoming_states + local_states))

    class_name = coupling_key.replace(" ", "").replace("-", "")
    base_class = "DelayedCoupling" if has_delay else "InstantaneousCoupling"
    interp_kw = "history_interpolation='linear', " if (has_delay and bool(getattr(coupling, "interpolate_delays", False))) else ""

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
    gx_symbols = ["gx_%d" % k for k in range(n_pre)] if n_pre > 1 else []
    post_alias_symbols = [a[0] for a in post_aliases_i]

    all_symbols = (
        param_names + incoming_states + local_states
        + ["gx", "G", "x_i", "x_j", "incoming_states", "local_states"]
        + alias_symbols + gx_symbols + post_alias_symbols
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
        "class_name": class_name,
        "base_class": base_class,
        "interp_kw": interp_kw,
        "post_aliases_i": post_aliases_i,
        "post_is_list": post_is_list,
        "state_aliases_j": state_aliases_j,
        "state_aliases_i": state_aliases_i,
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


def _reduction_init_value(sv: Any) -> float:
    """Initial scalar for an observer state (its declared value, else 0.0).

    Accumulators start at their reduction identity (0.0 for a sum); a memory state's
    init is irrelevant (it is overwritten on the first step), so 0.0 is a safe default.
    """
    for holder in (sv, get_attr(sv, "domain"), get_attr(sv, "distribution")):
        v = get_attr(holder, "value") if holder is not None else None
        if v is not None:
            try:
                return float(to_numeric(v))
            except Exception:
                pass
    return 0.0


def resolve_reduction(obs: Any) -> Optional[Dict[str, Any]]:
    """Lift an observation's auxiliary ``dynamics`` into a backend-agnostic reduction.

    An observation may declare a co-integrated auxiliary ``Dynamics`` (the observer)
    that computes it online as a time recurrence, instead of a post-scan ``pipeline``.
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
    vocabulary (its states, the source, and the framework scalars ``dt``/``count``;
    user functions become undefined ``Function``s). That makes the analysis symbolic,
    not string-based: accumulator classification is ``state_symbol in expr.free_symbols``,
    and an unknown symbol (a typo) is caught here rather than surfacing as a codegen
    error. The context carries sympy ``Expr`` objects; the partial renders them per
    backend via ``render_expression`` (which accepts sympy directly). Returns ``None``
    when the observation declares no ``dynamics`` (the post-scan path runs).
    """
    import sympy as sp

    dyn = get_attr(obs, "dynamics")
    if dyn is None:
        return None
    src = as_list(get_attr(obs, "source"))
    source = str(src[0]) if src else None

    svs = get_attr(dyn, "state_variables")
    sv_pairs = list(svs.items()) if hasattr(svs, "items") else []
    sv_names = [str(n) for n, _ in sv_pairs]

    # Symbolic vocabulary: observer states + source + framework scalars are Symbols;
    # the observer's user functions are undefined Functions. Parse every RHS against it.
    allowed = set(sv_names) | ({source} if source else set()) | {"dt", "count"}
    loc: Dict[str, Any] = {n: sp.Symbol(n) for n in allowed}
    loc["pi"] = sp.pi
    funcs = get_attr(dyn, "functions")
    functions: Dict[str, Any] = {}
    for fname, fn in (funcs.items() if hasattr(funcs, "items") else []):
        feq = get_attr(fn, "equation")
        fargs = [str(get_attr(a, "name", a)) for a in as_list(get_attr(fn, "arguments"))]
        frhs = get_attr(feq, "rhs") if feq is not None else None
        loc[str(fname)] = sp.Function(str(fname))
        fexpr = (sp.sympify(str(frhs), locals={**{a: sp.Symbol(a) for a in fargs}, "pi": sp.pi})
                 if frhs is not None else None)
        functions[str(fname)] = {"args": fargs, "expr": fexpr}

    def _parse(rhs_str: str, where: str):
        expr = sp.sympify(rhs_str, locals=loc)
        unknown = {str(s) for s in expr.free_symbols} - allowed
        if unknown:
            raise ValueError(
                f"Observation reduction {where}: unknown symbol(s) {sorted(unknown)}; "
                f"available are the observer states {sv_names}, the source {source!r}, "
                f"and dt/count."
            )
        return expr

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
        states.append({
            "name": name,
            "init": _reduction_init_value(sv),
            "update": expr,  # sympy Expr — the printer renders it
            "evict": evict,  # sympy Expr (sliding-window downdate) or None
            "is_accumulator": sp.Symbol(name) in expr.free_symbols,
        })

    outs = as_list(get_attr(dyn, "output"))
    dvs = get_attr(dyn, "derived_variables")
    dv_map = dict(dvs.items()) if hasattr(dvs, "items") else {}
    output = None
    if outs:
        out_name = str(outs[0])
        dv = dv_map.get(out_name)
        oeq = get_attr(dv, "equation") if dv is not None else None
        output = (_parse(str(get_attr(oeq, "rhs")), f"output {out_name!r}")
                  if oeq is not None else sp.Symbol(out_name))

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

    return {
        "source": source,
        "states": states,
        "output": output,
        "functions": functions,
        "statistic": statistic,   # 'mean' | 'median'
        "histogram": histogram,
        "windowed": windowed,     # True if any state declares an evict_equation (sliding window)
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
    class_info.setdefault("constructor_arg_codes", {
        name: _literal_code(value)
        for name, value in class_info.get("constructor_args", {}).items()
        if value is not None
    })
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
        tok for tok in (t.strip() for t in solver_kwargs.split(",")) if tok
        and not tok.startswith(("grad_horizon=", "block_size="))
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


def render_analysis_observations(
    analysis_obs: Dict[str, Any],
    coupling_keys: Set[str],
    solver_class: str,
    transient_time: float,
    t1_default: float,
    dt: float,
    solver_kwargs: str = "",
    model: Any = None,
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
    evaluation config. Returns a string whose lines are indented for a function body
    (4 spaces), empty string if there are no analysis observations.
    """
    window = f"t0=0.0 + {transient_time}, t1={transient_time} + {t1_default}, dt={dt}"
    solver_kwargs = _analysis_solver_kwargs(solver_kwargs)
    lines: List[str] = []

    # Linear-response analysis types (covariance / psd / fisher) are evaluated at the SAME
    # deterministic operating point via the metadata-symbolic network vector field + Jacobian.
    # Emit the vector field, the Jacobian, and the operating point (settle + A) ONCE — shared by
    # every such observation — then each observation is just a linear-algebra solve on the shared
    # ``_lr_A``. Structure comes from _linear_response.py.mako (backend-renderable); resolution
    # from linear_response_context — no Python string-emit of the code bodies.
    _LR_TYPES = {"covariance", "psd"}
    _lr_ctx = _lr_tpl = None

    def _emit_partial(_defname, **_kw):
        return _lr_tpl.get_def(_defname).render(**_kw).strip("\n").split("\n")

    if model is not None and any(
        str(getattr(a.analysis, "type", "") or "") in _LR_TYPES for a in analysis_obs.values()
    ):
        from tvbo import templates as _templates
        from tvbo.analysis.linear_response import linear_response_context

        _lr_ctx = linear_response_context(model)
        _lr_tpl = _templates.lookup.get_template("_linear_response.py.mako")
        _nsv = _lr_ctx["n_sv"]
        lines += [
            "_lr_weights = jnp.asarray(network.graph.weights)",
            "_lr_params = state.dynamics",
            f"_lr_x0 = jnp.broadcast_to(jnp.reshape(jnp.asarray(state.initial_state.dynamics), "
            f"({_nsv}, -1)), ({_nsv}, _lr_weights.shape[0]))",
        ]
        lines += _emit_partial("lr_vf", ctx=_lr_ctx)
        lines += _emit_partial("lr_jacobian", ctx=_lr_ctx)
        lines += _emit_partial("lr_operating_point", ctx=_lr_ctx)  # binds _lr_fp, _lr_A

    # Finite-difference observations that share the exact same per-seed computation
    # (same target, wrt, delta, seeds, seed_base) reuse ONE ``jax.lax.map`` — matching the
    # reference, which derives fd_mean and fd_sem from a single map — rather than
    # recomputing the seeds once per reduction. Assign a shared group id per signature.
    def _fd_signature(aobs):
        an = aobs.analysis
        p = {str(k): (v.value if hasattr(v, "value") else v)
             for k, v in (getattr(an, "parameters", None) or {}).items()}
        wrt = [str(w) for w in (getattr(an, "wrt", None) or [])]
        return (
            str(getattr(an, "target", None) or "loss"),
            _analysis_wrt_access(wrt, coupling_keys),
            float(p.get("delta", 0.3)), int(p.get("seeds", 8)), int(p.get("seed_base", 0)),
        )

    fd_group = {}  # signature -> group id
    for aobs in analysis_obs.values():
        if str(getattr(aobs.analysis, "type", "") or "") == "finite_difference":
            fd_group.setdefault(_fd_signature(aobs), f"fdgrp{len(fd_group)}")
    fd_emitted: Set[tuple] = set()

    for name, aobs in analysis_obs.items():
        an = aobs.analysis
        atype = str(getattr(an, "type", "") or "")
        params = {
            str(k): (v.value if hasattr(v, "value") else v)
            for k, v in (getattr(an, "parameters", None) or {}).items()
        }
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
                f"# Same integrator config as the main sim (coupling_evaluation) so lambda_1",
                f"# matches the trajectory it characterises. Emits the exponents and the",
                f"# per-node leading-vector profile ({name}_xi, paper's xi_i).",
                f"_le_solve, _le_cfg = prepare(network, {solver_class}({solver_kwargs}), t0=0.0, t1={seg}, dt={dt})",
            ]
            # Evaluate at the current operating point: sync `wrt` (e.g. the swept coupling)
            # into the segment config so a K-sweep reports lambda_1(K), not at the network's
            # fixed parameters. Optional — omit `wrt` for a static Lyapunov.
            if access:
                lines.append(f"_le_cfg = eqx.tree_at(lambda _c: _c.{access}, _le_cfg, state.{access})")
            lines.append(f"obs.{name}, obs.{name}_xi = benettin_spectrum_and_vectors(_le_solve, _le_cfg, t={seg}, n={n}, k={k})")
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
                    f"    _cs = eqx.tree_at(lambda _s: _s.noise.key, state, _key)",
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
        elif atype in ("covariance", "psd") and _lr_tpl is None:
            lines.append(f"# {name}: {atype} analysis needs the model threaded to "
                         f"render_analysis_observations — skipped.")
        elif atype == "covariance":
            # Stationary covariance (Lyapunov) on the shared operating point — Deco Fig 5, Eq 24.
            _sigma = float(params.get("sigma", 0.01))
            lines += _emit_partial("lr_covariance", ctx=_lr_ctx, name=f"_cov_{name}", sigma=_sigma)
            lines += [
                f"# {name}: excitatory-block stationary covariance via the Lyapunov equation",
                f"obs.{name} = _cov_{name}(_lr_A)",
            ]
        elif atype == "psd":
            # Analytic power spectrum per excitatory node on the shared A — Deco Fig 5, Eq 28.
            _sigma = float(params.get("sigma", 0.01))
            _flo = float(params.get("f_lo", 0.1))
            _fhi = float(params.get("f_hi", 50.0))
            _nf = int(params.get("n_freq", 128))
            lines += _emit_partial("lr_psd", ctx=_lr_ctx, name=f"_psd_{name}",
                                   sigma=_sigma, f_lo=_flo, f_hi=_fhi, n_freq=_nf)
            lines += [
                f"# {name}: analytic power spectrum per excitatory node (Eq 28)",
                f"obs.{name} = _psd_{name}(_lr_A)",
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
) -> str:
    """Render the body of an exploration ``observable_fn`` that records a `record:` list.

    Each recorded name resolves to ``compute_all_observations`` (derived / network /
    simulated observations) or ``compute_analysis_observations`` (the `analysis`
    diagnostics — Lyapunov, gradients). The observable returns a ``Bunch`` of the named
    values, which the exploration stacks over the grid into one array per name. Kept in
    the adapter (not the template) so the same routing serves any backend. Returns the
    function-body string (8-space indented for ``def observable_fn(s):`` inside the
    exploration function).
    """
    analysis_set = set(analysis_names)
    lines = ["result = _expl_model_fn(s)"]
    if any(n not in analysis_set for n in record_names):
        # Restrict the per-cell computation to the recorded observations and their
        # closure (passed by the caller), so non-recorded — possibly non-jittable —
        # observations never execute inside this jitted observable.
        if only_obs is not None:
            _only_lit = "{%s}" % ", ".join(repr(n) for n in sorted(only_obs))
            lines.append(f"_all_obs = compute_all_observations(result, s, result_transient, only={_only_lit})")
        else:
            lines.append("_all_obs = compute_all_observations(result, s, result_transient)")
    if any(n in analysis_set for n in record_names):
        lines.append("_an_obs = compute_analysis_observations(s, _network, result_transient)")
    entries = []
    for n in record_names:
        if n in analysis_set:
            entries.append(f"{n}=_an_obs.{n}")
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


def render_inference(inf: Any, coupling_keys: Set[str], external_keys: Set[str],
                     derived_names: Set[str], network_obs_names: Set[str]) -> str:
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
    noise = dist_expr(lik) if lik is not None else "dist.Normal"  # family (name), scale applied below
    noise_family = str(getattr(lik, "name", None) or "Normal")
    sampler = str(getattr(inf, "sampler", None) or "nuts")
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
        return [f"{indent}{tmp} = {acc}",
                f"{indent}{target} = {tmp}.data if hasattr({tmp}, 'data') else {tmp}"]

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
        if (s.startswith("network.observations") or s.startswith("network.edges")
                or s.startswith("dataset.subject")):
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
        for src in (dobs_def.source or []):
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


def parse_exploration(expl: Any, all_couplings: Dict, get_pipeline_output_key_fn=None) -> Dict:
    """Parse exploration specification from YAML.

    Returns dict with: name, label, mode, n_parallel, axes, observable_*
    """
    exp_info = {
        "name": getattr(expl, "name", ""),
        "label": getattr(expl, "label", "") or "",
        "mode": getattr(expl, "mode", None) or "product",
        "n_parallel": int(getattr(expl, "n_parallel", 1) or 1),
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

        exp_info["axes"].append(
            {
                "name": pname,
                "lo": float(getattr(domain, "lo", 0)),
                "hi": float(getattr(domain, "hi", 1)),
                "n": int(getattr(domain, "n", 10)),
                "is_coupling": is_coupling_param,
                "coupling_key": source_key if is_coupling_param else None,
                "dynamics_key": source_key if not is_coupling_param and source_key else None,
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
_NETWORK_EDGE_ALIASES = {"weight": "weight", "weights": "weight",
                         "length": "length", "lengths": "length"}


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
    r = ref[len("network."):] if ref.startswith("network.") else ref
    if r.startswith("edges."):
        return r.split("edges.", 1)[1] or None
    return _NETWORK_EDGE_ALIASES.get(r)


def edge_const(label: str) -> str:
    """Module-constant identifier holding the embedded matrix for ``label``."""
    import re
    return "_network_edge_" + re.sub(r"\W", "_", label)


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
        for src in (get_attr(obs, "source", None) or []):
            add(src)
        for stage in (get_attr(obs, "pipeline", None) or []):
            stage_args = get_attr(stage, "arguments", None) or {}
            for arg in (stage_args.values() if hasattr(stage_args, "values") else stage_args):
                add(get_attr(arg, "value", None))
    return arrays
