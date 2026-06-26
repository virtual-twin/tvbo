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


def as_list(obj: Any) -> list:
    """Convert dict or list to list of values."""
    if obj is None:
        return []
    if hasattr(obj, "values"):
        return list(obj.values())
    return list(obj)


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
    state_names = list(model.state_variables.keys()) if model and model.state_variables else []
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
    """Return the first named pipeline argument object."""
    for step in pipeline or []:
        for argument in getattr(step, "arguments", None) or []:
            if getattr(argument, "name", None) == name:
                return argument
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

    for _sv_name, sv in model.state_variables.items():
        lo, hi = None, None
        # A ``domain`` only constrains integration when its ``enforce`` attribute
        # opts in. ``enforce: clamp`` hard-clips to [lo, hi]; ``none`` (default)
        # treats the domain as descriptive metadata (expected/plot range,
        # optimisation hints, IC-sampling support) and never alters the dynamics
        # — so e.g. a phase θ with domain [0, 2π] and no enforcement is left
        # unclamped and may evolve past 2π. The legacy ``boundaries`` slot is
        # folded into ``domain`` with ``enforce: clamp`` by the Dynamics loader.
        from tvbo.utils import domain_enforcement

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

        def _finite(b):
            # A clamp bound is finite only if it is a real number that is not ±inf.
            # Range.lo/hi may also be an argument-name string or a sympy symbol
            # (schema permits both); those are treated as unbounded here.
            try:
                return b is not None and math.isfinite(float(b))
            except (TypeError, ValueError):
                return False

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
    """Check if observation is a network observation (static data from network).

    Network observations have source starting with 'network.observations' or 'network.edges'.
    The slot is multivalued; for raw network observations there is exactly
    one entry. Accept both scalar and list forms.
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
        if s.startswith("network.observations") or s.startswith("network.edges"):
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
        args = getattr(func, "arguments", None) or []
        for arg in args:
            if getattr(arg, "name", None) and getattr(arg, "value", None) is None:
                # First step's data-like args are satisfied by source
                if is_first_step and has_source and arg.name in ("data", "X", "x", "input", "timeseries", "a"):
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
    loss_args = getattr(loss_call, "arguments", None) or []
    parsed_args = []
    obs_refs = set()

    for arg in loss_args:
        arg_name = getattr(arg, "name", None)
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
        domain = getattr(param, "domain", None)
        if domain:
            lo = getattr(domain, "lo", None)
            hi = getattr(domain, "hi", None)
            try:
                return (float(lo) if lo is not None else None, float(hi) if hi is not None else None)
            except (TypeError, ValueError):
                pass
        return (None, None)

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
    }

    # Parse exploration axes (schema: `space` is a list of ExplorationAxis)
    axes_list = getattr(expl, "space", None) or []

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
        args = getattr(observable, "arguments", None) or []

        if args:
            exp_info["observable_type"] = "function_call"
            exp_info["observable_func"] = func_name
            exp_info["observable_args"] = []
            for arg in args:
                arg_name = getattr(arg, "name", None) or str(arg)
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
