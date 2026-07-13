# -*- coding: utf-8 -*-
<%doc>TVB-Optim Algorithm Template. Context: experiment (SimulationExperiment).</%doc>
<%
from tvbo.codegen import render_expression
from tvbo.codegen.streaming_reducers import lookup_streaming_reducer
from tvbo.templates.tvboptim.utils import (
    safe_name, as_list, get_attr, is_network_observation, is_external_observation,
    get_include_info, get_all_observations_from_algo, get_all_hyperparams
)

# Backend key this template targets — used for streaming-reducer registry lookups.
_STREAMING_BACKEND = 'tvboptim'

# Define jaxcode locally
_exp_functions = experiment.functions or {}
if hasattr(_exp_functions, 'items'):
    _user_functions = {str(fname): str(fname) for fname in _exp_functions.keys()}
else:
    _user_functions = {str(f.name): str(f.name) for f in _exp_functions}

# jaxcode with broadcasting inference for proper array dimension handling
# When expressions use indexed notation (e.g., a[i,k], rmse[i]), the code generator
# automatically adds broadcasting dimensions (e.g., rmse[:, None]) as needed.
jaxcode = lambda expr, params=None: render_expression(
    expr, format='jax', user_functions=_user_functions,
    parameters=params, infer_broadcasting=True
)

_algos_raw = experiment.algorithms or {}
algorithms_list = list(_algos_raw.values()) if hasattr(_algos_raw, 'values') else list(_algos_raw)
has_algorithms = len(algorithms_list) > 0

# Build algorithms_dict for looking up included algorithms
algorithms_dict = {}
for _algo in algorithms_list:
    algorithms_dict[str(_algo.name)] = _algo

def get_func_name(func_call):
    """Get function name from FunctionCall."""
    return str(func_call.function) if func_call.function else None

def get_func_args(func_call):
    """Get arguments from FunctionCall as dict {name: value}."""
    if not func_call.arguments:
        return {}
    return {str(name): arg.value for name, arg in func_call.arguments.items()}

def get_target_name(rule):
    """Get target parameter name from UpdateRule."""
    tp = rule.target_parameter
    return str(tp.name) if hasattr(tp, 'name') else str(tp)

# Note: get_include_info is imported from utils

def get_obs_names(algo):
    """Get observation names as strings."""
    return [str(o) for o in as_list(getattr(algo, 'observations', None))]

def get_all_observations(algo, algorithms_dict):
    """Get all observation names including from included algorithms."""
    return get_all_observations_from_algo(algo, algorithms_dict)

def get_all_update_rules(algo, algorithms_dict):
    """Get all update rules including from included algorithms.

    Returns list of (rule, source_algo_name, arg_overrides) tuples.
    arg_overrides is a dict of {param_name: value} for hyperparameter overrides.
    Included algorithm rules come first, then this algorithm's rules.
    """
    all_rules = []
    # First, add rules from COMBINED-mode included algorithms with their argument
    # overrides. nested-mode includes are NOT flattened here — they run as a
    # converging inner loop (see get_nested_includes) on each outer iteration.
    for inc in as_list(getattr(algo, 'includes', None)):
        if _include_mode(inc) == 'nested':
            continue
        inc_name, arg_overrides = get_include_info(inc)
        inc_algo = algorithms_dict.get(inc_name)
        if inc_algo:
            for rule in as_list(getattr(inc_algo, 'update_rules', None)):
                all_rules.append((rule, inc_name, arg_overrides))
    # Then add this algorithm's own rules (no overrides needed)
    for rule in as_list(getattr(algo, 'update_rules', None)):
        all_rules.append((rule, str(algo.name), {}))
    return all_rules

def _include_mode(inc):
    """Composition mode of an AlgorithmInclude ('combined' default, or 'nested')."""
    return str(getattr(inc, 'mode', 'combined') or 'combined')

def get_nested_includes(algo, algorithms_dict, obs_dict):
    """Return nested-mode includes as call-ready descriptors.

    Each entry: {name, inner_iterations, hyperparam_values, external_inputs}.
    A nested include runs the included algorithm's own generated run_<inner>() as
    a converging inner loop on each outer iteration, so the inner algorithm's
    invariant (e.g. FIC's working point) is re-settled before the outer update.
    """
    result = []
    for inc in as_list(getattr(algo, 'includes', None)):
        if _include_mode(inc) != 'nested':
            continue
        inc_name, arg_overrides = get_include_info(inc)
        inc_algo = algorithms_dict.get(inc_name)
        if not inc_algo:
            continue
        inc_hp = get_hyperparam_dict(inc_algo)
        ordered_vals = [arg_overrides.get(k, inc_hp[k]) for k in inc_hp.keys()]
        inner_iters = getattr(inc, 'inner_iterations', None)
        if inner_iters is None:
            inner_iters = int(inc_algo.n_iterations)
        result.append({
            'name': safe_name(inc_name),
            'inner_iterations': int(inner_iters),
            'hyperparam_values': ordered_vals,
            'external_inputs': get_external_inputs(inc_algo, obs_dict, algorithms_dict),
        })
    return result

def get_external_inputs(algo, obs_dict, algorithms_dict=None):
    """Get observations that have external data_source or network.observations source."""
    obs_names = get_all_observations(algo, algorithms_dict or {}) if algorithms_dict else get_obs_names(algo)
    return [o for o in obs_names if is_external_observation(obs_dict.get(o))]

def get_simulated_observations(algo, obs_dict, algorithms_dict=None):
    """Get observations that are simulated (not external)."""
    obs_names = get_all_observations(algo, algorithms_dict or {}) if algorithms_dict else get_obs_names(algo)
    return [o for o in obs_names if not is_external_observation(obs_dict.get(o))]

def get_hyperparam_dict(algo):
    """Build {name: value} dict from hyperparameters (THIS algorithm only)."""
    return {str(hp.name): hp.value for hp in as_list(getattr(algo, 'hyperparameters', None))}

def get_all_functions(algo, algorithms_dict):
    """Get all functions including from included algorithms.

    Returns list of FunctionCall objects. Included algorithm functions come first.
    """
    all_funcs = []
    # First, add functions from included algorithms
    for inc in as_list(getattr(algo, 'includes', None)):
        inc_name, _ = get_include_info(inc)
        inc_algo = algorithms_dict.get(inc_name)
        if inc_algo:
            all_funcs.extend(as_list(getattr(inc_algo, 'functions', None)))
    # Then add this algorithm's own functions
    all_funcs.extend(as_list(getattr(algo, 'functions', None)))
    return all_funcs

# Extract observations dict for reference
_obs_raw = experiment.observations or {}
_all_observations = dict(_obs_raw.items()) if hasattr(_obs_raw, 'items') else {}

# Split observations into raw vs derived views.
from tvbo.codegen.templater import is_derived as _is_derived
observations_dict = {n: o for n, o in _all_observations.items() if not _is_derived(o, experiment)}
derived_observations_dict = {n: o for n, o in _all_observations.items() if _is_derived(o, experiment)}

# State variable names from model
model = experiment.dynamics
state_var_names = list(model.state_variables.keys()) if model and model.state_variables else []
state_names = state_var_names

# Recorded variable layout (states + auxiliaries-in-VOI). Matches the dfun's
# VARIABLES_OF_INTEREST and solution.variable_names from tvboptim >= 0.2.7.
from tvbo.templates.tvboptim.utils import get_recorded_variable_names
_, _recorded_aux, var_names = get_recorded_variable_names(model, experiment) if model else ([], [], [])

# Build coupling parameter lookup: param_name -> coupling_key
coupling_param_to_key = {}
network = experiment.network
if network and network.coupling:
    for coupling_key, coupling_obj in network.coupling.items():
        if coupling_obj.parameters:
            for param_name in coupling_obj.parameters.keys():
                coupling_param_to_key[param_name] = coupling_key
%>
% if has_algorithms:

% for algo in algorithms_list:
<%
    algo_name = safe_name(algo.name)
    n_iterations = int(algo.n_iterations)
    simulation_period = float(algo.simulation_period)

    # Get all update rules (combined-mode includes + own rules; nested excluded)
    all_update_rules_with_source = get_all_update_rules(algo, algorithms_dict)
    update_rules = [r for r, src, args in all_update_rules_with_source]  # Just the rules

    # Nested-mode includes: inner algorithms run to convergence per outer iteration
    nested_includes = get_nested_includes(algo, algorithms_dict, observations_dict)

    # Get all observations (including from included algorithms)
    observations = get_all_observations(algo, algorithms_dict)
    external_inputs = get_external_inputs(algo, observations_dict, algorithms_dict)
    simulated_observations = get_simulated_observations(algo, observations_dict, algorithms_dict)

    # Get all functions (including from included algorithms)
    algo_functions = get_all_functions(algo, algorithms_dict)

    # Get ALL hyperparameters (including from included algorithms)
    # This is needed for detecting sliding window pattern (window_size) and function args
    hyperparam_dict = get_all_hyperparams(algo, algorithms_dict)
    description = algo.description

    # Check for learning rate warmup - examine ALL update rules (including included)
    # to see if any have warmup: true AND an eta parameter
    def _check_warmup_needed(all_rules, algos_dict):
        """Check if any update rule requires learning rate warmup."""
        for rule, rule_source, arg_overrides in all_rules:
            if not getattr(rule, 'warmup', False):
                continue
            # Rule has warmup: true, check if it has an eta parameter
            rule_rhs = rule.equation.rhs if hasattr(rule, 'equation') and rule.equation else ''
            source_algo = algos_dict.get(rule_source)
            source_hp = get_hyperparam_dict(source_algo) if source_algo else {}
            effective_hp = {**source_hp, **arg_overrides}
            for pname in effective_hp.keys():
                if pname.lower() in ['eta', 'learning_rate', 'lr'] and pname in rule_rhs:
                    return True
        return False

    has_warmup = _check_warmup_needed(all_update_rules_with_source, algorithms_dict)

    # Check for included algorithms
    included_algos = as_list(algo.includes)
    has_includes = len(included_algos) > 0
%>

<%
    # Detect sliding window pattern:
    # 1. Check for window_size hyperparameter
    # 2. Find observations that are DerivedObservations (from derived_observations_dict)
    has_window_size = 'window_size' in hyperparam_dict
    window_size_val = int(hyperparam_dict.get('window_size', 150)) if has_window_size else 0

    # Find derived observations and their sources
    dependent_observations = {}
    source_observations_needed = set()

    for obs in simulated_observations:
        derived_obs_def = derived_observations_dict.get(obs)
        if derived_obs_def:
            # Filter source entries to names resolving to other observations.
            src_obs_list = derived_obs_def.source or []
            src_names = []
            for s in src_obs_list:
                _sn = s.name if hasattr(s, 'name') else str(s)
                if _sn in _all_observations:
                    src_names.append(_sn)
            dependent_observations[obs] = src_names
            for src_name in src_names:
                if src_name in observations_dict:
                    source_observations_needed.add(src_name)

    use_sliding_window = has_window_size and len(source_observations_needed) > 0

    # ── Streaming-reducer resolution (backend-independent, registry-gated) ──────
    # For each derived observation computed over a sliding-window source, look up
    # whether its FIRST pipeline reducer (e.g. compute_fc) has a registered
    # streaming form for this backend. When it does, the loop below replaces the
    # O(window*N^2)/step recompute over the ring buffer with an O(N^2)/step
    # incremental (init/add/evict/emit/finalize) reducer; when it does not (or
    # streaming is disabled), the recompute path is kept unchanged. Transparent to
    # the recipe: nothing here reads schema fields the recompute path does not.
    def _pipeline_reducer_ref(dobs_def):
        """(module, name, skip_t, s_var) of a derived obs's FIRST pipeline step, else None."""
        pipeline = getattr(dobs_def, 'pipeline', None)
        if not pipeline:
            return None
        step = pipeline[0]
        call = getattr(step, 'callable', None)
        if not call or not getattr(call, 'name', None) or not getattr(call, 'module', None):
            return None
        skip_t, s_var = 0, 0
        for aname, arg in (getattr(step, 'arguments', None) or {}).items():
            val = getattr(arg, 'value', None)
            if val is None:
                continue
            if str(aname) == 'skip_t':
                skip_t = int(val)
            elif str(aname) == 's_var':
                s_var = int(val)
        return (str(call.module), str(call.name), skip_t, s_var)

    streaming_map = {}  # derived_obs_name -> {src_obs, factory, skip_t, s_var}
    if use_sliding_window:
        for _dobs_name, _src_list in dependent_observations.items():
            _dobs_def = derived_observations_dict.get(_dobs_name)
            if not _dobs_def or not _src_list:
                continue
            _ref = _pipeline_reducer_ref(_dobs_def)
            if not _ref:
                continue
            _mod, _name, _skip_t, _s_var = _ref
            _spec = lookup_streaming_reducer(_STREAMING_BACKEND, _mod, _name)
            # Only step-wise ('window') reducers are wired here; 'stride' (FCD)
            # specs keep the recompute fallback until the stride branch lands.
            if _spec is not None and _spec.emit_kind == 'window' and _src_list[0] in source_observations_needed:
                streaming_map[_dobs_name] = dict(
                    src_obs=_src_list[0], factory=_spec.factory,
                    skip_t=_skip_t, s_var=_s_var,
                )
%>
def run_${algo_name}(
    state: Any,
    model_fn: Callable,
    key: jax.random.PRNGKey,
    n_iterations: int,
% for pname in hyperparam_dict.keys():
    ${pname}: float,
% endfor
% for inp_name in external_inputs:
    ${inp_name}: jnp.ndarray,
% endfor
    post_model_fn: Callable = None,
    post_state: Any = None,
    history: Any = None,
    run_post_tuning: bool = True,  # set False when called as a nested inner loop
% if use_sliding_window:
% for src_obs in source_observations_needed:
    ${src_obs}_buffer: jnp.ndarray = None,  # Optional: passed from previous algorithm
% endfor
% endif
    monitors: dict = None,
    verbose: bool = True,
    print_every: int = None,
    save_every: int = None,
% if use_sliding_window:
    resync_every: int = None,  # streaming-reducer float-drift re-sync period (default = window_size; 0 disables)
% endif
) -> Bunch:
    import copy
    import equinox as eqx

    def _smart_interval(n):
        """Compute smart interval: 1 for 0-10, 10 for 10-100, 100 for 100-1000, etc."""
        if n <= 10:
            return 1
        return 10 ** (len(str(n)) - 2)

    if print_every is None:
        print_every = _smart_interval(n_iterations)
    if save_every is None:
        save_every = _smart_interval(n_iterations)
    state = copy.deepcopy(state)

    # Update rule functions
% for rule_idx, (rule, rule_source, arg_overrides) in enumerate(all_update_rules_with_source):
<%
    rule_name = safe_name(rule.name)
    target_name = str(rule.target_parameter.name if hasattr(rule.target_parameter, 'name') else rule.target_parameter)
    rule_rhs = rule.equation.rhs
    is_from_included = rule_source != str(algo.name)
    source_algo = algorithms_dict.get(rule_source)
    source_hyperparam_dict = get_hyperparam_dict(source_algo) if source_algo else {}
    effective_hyperparams = {**source_hyperparam_dict, **arg_overrides}
    rule_params_dict = {}
    for pname, pval in effective_hyperparams.items():
        if pname in rule_rhs:
            rule_params_dict[pname] = pval
    eta_param_names = [p for p in rule_params_dict.keys() if p.lower() in ['eta', 'learning_rate', 'lr']]
    has_eta_param = len(eta_param_names) > 0
    eta_param_name = eta_param_names[0] if has_eta_param else None
    bounds = rule.bounds
    has_bounds = bounds is not None
    lo_bound = bounds.lo if has_bounds else None
    hi_bound = bounds.hi if has_bounds else None
    all_known_symbols = [target_name] + list(observations) + list(rule_params_dict.keys())
    rule_has_warmup = getattr(rule, 'warmup', False)
    needs_warmup = rule_has_warmup and has_eta_param
    if needs_warmup:
        all_known_symbols.append('eta_scale')
%>
    def ${rule_name}(${target_name}, ${', '.join(observations)}${', ' + ', '.join(rule_params_dict.keys()) if rule_params_dict else ''}${',' if needs_warmup else ''} ${'eta_scale=1.0' if needs_warmup else ''}):
% if needs_warmup:
        _eta = ${eta_param_name} * eta_scale
% for pn in rule_params_dict.keys():
% if pn == eta_param_name:
        ${pn} = _eta
% endif
% endfor
% endif
        updated = ${jaxcode(rule_rhs, all_known_symbols)}
% if has_bounds and (lo_bound is not None or hi_bound is not None):
        return jnp.clip(updated, ${lo_bound if lo_bound is not None else 'None'}, ${hi_bound if hi_bound is not None else 'None'})
% else:
        return updated
% endif
% endfor

<%
    algo_func_names = [get_func_name(fc) for fc in algo_functions]
%>
    result_history = Bunch(
% for obs in simulated_observations:
        ${obs}=[],
% endfor
% for func_name in algo_func_names:
        ${func_name}=[],
% endfor
% for rule, rule_source, arg_overrides in all_update_rules_with_source:
<%
    target_name = str(rule.target_parameter.name) if hasattr(rule.target_parameter, 'name') else str(rule.target_parameter)
%>
        ${target_name}=[],
% endfor
    )

<%
    # Identify pipeline observations that need raw sample collection (for passing to next algorithm)
    # Exclude source_observations_needed since those are handled by sliding window buffer
    collectible_observations = []
    for obs in simulated_observations:
        obs_def = observations_dict.get(obs)
        if obs_def and hasattr(obs_def, 'pipeline') and obs_def.pipeline:
            # Skip if this is a source observation (already has a sliding window buffer)
            if obs not in source_observations_needed:
                collectible_observations.append(obs)
%>
% for obs in collectible_observations:
    _${obs}_samples = []
% endfor

    pre_tuning = model_fn(state)

    # Create observation monitors for pipeline-based observations (created once, reused in loop)
<%
    # Determine which observations need monitors
    pipeline_observations = []
    for obs in simulated_observations:
        obs_def = observations_dict.get(obs)
        if obs_def and hasattr(obs_def, 'pipeline') and obs_def.pipeline:
            obs_class = ''.join(w.capitalize() for w in obs.replace('_', ' ').split())
            pipeline_observations.append((obs, obs_class))

    # For source observations that need sliding-window buffers, we need their
    # monitors too. EVERY source observation requires a monitor for the warmup
    # loop, whether it is pipeline-based (e.g. a derived FC) or a raw monitor
    # observation (e.g. `bold` from a BOLD monitor, which has no pipeline). The
    # monitor class is the CamelCase of the observation name and is emitted for
    # all observations, so this is gated only by membership in
    # source_observations_needed (already filtered to real observations), not by
    # the presence of a pipeline. (Previously the pipeline gate left raw-source
    # monitors like `_bold_monitor` undefined -> UnboundLocalError.)
    source_monitors = []
    for src_obs in source_observations_needed:
        src_class = ''.join(w.capitalize() for w in src_obs.replace('_', ' ').split())
        source_monitors.append((src_obs, src_class))

    # Note: Derived observations don't have monitor classes - they're computed from other observations
    # So we don't need dependent_monitors anymore
%>
    if monitors is None:
        monitors = {}
% for obs, obs_class in pipeline_observations:
    _${obs}_monitor = monitors.get('${obs}') if monitors.get('${obs}') is not None else ${obs_class}(history=history)
% endfor
% for src_obs, src_class in source_monitors:
% if src_obs not in [o[0] for o in pipeline_observations]:
    _${src_obs}_monitor = monitors.get('${src_obs}') if monitors.get('${src_obs}') is not None else ${src_class}(history=history)
% endif
% endfor
    history_accessor = lambda tree: tree._history

% if use_sliding_window:
    n_nodes = state.dynamics.${list(model.state_variables.keys())[0] if model.state_variables else 'S_e'}.shape[0] if hasattr(state.dynamics, '${list(model.state_variables.keys())[0] if model.state_variables else 'S_e'}') else history.data.shape[2] if history is not None else N_NODES

% for src_obs in source_observations_needed:
    if ${src_obs}_buffer is not None:
        _passed_buffer = ${src_obs}_buffer
        _passed_len = _passed_buffer.shape[0]

        if _passed_len >= int(window_size):
            if _passed_buffer.ndim == 2:
                _${src_obs}_buffer = _passed_buffer[-int(window_size):].reshape((int(window_size), 1, n_nodes))
            elif _passed_buffer.ndim == 3:
                _${src_obs}_buffer = _passed_buffer[-int(window_size):]
            else:
                _${src_obs}_buffer = _passed_buffer[-int(window_size):].reshape((int(window_size), 1, n_nodes))
            _buffer_idx = int(window_size)
            if verbose:
                print(f"  Using passed ${src_obs} buffer ({_passed_len} samples, using last {int(window_size)})")
        else:
            _${src_obs}_buffer = jnp.zeros((int(window_size), 1, n_nodes))
            if _passed_buffer.ndim == 2:
                for _pi in range(_passed_len):
                    _${src_obs}_buffer = _${src_obs}_buffer.at[int(window_size) - _passed_len + _pi, 0, :].set(_passed_buffer[_pi, :])
            elif _passed_buffer.ndim == 3:
                _${src_obs}_buffer = _${src_obs}_buffer.at[-_passed_len:, :, :].set(_passed_buffer)
            else:
                for _pi in range(_passed_len):
                    _${src_obs}_buffer = _${src_obs}_buffer.at[int(window_size) - _passed_len + _pi, 0, :].set(_passed_buffer[_pi])
            _buffer_idx = _passed_len
            if verbose:
                print(f"  Passed ${src_obs} buffer too small ({_passed_len} < {int(window_size)}), running warmup for remaining {int(window_size) - _passed_len} samples...")
            for _warmup_i in range(int(window_size) - _passed_len):
                key, subkey = jax.random.split(key)
                _warmup_result = model_fn(state)
                state.initial_state.dynamics = _warmup_result.data[-1][:${len(state_names)}]
                if getattr(state, 'noise', None) is not None and getattr(state.noise, 'key', None) is not None:
                    # Resample the in-scan noise by swapping the PRNG key — the
                    # live randomness source. _internal.noise_samples is an
                    # optional injection slot that defaults to None. Matches the
                    # reference EI_Tuning workflow (state.noise.key = subkey).
                    state.noise.key = subkey
                elif hasattr(state, '_internal') and getattr(state._internal, 'noise_samples', None) is not None:
                    state._internal.noise_samples = jax.random.normal(
                        key=subkey, shape=state._internal.noise_samples.shape
                    )
                _warmup_${src_obs} = _${src_obs}_monitor(_warmup_result)
                _warmup_${src_obs}_data = _warmup_${src_obs}.data if hasattr(_warmup_${src_obs}, 'data') else _warmup_${src_obs}
                _${src_obs}_buffer = jnp.roll(_${src_obs}_buffer, -1, axis=0)
                if _warmup_${src_obs}_data.ndim == 2:
                    _${src_obs}_buffer = _${src_obs}_buffer.at[-1, 0, :].set(_warmup_${src_obs}_data[0, :])
                elif _warmup_${src_obs}_data.ndim == 3:
                    _${src_obs}_buffer = _${src_obs}_buffer.at[-1, :, :].set(_warmup_${src_obs}_data[0, :, :])
                else:
                    _${src_obs}_buffer = _${src_obs}_buffer.at[-1, 0, :].set(_warmup_${src_obs}_data)
                if hasattr(_${src_obs}_monitor, '_history') and _${src_obs}_monitor._history is not None:
                    _new_history = jnp.roll(_${src_obs}_monitor._history, -_warmup_result.data.shape[0], axis=0)
                    _new_history = _new_history.at[-_warmup_result.data.shape[0]:, :, :].set(_warmup_result.data[:, 0:1, :])
                    _${src_obs}_monitor = eqx.tree_at(history_accessor, _${src_obs}_monitor, _new_history)
            _buffer_idx = int(window_size)
            if verbose:
                print(f"  Warmup complete. Buffer filled with {_buffer_idx} samples.")
    else:
        # No buffer passed - run warmup phase
        _${src_obs}_buffer = jnp.zeros((int(window_size), 1, n_nodes))
        _buffer_idx = 0

        if verbose:
            print(f"  Warmup: filling ${src_obs} buffer with {int(window_size)} samples...")

        for _warmup_i in range(int(window_size)):
            key, subkey = jax.random.split(key)
            _warmup_result = model_fn(state)
            state.initial_state.dynamics = _warmup_result.data[-1][:${len(state_names)}]
            if getattr(state, 'noise', None) is not None and getattr(state.noise, 'key', None) is not None:
                # Resample the in-scan noise by swapping the PRNG key (live
                # randomness source); _internal.noise_samples defaults to None.
                state.noise.key = subkey
            elif hasattr(state, '_internal') and getattr(state._internal, 'noise_samples', None) is not None:
                state._internal.noise_samples = jax.random.normal(
                    key=subkey, shape=state._internal.noise_samples.shape
                )

            _warmup_${src_obs} = _${src_obs}_monitor(_warmup_result)
            _warmup_${src_obs}_data = _warmup_${src_obs}.data if hasattr(_warmup_${src_obs}, 'data') else _warmup_${src_obs}
            _${src_obs}_buffer = jnp.roll(_${src_obs}_buffer, -1, axis=0)
            if _warmup_${src_obs}_data.ndim == 2:
                _${src_obs}_buffer = _${src_obs}_buffer.at[-1, 0, :].set(_warmup_${src_obs}_data[0, :])
            elif _warmup_${src_obs}_data.ndim == 3:
                _${src_obs}_buffer = _${src_obs}_buffer.at[-1, :, :].set(_warmup_${src_obs}_data[0, :, :])
            else:
                _${src_obs}_buffer = _${src_obs}_buffer.at[-1, 0, :].set(_warmup_${src_obs}_data)

            if hasattr(_${src_obs}_monitor, '_history') and _${src_obs}_monitor._history is not None:
                _new_history = jnp.roll(_${src_obs}_monitor._history, -_warmup_result.data.shape[0], axis=0)
                _new_history = _new_history.at[-_warmup_result.data.shape[0]:, :, :].set(_warmup_result.data[:, 0:1, :])
                _${src_obs}_monitor = eqx.tree_at(history_accessor, _${src_obs}_monitor, _new_history)

        _buffer_idx = int(window_size)
        if verbose:
            print(f"  Warmup complete. Buffer filled with {_buffer_idx} samples.")
% endfor
% endif

% for dobs_name, sinfo in streaming_map.items():
    # Streaming reducer for '${dobs_name}' over the '${sinfo['src_obs']}' window:
    # replaces the O(window*N^2)/step ${sinfo['factory'].rsplit('.', 1)[-1]}-equivalent
    # recompute with an O(N^2)/step incremental accumulator. The accumulator is
    # (re)built EXACTLY from the ring buffer here (post-warmup / post-passed-buffer)
    # and re-synced every _${dobs_name}_resync_every steps to reset float drift.
    _${dobs_name}_reducer = ${sinfo['factory']}(s_var=${sinfo['s_var']}, skip_t=${sinfo['skip_t']})
    _${dobs_name}_resync_every = int(window_size) if resync_every is None else int(resync_every)
    _${dobs_name}_acc = _${dobs_name}_reducer.resync(_${sinfo['src_obs']}_buffer)
% endfor

    if verbose:
        print(f"Running ${algo_name} algorithm for {n_iterations} iterations...")

    for i in range(n_iterations):
        key, subkey = jax.random.split(key)
% for ni in nested_includes:
        # [NESTED INNER LOOP] Re-converge '${ni['name']}' before the outer update.
        # The outer update rule's validity depends on '${ni['name']}'s invariant
        # (e.g. FIC holding the E-I working point), which the previous outer step
        # perturbed. Running its own run_${ni['name']}() to (near-)convergence each
        # outer iteration re-settles that invariant. run_post_tuning=False skips the
        # inner post-tuning simulation (we only need the converged state).
        key, subkey = jax.random.split(key)
        state = run_${ni['name']}(
            state, model_fn, subkey, ${ni['inner_iterations']},
% for hv in ni['hyperparam_values']:
            ${hv},
% endfor
% for ext in ni['external_inputs']:
            ${ext},
% endfor
            history=history, run_post_tuning=False, verbose=False,
        ).state
% endfor
        result = model_fn(state)
        state.initial_state.dynamics = result.data[-1][:${len(state_names)}]
        if getattr(state, 'noise', None) is not None and getattr(state.noise, 'key', None) is not None:
            # Resample the in-scan noise by swapping the PRNG key (live
            # randomness source); _internal.noise_samples defaults to None.
            state.noise.key = subkey
        elif hasattr(state, '_internal') and getattr(state._internal, 'noise_samples', None) is not None:
            state._internal.noise_samples = jax.random.normal(
                key=subkey, shape=state._internal.noise_samples.shape
            )

% if use_sliding_window:
% for src_obs in source_observations_needed:
<%
    src_obs_class = ''.join(w.capitalize() for w in src_obs.replace('_', ' ').split())
%>
        _${src_obs}_sample = _${src_obs}_monitor(result)
        _${src_obs}_sample_data = _${src_obs}_sample.data if hasattr(_${src_obs}_sample, 'data') else _${src_obs}_sample
<%
    _stream_for_src = [(d, si) for d, si in streaming_map.items() if si['src_obs'] == src_obs]
%>
% for dobs_name, sinfo in _stream_for_src:
        # Sample leaving the effective window (ring index skip_t, read PRE-roll so
        # it is the sample that falls out as the window slides forward).
        _${dobs_name}_evict = _${src_obs}_buffer[${sinfo['skip_t']}, ${sinfo['s_var']}, :]
% endfor
        _${src_obs}_buffer = jnp.roll(_${src_obs}_buffer, -1, axis=0)
        if _${src_obs}_sample_data.ndim == 2:
            _${src_obs}_buffer = _${src_obs}_buffer.at[-1, 0, :].set(_${src_obs}_sample_data[0, :])
        elif _${src_obs}_sample_data.ndim == 3:
            _${src_obs}_buffer = _${src_obs}_buffer.at[-1, :, :].set(_${src_obs}_sample_data[0, :, :])
        else:
            _${src_obs}_buffer = _${src_obs}_buffer.at[-1, 0, :].set(_${src_obs}_sample_data)
% for dobs_name, sinfo in _stream_for_src:
        # Incremental update: drop the leaving sample, fold in the arriving one
        # (read from the buffer's newest slot — the canonical storage). The
        # accumulator now holds EXACTLY _${src_obs}_buffer[skip_t:] (the window).
        _${dobs_name}_acc = _${dobs_name}_reducer.evict(_${dobs_name}_acc, _${dobs_name}_evict)
        _${dobs_name}_acc = _${dobs_name}_reducer.add(_${dobs_name}_acc, _${src_obs}_buffer[-1, ${sinfo['s_var']}, :])
% endfor
        # Maintain the monitor's own rolling history only if it carries one.
        # Stateless monitors (e.g. a BOLD monitor) have no _history buffer.
        if hasattr(_${src_obs}_monitor, '_history') and _${src_obs}_monitor._history is not None:
            _new_history = jnp.roll(_${src_obs}_monitor._history, -result.data.shape[0], axis=0)
            _new_history = _new_history.at[-result.data.shape[0]:, :, :].set(result.data[:, 0:1, :])
            _${src_obs}_monitor = eqx.tree_at(history_accessor, _${src_obs}_monitor, _new_history)
% endfor
        _buffer_idx = min(_buffer_idx + 1, int(window_size))
% for dobs_name, sinfo in streaming_map.items():
        # Periodic exact re-sync: rebuild the accumulator from the ring buffer to
        # reset the float drift the running sums accumulate (add/evict are not
        # exactly reversible in floating point). resync_every<=0 disables it.
        if _${dobs_name}_resync_every > 0 and (i + 1) % _${dobs_name}_resync_every == 0:
            _${dobs_name}_acc = _${dobs_name}_reducer.resync(_${sinfo['src_obs']}_buffer)
% endfor

% for obs in simulated_observations:
<%
    # Check both regular and derived observations
    obs_def = observations_dict.get(obs)
    derived_obs_def = derived_observations_dict.get(obs)

    # dependent_observations maps obs_name -> [source_obs_names] (list)
    src_obs_list_for_this = dependent_observations.get(obs)  # Returns list or None
    # For derived observations, use the first source observation for buffer computation
    src_obs_for_this = src_obs_list_for_this[0] if src_obs_list_for_this else None

    # Use derived_obs_def for derived observations
    effective_obs_def = derived_obs_def if derived_obs_def else obs_def
    has_pipeline = effective_obs_def and effective_obs_def.pipeline
    obs_source = obs_def.source if obs_def else None  # Only regular obs have source
    # `source` is multivalued; for raw observations there is exactly one
    # entry (a state-variable reference). Take it.
    if isinstance(obs_source, (list, tuple)):
        obs_source = obs_source[0] if obs_source else None
    if obs_source is not None and hasattr(obs_source, 'name'):
        obs_source = obs_source.name
    obs_aggregation = obs_def.aggregation if obs_def else None
    agg_str = str(obs_aggregation) if obs_aggregation else None

    if obs_def and obs_source and agg_str == 'mean' and not has_pipeline:
        # Source must be a recorded variable (state or auxiliary in VOI).
        if str(obs_source) in var_names:
            state_idx = var_names.index(str(obs_source))
        else:
            raise ValueError(f"Observation '{obs}' source '{obs_source}' not in recorded variables: {var_names}. Add it to model.output or to an observation source so the solver records it.")
    else:
        state_idx = None

    # For dependent observations, extract the primary pipeline function
    # to call directly on buffer (e.g., compute_fc for fc observation)
    direct_call = None
    direct_call_args = []  # Additional keyword args from pipeline
    if src_obs_for_this and has_pipeline:
        pipeline = effective_obs_def.pipeline
        if pipeline and len(pipeline) > 0:
            first_step = pipeline[0]
            step_callable = first_step.callable
            if step_callable:
                callable_name = step_callable.name
                callable_module = step_callable.module
                if callable_name and callable_module:
                    direct_call = f"{callable_module}.{callable_name}"
                    # Extract additional arguments from pipeline (e.g., skip_t=20)
                    if hasattr(first_step, 'arguments') and first_step.arguments:
                        for arg_name, arg in first_step.arguments.items():
                            arg_value = getattr(arg, 'value', None)
                            # Skip the source observation argument (that's passed as the buffer)
                            if arg_name and arg_value is not None:
                                val_str = str(arg_value)
                                # Skip observation references (those are handled by buffer)
                                if val_str == src_obs_for_this:
                                    continue
                                # Include numeric arguments as keyword args
                                if val_str.replace('.', '').replace('-', '').isdigit():
                                    direct_call_args.append(f"{arg_name}={val_str}")
                elif callable_name:
                    raise ValueError(f"Derived observation '{obs}' pipeline callable '{callable_name}' missing module - must specify full path")
                else:
                    raise ValueError(f"Derived observation '{obs}' pipeline has callable without name attribute")
            else:
                raise ValueError(f"Derived observation '{obs}' has source_observations but pipeline step lacks callable")
        else:
            raise ValueError(f"Derived observation '{obs}' has source_observations '{src_obs_for_this}' but no pipeline defined")

    # Build argument string (buffer is first positional, then keyword args from pipeline)
    direct_call_kwargs = ", ".join(direct_call_args) if direct_call_args else ""
%>
% if obs in streaming_map:
        # ${obs}: streaming windowed reducer over _${streaming_map[obs]['src_obs']}_buffer.
        # emit() == ${direct_call}(buffer${', ' + direct_call_kwargs if direct_call_kwargs else ''}) up to float rounding
        # (re-synced every _${obs}_resync_every steps) at O(N^2)/step instead of O(window*N^2)/step.
        ${obs} = _${obs}_reducer.emit(_${obs}_acc)
% elif src_obs_for_this:
        # ${obs} depends on ${src_obs_for_this} - compute directly from accumulated buffer
        # Call: ${direct_call}(buffer${', ' + direct_call_kwargs if direct_call_kwargs else ''})
        ${obs} = ${direct_call}(_${src_obs_for_this}_buffer${', ' + direct_call_kwargs if direct_call_kwargs else ''})
% elif obs_def and obs_source and state_idx is not None:
        ${obs} = jnp.mean(result.data[:, ${state_idx}], axis=0)  # Mean over time, per node
% elif has_pipeline and obs not in source_observations_needed:
        # Use observation monitor for pipeline (squeeze to remove state dimension)
        _${obs}_result = _${obs}_monitor(result)
        ${obs} = _${obs}_result.data if hasattr(_${obs}_result, 'data') else _${obs}_result
        ${obs} = jnp.squeeze(${obs})  # Remove extra dimensions for scalar/vector operations
% elif obs in source_observations_needed:
        # ${obs} is a source observation - use the sample computed above
        ${obs} = jnp.squeeze(_${obs}_sample_data)
% else:
        # ERROR: ${obs} has no valid computation path defined in YAML
        raise RuntimeError("Observation '${obs}' requires explicit pipeline or aggregation in YAML")
% endif
% endfor
% else:
% for obs in simulated_observations:
<%
    # Check if observable is defined in observations section
    obs_def = observations_dict.get(obs)
    has_pipeline = obs_def and obs_def.pipeline
    if obs_def:
        # Observable defined in observations - check source and aggregation
        obs_source = obs_def.source
        # `source` is multivalued; for raw observations there is one entry.
        if isinstance(obs_source, (list, tuple)):
            obs_source = obs_source[0] if obs_source else None
        if obs_source is not None and hasattr(obs_source, 'name'):
            obs_source = obs_source.name
        obs_aggregation = obs_def.aggregation
        # Convert enum to string for comparison
        agg_str = str(obs_aggregation) if obs_aggregation else None
        if obs_source and agg_str == 'mean' and not has_pipeline:
            # Simple aggregation case: source must be a recorded variable.
            if str(obs_source) in var_names:
                state_idx = var_names.index(str(obs_source))
            else:
                state_idx = 0
        else:
            state_idx = None
    else:
        state_idx = None
        obs_source = None
        agg_str = None
        has_pipeline = False

    # Generate Python class name for pipeline-based observations
    obs_class_name = ''.join(w.capitalize() for w in obs.replace('_', ' ').split())
%>
% if obs_def and obs_source and state_idx is not None:
        ${obs} = jnp.mean(result.data[:, ${state_idx}], axis=0)  # Mean over time, per node
% elif has_pipeline:
        # Use observation monitor for pipeline (squeeze to remove state dimension)
        _${obs}_result = _${obs}_monitor(result)
        ${obs} = _${obs}_result.data if hasattr(_${obs}_result, 'data') else _${obs}_result
        ${obs} = jnp.squeeze(${obs})  # Remove extra dimensions for scalar/vector operations

% if obs in collectible_observations:
        # Collect raw sample for passing to next algorithm (handle different result shapes)
        if hasattr(_${obs}_result, 'ys') and _${obs}_result.ys.ndim >= 3:
            _${obs}_samples.append(_${obs}_result.ys[0, 0, :])
        else:
            _${obs}_samples.append(${obs})
% endif

        # Update monitor history for hemodynamic state continuity (only if monitor has history)
        if hasattr(_${obs}_monitor, '_history') and _${obs}_monitor._history is not None:
            _new_history = jnp.roll(_${obs}_monitor._history, -result.data.shape[0], axis=0)
            _new_history = _new_history.at[-result.data.shape[0]:, :, :].set(result.data[:, 0:1, :])
            _${obs}_monitor = eqx.tree_at(history_accessor, _${obs}_monitor, _new_history)
% else:
        # ${obs} requires observation pipeline (not simple aggregation)
        pass
% endif
% endfor
% endif

% if has_warmup:
        # Compute learning rate warmup scale: (i+1) / n_iterations
        eta_scale = (i + 1) / n_iterations
% endif

        # Record history at save_every intervals
        if (i + 1) % save_every == 0 or i == 0:
            # Record simulated observations (mean value for scalars)
% for obs in simulated_observations:
            result_history.${obs}.append(jnp.mean(${obs}))
% endfor

            # Record parameter values (captures current values)
% for rule, rule_source, arg_overrides in all_update_rules_with_source:
<%
    target_name = str(rule.target_parameter.name) if hasattr(rule.target_parameter, 'name') else str(rule.target_parameter)
    rec_coupling_key = coupling_param_to_key.get(target_name, None)
    is_rec_coupling_param = rec_coupling_key is not None
%>
% if is_rec_coupling_param:
            result_history.${target_name}.append(state.coupling.${rec_coupling_key}.${target_name})
% else:
            result_history.${target_name}.append(state.dynamics.${target_name})
% endif
% endfor

            # Compute and record algorithm functions (metrics, derived quantities)
% for func_call in algo_functions:
<%
    func_name = get_func_name(func_call)
    arg_values = get_func_args(func_call)
    call_args = ', '.join([f"{k}={v}" for k, v in arg_values.items()])
    func_call_str = f"{func_name}({call_args})"
%>
            _${func_name}_val = ${func_call_str}
            result_history.${func_name}.append(_${func_name}_val)
% endfor

        # NOW compute and apply update rules (after recording)
% for rule_idx, (rule, rule_source, arg_overrides) in enumerate(all_update_rules_with_source):
<%
    rule_name = safe_name(rule.name)
    target_name = str(rule.target_parameter.name) if hasattr(rule.target_parameter, 'name') else str(rule.target_parameter)
    rule_rhs = rule.equation.rhs

    is_from_included = rule_source != str(algo.name)
    source_algo = algorithms_dict.get(rule_source)
    source_hyperparam_dict = get_hyperparam_dict(source_algo) if source_algo else {}
    effective_hyperparams = {**source_hyperparam_dict, **arg_overrides}

    rule_params_dict = {}
    for pname, pval in effective_hyperparams.items():
        if pname in rule_rhs:
            rule_params_dict[pname] = pval

    eta_param_names = [p for p in rule_params_dict.keys() if p.lower() in ['eta', 'learning_rate', 'lr']]
    has_eta_param = len(eta_param_names) > 0
    rule_has_warmup = getattr(rule, 'warmup', False)
    needs_warmup = rule_has_warmup and has_eta_param

    coupling_key = coupling_param_to_key.get(target_name, None)
    is_coupling_param = coupling_key is not None
%>
        new_${target_name} = ${rule_name}(
% if is_coupling_param:
            state.coupling.${coupling_key}.${target_name},
% else:
            state.dynamics.${target_name},
% endif
% for obs in observations:
            ${obs},
% endfor
% for pname, pval in rule_params_dict.items():
            ${pval},
% endfor
% if needs_warmup:
            eta_scale,
% endif
        )
% if is_coupling_param:
        state.coupling.${coupling_key}.${target_name} = new_${target_name}
% else:
        state.dynamics.${target_name} = new_${target_name}
% endif
% endfor

        if verbose and (i + 1) % print_every == 0:
<%
    # Build progress output - show functions (if any) + all simulated observations
    progress_items = []

    # First add any algorithm functions (e.g., fc_corr, fc_rmse)
    if algo_functions:
        for fc in algo_functions:
            fname = get_func_name(fc)
            if fname:
                progress_items.append(fname)

    # Then add all simulated observations
    for obs in simulated_observations:
        if obs not in progress_items:
            progress_items.append(obs)

    # Build format string
    progress_parts = [f"{item}={{float(result_history.{item}[-1]):.4f}}" for item in progress_items]
    progress_str = ", ".join(progress_parts)
%>
% if progress_items:
            print(f"  {i+1}/{n_iterations}: ${progress_str}")
% else:
            raise ValueError("Algorithm must have functions or simulated_observations for progress display")
% endif

    # Run post-tuning simulation (use full integration setup if provided).
    # Carry over the TUNED PARAMETERS (dynamics J_i, coupling wLRE/wFFI) but
    # keep post_state's own settled initial conditions — NOT the tuning loop's
    # last mid-trajectory state. Seeding from the loop's last state lands a
    # different attractor in multistable models, corrupting the post-tuning FC
    # (it made identical-weight post sims score far below the base sim).
    # Skipped entirely when run as a nested inner loop (only the state is needed).
    if run_post_tuning:
        if post_model_fn is not None and post_state is not None:
            import copy
            _post_state = copy.deepcopy(post_state)
            _post_state.dynamics = state.dynamics
            _post_state.coupling = state.coupling
            post_tuning = post_model_fn(_post_state)
        else:
            post_tuning = model_fn(state)

        # Compute ALL observations from post-tuning simulation (in dependency order)
        # This uses the experiment-level compute_all_observations function
        # Pass history as result_transient for BOLD pipeline continuity
        post_tuning_observations = compute_all_observations(post_tuning, state, history)
    else:
        post_tuning = None
        post_tuning_observations = None

    # Convert result_history lists to arrays
    for k in list(result_history.keys()):
        v = result_history[k]
        if len(v) > 0:
            first = v[0]
            if isinstance(first, (int, float)):
                result_history[k] = jnp.array(v)
            elif hasattr(first, 'shape'):
                if first.shape == ():
                    result_history[k] = jnp.array([float(x) for x in v])
                else:
                    result_history[k] = jnp.stack(v, axis=0)

    # Convert collected observation samples to arrays (for passing to next algorithm)
% for obs in collectible_observations:
    _${obs}_buffer_out = jnp.array(_${obs}_samples) if _${obs}_samples else None
% endfor

    # Collect monitors for passing to next algorithm (hemodynamic continuity)
    _monitors_out = {}
% for obs, obs_class in pipeline_observations:
    _monitors_out['${obs}'] = _${obs}_monitor
% endfor
% for src_obs, src_class in source_monitors:
% if src_obs not in [o[0] for o in pipeline_observations]:
    _monitors_out['${src_obs}'] = _${src_obs}_monitor
% endif
% endfor

    if verbose:
        print(f"${algo_name} complete!")

    # Build hyperparameters Bunch for AlgorithmResult
    _hyperparams = Bunch(
% for pname, pval in hyperparam_dict.items():
        ${pname}=${pval},
% endfor
    )

    return AlgorithmResult(
        name='${algo_name}',
        state=state,
        history=result_history,
        pre_tuning=pre_tuning,
        post_tuning=post_tuning,
        post_tuning_observations=post_tuning_observations,
        n_iterations=n_iterations,
        hyperparameters=_hyperparams,
        state_names=${state_names},
        # Additional fields for algorithm chaining
        monitors=_monitors_out,
% for obs in collectible_observations:
        ${obs}_buffer=_${obs}_buffer_out,
% endfor
% if use_sliding_window:
% for src_obs in source_observations_needed:
        ${src_obs}_buffer=_${src_obs}_buffer,
% endfor
% endif
    )

% endfor
% endif
