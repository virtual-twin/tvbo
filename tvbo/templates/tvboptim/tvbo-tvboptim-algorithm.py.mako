# -*- coding: utf-8 -*-
<%doc>TVB-Optim Algorithm Template. Context: experiment (SimulationExperiment).</%doc>
<%
from tvbo.codegen import render_expression
from tvbo.codegen.streaming_reducers import lookup_streaming_reducer
from tvbo.templates.tvboptim.utils import (
    safe_name, as_list, get_attr, is_network_observation, is_external_observation,
    get_include_info, get_all_observations_from_algo, get_all_hyperparams,
    streaming_post_eval_plan,
)

# Backend key this template targets — used for streaming-reducer registry lookups.
_STREAMING_BACKEND = 'tvboptim'

# Shared with the experiment template, so the two sides cannot drift: non-empty `names` means post_model_fn returns streamed values rather than a trajectory.
_pp = streaming_post_eval_plan(experiment)
_pp_names = _pp['names']
_pp_deliverables = _pp['deliverables']

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
from tvbo.utils import network_couplings
coupling_param_to_key = {}
for coupling_key, coupling_obj in network_couplings(experiment.network).items():
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

    # A registered streaming form replaces the O(window*N^2)/step recompute with an O(N^2)/step incremental reducer; without one the recompute path stands.
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

    streaming_map = {}  # derived_obs_name -> {src_obs, spec, skip_t, s_var}
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
            # Only step-wise 'window' reducers are wired here; a 'stride' spec keeps the recompute fallback.
            if _spec is not None and _spec.emit_kind == 'window' and _src_list[0] in source_observations_needed:
                streaming_map[_dobs_name] = dict(
                    src_obs=_src_list[0], spec=_spec,
                    skip_t=_skip_t, s_var=_s_var,
                )

    # Sizing the ring at the largest stage window and tracing window_size compiles the scan once across stages, where a varying window would recompile per stage.
    _stage_windows = []
    for _st in (getattr(algo, 'stages', None) or []):
        _ws_over = None
        for _arg in (getattr(_st, 'arguments', None) or []):
            if str(_arg.name) == 'window_size':
                _ws_over = _arg.value
        _stage_windows.append(int(_ws_over) if _ws_over is not None else (window_size_val if has_window_size else None))
    _stage_windows = [w for w in _stage_windows if w is not None]
    _varying_window = len(set(_stage_windows)) > 1
    _all_specs_maskable = bool(streaming_map) and all(
        getattr(_si['spec'], 'resync_masked', ()) for _si in streaming_map.values())
    use_maxwin = use_sliding_window and _varying_window and _all_specs_maskable
    _accept_maxwin = use_sliding_window and _varying_window  # accept the kwarg whenever the varying-window caller passes it; the M-ring is only emitted when also maskable (use_maxwin), else it is ignored and the contiguous path runs
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
    raw: bool = False,  # vmap-safe: skip pre_tuning sim + AlgorithmResult wrapping, return a Bunch of raw JAX arrays (for jax.vmap over a subject cohort)
% if use_sliding_window:
% for src_obs in source_observations_needed:
    ${src_obs}_buffer: jnp.ndarray = None,  # Optional: passed from previous algorithm
% endfor
% endif
% if _accept_maxwin:
    max_window_size: int = None,  # M: fixed ring size (>= every stage's window_size). Set -> masked ring drives the sliding window with a TRACED window_size so the tuning scan compiles ONCE across stages; None (or a non-maskable reducer) -> contiguous per-stage path.
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

    _raw_model_fn = model_fn  # un-jitted; the tuning core takes it as a STATIC arg (stable identity across stages -> jit caches the scan once)
    model_fn = jax.jit(model_fn)  # tvboptim's solve fn is un-jitted by design (its tests jit it); jit once so warmup/tuning calls fuse+cache instead of eager per-step dispatch
    post_model_fn = None if post_model_fn is None else jax.jit(post_model_fn)  # same: the post-tuning evaluation is the longest solve in a fit, and it arrives un-jitted from prepare()

    def _smart_interval(n):
        """Compute smart interval: 1 for 0-10, 10 for 10-100, 100 for 100-1000, etc."""
        if n <= 10:
            return 1
        return 10 ** (len(str(n)) - 2)

    if print_every is None:
        print_every = _smart_interval(n_iterations)
    if save_every is None:
        save_every = _smart_interval(n_iterations)
    state = jax.tree_util.tree_map(lambda _leaf: _leaf, state)  # fresh container; trace-safe (deepcopy of a jax typed key asserts under trace)
    _algo_t0 = time.perf_counter()

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

    pre_tuning = None if raw else model_fn(state)  # raw skips the diagnostic pre-sim; the host-side numpy wrap of a traced trajectory is not vmap-safe

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
% if use_maxwin:
    _M = int(max_window_size) if max_window_size is not None else int(window_size)  # physical ring size (== window_size on the contiguous path)
% endif

% for src_obs in source_observations_needed:
    # Warmup fill as an inner lax.scan (functional; vmap-safe; == the host for-loop).
    def _warmup_step_${src_obs}(_wc, _):
        state, key, _buf, _mon = _wc
        key, subkey = jax.random.split(key)
        _wr = model_fn(state)
        state = eqx.tree_at(lambda s: s.initial_state.dynamics, state, _wr.data[-1][:${len(state_names)}])
        if getattr(state, 'noise', None) is not None and getattr(state.noise, 'key', None) is not None:
            state = eqx.tree_at(lambda s: s.noise.key, state, subkey)
        elif hasattr(state, '_internal') and getattr(state._internal, 'noise_samples', None) is not None:
            state = eqx.tree_at(lambda s: s._internal.noise_samples, state,
                                jax.random.normal(key=subkey, shape=state._internal.noise_samples.shape))
        _wd = _mon(_wr)
        _wd = _wd.data if hasattr(_wd, 'data') else _wd
        _buf = jnp.roll(_buf, -1, axis=0)
        if _wd.ndim == 2:
            _buf = _buf.at[-1, 0, :].set(_wd[0, :])
        elif _wd.ndim == 3:
            _buf = _buf.at[-1, :, :].set(_wd[0, :, :])
        else:
            _buf = _buf.at[-1, 0, :].set(_wd)
        if hasattr(_mon, '_history') and _mon._history is not None:
            _nh = jnp.roll(_mon._history, -_wr.data.shape[0], axis=0)
            _nh = _nh.at[-_wr.data.shape[0]:, :, :].set(_wr.data[:, 0:1, :])
            _mon = eqx.tree_at(history_accessor, _mon, _nh)
        return (state, key, _buf, _mon), None

    def _run_warmup_${src_obs}(state, key, _buf, _mon, _nsteps):
        if _nsteps <= 0:
            return state, key, _buf, _mon
        (state, key, _buf, _mon), _ = jax.lax.scan(
            _warmup_step_${src_obs}, (state, key, _buf, _mon), None, length=_nsteps)
        return state, key, _buf, _mon

% if use_maxwin:
    # The last window_size slots hold what the contiguous path holds, so the trajectory is identical; leading slots sit outside the window and stay zero.
    _${src_obs}_buffer = jnp.zeros((_M, 1, n_nodes))
    if ${src_obs}_buffer is not None:
        _passed_buffer = ${src_obs}_buffer
        _passed_len = _passed_buffer.shape[0]
        _take = min(int(_passed_len), int(window_size))
        if _passed_buffer.ndim == 3:
            _${src_obs}_buffer = _${src_obs}_buffer.at[-_take:, :, :].set(_passed_buffer[-_take:])
        elif _passed_buffer.ndim == 2:
            _${src_obs}_buffer = _${src_obs}_buffer.at[-_take:, 0, :].set(_passed_buffer[-_take:, :])
        else:
            _${src_obs}_buffer = _${src_obs}_buffer.at[-_take:, 0, :].set(_passed_buffer[-_take:])
        _remaining = int(window_size) - _take
        if verbose:
            logger.info(f"  Carried last {_take} ${src_obs} samples into ring (M={_M}); warmup remaining {_remaining}...")
        state, key, _${src_obs}_buffer, _${src_obs}_monitor = _run_warmup_${src_obs}(
            state, key, _${src_obs}_buffer, _${src_obs}_monitor, _remaining)
    else:
        if verbose:
            logger.info(f"  Warmup: filling last {int(window_size)} of ${src_obs} ring (M={_M})...")
        state, key, _${src_obs}_buffer, _${src_obs}_monitor = _run_warmup_${src_obs}(
            state, key, _${src_obs}_buffer, _${src_obs}_monitor, int(window_size))
% else:
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
                logger.info(f"  Using passed ${src_obs} buffer ({_passed_len} samples, using last {int(window_size)})")
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
                logger.info(f"  Passed ${src_obs} buffer too small ({_passed_len} < {int(window_size)}), running warmup for remaining {int(window_size) - _passed_len} samples...")
            state, key, _${src_obs}_buffer, _${src_obs}_monitor = _run_warmup_${src_obs}(
                state, key, _${src_obs}_buffer, _${src_obs}_monitor, int(window_size) - _passed_len)
            _buffer_idx = int(window_size)
            if verbose:
                logger.info(f"  Warmup complete. Buffer filled with {_buffer_idx} samples.")
    else:
        # No buffer passed - run warmup phase
        _${src_obs}_buffer = jnp.zeros((int(window_size), 1, n_nodes))
        _buffer_idx = 0

        if verbose:
            logger.info(f"  Warmup: filling ${src_obs} buffer with {int(window_size)} samples...")

        state, key, _${src_obs}_buffer, _${src_obs}_monitor = _run_warmup_${src_obs}(
            state, key, _${src_obs}_buffer, _${src_obs}_monitor, int(window_size))
        _buffer_idx = int(window_size)
        if verbose:
            logger.info(f"  Warmup complete. Buffer filled with {_buffer_idx} samples.")
% endif
% endfor
% endif

    # Routed through the module-level jitted core, so a multi-stage schedule reuses one compiled scan.
    _rec_positions = [_ri for _ri in range(n_iterations) if (_ri + 1) % save_every == 0 or _ri == 0]
    _rec_idx = jnp.asarray(_rec_positions)  # record positions for the post-scan result_history rebuild
% if streaming_map:
    _resync_period = jnp.asarray(int(window_size) if resync_every is None else int(resync_every), jnp.int32)
% endif
% if use_maxwin:
    _ws0 = jnp.asarray(int(window_size), jnp.int32)
    _use_ring = max_window_size is not None
% endif
    def _canon(_x):
        # Stripping weak_type lets a fresh first stage and the scan's own outputs share one jit specialization.
        _a = _x if hasattr(_x, "dtype") else jnp.asarray(_x)
        if jax.dtypes.issubdtype(_a.dtype, jax.dtypes.prng_key):
            return _a
        return jax.lax.convert_element_type(_a, _a.dtype)
    _canon_tree = lambda _t: jax.tree_util.tree_map(_canon, _t)
    _ls_final, _ys_all = _${algo_name}_tuning_core(
        _canon_tree(state),
        _canon_tree(key),
% for src_obs in source_observations_needed:
        _canon_tree(_${src_obs}_buffer),
% endfor
% for obs, obs_class in pipeline_observations:
        _canon_tree(_${obs}_monitor),
% endfor
% for src_obs, src_class in source_monitors:
% if src_obs not in [o[0] for o in pipeline_observations]:
        _canon_tree(_${src_obs}_monitor),
% endif
% endfor
% for pname in hyperparam_dict.keys():
% if pname != 'window_size':
        _canon_tree(${pname}),
% endif
% endfor
% for inp_name in external_inputs:
        _canon_tree(${inp_name}),
% endfor
% if streaming_map:
        _canon_tree(_resync_period),
% endif
% if use_maxwin:
        _canon_tree(_ws0),
% endif
        model_fn=_raw_model_fn,
        n_iterations=n_iterations,
        print_every=print_every,
        save_every=save_every,
        verbose=verbose,
% if use_maxwin:
        use_ring=_use_ring,
% endif
    )
    state = _ls_final['state']
    # Tuning ends here; the scan dispatches asynchronously, so its output must land before the split from the full-duration eval below is honest.
    if verbose:
        jax.block_until_ready(state)
    _tune_t1 = time.perf_counter()
% for src_obs in source_observations_needed:
% if use_maxwin:
    # Return the M-ring's last window_size rows, so a next-stage call carries what the contiguous path does.
    _${src_obs}_buffer = _ls_final['${src_obs}__buf'][-int(window_size):] if max_window_size is not None else _ls_final['${src_obs}__buf']
% else:
    _${src_obs}_buffer = _ls_final['${src_obs}__buf']
% endif
% endfor
% for obs, obs_class in pipeline_observations:
    _${obs}_monitor = _ls_final['${obs}__mon']
% endfor
% for src_obs, src_class in source_monitors:
% if src_obs not in [o[0] for o in pipeline_observations]:
    _${src_obs}_monitor = _ls_final['${src_obs}__mon']
% endif
% endfor
    # Rebuild result_history: scalars subsampled at record positions; param snapshots from the carried buffers.
    result_history = Bunch(
% for _k, obs in enumerate(simulated_observations):
        ${obs}=_ys_all[${_k}][_rec_idx],
% endfor
<% _nso = len(simulated_observations) %>
% for _j, func_call in enumerate(algo_functions):
        ${get_func_name(func_call)}=_ys_all[${_nso + _j}][_rec_idx],
% endfor
% for rule, rule_source, arg_overrides in all_update_rules_with_source:
<% _tn = str(rule.target_parameter.name) if hasattr(rule.target_parameter, 'name') else str(rule.target_parameter) %>
        ${_tn}=_ls_final['${_tn}__rec'],
% endfor
    )

    # Carries the tuned parameters but keeps post_state's settled initial conditions: the loop's last mid-trajectory state lands a different attractor in a multistable model.
    if run_post_tuning:
% if _pp_names:
        # ${', '.join(_pp_names)} fold into the integrator carry, so ${', '.join(_pp_deliverables)} come from the streamed value alone and no fit-scale trajectory is materialised.
        if post_model_fn is not None and post_state is not None:
            import copy
            _post_state = copy.deepcopy(post_state)
            _post_state.dynamics = state.dynamics
            _post_state.coupling = state.coupling
            _streamed = post_model_fn(_post_state)
            _stream_vals = {_n: _v for _n, _v in zip(
                ${repr(_pp_names)}, _streamed if isinstance(_streamed, (tuple, list)) else (_streamed,))}
            post_tuning = None
            post_tuning_observations = compute_all_observations(
                None, state, history,
                only=${repr(set(_pp_names) | set(_pp_deliverables))}, precomputed=_stream_vals,
% if external_inputs:
                network_obs={${', '.join("'%s': %s" % (n, n) for n in external_inputs)}},
% endif
            )
        else:
            post_tuning = None
            post_tuning_observations = None
% else:
        if post_model_fn is not None and post_state is not None:
            import copy
            _post_state = copy.deepcopy(post_state)
            _post_state.dynamics = state.dynamics
            _post_state.coupling = state.coupling
            post_tuning = post_model_fn(_post_state)
        else:
            post_tuning = model_fn(state)

        # History is passed as result_transient so the BOLD pipeline continues across the boundary.
% if external_inputs:
        # Scores against this call's `${', '.join(external_inputs)}`, since the module-level default would score a per-subject run against the wrong target.
        post_tuning_observations = compute_all_observations(
            post_tuning, state, history,
            network_obs={${', '.join("'%s': %s" % (n, n) for n in external_inputs)}},
        )
% else:
        post_tuning_observations = compute_all_observations(post_tuning, state, history)
% endif
% endif
    else:
        post_tuning = None
        post_tuning_observations = None

    # result_history is already stacked arrays (scan ys + carried buffers); no list->array rebuild here keeps run_${algo_name} vmap-traceable for subject-batching.

    # Convert collected observation samples to arrays (for passing to next algorithm)
<%
    _nso_c = len(simulated_observations)
    _nf_c = len(algo_functions)
%>
% for _ci, obs in enumerate(collectible_observations):
    # Collectible samples were emitted as scan ys (index after obs-means + functions).
    _${obs}_buffer_out = _ys_all[${_nso_c + _nf_c + _ci}]
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
        logger.info(f"${algo_name} complete! (tuning {_tune_t1 - _algo_t0:.1f}s, {n_iterations} iters; post-tuning eval {time.perf_counter() - _tune_t1:.1f}s)")

    if raw:
        # vmap-safe cohort return: pure jnp arrays only; no AlgorithmResult/DataArray wrapping (host-side numpy conversion breaks under jax.vmap). Wrap per-subject host-side after the vmap.
        return Bunch(
            state=state,
            history=result_history,
            post_tuning_observations=post_tuning_observations,
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

    # An update rule with no restoring force can drive estimates non-finite, and returning them would write a NaN result as a "successful" run.
% if all_update_rules_with_source:
    _nonfinite_estimates = [
        _nm for _nm, _arr in (
% for rule, rule_source, arg_overrides in all_update_rules_with_source:
<%
    _tn = str(rule.target_parameter.name) if hasattr(rule.target_parameter, 'name') else str(rule.target_parameter)
    _gk = coupling_param_to_key.get(_tn, None)
    _gsrc = ('state.coupling.%s.%s' % (_gk, _tn)) if _gk is not None else ('state.dynamics.%s' % _tn)
%>
            ('${_tn}', ${_gsrc}),
% endfor
        )
        if not bool(jnp.all(jnp.isfinite(jnp.asarray(_arr))))
    ]
    if _nonfinite_estimates:
        raise RuntimeError(
            "${algo_name} tuning diverged: non-finite estimate(s) "
            f"{_nonfinite_estimates}. The update rule has no restoring force, so "
            "online every-TR updates over a long/hot schedule let parameters run "
            "away — reduce eta or n_iterations (or batch weight updates). Failing "
            "loud so a NaN fit is not recorded as a successful result."
        )
% endif

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


<%
    _core_statics = ["model_fn", "n_iterations", "print_every", "save_every", "verbose"]
    if use_maxwin:
        _core_statics.append("use_ring")
    _core_static_tuple = ", ".join("'%s'" % _s for _s in _core_statics)
%>
def _${algo_name}_tuning_core_impl(
    state,
    key,
% for src_obs in source_observations_needed:
    _${src_obs}_buffer,
% endfor
% for obs, obs_class in pipeline_observations:
    _${obs}_monitor,
% endfor
% for src_obs, src_class in source_monitors:
% if src_obs not in [o[0] for o in pipeline_observations]:
    _${src_obs}_monitor,
% endif
% endfor
% for pname in hyperparam_dict.keys():
% if pname != 'window_size':
    ${pname},
% endif
% endfor
% for inp_name in external_inputs:
    ${inp_name},
% endfor
% if streaming_map:
    _resync_period,
% endif
% if use_maxwin:
    ws0,
% endif
    model_fn,
    n_iterations,
    print_every,
    save_every,
    verbose,
% if use_maxwin:
    use_ring,
% endif
):
    """Compile-once tuning core: the tuning `lax.scan` under a STABLE module-level
    `jax.jit` so a multi-stage schedule (varying eta / window / resync) reuses ONE
    compiled scan across stages instead of recompiling per stage. Per-stage-varying
    scalars enter TRACED (eta, resync period, ring window ws0); model_fn is a STATIC
    arg with stable identity so the jit cache keys the same across stages."""
    import equinox as eqx
    history_accessor = lambda tree: tree._history
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
% if streaming_map:
<%
    from tvbo.codegen.reducers import resolve_streaming_reducer
    # One recipe for every wired reducer: the scaffolding below emits the resolved state for any spec, with no use case baked in.
    _r = resolve_streaming_reducer(next(iter(streaming_map.values()))['spec'], 'jax')
    _acc = ', '.join(_r['state'])
%>
    # The scaffolding is backend-shaped and the accumulator math is the resolved spec: resync rebuilds from the window, add and evict fold one sample, emit reads out.
    from types import SimpleNamespace as _StreamReducer
    def _make_windowed_reducer(s_var=0, skip_t=0):
        def _samp(x):
            return x if x.ndim == 1 else x[s_var, :]
        def add(acc, x_new):
            ${_acc} = acc
            v = _samp(x_new)
% for _lhs, _rhs in _r['add']:
            ${_lhs} = ${_rhs}
% endfor
            return (${_acc})
        def evict(acc, x_old):
            ${_acc} = acc
            v = _samp(x_old)
% for _lhs, _rhs in _r['evict']:
            ${_lhs} = ${_rhs}
% endfor
            return (${_acc})
        def emit(acc):
            ${_acc} = acc
            return ${_r['emit']}
        def resync(buffer):
            b = buffer[skip_t:]
            x = b[:, s_var, :] if b.ndim == 3 else b
% for _lhs, _rhs in _r['resync']:
            ${_lhs} = ${_rhs}
% endfor
            return (${_acc})
% if use_maxwin:
        def resync_masked(buffer, ws):
            # Equals resync(buffer[-ws:]) with `ws` traced, so the scan compiles once across stage window sizes.
            _MM = buffer.shape[0]
            x = buffer[:, s_var, :] if buffer.ndim == 3 else buffer
            L = jnp.asarray(ws, jnp.int32) - skip_t
            m = (jnp.arange(_MM) >= (_MM - L)).astype(x.dtype).reshape((_MM, 1))
% for _lhs, _rhs in _r['resync_masked']:
            ${_lhs} = ${_rhs}
% endfor
            return (${_acc})
        return _StreamReducer(add=add, evict=evict, emit=emit, resync=resync, resync_masked=resync_masked)
% else:
        return _StreamReducer(add=add, evict=evict, emit=emit, resync=resync)
% endif
% endif
% for dobs_name, sinfo in streaming_map.items():
    # Built exactly from the ring buffer and re-synced every _${dobs_name}_resync_every steps to reset float drift.
    _${dobs_name}_reducer = _make_windowed_reducer(s_var=${sinfo['s_var']}, skip_t=${sinfo['skip_t']})
    _${dobs_name}_resync_every = _resync_period  # traced period -> tuning scan compiles once across stage window sizes
% if use_maxwin:
    _${dobs_name}_acc = (
        _${dobs_name}_reducer.resync_masked(_${sinfo['src_obs']}_buffer, ws0)
        if use_ring
        else _${dobs_name}_reducer.resync(_${sinfo['src_obs']}_buffer))
% else:
    _${dobs_name}_acc = _${dobs_name}_reducer.resync(_${sinfo['src_obs']}_buffer)
% endif
% endfor

    if verbose:
        logger.info(f"Running ${algo_name} algorithm for {n_iterations} iterations...")

% if nested_includes:
    raise NotImplementedError(
        "scan-based tuning does not yet support nested-mode algorithm includes "
        "(scan-in-scan needs an inner pure core); use mode: combined (the shipped path)."
    )
% endif
    # Functional lax.scan tuning loop, byte-identical to the host for-loop (mutations -> eqx.tree_at; record positions == host `(i+1)%save_every==0 or i==0`).
    _rec_positions = [_ri for _ri in range(n_iterations) if (_ri + 1) % save_every == 0 or _ri == 0]
    _rec_idx = jnp.asarray(_rec_positions)
    _n_records = len(_rec_positions)
% for rule, rule_source, arg_overrides in all_update_rules_with_source:
<%
    _tn = str(rule.target_parameter.name) if hasattr(rule.target_parameter, 'name') else str(rule.target_parameter)
    _ck = coupling_param_to_key.get(_tn, None)
    _src = ('state.coupling.%s.%s' % (_ck, _tn)) if _ck is not None else ('state.dynamics.%s' % _tn)
%>
    _rec_${_tn}_buf0 = jnp.zeros((_n_records,) + tuple(jnp.shape(${_src})), dtype=jnp.asarray(${_src}).dtype)
% endfor

    def _tuning_step(_ls, _i):
        state = _ls['state']
        key = _ls['key']
% for src_obs in source_observations_needed:
        _${src_obs}_buffer = _ls['${src_obs}__buf']
% endfor
% for dobs_name in streaming_map.keys():
        _${dobs_name}_acc = _ls['${dobs_name}__acc']
% endfor
% for obs, obs_class in pipeline_observations:
        _${obs}_monitor = _ls['${obs}__mon']
% endfor
% for src_obs, src_class in source_monitors:
% if src_obs not in [o[0] for o in pipeline_observations]:
        _${src_obs}_monitor = _ls['${src_obs}__mon']
% endif
% endfor
% for rule, rule_source, arg_overrides in all_update_rules_with_source:
<% _tn = str(rule.target_parameter.name) if hasattr(rule.target_parameter, 'name') else str(rule.target_parameter) %>
        _rec_${_tn}_buf = _ls['${_tn}__rec']
% endfor
        _wptr = _ls['wptr']
% if use_maxwin:
        _ws = _ls['ws']  # traced window length (masked ring); unused on the contiguous (use_ring=False) path
% endif
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
        state = eqx.tree_at(lambda s: s.initial_state.dynamics, state, result.data[-1][:${len(state_names)}])
        if getattr(state, 'noise', None) is not None and getattr(state.noise, 'key', None) is not None:
            state = eqx.tree_at(lambda s: s.noise.key, state, subkey)
        elif hasattr(state, '_internal') and getattr(state._internal, 'noise_samples', None) is not None:
            state = eqx.tree_at(lambda s: s._internal.noise_samples, state,
                                jax.random.normal(key=subkey, shape=state._internal.noise_samples.shape))

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
        # Read pre-roll, as the sample falls out when the window slides.
% if use_maxwin:
        if use_ring:
            _${dobs_name}_evict = _${src_obs}_buffer[_${src_obs}_buffer.shape[0] - (_ws - ${sinfo['skip_t']}), ${sinfo['s_var']}, :]
        else:
            _${dobs_name}_evict = _${src_obs}_buffer[${sinfo['skip_t']}, ${sinfo['s_var']}, :]
% else:
        _${dobs_name}_evict = _${src_obs}_buffer[${sinfo['skip_t']}, ${sinfo['s_var']}, :]
% endif
% endfor
        _${src_obs}_buffer = jnp.roll(_${src_obs}_buffer, -1, axis=0)
        if _${src_obs}_sample_data.ndim == 2:
            _${src_obs}_buffer = _${src_obs}_buffer.at[-1, 0, :].set(_${src_obs}_sample_data[0, :])
        elif _${src_obs}_sample_data.ndim == 3:
            _${src_obs}_buffer = _${src_obs}_buffer.at[-1, :, :].set(_${src_obs}_sample_data[0, :, :])
        else:
            _${src_obs}_buffer = _${src_obs}_buffer.at[-1, 0, :].set(_${src_obs}_sample_data)
% for dobs_name, sinfo in _stream_for_src:
        # Drop the leaving sample and fold in the arriving one, leaving the accumulator holding exactly the window.
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
% for dobs_name, sinfo in streaming_map.items():
        # Periodic exact re-sync (float-drift reset; add/evict not exactly reversible); resync_every<=0 disables it.
        _${dobs_name}_acc = jax.lax.cond(
            (_${dobs_name}_resync_every > 0) & ((_i + 1) % jnp.maximum(_${dobs_name}_resync_every, 1) == 0),
% if use_maxwin:
            (lambda _a: _${dobs_name}_reducer.resync_masked(_${sinfo['src_obs']}_buffer, _ws))
            if use_ring
            else (lambda _a: _${dobs_name}_reducer.resync(_${sinfo['src_obs']}_buffer)),
% else:
            lambda _a: _${dobs_name}_reducer.resync(_${sinfo['src_obs']}_buffer),
% endif
            lambda _a: _a, _${dobs_name}_acc)
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
        # emit() equals ${direct_call}(buffer) up to float rounding, at O(N^2)/step rather than O(window*N^2)/step.
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
        # Collect raw sample every step for chaining -> scan ys (a host-list append would leak a tracer out of the scan).
        if hasattr(_${obs}_result, 'ys') and _${obs}_result.ys.ndim >= 3:
            _${obs}_collect = _${obs}_result.ys[0, 0, :]
        else:
            _${obs}_collect = ${obs}
% endif

        # Update monitor history for hemodynamic state continuity (only if monitor has history)
        if hasattr(_${obs}_monitor, '_history') and _${obs}_monitor._history is not None:
            _new_history = jnp.roll(_${obs}_monitor._history, -result.data.shape[0], axis=0)
            _new_history = _new_history.at[-result.data.shape[0]:, :, :].set(result.data[:, 0:1, :])
            _${obs}_monitor = eqx.tree_at(history_accessor, _${obs}_monitor, _new_history)
% else:
<%
    _why = ("is not declared in the experiment's observations" if obs_def is None
            else "is declared but is neither a mean-aggregation of a recorded variable nor a pipeline")
    raise ValueError(
        f"Algorithm observation '{obs}' {_why}. Algorithm observations must resolve to either "
        f"`source: [<recorded variable>], aggregation: mean` or a `pipeline:` in the experiment's "
        f"`observations:` block; otherwise the tuning loop references an undefined value. "
        f"Recorded variables available: {var_names}."
    )
%>\
% endif
% endfor
% endif

% if has_warmup:
        eta_scale = (_i + 1) / n_iterations
% endif

        # Sparse recording: param snapshots -> carried buffers (lax.cond @ _do_rec); obs/function scalars -> scan ys, subsampled after. Byte-identical to the host `.append`.
% for func_call in algo_functions:
<%
    func_name = get_func_name(func_call)
    arg_values = get_func_args(func_call)
    call_args = ', '.join([f"{k}={v}" for k, v in arg_values.items()])
    func_call_str = f"{func_name}({call_args})"
%>
        _${func_name}_val = ${func_call_str}
% endfor
        _do_rec = ((_i + 1) % save_every == 0) | (_i == 0)
        def _rec_write(_carry_rec):
            _bufs, _p = _carry_rec
            _bufs = (
% for _ri2, (rule, rule_source, arg_overrides) in enumerate(all_update_rules_with_source):
<%
    target_name = str(rule.target_parameter.name) if hasattr(rule.target_parameter, 'name') else str(rule.target_parameter)
    rec_coupling_key = coupling_param_to_key.get(target_name, None)
    _rsrc = ('state.coupling.%s.%s' % (rec_coupling_key, target_name)) if rec_coupling_key is not None else ('state.dynamics.%s' % target_name)
%>
                jax.lax.dynamic_update_index_in_dim(_bufs[${_ri2}], ${_rsrc}, _p, 0),
% endfor
            )
            return _bufs, _p + 1
        (
% for rule, rule_source, arg_overrides in all_update_rules_with_source:
<% target_name = str(rule.target_parameter.name) if hasattr(rule.target_parameter, 'name') else str(rule.target_parameter) %>
            _rec_${target_name}_buf,
% endfor
        ), _wptr = jax.lax.cond(
            _do_rec, _rec_write, lambda _c: _c,
            ((
% for rule, rule_source, arg_overrides in all_update_rules_with_source:
<% target_name = str(rule.target_parameter.name) if hasattr(rule.target_parameter, 'name') else str(rule.target_parameter) %>
                _rec_${target_name}_buf,
% endfor
            ), _wptr))

        # NOW compute and apply update rules (after recording)
% if 'update_every' in hyperparam_dict:
        # Batched-update cadence (cf. EIBalance.update_every); ue==1 -> always-True predicate -> byte-identical online path.
        _apply_update = ((_i + 1) % jnp.maximum(jnp.asarray(update_every, jnp.int32), 1)) == 0
% endif
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
            ${pname},
% endfor
% if needs_warmup:
            eta_scale,
% endif
        )
% if 'update_every' in hyperparam_dict:
        # Hold the parameter on non-cadence steps (batched-update gate).
        new_${target_name} = jnp.where(
            _apply_update,
            new_${target_name},
% if is_coupling_param:
            state.coupling.${coupling_key}.${target_name},
% else:
            state.dynamics.${target_name},
% endif
        )
% endif
% if is_coupling_param:
        state = eqx.tree_at(lambda s: s.coupling.${coupling_key}.${target_name}, state, new_${target_name})
% else:
        state = eqx.tree_at(lambda s: s.dynamics.${target_name}, state, new_${target_name})
% endif
% endfor

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
        # ── Progress logging: scan-safe host callback, fires only at print_every ──
% if progress_items:
<%
    _prog_names = list(progress_items)
    _sim_obs_names = [str(o) for o in simulated_observations]
    _prog_vals = ['jnp.mean(%s)' % _pn if _pn in _sim_obs_names else '_%s_val' % _pn for _pn in _prog_names]
%>
        if verbose:  # trace-time gate: batched/vmap runs (verbose=False) emit no host callback
            def _log_cb(_ii, *_vals):
                if logger.isEnabledFor(logging.INFO):
                    _pn = ${repr(_prog_names)}
                    logger.info("  %d/%d: %s" % (int(_ii) + 1, n_iterations,
                        ", ".join("%s=%.4f" % (_pn[_z], float(_vals[_z])) for _z in range(len(_pn)))))
            def _do_log():
                jax.debug.callback(_log_cb, _i, ${', '.join(_prog_vals)})
                return 0
            jax.lax.cond((_i + 1) % print_every == 0, _do_log, lambda: 0)
% endif
        # Per-step ys: obs-mean/function scalars (subsampled after) + collectible raw samples (kept in full for chaining).
        _ys = (
% for obs in simulated_observations:
            jnp.mean(${obs}),
% endfor
% for func_call in algo_functions:
            _${get_func_name(func_call)}_val,
% endfor
% for cobs in collectible_observations:
            _${cobs}_collect,
% endfor
        )
        _ls_out = {
            'state': state, 'key': key,
% for src_obs in source_observations_needed:
            '${src_obs}__buf': _${src_obs}_buffer,
% endfor
% for dobs_name in streaming_map.keys():
            '${dobs_name}__acc': _${dobs_name}_acc,
% endfor
% for obs, obs_class in pipeline_observations:
            '${obs}__mon': _${obs}_monitor,
% endfor
% for src_obs, src_class in source_monitors:
% if src_obs not in [o[0] for o in pipeline_observations]:
            '${src_obs}__mon': _${src_obs}_monitor,
% endif
% endfor
% for rule, rule_source, arg_overrides in all_update_rules_with_source:
<% _tn = str(rule.target_parameter.name) if hasattr(rule.target_parameter, 'name') else str(rule.target_parameter) %>
            '${_tn}__rec': _rec_${_tn}_buf,
% endfor
            'wptr': _wptr,
% if use_maxwin:
            'ws': _ws,
% endif
        }
        return _ls_out, _ys

    _ls_init = {
        'state': state, 'key': key,
% for src_obs in source_observations_needed:
        '${src_obs}__buf': _${src_obs}_buffer,
% endfor
% for dobs_name in streaming_map.keys():
        '${dobs_name}__acc': _${dobs_name}_acc,
% endfor
% for obs, obs_class in pipeline_observations:
        '${obs}__mon': _${obs}_monitor,
% endfor
% for src_obs, src_class in source_monitors:
% if src_obs not in [o[0] for o in pipeline_observations]:
        '${src_obs}__mon': _${src_obs}_monitor,
% endif
% endfor
% for rule, rule_source, arg_overrides in all_update_rules_with_source:
<% _tn = str(rule.target_parameter.name) if hasattr(rule.target_parameter, 'name') else str(rule.target_parameter) %>
        '${_tn}__rec': _rec_${_tn}_buf0,
% endfor
        'wptr': jnp.asarray(0),
% if use_maxwin:
        'ws': ws0,
% endif
    }
    _ls_final, _ys_all = jax.lax.scan(_tuning_step, _ls_init, jnp.arange(n_iterations))
    return _ls_final, _ys_all


_${algo_name}_tuning_core = jax.jit(
    _${algo_name}_tuning_core_impl, static_argnames=(${_core_static_tuple}))


<%
    # The shared network stays in the model closure and the per-subject targets are the vmap axis.
    _ds_targets = set(getattr(experiment, 'dataset_observation_targets', None) or {})
    batched_inputs = [i for i in external_inputs if i in _ds_targets]
    shared_inputs = [i for i in external_inputs if i not in _ds_targets]
    emit_cohort = bool(getattr(experiment, 'dataset_on_device', lambda: False)()) and len(batched_inputs) > 0
    # One schedule either way: Algorithm.stages, or a single synthetic stage from the algo defaults.
    _c_stages = list(getattr(algo, 'stages', None) or [])
    c_stage_defs = []
    for _st in _c_stages:
        _over = {}
        for _arg in (getattr(_st, 'arguments', None) or []):
            _over[str(_arg.name)] = _arg.value
        _sd = {'n_iterations': int(_st.n_iterations)}
        for _hp_name, _hp_val in hyperparam_dict.items():
            _sd[_hp_name] = _over.get(_hp_name, _hp_val)
        c_stage_defs.append(_sd)
    if not c_stage_defs:
        c_stage_defs = [{'n_iterations': n_iterations, **dict(hyperparam_dict)}]
%>
% if emit_cohort:
def run_cohort_${algo_name}(
    algo_state,
    model_fn,
    key,
% for _bi in batched_inputs:
    ${_bi},
% endfor
% for _si in shared_inputs:
    ${_si}=None,
% endfor
    save_every: int = None,
    resync_every: int = None,
    batch_size: int = None,
):
    """On-device cohort: vectorise the full ${algo_name} fit over subjects.

    ${', '.join(batched_inputs)} carries a leading subject axis (n_subjects, ...); the
    shared network lives in model_fn's closure, so only the per-subject target(s) + RNG
    vary per lane. Every fit runs pure-jnp via run_${algo_name}(raw=True) — no host
    wrapping inside the vmap. Returns the batched tuned state (leading subject axis);
    wrap/save per subject on the host after this returns.

    ``batch_size`` bounds how many subjects vectorise together: the whole cohort in
    one ``vmap`` by default, or a Python loop of ``batch_size``-wide ``vmap`` slices
    concatenated on the subject axis when a smaller width is requested (or the memory
    budget resolves one). Each slice is a plain ``vmap`` run on its own, so only one
    slice's working memory is live at a time. A narrower slice changes XLA's fusion
    order, so per-subject results match the whole-cohort ``vmap`` to floating-point
    rounding (~1 ULP), not bit-for-bit."""
    _n_subjects = ${batched_inputs[0]}.shape[0]
    _keys = jax.random.split(key, _n_subjects)
    _stage_defs = [
% for sd in c_stage_defs:
        ${repr(sd)},
% endfor
    ]
    def _fit_one_subject(${', '.join('_lane_%d' % i for i in range(len(batched_inputs)))}, _skey):
        _st = algo_state
        _mon = None
% if use_sliding_window:
% for src_obs in source_observations_needed:
        _buf_${src_obs} = None
% endfor
% endif
        for _si, _sd in enumerate(_stage_defs):
            _r = run_${algo_name}(
                _st, model_fn, jax.random.fold_in(_skey, _si), _sd['n_iterations'],
% for hp in hyperparam_dict.keys():
                ${hp}=_sd['${hp}'],
% endfor
% for _i, _bi in enumerate(batched_inputs):
                ${_bi}=_lane_${_i},
% endfor
% for _si2 in shared_inputs:
                ${_si2}=${_si2},
% endfor
% if use_sliding_window:
% for src_obs in source_observations_needed:
                ${src_obs}_buffer=_buf_${src_obs},
% endfor
                resync_every=resync_every,
% endif
                monitors=_mon, raw=True, run_post_tuning=False, verbose=False,
                save_every=save_every,
            )
            _st = _r.state
% if use_sliding_window:
% for src_obs in source_observations_needed:
            _buf_${src_obs} = _r.get('${src_obs}_buffer', _buf_${src_obs})
% endfor
% endif
            _mon = _r.get('monitors', _mon)
        return _st
    _example = (${', '.join('%s[0]' % bi for bi in batched_inputs)}, _keys[0])
    _bs = resolve_cohort_batch_size(batch_size, _n_subjects, _fit_one_subject, _example)
    if _bs >= _n_subjects:
        return jax.vmap(_fit_one_subject)(${', '.join(batched_inputs)}, _keys)
    _chunks = [
        jax.vmap(_fit_one_subject)(${', '.join('%s[_i:_i + _bs]' % bi for bi in batched_inputs)}, _keys[_i:_i + _bs])
        for _i in range(0, _n_subjects, _bs)
    ]
    return jax.tree_util.tree_map(lambda *_c: jnp.concatenate(_c, axis=0), *_chunks)

% endif
% endfor
% endif
