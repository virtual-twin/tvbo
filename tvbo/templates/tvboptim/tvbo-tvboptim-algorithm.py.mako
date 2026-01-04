# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Algorithm Template
============================

Generates iterative parameter tuning algorithms from Algorithm definitions.

IMPORTANT: This template is FULLY GENERIC. ALL values come from YAML metadata.
No hardcoded defaults, no special cases for specific algorithms.

Context: experiment (SimulationExperiment instance)
</%doc>
<%
from tvbo.export.code import render_expression

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

# =============================================================================
# Extract Algorithms from experiment
# =============================================================================
# algorithms can be dict (keyed by name) or list - get values if dict
_algos_raw = experiment.algorithms or {}
algorithms_list = list(_algos_raw.values()) if hasattr(_algos_raw, 'values') else list(_algos_raw)
has_algorithms = len(algorithms_list) > 0

# Build algorithms_dict for looking up included algorithms
algorithms_dict = {}
for _algo in algorithms_list:
    algorithms_dict[str(_algo.name)] = _algo

def safe_name(name):
    """Convert name to valid Python identifier."""
    return str(name).replace(' ', '_').replace('-', '_').lower()

def as_list(obj):
    """Convert dict or list to list of values."""
    if obj is None:
        return []
    if hasattr(obj, 'values'):
        return list(obj.values())
    return list(obj)

def get_func_name(func_call):
    """Get function name from FunctionCall."""
    return str(func_call.function) if func_call.function else None

def get_func_args(func_call):
    """Get arguments from FunctionCall as dict {name: value}."""
    if not func_call.arguments:
        return {}
    return {str(arg.name): arg.value for arg in func_call.arguments}

def get_target_name(rule):
    """Get target parameter name from UpdateRule."""
    tp = rule.target_parameter
    return str(tp.name) if hasattr(tp, 'name') else str(tp)

def get_include_info(inc):
    """Extract algorithm name and argument overrides from AlgorithmInclude.
    
    Returns (algo_name, {param_name: value}) tuple.
    """
    # Handle AlgorithmInclude object
    if hasattr(inc, 'algorithm'):
        algo_name = str(inc.algorithm.name) if hasattr(inc.algorithm, 'name') else str(inc.algorithm)
        args = {}
        for arg in as_list(inc.arguments):
            args[str(arg.name)] = arg.value
        return algo_name, args
    # Fallback for simple string reference
    return str(inc), {}

def get_obs_names(algo):
    """Get observation names as strings."""
    return [str(o) for o in as_list(algo.observations)]

def get_all_observations(algo, algorithms_dict):
    """Get all observation names including from included algorithms.
    
    Preserves order: included algorithm observations first, then this algorithm's.
    """
    obs = []
    seen = set()
    # First from included algorithms
    for inc in as_list(algo.includes):
        inc_name, _ = get_include_info(inc)
        inc_algo = algorithms_dict.get(inc_name)
        if inc_algo:
            for o in get_obs_names(inc_algo):
                if o not in seen:
                    obs.append(o)
                    seen.add(o)
    # Then this algorithm's observations
    for o in get_obs_names(algo):
        if o not in seen:
            obs.append(o)
            seen.add(o)
    return obs

def get_all_update_rules(algo, algorithms_dict):
    """Get all update rules including from included algorithms.
    
    Returns list of (rule, source_algo_name, arg_overrides) tuples.
    arg_overrides is a dict of {param_name: value} for hyperparameter overrides.
    Included algorithm rules come first, then this algorithm's rules.
    """
    all_rules = []
    # First, add rules from included algorithms with their argument overrides
    for inc in as_list(algo.includes):
        inc_name, arg_overrides = get_include_info(inc)
        inc_algo = algorithms_dict.get(inc_name)
        if inc_algo:
            for rule in as_list(inc_algo.update_rules):
                all_rules.append((rule, inc_name, arg_overrides))
    # Then add this algorithm's own rules (no overrides needed)
    for rule in as_list(algo.update_rules):
        all_rules.append((rule, str(algo.name), {}))
    return all_rules

def get_external_inputs(algo, obs_dict, algorithms_dict=None):
    """Get observations that have external data_source."""
    obs_names = get_all_observations(algo, algorithms_dict or {}) if algorithms_dict else get_obs_names(algo)
    return [o for o in obs_names if obs_dict.get(o) and obs_dict[o].data_source]

def get_simulated_observations(algo, obs_dict, algorithms_dict=None):
    """Get observations that are simulated (no data_source)."""
    obs_names = get_all_observations(algo, algorithms_dict or {}) if algorithms_dict else get_obs_names(algo)
    return [o for o in obs_names if not (obs_dict.get(o) and obs_dict[o].data_source)]

def get_hyperparam_dict(algo):
    """Build {name: value} dict from hyperparameters (THIS algorithm only)."""
    return {str(hp.name): hp.value for hp in as_list(algo.hyperparameters)}

def get_all_hyperparams(algo, algorithms_dict):
    """Get all hyperparameters including from included algorithms.
    
    Returns dict {name: value}. Included algorithm hyperparameters come first,
    then this algorithm's hyperparameters override. Argument overrides from
    includes are applied.
    """
    all_hp = {}
    # First, add hyperparameters from included algorithms (with argument overrides)
    for inc in as_list(algo.includes):
        inc_name, arg_overrides = get_include_info(inc)
        inc_algo = algorithms_dict.get(inc_name)
        if inc_algo:
            # Get base hyperparameters from included algorithm
            for hp in as_list(inc_algo.hyperparameters):
                hp_name = str(hp.name)
                # Use override if present, else use original value
                if hp_name in arg_overrides:
                    all_hp[hp_name] = arg_overrides[hp_name]
                else:
                    all_hp[hp_name] = hp.value
    # Then add this algorithm's own hyperparameters (override included)
    for hp in as_list(algo.hyperparameters):
        all_hp[str(hp.name)] = hp.value
    return all_hp

def get_all_functions(algo, algorithms_dict):
    """Get all functions including from included algorithms.
    
    Returns list of FunctionCall objects. Included algorithm functions come first.
    """
    all_funcs = []
    # First, add functions from included algorithms
    for inc in as_list(algo.includes):
        inc_name, _ = get_include_info(inc)
        inc_algo = algorithms_dict.get(inc_name)
        if inc_algo:
            all_funcs.extend(as_list(inc_algo.functions))
    # Then add this algorithm's own functions
    all_funcs.extend(as_list(algo.functions))
    return all_funcs

# Extract observations dict for reference
_obs_raw = experiment.observations or {}
observations_dict = dict(_obs_raw.items()) if hasattr(_obs_raw, 'items') else {}

# State variable names from model
model = experiment.local_dynamics
state_var_names = list(model.state_variables.keys()) if model and model.state_variables else []

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

# =============================================================================
# ITERATIVE ALGORITHMS
# =============================================================================

% for algo in algorithms_list:
<%
    algo_name = safe_name(algo.name)
    n_iterations = int(algo.n_iterations)
    simulation_period = float(algo.simulation_period)

    # Get all update rules (including from included algorithms)
    all_update_rules_with_source = get_all_update_rules(algo, algorithms_dict)
    update_rules = [r for r, src, args in all_update_rules_with_source]  # Just the rules

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
# -----------------------------------------------------------------------------
# Algorithm: ${algo_name}
# -----------------------------------------------------------------------------

% for rule_idx, (rule, rule_source, arg_overrides) in enumerate(all_update_rules_with_source):
<%
    rule_name = safe_name(rule.name)
    target_name = str(rule.target_parameter.name if hasattr(rule.target_parameter, 'name') else rule.target_parameter)
    rule_rhs = rule.equation.rhs

    # Check if this rule comes from an included algorithm
    is_from_included = rule_source != str(algo.name)

    # Get hyperparameters from the source algorithm
    source_algo = algorithms_dict.get(rule_source)
    source_hyperparam_dict = get_hyperparam_dict(source_algo) if source_algo else {}

    # Apply argument overrides from includes (these take priority)
    effective_hyperparams = {**source_hyperparam_dict, **arg_overrides}

    # Parse equation RHS to extract parameter names (symbols that aren't observations or target)
    # We'll just use any hyperparameters that exist in effective_hyperparams
    rule_params_dict = {}
    for pname, pval in effective_hyperparams.items():
        # Only include parameters that appear in the equation RHS
        if pname in rule_rhs:
            rule_params_dict[pname] = pval

    # Find learning rate parameter
    eta_param_names = [p for p in rule_params_dict.keys() if p.lower() in ['eta', 'learning_rate', 'lr']]
    has_eta_param = len(eta_param_names) > 0
    eta_param_name = eta_param_names[0] if has_eta_param else None

    # Bounds
    bounds = rule.bounds
    has_bounds = bounds is not None
    lo_bound = bounds.lo if has_bounds else None
    hi_bound = bounds.hi if has_bounds else None

    # All known symbols - add eta_scale if warmup applies
    all_known_symbols = [target_name] + list(observations) + list(rule_params_dict.keys())

    # Check if warmup should apply to THIS rule
    # Warmup applies if: rule has warmup=True AND this rule has an eta parameter
    # Note: we use per-rule warmup, not algorithm-level, for fine-grained control
    rule_has_warmup = getattr(rule, 'warmup', False)
    needs_warmup = rule_has_warmup and has_eta_param
    if needs_warmup:
        all_known_symbols.append('eta_scale')
%>
def update_${algo_name}_${rule_name}(
    ${target_name}: jnp.ndarray,
% for obs in observations:
    ${obs}: jnp.ndarray,
% endfor
% for pname in rule_params_dict.keys():
    ${pname}: float,
% endfor
% if needs_warmup:
    eta_scale: float = 1.0,
% endif
) -> jnp.ndarray:
    """Update rule: ${rule_name} for ${target_name}${' (with learning rate warmup)' if needs_warmup else ''}."""
% if needs_warmup:
    # Apply learning rate warmup: scale eta by eta_scale (i+1)/n_iterations
    ${eta_param_name} = ${eta_param_name} * eta_scale
% endif
    updated = ${jaxcode(rule_rhs, all_known_symbols)}
% if has_bounds and (lo_bound is not None or hi_bound is not None):
    updated = jnp.clip(updated, ${lo_bound if lo_bound is not None else 'None'}, ${hi_bound if hi_bound is not None else 'None'})
% endif
    return updated

% endfor

<%
    # Detect sliding window pattern:
    # 1. Check for window_size hyperparameter
    # 2. Find observations with source_observation (depend on another observation)
    has_window_size = 'window_size' in hyperparam_dict
    window_size_val = int(hyperparam_dict.get('window_size', 150)) if has_window_size else 0

    # Find observations that have source_observation (dependent observations)
    # These need their source accumulated in a buffer
    dependent_observations = {}  # {obs_name: source_obs_name}
    source_observations_needed = set()  # Source observations that need buffers

    for obs in simulated_observations:
        obs_def = observations_dict.get(obs)
        if obs_def:
            src_obs = obs_def.source_observation
            if src_obs:
                dependent_observations[obs] = str(src_obs)
                source_observations_needed.add(str(src_obs))

    # Use sliding window if we have window_size AND dependent observations
    use_sliding_window = has_window_size and len(dependent_observations) > 0
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
    history: Any = None,
% if use_sliding_window:
% for src_obs in source_observations_needed:
    ${src_obs}_buffer: jnp.ndarray = None,  # Optional: passed from previous algorithm
% endfor
% endif
    verbose: bool = True,
    print_every: int = None,
) -> Bunch:
    """
    Run ${algo_name} algorithm for n_iterations.
% if description:

    ${description}
% endif

    Args:
        state: Simulation state (Bunch with dynamics, coupling, etc.)
        model_fn: Compiled model function from run_simulation
        key: Random key for noise generation (REQUIRED)
        n_iterations: Number of algorithm iterations
% for pname in hyperparam_dict.keys():
        ${pname}: Algorithm hyperparameter (from YAML)
% endfor
% for inp_name in external_inputs:
        ${inp_name}: External data (from observations section with data_source)
% endfor
% if use_sliding_window:
% for src_obs in source_observations_needed:
        ${src_obs}_buffer: Optional buffer from previous algorithm (skips warmup if provided)
% endfor
% endif
        verbose: Print progress messages
        print_every: Print frequency (defaults to n_iterations // 10)

    Returns:
        Bunch with: state, history, pre_tuning, post_tuning, ${', '.join([f'{obs}_buffer' for obs in source_observations_needed]) if use_sliding_window else ''}
    """
    import copy
    import equinox as eqx

    if print_every is None:
        print_every = max(1, n_iterations // 10)

    # Deep copy state to avoid modifying original
    state = copy.deepcopy(state)

<%
    # Extract function names from FunctionCall objects
    algo_func_names = [get_func_name(fc) for fc in algo_functions]
%>
    # History tracking for algorithm results
    result_history = Bunch(
% for obs in simulated_observations:
        ${obs}=[],
% endfor
% for func_name in algo_func_names:
        ${func_name}=[],  # Tracked from algorithm function
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
    # Buffer lists for collecting raw observation samples (passed to next algorithm)
% for obs in collectible_observations:
    _${obs}_samples = []  # Will hold raw samples for each iteration
% endfor

    # Run pre-tuning simulation for comparison
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

    # For source observations that need buffers, we need their monitors too
    source_monitors = []
    for src_obs in source_observations_needed:
        src_def = observations_dict.get(src_obs)
        if src_def and hasattr(src_def, 'pipeline') and src_def.pipeline:
            src_class = ''.join(w.capitalize() for w in src_obs.replace('_', ' ').split())
            source_monitors.append((src_obs, src_class))

    # For dependent observations (those with source_observation), we also need monitors
    dependent_monitors = []
    for obs, src_obs in dependent_observations.items():
        obs_def = observations_dict.get(obs)
        if obs_def and hasattr(obs_def, 'pipeline') and obs_def.pipeline:
            obs_class = ''.join(w.capitalize() for w in obs.replace('_', ' ').split())
            if (obs, obs_class) not in pipeline_observations:
                dependent_monitors.append((obs, obs_class))
%>
% for obs, obs_class in pipeline_observations:
    _${obs}_monitor = ${obs_class}(history=history)
% endfor
% for src_obs, src_class in source_monitors:
% if src_obs not in [o[0] for o in pipeline_observations]:
    _${src_obs}_monitor = ${src_class}(history=history)
% endif
% endfor
% for obs, obs_class in dependent_monitors:
    _${obs}_monitor = ${obs_class}(history=history)
% endfor
    # History accessor for updating monitor state (hemodynamic continuity)
    # Uses _history because generated Bold class stores history with underscore prefix
    history_accessor = lambda tree: tree._history

% if use_sliding_window:
    # Initialize sliding window buffers for source observations
    # Shape: (window_size, 1, n_nodes) - accumulated over iterations
    n_nodes = state.dynamics.${list(model.state_variables.keys())[0] if model.state_variables else 'S_e'}.shape[0] if hasattr(state.dynamics, '${list(model.state_variables.keys())[0] if model.state_variables else 'S_e'}') else history.data.shape[2] if history is not None else 68

% for src_obs in source_observations_needed:
    # Check if buffer was passed from previous algorithm
    if ${src_obs}_buffer is not None:
        # Use buffer directly from previous algorithm (e.g., from FIC)
        # Reshape to match expected shape (window_size, 1, n_nodes)
        if ${src_obs}_buffer.ndim == 2:
            _${src_obs}_buffer = ${src_obs}_buffer[-int(window_size):].reshape((int(window_size), 1, n_nodes))
        elif ${src_obs}_buffer.ndim == 3:
            _${src_obs}_buffer = ${src_obs}_buffer[-int(window_size):]
        else:
            _${src_obs}_buffer = ${src_obs}_buffer[-int(window_size):].reshape((int(window_size), 1, n_nodes))
        _buffer_idx = int(window_size)  # Buffer is already full
        if verbose:
            print(f"  Using passed ${src_obs} buffer ({_${src_obs}_buffer.shape[0]} samples)")
    else:
        # No buffer passed - run warmup phase
        _${src_obs}_buffer = jnp.zeros((int(window_size), 1, n_nodes))
        _buffer_idx = 0

        if verbose:
            print(f"  Warmup: filling ${src_obs} buffer with {int(window_size)} samples...")

        for _warmup_i in range(int(window_size)):
            key, subkey = jax.random.split(key)

            # Run simulation for one period
            _warmup_result = model_fn(state)

            # Update state for next iteration
            state.initial_state.dynamics = _warmup_result.data[-1]
            if hasattr(state, '_internal') and hasattr(state._internal, 'noise_samples'):
                state._internal.noise_samples = jax.random.normal(
                    key=subkey, shape=state._internal.noise_samples.shape
                )

            # Compute ${src_obs} and add to buffer
            _warmup_${src_obs} = _${src_obs}_monitor(_warmup_result)
            _warmup_${src_obs}_data = _warmup_${src_obs}.data if hasattr(_warmup_${src_obs}, 'data') else _warmup_${src_obs}

            # Roll buffer and add new sample
            _${src_obs}_buffer = jnp.roll(_${src_obs}_buffer, -1, axis=0)
            if _warmup_${src_obs}_data.ndim == 2:
                _${src_obs}_buffer = _${src_obs}_buffer.at[-1, 0, :].set(_warmup_${src_obs}_data[0, :])
            elif _warmup_${src_obs}_data.ndim == 3:
                _${src_obs}_buffer = _${src_obs}_buffer.at[-1, :, :].set(_warmup_${src_obs}_data[0, :, :])
            else:
                _${src_obs}_buffer = _${src_obs}_buffer.at[-1, 0, :].set(_warmup_${src_obs}_data)

            # === CRITICAL: Update ${src_obs.upper()} monitor history for hemodynamic state continuity ===
            # This matches the original: new_history = jnp.roll(...); bold_monitor = eqx.tree_at(...)
            _new_history = jnp.roll(_${src_obs}_monitor._history, -_warmup_result.data.shape[0], axis=0)
            _new_history = _new_history.at[-_warmup_result.data.shape[0]:, :, :].set(_warmup_result.data[:, 0:1, :])
            _${src_obs}_monitor = eqx.tree_at(history_accessor, _${src_obs}_monitor, _new_history)

        _buffer_idx = int(window_size)  # Buffer is now full
        if verbose:
            print(f"  Warmup complete. Buffer filled with {_buffer_idx} samples.")
% endfor
% endif

    if verbose:
        print(f"Running ${algo_name} algorithm for {n_iterations} iterations...")

    for i in range(n_iterations):
        # Split key for this iteration
        key, subkey = jax.random.split(key)

        # Run simulation for one period
        result = model_fn(state)

        # Update state for next iteration
        state.initial_state.dynamics = result.data[-1]
        if hasattr(state, '_internal') and hasattr(state._internal, 'noise_samples'):
            state._internal.noise_samples = jax.random.normal(
                key=subkey, shape=state._internal.noise_samples.shape
            )

        # Compute observations from simulation result
        # Simulated observations are computed here; external inputs are passed as arguments
% if use_sliding_window:
        # === SLIDING WINDOW PATTERN ===
        # 1. Compute source observations and add to buffer
% for src_obs in source_observations_needed:
<%
    src_obs_class = ''.join(w.capitalize() for w in src_obs.replace('_', ' ').split())
%>
        _${src_obs}_sample = _${src_obs}_monitor(result)
        _${src_obs}_sample_data = _${src_obs}_sample.data if hasattr(_${src_obs}_sample, 'data') else _${src_obs}_sample
        # Roll buffer and add new sample at end (shape: [1, 1, n_nodes] or [1, n_nodes])
        _${src_obs}_buffer = jnp.roll(_${src_obs}_buffer, -1, axis=0)
        if _${src_obs}_sample_data.ndim == 2:
            # Shape [1, n_nodes] -> expand to [1, 1, n_nodes]
            _${src_obs}_buffer = _${src_obs}_buffer.at[-1, 0, :].set(_${src_obs}_sample_data[0, :])
        elif _${src_obs}_sample_data.ndim == 3:
            # Shape [1, 1, n_nodes]
            _${src_obs}_buffer = _${src_obs}_buffer.at[-1, :, :].set(_${src_obs}_sample_data[0, :, :])
        else:
            # Shape [n_nodes]
            _${src_obs}_buffer = _${src_obs}_buffer.at[-1, 0, :].set(_${src_obs}_sample_data)

        # Update ${src_obs} monitor history for hemodynamic state continuity
        _new_history = jnp.roll(_${src_obs}_monitor._history, -result.data.shape[0], axis=0)
        _new_history = _new_history.at[-result.data.shape[0]:, :, :].set(result.data[:, 0:1, :])
        _${src_obs}_monitor = eqx.tree_at(history_accessor, _${src_obs}_monitor, _new_history)
% endfor
        _buffer_idx = min(_buffer_idx + 1, int(window_size))

        # 2. Compute dependent observations from buffer (only after warmup)
% for obs in simulated_observations:
<%
    obs_def = observations_dict.get(obs)
    src_obs_for_this = dependent_observations.get(obs)
    has_pipeline = obs_def and obs_def.pipeline
    obs_source = obs_def.source if obs_def else None
    obs_aggregation = obs_def.aggregation if obs_def else None
    agg_str = str(obs_aggregation) if obs_aggregation else None

    if obs_def and obs_source and agg_str == 'mean' and not has_pipeline:
        if str(obs_source) in state_var_names:
            state_idx = state_var_names.index(str(obs_source))
        else:
            raise ValueError(f"Observation '{obs}' source '{obs_source}' not found in state variables: {state_var_names}")
    else:
        state_idx = None

    # For dependent observations, extract the primary pipeline function
    # to call directly on buffer (e.g., compute_fc for fc observation)
    direct_call = None
    if src_obs_for_this and has_pipeline:
        pipeline = obs_def.pipeline
        if pipeline and len(pipeline) > 0:
            first_step = pipeline[0]
            step_callable = first_step.callable
            if step_callable:
                callable_name = step_callable.name
                callable_module = step_callable.module
                if callable_name and callable_module:
                    direct_call = f"{callable_module}.{callable_name}"
                elif callable_name:
                    raise ValueError(f"Observation '{obs}' pipeline callable '{callable_name}' missing module - must specify full path")
                else:
                    raise ValueError(f"Observation '{obs}' pipeline has callable without name attribute")
            else:
                raise ValueError(f"Observation '{obs}' has source_observation but pipeline step lacks callable")
        else:
            raise ValueError(f"Observation '{obs}' has source_observation '{src_obs_for_this}' but no pipeline defined")
%>
% if src_obs_for_this:
        # ${obs} depends on ${src_obs_for_this} - compute directly from accumulated buffer
        # Call pipeline function directly: ${direct_call}
        ${obs} = ${direct_call}(_${src_obs_for_this}_buffer)
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
        # === STANDARD PATTERN (no sliding window) ===
% for obs in simulated_observations:
<%
    # Check if observable is defined in observations section
    obs_def = observations_dict.get(obs)
    has_pipeline = obs_def and obs_def.pipeline
    if obs_def:
        # Observable defined in observations - check source and aggregation
        obs_source = obs_def.source
        obs_aggregation = obs_def.aggregation
        # Convert enum to string for comparison
        agg_str = str(obs_aggregation) if obs_aggregation else None
        if obs_source and agg_str == 'mean' and not has_pipeline:
            # Simple aggregation case
            if str(obs_source) in state_var_names:
                state_idx = state_var_names.index(str(obs_source))
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

        # Apply update rules
% for rule_idx, (rule, rule_source, arg_overrides) in enumerate(all_update_rules_with_source):
<%
    rule_name = safe_name(rule.name)
    target_name = str(rule.target_parameter.name) if hasattr(rule.target_parameter, 'name') else str(rule.target_parameter)
    rule_rhs = rule.equation.rhs

    # Check if this rule comes from an included algorithm
    is_from_included = rule_source != str(algo.name)

    # Get hyperparameters from the source algorithm
    source_algo = algorithms_dict.get(rule_source)
    source_hyperparam_dict = get_hyperparam_dict(source_algo) if source_algo else {}

    # Apply argument overrides from includes (these take priority)
    effective_hyperparams = {**source_hyperparam_dict, **arg_overrides}

    # Build rule_params_dict from effective hyperparameters
    rule_params_dict = {}
    for pname, pval in effective_hyperparams.items():
        if pname in rule_rhs:
            rule_params_dict[pname] = pval

    # Check if this rule has an eta parameter that needs warmup
    eta_param_names = [p for p in rule_params_dict.keys() if p.lower() in ['eta', 'learning_rate', 'lr']]
    has_eta_param = len(eta_param_names) > 0
    # Use per-rule warmup setting (matches function definition)
    rule_has_warmup = getattr(rule, 'warmup', False)
    needs_warmup = rule_has_warmup and has_eta_param

    # Determine where to update the parameter (dynamics or coupling)
    coupling_key = coupling_param_to_key.get(target_name, None)
    is_coupling_param = coupling_key is not None
%>
        # Update ${target_name}${' (from included: ' + rule_source + ')' if is_from_included else ''}
        new_${target_name} = update_${algo_name}_${rule_name}(
% if is_coupling_param:
            state.coupling.${coupling_key}.${target_name},
% else:
            state.dynamics.${target_name},
% endif
% for obs in observations:
            ${obs},
% endfor
% for pname, pval in rule_params_dict.items():
            ${pname}=${pval},
% endfor
% if needs_warmup:
            eta_scale=eta_scale,
% endif
        )
% if is_coupling_param:
        state.coupling.${coupling_key}.${target_name} = new_${target_name}
% else:
        state.dynamics.${target_name} = new_${target_name}
% endif
% endfor

        # Record result history (defer conversion to end)
        # Record simulated observations (mean value for scalars)
% for obs in simulated_observations:
        result_history.${obs}.append(jnp.mean(${obs}))
% endfor

        # Compute and record algorithm functions (metrics, derived quantities)
% for func_call in algo_functions:
<%
    # FunctionCall has function (name reference) and arguments
    func_name = get_func_name(func_call)
    arg_values = get_func_args(func_call)

    # Generate function call
    call_args = ', '.join([f"{k}={v}" for k, v in arg_values.items()])
    func_call_str = f"{func_name}({call_args})"
%>
        _${func_name}_val = ${func_call_str}
        result_history.${func_name}.append(_${func_name}_val)
% endfor

        # Record parameter updates
% for rule, rule_source, arg_overrides in all_update_rules_with_source:
<%
    target_name = str(rule.target_parameter.name) if hasattr(rule.target_parameter, 'name') else str(rule.target_parameter)
    # Use coupling_param_to_key lookup
    rec_coupling_key = coupling_param_to_key.get(target_name, None)
    is_coupling_param = rec_coupling_key is not None
%>
% if is_coupling_param:
        result_history.${target_name}.append(state.coupling.${rec_coupling_key}.${target_name})
% else:
        result_history.${target_name}.append(state.dynamics.${target_name})
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

    # Run post-tuning simulation
    post_tuning = model_fn(state)

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

    if verbose:
        print(f"${algo_name} complete!")

    return Bunch(
        state=state,
        history=result_history,
        pre_tuning=pre_tuning,
        post_tuning=post_tuning,
% for obs in collectible_observations:
        ${obs}_buffer=_${obs}_buffer_out,  # Raw samples for passing to next algorithm
% endfor
% if use_sliding_window:
% for src_obs in source_observations_needed:
        ${src_obs}_buffer=_${src_obs}_buffer,  # Current sliding window buffer
% endfor
% endif
    )

% endfor
% endif
