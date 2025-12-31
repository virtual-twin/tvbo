# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Algorithm Template
============================

Generates iterative parameter tuning algorithms from Algorithm definitions.

Algorithms are iterative (non-gradient) methods like:
- FIC (Feedback Inhibition Control): Adjusts J_i to maintain target S_e
- Custom: User-defined update rules

Each algorithm generates:
- update_<name>(): Update function that modifies target parameters
- run_<name>(): Main loop running n_iterations of simulate + update

Context: experiment (SimulationExperiment instance)
</%doc>
<%
from tvbo.export.code import render_expression

# Define jaxcode locally (same as parent template)
_exp_functions = getattr(experiment, 'functions', None) or {}
if hasattr(_exp_functions, 'items'):
    _user_functions = {str(fname): str(fname) for fname in _exp_functions.keys()}
elif hasattr(_exp_functions, '__iter__'):
    _user_functions = {str(getattr(f, 'name', f)): str(getattr(f, 'name', f)) for f in _exp_functions}
else:
    _user_functions = {}

jaxcode = lambda expr, params=None: render_expression(expr, format='jax', user_functions=_user_functions, parameters=params)

# =============================================================================
# Extract Algorithms from experiment
# =============================================================================
algorithms_raw = getattr(experiment, 'algorithms', None) or []
if isinstance(algorithms_raw, dict):
    algorithms_list = list(algorithms_raw.values())
elif isinstance(algorithms_raw, list):
    algorithms_list = algorithms_raw
else:
    algorithms_list = [algorithms_raw] if algorithms_raw else []

has_algorithms = len(algorithms_list) > 0

def safe_name(name):
    """Convert name to valid Python identifier."""
    return str(name).replace(' ', '_').replace('-', '_').lower()

def get_param_value(params_dict, name, default=None):
    """Extract parameter value from dict-like or object."""
    if hasattr(params_dict, 'items'):
        p = params_dict.get(name)
        if p is not None:
            return getattr(p, 'value', p)
    return default

%>
% if has_algorithms:
# =============================================================================
# ITERATIVE ALGORITHMS
# =============================================================================

% for algo in algorithms_list:
<%
    algo_name = safe_name(getattr(algo, 'name', 'algorithm'))
    algo_type = str(getattr(algo, 'type', 'custom')).lower()
    n_iterations = int(getattr(algo, 'n_iterations', 100))
    learning_rate = float(getattr(algo, 'learning_rate', 0.1))
    simulation_period = getattr(algo, 'simulation_period', None)

    # Extract update rules
    update_rules = getattr(algo, 'update_rules', None) or []
    if hasattr(update_rules, 'values'):
        update_rules = list(update_rules.values())

    # Extract hyperparameters for default values
    hyperparams = getattr(algo, 'hyperparameters', None) or []
    if hasattr(hyperparams, 'values'):
        hyperparams = list(hyperparams.values())
    hyperparam_defaults = {}
    for hp in hyperparams:
        hp_name = str(getattr(hp, 'name', 'param'))
        hp_val = getattr(hp, 'value', 0.0)
        hyperparam_defaults[hp_name] = hp_val

    # Extract observables needed for this algorithm
    observables = getattr(algo, 'observables', None) or []
    if hasattr(observables, '__iter__') and not isinstance(observables, str):
        observables = list(observables)
    else:
        observables = [observables] if observables else []
%>
# -----------------------------------------------------------------------------
# Algorithm: ${algo_name} (${algo_type})
# -----------------------------------------------------------------------------

% for rule in update_rules:
<%
    rule_name = safe_name(getattr(rule, 'name', 'update'))
    target_param = getattr(rule, 'target_parameter', None)
    if target_param:
        target_name = str(getattr(target_param, 'name', target_param))
    else:
        target_name = 'param'

    rule_eq = getattr(rule, 'equation', None)
    if rule_eq:
        rule_rhs = getattr(rule_eq, 'rhs', None)
        rule_params = getattr(rule_eq, 'parameters', None) or {}
        if hasattr(rule_params, 'items'):
            rule_params_dict = {k: getattr(v, 'value', v) for k, v in rule_params.items()}
        else:
            rule_params_dict = {}
    else:
        rule_rhs = target_name
        rule_params_dict = {}

    # Bounds for clipping
    bounds = getattr(rule, 'bounds', None)
    lo_bound = float(getattr(bounds, 'lo', 0.0)) if bounds else None
    hi_bound = float(getattr(bounds, 'hi', float('inf'))) if bounds else None
    has_bounds = bounds is not None
%>
def update_${algo_name}_${rule_name}(
    ${target_name}: jnp.ndarray,
% for obs in observables:
    ${obs}: jnp.ndarray,
% endfor
% for pname, pval in rule_params_dict.items():
    ${pname}: float = ${pval},
% endfor
) -> jnp.ndarray:
    """
    Update rule for ${target_name} in ${algo_name} algorithm.

    ${getattr(rule, 'description', f'Updates {target_name} based on observables.')}

    Args:
        ${target_name}: Current parameter value(s)
% for obs in observables:
        ${obs}: Observable from simulation
% endfor
% for pname in rule_params_dict.keys():
        ${pname}: Update rule hyperparameter
% endfor

    Returns:
        Updated ${target_name} value(s)
    """
    # Compute update
    # Pass only param names (not values) to keep them symbolic in the expression
    updated = ${jaxcode(rule_rhs, list(rule_params_dict.keys())) if rule_rhs else target_name}
% if has_bounds:
    # Apply bounds
    updated = jnp.clip(updated, ${lo_bound if lo_bound is not None else 'None'}, ${hi_bound if hi_bound is not None else 'None'})
% endif
    return updated

% endfor

def run_${algo_name}(
    model_fn: Callable,
    params: PyTree,
    state: jnp.ndarray,
    key: jax.random.PRNGKey,
    n_iterations: int = ${n_iterations},
% for pname, pval in hyperparam_defaults.items():
    ${pname}: float = ${pval},
% endfor
    verbose: bool = True,
) -> Tuple[PyTree, List[Dict]]:
    """
    Run the ${algo_name} algorithm for n_iterations.

    ${getattr(algo, 'description', f'Iterative algorithm that updates parameters based on simulation results.')}

    Args:
        model_fn: Simulation function (params, state, key) -> (state, result)
        params: Initial parameter values (PyTree)
        state: Initial simulation state
        key: Random key for simulation
        n_iterations: Number of iterations to run
% for pname in hyperparam_defaults.keys():
        ${pname}: Hyperparameter for algorithm
% endfor
        verbose: Print progress

    Returns:
        Tuple of (final_params, history) where history contains per-iteration info
    """
    history = []
    current_params = params

    for i in range(n_iterations):
        # Split key for this iteration
        key, subkey = jax.random.split(key)

        # Run simulation with current parameters
        new_state, result = model_fn(current_params, state, subkey)

        # Compute observables from result
% for obs in observables:
        ${obs} = compute_${obs}(result)  # User must define compute_${obs}
% endfor

        # Apply update rules
% for rule in update_rules:
<%
    rule_name = safe_name(getattr(rule, 'name', 'update'))
    target_param = getattr(rule, 'target_parameter', None)
    target_name = str(getattr(target_param, 'name', target_param)) if target_param else 'param'
    rule_eq = getattr(rule, 'equation', None)
    rule_params = getattr(rule_eq, 'parameters', None) or {} if rule_eq else {}
    if hasattr(rule_params, 'items'):
        rule_params_dict = {k: getattr(v, 'value', v) for k, v in rule_params.items()}
    else:
        rule_params_dict = {}
%>
        new_${target_name} = update_${algo_name}_${rule_name}(
            current_params['${target_name}'],
% for obs in observables:
            ${obs},
% endfor
% for pname in rule_params_dict.keys():
            ${pname}=${pname},
% endfor
        )
        current_params = {**current_params, '${target_name}': new_${target_name}}
% endfor

        # Record history
        history.append({
            'iteration': i,
% for obs in observables:
            '${obs}': float(jnp.mean(${obs})),
% endfor
% for rule in update_rules:
<%
    target_param = getattr(rule, 'target_parameter', None)
    target_name = str(getattr(target_param, 'name', target_param)) if target_param else 'param'
%>
            '${target_name}': current_params['${target_name}'].copy() if hasattr(current_params['${target_name}'], 'copy') else current_params['${target_name}'],
% endfor
        })

        if verbose and (i + 1) % max(1, n_iterations // 10) == 0:
            print(f"${algo_name} iteration {i + 1}/{n_iterations}")

    return current_params, history

% endfor
% endif
