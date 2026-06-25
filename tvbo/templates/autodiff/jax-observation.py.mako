<%!
from tvbo.codegen import render_expression

def get_callable_info(func):
    return func.callable.module, func.callable.name

def get_parameters(func, pipeline_outputs, obs_period=None, dt=None):
    """Return {arg_name: value_str} for arguments that have a resolved value.

    Arguments without an explicit value are skipped — they appear as required
    positional parameters in the generated function signature (see
    generate_function in jax-function.py.mako) and must be passed at the call
    site.

    ``obs_period`` and ``dt`` are used to derive windowing arguments
    (window_size, window) when the YAML omits an explicit value.
    """
    _WINDOW_ARGS = ('window_size', 'window')

    params = {}
    args = func.arguments or []
    # Handle dict-keyed arguments (from Function) or list (from FunctionCall)
    if hasattr(args, 'values'):
        args = args.values()
    for arg in args:
        value = getattr(arg, 'value', None)
        if value is None:
            # Derive windowing parameters from observation period and integration dt
            if obs_period is not None and dt is not None and arg.name in _WINDOW_ARGS:
                value = str(int(round(float(obs_period) / float(dt))))
        if value is None:
            continue
        # Quote string values that aren't numeric or pipeline references
        if isinstance(value, str):
            is_numeric = value.replace('.','').replace('-','').isdigit()
            is_pipeline_ref = value in pipeline_outputs
            if not is_numeric and not is_pipeline_ref:
                value = f"'{value}'"
        params[arg.name] = value
    return params

def get_equation_rhs(func):
    return func.equation.rhs

JAX_MODULE_MAP = {
    'scipy.signal': 'jax.scipy.signal',
    'scipy.linalg': 'jax.scipy.linalg',
    'scipy.special': 'jax.scipy.special',
    'scipy.stats': 'jax.scipy.stats',
    'numpy': 'jax.numpy',
    'np': 'jax.numpy',
}

# Functions that don't exist in JAX - use scipy directly
JAX_UNAVAILABLE = {
    'scipy.signal': ['decimate'],
    'numpy': ['corrcoef'],
}

def get_jax_module(module, func_name=None):
    """Map module to JAX equivalent, but check if function is available."""
    # Check if this specific function is known to be unavailable in JAX
    if func_name:
        for mod_prefix, unavailable_funcs in JAX_UNAVAILABLE.items():
            if module.startswith(mod_prefix) and func_name in unavailable_funcs:
                return module  # Use original module

    # Otherwise try to map to JAX
    for prefix, jax_prefix in JAX_MODULE_MAP.items():
        if module.startswith(prefix):
            return module.replace(prefix, jax_prefix, 1)
    return module
%>
<%

pipeline_imports = set()
callable_names = []
for func in observation.pipeline:
    if func.callable:
        module, qualname = get_callable_info(func)
        pipeline_imports.add((module, qualname))
        callable_names.append(qualname)
%>
import jax
import jax.numpy as jnp
from tvbo.data.types import TimeSeries
% for module, name in sorted(pipeline_imports):
<%
    jax_module = get_jax_module(module, name)
%>
from ${jax_module} import ${name} as ${name}
% endfor
<%namespace name="jaxfunc" file="jax-function.py.mako"/>
<%def name="create_observation_pipeline(observation, dt, functions=None)" filter="trim">
<%
    # Helper to get function name from FunctionCall or resolved Function
    def get_func_name(func):
        if hasattr(func, 'function') and func.function:
            return str(func.function)
        if hasattr(func, 'name') and func.name:
            return str(func.name)
        if hasattr(func, 'callable') and func.callable and hasattr(func.callable, 'name'):
            return str(func.callable.name)
        return None

    # Resolve function references: if a FunctionCall has only function: name,
    # look up the actual Function definition from experiment.functions
    resolved_pipeline = []
    for fc in observation.pipeline:
        fname = str(fc.function) if fc.function else None
        if fname and functions and fname in functions:
            # Use the resolved Function object for code generation,
            # but carry over FunctionCall-specific fields (output, arguments, apply_on_dimension)
            resolved = functions[fname]
            # If FunctionCall specifies output, use it (override Function default)
            if fc.output and not getattr(resolved, 'output', None):
                resolved.output = fc.output
            # If FunctionCall specifies arguments, merge them
            if fc.arguments:
                resolved._fc_arguments = fc.arguments
            resolved_pipeline.append(resolved)
        else:
            resolved_pipeline.append(fc)

    func_name_to_output = {get_func_name(func): func.output for func in resolved_pipeline if get_func_name(func)}
    # Collect imports for this observation
    obs_imports = set()
    for func in resolved_pipeline:
        if func.callable:
            module, qualname = get_callable_info(func)
            obs_imports.add((module, qualname))
%>
# Import callable functions for ${observation.name}
% for module, name in sorted(obs_imports):
<%
    jax_module = get_jax_module(module, name)
%>
from ${jax_module} import ${name}
% endfor

# Generate all transform functions from schema
% for func in resolved_pipeline:
% if func.callable:
_jax_${func.callable.name} = ${func.callable.name}  # Store reference to avoid recursion
% endif
% endfor

% for func in resolved_pipeline:
${jaxfunc.generate_function(func, get_func_name(func))}

% endfor

<%
    # ``source`` is multivalued; for raw observations there is one entry
    # (a state-variable reference). Take the first scalar value.
    _src = getattr(observation, 'source', None)
    if isinstance(_src, (list, tuple)):
        _src = _src[0] if _src else None
    if _src is not None and hasattr(_src, 'name'):
        _src = _src.name
    _state_default = "'" + str(_src) + "'" if _src else 'None'
%>
# Compose functions into observation pipeline
def ${observation.name}(ts: TimeSeries, state=${_state_default}):
    # Extract state variable if specified
    if state is not None:
        ts = ts.get_state(state)

% for i_step, func in enumerate(resolved_pipeline):
<%
    # Check if input is from pipeline outputs or if it's another observation (needs to be called)
    pipeline_outputs = set([f.output for f in resolved_pipeline if f.output])
    _func_input = getattr(func, 'input', None)

    if _func_input:
        if _func_input in pipeline_outputs:
            # Direct reference to an output variable name — use as-is.
            input_name = _func_input
        elif _func_input in func_name_to_output:
            # Reference to a *function* name — resolve to its output variable.
            # e.g. YAML says  input: temporal_average_interim
            #       → should use  interim_averaged  (that function's output var)
            input_name = func_name_to_output[_func_input]
        # Check if input matches the observation's source_observation (cross-observation dependency)
        elif hasattr(observation, 'source_observation') and _func_input == observation.source_observation:
            # Need to call the source observation function
            input_name = f"{_func_input}(ts)"
        else:
            # Unknown input - use as variable name
            input_name = _func_input
    elif i_step > 0 and resolved_pipeline[i_step - 1].output:
        # Auto-chain: use previous step's output
        input_name = resolved_pipeline[i_step - 1].output
    else:
        input_name = 'ts'

    params_dict = get_parameters(
        func, pipeline_outputs,
        obs_period=getattr(observation, 'period', None),
        dt=dt,
    )
    # Build argument list: input + schema arguments
    args = [input_name]
    for arg_name, arg_value in params_dict.items():
        args.append(f"{arg_name}={arg_value}")
    args_str = ', '.join(args)
%>
    ${func.output} = ${get_func_name(func)}(${args_str})
% endfor
    return ${resolved_pipeline[-1].output}
</%def>

% if 'observation' in context.keys():
${create_observation_pipeline(observation, dt if 'dt' in context.keys() else 0.1)}
% endif

<%def name="create_all_observations(experiment)" filter="trim">
<%
    dt = experiment.integration.step_size
    exp_functions = getattr(experiment, 'functions', None) or {}
%>
% for name, obs in experiment.observations.items():

${create_observation_pipeline(obs, dt, functions=exp_functions)}

% endfor

def apply_observations(trace, time_steps, dt=${dt}, params=None):
    results = {}
    % for name in experiment.observations.keys():
    results['${name}'] = observe_${name}(trace, time_steps, dt=dt, params=params)
    % endfor
    return results

</%def>
