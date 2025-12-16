<%!
from tvbo.export.code import render_expression

def get_callable_info(func):
    return func.callable.module, func.callable.name

def get_parameters(func, pipeline_outputs):
    params = {}
    for arg in func.arguments:
        value = arg.value
        # Quote string values that aren't numeric or pipeline references
        if isinstance(value, str):
            # Check if it's a number string or a pipeline reference (no quotes needed)
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
<%def name="create_observation_pipeline(observation, dt)" filter="trim">
<%
    func_name_to_output = {func.name: func.output for func in observation.pipeline}
    # Collect imports for this observation
    obs_imports = set()
    for func in observation.pipeline:
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
% for func in observation.pipeline:
% if func.callable:
_jax_${func.callable.name} = ${func.callable.name}  # Store reference to avoid recursion
% endif
% endfor

% for func in observation.pipeline:
${jaxfunc.generate_function(func, func.name)}

% endfor

# Compose functions into observation pipeline
def ${observation.name}(ts: TimeSeries, state=${"'" + observation.source + "'" if hasattr(observation, 'source') and observation.source else 'None'}):
    # Extract state variable if specified
    if state is not None:
        ts = ts.get_state(state)

% for func in observation.pipeline:
<%
    # Check if input is from pipeline outputs or if it's another observation (needs to be called)
    pipeline_outputs = set([f.output for f in observation.pipeline])

    if func.input:
        # Check if input is from this pipeline's outputs
        if func.input in pipeline_outputs or func.input in func_name_to_output.values():
            input_name = func_name_to_output.get(func.input, func.input)
        # Check if input matches the observation's source_observation (cross-observation dependency)
        elif hasattr(observation, 'source_observation') and func.input == observation.source_observation:
            # Need to call the source observation function
            input_name = f"{func.input}(ts)"
        else:
            # Unknown input - use as variable name
            input_name = func.input
    else:
        input_name = 'ts'

    params_dict = get_parameters(func, pipeline_outputs)
    # Build argument list: input + schema arguments
    args = [input_name]
    for arg_name, arg_value in params_dict.items():
        args.append(f"{arg_name}={arg_value}")
    args_str = ', '.join(args)
%>
    ${func.output} = ${func.name}(${args_str})
% endfor
    return ${observation.pipeline[-1].output}
</%def>

% if 'observation' in context.keys():
${create_observation_pipeline(observation, dt if 'dt' in context.keys() else 0.1)}
% endif

<%def name="create_all_observations(experiment)" filter="trim">
<%
    dt = experiment.integration.step_size
%>
% for name, obs in experiment.observations.items():

${create_observation_pipeline(obs, dt)}

% endfor

def apply_observations(trace, time_steps, dt=${dt}, params=None):
    results = {}
    % for name in experiment.observations.keys():
    results['${name}'] = observe_${name}(trace, time_steps, dt=dt, params=params)
    % endfor
    return results

</%def>
