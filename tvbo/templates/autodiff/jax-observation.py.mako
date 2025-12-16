<%
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

def get_jax_module(module):
    for prefix, jax_prefix in JAX_MODULE_MAP.items():
        if module.startswith(prefix):
            return module.replace(prefix, jax_prefix, 1)
    return module

pipeline_imports = set()
for func in observation.pipeline:
    if func.callable:
        module, qualname = get_callable_info(func)
        pipeline_imports.add((module, qualname))
%>
import jax
import jax.numpy as jnp
from tvbo.data.types import TimeSeries
% for module, name in sorted(pipeline_imports):
<%
    jax_module = get_jax_module(module)
%>
from ${jax_module} import ${name}
% endfor
<%namespace name="jaxfunc" file="/jax-function.py.mako"/>
<%def name="create_observation_pipeline(observation, dt)" filter="trim">
<%
    func_name_to_output = {func.name: func.output for func in observation.pipeline}
%>
# Generate all transform functions from schema
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
    input_name = func_name_to_output.get(func.input, func.input) if func.input else 'ts'
    pipeline_outputs = set([f.output for f in observation.pipeline])
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
