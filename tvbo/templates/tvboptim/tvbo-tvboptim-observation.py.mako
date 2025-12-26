# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Observation/Monitor Template
======================================

Generates observation functions from pipeline-based Observation definitions.

Each Observation can have:
- source: StateVariable to observe
- source_observation: Another Observation to derive from
- pipeline: List of Function steps with callable/equation definitions

Pipeline Functions support:
- callable: {name, module, qualname} for importing external functions
- equation: {rhs, parameters} for symbolic expressions
- arguments: Named arguments with values
- input/output: Data flow between pipeline steps

Context Variables:
- experiment: SimulationExperiment instance (required)

Output:
- Observation functions with proper imports and pipeline execution
</%doc>
<%
from tvbo.export.code import render_expression

# Get experiment info
model = experiment.local_dynamics
state_names = list(model.state_variables.keys()) if model else ['x']
dt = experiment.integration.step_size if experiment.integration else 0.1

jaxcode = lambda expr: render_expression(expr, format='jax')

def to_numeric(val):
    """Convert string to numeric if possible."""
    if isinstance(val, (int, float)):
        return val
    if isinstance(val, str):
        try:
            return int(val) if '.' not in val else float(val)
        except ValueError:
            return val
    return val

# Get observations
observations = getattr(experiment, 'observations', None) or {}
if hasattr(observations, 'values'):
    observations = dict(observations.items()) if hasattr(observations, 'items') else {}
elif hasattr(observations, '__iter__') and not isinstance(observations, dict):
    observations = {getattr(o, 'name', f'obs_{i}'): o for i, o in enumerate(observations)}

# Collect all unique imports from callables across all observations
callable_imports = {}  # {module: set(qualnames)}
for obs_name, obs in observations.items():
    pipeline = getattr(obs, 'pipeline', None) or []
    for func in pipeline:
        callable_def = getattr(func, 'callable', None)
        if callable_def:
            module = getattr(callable_def, 'module', None)
            qualname = getattr(callable_def, 'qualname', None)
            if module and qualname:
                if module not in callable_imports:
                    callable_imports[module] = set()
                callable_imports[module].add(qualname)

# Parse observations into structured info
obs_list = []
for obs_name, obs in observations.items():
    obs_info = {
        'name': obs_name,
        'label': getattr(obs, 'label', ''),
        'description': getattr(obs, 'description', ''),
        'source': None,
        'source_observation': None,
        'pipeline': [],
    }

    # Source state variable
    if hasattr(obs, 'source') and obs.source:
        src = obs.source
        obs_info['source'] = getattr(src, 'name', str(src)) if hasattr(src, 'name') else str(src)

    # Source observation (for derived observations)
    if hasattr(obs, 'source_observation') and obs.source_observation:
        src_obs = obs.source_observation
        obs_info['source_observation'] = getattr(src_obs, 'name', str(src_obs)) if hasattr(src_obs, 'name') else str(src_obs)

    # Parse pipeline steps
    pipeline = getattr(obs, 'pipeline', None) or []
    for func in pipeline:
        func_info = {
            'name': getattr(func, 'name', 'step'),
            'output': getattr(func, 'output', None),
            'input': None,
            'callable': None,
            'equation': None,
            'source_code': None,
            'arguments': {},
        }

        # Input reference
        if hasattr(func, 'input') and func.input:
            inp = func.input
            func_info['input'] = getattr(inp, 'name', str(inp)) if hasattr(inp, 'name') else str(inp)

        # Callable reference
        callable_def = getattr(func, 'callable', None)
        if callable_def:
            func_info['callable'] = {
                'name': getattr(callable_def, 'name', None),
                'module': getattr(callable_def, 'module', None),
                'qualname': getattr(callable_def, 'qualname', None),
            }

        # Source code (inline function definition)
        source_code = getattr(func, 'source_code', None)
        if source_code:
            func_info['source_code'] = str(source_code)

        # Equation
        eq = getattr(func, 'equation', None)
        if eq:
            func_info['equation'] = getattr(eq, 'rhs', None)
            # Extract equation parameters
            eq_params = getattr(eq, 'parameters', None) or {}
            if hasattr(eq_params, 'items'):
                for pname, pobj in eq_params.items():
                    if hasattr(pobj, 'value'):
                        func_info['arguments'][pname] = to_numeric(pobj.value)

        # Arguments
        args = getattr(func, 'arguments', None) or []
        if hasattr(args, '__iter__'):
            for arg in args:
                arg_name = getattr(arg, 'name', None)
                arg_value = getattr(arg, 'value', None)
                if arg_name and arg_value is not None:
                    func_info['arguments'][arg_name] = to_numeric(arg_value)

        obs_info['pipeline'].append(func_info)

    obs_list.append(obs_info)
%>
"""
Observation Functions for tvboptim
==================================

Auto-generated from TVBO pipeline-based Observation definitions.

Observations defined: ${len(obs_list)}
% for obs in obs_list:
- ${obs['name']}: ${obs['label'] or obs['description'][:50] + '...' if obs['description'] and len(obs['description']) > 50 else obs['description'] or 'No description'}
% endfor
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Any

# Pipeline callable imports
% for module, qualnames in sorted(callable_imports.items()):
% if '.' in module:
<%
    parts = module.split('.')
    base_module = parts[0]
%>
import ${base_module}
% for qn in sorted(qualnames):
${qn} = ${module}.${qn}
% endfor
% else:
from ${module} import ${', '.join(sorted(qualnames))}
% endif
% endfor


# =============================================================================
# Observation Functions
# =============================================================================
% for obs in obs_list:
<%
    obs_name = obs['name']
    obs_source = obs['source']
    obs_src_obs = obs['source_observation']
    pipeline = obs['pipeline']
    has_pipeline = len(pipeline) > 0
%>

def ${obs_name}(model_fn, state, dt: float = ${dt}, **kwargs) -> Any:
    """${obs['label'] or obs_name}

    ${obs['description'] or 'Auto-generated observation function.'}
% if has_pipeline:

    Pipeline steps:
% for i, step in enumerate(pipeline):
    ${i + 1}. ${step['name']}: ${'callable=' + step['callable']['module'] + '.' + step['callable']['qualname'] if step['callable'] and step['callable']['module'] else 'equation=' + (step['equation'][:40] + '...' if step['equation'] and len(step['equation']) > 40 else step['equation'] or 'N/A')}
% endfor
% endif
    """
    # Initialize outputs dictionary
    _outputs = {}

% if obs_src_obs:
    # Derived observation - get data from source observation
    _source_result = ${obs_src_obs}(model_fn, state, dt=dt, **kwargs)
    if isinstance(_source_result, dict):
        _outputs.update(_source_result)
    else:
        _outputs['data'] = _source_result
% elif obs_source:
    # Root observation - run simulation and extract state variable
    _result = model_fn(state)
    _outputs['data'] = _result.data[:, ${state_names.index(obs_source) if obs_source in state_names else 0}, :]
% else:
    # Generic observation - run simulation
    _result = model_fn(state)
    _outputs['data'] = _result.data
% endif
% if has_pipeline:

    # Execute pipeline
% for step in pipeline:
<%
    step_name = step['name']
    step_output = step['output'] or 'data'
    step_input = step['input'] or 'data'
    step_callable = step['callable']
    step_equation = step['equation']
    step_source_code = step['source_code']
    step_args = step['arguments']

    # Build keyword arguments string (repr handles quoting correctly for all types)
    arg_strs = [f"{aname}={repr(aval)}" for aname, aval in step_args.items()]
    args_str = ', '.join(arg_strs)
%>
    # Step: ${step_name} (${step_input} -> ${step_output})
% if step_source_code:
    _fn_${step_name} = ${step_source_code}
% if '_outputs' in step_source_code:
    _outputs['${step_output}'] = _fn_${step_name}(_outputs${', ' + args_str if args_str else ''})
% else:
    _outputs['${step_output}'] = _fn_${step_name}(_outputs['${step_input}']${', ' + args_str if args_str else ''})
% endif
% elif step_callable and step_callable.get('module') and step_callable.get('qualname'):
<%
    fn_ref = step_callable['qualname']
    # Check if output contains comma (multiple outputs)
    outputs = [o.strip() for o in step_output.split(',')]
    is_multi_output = len(outputs) > 1
%>
% if is_multi_output:
    ${', '.join(["_outputs['" + o + "']" for o in outputs])} = ${fn_ref}(_outputs['${step_input}']${', ' + args_str if args_str else ''})
% else:
    _outputs['${step_output}'] = ${fn_ref}(_outputs['${step_input}']${', ' + args_str if args_str else ''})
% endif
% elif step_equation:
<%
    # Render equation with JAX syntax - substitute output references
    rendered_eq = jaxcode(step_equation)
%>
    _outputs['${step_output}'] = ${rendered_eq}
% endif
% endfor
% endif

    # Return pipeline outputs
    return _outputs

% endfor
