# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Observation Template
==============================

Generates observation functions from pipeline-based Observation definitions.

Reference Syntax (declarative):
- step_name        → pipeline step output (_outputs['step_name'])
- obs.step_name    → self-referential observation.step (_outputs['step_name'])
- input.key        → pipeline local output (_outputs['key'])
- integration.transient → transient simulation data
- integration.result    → main simulation result
- observation.key  → another observation's output

Context: experiment (SimulationExperiment instance)
</%doc>
<%
from tvbo.export.code import render_expression

# =============================================================================
# Configuration
# =============================================================================
model = experiment.local_dynamics
state_names = list(model.state_variables.keys()) if model else ['x']
dt = experiment.integration.step_size if experiment.integration else 0.1

# =============================================================================
# Helper Functions
# =============================================================================
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

def get_attr(obj, name, default=None):
    """Safe attribute access."""
    return getattr(obj, name, default) if obj else default

def is_numeric_string(s):
    """Check if string represents a number."""
    return s.replace('.', '').replace('-', '').replace('_', '').isdigit()

# =============================================================================
# Reference Parser
# =============================================================================
def parse_reference(val, step_names=None, current_obs_name=None):
    """Parse argument value into (ref_type, ref_value).

    Reference types:
    - 'step'        : pipeline step output (_outputs[step_name])
    - 'input'       : pipeline local (_outputs[key])
    - 'integration' : simulation result (transient/result)
    - 'observation' : another observation's output
    - 'source_data' : data from source_observation (used when value matches source_observation name)
    - 'literal'     : direct value
    """
    if not isinstance(val, str):
        return ('literal', val)

    # Check for prefix.key syntax
    if '.' in val and not is_numeric_string(val):
        prefix, key = val.split('.', 1)

        if prefix == 'input':
            return ('input', key)
        if prefix == 'integration':
            return ('integration', key)
        # Self-referential: bold.hrf_kernel within bold observation
        if current_obs_name and prefix == current_obs_name:
            return ('step', key)
        if prefix in observations:
            return ('observation', (prefix, key))

    # Check for direct step name reference (e.g., hrf_kernel)
    if step_names and val in step_names:
        return ('step', val)

    # Check for source observation reference (e.g., 'bold' when source_observation: bold)
    # This happens when an observation references its source observation's data
    if val in observations:
        return ('source_data', val)

    # Try numeric conversion
    try:
        return ('literal', float(val) if '.' in val else int(val))
    except ValueError:
        return ('literal', val)

def ref_to_code(ref_type, ref_val, state_idx=None):
    """Convert reference to Python code expression."""
    if ref_type == 'step':
        return f"_outputs['{ref_val}']"
    if ref_type == 'input':
        return f"_outputs['{ref_val}']"
    if ref_type == 'source_data':
        # Reference to source observation data (already in _outputs['data'] or _outputs)
        return "_outputs.get('data', _outputs)"
    if ref_type == 'integration':
        if ref_val == 'transient':
            if state_idx is not None:
                # Slice to source state and squeeze state dimension → (time, nodes)
                return f"_result_transient.data[:, {state_idx}, :]"
            return "_result_transient.data"
        if ref_val == 'result':
            if state_idx is not None:
                # Slice to source state and squeeze state dimension → (time, nodes)
                return f"_result.data[:, {state_idx}, :]"
            return "_result.data"
        return f"_result_{ref_val}"
    if ref_type == 'observation':
        obs_name, key = ref_val
        return f"_{obs_name}_result['{key}']"
    return repr(ref_val)

# =============================================================================
# Data Collection
# =============================================================================

# Build function lookup from experiment.functions
_exp_functions = get_attr(experiment, 'functions', {})
if hasattr(_exp_functions, 'items'):
    functions_by_name = {str(k): v for k, v in _exp_functions.items()}
else:
    functions_by_name = {str(get_attr(f, 'name', '')): f for f in (_exp_functions or [])}

# User-defined function names for expression rendering
user_functions = {name: name for name in functions_by_name.keys()}
jaxcode = lambda expr, params=None: render_expression(expr, format='jax', user_functions=user_functions, parameters=params)

# Get observations dict
_obs_raw = get_attr(experiment, 'observations', {})
if hasattr(_obs_raw, 'items'):
    observations = dict(_obs_raw.items())
elif hasattr(_obs_raw, '__iter__') and not isinstance(_obs_raw, dict):
    observations = {get_attr(o, 'name', f'obs_{i}'): o for i, o in enumerate(_obs_raw)}
else:
    observations = {}

# =============================================================================
# Parse Pipeline Step
# =============================================================================
def parse_step(func, step_name):
    """Parse a pipeline step into a clean dict structure."""
    _inp = get_attr(func, 'input')
    step = {
        'name': step_name,
        'output': get_attr(func, 'output'),
        'input': str(_inp) if _inp else None,
        'callable': None,
        'equation': None,
        'equation_params': {},  # Local equation constants, not function args
        'source_code': get_attr(func, 'source_code'),
        'arguments': {},
        'arg_names': [],
        'apply_on_dimension': None,
    }

    # Parse apply_on_dimension (for vmap generation)
    apply_dim = get_attr(func, 'apply_on_dimension')
    if apply_dim:
        step['apply_on_dimension'] = str(apply_dim).split('.')[-1]  # Handle 'DimensionType.node' -> 'node'

    # Parse callable (inline on FunctionCall or on Function)
    c = get_attr(func, 'callable')
    if c:
        cname = get_attr(c, 'name')
        cmodule = get_attr(c, 'module')
        step['callable'] = {
            'name': cname,
            'module': cmodule,
            # Full qualified call: module.name
            'full_call': f"{cmodule}.{cname}" if cmodule else cname,
        }

    # Parse equation
    eq = get_attr(func, 'equation')
    if eq:
        step['equation'] = get_attr(eq, 'rhs')
        # Note: equation.parameters are LOCAL constants, not function arguments
        # They are used when rendering the equation, not passed as kwargs
        step['equation_params'] = {}
        for pname, pobj in (get_attr(eq, 'parameters') or {}).items():
            if hasattr(pobj, 'value'):
                step['equation_params'][str(pname)] = to_numeric(pobj.value)

    # Parse arguments (these ARE function arguments)
    for arg in (get_attr(func, 'arguments') or []):
        name = get_attr(arg, 'name')
        if name:
            step['arg_names'].append(name)
            val = get_attr(arg, 'value')
            if val is not None:
                step['arguments'][name] = val

    # Lookup from functions section if no inline definition
    if not (step['source_code'] or step['callable'] or step['equation']):
        fn_def = functions_by_name.get(step_name)
        if fn_def:
            step['source_code'] = get_attr(fn_def, 'source_code')

            fn_callable = get_attr(fn_def, 'callable')
            if fn_callable:
                cname = get_attr(fn_callable, 'name')
                cmodule = get_attr(fn_callable, 'module')
                step['callable'] = {
                    'name': cname,
                    'module': cmodule,
                    'full_call': f"{cmodule}.{cname}" if cmodule else cname,
                }

            fn_eq = get_attr(fn_def, 'equation')
            if fn_eq:
                step['equation'] = get_attr(fn_eq, 'rhs')

            # Merge function arguments (step args take precedence)
            for arg in (get_attr(fn_def, 'arguments') or []):
                name = get_attr(arg, 'name')
                if name and name not in step['arg_names']:
                    step['arg_names'].append(name)
                val = get_attr(arg, 'value')
                if name and val is not None and name not in step['arguments']:
                    step['arguments'][name] = to_numeric(val)

    return step

# =============================================================================
# Analyze Pipeline
# =============================================================================
def analyze_pipeline(pipeline):
    """Analyze pipeline to determine data requirements."""
    needs_transient = False
    needs_result = False

    for step in pipeline:
        # Check step input field
        step_input = step.get('input')
        if step_input:
            ref_type, ref_val = parse_reference(step_input)
            if ref_type == 'integration':
                if 'transient' in str(ref_val):
                    needs_transient = True
                if 'result' in str(ref_val):
                    needs_result = True

        # Check arguments
        for val in step.get('arguments', {}).values():
            ref_type, ref_val = parse_reference(val)
            if ref_type == 'integration':
                if 'transient' in str(ref_val):
                    needs_transient = True
                if 'result' in str(ref_val):
                    needs_result = True

    return needs_transient, needs_result

def is_kernel_generator(step_name):
    """Check if function is a kernel generator (has time_range)."""
    fn_def = functions_by_name.get(step_name)
    return fn_def and get_attr(fn_def, 'time_range')

def build_vmap_call(callable_ref, step, step_names, current_obs_name, state_idx):
    """Build a vmap-wrapped callable for apply_on_dimension: node.

    For fftconvolve: vmap(lambda x: fftconvolve(x, kernel, mode='valid'), in_axes=1, out_axes=1)(data)
    """
    args = step['arguments']
    arg_names = list(args.keys())

    # First argument is the data being vmapped over
    data_arg = arg_names[0] if arg_names else 'x'
    data_val = args.get(data_arg)
    ref_type, ref_val = parse_reference(data_val, step_names=step_names, current_obs_name=current_obs_name)
    data_code = ref_to_code(ref_type, ref_val, state_idx=state_idx)

    # Remaining arguments are constants (kernel, mode, etc.)
    const_args = []
    for name in arg_names[1:]:
        val = args[name]
        ref_type, ref_val = parse_reference(val, step_names=step_names, current_obs_name=current_obs_name)
        if ref_type == 'literal':
            # Quote strings
            if isinstance(ref_val, str):
                const_args.append(f"'{ref_val}'")
            else:
                const_args.append(str(ref_val))
        else:
            const_args.append(ref_to_code(ref_type, ref_val, state_idx=state_idx))

    const_str = ', '.join(const_args) if const_args else ''
    inner_call = f"{callable_ref}(x, {const_str})" if const_str else f"{callable_ref}(x)"

    return f"jax.vmap(lambda x: {inner_call}, in_axes=1, out_axes=1)({data_code})"

# =============================================================================
# Build Step Call
# =============================================================================
def build_step_call(step, step_input, step_names=None, current_obs_name=None, state_idx=None, is_first_step=False):
    """Build the function call for a pipeline step.

    For inline callables (step has 'callable'), use only explicit arguments.
    For defined functions, may add implicit input if needed.

    If is_first_step=True and an argument has no value, default to _outputs['data']
    (which comes from observation source or integration.result).
    """
    args = step['arguments']
    arg_names = step.get('arg_names', [])
    keyword = []
    obs_deps = set()
    has_inline_callable = step.get('callable') is not None

    # Build keyword args from explicit arguments
    for name, val in args.items():
        ref_type, ref_val = parse_reference(val, step_names=step_names, current_obs_name=current_obs_name)
        code = ref_to_code(ref_type, ref_val, state_idx=state_idx)

        if ref_type == 'observation':
            obs_deps.add(ref_val[0])

        keyword.append(f"{name}={code}")

    # Handle arguments that have names but no values
    # For first step, default to _outputs['data'] (from source)
    for name in arg_names:
        if name not in args:
            if is_first_step and name in ('data', 'X', 'x', 'input', 'timeseries'):
                # First step's primary input defaults to observation source data
                keyword.append(f"{name}=_outputs['data']")

    # For inline callables, use ONLY explicit arguments - no implicit input
    # For defined functions without explicit args, may need implicit input
    if not has_inline_callable and not args and not arg_names and not is_kernel_generator(step['name']):
        # Parse step_input as a reference
        ref_type, ref_val = parse_reference(step_input, step_names=step_names, current_obs_name=current_obs_name)
        input_code = ref_to_code(ref_type, ref_val, state_idx=state_idx)
        if ref_type == 'literal' and isinstance(ref_val, str):
            input_code = f"_outputs['{ref_val}']"
        keyword.insert(0, input_code)  # As positional first arg

    call_args = ', '.join(keyword)
    return call_args, obs_deps

# =============================================================================
# Parse All Observations
# =============================================================================
obs_list = []
callable_imports = {}  # {module: set(qualnames)}

for obs_name, obs in observations.items():
    info = {
        'name': obs_name,
        'label': get_attr(obs, 'label', ''),
        'description': get_attr(obs, 'description', ''),
        'source': None,
        'source_observation': None,
        'pipeline': [],
    }

    # Source state variable
    src = get_attr(obs, 'source')
    if src:
        info['source'] = get_attr(src, 'name', str(src)) if hasattr(src, 'name') else str(src)

    # Source observation
    src_obs = get_attr(obs, 'source_observation')
    if src_obs:
        info['source_observation'] = get_attr(src_obs, 'name', str(src_obs)) if hasattr(src_obs, 'name') else str(src_obs)

    # Parse pipeline - handle FunctionCall objects
    for func_call in (get_attr(obs, 'pipeline') or []):
        # FunctionCall can have:
        # 1. function: reference to a defined Function
        # 2. callable: inline callable specification (no function reference needed)

        func_ref = get_attr(func_call, 'function')
        inline_callable = get_attr(func_call, 'callable')

        if func_ref:
            # Referenced function - get name from function reference
            step_name = str(func_ref) if isinstance(func_ref, str) else get_attr(func_ref, 'name', str(func_ref))
            # Look up the actual Function definition
            func_def = functions_by_name.get(step_name)
            # Parse using Function definition with FunctionCall overrides
            step = parse_step(func_def or func_call, step_name)
            # Override output from FunctionCall if specified (func_call's output takes precedence)
            fc_output = get_attr(func_call, 'output')
            if fc_output:
                step['output'] = fc_output
        elif inline_callable:
            # Inline callable - use output or callable.name as step identifier
            cname = get_attr(inline_callable, 'name')
            step_name = get_attr(func_call, 'output') or cname or 'callable_step'
            # Parse the FunctionCall directly (it has the callable)
            step = parse_step(func_call, step_name)
        else:
            # Fallback
            step_name = get_attr(func_call, 'name', 'step')
            step = parse_step(func_call, step_name)

        # Override arguments from FunctionCall if provided
        for arg in (get_attr(func_call, 'arguments') or []):
            name = get_attr(arg, 'name')
            val = get_attr(arg, 'value')
            if name and val is not None:
                step['arguments'][str(name)] = val
                if str(name) not in step['arg_names']:
                    step['arg_names'].append(str(name))

        info['pipeline'].append(step)

        # Collect callable imports (use module for import)
        c = step['callable']
        if c and c.get('module'):
            # Import the top-level module
            callable_imports.setdefault(c['module'], set())

    obs_list.append(info)

# Also collect imports from functions section
for fname, fdef in functions_by_name.items():
    c = get_attr(fdef, 'callable')
    if c:
        module = get_attr(c, 'module')
        if module:
            callable_imports.setdefault(module, set())

# Determine unique top-level modules to import
top_level_modules = set()
for module in callable_imports.keys():
    top_level_modules.add(module.split('.')[0])
%>
"""
Observation Functions for tvboptim
==================================

Auto-generated from TVBO Observation definitions.

Observations: ${len(obs_list)}
% for obs in obs_list:
- ${obs['name']}: ${obs['label'] or obs['description'][:50] + '...' if obs['description'] and len(obs['description']) > 50 else obs['description'] or '(no description)'}
% endfor
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any
from tvboptim.experimental.network_dynamics.core.bunch import Bunch

# Callable module imports
% for module in sorted(top_level_modules):
% if module not in ('jax', 'numpy', 'np', 'jnp'):
import ${module}
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

    # Analyze pipeline for data requirements
    needs_transient, needs_result_from_pipeline = analyze_pipeline(pipeline)

    # Determine state index from source
    state_idx = state_names.index(obs_source) if obs_source and obs_source in state_names else None

    # Determine simulation needs
    needs_simulation = bool(obs_source) or needs_result_from_pipeline
    if not obs_src_obs and not obs_source and pipeline:
        # Check if any step needs implicit simulation input
        for step in pipeline:
            has_data_ref = any(
                parse_reference(v)[0] in ('observation', 'input', 'integration')
                for v in step.get('arguments', {}).values()
            )
            if not has_data_ref and not step.get('input') and not is_kernel_generator(step['name']):
                needs_simulation = True
                break

    # Determine final output key (defaults to observation name if last step has no explicit output)
    if pipeline:
        last_step_output = pipeline[-1].get('output')
        final_key = last_step_output if last_step_output else obs_name
    else:
        final_key = 'data'
%>

def ${obs_name}(model_fn, state, dt: float = ${dt}, outputs: Bunch = None, **kwargs) -> Bunch:
    """${obs['label'] or obs_name}

    ${obs['description'] or 'Auto-generated observation function.'}
% if pipeline:

    Pipeline: ${' -> '.join([s['name'] for s in pipeline])}
% endif

    Returns
    -------
    Bunch
        All pipeline outputs. Primary output: '${final_key}'
    """
    _outputs = Bunch() if outputs is None else outputs

% if obs_src_obs:
    # Source: ${obs_src_obs} observation
    _source = ${obs_src_obs}(model_fn, state, dt=dt, outputs=_outputs, **kwargs)
    _outputs.update(_source)
    _outputs['data'] = _source.get(_source._primary_key, _source.get('data'))
    if 'time' in _source:
        _outputs['time'] = _source['time']
% elif obs_source:
    # Source: ${obs_source} state variable (index ${state_idx})
    _result = kwargs.get('result') or model_fn(state)
    _outputs['data'] = _result.data[:, ${state_idx}, :]  # Shape: (time, nodes)
    _outputs['time'] = _result.time  # Time axis from simulation
% elif needs_simulation:
    # Get simulation result
    _result = kwargs.get('result') or model_fn(state)
    _outputs['data'] = _result.data
    _outputs['time'] = _result.time  # Time axis from simulation
% endif
% if needs_transient:
    # Integration transient data
    _result_transient = kwargs.get('result_transient')
% endif
% if pipeline:

    # Pipeline execution
<%
    # Collect all step names AND output names in this pipeline for reference resolution
    # References can use either the function name or the output key
    step_names = set()
    for s in pipeline:
        step_names.add(s['name'])
        if s.get('output'):
            # Also track individual outputs (for tuple unpacking like 'frequencies, psd')
            for out in s['output'].split(','):
                step_names.add(out.strip())
%>
% for step_idx, step in enumerate(pipeline):
<%
    step_name = step['name']
    is_last = step_idx == len(pipeline) - 1
    is_first_step = step_idx == 0
    # Output defaults: last step -> observation name, otherwise -> function name
    if is_last:
        step_output = step['output'] or obs_name
    else:
        step_output = step['output'] or step_name
    # Input defaults to previous step's output, or 'data' for first step
    prev_output = pipeline[step_idx - 1]['output'] or pipeline[step_idx - 1]['name'] if step_idx > 0 else 'data'
    step_input = step['input'] or prev_output

    call_args, obs_deps = build_step_call(step, step_input, step_names=step_names, current_obs_name=obs_name, state_idx=state_idx, is_first_step=is_first_step)

    outputs = [o.strip() for o in step_output.split(',')]
    output_lhs = ', '.join([f"_outputs['{o}']" for o in outputs])

    fn_is_global = step_name in functions_by_name
    step_callable = step['callable']
    apply_dim = step.get('apply_on_dimension')
%>
    # ${step_name}: ${step_input} -> ${step_output}
% for dep in sorted(obs_deps):
    _${dep}_result = _${dep}_result if '_${dep}_result' in dir() else ${dep}(model_fn, state, dt=dt, **kwargs)
% endfor
% if fn_is_global:
    ${output_lhs} = ${step_name}(${call_args})
% elif step_callable and step_callable.get('full_call'):
% if apply_dim == 'node':
    # apply_on_dimension: node - vmap over axis=1 (nodes)
    ${output_lhs} = ${build_vmap_call(step_callable['full_call'], step, step_names, obs_name, state_idx)}
% else:
    ${output_lhs} = ${step_callable['full_call']}(${call_args})
% endif
% elif step['source_code']:
    ${output_lhs} = (lambda _in: ${step['source_code'].replace('_input', '_in')})(_outputs['${step_input}'])
% elif step['equation']:
<%
    params = step['arg_names'] or list(step['arguments'].keys())
%>
    ${output_lhs} = ${jaxcode(step['equation'], params)}
% else:
    raise NotImplementedError("Step '${step_name}' has no implementation")
% endif
% endfor
% endif

    # Mark primary output for convenience
    _outputs._primary_key = '${final_key}'
    return _outputs

% endfor
