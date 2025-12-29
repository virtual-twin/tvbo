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

All monitors (BOLD, EEG, MEG, etc.) are handled via pipeline functions with callables.
No special-case handling - everything is declarative via function definitions.

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

# Collect user-defined functions from experiment.functions
# These are functions defined in YAML (e.g., correlation, cauchy_pdf) that need to be
# recognized by the code printer. Map function name -> function name (identity mapping)
# so the printer emits them as-is rather than raising PrintMethodNotImplementedError.
_exp_functions = getattr(experiment, 'functions', None) or {}
if hasattr(_exp_functions, 'items'):
    user_functions = {str(fname): str(fname) for fname in _exp_functions.keys()}
elif hasattr(_exp_functions, '__iter__'):
    user_functions = {str(getattr(f, 'name', f)): str(getattr(f, 'name', f)) for f in _exp_functions}
else:
    user_functions = {}

# JAX code generation helper (array function mappings are built into the printer)
# Pass user_functions so custom functions (correlation, cauchy_pdf, etc.) are recognized
# Parameters argument allows user-defined names (like 'gamma') to override SymPy builtins
jaxcode = lambda expr, parameters=None: render_expression(expr, format='jax', user_functions=user_functions, parameters=parameters)

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

def parse_arg_value(val):
    """Parse argument value and classify as reference or literal.

    Returns tuple: (type, value) where type is:
    - 'observation': value is (obs_name, key) - reference to another observation's output
    - 'input': value is key - reference to pipeline local _outputs dict
    - 'integration': value is key - reference to integration result (transient, result)
    - 'literal': value is the literal value
    """
    if isinstance(val, str):
        # Check for prefix.key reference syntax
        if '.' in val and not val.replace('.', '').replace('-', '').replace('_', '').isdigit():
            parts = val.split('.', 1)
            prefix, key = parts[0], parts[1]
            # Check if it's 'input.key' pattern (pipeline local reference)
            if prefix == 'input':
                return ('input', key)
            # Check if it's 'integration.key' pattern (simulation result reference)
            # integration.transient -> transient simulation data
            # integration.result -> main simulation data
            if prefix == 'integration':
                return ('integration', key)
            # Check if it's a known observation name
            if prefix in observations:
                return ('observation', (prefix, key))
        # Try to convert to numeric
        try:
            if '.' in val:
                return ('literal', float(val))
            else:
                return ('literal', int(val))
        except ValueError:
            pass
    return ('literal', val)

# Build a lookup dict for experiment.functions by name
# This allows pipeline steps to reference functions by name without inline definitions
exp_functions_by_name = {}
if hasattr(_exp_functions, 'items'):
    for fname, fdef in _exp_functions.items():
        exp_functions_by_name[str(fname)] = fdef
elif hasattr(_exp_functions, '__iter__'):
    for fdef in _exp_functions:
        name = getattr(fdef, 'name', None)
        if name:
            exp_functions_by_name[str(name)] = fdef

def lookup_function_def(step_name):
    """Look up function definition from experiment.functions by name."""
    return exp_functions_by_name.get(step_name)

# Get observations
observations = getattr(experiment, 'observations', None) or {}
if hasattr(observations, 'values'):
    observations = dict(observations.items()) if hasattr(observations, 'items') else {}
elif hasattr(observations, '__iter__') and not isinstance(observations, dict):
    observations = {getattr(o, 'name', f'obs_{i}'): o for i, o in enumerate(observations)}

# Collect all unique imports from callables across all observations AND experiment.functions
callable_imports = {}  # {module: set(qualnames)}

# First collect from experiment.functions
for fname, fdef in exp_functions_by_name.items():
    callable_def = getattr(fdef, 'callable', None)
    if callable_def:
        module = getattr(callable_def, 'module', None)
        qualname = getattr(callable_def, 'qualname', None)
        if module and qualname:
            if module not in callable_imports:
                callable_imports[module] = set()
            callable_imports[module].add(qualname)

# Then collect from inline observation pipeline callables
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
        step_name = getattr(func, 'name', 'step')
        func_info = {
            'name': step_name,
            'output': getattr(func, 'output', None),
            'input': None,
            'callable': None,
            'equation': None,
            'source_code': None,
            'arguments': {},
            'arg_names': [],  # All argument names (even without values) for equation parsing
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

        # Arguments from inline step - store raw values for reference resolution later
        args = getattr(func, 'arguments', None) or []
        if hasattr(args, '__iter__'):
            for arg in args:
                arg_name = getattr(arg, 'name', None)
                arg_value = getattr(arg, 'value', None)
                if arg_name:
                    # Track all argument names for equation parsing
                    if arg_name not in func_info['arg_names']:
                        func_info['arg_names'].append(arg_name)
                    # Store raw value (will be parsed for references in code generation)
                    if arg_value is not None:
                        func_info['arguments'][arg_name] = arg_value  # Keep raw for reference parsing

        # === LOOKUP from experiment.functions if no inline definition ===
        # If step has no source_code/callable/equation, look up by name in experiment.functions
        has_inline_def = bool(func_info['source_code'] or func_info['callable'] or func_info['equation'])
        if not has_inline_def:
            fn_def = lookup_function_def(step_name)
            if fn_def:
                # Get source_code from function definition
                fn_source = getattr(fn_def, 'source_code', None)
                if fn_source:
                    func_info['source_code'] = str(fn_source)

                # Get callable from function definition
                fn_callable = getattr(fn_def, 'callable', None)
                if fn_callable:
                    func_info['callable'] = {
                        'name': getattr(fn_callable, 'name', None),
                        'module': getattr(fn_callable, 'module', None),
                        'qualname': getattr(fn_callable, 'qualname', None),
                    }

                # Get equation from function definition
                fn_eq = getattr(fn_def, 'equation', None)
                if fn_eq:
                    func_info['equation'] = getattr(fn_eq, 'rhs', None)

                # Merge arguments from function definition (step args take precedence)
                fn_args = getattr(fn_def, 'arguments', None) or []
                if hasattr(fn_args, '__iter__'):
                    for arg in fn_args:
                        arg_name = getattr(arg, 'name', None)
                        arg_value = getattr(arg, 'value', None)
                        if arg_name:
                            # Track all argument names for equation parsing
                            if arg_name not in func_info['arg_names']:
                                func_info['arg_names'].append(arg_name)
                            # Only add to 'arguments' dict if has a value (for code generation)
                            if arg_value is not None and arg_name not in func_info['arguments']:
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
    ${i + 1}. ${step['name']}: ${'callable=' + step['callable']['module'] + '.' + step['callable']['name'] if step['callable'] and step['callable']['module'] else 'equation=' + (step['equation'][:40] + '...' if step['equation'] and len(step['equation']) > 40 else step['equation'] or 'N/A')}
% endfor
% endif
    """
    # Initialize outputs dictionary
    _outputs = {}

<%
    # Determine if this observation needs simulation data or is purely derived
    # An observation is purely derived if:
    # 1. No source state variable
    # 2. No source_observation
    # 3. Has a pipeline AND all pipeline data comes from observation references

    needs_simulation = False
    needs_transient = False  # Track if integration.transient is referenced

    if obs_src_obs:
        needs_simulation = False  # Uses another observation's output
    elif obs_source:
        needs_simulation = True  # Uses a state variable directly
    elif has_pipeline:
        # Check if any pipeline step has NO data references (i.e., uses implicit input from simulation)
        for step in pipeline:
            step_has_data_ref = False
            for aname, aval in step.get('arguments', {}).items():
                ref_type, ref_val = parse_arg_value(aval)
                if ref_type in ('observation', 'input'):
                    step_has_data_ref = True
                elif ref_type == 'integration':
                    # Track integration references
                    if ref_val == 'transient':
                        needs_transient = True
                    elif ref_val == 'result':
                        needs_simulation = True
            # Check if step is a kernel generator (time_range function) - doesn't need simulation
            step_fn_def = exp_functions_by_name.get(step['name'])
            step_is_kernel_gen = step_fn_def and getattr(step_fn_def, 'time_range', None)
            if not step_has_data_ref and not step.get('input') and not step_is_kernel_gen:
                # Step has no data references, no explicit input, and is not a kernel generator -> needs simulation
                needs_simulation = True
    else:
        # No pipeline, no source - assume needs simulation
        needs_simulation = True
%>
% if obs_src_obs:
    # Derived observation - get data from source observation
    _source_result = ${obs_src_obs}(model_fn, state, dt=dt, **kwargs)
    if isinstance(_source_result, dict):
        _outputs.update(_source_result)
    else:
        _outputs['data'] = _source_result
% elif obs_source:
    # Root observation - use pre-computed result if available, otherwise run simulation
    _result = kwargs.get('result', None)
    if _result is None:
        _result = model_fn(state)
    _outputs['data'] = _result.data[:, ${state_names.index(obs_source) if obs_source in state_names else 0}, :]
% elif needs_simulation:
    # Generic observation - use pre-computed result if available, otherwise run simulation
    _result = kwargs.get('result', None)
    if _result is None:
        _result = model_fn(state)
    _outputs['data'] = _result.data
% else:
    # Purely derived observation - all data comes from other observations via references
    pass
% endif
% if needs_transient:
    # Get transient simulation data (integration.transient reference)
    _result_transient = kwargs.get('result_transient', None)
% endif
% if has_pipeline:

    # Execute pipeline
% for step_idx, step in enumerate(pipeline):
<%
    step_name = step['name']
    # Last step defaults to observation name, intermediate steps default to 'data'
    is_last_step = (step_idx == len(pipeline) - 1)
    step_output = step['output'] or (obs_name if is_last_step else 'data')
    step_input = step['input'] or 'data'
    step_callable = step['callable']
    step_equation = step['equation']
    step_source_code = step['source_code']
    step_args = step['arguments']  # Arguments with values (for kwargs)
    step_arg_names = step.get('arg_names', [])  # All argument names

    # Check if this function is defined globally in experiment.functions
    fn_is_global = step_name in exp_functions_by_name

    # Get the global function definition if available
    fn_def = exp_functions_by_name.get(step_name)
    fn_def_args = []
    if fn_def:
        fn_args_raw = getattr(fn_def, 'arguments', None) or []
        if hasattr(fn_args_raw, '__iter__'):
            for a in fn_args_raw:
                aname = getattr(a, 'name', None)
                if aname:
                    fn_def_args.append(str(aname))

    # Parse argument values for references
    # Build list of (arg_name, code_expr) where code_expr is Python code to get the value
    arg_code_parts = []
    obs_calls_needed = set()  # Observations that need to be called first
    has_data_reference = False  # Track if any arg is a data reference (observation/input)

    for aname, aval in step_args.items():
        ref_type, ref_val = parse_arg_value(aval)
        if ref_type == 'observation':
            obs_name, key = ref_val
            obs_calls_needed.add(obs_name)
            code_expr = f"_{obs_name}_result['{key}']"
            has_data_reference = True
        elif ref_type == 'input':
            code_expr = f"_outputs['{ref_val}']"
            has_data_reference = True
        elif ref_type == 'integration':
            # Map integration.transient -> _result_transient, integration.result -> _result
            if ref_val == 'transient':
                code_expr = "_result_transient.data"
            elif ref_val == 'result':
                code_expr = "_result.data"
            else:
                code_expr = f"_result_{ref_val}"
        else:
            code_expr = repr(ref_val)
        arg_code_parts.append((aname, code_expr))

    # Determine the first positional argument (the input data)
    # If first explicit arg is named data/x/a/arr/input/array, use it positionally
    # If no data references in args, use _outputs[step_input] as implicit first arg
    # If explicit data references exist, only use those (no implicit input)
    positional_args = []
    kwarg_parts = []

    for i, (aname, code) in enumerate(arg_code_parts):
        if i == 0 and aname.lower() in ('data', 'x', 'a', 'arr', 'input', 'array'):
            # First arg is explicitly the data - use positionally
            positional_args.append(code)
        else:
            kwarg_parts.append(f"{aname}={code}")

    # Check if function has time_range (kernel generator - generates data, no input needed)
    fn_has_time_range = fn_def and getattr(fn_def, 'time_range', None)

    # Add implicit pipeline input if there are no data references
    # (i.e., args are only default values like step=10, fs=100.0)
    # BUT skip for kernel generators (time_range functions) that don't take data input
    if not has_data_reference and not fn_has_time_range:
        positional_args.insert(0, f"_outputs['{step_input}']")

    # Build full args string
    all_args = positional_args + kwarg_parts
    args_str = ', '.join(all_args)

    # Check if output contains comma (multiple outputs)
    outputs = [o.strip() for o in step_output.split(',')]
    is_multi_output = len(outputs) > 1
    output_lhs = ', '.join([f"_outputs['{o}']" for o in outputs])
%>
    # Step: ${step_name} (${step_input} -> ${step_output})
% for obs_call in sorted(obs_calls_needed):
    if '_${obs_call}_result' not in dir():
        _${obs_call}_result = ${obs_call}(model_fn, state, dt=dt, **kwargs)
% endfor
% if fn_is_global:
    ${output_lhs} = ${step_name}(${args_str})
% elif step_callable and step_callable.get('module') and step_callable.get('qualname'):
<%
    fn_ref = step_callable['name']
%>
% if is_multi_output:
    ${output_lhs} = ${fn_ref}(${args_str})
% else:
    _outputs['${step_output}'] = ${fn_ref}(${args_str})
% endif
% elif step_source_code:
    # Inline source_code (should be rare - prefer defining functions globally)
    _outputs['${step_output}'] = (lambda _in: ${step_source_code.replace('_input', '_in') if '_input' in step_source_code else step_source_code})(_outputs['${step_input}'])
% elif step_equation:
<%
    step_param_names = step_arg_names if step_arg_names else list(step_args.keys()) if step_args else []
    rendered_eq = jaxcode(step_equation, parameters=step_param_names)
%>
    # Inline equation (should be rare - prefer defining functions globally)
    _outputs['${step_output}'] = ${rendered_eq}
% else:
    raise NotImplementedError("Pipeline step '${step_name}' has no implementation - define it in functions section")
% endif
% endfor
% endif

    # Return: if only output is the observation name itself, return value directly
    # Otherwise return full dict
    if list(_outputs.keys()) == ['${obs_name}']:
        return _outputs['${obs_name}']
    return _outputs

% endfor
