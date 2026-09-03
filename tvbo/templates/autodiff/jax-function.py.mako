<%doc>
JAX Function Template (Pipeline-style)
======================================

Generates JAX functions for TimeSeries processing pipelines.
Uses the base function-def.mako for core function generation.

This template provides the `generate_function` def for pipeline transformations
that operate on TimeSeries objects (with .data, .time attributes).

For standalone mathematical functions, use base/function-def.mako directly.
</%doc>
<%namespace name="fn" file="/base/function-def.mako"/>
<%!
from tvbo.codegen import render_expression
from tvbo.templates.base.utils import get_source_code, retime, time_series_inputs
%>
<%def name="generate_function(func, func_name, supplied=())" filter="trim">
<%
    # Collect parameters - split into required (no default) and optional (with default)
    params_required = []  # argument names with no default value
    params = {}           # argument name -> default value
    if func.arguments:
        args = func.arguments.values() if hasattr(func.arguments, 'values') else func.arguments
        for arg in args:
            # Skip pipeline references (in1, in2, etc.) - these are variable names not parameters
            if arg.name not in ['in1', 'in2', 'in3']:
                value = arg.value if hasattr(arg, 'value') else None
                if value is not None:
                    if isinstance(value, str) and not value.replace('.','').replace('-','').isdigit():
                        value = f"'{value}'"
                    params[arg.name] = value
                else:
                    params_required.append(arg.name)
    if getattr(func, 'equation', None) and hasattr(func.equation, 'parameters') and func.equation.parameters:
        for name, param in func.equation.parameters.items():
            if hasattr(param, 'value') and param.value is not None:
                params[name] = param.value
    _kwargs = [f"{name}={value}" for name, value in params.items()]
    # A no-default argument is a parameter when the call site supplies it (`supplied`), and the pipeline input bound from the TimeSeries in the body when it does not.
    _in_signature = [name for name in params_required if name in supplied]
    _bound_in_body = [name for name in params_required if name not in supplied]
    param_args = ', '.join(_in_signature + _kwargs)
    signature_args = ', '.join(list(params_required) + _kwargs)

    # Store callable reference to avoid name collision
    callable_ref = f"_jax_{func.callable.name}" if func.callable else None

    # For callables: determine which arguments are TimeSeries (need .data extraction)
    # vs regular parameters (use as-is)
    callable_ts_args = []  # Arguments that are TimeSeries
    if func.callable and func.arguments:
        args = func.arguments.values() if hasattr(func.arguments, 'values') else func.arguments
        for arg in args:
            # in1, in2, etc. are TimeSeries inputs from pipeline
            if arg.name in ['in1', 'in2', 'in3']:
                callable_ts_args.append(arg.name)
%>
% if func.callable:
<%
    # Build function signature: primary input + secondary TimeSeries inputs + parameters
    sig_parts = ['signal_ts']  # First argument is always the primary signal
    sig_parts.extend(callable_ts_args)
    if signature_args:
        sig_parts.append(signature_args)
    func_signature = ', '.join(sig_parts)

    # Build kwargs string for lambda
    call_kwargs = ', '.join([f'{k}={k}' for k in params.keys()]) if params else ''
%>
def ${func_name}(${func_signature}):
% if hasattr(func, 'description') and func.description:
    """${func.description}"""
% endif
% if callable_ts_args:
    # Secondary inputs (e.g., kernels) - extract 1D data
    kernel_data = ${callable_ts_args[0]}.data.squeeze()  # Assume 1D kernel
    # Causal convolution: 'full' then keep the leading len(x) samples (matches
    # TVB's stock-buffer BOLD; 'same' would sample the kernel's decayed tail).
    _apply_1d = lambda x: ${callable_ref}(x, kernel_data, mode='full')[:x.shape[0]]
    _apply_nodes = lambda x: jax.vmap(_apply_1d, in_axes=1, out_axes=1)(x)
    _apply_svars = lambda x: jax.vmap(_apply_nodes, in_axes=1, out_axes=1)(x)
    data = jax.vmap(_apply_svars, in_axes=3, out_axes=3)(signal_ts.data)
% else:
    # No secondary inputs - apply directly with vmap over all dimensions
    _apply_1d = lambda x: ${callable_ref}(x${', ' + call_kwargs if call_kwargs else ''})
    _apply_nodes = lambda x: jax.vmap(_apply_1d, in_axes=1, out_axes=1)(x)
    _apply_svars = lambda x: jax.vmap(_apply_nodes, in_axes=1, out_axes=1)(x)
    data = jax.vmap(_apply_svars, in_axes=3, out_axes=3)(signal_ts.data)
% endif

    # Time axis: preserve if same length, otherwise reconstruct
    time = signal_ts.time if data.shape[0] == len(signal_ts.time) else jnp.arange(data.shape[0]) * signal_ts.sample_period
    return signal_ts.duplicate(time=time, data=data, title='${func_name}')
% elif hasattr(func, 'time_range') and func.time_range:
<%
    # Kernel generator: creates TimeSeries over specified time range
    lo = func.time_range.lo if func.time_range.lo else 0
    hi = func.time_range.hi
    # Translate language-independent 'input.' to Python-specific 'ts.'
    step = func.time_range.step.replace('input.', 'ts.') if func.time_range.step else 'ts.sample_period'

    # Collect expected time unit from equation parameters
    expected_time_unit = None
    if func.equation and hasattr(func.equation, 'parameters') and func.equation.parameters:
        for param_name, param in func.equation.parameters.items():
            if hasattr(param, 'unit') and str(param.unit) in ['s', 'ms', 'us']:
                expected_time_unit = str(param.unit)
                break
    # Also check arguments
    if not expected_time_unit and func.arguments:
        args = func.arguments.values() if hasattr(func.arguments, 'values') else func.arguments
        for arg in args:
            if hasattr(arg, 'unit') and str(arg.unit) in ['s', 'ms', 'us']:
                expected_time_unit = str(arg.unit)
                break
%>
def ${func_name}(ts, ${signature_args}):
% if hasattr(func, 'description') and func.description:
    """${func.description}"""
% endif
% if expected_time_unit:
    # Auto-convert time units if necessary
    if ts.units and ts.units.get('time') and ts.units.get('time') != '${expected_time_unit}':
        ts = ts.convert_units('time', '${expected_time_unit}')
% endif
    t = jnp.arange(${lo}, ${hi}, ${step})
    data = ${render_expression(func.equation.rhs, format='jax')}
    return ts.duplicate(time=t, data=data, title='${func_name}')
% else:
<%
    _eq = getattr(func, 'equation', None)
    rhs = _eq.rhs if _eq else ''
    # `source_code:` is already backend-ready text; only an `equation:` needs printing.
    _src = get_source_code(func)
    jax_code = render_expression(rhs, format='jax') if rhs else (_src or '0.0')
    # Check if transformation applies on time dimension (handle both string and enum)
    apply_on_dim = str(func.apply_on_dimension) if func.apply_on_dimension else None
    has_apply_on_time = apply_on_dim in ['time', 'DimensionType.time']
    # `X` or the spec's own unsupplied argument, whichever the body actually reads.
    inputs = time_series_inputs(['X'] + _bound_in_body, jax_code)
    time_code = retime(jax_code, inputs)
%>
def ${func_name}(ts, ${param_args}):
% if hasattr(func, 'description') and func.description:
    """${func.description}"""
% endif
% for name in inputs:
    ${name} = ts.data
% endfor
% if has_apply_on_time:
% for name in inputs:
    t_${name} = ts.time
% endfor
    data = ${jax_code}
    time = ${time_code}
    return ts.duplicate(time=time, data=data, title='${func_name}')
% else:
    data = ${jax_code}
    return ts.duplicate(data=data, title='${func_name}')
% endif
% endif
</%def>

## =============================================================================
## Standalone function generation (uses base template)
## =============================================================================
<%def name="standalone_function(func, format='jax', user_functions=None)">
${fn.function_def(func, format=format, user_functions=user_functions)}
</%def>

<%def name="loss_function(func, format='jax', user_functions=None, inner_func_names=None)">
${fn.function_def_with_aggregation(func, format=format, user_functions=user_functions, inner_func_names=inner_func_names)}
</%def>
