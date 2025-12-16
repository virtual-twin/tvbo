<%!
from tvbo.export.code import render_expression
%>
<%def name="generate_function(func, func_name)" filter="trim">
<%
    # Collect parameters
    params = {}
    if func.arguments:
        for arg in func.arguments:
            # Skip pipeline references (in1, in2, etc.) - these are variable names not parameters
            if arg.name not in ['in1', 'in2']:
                # Quote string values
                value = arg.value
                if isinstance(value, str) and not value.replace('.','').replace('-','').isdigit():
                    value = f"'{value}'"
                params[arg.name] = value
    if func.equation and func.equation.parameters:
        for name, param in func.equation.parameters.items():
            params[name] = param.value
    param_args = ', '.join([f"{name}={value}" for name, value in params.items()])

    # Store callable reference to avoid name collision
    callable_ref = f"_jax_{func.callable.name}" if func.callable else None

    # For callables: determine which arguments are TimeSeries (need .data extraction)
    # vs regular parameters (use as-is)
    callable_ts_args = []  # Arguments that are TimeSeries
    if func.callable and func.arguments:
        for arg in func.arguments:
            # in1, in2, etc. are TimeSeries inputs from pipeline
            if arg.name in ['in1', 'in2', 'in3']:
                callable_ts_args.append(arg.name)
%>
% if func.callable:
<%
    # Build function signature: primary input + secondary TimeSeries inputs + parameters
    sig_parts = ['signal_ts']  # First argument is always the primary signal
    sig_parts.extend(callable_ts_args)
    if param_args:
        sig_parts.append(param_args)
    func_signature = ', '.join(sig_parts)

    # Build kwargs string for lambda
    call_kwargs = ', '.join([f'{k}={k}' for k in params.keys()]) if params else ''
%>
def ${func_name}(${func_signature}):
% if callable_ts_args:
    # Secondary inputs (e.g., kernels) - extract 1D data
    kernel_data = ${callable_ts_args[0]}.data.squeeze()  # Assume 1D kernel
    # Apply callable with vmap over spatial dimensions only (nodes, state_vars, samples)
    # The kernel stays 1D and is broadcast
    _apply_1d = lambda x: ${callable_ref}(x, kernel_data${', ' + call_kwargs if call_kwargs else ''})
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
% elif func.time_range:
<%
    # Kernel generator: creates TimeSeries over specified time range
    lo = func.time_range.lo if func.time_range.lo else 0
    hi = func.time_range.hi
    # Translate language-independent 'input.' to Python-specific 'ts.'
    step = func.time_range.step.replace('input.', 'ts.') if func.time_range.step else 'ts.sample_period'

    # Collect expected time unit from equation parameters
    expected_time_unit = None
    if func.equation and hasattr(func.equation, 'parameters'):
        for param_name, param in func.equation.parameters.items():
            if hasattr(param, 'unit') and param.unit in ['s', 'ms', 'us', 'ns']:
                expected_time_unit = param.unit
                break
    # Also check arguments
    if not expected_time_unit and func.arguments:
        for arg in func.arguments:
            if hasattr(arg, 'unit') and arg.unit in ['s', 'ms', 'us', 'ns']:
                expected_time_unit = arg.unit
                break
%>
def ${func_name}(ts, ${param_args}):
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
    # Check if equation contains array slicing (e.g., X[::stepsize])
    rhs = func.equation.rhs
    if '[' in rhs and ']' in rhs:
        # Direct Python code - use as is
        jax_code = rhs
    else:
        # Symbolic expression - render it
        jax_code = render_expression(rhs, format='jax')

    # Check if transformation applies on time dimension (handle both string and enum)
    apply_on_dim = str(func.apply_on_dimension) if func.apply_on_dimension else None
    has_apply_on_time = apply_on_dim in ['time', 'DimensionType.time']
%>
def ${func_name}(ts, ${param_args}):
    data = ${jax_code.replace('X', 'ts.data')}
% if has_apply_on_time:
    time = ${jax_code.replace('X', 'ts.time')}
    return ts.duplicate(time=time, data=data, title='${func_name}')
% else:
    return ts.duplicate(data=data, title='${func_name}')
% endif
% endif
</%def>
