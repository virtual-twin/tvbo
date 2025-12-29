<%doc>
Base Function Definition Template
=================================

Provides reusable defs for generating Python function definitions from TVBO Function objects.
Supports all function types:

1. **Equation-based**: Symbolic expressions parsed with SymPy
2. **Callable-based**: References to external functions with vmap support
3. **Source-code**: Raw Python code (func.source_code or equation.pycode)
4. **Time-range**: Kernel/signal generators over time ranges
5. **Aggregation**: Loss functions with vmap + reduction (for LossFunction)

Usage in other templates:
    <%namespace name="fn" file="/base/function-def.mako"/>
    ${fn.function_def(func, format='jax')}
    ${fn.function_def_with_aggregation(loss_func, format='jax')}

Standalone usage (no model context):
    from tvbo.templates import lookup
    template = lookup.get_template("base/function-def.mako")
    code = template.get_def("function_def").render(func=my_func, format='jax')
</%doc>
<%!
from tvbo.export.code import render_expression, parse_eq

# =============================================================================
# Helper Functions (Python code block - available to all defs)
# =============================================================================

def get_func_args(func):
    """Extract argument names from a Function object.

    Handles both dict-style (func.arguments.values()) and list-style arguments.
    """
    if not func.arguments:
        return []
    args = func.arguments
    if hasattr(args, 'values'):
        args = args.values()
    return [arg.name if hasattr(arg, 'name') else str(arg) for arg in args]


def get_func_params_with_defaults(func):
    """Extract parameters with default values for function signature.

    Returns list of tuples: [(param_name, default_value), ...]
    Includes both argument defaults and equation parameters.
    """
    params = []

    # From arguments
    if func.arguments:
        args = func.arguments.values() if hasattr(func.arguments, 'values') else func.arguments
        for arg in args:
            if hasattr(arg, 'value') and arg.value is not None:
                # Skip pipeline inputs (in1, in2, etc.)
                if arg.name not in ['in1', 'in2', 'in3', 'x', 'y']:
                    value = arg.value
                    # Quote string values that aren't numbers
                    if isinstance(value, str) and not value.replace('.','').replace('-','').replace('e','').isdigit():
                        value = f"'{value}'"
                    params.append((arg.name, value))

    # From equation parameters
    if func.equation and hasattr(func.equation, 'parameters') and func.equation.parameters:
        for name, param in func.equation.parameters.items():
            if hasattr(param, 'value') and param.value is not None:
                params.append((name, param.value))

    return params


def get_equation_rhs(func):
    """Get the equation RHS string from a Function object."""
    if func.equation:
        if hasattr(func.equation, 'rhs') and func.equation.rhs:
            return func.equation.rhs
        # Direct string equation
        if isinstance(func.equation, str):
            return func.equation
    return None


def get_source_code(func):
    """Get raw source code if available."""
    if hasattr(func, 'source_code') and func.source_code:
        return func.source_code
    if func.equation and hasattr(func.equation, 'pycode') and func.equation.pycode:
        return func.equation.pycode
    return None


def get_enum_text(enum_val):
    """Extract text value from enum (handles both PermissibleValue and string)."""
    if enum_val is None:
        return None
    if hasattr(enum_val, 'text'):
        return enum_val.text
    return str(enum_val).split('.')[-1]  # Handle 'DimensionType.node' -> 'node'


def render_body(func, format='jax', user_functions=None, render_func=None):
    """Render the function body expression.

    Parameters
    ----------
    func : Function
        The function object
    format : str
        Output format ('jax', 'numpy', 'python')
    user_functions : dict, optional
        Custom function mappings for the printer
    render_func : callable, optional
        Custom render function (e.g., model.render_equation)
    """
    # Use custom render function if provided (for model context)
    if render_func is not None:
        return render_func(func)

    # Get equation RHS
    rhs = get_equation_rhs(func)
    if rhs is None:
        raise ValueError(f"Function {func.name} has no equation.rhs")

    # Check for raw Python code patterns (array slicing, etc.)
    if '[' in rhs and ('::' in rhs or ':' in rhs):
        # Likely raw Python slicing - use as-is
        return rhs

    # Parse and render symbolic expression
    return render_expression(rhs, format=format, user_functions=user_functions or {})
%>

## =============================================================================
## Main Function Definition Defs
## =============================================================================

<%def name="function_signature(func, extra_args=None, params_with_defaults=None)">
<%
    """Generate function signature: def name(args, params=defaults)"""
    args = get_func_args(func)
    if extra_args:
        args = list(extra_args) + args

    params = params_with_defaults or get_func_params_with_defaults(func)
    param_strs = [f"{name}={value}" for name, value in params]

    all_args = args + param_strs
    signature = ', '.join(all_args)
%>\
def ${func.name}(${signature}):\
</%def>

<%def name="function_def(func, format='jax', user_functions=None, render_func=None, docstring=None)">
<%
    """Generate a complete function definition.

    Handles equation-based, source-code, and time-range functions.
    For callable-based or aggregation, use specialized defs.
    """
    # Determine function type and generate body
    source = get_source_code(func)
    rhs = get_equation_rhs(func)

    # Build docstring
    doc = docstring or (func.description if hasattr(func, 'description') and func.description else None)

    args = get_func_args(func)
    params = get_func_params_with_defaults(func)
    param_strs = [f"{name}={value}" for name, value in params]
    all_args = args + param_strs
    signature = ', '.join(all_args)

    # Compute body inline (avoid calling render_body which confuses Mako context)
    body = None
    if rhs and not source and not (hasattr(func, 'time_range') and func.time_range):
        if render_func is not None:
            body = render_func(func)
        elif '[' in rhs and ('::' in rhs or ':' in rhs):
            # Likely raw Python slicing - use as-is
            body = rhs
        else:
            # Parse and render symbolic expression
            body = render_expression(rhs, format=format, user_functions=user_functions or {})
%>\
% if source:
## Source code function - use raw code
def ${func.name}(${signature}):
% if doc:
    """${doc}"""
% endif
${source | trim}
% elif hasattr(func, 'time_range') and func.time_range:
## Time-range kernel generator
${time_range_function(func, format=format)}
% elif rhs:
## Equation-based function
def ${func.name}(${signature}):
% if doc:
    """${doc}"""
% endif
    return ${body}
% else:
## Fallback - empty function
def ${func.name}(${signature}):
% if doc:
    """${doc}"""
% endif
    pass
% endif
</%def>

<%def name="time_range_function(func, format='jax')">
<%
    """Generate a time-range based kernel/signal generator function."""
    lo = func.time_range.lo if func.time_range.lo else 0
    hi = func.time_range.hi
    step = func.time_range.step.replace('input.', 'ts.') if func.time_range.step else 'ts.sample_period'

    params = get_func_params_with_defaults(func)
    param_strs = [f"{name}={value}" for name, value in params]
    signature = ', '.join(['ts'] + param_strs)

    # Render the equation body
    rhs = get_equation_rhs(func)
    body = render_expression(rhs, format=format) if rhs else '0.0'

    # Check for expected time unit
    expected_time_unit = None
    if func.equation and hasattr(func.equation, 'parameters') and func.equation.parameters:
        for param_name, param in func.equation.parameters.items():
            if hasattr(param, 'unit') and param.unit in ['s', 'ms', 'us', 'ns']:
                expected_time_unit = param.unit
                break
    if not expected_time_unit and func.arguments:
        args = func.arguments.values() if hasattr(func.arguments, 'values') else func.arguments
        for arg in args:
            if hasattr(arg, 'unit') and arg.unit in ['s', 'ms', 'us', 'ns']:
                expected_time_unit = arg.unit
                break

    np_module = 'jnp' if format == 'jax' else 'np'
%>\
def ${func.name}(${signature}):
% if hasattr(func, 'description') and func.description:
    """${func.description}"""
% endif
% if expected_time_unit:
    # Auto-convert time units if necessary
    if ts.units and ts.units.get('time') and ts.units.get('time') != '${expected_time_unit}':
        ts = ts.convert_units('time', '${expected_time_unit}')
% endif
    t = ${np_module}.arange(${lo}, ${hi}, ${step})
    data = ${body}
    return ts.duplicate(time=t, data=data, title='${func.name}')
</%def>

<%def name="callable_function(func, format='jax', callable_ref=None)">
<%
    """Generate a callable-based function with vmap support.

    Handles external function calls with proper array dimension mapping.
    """
    if not func.callable:
        raise ValueError(f"Function {func.name} has no callable")

    ref = callable_ref or f"_jax_{func.callable.name}"

    # Collect parameters
    params = get_func_params_with_defaults(func)
    param_strs = [f"{name}={value}" for name, value in params]

    # Determine TimeSeries inputs (in1, in2, etc.)
    ts_args = []
    if func.arguments:
        args = func.arguments.values() if hasattr(func.arguments, 'values') else func.arguments
        for arg in args:
            if arg.name in ['in1', 'in2', 'in3']:
                ts_args.append(arg.name)

    # Build signature
    sig_parts = ['signal_ts']
    sig_parts.extend(ts_args)
    if param_strs:
        sig_parts.extend(param_strs)
    signature = ', '.join(sig_parts)

    # Build kwargs for lambda
    call_kwargs = ', '.join([f'{name}={name}' for name, _ in params]) if params else ''

    np_module = 'jnp' if format == 'jax' else 'np'
%>\
def ${func.name}(${signature}):
% if hasattr(func, 'description') and func.description:
    """${func.description}"""
% endif
% if ts_args:
    # Secondary inputs (e.g., kernels) - extract 1D data
    kernel_data = ${ts_args[0]}.data.squeeze()  # Assume 1D kernel
    # Apply callable with vmap over spatial dimensions only (nodes, state_vars, samples)
    _apply_1d = lambda x: ${ref}(x, kernel_data${', ' + call_kwargs if call_kwargs else ''})
    _apply_nodes = lambda x: jax.vmap(_apply_1d, in_axes=1, out_axes=1)(x)
    _apply_svars = lambda x: jax.vmap(_apply_nodes, in_axes=1, out_axes=1)(x)
    data = jax.vmap(_apply_svars, in_axes=3, out_axes=3)(signal_ts.data)
% else:
    # Apply callable with vmap over all dimensions
    _apply_1d = lambda x: ${ref}(x${', ' + call_kwargs if call_kwargs else ''})
    _apply_nodes = lambda x: jax.vmap(_apply_1d, in_axes=1, out_axes=1)(x)
    _apply_svars = lambda x: jax.vmap(_apply_nodes, in_axes=1, out_axes=1)(x)
    data = jax.vmap(_apply_svars, in_axes=3, out_axes=3)(signal_ts.data)
% endif

    # Time axis: preserve if same length, otherwise reconstruct
    time = signal_ts.time if data.shape[0] == len(signal_ts.time) else ${np_module}.arange(data.shape[0]) * signal_ts.sample_period
    return signal_ts.duplicate(time=time, data=data, title='${func.name}')
</%def>

## =============================================================================
## Aggregation Support (for LossFunction)
## =============================================================================

<%def name="function_def_with_aggregation(func, format='jax', user_functions=None, inner_func_names=None)">
<%
    """Generate a loss function with aggregation (vmap + reduction).

    Supports:
    - SymPy indexed notation: Sum(f(x[i], y[i]), (i, 0, n-1))
    - Metadata aggregation: aggregate.over + aggregate.type

    Parameters
    ----------
    func : LossFunction
        The loss function with optional aggregate specification
    format : str
        Output format ('jax', 'numpy')
    user_functions : dict, optional
        Custom function mappings (e.g., {'correlation': 'correlation'})
    inner_func_names : list, optional
        Names of inner functions that should be recognized (not implicit multiplication)
    """
    # Get aggregation spec
    has_aggregate = hasattr(func, 'aggregate') and func.aggregate is not None

    if has_aggregate:
        agg_over = get_enum_text(func.aggregate.over)
        agg_type = get_enum_text(func.aggregate.type)
    else:
        agg_over = None
        agg_type = 'none'

    # Build user_functions dict including inner functions
    uf = dict(user_functions or {})
    if inner_func_names:
        for name in inner_func_names:
            if name not in uf:
                uf[name] = name

    # Render the per-element loss expression
    rhs = get_equation_rhs(func)
    if rhs is None:
        raise ValueError(f"LossFunction {func.name} has no equation.rhs")

    # Parse to check for indexed aggregation and extract free symbols
    from tvbo.parse.expression import parse_eq, Mean
    from sympy import Sum, Product, Symbol

    parsed = parse_eq(rhs, functions=list(uf.keys()))

    # Collect arguments - prefer explicit args, fallback to equation free symbols
    args = get_func_args(func)
    if not args:
        # Infer arguments from free symbols in the equation (excluding known functions)
        known_funcs = set(uf.keys()) | {'Sum', 'Mean', 'Product', 'sqrt', 'exp', 'log', 'abs', 'mean', 'std', 'var'}
        if hasattr(parsed, 'free_symbols'):
            inferred_args = sorted([str(s) for s in parsed.free_symbols if str(s) not in known_funcs])
            args = inferred_args

    params = get_func_params_with_defaults(func)
    param_strs = [f"{name}={value}" for name, value in params]
    all_args = args + param_strs
    signature = ', '.join(all_args) if all_args else '*args'

    # For vmap call, use args or *args depending on what we have
    inner_args = ', '.join(args) if args else '*args'
    uses_varargs = not args

    # Check if expression contains indexed aggregation (Sum, Mean, Product with limits)
    has_indexed_agg = any(
        isinstance(sub, (Sum, Product, Mean)) and hasattr(sub, 'limits') and sub.limits
        for sub in parsed.atoms() | {parsed}
    ) if hasattr(parsed, 'atoms') else False

    # Render body
    body = render_expression(parsed, format=format, user_functions=uf)

    np_module = 'jnp' if format == 'jax' else 'np'

    # Map dimension to axis
    dim_to_axis = {
        'time': 0,
        'node': 0,  # Assuming node is first dimension after vmap
        'region': 0,
        'state': 1,
        'mode': 2,
        'sample': -1,
    }

    # Reduction function mapping
    reduction_map = {
        'mean': f'{np_module}.mean',
        'sum': f'{np_module}.sum',
        'max': f'{np_module}.max',
        'min': f'{np_module}.min',
        'none': None,
    }
%>\
def ${func.name}(${signature}):
% if hasattr(func, 'description') and func.description:
    """${func.description}"""
% endif
% if has_indexed_agg:
    ## Expression already contains Sum/Mean with indices - printer handles it
    return ${body}
% elif has_aggregate and agg_type != 'none':
    ## Metadata-driven aggregation with vmap + reduction
% if uses_varargs:
    def _per_element_loss(*args):
        return ${body}

    # vmap over ${agg_over} dimension, then reduce with ${agg_type}
    per_element_losses = jax.vmap(_per_element_loss)(*args)
% else:
    def _per_element_loss(${inner_args}):
        return ${body}

    # vmap over ${agg_over} dimension, then reduce with ${agg_type}
    per_element_losses = jax.vmap(_per_element_loss)(${inner_args})
% endif
    return ${reduction_map[agg_type]}(per_element_losses)
% else:
    ## No aggregation - direct evaluation
    return ${body}
% endif
</%def>

## =============================================================================
## Indexed Mathematical Functions (Sum, Mean, Product)
## =============================================================================

<%def name="indexed_function(func, format='jax', user_functions=None)">
<%
    """Generate a function with proper indexed aggregation handling.

    This def properly handles mathematical notation like:
    - Sum(x[i] * y[i], (i, 0, n-1)) -> jnp.sum(x * y)
    - Mean((x[i] - mean(x))**2, (i, 0, n-1)) -> jnp.mean((x - jnp.mean(x))**2)
    - Sum((x[i] - mean(x)) * (y[i] - mean(y)), (i, 0, n-1)) / sqrt(...)

    The printers in tvbo.export.code handle the translation.
    """
    args = get_func_args(func)
    params = get_func_params_with_defaults(func)
    param_strs = [f"{name}={value}" for name, value in params]
    all_args = args + param_strs
    signature = ', '.join(all_args)

    rhs = get_equation_rhs(func)
    body = render_expression(rhs, format=format, user_functions=user_functions or {})
%>\
def ${func.name}(${signature}):
% if hasattr(func, 'description') and func.description:
    """${func.description}"""
% endif
    return ${body}
</%def>

## =============================================================================
## Utility Defs
## =============================================================================

<%def name="function_body_only(func, format='jax', user_functions=None, render_func=None)">
<%
    """Render just the function body (return expression), no def statement."""
    rhs = get_equation_rhs(func)
    if render_func is not None:
        body = render_func(func)
    elif rhs and '[' in rhs and ('::' in rhs or ':' in rhs):
        body = rhs
    elif rhs:
        body = render_expression(rhs, format=format, user_functions=user_functions or {})
    else:
        body = 'None'
%>\
${body}\
</%def>

<%def name="inline_function(func, format='jax', user_functions=None)">
<%
    """Generate a lambda expression for the function."""
    args = get_func_args(func)
    rhs = get_equation_rhs(func)
    if rhs and '[' in rhs and ('::' in rhs or ':' in rhs):
        body = rhs
    elif rhs:
        body = render_expression(rhs, format=format, user_functions=user_functions or {})
    else:
        body = 'None'
    arg_str = ', '.join(args)
%>\
lambda ${arg_str}: ${body}\
</%def>
