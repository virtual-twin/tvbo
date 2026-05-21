# -*- coding: utf-8 -*-
<%doc>TVB-Optim Observation Template. Context: experiment (SimulationExperiment).</%doc>
<%
from tvbo.codegen import render_expression
from tvbo.templates.tvboptim.utils import (
    get_attr, to_numeric, get_recorded_variable_names,
    adapt_class_reference_for_tvboptim,
)

model = experiment.dynamics
state_names = list(model.state_variables.keys()) if model else ['x']
# Recorded variable layout (states + auxiliaries-in-VOI). Matches solution.variable_names
# produced by tvboptim >= 0.2.7 and the dfun's VARIABLES_OF_INTEREST tuple.
_, _recorded_aux, var_names = get_recorded_variable_names(model, experiment) if model else ([], [], ['x'])
dt = experiment.integration.step_size if experiment.integration else 0.1


def resolve_var_index(source, label: str) -> int:
    """Resolve an observation source to its column in result.data / solution.ys.

    Returns the index of ``source`` within the recorded variable layout
    (states + auxiliaries-in-VOI). ``source`` may be a single name (legacy
    scalar form) or a list of names (multivalued); for lists the first
    state-variable-shaped entry wins. Network/edge/class-reference sources
    and missing sources fall back to ``0`` to preserve prior behavior for
    the external-monitor and BIDS paths.
    """
    if not source:
        return 0
    candidates = source if isinstance(source, (list, tuple)) else [source]
    plain_names = []
    for item in candidates:
        s = item.name if hasattr(item, 'name') else item
        s = str(s)
        if s.startswith('network.'):
            return 0
        if s in var_names:
            return var_names.index(s)
        plain_names.append(s)
    if not plain_names:
        return 0
    raise ValueError(
        f"Observation '{label}' source '{plain_names}' not in recorded "
        f"variables {var_names}. Add it to model.output or reference it "
        f"from an observation so the solver records it."
    )

def is_numeric_string(s):
    """Check if string represents a number."""
    return s.replace('.', '').replace('-', '').replace('_', '').isdigit()

# =============================================================================

# =============================================================================
def parse_reference(val, step_names=None, current_obs_name=None):
    """Parse argument value into (ref_type, ref_value).

    Reference types:
    - 'step'        : pipeline step output (_outputs[step_name])
    - 'input'       : pipeline local (_outputs[key])
    - 'integration' : simulation result (transient/result)
    - 'observation' : another observation's output
    - 'network'     : network property (e.g., network.observations.BoldCorrelation)
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
        if prefix == 'network':
            # network.observations.BoldCorrelation → ('network', 'observations.BoldCorrelation')
            return ('network', key)
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
    if ref_type == 'network':
        # network.observations.BoldCorrelation → _network_observations['BoldCorrelation']
        # ref_val is 'observations.BoldCorrelation'
        if ref_val.startswith('observations.'):
            obs_key = ref_val.split('.', 1)[1]
            return f"_network_observations['{obs_key}']"
        # For other network properties, use kwargs
        return f"kwargs.get('{ref_val}')"
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
from tvbo.codegen.templater import is_derived as _is_derived
_obs_raw = get_attr(experiment, 'observations', {})
if hasattr(_obs_raw, 'items'):
    _all_observations = dict(_obs_raw.items())
elif hasattr(_obs_raw, '__iter__') and not isinstance(_obs_raw, dict):
    _all_observations = {get_attr(o, 'name', f'obs_{i}'): o for i, o in enumerate(_obs_raw)}
else:
    _all_observations = {}
# Raw observations only — derived ones are handled by the experiment-level
# DAG walker, not by per-observation monitor rendering.
observations = {n: o for n, o in _all_observations.items() if not _is_derived(o, experiment)}

# =============================================================================

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

def can_precompute(step):
    """Check if a pipeline step can be precomputed (no data dependency).

    A step can be precomputed if:
    1. It's a kernel generator (has time_range), OR
    2. It has no arguments that reference runtime data (integration.*, observation.*, etc.)

    Currently we only precompute kernel generators since they're the most common case.
    """
    step_name = step.get('name')
    if is_kernel_generator(step_name):
        return True
    return False

def get_precompute_call(step):
    """Generate the function call for precomputation."""
    step_name = step['name']
    args = step.get('arguments', {})

    # Build keyword arguments (only literals, no data references)
    kwargs = []
    for name, val in args.items():
        ref_type, ref_val = parse_reference(val)
        if ref_type == 'literal':
            if isinstance(ref_val, str):
                kwargs.append(f"{name}='{ref_val}'")
            else:
                kwargs.append(f"{name}={ref_val}")

    return f"{step_name}({', '.join(kwargs)})"

def build_vmap_call(callable_ref, step, step_names, current_obs_name, state_idx):
    """Build a double-vmap-wrapped callable for apply_on_dimension: node.

    For 3D data (time, state, node), wraps with double vmap:
    - Outer vmap: iterate over states (axis=1)
    - Inner vmap: iterate over nodes (axis=1 of each state slice)

    For fftconvolve:
        jax.vmap(lambda y: jax.vmap(lambda x: fftconvolve(x, kernel, mode), in_axes=1, out_axes=1)(y), in_axes=1, out_axes=1)(data)
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

    # Double vmap for 3D data (time, state, node)
    inner_vmap = f"jax.vmap(lambda x: {inner_call}, in_axes=1, out_axes=1)"
    return f"jax.vmap(lambda y: {inner_vmap}(y), in_axes=1, out_axes=1)({data_code})"

# =============================================================================

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

# =============================================================================
def parse_class_reference(class_ref, obs):
    """Parse a ClassReference into a clean dict structure."""
    if not class_ref:
        return None

    result = {
        'name': get_attr(class_ref, 'name'),
        'module': get_attr(class_ref, 'module'),
        'constructor_args': {},
        'constructor_arg_codes': {},
        'call_args': {},
        'warmup_source': get_attr(class_ref, 'warmup_source') or get_attr(obs, 'warmup_source'),
        'accepts_voi': False,
        'accepts_history': False,
        'extra_imports': [],
    }

    # Parse constructor_args
    for arg in (get_attr(class_ref, 'constructor_args') or []):
        name = get_attr(arg, 'name')
        val = get_attr(arg, 'value')
        if name:
            parsed_value = to_numeric(val) if val is not None else None
            result['constructor_args'][str(name)] = parsed_value
            if parsed_value is not None:
                result['constructor_arg_codes'][str(name)] = repr(parsed_value)

    # Parse call_args
    for arg in (get_attr(class_ref, 'call_args') or []):
        name = get_attr(arg, 'name')
        val = get_attr(arg, 'value')
        if name:
            result['call_args'][str(name)] = val

    return adapt_class_reference_for_tvboptim(result, obs, dt)

# =============================================================================

# =============================================================================
obs_list = []
callable_imports = {}  # {module: set(qualnames)}
class_ref_imports = {}  # {module: set(class_names)}

for obs_name, obs in observations.items():
    info = {
        'name': obs_name,
        'label': get_attr(obs, 'label', ''),
        'description': get_attr(obs, 'description', ''),
        'source': None,
        # Regular observations derive from simulation state, not other observations
        # DerivedObservation uses source_observations (plural) - handled separately
        'pipeline': [],
        'class_reference': None,  # New: direct class reference
        'period': get_attr(obs, 'period'),  # Sampling period (ms) for time computation
        'tail_samples': get_attr(obs, 'tail_samples'),  # Last N samples before aggregation
        'aggregation': get_attr(obs, 'aggregation'),  # Aggregation type (mean, last, first, etc.)
    }

    # Check for class_reference first (takes precedence over pipeline)
    class_ref = get_attr(obs, 'class_reference')
    if class_ref:
        info['class_reference'] = parse_class_reference(class_ref, obs)
        # Collect import
        if info['class_reference'] and info['class_reference']['module']:
            class_ref_imports.setdefault(info['class_reference']['module'], set()).add(info['class_reference']['name'])
            for extra_module, extra_name in info['class_reference'].get('extra_imports', []):
                class_ref_imports.setdefault(extra_module, set()).add(extra_name)

    # Source for this observation. The slot is multivalued; for raw
    # observations (the path this template handles) there is exactly one
    # entry, a state-variable or `network.*` reference. Take the first
    # entry verbatim.
    src = get_attr(obs, 'source')
    if isinstance(src, (list, tuple)):
        src = src[0] if src else None
    if src:
        info['source'] = get_attr(src, 'name', str(src)) if hasattr(src, 'name') else str(src)

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

# =============================================================================

# =============================================================================
# Identify pipeline steps that can be precomputed (no data dependency)

precomputable_steps = {}  # {step_name: {'call': 'fn(...)', 'const_name': '_PRECOMPUTED_...'}}

for obs in obs_list:
    for step in obs.get('pipeline', []):
        step_name = step.get('name')
        if step_name and can_precompute(step) and step_name not in precomputable_steps:
            precomputable_steps[step_name] = {
                'call': get_precompute_call(step),
                'const_name': f'_PRECOMPUTED_{step_name.upper()}',
            }

# =============================================================================
# Check for Network Observations (loaded from BIDS or edge data)
# =============================================================================
# Identify observations that reference network.observations.* and extract the keys
network_obs_keys = set()
for obs in obs_list:
    src = obs.get('source')
    if src and str(src).startswith('network.observations.'):
        key = str(src).split('network.observations.')[1]
        network_obs_keys.add(key)

# Identify observations that reference network.edges.* and extract matrix data
network_edge_data = {}  # {edge_label: list} — embedded at codegen time
for obs in obs_list:
    src = obs.get('source')
    if src and str(src).startswith('network.edges.'):
        edge_label = str(src).split('network.edges.')[1]
        if edge_label not in network_edge_data:
            _net = get_attr(experiment, 'network', None)
            if _net:
                _matrix = _net.matrix(edge_label)
                if _matrix is not None:
                    network_edge_data[edge_label] = _matrix.tolist()

# Get BIDS directory from experiment network (resolve to absolute path)
bids_dir = None
if network_obs_keys:
    network = get_attr(experiment, 'network', None)
    if network:
        _bids_dir_raw = get_attr(network, 'bids_dir', None)
        if _bids_dir_raw:
            from pathlib import Path
            _bids_path = Path(_bids_dir_raw)
            if not _bids_path.is_absolute():
                # Resolve relative to YAML spec file's parent directory
                # e.g., YAML at database/experiments/foo.yaml with bids_dir: ../networks/bids/dk
                _source_file = get_attr(experiment, '_source_file', None)
                if _source_file:
                    _bids_path = (Path(_source_file).parent / _bids_dir_raw).resolve()
                else:
                    # Fallback: resolve relative to cwd
                    _bids_path = (Path.cwd() / _bids_dir_raw).resolve()
            bids_dir = str(_bids_path)
%>
"""Observation classes derived from AbstractMonitor for tvboptim."""

import math
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from types import SimpleNamespace
from tvboptim.experimental.network_dynamics.result import NativeSolution
from tvboptim.observations.tvb_monitors.downsampling import AbstractMonitor
from tvbo.data.types import ObservationResult

% if network_obs_keys and bids_dir:
from tvbo.classes.network import Network as _TvboNetwork

_bids_network = _TvboNetwork.from_bids('${bids_dir}', observational_measures=${list(network_obs_keys)})
% endif

% for module in sorted(callable_imports.keys()):
% if module not in ('jax', 'numpy', 'np', 'jnp', 'equinox', 'eqx'):
import ${module}
% endif
% endfor

% for module, class_names in sorted(class_ref_imports.items()):
% for class_name in sorted(class_names):
from ${module} import ${class_name} as _Ext${class_name}
% endfor
% endfor

% for obs in obs_list:
<%
    obs_name = obs['name']
    obs_source = obs['source']
    # Regular observations don't have source_observation - that's only for DerivedObservation
    pipeline = obs['pipeline']
    class_ref = obs.get('class_reference')
    class_name = ''.join(word.capitalize() for word in obs_name.split('_'))

    # Check if source is from network.observations (static data from BIDS)
    is_network_observation = obs_source and str(obs_source).startswith('network.observations.')
    network_obs_key = str(obs_source).split('network.observations.')[1] if is_network_observation else None

    # Check if source is from network.edges (edge matrix data)
    is_network_edge = obs_source and str(obs_source).startswith('network.edges.')
    network_edge_label = str(obs_source).split('network.edges.')[1] if is_network_edge else None

    # Resolve source to its column in the recorded variable layout (states + recorded aux).
    # Returns 0 for network/external/empty sources (back-compat for external monitor paths).
    state_idx = resolve_var_index(obs_source, obs_name)
%>
% if is_network_edge and network_edge_label in network_edge_data:
## =============================================================================
## Static Network Edge Data (embedded from network edge matrix)
## =============================================================================
# ${obs['label'] or obs_name} - edge data from network
${obs_name} = jnp.array(${repr(network_edge_data[network_edge_label])})

% elif is_network_observation and bids_dir:
## =============================================================================
## Static Network Observation (data from BIDS - module-level constant)
## =============================================================================
# ${obs['label'] or obs_name} - static data loaded from BIDS
${obs_name} = jnp.asarray(_bids_network.observations['${network_obs_key}'])

% elif class_ref:
## =============================================================================
## Class Reference Observation (direct external class usage)
## =============================================================================
<%
    ext_class_name = class_ref['name']
    ext_module = class_ref['module']
    constructor_args = class_ref['constructor_args']
    constructor_arg_codes = dict(class_ref.get('constructor_arg_codes') or {})
    call_args = class_ref['call_args']
    warmup_source = class_ref.get('warmup_source')
    accepts_history = class_ref.get('accepts_history') or bool(warmup_source)
    if class_ref.get('accepts_voi') and 'voi' not in constructor_arg_codes:
        constructor_arg_codes['voi'] = 'voi'

    # Build constructor kwargs string
    init_kwargs = []
    for arg_name, arg_val in constructor_args.items():
        if arg_val is not None:
            if isinstance(arg_val, str) and not arg_val.replace('.', '').replace('-', '').isdigit():
                init_kwargs.append(f"{arg_name}='{arg_val}'")
            else:
                init_kwargs.append(f"{arg_name}={arg_val}")
    init_kwargs_str = ', '.join(init_kwargs)

    # Check if history is needed (warmup_source specified)
    needs_history = bool(accepts_history)
%>

class ${class_name}(eqx.Module):
    """${obs['label'] or obs_name} observation (external class wrapper).

    ${obs['description'] or f'Wraps external class {ext_class_name} from {ext_module}.'}

    Uses: ${ext_module}.${ext_class_name}
    """
    # External monitor instance
    _monitor: eqx.Module

    def __init__(self, history=None, voi: int = ${state_idx}, dt: float = ${dt}, **kwargs):
        """Initialize observation using external class.

        Args:
            history: NativeSolution from transient simulation (for warmup)
            voi: Variable of interest index (default: ${state_idx})
            dt: Time step (default: ${dt})
            **kwargs: Additional arguments passed to ${ext_class_name}
        """
        # Merge default constructor args with kwargs (filter out internal keys)
        _init_kwargs = {${', '.join([f"'{k}': {v}" for k, v in constructor_arg_codes.items()])}}
        # Filter out internal keys not meant for external class
        _internal_keys = {'result', 'result_transient', 'model_fn', 'state', 'history'}
        _init_kwargs.update({k: v for k, v in kwargs.items() if k not in _internal_keys})
% if needs_history:
        _init_kwargs['history'] = history
% endif
    % if class_ref.get('accepts_voi') or 'voi' in constructor_arg_codes or needs_history:
        _init_kwargs.setdefault('voi', voi)
% endif

        self._monitor = _Ext${ext_class_name}(**_init_kwargs)

    def __call__(self, result):
        """Process simulation result using external class."""
        return self._monitor(result)

% else:
## =============================================================================
## Pipeline-based Observation (existing implementation)
## =============================================================================
<%
    # Analyze pipeline for data requirements
    needs_transient, needs_result_from_pipeline = analyze_pipeline(pipeline)

    # Resolve source to its column in the recorded variable layout.
    state_idx = resolve_var_index(obs_source, obs_name)

    # Declarative observation attributes (language-independent)
    tail_samples = obs.get('tail_samples')  # Last N samples before aggregation
    aggregation = obs.get('aggregation')  # Aggregation type (mean, last, first, etc.)

    # Identify precomputable steps (kernels) and history steps
    static_steps = []  # Kernel generators - computed in __init__, stored as static
    history_steps = []  # Steps that reference integration.transient
    dynamic_steps = []  # Everything else - computed in __call__

    for step in pipeline:
        step_name = step['name']
        if is_kernel_generator(step_name):
            static_steps.append(step)
        else:
            # Check if step references transient
            refs_transient = False
            for val in step.get('arguments', {}).values():
                if isinstance(val, str) and 'transient' in val:
                    refs_transient = True
                    break
            if refs_transient:
                history_steps.append(step)
            else:
                dynamic_steps.append(step)

    # Determine final output key - must match the actual variable name generated
    if pipeline:
        last_step = pipeline[-1]
        # Use the step's output if set, otherwise the step name (matches the variable generation)
        final_key = last_step.get('output') or last_step['name']
    else:
        final_key = 'data'

    # Collect step names and named outputs for reference resolution
    step_names = set()
    named_outputs = []  # All named outputs to expose on result
    for s in pipeline:
        step_names.add(s['name'])
        if s.get('output'):
            for out in s['output'].split(','):
                out_name = out.strip()
                step_names.add(out_name)
                named_outputs.append(out_name)

    # Collect all observation references in pipeline (for cross-observation access)
    # e.g., when a pipeline step needs another observation's output
    referenced_observations = set()
    for s in pipeline:
        for arg_val in s.get('arguments', {}).values():
            if isinstance(arg_val, str) and '.' in arg_val:
                prefix = arg_val.split('.')[0]
                if prefix in observations:
                    referenced_observations.add(prefix)

    # Check if pipeline has explicit prepend_history step (avoids redundant auto-generation)
    has_prepend_history_step = any(s['name'] == 'prepend_history' for s in pipeline)
%>

class ${class_name}(AbstractMonitor):
    """${obs['label'] or obs_name} observation.

    ${obs['description'] or 'Auto-generated observation class.'}
% if pipeline:
    Pipeline: ${' -> '.join([s['name'] for s in pipeline])}
% endif
    """
    # AbstractMonitor fields (voi and period are inherited)
    dt: float = eqx.field(static=True, default=${dt})

% for step in static_steps:
    # Precomputed: ${step['name']} (kernel)
    _${step['name']}: jax.Array
% endfor
% if needs_transient:
    # History buffer (preprocessed from transient)
    _history: jax.Array = None
% endif
% for ref_obs in referenced_observations:
    # Referenced observation monitor: ${ref_obs}
    _${ref_obs}_monitor: eqx.Module = None
% endfor

    def __init__(self, history=None, voi: int = ${state_idx}, period: float = None, dt: float = ${dt}${''.join([f", {step['name']}_params=None" for step in static_steps])}):
        self.voi = self._normalize_voi(voi)
        self.period = period if period is not None else dt
        self.dt = dt

% for step in static_steps:
<%
    step_name = step['name']
    fn_def = functions_by_name.get(step_name)
    # Build default args from function definition
    default_args = []
    for arg in (get_attr(fn_def, 'arguments') or []):
        arg_name = get_attr(arg, 'name')
        arg_val = get_attr(arg, 'value')
        if arg_name and arg_val is not None:
            default_args.append(f"{arg_name}={to_numeric(arg_val)}")
%>
        self._${step_name} = ${step_name}(${', '.join(default_args)})
% endfor

% if needs_transient:
        if history is not None:
            if hasattr(history, 'data'):
% if static_steps:
                n_samples = int(math.ceil(self._${static_steps[0]['name']}.shape[0]))
% else:
                n_samples = 5000  # Default history length
% endif
% if obs_source:
                self._history = history.data[-n_samples:, self.voi, :]
% else:
                self._history = history.data[-n_samples:, :, :]
% endif
            else:
                self._history = history
% endif

% for ref_obs in referenced_observations:
        self._${ref_obs}_monitor = ${ref_obs.replace('_', ' ').title().replace(' ', '')}(history=history, voi=voi, dt=dt)
% endfor

    def __call__(self, result):
% for ref_obs in referenced_observations:
        _${ref_obs}_result = self._${ref_obs}_monitor(result)
% endfor
% if obs_source:
        _data = result.data[:, self.voi, :]
        _time = result.time
% else:
        _data = result.data
        _time = result.time
% endif

% if needs_transient and not has_prepend_history_step:
        _data_with_history = jnp.concatenate([self._history, _data], axis=0)
% endif

% for step_idx, step in enumerate(pipeline):
<%
    step_name = step['name']
    step_output = step.get('output') or step_name
    step_callable = step['callable']
    apply_dim = step.get('apply_on_dimension')
    is_static = step in static_steps

    # Handle multi-output steps: "frequencies, psd" -> "_frequencies, _psd"
    def prefix_outputs(out_str):
        """Prefix all output names with underscore."""
        parts = [o.strip() for o in out_str.split(',')]
        return ', '.join(f'_{p}' for p in parts)

    prefixed_output = prefix_outputs(step_output)

    # Determine input variable
    if step_idx == 0:
        # First step always gets _data; prepend_history step handles concatenation if needed
        input_var = '_data'
    else:
        prev_step = pipeline[step_idx - 1]
        prev_output = prev_step.get('output') or prev_step['name']
        # For multi-output, use last output as input to next step
        prev_parts = [o.strip() for o in prev_output.split(',')]
        input_var = f"_{prev_parts[-1]}"
%>
% if is_static:
        ${prefixed_output} = self._${step_name}
% elif step_callable and step_callable.get('full_call'):
<%
    full_call = step_callable['full_call']
    args = step.get('arguments', {})

    # Build call arguments
    call_parts = []
    for arg_name, arg_val in args.items():
        if isinstance(arg_val, str):
            if arg_val in step_names or arg_val == 'data':
                # Reference to previous step output
                if arg_val == 'data':
                    call_parts.append(f"{arg_name}={input_var}")
                else:
                    call_parts.append(f"{arg_name}=_{arg_val}")
            elif 'transient' in arg_val:
                call_parts.append(f"{arg_name}=self._history")
            elif arg_val.startswith('integration.'):
                # Reference to simulation data
                call_parts.append(f"{arg_name}=_data")
            elif arg_val in observations:
                # Reference to another observation's data
                call_parts.append(f"{arg_name}=_data")
            elif '.' in arg_val and not arg_val.replace('.', '').replace('-', '').isdigit():
                # Dotted reference: check for observation.attribute pattern (e.g., simulated_psd.psd)
                prefix, attr = arg_val.split('.', 1)
                if prefix in referenced_observations:
                    # Reference to observation's named output (e.g., simulated_psd.frequencies)
                    # Observation result is stored in _<prefix>_result
                    call_parts.append(f"{arg_name}=_{prefix}_result.{attr}")
                elif prefix in observations:
                    # Reference to observation not in our dependency set - warn
                    call_parts.append(f"{arg_name}='{arg_val}'  # WARNING: observation not accessible")
                else:
                    # Unknown dotted reference - pass as string literal
                    call_parts.append(f"{arg_name}='{arg_val}'")
            elif arg_val.replace('.', '').replace('-', '').isdigit():
                call_parts.append(f"{arg_name}={arg_val}")
            else:
                call_parts.append(f"{arg_name}='{arg_val}'")
        else:
            call_parts.append(f"{arg_name}={repr(arg_val)}")

    # Handle vmap for node dimension
    # For 3D data (time, state, node), use double vmap: outer over states, inner over nodes
    if apply_dim == 'node':
        # For convolution, build the lambda
        kernel_arg = None
        mode_arg = "'valid'"
        for arg_name, arg_val in args.items():
            if arg_name in ('in2', 'kernel'):
                if arg_val in step_names:
                    kernel_arg = f"_{arg_val}"
                else:
                    kernel_arg = f"self._{arg_val}"
            elif arg_name == 'mode':
                mode_arg = f"'{arg_val}'" if isinstance(arg_val, str) else repr(arg_val)
        # Double vmap for 3D data (time, state, node):
        # - Outer vmap: iterate over states (axis=1)
        # - Inner vmap: iterate over nodes (axis=1 of each state slice)
        inner_vmap = f"jax.vmap(lambda x: {full_call}(x, {kernel_arg or '_kernel'}, {mode_arg}), in_axes=1, out_axes=1)"
        callable_code = f"jax.vmap(lambda y: {inner_vmap}(y), in_axes=1, out_axes=1)({input_var})"
        callable_comment = f"# {step_name}: double vmap over states and nodes"
    else:
        callable_code = f"{full_call}({', '.join(call_parts) if call_parts else input_var})"
        callable_comment = f"# {step_name}"
%>
        ${callable_comment}
        ${prefixed_output} = ${callable_code}
% elif step_name in functions_by_name:
<%
    # User-defined function call
    args = step.get('arguments', {})
    call_parts = []
    for arg_name, arg_val in args.items():
        if isinstance(arg_val, str):
            if arg_val in step_names:
                # Reference to a previous pipeline step output
                call_parts.append(f"{arg_name}=_{arg_val}")
            elif 'transient' in arg_val:
                # Reference to transient history
                call_parts.append(f"{arg_name}=self._history")
            elif arg_val.startswith('integration.'):
                # Reference to simulation data - use _data (raw simulation output)
                # Functions that need history should reference integration.transient separately
                call_parts.append(f"{arg_name}=_data")
            elif arg_val == 'data':
                # Generic data reference - use input_var (previous step output)
                call_parts.append(f"{arg_name}={input_var}")
            elif arg_val in observations:
                # Reference to another observation's data
                call_parts.append(f"{arg_name}=_data")
            elif '.' in arg_val and not arg_val.replace('.', '').replace('-', '').replace('_', '').isdigit():
                # Dotted reference: check for observation.attribute pattern (e.g., simulated_psd.psd)
                prefix, attr = arg_val.split('.', 1)
                if prefix in referenced_observations:
                    # Reference to observation's named output (e.g., simulated_psd.frequencies)
                    call_parts.append(f"{arg_name}=_{prefix}_result.{attr}")
                elif prefix in observations:
                    # Reference to observation not in our dependency set - warn
                    call_parts.append(f"{arg_name}='{arg_val}'  # WARNING: observation not accessible")
                else:
                    # Unknown dotted reference - pass as string literal
                    call_parts.append(f"{arg_name}='{arg_val}'")
            elif arg_val.replace('.', '').replace('-', '').replace('_', '').isdigit():
                # Numeric value
                call_parts.append(f"{arg_name}={arg_val}")
            else:
                # String literal
                call_parts.append(f"{arg_name}='{arg_val}'")
        else:
            call_parts.append(f"{arg_name}={repr(arg_val)}")
    if not call_parts:
        call_parts.append(input_var)
%>
        ${prefixed_output} = ${step_name}(${', '.join(call_parts)})
% elif step.get('source_code'):
        ${prefixed_output} = ${step['source_code'].replace('_input', input_var)}
% else:
        ${prefixed_output} = ${input_var}
% endif
% endfor

% if tail_samples or aggregation:
<%
    # Determine input variable for declarative processing
    if pipeline:
        last_step = pipeline[-1]
        last_output = last_step.get('output') or last_step['name']
        final_parts = [o.strip() for o in last_output.split(',')]
        decl_input = f"_{final_parts[-1]}"
    else:
        decl_input = '_data'
%>
% if tail_samples:
        ${decl_input} = ${decl_input}[-${tail_samples}:]
% endif
% if str(aggregation) == 'mean':
        # Declarative: mean over time axis (aggregation: mean)
        ${decl_input} = jnp.mean(${decl_input}, axis=0)
% elif str(aggregation) == 'last':
        ${decl_input} = ${decl_input}[-1]
% elif str(aggregation) == 'first':
        ${decl_input} = ${decl_input}[0]
% endif
% endif

<%
    # Determine primary data - last output name or 'data'
    final_outputs = [o.strip() for o in final_key.split(',')]
    primary_output = final_outputs[-1] if final_outputs else 'data'
%>
        _final = _${primary_output}
        if isinstance(_final, NativeSolution):
            return _final

        _is_scalar = _final.ndim == 0 if hasattr(_final, 'ndim') else True
        if _is_scalar:
            _ts = None  # Scalar result has no time dimension
        elif _time is not None and len(_time) >= len(_final):
            _ts = _time[:len(_final)]
        else:
            _ts = jnp.arange(len(_final)) * self.dt
% if named_outputs:
        return ObservationResult(
            ts=_ts,
            ys=_final,
            dt=self.dt,
% for out_name in named_outputs:
            ${out_name}=_${out_name},
% endfor
        )
% else:
        return NativeSolution(ts=_ts if _ts is not None else jnp.array([0.0]), ys=_final, dt=self.dt)
% endif

% endif
% endfor
