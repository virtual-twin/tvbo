# -*- coding: utf-8 -*-
<%doc>TVB-Optim Observation Template. Context: experiment (SimulationExperiment).</%doc>
<%
from tvbo.codegen import render_expression
from tvbo.templates.tvboptim.utils import (
    get_attr, to_numeric, get_recorded_variable_names,
    adapt_class_reference_for_tvboptim, resolve_reduction, iter_parameter_values,
    edge_label as _edge_label, edge_const as _edge_const, collect_network_edge_arrays,
    node_label as _node_label, node_const as _node_const, collect_network_node_arrays,
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
        if s.startswith('network.') or s.startswith('dataset.'):
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
        # network.weight(s)/length(s), or the explicit network.edges.<label>, → the
        # connectivity matrix embedded once as a module constant (network_edge_arrays
        # below), so a callable can receive the connectome (connectogram, betweenness,
        # moments, ...).
        _label = _edge_label(ref_val)
        if _label:
            return _edge_const(_label)
        # network.positions/instrength → the per-node vector embedded once as a module
        # constant (network_node_arrays below), mirroring the edge-matrix path, so a
        # callable or observer parameter can receive region centroids / in-strength.
        _nlabel = _node_label(ref_val)
        if _nlabel:
            return _node_const(_nlabel)
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
# Raw observations only — derived ones are handled by the experiment-level DAG
# walker, and analysis observations (gradient/lyapunov/...) by their own path.
observations = {n: o for n, o in _all_observations.items() if not _is_derived(o, experiment) and getattr(o, 'analysis', None) is None}

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

    # Parse arguments (these ARE function arguments), keyed by name (key == arg name).
    for name, arg in (get_attr(func, 'arguments') or {}).items():
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

            # Merge function arguments (step args take precedence), keyed by name.
            for name, arg in (get_attr(fn_def, 'arguments') or {}).items():
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
        # Aggregation knobs, resolved once from the observation's generic
        # `parameters` slot so any parametric aggregation (e.g. first_passage's
        # crossing `threshold`) reads its values by name.
        'agg_params': dict(iter_parameter_values(get_attr(obs, 'parameters'))),
        # An Observation.dynamics observer resolves to a streaming reducer (init, update,
        # finalize). None when the observation has no dynamics — the pipeline path applies.
        'reduction': resolve_reduction(obs, experiment),
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

        # Override arguments from FunctionCall if provided (keyed by name)
        for name, arg in (get_attr(func_call, 'arguments') or {}).items():
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

# Embed connectome matrices referenced by observations (source or callable argument,
# via the network.weight(s)/length(s) shortcut or the explicit network.edges.<label>
# form) as module constants. Covers derived and non-derived observations, so the same
# constant serves both the source path here and the derived resolver in the experiment
# template (which includes this module). See utils.collect_network_edge_arrays.
network_edge_arrays = collect_network_edge_arrays(experiment)

# Embed per-node vectors referenced by observations (network.positions/instrength) as
# module constants, the node-level analogue of the connectome matrices above. See
# utils.collect_network_node_arrays.
network_node_arrays = collect_network_node_arrays(experiment)

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
%>\
## Emit a tvboptim reducer (init, update, finalize) from a resolved Observation.dynamics
## observer (utils.resolve_reduction). The observer's per-step recurrence runs as an inner
## scan over each block — accumulators commit only after a global-step warmup (`_gstep >
## skip`), so the block trajectory is never held. `skip` skips leading samples from the
## reduction: `skip=0` reproduces the plain "no accumulate on the first step" behaviour
## (the first phase increment needs a previous sample); `skip=n_transient` streams a run
## whose transient has NOT been trimmed, folding only the post-transient window. Memory
## states (e.g. the previous phase) advance every step so the boundary increment is exact.
## Every expression is a sympy Expr rendered via render_expression — backend-independent.
<%def name="render_reduction(red, name, s_idx, dt)">\
% if red.get('kind') == 'convolution':
${render_convolution_reduction(red, name, s_idx, dt)}\
% elif red.get('kind') == 'stride':
${render_stride_reduction(red, name, s_idx, dt)}\
% elif red.get('kind') == 'comoment':
${render_comoment_reduction(red, name, s_idx, dt)}\
% elif red.get('kind') == 'wave':
${render_wave_reduction(red, name, s_idx, dt)}\
% else:
${render_recurrence_reduction(red, name, s_idx, dt)}\
% endif
</%def>\
## Cumulative co-moment FC reducer (a compute_fc pipeline marked reduce: streaming). Folds
## the whole post-transient window as a Welford co-moment (add-only, NO eviction — cumulative,
## not a sliding window), then reads the zero-diagonal Pearson correlation at finalize.
## Matrix-valued state: `comoment` is (n, n), `mean` is (n,), `count` a scalar. The `add`
## assignments and the Pearson `emit` are the declarative windowed_fc recipe already lowered to
## this backend (utils._resolve_fc_stream via resolve_streaming_reducer), so this partial only
## emits the block scaffolding — the accumulator math is the shared reducer spec, no FC logic
## baked here. Byte-identical to compute_fc(source, skip_t) to f64 summation order; the
## compute_fc skip_t adds to the transient `skip` so the same leading samples are dropped.
<%def name="render_comoment_reduction(red, name, s_idx, dt)">\
<%
    _states = list(red['states'])          # ['count', 'mean', 'comoment']
    _add = red['add']                      # [(lhs, jax_rhs), ...] sequential Welford update
    _emit = red['emit']                    # zero-diagonal Pearson correlation over comoment
    _skip_t = int(red.get('skip_t', 0))
    _acc = ", ".join(_states)
    _acc0 = ", ".join("_%s0" % _s for _s in _states)
    # Carry-init shape keyed by state ROLE (not position), so a reducer-spec state reorder
    # cannot silently mis-shape the accumulator; an unknown state fails loudly at emit.
    _INIT_SHAPE = {'count': 'jnp.array(0)', 'mean': 'jnp.zeros((n,))', 'comoment': 'jnp.zeros((n, n))'}
    _missing = [_s for _s in _states if _s not in _INIT_SHAPE]
    if _missing:
        raise ValueError("co-moment reducer: no carry-init shape for state(s) %s" % _missing)
    _init_tuple = ", ".join(_INIT_SHAPE[_s] for _s in _states)
%>\
def _reduction_${name}(s_var=${s_idx}, dt=${repr(dt)}, skip=0, warm_history=None, progress=False):
    # warm_history/progress are accepted-and-ignored so this reducer shares ONE call site
    # with the recurrence and convolution factories. The compute_fc skip_t (${_skip_t}) adds
    # to the transient `skip`: the first (skip + ${_skip_t}) samples are dropped, matching the
    # materialised compute_fc(source, skip_t=${_skip_t}).
    _skip = skip + ${_skip_t}
    def _init(template, n_steps):
        n = template.shape[-1]
        return (${_init_tuple}, jnp.array(0))
    def _update(acc, block):
        def _step(carry, s_row):
            ${_acc}, _gstep = carry
            v = s_row[s_var]
            _accept = _gstep >= _skip
            # Fold one sample via the Welford `add` recurrence, then commit only past `skip`
            # (a running correlation must not fold the dropped leading samples).
            ${_acc0} = ${_acc}
% for _lhs, _rhs in _add:
            ${_lhs} = ${_rhs}
% endfor
% for _s in _states:
            ${_s} = jnp.where(_accept, ${_s}, _${_s}0)
% endfor
            return (${_acc}, _gstep + 1), None
        return jax.lax.scan(_step, acc, block)[0]
    def _finalize(acc):
        ${_acc}, _gstep = acc
        return ${_emit}
    return (_init, _update, _finalize)
</%def>\
## Streaming HRF-Volterra BOLD reducer (Observation.reduce == 'streaming'). Recasts the
## post-scan bold pipeline (HRF kernel -> stride decimation -> prepend downsampled
## transient -> 'valid' HRF convolution -> Volterra scaling -> subsample at the TR) as a
## block reducer: a downsampled-history ring buffer folds each integration block, and
## strided_convolve (a backend-abstracted printer primitive) evaluates the 'valid'
## convolution ONLY at the TR boundaries, so the full trajectory is never held. The
## portable array ops (subsample / concatenate / strided_convolve / Volterra scaling) are
## rendered via render_expression; the ring/buffer scaffolding is the backend template's
## concern, exactly as the recurrence reducer mixes jax.lax.scan with printed exprs.
## Byte-identical to the materialised pipeline to f64 rounding (strided_convolve is ~1e-12
## vs the FFT fftconvolve). `warm_history` seeds the ring from the downsampled transient
## (HRF warm-up); None -> zeros (a transient-free run, e.g. the Schirner group fit).
<%def name="render_convolution_reduction(red, name, s_idx, dt)">\
<%
    from tvbo.codegen import render_expression
    _cv = ['_block_voi', '_ds', '_ring', '_block_ds', '_signal', '_kernel', '_tr', '_conv', 'k_1', 'V_0']
    _rc = lambda e: render_expression(e, format='jax', parameters=_cv)
%>\
def _reduction_${name}(s_var=${s_idx}, dt=${repr(dt)}, skip=0, warm_history=None, progress=False):
    k_1 = ${repr(red['k_1'])}
    V_0 = ${repr(red['V_0'])}
    _kernel = ${red['kernel_call']}   # HRF kernel array [K] (the pipeline's kernel function)
    _K = _kernel.shape[0]
    _ds = ${red['ds_steps']}          # decimation stride (raw integration steps / downsampled sample)
    _tr = ${red['tr_stride']}         # TR stride (downsampled samples / BOLD sample)
    def _init(template, n_steps):
        n = template.shape[-1]
        _n_ds = len(range(_ds - 1, n_steps, _ds))     # downsampled samples over the run
        _n_bold = len(range(_tr, _n_ds + 1, _tr))     # BOLD samples at TR boundaries
        if warm_history is None:
            _ring0 = jnp.zeros((_K, n))
        else:
            # Fit the downsampled transient to one kernel length, front-zero-padded when
            # short — the HRF warm-up, matching prepend_history / HRFBold._process_history.
            _wh = jnp.asarray(warm_history)
            _ring0 = (jnp.vstack([jnp.zeros((_K - _wh.shape[0], n), _wh.dtype), _wh])
                      if _wh.shape[0] < _K else _wh[-_K:])
        return (_ring0, jnp.zeros((_n_bold, n)), jnp.array(0))
    def _update(acc, block):
        _ring, _bold, _ds_count = acc
        _block_voi = block[:, s_var, :]                     # source column -> [block_len, n]
        _block_ds = ${_rc("subsample(_block_voi, _ds - 1, _ds)")}   # SubSampling decimation
        _m_b = _block_ds.shape[0]
        _signal = ${_rc("concatenate(_ring, _block_ds)")}          # [_K + m_b, n]
        _conv = ${_rc("strided_convolve(_signal, _kernel, _tr)")}  # 'valid' conv at TR boundaries
        _samples = ${_rc("k_1 * V_0 * (_conv - 1.0)")}             # Volterra BOLD scaling
        _start = _ds_count // _tr
        _bold = jax.lax.dynamic_update_slice(_bold, _samples, (_start, 0))
        _ring = _signal[-_K:]
        _ds_next = _ds_count + _m_b
        if progress:
            # JAX-native streaming progress: jax.debug.print fires from INSIDE the compiled
            # block scan, so a long post-tuning fold (e.g. the 36e6-step group-fit eval)
            # streams ~50 lines live instead of the log sitting silent until the whole scan
            # returns. Gated to the non-vmapped post-eval (progress=True) so the vmapped
            # exploration grid never floods; the guard is a plain Python bool, not traced.
            _done = _ds_next // _tr
            _ntot = _bold.shape[0]
            _every = jnp.maximum(1, _ntot // 50)
            jax.lax.cond(
                (_done // _every) > ((_ds_count // _tr) // _every),
                lambda: jax.debug.print("  post-eval: streamed {}/{} BOLD samples", _done, _ntot),
                lambda: None,
            )
        return (_ring, _bold, _ds_next)
    def _finalize(acc):
        _ring, _bold, _ds_count = acc
        return _bold
    return (_init, _update, _finalize)
</%def>\
## Streaming stride reducer (Observation.reduce == 'streaming' over a pure decimation
## pipeline). Keeps every _ds-th sample of the source column and writes it straight into a
## preallocated buffer, so a run that reports 1/_ds of its samples never materialises the
## other (_ds - 1)/_ds. Bit-identical to the materialised SubSampling: the kept samples are
## the same samples. Block boundaries are multiples of _ds (streaming_post_eval_plan), so a
## block's local decimation grid is the global one and the write slot is exact.
<%def name="render_stride_reduction(red, name, s_idx, dt)">\
<%
    from tvbo.codegen import render_expression
    _rc = lambda e: render_expression(e, format='jax', parameters=['_block_voi', '_ds'])
%>\
def _reduction_${name}(s_var=${s_idx}, dt=${repr(dt)}, skip=0, warm_history=None, progress=False):
    # warm_history/progress are accepted-and-ignored so this reducer shares ONE call site
    # with the recurrence, comoment and convolution factories.
    _ds = ${red['ds_steps']}           # decimation stride (integration steps per kept sample)
    # `skip` is the transient, in integration steps, for the sweep path, which folds
    # transient and main run into ONE window (the single-run path integrates them
    # separately and streams only the main one). Without this the sweep silently keeps
    # `skip/_ds` extra leading samples and a swept cell is NOT the same computation as the
    # same experiment run alone. Dropped at finalize rather than gated per block, so the
    # arithmetic never depends on the block size dividing the stride. The sample GRID stays
    # anchored at 0, so when `skip` is not a multiple of `_ds` the kept samples sit up to
    # one stride earlier than the single run's — same count, same post-transient window,
    # phase offset < 1 sample.
    _skip_n = max(0, -(-(skip + 1) // _ds) - 1) if skip else 0
    def _init(template, n_steps):
        n = template.shape[-1]
        return (jnp.zeros((len(range(_ds - 1, n_steps, _ds)), n)), jnp.array(0))
    def _update(acc, block):
        _out, _count = acc
        _block_voi = block[:, s_var, :]
        _samples = ${_rc("subsample(_block_voi, _ds - 1, _ds)")}
        _out = jax.lax.dynamic_update_slice(_out, _samples, (_count // _ds, 0))
        return (_out, _count + block.shape[0])
    def _finalize(acc):
        _out, _count = acc
        return _out[_skip_n:]
    return (_init, _update, _finalize)
</%def>\
## The two per-step observer fragments every reduction branch shares, emitted at the caller's
## indentation `ind` with its expression printer `jc`. Kept separate because the histogram
## fold evaluates its per-step sample BETWEEN them.
<%def name="render_observer_dvs(dvs, jc, ind)">\
% for _d in dvs:
% if 'surrogate' in _d:
${render_surrogate(_d['name'], _d['surrogate'], jc, ind)}\
% else:
${ind}${_d['name']} = ${jc(_d['expr'])}
% endif
% endfor
</%def>\
## A permutation-significance surrogate (DerivedVariable.surrogate): re-evaluate a named
## statistic under the fixed `(n_perm, n)` permutation table and report the per-element
## exceedance p-value. `expr` is the statistic as a self-contained function of the permuted
## symbol; the observed value reuses the already-computed statistic DV. `family_reduce`
## (e.g. `nanmax`) collapses the permuted statistic over its element axes to one family-wise
## extremum per permutation — the Westfall–Young max-T FWE null each observed element is
## tested against; absent → the symmetric per-element test.
<%def name="render_surrogate(sname, surr, jc, ind)">\
<%
    _perm = surr['permute']
    _perms = surr['perms']
    _cmp = surr['compare']
    _fam = surr.get('family_reduce')
    _obs = surr.get('statistic')
%>\
${ind}def _surrstat_${sname}(${_perm}):
${ind}    return ${jc(surr['expr'])}
${ind}_obs_${sname} = ${_obs if _obs else '_surrstat_%s(%s)' % (sname, _perm)}
${ind}_null_${sname} = jax.vmap(_surrstat_${sname})(${_perm}[${_perms}])
% if _fam:
${ind}_null_${sname} = jnp.${_fam}(_null_${sname}, axis=tuple(range(1, _null_${sname}.ndim)))
${ind}${sname} = jnp.mean((_null_${sname}.reshape((-1,) + (1,) * _obs_${sname}.ndim) ${_cmp} _obs_${sname}[None]) * 1.0, axis=0)
% else:
${ind}${sname} = jnp.mean((_null_${sname} ${_cmp} _obs_${sname}) * 1.0, axis=0)
% endif
</%def>\
## `states` is the subset this branch advances: all of them, or memory-only for the
## histogram fold (an accumulator there is the histogram itself).
<%def name="render_observer_states(states, jc, ind)">\
% for s in states:
${ind}_new_${s['name']} = ${jc(s['update'])}
% endfor
</%def>\
<%def name="render_recurrence_reduction(red, name, s_idx, dt)">\
<%
    from tvbo.codegen import render_expression
    from tvbo.templates.tvboptim.utils import render_jax_default
    _is_median = red.get('statistic', 'mean') == 'median'
    _period_steps = red.get('period_steps')
    # Pure accumulators (skip_inclusive) fold the sample AT skip; a memory-dependent
    # observer's first step has no predecessor, so it starts the step after (`>`).
    _gate = '>=' if red.get('skip_inclusive') else '>'
    _snames = [s['name'] for s in red['states']]
    _mem = [s for s in red['states'] if not s['is_accumulator']]   # memory-only states
    _mnames = [s['name'] for s in _mem]
    _src = red['source']
    _rpars = red.get('parameters') or {}   # observer constants, bound by name below
    _derived = red.get('derived', [])      # per-step derived-variable chain (sympy Exprs)
    # Non-output DVs are computed per step (they feed states / the median's per-step sample);
    # the output DV itself stays inlined at finalize, so it is excluded here. Inert (empty)
    # for a simple observer whose only DV IS its output.
    _step_dvs = [d for d in _derived if d['name'] != red.get('output_name')]
    _dconsts = red.get('derived_constants') or []
    _rparams = (_snames + [d['name'] for d in _derived] + [d['name'] for d in _dconsts]
                + list(_rpars) + [_src, 'dt', 'count'])
    _rufuncs = {f: f for f in red['functions']}
    _jc = lambda e, ps=_rparams: render_expression(e, format='jax', user_functions=_rufuncs, parameters=ps)
    _h = red.get('histogram')   # guaranteed present for median (resolve_reduction requires it)
    _mem_pre = "".join("%s, " % _n for _n in _mnames)        # "s_prev, "
    _mem_new = "".join("_new_%s, " % _n for _n in _mnames)   # "_new_s_prev, "
    _mem_ini = "".join("jnp.full((n,), %r), " % s['init'] for s in _mem)
    _ind = ' ' * 12   # the scan-step body's indentation, shared by the emitted fragments
%>\
def _reduction_${name}(s_var=${s_idx}, dt=${repr(dt)}, skip=0, warm_history=None, progress=False):
    # warm_history/progress are accepted-and-ignored so the recurrence and convolution
    # factories share ONE call site (the convolution reducer uses both; a recurrence does not).
% if _rpars:
    # Observer constants (Dynamics.parameters), bound by name in the closure the
    # init/update/finalize triple shares. A literal inlines; a sourced/produced operator
    # was materialised to a companion at codegen and is read once here (a large array
    # never enters this source), becoming a captured constant for the traced reduction.
% for _pname, _pdef in _rpars.items():
% if _pdef.get('lazy'):
    ${_pname} = _load_constant(${repr(_pdef['lazy'][0])}, ${repr(_pdef['lazy'][1])})
% elif 'value' in _pdef:
    ${_pname} = ${render_jax_default(_pdef['value'])}
% else:
<% raise ValueError("observer constant %r reached render unmaterialised; call resolve_reduction(obs, experiment) so a sourced/produced constant is written before emission" % _pname) %>
% endif
% endfor
% endif
% for _fname, _fdef in red['functions'].items():
    def ${_fname}(${", ".join(_fdef['args'])}):
        return ${_jc(_fdef['expr'], _fdef['args'])}
% endfor
% if _dconsts:
    # Observer constants derived from the parameters above (Dynamics.derived_parameters).
    # Bound once here, not per step — they cannot depend on a state or the observed signal.
% for _d in _dconsts:
    ${_d['name']} = ${_jc(_d['expr'])}
% endfor
% endif
% if _is_median:
    # Streaming median: fold a per-node histogram of the per-step output into the carry
    # (O(bins) memory, no trajectory) and read the 0.5 quantile at finalize — the robust
    # statistic a running sum cannot give. Memory states advance every step; the histogram
    # only accumulates after the warmup (_gstep > skip).
    _hlo, _hhi, _hbins = ${_h['lo']}, ${_h['hi']}, ${_h['bins']}
    _hbw = (_hhi - _hlo) / _hbins
    def _init(template, n_steps):
        n = template.shape[-1]
        return (${_mem_ini}jnp.zeros((_hbins, n)), jnp.array(0))
    def _update(acc, block):
        def _step(carry, s_row):
            ${_mem_pre}_counts, _gstep = carry
            ${_src} = s_row[s_var]
            _accumulate = _gstep ${_gate} skip
${render_observer_dvs(_step_dvs, _jc, _ind)}\
            _q_step = ${_jc(red['output'])}
            _b = jnp.clip(((_q_step - _hlo) / _hbw).astype(jnp.int32), 0, _hbins - 1)
            _counts = _counts.at[_b, jnp.arange(_counts.shape[1])].add(jnp.where(_accumulate, 1.0, 0.0))
${render_observer_states(_mem, _jc, _ind)}\
            return (${_mem_new}_counts, _gstep + 1), None
        return jax.lax.scan(_step, acc, block)[0]
    def _finalize(acc):
        ${_mem_pre}_counts, _gstep = acc
        _total = _counts.sum(0)
        _cum = jnp.cumsum(_counts, 0)
        _target = _total * 0.5   # median = 0.5 quantile
        _bi = jnp.clip(jnp.sum(_cum < _target[None, :], axis=0), 0, _hbins - 1)
        _cb = jnp.where(_bi > 0, jnp.take_along_axis(_cum, jnp.maximum(_bi - 1, 0)[None, :], 0)[0], 0.0)
        _frac = (_target - _cb) / jnp.maximum(jnp.take_along_axis(_counts, _bi[None, :], 0)[0], 1.0)
        return _hlo + (_bi + _frac) * _hbw
    return (_init, _update, _finalize)
% elif _period_steps:
    # Monitor form (Observation.period): emit the observer's output every _period steps into a preallocated series; states advance every step and `skip` drops the leading transient samples at finalize.
<% _out_per_step = red.get('output_per_step', True) %>\
    _period = ${_period_steps}
    def _init(template, n_steps):
        n = template.shape[-1]
        return (${", ".join("jnp.full((n,), %r)" % s['init'] for s in red['states'])},
                ${"jnp.zeros((n,)), " if _out_per_step else ""}jnp.zeros((n_steps // _period, n)), jnp.array(0))
    def _update(acc, block):
        *_st, _out, _count = acc
        def _step(carry, s_row):
            ${", ".join(_snames)}${", _emit" if _out_per_step else ","} = carry
            ${_src} = s_row[s_var]
${render_observer_dvs(_step_dvs, _jc, _ind)}\
${render_observer_states(red['states'], _jc, _ind)}\
% if _out_per_step:
            # readout evaluated from the advanced state each step, carried in one [n] slot (per-step values never stacked)
            ${", ".join(_snames)} = ${", ".join("_new_%s" % _n for _n in _snames)}
            return (${", ".join(_snames)}, ${_jc(red['output'])}), None
% else:
            return (${", ".join("_new_%s" % _n for _n in _snames)},), None
% endif
        def _chunk(carry, rows):
            carry = jax.lax.scan(_step, carry, rows)[0]
% if _out_per_step:
            return carry, carry[-1]
% else:
            # readout is a function of the states alone: evaluated once per emitted sample, not per step
            ${", ".join(_snames)}, = carry
            return carry, ${_jc(red['output'])}
% endif
        _n_full, _rem = divmod(block.shape[0], _period)
        if _rem and _n_full:
            # a block longer than one period must be a whole number of periods, else its samples shift off the global grid; a sub-period block is the final tail (handled below)
            raise ValueError(
                f"monitor reducer: a {block.shape[0]}-step block is not a whole number of "
                f"{_period}-step emission periods; size blocks from period_in_steps.")
        _chunks = block[:_n_full * _period].reshape(_n_full, _period, *block.shape[1:])
        _st, _samples = jax.lax.scan(_chunk, tuple(_st), _chunks)
        _out = jax.lax.dynamic_update_slice(_out, _samples, (_count, 0))
        if _rem:
            # A short tail block emits nothing; its steps still advance the observer.
            _st = jax.lax.scan(_step, _st, block[_n_full * _period:])[0]
        return (*_st, _out, _count + _n_full)
    def _finalize(acc):
        return acc[-2][skip // _period:]   # the sample buffer, past the transient
    return (_init, _update, _finalize)
% else:
    def _init(template, n_steps):
        n = template.shape[-1]
        return (${", ".join("jnp.full((n,), %r)" % s['init'] for s in red['states'])}, jnp.array(0), jnp.array(0))
    def _update(acc, block):
        def _step(carry, s_row):
            ${", ".join(_snames)}, _count, _gstep = carry
            ${_src} = s_row[s_var]
            _accumulate = _gstep ${_gate} skip
${render_observer_dvs(_step_dvs, _jc, _ind)}\
${render_observer_states(red['states'], _jc, _ind)}\
% for s in red['states']:
% if s['is_accumulator']:
            _new_${s['name']} = jnp.where(_accumulate, _new_${s['name']}, ${s['name']})
% endif
% endfor
            _count = _count + jnp.where(_accumulate, 1, 0)
            return (${", ".join("_new_%s" % _n for _n in _snames)}, _count, _gstep + 1), None
        return jax.lax.scan(_step, acc, block)[0]
    def _finalize(acc):
        ${", ".join(_snames)}, count, _gstep = acc
        return ${_jc(red['output'])}
    return (_init, _update, _finalize)
% endif
</%def>\
## Grouped wave-metrics reducer (Observation whose value collapses BOTH time and the node axis
## into per-group scalars — the cortical wave detector: proportion_waves, proportion_directed,
## rho per hemisphere). Unlike the per-node recurrence, the output is keyed by (group, metric),
## not node, so it cannot come from `template.shape[-1]`. The heavy per-emitted-step math
## (gradient → angular-similarity → surrogate → HHD → correlation) is the SAME declarative DV
## chain the other observers use — incl. the permutation surrogate (render_surrogate) — and
## produces one (n_groups,) vector per named output (`corr`, `wave_present`, `sig_corr`). Only
## the outer carry is bespoke: a monitor-style (n_ds, n_groups) buffer per output (n_ds is the
## downsampled sample count — small, so buffering is cheap and duration is never materialised at
## node resolution), reduced at finalize to the three metrics via exact masked statistics
## (`nanmedian` over wave-present samples is Koller's rho, no binning). One block per grid cell;
## blocks are period-aligned (streaming_post_eval_plan), matching the stride reducer.
## TRAVELING-WAVE GATE (active iff the chain has pgn0/1/2 and a `real_face_mask` param): a standing
## field passes Koller's per-frame test yet does not travel, so the carry also holds a running
## O(faces) post-transient sum of the per-face unit gradient direction, and finalize zeroes a
## group's proportion_waves when its direction dispersion mean_faces(1-|time-mean dir|) < 0.06. The
## non-gated path is byte-identical; byte-consistent with the host cortical_wave_metrics gate.
<%def name="render_wave_reduction(red, name, s_idx, dt)">\
<%
    from tvbo.codegen import render_expression
    from tvbo.templates.tvboptim.utils import render_jax_default
    _G = red['n_groups']
    _period = red['period_steps']
    _src = red['source']
    _rpars = red.get('parameters') or {}
    _derived = red.get('derived', [])   # per-emitted-step DV chain producing the named outputs
    _corr, _wave, _sig = red['corr'], red['wave_present'], red['sig_corr']
    _gv = red.get('group_vmap')   # {gather: <param>, over: [<group-indexed params>]} or None
    _rparams = ([d['name'] for d in _derived] + list(_rpars) + [_src, 'dt'])
    _rufuncs = {f: f for f in red.get('functions', {})}
    _jc = lambda e: render_expression(e, format='jax', user_functions=_rufuncs, parameters=_rparams)
    # Traveling-wave gate (see render_wave_reduction docstring): active iff the chain exposes pgn0/1/2 and a real_face_mask param.
    _dvnames = [d['name'] for d in _derived]
    _gate = ('real_face_mask' in _rpars) and all(('pgn%d' % _k) in _dvnames for _k in range(3))
    _pthr = 0.06
%>\
def _reduction_${name}(s_var=${s_idx}, dt=${repr(dt)}, skip=0, warm_history=None, progress=False):
    # warm_history/progress are accepted-and-ignored so this reducer shares ONE call site with
    # the recurrence, convolution, comoment and stride factories.
% if _rpars:
    # Observer constants (operators / permutation table) — a literal inlines, a produced/sourced
    # operator is read once here as a captured constant (a large array never enters this source).
% for _pname, _pdef in _rpars.items():
% if _pdef.get('lazy'):
    ${_pname} = _load_constant(${repr(_pdef['lazy'][0])}, ${repr(_pdef['lazy'][1])})
% elif 'value' in _pdef:
    ${_pname} = ${render_jax_default(_pdef['value'])}
% else:
<% raise ValueError("wave observer constant %r reached render unmaterialised; call resolve_reduction(obs, experiment)" % _pname) %>
% endif
% endfor
% endif
% for _fname, _fdef in red.get('functions', {}).items():
    def ${_fname}(${", ".join(_fdef['args'])}):
        return ${_jc(_fdef['expr'])}
% endfor
    _period = ${_period}
% if _gv:
    # Grouped detector: the per-timestep body is written ONCE for a single group (1-D, exactly
    # detect_matmul + the 1-D surrogates) and vmapped over the partition axis. The group-indexed
    # operators (`over`) carry a leading group axis and are sliced per call; `gather` selects each
    # group's vertices out of the whole-brain source. General to any partition; the surrogate
    # stays a clean per-vertex max-T inside the vmap.
    def _detect(${", ".join([_src] + _gv['over'])}):
${render_observer_dvs(_derived, _jc, ' ' * 8)}\
        return ${_corr}, ${_wave} * 1.0, ${_sig} * 1.0${', pgn0, pgn1, pgn2' if _gate else ''}
    def _sample(_theta_all):
        ${_src}_g = _theta_all[${_gv['gather']}]   # (n_groups, nv) per-group vertex gather
        return jax.vmap(_detect, in_axes=(0,) * ${1 + len(_gv['over'])})(${_src}_g, ${", ".join(_gv['over'])})
% else:
    def _sample(${_src}):
        # one downsampled sample -> (n_groups,) per named output; a body that is already
        # group-batched (operators with a leading group axis) produces the per-group vector directly.
${render_observer_dvs(_derived, _jc, ' ' * 8)}\
        return ${_corr}, ${_wave} * 1.0, ${_sig} * 1.0${', pgn0, pgn1, pgn2' if _gate else ''}
% endif
    def _init(template, n_steps):
        _n_ds = len(range(_period - 1, n_steps, _period))
        _z = jnp.zeros((_n_ds, ${_G}))
% if _gate:
        return (_z, _z, _z, jnp.array(0), jnp.zeros((${_G}, 3, real_face_mask.shape[-1])), jnp.array(0.0))
% else:
        return (_z, _z, _z, jnp.array(0))
% endif
    def _update(acc, block):
% if _gate:
        _corr_buf, _wave_buf, _sig_buf, _count, _dir, _dcnt = acc
% else:
        _corr_buf, _wave_buf, _sig_buf, _count = acc
% endif
        if block.shape[0] % _period:
            # A block must be a whole number of downsample periods, else its samples shift off
            # the global grid and the write row (_count // _period) drifts — the same alignment
            # the stride/monitor reducers require; size streaming blocks from period_in_steps.
            raise ValueError(
                f"wave reducer: a {block.shape[0]}-step block is not a whole number of "
                f"{_period}-step downsample periods; size streaming blocks from period_in_steps.")
        _theta = block[_period - 1 :: _period, s_var, :]        # downsample to (_m, n)
% if _gate:
        _c, _w, _s, _p0, _p1, _p2 = jax.vmap(_sample)(_theta)   # scalars (_m, G); unit gradient dir (_m, G, nf)
% else:
        _c, _w, _s = jax.vmap(_sample)(_theta)                  # batched update over the block's frames; block_size bounds the batch (tvboptim's native per-block fold)
% endif
        _row = _count // _period
        _corr_buf = jax.lax.dynamic_update_slice(_corr_buf, _c, (_row, 0))
        _wave_buf = jax.lax.dynamic_update_slice(_wave_buf, _w, (_row, 0))
        _sig_buf = jax.lax.dynamic_update_slice(_sig_buf, _s, (_row, 0))
% if _gate:
        # running post-transient sum of the per-face unit gradient direction (O(faces), no per-frame buffer)
        _keepm = (_row + jnp.arange(_theta.shape[0]) >= (skip // _period)).astype(_p0.dtype)
        _dir = _dir + jnp.sum(_keepm[:, None, None, None] * jnp.stack([_p0, _p1, _p2], axis=2), axis=0)
        _dcnt = _dcnt + jnp.sum(_keepm)
        return (_corr_buf, _wave_buf, _sig_buf, _count + block.shape[0], _dir, _dcnt)
% else:
        return (_corr_buf, _wave_buf, _sig_buf, _count + block.shape[0])
% endif
    def _finalize(acc):
% if _gate:
        _corr_buf, _wave_buf, _sig_buf, _count, _dir, _dcnt = acc
% else:
        _corr_buf, _wave_buf, _sig_buf, _count = acc
% endif
        _keep = skip // _period                                 # drop the transient samples
        _corr_buf, _wave_buf, _sig_buf = _corr_buf[_keep:], _wave_buf[_keep:], _sig_buf[_keep:]
        _nw = _wave_buf.sum(0)                                   # (n_groups,) wave-present count
% if _gate:
        # traveling-wave gate: direction-dispersion over real faces; standing field (< _pthr) -> zero wave count
        _Rf = jnp.linalg.norm(_dir / jnp.maximum(_dcnt, 1.0), axis=1)                       # (G, nf)
        _dd = jnp.sum(real_face_mask * (1.0 - _Rf), axis=1) / jnp.sum(real_face_mask, axis=1)
        _nw = jnp.where(_dd >= ${_pthr}, _nw, 0.0)              # standing -> no traveling waves
% endif
        _pw = _nw / _wave_buf.shape[0]                           # proportion of waves
        _pd = jnp.where(_nw > 0, (_sig_buf * _wave_buf).sum(0) / _nw, jnp.nan)  # proportion directed
% if _gate:
        _rho = jnp.where(_nw > 0, jnp.nanmedian(jnp.where(_wave_buf > 0, _corr_buf, jnp.nan), axis=0), jnp.nan)
% else:
        _rho = jnp.nanmedian(jnp.where(_wave_buf > 0, _corr_buf, jnp.nan), axis=0)  # exact masked median
% endif
        return jnp.stack([_pw, _pd, _rho], axis=-1)             # (n_groups, metric=3)
    return (_init, _update, _finalize)
</%def>\
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


def _load_constant(path, key):
    """Load a lazily-stored observer constant (a sourced/produced operator too large to
    inline) as a jax array. Read once when the reducer is built, so it is captured as a
    concrete constant in the traced update — never re-read per step. Reuses tvbo's
    array store, so an h5 or zarr companion is read the same way the network is.

    A packed kit stages these constants into its own ``constants/`` dir, so when the author's
    absolute path is absent (a frozen kit run on another machine) the file is resolved by
    basename under ``$TVBO_CONSTANTS_DIR`` or the run dir's ``constants/``."""
    import os
    from pathlib import Path
    from tvbo.data.matrix_io import LazyArrayStore
    p = Path(path)
    if not p.exists():
        for _base in (os.environ.get("TVBO_CONSTANTS_DIR"), "constants"):
            if _base and (Path(_base) / p.name).is_file():
                p = Path(_base) / p.name
                break
    return jnp.asarray(LazyArrayStore(p, {}).read_dataset(key))


% if network_obs_keys and bids_dir:
from tvbo.classes.network import Network as _TvboNetwork

_bids_network = _TvboNetwork.from_bids('${bids_dir}', observational_measures=${list(network_obs_keys)})
% endif

% if network_edge_arrays:
# Connectome matrices referenced by observations via network.weight(s)/length(s) or
# network.edges.<label> (embedded once; shared by observation sources and callable args).
% for _label in sorted(network_edge_arrays):
${_edge_const(_label)} = jnp.array(${repr(network_edge_arrays[_label])})
% endfor
% endif

% if network_node_arrays:
# Per-node vectors referenced by observations via network.positions/instrength
# (embedded once; shared by observation sources, callable args, and observer parameters).
% for _label in sorted(network_node_arrays):
${_node_const(_label)} = jnp.array(${repr(network_node_arrays[_label])})
% endfor
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

    # Check if source is a connectome matrix — bare network.weight(s)/length(s)
    # shortcut or explicit network.edges.<label>; resolves to the shared embedded
    # constant (network_edge_arrays / _edge_const). Only short-circuits to the raw
    # matrix when there is NO pipeline; a source matrix WITH a pipeline (e.g.
    # betweenness_centrality(sc=network.weight)) falls through to the pipeline path,
    # where the matrix reaches the callable as an argument via ref_to_code.
    _src_type, _src_val = parse_reference(obs_source) if obs_source else (None, None)
    network_edge_label = _edge_label(_src_val) if _src_type == 'network' else None
    is_network_edge = network_edge_label is not None and not pipeline

    # A per-subject dataset target (dataset.subject.<measure>) is materialized as a
    # module-level constant and bound at run_experiment time by
    # _bind_network_observations, so this template emits no monitor for it.
    is_dataset_target = obs_source and str(obs_source).startswith('dataset.subject.')

    # Resolve source to its column in the recorded variable layout (states + recorded aux).
    # Returns 0 for network/dataset/external/empty sources (back-compat for external monitors).
    state_idx = resolve_var_index(obs_source, obs_name)
%>
% if obs['reduction'] is not None:
## =============================================================================
## Streaming reducer for the Observation.dynamics observer `${obs_name}`.
## The (init, update, finalize) triple serves both paths: the host/base run passes
## the whole trajectory as one block (update scans it → post-scan value), while the
## exploration grid passes it to prepare(reduce=...) to fold the value in-carry with
## no trajectory held. `_reduction_${obs_name}(s_var, dt)` builds a fresh triple.
## =============================================================================
${render_reduction(obs['reduction'], obs_name, state_idx, dt)}
% endif
% if is_dataset_target:
## =============================================================================
## Per-subject dataset target — bound at run_experiment time (see
## _bind_network_observations); no monitor emitted here.
## =============================================================================
# ${obs['label'] or obs_name} <- dataset.subject.${str(obs_source).split('.')[-1]} (runtime-bound)

% elif is_network_edge and network_edge_label in network_edge_arrays:
## =============================================================================
## Static Network Edge Data (embedded connectome matrix — shared constant)
## =============================================================================
# ${obs['label'] or obs_name} - edge data from network
${obs_name} = ${_edge_const(network_edge_label)}

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

% elif obs['reduction'] is not None:
## =============================================================================
## Dynamics-observer Observation `${obs_name}` — host monitor backed by the
## streaming reducer `_reduction_${obs_name}`. The host/base run folds the reducer
## over the whole trajectory as a single block, which equals the in-carry value the
## exploration grid streams via prepare(reduce=...). There is no pipeline: the
## Observation.dynamics observer *is* the definition, one source of truth for both.
## =============================================================================
class ${class_name}(AbstractMonitor):
    """${obs['label'] or obs_name} observation (dynamics observer)."""
    dt: float = eqx.field(static=True, default=${dt})
% if obs['reduction'].get('kind') == 'convolution':
    _history: jax.Array = None

    def __init__(self, history=None, voi: int = ${state_idx}, period: float = None, dt: float = ${dt}, **kwargs):
        self.voi = voi
        self.period = period if period is not None else dt
        self.dt = dt
        if history is not None:
            _h = history.data if hasattr(history, 'data') else history
            # Source-column transient, decimated to the downsampled grid — the HRF warm-up
            # the reducer seeds its ring with (matches the pipeline's prepend_history).
            self._history = _h[:, ${state_idx}, :][${obs['reduction']['ds_steps'] - 1}::${obs['reduction']['ds_steps']}]
        else:
            self._history = None

    def __call__(self, result):
        # One big block over the whole trajectory == the in-carry value the grid streams;
        # seeded with the downsampled transient so the HRF warm-up matches the pipeline.
        _init, _update, _finalize = _reduction_${obs_name}(s_var=${state_idx}, dt=self.dt, warm_history=self._history)
        _data = result.data if hasattr(result, 'data') else result
        _acc = _init(_data[0], _data.shape[0])
        _acc = _update(_acc, _data)
        return _finalize(_acc)
% else:

<%
    # a monitor reports at its declared period; a folded statistic collapses to one value (dt)
    _obs_period = repr(float(to_numeric(obs['period']))) if obs['reduction'].get('kind') == 'monitor' else 'dt'
%>\
    def __init__(self, history=None, voi: int = ${state_idx}, period: float = None, dt: float = ${dt}, **kwargs):
        self.voi = voi
        self.period = period if period is not None else ${_obs_period}
        self.dt = dt

    def __call__(self, result):
        # One big block over the whole trajectory == the post-scan reduction.
        _init, _update, _finalize = _reduction_${obs_name}(s_var=${state_idx}, dt=self.dt)
        _data = result.data if hasattr(result, 'data') else result
        _acc = _init(_data[0], _data.shape[0])
        _acc = _update(_acc, _data)
        return _finalize(_acc)
% endif

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
    # Build default args from function definition (arguments keyed by name)
    default_args = []
    for arg_name, arg in (get_attr(fn_def, 'arguments') or {}).items():
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
% if network_edge_label:
        # Source is a connectome matrix (network.${obs_source}); the embedded constant
        # is the pipeline input, not a simulation trajectory. An argument naming the
        # source (e.g. sc=network.weight) is bound to `_data` below.
        _data = ${_edge_const(network_edge_label)}
        _time = result.time
% elif obs_source:
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
    # This observation's source state variable(s): an argument value naming one of
    # them refers to the source time series (bound above as `_data`), not a string.
    _src_vars = [str(s) for s in (obs_source if isinstance(obs_source, (list, tuple)) else [obs_source])] if obs_source else []

    # Build call arguments
    call_parts = []
    for arg_name, arg_val in args.items():
        if isinstance(arg_val, str):
            if arg_val in _src_vars:
                # Reference to this observation's SOURCE state variable → its data slice
                # (result.data[:, self.voi, :], bound above as `_data`).
                call_parts.append(f"{arg_name}=_data")
            elif arg_val in step_names or arg_val == 'data':
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
            elif _node_label(arg_val):
                # network.positions / network.instrength → the per-node vector embedded once
                # as a module constant (network_node_arrays), so a host callable receives the
                # region centroids / in-strength. Without this the ref falls through to the
                # string-literal branch below and the callable gets the raw 'network.positions'.
                call_parts.append(f"{arg_name}={_node_const(_node_label(arg_val))}")
            elif _edge_label(arg_val):
                # network.weight(s)/length(s)/edges.<label> → the embedded connectome matrix.
                call_parts.append(f"{arg_name}={_edge_const(_edge_label(arg_val))}")
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
% if aggregation:
        # Track whether a collapsing aggregation removed the time axis. A future
        # aggregation only needs to set this flag to be handled correctly below.
        _aggregated = False
% endif
% if str(aggregation) == 'mean':
        # Declarative: mean over time axis (aggregation: mean)
        ${decl_input} = jnp.mean(${decl_input}, axis=0)
        _aggregated = True
% elif str(aggregation) == 'variance':
        # Declarative: variance over time axis (aggregation: variance)
        ${decl_input} = jnp.var(${decl_input}, axis=0)
        _aggregated = True
% elif str(aggregation) == 'std':
        # Declarative: standard deviation over time axis (aggregation: std)
        ${decl_input} = jnp.std(${decl_input}, axis=0)
        _aggregated = True
% elif str(aggregation) == 'first_passage':
<%
    _fp_thr = (obs.get('agg_params') or {}).get('threshold')
    if _fp_thr is None:
        raise ValueError(f"Observation '{obs.get('name')}' uses aggregation: first_passage "
                         f"but has no parameters.threshold to cross.")
%>
        # Declarative: first-passage index over the time axis — the first sample
        # where the source crosses the threshold (>=), or the sample count if it
        # never crosses. Backend-independent (argmax of the threshold-crossing).
        _fp_cross = ${decl_input} >= ${_fp_thr}
        ${decl_input} = jnp.where(jnp.any(_fp_cross, axis=0),
                                  jnp.argmax(_fp_cross, axis=0), _fp_cross.shape[0])
        _aggregated = True
% elif str(aggregation) == 'last':
        ${decl_input} = ${decl_input}[-1]
        _aggregated = True
% elif str(aggregation) == 'first':
        ${decl_input} = ${decl_input}[0]
        _aggregated = True
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
% if aggregation:
        if _aggregated:
            # A collapsing aggregation removed the time axis: the result is a
            # plain per-node value (or scalar), so return it directly instead of
            # re-wrapping it in a NativeSolution — downstream loss / algorithm
            # arithmetic needs an array, not a solution object (else e.g.
            # `mean_activity + target` raises "unsupported operand type(s) for
            # +: 'float' and 'NativeSolution'").
            return _final
% endif

        _is_scalar = _final.ndim == 0 if hasattr(_final, 'ndim') else True
        if _is_scalar:
            _ts = None  # Scalar result has no time dimension
            _out_dt = self.dt
        elif _time is not None and len(_time) == len(_final):
            _ts = _time
            _out_dt = self.dt
<%
    # Declared output sampling period (e.g. BOLD TR). Used to label a subsampled
    # time axis at the true period instead of stretching it across the recorded span.
    _decl_period = obs.get('period')
    try:
        _decl_period = float(_decl_period)
    except (TypeError, ValueError):
        _decl_period = None
%>
% if _decl_period and _decl_period > float(dt):
        elif _time is not None and len(_time) > len(_final) > 1:
            # The pipeline subsampled to the declared output period (e.g. BOLD TR):
            # place samples at that exact period from the recording start, so the
            # time axis is not stretched across the full integration span.
            _ts = _time[0] + jnp.arange(len(_final)) * ${_decl_period}
            _out_dt = ${_decl_period}
% else:
        elif _time is not None and len(_time) > len(_final) > 1:
            # The pipeline subsampled the time series but declares no output period;
            # spread the outputs uniformly across the recorded span as a fallback.
            _ts = jnp.linspace(_time[0], _time[-1], len(_final))
            _out_dt = (_time[-1] - _time[0]) / (len(_final) - 1)
% endif
        else:
            _ts = jnp.arange(len(_final)) * self.dt
            _out_dt = self.dt
% if named_outputs:
        return ObservationResult(
            ts=_ts,
            ys=_final,
            dt=_out_dt,
% for out_name in named_outputs:
            ${out_name}=_${out_name},
% endfor
        )
% else:
        return NativeSolution(ts=_ts if _ts is not None else jnp.array([0.0]), ys=_final, dt=_out_dt)
% endif

% endif
% endfor
<%
    # Registry of every Observation.dynamics observer: name -> (reducer_factory, source
    # column index). The exploration grid uses it to stream recorded observables via
    # prepare(reduce=...) with no trajectory held (see the experiment template).
    _streaming = [(o['name'], resolve_var_index(o['source'], o['name']))
                  for o in obs_list if o.get('reduction') is not None]
%>
% if _streaming:


def _compose_reducers(*reducers):
    """Fuse several (init, update, finalize) triples into one whose carry is the tuple
    of per-reducer carries; finalize returns the tuple of per-reducer values. Each
    reducer scans the block independently — no cross-talk, order matches ``reducers``."""
    def _init(template, n_steps):
        return tuple(_r[0](template, n_steps) for _r in reducers)
    def _update(accs, block):
        return tuple(_r[1](_a, block) for _r, _a in zip(reducers, accs))
    def _finalize(accs):
        return tuple(_r[2](_a) for _r, _a in zip(reducers, accs))
    return (_init, _update, _finalize)


# name -> (reducer_factory, source column index) for every Observation.dynamics observer.
_STREAMING_REDUCERS = {
% for _sname, _sidx in _streaming:
    ${repr(_sname)}: (_reduction_${_sname}, ${_sidx}),
% endfor
}
% endif
