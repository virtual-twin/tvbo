<%!
from tvbo.codegen import render_expression

def get_callable_info(func):
    return func.callable.module, func.callable.name

def get_parameters(func, pipeline_outputs, obs_period=None, dt=None, sampling_overrides=None):
    """Return {arg_name: value_str} for arguments that have a resolved value.

    Arguments without an explicit value are skipped — they appear as required
    positional parameters in the generated function signature (see
    generate_function in jax-function.py.mako) and must be passed at the call
    site.

    ``obs_period`` and ``dt`` are used to derive windowing arguments
    (window_size, window) when the YAML omits an explicit value.

    ``sampling_overrides`` maps ``arg_name -> resolved integer`` for temporal
    sampling steps (interim window / subsample count). These are resolved once,
    backend-independently, by :func:`tvbo.adapters.observation_sampling.
    resolve_observation_sampling` (injected via the render adapter) and take
    precedence over any YAML literal so the template only emits resolved values.
    """
    _WINDOW_ARGS = ('window_size', 'window')
    sampling_overrides = sampling_overrides or {}

    params = {}
    args = func.arguments or []
    # Handle dict-keyed arguments (from Function) or list (from FunctionCall)
    if hasattr(args, 'values'):
        args = args.values()
    for arg in args:
        # Backend-shared resolved sampling counts win over the YAML literal.
        if arg.name in sampling_overrides:
            params[arg.name] = str(int(sampling_overrides[arg.name]))
            continue
        value = getattr(arg, 'value', None)
        if value is None:
            # Derive windowing parameters from observation period and integration dt
            if obs_period is not None and dt is not None and arg.name in _WINDOW_ARGS:
                value = str(int(round(float(obs_period) / float(dt))))
        if value is None:
            continue
        # Quote string values that aren't numeric or pipeline references
        if isinstance(value, str):
            is_numeric = value.replace('.','').replace('-','').isdigit()
            is_pipeline_ref = value in pipeline_outputs
            if not is_numeric and not is_pipeline_ref:
                value = f"'{value}'"
        params[arg.name] = value
    return params

def get_equation_rhs(func):
    return func.equation.rhs

# Argument names that carry a temporal-sampling step count.
_SAMPLING_ARG_NAMES = ('stepsize', 'step', 'window_size', 'window')


def _sampling_kind(func):
    """Classify a pipeline step as a temporal-sampling stage.

    Returns ``'window'`` for an averaging window (``window_mean``),
    ``'subsample'`` for a decimation (``subsample``), else ``None``.
    """
    equation = getattr(func, 'equation', None)
    rhs = str(getattr(equation, 'rhs', '') or '')
    if 'window_mean(' in rhs:
        return 'window'
    if 'subsample(' in rhs:
        return 'subsample'
    return None


def _sampling_arg_names(func):
    """Names of the step-size arguments on a sampling step."""
    args = getattr(func, 'arguments', None) or {}
    names = list(args.keys()) if hasattr(args, 'keys') else [getattr(a, 'name', None) for a in args]
    return [n for n in names if n in _SAMPLING_ARG_NAMES]


def build_sampling_overrides(resolved_pipeline, sampling):
    """Resolve sampling-step arguments to integer step counts.

    Uses the backend-shared :class:`ObservationSampling` (``sampling``) so the
    jax pipeline downsamples by exactly the same counts as tvboptim/tvb. The
    mapping from stage to count is derived from the pipeline structure, not from
    any raw-grid YAML literal:

      * two-stage (interim ``window_mean`` then ``subsample``, e.g. BOLD):
        window -> ``interim_istep``, subsample -> ``output_interim_count``
        (net factor = ``output_istep``);
      * single ``window_mean`` (e.g. TemporalAverage) -> ``output_istep``;
      * single ``subsample`` (e.g. SubSample)        -> ``output_istep``.

    Returns ``{id(func): {arg_name: int}}``. Empty when no sampling info.
    """
    if sampling is None:
        return {}

    stages = [(func, _sampling_kind(func)) for func in resolved_pipeline]
    stages = [(func, kind) for func, kind in stages if kind is not None]
    has_window = any(kind == 'window' for _, kind in stages)
    has_subsample = any(kind == 'subsample' for _, kind in stages)
    two_stage = has_window and has_subsample

    overrides = {}
    for func, kind in stages:
        if two_stage:
            count = sampling.interim_istep if kind == 'window' else sampling.output_interim_count
        else:
            count = sampling.output_istep
        sampling_args = _sampling_arg_names(func)
        if sampling_args and int(count) <= 0:
            # Without this the pipeline emits `X[::0]`, which fails far from the missing declaration.
            raise ValueError(
                f"observation pipeline step {getattr(func, 'name', func)!r} needs a "
                f"sampling stride, but the observation declares no output period. "
                f"Add `period: <ms>` (or a `TR` parameter) to the observation so the "
                f"stride resolves from the integration step size."
            )
        arg_map = {name: int(count) for name in sampling_args}
        # TVB emits the first subsample at step % istep == 0 (1-indexed), i.e.
        # array index istep - 1. Honour an explicit ``start`` argument so the jax
        # subsample aligns with tvboptim/tvb instead of starting at index 0.
        arg_names = getattr(func, 'arguments', None) or {}
        arg_names = list(arg_names.keys()) if hasattr(arg_names, 'keys') else [getattr(a, 'name', None) for a in arg_names]
        if kind == 'subsample' and 'start' in arg_names:
            arg_map['start'] = int(count) - 1
        if arg_map:
            overrides[id(func)] = arg_map
    return overrides

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
<%def name="create_observation_pipeline(observation, dt, functions=None, obs_sampling=None)" filter="trim">
<%
    # Helper to get function name from FunctionCall or resolved Function
    def get_func_name(func):
        if hasattr(func, 'function') and func.function:
            return str(func.function)
        if hasattr(func, 'name') and func.name:
            return str(func.name)
        if hasattr(func, 'callable') and func.callable and hasattr(func.callable, 'name'):
            return str(func.callable.name)
        return None

    # Resolve function references: if a FunctionCall has only function: name,
    # look up the actual Function definition from experiment.functions
    resolved_pipeline = []
    for fc in observation.pipeline:
        fname = str(fc.function) if fc.function else None
        if fname and functions and fname in functions:
            # Use the resolved Function object for code generation,
            # but carry over FunctionCall-specific fields (output, arguments, apply_on_dimension)
            resolved = functions[fname]
            # If FunctionCall specifies output, use it (override Function default)
            if fc.output and not getattr(resolved, 'output', None):
                resolved.output = fc.output
            # If FunctionCall specifies arguments, merge them
            if fc.arguments:
                resolved._fc_arguments = fc.arguments
            resolved_pipeline.append(resolved)
        else:
            resolved_pipeline.append(fc)

    if not resolved_pipeline:
        raise ValueError(
            f"Observation {observation.name!r} declares no pipeline. The jax backend "
            "renders observations as a post-scan pipeline and has no path for an "
            "Observation.dynamics observer or a bare class_reference; export it with the "
            "tvboptim backend, which folds an observer into the integrator carry."
        )

    func_name_to_output = {get_func_name(func): func.output for func in resolved_pipeline if get_func_name(func)}
    # `output:` is optional; an unnamed step needs a name anyway (`None = f(...)` is not Python).
    step_names = [f.output or f"_step{i}" for i, f in enumerate(resolved_pipeline)]
    # Resolve temporal-sampling step counts once, backend-independently.
    sampling_overrides_map = build_sampling_overrides(resolved_pipeline, obs_sampling)
    # Resolved once for both the definition and the call: a name here is a signature parameter, one absent is bound from the TimeSeries.
    pipeline_outputs = set(f.output for f in resolved_pipeline if f.output)
    params_map = {
        id(func): get_parameters(
            func, pipeline_outputs,
            obs_period=getattr(observation, 'period', None),
            dt=dt,
            sampling_overrides=sampling_overrides_map.get(id(func)),
        )
        for func in resolved_pipeline
    }
    # Collect imports for this observation
    obs_imports = set()
    for func in resolved_pipeline:
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
% for func in resolved_pipeline:
% if func.callable:
_jax_${func.callable.name} = ${func.callable.name}  # Store reference to avoid recursion
% endif
% endfor

% for func in resolved_pipeline:
${jaxfunc.generate_function(func, get_func_name(func), supplied=params_map[id(func)])}

% endfor

<%
    # ``source`` is multivalued; for raw observations there is one entry
    # (a state-variable reference). Take the first scalar value.
    _src = getattr(observation, 'source', None)
    if isinstance(_src, (list, tuple)):
        _src = _src[0] if _src else None
    if _src is not None and hasattr(_src, 'name'):
        _src = _src.name
    _state_default = "'" + str(_src) + "'" if _src else 'None'
%>
# Compose functions into observation pipeline
def ${observation.name}(ts: TimeSeries, state=${_state_default}):
    # Extract state variable if specified
    if state is not None:
        ts = ts.get_state(state)

% for i_step, func in enumerate(resolved_pipeline):
<%
    _func_input = getattr(func, 'input', None)

    if _func_input:
        if _func_input in pipeline_outputs:
            # Direct reference to an output variable name — use as-is.
            input_name = _func_input
        elif _func_input in func_name_to_output:
            # Reference to a *function* name — resolve to its output variable.
            # e.g. YAML says  input: temporal_average_interim
            #       → should use  interim_averaged  (that function's output var)
            input_name = func_name_to_output[_func_input]
        # Check if input matches the observation's source_observation (cross-observation dependency)
        elif hasattr(observation, 'source_observation') and _func_input == observation.source_observation:
            # Need to call the source observation function
            input_name = f"{_func_input}(ts)"
        else:
            # Unknown input - use as variable name
            input_name = _func_input
    elif i_step > 0:
        # Auto-chain: use previous step's result, named or positional
        input_name = step_names[i_step - 1]
    else:
        input_name = 'ts'

    params_dict = params_map[id(func)]
    # Build argument list: input + schema arguments
    args = [input_name]
    for arg_name, arg_value in params_dict.items():
        args.append(f"{arg_name}={arg_value}")
    args_str = ', '.join(args)
%>
    ${step_names[i_step]} = ${get_func_name(func)}(${args_str})
% endfor
    return ${step_names[-1]}
</%def>

% if 'observation' in context.keys():
${create_observation_pipeline(observation, dt if 'dt' in context.keys() else 0.1, obs_sampling=context.get('obs_sampling', {}).get(observation.name) if hasattr(context.get('obs_sampling', {}), 'get') else None)}
% endif

<%def name="create_all_observations(experiment, obs_sampling=None)" filter="trim">
<%
    dt = experiment.integration.step_size
    exp_functions = getattr(experiment, 'functions', None) or {}
    obs_sampling = obs_sampling or {}
%>
% for name, obs in experiment.observations.items():

${create_observation_pipeline(obs, dt, functions=exp_functions, obs_sampling=obs_sampling.get(name))}

% endfor

def apply_observations(trace, time_steps, dt=${dt}, params=None):
    results = {}
    % for name in experiment.observations.keys():
    results['${name}'] = observe_${name}(trace, time_steps, dt=dt, params=params)
    % endfor
    return results

</%def>
