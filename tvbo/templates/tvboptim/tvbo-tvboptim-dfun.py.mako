# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Dynamics (dfun) Template
==================================

Generates an AbstractDynamics subclass for tvboptim.experimental.network_dynamics.

Context Variables:
- experiment: SimulationExperiment instance (optional)
- model: Dynamics instance (required if no experiment)

Output:
- Python class inheriting from AbstractDynamics
</%doc>
<%namespace name="fn" file="/base/function-def.mako"/>
<%
import textwrap
from tvbo.codegen import render_expression
from tvbo.templates.tvboptim.utils import get_param_info

# Get model from context
if 'experiment' in context.keys():
    model = experiment.dynamics
    # Also collect experiment-level functions if available
    _exp_functions = getattr(experiment, 'functions', None) or {}
else:
    model = context['model']
    _exp_functions = {}

# Collect user-defined functions from model.functions and experiment.functions
# These are functions defined in YAML that need to be recognized by the code printer.
# Map function name -> function name (identity mapping) so printer emits them as-is.
_model_functions = getattr(model, 'functions', None) or {}
user_functions = {}
if hasattr(_model_functions, 'keys'):
    user_functions.update({str(fname): str(fname) for fname in _model_functions.keys()})
if hasattr(_exp_functions, 'items'):
    user_functions.update({str(fname): str(fname) for fname in _exp_functions.keys()})

jaxcode = lambda expr: render_expression(expr, format='jax', user_functions=user_functions)
jaxcode_obj = lambda obj: model.render_equation(obj, format='jax')

# Extract metadata
state_names = list(model.state_variables.keys())
initial_state = [float(sv.initial_value) if sv.initial_value is not None else 0.0
                 for sv in model.state_variables.values()]

# Determine auxiliary variables from 'output' attribute.
# output is a list of derived_variable names (schema: range: string, multivalued: true).
# State variables in output are silently ignored (already in state array).
output_vars = getattr(model, 'output', None) or []
if isinstance(output_vars, str):
    output_vars = [output_vars]

if output_vars:
    aux_names = [v for v in output_vars if v in (model.derived_variables or {})]
else:
    # Default: all derived variables become auxiliaries
    aux_names = list(model.derived_variables.keys()) if model.derived_variables else []

# Extract parameter info using shared utility
param_names, param_defaults, param_shapes = get_param_info(model.parameters)
derived_param_names = [p.name for p in model.derived_parameters.values()] if model.derived_parameters else []

# Detect parameters with distribution.axis == 'time' — these are stochastic
# time-varying inputs pre-generated as arrays and indexed per integration step.
# This avoids per-step RNG calls (fast) and works with vmap/pmap (pure arrays).
stochastic_params = {}
regular_param_names = []
# Get dt from experiment context (available when included from experiment template)
if 'experiment' in context.keys():
    _stoch_dt = float(experiment.integration.step_size)
else:
    _stoch_dt = 0.001  # fallback for standalone dfun rendering
_stoch_inv_dt = 1.0 / _stoch_dt
for pname in param_names:
    p_obj = (model.parameters[pname] if pname in model.parameters else None) if model.parameters else None
    if p_obj and getattr(p_obj, 'distribution', None):
        dist = p_obj.distribution
        axis = str(getattr(dist, 'axis', 'space'))
        if axis == 'time' or 'time' in axis:
            domain = getattr(dist, 'domain', None)
            dist_name = str(getattr(dist, 'name', 'Uniform')).lower()
            stochastic_params[pname] = {
                'dist': dist_name,
                'lo': float(getattr(domain, 'lo', 0)) if domain else 0.0,
                'hi': float(getattr(domain, 'hi', 1)) if domain else 1.0,
                'default': float(p_obj.value) if p_obj.value is not None else 0.0,
                'seed': int(getattr(dist, 'seed', None) or 42),
                'shape': str(getattr(p_obj, 'shape', '')) if getattr(p_obj, 'shape', None) else '',
            }
            continue
    regular_param_names.append(pname)
param_names = regular_param_names

# Build COUPLING_INPUTS from coupling_inputs
# Pattern: each coupling_input name → its dimension (default 1)
# If dimension > 1 and keys provided, keys are used as variable names
#
# Example 1: instant: {dimension: 1}, delayed: {dimension: 1}
#    → COUPLING_INPUTS = {'instant': 1, 'delayed': 1}
#    → c_instant = coupling.instant[0], c_delayed = coupling.delayed[0]
#
# Example 2: coupling: {dimension: 2, keys: [lre, ffi]}
#    → COUPLING_INPUTS = {'coupling': 2}
#    → lre = coupling.coupling[0], ffi = coupling.coupling[1]
#
coupling_inputs_dict = {}
coupling_keys = {}  # ci_name -> list of key names (for unpacking)

if hasattr(model, 'coupling_inputs') and model.coupling_inputs:
    for ci_name, ci in model.coupling_inputs.items():
        dim = getattr(ci, 'dimension', 1) or 1
        coupling_inputs_dict[ci_name] = dim
        keys = getattr(ci, 'keys', None)
        if keys:
            coupling_keys[ci_name] = list(keys)
elif hasattr(model, 'coupling_terms') and model.coupling_terms:
    # Deprecated fallback: use coupling_terms with dimension 1 each
    for ct_name in model.coupling_terms.keys():
        coupling_inputs_dict[ct_name] = 1

class_name = model.name.replace(' ', '').replace('-', '') if hasattr(model, 'name') and model.name else 'GeneratedDynamics'

# Build EXTERNAL_INPUTS from experiment.events (stimulus-type events)
# Each stimulus event name → dimension 1 (scalar signal per node)
external_inputs_dict = {}  # event_name -> dimension
if 'experiment' in context.keys():
    _events = list(experiment.events.values()) if experiment.events else []
    for ev in _events:
        ev_type = str(getattr(ev, 'event_type', 'stimulus'))
        if 'stimulus' in ev_type:
            external_inputs_dict[str(ev.name)] = 1
%>

class ${class_name}(AbstractDynamics):
    """${class_name} neural mass model."""

    STATE_NAMES = ${tuple(state_names)}
    INITIAL_STATE = ${tuple(initial_state)}
    % if aux_names:
    AUXILIARY_NAMES = ${tuple(aux_names)}
    % else:
    AUXILIARY_NAMES = ()
    % endif
<%
    # Build VARIABLES_OF_INTEREST: all states + only explicitly requested auxiliaries
    # When output lists derived variables, include them so the solver records them.
    # When output is empty, record states only (tvboptim default).
    requested_aux = [v for v in (output_vars or []) if v in aux_names]
    voi = tuple(state_names + requested_aux) if requested_aux else ()
%>\
    % if voi:
    VARIABLES_OF_INTEREST = ${voi}
    % endif

    DEFAULT_PARAMS = Bunch(
        % for name in param_names:
        ${name}=${param_defaults.get(name, 1.0)},
        % endfor
    )

    COUPLING_INPUTS = {
        % for ci_name, ci_dim in coupling_inputs_dict.items():
        '${ci_name}': ${ci_dim},
        % endfor
    }

    % if external_inputs_dict:
    EXTERNAL_INPUTS = {
        % for ei_name, ei_dim in external_inputs_dict.items():
        '${ei_name}': ${ei_dim},
        % endfor
    }
    % endif

    def dynamics(
        self,
        t: float,
        state: jnp.ndarray,
        params: Bunch,
        coupling: Bunch,
        external: Bunch,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        % for name in param_names:
        ${name} = params.${name}
        % endfor
        % for sp_name, sp_info in stochastic_params.items():
        # ${sp_name} ~ ${sp_info['dist'].capitalize()}(${sp_info['lo']}, ${sp_info['hi']}), pre-generated array indexed per step
        ${sp_name} = params._stoch_${sp_name}[jnp.int32(jnp.clip(t * ${_stoch_inv_dt}, 0, params._stoch_${sp_name}.shape[0] - 1))]
        % endfor

        % if derived_param_names:
        % for dp in model.derived_parameters.values():
        ${dp.name} = ${jaxcode_obj(dp)}
        % endfor
        % endif

        % for i, svar in enumerate(state_names):
        ${svar} = state[${i}]
        % endfor

        % for ci_name, ci_dim in coupling_inputs_dict.items():
        % if ci_name in coupling_keys:
        ## Multi-dimensional with named keys: unpack each key
        % for idx, key_name in enumerate(coupling_keys[ci_name]):
        ${key_name} = coupling.${ci_name}[${idx}] if hasattr(coupling, '${ci_name}') else 0.0
        % endfor
        % elif ci_dim == 1:
        ${ci_name} = coupling.${ci_name}[0] if hasattr(coupling, '${ci_name}') else 0.0
        % else:
        ${ci_name} = coupling.${ci_name} if hasattr(coupling, '${ci_name}') else jnp.zeros(${ci_dim})
        % endif
        % endfor

        % for ei_name in external_inputs_dict:
        ${ei_name} = external.${ei_name}[0] if hasattr(external, '${ei_name}') else 0.0
        % endfor

        % if model.functions:
        % for f in model.functions.values():
<%
    _fdef = capture(fn.function_def, f, format='jax', render_func=jaxcode_obj).strip()
    _fdef = textwrap.indent(textwrap.dedent(_fdef), '        ')
%>\
${_fdef}
        % endfor
        % endif

        % if model.derived_variables:
        % for dv in model.derived_variables.values():
        ${dv.name} = ${jaxcode_obj(dv)}
        % endfor
        % endif

        % for sv in model.state_variables.values():
        d${sv.name}_dt = ${jaxcode_obj(sv)}
        % endfor

        derivatives = jnp.stack([
            % for sv in model.state_variables.values():
            jnp.atleast_1d(d${sv.name}_dt),
            % endfor
        ], axis=0)

        % if aux_names:
        auxiliaries = jnp.stack([
            % for aux in aux_names:
            jnp.atleast_1d(${aux}),
            % endfor
        ], axis=0)
        % else:
        auxiliaries = jnp.array([])
        % endif

        return derivatives, auxiliaries
