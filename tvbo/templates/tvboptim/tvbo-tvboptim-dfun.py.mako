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
from tvbo.templates.tvboptim.utils import get_param_info, get_recorded_variable_names, render_jax_default, get_mode_layout

# Get model from context
if 'experiment' in context.keys():
    _experiment_ctx = experiment
    model = experiment.dynamics
    # Also collect experiment-level functions if available
    _exp_functions = getattr(experiment, 'functions', None) or {}
else:
    _experiment_ctx = None
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

## preserve_order: keep authored term order so generated dynamics match
## hand-written reference dynamics operation-for-operation (float byte-identity).
jaxcode = lambda expr: render_expression(expr, format='jax', user_functions=user_functions, preserve_order=True)
jaxcode_obj = lambda obj: model.render_equation(obj, format='jax', preserve_order=True)

# Extract metadata. For number_of_modes>1 the per-node mode axis is folded into
# the state axis (see get_mode_layout): each variable occupies n_modes scalar
# slots and the dfun reconstructs the (n_nodes, n_modes) mode-vector per variable.
# Single-mode models are unchanged (state_names == variable names).
n_modes, state_names, var_slots = get_mode_layout(model)
var_names = list(model.state_variables.keys())
_init_value = {
    sv_name: (float(sv.initial_value) if sv.initial_value is not None else 0.0)
    for sv_name, sv in model.state_variables.items()
}
initial_state = [_init_value[v] for v in var_names for _ in range(n_modes)]

# Determine auxiliaries to record. Includes:
#   * derived variables listed in model.output, and
#   * derived variables referenced as observation sources (auto-included so that
#     observing an auxiliary does not require also adding it to model.output).
# all_aux_names = every derived variable defined by the model (the AUXILIARY_NAMES tuple).
# requested_aux = subset that the solver should record (extends VARIABLES_OF_INTEREST).
all_aux_names = list(model.derived_variables.keys()) if model.derived_variables else []
_, requested_aux, recorded_var_names = get_recorded_variable_names(model, _experiment_ctx)
aux_names = all_aux_names

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
        if ('stimul' in ev_type) or (ev_type in ('continuous', 'discrete')):
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
    # VARIABLES_OF_INTEREST: all states (in order) + only auxiliaries that are
    # explicitly requested via model.output OR referenced by an observation source.
    # Empty tuple = tvboptim default (record all states only).
    voi = tuple(state_names + requested_aux) if requested_aux else ()
%>\
    % if voi:
    VARIABLES_OF_INTEREST = ${voi}
    % endif

    DEFAULT_PARAMS = Bunch(
        % for name in param_names:
        ${name}=${render_jax_default(param_defaults.get(name, 1.0))},
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

        % if n_modes > 1:
        ## Mode fold: rebuild each variable's (n_nodes, n_modes) mode-vector from
        ## its per-mode scalar slots so the mode-aware equations (mode_dot/mode_sum)
        ## see a real mode axis.
        % for v in var_names:
        ${v} = jnp.stack([${', '.join('state[%d]' % i for i in var_slots[v])}], axis=-1)
        % endfor
        % else:
        % for i, svar in enumerate(state_names):
        ${svar} = state[${i}]
        % endfor
        % endif

        % for ci_name, ci_dim in coupling_inputs_dict.items():
        % if n_modes > 1:
        ## Mode fold: the coupling emits per-mode output (n_modes, n_nodes); present
        ## it as (n_nodes, n_modes) so the mode-aware equations (mode_sum over the
        ## trailing axis) reduce it exactly as TVB's coupling[cvar].sum(over modes).
        ${ci_name} = jnp.moveaxis(coupling.${ci_name}, 0, -1) if hasattr(coupling, '${ci_name}') else 0.0
        % elif ci_name in coupling_keys:
        ## Multi-dimensional with named keys: unpack each key
        % for idx, key_name in enumerate(coupling_keys[ci_name]):
        ${key_name} = coupling.${ci_name}[${idx}] if hasattr(coupling, '${ci_name}') else 0.0
        % endfor
        % elif ci_dim == 1:
        ## `hasattr` only proves the attribute exists. An unsatisfied coupling input
        ## (e.g. local_coupling in a region simulation) arrives as a scalar rather than
        ## an array, and subscripting it raises "'int' object is not subscriptable".
        ## atleast_1d makes the scalar case indexable and is a no-op for real arrays.
        ${ci_name} = jnp.atleast_1d(coupling.${ci_name})[0] if hasattr(coupling, '${ci_name}') else 0.0
        % else:
        ${ci_name} = coupling.${ci_name} if hasattr(coupling, '${ci_name}') else jnp.zeros(${ci_dim})
        % endif
        % endfor

        % for ei_name in external_inputs_dict:
        ${ei_name} = jnp.atleast_1d(external.${ei_name})[0] if hasattr(external, '${ei_name}') else 0.0
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

        % if n_modes > 1:
        # Mode fold-back: split each variable's (n_nodes, n_modes) derivative into
        # its n_modes scalar slots, in the same (variable, mode) order as STATE_NAMES.
        _mode_ref = jnp.atleast_2d(d${var_names[0]}_dt)
        derivatives = jnp.concatenate([
            % for v in var_names:
            jnp.moveaxis(jnp.broadcast_to(jnp.atleast_2d(d${v}_dt), _mode_ref.shape), -1, 0),
            % endfor
        ], axis=0)
        % else:
        # Determine per-node shape from the first state variable; broadcast
        # any scalar derivatives or auxiliaries so jnp.stack sees uniform
        # rank-1 arrays.
        _per_node_shape = jnp.atleast_1d(d${list(model.state_variables.keys())[0]}_dt).shape

        derivatives = jnp.stack([
            % for sv in model.state_variables.values():
            jnp.broadcast_to(jnp.atleast_1d(d${sv.name}_dt), _per_node_shape),
            % endfor
        ], axis=0)
        % endif

        % if aux_names:
        % if n_modes > 1:
        # Fold auxiliaries per-mode too (same layout as the derivatives), so a
        # multi-mode model with recorded derived variables produces a valid,
        # mode-consistent auxiliary array (``_mode_ref`` from the derivative fold).
        auxiliaries = jnp.concatenate([
            % for aux in aux_names:
            jnp.moveaxis(jnp.broadcast_to(jnp.atleast_2d(${aux}), _mode_ref.shape), -1, 0),
            % endfor
        ], axis=0)
        % else:
        auxiliaries = jnp.stack([
            % for aux in aux_names:
            jnp.broadcast_to(jnp.atleast_1d(${aux}), _per_node_shape),
            % endfor
        ], axis=0)
        % endif
        % else:
        auxiliaries = jnp.array([])
        % endif

        return derivatives, auxiliaries
