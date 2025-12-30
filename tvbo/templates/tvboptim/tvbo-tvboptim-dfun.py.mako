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
from tvbo.export.code import render_expression

# Get model from context
if 'experiment' in context.keys():
    model = experiment.local_dynamics
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

# Determine auxiliary variables from 'output' attribute
# If output is specified, only derived variables in output list become auxiliaries
# State variables in output are already state variables, not auxiliaries
output_vars = getattr(model, 'output', None) or []
if isinstance(output_vars, str):
    output_vars = [output_vars]

if output_vars:
    # Filter output to only include derived variables (exclude state variables)
    aux_names = [v for v in output_vars if v in (model.derived_variables or {}).keys()]
else:
    # Default: all derived variables become auxiliaries
    aux_names = list(model.derived_variables.keys()) if model.derived_variables else []
param_names = [p.name for p in model.parameters.values()]
param_defaults = {p.name: float(p.value) if p.value is not None else 1.0
                  for p in model.parameters.values()}
derived_param_names = [p.name for p in model.derived_parameters.values()] if model.derived_parameters else []
coupling_terms_raw = list(model.coupling_terms.keys()) if model.coupling_terms else ['default']

# Map TVBO coupling term names to tvboptim convention
# TVBO uses c_instant/c_delayed, tvboptim uses instant/delayed
def to_tvboptim_coupling_name(name):
    """Strip 'c_' prefix for tvboptim compatibility."""
    if name.startswith('c_'):
        return name[2:]  # Remove 'c_' prefix
    return name

# Create mapping: TVBO name -> tvboptim name
coupling_term_map = {ct: to_tvboptim_coupling_name(ct) for ct in coupling_terms_raw}
coupling_terms = list(coupling_term_map.values())  # tvboptim names for COUPLING_INPUTS

class_name = model.name.replace(' ', '').replace('-', '') if hasattr(model, 'name') and model.name else 'GeneratedDynamics'
%>

class ${class_name}(AbstractDynamics):
    """${class_name} neural mass model.

    ${model.description if hasattr(model, 'description') and model.description else 'Auto-generated dynamics model.'}

    State variables: ${state_names}
    Parameters: ${param_names}
    """

    STATE_NAMES = ${tuple(state_names)}
    INITIAL_STATE = ${tuple(initial_state)}
    % if aux_names:
    AUXILIARY_NAMES = ${tuple(aux_names)}
    % else:
    AUXILIARY_NAMES = ()
    % endif

    DEFAULT_PARAMS = Bunch(
        % for name in param_names:
        ${name}=${param_defaults.get(name, 1.0)},
        % endfor
    )

    COUPLING_INPUTS = {
        % for ct in coupling_terms:
        '${ct}': 1,
        % endfor
    }

    def dynamics(
        self,
        t: float,
        state: jnp.ndarray,
        params: Bunch,
        coupling: Bunch,
        external: Bunch,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute ${class_name} dynamics."""
        # Unpack parameters
        % for name in param_names:
        ${name} = params.${name}
        % endfor

        % if derived_param_names:
        # Derived parameters
        % for dp in model.derived_parameters.values():
        ${dp.name} = ${jaxcode_obj(dp)}
        % endfor
        % endif

        # Unpack state variables
        % for i, svar in enumerate(state_names):
        ${svar} = state[${i}]
        % endfor

        # Unpack coupling (tvboptim uses instant/delayed, TVBO equations use c_instant/c_delayed)
        % for tvbo_name, tvboptim_name in coupling_term_map.items():
        ${tvbo_name} = coupling.${tvboptim_name}[0] if hasattr(coupling, '${tvboptim_name}') else 0.0
        % endfor

        % if model.functions:
        # Helper functions
        % for f in model.functions.values():
        ${fn.function_def(f, format='jax', render_func=jaxcode_obj) | trim,n}
        % endfor
        % endif

        % if model.derived_variables:
        # Derived variables
        % for dv in model.derived_variables.values():
        ${dv.name} = ${jaxcode_obj(dv)}
        % endfor
        % endif

        # State derivatives
        % for sv in model.state_variables.values():
        d${sv.name}_dt = ${jaxcode_obj(sv)}
        % endfor

        derivatives = jnp.array([
            % for sv in model.state_variables.values():
            d${sv.name}_dt,
            % endfor
        ])

        % if aux_names:
        auxiliaries = jnp.array([${', '.join(aux_names)}])
        % else:
        auxiliaries = jnp.array([])
        % endif

        return derivatives, auxiliaries
