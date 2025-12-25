# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Coupling (cfun) Template
==================================

Generates coupling classes for tvboptim.experimental.network_dynamics.

Context Variables:
- experiment: SimulationExperiment instance (optional)
- coupling: Coupling instance (required if no experiment)
- model: Dynamics instance (for n_output info)

Output:
- Python class(es) inheriting from InstantaneousCoupling or DelayedCoupling
</%doc>
<%
from tvbo.export.code import render_expression

# Get coupling and model from context
if 'experiment' in context.keys():
    coupling = experiment.coupling
    model = experiment.local_dynamics
else:
    coupling = context['coupling']
    model = context.get('model', None)

jaxcode = lambda expr: render_expression(expr, format='jax')

# Determine if coupling is delayed
has_delay = hasattr(coupling, 'delayed') and coupling.delayed

# Get coupling parameters
coupling_params = list(coupling.parameters.values()) if hasattr(coupling, 'parameters') and coupling.parameters else []
param_names = [p.name for p in coupling_params]
param_defaults = {p.name: float(p.value) if p.value is not None else 1.0
                  for p in coupling_params}

# Get incoming/local states
incoming_states = getattr(coupling, 'incoming_states', None) or []
if isinstance(incoming_states, str):
    incoming_states = [incoming_states]
incoming_states = list(incoming_states) if incoming_states else []

local_states = getattr(coupling, 'local_states', None) or []
if isinstance(local_states, str):
    local_states = [local_states]
local_states = list(local_states) if local_states else []

# Get pre and post expressions
pre_expr = coupling.pre_expression if hasattr(coupling, 'pre_expression') and coupling.pre_expression else None
post_expr = coupling.post_expression if hasattr(coupling, 'post_expression') and coupling.post_expression else None

# Class name and base class
class_name = coupling.name.replace(' ', '').replace('-', '') if hasattr(coupling, 'name') and coupling.name else 'GeneratedCoupling'
base_class = 'DelayedCoupling' if has_delay else 'InstantaneousCoupling'

# Number of output states - this is always 1 for a single coupling function
# (represents one scalar/vector output per node from this coupling)
n_output = 1
%>

class ${class_name}(${base_class}):
    """${class_name} coupling function.

    ${coupling.description if hasattr(coupling, 'description') and coupling.description else 'Auto-generated coupling function.'}
    """

    N_OUTPUT_STATES = ${n_output}

    DEFAULT_PARAMS = Bunch(
        % for name in param_names:
        ${name}=${param_defaults.get(name, 1.0)},
        % endfor
        % if not param_names:
        G=1.0,
        % endif
    )

    def __init__(self, **kwargs):
        % if incoming_states:
        super().__init__(
            incoming_states=${incoming_states},
            % if local_states:
            local_states=${local_states},
            % endif
            **kwargs
        )
        % else:
        super().__init__(**kwargs)
        % endif

    % if pre_expr:
    def pre(self, incoming_states, local_states, params):
        % for name in param_names:
        ${name} = params.${name}
        % endfor
        % for i, state_name in enumerate(incoming_states):
        ${state_name} = incoming_states[${i}]
        % endfor
        coupling_term = ${jaxcode(pre_expr.rhs)}
        % if has_delay:
        # Return as [1, n_nodes_target, n_nodes_source] for matrix multiplication
        return coupling_term[jnp.newaxis, :, :]
        % else:
        return coupling_term
        % endif
    % endif

    def post(self, summed_inputs, local_states, params):
        % for name in param_names:
        ${name} = params.${name}
        % endfor
        % if 'G' not in param_names:
        G = params.G if hasattr(params, 'G') else 1.0
        % endif
        gx = summed_inputs
        % if post_expr:
        return ${jaxcode(post_expr.rhs)}
        % else:
        return G * gx
        % endif
