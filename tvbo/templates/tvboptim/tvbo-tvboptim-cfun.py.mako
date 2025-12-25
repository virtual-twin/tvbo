# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Coupling (cfun) Template
==================================

Generates coupling classes for tvboptim.experimental.network_dynamics
from a TVBO Coupling specification.

Context Variables:
- experiment: SimulationExperiment instance (optional)
- coupling: Coupling instance (required if no experiment)
- model: Dynamics instance (required for state variable info)

Output:
- Python class(es) inheriting from InstantaneousCoupling or DelayedCoupling with:
  - N_OUTPUT_STATES specification
  - DEFAULT_PARAMS as Bunch
  - pre() and post() methods
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
coupling_params = list(coupling.parameters.values()) if hasattr(coupling, 'parameters') else []
param_names = [p.name for p in coupling_params]
param_defaults = {p.name: float(p.value) if p.value is not None else 1.0
                  for p in coupling_params}

# Get incoming states
incoming_states = getattr(coupling, 'incoming_states', None) or []
if isinstance(incoming_states, str):
    incoming_states = [incoming_states]
incoming_states = list(incoming_states) if incoming_states else []

# Get local states
local_states = getattr(coupling, 'local_states', None) or []
if isinstance(local_states, str):
    local_states = [local_states]
local_states = list(local_states) if local_states else []

# Get pre and post expressions
pre_expr = coupling.pre_expression if hasattr(coupling, 'pre_expression') and coupling.pre_expression else None
post_expr = coupling.post_expression if hasattr(coupling, 'post_expression') and coupling.post_expression else None

# Class name
class_name = coupling.name.replace(' ', '').replace('-', '') if hasattr(coupling, 'name') and coupling.name else 'GeneratedCoupling'
base_class = 'DelayedCoupling' if has_delay else 'InstantaneousCoupling'

# Number of output states (from model coupling terms)
n_output = len(model.coupling_terms) if model and hasattr(model, 'coupling_terms') else 1
%>
"""${class_name} coupling for tvboptim Network Dynamics.

Auto-generated from TVBO coupling specification.
"""

import jax.numpy as jnp

from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.coupling.base import ${base_class}


class ${class_name}(${base_class}):
    """${class_name} coupling function.

    ${coupling.description if hasattr(coupling, 'description') and coupling.description else 'Auto-generated coupling function.'}

    Attributes
    ----------
    N_OUTPUT_STATES : int
        Number of output coupling states: ${n_output}
    DEFAULT_PARAMS : Bunch
        Default coupling parameters
    """

    N_OUTPUT_STATES = ${n_output}

    DEFAULT_PARAMS = Bunch(
        % for name in param_names:
        ${name}=${param_defaults.get(name, 1.0)},
        % endfor
        % if not param_names:
        G=1.0,  # Default global coupling strength
        % endif
    )

    def __init__(self, **kwargs):
        """Initialize ${class_name} coupling.

        Parameters
        ----------
        % for name in incoming_states:
        incoming_states contains: '${name}'
        % endfor
        % for name in local_states:
        local_states contains: '${name}'
        % endfor
        **kwargs : dict
            Parameter overrides for DEFAULT_PARAMS
        """
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
    def pre(
        self,
        incoming_states: jnp.ndarray,
        local_states: jnp.ndarray,
        params: Bunch,
    ) -> jnp.ndarray:
        """Pre-coupling transformation.

        Parameters
        ----------
        incoming_states : jnp.ndarray
            States from connected nodes [n_incoming, n_nodes_target, n_nodes_source]
        local_states : jnp.ndarray
            Local states [n_local, n_nodes]
        params : Bunch
            Coupling parameters

        Returns
        -------
        jnp.ndarray
            Transformed values for weighted summation
        """
        # Unpack parameters
        % for name in param_names:
        ${name} = params.${name}
        % endfor

        % if incoming_states:
        # Unpack incoming states
        % for i, state_name in enumerate(incoming_states):
        ${state_name} = incoming_states[${i}]
        % endfor
        % endif

        % if local_states:
        # Unpack local states
        % for i, state_name in enumerate(local_states):
        ${state_name}_local = local_states[${i}]
        % endfor
        % endif

        # Pre-coupling expression
        pre = ${jaxcode(pre_expr.rhs)}

        return pre
    % endif

    def post(
        self,
        summed_inputs: jnp.ndarray,
        local_states: jnp.ndarray,
        params: Bunch,
    ) -> jnp.ndarray:
        """Post-coupling transformation.

        Parameters
        ----------
        summed_inputs : jnp.ndarray
            Weighted sum of pre-processed inputs [n_outputs, n_nodes]
        local_states : jnp.ndarray
            Local states [n_local, n_nodes]
        params : Bunch
            Coupling parameters

        Returns
        -------
        jnp.ndarray
            Final coupling values [n_outputs, n_nodes]
        """
        # Unpack parameters
        % for name in param_names:
        ${name} = params.${name}
        % endfor
        % if 'G' not in param_names:
        G = params.G if hasattr(params, 'G') else 1.0
        % endif

        gx = summed_inputs

        % if post_expr:
        # Post-coupling expression
        return ${jaxcode(post_expr.rhs)}
        % else:
        # Default: scale by global coupling
        return G * gx
        % endif


% if has_delay:
# Also provide instantaneous version for convenience
class ${class_name}Instant(InstantaneousCoupling):
    """Instantaneous version of ${class_name} (no delays)."""

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
        return ${jaxcode(pre_expr.rhs)}
    % endif

    def post(self, summed_inputs, local_states, params):
        % for name in param_names:
        ${name} = params.${name}
        % endfor
        gx = summed_inputs
        % if post_expr:
        return ${jaxcode(post_expr.rhs)}
        % else:
        return params.G * gx if hasattr(params, 'G') else gx
        % endif


from tvboptim.experimental.network_dynamics.coupling.base import InstantaneousCoupling
% endif
