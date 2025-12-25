# -*- coding: utf-8 -*-
<%doc>
TVB-Optim Dynamics (dfun) Template
==================================

Generates an AbstractDynamics subclass for tvboptim.experimental.network_dynamics
from a TVBO Dynamics/Model specification.

Context Variables:
- experiment: SimulationExperiment instance (optional)
- model: Dynamics instance (required if no experiment)

Output:
- Python class inheriting from AbstractDynamics with:
  - STATE_NAMES, INITIAL_STATE, AUXILIARY_NAMES
  - DEFAULT_PARAMS as Bunch
  - COUPLING_INPUTS specification
  - dynamics() method implementing the equations
</%doc>
<%
from tvbo.export.code import render_expression

# Get model from context
if 'experiment' in context.keys():
    model = experiment.local_dynamics
else:
    model = context['model']

jaxcode = lambda expr: render_expression(expr, format='jax')
jaxcode_obj = lambda obj: model.render_equation(obj, format='jax')

# Extract metadata
state_names = list(model.state_variables.keys())
initial_state = [float(sv.initial_value) if sv.initial_value is not None else 0.0
                 for sv in model.state_variables.values()]

# Auxiliary variables (derived variables that are computed but not integrated)
aux_names = list(model.derived_variables.keys()) if model.derived_variables else []

# Parameters
param_names = [p.name for p in model.parameters.values()]
param_defaults = {p.name: float(p.value) if p.value is not None else 1.0
                  for p in model.parameters.values()}

# Derived parameters
derived_param_names = [p.name for p in model.derived_parameters.values()] if model.derived_parameters else []

# Coupling inputs
coupling_inputs = {}
for i, ct in enumerate(model.coupling_terms.keys()):
    coupling_inputs[ct] = 1  # Each coupling term maps to 1 dimension

# External inputs (if any)
external_inputs = {}
if hasattr(model, 'external_inputs') and model.external_inputs:
    for name, inp in model.external_inputs.items():
        external_inputs[name] = 1

# Model class name
class_name = model.name.replace(' ', '').replace('-', '') if hasattr(model, 'name') and model.name else 'GeneratedDynamics'
%>
"""${class_name} dynamics for tvboptim Network Dynamics.

Auto-generated from TVBO model specification.
"""

from typing import Tuple

import jax.numpy as jnp

from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics


class ${class_name}(AbstractDynamics):
    """${class_name} neural mass model.

    ${model.description if hasattr(model, 'description') and model.description else 'Auto-generated dynamics model.'}

    Attributes
    ----------
    STATE_NAMES : tuple of str
        State variable names: ${tuple(state_names)}
    INITIAL_STATE : tuple of float
        Default initial conditions
    AUXILIARY_NAMES : tuple of str
        Auxiliary (derived) variable names
    DEFAULT_PARAMS : Bunch
        Default parameter values
    COUPLING_INPUTS : dict
        Coupling input specification
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
        % for name in derived_param_names:
        ${name}=None,  # Derived parameter - computed from others
        % endfor
    )

    % if coupling_inputs:
    COUPLING_INPUTS = {
        % for name, dims in coupling_inputs.items():
        '${name}': ${dims},
        % endfor
    }
    % else:
    COUPLING_INPUTS = {'default': 1}
    % endif

    % if external_inputs:
    EXTERNAL_INPUTS = {
        % for name, dims in external_inputs.items():
        '${name}': ${dims},
        % endfor
    }
    % else:
    EXTERNAL_INPUTS = {}
    % endif

    def dynamics(
        self,
        t: float,
        state: jnp.ndarray,
        params: Bunch,
        coupling: Bunch,
        external: Bunch,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute ${class_name} dynamics.

        Parameters
        ----------
        t : float
            Current time
        state : jnp.ndarray
            Current state with shape [${len(state_names)}, n_nodes]
        params : Bunch
            Model parameters
        coupling : Bunch
            Coupling inputs
        external : Bunch
            External inputs

        Returns
        -------
        derivatives : jnp.ndarray
            State derivatives with shape [${len(state_names)}, n_nodes]
        auxiliaries : jnp.ndarray
            Auxiliary variables with shape [${len(aux_names)}, n_nodes]
        """
        # Unpack parameters
        % for name in param_names:
        ${name} = params.${name}
        % endfor

        % if derived_param_names:
        # Compute derived parameters
        % for dp in model.derived_parameters.values():
        ${dp.name} = ${jaxcode_obj(dp)}
        % endfor
        % endif

        # Unpack state variables
        % for i, svar in enumerate(state_names):
        ${svar} = state[${i}]
        % endfor

        # Unpack coupling inputs
        % for i, (cname, cdims) in enumerate(coupling_inputs.items()):
        ${cname} = coupling.${cname}[0] if hasattr(coupling, '${cname}') else 0.0
        % endfor

        % if external_inputs:
        # Unpack external inputs
        % for ename in external_inputs.keys():
        ${ename} = external.${ename}[0] if hasattr(external, '${ename}') else 0.0
        % endfor
        % endif

        % if model.functions:
        # Helper functions
        % for f in model.functions.values():
        def ${f.name}(${', '.join([arg.name if hasattr(arg, 'name') else str(arg) for arg in (f.arguments.values() if hasattr(f.arguments, 'values') else f.arguments)])}):
            return ${jaxcode_obj(f)}
        % endfor
        % endif

        % if model.derived_variables:
        # Compute derived variables
        % for dv in model.derived_variables.values():
        ${dv.name} = ${jaxcode_obj(dv)}
        % endfor
        % endif

        # State derivatives
        % for sv in model.state_variables.values():
        d${sv.name}_dt = ${jaxcode_obj(sv)}
        % endfor

        # Package results
        derivatives = jnp.array([
            % for sv in model.state_variables.values():
            d${sv.name}_dt,
            % endfor
        ])

        % if aux_names:
        auxiliaries = jnp.array([
            % for aux in aux_names:
            ${aux},
            % endfor
        ])
        % else:
        auxiliaries = jnp.array([])
        % endif

        return derivatives, auxiliaries
