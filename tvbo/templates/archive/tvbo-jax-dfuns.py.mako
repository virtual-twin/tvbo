<%namespace name="utils" file="jax-utils.py.mako"/>

import jax.numpy as jnp
from collections import namedtuple

## Derivatives of state variables
def dfuns(current_state, cX, params_dfun, local_coupling=0):
    \
% for par in parameter_names:
${par}, \
% endfor
= params_dfun

    pi = jnp.pi
    exp = jnp.exp

    # unpack coupling terms and states as in dfuns
    % for i, cterm in enumerate(coupling_terms):
    ${cterm} = cX[${i}]
    % endfor
    ## ${','.join(coupling_terms)} = cX
    <%
    if non_integrated_variables == None:
        integrated_state_variables = state_variables
    else:
        integrated_state_variables = [var for var in state_variables if var not in non_integrated_variables]
    %>
    ${','.join(integrated_state_variables)} = current_state

    # compute internal states for dfuns
    % for var, term in non_integrated_variables.items():
        %if var not in integrated_state_variables:
        ${var} = ${term}
        %endif
    % endfor

    %if non_integrated_variables == None:
    return jnp.array([
        % for svar in state_variables:
            jnp.squeeze(${state_variable_dfuns[svar]}),
        % endfor
        ])
    %else:
    # compute integrated variables
    ivars = jnp.array([
        % for svar in state_variables:
            %if svar not in non_integrated_variables:
            jnp.squeeze(${state_variable_dfuns[svar]}),
            %endif
        % endfor
        ])
    # non-integrated variables
    nivars = jnp.array([
        % for var in non_integrated_variables:
            jnp.squeeze(${var}),
        % endfor
        ])
    return (ivars, nivars)
    %endif