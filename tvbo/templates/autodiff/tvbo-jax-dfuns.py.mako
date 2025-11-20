# -*- coding: utf-8 -*-
<%
if 'experiment' in context.keys():
    model = experiment.local_dynamics
else:
    model = context['model']

metadata = model

jaxcode = lambda obj: model.render_equation(obj, format='jax')

parameters = [par.name for par in metadata.parameters.values()] + [par.name for par in metadata.derived_parameters.values()]

%>
import jax.numpy as jnp
import jax.scipy as jsp

from collections import namedtuple

## Derivatives of state variables
def dfun(current_state, cX, _p , local_coupling=0):
    ${', '.join(parameters)} = _p.${', _p.'.join(parameters)}

    # unpack coupling terms and states as in dfun
    % for i, cterm in enumerate(metadata.coupling_terms):
    ${cterm} = cX[${i}]
    % endfor

    % for i, ivar in enumerate(metadata.state_variables):
    ${ivar} = current_state[${i}]
    % endfor

    % if metadata.functions:
    # Functions
    % for f in metadata.functions.values():
    def ${f.name}(${", ".join([arg.name for arg in f.arguments.values()])}):
        return ${jaxcode(f)}
    % endfor
    % endif

    # compute internal states for dfun
    % for dv in metadata.derived_variables.values():
    ${dv.name} = ${jaxcode(dv)}
    % endfor

    return jnp.array([
        % for sv in metadata.state_variables.values():
            ${jaxcode(sv)}, # ${sv.name}
        % endfor
        ])


