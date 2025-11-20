## -*- coding: utf-8 -*-
%if jax:
import jax.scipy as jsp
import jax.numpy as jnp
from jax import lax
%else:
import numpy as np
import scipy
%endif

def ${model.metadata.name}(
    current_state,t,
    % if coupling_as_argument:
    coupling,
    % endif
    % for p in model.metadata.parameters.values():
    ${p.name}=${p.value},
    % endfor
    % if not coupling_as_argument:
    % for cterm in model.metadata.coupling_terms:
    ${cterm}=0.0,
    % endfor
    % endif
    local_coupling=0.0,
    %if 'stim_t' not in model.metadata.derived_variables:
    stimulus=None,
    stimulus_scaling=1.0,
    %endif
):
% if 'e' not in model.metadata.parameters:
    % if jax:
    e = jnp.e
    % else:
    e = np.e
    % endif
% endif
%if 'stim_t' not in model.metadata.derived_variables:
    %if jax:
    stim_t = stimulus[t.astype(jnp.int32)]
    %else:
    stim_t = stimulus_scaling * stimulus(t) if stimulus is not None else 0.0
    %endif
%endif
% if coupling_as_argument:
    % for i, cterm in enumerate(model.metadata.coupling_terms):
    ${cterm} = coupling[${i}] ## TODO: model.metadata.state_variables
    % endfor
% endif
% if model.metadata.derived_parameters:
    % for dp in model.metadata.derived_parameters.values():
    ${dp.name} = ${model.render_equation(dp, format='jax' if jax else 'numpy')}
    % endfor
% endif

% for i, ivar in enumerate(model.metadata.state_variables):
    ${ivar} = current_state[${i}]
% endfor

% if model.metadata.functions:
    # Functions
% for f in model.metadata.functions.values():
    def ${f.name}(${", ".join([arg.name for arg in f.arguments.values()])}):
        return ${model.render_equation(f, format='jax' if jax else 'numpy')}
% endfor
% endif

    # Derived Variables
% for k,v in model.metadata.derived_variables.items():
    ${k} = ${model.render_equation(v, format='jax' if jax else 'numpy')}
% endfor

    # Time Derivatives
    next_state = ${'jnp' if jax else 'np'}.array([
        % for sv in model.metadata.state_variables.values():
        # ${sv.name}
        ${model.render_equation(sv, format='jax' if jax else 'numpy')} ${'+ stim_t' if sv.stimulation_variable else ''},
        % endfor
    ])

    return next_state
