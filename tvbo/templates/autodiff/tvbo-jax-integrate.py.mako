# -*- coding: utf-8 -*-
<%
import numpy as np
if 'integration' in context.keys():
    integration = context['integration']
    model = context['model']
elif 'experiment' in context.keys():
    integration = experiment.integration
    model = experiment.local_dynamics
else:
    raise ValueError("No integration metadata found")

stochastic = integration.noise is not None
delayed = integration.delayed

## Potential boundaries for clamping
boundaries = [
    (sv.boundaries.lo, sv.boundaries.hi)
    for sv in model.state_variables.values()
    if sv.boundaries is not None
]
def get_lower_bound(sv):
    if sv.boundaries and sv.boundaries.lo is not None:
        return sv.boundaries.lo
    return -np.inf
def get_upper_bound(sv):
    if sv.boundaries and sv.boundaries.hi is not None:
        return sv.boundaries.hi
    return np.inf
%>

## Helper that converts a array into a string that can be read as array again
<%def name = "array_input(array)" filter="trim">
        <%
        import numpy as np
        %>
        ## ${np.set_printoptions(threshold=sys.maxsize)}
        jnp.array(${np.array2string(array, separator = ",")})
</%def>

## TODO: Implement Stimulus

def integrate(state, weights, dt, params_integrate, delay_indices, external_input):
    """
    ${integration.method} Integration
    ${'=' * len(integration.method) + '='*len(" Integration")}
    """
    %if stochastic:
    t, noise = external_input
    % else:
    t, noise = external_input, 0
    ## ${'t, noise' if stochastic else 't'} = external_input
    % endif

    ## dt = ${integration.step_size}

    params_dfun, params_cfun, params_stimulus = params_integrate

    history, current_state = state
    % if has_stimulus:
    stimulus = get_stimulus(t, params_stimulus)
    % else:
    stimulus = 0
    % endif

    ## Clip state if bounds are defined
    % if boundaries:
    inf = jnp.inf
    ## Clip by boundaries if present else set bounderies to +-inf -> no clip
    min_bounds = jnp.array(${[[[get_lower_bound(sv)]] for sv in model.state_variables.values()]})
    max_bounds = jnp.array(${[[[get_upper_bound(sv)]] for sv in model.state_variables.values()]})
    % endif

    cX = jax.vmap(cfun, in_axes=(None, -1, -1, None, None, None), out_axes=-1)(weights, history, current_state, params_cfun, delay_indices, t)

    dX0 = dfun(current_state, cX, params_dfun)

    X = current_state

    ## % for i, exp in enumerate(integration.intermediate_expressions.values()):
    ## inter_k${i+1} = ${exp.equation.rhs}
    ## ${exp.name} = dfun(inter_k${i+1}, cX, params_dfun)

    ## % endfor
    ## ## Calculate the state change dX
    ## dX = ${integration.update_expression.equation.rhs}

    ###
    % for k, step in integration.intermediate_expressions.items():
    # Calculate intermediate step ${k}
    ${k} = ${step.equation.rhs}
    ## self.integration_bound_and_clamp(${k})
    % if boundaries:
    ${k} = jnp.clip(${k}, min_bounds, max_bounds)
    % endif

    # Calculate derivative ${k}
    d${k} = dfun(${k}, cX, params_dfun)
    % endfor
    # Calculate the state change dX
    dX = ${integration.update_expression.equation.rhs}
    ## # Calculate the next state of X, including noise
    ## X_next = X + dX + noise + stimulus * dt
    ## self.integration_bound_and_clamp(X_next)
    ## return X_next
    ###
    next_state = current_state + (dX)${" + dt * stimulus" if has_stimulus else ''}${' + noise' if stochastic else ''}
    % if boundaries:
    next_state = jnp.clip(next_state, min_bounds, max_bounds)
    % endif




## Return for scan: carry, result
% if delayed:
    <%
    cvar_list = [i for i, sv in enumerate(model.state_variables.values()) if sv.coupling_variable]
    %>
    %if len(cvar_list) > 0:
    cvar = ${array_input(np.array(cvar_list))}
    % if small_dt:
    history = history.at[:, t, :].set(next_state[cvar, :])
    % else:
    _h = jnp.roll(history, -1, axis=1)
    history = _h.at[:, -1, :].set(next_state[cvar, :])
    % endif
    %else:
    ## When no coupling variables are defined, store all state variables in history
    % if small_dt:
    history = history.at[:, t, :].set(next_state)
    % else:
    _h = jnp.roll(history, -1, axis=1)
    history = _h.at[:, -1, :].set(next_state)
    % endif
    %endif
% endif
## Return for scan coupling if monitored
% if monitor_node_coupling:
    return (history, next_state), (next_state, cX)
% else:
    return (history, next_state), next_state
% endif


