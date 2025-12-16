# -*- coding: utf-8 -*-
<%
from tvbo.export.code import render_expression
jaxcode = lambda expr: render_expression(expr, format='jax')

if 'coupling' not in context.keys():
    coupling = experiment.coupling
    model = experiment.local_dynamics
else:
    coupling = context['coupling']
    model = context.get('model', None)
has_delay = coupling.delayed and (experiment.horizon > 1)

# Get incoming_states names for variable assignment
incoming_states_names = getattr(coupling, 'incoming_states', None) or []
if isinstance(incoming_states_names, str):
    incoming_states_names = [incoming_states_names]
# Convert to list if it's some other iterable
incoming_states_names = list(incoming_states_names) if incoming_states_names else []

# Check if any incoming_states variable name is used in pre_expression
pre_rhs = str(coupling.pre_expression.rhs)
needs_x_j = 'x_j' in pre_rhs or any(str(name) in pre_rhs for name in incoming_states_names)
is_list_expr = pre_rhs.strip().startswith('[') and pre_rhs.strip().endswith(']')
%>

## Coupling function
def cfun(weights, history, current_state, p, delay_indices, t):
    n_node = weights.shape[0]
    ${', '.join([par.name for par in coupling.parameters.values()])} = p.${', p.'.join([par.name for par in coupling.parameters.values()])}
## History is "sparse", only states that are coupled are stored in history therefore we use cvar_idx (assumes cvar is ordered)
## JAX does not throw out of bounds errors but returns the last valid index, so be careful!
## Collect x_i and x_j as needed, pre needs all cvars at the same time
% if 'x_i' in pre_rhs: ## don't generate x_i if not required
    x_i = jnp.array([
% for i, sv in enumerate(model.state_variables.values()):
    % if sv.coupling_variable:
    current_state[${i}, :],
    % endif
% endfor
    ])
    x_i = x_i.transpose(1, 0)
    ## %if has_delay:
    x_i = jnp.expand_dims(x_i, axis=-1)
    ## %endif
% endif

## Collect incoming states from connected nodes (x_j)
% if needs_x_j:
    x_j = jnp.array([
<% cvar_idx = 0 %>
% for i, sv in enumerate(model.state_variables.values()):
    % if sv.coupling_variable:
    % if has_delay: # We need to collect delayed states
            % if small_dt: # history of nt + nh length
    history[${cvar_idx}, delay_indices[0].T + t, delay_indices[1]],
            % else: # rolling history of nh length
    history[${cvar_idx}, delay_indices[0].T, delay_indices[1]],
            % endif
        % else: # no delay
            % if not scalar_pre: # We need to collect all states (history is equal to current_state here) for non scalar operations like Differnce or SigmoidelJansenRit
    current_state[${i}, delay_indices[1]],
            %else:
    current_state[${i}],
            % endif
        % endif
    <% cvar_idx += 1 %>
    % endif
% endfor
    ])
% if not scalar_pre: # We need to collect all states (history is equal to current_state here) for non scalar operations like Difference or SigmoidelJansenRit
    ## x_j = x_j.transpose(1, 0, 2) ## (n_node, n_cvar, ...) old TVB convention
% endif
## Assign named variables from incoming_states
% for idx, state_name in enumerate(incoming_states_names):
    ${state_name} = x_j[${idx}]
% endfor
% endif

## Apply pre-expression this can reduce and collapse the cvar dimension, eg. SigmoidalJansenRit
% if is_list_expr:
    pre = jnp.stack(${jaxcode(pre_rhs)}, axis=0)
% else:
    pre = ${jaxcode(pre_rhs)}
% endif
    ## %if has_delay:
    % if not scalar_pre:
    ## pre = pre.reshape(n_node, -1 ,n_node) ## Restore collapsed dimension if necessary
    pre = pre.reshape(-1, n_node ,n_node) ## Restore collapsed dimension if necessary
    %endif

## Apply weights
## delay dotproduct -> sum: (nnodes x nnodes) x (nnodes x nnodes
% if not scalar_pre:
    def op(x): return jnp.sum(weights * x, axis=-1)
    gx = jax.vmap(op, in_axes=0)(pre)
% else:
    def op(x): return weights @ x
    gx = jax.vmap(op, in_axes=0)(pre)
## no-delay matmul: (nnodes x nnodes) x (nnodes x n_cvar) = (nnodes x n_cvar)
    ## gx = jnp.matmul(weights, pre)
    ## gx = gx.T
% endif
    return ${jaxcode(coupling.post_expression.rhs)}

