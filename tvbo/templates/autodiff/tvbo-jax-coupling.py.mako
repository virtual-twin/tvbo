# -*- coding: utf-8 -*-
<%
from tvbo.export.code import render_expression

# Generic jaxcode - pass parameters on each call
jaxcode = lambda expr, parameters=None: render_expression(expr, format='jax', parameters=parameters)

if 'coupling' not in context.keys():
    coupling = experiment.coupling
    model = experiment.dynamics
else:
    coupling = context['coupling']
    model = context.get('model', None)

# Collect coupling parameter names for use in expressions
coupling_param_names = [par.name for par in coupling.parameters.values()] if coupling.parameters else []

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

# Check if we need to return multiple coupling outputs
num_coupling_terms = len(model.coupling_terms) if hasattr(model, 'coupling_terms') else 1
needs_stacked_output = num_coupling_terms > 1

# Check if coupling has per-term weight parameters (e.g., wLRE, wFFI)
# These would be named like w_<term_name> or just listed as weight-type parameters
coupling_term_names = list(model.coupling_terms.keys()) if hasattr(model, 'coupling_terms') else []
has_weight_params = any(par.name.startswith('w') and par.name[1:] in ['LRE', 'FFI', '_'] or
                       par.name.lower() in ['wlre', 'wffi']
                       for par in coupling.parameters.values())
%>

def cfun(weights, history, current_state, p, delay_indices, t):
    n_node = weights.shape[0]
    ${', '.join([par.name for par in coupling.parameters.values()])} = p.${', p.'.join([par.name for par in coupling.parameters.values()])}

% if 'x_i' in pre_rhs:
    x_i = jnp.array([
% for i, sv in enumerate(model.state_variables.values()):
    % if sv.coupling_variable:
    current_state[${i}, :],
    % endif
% endfor
    ])
    x_i = x_i.transpose(1, 0)
    x_i = jnp.expand_dims(x_i, axis=-1)
% endif

% if needs_x_j:
    x_j = jnp.array([
<% cvar_idx = 0 %>
% for i, sv in enumerate(model.state_variables.values()):
    % if sv.coupling_variable:
    % if has_delay:
            % if small_dt:
    history[${cvar_idx}, delay_indices[0].T + t, delay_indices[1]],
            % else:
    history[${cvar_idx}, delay_indices[0].T, delay_indices[1]],
            % endif
        % else:
            % if not scalar_pre:
    current_state[${i}, delay_indices[1]],
            %else:
    current_state[${i}],
            % endif
        % endif
    <% cvar_idx += 1 %>
    % endif
% endfor
    ])
% if not scalar_pre:
% endif

% for idx, state_name in enumerate(incoming_states_names):
    ${state_name} = x_j[${idx}]
% endfor
% endif

% if is_list_expr:
    pre = jnp.stack(${jaxcode(pre_rhs, parameters=coupling_param_names + incoming_states_names)}, axis=0)
% else:
    pre = ${jaxcode(pre_rhs, parameters=coupling_param_names + incoming_states_names)}
% endif
    % if not scalar_pre:
    pre = pre.reshape(-1, n_node ,n_node)
    %endif

% if has_weight_params and needs_stacked_output:
    outputs = []
    % for i, term_name in enumerate(coupling_term_names):
<%
    weight_param = 'weights'
    for par in coupling.parameters.values():
        if (par.name.lower() == f'w{term_name.lower().replace("cpop_", "")}' or
            par.name.lower() == f'w_{i}' or
            (term_name == 'cpop_0' and par.name.lower() == 'wlre') or
            (term_name == 'cpop_1' and par.name.lower() == 'wffi')):
            weight_param = par.name
            break
%>
    % if not scalar_pre:
    def op_${i}(x): return jnp.sum(${weight_param} * x, axis=-1)
    gx_${i} = jax.vmap(op_${i}, in_axes=0)(pre)[0]
    % else:
    def op_${i}(x): return ${weight_param} @ x
    gx_${i} = jax.vmap(op_${i}, in_axes=0)(pre)[0]
    % endif
    outputs.append(gx_${i})
    % endfor
    gx = jnp.stack(outputs, axis=0)
    return ${jaxcode(coupling.post_expression.rhs, parameters=['gx'] + coupling_param_names)}
% else:
    % if not scalar_pre:
    def op(x): return jnp.sum(weights * x, axis=-1)
    gx = jax.vmap(op, in_axes=0)(pre)
    % else:
    def op(x): return weights @ x
    gx = jax.vmap(op, in_axes=0)(pre)
    % endif
    return ${jaxcode(coupling.post_expression.rhs, parameters=['gx'] + coupling_param_names)}
% endif

