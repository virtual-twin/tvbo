# -*- coding: utf-8 -*-
<%
from tvbo.codegen import render_expression

# Generic jaxcode - pass parameters on each call
jaxcode = lambda expr, parameters=None: render_expression(expr, format='jax', parameters=parameters)

coupling = context['coupling']
model = context.get('model')

_has_coupling = coupling is not None

if _has_coupling:
    # Collect coupling parameter names for use in expressions
    coupling_param_names = [par.name for par in coupling.parameters.values()] if coupling.parameters else []

    has_delay = coupling.delayed and (experiment.horizon > 1)

    from tvbo.templates.base.utils import coupling_bindings
    _b = coupling_bindings(model, coupling)
    vec_states, cvar_names, _cvar_index, _sv_index = (
        _b['vec_states'], _b['cvar_names'], _b['cvar_index'], _b['sv_index'])
    bare_states, pre_j_aliases, pre_i_aliases, post_i_aliases = (
        _b['bare'], _b['pre_j'], _b['pre_i'], _b['post_i'])
    pre_rhs, needs_x_i, needs_x_j, is_list_expr, gx_indices = (
        _b['pre_rhs'], _b['needs_x_i'], _b['needs_x_j'], _b['is_list_expr'], _b['gx_indices'])
    pre_code = jaxcode(pre_rhs, parameters=coupling_param_names + vec_states + [a for a, _ in pre_j_aliases] + [a for a, _ in pre_i_aliases])
%>

% if not _has_coupling:
def cfun(weights, history, current_state, p, delay_indices, t):
    return jnp.zeros_like(current_state[0])
% else:
def cfun(weights, history, current_state, p, delay_indices, t):
    n_node = weights.shape[0]
## Unconditional emission yields the bare `= p.` for a coupling with no parameters.
% if coupling_param_names:
    ${', '.join(coupling_param_names)} = p.${', p.'.join(coupling_param_names)}
% endif

% if needs_x_i:
    x_i = jnp.array([
% for name in cvar_names:
    current_state[${_sv_index[name]}, :],
% endfor
    ])
    x_i = x_i.transpose(1, 0)
    x_i = jnp.expand_dims(x_i, axis=-1)
% endif

% if needs_x_j:
    x_j = jnp.array([
% for cvar_idx, name in enumerate(cvar_names):
    % if has_delay:
        % if small_dt:
    history[${cvar_idx}, delay_indices[0].T + t, delay_indices[1]],
        % else:
    history[${cvar_idx}, delay_indices[0].T, delay_indices[1]],
        % endif
    % else:
        % if not scalar_pre:
    current_state[${_sv_index[name]}, delay_indices[1]],
        % else:
    current_state[${_sv_index[name]}],
        % endif
    % endif
% endfor
    ])
% if not scalar_pre:
% endif

% for state_name in bare_states:
    ${state_name} = x_j[${_cvar_index[state_name]}]
% endfor
% for alias, row in pre_j_aliases:
    ${alias} = x_j[${row}]
% endfor
% for alias, sv_idx in pre_i_aliases:
    ${alias} = current_state[${sv_idx}]
% endfor
% endif

% if is_list_expr:
    pre = jnp.stack(${pre_code}, axis=0)
% else:
    pre = ${pre_code}
% endif
    % if not scalar_pre:
    pre = pre.reshape(-1, n_node ,n_node)
    %endif

% if not scalar_pre:
    def op(x):
        return jnp.sum(weights * x, axis=-1)
% else:
    def op(x):
        return weights @ x
% endif
    gx = jax.vmap(op, in_axes=0)(pre)
% for k in gx_indices:
    gx_${k} = gx[${k}]
% endfor
% for alias, sv_idx in post_i_aliases:
    ${alias} = current_state[${sv_idx}]
% endfor
    return ${jaxcode(coupling.post_expression.rhs, parameters=['gx'] + coupling_param_names)}
% endif
