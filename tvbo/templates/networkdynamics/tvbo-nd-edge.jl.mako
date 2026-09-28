## -*- coding: utf-8 -*-
<%doc>
NetworkDynamics.jl EdgeModel from tvbo Coupling.

A tvbo coupling gives node i the input post(sum_j w_ij * pre(x_i, x_j)). The edge j -> i evaluates the weighted pre-expression, with x_j the source's output (v_src) and x_i the destination's (v_dst); NetworkDynamics sums the edges into each node, and the vertex f! applies the post-expression to that sum through the function post_function defines, which the experiment template emits ahead of the vertex models that call it. Every graph is a SimpleDiGraph, so the edge is always Directed: its output goes to the destination alone.

Supports:
  - Single-line expressions: e_dst[1] = w * (K * sin(v_src[1] - v_dst[1]))
  - Multi-line custom edge functions: beam force, etc.
  - Multi-dimensional coupling: outdim > 1 (e.g. 2D diffusion, 2D beams)
  - Observed functions (obsf/obssym): post-hoc edge observables
  - Coupling-defined outsym: explicit output symbol names

Context: coupling (Coupling instance), split (its coupling_split: the weight parameter, the edge and post parameters, the post-expression), outdim (int), outsym_names (list[str])
</%doc>
<%page args="coupling, split, outdim=1, outsym_names=None"/>
<%!
from tvbo.codegen import render_expression
%>
<%
edge_names = [name for name, _ in split['edge_parameters']]
weight = split['weight']
implicit_weight = split['implicit_weight']

# All symbol names: coupling params + placeholder coupling variables
all_symbols = edge_names + ['x_j', 'x_i']
juliacode = lambda expr: render_expression(expr, format='julia', parameters=all_symbols)

# Get pre-expression
pre_rhs = str(coupling.pre_expression.rhs) if coupling.pre_expression else "v_src[1] - v_dst[1]"

# Detect multi-line custom function body (contains newlines or e_dst assignments)
is_custom_body = '\n' in pre_rhs.strip() or 'e_dst[' in pre_rhs

# Detect antisymmetric pattern (f(x_j) - f(x_i)), which a multi-dimensional edge broadcasts over every output
is_antisymmetric = not is_custom_body and 'x_j' in pre_rhs and 'x_i' in pre_rhs and '-' in pre_rhs

# For standard expressions: translate x_j/x_i to v_src/v_dst
if not is_custom_body:
    if outdim > 1 and is_antisymmetric:
        julia_pre = juliacode(pre_rhs).replace('x_j', 'v_src').replace('x_i', 'v_dst')
        use_broadcast = True
    else:
        julia_pre = juliacode(pre_rhs).replace('x_j', 'v_src[1]').replace('x_i', 'v_dst[1]')
        use_broadcast = False

# Output symbol names: prefer coupling.outsym, then fallback
coupling_outsym = list(coupling.outsym) if getattr(coupling, 'outsym', None) else None
if coupling_outsym:
    outsym_names = coupling_outsym
elif outsym_names is None:
    outsym_names = ['coupling']

# Observed variables (explicit definitions only - no auto-generation)
coupling_obs = list((coupling.observed or {}).values()) if getattr(coupling, 'observed', None) else []
has_observed = len(coupling_obs) > 0
%>

## ── Edge coupling function ──────────────────────────────────────────────────
function ${coupling.name}_edge_g!(e_dst, v_src, v_dst, (${", ".join(edge_names)},), t)
% if is_custom_body:
    ## Custom multi-line edge function body
% for line in pre_rhs.strip().splitlines():
    ${line.strip()}
% endfor
% if implicit_weight:
    e_dst .*= ${weight}
% endif
% elif use_broadcast:
    e_dst .= ${f"{weight} .* ({julia_pre})" if implicit_weight else julia_pre}
% else:
    e_dst[1] = ${f"{weight} * ({julia_pre})" if implicit_weight else julia_pre}
% endif
    nothing
end
% if has_observed:

## ── Edge observed function ──────────────────────────────────────────────────
function ${coupling.name}_obsf!(obsout, u, v_src, v_dst, (${", ".join(edge_names)},), t)
% for i, obs in enumerate(coupling_obs):
<%
    obs_rhs = str(obs.equation.rhs).strip()
    is_multiline_obs = '\n' in obs_rhs or 'obsout[' in obs_rhs
%>
    ## ${obs.name}: ${obs.description or ''}
% if is_multiline_obs:
    ## Custom multi-line observed body
% for line in obs_rhs.splitlines():
    ${line.strip()}
% endfor
% else:
<%
    obs_code = obs_rhs.replace('x_j', 'v_src[1]').replace('x_i', 'v_dst[1]')
%>
    obsout[${i+1}] = ${obs_code}
% endif
% endfor
    nothing
end
% endif

edge_${coupling.name} = EdgeModel(;
    g = Directed(${coupling.name}_edge_g!),
    outsym = [${", ".join(f':{s}' for s in outsym_names)}],
    psym = [${", ".join(f':{name} => {value}' for name, value in split['edge_parameters'])}],
    % if has_observed:
    obsf = ${coupling.name}_obsf!,
    obssym = [${", ".join(f':{obs.name}' for obs in coupling_obs)}],
    % endif
    name = :${coupling.name},
)
<%def name="post_function(split)">\
<% post_names = [name for name, _, _ in split['post_parameters']] %>\
% if split['post'] is not None and post_names:
${split['post_function']}(gx, (${", ".join(post_names)},)) = ${render_expression(split['post'], format='julia', parameters=post_names + ['gx'])}
% elif split['post'] is not None:
${split['post_function']}(gx) = ${render_expression(split['post'], format='julia', parameters=['gx'])}
% endif
</%def>
