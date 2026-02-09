## -*- coding: utf-8 -*-
<%doc>
NetworkDynamics.jl EdgeModel from tvbo Coupling.

tvbo coupling has:
  pre_expression: per-edge function of x_j (source), x_i (destination)
  post_expression: applied after aggregation

In NetworkDynamics.jl the edge g! computes the pre_expression.
The post_expression scaling is folded into the vertex f!.

Supports:
  - Single-line expressions: e_dst[1] = K * sin(v_src[1] - v_dst[1])
  - Multi-line custom edge functions: beam force, etc.
  - Multi-dimensional coupling: outdim > 1 (e.g. 2D diffusion, 2D beams)
  - Observed functions (obsf/obssym): post-hoc edge observables
  - Coupling-defined outsym: explicit output symbol names

Context: coupling (Coupling instance), outdim (int), outsym_names (list[str])
</%doc>
<%page args="coupling, is_directed=False, outdim=1, outsym_names=None"/>
<%!
from tvbo.export.code import render_expression
%>
<%
cparam_names = list(coupling.parameters.keys()) if coupling.parameters else []
has_params = len(cparam_names) > 0

# All symbol names: coupling params + placeholder coupling variables
all_symbols = cparam_names + ['x_j', 'x_i']
juliacode = lambda expr: render_expression(expr, format='julia', parameters=all_symbols)

# Get pre-expression
pre_rhs = str(coupling.pre_expression.rhs) if coupling.pre_expression else "v_src[1] - v_dst[1]"

# Detect multi-line custom function body (contains newlines or e_dst assignments)
is_custom_body = '\n' in pre_rhs.strip() or 'e_dst[' in pre_rhs

# Detect antisymmetric pattern (f(x_j) - f(x_i))
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
% if has_params:
function ${coupling.name}_edge_g!(e_dst, v_src, v_dst, (${", ".join(cparam_names)},), t)
% else:
function ${coupling.name}_edge_g!(e_dst, v_src, v_dst, p, t)
% endif
% if is_custom_body:
    ## Custom multi-line edge function body
% for line in pre_rhs.strip().splitlines():
    ${line.strip()}
% endfor
% elif use_broadcast:
    e_dst .= ${julia_pre}
% else:
    e_dst[1] = ${julia_pre}
% endif
    nothing
end
% if has_observed:

## ── Edge observed function ──────────────────────────────────────────────────
function ${coupling.name}_obsf!(obsout, u, v_src, v_dst, (${", ".join(cparam_names)},), t)
% for i, obs in enumerate(coupling_obs):
<%
    obs_rhs = str(obs.equation.rhs).strip()
    # Translate variable names directly without sympy processing to avoid rewriting
    obs_code = obs_rhs.replace('x_j', 'v_src[1]').replace('x_i', 'v_dst[1]')
%>
    ## ${obs.name}: ${obs.description or ''}
    obsout[${i+1}] = ${obs_code}
% endfor
    nothing
end
% endif

% if (is_antisymmetric or is_custom_body) and not is_directed:
edge_${coupling.name} = EdgeModel(;
    g = AntiSymmetric(${coupling.name}_edge_g!),
    outsym = [${", ".join(f':{s}' for s in outsym_names)}],
    % if has_params:
    psym = [${", ".join(f':{p} => {coupling.parameters[p].value}' for p in cparam_names)}],
    % endif
    % if has_observed:
    obsf = ${coupling.name}_obsf!,
    obssym = [${", ".join(f':{obs.name}' for obs in coupling_obs)}],
    % endif
    name = :${coupling.name},
)
% elif is_directed:
edge_${coupling.name} = EdgeModel(;
    g = Directed(${coupling.name}_edge_g!),
    outsym = [${", ".join(f':{s}' for s in outsym_names)}],
    % if has_params:
    psym = [${", ".join(f':{p} => {coupling.parameters[p].value}' for p in cparam_names)}],
    % endif
    % if has_observed:
    obsf = ${coupling.name}_obsf!,
    obssym = [${", ".join(f':{obs.name}' for obs in coupling_obs)}],
    % endif
    name = :${coupling.name},
)
% else:
edge_${coupling.name} = EdgeModel(;
    g = Symmetric(${coupling.name}_edge_g!),
    outsym = [${", ".join(f':{s}' for s in outsym_names)}],
    % if has_params:
    psym = [${", ".join(f':{p} => {coupling.parameters[p].value}' for p in cparam_names)}],
    % endif
    % if has_observed:
    obsf = ${coupling.name}_obsf!,
    obssym = [${", ".join(f':{obs.name}' for obs in coupling_obs)}],
    % endif
    name = :${coupling.name},
)
% endif
