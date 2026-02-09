## -*- coding: utf-8 -*-
<%doc>
NetworkDynamics.jl EdgeModel from tvbo Coupling.

tvbo coupling has:
  pre_expression: per-edge function of x_j (source), x_i (destination)
  post_expression: applied after aggregation

In NetworkDynamics.jl the edge g! computes the pre_expression.
The post_expression scaling is folded into the vertex f!.

Supports multi-dimensional coupling: when outdim > 1 (e.g. 2D diffusion),
the edge function uses broadcasting (e_dst .= v_src .- v_dst) and outsym
lists multiple flow symbols.

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

# Detect antisymmetric pattern (f(x_j) - f(x_i))
is_antisymmetric = 'x_j' in pre_rhs and 'x_i' in pre_rhs and '-' in pre_rhs

# For multi-dim: decide between scalar indexing and broadcasting
if outdim > 1 and is_antisymmetric:
    # Pure difference: use broadcasting  e_dst .= v_src .- v_dst
    # Replace x_j/x_i with v_src/v_dst (no indexing)
    julia_pre = juliacode(pre_rhs).replace('x_j', 'v_src').replace('x_i', 'v_dst')
    use_broadcast = True
else:
    # Scalar or non-trivial expression: use [1] indexing
    julia_pre = juliacode(pre_rhs).replace('x_j', 'v_src[1]').replace('x_i', 'v_dst[1]')
    use_broadcast = False

# Output symbol names
if outsym_names is None:
    outsym_names = ['coupling']
%>

## ── Edge coupling function ──────────────────────────────────────────────────
% if has_params:
function ${coupling.name}_edge_g!(e_dst, v_src, v_dst, (${", ".join(cparam_names)},), t)
% else:
function ${coupling.name}_edge_g!(e_dst, v_src, v_dst, p, t)
% endif
% if use_broadcast:
    e_dst .= ${julia_pre}
% else:
    e_dst[1] = ${julia_pre}
% endif
    nothing
end

% if is_antisymmetric and not is_directed:
edge_${coupling.name} = EdgeModel(;
    g = AntiSymmetric(${coupling.name}_edge_g!),
    outsym = [${", ".join(f':{s}' for s in outsym_names)}],
    % if has_params:
    psym = [${", ".join(f':{p} => {coupling.parameters[p].value}' for p in cparam_names)}],
    % endif
    name = :${coupling.name},
)
% else:
edge_${coupling.name} = EdgeModel(;
    g = Directed(${coupling.name}_edge_g!),
    outsym = [${", ".join(f':{s}' for s in outsym_names)}],
    % if has_params:
    psym = [${", ".join(f':{p} => {coupling.parameters[p].value}' for p in cparam_names)}],
    % endif
    name = :${coupling.name},
)
% endif
