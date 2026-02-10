## -*- coding: utf-8 -*-
<%doc>
MTK EdgeModel from tvbo Coupling using @component (MTK v11+).

Generates:
  - @component function returning System with input/output annotations
  - EdgeModel(...) with appropriate symmetry wrapper

tvbo coupling variables:
  - x_j (source vertex output) -> srcin
  - x_i (destination vertex output) -> dstin
  - outsym -> output (flow/coupling)

Symmetry:
  - antisymmetric -> AntiSymmetric([outputs])
  - symmetric -> Symmetric([outputs])
  - directed -> Directed([outputs])

Context: coupling (Coupling instance), outdim (int), outsym_names (list)
</%doc>
<%page args="coupling, is_directed=False, outdim=1, outsym_names=None"/>
<%!
from tvbo.export.code import render_expression
%>
<%
cparam_names = list(coupling.parameters.keys()) if coupling.parameters else []

# Output symbol names
coupling_outsym = list(coupling.outsym) if getattr(coupling, 'outsym', None) else None
if coupling_outsym:
    outsym_names = coupling_outsym
elif outsym_names is None:
    outsym_names = ['coupling']

# Observed variables
coupling_obs = list((coupling.observed or {}).values()) if getattr(coupling, 'observed', None) else []

# Determine symmetry from coupling metadata
symmetry = str(getattr(coupling, 'symmetry', '') or '').lower()

# All symbols for expression parsing
all_symbols = cparam_names + ['x_j', 'x_i'] + outsym_names + [obs.name for obs in coupling_obs]
juliacode = lambda expr: render_expression(expr, format='mtk', parameters=all_symbols)

# Get pre-expression
pre_rhs = str(coupling.pre_expression.rhs) if coupling.pre_expression else "x_j - x_i"
is_custom_body = '\n' in pre_rhs.strip() or 'e_dst[' in pre_rhs

# For standard expressions: substitute x_j/x_i with src/dst variable names
if not is_custom_body:
    julia_pre = juliacode(pre_rhs)
    edge_sv = outsym_names[0] if outsym_names else 'coupling'

# Determine edge state variables (from outsym that appear in pre_expression RHS)
edge_svs = set()
if hasattr(coupling, 'pre_expression') and coupling.pre_expression:
    for s in outsym_names:
        if s in pre_rhs:
            edge_svs.add(s)

# Determine if edge is dynamic (has state variables) or static (algebraic)
is_dynamic = len(edge_svs) > 0
%>

## -- @component ${coupling.name} --------------------------------------------
@component function ${coupling.name}(; name)
% if cparam_names:
    @parameters begin
% for p_name in cparam_names:
<%
    p = coupling.parameters[p_name]
    p_val = p.value if p.value is not None else ''
    p_desc = p.description or p.label or ''
    meta = []
    if p_desc:
        meta.append(f'description="{p_desc}"')
    meta_str = ', [' + ', '.join(meta) + ']' if meta else ''
    default_str = f' = {p_val}' if p_val != '' else ''
%>
        ${p_name}${default_str}${meta_str}
% endfor
    end
% endif
    @variables begin
        ## Source/destination inputs
        p_src(t), [input=true]
        p_dst(t), [input=true]
        ## Edge output(s)
% for s in outsym_names:
<%
    s_init = ''
    s_meta = ['output=true']
    s_meta_str = ', [' + ', '.join(s_meta) + ']'
    s_default = f'=1' if is_dynamic and s in edge_svs else ''
%>
        ${s}(t)${s_default}${s_meta_str}
% endfor
% for obs in coupling_obs:
<%
    obs_desc = obs.description or ''
    obs_meta = []
    if obs_desc:
        obs_meta.append(f'description="{obs_desc}"')
    obs_meta_str = ', [' + ', '.join(obs_meta) + ']' if obs_meta else ''
%>
        ${obs.name}(t)${obs_meta_str}
% endfor
    end
    eqs = [
% for obs in coupling_obs:
<%
    obs_rhs = str(obs.equation.rhs) if obs.equation else obs.name
    obs_julia = juliacode(obs_rhs)
%>
        ${obs.name} ~ ${obs_julia},
% endfor
% if is_custom_body:
        ## Custom multi-line edge equations
% for line in pre_rhs.strip().splitlines():
        ${line.strip()},
% endfor
% elif is_dynamic:
<%
    # Dynamic edge: Dt(q) ~ expression with x_j->p_src, x_i->p_dst
    dyn_rhs = julia_pre.replace('x_j', 'p_src').replace('x_i', 'p_dst')
%>
        Dt(${edge_sv}) ~ ${dyn_rhs},
% else:
<%
    # Static edge: q ~ expression
    static_rhs = julia_pre.replace('x_j', 'p_src').replace('x_i', 'p_dst')
%>
        ${edge_sv} ~ ${static_rhs},
% endif
    ]
    return System(eqs, t; name)
end

## -- EdgeModel ---------------------------------------------------------------
<%
    sym_wrapper = 'AntiSymmetric' if symmetry == 'antisymmetric' else ('Symmetric' if not is_directed else 'Directed')
    if not symmetry:
        # Infer from expression pattern
        is_antisymmetric = not is_custom_body and 'x_j' in pre_rhs and 'x_i' in pre_rhs and '-' in pre_rhs
        sym_wrapper = 'AntiSymmetric' if (is_antisymmetric and not is_directed) else ('Directed' if is_directed else 'Symmetric')
%>
@named ${coupling.name.lower()}_mtk = ${coupling.name}()
edge_${coupling.name} = EdgeModel(${coupling.name.lower()}_mtk, [:p_src], [:p_dst], ${sym_wrapper}([:${", :".join(outsym_names)}]))

