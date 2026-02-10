## -*- coding: utf-8 -*-
<%doc>
MTK VertexModel from tvbo Dynamics using @component (MTK v11+).

Generates:
  - @component function returning System with @variables, @parameters, equations
  - VertexModel(...) wrapping the MTK system

Input/output annotations:
  - coupling_terms -> [input=true]
  - coupling_variable=true state vars -> [output=true]

Context: model (Dynamics instance)
</%doc>
<%page args="model, all_couplings=None"/>
<%!
from tvbo.export.code import render_expression
%>
<%
sv_names = list(model.state_variables.keys())
param_names = list(model.parameters.keys())
ct_names = list(model.coupling_terms.keys()) if model.coupling_terms else []
dv_names = list(model.derived_variables.keys()) if model.derived_variables else []
dp_names = list(model.derived_parameters.keys()) if model.derived_parameters else []
n_sv = len(sv_names)

# Output variables: those marked coupling_variable=true, or all SVs
coupling_vars = [name for name, sv in model.state_variables.items()
                 if getattr(sv, 'coupling_variable', False)]
output_vars = coupling_vars if coupling_vars else sv_names

# All symbol names for the expression parser
all_symbols = sv_names + param_names + ct_names + dv_names + dp_names
juliacode = lambda expr: render_expression(expr, format='mtk', parameters=all_symbols)
%>

## -- @component ${model.name} -----------------------------------------------
@component function ${model.name}(; name)
    @parameters begin
% for p_name, p in model.parameters.items():
<%
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
    @variables begin
% for ct_name in ct_names:
<%
    ct = model.coupling_terms[ct_name]
    ct_desc = ct.description if hasattr(ct, 'description') and ct.description else ''
    ct_meta = ['input=true']
    if ct_desc:
        ct_meta.append(f'description="{ct_desc}"')
    ct_meta_str = ', [' + ', '.join(ct_meta) + ']'
%>
        ${ct_name}(t)${ct_meta_str}
% endfor
% for sv_name, sv in model.state_variables.items():
<%
    sv_init = sv.initial_value if sv.initial_value is not None else ''
    sv_desc = sv.description or sv.label or ''
    is_output = sv_name in output_vars
    meta = []
    if is_output:
        meta.append('output=true')
    if sv_desc:
        meta.append(f'description="{sv_desc}"')
    meta_str = ', [' + ', '.join(meta) + ']' if meta else ''
    default_str = f'={sv_init}' if sv_init != '' else ''
%>
        ${sv_name}(t)${default_str}${meta_str}
% endfor
% for dv_name, dv in (model.derived_variables or {}).items():
<%
    dv_desc = dv.description or ''
    dv_meta = []
    if dv_desc:
        dv_meta.append(f'description="{dv_desc}"')
    dv_meta_str = ', [' + ', '.join(dv_meta) + ']' if dv_meta else ''
%>
        ${dv_name}(t)${dv_meta_str}
% endfor
    end
    eqs = [
% for dp in (model.derived_parameters or {}).values():
        ${dp.name} ~ ${juliacode(dp.equation.rhs)},
% endfor
% for dv in (model.derived_variables or {}).values():
<%
    is_conditional = getattr(dv, 'conditional', False) and getattr(dv, 'cases', None)
%>
% if is_conditional:
<%
    # Build nested ifelse chain from cases
    cases = list(dv.cases)
    def build_ifelse(cases):
        if len(cases) == 1:
            return juliacode(cases[0].equation.rhs)
        c = cases[0]
        cond = str(c.condition).strip()
        if cond.lower() == 'true':
            return juliacode(c.equation.rhs)
        expr = juliacode(c.equation.rhs)
        rest = build_ifelse(cases[1:])
        return f'ifelse({cond}, {expr}, {rest})'
    ifelse_expr = build_ifelse(cases)
%>
        ${dv.name} ~ ${ifelse_expr},
% else:
        ${dv.name} ~ ${juliacode(dv.equation.rhs)},
% endif
% endfor
% for sv_name, sv in model.state_variables.items():
<%
    eq_type = getattr(sv, 'equation_type', None)
    is_algebraic = str(eq_type) == 'algebraic' if eq_type else False
%>
% if is_algebraic:
        ${sv_name} ~ ${juliacode(sv.equation.rhs)},
% else:
        Dt(${sv_name}) ~ ${juliacode(sv.equation.rhs)},
% endif
% endfor
    ]
    return System(eqs, t; name)
end

