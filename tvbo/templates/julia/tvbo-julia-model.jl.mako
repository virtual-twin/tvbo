## -*- coding: utf-8 -*-
<%page args="model"/>
<%!
from tvbo.export.code import render_expression
%>
<%
sv_names = list(model.state_variables.keys())
param_names = list(model.parameters.keys())
ct_names = list(model.coupling_terms.keys()) if model.coupling_terms else []
dv_names = list(model.derived_variables.keys()) if model.derived_variables else []
dp_names = list(model.derived_parameters.keys()) if model.derived_parameters else []
all_symbols = sv_names + param_names + ct_names + dv_names + dp_names
juliacode = lambda expr: render_expression(expr, format='julia', parameters=all_symbols)
%>
using SpecialFunctions

function ${model.name}!(dx, x, p, t = 0)

    (;${", ".join(param_names + ct_names)}) = p

    ${", ".join(sv_names)} = x

    ${"\n    ".join([f"{dp.name} = {juliacode(dp.equation.rhs)}" for dp in model.derived_parameters.values()])}

    ${"\n    ".join([f"{dv.name} = {juliacode(dv.equation.rhs)}" for dv in model.derived_variables.values()])}

    ${"\n    ".join([f"dx[{i+1}] = {juliacode(sv.equation.rhs)}" for i, sv in enumerate(model.state_variables.values())])}
    dx
end

# Parameter values
p = (${", ".join([f"{p.name} = {p.value}" for p in model.parameters.values()] + [f"{ct} = 0.0" for ct in ct_names])})
