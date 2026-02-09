## -*- coding: utf-8 -*-
<%page args="model"/>
<%!
from tvbo.export.code import render_expression
juliacode = lambda expr: render_expression(expr, format='julia')
from tvbo.knowledge.simulation import equations
%>
using SpecialFunctions

function ${model.name}!(dx, x, p, t = 0, local_coupling = 0)

    exp = Base.exp
    sqrt = Base.sqrt
    tanh = Base.tanh
    e = Base.MathConstants.e
    pi = π

    (;${", ".join([p.name for p in model.parameters.values()] + [p.name for p in model.coupling_terms.values()])}) = p

    ${", ".join([sv.name for sv in model.state_variables.values()])} = x

    ${"\n    ".join([f"{dp.name} = {juliacode(dp.equation.rhs)}" for dp in model.derived_parameters.values()])}

    ${"\n    ".join([f"{dv.name} = {juliacode(dv.equation.rhs)}" for dv in model.derived_variables.values()])}

    ${"\n    ".join([f"dx[{i+1}] = {juliacode(sv.equation.rhs)}" for i, sv in enumerate(model.state_variables.values())])}
    dx
end

# Parameter values
p = (${", ".join([f"{p.name} = {p.value}" for p in model.parameters.values()] + [f"{p.name} = 0.0" for p in model.coupling_terms.values()])})
