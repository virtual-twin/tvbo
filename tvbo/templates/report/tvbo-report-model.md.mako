<%

from sympy import latex, Eq, symbols, sympify, Symbol, Function, Derivative
from tvbo.utils import report
from tvbo.utils.units import unit_to_latex
from tvbo import Dynamics

derivative_notation = context.get('derivative_notation', 'd')

def _dot_lhs(deriv, mul_symbol='*'):
    try:
        t = Symbol("t")
        order = sum(1 for v in deriv.variables if v == t)
        base = deriv.expr
        base_latex = latex(base, mul_symbol=mul_symbol)
        if order == 1:
            return f"\\dot{{{base_latex}}}"
        if order == 2:
            return f"\\ddot{{{base_latex}}}"
        if order == 3:
            return f"\\dddot{{{base_latex}}}"
        return f"\\frac{{d^{order}}}{{d t^{order}}} {base_latex}"
    except Exception:
        return latex(deriv, mul_symbol=mul_symbol)

def latex_equation(eq, mul_symbol='*'):
    if derivative_notation == 'dot' and isinstance(eq, Eq) and isinstance(eq.lhs, Derivative):
        lhs = _dot_lhs(eq.lhs, mul_symbol=mul_symbol)
        rhs = latex(eq.rhs, mul_symbol=mul_symbol)
        return f"{lhs} = {rhs}"
    return latex(eq, mul_symbol=mul_symbol)

if 'experiment' in context.keys():
    model = context.get('experiment').dynamics
else:
    model = context.get('dynamics', context.get('model'))

state_equations = [eq for k, eq in model.get_equations().items() if k in model.state_variables]

derived_variables = [eq for k, eq in model.get_equations().items() if k in model.derived_variables]

if isinstance(model.output, list):
    output = [eq for k, eq in model.get_equations().items() if k in model.output]
else:
    output = [
        Eq(symbols(p.name), sympify(p.equation.rhs, strict=False))
        for p in model.output.values()
    ]

derived_parameters = [
    Eq(symbols(p.name), sympify(p.equation.rhs, strict=False))
    for p in model.derived_parameters.values()
]

functions = [Eq(Function(f.name)(*[Symbol(arg.name if hasattr(arg, 'name') else str(arg)) for arg in (f.arguments.values() if hasattr(f.arguments, 'values') else f.arguments)]), sympify(f.equation.rhs, strict=False)) for f in model.functions.values()]
%>

${'## ' + model.name}
${model.description if model.description else ""}

${"### State Equations"}
${'\n'.join([f"$$\n{latex_equation(eq, mul_symbol='*')}\n$$" for eq in state_equations])}

${"### Parameters"}

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
${'\n'.join([f"| ${latex(Symbol(p.name))}$ | {p.value} | {'$' + unit_to_latex(p.unit) + '$' if p.unit and unit_to_latex(p.unit) else ('—' if p.unit else 'N/A')} | {p.description} |" for p in model.parameters.values()])}

% if derived_parameters or derived_variables or functions:
${"### Derived Quantities"}
% endif
% if derived_parameters:
${"#### Derived Parameters"}
${'\n'.join([f"$$\n{latex_equation(eq, mul_symbol='*')}\n$$" for eq in derived_parameters])}
% endif
% if derived_variables:
${"#### Derived Variables"}
${'\n'.join([f"$$\n{latex_equation(eq, mul_symbol='*')}\n$$" for eq in derived_variables])}
% endif
% if functions:
${"#### Functions"}
${'\n'.join([f"$$\n{latex_equation(eq, mul_symbol='*')}\n$$" for eq in functions])}
% endif

% if output:
${"### Output Transforms"}
${'\n'.join([f"$$\n{latex_equation(eq, mul_symbol='*')}\n$$" for eq in output])}
% endif


<%
refs_src = None
if getattr(model, 'ontology', None) is not None:
    refs_src = getattr(model.ontology, 'has_reference', None)
else:
    refs_src = getattr(model, 'has_reference', None)

# Normalize to a list
refs = list(refs_src) if refs_src else []

# Extract safe names
ref_names = []
for r in refs:
    name = getattr(r, 'name', None)
    if name:
        ref_names.append(name)
%>
% if ref_names:
${"## References"}
${"\n\n".join([report.get_citation(n) for n in ref_names])}
% endif
