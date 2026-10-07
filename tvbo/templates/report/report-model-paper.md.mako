<%
from sympy import latex, Symbol
from tvbo.utils import report

derivative_notation = context.get('derivative_notation', 'd')

def latex_equation(eq):
    return report.equation_latex(eq, derivative_notation, None, 'dot')

def format_aligned_equations(equations):
    lines = [latex_equation(eq).replace('=', '&=') for eq in equations]
    joined = ' \\\\\n'.join(lines)
    return f"$$\n\\begin{{aligned}}\n{joined}\n\\end{{aligned}}\n$$"

_equations = report.model_equation_groups(model)
state_equations = _equations['state']
derived_variables = _equations['derived']
derived_parameters = _equations['derived_parameters']
functions = _equations['functions']
output = _equations['output']

rows = "\n".join([
    f"${latex(Symbol(p.name))}$ & {p.value} & {p.unit if p.unit else '1'} & {p.definition or p.description} \\\\"
    for p in model.parameters.values()
])


table_latex = (
    "\\begin{center}\n"
    "\\begin{tabular}{l l l p{10cm}}\n"
    "\\textbf{Parameter} & \\textbf{Value} & \\textbf{Unit} & \\textbf{Description} \\\\\n"
    "\\hline\n"
    f"{rows}\n"
    "\\end{tabular}\n"
    "\\end{center}\n"
)

%># ${model.name}
${model.description if model.description else ""}

${"### Equations"}
${format_aligned_equations(state_equations)}

with

% if derived_parameters:
${format_aligned_equations(derived_parameters)}
% endif
% if functions:
${format_aligned_equations(functions)}
% endif
% if derived_variables:
${format_aligned_equations(derived_variables)}
% endif

% if output:
${format_aligned_equations(output)}
% endif

${"### Parameters"}

${table_latex}

${"### References"}
${"\n\n".join([report.get_citation(r.name) for r in model.ontology.has_reference])}
