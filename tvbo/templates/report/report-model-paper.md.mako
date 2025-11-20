<%
from sympy import latex, Eq, symbols, sympify, Symbol, Function
from tvbo.export import report

def format_aligned_equations(equations):
    lines = [latex(eq, mul_symbol='dot').replace('=', '&=') for eq in equations]
    joined = ' \\\\\n'.join(lines)
    return f"$$\n\\begin{{aligned}}\n{joined}\n\\end{{aligned}}\n$$"

state_equations = [eq for k, eq in model.get_equations().items() if k in model.metadata.state_variables]

derived_variables = [eq for k, eq in model.get_equations().items() if k in model.metadata.derived_variables]

output_transforms = [
    Eq(symbols(p.name), sympify(p.equation.rhs, strict=False))
    for p in model.metadata.output_transforms.values()
]

derived_parameters = [
    Eq(symbols(p.name), sympify(p.equation.rhs, strict=False))
    for p in model.metadata.derived_parameters.values()
]

functions = [Eq(Function(f.name)(*[Symbol(arg) for arg in f.arguments.keys()]), sympify(f.equation.rhs, strict=False)) for f in model.metadata.functions.values()]

rows = "\n".join([
    f"${latex(Symbol(p.name))}$ & {p.value} & {p.unit if p.unit else '1'} & {p.definition or p.description} \\\\"
    for p in model.metadata.parameters.values()
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

%># ${model.metadata.name}
${model.metadata.description if model.metadata.description else ""}

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

% if output_transforms:
${format_aligned_equations(output_transforms)}
% endif

${"### Parameters"}

${table_latex}

${"### References"}
${"\n\n".join([report.get_citation(r.name) for r in model.ontology.has_reference])}
