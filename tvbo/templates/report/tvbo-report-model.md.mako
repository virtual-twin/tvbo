<%

from sympy import latex, Eq, symbols, sympify, Symbol, Function
from tvbo.export import report
from tvbo import Dynamics

if 'experiment' in context.keys():
    model = context.get('experiment').local_dynamics
else:
    model = context.get('local_dynamics', context.get('model'))

state_equations = [eq for k, eq in model.get_equations().items() if k in model.state_variables]

derived_variables = [eq for k, eq in model.get_equations().items() if k in model.derived_variables]

output_transforms = [
    Eq(symbols(p.name), sympify(p.equation.rhs, strict=False))
    for p in model.output_transforms.values()
]

derived_parameters = [
    Eq(symbols(p.name), sympify(p.equation.rhs, strict=False))
    for p in model.derived_parameters.values()
]

functions = [Eq(Function(f.name)(*[Symbol(arg) for arg in f.arguments.keys()]), sympify(f.equation.rhs, strict=False)) for f in model.functions.values()]
%>

# ${model.name}
${model.description if model.description else ""}

${"## Equations"}

% if derived_parameters:
${"### Derived Parameters"}
${'\n'.join([f"$$\n{latex(eq, mul_symbol='*')}\n$$" for eq in derived_parameters])}
% endif
% if functions:
${"### Functions"}
${'\n'.join([f"$$\n{latex(eq, mul_symbol='*')}\n$$" for eq in functions])}
% endif
% if derived_variables:
${"### Derived Variables"}
${'\n'.join([f"$$\n{latex(eq, mul_symbol='*')}\n$$" for eq in derived_variables])}
% endif

${"### State Equations"}
${'\n'.join([f"$$\n{latex(eq, mul_symbol='*')}\n$$" for eq in state_equations])}

% if output_transforms:
${"### Output Transforms"}
${'\n'.join([f"$$\n{latex(eq, mul_symbol='*')}\n$$" for eq in output_transforms])}
% endif

${"## Parameters"}

| **Parameter** | **Value** | **Unit** | **Description** |
|---------------|-----------|----------|-----------------|
${'\n'.join([f"| ${latex(Symbol(p.name))}$ | {p.value} | {p.unit if p.unit else 'N/A'} | {p.description} |" for p in model.parameters.values()])}


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
