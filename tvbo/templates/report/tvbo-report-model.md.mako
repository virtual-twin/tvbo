<%doc>
Model / Dynamics Report Template
================================

Embeddable methods-style block describing a Dynamics (neural mass) model.
No '#' headings: section labels use bold so the rendered block can be
copied into a manuscript chapter, slide deck or larger report without
breaking the host document's heading hierarchy.

Order (mirrors a typical "Model" methods sub-section):
  1. Title (bold) + description
  2. State Equations
  3. Derived Variables (the "where ..." block)
  4. Functions (helpers used in the equations)
  5. State Variables (initial conditions table)
  6. Parameters table
  7. Derived Parameters (auxiliary expressions)
  8. References
</%doc>
<%
from sympy import latex, Eq, symbols, sympify, Symbol, Function, Derivative
from tvbo.utils import report

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

def _slot(obj, name, default=None):
    return getattr(obj, name, default) if obj is not None else default

def _present(value):
    return value not in (None, '', [], {})

# Cell formatters live in the adapter (tvbo.utils.report) to avoid duplicating
# them across the report templates; alias for the local call sites below.
_unit_text = report.unit_text
_range_text = report.range_text
_distribution_text = report.distribution_text
_metadata_text = report.metadata_text
_flag_text = report.flag_text

if 'experiment' in context.keys():
    model = context.get('experiment').dynamics
else:
    model = context.get('dynamics', context.get('model'))

state_equations = [eq for k, eq in model.get_equations().items() if k in model.state_variables]

derived_variables = [eq for k, eq in model.get_equations().items() if k in model.derived_variables]

derived_parameters = [
    Eq(symbols(p.name), sympify(p.equation.rhs, strict=False))
    for p in model.derived_parameters.values()
]

functions = [
    Eq(
        Function(f.name)(*[
            Symbol(arg.name if hasattr(arg, 'name') else str(arg))
            for arg in (f.arguments.values() if hasattr(f.arguments, 'values') else f.arguments)
        ]),
        sympify(f.equation.rhs, strict=False),
    )
    for f in model.functions.values()
]

outputs = list(model.output or [])
coupling_inputs = getattr(model, 'coupling_inputs', {}) or {}
coupling_terms = getattr(model, 'coupling_terms', {}) or {}
model_summary = []
if getattr(model, 'model_type', None):
    model_summary.append(f"type: {model.model_type}")
if getattr(model, 'system_type', None):
    model_summary.append(f"system: {model.system_type}")
if getattr(model, 'autonomous', None) is not None:
    model_summary.append(f"autonomous: {model.autonomous}")
if getattr(model, 'number_of_modes', None):
    model_summary.append(f"modes: {model.number_of_modes}")
model_summary.append(f"state variables: {len(model.state_variables)}")
model_summary.append(f"parameters: {len(model.parameters)}")
if model.derived_variables:
    model_summary.append(f"derived variables: {len(model.derived_variables)}")
if model.derived_parameters:
    model_summary.append(f"derived parameters: {len(model.derived_parameters)}")
if model.functions:
    model_summary.append(f"functions: {len(model.functions)}")
if outputs:
    model_summary.append("outputs: " + ", ".join(outputs))

# Resolve references
refs_src = None
if getattr(model, 'ontology', None) is not None:
    refs_src = getattr(model.ontology, 'has_reference', None)
if not refs_src:
    refs_src = getattr(model, 'has_reference', None)
ref_names = []
if refs_src:
    if isinstance(refs_src, str):
        refs_src = [refs_src]
    for r in refs_src:
        name = getattr(r, 'name', None) or (r if isinstance(r, str) else None)
        if name:
            ref_names.append(name)
if not ref_names:
    raw_refs = getattr(model, 'references', None) or []
    if isinstance(raw_refs, str):
        raw_refs = [raw_refs]
    ref_names = list(raw_refs)
%>\
**${model.name}**

% if model.description:
${model.description}

% endif
${'; '.join(model_summary)}.

% if state_equations:
**State Equations**

${'\n'.join([f"$$\n{latex_equation(eq, mul_symbol='*')}\n$$" for eq in state_equations])}

% endif
% if derived_variables:
where

${'\n'.join([f"$$\n{latex_equation(eq, mul_symbol='*')}\n$$" for eq in derived_variables])}

% endif
% if functions:
with

${'\n'.join([f"$$\n{latex_equation(eq, mul_symbol='*')}\n$$" for eq in functions])}

% endif
% if model.state_variables:
**State Variables**

${report.state_variable_table(model.state_variables)}

% endif
% if model.parameters:
**Parameters**

${report.parameter_table(model.parameters)}

% endif
% if coupling_inputs:
**Coupling Inputs**

| Input | Source | Dimension | Keys | Description |
|:------|:-------|----------:|:-----|:------------|
% for input_name, input_obj in coupling_inputs.items():
| ${input_name} | ${_slot(input_obj, 'source', '—') or '—'} | ${_slot(input_obj, 'dimension', 1)} | ${', '.join(_slot(input_obj, 'keys', []) or []) or '—'} | ${_slot(input_obj, 'description', '') or ''} |
% endfor

% endif
% if coupling_terms:
**Coupling Terms**

${report.param_table(coupling_terms, name_header='Term')}

% endif
% if derived_parameters:
**Derived Parameters**

${'\n'.join([f"$$\n{latex_equation(eq, mul_symbol='*')}\n$$" for eq in derived_parameters])}

% endif
% if ref_names:
**References**

${"\n\n".join([report.get_citation(n) for n in ref_names])}
% endif
