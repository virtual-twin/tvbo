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
from tvbo.utils.units import unit_to_latex

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

def _unit_text(unit):
    unit_ltx = unit_to_latex(unit) if unit else ''
    return '$' + unit_ltx + '$' if unit_ltx else '—'

def _range_text(range_obj):
    if not range_obj:
        return ''
    values = _slot(range_obj, 'explored_values', None)
    if values:
        values = [str(v) for v in values]
        return '{' + ', '.join(values[:8]) + ('...' if len(values) > 8 else '') + '}'
    lo = _slot(range_obj, 'lo', None)
    hi = _slot(range_obj, 'hi', None)
    step = _slot(range_obj, 'step', None)
    n_points = _slot(range_obj, 'n', None)
    log_scale = _slot(range_obj, 'log_scale', False)
    parts = []
    if lo is not None or hi is not None:
        parts.append(f"[{lo if lo is not None else '-∞'}, {hi if hi is not None else '∞'}]")
    if step is not None:
        parts.append(f"step={step}")
    if n_points is not None:
        parts.append(f"n={n_points}")
    if log_scale:
        parts.append('log')
    return ', '.join(parts)

def _distribution_text(distribution):
    if not distribution:
        return ''
    name = _slot(distribution, 'name', 'Distribution')
    domain = _range_text(_slot(distribution, 'domain', None))
    axis = _slot(distribution, 'axis', None)
    seed = _slot(distribution, 'seed', None)
    parts = [str(name)]
    if domain:
        parts.append(domain)
    if axis:
        parts.append(f"axis={axis}")
    if seed is not None:
        parts.append(f"seed={seed}")
    return ' '.join(parts)

def _metadata_text(obj):
    from tvbo.utils import domain_enforcement
    bits = []
    _dom = _slot(obj, 'domain', None)
    if _present(_dom):
        bits.append(_range_text(_dom))
        _enf = domain_enforcement(_dom)   # none / clamp / wrap (boundaries folded into domain)
        if _enf != 'none':
            bits.append(f'enforce={_enf}')
    if _present(_slot(obj, 'distribution', None)):
        bits.append(_distribution_text(_slot(obj, 'distribution')))
    return '; '.join([b for b in bits if b]) or '—'

def _flag_text(obj, names):
    flags = []
    for name, label in names:
        if _slot(obj, name, False):
            flags.append(label)
    shape = _slot(obj, 'shape', None)
    if shape:
        flags.append(f"shape={shape}")
    dataset_path = _slot(obj, 'dataset_path', None)
    if dataset_path:
        flags.append(f"data={dataset_path}")
    optimum = _slot(obj, 'reported_optimum', None)
    if optimum is not None:
        flags.append(f"optimum={optimum}")
    return ', '.join(flags) or '—'

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

| Variable | Initial Value | Unit | Equation | Domain / Sampling | Flags | Description |
|:---------|:--------------|:-----|:---------|:------------------|:------|:------------|
${'\n'.join([f"| ${latex(Symbol(sv.name))}$ | {sv.initial_value if sv.initial_value is not None else '—'} | {_unit_text(sv.unit)} | {sv.equation_type or 'differential'} (order {sv.equation_order or 1}) | {_metadata_text(sv)} | {_flag_text(sv, [('coupling_variable', 'coupling'), ('stimulation_variable', 'stimulation'), ('record', 'recorded')])} | {sv.description or sv.definition or ''} |" for sv in model.state_variables.values()])}

% endif
% if model.parameters:
**Parameters**

| Parameter | Value | Default | Unit | Domain / Sampling | Flags | Description |
|:----------|------:|:--------|:-----|:------------------|:------|:------------|
${'\n'.join([f"| ${latex(Symbol(p.name))}$ | {p.value} | {p.default if p.default is not None else '—'} | {_unit_text(p.unit)} | {_metadata_text(p)} | {_flag_text(p, [('free', 'free'), ('heterogeneous', 'heterogeneous')])} | {p.description or p.definition or ''} |" for p in model.parameters.values()])}

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

| Term | Value | Domain / Sampling | Flags | Description |
|:-----|------:|:------------------|:------|:------------|
% for term_name, term in coupling_terms.items():
| $${latex(Symbol(term_name))}$ | ${_slot(term, 'value', '—')} | ${_metadata_text(term)} | ${_flag_text(term, [('free', 'free'), ('heterogeneous', 'heterogeneous')])} | ${_slot(term, 'description', '') or _slot(term, 'definition', '') or ''} |
% endfor

% endif
% if derived_parameters:
**Derived Parameters**

${'\n'.join([f"$$\n{latex_equation(eq, mul_symbol='*')}\n$$" for eq in derived_parameters])}

% endif
% if ref_names:
**References**

${"\n\n".join([report.get_citation(n) for n in ref_names])}
% endif
