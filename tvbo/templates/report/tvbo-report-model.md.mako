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
mul_symbol = context.get('mul_symbol', None)

# Per-symbol display overrides {identifier Symbol -> LaTeX string}, populated from the
# model's `symbol` slots once `model` is resolved (see _collect_symbols below). Passed to
# every sympy.latex call so rendered equations use the source's own notation (fully
# sympy-native): e.g. identifier ``w_plus`` renders as ``w_+``, ``S_e`` as ``S^{(E)}``.
symbol_names = {}

def _dot_lhs(deriv, mul_symbol='*'):
    try:
        t = Symbol("t")
        order = sum(1 for v in deriv.variables if v == t)
        base = deriv.expr
        base_latex = latex(base, mul_symbol=mul_symbol, symbol_names=symbol_names)
        if order == 1:
            return f"\\dot{{{base_latex}}}"
        if order == 2:
            return f"\\ddot{{{base_latex}}}"
        if order == 3:
            return f"\\dddot{{{base_latex}}}"
        return f"\\frac{{d^{order}}}{{d t^{order}}} {base_latex}"
    except Exception:
        return latex(deriv, mul_symbol=mul_symbol, symbol_names=symbol_names)

def latex_equation(eq, mul_symbol=mul_symbol):
    if derivative_notation == 'dot' and isinstance(eq, Eq) and isinstance(eq.lhs, Derivative):
        lhs = _dot_lhs(eq.lhs, mul_symbol=mul_symbol)
        rhs = latex(eq.rhs, mul_symbol=mul_symbol, symbol_names=symbol_names)
        return f"{lhs} = {rhs}"
    return latex(eq, mul_symbol=mul_symbol, symbol_names=symbol_names)

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

# Display-symbol overrides {identifier Symbol -> LaTeX}; resolution lives on the model
# (Dynamics.symbol_map), the template only consumes it.
symbol_names.update(model.symbol_map())

# Optional baseline diff: when a `baseline` model is passed, render only what this
# model adds or changes relative to it. report.model_delta does the comparison in
# Python; the template just filters its collections by the returned name sets.
_baseline = context.get('baseline', None)
_delta = report.model_delta(model, _baseline) if _baseline is not None else None

state_equations = [eq for k, eq in model.get_equations().items()
                   if k in model.state_variables and (_delta is None or k in _delta.eq_svars)]

derived_variables = [eq for k, eq in model.get_equations().items()
                     if k in model.derived_variables and (_delta is None or k in _delta.dvars)]

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

# Events (discrete condition -> affect updates, e.g. a spike threshold/reset or a
# spike-driven increment). Rendered from the model's own event declarations so the
# native report is complete for spiking models, not just continuous ODEs.
events_lines = []
for _en, _ev in (getattr(model, 'events', None) or {}).items():
    _cond = _slot(_slot(_ev, 'condition', None), 'rhs', None)
    _aff = _slot(_slot(_ev, 'affect', None), 'rhs', None)
    _parts = []
    if _present(_cond):
        try:
            _c = latex(sympify(str(_cond), strict=False), symbol_names=symbol_names)
        except Exception:
            _c = str(_cond)
        _parts.append(f"when ${_c}$")
    # The affect may be SEVERAL assignments separated by ';' (a spike updates u and x and delivers
    # to v). Parse and render EACH natively with sympy, so multiplication is implicit and every
    # update shows as its own $lhs \leftarrow rhs$ — not the raw semicolon-joined string.
    _updates = []
    for _stmt in str(_aff or '').split(';'):
        _stmt = _stmt.strip()
        if not _stmt or '=' not in _stmt:
            continue
        _l, _r = _stmt.split('=', 1)
        try:
            _updates.append(f"{latex(Symbol(_l.strip()), symbol_names=symbol_names)} \\leftarrow "
                            f"{latex(sympify(_r, strict=False), symbol_names=symbol_names)}")
        except Exception:
            _updates.append(_stmt)
    if _updates:
        _parts.append("$" + ",\\; ".join(_updates) + "$")
    events_lines.append(f"- *{_en}*: " + ", ".join(_parts))

# coupling_inputs is the supported surface; coupling_terms duplicated the same names
# (each input IS a term) and is no longer rendered.
coupling_inputs = getattr(model, 'coupling_inputs', {}) or {}
if _delta is not None:
    coupling_inputs = {n: o for n, o in report.name_items(coupling_inputs) if n in _delta.coupling_inputs}
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
if events_lines:
    model_summary.append(f"events: {len(events_lines)}")
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
# citeformat: 'quarto' -> inline @key citations in the fulltext and NO References list (a host
# Quarto doc's own bibliography resolves them into one bibliography); else a formatted list below.
citeformat = context.get('citeformat', None)
_quarto_cites = ("[" + "; ".join("@" + n for n in ref_names) + "]") if (citeformat == 'quarto' and ref_names) else ""
%>\
**${model.name}**${' ' + _quarto_cites if _quarto_cites else ''}

% if model.description:
${model.description}

% endif
${'; '.join(model_summary)}.

% if _delta is not None:
Shown **relative to the base model** (${_delta.base_label}) — only new or changed state variables, parameters, derived variables and couplings are listed; everything else is inherited unchanged.

% endif
% if state_equations:
**State Equations**

${'\n'.join([f"$$\n{latex_equation(eq)}\n$$" for eq in state_equations])}

% endif
% if derived_variables:
where

${'\n'.join([f"$$\n{latex_equation(eq)}\n$$" for eq in derived_variables])}

% endif
% if functions:
with

${'\n'.join([f"$$\n{latex_equation(eq)}\n$$" for eq in functions])}

% endif
<%
_svars_show = model.state_variables if _delta is None else {n: s for n, s in report.name_items(model.state_variables) if n in _delta.new_svars}
%>\
% if _svars_show:
**State Variables**

${report.state_variable_table(_svars_show)}

% endif
<%
_params_show = model.parameters if _delta is None else {n: p for n, p in report.name_items(model.parameters) if n in _delta.params}
%>\
% if _params_show:
**Parameters**

${report.parameter_table(_params_show)}

% endif
% if coupling_inputs:
<%
# One md_table (empty columns dropped; a scalar-only input list collapses to plain
# text) instead of a hand-rolled table — a trivial dimension=1 carries no information.
ci_rows = [[nm, report.slot(o, "source", ""),
            ("" if report.slot(o, "dimension", "") in (1, "1", None, "") else report.slot(o, "dimension", "")),
            ", ".join(report.slot(o, "keys", []) or []), report.slot(o, "description", "") or ""]
           for nm, o in coupling_inputs.items()]
%>\
**Coupling Inputs**

${report.md_table(["Input", "Source", "Dimension", "Keys", "Description"], ci_rows, aligns=["l", "l", "r", "l", "l"])}

% endif
% if events_lines:
**Events**

${'\n'.join(events_lines)}

% endif
% if derived_parameters:
**Derived Parameters**

${'\n'.join([f"$$\n{latex_equation(eq)}\n$$" for eq in derived_parameters])}

% endif
% if ref_names and citeformat != 'quarto':
**References**

${"\n\n".join([report.get_citation(n) for n in ref_names])}
% endif
