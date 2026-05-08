<%doc>
Coupling Report Template
========================

Embeddable methods-style block for a Coupling instance.
No '#' headings: uses bold labels so the block can be inserted into a
manuscript chapter, slide deck or larger report without breaking the
host document's heading hierarchy.

Order (mirrors a typical "Coupling" methods sub-section):
  1. Description
  2. Full coupling equation
  3. Pre / post decomposition
  4. Incoming / local states & meta info
  5. Parameter table

Context variables:
  - coupling: tvbo.classes.coupling.Coupling instance
</%doc>
<%
from sympy import latex, Symbol
from tvbo.parse.expression import parse_eq

cpl = coupling
cpl_label = getattr(cpl, "label", None) or getattr(cpl, "name", "Coupling")
cpl_desc = getattr(cpl, "description", "") or ""

incoming = list(getattr(cpl, "incoming_states", None) or [])
local = list(getattr(cpl, "local_states", None) or [])

cpl_params_obj = getattr(cpl, "parameters", {}) or {}
if hasattr(cpl_params_obj, "items"):
    cpl_items = list(cpl_params_obj.items())
else:
    cpl_items = [(getattr(p, "name", "?"), p) for p in (list(cpl_params_obj) if cpl_params_obj else [])]
cpl_pnames = [n for n, _ in cpl_items]

extra_syms = (
    cpl_pnames + incoming + local
    + [f"{s}_i" for s in local] + [f"{s}_j" for s in incoming]
    + ["gx", "x_i", "x_j"]
)

def _safe_latex(rhs):
    if not rhs:
        return ""
    try:
        return latex(parse_eq(str(rhs), parameters=extra_syms), mul_symbol="dot")
    except Exception:
        return str(rhs)

def _slot(obj, name, default=None):
    return getattr(obj, name, default) if obj is not None else default

def _range_text(range_obj):
    if not range_obj:
        return ""
    values = _slot(range_obj, "explored_values", None)
    if values:
        values = [str(v) for v in values]
        return "{" + ", ".join(values[:8]) + ("..." if len(values) > 8 else "") + "}"
    lo = _slot(range_obj, "lo", None)
    hi = _slot(range_obj, "hi", None)
    step = _slot(range_obj, "step", None)
    n_points = _slot(range_obj, "n", None)
    log_scale = _slot(range_obj, "log_scale", False)
    parts = []
    if lo is not None or hi is not None:
        parts.append(f"[{lo if lo is not None else '-∞'}, {hi if hi is not None else '∞'}]")
    if step is not None:
        parts.append(f"step={step}")
    if n_points is not None:
        parts.append(f"n={n_points}")
    if log_scale:
        parts.append("log")
    return ", ".join(parts)

def _distribution_text(distribution):
    if not distribution:
        return ""
    name = _slot(distribution, "name", "Distribution")
    domain = _range_text(_slot(distribution, "domain", None))
    axis = _slot(distribution, "axis", None)
    seed = _slot(distribution, "seed", None)
    parts = [str(name)]
    if domain:
        parts.append(domain)
    if axis:
        parts.append(f"axis={axis}")
    if seed is not None:
        parts.append(f"seed={seed}")
    return " ".join(parts)

def _metadata_text(obj):
    bits = []
    domain = _slot(obj, "domain", None)
    distribution = _slot(obj, "distribution", None)
    if domain:
        bits.append(_range_text(domain))
    if distribution:
        bits.append(_distribution_text(distribution))
    return "; ".join([b for b in bits if b]) or "—"

def _flag_text(obj):
    flags = []
    if _slot(obj, "free", False):
        flags.append("free")
    if _slot(obj, "heterogeneous", False):
        flags.append("heterogeneous")
    shape = _slot(obj, "shape", None)
    if shape:
        flags.append(f"shape={shape}")
    dataset_path = _slot(obj, "dataset_path", None)
    if dataset_path:
        flags.append(f"data={dataset_path}")
    optimum = _slot(obj, "reported_optimum", None)
    if optimum is not None:
        flags.append(f"optimum={optimum}")
    return ", ".join(flags) or "—"

pre_rhs = getattr(getattr(cpl, "pre_expression", None), "rhs", None)
post_rhs = getattr(getattr(cpl, "post_expression", None), "rhs", None)
coupling_meta = []
for attr, label in (
    ("coupling_function", "function"),
    ("aggregation", "aggregation"),
    ("inner_coupling", "inner"),
    ("outsym", "output symbol"),
    ("observed", "observed"),
):
    value = getattr(cpl, attr, None)
    if value:
        coupling_meta.append(f"{label}: {value}")

# Full equation (indexed-symbolic form preserves sign ordering).
full_latex = ""
try:
    sym = cpl.symbolic()
    if sym is not None:
        full_latex = latex(sym, mul_symbol="dot")
except Exception:
    try:
        eq = cpl.equation
        if eq is not None:
            full_latex = latex(eq, mul_symbol="dot")
    except Exception:
        full_latex = ""
%>\
**Coupling: ${cpl_label}**

% if cpl_desc:
${cpl_desc.strip()}

% endif
% if full_latex:
$$c = ${full_latex}$$

% endif
% if coupling_meta:
${'; '.join(coupling_meta)}.

% endif
% if pre_rhs:
**Pre-synaptic:** $c_{\text{pre}} = ${_safe_latex(pre_rhs)}$

% endif
% if post_rhs:
**Post-synaptic:** $c_{\text{post}} = ${_safe_latex(post_rhs)}$

% endif
<%
meta_lines = []
if incoming:
    meta_lines.append("Incoming states: " + ", ".join(["$" + latex(Symbol(s)) + "$" for s in incoming]))
if local:
    meta_lines.append("Local states: " + ", ".join(["$" + latex(Symbol(s)) + "$" for s in local]))
if getattr(cpl, "delayed", False):
    meta_lines.append("Conduction delays enabled")
if getattr(cpl, "sparse", False):
    meta_lines.append("Sparse connectivity")
sym_val = getattr(cpl, "symmetry", None)
if sym_val:
    meta_lines.append(f"Symmetry: {sym_val}")
%>\
% if meta_lines:
${" — ".join(meta_lines)}.

% endif
% if cpl_items:
**Coupling parameters**

| Parameter | Value | Unit | Domain / Sampling | Flags | Description |
|:----------|------:|:-----|:------------------|:------|:------------|
% for pname, param in cpl_items:
<% pval = getattr(param, "value", "—"); punit = getattr(param, "unit", "") or "—"; pdesc = getattr(param, "description", "") or "" %>\
| $${latex(Symbol(pname))}$ | ${pval} | ${punit} | ${_metadata_text(param)} | ${_flag_text(param)} | ${pdesc} |
% endfor
% endif
