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
  - parameters: emit the parameter table (False where the host report already
    lists these symbols, e.g. a study's own glossary)
  - equations: a report.Equations to number and anchor the coupling equation with,
    or None to emit it bare (standalone rendering, which has nothing to number into)
</%doc>
<%
from sympy import latex, Symbol
from tvbo.parse.expression import parse_eq
from tvbo.utils import report

cpl = coupling
cpl_label = getattr(cpl, "label", None) or getattr(cpl, "name", "Coupling")
cpl_desc = getattr(cpl, "description", "") or ""

def _md_escape(s):
    """Escape markdown emphasis characters so a free-text label renders
    literally inside the surrounding ``**...**`` — an odd number of
    underscores or asterisks in the label would otherwise break the bold
    span (e.g. ``sum_j (A+I)_ij (theta_j(t-tau_ij) - theta_i)``)."""
    s = str(s)
    for ch in ("\\", "`", "*", "_"):
        s = s.replace(ch, "\\" + ch)
    return s

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

# Full equation (indexed-symbolic form preserves sign ordering). A delayed coupling
# must show its delay -- theta_j(t - tau_ij) -- otherwise it reads identically to its
# instantaneous twin, so pass the coupling's own `delayed` flag through.
full_latex = ""
try:
    sym = cpl.symbolic(delays=bool(getattr(cpl, "delayed", False)))
    if sym is not None:
        # order="none" keeps the constructed term order (e.g. theta_j - theta_i) instead
        # of sympy's canonical reordering to -theta_i + theta_j.
        full_latex = latex(sym, mul_symbol="dot", order="none")
except Exception:
    try:
        eq = cpl.equation
        if eq is not None:
            full_latex = latex(eq, mul_symbol="dot")
    except Exception:
        full_latex = ""

# Factored / vectorized couplings: define gx_k = Sum_j w_ij (c_pre)_k so the post's
# gx_0, gx_1, ... are mathematically precise, not bare symbols.
try:
    gx_defs = cpl.summed_inputs(delays=bool(getattr(cpl, "delayed", False)))
except Exception:
    gx_defs = []
gx_line = ", ".join(
    f"${latex(g)} = {latex(s, mul_symbol='dot', order='none')}$" for g, s in gx_defs
)


def _summed_inputs():
    """The ``gx_k`` intermediates, numbered where the post-synaptic term can cite them.

    ``c_post`` is written in terms of ``gx``, so once that is a numbered equation its
    operand has to be one too — otherwise the section defines a symbol it uses in a
    numbered equation only in a passing clause.
    """
    aside = "the pre-synaptic components summed over the graph, which the post-synaptic term recombines"
    if not equations:
        return f"with {gx_line} — {aside}."
    blocks = "\n".join(
        equations.block(f"{latex(g)} = {latex(s, mul_symbol='dot', order='none')}", cpl, f"gx-{i}")
        for i, (g, s) in enumerate(gx_defs))
    return f"with {aside}:\n\n{blocks}"


def _decomposition(label, symbol, rhs, key):
    """One half of the pre/post split, displayed and numbered where it can be cited.

    An equation a reader cannot refer to does not need to be displayed, so without a
    numbering registry these stay inline and compact (the standalone rendering, embedded
    in docs). With one they become numbered display equations like every other equation
    in the section — these define what the backend actually computes per source node and
    at the target, and a Methods section that numbers the state equations but not the
    coupling decomposition leaves its most-cited step uncitable.
    """
    body = f"{symbol} = {_safe_latex(rhs)}"
    if equations:
        return f"**{label}:**\n\n{equations.block(body, cpl, key)}"
    return f"**{label}:** ${body}$"
%>\
**Coupling: ${_md_escape(cpl_label)}**

% if cpl_desc:
${cpl_desc.strip()}

% endif
% if full_latex:
${equations.block("c = " + full_latex, cpl, "c") if equations else "$$c = " + full_latex + "$$"}

% endif
% if coupling_meta:
${'; '.join(coupling_meta)}.

% endif
% if pre_rhs:
${_decomposition("Pre-synaptic", r"c_{\text{pre}}", pre_rhs, "c-pre")}

% endif
% if post_rhs:
${_decomposition("Post-synaptic", r"c_{\text{post}}", post_rhs, "c-post")}

% endif
% if gx_line:
${_summed_inputs()}

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
% if cpl_items and parameters:
**Coupling parameters**

${report.parameter_table(cpl_params_obj)}

% endif
