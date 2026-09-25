## -*- coding: utf-8 -*-
<%doc>
NetworkDynamics.jl VertexModel from tvbo Dynamics.

Generates:
  - f!(dx, x, esum, p, t): node-local dynamics
  - VertexModel constructor with symbolic state/param names

Supports multi-dimensional coupling: when multiple state variables are
marked coupling_variable=true, the vertex outputs all of them via g=1:n_out
and esum has dimension n_out.

Context: model (Dynamics instance)
</%doc>
<%page args="model, all_couplings=None, outdim=None"/>
<%!
from tvbo.codegen import render_expression
from tvbo.templates.base.utils import get_coupling_terms
%>
<%
sv_names = list(model.state_variables.keys())
param_names = list(model.parameters.keys())
ct_names = list(model.coupling_inputs.keys()) if model.coupling_inputs else []
dv_names = list(model.in_dependency_order('derived_variables').keys()) if model.derived_variables else []
dp_names = list(model.in_dependency_order('derived_parameters').keys()) if model.derived_parameters else []
n_sv = len(sv_names)

# Determine output dimension: coupling variables → vertex output
# If coupling_variable is marked, only those are output; otherwise all state vars
coupling_vars = [name for name, sv in model.state_variables.items()
                 if getattr(sv, 'coupling_variable', False)]
n_out = len(coupling_vars) if coupling_vars else n_sv

# Edge output dimension: how many values esum actually contains.
# This must match the edge model's outsym length.
# Global coupling terms map to esum; a local input is not driven by the connectome.
# If outdim not provided, compute from n_out.
global_ct_names = get_coupling_terms(model)[1]
if outdim is None:
    outdim = n_out
# Only the first `outdim` global coupling terms read from esum;
# the rest (and local_coupling) are zero.
esum_ct_names = global_ct_names[:outdim]

# Compute StateMask range: indices (1-based) of coupling variables in state vector
if coupling_vars:
    cvar_indices = [i + 1 for i, name in enumerate(sv_names) if name in coupling_vars]
    # Check if indices are contiguous for StateMask(start:end)
    g_start = cvar_indices[0]
    g_end = cvar_indices[-1]
else:
    g_start = 1
    g_end = n_sv

# Compute insym from coupling outsym (if multi-dimensional coupling)
insym = None
if all_couplings and len(ct_names) > 1:
    # Get the default coupling's outsym
    default_coupling = next(iter(all_couplings.values())) if all_couplings else None
    if default_coupling and getattr(default_coupling, 'outsym', None):
        insym = list(default_coupling.outsym)

# All symbol names the parser must recognize (prevents omega0 → omega*0 etc.)
all_symbols = sv_names + param_names + ct_names + dv_names + dp_names
func_names = {str(fname): str(fname) for fname in (getattr(model, 'functions', None) or {}).keys()}
juliacode = lambda expr: render_expression(expr, format='julia', parameters=all_symbols, user_functions=func_names)
%>

## ── Node dynamics (f!) ──────────────────────────────────────────────────────
<%
# Detect if any state var name shadows the function argument 'x'
# In that case, rename the argument to '_x' to avoid collision
arg_x = '_x' if 'x' in sv_names else 'x'
%>
% if param_names:
function ${model.name}_f!(dx, ${arg_x}, esum, (${", ".join(param_names)},), t)
% else:
function ${model.name}_f!(dx, ${arg_x}, esum, p, t)
% endif
% if n_sv > 1:

    ${", ".join(sv_names)} = ${arg_x}
% elif n_sv == 1:
    ${sv_names[0]} = ${arg_x}[1]
% endif

<%
    # Check if multi-dim esum can use broadcasting: single coupling term,
    # n_out > 1, and every SV equation is just the coupling term name
    use_broadcast = (
        len(ct_names) == 1 and n_out > 1
        and all(str(sv.equation.rhs).strip() == ct_names[0]
                for sv in model.state_variables.values())
    )
%>\
    % if use_broadcast:
    ## Multi-dim coupling: all SVs = coupling term → broadcast esum directly
    dx .= esum
    % else:
    ## Coupling terms: map edge outputs to named coupling variables
    % for ct in ct_names:
    % if ct in esum_ct_names:
    % if len(esum_ct_names) == 1:
    ${ct} = esum[1]
    % else:
    ${ct} = esum[${esum_ct_names.index(ct) + 1}]
    % endif
    % else:
    ${ct} = 0.0
    % endif
    % endfor
    % for fname, fdef in (model.functions or {}).items():
<%
    _fargs = fdef.arguments or {}
    fargs = [str(getattr(arg, "name", arg)) for arg in (_fargs.values() if hasattr(_fargs, "values") else _fargs)]
    fbody = juliacode(fdef.equation.rhs)
%>
    ${fname}(${", ".join(fargs)}) = ${fbody}
    % endfor
    % for dp in model.in_dependency_order('derived_parameters').values():
    ${dp.name} = ${juliacode(dp.equation.rhs)}
    % endfor
    % for dv in model.in_dependency_order('derived_variables').values():
    % if getattr(dv.equation, 'conditionals', None):
<%
    parts = []
    for case in dv.equation.conditionals:
        cond_str = str(case.condition).strip()
        eq_rhs = juliacode(case.expression)
        if cond_str.lower() == 'true':
            parts.append(eq_rhs)
        else:
            parts.append((cond_str, eq_rhs))
    # Build nested ifelse chain
    def build_ifelse(parts):
        if len(parts) == 1:
            return parts[0] if isinstance(parts[0], str) else parts[0][1]
        cond, val = parts[0]
        return f"ifelse({cond}, {val}, {build_ifelse(parts[1:])})"
    ifelse_expr = build_ifelse(parts)
%>
    ${dv.name} = ${ifelse_expr}
    % else:
    ${dv.name} = ${juliacode(dv.equation.rhs)}
    % endif
    % endfor
    % for i, sv in enumerate(model.state_variables.values()):
    dx[${i + 1}] = ${juliacode(sv.equation.rhs)}
    % endfor
    % endif
    nothing
end

## ── VertexModel ─────────────────────────────────────────────────────────────
vertex_${model.name} = VertexModel(;
    f = ${model.name}_f!,
    g = StateMask(${g_start}:${g_end}),
    sym = [${", ".join(f':{sv}' for sv in sv_names)}],
% if param_names:
    psym = [${", ".join(f':{p} => {model.parameters[p].value}' for p in param_names)}],
% endif
% if insym:
    insym = [${", ".join(f':{s}' for s in insym)}],
% endif
    name = :${model.name},
)
