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
<%page args="model, all_couplings=None"/>
<%!
from tvbo.export.code import render_expression
%>
<%
sv_names = list(model.state_variables.keys())
param_names = list(model.parameters.keys())
ct_names = list(model.coupling_terms.keys()) if model.coupling_terms else []
dv_names = list(model.derived_variables.keys()) if model.derived_variables else []
dp_names = list(model.derived_parameters.keys()) if model.derived_parameters else []
n_sv = len(sv_names)

# Determine output dimension: coupling variables → vertex output
# If coupling_variable is marked, only those are output; otherwise all state vars
coupling_vars = [name for name, sv in model.state_variables.items()
                 if getattr(sv, 'coupling_variable', False)]
n_out = len(coupling_vars) if coupling_vars else n_sv

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
juliacode = lambda expr: render_expression(expr, format='julia', parameters=all_symbols)
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
    % for i, ct in enumerate(ct_names):
    % if len(ct_names) == 1 and n_out == 1:
    ${ct} = esum[1]
    % else:
    ${ct} = esum[${i + 1}]
    % endif
    % endfor
    % for dp in (model.derived_parameters or {}).values():
    ${dp.name} = ${juliacode(dp.equation.rhs)}
    % endfor
    % for dv in (model.derived_variables or {}).values():
    % if getattr(dv, 'conditional', False) and getattr(dv, 'cases', None):
<%
    cases = list(dv.cases)
    parts = []
    for case in cases:
        cond_str = str(case.condition).strip()
        eq_rhs = juliacode(case.equation.rhs)
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
