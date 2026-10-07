## -*- coding: utf-8 -*-
<%doc>
NetworkDynamics.jl VertexModel from tvbo Dynamics.

Generates:
  - f!(dx, x, esum, p, t): node-local dynamics
  - VertexModel constructor with symbolic state/param names

The vertex outputs the states the coupling transmits, the StateMask of layout['mask'], and receives esum, the sum of its edges' outputs, with outdim components. Each global coupling input is the coupling's post-expression applied once to its component of esum, with the post-expression's parameters carried by the vertex under their coupling_split symbols; the global inputs past outdim, and a local input, which the connectome does not drive, are zero. A vertex that holds its input per step (coupling_evaluation: per_step) reads esum off its layout['held'] parameters, which a callback sets at the start of every step.

Context: model (Dynamics instance), outdim (components of esum), split (coupling_split of the coupling on the edges, or None), layout (NetworkDynamicsAdapter.vertex_layout)
</%doc>
<%page args="model, outdim, layout, split=None"/>
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
n_out = len(layout['outputs'])
esum_ct_names = get_coupling_terms(model)[1][:outdim]
insym = layout['insym']
held = layout['held']

# All symbol names the parser must recognize (prevents omega0 → omega*0 etc.)
all_symbols = sv_names + param_names + ct_names + dv_names + dp_names
func_names = {str(fname): str(fname) for fname in (getattr(model, 'functions', None) or {}).keys()}
juliacode = lambda expr: render_expression(expr, format='julia', parameters=all_symbols, user_functions=func_names)

# Multi-dim esum is broadcast when there is a single coupling term, n_out > 1, and every SV equation is just that term
use_broadcast = (
    len(ct_names) == 1 and n_out > 1
    and all(str(sv.equation.rhs).strip() == ct_names[0]
            for sv in model.state_variables.values())
)
# The post-expression and its parameters reach only a vertex that reads esum
post = split if split is not None and split['post'] is not None and (use_broadcast or esum_ct_names) else None
post_parameters = post['post_parameters'] if post else []
post_syms = [sym for _, sym, _ in post_parameters]
f_params = param_names + post_syms + held
%>

## ── Node dynamics (f!) ──────────────────────────────────────────────────────
<%
# Detect if any state var name shadows the function argument 'x'
# In that case, rename the argument to '_x' to avoid collision
arg_x = '_x' if 'x' in sv_names else 'x'
%>
% if f_params:
function ${model.name}_f!(dx, ${arg_x}, esum, (${", ".join(f_params)},), t)
% else:
function ${model.name}_f!(dx, ${arg_x}, esum, p, t)
% endif
% if n_sv > 1:

    ${", ".join(sv_names)} = ${arg_x}
% elif n_sv == 1:
    ${sv_names[0]} = ${arg_x}[1]
% endif
% if held:
    esum = (${", ".join(held)},)
% endif

    % if use_broadcast:
    ## Multi-dim coupling: all SVs = coupling term → broadcast the (post-applied) esum directly
    dx .= ${summed('esum', post, post_syms, dotted=True)}
    % else:
    ## Coupling terms: map edge outputs to named coupling variables
    % for ct in ct_names:
    % if ct in esum_ct_names:
    % if len(esum_ct_names) == 1:
    ${ct} = ${summed('esum[1]', post, post_syms)}
    % else:
    ${ct} = ${summed(f'esum[{esum_ct_names.index(ct) + 1}]', post, post_syms)}
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
    g = StateMask(${layout['mask']}),
    sym = [${", ".join(f':{sv}' for sv in sv_names)}],
% if f_params:
    psym = [${", ".join([f':{p} => {model.parameters[p].value}' for p in param_names] + [f':{sym} => {value}' for _, sym, value in post_parameters] + [f':{h} => 0.0' for h in held])}],
% endif
% if insym:
    insym = [${", ".join(f':{s}' for s in insym)}],
% endif
    name = :${model.name},
)
<%def name="summed(x, post, post_syms, dotted=False)">\
## The coupling input from summed edge input x: the post-expression's function applied once, broadcast over esum when dotted, or x itself where the post-expression is the identity.
<% dot = '.' if dotted else '' %>\
% if post is None:
${x}\
% elif post_syms:
${post['post_function']}${dot}(${x}, ${'Ref(' if dotted else ''}(${", ".join(post_syms)},)${')' if dotted else ''})\
% else:
${post['post_function']}${dot}(${x})\
% endif
</%def>
