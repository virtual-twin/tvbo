## -*- coding: utf-8 -*-
<%doc>
Standalone ModelingToolkit.jl experiment template.

Generates a self-contained Julia script using pure MTK (no NetworkDynamics.jl):
  1. Package imports (ModelingToolkit + ODE solver)
  2. @component function from Dynamics definition
  3. System instantiation + mtkcompile
  4. ODEProblem + solve
  5. Plot

This template handles single-model ODE systems. For network simulations,
use the NetworkDynamics.jl backend.

Context: Pre-computed dict from BaseAdapter.prepare_context()
</%doc>
<%page args="experiment, model, integration, \
dynamics_dict, sv_names, n_sv, \
is_stochastic, dt, duration, solver_method, needs_stiff, \
tstops, \
**kwargs"/>
<%!
from tvbo.export.code import render_expression
%>
<%
# Check if any parameter or state variable has units
has_units = False
unit_map = {}
for p_name, p in (model.parameters or {}).items():
    u = getattr(p, 'unit', None)
    if u:
        has_units = True
        unit_map[p_name] = str(u)
for sv_name, sv in (model.state_variables or {}).items():
    u = getattr(sv, 'unit', None)
    if u:
        has_units = True
        unit_map[sv_name] = str(u)

# Prepare expression renderer
param_names = list((model.parameters or {}).keys())
ct_names = list((model.coupling_terms or {}).keys()) if model.coupling_terms else []
dv_names = list((model.derived_variables or {}).keys()) if getattr(model, 'derived_variables', None) else []
dp_names = list((model.derived_parameters or {}).keys()) if getattr(model, 'derived_parameters', None) else []
all_symbols = sv_names + param_names + ct_names + dv_names + dp_names
juliacode = lambda expr: render_expression(expr, format='mtk', parameters=all_symbols)

# Output variables: those marked coupling_variable=true
coupling_vars = [name for name, sv in model.state_variables.items()
                 if getattr(sv, 'coupling_variable', False)]

# Detect higher-order ODEs: equation_order > 1
higher_order_svs = {}
for sv_name, sv in model.state_variables.items():
    order = getattr(sv, 'equation_order', None)
    if order and int(order) > 1:
        higher_order_svs[sv_name] = int(order)
has_higher_order = len(higher_order_svs) > 0
%>

## ── Packages ────────────────────────────────────────────────────────────────
using ModelingToolkit
% if has_units:
using ModelingToolkit: t_units as t, D_units as Dt
using DynamicQuantities
% else:
using ModelingToolkit: t_nounits as t, D_nounits as Dt
% endif
% if is_stochastic:
using StochasticDiffEq
% else:
using OrdinaryDiffEqTsit5
% endif
% if needs_stiff:
using OrdinaryDiffEqSDIRK
% endif

## ── Model definition ────────────────────────────────────────────────────────
@component function ${model.name}(; name)
% if param_names:
    @parameters begin
% for p_name, p in model.parameters.items():
<%
    p_val = p.value if p.value is not None else ''
    p_desc = p.description or p.label or ''
    meta = []
    if p_desc:
        meta.append(f'description="{p_desc}"')
    meta_str = ', [' + ', '.join(meta) + ']' if meta else ''
    default_str = f' = {p_val}' if p_val != '' else ''
%>\
        ${p_name}${default_str}${meta_str}
% endfor
    end
% endif
    @variables begin
% for sv_name, sv in model.state_variables.items():
<%
    sv_init = sv.initial_value if sv.initial_value is not None else ''
    sv_desc = sv.description or sv.label or ''
    meta = []
    if sv_desc:
        meta.append(f'description="{sv_desc}"')
    meta_str = ', [' + ', '.join(meta) + ']' if meta else ''
    default_str = f'={sv_init}' if sv_init != '' else ''
%>\
        ${sv_name}(t)${default_str}${meta_str}
% endfor
% for dv_name, dv in (getattr(model, 'derived_variables', None) or {}).items():
<%
    dv_desc = getattr(dv, 'description', '') or ''
    dv_meta = []
    if dv_desc:
        dv_meta.append(f'description="{dv_desc}"')
    dv_meta_str = ', [' + ', '.join(dv_meta) + ']' if dv_meta else ''
%>\
        ${dv_name}(t)${dv_meta_str}
% endfor
    end
    eqs = [
## Derived parameters (algebraic definitions that use parameters only)
% for dp in (getattr(model, 'derived_parameters', None) or {}).values():
        ${dp.name} ~ ${juliacode(dp.equation.rhs)},
% endfor
## Derived variables (algebraic equations involving t-dependent variables)
% for dv in (getattr(model, 'derived_variables', None) or {}).values():
<%
    is_conditional = getattr(dv, 'conditional', False) and getattr(dv, 'cases', None)
%>\
% if is_conditional:
<%
    cases = list(dv.cases)
    def build_ifelse(cases):
        if len(cases) == 1:
            return juliacode(cases[0].equation.rhs)
        c = cases[0]
        cond = str(c.condition).strip()
        if cond.lower() == 'true':
            return juliacode(c.equation.rhs)
        expr = juliacode(c.equation.rhs)
        rest = build_ifelse(cases[1:])
        return f'ifelse({cond}, {expr}, {rest})'
    ifelse_expr = build_ifelse(cases)
%>\
        ${dv.name} ~ ${ifelse_expr},
% else:
        ${dv.name} ~ ${juliacode(dv.equation.rhs)},
% endif
% endfor
## State variable equations
% for sv_name, sv in model.state_variables.items():
<%
    eq_type = getattr(sv, 'equation_type', None)
    is_algebraic = str(eq_type) == 'algebraic' if eq_type else False
    order = higher_order_svs.get(sv_name, 1)
%>\
% if is_algebraic:
        ${sv_name} ~ ${juliacode(sv.equation.rhs)},
% elif order > 1:
<%
    # Build nested Dt: D(D(x)) for order 2, D(D(D(x))) for order 3, etc.
    dt_chain = sv_name
    for _ in range(order):
        dt_chain = f'Dt({dt_chain})'
%>\
        ${dt_chain} ~ ${juliacode(sv.equation.rhs)},
% else:
        Dt(${sv_name}) ~ ${juliacode(sv.equation.rhs)},
% endif
% endfor
    ]
    return System(eqs, t; name)
end

## ── Instantiate and compile ─────────────────────────────────────────────────
@named sys_model = ${model.name}()
sys = mtkcompile(sys_model)

## ── Initial conditions ──────────────────────────────────────────────────────
<%
# Collect initial conditions that need to be set explicitly
# For higher-order ODEs, we may need D(x) initial conditions too
u0_entries = []
for sv_name, sv in model.state_variables.items():
    if sv.initial_value is not None:
        # Already set via default in @variables block
        pass
    order = higher_order_svs.get(sv_name, 1)
    if order > 1:
        # Need derivative initial conditions: D(x)(0) etc.
        deriv_init = getattr(sv, 'derivative_initial_value', None)
        if deriv_init is not None:
            u0_entries.append((f'Dt(sys.{sv_name})', deriv_init))
%>\
% if u0_entries:
u0 = [${', '.join(f'{name} => {val}' for name, val in u0_entries)}]
% else:
u0 = []
% endif

## ── Problem + solve ─────────────────────────────────────────────────────────
tspan = (0.0, ${duration})
<%
    tstops_str = ""
    if tstops:
        tstops_str = ", tstops=[" + ", ".join(str(t) for t in tstops) + "]"
%>\
prob = ODEProblem(sys, u0, tspan, jac=true)
sol = solve(prob, ${solver_method}(); saveat=${dt}${tstops_str})

## ── Plot ────────────────────────────────────────────────────────────────────
using Plots
plot(sol; title="${model.name}")
