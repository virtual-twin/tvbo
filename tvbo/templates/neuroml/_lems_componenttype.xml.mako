<%!
from tvbo.utils import initial_value as _initial_value
%>\
## -*- coding: utf-8 -*-
<%doc>
Shared fragment: LEMS ComponentType + Coupling + Component instances
=====================================================================
Single source of truth for the dynamics ComponentType, Coupling ComponentType,
and their default Component instances.  Included by both the monolithic
template (tvbo-neuroml-lems.xml.mako) and the standalone dynamics template
(tvbo-neuroml-dynamics.xml.mako).

All template variables are injected by the calling template's render context
(from build_lems_context()).
</%doc>
## lems_dim() (from build_lems_context) already suppresses non-time dimensions
## to "none" for custom types.  No extra wrapper needed.
<%
  _dim = lems_dim  # alias for backward compat in template expressions
%>\
  <!-- ════════════════════════════════════════════════════════════════
       ComponentType: ${dyn_id}
       Generated from TVBO Dynamics: ${dyn.name or '(unnamed)'}
       ════════════════════════════════════════════════════════════════ -->
  <ComponentType name="${dyn_id}">

    <!-- Parameters -->
% for pname, p in params.items():
    <Parameter name="${pname}" dimension="${_dim(getattr(p, 'unit', None))}"/>
% endfor
% if regime_data and 'refract' not in params:
    <Parameter name="refract" dimension="time"/>
% endif
    <!-- Coupling inputs -->
% for ci in coupling_inputs:
<%  ci_name = str(ci) %>\
% if ci_name not in sv_names_set and ci_name not in [str(k) for k in params.keys()]:
    <Parameter name="${ci_name}" dimension="none"/>
% endif
% endfor
    <!-- Initial condition parameters -->
% for sv_name in svs:
    <Parameter name="${sv_name}_0" dimension="${_dim(getattr(svs[sv_name], 'unit', None))}"/>
% endfor

    <!-- Time conversion for derivatives.
         When all parameters and state variables carry proper LEMS dimensions,
         LEMS handles unit conversion natively (e.g. tau="30 ms" → 0.03 s).
         No SEC constant is needed and TimeDerivatives use the RHS directly.
         When dimensions are "none" (dimensionless models), / SEC converts
         from model time to SI seconds.
         all_dimensioned=${all_dimensioned}  needs_sec=${needs_sec}  time_scale=${time_scale} -->
% if not all_dimensioned:
% if needs_sec:
    <Constant name="SEC" dimension="time" value="1${time_scale}"/>
% else:
    <Constant name="SEC" dimension="time" value="1s"/>
% endif
% endif

    <!-- Exposures (one per state variable) -->
% for sv_name, sv in svs.items():
    <Exposure name="${sv_name}" dimension="${_dim(getattr(sv, 'unit', None))}"/>
% endfor
% if regime_data:
    <EventPort name="spike" direction="out"/>
% endif

    <Dynamics>

      <!-- State variables -->
% for sv_name, sv in svs.items():
      <StateVariable name="${sv_name}" dimension="${_dim(getattr(sv, 'unit', None))}" exposure="${sv_name}"/>
% endfor
% if regime_data:
      <StateVariable name="lastSpikeTime" dimension="time"/>
% endif

      <!-- Derived variables (simple and conditional/piecewise) -->
% for dv_name, dv in dvs.items():
<%
  eq = getattr(dv, 'equation', None)
  rhs = eq  # the Equation itself; lems_expr resolves rhs or conditionals
  dv_dim = _dim(getattr(dv, 'unit', None))
  pw_cases = _parse_piecewise(rhs) if rhs else None
%>\
% if rhs:
% if pw_cases:
% if len(pw_cases) == 1 and pw_cases[0][0] is None:
      <DerivedVariable name="${dv_name}" dimension="${dv_dim}" value="${pw_cases[0][1]}"/>
% else:
      <ConditionalDerivedVariable name="${dv_name}" dimension="${dv_dim}">
% for (cond_str, val_str) in pw_cases:
% if cond_str is not None:
        <Case condition="${cond_str}" value="${val_str}"/>
% else:
        <Case value="${val_str}"/>
% endif
% endfor
      </ConditionalDerivedVariable>
% endif
% else:
      <DerivedVariable name="${dv_name}" dimension="${dv_dim}" value="${lems_expr(rhs)}"/>
% endif
% endif
% endfor

% if regime_data:
      <!-- ── Regime-based dynamics (spike model) ── -->

      <!-- Initial conditions -->
      <OnStart>
% for sv_name, sv in svs.items():
        <StateAssignment variable="${sv_name}" value="${sv_name}_0"/>
% endfor
      </OnStart>

      <Regime name="integrating" initial="true">
        <!-- All time derivatives active -->
% for sv_name, sv in svs.items():
<%
  eq = getattr(sv, 'equation', None)
  rhs = eq  # the Equation itself; lems_expr resolves rhs or conditionals
%>\
% if rhs:
% if all_dimensioned:
        <TimeDerivative variable="${sv_name}" value="${lems_expr(rhs)}"/>
% else:
        <TimeDerivative variable="${sv_name}" value="(${lems_expr(rhs)}) / SEC"/>
% endif
% endif
% endfor
        <OnCondition test="${lems_expr(regime_data['condition'])}">
          <EventOut port="spike"/>
          <Transition regime="refractory"/>
        </OnCondition>
      </Regime>

      <Regime name="refractory">
        <!-- Only non-reset SVs evolve during refractory -->
% for sv_name, sv in svs.items():
<%
  eq = getattr(sv, 'equation', None)
  rhs = eq  # the Equation itself; lems_expr resolves rhs or conditionals
%>\
% if rhs and sv_name not in regime_data['reset_vars']:
% if all_dimensioned:
        <TimeDerivative variable="${sv_name}" value="${lems_expr(rhs)}"/>
% else:
        <TimeDerivative variable="${sv_name}" value="(${lems_expr(rhs)}) / SEC"/>
% endif
% endif
% endfor
        <OnEntry>
          <StateAssignment variable="lastSpikeTime" value="t"/>
% for lhs, rhs_val in regime_data['assignments']:
          <StateAssignment variable="${lhs}" value="${lems_expr(rhs_val)}"/>
% endfor
        </OnEntry>
        <OnCondition test="t .gt. lastSpikeTime + refract">
          <Transition regime="integrating"/>
        </OnCondition>
      </Regime>

% else:
      <!-- ── Flat dynamics (no spike events) ── -->

      <!-- Time derivatives -->
% for sv_name, sv in svs.items():
<%
  eq = getattr(sv, 'equation', None)
  rhs = eq  # the Equation itself; lems_expr resolves rhs or conditionals
%>\
% if rhs:
% if all_dimensioned:
      <TimeDerivative variable="${sv_name}" value="${lems_expr(rhs)}"/>
% else:
      <TimeDerivative variable="${sv_name}" value="(${lems_expr(rhs)}) / SEC"/>
% endif
% endif
% endfor

      <!-- Initial conditions -->
      <OnStart>
% for sv_name, sv in svs.items():
        <StateAssignment variable="${sv_name}" value="${sv_name}_0"/>
% endfor
      </OnStart>

      <!-- Events (non-spike) -->
% for ev_name, ev in events.items():
<%
  cond = getattr(ev, 'condition', None)
  affect = getattr(ev, 'affect', None)
  cond_rhs = getattr(cond, 'rhs', None) if cond else None
  affect_rhs = getattr(affect, 'rhs', None) if affect else None
%>\
% if cond_rhs:
      <OnCondition test="${lems_expr(cond_rhs)}">
% if affect_rhs:
% for assignment in str(affect_rhs).split(";"):
<% parts = assignment.strip().split("=", 1) if "=" in assignment else [] %>\
% if len(parts) == 2:
        <StateAssignment variable="${parts[0].strip()}" value="${lems_expr(parts[1].strip())}"/>
% endif
% endfor
% endif
      </OnCondition>
% endif
% endfor
% endif

    </Dynamics>

  </ComponentType>

% if n_nodes > 1:
  <!-- ════════════════════════════════════════════════════════════════
       Coupling ComponentType
       Provides pre/post expressions + global coupling parameter.
       The DerivedVariable ${coupling_output_name} is the coupling input consumed
       by the dynamics ComponentType's coupling_inputs.
       ════════════════════════════════════════════════════════════════ -->
  <ComponentType name="Coupling">
    <Parameter name="global_coupling" dimension="none"/>
% for pname, p in coupling_params.items():
<% pname_str = str(pname) %>\
% if pname_str != "global_coupling":
    <Parameter name="${pname_str}" dimension="${_dim(getattr(p, 'unit', None))}"/>
% endif
% endfor
    <Dynamics>
      <DerivedVariable name="pre"  dimension="none" value="${lems_expr(coupling_pre_rhs)}"/>
      <DerivedVariable name="gx"   dimension="none" value="global_coupling * pre"/>
      <DerivedVariable name="post" dimension="none" value="${lems_expr(coupling_post_rhs) if coupling_post_rhs != 'global_coupling * pre' else 'gx'}"/>
      <DerivedVariable name="${coupling_output_name}" dimension="none" value="post"/>
    </Dynamics>
  </ComponentType>
% endif

  <!-- ════════════════════════════════════════════════════════════════
       Component instances (default parameter values)
       ════════════════════════════════════════════════════════════════ -->
  <Component id="${dyn_id}_inst" type="${dyn_id}"\
% for pname, p in params.items():
<% p_unit = lems_sym(getattr(p, 'unit', None)) %>\
 ${pname}="${getattr(p, 'value', 0)}${(' ' + p_unit) if p_unit else ''}"\
% endfor
% if regime_data and 'refract' not in params:
 refract="0 ${time_scale}"\
% endif
% for ci in coupling_inputs:
<% ci_name = str(ci) %>\
% if ci_name not in sv_names_set and ci_name not in [str(k) for k in params.keys()]:
 ${ci_name}="0"\
% endif
% endfor
% for sv_name, sv in svs.items():
<% iv = _initial_value(sv) %>\
<% sv_unit = lems_sym(getattr(sv, 'unit', None)) %>\
 ${sv_name}_0="${iv if iv is not None else 0.0}${(' ' + sv_unit) if sv_unit else ''}"\
% endfor
/>

% if n_nodes > 1:
  <Component id="coupling" type="Coupling" global_coupling="${coupling_global}"\
% for pname, p in coupling_params.items():
<% pname_str = str(pname) %>\
% if pname_str != "global_coupling":
 ${pname_str}="${getattr(p, 'value', 0)}"\
% endif
% endfor
/>
% endif
