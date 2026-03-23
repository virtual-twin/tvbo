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
  <!-- ════════════════════════════════════════════════════════════════
       ComponentType: ${dyn_id}
       Generated from TVBO Dynamics: ${dyn.name or '(unnamed)'}
       ════════════════════════════════════════════════════════════════ -->
  <ComponentType name="${dyn_id}">

    <!-- Parameters -->
% for pname, p in params.items():
    <Parameter name="${pname}" dimension="${lems_dim(getattr(p, 'unit', None))}"/>
% endfor
    <!-- Coupling inputs -->
% for ci in coupling_inputs:
<%  ci_name = str(ci) %>\
% if ci_name not in sv_names_set and ci_name not in [str(k) for k in params.keys()]:
    <Parameter name="${ci_name}" dimension="none"/>
% endif
% endfor
    <!-- Initial condition parameters -->
% for sv_name in svs:
    <Parameter name="${sv_name}_0" dimension="${lems_dim(getattr(svs[sv_name], 'unit', None))}"/>
% endfor

    <!-- Time constant for derivatives.
         When the model is fully dimensionless (no time-bearing parameter units),
         dividing by SEC converts the expression from 'none' to 'per_time' as
         LEMS requires.  When parameters already carry physical time dimensions
         (e.g., tau_e in ms, or rate constant a in ms⁻¹), the equation naturally
         has the correct dimension and / SEC is omitted.
         needs_sec=${needs_sec}  time_scale=${time_scale} -->
% if needs_sec:
    <Constant name="SEC" dimension="time" value="1${time_scale}"/>
% endif

    <!-- Exposures (one per state variable) -->
% for sv_name, sv in svs.items():
    <Exposure name="${sv_name}" dimension="${lems_dim(getattr(sv, 'unit', None))}"/>
% endfor

    <Dynamics>

      <!-- State variables -->
% for sv_name, sv in svs.items():
      <StateVariable name="${sv_name}" dimension="${lems_dim(getattr(sv, 'unit', None))}" exposure="${sv_name}"/>
% endfor

      <!-- Derived variables (simple and conditional/piecewise) -->
% for dv_name, dv in dvs.items():
<%
  eq = getattr(dv, 'equation', None)
  rhs = getattr(eq, 'rhs', None) if eq else None
  dv_dim = lems_dim(getattr(dv, 'unit', None))
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

      <!-- Time derivatives -->
% for sv_name, sv in svs.items():
<%
  eq = getattr(sv, 'equation', None)
  rhs = getattr(eq, 'rhs', None) if eq else None
%>\
% if rhs:
% if needs_sec:
      <TimeDerivative variable="${sv_name}" value="(${lems_expr(rhs)}) / SEC"/>
% else:
      <TimeDerivative variable="${sv_name}" value="${lems_expr(rhs)}"/>
% endif
% endif
% endfor

      <!-- Initial conditions -->
      <OnStart>
% for sv_name, sv in svs.items():
        <StateAssignment variable="${sv_name}" value="${sv_name}_0"/>
% endfor
      </OnStart>

      <!-- Events (spike / reset) -->
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

    </Dynamics>

  </ComponentType>

  <!-- ════════════════════════════════════════════════════════════════
       Coupling ComponentType
       Provides pre/post expressions + global coupling parameter.
       The DerivedParameter c_pop0 is the coupling input consumed by
       the dynamics ComponentType's coupling_inputs.
       ════════════════════════════════════════════════════════════════ -->
  <ComponentType name="Coupling">
    <Parameter name="global_coupling" dimension="none"/>
% for pname, p in coupling_params.items():
<% pname_str = str(pname) %>\
% if pname_str != "global_coupling":
    <Parameter name="${pname_str}" dimension="${lems_dim(getattr(p, 'unit', None))}"/>
% endif
% endfor
    <Dynamics>
      <DerivedVariable name="pre"  dimension="none" value="${lems_expr(coupling_pre_rhs)}"/>
      <DerivedVariable name="gx"   dimension="none" value="global_coupling * pre"/>
      <DerivedVariable name="post" dimension="none" value="${lems_expr(coupling_post_rhs) if coupling_post_rhs != 'global_coupling * pre' else 'gx'}"/>
      <DerivedVariable name="c_pop0" dimension="none" value="post"/>
    </Dynamics>
  </ComponentType>

  <!-- ════════════════════════════════════════════════════════════════
       Component instances (default parameter values)
       ════════════════════════════════════════════════════════════════ -->
  <Component id="${dyn_id}_inst" type="${dyn_id}"\
% for pname, p in params.items():
<% p_unit = getattr(p, 'unit', '') or '' %>\
 ${pname}="${getattr(p, 'value', 0)}${p_unit}"\
% endfor
% for ci in coupling_inputs:
<% ci_name = str(ci) %>\
% if ci_name not in sv_names_set and ci_name not in [str(k) for k in params.keys()]:
 ${ci_name}="0"\
% endif
% endfor
% for sv_name, sv in svs.items():
<% iv = getattr(sv, 'initial_value', None) %>\
 ${sv_name}_0="${iv if iv is not None else 0.0}"\
% endfor
/>

  <Component id="coupling" type="Coupling" global_coupling="${coupling_global}"\
% for pname, p in coupling_params.items():
<% pname_str = str(pname) %>\
% if pname_str != "global_coupling":
 ${pname_str}="${getattr(p, 'value', 0)}"\
% endif
% endfor
/>
