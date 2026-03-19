## -*- coding: utf-8 -*-
<%doc>
TVBO → LEMS XML Template
=========================
Generates a complete, self-contained LEMS simulation file from
a TVBO SimulationExperiment.  No NeuroML includes — everything is
generated from scratch for 100% flexibility with ANY dynamics model.

Template receives ``experiment`` (SimulationExperiment) directly
and reads all metadata from it (Pattern A).
</%doc>
<%
from tvbo.adapters.neuroml import sympy_to_lems as lems_expr
from tvbo.adapters.neuroml import unit_to_dimension as lems_dim
from tvbo.adapters.neuroml import safe_id

# ── Unpack from experiment ──────────────────────────────────────
dyn = experiment.dynamics
dyn_id = safe_id(dyn.name or "dynamics")

params = dyn.parameters or {}
svs = dyn.state_variables or {}
dvs = getattr(dyn, "derived_variables", None) or {}
events = getattr(dyn, "events", None) or {}

coupling_inputs = getattr(dyn, "coupling_inputs", None) or []

integration = experiment.integration if hasattr(experiment, 'integration') else None
network = experiment.network if hasattr(experiment, 'network') else None
n_nodes = network.number_of_nodes if network and hasattr(network, 'number_of_nodes') else 1
dt = integration.step_size if integration else 0.01
duration = integration.duration if integration else 1000.0
%>\
<Lems>

  <!-- ════════════════════════════════════════════════════════════════
       Dimensions & Units
       ════════════════════════════════════════════════════════════════ -->
  <Dimension name="voltage"      m="1"  l="2"  t="-3" i="-1"/>
  <Dimension name="time"                        t="1"/>
  <Dimension name="per_time"                    t="-1"/>
  <Dimension name="current"                              i="1"/>
  <Dimension name="conductance"  m="-1" l="-2" t="3"  i="2"/>
  <Dimension name="capacitance"  m="-1" l="-2" t="4"  i="2"/>
  <Dimension name="resistance"   m="1"  l="2"  t="-3" i="-2"/>
  <Dimension name="concentration"       l="-3"               j="1"/>
  <Dimension name="length"              l="1"/>
  <Dimension name="none"/>

  <Unit name="second"      symbol="s"    dimension="time"        power="0"/>
  <Unit name="milliSecond" symbol="ms"   dimension="time"        power="-3"/>
  <Unit name="milliVolt"   symbol="mV"   dimension="voltage"     power="-3"/>
  <Unit name="volt"        symbol="V"    dimension="voltage"     power="0"/>
  <Unit name="milliAmpere" symbol="mA"   dimension="current"     power="-3"/>
  <Unit name="nanoAmpere"  symbol="nA"   dimension="current"     power="-9"/>
  <Unit name="picoAmpere"  symbol="pA"   dimension="current"     power="-12"/>
  <Unit name="siemens"     symbol="S"    dimension="conductance" power="0"/>
  <Unit name="milliSiemens" symbol="mS"  dimension="conductance" power="-3"/>
  <Unit name="nanoSiemens" symbol="nS"   dimension="conductance" power="-9"/>
  <Unit name="microFarad"  symbol="uF"   dimension="capacitance" power="-6"/>
  <Unit name="nanoFarad"   symbol="nF"   dimension="capacitance" power="-9"/>
  <Unit name="picoFarad"   symbol="pF"   dimension="capacitance" power="-12"/>
  <Unit name="ohm"         symbol="ohm"  dimension="resistance"  power="0"/>
  <Unit name="per_second"  symbol="per_s" dimension="per_time"   power="0"/>
  <Unit name="hertz"       symbol="Hz"   dimension="per_time"    power="0"/>
  <Unit name="metre"       symbol="m"    dimension="length"      power="0"/>
  <Unit name="centimetre"  symbol="cm"   dimension="length"      power="-2"/>
  <Unit name="micrometre"  symbol="um"   dimension="length"      power="-6"/>

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
% if ci_name not in [str(k) for k in params.keys()]:
    <Parameter name="${ci_name}" dimension="none"/>
% endif
% endfor
    <!-- Initial condition parameters -->
% for sv_name in svs:
    <Parameter name="${sv_name}_0" dimension="${lems_dim(getattr(svs[sv_name], 'unit', None))}"/>
% endfor

    <!-- Time constant for derivatives -->
    <Constant name="SEC" dimension="time" value="1s"/>

    <!-- Exposures (one per state variable) -->
% for sv_name, sv in svs.items():
    <Exposure name="${sv_name}" dimension="${lems_dim(getattr(sv, 'unit', None))}"/>
% endfor

    <Dynamics>

      <!-- State variables -->
% for sv_name, sv in svs.items():
      <StateVariable name="${sv_name}" dimension="${lems_dim(getattr(sv, 'unit', None))}" exposure="${sv_name}"/>
% endfor

      <!-- Derived variables -->
% for dv_name, dv in dvs.items():
<%
  eq = getattr(dv, 'equation', None)
  rhs = getattr(eq, 'rhs', None) if eq else None
%>\
% if rhs:
      <DerivedVariable name="${dv_name}" dimension="${lems_dim(getattr(dv, 'unit', None))}" value="${lems_expr(rhs)}"/>
% endif
% endfor

      <!-- Time derivatives -->
% for sv_name, sv in svs.items():
<%
  eq = getattr(sv, 'equation', None)
  rhs = getattr(eq, 'rhs', None) if eq else None
%>\
% if rhs:
      <TimeDerivative variable="${sv_name}" value="(${lems_expr(rhs)}) / SEC"/>
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
<%
  # Convert condition to LEMS test format
  cond_str = str(cond_rhs).strip()
  lems_test = cond_str
  for op, lop in [(">=", ".geq."), ("<=", ".leq."), (">", ".gt."), ("<", ".lt."), ("==", ".eq."), ("!=", ".neq.")]:
      if op in lems_test:
          lems_test = lems_test.replace(op, " " + lop + " ")
          break
  else:
      if " - " in lems_test:
          parts = lems_test.split(" - ", 1)
          lems_test = parts[0].strip() + " .gt. " + parts[1].strip()
      else:
          lems_test = lems_test + " .gt. 0"
%>\
      <OnCondition test="${lems_test}">
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
       Component instance (default parameter values)
       ════════════════════════════════════════════════════════════════ -->
  <Component id="${dyn_id}" type="${dyn_id}"\
% for pname, p in params.items():
 ${pname}="${getattr(p, 'value', 0)}"\
% endfor
% for ci in coupling_inputs:
<% ci_name = str(ci) %>\
% if ci_name not in [str(k) for k in params.keys()]:
 ${ci_name}="0"\
% endif
% endfor
% for sv_name, sv in svs.items():
<% iv = getattr(sv, 'initial_value', None) %>\
 ${sv_name}_0="${iv if iv is not None else 0.0}"\
% endfor
/>

  <!-- ════════════════════════════════════════════════════════════════
       Network
       ════════════════════════════════════════════════════════════════ -->
  <Component id="net" type="network">
    <Component id="pop0" type="population" component="${dyn_id}" size="${n_nodes}"/>
  </Component>

  <!-- ════════════════════════════════════════════════════════════════
       Simulation
       ════════════════════════════════════════════════════════════════ -->
  <Simulation id="sim1" length="${duration}ms" step="${dt}ms" target="net">

% for sv_name in svs:
    <OutputFile id="of_${sv_name}" fileName="results/${dyn_id}_${sv_name}.dat">
% for node_idx in range(min(n_nodes, 100)):
      <OutputColumn id="${sv_name}_${node_idx}" quantity="pop0[${node_idx}]/${sv_name}"/>
% endfor
    </OutputFile>
% endfor

  </Simulation>

</Lems>
