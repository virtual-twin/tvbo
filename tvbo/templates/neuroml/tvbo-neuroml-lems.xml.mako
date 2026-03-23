## -*- coding: utf-8 -*-
<%doc>
TVBO → LEMS XML Template  (monolithic)
=======================================
Generates a complete, self-contained LEMS simulation file.
All dimensions, units, and LEMS infrastructure types (Simulation,
OutputFile, OutputColumn) are defined inline — NO external
<Include> files are used.  This avoids the jNeuroML double-read bug
where included NeuroML type files cause "Duplicate name for
ComponentType" or "no such dimension" errors.

All template variables are pre-computed by NeuroMLAdapter._ctx()
via tvbo.adapters.neuroml.build_lems_context() and injected directly
into the Mako namespace.

Variables available: dyn, dyn_id, params, svs, dvs, events,
  coupling_inputs, coupling_params, coupling_pre_rhs, coupling_post_rhs,
  coupling_global, sv_names_set, n_nodes, dt, duration,
  lems_expr (callable), _parse_piecewise (callable), lems_dim (callable),
  max_output_nodes (int), sim_id, time_scale, safe_id
</%doc>
<Lems>

  <!-- Tell jLEMS/jNeuroML which component is the simulation entry point. -->
  <Target component="${sim_id}"/>

  <!-- ════════════════════════════════════════════════════════════════
       Dimensions & Units (inline — no external includes needed)
       ════════════════════════════════════════════════════════════════ -->
<%include file="_lems_dims_units.xml.mako"/>

  <!-- ════════════════════════════════════════════════════════════════
       LEMS Simulation infrastructure types
       (normally provided by Simulation.xml — defined inline to avoid
       the jNeuroML double-read bug)
       ════════════════════════════════════════════════════════════════ -->
  <ComponentType name="Simulation">
    <Parameter name="length" dimension="time"/>
    <Parameter name="step" dimension="time"/>
    <Children name="outputs" type="OutputFile"/>
    <ComponentReference name="target" type="Component"/>
    <Dynamics>
      <StateVariable name="t" dimension="time"/>
    </Dynamics>
    <Simulation>
      <Run component="target" variable="t" increment="step" total="length"/>
    </Simulation>
  </ComponentType>

  <ComponentType name="OutputFile">
    <Children name="outputColumn" type="OutputColumn"/>
    <Text name="fileName"/>
    <Text name="path"/>
    <Simulation>
      <DataWriter path="path" fileName="fileName"/>
    </Simulation>
  </ComponentType>

  <ComponentType name="OutputColumn">
    <Path name="quantity"/>
    <Simulation>
      <Record quantity="quantity"/>
    </Simulation>
  </ComponentType>

  <!-- ════════════════════════════════════════════════════════════════
       Dynamics ComponentType & Component instances
       ════════════════════════════════════════════════════════════════ -->
% if has_network and cell_contexts:
## ── Multi-population: render ComponentTypes for each cell & synapse type ──

  <!-- baseSynapse: minimal base type for spike-triggered synapses.
       All custom synapse ComponentTypes reference EventPort "in" and
       expose "i" (synaptic current).  Post-cells collect synaptic current
       via <Children> + DerivedVariable select/reduce. -->
  <ComponentType name="baseSynapse">
    <EventPort name="in" direction="in"/>
    <Exposure name="i" dimension="current"/>
  </ComponentType>

## ── Cell-type ComponentTypes (non-synapse) ──
% for ct_name, ct in cell_contexts.items():
% if not ct.get('is_synapse'):
<%
  ct_dyn = ct['dyn']
  ct_dyn_id = ct['dyn_id']
  ct_params = ct['params']
  ct_svs = ct['svs']
  ct_dvs = ct['dvs']
  ct_events = ct['events']
  ct_coupling_inputs = ct['coupling_inputs']
  ct_sv_names_set = ct['sv_names_set']
  ct_needs_sec = ct['needs_sec']
  ct_lems_expr = ct['lems_expr']
  ct_parse_piecewise = ct['_parse_piecewise']
  ct_has_threshold = ct.get('has_threshold_events', False)
  ct_threshold_ev = set(ct.get('threshold_event_names', []))
  # coupling_inputs that are NOT parameters or state variables receive synaptic current
  ct_syn_inputs = [
      str(ci) for ci in ct_coupling_inputs
      if str(ci) not in ct_sv_names_set
      and str(ci) not in [str(k) for k in ct_params.keys()]
  ]
%>\

  <!-- ── ComponentType: ${ct_dyn_id} ── -->
  <ComponentType name="${ct_dyn_id}">
% for pname, p in ct_params.items():
    <Parameter name="${pname}" dimension="${lems_dim(getattr(p, 'unit', None))}"/>
% endfor
% for sv_name in ct_svs:
    <Parameter name="${sv_name}_0" dimension="${lems_dim(getattr(ct_svs[sv_name], 'unit', None))}"/>
% endfor
% if ct_needs_sec:
    <Constant name="SEC" dimension="time" value="1${time_scale}"/>
% endif
% for sv_name, sv in ct_svs.items():
    <Exposure name="${sv_name}" dimension="${lems_dim(getattr(sv, 'unit', None))}"/>
% endfor
% if ct_has_threshold:
    <EventPort name="spike" direction="out"/>
% endif
% if ct_syn_inputs:
    <!-- Collects all attached spike-triggered synapses; their currents are summed -->
    <Children name="synapses" type="baseSynapse"/>
% endif

    <Dynamics>
% for sv_name, sv in ct_svs.items():
      <StateVariable name="${sv_name}" dimension="${lems_dim(getattr(sv, 'unit', None))}" exposure="${sv_name}"/>
% endfor
% if ct_syn_inputs:
<%  ## Emit one DerivedVariable per coupling-input that sums synaptic currents.
    ## All conductance-based synapses expose dimension="current" via "i".
%>\
% for ci_name in ct_syn_inputs:
      <DerivedVariable name="${ci_name}" dimension="current" select="synapses[*]/i" reduce="add"/>
% endfor
% endif
% for dv_name, dv in ct_dvs.items():
<%
  eq = getattr(dv, 'equation', None)
  rhs = getattr(eq, 'rhs', None) if eq else None
  dv_dim = lems_dim(getattr(dv, 'unit', None))
  pw_cases = ct_parse_piecewise(rhs) if rhs else None
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
      <DerivedVariable name="${dv_name}" dimension="${dv_dim}" value="${ct_lems_expr(rhs)}"/>
% endif
% endif
% endfor
% for sv_name, sv in ct_svs.items():
<%
  eq = getattr(sv, 'equation', None)
  rhs = getattr(eq, 'rhs', None) if eq else None
%>\
% if rhs:
% if ct_needs_sec:
      <TimeDerivative variable="${sv_name}" value="(${ct_lems_expr(rhs)}) / SEC"/>
% else:
      <TimeDerivative variable="${sv_name}" value="${ct_lems_expr(rhs)}"/>
% endif
% endif
% endfor
      <OnStart>
% for sv_name, sv in ct_svs.items():
        <StateAssignment variable="${sv_name}" value="${sv_name}_0"/>
% endfor
      </OnStart>
% for ev_name, ev in ct_events.items():
<%
  cond = getattr(ev, 'condition', None)
  affect = getattr(ev, 'affect', None)
  cond_rhs = getattr(cond, 'rhs', None) if cond else None
  affect_rhs = getattr(affect, 'rhs', None) if affect else None
%>\
% if cond_rhs:
      <OnCondition test="${ct_lems_expr(cond_rhs)}">
% if affect_rhs:
% for assignment in str(affect_rhs).split(";"):
<% parts = assignment.strip().split("=", 1) if "=" in assignment else [] %>\
% if len(parts) == 2:
        <StateAssignment variable="${parts[0].strip()}" value="${ct_lems_expr(parts[1].strip())}"/>
% endif
% endfor
% endif
% if ev_name in ct_threshold_ev:
        <EventOut port="spike"/>
% endif
      </OnCondition>
% endif
% endfor
    </Dynamics>
  </ComponentType>

  <Component id="${ct_dyn_id}_inst" type="${ct_dyn_id}"\
% for pname, p in ct_params.items():
<% p_unit = getattr(p, 'unit', '') or '' %>\
 ${pname}="${getattr(p, 'value', 0)}${p_unit}"\
% endfor
% for sv_name, sv in ct_svs.items():
<% iv = getattr(sv, 'initial_value', None) %>\
 ${sv_name}_0="${iv if iv is not None else 0.0}"\
% endfor
/>
% endif
% endfor

## ── Synapse ComponentTypes (from edge dynamics) ──
% for ct_name, ct in cell_contexts.items():
% if ct.get('is_synapse'):
<%
  ct_dyn = ct['dyn']
  ct_dyn_id = ct['dyn_id']
  ct_params = ct['params']
  ct_svs = ct['svs']
  ct_dvs = ct['dvs']
  ct_events = ct['events']
  ct_coupling_inputs = ct['coupling_inputs']
  ct_sv_names_set = ct['sv_names_set']
  ct_needs_sec = ct['needs_sec']
  ct_lems_expr = ct['lems_expr']
  ct_parse_piecewise = ct['_parse_piecewise']
  ct_has_i = ct.get('has_i_exposure', False)
  ct_has_v = ct.get('has_v_req', False)
  ct_ext_evs = set(ct.get('external_event_names', []))
%>\

  <!-- ── Synapse ComponentType: ${ct_dyn_id} (extends baseSynapse) ── -->
  <ComponentType name="${ct_dyn_id}" extends="baseSynapse">
% for pname, p in ct_params.items():
    <Parameter name="${pname}" dimension="${lems_dim(getattr(p, 'unit', None))}"/>
% endfor
% for sv_name in ct_svs:
    <Parameter name="${sv_name}_0" dimension="${lems_dim(getattr(ct_svs[sv_name], 'unit', None))}"/>
% endfor
% if ct_needs_sec:
    <Constant name="SEC" dimension="time" value="1${time_scale}"/>
% endif
% if ct_has_v:
    <InstanceRequirement name="v" type="voltage"/>
% endif

    <Dynamics>
% for sv_name, sv in ct_svs.items():
      <StateVariable name="${sv_name}" dimension="${lems_dim(getattr(sv, 'unit', None))}"/>
% endfor
% for dv_name, dv in ct_dvs.items():
<%
  eq = getattr(dv, 'equation', None)
  rhs = getattr(eq, 'rhs', None) if eq else None
  dv_dim = lems_dim(getattr(dv, 'unit', None))
  pw_cases = ct_parse_piecewise(rhs) if rhs else None
  is_i = (dv_name == 'i')
%>\
% if rhs:
% if pw_cases:
% if len(pw_cases) == 1 and pw_cases[0][0] is None:
      <DerivedVariable name="${dv_name}" dimension="${dv_dim}"${ ' exposure="i"' if is_i else ''} value="${pw_cases[0][1]}"/>
% else:
      <ConditionalDerivedVariable name="${dv_name}" dimension="${dv_dim}"${ ' exposure="i"' if is_i else ''}>
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
      <DerivedVariable name="${dv_name}" dimension="${dv_dim}"${ ' exposure="i"' if is_i else ''} value="${ct_lems_expr(rhs)}"/>
% endif
% endif
% endfor
% for sv_name, sv in ct_svs.items():
<%
  eq = getattr(sv, 'equation', None)
  rhs = getattr(eq, 'rhs', None) if eq else None
%>\
% if rhs:
% if ct_needs_sec:
      <TimeDerivative variable="${sv_name}" value="(${ct_lems_expr(rhs)}) / SEC"/>
% else:
      <TimeDerivative variable="${sv_name}" value="${ct_lems_expr(rhs)}"/>
% endif
% endif
% endfor
      <OnStart>
% for sv_name, sv in ct_svs.items():
        <StateAssignment variable="${sv_name}" value="${sv_name}_0"/>
% endfor
      </OnStart>
% for ev_name, ev in ct_events.items():
<%
  cond = getattr(ev, 'condition', None)
  affect = getattr(ev, 'affect', None)
  cond_rhs = getattr(cond, 'rhs', None) if cond else None
  affect_rhs = getattr(affect, 'rhs', None) if affect else None
%>\
% if ev_name in ct_ext_evs:
## External event (no condition) → OnEvent port="in" (fired by incoming spike)
      <OnEvent port="in">
% if affect_rhs:
% for assignment in str(affect_rhs).split(";"):
<% parts = assignment.strip().split("=", 1) if "=" in assignment else [] %>\
% if len(parts) == 2:
        <StateAssignment variable="${parts[0].strip()}" value="${ct_lems_expr(parts[1].strip())}"/>
% endif
% endfor
% endif
      </OnEvent>
% elif cond_rhs:
      <OnCondition test="${ct_lems_expr(cond_rhs)}">
% if affect_rhs:
% for assignment in str(affect_rhs).split(";"):
<% parts = assignment.strip().split("=", 1) if "=" in assignment else [] %>\
% if len(parts) == 2:
        <StateAssignment variable="${parts[0].strip()}" value="${ct_lems_expr(parts[1].strip())}"/>
% endif
% endfor
% endif
      </OnCondition>
% endif
% endfor
    </Dynamics>
  </ComponentType>
% endif
% endfor

  <!-- ════════════════════════════════════════════════════════════════
       Built-in/named synapse definitions (no custom ODE dynamics)
       ════════════════════════════════════════════════════════════════ -->
% for syn in net_ctx['synapses']:
<%
  syn_id = syn['id']
  syn_type = syn['type']
  syn_params = syn['params']
  has_custom_dyn = syn['id'] in cell_contexts and cell_contexts[syn['id']].get('is_synapse')
%>\
% if not has_custom_dyn:
  <!-- Synapse: ${syn_id} (${syn_type}) -->
  <${syn_type} id="${syn_id}"\
% for pk, pinfo in syn_params.items():
<% pv = pinfo['value'] if isinstance(pinfo, dict) else pinfo; pu = (pinfo.get('unit') if isinstance(pinfo, dict) else None) or '' %>\
 ${pk}="${pv}${pu}"\
% endfor
/>
% endif
% endfor

  <!-- ════════════════════════════════════════════════════════════════
       Pulse Generators / Input Sources
       ════════════════════════════════════════════════════════════════ -->
% for inp in net_ctx.get('inputs', []):
  <${inp['type']} id="${inp['id']}"\
% for pk, pv in inp.get('params', {}).items():
 ${pk}="${pv}"\
% endfor
/>
% endfor

  <!-- ════════════════════════════════════════════════════════════════
       Network
       ════════════════════════════════════════════════════════════════ -->
  <network id="net1">
% for pop in net_ctx['populations']:
    <population id="${pop['id']}" component="${pop['component']}" size="${pop['size']}"/>
% endfor

% for conn in net_ctx['connections']:
<%
  has_wd = conn.get('weight') is not None or conn.get('delay') is not None
  conn_delay_unit = conn.get('delay_unit') or time_scale
%>\
% if has_wd:
    <synapticConnectionWD from="${conn['from_pop']}[${conn['from_idx']}]" to="${conn['to_pop']}[${conn['to_idx']}]" synapse="${conn['synapse']}" destination="synapses"\
% if conn.get('weight') is not None:
 weight="${conn['weight']}"\
% endif
% if conn.get('delay') is not None:
 delay="${conn['delay']}${conn_delay_unit}"\
% endif
/>
% else:
    <synapticConnection from="${conn['from_pop']}[${conn['from_idx']}]" to="${conn['to_pop']}[${conn['to_idx']}]" synapse="${conn['synapse']}" destination="synapses"/>
% endif
% endfor

% for inp in net_ctx.get('inputs', []):
    <explicitInput target="${inp['target_pop']}[${inp['target_idx']}]" input="${inp['id']}" destination="synapses"/>
% endfor
  </network>

  <!-- ════════════════════════════════════════════════════════════════
       Simulation — target the network
       ════════════════════════════════════════════════════════════════ -->
  <Simulation id="${sim_id}" length="${duration}${time_scale}" step="${dt}${time_scale}" target="net1">
<%
  ## Collect output columns: all state variables from non-synapse populations.
  out_cols = []
  for pop in net_ctx['populations']:
    ct = cell_contexts.get(pop['dyn_name'], {})
    if ct.get('is_synapse'):
        continue  # synapses are not directly recorded
    for sv_name in ct.get('svs', {}):
      for idx in range(pop['size']):
        col_id = f"{pop['id']}_{idx}_{sv_name}"
        quantity = f"{pop['id']}[{idx}]/{sv_name}"
        out_cols.append((col_id, quantity))
%>
    <OutputFile id="of1" fileName="results/${dyn_id}.dat">
% for col_id, quantity in out_cols:
      <OutputColumn id="${col_id}" quantity="${quantity}"/>
% endfor
    </OutputFile>
  </Simulation>

% else:
## ── Single-population: original behavior ──
<%include file="_lems_componenttype.xml.mako"/>

  <!-- ════════════════════════════════════════════════════════════════
       Simulation — target the component instance directly.
       Output: one file with all state variables.
       ════════════════════════════════════════════════════════════════ -->
  <Simulation id="${sim_id}" length="${duration}${time_scale}" step="${dt}${time_scale}" target="${dyn_id}_inst">

    <OutputFile id="of1" fileName="results/${dyn_id}.dat">
% for sv_name in svs:
      <OutputColumn id="${sv_name}" quantity="${sv_name}"/>
% endfor
    </OutputFile>

  </Simulation>
% endif

</Lems>
