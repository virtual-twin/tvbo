<%!
from tvbo.utils import initial_value as _initial_value
%>\
## -*- coding: utf-8 -*-
<%doc>
TVBO → LEMS XML Template  (monolithic / include-based)
==========================================================
Generates a complete LEMS simulation file.

Single-population mode (no network):
  All dimensions, units, and LEMS infrastructure types (Simulation,
  OutputFile, OutputColumn) are defined inline.

Multi-population network mode:
  Includes standard NeuroML2 type files (Cells.xml, Networks.xml,
  Simulation.xml) which provide all standard types (synapse types,
  input types, network infrastructure, simulation types, dimensions
  and units).  Custom cell/synapse dynamics are rendered as additional
  custom ComponentTypes alongside the standard types.

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

% if has_network and cell_contexts:
  <!-- ════════════════════════════════════════════════════════════════
       Standard NeuroML2 type includes for network mode.
       Provides all standard dimensions, units, synapse types, input
       types, network infrastructure, and simulation types.
       ════════════════════════════════════════════════════════════════ -->
  <Include file="Cells.xml"/>
  <Include file="Networks.xml"/>
  <Include file="Simulation.xml"/>
% else:
  <Include file="Cells.xml"/>
  <Include file="Networks.xml"/>
  <Include file="Simulation.xml"/>
% endif

  <!-- ════════════════════════════════════════════════════════════════
       Dynamics ComponentType & Component instances
       ════════════════════════════════════════════════════════════════ -->
% if has_network and cell_contexts:

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
  ct_lems_dim = ct.get('lems_dim', lems_dim)
  ct_lems_sym = ct.get('lems_sym', lems_sym)
  ct_has_threshold = ct.get('has_threshold_events', False)
  ct_threshold_ev = set(ct.get('threshold_event_names', []))
  ct_regime_data = ct.get('regime_data')
  # coupling_inputs that are NOT parameters or state variables receive synaptic current
  ct_syn_inputs = [
      str(ci) for ci in ct_coupling_inputs
      if str(ci) not in ct_sv_names_set
      and str(ci) not in [str(k) for k in ct_params.keys()]
  ]
  # In network mode, all non-synapse cells extend baseCellMembPot and get
  # Attachments so they can receive current from explicitInput / synapses.
  ct_extends_cell = True
  # v Exposure and spike EventPort come from the base type when extending
  ct_v_from_base = ct_extends_cell and 'v' in ct_sv_names_set
%>\

  <!-- ── ComponentType: ${ct_dyn_id} ── -->
% if ct_extends_cell:
  <ComponentType name="${ct_dyn_id}" extends="baseCellMembPot">
% else:
  <ComponentType name="${ct_dyn_id}">
% endif
% for pname, p in ct_params.items():
    <Parameter name="${pname}" dimension="${ct_lems_dim(getattr(p, 'unit', None))}"/>
% endfor
## Default `refract` in for the refractory regime, unless the model declares its own and it would be emitted twice.
% if ct_regime_data and 'refract' not in ct_params:
    <Parameter name="refract" dimension="time"/>
% endif
% for sv_name in ct_svs:
    <Parameter name="${sv_name}_0" dimension="${ct_lems_dim(getattr(ct_svs[sv_name], 'unit', None))}" />
% endfor
% if ct_needs_sec:
    <Constant name="SEC" dimension="time" value="1${time_scale}"/>
% endif
% for sv_name, sv in ct_svs.items():
% if ct_v_from_base and sv_name == 'v':
## v Exposure is inherited from baseCellMembPot — do not redeclare
% else:
    <Exposure name="${sv_name}" dimension="${ct_lems_dim(getattr(sv, 'unit', None))}" />
% endif
% endfor
% if ct_has_threshold and not ct_extends_cell:
    <EventPort name="spike" direction="out"/>
% endif
% if ct_extends_cell:
    <!-- Dynamically attached synapses/inputs from network connections -->
    <Attachments name="synapses" type="basePointCurrent"/>
% endif

    <Dynamics>
% for sv_name, sv in ct_svs.items():
      <StateVariable name="${sv_name}" dimension="${ct_lems_dim(getattr(sv, 'unit', None))}" exposure="${sv_name}"/>
% endfor
% if ct_regime_data:
      <StateVariable name="lastSpikeTime" dimension="time"/>
% endif
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
  rhs = eq if states_an_expression(eq) else None
  dv_dim = ct_lems_dim(getattr(dv, 'unit', None))
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
% if ct_regime_data:
      <!-- ── Regime-based dynamics (spike model) ── -->
      <OnStart>
% for sv_name, sv in ct_svs.items():
        <StateAssignment variable="${sv_name}" value="${sv_name}_0"/>
% endfor
      </OnStart>

      <Regime name="integrating" initial="true">
% for sv_name, sv in ct_svs.items():
<%
  eq = getattr(sv, 'equation', None)
  rhs = eq if states_an_expression(eq) else None
%>\
% if rhs:
% if ct_needs_sec:
        <TimeDerivative variable="${sv_name}" value="(${ct_lems_expr(rhs)}) / SEC"/>
% else:
        <TimeDerivative variable="${sv_name}" value="${ct_lems_expr(rhs)}"/>
% endif
% endif
% endfor
        <OnCondition test="${ct_lems_expr(ct_regime_data['condition'])}">
          <EventOut port="spike"/>
          <Transition regime="refractory"/>
        </OnCondition>
      </Regime>

      <Regime name="refractory">
% for sv_name, sv in ct_svs.items():
<%
  eq = getattr(sv, 'equation', None)
  rhs = eq if states_an_expression(eq) else None
%>\
% if rhs and sv_name not in ct_regime_data['reset_vars']:
% if ct_needs_sec:
        <TimeDerivative variable="${sv_name}" value="(${ct_lems_expr(rhs)}) / SEC"/>
% else:
        <TimeDerivative variable="${sv_name}" value="${ct_lems_expr(rhs)}"/>
% endif
% endif
% endfor
        <OnEntry>
          <StateAssignment variable="lastSpikeTime" value="t"/>
% for lhs, rhs_val in ct_regime_data['assignments']:
          <StateAssignment variable="${lhs}" value="${ct_lems_expr(rhs_val)}"/>
% endfor
        </OnEntry>
        <OnCondition test="t .gt. lastSpikeTime + refract">
          <Transition regime="integrating"/>
        </OnCondition>
      </Regime>
% elif ct_has_threshold and ct_extends_cell:
<%
  ## ── Edge-triggered spike detection via Regime pattern ──
  ## jLEMS fires plain OnCondition at EVERY timestep where the condition
  ## is true (level-triggered) for custom ComponentTypes. Standard types
  ## like pointCellCondBased are edge-triggered. Using Regime transitions
  ## ensures the spike fires only once per upward threshold crossing.
  ##
  ## Negate comparison operators for the exit condition:
  _NEGATE_CMP = {'.gt.': '.leq.', '.lt.': '.geq.',
                 '.geq.': '.lt.', '.leq.': '.gt.',
                 '.GT.': '.LEQ.', '.LT.': '.GEQ.',
                 '.GEQ.': '.LT.', '.LEQ.': '.GT.'}
  import re as _re
  _CMP_PAT = _re.compile(r'\.(?:gt|lt|geq|leq)\.', _re.IGNORECASE)
  def _negate_cond(cond_str):
      return _CMP_PAT.sub(lambda m: _NEGATE_CMP[m.group(0)], cond_str)
%>\
      <!-- ── Edge-triggered spike dynamics (Regime pattern for network cells) ── -->
      <OnStart>
% for sv_name, sv in ct_svs.items():
        <StateAssignment variable="${sv_name}" value="${sv_name}_0"/>
% endfor
      </OnStart>

      <Regime name="integrating" initial="true">
% for sv_name, sv in ct_svs.items():
<%
  eq = getattr(sv, 'equation', None)
  rhs = eq if states_an_expression(eq) else None
%>\
% if rhs:
% if ct_needs_sec:
        <TimeDerivative variable="${sv_name}" value="(${ct_lems_expr(rhs)}) / SEC"/>
% else:
        <TimeDerivative variable="${sv_name}" value="${ct_lems_expr(rhs)}"/>
% endif
% endif
% endfor
% for ev_name, ev in ct_events.items():
<%
  cond = getattr(ev, 'condition', None)
  cond_rhs = getattr(cond, 'rhs', None) if cond else None
%>\
% if cond_rhs and ev_name in ct_threshold_ev:
        <OnCondition test="${ct_lems_expr(cond_rhs)}">
          <EventOut port="spike"/>
          <Transition regime="refractory"/>
        </OnCondition>
% endif
% endfor
      </Regime>

      <Regime name="refractory">
% for sv_name, sv in ct_svs.items():
<%
  eq = getattr(sv, 'equation', None)
  rhs = eq if states_an_expression(eq) else None
%>\
% if rhs:
% if ct_needs_sec:
        <TimeDerivative variable="${sv_name}" value="(${ct_lems_expr(rhs)}) / SEC"/>
% else:
        <TimeDerivative variable="${sv_name}" value="${ct_lems_expr(rhs)}"/>
% endif
% endif
% endfor
% for ev_name, ev in ct_events.items():
<%
  cond = getattr(ev, 'condition', None)
  cond_rhs = getattr(cond, 'rhs', None) if cond else None
%>\
% if cond_rhs and ev_name in ct_threshold_ev:
        <OnCondition test="${_negate_cond(ct_lems_expr(cond_rhs))}">
          <Transition regime="integrating"/>
        </OnCondition>
% endif
% endfor
      </Regime>
% else:
      <!-- ── Flat dynamics ── -->
% for sv_name, sv in ct_svs.items():
<%
  eq = getattr(sv, 'equation', None)
  rhs = eq if states_an_expression(eq) else None
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
% endif
    </Dynamics>
  </ComponentType>

  <Component id="${ct_dyn_id}_inst" type="${ct_dyn_id}"\
% for pname, p in ct_params.items():
<% p_unit = ct_lems_sym(getattr(p, 'unit', None)) %>\
## Omit an unset parameter so LEMS names it, rather than writing "None" and failing on the quantity.
% if getattr(p, 'value', None) is not None:
 ${pname}="${p.value}${(' ' + p_unit) if p_unit else ''}"\
% endif
% endfor
% if ct_regime_data and 'refract' not in ct_params:
 refract="0 ${time_scale}"\
% endif
% for sv_name, sv in ct_svs.items():
<% iv = _initial_value(sv) %>\
<% sv_unit = ct_lems_sym(getattr(sv, 'unit', None)) %>\
 ${sv_name}_0="${iv if iv is not None else 0.0}${(' ' + sv_unit) if sv_unit else ''}"\
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
  ct_lems_dim = ct.get('lems_dim', lems_dim)
  ct_lems_sym = ct.get('lems_sym', lems_sym)
  ct_has_v = ct.get('has_v_req', False)
  ct_ext_evs = set(ct.get('external_event_names', []))
  ct_synapse_extends = ct.get('synapse_extends', 'baseSynapse')
  ct_syn_inherited = set(ct.get('synapse_inherited_params', ()))
  ct_exposure_names = set(ct.get('synapse_exposure_names', ('i',)))
  # The per-connection weight scales the current the postsynaptic cell sums.
  ct_weighted = ct.get('weighted_exposure')
  ct_weigh = lambda n, v: 'weight * (%s)' % v if n == ct_weighted else v
  # A variable named for one of the base type's exposures fulfils it, which LEMS requires the subtype to do.
  ct_expose = lambda n: ' exposure="%s"' % n if n in ct_exposure_names else ''
%>\

  <!-- ── Synapse ComponentType: ${ct_dyn_id} (extends ${ct_synapse_extends}) ── -->
  <ComponentType name="${ct_dyn_id}" extends="${ct_synapse_extends}">
% if ct_weighted:
## jLEMS hides baseSynapse's inherited `weight` from the DerivedVariable checker, so re-declare it as gradedSynapse does.
    <Property name="weight" dimension="none" defaultValue="1"/>
% endif
% for pname, p in ct_params.items():
% if pname not in ct_syn_inherited:
    <Parameter name="${pname}" dimension="${ct_lems_dim(getattr(p, 'unit', None))}"/>
% endif
% endfor
% for sv_name in ct_svs:
    <Parameter name="${sv_name}_0" dimension="${ct_lems_dim(getattr(ct_svs[sv_name], 'unit', None))}"/>
% endfor
% if ct_needs_sec:
    <Constant name="SEC" dimension="time" value="1${time_scale}"/>
% endif
% if ct_has_v:
    <InstanceRequirement name="v" type="voltage"/>
% endif

    <Dynamics>
% for sv_name, sv in ct_svs.items():
      <StateVariable name="${sv_name}" dimension="${ct_lems_dim(getattr(sv, 'unit', None))}"${ct_expose(sv_name)}/>
% endfor
% for dv_name, dv in ct_dvs.items():
<%
  eq = getattr(dv, 'equation', None)
  rhs = eq if states_an_expression(eq) else None
  dv_dim = ct_lems_dim(getattr(dv, 'unit', None))
  pw_cases = ct_parse_piecewise(rhs) if rhs else None
%>\
% if rhs:
% if pw_cases:
% if len(pw_cases) == 1 and pw_cases[0][0] is None:
      <DerivedVariable name="${dv_name}" dimension="${dv_dim}"${ct_expose(dv_name)} value="${ct_weigh(dv_name, pw_cases[0][1])}"/>
% else:
      <ConditionalDerivedVariable name="${dv_name}" dimension="${dv_dim}"${ct_expose(dv_name)}>
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
      <DerivedVariable name="${dv_name}" dimension="${dv_dim}"${ct_expose(dv_name)} value="${ct_weigh(dv_name, ct_lems_expr(rhs))}"/>
% endif
% endif
% endfor
% for sv_name, sv in ct_svs.items():
<%
  eq = getattr(sv, 'equation', None)
  rhs = eq if states_an_expression(eq) else None
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

  <Component id="${ct_dyn_id}_inst" type="${ct_dyn_id}"\
% for pname, p in ct_params.items():
<% p_unit = ct_lems_sym(getattr(p, 'unit', None)) %>\
## Omit an unset parameter so LEMS names it, rather than writing "None" and failing on the quantity.
% if getattr(p, 'value', None) is not None:
 ${pname}="${p.value}${(' ' + p_unit) if p_unit else ''}"\
% endif
% endfor
% for sv_name, sv in ct_svs.items():
<% iv = _initial_value(sv) %>\
<% sv_unit = ct_lems_sym(getattr(sv, 'unit', None)) %>\
 ${sv_name}_0="${iv if iv is not None else 0.0}${(' ' + sv_unit) if sv_unit else ''}"\
% endfor
/>
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
<% pv = pinfo['value'] if isinstance(pinfo, dict) else pinfo; pu = lems_sym_real((pinfo.get('unit') if isinstance(pinfo, dict) else None)) %>\
 ${pk}="${pv}${(' ' + pu) if pu else ''}"\
% endfor
/>
% endif
% endfor

  <!-- ════════════════════════════════════════════════════════════════
       Input Sources (pulseGenerator, spikeGenerator, spikeArray, etc.)
       ════════════════════════════════════════════════════════════════ -->
## ── Current-injection sources (standalone components, one per input) ──
% for inp in net_ctx.get('input_components', []):
  <${inp['type']} id="${inp['id']}"\
% for pk, pv in inp.get('params', {}).items():
 ${pk}="${pv}"\
% endfor
/>
% endfor
## ── Event sources (spikeGenerator, spikeArray — these are populations) ──
% for pop in net_ctx['populations']:
% if pop.get('is_input'):
<%
  inp_type = pop['input_type']
  inp_id = pop['input_id']
  inp_params = pop.get('input_params', {})
  spike_children_xml = pop.get('spike_children_xml', '')
%>\
% if spike_children_xml:
  <${inp_type} id="${inp_id}"\
% for pk, pv in inp_params.items():
 ${pk}="${pv}"\
% endfor
>
${spike_children_xml}
  </${inp_type}>
% else:
  <${inp_type} id="${inp_id}"\
% for pk, pv in inp_params.items():
 ${pk}="${pv}"\
% endfor
/>
% endif
% endif
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
## synapticConnectionWD requires both weight and delay, so supply the neutral default for whichever the edge left unset.
    <synapticConnectionWD from="${conn['from_pop']}[${conn['from_idx']}]" to="${conn['to_pop']}[${conn['to_idx']}]" synapse="${conn['synapse']}" destination="synapses" weight="${1.0 if conn.get('weight') is None else conn['weight']}" delay="${0 if conn.get('delay') is None else conn['delay']}${conn_delay_unit}"/>
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
  ## Collect output columns: all state variables from non-synapse, non-input populations.
  out_cols = []
  for pop in net_ctx['populations']:
    if pop.get('is_input'):
        continue  # input source populations don't have output columns
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
## ── Single-population: wrapped in NeuroML network for all-backend compatibility ──
## Quantity paths of the form pop[0]/sv_name are required by DLemsWriter (NetPyNE)
## and are also resolved correctly by the jNeuroML reference simulation engine.
<%include file="_lems_componenttype.xml.mako"/>

  <!-- Wrap the single cell in a network so quantity paths use the standard
       pop[idx]/variable form, which works across jNeuroML, NEURON, Brian2,
       NetPyNE and EDEN backends. -->
  <network id="net">
    <population id="pop" component="${dyn_id}_inst" size="1"/>
  </network>

  <!-- ════════════════════════════════════════════════════════════════
       Simulation — target the network (all-backend compatible)
       ════════════════════════════════════════════════════════════════ -->
  <Simulation id="${sim_id}" length="${duration}${time_scale}" step="${dt}${time_scale}" target="net">

    <OutputFile id="of1" fileName="results/${dyn_id}.dat">
% for sv_name in svs:
      <OutputColumn id="${sv_name}" quantity="pop[0]/${sv_name}"/>
% endfor
    </OutputFile>

  </Simulation>
% endif

</Lems>
