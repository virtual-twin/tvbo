## -*- coding: utf-8 -*-
##
## Hierarchical Custom LEMS Template
## ==================================
## Generates a self-contained LEMS simulation file with custom ComponentTypes
## that extend LEMS base types (baseCellMembPot, baseIonChannel, baseGate,
## baseVoltageDepRate, etc.).  All equations are explicit — no standard
## NeuroML biological types used, only the type-system infrastructure.
##
## Context variables (from _build_hier_custom_context):
##   component_types  - list of ComponentType definition dicts (leaf-first)
##   channels         - list of channel instance dicts with nested children
##   cell_type_name   - custom cell ComponentType name
##   cell_id          - cell instance id
##   cell_attrs_str   - cell instance XML attributes string
##   channel_pops     - list of channelPopulation dicts
##   inputs           - list of input generator dicts
##   network_id       - network id
##   population_id    - population id
##   sim_id           - simulation id
##   sim_length       - simulation length with unit (e.g. "150ms")
##   sim_step         - simulation step with unit (e.g. "0.01ms")
##   output_var       - output variable name (e.g. "v")
##   dyn_id           - dynamics id for file naming
##
<%!
def _attr(name, val):
    """Format an XML attribute pair."""
    return '%s="%s"' % (name, val)
%>\
<Lems>
  <Target component="${sim_id}" reportFile="report.txt"/>
  <Include file="Cells.xml"/>
  <Include file="Networks.xml"/>
  <Include file="Simulation.xml"/>
% for ct in component_types:

  <ComponentType name="${ct['name']}" extends="${ct['extends']}">
  % for p in ct.get('parameters', []):
    <Parameter name="${p['name']}" dimension="${p['dimension']}"/>
  % endfor
  % for name, typ in ct.get('child_slots', []):
    <Child name="${name}" type="${typ}"/>
  % endfor
  % for name, typ in ct.get('children_slots', []):
    <Children name="${name}" type="${typ}"/>
  % endfor
  % for name, typ in ct.get('attachments', []):
    <Attachments name="${name}" type="${typ}"/>
  % endfor
    <Dynamics>
    % for dv in ct['dynamics'].get('derived_variables', []):
      % if dv.get('select'):
      <DerivedVariable name="${dv['name']}" dimension="${dv['dimension']}"\
${' exposure="%s"' % dv['exposure'] if dv.get('exposure') else ''}\
${' required="false"' if dv.get('required_false') else ''}\
 select="${dv['select']}"\
${' reduce="%s"' % dv['reduce'] if dv.get('reduce') else ''}/>
      % else:
      <DerivedVariable name="${dv['name']}" dimension="${dv['dimension']}"\
${' exposure="%s"' % dv['exposure'] if dv.get('exposure') else ''}\
 value="${dv['value']}"/>
      % endif
    % endfor
    % for cdv in ct['dynamics'].get('cdvs', []):
      <ConditionalDerivedVariable name="${cdv['name']}" dimension="${cdv['dimension']}" exposure="${cdv['exposure']}">
      % for case in cdv['cases']:
        % if case.get('condition'):
        <Case condition="${case['condition']}" value="${case['value']}"/>
        % else:
        <Case value="${case['value']}"/>
        % endif
      % endfor
      </ConditionalDerivedVariable>
    % endfor
    % for sv in ct['dynamics'].get('state_variables', []):
      <StateVariable name="${sv['name']}" dimension="${sv['dimension']}" exposure="${sv['exposure']}"/>
    % endfor
    % for td in ct['dynamics'].get('time_derivatives', []):
      <TimeDerivative variable="${td['variable']}" value="${td['value']}"/>
    % endfor
    % for os in ct['dynamics'].get('on_start', []):
      <OnStart>
        <StateAssignment variable="${os['variable']}" value="${os['value']}"/>
      </OnStart>
    % endfor
    % for oc in ct['dynamics'].get('on_condition', []):
      <OnCondition test="${oc['test']}">
        <EventOut port="${oc['port']}"/>
      </OnCondition>
    % endfor
    </Dynamics>
  </ComponentType>
% endfor

  <!-- Concrete component instances -->
% for ch in channels:
  % if ch.get('children'):
  <${ch['type_name']} id="${ch['id']}" ${ch['attrs_str']}>
    % for gate in ch['children']:
    <${gate['type_name']} id="${gate['id']}" ${gate['attrs_str']}>
      % for role_name in gate.get('role_children', {}):
<% rate = gate['role_children'][role_name] %>\
      <${role_name} type="${rate['type_name']}" ${rate['attrs_str']}/>
      % endfor
    </${gate['type_name']}>
    % endfor
  </${ch['type_name']}>
  % else:
  <${ch['type_name']} id="${ch['id']}" ${ch['attrs_str']}/>
  % endif
% endfor

% for inp in inputs:
  <${inp['type']} id="${inp['id']}" delay="${inp['delay']}" duration="${inp['duration']}" amplitude="${inp['amplitude']}"/>
% endfor

  <${cell_type_name} id="${cell_id}" ${cell_attrs_str}>
% for pop in channel_pops:
    <channelPopulation id="${pop['id']}" ionChannel="${pop['ion_channel']}" number="${pop['number']}" erev="${pop['erev']}"/>
% endfor
  </${cell_type_name}>

  <network id="${network_id}">
    <population id="${population_id}" component="${cell_id}" size="1"/>
% for inp in inputs:
    <explicitInput target="${population_id}[0]" input="${inp['id']}" destination="synapses"/>
% endfor
  </network>

  <Simulation id="${sim_id}" length="${sim_length}" step="${sim_step}" target="${network_id}">
    <OutputFile id="of0" fileName="results/${dyn_id}_v.dat">
      <OutputColumn id="${output_var}" quantity="${population_id}[0]/${output_var}"/>
    </OutputFile>
  </Simulation>
</Lems>
