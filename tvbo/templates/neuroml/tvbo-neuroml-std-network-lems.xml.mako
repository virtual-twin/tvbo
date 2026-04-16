## -*- coding: utf-8 -*-
<%doc>
Standard NeuroML types: multi-population network LEMS file
==========================================================
Renders a self-contained <Lems> file for a network of standard NeuroML types.

Sections (in order):
  1. Header & Includes
  2. Custom ComponentType definitions
  3. Cell type definitions (channels + cell)
  4. Input components
  5. Synapse components
  6. Silent synapses (for continuous connections)
  7. Network block (populations, projections, connections, inputs)
  8. Simulation block (OutputFile / OutputColumn)

All template variables are injected by ``build_std_lems_context()``.
</%doc>\
<%
    _SEG_KEYS = ('preSegmentId', 'preFractionAlong', 'postSegmentId', 'postFractionAlong')

    def _cell_ref(pop_obj, pop_id, idx, prefix='../'):
        """Build a cell reference path for connections."""
        if pop_obj and pop_obj.get('is_populationlist'):
            return f'{prefix}{pop_id}/{idx}/{pop_obj["component"]}'
        return f'{prefix}{pop_id}[{idx}]'

    def _cell_ref_no_prefix(pop_obj, pop_id, idx):
        """Build a cell reference path without ../ prefix."""
        if pop_obj and pop_obj.get('is_populationlist'):
            return f'{pop_id}/{idx}/{pop_obj["component"]}'
        return f'{pop_id}[{idx}]'
%>\
<Lems>
  <Target component="${sim_id}"/>

  <Include file="Cells.xml"/>
% if needs_pynn:
  <Include file="PyNN.xml"/>
% endif
  <Include file="Networks.xml"/>
% if needs_inputs_include:
  <Include file="Inputs.xml"/>
% endif
  <Include file="Simulation.xml"/>

## ── Custom ComponentType definitions ──
% for ct_xml in custom_type_xmls:
${ct_xml}

% endfor
## ── Cell type definitions (channels + cell) ──
% for cell_xml in cell_xmls_all:
${cell_xml}

% endfor
## ── Input components ──
% for inp_xml in input_xmls_all:
${inp_xml}

% endfor
## ── Synapse components ──
% for syn_xml in synapse_xmls:
${syn_xml}

% endfor
## ── Silent synapses for continuous connections ──
% for silent_id in silent_ids:
    <silentSynapse id="${silent_id}"/>

% endfor
## ══════════════════════════════════════════════════════════════════════
## Network block
## ══════════════════════════════════════════════════════════════════════
    <network id="net1">
## ── Cell populations ──
% for pop in populations:
% if not pop.get('is_input'):
% if pop.get('is_populationlist'):
        <population id="${pop['id']}" type="populationList" component="${pop['component']}">
% for idx, pos in enumerate(pop['node_positions']):
            <instance id="${idx}"><location x="${pos[0]}" y="${pos[1]}" z="${pos[2]}"/></instance>
% endfor
        </population>
% else:
        <population id="${pop['id']}" component="${pop['component']}" size="${pop['size']}"/>
% endif
% endif
% endfor
## ── Event source populations ──
% for pop in populations:
% if pop.get('is_input'):
        <population id="${pop['id']}" component="${pop['component']}" size="${pop['size']}"/>
% endif
% endfor
## ── Chemical synaptic connections ──
% if chem_conns and needs_projection:
## Projection mode (segment targeting)
% for pidx, ((fp, tp, syn), proj_conns) in enumerate(chem_projs.items()):
        <projection id="projection${pidx}" presynapticPopulation="${fp}" postsynapticPopulation="${tp}" synapse="${syn}">
% for cidx, conn in enumerate(proj_conns):
<%
    fp_pop = pop_map.get(fp)
    tp_pop = pop_map.get(tp)
    pre = _cell_ref(fp_pop, fp, conn['from_idx'])
    post = _cell_ref(tp_pop, tp, conn['to_idx'])
    seg_attrs = ''.join(f' {sk}="{conn[sk]}"' for sk in _SEG_KEYS if conn.get(sk) is not None)
%>\
% if conn.get('weight') is not None or conn.get('delay') is not None:
            <connectionWD id="${cidx}" preCellId="${pre}" postCellId="${post}" weight="${conn.get('weight', 1.0) or 1.0}" delay="${conn.get('delay', '0ms') or '0ms'}"${seg_attrs}/>
% else:
            <connection id="${cidx}" preCellId="${pre}" postCellId="${post}"${seg_attrs}/>
% endif
% endfor
        </projection>
% endfor
% elif chem_conns:
## synapticConnection mode (no segment targeting)
% for conn in chem_conns:
<%
    fp = conn['from_pop']
    tp = conn['to_pop']
    syn = conn['synapse']
    fp_pop = pop_map.get(fp)
    tp_pop = pop_map.get(tp)
    pre = _cell_ref_no_prefix(fp_pop, fp, conn['from_idx'])
    post = _cell_ref_no_prefix(tp_pop, tp, conn['to_idx'])
%>\
% if conn.get('weight') is not None or conn.get('delay') is not None:
        <synapticConnectionWD from="${pre}" to="${post}" synapse="${syn}" destination="synapses" weight="${conn.get('weight', 1.0) or 1.0}" delay="${conn.get('delay', '0ms') or '0ms'}"/>
% else:
        <synapticConnection from="${pre}" to="${post}" synapse="${syn}" destination="synapses"/>
% endif
% endfor
% endif
## ── Electrical projections (gap junctions) ──
% for pidx, ((fp, tp, syn), proj_conns) in enumerate(elec_projs.items()):
        <electricalProjection id="elecProj_${pidx}" presynapticPopulation="${fp}" postsynapticPopulation="${tp}">
% for cidx, conn in enumerate(proj_conns):
% if conn.get('weight') is not None:
            <electricalConnectionInstanceW id="${cidx}" preCell="../${fp}[${conn['from_idx']}]" postCell="../${tp}[${conn['to_idx']}]" synapse="${syn}" weight="${conn['weight']}"/>
% else:
            <electricalConnection id="${cidx}" preCell="${conn['from_idx']}" postCell="${conn['to_idx']}" synapse="${syn}"/>
% endif
% endfor
        </electricalProjection>
% endfor
## ── Continuous projections (graded synapses) ──
% for pidx, ((fp, tp, syn), proj_conns) in enumerate(cont_projs.items()):
        <continuousProjection id="contProj_${pidx}" presynapticPopulation="${fp}" postsynapticPopulation="${tp}">
% for cidx, conn in enumerate(proj_conns):
<%  pre_comp = conn.get('pre_component') or 'silent1' %>\
% if conn.get('weight') is not None:
            <continuousConnectionInstanceW id="${cidx}" preCell="../${fp}[${conn['from_idx']}]" postCell="../${tp}[${conn['to_idx']}]" preComponent="${pre_comp}" postComponent="${syn}" weight="${conn['weight']}"/>
% else:
            <continuousConnection id="${cidx}" preCell="${conn['from_idx']}" postCell="${conn['to_idx']}" preComponent="${pre_comp}" postComponent="${syn}"/>
% endif
% endfor
        </continuousProjection>
% endfor
## ── Explicit inputs ──
% if needs_input_list:
% for gidx, ((inp_id, tgt_pop), group) in enumerate(input_groups.items()):
        <inputList id="inputList_${gidx}" component="${inp_id}" population="${tgt_pop}">
% for iidx, inp in enumerate(group):
<%
    tp_pop = pop_map.get(tgt_pop)
    if tp_pop and tp_pop.get('is_populationlist'):
        target = f'../{tgt_pop}/{inp["target_idx"]}/{tp_pop["component"]}'
    else:
        target = f'../{tgt_pop}[{inp["target_idx"]}]'
    seg_attrs = ''
    if inp.get('segmentId') is not None:
        seg_attrs += f' segmentId="{inp["segmentId"]}"'
    if inp.get('fractionAlong') is not None:
        seg_attrs += f' fractionAlong="{inp["fractionAlong"]}"'
%>\
% if inp.get('weight') is not None:
            <inputW id="${iidx}" target="${target}"${seg_attrs} destination="synapses" weight="${inp['weight']}"/>
% else:
            <input id="${iidx}" target="${target}"${seg_attrs} destination="synapses"/>
% endif
% endfor
        </inputList>
% endfor
% else:
% for inp in explicit_inputs:
        <explicitInput target="${inp['target_pop']}[${inp['target_idx']}]" input="${inp['input_id']}" destination="synapses"/>
% endfor
% endif
    </network>

## ══════════════════════════════════════════════════════════════════════
## Simulation block
## ══════════════════════════════════════════════════════════════════════
    <Simulation id="${sim_id}" length="${duration}${time_scale}" step="${dt}${time_scale}" target="net1"${seed_attr}>
<%  of_idx = 0 %>\
% for pop_id_out, pop_size, sv_var in output_pops:
<%
    pop_obj = pop_map.get(pop_id_out)
    seg_ids = pop_obj.get('segment_ids', []) if pop_obj else []
    is_poplist = pop_obj.get('is_populationlist', False) if pop_obj else False
    comp = pop_obj.get('component', '') if pop_obj else ''
%>\
% if seg_ids and len(seg_ids) > 1 and is_poplist:
## Multi-compartment: separate OutputFile per cell instance
% for idx in range(pop_size):
        <OutputFile id="of${of_idx}" fileName="results/${comp}_${idx}.dat">
% for seg_id in seg_ids:
            <OutputColumn id="${idx}_${seg_id}" quantity="${pop_id_out}/${idx}/${comp}/${seg_id}/${sv_var}"/>
% endfor
        </OutputFile>
<%  of_idx += 1 %>\
% endfor
% else:
        <OutputFile id="of${of_idx}" fileName="results/${comp or dyn_id}.dat">
% for idx in range(pop_size):
<%
    col_id = f'{pop_id_out}_{idx}_{sv_var}'
    if is_poplist:
        quantity = f'{pop_id_out}/{idx}/{comp}/{sv_var}'
    else:
        quantity = f'{pop_id_out}[{idx}]/{sv_var}'
%>\
            <OutputColumn id="${col_id}" quantity="${quantity}"/>
% endfor
        </OutputFile>
<%  of_idx += 1 %>\
% endif
% endfor
    </Simulation>

</Lems>
