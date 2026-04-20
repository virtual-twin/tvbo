## -*- coding: utf-8 -*-
<%doc>
Standard NeuroML types: single-cell LEMS file
==============================================
Renders a self-contained <Lems> file for a single NeuroML standard-type cell.

Handles two modes (selected by ``is_fhn``):
  1. FitzHugh-Nagumo cells (with Display + OutputFile)
  2. Regular standard cells (pointCellCondBased, cell, flat types)
     with optional tissue-temperature wrapper.

All template variables are injected by ``build_std_lems_context()``.
</%doc>
\
% if is_fhn:
## ─── FitzHugh-Nagumo cell ───────────────────────────────────────────
<Lems>
  <Target component="${sim_id}"/>

  <Include file="Cells.xml"/>
  <Include file="Networks.xml"/>
  <Include file="Inputs.xml"/>
  <Include file="Simulation.xml"/>

  <${cell_tag} id="${dyn_id}" ${'  '.join(f'{k}="{v}"' for k, v in cell_attrs.items())}/>

  <network id="net1">
    <population id="${pop_id}" component="${dyn_id}" size="1"/>
  </network>

  <Simulation id="${sim_id}" length="${duration}${time_scale}" step="${dt}${time_scale}" target="net1">

    <Display id="d1" title="${dyn_name}" timeScale="1${time_scale}" xmin="0" xmax="${int(duration)}" ymin="-2.5" ymax="2.5">
% for i, sv_name in enumerate(sv_names):
      <Line id="${sv_name}" quantity="${pop_id}[0]/${sv_name}" scale="1" color="${colors[i % len(colors)]}" timeScale="1${time_scale}"/>
% endfor
    </Display>

    <OutputFile id="of1" fileName="results/${dyn_id}.dat">
% for sv_name in sv_names:
      <OutputColumn id="${sv_name}" quantity="${pop_id}[0]/${sv_name}"/>
% endfor
    </OutputFile>

  </Simulation>

</Lems>\
% else:
## ─── Standard cell (pointCellCondBased / cell / flat types) ─────────
<Lems>
  <Target component="${sim_id}"/>

  <Include file="Cells.xml"/>
  <Include file="Networks.xml"/>
% if has_inputs:
  <Include file="Inputs.xml"/>
% endif
  <Include file="Simulation.xml"/>

% for ct_xml in custom_type_xmls:
${ct_xml}

% endfor
% for cm_xml in conc_xmls:
${cm_xml}

% endfor
% for ch_xml in channel_xmls:
${ch_xml}

% endfor
${cell_xml}

% for inp_xml in input_xmls:
${inp_xml}

% endfor
% if use_tissue:
    <ComponentType name="baseTissue" description="...">
        <Child name="network" type="network"/>
    </ComponentType>

    <ComponentType name="tissueWithVaryingTemperature" description="..." extends="baseTissue">
        <Exposure name="temperature" dimension="temperature"/>
        <Parameter name="startTemperature" dimension="temperature"/>
        <Parameter name="endTemperature" dimension="temperature"/>
        <Parameter name="changeTime" dimension="time"/>
        <Dynamics>
            <StateVariable name="temperature" exposure="temperature" dimension="temperature"/>
            <OnStart>
                <StateAssignment variable="temperature" value="startTemperature"/>
            </OnStart>
            <OnCondition test="t .gt. changeTime">
                <StateAssignment variable="temperature" value="endTemperature"/>
            </OnCondition>
        </Dynamics>
    </ComponentType>

    <tissueWithVaryingTemperature id="slice" startTemperature="${tissue_start}" endTemperature="${tissue_end}" changeTime="${tissue_change}">
        <network id="net1">
            <population id="pop" component="${dyn_id}" size="1"/>
% for inp_ref in input_refs:
    ${inp_ref}
% endfor
        </network>
    </tissueWithVaryingTemperature>
% elif net_temp:
    <network id="net1" type="networkWithTemperature" temperature="${net_temp}">
        <population id="pop" component="${dyn_id}" size="1"/>
% for inp_ref in input_refs:
${inp_ref}
% endfor
    </network>
% else:
    <network id="net1">
        <population id="pop" component="${dyn_id}" size="1"/>
% for inp_ref in input_refs:
${inp_ref}
% endfor
    </network>
% endif

    <Simulation id="${sim_id}" length="${duration}${time_scale}" step="${dt}${time_scale}" target="${sim_target}">
        <OutputFile id="of0" fileName="results/${dyn_id}.dat">
            <OutputColumn id="v" quantity="${quantity_prefix}pop[0]/v"/>
        </OutputFile>
    </Simulation>

</Lems>\
% endif
