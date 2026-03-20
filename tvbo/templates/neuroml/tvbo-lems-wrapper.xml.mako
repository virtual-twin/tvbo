## -*- coding: utf-8 -*-
<%doc>
TVBO → LEMS Simulation Wrapper Template
=========================================
Generates a LEMS simulation wrapper that includes a NeuroML (.nml) file
and provides the Simulation block needed to run it.

The NeuroML file is expected to contain:
  - Custom ComponentType definitions (the dynamics model + coupling)
  - Component instances (default parameters)
  - A <network id="net"> with populations

This wrapper adds:
  - Standard NeuroML type includes (Cells.xml, Networks.xml, Simulation.xml)
  - The <Include> for the NeuroML file
  - A <Simulation> block targeting the network

Generate the companion NeuroML file via NeuroMLAdapter.render_neuroml().

All template variables are pre-computed by NeuroMLAdapter._ctx()
via tvbo.adapters.neuroml.build_lems_context().
</%doc>
<Lems>

  <!-- Simulation entry point -->
  <Target component="${sim_id}"/>

  <!-- Standard NeuroML type definitions -->
  <Include file="Cells.xml"/>
  <Include file="Networks.xml"/>
  <Include file="Simulation.xml"/>

  <!-- NeuroML model file (ComponentTypes, Components, Network) -->
% if neuroml_file:
  <Include file="${neuroml_file}"/>
% endif

  <!-- ════════════════════════════════════════════════════════════════
       Simulation
       length=${duration}${time_scale}   step=${dt}${time_scale}
       target=net (defined in the included NeuroML file)
       ════════════════════════════════════════════════════════════════ -->
  <Simulation id="${sim_id}" length="${duration}${time_scale}" step="${dt}${time_scale}" target="net">

    <OutputFile id="of1" fileName="results/${dyn_id}.dat">
<%
  n_out = min(n_nodes, max_output_nodes)
%>\
% for sv_name in svs:
% for node_idx in range(n_out):
      <OutputColumn id="${sv_name}_${node_idx}" quantity="pop0[${node_idx}]/${sv_name}"/>
% endfor
% endfor
    </OutputFile>

  </Simulation>

</Lems>
