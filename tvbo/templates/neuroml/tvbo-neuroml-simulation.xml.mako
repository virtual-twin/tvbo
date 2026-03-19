## -*- coding: utf-8 -*-
<%doc>
TVBO → LEMS Simulation Template
=================================
Renders a LEMS document containing only the Simulation block.
Optionally includes a network file via <Include file="..."/>.

Inject network_file=None or a filename string from NeuroMLAdapter.render_simulation().
All other template variables from tvbo.adapters.neuroml.build_lems_context().
</%doc>
<Lems>

  <Include file="Simulation.xml"/>

% if network_file:
  <Include file="${network_file}"/>
% endif

  <Target component="${sim_id}"/>

  <!-- ════════════════════════════════════════════════════════════════
       Simulation
       length=${duration}${time_scale}   step=${dt}${time_scale}
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
