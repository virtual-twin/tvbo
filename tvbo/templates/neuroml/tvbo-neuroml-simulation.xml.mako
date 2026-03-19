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

% if network_file:
  <Include file="${network_file}"/>
% endif

  <!-- ════════════════════════════════════════════════════════════════
       Simulation
       length=${duration}ms   step=${dt}ms
       ════════════════════════════════════════════════════════════════ -->
  <Simulation id="sim1" length="${duration}ms" step="${dt}ms" target="net">

% for sv_name in svs:
    <OutputFile id="of_${sv_name}" fileName="results/${dyn_id}_${sv_name}.dat">
% for node_idx in range(min(n_nodes, max_output_nodes)):
      <OutputColumn id="${sv_name}_${node_idx}" quantity="pop0[${node_idx}]/${sv_name}"/>
% endfor
    </OutputFile>
% endfor

  </Simulation>

</Lems>
