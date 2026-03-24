## -*- coding: utf-8 -*-
<%doc>
TVBO → LEMS Network Template
=============================
Renders a LEMS document containing only the Network component.
Optionally includes a dynamics file via <Include file="..."/>.

Inject dynamics_file=None or a filename string from NeuroMLAdapter.render_network().
All other template variables from tvbo.adapters.neuroml.build_lems_context().
</%doc>
<Lems>

  <Include file="Networks.xml"/>

% if dynamics_file:
  <Include file="${dynamics_file}"/>
% endif

  <!-- ════════════════════════════════════════════════════════════════
       Network: ${n_nodes} node(s), population of ${dyn_id}
       ════════════════════════════════════════════════════════════════ -->
  <network id="net">
    <population id="pop0" component="${dyn_id}_inst" size="${n_nodes}"/>
  </network>

</Lems>
