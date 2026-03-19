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

% if dynamics_file:
  <Include file="${dynamics_file}"/>
% endif

  <!-- ════════════════════════════════════════════════════════════════
       Network: ${n_nodes} node(s), population of ${dyn_id}
       ════════════════════════════════════════════════════════════════ -->
  <Component id="net" type="network">
    <Component id="pop0" type="population" component="${dyn_id}" size="${n_nodes}"/>
  </Component>

</Lems>
