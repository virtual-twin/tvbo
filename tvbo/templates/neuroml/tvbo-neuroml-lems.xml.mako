## -*- coding: utf-8 -*-
<%doc>
TVBO → LEMS XML Template  (monolithic)
=======================================
Generates a complete, self-contained LEMS simulation file.
All template variables are pre-computed by NeuroMLAdapter._ctx()
via tvbo.adapters.neuroml.build_lems_context() and injected directly
into the Mako namespace — no Python setup needed here.

The ComponentType + Coupling definitions are shared with the split-file
dynamics template via <%include> fragments to avoid duplication.

Variables available: dyn, dyn_id, params, svs, dvs, events,
  coupling_inputs, coupling_params, coupling_pre_rhs, coupling_post_rhs,
  coupling_global, sv_names_set, n_nodes, dt, duration,
  lems_expr (callable), _parse_piecewise (callable), lems_dim (callable),
  max_output_nodes (int)
</%doc>
<Lems>

<%include file="_lems_dims_units.xml.mako"/>

<%include file="_lems_componenttype.xml.mako"/>

  <!-- ════════════════════════════════════════════════════════════════
       Network
       ════════════════════════════════════════════════════════════════ -->
  <Component id="net" type="network">
    <Component id="pop0" type="population" component="${dyn_id}" size="${n_nodes}"/>
  </Component>

  <!-- ════════════════════════════════════════════════════════════════
       Simulation
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
