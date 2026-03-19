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

  <!-- Tell jLEMS/jNeuroML which component is the simulation entry point. -->
  <Target component="${sim_id}"/>

  <!-- Simulation.xml chain-includes NeuroMLCoreDimensions.xml (all standard dims/units).
       Networks.xml chain-includes Cells.xml which defines baseCell/baseStandalone,
       enabling <population> and <network> for single- and multi-node networks.
       Both files pull in NeuroMLCoreDimensions.xml; jLEMS deduplicates includes. -->
  <Include file="Simulation.xml"/>
  <Include file="Networks.xml"/>

<%include file="_lems_dims_units.xml.mako"/>

<%include file="_lems_componenttype.xml.mako"/>

  <!-- ════════════════════════════════════════════════════════════════
       Network — works for any n_nodes (single neuron or population).
       Our ComponentType extends baseCell, so it is valid in a population.
       ════════════════════════════════════════════════════════════════ -->
  <network id="net">
    <population id="pop0" component="${dyn_id}_inst" size="${n_nodes}"/>
  </network>

  <!-- ════════════════════════════════════════════════════════════════
       Simulation — target the network.
       Output: one file with all (node, SV) combinations.
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
