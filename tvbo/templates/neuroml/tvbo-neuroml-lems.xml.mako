## -*- coding: utf-8 -*-
<%doc>
TVBO → LEMS XML Template  (monolithic)
=======================================
Generates a complete, self-contained LEMS simulation file.
All dimensions, units, and LEMS infrastructure types (Simulation,
OutputFile, OutputColumn) are defined inline — NO external
<Include> files are used.  This avoids the jNeuroML double-read bug
where included NeuroML type files cause "Duplicate name for
ComponentType" or "no such dimension" errors.

All template variables are pre-computed by NeuroMLAdapter._ctx()
via tvbo.adapters.neuroml.build_lems_context() and injected directly
into the Mako namespace.

Variables available: dyn, dyn_id, params, svs, dvs, events,
  coupling_inputs, coupling_params, coupling_pre_rhs, coupling_post_rhs,
  coupling_global, sv_names_set, n_nodes, dt, duration,
  lems_expr (callable), _parse_piecewise (callable), lems_dim (callable),
  max_output_nodes (int), sim_id, time_scale, safe_id
</%doc>
<Lems>

  <!-- Tell jLEMS/jNeuroML which component is the simulation entry point. -->
  <Target component="${sim_id}"/>

  <!-- ════════════════════════════════════════════════════════════════
       Dimensions & Units (inline — no external includes needed)
       ════════════════════════════════════════════════════════════════ -->
<%include file="_lems_dims_units.xml.mako"/>

  <!-- ════════════════════════════════════════════════════════════════
       LEMS Simulation infrastructure types
       (normally provided by Simulation.xml — defined inline to avoid
       the jNeuroML double-read bug)
       ════════════════════════════════════════════════════════════════ -->
  <ComponentType name="Simulation">
    <Parameter name="length" dimension="time"/>
    <Parameter name="step" dimension="time"/>
    <Children name="outputs" type="OutputFile"/>
    <ComponentReference name="target" type="Component"/>
    <Dynamics>
      <StateVariable name="t" dimension="time"/>
    </Dynamics>
    <Simulation>
      <Run component="target" variable="t" increment="step" total="length"/>
    </Simulation>
  </ComponentType>

  <ComponentType name="OutputFile">
    <Children name="outputColumn" type="OutputColumn"/>
    <Text name="fileName"/>
    <Text name="path"/>
    <Simulation>
      <DataWriter path="path" fileName="fileName"/>
    </Simulation>
  </ComponentType>

  <ComponentType name="OutputColumn">
    <Path name="quantity"/>
    <Simulation>
      <Record quantity="quantity"/>
    </Simulation>
  </ComponentType>

  <!-- ════════════════════════════════════════════════════════════════
       Dynamics ComponentType & Component instances
       ════════════════════════════════════════════════════════════════ -->
<%include file="_lems_componenttype.xml.mako"/>

  <!-- ════════════════════════════════════════════════════════════════
       Simulation — target the component instance directly.
       Output: one file with all state variables.
       ════════════════════════════════════════════════════════════════ -->
  <Simulation id="${sim_id}" length="${duration}${time_scale}" step="${dt}${time_scale}" target="${dyn_id}_inst">

    <OutputFile id="of1" fileName="results/${dyn_id}.dat">
% for sv_name in svs:
      <OutputColumn id="${sv_name}" quantity="${sv_name}"/>
% endfor
    </OutputFile>

  </Simulation>

</Lems>
