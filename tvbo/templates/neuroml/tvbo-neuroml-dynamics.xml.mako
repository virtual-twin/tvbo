## -*- coding: utf-8 -*-
<%doc>
TVBO → LEMS Dynamics Template
==============================
Renders a standalone LEMS document containing only ComponentType definitions
(the dynamics model + Coupling) and their default Component instances.

No Network or Simulation elements are included.  This file can be used as
a reusable component definition that other LEMS files reference via
  <Include file="..."/>

Shares the ComponentType and Dims/Units blocks with the monolithic template
via <%include> fragments — single source of truth for all LEMS XML output.

All template variables are injected by NeuroMLAdapter.render_dynamics()
via tvbo.adapters.neuroml.build_lems_context().
</%doc>
<Lems>

<%include file="_lems_dims_units.xml.mako"/>

<%include file="_lems_componenttype.xml.mako"/>

</Lems>
