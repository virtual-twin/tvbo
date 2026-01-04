<%doc>
Constants Template
==================

Provides reusable constants derived from experiment metadata.
These constants are available at module level in generated code.

True constants (invariant for a given network):
- n_nodes, N_nodes, N_NODES: Number of nodes/regions in the network

Usage in other templates:
    <%namespace name="const" file="/base/constants.mako"/>
    ${const.all_constants(experiment)}

Note: Constants use both lowercase (n_nodes) and uppercase (N_NODES)
variants for compatibility with different equation conventions.

NOT constants (these are experiment parameters):
- dt, duration, transient_time, conduction_speed, noise_sigma
</%doc>

## =============================================================================
## All Constants (main entry point)
## =============================================================================
<%def name="all_constants(experiment)">
<%
    n_nodes = experiment.network.number_of_regions if experiment.network else 1
    # Get BIDS observations from network
    bids_obs = experiment.network.observations if hasattr(experiment.network, 'observations') else {}
%>
# =============================================================================
# Constants (derived from experiment metadata)
# =============================================================================
# Network size (invariant for a given connectome)
n_nodes = N_nodes = N_NODES = ${n_nodes}

</%def>
