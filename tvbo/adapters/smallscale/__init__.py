"""Shared small-scale network-lowering core.

TVB-O's small-scale backends (NeuroML/LEMS, Brian2, later NEST/Jaxley) all turn
the same declarative primitives — ``Network``, ``Node(size)``,
``Edge(dynamics, coupling, events, connectivity)`` — into populations of cells and
an explicit set of cell-to-cell connections. That lowering is backend-neutral and
lives here **once**; each backend contributes only a role vocabulary, an
expression printer, a template tree, and its synapse/cell rendering.

This is a lowering *function library*, not a schema addition: no new classes enter
the datamodel. Biological identity stays on ``Dynamics.iri``.
"""

from tvbo.adapters.smallscale.lowering import (
    ConnectionRecord,
    assign_cell_population,
    classify_node_role,
    connectivity_pairs,
    expand_edge_connections,
    expand_input_targets,
    group_nodes_by_dynamics,
    merge_params,
    normalize_edge_params,
    safe_id,
    unique_component_id,
)

__all__ = [
    "ConnectionRecord",
    "assign_cell_population",
    "classify_node_role",
    "connectivity_pairs",
    "expand_edge_connections",
    "expand_input_targets",
    "group_nodes_by_dynamics",
    "merge_params",
    "normalize_edge_params",
    "safe_id",
    "unique_component_id",
]
