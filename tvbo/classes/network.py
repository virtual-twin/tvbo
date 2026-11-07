"""Brain-network connectivity classes for TVBO.

[`Network`](#tvbo.classes.network.Network) is the generated class with its behaviour (`tvbo.behaviour.network.NetworkBehaviour`): the structural connectivity (weights and tract lengths), parcellation, nodes/edges and declarative transforms of a virtual brain, loaders from HDF5/YAML database files and BIDS derivatives, JAX pytree registration, and graph-Laplacian coupling primitives. [`Connectome`](#tvbo.classes.network.Connectome) is its deprecated alias.
"""

from tvbo import datamodel as tvbo_datamodel
from tvbo.behaviour.network import (
    NETWORK_DIR,
    LazyMaterializationWarning,
    get_normative_connectome_data,
    graph_laplacian,
    max_row_sum_normalized,
    normalized_graph_laplacian,
)

__all__ = [
    "NETWORK_DIR",
    "Connectome",
    "LazyMaterializationWarning",
    "Network",
    "get_normative_connectome_data",
    "graph_laplacian",
    "max_row_sum_normalized",
    "normalized_graph_laplacian",
]

Network = tvbo_datamodel.Network
"""The generated `Network`, which carries its behaviour itself; this is its public import location."""


class Connectome(Network):
    """Deprecated alias for Network. Use Network instead."""

    def __init__(self, *args, **kwargs):
        import warnings

        warnings.warn(
            "Connectome is deprecated and will be removed in a future version. "
            "Use tvbo.Network (tvbo.classes.network.Network) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
