# -*- coding: utf-8 -*-
"""tvboptim adapter for tvbo.

Export (tvbo → tvboptim)

- :func:`to_tvboptim` — Network → tvboptim Network or DenseGraph / DenseDelayGraph
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tvbo.classes.network import Network


def _build_graph(network: "Network", delays: bool = True):
    """Build a tvboptim graph from a tvbo Network.

    Returns a ``DenseDelayGraph`` when *delays* is True and the network
    has non-zero tract lengths, otherwise a ``DenseGraph``.
    """
    import jax.numpy as jnp
    from tvboptim.experimental.network_dynamics.graph import DenseGraph
    from tvboptim.experimental.network_dynamics.graph.base import DenseDelayGraph

    weights = jnp.asarray(np.asarray(network.weights_matrix, dtype=float))
    labels = network.node_labels
    lengths = network.lengths_matrix

    if delays and lengths is not None and np.any(lengths > 0):
        delay_matrix = jnp.asarray(np.asarray(network.calculate_delays(), dtype=float))
        return DenseDelayGraph(
            weights=weights,
            delays=delay_matrix,
            region_labels=labels,
        )
    return DenseGraph(weights=weights, region_labels=labels)


def _extract_noise(dyn_obj):
    """Extract tvboptim noise from tvbo dynamics state variable metadata.

    Iterates state variables looking for noise definitions.  Returns a
    tvboptim ``AdditiveNoise`` or ``MultiplicativeNoise`` when found,
    ``None`` otherwise.
    """
    svs = getattr(dyn_obj, "state_variables", None)
    if not svs:
        return None

    noisy_states = []
    sigma = None
    additive = True

    for sv_name, sv in svs.items():
        sv_noise = getattr(sv, "noise", None)
        if sv_noise is None:
            continue
        noisy_states.append(sv_name)
        # Extract sigma
        sv_sigma = getattr(sv_noise, "sigma", None)
        if sv_sigma is not None:
            sigma = float(sv_sigma)
        # Check additive flag
        if getattr(sv_noise, "additive", True) is False:
            additive = False

    if not noisy_states:
        return None

    if sigma is None:
        sigma = 0.01

    # Determine apply_to: None means all states
    all_sv_names = list(svs.keys())
    apply_to = noisy_states if set(noisy_states) != set(all_sv_names) else None

    if additive:
        from tvboptim.experimental.network_dynamics.noise import AdditiveNoise

        return AdditiveNoise(apply_to=apply_to, sigma=sigma)
    else:
        from tvboptim.experimental.network_dynamics.noise import MultiplicativeNoise

        return MultiplicativeNoise(apply_to=apply_to, sigma=sigma)


def to_tvboptim(
    network: "Network",
    delays: bool | None = None,
    return_type: str = "network",
    dynamics=None,
    coupling=None,
    noise=None,
    **kwargs,
):
    """Export a tvbo Network to a tvboptim Network or graph object.

    When *dynamics* / *coupling* are not provided explicitly, they are
    auto-extracted from ``network.dynamics`` and ``network.coupling``
    using each object's ``.execute('tvboptim')`` method.

    Parameters
    ----------
    network : Network
        tvbo Network instance with weights (and optionally lengths) matrices.
    delays : bool or None, default=None
        Whether to include delay matrices in the graph.  When ``None``
        (default), auto-inferred from ``network.coupling``: uses delays
        only when at least one coupling has ``delayed=True``.
    return_type : str, default="network"
        ``"network"`` — return a full ``tvboptim.experimental.network_dynamics.Network``
        (requires *dynamics* and *coupling*).
        ``"graph"`` — return only the ``DenseGraph`` / ``DenseDelayGraph``.
    dynamics : AbstractDynamics, optional
        tvboptim dynamics instance. If not given, auto-extracted from
        ``network.dynamics``.
    coupling : AbstractCoupling | dict, optional
        tvboptim coupling instance(s). If not given, auto-extracted from
        ``network.coupling``.
    noise : AbstractNoise, optional
        tvboptim noise instance. Optional.
    **kwargs
        Extra keyword arguments forwarded to the tvboptim ``Network`` constructor.

    Returns
    -------
    Network or DenseGraph or DenseDelayGraph
    """
    # Auto-infer delays from coupling metadata if not specified
    if delays is None:
        delays = False
        if hasattr(network, "coupling") and network.coupling:
            for coup_obj in network.coupling.values():
                if getattr(coup_obj, "delayed", False):
                    delays = True
                    break

    graph = _build_graph(network, delays=delays)

    if return_type == "graph":
        return graph

    # Auto-extract dynamics from network if not provided
    if dynamics is None and hasattr(network, "dynamics") and network.dynamics:
        dyn_key = next(iter(network.dynamics))
        dyn_obj = network.dynamics[dyn_key]
        dynamics = dyn_obj.execute("tvboptim")
    else:
        dyn_obj = None

    # Auto-extract coupling from network if not provided.
    # Resolution: use CouplingInput.source to remap function keys → CI keys,
    # then fall back to name matching, then positional order.
    if coupling is None and hasattr(network, "coupling") and network.coupling:
        coup_dict = {key: coup_obj.execute("tvboptim") for key, coup_obj in network.coupling.items()}
        if dynamics is not None and hasattr(dynamics, "COUPLING_INPUTS"):
            ci_keys = set(dynamics.COUPLING_INPUTS.keys())
            func_keys = list(coup_dict.keys())

            # Build func_name → ci_name mapping from source attribute
            remap = {}
            if dyn_obj is not None and hasattr(dyn_obj, "coupling_inputs") and dyn_obj.coupling_inputs:
                for ci_name, ci_obj in dyn_obj.coupling_inputs.items():
                    src = getattr(ci_obj, "source", None)
                    if src and src in coup_dict:
                        remap[src] = ci_name

            if remap:
                # Apply explicit source remapping
                coupling = {}
                for fk, fv in coup_dict.items():
                    ci_name = remap.get(fk, fk)
                    coupling[ci_name] = fv
            elif set(func_keys) <= ci_keys:
                # Names already match COUPLING_INPUTS
                coupling = coup_dict
            else:
                # Positional fallback
                coupling = list(coup_dict.values())
        else:
            coupling = coup_dict

    if dynamics is None or coupling is None:
        raise ValueError(
            "dynamics and coupling are required for return_type='network'. "
            "Set network.dynamics/coupling, pass them as kwargs, "
            "or use return_type='graph' for just the graph."
        )

    # Auto-extract noise from dynamics state variables if not provided
    if noise is None and dyn_obj is not None:
        noise = _extract_noise(dyn_obj)

    from tvboptim.experimental.network_dynamics import Network as TvboptimNetwork

    return TvboptimNetwork(
        dynamics=dynamics,
        coupling=coupling,
        graph=graph,
        noise=noise,
        **kwargs,
    )
