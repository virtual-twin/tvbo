#
# Module: network.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""
Network graph plotting utilities.

Three rendering backends:

1. **networkx** — standard ``nx.draw_networkx_*`` rendering
2. **bsplot**  — text-box nodes + curved edges via ``bsplot.graph``
3. **brain**   — 3-D brain surface with sphere nodes + tube edges
                  via ``bsplot.graph.plot_network_on_surface``
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _safe_norm(arr) -> np.ndarray:
    """Normalise array to [0, 1], handling empty / constant data."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return arr
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


def _node_in_strengths(G, log: bool = True) -> np.ndarray:
    """Return per-node incoming strength vector (ordered by ``G.nodes()``)."""
    s = np.zeros(G.number_of_nodes())
    for i, node_id in enumerate(G.nodes()):
        s[i] = sum(abs(d.get("weight", 1.0)) for _, _, d in G.in_edges(node_id, data=True))
    if log:
        s = np.log1p(s)
    return s


def _threshold_graph(G, W_flat, percentile: float):
    """Remove edges whose weight is below *percentile* of *W_flat*."""
    if percentile <= 0:
        return
    cutoff = np.percentile(W_flat, percentile)
    to_remove = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if abs(d.get("weight", 1.0)) < cutoff]
    G.remove_edges_from(to_remove)


def _resolve_positions(G, pos, network=None, plot_brain=None):
    """Return a ``{node_id: (x, y)}`` dict for 2-D layout.

    Parameters
    ----------
    G : networkx graph
    pos : str or dict
        ``"spring"`` / ``"graphviz"`` / brain-view string / explicit dict.
    network : Network, optional
        Used to read explicit node positions or atlas centres.
    plot_brain : str, optional
        ``"horizontal"`` / ``"sagittal"`` / ``"coronal"``.
    """
    # Check for explicit node positions from Network metadata
    nodes_have_positions = False
    if network is not None:
        nodes_obj = getattr(network, "nodes", None)
        if nodes_obj and len(nodes_obj) > 0:
            positions_from_nodes = {}
            for i, node in enumerate(nodes_obj):
                node_pos = getattr(node, "position", None)
                if node_pos is not None:
                    x = getattr(node_pos, "x", None)
                    y = getattr(node_pos, "y", None)
                    if x is not None and y is not None:
                        node_id = getattr(node, "id", i) or i
                        positions_from_nodes[node_id] = [float(x), float(y)]
            if len(positions_from_nodes) == len(nodes_obj):
                nodes_have_positions = True
                pos = positions_from_nodes

    if isinstance(pos, str):
        if pos == "spring":
            pos = nx.spring_layout(G, k=1 / np.sqrt(max(len(G.nodes), 1)), seed=1312)
        elif pos == "graphviz":
            pos = nx.nx_agraph.graphviz_layout(G, prog="neato")

    if plot_brain and not nodes_have_positions and network is not None:
        centres = network.get_centers()
        proj = {"horizontal": (0, 1), "sagittal": (1, 2), "coronal": (0, 2)}
        xi, yi = proj.get(plot_brain, (0, 1))
        pos = {i: [c[xi], c[yi]] for i, c in centres.items()}

    return pos


def _edge_data(G, attr: str = "weight"):
    """Return ``(edges_list, attr_vals)`` aligned arrays."""
    edges_list = list(G.edges(keys=True, data=True))
    vals = np.array([d.get(attr, 0.0) for _, _, _, d in edges_list], dtype=float) if edges_list else np.array([])
    return edges_list, vals


def _make_mappable(vals, cmap):
    """Build a ``ScalarMappable`` suitable for ``fig.colorbar(...)``."""
    data = vals if vals.size > 0 else np.array([0.0])
    vmin, vmax = float(np.min(data)), float(np.max(data))
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return cm.ScalarMappable(norm=norm, cmap=cmap)


# ---------------------------------------------------------------------------
# 1.  NetworkX native
# ---------------------------------------------------------------------------


def plot_graph_networkx(
    G: nx.MultiDiGraph,
    *,
    ax: Optional[Axes] = None,
    node_cmap: Union[str, Any] = "viridis",
    edge_cmap: Union[str, Any] = "viridis",
    node_color_by: str = "in-strength",
    node_size_by: Union[str, float] = "in-strength",
    node_size_scaling: float = 100,
    log_in_strength: bool = True,
    edge_color_by: str = "weight",
    pos: Optional[dict] = None,
    node_labels: bool = True,
    edge_labels: bool = True,
    fontsize: float = 8,
    edge_kwargs: Optional[Dict[str, Any]] = None,
    node_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[Figure, cm.ScalarMappable]:
    """Render a network graph with standard NetworkX drawing.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        Graph (typically from ``Network.graph``).
    ax : Axes, optional
        If *None*, a new figure is created and returned.
    node_cmap, edge_cmap : str or Colormap
        Colormaps for nodes / edges.
    node_color_by : str
        ``"in-strength"`` or ``"node"`` (index-based).
    node_size_by : str or float
        ``"in-strength"`` for proportional sizing, or a fixed float.
    node_size_scaling : float
        Scaling factor for node circles.
    log_in_strength : bool
        Apply ``log1p`` to node in-strength.
    edge_color_by : str
        Edge attribute used for colour mapping.
    pos : dict
        ``{node: (x, y)}``.  If *None*, spring layout is used.
    node_labels, edge_labels : bool
        Toggle label rendering.
    fontsize : float
        Label font size.
    edge_kwargs, node_kwargs : dict
        Extra kwargs forwarded to ``nx.draw_networkx_edges`` /
        ``nx.draw_networkx_nodes``.

    Returns
    -------
    Figure  (when *ax* is None) **or**  ScalarMappable  (when *ax* given).
    """
    edge_kwargs = {
        "min_source_margin": 15,
        "min_target_margin": 15,
        **(edge_kwargs or {}),
    }
    node_kwargs = {**(node_kwargs or {})}

    if isinstance(node_cmap, str):
        node_cmap = plt.get_cmap(node_cmap)
    if isinstance(edge_cmap, str):
        edge_cmap = plt.get_cmap(edge_cmap)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    if pos is None:
        pos = nx.spring_layout(G, k=1 / np.sqrt(max(len(G.nodes), 1)), seed=1312)

    # Edge colours
    edges_list, edge_vals = _edge_data(G, edge_color_by)
    norm_ev = _safe_norm(edge_vals)
    if norm_ev.size == 0:
        edge_colors: Any = []
    elif np.all(norm_ev == 0):
        edge_colors = ["black"] * len(edges_list)
    else:
        edge_colors = edge_cmap(norm_ev)

    # Node colours / sizes
    in_str = _node_in_strengths(G, log=log_in_strength)
    norm_in = _safe_norm(in_str)

    if node_size_by == "in-strength":
        node_sizes = 300 + norm_in * node_size_scaling
    else:
        node_sizes = np.full(len(G.nodes), 100 * float(node_size_by))

    if node_color_by == "in-strength":
        node_coloring = node_cmap(norm_in)
    elif node_color_by == "node":
        node_coloring = node_cmap(_safe_norm(np.arange(len(G.nodes), dtype=float)))
    else:
        node_coloring = node_cmap(np.zeros(len(G.nodes)))

    # Draw
    edgelist_draw = [(u, v, k) for u, v, k, _ in edges_list]
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edgelist_draw if edges_list else None,
        edge_color=edge_colors,
        edge_cmap=edge_cmap,
        ax=ax,
        **edge_kwargs,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color="white",
        edgecolors=node_coloring,
        linewidths=1,
        ax=ax,
        **node_kwargs,
    )
    if node_labels:
        lbl = {n: G.nodes[n].get("label", str(n)) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=lbl, ax=ax, font_size=fontsize)
    if edge_labels:
        if edges_list:
            elbl = {}
            for u, v, k, d in edges_list:
                val = d.get(edge_color_by, None)
                try:
                    elbl[(u, v, k)] = f"{float(val):.2f}"
                except (TypeError, ValueError):
                    elbl[(u, v, k)] = str(val)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=elbl, ax=ax, font_size=fontsize)

    if fig is not None:
        ax.axis("off")
        plt.close()
        return fig

    return _make_mappable(edge_vals, edge_cmap)


# ---------------------------------------------------------------------------
# 2.  bsplot  (text-box nodes + curved edges)
# ---------------------------------------------------------------------------


def plot_graph_bsplot(
    G: nx.MultiDiGraph,
    *,
    ax: Optional[Axes] = None,
    node_cmap: Union[str, Any] = "viridis",
    edge_cmap: Union[str, Any] = "viridis",
    node_color_by: str = "in-strength",
    log_in_strength: bool = True,
    edge_color_by: str = "type",
    pos: Optional[dict] = None,
    node_labels: bool = True,
    edge_labels: bool = True,
    fontsize: float = 8,
    edge_radius: float = 0.1,
    linewidth: float = 1.5,
    alpha: float = 0.9,
) -> Union[Figure, cm.ScalarMappable]:
    """Render a network graph with bsplot text-box nodes and curved edges.

    Requires ``bsplot`` to be installed.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        Graph (typically from ``Network.graph``).
    ax : Axes, optional
        If *None*, a new figure is created and returned.
    node_cmap, edge_cmap : str or Colormap
        Colormaps for nodes / edges.
    node_color_by : str
        ``"in-strength"`` or ``"node"``.
    log_in_strength : bool
        Apply ``log1p`` to in-strength.
    edge_color_by : str
        Edge attribute used for colour classification (default ``"type"``).
    pos : dict
        ``{node: (x, y)}``.  If *None*, spring layout is used.
    node_labels, edge_labels : bool
        Toggle label rendering.
    fontsize : float
        Label font size.
    edge_radius : float
        Curvature radius for edges.
    linewidth : float
        Edge line width.
    alpha : float
        Node box transparency.

    Returns
    -------
    Figure  (when *ax* is None) **or**  ScalarMappable  (when *ax* given).
    """
    from bsplot.graph.edges import draw_custom_edges
    from bsplot.graph.nodes import draw_custom_nodes

    if isinstance(node_cmap, str):
        node_cmap = plt.get_cmap(node_cmap)
    if isinstance(edge_cmap, str):
        edge_cmap = plt.get_cmap(edge_cmap)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    if pos is None:
        pos = nx.spring_layout(G, k=1 / np.sqrt(max(len(G.nodes), 1)), seed=1312)

    # Node colours
    in_str = _node_in_strengths(G, log=log_in_strength)
    norm_in = _safe_norm(in_str)
    if node_color_by == "in-strength":
        node_coloring = node_cmap(norm_in)
    elif node_color_by == "node":
        node_coloring = node_cmap(_safe_norm(np.arange(len(G.nodes), dtype=float)))
    else:
        node_coloring = node_cmap(np.zeros(len(G.nodes)))

    # Build label + colour dicts
    label_dict = {}
    node_colors_dict = {}
    for i, node_id in enumerate(G.nodes()):
        lbl = G.nodes[node_id].get("label", None) or str(node_id)
        label_dict[node_id] = lbl
        node_colors_dict[node_id] = node_coloring[i]

    # bsplot expects string keys
    G_str = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})
    pos_str = {str(k): v for k, v in pos.items()}
    label_dict_str = {str(k): v for k, v in label_dict.items()}
    node_colors_dict_str = {str(k): v for k, v in node_colors_dict.items()}

    # Ensure 'type' attr on edges
    for u, v, k, d in G_str.edges(keys=True, data=True):
        if "type" not in d:
            d["type"] = d.get("label", f"w={d.get('weight', 1.0):.2f}")

    draw_custom_edges(
        G_str,
        pos_str,
        ax=ax,
        edge_labels=edge_labels,
        edge_colors=edge_cmap.name if hasattr(edge_cmap, "name") else "viridis",
        color_by=edge_color_by,
        edge_radius=edge_radius,
        linewidth=linewidth,
        font_size=fontsize,
    )
    draw_custom_nodes(
        G_str,
        pos_str,
        labels=label_dict_str,
        font_size=fontsize,
        ax=ax,
        node_colors=node_colors_dict_str,
        alpha=alpha,
    )

    ax.axis("off")

    if fig is not None:
        plt.close()
        return fig

    _, edge_vals = _edge_data(G, "weight")
    return _make_mappable(edge_vals, edge_cmap)


# ---------------------------------------------------------------------------
# 3.  Brain surface  (3-D spheres + tubes on cortex)
# ---------------------------------------------------------------------------


def plot_graph_brain(
    network,
    *,
    ax: Optional[Axes] = None,
    weight_matrix: Optional[np.ndarray] = None,
    template: str = "fsaverage",
    density: str = "164k",
    hemi: Union[str, Tuple[str, str]] = ("lh", "rh"),
    view: str = "lateral",
    surface_alpha: float = 0.3,
    threshold_percentile: float = 90,
    node_radius: float = 1.5,
    node_color: str = "auto",
    node_cmap: str = "YlOrRd",
    node_data_key: str = "strength",
    node_scale: Optional[dict] = None,
    edge_radius: float = 0.15,
    edge_color: str = "steelblue",
    edge_cmap: Optional[str] = None,
    edge_data_key: str = "weight",
    edge_scale: Optional[dict] = None,
    log_weights: bool = False,
    figsize: Tuple[float, float] = (6, 5),
    **kwargs,
) -> Tuple[Figure, Axes, dict]:
    """Render a network on the cortical brain surface using ``bsplot``.

    Nodes are rendered as coloured spheres at their MNI coordinates
    (from the atlas metadata); edges as coloured tubes between them.

    Parameters
    ----------
    network : Network
        Network object with a parcellation atlas whose entities have
        ``center`` coordinates (x, y, z in MNI space).
    ax : Axes, optional
        If *None*, a new figure is created.
    template : str
        Surface template (``"fsaverage"``).
    density : str
        Surface mesh density (``"164k"``).
    hemi : str or tuple
        ``"lh"``, ``"rh"``, or ``("lh", "rh")``.
    view : str
        Camera view (``"lateral"``, ``"top"``, ``"front"``, etc.).
    surface_alpha : float
        Brain surface transparency.
    threshold_percentile : float
        Only render edges above this percentile weight.
    node_radius : float
        Sphere radius for nodes.
    node_color : str
        Uniform colour (or ``"auto"`` to use *node_cmap*).
    node_cmap : str
        Colormap for node data.
    node_data_key : str
        Node attribute for colour mapping (default ``"strength"``).
    node_scale : dict, optional
        Radius scaling, e.g. ``{"strength": 2, "mode": "exp"}``.
    edge_radius : float
        Tube radius for edges.
    edge_color : str
        Uniform edge colour (used when *edge_cmap* is ``None``).
    edge_cmap : str, optional
        Colormap for edge data.
    edge_data_key : str
        Edge attribute for colour mapping.
    edge_scale : dict, optional
        Tube radius scaling, e.g. ``{"weight": 8, "mode": "exp"}``.
    figsize : tuple
        Figure size if no *ax* is given.
    **kwargs
        Forwarded to ``bsplot.graph.plot_network_on_surface``.

    Returns
    -------
    fig : Figure
    ax : Axes
    mappables : dict
        ``ScalarMappable`` objects (keys ``"nodes"`` / ``"edges"``).
    """
    from bsplot.graph import create_network, plot_network_on_surface

    if node_scale is None:
        node_scale = {node_data_key: 2, "mode": "exp"}

    # Use MNI coordinates from atlas metadata
    centers = network.get_centers()
    W = weight_matrix if weight_matrix is not None else network.weights_matrix

    # Filter out nodes with no valid coordinates (0,0,0)
    valid_idx = [i for i, coord in centers.items() if not (coord[0] == 0 and coord[1] == 0 and coord[2] == 0)]

    if not valid_idx:
        raise ValueError("No nodes with valid coordinates found. Ensure the atlas metadata contains center coordinates.")

    # Extract sub-matrix for valid nodes
    idx = np.array(valid_idx)
    W_sub = W[np.ix_(idx, idx)]

    if log_weights:
        W_sub = np.where(W_sub > 0, np.log1p(W_sub), 0.0)

    # Build sequential centers dict for create_network
    seq_centers = {j: np.array(centers[i]) for j, i in enumerate(idx)}
    seq_labels = list(range(len(idx)))

    G = create_network(
        seq_centers,
        W_sub,
        labels=seq_labels,
        threshold_percentile=threshold_percentile,
    )

    # Annotate node strength
    for node in G.nodes():
        G.nodes[node]["strength"] = sum(d["weight"] for _, _, d in G.edges(node, data=True))

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    _, ax, mappables = plot_network_on_surface(
        G,
        ax=ax,
        template=template,
        density=density,
        hemi=hemi,
        view=view,
        surface_alpha=surface_alpha,
        node_radius=node_radius,
        node_color=node_color,
        node_data_key=node_data_key,
        node_cmap=node_cmap,
        node_scale=node_scale,
        edge_radius=edge_radius,
        edge_color=edge_color,
        edge_cmap=edge_cmap,
        edge_data_key=edge_data_key,
        edge_scale=edge_scale,
        **kwargs,
    )

    ax.axis("off")
    return fig, ax, mappables
