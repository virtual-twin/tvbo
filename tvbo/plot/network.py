#
# Module: network.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""
Plotting functions for visualizing networks
-------------------------------------------

```{python}
# | fig-align: center
import matplotlib.pyplot as plt
from tvbo import plot

fig, ax = plt.subplots()
plot.network.plot_model('Generic2dOscillator', ax=ax)
```
"""

import pickle
from collections import defaultdict
from itertools import count
from typing import Any, Dict, Optional, Tuple

import matplotlib
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import colormaps, rcParams
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch, Patch
from sympy import latex, symbols

from tvbo.knowledge import graph, ontology
from tvbo.knowledge.simulation import equations
from tvbo.utils import get_cmap, get_continuous_cmap, tvb_colors

######################
# Default Parameters #
######################
node_params = {
    "node_size_factor": 100,
    "node_filling": "category",
    "node_linewidth": 1,
    "node_size_by": "degree",
    "label_as_symbol": False,
    "relabel": None,
    "draw_labels": True,
}

edge_params = {
    "draw_edgelabels": False,
    "colored_edges": False,
    "edge_width": 2,
    "edge_font_size": 12,
}

figure_params = {
    "figsize": (19, 15),
    "font_size": 15,
    "alpha": 1,
    "xpad": 0.5,
    "ypad": 0.5,
    "return_fig": False,
}

color_params = {
    "c_order": None,
    "colors": tvb_colors,
}

logo_params = {
    "logo_pos": [0.7, 0.3, 0.2, 0.2],
}

layout_params = {
    "layout": "kamada",
    "ax": None,
}

legend_params = {
    "legend": True,
    "legend_size": 10,
    "legend_bbox": (0.68, 0.24),
    "legend_loc": "lower right",
}


def get_default_params():
    default_params = {
        **node_params,
        **edge_params,
        **figure_params,
        **color_params,
        **logo_params,
        **layout_params,
        **legend_params,
    }

    return default_params


##################
# Graph Plotting #
##################
def reverse_edges(pos):
    """
    Reverse the coordinates of the positions in the pos dictionary.

    Parameters:
    ----------
    pos : dict
        A dictionary with nodes as keys and (x, y) coordinates as values.

    Returns:
    -------
    dict
        A dictionary with nodes as keys and reversed (y, x) coordinates as values.
    """
    reverse_pos = dict()
    for k, (x, y) in pos.items():
        reverse_pos[k] = np.array([y, x])
    return reverse_pos


def get_edge_color_mapping(G, colormap="viridis", color_by="type"):
    """
    Generate a color mapping for edges in graph G based on their specified attribute.

    Parameters:
    ----------
    G : networkx.Graph
        The graph containing the edges with the specified attribute.
    colormap : str, optional
        The name of the colormap to use (default is 'viridis').
    color_by : str, optional
        The edge attribute to color by (default is 'type').

    Returns:
    -------
    edge_colors : dict
        A dictionary mapping edge attributes to colors.
    """
    # Extract all edge attributes for the specified attribute
    edge_attributes = [
        data[color_by] if color_by in data.keys() else "n.a."
        for u, v, data in G.edges(data=True)
    ]
    unique_edge_attributes = list(set(edge_attributes))

    # Ensure "n.a." is first in the list
    if "n.a." in unique_edge_attributes:
        unique_edge_attributes.remove("n.a.")
    unique_edge_attributes.insert(0, "n.a.")

    # Create a colormap
    cmap = colormaps[colormap]
    norm = Normalize(vmin=0, vmax=len(unique_edge_attributes) - 1)

    # Map edge attributes to colors
    edge_colors = {
        attr: cmap(norm(i)) if attr != "n.a." else (0.6, 0.6, 0.6, 1)
        for i, attr in enumerate(unique_edge_attributes)
    }

    return edge_colors


def compute_bezier_midpoint(p0, p1, p2, t=0.5):
    """
    Compute the midpoint of a quadratic Bézier curve at a given t.

    Parameters:
    - p0, p1, p2: Control points of the Bézier curve.
    - t: The parameter at which to evaluate the midpoint (default is 0.5 for true midpoint).

    Returns:
    - tuple of (x, y) coordinates of the midpoint.
    """
    midpoint_x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
    midpoint_y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
    return (midpoint_x, midpoint_y)


def compute_midpoint(start, end, rad):
    """
    Compute the midpoint of a quadratic Bézier curve.

    Parameters:
    - start: tuple of (x, y) coordinates of the start point.
    - end: tuple of (x, y) coordinates of the end point.
    - rad: curvature radius as a percentage.

    Returns:
    - tuple of (x, y) coordinates of the midpoint.
    """
    x1, y1 = start
    x2, y2 = end
    y12 = (y1 + y2) / 2
    dy = y2 - y1
    cy = y12 + rad * dy

    # Ensure cy is a single value
    if isinstance(cy, np.ndarray):
        cy = cy.item()

    # Midpoint of the quadratic Bézier curve
    t = 0.5
    midpoint_x = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * (x1 + (x2 - x1) / 2) + t**2 * x2
    midpoint_y = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t**2 * y2
    return (midpoint_x, midpoint_y)


def add_arrow(line, ax, position=None, direction="right", color=None, label=""):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    color:      if None, line color is taken.
    label:      label for arrow
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == "right":
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    dx = xdata[end_ind] - xdata[start_ind]
    dy = ydata[end_ind] - ydata[start_ind]
    size = abs(dx) * 5.0
    x = xdata[start_ind] + (np.sign(dx) * size / 2.0)
    y = ydata[start_ind] + (np.sign(dy) * size / 2.0)

    arrow = patches.FancyArrow(
        x,
        y,
        dx,
        dy,
        color=color,
        width=0,
        head_width=size,
        head_length=size,
        label=label,
        length_includes_head=True,
        overhang=0.3,
        zorder=10,
    )
    ax.add_patch(arrow)


def plot_curve(
    ax,
    start,
    end,
    rad,
    color="black",
    shrinkA=0,
    shrinkB=0.03,
    arrow_size=0.1,
    arrow_style="->",
    annotate=None,
    **kwargs,
):
    font_size = kwargs.pop("font_size") if "font_size" in kwargs else 12

    x1, y1 = start
    x2, y2 = end
    y12 = (y1 + y2) / 2
    dy = y2 - y1
    cy = y12 + rad * dy
    midpoint = compute_midpoint(start, end, rad)

    tau = np.linspace(0, 1, 100)
    xsupport = np.linspace(x1, x2, 100)
    ysupport = [(1 - i) ** 2 * y1 + 2 * (1 - i) * i * cy + i**2 * y2 for i in tau]

    # Apply shrink to the start and end
    shrink_index_start = int(len(xsupport) * shrinkA)
    shrink_index_end = int(len(xsupport) * (1 - shrinkB))

    line = ax.plot(
        xsupport[shrink_index_start:shrink_index_end],
        ysupport[shrink_index_start:shrink_index_end],
        color=color,
        **kwargs,
    )
    line = line[0]
    # ax.scatter(midpoint[0], midpoint[1], color="red", s=100)
    # ax.scatter([x1, x2], [y1, y2], color="blue")
    if isinstance(annotate, str):
        ax.annotate(
            annotate,
            xy=midpoint,
            xytext=(midpoint[0], midpoint[1]),  # + 0.05 * ax.get_xlim()[1],
            xycoords="data",
            ha="left",
            va="center",
            color="k",
            fontsize=font_size,
        )

    # add_arrow(line, ax)
    # arrow_head_length = arrow_size
    # arrow_head_width = arrow_size / 2
    # # Add arrow tip at the end of the line
    # end_arrow_index = shrink_index_end - 1

    # # Compute the derivative at the last point (dx/dy) for proper arrow tilt
    # dx = xsupport[end_arrow_index] - xsupport[end_arrow_index - 1]
    # dy = ysupport[end_arrow_index] - ysupport[end_arrow_index - 1]

    # # Normalize the direction (dx, dy) to ensure consistent arrow scaling
    # norm = (dx**2 + dy**2) ** 0.5
    # dx /= norm
    # dy /= norm

    # ax.arrow(
    #     xsupport[end_arrow_index],
    #     ysupport[end_arrow_index],
    #     dx * arrow_head_length,  # Scale the arrow direction by the head length
    #     dy * arrow_head_length,
    #     lw=0,
    #     length_includes_head=True,
    #     head_width=arrow_head_width,
    #     head_length=arrow_head_length,
    #     color=color,
    # )


def get_edge_info(n1, n2, adj_matrix):
    # Get edges for n1 -> n2
    edges_n1_n2 = adj_matrix.get((n1, n2), [])
    num_edges_n1_n2 = len(edges_n1_n2)

    # Get edges for n2 -> n1 (reverse direction)
    edges_n2_n1 = adj_matrix.get((n2, n1), [])
    num_edges_n2_n1 = len(edges_n2_n1)

    return num_edges_n1_n2, edges_n1_n2, num_edges_n2_n1, edges_n2_n1


def create_adj_matrix(G):
    # Initialize adjacency matrix dictionary
    adj_matrix = {}

    # Loop through adjacency structure to build adjacency matrix
    for node, neighbors in G.adj.items():
        for neighbor, edge_attrs in neighbors.items():
            # Initialize an entry for the node pair if it doesn't exist
            if (node, neighbor) not in adj_matrix:
                adj_matrix[(node, neighbor)] = []

            # Loop through the edges between node and neighbor
            for edge_index, attrs in edge_attrs.items():
                # Append edge type and direction to the adjacency matrix
                adj_matrix[(node, neighbor)].append(
                    {"type": attrs["type"], "direction": f"{node} -> {neighbor}"}
                )
    return adj_matrix


def get_unique_node_pairs(G):
    unique_pairs = set()  # Use a set to store unique node pairs
    for n1, n2 in G.edges():
        # Sort using the 'name' attribute of the nodes (assuming each node has a 'name' attribute)
        pair = tuple(
            sorted(
                [n1, n2], key=lambda node: node if isinstance(node, str) else node.name
            )
        )
        unique_pairs.add(pair)  # Add the sorted pair to the set
    return unique_pairs


def draw_custom_edges(
    G,
    pos,
    ax=None,
    edge_labels=False,
    color_by="type",
    edge_colors="#606060",
    edge_radius=0,
    **kwargs,
):
    # if "shrinkA" not in kwargs.keys():
    #     kwargs["shrinkA"] = .1
    # if "shrinkB" not in kwargs.keys():
    #     kwargs["shrinkB"] = .3
    # if "arrowstyle" not in kwargs.keys():
    #     kwargs["arrowstyle"] = "-|>,head_length=0.4,head_width=0.2"
    # if "mutation_scale" not in kwargs.keys():
    #     kwargs["mutation_scale"] = 10.0

    if edge_colors in colormaps.keys():
        cmap = (
            edge_colors
            if isinstance(edge_colors, str) and edge_colors in colormaps.keys()
            else "viridis"
        )
        colmap = get_edge_color_mapping(G, colormap=cmap, color_by=color_by)
        edge_colors = [
            colmap[data[color_by] if color_by in data else "n.a."]
            for (u, v, data) in G.edges(data=True)
        ]
    elif isinstance(edge_colors, str):
        edge_colors = np.repeat(edge_colors, len(G.edges()))

    for n1, n2 in get_unique_node_pairs(G):
        point1 = pos[n1]
        point2 = pos[n2]
        types12 = {e["type"] for e in G[n1][n2].values()} if n2 in G[n1] else set()
        types21 = {e["type"] for e in G[n2][n1].values()} if n1 in G[n2] else set()

        num_edges = len(types12) + len(types21)

        if num_edges == 1:
            rad = [edge_radius]
        else:
            rad = np.linspace(-0.5, 0.5, num_edges)

        i = 0
        for t in types12:
            plot_curve(
                ax,
                point1,
                point2,
                rad[i],
                color=edge_colors[i],
                annotate=t if edge_labels else None,
                **kwargs,
            )
            i += 1
        for t in types21:
            plot_curve(
                ax,
                point2,
                point1,
                rad[i],
                color=edge_colors[i],
                annotate=t if edge_labels else None,
                **kwargs,
            )
            i += 1


def count_directed_edges(G) -> Dict[Tuple[Any, Any], int]:
    """
    Count number of edges between each directed node pair (u, v), not treating (u, v) ≡ (v, u).

    Returns:
        Dict[Tuple[Any, Any], int]: Mapping (u, v) → count of edges from u to v.
    """
    counts = defaultdict(int)
    for u, v, _ in G.edges(data=True):
        counts[(u, v)] += 1
    return counts

def n1n2_edgecounts(G, n1, n2, edge_counts: Optional[Dict[Tuple[Any, Any], int]] = None) -> int:
    """
    Return the number of edges between nodes n1 and n2, treating (n1, n2) ≡ (n2, n1).

    Args:
        G (networkx.Graph or MultiGraph): Input graph.
        n1 (hashable): First node identifier.
        n2 (hashable): Second node identifier.
        edge_counts (dict, optional): Precomputed output from count_directed_edges(G).

    Returns:
        int: Number of edges between n1 and n2, regardless of direction.
    """
    if edge_counts is None:
        edge_counts = edge_counts = count_directed_edges(G)
    return edge_counts.get(tuple(sorted((n1, n2))), 0)


def draw_custom_arrows(
    G,
    pos,
    edge_width: int = 1,
    ax: Optional[Any] = None,
    edge_colors: str = "grey",
    edge_labels: bool = False,
    scatter_edges: bool = True,
    color_by: Optional[str] = None,
    return_color_mapping: bool = False,
    radius: float = -0.3,
    **kwargs: Any,
) -> Optional[Dict[Any, Any]]:
    """
    Draw custom arrows on a graph `G` with given positions and styles.

    Args:
        G (networkx.Graph): The graph on which arrows will be drawn.
        pos (dict): A dictionary with nodes as keys and positions as values.
            Positions should be tuples of (x, y) coordinates.
        edge_width (int, optional): Width of the edges. Defaults to 1.
        ax (matplotlib.axes.Axes, optional): Matplotlib axes object to draw the arrows on.
            If None, the current axes will be used. Defaults to None.
        edge_colors (str or list, optional): Color(s) of the edges. If a single string,
            all edges will have the same color. If a list, it should be the same length
            as the number of edges. Defaults to "grey".
        scatter_edges (bool, optional): If True, scatter multiple edges between nodes
            to make them distinguishable. Defaults to False.
        **kwargs (Any): Additional keyword arguments to customize arrow properties. This can include:
            connectionstyle (str, optional): The connection style of the arrows.
                More info: https://matplotlib.org/stable/gallery/userdemo/connectionstyle_demo.html.
                Defaults to "arc3, rad=-0.3".
            arrowstyle (str, optional): The style of the arrow.
                Defaults to "-|>,head_length=0.4,head_width=0.2".
            mutation_scale (float, optional): The scale factor for the arrow head.
                Defaults to 10.0.
            shrinkA (float, optional): Shrink factor at the start of the arrow.
                Defaults to 15.
            shrinkB (float, optional): Shrink factor at the end of the arrow.
                Defaults to 15.

    Returns:
        Optional[Dict[Any, Any]]: Color mapping if return_color_mapping is True, otherwise None.

    Example:
        ```{python}
        import networkx as nx
        import matplotlib.pyplot as plt
        from tvbo.plot.network import draw_custom_arrows, draw_custom_nodes

        G = nx.DiGraph()
        G.add_edges_from([(0, 1, {"type": 1}), (1, 2, {"type": 2}), (0, 2, {"type": 3})])
        pos = {0: (0, 0), 1: (1, 1), 2: (2, 0)}
        fig, ax = plt.subplots(figsize=(2,2))
        draw_custom_nodes(G, pos, ax=ax)
        draw_custom_arrows(G, pos, ax=ax, color_by="type", edge_colors="viridis")
        ax.axis("off");
        ```

    Notes:
        This function uses `FancyArrowPatch` from `matplotlib.patches` to draw arrows.
    """
    edge_counts = count_directed_edges(G)
    if color_by:
        cmap = (
            edge_colors
            if isinstance(edge_colors, str) and edge_colors in colormaps.keys()
            else "viridis"
        )
        colmap = get_edge_color_mapping(G, colormap=cmap, color_by=color_by)
        edge_colors = [
            colmap[data[color_by] if color_by in data else "n.a."]
            for (u, v, data) in G.edges(data=True)
        ]

    if isinstance(edge_colors, str):
        edge_colors = np.repeat(edge_colors, len(G.edges()))

    if "shrinkA" not in kwargs.keys():
        kwargs["shrinkA"] = 15
    if "shrinkB" not in kwargs.keys():
        kwargs["shrinkB"] = 15
    if "arrowstyle" not in kwargs.keys():
        kwargs["arrowstyle"] = "-|>,head_length=0.4,head_width=0.2"
    if "mutation_scale" not in kwargs.keys():
        kwargs["mutation_scale"] = 10.0

    if "connectionstyle" not in kwargs.keys():
        kwargs["connectionstyle"] = f"arc3,rad={radius}"

    if ax is None:
        ax = plt.gca()

    from collections import defaultdict
    # Group edges by unordered node pair
    edge_groups = defaultdict(list)
    for i, (u, v, data) in enumerate(G.edges(data=True)):
        edge_groups[frozenset({u, v})].append((u, v, i, data))

    used_annotation_coords = []
    for node_pair, edges in edge_groups.items():
        u, v = tuple(node_pair)
        both_directions = G.has_edge(u, v) and G.has_edge(v, u)
        num_edges = len(edges)
        rad_vals = np.linspace(-0.3, 0.3, num_edges) if both_directions or num_edges > 1 else [0]

        for (r, (n1, n2, i, d)) in zip(rad_vals, edges):
            if (n1, n2) in G.edges():
                r = abs(r)
            else:
                r = -abs(r)
            point1 = pos[n1]
            point2 = pos[n2]

            local_kwargs = kwargs.copy()
            local_kwargs["connectionstyle"] = f"arc3,rad={r}"

            arrow = FancyArrowPatch(
                point1, point2, color=edge_colors[i], lw=edge_width, **local_kwargs
            )
            ax.add_patch(arrow)

            if edge_labels:
                connector = arrow.get_connectionstyle()
                path = connector.connect(point1, point2)
                p0, p1_, p2 = path.vertices[:3]

                midpoint = compute_bezier_midpoint(p0, p1_, p2)

                x, y = midpoint
                xrange = ax.get_xlim()[1] - ax.get_xlim()[0]
                yrange = ax.get_ylim()[1] - ax.get_ylim()[0]

                min_x_dist = 0.1 * xrange  # 1% of axis width
                min_y_dist = 0.1 * yrange  # 1% of axis height

                attempts = 0
                max_attempts = 20
                while any(abs(x - ux) < min_x_dist and abs(y - uy) < min_y_dist for ux, uy in used_annotation_coords) and attempts < max_attempts:
                    y += min_y_dist
                    attempts += 1

                used_annotation_coords.append((x, y))

                ax.annotate(
                    d["type"],
                    xy=(x, y),
                    xycoords="data",
                    ha="left",
                    va="center",
                    color="k",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        boxstyle="round,pad=0.2",
                        alpha=0.7,
                    ),
                )

    if return_color_mapping:
        return colmap


def get_actual_bounds(ax, axis="x"):
    renderer = ax.figure.canvas.get_renderer()
    xcoords = []
    ycoords = []

    for artist in ax.get_children():
        if not isinstance(artist, matplotlib.text.Text):
            continue  # Only process text elements
        elif artist.get_text() == "":
            continue
        if hasattr(artist, "get_window_extent"):
            try:
                # Get the bounding box in display coordinates
                bbox = artist.get_window_extent(renderer=renderer)
                # print(f"Bbox for '{artist.get_text()}': {bbox}")

                # Transform the bounding box corners to data coordinates
                bbox_data = ax.transData.inverted().transform(
                    [
                        [bbox.x0, bbox.y0],  # Bottom-left
                        [bbox.x1, bbox.y1],  # Top-right
                    ]
                )
                xcoords.extend([bbox_data[0, 0], bbox_data[1, 0]])
                ycoords.extend([bbox_data[0, 1], bbox_data[1, 1]])
            except Exception as e:
                print(f"Error processing '{artist.get_text()}': {e}")

    if axis == "x":
        return np.min(xcoords), np.max(xcoords)
    elif axis == "y":
        return np.min(ycoords), np.max(ycoords)


def draw_custom_nodes(
    G,
    pos,
    labels=None,
    font_size=10,
    ax=None,
    node_colors=None,
    alpha=0.8,
    facecolor=None,
    edgecolor=None,
):
    """
    Custom function to draw nodes as text in a network graph.

    Parameters:
    ----------
    G : networkx.Graph
        The graph on which nodes will be drawn.
    pos : dict
        A dictionary with nodes as keys and positions as values. Positions should be tuples of (x, y) coordinates.
    labels : dict, optional
        A dictionary with node labels. If None, nodes are labeled with node names.
    font_size : int, optional
        The font size of the node labels (default is 10).
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes object to draw the nodes on. If None, the current axes will be used (default is None).
    node_colors : dict or str, optional
        A dictionary specifying the color for each node or a single color for all nodes (default is None).

    Returns:
    -------
    list
        A list of text objects for the node labels.
    """
    x = [v[0] for v in pos.values()]
    y = [v[1] for v in pos.values()]

    if ax is None:
        ax = plt.gca()

    if min(x) != max(x):
        ax.set_xlim([min(x), max(x)])
    if min(y) != max(y):
        ax.set_ylim([min(y), max(y)])

    if labels is None:
        labels = {node: node for node in G.nodes()}

    if node_colors is None:
        node_colors = ["grey" for node in G.nodes()]
    elif isinstance(node_colors, str):
        if node_colors.startswith("#"):
            node_colors = mcolors.to_rgba(node_colors)
        node_colors = [node_colors for node in G.nodes()]
    elif isinstance(node_colors, dict):
        node_colors = [
            (
                mcolors.to_rgba(node_colors[node])
                if isinstance(node_colors[node], str)
                else node_colors[node]
            )
            for node in G.nodes()
        ]

    # TODO: buffer_factor not used, remove?
    buffer_factor = 1.1  # 10% buffer
    texts = {}  # To store the text objects
    bbox_pad = 0.3  # Padding for the bounding box
    for i, (node, position) in enumerate(pos.items()):
        text = labels[node]
        x, y = position
        txt_obj = ax.text(
            x,
            y,
            text,
            bbox=dict(
                facecolor=node_colors[i] if facecolor is None else facecolor,
                edgecolor=node_colors[i] if edgecolor is None else edgecolor,
                alpha=alpha,
                boxstyle=f"round,pad={bbox_pad}",
            ),
            ha="center",
            va="top",
            fontsize=font_size,
        )
        texts[node] = txt_obj
    ax.figure.canvas.draw()
    # Force rendering to ensure bounding boxes are accurate
    bbox_positions = {
        node: txt.get_window_extent(
            renderer=ax.figure.canvas.get_renderer()
        ).transformed(ax.transData.inverted())
        for node, txt in texts.items()
    }

    xmin, xmax = get_actual_bounds(ax, axis="x")
    ymin, ymax = get_actual_bounds(ax, axis="y")
    ax.set_xlim([xmin - bbox_pad, xmax + bbox_pad])
    ax.set_ylim([ymin - bbox_pad, ymax + bbox_pad])

    return texts, bbox_positions

    # return texts


def get_layout(
    G,
    layout_type="spring",
    k_factor=1,
    save_pos=None,
    use_precomputed_pos=None,
):
    """
    Get the layout positions for the nodes in the graph.

    Parameters:
    ----------
    G : networkx.Graph
        The graph for which the layout positions are calculated.
    layout_type : str, optional
        The type of layout algorithm to use ('spring', 'kamada', etc.) (default is "spring").
    k_factor : float, optional
        Scaling factor for the layout algorithm (default is 1).
    save_pos : str, optional
        Path to save the computed positions to a file (default is None).
    use_precomputed_pos : str, optional
        Path to a file with precomputed positions (default is None).

    Returns:
    -------
    dict
        A dictionary with nodes as keys and positions as values.
    """
    if not isinstance(use_precomputed_pos, type(None)):
        with open(use_precomputed_pos, "rb") as f:
            pos = pickle.load(f)
        return pos

    if layout_type == "spring":
        # Get Graph postions
        layout_k = k_factor / np.sqrt(G.number_of_nodes())
        pos = nx.spring_layout(G, k=layout_k, seed=1312)
    elif layout_type == "graphviz":
        pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
    else:
        pos = nx.kamada_kawai_layout(G, scale=k_factor)
    if not isinstance(save_pos, type(None)):
        with open(save_pos, "wb") as f:
            pickle.dump(pos, f)

    return pos


def get_category_from_graph(G, node):
    """
    Retrieve the category of a given node from the graph.

    Parameters:
    ----------
    G : networkx.Graph
        The graph containing the node.
    node : hashable
        The node for which the category is to be retrieved.

    Returns:
    -------
    str
        The category of the node.
    """
    category = G.nodes[node].get("category", "unknown")
    if category == "unknown":
        category = G.nodes[node].get("type", "unknown")

    # TODO: review
    # if category == "unknown":
    #     tvbocls = ontology.search_class(node)
    #     if isinstance(tvbocls, owl.ThingClass):
    #         tvbocls.is_a.first().label.first()

    category = category.replace("parameter", "Parameter")
    return category


def get_categories_from_graph(G):
    """
    Retrieve all unique categories from the graph's nodes.

    Parameters:
    ----------
    G : networkx.Graph
        The graph from which categories are to be retrieved.

    Returns:
    -------
    list
        A sorted list of unique categories from the graph.
    """
    categories = set(nx.get_node_attributes(G, "category").values())
    categories.update(set(nx.get_node_attributes(G, "type").values()))
    categories = sorted(categories)
    categories = [c.replace("parameter", "Parameter") for c in categories]
    return sorted(list(set(categories)))


def set_axis_limits(pos, ax):
    """
    Set the axis limits based on node positions.

    Parameters:
    ----------
    pos : dict
        A dictionary with nodes as keys and positions as values.
    ax : matplotlib.axes.Axes
        The matplotlib axes object to set the limits on.
    """
    x_values, y_values = zip(*pos.values())
    ax.set_xlim(min(x_values) - 0.15, max(x_values) + 0.1)
    ax.set_ylim(min(y_values) - 0.15, max(y_values) + 0.1)


def plot_ontology_graph(
    G,
    node_size_factor=10,
    draw_labels=False,
    edge_width: float = 1.0,
    k_factor: float = 1,
    colors: str = "viridis",
    edge_colors: str = "grey",
    preferred_label: str = "symbol",
    alternative_label: str = "label",
    font_size: int = 12,
    legend: bool = True,
    edge_legend: bool = True,
    ax: Optional[Any] = None,
    colorby: str = "category",
    **kwargs: Any,
):
    """
    Plot the ontology graph using matplotlib.

    Args:
        G (networkx.DiGraph): A directed graph representation of the ontology.
        node_size_factor (int, optional): Factor by which node size is multiplied. Defaults to 10.
        draw_labels (bool, optional): Whether to draw node labels. Defaults to False.
        edge_width (float, optional): Width of the edges. Defaults to 1.0.
        k_factor (float, optional): Scaling factor for the layout algorithm. Defaults to 1.
        colors (str or list, optional): Color map or list of colors for nodes. Defaults to "viridis".
        edge_colors (str or list, optional): Color map or list of colors for edges. Defaults to "grey".
        preferred_label (str, optional): Node attribute to use as the preferred label. Defaults to "symbol".
        alternative_label (str, optional): Node attribute to use as the alternative label. Defaults to "label".
        font_size (int, optional): Font size for labels. Defaults to 12.
        legend (bool, optional): Whether to draw the legend. Defaults to True.
        edge_legend (bool, optional): Whether to include edge types in the legend. Defaults to True.
        ax (matplotlib.axes.Axes, optional): Matplotlib axes object to draw the graph on. If None, a new figure is created. Defaults to None.
        colorby (str, optional): Node attribute to determine the node colors. Defaults to "category".
        **kwargs (Any): Additional keyword arguments for node and edge drawing functions.
    """
    rcParams["font.size"] = font_size

    pos = get_layout(
        G,
        k_factor=k_factor,
    )

    # Create figure
    if isinstance(ax, type(None)):
        fig, ax = plt.subplots(constrained_layout=True, figsize=(10, 10))
        return_fig = True
    else:
        return_fig = False

    x_values, y_values = zip(*pos.values())
    ax.set_xlim(min(x_values) - 0.15, max(x_values) + 0.1)
    ax.set_ylim(min(y_values) - 0.15, max(y_values) + 0.1)
    ax.axis("off")

    # Determine node colors
    color_map = []
    categories = get_categories_from_graph(G)

    if isinstance(colors, str):
        color_list = colormaps[colors](np.linspace(0, 1, len(categories)))
    else:
        color_list = colors

    color_dict = dict(zip(sorted(categories), color_list))
    # print(color_dict)
    for node in G.nodes():
        category = get_category_from_graph(G, node)
        category = category.replace("Time Derivative", "TimeDerivative")

        color_map.append(color_dict.get(category, (0.3, 0.3, 0.3, 1)))

    edge_types = sorted(set(nx.get_edge_attributes(G, "type").values()))
    edge_types.append("is_a")
    edge_types = list(set(sorted(edge_types)))

    if isinstance(edge_colors, str) and edge_colors in colormaps:
        # Determine edge colors based on edge types
        edge_colors_list = colormaps[edge_colors](np.linspace(0, 1, len(edge_types)))
    elif isinstance(edge_colors, str) and edge_colors not in colormaps:
        edge_colors_list = np.repeat(edge_colors, len(edge_types))
    else:
        edge_colors_list = edge_colors

    if len(edge_types) <= len(edge_colors_list):
        edge_color_dict = dict(zip(edge_types, edge_colors_list))
        edge_colors = [
            edge_color_dict[data["type"]] for _, _, data in G.edges(data=True)
        ]

    # else:
    #     edge_legend = False

    labels = {}
    for node in G.nodes():
        # Get the symbol or label for the node
        symb = G.nodes[node].get(
            preferred_label,
            G.nodes[node].get(alternative_label, node.replace("_G2D", "")),
        )

        # If the node has a 'defaultValue' attribute, append it to the symbol
        if preferred_label == "defaultValue" or alternative_label == "defaultValue":
            if "defaultValue" in G.nodes[node]:
                symb = G.nodes[node].get(
                    "symbol",
                    G.nodes[node].get("label", node.replace("_G2D", "")),
                )
                defaultValue = G.nodes[node]["defaultValue"]
                symb = f"{symb} = {defaultValue}"

        labels[node] = f"${symb}$"

    # Draw the graph

    if node_size_factor == 0:
        colorbar = False
        draw_custom_nodes(
            G,
            pos,
            ax=ax,
            labels=labels,
            node_colors=color_map,
            font_size=font_size,
        )
        draw_labels = False  # TODO: draw_labels not used, remove?

        label_shift = 0  # TODO: label_shift in this if is not necessary, remove?
        draw_custom_arrows(
            G,
            pos,
            ax=ax,
            edge_colors=edge_colors,
            edge_width=edge_width,
        )
    else:
        # Set node size based on degree
        color_dict.update({"unknown": np.array([0.3, 0.3, 0.3])})
        node_size = {
            k: v * node_size_factor for k, v in dict(G.degree()).items()
        }  # multiply by a factor for visualization
        label_shift = 0.02 + (0.0002 * node_size_factor)

        node_degree = list(dict(G.degree).values())
        node_degree = np.log1p(node_degree)
        node_degree = (np.array(node_degree) - np.min(node_degree)) / (
            np.max(node_degree) - np.min(node_degree)
        )

        node_degree = {n: d for n, d in zip(dict(G.degree).keys(), node_degree)}

        for node, (x, y) in pos.items():
            category = get_category_from_graph(G, node)

            if colorby == "degree":
                color = mpl.cm.viridis(node_degree[node])
                legend = False
                colorbar = True
            else:
                color = color_dict[category]
                colorbar = False

            plt.scatter(
                x=x,
                y=y,
                s=node_size[node],
                color=color,
                zorder=2,
                axes=ax,
                **kwargs,
            )

            draw_labels = True
            if (
                "Cakan" not in node
                and G.degree()[node] > 50
                or node == "Generic2DOscillator"
            ):
                if draw_labels:
                    labelx = x - label_shift
                    labely = y + label_shift

                    plt.text(
                        labelx,
                        labely,
                        labels[node],
                        color="black",
                        fontweight="bold",
                        ha="center",
                        va="center",
                        fontsize=font_size,
                        bbox=dict(
                            facecolor=(0.7, 0.7, 0.7),
                            edgecolor=(0.5, 0.5, 0.5),
                            alpha=0.8,
                            boxstyle="round,pad=0.3",
                        ),
                    )
            draw_labels = False
        # TODO: review
        # draw_custom_arrows(
        #     G, pos, edge_colors="grey", edge_width=1, connectionstyle="bar"
        # )
        nx.draw_networkx_edges(G, pos, edge_color="grey", width=edge_width, alpha=0.8)

        # TODO: review
        # nx.draw_networkx_nodes(
        #     G,
        #     pos,
        #     with_labels=False,
        #     # labels=labels,
        #     node_color=color_map,
        #     node_size=node_size,
        #     # edge_color=edge_colors,
        #     # width=edge_width,
        #     ax=ax,
        # )

        if draw_labels:
            label_pos = {
                node: (
                    x - label_shift,
                    y + label_shift,
                )
                for node, (x, y) in pos.items()
            }
            for node, (x, y) in label_pos.items():
                plt.text(
                    x,
                    y,
                    labels[node],
                    color="black",
                    fontweight="bold",
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    zorder=3,
                )

    if colorbar:
        plt.colorbar(ax=ax, label="Degree", shrink=0.3, location="left", pad=-0.05)
    if legend:
        # Create legend
        legend_handles = [
            mpatches.Patch(color=color_dict[cat], label=cat)
            for cat in sorted(categories)
        ]
        if edge_legend:
            # Create a second legend for edge types
            legend_handles_edges = [
                mlines.Line2D(
                    [], [], color=edge_color_dict[etype], lw=edge_width, label=etype
                )
                for etype in edge_types
            ]

        # Assuming legend_handles contains handles for nodes
        all_handles = []
        all_labels = []

        # Add node legend with subtitle
        all_handles.append(mpatches.Patch(color="none", label="Node Types"))
        all_labels.append("Node Types")
        for handle in legend_handles:
            all_handles.append(handle)
            all_labels.append(handle.get_label())

        # Add edge legend with subtitle if edge_legend is True
        if edge_legend:
            all_handles.append(mpatches.Patch(color="none", label="Edge Types"))
            all_labels.append("Edge Types")
            for handle in legend_handles_edges:
                all_handles.append(handle)
                all_labels.append(handle.get_label())

        # Now, use plt.legend() once to create the unified legend
        leg = plt.legend(
            all_handles, all_labels, loc="best", handlelength=1, handletextpad=0.5
        )

        # Hide the patches that we added just for the subtitles
        for text in leg.get_texts():
            if text.get_text() == "Node Types" or text.get_text() == "Edge Types":
                text.set_weight(
                    "bold"
                )  # You can set the subtitles to bold for better distinction

    # plt.close()
    if return_fig:
        return fig


def hierarchy_pos(
    G,
    root=None,
    width=1.0,
    vert_gap=0.2,
    hor_gap=1,
    vert_loc=0,
    xcenter=0.5,
    direction="down",
    vert_scatter=0.0,
):
    """
    If the graph is a DAG this will return the positions to plot this in a hierarchical layout.

    G: the graph (must be a DAG)
    root: the root node of current branch
    width: horizontal space allocated for this branch
    vert_gap: gap between levels of hierarchy
    hor_gap: gap between nodes within the same level
    vert_loc: vertical location of root
    xcenter: horizontal location of root
    direction: 'down' for top-down layout, 'up' for bottom-up layout
    vert_scatter: vertical scatter factor for nodes on the same level
    """
    vert_scatter /= 10
    if not nx.is_directed_acyclic_graph(G):
        raise TypeError(
            "cannot use hierarchy_pos on a graph that is not a directed acyclic graph (DAG)"
        )

    def _hierarchy_pos(
        G,
        root,
        width=1.0,
        vert_gap=0.2,
        hor_gap=1,
        vert_loc=0,
        xcenter=0.5,
        pos=None,
        parent=None,
        parsed=[],
    ):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        if direction == "down":
            children = list(G.successors(root))
        else:
            children = list(G.predecessors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children) * hor_gap
            nextx = xcenter - width / 2 + dx / 2
            for i, child in enumerate(children):
                scatter = vert_scatter if i % 2 == 0 else -vert_scatter
                pos = _hierarchy_pos(
                    G,
                    child,
                    width=dx,
                    vert_gap=vert_gap,
                    hor_gap=hor_gap,
                    vert_loc=(
                        vert_loc - vert_gap + scatter
                        if direction == "down"
                        else vert_loc + vert_gap + scatter
                    ),
                    xcenter=nextx,
                    pos=pos,
                    parent=root,
                    parsed=parsed,
                )
                nextx += dx
        return pos

    if root is None:
        if direction == "down":
            root = [n for n, d in G.in_degree() if d == 0][0]
        else:
            root = [n for n, d in G.out_degree() if d == 0][0]

    return _hierarchy_pos(G, root, width, vert_gap, hor_gap, vert_loc, xcenter)


########################
# DEPRECATED FUNCTIONS #
########################


def plot_tvbo_graph(g, ax=None, **kwargs):
    """
    A function to plot a TVBO graph with various customizable parameters.

    Parameters:
    g (networkx.Graph): The graph to be plotted.
    kwargs: Various other optional parameters to customize the plot.

    Returns:
    None
    """
    params = validate_parameters(kwargs)
    relabel_graph(g, params["relabel"])
    set_plt_params()

    labels = get_labels(
        g, params["label_as_symbol"]
    )  # TODO: labels not used, remove or use?
    node_color, cmap = get_node_color_and_cmap(g, params)
    pos = get_layout(g, params["layout"])
    node_size = get_node_size(g, params)

    if isinstance(ax, type(None)):
        ax = get_ax(params["ax"], params["figsize"])
        params["return_fig"] = False

    draw_nodes(g, pos, node_color, cmap, node_size, ax, params)
    draw_edges(g, pos, ax, params)
    if params["draw_edgelabels"]:
        draw_edge_labels(g, pos, ax, params)

    if params["legend"]:
        draw_legend(g, ax, params)

    if params["return_fig"]:
        return plt.figure()  # Adjusted to return the figure object
    else:
        plt.show()


def validate_parameters(params=None):
    """
    A function to validate and set default values for various parameters.

    Parameters:
    params (dict): The parameters passed to the function.

    Returns:
    dict: The validated and updated parameters.
    """
    if params is None:
        params = {}
    default_params = get_default_params()
    for key, value in default_params.items():
        if key not in params:
            params[key] = value

    return params


def relabel_graph(g, relabel):
    """
    A function to relabel the nodes of the graph if a relabel dictionary is provided.

    Parameters:
    g (networkx.Graph): The graph to be relabeled.
    relabel (dict): A dictionary with the relabeling information.

    Returns:
    None
    """
    if isinstance(relabel, dict):
        g = nx.relabel_nodes(
            g, relabel
        )  # TODO: g is not returned, so this method will not actually modify any graph;
        # TODO: either return g or modify it in place ("nx.relabel_nodes(g, relabel, copy=False)")


def set_plt_params(rc_params=None):
    """
    A function to set certain plt parameters.

    Returns:
    None
    """
    params = {"text.usetex": False}
    if not isinstance(rc_params, type(None)):
        params.update(rc_params)

    plt.rcParams.update(params)


def get_labels(g, label_as_symbol):
    """
    Generate labels for the nodes in the graph based on the node names and ontology search results. If the label_as_symbol parameter is True,
    the labels are formatted as LaTeX symbols. Additionally, specific substrings ("_RWW") are removed from the labels.

    Parameters:
    g (networkx.Graph): The graph object containing the nodes for which labels are to be generated.
    label_as_symbol (bool): A flag indicating whether to format the labels as LaTeX symbols.

    Returns:
    dict: A dictionary where keys are node names and values are the corresponding labels.
    """
    labels = dict()
    for node in g.nodes:
        label = str(node)
        searchlist = ontology.onto.search(label=label)
        if len(searchlist) > 0:
            sym = searchlist.first().symbol.first()
            acr = searchlist.first().acronym.first()

            if sym:
                label = sym
            elif not sym and acr:
                label = acr

            if len(label.split("_")) > 2:
                label = "_".join(label.split("_")[:-1])

            if label_as_symbol:
                label = r"${}$".format(label)
        labels[node] = label.replace("_RWW", "")
    return labels


def get_node_color_and_cmap(g, params=None):
    """
    A function to get the node colors and the colormap based on node attributes such as type or degree.

    Parameters:
    g (networkx.Graph): The graph for which the node colors and colormap are to be generated.
    params (dict): A dictionary with various parameters including node_filling and tvb_colors.

    Returns:
    tuple: A tuple with the node colors and the colormap.
    """
    if params is None:
        params = {}
    params = validate_parameters(params)

    handles = list()
    colors = list()

    if isinstance(params["c_order"], list):
        params["colors"] = np.array(params["colors"])[params["c_order"]]

    if params["node_filling"] == "type" or params["node_filling"] == "category":
        filling_key = params["node_filling"]
        groups = set(nx.get_node_attributes(g, filling_key).values())
        mapping = dict(zip(sorted(groups), count()))
        nodes = g.nodes()
        color_mapping = [mapping[g.nodes[n][filling_key]] for n in nodes]
        node_color = color_mapping
        cmap = get_cmap(colors)

        for k, v in mapping.items():
            handles.append(mpatches.Patch(color=params["colors"][v], label=k))
            colors.append(params["colors"][v])

    elif params["node_filling"] == "degree":
        node_degree = list(dict(g.degree).values())
        node_color = node_degree
        cmap = get_continuous_cmap(["#FFFFFF", colors[0]])
        cmap = get_continuous_cmap([mcolors.to_hex(cmap(0.1)), colors[0]])
    else:
        node_color = "white"
        cmap = get_cmap(colors)

    return node_color, cmap


def get_node_size(g, params=None):
    """
    A function to get the node size for the graph, which can be determined by various parameters including
    the degree of the nodes, the length of the labels, or a specified factor.

    Parameters:
    g (networkx.Graph): The graph for which the node size is to be generated.
    pos (dict): The positions of the nodes.
    labels (dict): The labels of the nodes.
    params (dict): A dictionary with various parameters including 'node_size_by' and 'node_size_factor'.

    Returns:
    list or int: A list with the node sizes or a single integer representing a uniform node size for all nodes.
    """
    if params is None:
        params = {}
    params = validate_parameters(params)

    if params["node_size_by"] == "degree":
        d = dict(g.degree)
        node_size = [v * params["node_size_factor"] for v in d.values()]
    # elif params["node_size_by"] == "text":
    #     node_size = [2 * 60 * params["font_size"] / 2 for i in pos]
    elif params["node_size_by"] == "factor":
        node_size = params["node_size_factor"]
    else:
        raise ValueError(
            "Invalid 'node_size_by' parameter. Supported values are: 'degree', 'text', 'factor'"
        )

    return node_size


def get_ax(ax, figsize):
    """
    A function to get the ax for the plot. If ax is None, a new figure and ax are created using the specified figsize.

    Parameters:
    ax (matplotlib.axes._axes.Axes or None): The ax parameter passed to the function. If None, a new ax is created.
    figsize (tuple): The size of the figure, specified as a tuple of width and height.

    Returns:
    matplotlib.axes._axes.Axes: The ax for the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return ax
    else:
        return ax


def draw_nodes(g, pos, node_color, cmap, node_size, ax, params):
    """
    A function to draw the nodes on the graph with specified attributes such as color, size, and using a specified colormap.
    Additional parameters like linewidth and alpha are taken from the params dictionary.

    Parameters:
    g (networkx.Graph): The graph on which the nodes are to be drawn.
    pos (dict): The positions of the nodes.
    node_color: The color of the nodes, specified either as a single color or a list of colors.
    cmap: The colormap for the nodes.
    node_size: The size of the nodes, specified either as a single size or a list of sizes.
    ax (matplotlib.axes._axes.Axes): The ax on which the nodes are to be drawn.
    params (dict): A dictionary with various parameters including 'alpha' and 'node_linewidth'.

    Returns:
    None
    """
    groups = set(nx.get_node_attributes(g, "type").values())
    mapping = dict(zip(sorted(groups), count()))
    nodes = g.nodes()
    color_mapping = [
        mapping[g.nodes[n]["type"]] for n in nodes
    ]  # TODO: color_mapping not used, remove?

    handles = list()
    colors = list()

    for k, v in mapping.items():
        handles.append(mpatches.Patch(color=params["colors"][v], label=k))
        colors.append(params["colors"][v])

    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=nodes,
        node_color=node_color,
        cmap=cmap,
        node_size=node_size,
        alpha=params["alpha"],
        linewidths=params["node_linewidth"],
        ax=ax,
    )


def draw_edges(g, pos, ax, params):
    """
    A function to draw the edges on the graph. Depending on the parameters, it can draw colored edges
    and adjust the width of the edges.

    Parameters:
    g (networkx.Graph): The graph on which the edges are to be drawn.
    pos (dict): The positions of the nodes.
    ax (matplotlib.axes._axes.Axes): The ax on which the edges are to be drawn.
    params (dict): A dictionary with various parameters including 'colored_edges' and 'edge_width'.

    Returns:
    None
    """

    if params["colored_edges"]:
        edge_types = list(set(nx.get_edge_attributes(g, "type").values()))
        edge_colors = plt.cm.viridis(np.linspace(0, 1, len(edge_types)))

        type_cmapping = dict()  # TODO: type_cmapping not used, remove?
        for i, edge_type in enumerate(edge_types):
            edges_of_type = [
                (u, v)
                for (u, v, data) in g.edges(data=True)
                if data["type"] == edge_type
            ]
            nx.draw_networkx_edges(
                g,
                pos,
                edgelist=edges_of_type,
                edge_color=edge_colors[i],
                width=params["edge_width"],
                ax=ax,
                alpha=0.6,
            )

        # Create a legend for edge types and their colors
        edge_type_labels = {edge_type: f"{edge_type}" for edge_type in edge_types}
        ax.legend(
            handles=[
                plt.Line2D([], [], linestyle="-", color=color, lw=params["edge_width"])
                for color in edge_colors
            ],
            labels=edge_type_labels.values(),
            title="Edge Types",
            loc="upper left",
            bbox_to_anchor=(0, 0),
            ncols=2,
        )
    else:
        nx.draw_networkx_edges(g, pos, alpha=0.2, width=params["edge_width"], ax=ax)


def draw_edge_labels(g, pos, ax, params):
    """
    A function to draw the edge labels on the graph. The labels are extracted from the edge attributes
    and can be customized using various parameters in the params dictionary.

    Parameters:
    g (networkx.Graph): The graph on which the edge labels are to be drawn.
    pos (dict): The positions of the nodes.
    ax (matplotlib.axes._axes.Axes): The ax on which the edge labels are to be drawn.
    params (dict): A dictionary with various parameters including 'edge_font_size'.

    Returns:
    None
    """
    if params.get("draw_edgelabels", False):
        nx.draw_networkx_edge_labels(
            g,
            pos,
            edge_labels=nx.get_edge_attributes(g, "type"),
            font_color="grey",
            font_size=params.get("edge_font_size", 12),
            clip_on=False,
            ax=ax,
        )


def draw_legend(g, ax, params):
    """
    A function to draw the legend on the graph. The legend is created based on the node types and
    their respective colors, which are derived from the 'tvb_colors' parameter in the params dictionary.

    Parameters:
    g (networkx.Graph): The graph on which the legend is to be drawn.
    ax (matplotlib.axes._axes.Axes): The ax on which the legend is to be drawn.
    params (dict): A dictionary with various parameters including 'tvb_colors' and 'c_order'.

    Returns:
    None
    """
    groups = set(nx.get_node_attributes(g, "type").values())
    mapping = dict(zip(sorted(groups), count()))
    nodes = g.nodes()
    color_mapping = [mapping[g.nodes[n]["type"]] for n in nodes]

    handles = list()
    colors = list()

    if isinstance(params.get("c_order"), list):
        params["colors"] = np.array(params["colors"])[params["c_order"]]

    for k, v in mapping.items():
        handles.append(mpatches.Patch(color=params["colors"][v], label=k))
        colors.append(params["colors"][v])

    ax.legend(handles=handles, loc=params["legend_loc"], title="Node Types")


########################
# Graph Plotting Model #
########################


def plot_model(
    model,
    k_factor=1,
    edge_cmap="viridis",
    node_cmap="viridis",
    edge_width=1,
    font_size=20,
    node_colors="math_type",
    add_equations_to_labels=False,
    add_parameter_values=False,
    circle_matches=False,
    ax=None,
    legend=True,
    figsize=(10, 6),
    legend_kwargs={},
    edge_kwargs={"shrinkA": 15, "shrinkB": 15},
    node_kwargs={},
    **kwargs,
):
    edge_width = font_size / 20
    edge_kwargs.update(
        {"shrinkA": 15 * (font_size / 20), "shrinkB": 15 * (font_size / 20)}
    )
    if isinstance(model, str):
        model = ontology.get_model(model)

    default_legend_kwargs = dict(handlelength=1, handletextpad=0.5, loc="best", ncol=2)
    default_legend_kwargs.update(legend_kwargs)

    G = graph.model2graph(model)
    # Collect the edges to be removed
    edges_to_remove = [
        (u, v, key)
        for u, v, key, attrs in G.edges(keys=True, data=True)
        if not attrs.get("type", "").startswith("is")
    ]

    # Remove the collected edges
    for u, v, key in edges_to_remove:
        G.remove_edge(u, v, key)

    layout_type = kwargs.pop("layout_type", "spring")
    pos = get_layout(
        G,
        k_factor=k_factor,
        layout_type=layout_type,
    )

    if isinstance(ax, type(None)):
        # Initialize figure
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(model.name, bbox=dict(dict(facecolor="none", edgecolor="none")))
        return_fig = True
    else:
        return_fig = False

    x_values, y_values = zip(*pos.values())

    if layout_type != "graphviz":
        ax.set_xlim(min(x_values) - 0.15, max(x_values) + 0.1)
        ax.set_ylim(min(y_values) - 0.15, max(y_values) + 0.1)

    # Node colors
    cat_dict, categories = get_node_color_mapping(
        G, node_colors, return_categories=True
    )
    cat_keys = sorted(cat_dict.keys())

    labels = {
        n: latex(equations.sub_equation(symbols(ontology.replace_suffix(n)), model))
        for n in G.nodes()
    }
    if add_equations_to_labels:
        eqs = equations.symbolic_model_equations(model)
        # print(eqs)
        for n in labels.keys():
            nlab = n.label.first().replace(ontology.get_model_suffix(model), "")
            if nlab in eqs.keys():
                labels[n] = (
                    labels[n] + " = " + latex(equations.sub_equation(eqs[nlab], model))
                )
    if add_parameter_values:
        param_vals = ontology.get_default_values(model)
        for n in labels.keys():
            nlab = ontology.replace_suffix(n)
            if nlab in param_vals.keys():
                unit = n.unit.first() if n.unit else n.has_unit.first()
                unit = latex(symbols(unit)) if unit else ""
                labels[n] = labels[n] + " = " + str(round(param_vals[nlab], 2)) + unit

    draw_custom_nodes(
        G,
        pos,
        ax=ax,
        labels={n: f"${l}$" for n, l in labels.items()},
        node_colors=[cat_dict[categories[n]] for n in G.nodes],
        **node_kwargs,
        font_size=font_size,
    )

    # Edge colors
    edge_types = sorted(set(nx.get_edge_attributes(G, "type").values()))
    edge_types = list(set(sorted(edge_types)))

    edge_colors_list = colormaps[edge_cmap](np.linspace(0, 1, len(edge_types) + 1))[:-1]

    edge_color_dict = dict(zip(edge_types, edge_colors_list))
    edge_colors = [edge_color_dict[data["type"]] for _, _, data in G.edges(data=True)]
    draw_custom_arrows(
        G,
        pos,
        ax=ax,
        edge_colors=edge_colors,
        edge_width=edge_width,
        alpha=0.7,
        **edge_kwargs,
    )

    cat_labels = [
        cat.label.first() if not isinstance(cat, str) else cat for cat in cat_keys
    ]
    legend_handles = [
        mpatches.Patch(color=cat_dict[cat], label=label)
        for cat, label in zip(cat_keys, cat_labels)
    ]
    # Create a second legend for edge types
    legend_handles_edges = [
        mlines.Line2D([], [], color=edge_color_dict[etype], lw=edge_width, label=etype)
        for etype in edge_types
    ]

    # Assuming legend_handles contains handles for nodes
    all_handles = []
    all_labels = []

    # Add node legend with subtitle
    all_handles.append(mpatches.Patch(color="none", label="Node Types"))
    all_labels.append("Node Types")
    for handle in legend_handles:
        all_handles.append(handle)
        all_labels.append(handle.get_label())

    if default_legend_kwargs.get("ncol", 2) == 2:
        while len(all_handles) < len(edge_types) + 1:
            all_handles.append(mpatches.Patch(color="none", label=""))
        while len(all_labels) < len(edge_types) + 1:
            all_labels.append("")

    # Add edge legend with subtitle if edge_legend is True
    all_handles.append(mpatches.Patch(color="none", label="Edge Types"))
    all_labels.append("Edge Types")
    for handle in legend_handles_edges:
        all_handles.append(handle)
        all_labels.append(handle.get_label())

    # Now, use plt.legend() once to create the unified legend
    if legend:
        leg = ax.legend(
            all_handles,
            all_labels,
            # ncols=2,
            **default_legend_kwargs,
        )

        # Hide the patches that we added just for the subtitles
        for text in leg.get_texts():
            if text.get_text() == "Node Types" or text.get_text() == "Edge Types":
                text.set_weight(
                    "bold"
                )  # You can set the subtitles to bold for better distinction

    if return_fig:
        ax.axis('off')
        plt.tight_layout()
        plt.close()
        return fig


def plot_hierarchy(cls, hierarchy_type="ancestors", ax=None, **kwargs):
    if hierarchy_type.lower() == "ancestors":
        subset = cls.ancestors()
        G = graph.subset2graph(subset)
        direction = "down"
    elif hierarchy_type.lower() in ["parents", "is_a", "isa", "superclasses"]:
        subset = cls.is_a
        G = graph.subset2graph(subset)
        for v in subset:
            G.add_edge(cls, v, type="is_a")
        direction = "down"
    elif hierarchy_type.lower() in ["children", "subclasses"]:
        subset = set(cls.subclasses())
        G = graph.subset2graph(subset)
        for v in subset:
            G.add_edge(v, cls, type="is_a")
        direction = "up"
    elif hierarchy_type.lower() == "descendants":
        subset = cls.descendants()
        G = graph.subset2graph(subset)
        direction = "up"
    else:
        print("Error")

    # Create a new graph to hold only 'is_a' relationships
    G = graph.hierarchy_graph(G)

    # Define the position for each node in a hierarchical layout
    pos = hierarchy_pos(G, root=cls, direction=direction, **kwargs)
    # pos = nx.shell_layout(G)
    G = G.subgraph(pos.keys())

    # Draw the graph
    if isinstance(ax, type(None)):
        fig, ax = plt.subplots(figsize=(16, 9))
        return_fig = True
    else:
        return_fig = False
    # draw_custom_nodes(G, pos, ax=ax)
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels={
            node: (
                f"${latex(symbols(node.symbol.first()))}$"
                if hasattr(node, "symbol") and node.symbol.first()
                else node
            )
            for node in G.nodes()
        },
        node_size=2000,
        node_color="lightblue",
        font_size=10,
        font_weight="bold",
        arrowsize=10,
        edge_color="grey",
        ax=ax,
    )

    plt.title("Ontology Class Hierarchy")

    plt.close()
    if return_fig:
        return fig


def plot_multidigraph(G, figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Plots a MultiDiGraph with curved edges to visualize multiple edges between two nodes.

    Args:
        G (networkx.MultiDiGraph): The MultiDiGraph to be plotted.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 8).

    Returns:
        None
    """
    pos = nx.kamada_kawai_layout(G)
    fig, ax = plt.subplots(figsize=figsize)

    draw_custom_arrows(G, pos, scatter_edges=True, ax=ax)

    # nx.draw_networkx_nodes(G, pos, node_size=700, ax=ax)
    draw_custom_nodes(G, pos, ax=ax)
    ax.axis("off")
    plt.title("MultiDiGraph with Curved Edges")
    plt.show()


def create_colormap_legend(
    edge_colmap=None,
    node_colmap=None,
    ax=None,
    fontsize=12,
    title_fontsize=14,
    edge_cols=8,
    node_cols=8,  # TODO: not used, remove?
):
    """
    Create a space-efficient legend for given edge and node colormap dictionaries.

    Args:
    edge_colmap (dict): A dictionary where keys are labels and values are RGBA tuples for edges.
    node_colmap (dict): A dictionary where keys are labels and values are RGBA tuples for nodes.
    ax (matplotlib.axes.Axes, optional): Matplotlib axes object to draw the legend on. If None, a new figure is created.
    fontsize (int, optional): Font size for the legend labels.
    title_fontsize (int, optional): Font size for the legend titles.
    """

    def add_legend(ax, patches, title, loc, bbox_y):
        legend = ax.legend(
            handles=patches,
            loc=loc,
            frameon=False,
            ncol=edge_cols,
            handleheight=1,
            handlelength=1,
            mode="expand",
            borderaxespad=0,
            title=title,
            fontsize=fontsize,
            bbox_to_anchor=(0, bbox_y, 1, 1),
            bbox_transform=ax.transAxes,
        )
        plt.setp(legend.get_title(), fontsize=title_fontsize)
        ax.add_artist(legend)
        return legend

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 9))
        return_fig = True
    else:
        fig = ax.figure
        return_fig = False

    inflate = 0.2 if edge_colmap and node_colmap else 0.1
    inflate *= 1 + edge_cols / 8

    if edge_colmap:
        edge_patches = [
            Patch(color=value, label=key) for key, value in edge_colmap.items()
        ]
        # TODO: edge_legend not used
        edge_legend = add_legend(
            ax,
            edge_patches,
            "Edge Type",
            "upper center",
            bbox_y=-0.8,
        )

    if node_colmap:
        node_patches = [
            Patch(color=value, label=key) for key, value in node_colmap.items()
        ]
        add_legend(
            ax,
            node_patches,
            "Node Type",
            "upper center",
            bbox_y=-0.95 if edge_colmap else -0.8,
        )
    y_min, y_max = ax.get_ylim()

    ax.set_ylim(y_min - (y_max - y_min) * inflate, y_max)
    if return_fig:
        return fig, ax


def get_node_color_mapping(
    G, node_colors="math_type", colors="tvb", return_categories=False
):
    """
    Generate a color mapping for nodes in graph G based on their specified attribute.

    Parameters:
    ----------
    G : networkx.Graph
        The graph containing the nodes with the specified attribute.
    node_colors : str, optional
        The node attribute to color by (default is 'math_type').
    colors : str, optional
        The color scheme to use (default is 'tvb').

    Returns:
    -------
    node_color_dict : dict
        A dictionary mapping node attributes to colors.
    categories : dict
        A dictionary mapping nodes to their categories.
    """
    categories = {}
    cat_dict = {}

    if node_colors == "math_type":
        categories = {
            n: (
                "Parameter"
                if ontology.onto.Parameter in n.is_a
                else (
                    "StateVariable"
                    if ontology.onto.StateVariable in n.is_a
                    else (
                        "Function"
                        if ontology.onto.Function in n.is_a
                        else (
                            "Constant"
                            if ontology.onto.Constant in n.is_a
                            else (
                                "TimeDerivative"
                                if ontology.onto.TimeDerivative in n.is_a
                                else "other"
                            )
                        )
                    )
                )
            )
            for n in G.nodes
        }
    elif node_colors == "bnm_component":
        categories = {
            n: (
                "LocalDynamics"
                if ontology.onto.LocalNeuralDynamics in n.ancestors()
                and not (
                    ontology.onto.GlobalConnectivity in n.ancestors()
                    or ontology.onto.LocalConnectivity in n.ancestors()
                )
                else (
                    "GlobalCoupling"
                    if ontology.onto.GlobalConnectivity in n.ancestors()
                    else (
                        "LocalCoupling"
                        if ontology.onto.LocalConnectivity in n.ancestors()
                        else "other"
                    )
                )
            )
            for n in G.nodes
        }
    elif node_colors == "excinh":
        categories = {
            n: (
                "excitatory"
                if (
                    n.definition.first()
                    and "excit" in n.definition.first().lower()
                    and not "inhib" in n.definition.first().lower()
                )
                or len(
                    ontology.intersection(
                        n.is_a,
                        list(ontology.onto.Excitation.subclasses())
                        + [ontology.onto.Excitation],
                    )
                )
                > 0
                else (
                    "inhibitory"
                    if (
                        n.definition.first()
                        and "inhib" in n.definition.first().lower()
                        and not "excit" in n.definition.first().lower()
                    )
                    or len(
                        ontology.intersection(
                            n.is_a,
                            list(ontology.onto.Inhibition.subclasses())
                            + [ontology.onto.Inhibition],
                        )
                    )
                    > 0
                    else "other"
                )
            )
            for n in G.nodes
        }
        cat_dict = dict(excitatory="#BE4E30", inhibitory="#4EA8E5", other="#657684")

    elif isinstance(node_colors, str):
        categories = {
            n: (
                node_colors
                if (
                    n.definition.first()
                    and node_colors.lower() in n.definition.first().lower()
                )
                or node_colors.lower() in n.label.first().lower()
                else "other"
            )
            for n in G.nodes
        }
        cat_keys = [node_colors, "other"]
        cat_dict = dict(zip(cat_keys, ["#30be80", "#dcdedd"]))
    else:
        cat_dict = node_colors
        categories = {n: cat_dict.get(n) for n in G.nodes()}

    if not cat_dict:
        cat_keys = sorted(set(categories.values()))
        if colors == "tvb":
            color_list = tvb_colors
        elif isinstance(colors, str) and colors in colormaps:
            color_list = colormaps[colors](np.linspace(0, 1, len(cat_keys)))
        cat_dict = dict(zip(cat_keys, color_list))

    if "other" in categories.values():
        cat_dict.update({"other": "#dcdedd"})
    if return_categories:
        return cat_dict, categories
    else:
        return {n: cat_dict[categories[n]] for n in G.nodes()}


########
def adjust_arrow_end(start, end, bbox_end):
    x_start, y_start = start
    x_end, y_end = end

    x_min_end, y_min_end, width_end, height_end = bbox_end.bounds
    x_max_end, y_max_end = x_min_end + width_end, y_min_end + height_end

    # x_min_end, y_min_end = bbox_end[0]
    # x_max_end, y_max_end = bbox_end[1]

    # Determine which edge of the bounding box to intersect with
    if y_start > y_end:  # Arrow goes downward; intersect with top edge
        y_bbox = y_max_end
    else:  # Arrow goes upward; intersect with bottom edge
        y_bbox = y_min_end

    # Calculate the intersection point
    if x_start == x_end:  # Vertical line case
        x_intersect = x_start
    else:
        slope = (y_end - y_start) / (x_end - x_start)
        x_intersect = (y_bbox - y_start) / slope + x_start

    # Clip x_intersect to within the bounding box's horizontal bounds
    x_intersect = max(x_min_end, min(x_intersect, x_max_end))

    return (x_intersect, y_bbox)


def plot_edge(edge, pos, bbox_positions_data, ax, **kwargs):
    start, end = pos[edge[0]], pos[edge[1]]
    xstart, ystart = start
    xend, yend = end

    # bbox_start = bbox_positions_data[edge[0]].bounds
    bbox_end = bbox_positions_data[edge[1]]

    # x_min_start, y_min_start = bbox_start[0]
    # x_max_start, y_max_start = bbox_start[1]

    # x_min_end, y_min_end = bbox_end[0]
    # x_max_end, y_max_end = bbox_end[1]

    start = pos[edge[0]]
    end = pos[edge[1]]
    adjusted_end = adjust_arrow_end(start, end, bbox_end)

    # Plot the edge with adjusted endpoint
    arrow = FancyArrowPatch(
        start,
        adjusted_end,
        **kwargs,
    )
    ax.add_patch(arrow)
