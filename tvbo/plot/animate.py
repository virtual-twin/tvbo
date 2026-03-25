#
# Module: animate.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""Animation utilities for SimulationResult."""

import matplotlib.pyplot as plt
import numpy as np


def _graph_from_experiment(result):
    """Extract graph layout dict from experiment metadata on *result*.

    Returns ``{'positions': ndarray(N,2), 'adjacency': ndarray(N,N)}`` or None.
    """
    exp_result = getattr(result, '_source', None)
    if exp_result is None:
        return None
    experiment = getattr(exp_result, 'source', None)
    if experiment is None:
        return None
    network = getattr(experiment, 'network', None)
    if network is None:
        return None

    import networkx as nx
    G = network.graph
    from tvbo.plot.network import _resolve_positions
    pos_dict = _resolve_positions(G, "spring", network=network)

    node_ids = sorted(G.nodes)
    positions = np.array([pos_dict[n][:2] for n in node_ids])

    W = getattr(network, 'weights_matrix', None)
    if W is None:
        W = np.zeros((len(node_ids), len(node_ids)))

    labels = [G.nodes[n].get('label', f'node_{n}') for n in node_ids]
    return {'positions': positions, 'adjacency': np.asarray(W), 'labels': labels}


def _extract_node_timeseries(data):
    """Extract a list of ``(var_name, vals[time, node], time)`` from data.

    Handles data that may or may not still have a *variable* dimension
    (e.g. after ``.sel(variable='V')``).
    """
    dims = data.dims if hasattr(data, 'dims') else ()
    time = data.coords['time'].values if 'time' in data.coords else np.arange(data.shape[0])

    if 'variable' in dims:
        var_names = list(np.atleast_1d(data.coords['variable'].values))
        slices = []
        for vn in var_names:
            arr = np.asarray(data.sel(variable=vn))
            # After sel: (time,) or (time, node) or (time, node, mode)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            elif arr.ndim >= 3:
                arr = arr[..., 0]   # drop mode dim
            slices.append((str(vn), arr, time))
        return slices

    # Variable dim already selected away
    vn = str(data.coords['variable'].values) if 'variable' in data.coords else 'state'
    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    elif arr.ndim >= 3:
        arr = arr[..., 0]
    return [(vn, arr, time)]


def animate_network(result, state=None, interval=50, cmap='viridis',
                    node_size=120, figsize=None, format=None):
    """Animate time-series on a graph layout (nodes colored by state value).

    Supports the ``.sel(variable='V').animate()`` pattern: if the variable
    dimension is already selected, animates that variable.  With no selection
    all variables are shown (one row per variable).

    Graph data is resolved in order:
    1. ``result.graph`` / ``result._extras['graph']`` (explicitly attached)
    2. Experiment metadata via ``result._source.source.network``

    Parameters
    ----------
    result : SimulationResult
        Simulation result, ideally linked to an experiment with a network.
    state : int, str or None
        *Deprecated* — prefer ``.sel(variable=...)``.  If given, selects a
        single variable by index or name.
    interval : int
        Milliseconds between frames.
    cmap : str
        Matplotlib colormap.
    node_size : int
        Scatter point size.
    figsize : tuple, optional

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    from matplotlib.animation import FuncAnimation

    graph = getattr(result, 'graph', None)
    if graph is None:
        extras = getattr(result, '_extras', {})
        graph = extras.get('graph')
    if graph is None:
        graph = _graph_from_experiment(result)
    if graph is None:
        raise ValueError(
            "No network metadata available for graph animation. "
            "Use result.animate(type='timeseries') instead."
        )

    data = result.data

    # Legacy state= argument: reduce to single-variable data
    if state is not None:
        if isinstance(state, str):
            data = data.sel(variable=state)
        else:
            var_names = list(np.atleast_1d(data.coords['variable'].values))
            data = data.sel(variable=var_names[state])

    slices = _extract_node_timeseries(data)
    n_vars = len(slices)

    pos = graph['positions']
    adj = graph['adjacency']
    x, y = pos[:, 0], pos[:, 1]
    labels = graph.get('labels')

    if figsize is None:
        figsize = (10, 3.5 * n_vars)

    fig, axes = plt.subplots(
        n_vars, 2, figsize=figsize, squeeze=False,
        gridspec_kw={'width_ratios': [1, 1.2]},
    )

    all_artists = []  # (scatter, lines, avg_line) per row

    for row, (vn, vals, time) in enumerate(slices):
        ax_graph, ax_ts = axes[row]
        vmin, vmax = float(vals.min()), float(vals.max())
        n_nodes = vals.shape[1]

        # Draw edges
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i, j] != 0:
                    ax_graph.plot(
                        [x[i], x[j]], [y[i], y[j]],
                        color='lightgray', linewidth=0.5, zorder=0,
                    )

        sc = ax_graph.scatter(
            x, y, c=vals[0], cmap=cmap, s=node_size,
            vmin=vmin, vmax=vmax, zorder=2,
            edgecolors='k', linewidths=0.5,
        )
        if labels:
            for k, lbl in enumerate(labels):
                ax_graph.annotate(
                    lbl, (x[k], y[k]), fontsize=7, ha='center',
                    va='bottom', xytext=(0, 5),
                    textcoords='offset points',
                )
        ax_graph.set_aspect('equal')
        ax_graph.set_title(f'{vn}  t = {time[0]:.2f}')
        ax_graph.axis('off')
        fig.colorbar(sc, ax=ax_graph, shrink=0.7)

        # Time-series panel — raw per-node traces
        cm = plt.get_cmap(cmap)
        node_norm = plt.Normalize(vmin=0, vmax=max(n_nodes - 1, 1))
        lines = []
        for i in range(n_nodes):
            lbl_i = labels[i] if labels else None
            ln, = ax_ts.plot(
                [], [], color=cm(node_norm(i)),
                linewidth=0.8, alpha=0.7, label=lbl_i,
            )
            lines.append(ln)
        ax_ts.set_xlim(time[0], time[-1])
        margin = 0.05 * abs(vmax - vmin) if vmax != vmin else 0.1
        ax_ts.set_ylim(vmin - margin, vmax + margin)
        ax_ts.set_xlabel('time')
        ax_ts.set_ylabel(vn)
        ax_ts.legend(loc='upper right', fontsize='x-small')

        all_artists.append((sc, lines, vals, time, ax_graph, vn))

    fig.tight_layout()

    # Shared frame list from the first variable's time axis
    ref_time = slices[0][2]
    step = max(1, len(ref_time) // 200)
    frames = list(range(0, len(ref_time), step))

    def update(frame):
        arts = []
        for sc, lines, vals, time, ax_g, vn in all_artists:
            sc.set_array(vals[frame])
            ax_g.set_title(f'{vn}  t = {time[frame]:.2f}')
            for i, ln in enumerate(lines):
                ln.set_data(time[:frame + 1], vals[:frame + 1, i])
            arts.extend([sc] + lines)
        return arts

    ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    plt.close(fig)
    return ani


def animate_timeseries(result, state=None, interval=50, cmap='viridis',
                       figsize=None):
    """Animate evolving time-series traces (no graph layout needed).

    Supports ``.sel(variable='V').animate(type='timeseries')``.
    With no selection, all variables are shown (one panel per variable).

    Parameters
    ----------
    result : SimulationResult
    state : int, str or None
        If given, selects a single variable by index or name.
    interval : int
        Milliseconds between frames.
    cmap : str
        Matplotlib colormap.
    figsize : tuple, optional

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    from matplotlib.animation import FuncAnimation

    data = result.data

    if state is not None:
        if isinstance(state, str):
            data = data.sel(variable=state)
        else:
            var_names = list(np.atleast_1d(data.coords['variable'].values))
            data = data.sel(variable=var_names[state])

    slices = _extract_node_timeseries(data)
    n_vars = len(slices)

    if figsize is None:
        figsize = (10, 3 * n_vars)

    fig, axes = plt.subplots(n_vars, 1, figsize=figsize, squeeze=False)

    all_artists = []
    for row, (vn, vals, time) in enumerate(slices):
        ax = axes[row, 0]
        vmin, vmax = float(vals.min()), float(vals.max())
        n_traces = vals.shape[1]
        cm = plt.get_cmap(cmap)
        norm = plt.Normalize(vmin=0, vmax=max(n_traces - 1, 1))
        lines = []
        for i in range(n_traces):
            ln, = ax.plot([], [], color=cm(norm(i)), linewidth=0.8, alpha=0.7)
            lines.append(ln)
        ax.set_xlim(time[0], time[-1])
        margin = 0.05 * abs(vmax - vmin) if vmax != vmin else 0.1
        ax.set_ylim(vmin - margin, vmax + margin)
        ax.set_xlabel('time')
        ax.set_ylabel(vn)
        all_artists.append((lines, vals, time, ax))

    fig.tight_layout()

    ref_time = slices[0][2]
    step = max(1, len(ref_time) // 200)
    frames = list(range(0, len(ref_time), step))

    def update(frame):
        arts = []
        for lines, vals, time, ax in all_artists:
            ax.set_title(f't = {time[frame]:.2f}')
            for i, ln in enumerate(lines):
                ln.set_data(time[:frame + 1], vals[:frame + 1, i])
            arts.extend(lines)
        return arts

    ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    plt.close(fig)
    return ani


def animate_phase(result, x_var=None, y_var=None, region=0, mode=0,
                  interval=50, trail=200, figsize=(6, 5)):
    """Animate phase-space trajectory with a trailing tail.

    Parameters
    ----------
    result : SimulationResult
    x_var, y_var : str, optional
    region, mode : int
    interval : int
    trail : int
        Number of trailing points.
    figsize : tuple

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    from matplotlib.animation import FuncAnimation
    from tvbo.plot.phase import _extract_2d

    time, x, y, xlabel, ylabel = _extract_2d(result, x_var, y_var, region, mode)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(x.min() - 0.05 * np.ptp(x), x.max() + 0.05 * np.ptp(x))
    ax.set_ylim(y.min() - 0.05 * np.ptp(y), y.max() + 0.05 * np.ptp(y))

    units = getattr(result, '_units', {})
    xu, yu = units.get(xlabel, ''), units.get(ylabel, '')
    ax.set_xlabel(f'{xlabel} [{xu}]' if xu else xlabel)
    ax.set_ylabel(f'{ylabel} [{yu}]' if yu else ylabel)

    line, = ax.plot([], [], 'b-', linewidth=0.8, alpha=0.7)
    point, = ax.plot([], [], 'ro', markersize=6)

    step = max(1, len(time) // 400)
    frames = list(range(0, len(time), step))

    def update(frame):
        lo = max(0, frame - trail)
        line.set_data(x[lo:frame + 1], y[lo:frame + 1])
        point.set_data([x[frame]], [y[frame]])
        ax.set_title(f't = {time[frame]:.1f} ms')
        return line, point

    ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    plt.close(fig)
    return ani
