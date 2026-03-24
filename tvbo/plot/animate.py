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


def animate_network(result, state=0, interval=50, cmap='viridis',
                    node_size=120, figsize=(10, 4)):
    """Animate time-series on a graph layout (nodes colored by state value).

    Parameters
    ----------
    result : SimulationResult
        Must have graph data attached (``result.graph``).
    state : int or str
        State variable index or name.
    interval : int
        Milliseconds between frames.
    cmap : str
        Matplotlib colormap.
    node_size : int
        Scatter point size.
    figsize : tuple

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
        raise ValueError(
            "No graph data attached. Run with format='networkdynamics' "
            "to get graph positions."
        )

    data = result.data
    var_names = list(np.atleast_1d(data.coords['variable'].values)) if 'variable' in data.coords else []
    time = data.coords['time'].values if 'time' in data.coords else np.arange(data.shape[0])

    if isinstance(state, str):
        state = var_names.index(state)

    # Extract (time, nodes) for selected state
    arr = np.asarray(data)
    if arr.ndim == 4:
        vals = arr[:, state, :, 0]
    elif arr.ndim == 3:
        vals = arr[:, state, :]
    else:
        raise ValueError("Need at least (time, variable, node) dimensions for animation")

    pos = graph['positions']
    adj = graph['adjacency']
    vmin, vmax = float(vals.min()), float(vals.max())
    x, y = pos[:, 0], pos[:, 1]

    fig, (ax_graph, ax_ts) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, 1.2]},
    )

    # Draw edges
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j] != 0:
                ax_graph.plot([x[i], x[j]], [y[i], y[j]],
                              color='lightgray', linewidth=0.5, zorder=0)

    sc = ax_graph.scatter(x, y, c=vals[0], cmap=cmap, s=node_size,
                          vmin=vmin, vmax=vmax, zorder=2,
                          edgecolors='k', linewidths=0.5)
    ax_graph.set_aspect('equal')
    ax_graph.set_title(f't = {time[0]:.2f}')
    ax_graph.axis('off')
    fig.colorbar(sc, ax=ax_graph, shrink=0.7)

    # Time-series panel
    n_nodes = vals.shape[1]
    cm = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin=0, vmax=n_nodes - 1)
    lines = []
    for i in range(n_nodes):
        ln, = ax_ts.plot([], [], color=cm(norm(i)), linewidth=0.5, alpha=0.6)
        lines.append(ln)
    avg_ln, = ax_ts.plot([], [], color='k', linewidth=1.5, label='mean')
    ax_ts.set_xlim(time[0], time[-1])
    ax_ts.set_ylim(vmin - 0.05 * abs(vmax - vmin), vmax + 0.05 * abs(vmax - vmin))
    sv_name = var_names[state] if state < len(var_names) else f'state {state}'
    ax_ts.set_xlabel('time')
    ax_ts.set_ylabel(sv_name)
    ax_ts.legend(loc='upper right', fontsize='small')
    fig.tight_layout()

    step = max(1, len(time) // 200)
    frames = list(range(0, len(time), step))

    def update(frame):
        sc.set_array(vals[frame])
        ax_graph.set_title(f't = {time[frame]:.2f}')
        for i, ln in enumerate(lines):
            ln.set_data(time[:frame + 1], vals[:frame + 1, i])
        avg_ln.set_data(time[:frame + 1], vals[:frame + 1].mean(axis=1))
        return [sc] + lines + [avg_ln]

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
