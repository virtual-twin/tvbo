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
    exp_result = getattr(result, "_source", None)
    if exp_result is None:
        return None
    experiment = getattr(exp_result, "source", None)
    if experiment is None:
        return None
    network = getattr(experiment, "network", None)
    if network is None:
        return None

    G = network.graph
    from tvbo.plot.network import _resolve_positions

    pos_dict = _resolve_positions(G, "spring", network=network)

    node_ids = sorted(G.nodes)
    positions = np.array([pos_dict[n][:2] for n in node_ids])

    W = getattr(network, "weights_matrix", None)
    if W is None:
        W = np.zeros((len(node_ids), len(node_ids)))

    labels = [G.nodes[n].get("label", f"node_{n}") for n in node_ids]
    return {"positions": positions, "adjacency": np.asarray(W), "labels": labels}


def _extract_node_timeseries(data, max_points=200):
    """Extract a list of ``(var_name, vals[time, node], time)`` from data.

    Handles data that may or may not still have a *variable* dimension
    (e.g. after ``.sel(variable='V')``).  When the time axis has more than
    *max_points* samples the arrays are downsampled so that animations
    stay lightweight.
    """
    dims = data.dims if hasattr(data, "dims") else ()
    time = data.coords["time"].values if "time" in data.coords else np.arange(data.shape[0])

    # Compute downsample indices once (shared across variables)
    n_t = len(time)
    if n_t > max_points:
        idx = np.linspace(0, n_t - 1, max_points, dtype=int)
        time = time[idx]
    else:
        idx = None

    if "variable" in dims:
        var_names = list(np.atleast_1d(data.coords["variable"].values))
        slices = []
        for vn in var_names:
            arr = np.asarray(data.sel(variable=vn))
            # After sel: (time,) or (time, node) or (time, node, mode)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis]
            elif arr.ndim >= 3:
                arr = arr[..., 0]  # drop mode dim
            if idx is not None:
                arr = arr[idx]
            slices.append((str(vn), arr, time))
        return slices

    # Variable dim already selected away
    vn = str(data.coords["variable"].values) if "variable" in data.coords else "state"
    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    elif arr.ndim >= 3:
        arr = arr[..., 0]
    if idx is not None:
        arr = arr[idx]
    return [(vn, arr, time)]


def animate_network(result, state=None, interval=50, cmap="viridis", node_size=120, figsize=None, format=None):
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

    graph = getattr(result, "graph", None)
    if graph is None:
        extras = getattr(result, "_extras", {})
        graph = extras.get("graph")
    if graph is None:
        graph = _graph_from_experiment(result)
    if graph is None:
        raise ValueError("No network metadata available for graph animation. Use result.animate(type='timeseries') instead.")

    data = result.data

    # Legacy state= argument: reduce to single-variable data
    if state is not None:
        if isinstance(state, str):
            data = data.sel(variable=state)
        else:
            var_names = list(np.atleast_1d(data.coords["variable"].values))
            data = data.sel(variable=var_names[state])

    slices = _extract_node_timeseries(data)
    n_vars = len(slices)

    pos = graph["positions"]
    adj = graph["adjacency"]
    x, y = pos[:, 0], pos[:, 1]
    labels = graph.get("labels")

    if figsize is None:
        figsize = (10, 3.5 * n_vars)

    fig, axes = plt.subplots(
        n_vars,
        2,
        figsize=figsize,
        squeeze=False,
        gridspec_kw={"width_ratios": [1, 1.2]},
    )
    fig.dpi = 72  # keep PNGs small for to_jshtml() embed_limit

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
                        [x[i], x[j]],
                        [y[i], y[j]],
                        color="lightgray",
                        linewidth=0.5,
                        zorder=0,
                    )

        sc = ax_graph.scatter(
            x,
            y,
            c=vals[0],
            cmap=cmap,
            s=node_size,
            vmin=vmin,
            vmax=vmax,
            zorder=2,
            edgecolors="k",
            linewidths=0.5,
        )
        if labels:
            for k, lbl in enumerate(labels):
                ax_graph.annotate(
                    lbl,
                    (x[k], y[k]),
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    xytext=(0, 5),
                    textcoords="offset points",
                )
        ax_graph.set_aspect("equal")
        ax_graph.set_title(f"{vn}  t = {time[0]:.2f}")
        ax_graph.axis("off")
        fig.colorbar(sc, ax=ax_graph, shrink=0.7)

        # Time-series panel — raw per-node traces (use default cycler for node identity)
        lines = []
        for i in range(n_nodes):
            lbl_i = labels[i] if labels else None
            (ln,) = ax_ts.plot(
                [],
                [],
                linewidth=0.8,
                alpha=0.7,
                label=lbl_i,
            )
            lines.append(ln)
        ax_ts.set_xlim(time[0], time[-1])
        margin = 0.05 * abs(vmax - vmin) if vmax != vmin else 0.1
        ax_ts.set_ylim(vmin - margin, vmax + margin)
        ax_ts.set_xlabel("time")
        ax_ts.set_ylabel(vn)
        ax_ts.legend(loc="upper right", fontsize="x-small")

        all_artists.append((sc, lines, vals, time, ax_graph, vn))

    fig.tight_layout()

    # Data is already downsampled by _extract_node_timeseries;
    # animate sequentially over all points.
    n_frames = len(slices[0][2])

    def update(i):
        """Recolor nodes and extend per-node traces for frame `i`.

        Args:
            i: Frame index into the downsampled time axis.

        Returns:
            The updated scatter and line artists to redraw.
        """
        arts = []
        for sc, lines, vals, time, ax_g, vn in all_artists:
            sc.set_array(vals[i])
            ax_g.set_title(f"{vn}  t = {time[i]:.2f}")
            for j, ln in enumerate(lines):
                ln.set_data(time[: i + 1], vals[: i + 1, j])
            arts.extend([sc] + lines)
        return arts

    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
    plt.close(fig)
    return ani


def animate_timeseries(result, state=None, interval=50, cmap=None, figsize=None):
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
            var_names = list(np.atleast_1d(data.coords["variable"].values))
            data = data.sel(variable=var_names[state])

    slices = _extract_node_timeseries(data)
    n_vars = len(slices)

    if figsize is None:
        figsize = (10, 3 * n_vars)

    fig, axes = plt.subplots(n_vars, 1, figsize=figsize, squeeze=False)
    fig.dpi = 72  # keep PNGs small for to_jshtml() embed_limit

    all_artists = []
    for row, (vn, vals, time) in enumerate(slices):
        ax = axes[row, 0]
        vmin, vmax = float(vals.min()), float(vals.max())
        n_traces = vals.shape[1]
        if cmap is not None:
            cm = plt.get_cmap(cmap)
            norm = plt.Normalize(vmin=0, vmax=max(n_traces - 1, 1))
            colors = [cm(norm(i)) for i in range(n_traces)]
        else:
            colors = [None] * n_traces
        lines = []
        for i in range(n_traces):
            kw = {} if colors[i] is None else {"color": colors[i]}
            (ln,) = ax.plot([], [], linewidth=0.8, alpha=0.7, **kw)
            lines.append(ln)
        ax.set_xlim(time[0], time[-1])
        margin = 0.05 * abs(vmax - vmin) if vmax != vmin else 0.1
        ax.set_ylim(vmin - margin, vmax + margin)
        ax.set_xlabel("time")
        ax.set_ylabel(vn)
        all_artists.append((lines, vals, time, ax))

    fig.tight_layout()

    # Data is already downsampled by _extract_node_timeseries;
    # animate sequentially over all points.
    n_frames = len(slices[0][2])

    def update(i):
        """Update each panel's title and traces for frame `i`.

        Args:
            i: Frame index into the downsampled time axis.

        Returns:
            The updated line artists to redraw.
        """
        arts = []
        for lines, vals, time, ax in all_artists:
            ax.set_title(f"t = {time[i]:.2f}")
            for j, ln in enumerate(lines):
                ln.set_data(time[: i + 1], vals[: i + 1, j])
            arts.extend(lines)
        return arts

    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
    plt.close(fig)
    return ani


def animate_phase(result, x_var=None, y_var=None, region=0, mode=0, interval=50, trail=200, figsize=(6, 5)):
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

    units = getattr(result, "_units", {})
    xu, yu = units.get(xlabel, ""), units.get(ylabel, "")
    ax.set_xlabel(f"{xlabel} [{xu}]" if xu else xlabel)
    ax.set_ylabel(f"{ylabel} [{yu}]" if yu else ylabel)

    (line,) = ax.plot([], [], "b-", linewidth=0.8, alpha=0.7)
    (point,) = ax.plot([], [], "ro", markersize=6)

    step = max(1, len(time) // 400)
    frames = list(range(0, len(time), step))

    def update(frame):
        """Redraw the trailing tail and current point for `frame`.

        Args:
            frame: Index into the time axis for the current animation step.

        Returns:
            The trail line and current-position marker artists.
        """
        lo = max(0, frame - trail)
        line.set_data(x[lo : frame + 1], y[lo : frame + 1])
        point.set_data([x[frame]], [y[frame]])
        ax.set_title(f"t = {time[frame]:.1f} ms")
        return line, point

    ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    plt.close(fig)
    return ani


# =========================================================================
# Composable Panel System
# =========================================================================
#
# Each panel function has the signature:
#     panel_fn(result, ax, max_points=200) -> (n_frames, update_fn)
#
# update_fn(i) updates artists for frame i and returns a list of artists.
# Panels share a common frame count (the minimum across all panels).

# Built-in composite types: string -> list of panel names
_COMPOSITE_TYPES = {
    "pendulum": ["pendulum_bob", "timeseries"],
}


def _panel_timeseries(result, ax, max_points=200):
    """Panel: evolving time-series traces."""
    slices = _extract_node_timeseries(result.data, max_points=max_points)

    all_lines = []
    all_data = []
    for vn, vals, time in slices:
        n_traces = vals.shape[1]
        for j in range(n_traces):
            (ln,) = ax.plot([], [], linewidth=0.8, alpha=0.7, label=vn if n_traces == 1 else f"{vn}[{j}]")
            all_lines.append(ln)
            all_data.append((vals[:, j], time))

    if slices:
        time = slices[0][2]
        all_vals = np.concatenate([v for _, v, _ in slices], axis=1)
        vmin, vmax = float(all_vals.min()), float(all_vals.max())
        margin = 0.05 * abs(vmax - vmin) if vmax != vmin else 0.1
        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(vmin - margin, vmax + margin)
    ax.set_xlabel("time")
    ax.legend(loc="upper right", fontsize="small")
    n_frames = len(slices[0][2]) if slices else 0

    def update(i):
        for ln, (vals, time) in zip(all_lines, all_data):
            ln.set_data(time[: i + 1], vals[: i + 1])
        return all_lines

    return n_frames, update


def _panel_pendulum_bob(result, ax, max_points=200):
    """Panel: pendulum bob swinging from origin."""
    data = result.data
    time_raw = data.coords["time"].values if "time" in data.coords else np.arange(data.shape[0])
    var_names = list(np.atleast_1d(data.coords["variable"].values)) if "variable" in data.coords else []

    # Try x/y derived variables first, fall back to sin/cos of first state var
    if "x" in var_names and "y" in var_names:
        x_raw = np.asarray(data.sel(variable="x")).ravel()
        y_raw = np.asarray(data.sel(variable="y")).ravel()
    elif "theta" in var_names:
        theta = np.asarray(data.sel(variable="theta")).ravel()
        x_raw = np.sin(theta)
        y_raw = -np.cos(theta)
    else:
        raise ValueError("Pendulum panel requires 'x'+'y' or 'theta' variables.")

    # Downsample
    n_t = len(time_raw)
    if n_t > max_points:
        idx = np.linspace(0, n_t - 1, max_points, dtype=int)
        x_ds, y_ds = x_raw[idx], y_raw[idx]
    else:
        idx = None
        x_ds, y_ds = x_raw, y_raw

    pad = 0.3
    lim = max(abs(x_ds).max(), abs(y_ds).max()) + pad
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    (line,) = ax.plot([], [], "o-", lw=2, markersize=8)

    n_frames = len(x_ds)

    def update(i):
        line.set_data([0, x_ds[i]], [0, y_ds[i]])
        return [line]

    return n_frames, update


def _panel_phase(result, ax, max_points=200, trail=None):
    """Panel: phase-space trajectory with trailing tail."""
    from tvbo.plot.phase import _extract_2d

    time, x, y, xlabel, ylabel = _extract_2d(result, None, None, 0, 0)

    n_t = len(time)
    if n_t > max_points:
        idx = np.linspace(0, n_t - 1, max_points, dtype=int)
        time, x, y = time[idx], x[idx], y[idx]

    if trail is None:
        trail = max_points // 2

    ax.set_xlim(x.min() - 0.05 * np.ptp(x), x.max() + 0.05 * np.ptp(x))
    ax.set_ylim(y.min() - 0.05 * np.ptp(y), y.max() + 0.05 * np.ptp(y))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    (line,) = ax.plot([], [], "b-", linewidth=0.8, alpha=0.7)
    (point,) = ax.plot([], [], "ro", markersize=6)

    n_frames = len(time)

    def update(i):
        lo = max(0, i - trail)
        line.set_data(x[lo : i + 1], y[lo : i + 1])
        point.set_data([x[i]], [y[i]])
        return [line, point]

    return n_frames, update


# Registry of panel functions
_PANEL_REGISTRY = {
    "timeseries": _panel_timeseries,
    "pendulum_bob": _panel_pendulum_bob,
    "phase": _panel_phase,
}


def animate_multi(result, panels, interval=50, figsize=None, max_points=200, save=None, fps=20):
    """Compose multiple animation panels into a single FuncAnimation.

    Parameters
    ----------
    result : SimulationResult
    panels : list of str
        Panel type names (e.g. ['pendulum_bob', 'timeseries']).
    interval : int
        Milliseconds between frames.
    figsize : tuple, optional
    max_points : int
        Maximum time points per panel (downsampled for performance).
    save : str, optional
        If given, save the animation to this path (e.g. 'anim.gif').
    fps : int
        Frames per second when saving.

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    from matplotlib.animation import FuncAnimation

    n_panels = len(panels)
    if figsize is None:
        figsize = (4 * n_panels, 3)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, layout="compressed")
    if n_panels == 1:
        axes = [axes]

    updaters = []
    n_frames = None
    for ax, panel_name in zip(axes, panels):
        panel_fn = _PANEL_REGISTRY.get(panel_name)
        if panel_fn is None:
            raise ValueError(f"Unknown panel type '{panel_name}'. Available: {list(_PANEL_REGISTRY.keys())}")
        nf, update_fn = panel_fn(result, ax, max_points=max_points)
        updaters.append(update_fn)
        n_frames = min(n_frames, nf) if n_frames is not None else nf

    def update(i):
        """Advance every composed panel to frame `i` and gather their artists.

        Args:
            i: Frame index shared across all panels.

        Returns:
            The combined list of artists from every panel updater.
        """
        arts = []
        for fn in updaters:
            arts.extend(fn(i))
        return arts

    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=True)
    plt.close(fig)

    if save:
        ani.save(save, writer="pillow", fps=fps)

    return ani
