#
# Module: dynamics.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""General-purpose plotting for ``Dynamics`` objects.

A single entry point ``plot_dynamics`` (also exposed as ``Dynamics.plot``)
that selects between several ``kind`` of representations: time series,
1D/2D/3D phase portraits, and 2D vector fields. Dimensions can be
state-variable names, derived-variable names, or arbitrary
sympy-parseable expressions over them. Labels are rendered as LaTeX via
``sympy.latex``. Styling and colormaps come from ``bsplot``.
"""

import bsplot
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr


# ---------------------------------------------------------------------------
# Symbolic helpers
# ---------------------------------------------------------------------------


def _scope(dynamics):
    return dynamics.get_symbolic_elements(include_time_symbol=True)


def _latex_label(dim, dynamics):
    """Render an arbitrary expression / name as a LaTeX axis label."""
    expr = parse_expr(str(dim), local_dict=_scope(dynamics))
    return f"${sp.latex(expr)}$"


def _resolve(dim, dynamics):
    """Parse ``dim`` and substitute derived vars/parameters → expression in state vars only.

    Returns ``(latex_label, expr)``.
    """
    scope = _scope(dynamics)
    state_names = list(dynamics.state_variables)
    expr = parse_expr(str(dim), local_dict=scope)

    derived_subs = {
        sp.Symbol(name): parse_expr(str(dv.equation.rhs), local_dict=scope)
        for name, dv in (getattr(dynamics, "derived_variables", {}) or {}).items()
        if getattr(dv, "equation", None) is not None
    }
    for _ in range(20):
        new = expr.xreplace(derived_subs)
        if new == expr:
            break
        expr = new

    from tvbo.utils import initial_value, is_array_valued

    param_subs = {
        sp.Symbol(name): float(p.value)
        for name, p in dynamics.parameters.items()
        if p.value is not None and not is_array_valued(p.value)
    }
    for name, dp in (getattr(dynamics, "derived_parameters", {}) or {}).items():
        if getattr(dp, "equation", None) is not None:
            param_subs[sp.Symbol(name)] = parse_expr(
                str(dp.equation.rhs), local_dict=scope
            ).xreplace(param_subs)
    expr = expr.xreplace(param_subs)

    extras = expr.free_symbols - {sp.Symbol(n) for n in state_names} - {sp.Symbol("t")}
    if extras:
        raise ValueError(
            f"Expression {dim!r} has unresolved symbols: {sorted(map(str, extras))}"
        )

    return _latex_label(dim, dynamics), expr


def _evaluator(expr, dynamics):
    """Lambdify an expression over state variables and time."""
    state_syms = [sp.Symbol(n) for n in dynamics.state_variables]
    t_sym = sp.Symbol("t")
    f = sp.lambdify(state_syms + [t_sym], expr, modules="numpy")

    def _eval(traj, time):
        cols = [traj[:, i] for i in range(traj.shape[1])]
        out = f(*cols, time)
        return np.broadcast_to(np.asarray(out, dtype=float), (traj.shape[0],)).copy()

    return _eval


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


def _initial_conditions(dynamics, n_trials, u_0):
    n_sv = len(dynamics.state_variables)
    if u_0 is not None:
        u_0 = np.asarray(u_0, dtype=float)
        if u_0.ndim == 1:
            u_0 = np.broadcast_to(u_0, (n_trials, n_sv)).copy()
        return u_0
    if n_trials == 1:
        return np.asarray(dynamics.get_initial_values(), dtype=float).reshape(1, n_sv)
    return np.asarray(dynamics.get_initial_values(random=True, N=n_trials)).T


def _run_trials(dynamics, n_trials, duration, dt, u_0, transient):
    u_0 = _initial_conditions(dynamics, n_trials, u_0)
    trajs, time = [], None
    for i in range(n_trials):
        ts = dynamics.run(duration=duration, dt=dt, u_0=u_0[i], save=False)
        traj = ts.data[:, :, 0, 0]
        t = ts.time
        if transient > 0:
            keep = t >= transient
            traj, t = traj[keep], t[keep]
        trajs.append(traj)
        time = t
    return trajs, time


def _trial_colors(n_trials, cmap):
    cm = plt.colormaps[cmap]
    if n_trials == 1:
        return [None], None
    norm = plt.Normalize(0, n_trials - 1)
    return [cm(norm(i)) for i in range(n_trials)], norm


def _render_dynamics_panel(dynamics, panel, ax, cache):
    # Compatibility shim: moved to tvbo.plot.dynamics_layout
    from tvbo.plot.dynamics_layout import render_dynamics_panel

    return render_dynamics_panel(dynamics, panel, ax, cache)


def _finish_panel(ax, panel):
    # Compatibility shim: moved to tvbo.plot.layout_mosaic
    from tvbo.plot.layout_mosaic import finish_panel

    return finish_panel(ax, panel)


def plot_dynamics_layout(
    dynamics,
    layout=None,
    panels=None,
    figsize=None,
    subplot_kwargs=None,
    fig=None,
    axes=None,
):
    """Backwards-compatible shim — delegates to [`tvbo.plot.dynamics_layout.plot_dynamics_layout`](./dynamics_layout.qmd#tvbo.plot.dynamics_layout.plot_dynamics_layout)."""
    # Compatibility shim: moved to tvbo.plot.dynamics_layout
    from tvbo.plot.dynamics_layout import plot_dynamics_layout as _plot_dynamics_layout

    return _plot_dynamics_layout(
        dynamics,
        layout=layout,
        panels=panels,
        figsize=figsize,
        subplot_kwargs=subplot_kwargs,
        fig=fig,
        axes=axes,
    )


def plot_experiment_layout(
    experiment,
    layout=None,
    panels=None,
    figsize=None,
    subplot_kwargs=None,
    run_kwargs=None,
    fig=None,
    axes=None,
):
    """Backwards-compatible shim — delegates to [`tvbo.plot.experiment_layout.plot_experiment_layout`](./experiment_layout.qmd#tvbo.plot.experiment_layout.plot_experiment_layout)."""
    # Compatibility shim: moved to tvbo.plot.experiment_layout
    from tvbo.plot.experiment_layout import plot_experiment_layout as _plot_experiment_layout

    return _plot_experiment_layout(
        experiment,
        layout=layout,
        panels=panels,
        figsize=figsize,
        subplot_kwargs=subplot_kwargs,
        run_kwargs=run_kwargs,
        fig=fig,
        axes=axes,
    )


# ---------------------------------------------------------------------------
# Kinds
# ---------------------------------------------------------------------------


def _kind_timeseries(dynamics, resolved, trials, time, ax, cmap, alpha, lw):
    n_dims = len(resolved)
    if ax is None:
        fig, axes = plt.subplots(n_dims, 1, sharex=True, squeeze=False,
                                 figsize=(6, 1.8 * n_dims))
        axes = axes[:, 0]
    else:
        axes = np.atleast_1d(ax)
        fig = axes[0].figure

    colors, _ = _trial_colors(len(trials), cmap)
    for d_idx, (label, expr) in enumerate(resolved):
        evaluator = _evaluator(expr, dynamics)
        for t_idx, traj in enumerate(trials):
            bsplot.timeseries.plot_ts(
                axes[d_idx], time, evaluator(traj, time),
                label=f"trial {t_idx}" if d_idx == 0 and len(trials) > 1 else None,
                data_label=label, time_unit="ms", box_aspect=None,
            )
            line = axes[d_idx].lines[-1]
            if colors[t_idx] is not None:
                line.set_color(colors[t_idx])
            line.set_alpha(alpha)
            line.set_linewidth(lw)
        axes[d_idx].set_box_aspect(None)
    if len(trials) > 1:
        axes[0].legend(loc="best", fontsize="small", frameon=False)
    fig.tight_layout()
    return fig


def _kind_phase(dynamics, resolved, trials, time, ax, cmap, alpha, lw,
                show_ic, color_by, ax_given=False):
    n_dims = len(resolved)
    if ax is None:
        if n_dims == 3:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots(figsize=(5, 4.5))
    else:
        fig = ax.figure

    evaluators = [_evaluator(expr, dynamics) for _, expr in resolved]
    labels = [lbl for lbl, _ in resolved]
    n_trials = len(trials)
    cm = plt.colormaps[cmap]

    if color_by is not None:
        _, color_expr = _resolve(color_by, dynamics)
        ce = _evaluator(color_expr, dynamics)
        color_vals = np.array([float(ce(traj[:1], time[:1])[0]) for traj in trials])
        color_label = _latex_label(color_by, dynamics)
    elif n_trials > 1:
        color_vals = np.arange(n_trials, dtype=float)
        color_label = "trial"
    else:
        color_vals = None
        color_label = None

    if color_vals is not None:
        lo, hi = float(color_vals.min()), float(color_vals.max())
        norm = plt.Normalize(lo, hi if hi != lo else lo + 1)

    single_traj_time_color = (n_trials == 1 and color_by is None and n_dims >= 2)

    for i, traj in enumerate(trials):
        ys = [evaluators[d](traj, time) for d in range(n_dims)]
        color = cm(norm(color_vals[i])) if color_vals is not None else None
        if n_dims == 1:
            ax.plot(time, ys[0], color=color, lw=lw, alpha=alpha)
            if show_ic:
                ax.plot(time[0], ys[0][0], "o", color=color or "k", ms=4, zorder=5)
        elif n_dims == 2:
            if single_traj_time_color:
                sc = ax.scatter(ys[0], ys[1], c=time, cmap=cmap, s=2)
                if not ax_given:
                    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
                    cb.set_label("time [ms]")
            else:
                ax.plot(ys[0], ys[1], color=color, lw=lw, alpha=alpha)
            if show_ic:
                ax.plot(ys[0][0], ys[1][0], "o", color=color or "k", ms=4, zorder=5)
        else:
            if single_traj_time_color:
                # Color a 3D line by time using a LineCollection-like trick via segments
                t_norm = plt.Normalize(time.min(), time.max())
                for k in range(len(time) - 1):
                    ax.plot(ys[0][k:k + 2], ys[1][k:k + 2], ys[2][k:k + 2],
                            color=cm(t_norm(time[k])), lw=lw, alpha=alpha)
                if not ax_given:
                    sm_t = plt.cm.ScalarMappable(cmap=cmap, norm=t_norm)
                    sm_t.set_array([])
                    cb = fig.colorbar(sm_t, ax=ax, shrink=0.8, pad=0.1)
                    cb.set_label("time [ms]")
            else:
                ax.plot(ys[0], ys[1], ys[2], color=color, lw=lw, alpha=alpha)
            if show_ic:
                ax.scatter(ys[0][0], ys[1][0], ys[2][0],
                           color=color or "k", s=20, zorder=5)

    if color_vals is not None and not single_traj_time_color and not ax_given:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, shrink=0.8)
        cb.set_label(color_label)

    if n_dims == 1:
        ax.set_xlabel("time [ms]"); ax.set_ylabel(labels[0])
    elif n_dims == 2:
        ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1])
        ax.set_box_aspect(1)
    else:
        ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1]); ax.set_zlabel(labels[2])
    return fig


def _kind_vectorfield(dynamics, resolved, ax, grid_n, cmap, stream, ax_given=False,
                      alpha=0.8, lw=1.5, quiver_scale=None,
                      quiver_width=None, stream_density=1.2,
                      stream_arrowsize=1.0, _return_axes_only=False):
    if len(resolved) != 2:
        raise ValueError("vectorfield requires exactly 2 dims (state variables)")
    state_names = list(dynamics.state_variables)
    sv_idx = []
    for label, expr in resolved:
        if not (isinstance(expr, sp.Symbol) and str(expr) in state_names):
            raise ValueError("vectorfield dims must be state-variable names")
        sv_idx.append(state_names.index(str(expr)))

    sv_objs = list(dynamics.state_variables.values())

    def _range(sv):
        if sv.domain and sv.domain.lo is not None and sv.domain.hi is not None:
            return float(sv.domain.lo), float(sv.domain.hi)
        v = initial_value(sv)
        return v - 1.0, v + 1.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.5))
        lo_x, hi_x = _range(sv_objs[sv_idx[0]])
        lo_y, hi_y = _range(sv_objs[sv_idx[1]])
    else:
        fig = ax.figure
        lo_x, hi_x = ax.get_xlim()
        lo_y, hi_y = ax.get_ylim()
        # If caller gave an empty axis with default (0,1) limits, fall back to domain
        if (lo_x, hi_x) == (0.0, 1.0):
            lo_x, hi_x = _range(sv_objs[sv_idx[0]])
        if (lo_y, hi_y) == (0.0, 1.0):
            lo_y, hi_y = _range(sv_objs[sv_idx[1]])

    X, Y = np.meshgrid(np.linspace(lo_x, hi_x, grid_n), np.linspace(lo_y, hi_y, grid_n))
    base_ic = np.asarray(dynamics.get_initial_values(), dtype=float).reshape(-1)
    f = dynamics.execute(format="python")
    U, V = np.zeros_like(X), np.zeros_like(Y)
    for i in range(grid_n):
        for j in range(grid_n):
            u = base_ic.copy()
            u[sv_idx[0]] = X[i, j]
            u[sv_idx[1]] = Y[i, j]
            du = np.asarray(f(u, 0.0))
            U[i, j] = du[sv_idx[0]]
            V[i, j] = du[sv_idx[1]]

    speed = np.hypot(U, V)
    if stream:
        strm = ax.streamplot(X, Y, U, V, color=speed, cmap=cmap,
                             density=stream_density,
                             arrowsize=stream_arrowsize, linewidth=lw)
        strm.lines.set_alpha(alpha)
        if not ax_given and not _return_axes_only:
            cb = fig.colorbar(strm.lines, ax=ax, shrink=0.8)
            cb.set_label("|f|")
    else:
        quiver_kwargs = {"cmap": cmap, "alpha": alpha}
        if quiver_scale is not None:
            quiver_kwargs.update(angles="xy", scale_units="xy", scale=quiver_scale)
        if quiver_width is not None:
            quiver_kwargs["width"] = quiver_width
        q = ax.quiver(X, Y, U, V, speed, **quiver_kwargs)
        if not ax_given and not _return_axes_only:
            cb = fig.colorbar(q, ax=ax, shrink=0.8)
            cb.set_label("|f|")
    ax.set_xlabel(resolved[0][0]); ax.set_ylabel(resolved[1][0])
    ax.set_xlim(lo_x, hi_x); ax.set_ylim(lo_y, hi_y)
    ax.set_box_aspect(1)
    if _return_axes_only:
        return fig, (X, Y, U, V)
    return fig


def _kind_phaseplane(dynamics, resolved, ax, grid_n, cmap, stream, ax_given,
                     alpha, lw, n_trajectories=0, traj_duration=200, traj_dt=0.01,
                     show_nullclines=True, show_fixed_points=True,
                     show_limit_cycle=True, trajectory_color="red"):
    """Vector field + nullclines + (optional) sample trajectories.

    Nullclines are the zero-level contours of each component of the vector
    field (``\\dot x_i = 0``). Their intersections are equilibria.
    """
    fig, (X, Y, U, V) = _kind_vectorfield(
        dynamics, resolved, ax=ax, grid_n=max(grid_n, 30),
        cmap=cmap, stream=stream, ax_given=ax_given,
        alpha=alpha * 0.6, lw=lw * 0.6, _return_axes_only=True,
    )
    if ax is None:
        ax = fig.axes[0]

    if show_nullclines:
        ax.contour(X, Y, U, levels=[0], colors="C0", linewidths=1.5,
                   linestyles="-", alpha=0.9)
        ax.contour(X, Y, V, levels=[0], colors="C3", linewidths=1.5,
                   linestyles="-", alpha=0.9)
        # Legend proxies
        from matplotlib.lines import Line2D
        proxies = [
            Line2D([0], [0], color="C0", lw=1.5,
                   label=f"$\\dot {{{resolved[0][0].strip('$')}}} = 0$"),
            Line2D([0], [0], color="C3", lw=1.5,
                   label=f"$\\dot {{{resolved[1][0].strip('$')}}} = 0$"),
        ]
        ax.legend(handles=proxies, loc="best", fontsize="small", frameon=False)

    if show_fixed_points:
        # Detect fixed points by sign-change crossings in both U and V.
        # Robust grid-based heuristic: cells where U and V both straddle 0.
        from scipy.optimize import fsolve
        f = dynamics.execute(format="python")
        state_names = list(dynamics.state_variables)
        sv_idx = [state_names.index(str(expr)) for _, expr in resolved]
        base_ic = np.asarray(dynamics.get_initial_values(), dtype=float).reshape(-1)

        def _rhs(xy):
            u = base_ic.copy()
            u[sv_idx[0]] = xy[0]; u[sv_idx[1]] = xy[1]
            du = np.asarray(f(u, 0.0))
            return [du[sv_idx[0]], du[sv_idx[1]]]

        seen = []
        for i in range(X.shape[0] - 1):
            for j in range(X.shape[1] - 1):
                u_box = U[i:i + 2, j:j + 2]
                v_box = V[i:i + 2, j:j + 2]
                if (u_box.min() <= 0 <= u_box.max()) and (v_box.min() <= 0 <= v_box.max()):
                    sol, _, ier, _ = fsolve(
                        _rhs, [X[i, j], Y[i, j]], full_output=True
                    )
                    if ier == 1 and not any(np.allclose(sol, s, atol=1e-3) for s in seen):
                        seen.append(sol)
        for x_fp, y_fp in seen:
            ax.plot(x_fp, y_fp, "ko", ms=7, mec="white", mew=1.2, zorder=10)

    if show_limit_cycle:
        # Integrate one long trajectory from a small offset, discard
        # transient, and check if the tail forms a closed loop. If yes,
        # draw it as a red ring (and mark the inner unstable focus).
        state_names = list(dynamics.state_variables)
        sv_idx = [state_names.index(str(expr)) for _, expr in resolved]
        base_ic = np.asarray(dynamics.get_initial_values(), dtype=float).reshape(-1)
        u0 = base_ic.copy()
        u0[sv_idx[0]] += 0.05
        u0[sv_idx[1]] += 0.05
        ts = dynamics.run(duration=max(traj_duration * 4, 200),
                          dt=traj_dt, u_0=u0, save=False)
        traj = ts.data[:, :, 0, 0]
        x = traj[:, sv_idx[0]]; y = traj[:, sv_idx[1]]
        n = len(x)
        tail = slice(int(n * 0.7), n)
        xt, yt = x[tail], y[tail]
        amp = np.hypot(xt.max() - xt.min(), yt.max() - yt.min())
        closes = np.hypot(xt[0] - xt[-1], yt[0] - yt[-1]) < 0.1 * amp
        if amp > 0.05 and closes:
            ax.plot(xt, yt, color="red", lw=1.8, zorder=9)

    if n_trajectories > 0:
        from numpy.random import default_rng
        rng = default_rng(0)
        lo_x, hi_x = ax.get_xlim(); lo_y, hi_y = ax.get_ylim()
        state_names = list(dynamics.state_variables)
        sv_idx = [state_names.index(str(expr)) for _, expr in resolved]
        base_ic = np.asarray(dynamics.get_initial_values(), dtype=float).reshape(-1)
        for _ in range(n_trajectories):
            u0 = base_ic.copy()
            u0[sv_idx[0]] = rng.uniform(lo_x, hi_x)
            u0[sv_idx[1]] = rng.uniform(lo_y, hi_y)
            ts = dynamics.run(duration=traj_duration, dt=traj_dt, u_0=u0, save=False)
            traj = ts.data[:, :, 0, 0]
            ax.plot(traj[:, sv_idx[0]], traj[:, sv_idx[1]],
                    color=trajectory_color, lw=0.8, alpha=0.7)
    return fig


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_KINDS = ("auto", "timeseries", "phase", "vectorfield", "phaseplane")


def plot_dynamics(
    dynamics,
    *dims,
    kind="auto",
    duration=2000,
    dt=0.1,
    n_trials=1,
    u_0=None,
    color_by=None,
    cmap="viridis",
    transient=0.0,
    ax=None,
    show_ic=True,
    lw=0.8,
    alpha=0.8,
    grid_n=20,
    stream=True,
    stream_density=1.2,
    stream_arrowsize=1.0,
    quiver_scale=None,
    quiver_width=None,
    n_trajectories=0,
    show_nullclines=True,
    show_fixed_points=True,
    trajectory_color="red",
):
    """Plot a ``Dynamics`` in several ways.

    Parameters
    ----------
    dynamics : Dynamics
    *dims : str
        1-3 dimensions: state-variable name, derived-variable name, or any
        sympy-parseable expression over them. Defaults: first SV (timeseries),
        first 2 SVs (phase / vectorfield / phaseplane).
    kind : {"auto", "timeseries", "phase", "vectorfield", "phaseplane"}
        - ``timeseries``: stacked time series, one panel per dim.
        - ``phase``: 1D/2D/3D phase portrait.
        - ``vectorfield``: 2D vector field (stream- or quiver-plot).
        - ``phaseplane``: vector field + nullclines + (optional) sample
          trajectories + auto-detected fixed points.
        - ``auto``: timeseries for 1 dim, phase for 2-3 dims.
    duration, dt : float
        Integration parameters in ms.
    n_trials : int
        Number of trajectories. ``> 1`` resamples ICs from each SV's distribution.
    u_0 : array-like, optional
        Explicit IC(s); shape ``(n_sv,)`` or ``(n_trials, n_sv)``.
    color_by : str, optional
        Variable / expression to color trajectories by (phase). Defaults to
        trial index for multi-trial plots, time for single-trial 2D/3D.
    cmap : str
        Matplotlib / bsplot colormap name.
    transient : float
        Discard this many ms from the start of each trajectory.
    ax : matplotlib.axes.Axes, optional
    show_ic : bool
    lw, alpha : float
    grid_n : int
        Grid resolution for ``kind="vectorfield"`` / ``"phaseplane"``.
    stream : bool
        Stream- vs quiver-plot for vectorfield / phaseplane.
    stream_density, stream_arrowsize : float
        Matplotlib streamplot controls used when ``stream=True``.
    quiver_scale, quiver_width : float, optional
        Matplotlib quiver sizing controls used when ``stream=False``.
    n_trajectories : int
        Number of sample trajectories overlaid on a phase plane (random ICs).
    show_nullclines, show_fixed_points : bool
        Toggles for ``kind="phaseplane"``.
    trajectory_color : str
        Color of overlaid sample trajectories on a ``kind="phaseplane"`` plot.
        Defaults to ``"red"`` for clear contrast against the streamline /
        nullcline overlay.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")

    ax_given = ax is not None
    state_names = list(dynamics.state_variables)
    if not dims:
        dims = (
            tuple(state_names[: min(2, len(state_names))])
            if kind in ("phase", "vectorfield", "phaseplane")
            else (state_names[0],)
        )
    if not 1 <= len(dims) <= 3:
        raise ValueError(f"plot supports 1-3 dims, got {len(dims)}")
    if kind == "auto":
        kind = "timeseries" if len(dims) == 1 else "phase"

    resolved = [_resolve(d, dynamics) for d in dims]

    if kind == "vectorfield":
        fig = _kind_vectorfield(dynamics, resolved, ax=ax, grid_n=grid_n,
                                cmap=cmap, stream=stream, ax_given=ax_given,
                                alpha=alpha, lw=lw,
                                quiver_scale=quiver_scale,
                                quiver_width=quiver_width,
                                stream_density=stream_density,
                                stream_arrowsize=stream_arrowsize)
    elif kind == "phaseplane":
        fig = _kind_phaseplane(dynamics, resolved, ax=ax, grid_n=grid_n,
                               cmap=cmap, stream=stream, ax_given=ax_given,
                               alpha=alpha, lw=lw,
                               n_trajectories=n_trajectories,
                               traj_duration=duration, traj_dt=dt,
                               show_nullclines=show_nullclines,
                               show_fixed_points=show_fixed_points,
                               trajectory_color=trajectory_color)
    else:
        trials, time = _run_trials(dynamics, n_trials, duration, dt, u_0, transient)
        if kind == "timeseries":
            fig = _kind_timeseries(dynamics, resolved, trials, time, ax=ax,
                                   cmap=cmap, alpha=alpha, lw=lw)
        else:
            fig = _kind_phase(dynamics, resolved, trials, time, ax=ax, cmap=cmap,
                              alpha=alpha, lw=lw, show_ic=show_ic, color_by=color_by,
                              ax_given=ax_given)

    if not ax_given:
        plt.close(fig)
        return fig
    return None


def animate_dynamics(
    dynamics,
    parameter,
    values,
    *dims,
    kind="vectorfield",
    interval=80,
    figsize=(5, 4.5),
    title_fmt="{name} = {value:.3f}",
    **kwargs,
):
    """Animate a :class:`Dynamics` by sweeping one parameter through ``values``.

    For each frame the parameter is set, the chosen ``kind`` of plot is drawn
    on the same axes (cleared between frames), and a title shows the current
    value. Returns a :class:`matplotlib.animation.FuncAnimation` that you can
    display with ``anim.to_jshtml()`` (Quarto / notebooks) or save with
    ``anim.save("foo.mp4")``.

    Parameters
    ----------
    dynamics : Dynamics
    parameter : str
        Name of the parameter to sweep (must be in ``dynamics.parameters``).
    values : sequence of float
        Parameter values, one per frame.
    *dims, kind, **kwargs
        Forwarded to :func:`plot_dynamics`.
    interval : int
        Delay between frames in ms.
    figsize : (float, float)
    title_fmt : str
        Format string with ``{name}`` and ``{value}`` placeholders.
    """
    import copy
    from matplotlib.animation import FuncAnimation

    if parameter not in dynamics.parameters:
        raise ValueError(
            f"parameter {parameter!r} not in dynamics "
            f"(available: {list(dynamics.parameters)})"
        )

    dyn = copy.deepcopy(dynamics)
    fig, ax = plt.subplots(figsize=figsize)

    def _update(frame_idx):
        ax.clear()
        val = float(values[frame_idx])
        dyn.parameters[parameter].value = val
        plot_dynamics(dyn, *dims, kind=kind, ax=ax, **kwargs)
        ax.set_title(title_fmt.format(name=parameter, value=val))
        return ax.get_children()

    anim = FuncAnimation(fig, _update, frames=len(values),
                         interval=interval, blit=False)
    plt.close(fig)
    return anim
