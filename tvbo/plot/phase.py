#
# Module: phase.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""Phase-space trajectory and vector-field plots for SimulationResult."""

import matplotlib.pyplot as plt
import numpy as np


def _extract_2d(result, x_var=None, y_var=None, region=0, mode=0):
    """Extract two variable time courses from a SimulationResult → (time, x, y, labels)."""
    data = result.data
    var_names = list(np.atleast_1d(data.coords["variable"].values)) if "variable" in data.coords else []

    if x_var is None:
        x_var = var_names[0] if len(var_names) >= 1 else None
    if y_var is None:
        y_var = var_names[1] if len(var_names) >= 2 else None

    if x_var is None or y_var is None:
        raise ValueError("Phase plot requires at least two state variables")

    sel_kw = {}
    if "node" in data.dims:
        sel_kw["node"] = data.coords["node"].values[region]
    if "mode" in data.dims:
        sel_kw["mode"] = mode

    x = np.asarray(data.sel(variable=x_var, **sel_kw)).ravel()
    y = np.asarray(data.sel(variable=y_var, **sel_kw)).ravel()
    time = data.coords["time"].values if "time" in data.coords else np.arange(len(x))
    return time, x, y, str(x_var), str(y_var)


def plot_phase(result, x_var=None, y_var=None, region=0, mode=0, ax=None, colorbar=True, **kwargs):
    """2D phase-space trajectory colored by time.

    Parameters
    ----------
    result : SimulationResult
    x_var, y_var : str, optional
        Variable names for x/y axes. Defaults to first two.
    region, mode : int
        Index selection when node/mode dims exist.
    ax : matplotlib.axes.Axes, optional
    colorbar : bool
        Show time colorbar.
    **kwargs
        Forwarded to ``ax.scatter()``.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    time, x, y, xlabel, ylabel = _extract_2d(result, x_var, y_var, region, mode)

    created = ax is None
    if created:
        fig, ax = plt.subplots()

    kwargs.setdefault("s", 1)
    kwargs.setdefault("c", time)
    kwargs.setdefault("cmap", "viridis")
    sc = ax.scatter(x, y, **kwargs)
    if colorbar and kwargs.get("c") is not None:
        cb = ax.figure.colorbar(sc, ax=ax, shrink=0.8)
        cb.set_label("time [ms]")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    units = getattr(result, "_units", {})
    xu, yu = units.get(xlabel, ""), units.get(ylabel, "")
    if xu:
        ax.set_xlabel(f"{xlabel} [{xu}]")
    if yu:
        ax.set_ylabel(f"{ylabel} [{yu}]")

    if created:
        plt.close()
        return fig
    return None


def plot_vector_field(
    result,
    x_var=None,
    y_var=None,
    region=0,
    mode=0,
    grid_n=20,
    ax=None,
    stream=True,
    trajectory=True,
    inputs=None,
    **kwargs,
):
    """Vector field (streamplot or quiver) from the dynamics RHS.

    Requires ``result`` to carry a reference to the source experiment
    (via ``ExperimentResult.source``) so the dynamics equations are available.

    Parameters
    ----------
    result : SimulationResult
    x_var, y_var : str, optional
    region, mode : int
    grid_n : int
        Grid resolution per axis.
    ax : matplotlib.axes.Axes, optional
    stream : bool
        If True use streamplot, otherwise quiver.
    trajectory : bool
        Overlay the simulation trajectory.
    inputs : dict, optional
        Values for symbols the dfun expects but the phase plane does not supply —
        typically coupling inputs such as ``c_glob``. Anything unspecified is
        evaluated at ``0``, giving the isolated-node vector field. Pass e.g.
        ``inputs={"c_glob": 0.3}`` to draw the field at a fixed coupling drive.
    **kwargs
        Forwarded to ``streamplot`` / ``quiver``.
    """
    from sympy import lambdify, symbols

    time, traj_x, traj_y, xlabel, ylabel = _extract_2d(result, x_var, y_var, region, mode)

    # Walk back to the object that owns the dynamics. A SimulationResult links to
    # its ExperimentResult, which in turn links to the SimulationExperiment — so a
    # single hop lands on a container that has no `.dynamics`. Follow the chain
    # instead of falling back to the container itself, which used to surface as
    # "'ExperimentResult' has no attribute 'state_variables'".
    dynamics = None
    seen = set()
    candidate = result
    while candidate is not None and id(candidate) not in seen:
        seen.add(id(candidate))
        found = getattr(candidate, "dynamics", None)
        if found is not None and hasattr(found, "state_variables"):
            dynamics = found
            break
        if hasattr(candidate, "state_variables"):
            dynamics = candidate
            break
        candidate = getattr(candidate, "_source", None) or getattr(candidate, "source", None)

    if dynamics is None:
        raise ValueError(
            "Vector field requires access to the dynamics equations, but none were "
            "reachable from this result. Plot from the ExperimentResult level, or "
            "pass the experiment explicitly."
        )

    # Build lambdified RHS for the two selected variables
    all_svs = dynamics.state_variables
    all_params = dynamics.parameters
    sv_names = list(all_svs.keys())

    # Symbol mapping: state variables + parameters
    sym_dict = {n: symbols(n) for n in sv_names}
    for pname in all_params:
        sym_dict[pname] = symbols(pname)

    # Parse RHS expressions
    from sympy.parsing.sympy_parser import parse_expr

    rhs_x = parse_expr(str(all_svs[xlabel].equation.rhs), local_dict=sym_dict)
    rhs_y = parse_expr(str(all_svs[ylabel].equation.rhs), local_dict=sym_dict)

    # Parameter values
    from tvbo.utils import initial_value, is_array_valued

    param_vals = {
        pname: float(p.value)
        for pname, p in all_params.items()
        if p.value is not None and not is_array_valued(p.value)
    }

    # Substitute fixed params and non-plotted state variables at their mean
    data = result.data
    subs = dict(param_vals)
    for sv_name in sv_names:
        if sv_name not in (xlabel, ylabel):
            if "variable" in data.coords and sv_name in data.coords["variable"].values:
                sel_kw = {"variable": sv_name}
                if "node" in data.dims:
                    sel_kw["node"] = data.coords["node"].values[region]
                if "mode" in data.dims:
                    sel_kw["mode"] = mode
                subs[sv_name] = float(np.asarray(data.sel(**sel_kw)).mean())
            else:
                sv_obj = all_svs.get(sv_name)
                subs[sv_name] = initial_value(sv_obj)

    rhs_x_sub = rhs_x.subs({sym_dict[k]: v for k, v in subs.items() if k in sym_dict})
    rhs_y_sub = rhs_y.subs({sym_dict[k]: v for k, v in subs.items() if k in sym_dict})

    x_sym, y_sym = sym_dict[xlabel], sym_dict[ylabel]

    # Anything still free after substitution is an input the dfun expects but the
    # phase plane does not supply — coupling inputs such as `c_glob` /
    # `local_coupling`, which are network quantities rather than model parameters.
    # Evaluate them at `inputs`(default 0), i.e. draw the *isolated-node* vector
    # field. Leaving them symbolic makes lambdify return an expression, which then
    # fails with "Cannot convert expression to float".
    residual = (rhs_x_sub.free_symbols | rhs_y_sub.free_symbols) - {x_sym, y_sym}
    if residual:
        fill = {s: float(inputs.get(str(s), 0.0)) if inputs else 0.0 for s in residual}
        rhs_x_sub = rhs_x_sub.subs(fill)
        rhs_y_sub = rhs_y_sub.subs(fill)
    fx = lambdify((x_sym, y_sym), rhs_x_sub, modules="numpy")
    fy = lambdify((x_sym, y_sym), rhs_y_sub, modules="numpy")

    # Build grid
    margin = 0.1
    x_lo, x_hi = float(traj_x.min()), float(traj_x.max())
    y_lo, y_hi = float(traj_y.min()), float(traj_y.max())
    dx, dy = (x_hi - x_lo) * margin, (y_hi - y_lo) * margin
    xg = np.linspace(x_lo - dx, x_hi + dx, grid_n)
    yg = np.linspace(y_lo - dy, y_hi + dy, grid_n)
    X, Y = np.meshgrid(xg, yg)
    U = np.asarray(fx(X, Y), dtype=float)
    V = np.asarray(fy(X, Y), dtype=float)

    created = ax is None
    if created:
        fig, ax = plt.subplots()

    if stream:
        speed = np.sqrt(U**2 + V**2)
        kwargs.setdefault("color", speed)
        kwargs.setdefault("cmap", "coolwarm")
        kwargs.setdefault("linewidth", 0.8)
        ax.streamplot(xg, yg, U, V, **kwargs)
    else:
        kwargs.setdefault("scale", None)
        ax.quiver(X, Y, U, V, **kwargs)

    if trajectory:
        ax.plot(traj_x, traj_y, "k-", linewidth=1, alpha=0.7, label="trajectory")
        ax.plot(traj_x[0], traj_y[0], "go", markersize=6, label="start")
        ax.plot(traj_x[-1], traj_y[-1], "rs", markersize=6, label="end")
        ax.legend(fontsize="smaller")

    units = getattr(result, "_units", {})
    xu, yu = units.get(xlabel, ""), units.get(ylabel, "")
    ax.set_xlabel(f"{xlabel} [{xu}]" if xu else xlabel)
    ax.set_ylabel(f"{ylabel} [{yu}]" if yu else ylabel)

    if created:
        plt.close()
        return fig
    return None
