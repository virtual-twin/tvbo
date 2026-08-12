#
# Module: dynamics_layout.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""Declarative layout composition for Dynamics plotting."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from tvbo.plot.layout_mosaic import prepare_mosaic, finish_panel


def _coerce_dims(panel):
    dims = panel.get("dims")
    if dims is None and panel.get("VOI") is not None:
        dims = (panel["VOI"],)
    if dims is None:
        return ()
    if isinstance(dims, str):
        return (dims,)
    return tuple(dims)


def _parameter_values_from_result(result, n_values):
    df = getattr(result, "df", None)
    if df is None or "param" not in df:
        raise ValueError("source panel does not expose continuation parameters")
    p_min = float(df["param"].min())
    p_max = float(df["param"].max())
    return np.linspace(p_min, p_max, int(n_values))


def _plot_parameter_sweep_timeseries(dynamics, panel, ax, cache):
    from tvbo.plot.dynamics import _resolve, _evaluator

    parameter = panel.get("parameter") or panel.get("ICS")
    if parameter is None:
        raise ValueError("parameter_sweep_timeseries requires 'parameter' or 'ICS'")

    dims = _coerce_dims(panel)
    if not dims:
        dims = (next(iter(dynamics.state_variables)),)
    resolved = [_resolve(dim, dynamics) for dim in dims]
    evaluators = [_evaluator(expr, dynamics) for _, expr in resolved]

    values = panel.get("values")
    if values is None:
        source_panel = panel.get("from_panel")
        if source_panel is None or source_panel not in cache:
            raise ValueError("parameter_sweep_timeseries needs 'values' or a valid 'from_panel'")
        values = _parameter_values_from_result(cache[source_panel], panel.get("n_values", 3))
    values = np.asarray(values, dtype=float)

    plot_kwargs = dict(panel.get("plot", {}))
    run_kwargs = dict(panel.get("run", {}))
    run_format = run_kwargs.pop("format", panel.get("format", "python"))
    if "t" in panel:
        run_kwargs["t"] = panel["t"]
    else:
        if "duration" in panel:
            run_kwargs["duration"] = panel["duration"]
        if "dt" in panel:
            run_kwargs["dt"] = panel["dt"]

    colors = plt.colormaps[panel.get("cmap", "viridis")](np.linspace(0, 1, len(values) + 1))[:-1]
    label_fmt = panel.get("label_fmt", "{value:.2f}")
    dyn = dynamics.copy()

    for idx, value in enumerate(values):
        dyn.parameters[parameter].value = float(value)
        ts = dyn.run(format=run_format, **run_kwargs)
        traj = ts.data[:, :, 0, 0]
        time = ts.time
        for dim_idx, ((label, _), evaluator) in enumerate(zip(resolved, evaluators)):
            label_text = label_fmt.format(value=float(value), parameter=parameter)
            if len(resolved) > 1:
                label_text = f"{label_text} · {label}"
            line_kwargs = dict(plot_kwargs)
            line_kwargs.setdefault("alpha", panel.get("alpha", 0.8))
            line_kwargs.setdefault("linewidth", panel.get("lw", 1.0))
            line_kwargs.setdefault("color", colors[idx])
            ax.plot(time, evaluator(traj, time), label=label_text if dim_idx == 0 else None, **line_kwargs)

    ax.set_xlabel(panel.get("xlabel", "Time in ms"))
    if len(resolved) == 1:
        ax.set_ylabel(panel.get("ylabel", resolved[0][0]))
    if panel.get("legend", True):
        ax.legend(
            title=panel.get("legend_title", parameter),
            handlelength=0.8,
            fontsize="small",
            frameon=False,
            loc=panel.get("legend_loc", "best"),
        )


def render_dynamics_panel(dynamics, panel, ax, cache):
    """Render a single dynamics panel onto an axes according to its kind.

    Dispatches on `panel["kind"]` (default `"timeseries"`): standard plot
    kinds are drawn with `plot_dynamics`, `"bifurcation"`/`"continuation"`
    run a continuation and plot the result, and `"parameter_sweep_timeseries"`
    (or `"timeseries_parameter_sweep"`) overlays trajectories across a
    parameter range.

    Args:
        dynamics: The `Dynamics` model to render; copied before running so the
            original is left unmodified.
        panel: Panel specification mapping. Recognized keys include `kind`,
            `dims`/`VOI`, `run`, `plot`, and `format`, plus kind-specific keys
            such as `parameter`, `values`, and `from_panel` for parameter
            sweeps.
        ax: The Matplotlib axes to draw the panel on.
        cache: Mapping of previously rendered panel names to their results,
            used to resolve continuation parameters referenced via `from_panel`.

    Returns:
        The continuation/bifurcation result object for those kinds (so later
        panels can reference it), or `None` for plot and parameter-sweep kinds.

    Raises:
        ValueError: If `kind` is not one of the supported panel kinds.
    """
    from tvbo.plot.dynamics import _KINDS, plot_dynamics

    kind = panel.get("kind", "timeseries")
    if kind in _KINDS:
        dims = _coerce_dims(panel)
        kwargs = {
            k: v
            for k, v in panel.items()
            if k
            not in {
                "kind",
                "dims",
                "VOI",
                "title",
                "xlabel",
                "ylabel",
                "box_aspect",
                "legend",
                "legend_title",
                "legend_loc",
                "plot",
                "run",
                "parameter",
                "ICS",
                "values",
                "from_panel",
                "n_values",
                "label_fmt",
            }
        }
        plot_dynamics(dynamics.copy(), *dims, kind=kind, ax=ax, **kwargs)
        return None

    if kind in ("bifurcation", "continuation"):
        run_kwargs = dict(panel.get("run", {}))
        run_kwargs.setdefault("format", panel.get("format", "bifurcation-julia"))
        result = dynamics.copy().run(**run_kwargs)
        result.plot(ax=ax, **panel.get("plot", {}))
        return result

    if kind in ("parameter_sweep_timeseries", "timeseries_parameter_sweep"):
        _plot_parameter_sweep_timeseries(dynamics, panel, ax, cache)
        return None

    raise ValueError(f"Unsupported dynamics panel kind: {kind!r}")


def plot_dynamics_layout(
    dynamics,
    layout=None,
    panels=None,
    figsize=None,
    subplot_kwargs=None,
    fig=None,
    axes=None,
):
    """Compose multiple dynamics-related panels via ``subplot_mosaic``."""
    panels = panels or {}
    fig_obj, ax_map, created = prepare_mosaic(
        layout=layout,
        panels=panels,
        fig=fig,
        axes=axes,
        figsize=figsize,
        subplot_kwargs=subplot_kwargs,
    )

    cache = {}
    for name, ax in ax_map.items():
        if name not in panels:
            continue
        result = render_dynamics_panel(dynamics, panels[name], ax, cache)
        cache[name] = result
        finish_panel(ax, panels[name])

    if created:
        fig_obj.tight_layout()
    if fig is None and axes is None:
        plt.close(fig_obj)
    return fig_obj
