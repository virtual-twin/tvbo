#
# Module: experiment_layout.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""Declarative and auto-configured layout composition for Experiment plotting."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from tvbo.plot.layout_mosaic import prepare_mosaic, finish_panel
from tvbo.plot.dynamics_layout import render_dynamics_panel


def _freeze_config(value):
    if isinstance(value, dict):
        return tuple((k, _freeze_config(v)) for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_config(v) for v in value)
    if isinstance(value, np.ndarray):
        return tuple(np.asarray(value).tolist())
    return value


def _pick_named_result(mapping, panel, kind):
    if not mapping:
        raise ValueError(f"experiment result has no {kind}")
    name = panel.get("name")
    if name is None:
        return next(iter(mapping.values()))
    if name not in mapping:
        raise ValueError(f"Unknown {kind} panel name: {name!r}")
    return mapping[name]


def _plot_exploration_timeseries_overlay(exploration, panel, ax):
    if exploration.results is None or not exploration.is_timeseries:
        raise ValueError("exploration timeseries overlay requires timeseries exploration results")

    axis_info = exploration.axes[0] if exploration.axes else None
    values = None
    if axis_info is not None:
        values = axis_info.get("explored_values") if isinstance(axis_info, dict) else getattr(axis_info, "explored_values", None)
    if values is None:
        values = np.arange(exploration.results.shape[0])

    time = exploration._get_time_axis()
    if time is None:
        time = np.arange(exploration.results.shape[1])

    plot_kwargs = dict(panel.get("plot", {}))
    colors = plt.colormaps[panel.get("cmap", "viridis")](np.linspace(0, 1, len(values) + 1))[:-1]
    label_fmt = panel.get("label_fmt", "{value:.2f}")
    component = int(panel.get("component", 0))

    results = exploration.results
    lead_dim = results.dims[0] if hasattr(results, "dims") and results.dims else None
    for idx, value in enumerate(values):
        # Select the run by its named leading dim (the swept parameter, trial, or
        # flat point) rather than by position, so a change in layout cannot quietly
        # read the wrong slice.
        if lead_dim:
            run = results.isel({lead_dim: idx})
            # Variable is selected by name; do not also index it positionally.
            if "variable" in run.dims:
                run = run.isel(variable=component)
            data = np.asarray(run).squeeze()
        else:
            # Unlabelled fallback: variable lives on axis 1, selected positionally.
            data = np.asarray(results[idx]).squeeze()
            if data.ndim > 1:
                data = data[:, component]
        line_kwargs = dict(plot_kwargs)
        line_kwargs.setdefault("alpha", panel.get("alpha", 0.8))
        line_kwargs.setdefault("linewidth", panel.get("lw", 1.0))
        line_kwargs.setdefault("color", colors[idx])
        ax.plot(
            time,
            data,
            label=label_fmt.format(value=float(value), index=idx),
            **line_kwargs,
        )

    axis_name = "parameter"
    if axis_info is not None:
        axis_name = axis_info.get("name", axis_name) if isinstance(axis_info, dict) else getattr(axis_info, "name", axis_name)
    ylabel = panel.get("ylabel") or exploration.observable or ", ".join(exploration.output_names) or "value"
    ax.set_xlabel(panel.get("xlabel", "Time"))
    ax.set_ylabel(ylabel)
    if panel.get("legend", True):
        ax.legend(
            title=panel.get("legend_title", axis_name),
            handlelength=0.8,
            fontsize="small",
            frameon=False,
            loc=panel.get("legend_loc", "best"),
        )


def _auto_experiment_panels(result):
    """Infer panel config from executed tasks in an ExperimentResult."""
    panels = {}

    def _next_key():
        return chr(ord("a") + len(panels))

    has_integration = result.integration is not None
    ts_explorations = {
        name: expl for name, expl in result.explorations.items()
        if getattr(expl, "is_timeseries", False)
    } if result.explorations else {}
    default_voi = None
    if ts_explorations:
        first_exploration = next(iter(ts_explorations.values()))
        output_names = getattr(first_exploration, "output_names", None) or []
        if output_names:
            default_voi = output_names[0]

    if has_integration:
        panels["a"] = {
            "kind": "integration",
            "modality": "timeseries",
            "title": "Integration",
        }

    if has_integration and ts_explorations:
        first_name = next(iter(ts_explorations))
        panels.setdefault("a", {}).setdefault("overlay", []).append(
            {
                "kind": "exploration",
                "name": first_name,
                "modality": "timeseries_overlay",
            }
        )

    if result.continuations:
        for name in result.continuations.keys():
            key = _next_key()
            panels[key] = {
                "kind": "continuation",
                "name": name,
                "title": f"Continuation: {name}",
            }
            if default_voi is not None:
                panels[key]["plot"] = {"VOI": default_voi}

    if not has_integration and ts_explorations:
        for name in ts_explorations.keys():
            key = _next_key()
            panels[key] = {
                "kind": "exploration",
                "name": name,
                "modality": "timeseries_overlay",
                "title": f"Exploration: {name}",
            }

    if result.explorations:
        for name, exploration in result.explorations.items():
            if name in ts_explorations:
                continue
            key = _next_key()
            panels[key] = {
                "kind": "exploration",
                "name": name,
                "title": f"Exploration: {name}",
            }

    if result.optimizations:
        for name in result.optimizations.keys():
            key = _next_key()
            panels[key] = {
                "kind": "optimization",
                "name": name,
                "title": f"Optimization: {name}",
            }

    if result.algorithms:
        for name in result.algorithms.keys():
            key = _next_key()
            panels[key] = {
                "kind": "algorithm",
                "name": name,
                "title": f"Algorithm: {name}",
            }

    if not panels:
        panels["a"] = {"kind": "integration", "modality": "timeseries", "title": result.name or "Experiment"}

    return panels


def _render_experiment_panel(experiment, panel, ax, cache, default_run_kwargs=None):
    from tvbo.plot.dynamics import _KINDS

    kind = panel.get("kind", "integration")
    if kind in _KINDS or kind in ("parameter_sweep_timeseries", "timeseries_parameter_sweep"):
        return render_dynamics_panel(experiment.dynamics, panel, ax, cache)

    run_kwargs = dict(default_run_kwargs or {})
    run_kwargs.update(panel.get("run", {}))
    run_key = ("experiment", _freeze_config(run_kwargs))
    if run_key not in cache:
        cache[run_key] = experiment.run(**run_kwargs)
    result = cache[run_key]

    if kind == "integration":
        if result.integration is None:
            raise ValueError("experiment result has no integration output")
        plot_kwargs = dict(panel.get("plot", {}))
        modality = panel.get("modality") or panel.get("type")
        if modality is not None:
            plot_kwargs.setdefault("type", modality)
        result.integration.plot(ax=ax, **plot_kwargs)

        for overlay in panel.get("overlay", []):
            if overlay.get("kind") == "exploration":
                expl = _pick_named_result(result.explorations, overlay, "exploration")
                _plot_exploration_timeseries_overlay(expl, overlay, ax)
        return result.integration

    if kind in ("bifurcation", "continuation"):
        continuation = _pick_named_result(result.continuations, panel, "continuation")
        continuation.plot(ax=ax, **panel.get("plot", {}))
        return continuation

    if kind == "exploration":
        exploration = _pick_named_result(result.explorations, panel, "exploration")
        modality = panel.get("modality")
        if modality in ("timeseries", "timeseries_overlay", "overlay"):
            _plot_exploration_timeseries_overlay(exploration, panel, ax)
        else:
            exploration.plot(ax=ax, **panel.get("plot", {}))
        return exploration

    if kind == "optimization":
        optimization = _pick_named_result(result.optimizations, panel, "optimization")
        optimization.plot(ax=ax, **panel.get("plot", {}))
        return optimization

    if kind == "algorithm":
        algorithm = _pick_named_result(result.algorithms, panel, "algorithm")
        algorithm.plot(ax=ax, **panel.get("plot", {}))
        return algorithm

    raise ValueError(f"Unsupported experiment panel kind: {kind!r}")


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
    """Compose integration, continuation, exploration, or dynamics panels.

    If panels are not provided, they are inferred from a single experiment run.
    """
    cache = {}
    if panels is None:
        result = experiment.run(**(run_kwargs or {}))
        cache[("experiment", _freeze_config(run_kwargs or {}))] = result
        panels = _auto_experiment_panels(result)

    fig_obj, ax_map, created = prepare_mosaic(
        layout=layout,
        panels=panels,
        fig=fig,
        axes=axes,
        figsize=figsize,
        subplot_kwargs=subplot_kwargs,
    )

    for name, ax in ax_map.items():
        if name not in panels:
            continue
        rendered = _render_experiment_panel(experiment, panels[name], ax, cache, default_run_kwargs=run_kwargs)
        cache[name] = rendered
        finish_panel(ax, panels[name])

    if created:
        fig_obj.tight_layout()
    if fig is None and axes is None:
        plt.close(fig_obj)
    return fig_obj
