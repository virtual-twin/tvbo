#
# Module: layout_mosaic.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""Reusable subplot_mosaic helpers for declarative panel layouts."""

from __future__ import annotations

import matplotlib.pyplot as plt


def default_layout(panels):
    """Return a compact default layout string from panel keys."""
    keys = list(panels)
    if not keys:
        raise ValueError("panels must not be empty")
    return "".join(keys)


def finish_panel(ax, panel):
    """Apply common axis-level decorations from panel config."""
    if panel.get("title"):
        ax.set_title(panel["title"])
    if panel.get("xlabel"):
        ax.set_xlabel(panel["xlabel"])
    if panel.get("ylabel"):
        ax.set_ylabel(panel["ylabel"])
    if panel.get("box_aspect") is not None:
        ax.set_box_aspect(panel["box_aspect"])


def prepare_mosaic(layout=None, panels=None, fig=None, axes=None, figsize=None, subplot_kwargs=None):
    """Prepare figure + axes mapping for mosaic rendering.

    Parameters
    ----------
    layout : str | list[list[str]] | None
        subplot_mosaic layout specification.
    panels : dict | None
        Panel config keyed by mosaic labels.
    fig : matplotlib.figure.Figure | None
        Existing figure to render into.
    axes : dict[str, matplotlib.axes.Axes] | None
        Existing axes mapping keyed by panel labels.
    figsize : tuple | None
        Figure size used only when creating a new figure.
    subplot_kwargs : dict | None
        Forwarded to ``subplot_mosaic`` when axes are created.

    Returns
    -------
    tuple
        ``(figure, axes_mapping, created_axes)``.
    """
    panels = panels or {}
    if axes is not None:
        fig_obj = fig or next(iter(axes.values())).figure
        return fig_obj, axes, False

    use_layout = layout or default_layout(panels)
    if fig is not None:
        ax_map = fig.subplot_mosaic(use_layout, **(subplot_kwargs or {}))
        return fig, ax_map, True

    fig_obj, ax_map = plt.subplot_mosaic(
        use_layout,
        figsize=figsize,
        **(subplot_kwargs or {}),
    )
    return fig_obj, ax_map, True
