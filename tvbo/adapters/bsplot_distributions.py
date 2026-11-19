"""Built-in panels for distributions: a Histogram container drawn on its own bins, and two quantities of the same entries as a joint density with each one's distribution on its margin.

Registered into :data:`tvbo.adapters.bsplot.CUSTOM_PANELS` when :mod:`tvbo.adapters.bsplot` is imported, as the surface and network panels are, so a spec reaches them with ``kind: custom`` and ``render: histogram`` or ``render: joint_density`` and needs no ``code_modules``.
"""

from __future__ import annotations

import numpy as np

from tvbo.adapters.bsplot import _open_ds, load_layer, register_panel, resolve_colormap
from tvbo.adapters.colormaps import resolve as _resolve_cmap
from tvbo.data.dataref import match_output


@register_panel("histogram")
def histogram_panel(fig, ax, ctx):
    """The counts of a ``tvbo:functions/Histogram`` container as bars that each fill their bin, the bins read from the container's own ``edge``.

    A bar width typed into the spec is a number derived from the data, wrong as soon as the range moves; the edges the analysis cut keep every bar on its bin, uneven bins included. The layer is the container's ``count``, and its style colours the bars (``edgecolor`` and ``linewidth`` in its opts draw their outlines). With ``percent: true`` each count is drawn as a percentage of the values counted, the container's ``n``.

    Raises:
        ValueError: If the container's ``edge`` does not hold one more value than the layer has counts.
    """
    opts = ctx.get("opts") or {}
    layer = ctx["layers"][0]
    count = np.asarray(load_layer(layer).values, dtype=float).ravel()
    ds = _open_ds(layer["container"])
    edge = np.asarray(ds[match_output(ds.data_vars, "edge")].values, dtype=float)
    if edge.size != count.size + 1:
        raise ValueError(
            f"histogram panel: {count.size} counts need {count.size + 1} bin edges, but the container holds {edge.size}; bind the Histogram's `count`."
        )
    if opts.get("percent"):
        count = 100.0 * count / float(ds[match_output(ds.data_vars, "n")].values)
    ax.bar(edge[:-1], count, width=np.diff(edge), align="edge", **{"linewidth": 0.0, **(layer.get("style") or {})})


def _xy(layers) -> tuple[np.ndarray, np.ndarray]:
    """The two quantities of a joint density: two layers paired by label, or one layer and its coordinate named by ``encoding.x``."""
    if len(layers) == 2:
        x, y = load_layer(layers[0]), load_layer(layers[1])
        shared = [d for d in x.dims if d in x.coords and d in y.coords]
        for d in shared:
            if sorted(map(str, x[d].values)) != sorted(map(str, y[d].values)):
                raise ValueError(
                    f"joint_density panel: the two layers label `{d}` differently, so their entries cannot be paired."
                )
        y = y.sel({d: x[d].values for d in shared}).transpose(*x.dims)
        return np.asarray(x.values, dtype=float).ravel(), np.asarray(y.values, dtype=float).ravel()
    if len(layers) == 1 and layers[0].get("x"):
        da = load_layer(layers[0])
        return np.asarray(da.coords[layers[0]["x"]].values, dtype=float).ravel(), np.asarray(da.values, dtype=float).ravel()
    raise ValueError(
        "joint_density panel: bind two layers, x and then y, or one layer whose `encoding.x` names the coordinate that holds x."
    )


def _margin(m, values: np.ndarray, kind: str, bins: int, color, sideways: bool):
    """One margin: the distribution of *values* as bars, a density curve scaled to them, or both."""
    counts, edges = np.histogram(values, bins=bins)
    if kind in ("hist", "both"):
        m.hist(
            values,
            bins=edges,
            color=color,
            alpha=0.5 if kind == "both" else 1.0,
            linewidth=0.0,
            orientation="horizontal" if sideways else "vertical",
        )
    if kind in ("kde", "both"):
        from scipy.stats import gaussian_kde

        grid = np.linspace(edges[0], edges[-1], 200)
        curve = gaussian_kde(values)(grid) * values.size * np.diff(edges)[0]
        m.plot(*((curve, grid) if sideways else (grid, curve)), color=color, linewidth=1.0)
    m.set_axis_off()


@register_panel("joint_density")
def joint_density_panel(fig, ax, ctx):
    """Two quantities of the same entries as a hexagonal-bin density, each one's distribution on its margin, and optionally a scale of counts.

    With two layers the first is x and the second y, paired by label along the dimension they share, so two arrays stored in different orders still meet entry by entry; with one layer, y is the layer and x its coordinate named by the layer's ``encoding.x``. The top margin shows x and the right margin y, in inset axes that move with the panel.

    Opts: ``gridsize``, the hexagons across the panel (default 40); ``mincnt``, the fewest points a drawn cell holds (default 1; 0 draws the empty cells in the map's lowest colour, as seaborn's hex jointplot does); ``cmap`` (else the last layer's colormap); ``log_counts``; ``marginal``, ``hist``, ``kde`` or ``both`` (default ``hist``); ``marginal_bins`` (default 50); ``marginal_color`` (default a light shade of the map); and ``marginal_size``, the margins' depth as a fraction of the panel (default 0.18). A declared ``colorbar`` slot adds a vertical scale of counts beyond the right margin, under its ``label``.
    """
    opts = ctx.get("opts") or {}
    x, y = _xy(ctx["layers"])
    cmap = _resolve_cmap(resolve_colormap(opts.get("cmap") or ctx["layers"][-1].get("cmap")))
    hexes = ax.hexbin(
        x,
        y,
        gridsize=int(opts.get("gridsize", 40)),
        cmap=cmap,
        mincnt=int(opts.get("mincnt", 1)),
        bins="log" if opts.get("log_counts") else None,
        linewidths=0.0,
    )
    size, gap = float(opts.get("marginal_size", 0.18)), 0.02
    color = opts.get("marginal_color") or hexes.cmap(0.15)
    kind, bins = str(opts.get("marginal", "hist")), int(opts.get("marginal_bins", 50))
    _margin(ax.inset_axes([0.0, 1.0 + gap, 1.0, size], sharex=ax), x, kind, bins, color, sideways=False)
    _margin(ax.inset_axes([1.0 + gap, 0.0, size, 1.0], sharey=ax), y, kind, bins, color, sideways=True)
    if opts.get("colorbar", bool(opts.get("colorbar_label"))):
        bar = ax.inset_axes([1.0 + 3 * gap + size, 0.0, 0.05, 1.0])
        fig.colorbar(hexes, cax=bar, label=str(opts.get("colorbar_label", "")))
