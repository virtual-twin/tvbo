"""bsplot figure codegen adapter.

Resolves a declarative ``Figure`` (see ``schema/figure.yaml``) into a codegen
context and renders the ``tvbo/templates/bsplot/`` Mako tree into a self-contained,
user-editable ``plot.py``. This is the figure sibling of the simulation adapters
(``julia_model``, ``pyrates``, …): resolution lives here in Python, code *structure*
lives in the Mako template.

``render_code(figure, base_dir)`` returns the script; ``render(...)`` emits and runs
it — mirroring ``experiment.render_code`` / ``.run``.
"""
from __future__ import annotations

import functools
import re
from pathlib import Path

from tvbo.templates import lookup


@functools.lru_cache(maxsize=None)
def _open_ds(path):
    """Open a result container once per distinct path (shared by the adapter's custom panels)."""
    import xarray as xr
    return xr.open_dataset(path, engine="h5netcdf")

_TEMPLATE = "bsplot/tvbo-bsplot-figure.py.mako"


# --------------------------------------------------------------------------- transforms
# Generic, presentation-only reductions usable as a layer `transform`. Registered here
# so the emitted plot.py can import them; a study can register its own the same way.

def up_branch(da):
    """The up-sweep half of a hysteresis scan (up to the sweep reversal at argmax of the swept coord)."""
    import numpy as np
    dim = da.dims[0]
    coord = da[dim].values if dim in da.coords else np.arange(da.sizes[dim])
    n = int(np.argmax(coord)) + 1
    return da.isel({dim: slice(0, n)})


def down_branch(da):
    """The down-sweep half of a hysteresis scan (from the reversal onward)."""
    import numpy as np
    dim = da.dims[0]
    coord = da[dim].values if dim in da.coords else np.arange(da.sizes[dim])
    n = int(np.argmax(coord)) + 1
    return da.isel({dim: slice(n - 1, None)})


def order_by_branch(da):
    """Restore a branch-restart run's source traversal order (sort by ``branch_point``).

    A paired Lyapunov/branch-following experiment records its points in solve order,
    not in the swept-coordinate order. Sorting by the ``branch_point`` ordinal realigns
    it to the hysteresis scan's up-then-down K traversal so it can be read against that
    K axis.
    """
    if "branch_point" in da.dims or "branch_point" in da.coords:
        return da.sortby("branch_point")
    return da


TRANSFORMS = {"up_branch": up_branch, "down_branch": down_branch,
              "order_by_branch": order_by_branch}


def _style_entries(figure) -> list:
    """Classify each figure style as a bsplot named style or an .mplstyle path.

    bsplot.style.use only knows its registered names; a study's own .mplstyle is a
    filesystem path, applied via matplotlib's plt.style.use instead. This lets a
    study carry its own design rules (Figure.style: ['<path>/study.mplstyle']).
    """
    styles = list(getattr(figure, "style", None) or []) or ["tvbo"]
    out = []
    for s in styles:
        s = str(s)
        is_path = s.endswith(".mplstyle") or "/" in s or "\\" in s
        out.append({"path": is_path, "value": s})
    return out


def _arg_dict(coll) -> dict:
    """Resolve an Argument collection (dict or list-of-Argument) into ``{name: value}``."""
    if not coll:
        return {}
    items = coll.items() if isinstance(coll, dict) else [(a.name, a) for a in coll]
    return {key: getattr(arg, "value", arg) for key, arg in items}


def _style_kwargs(style) -> dict:
    """Resolve a layer/panel Style into matplotlib kwargs."""
    if style is None:
        return {}
    kw: dict = {}
    if getattr(style, "color", None):
        kw["color"] = style.color
    if getattr(style, "opacity", None) is not None:
        kw["alpha"] = style.opacity
    kw.update(_arg_dict(getattr(style, "opts", None)))
    return kw


# Axis directives a grammar (cartesian/heatmap) panel may carry on ``Panel.opts``.
# Kept to a small, backend-independent set so the template can apply them uniformly;
# a ``custom`` panel routes ``opts`` to its callable instead (see ``build_context``).
_AXIS_OPTS = {
    "xlabel", "ylabel", "title", "xlim", "ylim", "xticks", "yticks",
    "hide_xticklabels", "hide_yticklabels", "axhline", "legend",
}


def _panel_opts(panel) -> dict:
    """Resolve ``Panel.opts`` (Argument dict) into a plain ``{name: value}`` dict."""
    return _arg_dict(getattr(panel, "opts", None))


def _axopts(panel) -> dict:
    """Axis-level directives for a grammar panel (labels, limits, ticks, legend).

    Draws from ``Panel.opts`` (the recognised ``_AXIS_OPTS`` keys) plus the boolean
    ``Panel.legend`` slot. This is the minimal per-panel label/limit override: the paper's
    LaTeX axis labels and shared ranges live here rather than defaulting to the bare
    variable name.
    """
    o = {k: v for k, v in _panel_opts(panel).items() if k in _AXIS_OPTS}
    if getattr(panel, "legend", None):
        o.setdefault("legend", "best")
    return o


_ANNOT_LOC = {"upper left": (0.03, 0.95), "upper right": (0.97, 0.95),
              "lower left": (0.03, 0.05), "lower right": (0.97, 0.05), "center": (0.5, 0.5)}

# Panel-number placement per corner -> kwargs for bsplot.panels.add_panel_number.
# In its coord="axes" mode the label lands at (x_shift, 1.0 + y_shift), so ha/va anchor
# the text and the shifts hug it just inside the corresponding spine.
_PANEL_NUM_LOC = {
    "upper left":  {"x_shift": 0.02, "y_shift": -0.02, "ha": "left",  "va": "top"},
    "upper right": {"x_shift": 0.98, "y_shift": -0.02, "ha": "right", "va": "top"},
    "lower left":  {"x_shift": 0.02, "y_shift": 0.02,  "ha": "left",  "va": "bottom"},
    "lower right": {"x_shift": 0.98, "y_shift": 0.02,  "ha": "right", "va": "bottom"},
}


def _annotations(panel) -> list:
    """Resolve ``Panel.annotations`` into ``[{text, x, y}]`` in axes-fraction coords."""
    out = []
    for a in (getattr(panel, "annotations", None) or []):
        loc = getattr(a, "loc", None)
        if loc in _ANNOT_LOC:
            x, y = _ANNOT_LOC[loc]
        else:
            x = a.x if getattr(a, "x", None) is not None else 0.5
            y = a.y if getattr(a, "y", None) is not None else 0.95
        out.append({"text": a.text, "x": x, "y": y})
    return out


@functools.lru_cache(maxsize=None)
def _container_path(iri, base_dir: Path) -> str:
    """Resolve an experiment IRI/key to its result container (skips ``*_network.h5``).

    The PROV ``used`` edge points at an experiment; its container lives under
    ``<base_dir>/output/nc/<exp>/``. Returns ``""`` when unresolved.
    """
    if not iri:
        return ""
    key = re.split(r"[:/#]", str(iri))[-1]          # last IRI segment (e.g. "exp-3")
    digits = re.sub(r"\D", "", key)                 # trailing experiment number
    for cand in [key, *([f"exp{digits}", f"exp-{digits}"] if digits else [])]:
        d = base_dir / "output" / "nc" / cand
        if d.is_dir():
            files = [f for f in sorted(d.glob("*.h5")) if "network" not in f.name]
            if files:
                return str(files[0].resolve())
    return ""


# --------------------------------------------------------------------------- custom panels
# The ``custom`` escape hatch: a registered ``fn(fig, ax, ctx)`` draws a bespoke sub-panel
# the grammar can't (yet) express. ``ctx`` carries the resolved layers (container paths,
# transforms, selectors already resolved by ``build_context``) plus the panel's ``opts``, so
# a callable opens the container(s) itself and draws exactly what the paper needs. A study
# registers its own the same way it registers a transform.

def _load_layer(layer: dict):
    """Open a resolved-layer dict into a DataArray (transform + selector applied)."""
    da = _open_ds(layer["container"])[layer["output"]]
    if layer.get("transform"):
        da = TRANSFORMS[layer["transform"]](da)
    if layer.get("sel"):
        da = da.sel(layer["sel"], method=layer.get("sel_method"))
    return da


def _sweep_axis(da):
    """The swept coordinate of a hysteresis scan and its reversal index ``n_up``."""
    import numpy as np
    dim = da.dims[0]
    K = np.asarray(da[dim].values) if dim in da.coords else np.arange(da.sizes[dim])
    return K, int(np.argmax(K)) + 1


def lyapunov_vs_k(fig, ax, ctx):
    """(b) largest Lyapunov exponent lambda_1 vs K, up/down branches, sampled K circled.

    Cross-experiment: ``layers[0]`` supplies the K axis + reversal (the hysteresis scan),
    ``layers[1]`` supplies lambda_1 from the paired branch-restart run, already reordered to
    the K traversal by the ``order_by_branch`` transform. This x-from-one-run / y-from-another
    merge is why the panel is a callable and not a grammar layer.
    """
    import numpy as np
    o = ctx["opts"]
    k_targets = list(o.get("k_targets", []))
    k_max = float(o.get("k_max", np.inf))
    color = o.get("color", "#ff7f0e")
    K, nup = _sweep_axis(_load_layer(ctx["layers"][0]))
    lam = np.asarray(_load_layer(ctx["layers"][1]).values).ravel()
    upK, dnK = K[:nup], K[nup:]
    um, dm = upK <= k_max, dnK <= k_max
    ax.axhline(0.0, color="0.6", lw=0.7, ls=":")            # lambda_1 = 0 stability threshold
    ax.plot(upK[um], lam[:nup][um], "-", color=color, lw=1.3, label="up-sweep")
    ax.plot(dnK[dm], lam[nup:][dm], "--", color=color, lw=1.3, label="down-sweep")
    for kt in k_targets:
        j = int(np.argmin(np.abs(upK - kt)))
        ax.plot(upK[j], lam[:nup][j], "o", mfc="none", mec="k", ms=7, mew=1.1)
    ax.set_xlim(0, k_max if np.isfinite(k_max) else float(K.max()))
    if o.get("legend"):
        ax.legend(frameon=False,
                  loc=o["legend"] if isinstance(o["legend"], str) else "upper left")
    ax.set_ylabel(o.get("ylabel", r"$\lambda_1$"))
    if o.get("xlabel"):
        ax.set_xlabel(o["xlabel"])


def node_profile(fig, ax, ctx):
    """(c-e) per-node time-mean profile <omega_i>_t at one sampled K.

    ``opts.col`` selects which of ``opts.k_targets`` this cell shows; the shared y-range is
    derived across all three so the columns are directly comparable (the paper's convention).
    """
    import numpy as np
    o = ctx["opts"]
    k_targets = list(o.get("k_targets", []))
    col = int(o.get("col", 0))
    color = o.get("color", "#1f77b4")
    da = _load_layer(ctx["layers"][0])
    K, nup = _sweep_axis(da)
    upK, vals = K[:nup], np.asarray(da.values)
    prof_at = [(float(upK[int(np.argmin(np.abs(upK - kt)))]),
                vals[int(np.argmin(np.abs(upK - kt)))]) for kt in k_targets]
    allp = np.concatenate([p for _, p in prof_at])
    lo, hi = float(allp.min()), float(allp.max())
    pad = 0.08 * ((hi - lo) or 1.0)
    kk, p = prof_at[col]
    n = p.size
    ax.plot(np.arange(n), p, ".", color=color, ms=2.2)
    if o.get("k_title"):                                     # column header, once per column (dedup)
        ax.set_title(r"$K\approx%.0f$" % kk)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(0, n)
    ax.set_xticks([1, n // 2, n])
    ax.set_xticklabels([])
    ax.tick_params(labelsize=6)
    if col == 0:
        ax.set_ylabel(o.get("ylabel", r"$\langle\omega_i\rangle_t$ (Hz)"))
    else:
        ax.set_yticklabels([])


def lyapunov_vector(fig, ax, ctx):
    """(f-h) covariant Lyapunov vector xi_i at one sampled K.

    ``layers[0]`` fixes the K axis (hysteresis scan); ``layers[1]`` is xi_i from the paired
    branch-restart run (``order_by_branch``-sorted). Shared 0-based y-range across the three K.
    """
    import numpy as np
    o = ctx["opts"]
    k_targets = list(o.get("k_targets", []))
    col = int(o.get("col", 0))
    color = o.get("color", "#ff7f0e")
    K, nup = _sweep_axis(_load_layer(ctx["layers"][0]))
    upK = K[:nup]
    xi_up = np.asarray(_load_layer(ctx["layers"][1]).values)[:nup]
    xi_at = [(float(upK[int(np.argmin(np.abs(upK - kt)))]),
              xi_up[int(np.argmin(np.abs(upK - kt)))]) for kt in k_targets]
    allx = np.concatenate([x for _, x in xi_at])
    top = (float(allx.max()) * 1.08) or 1.0
    kk, xv = xi_at[col]
    n = xv.size
    ax.plot(np.arange(n), xv, ".", color=color, ms=2.2)
    ax.set_ylim(0.0, top)
    ax.set_xlim(0, n)
    ax.set_xticks([1, n // 2, n])
    ax.set_xlabel(o.get("xlabel", r"Index $i$"))
    ax.tick_params(labelsize=6)
    if col == 0:
        ax.set_ylabel(o.get("ylabel", r"$\xi_i$"))
    else:
        ax.set_yticklabels([])


CUSTOM_PANELS = {
    "lyapunov_vs_k": lyapunov_vs_k,
    "node_profile": node_profile,
    "lyapunov_vector": lyapunov_vector,
}


def _items(coll):
    """Yield ``(key, value)`` from a keyed dict or a list of keyed objects."""
    if coll is None:
        return []
    if isinstance(coll, dict):
        return list(coll.items())
    return [(getattr(v, "panel_key", None) or getattr(v, "name", i), v)
            for i, v in enumerate(coll)]


def _sel_dict(used):
    """Resolve ``DataRef.sel`` (Argument dict) into ``({dim: value}, method)`` or ``(None, None)``.

    A numeric selection uses ``method="nearest"`` (label-based nearest coordinate, e.g. the
    sampled K-values on a continuous sweep); a non-numeric one is an exact label match.
    """
    resolved = _arg_dict(getattr(used, "sel", None))
    if not resolved:
        return None, None

    def _numeric(v):
        vals = v if isinstance(v, (list, tuple)) else [v]
        return all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in vals)

    method = "nearest" if all(_numeric(v) for v in resolved.values()) else None
    return resolved, method


def _resolve_layer(layer, panel_kind, base_dir):
    """Resolve one ``Layer`` into the flat dict the template/callables consume."""
    used, enc = layer.used, getattr(layer, "encoding", None)
    sel, method = _sel_dict(used)
    return {
        "container": _container_path(getattr(used, "iri", None), base_dir),
        "output": used.output,
        "mark": layer.mark or ("heatmap" if panel_kind == "heatmap" else "line"),
        "x": getattr(enc, "x", None),
        "y": getattr(enc, "y", None),
        "transform": getattr(layer, "transform", None),
        "sel": sel,
        "sel_method": method,
        "style": _style_kwargs(getattr(layer, "style", None)),
    }


def build_context(figure, base_dir, outfile: str) -> dict:
    """Resolve a ``Figure`` into the template context (all IO paths + names resolved)."""
    base_dir = Path(base_dir)
    panels = []
    for key, panel in _items(figure.panels):
        layers = [_resolve_layer(l, panel.kind, base_dir)
                  for l in (getattr(panel, "layers", None) or [])]
        # ``custom`` routes Panel.opts to its callable; grammar panels read the axis subset.
        ctx = ({"layers": layers, "opts": _panel_opts(panel), "key": key}
               if panel.kind == "custom" else None)
        # Default the axis labels to the first layer's x-dim / output; opts override them.
        axopts = _axopts(panel)
        if panel.kind in ("cartesian", "heatmap") and layers:
            axopts.setdefault("xlabel", layers[0]["x"] or "")
            axopts.setdefault("ylabel", layers[0]["y"] or layers[0]["output"])
        panels.append({
            "key": key,
            "kind": panel.kind,
            "title": getattr(panel, "label", None),
            "path": getattr(panel, "path", None),
            "render": getattr(panel, "render", None),
            "placeholder": getattr(panel, "placeholder", None),
            "layers": layers,
            "axopts": axopts,
            "ctx": ctx,
            "annotations": _annotations(panel),
            "number_loc": getattr(panel, "number_loc", None),
        })
    layout = (figure.layout or "".join(str(p["key"]) for p in panels) or "a")
    layout = layout.replace("/", "\n")                  # bsplot mosaics split rows on newline
    fmt = getattr(figure, "panel_number_format", None) or "{}"
    fig_loc = getattr(figure, "panel_number_loc", None)   # unset -> keep bsplot's own default placement
    font_size = getattr(figure, "font_size", None)
    for p in panels:
        p["letter"] = fmt.format(p["key"])          # the mosaic key IS the panel letter
        place = {"option": "numbers"}               # label is given verbatim, no int->letter conversion
        loc = p["number_loc"] or fig_loc
        if loc:                                     # only override placement when a corner was asked for
            place.update(_PANEL_NUM_LOC.get(loc, _PANEL_NUM_LOC["upper left"]))
        if font_size:
            place["fontsize"] = font_size
        p["number_kwargs"] = place                  # resolved here; the template just splats it

    # bsplot.figure.subplots kwargs, resolved here so the template just splats them.
    dpi = getattr(figure, "dpi", None) or 200
    subplots_kwargs = {"layout": layout, "dpi": dpi}
    width, height = getattr(figure, "width", None), getattr(figure, "height", None)
    if width and height:                            # physical size in mm -> inches
        subplots_kwargs["figsize"] = (width / 25.4, height / 25.4)
    for key in ("height_ratios", "width_ratios"):
        ratios = list(getattr(figure, key, None) or [])
        if ratios:
            subplots_kwargs[key] = ratios

    spines = getattr(figure, "spines", None)
    spine_rcparams = {}
    if spines == "box":
        spine_rcparams = {f"axes.spines.{s}": True for s in ("top", "right", "left", "bottom")}
    elif spines == "open":
        spine_rcparams = {"axes.spines.top": False, "axes.spines.right": False}

    return {
        "name": figure.name or "figure",
        "style": _style_entries(figure),
        "outfile": outfile,
        "panels": panels,
        "subplots_kwargs": subplots_kwargs,
        "spine_rcparams": spine_rcparams,
        "dpi": dpi,
        "font_size": font_size,
        "auto_format": getattr(figure, "auto_format", None) is not False,
        "panel_numbers": getattr(figure, "panel_numbers", None) is not False,
    }


def render_code(figure, base_dir=".", outfile="figure.png") -> str:
    """Render the ``Figure`` into a self-contained ``plot.py`` string."""
    ctx = build_context(figure, base_dir, outfile)
    return lookup.get_template(_TEMPLATE).render(**ctx)


def render(figure, base_dir=".", outfile="figure.png", script_path=None):
    """Render and run the figure. Writes the script to *script_path* if given."""
    code = render_code(figure, base_dir, outfile)
    if script_path:
        Path(script_path).write_text(code, encoding="utf-8")
    namespace: dict = {}
    exec(compile(code, script_path or "<figure>", "exec"), namespace)
    return namespace["main"]()
