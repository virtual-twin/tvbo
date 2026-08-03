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


# --------------------------------------------------------------------------- registries
# Extension points for the layer `transform` and the `custom` panel escape hatch. Core ships
# no built-ins: a study ships figure-specific transforms/panels in its code_source module and
# decorates them with these; the emitted plot.py imports that module so the registration fires
# before lookup (see Figure.code_modules).

TRANSFORMS: dict = {}       # name -> fn(da) -> da       presentation-only layer reductions
CUSTOM_PANELS: dict = {}    # name -> fn(fig, ax, ctx)   bespoke `custom` panel drawers


def register_transform(name):
    """Register a presentation-only layer transform ``fn(da) -> da`` under *name* (decorator)."""
    def deco(fn):
        TRANSFORMS[name] = fn
        return fn
    return deco


def register_panel(name):
    """Register a ``custom``-panel callable ``fn(fig, ax, ctx)`` under *name* (decorator)."""
    def deco(fn):
        CUSTOM_PANELS[name] = fn
        return fn
    return deco


def registered(registry, name, kind):
    """Look a spec-declared name up in a registry, or raise an actionable error (public API).

    Shared by the adapter and the emitted plot.py, which imports it, so both report a miss
    the same way. The registries are empty until a figure's code_modules are imported and
    their register_* decorators run, so a miss almost always means code_modules is missing
    the module, or importing it failed.
    """
    try:
        return registry[name]
    except KeyError:
        known = ", ".join(sorted(registry)) or "none — no code_modules registered anything"
        raise KeyError(
            f"{kind} {name!r} is not registered (registered: {known}). Declare the module "
            f"that defines it in the figure's code_modules; it must be importable from here."
        ) from None


def resolve_path(p, base_dir):
    """Resolve a spec-relative file reference against *base_dir* (the study dir) — public API.

    A file a figure points at (an ``image`` panel's path, a study .mplstyle) is written
    relative to the spec that declares it, so the spec stays portable; the emitted plot.py
    runs from an arbitrary cwd and needs an absolute one. An absolute path is passed
    through untouched. Returns *p* unchanged when it is empty.

    A ``custom`` panel resolving its own study-relative input (``ctx["opts"]`` naming a
    tvbo Network yaml, say) should call this with ``ctx["base_dir"]`` so it follows the
    same rule as the rest of the spec rather than re-implementing the join.
    """
    if not p:
        return p
    p = str(p)
    return p if Path(p).is_absolute() else str((Path(base_dir) / p).resolve())


def _style_entries(figure, base_dir) -> list:
    """Classify each figure style as a bsplot named style or an .mplstyle path.

    bsplot.style.use only knows its registered names; a study's own .mplstyle is a
    filesystem path, applied via matplotlib's plt.style.use instead. This lets a
    study carry its own design rules (Figure.style: ['<path>/study.mplstyle']). Only
    the path form is resolved against base_dir — a named style is not a filesystem
    reference.
    """
    styles = list(getattr(figure, "style", None) or []) or ["tvbo"]
    out = []
    for s in styles:
        s = str(s)
        is_path = s.endswith(".mplstyle") or "/" in s or "\\" in s
        out.append({"path": is_path, "value": resolve_path(s, base_dir) if is_path else s})
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


def _heatmap_kwargs(style) -> dict:
    """Resolve a Style into pcolormesh kwargs: the colormap, opacity and raw opts.

    A field's colour scale is part of what it shows (a diverging map centred on zero for a
    correlation matrix), so ``Style.colormap`` and explicit ``vmin``/``vmax`` opts route
    here. ``Style.color`` — a line colour — is not a mesh property and is dropped.
    """
    if style is None:
        return {}
    kw: dict = {}
    if getattr(style, "colormap", None):
        kw["cmap"] = style.colormap
    if getattr(style, "opacity", None) is not None:
        kw["alpha"] = style.opacity
    kw.update(_arg_dict(getattr(style, "opts", None)))
    return kw


# Grammar-panel axis directives on ``Panel.opts`` — a small backend-independent set the template applies uniformly (a ``custom`` panel routes ``opts`` to its callable; see ``build_context``).
_AXIS_OPTS = {
    "xlabel", "ylabel", "title", "xlim", "ylim", "xticks", "yticks",
    "hide_xticklabels", "hide_yticklabels", "axhline", "axvline", "legend",
    "xscale", "yscale",   # axis scale (log/symlog/linear): part of the claim, not cosmetic
    "nbins",              # tick budget: a small multi-panel slot cannot hold the automatic count
    "aspect", "box_aspect", "invert_x", "invert_y", "frame",   # frame geometry/direction/visibility
    "zlabel", "zlim", "elev", "azim", "invert_z", "zoom",  # line3d only
}


# Axis directives the format pass can overwrite, so they are re-applied after it.
_POST_FORMAT_OPTS = {"xticks", "yticks", "xlim", "ylim", "xscale", "yscale",
                     "hide_xticklabels", "hide_yticklabels", "aspect", "box_aspect",
                     "frame", "nbins"}


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

# How much larger than the body font a panel letter is drawn when the figure does not say.
# Journals set panel letters well above the body size; matching the body size makes the
# letter read as another tick label.
_PANEL_NUMBER_SCALE = 1.6

# Panel-number placement per corner -> kwargs for bsplot.panels.add_panel_number.
# In its coord="axes" mode the label lands at (x_shift, 1.0 + y_shift), so ha/va anchor
# the text and the shifts hug it just inside the corresponding spine.
_PANEL_NUM_LOC = {
    "upper left":  {"x_shift": 0.02, "y_shift": -0.02, "ha": "left",  "va": "top"},
    "upper right": {"x_shift": 0.98, "y_shift": -0.02, "ha": "right", "va": "top"},
    "lower left":  {"x_shift": 0.02, "y_shift": 0.02,  "ha": "left",  "va": "bottom"},
    "lower right": {"x_shift": 0.98, "y_shift": 0.02,  "ha": "right", "va": "bottom"},
}


def _annotations(panel, base_dir=Path(".")) -> list:
    """Resolve ``Panel.annotations`` into ``[{text, x, y, layer}]`` in axes-fraction coords.

    An annotation with a ``used:`` binding carries its resolved layer, so the emitted
    script reads the number out of the container and formats ``text`` with it — a panel's
    printed statistic is computed from the run, never typed into the spec.
    """
    out = []
    for a in (getattr(panel, "annotations", None) or []):
        loc = getattr(a, "loc", None)
        if loc in _ANNOT_LOC:
            x, y = _ANNOT_LOC[loc]
        else:
            x = a.x if getattr(a, "x", None) is not None else 0.5
            y = a.y if getattr(a, "y", None) is not None else 0.95
        used = getattr(a, "used", None)
        layer = _resolve_layer(_UsedOnly(used), "cartesian", base_dir) if used is not None else None
        arrow = [float(v) for v in (getattr(a, "arrow", None) or [])] or None
        tail_used = getattr(a, "tail_used", None)
        tail = None
        if tail_used is not None and getattr(a, "tail_x", None) is not None:
            tail = {"x": float(a.tail_x),
                    "layer": _resolve_layer(_UsedOnly(tail_used), "cartesian", base_dir)}
        text_kwargs = {k: getattr(a, k) for k in ("rotation", "ha", "va", "size", "color")
                       if getattr(a, k, None) is not None}
        text_kwargs.setdefault("ha", "center")
        text_kwargs.setdefault("va", "center")
        out.append({"text": a.text, "x": x, "y": y, "layer": layer, "arrow": arrow,
                    "tail": tail, "kwargs": text_kwargs})
    return out


class _UsedOnly:
    """A Layer-shaped view of a bare ``DataRef``, so an annotation binding resolves through
    the one layer resolver instead of a copy of it."""

    mark = None
    encoding = None
    transform = None
    style = None
    triangle = None

    def __init__(self, used):
        self.used = used


@functools.lru_cache(maxsize=None)
def _container_path(iri, base_dir: Path) -> str:
    """Resolve an experiment IRI/key to its result container (skips ``*_network.h5``).

    The PROV ``used`` edge points at a result; its container lives under either
    ``<base_dir>/output/nc/<exp>/`` (a per-experiment ``tvbo run``), flat BIDS-style
    files directly inside ``<base_dir>/output/nc/`` (``<exp>[_desc-...]_result.h5`` —
    the layout a whole-study ``tvbo run`` writes into ``nc/``), the flat
    ``<base_dir>/output/<exp>[_desc-...]_result.h5`` at the output root, or
    ``<base_dir>/output/results/<name>/result.h5`` (a derived-figure container a
    replication study writes with ``ExperimentResult.save``, the ``results_io``
    convention). All are tried so a figure layer can bind any of them. Returns ``""``
    when unresolved.
    """
    if not iri:
        return ""
    key = re.split(r"[:/#]", str(iri))[-1]          # last IRI segment (e.g. "exp-3" or "fig3")
    # Only an experiment reference (exp-N / expN / bare N) yields exp-<id> candidates. A
    # digit-bearing but non-experiment IRI (e.g. rec-avgMatrix_atlas-HCPMMP1) must NOT be
    # misread as exp-1 — reuse the strict matcher DataRef.experiment_id already uses.
    from tvbo.data.dataref import experiment_id as _experiment_id
    eid = _experiment_id(iri)
    cands = [key, *([f"exp-{eid}", f"exp{eid}"] if eid else [])]
    nc = base_dir / "output" / "nc"
    for cand in cands:
        d = nc / cand
        if d.is_dir():
            files = [f for f in sorted(d.glob("*.h5")) if "network" not in f.name]
            if files:
                return str(files[0].resolve())
        if nc.is_dir():                              # flat BIDS files directly inside output/nc/
            files = [f for f in sorted(nc.glob(f"{cand}_*result.h5")) if "network" not in f.name]
            if files:
                return str(files[0].resolve())
        result = base_dir / "output" / "results" / cand / "result.h5"
        if result.is_file():
            return str(result.resolve())
    out = base_dir / "output"                        # flat whole-study layout: output/<exp>_*result.h5
    if out.is_dir():
        for cand in cands:                           # `_` boundary so exp-1 never matches exp-10
            files = [f for f in sorted(out.glob(f"{cand}_*result.h5")) if "network" not in f.name]
            if files:
                return str(files[0].resolve())
    return ""


# --------------------------------------------------------------------------- custom panels
# The ``custom`` escape hatch: a registered ``fn(fig, ax, ctx)`` draws a bespoke sub-panel
# the grammar can't (yet) express. ``ctx`` carries the resolved layers (container paths,
# transforms, selectors already resolved by ``build_context``) plus the panel's ``opts``, so
# a callable opens the container(s) itself and draws exactly what the paper needs. A study
# registers its own the same way it registers a transform.

def load_layer(layer: dict):
    """Open a custom panel's resolved layer into a DataArray (public API).

    A registered ``custom`` panel receives ``ctx`` with a ``layers`` list of resolved-layer
    dicts (container path, output, transform, selector — all resolved by ``build_context``);
    it calls ``bsplot.load_layer(ctx["layers"][i])`` to open the i-th one as an xarray
    ``DataArray`` with the declared ``transform`` and ``.sel`` already applied. The shared
    container cache means opening the same file across panels is free.
    """
    name = layer.get("transform")
    fn = registered(TRANSFORMS, name, "transform") if name else None   # spec error before any IO
    ds = _open_ds(layer["container"])
    from tvbo.data.dataref import match_output

    da = ds[match_output(ds.data_vars, layer["output"])]
    if fn:
        da = fn(da)
    if layer.get("sel"):
        da = da.sel(layer["sel"], method=layer.get("sel_method"))
    return da


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


def _used_ref(used):
    """The container key for a figure layer's ``used`` DataRef.

    An explicit ``iri`` pointer wins; otherwise an in-study ``experiment`` id resolves to its
    ``exp-<id>`` key and an in-study ``analysis`` to its own name (whose container
    ``_container_path`` finds under ``output/results/<name>/``). The short forms are preferred
    for same-study bindings — they need no hardcoded study key in an IRI string and (via the
    ``used`` edge) register the dependency, so the source runs first.
    """
    iri = getattr(used, "iri", None)
    if iri:
        return str(iri)
    exp = getattr(used, "experiment", None)
    if exp is not None:
        return f"exp-{exp}"
    ana = getattr(used, "analysis", None)
    return str(ana) if ana is not None else None


def _resolve_layer(layer, panel_kind, base_dir):
    """Resolve one ``Layer`` into the flat dict the template/callables consume."""
    used, enc = layer.used, getattr(layer, "encoding", None)
    sel, method = _sel_dict(used)
    # str() collapses the MarkType enum (dataclass flavor) to a plain string the template compares.
    mark = str(layer.mark) if layer.mark else ("heatmap" if panel_kind == "heatmap" else "line")
    style = getattr(layer, "style", None)
    triangle = getattr(layer, "triangle", None)
    color = getattr(enc, "color", None)
    kwargs = _heatmap_kwargs(style) if mark == "heatmap" else _style_kwargs(style)
    label = getattr(layer, "label", None)
    # A `color` ENCODING fans one artist per entry and labels each with its own coordinate
    # value, so a layer-wide colour or label would collide with the per-entry ones. Only
    # the marks the template routes through that fan-out are affected: `scatter`/`bar` keep
    # their own colour, and so does any mark drawn before the colour branch is reached.
    _fans_by_color = bool(color) and mark not in ("scatter", "bar", "area", "heatmap")
    if label and mark != "heatmap" and not _fans_by_color:
        kwargs["label"] = str(label)    # matplotlib reads the legend entry off the artist
    if _fans_by_color:
        kwargs.pop("color", None)
    return {
        "container": _container_path(_used_ref(used), base_dir),
        "output": used.output,
        "mark": mark,
        "x": getattr(enc, "x", None),
        "y": getattr(enc, "y", None),
        "z": getattr(enc, "z", None),
        "color": color,
        "cmap": getattr(style, "colormap", None),
        "transform": getattr(layer, "transform", None),
        "sel": sel,
        "sel_method": method,
        "triangle": str(triangle) if triangle else None,
        "style": kwargs,
    }


def _resolve_drawable(panel, key, base_dir) -> dict:
    """Resolve one drawable — a mosaic Panel or an Inset — into its template entry.

    An inset is a panel in everything that draws, so both go through this one function and
    the template emits both from one partial. Splitting them would let an inset's heatmap,
    triangle gap or colourbar quietly diverge from the identical panel beside it.
    """
    kind = str(panel.kind)                          # datamodel enum -> plain string (flavor-agnostic)
    layers = [_resolve_layer(l, kind, base_dir)
              for l in (getattr(panel, "layers", None) or [])]
    # ``custom`` routes opts to its callable; grammar panels read the axis subset.
    # base_dir lets a panel resolve study-relative inputs (e.g. a tvbo Network yaml).
    opts = _panel_opts(panel)
    ctx = ({"layers": layers, "opts": opts, "key": key, "base_dir": str(base_dir)}
           if kind == "custom" else None)
    # One colourbar per panel (not per layer — a split matrix is two layers, one scale),
    # suppressed with `colorbar: false` where the paper prints none. It is slim by
    # default: matplotlib's own default steals ~20% of a small panel's width.
    colorbar = (any(l["mark"] == "heatmap" for l in layers)
                and opts.get("colorbar", True) is not False)
    colorbar_kwargs = {"fraction": opts.get("colorbar_fraction", 0.046),
                       "pad": opts.get("colorbar_pad", 0.04)}
    # Default the axis labels to the first layer's x-dim / output; opts override them.
    axopts = _axopts(panel)
    # bsplot's format pass re-derives ticks and can re-normalise limits, so a DECLARED
    # frame (the paper's own tick marks and ranges) is re-applied after it. Intent
    # written in the spec must not be silently replaced by the tidy-up.
    post = {k: v for k, v in axopts.items() if k in _POST_FORMAT_OPTS}
    if kind in ("cartesian", "heatmap", "line3d") and layers:
        axopts.setdefault("xlabel", layers[0]["x"] or "")
        axopts.setdefault("ylabel", layers[0]["y"] or layers[0]["output"])
    if kind == "line3d" and layers:
        axopts.setdefault("zlabel", layers[0]["z"] or "")
    placeholder = getattr(panel, "placeholder", None)
    render = getattr(panel, "render", None)
    path = resolve_path(getattr(panel, "path", None), base_dir)
    return {
        "key": key,
        "kind": kind,
        "title": getattr(panel, "label", None),
        "path": path,
        "render": render,
        "placeholder": placeholder,
        # Nothing to draw at all: the placeholder IS the panel, not a fallback. Without
        # this the guarded draw succeeds silently (no layer raises) and the slot renders
        # as an empty 0-1 axes instead of the honest label.
        "placeholder_only": bool(placeholder) and not layers and not render and not path,
        "layers": layers,
        "colorbar": colorbar,
        "colorbar_kwargs": colorbar_kwargs,
        "colorbar_label": opts.get("colorbar_label"),
        # Cells of blank band between two triangle layers, so the halves read as two
        # quantities rather than one field (the paper's own `extra_diagonal`).
        "triangle_gap": int(opts.get("triangle_gap", 0) or 0),
        "axopts": axopts,
        "post_axopts": {} if (placeholder and not layers and not render and not path) else post,
        "ctx": ctx,
        "annotations": _annotations(panel, base_dir),
        "number_loc": getattr(panel, "number_loc", None),
        "number": getattr(panel, "number", None),
        "insets": [
            dict(_resolve_drawable(inset, f"{key}_inset{i}", base_dir),
                 bounds=[float(b) for b in (getattr(inset, "bounds", None) or [])])
            for i, inset in enumerate(getattr(panel, "insets", None) or [])
        ],
    }


def build_context(figure, base_dir, outfile: str) -> dict:
    """Resolve a ``Figure`` into the template context (all IO paths + names resolved)."""
    base_dir = Path(base_dir)
    panels = [_resolve_drawable(panel, key, base_dir)
              for key, panel in _items(figure.panels)]
    layout = (figure.layout or "".join(str(p["key"]) for p in panels) or "a")
    layout = layout.replace("/", "\n")                  # bsplot mosaics split rows on newline
    fmt = getattr(figure, "panel_number_format", None) or "{}"
    fig_loc = getattr(figure, "panel_number_loc", None)   # unset -> keep bsplot's own default placement
    font_size = getattr(figure, "font_size", None)
    number_size = getattr(figure, "panel_number_size", None) or (
        font_size * _PANEL_NUMBER_SCALE if font_size else None)
    offset = [float(v) for v in (getattr(figure, "panel_number_offset", None) or [])]
    for p in panels:
        override = p.pop("number", None)   # overrides the mosaic key; "false" suppresses the letter (many cells = one paper panel)
        if override is not None and str(override).lower() in ("false", "none", ""):
            p["letter"] = None
            p["number_kwargs"] = {}
            continue
        p["letter"] = fmt.format(override if override is not None else p["key"])
        place = {"option": "numbers"}               # label is given verbatim, no int->letter conversion
        loc = p["number_loc"] or fig_loc
        if loc:                                     # only override placement when a corner was asked for
            # loc is a Corner enum whose str() is the corner text in both datamodel flavors.
            place.update(_PANEL_NUM_LOC.get(str(loc), _PANEL_NUM_LOC["upper left"]))
        if number_size:
            place["fontsize"] = number_size
        if offset:
            place["x_shift"] = place.get("x_shift", 0.0) + offset[0]
            place["y_shift"] = place.get("y_shift", 0.0) + offset[1]
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

    # fig.savefig kwargs, resolved here so the template just splats them. Trimming re-crops
    # to content, so it is what makes a saved figure's aspect drift from the declared w×h.
    savefig_kwargs = {"dpi": dpi}
    if getattr(figure, "trim_margins", None) is not False:
        savefig_kwargs["bbox_inches"] = "tight"

    spines = getattr(figure, "spines", None)
    spine_rcparams = {}
    if spines == "box":
        spine_rcparams = {f"axes.spines.{s}": True for s in ("top", "right", "left", "bottom")}
    elif spines == "open":
        spine_rcparams = {"axes.spines.top": False, "axes.spines.right": False}
    if getattr(figure, "trim_margins", None) is False:
        # matplotlib reads `bbox_inches=None` as "use rcParams['savefig.bbox']", which the
        # stylesheet sets to "tight" — so the declared `trim_margins: false` only takes effect
        # if the rcParam itself is cleared.
        spine_rcparams = {**spine_rcparams, "savefig.bbox": None}

    offset = getattr(figure, "spine_offset", None)
    format_kwargs = {} if offset is None else {
        "shift_left_spine": -float(offset), "shift_bottom_spine": -float(offset)}

    return {
        "name": figure.name or "figure",
        "style": _style_entries(figure, base_dir),
        "outfile": outfile,
        "panels": panels,
        "subplots_kwargs": subplots_kwargs,
        "spine_rcparams": spine_rcparams,
        "dpi": dpi,
        "font_size": font_size,
        "auto_format": getattr(figure, "auto_format", None) is not False,
        "format_kwargs": format_kwargs,
        "panel_numbers": getattr(figure, "panel_numbers", None) is not False,
        "savefig_kwargs": savefig_kwargs,
        # study-shipped custom panels/transforms register when plot.py imports these
        "code_modules": [str(m) for m in (getattr(figure, "code_modules", None) or [])],
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
