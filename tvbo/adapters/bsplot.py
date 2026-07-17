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
    da = _open_ds(layer["container"])[layer["output"]]
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


def _resolve_layer(layer, panel_kind, base_dir):
    """Resolve one ``Layer`` into the flat dict the template/callables consume."""
    used, enc = layer.used, getattr(layer, "encoding", None)
    sel, method = _sel_dict(used)
    return {
        "container": _container_path(getattr(used, "iri", None), base_dir),
        "output": used.output,
        # str() collapses the MarkType enum (dataclass flavor) to a plain string the template compares.
        "mark": str(layer.mark) if layer.mark else ("heatmap" if panel_kind == "heatmap" else "line"),
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
        kind = str(panel.kind)                      # datamodel enum -> plain string (flavor-agnostic)
        layers = [_resolve_layer(l, kind, base_dir)
                  for l in (getattr(panel, "layers", None) or [])]
        # ``custom`` routes Panel.opts to its callable; grammar panels read the axis subset.
        # base_dir lets a panel resolve study-relative inputs (e.g. a tvbo Network yaml).
        ctx = ({"layers": layers, "opts": _panel_opts(panel), "key": key, "base_dir": str(base_dir)}
               if kind == "custom" else None)
        # Default the axis labels to the first layer's x-dim / output; opts override them.
        axopts = _axopts(panel)
        if kind in ("cartesian", "heatmap") and layers:
            axopts.setdefault("xlabel", layers[0]["x"] or "")
            axopts.setdefault("ylabel", layers[0]["y"] or layers[0]["output"])
        panels.append({
            "key": key,
            "kind": kind,
            "title": getattr(panel, "label", None),
            "path": resolve_path(getattr(panel, "path", None), base_dir),
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
            # loc is a Corner enum whose str() is the corner text in both datamodel flavors.
            place.update(_PANEL_NUM_LOC.get(str(loc), _PANEL_NUM_LOC["upper left"]))
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
