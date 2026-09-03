"""bsplot figure codegen adapter.

Resolves a declarative ``Figure`` (see ``schema/figure.yaml``) into a codegen context and renders the ``tvbo/templates/bsplot/`` Mako tree into a self-contained, user-editable ``plot.py``. This is the figure sibling of the simulation adapters (``julia_model``, ``pyrates``, …): resolution lives here in Python, code *structure* lives in the Mako template.

``render_code(figure, base_dir)`` returns the script; ``render(...)`` emits and runs it — mirroring ``experiment.render_code`` / ``.run``.
"""

from __future__ import annotations

import functools
import itertools
import re
from pathlib import Path

from tvbo.templates import lookup
from tvbo.utils import as_list


@functools.cache
def _open_ds(path):
    """Open a result container once per distinct path (shared by the adapter's custom panels)."""
    import xarray as xr

    return xr.open_dataset(path, engine="h5netcdf")


_TEMPLATE = "bsplot/tvbo-bsplot-figure.py.mako"


# --------------------------------------------------------------------------- registries Extension points for the layer `transform` and the `custom` panel escape hatch. Core ships no built-ins: a study ships figure-specific transforms/panels in its code_source module and decorates them with these; the emitted plot.py imports that module so the registration fires before lookup (see Figure.code_modules).

TRANSFORMS: dict = {}  # name -> fn(da) -> da       presentation-only layer reductions
CUSTOM_PANELS: dict = {}  # name -> fn(fig, ax, ctx)   bespoke `custom` panel drawers


def register_transform(name):
    """Register a presentation-only layer transform ``fn(da) -> da`` under *name* (decorator)."""

    def deco(fn):
        TRANSFORMS[name] = fn
        return fn

    return deco


def _enum_value(value):
    """The plain string behind a schema enum member, or *value* unchanged when it is already one (or None).

    The two loaders disagree on what an enum slot holds: the pydantic model (``use_enum_values``) gives the plain string, the LinkML runtime gives a permissible-value object carrying ``.text``. A value the template embeds by ``repr()`` must be collapsed here first — ``repr()`` of a permissible value is its whole constructor call, which lands in the emitted script as a ``NameError``. A value the adapter only compares or looks up needs no collapsing, since ``str()`` of a permissible value is already its text.
    """
    if value is None:
        return None
    for attr in ("text", "value"):
        if hasattr(value, attr):
            return str(getattr(value, attr))
    return str(value)


def _style_dpi(entries, fallback: int = 200) -> int:
    """The resolution a figure that declares none inherits from its style sheets.

    A spec's own `dpi:` still wins. Without one the sheet decides, because the sheet is where every other size in the figure comes from and a resolution set there was silently ignored before.
    """
    import matplotlib as mpl
    import matplotlib.style  # noqa: F401 — `mpl.style` is a submodule, and reaching it off the package alone is an AttributeError the loop below would swallow

    dpi = None
    for entry in entries or ():
        if entry.get("kind") != "mplstyle":
            continue
        try:
            with mpl.rc_context():
                mpl.style.use(entry["value"])
                found = mpl.rcParams.get("savefig.dpi")
                dpi = float(found) if isinstance(found, (int, float)) else dpi
        except Exception:
            continue
    return int(dpi) if dpi else fallback


def expand_bare_axes(fig, axes, pad: float = 0.004) -> list:
    """Grow each of *axes* until it meets a neighbour's ink, and return the ones moved.

    A layout engine sizes rows and columns uniformly and reserves each cell's margins for whichever panel in that row or column needs the most of them: ticks, tick labels, an axis label. A panel that draws none of that — a schematic, a chain diagram, a captured graph — is padded to match its neighbours and prints smaller than the slot it owns, which on a print figure is millimetres of a panel that had to fit in the first place.

    The rule is about ink rather than about cells: a panel expands in each direction until it reaches the *drawn* extent of the nearest panel that could collide with it, so it can never overlap anything and needs to know nothing about how the grid was divided. Nothing shrinks, and only space no one draws in is taken. Panels are grown one at a time against where the previous one now ends, so two panels sharing a gap cannot both claim it. The layout engine is switched off first, since it would recompute these positions away on the next draw.

    Opt-in per panel (``opts: {fill_cell: true}``): a panel whose position carries meaning — aligned with the one above it, or sized to match a sibling — must keep the position the engine gave it.
    """
    from matplotlib.transforms import Bbox

    fig.canvas.draw()  # the engine has to settle before its result can be read or replaced
    renderer = fig.canvas.get_renderer()
    inverse = fig.transFigure.inverted()
    boxes = {ax: ax.get_tightbbox(renderer).transformed(inverse) for ax in fig.axes}
    fig.set_layout_engine("none")

    moved = []
    for ax in axes:
        if hasattr(ax, "zaxis"):
            continue
        # The engine pinned this panel's axis labels at an offset measured for the box it is about to leave; only a panel that moves needs its labels placed again from scratch.
        ax.xaxis._autolabelpos = True
        ax.yaxis._autolabelpos = True
        here = ax.get_position()
        others = [box for other, box in boxes.items() if other is not ax]
        beside = [b for b in others if b.y1 > here.y0 and b.y0 < here.y1]
        stacked = [b for b in others if b.x1 > here.x0 and b.x0 < here.x1]
        # With nothing in the way the panel stops at the figure's own ink, not at its edge: growing past everything else would enlarge the figure rather than fill it, and the panel would hang below its neighbours' axis labels.
        edge = Bbox.union(others)
        # A neighbour is a limit as soon as its ink reaches past this panel's edge, even where it already overlaps the cell: a panel whose labels hang into this one's slot must stop it here rather than be stepped over.
        left = max([b.x1 for b in beside if b.x0 < here.x0], default=edge.x0) + pad
        right = min([b.x0 for b in beside if b.x1 > here.x1], default=edge.x1) - pad
        bottom = max([b.y1 for b in stacked if b.y0 < here.y0], default=edge.y0) + pad
        top = min([b.y0 for b in stacked if b.y1 > here.y1], default=edge.y1) - pad
        left, right = min(left, here.x0), max(right, here.x1)  # never shrink: the engine's box is the floor
        bottom, top = min(bottom, here.y0), max(top, here.y1)
        if right <= left or top <= bottom:
            continue  # hemmed in on both sides: the engine's own box is already the honest one
        ax.set_position([left, bottom, right - left, top - bottom])
        if ax.get_aspect() != "auto" and ax.get_images():
            ax.set_aspect(
                "auto"
            )  # a picture holding its own aspect re-shrinks inside the box it was just given, which is the opposite of filling the cell
        # A label is placed below the lowest tick box among the axes it is aligned with, and these axes draw no ticks but still carry them. Left in place, a panel grown to the figure floor drags its neighbours' axis labels down there with it.
        ax.set_xticks([])
        ax.set_yticks([])
        boxes[ax] = Bbox([[left, bottom], [right, top]])  # the next panel grows against where this one now ends
        moved.append(ax)

    _trim_to_content(fig, moved)
    return moved


def _trim_to_content(fig, moved: list) -> None:
    """Give back any side of a grown panel that reaches past everything the figure draws.

    A panel spanning several rows is handed the margin the bottom row keeps for its axis label, and a panel hanging past every label on the page is as wrong as one printed too small. The reference is measured only now, because clearing the grown panels' ticks lets their neighbours' axis labels rise back to where they belong, and a side is never given back so far that the panel leaves the cell it was dealt.
    """
    from matplotlib.transforms import Bbox

    rest = [ax for ax in fig.axes if ax not in set(moved)]
    if not moved or not rest:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = fig.transFigure.inverted()
    content = Bbox.union([ax.get_tightbbox(renderer).transformed(inverse) for ax in rest])
    for ax in moved:
        here = ax.get_position()
        mid_x, mid_y = (here.x0 + here.x1) / 2, (here.y0 + here.y1) / 2
        left = content.x0 if here.x0 < content.x0 < mid_x else here.x0
        right = content.x1 if mid_x < content.x1 < here.x1 else here.x1
        bottom = content.y0 if here.y0 < content.y0 < mid_y else here.y0
        top = content.y1 if mid_y < content.y1 < here.y1 else here.y1
        if right > left and top > bottom:
            ax.set_position([left, bottom, right - left, top - bottom])


def _recipe_dict(spec):
    """A ``RenderSpec`` as a plain dict of the options it sets, so a capture recipe survives into the emitted plot script."""
    if spec is None:
        return None
    fields = spec if isinstance(spec, dict) else getattr(spec, "__dict__", {})
    return {k: v for k, v in fields.items() if v is not None and not k.startswith("_")} or None


def register_panel(name):
    """Register a ``custom``-panel callable ``fn(fig, ax, ctx)`` under *name* (decorator)."""

    def deco(fn):
        CUSTOM_PANELS[name] = fn
        return fn

    return deco


@functools.cache
def _read_mesh_cached(path: str, kind: str, mesh_format):
    """One parse per distinct mesh, so a grid of surfaces reads its file once."""
    import numpy as _np

    if kind == "network":
        from tvbo.classes.network import Network

        net = Network.from_file(path)
        try:
            return (
                _np.asarray(object.__getattribute__(net, "_mesh_vertices")),
                _np.asarray(object.__getattribute__(net, "_mesh_elements")),
            )
        except AttributeError:
            raise ValueError(
                f"surface panel: network {path!r} carries no mesh. Its companion needs a "
                "`mesh` group (vertices + elements); a network built from edges alone has "
                "no geometry to paint on."
            ) from None
    if path.endswith(".npz"):
        with _np.load(path) as data:
            missing = {"vertices", "faces"} - set(data)
            if missing:
                raise ValueError(
                    f"surface panel: mesh {path!r} is missing {sorted(missing)}; an npz "
                    "mesh names its arrays `vertices` (n, 3) and `faces` (m, 3)."
                )
            return _np.asarray(data["vertices"]), _np.asarray(data["faces"])
    from tvbo.data.mesh_io import read_mesh

    return read_mesh(path, mesh_format)


def _read_mesh(path: str, kind: str, mesh_format):
    """Per-caller mesh: the parse is cached (once per file), but each caller gets its OWN arrays. The cached tuple must never be handed out directly — a grid of surfaces would then share one geometry, and any in-place consumer (recenter/normalize) would corrupt it for every later cell. Copying outside the cache (not freezing) also avoids aliasing a network's own live ``_mesh_vertices`` array."""
    vertices, faces = _read_mesh_cached(path, kind, mesh_format)
    return vertices.copy(), faces.copy()


def _surface_mesh(ctx):
    """``(vertices, faces)`` for a ``surface`` panel, from whichever source it declares.

    Four tvbo-native sources: a tvbo ``Network`` whose companion carries a ``mesh`` group (geometry belongs to the network, and ``network_io`` already writes it), a surface mesh FILE in any format :mod:`tvbo.data.mesh_io` reads — the GIFTI/VTK/FreeSurfer that ``Mesh.mesh_file`` has always declared — an ``.npz`` holding ``vertices``/``faces`` (what an analysis emits when the mesh is derived rather than measured), or a named ``template:`` surface, fetched rather than committed so a parcellated map needs no mesh checked in beside the page.
    """
    opts, base = ctx.get("opts", {}), ctx.get("base_dir")
    net_path, mesh_path = opts.get("network"), opts.get("mesh")
    if net_path:
        return _read_mesh(str(resolve_path(net_path, base)), "network", None)
    if mesh_path:
        return _read_mesh(str(resolve_path(mesh_path, base)), "file", opts.get("mesh_format"))
    if opts.get("template"):
        import numpy as _np
        from bsplot.data.surface import get_surface_geometry

        template, density = str(opts["template"]), str(opts.get("density", "164k"))
        hemi = str(opts.get("hemi", "lh"))
        if hemi != "both":
            return get_surface_geometry(template=template, hemi=hemi, density=density)
        # One mesh out of two: the template ships a hemisphere at a time, while a whole-brain dorsal view is what a paper prints. Right-hemisphere faces shift past the left's vertices, which is the order `load_atlas_labels(hemi="both")` concatenates its parcels in too, so a vertex means the same region to both.
        lv, lf = get_surface_geometry(template=template, hemi="lh", density=density)
        rv, rf = get_surface_geometry(template=template, hemi="rh", density=density)
        return _np.vstack([lv, rv]), _np.vstack([lf, rf + len(lv)])
    raise ValueError(
        "surface panel: declare where the mesh comes from — `network:` (a tvbo Network whose "
        "companion carries a mesh group), `mesh:` (a GIFTI/VTK/FreeSurfer surface, or an "
        "npz with `vertices`/`faces`), or `template:` (a named surface such as fsaverage)."
    )


def _vertex_values(da, n_vertices):
    """A layer's values on the FULL mesh, placed BY LABEL from a vertex-subset array.

    An analysis that runs on a subset of the mesh — the cortex vertices, with the medial wall cut away — returns fewer values than the mesh has vertices and carries the kept indices as a ``vertex`` coordinate. Scattering by that coordinate puts each value on its own vertex and leaves the rest NaN, which renders grey; placing them positionally would silently rotate the map into a plausible-looking wrong one.
    """
    import numpy as _np

    values = _np.asarray(da.values, dtype=float).squeeze()
    if values.ndim != 1:
        raise ValueError(
            f"surface panel: expected one value per vertex, got shape {values.shape} over "
            f"dims {tuple(da.dims)}. Add a `sel:` to the layer picking a single map "
            "(e.g. `sel: {mode: 1}`)."
        )
    if values.size == n_vertices:
        return values
    idx = da.coords["vertex"].values if "vertex" in da.coords else None
    if idx is None or _np.size(idx) != values.size:
        raise ValueError(
            f"surface panel: the layer supplies {values.size} values for a mesh of "
            f"{n_vertices} vertices and carries no `vertex` coordinate to place them by. "
            "This kind paints PER-VERTEX values; an analysis on a vertex subset must carry "
            "the kept indices, and a parcellated quantity must be mapped to vertices "
            "upstream, where the mapping is declared and testable."
        )
    idx = _np.asarray(idx, dtype=int)
    if idx.size and (idx.min() < 0 or idx.max() >= n_vertices):
        raise ValueError(
            f"surface panel: the layer's `vertex` coordinate reaches vertex {int(idx.max())} on "
            f"a {n_vertices}-vertex mesh — the kept-index sidecar does not match this mesh "
            "(wrong parcellation or hemisphere?)."
        )
    full = _np.full(n_vertices, _np.nan)
    full[idx] = values
    return full


@register_panel("surface")
def surface_panel(fig, ax, ctx):
    """Per-vertex values painted on a mesh — the built-in ``kind: surface``.

    A brain map is the most-drawn panel in a network-neuroscience paper and needs no study code: the mesh is geometry the network already carries, the values are a layer like any other, and everything else is presentation. Registered here rather than shipped per study, so ``kind: surface`` works with no ``code_modules``.

    A spec declares all of this as ``surface:`` — the `Surface` class names and documents every attribute, and ``_KIND_RENAMES`` says where one reaches this callable under a different word. What is only true here:

    ``symmetric`` defaults to TRUE, because a cortical map is usually signed deviations and reading those on an off-centre scale is misleading in a way no axis label catches. ``percentile`` defaults to 100. Geometry comes from whichever of ``connectome`` / ``mesh`` / ``template`` is given (see :func:`_surface_mesh`). With no layer the panel draws the bare mesh; with ``geometry: true`` the layer supplies (V, 3) vertex COORDINATES, so what it shows is the surface a reconstruction rebuilt rather than a field living on a fixed one.
    """
    import bsplot
    import numpy as _np

    from tvbo.adapters.colormaps import resolve as _resolve_cmap

    opts, base = ctx.get("opts", {}), ctx.get("base_dir")
    verts, faces = _surface_mesh(ctx)
    layers = ctx.get("layers") or []
    geometry = bool(opts.get("geometry", False))
    if not layers and not (opts.get("color") or opts.get("edgecolor")):
        raise ValueError(
            "surface panel: needs a layer supplying the per-vertex values, or a `color:` / "
            "`edgecolor:` to draw the bare geometry."
        )

    values, grey = None, None
    if layers and geometry:
        verts = _np.asarray(load_layer(layers[0]).values, dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError(
                f"surface panel: `geometry: true` means the layer supplies (V, 3) vertex coordinates; got shape {verts.shape}."
            )
    elif layers and opts.get("atlas"):
        # A parcellated layer holds one value per REGION, so it is spread onto the vertices of the parcel it names. Placing it by name keeps a cortical map honest the way the volume kind does: the alternative, trusting array order to match the mesh atlas's own, silently rotates the map.
        da = load_layer(layers[0])
        region_values = _np.asarray(da.values, dtype=float).squeeze()
        labels = _region_labels(da)
        if labels is None:
            raise ValueError(
                "surface panel: `atlas:` paints per-region values, but the layer carries no region-label "
                "coordinate to place them by. The container must name its regions."
            )
        vertex_parcels, parcel_of = _surface_parcellation(
            str(opts["atlas"]),
            base,
            str(opts.get("surface_atlas", "Desikan2006")),
            str(opts.get("template", "fsaverage")),
            str(opts.get("density", "164k")),
            str(opts.get("hemi", "lh")),
        )
        if len(vertex_parcels) != len(verts):
            raise ValueError(
                f"surface panel: the parcellation covers {len(vertex_parcels)} vertices but the mesh has "
                f"{len(verts)} — `atlas:` places values through the surface atlas's own geometry, so `template:`, "
                "`density:` and `hemi:` must name the mesh the panel draws rather than a different one."
            )
        # A region the surface atlas does not carry is skipped rather than refused, because a cortical mesh legitimately has no parcel for a subcortical one; what is refused is a layer none of whose regions land, which is a mismatched atlas rather than a whole-brain map drawn on cortex.
        values = _np.full(len(verts), _np.nan)
        placed = 0
        for nm, value in zip(labels, region_values, strict=True):
            index = parcel_of.get(nm)
            if index is not None:
                values[vertex_parcels == index] = value
                placed += 1
        if not placed:
            raise ValueError(
                f"surface panel: none of the layer's {len(labels)} region labels are parcels of "
                f"{opts.get('surface_atlas', 'Desikan2006')!r} through atlas {str(opts['atlas'])!r} "
                f"({labels[:5]}{' …' if len(labels) > 5 else ''}). The panel would draw an empty mesh."
            )
    elif layers:
        values = _vertex_values(load_layer(layers[0]), len(verts))
        if opts.get("mask"):
            grey = ~_np.loadtxt(str(resolve_path(opts["mask"], base))).astype(bool)
            if grey.shape != (len(verts),):
                raise ValueError(
                    f"surface panel: mask {opts['mask']!r} has shape {grey.shape} but the mesh "
                    f"has {len(verts)} vertices — a per-vertex 0/1 mask needs exactly one value "
                    "per vertex (wrong parcellation or hemisphere sidecar?)."
                )
            values = _np.where(grey, _np.nan, values)  # also keeps it out of the limits

    vmin, vmax = _colour_limits(_limits_source(layers, values), opts, True)

    edges = (
        {"edgecolor": str(opts["edgecolor"]), "linewidth": float(opts.get("edge_linewidth", 0.08))}
        if opts.get("edgecolor")
        else None
    )
    bsplot.plot_surf(
        vertices=verts,
        faces=faces,
        overlay=values,
        ax=ax,
        mask=grey,
        # A merged bilateral mesh is already whole, and the backend reads `hemi` as an instruction to go find one hemisphere's geometry — which it then cannot resolve.
        hemi="lh" if str(opts.get("hemi", "lh")) == "both" else str(opts.get("hemi", "lh")),
        view=str(opts.get("view", "lateral")),
        cmap=_resolve_cmap(resolve_colormap(opts.get("cmap"))),
        vmin=vmin,
        vmax=vmax,
        color=opts.get("color"),
        faces_kwargs=edges,
    )
    ax.set_aspect("equal")  # a surface's frame is anatomy, not a coordinate system
    ax.axis("off")
    if opts.get("title"):
        ax.set_title(str(opts["title"]))


def _atlas_spec(atlas: str, base_dir) -> Path:
    """The curated terminology file an ``atlas:`` opt names, by IRI or by path."""
    from tvbo.data.registry import iri_target, resolve_iri

    return Path(resolve_iri(atlas) if iri_target(atlas) else resolve_path(atlas, base_dir))


@functools.cache
def _atlas_crosswalk(atlas: str, base_dir):
    """``{name: entity_key}`` over every name an atlas's regions answer to.

    An entity is reachable by its key, its ``name``, each ``alternateName`` and its hemisphere-qualified ``abbreviation``. One crosswalk serves both brain-map kinds, so a connectome labelled ``L.LOG`` lands on the same region whether it is painted into a volume or onto a surface — two panels of one figure cannot disagree about which region a value belongs to.
    """
    from tvbo.utils import yaml_loader

    spec = _atlas_spec(atlas, base_dir)
    entities = (yaml_loader.load_as_dict(str(spec)).get("terminology") or {}).get("entities") or {}
    names = {}
    for key, ent in entities.items():
        side = {"left": "L", "right": "R"}.get(str(ent.get("hemisphere") or ""))
        abbreviation = ent.get("abbreviation")
        aliases = [key, ent.get("name"), *(ent.get("alternateName") or [])]
        if abbreviation and side:
            aliases.append(f"{side}.{abbreviation}")
        for nm in aliases:
            if nm:
                names[str(nm)] = key
    return names, entities


@functools.cache
def _atlas_segmentation(atlas: str, base_dir):
    """``(image, {label: code})`` for a labelled volume and every name its regions answer to.

    The segmentation is the ``dseg`` raster beside the atlas terminology, and each name resolves to its entity's ``originalLookupLabel`` — the integer the raster actually carries. That is what lets a connectome labelled ``L.LOG`` paint the voxels FreeSurfer labelled ``1011`` with no per-study translation table.
    """
    import nibabel as nib

    spec = _atlas_spec(atlas, base_dir)
    raster = spec.parent / (spec.name.split(".")[0] + ".nii.gz")
    if not raster.exists():
        raise FileNotFoundError(
            f"volume panel: atlas {atlas!r} has no segmentation beside its terminology (looked for "
            f"{raster.name} in {raster.parent})."
        )
    names, entities = _atlas_crosswalk(atlas, base_dir)
    codes = {}
    for nm, key in names.items():
        code = entities[key].get("originalLookupLabel")
        if code is not None:
            codes[nm] = int(code)
    return nib.load(str(raster)), codes


@functools.cache
def _surface_parcellation(atlas: str, base_dir, surface_atlas: str, template: str, density: str, hemi: str):
    """``(vertex_labels, {node_label: parcel_index})`` for painting per-region values on a mesh.

    The surface atlas supplies one integer parcel per vertex and the region names those integers stand for; the curated terminology supplies every other name each region answers to. Joining them on the entity key is what places a value by NAME on a mesh, so a parcellated map needs neither a committed ``.annot`` beside the page nor a hand-written abbreviation table.

    Across ``hemi: both`` the surface atlas numbers its parcels from zero again in the right hemisphere, so a bare index says ``lateraloccipital`` without saying whose. The side is folded into the id, which is the difference between a bilateral map and one that paints both hemispheres with the left one's values.
    """
    import bsplot
    import numpy as _np

    SIDE_STRIDE = 1000

    def _load(h):
        return bsplot.load_atlas_labels(atlas=surface_atlas, hemi=h, template=template, density=density)

    if hemi == "both":
        left, region_names = _load("lh")
        right, _ = _load("rh")
        labels = _np.concatenate([_np.asarray(left), _np.asarray(right) + SIDE_STRIDE])
        sides = ("lh", "rh")
    else:
        labels, region_names = _load(hemi)
        labels = _np.asarray(labels)
        sides = (hemi,)

    names, _entities = _atlas_crosswalk(atlas, base_dir)
    by_key = {}
    for offset, side in enumerate(sides):
        for index, region in enumerate(region_names):
            by_key[f"ctx-{side}-{str(region)}"] = index + offset * SIDE_STRIDE
    return labels, {nm: by_key[key] for nm, key in names.items() if key in by_key}


def _check_placeable(labels, table, kind: str, atlas: str) -> None:
    """Refuse a layer carrying a region the atlas does not know.

    A name with no entry cannot be placed, and the two ways of not placing it are both silent: dropping it leaves a hole that reads as missing data, and falling back to array position paints the region next to it. Asked by the VOLUME kind, whose segmentation covers the whole brain, so any unplaceable name there is a mismatched atlas. The surface kind cannot ask it: a cortical mesh legitimately carries no parcel for a subcortical region, so it skips what it cannot place and refuses only a layer none of whose regions land.
    """
    unknown = [nm for nm in labels if nm not in table]
    if unknown:
        raise ValueError(
            f"{kind} panel: {len(unknown)} region label(s) are not in atlas {atlas!r}: "
            f"{unknown[:5]}{' …' if len(unknown) > 5 else ''}. Add these spellings as `alternateName` "
            "on the atlas entities rather than translating them per study."
        )


def _colour_limits(values, opts, symmetric_default: bool):
    """``(vmin, vmax)`` for a brain map, from whichever of them the panel did not declare.

    A declared end is a fixed scale to honour, so each is filled independently. ``symmetric`` centres the range on zero, for a map of signed deviations that an off-centre scale would misread; ``percentile`` clips the range so one outlier region cannot wash the map out, and it applies whether or not the range is centred — a one-sided quantity has outliers too.
    """
    import numpy as _np

    vmin, vmax = opts.get("vmin"), opts.get("vmax")
    if values is None or (vmin is not None and vmax is not None):
        return vmin, vmax
    finite = _np.asarray(values)[_np.isfinite(values)]
    if not finite.size:
        return vmin, vmax
    percentile = float(opts.get("percentile", 100.0))
    if opts.get("symmetric", symmetric_default):
        lim = float(_np.percentile(_np.abs(finite), percentile)) or 1.0
        lo, hi = -lim, lim
    elif percentile < 100.0:
        lo, hi = (float(v) for v in _np.percentile(finite, [100.0 - percentile, percentile]))
    else:
        return vmin, vmax
    return (lo if vmin is None else vmin), (hi if vmax is None else vmax)


def _limits_source(layers, values):
    """The values a brain map's colour limits are read from.

    A map that animates takes them from the WHOLE animated range rather than from the frame on screen, because limits recomputed per frame make the same colour mean a different number in every frame — an animation that looks like activity spreading when nothing but the scale moved. A still figure has one frame, so this is what it draws.
    """
    import numpy as _np

    layer = layers[0] if layers else None
    if not layer or layer.get("frame_dim") is None or layer.get("frame_pos") is None:
        return values
    return _np.asarray(load_layer({**layer, "frame_pos": None}).values, dtype=float)


def _region_labels(da):
    """The region name of each value in a per-region layer, or ``None`` when it carries none.

    A per-node array is written with its node labels as a coordinate, so the name of each region travels with its value. Reading them off the container is what keeps a brain map a by-label placement; a container that lost them is a defect at the writer, not something to guess around here.
    """
    for dim in ("node", "region", "label", "nodes", "regions"):
        if dim in da.coords:
            values = da.coords[dim].values
            if values.dtype.kind in "US" or values.dtype == object:
                return [str(v) for v in values]
    return None


@register_panel("volume")
def volume_panel(fig, ax, ctx):
    """Per-region values painted into a labelled volume and projected — the built-in ``kind: volume``.

    The volumetric counterpart of ``kind: surface``: same layer, same colour options, geometry from an atlas instead of a mesh. Registered here rather than shipped per study, so a glass brain needs no ``code_modules`` and no per-study index table.

    Values are placed BY LABEL through the atlas crosswalk, never by array position — a per-region array ordered differently from the atlas would otherwise paint every region with its neighbour's number and still look like a brain.

    A spec declares all of this as ``volume:`` — the `Volume` class names and documents every attribute. What is only true here: ``symmetric`` defaults to FALSE, because unlike a surface map of signed deviations a volume panel is as often a positive quantity such as a frequency; ``percentile`` defaults to 100 and ``intensity_projection`` to ``absmax``.
    """
    import bsplot
    import nibabel as nib
    import numpy as _np

    from tvbo.adapters.colormaps import resolve as _resolve_cmap

    opts, base = ctx.get("opts", {}), ctx.get("base_dir")
    source = opts.get("atlas") or opts.get("volume")
    if not source:
        raise ValueError(
            "volume panel: declare where the segmentation comes from — `atlas:` (a curated atlas) or `volume:` (a dseg file)."
        )
    layers = ctx.get("layers") or []
    if not layers:
        raise ValueError("volume panel: needs a layer supplying the per-region values.")

    img, codes = _atlas_segmentation(str(source), base)
    da = load_layer(layers[0])
    values = _np.asarray(da.values, dtype=float).squeeze()
    if values.ndim != 1:
        raise ValueError(
            f"volume panel: expected one value per region, got shape {values.shape} over dims "
            f"{tuple(da.dims)}. Add a `sel:` to the layer picking a single map."
        )
    labels = _region_labels(da)
    if labels is None:
        raise ValueError(
            "volume panel: the layer carries no region-label coordinate to place its values by. Placing "
            "them positionally would paint each region with its neighbour's value, so the container must name its regions."
        )
    _check_placeable(labels, codes, "volume", str(source))

    # One pass over the segmentation, not one per region: a table indexed by label code turns a whole-volume comparison per region into a single gather.
    seg = _np.rint(_np.asarray(img.dataobj)).astype(_np.int64, copy=False)
    lookup = _np.full(max([int(seg.max(initial=0)), *(codes[nm] for nm in labels)], default=0) + 1, _np.nan)
    for nm, value in zip(labels, values, strict=True):
        lookup[codes[nm]] = value
    painted = lookup[_np.where(seg >= 0, seg, 0)]
    painted[seg < 0] = _np.nan

    vmin, vmax = _colour_limits(_limits_source(layers, values), opts, False)

    view = str(opts.get("view", "sagittal")).lower()
    if view not in ("sagittal", "coronal", "horizontal"):
        raise ValueError(f"volume panel: view {view!r} is not one of sagittal, coronal, horizontal.")
    bsplot.glass_brain(
        nib.Nifti1Image(painted, img.affine),
        ax=ax,
        view=view,
        cmap=_resolve_cmap(resolve_colormap(opts.get("cmap"))),
        intensity_projection=str(opts.get("intensity_projection", "absmax")),
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_aspect("equal")  # a projection's frame is anatomy, not a coordinate system
    ax.axis("off")
    if opts.get("title"):
        ax.set_title(str(opts["title"]))


# --------------------------------------------------------------------------- network

# Which two of a region centre's three coordinates a projection draws. Named for the radiological plane, so a spec says which view it wants rather than which array columns to keep.
_PROJECTIONS = {"axial": (0, 1), "coronal": (0, 2), "sagittal": (1, 2)}


@functools.cache
def _network_geometry(ref: str, base_dir: Path):
    """``(centres, weights, labels)`` of the connectome a ``network`` panel draws.

    Cached because an animation redraws the panel once per frame and the connectome is the one thing in it that never moves. The reference is a curated IRI or a path to a network beside the study; a path that does not exist is not quietly retried as an IRI, so a mistyped file is an error rather than a silent fall-through to some other network.
    """
    import numpy as _np

    from tvbo.classes.network import Network

    candidate = Path(str(resolve_path(ref, base_dir)))
    if ":" in str(ref) and not candidate.exists():
        network = Network(iri=str(ref))
    elif candidate.exists():
        network = Network.from_file(str(candidate))
    else:
        raise FileNotFoundError(
            f"network panel: {ref!r} is neither a curated IRI nor a file under {base_dir}. "
            "Name the connectome the panel draws as an IRI (tvbo:DesikanKilliany) or as a path to a network beside the study."
        )
    labels = [str(nm) for nm in network.node_labels]
    centres = network.get_centers()
    coords = _np.asarray([centres[i] for i in range(len(labels))], dtype=float)
    weights = _np.asarray(network.matrix("weight"), dtype=float)
    return coords, weights, labels


@register_panel("network")
def network_panel(fig, ax, ctx):
    """The connectome as a node-link graph, its nodes coloured by a layer — the built-in ``kind: network``.

    The panel a whole-brain paper draws beside its time courses: the network the model actually ran on, with the state living on it. Registered here rather than shipped per study, so ``kind: network`` needs no ``code_modules``; the graph itself is built by bsplot's own ``create_network``, which is where the edge threshold and the node/edge attributes are defined.

    A spec declares all of this as ``network:`` — the `Graph` class names and documents every attribute, ``connectome`` being the only required one. What is only true here: ``symmetric`` defaults to FALSE, because a state variable is not a signed deviation from zero the way a brain map of differences is; ``projection`` defaults to axial, ``edge_percentile`` to 92, and ``labels`` to off, since 87 region names on one panel is a block of text rather than a figure. Edge widths scale with the weight on top of ``edge_width``, so a strong connection reads as strong. With no layer the panel draws the bare graph in ``color``.
    """
    import numpy as _np
    from bsplot.graph import create_network
    from matplotlib.collections import LineCollection

    from tvbo.adapters.colormaps import resolve as _resolve_cmap

    opts, base = ctx.get("opts", {}), Path(ctx.get("base_dir") or ".")
    ref = opts.get("network")
    if not ref:
        raise ValueError(
            "network panel: declare `network:` — the connectome whose nodes and edges the panel draws, "
            "as a curated IRI (tvbo:DesikanKilliany) or a path to a network beside the study."
        )
    coords, weights, labels = _network_geometry(str(ref), base)
    plane = str(opts.get("projection", "axial"))
    if plane not in _PROJECTIONS:
        raise ValueError(f"network panel: projection {plane!r} is not one of {sorted(_PROJECTIONS)}.")
    ix, iy = _PROJECTIONS[plane]

    values = None
    layers = ctx.get("layers") or []
    if layers:
        values = _node_values(load_layer(layers[0]), labels, str(ref))

    graph = create_network(
        {nm: tuple(coords[i]) for i, nm in enumerate(labels)},
        weights,
        labels=labels,
        threshold_percentile=float(opts.get("edge_percentile", 92.0)),
    )
    index = {nm: i for i, nm in enumerate(labels)}
    segments = [
        [(coords[index[u], ix], coords[index[u], iy]), (coords[index[v], ix], coords[index[v], iy])] for u, v in graph.edges()
    ]
    if segments:
        strength = _np.asarray([float(d.get("weight", 1.0)) for _, _, d in graph.edges(data=True)])
        span = float(strength.max()) or 1.0
        ax.add_collection(
            LineCollection(
                segments,
                colors=str(opts.get("edge_color", "#9aa7ab")),
                # Width carries the weight, so the backbone reads as the backbone; the floor keeps a weak kept edge visible.
                linewidths=float(opts.get("edge_linewidth", 0.7)) * (0.3 + 0.7 * strength / span),
                alpha=float(opts.get("edge_alpha", 0.5)),
                zorder=1,
            )
        )

    marker = {
        "s": float(opts.get("node_size", 26.0)),
        "linewidths": float(opts.get("node_linewidth", 0.3)),
        "edgecolors": str(opts.get("node_edgecolor", "#2f3437")),
        "zorder": 2,
    }
    if values is None:
        ax.scatter(coords[:, ix], coords[:, iy], c=str(opts.get("color", "#2a9d8f")), **marker)
    else:
        vmin, vmax = _colour_limits(_limits_source(layers, values), opts, False)
        ax.scatter(
            coords[:, ix],
            coords[:, iy],
            c=values,
            cmap=_resolve_cmap(resolve_colormap(opts.get("cmap"))),
            vmin=vmin,
            vmax=vmax,
            **marker,
        )
    if opts.get("labels"):
        for i, nm in enumerate(labels):
            ax.annotate(
                nm,
                (coords[i, ix], coords[i, iy]),
                fontsize=4,
                ha="center",
                va="bottom",
                xytext=(0, 4),
                textcoords="offset points",
            )

    ax.set_aspect("equal")  # a connectome's frame is anatomy, not a coordinate system
    ax.margins(0.06)
    ax.axis("off")
    if opts.get("title"):
        ax.set_title(str(opts["title"]))


def _node_values(da, labels: list[str], ref: str):
    """A layer's values in the network's own node order, placed BY LABEL.

    A container writes its node labels alongside its values, so the two orders are reconciled by name rather than trusted to agree. A layer that carries no labels is accepted only when it already has one value per node, which is the one case where position cannot be wrong.
    """
    import numpy as _np

    values = _np.asarray(da.values, dtype=float).squeeze()
    if values.ndim != 1:
        raise ValueError(
            f"network panel: expected one value per node, got shape {values.shape} over dims {tuple(da.dims)}. "
            "Add a `sel:` picking a single map, or animate the panel over the extra dimension."
        )
    names = _region_labels(da)
    if names is None:
        if values.size != len(labels):
            raise ValueError(
                f"network panel: the layer supplies {values.size} values for a network of {len(labels)} nodes "
                "and carries no node-label coordinate to place them by."
            )
        return values
    index = {nm: i for i, nm in enumerate(labels)}
    _check_placeable(names, index, "network", ref)
    placed = _np.full(len(labels), _np.nan)
    for nm, value in zip(names, values, strict=True):
        placed[index[nm]] = value
    return placed


def scale_colormap(name, lo=None, hi=None, center=None):
    """A declared colormap name as the map a field is drawn with, pinned to *center* where one is asked for.

    An undeclared map takes the theme's, and which of the theme's depends on what the field is: a centred scale is signed, so it takes ``diverging``; anything else takes ``sequential``. That is the rule the ``symmetric`` default already encodes for the limits, applied to the map as well, so a field of signed deviations is not drawn on a one-directional ramp.

    ``center`` fixes the map's neutral colour at a value — zero for a signed change, where the sign is the reading. The limits stay the data's own and the map is truncated to the half-range the data actually reaches, so a unit of change is the same colour distance either side of the centre and the scale carries no colour the field never takes.

    The mesh and the bar that keys it both resolve here. A bar built from the untruncated map would put the neutral colour at the middle of a scale whose field crosses the centre anywhere else, and every value the reader looks up would be wrong.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as _np

    from tvbo.adapters.colormaps import resolve as _resolve_cmap

    cmap = _resolve_cmap(resolve_colormap(name, "diverging" if center is not None else "sequential"))
    if center is None:
        return cmap
    base = plt.get_cmap(cmap)
    lo, hi, center = float(lo), float(hi), float(center)
    span = max(hi - center, center - lo) or 1.0
    frac = _np.linspace((lo - center + span) / (2 * span), (hi - center + span) / (2 * span), 256)
    return mcolors.ListedColormap(base(frac))


@register_panel("colorbar")
def colorbar_panel(fig, ax, ctx):
    """A colour scale occupying its own mosaic slot — the built-in ``kind: colorbar``.

    Panels that share one scale cannot each own the bar: attaching it to any one of them steals that panel's width and implies the scale is local to it. The paper puts it in an empty cell instead, and so does this.

    A spec declares all of this as ``colorbar:`` — the `Colorbar` class names and documents every attribute. This is the one panel that states its own scale: a bar attached to another panel keys the field beside it and is refused a ``colormap`` of its own. With a layer bound and no explicit limits the limits are read from the data, so the bar cannot drift from what it describes, and ``center`` resolves exactly as the mesh resolves its own, so a bar keying centred heatmaps is the scale they were drawn on. A quantity in arbitrary units is labelled at its ends (Minimum..Maximum) rather than with numbers that mean nothing.

    Returns the bar's own axes, so a declared frame (ticks, formats, label padding) lands on the scale rather than on the blanked slot behind it.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as _np

    opts = ctx.get("opts", {})
    vmin, vmax = opts.get("vmin"), opts.get("vmax")
    layers = ctx.get("layers") or []
    if (vmin is None or vmax is None) and layers:
        values = _np.asarray(load_layer(layers[0]).values, dtype=float)
        finite = values[_np.isfinite(values)]
        if finite.size:
            vmin = float(finite.min()) if vmin is None else vmin
            vmax = float(finite.max()) if vmax is None else vmax

    vertical = str(opts.get("orientation", "vertical")) == "vertical"
    frac = float(opts.get("width", 0.22))
    ax.axis("off")
    box = [0.0, 0.18, frac, 0.64] if vertical else [0.18, 0.0, 0.64, frac]
    lo, hi = float(vmin if vmin is not None else 0.0), float(vmax if vmax is not None else 1.0)
    norm = mpl.colors.Normalize(vmin=lo, vmax=hi)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=scale_colormap(opts.get("cmap"), lo, hi, opts.get("center")))
    cb = fig.colorbar(mappable, cax=ax.inset_axes(box), orientation="vertical" if vertical else "horizontal")
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(direction="in", labelsize=plt.rcParams["ytick.labelsize"])
    if opts.get("ticks") is not None:
        cb.set_ticks([float(t) for t in opts["ticks"]])
    if opts.get("ticklabels") is not None:
        cb.set_ticklabels([str(t) for t in opts["ticklabels"]])
    if opts.get("label"):
        cb.set_label(str(opts["label"]))
    return cb.ax


@register_panel("legend")
def legend_panel(fig, ax, ctx):
    """A free-standing key occupying its own mosaic slot — the built-in ``kind: legend``.

    A convention shared by several panels belongs to none of them; drawing it inside one both shrinks that panel and implies the convention is local to it. Papers put it in the grid's spare cell, which is what this kind is.

    The entries are parallel declared lists rather than one encoded string per entry, so each is a typed value the spec can validate: ``labels`` names them and ``colors`` / ``linestyles`` / ``markers`` style them, each falling back to a sensible default when shorter than ``labels``.

    A spec declares all of this as ``legend:`` — the same slot every other panel uses for its own key, because this panel IS one. The `Legend` class names and documents every attribute; here the entry lists are what it is for.
    """
    from matplotlib.lines import Line2D

    opts = ctx.get("opts", {})
    labels = [str(t) for t in (opts.get("labels") or [])]
    if not labels:
        raise ValueError("legend panel: declare the entries it names in `labels:`.")

    def _at(name, i, default):
        values = list(opts.get(name) or [])
        return values[i] if i < len(values) else default

    handles = []
    for i in range(len(labels)):
        marker = _at("markers", i, None)
        handles.append(
            Line2D(
                [],
                [],
                color=str(_at("colors", i, "k")),
                linestyle=str(_at("linestyles", i, "-")),
                **({"marker": str(marker)} if marker else {}),
            )
        )
    ax.axis("off")
    extra = {k: opts[k] for k in ("ncols", "frameon") if opts.get(k) is not None}
    ax.legend(
        handles,
        labels,
        loc=str(opts.get("loc", "center")),
        title=opts.get("title"),
        handlelength=float(opts.get("handlelength", 2.2)),
        **extra,
    )


def heatmap_orientation(da, C, x, y, nx, ny):
    """A heatmap's array as ``(y, x)``, the order ``pcolormesh`` reads (public API).

    Decided by DIM NAME whenever the encoding names dims of the array. A SQUARE grid makes the two orientations indistinguishable by shape, so a shape test transposes half of them at random — silently swapping which axis the field varies along, which is a wrong figure rather than an ugly one. Falls back to the shape test only when the encoded channels are not dims of the array (a matrix addressed by index), where names cannot decide it.

    Shared with the emitted plot.py, which imports it: the orientation is a keying decision, so it lives beside the other reference resolvers rather than being inlined per script.

    Args:
        da: The layer's DataArray, consulted for its dim ORDER only.
        C: Its values.
        x: Name encoded on the x channel.
        y: Name encoded on the y channel.
        nx: Length of the x coordinate, for the fallback.
        ny: Length of the y coordinate, for the fallback.
    """
    dims = [str(d) for d in getattr(da, "dims", ())]
    if str(x) in dims and str(y) in dims:
        return C.T if dims.index(str(x)) < dims.index(str(y)) else C
    return C.T if C.shape == (nx, ny) else C


def registered(registry, name, kind):
    """Look a spec-declared name up in a registry, or raise an actionable error (public API).

    Shared by the adapter and the emitted plot.py, which imports it, so both report a miss the same way. The registries are empty until a figure's code_modules are imported and their register_* decorators run, so a miss almost always means code_modules is missing the module, or importing it failed.
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

    A file a figure points at (an ``image`` panel's path, a study .mplstyle) is written relative to the spec that declares it, so the spec stays portable; the emitted plot.py runs from an arbitrary cwd and needs an absolute one. An absolute path is passed through untouched. Returns *p* unchanged when it is empty.

    A ``custom`` panel resolving its own study-relative input (``ctx["opts"]`` naming a tvbo Network yaml, say) should call this with ``ctx["base_dir"]`` so it follows the same rule as the rest of the spec rather than re-implementing the join.
    """
    if not p:
        return p
    p = str(p)
    return p if Path(p).is_absolute() else str((Path(base_dir) / p).resolve())


_SHIPPED_PALETTE = "tvbo-palette"
"""A style layer spelling ``theme: {iri: tvbo:theme/default}``, kept because it reads well in a list of layers."""

_SHIPPED_THEME = "tvbo:theme/default"

COLOR_OPTS = frozenset(
    {
        "color",
        "axhline_color",
        "axvline_color",
        "axline_color",
        "region_color",
        "edge_color",
        "edgecolor",
        "node_edgecolor",
        "facecolor",
    }
)
"""Every option whose value is a colour, and so the only ones a palette role is resolved in.

An allowlist rather than a name match, because a match on ``color`` would also catch ``colorbar_label`` (display text), ``colorbar_ticks`` and six more ``colorbar_*`` options that are not colours — a project key could then silently replace a caption with a hex. ``edgecolors`` is deliberately absent: a scatter takes a sequence there, which no role can name.
"""


def resolve_color(value):
    """A declared colour as a drawn one — public API, and the only place the translation happens.

    A palette role (``highlight``) or hue (``palette.2``) resolves against the palette in force; a hex, a backend colour name, a sequence or ``None`` passes through untouched, so a spec written before the palette existed draws exactly as it did. A ``custom`` panel colouring by role should call this rather than reading the palette itself, so one rule covers the grammar and the escape hatch alike.
    """
    from tvbo.plot import palette

    return palette.as_color(value)


def resolve_colormap(value, default: str = "sequential"):
    """A declared colormap as a resolvable name — public API, and the only place the translation happens.

    A key the project's theme declares (``diverging``, or its own ``meg``) becomes whatever that key holds; anything else is left for the backend's own registry, so ``cividis`` and ``parula`` are untouched. ``None`` takes *default*, which is how a mark that names no scale lands on the project's rather than on matplotlib's.
    """
    from tvbo.plot import palette

    return palette.as_colormap(value, default)


def fan_colors(n: int) -> list:
    """Colours for a categorical line fan — public API, imported by the emitted plot script.

    A fan over a categorical dim asserts unrelated entries, and the palette's own doctrine (see ``palette.ramp``) gives unrelated entries hues rather than samples of an ordered ramp. Beyond the distinct hues a cycle would hand two categories one colour, so a larger fan falls back to even samples of the sequential scale, which stay pairwise distinct.
    """
    from tvbo.plot import palette

    hues = palette.palette()
    if n <= len(hues):
        return list(hues[:n])
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(resolve_colormap(None))
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _resolve_colors(kw: dict) -> dict:
    """*kw* with every colour-bearing option resolved against the palette."""
    return {k: (resolve_color(v) if k in COLOR_OPTS else v) for k, v in kw.items()}


def theme_spec(figure, base_dir) -> dict:
    """The look a figure declares, with the curated theme it names merged underneath it.

    Returns the whole Theme as a plain dict — colours and geometry together — because both halves travel to the same place: the colours become the palette the emitted script puts in force, the geometry becomes the rcParams applied after every style layer. The curated theme is the base under both halves, so a figure that declares nothing, and one that declares a single tick length, get the same look everywhere it did not speak.
    """
    from tvbo.plot import palette
    from tvbo.utils.yaml_loader import load_as_dict, strip_envelope

    declared = {k: v for k, v in (_plain(getattr(figure, "theme", None)) or {}).items() if v is not None}
    iri = declared.pop("iri", None) or (_SHIPPED_THEME if _names_shipped_palette(figure) or not declared else None)
    curated = {}
    if iri:
        from tvbo.data.registry import resolve_iri

        curated = strip_envelope(load_as_dict(str(resolve_iri(str(iri)))))
        curated.pop("iri", None)
    merged = {**palette.DEFAULT, **palette.DEFAULT_GEOMETRY, **curated, **declared}
    merged["colormaps"] = {
        **palette.DEFAULT["colormaps"],
        **(curated.get("colormaps") or {}),
        **(declared.get("colormaps") or {}),
    }
    return merged


def _names_shipped_palette(figure) -> bool:
    """Whether the figure asks for TVB-O's own colours through the ``tvbo-palette`` style layer."""
    return _SHIPPED_PALETTE in [str(s) for s in (getattr(figure, "style", None) or [])]


_THEME_RCPARAMS = {
    "tick_length": ("xtick.major.size", "ytick.major.size"),
    "tick_width": ("xtick.major.width", "ytick.major.width"),
    "tick_direction": ("xtick.direction", "ytick.direction"),
    "tick_pad": ("xtick.major.pad", "ytick.major.pad"),
    "minor_ticks": ("xtick.minor.visible", "ytick.minor.visible"),
    "minor_tick_length": ("xtick.minor.size", "ytick.minor.size"),
    "minor_tick_width": ("xtick.minor.width", "ytick.minor.width"),
    "axis_width": ("axes.linewidth",),
    "label_pad": ("axes.labelpad",),
    "title_pad": ("axes.titlepad",),
    "line_width": ("lines.linewidth",),
    "marker_size": ("lines.markersize",),
    "legend_frame": ("legend.frameon",),
    "legend_handle_length": ("legend.handlelength",),
    "legend_pad": ("legend.borderaxespad",),
    "grid_lines": ("axes.grid",),
}
"""Which rcParams each Theme slot fixes. The one place the spec's vocabulary meets matplotlib's, so a second backend replaces this table rather than the schema."""

_GENERIC_FONT_FAMILIES = ("sans-serif", "serif", "monospace", "cursive", "fantasy")


def theme_rcparams(theme: dict) -> dict:
    """The Theme's geometry as rcParams, applied after every style layer so a declared look always wins.

    Only what the theme states: a slot left unset is not a value, so the layer underneath keeps it.
    """
    out: dict = {}
    for slot, params in _THEME_RCPARAMS.items():
        value = theme.get(slot)
        if value is None:
            continue
        out.update(dict.fromkeys(params, value))
    faces = [str(f) for f in (theme.get("font_family") or [])]
    if faces:
        if faces[0] in _GENERIC_FONT_FAMILIES:
            out["font.family"] = faces[0]
            if len(faces) > 1:
                out[f"font.{faces[0]}"] = faces[1:]
        else:
            out["font.family"] = "sans-serif"
            out["font.sans-serif"] = faces
    return out


def _style_entries(figure, base_dir) -> list:
    """Classify each style layer as a registered bsplot style or a path to an .mplstyle.

    These are the looks TVB-O does not own: bsplot's registered styles, a journal's sheet, the rcParams of a paper being reproduced. Colour is not among them — a figure's colours come from its ``theme``, which is applied after every layer here, so a sheet cannot quietly reintroduce a cycle the project has replaced. ``tvbo-palette`` is read as the theme rather than as a layer, and is dropped here. Only the path form is resolved against base_dir; a registered name is not a filesystem reference.
    """
    styles = [s for s in (list(getattr(figure, "style", None) or []) or ["tvbo"]) if str(s) != _SHIPPED_PALETTE]
    out = []
    for s in styles:
        s = str(s)
        if s.endswith((".yaml", ".yml")):
            raise ValueError(
                f"figure style {s!r}: a palette is no longer a style layer. Declare it as the figure's theme instead "
                f"— `theme: {{iri: {_SHIPPED_THEME}}}` for TVB-O's own, or `theme: !include {s}` for this file."
            )
        kind = "mplstyle" if (s.endswith(".mplstyle") or "/" in s or "\\" in s) else "named"
        out.append({"kind": kind, "path": kind == "mplstyle", "value": resolve_path(s, base_dir) if kind == "mplstyle" else s})
    return out


def _plain(value):
    """*value* as plain Python, whatever LinkML flavor wrapped it.

    A structured opt — a colour key, a group spec — arrives as a JsonObj whose ``repr`` is ``JsonObj(...)``. The emitted script embeds opts by ``repr``, so anything that is not already a builtin becomes a NameError at render time rather than a value. One conversion at the boundary, so no consumer has to unwrap the datamodel's representation itself.
    """
    if isinstance(value, (str, bytes)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "__dict__") and not isinstance(value, (int, float, bool)):
        return {k: _plain(v) for k, v in vars(value).items() if not k.startswith("_")}
    return value


def _arg_dict(coll) -> dict:
    """Resolve an Argument collection (dict or list-of-Argument) into ``{name: value}``."""
    if not coll:
        return {}
    items = coll.items() if isinstance(coll, dict) else [(a.name, a) for a in coll]
    return {key: _plain(getattr(arg, "value", arg)) for key, arg in items}


def _style_kwargs(style) -> dict:
    """Resolve a layer/panel Style into matplotlib kwargs, colours included."""
    if style is None:
        return {}
    kw: dict = {}
    if getattr(style, "color", None):
        kw["color"] = style.color
    if getattr(style, "opacity", None) is not None:
        kw["alpha"] = style.opacity
    kw.update(_arg_dict(getattr(style, "opts", None)))
    return _resolve_colors(kw)


def _heatmap_kwargs(style) -> dict:
    """Resolve a Style into pcolormesh kwargs: the colormap, opacity and raw opts.

    A field's colour scale is part of what it shows (a diverging map centred on zero for a correlation matrix), so ``Style.colormap`` and explicit ``vmin``/``vmax`` opts route here. ``Style.color`` — a line colour — is not a mesh property and is dropped.

    ``center`` pins the map's neutral colour to a value and truncates the map to the half-range the data reaches, keeping the limits the data's own: a unit of change is the same colour distance either side of the centre, and the bar shows no colour the field never takes.
    """
    if style is None:
        return {}
    kw: dict = {}
    if getattr(style, "colormap", None):
        kw["cmap"] = style.colormap
    if getattr(style, "opacity", None) is not None:
        kw["alpha"] = style.opacity
    kw.update(_arg_dict(getattr(style, "opts", None)))
    if kw.get("cmap") is not None:
        kw["cmap"] = resolve_colormap(kw["cmap"])
    return _resolve_colors(kw)


_AXIS_SLOTS = (
    "xlabel",
    "ylabel",
    "zlabel",
    "title",
    "xlabel_pad",
    "ylabel_pad",
    "zlabel_pad",
    "xlabel_side",
    "ylabel_side",
    "xlim",
    "ylim",
    "zlim",
    "xscale",
    "yscale",
    "aspect",
    "box_aspect",
    "invert_x",
    "invert_y",
    "invert_z",
    "frame",
)
"""What a panel's axes say, how far they run and what shape they are drawn in — declared outright, beside the tick family. Each is named as the renderer's own directive, so the slot and the thing it sets are one word."""

_AXIS_ENUMS = frozenset({"xlabel_side", "ylabel_side", "xscale", "yscale"})
_AXIS_LIMITS = frozenset({"xlim", "ylim", "zlim"})
_TICK_SLOTS = (
    "tick_size",
    "tick_length",
    "tick_prune",
    "nbins",
    "xticks",
    "yticks",
    "hide_xticklabels",
    "hide_yticklabels",
    "xtick_rotation",
    "ytick_rotation",
    "xtick_format",
    "ytick_format",
    "xtick_side",
    "ytick_side",
)
"""The tick directives a panel declares outright. Named exactly as the theme's own, so a figure-wide default and a panel's override are the same word."""

_TICK_ENUMS = frozenset({"tick_prune", "xtick_format", "ytick_format", "xtick_side", "ytick_side"})

_PANEL_SLOTS = ("fill_cell", "triangle_gap")
"""Directives belonging to no family: one panel's answer to a layout question, declared on the panel itself."""

_FLAT_SLOTS = (*_TICK_SLOTS, *_AXIS_SLOTS, *_PANEL_SLOTS)
"""Every directive a panel declares as a plain value rather than inside an object. One loop resolves them all; the two sets below say which need converting on the way."""

_SLOT_ENUMS = _AXIS_ENUMS | _TICK_ENUMS
_SLOT_FLOATS = _AXIS_LIMITS | {"xticks", "yticks"}

_RULE_DECOR = ("color", "width", "dash", "label")
_REGION_DECOR = ("color", "fill", "opacity")
_RULE_DIRECTIVES = tuple(f"{k}{s}" for k in ("axhline", "axvline", "axline") for s in ("", *(f"_{d}" for d in _RULE_DECOR)))
_REGION_DIRECTIVES = ("region", *(f"region_{d}" for d in _REGION_DECOR))
_LEGEND_DIRECTIVES = ("legend", "legend_frame", "legend_columns", "legend_title")
_CAMERA_DIRECTIVES = ("elev", "azim", "zoom")

# The flat directives a grammar panel's axes are drawn from — the backend-independent set the template applies uniformly. Every one now has a declared slot behind it; the names survive as the renderer's own vocabulary, which is where the spec's words stop and matplotlib's begin.
_AXIS_OPTS = {
    *_AXIS_SLOTS,
    *_TICK_SLOTS,
    *_RULE_DIRECTIVES,
    *_REGION_DIRECTIVES,
    *_LEGEND_DIRECTIVES,
    *_CAMERA_DIRECTIVES,
}

_RESHAPED_OPTS = {
    "axhline": "rules: [{orientation: horizontal, at: <value>}]",
    "axvline": "rules: [{orientation: vertical, at: <value>}]",
    "axline": "rules: [{orientation: diagonal, at: [<slope>, <intercept>]}]",
    "axhline_color": "the `color` of the rule it belongs to",
    "axvline_color": "the `color` of the rule it belongs to",
    "axline_color": "the `color` of the rule it belongs to",
    "region": "regions: [{bounds: [x0, x1, y0, y1]}]",
    "region_color": "the `color` of the region it belongs to",
    "elev": "camera: {elevation: <deg>}",
    "azim": "camera: {azimuth: <deg>}",
    "zoom": "camera: {zoom: <factor>}",
    "legend": "legend: <corner>, or legend: {loc: <corner>, frame: <bool>, columns: <n>} — a panel slot either way",
    "legend_loc": "legend: {loc: <corner>}",
}
"""The retirements that changed a spec's SHAPE rather than a name, so there is no attribute to point at and the replacement is spelled out. Every retirement that IS a rename is derived by :func:`retired_options`.

Kept as a lookup rather than dropped silently: a retired option would otherwise be carried into the emitted script as an unknown keyword and surface as a matplotlib TypeError with nothing pointing back at the spec that wrote it.
"""


# Axis directives the format pass can overwrite, so they are re-applied after it. The whole tick family plus the frame the panel declared: what a paper prints ranges and tick marks for is intent, and a tidy-up pass must not quietly replace it.
_POST_FORMAT_OPTS = {
    *_TICK_SLOTS,
    "xlabel_side",  # re-applied beside the tick side, which also moves the label and would win alone
    "ylabel_side",
    "xlim",
    "ylim",
    "zlim",
    "xscale",
    "yscale",
    "aspect",
    "box_aspect",
    "frame",
}


_KIND_SLOTS = ("surface", "volume", "network", "grid", "colorbar")
"""The panel slots that carry a whole kind's options as one object. Ordered, so what a panel resolves to does not depend on the order its slots happen to be declared in."""

_SCALE_ATTRS = ("colormap", "vmin", "vmax", "symmetric", "percentile")
"""What a `ColorScale` answers, spliced into each kind below exactly as `is_a ColorScale` splices it in the schema."""

_KIND_ATTRS = {
    "surface": (
        *_SCALE_ATTRS,
        "connectome",
        "mesh",
        "mesh_format",
        "template",
        "density",
        "hemi",
        "view",
        "atlas",
        "surface_atlas",
        "mask",
        "geometry",
        "color",
        "edge_color",
        "edge_width",
    ),
    "volume": (*_SCALE_ATTRS, "atlas", "image", "view", "intensity_projection"),
    "network": (
        *_SCALE_ATTRS,
        "connectome",
        "projection",
        "labels",
        "color",
        "node_size",
        "node_edge_color",
        "node_edge_width",
        "edge_color",
        "edge_alpha",
        "edge_width",
        "edge_percentile",
    ),
    "grid": (
        "ncols",
        "nrows",
        "left",
        "right",
        "top",
        "bottom",
        "wspace",
        "hspace",
        "col_labels",
        "row_labels",
        "col_label_size",
        "col_label_pad",
        "row_label_rotation",
        "between",
        "trailing",
    ),
    "colorbar": (
        *_SCALE_ATTRS,
        "center",
        "show",
        "label",
        "ticks",
        "ticklabels",
        "decimals",
        "fraction",
        "pad",
        "aspect",
        "orientation",
        "location",
        "width",
    ),
    "legend": ("show", "loc", "frame", "columns", "title", "labels", "colors", "linestyles", "markers", "handle_length"),
}
"""Every attribute of the object each built-in kind's loose options became, in the schema's own words. The one list per kind: what a panel may declare, what the drawing code is handed, and what the retired spelling was are all read off it below."""

_KIND_RENAMES = {
    "surface": {"connectome": "network", "colormap": "cmap", "edge_color": "edgecolor", "edge_width": "edge_linewidth"},
    "volume": {"image": "volume", "colormap": "cmap"},
    "network": {
        "connectome": "network",
        "colormap": "cmap",
        "edge_width": "edge_linewidth",
        "node_edge_color": "node_edgecolor",
        "node_edge_width": "node_linewidth",
    },
    "grid": {},
    "colorbar": {
        "show": "colorbar",
        **{
            a: f"colorbar_{a}"
            for a in ("label", "ticks", "ticklabels", "decimals", "fraction", "pad", "aspect", "orientation", "location")
        },
    },
    "legend": {"handle_length": "handlelength", "columns": "ncols", "frame": "frameon"},
}
"""Only where a drawer spells an attribute differently from the spec. Everything absent here keeps the word the spec uses, which is most of them."""


def _directives(kind: str, renames: dict) -> dict:
    """One kind's attributes mapped to the names its drawing code reads."""
    return {attr: renames.get(attr, attr) for attr in _KIND_ATTRS[kind]}


_KIND_DIRECTIVES = {kind: _directives(kind, _KIND_RENAMES[kind]) for kind in _KIND_ATTRS}

# The standalone `kind: colorbar` panel IS the bar, so its drawer reads the unprefixed names. Same object and same words in the spec; only where they land differs.
_COLORBAR_PANEL_DIRECTIVES = _directives("colorbar", {"colormap": "cmap"})

# What the drawing code reads for a panel OF that kind, which is where the standalone bar parts company with an attached one.
_KIND_SPELLING = {**_KIND_DIRECTIVES, "colorbar": _COLORBAR_PANEL_DIRECTIVES}

_RETIRED_BY_KIND = {kind: {v: a for a, v in spelling.items()} for kind, spelling in _KIND_SPELLING.items()}
"""Each kind's retired option, and the attribute that replaced it. Derived, so a kind's vocabulary is stated once and the refusal cannot fall behind it. Per kind because `color`, `cmap` and `labels` are still perfectly good keywords for a `custom` callable and must not be refused there."""

HEMISPHERES = {"left": "lh", "right": "rh", "both": "both"}
"""The schema's hemisphere vocabulary in the surface backend's spelling. One vocabulary in the spec, whatever each backend calls it."""


def retired_options(kind: str = "") -> dict[str, tuple[str, str]]:
    """Every panel option of *kind* that is a declared slot now, mapped to the ``(slot, attribute)`` that replaced it.

    An empty ``slot`` means the option became a slot of the panel itself. This is the single statement of the renaming half of the retirement: :func:`_panel_opts` refuses from it and ``scripts/migrate_panel_marks.py`` rewrites from it, so what a spec is converted into is exactly what the renderer will accept. The retirements that are a change of SHAPE rather than of name — a rule, a region, a viewpoint — are ``_RESHAPED_OPTS``, because there is no attribute to point at.
    """
    return {
        **{name: ("", name) for name in _FLAT_SLOTS},
        **{
            option: ("colorbar", attr)
            for attr, option in _KIND_DIRECTIVES["colorbar"].items()
            if option.startswith(
                "colorbar"
            )  # only what was literally spelled `colorbar_*`: the scale half is a standalone bar's alone, and `vmin` is a perfectly good keyword anywhere else
        },
        **{option: (kind, attr) for option, attr in _RETIRED_BY_KIND.get(kind, {}).items()},
    }


def _declared_kind(panel) -> dict:
    """A panel's kind object — its surface, volume, graph, tiling or colour bar — as the flat directives its drawer reads.

    One object per kind rather than loose options, so the three mutually exclusive places a mesh comes from sit together with the settings that apply only to each, and so a misspelt attribute is a validation error instead of a default nobody was told about.
    """
    out: dict = {}
    kind = _enum_value(getattr(panel, "kind", None))
    for slot in _KIND_SLOTS:
        obj = getattr(panel, slot, None)
        if obj is None:
            continue
        renames = _COLORBAR_PANEL_DIRECTIVES if (slot == "colorbar" and kind == "colorbar") else _KIND_DIRECTIVES[slot]
        items = obj.items() if isinstance(obj, dict) else vars(obj).items()
        if slot == "colorbar" and kind != "colorbar":
            stated = [a for a in _SCALE_ATTRS if dict(items).get(a) is not None]
            if stated:
                raise ValueError(
                    f"panel colorbar: {', '.join(stated)} names a colour scale, and a bar attached to a panel keys the "
                    f"field beside it rather than a scale of its own — state it on that layer's `style:`, or use a "
                    f"standalone `kind: colorbar` panel, which does carry its own."
                )
        for attr, raw in items:
            if attr.startswith("_") or raw is None:
                continue
            if isinstance(raw, (list, tuple)):
                if not raw:
                    continue  # a multivalued slot nobody declared arrives as [], which is not a declaration of nothing
                value = [_plain(v) for v in raw]
            elif isinstance(raw, (bool, int, float)):
                value = raw
            else:
                value = _enum_value(raw)  # an enum member, whichever of the two loaders produced it
            out[renames.get(attr, attr)] = HEMISPHERES.get(value, value) if attr == "hemi" else value
    if out.get("cmap") is not None:
        out["cmap"] = resolve_colormap(out["cmap"])
    return _resolve_colors(out)


# What each kind's own options were before they became one object, per kind, because most of these words — `color`, `labels`, `cmap`, `width` — are still perfectly good keywords for a `custom` callable and must not be refused there.
def _panel_opts(panel) -> dict:
    """Resolve ``Panel.opts`` (Argument dict) into a plain ``{name: value}`` dict, refusing the retired spellings.

    The refusal is for what a person wrote. A grid cell arrives with its directives already resolved out of the slots the template and the cell declared, so it says so and is passed through — checking a machine-built dict against the retired names would reject the very spelling the resolution produces.
    """
    opts = _arg_dict(getattr(panel, "opts", None))
    kind = _enum_value(getattr(panel, "kind", None)) or ""
    renamed = {
        option: f"{slot}: {{{attr}: <value>}}" if slot else f"{attr}: <value> as a panel slot, not an option"
        for option, (slot, attr) in retired_options(kind).items()
    }
    replaced = {**_RESHAPED_OPTS, **renamed}
    retired = [] if getattr(panel, "opts_are_resolved", False) else [k for k in opts if k in replaced]
    if retired:
        lines = "\n  ".join(f"{k}: → {replaced[k]}" for k in sorted(retired))
        raise ValueError(f"panel opts {', '.join(sorted(retired))} are declared objects now, not options:\n  {lines}")
    return {**opts, **_declared_marks(panel)}


def _per_mark(marks: list, prefix: str, decor: tuple) -> dict:
    """One entry per mark for each decoration any of them states, keyed ``<prefix>_<attribute>``.

    Parallel lists rather than one value, so two marks of a family keep the different colours they were given instead of the last one winning for both. A decoration none of them states is left out entirely, so the renderer's own default stands.
    """
    out = {}
    for attr in decor:
        given = [m.get(attr) for m in marks]
        if any(v is not None for v in given):
            out[f"{prefix}_{attr}"] = [resolve_color(v) if attr == "color" and v is not None else v for v in given]
    return out


def _declared_marks(panel) -> dict:
    """A panel's declared axes, ticks, legend, rules, regions and camera as the directives the renderer draws them with.

    The spec states them outright — a rule carries the colour it is drawn in, so the two cannot be given apart; a label, a limit and a scale are slots the schema declares rather than strings in a free-form bag — and this is the one place that shape meets the flat directives the drawing code takes.

    Rules and regions come out as parallel lists, one entry per mark, so two rules of the same orientation keep the different colours they were given rather than the last one winning for both.

    A multivalued slot cannot tell "unset" from "empty": both arrive as ``[]``. Empty is therefore read as unset, because the alternative silently strips every tick from every axis of every figure whose panels declare none. A panel that wants a bare axis says ``hide_xticklabels``, which is a slot of its own.
    """
    out: dict = {}
    rules = _plain(getattr(panel, "rules", None)) or []
    for orientation, kind in (("horizontal", "axhline"), ("vertical", "axvline"), ("diagonal", "axline")):
        drawn = [r for r in rules if str(r["orientation"]) == orientation and r.get("at")]
        if not drawn:
            continue
        at = [[float(v) for v in r["at"]] for r in drawn]
        out[kind] = at if kind == "axline" else [a[0] for a in at]
        out.update(_per_mark(drawn, kind, _RULE_DECOR))
    regions = _plain(getattr(panel, "regions", None)) or []
    if regions:
        out["region"] = [[float(v) for v in r["bounds"]] for r in regions]
        out.update(_per_mark(regions, "region", _REGION_DECOR))
    for slot in _FLAT_SLOTS:
        raw = getattr(panel, slot, None)
        if raw is None or (slot in _SLOT_FLOATS and len(raw) == 0):
            continue  # a multivalued slot nobody declared arrives as [], which must not read as a declaration of nothing
        out[slot] = (
            [float(v) for v in raw] if slot in _SLOT_FLOATS else (_enum_value(raw) if slot in _SLOT_ENUMS else _plain(raw))
        )
    out.update(_declared_kind(panel))
    out.update(_declared_legend(panel))
    camera = _plain(getattr(panel, "camera", None)) or {}
    for slot, key in (("elevation", "elev"), ("azimuth", "azim"), ("zoom", "zoom")):
        if camera.get(slot) is not None:
            out[key] = float(camera[slot])
    return out


def _declared_legend(panel) -> dict:
    """A panel's ``legend`` slot as the renderer's legend directives, in whichever of its three spellings it was written.

    ``legend: true`` asks for a key wherever the backend finds room, ``legend: upper right`` fixes the corner, and the object states the rest. Absent, ``false`` and ``show: false`` all mean no key — the last so a panel can switch off what a shared inclusion turned on without deleting what it says.

    ``frame`` is deliberately absent from the result when the spec does not state it: the renderer then leaves ``legend.frameon`` to the theme, rather than boxing or unboxing a key nobody asked about.

    A standalone ``kind: legend`` panel IS the key rather than carrying one, so the same object resolves to the entries and placement its drawer reads directly. One word in the spec either way.
    """
    raw = getattr(panel, "legend", None)
    if raw is None or raw is False:
        return {}
    if isinstance(raw, str):
        spec = {"loc": raw}
    elif raw is True:
        spec = {}
    else:
        items = raw.items() if isinstance(raw, dict) else vars(raw).items()
        spec = {k: v for k, v in items if not k.startswith("_") and v is not None and v != []}
    if spec.get("show") is False:
        return {}
    if _enum_value(getattr(panel, "kind", None)) == "legend":
        return {  # the panel IS the key, so its drawer reads the entries and the placement unprefixed
            _KIND_DIRECTIVES["legend"].get(k, k): (_enum_value(v) if k == "loc" else _plain(v))
            for k, v in spec.items()
            if k != "show"
        }
    out: dict = {"legend": _enum_value(spec["loc"]) if spec.get("loc") is not None else True}
    for slot, key in (("frame", "legend_frame"), ("columns", "legend_columns"), ("title", "legend_title")):
        if spec.get(slot) is not None:
            out[key] = _plain(spec[slot])
    return out


# Axis directives whose only sensible value is a number, so a numeric-looking string is a parse accident rather than an intent. `aspect: equal` and the tick-format words are deliberately absent.
_NUMERIC_AXIS_OPTS = {
    "xlim",
    "ylim",
    "zlim",
    "region",
    "axhline",
    "axvline",
    "axline",
    "nbins",
    "tick_size",
    "tick_length",
    "xtick_rotation",
    "ytick_rotation",
    "xlabel_pad",
    "ylabel_pad",
    "zlabel_pad",
    "elev",
    "azim",
    "zoom",
}


def _as_number(v):
    """A numeric-looking string as a number, recursively through lists; anything else unchanged.

    YAML 1.1 needs a signed exponent to read a float, so ``[1.0e-4, 1.0e4]`` parses as one float and one string and matplotlib turns that axis categorical -- a declared limit then fails several layers down as "Failed to convert value(s) to axis units", naming neither the figure nor the key.
    """
    if isinstance(v, (list, tuple)):
        return type(v)(_as_number(x) for x in v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def _axopts(panel) -> dict:
    """Axis-level directives for a grammar panel (labels, limits, ticks, legend).

    Every one of them is a declared slot, resolved by :func:`_declared_marks`; ``Panel.opts`` reaches this only for a figure whose custom callable shares a name with one, and is filtered to the same recognised set. The paper's LaTeX axis labels and its shared ranges live here rather than defaulting to the bare variable name.
    """
    return {k: (_as_number(v) if k in _NUMERIC_AXIS_OPTS else v) for k, v in _panel_opts(panel).items() if k in _AXIS_OPTS}


_ANNOT_LOC = {
    "upper left": (0.03, 0.95),
    "upper right": (0.97, 0.95),
    "lower left": (0.03, 0.05),
    "lower right": (0.97, 0.05),
    "center": (0.5, 0.5),
}

# How much larger than the body font a panel letter is drawn when the figure does not say. Journals set panel letters well above the body size; matching the body size makes the letter read as another tick label.
_PANEL_NUMBER_SCALE = 1.6

# Panel-number placement per corner -> kwargs for bsplot.panels.add_panel_number. In its coord="axes" mode the label lands at (x_shift, 1.0 + y_shift), so ha/va anchor the text and the shifts hug it just inside the corresponding spine.
_PANEL_NUM_LOC = {
    "upper left": {"x_shift": 0.02, "y_shift": -0.02, "ha": "left", "va": "top"},
    "upper right": {"x_shift": 0.98, "y_shift": -0.02, "ha": "right", "va": "top"},
    "lower left": {"x_shift": 0.02, "y_shift": 0.02, "ha": "left", "va": "bottom"},
    "lower right": {"x_shift": 0.98, "y_shift": 0.02, "ha": "right", "va": "bottom"},
}


def _group_axis(opts, axis: str) -> dict | None:
    """A categorical axis whose entries fall into named groups, from ``<axis>groups`` opts.

    A paper labels 47 task contrasts as seven families, not as 47 tick labels: one name per family, centred on its block, with a rule between blocks. The same shape recurs wherever a categorical axis has structure — ROIs by system, nodes by module, subjects by cohort — so it is an axis feature rather than something a bespoke panel redraws each time.

    ``bounds`` are the cumulative COUNTS at which each group ends, which is what makes the group sizes readable off the declaration and the last bound the axis length. Entry *i* of a categorical axis is drawn centred on coordinate *i*, so the boundary after count
    *n* lies half a unit below it — the rules and the label centres carry that shift, and a bound is therefore declared as "how many", never as a plotted coordinate.
    """
    spec = opts.get(f"{axis}groups")
    if not spec:
        return None
    spec = dict(spec) if isinstance(spec, dict) else {"bounds": spec}
    bounds = [float(b) for b in (spec.get("bounds") or [])]
    labels = [str(t) for t in (spec.get("labels") or [])]
    if not bounds:
        raise ValueError(f"`{axis}groups` needs `bounds:` — the cumulative index where each group ends.")
    if labels and len(labels) != len(bounds):
        raise ValueError(
            f"`{axis}groups` has {len(bounds)} bounds and {len(labels)} labels; a group is "
            "one bound and one name, so the two lists have to be parallel."
        )
    starts = [0.0] + bounds[:-1]
    edge = float(spec.get("edge_offset", -0.5))  # count -> plotted coordinate of the gap
    return {
        "axis": axis,
        "rules": [b + edge for b in bounds[:-1]],  # interior only: the last is the axis end
        "rule_kwargs": {"color": str(spec.get("color", "k")), "linewidth": float(spec.get("linewidth", 0.6))},
        "labels": [
            # Not strict on `labels`: it is optional, and the check above already rejects a partial list.
            {"text": t, "at": (s + e) / 2.0 + edge}
            for (s, e), t in zip(zip(starts, bounds, strict=True), labels, strict=False)
        ],
        "pad": float(spec.get("pad", 0.04)),
        "kwargs": {
            "ha": "right" if axis == "y" else "center",
            "va": "center" if axis == "y" else "top",
            **{k: spec[k] for k in ("rotation", "size", "color") if k in spec},
        },
    }


def _annotations(panel, base_dir=Path(".")) -> list:
    """Resolve ``Panel.annotations`` into ``[{text, x, y, layer}]`` in axes-fraction coords.

    An annotation with a ``used:`` binding carries its resolved layer, so the emitted script reads the number out of the container and formats ``text`` with it — a panel's printed statistic is computed from the run, never typed into the spec.
    """
    out = []
    for a in getattr(panel, "annotations", None) or []:
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
            tail = {"x": float(a.tail_x), "layer": _resolve_layer(_UsedOnly(tail_used), "cartesian", base_dir)}
        text_kwargs = {k: getattr(a, k) for k in ("rotation", "ha", "va", "size", "color") if getattr(a, k, None) is not None}
        text_kwargs.setdefault("ha", "center")
        text_kwargs.setdefault("va", "center")
        out.append({"text": a.text, "x": x, "y": y, "layer": layer, "arrow": arrow, "tail": tail, "kwargs": text_kwargs})
    return out


class _UsedOnly:
    """A Layer-shaped view of a bare ``DataRef``, so an annotation binding resolves through the one layer resolver instead of a copy of it."""

    mark = None
    encoding = None
    transform = None
    style = None
    triangle = None

    def __init__(self, used):
        self.used = used


@functools.cache
def _container_path(iri, base_dir: Path) -> str:
    """Resolve a ``used`` edge's IRI/key to its result container under ``base_dir``.

    One layout: the study's results directory (:mod:`tvbo.utils.study_layout`, role ``results``) holds every container flat, ``exp-<id>[_<entities>]_result.h5`` for a run and ``ana-<name>_result.h5`` for an analysis, so a figure reads the same directory the run and the analyses wrote. The ``_`` boundary keeps ``exp-1`` from matching ``exp-10`` and the network companion is skipped by name.

    An IRI that names a study (``tvbo:exp/<study>/exp-N``) is resolved against that study's root rather than ``base_dir``, so a figure owned by a study-of-studies can read a member's run; the name is matched within the tree the referring study belongs to and an unmatched or ambiguous one raises.

    Returns ``""`` when the container is not there, which a panel declaring a ``placeholder`` relies on: the generated script draws the honest label instead of a plot, so a partially-run study still renders. A panel without a placeholder gets a named error from ``_open`` at render time. What is gone is the guessing — four candidate layouts tried in turn, which is how a figure came to read one run's experiments against another run's analyses.
    """
    if not iri:
        return ""
    from tvbo.adapters.bids import entity_value
    from tvbo.data.dataref import experiment_id as _experiment_id
    from tvbo.data.dataref import iri_scope
    from tvbo.utils.study_layout import is_network_companion, sibling_study_root, study_path

    # An IRI naming a study names a container in THAT study's results, not in the referring study's: `tvbo:exp/Jansen1995/exp-1` read from the manuscript root must not find the root's own exp-1.
    _kind, owner, _name = iri_scope(iri)
    if owner:
        base_dir = sibling_study_root(owner, base_dir) or base_dir
    key = re.split(r"[:/#]", str(iri))[-1]  # last IRI segment (e.g. "exp-3" or "fig3")
    # Only an experiment reference (exp-N / expN / bare N) yields an exp-<id> stem. A digit-bearing but non-experiment IRI (e.g. rec-avgMatrix_atlas-HCPMMP1) must NOT be misread as exp-1 — reuse the strict matcher DataRef.experiment_id already uses.
    eid = _experiment_id(iri)
    if eid:
        stems = [f"exp-{eid}"]
    else:
        # The writer's own stem comes first, built by the same `entity_value` the analysis container name is: an analysis named `abeta_transfer` is WRITTEN as `ana-abetatransfer_result.h5`, so a literal `ana-<name>` glob would never find it. The literal forms stay behind it for an IRI that is already a stem, and a key with no alphanumeric character names no container rather than raising.
        written = entity_value(key)
        stems = ([f"ana-{written}"] if written else []) + [f"ana-{key}", key]
    results = study_path("results", root=base_dir)
    for stem in stems:
        files = [f for f in sorted(results.glob(f"{stem}_*result.h5")) if not is_network_companion(f)]
        if files:
            return str(files[0].resolve())
    return ""


# --------------------------------------------------------------------------- custom panels The ``custom`` escape hatch: a registered ``fn(fig, ax, ctx)`` draws a bespoke sub-panel the grammar can't (yet) express. ``ctx`` carries the resolved layers (container paths, transforms, selectors already resolved by ``build_context``) plus the panel's ``opts``, so a callable opens the container(s) itself and draws exactly what the paper needs. A study registers its own the same way it registers a transform.


def load_layer(layer: dict):
    """Open a custom panel's resolved layer into a DataArray (public API).

    A registered ``custom`` panel receives ``ctx`` with a ``layers`` list of resolved-layer dicts (container path, output, transform, selector — all resolved by ``build_context``); it calls ``bsplot.load_layer(ctx["layers"][i])`` to open the i-th one as an xarray ``DataArray`` with the declared ``transform`` and ``.sel`` already applied. A ``Layer.transform`` runs before the selection and a ``DataRef.transform`` after it, which is the order the two slots are documented in and the only thing that distinguishes them. The shared container cache means opening the same file across panels is free.

    Under an animation the emitted script binds ``frame_dim``/``frame_pos`` onto each layer before handing the context over, so a drawer that knows nothing about animation still draws the frame the rest of the figure is at.
    """
    name, ref_name = layer.get("transform"), layer.get("ref_transform")
    fn = registered(TRANSFORMS, name, "transform") if name else None  # spec error before any IO
    ref_fn = registered(TRANSFORMS, ref_name, "transform") if ref_name else None
    ds = _open_ds(layer["container"])
    from tvbo.data.dataref import match_output

    da = ds[match_output(ds.data_vars, layer["output"])]
    if fn:
        da = fn(da)
    if layer.get("sel"):
        da = da.sel(layer["sel"], method=layer.get("sel_method"))
    if ref_fn:
        da = ref_fn(da)
    dim, pos = layer.get("frame_dim"), layer.get("frame_pos")
    if dim is not None and pos is not None and layer.get("frame") != "static" and dim in da.dims:
        da = da.isel({dim: pos})
    return da


def _items(coll):
    """Yield ``(key, value)`` from a keyed dict or a list of keyed objects."""
    if coll is None:
        return []
    if isinstance(coll, dict):
        return list(coll.items())
    return [(getattr(v, "panel_key", None) or getattr(v, "name", i), v) for i, v in enumerate(coll)]


def _sel_dict(used):
    """Resolve ``DataRef.sel`` (Argument dict) into ``({dim: value}, method)`` or ``(None, None)``.

    A numeric selection uses ``method="nearest"`` (label-based nearest coordinate, e.g. the sampled K-values on a continuous sweep); a non-numeric one is an exact label match.
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

    An explicit ``iri`` pointer wins; otherwise an in-study ``experiment`` id resolves to its ``exp-<id>`` stem and an in-study ``analysis`` to its ``ana-<name>`` one, both of which ``_container_path`` finds in the study's results directory. The short forms are preferred for same-study bindings — they need no hardcoded study key in an IRI string and (via the ``used`` edge) register the dependency, so the source runs first.
    """
    iri = getattr(used, "iri", None)
    if iri:
        return str(iri)
    exp = getattr(used, "experiment", None)
    if exp is not None:
        return f"exp-{exp}"
    ana = getattr(used, "analysis", None)
    return str(ana) if ana is not None else None


def _resolve_layer(layer, panel_kind, base_dir, animation=None):
    """Resolve one ``Layer`` into the flat dict the template/callables consume.

    Under an ``animation`` the layer also gets its frame role. Undeclared, the role is read off the data: a layer carrying the animated dimension is what moves, one that does not is the fixed backdrop. That inference is the whole reason a two-panel movie needs no per-layer annotation, and a `frame:` on the layer overrides it.
    """
    used, enc = layer.used, getattr(layer, "encoding", None)
    sel, method = _sel_dict(used)
    # str() collapses the MarkType enum (dataclass flavor) to a plain string the template compares.
    mark = str(layer.mark) if layer.mark else ("heatmap" if panel_kind == "heatmap" else "line")
    style = getattr(layer, "style", None)
    triangle = getattr(layer, "triangle", None)
    color = getattr(enc, "color", None)
    kwargs = _heatmap_kwargs(style) if mark == "heatmap" else _style_kwargs(style)
    label = getattr(layer, "label", None)
    # A `color` ENCODING means one of two things, and the mark decides which. On a `scatter` it is a THIRD QUANTITY per point — the paper's convention of shading a cloud by the variable it is not plotted against — drawn as one artist with `c=`. On a line it fans one artist per entry along the named dim, each labelled with its own coordinate value. Every other mark draws a single artist that keeps its own colour and label.
    _shades_points = bool(color) and mark == "scatter"
    _fans_by_color = bool(color) and mark not in ("scatter", "bar", "area", "heatmap", "band", "rule")
    if label and mark != "heatmap" and not _fans_by_color:
        kwargs["label"] = str(label)  # matplotlib reads the legend entry off the artist
    if _fans_by_color or _shades_points:
        kwargs.pop("color", None)  # a per-entry/per-point colour and a layer-wide one collide
    return {
        "container": _container_path(_used_ref(used), base_dir),
        "output": used.output,
        "mark": mark,
        "x": getattr(enc, "x", None),
        "y": getattr(enc, "y", None),
        "z": getattr(enc, "z", None),
        "color": color,
        "cmap": resolve_colormap(getattr(style, "colormap", None)),
        "cmap_declared": bool(getattr(style, "colormap", None)),
        "transform": getattr(layer, "transform", None),
        "ref_transform": getattr(used, "transform", None),
        "sel": sel,
        "sel_method": method,
        "triangle": str(triangle) if triangle else None,
        "style": kwargs,
        "frame": _frame_role(layer, animation),
    }


def _frame_role(layer, animation) -> str | None:
    """This layer's declared frame role, or ``None`` for a figure that does not animate.

    The role is resolved here rather than in the template so the emitted script carries a decision, not a rule to re-derive. An undeclared role stays ``None`` and the script reads it off the opened data — the dimension a layer carries is only knowable once the container is open, which is after codegen.
    """
    if animation is None:
        return None
    declared = getattr(layer, "frame", None)
    return str(declared) if declared else None


# Interior drawn by a callable or sub-axes, so its ticks must survive the format pass.
_DRAWER_KINDS = {"custom", "surface", "volume", "network", "grid", "line3d"}

# Kinds whose interior is a built-in callable, needing no `render:` and no code_modules.
_BUILTIN_PANELS = {"surface", "volume", "network", "colorbar", "legend"}


class _GridCell:
    """One cell of a ``grid`` panel: the shared ``cell:`` template with this cell's overrides.

    Shaped like a Panel so it resolves through :func:`_resolve_drawable` — a grid cell must draw exactly as the same kind draws in a mosaic slot, and a second resolution path is how the two would drift apart.
    """

    opts_are_resolved = True
    """Its ``opts`` are the directives the template's and the cell's declared slots resolved to, not what anyone wrote."""

    def __init__(self, template, cell, layers, opts):
        self.kind = getattr(cell, "kind", None) or getattr(template, "kind", None)
        self.render = getattr(cell, "render", None) or getattr(template, "render", None)
        self.path = getattr(cell, "path", None) or getattr(template, "path", None)
        self.label = getattr(cell, "label", None)
        self.layers = layers
        self.opts = opts
        self.annotations = getattr(cell, "annotations", None)
        self.legend = getattr(cell, "legend", None)
        self.placeholder = self.number = self.number_loc = self.insets = None


def _grid_geometry(opts, n_cells):
    """Cell boxes and label anchors of a ``grid``, in the host panel's axes fractions.

    Rows and columns are labelled ONCE, at the left and top, which is the whole reason a paper's composite panel is one lettered panel rather than n of them: declaring each cell as its own mosaic entry repeats the row name in every cell and renumbers panels the paper letters once. The label strips are reserved out of the drawable area (``left`` and ``top``), so the cells shrink to make room instead of being drawn over.

    A column header sits just above the cells rather than at the panel's own top: parked at a fixed fraction it would leave a dead band whenever ``top`` is opened up for something else, and stop reading as belonging to its column. ``between`` writes text in the gap BEFORE each cell, which is what turns a row of maps into a paper's decomposition equation ``y = a1 x psi_1 + a2 x psi_2 + ...``.

    An unset ``nrows`` holds every cell, so declaring only ``ncols`` wraps rather than drops. An explicit ``nrows`` still caps the grid — cropping the extras deliberately.
    ``bottom`` is ``top``'s counterpart and ``right`` is ``left``'s: cells that carry tick labels or a shared axis label need that strip reserved, or the labels are drawn outside the panel's own box — and on the right that is the panel next door.

    ``xlabel``/``ylabel`` name the quantity the cells SHARE, once, in the reserved strip — the same argument that makes row and column labels grid-level rather than per-cell, applied to the axes. They are what a paper prints under and beside such a block instead of repeating one axis title ten times, and they are inert as per-cell opts because a grid's own axes is turned off. ``ylabel_side: right`` puts the label opposite ``left``, which is where it belongs when the cells carry their ticks on the right. Both sit at the strip's OUTER edge, because the strip also holds the tick labels of the cells nearest it and an axis title anchored just outside the cells is drawn straight over them.

    Every fraction is of the HOST PANEL, cells included: a cell is ``cw - wspace`` wide, so a ``wspace`` approaching ``cw`` collapses the cells rather than merely separating them.
    """
    rows = list(opts.get("row_labels") or [])
    cols = list(opts.get("col_labels") or [])
    ncols = int(opts.get("ncols", n_cells) or 1)
    nrows = int(opts.get("nrows") or -(-n_cells // ncols))
    wspace, hspace = float(opts.get("wspace", 0.02)), float(opts.get("hspace", 0.02))
    left = float(opts.get("left", 0.16 if rows else 0.0))
    right = float(opts.get("right", 0.0))
    top = float(opts.get("top", 0.10 if cols else 0.0))
    bottom = float(opts.get("bottom", 0.0))
    cw, ch = (1.0 - left - right) / ncols, (1.0 - top - bottom) / nrows

    boxes = []
    for n in range(n_cells):
        r, c = divmod(n, ncols)
        if r >= nrows:
            break
        boxes.append([left + c * cw + wspace / 2, 1.0 - top - (r + 1) * ch + hspace / 2, cw - wspace, ch - hspace])

    def _text(text, x, y, **kw):
        kwargs = {"ha": "center", "va": "center", **kw}
        return {"text": str(text), "x": x, "y": y, "layer": None, "arrow": None, "tail": None, "kwargs": kwargs}

    labels = []
    _col_size = {"size": float(opts["col_label_size"])} if opts.get("col_label_size") else {}
    for c, text in enumerate(cols[:ncols]):
        labels.append(
            _text(text, left + (c + 0.5) * cw, 1.0 - top + float(opts.get("col_label_pad", 0.012)), va="bottom", **_col_size)
        )
    rotation = float(opts.get("row_label_rotation", 0.0))
    for r, text in enumerate(rows[:nrows]):
        labels.append(
            _text(
                text,
                left * (0.55 if rotation else 0.9),
                1.0 - top - (r + 0.5) * ch,
                ha="center" if rotation else "right",
                **({"rotation": rotation} if rotation else {}),
            )
        )
    for n, text in enumerate(list(opts.get("between") or [])[:n_cells]):
        if str(text).strip():
            r, c = divmod(n, ncols)
            labels.append(_text(text, left + c * cw, 1.0 - top - (r + 0.5) * ch))
    if opts.get("trailing"):
        r, c = divmod(n_cells - 1, ncols)
        labels.append(_text(opts["trailing"], min(left + (c + 1) * cw + wspace / 4, 0.99), 1.0 - top - (r + 0.5) * ch))
    if opts.get("xlabel"):
        labels.append(_text(opts["xlabel"], left + ncols * cw / 2, float(opts.get("xlabel_pad", 0.01)), va="bottom"))
    if opts.get("ylabel"):
        pad = float(opts.get("ylabel_pad", 0.01))
        on_right = str(opts.get("ylabel_side", "left")) == "right"
        labels.append(_text(opts["ylabel"], 1.0 - pad if on_right else pad, 1.0 - top - nrows * ch / 2, rotation=90))
    return boxes, labels


def _grid_cells(panel, key, base_dir, opts) -> tuple:
    """Resolve a ``grid`` panel into positioned cell drawables plus its label annotations.

    Cells come either from ``cells:`` (one entry each, with its own layers) or from the panel's ``layers:``, one cell per layer drawn by the shared ``cell:`` template.

    An opt named ``row.<name>`` or ``col.<name>`` supplies ``<name>`` one value per row or column, which is how an option that belongs to the ROW — the same frames shown laterally and then medially — is declared once instead of repeated in every cell of it.
    """
    template = getattr(panel, "cell", None)
    template_opts = _panel_opts(template) if template is not None else {}
    declared = list(getattr(panel, "cells", None) or [])
    layers = list(getattr(panel, "layers", None) or [])
    if declared and layers:
        raise ValueError(
            f"grid panel {key!r}: declare either `cells:` (a cell each, with its own "
            "layers) or `layers:` (one cell per layer, all drawn by `cell:`) — not both."
        )
    if not declared and not layers:
        raise ValueError(f"grid panel {key!r}: nothing to draw. Give it `layers:` (one cell each) or `cells:`.")
    if template is None and not declared:
        raise ValueError(
            f"grid panel {key!r}: `layers:` fills the grid with cells drawn by `cell:`, "
            "which declares what kind of cell they are; it is missing."
        )

    ncols = int(opts.get("ncols", len(declared or layers)) or 1)
    row_opts = {k.split(".", 1)[1]: v for k, v in opts.items() if k.startswith("row.")}
    col_opts = {k.split(".", 1)[1]: v for k, v in opts.items() if k.startswith("col.")}
    cells = []
    for n, item in enumerate(declared or layers):
        r, c = divmod(n, ncols)
        merged = dict(template_opts)
        merged.update({k: v[min(r, len(v) - 1)] for k, v in row_opts.items() if len(v)})
        merged.update({k: v[min(c, len(v) - 1)] for k, v in col_opts.items() if len(v)})
        cell = item if declared else None
        merged.update(_panel_opts(cell) if cell is not None else {})
        cell_layers = list(getattr(cell, "layers", None) or []) if declared else [item]
        cells.append(_GridCell(template, cell, cell_layers, merged))

    boxes, labels = _grid_geometry(opts, len(cells))
    resolved = [
        dict(_resolve_drawable(cell, f"{key}_cell{i}", base_dir), bounds=box)
        for i, (cell, box) in enumerate(zip(cells, boxes, strict=False))  # nrows crops: fewer boxes than cells is legal
    ]
    return resolved, labels


def _inset_bounds(inset, key, i) -> list:
    """An inset's declared ``[x0, y0, w, h]``, which only a grid cell may leave unset."""
    bounds = [float(b) for b in (getattr(inset, "bounds", None) or [])]
    if len(bounds) != 4:
        raise ValueError(
            f"panel {key!r} inset {i}: `bounds:` must be [x0, y0, width, height] in host-axes "
            f"fractions, got {bounds or 'nothing'}. Only a `grid` cell may omit them — there "
            "the grid computes the position from the row and column the cell lands in."
        )
    return bounds


def _resolve_drawable(panel, key, base_dir, animation=None) -> dict:
    """Resolve one drawable — a mosaic Panel or an Inset — into its template entry.

    An inset is a panel in everything that draws, so both go through this one function and the template emits both from one partial. Splitting them would let an inset's heatmap, triangle gap or colourbar quietly diverge from the identical panel beside it.
    """
    kind = str(panel.kind)  # datamodel enum -> plain string (flavor-agnostic)
    opts = _panel_opts(panel)
    # A grid's `layers:` belong to its cells, so they are not also drawn on the host axes.
    cells, cell_labels = _grid_cells(panel, key, base_dir, opts) if kind == "grid" else ([], [])
    layers = (
        []
        if kind == "grid"
        else [_resolve_layer(layer, kind, base_dir, animation) for layer in (getattr(panel, "layers", None) or [])]
    )
    # A callable-drawn kind gets the whole opts dict; grammar panels read the axis subset.
    ctx = (
        {
            "layers": layers,
            "opts": opts,
            "key": key,
            "base_dir": str(base_dir),
            "path": resolve_path(getattr(panel, "path", None), base_dir),
            "source": getattr(panel, "source", None),
            "capture": _recipe_dict(getattr(panel, "capture", None)),
        }
        if kind == "custom" or kind in _BUILTIN_PANELS
        else None
    )
    # One colourbar per panel (not per layer — a split matrix is two layers, one scale), suppressed with `colorbar: false` where the paper prints none. It is slim by default: matplotlib's own default steals ~20% of a small panel's width. A heatmap is unreadable without its scale, so it carries one by default; a scatter shaded by a third quantity still reads as a scatter, and a row of them conventionally shares ONE bar, so that case opts in with `colorbar: true`.
    _mappable = any(layer["mark"] == "heatmap" or (layer["mark"] == "scatter" and layer["color"]) for layer in layers)
    _default_on = any(layer["mark"] == "heatmap" for layer in layers)
    colorbar = _mappable and bool(opts.get("colorbar", _default_on))
    colorbar_kwargs = {"fraction": opts.get("colorbar_fraction", 0.046), "pad": opts.get("colorbar_pad", 0.04)}
    # A scale drawn under its own x-axis, the way a scatter's third quantity is conventionally keyed.
    for name, kw in (("colorbar_orientation", "orientation"), ("colorbar_location", "location")):
        if opts.get(name):
            colorbar_kwargs[kw] = str(opts[name])
    if opts.get("colorbar_aspect"):
        colorbar_kwargs["aspect"] = float(opts["colorbar_aspect"])
    # Default the axis labels to the first layer's x-dim / output; opts override them.
    axopts = _axopts(panel)
    # bsplot's format pass re-derives ticks and can re-normalise limits, so a DECLARED frame (the paper's own tick marks and ranges) is re-applied after it. Intent written in the spec must not be silently replaced by the tidy-up.
    post = {k: v for k, v in axopts.items() if k in _POST_FORMAT_OPTS}
    if kind in ("cartesian", "heatmap", "line3d") and layers:
        axopts.setdefault("xlabel", layers[0]["x"] or "")
        axopts.setdefault("ylabel", layers[0]["y"] or layers[0]["output"])
    if kind == "line3d" and layers:
        axopts.setdefault("zlabel", layers[0]["z"] or "")
    placeholder = getattr(panel, "placeholder", None)
    # A built-in kind names no `render:` and falls back to the callable core registers.
    render = getattr(panel, "render", None) or (kind if kind in _BUILTIN_PANELS else None)
    path = resolve_path(getattr(panel, "path", None), base_dir)
    source = getattr(panel, "source", None)
    if kind == "image" and source and path:
        # The panel draws a file; with a `source` it is a build product, re-rendered here so the figure cannot go out with a stale screenshot of a page that has since changed.
        from tvbo.plot.capture import capture

        capture(source, path, getattr(panel, "capture", None), base_dir=base_dir)
    return {
        "key": key,
        "kind": kind,
        "fill_cell": bool(opts.get("fill_cell")),
        "aspect": opts.get("aspect"),
        "drawer": kind in _DRAWER_KINDS,
        "title": getattr(panel, "label", None),
        "path": path,
        "render": render,
        "placeholder": placeholder,
        # Nothing to draw at all: the placeholder IS the panel, not a fallback. Without this the guarded draw succeeds silently (no layer raises) and the slot renders as an empty 0-1 axes instead of the honest label.
        "placeholder_only": bool(placeholder) and not layers and not render and not path,
        "layers": layers,
        "colorbar": colorbar,
        "colorbar_kwargs": colorbar_kwargs,
        "colorbar_label": opts.get("colorbar_label"),
        # Ticks written out to this many decimals instead of a shared multiplier. A slim bar has nowhere to put an exponent, so a range like 1e-4 otherwise reads as "3, 0, -1" — the axis is then silently wrong by four orders of magnitude.
        "colorbar_decimals": opts.get("colorbar_decimals"),
        # A paper's colourbar tick set is declared intent exactly as `xticks` is, and often carries meaning of its own — the two end values of a deliberately narrow range.
        "colorbar_ticks": opts.get("colorbar_ticks"),
        # Cells of blank band between two triangle layers, so the halves read as two quantities rather than one field (the paper's own `extra_diagonal`).
        "triangle_gap": int(opts.get("triangle_gap", 0) or 0),
        "axopts": axopts,
        "post_axopts": {} if (placeholder and not layers and not render and not path) else post,
        "ctx": ctx,
        "annotations": _annotations(panel, base_dir) + cell_labels,
        "groups": [g for g in (_group_axis(opts, "x"), _group_axis(opts, "y")) if g],
        "number_loc": getattr(panel, "number_loc", None),
        "number": getattr(panel, "number", None),
        "insets": cells
        + [  # a grid cell IS an inset; only who computes the bounds differs
            dict(_resolve_drawable(inset, f"{key}_inset{i}", base_dir), bounds=_inset_bounds(inset, key, i))
            for i, inset in enumerate(getattr(panel, "insets", None) or [])
        ],
    }


def output_format(figure) -> str:
    """The file extension a figure's render writes, without the dot.

    An animated figure's output is its movie, so the animation's container replaces the still's format rather than sitting beside it — a figure written as both a png and a gif is two artefacts claiming to be one figure. Asked by ``figure_outputs`` and by the renderer, so the name they agree on is derived once.
    """
    animation = _animation(figure)
    if animation:
        return animation["format"]
    return str(getattr(figure, "format", None) or "png").lstrip(".")


def _animated_sources(drawables) -> list:
    """``(container, output)`` of every layer whose data the frames advance through.

    Collected across panels, their insets and their grid cells, because the frame count is a property of the FIGURE: two panels reading different runs must not end up at different instants. A `static` layer is left out — it is the backdrop, and its length says nothing about how long the movie is.
    """
    sources: list = []
    for drawable in drawables:
        for layer in drawable.get("layers") or []:
            if layer.get("frame") in ("static", "cursor"):
                continue
            pair = [layer["container"], layer["output"]]
            if pair not in sources:
                sources.append(pair)
        sources.extend(pair for pair in _animated_sources(drawable.get("insets") or []) if pair not in sources)
    return sources


def _animation(figure) -> dict | None:
    """The figure's animation as the flat dict the template consumes, or ``None`` for a still."""
    spec = getattr(figure, "animation", None)
    if spec is None:
        return None
    fmt = str(getattr(spec, "format", None) or "gif").lstrip(".").lower()
    if fmt not in ("gif", "mp4"):
        raise ValueError(f"figure animation: format {fmt!r} is not one of gif, mp4.")
    frames = getattr(spec, "frames", None)
    still = getattr(spec, "still", None)
    return {
        "over": str(spec.over),
        "frames": int(frames) if frames is not None else None,
        "fps": int(getattr(spec, "fps", None) or 20),
        "format": fmt,
        "still": int(still) if still is not None else None,
    }


def build_context(figure, base_dir, outfile: str) -> dict:
    """Resolve a ``Figure`` into the template context (all IO paths + names resolved)."""
    from tvbo.plot import palette as _palette_mod

    base_dir = Path(base_dir)
    theme = theme_spec(figure, base_dir)
    _palette_mod.use(
        {k: v for k, v in theme.items() if k in _palette_mod.FIELDS}
    )  # in force while the panels resolve, so a role a layer names becomes a colour here
    animation = _animation(figure)
    panels = [_resolve_drawable(panel, key, base_dir, animation) for key, panel in _items(figure.panels)]
    if figure.layout:
        layout = _mosaic(str(figure.layout).replace("/", "\n"))  # bsplot mosaics split rows on newline
    else:
        # One row of the declared keys. A multi-character key has to go out as a token row: concatenated it would read as one cell per character.
        keys = [str(p["key"]) for p in panels] or ["a"]
        layout = [keys] if any(len(k) > 1 for k in keys) else "".join(keys)
    fmt = getattr(figure, "panel_number_format", None) or "{}"
    fig_loc = _enum_value(getattr(figure, "panel_number_loc", None))  # unset -> keep bsplot's own default placement
    font_size = getattr(figure, "font_size", None)
    number_size = getattr(figure, "panel_number_size", None) or (font_size * _PANEL_NUMBER_SCALE if font_size else None)
    offset = [float(v) for v in (getattr(figure, "panel_number_offset", None) or [])]
    reading = {key: n for n, key in enumerate(_panel_layout_order(figure))}
    seen: set[str] = set()
    for p in sorted(panels, key=lambda p: reading.get(p["key"], len(reading))):
        override = p.pop(
            "number", None
        )  # overrides the mosaic key; "false" suppresses the letter (many cells = one paper panel)
        ident = _letter_identity(override, p["key"], seen)
        if ident is None:
            p["letter"] = None
            p["number_kwargs"] = {}
            continue
        p["letter"] = fmt.format(ident)
        place = {"option": "numbers"}  # label is given verbatim, no int->letter conversion
        loc = p["number_loc"] or fig_loc
        if loc:  # only override placement when a corner was asked for
            # loc is a Corner enum whose str() is the corner text in both datamodel flavors.
            place.update(_PANEL_NUM_LOC.get(str(loc), _PANEL_NUM_LOC["upper left"]))
        if number_size:
            place["fontsize"] = number_size
        if offset:
            place["x_shift"] = place.get("x_shift", 0.0) + offset[0]
            place["y_shift"] = place.get("y_shift", 0.0) + offset[1]
        p["number_kwargs"] = place  # resolved here; the template just splats it

    # bsplot.figure.subplots kwargs, resolved here so the template just splats them.
    style_entries = _style_entries(figure, base_dir)
    dpi = getattr(figure, "dpi", None) or _style_dpi(style_entries)
    subplots_kwargs = {"layout": layout, "dpi": dpi}
    width, height = getattr(figure, "width", None), getattr(figure, "height", None)
    if width and height:  # physical size in mm -> inches
        subplots_kwargs["figsize"] = (width / 25.4, height / 25.4)
    for key in ("height_ratios", "width_ratios"):
        ratios = list(getattr(figure, key, None) or [])
        if ratios:
            subplots_kwargs[key] = ratios

    # fig.savefig kwargs, resolved here so the template just splats them. Trimming re-crops to content, so it is what makes a saved figure's aspect drift from the declared w×h.
    savefig_kwargs = {"dpi": dpi}
    if getattr(figure, "trim_margins", None) is not False:
        savefig_kwargs["bbox_inches"] = "tight"
        pad = getattr(figure, "pad", None)
        if pad is not None:
            savefig_kwargs["pad_inches"] = float(pad)

    spines = getattr(figure, "spines", None)
    spine_rcparams = {}
    if spines == "box":
        spine_rcparams = {f"axes.spines.{s}": True for s in ("top", "right", "left", "bottom")}
    elif spines == "open":
        spine_rcparams = {"axes.spines.top": False, "axes.spines.right": False}
    if getattr(figure, "trim_margins", None) is False:
        # matplotlib reads `bbox_inches=None` as "use rcParams['savefig.bbox']", which the stylesheet sets to "tight" — so the declared `trim_margins: false` only takes effect if the rcParam itself is cleared.
        spine_rcparams = {**spine_rcparams, "savefig.bbox": None}
    spine_rcparams = {
        **theme_rcparams(theme),
        **spine_rcparams,
    }  # the theme is declared too, and a spine the figure states beats the theme's own

    offset = getattr(figure, "spine_offset", None)  # undeclared defers to bsplot's own offset, as the slot documents
    format_kwargs = {} if offset is None else {"shift_left_spine": -float(offset), "shift_bottom_spine": -float(offset)}

    shared = {
        axis: [[k.strip() for k in str(g).split(",") if k.strip()] for g in (getattr(figure, f"share_{axis}", None) or [])]
        for axis in ("x", "y")
    }

    return {
        "name": figure.name or "figure",
        "animation": animation,
        "animated_sources": _animated_sources(panels) if animation else [],
        "still_outfile": str(Path(outfile).with_suffix(".png")) if animation and animation["still"] is not None else None,
        # A movie rebuilds the mosaic per frame, so the emitted script keeps the arguments that made it.
        "mosaic_kwargs": {"mosaic": layout, **{k: v for k, v in subplots_kwargs.items() if k.endswith("_ratios")}},
        "shared_scales": shared,
        "style": style_entries,
        "palette": {k: v for k, v in theme.items() if k in _palette_mod.FIELDS},
        "outfile": outfile,
        "panels": panels,
        "subplots_kwargs": subplots_kwargs,
        "layout_engine": _enum_value(getattr(figure, "layout_engine", None)),
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


# --------------------------------------------------------------------------- captions


def _group_letter(key):
    """The paper letter a mosaic key belongs to: its leading alphabetic run, so ``f1``, ``f2`` and ``f`` are all panel (f)."""
    text = str(key)
    head = "".join(itertools.takewhile(str.isalpha, text))
    return head or text


def _letter_identity(number, key, seen=None):
    """The letter a panel carries — its ``number`` override, else the paper letter of its mosaic ``key`` — or None when the override suppresses it (``false``/``none``/``""``: many cells under one paper letter).

    Pass *seen* to letter a group once: keys sharing a leading letter (``f1``, ``f2``, ...) are one paper panel, so only the first cell of the group draws it. Without *seen* every key answers for itself.

    Shared by :func:`build_context` (which formats it onto the figure) and the caption composer, so a caption's ``(a)`` can never disagree with the letter drawn on the panel.
    """
    if number is not None and str(number).lower() in ("false", "none", ""):
        return None
    if number is not None:
        if seen is not None:
            seen.add(_group_letter(key))  # an explicit letter still spends the group, or a sibling cell draws a second one
        return number
    ident = _group_letter(key)
    if seen is None:
        return ident
    if ident in seen:
        return None
    seen.add(ident)
    return ident


def _mosaic(layout: str):
    """The mosaic bsplot is handed: the layout string as-is, or a nested list when its rows are whitespace-separated.

    A row of single characters is the compact form and stays a string. Writing a row as tokens (``f1 f1 f2 f2``) is what lets a mosaic name its cells beyond one character, which is how several cells declare themselves parts of one paper panel. The two are told apart by a row holding more than one token, so surrounding whitespace of either kind leaves a compact layout compact.
    """
    rows = [row for row in layout.split("\n") if row.strip()]
    if not any(len(row.split()) > 1 for row in rows):
        return layout
    grid = [row.split() for row in rows]
    if len({len(row) for row in grid}) != 1:
        raise ValueError("every row of a token mosaic must hold the same number of cells")
    return grid


def _panel_layout_order(figure) -> list[str]:
    """Panel keys in reading order — first appearance in the ``layout`` mosaic, then any declared panel the mosaic omits, so a caption walks panels the way a reader meets them."""
    declared = [k for k, _ in _items(figure.panels)]
    layout = getattr(figure, "layout", None)
    if not layout:
        return declared
    grid = _mosaic(str(layout).replace("/", "\n"))
    cells = itertools.chain.from_iterable(grid) if isinstance(grid, list) else str(grid)
    order: list[str] = []
    for cell in cells:
        if cell in declared and cell not in order:
            order.append(cell)
    for k in declared:
        if k not in order:
            order.append(k)
    return order


def _used_source(used) -> tuple:
    """``(source, output)`` for a layer's ``used:`` DataRef — ``("analysis y", "fc")`` — or ``(None, None)`` for a local/unbound layer.

    Split rather than pre-joined so a panel drawing several outputs of ONE analysis (a density with its mean and a reference line on it) names that analysis once. An output that merely repeats its analysis's own name is dropped: it says one thing twice.
    """
    if used is None:
        return None, None
    out = getattr(used, "output", None)
    for attr, word in (("experiment", "experiment"), ("analysis", "analysis")):
        val = getattr(used, attr, None)
        if val:
            return f"{word} {val}", (str(out) if out and str(out) != str(val) else None)
    iri = getattr(used, "iri", None)
    return (Path(str(iri)).name, str(out) if out else None) if iri else (None, None)


def _panel_descriptor(panel) -> str:
    """The auto-derived structural half of a panel's caption clause, read from its spec.

    From each layer's ``mark`` + ``encoding`` (which quantity is on which axis) and its ``used:`` DataRef (which run/analysis it came from), so the description follows the figure: move a panel or rebind a layer and the sentence changes with it. Units live in the runtime container and are folded in by the renderer, not typed here.

    Clauses are deduplicated. A grid draws one binding per CELL — the same frames laterally and then medially, one analysis per row — so listing them undeduplicated repeated a single source once per cell and buried the authored caption behind eight identical phrases.
    """
    kind = str(getattr(panel, "kind", "") or "")
    if kind == "image":
        # Provenance, and only where nothing better was written: a filename in a journal caption is noise beside an authored sentence.
        src = getattr(panel, "source", None)
        return f"rendered from {Path(str(src)).name}" if src and not getattr(panel, "description", None) else ""
    # A grid's cells all draw the same kind, which names them better than "grid" does.
    cell_kind = str(getattr(getattr(panel, "cell", None), "kind", "") or "") if kind == "grid" else ""

    by_source: dict = {}
    for layer in as_list(getattr(panel, "layers", None)):
        enc = getattr(layer, "encoding", None)
        x = getattr(enc, "x", None) if enc else None
        y = getattr(enc, "y", None) if enc else None
        z = getattr(enc, "z", None) if enc else None
        mark = str(getattr(layer, "mark", None) or "")
        src, out = _used_source(getattr(layer, "used", None))
        if kind == "heatmap":
            body = f"{y or x or 'field'} as a matrix"
        elif mark == "rule":
            # A rule's subject is the VALUE it stands at, not the axis it crosses.
            body = f"rule at {out or y or x or 'a value'}"
        elif z:
            body = f"{mark or 'trajectory'} of {y} vs {x} vs {z}"
        elif x and y:
            body = f"{mark or 'line'} of {y} vs {x}"
        elif y or x:
            body = f"{mark or 'line'} of {y or x}"
        else:
            body = cell_kind or mark or kind
        if out and mark != "rule" and out not in body:
            body += f" ({out})"
        clauses = by_source.setdefault(src or "", [])
        if body and body not in clauses:
            clauses.append(body)
    # A custom panel draws through registered code, so the spec holds nothing structural to say about it: its authored description is the whole clause.
    return "; ".join(", ".join(c) + (f" from {s}" if s else "") for s, c in by_source.items()) or (
        "" if kind == "custom" else kind
    )


def _sentence(text: str) -> str:
    """Trim and terminate a clause with a single period, for concatenation into a caption."""
    text = (text or "").strip()
    if not text:
        return ""
    return text if text.endswith((".", "!", "?", ":")) else text + "."


def compose_caption(figure) -> str:
    """Compose a figure's caption from its spec — the authored lead plus one clause per panel.

    Each panel contributes ``(letter) label — <structural descriptor> <Panel.description>`` in layout order, the letter taken from the same identity the panel draws (:func:`_letter_identity`) so caption and figure cannot disagree. Cells sharing a paper letter share its clause, each adding only what the clause does not already say, so a grid does not repeat one descriptor per cell and a sibling's authored prose is not dropped with its letter. A sibling that authors no prose and derives no descriptor contributes nothing at all: its drawn title is a label on the figure, not a sentence in the caption. The structural descriptor is derived from the panel's layers (:func:`_panel_descriptor`); the authored ``Figure.description`` (lead) and ``Panel.description`` (per-panel interpretation) are the only parts a human writes.
    """
    spec_by_key = {k: p for k, p in _items(figure.panels)}
    lead: list[str] = []
    label = getattr(figure, "label", None)
    if label:
        lead.append(f"**{str(label).strip()}.**")  # journal convention: a bold figure title
    lead.append(_sentence(getattr(figure, "description", None) or ""))
    clauses: list[str] = []
    seen: set[str] = set()
    group_clause: dict[str, int] = {}
    # What each group's clause has already said, held as whole parts. Testing a part for containment in the clause TEXT instead drops any label that happens to be a substring of a sibling's prose, so a cell called "network" loses its name to an earlier sentence that merely used the word.
    group_parts: dict[str, set[str]] = {}
    for key in _panel_layout_order(figure):
        panel = spec_by_key.get(key)
        if panel is None:
            continue
        ident = _letter_identity(getattr(panel, "number", None), key, seen)
        parts = [
            _sentence(getattr(panel, "label", None) or ""),
            _sentence(_panel_descriptor(panel)),
            _sentence(getattr(panel, "description", None) or ""),
        ]
        group = _group_letter(key)
        said = group_parts.setdefault(group, set())
        if ident is not None:
            group_clause[group] = len(clauses)
            said.update(s for s in parts if s)
            clauses.append(f"**({ident})** {' '.join(s for s in parts if s)}".strip())
            continue
        # A cell of a paper panel already lettered, or whose letter the author suppressed: it shares that panel's clause.
        index = group_clause.get(group)
        # A sibling joins the clause for what it SAYS, never for its own title: a grid whose cells are titled and whose prose is authored once on the lettered cell would otherwise append a run of bare labels ("... power grid. spiking neurons. mean field.") that reads as debris rather than as caption.
        carried = [s for s in parts[1:] if s and s not in said]
        fresh = " ".join([s for s in parts[:1] if s and s not in said] + carried) if carried else ""
        if not fresh:
            continue
        said.update(s for s in parts if s)
        if index is None:
            group_clause[group] = len(clauses)
            clauses.append(fresh)
        else:
            clauses[index] = f"{clauses[index]} {fresh}".strip()
    return " ".join(s for s in (lead + clauses) if s).strip()


def write_caption(figure, out_dir, *, name: str | None = None) -> Path:
    """Write a figure's composed caption to ``<out_dir>/<name>.caption.qmd`` and return the path.

    A Quarto partial the manuscript pulls in with ``{{< include <name>.caption.qmd >}}``, so the caption is generated from the figure spec and regenerates whenever a panel moves or a layer is rebound — never hand-maintained beside the figure it describes.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = name or getattr(figure, "name", None) or "figure"
    path = out_dir / f"{stem}.caption.qmd"
    path.write_text(compose_caption(figure) + "\n", encoding="utf-8")
    return path
