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

    opts:
        network / mesh: where the geometry comes from (see :func:`_surface_mesh`).
        view: bsplot's camera name (lateral, medial, dorsal, ventral, anterior, posterior).
        hemi: 'lh' | 'rh' (default 'lh').
        cmap: colormap name; symmetric/vmin/vmax/percentile set the colour limits.
        symmetric: centre the limits on zero so zero is the colormap's midpoint (default
            true, because a map of signed deviations read on an off-centre scale is
            misleading in a way no axis label catches).
        percentile: clip the symmetric limit to this percentile of |values| (default 100),
            so one outlier vertex cannot wash the map out.
        mask: per-vertex 0/1 file; vertices OUTSIDE it (a medial wall) are drawn grey and
            excluded from the colour range rather than being coloured as zeros.
        color: draw the mesh itself in one flat colour instead of painting a map on it.
            With no layer that is the bare geometry; with ``geometry: true`` the layer
            supplies (V, 3) vertex COORDINATES, so what the panel shows is the surface a
            reconstruction rebuilt rather than a field living on a fixed surface.
        edgecolor / edge_linewidth: draw the triangulation, for a panel whose subject is
            the mesh the model is solved on.
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

    vmin, vmax = _colour_limits(values, opts, True)

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
        cmap=_resolve_cmap(opts.get("cmap", "viridis")),
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
            f"{raster.name}). The labelled raster is a fetched asset rather than a tracked one."
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

    opts:
        - atlas / volume: where the segmentation comes from (a curated atlas IRI, or a dseg file).
        - view: bsplot's camera name (sagittal, coronal, horizontal).
        - intensity_projection: how voxels collapse along the view axis (default 'absmax').
        - cmap: colormap name; symmetric/percentile/vmin/vmax set the colour limits.
        - symmetric: centre the limits on zero (default false — unlike a surface map of signed deviations, a volume panel is as often a positive quantity such as a frequency).
        - percentile: clip the limit to this percentile of |values| (default 100).
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

    vmin, vmax = _colour_limits(values, opts, False)

    view = str(opts.get("view", "sagittal")).lower()
    if view not in ("sagittal", "coronal", "horizontal"):
        raise ValueError(f"volume panel: view {view!r} is not one of sagittal, coronal, horizontal.")
    bsplot.glass_brain(
        nib.Nifti1Image(painted, img.affine),
        ax=ax,
        view=view,
        cmap=_resolve_cmap(opts.get("cmap", "viridis")),
        intensity_projection=str(opts.get("intensity_projection", "absmax")),
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_aspect("equal")  # a projection's frame is anatomy, not a coordinate system
    ax.axis("off")
    if opts.get("title"):
        ax.set_title(str(opts["title"]))


def scale_colormap(name, lo=None, hi=None, center=None):
    """A declared colormap name as the map a field is drawn with, pinned to *center* where one is asked for.

    ``center`` fixes the map's neutral colour at a value — zero for a signed change, where the sign is the reading. The limits stay the data's own and the map is truncated to the half-range the data actually reaches, so a unit of change is the same colour distance either side of the centre and the scale carries no colour the field never takes.

    The mesh and the bar that keys it both resolve here. A bar built from the untruncated map would put the neutral colour at the middle of a scale whose field crosses the centre anywhere else, and every value the reader looks up would be wrong.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as _np

    from tvbo.adapters.colormaps import resolve as _resolve_cmap

    cmap = _resolve_cmap(name)
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

    opts:
        cmap / vmin / vmax: the scale. With a layer bound and no explicit limits, the
            limits are read from the data, so the bar cannot drift from what it describes.
        center: the value the map's neutral colour sits at, resolved exactly as the mesh
            resolves its own, so a bar keying centred heatmaps is the scale they were drawn on.
        orientation: 'vertical' (default) or 'horizontal'.
        width: fraction of the slot the bar itself occupies across its short axis.
        ticks / ticklabels: the marks. A quantity in arbitrary units is labelled at its
            ends (Minimum..Maximum) rather than with numbers that mean nothing.
        label: the axis label beside the bar.

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
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=scale_colormap(opts.get("cmap", "viridis"), lo, hi, opts.get("center")))
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

    opts: labels, colors, linestyles, markers, title, loc, handlelength.
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
    ax.legend(
        handles,
        labels,
        loc=str(opts.get("loc", "center")),
        frameon=False,
        title=opts.get("title"),
        handlelength=float(opts.get("handlelength", 2.2)),
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


def _style_entries(figure, base_dir) -> list:
    """Classify each figure style as a bsplot named style or an .mplstyle path.

    bsplot.style.use only knows its registered names; a study's own .mplstyle is a filesystem path, applied via matplotlib's plt.style.use instead. This lets a study carry its own design rules (Figure.style: ['<path>/study.mplstyle']). Only the path form is resolved against base_dir — a named style is not a filesystem reference.
    """
    styles = list(getattr(figure, "style", None) or []) or ["tvbo"]
    out = []
    for s in styles:
        s = str(s)
        is_path = s.endswith(".mplstyle") or "/" in s or "\\" in s
        out.append({"path": is_path, "value": resolve_path(s, base_dir) if is_path else s})
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
    return kw


# Grammar-panel axis directives on ``Panel.opts`` — a small backend-independent set the template applies uniformly (a ``custom`` panel routes ``opts`` to its callable; see ``build_context``).
_AXIS_OPTS = {
    "xlabel",
    "ylabel",
    "title",
    "xlim",
    "ylim",
    "xticks",
    "yticks",
    "hide_xticklabels",
    "hide_yticklabels",
    "axhline",
    "axvline",
    "axline",
    "axhline_color",
    "axvline_color",
    "axline_color",
    "xtick_side",
    "ytick_side",
    "xlabel_side",  # the axis label alone crosses to the far side, leaving the tick numbers where they are
    "ylabel_side",
    "region",  # [x0, x1, y0, y1] outline: the window a paper rings on a field
    "region_color",
    "legend",
    "xscale",
    "yscale",  # axis scale (log/symlog/linear): part of the claim, not cosmetic
    "nbins",  # tick budget: a small multi-panel slot cannot hold the automatic count
    "tick_size",  # tick-label type size, when the figure's own would crowd the cell out
    "tick_length",  # tick mark length, the same concern as tick_size on the other axis of the gap
    "xtick_rotation",  # slant the tick labels of a dense row of panels so neighbours stop running together
    "ytick_rotation",
    "xtick_format",  # 'sci' moves a shared exponent to the axis corner; 'plain' writes every tick in full
    "ytick_format",
    "tick_prune",  # drop the first/last tick ('lower'|'upper'|'both') where a neighbouring panel starts
    "xlabel_pad",  # gap in points between an axis label and its tick numbers, when the default lets them touch
    "ylabel_pad",
    "aspect",
    "box_aspect",
    "invert_x",
    "invert_y",
    "frame",  # frame geometry/direction/visibility
    "zlabel",
    "zlabel_pad",  # 3-D axis labels land where matplotlib puts them, which on a narrow slot is over the neighbour
    "zlim",
    "elev",
    "azim",
    "invert_z",
    "zoom",  # line3d only
}


# Axis directives the format pass can overwrite, so they are re-applied after it.
_POST_FORMAT_OPTS = {
    "xtick_side",
    "ytick_side",
    "xlabel_side",  # re-applied beside the tick side, which also moves the label and would win alone
    "ylabel_side",
    "xticks",
    "yticks",
    "xlim",
    "ylim",
    "xscale",
    "yscale",
    "hide_xticklabels",
    "hide_yticklabels",
    "aspect",
    "box_aspect",
    "frame",
    "nbins",
    "tick_size",
    "tick_length",
    "xtick_rotation",
    "ytick_rotation",
    "xtick_format",
    "ytick_format",
    "tick_prune",
}


def _panel_opts(panel) -> dict:
    """Resolve ``Panel.opts`` (Argument dict) into a plain ``{name: value}`` dict."""
    return _arg_dict(getattr(panel, "opts", None))


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

    Draws from ``Panel.opts`` (the recognised ``_AXIS_OPTS`` keys) plus the boolean ``Panel.legend`` slot. This is the minimal per-panel label/limit override: the paper's LaTeX axis labels and shared ranges live here rather than defaulting to the bare variable name.
    """
    o = {k: (_as_number(v) if k in _NUMERIC_AXIS_OPTS else v) for k, v in _panel_opts(panel).items() if k in _AXIS_OPTS}
    if getattr(panel, "legend", None):
        o.setdefault("legend", "best")
    return o


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
    *n* lies half a unit below it — the rules and the label centres carry that shift, and a
    bound is therefore declared as "how many", never as a plotted coordinate.
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

    Returns ``""`` when the container is not there, which a panel declaring a ``placeholder`` relies on: the generated script draws the honest label instead of a plot, so a partially-run study still renders. A panel without a placeholder gets a named error from ``_open`` at render time. What is gone is the guessing — four candidate layouts tried in turn, which is how a figure came to read one run's experiments against another run's analyses.
    """
    if not iri:
        return ""
    from tvbo.adapters.bids import entity_value
    from tvbo.data.dataref import experiment_id as _experiment_id
    from tvbo.utils.study_layout import study_path

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
        files = [f for f in sorted(results.glob(f"{stem}_*result.h5")) if "network" not in f.name]
        if files:
            return str(files[0].resolve())
    return ""


# --------------------------------------------------------------------------- custom panels The ``custom`` escape hatch: a registered ``fn(fig, ax, ctx)`` draws a bespoke sub-panel the grammar can't (yet) express. ``ctx`` carries the resolved layers (container paths, transforms, selectors already resolved by ``build_context``) plus the panel's ``opts``, so a callable opens the container(s) itself and draws exactly what the paper needs. A study registers its own the same way it registers a transform.


def load_layer(layer: dict):
    """Open a custom panel's resolved layer into a DataArray (public API).

    A registered ``custom`` panel receives ``ctx`` with a ``layers`` list of resolved-layer dicts (container path, output, transform, selector — all resolved by ``build_context``);
    it calls ``bsplot.load_layer(ctx["layers"][i])`` to open the i-th one as an xarray ``DataArray`` with the declared ``transform`` and ``.sel`` already applied. A ``Layer.transform`` runs before the selection and a ``DataRef.transform`` after it, which is the order the two slots are documented in and the only thing that distinguishes them. The shared container cache means opening the same file across panels is free.
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
        "cmap": getattr(style, "colormap", None),
        "transform": getattr(layer, "transform", None),
        "ref_transform": getattr(used, "transform", None),
        "sel": sel,
        "sel_method": method,
        "triangle": str(triangle) if triangle else None,
        "style": kwargs,
    }


# Interior drawn by a callable or sub-axes, so its ticks must survive the format pass.
_DRAWER_KINDS = {"custom", "surface", "volume", "grid", "line3d"}

# Kinds whose interior is a built-in callable, needing no `render:` and no code_modules.
_BUILTIN_PANELS = {"surface", "volume", "colorbar", "legend"}


class _GridCell:
    """One cell of a ``grid`` panel: the shared ``cell:`` template with this cell's overrides.

    Shaped like a Panel so it resolves through :func:`_resolve_drawable` — a grid cell must draw exactly as the same kind draws in a mosaic slot, and a second resolution path is how the two would drift apart.
    """

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


def _resolve_drawable(panel, key, base_dir) -> dict:
    """Resolve one drawable — a mosaic Panel or an Inset — into its template entry.

    An inset is a panel in everything that draws, so both go through this one function and the template emits both from one partial. Splitting them would let an inset's heatmap, triangle gap or colourbar quietly diverge from the identical panel beside it.
    """
    kind = str(panel.kind)  # datamodel enum -> plain string (flavor-agnostic)
    opts = _panel_opts(panel)
    # A grid's `layers:` belong to its cells, so they are not also drawn on the host axes.
    cells, cell_labels = _grid_cells(panel, key, base_dir, opts) if kind == "grid" else ([], [])
    layers = (
        [] if kind == "grid" else [_resolve_layer(layer, kind, base_dir) for layer in (getattr(panel, "layers", None) or [])]
    )
    # A callable-drawn kind gets the whole opts dict; grammar panels read the axis subset.
    ctx = (
        {"layers": layers, "opts": opts, "key": key, "base_dir": str(base_dir)}
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
    return {
        "key": key,
        "kind": kind,
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


def build_context(figure, base_dir, outfile: str) -> dict:
    """Resolve a ``Figure`` into the template context (all IO paths + names resolved)."""
    base_dir = Path(base_dir)
    panels = [_resolve_drawable(panel, key, base_dir) for key, panel in _items(figure.panels)]
    if figure.layout:
        layout = _mosaic(str(figure.layout).replace("/", "\n"))  # bsplot mosaics split rows on newline
    else:
        # One row of the declared keys. A multi-character key has to go out as a token row: concatenated it would read as one cell per character.
        keys = [str(p["key"]) for p in panels] or ["a"]
        layout = [keys] if any(len(k) > 1 for k in keys) else "".join(keys)
    fmt = getattr(figure, "panel_number_format", None) or "{}"
    fig_loc = getattr(figure, "panel_number_loc", None)  # unset -> keep bsplot's own default placement
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
    dpi = getattr(figure, "dpi", None) or 200
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

    offset = getattr(figure, "spine_offset", None)  # undeclared defers to bsplot's own offset, as the slot documents
    format_kwargs = {} if offset is None else {"shift_left_spine": -float(offset), "shift_bottom_spine": -float(offset)}

    shared = {
        axis: [[k.strip() for k in str(g).split(",") if k.strip()] for g in (getattr(figure, f"share_{axis}", None) or [])]
        for axis in ("x", "y")
    }

    return {
        "name": figure.name or "figure",
        "shared_scales": shared,
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
        src = getattr(panel, "source", None)
        return f"rendered from {Path(str(src)).name}" if src else ""
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

    Each panel contributes ``(letter) label — <structural descriptor> <Panel.description>`` in layout order, the letter taken from the same identity the panel draws (:func:`_letter_identity`) so caption and figure cannot disagree. Cells sharing a paper letter share its clause, each adding only what the clause does not already say, so a grid does not repeat one descriptor per cell and a sibling's authored prose is not dropped with its letter. The structural descriptor is derived from the panel's layers (:func:`_panel_descriptor`); the authored ``Figure.description`` (lead) and ``Panel.description`` (per-panel interpretation) are the only parts a human writes.
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
        if ident is not None:
            group_clause[group] = len(clauses)
            clauses.append(f"**({ident})** {' '.join(s for s in parts if s)}".strip())
            continue
        # A cell of a paper panel already lettered, or whose letter the author suppressed: it shares that panel's clause.
        index = group_clause.get(group)
        said = clauses[index] if index is not None else ""
        fresh = " ".join(s for s in parts if s and s not in said)
        if not fresh:
            continue
        if index is None:
            group_clause[group] = len(clauses)
            clauses.append(fresh)
        else:
            clauses[index] = f"{said} {fresh}".strip()
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
