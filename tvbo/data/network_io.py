"""High-level Network I/O. Dispatches by companion file extension.

Supported companion formats:
  .h5 / .hdf5  — HDF5 (default, best compression)
  .zarr/        — Zarr (cloud-native, S3-compatible)
  .csv          — CSV legacy (one file = one matrix = first template edge)

YAML sidecars are loaded via linkml_runtime.loaders.yaml_loader — the same loader used by Dynamics, Coupling, and SimulationExperiment. This ensures schema validation and proper nested object construction. Never use raw yaml.safe_load → cls(**dict) for LinkML classes.

See §12.2 of the tvbo HDF5 format proposal v0.7.
"""

import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import yaml as _yaml
from linkml_runtime.dumpers import yaml_dumper
from linkml_runtime.loaders import yaml_loader

from tvbo.utils import yaml_loader as tvbo_yaml_loader

SCHEMA_VERSION = "tvb-datamodel/0.7.0"

from tvbo.data.matrix_io import (
    LazyArrayStore,
    auto_format,
    edge_name,
    read_edge,
    template_edges,
    write_matrix,
)

# ── BIDS filename patterns (canonical definition, used by converters) ─

RELMAT_PATTERNS = [
    # Normative (template-level) connectome — tpl- replaces sub-
    "tpl-{template}[_cohort-{cohort}][_rec-{reconstruction}]"
    "_atlas-{atlas}[_seg-{segmentation}][_scale-{scale}]"
    "[_desc-{description}]_relmat{extension|.h5}",
    # Subject-specific connectome — sub- with space-
    "sub-{subject}[_ses-{session}]_space-{space}"
    "[_rec-{reconstruction}]_atlas-{atlas}"
    "[_seg-{segmentation}][_scale-{scale}]"
    "[_desc-{description}]_relmat{extension|.h5}",
]

SENSOR_PATTERNS = [
    # Template-level sensor network with atlas (forward-model projection)
    "tpl-{template}_acq-{acquisition}[_atlas-{atlas}][_desc-{description}]_sensors{extension|.h5}",
    # Template-level sensor network without atlas
    "tpl-{template}_acq-{acquisition}[_desc-{description}]_sensors{extension|.h5}",
    # Sensor network without standard template
    "acq-{acquisition}[_desc-{description}]_sensors{extension|.h5}",
]

# ── Helpers ───────────────────────────────────────────────────────────


_template_edges = template_edges


def _read_edges(store, meta: dict) -> tuple[dict, dict]:
    """Every template-edge matrix and its edge parameters, from an open store.

    Names come from the sidecar when it declares any, else from the store's own ``edges/`` listing. Each edge is read through :func:`tvbo.data.matrix_io.read_edge`, in its stored format.
    """
    edges = template_edges(meta.get("edges", []))
    names = [edge_name(e) for e in edges] if edges else (list(store["edges"]) if "edges" in store else [])
    arrays, params = {}, {}
    for name in names:
        try:
            arrays[name], params[name] = read_edge(store, name)
        except KeyError:
            continue
    return arrays, params


def _write_dimension_labels(dataset, meta: dict, column_labels: list[str]):
    """Attach HDF5 dimension scales to a matrix dataset.

    dim-0 (rows) is labelled from the parent Network's node labels.
    dim-1 (columns) is labelled from the edge's ``dimension_labels``.
    Scale datasets are created inside the same group as *dataset*.
    """
    import h5py

    if not isinstance(dataset.id.id, int):
        return  # skip for zarr
    grp = dataset.parent

    # Row labels from network nodes
    nodes = meta.get("nodes", [])
    if nodes:
        row_labels = [str(n.get("label", f"node_{n.get('id', i)}")) for i, n in enumerate(nodes)]
        row_ds = grp.create_dataset(
            "_dim_row_labels",
            data=np.array(row_labels, dtype=h5py.string_dtype()),
        )
        row_ds.make_scale("row_labels")
        dataset.dims[0].attach_scale(row_ds)
        dataset.dims[0].label = "rows"

    # Column labels from dimension_labels
    if column_labels:
        col_ds = grp.create_dataset(
            "_dim_col_labels",
            data=np.array(column_labels, dtype=h5py.string_dtype()),
        )
        col_ds.make_scale("column_labels")
        dataset.dims[1].attach_scale(col_ds)
        dataset.dims[1].label = "columns"


def _write_edges(store, meta: dict, arrays: dict, edge_params: dict):
    """Write all template-edge matrices + edge parameters to a store."""
    edge_meta = {}
    for e in _template_edges(meta.get("edges", [])):
        edge_meta[e.get("name") or e.get("label")] = e

    store.attrs["tvbo_class"] = "tvbo:Network"
    store.attrs["sidecar_file"] = str(meta.get("_sidecar_name", ""))
    store.attrs["schema_version"] = str(meta.get("schema_version", "tvb-datamodel/0.7.0"))

    for name, matrix in arrays.items():
        m = edge_meta.get(name, {})
        fmt = m.get("format", auto_format(matrix))
        dtype = m.get("dtype")
        grp = store.create_group(f"edges/{name}")
        grp.attrs["tvbo_class"] = "tvbo:Matrix"
        for attr in ("directed", "unit"):
            if attr in m:
                # Cast to native Python types — LinkML extended_str and extended_bool are not directly serializable by h5py.
                val = m[attr]
                if isinstance(val, bool):
                    grp.attrs[attr] = val
                else:
                    grp.attrs[attr] = str(val)
        write_matrix(grp, matrix, fmt=str(fmt), dtype=dtype)

        # HDF5 dimension scales for labelled axes (§12.2)
        dim_labels = m.get("dimension_labels")
        if dim_labels and "data" in grp:
            _write_dimension_labels(grp["data"], meta, dim_labels)

        for pname, pmatrix in edge_params.get(name, {}).items():
            pg = grp.require_group("edge_parameters").create_group(pname)
            pg.attrs["tvbo_class"] = "tvbo:Parameter"
            p_meta = (m.get("parameters") or {}).get(pname) if isinstance(m.get("parameters"), dict) else None
            p_meta = p_meta if isinstance(p_meta, dict) else {}
            write_matrix(pg, pmatrix, fmt=p_meta.get("format", fmt), dtype=p_meta.get("dtype", dtype))


def _nodes_are_placeholders(nodes, number_of_nodes) -> bool:
    """True when ``nodes`` is exactly what ``Network(number_of_nodes=N)`` would synthesise.

    A Network materialises `node_0 … node_{N-1}` whenever nodes are not authored, so such a list carries no information the node count does not already hold — and at mesh scale (32,492 vertices) writing it out makes the sidecar larger than the matrices it describes.
    """
    if not nodes or len(nodes) != (number_of_nodes or 0):
        return False
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            return False
        extra = set(node) - {"id", "label", "record", "size"}
        if extra or node.get("id") != i or node.get("label") != f"node_{i}":
            return False
        if node.get("record", True) is not True or node.get("size", 1) != 1:
            return False
    return True


def _v07_postprocess(meta: dict) -> dict:
    """Transform a LinkML-dumped dict into strict v0.7 sidecar format.

    Applies the following transforms:
    1. Injects tvbo_class and schema_version at the top.
    2. Removes redundant inner name: from dict-keyed parameters.
    3. Drops a node list that is purely the default placeholders.
    4. Reorders top-level keys to match §4 of the v0.7 spec.
    """
    # 1. Inject header fields
    meta["tvbo_class"] = "tvbo:Network"
    meta["schema_version"] = SCHEMA_VERSION

    # 2. Parameters: remove redundant inner name (dict key IS the name)
    def _strip_param_name(params):
        if not isinstance(params, dict):
            return
        for body in params.values():
            if isinstance(body, dict):
                body.pop("name", None)

    _strip_param_name(meta.get("parameters"))
    for node in meta.get("nodes", []):
        if isinstance(node, dict):
            _strip_param_name(node.get("parameters"))
    for edge in meta.get("edges", []):
        if isinstance(edge, dict):
            _strip_param_name(edge.get("parameters"))

    # 3. Placeholder nodes: `number_of_nodes` alone reconstitutes them on load
    if _nodes_are_placeholders(meta.get("nodes"), meta.get("number_of_nodes")):
        meta.pop("nodes")

    # 4. Reorder keys: v0.7 §4 ordering
    key_order = [
        "tvbo_class",
        "schema_version",
        "label",
        "description",
        "number_of_nodes",
        "descriptor",
        "distance_unit",
        "time_unit",
        "data_file",
        "parent_network",
        "node_mapping",
        "provenance",
        "bids",
        "dynamics",
        "coupling",
        "parameters",
        "parcellation",
        "tractogram",
        "nodes",
        "edges",
    ]
    ordered = {}
    for k in key_order:
        if k in meta:
            ordered[k] = meta[k]
    # Append any remaining keys not in the order list
    for k, v in meta.items():
        if k not in ordered:
            ordered[k] = v
    return ordered


def _purify(obj):
    """Recursively convert LinkML extended types to plain Python types.

    yaml_loader.load_as_dict returns extended_str, extended_int, extended_bool etc. which yaml.dump serializes with Python tags.
    """
    if isinstance(obj, dict):
        return {_purify(k): _purify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_purify(v) for v in obj]
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, str):
        return str(obj)
    return obj


def _write_v07_sidecar(network, sidecar_path: Path, sidecar_format: str, data_file: str = None):
    """Write a strict v0.7-compliant YAML or JSON sidecar.

    Dumps the network via LinkML, post-processes to v0.7 form, and writes with controlled key ordering.

    A ``data_file``-backed connectome is materialised — its arrays live in the companion file — so the directives that would rebuild them from a source are dropped and a reload uses the companion. ``parcellation`` is deliberately not one of them: it also states which atlas this network's node labels belong to, which is what a ``by_label`` crosswalk resolves against, so dropping it would leave a saved network unable to reconcile any label the atlas spells differently. Keeping it is safe because ``load_network`` defers connectivity until the companion store is attached, so it no longer re-expands to the atlas node set.
    """
    if data_file:
        network.data_file = data_file

    meta = yaml_loader.load_as_dict(yaml_dumper.dumps(network))
    meta = _purify(meta)
    if data_file:
        for _directive in ("bids_dir", "graph_generator", "tractogram"):
            meta.pop(_directive, None)
    meta = _v07_postprocess(meta)

    if sidecar_format == "json":
        import json

        with open(sidecar_path, "w") as f:
            json.dump(meta, f, indent=2)
    else:
        with open(sidecar_path, "w") as f:
            _yaml.dump(meta, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _create_ds(grp, name, *, data, **kwargs):
    """Create a dataset compatible with both h5py and zarr v3."""
    try:
        import zarr

        if isinstance(grp, zarr.Group):
            return grp.create_array(name, data=data, **kwargs)
    except ImportError:
        pass
    return grp.create_dataset(name, data=data, **kwargs)


def _carry_through(network) -> dict:
    """The mesh and per-node datasets a re-save copies across, read while the source companion is still readable.

    ``save_network`` opens the destination with truncation, and a network saved back onto its own companion would then find every lazy dataset gone — silently, because a missing mesh is indistinguishable from a network that never had one.
    """
    if not hasattr(network, "array"):
        return {}
    nm_path = getattr(network, "node_mapping", None) or "/nodes/parent_index"
    paths = ["mesh/vertices", "mesh/elements", "mesh/normals", nm_path.lstrip("/")]
    src = getattr(network, "_store", None)
    if src is not None and hasattr(src, "dataset_keys"):
        try:
            paths.extend(src.dataset_keys("nodes"))
        except (AttributeError, OSError, KeyError):
            pass
    out = {}
    for path in paths:
        value = network.array(path)
        if value is not None:
            out[path] = value
    return out


def _write_nodes(store, network, carried: dict):
    """Write node-level data to a store.

    Persists:
    - ``nodes/parent_index``: hierarchical node mapping (if present)
    - ``nodes/coordinates``: (N,3) float32 array from Node.position (if present)
    - every other ``nodes/<name>`` dataset the network was loaded from, copied through

    The copy-through is what makes a re-save lossless. A companion may carry per-node arrays this writer knows nothing about — a study's own fitted intervals, a per-region scaling — and an observation may reference one as ``network.nodes.<name>``. Dropping them here turns a packed kit into a spec whose references cannot resolve on the target host.

    Called after ``_write_edges`` during ``save_network``.
    """
    written: set[str] = set()

    nm_path = getattr(network, "node_mapping", None) or "/nodes/parent_index"
    data = carried.get(nm_path.lstrip("/"))
    if data is not None:
        key = nm_path.lstrip("/")
        written.add(key)
        parts = key.rsplit("/", 1)
        if len(parts) == 2:
            grp_path, ds_name = parts
            grp = store.require_group(grp_path)
            _create_ds(grp, ds_name, data=data, dtype="int32")
        else:
            _create_ds(store, key, data=data, dtype="int32")

    # Node coordinates: prefer explicit Node.position (only when EVERY node has one — a partial list would misalign with node order). Otherwise fall back to network.get_centers(), which resolves centres from the atlas or by region-label mapping (e.g. DesikanKilliany), so BIDS connectomes without inline positions still ship coordinates in the companion file.
    nodes = getattr(network, "nodes", None) or []
    coords = []
    if nodes and all(getattr(node, "position", None) is not None for node in nodes):
        coords = [[float(n.position.x), float(n.position.y), float(n.position.z)] for n in nodes]
    elif nodes:
        try:
            centres = network.get_centers()
        except Exception:  # noqa: BLE001
            centres = None
        if centres and len(centres) == len(nodes) and all(i in centres for i in range(len(nodes))):
            coords = [list(centres[i]) for i in range(len(nodes))]
    if coords:
        import numpy as _np

        grp = store.require_group("nodes")
        _create_ds(
            grp,
            "coordinates",
            data=_np.array(coords, dtype="float32"),
        )
        written.add("nodes/coordinates")

    # `written` holds what was ACTUALLY emitted, so a dataset with no in-memory value is carried across from the source rather than silently dropped.
    keys = [k for k in carried if k.startswith("nodes/") and k not in written]
    if not keys:
        return
    grp = store.require_group("nodes")
    for key in keys:
        _create_ds(grp, key.split("/", 1)[1], data=carried[key])


def _write_mesh(store, network, carried: dict):
    """Write mesh data (vertices, elements, normals) to ``/mesh/`` group.

    Each array comes from :func:`_carry_through` — resident if an adapter such as ``from_tvb_surface`` set it, read off the source companion before this store was opened otherwise — so a re-save of a surface network it never touched is lossless, including a re-save onto that same companion.

    Called after ``_write_nodes`` during ``save_network``.
    """
    import numpy as _np

    vertices = carried.get("mesh/vertices")
    if vertices is None:
        return

    mesh_grp = store.require_group("mesh")

    mesh_obj = getattr(network, "mesh", None)
    if mesh_obj:
        mesh_grp.attrs["tvbo_class"] = "tvbo:Mesh"
        et = getattr(mesh_obj, "element_type", None)
        if et:
            mesh_grp.attrs["element_type"] = str(getattr(et, "text", et)).encode("utf-8")
        nv = getattr(mesh_obj, "number_of_vertices", None)
        if nv is not None:
            mesh_grp.attrs["number_of_vertices"] = int(nv)
        ne = getattr(mesh_obj, "number_of_elements", None)
        if ne is not None:
            mesh_grp.attrs["number_of_elements"] = int(ne)

    v = _np.asarray(vertices, dtype="float32")
    _create_ds(mesh_grp, "vertices", data=v, chunks=(min(v.shape[0], 4096), 3), compression="gzip")

    elements = carried.get("mesh/elements")
    if elements is not None:
        e = _np.asarray(elements, dtype="int32")
        _create_ds(mesh_grp, "elements", data=e, chunks=(min(e.shape[0], 4096), e.shape[1]), compression="gzip")

    normals = carried.get("mesh/normals")
    if normals is not None:
        n = _np.asarray(normals, dtype="float32")
        _create_ds(mesh_grp, "normals", data=n, chunks=(min(n.shape[0], 4096), 3), compression="gzip")


# ── Load ──────────────────────────────────────────────────────────────


EMBEDDED_METADATA_ATTR = "tvbo_metadata"
"""Root attribute holding a self-describing companion's own sidecar, as JSON.

A companion carrying this attribute needs no sidecar file: ``load_network`` reads the
metadata out of the binary and uses the same file as its own array store. Written by
:func:`write_embedded_metadata` and read back by :func:`read_embedded_metadata`. Namespaced
like the module's other root attrs (``tvbo_class``, ``schema_version``) so a foreign tool's
generic ``metadata`` key cannot be mistaken for one of ours.
"""


def read_embedded_metadata(path) -> dict | None:
    """The sidecar dict embedded in a self-describing companion, or ``None``.

    ``None`` covers every "this is not a self-describing binary" case — a YAML/JSON sidecar path, a companion written without the attribute, an unreadable or non-JSON payload — so callers can treat it as a plain feature probe.
    """
    path = Path(path)
    ext = path.suffix.lower()
    raw = None
    try:
        if ext in (".h5", ".hdf5"):
            import h5py

            with h5py.File(path, "r") as f:
                raw = f.attrs.get(EMBEDDED_METADATA_ATTR)
        # Bare-suffix guard first: a sidecar path must not pay a stat() to learn it is not a zarr directory, and load_network probes every load.
        elif ext == "" and path.is_dir() and (path / ".zgroup").exists():
            import zarr

            raw = zarr.open(str(path), "r").attrs.get(EMBEDDED_METADATA_ATTR)
        elif ext == ".zarr":
            import zarr

            raw = zarr.open(str(path), "r").attrs.get(EMBEDDED_METADATA_ATTR)
    except (ImportError, OSError, KeyError):
        # Missing backend, or a corrupt/foreign file: "not self-describing", not an error.
        return None
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, dict):
        # zarr decodes JSON attrs natively, so an externally-authored store may hand back the object rather than the string this module writes.
        return raw
    try:
        meta = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def write_embedded_metadata(companion_path, meta: dict) -> None:
    """Embed a sidecar dict into a companion so the file needs no sidecar.

    The dict is stored as JSON under the ``metadata`` root attribute; ``data_file`` is dropped because a self-describing file *is* its own array store.
    :func:`load_network` reads it back, so one path is the whole Network.
    """
    companion_path = Path(companion_path)
    payload = {k: v for k, v in meta.items() if k != "data_file"}
    blob = json.dumps(payload, sort_keys=False)
    ext = companion_path.suffix.lower()
    if ext in (".h5", ".hdf5"):
        import h5py

        with h5py.File(companion_path, "a") as f:
            f.attrs[EMBEDDED_METADATA_ATTR] = blob
    elif ext == ".zarr" or companion_path.is_dir():
        import zarr

        zarr.open(str(companion_path), "a").attrs[EMBEDDED_METADATA_ATTR] = blob
    else:
        raise ValueError(f"cannot embed metadata in {companion_path.suffix!r} companion")


def load_network(path):
    """Load a tvbo Network from a sidecar, or from a self-describing companion.

    Uses linkml yaml_loader or json_loader to construct a schema-validated Network instance directly — same pattern as Dynamics.from_file().

    Two layouts are accepted:

    - **sidecar + companion** — a ``.yaml``/``.json`` metadata file whose ``data_file``
      names the binary beside it;
    - **single self-describing file** — a ``.h5``/``.zarr`` carrying its own sidecar in
      the ``metadata`` root attribute (see :data:`EMBEDDED_METADATA_ATTR`). The file is its own array store, so one path is the whole Network.

    Arrays are NOT loaded into memory. A LazyArrayStore is attached that loads arrays on first access (e.g., net.matrix("weight")).

    Parameters
    ----------
    path : str or Path
        Path to a YAML/JSON sidecar, or to a self-describing ``.h5``/``.zarr``.

    Returns:
    -------
    Network
        Fully constructed tvbo.Network with lazy array references.
    """
    from tvbo.classes.network import Network

    path = Path(path)

    embedded = read_embedded_metadata(path)
    if embedded is not None:
        meta_dict = embedded
        meta_dict.pop("data_file", None)
        data_file = path.name
    else:
        # Load as dict first so data_file can be handled here. It IS a schema slot, but `Network._resolve_from_data_file` treats it as an indirect reference to ANOTHER network's sidecar, so leaving it on the constructor kwargs would recurse.
        meta_dict = yaml_loader.load_as_dict(str(path))
        data_file = meta_dict.pop("data_file", None)
    # Extract non-schema fields that are YAML-only metadata
    bids_meta = meta_dict.pop("bids", None)
    descriptor = meta_dict.pop("descriptor", None)
    meta_dict = tvbo_yaml_loader.strip_envelope(meta_dict)

    # Reconstruct clean YAML without non-schema fields for LinkML loader
    import yaml as _yaml

    # This loader attaches connectivity itself, so the constructor must not resolve it: `data_file` is stripped above (it is an INDIRECT reference — `_resolve_from_data_file` reads the companion's own sidecar, which would recurse into this very load), and a sidecar that also declares `parcellation:` would otherwise fall through to the normative-database branch and cache an atlas connectome that shadows this file's real matrices. `_resolve` runs below, once `_store` is in place.
    clean_yaml = _yaml.dump({**meta_dict, "_defer_connectivity": True}, Dumper=_yaml.SafeDumper)
    net = yaml_loader.loads(clean_yaml, Network)

    # Attach non-schema metadata as attributes (used by bids_filename)
    if bids_meta:
        net.bids = bids_meta
    if descriptor:
        net.descriptor = descriptor

    # Attach lazy array store (no arrays loaded yet).
    if data_file:
        data_path = path.parent / data_file
        net._store = LazyArrayStore(data_path, meta_dict)
        net.data_file = data_file

        # Restore mesh data from companion if present
        _load_mesh(net, data_path)
    else:
        net._store = None

    # `_store` now makes the network materialised, so this expands node/edge templates and subnetworks without re-entering the connectivity branches; a sidecar carrying only a `parcellation:` still resolves its normative connectome here as before.
    net._resolve(source_dir=str(path.parent))

    return net


def _load_mesh(network, companion_path):
    """Restore the ``Mesh`` record from a companion's ``/mesh/`` group attributes; its arrays stay on the store until asked for."""
    ext = Path(companion_path).suffix.lower()
    if ext not in (".h5", ".hdf5"):
        return
    import h5py

    with h5py.File(companion_path, "r") as f:
        if "mesh" not in f:
            return
        mg = f["mesh"]
        from tvbo.datamodel import schema as tvbo_datamodel

        et = mg.attrs.get("element_type", None)
        if isinstance(et, bytes):
            et = et.decode("utf-8")
        nv = mg.attrs.get("number_of_vertices", None)
        ne = mg.attrs.get("number_of_elements", None)
        mesh_obj = tvbo_datamodel.Mesh(
            element_type=et,
            number_of_vertices=int(nv) if nv is not None else None,
            number_of_elements=int(ne) if ne is not None else None,
        )
        object.__setattr__(network, "_mesh", mesh_obj)


# ── Save ──────────────────────────────────────────────────────────────


def save_network(network, yaml_path, binary_format: str = "h5", sidecar_format: str = "yaml"):
    """Save a tvbo Network as sidecar + binary companion.

    Uses LinkML yaml_dumper or json_dumper for schema-valid sidecar output — no manual field unpacking or yaml.dump() calls.

    The generic canonical keys `weight` and `length` that `from_matrix` produces are remapped onto the edge names the sidecar declares, because they are only two of arbitrarily many edge attributes a sidecar may bundle (`weight_NMF_*`, `fc`, `local_connectivity`). Two guards keep that from doing harm: an array already named by a template edge is left alone, so a directly-named `length` edge is never renamed away; and where no template edge matches, the generic key stays, since the loader resolves `weight`/`length` directly and renaming on a guess would misplace the matrix.

    Parameters
    ----------
    network : Network
        Network instance to save.
    yaml_path : str or Path
        Output path for sidecar (extension overridden by sidecar_format).
    binary_format : str
        "h5" (default), "zarr", or "csv".
    sidecar_format : str
        "yaml" (default) or "json".
    """
    yaml_path = Path(yaml_path)
    sidecar_ext = ".json" if sidecar_format == "json" else ".yaml"
    sidecar_path = yaml_path.with_suffix(sidecar_ext)

    # Every edge matrix, resident or still in the source companion (the resident one wins), keyed by edge name for the writer. Read under one open handle, and read BEFORE the companion is opened for writing: a save onto the network's own companion truncates it, and everything still lazy would be gone.
    store = getattr(network, "_store", None)
    with store if hasattr(store, "__enter__") else nullcontext():
        arrays = dict(store.arrays) if store else {}
        edge_params = network.edge_parameter_arrays() if hasattr(network, "edge_parameter_arrays") else {}
        carried = _carry_through(network)
    arrays.update(network.edge_arrays() if hasattr(network, "edge_arrays") else {})

    # Network._items() hides _cached_* attrs, so yaml_dumper works directly
    meta = yaml_loader.load_as_dict(yaml_dumper.dumps(network))

    # Remap the generic `weight`/`length` keys onto the sidecar's declared edge names — see this function's docstring for why, and for the two guards.
    if arrays:
        from tvbo.classes.network import _LENGTH_MEASURES, _WEIGHT_MEASURES

        names = [te.get("name") or te.get("label") for te in _template_edges(meta.get("edges", []))]
        names = [nm for nm in names if nm]
        nameset = set(names)
        if "weight" in arrays and "weight" not in nameset:
            w_name = next((nm for nm in names if nm.lower() in _WEIGHT_MEASURES), None) or next(
                (nm for nm in names if nm.lower() not in _LENGTH_MEASURES), None
            )
            if w_name and w_name != "weight":
                arrays[w_name] = arrays.pop("weight")
        if "length" in arrays and "length" not in nameset:
            l_name = next((nm for nm in names if nm.lower() in _LENGTH_MEASURES), None)
            if l_name and l_name != "length":
                arrays[l_name] = arrays.pop("length")

    if not arrays and carried.get("mesh/vertices") is None:
        _write_v07_sidecar(network, sidecar_path, sidecar_format)
        return

    companion = sidecar_path.with_suffix(f".{binary_format}")
    meta["data_file"] = companion.name
    meta["_sidecar_name"] = sidecar_path.name

    if binary_format in ("h5", "hdf5"):
        import h5py

        with h5py.File(companion, "w") as f:
            _write_edges(f, meta, arrays, edge_params)
            _write_nodes(f, network, carried)
            _write_mesh(f, network, carried)

    elif binary_format == "zarr":
        import zarr

        z = zarr.open(str(companion), mode="w")
        _write_edges(z, meta, arrays, edge_params)
        _write_nodes(z, network, carried)
        _write_mesh(z, network, carried)

    elif binary_format == "csv":
        # CSV: one file = one matrix = first template edge only
        edges = _template_edges(meta.get("edges", []))
        name = (edges[0].get("name") or edges[0].get("label")) if edges else next(iter(arrays))
        np.savetxt(companion, arrays[name], delimiter=" ", fmt="%.8g")

    meta.pop("_sidecar_name", None)
    # Write v0.7-compliant sidecar
    _write_v07_sidecar(network, sidecar_path, sidecar_format, data_file=companion.name)
