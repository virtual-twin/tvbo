"""High-level Network I/O. Dispatches by companion file extension.

Supported companion formats:
  .h5 / .hdf5  — HDF5 (default, best compression)
  .zarr/        — Zarr (cloud-native, S3-compatible)
  .csv          — CSV (one file per array, each named on the sidecar record of what it holds)

Sidecars are parsed and built by tvbo's own loader (:func:`tvbo.utils.yaml_loader.load_as_dict` + :func:`~tvbo.utils.yaml_loader.construct`), the one SimulationExperiment and SimulationStudy load through, so a sidecar may use ``!include``, ``<<:`` merge keys and anchors, and construction goes through the schema's constructor classes. Never use raw yaml.safe_load → cls(**dict) for LinkML classes.

See §12.2 of the tvbo HDF5 format proposal v0.7.
"""

import json
import os
import shutil
import tempfile
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import yaml as _yaml
from linkml_runtime.dumpers import yaml_dumper
from linkml_runtime.loaders import yaml_loader

from tvbo.utils import yaml_loader as tvbo_yaml_loader

SCHEMA_VERSION = "tvb-datamodel/0.7.0"

from tvbo.data.matrix_io import (
    MESH_FIELDS,
    LazyArrayStore,
    _create_ds,
    auto_format,
    declared_arrays,
    edge_name,
    is_csv,
    named_entries,
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
    for e in template_edges(meta.get("edges", [])):
        edge_meta[edge_name(e)] = e

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


def _write_v07_sidecar(network, sidecar_path: Path, sidecar_format: str, data_file: str = None, links: dict = None):
    """Write a strict v0.7-compliant YAML or JSON sidecar.

    Dumps the network via LinkML, post-processes to v0.7 form, and writes with controlled key ordering.

    A ``data_file``-backed connectome is materialised — its arrays live in the companion file — so the directives that would rebuild them from a source are dropped and a reload uses the companion. ``parcellation`` is deliberately not one of them: it also states which atlas this network's node labels belong to, which is what a ``by_label`` crosswalk resolves against, so dropping it would leave a saved network unable to reconcile any label the atlas spells differently. Keeping it is safe because ``load_network`` defers connectivity until the companion store is attached, so it no longer re-expands to the atlas node set.

    *links* names the CSVs of a CSV save on the records of what they hold, and every save takes the CSV links a network read from CSVs carries off its records first (:func:`_place_links`).
    """
    if data_file:
        network.data_file = data_file

    meta = yaml_loader.load_as_dict(yaml_dumper.dumps(network))
    meta = _purify(meta)
    if data_file:
        for _directive in ("bids_dir", "graph_generator", "tractogram"):
            meta.pop(_directive, None)
    _place_links(meta, links, len(network.nodes or []))
    meta = _v07_postprocess(meta)

    if sidecar_format == "json":
        import json

        with open(sidecar_path, "w") as f:
            json.dump(meta, f, indent=2)
    else:
        with open(sidecar_path, "w") as f:
            _yaml.dump(meta, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _carry_through(network) -> dict:
    """The mesh, per-node and declared datasets a re-save copies across, read while the source companion is still readable.

    ``save_network`` opens the destination with truncation, and a network saved back onto its own companion would then find every lazy dataset gone — silently, because a missing mesh is indistinguishable from a network that never had one. Carried are the mesh, the node mapping, every ``nodes/<name>`` dataset the companion holds or the network holds in memory, and every array the network declares as its own (:func:`~tvbo.data.matrix_io.declared_arrays`), wherever its path.
    """
    if not hasattr(network, "array"):
        return {}
    nm_path = getattr(network, "node_mapping", None) or "/nodes/parent_index"
    paths = [*MESH_FIELDS, nm_path.lstrip("/")]
    src = getattr(network, "_store", None)
    if src is not None and hasattr(src, "dataset_keys"):
        try:
            paths.extend(src.dataset_keys("nodes"))
        except (AttributeError, OSError, KeyError):
            pass
    paths.extend(key for key in network._resident() if key.startswith("nodes/"))
    paths.extend(declared_arrays(network))
    out = {}
    for path in dict.fromkeys(paths):
        value = network.array(path)
        if value is not None:
            out[path] = value
    return out


def _write_nodes(store, network, carried: dict):
    """Write node-level data to a store.

    Persists:
    - ``nodes/parent_index``: hierarchical node mapping (if present)
    - ``nodes/coordinates``: (N,3) float32 array from Node.position (if present)
    - every other array :func:`_carry_through` carries but the mesh, copied through at its path

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
    for key in (k for k in carried if k not in written and k not in MESH_FIELDS):
        group, _, name = key.rpartition("/")
        _create_ds(store.require_group(group) if group else store, name, data=carried[key])


def _write_companion(store, network, meta: dict, arrays: dict, edge_params: dict, carried: dict):
    """Write the layers (:func:`_write_edges`), the node arrays (:func:`_write_nodes`) and the mesh (:func:`_write_mesh`) to *store*, an HDF5 file or a Zarr group."""
    _write_edges(store, meta, arrays, edge_params)
    _write_nodes(store, network, carried)
    _write_mesh(store, meta.get("mesh"), carried)


def _write_mesh(store, record: dict | None, carried: dict):
    """Write mesh data (vertices, elements, normals) to ``/mesh/`` group, with the element type and counts of the sidecar's ``mesh`` *record*, as dumped, as the group's attributes.

    Each array comes from :func:`_carry_through` — resident if an adapter such as ``from_tvb_surface`` set it, read off the source companion before this store was opened otherwise — so a re-save of a surface network it never touched is lossless, including a re-save onto that same companion.
    """
    import numpy as _np

    vertices = carried.get("mesh/vertices")
    if vertices is None:
        return

    mesh_grp = store.require_group("mesh")

    if record:
        mesh_grp.attrs["tvbo_class"] = "tvbo:Mesh"
        for key in ("element_type", "number_of_vertices", "number_of_elements"):
            if record.get(key) is not None:
                mesh_grp.attrs[key] = json.loads(json.dumps(record[key]))  # h5py refuses a str subclass

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


def _csv_entities(path: str) -> list[str]:
    """The entities naming the CSV of the array at *path*: ``meas-<layer>`` (BEP017) for an edge layer, with ``param-<name>`` for one of its edge parameters, and ``<group>-<name>`` for any other array, each value cut to letters and digits as every BIDS entity value is."""
    from tvbo.adapters.bids import entity_value

    group, _, rest = path.partition("/")
    if group == "edges":
        layer, _, param = rest.partition("/edge_parameters/")
        return [f"meas-{entity_value(layer)}", *([f"param-{entity_value(param)}"] if param else [])]
    return [f"{entity_value(group)}-{entity_value(rest)}"]


def _write_csv_arrays(
    network, sidecar_path: Path, into: Path, meta: dict, arrays: dict, edge_params: dict, carried: dict
) -> dict:
    """Write every array an HDF5 companion would hold to a CSV of its own in *into*, named for the sidecar at *sidecar_path*, and return the links :func:`_place_links` names them by.

    The arrays are what :func:`_write_edges`, :func:`_write_nodes` and :func:`_write_mesh` store, written to an HDF5 file in memory and read back, so the CSVs hold the arrays an HDF5 save holds, at the same dtype and sparse format; a sparse matrix is written as one line per stored entry and never as the dense matrix (:func:`~tvbo.data.matrix_io.write_csv_array`). The coordinates are left out when every node has a position, which the sidecar then holds them as. A file takes the sidecar's name with the entities of :func:`_csv_entities`: before the suffix of a BIDS name, where BEP017 places ``meas-`` in a relationship matrix's name (``exp-1_network.yaml`` gives ``exp-1_meas-weight_network.csv``), and at the end of any other name (``n.yaml`` gives ``n_nodes-parentindex.csv``). Declared layers come first, so the first file is the first declared layer's.

    Refused before any file is written, naming each: an array that is not 1-D or 2-D, which a CSV holds as no rows; a sparse layer that is not a row for each node by a column for each node or for each of its ``dimension_labels``, the shape it reads back at; and arrays whose names give one file name.
    """
    import io

    import h5py
    from scipy import sparse

    from tvbo.data.matrix_io import _read_values, read_matrix, write_csv_array

    held = {}
    with h5py.File(io.BytesIO(), "w") as f:
        _write_companion(f, network, meta, arrays, edge_params, carried)

        def collect(path, node):
            if isinstance(node, h5py.Group) and "format" in node.attrs:
                held[path] = read_matrix(node)
            elif isinstance(node, h5py.Dataset) and "format" not in node.parent.attrs:
                held[path] = _read_values(node)

        f.visititems(collect)
    nodes = network.nodes or []
    if nodes and all(getattr(node, "position", None) is not None for node in nodes):
        held.pop("nodes/coordinates", None)
    edges = {edge_name(e): e for e in template_edges(meta.get("edges", []))}
    rank = {f"edges/{name}": i for i, name in enumerate(edges)}
    held = dict(sorted(held.items(), key=lambda item: (rank.get(item[0], len(rank)), item[0])))

    flat = [path for path, value in held.items() if np.ndim(value) not in (1, 2)]
    if flat:
        raise ValueError(
            f"A CSV holds an array as rows, so saving {sidecar_path.name} with CSV companions would lose {', '.join(flat)}, which {'is' if len(flat) == 1 else 'are'} not 1-D or 2-D. Save it with binary_format='h5' or 'zarr', which hold them."
        )
    n = len(nodes)
    misshapen = [
        f"{path} ({value.shape[0]} x {value.shape[1]})"
        for path, value in held.items()
        if sparse.issparse(value)
        and value.shape != (n, len((edges.get(path.split("/")[1]) or {}).get("dimension_labels") or []) or n)
    ]
    if misshapen:
        raise ValueError(
            f"A sparse layer's CSV holds its stored entries and not its shape, which reads back as a row for each of the {n} nodes by a column for each node or for each of its edge's dimension_labels, so saving {sidecar_path.name} with CSV companions would misread {', '.join(misshapen)}. Save it with binary_format='h5' or 'zarr'."
        )
    *head, suffix = sidecar_path.stem.split("_")
    bids = bool(head) and all("-" in part for part in head) and "-" not in suffix
    files = {
        path: "_".join([*head, *_csv_entities(path), suffix] if bids else [sidecar_path.stem, *_csv_entities(path)]) + ".csv"
        for path in held
    }
    clash = [path for path, file in files.items() if list(files.values()).count(file) > 1]
    if clash:
        raise ValueError(
            f"The arrays {clash} would share one CSV, since a file name keeps only the letters and digits of each name. Rename them, or save with binary_format='h5' or 'zarr'."
        )

    links = {}
    for path, value in held.items():
        write_csv_array(into / files[path], value)
        links[path] = {
            "file": files[path],
            "dtype": str(value.dtype),
            "shape": tuple(value.shape),
            "format": value.format if sparse.issparse(value) else "dense",
        }
    return links


def _place_links(meta: dict, links: dict | None, n_nodes: int) -> None:
    """Name each CSV of a CSV save on the sidecar record of the array it holds, once every CSV link is off the records, which is all an HDF5 or Zarr save does: its companion holds each array under its path.

    *links* maps each array's path to its CSV's ``file``, ``dtype``, ``shape`` and sparse ``format`` (:func:`_write_csv_arrays`). A layer's CSV becomes its template edge's ``data_file``, with the layer's dtype unless float64 and its format unless dense; an edge parameter's the parameter's ``source``, with its dtype unless float64; a mesh array's the ``mesh`` field :data:`~tvbo.data.matrix_io.MESH_FIELDS` names; and any other array's the ``source`` of the parameter that declares it (:func:`~tvbo.data.matrix_io.declared_arrays`), with its ``shape`` and its dtype unless float64. An array no record declares is declared with the least the save knows: a template edge for a layer, and otherwise a parameter named by the last part of its path, whose ``measure`` is that path, on ``node_template`` when the array has a row for each of the *n_nodes* nodes and on the network when it has not. A name another parameter already has is refused, naming both.
    """
    edges = {edge_name(e): e for e in template_edges(meta.get("edges"))}
    for edge in edges.values():
        edge.pop("data_file", None)
        for _, param in named_entries(edge.get("parameters")):
            if is_csv(param.get("source")):
                param.pop("source")
    for _, param in declared_arrays(meta).values():
        if is_csv(param.get("source")):
            param.pop("source")
    mesh = meta.get("mesh")
    if isinstance(mesh, dict):
        for field, _ in MESH_FIELDS.values():
            if is_csv(mesh.get(field)):
                mesh.pop(field)
        if not mesh:
            meta.pop("mesh")
    declared = declared_arrays(meta)
    for path, link in (links or {}).items():
        dtype = {} if link["dtype"] == "float64" else {"dtype": link["dtype"]}
        group, _, rest = path.partition("/")
        if group == "edges":
            layer, _, param = rest.partition("/edge_parameters/")
            if layer not in edges:
                edges[layer] = {"label": layer}
                meta.setdefault("edges", []).append(edges[layer])
            if param:
                edges[layer].setdefault("parameters", {}).setdefault(param, {}).update(source=link["file"], **dtype)
            else:
                edges[layer].update(
                    data_file=link["file"], **dtype, **({} if link["format"] == "dense" else {"format": link["format"]})
                )
        elif path in MESH_FIELDS:
            meta.setdefault("mesh", {})[MESH_FIELDS[path][0]] = link["file"]
        else:
            record = declared[path][1] if path in declared else _declare_array(meta, path, link["shape"][0] == n_nodes)
            record.pop("dtype", None)
            record.update(source=link["file"], shape=str(link["shape"]), **dtype)


def _declare_array(meta: dict, path: str, per_node: bool) -> dict:
    """The parameter :func:`_place_links` declares an array at *path* no record declares with: named by the last part of the path, on ``node_template`` when it runs over the nodes (*per_node*) and on the network otherwise."""
    name = path.rpartition("/")[2]
    template = meta.get("node_template") or {}
    taken = {
        key: where
        for where, params in (("network", meta.get("parameters")), ("node_template", template.get("parameters")))
        for key, _ in named_entries(params)
    }
    if name in taken:
        raise ValueError(
            f"A CSV save declares the array {path}, which no record declares, as the parameter {name!r}, and the {taken[name]} has a parameter {name!r} already. Declare {path} under another name, with `measure: {path}`, or save with binary_format='h5' or 'zarr'."
        )
    owner = meta.setdefault("node_template", template) if per_node else meta
    record = owner.setdefault("parameters", {}).setdefault(name, {})
    record["measure"] = path
    return record


def _unholdable(binary_format: str, dtype) -> bool:
    """Whether *binary_format* cannot hold an array of *dtype*: an ``object`` array in any format, a datetime64 or timedelta64 in HDF5 and in CSV, whose arrays pass through HDF5 (:func:`_write_csv_arrays`), a long double in Zarr, and a structured or void array in CSV. Text (``<U``) every format holds (:func:`_as_stored`)."""
    dtype = np.dtype(dtype)
    return (
        dtype.kind == "O"
        or (dtype.kind in "Mm" and binary_format in ("h5", "hdf5", "csv"))
        or (dtype.char in "gG" and binary_format == "zarr")
        or (dtype.kind == "V" and binary_format == "csv")
    )


def _refuse_unholdable(network, sidecar_path: Path, binary_format: str, arrays: dict, edge_params: dict, carried: dict):
    """Refuse a save of arrays *binary_format* cannot hold (:func:`_unholdable`) before anything is written, naming the network and each array with its dtype and the formats that hold it."""
    every = {
        **{f"edges/{name}": value for name, value in arrays.items()},
        **{
            f"edges/{layer}/edge_parameters/{name}": value
            for layer, params in edge_params.items()
            for name, value in params.items()
        },
        **carried,
    }
    refused = {path: value.dtype for path, value in every.items() if _unholdable(binary_format, value.dtype)}
    if not refused:
        return

    def described(path, dtype) -> str:
        others = [repr(other) for other in ("h5", "zarr", "csv") if not _unholdable(other, dtype)]
        held = f"which binary_format={' or '.join(others)} holds" if others else "which no format holds"
        return f"{path} ({dtype.type.__name__ if dtype.char in 'gG' else dtype}), {held}"

    label = getattr(network, "label", None)
    named = f"{label!r} ({sidecar_path.name})" if label else sidecar_path.name
    advice = (
        " An object array holds Python objects, which no format stores: give text as a text array (np.asarray(values, dtype=str)), which every format holds, and numbers at a numeric dtype."
        if any(dtype.kind == "O" for dtype in refused.values())
        else ""
    )
    raise ValueError(
        f"The network {named} holds arrays binary_format={binary_format!r} cannot store, so nothing was written: {'; '.join(described(path, dtype) for path, dtype in refused.items())}.{advice}"
    )


def _as_stored(value, binary_format: str):
    """*value* as *binary_format* stores it: a text array (``<U``) as h5py's UTF-8 string type in HDF5 and in CSV, whose arrays pass through HDF5, which :func:`~tvbo.data.matrix_io._read_values` reads back as text; a mapping of arrays, as the edge parameters of each layer are held, with each array so; any other array as it is."""
    if isinstance(value, dict):
        return {key: _as_stored(item, binary_format) for key, item in value.items()}
    if binary_format == "zarr" or value.dtype.kind != "U":
        return value
    import h5py

    return value.astype(h5py.string_dtype())


def _staging_prefix(last: str) -> str:
    """The start of the name of each directory a save of the file *last* writes into (:func:`_saving_into`), which says what a killed save left behind."""
    return f".{last}.unfinished-tvbo-save-"


def _remove_stale_staging(folder: Path, last: str) -> None:
    """Remove each directory a save of *last* into *folder* left behind when it was killed, and no other: one named by :func:`_staging_prefix` for *last* whose process has ended."""
    prefix = _staging_prefix(last)
    for leftover in folder.iterdir():
        pid = leftover.name[len(prefix) :].partition("-")[0]
        if leftover.name.startswith(prefix) and leftover.is_dir() and pid.isdigit() and not _process_runs(int(pid)):
            shutil.rmtree(leftover)


def _process_runs(pid: int) -> bool:
    """Whether a process with the id *pid* runs, one this user may not signal included."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


@contextmanager
def _saving_into(folder: Path, last: str):
    """A fresh directory in *folder* for a save to write its files into, each moved onto its own name in *folder* once the block succeeds, *last* (the sidecar) after the others, replacing what was there. A block that raises leaves *folder* as it was, so a save that fails truncates no companion and leaves no sidecar naming files it did not write.

    The directory is named ``.<last>.unfinished-tvbo-save-<process id>-<random>`` (:func:`_staging_prefix`), so a save that is killed outright, which runs no cleanup, leaves a directory that says what it is, and the next save of *last* into *folder* removes it (:func:`_remove_stale_staging`).
    """
    _remove_stale_staging(folder, last)
    staging = Path(tempfile.mkdtemp(prefix=f"{_staging_prefix(last)}{os.getpid()}-", dir=folder))
    try:
        yield staging
        for written in sorted(staging.iterdir(), key=lambda p: p.name == last):
            target = folder / written.name
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            os.replace(written, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


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


def load_declared_network(path):
    """The network the file at *path* declares, read the way whatever uses that file reads it.

    A sidecar names its companion relative to itself, so it is loaded where it stands (:func:`load_network`). A fragment in a study's spec directory is written to be included by the study's recipe, so its relative paths mean the study root: it is constructed as the recipe constructs it, with its ``data_file``, generator or parcellation resolved against that root.
    """
    from tvbo.classes.network import Network
    from tvbo.utils import study_layout
    from tvbo.utils.source import loading_from

    path = Path(path).resolve()
    try:
        root = study_layout.study_root(path)
    except (StopIteration, LookupError, OSError):
        return load_network(path)
    if study_layout.study_path("spec", root).resolve() not in path.parents:
        return load_network(path)
    with loading_from(root / path.name):
        return tvbo_yaml_loader.construct(tvbo_yaml_loader.load_as_dict(path), Network)


def load_network(path):
    """Load a tvbo Network from a sidecar, or from a self-describing companion.

    Parsed and built by the loader ``SimulationExperiment.from_file`` and ``SimulationStudy.from_file`` use: :func:`tvbo.utils.yaml_loader.load_as_dict` expands ``!include``, ``<<:`` merge keys and anchors and normalises the document once, and :func:`~tvbo.utils.yaml_loader.construct` builds the Network from that mapping without dumping and reparsing it. The load runs inside ``loading_from(path)``, so relative paths anywhere in the sidecar resolve against its own directory, as a recipe's do.

    Two layouts are accepted:

    - **sidecar + companion** — a ``.yaml``/``.json`` metadata file whose ``data_file`` names the binary beside it;
    - **single self-describing file** — a ``.h5``/``.zarr`` carrying its own sidecar in the ``metadata`` root attribute (see :data:`EMBEDDED_METADATA_ATTR`). The file is its own array store, so one path is the whole Network.

    A sidecar whose ``data_file`` names another network's sidecar (:data:`~tvbo.data.dataref.NETWORK_SIDECARS`), as the dump of a curated network does, is no companion: that network is read as a ``data_file`` network reads it (``Network._resolve_from_data_file``), so the two readers of the slot agree. A sidecar that names itself is refused.

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
    from tvbo.data.dataref import NETWORK_SIDECARS
    from tvbo.utils.source import loading_from

    path = Path(path)

    embedded = read_embedded_metadata(path)
    if embedded is not None:
        meta_dict = tvbo_yaml_loader.load_as_dict(embedded)
        meta_dict.pop("data_file", None)
        data_file = path.name
    else:
        # `data_file` IS a schema slot, but `Network._resolve_from_data_file` treats it as an indirect reference to ANOTHER network's sidecar, so leaving it on the constructor kwargs would recurse.
        meta_dict = tvbo_yaml_loader.load_as_dict(path)
        data_file = meta_dict.pop("data_file", None)
    # Extract non-schema fields that are YAML-only metadata
    bids_meta = meta_dict.pop("bids", None)
    descriptor = meta_dict.pop("descriptor", None)
    meta_dict = tvbo_yaml_loader.strip_envelope(meta_dict)

    with loading_from(path):
        # This loader attaches connectivity itself, so the constructor must not resolve it: `data_file` is stripped above (it is an INDIRECT reference — `_resolve_from_data_file` reads the companion's own sidecar, which would recurse into this very load), and a sidecar that also declares `parcellation:` would otherwise fall through to the normative-database branch and cache an atlas connectome that shadows this file's real matrices. `_resolve` runs below, once `_store` is in place.
        net = tvbo_yaml_loader.construct({**meta_dict, "_defer_connectivity": True}, Network)

        # Attach non-schema metadata as attributes (used by bids_filename)
        if bids_meta:
            net.bids = bids_meta
        if descriptor:
            net.descriptor = descriptor

        names_sidecar = bool(data_file) and Path(str(data_file)).suffix.lower() in NETWORK_SIDECARS
        if names_sidecar and (path.parent / data_file).resolve() == path.resolve():
            raise ValueError(
                f"{path.name} names itself as its `data_file`; a sidecar names the companion that holds its arrays."
            )
        if data_file and not names_sidecar:
            data_path = path.parent / data_file
            net._store = LazyArrayStore(data_path, meta_dict)
            net.data_file = data_file

            # Restore mesh data from companion if present
            _load_mesh(net, data_path)
        else:
            net._store = None
            net.data_file = data_file

        # With a store attached the network is materialised, so this expands node/edge templates and subnetworks without re-entering the connectivity branches; without one, a `data_file` naming another sidecar and a bare `parcellation:` resolve here.
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

    With ``binary_format="csv"`` every array an HDF5 companion would hold is written to a CSV of its own (:func:`_write_csv_arrays`), which the sidecar names on the record of what it holds (:func:`_place_links`); the network's ``data_file`` names the first of them.

    Every array is read, and an array declared per node whose length is not the node count refused (``Network._check_per_node_lengths``), as is an array the format cannot hold (:func:`_refuse_unholdable`), before anything is written. A text array is stored as text in every format (:func:`_as_stored`). The files are written beside the sidecar and moved onto their names once all are written (:func:`_saving_into`), so a save that fails leaves the files it would have replaced as they were.

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

    from tvbo.classes.network import _is_weight_name, _read_held, _unheld_weight

    # Every edge matrix, resident or still in the source companion (the resident one wins), keyed by edge name for the writer. Read under one open handle, and read BEFORE the companion is opened for writing: a save onto the network's own companion truncates it, and everything still lazy would be gone. A layer declared with `used:` is never read from the companion, under any spelling of its name: once frozen the companion holds no matrix for it, and its value is the one its binding read, if any. A layer the companion holds is read, and an error reading it is raised (`_read_held`); a declared layer it does not hold is carried as in memory, declared and with no matrix, except a weight with nothing in memory either, which is refused below as `matrix()` refuses it.
    store = getattr(network, "_store", None)
    arrays, unheld = {}, []
    with store if hasattr(store, "__enter__") else nullcontext():
        for name in store.arrays if store else ():
            if network.sources_layer(name):
                continue
            if name in store:
                arrays[name] = _read_held(store, name)
            else:
                unheld.append(name)
        edge_params = network.edge_parameter_arrays() if hasattr(network, "edge_parameter_arrays") else {}
        carried = _carry_through(network)
    arrays.update(network.edge_arrays() if hasattr(network, "edge_arrays") else {})

    # Network._items() hides _cached_* attrs, so yaml_dumper works directly
    meta = yaml_loader.load_as_dict(yaml_dumper.dumps(network))

    # Remap the generic `weight`/`length` keys onto the sidecar's declared edge names — see this function's docstring for why, and for the two guards.
    if arrays:
        from tvbo.classes.network import _LENGTH_MEASURES, _WEIGHT_MEASURES

        names = [edge_name(te) for te in template_edges(meta.get("edges", []))]
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

    if unheld:
        missing = next((name for name in unheld if _is_weight_name(name) and name not in arrays), None)
        if missing is not None:
            raise _unheld_weight(missing, store)

    unread = network.unread_sourced_layers() if hasattr(network, "unread_sourced_layers") else []
    for name in unread:
        arrays.pop(name, None)  # a stored matrix under a layer read per run would stand in for it on reload

    if hasattr(network, "_check_per_node_lengths"):
        network._check_per_node_lengths()
    _refuse_unholdable(network, sidecar_path, binary_format, arrays, edge_params, carried)
    arrays, edge_params, carried = (_as_stored(group, binary_format) for group in (arrays, edge_params, carried))

    with _saving_into(sidecar_path.parent, sidecar_path.name) as staging:
        if not arrays and not carried and not unread:
            _write_v07_sidecar(network, staging / sidecar_path.name, sidecar_format)
            return

        companion = staging / f"{sidecar_path.stem}.{binary_format}"
        meta["data_file"] = companion.name
        meta["_sidecar_name"] = sidecar_path.name
        data_file, links = companion.name, None

        if binary_format in ("h5", "hdf5"):
            import h5py

            with h5py.File(companion, "w") as f:
                _write_companion(f, network, meta, arrays, edge_params, carried)

        elif binary_format == "zarr":
            import zarr

            z = zarr.open(str(companion), mode="w")
            _write_companion(z, network, meta, arrays, edge_params, carried)

        elif binary_format == "csv":
            links = _write_csv_arrays(network, sidecar_path, staging, meta, arrays, edge_params, carried)
            data_file = next(link["file"] for link in links.values())

        meta.pop("_sidecar_name", None)
        # Write v0.7-compliant sidecar
        _write_v07_sidecar(network, staging / sidecar_path.name, sidecar_format, data_file=data_file, links=links)
