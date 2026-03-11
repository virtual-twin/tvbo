"""High-level Network I/O. Dispatches by companion file extension.

Supported companion formats:
  .h5 / .hdf5  — HDF5 (default, best compression)
  .zarr/        — Zarr (cloud-native, S3-compatible)
  .csv          — CSV legacy (one file = one matrix = first template edge)

YAML sidecars are loaded via linkml_runtime.loaders.yaml_loader — the
same loader used by Dynamics, Coupling, and SimulationExperiment. This
ensures schema validation and proper nested object construction. Never
use raw yaml.safe_load → cls(**dict) for LinkML classes.

See §12.2 of the tvbo HDF5 format proposal v0.7.
"""
import numpy as np
from pathlib import Path
from linkml_runtime.loaders import yaml_loader, json_loader
from linkml_runtime.dumpers import yaml_dumper, json_dumper

from tvbo.data.matrix_io import (
    read_matrix, write_matrix, auto_format, LazyArrayStore,
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

# ── Helpers ───────────────────────────────────────────────────────────


def _template_edges(edges) -> list:
    """Template edges = entries without source/target (matrix measures).

    Works with both dicts (from yaml_loader.load_as_dict) and
    LinkML Edge objects (from Network.edges).
    """
    if not edges:
        return []
    result = []
    for e in edges:
        if isinstance(e, dict):
            if not e.get("source"):
                result.append(e)
        else:
            if not getattr(e, "source", None):
                result.append(e)
    return result


def _read_edges(store, meta: dict) -> tuple[dict, dict]:
    """Read all template-edge matrices + edge parameters from a store.

    Works identically for h5py.File and zarr.Group — both support
    ``"path" in store`` and ``store["path"]`` access.
    """
    edges = _template_edges(meta.get("edges", []))
    arrays, params = {}, {}

    # Determine edge names: from template metadata or store discovery
    if edges:
        edge_names = [e.get("name") or e.get("label") for e in edges]
    elif "edges" in store:
        edge_names = list(store["edges"])
    else:
        edge_names = []

    for name in edge_names:
        edge_path = f"edges/{name}"
        if edge_path not in store:
            continue
        arrays[name] = read_matrix(store[edge_path])
        params[name] = {}
        ep_path = f"{edge_path}/edge_parameters"
        if ep_path in store:
            for pname in store[ep_path]:
                params[name][pname] = read_matrix(store[f"{ep_path}/{pname}"])
    return arrays, params


def _write_edges(store, meta: dict, arrays: dict, edge_params: dict):
    """Write all template-edge matrices + edge parameters to a store."""
    edge_meta = {}
    for e in _template_edges(meta.get("edges", [])):
        edge_meta[e.get("name") or e.get("label")] = e

    store.attrs["tvbo_class"] = "tvbo:Network"
    store.attrs["sidecar_file"] = str(meta.get("_sidecar_name", ""))
    store.attrs["schema_version"] = str(meta.get(
        "schema_version", "tvb-datamodel/0.7.0"))

    for name, matrix in arrays.items():
        m = edge_meta.get(name, {})
        fmt = m.get("format", auto_format(matrix))
        grp = store.create_group(f"edges/{name}")
        grp.attrs["tvbo_class"] = "tvbo:Matrix"
        for attr in ("directed", "unit"):
            if attr in m:
                # Cast to native Python types — LinkML extended_str
                # and extended_bool are not directly serializable by h5py.
                val = m[attr]
                if isinstance(val, bool):
                    grp.attrs[attr] = val
                else:
                    grp.attrs[attr] = str(val)
        write_matrix(grp, matrix, fmt=str(fmt))

        for pname, pmatrix in edge_params.get(name, {}).items():
            pg = grp.require_group("edge_parameters").create_group(pname)
            pg.attrs["tvbo_class"] = "tvbo:Parameter"
            pfmt = fmt  # default to same format as parent edge
            if "parameters" in m and isinstance(m["parameters"], dict):
                p_meta = m["parameters"].get(pname, {})
                if isinstance(p_meta, dict) and "format" in p_meta:
                    pfmt = p_meta["format"]
            write_matrix(pg, pmatrix, fmt=pfmt)


def _write_nodes(store, network):
    """Write node-level data (e.g. ``nodes/parent_index``) to a store.

    Called after ``_write_edges`` during ``save_network``.
    """
    try:
        data = object.__getattribute__(network, "_node_mapping_data")
    except AttributeError:
        data = None
    if data is None:
        return

    nm_path = getattr(network, "node_mapping", None) or "/nodes/parent_index"
    # Strip leading "/" and split into group + dataset name
    key = nm_path.lstrip("/")
    parts = key.rsplit("/", 1)
    if len(parts) == 2:
        grp_path, ds_name = parts
        grp = store.require_group(grp_path)
        grp.create_dataset(ds_name, data=data, dtype="int32")
    else:
        store.create_dataset(key, data=data, dtype="int32")


# ── Load ──────────────────────────────────────────────────────────────

def load_network(yaml_path):
    """Load a tvbo Network from YAML/JSON sidecar + companion reference.

    Uses linkml yaml_loader or json_loader to construct a schema-validated
    Network instance directly — same pattern as Dynamics.from_file().

    Arrays are NOT loaded into memory. A LazyArrayStore is attached
    that loads arrays on first access (e.g., net.weights_matrix).

    Parameters
    ----------
    yaml_path : str or Path
        Path to YAML or JSON sidecar file.

    Returns
    -------
    Network
        Fully constructed tvbo.Network with lazy array references.
    """
    from tvbo.data.tvbo_data.connectomes import Network

    yaml_path = Path(yaml_path)
    ext = yaml_path.suffix.lower()

    # Load as dict first to extract data_file (not a schema field)
    meta_dict = yaml_loader.load_as_dict(str(yaml_path))
    data_file = meta_dict.pop("data_file", None)
    # Extract non-schema fields that are YAML-only metadata
    bids_meta = meta_dict.pop("bids", None)
    descriptor = meta_dict.pop("descriptor", None)
    meta_dict.pop("schema_version", None)

    # Reconstruct clean YAML without non-schema fields for LinkML loader
    import yaml as _yaml
    clean_yaml = _yaml.dump(meta_dict, Dumper=_yaml.SafeDumper)
    net = yaml_loader.loads(clean_yaml, Network)

    # Attach non-schema metadata as attributes (used by bids_filename)
    if bids_meta:
        net.bids = bids_meta
    if descriptor:
        net.descriptor = descriptor

    # Attach lazy array store (no arrays loaded yet).
    if data_file:
        data_path = yaml_path.parent / data_file
        net._store = LazyArrayStore(data_path, meta_dict)
        net.data_file = data_file
    else:
        net._store = None

    return net


# ── Save ──────────────────────────────────────────────────────────────

def save_network(network, yaml_path, binary_format: str = "h5",
                 sidecar_format: str = "yaml"):
    """Save a tvbo Network as sidecar + binary companion.

    Uses LinkML yaml_dumper or json_dumper for schema-valid sidecar
    output — no manual field unpacking or yaml.dump() calls.

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

    # Get arrays: merge lazy store with user-set _arrays (user wins)
    store = getattr(network, "_store", None)
    base_arrays = dict(store.arrays) if store else {}
    user_arrays = getattr(network, "_arrays", {})
    arrays = {**base_arrays, **user_arrays}

    edge_params = (dict(store.edge_params) if store
                   else getattr(network, "_edge_params", {}))

    # Also pick up _cached_weights / _cached_lengths from from_matrix()
    if not arrays:
        cw = getattr(network, "_cached_weights", None)
        cl = getattr(network, "_cached_lengths", None)
        if cw is not None:
            arrays = {"weights": cw}
            if cl is not None:
                arrays["lengths"] = cl

    # Network._items() hides _cached_* attrs, so yaml_dumper works directly
    meta = yaml_loader.load_as_dict(yaml_dumper.dumps(network))

    if not arrays:
        # Metadata-only sidecar (no companion file)
        if sidecar_format == "json":
            json_dumper.dump(network, str(sidecar_path), inject_type=False)
        else:
            yaml_dumper.dump(network, str(sidecar_path))
        return

    companion = sidecar_path.with_suffix(f".{binary_format}")
    meta["data_file"] = companion.name
    meta["_sidecar_name"] = sidecar_path.name

    if binary_format in ("h5", "hdf5"):
        import h5py
        with h5py.File(companion, "w") as f:
            _write_edges(f, meta, arrays, edge_params)
            _write_nodes(f, network)

    elif binary_format == "zarr":
        import zarr
        z = zarr.open(str(companion), mode="w")
        _write_edges(z, meta, arrays, edge_params)
        _write_nodes(z, network)

    elif binary_format == "csv":
        # CSV: one file = one matrix = first template edge only
        edges = _template_edges(meta.get("edges", []))
        name = (edges[0].get("name") or edges[0].get("label")) if edges else next(iter(arrays))
        np.savetxt(companion, arrays[name], delimiter=" ", fmt="%.8g")

    meta.pop("_sidecar_name", None)
    # Write sidecar via LinkML dumper for schema-valid output
    network.data_file = companion.name
    if sidecar_format == "json":
        json_dumper.dump(network, str(sidecar_path), inject_type=False)
    else:
        yaml_dumper.dump(network, str(sidecar_path))
