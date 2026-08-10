"""Format converters: BEP017 export.

Uses relmat_entities() + build_path() from §6.5 for BIDS-compliant filenames — no manual filename construction.

TVB converters have moved to :mod:`tvbo.adapters.tvb`.
"""

import json
import numpy as np
from pathlib import Path

from tvbo.data.network_io import _template_edges


def relmat_entities(network) -> dict:
    """Extract pybids entities from a tvbo Network instance or dict (§6.5).

    Accepts both a Network object (direct attribute access) and a raw sidecar dict (for standalone usage). Returns entity dict for
    ``build_path()``.

    Parameters
    ----------
    network : Network or dict
        Network instance or raw sidecar dict.

    Returns
    -------
    dict
        Entity dict with keys matching RELMAT_PATTERNS placeholders.
    """
    if isinstance(network, dict):
        bids = network.get("bids", {})
        atlas = network.get("parcellation", {}).get("atlas", {}).get("name", "")
        descriptor = network.get("descriptor")
    else:
        bids = getattr(network, "bids", None) or {}
        if not isinstance(bids, dict):
            bids = {
                k: getattr(bids, k, None)
                for k in ("template", "subject", "session", "cohort", "reconstruction", "space", "segmentation", "scale")
            }
        parc = getattr(network, "parcellation", None)
        atlas = ""
        if parc:
            atlas_obj = getattr(parc, "atlas", None)
            if atlas_obj:
                atlas = getattr(atlas_obj, "name", "") or ""
        descriptor = getattr(network, "descriptor", None)

    return {
        "template": bids.get("template"),
        "subject": bids.get("subject"),
        "session": bids.get("session"),
        "cohort": bids.get("cohort"),
        "reconstruction": bids.get("reconstruction"),
        "space": bids.get("space"),
        "atlas": atlas or None,
        "segmentation": bids.get("segmentation"),
        "scale": bids.get("scale"),
        "description": descriptor,
    }


def sensor_entities(network) -> dict:
    """Extract pybids entities from a sensor Network (§6.5).

    Returns entity dict suitable for ``build_path()`` with
    ``SENSOR_PATTERNS``.
    """
    if isinstance(network, dict):
        bids = network.get("bids", {})
        descriptor = network.get("descriptor")
    else:
        bids = getattr(network, "bids", None) or {}
        if not isinstance(bids, dict):
            bids = {k: getattr(bids, k, None) for k in ("template", "acquisition", "atlas")}
        descriptor = getattr(network, "descriptor", None)

    return {
        "template": bids.get("template"),
        "acquisition": bids.get("acquisition"),
        "atlas": bids.get("atlas"),
        "description": descriptor,
    }


def to_bep017(network, output_dir):
    """Export a tvbo Network to BEP017-compatible per-measure files.

    Each template edge becomes one ``meas-<name>_relmat.dense.tsv`` +
    JSON sidecar. Filenames use pybids ``build_path`` (§6.5). Reads data directly from the Network object — no YAML round-trip needed.

    Parameters
    ----------
    network : Network
        Network instance with loaded arrays.
    output_dir : str or Path
        Output directory for BEP017 files.
    """
    store = getattr(network, "_store", None)
    arrays = store.arrays if store else getattr(network, "_arrays", {})
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    base_entities = relmat_entities(network)
    atlas = base_entities.get("atlas", "unknown")

    edges = network.edges or []
    for e in _template_edges(edges):
        name = e.label if hasattr(e, "label") else e["label"]
        if name not in arrays:
            continue

        # BEP017 uses meas-<name> per file (one matrix per file)
        tsv_name = f"atlas-{atlas}_meas-{name}_relmat.dense.tsv"
        np.savetxt(out / tsv_name, arrays[name], delimiter="\t", fmt="%.8g")

        # JSON sidecar (tvbo fields → BEP017 field names)
        _get = (lambda k, d: e.get(k, d)) if isinstance(e, dict) else (lambda k, d: getattr(e, k, d))
        sidecar = {
            "RelationshipMeasure": name,
            "Directed": _get("directed", False),
            "Weighted": _get("weighted", True),
            "ValidDiagonal": _get("valid_diagonal", False),
            "NonNegative": _get("non_negative", True),
        }
        prov = getattr(network, "provenance", None)
        if prov and getattr(prov, "generated_by", None):
            sidecar["Software"] = prov.generated_by
        unit = _get("unit", None)
        if unit:
            sidecar["MeasureUnits"] = unit

        json_name = tsv_name.replace(".dense.tsv", ".json")
        (out / json_name).write_text(json.dumps(sidecar, indent=2))

    # Node indices TSV
    nodes = network.nodes or []
    if nodes:
        lines = ["matrix_index\tnode_file\tnode_index\tlabel"]
        for node in nodes:
            nid = node.id if hasattr(node, "id") else node["id"]
            nlabel = node.label if hasattr(node, "label") else node.get("label", "")
            lines.append(f"{nid}\tatlas-{atlas}\t{nid}\t{nlabel}")
        (out / f"atlas-{atlas}_nodeindices.tsv").write_text("\n".join(lines))


# TVB converters have been consolidated into tvbo.adapters.tvb
# Backward-compatible re-exports:
from tvbo.adapters.tvb import from_tvb_zip, from_tvb, from_tvb_surface  # noqa: F401, E402
