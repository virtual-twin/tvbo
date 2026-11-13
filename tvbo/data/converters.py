"""Format converters: BEP017 export.

Uses relmat_entities() + build_path() from §6.5 for BIDS-compliant filenames — no manual filename construction.

TVB converters have moved to :mod:`tvbo.adapters.tvb`.
"""

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from tvbo.data.matrix_io import edge_name, template_edges


def _get(obj, key, default=None):
    """Read *key* off a datamodel object or the raw sidecar dict it round-trips through.

    Every public function here accepts both forms, so the accessor is written once instead of a `hasattr` ternary per field.
    """
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def relmat_entities(network) -> dict:
    """Extract pybids entities from a tvbo Network instance or dict (§6.5).

    Accepts both a Network object (direct attribute access) and a raw sidecar dict (for standalone usage). Returns entity dict for ``build_path()``.

    Parameters
    ----------
    network : Network or dict
        Network instance or raw sidecar dict.

    Returns:
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
                for k in (
                    "template",
                    "subject",
                    "session",
                    "cohort",
                    "reconstruction",
                    "hemi",
                    "space",
                    "segmentation",
                    "scale",
                    "density",
                )
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
        "hemi": bids.get("hemi"),
        "space": bids.get("space"),
        "atlas": atlas or None,
        "segmentation": bids.get("segmentation"),
        "scale": bids.get("scale"),
        "density": bids.get("density"),
        "description": descriptor,
    }


def sensor_entities(network) -> dict:
    """Extract pybids entities from a sensor Network (§6.5).

    Returns entity dict suitable for ``build_path()`` with ``SENSOR_PATTERNS``.
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


def _bep017_stem(network) -> str:
    """The name a BEP017 export gives its files before ``meas-`` and the suffix: the network's own relmat name without its suffix (:data:`~tvbo.data.network_io.RELMAT_PATTERNS`), else ``atlas-<atlas>`` for a network that names no template and no subject, else nothing."""
    from bids.layout.writing import build_path

    from tvbo.data.network_io import RELMAT_PATTERNS

    entities = relmat_entities(network)
    named = build_path(entities, RELMAT_PATTERNS)
    if named:
        return named.removesuffix("_relmat.h5")
    return f"atlas-{entities['atlas']}" if entities["atlas"] else ""


def to_bep017(network, output_dir):
    """Export a tvbo Network to BEP017-compatible per-measure files.

    Each template edge becomes one ``<stem>_meas-<label>_relmat.dense.tsv`` + JSON sidecar, ``<label>`` being the edge name as a BIDS label (:func:`~tvbo.adapters.bids.entity_value`) and the sidecar's ``RelationshipMeasure`` the name itself, and the nodes one ``<stem>_nodeindices.tsv``, where ``<stem>`` is the network's own BIDS name in the entity order of :data:`~tvbo.data.network_io.RELMAT_PATTERNS` (:func:`_bep017_stem`, through pybids ``build_path``, §6.5), so a subject's or a template's connectome keeps its ``sub-`` or ``tpl-``. Reads data directly from the Network object — no YAML round-trip needed.

    Parameters
    ----------
    network : Network
        Network instance with loaded arrays.
    output_dir : str or Path
        Output directory for BEP017 files.
    """
    from tvbo.adapters.bids import entity_value

    store = getattr(network, "_store", None)
    # `Network.arrays` is keyed by companion path; `edge_arrays` is the same set keyed by the edge name a BEP017 filename is built from.
    arrays = dict(store.arrays) if store else {}
    arrays.update(network.edge_arrays() if hasattr(network, "edge_arrays") else {})
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stem = _bep017_stem(network)
    atlas = relmat_entities(network)["atlas"]

    edges = network.edges or []
    labelled = {}
    for e in template_edges(edges):
        name = edge_name(e)
        if name not in arrays:
            continue
        label = entity_value(name)
        key = label.lower()  # macOS and Windows file systems ignore case, so meas-Weight and meas-weight are one file
        if key in labelled:
            raise ValueError(
                f"The edges {labelled[key]!r} and {name!r} are both written as meas-{label}, without regard to case, so a BEP017 export cannot hold both; rename one of them."
            )
        labelled[key] = name

        tsv_name = "_".join(filter(None, [stem, f"meas-{label}_relmat.dense.tsv"]))
        matrix = arrays[name]
        # 17 significant digits read every float64 back as itself.
        np.savetxt(out / tsv_name, matrix.toarray() if sp.issparse(matrix) else matrix, delimiter="\t", fmt="%.17g")

        # JSON sidecar (tvbo fields → BEP017 field names)
        sidecar = {
            "RelationshipMeasure": name,
            "Directed": _get(e, "directed", False),
            "Weighted": _get(e, "weighted", True),
            "ValidDiagonal": _get(e, "valid_diagonal", False),
            "NonNegative": _get(e, "non_negative", True),
        }
        prov = getattr(network, "provenance", None)
        if prov and getattr(prov, "generated_by", None):
            sidecar["Software"] = prov.generated_by
        unit = _get(e, "unit")
        if unit:
            sidecar["MeasureUnits"] = unit

        json_name = tsv_name.replace(".dense.tsv", ".json")
        (out / json_name).write_text(json.dumps(sidecar, indent=2))

    # Node indices TSV
    nodes = network.nodes or []
    if nodes:
        lines = ["matrix_index\tnode_file\tnode_index\tlabel"]
        for node in nodes:
            nid = _get(node, "id")
            lines.append(f"{nid}\tatlas-{atlas}\t{nid}\t{_get(node, 'label', '')}")
        (out / "_".join(filter(None, [stem, "nodeindices.tsv"]))).write_text("\n".join(lines))


# Backward-compatible re-exports; the converters now live in tvbo.adapters.tvb.
from tvbo.adapters.tvb import from_tvb, from_tvb_surface, from_tvb_zip  # noqa: F401, E402
