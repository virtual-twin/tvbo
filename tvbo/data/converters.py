"""Format converters: BEP017 export, TVB ZIP import.

Uses relmat_entities() + build_path() from §6.5 for BIDS-compliant
filenames — no manual filename construction.

See §12.3 of the tvbo HDF5 format proposal v0.7.
"""
import json
import numpy as np
from pathlib import Path
from bids.layout.writing import build_path

from tvbo.data.matrix_io import auto_format
from tvbo.data.network_io import RELMAT_PATTERNS, _template_edges


def relmat_entities(network) -> dict:
    """Extract pybids entities from a tvbo Network instance or dict (§6.5).

    Accepts both a Network object (direct attribute access) and a raw
    sidecar dict (for standalone usage). Returns entity dict for
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
        atlas = (network.get("parcellation", {})
                 .get("atlas", {}).get("name", ""))
        descriptor = network.get("descriptor")
    else:
        bids = getattr(network, "bids", None) or {}
        if not isinstance(bids, dict):
            bids = {
                k: getattr(bids, k, None)
                for k in ("template", "subject", "session", "cohort",
                           "reconstruction", "space", "segmentation", "scale")
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


def to_bep017(network, output_dir):
    """Export a tvbo Network to BEP017-compatible per-measure files.

    Each template edge becomes one ``meas-<name>_relmat.dense.tsv`` +
    JSON sidecar. Filenames use pybids ``build_path`` (§6.5). Reads
    data directly from the Network object — no YAML round-trip needed.

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
        _get = ((lambda k, d: e.get(k, d)) if isinstance(e, dict)
                else (lambda k, d: getattr(e, k, d)))
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
            nlabel = (node.label if hasattr(node, "label")
                      else node.get("label", ""))
            lines.append(f"{nid}\tatlas-{atlas}\t{nid}\t{nlabel}")
        (out / f"atlas-{atlas}_nodeindices.tsv").write_text("\n".join(lines))


def from_tvb_zip(zip_path):
    """Import a TVB connectivity ZIP into a tvbo Network.

    TVB ZIPs contain: ``weights.txt``, ``tract_lengths.txt``,
    ``centres.txt``. Returns a Network with arrays loaded — call
    ``net.save()`` to persist.

    Parameters
    ----------
    zip_path : str or Path
        Path to TVB connectivity ZIP file.

    Returns
    -------
    Network
        Network instance with loaded arrays ready for ``save()``.
    """
    import zipfile
    import io
    from tvbo.classes.network import Network

    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        weights = np.loadtxt(io.BytesIO(zf.read("weights.txt")))
        lengths = np.loadtxt(io.BytesIO(zf.read("tract_lengths.txt")))
        centres_raw = zf.read("centres.txt").decode()

    labels, coords = [], []
    for line in centres_raw.strip().split("\n"):
        parts = line.split()
        labels.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])

    n = weights.shape[0]
    from tvbo.datamodel import schema as tvbo_datamodel
    nodes = [
        tvbo_datamodel.Node(
            id=i, label=labels[i],
            position=tvbo_datamodel.Coordinate(
                x=float(coords[i][0]),
                y=float(coords[i][1]),
                z=float(coords[i][2]),
            ),
        )
        for i in range(n)
    ]
    net = Network(
        nodes=nodes,
        edges=[],
        number_of_nodes=n,
    )
    net.set_matrix("weight", weights)
    net.set_matrix("length", lengths)
    net.label = zip_path.stem
    net.descriptor = "SC"  # TVB connectivity = structural
    return net


def from_tvb(connectivity):
    """Import a live TVB Connectivity object into a tvbo Network.

    Lossless conversion — all TVB fields are preserved as tvbo Node
    parameters and Network-level metadata.

    Parameters
    ----------
    connectivity : tvb.datatypes.connectivity.Connectivity
        A configured TVB Connectivity instance.

    Returns
    -------
    Network
        Network instance with arrays loaded, ready for ``save()``.
    """
    from tvbo.classes.network import Network
    from tvbo.datamodel import schema as tvbo_datamodel

    conn = connectivity
    weights = np.asarray(conn.weights, dtype="float64")
    lengths = np.asarray(conn.tract_lengths, dtype="float64")
    centres = np.asarray(conn.centres, dtype="float64")
    labels = list(conn.region_labels)
    n = weights.shape[0]

    # Build nodes with position + all per-node TVB metadata
    nodes = []
    for i in range(n):
        node = tvbo_datamodel.Node(
            id=i,
            label=str(labels[i]),
            position=tvbo_datamodel.Coordinate(
                x=float(centres[i, 0]),
                y=float(centres[i, 1]),
                z=float(centres[i, 2]),
            ),
        )
        # Preserve TVB per-node arrays as parameters
        if conn.cortical is not None and len(conn.cortical) == n:
            node.parameters = node.parameters or {}
            node.parameters["cortical"] = tvbo_datamodel.Parameter(
                name="cortical", value=float(conn.cortical[i]),
            )
        if conn.areas is not None and len(conn.areas) == n:
            node.parameters = node.parameters or {}
            node.parameters["area"] = tvbo_datamodel.Parameter(
                name="area", value=float(conn.areas[i]),
            )
        if (conn.hemispheres is not None and len(conn.hemispheres) == n):
            node.parameters = node.parameters or {}
            node.parameters["hemisphere"] = tvbo_datamodel.Parameter(
                name="hemisphere", value=float(conn.hemispheres[i]),
            )
        nodes.append(node)

    net = Network(nodes=nodes, edges=[], number_of_nodes=n)
    net.set_matrix("weight", weights)
    net.set_matrix("length", lengths)
    net.label = getattr(conn, "title", None) or "TVB Connectivity"
    net.descriptor = "SC"

    # Store orientations for lossless reconstruction
    if conn.orientations is not None and len(conn.orientations) == n:
        object.__setattr__(net, '_orientations', np.asarray(conn.orientations))

    # Conduction speed
    speed = np.asarray(conn.speed).ravel()
    cs_val = float(speed[0]) if len(speed) > 0 else 3.0
    net.parameters["conduction_speed"] = tvbo_datamodel.Parameter(
        name="conduction_speed", label="v", value=cs_val, unit="mm/ms",
    )

    return net


def from_tvb_surface(connectivity, surface, region_mapping):
    """Create a multi-level tvbo Network from TVB surface simulation data.

    Produces two linked networks:
    1. **Region-level** (parent): from TVB Connectivity
    2. **Vertex-level** (child): mesh + region_mapping linking vertices
       to regions via the hierarchical ``node_mapping`` pattern

    Parameters
    ----------
    connectivity : tvb.datatypes.connectivity.Connectivity
        Configured TVB Connectivity (region-level).
    surface : tvb.datatypes.surfaces.Surface
        TVB CorticalSurface (or any Surface subclass).
    region_mapping : tvb.datatypes.region_mapping.RegionMapping
        TVB RegionMapping mapping vertices to regions.

    Returns
    -------
    tuple[Network, Network]
        ``(region_network, surface_network)`` — the surface network
        references the region network as its parent via ``parent_network``
        and stores the region mapping in ``node_mapping``.
    """
    from tvbo.classes.network import Network
    from tvbo.datamodel import schema as tvbo_datamodel

    # 1. Region-level network from Connectivity
    region_net = from_tvb(connectivity)

    # 2. Surface-level network
    vertices = np.asarray(surface.vertices, dtype="float32")
    triangles = np.asarray(surface.triangles, dtype="int32")
    normals = np.asarray(surface.vertex_normals, dtype="float32")
    mapping = np.asarray(region_mapping.array_data, dtype="int32")
    n_vertices = vertices.shape[0]
    n_elements = triangles.shape[0]

    # Create surface network (nodes = vertices)
    surface_nodes = [
        tvbo_datamodel.Node(
            id=i, label=f"vertex_{i}",
            position=tvbo_datamodel.Coordinate(
                x=float(vertices[i, 0]),
                y=float(vertices[i, 1]),
                z=float(vertices[i, 2]),
            ),
        )
        for i in range(n_vertices)
    ]

    surface_net = Network(
        nodes=surface_nodes,
        edges=[],
        number_of_nodes=n_vertices,
    )
    surface_net.label = f"Surface ({n_vertices} vertices, {n_elements} triangles)"
    surface_net.descriptor = "surface"

    # Store mesh data for HDF5 companion
    mesh = tvbo_datamodel.Mesh(
        label=getattr(surface, "surface_type", "CorticalSurface"),
        element_type="triangle",
        number_of_vertices=n_vertices,
        number_of_elements=n_elements,
    )
    surface_net.mesh = mesh
    object.__setattr__(surface_net, '_mesh_vertices', vertices)
    object.__setattr__(surface_net, '_mesh_elements', triangles)
    object.__setattr__(surface_net, '_mesh_normals', normals)

    # Set hierarchical node mapping (vertex → region)
    surface_net.set_node_mapping(
        mapping,
        parent_network=region_net,
        dataset_path="/mesh/region_mapping",
    )

    return region_net, surface_net
