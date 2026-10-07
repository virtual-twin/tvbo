"""Network retrieval API — serves normative connectomes from the tvbo database.

Endpoints:
  GET /api/v1/networks              — list available networks (filterable)
  GET /api/v1/networks/{id}/sidecar — LinkML-valid YAML or JSON sidecar
  GET /api/v1/networks/{id}/data    — binary companion file (HDF5 or CSV)

See §12.8 of the tvbo HDF5 format proposal v0.7.
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from linkml_runtime.dumpers import json_dumper, yaml_dumper

from tvbo.data.registry import database_dir

router = APIRouter(prefix="/api/v1/networks", tags=["networks"])

# Networks are loaded from tvbo/database/networks/ YAML files
NETWORK_DIR = database_dir("Network")


def _index_networks() -> dict:
    """Build {id: Network} index from tvbo/database/networks/ directory.

    Cached at startup. Metadata only — arrays are lazy.
    """
    from tvbo.data.network_io import load_network

    networks = {}
    if not NETWORK_DIR.exists():
        return networks
    for yaml_path in sorted(NETWORK_DIR.glob("*.yaml")):
        try:
            net = load_network(yaml_path)
            networks[yaml_path.stem] = net
        except Exception:
            continue  # skip malformed sidecars
    return networks


_NETWORKS: dict | None = None


def _get_networks() -> dict:
    global _NETWORKS
    if _NETWORKS is None:
        _NETWORKS = _index_networks()
    return _NETWORKS


@router.get("")
def list_networks(
    atlas: str | None = Query(None),
    tractogram: str | None = Query(None),
):
    """List available normative connectivity networks."""
    result = []
    for net_id, net in _get_networks().items():
        # Extract atlas name
        parc = getattr(net, "parcellation", None)
        atlas_obj = getattr(parc, "atlas", None) if parc else None
        atlas_name = getattr(atlas_obj, "name", None) if atlas_obj else None

        # Extract tractogram name
        tract = getattr(net, "tractogram", None)
        tract_name = getattr(tract, "name", None) if tract else str(tract) if tract else None

        if atlas and atlas_name and atlas.lower() not in atlas_name.lower():
            continue
        if tractogram and tract_name and tractogram.lower() not in tract_name.lower():
            continue

        result.append(
            {
                "id": net_id,
                "label": getattr(net, "label", net_id),
                "atlas": atlas_name,
                "tractogram": tract_name,
                "number_of_nodes": getattr(net, "number_of_nodes", None),
                "descriptor": getattr(net, "descriptor", None),
                "bids_filename": getattr(net, "bids_filename", None),
            }
        )
    return result


@router.get("/{network_id}/sidecar")
def get_sidecar(network_id: str, format: str = Query("yaml")):
    """Download LinkML-valid sidecar (YAML or JSON)."""
    net = _get_networks().get(network_id)
    if not net:
        raise HTTPException(404, f"Network '{network_id}' not found")

    if format == "json":
        content = json_dumper.dumps(net, inject_type=False)
        return Response(content, media_type="application/json")
    else:
        content = yaml_dumper.dumps(net)
        return Response(content, media_type="application/x-yaml")


@router.get("/{network_id}/data")
def get_data(network_id: str, format: str = Query("h5")):
    """Download binary companion file (HDF5)."""
    net = _get_networks().get(network_id)
    if not net:
        raise HTTPException(404, f"Network '{network_id}' not found")

    # Serve existing HDF5 companion from tvbo/database/networks/
    data_file = getattr(net, "data_file", None)
    if data_file:
        companion = NETWORK_DIR / data_file
        if companion.exists():
            media = "application/x-hdf5"
            return FileResponse(companion, media_type=media, filename=companion.name)

    raise HTTPException(404, f"No binary data for network '{network_id}'")
