"""Dynamics model API — serves neural mass models from the tvbo database.

Endpoints:
  GET /api/v1/dynamics              — list available models (filterable)
  GET /api/v1/dynamics/{id}/sidecar — LinkML-valid YAML or JSON
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from linkml_runtime.dumpers import yaml_dumper, json_dumper

from tvbo.data.registry import database_dir

router = APIRouter(prefix="/api/v1/dynamics", tags=["dynamics"])

MODEL_DIR = database_dir("Dynamics")


def _index_models() -> dict:
    """Build {name: (Dynamics, yaml_path)} index from tvbo/database/models/."""
    from tvbo.classes.dynamics import Dynamics

    models = {}
    if not MODEL_DIR.exists():
        return models
    for yaml_path in sorted(MODEL_DIR.rglob("*.yaml")):
        try:
            d = Dynamics.from_file(str(yaml_path))
            models[d.name] = (d, yaml_path)
        except Exception:
            continue
    return models


_MODELS: Optional[dict] = None


def _get_models() -> dict:
    global _MODELS
    if _MODELS is None:
        _MODELS = _index_models()
    return _MODELS


@router.get("")
def list_dynamics(
    system_type: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
):
    """List available dynamics models."""
    result = []
    for model_name, (d, _path) in _get_models().items():
        st = getattr(d, "system_type", None)
        if system_type and st and system_type.lower() not in str(st).lower():
            continue
        if name and name.lower() not in model_name.lower():
            continue

        n_sv = len(d.state_variables) if d.state_variables else 0
        n_params = len(d.parameters) if d.parameters else 0

        result.append(
            {
                "id": model_name,
                "name": model_name,
                "label": getattr(d, "label", model_name),
                "system_type": str(st) if st else None,
                "autonomous": getattr(d, "autonomous", None),
                "number_of_modes": getattr(d, "number_of_modes", 1),
                "n_state_variables": n_sv,
                "n_parameters": n_params,
                "description": getattr(d, "description", None),
            }
        )
    return result


@router.get("/{model_id}/sidecar")
def get_sidecar(model_id: str, format: str = Query("yaml")):
    """Download full LinkML-valid model definition (YAML or JSON)."""
    entry = _get_models().get(model_id)
    if not entry:
        raise HTTPException(404, f"Dynamics model '{model_id}' not found")
    d, _path = entry

    if format == "json":
        content = json_dumper.dumps(d, inject_type=False)
        return Response(content, media_type="application/json")
    else:
        content = yaml_dumper.dumps(d)
        return Response(content, media_type="application/x-yaml")
