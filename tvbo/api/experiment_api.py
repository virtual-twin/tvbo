"""Experiment API — serves simulation experiments from the tvbo database.

Endpoints:
  GET /api/v1/experiments              — list available experiments
  GET /api/v1/experiments/{id}/sidecar — LinkML-valid YAML or JSON
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from linkml_runtime.dumpers import yaml_dumper, json_dumper

from tvbo.data.registry import database_dir

router = APIRouter(prefix="/api/v1/experiments", tags=["experiments"])

EXPERIMENT_DIR = database_dir("SimulationExperiment")


def _index_experiments() -> dict:
    """Build {id: (SimulationExperiment, yaml_path)} index."""
    from tvbo.classes.experiment import SimulationExperiment

    experiments = {}
    if not EXPERIMENT_DIR.exists():
        return experiments
    for yaml_path in sorted(EXPERIMENT_DIR.rglob("*.yaml")):
        try:
            exp = SimulationExperiment.from_file(str(yaml_path))
            key = yaml_path.stem
            experiments[key] = (exp, yaml_path)
        except Exception:
            continue
    return experiments


_EXPERIMENTS: Optional[dict] = None


def _get_experiments() -> dict:
    global _EXPERIMENTS
    if _EXPERIMENTS is None:
        _EXPERIMENTS = _index_experiments()
    return _EXPERIMENTS


@router.get("")
def list_experiments():
    """List available simulation experiments."""
    result = []
    for exp_id, (exp, _path) in _get_experiments().items():
        dynamics_name = None
        if exp.dynamics:
            dynamics_name = getattr(exp.dynamics, "name", None)

        result.append(
            {
                "id": exp_id,
                "label": getattr(exp, "label", exp_id),
                "description": getattr(exp, "description", None),
                "dynamics": dynamics_name,
            }
        )
    return result


@router.get("/{experiment_id}/sidecar")
def get_sidecar(experiment_id: str, format: str = Query("yaml")):
    """Download full LinkML-valid experiment definition (YAML or JSON)."""
    entry = _get_experiments().get(experiment_id)
    if not entry:
        raise HTTPException(404, f"Experiment '{experiment_id}' not found")
    exp, _path = entry

    if format == "json":
        content = json_dumper.dumps(exp, inject_type=False)
        return Response(content, media_type="application/json")
    else:
        content = yaml_dumper.dumps(exp)
        return Response(content, media_type="application/x-yaml")
