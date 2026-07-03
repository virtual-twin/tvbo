"""Experiment API — serves simulation experiments from the tvbo database.

Endpoints:
  GET /api/v1/experiments              — list available experiments
  GET /api/v1/experiments/formats      — list supported export formats
  GET /api/v1/experiments/{id}/sidecar — LinkML-valid YAML or JSON
  GET /api/v1/experiments/{id}/render  — render experiment in any supported format
  POST /api/v1/experiments/render      — render or save an experiment from a payload
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

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


# ---------------------------------------------------------------------------
# Export format discovery & render endpoints
# ---------------------------------------------------------------------------

class RenderExperimentRequest(BaseModel):
    """Request body for rendering or saving an experiment from an inline payload.

    Describes a [`SimulationExperiment`](../classes/experiment.qmd) to render (or
    persist) via the `POST /api/v1/experiments/render` endpoint. The experiment is
    supplied inline as a mapping rather than referenced by stored id.

    Args:
        experiment: Mapping of `SimulationExperiment` constructor keyword arguments
            used to build the experiment to render.
        format: Export format key or alias to render into (e.g. `"yaml"`).
        filename: Suggested download filename; falls back to a default derived from
            the format's extension when omitted.
        save_path: Destination path to persist the rendered experiment to. When set,
            the experiment is saved there instead of being returned as a download.
        metadata_only: Whether to render only the experiment metadata rather than the
            fully expanded definition when saving.
        render_kwargs: Extra keyword arguments forwarded to the render or save call.
    """

    experiment: dict
    format: str = "yaml"
    filename: Optional[str] = None
    save_path: Optional[str] = None
    metadata_only: bool = True
    render_kwargs: dict = Field(default_factory=dict)


def _resolve_format(fmt: str):
    from tvbo import export as _export
    try:
        return _export.resolve(fmt)
    except ValueError as e:
        raise HTTPException(400, str(e))


def _render_response(content: str, fmt, filename: Optional[str]) -> Response:
    fname = filename or f"experiment{fmt.extension}"
    headers = {"Content-Disposition": f'attachment; filename="{fname}"'}
    return Response(content, media_type=fmt.media_type, headers=headers)


@router.get("/formats")
def get_export_formats():
    """Return supported export formats for UI dropdowns."""
    from tvbo import export as _export
    return _export.list_format_dicts()


@router.get("/{experiment_id}/render")
def render_experiment(
    experiment_id: str,
    format: str = Query("yaml"),
    filename: Optional[str] = Query(None),
    metadata_only: bool = Query(True),
):
    """Render a stored experiment in the requested format."""
    entry = _get_experiments().get(experiment_id)
    if not entry:
        raise HTTPException(404, f"Experiment '{experiment_id}' not found")
    exp, yaml_path = entry

    fmt = _resolve_format(format)

    # YAML: return source file directly to avoid numpy serialization errors
    if fmt.key == "yaml":
        with open(yaml_path, encoding="utf-8") as f:
            content = f.read()
    else:
        content = exp.render(format=fmt.key)

    return _render_response(content, fmt, filename)


@router.post("/render")
def render_experiment_payload(request: RenderExperimentRequest):
    """Render or save an experiment described by an inline payload."""
    from tvbo.classes.experiment import SimulationExperiment

    fmt = _resolve_format(request.format)
    exp = SimulationExperiment(**request.experiment)

    if request.save_path:
        saved = exp.save(
            request.save_path,
            format=fmt.key,
            metadata_only=request.metadata_only,
            **request.render_kwargs,
        )
        return {"status": "saved", "path": saved}

    content = exp.render(format=fmt.key, **request.render_kwargs)
    return _render_response(content, fmt, request.filename)

