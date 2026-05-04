"""``tvbo info`` — inspect a SPEC without running it."""
from __future__ import annotations

import typer

from . import _common


def info(
    spec: str = typer.Argument(..., help="Path, CURIE, or DB name."),
    json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    kind, obj = _common.resolve_spec(spec)
    payload = _summarize(kind, obj)
    if json:
        _common.emit_json(payload)
        return
    _print_human(payload)


def _summarize(kind: str, obj) -> dict:
    out: dict = {"kind": kind, "type": type(obj).__name__}
    for attr in ("key", "name", "label", "title", "year", "doi", "description"):
        val = getattr(obj, attr, None)
        if val:
            out[attr] = val

    if kind == "study":
        exps = getattr(obj, "experiments", None) or getattr(obj, "simulation_experiments", None) or []
        try:
            exps_list = list(exps.values()) if hasattr(exps, "values") else list(exps)
        except Exception:
            exps_list = []
        out["experiments"] = [
            {"key": getattr(e, "key", None) or getattr(e, "id", None) or getattr(e, "label", None)}
            for e in exps_list
        ]

    if kind == "experiment":
        out["has_integration"] = getattr(obj, "integration", None) is not None
        out["has_network"] = getattr(obj, "network", None) is not None
        explorations = getattr(obj, "explorations", None)
        if explorations:
            try:
                names = list(explorations.keys()) if hasattr(explorations, "keys") else [
                    getattr(e, "name", None) or getattr(e, "key", None) for e in explorations
                ]
            except Exception:
                names = []
            out["explorations"] = names

    if kind == "dynamics":
        params = getattr(obj, "parameters", None)
        svs = getattr(obj, "state_variables", None)
        out["n_parameters"] = len(params) if params is not None else 0
        out["n_state_variables"] = len(svs) if svs is not None else 0

    return out


def _print_human(p: dict) -> None:
    typer.echo(f"{p['type']} ({p['kind']})")
    for k, v in p.items():
        if k in ("kind", "type"):
            continue
        if isinstance(v, list):
            typer.echo(f"  {k}: [{len(v)}]")
            for item in v[:10]:
                typer.echo(f"    - {item}")
        else:
            typer.echo(f"  {k}: {v}")
