"""Inspect and reclaim the produced-constant store.

A parameter with a ``producer:`` is materialised once to a content-addressed artifact under ``~/.tvbo/constants``, keyed on the producing call AND on that module's source. Editing the callable is therefore supposed to write a NEW artifact — which is what keeps a run from reading arrays computed by code that no longer exists — and the old one is left behind deliberately, because nothing at write time knows whether another study still reaches it.

That is what this reclaims: given a study, the artifacts of ITS producers that IT no longer reaches. Producers the study does not declare are never touched.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

app = typer.Typer(help="Inspect and reclaim tvbo's caches.", no_args_is_help=True)


def _mb(n: int) -> str:
    return f"{n / 1e6:,.1f} MB"


@app.command("prune")
def prune(
    specs: List[Path] = typer.Argument(..., help="Study / experiment YAML to read."),
    delete: bool = typer.Option(False, "--delete", help="Actually remove them (default: list only)."),
    cache_dir: Optional[Path] = typer.Option(None, help="Store to prune (default ~/.tvbo/constants)."),
) -> None:
    """List — or with --delete remove — produced constants these studies have superseded."""
    from tvbo.classes.study import SimulationStudy
    from tvbo.data import param_io

    dead, live = [], set()
    for spec in specs:
        study = SimulationStudy.from_file(spec)
        keep, _ = param_io.live_artifacts(study, cache_dir)
        live |= keep
        dead += param_io.superseded_artifacts(study, cache_dir)

    dead = [p for p in dict.fromkeys(dead) if p not in live]  # live for ANY spec wins
    if not dead:
        typer.echo(f"nothing superseded ({len(live)} artifact(s) still reached)")
        return

    total = sum(p.stat().st_size for p in dead)
    for p in dead:
        typer.echo(f"  {'removed' if delete else 'superseded'}  {_mb(p.stat().st_size):>10s}  {p.name}")
        if delete:
            p.unlink()
    verb = "reclaimed" if delete else "reclaimable"
    typer.echo(f"{len(dead)} artifact(s), {_mb(total)} {verb}" + ("" if delete else "  — re-run with --delete"))
