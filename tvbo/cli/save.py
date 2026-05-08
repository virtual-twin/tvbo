"""``tvbo save`` — alias for ``tvbo export --with-data``."""
from __future__ import annotations

from pathlib import Path

import typer

from . import export as _export_cmd


def save(
    format: str = typer.Argument(..., help="Format key (see `tvbo formats`)."),
    spec: str = typer.Argument(..., help="Path, CURIE, or DB name."),
    output: Path = typer.Option(..., "-o", "--output", help="Output file or directory."),
) -> None:
    _export_cmd.export(format=format, spec=spec, output=output, metadata_only=False)
