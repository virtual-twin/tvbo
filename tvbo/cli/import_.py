"""``tvbo import`` — load a foreign file via the registry's importer dispatch."""
from __future__ import annotations

from pathlib import Path

import typer

from tvbo import export as _export
from . import _common


def import_(
    path: Path = typer.Argument(..., exists=True, readable=True, help="Input file."),
    format: str = typer.Option(
        None, "--format", help="Force a format key (otherwise inferred from extension)."
    ),
    output: Path = typer.Option(
        None, "-o", "--output",
        help="If given, save the loaded object as YAML to this path.",
    ),
) -> None:
    if format is None:
        fmt = _export.resolve_by_extension(path.suffix)
    else:
        fmt = _export.resolve(format)

    if fmt.importer is None:
        _common.die(f"Format {fmt.key!r} has no importer.")

    obj = _export.load(fmt.key, path)
    _common.info(f"loaded {type(obj).__name__} from {path} via {fmt.key!r}")

    if output is not None and hasattr(obj, "save"):
        out_path = obj.save(str(output), format="yaml")
        _common.info(f"wrote {out_path}")
