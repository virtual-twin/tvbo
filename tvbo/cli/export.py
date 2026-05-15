"""``tvbo export`` — render a SPEC into a target format (no execution)."""
from __future__ import annotations

from pathlib import Path

import typer

from tvbo import export as _export
from . import _common


def export(
    format: str = typer.Argument(..., help="Format key or alias (see `tvbo formats`)."),
    spec: str = typer.Argument(..., help="Path, CURIE, or DB name."),
    output: Path = typer.Option(
        None, "-o", "--output",
        help="Output file or directory. When omitted, writes to stdout (text formats only).",
    ),
    metadata_only: bool = typer.Option(
        True, "--metadata-only/--with-data",
        help="Bundle binary network/data alongside the metadata file.",
    ),
) -> None:
    """Render *spec* into *format* without executing it.

    Resolves *spec* (file path, CURIE, or DB name) to a TVBO object and
    writes the rendered output to *--output* or stdout. Use `tvbo formats`
    to list available format keys.
    """
    kind, obj = _common.resolve_spec(spec)

    fmt = _export.resolve(format)

    if output is None:
        if hasattr(obj, "render"):
            text = obj.render(format=fmt.key)
        else:
            text = _export.render(obj, fmt.key)
        typer.echo(text)
        return

    if hasattr(obj, "save"):
        out_path = obj.save(str(output), format=fmt.key, metadata_only=metadata_only)
        _common.info(f"wrote {out_path}")
        return

    # Fallback: render + write
    text = _export.render(obj, fmt.key)
    output = Path(output)
    if output.is_dir() or not output.suffix:
        output.mkdir(parents=True, exist_ok=True)
        output = output / f"{getattr(obj, 'name', 'tvbo')}{fmt.extension}"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    _common.info(f"wrote {output}")
