"""``tvbo formats`` — list every registered I/O format."""
from __future__ import annotations

import typer

from tvbo import export as _export


def formats(
    json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    rows = _export.list_format_dicts()
    if json:
        from . import _common
        _common.emit_json(rows)
        return

    # Human table
    headers = ["key", "extension", "media_type", "with_data", "importer", "aliases"]
    widths = {h: len(h) for h in headers}
    table = []
    for r in rows:
        row = {
            "key": r.get("format", ""),
            "extension": r.get("extension", "") or "",
            "media_type": r.get("media_type", "") or "",
            "with_data": "yes" if r.get("supports_with_data") else "",
            "importer": "yes" if _export.resolve(r["format"]).importer else "",
            "aliases": ",".join(r.get("aliases", []) or []),
        }
        for h in headers:
            widths[h] = max(widths[h], len(row[h]))
        table.append(row)

    fmt = "  ".join(f"{{{h}:<{widths[h]}}}" for h in headers)
    typer.echo(fmt.format(**{h: h for h in headers}))
    typer.echo(fmt.format(**{h: "-" * widths[h] for h in headers}))
    for row in sorted(table, key=lambda r: r["key"]):
        typer.echo(fmt.format(**row))
