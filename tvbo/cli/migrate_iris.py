"""``tvbo migrate-iris`` — rewrite the curated-record IRIs specs name into their ``tvbo:<folder>/<file stem>`` form."""

from __future__ import annotations

from pathlib import Path

import typer

from . import _common


def migrate_iris(
    paths: list[Path] = typer.Argument(
        ..., exists=True, help="YAML, Markdown or Quarto files, or directories to search for them."
    ),
    check: bool = typer.Option(
        False, "--check", help="Report what would change and exit 1 if anything would, writing nothing."
    ),
) -> None:
    """Rewrite every singular-scope (``tvbo:observation/bold_tvb``) and unscoped (``tvbo:Linear``) curated-record IRI under *paths*.

    Each rewrite is printed as ``path:line: old -> new``, and each IRI that names a curated record but could not be placed as ``path:line: old: reason``, for a person to settle by hand. YAML blocks a page fences that do not parse are reported and left alone.
    """
    from tvbo.data.iri_migration import migrate_text, spec_files

    rewrites = 0
    for path in spec_files(paths):
        text = path.read_bytes().decode("utf-8")  # bytes, so a CRLF file keeps its line endings
        migrated, changes, skipped = migrate_text(text, path)
        for change in changes:
            typer.echo(
                f"{path}:{change.line}: {change.old} -> {change.new}"
                if change.new
                else f"{path}:{change.line}: {change.old}: {change.reason}"
            )
        for reason in skipped:
            _common.warn(f"{path}: not YAML, left alone ({reason})")
        if migrated != text:
            rewrites += 1
            if not check:
                path.write_bytes(migrated.encode("utf-8"))
    verb = "would rewrite" if check else "rewrote"
    _common.info(f"{verb} {rewrites} file(s)")
    if check and rewrites:
        raise typer.Exit(1)
