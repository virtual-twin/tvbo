"""``tvbo version`` — print the package version."""

from __future__ import annotations

import typer

import tvbo


def version() -> None:
    """Print the installed TVBO package version."""
    typer.echo(tvbo.__version__)
