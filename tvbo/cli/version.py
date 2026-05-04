"""``tvbo version`` — print the package version."""
from __future__ import annotations

import typer

import tvbo


def version() -> None:
    typer.echo(tvbo.__version__)
