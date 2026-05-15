"""``tvbo config`` — view CLI configuration (full impl in C2)."""
from __future__ import annotations

import os
from pathlib import Path

import typer


app = typer.Typer(name="config", no_args_is_help=True)


def _config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "tvbo" / "config.toml"


@app.command("path", help="Print the resolved path to the user config file.")
def path() -> None:
    """Print the resolved path to the user config file (`$XDG_CONFIG_HOME/tvbo/config.toml`)."""
    typer.echo(_config_path())


@app.command("show", help="Print the contents of the user config file.")
def show() -> None:
    """Print the contents of the user config file, or a `# (no config)` notice if absent."""
    p = _config_path()
    if not p.exists():
        typer.echo(f"# (no config at {p})", err=True)
        raise typer.Exit(0)
    typer.echo(p.read_text(encoding="utf-8"))
