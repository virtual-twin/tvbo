"""``tvbo brain`` — the ASCII spec-to-cortex portrait, and the bare-``tvbo`` splash."""

from __future__ import annotations

import shutil
from pathlib import Path

import typer

from . import _common, _portrait


def brain(
    spec: str = typer.Argument(None, help="Path, CURIE, or DB name whose text draws the cortex."),
    animate: bool = typer.Option(False, "--animate", "-a", help="Play the dissolve instead of the still frame."),
    background: str = typer.Option("dark", "--bg", help="Background the output is viewed on (dark|light)."),
    width: int = typer.Option(None, "--width", "-w", help="Portrait width in characters (default: the terminal)."),
    plain: bool = typer.Option(False, "--plain", help="Plain characters, no ANSI colour."),
    no_flow: bool = typer.Option(False, "--no-flow", help="Drop the letters caught mid-flight."),
    hemi: str = typer.Option("lh", "--hemi", help="Hemisphere to render live (lh|rh|both)."),
    view: str = typer.Option("lateral", "--view", help="View to render live (lateral, medial, top, ...)."),
    render: bool = typer.Option(False, "--render", help="Re-render the surface with bsplot instead of the shipped one."),
    save_asset: Path = typer.Option(None, "--save-asset", help="Write the rendered surface as a shippable asset."),
    save_logo: Path = typer.Option(
        None, "--save-logo", help="Re-render the TVB-O logo from the project artwork into an asset."
    ),
) -> None:
    """Draw a model spec dissolving into a cortical surface built from its own characters.

    The surface ships precomputed, so the still frame is instant. ``--render`` recomputes the geometry with bsplot for another hemisphere or view, and ``--save-asset`` stores that render for reuse.
    """
    if background not in _portrait.THEMES:
        raise typer.BadParameter(f"--bg must be one of {', '.join(_portrait.THEMES)}")
    if save_logo is not None:
        _common.logger.info(f"wrote {_portrait.build_logo_asset(out=save_logo) and save_logo}")
        return
    text = _spec_text(spec)
    cortex_path = _portrait.ASSET

    if render or save_asset is not None:
        target = save_asset or (Path(typer.get_app_dir("tvbo")) / "cortex_ascii.txt")
        _common.logger.info(f"rendering {hemi} {view} with bsplot …")
        _portrait.build_asset(hemi=hemi, view=view, out=target)
        _common.logger.info(f"wrote {target}")
        cortex_path = target

    opts = dict(
        theme=background,
        color_mode="none" if plain else None,
        width=width,
        cortex_path=cortex_path,
    )
    if animate:
        _portrait.play(spec_text=text, **opts)
        return
    typer.echo(_portrait.render(text, flow=not no_flow, **opts))


def splash(ctx: typer.Context) -> None:
    """What a bare ``tvbo`` prints: the wordmark beside a small cortex, then the commands."""
    import tvbo

    cols = shutil.get_terminal_size((96, 30)).columns - 4
    banner = _portrait.hero(subtitle=f"tvbo {tvbo.__version__}", width=cols)
    typer.echo()
    typer.echo("\n".join(f"  {line}" for line in banner.split("\n")))
    typer.echo(ctx.get_help())


def _first_field(text: str, field: str) -> str | None:
    for line in text.splitlines():
        if line.startswith(f"{field}:"):
            return line.split(":", 1)[1].strip().strip('"')
    return None


def _spec_text(spec: str | None) -> str:
    """The raw text of *spec* — a file, a CURIE, or a database name."""
    if not spec:
        return Path(_portrait.DEFAULT_SPEC).read_text(encoding="utf-8")
    path = Path(spec).expanduser()
    if path.exists():
        return path.read_text(encoding="utf-8")

    from tvbo.data import registry

    prefix, _, name = spec.rpartition(":")
    classes = (
        [_common._CURIE_TO_CLASS[prefix]] if prefix in _common._CURIE_TO_CLASS else list(_common._CURIE_TO_CLASS.values())
    )
    for cls in dict.fromkeys(classes):
        try:
            return Path(registry.resolve(cls, name)).read_text(encoding="utf-8")
        except Exception:
            continue
    raise typer.BadParameter(f"No spec text found for {spec!r} (pass a path, a CURIE, or a database name)")
