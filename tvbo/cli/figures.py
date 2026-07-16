"""``tvbo figure`` — render declarative figures from a Figure or Study YAML.

A :class:`~tvbo.datamodel.Figure` is TVBO's backend-independent spec of a
publication figure: a mosaic of panels whose layers bind experiment result
containers to visual channels (see ``schema/figure.yaml``). This verb resolves
such a spec and drives :mod:`tvbo.adapters.bsplot` to emit a self-contained
``plot.py`` and run it, mirroring how ``tvbo run`` drives the simulation
adapters.

The spec may be a standalone ``Figure`` (top-level ``panels:``) or a
``SimulationStudy`` carrying a ``figures:`` list — the latter closes the
replication loop (a study is its experiments plus the figures that read them).
"""
from __future__ import annotations

from pathlib import Path

import typer

from . import _common
from tvbo.utils import as_list, sanitize_name

app = typer.Typer(name="figure", no_args_is_help=True)


def _load_figures(spec_path: Path) -> tuple[list, str]:
    """Resolve *spec_path* to ``(figures, kind)``.

    A spec with a top-level ``panels:`` (and no study markers) is a standalone
    ``Figure``; otherwise it is loaded as a ``SimulationStudy`` and its
    ``figures`` slot is returned. *kind* is ``"figure"`` or ``"study"`` so the
    caller can phrase the empty case correctly.
    """
    from tvbo.utils import yaml_loader

    data = yaml_loader.load_as_dict(str(spec_path))
    if not isinstance(data, dict):
        _common.die(f"{spec_path} is not a Figure or SimulationStudy spec (got {type(data).__name__}).")

    study_markers = ("experiments", "simulation_experiments", "figures")
    if "panels" in data and not any(k in data for k in study_markers):
        from tvbo.datamodel import schema as dm

        figure = yaml_loader.load(str(spec_path), dm.Figure)
        return [figure], "figure"

    import tvbo

    study = tvbo.SimulationStudy.from_file(str(spec_path))
    return as_list(getattr(study, "figures", None)), "study"


@app.command("render", help="Render declarative figures from a Figure or SimulationStudy YAML.")
def render(
    spec: str = typer.Argument(
        ..., help="Path to a Figure YAML (top-level panels:) or a SimulationStudy YAML with a figures: list."
    ),
    out: Path = typer.Option(
        None, "-o", "--out",
        help="Directory for the rendered figures + plot scripts (default: <base-dir>/figures).",
    ),
    base_dir: Path = typer.Option(
        None, "--base-dir",
        help="Root the experiment result containers live under (where output/nc/ sits). "
             "Defaults to the spec file's directory.",
    ),
) -> None:
    """Render every figure in *spec* via bsplot codegen.

    For each figure a self-contained ``plot_<name>.py`` is emitted next to the
    output image and executed, so the ``<name>.<format>`` file is produced and
    the script stays as an editable, rerunnable artefact. ``<base-dir>`` is the
    root each layer's ``used`` IRI resolves against (``<base-dir>/output/nc/…``).
    """
    from tvbo.adapters import bsplot

    spec_path = Path(spec).expanduser()
    if not spec_path.is_file():
        _common.die(f"No such spec file: {spec_path}")

    base = base_dir.expanduser().resolve() if base_dir else spec_path.resolve().parent
    out_dir = out.expanduser().resolve() if out else base / "figures"

    figures, kind = _load_figures(spec_path)
    if not figures:
        detail = ("study has an empty `figures:` list" if kind == "study"
                  else "spec has no panels")
        _common.info(f"{spec_path.name}: no figures to render ({detail}).")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for figure in figures:
        name = getattr(figure, "name", None) or "figure"
        fmt = (getattr(figure, "format", None) or "png").lstrip(".")
        outfile = out_dir / f"{name}.{fmt}"
        script_path = out_dir / f"plot_{sanitize_name(name)}.py"
        bsplot.render(figure, base_dir=str(base), outfile=str(outfile),
                      script_path=str(script_path))
        _common.info(f"wrote {outfile}")
        _common.info(f"wrote {script_path}")
