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

    study_markers = ("experiments", "simulation_experiments", "figures", "members")
    if "panels" in data and not any(k in data for k in study_markers):
        from tvbo.datamodel import schema as dm

        figure = yaml_loader.load(str(spec_path), dm.Figure)
        return [figure], "figure"

    import tvbo

    loader = tvbo.Investigation if "members" in data else tvbo.SimulationStudy
    study = loader.from_file(str(spec_path))
    return as_list(getattr(study, "figures", None)), "study"


def render_figures(figures, base_dir: Path, out_dir: Path) -> list[Path]:
    """Emit + run each figure's render script and return the written images.

    The single home for the per-figure render loop, shared by the ``figure
    render`` command and by ``tvbo run`` (which renders a study's figures after
    its experiments, so one command closes the replication loop). ``base_dir`` is
    the root each layer's ``used`` IRI resolves against (``<base_dir>/output/…``).

    The image lands directly in ``out_dir`` — the one place the report and every
    other consumer reads a figure from — while its self-contained, editable
    ``plot_<name>.py`` goes to ``out_dir/scripts/``. Both are regenerable and
    gitignored together; separating them just keeps a study with many figures from
    interleaving twice as many files in the directory people actually browse. The
    subdirectory is deliberately NOT called ``code``: in a study that name means
    the authored, tracked, importable code the recipe references by bare module
    name, which this is not.
    """
    from tvbo.adapters import bsplot

    out_dir.mkdir(parents=True, exist_ok=True)
    script_dir = out_dir / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for figure in figures:
        name = getattr(figure, "name", None) or "figure"
        fmt = (getattr(figure, "format", None) or "png").lstrip(".")
        outfile = out_dir / f"{name}.{fmt}"
        script_path = script_dir / f"plot_{sanitize_name(name)}.py"
        bsplot.render(figure, base_dir=str(base_dir), outfile=str(outfile),
                      script_path=str(script_path))
        _common.info(f"wrote {outfile}")
        _common.info(f"wrote {script_path}")
        written.append(outfile)
        # Caption partial beside the image, for `{{< include >}}` in the prose.
        try:
            cap = (bsplot.write_caption(figure, out_dir, name=name)
                   if bsplot.compose_caption(figure) else None)
            if cap:
                _common.info(f"wrote {cap}")
        except Exception as e:  # noqa: BLE001 — a caption must never lose a rendered figure
            _common.info(f"caption for {name} not written ({type(e).__name__}: {e})")
    return written


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
    name: str = typer.Option(
        None, "-n", "--name",
        help="Render only the figure(s) with this name (comma-separated for several). "
             "Default: render every figure in the spec. Iterating one panel? "
             "`tvbo figure render <Study>.yaml -n Fig3_<study>` re-renders just that figure.",
    ),
) -> None:
    """Render figures in *spec* via bsplot codegen (all of them, or the ``--name`` subset).

    Each figure's ``<name>.<format>`` image lands in ``<out-dir>`` and its
    self-contained, editable ``plot_<name>.py`` in ``<out-dir>/scripts/``, so the
    directory the report reads holds images only. ``<base-dir>`` is the
    root each layer's ``used`` IRI resolves against (``<base-dir>/output/nc/…``).
    """
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

    figures = _select(figures, name, spec_path)
    render_figures(figures, base, out_dir)


def _select(figures, name: str | None, spec_path: Path) -> list:
    """The ``--name`` subset of *figures*, or all of them."""
    if not name:
        return figures
    wanted = [n.strip() for n in name.split(",") if n.strip()]
    available = [getattr(f, "name", None) for f in figures]
    chosen = [f for f in figures if getattr(f, "name", None) in wanted]
    if not chosen:
        _common.die(
            f"{spec_path.name}: no figure named {wanted!r} "
            f"(available: {[n for n in available if n]})."
        )
    return chosen


@app.command("caption", help="Compose figure captions from the spec (no rendering) into .caption.qmd partials.")
def caption(
    spec: str = typer.Argument(
        ..., help="Path to a Figure / SimulationStudy / Investigation YAML."
    ),
    out: Path = typer.Option(
        None, "-o", "--out",
        help="Directory for the .caption.qmd partials (default: <base-dir>/figures).",
    ),
    name: str = typer.Option(
        None, "-n", "--name",
        help="Caption only the figure(s) with this name (comma-separated). Default: all.",
    ),
) -> None:
    """Compose each figure's caption from its spec and write a ``<name>.caption.qmd`` partial.

    The render-free sibling of ``figure render``: it emits only the composed captions (figure
    lead + per-panel structural descriptor + authored ``Panel.description``), so the prose can
    ``{{< include >}}`` a caption that regenerates from the spec without recomputing the figure.
    """
    from tvbo.adapters import bsplot

    spec_path = Path(spec).expanduser()
    if not spec_path.is_file():
        _common.die(f"No such spec file: {spec_path}")
    figures, kind = _load_figures(spec_path)
    if not figures:
        _common.info(f"{spec_path.name}: no figures to caption.")
        return
    figures = _select(figures, name, spec_path)
    out_dir = out.expanduser().resolve() if out else spec_path.resolve().parent / "figures"
    for figure in figures:
        fig_name = getattr(figure, "name", None) or "figure"
        if not bsplot.compose_caption(figure):
            _common.info(f"{fig_name}: nothing to caption (no description or panel labels).")
            continue
        _common.info(f"wrote {bsplot.write_caption(figure, out_dir)}")


@app.command("compare", help="Measure how a rendered figure's LAYOUT differs from a reference image.")
def compare(
    spec: str = typer.Argument(
        ..., help="Path to a Figure YAML or a SimulationStudy YAML with a figures: list."
    ),
    reference: Path = typer.Option(
        None, "-r", "--reference",
        help="Reference image, or a directory of them. Optional: by default each figure's "
             "declared `reference_image:` is used, resolved against the study root.",
    ),
    figures_dir: Path = typer.Option(
        None, "-f", "--figures",
        help="Directory holding the rendered figures (default: <base-dir>/figures).",
    ),
    out: Path = typer.Option(
        None, "-o", "--out",
        help="Directory for the side-by-side overlays and the markdown summary "
             "(default: <base-dir>/output/figure-compare).",
    ),
    base_dir: Path = typer.Option(
        None, "--base-dir", help="Study root. Defaults to the spec file's directory."
    ),
    name: str = typer.Option(
        None, "-n", "--name", help="Compare only the figure(s) with this name (comma-separated)."
    ),
) -> None:
    """Compare each rendered figure against its published counterpart, by panel geometry.

    Replication asks a figure to land on the original's layout — same aspect, same panel
    grid, panels the same relative size in the same places. This decomposes both images
    into panel boxes and reports the offsets, so "not well aligned" becomes a number per
    panel rather than an impression. Writes one overlay PNG per figure plus a markdown
    summary; the summary is what a report reads.
    """
    from tvbo.utils import figure_compare as fc

    spec_path = Path(spec).expanduser()
    if not spec_path.is_file():
        _common.die(f"No such spec file: {spec_path}")
    base = base_dir.expanduser().resolve() if base_dir else spec_path.resolve().parent
    fig_dir = figures_dir.expanduser().resolve() if figures_dir else base / "figures"
    out_dir = out.expanduser().resolve() if out else base / "output/figure-compare"
    ref = reference.expanduser().resolve() if reference else None

    figures = _select(_load_figures(spec_path)[0], name, spec_path)
    if not figures:
        _common.info(f"{spec_path.name}: no figures to compare.")
        return

    sections, summary_rows = [], []
    for figure in figures:
        fname = getattr(figure, "name", None) or "figure"
        fmt = (getattr(figure, "format", None) or "png").lstrip(".")
        ours = fig_dir / f"{fname}.{fmt}"
        theirs = ref if (ref and ref.is_file()) else reference_image_for(figure, ref or base)
        if not ours.exists():
            _common.info(f"skip {fname}: not rendered ({ours})")
            continue
        if theirs is None or not theirs.exists():
            _common.info(f"skip {fname}: no reference image (declare `reference_image:`, "
                         f"or pass --reference)")
            continue

        result = fc.compare(ours, theirs)
        image = fc.overlay(result, out_dir / f"{fname}_compare.png", titles=(fname, theirs.name))
        _common.info(f"wrote {image}")
        summary_rows.append([
            fname, f'{result["ours"]["aspect"]:.3f}', f'{result["theirs"]["aspect"]:.3f}',
            f'{result["ours"]["n_panels"]}/{result["theirs"]["n_panels"]}',
            f'{result["mean_iou"]:.3f}', f'{result["mean_offset"]:.1f}',
            f'{result["max_offset"]:.1f}',
        ])
        sections.append(f"### {fname}\n\nReference: `{theirs.name}`\n\n"
                        f"{fc.report_table(result)}\n\n"
                        f"![]({image.name})\n")

    if not sections:
        _common.info("nothing compared.")
        return

    from tvbo.utils.report import md_table

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = md_table(
        ["Figure", "Our aspect", "Ref aspect", "Panels (ours/ref)", "Mean IoU",
         "Mean offset %", "Max offset %"],
        summary_rows, aligns=["l", "r", "r", "r", "r", "r", "r"],
    )
    doc = out_dir / "figure-compare.md"
    doc.write_text(
        "# Figure A/B — layout against the published figures\n\n"
        "Panel boxes are found by recursive XY-cut on each image and matched by overlap. "
        "Offsets are in percent of page; IoU is 1.0 for a panel that coincides exactly.\n\n"
        f"{summary}\n\n" + "\n".join(sections),
        encoding="utf-8",
    )
    _common.info(f"wrote {doc}")


def reference_image_for(figure, root: Path) -> Path | None:
    """The published image *figure* reproduces, resolved against *root*.

    Prefers the figure's declared `reference_image:`; otherwise falls back to a file
    named after the figure. Returns None when neither exists.
    """
    declared = getattr(figure, "reference_image", None)
    if declared:
        candidate = Path(declared)
        return candidate if candidate.is_absolute() else root / candidate
    name = getattr(figure, "name", None) or "figure"
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = root / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    return None
