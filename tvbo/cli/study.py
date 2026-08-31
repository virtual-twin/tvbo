"""``tvbo study`` — scaffold and inspect a study dataset.

Two subcommands, both reading the one layout record (``schema/study_layout.yaml``, see :mod:`tvbo.utils.study_layout`):

* ``tvbo study init <Name>`` creates a BIDS study dataset: the directories, the two ignore files derived from the record, ``dataset_description.json`` for the study and its derivative, and a seed for every file the record gives a template. ``-t <variant>`` adds the entries that variant declares, and swaps in any seed it names in place of the general one, so a replication gets a report about reproducing a paper and every other study does not.
* ``tvbo study layout`` prints the tree or either ignore file, and ``--sync`` rewrites the layout region of a document in place so no document retypes the tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

import tvbo
from tvbo.utils import study_layout as layout_rules

app = typer.Typer(name="study", no_args_is_help=True)

TEMPLATE_DIR = Path(tvbo.__file__).resolve().parent / "templates" / "study"
"""Seeds for the files the record gives a ``template``."""

_KEEP = ".gitkeep"

SLIM_ROLES = ("recipe", "gitignore", "dataset_description")
"""What ``--slim`` writes, at the study root only: the one specification a run is given, the rules that keep the run's products out of version control, and the file that makes the directory identifiable as a study.

That last one is not documentation and cannot be dropped: :func:`~tvbo.utils.study_layout.study_root` finds a study by walking up for it, so without it nothing inside can resolve a layout role. What ``--slim`` does leave out is a study of record's documentation — the README, the sourcedata provenance note, the citation file, the report — which a demonstration has no use for. ``tvbo validate study`` reports those as missing, correctly: a slim study is a working recipe, not a complete BIDS dataset."""


def _dataset_description(name: str, dataset_type: str, bids_version: str, sources: list[dict] | None = None) -> dict:
    """The ``dataset_description.json`` body for a dataset of ``dataset_type``."""
    body: dict = {"Name": name, "BIDSVersion": bids_version, "DatasetType": dataset_type}
    if dataset_type == "derivative":
        body["GeneratedBy"] = [{"Name": "tvbo", "Version": tvbo.__version__}]
    if sources:
        body["SourceDatasets"] = sources
    return body


def _seed(dest: Path, template: str, study: str) -> None:
    """Write ``dest`` from its template, substituting the study's own name.

    Three spellings, because the templates predate the scaffolder and use the angle-bracket placeholders an author filled in by hand: ``{study}`` and ``<Study>`` take the name as given, ``<study>`` its lowercase form (the one that appears inside identifiers and filenames).
    """
    source = TEMPLATE_DIR / template
    if not source.is_file():
        raise typer.BadParameter(f"The layout names template {template!r} for {dest.name}, but {source} does not exist.")
    text = source.read_text(encoding="utf-8")
    for placeholder, value in (("{study}", study), ("<Study>", study), ("<study>", study.lower())):
        text = text.replace(placeholder, value)
    dest.write_text(text, encoding="utf-8")


@app.command("init")
def init(
    name: str = typer.Argument(..., help="Study name; also the dataset name and the entry recipe's stem."),
    parent: Path = typer.Option(Path(), "--in", "-C", help="Directory to create the study in."),
    template: list[str] = typer.Option(
        [],
        "--template",
        "-t",
        help="Layout variant to include, e.g. `replication`. Repeatable; omit for the general layout.",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite files that already exist."),
    slim: bool = typer.Option(
        False,
        "--slim/--full",
        help="`--slim` writes the entry recipe, a `.gitignore` and `dataset_description.json`, "
        "and stops. The result is a runnable study, NOT a complete BIDS dataset: `tvbo validate "
        "study` will report the documentation a study of record needs and a demonstration does "
        "not. For a docs page, a notebook, or anything driven from the Python API. `--full` "
        "(default) scaffolds the whole BIDS study dataset.",
    ),
) -> None:
    """Scaffold a BIDS study dataset from the layout record.

    Every directory, both ignore files and every ``dataset_description.json`` are derived from the record, so a study's shape is never typed out a second time. An empty directory gets a ``.gitkeep`` only when it is tracked; an untracked one is left for the run to create.

    ``--template`` selects a layout variant. A variant both adds entries of its own and may supersede the seed of one every study has, since the same file needs different starting text once the variant says what kind of study this is: without one the study gets a report of its own results, and with ``-t replication`` it gets the report, the scorecard and the copyright-safe figure split a replication needs.

    ``--slim`` writes only what a human authors (:data:`SLIM_ROLES`), for a study that demonstrates something rather than being archived.
    """
    record = layout_rules.load_layout()
    templates = tuple(template)
    root = (parent / name).resolve()
    root.mkdir(parents=True, exist_ok=True)

    def write(rel: str, text: str) -> None:
        target = root / rel
        if target.exists() and not force:
            typer.echo(f"  skip     {rel} (exists; --force to overwrite)")
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        typer.echo(f"  write    {rel}")

    typer.echo(f"{root}")
    if not slim:
        for rel, _ in layout_rules.walk(record, templates):
            (root / rel).mkdir(parents=True, exist_ok=True)
            typer.echo(f"  mkdir    {rel}/")

    bids_version = str(record.bids_version)
    for rel, entry in layout_rules.iter_files(record, templates):
        rel = layout_rules.interpolate(rel, name)
        role = str(entry.role)
        # Slim writes at the root only: the nested `derivatives/tvbo/dataset_description.json` declares a derivative dataset the run has not produced yet, and writing it would announce one that is not there.
        if slim and (role not in SLIM_ROLES or "/" in rel):
            continue
        if role == "gitignore":
            write(rel, "\n".join(layout_rules.gitignore_lines(record, templates, name)))
        elif role == "bidsignore":
            write(rel, "\n".join(layout_rules.bidsignore_lines(record, templates, name)))
        elif role == "dataset_description":
            nested = "/" in rel
            body = _dataset_description(
                name,
                "derivative" if nested else str(record.dataset_type),
                bids_version,
                [{"URL": "../.."}] if nested else None,
            )
            write(rel, json.dumps(body, indent=2))
        elif (seed := layout_rules.template_for(entry, templates)) is not None:
            target = root / rel
            if target.exists() and not force:
                typer.echo(f"  skip     {rel} (exists; --force to overwrite)")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                _seed(target, seed, name)
                typer.echo(f"  write    {rel}")

    if not slim:
        # Last, so a directory that a seed has just filled is not given a placeholder it does not need.
        for rel, _ in layout_rules.walk(record, templates):
            target = root / rel
            if layout_rules.is_tracked(rel, record, templates) and not any(target.iterdir()):
                (target / _KEEP).touch()
                typer.echo(f"  keep     {rel}/{_KEEP}")

        readme = root / layout_rules.file_relpath("readme", name, record)
        if readme.is_file():
            layout_rules.sync_layout(readme, record, name, templates)
            typer.echo(f"  layout   {readme.name}")
    typer.echo(f"\nNext: fill in {name}.yaml, then `tvbo run {name}.yaml`.")
    if slim:
        typer.echo(f"Or from Python: SimulationStudy.from_file('{name}.yaml').run()")


@app.command("layout")
def show_layout(
    what: str = typer.Argument("tree", help="tree | gitignore | bidsignore"),
    study: str = typer.Option("<Study>", "--study", "-s", help="Study name to interpolate into the layout."),
    template: list[str] = typer.Option([], "--template", "-t", help="Layout variant to include. Repeatable."),
    sync: list[Path] = typer.Option(
        [],
        "--sync",
        help="Rewrite each file's marker-delimited layout region in place instead of printing.",
    ),
) -> None:
    """Print part of the layout record, or splice the tree into a document.

    ``--sync`` is how documentation stops restating the layout: a file carrying the layout markers gets the current tree written into them, so a tree in prose can no longer fall behind the record.
    """
    record = layout_rules.load_layout()
    templates = tuple(template)
    if sync:
        for dest in sync:
            if not dest.is_file():
                raise typer.BadParameter(f"{dest} does not exist.")
            changed = layout_rules.sync_layout(dest, record, study, templates)
            typer.echo(f"{'updated' if changed else 'unchanged'}  {dest}")
        return
    if what == "tree":
        typer.echo(layout_rules.tree(record, study, templates))
    elif what == "gitignore":
        typer.echo("\n".join(layout_rules.gitignore_lines(record, templates, study)))
    elif what == "bidsignore":
        typer.echo("\n".join(layout_rules.bidsignore_lines(record, templates, study)))
    else:
        raise typer.BadParameter(f"Unknown view {what!r}. Choose tree, gitignore or bidsignore.")
