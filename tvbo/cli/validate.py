"""``tvbo validate`` — sub-tree of validators (schema today; OMEX/SED-ML/BIDS later)."""

from __future__ import annotations

import re
from pathlib import Path

import typer

from tvbo.utils.yaml_loader import declared_class

from . import _common

app = typer.Typer(name="validate", no_args_is_help=True)


@app.command("schema", help="Structural JSON Schema validation of a YAML file.")
def schema(
    path: Path = typer.Argument(..., exists=True, readable=True, help="YAML file."),
    target_class: str = typer.Option(
        None,
        "--class",
        help="Target class (auto-detected from the file's `tvbo_class:` envelope when omitted).",
    ),
) -> None:
    """Validate *path* against the shipped JSON Schema; auto-detects the target class.

    Uses the lightweight ``jsonschema`` library against the pre-generated ``tvbo/datamodel/tvbo_datamodel.schema.json`` (produced from the LinkML schema at build time), so validation needs no runtime ``linkml``. The file is parsed with TVBO's loader so ``!include``/merge-key extensions and slot aliases resolve exactly as they do when the model is loaded.
    """
    import json

    import jsonschema

    import tvbo
    from tvbo.utils import yaml_loader

    schema_json = Path(tvbo.__file__).parent / "datamodel" / "tvbo_datamodel.schema.json"
    if not schema_json.exists():
        _common.die(
            f"Cannot locate the generated JSON Schema at {schema_json}. "
            "Run `make gen-linkml` (or reinstall) to regenerate the datamodel."
        )
    full = json.loads(schema_json.read_text(encoding="utf-8"))
    data = yaml_loader.load_as_dict(str(path))

    if target_class is None:
        target_class = declared_class(data) or "SimulationExperiment"

    defs = full.get("$defs", {})
    if target_class not in defs:
        _common.die(f"Unknown target class '{target_class}' (not in the schema's $defs).")

    # Validate the document as an instance of `target_class` via a $ref into $defs.
    class_schema = {"$schema": full.get("$schema"), "$defs": defs, "$ref": f"#/$defs/{target_class}"}
    validator_cls = jsonschema.validators.validator_for(class_schema)
    errors = sorted(
        validator_cls(class_schema).iter_errors(data),
        key=lambda e: list(e.absolute_path),
    )
    if errors:
        for e in errors:
            loc = "/".join(str(p) for p in e.absolute_path) or "<root>"
            typer.echo(f"  ERROR at {loc}: {e.message}", err=True)
        _common.die(f"{len(errors)} validation issue(s) in {path}.")
    typer.echo(f"OK — {path} is a valid {target_class}.")


# C5 stubs: bids / sedml / omex / all


@app.command("bids", help="Validate a BIDS dataset directory.")
def bids(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True, readable=True, help="BIDS dataset root."),
) -> None:
    """Validate a BIDS dataset directory using the `bids_validator` package."""
    try:
        from bids_validator import BIDSValidator  # type: ignore
    except ImportError:
        _common.die("bids_validator not installed. Install with: uv pip install bids-validator")

    v = BIDSValidator()
    bad: list[str] = []
    n_checked = 0
    for fp in path.rglob("*"):
        if not fp.is_file():
            continue
        rel = "/" + str(fp.relative_to(path))
        n_checked += 1
        if not v.is_bids(rel):
            bad.append(rel)
    if bad:
        for b in bad:
            typer.echo(f"  invalid: {b}", err=True)
        _common.die(f"{len(bad)} of {n_checked} files are not valid BIDS.")
    typer.echo(f"OK — {n_checked} files in {path} are valid BIDS.")


@app.command("sedml", help="Validate a SED-ML file (stub — full validator post-P3).")
def sedml(
    path: Path = typer.Argument(..., exists=True, readable=True, help="SED-ML XML file."),
) -> None:
    """Shallow SED-ML check — confirms the file has a `<sedML>` root element.

    Full L1V4 validation is scheduled for the post-P3 milestone.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    if "<sedML" not in text and "<sedml" not in text:
        _common.die(f"{path}: does not look like SED-ML (no <sedML> root element).")
    typer.echo(
        f"OK (shallow) — {path} contains a SED-ML root element. "
        "Full SED-ML L1V4 validation is scheduled for the post-P3 milestone."
    )


@app.command("omex", help="Validate a COMBINE/OMEX archive (stub — full validator post-P3).")
def omex(
    path: Path = typer.Argument(..., exists=True, readable=True, help="OMEX archive (.omex / .zip)."),
) -> None:
    """Shallow OMEX check — confirms the archive is a zip with `manifest.xml` at the root.

    Full COMBINE-archive validation is scheduled for the post-P3 milestone.
    """
    import zipfile

    if not zipfile.is_zipfile(path):
        _common.die(f"{path}: not a zip archive (OMEX must be a zip).")
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if "manifest.xml" not in names:
            _common.die(f"{path}: missing manifest.xml at archive root.")
    typer.echo(
        f"OK (shallow) — {path} is a zip with manifest.xml at the root. "
        "Full COMBINE archive validation is scheduled for the post-P3 milestone."
    )


@app.command("all", help="Recursively validate every YAML file under DIR via `validate schema`.")
def all_(
    directory: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True, readable=True, help="Directory to walk."
    ),
    pattern: str = typer.Option("*.yaml", "--pattern", help="Glob pattern, e.g. '*.yml' or '**/*.yaml'."),
    fail_fast: bool = typer.Option(False, "--fail-fast", help="Stop at first failure."),
) -> None:
    """Recursively run `validate schema` on every YAML file under *directory* matching *pattern*."""
    files = sorted(directory.rglob(pattern))
    if not files:
        _common.die(f"No files matching {pattern!r} under {directory}.")
    failures: list[tuple[Path, str]] = []
    for fp in files:
        try:
            schema(path=fp, target_class=None)  # type: ignore[arg-type]
        except (typer.Exit, SystemExit) as exc:
            failures.append((fp, str(exc)))
            if fail_fast:
                break
    if failures:
        typer.echo(f"\n{len(failures)} file(s) failed validation:", err=True)
        for fp, _msg in failures:
            typer.echo(f"  - {fp}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"\nOK — all {len(files)} files validated.")


def _entity_problems(name: str) -> list[str]:
    """BIDS entity values in ``name`` that are not alphanumeric.

    BIDS requires an entity value to be alphanumeric, so a hyphen or underscore inside one silently changes where the key/value boundary falls and makes the file unqueryable.
    """
    stem = name.split(".")[0]
    bad = []
    for part in stem.split("_")[:-1]:
        if "-" not in part:
            bad.append(f"{part!r} is not a `key-value` entity")
            continue
        key, _, value = part.partition("-")
        if not value.isalnum():
            bad.append(f"entity {key}- has the non-alphanumeric value {value!r}")
    return bad


FIGURE_NAME = re.compile(r"^fig-[A-Za-z0-9]+_desc-[a-z][A-Za-z0-9]*$")
"""The one naming rule for a rendered display item: ``fig-<id>_desc-<descriptor>``, the identifier the document prints and a camelCase descriptor, so the record, the script that draws it, the image and the caption all carry the same name."""


def _figure_name_problems(name: str) -> list[str]:
    """Whether a file in the figures role is named by the central rule.

    Checked here rather than left to each study because the name is the join between a figure's record, its generated script, its image and its caption: when one study invents its own spelling, that figure exists under two names at once and neither side can tell which is current.
    """
    stem = name.split(".")[0]
    if FIGURE_NAME.match(stem):
        return []
    return ["not named `fig-<id>_desc-<descriptor>` (descriptor camelCase)"]


def _is_rule(line: str) -> bool:
    """Whether a generated ignore line carries a rule, as opposed to the blank lines and commentary that only make the file readable."""
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith("#")


def _suffix_problems(path: Path) -> list[str]:
    """Whether ``path``'s BIDS suffix agrees with the class its own envelope declares.

    A file named ``*_dynamics.yaml`` whose envelope says ``tvbo:Network`` is a rename that went wrong, and nothing else notices: both halves are individually valid.
    """
    from tvbo.adapters.bids import SPEC_SUFFIXES
    from tvbo.utils import yaml_loader

    suffix = path.name.split(".")[0].rsplit("_", 1)[-1]
    expected = SPEC_SUFFIXES.get(suffix)
    if expected is None:
        return [f"suffix {suffix!r} is not in the tvbo suffix vocabulary ({', '.join(sorted(SPEC_SUFFIXES))})"]
    # The include-aware loader, because a fragment may itself `!include` a sibling (a network naming its coupling), which a plain safe_load rejects as an unknown tag. A fragment that will not parse at all is one of this command's problems to report, not a traceback out of it.
    try:
        declared = declared_class(yaml_loader.load_as_dict(str(path)))
    except Exception as exc:
        return [f"cannot be parsed ({type(exc).__name__}: {exc})"]
    if declared is None:
        return [f"named `_{suffix}` but carries no `tvbo_class` envelope to check it against"]
    if declared != expected:
        return [f"named `_{suffix}` (which means {expected}) but its envelope declares {declared}"]
    return []


def _templates_of(root: Path, name: str, record) -> tuple[str, ...]:
    """Which layout variant this study was built with, read from its own generated `.gitignore`.

    A replication ignores entries a plain study does not have, so checking one against the other reports a difference that is not a fault. `tvbo study init` stamps the variant into the file's header, which makes the study say what it is instead of the caller having to remember.
    """
    from tvbo.utils import study_layout as layout_rules

    gate = root / layout_rules.file_relpath("gitignore", name, record)
    return layout_rules.templates_of(gate.read_text(encoding="utf-8")) if gate.is_file() else ()


@app.command("study", help="Validate a study dataset against the layout record.")
def study(
    path: Path = typer.Argument(
        Path(), exists=True, file_okay=False, dir_okay=True, readable=True, help="Study dataset root."
    ),
    template: list[str] = typer.Option([], "--template", "-t", help="Layout variant the study was built with."),
) -> None:
    """Check a study dataset against the one layout record.

    Three things, none of which any other check catches: the tree matches the record (:mod:`tvbo.utils.study_layout`), each spec fragment's BIDS suffix agrees with the class its envelope declares, and its entity values are legal BIDS. Every rule the record generates must still be present in the ignore files; a study may add its own rules and order them as git requires, but silently dropping one of the record's is reported.
    """
    import json

    from tvbo.cli.study import _checked_templates
    from tvbo.utils import study_layout as layout_rules

    record = layout_rules.load_layout()
    root = path.resolve()
    name = root.name
    problems: list[str] = []
    templates = _checked_templates(template, record) or _templates_of(root, name, record)

    for rel, directory in layout_rules.walk(record, templates):
        if str(directory.level) == "required" and not (root / rel).is_dir():
            problems.append(f"{rel}/: required by the layout but missing")
    for rel, entry in layout_rules.iter_files(record, templates):
        rel = layout_rules.interpolate(rel, name)
        if str(entry.level) == "required" and not (root / rel).is_file():
            problems.append(f"{rel}: required by the layout but missing")

    description = root / "dataset_description.json"
    if description.is_file():
        declared_type = (json.loads(description.read_text(encoding="utf-8")) or {}).get("DatasetType")
        if declared_type != str(record.dataset_type):
            problems.append(f"dataset_description.json declares DatasetType {declared_type!r}, not {record.dataset_type!r}")

    for role, lines in (
        ("gitignore", layout_rules.gitignore_lines(record, templates, name)),
        ("bidsignore", layout_rules.bidsignore_lines(record, templates, name)),
    ):
        generated = root / layout_rules.file_relpath(role, name, record)
        if not generated.is_file():
            continue
        present = set(generated.read_text(encoding="utf-8").splitlines())
        dropped = [rule for rule in lines if _is_rule(rule) and rule not in present]
        if dropped:
            shown = ", ".join(dropped[:3]) + (", ..." if len(dropped) > 3 else "")
            problems.append(
                f"{generated.name}: {len(dropped)} rule(s) the layout record declares are missing ({shown}) — "
                "restore them, or change schema/study_layout.yaml if the record is what is wrong"
            )

    spec_dir = root / layout_rules.relpath("spec", record)
    n_spec = 0
    if spec_dir.is_dir():
        for fp in sorted(spec_dir.rglob("*.yaml")):
            n_spec += 1
            problems.extend(f"{fp.relative_to(root)}: {p}" for p in _suffix_problems(fp) + _entity_problems(fp.name))

    figures_dir = root / layout_rules.relpath("figures", record)
    n_figures = 0
    if figures_dir.is_dir():
        for fp in sorted(figures_dir.iterdir()):
            if not fp.is_file() or fp.name.startswith("."):
                continue
            n_figures += 1
            problems.extend(f"{fp.relative_to(root)}: {p}" for p in _figure_name_problems(fp.name))

    results_dir = root / layout_rules.relpath("results", record)
    n_results = 0
    if results_dir.is_dir():
        for fp in sorted(results_dir.glob("*_result.*")):
            n_results += 1
            problems.extend(f"{fp.relative_to(root)}: {p}" for p in _entity_problems(fp.name))

    if problems:
        for p in problems:
            typer.echo(f"  {p}", err=True)
        _common.die(f"{len(problems)} problem(s) in {root}.")
    typer.echo(f"OK — {root} conforms to the layout ({n_spec} spec fragment(s), {n_results} result file(s)).")
