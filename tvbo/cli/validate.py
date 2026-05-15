"""``tvbo validate`` — sub-tree of validators (schema today; OMEX/SED-ML/BIDS later)."""
from __future__ import annotations

from pathlib import Path

import typer

from . import _common


app = typer.Typer(name="validate", no_args_is_help=True)


@app.command("schema", help="LinkML structural validation of a YAML file.")
def schema(
    path: Path = typer.Argument(..., exists=True, readable=True, help="YAML file."),
    target_class: str = typer.Option(
        None, "--class",
        help="LinkML target class (auto-detected from `class:` key when omitted).",
    ),
) -> None:
    """Validate *path* against the LinkML schema; auto-detects target class from `class:` key."""
    from linkml_runtime.loaders import yaml_loader
    from linkml.validator import Validator
    from linkml.validator.plugins import JsonschemaValidationPlugin

    # Locate the shipped schema
    import tvbo
    schema_path = Path(tvbo.__file__).parent.parent / "schema" / "tvbo_datamodel.yaml"
    if not schema_path.exists():
        # Fall back to the installed copy under tvbo/datamodel
        schema_path = Path(tvbo.__file__).parent / "datamodel" / "tvbo_datamodel.yaml"
    if not schema_path.exists():
        _common.die("Cannot locate tvbo_datamodel.yaml schema file.")

    if target_class is None:
        text = path.read_text(encoding="utf-8")
        # Extremely lightweight detection — accept either explicit `class:`
        # or fall back to common roots.
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("class:"):
                target_class = stripped.split(":", 1)[1].strip()
                break
        if target_class is None:
            target_class = "SimulationExperiment"

    validator = Validator(
        schema=str(schema_path),
        validation_plugins=[JsonschemaValidationPlugin(closed=False)],
    )
    report = validator.validate_file(str(path), target_class=target_class)
    if report.results:
        for r in report.results:
            typer.echo(f"  {r.severity}: {r.message}", err=True)
        _common.die(f"{len(report.results)} validation issue(s) in {path}.")
    typer.echo(f"OK — {path} is a valid {target_class}.")


# ---------------------------------------------------------------------------
# C5 stubs: bids / sedml / omex / all
# ---------------------------------------------------------------------------

@app.command("bids", help="Validate a BIDS dataset directory.")
def bids(
    path: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True,
                                readable=True, help="BIDS dataset root."),
) -> None:
    try:
        from bids_validator import BIDSValidator  # type: ignore
    except ImportError:
        _common.die(
            "bids_validator not installed. Install with: uv pip install bids-validator"
        )

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
    directory: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True,
                                     readable=True, help="Directory to walk."),
    pattern: str = typer.Option("*.yaml", "--pattern",
                                help="Glob pattern, e.g. '*.yml' or '**/*.yaml'."),
    fail_fast: bool = typer.Option(False, "--fail-fast", help="Stop at first failure."),
) -> None:
    files = sorted(directory.rglob(pattern))
    if not files:
        _common.die(f"No files matching {pattern!r} under {directory}.")
    failures: list[tuple[Path, str]] = []
    for fp in files:
        try:
            schema(path=fp, target_class=None)  # type: ignore[arg-type]
        except SystemExit as exc:
            failures.append((fp, str(exc)))
            if fail_fast:
                break
    if failures:
        typer.echo(f"\n{len(failures)} file(s) failed validation:", err=True)
        for fp, msg in failures:
            typer.echo(f"  - {fp}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"\nOK — all {len(files)} files validated.")
