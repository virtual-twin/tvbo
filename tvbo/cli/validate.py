"""``tvbo validate`` — sub-tree of validators (schema today; OMEX/SED-ML/BIDS later)."""

from __future__ import annotations

from pathlib import Path

import typer

from . import _common

app = typer.Typer(name="validate", no_args_is_help=True)


def _declared_class(data: object) -> str | None:
    """Target class named by the document's own file envelope, or None.

    Reads ``tvbo_class`` (the envelope key every self-describing TVBO file carries, CURIE-prefixed as ``tvbo:Network``) and strips the prefix to the bare class name.
    """
    if not isinstance(data, dict):
        return None
    declared = data.get("tvbo_class")
    return str(declared).split(":")[-1] if declared else None


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
        target_class = _declared_class(data) or "SimulationExperiment"

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
