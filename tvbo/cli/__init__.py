"""TVB-O command-line interface.

See ``dev/tvbo-cli.md`` for the full design. The CLI is Typer-based,
registry-driven, and transport-aware. This module assembles the top-level
:class:`typer.Typer` ``app`` from per-verb sub-modules.
"""
from __future__ import annotations

import typer

from . import (
    config as _config_cmd,
    export as _export_cmd,
    formats as _formats_cmd,
    info as _info_cmd,
    import_ as _import_cmd,
    run as _run_cmd,
    save as _save_cmd,
    validate as _validate_cmd,
    version as _version_cmd,
    workflow as _workflow_cmd,
)

app = typer.Typer(
    name="tvbo",
    help="The Virtual Brain Ontology — command-line interface.",
    no_args_is_help=True,
    add_completion=True,
    pretty_exceptions_show_locals=False,
)

# Top-level verbs (single-command, no sub-tree)
app.command("run", help="Execute a SimulationStudy or SimulationExperiment.")(_run_cmd.run)
app.command("export", help="Render a SPEC into a target format (no execution).")(_export_cmd.export)
app.command("save", help="Like export, with bundled data when supported.")(_save_cmd.save)
app.command("import", help="Load a foreign file (auto-dispatch by extension).")(_import_cmd.import_)
app.command("info", help="Inspect a SPEC (tasks, outputs, declared backends).")(_info_cmd.info)
app.command("formats", help="List all registered I/O formats.")(_formats_cmd.formats)
app.command("version", help="Print the tvbo version.")(_version_cmd.version)

# Sub-trees (registered as their own Typer apps)
app.add_typer(_validate_cmd.app, name="validate", help="Validate YAML / OMEX / BIDS / SED-ML files.")
app.add_typer(_config_cmd.app, name="config", help="Manage CLI configuration.")
app.add_typer(_workflow_cmd.app, name="workflow", help="Plan / emit HPC + pipeline artefacts (slurm, snakemake, nextflow).")


__all__ = ["app"]
