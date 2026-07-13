"""TVB-O command-line interface.

See ``dev/tvbo-cli.md`` for the full design. The CLI is Typer-based,
registry-driven, and transport-aware. This module assembles the top-level
:class:`typer.Typer` ``app`` from per-verb sub-modules.
"""
from __future__ import annotations

import typer

# Shared helper sub-modules used by the verb modules below (which do
# ``from . import _common`` / ``_workflow`` at import time). Importing them here
# first binds them on the package so those relative imports resolve by attribute.
from . import _backends, _common, _workflow  # noqa: F401

from . import (
    config as _config_cmd,
    export as _export_cmd,
    formats as _formats_cmd,
    info as _info_cmd,
    import_ as _import_cmd,
    network as _network_cmd,
    run as _run_cmd,
    save as _save_cmd,
    skills as _skills_cmd,
    validate as _validate_cmd,
    version as _version_cmd,
    workflow as _workflow_cmd,
)

app = typer.Typer(
    name="tvbo",
    help=(
        "The Virtual Brain Ontology — command-line interface.\n"
        "\n"
        "Quick start:\n"
        "  tvbo run experiment:JR_MEG_FrequencyGradient_Optimization\n"
        "  tvbo info <SPEC>            inspect tasks, outputs, backends\n"
        "  tvbo export jax <SPEC>      render code without executing\n"
        "  tvbo workflow snakemake <SPEC> -o ./kit\n"
        "\n"
        "AI-assistant skills (Claude Code, Cursor, Copilot, raw API):\n"
        "  tvbo skills install --target claude-code     install user skills (default)\n"
        "  tvbo skills install --target cursor          install Cursor rules\n"
        "  tvbo skills install --target prompt > p.md   dump as a system prompt\n"
        "  tvbo skills uninstall                        remove TVBO-managed skill files\n"
        "  tvbo skills --help                           all targets, scopes, flags\n"
        "\n"
        "Docs: https://thevirtualbrain.github.io/tvbo/  (see Agentic Coding section)"
    ),
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
app.add_typer(_network_cmd.app, name="network", help="Build connectomes from a tractogram + parcellation (MRtrix wrapper).")
app.add_typer(_workflow_cmd.app, name="workflow", help="Plan / emit HPC + pipeline artefacts (slurm, snakemake, nextflow).")
app.add_typer(_skills_cmd.app, name="skills", help="Render skills for Claude Code / Copilot / Cursor; install user skills locally.")


__all__ = ["app"]
