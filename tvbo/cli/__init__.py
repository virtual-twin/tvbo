"""TVB-O command-line interface.

See ``dev/tvbo-cli.md`` for the full design. The CLI is Typer-based, registry-driven, and transport-aware. This module assembles the top-level :class:`typer.Typer` ``app`` from per-verb sub-modules.
"""

from __future__ import annotations

import typer

# Bound on the package first, so the verb modules' relative imports resolve by attribute.
from . import _backends, _common, _workflow  # noqa: F401
from . import (
    cache as _cache_cmd,
)
from . import (
    config as _config_cmd,
)
from . import (
    export as _export_cmd,
)
from . import (
    figures as _figures_cmd,
)
from . import (
    formats as _formats_cmd,
)
from . import (
    import_ as _import_cmd,
)
from . import (
    info as _info_cmd,
)
from . import (
    install as _install_cmd,
)
from . import (
    network as _network_cmd,
)
from . import (
    run as _run_cmd,
)
from . import (
    save as _save_cmd,
)
from . import (
    skills as _skills_cmd,
)
from . import (
    units as _units_cmd,
    study as _study_cmd,
)
from . import (
    validate as _validate_cmd,
)
from . import (
    verify as _verify_cmd,
)
from . import (
    version as _version_cmd,
)
from . import (
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
        "Docs: https://virtual-twin.github.io/tvbo/  (see Agentic Coding section)"
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
app.command("verify", help="Check a study-of-studies is buildable (completeness / staleness / manifest coverage).")(
    _verify_cmd.verify
)
app.command("version", help="Print the tvbo version.")(_version_cmd.version)

# Sub-trees (registered as their own Typer apps)
app.add_typer(_validate_cmd.app, name="validate", help="Validate YAML / OMEX / BIDS / SED-ML files.")
app.add_typer(_config_cmd.app, name="config", help="Manage CLI configuration.")
app.add_typer(_cache_cmd.app, name="cache", help="Inspect and reclaim tvbo's caches.")
app.add_typer(_network_cmd.app, name="network", help="Build connectomes from a tractogram + parcellation (MRtrix wrapper).")
app.add_typer(
    _figures_cmd.app, name="figure", help="Render declarative figures (Figure / SimulationStudy YAML) via bsplot codegen."
)
app.add_typer(_workflow_cmd.app, name="workflow", help="Plan / emit HPC + pipeline artefacts (slurm, snakemake, nextflow).")
app.add_typer(
    _skills_cmd.app, name="skills", help="Render skills for Claude Code / Copilot / Cursor; install user skills locally."
)
app.add_typer(_units_cmd.app, name="units", help="Inspect the QUDT-vendored unit vocabulary, and curate a new unit into it.")
app.add_typer(_study_cmd.app, name="study", help="Scaffold and inspect a BIDS study dataset from the layout record.")
app.add_typer(_install_cmd.app, name="install", help="Provision optional native components pip cannot place (e.g. AUTO-07p).")


def _print_version(value: bool) -> None:
    """Print the version and exit, so `--version` answers before any verb is dispatched."""
    if value:
        import tvbo

        typer.echo(tvbo.__version__)
        raise typer.Exit


@app.callback()
def _configure(
    log_level: str = typer.Option(
        None,
        "--log-level",
        "-L",
        metavar="LEVEL",
        help="tvbo log level (DEBUG|INFO|WARNING|ERROR|OFF); overrides TVBO_LOG_LEVEL.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output (DEBUG)."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Errors only — suppress progress."),
    version: bool = typer.Option(
        False, "--version", "-V", help="Print the tvbo version and exit.", callback=_print_version, is_eager=True
    ),
) -> None:
    """Configure tvbo logging once for every verb.

    Progress and status flow through the central ``tvbo`` logger (see :mod:`tvbo.log`), so ``tvbo run`` and the in-process ``.run()`` API behave identically. ``--log-level`` wins over ``--quiet``/``--verbose``; with none set the level falls back to ``TVBO_LOG_LEVEL`` and then INFO.
    """
    from tvbo.log import configure_logging

    level = None
    if verbose:
        level = "DEBUG"
    if quiet:
        level = "ERROR"
    if log_level:
        level = log_level
    # CLI output is user-facing: keep it bare (no "LEVEL [name]" diagnostic prefix), matching the plain lines the CLI printed before. force=True so this format wins even if ``import tvbo`` already installed the default (diagnostic) handler because ``TVBO_LOG_LEVEL`` was set in the environment.
    configure_logging(level, fmt="%(message)s", force=True)


__all__ = ["app"]
