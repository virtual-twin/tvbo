"""CLI smoke tests using typer.testing.CliRunner.

These tests exercise the C0/C1 surface (no execution-heavy paths).
"""

from __future__ import annotations

from typer.testing import CliRunner

from tvbo.cli import app

runner = CliRunner()


def test_root_help_shows_subcommands():
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    for verb in ("run", "export", "save", "import", "info", "formats", "validate", "config", "version"):
        assert verb in res.stdout


def test_version_prints_semver():
    import tvbo

    res = runner.invoke(app, ["version"])
    assert res.exit_code == 0
    assert tvbo.__version__ in res.stdout


def test_formats_lists_yaml():
    res = runner.invoke(app, ["formats"])
    assert res.exit_code == 0
    assert "yaml" in res.stdout.lower()


def test_formats_json_is_machine_readable():
    import json

    res = runner.invoke(app, ["formats", "--json"])
    assert res.exit_code == 0
    payload = json.loads(res.stdout.strip().splitlines()[-1])
    assert isinstance(payload, list)
    keys = {row.get("format") for row in payload}
    assert "yaml" in keys


def test_info_unknown_spec_errors():
    res = runner.invoke(app, ["info", "this-name-does-not-exist-anywhere"])
    assert res.exit_code != 0


def test_validate_help():
    res = runner.invoke(app, ["validate", "--help"])
    assert res.exit_code == 0
    assert "schema" in res.stdout


def test_config_path():
    res = runner.invoke(app, ["config", "path"])
    assert res.exit_code == 0
    assert "tvbo" in res.stdout
