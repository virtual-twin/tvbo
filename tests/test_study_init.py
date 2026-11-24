"""`tvbo study init` against the project-scaffold guidelines: the seeds load, render and carry what an author and a coding agent need, and version control, licences and CI follow the options without committing anything for the author."""

from __future__ import annotations

import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from tvbo.cli import app


def _init(tmp_path, *args):
    result = CliRunner().invoke(app, ["study", "init", *args, "--in", str(tmp_path)])
    assert result.exit_code == 0, result.output
    return tmp_path / args[0]


def _validate(root):
    return CliRunner().invoke(app, ["validate", "study", str(root)])


def test_the_seeded_recipe_loads(tmp_path):
    from tvbo import SimulationStudy

    root = _init(tmp_path, "Plain")
    study = SimulationStudy.from_file(root / "Plain.yaml")
    assert [e.id for e in study.experiments] == [1]


def test_validate_loads_the_recipe(tmp_path):
    root = _init(tmp_path, "Plain")
    assert _validate(root).exit_code == 0
    (root / "Plain.yaml").write_text((root / "Plain.yaml").read_text().replace("id: 1", "name: '1'"))
    result = _validate(root)
    assert result.exit_code != 0 and "does not load" in result.output


def test_every_file_the_report_project_reads_is_seeded(tmp_path):
    root = _init(tmp_path, "Plain")
    bib = yaml.safe_load((root / "docs" / "_quarto.yml").read_text()).get("bibliography")
    assert (root / "docs" / bib).is_file()


def test_the_report_seed_runs_no_placeholder_as_code(tmp_path):
    root = _init(tmp_path, "Plain")
    report = (root / "docs" / "report.qmd").read_text()
    for cell in report.split("```{python}")[1:]:
        compile(cell.split("```")[0], "report.qmd", "exec")
    assert '`{python} fmt(M["..."])`' not in report


def test_a_collection_names_its_nested_study_apart_from_itself(tmp_path):
    root = _init(tmp_path, "Coll", "-t", "collection")
    assert "Coll/Coll.yaml" not in (root / "Coll.yaml").read_text()


def test_the_replication_readme_names_only_directories_the_layout_has(tmp_path):
    root = _init(tmp_path, "Rep", "-t", "replication")
    assert "prov/" not in (root / "README.md").read_text()


def test_agents_and_changes_and_licences_are_seeded(tmp_path):
    root = _init(tmp_path, "Plain")
    agents = (root / "AGENTS.md").read_text()
    assert "tvbo validate study" in agents and "tvbo skills install" in agents and "Plain/" in agents
    assert (root / "CHANGES").read_text().startswith("0.1.0 ")
    assert sorted(p.name for p in (root / "LICENSES").iterdir()) == ["CC-BY-4.0.txt", "MIT.txt"]
    reuse = (root / "REUSE.toml").read_text()
    assert 'SPDX-License-Identifier = "CC-BY-4.0"' in reuse and 'path = "code/**"' in reuse
    assert "license: CC-BY-4.0" in (root / "CITATION.cff").read_text()


def test_the_data_licence_is_chosen_and_an_unknown_one_refused(tmp_path):
    root = _init(tmp_path, "Open", "--license", "CC0-1.0")
    assert (root / "LICENSES" / "CC0-1.0.txt").is_file()
    assert "license: CC0-1.0" in (root / "CITATION.cff").read_text()
    result = CliRunner().invoke(app, ["study", "init", "Odd", "--license", "WTFPL", "--in", str(tmp_path)])
    assert result.exit_code != 0


@pytest.mark.parametrize("ci,path", [("github", ".github/workflows/study.yml"), ("gitlab", ".gitlab-ci.yml")])
def test_ci_validates_the_study_in_the_pinned_image(tmp_path, ci, path, monkeypatch):
    import tvbo
    from tvbo.run.workflow import container_tag

    monkeypatch.delenv("TVBO_CONTAINER_IMAGE", raising=False)
    monkeypatch.delenv("TVBO_CONTAINER_TAG", raising=False)
    root = _init(tmp_path, "Plain", "--ci", ci)
    text = (root / path).read_text()
    assert f"ghcr.io/virtual-twin/tvbo:{container_tag(tvbo.__version__)}" in text and "tvbo validate study ." in text


def test_ci_runs_in_the_image_the_environment_names(tmp_path, monkeypatch):
    """The scaffolded CI pulls the image a kit pulls, so a site mirror set with `TVBO_CONTAINER_IMAGE` / `TVBO_CONTAINER_TAG` reaches both."""
    monkeypatch.setenv("TVBO_CONTAINER_IMAGE", "mirror.local/tvbo")
    monkeypatch.setenv("TVBO_CONTAINER_TAG", "9.9.9")
    root = _init(tmp_path, "Plain", "--ci", "gitlab")
    assert "name: mirror.local/tvbo:9.9.9" in (root / ".gitlab-ci.yml").read_text()


@pytest.mark.parametrize(
    "version,tag",
    [
        ("0.9.0", "0.9.0"),
        ("0.9.0rc1", "0.9.0-rc.1"),
        ("1.0.0b2", "1.0.0-beta.2"),
        ("0.9.1.dev3+gabc", "latest"),
        ("0.9.0.post1", "latest"),
    ],
)
def test_the_ci_image_is_the_tag_the_docker_workflow_publishes(version, tag, monkeypatch):
    from tvbo.run import workflow

    monkeypatch.delenv("TVBO_CONTAINER_TAG", raising=False)
    monkeypatch.setattr("tvbo.__version__", version)
    assert workflow.container_tag(version) == tag
    assert workflow.default_container_tag() == tag


def test_a_rerun_keeps_an_edited_ci_configuration(tmp_path):
    root = _init(tmp_path, "Plain")
    workflow = root / ".github" / "workflows" / "study.yml"
    workflow.write_text("edited\n")
    _init(tmp_path, "Plain")
    assert workflow.read_text() == "edited\n"


@pytest.mark.parametrize("option", [["--git"], ["--datalad"], ["--ci", "gitlab"]])
def test_slim_refuses_version_control_and_ci(tmp_path, option):
    result = CliRunner().invoke(app, ["study", "init", "Slim", "--slim", *option, "--in", str(tmp_path)])
    assert result.exit_code != 0 and "--slim" in result.output


def test_no_ci_writes_none(tmp_path):
    root = _init(tmp_path, "Plain", "--ci", "none")
    assert not (root / ".github").exists() and not (root / ".gitlab-ci.yml").exists()


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_git_makes_a_repository_and_commits_nothing(tmp_path):
    root = _init(tmp_path, "Plain", "--git")
    assert (root / ".git").is_dir()
    assert subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True).returncode != 0


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_no_repository_inside_an_existing_one(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    assert not (_init(tmp_path, "Inner", "--git") / ".git").exists()


def test_no_repository_unless_asked(tmp_path):
    assert not (_init(tmp_path, "Plain") / ".git").exists()


@pytest.mark.skipif(shutil.which("datalad") is None, reason="needs datalad")
def test_datalad_annexes_the_data_and_saves_nothing_of_the_scaffold(tmp_path):
    root = _init(tmp_path, "Dl", "--datalad")
    assert (root / ".datalad").is_dir()
    ignored = (root / ".gitignore").read_text().splitlines()
    assert not any(line.startswith(("sourcedata/*", "derivatives/*")) for line in ignored)
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True).stdout
    assert "Dl.yaml" in status
    assert "datalad run" in (root / "README.md").read_text()
    assert _validate(root).exit_code == 0


@pytest.mark.skipif(shutil.which("datalad") is None, reason="needs datalad")
def test_a_datalad_rerun_adds_its_section_once(tmp_path):
    root = _init(tmp_path, "Dl", "--datalad")
    _init(tmp_path, "Dl", "--datalad")
    assert [(root / doc).read_text().count("## DataLad") for doc in ("README.md", "AGENTS.md")] == [1, 1]


def test_the_network_a_new_study_is_seeded_with_ships_in_the_wheel():
    """`tvbo study init` seeds a runnable recipe, so the network it names must be in the wheel, which excludes network files unless they are force-included."""
    shipped = tomllib.loads((Path(__file__).resolve().parent.parent / "pyproject.toml").read_text())["tool"]["hatch"]["build"][
        "force-include"
    ]
    templates = (Path(__file__).resolve().parent.parent / "tvbo" / "templates" / "study").glob("*.tmpl")
    seeded = {name for t in templates for name in re.findall(r"tvbo:networks/(\S+)", t.read_text())}
    assert seeded, "the study template seeds a network"
    assert {f"tvbo/database/networks/{name}.h5" for name in seeded} <= set(shipped)
