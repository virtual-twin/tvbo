"""The repository's licensing as `REUSE.toml` states it: every file of the package has a licence of its own, the licences named and the texts in `LICENSES/` match, the wheel declares the licences of the files it ships, each LinkML schema module declares CC BY 4.0, and the schema carries the package version."""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

import pathspec
import pytest
import yaml

import tvbo

REPO = Path(__file__).resolve().parents[1]
ANNOTATIONS = tomllib.loads((REPO / "REUSE.toml").read_text(encoding="utf-8"))["annotations"]
PYPROJECT = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
SCHEMA_MODULES = [
    path for path in sorted((REPO / "schema").glob("*.yaml")) if "id" in yaml.safe_load(path.read_text(encoding="utf-8"))
]
CC_BY = "https://creativecommons.org/licenses/by/4.0/"


def glob_pattern(glob: str) -> re.Pattern[str]:
    """Translate a `REUSE.toml` path glob: `*` stays inside one directory, `**` and `**/` span any number of them, and a backslash escapes the next character."""
    tokens = re.findall(r"\\.|\*\*/?|\*|[^\\*]+", glob)
    return re.compile(
        "".join(".*" if t.startswith("**") else "[^/]*" if t == "*" else re.escape(t.removeprefix("\\")) for t in tokens)
    )


def licence_of(path: str) -> str | None:
    """The `SPDX-License-Identifier` of the last annotation whose path matches `path`, the one REUSE applies."""
    for annotation in reversed(ANNOTATIONS):
        globs = annotation["path"] if isinstance(annotation["path"], list) else [annotation["path"]]
        if any(glob_pattern(glob).fullmatch(path) for glob in globs):
            return annotation.get("SPDX-License-Identifier")
    return None


def licence_ids(expression: str) -> set[str]:
    """The licence identifiers an SPDX expression names."""
    return set(re.findall(r"[\w.+-]+", expression)) - {"AND", "OR", "WITH"}


def tracked_files(*folders: str) -> list[str]:
    """The files under `folders`: those git tracks in a checkout, and every file outside one."""
    if (REPO / ".git").exists():
        listed = subprocess.run(
            ["git", "ls-files", "-z", *folders], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout
        return [path for path in listed.split("\0") if path]
    return [
        path.relative_to(REPO).as_posix()
        for folder in folders
        for path in (REPO / folder).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and not path.name.startswith(".")
    ]


def wheel_files() -> set[str]:
    """The files the wheel ships, selected as hatchling selects them: the tracked files of the package and the build `artifacts`, less the `exclude` globs, plus the `force-include` sources."""
    build = PYPROJECT["tool"]["hatch"]["build"]
    excluded = pathspec.GitIgnoreSpec.from_lines(build["exclude"])
    selected = [*tracked_files(*build["targets"]["wheel"]["packages"]), *build["artifacts"]]
    return {path for path in selected if not excluded.match_file(path)} | set(build["force-include"])


def test_every_file_of_the_package_is_licensed():
    """`tvbo/` has no default licence, so a file added there, or shipped in the wheel, without an entry that covers it fails here instead of passing as EUPL."""
    unlicensed = sorted(path for path in {*tracked_files("tvbo"), *wheel_files()} if licence_of(path) is None)
    assert not unlicensed, unlicensed


def test_the_licences_named_and_their_texts_match():
    expressions = [
        annotation["SPDX-License-Identifier"] for annotation in ANNOTATIONS if "SPDX-License-Identifier" in annotation
    ]
    expressions.append(PYPROJECT["project"]["license"])
    named = set().union(*map(licence_ids, expressions))
    texts = {path.stem for path in (REPO / "LICENSES").glob("*.txt")}
    assert sorted(named - texts) == [], "named without a text"
    assert sorted(texts - named) == [], "a text no expression names"


def test_the_wheel_declares_exactly_the_licences_of_the_files_it_ships():
    """`project.license` names every licence a shipped file carries and no other, and `project.license-files` ships exactly their texts."""
    licences = {path: licence_of(path) for path in wheel_files()}
    assert not [path for path, licence in licences.items() if licence is None]
    shipped = set().union(*map(licence_ids, licences.values()))
    assert licence_ids(PYPROJECT["project"]["license"]) == shipped
    assert sorted(PYPROJECT["project"]["license-files"]) == sorted(f"LICENSES/{name}.txt" for name in shipped)


@pytest.mark.parametrize("module", SCHEMA_MODULES, ids=lambda path: path.name)
def test_schema_module_declares_cc_by(module: Path):
    assert yaml.safe_load(module.read_text(encoding="utf-8")).get("license") == CC_BY


def test_schema_version_is_the_package_version():
    """`scripts/release.sh` writes both, so the schema a release ships names that release."""
    assert yaml.safe_load((REPO / "schema" / "tvbo_datamodel.yaml").read_text(encoding="utf-8"))["version"] == tvbo.__version__
