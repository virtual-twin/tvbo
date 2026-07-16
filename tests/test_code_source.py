"""CodeSource resolution — a recipe's callable code as a local dir or git repo.

Covers ``register_recipe_code_paths`` / ``_resolve_code_source`` (tvbo.utils):
the ``code/`` convention, an explicit local ``path``, a ``git`` source (cloned +
cached, with ``ref``/``subdir``), and the ``path``/``git`` mutual-exclusion guard.
"""
import subprocess
import sys

import pytest

from tvbo.datamodel import pydantic as p
from tvbo.utils import (
    _resolve_code_source,
    register_recipe_code_paths,
)


def _write_module(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"VALUE = {value!r}\n")


def _import_fresh(name):
    sys.modules.pop(name, None)
    return __import__(name)


def test_code_dir_convention(tmp_path):
    """No code_source -> the conventional ``code/`` subdir beside the YAML."""
    _write_module(tmp_path / "code" / "conv_mod.py", "convention")
    inserted = register_recipe_code_paths(str(tmp_path / "Study.yaml"))
    try:
        assert inserted and inserted[0].endswith("/code")
        assert _import_fresh("conv_mod").VALUE == "convention"
    finally:
        sys.modules.pop("conv_mod", None)
        for e in inserted:
            sys.path.remove(e)


def test_local_path_relative_to_yaml(tmp_path):
    """code_source.path resolves relative to the recipe YAML's directory."""
    _write_module(tmp_path / "src" / "path_mod.py", "from_path")
    cs = p.CodeSource(path="src")
    inserted = register_recipe_code_paths(str(tmp_path / "Study.yaml"), cs)
    try:
        assert inserted and inserted[0].endswith("/src")
        assert _import_fresh("path_mod").VALUE == "from_path"
    finally:
        sys.modules.pop("path_mod", None)
        for e in inserted:
            sys.path.remove(e)


def test_git_source_ref_subdir_and_cache(tmp_path, monkeypatch):
    """code_source.git shallow-clones a repo, checks out ref, uses subdir, caches."""
    repo = tmp_path / "repo"
    _write_module(repo / "recipe" / "git_mod.py", "from_git")
    run = lambda *a: subprocess.run(a, cwd=repo, check=True, capture_output=True)
    run("git", "init", "-q")
    run("git", "add", "-A")
    run("git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init")
    run("git", "branch", "-M", "main")

    monkeypatch.setenv("TVBO_CACHE", str(tmp_path / "cache"))
    cs = p.CodeSource(git=f"file://{repo}", ref="main", subdir="recipe")
    inserted = register_recipe_code_paths(str(tmp_path / "Study.yaml"), cs)
    try:
        assert inserted and "cache/code_sources" in inserted[0]
        assert _import_fresh("git_mod").VALUE == "from_git"
        # a second resolve reuses the clone (same path, no re-clone)
        assert str(_resolve_code_source(cs, None)) == inserted[0]
    finally:
        sys.modules.pop("git_mod", None)
        for e in inserted:
            sys.path.remove(e)


def test_path_and_git_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        _resolve_code_source(p.CodeSource(path="a", git="b"), None)


def test_absent_code_source_falls_back_to_none(tmp_path):
    """A study without code_source and without a code/ dir registers nothing."""
    assert register_recipe_code_paths(str(tmp_path / "Study.yaml")) == []


def test_empty_code_source_falls_back_to_convention(tmp_path):
    """A declared-but-empty code_source (no path/git) still uses the code/ convention."""
    _write_module(tmp_path / "code" / "empty_cs_mod.py", "convention")
    inserted = register_recipe_code_paths(str(tmp_path / "Study.yaml"), p.CodeSource())
    try:
        assert inserted and inserted[0].endswith("/code")
        assert _import_fresh("empty_cs_mod").VALUE == "convention"
    finally:
        sys.modules.pop("empty_cs_mod", None)
        for e in inserted:
            sys.path.remove(e)
