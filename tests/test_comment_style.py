"""The stacked-comment rule is checked by a script, so the script is checked here.

A run of two or more consecutive whole-line ``#`` comments is what the convention
forbids. The check reads a diff rather than the tree, so that the blocks already in the
repository need not be cleared before the rule can hold for everything written after it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_comment_blocks import added_blocks  # noqa: E402

CHECKER = Path(__file__).resolve().parents[1] / "scripts" / "check_comment_blocks.py"


def _diff(path: str, added: list[str], start: int = 10) -> str:
    """A one-hunk diff adding *added* at *start*, away from line 1's header exemption."""
    body = "".join(f"+{line}\n" for line in added)
    return f"--- a/{path}\n+++ b/{path}\n@@ -{start},0 +{start},{len(added)} @@\n{body}"


def test_a_stacked_block_is_reported():
    diff = _diff("tvbo/x.py", ["# one line of it", "# and a second", "value = 1"])

    (path, block), = added_blocks(diff)
    assert path == "tvbo/x.py"
    assert [line for _, line in block] == ["# one line of it", "# and a second"]
    assert block[0][0] == 10, "reported against the first line of the run"


def test_one_line_of_rationale_is_allowed():
    assert not list(added_blocks(_diff("tvbo/x.py", ["# just the one", "value = 1"])))


def test_a_run_broken_by_code_is_two_separate_lines():
    diff = _diff("tvbo/x.py", ["# first", "a = 1", "# second", "b = 2"])

    assert not list(added_blocks(diff))


def test_a_shebang_is_not_a_comment_block():
    assert not list(added_blocks(_diff("run.sh", ["#!/bin/bash", "# what it does"], start=1)))


def test_a_file_header_is_not_a_comment_block():
    """Every TVBO source file opens with a copyright and licence block."""
    header = ["#  module.py", "#", "# Copyright (c) 2024 Charite", "# Licensed under the EUPL"]

    assert not list(added_blocks(_diff("tvbo/x.py", header, start=1)))


def test_an_attribute_doc_is_not_a_comment_block():
    """``#:`` is the only way to document a module constant — it has no docstring slot."""
    docs = ["#: the first thing", "#: the second thing"]

    assert not list(added_blocks(_diff("tvbo/x.py", docs)))


def test_the_run_at_the_end_of_a_hunk_is_still_reported():
    (_, block), = added_blocks(_diff("tvbo/x.py", ["# trailing", "# block"]))

    assert len(block) == 2


def _repo(tmp_path: Path, *files: tuple[str, str]) -> Path:
    """A git repo with one base commit, then *files* written on top but uncommitted.

    ``git add -N`` registers the intent to add, which is what puts an as-yet-untracked
    file into ``git diff`` at all — without it the checker would see an empty diff and
    the test would pass for the wrong reason.
    """
    def run(*cmd):
        subprocess.run(cmd, cwd=tmp_path, check=True, capture_output=True)

    run("git", "init", "-q", "-b", "base")
    run("git", "config", "user.email", "t@t")
    run("git", "config", "user.name", "t")
    (tmp_path / "seed.py").write_text("x = 1\n")
    run("git", "add", "-A")
    run("git", "commit", "-qm", "base")
    for name, text in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    run("git", "add", "-N", ".")
    return tmp_path


def _check(repo: Path, ref: str = "base"):
    return subprocess.run(
        [sys.executable, str(CHECKER), ref], cwd=repo, capture_output=True, text=True
    )


def test_the_checker_fails_on_a_block_a_real_diff_adds(tmp_path: Path):
    """The end-to-end path CI runs, against a repo whose diff is known to be dirty."""
    repo = _repo(tmp_path, ("new.py", "def f():\n    # one\n    # two\n    return 1\n"))

    proc = _check(repo)

    assert proc.returncode == 1, proc.stdout
    assert "new.py:2" in proc.stdout


def test_the_checker_passes_on_a_clean_diff(tmp_path: Path):
    """And exits 0 when the same path is walked over compliant code."""
    repo = _repo(tmp_path, ("new.py", "def f():\n    # just the one\n    return 1\n"))

    proc = _check(repo)

    assert proc.returncode == 0, proc.stdout


def test_a_generated_file_is_not_the_authors_to_fix(tmp_path: Path):
    """A frozen reference dump carries whatever it was generated from."""
    repo = _repo(tmp_path, ("tests/reference_data/database/x.yaml", "a: 1\n# one\n# two\n"))

    assert _check(repo).returncode == 0


def test_a_missing_base_ref_is_reported_rather_than_passed(tmp_path: Path):
    """A gate that cannot run says so; exiting 0 would read as 'nothing to report'."""
    proc = _check(_repo(tmp_path), ref="no/such/ref")

    assert proc.returncode == 2
    assert "no such ref" in proc.stdout
