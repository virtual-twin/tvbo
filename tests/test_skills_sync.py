"""Tests for the ``tvbo skills sync`` repo-side guard.

Covers the failure modes the check has to catch: rendered copies that have
drifted from their canonical source, rendered copies with no canonical source
behind them (a personal skill committed by accident), shipped user skills
pointing at maintainer skills that never ship, and ``requires_extras`` naming
a group that does not exist in ``pyproject.toml``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from tvbo.cli.skills import (
    _find_bad_extras,
    _find_leaked_refs,
    _find_orphans,
    _references,
    _sync_check,
)
from tvbo.skills import load_canonical
from tvbo.skills._render import render_agents_md, render_claude_code, render_copilot


def _canonical(
    root: Path,
    name: str,
    *,
    audience: str = "maintainer",
    body: str = "body",
    extras: list[str] | None = None,
) -> None:
    """Write a minimal canonical skill under *root*."""
    d = root / name
    d.mkdir(parents=True)
    extras_line = f"  requires_extras: {extras!r}\n" if extras is not None else ""
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test skill\n"
        f"metadata:\n  audience: {audience}\n{extras_line}---\n\n# {name}\n\n{body}\n",
        encoding="utf-8",
    )


@pytest.fixture
def synced(tmp_path):
    """A canonical root plus its freshly-rendered, in-sync output dirs."""
    src = tmp_path / "skills"
    _canonical(src, "real-skill")
    skills = load_canonical([src])

    claude_dir = tmp_path / ".claude" / "skills"
    copilot_dir = tmp_path / ".github" / "instructions"
    agents_md = tmp_path / "AGENTS.md"
    for s in skills:
        render_claude_code(s, claude_dir)
        render_copilot(s, copilot_dir)
    render_agents_md(skills, agents_md)
    return skills, claude_dir, copilot_dir, agents_md, tmp_path


# ------------------------------------------------------------- _find_orphans


def test_no_orphans_when_every_render_has_a_source(synced):
    skills, claude_dir, copilot_dir, *_ = synced
    assert _find_orphans(skills, claude_dir, copilot_dir) == []


def test_detects_stray_claude_skill(synced):
    skills, claude_dir, copilot_dir, *_ = synced
    stray = claude_dir / "personal-thing"
    stray.mkdir()
    (stray / "SKILL.md").write_text("---\nname: personal-thing\n---\n\nmine\n")

    assert _find_orphans(skills, claude_dir, copilot_dir) == [".claude/skills/personal-thing/"]


def test_detects_stray_copilot_instructions(synced):
    skills, claude_dir, copilot_dir, *_ = synced
    (copilot_dir / "ghost.instructions.md").write_text("---\napplyTo: '**'\n---\n\nx\n")

    assert _find_orphans(skills, claude_dir, copilot_dir) == [".github/instructions/ghost.instructions.md"]


def test_user_skill_needs_no_copilot_render(tmp_path):
    """`sync` only renders copilot files for maintainer skills, so a user
    skill's *absent* instructions file is not an orphan."""
    src = tmp_path / "skills"
    _canonical(src, "user-skill", audience="user")
    skills = load_canonical([src])

    claude_dir = tmp_path / ".claude" / "skills"
    copilot_dir = tmp_path / ".github" / "instructions"
    copilot_dir.mkdir(parents=True)
    for s in skills:
        render_claude_code(s, claude_dir)

    assert _find_orphans(skills, claude_dir, copilot_dir) == []


# ----------------------------------------------------------- _find_leaked_refs


@pytest.mark.parametrize(
    "body, expected",
    [
        ("defers to **codegen-templates** (render internals)", True),
        ("resolve it with /codegen-templates first", True),
        ("see the `codegen-templates` skill for details", True),
        ("nothing to see here", False),
    ],
)
def test_reference_forms(body, expected):
    assert _references(body, "codegen-templates") is expected


def test_bare_substring_is_not_a_reference():
    """A skill named `git` must not match every mention of gitignore."""
    assert not _references("everything else is gitignored; do not commit it", "git")


def test_user_skill_referencing_maintainer_skill_is_a_leak(tmp_path):
    src = tmp_path / "skills"
    _canonical(src, "internal-only", audience="maintainer")
    _canonical(src, "shipped", audience="user", body="see the `internal-only` skill for more")
    leaks = _find_leaked_refs(load_canonical([src]))
    assert len(leaks) == 1 and "internal-only" in leaks[0]


def test_maintainer_skill_may_reference_maintainer_skill(tmp_path):
    """Maintainer skills never ship, so cross-refs between them are fine."""
    src = tmp_path / "skills"
    _canonical(src, "internal-only", audience="maintainer")
    _canonical(src, "other", audience="maintainer", body="see the `internal-only` skill")
    assert _find_leaked_refs(load_canonical([src])) == []


def test_user_skill_referencing_user_skill_is_fine(tmp_path):
    src = tmp_path / "skills"
    _canonical(src, "sibling", audience="user")
    _canonical(src, "shipped", audience="user", body="see the `sibling` skill")
    assert _find_leaked_refs(load_canonical([src])) == []


# ------------------------------------------------------------ _find_bad_extras


def _pyproject(root: Path, extras: list[str]) -> None:
    groups = "\n".join(f'{e} = ["x"]' for e in extras)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "t"\n\n[project.optional-dependencies]\n{groups}\n',
        encoding="utf-8",
    )


def test_real_extra_passes(tmp_path):
    src = tmp_path / "skills"
    _canonical(src, "s", audience="user", extras=["julia"])
    _pyproject(tmp_path, ["julia", "tvb"])
    assert _find_bad_extras(load_canonical([src]), tmp_path) == []


def test_nonexistent_extra_is_reported(tmp_path):
    """The real bug: skills declared requires_extras: ["jax"], but jax is core."""
    src = tmp_path / "skills"
    _canonical(src, "s", audience="user", extras=["jax"])
    _pyproject(tmp_path, ["julia", "tvb"])
    bad = _find_bad_extras(load_canonical([src]), tmp_path)
    assert len(bad) == 1 and "jax" in bad[0]


def test_missing_pyproject_skips_the_check(tmp_path):
    src = tmp_path / "skills"
    _canonical(src, "s", audience="user", extras=["anything"])
    assert _find_bad_extras(load_canonical([src]), tmp_path) == []


# --------------------------------------------------------------- _sync_check


def test_sync_check_passes_when_clean(synced):
    _sync_check(*synced)


def test_sync_check_fails_on_orphan(synced):
    skills, claude_dir, copilot_dir, agents_md, root = synced
    stray = claude_dir / "personal-thing"
    stray.mkdir()
    (stray / "SKILL.md").write_text("---\nname: personal-thing\n---\n\nmine\n")

    with pytest.raises(typer.Exit) as exc:
        _sync_check(skills, claude_dir, copilot_dir, agents_md, root)
    assert exc.value.exit_code == 1


def test_sync_check_fails_on_drift(synced):
    skills, claude_dir, copilot_dir, agents_md, root = synced
    edited = claude_dir / "real-skill" / "SKILL.md"
    edited.write_text(edited.read_text() + "\nhand-edited\n", encoding="utf-8")

    with pytest.raises(typer.Exit) as exc:
        _sync_check(skills, claude_dir, copilot_dir, agents_md, root)
    assert exc.value.exit_code == 1
