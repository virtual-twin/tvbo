"""Tests for per-skill ``assets/`` handling in the skills pipeline.

Locks in the behaviour that lets a canonical user skill ship an ``assets/`` directory: ``parse_skill`` discovers it, ``render_claude_code`` mirrors it next
to ``SKILL.md`` (pruning stale files, skipping bytecode), the flat targets inline the reference chapters they cannot mirror, and the CLI ``install`` /
``uninstall`` round-trip carries then removes it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from typer.testing import CliRunner

from tvbo.cli import app
from tvbo.cli.skills import _find_dead_asset_refs
from tvbo.skills import CANONICAL_PACKAGE_DIR, load_canonical
from tvbo.skills._render import (
    asset_refs,
    flat_body,
    parse_skill,
    render_claude_code,
    render_copilot,
)

runner = CliRunner()


def _make_skill(root: Path, name: str, *, assets: dict[str, str] | None = None) -> Path:
    """Create a minimal canonical skill dir; return its ``SKILL.md`` path."""
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test skill\nmetadata:\n  audience: user\n---\n\n# {name}\n\nbody\n",
        encoding="utf-8",
    )
    for rel, content in (assets or {}).items():
        p = d / "assets" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return d / "SKILL.md"


# --------------------------------------------------------------- parse_skill


def test_parse_skill_detects_assets_dir(tmp_path):
    md = _make_skill(tmp_path, "with-assets", assets={"tmpl.txt": "x"})
    assert parse_skill(md).assets_dir == md.parent / "assets"


def test_parse_skill_without_assets_is_none(tmp_path):
    md = _make_skill(tmp_path, "no-assets")
    assert parse_skill(md).assets_dir is None


# ---------------------------------------------------------- render_claude_code


def test_render_mirrors_assets_tree(tmp_path):
    md = _make_skill(tmp_path / "src", "s", assets={"a.tmpl": "A", "sub/b.py": "print('b')"})
    dest = tmp_path / "dest"
    out = render_claude_code(parse_skill(md), dest)
    assert out == dest / "s" / "SKILL.md"
    assert (dest / "s" / "assets" / "a.tmpl").read_text() == "A"
    assert (dest / "s" / "assets" / "sub" / "b.py").read_text() == "print('b')"


def test_render_without_assets_writes_no_assets_dir(tmp_path):
    md = _make_skill(tmp_path / "src", "s")
    dest = tmp_path / "dest"
    render_claude_code(parse_skill(md), dest)
    assert (dest / "s" / "SKILL.md").exists()
    assert not (dest / "s" / "assets").exists()


def test_render_prunes_upstream_deleted_asset(tmp_path):
    src = tmp_path / "src"
    md = _make_skill(src, "s", assets={"keep.txt": "k", "drop.txt": "d"})
    dest = tmp_path / "dest"
    render_claude_code(parse_skill(md), dest)
    assert (dest / "s" / "assets" / "drop.txt").exists()

    (md.parent / "assets" / "drop.txt").unlink()  # remove upstream, re-render
    render_claude_code(parse_skill(md), dest)
    assert (dest / "s" / "assets" / "keep.txt").exists()
    assert not (dest / "s" / "assets" / "drop.txt").exists()


def test_render_removes_assets_dir_when_source_loses_all(tmp_path):
    src = tmp_path / "src"
    md = _make_skill(src, "s", assets={"a.txt": "a"})
    dest = tmp_path / "dest"
    render_claude_code(parse_skill(md), dest)
    assert (dest / "s" / "assets").is_dir()

    shutil.rmtree(md.parent / "assets")  # skill no longer ships assets
    render_claude_code(parse_skill(md), dest)
    assert not (dest / "s" / "assets").exists()


def test_render_ignores_bytecode_and_os_noise(tmp_path):
    md = _make_skill(
        tmp_path / "src",
        "s",
        assets={
            "m.py": "x = 1",
            "__pycache__/m.cpython-313.pyc": "junk",
            ".DS_Store": "junk",
        },
    )
    dest = tmp_path / "dest"
    render_claude_code(parse_skill(md), dest)
    assert (dest / "s" / "assets" / "m.py").exists()
    assert not (dest / "s" / "assets" / "__pycache__").exists()
    assert not (dest / "s" / "assets" / ".DS_Store").exists()


# ------------------------------------------ flat targets inline md chapters


def _skill_deferring_to(root: Path, name: str, ref: str, chapter: str) -> Path:
    md = _make_skill(root, name, assets={ref: chapter})
    md.write_text(md.read_text().replace("body", f"body — see `assets/{ref}`"), "utf-8")
    return md


def test_asset_refs_are_deduped_in_mention_order():
    body = "see `assets/b.md`, then `assets/a.md`, then `assets/b.md` and `assets/skeleton/`"
    assert asset_refs(body) == ["b.md", "a.md", "skeleton"]


def test_flat_body_inlines_referenced_chapter(tmp_path):
    md = _skill_deferring_to(tmp_path, "s", "ref.md", "# Ref\n\nthe deferred detail\n")
    out = flat_body(parse_skill(md))
    assert "the deferred detail" in out
    assert "<!-- inlined from assets/ref.md -->" in out


def test_flat_body_leaves_non_markdown_assets_alone(tmp_path):
    md = _skill_deferring_to(tmp_path, "s", "tmpl.yaml", "a: 1\n")
    assert flat_body(parse_skill(md)) == parse_skill(md).body


def test_claude_code_keeps_the_pointer_and_mirrors_the_chapter(tmp_path):
    md = _skill_deferring_to(tmp_path / "src", "s", "ref.md", "# Ref\n\ndetail\n")
    dest = tmp_path / "dest"
    render_claude_code(parse_skill(md), dest)
    rendered = (dest / "s" / "SKILL.md").read_text()
    assert "`assets/ref.md`" in rendered
    assert "detail" not in rendered  # deferred, not inlined
    assert (dest / "s" / "assets" / "ref.md").read_text() == "# Ref\n\ndetail\n"


def test_copilot_inlines_because_it_has_no_assets_dir(tmp_path):
    md = _skill_deferring_to(tmp_path / "src", "s", "ref.md", "# Ref\n\ndetail\n")
    dest = tmp_path / "dest"
    out = render_copilot(parse_skill(md), dest)
    assert "detail" in out.read_text()
    assert not (dest / "assets").exists()


def test_dead_asset_reference_is_linted(tmp_path):
    md = _make_skill(tmp_path, "s", assets={"real.md": "x"})
    md.write_text(md.read_text().replace("body", "see `assets/gone.md`"), "utf-8")
    dead = _find_dead_asset_refs([parse_skill(md)])
    assert len(dead) == 1 and "gone.md" in dead[0]


def test_shipped_skills_have_no_dead_asset_references():
    assert _find_dead_asset_refs(load_canonical([CANONICAL_PACKAGE_DIR])) == []


# ------------------------------------------------- CLI install / uninstall


def _shipped_user_skill_with_assets() -> str:
    """Name of a bundled user skill that ships an ``assets/`` dir (or "")."""
    for s in load_canonical([CANONICAL_PACKAGE_DIR]):
        if s.audience in {"user", "both"} and s.assets_dir is not None:
            return s.name
    return ""


def test_install_then_uninstall_carries_and_removes_assets(tmp_path):
    name = _shipped_user_skill_with_assets()
    assert name, "expected at least one shipped user skill with an assets/ dir"

    res = runner.invoke(
        app,
        ["skills", "install", "--target", "claude-code", "--skill", name, "--out", str(tmp_path)],
    )
    assert res.exit_code == 0, res.stdout
    skill_dir = tmp_path / f"tvbo-{name}"
    assert "managed-by: tvbo" in (skill_dir / "SKILL.md").read_text()
    assert (skill_dir / "assets").is_dir()
    assert any((skill_dir / "assets").rglob("*")), "installed assets/ is empty"

    res = runner.invoke(app, ["skills", "uninstall", "--target", "claude-code", "--out", str(tmp_path)])
    assert res.exit_code == 0, res.stdout
    assert not skill_dir.exists()  # SKILL.md + assets/ + dir all gone
