"""TVBO skill system.

Canonical sources live in:

* ``tvbo/skills/canonical/<name>/SKILL.md``  — user skills, ship in the wheel
* ``skills/<name>/SKILL.md`` (repo root)     — maintainer skills, repo-only

Both are rendered to per-tool adapter formats by :mod:`tvbo.skills._render`.
"""
from __future__ import annotations

from ._render import (
    CANONICAL_PACKAGE_DIR,
    CANONICAL_REPO_DIR,
    Skill,
    is_asset_noise,
    is_managed_file,
    load_canonical,
    render_agents_md,
    render_claude_code,
    render_copilot,
    render_cursor,
    render_prompt,
)

__all__ = [
    "CANONICAL_PACKAGE_DIR",
    "CANONICAL_REPO_DIR",
    "Skill",
    "is_asset_noise",
    "is_managed_file",
    "load_canonical",
    "render_agents_md",
    "render_claude_code",
    "render_copilot",
    "render_cursor",
    "render_prompt",
]