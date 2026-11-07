"""Every file the build hook generates must be declared as a wheel artifact.

The generated datamodel is gitignored, and hatchling selects files by VCS, so anything the hook writes is dropped from the wheel unless ``[tool.hatch.build] artifacts`` names it. ``dialect_tables.py`` was written by the hook and never declared, so every wheel built from this repo omitted it and ``import tvbo`` raised ``ModuleNotFoundError`` on the published container — where it went unnoticed because CI installs the checkout over the wheel. The two lists are kept in step here rather than by memory.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def _generated_paths() -> set[str]:
    """The files ``hatch_build.py`` writes, read off its own ``_write(out_dir / "...")`` calls."""
    source = (REPO_ROOT / "hatch_build.py").read_text(encoding="utf-8")
    return {f"tvbo/datamodel/{name}" for name in re.findall(r'out_dir\s*/\s*"([^"]+)"', source)}


def _declared_artifacts() -> set[str]:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return set(config["tool"]["hatch"]["build"]["artifacts"])


def test_the_hook_writes_nothing_the_wheel_would_drop():
    generated = _generated_paths()
    assert generated, "no generated files found — the parse of hatch_build.py stopped matching it"
    missing = sorted(generated - _declared_artifacts())
    assert not missing, f"generated but not declared in [tool.hatch.build] artifacts, so the wheel omits them: {missing}"
