"""
Execute all Jupyter notebooks under the project `docs/` folder, excluding any
paths that include 'deprecated' (case-insensitive), to ensure they run without
errors. Notebooks can opt out of CI execution by adding a cell tag 'skip-ci'.

Environment variables:
- TVBO_NOTEBOOK_TIMEOUT: per-notebook timeout in seconds (default: 600)
- TVBO_NOTEBOOK_PATTERN: substring to filter which notebooks to run (optional)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List

import nbformat
from nbclient import NotebookClient
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"


def _is_deprecated(path: Path) -> bool:
    # exclude any path component matching 'deprecated' (case-insensitive)
    return any("deprecated" in part.lower() for part in path.parts)


def _discover_notebooks(base: Path) -> List[Path]:
    notebooks: List[Path] = []
    for p in base.rglob("*.ipynb"):
        if _is_deprecated(p):
            continue
        notebooks.append(p)
    return sorted(notebooks)


def _has_skip_ci_tag(nb) -> bool:
    # Check notebook-level metadata tags
    tags = set()
    md = getattr(nb, "metadata", {}) or {}
    # nbformat uses attribute access for metadata fields
    if hasattr(md, "get"):
        # type: ignore[attr-defined]
        tags.update(md.get("tags", []) or [])

    # Check cell-level tags
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        cell_tags = (cell.get("metadata", {}) or {}).get("tags", []) or []
        tags.update(cell_tags)
    normalized = {str(t).strip().lower() for t in tags}
    return any(t in {"skip-ci", "skip_ci", "no-ci"} for t in normalized)


def _load_notebook(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return nbformat.read(f, as_version=4)


ALL_NOTEBOOKS = _discover_notebooks(DOCS_DIR)


def _apply_pattern_filter(paths: Iterable[Path]) -> List[Path]:
    pat = os.environ.get("TVBO_NOTEBOOK_PATTERN")
    if not pat:
        return list(paths)
    pat_lower = pat.lower()
    return [p for p in paths if pat_lower in str(p).lower()]


PARAM_NOTEBOOKS = _apply_pattern_filter(ALL_NOTEBOOKS)


@pytest.mark.parametrize("nb_path", PARAM_NOTEBOOKS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_docs_notebook_executes(nb_path: Path, tmp_path: Path):
    nb = _load_notebook(nb_path)

    if _has_skip_ci_tag(nb):
        pytest.skip("Notebook opted out via skip-ci tag")

    # Execute in the notebook's directory to preserve relative paths
    workdir = nb_path.parent

    timeout = int(os.environ.get("TVBO_NOTEBOOK_TIMEOUT", "600"))

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(workdir)}},
        allow_errors=False,
    )

    # Some notebooks may rely on repository-local imports; ensure repo root on sys.path
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    client.execute()
