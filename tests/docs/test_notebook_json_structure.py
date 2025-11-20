"""
Lightweight smoke tests for docs notebooks.

These tests only parse the notebooks with nbformat to ensure they are
well-formed JSON and conform to nbformat v4. They do not execute any cells,
so they are safe to run in CI on main.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


NOTEBOOK_PATHS = [
    REPO_ROOT / "docs" / "TestJax.ipynb",
    REPO_ROOT / "docs" / "Usage" / "2DSweep_RWW.ipynb",
    REPO_ROOT / "docs" / "Usage" / "NavigatingTheOntology.ipynb",
    REPO_ROOT / "docs" / "Use-Cases" / "LB_Coupling-Noise-Exploration.ipynb",
]


@pytest.mark.parametrize("nb_path", NOTEBOOK_PATHS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_notebook_parses_and_has_cells(nb_path: Path):
    if not nb_path.exists():
        pytest.skip(f"Notebook missing: {nb_path}")

    nb = nbformat.read(nb_path, as_version=4)

    # Basic structural assertions
    assert nb.nbformat == 4, "Notebook must be nbformat v4"
    assert isinstance(nb.cells, list) and len(nb.cells) >= 1, "Notebook must contain at least one cell"
