"""
Test that all documentation notebooks execute without errors.

This test discovers all .qmd files with Python code cells in docs/,
converts them to notebooks, and executes them to ensure documentation
examples remain functional.

Run with: pytest tests/test_docs.py -v
Run single doc: pytest tests/test_docs.py -k "Network" -v
"""

import os
import re
import glob
import subprocess
import tempfile
import shutil
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DOCS_DIR = REPO_ROOT / "docs"


def get_all_qmd_files():
    """Get all .qmd files from docs/ directory."""
    pattern = str(DOCS_DIR / "**" / "*.qmd")
    return glob.glob(pattern, recursive=True)


def has_python_cells(qmd_path: str) -> bool:
    """Check if a .qmd file contains Python code cells."""
    with open(qmd_path, "r", encoding="utf-8") as f:
        content = f.read()
    return "```{python}" in content


def get_doc_name(path: str) -> str:
    """Extract a readable name from the qmd path."""
    rel_path = Path(path).relative_to(DOCS_DIR)
    return str(rel_path.with_suffix(""))


# Discover all testable qmd files
qmd_files = [f for f in get_all_qmd_files() if has_python_cells(f)]
test_params = [(path, get_doc_name(path)) for path in qmd_files]


@pytest.mark.docs
@pytest.mark.parametrize(
    "qmd_path,doc_name",
    test_params,
    ids=lambda x: x if isinstance(x, str) else Path(x).stem
)
def test_doc_executes(qmd_path, doc_name):
    """Test that a documentation notebook executes without errors."""
    qmd_path = Path(qmd_path)

    # Create temp directory for conversion
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy qmd to temp dir to avoid polluting docs/
        tmp_qmd = Path(tmpdir) / qmd_path.name
        shutil.copy(qmd_path, tmp_qmd)

        # Convert qmd to ipynb
        ipynb_path = tmp_qmd.with_suffix(".ipynb")
        result = subprocess.run(
            ["quarto", "convert", str(tmp_qmd), "--output", str(ipynb_path)],
            capture_output=True,
            text=True,
            cwd=tmpdir
        )
        assert result.returncode == 0, f"quarto convert failed: {result.stderr}"
        assert ipynb_path.exists(), f"Notebook not created: {ipynb_path}"

        # Execute the notebook
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"  # Non-interactive matplotlib backend

        result = subprocess.run(
            ["jupyter", "execute", str(ipynb_path)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),  # Run from repo root for correct imports
            env=env
        )

        # Check for execution errors
        if result.returncode != 0:
            # Try to extract meaningful error message
            error_msg = result.stderr.strip().split("\n")[-1] if result.stderr else "Unknown error"
            pytest.fail(f"Notebook execution failed: {error_msg}\n\nFull stderr:\n{result.stderr}")


# Allow running docs tests separately
def pytest_configure(config):
    config.addinivalue_line("markers", "docs: mark test as documentation test")
