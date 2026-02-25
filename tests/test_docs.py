"""
Test that all documentation notebooks execute without errors.

This test discovers all .qmd files with Python code cells in docs/,
converts them to notebooks, and executes them to ensure documentation
examples remain functional.

Docs with ``execute: eval: false`` in their YAML frontmatter are excluded
from collection (Quarto itself would not execute them).

Run with: pytest tests/test_docs.py -v
Run single doc: pytest tests/test_docs.py -k "Network" -v
"""

import os
import re
import glob
import subprocess
import pytest
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DOCS_DIR = REPO_ROOT / "docs"


def parse_frontmatter(qmd_path: str) -> dict:
    """Parse YAML frontmatter from a .qmd file."""
    with open(qmd_path, "r", encoding="utf-8") as f:
        content = f.read()
    match = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
    if not match:
        return {}
    return yaml.safe_load(match.group(1)) or {}


def is_eval_disabled(qmd_path: str) -> bool:
    """Return True if the doc has execute.eval: false in frontmatter."""
    fm = parse_frontmatter(qmd_path)
    execute = fm.get("execute", {})
    return isinstance(execute, dict) and execute.get("eval") is False


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


# Discover all testable qmd files (exclude eval:false docs at collection time)
qmd_files = [
    f for f in get_all_qmd_files()
    if has_python_cells(f) and not is_eval_disabled(f)
]
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
    doc_dir = qmd_path.parent  # Original doc directory for relative path resolution

    # Convert qmd to ipynb in the doc's own directory (clean up after)
    ipynb_path = qmd_path.with_suffix(".ipynb")
    try:
        result = subprocess.run(
            ["quarto", "convert", str(qmd_path), "--output", str(ipynb_path)],
            capture_output=True,
            text=True,
            cwd=str(doc_dir)
        )
        assert result.returncode == 0, f"quarto convert failed: {result.stderr}"
        assert ipynb_path.exists(), f"Notebook not created: {ipynb_path}"

        # Ensure _output directory exists (some docs write files there)
        (doc_dir / "_output").mkdir(exist_ok=True)

        # Execute the notebook with cwd = doc's directory
        # so relative paths (yaml/, _output/, ../../../database/) resolve correctly
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"  # Non-interactive matplotlib backend
        # Ensure repo root is on PYTHONPATH for imports
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

        result = subprocess.run(
            ["jupyter", "execute", str(ipynb_path)],
            capture_output=True,
            text=True,
            cwd=str(doc_dir),  # Run from doc's directory for correct relative paths
            env=env
        )

        # Check for execution errors
        if result.returncode != 0:
            # Try to extract meaningful error message
            error_msg = result.stderr.strip().split("\n")[-1] if result.stderr else "Unknown error"
            pytest.fail(f"Notebook execution failed: {error_msg}\n\nFull stderr:\n{result.stderr}")
    finally:
        # Clean up generated notebook
        if ipynb_path.exists():
            ipynb_path.unlink()


# Allow running docs tests separately
def pytest_configure(config):
    config.addinivalue_line("markers", "docs: mark test as documentation test")
