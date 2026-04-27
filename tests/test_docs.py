"""
Test that all documentation notebooks execute without errors.

This test discovers all .qmd files with Python code cells in docs/,
converts them to notebooks, and executes them to ensure documentation
examples remain functional.

Run with: pytest tests/test_docs.py -v
Run single doc: pytest tests/test_docs.py -k "Network" -v
"""

import json
import os
import glob
import subprocess
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

# Directories whose doc tests are slow and excluded by default (use --run-slow)
SLOW_DIRS = {"Interoperability/tvboptim"}

test_params = []
for path in qmd_files:
    doc_name = get_doc_name(path)
    marks = [pytest.mark.slow] if any(d in doc_name for d in SLOW_DIRS) else []
    test_params.append(pytest.param(path, doc_name, marks=marks))


@pytest.mark.docs
@pytest.mark.parametrize(
    "qmd_path,doc_name",
    test_params,
    ids=lambda x: x if isinstance(x, str) else Path(x).stem,
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
            cwd=str(doc_dir),
        )
        assert result.returncode == 0, f"quarto convert failed: {result.stderr}"
        assert ipynb_path.exists(), f"Notebook not created: {ipynb_path}"

        # Ensure _output directory exists (some docs write files there)
        (doc_dir / "_output").mkdir(exist_ok=True)

        # Inject a setup cell so the kernel can find local modules (e.g. Jansen1995_plot)
        # PYTHONPATH isn't reliably inherited by Jupyter kernels.
        with open(ipynb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        setup_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"tags": ["setup"]},
            "outputs": [],
            "source": [
                "import sys, os\n",
                f"sys.path.insert(0, {str(doc_dir)!r})\n",
                f"os.chdir({str(doc_dir)!r})\n",
                "try:\n",
                "    import pyrates.backend.fortran.fortran_backend as _fb\n",
                "    import subprocess as _sp, glob as _glob\n",
                "    _orig_generate = _fb.FortranBackend.generate_func\n",
                "    def _patched_generate(self, *args, **kwargs):\n",
                "        _cwd = os.getcwd()\n",
                "        if _cwd not in sys.path:\n",
                "            sys.path.insert(0, _cwd)\n",
                "        # Find .so after orig runs to diagnose where it landed\n",
                "        result = _orig_generate(self, *args, **kwargs)\n",
                "        return result\n",
                "    _fb.FortranBackend.generate_func = _patched_generate\n",
                "    # Patch f2py call to force cwd=doc_dir\n",
                "    _orig_subprocess_run = _fb.subprocess.run\n",
                f"    _forced_cwd = {str(doc_dir)!r}\n",
                "    def _patched_run(cmd, *a, **kw):\n",
                "        if 'f2py' in str(cmd):\n",
                "            kw['cwd'] = _forced_cwd\n",
                "            kw.pop('stdout', None); kw.pop('stderr', None)  # show output\n",
                "        return _orig_subprocess_run(cmd, *a, **kw)\n",
                "    _fb.subprocess.run = _patched_run\n",
                "except ImportError:\n",
                "    pass\n",
            ],
        }
        nb["cells"].insert(0, setup_cell)
        diagnostic_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"tags": ["setup"]},
            "outputs": [],
            "source": [
                "import glob, sys, os\n",
                "print('CWD:', os.getcwd())\n",
                "print('sys.path:', sys.path)\n",
                "print('.so files found:', glob.glob('**/*.so', recursive=True))\n",
                "print('.so in /tmp:', glob.glob('/tmp/**/*.so', recursive=True))\n",
            ],
        }
        nb["cells"].insert(1, diagnostic_cell)
        with open(ipynb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f)

        # Execute the notebook
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"  # Non-interactive matplotlib backend

        result = subprocess.run(
            ["jupyter", "execute", str(ipynb_path)],
            capture_output=True,
            text=True,
            cwd=str(doc_dir),  # Run from doc's directory for correct relative paths
            env=env,
        )

        # Check for execution errors
        if result.returncode != 0:
            # Try to extract meaningful error message
            error_msg = (
                result.stderr.strip().split("\n")[-1]
                if result.stderr
                else "Unknown error"
            )
            pytest.fail(
                f"Notebook execution failed: {error_msg}\n\nFull stderr:\n{result.stderr}"
            )
    finally:
        # Clean up generated notebook
        if ipynb_path.exists():
            ipynb_path.unlink()
