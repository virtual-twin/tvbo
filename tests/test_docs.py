"""Test that all documentation notebooks execute without errors.

This test discovers all .qmd files with Python code cells in docs/, converts them to notebooks, and executes them to ensure documentation examples remain functional.

Run with: pytest tests/test_docs.py -v
Run single doc: pytest tests/test_docs.py -k "Network" -v
"""

import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
DOCS_DIR = REPO_ROOT / "docs"
KERNEL_NAME = "tvbo-docs"


def jupyter_executable() -> str:
    """Path to the ``jupyter`` shipped with the interpreter running the tests.

    Falls back to the bare name so the failure, if any, is a plain "not found" rather than a silent execution against some other environment's Jupyter.
    """
    candidate = Path(sys.executable).with_name("jupyter")
    return str(candidate) if candidate.exists() else "jupyter"


@pytest.fixture(scope="session")
def docs_kernel(tmp_path_factory):
    """Register a kernelspec bound to ``sys.executable`` and return its search root.

    The docs declare ``jupyter: python3``, and which interpreter that name resolves to is ambient: Jupyter searches the user, environment and system kernel directories, and the ``python3`` spec ipykernel installs into a virtualenv launches a bare ``python`` taken from ``PATH``. On a machine with several project virtualenvs that lands wherever ``PATH`` happens to point -- typically another project's environment holding a stale released ``tvbo`` -- so the docs never exercise this checkout. Pinning the kernel to the absolute interpreter running pytest makes the notebooks execute against the code under test.
    """
    root = tmp_path_factory.mktemp("jupyter-kernels")
    spec_dir = root / "kernels" / KERNEL_NAME
    spec_dir.mkdir(parents=True)
    spec = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": f"tvbo docs ({sys.executable})",
        "language": "python",
    }
    (spec_dir / "kernel.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    return root


def get_all_qmd_files():
    """Every ``.qmd`` SOURCE page under ``docs/``.

    Quarto keeps its build and cache output beside the sources, in directories it marks with a leading underscore or dot (``_site``, ``_freeze``, ``_archive``, ``.jupyter_cache``). Those hold rendered copies of the very pages collected here, so globbing them runs each doc twice — once as written and once as it stood at the last render. The stale copy then fails on an API the source has already migrated off, blaming a page whose committed text is correct.
    """
    pattern = str(DOCS_DIR / "**" / "*.qmd")
    return [
        path
        for path in glob.glob(pattern, recursive=True)
        if not any(part.startswith(("_", ".")) for part in Path(path).relative_to(DOCS_DIR).parts[:-1])
    ]


def has_python_cells(qmd_path: str) -> bool:
    """Check if a .qmd file contains Python code cells."""
    with open(qmd_path, encoding="utf-8") as f:
        content = f.read()
    return "```{python}" in content


def is_eval_false(cell: dict) -> bool:
    """Whether a notebook cell carries Quarto's ``#| eval: false`` directive.

    Quarto renders such cells without running them, so they are free to show illustrative code referencing names the page never defines. ``jupyter execute`` knows nothing of the directive and would run them anyway, turning a deliberately non-executable snippet into a spurious NameError.
    """
    if cell.get("cell_type") != "code":
        return False
    for line in cell.get("source", []):
        if not line.lstrip().startswith("#"):
            break
        if re.match(r"\s*#\s*\|\s*eval\s*:\s*false\b", line, re.IGNORECASE):
            return True
    return False


def get_doc_name(path: str) -> str:
    """Extract a readable name from the qmd path."""
    rel_path = Path(path).relative_to(DOCS_DIR)
    return str(rel_path.with_suffix(""))


qmd_files = [f for f in get_all_qmd_files() if has_python_cells(f)]

SLOW_DIRS = {"Interoperability/tvboptim", "_tvboptim"}
"""Substrings of a doc's path, relative to docs/, marking it slow (needs --run-slow).

The heavy pages are the tvboptim optimization workflows, which moved from
Interoperability/tvboptim/ into the Fitting/ topic, hence the `_tvboptim` filename
match: keying on "Fitting" would also catch neighbours like Fitting/ModelFitting.qmd,
which is not slow.
"""

EXCLUDED_DOCS = {"Replication/Koller2024/Run_Koller2024"}
"""Docs skipped entirely. Koller2024 exceeds the CI timeout until its replication
converges; re-add it once the runtime is bounded."""

test_params = []
for path in qmd_files:
    doc_name = get_doc_name(path)
    if doc_name in EXCLUDED_DOCS:
        continue
    marks = [pytest.mark.slow] if any(d in doc_name for d in SLOW_DIRS) else []
    test_params.append(pytest.param(path, doc_name, marks=marks))


CONVERT_TIMEOUT_S = 120
"""A `quarto convert` is a format translation and never legitimately takes minutes."""

EXECUTE_TIMEOUT_S = 600
SLOW_EXECUTE_TIMEOUT_S = 3600
"""Wall-clock ceilings on executing one notebook. Generous, because a doc that runs a simulation is allowed to be slow; finite, because one that hangs must not be allowed to stall the run."""


def timeout_for(doc_name: str) -> int:
    """The execution ceiling for one doc: longer where the doc is marked slow."""
    return SLOW_EXECUTE_TIMEOUT_S if any(d in doc_name for d in SLOW_DIRS) else EXECUTE_TIMEOUT_S


def _run(cmd, doc_name: str, timeout: int, **kwargs):
    """Run *cmd*, failing the test rather than the run when it outlives *timeout*.

    Without a ceiling here a single wedged notebook stops the whole suite indefinitely, and it does so invisibly: the main thread blocks inside `subprocess.run`, so pytest-timeout cannot fire in either of its modes and the run reports no test, no failure and no name -- just silence at whatever percentage it had reached. Diagnosing that costs far more than the ceiling does.
    """
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"{doc_name}: `{Path(cmd[0]).name}` did not finish within {timeout} s and was killed. "
            "Either the doc has become slower than its ceiling, or it is wedged -- a notebook that "
            "runs a simulation can park its whole thread pool and never return."
        )


@pytest.mark.docs
@pytest.mark.parametrize("qmd_path,doc_name", test_params, ids=lambda x: x if isinstance(x, str) else Path(x).stem)
def test_doc_executes(qmd_path, doc_name, docs_kernel):
    """Test that a documentation notebook executes without errors."""
    qmd_path = Path(qmd_path)
    doc_dir = qmd_path.parent  # Original doc directory for relative path resolution

    # Beside the doc for relative paths; pid-tagged so concurrent runs don't delete each other's.
    ipynb_path = doc_dir / f"{qmd_path.stem}.{os.getpid()}.ipynb"
    try:
        result = _run(
            ["quarto", "convert", str(qmd_path), "--output", str(ipynb_path)], doc_name, CONVERT_TIMEOUT_S, cwd=str(doc_dir)
        )
        assert result.returncode == 0, f"quarto convert failed: {result.stderr}"
        assert ipynb_path.exists(), f"Notebook not created: {ipynb_path}"

        # Ensure _output directory exists (some docs write files there)
        (doc_dir / "_output").mkdir(exist_ok=True)

        # Kernels do not reliably inherit PYTHONPATH, so put the doc dir on sys.path in-band.
        with open(ipynb_path, encoding="utf-8") as f:
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
            ],
        }
        nb["cells"] = [setup_cell] + [c for c in nb["cells"] if not is_eval_false(c)]
        with open(ipynb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f)

        # Execute the notebook
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"  # Non-interactive matplotlib backend
        # Keep JAX memory usage bounded in CI notebooks (notably tvboptim docs).
        env.setdefault("JAX_PLATFORMS", "cpu")
        env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        env["JUPYTER_PATH"] = str(docs_kernel)
        # Console scripts the docs shell out to (meson, ninja, jnml) live beside the interpreter.
        env["PATH"] = os.pathsep.join([str(Path(sys.executable).parent), env.get("PATH", "")])

        result = _run(
            [jupyter_executable(), "execute", "--kernel_name", KERNEL_NAME, str(ipynb_path)],
            doc_name,
            timeout_for(doc_name),
            cwd=str(doc_dir),  # Run from doc's directory for correct relative paths
            env=env,
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
