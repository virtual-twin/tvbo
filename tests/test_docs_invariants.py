"""The page-level rules that hold whether or not anyone runs `make docs-lint`.

Both checks catch a defect that renders successfully. An unresolved citation key reaches the published site as literal **[@Key]** because citeproc exits 0 on it, and a page that declares `native: true` while plotting by hand looks perfect — it just no longer says what it does. Neither is visible in a build log, so they belong in the suite rather than in a lint step someone remembers to run.
"""

import subprocess
import sys
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"


@pytest.mark.parametrize(
    "script",
    ["check_citations.py", "check_native_pages.py"],
    ids=["every citation resolves", "native pages draw nothing by hand"],
)
def test_docs_invariant_holds(script):
    done = subprocess.run([sys.executable, f"scripts/{script}"], cwd=DOCS, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr or done.stdout
