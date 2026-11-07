"""The doc harness must report why a page failed, not merely that it did.

``jupyter execute`` discards the executed notebook when a cell raises, so the only thing reaching the log was the failing cell's own traceback — and the cause is usually in an earlier cell, which printed a warning and carried on. Two documentation pages failed for a week with no trace of the reason. Execution now continues past a failing cell so the notebook is written back, which moves the pass/fail decision off the exit status and onto the errors the notebook records; these pin both halves of that trade.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_docs import _cell_error, _cell_streams, jupyter_executable

KERNEL = "tvbo-harness-probe"


def _notebook(path: Path, sources: list[str]) -> Path:
    cells = [{"cell_type": "code", "metadata": {}, "source": [s], "outputs": [], "execution_count": None} for s in sources]
    path.write_text(
        json.dumps(
            {
                "cells": cells,
                "metadata": {"kernelspec": {"name": KERNEL, "display_name": KERNEL, "language": "python"}},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="module")
def kernel_root(tmp_path_factory):
    """A kernelspec bound to the interpreter running the tests, so the probe notebooks execute against it."""
    root = tmp_path_factory.mktemp("harness-kernels")
    spec = root / "kernels" / KERNEL
    spec.mkdir(parents=True)
    argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    (spec / "kernel.json").write_text(
        json.dumps({"argv": argv, "display_name": KERNEL, "language": "python"}), encoding="utf-8"
    )
    return root


def _execute(path: Path, kernel_root: Path):
    env = {**dict(__import__("os").environ), "JUPYTER_PATH": str(kernel_root)}
    return subprocess.run(
        [jupyter_executable(), "execute", "--kernel_name", KERNEL, "--inplace", "--allow-errors", str(path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )


def test_a_failing_page_reports_its_error_and_what_ran_before_it(tmp_path, kernel_root):
    """The whole point: the cause is in the earlier cell, and it must survive into the failure message."""
    nb = _notebook(tmp_path / "probe.ipynb", ["print('the figure did not render')", "raise RuntimeError('boom')"])
    _execute(nb, kernel_root)
    assert _cell_error(nb) == "RuntimeError: boom"
    assert "the figure did not render" in _cell_streams(nb)


def test_a_page_that_ran_records_no_error(tmp_path, kernel_root):
    """`--allow-errors` makes the exit status always zero, so a clean run must be distinguishable by this alone."""
    nb = _notebook(tmp_path / "clean.ipynb", ["print('fine')"])
    result = _execute(nb, kernel_root)
    assert result.returncode == 0
    assert _cell_error(nb) == ""
    assert "fine" in _cell_streams(nb)


def test_an_unwritten_notebook_is_not_read_as_success(tmp_path):
    """A notebook the run never wrote yields no error string, which is why the exit status is still checked beside it."""
    assert _cell_error(tmp_path / "absent.ipynb") == ""
    assert _cell_streams(tmp_path / "absent.ipynb") == ""
