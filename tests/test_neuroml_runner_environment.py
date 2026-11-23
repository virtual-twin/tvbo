"""A NeuroML run leaves the process environment as it found it.

pyNeuroML's NEURON runner prepends every `sys.path` entry to `os.environ["PYTHONPATH"]` before it starts jNeuroML, and never undoes it, so every later subprocess of the session (a figure's `plot.py`, a kit's `tvbo run`) imported packages from the order that rewrite left: on a venv with an editable install, a different tvbo than the session's (G297). The adapter restores every variable it or the runner rewrites.
"""

from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("pyneuroml")

from tvbo.adapters.neuroml import NeuroMLAdapter


@pytest.mark.backend_neuroml
@pytest.mark.parametrize("backend", ["jneuroml", "neuron"])
def test_a_runner_s_rewrite_of_pythonpath_does_not_outlive_the_run(backend, monkeypatch, tmp_path):
    """A runner that rewrites ``PYTHONPATH`` as pyNeuroML's NEURON runner does leaves it as it was once the run returns."""
    from pyneuroml import pynml

    monkeypatch.setenv("PYTHONPATH", "/the/session/path")
    monkeypatch.setenv("NEURON_HOME", str(tmp_path))

    def runner(*args, **kwargs):
        os.environ["PYTHONPATH"] = os.pathsep.join([*sys.path, os.environ["PYTHONPATH"]])
        return True

    monkeypatch.setattr(pynml, NeuroMLAdapter._BACKENDS[backend], runner)
    NeuroMLAdapter._invoke_runner(backend, tmp_path / "LEMS_sim.xml", tmp_path)
    assert os.environ["PYTHONPATH"] == "/the/session/path"
