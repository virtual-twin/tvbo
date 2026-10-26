"""An experiment prints as a plain `str`, which a pytest-xdist failure report can carry.

pytest renders a failing test's arguments with ``repr``, and pytest-xdist sends the report to the controller through execnet, which serializes builtin types only. The label is LinkML's `extended_str`, and returned as it is the worker dies with ``DumpError`` and the run reports the tests before it as the whole outcome.
"""

from __future__ import annotations

import pytest

from tvbo.classes.experiment import SimulationExperiment


def test_an_experiment_prints_as_a_plain_str(tmp_path):
    execnet = pytest.importorskip("execnet")
    spec = tmp_path / "labelled.yaml"
    spec.write_text("id: 1\nlabel: a labelled experiment\ndynamics: {name: Generic2dOscillator}\n")
    experiment = SimulationExperiment.from_file(str(spec))
    for printed in (repr(experiment), str(experiment)):
        assert type(printed) is str
        assert execnet.loads(execnet.dumps(printed)) == "a labelled experiment"
