"""Functional tests for PyRates backend."""

import pytest

from tvbo.export.experiment import SimulationExperiment
from tvbo.knowledge.simulation.localdynamics import Dynamics
from tests.functional.simulation_backends_shared import MODEL_FILES, MODEL_IDS, _HAVE_PYRATES


@pytest.mark.backend_pyrates
@pytest.mark.skipif(not _HAVE_PYRATES, reason="pyrates not installed")
class TestPyRatesBackend:
    """Single-node tests for the PyRates backend."""

    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_run_pyrates(self, model_file):
        """Run single-node simulation with PyRates backend."""
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(dynamics=model)
        result = exp.run("pyrates")
        assert result is not None
