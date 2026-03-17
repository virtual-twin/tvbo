"""Functional tests for TVB backend."""

import pytest

from tvbo.classes.experiment import SimulationExperiment
from tvbo.classes.dynamics import Dynamics
from tests.functional.simulation_backends_shared import (
    TVB_MODEL_FILES,
    TVB_MODEL_IDS,
    _HAVE_TVB,
)


@pytest.mark.backend_tvb
@pytest.mark.skipif(not _HAVE_TVB, reason="tvb-library not installed")
class TestTVBBackend:
    """Single-node tests for The Virtual Brain backend."""

    @pytest.mark.parametrize("model_file", TVB_MODEL_FILES, ids=TVB_MODEL_IDS)
    def test_run_tvb(self, model_file):
        """Run single-node simulation with The Virtual Brain backend."""
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(dynamics=model)
        result = exp.run("tvb")
        assert result is not None
