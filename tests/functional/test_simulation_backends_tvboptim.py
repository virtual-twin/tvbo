"""Functional tests for tvboptim backend."""

import pytest

from tvbo.export.experiment import SimulationExperiment
from tvbo.knowledge.simulation.localdynamics import Dynamics
from tests.functional.simulation_backends_shared import MODEL_FILES, MODEL_IDS, _HAVE_TVBOPTIM


@pytest.mark.backend_tvboptim
@pytest.mark.skipif(not _HAVE_TVBOPTIM, reason="tvboptim not installed")
class TestTvboptimBackend:
    """Single-node tests for the tvboptim JAX backend."""

    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_run_tvboptim(self, model_file):
        """Run single-node simulation with tvboptim JAX backend."""
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(dynamics=model)
        result = exp.run("tvboptim")
        assert result is not None
