"""Functional tests for TVB backend."""

import pytest

from tests.functional.simulation_backends_shared import (
    _HAVE_TVB,
    TVB_MODEL_FILES,
    TVB_MODEL_IDS,
)
from tvbo.classes.dynamics import Dynamics
from tvbo.classes.experiment import SimulationExperiment

# Models whose numba compilation on TVB is legitimately slow (huge erfc / quadrature expressions) and exceed the CI-wide per-test timeout. They are skipped on the non-TVB backends (see simulation_backends_shared._SKIP_NON_TVB) and kept here with a longer per-test budget.
_TVB_SLOW = {"ZerlautAdaptationSecondOrder"}
_TVB_PARAMS = [pytest.param(f, marks=pytest.mark.timeout(1800)) if f.stem in _TVB_SLOW else f for f in TVB_MODEL_FILES]


@pytest.mark.backend_tvb
@pytest.mark.skipif(not _HAVE_TVB, reason="tvb-library not installed")
class TestTVBBackend:
    """Single-node tests for The Virtual Brain backend."""

    @pytest.mark.parametrize("model_file", _TVB_PARAMS, ids=TVB_MODEL_IDS)
    def test_run_tvb(self, model_file):
        """Run single-node simulation with The Virtual Brain backend."""
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(dynamics=model)
        result = exp.run("tvb")
        assert result is not None
