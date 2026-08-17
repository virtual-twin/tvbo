"""Functional tests for PyRates backend."""

import pytest

from tests.functional.simulation_backends_shared import _HAVE_PYRATES, MODEL_FILES, MODEL_IDS
from tvbo.classes.dynamics import Dynamics
from tvbo.classes.experiment import SimulationExperiment

_PYRATES_UNSUPPORTED = {
    "ReducedSetHindmarshRose": "mode-axis model: PyRates has no mode axis (per-mode matrix params)",
    "ReducedSetFitzHughNagumo": "mode-axis model: PyRates has no mode axis (per-mode matrix params)",
    "EpileptorCodim3": "Piecewise regime traits unsupported by the PyRates sympy parser",
    "EpileptorCodim3SlowMod": "Piecewise regime traits unsupported by the PyRates sympy parser",
    "ZerlautAdaptationSecondOrder": "symbolic form unsupported by PyRates codegen",
}
"""Models the PyRates backend cannot represent, xfailed (not failed) as out of scope.

Two families are excluded: multi-mode models, whose per-mode matrix parameters (``A_ik``/``B_ik``/``C_ik``) appear in the equations and have no mode axis to live on in PyRates; and the Piecewise regime traits and symbolic forms from the richer TVB model capture that the PyRates sympy parser cannot handle. Full support for these lives in the tvb, tvboptim and jax backends.
"""


@pytest.mark.backend_pyrates
@pytest.mark.skipif(not _HAVE_PYRATES, reason="pyrates not installed")
class TestPyRatesBackend:
    """Single-node tests for the PyRates backend."""

    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_run_pyrates(self, model_file):
        """Run single-node simulation with PyRates backend."""
        reason = _PYRATES_UNSUPPORTED.get(model_file.stem)
        if reason:
            pytest.xfail(reason)
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(dynamics=model)
        result = exp.run("pyrates")
        assert result is not None
