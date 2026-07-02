"""Functional tests for Julia-based backends."""

import pytest

from tvbo.classes.experiment import SimulationExperiment
from tvbo.classes.dynamics import Dynamics
from tests.functional.simulation_backends_shared import (
    MODEL_FILES,
    MODEL_IDS,
    _HAVE_JULIACALL,
    _assert_timeseries,
)


# Multi-mode models (number_of_modes > 1) whose state variables are per-mode
# vectors. mode_dot/mode_sum now render on Julia, but the Julia templates still
# lay out state as flat scalars (`u0`/`dx[i]` are scalar slots), so a length-N
# mode vector can't be written into `dx[i]`. xfail'd (not failed) until the Julia
# backend grows a mode-axis state layout; full support lives in the
# tvb / tvboptim / jax backends. Mirrors _PYRATES_UNSUPPORTED.
_JULIA_MODE_UNSUPPORTED = {
    "ReducedSetHindmarshRose": "mode-axis model: Julia backend has no mode-axis state layout yet",
    "ReducedSetFitzHughNagumo": "mode-axis model: Julia backend has no mode-axis state layout yet",
    "StefanescuJirsa2D": "mode-axis model: Julia backend has no mode-axis state layout yet",
    "StefanescuJirsa3D": "mode-axis model: Julia backend has no mode-axis state layout yet",
}


@pytest.mark.backend_julia
@pytest.mark.xdist_group("julia")
@pytest.mark.skipif(not _HAVE_JULIACALL, reason="juliacall not installed")
class TestJuliaBackends:
    """Single-node tests for Julia-based simulation backends."""

    @pytest.mark.backend_julia_diffeq
    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_run_julia(self, model_file):
        """Run single-node simulation via DifferentialEquations.jl."""
        reason = _JULIA_MODE_UNSUPPORTED.get(model_file.stem)
        if reason:
            pytest.xfail(reason)
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(dynamics=model)

        result = exp.run("julia")
        _assert_timeseries(result, model)

    @pytest.mark.backend_networkdynamics
    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_run_networkdynamics(self, model_file):
        """Run single-node simulation via NetworkDynamics.jl."""
        reason = _JULIA_MODE_UNSUPPORTED.get(model_file.stem)
        if reason:
            pytest.xfail(reason)
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(dynamics=model)

        result = exp.run("networkdynamics")
        _assert_timeseries(result, model)

    @pytest.mark.backend_mtk
    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_run_mtk(self, model_file):
        """Run single-node simulation via ModelingToolkit.jl."""
        reason = _JULIA_MODE_UNSUPPORTED.get(model_file.stem)
        if reason:
            pytest.xfail(reason)
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(dynamics=model)

        result = exp.run("mtk")
        _assert_timeseries(result, model)
