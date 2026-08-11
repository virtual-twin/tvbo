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


# Multi-mode models (number_of_modes > 1). The DifferentialEquations.jl backend
# (``test_run_julia``) lays each state variable out as a length-n_modes block and
# contracts mode_dot/mode_sum over the mode axis, matching jax/tvb (per-mode corr
# 1.0), so those models run there. The NetworkDynamics.jl and ModelingToolkit.jl
# backends still use scalar-per-variable templates without a mode-axis state layout,
# so they stay xfail'd. Mirrors _PYRATES_UNSUPPORTED.
_JULIA_MODE_UNSUPPORTED = {
    "ReducedSetHindmarshRose": "mode-axis model: nd/mtk backends have no mode-axis state layout yet",
    "ReducedSetFitzHughNagumo": "mode-axis model: nd/mtk backends have no mode-axis state layout yet",
    "StefanescuJirsa2D": "mode-axis model: nd/mtk backends have no mode-axis state layout yet",
    "StefanescuJirsa3D": "mode-axis model: nd/mtk backends have no mode-axis state layout yet",
}

# NetworkDynamics-specific: the stiff KIonEx ion-exchange model diverges under the
# nd solver, driving a concentration negative so a Nernst-potential log() hits a
# negative argument (Julia raises DomainError where numpy returns NaN). It runs
# correctly on the diffeq / jax / tvb backends.
_ND_UNSUPPORTED = {
    "KIonEx": "stiff ion-exchange model diverges under the nd solver (log of negative concentration)",
}


@pytest.mark.julia
@pytest.mark.backend_julia
@pytest.mark.xdist_group("julia")
@pytest.mark.skipif(not _HAVE_JULIACALL, reason="juliacall not installed")
class TestJuliaBackends:
    """Single-node tests for Julia-based simulation backends."""

    @pytest.mark.backend_julia_diffeq
    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_run_julia(self, model_file):
        """Run single-node simulation via DifferentialEquations.jl.

        Multi-mode models run here too: the DifferentialEquations.jl template lays
        each state variable out as a length-n_modes block (see tvbo-julia-model.jl.mako)
        and the result carries a ``mode`` axis.
        """
        model = Dynamics.from_file(model_file)
        exp = SimulationExperiment(dynamics=model)

        result = exp.run("julia")
        _assert_timeseries(result, model)

    @pytest.mark.backend_networkdynamics
    @pytest.mark.parametrize("model_file", MODEL_FILES, ids=MODEL_IDS)
    def test_run_networkdynamics(self, model_file):
        """Run single-node simulation via NetworkDynamics.jl."""
        reason = _JULIA_MODE_UNSUPPORTED.get(model_file.stem) or _ND_UNSUPPORTED.get(model_file.stem)
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
