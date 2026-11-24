"""A backend with no stimulus path refuses an event the experiment declares, by name, rather than run the unstimulated model.

DifferentialEquations.jl, ModelingToolkit.jl, gillespie and the CUDA kernel have no stimulus path, and the continuation backends continue an equilibrium, where a stimulus has no place. Every id goes through a public path: rendering (`render_code`, which renders through the export registry) or, for gillespie, which no format registers, `run`. An event the dynamics reads by name is refused as well, before it can reach a render as a free symbol. tvboptim, tvb, jax, python and pyrates pass, and tvboptim still runs the event.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from tvbo import SimulationExperiment

from .test_initial_state_distribution import ONE_VARIABLE

DRIVE = {
    "event_type": "stimulation",
    "equation": {"rhs": "amp"},
    "parameters": {"amp": {"name": "amp", "value": 1.0}},
    "target_variable": "x",
}
NO_STIMULUS_PATH = "has no stimulus path, so it would run the unstimulated model, and the experiment declares event 'drive'"
CONTINUATION = "continues an equilibrium, where a stimulus has no place, and the experiment declares event 'drive'"


def _driven(rhs="-a * x", event=DRIVE):
    """G240's one-variable recipe as one node, ``dx/dt = rhs``, with *event* declared as ``drive``."""
    spec = copy.deepcopy(ONE_VARIABLE)
    del spec["network"]
    spec["dynamics"]["state_variables"]["x"]["equation"]["rhs"] = rhs
    spec["events"] = {"drive": copy.deepcopy(event)}
    return SimulationExperiment(**spec)


@pytest.mark.parametrize(
    "key, backend",
    [("julia", "julia"), ("modelingtoolkit", "modelingtoolkit"), ("cuda", "rateml-cuda")],
    ids=["julia", "modelingtoolkit", "cuda"],
)
def test_a_render_with_no_stimulus_path_refuses_a_declared_event(key, backend):
    """The render names the backend and the event, and says the run would integrate the unstimulated model."""
    with pytest.raises(NotImplementedError, match=rf"the {backend} backend {NO_STIMULUS_PATH}"):
        _driven().render_code(key)


def test_gillespie_refuses_a_declared_event_before_it_runs():
    """gillespie, which no format registers, meets the check where every run does, before its dispatch."""
    with pytest.raises(NotImplementedError, match=rf"the gillespie backend {NO_STIMULUS_PATH}"):
        _driven().run("gillespie")


@pytest.mark.parametrize("backend", ["bifurcationkit", "numcont", "pyrates-bifurcation"])
def test_a_continuation_backend_refuses_a_declared_event(backend):
    """A continuation follows an equilibrium, where a stimulus has no place; the render says so before it reads any continuation."""
    with pytest.raises(NotImplementedError, match=rf"the {backend} backend {CONTINUATION}"):
        _driven().render_code(backend)


def test_an_event_the_dynamics_reads_by_name_is_refused_at_render():
    """An event the equation names, with no target, is refused when the Julia renders rather than reaching it as a free symbol."""
    referenced = {"event_type": "stimulation", "equation": {"rhs": "sin(t)"}}
    with pytest.raises(NotImplementedError, match=rf"the julia backend {NO_STIMULUS_PATH}"):
        _driven(rhs="-a * x + drive", event=referenced).render_code("julia")


@pytest.mark.parametrize("backend", ["tvboptim", "tvb", "jax"])
def test_a_render_that_integrates_a_stimulus_passes(backend):
    """tvboptim, tvb and jax render the declared stimulus."""
    assert _driven().render_code(backend)


@pytest.mark.parametrize("backend", ["python", "pyrates"])
def test_a_run_only_backend_that_integrates_a_stimulus_passes(backend):
    """The run-only python and pyrates, which no format registers, evaluate a stimulus and run."""
    assert _driven().run(backend).integration is not None


def test_tvboptim_still_runs_the_event():
    """The control: ``dx/dt = drive`` with a constant drive of 1 rises on tvboptim, where the unstimulated model would stay at 0."""
    data = _driven(rhs="0 * x").run("tvboptim").integration.data
    final = float(np.asarray(data.sel(variable="x")).ravel()[-1])
    assert final > 0.5, final
