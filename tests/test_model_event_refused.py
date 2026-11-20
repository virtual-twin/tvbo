"""A backend with no path for the affect of an event a model declares, such as a spike's reset, refuses the model by that declaration rather than integrate it without the affect.

The refusal reads what the model declares, the event and the affect it sets, never its name: the curated Izhikevich2007 cell, whose ``spike`` event resets ``v = c; u = u + d``, is refused on every backend that has no reset path (DifferentialEquations.jl, ModelingToolkit.jl, the CUDA kernel, tvb, NetworkDynamics.jl, python, pyrates and gillespie), by render where a format renders it and by run where only a run reaches the backend, and the same dynamics under another name is refused with the same message. An event with a condition alone, which only marks a spike, passes, and LEMS still renders the reset.
"""

from __future__ import annotations

import re

import pytest

from tvbo import Dynamics, SimulationExperiment

RESET = re.escape(
    "has no path for the affect of an event a model declares, so it would integrate the model without it: "
    "event 'spike' sets v = c; u = u + d."
)


def _experiment(dynamics):
    """One node of *dynamics*, integrated for a millisecond."""
    return SimulationExperiment(dynamics=dynamics, integration={"duration": 1.0, "step_size": 0.1})


def _reset_cell(name="Izhikevich2007"):
    """The curated Izhikevich2007 cell under *name*."""
    return _experiment(Dynamics.from_db("Izhikevich2007").copy(name=name))


@pytest.mark.parametrize(
    "key, backend",
    [
        ("julia", "julia"),
        ("modelingtoolkit", "modelingtoolkit"),
        ("cuda", "rateml-cuda"),
        ("tvb", "tvb"),
        ("networkdynamics", "networkdynamics"),
    ],
    ids=["julia", "modelingtoolkit", "cuda", "tvb", "networkdynamics"],
)
def test_a_render_with_no_reset_path_refuses_a_reset(key, backend):
    """The render names the backend, the event and the affect it would drop."""
    with pytest.raises(NotImplementedError, match=rf"the {backend} backend {RESET}"):
        _reset_cell().render_code(key)


@pytest.mark.parametrize("backend", ["python", "pyrates", "gillespie"])
def test_a_run_only_backend_refuses_a_reset_before_it_runs(backend):
    """python, pyrates and gillespie, which no format registers, meet the check where every run does, before the dispatch."""
    with pytest.raises(NotImplementedError, match=rf"the {backend} backend {RESET}"):
        _reset_cell().run(backend)


def test_the_refusal_reads_the_declaration_not_the_name():
    """The same dynamics under another name is refused with the same message."""
    messages = []
    for name in ("Izhikevich2007", "AnyCell"):
        with pytest.raises(NotImplementedError) as refused:
            _reset_cell(name).render_code("julia")
        messages.append(str(refused.value))
    assert messages[0] == messages[1]


def test_an_event_with_a_condition_alone_passes():
    """hhcell_1's ``spike`` event declares a threshold and no affect, so it changes no trajectory and the render goes on, on tvb, which applies no affect."""
    assert _experiment(Dynamics.from_db("hhcell_1")).render_code("tvb")


def test_lems_still_renders_the_reset():
    """LEMS applies a model's events, so the reset cell renders, its affect among them as the printer orders it."""
    assert '<StateAssignment variable="u" value="d + u"/>' in _reset_cell().render_code("lems")
