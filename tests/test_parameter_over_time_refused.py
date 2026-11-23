"""A parameter declared with `axis: time` is refused by name on every backend that does not list its kind as one it draws at every step; tvboptim still draws a dynamics parameter and a stimulus's one such parameter.

The export registry's `StartKind`s say what a backend draws per step: tvboptim lists a dynamics parameter (`TIME`) and the one such parameter of a stimulus that is an equation of time (`STIMULUS_TIME`); no backend lists a coupling parameter (`COUPLING_TIME`) or any other event's (`EVENT_TIME`). The check is `refuse_unlowered_parameter_starts`, which rendering and every run path call, so the refusals are read off it, and for tvb, jax and NetworkDynamics.jl off the render; the tvboptim control runs.
"""

from __future__ import annotations

import copy
import re

import numpy as np
import pytest

from tvbo import SimulationExperiment
from tvbo.export import refuse_unlowered_parameter_starts

from .test_initial_state_distribution import ONE_VARIABLE, SEED
from .test_parameter_start_from_distribution import EDGE

OVER_TIME = "distribution: {name: Uniform, domain: {lo: 0.0, hi: 2.0}, axis: time}"
DRAWN_OVER_TIME = {"name": "Uniform", "domain": {"lo": 0.0, "hi": 2.0}, "axis": "time"}
EVERY_OTHER_BACKEND = [
    "tvb", "jax", "networkdynamics", "pyrates", "brian2", "julia", "modelingtoolkit", "python", "gillespie",
    "neuroml", "lems", "pde", "rateml-cuda", "bifurcationkit", "pyrates-bifurcation", "numcont",
]  # fmt: skip
RENDERED = ["tvb", "jax", "networkdynamics"]
KINDS = [("dynamics", "parameter 'a' of dynamics 'Leak'"), ("coupling", "parameter 'w' of coupling 'LinearCoupling'")]


def _refusal(backend, held):
    """The refusal *backend* gives for the parameters *held*, as a pattern."""
    return rf"the {re.escape(backend)} backend sets these parameters once and never per step, so {held}, declared with `axis: time`"


def _over_time(kind):
    """The coupled leak of `EDGE`, without its observation, whose dynamics parameter ``a`` or coupling parameter ``w`` (*kind*) is drawn over time."""
    w = "        w: {value: 1.0" + (f", {OVER_TIME}" if kind == "coupling" else "") + "}"
    spec = EDGE.format(parameters=w, pre="x_j * w", seed=SEED).split("observations:")[0]
    if kind == "dynamics":
        spec = spec.replace("a: {value: -0.3}", f"a: {{value: -0.3, {OVER_TIME}}}")
    return SimulationExperiment.from_string(spec)


def _with_event(name, event):
    """G240's one-variable recipe as one node, with *event* declared under *name*."""
    spec = copy.deepcopy(ONE_VARIABLE)
    del spec["network"]
    spec["events"] = {name: event}
    return SimulationExperiment(**spec)


def _drawn(*names):
    """Event parameters *names*, each drawn over time."""
    return {name: {"name": name, "value": 0.0, "distribution": dict(DRAWN_OVER_TIME)} for name in names}


STIMULUS = {"event_type": "stimulation", "equation": {"rhs": "u"}, "parameters": _drawn("u"), "target_variable": "x"}
STATE_TRIGGERED = {
    "event_type": "continuous",
    "condition": {"rhs": "x - 1"},
    "affect": {"rhs": "u"},
    "parameters": _drawn("u"),
    "target_variable": "x",
}
TWO_DRAWS = {"event_type": "stimulation", "equation": {"rhs": "u + v"}, "parameters": _drawn("u", "v"), "target_variable": "x"}


@pytest.mark.parametrize("backend", EVERY_OTHER_BACKEND)
def test_every_backend_but_tvboptim_refuses_a_dynamics_parameter_drawn_over_time(backend):
    """The registry check names the backend and the parameter, offers `axis: space` only where the backend draws a dynamics parameter per node, and points to tvboptim."""
    with pytest.raises(NotImplementedError, match=_refusal(backend, "parameter 'a' of dynamics 'Leak'")) as refused:
        refuse_unlowered_parameter_starts(_over_time("dynamics"), backend)
    assert ("`axis: space`" in str(refused.value)) == (backend == "networkdynamics")
    assert "run the experiment on tvboptim" in str(refused.value)


@pytest.mark.parametrize("backend", RENDERED)
@pytest.mark.parametrize("kind, held", KINDS, ids=[kind for kind, _ in KINDS])
def test_the_render_refuses_a_parameter_drawn_over_time(backend, kind, held):
    """tvb, jax and NetworkDynamics.jl meet the check when they render, for a dynamics and a coupling parameter alike."""
    with pytest.raises(NotImplementedError, match=_refusal(backend, held)):
        _over_time(kind).render_code(backend)


def test_tvboptim_refuses_a_coupling_parameter_drawn_over_time():
    """A coupling parameter drawn over time is refused on tvboptim, which offers the draw over space it does make and no other backend."""
    held = "parameter 'w' of coupling 'LinearCoupling'"
    with pytest.raises(NotImplementedError, match=_refusal("tvboptim", held)) as refused:
        refuse_unlowered_parameter_starts(_over_time("coupling"), "tvboptim")
    assert "`axis: space`" in str(refused.value) and "run the experiment on tvboptim" not in str(refused.value)


@pytest.mark.parametrize("backend", ["tvb", "python"])
def test_a_stimulus_parameter_drawn_over_time_is_refused_where_no_step_draws_it(backend):
    """A stimulus's one parameter drawn over time is refused by name on tvb and python, which evaluate a stimulus at its parameters' values."""
    with pytest.raises(NotImplementedError, match=_refusal(backend, "parameter 'u' of event 'u'")):
        refuse_unlowered_parameter_starts(_with_event("u", STIMULUS), backend)


@pytest.mark.parametrize(
    "name, event, held",
    [
        ("trigger", STATE_TRIGGERED, "parameter 'u' of event 'trigger'"),
        ("s", TWO_DRAWS, "parameter 'u' of event 's', parameter 'v' of event 's'"),
    ],
    ids=["state-triggered", "two-on-one-event"],
)
def test_tvboptim_refuses_an_event_parameter_it_draws_no_step_for(name, event, held):
    """Only the one parameter drawn over time of a stimulus that is an equation of time is drawn on tvboptim: a state-triggered event's is refused, and so are both of two on one event, since which one it drew would rest on declaration order."""
    with pytest.raises(NotImplementedError, match=_refusal("tvboptim", held)):
        refuse_unlowered_parameter_starts(_with_event(name, event), "tvboptim")


def test_tvboptim_accepts_the_one_stimulus_parameter_it_draws_per_step():
    """The control for events: the registry check passes the stimulus tvboptim draws per step."""
    refuse_unlowered_parameter_starts(_with_event("u", STIMULUS), "tvboptim")


def test_tvboptim_still_draws_a_dynamics_parameter_at_every_step():
    """The control: on tvboptim ``dx/dt = a`` under Euler advances by the step size times that step's draw of ``a``, so the rates vary from step to step and stay within the declared bounds."""
    spec = copy.deepcopy(ONE_VARIABLE)
    del spec["network"]
    spec["dynamics"]["state_variables"]["x"]["equation"]["rhs"] = "a"
    spec["dynamics"]["parameters"]["a"]["distribution"] = dict(DRAWN_OVER_TIME)
    data = SimulationExperiment(**spec).run("tvboptim").integration.data
    rates = np.diff(np.asarray(data.sel(variable="x")).ravel()) / spec["integration"]["step_size"]
    assert np.all((rates > -1e-9) & (rates < 2.0 + 1e-9)), rates
    assert np.unique(rates.round(9)).size > 1, rates
