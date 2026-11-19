"""The tvb backend names and orders its recorded variables as tvboptim presents them, each once, whatever ``output`` and ``record`` declare.

``variables_of_interest`` was the recorded states followed by every ``output`` name that is not a recorded derived variable, so ``output: [x, y]`` on two states recorded ``(x, y, x, y)``, the observer returned four rows and ``trace.sel(variable="x")`` two. The model is ``x' = -a x`` beside ``y' = x``, with ``s = 2 x`` derived, so a row's label can be checked against what it holds.
"""

from __future__ import annotations

import copy
import re

import numpy as np
import pytest

pytest.importorskip("tvb")

from tvbo import SimulationExperiment

pytestmark = pytest.mark.backend_tvb

_SPEC = """
id: 1
dynamics:
  name: Pair
  parameters:
    a: {value: 1.0}
  state_variables:
    x: {equation: {rhs: "-a*x"}, initial_value: 1.0%(x_record)s}
    y: {equation: {rhs: "x"}, initial_value: 0.0}
  derived_variables:
    s: {equation: {rhs: "2*x"}}
%(output)snetwork:
  number_of_nodes: 2
  nodes:
    - {id: 0, label: p}
    - {id: 1, label: q}
integration: {method: Euler, step_size: 0.1, duration: 1.0}
"""


def _spec(output=None, x_recorded=True):
    """The pair with the legacy ``output`` list *output*, if any, and ``x`` declaring ``record: false`` unless *x_recorded*."""
    return _SPEC % {
        "output": f"  output: [{', '.join(output)}]\n" if output is not None else "",
        "x_record": "" if x_recorded else ", record: false",
    }


def _names(spec, backend):
    """The ``variable`` labels of *spec*'s trace on *backend*, in their order."""
    trace = SimulationExperiment.from_string(spec).run(backend).integration.data
    return [str(name) for name in trace["variable"].values]


@pytest.mark.backend_tvboptim
@pytest.mark.parametrize(
    "output, x_recorded",
    [(["x", "y"], True), (["y", "x"], True), (None, True), (None, False), (["x"], False), (["x", "s"], True)],
    ids=["declared", "reversed", "no-output", "x-unrecorded", "x-unrecorded-named", "derived"],
)
def test_tvb_records_the_names_tvboptim_presents(output, x_recorded):
    """The same names in the same order on both backends, none twice, and each a choice the tvb model class declares once."""
    pytest.importorskip("tvboptim")
    spec = _spec(output, x_recorded)
    names = _names(spec, "tvb")
    assert names == _names(spec, "tvboptim")
    assert len(set(names)) == len(names)
    model = SimulationExperiment.from_string(spec).execute("tvb").model
    choices = tuple(type(model).variables_of_interest.element_choices)
    assert len(set(choices)) == len(choices)
    assert set(names) <= set(choices)


def test_the_observer_computes_only_the_derived_outputs():
    """With a state and a derived variable in ``output``, the observer computes ``s`` and passes the states through as they are."""
    spec = _spec(["x", "s"])
    code = SimulationExperiment.from_string(spec).execute("tvb").model.observe.code
    assert re.search(r"^\s+s = ", code, re.M)
    assert not re.search(r"^\s+[xy] = ", code, re.M)
    trace = SimulationExperiment.from_string(spec).run("tvb").integration.data.isel(mode=0)
    np.testing.assert_allclose(trace.sel(variable="s").values, 2 * trace.sel(variable="x").values, rtol=1e-12)


K = 0.5
X0 = 0.3
COUPLING_RTOL = 1e-12
"""The tolerance on what the coupling delivered, at the float64 the declared precision gives TVB's delay history; the opening 0.3 is not exact in float32, so a float32 history misses it."""

COUPLED = {
    "id": 1,
    "label": "coupling read by an observation",
    "dynamics": {
        "name": "Relay",
        "parameters": {"k": {"value": K}},
        "coupling_inputs": {"c": {"name": "c"}},
        "state_variables": {"x": {"equation": {"rhs": "c - k * x"}, "initial_value": X0, "coupling_variable": True}},
        "derived_variables": {"drive": {"equation": {"rhs": "c - k * x"}}},
    },
    "network": {
        "number_of_nodes": 2,
        "nodes": [{"id": 0, "label": "r0"}, {"id": 1, "label": "r1"}],
        "edges": [
            {
                "source": i,
                "target": j,
                "directed": True,
                "parameters": {"weight": {"name": "weight", "value": 1.0}, "length": {"name": "length", "value": 0.0}},
            }
            for i, j in ((0, 1), (1, 0))
        ],
        "parameters": {"conduction_speed": {"name": "conduction_speed", "value": 1.0}},
        "coupling": {
            "LinearCoupling": {
                "delayed": False,
                "parameters": {"a": {"value": 1.0}},
                "pre_expression": {"rhs": "x_j"},
                "post_expression": {"rhs": "a * gx"},
                "incoming_states": ["x"],
                "local_states": ["x"],
            }
        },
    },
    "observations": {"drive_seen": {"source": ["drive"], "period": 0.1}},
    "integration": {"method": "Euler", "step_size": 0.1, "duration": 8.0, "transient_time": 0.0},
}
"""Two nodes sending ``x`` to each other at weight 1 with no delay; ``drive = c - k x`` reads the coupling and is recorded only because an observation reads it."""


@pytest.mark.parametrize("method, stages", [("Euler", 1), ("Heun", 2)])
def test_an_observed_derived_variable_reading_the_coupling_is_kept_from_the_step_s_first_evaluation(method, stages):
    """TVB's observer sees the state alone, so tvb records ``drive`` as tvboptim does: from the first evaluation of each step, at the state the step started from and the coupling it delivered, ``W x[t-1] - k x[t-1]`` on the row of ``x[t]``. Heun evaluates twice, the second time at its predictor, so keeping the last evaluation would be off that line."""
    spec = copy.deepcopy(COUPLED)
    spec["integration"].update(method=method, number_of_stages=stages)
    experiment = SimulationExperiment(**spec)
    trace = experiment.run("tvb").integration.data.isel(mode=0)
    assert [str(name) for name in trace["variable"].values] == ["x", "drive"]
    x = trace.sel(variable="x").transpose("time", "node").values
    drive = trace.sel(variable="drive").transpose("time", "node").values
    weights = np.asarray(experiment.network.weights)
    opening = np.full(x.shape[1], X0)
    np.testing.assert_allclose(
        drive[0],
        opening @ weights.T - K * opening,
        rtol=COUPLING_RTOL,
        err_msg="the first row is the first step's evaluation at the declared opening",
    )
    np.testing.assert_allclose(drive[1:], x[:-1] @ weights.T - K * x[:-1], rtol=COUPLING_RTOL)
    unaligned = np.max(np.abs(drive[1:] - (x[1:] @ weights.T - K * x[1:])))
    assert unaligned > 1e-3, "the run must move enough to tell a step's start from its end"
