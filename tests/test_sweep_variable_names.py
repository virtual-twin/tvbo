"""A sweep's grid names its ``variable`` axis as the single run names it, whether or not the model declares ``output``.

The tvboptim backend named an exploration's ``variable`` axis only from ``model.output``, so the grid of a model that declares none kept a positional axis and ``grid.sel(variable="x")`` raised ``KeyError: 'variable'``. A cell now returns the channels a single run presents, its ``record: true`` variables, under the names that run gives them. The reference for every cell is the experiment run alone at the cell's values, compared channel by channel by name; a model that declares ``output`` keeps the names and order it had.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment

KS = (-0.5, -0.1)
"""The swept values of ``Probe.k``, the base run's value last."""

N_TRIALS = 3

PROBE = {
    "id": 1,
    "label": "two-variable probe",
    "dynamics": {
        "name": "Probe",
        "parameters": {"k": {"value": KS[1]}},
        "coupling_inputs": {"c": {"name": "c"}},
        "state_variables": {
            "x": {"equation": {"rhs": "k * x"}, "initial_value": 1.0, "coupling_variable": True},
            "y": {"equation": {"rhs": "c - y"}, "initial_value": 0.0},
        },
    },
    "network": {
        "number_of_nodes": 2,
        "nodes": [{"id": 0, "label": "r0"}, {"id": 1, "label": "r1"}],
        "edges": [
            {"source": 0, "target": 1, "parameters": {"weight": {"value": 0.5}}},
            {"source": 1, "target": 0, "parameters": {"weight": {"value": 0.25}}},
        ],
        "coupling": {
            "LinearCoupling": {
                "parameters": {"a": {"value": 1.0}},
                "pre_expression": {"rhs": "x_j"},
                "post_expression": {"rhs": "a * gx"},
                "incoming_states": ["x"],
                "local_states": ["x"],
            }
        },
    },
    "execution": {"random_seed": 0, "precision": "float64"},
    "integration": {"method": "Euler", "step_size": 1.0, "duration": 5.0, "transient_time": 0.0},
}
"""Two undelayed nodes without ``output``: ``x`` decays at rate ``k`` and is what the coupling sends, ``y`` relaxes towards what arrives, so both channels move and differ."""

SWEEP = {"space": [{"parameter": "Probe.k", "explored_values": list(KS)}]}


def _with(spec=PROBE, k=None, **dynamics):
    """*spec* with ``Probe.k`` set to *k* when given and the *dynamics* entries added."""
    spec = copy.deepcopy(spec)
    if k is not None:
        spec["dynamics"]["parameters"]["k"]["value"] = k
    spec["dynamics"].update(copy.deepcopy(dynamics))
    return spec


def _samples(da):
    """*da* as ``(time, variable, node)``, every other length-one dim dropped."""
    return da.squeeze([d for d in da.dims if d not in ("time", "variable", "node") and da.sizes[d] == 1]).transpose(
        "time", "variable", "node"
    )


def _grid(spec, exploration):
    """*spec* run on tvboptim with *exploration* as its one exploration, that exploration's ``as_grid()``."""
    spec = copy.deepcopy(spec)
    spec["explorations"] = {"e": {"name": "e", "mode": "product", **exploration}}
    return SimulationExperiment(**spec).run("tvboptim").explorations["e"].as_grid()


def _alone(spec):
    """*spec*'s single run on tvboptim, its measured window as ``(time, variable, node)``."""
    return _samples(SimulationExperiment(**copy.deepcopy(spec)).run("tvboptim").integration.data)


def _swept_cells(spec):
    """Each cell of *spec* swept over ``KS``, in the order of ``KS``, as ``(time, variable, node)``."""
    grid = _grid(spec, SWEEP)
    (axis,) = [d for d in grid.dims if d not in ("time", "variable", "node", "mode")]
    cells = [_samples(grid.sel({axis: k}, method="nearest")) for k in KS]
    assert np.abs(cells[0].values - cells[1].values).max() > 1e-3, "the cells must differ for the comparison to mean anything"
    return cells


def _assert_cell_is_run_alone(cell, alone, names):
    """*cell* carries *names* on ``variable`` in that order, and each named channel equals *alone*'s sample for sample."""
    assert cell.coords["variable"].values.tolist() == names
    for name in names:
        np.testing.assert_allclose(cell.sel(variable=name).values, alone.sel(variable=name).values, rtol=1e-12, atol=1e-12)


def test_a_sweep_of_a_model_without_output_names_its_variables():
    """Each cell of a parameter sweep is selectable by ``variable="x"`` and ``"y"`` and equals the experiment run alone at its value."""
    for k, cell in zip(KS, _swept_cells(PROBE), strict=True):
        alone = _alone(_with(k=k))
        assert alone.coords["variable"].values.tolist() == ["x", "y"]
        _assert_cell_is_run_alone(cell, alone, ["x", "y"])


def test_an_n_trials_ensemble_of_a_model_without_output_names_its_variables():
    """Each trial of an ensemble that draws ``x`` per node is selectable by name and equals the experiment run alone from that trial's opening.

    At ``k = 0`` ``x`` holds its opening, so a trial's opening is its ``x`` at any sample, and the run alone declares it per node.
    """
    spec = _with(k=0.0)
    spec["dynamics"]["state_variables"]["x"]["distribution"] = {"name": "Uniform", "domain": {"lo": 1.0, "hi": 2.0}}
    grid = _grid(spec, {"n_trials": N_TRIALS, "space": [{"parameter": "Probe.k", "explored_values": [0.0]}]})
    trials = [_samples(grid.isel(trial=i)) for i in range(N_TRIALS)]
    openings = [trial.sel(variable="x").values[0] for trial in trials]
    assert min(np.abs(a - b).max() for i, a in enumerate(openings) for b in openings[i + 1 :]) > 1e-6, (
        "the trials must draw different openings for the comparison to mean anything"
    )
    for trial, opening in zip(trials, openings, strict=True):
        alone = copy.deepcopy(spec)
        del alone["dynamics"]["state_variables"]["x"]["distribution"]
        for node, x in zip(alone["network"]["nodes"], opening, strict=True):
            node["state"] = {"x": {"value": float(x)}}
        _assert_cell_is_run_alone(trial, _alone(alone), ["x", "y"])


def test_a_sweep_presents_only_the_variables_a_single_run_records():
    """A state declared ``record: false`` is left out of each cell as it is out of the single run, and the kept one keeps its name."""
    spec = copy.deepcopy(PROBE)
    spec["dynamics"]["state_variables"]["y"]["record"] = False
    for k, cell in zip(KS, _swept_cells(spec), strict=True):
        alone = _alone(_with(spec, k=k))
        assert alone.coords["variable"].values.tolist() == ["x"]
        _assert_cell_is_run_alone(cell, alone, ["x"])


@pytest.mark.parametrize(
    ("dynamics", "names"),
    [
        ({"output": ["y", "x"]}, ["y", "x"]),
        ({"derived_variables": {"total": {"equation": {"rhs": "x + y"}}}, "output": ["total", "x"]}, ["total", "x"]),
    ],
    ids=["state-outputs", "derived-output"],
)
def test_a_declared_output_keeps_its_names_and_order(dynamics, names):
    """A model that declares ``output`` keeps it on ``variable``, in the declared order, a derived output included, each channel the single run's of that name."""
    spec = _with(**dynamics)
    for k, cell in zip(KS, _swept_cells(spec), strict=True):
        _assert_cell_is_run_alone(cell, _alone(_with(spec, k=k)), names)
