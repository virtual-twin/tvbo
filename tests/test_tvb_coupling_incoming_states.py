"""tvb couples a network through the states its coupling declares as incoming, in their declared order, as tvboptim does.

A coupling's ``incoming_states`` name what ``x_j`` holds, row by row, and its ``local_states`` what ``x_i`` holds: the model's ``coupling_variable`` flags are only the default for a coupling that declares neither. tvb sizes its history and both arrays from ``model.cvar``, so a ``cvar`` taken from the flags read another state where the identities differ and raised ``IndexError`` where the counts differ, and a source state named bare or as ``<s>_j`` reads its own row of ``x_j``. Each case runs on tvb and on tvboptim and compares the two trajectories.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from tvbo import SimulationExperiment

pytestmark = pytest.mark.backend_tvb

THREE_NODES = [
    {"source": 0, "target": 1, "weight": 1.0},
    {"source": 1, "target": 0, "weight": 1.0},
    {"source": 1, "target": 2, "weight": 0.5},
    {"source": 2, "target": 1, "weight": 0.5},
    {"source": 0, "target": 2, "weight": 0.25},
    {"source": 2, "target": 0, "weight": 0.25},
]
"""Undelayed directed edges among three nodes."""


def _network(states, coupling):
    """Three nodes opening at *states*, joined by `THREE_NODES` and coupled by *coupling*."""
    return {
        "number_of_nodes": 3,
        "nodes": [
            {"id": i, "label": f"n{i}", "state": {k: {"value": v} for k, v in state.items()}} for i, state in enumerate(states)
        ],
        "edges": [{**edge, "directed": True} for edge in THREE_NODES],
        "coupling": coupling,
    }


OSCILLATOR = {
    "id": 1,
    "dynamics": {"name": "Generic2dOscillator", "iri": "tvbo:models/Generic2dOscillator"},
    "network": _network(
        [{"V": 0.5, "W": -0.3}, {"V": -0.2, "W": 0.4}, {"V": 0.1, "W": -0.6}],
        {
            "c_glob": {
                "iri": "tvbo:coupling_functions/Linear",
                "delayed": False,
                "incoming_states": ["W"],
                "parameters": {"a": {"value": 1.0}, "b": {"value": 0.0}},
            }
        },
    ),
    "integration": {"method": "Euler", "step_size": 0.1, "duration": 50.0, "transient_time": 0.0},
    "execution": {"random_seed": 0, "precision": "float64"},
}
"""``Generic2dOscillator`` flags ``V``, and the coupling transmits ``W``: one state each, of different identity."""

SIGMOID = "cmin + (cmax - cmin) / (1.0 + exp(r * (midpoint - ({difference}))))"
COLUMNS = {
    "id": 1,
    "dynamics": {"name": "JansenRit1995", "iri": "tvbo:models/JansenRit1995"},
    "network": _network(
        [{"y0": 0.1, "y1": y1, "y2": 12.0, "y3": 0.0, "y4": 0.0, "y5": 0.0} for y1 in (12.0, 16.0, 20.0)],
        {
            "c_intercolumn": {
                "delayed": False,
                "incoming_states": ["y1", "y2"],
                "parameters": {
                    "G": {"value": 10.0},
                    "cmin": {"value": 0.0},
                    "cmax": {"value": 5.0},
                    "midpoint": {"value": 6.0},
                    "r": {"value": 0.56},
                },
                "pre_expression": {"rhs": None},
                "post_expression": {"rhs": "G * gx"},
            }
        },
    ),
    "integration": {"method": "Euler", "step_size": 0.0001220703125, "duration": 0.0625, "transient_time": 0.0},
    "execution": {"random_seed": 0, "precision": "float64"},
}
"""``JansenRit1995`` flags ``y4``, and the inter-column coupling transmits ``y1`` and ``y2``, the sigmoid of their difference, as the curated Jansen-Rit optimisations declare it; `_columns` writes the difference into its pre-expression."""

DIFFERENCES = {"bare": "y1 - y2", "named": "y1_j - y2_j"}
"""The source's ``y1 - y2`` by bare name, as ``TBPTT_JansenRit_FC_Optimization`` and ``JR_MEG_FrequencyGradient_Optimization`` write it, and by ``<s>_j``."""


def _columns(naming):
    """`COLUMNS` with its sigmoid reading the source's difference as *naming* writes it."""
    spec = copy.deepcopy(COLUMNS)
    spec["network"]["coupling"]["c_intercolumn"]["pre_expression"]["rhs"] = SIGMOID.format(difference=DIFFERENCES[naming])
    return SimulationExperiment(**spec)


def _by_step(experiment, backend):
    """*experiment*'s recorded states on *backend* as ``(time, variable, node)``, each sample keyed by the integration step it is stamped at."""
    pytest.importorskip(backend)
    trace = experiment.run(backend).integration
    if "mode" in trace.dims:
        trace = trace.isel(mode=0)
    trace = trace.transpose("time", "variable", "node")
    return trace.assign_coords(time=np.rint(trace["time"].values / float(experiment.integration.step_size)).astype(int))


def assert_backends_agree(make_experiment, backend, reference="tvboptim"):
    """The experiment *make_experiment* builds records the same states on *backend* as on *reference*, each run on its own copy, at every integration step both report: all of each, but for at most one opening sample."""
    ours, theirs = _by_step(make_experiment(), backend), _by_step(make_experiment(), reference)
    assert list(ours["variable"].values) == list(theirs["variable"].values)
    shared = np.intersect1d(ours["time"].values, theirs["time"].values)
    assert len(shared) >= max(ours.sizes["time"], theirs.sizes["time"]) - 1
    np.testing.assert_allclose(ours.sel(time=shared).values, theirs.sel(time=shared).values, rtol=1e-8, atol=1e-8)


def test_tvb_transmits_the_declared_state_where_the_flags_name_another():
    """The coupling transmits ``W`` on tvb, not the flagged ``V``."""
    assert_backends_agree(lambda: SimulationExperiment(**copy.deepcopy(OSCILLATOR)), "tvb")


@pytest.mark.parametrize("naming", list(DIFFERENCES))
def test_tvb_reads_each_incoming_state_from_its_own_row(naming):
    """The sigmoid reads the source's ``y1`` and ``y2`` from their two rows, by bare name or as ``<s>_j``, where the flags name the one state ``y4``."""
    assert_backends_agree(lambda: _columns(naming), "tvb")


LENGTHS = {(0, 1): 40.0, (1, 0): 40.0, (1, 2): 60.0, (2, 1): 60.0, (0, 2): 80.0, (2, 0): 80.0}
"""Tract lengths in mm, 40, 60 and 80 steps of 2^-13 s at 8192 mm/s."""


def _delayed_columns():
    """`COLUMNS` declared as ``ThreeNode_JansenRit1995_Parity`` declares it: the curated ``SigmoidalJansenRit``, which reads ``x_j[0] - x_j[1]``, delayed over `LENGTHS`, integrated by Heun."""
    spec = copy.deepcopy(COLUMNS)
    network = spec["network"]
    for edge in network["edges"]:
        edge["parameters"] = {
            "weight": {"name": "weight", "value": edge.pop("weight")},
            "length": {"name": "length", "value": LENGTHS[edge["source"], edge["target"]]},
        }
    network["parameters"] = {"conduction_speed": {"name": "conduction_speed", "value": 8192.0}}
    network["coupling"] = {
        "c_intercolumn": {
            "iri": "tvbo:coupling_functions/SigmoidalJansenRit",
            "delayed": True,
            "incoming_states": ["y1", "y2"],
            "parameters": {"a": {"value": 10.0}, "cmax": {"value": 5.0}},
        }
    }
    spec["integration"] = {"method": "Heun", "step_size": 0.0001220703125, "duration": 0.125, "transient_time": 0.0}
    return SimulationExperiment(**spec)


def test_tvb_reads_indexed_rows_over_a_delayed_coupling():
    """The parity experiment's declaration, whose delayed coupling reads ``x_j[0] - x_j[1]`` over ``incoming_states: [y1, y2]``, integrates on tvb the trajectory tvboptim integrates."""
    assert_backends_agree(_delayed_columns, "tvb")


def test_tvb_refuses_a_local_state_it_does_not_gather():
    """TVB hands a coupling's ``pre`` only the gathered states, so a ``V_i`` beside ``incoming_states: [W]`` has no row and is refused by name."""
    pytest.importorskip("tvb")
    spec = copy.deepcopy(OSCILLATOR)
    spec["network"]["coupling"]["c_glob"].update(
        {"local_states": ["V"], "pre_expression": {"rhs": "W_j - V_i"}, "post_expression": {"rhs": "a * gx + b"}}
    )
    with pytest.raises(
        NotImplementedError, match=r"reads the target's own 'V' as `V_i`, and the tvb backend hands a coupling only"
    ):
        SimulationExperiment(**spec).render_code("tvb")
