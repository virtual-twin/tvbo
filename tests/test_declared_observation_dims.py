"""An observation declares its own axes, so nothing has to infer them from a length.

A standard reduction states its axes at codegen; a `pipeline` of user functions has an output shape only its author knows. `Observation.dims` is where that author says it, and it is the only thing that names those axes: guessing from a length is how a four-sample trace in a four-node network comes to be labelled by region.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tvbo.data.types import ObservationResult, SimulationResult
from tvbo.templates.tvboptim.utils import observation_dims

LABELS = ["L.LOG", "L.SFG", "R.LOG", "R.SFG"]


def _experiment(**observations):
    return SimpleNamespace(observations=observations, network=SimpleNamespace(node_labels=LABELS))


def _observation(**kwargs):
    return SimpleNamespace(**{"name": "obs", "reduce": None, "pipeline": None, "source": None, "dims": None, **kwargs})


def test_a_declared_dims_names_the_axes():
    """The whole point of the slot: a pipeline observation gets its axes from the recipe."""
    exp = _experiment(target_peak_frequencies=_observation(name="target_peak_frequencies", dims=["node"]))
    assert observation_dims(exp)["target_peak_frequencies"] == ("node",)


def test_an_observation_that_declares_nothing_stays_absent():
    """Absent, not guessed — the container then falls back to its positional names for that one alone."""
    exp = _experiment(mystery=_observation(name="mystery"))
    assert "mystery" not in observation_dims(exp)


def test_a_declared_axis_reaches_the_array_through_its_wrapper():
    """A pipeline observation arrives wrapped, and labelling only bare arrays left exactly those unlabelled."""
    wrapped = ObservationResult(ys=np.arange(4, dtype=float))
    res = SimulationResult(
        observations={"peak_frequencies": wrapped},
        nodes=LABELS,
        observation_dims={"peak_frequencies": ("node",)},
    )
    got = res.observations["peak_frequencies"].data
    assert got.dims == ("node",)
    assert list(got.coords["node"].values) == LABELS


def test_the_wrapper_keeps_its_other_outputs():
    """A pipeline produces several named outputs; binding axes must not cost the ones that are not the array."""
    wrapped = ObservationResult(ys=np.arange(4, dtype=float), frequencies=np.arange(4, dtype=float))
    res = SimulationResult(
        observations={"peak_frequencies": wrapped},
        nodes=LABELS,
        observation_dims={"peak_frequencies": ("node",)},
    )
    assert "frequencies" in res.observations["peak_frequencies"]


def test_a_declaration_of_the_wrong_rank_is_refused_by_name():
    """Neither reshape the data nor ignore the author: a `dims:` that contradicts its value is the defect this slot exists to end, so it is named and raised rather than dropped."""
    with pytest.raises(ValueError, match="peak_frequencies.*declares 1 axis.*has 2"):
        SimulationResult(
            observations={"peak_frequencies": np.zeros((4, 3))},
            nodes=LABELS,
            observation_dims={"peak_frequencies": ("node",)},
        )


def test_a_value_with_no_axes_is_not_a_contradiction():
    """A reduction that collapsed every axis leaves nothing to name, so a declaration on it is inapplicable rather than wrong."""
    res = SimulationResult(
        observations={"peak_frequency": np.float64(10.0)},
        nodes=LABELS,
        observation_dims={"peak_frequency": ("node",)},
    )
    assert not hasattr(res.observations["peak_frequency"], "dims")
