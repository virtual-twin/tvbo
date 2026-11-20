"""A spatial average reports one mean per area, its areas numbered from 0 whatever labels the mask uses.

tvb's generated ``SpatialAverage`` builds its mask from the connectivity's ``cortical`` flags, which tvbo leaves unset (``None``), and every node then counts as cortical. An all-cortical mask is all 1s: one area labelled 1, which must be column 0 of a one-column assignment matrix, not column 1. The probe of ``test_delay_history_from_opening`` reads the result in closed form: ``y(t) = t a W x(0)`` per node at the sample stamped ``t``, and an area holds the mean of its nodes' ``y``. jax computes the curated pipeline ``spatial_mean @ X``, in which nothing supplies ``spatial_mean``; the step is refused by name.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from tvbo import ExperimentResult, SimulationExperiment

from .test_delay_history_from_opening import DT, DURATION, RTOL, A, W
from .test_tvb_delayed_clock import _undelayed

PERIOD = 3 * DT
"""A sample every three steps."""

OBSERVATION = {
    "Raw": {"iri": "tvbo:observation_models/raw"},
    "Spatial": {"iri": "tvbo:observation_models/spatial_average", "period": PERIOD},
}
"""The curated spatial average, named so that tvb's generated monitor class is ``Spatial``, beside the curated raw, so that the average is a keyed observation on the run path and through ``from_tvb``, which takes the first monitor as the integration series."""

AREAS = {
    "unset": (None, [[0, 1]]),
    "all-cortical": ([True, True], [[0, 1]]),
    "two-area": ([True, False], [[1], [0]]),
}
"""Per network, the ``cortical`` flags it carries (None: tvbo's, unset) and the nodes of each area in area order: one area of both nodes, or the non-cortical node 1 as area 0 and the cortical node 0 as area 1."""


def _spec():
    """The undelayed probe with the curated spatial average."""
    spec = copy.deepcopy(_undelayed())
    spec["observations"] = copy.deepcopy(OBSERVATION)
    return spec


@pytest.mark.backend_tvb
@pytest.mark.parametrize("network", list(AREAS))
def test_a_spatial_average_holds_each_area_s_mean(network):
    """Sampled at 3, 6, 9 and 12 ms, each area holds the mean ``y`` of its nodes, on an all-cortical network and on one of two areas."""
    pytest.importorskip("tvb")
    cortical, areas = AREAS[network]
    spec = _spec()
    sim = SimulationExperiment(**spec).execute("tvb")
    if cortical is not None:
        sim.connectivity.cortical = np.array(cortical)
        next(m for m in sim.monitors if type(m).__name__ == "Spatial").config_for_sim(sim)
    result = ExperimentResult.from_tvb(sim, transient_time=0.0)
    data = result.integration.observations["Spatial"].data
    if "variable" in data.dims:
        data = data.sel(variable="y")
    stamps = PERIOD * np.arange(1, int(DURATION / PERIOD) + 1)
    np.testing.assert_allclose(data["time"].values, stamps, rtol=0, atol=1e-9)
    opening = np.array([node["state"]["x"]["value"] for node in spec["network"]["nodes"]])
    per_node = stamps[:, None] * A * (W @ opening)[None, :]
    expected = np.stack([per_node[:, nodes].mean(axis=1) for nodes in areas], axis=1)
    values = data.transpose("time", ...).values.reshape(len(stamps), -1)
    np.testing.assert_allclose(values, expected, rtol=RTOL, atol=RTOL)


@pytest.mark.backend_jax
def test_jax_refuses_a_spatial_mean_nothing_supplies():
    """The curated pipeline reads ``spatial_mean`` beside ``X`` with no value for it, and jax names it instead of binding it to the samples."""
    with pytest.raises(ValueError, match=r"jax: the pipeline step 'spatial_average'.*'spatial_mean'"):
        SimulationExperiment(**_spec()).render_code("jax")
