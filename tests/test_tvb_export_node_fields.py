"""The exported tvb script loads the connectivity a tvbo run builds: the same centres, labels, cortical flags, areas, hemispheres and orientations.

`run("tvb")` builds its connectivity with `to_tvb`, which takes the per-region fields from the network (`tvb_node_fields`). The standalone script's `load_connectivity` read the weights and lengths CSVs and set zero centres and generated labels, and left the cortical flags, areas and orientations unset and the hemispheres to TVB's guess from those labels, so the script ran a network with subcortical regions as all cortical. The script now embeds the fields the network gives.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from tvbo import SimulationExperiment

from .test_tvb_delayed_clock import _undelayed

PER_NODE = [
    ({"x": -10.0, "y": 0.0, "z": 0.0}, {"cortical": 1.0, "area": 2.0, "hemisphere": 0.0, "orientation": [-1.0, 0.0, 0.0]}),
    ({"x": 10.0, "y": 0.0, "z": 0.0}, {"cortical": 0.0, "area": 3.0, "hemisphere": 1.0, "orientation": [0.0, 1.0, 0.0]}),
]
"""Each probe node's position and the node parameters `from_tvb` stores: node 1 subcortical and in the right hemisphere."""

FIELDS = ["centres", "region_labels", "cortical", "areas", "hemispheres", "orientations"]
"""The per-region ``Connectivity`` fields a run and the script must agree on."""


def _experiment():
    """The undelayed probe, its nodes placed and carrying `PER_NODE`'s parameters."""
    spec = copy.deepcopy(_undelayed())
    for node, (position, parameters) in zip(spec["network"]["nodes"], PER_NODE, strict=True):
        node["position"] = position
        node["parameters"] = {name: {"name": name, "value": value} for name, value in parameters.items()}
    return SimulationExperiment(**spec)


@pytest.fixture(scope="module")
def connectivities(tmp_path_factory):
    """The connectivity a tvb run of the probe builds, and the one its exported script loads from the run's weights and lengths."""
    pytest.importorskip("tvb")
    experiment = _experiment()
    namespace = {"__name__": "exported"}
    exec(experiment.render_code("tvb"), namespace)
    run = experiment.network.execute("tvb")
    directory = tmp_path_factory.mktemp("export")
    np.savetxt(directory / "weights.csv", run.weights, delimiter=",")
    np.savetxt(directory / "lengths.csv", run.tract_lengths, delimiter=",")
    return run, namespace["load_connectivity"](directory / "weights.csv", directory / "lengths.csv")


@pytest.mark.backend_tvb
@pytest.mark.parametrize("field", FIELDS)
def test_the_exported_script_loads_the_run_s_connectivity_field(field, connectivities):
    """The script's connectivity holds the run's value of *field*."""
    run, exported = connectivities
    np.testing.assert_array_equal(getattr(exported, field), getattr(run, field))
