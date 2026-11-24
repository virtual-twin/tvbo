"""The TVB connectivity tvbo builds carries the cortical flags, areas, hemispheres and orientations its nodes carry, as `from_tvb` stores them.

`Network.from_tvb` stores a TVB connectivity's ``cortical``, ``areas``, ``hemispheres`` and ``orientations`` as the node parameters ``cortical``, ``area``, ``hemisphere`` and ``orientation``, and `to_tvb` built the connectivity for a run without them: ``cortical``, ``areas`` and ``orientations`` unset and ``hemispheres`` guessed from the region labels. A network imported with subcortical regions then ran as all cortical, and its spatial average reported one area. The probe of ``test_delay_history_from_opening`` reads the average in closed form, as ``test_spatial_average_areas`` does. An ``orientation`` is three numbers per node, which the run reads past as no per-node scalar of the dynamics (`get_node_param_overrides`).
"""

from __future__ import annotations

import numpy as np
import pytest

from tvbo import Network, SimulationExperiment

from .test_delay_history_from_opening import W
from .test_spatial_average_areas import _assert_area_means, _spec

FIELDS = {
    "cortical": [True, False],
    "areas": [2.0, 3.0],
    "hemispheres": [False, True],
    "orientations": [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
}
"""The TVB connectivity's per-region fields: node 1 subcortical and in the right hemisphere, which the labels ``r0`` and ``r1`` would not give (TVB reads both as right), and a unit vector per node."""


def _connectivity():
    """A two-region TVB connectivity on the probe's weights, carrying `FIELDS`."""
    from tvb.datatypes.connectivity import Connectivity

    conn = Connectivity(
        weights=W,
        tract_lengths=np.zeros_like(W),
        centres=np.array([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        region_labels=np.array(["r0", "r1"]),
        speed=np.array([1.0]),
        **{field: np.array(values) for field, values in FIELDS.items()},
    )
    conn.configure()
    return conn


@pytest.mark.backend_tvb
@pytest.mark.parametrize("field", list(FIELDS))
def test_a_tvb_connectivity_s_regions_survive_import_and_export(field):
    """Imported with `from_tvb` and built back for a run, the connectivity holds the field it was imported with."""
    pytest.importorskip("tvb")
    exported = Network.from_tvb(_connectivity()).execute("tvb")
    np.testing.assert_array_equal(getattr(exported, field), FIELDS[field])


@pytest.mark.backend_tvb
def test_an_imported_network_s_spatial_average_reports_both_areas():
    """The probe's nodes carry the parameters `from_tvb` stored, and the spatial average holds the subcortical node 1 as area 0 and the cortical node 0 as area 1."""
    pytest.importorskip("tvb")
    spec = _spec()
    for node, imported in zip(spec["network"]["nodes"], Network.from_tvb(_connectivity()).nodes, strict=True):
        node["parameters"] = {name: {"name": name, "value": p.value} for name, p in imported.parameters.items()}
    _assert_area_means(spec, SimulationExperiment(**spec).run("tvb"), [[1], [0]])
