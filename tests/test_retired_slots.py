"""Slots tvbo retired are refused at load, naming where their meaning is written now, and a subnetwork reaches its parent as ``parent`` only.

The hierarchical coupling attributes (``inner_coupling``, ``region_mapping``, ``regional_connectivity``, ``aggregation``, ``distribution``) and the ``RegionMapping`` class they used are gone: a coupling on a subnetwork, ``node_mapping`` on the network and the coupling functions of up and down edges say the same. No retired slot has a counterpart on its own class, so a record that still carries one is refused by the dialect on every constructor, rather than dropped. The path spellings ``..`` and ``../..`` for the parent and its parent are refused the same way, at the edge that carries them, and so is a down edge the reservoir flattener reads from a raw recipe.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import tvbo.datamodel.pydantic as pydantic_model
import tvbo.datamodel.schema as dataclass_model
from tvbo import Network, SimulationExperiment
from tvbo.multiscale import _project_weights
from tvbo.utils import pydantic_loader

RETIRED = {
    "inner_coupling": ({"name": "Inner"}, "write the coupling on the subnetwork"),
    "region_mapping": ({"n_regions": 2}, "write `node_mapping` on the network"),
    "regional_connectivity": ({"label": "regions"}, "the parent network's connectivity is the regional coupling's graph"),
    "aggregation": ("mean", "write the aggregation as an up edge's coupling"),
    "distribution": ("broadcast", "write the distribution as a down edge's coupling"),
}

SPEC = """
label: Two nodes
dynamics:
  name: Leak
  state_variables:
    x: {{equation: {{rhs: "-x + c"}}, initial_value: 0.0}}
  coupling_inputs: {{c: {{}}}}
network:
  label: Pair
  number_of_nodes: 2
  nodes: [{{id: 0, label: A, dynamics: Leak}}, {{id: 1, label: B, dynamics: Leak}}]
  edges:
    - {{source: 0, target: 1, parameters: {{weight: {{value: 0.3}}}}, source_var: x, target_var: c, directed: true}}
  coupling:
    c:
      name: Linear
      {slot}: {value}
integration: {{method: euler, step_size: 1.0, duration: 4.0}}
"""


@pytest.mark.parametrize("slot", RETIRED, ids=list(RETIRED))
def test_a_retired_coupling_slot_is_refused_naming_where_it_is_written_now(slot):
    value, instead = RETIRED[slot]
    message = rf"Coupling\.{slot} was retired in tvbo 0\.9\.0: {instead}"
    with pytest.raises(ValueError, match=message):
        dataclass_model.Coupling(name="Linear", **{slot: value})
    with pytest.raises(ValueError, match=message):
        SimulationExperiment.from_string(SPEC.format(slot=slot, value=repr(value) if isinstance(value, str) else value))
    with pytest.raises(ValueError, match=message):
        pydantic_loader.validate({"name": "Linear", slot: value}, "Coupling")


def test_the_region_mapping_class_is_gone_from_both_datamodels():
    assert not hasattr(dataclass_model, "RegionMapping") and not hasattr(pydantic_model, "RegionMapping")


def test_an_observation_keeps_its_aggregation():
    assert dataclass_model.Observation(name="mean_x", source=["x"], aggregation="mean").aggregation == "mean"


PARENT_PATHS = [("..", "parent"), ("../..", "parent/parent")]


@pytest.mark.parametrize("path, spelled", PARENT_PATHS, ids=["parent", "grandparent"])
@pytest.mark.parametrize("slot", ["source_network", "target_network"])
def test_the_parent_network_is_spelled_parent(slot, path, spelled):
    assert str(getattr(Network(label="units", edges=[{"label": "drive", slot: spelled}]).edges[0], slot)) == spelled
    with pytest.raises(ValueError, match=rf"Edge 'drive': `{slot}: {re.escape(path)}` is spelled `{slot}: {spelled}`"):
        Network(label="units", edges=[{"label": "drive", slot: path}])


DRIVE = {
    "label": "sc_drive",
    "target_var": "sc_drive",
    "coupling": {
        "parameters": {
            "W_in": {
                "shape": "(n,)",
                "distribution": {"name": "Uniform", "seed": 13, "parameters": {"lo": {"value": -1.0}, "hi": {"value": 1.0}}},
            }
        }
    },
}


def test_the_reservoir_flattener_draws_the_projection_of_a_down_edge_from_parent():
    np.testing.assert_allclose(
        _project_weights({"edges": [{**DRIVE, "source_network": "parent"}]}, 6),
        [0.7296, 0.7106, 0.622, -0.4771, -0.8456, 0.8929],
        atol=1e-4,
    )


@pytest.mark.parametrize("path, spelled", PARENT_PATHS, ids=["parent", "grandparent"])
def test_the_reservoir_flattener_refuses_a_dotted_parent(path, spelled):
    with pytest.raises(
        ValueError, match=rf"Edge 'sc_drive': `source_network: {re.escape(path)}` is spelled `source_network: {spelled}`"
    ):
        _project_weights({"edges": [{**DRIVE, "source_network": path}]}, 6)
