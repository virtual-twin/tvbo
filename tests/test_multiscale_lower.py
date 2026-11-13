"""One seam lowers a layered network before a backend sees it, and a network without subnetworks passes through it untouched.

:func:`tvbo.multiscale.lower` runs where a render and a run check what the backend integrates. It routes a layered network to one strategy, and refuses it while tvbo builds none, so a subnetwork is never left out of a run without a word, as it was when a backend read only the parent nodes.
"""

from __future__ import annotations

import pytest
import yaml
from typer.testing import CliRunner

from tvbo import SimulationExperiment
from tvbo.classes.network import Network
from tvbo.cli import app
from tvbo.multiscale import HETEROGENEOUS, SUBNETWORK_TO_VECTOR, lower, strategy_for

SPEC = """
id: 1
label: Layered
dynamics:
  name: Leak
  parameters: {tau: {value: 5.0}}
  state_variables:
    x: {initial_value: 0.0, equation: {rhs: "-x / tau + c"}, coupling_variable: true}
  coupling_inputs: {c: {}}
  output: [x]
network:
  label: Regions
  number_of_nodes: 2
  NODES
  edges:
    - {source: 0, target: 1, parameters: {weight: {value: 0.3}}, source_var: x, target_var: c, directed: true}
integration: {method: euler, step_size: 0.1, duration: 1.0, unit: ms}
"""
UNITS = "node_template: {subnetwork: {label: Units, number_of_nodes: 3, node_template: {dynamics: Leak}}}"


def _experiment(nodes=UNITS):
    return SimulationExperiment.from_string(SPEC.replace("NODES", nodes))


def test_lower_returns_a_network_without_subnetworks_unchanged():
    experiment = _experiment("nodes: [{id: 0, label: A}, {id: 1, label: B}]")
    assert strategy_for(experiment.network) is None
    assert lower(experiment) is experiment


@pytest.mark.parametrize(
    "nodes, strategy",
    [
        (
            {"node_template": {"subnetwork": {"number_of_nodes": 3, "node_template": {"dynamics": "Leak"}}}},
            SUBNETWORK_TO_VECTOR,
        ),
        (
            {
                "nodes": [
                    {"id": 0, "subnetwork": {"nodes": [{"id": 0, "dynamics": "Leak"}, {"id": 1, "dynamics": "Leak"}]}},
                    {"id": 1, "subnetwork": {"nodes": [{"id": 0, "dynamics": "Leak"}]}},
                ]
            },
            SUBNETWORK_TO_VECTOR,
        ),
        (
            {"nodes": [{"id": 0, "subnetwork": {"nodes": [{"id": 0, "dynamics": "Leak"}, {"id": 1, "dynamics": "Spiking"}]}}]},
            HETEROGENEOUS,
        ),
        (
            {
                "nodes": [
                    {"id": 0, "subnetwork": {"nodes": [{"id": 0, "dynamics": "Leak"}]}},
                    {"id": 1, "subnetwork": {"nodes": [{"id": 0, "dynamics": "Spiking"}]}},
                ]
            },
            HETEROGENEOUS,
        ),
    ],
    ids=["one-template", "listed-alike", "units-differ", "regions-differ"],
)
def test_a_layered_network_is_routed_to_one_strategy(nodes, strategy):
    assert strategy_for(Network(label="Regions", number_of_nodes=2, **nodes)) == strategy


REFUSED = r"network 'Regions' places 'Units' on node '\w+', 'Units' on node '\w+'; its subnetworks are lowered by the 'subnetwork_to_vector' strategy, which tvbo does not build"


@pytest.mark.backend_tvboptim
def test_a_network_whose_subnetworks_no_strategy_lowers_is_refused(tmp_path):
    pytest.importorskip("tvboptim")
    experiment = _experiment()
    with pytest.raises(ValueError, match=REFUSED):
        experiment.render_code("tvboptim")
    with pytest.raises(ValueError, match=REFUSED):
        experiment.run("tvboptim")
    assert yaml.safe_load(experiment.render("yaml"))["network"]["node_template"]["subnetwork"]["label"] == "Units"
    spec = tmp_path / "layered.yaml"
    spec.write_text(SPEC.replace("NODES", UNITS))
    kit = tmp_path / "kit"
    emitted = CliRunner().invoke(app, ["workflow", "snakemake", str(spec), "--backend", "tvboptim", "-o", str(kit)])
    assert emitted.exit_code == 0, emitted.stdout
    assert not (kit / "scripts").exists()
    (frozen,) = kit.glob("spec/*/experiment.yaml")
    with pytest.raises(ValueError, match=REFUSED):
        SimulationExperiment.from_file(frozen).run("tvboptim")
