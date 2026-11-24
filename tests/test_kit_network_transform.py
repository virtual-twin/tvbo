"""A weight transform whose callable takes `network` runs in a rendered tvboptim script, in a kit and in a lanes run, as it runs in process.

The runtime calls such a callable with the network it transforms (`Network._apply_transform`). The generated `create_network` takes that network as `transform_network`, and every caller of the module hands it over (`tvbo.adapters.tvboptim.handed_network`, read by `SimulationExperiment.run` and `tvbo.run.lanes.build_network`), so a frozen script calls the callable with the network the runtime would: the same nodes, labels and layers. The callable here reads all three. Run on its own, the script has no network to hand over and says which run does. A renamed copy of the callable keys the tests on nothing by name.
"""

import re

import numpy as np
import pytest

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment  # noqa: E402
from tvbo.run import lanes  # noqa: E402

pytestmark = pytest.mark.backend_tvboptim

_CALLABLE = '''
import numpy as np


def {name}(M, network, gain=1.0):
    """Each hemisphere's own edges, scaled by the number in the target node's label, over one plus the tract length."""
    labels = [node.label for node in network.nodes]
    left = np.array([label.startswith("L_") for label in labels])
    number = np.array([float(label.split("_")[1]) for label in labels])
    lengths = np.asarray(network.matrix("length", format="dense"), dtype=np.float64)
    W = np.asarray(M, dtype=np.float64) * (left[:, None] == left[None, :])
    return gain * W * number[:, None] / (1.0 + lengths)
'''

LABELS = ["L_3", "R_1", "L_4", "R_2"]


def _spec(module: str, name: str) -> dict:
    """A four-node Kuramoto network whose weight transform is *module*.*name*, every ordered pair an edge of its own weight and length."""
    edges = [
        {
            "source": i,
            "target": j,
            "parameters": {"weight": {"value": 1.0 + i + 4.0 * j}, "length": {"value": 10.0 + i + 2.0 * j}},
        }
        for i in range(4)
        for j in range(4)
        if i != j
    ]
    return {
        "id": 1,
        "label": "network-reading transform",
        "dynamics": {
            "name": "MiniKuramoto",
            "system_type": "continuous",
            "output": ["theta"],
            "parameters": {"K": {"value": 1.0}},
            "coupling_inputs": {"c": {}},
            "state_variables": {"theta": {"equation": {"rhs": "K * c"}, "initial_value": 0.1, "coupling_variable": True}},
        },
        "network": {
            "number_of_nodes": len(LABELS),
            "nodes": [{"id": i, "label": label} for i, label in enumerate(LABELS)],
            "edges": edges,
            "transforms": [
                {
                    "name": "weight",
                    "callable": {"module": module, "name": name},
                    "arguments": {"gain": {"name": "gain", "value": 2.0}},
                }
            ],
            "coupling": {
                "c": {
                    "delayed": False,
                    "incoming_states": ["theta"],
                    "local_states": ["theta"],
                    "pre_expression": {"rhs": "sin(x_j - x_i)"},
                    "post_expression": {"rhs": "gx"},
                }
            },
        },
        "integration": {"method": "heun", "step_size": 0.1, "duration": 1.0, "transient_time": 0.0, "unit": "s"},
    }


@pytest.fixture(
    params=[("hemisphere_rules", "within_hemisphere"), ("block_layers", "kept_blocks")], ids=["callable", "renamed"]
)
def experiment(request, tmp_path, monkeypatch):
    """The experiment, its transform callable written to a module on the path as a kit's `code/` is."""
    module, name = request.param
    (tmp_path / f"{module}.py").write_text(_CALLABLE.format(name=name))
    monkeypatch.syspath_prepend(str(tmp_path))
    return SimulationExperiment(**_spec(module, name))


def _script_weights(experiment) -> np.ndarray:
    """The weights the network of the rendered script integrates, the script run as `tvbo run --rendered` runs a frozen one."""
    code = experiment.render_code("tvboptim")
    result = experiment.run(format="tvboptim", rendered_code=code)
    return np.asarray(result.network.graph.weights)


def test_a_rendered_script_hands_the_callable_the_network_of_its_run(experiment):
    expected = np.asarray(experiment.network.matrix("weight", format="dense"))
    raw = np.asarray(experiment.network.matrix("weight", format="dense", apply_transforms=False))
    assert not np.allclose(expected, raw), "the transform changed nothing: test is vacuous"
    np.testing.assert_array_equal(_script_weights(experiment), expected)


def test_a_kit_hands_the_callable_the_network_it_froze(experiment, tmp_path):
    expected = np.asarray(experiment.network.matrix("weight", format="dense"))
    kit = tmp_path / "kit"
    recipe = experiment.freeze_yaml(kit, network_stem="network")
    (kit / "experiment.yaml").write_text(recipe)
    frozen = SimulationExperiment.from_file(str(kit / "experiment.yaml"))
    assert [node.label for node in frozen.network.nodes] == LABELS
    np.testing.assert_array_equal(_script_weights(frozen), expected)


def test_a_lanes_run_hands_the_callable_the_network_of_its_experiment(experiment):
    expected = np.asarray(experiment.network.matrix("weight", format="dense"))
    built = lanes.build_network(experiment, experiment.execute("tvboptim"))
    np.testing.assert_array_equal(np.asarray(built.graph.weights), expected)


def test_the_script_run_on_its_own_names_the_run_that_hands_it_the_network(experiment, tmp_path):
    code = experiment.render_code("tvboptim")
    said = re.escape(
        "this script's weight transform reads the network; run it through `tvbo run`, which hands it the frozen network"
    )
    with pytest.raises(SystemExit, match=said):
        exec(
            compile(code, str(tmp_path / "script.py"), "exec"),
            {"__name__": "__main__", "__file__": str(tmp_path / "script.py")},
        )
