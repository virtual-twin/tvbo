"""A network written out, with ``to_yaml()`` or as part of a saved result, and read back from that file is the network that was written.

A curated network, and any network whose ``data_file`` is another network's sidecar, keeps that sidecar as its ``data_file`` in the dump, so the file read back names a sidecar, not a companion of its own. Every network here is synthetic or ships with tvbo.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import yaml

from tvbo import Network, SimulationExperiment

CURATED = "tvbo:networks/tpl-MNI152NLin2009cAsym_cohort-HCPYA_rec-dTOR_atlas-Lobar_desc-SC_relmat"
W = np.array([[0.0, 0.2, 0.5], [0.2, 0.0, 0.9], [0.5, 0.9, 0.0]])
RING = {
    "id": 1,
    "label": "ringlen",
    "dynamics": {
        "name": "Relaxation",
        "system_type": "continuous",
        "output": ["x"],
        "coupling_inputs": {"c": {}},
        "state_variables": {"x": {"equation": {"rhs": "1 - x + c"}, "initial_value": 0.1, "coupling_variable": True}},
    },
    "network": {
        "graph_generator": {"name": "ring", "type": "Cycle"},
        "number_of_nodes": 4,
        "edges": [{"label": "weight"}, {"label": "length"}],
        "coupling": {"c": {"incoming_states": ["x"], "pre_expression": {"rhs": "x"}, "post_expression": {"rhs": "gx"}}},
    },
    "integration": {"method": "euler", "step_size": 1.0, "duration": 5.0, "transient_time": 0.0, "unit": "ms"},
}


def _dense(net, layer):
    return np.asarray(net.matrix(layer, format="dense"))


def test_a_curated_network_written_out_reads_back_with_its_layers(tmp_path):
    written = Network(iri=CURATED)
    (tmp_path / "curated.yaml").write_text(written.to_yaml())
    back = Network.from_file(str(tmp_path / "curated.yaml"))
    assert back.node_labels == written.node_labels
    for layer in ("weight", "length"):
        assert _dense(written, layer).max() > 0
        np.testing.assert_array_equal(_dense(back, layer), _dense(written, layer))


def test_a_network_whose_data_file_is_a_sidecar_reads_back_from_its_dump(tmp_path):
    Network.from_matrix(weights=W, labels=["A", "B", "C"]).save(tmp_path / "saved.yaml")
    written = Network(data_file=str(tmp_path / "saved.yaml"))
    (tmp_path / "dumped.yaml").write_text(written.to_yaml())
    np.testing.assert_array_equal(_dense(Network.from_file(str(tmp_path / "dumped.yaml")), "weight"), W)


def test_a_sidecar_that_names_itself_as_its_data_file_is_refused(tmp_path):
    (tmp_path / "itself.yaml").write_text(
        "tvbo_class: tvbo:Network\nnodes: [{id: 0, label: A}, {id: 1, label: B}]\ndata_file: itself.yaml\n"
    )
    with pytest.raises(ValueError, match=r"itself\.yaml names itself as its `data_file`"):
        Network.from_file(str(tmp_path / "itself.yaml"))


def test_a_declared_weight_the_companion_cannot_give_raises_rather_than_serving_zeros(tmp_path):
    """A weight the sidecar declares and the companion does not hold, or a companion that cannot be read, is an error; zero weights stand in only for a network that declares no layer."""
    Network.from_matrix(weights=W, labels=["A", "B", "C"]).save(tmp_path / "saved.yaml")
    with h5py.File(tmp_path / "saved.h5", "a") as companion:
        del companion["edges/weight"]
    with pytest.raises(KeyError, match=r"declares the layer 'weight', and saved\.h5 does not hold it"):
        _dense(Network.from_file(str(tmp_path / "saved.yaml")), "weight")
    Network.from_matrix(weights=W, labels=["A", "B", "C"]).save(tmp_path / "saved.yaml")
    net = Network.from_file(str(tmp_path / "saved.yaml"))
    (tmp_path / "saved.h5").write_bytes(b"not an HDF5 file")
    with pytest.raises(OSError):
        _dense(net, "weight")
    np.testing.assert_array_equal(
        _dense(Network(nodes=[{"id": 0, "label": "A"}, {"id": 1, "label": "B"}]), "weight"), np.zeros((2, 2))
    )


def test_a_save_refuses_a_declared_weight_the_companion_does_not_hold(tmp_path):
    """Read back from a companion that has lost its declared weight, a network is refused at save with the error `matrix()` raises for that weight, and nothing is written."""
    Network.from_matrix(weights=W, lengths=10 * W, labels=["A", "B", "C"]).save(tmp_path / "saved.yaml")
    with h5py.File(tmp_path / "saved.h5", "a") as companion:
        del companion["edges/weight"]
    (tmp_path / "again").mkdir()
    with pytest.raises(KeyError, match=r"declares the layer 'weight', and saved\.h5 does not hold it"):
        Network.from_file(str(tmp_path / "saved.yaml")).save(tmp_path / "again" / "saved.yaml")
    assert not any((tmp_path / "again").iterdir())


def test_a_layer_held_in_a_damaged_group_raises_naming_it_when_read_and_when_saved(tmp_path):
    """A layer whose group the companion holds without its ``data`` raises ``OSError`` naming the layer and the companion, chained to h5py's ``KeyError``, when `matrix()` or `array()` reads it and when the network is saved; as that ``KeyError`` it read as absent to every caller that takes one for a miss."""
    Network.from_matrix(weights=W, lengths=10 * W, labels=["A", "B", "C"]).save(tmp_path / "saved.yaml")
    with h5py.File(tmp_path / "saved.h5", "a") as companion:
        del companion["edges/length/data"]
    message = r"saved\.h5 holds the layer 'length', and it cannot be read: .*'data'"
    for read in (lambda net: net.matrix("length"), lambda net: net.array("length")):
        with pytest.raises(OSError, match=message) as raised:
            read(Network.from_file(str(tmp_path / "saved.yaml")))
        assert isinstance(raised.value.__cause__, KeyError)
    (tmp_path / "again").mkdir()
    with pytest.raises(OSError, match=message):
        Network.from_file(str(tmp_path / "saved.yaml")).save(tmp_path / "again" / "saved.yaml")
    assert not any((tmp_path / "again").iterdir())


def test_a_companion_that_is_not_an_array_store_raises_for_every_layer(tmp_path):
    """A companion of a kind the store does not read (an ``.npz``) raises ``OSError`` for every layer, when the network is read or when the layer is, as one that cannot be opened does, rather than reading as a companion that lacks the layer."""
    Network.from_matrix(weights=W, lengths=10 * W, labels=["A", "B", "C"]).save(tmp_path / "saved.yaml")
    (tmp_path / "saved.h5").rename(tmp_path / "saved.npz")
    sidecar = tmp_path / "saved.yaml"
    text = sidecar.read_text()
    assert text.count("data_file: saved.h5") == 1
    sidecar.write_text(text.replace("data_file: saved.h5", "data_file: saved.npz"))
    for layer in ("weight", "length"):
        with pytest.raises(OSError, match=r"saved\.npz is not an array store"):
            Network.from_file(str(sidecar)).matrix(layer)


def test_a_ring_that_declares_a_length_it_does_not_hold_runs_and_saves_again_from_its_saved_result(tmp_path):
    """A ring that declares `weight` and `length` and has no lengths saves a companion holding the weights only; read back, its length is absent as it was in memory, so the experiment runs and saves again, and so does the experiment read back from that second save."""
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump(RING, sort_keys=False))
    experiment = SimulationExperiment.from_file(str(tmp_path / "exp.yaml"))
    assert experiment.network.lengths_matrix is None
    weights = _dense(experiment.network, "weight")
    for generation in ("first", "second"):
        experiment.run("tvboptim").save(tmp_path / generation)
        (companion,) = (tmp_path / generation).glob("*_network.h5")
        with h5py.File(companion, "r") as stored:
            assert "edges/weight" in stored and "edges/length" not in stored
        (recipe,) = (tmp_path / generation).glob("*_result.yaml")
        experiment = SimulationExperiment.from_file(str(recipe))
        assert experiment.max_delay == 0.0 and experiment.network.lengths_matrix is None
        np.testing.assert_array_equal(_dense(experiment.network, "weight"), weights)
    assert np.isfinite(np.asarray(experiment.run("tvboptim").integration.data)).all()
