"""A save writes files and leaves the network as it was: a record of the network written after the save, an experiment's export among them, reads back as one written before it.

Every network is synthetic: three nodes with a weight and a length layer, saved once as HDF5 and named by the absolute path of that sidecar, so any record of it resolves wherever it is written.
"""

from __future__ import annotations

import numpy as np
import pytest

from tvbo import Network
from tvbo.classes.experiment import SimulationExperiment
from tvbo.data.types import AlgorithmResult, ExperimentResult
from tvbo.utils import Bunch

W = np.array([[0.0, 0.5, 0.25], [0.5, 0.0, 0.75], [0.25, 0.75, 0.0]])
L = np.array([[0.0, 10.0, 20.0], [10.0, 0.0, 30.0], [20.0, 30.0, 0.0]])


def _source(tmp_path) -> Network:
    """The network named by the absolute path of the HDF5 save of its weight and length layers."""
    net = Network(nodes=[{"id": i, "label": k} for i, k in enumerate("ABC")], edges=[{"label": "weight"}, {"label": "length"}])
    net.set_matrix("weight", W)
    net.set_matrix("length", L)
    (tmp_path / "source").mkdir()
    net.save(tmp_path / "source" / "n.yaml")
    return Network(data_file=str(tmp_path / "source" / "n.yaml"))


@pytest.mark.parametrize("binary_format", ["h5", "zarr", "csv"])
def test_a_save_leaves_the_network_s_record_as_it_was(tmp_path, binary_format):
    """The network's dump after a save in any format is its dump before: the save names its own companion in the sidecar it writes, not on the network."""
    net = _source(tmp_path)
    before = net.to_yaml()
    (tmp_path / "saved").mkdir()
    net.save(tmp_path / "saved" / "n.yaml", binary_format=binary_format)
    assert net.to_yaml() == before


def test_a_dump_after_a_csv_save_reads_back_every_layer(tmp_path):
    """``to_yaml()`` after a CSV save, written beside it, reads back the length layer as well as the weights; it named the first CSV as the network's ``data_file``, which a reader takes for a one-matrix CSV holding the weights alone."""
    net = _source(tmp_path)
    (tmp_path / "saved").mkdir()
    net.save(tmp_path / "saved" / "n.yaml", binary_format="csv")
    (tmp_path / "saved" / "dump.yaml").write_text(net.to_yaml())
    back = Network.from_file(str(tmp_path / "saved" / "dump.yaml"))
    np.testing.assert_allclose(back.matrix("weight", format="dense"), W)
    np.testing.assert_allclose(back.matrix("length", format="dense"), L)


def test_an_export_after_a_save_reads_back_with_the_weights_that_ran(tmp_path):
    """``ExperimentResult.save`` freezes the experiment's network into the results folder, and ``export`` after it writes an experiment that reads back with the weights that ran; it named the frozen companion by its bare name, which is not beside the export."""
    exp = SimulationExperiment(network=_source(tmp_path))
    result = ExperimentResult(algorithms={"fic": AlgorithmResult(name="fic", history=Bunch(J_i=np.zeros((2, 3))))}, source=exp)
    result.save(str(tmp_path / "results"), compress=False, record_only=False)
    assert any((tmp_path / "results").glob("*_network.h5")), "precondition lost: the save froze no network"
    result.export(tmp_path / "export")
    exported = next((tmp_path / "export").rglob("*_experiment.yaml"))
    back = SimulationExperiment.from_file(str(exported))
    np.testing.assert_allclose(back.network.matrix("weight", format="dense"), W)
