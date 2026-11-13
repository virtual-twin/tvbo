"""A record written into another folder than the one its network was read from names the network's files relative to its own folder, or by absolute path when the two folders share no directory below the root, so it reads back with the same matrices.

This holds for a network read from its sidecar and for an experiment naming its network by ``data_file``, for HDF5, Zarr and CSV companions, whether or not the network was saved or frozen elsewhere first.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from tvbo import Network, SimulationExperiment

W = np.array([[0.0, 0.5, 0.25], [0.5, 0.0, 0.75], [0.25, 0.75, 0.0]])


def _saved(folder: Path, binary_format: str) -> Path:
    """A three-node network saved into *folder* with companions in *binary_format*; its sidecar."""
    folder.mkdir(parents=True)
    net = Network(nodes=[{"id": i, "label": k} for i, k in enumerate("ABC")], edges=[{"label": "weight"}])
    net.set_matrix("weight", W)
    net.save(folder / "n.yaml", binary_format=binary_format)
    return folder / "n.yaml"


def _experiment(sidecar: Path, ref: str) -> SimulationExperiment:
    """An experiment beside *sidecar* naming its network by ``data_file: <ref>``, read from its file."""
    (sidecar.parent / "e.yaml").write_text(yaml.safe_dump({"network": {"data_file": ref}}))
    return SimulationExperiment.from_file(str(sidecar.parent / "e.yaml"))


def _weight(net) -> np.ndarray:
    return np.asarray(net.matrix("weight", format="dense"))


REFERENCES = [("h5", "n.h5"), ("h5", "n.yaml"), ("zarr", "n.zarr"), ("zarr", "n.yaml"), ("csv", "n.yaml")]


@pytest.mark.parametrize(("binary_format", "ref"), REFERENCES)
def test_an_experiment_written_elsewhere_names_its_network_from_there(tmp_path, binary_format, ref):
    sidecar = _saved(tmp_path / "src", binary_format)
    exp = _experiment(sidecar, ref)
    (tmp_path / "out").mkdir()
    exp.to_yaml(str(tmp_path / "out" / "e.yaml"))
    named = yaml.safe_load((tmp_path / "out" / "e.yaml").read_text())["network"]
    assert named["data_file"] == os.path.join("..", "src", ref)
    assert not any(edge.get("data_file") for edge in named.get("edges") or [])
    assert exp.network.data_file == ref
    np.testing.assert_array_equal(_weight(SimulationExperiment.from_file(str(tmp_path / "out" / "e.yaml")).network), W)


@pytest.mark.parametrize("binary_format", ["h5", "zarr", "csv"])
def test_a_network_read_from_its_sidecar_and_written_elsewhere_reads_back(tmp_path, binary_format):
    """Its ``data_file`` and, for CSV companions, each CSV its records name are named from the new folder."""
    net = Network.from_file(str(_saved(tmp_path / "src", binary_format)))
    (tmp_path / "out").mkdir()
    net.to_yaml(str(tmp_path / "out" / "n.yaml"))
    text = (tmp_path / "out" / "n.yaml").read_text()
    assert "data_file: ../src/" in text and "data_file: n" not in text
    np.testing.assert_array_equal(_weight(Network.from_file(str(tmp_path / "out" / "n.yaml"))), W)


@pytest.mark.parametrize("first", ["save", "freeze"])
def test_a_record_written_elsewhere_after_a_save_or_a_freeze_reads_back(tmp_path, first):
    exp = _experiment(_saved(tmp_path / "src", "h5"), "n.h5")
    if first == "save":
        (tmp_path / "saved").mkdir()
        exp.network.save(tmp_path / "saved" / "net.yaml")
    else:
        exp.freeze_yaml(tmp_path / "frozen", network_stem="net")
    (tmp_path / "out").mkdir()
    exp.to_yaml(str(tmp_path / "out" / "e.yaml"))
    np.testing.assert_array_equal(_weight(SimulationExperiment.from_file(str(tmp_path / "out" / "e.yaml")).network), W)


def test_a_record_written_beside_its_files_names_them_as_read(tmp_path):
    exp = _experiment(_saved(tmp_path / "src", "h5"), "n.h5")
    exp.to_yaml(str(tmp_path / "src" / "again.yaml"))
    assert yaml.safe_load((tmp_path / "src" / "again.yaml").read_text())["network"]["data_file"] == "n.h5"


def test_a_file_sharing_no_directory_below_the_root_with_the_record_is_named_absolute():
    from tvbo.classes.network import _reference_from

    assert _reference_from(Path("/a/b/out"), Path("/a/b/src/n.h5")) == os.path.join("..", "src", "n.h5")
    assert _reference_from(Path("/Volumes/other/out"), Path("/a/b/src/n.h5")) == "/a/b/src/n.h5"
