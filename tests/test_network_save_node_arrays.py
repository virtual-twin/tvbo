"""A save writes a network's node arrays, or nothing: it keeps the arrays of a network that holds no layer, stores a text array as text in every format, refuses an array declared per node whose length is not the node count and one the format cannot hold, and a save that fails leaves the files it would have replaced as they were, and a killed one a folder the next save removes.

Every network is synthetic except the Desikan-Killiany one, whose 87 nodes beside a 68-value gradient are the case a 68-region measure meets on the 87-region atlas.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys

import h5py
import numpy as np
import pytest
import yaml

from tvbo import Network
from tvbo.data import network_io
from tvbo import datamodel as tvbo_datamodel

FORMATS = ["h5", "zarr", "csv"]
W = np.array([[0.0, 0.5, 0.25], [0.5, 0.0, 0.75], [0.25, 0.75, 0.0]])


def _network():
    """A network on A, B, C with a weight layer and a per-node array, both held in memory."""
    net = Network(nodes=[{"id": i, "label": k} for i, k in enumerate("ABC")], edges=[{"label": "weight"}])
    net.set_matrix("weight", W)
    net.set_array("nodes/scale", np.array([1 / 3, 2.0, -0.5]))
    return net


def _files(folder) -> dict:
    """Every file under *folder*, by its path relative to it, with its bytes."""
    return {str(p.relative_to(folder)): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("binary_format", FORMATS)
def test_a_save_that_fails_leaves_the_saved_network_as_it_was(tmp_path, monkeypatch, binary_format):
    """A save onto a network saved before, its own companion included, that fails once its companion is written leaves every file as it was and nothing beside them, and the network reads back as saved."""
    _network().save(tmp_path / "n.yaml", binary_format=binary_format)
    before = _files(tmp_path)
    back = Network.from_file(str(tmp_path / "n.yaml"))
    back.set_array("nodes/scale", np.array([7.0, 8.0, 9.0]))

    def fail(*_, **__):
        raise RuntimeError("the sidecar could not be written")

    monkeypatch.setattr(network_io, "_write_v07_sidecar", fail)
    with pytest.raises(RuntimeError, match="the sidecar could not be written"):
        back.save(tmp_path / "n.yaml", binary_format=binary_format)
    assert _files(tmp_path) == before
    monkeypatch.undo()
    np.testing.assert_array_equal(Network.from_file(str(tmp_path / "n.yaml")).array("nodes/scale"), [1 / 3, 2.0, -0.5])


@pytest.mark.parametrize("binary_format", FORMATS)
def test_a_network_without_a_layer_saves_its_node_arrays(tmp_path, binary_format):
    """A network holding per-node arrays and no layer writes them to its companion, which a save of the sidecar alone dropped, and reads them back equal."""
    net = Network(number_of_nodes=3)
    net.set_array("nodes/scale", np.array([1 / 3, 2.0, -0.5]))
    net.save(tmp_path / "n.yaml", binary_format=binary_format)
    np.testing.assert_array_equal(Network.from_file(str(tmp_path / "n.yaml")).array("nodes/scale"), [1 / 3, 2.0, -0.5])


def _declare(net, name, path, *, per_node):
    """Declare the array at *path* on *net* as *name*: on its ``node_template`` when *per_node*, on the network otherwise."""
    (net._template_parameters() if per_node else net.parameters)[name] = tvbo_datamodel.Parameter(name=name, measure=path)


REFUSAL = (
    "The array nodes/gradient, which `node_template` declares as `gradient`, holds 68 values, and the network has 87 nodes"
)


@pytest.mark.parametrize("binary_format", FORMATS)
def test_a_68_value_gradient_declared_per_node_on_87_nodes_is_refused_at_save(tmp_path, binary_format):
    """A 68-region gradient declared on the nodes of the 87-region Desikan-Killiany network cannot give each node its value, so the save is refused, naming the array and both counts, and writes nothing."""
    net = Network(parcellation={"atlas": {"name": "DesikanKilliany"}})
    net.set_array("nodes/gradient", np.linspace(0.0, 1.0, 68))
    _declare(net, "gradient", "nodes/gradient", per_node=True)
    (tmp_path / "out").mkdir()
    with pytest.raises(ValueError, match=re.escape(REFUSAL)):
        net.save(tmp_path / "out" / "n.yaml", binary_format=binary_format)
    assert not any((tmp_path / "out").iterdir())


@pytest.mark.parametrize("binary_format", FORMATS)
def test_a_68_value_gradient_declared_per_node_on_87_nodes_is_refused_at_load(tmp_path, binary_format):
    """A sidecar declaring a 68-value companion array on the nodes of an 87-node network is refused when read, naming the array and both counts, and saying to declare an array that does not run over the nodes on the network."""
    net = Network(parcellation={"atlas": {"name": "DesikanKilliany"}})
    net.set_array("nodes/gradient", np.linspace(0.0, 1.0, 68))
    _declare(net, "gradient", "nodes/gradient", per_node=False)
    net.save(tmp_path / "n.yaml", binary_format=binary_format)
    meta = yaml.safe_load((tmp_path / "n.yaml").read_text())
    meta["node_template"]["parameters"]["gradient"] = meta["parameters"].pop("gradient")
    (tmp_path / "n.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    with pytest.raises(ValueError, match=re.escape(REFUSAL) + ".*declare it as a network parameter"):
        Network.from_file(str(tmp_path / "n.yaml"))


@pytest.mark.parametrize("binary_format", FORMATS)
def test_an_array_declared_on_the_network_has_no_length_rule(tmp_path, binary_format):
    """A vocabulary of two names on a three-node network is a network parameter's array, which runs over no nodes, so it saves and reads back whatever its length."""
    net = _network()
    net.set_array("nodes/names", np.array([b"Vis", b"Default"]))
    _declare(net, "names", "nodes/names", per_node=False)
    net.set_array("nodes/code", np.array([0, 1, 1], dtype="int32"))
    _declare(net, "network", "nodes/code", per_node=True)
    net.save(tmp_path / "n.yaml", binary_format=binary_format)
    back = Network.from_file(str(tmp_path / "n.yaml"))
    np.testing.assert_array_equal(back.array("nodes/names"), [b"Vis", b"Default"])
    np.testing.assert_array_equal(back.array("nodes/code"), [0, 1, 1])


TEXT = {
    "1-D": np.array(["Vis", "Default mode", "ç"]),
    "2-D": np.array([["a", "b"], ["c", ""], ["e", "f g"]]),
    "empty": np.array([], dtype="<U1"),
}


@pytest.mark.parametrize("text", sorted(TEXT))
@pytest.mark.parametrize("binary_format", FORMATS)
def test_a_text_array_is_stored_as_text_in_every_format(tmp_path, binary_format, text):
    """A text array (``<U``), non-ASCII, empty and 2-D ones included, saves in every format and reads back as the same text, which HDF5 stores as UTF-8 strings."""
    net = _network()
    net.set_array("nodes/names", TEXT[text])
    net.save(tmp_path / "n.yaml", binary_format=binary_format)
    back = Network.from_file(str(tmp_path / "n.yaml")).array("nodes/names")
    assert (back.dtype.kind, back.shape) == ("U", TEXT[text].shape)
    np.testing.assert_array_equal(back, TEXT[text])
    if binary_format == "h5":
        with h5py.File(tmp_path / "n.h5") as stored:
            assert h5py.check_string_dtype(stored["nodes/names"].dtype).encoding == "utf-8"


UNHOLDABLE = {
    "object": (np.array([{}, 1, "a"], dtype=object), "object", []),
    "datetime64": (np.array(["2026-01-01", "2026-01-02", "2026-01-03"], dtype="datetime64[ns]"), "datetime64[ns]", ["zarr"]),
    "timedelta64": (np.array([1, 2, 3], dtype="timedelta64[s]"), "timedelta64[s]", ["zarr"]),
    "longdouble": (np.array([1.5, 2.0, 3.0], dtype=np.longdouble), "longdouble", ["h5", "csv"]),
    "structured": (
        np.array([(1, 2.0), (3, 4.0), (5, 6.0)], dtype=[("a", "i4"), ("b", "f8")]),
        "[('a', '<i4'), ('b', '<f8')]",
        ["h5", "zarr"],
    ),
    "void": (np.array([b"ab", b"cd", b"ef"], dtype="V2"), "|V2", ["h5", "zarr"]),
}


@pytest.mark.filterwarnings("ignore::zarr.errors.UnstableSpecificationWarning")
@pytest.mark.parametrize("kind", sorted(UNHOLDABLE))
@pytest.mark.parametrize("binary_format", FORMATS)
def test_an_array_a_format_cannot_hold_is_refused_naming_it_and_the_network(tmp_path, binary_format, kind):
    """An array a format cannot store is refused before anything is written, naming the network, the array, its dtype and the formats that hold it, each of which saves it and reads it back equal."""
    value, dtype, holding = UNHOLDABLE[kind]
    net = _network()
    net.label = "three regions"
    net.set_array("nodes/when", value)
    if binary_format in holding:
        net.save(tmp_path / "n.yaml", binary_format=binary_format)
        np.testing.assert_array_equal(Network.from_file(str(tmp_path / "n.yaml")).array("nodes/when"), value)
        return
    held = f"which binary_format={' or '.join(map(repr, holding))} holds" if holding else "which no format holds"
    with pytest.raises(ValueError) as refused:
        net.save(tmp_path / "n.yaml", binary_format=binary_format)
    assert refused.value.args[0].startswith(
        f"The network 'three regions' (n.yaml) holds arrays binary_format={binary_format!r} cannot store, so nothing was written: nodes/when ({dtype}), {held}."
    )
    assert not any(tmp_path.iterdir())


def _ended_process_id() -> int:
    """The id of a process that has run and ended."""
    child = subprocess.Popen([sys.executable, "-c", ""])
    child.wait()
    return child.pid


def test_a_killed_save_leaves_a_folder_naming_it_that_the_next_save_of_its_sidecar_removes(tmp_path, monkeypatch):
    """A save writes into ``.<sidecar>.unfinished-tvbo-save-<host>@<process id>-…``, so the folder a killed save leaves says what it is. The next save of that sidecar on the same host removes it, and no folder of a running save, of another sidecar or of a save on another host, whose process id says nothing here."""
    ended, host = _ended_process_id(), socket.gethostname()
    left = [
        f".n.yaml.unfinished-tvbo-save-{host}@{ended}-killed",
        f".n.yaml.unfinished-tvbo-save-{host}@{os.getpid()}-running",
        f".n.yaml.unfinished-tvbo-save-{host}-node2@{ended}-elsewhere",
        f".m.yaml.unfinished-tvbo-save-{host}@{ended}-killed",
    ]
    for name in left:
        (tmp_path / name).mkdir()
        (tmp_path / name / "n.h5").write_bytes(b"partial")
    staging = []
    write_sidecar = network_io._write_v07_sidecar

    def recording(network, path, *args, **kwargs):
        staging.append(path.parent.name)
        return write_sidecar(network, path, *args, **kwargs)

    monkeypatch.setattr(network_io, "_write_v07_sidecar", recording)
    _network().save(tmp_path / "n.yaml")
    assert staging[0].startswith(f".n.yaml.unfinished-tvbo-save-{host}@{os.getpid()}-")
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(["n.h5", "n.yaml", *left[1:]])
