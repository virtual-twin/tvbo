"""Saving a network keeps the per-node arrays it was loaded with.

A companion may carry per-node datasets the writer does not model — a study's fitted per-region intervals, a per-node scaling — and an observation may reference one as ``network.nodes.<name>``. When ``tvbo workflow`` bakes a network into a kit it re-saves it, so a writer that only persists what it models produces a kit whose spec cannot resolve on the target host.
"""

import h5py
import numpy as np
import pytest

from tvbo.classes.network import Network


def _network_with_node_arrays(tmp_path, arrays):
    """A two-node network whose companion carries extra ``nodes/<name>`` datasets."""
    net = Network(number_of_nodes=2)
    net._cached_weights = np.array([[0.0, 1.0], [1.0, 0.0]])
    src = tmp_path / "src" / "network.yaml"
    src.parent.mkdir(parents=True, exist_ok=True)
    net.save(str(src))
    with h5py.File(src.parent / "network.h5", "r+") as f:
        grp = f.require_group("nodes")
        for name, data in arrays.items():
            grp.create_dataset(name, data=data)
    return src


def test_extra_node_arrays_survive_a_resave(tmp_path):
    arrays = {"scale_lo": np.array([-0.117, 0.001]), "scale_hi": np.array([-0.019, 0.010])}
    src = _network_with_node_arrays(tmp_path, arrays)

    out = tmp_path / "kit" / "network.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    Network.load(str(src)).save(str(out))

    with h5py.File(out.parent / "network.h5", "r") as f:
        for name, expected in arrays.items():
            assert f"nodes/{name}" in f, f"nodes/{name} was dropped by the re-save"
            assert np.array_equal(np.asarray(f[f"nodes/{name}"][()], float), expected)


def test_resaved_node_array_still_resolves_as_a_reference(tmp_path):
    src = _network_with_node_arrays(tmp_path, {"scale_lo": np.array([-0.117, 0.001])})
    out = tmp_path / "kit" / "network.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    Network.load(str(src)).save(str(out))

    from tvbo.data.param_io import resolve_network_node

    resolved = resolve_network_node(Network.load(str(out)), "scale_lo")
    assert resolved is not None, "a re-saved node array must resolve as network.nodes.<name>"
    assert np.allclose(np.asarray(resolved, float), [-0.117, 0.001])


def test_coordinates_are_not_duplicated_by_the_copy_through(tmp_path):
    src = _network_with_node_arrays(tmp_path, {"scale_lo": np.array([1.0, 2.0])})
    out = tmp_path / "kit" / "network.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    net = Network.load(str(src))
    net.save(str(out))
    with h5py.File(out.parent / "network.h5", "r") as f:
        names = sorted(f["nodes"].keys())
    assert len(names) == len(set(names))
    assert "scale_lo" in names


@pytest.mark.parametrize("attr", ["missing", "scale_hi"])
def test_absent_node_array_does_not_resolve_silently(tmp_path, attr):
    src = _network_with_node_arrays(tmp_path, {"scale_lo": np.array([1.0, 2.0])})
    from tvbo.data.param_io import resolve_network_node

    assert resolve_network_node(Network.load(str(src)), attr) is None, (
        f"{attr!r} names no node array here, so it must not resolve to anything"
    )
