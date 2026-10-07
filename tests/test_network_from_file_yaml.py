"""`Network.from_file` reads a sidecar through the loader recipes use: `!include`, `<<:` merge keys and anchors work, and relative paths resolve against the sidecar's own directory."""

import numpy as np
import yaml

WEIGHTS = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])


def _saved(tmp_path):
    """A plain sidecar + companion pair, `base.yaml` and `base.h5`, as `save_network` writes them."""
    from tvbo.classes.network import Network
    from tvbo.data.network_io import save_network

    net = Network.from_matrix(WEIGHTS, np.full_like(WEIGHTS, 10.0), labels=["a", "b", "c"])
    save_network(net, tmp_path / "base.yaml")
    return tmp_path / "base.yaml"


def test_a_plain_sidecar_loads_as_before(tmp_path):
    from tvbo.classes.network import Network

    net = Network.from_file(_saved(tmp_path))
    assert [n.label for n in net.nodes] == ["a", "b", "c"]
    np.testing.assert_allclose(np.asarray(net.matrix("weight")), WEIGHTS)


def test_a_sidecar_merges_an_included_fragment_and_resolves_its_own_paths(tmp_path):
    from tvbo.classes.network import Network

    _saved(tmp_path)
    (tmp_path / "spec").mkdir()
    fragment = tmp_path / "spec" / "net.yaml"
    fragment.write_text("<<: !include ../base.yaml\nlabel: merged\ndata_file: ../base.h5\n")

    net = Network.from_file(fragment)
    assert net.label == "merged"
    assert [n.label for n in net.nodes] == ["a", "b", "c"]
    np.testing.assert_allclose(np.asarray(net.matrix("weight")), WEIGHTS)


def test_anchors_and_merge_keys_work_inside_a_sidecar(tmp_path):
    from tvbo.classes.network import Network

    base = _saved(tmp_path)
    meta = yaml.safe_load(base.read_text())
    meta.pop("nodes")
    anchored = tmp_path / "anchored.yaml"
    anchored.write_text(
        yaml.safe_dump(meta, sort_keys=False)
        + "nodes:\n  - &node {id: 0, label: a, record: true}\n  - {<<: *node, id: 1, label: b}\n  - {<<: *node, id: 2, label: c}\n"
    )

    net = Network.from_file(anchored)
    assert [(n.id, n.label) for n in net.nodes] == [(0, "a"), (1, "b"), (2, "c")]
    np.testing.assert_allclose(np.asarray(net.matrix("weight")), WEIGHTS)
