"""A network saved with ``binary_format="csv"`` writes every array an HDF5 companion holds to a CSV of its own, named on the sidecar record of what it holds, and reads back as an HDF5 save reads back: every array, at its dtype and in its sparse format.

A layer's CSV is its template edge's ``data_file``, an edge parameter's its ``source``, a per-node or network array's the ``source`` of the parameter declaring it by ``measure``, and a mesh array's a ``mesh`` field. A CSV companion written when a CSV held one matrix reads as the one layer that writer stored. Every network here is synthetic except the curated ones of the slow scan.
"""

from __future__ import annotations

import re

import h5py
import numpy as np
import pytest
import yaml
from scipy import sparse

from tvbo import Network, SimulationExperiment
from tvbo.classes.network import NETWORK_DIR
from tvbo.datamodel import schema as tvbo_datamodel
from tvbo.utils.study_layout import is_network_companion

W = np.array([[0.0, 2 / 7, 5 / 7], [2 / 7, 0.0, 9 / 7], [5 / 7, 9 / 7, 0.0]])
L = 10 * W + 1 / 3
F = np.array([[1.0, -1 / 3, 1 / 9], [-1 / 3, 1.0, 2 / 3], [1 / 9, 2 / 3, 1.0]])
LAYERS = {"weight": W, "length": L, "fc_alpha": F}
CENTRES = {0: (1 / 3, 2.5, -7.0), 1: (-11 / 9, 0.0, 4 / 3), 2: (6.25, -1 / 7, 0.5)}
NODE_DATASETS = {
    "nodes/coordinates": np.array(list(CENTRES.values()), dtype="float32"),
    "nodes/scaling": np.array([1 / 3, 2.0, np.nan]),
    "nodes/rank": np.array([3, -1, 2], dtype="int16"),
    "nodes/names": np.array([b"left V1", b"", b"right,A1"]),
    "nodes/groups": np.array([b"Default mode", b"Vis", b'"quoted"'], dtype=h5py.string_dtype()),
    "nodes/pairs": np.array([[0.5, 1 / 7], [2.0, 3.0], [-1.0, 1e-300]], dtype="float32"),
    "nodes/phase": np.array([1 / 3 + 2j / 7, -1e-300 + 0j, np.nan + 1j * np.inf]),
    "nodes/bands": np.array([0.5, 1.5, 4.0, 8.0, 13.0]),
}


def _network(order, positioned=True):
    """A network on A, B, C that declares the layers of LAYERS in *order* and holds all three."""
    nodes = [{"id": i, "label": k} for i, k in enumerate("ABC")]
    if positioned:
        for i, node in enumerate(nodes):
            node["position"] = dict(zip("xyz", CENTRES[i], strict=True))
    net = Network(nodes=nodes, edges=[{"label": name} for name in order])
    for name in order:
        net.set_matrix(name, LAYERS[name])
    return net


def _sidecar(path) -> dict:
    return yaml.safe_load(path.read_text())


def _edges(sidecar) -> dict:
    return {edge["label"]: edge for edge in _sidecar(sidecar).get("edges") or []}


def _companion_arrays(net) -> dict:
    """Every array the network's companion holds, by its path: the layers, their edge parameters, and the node and mesh datasets."""
    store = net._store
    out = {f"edges/{name}": store[name] for name in store.names}
    out |= {
        f"edges/{e}/edge_parameters/{p}": v for e, params in net.edge_parameter_arrays().items() for p, v in params.items()
    }
    return out | {path: store.read_dataset(path) for group in ("nodes", "mesh") for path in store.dataset_keys(group)}


def _assert_same_arrays(got: dict, want: dict):
    """The same paths, each array of the same type, dtype and shape, every value equal and every NaN in the same place."""
    assert sorted(got) == sorted(want)
    for path, value in want.items():
        assert (type(got[path]), got[path].dtype, got[path].shape) == (type(value), value.dtype, value.shape), path
        dense = [v.toarray() if sparse.issparse(v) else v for v in (got[path], value)]
        np.testing.assert_array_equal(*dense, err_msg=path)


def _full_network(tmp_path):
    """A network holding one of each kind of array an HDF5 companion stores, read back from that companion.

    A csr float32 layer beside two dense float64 ones, an edge parameter, a node mapping, a mesh, per-node arrays of six dtypes, fixed-width and variable-length text with a space, a comma, a quote and an empty string among them, a complex one, a 2-D one, an array that does not run over the nodes, and coordinates no node position carries.
    """
    net = _network(tuple(LAYERS), positioned=False)
    fc_alpha = next(e for e in net.edges if e.label == "fc_alpha")
    fc_alpha.format, fc_alpha.dtype = "csr", "float32"
    net.set_array("edges/weight/edge_parameters/gain", 3 * W)
    net.set_node_mapping(np.array([0, 0, 1], dtype="int32"))
    net.set_array("mesh/vertices", np.arange(12, dtype="float32").reshape(4, 3) / 7)
    net.set_array("mesh/elements", np.array([[0, 1, 2], [1, 2, 3]], dtype="int32"))
    net.set_array("mesh/normals", np.ones((4, 3), dtype="float32"))
    (tmp_path / "source").mkdir()
    net.save(tmp_path / "source" / "n.yaml")
    with h5py.File(tmp_path / "source" / "n.h5", "a") as stored:
        for path, value in NODE_DATASETS.items():
            stored[path] = value
    return Network.from_file(str(tmp_path / "source" / "n.yaml"))


def _saved_and_read_back(net, folder, binary_format):
    folder.mkdir()
    net.save(folder / "n.yaml", binary_format=binary_format)
    return Network.from_file(str(folder / "n.yaml"))


@pytest.mark.parametrize("order", [("weight", "length", "fc_alpha"), ("length", "weight", "fc_alpha")])
def test_every_layer_reads_back_from_its_own_csv_after_two_round_trips(tmp_path, order):
    """Saved, read back, saved again and read back, each layer is the matrix that was saved, exactly, in either order of declaration; each layer's template edge names its CSV, the first as the network's ``data_file``, and the node positions stay the coordinates, with no CSV of their own."""
    net = _network(order)
    for generation in ("first", "second"):
        net = _saved_and_read_back(net, tmp_path / generation, "csv")
        sidecar = tmp_path / generation / "n.yaml"
        files = {name: edge["data_file"] for name, edge in _edges(sidecar).items()}
        assert files == {name: f"n_meas-{name.replace('_', '')}.csv" for name in order}
        assert _sidecar(sidecar)["data_file"] == files[order[0]]
        assert sorted(p.name for p in sidecar.parent.glob("*.csv")) == sorted(files.values())
        for name, matrix in LAYERS.items():
            np.testing.assert_array_equal(net.matrix(name, format="dense"), matrix)
        assert net.get_centers() == CENTRES


def test_a_bids_named_sidecar_takes_each_entity_before_its_suffix(tmp_path):
    """The CSVs of ``exp-1_model-X_network.yaml`` carry their entity before the ``_network`` suffix, so each is still a network companion by name."""
    sidecar = tmp_path / "exp-1_model-X_network.yaml"
    _network(("weight", "length")).save(sidecar, binary_format="csv")
    files = [edge["data_file"] for edge in _edges(sidecar).values()]
    assert files == ["exp-1_model-X_meas-weight_network.csv", "exp-1_model-X_meas-length_network.csv"]
    assert all((tmp_path / f).is_file() and is_network_companion(f) for f in files)


def test_a_layer_no_template_edge_declares_is_declared_with_its_csv(tmp_path):
    """A layer held without a template edge, as `from_bids` and a parcellation load hold theirs, gets the template edge that names its CSV, and reads back from it."""
    net = Network(nodes=[{"id": i, "label": k} for i, k in enumerate("ABC")])
    net.set_array("weight", W)
    net.set_array("length", L)
    back = _saved_and_read_back(net, tmp_path / "csv", "csv")
    assert _edges(tmp_path / "csv" / "n.yaml") == {
        "length": {"label": "length", "data_file": "n_meas-length.csv"},
        "weight": {"label": "weight", "data_file": "n_meas-weight.csv"},
    }
    np.testing.assert_array_equal(back.matrix("weight", format="dense"), W)
    np.testing.assert_array_equal(back.matrix("length", format="dense"), L)


def test_a_csv_save_reads_back_as_an_hdf5_save_does(tmp_path):
    """Layers of two formats and two dtypes, an edge parameter, a node mapping, a mesh and per-node arrays, a complex one among them, read back from CSV companions as they read back from an HDF5 one, and again after a second CSV save and after an HDF5 save of what the CSVs gave."""
    net = _full_network(tmp_path)
    want = _companion_arrays(_saved_and_read_back(net, tmp_path / "h5", "h5"))
    assert {"edges/weight/edge_parameters/gain", "nodes/parent_index", "mesh/elements", *NODE_DATASETS} <= set(want)
    back = _saved_and_read_back(net, tmp_path / "csv", "csv")
    _assert_same_arrays(_companion_arrays(back), want)
    again = _saved_and_read_back(back, tmp_path / "csv-again", "csv")
    _assert_same_arrays(_companion_arrays(again), want)
    _assert_same_arrays(_companion_arrays(_saved_and_read_back(again, tmp_path / "h5-again", "h5")), want)
    np.testing.assert_array_equal(again.node_mapping_data, [0, 0, 1])


def test_each_csv_is_named_on_the_record_of_what_it_holds(tmp_path):
    """A layer's CSV is its edge's ``data_file`` with its dtype and sparse format, an edge parameter's its ``source``, a mesh array's a ``mesh`` field, and any other array's the ``source`` of a parameter declaring it by ``measure``, with its shape and dtype: on ``node_template`` when it has a row for each node, on the network otherwise. Every CSV written is named once."""
    _full_network(tmp_path).save(tmp_path / "n.yaml", binary_format="csv")
    sidecar = _sidecar(tmp_path / "n.yaml")
    edges = _edges(tmp_path / "n.yaml")
    assert edges["fc_alpha"]["data_file"] == "n_meas-fcalpha.csv"
    assert (edges["fc_alpha"]["dtype"], edges["fc_alpha"]["format"]) == ("float32", "csr")
    assert "dtype" not in edges["weight"] and edges["weight"]["parameters"]["gain"]["source"] == "n_meas-weight_param-gain.csv"
    assert sidecar["mesh"] == {
        "vertices_field": "n_mesh-vertices.csv",
        "elements": "n_mesh-elements.csv",
        "normals": "n_mesh-normals.csv",
    }
    per_node = sidecar["node_template"]["parameters"]
    assert per_node["parent_index"] | {"name": "parent_index"} == {
        "name": "parent_index",
        "measure": "nodes/parent_index",
        "source": "n_nodes-parentindex.csv",
        "shape": "(3,)",
        "dtype": "int32",
    }
    assert {per_node[k]["dtype"] for k in ("names", "groups", "pairs", "phase")} == {"|S8", "<U12", "float32", "complex128"}
    assert per_node["coordinates"]["shape"] == "(3, 3)" and "dtype" not in per_node["scaling"]
    assert sidecar["parameters"]["bands"] == {"measure": "nodes/bands", "source": "n_nodes-bands.csv", "shape": "(5,)"}
    named = [edge["data_file"] for edge in edges.values()] + [edges["weight"]["parameters"]["gain"]["source"]]
    named += list(sidecar["mesh"].values()) + [p["source"] for p in [*per_node.values(), sidecar["parameters"]["bands"]]]
    assert sorted(named) == sorted(p.name for p in tmp_path.glob("*.csv"))


def _categorical_network():
    """A network on A, B, C whose nodes each belong to one of two functional networks, declared as a categorical attribute beside the vocabulary its codes index, and held in memory."""
    net = Network(
        nodes=[{"id": i, "label": k, "position": dict(zip("xyz", CENTRES[i], strict=True))} for i, k in enumerate("ABC")],
        edges=[{"label": "weight"}],
        parameters={"network_names": {"measure": "nodes/network_names", "description": "the networks"}},
        node_template={"parameters": {"network": {"measure": "nodes/network_code", "levels": "network_names"}}},
    )
    net.set_matrix("weight", W)
    net.set_array("nodes/network_code", np.array([1, 0, 1], dtype="int32"))
    net.set_array("nodes/network_names", np.array([b"Vis", b"Default"]))
    return net


@pytest.mark.parametrize("binary_format", ["h5", "zarr", "csv"])
def test_a_categorical_attribute_reads_back_declared_in_every_format(tmp_path, binary_format):
    """A per-node code into a vocabulary reads back with its declaration in each format, as ``network.nodes.<name>``, and stays off the nodes themselves; an HDF5 companion holds both arrays at their paths and nothing else for them."""
    from tvbo.data.param_io import resolve_network_node

    back = _saved_and_read_back(_categorical_network(), tmp_path / binary_format, binary_format)
    template = back.node_template.parameters["network"]
    assert (template.measure, template.levels) == ("nodes/network_code", "network_names")
    assert back.parameters["network_names"].measure == "nodes/network_names"
    np.testing.assert_array_equal(resolve_network_node(back, "network"), [1, 0, 1])
    np.testing.assert_array_equal(back.array("nodes/network_names"), [b"Vis", b"Default"])
    assert all("network" not in (node.parameters or {}) for node in back.nodes)
    if binary_format == "h5":
        with h5py.File(tmp_path / "h5" / "n.h5") as stored:
            assert sorted(stored["nodes"]) == ["coordinates", "network_code", "network_names"]


def test_a_declared_per_node_array_is_not_copied_onto_each_node(tmp_path):
    """A ``node_template`` parameter naming its array by ``measure`` declares the array, so neither the nodes in memory nor a dump of them carry a copy of it."""
    net = _categorical_network()
    assert all(not node.parameters for node in net.nodes)
    assert all("parameters" not in node for node in yaml.safe_load(net.to_yaml())["nodes"])


def test_levels_naming_no_network_parameter_is_refused():
    """Codes into a vocabulary the network does not carry say nothing, so the network is refused, naming the parameter and the vocabulary."""
    with pytest.raises(ValueError, match=r"'network' gives its values as codes into the vocabulary 'names' \(`levels`\)"):
        Network(
            nodes=[{"id": 0, "label": "A"}],
            node_template={"parameters": {"network": {"measure": "nodes/c", "levels": "names"}}},
        )


def _one_matrix_csv(tmp_path, declared, held):
    """The network read back from ``n.yaml`` beside ``n.csv`` holding *held*, as the one-matrix writer left them for a network declaring *declared*."""
    np.savetxt(tmp_path / "n.csv", held, delimiter=" ", fmt="%.8g")
    sidecar = {
        "label": "n",
        "number_of_nodes": 3,
        "data_file": "n.csv",
        "nodes": [{"id": i, "label": k} for i, k in enumerate("ABC")],
    }
    if declared:
        sidecar["edges"] = [{"label": name} for name in declared]
    (tmp_path / "n.yaml").write_text(yaml.safe_dump(sidecar, sort_keys=False))
    return Network.from_file(str(tmp_path / "n.yaml"))


@pytest.mark.parametrize(
    ("declared", "held"),
    [(["weight", "length"], W), (["length", "weight"], L), ([], W)],
    ids=["weight-first", "length-first", "none-declared"],
)
def test_a_csv_no_record_names_reads_as_the_layer_it_holds(tmp_path, declared, held):
    """A sidecar whose ``data_file`` is one CSV that no record names was written by the one-matrix writer, which stored the first template edge, or ``weight`` with none declared; that layer reads from it and every other declared layer reads as missing."""
    back = _one_matrix_csv(tmp_path, declared, held)
    first = declared[0] if declared else "weight"
    np.testing.assert_allclose(back._store[first], held, rtol=1e-7)
    for other in declared[1:]:
        assert other not in back._store
        with pytest.raises(KeyError):
            back._store[other]
    if first == "weight":
        np.testing.assert_allclose(back.matrix("weight", format="dense"), held, rtol=1e-7)
        assert back.matrix("length") is None


@pytest.mark.parametrize("binary_format", ["h5", "csv"])
def test_a_one_matrix_csv_saves_again_without_a_layer_it_never_held(tmp_path, binary_format):
    """Read back from a one-matrix CSV holding the weights beside a sidecar that also declares ``length``, a network saves again with the weights alone: no companion it writes holds a ``length``, which the one-matrix reader served from the weights."""
    back = _one_matrix_csv(tmp_path, ["weight", "length"], W)
    (tmp_path / "again").mkdir()
    back.save(tmp_path / "again" / "n.yaml", binary_format=binary_format)
    if binary_format == "h5":
        with h5py.File(tmp_path / "again" / "n.h5") as stored:
            assert "edges/weight" in stored and "edges/length" not in stored
    else:
        assert {name: edge.get("data_file") for name, edge in _edges(tmp_path / "again" / "n.yaml").items()} == {
            "weight": "n_meas-weight.csv",
            "length": None,
        }
        assert sorted(p.name for p in (tmp_path / "again").glob("*.csv")) == ["n_meas-weight.csv"]
    again = Network.from_file(str(tmp_path / "again" / "n.yaml"))
    np.testing.assert_allclose(again.matrix("weight", format="dense"), W, rtol=1e-7)
    assert again.matrix("length") is None


def test_a_one_matrix_csv_with_its_length_first_is_refused_at_save(tmp_path):
    """A one-matrix CSV beside a sidecar declaring ``length`` first holds the lengths, so the weight it declares is held nowhere; a save is refused as a read of that weight is, rather than writing a companion without it."""
    back = _one_matrix_csv(tmp_path, ["length", "weight"], L)
    (tmp_path / "again").mkdir()
    with pytest.raises(KeyError, match=r"declares the layer 'weight', and n\.csv does not hold it"):
        back.save(tmp_path / "again" / "n.yaml")
    assert not any((tmp_path / "again").iterdir())


@pytest.mark.parametrize("binary_format", ["h5", "csv"])
def test_a_network_holding_a_mesh_and_no_layer_reads_back_without_one(tmp_path, binary_format):
    """A network holding its mesh and no layer, as each curated surface network does, reads back with no layer and its mesh: the CSV its ``data_file`` names holds the mesh's elements, which a CSV save names on the mesh, so no reader takes it for a one-matrix CSV's weight."""
    net = Network(number_of_nodes=4)
    net.set_array("mesh/vertices", np.arange(12, dtype="float32").reshape(4, 3) / 7)
    net.set_array("mesh/elements", np.array([[0, 1, 2], [1, 2, 3]], dtype="int32"))
    net.save(tmp_path / "n.yaml", binary_format=binary_format)
    back = Network.from_file(str(tmp_path / "n.yaml"))
    assert back._store.names == [] and "weight" not in back._store
    np.testing.assert_array_equal(back._store.read_dataset("mesh/elements"), [[0, 1, 2], [1, 2, 3]])


def test_an_array_that_is_not_1d_or_2d_is_refused_naming_it(tmp_path):
    """A three-axis layer and a three-axis per-node array have no rows a CSV holds them as, so the save is refused, naming both, and writes nothing."""
    _network(("weight",)).save(tmp_path / "h.yaml")
    with h5py.File(tmp_path / "h.h5", "a") as stored:
        stored["nodes/band"] = np.zeros((3, 2, 2))
    net = Network.from_file(str(tmp_path / "h.yaml"))
    net.set_array("cube", np.zeros((3, 3, 2)))
    (tmp_path / "csv").mkdir()
    with pytest.raises(ValueError, match=re.escape("would lose edges/cube, nodes/band, which are not 1-D or 2-D")):
        net.save(tmp_path / "csv" / "n.yaml", binary_format="csv")
    assert not any((tmp_path / "csv").iterdir())


def test_arrays_whose_names_give_one_file_name_are_refused(tmp_path):
    """``fc_1`` and ``fc1`` both give the entity ``meas-fc1``, so neither would have a CSV of its own; the save is refused, naming both, and writes nothing."""
    net = _network(("weight",))
    net.set_matrix("fc_1", F)
    net.set_matrix("fc1", 2 * F)
    with pytest.raises(ValueError, match=re.escape("['edges/fc_1', 'edges/fc1']")):
        net.save(tmp_path / "n.yaml", binary_format="csv")
    assert not any(tmp_path.iterdir())


def test_a_sparse_layer_of_another_shape_than_it_reads_back_at_is_refused(tmp_path):
    """A sparse layer's CSV holds its entries, not its shape, which reads back as a row for each node by a column for each node; a 3 x 2 one is refused, naming it, and nothing is written."""
    net = _network(("weight",))
    net.set_matrix("projection", sparse.csr_matrix(np.ones((3, 2))))
    next(e for e in net.edges if e.label == "projection").format = "csr"
    with pytest.raises(ValueError, match=re.escape("would misread edges/projection (3 x 2)")):
        net.save(tmp_path / "n.yaml", binary_format="csv")
    assert not any(tmp_path.iterdir())


def test_a_layer_naming_an_hdf5_file_as_its_own_is_refused(tmp_path):
    """An edge's ``data_file`` names the CSV holding its layer alone; one naming an HDF5 file is refused when the network is read, naming it."""
    _network(("weight", "length")).save(tmp_path / "n.yaml", binary_format="csv")
    meta = _sidecar(tmp_path / "n.yaml")
    meta["edges"][1]["data_file"] = "other.h5"
    (tmp_path / "n.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    with pytest.raises(ValueError, match=r"The layer 'length' names other\.h5 as its `data_file`"):
        Network.from_file(str(tmp_path / "n.yaml"))


@pytest.mark.parametrize("missing", ["n_meas-length.csv", "n_nodes-parentindex.csv"])
def test_a_csv_the_sidecar_names_and_that_is_missing_is_refused_by_every_reader(tmp_path, missing):
    """A layer's or a per-node array's CSV that is gone is still held by the companion, which the sidecar says without a file read, so ``matrix``/``array`` and a save raise naming the file rather than serve nothing or save without it."""
    _full_network(tmp_path).save(tmp_path / "n.yaml", binary_format="csv")
    (tmp_path / missing).unlink()
    back = Network.from_file(str(tmp_path / "n.yaml"))
    held = "length" if "length" in missing else "nodes/parent_index"
    assert (held in back._store) if held == "length" else back._store.holds(held)
    with pytest.raises(OSError, match=re.escape(missing)):
        back.matrix("length") if held == "length" else back.array(held)
    (tmp_path / "again").mkdir()
    with pytest.raises(OSError, match=re.escape(missing)):
        back.save(tmp_path / "again" / "n.yaml", binary_format="h5")


def test_a_csv_companion_reports_each_array_without_reading_it(tmp_path):
    """``info`` answers for a CSV companion from its links and its files' lines: a dense and a sparse layer, a declared array and the positions; so the node-count check refuses a sidecar listing fewer nodes than its layers have rows, as it does beside HDF5."""
    _full_network(tmp_path).save(tmp_path / "n.yaml", binary_format="csv")
    store = Network.from_file(str(tmp_path / "n.yaml"))._store
    assert (store.info("weight").shape, store.info("weight").format) == ((3, 3), "dense")
    sparse_info = store.info("fc_alpha")
    assert (sparse_info.shape, sparse_info.dtype, sparse_info.format) == ((3, 3), np.dtype("float32"), "csr")
    assert (store.info("nodes/names").shape, store.info("nodes/names").dtype) == ((3,), np.dtype("S8"))
    meta = _sidecar(tmp_path / "n.yaml")
    meta["nodes"], meta["number_of_nodes"] = meta["nodes"][:2], 2
    (tmp_path / "n.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    with pytest.raises(
        ValueError, match=r"lists 2 node\(s\) and its companion n_meas-weight\.csv stores the layer 'weight' as 3 x 3"
    ):
        Network.from_file(str(tmp_path / "n.yaml"))


def test_a_sparse_layer_is_written_as_its_entries_and_never_dense(tmp_path, monkeypatch):
    """A sparse layer's CSV holds one ``row column value`` line per stored entry, explicit zeros included, in row order, and the network reads it back as the sparse matrix it was, with nothing densified on the way: a 327,684-node layer dense is 860 GB."""
    layer = sparse.csr_matrix((np.array([0.25, -1.5, 0.0, 3.0], dtype="float32"), ([0, 1, 1, 2], [2, 1, 2, 0])), shape=(3, 3))
    net = _network(("weight",))
    net.set_matrix("local_connectivity", layer)
    next(e for e in net.edges if e.label == "local_connectivity").format = "csr"

    def refuse(*_, **__):
        raise AssertionError("a sparse layer was densified")

    for method in ("toarray", "todense"):
        monkeypatch.setattr(sparse.csr_matrix, method, refuse)
        monkeypatch.setattr(sparse.coo_matrix, method, refuse)
    back = _saved_and_read_back(net, tmp_path / "csv", "csv")
    assert (tmp_path / "csv" / "n_meas-localconnectivity.csv").read_text().splitlines() == [
        "0 2 0.25",
        "1 1 -1.5",
        "1 2 0",
        "2 0 3",
    ]
    read = back.matrix("local_connectivity", apply_transforms=False)
    assert (read.format, read.shape, read.nnz, read.dtype) == ("csr", (3, 3), 4, np.dtype("float32"))
    assert (read != layer).nnz == 0


@pytest.mark.slow
def test_a_sparse_layer_of_a_surface_network_saves_as_csv_without_densifying(tmp_path):
    """The 327,684-vertex surface network with a sparse local-connectivity layer saves with CSV companions and reads the layer back sparse and equal, under any memory guard a dense layer could not pass."""
    net = Network(iri="tvbo:networks/tpl-MNI152NLin2009cAsym_cohort-HCPYA_atlas-Lobar8_desc-surf_relmat")
    rng = np.random.default_rng(0)
    rows, cols = rng.integers(0, net.number_of_nodes, (2, 100_000))
    layer = sparse.coo_matrix((rng.random(100_000).astype("float32"), (rows, cols)), shape=(net.number_of_nodes,) * 2).tocsr()
    net.set_matrix("local_connectivity", layer)
    back = _saved_and_read_back(net, tmp_path / "csv", "csv")
    read = back.matrix("local_connectivity", apply_transforms=False)
    assert sparse.issparse(read) and (read != layer).nnz == 0


def test_a_kit_bundle_carries_every_csv_a_network_sidecar_names(tmp_path):
    """The files a network sidecar references, which a kit copies beside it, are the CSVs its records name, each once."""
    _full_network(tmp_path).save(tmp_path / "n.yaml", binary_format="csv")
    companions = [p.name for p in SimulationExperiment._sidecar_companions(tmp_path / "n.yaml")]
    assert sorted(companions) == sorted(p.name for p in tmp_path.glob("*.csv"))
    assert len(companions) == len(set(companions)) == 16
    assert {"n_meas-weight_param-gain.csv", "n_mesh-elements.csv", "n_nodes-phase.csv"} <= set(companions)


def test_an_hdf5_save_of_a_network_read_from_csvs_names_no_csv(tmp_path):
    """Saved with an HDF5 companion, a network read from CSVs keeps no link to them: its records name no file, and the companion holds what an HDF5 save of the original holds."""
    net = _full_network(tmp_path)
    back = _saved_and_read_back(net, tmp_path / "csv", "csv")
    _saved_and_read_back(back, tmp_path / "h5", "h5")
    text = (tmp_path / "h5" / "n.yaml").read_text()
    assert ".csv" not in text
    direct = _saved_and_read_back(net, tmp_path / "h5-direct", "h5")
    with h5py.File(tmp_path / "h5" / "n.h5") as again, h5py.File(tmp_path / "h5-direct" / "n.h5") as first:
        paths = []
        again.visit(paths.append)
        expected = []
        first.visit(expected.append)
        assert paths == expected
    assert direct is not None


SCHAEFER = "tpl-FSLMNI152_cohort-HCPYA_rec-dTOR_atlas-Schaefer2018_seg-17Networks_scale-100_desc-SC_relmat"


def test_a_curated_schaefer_network_declares_each_parcel_s_functional_network():
    """The curated Schaefer2018 networks declare ``parent_index`` as each parcel's functional network, a code into ``functional_network_labels``, whose entry it names is the network part of the parcel's label."""
    from tvbo.data.param_io import resolve_network_node

    net = Network(iri=f"tvbo:networks/{SCHAEFER}")
    declared = net.node_template.parameters["functional_network"]
    assert (declared.measure, declared.levels) == ("nodes/parent_index", "functional_network_labels")
    names = net.array(net.parameters["functional_network_labels"].measure)
    codes = resolve_network_node(net, "functional_network").astype(int)
    assert all(f"_{names[c].decode()}_" in f"_{label}_" for c, label in zip(codes, net.node_labels, strict=True))


@pytest.mark.parametrize("sidecar", sorted(NETWORK_DIR.glob("*atlas-Schaefer2018_seg-*.yaml")), ids=lambda f: f.stem)
def test_a_schaefer_vocabulary_s_example_is_a_parcel_of_its_own_network(sidecar):
    """The description of ``functional_network_labels`` gives as its example a parcel of the network it describes and that parcel's functional network: ``LH_Vis in 7Networks_LH_Vis_1`` in a 7Networks network."""
    meta = _sidecar(sidecar)
    example = re.search(r"\((\S+) in (\S+)\)", meta["parameters"]["functional_network_labels"]["description"])
    network, label = example.groups()
    assert label in [node["label"] for node in meta["nodes"]]
    assert f"_{network}_" in f"_{label}_"


@pytest.mark.parametrize(
    "names", [np.array(["Vis", "Default", "Vis"]), np.array([b"Vis", b"Default", b"Vis"])], ids=["str", "bytes"]
)
def test_a_categorical_attribute_held_as_text_resolves_as_its_text(names):
    """An attribute whose array holds names rather than codes into a vocabulary resolves as those names, never converted to numbers."""
    from tvbo.data.param_io import resolve_network_node

    net = _network(("weight",))
    net.set_array("nodes/network", names)
    net._template_parameters()["network"] = tvbo_datamodel.Parameter(name="network", measure="nodes/network")
    resolved = resolve_network_node(net, "network")
    assert resolved.dtype == names.dtype
    np.testing.assert_array_equal(resolved, names)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("shape", [(0,), (0, 2), (3, 0)], ids=str)
def test_an_empty_array_reads_back_from_its_csv_without_a_warning(tmp_path, shape):
    """An empty array's CSV is an empty file, or blank lines for an array of rows with no column, which reads back as the empty array it holds and warns of nothing."""
    net = _network(("weight",))
    net.set_array("nodes/none", np.zeros(shape))
    back = _saved_and_read_back(net, tmp_path / "csv", "csv")
    assert back.array("nodes/none").shape == shape


def test_a_selected_schaefer_network_saves_its_vocabulary_whole(tmp_path):
    """A node selection cuts each parcel's code to the kept parcels and serves the vocabulary of 34 networks whole, since its declaration says it does not run over the nodes, so the selected network saves and reads back."""
    net = Network(iri=f"tvbo:networks/{SCHAEFER}", node_selection={"drop": ["17Networks_LH_VisCent_ExStr_1"]})
    for binary_format in ("h5", "csv"):
        back = _saved_and_read_back(net, tmp_path / binary_format, binary_format)
        assert back.array("nodes/functional_network_labels").shape == (34,)
        assert back.array("nodes/parent_index").shape == (99,)


ATLASES = {
    "DesikanKilliany": ({"parcellation": {"atlas": {"name": "DesikanKilliany"}}}, {"hemisphere": "hemispheres"}),
    "Schaefer2018": (
        {"parcellation": {"atlas": {"name": "Schaefer2018"}}, "bids": {"segmentation": "17Networks", "scale": "200"}},
        {"functional_network": "functional_network_labels"},
    ),
}


@pytest.mark.parametrize("atlas", sorted(ATLASES))
def test_an_atlas_network_declares_its_categories_in_one_line(atlas):
    """A network named by its atlas alone declares the categories the atlas gives its regions, each with its vocabulary: Desikan-Killiany each region's hemisphere, from the atlas's terminology, and Schaefer2018 each parcel's functional network, from its curated network."""
    from tvbo.data.param_io import resolve_network_node

    spec, categories = ATLASES[atlas]
    net = Network(**spec)
    for name, vocabulary in categories.items():
        assert net.node_template.parameters[name].levels == vocabulary
        entries = net.parameters[vocabulary].value or net.array(net.parameters[vocabulary].measure)
        codes = resolve_network_node(net, name).astype(int)
        assert len(codes) == net.number_of_nodes and codes.max() < len(entries)
    if atlas == "DesikanKilliany":
        sides = dict(zip(net.node_labels, (net.parameters["hemispheres"].value[c] for c in codes), strict=True))
        assert (sides["left-thalamus"], sides["right-thalamus"], sides["brain-stem"]) == ("left", "right", "both")


def _short_run(network):
    """The first ten steps of a Generic2dOscillator on *network*, on tvboptim."""
    experiment = SimulationExperiment(
        dynamics={"name": "Generic2dOscillator", "iri": "tvbo:models/Generic2dOscillator"},
        network=network,
        integration={"method": "Heun", "step_size": 0.1, "duration": 1.0},
    )
    return np.asarray(experiment.run(format="tvboptim").integration.values)


@pytest.mark.parametrize("binary_format", ["h5", "csv"])
@pytest.mark.parametrize("atlas", sorted(ATLASES))
def test_an_atlas_network_reloads_and_runs_without_the_atlas(tmp_path, request, atlas, binary_format):
    """Saved, a network named by its atlas reads back with its categories and runs as it ran with the atlas there, once no atlas and no curated network can be read."""
    spec, categories = ATLASES[atlas]
    net = Network(**spec)
    (tmp_path / "saved").mkdir()
    net.save(tmp_path / "saved" / "n.yaml", binary_format=binary_format)
    expected = _short_run(net)
    request.getfixturevalue("hidden_atlases")
    back = Network.from_file(str(tmp_path / "saved" / "n.yaml"))
    for name, vocabulary in categories.items():
        declared = back.node_template.parameters[name]
        assert declared.levels == vocabulary
        np.testing.assert_array_equal(back.array(declared.measure), net.array(declared.measure))
    assert back.node_labels == net.node_labels
    np.testing.assert_array_equal(back.matrix("weight", format="dense"), net.matrix("weight", format="dense"))
    np.testing.assert_array_equal(_short_run(back), expected)


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(Network.list_db()))
def test_every_curated_network_reads_back_from_csv_as_from_hdf5(tmp_path, name):
    """Each curated network saved with CSV companions reads back as it reads back from an HDF5 save: every array, dtype and sparse format, every node label, and every declaration, beside those the CSV save adds for the arrays it links. One with no array to store is saved as its sidecar alone, the same for both."""
    net = Network(iri=f"tvbo:networks/{name}")
    for fmt in ("h5", "csv"):
        (tmp_path / fmt).mkdir()
        net.save(tmp_path / fmt / "n.yaml", binary_format=fmt)
    if not (tmp_path / "h5" / "n.h5").exists():
        assert sorted(p.name for p in (tmp_path / "csv").iterdir()) == ["n.yaml"]
        assert (tmp_path / "csv" / "n.yaml").read_text() == (tmp_path / "h5" / "n.yaml").read_text()
        return
    back = {fmt: Network.from_file(str(tmp_path / fmt / "n.yaml")) for fmt in ("h5", "csv")}
    _assert_same_arrays(_companion_arrays(back["csv"]), _companion_arrays(back["h5"]))
    assert back["csv"].node_labels == back["h5"].node_labels
    from tvbo.data.matrix_io import declared_arrays

    declared = {
        fmt: {path: (scope, param.name, param.levels) for path, (scope, param) in declared_arrays(b).items()}
        for fmt, b in back.items()
    }
    assert declared["h5"].items() <= declared["csv"].items()
