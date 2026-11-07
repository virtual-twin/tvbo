"""The array formats tvbo reads, each through its store, and `read_array`, the one function a matrix outside a companion is read with.

A store answers names and headers without reading a value, which the `_read_values` seam counts; the delimiter of a text file follows its suffix unless declared; a matrix is a DataArray under declared dims, never under dims guessed from its shape.
"""

import io
import zipfile

import numpy as np
import pytest
from scipy import sparse

from tvbo.data import matrix_io
from tvbo.data.array_formats import (
    CANONICAL_EDGE_DIMS,
    MatH5Store,
    TextStore,
    declare,
    is_symmetric,
    open_store,
    read_array,
    sender_row,
)
from tvbo.export.registry import array_format_for, resolve_array_format

from .array_files import write_arrays

W = np.array([[0.0, 0.0, 3.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
"""Receiver-row: ``W[1, 0]`` is the edge ``0 -> 1``."""
S = W + W.T


@pytest.fixture
def value_reads(monkeypatch):
    reads = []
    original = matrix_io._read_values
    monkeypatch.setattr(matrix_io, "_read_values", lambda dataset: reads.append(dataset) or original(dataset))
    return reads


KEYED = ["h5", "zarr", "npz", "mat", "mat73"]


@pytest.mark.parametrize("kind", KEYED)
def test_a_keyed_format_lists_and_describes_without_reading(tmp_path, kind, value_reads):
    path, key = write_arrays(tmp_path, kind, weight=W, other=S)
    store = open_store(path)
    with store:
        listed = store.keys(key.rsplit("/", 1)[0]) if "/" in key else store.keys()
        info = store.info(key)
    assert key in listed and "other" in store.keys()
    assert info.shape == (3, 3) and info.dtype == np.float64
    with pytest.raises(ValueError, match="holds its arrays by key"):
        read_array(path, dims=CANONICAL_EDGE_DIMS)
    assert value_reads == [], "names and headers read no values"
    with store:
        np.testing.assert_array_equal(store.read(key), W)
    assert len(value_reads) == 1


def test_a_matlab_v73_file_is_read_in_matlabs_orientation(tmp_path):
    path, _key = write_arrays(tmp_path, "mat73", weight=W, other=S)
    store = open_store(path)
    assert isinstance(store, MatH5Store)
    with store:
        assert store.info("weight").shape == (3, 3)
        np.testing.assert_array_equal(store.read("weight"), W)
    rect, _key = write_arrays(tmp_path / "rect", "mat73", m=np.arange(6.0).reshape(2, 3))
    with open_store(rect) as store:
        assert store.info("m").shape == (2, 3)
        np.testing.assert_array_equal(store.read("m"), np.arange(6.0).reshape(2, 3))


def test_a_numpy_array_is_memory_mapped_and_holds_any_key(tmp_path, value_reads):
    path = tmp_path / "w.npy"
    np.save(path, W)
    store = open_store(path)
    assert store.keys() == []
    assert store.info("weight").shape == (3, 3) and value_reads == []
    assert isinstance(store.read("anything"), np.memmap)
    np.testing.assert_array_equal(store.read("weight"), W)


@pytest.mark.parametrize(("suffix", "delimiter"), [(".csv", ","), (".tsv", "\t"), (".txt", " ")])
def test_a_text_file_is_split_by_what_its_suffix_means(tmp_path, suffix, delimiter):
    path = tmp_path / f"w{suffix}"
    np.savetxt(path, W, delimiter=delimiter)
    store = open_store(path)
    assert isinstance(store, TextStore) and store.keys() == []
    with pytest.raises(KeyError, match="no header"):
        store.info("weight")
    np.testing.assert_array_equal(store.read("weight"), W)


def test_a_declared_delimiter_overrides_the_suffix(tmp_path):
    path = tmp_path / "w.txt"
    np.savetxt(path, W, delimiter=",")
    with pytest.raises(ValueError):
        open_store(path).read("weight")
    np.testing.assert_array_equal(open_store(path, delimiter=",").read("weight"), W)
    np.testing.assert_array_equal(read_array(path, dims=CANONICAL_EDGE_DIMS, delimiter=",").values, W)
    np.savetxt(tmp_path / "w.dat", W, delimiter=" ")
    np.testing.assert_array_equal(open_store(tmp_path / "w.dat", format="text").read("weight"), W)
    np.save(tmp_path / "w.npy", W)
    with pytest.raises(ValueError, match=r"a 'npy' store takes none of the options \['delimiter'\]"):
        open_store(tmp_path / "w.npy", delimiter=",")


def test_an_open_file_is_read_as_text_by_the_format_it_names(tmp_path):
    archive = tmp_path / "c.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("weights.txt", "\n".join(" ".join(f"{x:g}" for x in row) for row in W))
    with zipfile.ZipFile(archive) as zf:
        da = read_array(zf.open("weights.txt"), format="text", dims=CANONICAL_EDGE_DIMS)
    np.testing.assert_array_equal(da.values, W)
    np.testing.assert_array_equal(read_array(io.BytesIO(b"0 1\n1 0\n"), format="text").values, [[0, 1], [1, 0]])


def test_a_suffix_no_format_claims_is_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match=r"no array format claims '\.yaml'"):
        array_format_for(tmp_path / "net.yaml")
    with pytest.raises(ValueError, match="No array format registered as 'parquet'"):
        resolve_array_format("parquet")


def test_a_bare_directory_with_a_zarr_marker_is_zarr(tmp_path):
    zarr = pytest.importorskip("zarr")
    path = tmp_path / "store"
    zarr.open(str(path), mode="w").create_array("x", data=W)
    assert array_format_for(path).key == "zarr"
    np.testing.assert_array_equal(open_store(path).read("x"), W)


def test_symmetry_is_judged_without_densifying():
    assert is_symmetric(S) and not is_symmetric(W)
    csr = sparse.csr_matrix(S)
    assert is_symmetric(csr) and not is_symmetric(sparse.csr_matrix(W))
    assert is_symmetric(sparse.csr_matrix((3, 3)))
    assert not is_symmetric(np.ones((2, 3)))


def test_dims_are_declared_never_taken_from_the_shape(tmp_path):
    da = declare(W, CANONICAL_EDGE_DIMS, "probe")
    assert da.dims == CANONICAL_EDGE_DIMS
    flipped = declare(W.T, ("source", "target"), "probe")
    assert flipped.dims == CANONICAL_EDGE_DIMS
    np.testing.assert_array_equal(flipped.values, W)
    undeclared = declare(S, None, "probe")
    assert undeclared.dims == CANONICAL_EDGE_DIMS
    with pytest.raises(ValueError, match="probe: no dims are declared .* not symmetric"):
        declare(W, None, "probe")
    with pytest.raises(ValueError, match=r"probe: is not a square matrix: shape \(2, 3\)"):
        declare(np.ones((2, 3)), None, "probe")
    with pytest.raises(ValueError, match="probe: has non-finite entries"):
        declare(np.array([[0.0, np.nan], [1.0, 0.0]]), ("source", "target"), "probe")
    with pytest.raises(ValueError, match="2 dims .* shape \\(3,\\)"):
        declare(np.ones(3), ("node", "axis"), "probe")
    with pytest.raises(TypeError, match="sparse matrix is read through its store"):
        declare(sparse.csr_matrix(S), CANONICAL_EDGE_DIMS, "probe")
    assert declare(np.arange(3.0), ("node",), "probe").dims == ("node",)


def test_sender_row_is_the_named_transpose():
    np.testing.assert_array_equal(sender_row(W), W.T)


def test_a_companion_in_any_format_feeds_the_lazy_store(tmp_path):
    from tvbo.data.matrix_io import LazyArrayStore

    meta = {"edges": [{"label": "weight"}]}
    for kind in ("npz", "mat"):
        path, _key = write_arrays(tmp_path / kind, kind, weight=W, other=S)
        store = LazyArrayStore(path, meta)
        assert store.names == ["weight"] and "weight" in store
        np.testing.assert_array_equal(store["weight"], W)
        assert store.edge_params_of("weight") == {}
    np.save(tmp_path / "w.npy", W)
    np.testing.assert_array_equal(LazyArrayStore(tmp_path / "w.npy", {})["weight"], W)


def test_a_data_file_in_no_array_format_fails_on_read_not_on_load(tmp_path):
    from tvbo.data.matrix_io import LazyArrayStore

    store = LazyArrayStore(tmp_path / "other.yaml", {})
    with pytest.raises(KeyError, match="other.yaml"):
        store["weight"]


def test_tvbo_writes_h5_zarr_and_csv_companions_only(tmp_path):
    from tvbo import Network
    from tvbo.data.network_io import save_network

    with pytest.raises(ValueError, match="hdf5, zarr and csv companions, not 'npy'"):
        save_network(Network.from_matrix(S), tmp_path / "net.yaml", binary_format="npy")
