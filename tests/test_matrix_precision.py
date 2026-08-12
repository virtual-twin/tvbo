"""A stored matrix keeps the precision it was computed at.

Every writer in `matrix_io` used to cast to float32 unconditionally, so the store
decided a numerical property of data it did not compute. For a measured connectome
that is a fair trade — half the file, well inside the measurement's own error. For a
matrix the spec *computes*, it is silently a different matrix: Pang2023's experiment 4
declares `precision: float64` and integrated a cotangent Laplace-Beltrami operator that
had been round-tripped through float32, with nothing anywhere reporting the narrowing.

So the contract these pin is round-trip identity — what is written is what is read —
with narrowing available as a declared choice on the edge rather than a default. The
sparse index arrays are part of it: forcing int32 on `indptr`, a cumulative count of
nonzeros, silently wraps on a matrix scipy had already widened to int64 for.
"""

from __future__ import annotations

import numpy as np
import pytest

from tvbo.data.matrix_io import read_matrix, write_matrix

FORMATS = ["dense", "csr", "coo"]


@pytest.fixture
def store(tmp_path):
    import h5py

    path = tmp_path / "m.h5"

    def _roundtrip(matrix, fmt="dense", **kwargs):
        """The matrix as read back, and the dtype of every dataset that holds it."""
        with h5py.File(path, "w") as f:
            write_matrix(f.create_group("m"), matrix, fmt=fmt, **kwargs)
        with h5py.File(path, "r") as f:
            return read_matrix(f["m"]), {k: v.dtype for k, v in f["m"].items()}

    return _roundtrip


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_a_matrix_is_read_back_at_the_precision_it_was_written(store, fmt, dtype):
    m = np.eye(20, dtype=dtype) * np.pi
    got, _ = store(m, fmt=fmt)
    assert got.dtype == np.dtype(dtype)


@pytest.mark.parametrize("fmt", FORMATS)
def test_a_float64_matrix_survives_bit_for_bit(store, fmt):
    """The failure this replaces: a value that looks right and is not the one stored."""
    rng = np.random.default_rng(0)
    m = rng.random((30, 30))
    got, _ = store(m, fmt=fmt)
    np.testing.assert_array_equal(got, m)


@pytest.mark.parametrize("fmt", FORMATS)
def test_a_declared_precision_narrows_deliberately(store, fmt):
    m = np.eye(20) * np.pi
    got, _ = store(m, fmt=fmt, dtype="float32")
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, m.astype("float32"))


def test_an_integer_matrix_is_not_turned_into_floats(store):
    """A binary adjacency is not a measurement; widening it is as wrong as narrowing."""
    m = (np.eye(10) + np.eye(10, k=1)).astype("int8")
    got, _ = store(m)
    assert got.dtype == np.int8


def test_a_scipy_matrix_keeps_its_own_precision(store):
    import scipy.sparse as sp

    m = sp.random(600, 600, density=0.01, format="csr", random_state=0, dtype=np.float64)
    got, _ = store(m, fmt="csr")
    assert got.dtype == np.float64
    np.testing.assert_array_equal(got, m.toarray())


def test_sparse_indices_keep_the_width_scipy_chose(store):
    """`indptr` counts nonzeros cumulatively, so a forced int32 wraps where scipy widened."""
    import scipy.sparse as sp

    m = sp.random(500, 500, density=0.02, format="csr", random_state=0)
    m.indptr = m.indptr.astype(np.int64)
    m.indices = m.indices.astype(np.int64)
    _, dtypes = store(m, fmt="csr")
    assert dtypes["indptr"] == np.int64 and dtypes["indices"] == np.int64


def test_a_narrowed_write_leaves_the_indices_alone(store):
    """`dtype` is about values; casting positions with them would be a different bug."""
    import scipy.sparse as sp

    m = sp.random(600, 600, density=0.01, format="csr", random_state=0)
    _, dtypes = store(m, fmt="csr", dtype="float32")
    assert dtypes["data"] == np.float32
    assert np.issubdtype(dtypes["indptr"], np.integer)


class TestDeclaredOnTheEdge:
    """`Edge.dtype` is the declarative form of the same choice, beside `Edge.format`."""

    @staticmethod
    def _write(tmp_path, matrix, **edge_kwargs):
        import h5py

        from tvbo.data.network_io import _write_edges

        meta = {"edges": [{"name": "weight", "format": "dense", **edge_kwargs}]}
        path = tmp_path / "net.h5"
        with h5py.File(path, "w") as f:
            _write_edges(f, meta, {"weight": matrix}, {})
        with h5py.File(path, "r") as f:
            return np.asarray(f["edges/weight/data"])

    def test_an_edge_without_dtype_stores_what_it_was_given(self, tmp_path):
        got = self._write(tmp_path, np.eye(4) * np.pi)
        assert got.dtype == np.float64

    def test_an_edge_may_declare_a_narrower_store(self, tmp_path):
        got = self._write(tmp_path, np.eye(4) * np.pi, dtype="float32")
        assert got.dtype == np.float32

    def test_an_edge_parameter_inherits_the_edge_dtype(self, tmp_path):
        import h5py

        from tvbo.data.network_io import _write_edges

        meta = {"edges": [{"name": "weight", "format": "dense", "dtype": "float32"}]}
        path = tmp_path / "net.h5"
        with h5py.File(path, "w") as f:
            _write_edges(f, meta, {"weight": np.eye(4)}, {"weight": {"probability": np.eye(4) * np.pi}})
        with h5py.File(path, "r") as f:
            assert f["edges/weight/edge_parameters/probability/data"].dtype == np.float32

    def test_a_parameter_may_override_the_edge_it_hangs_off(self, tmp_path):
        import h5py

        from tvbo.data.network_io import _write_edges

        meta = {
            "edges": [
                {"name": "weight", "format": "dense", "dtype": "float32", "parameters": {"probability": {"dtype": "float64"}}}
            ]
        }
        path = tmp_path / "net.h5"
        with h5py.File(path, "w") as f:
            _write_edges(f, meta, {"weight": np.eye(4)}, {"weight": {"probability": np.eye(4) * np.pi}})
        with h5py.File(path, "r") as f:
            assert f["edges/weight/data"].dtype == np.float32
            assert f["edges/weight/edge_parameters/probability/data"].dtype == np.float64

    def test_the_slot_exists_on_the_generated_class(self):
        from tvbo.datamodel.schema import Edge

        assert Edge(label="weight", dtype="float32").dtype == "float32"
        assert Edge(label="weight").dtype is None


def test_a_result_sidecar_stores_what_its_descriptor_claims(tmp_path):
    """The descriptor reports each array's dtype; the writer used to contradict it."""
    import h5py
    import yaml

    from tvbo.data.experiment_result_io import save_sidecar

    J_i = np.linspace(0.0, 1.0, 8, dtype=np.float64)
    yaml_path = tmp_path / "exp_1_seed0.yaml"
    save_sidecar(yaml_path=yaml_path, parameters={"J_i": J_i}, experiment_yaml_hash="0" * 64)

    declared = yaml.safe_load(yaml_path.read_text())["parameters"][0]["dtype"]
    with h5py.File(yaml_path.with_suffix(".h5"), "r") as f:
        stored = f["parameters/J_i"]
        assert str(stored.dtype) == declared
        np.testing.assert_array_equal(np.asarray(stored), J_i)


def test_a_phenotype_measure_round_trips(tmp_path):
    """Per-subject scalars: nothing to save by narrowing, and a measure to corrupt."""
    from tvbo.classes.phenotype import Phenotype

    scores = np.array([0.1, 0.2, 0.30000000000000004, 0.4])
    pheno = Phenotype(dataset_id="cohort", data_file="cohort.h5", subjects=[f"s{i}" for i in range(4)], measures=["iq"])
    pheno.to_file(tmp_path / "cohort.yaml", values={"iq": scores})

    got = Phenotype.from_file(tmp_path / "cohort.yaml").get("iq")
    np.testing.assert_array_equal(got, scores)
