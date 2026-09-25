"""Low-level matrix read/write for HDF5 groups and Zarr groups.

Supports dense, CSR, and COO formats. Both HDF5 (h5py.Group) and Zarr (zarr.Group) implement the same array-store interface, so a single pair of read/write functions handles both backends.

See §12.1 of the tvbo HDF5 format proposal v0.7.
"""

import os
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def resolve_staged_path(path) -> Path:
    """Resolve an artifact path against a packed kit's staging directory.

    A frozen backend script carries the absolute path its author read. A packed kit copies every artifact that script loads — sourced/produced observer constants and sourced model or coupling parameters alike — into its own ``constants/`` directory, keyed by basename.
    When the author's path is absent, as it is on any other machine, the file is looked up under ``$TVBO_CONSTANTS_DIR`` and then the run directory's ``constants/``.

    An existing path is returned untouched, so a run on the authoring machine never consults the staging directory and cannot pick up a same-named file by accident.
    """
    p = Path(path)
    if p.exists():
        return p
    for base in (os.environ.get("TVBO_CONSTANTS_DIR"), "constants"):
        if base and (Path(base) / p.name).is_file():
            return Path(base) / p.name
    return p


def _create_ds(grp, name, *, data, **kwargs):
    """Create a dataset compatible with both h5py and zarr v3."""
    try:
        import zarr

        if isinstance(grp, zarr.Group):
            return grp.create_array(name, data=data, **kwargs)
    except ImportError:
        pass
    return grp.create_dataset(name, data=data, **kwargs)


# ── Format selection ──────────────────────────────────────────────────


def auto_format(matrix) -> str:
    """Select optimal storage format based on empirical analysis (§11).

    Rules (data-driven from tvbo corpus measurements):
    - N < 500 or fill > 30%: dense + gzip wins
    - Otherwise: CSR

    Handles both dense arrays and scipy sparse matrices without densifying the input.

    Parameters
    ----------
    matrix : array-like or scipy.sparse matrix
        Matrix to analyze.

    Returns:
    -------
    str
        "dense" or "csr"
    """
    from scipy import sparse

    if sparse.issparse(matrix):
        n = max(matrix.shape)
        fill = matrix.nnz / (matrix.shape[0] * matrix.shape[1]) if matrix.shape[0] > 0 else 0
    else:
        arr = np.asarray(matrix)
        n = max(arr.shape)
        fill = np.count_nonzero(arr) / arr.size if arr.size > 0 else 0
    if n < 500 or fill > 0.30:
        return "dense"
    return "csr"


# ── Write ─────────────────────────────────────────────────────────────


def _at_precision(matrix, dtype):
    """The matrix as it goes to disk: its own precision unless one is declared.

    A writer that picks the precision decides a numerical property of data it did not compute. Narrowing a measured connectome is a fair trade for half the file; narrowing a differential operator someone integrates at float64 is a different operator, and nothing downstream can tell it happened. So the cast is opt-in.
    """
    return matrix if dtype is None else matrix.astype(dtype)


def _write_dense(grp, matrix, dtype=None):
    from scipy import sparse

    arr = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    arr = _at_precision(arr, dtype)
    chunks = tuple(min(s, 128) for s in arr.shape)
    _create_ds(grp, "data", data=arr, chunks=chunks)


def _write_csr(grp, matrix, dtype=None):
    m = _at_precision(csr_matrix(matrix), dtype)
    _create_ds(grp, "data", data=m.data)
    _create_ds(grp, "indices", data=m.indices)
    _create_ds(grp, "indptr", data=m.indptr)


def _write_coo(grp, matrix, dtype=None):
    m = _at_precision(coo_matrix(matrix), dtype)
    _create_ds(grp, "data", data=m.data)
    _create_ds(grp, "row", data=m.row)
    _create_ds(grp, "col", data=m.col)


_WRITERS = {"dense": _write_dense, "csr": _write_csr, "coo": _write_coo}


def write_matrix(grp, matrix, fmt: str = "dense", dtype=None):
    """Write a matrix to an HDF5/Zarr group in the specified format.

    Parameters
    ----------
    grp : h5py.Group or zarr.Group
        Target group.
    matrix : array-like
        Matrix data to write.
    fmt : str
        Storage format: "dense", "csr", or "coo".
    dtype : str or numpy.dtype, optional
        Store at this precision instead of the matrix's own. Index arrays of the sparse formats are unaffected — they keep the width scipy chose, which is int64 exactly when the matrix is too large for int32 to address.

    The shape is read off ``.shape`` directly, never via ``np.asarray`` — a scipy sparse matrix survives that call as a 0-d object array, which would record an empty shape.
    """
    grp.attrs["format"] = str(fmt)
    shape = matrix.shape if hasattr(matrix, "shape") else np.asarray(matrix).shape
    grp.attrs["shape"] = list(shape)
    _WRITERS[str(fmt)](grp, matrix, dtype)


# ── Read ──────────────────────────────────────────────────────────────


def _read_values(dataset):
    """Every value read in this module, in one place.

    The array layer's contract is that metadata questions read no values, and a contract nobody can measure is a wish. Routing each read through here gives a test one seam to count.
    """
    return dataset[()]


def read_matrix(grp):
    """Read a matrix from an HDF5/Zarr group, in the format it is stored in.

    A csr or coo group comes back as the scipy sparse matrix it describes; a dense group as an ndarray. Densifying is the caller's decision — a 32k-vertex surface stored sparse is 8.4 GB dense, and a reader that made that decision on its own would make it for the consumer that cannot afford it.

    Parameters
    ----------
    grp : h5py.Group or zarr.Group
        Source group containing format/shape attrs and data datasets.

    Returns:
    -------
    np.ndarray or scipy.sparse matrix
    """
    fmt = grp.attrs["format"]
    shape = tuple(grp.attrs["shape"])
    if fmt == "dense":
        return np.asarray(_read_values(grp["data"]))
    elif fmt == "csr":
        return csr_matrix((_read_values(grp["data"]), _read_values(grp["indices"]), _read_values(grp["indptr"])), shape=shape)
    elif fmt == "coo":
        return coo_matrix((_read_values(grp["data"]), (_read_values(grp["row"]), _read_values(grp["col"]))), shape=shape)
    else:
        raise ValueError(f"Unknown matrix format: {fmt}")


def read_edge(store, name: str) -> tuple:
    """One template edge's matrix and its edge-parameter matrices, from an open store.

    Works identically for ``h5py.File`` and ``zarr.Group`` — both support ``"path" in store`` and ``store["path"]``. Raises ``KeyError`` when the store holds no such edge, so a miss is a miss and not an empty result a caller has to test for.
    """
    edge_path = f"edges/{name}"
    if edge_path not in store:
        raise KeyError(name)
    matrix = read_matrix(store[edge_path])
    params = {}
    ep_path = f"{edge_path}/edge_parameters"
    if ep_path in store:
        for pname in store[ep_path]:
            params[pname] = read_matrix(store[f"{ep_path}/{pname}"])
    return matrix, params


def template_edges(edges) -> list:
    """Template edges = entries without source/target (matrix measures).

    Works with both dicts (from ``yaml_loader.load_as_dict``) and LinkML ``Edge`` objects (from ``Network.edges``).
    """
    if not edges:
        return []
    return [e for e in edges if (e.get("source") if isinstance(e, dict) else getattr(e, "source", None)) is None]


def edge_name(e) -> str:
    """An edge's matrix name, from either spelling the sidecars use."""
    if isinstance(e, dict):
        return e.get("name") or e.get("label") or "weight"
    return getattr(e, "name", None) or getattr(e, "label", None) or "weight"


# ── Lazy array store ──────────────────────────────────────────────────


@dataclass(frozen=True)
class ArrayInfo:
    """What a companion says about one array without reading its values."""

    path: str
    shape: tuple
    dtype: np.dtype
    format: str
    nbytes: int


class _LazyView(Mapping):
    """A read-per-key view over a store: ``keys()`` from metadata, ``[]`` from the file."""

    def __init__(self, keys, read):
        self._keys = keys
        self._read = read

    def __getitem__(self, key):
        return self._read(key)

    def __iter__(self):
        return iter(self._keys())

    def __len__(self):
        return len(self._keys())

    def __contains__(self, key):
        return key in self._keys()

    def __repr__(self):
        return f"{type(self).__name__}({list(self)})"


class LazyArrayStore:
    """A companion binary file (HDF5 / Zarr / CSV), read one array at a time.

    Holds the path and the sidecar's edge declarations and nothing else at construction. Names, shapes, dtypes and formats come from the sidecar and the file's header; a value is read the first time it is asked for by name, and only that value. So ``"weight" in store`` costs a header read on a companion that also carries a leadfield and a mesh, and ``store["weight"]`` reads the weight matrix and leaves the rest on disk.

    A read returns the array in the format it is stored in — a csr companion yields a csr matrix. Every read routes through :func:`_read_values`, the seam an I/O-counting test measures.

    Use as a context manager to hold the file open across many reads; outside one, each read opens and closes the file, which is correct and safe and wrong under iteration.

    Args:
        companion_path: Path to the companion binary file (.h5, .zarr, .csv).
        meta_dict: Raw sidecar dict (from ``yaml_loader.load_as_dict``) for edge declarations.
    """

    def __init__(self, companion_path: Path, meta_dict: dict):
        self._path = Path(companion_path)
        self._meta = meta_dict
        self._ext = self._path.suffix.lower()
        self._cache: dict[str, Any] = {}
        self._params_cache: dict[str, dict[str, Any]] = {}
        self._names: list[str] | None = None
        self._handle = None
        self._depth = 0

    def _template_edges(self) -> list:
        return template_edges(self._meta.get("edges", []) or [])

    @contextmanager
    def _open(self):
        """The open store, held for the duration if a ``with store:`` block already holds it."""
        if self._handle is not None:
            yield self._handle
            return
        if self._ext in (".h5", ".hdf5"):
            import h5py

            with h5py.File(self._path, "r") as f:
                yield f
        elif self._ext == ".zarr" or self._path.is_dir():
            import zarr

            yield zarr.open(str(self._path), "r")
        else:
            raise KeyError(f"{self._path} is not an array store")

    def __enter__(self):
        """Hold the file open for the block. Nesting is counted, so an inner block cannot close the handle the outer one is still reading through."""
        self._depth += 1
        if self._handle is not None:
            return self
        if self._ext in (".h5", ".hdf5"):
            import h5py

            self._handle = h5py.File(self._path, "r")
        elif self._ext == ".zarr" or self._path.is_dir():
            import zarr

            self._handle = zarr.open(str(self._path), "r")
        return self

    def __exit__(self, *exc):
        self._depth = max(self._depth - 1, 0)
        if self._depth:
            return
        handle, self._handle = self._handle, None
        if handle is not None and hasattr(handle, "close"):
            handle.close()

    @property
    def names(self) -> list[str]:
        """The edge matrices this companion carries, without reading a value.

        From the sidecar's template edges when it declares any; otherwise from the file's ``edges/`` group listing, which is a header read. A CSV companion carries exactly one.
        """
        if self._names is None:
            declared = [edge_name(e) for e in self._template_edges()]
            if declared:
                self._names = declared
            elif self._ext == ".csv":
                self._names = ["weight"]
            else:
                with self._open() as store:
                    self._names = list(store["edges"]) if "edges" in store else []
        return list(self._names)

    def info(self, key: str) -> ArrayInfo:
        """Shape, dtype and format of one array, from the file's header alone.

        ``key`` is an edge name (``"weight"``) or any dataset path (``"nodes/coordinates"``, ``"mesh/vertices"``). Raises ``KeyError`` when the file holds neither.
        """
        if self._ext == ".csv":
            raise KeyError(f"{self._path} has no header to read; load it to learn its shape")
        with self._open() as store:
            for path in (f"edges/{key}", key):
                if path not in store:
                    continue
                node = store[path]
                if hasattr(node, "attrs") and "format" in node.attrs:
                    data = node["data"]
                    return ArrayInfo(
                        path, tuple(node.attrs["shape"]), np.dtype(data.dtype), str(node.attrs["format"]), int(data.nbytes)
                    )
                if hasattr(node, "shape"):
                    return ArrayInfo(path, tuple(node.shape), np.dtype(node.dtype), "dataset", int(node.nbytes))
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return key in self._cache or key in self.names

    def __getitem__(self, key: str):
        """One edge matrix, in its stored format, read on first access and kept."""
        if key not in self._cache:
            if self._ext == ".csv":
                if key not in self.names:
                    raise KeyError(key)
                self._cache[key] = np.loadtxt(self._path, delimiter=" ")
                self._params_cache[key] = {}
            else:
                with self._open() as store:
                    self._cache[key], self._params_cache[key] = read_edge(store, key)
        return self._cache[key]

    def edge_params_of(self, key: str) -> dict:
        """The edge-parameter matrices stored beside one edge matrix."""
        self[key]
        return self._params_cache[key]

    @property
    def arrays(self) -> Mapping:
        """Edge matrices as a read-per-key mapping.

        ``dict(store.arrays)`` reads every matrix, which is what a caller spelling that asks for; ``store.arrays["weight"]`` reads one.
        """
        return _LazyView(lambda: self.names, self.__getitem__)

    @property
    def edge_params(self) -> Mapping:
        """Edge parameters as a read-per-key mapping, keyed like :attr:`arrays`."""
        return _LazyView(lambda: self.names, self.edge_params_of)

    @property
    def _loaded(self) -> bool:
        """True once every declared matrix is resident."""
        return bool(self._names is not None and all(n in self._cache for n in self._names))

    def dataset_keys(self, prefix: str = "") -> list[str]:
        """Dataset paths under ``prefix`` (e.g. ``"nodes"``), empty when it holds none.

        Lets a caller carry datasets across a re-save without modelling each one, which is what keeps a companion's per-node arrays alive through ``save_network``.
        """
        if self._ext == ".csv":
            return []
        with self._open() as store:
            grp = store.get(prefix) if prefix else store
            if grp is None:
                return []
            if self._ext in (".h5", ".hdf5"):
                import h5py

                names = [name for name, obj in grp.items() if isinstance(obj, h5py.Dataset)]
            else:
                names = list(getattr(grp, "array_keys", lambda: [])())
        return [f"{prefix}/{n}" if prefix else n for n in names]

    def read_dataset(self, key: str):
        """Read an arbitrary dataset by path (e.g. ``"nodes/parent_index"``)."""
        with self._open() as store:
            if key in store:
                return np.asarray(_read_values(store[key]))
        raise KeyError(key)
