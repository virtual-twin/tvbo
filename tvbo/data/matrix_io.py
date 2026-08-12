"""Low-level matrix read/write for HDF5 groups and Zarr groups.

Supports dense, CSR, and COO formats. Both HDF5 (h5py.Group) and
Zarr (zarr.Group) implement the same array-store interface, so a single pair of read/write functions handles both backends.

See §12.1 of the tvbo HDF5 format proposal v0.7.
"""

import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix, coo_matrix


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

    Returns
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

    A writer that picks the precision decides a numerical property of data it did not compute. Narrowing a measured connectome is a fair trade for half the file;
    narrowing a differential operator someone integrates at float64 is a different operator, and nothing downstream can tell it happened. So the cast is opt-in.
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
        Store at this precision instead of the matrix's own. Index arrays of the
        sparse formats are unaffected — they keep the width scipy chose, which is
        int64 exactly when the matrix is too large for int32 to address.

    The shape is read off ``.shape`` directly, never via ``np.asarray`` — a scipy sparse matrix survives that call as a 0-d object array, which would record an empty shape.
    """
    grp.attrs["format"] = str(fmt)
    shape = matrix.shape if hasattr(matrix, "shape") else np.asarray(matrix).shape
    grp.attrs["shape"] = list(shape)
    _WRITERS[str(fmt)](grp, matrix, dtype)


# ── Read ──────────────────────────────────────────────────────────────


def read_matrix(grp) -> np.ndarray:
    """Read a matrix from an HDF5/Zarr group, returning dense numpy array.

    Parameters
    ----------
    grp : h5py.Group or zarr.Group
        Source group containing format/shape attrs and data datasets.

    Returns
    -------
    np.ndarray
        Dense numpy array.
    """
    fmt = grp.attrs["format"]
    shape = tuple(grp.attrs["shape"])
    if fmt == "dense":
        return np.asarray(grp["data"])
    elif fmt == "csr":
        return csr_matrix(
            (np.asarray(grp["data"]), np.asarray(grp["indices"]), np.asarray(grp["indptr"])), shape=shape
        ).toarray()
    elif fmt == "coo":
        return coo_matrix((np.asarray(grp["data"]), (np.asarray(grp["row"]), np.asarray(grp["col"]))), shape=shape).toarray()
    else:
        raise ValueError(f"Unknown matrix format: {fmt}")


# ── Lazy array store ──────────────────────────────────────────────────


class LazyArrayStore:
    """Lazy-loading wrapper for companion binary files (HDF5/Zarr/CSV).

    Stores a reference to the companion path and sidecar metadata.
    Arrays are loaded on first access and cached thereafter.

    Parameters
    ----------
    companion_path : Path
        Path to the companion binary file (.h5, .zarr, .csv).
    meta_dict : dict
        Raw sidecar dict (from yaml_loader.load_as_dict) for edge traversal.
    """

    def __init__(self, companion_path: Path, meta_dict: dict):
        self._path = companion_path
        self._meta = meta_dict
        self._ext = companion_path.suffix.lower()
        self._cache: dict[str, np.ndarray] = {}
        self._params_cache: dict[str, dict[str, np.ndarray]] = {}
        self._loaded = False

    def _template_edges(self) -> list:
        """Template edges = entries without source/target (matrix measures)."""
        return [e for e in (self._meta.get("edges", []) or []) if e.get("source") is None]

    @staticmethod
    def _edge_name(e: dict) -> str:
        """Get edge name from dict — supports both 'name' (new) and 'label' (old)."""
        return e.get("name") or e.get("label", "weight")

    def _ensure_loaded(self):
        """Load all arrays on first access."""
        if self._loaded:
            return
        edges = self._template_edges()

        if self._ext == ".csv":
            name = self._edge_name(edges[0]) if edges else "weight"
            self._cache[name] = np.loadtxt(self._path, delimiter=" ")

        elif self._ext in (".h5", ".hdf5"):
            import h5py

            with h5py.File(self._path, "r") as f:
                self._cache, self._params_cache = _read_edges_from_store(f, edges)

        elif self._ext == ".zarr" or self._path.is_dir():
            import zarr

            self._cache, self._params_cache = _read_edges_from_store(zarr.open(str(self._path), "r"), edges)

        self._loaded = True

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        """Edge matrices, loaded lazily on first access."""
        self._ensure_loaded()
        return self._cache

    @property
    def edge_params(self) -> dict[str, dict[str, np.ndarray]]:
        """Edge parameters, loaded lazily on first access."""
        self._ensure_loaded()
        return self._params_cache

    def __contains__(self, key: str) -> bool:
        self._ensure_loaded()
        return key in self._cache

    def __getitem__(self, key: str) -> np.ndarray:
        self._ensure_loaded()
        return self._cache[key]

    def read_dataset(self, key: str) -> np.ndarray:
        """Read an arbitrary dataset by path (e.g. ``"nodes/parent_index"``)."""
        if self._ext in (".h5", ".hdf5"):
            import h5py

            with h5py.File(self._path, "r") as f:
                if key in f:
                    return f[key][()]
                raise KeyError(key)
        elif self._ext == ".zarr" or self._path.is_dir():
            import zarr

            z = zarr.open(str(self._path), "r")
            return np.asarray(z[key])
        raise KeyError(key)


def _read_edges_from_store(store, template_edges: list) -> tuple[dict, dict]:
    """Read all template-edge matrices + edge parameters from a store.

    Works identically for h5py.File and zarr.Group — both support
    `"path" in store` and `store["path"]` access.
    """
    arrays, params = {}, {}

    # Determine edge names: from template metadata or store discovery
    if template_edges:
        edge_names = [e.get("name") or e.get("label") for e in template_edges]
    elif "edges" in store:
        edge_names = list(store["edges"])
    else:
        edge_names = []

    for name in edge_names:
        edge_path = f"edges/{name}"
        if edge_path not in store:
            continue
        arrays[name] = read_matrix(store[edge_path])
        params[name] = {}
        ep_path = f"{edge_path}/edge_parameters"
        if ep_path in store:
            for pname in store[ep_path]:
                params[name][pname] = read_matrix(store[f"{ep_path}/{pname}"])
    return arrays, params
