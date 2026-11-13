"""Low-level matrix read/write for HDF5 groups and Zarr groups.

Supports dense, CSR, and COO formats. Both HDF5 (h5py.Group) and Zarr (zarr.Group) implement the same array-store interface, so a single pair of read/write functions handles both backends. A CSV companion holds one array per file, which the sidecar names on the record of what it holds (:func:`csv_links`, :func:`write_csv_array`, :func:`read_csv_array`).

See §12.1 of the tvbo HDF5 format proposal v0.7.
"""

import ast
import csv
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


# ── CSV ───────────────────────────────────────────────────────────────

MESH_FIELDS = {
    "mesh/vertices": ("vertices_field", "float32"),
    "mesh/elements": ("elements", "int32"),
    "mesh/normals": ("normals", "float32"),
}
"""Each mesh array by its path in a companion: the ``Mesh`` field a CSV save names its CSV in, and the dtype every writer stores it at."""


def is_csv(path) -> bool:
    """Whether *path* names a CSV file."""
    return isinstance(path, str) and path.lower().endswith(".csv")


def _slot(record, name):
    """The slot *name* of *record*, a sidecar mapping or a datamodel object."""
    return record.get(name) if isinstance(record, Mapping) else getattr(record, name, None)


def named_entries(collection) -> list:
    """``(name, entry)`` for each entry of a keyed collection, in either spelling: the mapping tvbo writes, or a list of entries that carry their ``name``."""
    if isinstance(collection, Mapping):
        return list(collection.items())
    return [(_slot(entry, "name"), entry) for entry in collection or []]


def declared_arrays(network) -> dict:
    """``{path: (scope, parameter)}`` for each array *network* declares as its own: a parameter whose ``measure`` is the array's path in the network's companion, with no ``source`` but the CSV a CSV save writes for it.

    *scope* is ``"node"`` for a ``node_template`` parameter, whose array runs over the nodes, and ``"network"`` for a network parameter, such as the vocabulary a categorical attribute's ``levels`` names. *network* is a sidecar mapping or a ``Network``.
    """
    template = _slot(network, "node_template")
    out = {}
    for scope, params in (
        ("node", _slot(template, "parameters") if template is not None else None),
        ("network", _slot(network, "parameters")),
    ):
        for _, param in named_entries(params):
            measure, source = _slot(param, "measure"), _slot(param, "source")
            if measure and (not source or is_csv(str(source))):
                out.setdefault(str(measure).lstrip("/"), (scope, param))
    return out


def csv_links(meta: Mapping) -> dict:
    """``{path: link}`` for each array the sidecar *meta* names a CSV for, keyed by the path the array has in an HDF5 companion.

    A layer's CSV is its template edge's ``data_file``, an edge parameter's its ``source``, a declared array's (:func:`declared_arrays`) its ``source``, and a mesh array's the ``mesh`` field of :data:`MESH_FIELDS`. A link holds the CSV's ``file``, relative to the sidecar, and what the file does not record: the ``dtype``, for a layer the sparse ``format``, and for a declared array the ``shape``. A layer whose ``data_file`` names any other file is refused.
    """
    links = {}
    for edge in template_edges(meta.get("edges")):
        layer, fmt = edge_name(edge), edge.get("format")
        if edge.get("data_file") and not is_csv(edge["data_file"]):
            raise ValueError(
                f"The layer {layer!r} names {edge['data_file']} as its `data_file`, which names the CSV holding that layer alone. An HDF5 or Zarr companion holds every layer under its path and is the network's own `data_file`."
            )
        if is_csv(edge.get("data_file")):
            links[f"edges/{layer}"] = {"file": edge["data_file"], "dtype": edge.get("dtype"), "format": fmt}
        for name, param in named_entries(edge.get("parameters")):
            if isinstance(param, Mapping) and is_csv(param.get("source")):
                links[f"edges/{layer}/edge_parameters/{name}"] = {
                    "file": param["source"],
                    "dtype": param.get("dtype") or edge.get("dtype"),
                    "format": fmt,
                }
    for path, (_, param) in declared_arrays(meta).items():
        if is_csv(param.get("source")):
            links[path] = {"file": param["source"], "dtype": param.get("dtype"), "shape": param.get("shape")}
    mesh = meta.get("mesh") or {}
    for path, (field, dtype) in MESH_FIELDS.items():
        if is_csv(mesh.get(field)):
            links[path] = {"file": mesh[field], "dtype": dtype}
    return links


def parse_shape(text):
    """The shape *text* states, ``(100,)`` or ``(100, 3)``, as a tuple of ints, or ``None`` when it names a symbol (``(n_nodes,)``) or is unset."""
    try:
        value = ast.literal_eval(str(text))
    except (ValueError, SyntaxError):
        return None
    value = value if isinstance(value, (tuple, list)) else (value,)
    return tuple(value) if all(isinstance(side, int) for side in value) else None


def _text_rows(values):
    """The rows of a 1-D or 2-D array, a 1-D array one value to a row."""
    return values[:, None] if values.ndim == 1 else values


def _formatted(values) -> list[str]:
    """Each row of a numeric array as one line of space-separated fields: integers exactly, other numbers to 17 significant digits, which reads every float64 back exactly, and a complex number as ``<real><+imag>j``."""
    if values.dtype.kind == "c":
        return [" ".join(f"{v.real:.17g}{v.imag:+.17g}j" for v in row) for row in _text_rows(values)]
    fmt = "%d" if values.dtype.kind in "biu" else "%.17g"
    return [" ".join(fmt % v for v in row) for row in _text_rows(values).tolist()]


def write_csv_array(path, values) -> None:
    """Write a 1-D or 2-D array to *path* as rows of space-separated fields, a 1-D array one value to a line, or a scipy sparse matrix as one ``row column value`` line per stored entry, in row order.

    Numbers are written as :func:`_formatted` writes them. Text is written one value to a field through the ``csv`` module, quoted where it holds a space or a quote; bytes, fixed-width or the variable-length text h5py reads into an ``object`` array, as the UTF-8 text they encode. The rows record neither the dtype, nor the shape of one row or one column, nor a sparse matrix's shape, so the caller keeps them.
    """
    from scipy import sparse

    if sparse.issparse(values):
        coo = values.tocoo()
        cells = zip(coo.row.tolist(), coo.col.tolist(), _formatted(coo.data), strict=True)
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(f"{r} {c} {v}\n" for r, c, v in cells)
        return
    values = np.asarray(values)
    if values.dtype.kind in "OSU":
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f, delimiter=" ", lineterminator="\n").writerows(
                [v.decode("utf-8") if isinstance(v, bytes) else v for v in row] for row in _text_rows(values)
            )
    else:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(line + "\n" for line in _formatted(values))


def read_csv_array(path, shape=None, dtype=None, fmt=None):
    """The array :func:`write_csv_array` wrote to *path*, at *dtype*, float64 when ``None``.

    Dense, it comes back at *shape*, or 2-D as the rows give it when that is ``None``; text at a bytes or ``object`` dtype comes back as bytes, as h5py reads it. With *fmt* csr or coo the file holds one line per stored entry and the matrix comes back in that format at *shape*, which a sparse matrix needs; an entry outside it raises ``ValueError``, naming the file.
    """
    dtype = np.dtype(dtype or "float64")
    if fmt in ("csr", "coo"):
        cells = np.dtype([("row", np.int64), ("col", np.int64), ("value", dtype)])
        held = np.loadtxt(path, delimiter=" ", dtype=cells, ndmin=1) if os.path.getsize(path) else np.zeros(0, cells)
        outside = (held["row"] >= shape[0]) | (held["col"] >= shape[1])
        if outside.any():
            row, col = int(held["row"][outside][0]), int(held["col"][outside][0])
            raise ValueError(
                f"{path} holds an entry at row {row}, column {col}, outside the {shape[0]} x {shape[1]} matrix its sidecar gives."
            )
        matrix = coo_matrix((held["value"], (held["row"], held["col"])), shape=tuple(shape))
        return matrix.tocsr() if fmt == "csr" else matrix
    if dtype.kind in "OSU":
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f, delimiter=" "))
        values = np.array(rows if dtype.kind == "U" else [[v.encode("utf-8") for v in row] for row in rows], dtype=dtype)
    else:
        values = np.loadtxt(path, delimiter=" ", ndmin=2, dtype=dtype)
    return values.reshape(shape) if shape is not None else values


def _lines_and_fields(path) -> tuple[int, int]:
    """How many lines the CSV at *path* holds and how many fields its first line has, read without parsing a value."""
    with open(path, "rb") as f:
        first = f.readline()
        return (1 if first else 0) + sum(1 for _ in f), len(first.split())


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

    A read returns the array in the format it is stored in — a csr companion yields a csr matrix. Every read routes through :func:`_read_values`, the seam an I/O-counting test measures. A CSV holds one array and no header, so a CSV companion is one file per array, each named on the sidecar's record of what it holds (:func:`csv_links`), which also gives what the file does not record. What it holds is read from those links and no file, so ``"length" in store`` is answered from the sidecar, and a listed file that is missing raises when it is read.

    Use as a context manager to hold the file open across many reads; outside one, each read opens and closes the file, which is correct and safe and wrong under iteration.

    Args:
        companion_path: Path to the companion binary file (.h5, .zarr, .csv).
        meta_dict: Raw sidecar dict (from ``yaml_loader.load_as_dict``) for edge declarations.
    """

    def __init__(self, companion_path: Path, meta_dict: dict):
        from tvbo.utils.source import current_source_dir

        self._path = Path(companion_path)
        self._meta = meta_dict
        self._ext = self._path.suffix.lower()
        root = Path(current_source_dir() or self._path.parent)  # what a CSV's link is relative to
        self._files = {path: {**link, "path": root / link["file"]} for path, link in csv_links(meta_dict).items()}
        if self._files and self._ext != ".csv":
            raise ValueError(
                f"The sidecar of {self._path.name} names CSVs for {sorted(self._files)}, and {self._path.name} holds every array under its path. A network keeps its arrays in one companion format: save it again with the format it is to have."
            )
        self._cache: dict[str, Any] = {}
        self._params_cache: dict[str, dict[str, Any]] = {}
        self._names: list[str] | None = None
        self._handle = None
        self._depth = 0

    def _template_edges(self) -> list:
        return template_edges(self._meta.get("edges", []) or [])

    def _open_handle(self):
        """A fresh read handle on the store — an HDF5 file or a Zarr group — or ``None`` for a CSV, which has none to hold. Any other file raises ``OSError``, as an HDF5 file that cannot be opened does, so no reader takes it for a store that lacks the array asked for."""
        if self._ext in (".h5", ".hdf5"):
            import h5py

            return h5py.File(self._path, "r")
        if self._ext == ".zarr" or self._path.is_dir():
            import zarr

            return zarr.open(str(self._path), mode="r")
        if self._ext == ".csv":
            return None
        raise OSError(
            f"{self._path} is not an array store: a companion is an HDF5 file (.h5, .hdf5), a Zarr group (.zarr) or a CSV matrix (.csv)."
        )

    @contextmanager
    def _open(self):
        """The open store, held for the duration if a ``with store:`` block already holds it."""
        if self._handle is not None:
            yield self._handle
            return
        handle = self._open_handle()
        if handle is None:
            raise KeyError(f"{self._path} is not an array store")
        try:
            yield handle
        finally:
            if hasattr(handle, "close"):
                handle.close()

    def __enter__(self):
        """Hold the file open for the block. Nesting is counted, so an inner block cannot close the handle the outer one is still reading through."""
        self._depth += 1
        if self._handle is None:
            self._handle = self._open_handle()
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

        From the sidecar's template edges when it declares any; otherwise from the file's ``edges/`` group listing, which is a header read. A CSV save declares every layer it writes, so a CSV companion with links and no declared edge holds none, and one no record names is the one-matrix CSV, which holds ``weight``.
        """
        if self._names is None:
            declared = [edge_name(e) for e in self._template_edges()]
            if declared:
                self._names = declared
            elif self._ext == ".csv":
                self._names = [] if self._files else ["weight"]
            else:
                with self._open() as store:
                    self._names = list(store["edges"]) if "edges" in store else []
        return list(self._names)

    def info(self, key: str) -> ArrayInfo:
        """Shape, dtype and format of one array, from the file's header alone.

        ``key`` is an edge name (``"weight"``) or any dataset path (``"nodes/coordinates"``, ``"mesh/vertices"``). Raises ``KeyError`` when the file holds neither. A CSV companion answers from its links and from its file's lines and first line's fields, which reads no value (:meth:`_csv_info`).
        """
        if self._ext == ".csv":
            path = next((path for path in (f"edges/{key}", key) if self.holds(path)), None)
            if path is None:
                raise KeyError(key)
            return self._csv_info(path)
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
        """Whether the companion holds the layer *key*, which a sidecar's declaration does not promise: ``edges/<key>`` in the file's header, or for a CSV companion in its links, read with no value."""
        return key in self._cache or self.holds(f"edges/{key}")

    def holds(self, path: str) -> bool:
        """Whether the companion holds an array at *path* (``edges/weight``, ``nodes/parent_index``), from the file's header, or for a CSV companion from its links, without reading a value.

        A CSV companion holds each array the sidecar links (:func:`csv_links`) and, when every node it lists has a position, ``nodes/coordinates``, which those positions are. A .csv companion no record names was written when a CSV held one matrix: that of the first template edge, or ``weight`` when none is declared.
        """
        if self._ext == ".csv":
            if self._files:
                return path in self._files or (path == "nodes/coordinates" and self._positions() is not None)
            return path == f"edges/{self.names[0]}"
        with self._open() as store:
            return path in store

    def __getitem__(self, key: str):
        """One edge matrix, in its stored format, read on first access and kept."""
        if key not in self._cache:
            if self._ext == ".csv":
                self._cache[key] = self._read_csv(f"edges/{key}")
                prefix = f"edges/{key}/edge_parameters/"
                self._params_cache[key] = {
                    path[len(prefix) :]: self._read_csv(path) for path in self._files if path.startswith(prefix)
                }
            else:
                with self._open() as store:
                    self._cache[key], self._params_cache[key] = read_edge(store, key)
        return self._cache[key]

    def edge_params_of(self, key: str) -> dict:
        """The edge-parameter matrices stored beside one edge matrix."""
        self[key]
        return self._params_cache[key]

    def _positions(self):
        """The node coordinates the sidecar's nodes give, an ``(n, 3)`` float32 array as every writer stores them, or ``None`` unless every node it lists has a position."""
        nodes = self._meta.get("nodes") or []
        places = [_slot(node, "position") for node in nodes]
        if not places or any(place is None for place in places):
            return None
        return np.array([[_slot(place, axis) for axis in "xyz"] for place in places], dtype="float32")

    def _csv_link(self, path: str) -> dict:
        """The link of the CSV holding the array at *path*; raises ``KeyError`` when the companion holds none (:meth:`holds`)."""
        if not self.holds(path):
            raise KeyError(path)
        if self._files:
            return self._files[path]
        dtype = next((e.get("dtype") for e in self._template_edges() if edge_name(e) == self.names[0]), None)
        return {"path": self._path, "dtype": dtype}

    def _sparse_shape(self, path: str) -> tuple:
        """The shape of the sparse layer, or of its edge parameter, at *path*: a row for each node the sidecar lists, and a column for each of its edge's ``dimension_labels`` or, without them, for each node."""
        rows = int(self._meta.get("number_of_nodes") or len(self._meta.get("nodes") or []))
        layer = path.split("/")[1]
        edge = next((e for e in self._template_edges() if edge_name(e) == layer), {})
        return rows, len(edge.get("dimension_labels") or []) or rows

    def _read_csv(self, path: str):
        """The array at *path* from the CSV holding it, at the dtype, format and shape its link gives; ``nodes/coordinates`` the sidecar's positions give is :meth:`_positions`. Raises ``KeyError`` when no CSV holds it, and ``OSError`` when the CSV a link names cannot be read."""
        if path == "nodes/coordinates" and path not in self._files and self.holds(path):
            return self._positions()
        link = self._csv_link(path)
        fmt = link.get("format")
        shape = self._sparse_shape(path) if fmt in ("csr", "coo") else parse_shape(link.get("shape"))
        return read_csv_array(link["path"], shape, link.get("dtype"), fmt)

    def _csv_info(self, path: str) -> ArrayInfo:
        """What :meth:`info` answers for an array of a CSV companion, from its link and from its file's lines and first line's fields, without parsing a value: a sparse layer has the shape :meth:`_sparse_shape` gives and a stored entry for each line, a declared array the ``shape`` its link states, and any other array a row for each line and a column for each field."""
        if path == "nodes/coordinates" and path not in self._files:
            coordinates = self._positions()
            return ArrayInfo(path, coordinates.shape, coordinates.dtype, "dataset", int(coordinates.nbytes))
        link = self._csv_link(path)
        dtype, fmt = np.dtype(link.get("dtype") or "float64"), link.get("format") or "dense"
        lines, fields = _lines_and_fields(link["path"])
        if fmt in ("csr", "coo"):
            return ArrayInfo(path, self._sparse_shape(path), dtype, fmt, lines * dtype.itemsize)
        shape = parse_shape(link.get("shape")) or (lines, fields)
        return ArrayInfo(
            path, tuple(shape), dtype, fmt if path.startswith("edges/") else "dataset", int(np.prod(shape)) * dtype.itemsize
        )

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
            held = [*self._files, *(["nodes/coordinates"] if self.holds("nodes/coordinates") else [])]
            return [path for path in dict.fromkeys(held) if path.rpartition("/")[0] == prefix]
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
        if self._ext == ".csv":
            return self._read_csv(key)
        with self._open() as store:
            if key in store:
                return np.asarray(_read_values(store[key]))
        raise KeyError(key)


# ── Node selection ────────────────────────────────────────────────────


def select_edge_nodes(matrix, index, n_nodes: int, what: str):
    """The rows and columns *index* of an edge matrix over *n_nodes* nodes, in the storage format it came in. A matrix of another shape raises, naming it as *what*, since nothing then says which of its axes run over the nodes."""
    from scipy import sparse

    if tuple(matrix.shape) != (n_nodes, n_nodes):
        raise ValueError(
            f"{what} is {' x '.join(str(side) for side in matrix.shape)} on a network of {n_nodes} nodes, so a node selection cannot restrict it."
        )
    if sparse.issparse(matrix):
        return matrix.tocsr()[index][:, index].asformat(matrix.format)
    return matrix[np.ix_(index, index)]


def select_node_entries(array, index, n_nodes: int, what: str):
    """The entries *index* along the first axis of a per-node array over *n_nodes* nodes. An array whose first axis has another length raises, naming it as *what*."""
    if not getattr(array, "shape", ()) or array.shape[0] != n_nodes:
        raise ValueError(
            f"{what} has shape {tuple(getattr(array, 'shape', ()))} on a network of {n_nodes} nodes, so a node selection cannot restrict it."
        )
    return array[index]


class NodeSelectedStore:
    """A companion store read through a node selection: every array comes back for the kept nodes only.

    Stands in for the store a network's source attached (:class:`LazyArrayStore`) once the network's ``node_selection`` is applied, and answers the same calls, so nothing that reads the companion sees a dropped node. An edge matrix and its edge-parameter matrices lose the dropped rows and columns (:func:`select_edge_nodes`), a ``nodes/<name>`` dataset or an array the sidecar declares on its ``node_template`` loses the dropped entries (:func:`select_node_entries`), an array it declares as a network parameter, such as a vocabulary, is served whole (:func:`declared_arrays`), and :meth:`info` reports the selected shape beside the bytes the stored array costs to read. Any other dataset raises, since nothing says whether it runs over the nodes. An edge matrix is read and cut once and then kept, as the store it stands in for keeps what it reads.

    Args:
        store: The store to read from; its own selection, if it is one of these, is applied first.
        index: Positions of the kept nodes among the store's nodes, ascending.
        n_nodes: How many nodes the store's arrays run over.
    """

    def __init__(self, store, index, n_nodes: int):
        self._inner = store
        self._index = np.asarray(index, dtype=int)
        self._n_nodes = int(n_nodes)
        self._cut: dict[str, Any] = {}
        self._cut_params: dict[str, dict[str, Any]] = {}

    @property
    def _path(self) -> Path:
        return self._inner._path

    @property
    def _meta(self) -> dict:
        return self._inner._meta

    @property
    def _loaded(self) -> bool:
        return self._inner._loaded

    @property
    def names(self) -> list[str]:
        """The edge matrices the companion carries, without reading a value."""
        return self._inner.names

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)

    def __contains__(self, key: str) -> bool:
        return key in self._inner

    def holds(self, path: str) -> bool:
        """Whether the companion holds an array at *path*, as the store it stands in for answers."""
        return self._inner.holds(path)

    def _scope(self, key: str) -> str | None:
        """``"node"`` or ``"network"`` for an array the sidecar declares (:func:`declared_arrays`), ``None`` for one it does not."""
        return declared_arrays(self._meta).get(key, (None,))[0]

    def __getitem__(self, key: str):
        """One edge matrix on the kept nodes, in its stored format."""
        if key not in self._cut:
            self._cut[key] = select_edge_nodes(self._inner[key], self._index, self._n_nodes, self._what(f"edges/{key}"))
        return self._cut[key]

    def edge_params_of(self, key: str) -> dict:
        """The edge-parameter matrices stored beside one edge matrix, on the kept nodes."""
        if key not in self._cut_params:
            self._cut_params[key] = {
                name: select_edge_nodes(matrix, self._index, self._n_nodes, self._what(f"edges/{key}/edge_parameters/{name}"))
                for name, matrix in self._inner.edge_params_of(key).items()
            }
        return self._cut_params[key]

    @property
    def arrays(self) -> Mapping:
        """Edge matrices on the kept nodes as a read-per-key mapping."""
        return _LazyView(lambda: self.names, self.__getitem__)

    @property
    def edge_params(self) -> Mapping:
        """Edge parameters on the kept nodes as a read-per-key mapping, keyed like :attr:`arrays`."""
        return _LazyView(lambda: self.names, self.edge_params_of)

    def info(self, key: str) -> ArrayInfo:
        """Shape, dtype and format of one array as this store serves it: the stored header with every node axis at the kept count."""
        stored = self._inner.info(key)
        node_axes = len(stored.shape) if stored.path.startswith("edges/") else int(self._scope(stored.path) != "network")
        shape = tuple(
            len(self._index) if axis < node_axes and side == self._n_nodes else side for axis, side in enumerate(stored.shape)
        )
        return ArrayInfo(stored.path, shape, stored.dtype, stored.format, stored.nbytes)

    def dataset_keys(self, prefix: str = "") -> list[str]:
        """Dataset paths under ``prefix``, as the companion lists them."""
        return self._inner.dataset_keys(prefix)

    def read_dataset(self, key: str):
        """A ``nodes/<name>`` dataset, or an array the sidecar declares on its ``node_template``, for the kept nodes, and an array it declares as a network parameter whole. A dataset the companion holds at any other path raises ``ValueError``, since nothing says whether it runs over the nodes; one it does not hold raises ``KeyError`` as on the store itself."""
        stored = self._inner.read_dataset(key)
        scope = self._scope(key)
        if scope == "network":
            return stored
        if not key.startswith("nodes/") and scope != "node":
            raise ValueError(
                f"{self._what(key)} is not a per-node dataset, so a node selection cannot say what is left of it."
            )
        return select_node_entries(stored, self._index, self._n_nodes, self._what(key))

    def _what(self, key: str) -> str:
        """How an error names one stored array: its path and the companion that holds it."""
        return f"`{key}` in {Path(self._path).name}"
