"""Low-level matrix read/write for HDF5 groups and Zarr groups, and the lazy companion store a Network reads through.

Supports dense, CSR, and COO formats. Both HDF5 (h5py.Group) and Zarr (zarr.Group) implement the same array-store interface, so a single pair of read/write functions handles both backends. A CSV companion holds one array per file, which the sidecar names on the record of what it holds (:func:`csv_links`, :func:`write_csv_array`, :func:`read_csv_array`). The file formats themselves, and the one `read_array` for a matrix outside a companion, are in `tvbo.data.array_formats`.

See §12.1 of the tvbo HDF5 format proposal v0.7.
"""

import ast
import csv
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
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
    """Create a dataset in an HDF5 or a Zarr v3 group, with h5py's keywords: ``compression="gzip"`` becomes Zarr's :class:`~zarr.codecs.GzipCodec`, and any other compression raises ``KeyError`` naming it."""
    try:
        import zarr

        if isinstance(grp, zarr.Group):
            if "compression" in kwargs:
                kwargs["compressors"] = {"gzip": zarr.codecs.GzipCodec()}[kwargs.pop("compression")]
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
    """Every value read in this module, in one place, a variable-length HDF5 string read as text (``<U``), as a save writes a text array, where h5py reads it as bytes.

    The array layer's contract is that metadata questions read no values, and a contract nobody can measure is a wish. Routing each read through here gives a test one seam to count.
    """
    import h5py

    info = h5py.check_string_dtype(dataset.dtype) if isinstance(dataset, h5py.Dataset) else None
    if info is not None and info.length is None:
        return np.asarray(dataset.asstr()[()], dtype=str)
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


def csv_link_slots(network) -> list:
    """``(path, record, slot, extra)`` for each CSV *network*, a sidecar mapping or a ``Network``, names on one of its records: a layer's on its template edge's ``data_file``, an edge parameter's on its ``source``, a declared array's (:func:`declared_arrays`) on its ``source``, and a mesh array's on the ``mesh`` field :data:`MESH_FIELDS` names. *path* is the array's path in an HDF5 companion, and *extra* what the file does not record: the ``dtype``, for a layer the sparse ``format``, and for a declared array the ``shape``."""
    slots = []
    for edge in template_edges(_slot(network, "edges")):
        layer, fmt, dtype = edge_name(edge), _slot(edge, "format"), _slot(edge, "dtype")
        slots.append((f"edges/{layer}", edge, "data_file", {"dtype": dtype, "format": fmt}))
        slots.extend(
            (
                f"edges/{layer}/edge_parameters/{name}",
                param,
                "source",
                {"dtype": _slot(param, "dtype") or dtype, "format": fmt},
            )
            for name, param in named_entries(_slot(edge, "parameters"))
        )
    slots.extend(
        (path, param, "source", {"dtype": _slot(param, "dtype"), "shape": _slot(param, "shape")})
        for path, (_, param) in declared_arrays(network).items()
    )
    mesh = _slot(network, "mesh")
    if mesh is not None:
        slots.extend((path, mesh, field, {"dtype": dtype}) for path, (field, dtype) in MESH_FIELDS.items())
    return [entry for entry in slots if is_csv(_slot(entry[1], entry[2]))]


def csv_links(meta: Mapping) -> dict:
    """``{path: link}`` for each array the sidecar *meta* names a CSV for (:func:`csv_link_slots`), keyed by the path the array has in an HDF5 companion. A link holds the CSV's ``file``, relative to the sidecar, and what the file does not record. A layer whose ``data_file`` names any other file is refused."""
    for edge in template_edges(meta.get("edges")):
        if edge.get("data_file") and not is_csv(edge["data_file"]):
            raise ValueError(
                f"The layer {edge_name(edge)!r} names {edge['data_file']} as its `data_file`, which names the CSV holding that layer alone. An HDF5 or Zarr companion holds every layer under its path and is the network's own `data_file`."
            )
    return {path: {"file": _slot(record, slot), **extra} for path, record, slot, extra in csv_link_slots(meta)}


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

    Dense, it comes back at *shape*, or 2-D as the rows give it when that is ``None``, a file that holds no value (an empty one, or the blank lines of an array with no column) as an empty array of as many rows as it has lines; text at a bytes or ``object`` dtype comes back as bytes, as h5py reads it. With *fmt* csr or coo the file holds one line per stored entry and the matrix comes back in that format at *shape*, which a sparse matrix needs; an entry outside it raises ``ValueError``, naming the file.
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
    elif _first_line(path).split():
        values = np.loadtxt(path, delimiter=" ", ndmin=2, dtype=dtype)
    else:
        values = np.zeros((_lines_and_fields(path)[0], 0), dtype)
    return values.reshape(shape) if shape is not None else values


def _first_line(path) -> bytes:
    """The first line of the file at *path*, empty for an empty file."""
    with open(path, "rb") as f:
        return f.readline()


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
    """A companion array file, read one array at a time through the store its format registers (`tvbo.data.array_formats`).

    Holds the path and the sidecar's edge declarations and nothing else at construction. Names, shapes, dtypes and formats come from the sidecar and the file's header; a value is read the first time it is asked for and kept.

    A read returns the array in the format it is stored in — a csr companion yields a csr matrix. Every read routes through :func:`_read_values`, the seam an I/O-counting test measures, a CSV companion's through :func:`read_csv_array`. A CSV holds one array and no header, so a CSV companion is one file per array, each named on the sidecar's record of what it holds (:func:`csv_links`), which also gives what the file does not record. What it holds is read from those links and no file, so ``"length" in store`` is answered from the sidecar, and a listed file that is missing raises when it is read.

    Use as a context manager to hold the file open across many reads; outside one, each read opens and closes the file, which is correct and safe and wrong under iteration.

    Args:
        companion_path: Path to the companion file, in any registered array format.
        meta_dict: Raw sidecar dict (from ``yaml_loader.load_as_dict``) for edge declarations.
    """

    def __init__(self, companion_path: Path, meta_dict: dict):
        from tvbo.utils.source import current_source_dir

        self._path = Path(companion_path)
        self._meta = meta_dict
        root = Path(current_source_dir() or self._path.parent)  # what a CSV's link is relative to
        self._files = {path: {**link, "path": root / link["file"]} for path, link in csv_links(meta_dict).items()}
        if self._files and not is_csv(str(self._path)):
            raise ValueError(
                f"The sidecar of {self._path.name} names CSVs for {sorted(self._files)}, and {self._path.name} holds every array under its path. A network keeps its arrays in one companion format: save it again with the format it is to have."
            )
        self._cache: dict[str, Any] = {}
        self._params_cache: dict[str, dict[str, Any]] = {}
        self._paths: dict[str, str] = {}
        self._names: list[str] | None = None

    @cached_property
    def _reader(self):
        """The format store on the companion, a CSV companion's its links (`tvbo.data.array_formats.CsvCompanionStore`), opened on first use. A path in no array format raises ``OSError``, as a file that cannot be opened does, so no reader takes it for a companion that lacks the array asked for."""
        from tvbo.data.array_formats import CsvCompanionStore, open_store

        if is_csv(str(self._path)):
            return CsvCompanionStore(self._path, self._files, self._meta)
        try:
            return open_store(self._path)
        except ValueError as e:
            raise OSError(f"{self._path} is not an array store: {e}") from None

    def __enter__(self):
        """Hold the file open for the block. Nesting is counted, so an inner block cannot close the handle the outer one is still reading through."""
        self._reader.__enter__()
        return self

    def __exit__(self, *exc):
        self._reader.__exit__(*exc)

    @property
    def names(self) -> list[str]:
        """The edge matrices this companion carries, without reading a value.

        From the sidecar's template edges when it declares any; otherwise from the file's ``edges/`` listing, which is a header read, a CSV companion's from its links. A single-array companion carries exactly one, ``weight``; a keyed file with no ``edges/`` group carries none.
        """
        if self._names is None:
            names = [edge_name(e) for e in template_edges(self._meta.get("edges"))]
            if not names:
                names = [key.removeprefix("edges/") for key in self._reader.keys("edges")]
            self._names = names or (["weight"] if self._reader.single else [])
        return list(self._names)

    def _path_of(self, key: str) -> str:
        """The stored path of ``key``: ``edges/<key>`` when the companion keys its edges that way, else ``key`` itself; probed once under one handle and kept. ``KeyError`` when the file holds neither."""
        if key not in self._paths:
            with self._reader:
                path = next((path for path in (f"edges/{key}", key) if self.holds(path)), None)
            if path is None:
                raise KeyError(f"{self._path} holds no {key!r}")
            self._paths[key] = path
        return self._paths[key]

    def info(self, key: str) -> ArrayInfo:
        """Shape, dtype and format of one array, from the file's header alone.

        ``key`` is an edge name (``"weight"``) or any dataset path (``"nodes/coordinates"``, ``"mesh/vertices"``). Raises ``KeyError`` when the file holds neither. A CSV companion answers from its links and from its file's lines and first line's fields, which reads no value.
        """
        with self._reader:
            return self._reader.info(self._path_of(key))

    def __contains__(self, key: str) -> bool:
        """Whether the companion holds the layer *key*, which a sidecar's declaration does not promise: ``edges/<key>`` or ``<key>`` in the file's header, or for a CSV companion in its links, read with no value."""
        if key in self._cache:
            return True
        try:
            self._path_of(key)
        except KeyError:
            return False
        return True

    def attributes(self, path: str) -> dict:
        """The attributes of the group or dataset at *path* (``mesh``), from the file's header: empty for a CSV companion, which records none, and when the file holds nothing at *path*."""
        return self._reader.attrs(path)

    def holds(self, path: str) -> bool:
        """Whether the companion holds an array at *path* (``edges/weight``, ``nodes/parent_index``), from the file's header, or for a CSV companion from its links, without reading a value; a single-array companion holds its one edge matrix."""
        if self._reader.single:
            return path == f"edges/{self.names[0]}"
        return self._reader.holds(path)

    def __getitem__(self, key: str):
        """One edge matrix, in its stored format, read on first access and kept."""
        if key not in self._cache:
            with self._reader:
                path = self._path_of(key)
                self._cache[key] = self._reader.read(path)
                self._params_cache[key] = {
                    param.rsplit("/", 1)[1]: self._reader.read(param) for param in self._reader.keys(f"{path}/edge_parameters")
                }
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
        return self._reader.keys(prefix)

    def read_dataset(self, key: str):
        """Read an arbitrary dataset by path (e.g. ``"nodes/parent_index"``)."""
        return self._reader.read(key)


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
