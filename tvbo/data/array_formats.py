"""The array file formats tvbo reads, one store class each, and `read_array`, the one function every matrix read in tvbo takes.

A store answers `keys(prefix)`, `info(key)` and `attrs(key)` from the file's header and reads values only in `read(key)`, every value through `matrix_io._read_values`, the seam `tests/test_array_source.py` counts, and a network's CSV companion's through `matrix_io.read_csv_array`. `with store:` holds the file open across reads. A single-array format (NumPy `.npy`, delimited text) has no keys of its own: it holds the one array the caller names, under whatever key the caller asks for. Edge matrices are receiver-row, dims `(target, source)`: `W[i, j]` is the weight of the edge `j -> i`, the convention TVB, tvboptim and every tvbo backend reduce with.
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import sparse

from tvbo.data import matrix_io
from tvbo.data.matrix_io import ArrayInfo
from tvbo.export.registry import ArrayFormat, array_format_for, register_array_format, resolve_array_format

CANONICAL_EDGE_DIMS = ("target", "source")
"""The dims of every edge matrix a Network holds: row = receiving node, column = sending node."""

TEXT_DELIMITERS = {".csv": ",", ".tsv": "\t", ".txt": None}
"""The delimiter a text suffix means; ``None`` is any run of whitespace."""

SYMMETRY_RTOL = 1e-9


class _Store:
    """What every store shares: the path, the open mode, and a handle held for the duration of a ``with`` block and opened per call outside one. ``single`` marks a format holding one array under whatever key is asked for."""

    single = False

    def __init__(self, path, *, mode: str = "r"):
        self.path = Path(path) if not hasattr(path, "read") else path
        self.mode = mode
        self._handle = None
        self._depth = 0

    def _open_handle(self):
        return None

    def _close_handle(self, handle) -> None:
        if hasattr(handle, "close"):
            handle.close()

    @contextmanager
    def open(self):
        """The open file, held for the block; the handle a surrounding ``with store:`` block holds when there is one."""
        with self:
            yield self._handle

    def __enter__(self):
        self._depth += 1
        if self._handle is None:
            self._handle = self._open_handle()
        return self

    def __exit__(self, *exc):
        self._depth = max(self._depth - 1, 0)
        if self._depth:
            return
        handle, self._handle = self._handle, None
        self._close_handle(handle)

    def keys(self, prefix: str = "") -> list[str]:
        """The array paths under ``prefix``, from the header; a single-array format lists none."""
        return []

    def attrs(self, key: str = "") -> dict:
        """The attributes stored on one node, the root's for ``""``; empty for a format that carries none."""
        return {}

    def info(self, key: str) -> ArrayInfo:
        """Shape, dtype and storage format of one array without reading its values; ``KeyError`` when the file holds no such array."""
        raise KeyError(f"{self.path} has no header to read; read {key!r} to learn its shape")

    def holds(self, key: str) -> bool:
        """Whether the file holds an array at ``key``, from its header."""
        try:
            self.info(key)
        except KeyError:
            return False
        return True

    def read(self, key: str):
        """One array in its stored form, an ndarray or a scipy sparse matrix; ``KeyError`` when the file holds no such array."""
        raise NotImplementedError


def _is_array(node) -> bool:
    return hasattr(node, "shape") and hasattr(node, "dtype")


def _is_matrix_group(node) -> bool:
    return hasattr(node, "attrs") and "format" in node.attrs


def _under(names, prefix: str) -> list[str]:
    """The names under ``prefix`` of a flat keyed file, all of them for ``""``."""
    return [n for n in names if not prefix or n.startswith(f"{prefix}/")]


class GroupStore(_Store):
    """An HDF5 file or a Zarr group: datasets at paths, edge matrices as ``format``-attributed groups (`matrix_io.read_matrix`)."""

    def _node(self, handle, key: str):
        if key not in handle:
            raise KeyError(f"{self.path} holds no {key!r}")
        return handle[key]

    def keys(self, prefix: str = "") -> list[str]:
        """The datasets and matrix groups directly under ``prefix``, from one listing of the group."""
        with self.open() as handle:
            if prefix and prefix not in handle:
                return []
            grp = handle[prefix] if prefix else handle
            if _is_array(grp):
                return []
            children = grp.items() if hasattr(grp, "items") else grp.members()
            names = [name for name, node in children if _is_array(node) or _is_matrix_group(node)]
        return [f"{prefix}/{n}" if prefix else n for n in names]

    def holds(self, key: str) -> bool:
        """Whether the file lists a node at ``key``, so a matrix group that lost its ``data`` is held and raises when read."""
        with self.open() as handle:
            return key in handle

    def attrs(self, key: str = "") -> dict:
        """The attributes of one node, the root's for ``""``; empty when the file holds no such node."""
        with self.open() as handle:
            if key and key not in handle:
                return {}
            return dict((handle[key] if key else handle).attrs)

    def info(self, key: str) -> ArrayInfo:
        """The header of a dataset or matrix group, no value read."""
        with self.open() as handle:
            node = self._node(handle, key)
            if _is_matrix_group(node):
                data = node["data"]
                shape = tuple(int(s) for s in node.attrs["shape"])
                return ArrayInfo(key, shape, np.dtype(data.dtype), str(node.attrs["format"]), int(data.nbytes))
            if _is_array(node):
                return ArrayInfo(key, tuple(node.shape), np.dtype(node.dtype), "dataset", int(node.nbytes))
        raise KeyError(f"{self.path}: {key!r} is a group, not an array")

    def read(self, key: str):
        """A dataset as an ndarray, a matrix group in its stored format."""
        with self.open() as handle:
            node = self._node(handle, key)
            if _is_matrix_group(node):
                return matrix_io.read_matrix(node)
            if _is_array(node):
                return np.asarray(matrix_io._read_values(node))
        raise KeyError(f"{self.path}: {key!r} is a group, not an array")


class H5Store(GroupStore):
    """An HDF5 file."""

    def _open_handle(self):
        import h5py

        return h5py.File(self.path, self.mode)


class ZarrStore(GroupStore):
    """A Zarr group; the optional ``zarr`` dependency is imported on first open."""

    def _open_handle(self):
        import zarr

        return zarr.open(str(self.path), mode=self.mode)


class MatH5Store(H5Store):
    """A MATLAB v7.3 file: HDF5 whose matrices are stored column-major, so a dataset comes back transposed to the shape MATLAB holds."""

    def info(self, key: str) -> ArrayInfo:
        """The header with its shape in MATLAB's order."""
        info = super().info(key)
        return dataclasses.replace(info, shape=info.shape[::-1])

    def read(self, key: str):
        """The dataset in MATLAB's orientation."""
        return np.transpose(super().read(key))


class MatStore(_Store):
    """A MATLAB v5 file through `scipy.io`; one open file serves the header listing and the reads."""

    def _open_handle(self):
        return open(self.path, "rb")

    def _variables(self, handle) -> list:
        from scipy.io import whosmat

        handle.seek(0)
        return whosmat(handle)

    def keys(self, prefix: str = "") -> list[str]:
        """The variables the file's header lists."""
        with self.open() as handle:
            return _under([name for name, _shape, _kind in self._variables(handle)], prefix)

    def info(self, key: str) -> ArrayInfo:
        """One variable's shape and class from the file's header."""
        with self.open() as handle:
            for name, shape, kind in self._variables(handle):
                if name == key:
                    dtype = np.dtype(_MAT_DTYPES.get(kind, "float64"))
                    nbytes = int(np.prod(shape)) * dtype.itemsize
                    return ArrayInfo(key, tuple(shape), dtype, "sparse" if kind == "sparse" else "dataset", nbytes)
        raise KeyError(f"{self.path} holds no {key!r}")

    def read(self, key: str):
        """One variable, loaded alone."""
        from scipy.io import loadmat

        with self.open() as handle:
            handle.seek(0)
            values = loadmat(handle, variable_names=[key], squeeze_me=False)
        if key not in values:
            raise KeyError(f"{self.path} holds no {key!r}")
        return matrix_io._read_values(values[key])


_MAT_DTYPES = {
    "double": "float64",
    "single": "float32",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "logical": "bool",
}


def mat_store(path, **options) -> _Store:
    """The store for a ``.mat`` file: v7.3 is HDF5 and reads through h5py, v5 through scipy."""
    with open(path, "rb") as f:
        header = f.read(128)
    return MatH5Store(path, **options) if b"MATLAB 7.3" in header else MatStore(path, **options)


class NpyStore(_Store):
    """One NumPy array, memory-mapped; it holds the one key the caller asks for."""

    single = True

    def _open_handle(self):
        return np.load(self.path, mmap_mode="r")

    def info(self, key: str) -> ArrayInfo:
        """The array's header, from the memory map."""
        with self.open() as mm:
            return ArrayInfo(key, tuple(mm.shape), np.dtype(mm.dtype), "dataset", int(mm.nbytes))

    def read(self, key: str):
        """The memory-mapped array."""
        with self.open() as mm:
            return matrix_io._read_values(mm)


class NpzStore(_Store):
    """A NumPy archive; each member is a key, its header read without its values."""

    def _open_handle(self):
        return np.load(self.path)

    def keys(self, prefix: str = "") -> list[str]:
        """The archive's members."""
        with self.open() as npz:
            return _under(npz.files, prefix)

    def info(self, key: str) -> ArrayInfo:
        """One member's header, read from the archive without its values."""
        with self.open() as npz:
            if key not in npz.files:
                raise KeyError(f"{self.path} holds no {key!r}")
            with npz.zip.open(f"{key}.npy") as member:
                version = np.lib.format.read_magic(member)
                read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _fortran, dtype = read_header(member)
        return ArrayInfo(key, tuple(shape), np.dtype(dtype), "dataset", int(np.prod(shape)) * np.dtype(dtype).itemsize)

    def read(self, key: str):
        """One member."""
        with self.open() as npz:
            if key not in npz.files:
                raise KeyError(f"{self.path} holds no {key!r}")
            return matrix_io._read_values(npz[key])


class TextStore(_Store):
    """One delimited text matrix, the one `np.loadtxt` a matrix is read with; the delimiter follows the suffix (`TEXT_DELIMITERS`) unless one is declared.

    The source may be an open binary file instead of a path (a member of a zip archive); the delimiter is then whitespace unless declared, since a file object has no suffix.
    """

    single = True

    def __init__(self, path, *, mode: str = "r", delimiter: str | None = None):
        super().__init__(path, mode=mode)
        if delimiter is None and isinstance(self.path, Path):
            delimiter = TEXT_DELIMITERS.get(self.path.suffix.lower())
        self.delimiter = delimiter

    def read(self, key: str):
        """The file's one matrix."""
        return matrix_io._read_values(np.loadtxt(self.path, delimiter=self.delimiter))


class CsvCompanionStore(_Store):
    """A network's CSV companion: one space-separated CSV per array (`matrix_io.write_csv_array`), each named on the sidecar's record of what it holds (`matrix_io.csv_links`), which also gives the dtype, format and shape the file does not record. What it holds comes from those links, and a header answer from them and, where they state no shape, the file's line and field counts, no value parsed.

    A CSV companion no record names was written when a CSV held one matrix: that of the sidecar's first template edge, or ``weight`` when none is declared. A linked companion also holds ``nodes/coordinates`` when every node the sidecar lists has a position, which those positions are.

    Args:
        path: The CSV the sidecar's ``data_file`` names.
        files: ``{array path: link}``, each link's ``path`` the CSV it names, resolved.
        meta: The raw sidecar dict.
    """

    def __init__(self, path, files: dict, meta: dict):
        super().__init__(path)
        self.files = files
        self.meta = meta
        self.edges = {matrix_io.edge_name(e): e for e in matrix_io.template_edges(meta.get("edges"))}

    @cached_property
    def links(self) -> dict:
        """``{array path: link}``: the sidecar's links, or the link of the one matrix a CSV companion no record names holds."""
        if self.files:
            return self.files
        name, edge = next(iter(self.edges.items()), ("weight", {}))
        return {f"edges/{name}": {"path": self.path, "dtype": edge.get("dtype")}}

    @cached_property
    def positions(self):
        """The ``(n, 3)`` float32 positions of the sidecar's nodes, as every writer stores them, which a linked companion with no CSV of its own for ``nodes/coordinates`` holds there; ``None`` unless every node it lists has one."""
        if not self.files or "nodes/coordinates" in self.files:
            return None
        places = [matrix_io._slot(node, "position") for node in self.meta.get("nodes") or []]
        if not places or any(place is None for place in places):
            return None
        return np.array([[matrix_io._slot(place, axis) for axis in "xyz"] for place in places], dtype="float32")

    def _coordinates(self, key: str):
        return self.positions if key == "nodes/coordinates" else None

    def _link(self, key: str) -> dict:
        if key not in self.links:
            raise KeyError(f"{self.path} holds no {key!r}")
        return self.links[key]

    def _sparse_shape(self, key: str) -> tuple:
        """The shape of the sparse layer, or of its edge parameter, at ``key``: a row for each node the sidecar lists, and a column for each of its edge's ``dimension_labels`` or, without them, for each node."""
        rows = int(self.meta.get("number_of_nodes") or len(self.meta.get("nodes") or []))
        edge = self.edges.get(key.split("/")[1], {})
        return rows, len(edge.get("dimension_labels") or []) or rows

    def holds(self, key: str) -> bool:
        """Whether a link names a CSV for ``key``, or the nodes' positions give it; no file is read."""
        return key in self.links or self._coordinates(key) is not None

    def keys(self, prefix: str = "") -> list[str]:
        """The arrays this companion holds directly under ``prefix``."""
        held = [*self.links, *(["nodes/coordinates"] if self.positions is not None else [])]
        return [key for key in held if key.rpartition("/")[0] == prefix]

    def info(self, key: str) -> ArrayInfo:
        """A sparse layer has the shape `_sparse_shape` gives and a stored entry for each line, a declared array the ``shape`` its link states, and any other array a row for each line and a column for each field."""
        coordinates = self._coordinates(key)
        if coordinates is not None:
            return ArrayInfo(key, coordinates.shape, coordinates.dtype, "dataset", int(coordinates.nbytes))
        link = self._link(key)
        dtype, fmt = np.dtype(link.get("dtype") or "float64"), link.get("format") or "dense"
        if fmt in ("csr", "coo"):
            lines, _ = matrix_io._lines_and_fields(link["path"])
            return ArrayInfo(key, self._sparse_shape(key), dtype, fmt, lines * dtype.itemsize)
        shape = matrix_io.parse_shape(link.get("shape")) or matrix_io._lines_and_fields(link["path"])
        return ArrayInfo(
            key, tuple(shape), dtype, fmt if key.startswith("edges/") else "dataset", int(np.prod(shape)) * dtype.itemsize
        )

    def read(self, key: str):
        """The array from the CSV holding it, at the dtype, format and shape its link gives; ``OSError`` when that CSV cannot be read."""
        coordinates = self._coordinates(key)
        if coordinates is not None:
            return coordinates
        link = self._link(key)
        fmt = link.get("format")
        shape = self._sparse_shape(key) if fmt in ("csr", "coo") else matrix_io.parse_shape(link.get("shape"))
        return matrix_io.read_csv_array(link["path"], shape, link.get("dtype"), fmt)


for _fmt in (
    ArrayFormat("hdf5", (".h5", ".hdf5"), H5Store, writes=True),
    ArrayFormat("zarr", (".zarr",), ZarrStore, writes=True),
    ArrayFormat("mat", (".mat",), mat_store),
    ArrayFormat("npy", (".npy",), NpyStore),
    ArrayFormat("npz", (".npz",), NpzStore),
    ArrayFormat("text", tuple(TEXT_DELIMITERS), TextStore),
):
    register_array_format(_fmt)


def open_store(source, *, format: str | None = None, **options) -> _Store:
    """The store that reads ``source``: by the registered format ``format`` names, else by the suffix. ``options`` are the store's constructor options (``mode``, a text store's ``delimiter``); one the format does not take is refused by name."""
    fmt = resolve_array_format(format) if format else array_format_for(source)
    try:
        return fmt.store(source, **options)
    except TypeError:
        raise ValueError(f"{source}: a {fmt.key!r} store takes none of the options {sorted(options)}.") from None


def is_symmetric(matrix, rtol: float = SYMMETRY_RTOL) -> bool:
    """Whether a square matrix equals its transpose to ``rtol`` of its largest magnitude, a sparse matrix against its own transpose without densifying."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    if sparse.issparse(matrix):
        m = matrix.tocsr()
        if not m.nnz:
            return True
        return abs(m - m.T).max() <= rtol * abs(m).max()
    scale = np.abs(matrix).max() if matrix.size else 0.0
    return bool(np.allclose(matrix, matrix.T, rtol=0.0, atol=rtol * scale))


def declare(data, dims, what: str) -> xr.DataArray:
    """``data`` as a DataArray under the declared ``dims``, an edge matrix transposed by name onto `CANONICAL_EDGE_DIMS`.

    Dims never come from the shape: with none declared, a square 2-D array is accepted when symmetric, where orientation cannot matter, and anything else is refused by name (``what`` says which read). An edge matrix, declared either way round or undeclared, must be square and finite.
    """
    if sparse.issparse(data):
        raise TypeError(f"{what}: a sparse matrix is read through its store, not as a DataArray.")
    data = np.asarray(data)
    if dims is None or set(dims) == set(CANONICAL_EDGE_DIMS):
        if data.ndim != 2 or data.shape[0] != data.shape[1]:
            raise ValueError(f"{what}: is not a square matrix: shape {data.shape}.")
        if not np.isfinite(data).all():
            raise ValueError(f"{what}: has non-finite entries.")
    if dims is None:
        if not is_symmetric(data):
            raise ValueError(
                f"{what}: no dims are declared for a matrix of shape {data.shape} that is not symmetric; declare them (`dims: [target, source]` for a receiver-row edge matrix, `[source, target]` for a sender-row one)."
            )
        dims = CANONICAL_EDGE_DIMS
    dims = tuple(dims)
    if len(dims) != data.ndim:
        raise ValueError(f"{what}: {len(dims)} dims {dims} are declared for an array of shape {data.shape}.")
    da = xr.DataArray(data, dims=dims)
    return da.transpose(*CANONICAL_EDGE_DIMS) if set(dims) == set(CANONICAL_EDGE_DIMS) else da


def sender_row(matrix):
    """A receiver-row edge matrix as the sender-row matrix a Graphs.jl ``SimpleWeightedDiGraph`` reads: ``M[i, j]`` is the weight of the edge ``i -> j``. The one place tvbo transposes a connectivity, by name."""
    return xr.DataArray(np.asarray(matrix), dims=CANONICAL_EDGE_DIMS).transpose("source", "target").values


def read_array(
    source, key: str | None = None, *, dims=None, format: str | None = None, delimiter: str | None = None
) -> xr.DataArray:
    """The array ``key`` of ``source`` under declared ``dims`` (`declare`), read through the store its format registers.

    ``source`` is a path, or an open binary file with ``format`` naming its format. ``key`` is the dataset path inside a keyed format and is ignored by a single-array one.
    """
    what = f"{getattr(source, 'name', source)}" + (f":{key}" if key else "")
    store = open_store(source, format=format, **({"delimiter": delimiter} if delimiter is not None else {}))
    if key is None and not store.single:
        raise ValueError(f"{what}: a {type(store).__name__} holds its arrays by key; name the one to read.")
    with store:
        data = store.read(key or "")
    return declare(data, dims, what)
