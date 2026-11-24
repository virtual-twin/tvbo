"""One writer for the array files the I/O tests read back.

`write_arrays(where, kind, **arrays)` writes every named array in the format ``kind`` names and returns the path and the key the first array is read by. In a companion layout (h5, zarr) the first array is the edge matrix under ``edges/<name>`` and the rest are root datasets; a single-array format (npy, csv, tsv, txt) takes the first array alone and has no key.
"""

import numpy as np
import pytest
import scipy.io

from tvbo.data import matrix_io

TEXT = {"csv": ",", "tsv": "\t", "txt": " "}


def _write_mat73(path, arrays: dict) -> None:
    """A MATLAB v7.3 file: HDF5 behind a 512-byte MATLAB header, arrays stored column-major."""
    import h5py

    with h5py.File(path, "w", userblock_size=512) as f:
        for name, value in arrays.items():
            f.create_dataset(name, data=np.asarray(value).T)
    with open(path, "r+b") as f:
        f.write(b"MATLAB 7.3 MAT-file".ljust(116) + b"\x00" * 8 + b"\x00\x02IM")


def write_arrays(where, kind: str, **arrays):
    """The file ``kind`` names under ``where`` holding ``arrays``, and the key its first array is read by (``None`` for a single-array format)."""
    where.mkdir(parents=True, exist_ok=True)
    (first, matrix), *rest = arrays.items()
    if kind in ("h5", "zarr"):
        if kind == "h5":
            import h5py

            path = where / "a.h5"
            root = h5py.File(path, "w")
        else:
            zarr = pytest.importorskip("zarr")
            path = where / "a.zarr"
            root = zarr.open(str(path), mode="w")
        with root if kind == "h5" else _nullcontext(root):
            matrix_io.write_matrix(root.create_group(f"edges/{first}"), matrix)
            for name, value in rest:
                (root.create_dataset if kind == "h5" else root.create_array)(name, data=value)
        return path, f"edges/{first}"
    if kind == "npy":
        np.save(where / "a.npy", matrix)
        return where / "a.npy", None
    if kind == "npz":
        np.savez(where / "a.npz", **arrays)
        return where / "a.npz", first
    if kind == "mat":
        scipy.io.savemat(where / "a.mat", dict(arrays))
        return where / "a.mat", first
    if kind == "mat73":
        _write_mat73(where / "a.mat", dict(arrays))
        return where / "a.mat", first
    path = where / f"a.{kind}"
    np.savetxt(path, matrix, delimiter=TEXT[kind])
    return path, None


class _nullcontext:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *exc):
        return None
