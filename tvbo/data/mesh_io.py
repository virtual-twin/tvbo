"""Read a surface mesh file into vertices and faces.

``Mesh.mesh_file`` has always declared that a mesh may live in an external GIFTI / VTK /
FreeSurfer / MSH file, and ``Mesh.mesh_format`` that the format may be stated rather than guessed — but nothing in core read one, so every study that draws a cortical surface
shipped its own reader in ``code/``. That is a format-support gap, not a modelling decision: which reader parses a ``.surf.gii`` is exactly the kind of mechanism a
declarative spec should not have to name.

The formats here are the ones ``mesh_format`` already lists. Each returns the SAME pair —
``(vertices (V, 3) float64, faces (F, 3) int64)`` — so a mesh is interchangeable across them and nothing downstream can tell which reader produced it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_GIFTI_SUFFIXES = (".gii",)
_FREESURFER_STEMS = (".pial", ".white", ".inflated", ".sphere", ".midthickness", ".orig")

_VTK_SCALARS = {"float": np.float32, "double": np.float64, "int": np.int32, "long": np.int64, "short": np.int16}


def detect_format(path: str | Path) -> str:
    """The reader a mesh file's name implies — what ``mesh_format`` overrides.

    FreeSurfer geometry carries no extension of its own (``lh.pial``), so it is recognised by its surface-name suffix; everything unrecognised falls through to meshio, which
    reads the long tail of mesh formats and raises its own error on a genuine miss.
    """
    path = Path(path)
    suffixes = [s.lower() for s in path.suffixes]
    if any(s in _GIFTI_SUFFIXES for s in suffixes):
        return "gifti"
    if ".vtk" in suffixes:
        return "vtk"
    if path.suffix.lower() in _FREESURFER_STEMS:
        return "freesurfer"
    return "meshio"


def _read_gifti(path: Path):
    import nibabel as nib

    gii = nib.load(str(path))
    return (gii.agg_data("NIFTI_INTENT_POINTSET"), gii.agg_data("NIFTI_INTENT_TRIANGLE"))


def _read_freesurfer(path: Path):
    from nibabel.freesurfer.io import read_geometry

    return read_geometry(str(path))


def _read_vtk(path: Path):
    """ASCII VTK PolyData, parsed directly rather than through a mesh library.

    This is the format brain-surface templates ship in, and its PolyData form is a flat
    POINTS block followed by a POLYGONS block whose every entry is prefixed by its vertex count. Parsing it here keeps a common case free of a heavyweight dependency; anything
    else in a ``.vtk`` wrapper (binary, unstructured grid) is handed to meshio, which reads those properly.

    The POINTS block names its scalar type and that type is honoured: VTK's ``float`` is single precision, so reading its decimals at full width would give coordinates the
    file does not claim to carry and leave this reader a rounding step away from every other one. The array is widened to float64 afterwards, which is lossless.

    A pure-triangle POLYGONS block is exactly 4·n_faces ints (``[3, v0, v1, v2]`` per face), so the total is checked before the per-face reshape — reshaping a mixed block first
    fails with an opaque numpy error instead of the actionable one raised here.
    """
    text = path.read_text(errors="ignore")
    if "POLYGONS" not in text or "ASCII" not in text.split("DATASET")[0].upper():
        return _read_meshio(path)

    tokens = text.split()
    start = tokens.index("POINTS")
    n_points = int(tokens[start + 1])
    dtype = _VTK_SCALARS.get(tokens[start + 2].lower(), np.float64)
    flat = np.asarray(tokens[start + 3 : start + 3 + 3 * n_points], dtype=dtype)
    vertices = flat.reshape(n_points, 3)

    start = tokens.index("POLYGONS")
    n_faces, n_entries = int(tokens[start + 1]), int(tokens[start + 2])
    entries = np.asarray(tokens[start + 3 : start + 3 + n_entries], dtype=np.int64)
    if n_entries != 4 * n_faces:
        raise ValueError(
            f"{path.name}: only triangular POLYGONS are read here (its POLYGONS block has "
            f"{n_entries} entries for {n_faces} faces, not the 4·n_faces of a pure-triangle mesh "
            "— the faces are non-triangular or mixed). Triangulate the mesh, or supply it in a "
            "format meshio reads."
        )
    faces = entries.reshape(n_faces, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError(
            f"{path.name}: only triangular POLYGONS are read here (found face sizes "
            f"{sorted(set(faces[:, 0].tolist()))}). Triangulate the mesh, or supply it in a "
            "format meshio reads."
        )
    return vertices, faces[:, 1:]


def _read_meshio(path: Path):
    import meshio

    mesh = meshio.read(str(path))
    for block in mesh.cells:
        if block.type == "triangle":
            return mesh.points, block.data
    raise ValueError(
        f"{path.name}: no triangle cells found (blocks: {sorted({b.type for b in mesh.cells})}); a surface mesh is triangular."
    )


_READERS = {
    "gifti": _read_gifti,
    "freesurfer": _read_freesurfer,
    "vtk": _read_vtk,
    "meshio": _read_meshio,
    "gmsh": _read_meshio,
}


def read_mesh(path: str | Path, mesh_format: str | None = None):
    """``(vertices, faces)`` of a surface mesh file (public API).

    Args:
        path: The mesh file.
        mesh_format: One of ``Mesh.mesh_format``'s values, overriding what the file name
            implies. Needed only where the name is ambiguous or wrong.

    Returns:
        ``vertices`` (V, 3) float64 and ``faces`` (F, 3) int64.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"mesh file not found: {path}")
    fmt = str(mesh_format or detect_format(path)).lower()
    try:
        reader = _READERS[fmt]
    except KeyError:
        raise ValueError(f"unknown mesh_format {fmt!r}; expected one of {sorted(_READERS)}.") from None
    vertices, faces = reader(path)
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)
