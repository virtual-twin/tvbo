"""``Mesh.mesh_file`` finally has a reader, and it must not invent precision.

The slot has always said a mesh may live in an external GIFTI/VTK/FreeSurfer file; nothing
in core read one, so every study that drew a cortical surface shipped its own parser. The
one behaviour worth pinning beyond "it parses" is the declared scalar type: VTK's ``float``
is single precision, and a reader that helpfully keeps the full decimal width returns
coordinates the file does not claim to carry — which is enough to move a rendered surface
by a sub-pixel and put two readers of the same file permanently out of agreement.
"""
from __future__ import annotations

import numpy as np
import pytest

from tvbo.data.mesh_io import detect_format, read_mesh

_VTK = """# vtk DataFile Version 2.0
test
ASCII
DATASET POLYDATA
POINTS 4 {scalar}
0.0 0.0 0.0
1.234567891 0.0 0.0
0.0 1.0 0.0
1.0 1.0 0.0
POLYGONS 2 8
3 0 1 2
3 1 3 2
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


@pytest.mark.parametrize("name,expected", [
    ("s.vtk", "vtk"), ("s.surf.gii", "gifti"), ("lh.pial", "freesurfer"),
    ("m.msh", "meshio"), ("m.obj", "meshio"),
])
def test_the_format_is_read_off_the_name(name, expected):
    assert detect_format(name) == expected


def test_a_triangulated_polydata_round_trips(tmp_path):
    v, f = read_mesh(_write(tmp_path, "s.vtk", _VTK.format(scalar="double")))
    assert v.shape == (4, 3) and f.shape == (2, 3)
    np.testing.assert_array_equal(f, [[0, 1, 2], [1, 3, 2]])


def test_a_float_block_is_read_at_single_precision(tmp_path):
    """VTK's `float` is float32; the value must be the one the file can hold."""
    v, _ = read_mesh(_write(tmp_path, "s.vtk", _VTK.format(scalar="float")))
    assert v[1, 0] == float(np.float32(1.234567891))
    assert v[1, 0] != 1.234567891


def test_a_double_block_keeps_every_digit(tmp_path):
    v, _ = read_mesh(_write(tmp_path, "s.vtk", _VTK.format(scalar="double")))
    assert v[1, 0] == 1.234567891


def test_the_result_is_always_widened_and_integer_indexed(tmp_path):
    """Whatever the file declares, downstream sees one dtype pair."""
    v, f = read_mesh(_write(tmp_path, "s.vtk", _VTK.format(scalar="float")))
    assert v.dtype == np.float64 and f.dtype == np.int64


def test_a_declared_format_overrides_the_name(tmp_path):
    path = _write(tmp_path, "surface.dat", _VTK.format(scalar="double"))
    assert detect_format(path) == "meshio"
    v, _ = read_mesh(path, mesh_format="vtk")
    assert v.shape == (4, 3)


def test_a_quad_mesh_is_refused_by_name(tmp_path):
    """Silently dropping the 4th vertex of every face would give a plausible wrong mesh."""
    quads = _VTK.format(scalar="double").replace("POLYGONS 2 8", "POLYGONS 1 5") \
                                        .replace("3 0 1 2\n3 1 3 2\n", "4 0 1 3 2\n")
    with pytest.raises(ValueError, match="only triangular POLYGONS"):
        read_mesh(_write(tmp_path, "q.vtk", quads))


def test_an_unknown_format_names_the_ones_it_knows(tmp_path):
    with pytest.raises(ValueError, match="unknown mesh_format"):
        read_mesh(_write(tmp_path, "s.vtk", _VTK.format(scalar="double")), mesh_format="obj?")


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_mesh(tmp_path / "nope.vtk")
