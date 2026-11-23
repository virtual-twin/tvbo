"""The curated ``tvbo:functions/MeshOperators``: the fixed operators of a triangulated mesh a travelling-wave detector reads, on meshes small enough to work by hand and on geometry whose answer is known.

Each point Koller's ``modules/wave_detection_methods.py`` settles is pinned on such a mesh: the k-ring walk with its face multiplicity, the border dilation, the gradient basis read at the face corners in x, y and z blocks, and the potential against his regularised solve. His code is never imported here; where a test follows one of his functions, its docstring works the expected value out.
"""

import numpy as np
import pytest
import xarray as xr

from tvbo.analysis.mesh import mesh_operators
from tvbo.data import param_io
from tvbo.datamodel import FunctionCall, Parameter

EXTRA = "requires the 'mesh' extra (pip install tvbo[mesh])"
pytest.importorskip("igl", reason=EXTRA)
pytest.importorskip("potpourri3d", reason=EXTRA)


def _labels(n: int) -> list[str]:
    return [f"n{i}" for i in range(n)]


def _operators(vertices, faces=None, groups="mesh", **rings) -> dict:
    """The operators of a mesh whose vertices are the network's nodes in order, one group unless *groups* says otherwise."""
    return mesh_operators(vertices, _labels(len(vertices)), groups, faces=faces, **rings)


def _grid(nx: int, ny: int, dx: float = 1.0, dy: float = 1.0):
    """An *nx* by *ny* grid in the plane, vertex ``i * ny + j`` at ``(i dx, j dy)``, each cell cut along the same diagonal into two triangles."""
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    corner = (i[:-1, :-1] * ny + j[:-1, :-1]).ravel()
    faces = np.r_[np.c_[corner, corner + ny, corner + ny + 1], np.c_[corner, corner + ny + 1, corner + 1]]
    return np.c_[i.ravel() * dx, j.ravel() * dy], faces


def _jittered_grid(nx: int, ny: int):
    """A :func:`_grid` of unit cells whose interior vertices move by up to a fifth of a cell, too little to fold a triangle."""
    vertices, faces = _grid(nx, ny)
    interior = (vertices > 0).all(axis=1) & (vertices < [nx - 1, ny - 1]).all(axis=1)
    vertices[interior] += np.random.default_rng(0).uniform(-0.2, 0.2, (interior.sum(), 2))
    return vertices, faces


def _ring_distance(vertices, centre: int):
    """The number of edges from vertex *centre* to each vertex of a unit :func:`_grid`, whose diagonals join steps of equal sign."""
    di, dj = (vertices - vertices[centre]).T
    return np.where(di * dj >= 0, np.maximum(abs(di), abs(dj)), abs(di) + abs(dj))


def _face_gradient(bundle: dict, values):
    """The gradient of the vertex field *values* on every face through the bundle's basis: (3, F), or (3, T, F) for frames (T, V)."""
    corners = values[..., bundle["faces"][0]]
    return np.stack([(bundle[f"grad_{axis}"][0] * corners).sum(axis=-2) for axis in "xyz"])


def _potential(bundle: dict, field):
    """The chain's Helmholtz-Hodge potential of the face field *field* (3, F): the area times the field against each corner's gradient, summed onto the vertices through the face table, then the Poisson operator."""
    flux = np.zeros(bundle["poisson"].shape[-1])
    divergence = bundle["area"][0] * sum(bundle[f"grad_{axis}"][0] * field[k] for k, axis in enumerate("xyz"))
    np.add.at(flux, bundle["faces"][0], divergence)
    return bundle["poisson"][0] @ flux


def test_the_gradient_basis_differentiates_a_linear_field_exactly_on_every_face():
    """On a planar mesh, the Delaunay triangulation of jittered grid points, a x + b y + c has the gradient (a, b, 0) on every face, to rounding."""
    vertices, _ = _jittered_grid(7, 6)
    gradient = _face_gradient(_operators(vertices), 0.7 * vertices[:, 0] - 1.3 * vertices[:, 1] + 2.0)
    np.testing.assert_allclose(gradient, np.broadcast_to([[0.7], [-1.3], [0.0]], gradient.shape), rtol=0, atol=1e-12)


def test_the_gradient_basis_is_read_at_each_face_corner_in_x_y_z_blocks():
    """Two right triangles in the x-z plane, whose hat functions can be written down.

    Face 0 has corners 0 (0, 0, 0), 1 (1, 0, 0) and 2 (0, 0, 1), with hat functions 1 - x - z, x and z, whose gradients are (-1, 0, -1), (1, 0, 0) and (0, 0, 1). Face 1 has corners 1, 3 (1, 0, 1) and 2, with hat functions 1 - z, x + z - 1 and 1 - x, whose gradients are (0, 0, -1), (1, 0, 1) and (-1, 0, 0). ``igl.grad`` stacks its rows in an x, a y and a z block of one row per face, and corner j of face f is read at column ``faces[f, j]`` of row f of each block, as ``compute_phase_gradient`` reshapes them to (3, F). So the corner-major x table is (-1, 1, 0) on face 0 and (0, 1, -1) on face 1, the z table (-1, 0, 1) and (-1, 1, 0), and the y table is zero.
    """
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    bundle = _operators(vertices, np.array([[0, 1, 2], [1, 3, 2]]))
    np.testing.assert_array_equal(bundle["faces"][0], [[0, 1], [1, 3], [2, 2]])
    expected = {"x": [[-1, 0], [1, 1], [0, -1]], "y": np.zeros((3, 2)), "z": [[-1, -1], [0, 1], [1, 0]]}
    for axis, table in expected.items():
        np.testing.assert_allclose(bundle[f"grad_{axis}"][0], table, rtol=0, atol=1e-12, err_msg=axis)


def test_the_face_areas_sum_to_the_area_of_the_domain():
    vertices, faces = _grid(6, 5, dx=0.5, dy=2.0)
    assert _operators(vertices, faces)["area"].sum() == pytest.approx(2.5 * 8.0, rel=1e-12)


def test_the_poisson_operator_inverts_the_cotangent_laplacian_up_to_the_mean():
    """``-L_cot`` applied to ``poisson f`` gives back f less its mean, the one component the Laplacian cannot see."""
    import igl

    vertices, faces = _jittered_grid(7, 6)
    laplacian = igl.cotmatrix(np.c_[vertices, np.zeros(len(vertices))], faces).toarray()
    f = np.random.default_rng(1).standard_normal(len(vertices))
    np.testing.assert_allclose(-laplacian @ (_operators(vertices, faces)["poisson"][0] @ f), f - f.mean(), rtol=0, atol=1e-10)


@pytest.mark.parametrize("surface", ["plane", "sphere"])
def test_the_potential_of_a_gradient_field_recovers_its_potential_up_to_a_constant(surface, icosphere):
    """The Helmholtz-Hodge potential of a pure gradient field is that field's potential less its mean, on a planar mesh and on a closed sphere."""
    vertices, faces = _jittered_grid(7, 6) if surface == "plane" else icosphere(2)
    bundle = _operators(vertices, faces)
    potential = np.sin(vertices[:, 0]) * vertices[:, 1] + vertices[:, 0] ** 2
    recovered = _potential(bundle, _face_gradient(bundle, potential))
    np.testing.assert_allclose(recovered, potential - potential.mean(), rtol=0, atol=1e-9)


@pytest.mark.parametrize("mesh", ["square", "spiral"])
def test_the_potential_is_kollers_regularised_solve_up_to_a_constant(mesh):
    """``compute_helmholtz_hodge_decomposition`` worked through: the divergence ``-Gᵀ diag(A, A, A)`` of the field he passes, the negated unit gradient, flattened axis by axis, with A the face areas, solved against ``-L + 1e-12 I``. The 1e-12 lifts the constant null space of -L, so his solve carries a constant the pseudo-inverse does not; less their means, the two potentials agree.

    By hand on the unit square cut along its diagonal from (0, 0) to (1, 1), the field (1, 0) on both faces: on face (0, 1, 2) the hat functions 1 - x, x - y and y have x slopes -1, 1 and 0, on face (0, 2, 3) the hat functions 1 - y, x and y - x have 0, 1 and -1, and each face has area 1/2, so the divergence is (-1/2, 1/2, 1/2, -1/2). Because -L = Gᵀ diag(A, A, A) G and x has the gradient (1, 0) on every face, -L x is that divergence, and both potentials are x up to a constant: (-1/2, 1/2, 1/2, -1/2) less the mean. The spiral case, a unit field half outward and half rotating on the jittered 7 by 6 grid, has no closed form and compares the two solves alone, to 1e-9 of the largest value.
    """
    import igl
    from scipy.sparse import identity
    from scipy.sparse.linalg import splu

    if mesh == "square":
        vertices, faces = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), np.array([[0, 1, 2], [0, 2, 3]])
        field = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    else:
        vertices, faces = _jittered_grid(7, 6)
        x, y = (vertices[faces].mean(axis=1) - [3.0, 2.5]).T
        field = np.stack([x - y, x + y, np.zeros(len(faces))])
        field /= np.linalg.norm(field, axis=0)
    ours = _potential(_operators(vertices, faces), field)
    solid = np.c_[vertices, np.zeros(len(vertices))]
    divergence = -igl.grad(solid, faces).T @ (np.tile(igl.doublearea(solid, faces) / 2, 3) * (-field).ravel())
    regularised = (-igl.cotmatrix(solid, faces) + 1e-12 * identity(len(vertices))).tocsc()
    theirs = splu(regularised).solve(divergence)
    theirs -= theirs.mean()
    np.testing.assert_allclose(ours - ours.mean(), theirs, rtol=0, atol=1e-9 * np.abs(theirs).max())
    if mesh == "square":
        np.testing.assert_allclose(ours - ours.mean(), [-0.5, 0.5, 0.5, -0.5], rtol=0, atol=1e-12)


@pytest.mark.parametrize("k", [1, 2])
def test_a_corner_vertex_lists_each_face_once_per_ring_vertex_it_touches(k):
    """``k_ring_neighbours`` and the listing loop of ``compute_wave_template`` at vertex 0 of the 3 by 3 grid, worked by hand.

    The grid's faces are 0 (0, 3, 4), 1 (1, 4, 5), 2 (3, 6, 7), 3 (4, 7, 8), 4 (0, 4, 1), 5 (1, 5, 2), 6 (3, 7, 4) and 7 (4, 8, 5). The walk replaces the ring with the neighbours of its vertices at each step and leaves out vertex 0 itself; on a triangle mesh that reaches every vertex within k edges. One step reaches 1, 3 and 4, and a face is listed once for each of them it has as a corner: faces 0, 1, 4 and 6 twice, faces 2, 3, 5 and 7 once, 12 entries, the 3 + 3 + 6 faces incident to vertices 1, 3 and 4. Two steps reach all eight other vertices, so each face is listed once per corner other than 0: twice for faces 0 and 4, which have 0 as a corner, three times for the other six, 22 entries.
    """
    multiplicity = {1: [2, 2, 1, 1, 2, 1, 2, 1], 2: [2, 3, 3, 3, 2, 3, 3, 3]}[k]
    vertices, faces = _grid(3, 3)
    bundle = _operators(vertices, faces, k_ring=k)
    listed = bundle["kring_faces"][0, 0][bundle["kring_mask"][0, 0] == 1]
    np.testing.assert_array_equal(np.bincount(listed, minlength=len(faces)), multiplicity)
    assert bundle["kring_count"][0, 0] == sum(multiplicity)


@pytest.mark.parametrize(("k", "entries", "faces_inside"), [(1, 36, 24), (2, 108, 54)])
def test_an_interior_vertex_lists_the_faces_of_its_ring_once_per_ring_vertex_they_touch(k, entries, faces_inside):
    """At a vertex four edges from the border of a regular grid, a face is listed once for each of its corners 1 to k edges away, which lists the faces inside the (k + 1)-ring: Koller's k = 2 is the 3-ring neighbourhood of his Methods, as the comment in compute_wave_template says."""
    vertices, faces = _grid(9, 9)
    centre = 4 * 9 + 4
    bundle = _operators(vertices, faces, k_ring=k)
    listed = bundle["kring_faces"][0, centre][bundle["kring_mask"][0, centre] == 1]
    distance = _ring_distance(vertices, centre)[faces]
    touching = ((distance >= 1) & (distance <= k)).sum(axis=1)
    assert bundle["kring_count"][0, centre] == len(listed) == entries
    np.testing.assert_array_equal(np.bincount(listed, minlength=len(faces)), touching)
    assert np.count_nonzero(touching) == np.count_nonzero((distance <= k + 1).all(axis=1)) == faces_inside


@pytest.mark.parametrize(("null_ring", "decision_ring"), [(-1, -1), (0, 0), (1, 1), (2, 2), (-1, 1), (0, 1)])
def test_a_border_ring_leaves_out_every_vertex_within_that_many_edges_of_the_border(null_ring, decision_ring):
    """``k_ring_boundary`` marks the border vertices at ring 0 and, for each further ring, every vertex adjacent to a marked one, ``(adj @ mask) != 0``. The 7 by 7 grid has 49 - 25 = 24 vertices on its border, so ring -1 keeps all 49, ring 0 the 5 by 5 = 25 inside the border, ring 1 the 3 by 3 = 9 and ring 2 the centre alone, each kept vertex more than that many edges from the border."""
    kept = {-1: 49, 0: 25, 1: 9, 2: 1}
    vertices, faces = _grid(7, 7)
    bundle = _operators(vertices, faces, null_ring=null_ring, decision_ring=decision_ring)
    edges_from_border = np.minimum(vertices, 6 - vertices).min(axis=1)
    assert (bundle["null_mask"][0].sum(), bundle["decision_mask"][0].sum()) == (kept[null_ring], kept[decision_ring])
    np.testing.assert_array_equal(bundle["null_mask"][0], edges_from_border > null_ring)
    np.testing.assert_array_equal(bundle["decision_mask"][0], edges_from_border > decision_ring)


@pytest.mark.parametrize(("null_ring", "decision_ring"), [(1, 0), (1, -1), (2, 1)])
def test_a_decision_ring_inside_the_null_ring_is_refused(null_ring, decision_ring):
    vertices, faces = _grid(4, 4)
    with pytest.raises(ValueError, match=f"decision_ring {decision_ring} lies inside null_ring {null_ring}"):
        _operators(vertices, faces, null_ring=null_ring, decision_ring=decision_ring)


def test_the_groups_carry_their_labels_and_no_order_of_the_inputs_changes_the_bundle():
    """Two hemispheres, the right one's nodes first in the network, given once in network order and once with every row reversed and the vertices, groups and reference labelled: the bundle is the same array for array, its groups ordered by label, each with its own nodes (SO1)."""
    hemisphere, faces = _grid(3, 3)
    vertices, faces = np.r_[hemisphere, hemisphere + [5.0, 0.0]], np.r_[faces, faces + 9]
    labels = np.array(_labels(18))
    groups, reference = np.repeat(["rh", "lh"], 9), np.arange(18.0) ** 2
    flip = np.arange(18)[::-1]

    def labelled(values):
        return xr.DataArray(values[flip], dims=("node", "axis")[: values.ndim], coords={"node": labels[flip]})

    forward = mesh_operators(vertices, list(labels), groups, faces=faces, reference=reference)
    backward = mesh_operators(
        labelled(vertices), list(labels), labelled(groups), faces=np.argsort(flip)[faces], reference=labelled(reference)
    )

    assert forward["group"].tolist() == ["lh", "rh"]
    np.testing.assert_array_equal(forward["members"], [np.arange(9, 18), np.arange(9)])
    np.testing.assert_array_equal(forward["reference"], reference[forward["members"]])
    for key, value in forward.items():
        np.testing.assert_array_equal(backward[key], value, err_msg=key)


def test_a_smaller_group_is_padded_with_entries_that_change_nothing():
    """A 3 by 3 group beside a 3 by 4 one: the smaller group's real block is what it gives alone, and every padded entry is zero, its count 1."""
    small, small_faces = _grid(3, 3)
    large, large_faces = _grid(3, 4)
    vertices, faces = np.r_[small, large + [5.0, 0.0]], np.r_[small_faces, large_faces + 9]
    bundle = mesh_operators(vertices, _labels(21), np.repeat(["a", "b"], [9, 12]), faces=faces)
    alone = _operators(small, small_faces, groups="a")

    assert bundle["poisson"].shape == (2, 12, 12)
    assert (bundle["kring_count"][0, 9:] == 1).all()
    for key in sorted(alone.keys() - {"group"}):
        ours, theirs = np.asarray(bundle[key][0]), np.asarray(alone[key][0])
        real = tuple(slice(0, size) for size in theirs.shape)
        np.testing.assert_array_equal(ours[real], theirs, err_msg=key)
        padded = np.ones(ours.shape, dtype=bool)
        padded[real] = False
        assert key == "kring_count" or not ours[padded].any(), key


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("spanning", "corners in two groups"),
        ("solid", "3-D vertices need faces"),
        ("short", "vertices has 8 entries for 9 labels"),
        ("stranger", "no network node has"),
        ("loose", "lie in no face"),
    ],
)
def test_inputs_that_do_not_make_a_mesh_of_each_group_are_refused(case, message):
    vertices, faces = _grid(3, 3)
    labels = _labels(9)
    renamed = xr.DataArray(vertices, dims=("node", "axis"), coords={"node": [*labels[:8], "elsewhere"]})
    vertices, groups, faces = {
        "spanning": (vertices, np.repeat(["a", "b"], [4, 5]), faces),
        "solid": (np.c_[vertices, np.zeros(9)], "a", None),
        "short": (vertices[:8], "a", faces[:1]),
        "stranger": (renamed, "a", faces),
        "loose": (vertices, "a", faces[:2]),
    }[case]
    with pytest.raises(ValueError, match=message):
        mesh_operators(vertices, labels, groups, faces=faces)


def test_producer_by_iri_materialises(tmp_path):
    """``producer: {iri: tvbo:functions/MeshOperators}`` takes the curated record's callable and outputs and binds every array the function returns from one artifact, the group labels read back as text."""
    import h5py

    hemisphere, faces = _grid(3, 3)
    arguments = {
        "vertices": np.r_[hemisphere, hemisphere + [5.0, 0.0]].tolist(),
        "faces": np.r_[faces, faces + 9].tolist(),
        "node_labels": _labels(18),
        "groups": ["rh"] * 9 + ["lh"] * 9,
    }
    mesh = Parameter(name="mesh", producer=FunctionCall(iri="tvbo:functions/MeshOperators", arguments=arguments))

    bound = param_io.materialise_bundle(mesh, cache_dir=tmp_path)

    (artifact,) = {path for path, _ in bound.values()}
    with h5py.File(artifact, "r") as stored:
        assert sorted(stored) == sorted(bound) == sorted(param_io.bundle_outputs(mesh))
    assert param_io.read_artifact(artifact, "group").tolist() == ["lh", "rh"]
    np.testing.assert_array_equal(param_io.read_artifact(artifact, "members"), [np.arange(9, 18), np.arange(9)])
