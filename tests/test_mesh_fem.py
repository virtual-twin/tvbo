"""P1 operators on a triangulated surface are the ones a FEM library would assemble.

A cortical mesh is a 2-manifold carrying three coordinates, which the general-purpose FEM libraries do not assemble on — scikit-fem builds the mesh and then divides by zero inside its affine mapping. So the operators are written in closed form, and these tests pin them against the two references that can settle it: scikit-fem itself wherever the two agree (a flat mesh), and the analytic Laplace-Beltrami spectrum of a sphere where only the closed form applies.
"""

import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from tvbo.data.mesh_fem import (
    boundary_vertices,
    p1_mass,
    p1_stiffness,
    triangle_areas,
)


@pytest.fixture(scope="module")
def flat():
    """A planar mesh with its scikit-fem basis, the case both assemblers cover."""
    skfem = pytest.importorskip("skfem")
    mesh = skfem.MeshTri.init_symmetric().refined(4)
    return mesh, skfem.Basis(mesh, skfem.ElementTriP1()), mesh.p.T, mesh.t.T


def test_planar_operators_equal_scikit_fems(flat):
    """Not merely close: the same P1 element assembled two ways is the same matrix."""
    import skfem
    from skfem.models.poisson import laplace, mass

    mesh, basis, vertices, faces = flat
    assert abs(p1_mass(vertices, faces) - skfem.asm(mass, basis)).max() < 1e-15
    assert abs(p1_stiffness(vertices, faces) - skfem.asm(laplace, basis)).max() < 1e-15
    assert np.array_equal(
        boundary_vertices(faces),
        np.unique(basis.get_dofs(mesh.boundary_facets()).flatten()),
    )


def test_operators_do_not_change_when_the_mesh_is_moved_through_space(flat):
    """The same triangles rotated out of the plane and translated: a surface operator depends on the metric, not on the coordinates it happens to be written in.

    This is the concrete regression. Handed a cortical surface, scikit-fem does not raise — it assembles the mesh's shadow on the xy-plane, and the mass matrix it returns totals the projected area rather than the surface area (33,070 mm2 against the true 69,589 mm2 on fsLR-32k). A rotation invariance check is what turns that from a plausible animation into a failing test.
    """
    _, _, vertices, faces = flat
    angle = 0.7
    rotation = np.array([[np.cos(angle), 0, -np.sin(angle)], [0, 1, 0], [np.sin(angle), 0, np.cos(angle)]])
    moved = np.column_stack([vertices, np.zeros(len(vertices))]) @ rotation.T + [5.0, -2.0, 3.0]

    assert abs(p1_stiffness(moved, faces) - p1_stiffness(vertices, faces)).max() < 1e-13
    assert abs(p1_mass(moved, faces) - p1_mass(vertices, faces)).max() < 1e-15


def test_the_sphere_spectrum_converges_to_the_analytic_one(icosphere):
    """``l(l+1)/R**2`` with multiplicity ``2l+1``, at second order in the mesh spacing.

    The eigenvalue test is the one that would catch a plausible-but-wrong operator: a stiffness matrix can be symmetric, positive semi-definite and have the right nullspace while still discretising the wrong metric.

    Solved densely. These meshes reach 2562 vertices, so a direct solve costs about a second, and it is the only way the assertion is deterministic: the spectrum is degenerate in blocks of ``2l+1``, and an Arnoldi solve started from the random vector ``eigsh`` draws by default returns the ``l=3`` eigenvalue in place of an ``l=2`` one on roughly one run in sixty.
    """
    radius = 2.0
    exact = np.concatenate([[deg * (deg + 1) / radius**2] * (2 * deg + 1) for deg in range(4)])[:10]

    errors = []
    for subdivisions in (2, 3, 4):
        vertices, faces = icosphere(subdivisions, radius)
        M, K = p1_mass(vertices, faces), p1_stiffness(vertices, faces)
        found = np.sort(eigh(K.toarray(), M.toarray(), eigvals_only=True, subset_by_index=[0, 9]))
        assert abs(found[0]) < 1e-12, "a closed surface has exactly one zero mode"
        errors.append(np.abs(found[1:] - exact[1:]).max() / exact[-1])

    assert errors[-1] < 5e-3
    assert np.log2(errors[0] / errors[1]) == pytest.approx(2.0, abs=0.2)
    assert np.log2(errors[1] / errors[2]) == pytest.approx(2.0, abs=0.2)


def test_a_closed_surface_has_no_boundary_and_carries_its_whole_area(icosphere):
    """The mass matrix's total is the surface area, and the stiffness annihilates constants — the discrete statement that no flux leaves a closed surface, which is why a Dirichlet condition declared on a cortical mesh constrains nothing."""
    vertices, faces = icosphere(4, 2.0)
    M, K = p1_mass(vertices, faces), p1_stiffness(vertices, faces)
    ones = np.ones(len(vertices))

    assert boundary_vertices(faces).size == 0
    assert ones @ (M @ ones) == pytest.approx(triangle_areas(vertices, faces).sum(), rel=1e-12)
    assert ones @ (M @ ones) == pytest.approx(4 * np.pi * 4.0, rel=2e-3)
    assert np.abs(K @ ones).max() < 1e-12


def test_a_constant_coefficient_just_scales_the_operator(icosphere):
    """The gate on the weighted assembly: where the weighted and plain forms describe the same operator they must be the same matrix, not merely a similar one."""
    vertices, faces = icosphere(3, 2.0)
    for build in (p1_mass, p1_stiffness):
        plain, weighted = build(vertices, faces), build(vertices, faces, 0.37)
        assert abs(weighted - 0.37 * plain).max() < 1e-14
        assert abs(weighted - build(vertices, faces, np.full(len(vertices), 0.37))).max() < 1e-14


def test_a_varying_coefficient_integrates_exactly(icosphere):
    """Both weighted forms are exact for a P1 coefficient rather than a one-point rule, so each has an integral identity that pins it: ``1.M(c).1`` is the integral of ``c``, and ``M(c)`` stays symmetric positive-definite."""
    vertices, faces = icosphere(3, 2.0)
    c = 1.0 + 0.5 * np.sin(vertices[:, 0]) * np.cos(vertices[:, 2])
    ones = np.ones(len(vertices))

    exact = float(triangle_areas(vertices, faces) @ c[faces].mean(axis=1))
    weighted = p1_mass(vertices, faces, c)
    assert ones @ (weighted @ ones) == pytest.approx(exact, rel=1e-12)
    assert abs(weighted - weighted.T).max() < 1e-15
    assert np.all(eigsh(weighted.tocsc(), k=1, which="SA")[0] > 0)

    stiffness = p1_stiffness(vertices, faces, c)
    assert abs(stiffness - stiffness.T).max() < 1e-15
    assert np.abs(stiffness @ ones).max() < 1e-12, "a varying diffusivity still conserves"


def test_a_degenerate_triangle_is_refused(icosphere):
    """A zero-area element has no P1 gradient. Saying so beats dividing by zero and returning an operator full of infinities."""
    vertices, faces = icosphere(1, 1.0)
    vertices = vertices.copy()
    vertices[faces[0, 1]] = vertices[faces[0, 0]]
    with pytest.raises(ValueError, match="zero area"):
        p1_stiffness(vertices, faces)


def test_a_coefficient_of_the_wrong_length_is_refused(icosphere):
    vertices, faces = icosphere(1, 1.0)
    with pytest.raises(ValueError, match="vertices"):
        p1_stiffness(vertices, faces, np.ones(len(vertices) + 1))
