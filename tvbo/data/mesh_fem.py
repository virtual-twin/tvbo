"""P1 finite-element operators for a triangular mesh, flat or curved.

:func:`~tvbo.data.mesh_io.read_mesh` turns a mesh file into vertices and faces; this turns
those into the two matrices a field equation is discretised with — the mass matrix
``int(u v)`` and the stiffness matrix ``int(grad(u).grad(v))``. Together they are the whole
of a P1 discretisation, so ``a*laplacian(u)`` assembles as ``-a*K`` and ``a*u`` as ``a*M``.

The reason this exists rather than a call into a FEM library: a cortical surface is a
2-manifold embedded in 3-space, and the general-purpose libraries assemble on domains whose
element dimension equals their coordinate dimension. Their affine mapping inverts a square
Jacobian, which a triangle carrying three coordinates does not have — scikit-fem builds such
a mesh and then fails inside the mapping. For P1 the manifold case needs no mapping at all:
the basis gradients are tangential and are written in closed form below, which makes the
brain-surface case exact rather than unsupported.

Coefficients may vary per vertex, which is what lets a propagation scale differ across
cortex. Both weighted forms are exact for a P1 coefficient, not a one-point approximation
of one: a gradient product is constant on a triangle so the weighted stiffness needs only
the coefficient's element mean, while the weighted mass integrates the full cubic.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sps


def _as_faces(faces) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (F, 3) triangles, got shape {faces.shape}")
    return faces


def _opposite_edges(vertices, faces):
    """Per-triangle areas and the edge vector opposite each of its three vertices.

    Returns ``(areas (F,), edges (F, 3, dim))``. The edges are ordered so that ``edges[:, i]``
    faces vertex ``i``, which is what makes the closed-form gradient below hold in any
    embedding dimension.
    """
    vertices = np.asarray(vertices, dtype=float)
    faces = _as_faces(faces)
    if vertices.ndim != 2 or vertices.shape[1] not in (2, 3):
        raise ValueError(f"vertices must be (V, 2) or (V, 3), got shape {vertices.shape}")

    p = vertices[faces]
    edges = np.stack([p[:, 2] - p[:, 1], p[:, 0] - p[:, 2], p[:, 1] - p[:, 0]], axis=1)

    span = -edges[:, 1], edges[:, 2]
    if vertices.shape[1] == 2:
        cross = span[1][:, 0] * span[0][:, 1] - span[1][:, 1] * span[0][:, 0]
        areas = 0.5 * np.abs(cross)
    else:
        areas = 0.5 * np.linalg.norm(np.cross(span[1], span[0]), axis=1)

    if not np.all(areas > 0):
        bad = int(np.count_nonzero(areas <= 0))
        raise ValueError(
            f"{bad} of {len(areas)} triangles have zero area; a degenerate element has no "
            f"P1 gradient and would divide by zero. Repair or decimate the mesh first."
        )
    return areas, edges


def _assemble(faces, local, n_vertices):
    """Scatter per-triangle ``(F, 3, 3)`` blocks into a sparse ``(V, V)`` matrix."""
    rows = np.repeat(faces, 3, axis=1).ravel()
    cols = np.tile(faces, (1, 3)).ravel()
    return sps.coo_matrix((local.reshape(-1), (rows, cols)), shape=(n_vertices, n_vertices)).tocsr()


def _element_values(coefficient, faces, n_vertices):
    """A coefficient as per-triangle vertex values ``(F, 3)``, or ``None`` when unit."""
    if coefficient is None:
        return None
    arr = np.asarray(coefficient, dtype=float)
    if arr.ndim == 0:
        return np.full((len(faces), 3), float(arr))
    arr = arr.ravel()
    if arr.size != n_vertices:
        raise ValueError(f"coefficient has {arr.size} values but the mesh has {n_vertices} vertices")
    return arr[faces]


def triangle_areas(vertices, faces) -> np.ndarray:
    """Area of every triangle, in the mesh's own length unit squared."""
    return _opposite_edges(vertices, faces)[0]


def p1_stiffness(vertices, faces, coefficient=None) -> sps.csr_matrix:
    """``int(c grad(u) . grad(v))`` — the cotangent operator, weighted by ``c``.

    Positive semi-definite with the constant vector in its kernel, so a Laplacian term
    enters an operator NEGATED. On a closed surface that kernel is exactly one-dimensional,
    which is the discrete statement that no flux leaves the domain.

    Args:
        vertices: ``(V, 2)`` or ``(V, 3)`` coordinates. Three columns is a surface in space.
        faces: ``(F, 3)`` triangles.
        coefficient: ``None``, a scalar, or a per-vertex array — a spatially varying
            diffusivity, assembled as ``div(c grad(u))`` rather than ``c laplacian(u)``.
    """
    areas, edges = _opposite_edges(vertices, faces)
    faces = _as_faces(faces)
    n = int(np.asarray(vertices).shape[0])

    local = np.einsum("fid,fjd->fij", edges, edges) / (4.0 * areas)[:, None, None]
    values = _element_values(coefficient, faces, n)
    if values is not None:
        local = local * values.mean(axis=1)[:, None, None]
    return _assemble(faces, local, n)


def p1_mass(vertices, faces, coefficient=None) -> sps.csr_matrix:
    """``int(c u v)`` — the consistent mass matrix, weighted by ``c``.

    Consistent, not lumped: the row sums give the area each vertex carries, but the
    off-diagonal entries are kept. Lumping is a first-order approximation whose error grows
    with spatial frequency, so it distorts exactly the high modes a wave equation resolves.
    """
    areas, _ = _opposite_edges(vertices, faces)
    faces = _as_faces(faces)
    n = int(np.asarray(vertices).shape[0])

    unit = (np.ones((3, 3)) + np.eye(3)) / 12.0
    values = _element_values(coefficient, faces, n)
    if values is None:
        local = areas[:, None, None] * unit
    else:
        total = values.sum(axis=1)
        local = (values[:, :, None] + values[:, None, :] + total[:, None, None]) / 60.0
        diagonal = (4.0 * values + 2.0 * total[:, None]) / 60.0
        idx = np.arange(3)
        local[:, idx, idx] = diagonal
        local = areas[:, None, None] * local
    return _assemble(faces, local, n)


def boundary_vertices(faces) -> np.ndarray:
    """Vertices on the mesh boundary — empty for a closed surface.

    An edge shared by one triangle is a boundary edge. A closed cortical surface has none,
    which is why a Dirichlet condition declared on one constrains nothing.
    """
    faces = _as_faces(faces)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, index, counts = np.unique(edges, axis=0, return_index=True, return_counts=True)
    return np.unique(edges[index[counts == 1]])
