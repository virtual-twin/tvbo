"""The fixed arrays a travelling-wave detector on a triangulated mesh reads: the operators of the mesh, per group of its vertices, and the resampling tables of its two tests, the implementations behind the curated ``tvbo:functions/MeshOperators``, ``tvbo:functions/NodePermutations`` and ``tvbo:functions/SpinPermutations``.

They follow the detector of Koller et al. (2024, doi:10.1038/s41467-024-47860-x), which compares each face's phase gradient with the ideal divergent field around every vertex and takes the curl-free Helmholtz-Hodge potential of the unit gradient field. Each operator is laid out in gather form, a table indexed by face, corner or vertex, so that a frame's chain is a few takes and sums and one matrix product, and every group (a hemisphere) is padded to the largest, so that one chain runs over all of them. libigl and potpourri3d build them once, where the producer runs, and come with the optional extra ``tvbo[mesh]``.

The resampling tables index network nodes, not mesh vertices, so they need no mesh and neither library: each row moves the nodes of every group among themselves, by a uniform permutation or by a rotation of their positions on a sphere, and never into another group.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def mesh_operators(
    vertices,
    node_labels,
    groups,
    faces=None,
    reference=None,
    k_ring: int = 2,
    null_ring: int = -1,
    decision_ring: int = -1,
) -> dict:
    """Every fixed array a travelling-wave detector reads on a triangulated mesh, per group of its vertices, padded to the largest group.

    Mesh vertices meet network nodes by label. A labelled *vertices* (a ``node`` coordinate) names each row's node, and *faces* index its rows; an unlabelled one has one row per network node, in network order. Planar vertices get z = 0, and without *faces* each group of them is triangulated by Delaunay, as Koller triangulates his sheet (``spatial.Delaunay(pos).simplices``). The groups are ordered by label and each group's vertices by network node, so no order of the rows, the groups or the labels changes a value.

    The operators are those of Koller's ``modules/wave_detection_methods.py``. The gradient basis is ``igl.grad`` read at each face's corners, the operator ``compute_phase_gradient`` applies. Each vertex lists the faces ``compute_wave_template`` lists, a face once per ring vertex it touches, with the unit gradient of the heat-method geodesic distance from the vertex on each (``potpourri3d.MeshHeatMethodDistanceSolver`` with its defaults, as there). A border ring dilates ``igl.is_border_vertex`` as ``k_ring_boundary`` does. ``compute_helmholtz_hodge_decomposition`` solves ``-L + 1e-12 I`` per frame; ``poisson`` is the pseudo-inverse of ``-L`` with singular values below 1e-12 of the largest dropped, so a potential has zero mean on each connected component where Koller's carries a small constant.

    Args:
        vertices: Positions, (n, 2) or (n, 3), in the length unit the potential is wanted in, since a potential scales with it.
        node_labels: The network's node labels, in network order (``network.labels``).
        groups: The group of each vertex, labelled on ``node`` or in vertex order, or one label for a mesh of one group.
        faces: Triangles, (m, 3), indexing the rows of *vertices*; needed for 3-D vertices.
        reference: A per-node field, labelled on ``node`` or in network order, laid out on each group's vertices; NaN without one.
        k_ring: Each vertex lists the faces incident to every vertex of its k-ring except itself. Koller's ``k=2`` gives the faces inside the 3-ring, the "3-ring neighbourhood" of his Methods.
        null_ring: The border ring the null family of the wave test leaves out: -1 none, 0 the border vertices, k every vertex within k edges of them.
        decision_ring: The border ring left out of the vertices that decide a wave, counted as *null_ring*.

    Returns:
        A mapping of arrays, each on a leading group axis of length G and padded to the largest group's V vertices, F faces and K listed faces, every pad entry zero:

        - ``members`` (G, V): the network node of each vertex.
        - ``group`` (G,): each group's label, as text in HDF5's variable-length string type, which the produced-constant store writes and reads back as text.
        - ``vertex_mask`` (G, V) and ``face_mask`` (G, F): 1 for a real entry.
        - ``faces`` (G, 3, F): the corner-major face table, in the group's vertex numbers.
        - ``grad_x``, ``grad_y``, ``grad_z`` (G, 3, F): the gradient of each corner's hat function on its face, zero along z on a planar mesh.
        - ``area`` (G, F): the face areas.
        - ``poisson`` (G, V, V): the pseudo-inverse of the negated cotangent Laplacian.
        - ``kring_faces``, ``kring_mask`` (G, V, K) and ``kring_count`` (G, V): each vertex's listed faces, which entries are real, and how many, 1 on a padded vertex.
        - ``divergent_x``, ``divergent_y``, ``divergent_z`` (G, V, K): the ideal divergent field on each listed face.
        - ``null_mask``, ``decision_mask`` (G, V): 1 outside the border ring.
        - ``reference`` (G, V): the reference field on each vertex.
        - ``components`` (G,): the number of connected components of each group's mesh.

    Raises:
        ImportError: libigl or potpourri3d is not installed.
        ValueError: A ring out of range or a decision ring inside the null ring; a label that repeats, that no network node has or that a labelled input lacks; an unlabelled input of the wrong length; 3-D vertices without faces; a face across two groups; a vertex in no face.
    """
    if k_ring < 1 or min(null_ring, decision_ring) < -1:
        raise ValueError(
            f"k_ring must be at least 1, and null_ring and decision_ring at least -1; got k_ring={k_ring}, null_ring={null_ring}, decision_ring={decision_ring}."
        )
    if decision_ring < null_ring:
        raise ValueError(
            f"decision_ring {decision_ring} lies inside null_ring {null_ring}, so a wave would be decided on vertices the null family leaves out; the decision ring must be at least the null ring."
        )
    import h5py
    from scipy.spatial import Delaunay

    node_index = _label_index(node_labels, "node_labels")
    if _is_labelled(vertices):
        vertex_labels = list(_label_index(vertices["node"].values, "vertices"))
        positions = vertices.transpose("node", ...).values
    else:
        vertex_labels = list(node_index)
        positions = _aligned(vertices, vertex_labels, "vertices")
    unknown = [label for label in vertex_labels if label not in node_index]
    if unknown:
        raise ValueError(f"vertices name {len(unknown)} label(s) no network node has, {unknown[:5]} among them.")
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] not in (2, 3):
        raise ValueError(f"vertices must be positions of shape (n, 2) or (n, 3), got {positions.shape}.")
    if faces is None and positions.shape[1] == 3:
        raise ValueError("3-D vertices need faces; only planar vertices are triangulated by Delaunay.")
    vertex_nodes = np.array([node_index[label] for label in vertex_labels], dtype=np.int64)
    if np.ndim(groups) == 0:
        vertex_groups = np.full(len(positions), str(groups))
    else:
        vertex_groups = _aligned(groups, vertex_labels, "groups").astype(str)
    if reference is None:
        node_reference = np.full(len(node_index), np.nan)
    else:
        node_reference = _aligned(reference, list(node_index), "reference").astype(float)
    if faces is not None:
        faces = np.asarray(faces, dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must be triangles of shape (m, 3), got {faces.shape}.")
        face_groups = vertex_groups[faces]
        spanning = int((face_groups != face_groups[:, :1]).any(axis=1).sum())
        if spanning:
            raise ValueError(f"{spanning} face(s) have corners in two groups; each group must be a mesh of its own.")

    labels = sorted({str(group) for group in vertex_groups})
    per_group = []
    for label in labels:
        rows = np.flatnonzero(vertex_groups == label)
        rows = rows[np.argsort(vertex_nodes[rows])]
        if faces is None:
            local_faces = Delaunay(positions[rows]).simplices
        else:
            local = np.full(len(positions), -1)
            local[rows] = np.arange(len(rows))
            local_faces = local[faces[face_groups[:, 0] == label]]
        unused = len(rows) - np.unique(local_faces).size
        if unused:
            raise ValueError(
                f"group {label!r}: {unused} vertex/vertices lie in no face, so they have no gradient, ring or potential."
            )
        mesh = np.pad(positions[rows], ((0, 0), (0, 3 - positions.shape[1])))
        operators = _group_operators(mesh, np.asarray(local_faces, dtype=np.int64), k_ring, null_ring, decision_ring)
        per_group.append({"members": vertex_nodes[rows], "reference": node_reference[vertex_nodes[rows]], **operators})
    bundle = {key: _stacked([operators[key] for operators in per_group]) for key in per_group[0]}
    bundle["kring_count"] = np.maximum(bundle["kring_mask"].sum(axis=-1), 1.0)
    bundle["group"] = np.array(labels, dtype=h5py.string_dtype())
    return bundle


def _group_operators(vertices: np.ndarray, faces: np.ndarray, k_ring: int, null_ring: int, decision_ring: int) -> dict:
    """The operators of one group's mesh, *vertices* (n, 3) and *faces* (m, 3) in the group's vertex numbers, every vertex in a face, unpadded."""
    try:
        import igl
        import potpourri3d
    except ImportError as exc:
        raise ImportError(
            f"MeshOperators builds its operators with libigl and potpourri3d, and {exc.name} is not installed: pip install 'tvbo[mesh]'"
        ) from exc
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    n, m = len(vertices), len(faces)
    rows = np.broadcast_to(np.arange(3)[:, None, None] * m + np.arange(m), (3, 3, m))
    corners = np.broadcast_to(faces.T, (3, 3, m))
    basis = np.asarray(igl.grad(vertices, faces).tocsr()[rows.ravel(), corners.ravel()]).reshape(3, 3, m)
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2).T
    adjacency = sparse.csr_matrix((np.ones(3 * m, dtype=np.int64), (edges[0], edges[1])), shape=(n, n))
    adjacency = ((adjacency + adjacency.T) > 0).astype(np.int64)
    listed = _kring_lists(adjacency, faces, k_ring)
    width = max(len(entry) for entry in listed)
    kring_faces = np.zeros((n, width), dtype=np.int64)
    kring_mask = np.zeros((n, width))
    divergent = np.zeros((3, n, width))
    solver = potpourri3d.MeshHeatMethodDistanceSolver(vertices, faces)
    for vertex, entry in enumerate(listed):
        field = (basis[:, :, entry] * solver.compute_distance(vertex)[faces[entry].T]).sum(axis=1)
        kring_faces[vertex, : len(entry)] = entry
        kring_mask[vertex, : len(entry)] = 1.0
        divergent[:, vertex, : len(entry)] = field / np.linalg.norm(field, axis=0)
    border = np.asarray(igl.is_border_vertex(faces), dtype=bool)
    return {
        "vertex_mask": np.ones(n),
        "face_mask": np.ones(m),
        "faces": faces.T,
        "grad_x": basis[0],
        "grad_y": basis[1],
        "grad_z": basis[2],
        "area": igl.doublearea(vertices, faces) / 2,
        "poisson": np.linalg.pinv(-igl.cotmatrix(vertices, faces).toarray(), rtol=1e-12, hermitian=True),
        "kring_faces": kring_faces,
        "kring_mask": kring_mask,
        "divergent_x": divergent[0],
        "divergent_y": divergent[1],
        "divergent_z": divergent[2],
        "null_mask": _outside_ring(adjacency, border, null_ring),
        "decision_mask": _outside_ring(adjacency, border, decision_ring),
        "components": np.int64(connected_components(adjacency, directed=False)[0]),
    }


def _kring_lists(adjacency, faces: np.ndarray, k_ring: int) -> list[np.ndarray]:
    """Per vertex, the faces incident to every vertex of its *k_ring*-ring except itself, each once per such vertex it has as a corner, in face order.

    The ring is what ``k_ring_neighbours`` walks, the vertices adjacent to the previous ring, *k_ring* times from the vertex, and a face is listed as often as ``compute_wave_template`` appends it, once for each ring vertex whose incident faces include it.
    """
    from scipy import sparse

    n, m = adjacency.shape[0], len(faces)
    ring = sparse.identity(n, dtype=np.int64, format="csr")
    for _ in range(k_ring):
        ring = ((adjacency @ ring) > 0).astype(np.int64)
    ring = sparse.triu(ring, 1) + sparse.tril(ring, -1)
    incidence = sparse.csr_matrix((np.ones(3 * m, dtype=np.int64), (faces.ravel(), np.repeat(np.arange(m), 3))), shape=(n, m))
    multiplicity = (ring @ incidence).tocsr()
    multiplicity.sort_indices()
    bounds = zip(multiplicity.indptr[:-1], multiplicity.indptr[1:], strict=True)
    return [np.repeat(multiplicity.indices[start:stop], multiplicity.data[start:stop]) for start, stop in bounds]


def _outside_ring(adjacency, border: np.ndarray, ring: int) -> np.ndarray:
    """1 on each vertex outside the border ring *ring* and 0 inside it, the ring dilated from *border* as ``k_ring_boundary`` dilates it: none for -1, the border for 0, and each step the vertices adjacent to the last."""
    inside = border if ring >= 0 else np.zeros(len(border), dtype=bool)
    for _ in range(ring):
        inside = (adjacency @ inside) != 0
    return (~inside).astype(float)


def _stacked(arrays: list) -> np.ndarray:
    """The per-group *arrays* on a leading group axis, each padded with zeros at the end of every axis to the largest shape among them."""
    shape = np.max([np.shape(array) for array in arrays], axis=0).astype(int)
    stacked = np.zeros((len(arrays), *shape), dtype=np.result_type(*arrays))
    for group, array in enumerate(arrays):
        stacked[(group, *(slice(0, size) for size in np.shape(array)))] = array
    return stacked


def _label_index(labels, what: str) -> dict:
    """``{label: position}`` over *labels* read as text. A label that occurs twice is refused, since it would pair with two entries."""
    labels = [str(label) for label in np.asarray(labels).ravel()]
    repeated = sorted(label for label, count in Counter(labels).items() if count > 1)
    if repeated:
        raise ValueError(f"{what} names {repeated[:5]} more than once; a label pairs with one entry.")
    return {label: position for position, label in enumerate(labels)}


def _is_labelled(values) -> bool:
    """Whether *values* is a labelled array with a ``node`` coordinate."""
    return "node" in getattr(values, "coords", ())


def _aligned(values, labels: list, what: str) -> np.ndarray:
    """*values* in the order of *labels*: a labelled array is reordered by its ``node`` labels, refusing one it lacks, and anything else is taken as already in that order, one entry per label."""
    if _is_labelled(values):
        index = _label_index(values["node"].values, what)
        missing = [label for label in labels if label not in index]
        if missing:
            raise ValueError(f"{what} has no entry for {len(missing)} label(s), {missing[:5]} among them.")
        return np.asarray(values.transpose("node", ...).values)[[index[label] for label in labels]]
    values = np.asarray(values)
    if len(values) != len(labels):
        raise ValueError(
            f"{what} has {len(values)} entries for {len(labels)} labels; label it on `node` or give one entry per label, in order."
        )
    return values


def node_permutations(node_labels, groups, n: int, seed: int) -> dict:
    """A table of *n* uniform random permutations of the network nodes within each group: the implementation behind the curated ``tvbo:functions/NodePermutations``.

    Each row permutes the nodes of every group among themselves and never moves a node into another group, as Koller permutes each hemisphere's phases with one ``rng.permutation`` per surrogate (``scripts/analysis/wave_detection_cortical.py``, ``40_analysis.py``, ``50_analysis.py``; one group on the sheet, ``wave_detection_2d.py``). Each group draws its rows from its own ``numpy.random.default_rng`` stream, keyed by the seed and the group's label, and permutes its nodes in label order, so no order of the nodes or the groups, and no other group, changes its rows.

    Args:
        node_labels: The network's node labels, in network order (``network.labels``).
        groups: The group of each node, labelled on ``node`` or in network order, or one label for all nodes.
        n: The number of permutations, the rows of the table.
        seed: The seed every group's stream is keyed by.

    Returns:
        A mapping of two arrays:

        - ``index`` (n, N): row r names, for each node, the node it takes its value from, so a node field ``x`` permuted by row r is ``x[index[r]]``.
        - ``group`` (N,): each node's group label, in network order, as text in HDF5's variable-length string type.

    Raises:
        ValueError: No *n* or *seed*, or *n* below 1; a node label that repeats; a node without a group; a group of one node.
    """
    labels = list(_label_index(node_labels, "node_labels"))
    node_group, members = _resampling_groups(labels, groups, n, seed)
    index = np.empty((n, len(labels)), dtype=np.int64)
    for label, nodes in members.items():
        rng = np.random.default_rng([seed, *label.encode()])
        index[:, nodes] = nodes[np.stack([rng.permutation(len(nodes)) for _ in range(n)])]
    return _table(index, node_group)


def spin_permutations(sphere, node_labels, groups, n: int, seed: int) -> dict:
    """A table of *n* spin permutations of the network nodes within each group (Alexander-Bloch et al. 2018, doi:10.1016/j.neuroimage.2018.05.070): the implementation behind the curated ``tvbo:functions/SpinPermutations``.

    Row r turns every node's position on the sphere by the r-th of *n* random rotations, and each node takes the node of its own group nearest to where it was turned to. This is BrainSpace's ``SpinPermutations`` (``null_models/spin.py`` of v0.1.4, the version Koller pins), fitted once per group with the same ``random_state``. The rotations are drawn as BrainSpace draws them, from ``numpy.random.RandomState(seed)``, so they are uniform over rotations, and the same rotations turn every group. The reassignment is BrainSpace's default (``unique=False``): a row may give two nodes one source and leave another node out, and an exact tie goes to the first node by label.

    The groups decide which nodes compete for one reassignment. A group per hemisphere is how Koller's directed-wave test fits the spins (``scripts/analysis/40_analysis_potentials.py``, ``flow_analysis_cortical.py``); one group holding both hemispheres is how his whole-brain ``correlation_spin_permutation_testing`` fits them (``modules/helpers.py``), and a node may then take a node of the other hemisphere.

    Args:
        sphere: Each node's position on a sphere centred at the origin, (N, 3), labelled on ``node`` or in network order, turned as given.
        node_labels: The network's node labels, in network order (``network.labels``).
        groups: The group of each node, labelled on ``node`` or in network order, or one label for all nodes; a node is reassigned only to nodes of its own group.
        n: The number of rotations, the rows of the table.
        seed: The seed of the rotations.

    Returns:
        The mapping :func:`node_permutations` returns, row r of ``index`` naming each node's source under the r-th rotation.

    Raises:
        ValueError: As :func:`node_permutations`, and a sphere without three coordinates for every node.
    """
    labels = list(_label_index(node_labels, "node_labels"))
    node_group, members = _resampling_groups(labels, groups, n, seed)
    points = np.asarray(_aligned(sphere, labels, "sphere"), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"sphere must hold three coordinates for each node, got shape {points.shape}.")
    rotations = _rotations(n, seed)
    index = np.empty((n, len(labels)), dtype=np.int64)
    for nodes in members.values():
        index[:, nodes] = nodes[_nearest(points[nodes], rotations)]
    return _table(index, node_group)


def _resampling_groups(labels: list, groups, n, seed) -> tuple[np.ndarray, dict]:
    """Each node's group label in the order of *labels*, and per group label the positions of its nodes in label order: the layout a resampling table of *n* rows drawn from *seed* is built on. A missing count or seed, a count below 1, a node without a group and a group of one node are refused."""
    if n is None or seed is None or n < 1:
        raise ValueError(
            f"A resampling table needs a declared seed and a declared number of rows n of at least 1; got n={n!r}, seed={seed!r}."
        )
    node_group = np.full(len(labels), str(groups)) if np.ndim(groups) == 0 else _aligned(groups, labels, "groups").astype(str)
    order = np.argsort(labels)
    members = {}
    for label in sorted(set(node_group.tolist())):
        nodes = order[node_group[order] == label]
        if len(nodes) < 2:
            raise ValueError(
                f"group {label!r} holds the one node {labels[nodes[0]]!r}, which has no other node to trade places with."
            )
        members[label] = nodes
    return node_group, members


def _rotations(n: int, seed: int) -> np.ndarray:
    """*n* rotations (n, 3, 3) acting on row vectors, ``points @ rotation``, drawn as BrainSpace's ``_generate_spins`` draws them: the Q factor of a standard normal matrix from ``RandomState(seed)`` with each column signed by R's diagonal entry, which makes it uniform over the orthogonal matrices, and its first column flipped where that leaves a reflection."""
    q, r = np.linalg.qr(np.random.RandomState(seed).normal(size=(n, 3, 3)))
    q *= np.sign(np.diagonal(r, axis1=1, axis2=2))[:, None, :]
    q[:, :, 0] *= np.sign(np.linalg.det(q))[:, None]
    return q


def _nearest(points: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    """(n, V): for each rotation, the row of *points* nearest to each turned row, ``points @ rotation``, as BrainSpace's ``cKDTree(points).query(points @ rotation, k=1)`` finds it; an exact tie goes to the first row."""
    from scipy.spatial.distance import cdist

    return np.stack([cdist(points @ rotation, points).argmin(axis=1) for rotation in rotations])


def _table(index: np.ndarray, node_group: np.ndarray) -> dict:
    """A resampling table's two outputs, the group labels as text the produced-constant store writes and reads back as text."""
    import h5py

    return {"index": index, "group": np.array(node_group.tolist(), dtype=h5py.string_dtype())}
