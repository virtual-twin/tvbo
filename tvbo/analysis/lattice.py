"""Node values on the regular lattice their positions form, the image a sheet model is drawn as."""

from __future__ import annotations

import numpy as np
import xarray as xr

from tvbo.analysis.statistics import _paired_along, _some

TOLERANCE = 1e-6
"""How far, as a fraction of a lattice's larger extent, a node may lie off its lattice line and two spacings between lines may differ, with the positions still read as a regular lattice; positions held in single precision, good to about 1e-7 of their size, pass."""


def lattice_field(values, positions, dim: str) -> dict:
    """*values* placed at the *positions* of their nodes on a regular, axis-aligned lattice, as an image whose rows run along y and columns along x.

    *positions* holds one row per node along *dim*, its x, y and optionally z in mm, as :meth:`tvbo.classes.network.Network.node_positions` returns them; a z is the same for every node, since a lattice lies in a plane. Labelled inputs are paired on the labels of *dim*, which occur once each and which both carry, and a bare *positions* is in the node order of *values* (:func:`_node_positions`). Coordinates along an axis that lie within :data:`TOLERANCE` of the lattice's larger extent of their neighbour are one lattice line, at the smallest of them, and the lines along each axis are evenly spaced to the same tolerance. Rows and columns ascend, so neither the order of the nodes nor that of their positions changes the image; for nodes in the order Koller et al. (2024) place their sheet in, x slowest, it is their ``values.reshape(nx, ny).T``. Every other dimension of *values* is kept with its labels.

    Returns:
        ``field``, on the other dimensions of *values* followed by ``y`` and ``x``, whose coordinates are the lattice lines in mm.

    Raises:
        ValueError: If the positions vary in z, lie off a regular lattice, put two nodes on one lattice point or leave a lattice point without a node, and as :func:`_node_positions` refuses them.
    """
    values, xyz = _node_positions(values, positions, dim)
    tolerance = TOLERANCE * np.ptp(xyz[:, :2], axis=0).max()
    if xyz.shape[1] == 3 and np.ptp(xyz[:, 2]) > tolerance:
        raise ValueError(f"`positions` vary in z by {np.ptp(xyz[:, 2]):g} mm; the nodes of a lattice lie in one plane.")
    x, column = _lattice_lines(xyz[:, 0], "x", tolerance)
    y, row = _lattice_lines(xyz[:, 1], "y", tolerance)
    cell = row * x.size + column
    points, counts = np.unique(cell, return_counts=True)
    if (counts > 1).any():
        shared = np.isin(cell, points[counts > 1])
        nodes = values.indexes.get(dim)
        named = list(nodes[shared]) if nodes is not None else np.flatnonzero(shared).tolist()
        raise ValueError(
            f"`positions` put the nodes {_some(named)} on lattice points another of them occupies too; each point holds one node."
        )
    empty = np.setdiff1d(np.arange(x.size * y.size), cell)
    if empty.size:
        raise ValueError(
            f"`positions` leave {empty.size} of the {y.size} x {x.size} lattice points without a node, the first at x = {x[empty[0] % x.size]:g} mm, y = {y[empty[0] // x.size]:g} mm; each point holds one."
        )
    placed = values.transpose(..., dim).isel({dim: np.argsort(cell)})
    kept = {name: coord for name, coord in placed.coords.items() if dim not in coord.dims}
    mm = {"units": "mm"}
    field = xr.DataArray(
        placed.values.reshape(*placed.shape[:-1], y.size, x.size),
        dims=[*placed.dims[:-1], "y", "x"],
        coords={**kept, "y": ("y", y, mm), "x": ("x", x, mm)},
    )
    return {"field": field}


def _node_positions(values, positions, dim: str) -> tuple:
    """*values* as a labelled array along *dim*, and the *positions* of its nodes as an ``(n_nodes, 2)`` or ``(n_nodes, 3)`` float array in the same node order.

    A bare *positions* is read in the node order of *values*. A labelled one runs along *dim* and one axis holding x, y and optionally z, labelled so or not at all, and is paired with *values* on the labels of *dim* (:func:`tvbo.analysis.statistics._paired_along`). Every coordinate is finite.
    """
    if not isinstance(positions, xr.DataArray):
        array = np.asarray(positions, dtype=float)
        if array.ndim != 2:
            raise ValueError(
                f"`positions` is a bare array of rank {array.ndim}; it holds one row per node, its x, y and optionally z."
            )
        positions = xr.DataArray(array, dims=[dim, "coordinate"])
    values, positions = _paired_along(values, positions, dim, ("values", "positions"))
    axes = [d for d in positions.dims if d != dim]
    if len(axes) != 1 or positions.sizes[axes[0]] not in (2, 3):
        raise ValueError(
            f"`positions` has the sizes {dict(positions.sizes)}; beside {dim!r} it has one axis holding x, y and optionally z."
        )
    labels = positions.indexes.get(axes[0])
    if labels is not None and list(labels) != ["x", "y", "z"][: len(labels)]:
        raise ValueError(f"`positions` labels its coordinates {list(labels)}; they are x, y and optionally z, in that order.")
    if positions.sizes[dim] != values.sizes[dim]:
        raise ValueError(
            f"`values` holds {values.sizes[dim]} nodes along {dim!r} and `positions` {positions.sizes[dim]}; each node has one position."
        )
    xyz = np.asarray(positions.transpose(dim, axes[0]).values, dtype=float)
    if not np.isfinite(xyz).all():
        raise ValueError("`positions` hold a NaN or infinite coordinate; each node has a place on the lattice.")
    return values, xyz


def _lattice_lines(coordinate: np.ndarray, axis: str, tolerance: float) -> tuple:
    """The ascending lattice lines the nodes' *coordinate* along *axis* lies on, and the index of each node's line.

    Coordinates that follow each other in sorted order no more than *tolerance* apart are one line, placed at the smallest of them, and the spacings between lines differ by at most *tolerance*.
    """
    order = np.argsort(coordinate)
    ordered = coordinate[order]
    new_line = np.r_[True, np.diff(ordered) > tolerance]
    line = np.empty(coordinate.size, dtype=int)
    line[order] = np.cumsum(new_line) - 1
    lines = ordered[new_line]
    spacing = np.diff(lines)
    if spacing.size and np.ptp(spacing) > tolerance:
        raise ValueError(
            f"`positions` do not lie on a regular lattice: their {lines.size} distinct {axis} coordinates are {spacing.min():g} to {spacing.max():g} mm apart, unequal by more than the tolerance of {tolerance:g} mm."
        )
    return lines, line
