"""Node values placed on the lattice their positions form, reached through the curated record ``LatticeField`` as a study reaches it.

``LatticeField`` returns ``field`` on the other dimensions of ``values`` followed by ``y`` and ``x``, rows and columns ascending and their coordinates in mm, whatever order the nodes come in. A known 3 x 4 lattice puts each value in the cell written out by hand in ``GRID_FIELD``, and on Koller et al.'s sheet the field is their ``values.reshape(nx, ny).T`` (doi:10.1038/s41467-024-47860-x; released ``scripts/results/01_results.ipynb`` cell 5, file l. 193, and every other sheet map of ``01_results.ipynb``, ``03_results.ipynb`` and ``06_results.ipynb``, on the node order of ``scripts/simulation/01_simulation.py`` l. 86-88; transcribed in :func:`_koller_sheet` and :func:`_koller_image`). Positions off a regular lattice, two nodes on one point, a point without a node, a z that varies and positions that cannot be paired with the values are refused by name.
"""

import numpy as np
import pytest
import xarray as xr

from tvbo.data import analysis_io
from tvbo.datamodel import Analysis

from .test_curated_statistics import _curated, _seed

GRID = np.array(
    [
        [4.0, 7.0, 22.0],
        [0.0, 1.0, 0.0],
        [6.0, 4.0, 13.0],
        [2.0, 7.0, 21.0],
        [0.0, 4.0, 10.0],
        [6.0, 1.0, 3.0],
        [2.0, 1.0, 1.0],
        [4.0, 4.0, 12.0],
        [0.0, 7.0, 20.0],
        [6.0, 7.0, 23.0],
        [2.0, 4.0, 11.0],
        [4.0, 1.0, 2.0],
    ]
)
"""Twelve nodes of a lattice of three rows, at y = 1, 4 and 7 mm, and four columns, at x = 0, 2, 4 and 6 mm, in no order: each row holds a node's x, y and value, the value 10 times the node's row plus its column."""

GRID_FIELD = [[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]]
"""The field of ``GRID``, one list per row from the smallest y, each from the smallest x."""

NODES = [f"n{i}" for i in range(12)]


@pytest.fixture
def rng():
    """A seeded generator, the same in every test."""
    return np.random.default_rng(0)


def _field(values, positions):
    """The ``field`` of ``LatticeField`` for *values* at *positions*, the nodes along ``node``."""
    return _curated("LatticeField")(values, positions, dim="node")["field"]


def _moved(node, x, y):
    """The positions of ``GRID`` with node *node* moved to *x*, *y*."""
    positions = GRID[:, :2].copy()
    positions[node] = x, y
    return positions


def _koller_sheet(nx, ny, x_extent, y_extent):
    """The node positions of Koller et al.'s sheet in their order, x slowest: ``scripts/simulation/01_simulation.py`` l. 86-88, as written there."""
    y = np.linspace(0, y_extent, ny)
    x = np.linspace(0, x_extent, nx)
    return np.array(np.meshgrid(x, y)).T.reshape(-1, 2)


def _koller_image(values, nx, ny):
    """A map of the sheet as Koller et al. draw it with ``imshow(..., origin='lower')``: ``values.reshape(nx, ny).T`` (``01_results.ipynb`` cell 5, file l. 193)."""
    return values.reshape(nx, ny).T


def test_a_known_three_by_four_lattice_puts_each_value_in_its_cell():
    """Twelve nodes listed in no order land each in the cell of its row and column, rows along y and columns along x."""
    field = _field(GRID[:, 2], GRID[:, :2])
    assert field.dims == ("y", "x")
    np.testing.assert_array_equal(field.values, GRID_FIELD)


def test_the_coordinates_are_the_lattice_lines_in_mm():
    """The y and x coordinates are the distinct positions, ascending, in mm, and the constant z a network's node positions carry changes nothing."""
    field = _field(GRID[:, 2], np.c_[GRID[:, :2], np.zeros(12)])
    assert field["y"].values.tolist() == [1.0, 4.0, 7.0] and field["x"].values.tolist() == [0.0, 2.0, 4.0, 6.0]
    assert field["y"].attrs["units"] == field["x"].attrs["units"] == "mm"
    np.testing.assert_array_equal(field.values, GRID_FIELD)


@pytest.mark.parametrize("labelled", [False, True], ids=["bare", "labelled"])
def test_shuffled_node_order_gives_the_same_field(labelled, rng):
    """The nodes in another order give the same field: bare values and positions shuffled together, or labelled ones each in an order of its own, paired by label."""
    want = _field(GRID[:, 2], GRID[:, :2])
    order = rng.permutation(12)
    values, positions = GRID[order, 2], GRID[order, :2]
    if labelled:
        other = rng.permutation(12)
        names = np.array(NODES)
        values = xr.DataArray(values, dims=["node"], coords={"node": names[order]})
        positions = xr.DataArray(GRID[other, :2], dims=["node", "coordinate"], coords={"node": names[other]})
    got = _field(values, positions)
    np.testing.assert_array_equal(got.values, want.values)
    np.testing.assert_array_equal(got["x"].values, want["x"].values)
    np.testing.assert_array_equal(got["y"].values, want["y"].values)


def test_every_other_dimension_is_kept_with_its_labels(rng):
    """Values on trial x node x time give a field on trial x time x y x x with the trial and time labels, each slice the field of that slice's values."""
    values = xr.DataArray(
        rng.normal(size=(2, 12, 3)),
        dims=("trial", "node", "time"),
        coords={"trial": ["a", "b"], "node": NODES, "time": [0.0, 5.0, 10.0]},
    )
    positions = xr.DataArray(GRID[:, :2], dims=["node", "coordinate"], coords={"node": NODES})
    field = _field(values, positions)
    assert field.dims == ("trial", "time", "y", "x")
    assert field["trial"].values.tolist() == ["a", "b"] and field["time"].values.tolist() == [0.0, 5.0, 10.0]
    for i, j in np.ndindex(2, 3):
        np.testing.assert_array_equal(field.values[i, j], _field(values.values[i, :, j], GRID[:, :2]).values)


@pytest.mark.parametrize(
    ("positions", "match"),
    [
        (
            _moved(2, 6.3, 4.0),
            "`positions` do not lie on a regular lattice: their 5 distinct x coordinates are 0.3 to 2 mm apart",
        ),
        (_moved(2, 4.0, 4.0), "`positions` put the nodes 'n2', 'n7' on lattice points another of them occupies too"),
        (GRID[1:, :2], "`positions` leave 1 of the 3 x 4 lattice points without a node, the first at x = 4 mm, y = 7 mm"),
    ],
    ids=["off_lattice", "shared_point", "empty_point"],
)
def test_a_lattice_that_is_not_regular_or_not_filled_one_node_per_point_is_refused_by_name(positions, match):
    """A node moved off its column, a node moved onto another's point and a node taken away are each refused, naming the spacings, the nodes or the empty point."""
    names = NODES[: len(positions)]
    values = xr.DataArray(np.arange(len(positions), dtype=float), dims=["node"], coords={"node": names})
    with pytest.raises(ValueError, match=match):
        _field(values, xr.DataArray(positions, dims=["node", "coordinate"], coords={"node": names}))


def test_positions_that_leave_the_plane_or_cannot_be_paired_with_the_values_are_refused():
    """A z that varies, positions that are not one row of x, y and optionally z per node, a NaN coordinate, a count that differs from the values' and a label only one side carries are refused by name."""
    values = GRID[:, 2]
    with pytest.raises(ValueError, match="`positions` vary in z by 0.5 mm; the nodes of a lattice lie in one plane"):
        _field(values, np.c_[GRID[:, :2], np.r_[0.5, np.zeros(11)]])
    with pytest.raises(ValueError, match="`positions` is a bare array of rank 1"):
        _field(values, GRID[:, 0])
    with pytest.raises(ValueError, match="beside 'node' it has one axis holding x, y and optionally z"):
        _field(values, np.c_[GRID, GRID])
    with pytest.raises(ValueError, match=r"`positions` labels its coordinates \['y', 'x'\]"):
        _field(values, xr.DataArray(GRID[:, 1::-1], dims=["node", "axis"], coords={"axis": ["y", "x"]}))
    with pytest.raises(ValueError, match="`positions` hold a NaN or infinite coordinate"):
        _field(values, _moved(3, np.nan, 7.0))
    with pytest.raises(ValueError, match="`values` holds 12 nodes along 'node' and `positions` 11"):
        _field(values, GRID[1:, :2])
    with pytest.raises(
        ValueError,
        match="`values` and `positions` do not carry the same labels along 'node': 'n11', 'p11' are in only one of them",
    ):
        _field(
            xr.DataArray(values, dims=["node"], coords={"node": NODES}),
            xr.DataArray(GRID[:, :2], dims=["node", "coordinate"], coords={"node": [*NODES[:11], "p11"]}),
        )


@pytest.mark.parametrize(
    ("nx", "ny", "x_extent", "y_extent"), [(30, 30, 140.0, 140.0), (7, 4, 140.0, 60.0)], ids=["published", "oblong"]
)
def test_the_field_is_kollers_reshape_transposed_on_his_node_order(nx, ny, x_extent, y_extent, rng):
    """On Koller et al.'s node order, for their 30 x 30 sheet of 140 mm and for an oblong one where a transposition would show, the field is their image, its coordinates their two linspaces, with the z of 0 a network adds."""
    positions = _koller_sheet(nx, ny, x_extent, y_extent)
    values = rng.normal(size=nx * ny)
    field = _field(values, np.c_[positions, np.zeros(nx * ny)])
    assert field.shape == (ny, nx)
    np.testing.assert_array_equal(field.values, _koller_image(values, nx, ny))
    np.testing.assert_array_equal(field["x"].values, np.linspace(0, x_extent, nx))
    np.testing.assert_array_equal(field["y"].values, np.linspace(0, y_extent, ny))


def test_it_runs_as_a_study_analysis_and_stores_the_field_with_its_coordinates(tmp_path, rng):
    """Named by ``iri`` and run on seeded containers, it stores ``observation__field`` on trial x y x x, with the x coordinate in mm, read back from the file."""
    values = xr.DataArray(rng.normal(size=(2, 12)), dims=("trial", "node"), coords={"trial": [0, 1], "node": NODES})
    positions = xr.DataArray(GRID[:, :2], dims=["node", "coordinate"], coords={"node": NODES})
    _seed(tmp_path, "maps", values)
    _seed(tmp_path, "sheet", positions)
    analysis = Analysis(
        name="maps_on_lattice",
        iri="tvbo:functions/LatticeField",
        arguments={
            "values": {"used": {"analysis": "maps", "output": "maps"}},
            "positions": {"used": {"analysis": "sheet", "output": "sheet"}},
            "dim": "node",
        },
    )
    analysis_io.run_analysis(analysis, tmp_path)

    want = _field(values, positions)
    with xr.open_dataset(analysis_io.container_path("maps_on_lattice", tmp_path), engine="h5netcdf") as stored:
        assert set(stored.data_vars) == {"observation__field"}
        assert stored["observation__field"].dims == ("trial", "y", "x")
        np.testing.assert_array_equal(stored["observation__field"].values, want.values)
        assert stored["x"].values.tolist() == [0.0, 2.0, 4.0, 6.0] and stored["x"].attrs["units"] == "mm"
