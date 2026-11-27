"""An array-valued sweep axis whose declared points repeat keeps every cell, run on tvboptim.

A control-mask sweep declares one per-node vector per (strategy, count), and strategies that select the same nodes declare the same vector more than once: every strategy's zero-count mask is the uncontrolled run. A repeated point is the same simulation, so its cells are dealt out over the repeats, once per combination of the other swept axes, instead of all landing on the first repeat and colliding there.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

from tvbo.templates.tvboptim.callbacks import point_indices

ZERO, ONE, TWO = np.zeros(3), np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0])
POINTS = np.stack([ZERO, ONE, ZERO, TWO, ZERO])


def test_cells_of_a_repeated_point_are_dealt_over_its_repeats():
    """Five cells in declared order take the five declared indices, the three zero masks included."""
    assert list(point_indices(list(POINTS), POINTS)) == [0, 1, 2, 3, 4]


def test_scrambled_cells_still_land_one_per_declared_point():
    """Cells arriving in any order name each declared point once, and each lands on a point equal to its own."""
    cells = [TWO, ZERO, ONE, ZERO, ZERO]
    idx = point_indices(cells, POINTS)
    assert sorted(idx) == [0, 1, 2, 3, 4]
    for cell, i in zip(cells, idx, strict=True):
        np.testing.assert_array_equal(POINTS[i], cell)


def test_repeats_are_dealt_once_per_combination_of_the_other_axes():
    """Two zero masks against two values of another axis, in an arrival order that a plain round-robin over all cells would collide on."""
    points = np.stack([ZERO, ZERO])
    cells = [ZERO, ZERO, ZERO, ZERO]
    other = np.array([0.5, 0.5, 0.9, 0.9])
    idx = point_indices(cells, points, others=[other])
    assert sorted(zip(idx.tolist(), other.tolist(), strict=True)) == [(0, 0.5), (0, 0.9), (1, 0.5), (1, 0.9)]


def test_two_axes_with_repeated_points_split_each_others_cells():
    """Two crossed mask axes, each declaring the zero mask twice: the second axis reads the first one's indices, so the four cells take the four index pairs."""
    points = np.stack([ZERO, ZERO])
    cells = np.empty(4, dtype=object)
    cells[:] = [ZERO] * 4
    first = point_indices(cells, points, others=[cells])
    second = point_indices(cells, points, others=[first])
    assert sorted(zip(first.tolist(), second.tolist(), strict=True)) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_a_cell_between_two_distinct_points_is_not_dealt():
    """Only copies of one declared point share cells: two cells equally far from two different points both keep the first."""
    midway = np.array([0.5, 0.0, 0.0])
    assert list(point_indices([midway, midway], np.stack([ZERO, ONE]))) == [0, 0]


def test_distinct_points_still_match_by_nearest_value():
    """A point that round-tripped through float32 is matched to the declared point it names."""
    cells = [p.astype(np.float32) + 1e-7 for p in (TWO, ONE)]
    assert list(point_indices(cells, np.stack([ZERO, ONE, TWO]))) == [2, 1]


xr = pytest.importorskip("xarray")

SPEC = """
id: 1
label: Per-node rate masks
dynamics:
  name: MaskDecay
  parameters:
    k: {value: 0.0, heterogeneous: true, shape: "(n_nodes,)"}
    c: {value: 0.0}
  state_variables:
    x: {equation: {rhs: "-k*x + c"}, initial_value: 0.7}
network:
  label: Trio
  number_of_nodes: 3
integration: {method: euler, step_size: 0.5, duration: 4.0, unit: ms}
observations:
  x_last:
    label: Final x
    source: [x]
    aggregation: last
explorations:
  masks:
    label: Rate mask x drive
    mode: product
    space:
      - parameter: MaskDecay.k
        builder:
          callable: {name: masks, module: mask_builder}
      - parameter: MaskDecay.c
        explored_values: [0.0, 0.2]
"""


def _final_x(k: np.ndarray, c: float) -> np.ndarray:
    """The state after the eight Euler steps of the run, per node."""
    x = np.full_like(k, 0.7)
    for _ in range(8):
        x = x + 0.5 * (-k * x + c)
    return x


def _run(tmp_path, monkeypatch, **options) -> list[Path]:
    """Run the mask sweep on tvboptim under *options* and return the containers it has written so far."""
    pytest.importorskip("jax")
    pytest.importorskip("tvboptim")
    from tvbo import SimulationExperiment
    from tvbo.run import study as study_run

    (tmp_path / "mask_builder.py").write_text(
        "import numpy as np\n\ndef masks():\n    return np.array(" + repr((0.5 * POINTS).tolist()) + ")\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "mask_builder", raising=False)
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(SPEC, encoding="utf-8")
    run_options = study_run.RunOptions(backend="tvboptim", **options)
    study_run.run_experiment(SimulationExperiment.from_string(SPEC), str(recipe), tmp_path / "results", options=run_options)
    return sorted(Path(study_run.results_root(str(recipe), tmp_path / "results")).glob("*_result.h5"))


@pytest.mark.backend_tvboptim
def test_a_mask_sweep_with_repeated_masks_runs_every_cell(tmp_path, monkeypatch):
    """Five masks, three of them all-zero, times two drives: the container holds all ten cells, each at its own mask's final state."""
    (container,) = _run(tmp_path, monkeypatch)
    with xr.open_dataset(container, engine="h5netcdf") as ds:
        final = ds["x_last"].load().squeeze()
    assert dict(final.sizes) == {"MaskDecay.k": 5, "MaskDecay.c": 2, "node": 3}, dict(final.sizes)
    for i, mask in enumerate(0.5 * POINTS):
        for c in (0.0, 0.2):
            np.testing.assert_allclose(final.isel({"MaskDecay.k": i}).sel({"MaskDecay.c": c}), _final_x(mask, c), rtol=1e-5)


@pytest.mark.backend_tvboptim
def test_the_shards_of_a_mask_sweep_name_every_cell_once(tmp_path, monkeypatch):
    """Three shards of the ten cells: each shard deals the zero masks as the whole sweep does, so together they name the ten (mask, drive) cells once each."""
    named = []
    for i in range(3):
        containers = _run(tmp_path, monkeypatch, shard=(i, 3))
    for container in containers:
        with xr.open_dataset(container, engine="h5netcdf") as ds:
            named += zip(ds["MaskDecay.k"].values.tolist(), ds["MaskDecay.c"].values.round(6).tolist(), strict=True)
    assert sorted(named) == [(k, c) for k in range(5) for c in (0.0, 0.2)]
