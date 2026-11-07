"""``--limit``, ``--shard`` and ``--pin`` address the cells of a sweep declared per element, run on tvboptim.

A heterogeneous parameter swept through ``element_domains`` spans one result dimension per element, named ``<parameter>[<element>]``. The run options that pick cells out of a sweep count and name those cells as the result container does: a limit is a sample in a directory of its own, a shard knows the size of the grid it slices, and one element can be held while the others sweep. The script that builds the sweep counts and slices it, so a sweep whose cells a builder supplies at run time is sampled as well. What is decided before a backend runs lives in ``test_sweep_planning_and_pins``.
"""

import itertools
import logging
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("tvboptim")
xr = pytest.importorskip("xarray")

from tests.test_sweep_planning_and_pins import BUILT, ELEMENT_0, ELEMENT_1, GRID, K0, K1, LABELLED, MODEL  # noqa: E402
from tvbo import SimulationExperiment  # noqa: E402
from tvbo.run import study as study_run  # noqa: E402

pytestmark = pytest.mark.backend_tvboptim


def _final_x(k: float) -> float:
    """The state after the eight Euler steps of the run, for decay rate *k*."""
    return 0.7 * (1 - 0.5 * k) ** 8


def _launch(out: Path, spec: str = GRID, **options) -> list[Path]:
    """Run *spec* on tvboptim under *options* with ``out/results`` as ``-o``, and return the containers in the directory the run resolves that to (a ``--limit`` sample's own)."""
    out.mkdir(parents=True, exist_ok=True)
    recipe = out / "recipe.yaml"
    recipe.write_text(spec, encoding="utf-8")
    run_options = study_run.RunOptions(backend="tvboptim", **options)
    study_run.run_experiment(SimulationExperiment.from_string(spec), str(recipe), out / "results", options=run_options)
    return sorted(study_run.results_root(str(recipe), out / "results", sample=run_options.sample).glob("*_result.h5"))


def _builder(root: Path, monkeypatch) -> None:
    """Make the module ``rate_builder`` importable from *root*, its ``rates(n)`` returning the ``n`` rates 0.1, 0.2 and so on."""
    (root / "rate_builder.py").write_text(
        "def rates(n):\n    return [0.1 * (i + 1) for i in range(int(n))]\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(root))
    sys.modules.pop("rate_builder", None)


def _run(out: Path, spec: str = GRID, **options):
    """Run *spec* on tvboptim under *options* and return the one container it wrote."""
    (container,) = _launch(out, spec, **options)
    with xr.open_dataset(container, engine="h5netcdf") as ds:
        return ds.load()


def _cells(result) -> int:
    """How many sweep cells a container holds, whether it stores them as a grid or as a list of points."""
    if "point" in result.sizes:
        return result.sizes["point"]
    return math.prod(n for dim, n in result.sizes.items() if str(dim).startswith("ElemDecay.k"))


def test_limit_runs_a_sample_of_an_element_grid(tmp_path):
    """``--limit 4`` on the 3 x 4 grid runs at most four of its 12 cells."""
    ran = _cells(_run(tmp_path, limit=4))
    assert 0 < ran <= 4, (
        f"--limit 4 ran {ran} of the grid's 12 cells: the limit is dropped when the element axes count as zero cells"
    )


def test_shard_reports_the_size_of_the_element_grid_it_slices(tmp_path, caplog):
    """``--shard 0/3`` names the 12 cells of the grid and the four its task runs."""
    with caplog.at_level(logging.INFO, logger=study_run.__name__):
        _run(tmp_path, shard=(0, 3))
    report = next((r.getMessage() for r in caplog.records if r.getMessage().startswith("sharding:")), "")
    counts = re.search(r"(\d+) of (\d+) sweep cells", report)
    assert counts is not None, f"the shard reported no cell count: {report!r}"
    assert counts.groups() == ("4", "12"), (
        f"the shard reported {counts.group(0)!r} for a task that runs 4 of a 3 x 4 grid's 12 cells"
    )


def test_the_shards_of_an_element_grid_run_every_cell_once(tmp_path):
    """Three shards of the 3 x 4 grid hold its 12 cells between them, none twice."""
    cells = []
    for task in range(3):
        share = _run(tmp_path / f"task{task}", shard=(task, 3))
        assert _cells(share) == 4
        cells += zip(share[ELEMENT_0].values.round(6).tolist(), share[ELEMENT_1].values.round(6).tolist(), strict=True)
    assert sorted(cells) == sorted(itertools.product(K0, K1))


def test_pin_holds_one_element_and_leaves_the_other_sweeping(tmp_path):
    """``--pin 'ElemDecay.k[0]=0.9'`` runs element 0 at 0.9 across the four values of element 1."""
    try:
        result = _run(tmp_path, pins=[(ELEMENT_0, 0.9)])
    except study_run.StudyRunError as refusal:
        pytest.fail(f"--pin refuses the element under the name its result dimension carries, {ELEMENT_0!r}: {refusal}")
    final = result["x_last"].squeeze()
    assert dict(final.sizes) == {ELEMENT_1: 4, "node": 2}, f"a run pinned on {ELEMENT_0!r} still spans {dict(final.sizes)}"
    np.testing.assert_allclose(
        final.isel(node=0), _final_x(0.9), rtol=1e-5, err_msg="element 0 did not run at its pinned rate"
    )
    np.testing.assert_allclose(
        final.isel(node=1), [_final_x(k) for k in K1], rtol=1e-5, err_msg="element 1 did not sweep its declared rates"
    )


def test_limit_samples_a_sweep_whose_cells_a_builder_supplies(tmp_path, monkeypatch, caplog):
    """A builder supplies its values at run time, and the script that builds the sweep counts them, so ``--limit 2`` samples two of its five cells and says so."""
    _builder(tmp_path, monkeypatch)
    with caplog.at_level(logging.INFO, logger=study_run.__name__):
        ran = _cells(_run(tmp_path, BUILT, limit=2))
    assert ran == 2, f"--limit 2 ran {ran} of the builder's five cells"
    report = next((r.getMessage() for r in caplog.records if r.getMessage().startswith("--limit")), "")
    assert re.search(r"ran 2 of 5 sweep cells", report), f"the limit reported {report!r}"


def test_pin_names_an_element_by_its_node_label(tmp_path):
    """``--pin 'ElemDecay.k[V1]=0.9'`` pins the element of node V1, the same cells as its index 0."""
    final = _run(tmp_path, LABELLED, pins=[("ElemDecay.k[V1]", 0.9)])["x_last"].squeeze()
    assert dict(final.sizes) == {ELEMENT_1: 4, "node": 2}, f"a run pinned on node V1 still spans {dict(final.sizes)}"
    np.testing.assert_allclose(final.isel(node=0), _final_x(0.9), rtol=1e-5)
    np.testing.assert_allclose(final.isel(node=1), [_final_x(k) for k in K1], rtol=1e-5)


def test_pin_at_the_shared_value_keeps_a_parameter_its_nodes_make_per_node(tmp_path):
    """``k`` is per node only because node 0 declares 0.3; pinning node 0 at the shared 0.5 leaves no node differing, and element 1 still sweeps."""
    spec = GRID.replace('{value: 0.5, heterogeneous: true, shape: "(n_nodes,)"}', "{value: 0.5}").replace(
        "  number_of_nodes: 2\n",
        "  number_of_nodes: 2\n  nodes:\n    - {id: 0, parameters: {k: {value: 0.3}}}\n    - {id: 1}\n",
    )
    final = _run(tmp_path, spec, pins=[(ELEMENT_0, 0.5)])["x_last"].squeeze()
    assert dict(final.sizes) == {ELEMENT_1: 4, "node": 2}, f"a run pinned at the shared value spans {dict(final.sizes)}"
    np.testing.assert_allclose(final.isel(node=0), _final_x(0.5), rtol=1e-5)
    np.testing.assert_allclose(final.isel(node=1), [_final_x(k) for k in K1], rtol=1e-5)


@pytest.mark.parametrize(
    ("n_cells", "shard", "limit", "cells"),
    [
        (12, None, 4, [0, 3, 6, 9]),
        (5, None, 2, [0, 3]),
        (12, (1, 3), None, [1, 4, 7, 10]),
        (12, (1, 3), 2, [1, 7]),
        (3, None, 10, [0, 1, 2]),
    ],
)
def test_a_sweep_slice_takes_the_shard_then_an_even_sample_of_it(n_cells, shard, limit, cells):
    """One slice covers a shard, a ``--limit`` sample, and a sample of a shard, never more cells than the limit."""
    from tvbo.templates.tvboptim.callbacks import sweep_slice

    assert list(range(n_cells))[sweep_slice(n_cells, shard, limit)] == cells


def test_an_element_without_a_domain_keeps_its_declared_value(tmp_path):
    """An axis that declares a domain for element 1 only sweeps element 1, and element 0 runs at the parameter's declared value."""
    spec = MODEL.format(
        parameter='{value: 0.5, heterogeneous: true, shape: "(n_nodes,)"}',
        axis=f"        element_domains:\n          - {{element: 1, explored_values: {list(K1)}}}",
    )
    final = _run(tmp_path, spec)["x_last"].squeeze()
    assert dict(final.sizes) == {ELEMENT_1: 4, "node": 2}, f"a sweep of element 1 alone spans {dict(final.sizes)}"
    np.testing.assert_allclose(final.isel(node=0), _final_x(0.5), rtol=1e-5)
    np.testing.assert_allclose(final.isel(node=1), [_final_x(k) for k in K1], rtol=1e-5)


@pytest.mark.parametrize(("limit", "split"), [(4, True), (12, False)])
def test_a_limited_run_is_named_a_slice_only_when_it_sampled(tmp_path, limit, split):
    """``--limit 4`` on the 12-cell grid writes ``split-0000``; ``--limit 12`` runs the whole grid and is named as the whole run. Either way it lands in the sample's own directory and leaves the results directory empty."""
    (container,) = _launch(tmp_path, limit=limit)
    assert ("_split-0000_" in container.name) is split, f"--limit {limit} wrote {container.name}"
    assert container.parent.name == f"results-limit-{limit}"
    assert not list((tmp_path / "results").glob("*_result.h5")), "a sample was written beside the whole run's containers"


def test_the_tasks_past_the_last_cell_of_a_built_sweep_write_nothing_and_the_gather_holds_it(tmp_path, monkeypatch):
    """Three tasks over a two-cell builder sweep: the third holds no cell, succeeds and writes no container, and the two containers reassemble into the whole sweep."""
    from tvbo.data.types import reassemble_experiment_results

    _builder(tmp_path, monkeypatch)
    spec = BUILT.replace("{n: {value: 5}}", "{n: {value: 2}}")
    written = [len(_launch(tmp_path, spec, shard=(task, 3))) for task in range(3)]
    assert written == [1, 2, 2], f"the three tasks left {written} containers behind them"
    (merged,) = reassemble_experiment_results(str(tmp_path / "results"), str(tmp_path / "merged"))
    with xr.open_dataset(merged, engine="h5netcdf") as gathered:
        np.testing.assert_allclose(sorted(gathered["ElemDecay.k"].values), [0.1, 0.2])
