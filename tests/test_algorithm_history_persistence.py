"""An algorithm's tracked observations belong in the container, per iteration.

`algorithms.<name>.observations:` is the recipe declaring what the loop should TRACK, and `AlgorithmResult.history` is what it tracked. Persisting only `post_tuning` left a convergence panel with nothing to bind: the endpoint was in the artifact and the approach to it was not.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tvbo.data.experiment_result_io import result_tree
from tvbo.data.types import AlgorithmResult, ExperimentResult
from tvbo.utils import Bunch

LABELS = ["n0", "n1", "n2"]


@pytest.fixture
def container(tmp_path):
    xr = pytest.importorskip("xarray")
    observations = {"mean_S_e": SimpleNamespace(name="mean_S_e", dims=None, reduce=None, pipeline=None, source=None)}
    source = SimpleNamespace(
        network=SimpleNamespace(node_labels=LABELS), dynamics=None, coupling=None, observations=observations
    )
    algo = AlgorithmResult(
        name="fic",
        history=Bunch(
            mean_S_e=np.linspace(0.4, 0.25, 6),
            J_i=np.zeros((6, 3)),
            wLRE=np.zeros((6, 3, 3)),
        ),
    )
    result = ExperimentResult(algorithms={"fic": algo}, source=source)
    written = result.save(str(tmp_path), compress=False, record_only=False)
    with xr.open_dataset([p for p in written if p.endswith(".h5")][0], engine="h5netcdf") as ds:
        yield ds.load()


def test_a_tracked_scalar_is_recorded_per_iteration(container):
    """The convergence curve itself — one value per recorded iteration."""
    track = container["algorithm__fic__history__mean_S_e"]
    assert track.dims == ("iteration",)
    assert float(track[0]) == pytest.approx(0.4)


def test_a_tracked_per_node_parameter_keeps_its_node_axis(container):
    """An update rule writes one value per region, so the record is (iteration, node) — the convention `estimate__` uses beside it."""
    assert container["algorithm__fic__history__J_i"].dims == ("iteration", "node")


def test_a_tracked_per_edge_parameter_keeps_both_node_axes(container):
    assert container["algorithm__fic__history__wLRE"].dims == ("iteration", "node_i", "node_j")


def test_the_tree_reads_it_below_the_algorithm_it_belongs_to(container):
    """`history` nests under the algorithm, one level below the post-tuning outputs it converges to."""
    tree = result_tree(container)
    assert float(tree.algorithms.fic.history["mean_S_e"][0]) == pytest.approx(0.4)
