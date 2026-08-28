"""A posterior belongs in the result container, like every other kind of result.

Without it a page has to re-run the sampler to redraw its own figure, which is how a Bayesian result ends up plotted by hand instead of declared.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tvbo.data.types import ExperimentResult, InferenceResult


@pytest.fixture
def container(tmp_path):
    xr = pytest.importorskip("xarray")
    source = SimpleNamespace(network=SimpleNamespace(node_labels=["n0"]), dynamics=None, coupling=None)
    inference = InferenceResult(
        name="scenario_A",
        posterior={"stimulus.amplitude": np.linspace(0.3, 0.5, 8), "Generic2dOscillator.I": np.linspace(0.05, 0.15, 8)},
        diagnostics={"stimulus.amplitude": {"mean": 0.4, "r_hat": 1.01}},
    )
    result = ExperimentResult(inferences={"scenario_A": inference}, source=source)
    written = result.save(str(tmp_path), compress=False, record_only=False)
    with xr.open_dataset([p for p in written if p.endswith(".h5")][0], engine="h5netcdf") as ds:
        yield ds.load()


def test_the_posterior_reaches_the_container(container):
    """The samples themselves, so a figure can bind them with `used:` like any other output."""
    samples = container["inference__scenario_A__posterior__stimulus.amplitude"]
    assert samples.shape == (8,)
    assert float(samples[0]) == pytest.approx(0.3)


def test_diagnostics_ride_along_with_the_samples(container):
    """A posterior read without its r_hat is a number with no standing."""
    assert float(container["inference__scenario_A__diagnostics__stimulus.amplitude__r_hat"]) == pytest.approx(1.01)


def test_the_tree_reads_it_at_the_path_the_recipe_declares(container):
    """`inferences:` is a recipe section, so the tree spells it the way the recipe does."""
    from tvbo.data.experiment_result_io import result_tree

    tree = result_tree(container)
    assert "inferences" in tree.children
    posterior = tree.inferences.scenario_A.posterior
    assert float(posterior["stimulus.amplitude"][0]) == pytest.approx(0.3)


def test_every_parameter_of_a_posterior_shares_the_draw_axis(container):
    """A joint plot of two marginals is a scatter only if they sit on ONE axis; unnamed positional axes make it an outer product."""
    amplitude = container["inference__scenario_A__posterior__stimulus.amplitude"]
    excitability = container["inference__scenario_A__posterior__Generic2dOscillator.I"]
    assert amplitude.dims == ("draw",)
    assert excitability.dims == amplitude.dims
