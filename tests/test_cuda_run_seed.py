"""The CUDA kernel seeds its noise from the run's seed, not from the GPU clock, so a noisy run is reproducible from its recipe.

`curand_init` took ``id + clock64()``, the cycle counter at launch, so two runs of one recipe drew different noise and nothing in the recipe could repeat one. The kernel now seeds every work item from `execution.random_seed` on a subsequence of its own, and a launch that resumes at ``i_step``, as the PyCUDA driver's chunked launches do, starts past every draw the launches before it could have made. The run test needs a GPU and is skipped without one.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from tvbo import SimulationExperiment

from .test_initial_state_distribution import ONE_VARIABLE

NOISE = {"parameters": {"sigma": {"value": 0.01}}}
RECIPE = "tvbo/database/experiments/RateML_ReducedWongWang.yaml"


def _noisy_kernel(seed):
    """The CUDA kernel of G240's one-variable recipe on one node, with additive noise, on *seed*."""
    spec = copy.deepcopy(ONE_VARIABLE)
    del spec["network"]
    spec["execution"] = {"random_seed": seed}
    spec["integration"]["noise"] = NOISE
    return SimulationExperiment(**spec).render_code("cuda")


def test_every_work_item_draws_from_the_run_seed():
    """The seed is the declared one, each work item its own subsequence, and no clock is read."""
    source = _noisy_kernel(11)
    assert "clock64" not in source
    assert re.search(r"curand_init\(\s*11ULL,\s*id,", source)


def test_a_resumed_launch_starts_past_the_draws_before_it():
    """A launch at ``i_step`` skips two draws per state variable, node and earlier step, so chunked launches never repeat a stream."""
    assert re.search(r"curand_init\(\s*11ULL,\s*id,\s*2ULL \* i_step \* n_node \* 1,\s*&crndst\)", _noisy_kernel(11))


def _gpu():
    """Whether a CUDA device is present, through PyCUDA."""
    driver = pytest.importorskip("pycuda.driver")
    try:
        driver.init()
        return driver.Device.count() > 0
    except driver.Error:
        return False


def _noisy_run(seed):
    """The time average the curated RateML recipe's kernel returns with additive noise, on *seed*."""
    from tvbo.codegen.cuda import run_cuda

    spec = yaml.safe_load(Path(RECIPE).read_text())
    spec["integration"]["noise"] = NOISE
    spec["execution"] = {"random_seed": seed}
    return run_cuda(SimulationExperiment(**spec))["tavg"]


def test_two_runs_on_one_seed_draw_the_same_noise():
    """On a GPU two runs on one seed agree exactly and a run on another seed does not."""
    if not _gpu():
        pytest.skip("no CUDA device")
    first = _noisy_run(5)
    np.testing.assert_array_equal(first, _noisy_run(5))
    assert not np.array_equal(first, _noisy_run(6))
