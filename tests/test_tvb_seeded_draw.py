"""A tvb run opens on the draw its seed names: a distribution's own ``seed``, else ``execution.random_seed``.

The initial state of a variable that declares a ``distribution`` was drawn from numpy's global generator, seeded once when tvbo was imported, so the same recipe opened on another draw every time it was run in one process, and neither seed reached it. It now draws from the seed ``resolve_distribution`` gives it, on a generator keyed also by the variable's position among those that declare a distribution, as tvboptim folds that position into its key.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("tvb")

from tvbo import SimulationExperiment

pytestmark = pytest.mark.backend_tvb

N_NODES = 4
LO, HI = 1.0, 2.0
RTOL = 1e-7
"""TVB may hold the opening in a float32 array."""

DRAWN = {
    "id": 1,
    "label": "drawn opening",
    "dynamics": {
        "name": "Drawn",
        "parameters": {"k": {"value": 0.0}},
        "state_variables": {
            name: {
                "equation": {"rhs": f"k * {name}"},
                "initial_value": 0.0,
                "distribution": {"name": "Uniform", "domain": {"lo": LO, "hi": HI}},
            }
            for name in ("x", "y")
        },
    },
    "network": {"number_of_nodes": N_NODES, "nodes": [{"id": i, "label": f"r{i}"} for i in range(N_NODES)]},
    "integration": {"method": "Euler", "step_size": 1.0, "duration": 4.0},
}
"""Four uncoupled nodes whose ``x`` and ``y`` both draw their opening from Uniform(1, 2) and hold it."""


def _opening(run_seed, own_seed=None):
    """The ``(variable, node)`` state a tvb run of the recipe opens on, under *run_seed* and, if given, a distribution ``seed`` of *own_seed* on both variables."""
    spec = copy.deepcopy(DRAWN)
    spec["execution"] = {"random_seed": run_seed}
    if own_seed is not None:
        for variable in spec["dynamics"]["state_variables"].values():
            variable["distribution"]["seed"] = own_seed
    return np.asarray(SimulationExperiment(**spec).execute("tvb").current_state)[:, :, 0]


def test_the_same_seed_opens_a_tvb_run_on_the_same_draw():
    """Run twice in one process, the recipe opens on one draw: the run's seed with each variable's position, ``x`` first."""
    first, second = _opening(7), _opening(7)
    np.testing.assert_array_equal(first, second)
    expected = [np.random.default_rng((7, position)).uniform(LO, HI, size=N_NODES) for position in range(2)]
    np.testing.assert_allclose(first, expected, rtol=RTOL)


def test_another_seed_opens_it_on_another_draw():
    assert not np.allclose(_opening(7), _opening(8))


def test_a_distribution_s_own_seed_overrides_the_run_s():
    """Under one own seed the run's seed does not reach the draw."""
    np.testing.assert_array_equal(_opening(7, own_seed=3), _opening(8, own_seed=3))
    expected = [np.random.default_rng((3, position)).uniform(LO, HI, size=N_NODES) for position in range(2)]
    np.testing.assert_allclose(_opening(7, own_seed=3), expected, rtol=RTOL)


def test_two_variables_under_one_seed_draw_independently():
    x, y = _opening(7)
    assert not np.allclose(x, y)
