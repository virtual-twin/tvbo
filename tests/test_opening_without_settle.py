"""A tvboptim run reports the opening it integrates from, with a settle or without one.

``run_simulation`` returns ``opening``, the state a run opens on before any settle, which a sweep cell and the evaluation after tuning reopen from. With a settle that is the settle's opening, ``_tr_state``; without one it is the measured window's own ``state``, and the render must not read ``_tr_state``, which nothing binds then. The probe is the factored Kuramoto of ``test_codegen_style_contract`` with each node declaring its own phase and ``K = 0``, so every phase holds its opening and the measured window shows where the run opened, with or without a settle before it.
"""

from __future__ import annotations

import copy

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment

from .test_codegen_style_contract import KURAMOTO_FACTORED, _lint, _ruff

PHASES = (0.5, -0.5, 1.0, 2.0)
"""The phase each node declares, away from the declared ``initial_value`` of 0.0."""

SETTLES = pytest.mark.parametrize("settle", [0.0, 0.5], ids=["no-settle", "settle"])


def _spec(settle):
    """The factored Kuramoto at ``K = 0`` with each node opening at its own phase from :data:`PHASES` and a settle of *settle*."""
    spec = copy.deepcopy(KURAMOTO_FACTORED)
    spec["dynamics"]["parameters"]["K"]["value"] = 0.0
    spec["network"]["nodes"] = [
        {"id": i, "label": f"r{i}", "state": {"theta": {"value": phase}}} for i, phase in enumerate(PHASES)
    ]
    spec["integration"]["transient_time"] = settle
    return spec


@SETTLES
def test_the_rendered_run_binds_every_name_it_reads(settle):
    """The module, ``run_simulation``'s report of its opening included, reads no name it leaves unbound."""
    code = SimulationExperiment(**_spec(settle)).render_code(format="tvboptim")
    assert "F821" not in _lint(code, _ruff())


@SETTLES
def test_the_reported_opening_is_where_the_run_opens(settle):
    """``opening`` holds each node's declared phase, and the measured window, which ``K = 0`` keeps at its opening, holds the same phases from its first sample."""
    module = SimulationExperiment(**_spec(settle)).execute("tvboptim")
    network = module.create_network(jnp.ones((len(PHASES), len(PHASES))) - jnp.eye(len(PHASES)))
    sim = module.run_simulation(network)
    opening = np.asarray(sim.opening)
    np.testing.assert_array_equal(opening[module._STATE_INDEX["theta"]], np.asarray(PHASES))
    ys = np.asarray(sim.result.ys)
    np.testing.assert_array_equal(ys, np.broadcast_to(opening, ys.shape))
