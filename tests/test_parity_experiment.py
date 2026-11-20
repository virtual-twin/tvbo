"""``ThreeNode_JansenRit1995_Parity`` integrates the same trajectory on tvboptim, jax and tvb, as its description and the parity page say.

The experiment is deterministic by construction: every node declares its opening state, nothing is drawn, and the delays and the duration are whole numbers of steps. The docs run it on the three backends; this runs it too, each backend against tvboptim, rather than rendering it alone.
"""

from __future__ import annotations

import pytest

from tvbo import SimulationExperiment

from .test_tvb_coupling_incoming_states import assert_backends_agree

pytestmark = pytest.mark.backend_tvboptim


@pytest.mark.parametrize(
    "backend", [pytest.param("jax", marks=pytest.mark.backend_jax), pytest.param("tvb", marks=pytest.mark.backend_tvb)]
)
def test_the_parity_experiment_integrates_tvboptim_s_trajectory(backend):
    """The backend records every state of the three columns as tvboptim does, at every integration step both report."""
    assert_backends_agree(lambda: SimulationExperiment.from_db("ThreeNode_JansenRit1995_Parity"), backend)
