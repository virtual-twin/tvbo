"""A settle no longer than the longest delay pads the measured window's delay lines with its opening, whichever way the settle is handed over.

A solve records the state after each of its steps, never the opening it starts from, and tvboptim pads a stored history that covers less than the longest delay with its first sample. The base run hands its settle to the measured window through that stored history (``network.update_history``, route A); a swept cell and the evaluation after tuning carry their own settle onto a buffer filled with the opening (``_own_settle``, route B). The probe is the one of ``test_delay_history_from_opening``: its delay is 4 ms at dt = 1 ms and its buffer holds 6 rows, the 5 the delay spans and the older one an interpolating coupling reads, so a settle of at most 5 steps leaves rows older than its first recorded sample, and at ``k = -0.1`` the settle's first step moves ``x`` by a tenth, so such a row tells the opening from that sample.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tvboptim")

from .test_delay_history_from_opening import DELAY, DT, _buffers, _built, _drawn


def _settled(settle, k=-0.1, sigma=None):
    """The drawn probe with a settle of *settle* ms, ``x`` changing at rate *k* under additive noise of *sigma*, and a one-cell sweep so the script carries ``_own_settle``."""
    spec = _drawn()
    spec["dynamics"]["parameters"]["k"]["value"] = k
    spec["integration"]["transient_time"] = settle
    spec["explorations"] = {
        "sweep": {"name": "sweep", "mode": "product", "space": [{"parameter": "Probe.k", "explored_values": [k]}]}
    }
    if sigma is not None:
        spec["dynamics"]["state_variables"]["x"]["noise"] = {
            "additive": True,
            "gaussian": True,
            "parameters": {"sigma": {"value": sigma}},
        }
    return spec


def _run(spec):
    """The generated module of *spec*, its base run without the measured window, and the opening's ``x`` as a ``(1, node)`` row."""
    module, network = _built(spec)
    sim = module.run_simulation(network, run_main=False)
    return module, network, sim, np.asarray(sim.opening)[[module._STATE_INDEX["x"]]]


def _routes(spec):
    """The opening's ``x`` and the measured window's delay buffers after the base run's settle (route A) and after ``_own_settle`` from the same opening (route B)."""
    module, network, sim, x = _run(spec)
    cell = sim.state.copy()
    cell.initial_state.dynamics = sim.opening
    settled, _ = module._own_settle(network, sim.result_transient)(cell)
    route_a, route_b = _buffers(sim.state), _buffers(settled)
    assert route_a, "the probe must run a delayed coupling"
    return x, route_a, route_b


def _assert_padded_with(buffer, x, recorded):
    """Every row of *buffer* older than its *recorded* settle samples holds *x*, and the first recorded sample is not *x*, so the comparison means something."""
    pad = buffer.shape[0] - recorded
    assert pad >= 1
    np.testing.assert_array_equal(buffer[:pad], np.broadcast_to(x, buffer[:pad].shape))
    assert np.abs(buffer[pad] - x).max() > 1e-3, "the settle's first step must move x away from the opening"


@pytest.mark.parametrize("settle", [2.0, DELAY, DELAY + DT], ids=["shorter", "equal", "one-step-longer"])
def test_a_settle_shorter_than_the_buffer_pads_with_its_opening(settle):
    """The base run's window reads the delay lines a cell's own settle gives it: the opening in every row older than the settle's first recorded step, then the settle's steps.

    The buffer holds one row more than the delay spans, the oldest one an interpolating coupling reads, so a settle as long as the delay and one a step longer still leave rows older than their first recorded step, and those rows are the opening too.
    """
    x, route_a, route_b = _routes(_settled(settle))
    for a, b in zip(route_a, route_b, strict=True):
        _assert_padded_with(a, x, int(settle / DT))
        _assert_padded_with(b, x, int(settle / DT))
        np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("settle", [DELAY + 2 * DT, 3 * DELAY], ids=["fills-the-buffer", "longer"])
def test_a_settle_that_fills_the_buffer_hands_over_the_rows_it_did(settle):
    """A settle with a recorded step for every row of the buffer never reaches back to its opening, so the window's delay lines are tvboptim's own cut of the settle as the solve recorded it, bit for bit.

    That cut is the hand-over without the opening, rebuilt here rather than stored: a network whose stored history is the settle as recorded, read through tvboptim's ``get_history``.
    """
    module, network, sim, x = _run(_settled(settle))
    recorded = module._rebuilt_on(network, network.graph, history=sim.result_transient)
    before = np.asarray(recorded.get_history(DT))[:, [module._STATE_INDEX["x"]], :]
    buffers = _buffers(sim.state)
    assert buffers, "the probe must run a delayed coupling"
    for buffer in buffers:
        assert buffer.shape[0] <= int(settle / DT), "the settle must record a step for every row of the buffer"
        assert np.abs(buffer - x).min() > 1e-3, "the opening must be in no row for the guard to mean anything"
        np.testing.assert_array_equal(buffer, before)


def test_the_padded_rows_of_a_noisy_settle_carry_no_noise_increment():
    """Under noise the settle's first recorded sample is the opening plus one noise increment, and the rows older than it hold the opening itself on both routes. ``x`` has no drift here, so only the noise moves it."""
    x, route_a, route_b = _routes(_settled(2.0, k=0.0, sigma=0.3))
    for buffer in route_a + route_b:
        _assert_padded_with(buffer, x, 2)
