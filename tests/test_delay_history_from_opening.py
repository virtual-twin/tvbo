"""A delayed run reads its own opening over its first max-delay window, as TVB does.

tvboptim's ``prepare`` fills a delayed coupling's history buffer from the dynamics' ``INITIAL_STATE`` on every node, and the generated script writes the opening (an initial-condition draw, a node's declared state, a trial's draw) into the current state alone, so for its first max-delay window such a run coupled the default states. TVB, as tvbo runs it, fills the history with the opening repeated. The probe here reads that window in closed form: ``x`` holds its opening (``dx/dt = k x`` at ``k = 0``) and is what a linear coupling sends with a 4 ms delay, and ``y`` integrates what arrives (``dy/dt = c``), so with the history filled by the opening ``y(t) = (settle + t) a W x(0)`` from the first step on, on every backend, while a buffer of defaults bends the first 4 ms towards ``a W x_default``.
"""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("tvboptim")

from tvboptim.experimental.network_dynamics import Network, prepare
from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph
from tvboptim.experimental.network_dynamics.result import NativeSolution
from tvboptim.experimental.network_dynamics.solvers import Euler

from tvbo import SimulationExperiment

from .test_tvboptim_delay_history_carry import COUPLINGS, NS, TwoChannelEI

A, DELAY, DT, DURATION, DEFAULT_X = 1.0, 4.0, 1.0, 12.0, 0.1
W = np.array([[0.0, 0.5], [0.5, 0.0]])
"""The probe's weights, symmetric so the weight a node reads from the other does not depend on the orientation convention."""

PROBE = {
    "id": 1,
    "label": "delayed probe",
    "dynamics": {
        "name": "Probe",
        "parameters": {"k": {"value": 0.0}},
        "coupling_inputs": {"c": {"name": "c"}},
        "state_variables": {
            "x": {
                "equation": {"rhs": "k * x"},
                "initial_value": DEFAULT_X,
                "variable_of_interest": True,
                "coupling_variable": True,
            },
            "y": {"equation": {"rhs": "c"}, "initial_value": 0.0, "variable_of_interest": True},
        },
    },
    "network": {
        "number_of_nodes": 2,
        "nodes": [{"id": 0, "label": "r0"}, {"id": 1, "label": "r1"}],
        "edges": [
            {
                "source": i,
                "target": j,
                "directed": True,
                "parameters": {"weight": {"name": "weight", "value": 0.5}, "length": {"name": "length", "value": DELAY}},
            }
            for i, j in ((0, 1), (1, 0))
        ],
        "parameters": {"conduction_speed": {"name": "conduction_speed", "value": 1.0}},
        "coupling": {
            "LinearCoupling": {
                "delayed": True,
                "parameters": {"a": {"value": A}},
                "pre_expression": {"rhs": "x_j"},
                "post_expression": {"rhs": "a * gx"},
                "incoming_states": ["x"],
                "local_states": ["x"],
            }
        },
    },
    "execution": {"random_seed": 0, "precision": "float64"},
    "integration": {"method": "Euler", "step_size": DT, "duration": DURATION, "transient_time": 0.0},
}
"""Two nodes whose ``x`` opens at the declared 0.1 and is sent to the other node over a 4 ms tract at 1 mm/ms."""

BACKENDS = [
    pytest.param("tvboptim", marks=pytest.mark.backend_tvboptim),
    pytest.param("tvb", marks=pytest.mark.backend_tvb),
]

RTOL = {"tvboptim": 1e-8, "tvb": 1e-7}
"""The closed form's tolerance per backend: the tvboptim run's float64, and on tvb the float32 of TVB's own delay buffer (``NDArray(..., 'f')`` in ``tvb/simulator/history.py``), which rounds what the coupling reads to about 6e-8."""


def _drawn(spec=PROBE):
    """*spec* with ``x`` drawn per node from [1, 2], away from its declared 0.1."""
    spec = copy.deepcopy(spec)
    spec["dynamics"]["state_variables"]["x"]["distribution"] = {"name": "Uniform", "domain": {"lo": 1.0, "hi": 2.0}}
    return spec


def _declared_per_node(spec=PROBE):
    """*spec* with each node declaring its own ``x``."""
    spec = copy.deepcopy(spec)
    for node, x in zip(spec["network"]["nodes"], (1.5, -0.5), strict=True):
        node["state"] = {"x": {"value": x}}
    return spec


def _trace(spec, backend="tvboptim"):
    """*spec*'s measured window on *backend* as ``(time, variable, node)``."""
    pytest.importorskip(backend)
    trace = SimulationExperiment(**copy.deepcopy(spec)).run(backend).integration
    if "mode" in trace.dims:
        trace = trace.isel(mode=0)
    return trace.transpose("time", "variable", "node")


def _assert_reads_its_opening(x, y, t, settle=0.0, rtol=RTOL["tvboptim"]):
    """``x`` holds one opening per node, away from the default, and ``y`` is what a history filled with that opening delivers from the first step: ``(settle + t) a W x(0)``."""
    assert np.all(np.abs(x[0] - DEFAULT_X) > 0.5), (
        "the opening must differ from the declared initial value for the comparison to mean anything"
    )
    np.testing.assert_array_equal(x, np.broadcast_to(x[0], x.shape))
    np.testing.assert_allclose(y, (settle + t[:, None]) * A * (W @ x[0])[None, :], rtol=rtol, atol=rtol)


def _sample_times(n):
    """The time of each of *n* samples counted from the window's opening, the state after ``t / dt`` steps, read by sample index so every backend is read on the same clock."""
    return DT * (1.0 + np.arange(n))


def _assert_trace_reads_its_opening(trace, settle=0.0, rtol=RTOL["tvboptim"]):
    _assert_reads_its_opening(
        trace.sel(variable="x").values, trace.sel(variable="y").values, _sample_times(trace.sizes["time"]), settle, rtol
    )


def _built(spec):
    """The generated tvboptim module of *spec* and the probe network built by it."""
    module = SimulationExperiment(**copy.deepcopy(spec)).execute("tvboptim")
    return module, module.create_network(jnp.asarray(W), distances=jnp.asarray(np.where(W != 0, DELAY, 0.0)))


def _buffers(state):
    return [np.asarray(c.history) for c in state.initial_state.coupling.values() if "history" in c]


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_drawn_opening_is_what_the_first_delayed_coupling_reads(backend):
    _assert_trace_reads_its_opening(_trace(_drawn(), backend), rtol=RTOL[backend])


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_node_s_declared_state_is_what_the_first_delayed_coupling_reads(backend):
    trace = _trace(_declared_per_node(), backend)
    np.testing.assert_array_equal(trace.sel(variable="x").values[0], [1.5, -0.5])
    _assert_trace_reads_its_opening(trace, rtol=RTOL[backend])


@pytest.mark.backend_tvb
def test_tvb_and_tvboptim_agree_over_the_first_max_delay_window():
    """Both backends open from the same declared state, so they integrate the same numbers sample for sample, the first 4 ms included."""
    spec = _declared_per_node()
    tvb, tvboptim = _trace(spec, "tvb"), _trace(spec, "tvboptim")
    np.testing.assert_allclose(tvboptim.values, tvb.values, rtol=1e-8, atol=1e-8)


def test_a_settle_opens_on_the_drawn_values():
    """The 6 ms settle reads the opening from its first step, so ``y`` has gathered ``6 a W x(0)`` when the measured window opens."""
    spec = _drawn()
    spec["integration"]["transient_time"] = 6.0
    _assert_trace_reads_its_opening(_trace(spec), settle=6.0)


def test_each_trial_reads_its_own_draw_in_its_first_delayed_coupling():
    """Trials draw their openings inside the mapped function, and each trial's delay lines hold its own draw, not the default and not another trial's.

    The spec declares ``output``, from which a sweep's ``variable`` axis takes its names.
    """
    spec = _drawn()
    spec["dynamics"]["output"] = ["x", "y"]
    spec["explorations"] = {
        "ens": {"name": "ens", "mode": "product", "n_trials": 3, "space": [{"parameter": "Probe.k", "explored_values": [0.0]}]}
    }
    grid = SimulationExperiment(**spec).run("tvboptim").explorations["ens"].as_grid()
    grid = grid.squeeze([d for d in grid.dims if d not in ("trial", "time", "variable", "node")])
    t = _sample_times(grid.sizes["time"])
    openings = []
    for trial in grid.transpose("trial", "time", "variable", "node"):
        x, y = trial.sel(variable="x").values, trial.sel(variable="y").values
        _assert_reads_its_opening(x, y, t)
        openings.append(x[0])
    assert np.unique(np.round(openings, 9), axis=0).shape[0] == 3, "the trials must draw three openings"


def test_a_settled_window_keeps_the_settle_s_trajectory_as_its_history():
    """The measured window after a settle continues the settle's delay lines: its buffer is the window tvboptim cuts from the settle, which varies over its rows, and no refill replaces it with the opening."""
    spec = _drawn()
    spec["dynamics"]["parameters"]["k"]["value"] = -0.1
    spec["integration"]["transient_time"] = 6.0
    module, network = _built(spec)
    sim = module.run_simulation(network)
    settled = np.asarray(network.get_history(DT))[:, [module._STATE_INDEX["x"]], :]
    buffers = _buffers(sim.state)
    assert buffers, "the probe must run a delayed coupling"
    for buffer in buffers:
        assert np.ptp(buffer, axis=0).max() > 1e-3
        np.testing.assert_array_equal(buffer, settled)


def test_an_opening_at_the_declared_initial_values_integrates_bit_for_bit_as_before():
    """Without a write to the opening the refill rebuilds the buffer ``prepare`` built, so the run is the unpatched one.

    The unpatched run is reconstructed here rather than stored: it is the same model function on the same state with the buffer ``prepare`` fills from ``INITIAL_STATE``, which before the fix nothing replaced.
    """
    module, network = _built(PROBE)
    sim = module.run_simulation(network)
    _, prepared = prepare(network, module.get_solver(), t0=0.0, t1=DURATION, dt=DT)
    np.testing.assert_array_equal(np.asarray(sim.state.initial_state.dynamics), np.asarray(prepared.initial_state.dynamics))
    for refilled, default in zip(_buffers(sim.state), _buffers(prepared), strict=True):
        np.testing.assert_array_equal(refilled, default)
    unpatched = sim.state.copy()
    unpatched.initial_state.coupling = prepared.initial_state.coupling
    solve = jax.jit(sim.model_fn)
    np.testing.assert_array_equal(np.asarray(solve(sim.state).ys), np.asarray(solve(unpatched).ys))


def test_an_undelayed_run_is_left_as_it_is():
    spec = _drawn()
    spec["network"]["coupling"]["LinearCoupling"]["delayed"] = False
    module = SimulationExperiment(**spec).execute("tvboptim")
    state = module.run_simulation(module.create_network(jnp.asarray(W)), run_main=False).state
    assert not _buffers(state)
    assert module._refill_history(state) is state


@pytest.mark.parametrize("kind", ["stock", "compact"])
def test_the_refill_is_the_buffer_tvboptim_builds_on_the_opening(kind):
    """Refilled, a prepared buffer equals the one tvboptim's own ``get_history`` builds on a network whose stored history is the opening, for the stock dense coupling and tvbo's compact one alike."""
    rng = np.random.default_rng(3)
    n = 30
    mask = rng.random((n, n)) < 0.4
    np.fill_diagonal(mask, False)
    graph = DenseDelayGraph(jnp.asarray(np.where(mask, rng.random((n, n)), 0.0)), jnp.asarray(rng.uniform(1.0, 9.0, (n, n))))
    opening = jnp.asarray(rng.uniform(0.05, 0.4, (2, n)))

    def network(history=None):
        return Network(
            TwoChannelEI(), {"eib": COUPLINGS[kind](wLRE=jnp.ones((n, n)), wFFI=jnp.ones((n, n)))}, graph, history=history
        )

    _, state = prepare(network(), Euler(), t1=10.0, dt=1.0)
    state.initial_state.dynamics = opening
    refilled = NS["_refill_history"](state).initial_state.coupling.eib.history
    with pytest.warns(UserWarning, match="History covers"):
        stored = network(NativeSolution(ts=jnp.zeros(1), ys=opening[None], dt=1.0, variable_names=TwoChannelEI.STATE_NAMES))
    _, reference = prepare(stored, Euler(), t1=10.0, dt=1.0)
    assert not np.allclose(np.asarray(state.initial_state.coupling.eib.history), np.asarray(refilled))
    np.testing.assert_array_equal(np.asarray(refilled), np.asarray(reference.initial_state.coupling.eib.history))
