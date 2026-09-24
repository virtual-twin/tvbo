"""Chained tvboptim solves carry their delay history, so a TR-by-TR loop integrates the same network as one continuous run.

A fitting loop (FIC, FIC/EIB) runs its prepared solve once per TR and hands the final state to the next call. The delayed couplings' history buffers live in ``initial_state.coupling`` and are not returned by the solve, so the generated experiment rebuilds them from the trajectory with ``_carry_delay_history``. These tests run that helper, taken verbatim from the experiment template, on the stock tvboptim DelayedCoupling and on tvbo's CompactDelayedCoupling, and check a whole fic_eib run changes once the history is carried.
"""

from __future__ import annotations

import re

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mako.template import Template

jax.config.update("jax_enable_x64", True)

pytest.importorskip("tvboptim")
pytestmark = pytest.mark.backend_tvboptim

from tvboptim.experimental.network_dynamics import Network, prepare
from tvboptim.experimental.network_dynamics.core.bunch import Bunch
from tvboptim.experimental.network_dynamics.coupling.base import DelayedCoupling
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Euler

EXPERIMENT_TEMPLATE = "tvbo/templates/tvboptim/tvbo-tvboptim-experiment.py.mako"
CARRY_CALL = "state = _carry_delay_history("


def _template_namespace() -> dict:
    """``_carry_delay_history`` and its registry exactly as the experiment template emits them, plus the emitted CompactDelayedCoupling."""
    source = open(EXPERIMENT_TEMPLATE).read()
    helper = source[source.index("_DELAY_SOURCE_STATES = {}") : source.index("def create_network(")]
    namespace = {"jax": jax, "jnp": jnp, "eqx": eqx, "Bunch": Bunch, "DelayedCoupling": DelayedCoupling}
    exec(compile(helper, "<carry_delay_history>", "exec"), namespace)
    compact = Template(filename="tvbo/templates/tvboptim/tvbo-tvboptim-compact-coupling.py.mako")
    exec(compile(compact.get_def("render_compact_delayed_coupling").render(), "<compact_coupling>", "exec"), namespace)
    return namespace


NS = _template_namespace()
NS["_DELAY_SOURCE_STATES"]["eib"] = ["S_e"]


class TwoChannelEI(AbstractDynamics):
    STATE_NAMES = ("S_e", "S_i")
    INITIAL_STATE = (0.1, 0.1)
    DEFAULT_PARAMS = Bunch(tau_e=10.0, tau_i=5.0)
    COUPLING_INPUTS = {"eib": 2}

    def dynamics(self, t, state, params, coupling, external):
        drive_e = jnp.tanh(coupling.eib[0] - 0.8 * state[1])
        drive_i = jnp.tanh(0.5 * coupling.eib[1] + state[0])
        return jnp.stack([(-state[0] + (1.0 - state[0]) * drive_e) / params.tau_e, (-state[1] + drive_i) / params.tau_i])


def _eib_class(base):
    return type(
        f"EIB{base.__name__}",
        (base,),
        {
            "N_OUTPUT_STATES": 2,
            "EDGE_PARAMS": ("wFFI", "wLRE"),
            "DEFAULT_PARAMS": Bunch(G=2.0, wFFI=1.0, wLRE=1.0),
            "__init__": lambda self, **kw: base.__init__(self, incoming_states=["S_e"], **kw),
            "pre": lambda self, x, local, p: jnp.stack([p.G * x[0] * p.wLRE, p.G * x[0] * p.wFFI], axis=0),
            "post": lambda self, summed, local, params: summed,
        },
    )


COUPLINGS = {"stock": _eib_class(DelayedCoupling), "compact": _eib_class(NS["CompactDelayedCoupling"])}
N_NODES, CHUNK, N_CHUNKS = 30, 40, 5


def _prepared(coupling_cls, n_steps, noise):
    rng = np.random.default_rng(0)
    mask = rng.random((N_NODES, N_NODES)) < 0.4
    np.fill_diagonal(mask, False)
    weights = jnp.asarray(np.where(mask, rng.random((N_NODES, N_NODES)), 0.0) / (0.4 * N_NODES))
    delays = jnp.asarray(rng.uniform(1.0, 15.0, (N_NODES, N_NODES)))
    edge_a, edge_b = (jnp.asarray(a) for a in rng.uniform(0.5, 1.5, (2, N_NODES, N_NODES)))
    network = Network(
        TwoChannelEI(),
        {"eib": coupling_cls(wLRE=edge_a, wFFI=edge_b)},
        DenseDelayGraph(weights, delays),
        noise=AdditiveNoise(sigma=0.02, apply_to=["S_e", "S_i"], key=jax.random.key(1)) if noise else None,
    )
    solver = BoundedSolver(Euler(), low=jnp.zeros((2, 1)), high=jnp.ones((2, 1)))
    solve_fn, config = prepare(network, solver, t1=float(n_steps), dt=1.0)
    config.initial_state.dynamics = jnp.asarray(rng.uniform(0.05, 0.4, (2, N_NODES)))
    return jax.jit(solve_fn), config


def _continuous_and_chunked(coupling_cls, noise, carry):
    """One run of N_CHUNKS * CHUNK steps, and the same run as N_CHUNKS chained solves fed the same noise increments."""
    increments = jax.random.normal(jax.random.key(7), (N_CHUNKS * CHUNK, 2, N_NODES))
    whole_fn, whole = _prepared(coupling_cls, N_CHUNKS * CHUNK, noise)
    if noise:
        whole._internal.noise_samples = increments
    continuous = np.asarray(whole_fn(whole).ys)

    chunk_fn, state = _prepared(coupling_cls, CHUNK, noise)
    pieces = []
    for k in range(N_CHUNKS):
        if noise:
            state = eqx.tree_at(lambda s: s._internal.noise_samples, state, increments[k * CHUNK : (k + 1) * CHUNK], is_leaf=lambda x: x is None)
        result = chunk_fn(state)
        pieces.append(np.asarray(result.ys))
        state = eqx.tree_at(lambda s: s.initial_state.dynamics, state, result.ys[-1][:2])
        if carry:
            state = NS["_carry_delay_history"](state, result)
    return continuous, np.concatenate(pieces)


@pytest.mark.parametrize("noise", [False, True])
@pytest.mark.parametrize("kind", ["stock", "compact"])
def test_chained_solves_equal_one_continuous_run(kind, noise):
    continuous, chained = _continuous_and_chunked(COUPLINGS[kind], noise, carry=True)
    assert np.ptp(continuous) > 0.05
    np.testing.assert_allclose(chained, continuous, rtol=1e-11, atol=1e-13)


def test_chained_solves_without_the_carry_restart_every_delay_line():
    continuous, chained = _continuous_and_chunked(COUPLINGS["compact"], False, carry=False)
    np.testing.assert_allclose(chained[:CHUNK], continuous[:CHUNK], rtol=1e-11, atol=1e-13)
    assert np.max(np.abs(chained - continuous)) > 1e-4


def test_an_unrecorded_transmitted_state_cannot_be_carried():
    chunk_fn, state = _prepared(COUPLINGS["stock"], CHUNK, False)
    result = chunk_fn(state)
    NS["_DELAY_SOURCE_STATES"]["eib"] = ["V"]
    try:
        with pytest.raises(ValueError, match="cannot carry its delay history"):
            NS["_carry_delay_history"](state, result)
    finally:
        NS["_DELAY_SOURCE_STATES"]["eib"] = ["S_e"]


def _delayed_fic_eib_run(tmp_path, carry):
    from tvbo import SimulationExperiment, database_path

    exp = SimulationExperiment.from_file(str(database_path / "experiments" / "EI_Tuning_FIC_EIB_Optimization.yaml"))
    exp.configure()
    exp.network.coupling["EIBLinearCoupling"].delayed = True
    exp.integration.duration = 7200.0
    for name in ("fic", "fic_eib"):
        exp.algorithms[name].n_iterations = 3
    code = exp.render_code("tvboptim")
    assert code.count(CARRY_CALL) == 3  # fic tuning, fic_eib warm-up, fic_eib tuning
    if not carry:
        code = re.sub(r"\n +state = _carry_delay_history\([^)]*\)", "", code)
        assert CARRY_CALL not in code
    result = exp.run("tvboptim", mode="algorithms", rendered_code=code, results_root=str(tmp_path))
    fitted = result.fic_eib.state
    return np.asarray(fitted.dynamics.J_i), np.asarray(fitted.coupling.EIBLinearCoupling.wLRE)


def test_carrying_the_history_changes_a_delayed_fic_eib_fit(tmp_path):
    """Guards the carry: the same seeded fit with the pre-fix loop, which restarted the delay lines every TR, ends elsewhere."""
    carried = _delayed_fic_eib_run(tmp_path, carry=True)
    restarted = _delayed_fic_eib_run(tmp_path, carry=False)
    assert max(float(np.max(np.abs(a - b))) for a, b in zip(carried, restarted)) > 1e-8
