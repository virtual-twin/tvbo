"""Chained tvboptim solves carry their delay history, so a TR-by-TR loop integrates the same network as one continuous run.

A fitting loop (FIC, FIC/EIB) runs its prepared solve once per TR and hands the final state to the next call. The delayed couplings' history buffers live in ``initial_state.coupling`` and are not returned by the solve, so the generated experiment rebuilds them from the trajectory with ``_carry_delay_history``. These tests run that helper, taken verbatim from the experiment template, on the stock tvboptim DelayedCoupling and on tvbo's CompactDelayedCoupling, through a roll and a circular history buffer named by ``buffer_strategy`` (both tvboptim 0.4.0 and the development line take it), and check a whole fic_eib run changes once the history is carried and ends on the same bits through either buffer.
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
    """``_carry_delay_history`` and its registry exactly as the experiment template emits them, plus the emitted CompactDelayedCoupling; the registry is the last top-level ``_DELAY`` name above the carry, whatever the template calls it."""
    source = open(EXPERIMENT_TEMPLATE).read()
    carry = source.index("def _carry_delay_history(")
    helper = source[source.rindex("\n_DELAY", 0, carry) + 1 : source.index("def create_network(")]
    namespace = {"jax": jax, "jnp": jnp, "eqx": eqx, "Bunch": Bunch, "DelayedCoupling": DelayedCoupling}
    exec(compile(helper, "<carry_delay_history>", "exec"), namespace)
    compact = Template(filename="tvbo/templates/tvboptim/tvbo-tvboptim-compact-coupling.py.mako")
    exec(compile(compact.get_def("render_compact_delayed_coupling").render(), "<compact_coupling>", "exec"), namespace)
    return namespace


NS = _template_namespace()


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


BUFFERS = pytest.mark.parametrize("buffer", ["roll", "circular"])
"""The history buffers the carry hands on, each test run on both."""


def _prepared(coupling_cls, n_steps, noise, buffer):
    """A prepared solve of *n_steps* on a delayed network whose coupling holds a *buffer* history, recorded in the template's registry as create_network() records a built coupling."""
    rng = np.random.default_rng(0)
    mask = rng.random((N_NODES, N_NODES)) < 0.4
    np.fill_diagonal(mask, False)
    weights = jnp.asarray(np.where(mask, rng.random((N_NODES, N_NODES)), 0.0) / (0.4 * N_NODES))
    delays = jnp.asarray(rng.uniform(1.0, 15.0, (N_NODES, N_NODES)))
    edge_a, edge_b = (jnp.asarray(a) for a in rng.uniform(0.5, 1.5, (2, N_NODES, N_NODES)))
    coupling = coupling_cls(wLRE=edge_a, wFFI=edge_b, buffer_strategy=buffer)
    NS["_DELAY_LINES"]["eib"] = (["S_e"], coupling.buffer_strategy)
    network = Network(
        TwoChannelEI(),
        {"eib": coupling},
        DenseDelayGraph(weights, delays),
        noise=AdditiveNoise(sigma=0.02, apply_to=["S_e", "S_i"], key=jax.random.key(1)) if noise else None,
    )
    solver = BoundedSolver(Euler(), low=jnp.zeros((2, 1)), high=jnp.ones((2, 1)))
    solve_fn, config = prepare(network, solver, t1=float(n_steps), dt=1.0)
    config.initial_state.dynamics = jnp.asarray(rng.uniform(0.05, 0.4, (2, N_NODES)))
    return jax.jit(solve_fn), config


def _continuous_and_chunked(coupling_cls, noise, carry, buffer):
    """One run of N_CHUNKS * CHUNK steps, the same run as N_CHUNKS chained solves fed the same noise increments, and the (history, write pointer or None) each carry handed on."""
    increments = jax.random.normal(jax.random.key(7), (N_CHUNKS * CHUNK, 2, N_NODES))
    whole_fn, whole = _prepared(coupling_cls, N_CHUNKS * CHUNK, noise, buffer)
    if noise:
        whole._internal.noise_samples = increments
    continuous = np.asarray(whole_fn(whole).ys)

    chunk_fn, state = _prepared(coupling_cls, CHUNK, noise, buffer)
    pieces, carried = [], []
    for k in range(N_CHUNKS):
        if noise:
            state = eqx.tree_at(
                lambda s: s._internal.noise_samples,
                state,
                increments[k * CHUNK : (k + 1) * CHUNK],
                is_leaf=lambda x: x is None,
            )
        result = chunk_fn(state)
        pieces.append(np.asarray(result.ys))
        state = eqx.tree_at(lambda s: s.initial_state.dynamics, state, result.ys[-1][:2])
        if carry:
            state = NS["_carry_delay_history"](state, result)
            line = state.initial_state.coupling.eib
            carried.append((np.asarray(line.history), np.asarray(line.write_idx) if "write_idx" in line else None))
    return continuous, np.concatenate(pieces), carried


@BUFFERS
@pytest.mark.parametrize("noise", [False, True])
@pytest.mark.parametrize("kind", ["stock", "compact"])
def test_chained_solves_equal_one_continuous_run(kind, noise, buffer):
    continuous, chained, _ = _continuous_and_chunked(COUPLINGS[kind], noise, carry=True, buffer=buffer)
    assert np.ptp(continuous) > 0.05
    np.testing.assert_allclose(chained, continuous, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("noise", [False, True])
@pytest.mark.parametrize("kind", ["stock", "compact"])
def test_the_circular_carry_equals_the_roll_carry_bitwise(kind, noise):
    """Chained solves through a circular buffer integrate the bits chained solves through a roll buffer integrate, and the one continuous run does too."""
    roll_continuous, roll_chained, _ = _continuous_and_chunked(COUPLINGS[kind], noise, carry=True, buffer="roll")
    continuous, chained, _ = _continuous_and_chunked(COUPLINGS[kind], noise, carry=True, buffer="circular")
    np.testing.assert_array_equal(chained, roll_chained)
    np.testing.assert_array_equal(continuous, roll_continuous)


@pytest.mark.parametrize("kind", ["stock", "compact"])
def test_a_carry_into_the_prepare_time_pointer_changes_no_bit(kind):
    """Every carry hands on the buffer a roll carry hands on, bit for bit, with the circular write pointer back at the 0 prepare() gives it: tvbo only ever carries into a prepared buffer, so unrolling it from its pointer changes nothing."""
    *_, roll = _continuous_and_chunked(COUPLINGS[kind], True, carry=True, buffer="roll")
    *_, circular = _continuous_and_chunked(COUPLINGS[kind], True, carry=True, buffer="circular")
    assert len(circular) == N_CHUNKS
    for (roll_history, roll_pointer), (history, pointer) in zip(roll, circular, strict=True):
        assert roll_pointer is None
        assert pointer is not None and int(pointer) == 0
        np.testing.assert_array_equal(history, roll_history)


@BUFFERS
def test_chained_solves_without_the_carry_restart_every_delay_line(buffer):
    continuous, chained, _ = _continuous_and_chunked(COUPLINGS["compact"], False, carry=False, buffer=buffer)
    np.testing.assert_allclose(chained[:CHUNK], continuous[:CHUNK], rtol=1e-11, atol=1e-13)
    assert np.max(np.abs(chained - continuous)) > 1e-4


@BUFFERS
def test_an_unrecorded_transmitted_state_cannot_be_carried(buffer):
    chunk_fn, state = _prepared(COUPLINGS["stock"], CHUNK, False, buffer)
    result = chunk_fn(state)
    NS["_DELAY_LINES"]["eib"] = (["V"], buffer)
    with pytest.raises(ValueError, match="cannot carry its delay history"):
        NS["_carry_delay_history"](state, result)


def test_a_preallocated_buffer_is_refused_by_name():
    chunk_fn, state = _prepared(COUPLINGS["stock"], CHUNK, False, "preallocated")
    result = chunk_fn(state)
    with pytest.raises(
        NotImplementedError,
        match="coupling 'eib': carrying a delay history across chained solves does not support the 'preallocated' buffer",
    ):
        NS["_carry_delay_history"](state, result)


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
    assert max(float(np.max(np.abs(a - b))) for a, b in zip(carried, restarted, strict=True)) > 1e-8


def _build_delayed_couplings_with(monkeypatch, buffer):
    """Make every tvboptim DelayedCoupling built from here on hold a *buffer* history, the generated module's included, and return the list each one joins once built."""
    built, construct = [], DelayedCoupling.__init__

    def init(self, *args, **kwargs):
        construct(self, *args, **{**kwargs, "buffer_strategy": buffer})
        built.append(self)

    monkeypatch.setattr(DelayedCoupling, "__init__", init)
    return built


def test_a_delayed_fic_eib_fit_ends_on_the_same_bits_through_either_buffer(tmp_path, monkeypatch):
    """The flow that carries, a delayed fic_eib fit, ends on the same bits whether its delayed coupling holds a roll or a circular history; each run's built EIBLinearCoupling reports the buffer it was given."""
    fits = {}
    for buffer in ("roll", "circular"):
        with monkeypatch.context() as patch:
            built = _build_delayed_couplings_with(patch, buffer)
            fits[buffer] = _delayed_fic_eib_run(tmp_path, carry=True)
        assert "EIBLinearCoupling" in {type(c).__name__ for c in built}
        assert {c.buffer_strategy for c in built} == {buffer}
    for a, b in zip(fits["circular"], fits["roll"], strict=True):
        np.testing.assert_array_equal(a, b)
