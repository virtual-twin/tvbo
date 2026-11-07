"""The CompactDelayedCoupling tvbo emits for delayed tvboptim couplings integrates the same network as tvboptim's stock DelayedCoupling.

The class is rendered from its template def and executed on its own, exactly as a generated module defines it, then subclassed the way the cfun template subclasses it. Each test integrates one network twice, once through tvboptim's stock path and once through the compact path, and compares the trajectories, on a dense delay graph and on a sparse one.
"""

from __future__ import annotations

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
from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph, SparseDelayGraph
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.experimental.network_dynamics.solvers import BoundedSolver, Euler

TEMPLATE = Template(filename="tvbo/templates/tvboptim/tvbo-tvboptim-compact-coupling.py.mako")


def _compact_namespace() -> dict:
    """The emitted CompactDelayedCoupling and its helper, defined in the scope a generated module gives them."""
    namespace = {"jax": jax, "jnp": jnp, "Bunch": Bunch, "DelayedCoupling": DelayedCoupling}
    source = TEMPLATE.get_def("render_compact_delayed_coupling").render()
    exec(compile(source, "<compact_coupling>", "exec"), namespace)
    return namespace


COMPACT = _compact_namespace()["CompactDelayedCoupling"]


class TwoChannelEI(AbstractDynamics):
    """A bounded two-population node driven by both coupling channels, so each channel's sum shows in the trajectory."""

    STATE_NAMES = ("S_e", "S_i")
    INITIAL_STATE = (0.1, 0.1)
    DEFAULT_PARAMS = Bunch(tau_e=10.0, tau_i=5.0)
    COUPLING_INPUTS = {"eib": 2}

    def dynamics(self, t, state, params, coupling, external):
        drive_e = jnp.tanh(coupling.eib[0] - 0.8 * state[1])
        drive_i = jnp.tanh(0.5 * coupling.eib[1] + state[0])
        return jnp.stack([(-state[0] + (1.0 - state[0]) * drive_e) / params.tau_e, (-state[1] + drive_i) / params.tau_i])


def _eib_pre(self, incoming_states, local_states, params):
    source = incoming_states[0]
    return jnp.stack([params.G * source * params.wLRE, params.G * source * params.wFFI], axis=0)


def _eib_class(base):
    """A dual-output delayed coupling with per-edge wLRE/wFFI, rendered the way the cfun template renders EIBLinearCoupling."""
    return type(
        f"EIB{base.__name__}",
        (base,),
        {
            "N_OUTPUT_STATES": 2,
            "EDGE_PARAMS": ("wFFI", "wLRE"),
            "DEFAULT_PARAMS": Bunch(G=2.0, wFFI=1.0, wLRE=1.0),
            "__init__": lambda self, **kw: base.__init__(self, incoming_states=["S_e"], **kw),
            "pre": _eib_pre,
            "post": lambda self, summed, local, params: summed,
        },
    )


STOCK_EIB, COMPACT_EIB = _eib_class(DelayedCoupling), _eib_class(COMPACT)


def _connectome(n_nodes, density, max_delay_steps, seed=0):
    rng = np.random.default_rng(seed)
    mask = rng.random((n_nodes, n_nodes)) < density
    np.fill_diagonal(mask, False)
    mask[1] = False
    weights = np.where(mask, rng.random((n_nodes, n_nodes)), 0.0) / (density * n_nodes)
    delays = rng.uniform(1.0, float(max_delay_steps), (n_nodes, n_nodes))
    edge_a, edge_b = rng.uniform(0.5, 1.5, (2, n_nodes, n_nodes))
    return (jnp.asarray(a) for a in (weights, delays, edge_a, edge_b))


def _solve(
    coupling_cls,
    n_nodes=40,
    density=0.35,
    max_delay_steps=12,
    n_steps=300,
    noise=True,
    sparse=False,
    per_edge=False,
    **coupling_kw,
):
    """A prepared network on a dense or a sparse delay graph, with the edge parameters as matrices or, on a sparse graph, per stored edge."""
    weights, delays, edge_a, edge_b = _connectome(n_nodes, density, max_delay_steps)
    graph = SparseDelayGraph(weights, delays) if sparse else DenseDelayGraph(weights, delays)
    if per_edge:
        edge_a, edge_b = graph.gather_edges(edge_a), graph.gather_edges(edge_b)
    network = Network(
        TwoChannelEI(),
        {"eib": coupling_cls(wLRE=edge_a, wFFI=edge_b, **coupling_kw)},
        graph,
        noise=AdditiveNoise(sigma=0.01, apply_to=["S_e", "S_i"], key=jax.random.key(3)) if noise else None,
    )
    solver = BoundedSolver(Euler(), low=jnp.zeros((2, 1)), high=jnp.ones((2, 1)))
    solve_fn, config = prepare(network, solver, t1=float(n_steps), dt=1.0)
    return solve_fn, config


def _trajectory(coupling_cls, **kw):
    solve_fn, config = _solve(coupling_cls, **kw)
    return np.asarray(jax.jit(solve_fn)(config).ys), config


@pytest.mark.parametrize("noise", [False, True])
def test_compact_matches_stock_delayed_coupling(noise):
    stock, _ = _trajectory(STOCK_EIB, noise=noise)
    compact, config = _trajectory(COMPACT_EIB, noise=noise)
    assert "_compact" in config._internal.coupling.eib
    assert np.ptp(stock) > 0.05
    np.testing.assert_allclose(compact, stock, rtol=1e-11, atol=1e-13)


def test_compact_matches_stock_on_a_379_node_connectome():
    stock, _ = _trajectory(STOCK_EIB, n_nodes=379, density=0.4, max_delay_steps=18, n_steps=720)
    compact, _ = _trajectory(COMPACT_EIB, n_nodes=379, density=0.4, max_delay_steps=18, n_steps=720)
    np.testing.assert_allclose(compact, stock, rtol=1e-10, atol=1e-12)


def test_compact_follows_live_weights_and_edge_parameters():
    stock_fn, stock_config = _solve(STOCK_EIB)
    compact_fn, compact_config = _solve(COMPACT_EIB)
    for config in (stock_config, compact_config):
        config.graph.weights = 1.7 * config.graph.weights
        config.coupling.eib.wLRE = 0.5 * config.coupling.eib.wLRE
    np.testing.assert_allclose(
        jax.jit(compact_fn)(compact_config).ys, jax.jit(stock_fn)(stock_config).ys, rtol=1e-11, atol=1e-13
    )


@pytest.mark.parametrize("per_edge", [False, True], ids=["matrix-edge-params", "per-edge-params"])
@pytest.mark.parametrize("noise", [False, True])
def test_compact_matches_stock_on_a_sparse_delay_graph(noise, per_edge):
    """On a sparse graph the compact path reads the stored edges with their own weights, delays and edge parameters, and integrates what the stock sparse path and the dense path integrate."""
    stock, _ = _trajectory(STOCK_EIB, noise=noise, sparse=True, per_edge=per_edge)
    compact, config = _trajectory(COMPACT_EIB, noise=noise, sparse=True, per_edge=per_edge)
    dense, _ = _trajectory(STOCK_EIB, noise=noise)
    assert "_compact" in config._internal.coupling.eib
    assert "off_pattern" not in config._internal.coupling.eib._compact
    assert np.ptp(stock) > 0.05
    np.testing.assert_allclose(compact, stock, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(compact, dense, rtol=1e-11, atol=1e-13)


def test_compact_follows_live_edge_parameters_on_a_sparse_graph():
    stock_fn, stock_config = _solve(STOCK_EIB, sparse=True)
    compact_fn, compact_config = _solve(COMPACT_EIB, sparse=True)
    for config in (stock_config, compact_config):
        config.coupling.eib.wLRE = 0.5 * config.coupling.eib.wLRE
    np.testing.assert_allclose(
        jax.jit(compact_fn)(compact_config).ys, jax.jit(stock_fn)(stock_config).ys, rtol=1e-11, atol=1e-13
    )


def test_compact_refuses_a_weight_outside_the_prepared_pattern():
    compact_fn, config = _solve(COMPACT_EIB)
    config.graph.weights = config.graph.weights.at[1, 0].set(0.3)
    with pytest.raises(Exception, match="outside the pattern"):
        jax.block_until_ready(jax.jit(compact_fn)(config).ys)


@pytest.mark.parametrize(
    "coupling_kw",
    [dict(history_interpolation="linear"), dict(buffer_strategy="circular")],
)
def test_configurations_outside_the_compact_path_run_the_stock_path(coupling_kw):
    stock, _ = _trajectory(STOCK_EIB, **coupling_kw)
    compact, config = _trajectory(COMPACT_EIB, **coupling_kw)
    assert "_compact" not in config._internal.coupling.eib
    np.testing.assert_array_equal(compact, stock)


def test_generated_delayed_couplings_inherit_the_compact_class():
    from tvbo import SimulationExperiment, database_path

    exp = SimulationExperiment.from_file(str(database_path / "experiments" / "Delay_Speed_Synchronization.yaml"))
    code = exp.render(format="tvboptim")
    assert code.count("class CompactDelayedCoupling(DelayedCoupling):") == 1
    assert "(CompactDelayedCoupling):" in code.split("class CompactDelayedCoupling(DelayedCoupling):")[1]
