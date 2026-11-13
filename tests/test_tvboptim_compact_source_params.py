"""A per-source parameter reaches the CompactDelayedCoupling tvbo emits gathered per slot, and tvboptim's stock dense path broadcast over the source axis.

The class is rendered from its template def, as ``test_tvboptim_compact_delayed_coupling.py`` renders it, and subclassed with a ``SOURCE_PARAMS`` gather written here: a declared parameter is held at ``(m, 1, n_source)``, gathered as ``value[:, 0, slots.source]`` when the compact path runs, and passed unchanged to the stock path, where it broadcasts against the ``(m, n_target, n_source)`` delayed states. ``pre`` contracts each source's m-element state with that source's readout, so the coupling is ``kappa * sum_j w_ij W_out[j] . z_j(t - d_ij)``, which numpy computes from the same history. Two graphs pick the two paths: four regions with few connections, which the compact path takes, and the full four-by-four graph with its diagonal, whose density sends it to the stock path. The file imports nothing from tvbo, so it also runs against the released tvboptim.
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
from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph
from tvboptim.experimental.network_dynamics.solvers import Euler

TEMPLATE = Template(filename="tvbo/templates/tvboptim/tvbo-tvboptim-compact-coupling.py.mako")
N_REGIONS = 4
KAPPA = 0.3


def _compact_class():
    """The emitted CompactDelayedCoupling, defined in the scope a generated module gives it."""
    namespace = {"jax": jax, "jnp": jnp, "Bunch": Bunch, "DelayedCoupling": DelayedCoupling}
    exec(compile(TEMPLATE.get_def("render_compact_delayed_coupling").render(), "<compact_coupling>", "exec"), namespace)
    return namespace["CompactDelayedCoupling"]


def _readout_pre(self, incoming_states, local_states, params):
    return jnp.sum(params._W_out_src * incoming_states, axis=0, keepdims=True)


def _gather_source_params(self, coupling_data, params, graph):
    """The compact precompute, then every ``SOURCE_PARAMS`` entry gathered to the slots' sources."""
    data = COMPACT.precompute(self, coupling_data, params, graph)
    slots = data.get("_compact")
    if slots is not None:
        pre_params = Bunch(data._compact_pre_params)
        for name in self.SOURCE_PARAMS:
            pre_params[name] = jnp.asarray(params[name])[:, 0, slots.source]
        data._compact_pre_params = pre_params
    return data


def _readout_class(base, **body):
    """A delayed coupling reading every state of its source through the source's readout, on *base*."""
    return type(
        f"Readout{base.__name__}",
        (base,),
        {
            "N_OUTPUT_STATES": 1,
            "SOURCE_PARAMS": ("_W_out_src",),
            "DEFAULT_PARAMS": Bunch(kappa=KAPPA, _W_out_src=0.0),
            "pre": _readout_pre,
            "post": lambda self, summed, local_states, params: params.kappa * summed,
            **body,
        },
    )


COMPACT = _compact_class()
COMPACT_READOUT = _readout_class(COMPACT, precompute=_gather_source_params)
STOCK_READOUT = _readout_class(DelayedCoupling)


def _units(m):
    """A node of m leaky units, each driven by the coupling."""
    return type(
        f"Units{m}",
        (AbstractDynamics,),
        {
            "STATE_NAMES": tuple(f"z{k}" for k in range(m)),
            "INITIAL_STATE": tuple(0.1 * (k + 1) for k in range(m)),
            "DEFAULT_PARAMS": Bunch(tau=5.0),
            "COUPLING_INPUTS": {"sc": 1},
            "dynamics": lambda self, t, state, params, coupling, external: (-state + jnp.tanh(coupling.sc)) / params.tau,
        },
    )()


def _graph(n_edges):
    """Four regions with *n_edges* connections at whole-step delays; 16 is the full graph with its diagonal."""
    rng = np.random.default_rng(n_edges)
    weights = np.zeros((N_REGIONS, N_REGIONS))
    chosen = rng.permutation(N_REGIONS * N_REGIONS)[:n_edges]
    weights.flat[chosen] = rng.uniform(0.2, 1.0, n_edges)
    delays = rng.integers(1, 6, (N_REGIONS, N_REGIONS)).astype(float)
    return weights, delays


def _readout(m, seed=0):
    """A ``(n_regions, m)`` readout and the ``(m, 1, n_regions)`` view a source-node read holds."""
    w_out = np.random.default_rng(seed).normal(size=(N_REGIONS, m))
    return w_out, jnp.swapaxes(jnp.asarray(w_out), 0, 1)[:, None, :]


def _coupling_at_random_history(coupling_cls, m, n_edges):
    """The coupling's prepared data, a random history, and the coupling it computes from them as a function of the view."""
    weights, delays = _graph(n_edges)
    w_out, view = _readout(m)
    graph = DenseDelayGraph(jnp.asarray(weights), jnp.asarray(delays))
    coupling = coupling_cls(incoming_states=[f"z{k}" for k in range(m)], _W_out_src=view)
    network = Network(_units(m), {"sc": coupling}, graph)
    data, coupling_state = coupling.prepare(network, 1.0, 0.0, 10.0)
    history = jnp.asarray(np.random.default_rng(1).normal(size=coupling_state.history.shape))
    coupling_state = Bunch(coupling_state, history=history)
    local = jnp.zeros((m, N_REGIONS))

    def computed(source_view):
        params = Bunch(coupling.params, _W_out_src=source_view)
        return coupling.compute(0.0, local, coupling.precompute(data, params, graph), coupling_state, params, graph)

    return data, computed, view, (weights, delays, w_out, np.asarray(history))


def _numpy_coupling(weights, delays, w_out, history):
    """``kappa * sum_j w_ij sum_c W_out[j, c] z_c,j(t - d_ij)``, with the newest sample last in the history."""
    newest = history.shape[0] - 1
    steps = np.clip(np.rint(delays), 0, newest).astype(int)
    target, source = np.indices(weights.shape)
    delayed = history[newest - steps, :, source]
    return KAPPA * np.einsum("ij,ijc,jc->i", weights, delayed, w_out)[None, :]


PATHS = [
    pytest.param(8, True, id="compact-8-edges"),
    pytest.param(2, True, id="compact-2-edges"),
    pytest.param(16, False, id="stock-full"),
]


@pytest.mark.parametrize("m", [5, 4], ids=["m5", "m-equals-regions"])
@pytest.mark.parametrize("n_edges, compact", PATHS)
def test_a_per_source_parameter_is_read_at_each_slots_source(m, n_edges, compact):
    data, computed, view, arrays = _coupling_at_random_history(COMPACT_READOUT, m, n_edges)
    assert ("_compact" in data) is compact
    np.testing.assert_allclose(np.asarray(computed(view)), _numpy_coupling(*arrays), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("m", [5, 4], ids=["m5", "m-equals-regions"])
@pytest.mark.parametrize("n_edges, compact", PATHS)
def test_the_gathered_parameter_keeps_its_gradient(m, n_edges, compact):
    _, computed, view, (weights, delays, w_out, history) = _coupling_at_random_history(COMPACT_READOUT, m, n_edges)
    newest = history.shape[0] - 1
    steps = np.clip(np.rint(delays), 0, newest).astype(int)
    target, source = np.indices(weights.shape)
    expected = np.zeros((N_REGIONS, m))
    np.add.at(expected, source, KAPPA * weights[..., None] * history[newest - steps, :, source])
    gradient = jax.grad(lambda v: jnp.sum(computed(v)))(view)
    np.testing.assert_allclose(np.asarray(gradient)[:, 0, :].T, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n_edges", [8, 16], ids=["compact", "stock"])
def test_a_solve_through_prepare_equals_the_stock_coupling(n_edges):
    weights, delays = _graph(n_edges)
    _, view = _readout(5)

    def trajectory(coupling_cls):
        coupling = coupling_cls(incoming_states=[f"z{k}" for k in range(5)], _W_out_src=view)
        network = Network(_units(5), {"sc": coupling}, DenseDelayGraph(jnp.asarray(weights), jnp.asarray(delays)))
        solve_fn, config = prepare(network, Euler(), t1=40.0, dt=1.0)
        return np.asarray(jax.jit(solve_fn)(config).ys)

    compact, stock = trajectory(COMPACT_READOUT), trajectory(STOCK_READOUT)
    assert np.all(np.isfinite(compact)) and np.ptp(compact) > 0
    np.testing.assert_allclose(compact, stock, rtol=1e-12, atol=1e-12)
