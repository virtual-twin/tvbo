"""One pytree registration in tvbo: a class declares its leaves and its static fields, `tvbo.utils.pytree` does the rest.

The bespoke `tree_flatten` pairs are gone, and with them their hazards: a dict in the static half made a treedef unhashable, so a class could be traced once but never cached; a fixed positional tuple of children made `vmap` over one field a reshuffle of every other. Here the static half is one JSON string, the children are keyed by name, and the same rules hold for `Network`, `Noise`, `TimeSeries`, `SimulationState` and `Bunch`.
"""

import pathlib
import re

import jax
import jax.numpy as jnp
import numpy as np

from tvbo.utils import Bunch

ROOT = pathlib.Path(__file__).resolve().parents[1] / "tvbo"


def test_no_class_in_tvbo_flattens_itself():
    """`register` is the only way into JAX: no `tree_flatten` method and no `register_pytree_node_class` anywhere under tvbo/."""
    offenders = []
    for path in ROOT.rglob("*.py"):
        text = path.read_text()
        if re.search(r"def tree_flatten\(|def tree_unflatten\(|register_pytree_node_class", text):
            offenders.append(str(path.relative_to(ROOT.parent)))
    assert offenders == []


def test_a_timeseries_is_traced_by_its_arrays_and_cached_by_its_metadata():
    from tvbo.data.types import TimeSeries

    def make(title="TimeSeries", scale=1.0):
        return TimeSeries(
            np.arange(4.0),
            scale * np.ones((4, 2, 3, 1)),
            title=title,
            labels_dimensions={"state": ["V", "W"]},
            units={"time": "ms"},
        )

    traces = []

    @jax.jit
    def total(ts):
        traces.append(1)
        return jnp.sum(ts.data)

    assert float(total(make())) == 24.0
    assert float(total(make(scale=2.0))) == 48.0
    assert len(traces) == 1, "the same metadata is the same treedef, whatever the arrays hold"
    total(make(title="other"))
    assert len(traces) == 2, "different metadata is a different treedef"
    back = jax.tree_util.tree_map(lambda x: x, make())
    assert isinstance(back, TimeSeries) and back.labels_dimensions == {"state": ["V", "W"]} and back.units == {"time": "ms"}
    assert hash(jax.tree_util.tree_structure(make())), "the static half is hashable"


def test_a_simulation_state_carries_its_noise_sigma_as_a_leaf_and_nt_as_static():
    from tvbo.classes.network import Network
    from tvbo.classes.noise import Noise
    from tvbo.data.types import SimulationState

    noise = Noise(noise_type="gaussian", parameters={"sigma": {"value": 0.1}})
    noise.sigma_vec = jnp.array([0.1, 0.2])
    state = SimulationState(
        initial_conditions=jnp.zeros((3, 2)),
        network=Network.from_matrix(np.eye(3)),
        dt=0.1,
        noise=noise,
        parameters=Bunch(a=jnp.array(1.0)),
        stimulus=None,
        monitor_parameters=None,
        nt=50,
    )
    doubled = jax.tree_util.tree_map(lambda x: 2 * x, state)
    assert isinstance(doubled, SimulationState) and isinstance(doubled.noise, Noise) and isinstance(doubled.parameters, Bunch)
    np.testing.assert_allclose(doubled.noise.sigma_vec, [0.2, 0.4], rtol=1e-6)
    np.testing.assert_array_equal(doubled.network.arrays["edges/weight"], 2 * np.eye(3))
    assert doubled.nt == 50 and doubled.noise.noise_type == "gaussian"

    @jax.jit
    def steps(s):
        return jnp.arange(s.nt) * s.dt + jnp.sum(s.noise.sigma_vec)

    assert steps(state).shape == (50,), "nt is a Python int inside the trace"


def test_a_bunch_flattens_by_sorted_key_and_comes_back_a_bunch():
    b = Bunch(z=jnp.array(1.0), a=jnp.array(2.0))
    leaves, treedef = jax.tree_util.tree_flatten(b)
    np.testing.assert_array_equal(leaves, [2.0, 1.0])
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(back, Bunch) and back.a == 2.0 and back["z"] == 1.0
    assert jax.tree_util.tree_structure(Bunch(a=1, z=2)) == jax.tree_util.tree_structure(Bunch(z=1, a=2))
