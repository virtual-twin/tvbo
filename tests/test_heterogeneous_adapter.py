"""Heterogeneous-network tvboptim adapter (``tvbo.adapters.tvboptim``).

Covers the P1 interoperability path: a network with different dynamics per node is lowered to a tvboptim ``HeterogeneousNetwork`` (nodes partitioned into ``NodeGroup``s, edges collapsed into a ``SignalRoute``) and run in process via ``exp.run("tvboptim")``.

The module skips only when the installed tvboptim ships no ``network_dynamics`` module at all. Presence is decided by ``find_spec``, which does not execute the module, so every other import failure — a renamed member, a broken upstream import — raises here instead of reading as "API absent"; that silent skip is what left the adapter broken until a doc notebook hit the same import. The names imported below are exactly the ones the adapter imports.
"""

import importlib.util

import numpy as np
import pytest
import yaml

pytest.importorskip("jax")
pytest.importorskip("tvboptim")
if importlib.util.find_spec("tvboptim.experimental.network_dynamics") is None:
    pytest.skip("tvboptim has no heterogeneous network-dynamics API", allow_module_level=True)

from tvboptim.experimental.network_dynamics import (  # noqa: E402, F401
    HeterogeneousNetwork,
    NodeGroup,
    SignalRoute,
)

from tvbo import Dynamics, Network, SimulationExperiment  # noqa: E402
from tvbo.adapters.tvboptim import (  # noqa: E402
    is_heterogeneous,
    to_heterogeneous_network,
    to_tvboptim,
)


def test_single_group_equivalence():
    """The heterogeneous engine reproduces the homogeneous engine exactly on a degenerate one-group partition (same model on every node).

    This is the strongest guard: if the segmented pack/route/scatter machinery were subtly wrong it would show up as a nonzero difference here.
    """
    from tvboptim.experimental.network_dynamics import prepare, solve
    from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
    from tvboptim.experimental.network_dynamics.solvers import Heun

    model = Dynamics.from_string(
        "name: Decay\n"
        "parameters: {k: {value: 0.5}}\n"
        "state_variables:\n"
        "  x: {equation: {rhs: '-k*x + c_in'}, initial_value: 0.7}\n"
        "coupling_inputs: {c_in: {}}"
    )
    network = Network(
        **yaml.safe_load(
            """
            label: Homo2
            number_of_nodes: 2
            nodes: [{id: 0, label: A, dynamics: Decay}, {id: 1, label: B, dynamics: Decay}]
            edges:
              - {source: 0, target: 1, parameters: {weight: {value: 0.3}}, source_var: x_out, target_var: c_in, directed: true}
              - {source: 1, target: 0, parameters: {weight: {value: 0.2}}, source_var: x_out, target_var: c_in, directed: true}
            """
        )
    )
    network.dynamics["Decay"] = model
    assert not is_heterogeneous(SimulationExperiment(network=network))  # one dynamics -> homogeneous

    solver = Heun(block_size=100)
    kw = dict(t0=0.0, t1=50.0, dt=0.1)

    net_h = to_tvboptim(
        network,
        dynamics=model.execute("tvboptim"),
        coupling={"c_in": LinearCoupling(incoming_states="x")},
    )
    ys_h = np.asarray(solve(net_h, solver, **kw).ys)

    het = to_heterogeneous_network(network)
    simulate, config = prepare(het, solver, **kw)
    ys_e = np.asarray(simulate(config).ys["Decay"])

    assert ys_h.shape == ys_e.shape
    assert np.max(np.abs(ys_h - ys_e)) < 1e-9


def _hetero_experiment():
    driver = Dynamics.from_string(
        "name: SlowDriver\n"
        "parameters: {a: {value: 0.5}, omega: {value: 0.3}}\n"
        "state_variables:\n"
        "  x: {equation: {rhs: 'a*x - omega*z - x*(x**2 + z**2) + c_in'}, initial_value: 1.0}\n"
        "  z: {equation: {rhs: 'omega*x + a*z - z*(x**2 + z**2)'}, initial_value: 0.0}\n"
        "coupling_inputs: {c_in: {}}"
    )
    fhn = Dynamics.from_string(
        "name: Excitable\n"
        "parameters: {a: {value: 0.7}, b: {value: 0.8}, tau: {value: 12.5}, I_ext: {value: 0.3}}\n"
        "state_variables:\n"
        "  v: {equation: {rhs: 'v - v**3/3 - w + I_ext + c_in'}, initial_value: -1.0}\n"
        "  w: {equation: {rhs: '(v + a - b*w)/tau'}, initial_value: -0.5}\n"
        "coupling_inputs: {c_in: {}}"
    )
    vdp = Dynamics.from_string(
        "name: Relaxation\n"
        "parameters: {mu: {value: 2.0}}\n"
        "state_variables:\n"
        "  x: {equation: {rhs: 'mu*(x - x**3/3 - w) + c_in'}, initial_value: -1.5}\n"
        "  w: {equation: {rhs: 'x/mu'}, initial_value: 0.0}\n"
        "coupling_inputs: {c_in: {}}"
    )
    network = Network(
        **yaml.safe_load(
            """
            label: HeterogeneousModulation
            number_of_nodes: 3
            nodes:
              - {id: 0, label: Driver, dynamics: SlowDriver}
              - {id: 1, label: Excitable, dynamics: Excitable}
              - {id: 2, label: Relaxation, dynamics: Relaxation}
            edges:
              - {source: 0, target: 1, parameters: {weight: {value: 0.8}}, source_var: x_out, target_var: c_in, directed: true}
              - {source: 0, target: 2, parameters: {weight: {value: -0.6}}, source_var: x_out, target_var: c_in, directed: true}
              - {source: 1, target: 2, parameters: {weight: {value: 0.1}}, source_var: v_out, target_var: c_in, directed: true}
              - {source: 2, target: 1, parameters: {weight: {value: 0.1}}, source_var: x_out, target_var: c_in, directed: true}
            """
        )
    )
    network.dynamics["SlowDriver"] = driver
    network.dynamics["Excitable"] = fhn
    network.dynamics["Relaxation"] = vdp
    exp = SimulationExperiment(network=network)
    exp.integration.duration = 300.0
    exp.integration.step_size = 0.1
    return exp


def test_adapter_partitions_and_routes():
    """to_heterogeneous_network builds one group per dynamics and one route."""
    exp = _hetero_experiment()
    assert is_heterogeneous(exp)
    het = to_heterogeneous_network(exp.network)
    assert set(het.group_names) == {"SlowDriver", "Excitable", "Relaxation"}
    assert het.group_nodes["SlowDriver"] == (0,)
    assert het.route_names == ("coupling",)
    route = het.routes["coupling"]
    # every group emits; only the two driven nodes receive
    assert set(route.source) == {"SlowDriver", "Excitable", "Relaxation"}
    assert set(route.target) == {"Excitable", "Relaxation"}


def test_heterogeneous_run_regions_and_union():
    """exp.run('tvboptim') integrates the heterogeneous network; per-region / per-variable indexing works and a node holds NaN for variables it lacks."""
    exp = _hetero_experiment()
    res = exp.run("tvboptim")

    assert np.asarray(res.time).shape == (3000,)
    for region, var in [("Driver", "x"), ("Excitable", "v"), ("Relaxation", "x")]:
        d = np.asarray(res.get_region(region).get_state_variable(var).data).squeeze()
        assert d.shape == (3000,)
        assert np.all(np.isfinite(d))

    # union variable axis: the Driver group has no 'v', so its column is NaN
    driver_v = np.asarray(res.get_region("Driver").get_state_variable("v").data)
    assert np.all(np.isnan(driver_v))
