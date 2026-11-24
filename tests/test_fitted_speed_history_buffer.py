"""A fitted conduction speed sizes the history buffer from its domain's lower bound.

The history buffer of a length graph is static: it is sized once, outside jit, for the longest delay, the longest tract over the slowest speed. An optimization that frees ``network.conduction_speed`` moves the speed during the fit, so a step below the build speed asks for a delay longer than the buffer holds. JAX clamps the out-of-range gather index rather than raising, so such a step reads the wrong history and the fit carries on with a silently wrong loss (G197). These pin the buffer to the slowest speed the fit can reach, and refuse a free speed with no positive lower bound to size it from.
"""

import numpy as np
import pytest

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment

_FITTED_SPEED_SPEC = """
id: 9
dynamics:
  name: Kuramoto
  label: Kuramoto
  parameters:
    omega: {name: omega, value: 0.0628, unit: rad_per_ms}
  coupling_inputs:
    c: {name: c, description: "coupling"}
  state_variables:
    theta:
      unit: rad
      equation: {lhs: "Derivative(theta, t)", rhs: "omega + c"}
      variable_of_interest: true
      coupling_variable: true
      distribution: {name: Uniform, domain: {lo: 0.0, hi: 6.283185}}
  output: [theta]
  number_of_modes: 1
execution: {random_seed: 0}
network:
  number_of_nodes: 2
  parameters:
    conduction_speed: {name: conduction_speed, value: BUILD_SPEED, unit: mm_per_ms}
  nodes:
    - {id: 0, label: r0}
    - {id: 1, label: r1}
  edges:
    - {source: 0, target: 1, weight: 0.5, distance: 30.0}
    - {source: 1, target: 0, weight: 0.5, distance: 30.0}
  coupling:
    KuramotoCoupling:
      label: KuramotoCoupling
      delayed: true
      parameters:
        a: {name: a, value: 0.05}
        N: {name: N, value: 1.0}
      pre_expression: {rhs: "sin(theta_j - theta_i)"}
      post_expression: {rhs: "a * gx / N"}
      incoming_states: [theta]
      local_states: [theta]
integration:
  method: RungeKutta4thOrder
  duration: 200.0
  step_size: 1.0
  transient_time: 0.0
functions:
  kuramoto_order:
    description: "Kuramoto order parameter R = <| <exp(i*theta)>_nodes |>_time over the whole window."
    source_code: "jnp.mean(jnp.abs(jnp.mean(jnp.exp(1j * data[:, 0, :]), axis=1)))"
    arguments:
      data:
        description: "Trajectory (time, states, nodes)."
  mse:
    description: "Mean square of the order parameter, a loss whose value depends on the trajectory."
    source_code: "jnp.mean(x ** 2)"
    arguments:
      x:
        description: "The order parameter."
observations:
  order_parameter:
    label: "Kuramoto order parameter"
    source: [theta]
    pipeline:
      - function: kuramoto_order
        arguments:
          data:
            value: integration.result
optimizations:
  fit:
    loss: {function: mse, arguments: {x: {value: observations.order_parameter.data}}}
    free_parameters:
      - parameter: network.conduction_speed
        initial_value: 1.2
        DOMAIN
    algorithm: adam
    learning_rate: 1.0e-12
    max_iterations: 1
"""
"""A 2-node delayed Kuramoto on 30 mm tracts whose fit frees the conduction speed and starts it at 1.2 mm/ms.

At the build speed of 3 mm/ms the delay is 10 ms; at the fit's start it is 25 ms, so a buffer sized for the build speed is 15 ms short. The initial phases are drawn from a Uniform under a fixed seed, so the coupling acts from the first step and a wrong history changes the trajectory. The learning rate is tiny so the one step leaves the speed where it started.
"""


def _experiment(tmp_path, build_speed=3.0, domain="domain: {lo: 1.0, hi: 6.0}"):
    spec = tmp_path / f"fitted_speed_{build_speed}.yaml"
    spec.write_text(_FITTED_SPEED_SPEC.replace("BUILD_SPEED", str(build_speed)).replace("DOMAIN", domain))
    exp = SimulationExperiment.from_file(str(spec))
    exp.configure()
    return exp


def _network_builds(code):
    """The emitted ``create_network(`` calls of the run, the base network's and the fit's own."""
    return "".join(code.split()).split("network=create_network(")[1:]


def _final_loss(result):
    """The fit's last recorded loss, by whichever name the result stores it under."""
    for fit in (getattr(result, "optimizations", None) or {}).values():
        for attr in ("final_loss", "loss", "loss_history"):
            value = getattr(fit, attr, None)
            if value is None:
                continue
            try:
                arr = np.asarray(value, dtype=float).ravel()
            except (TypeError, ValueError):
                continue
            if arr.size:
                return float(arr[-1])
    return None


def test_a_fitted_speed_sizes_the_buffer_from_its_lower_bound(tmp_path):
    """The base network is built for the slowest speed the fit can reach, the domain's 1.0, not the build speed's 3.0."""
    builds = _network_builds(_experiment(tmp_path).render_code("tvboptim"))
    assert builds, "no create_network call found in the emitted run"
    assert "max_delay=float(jnp.max(distances))/1.0*(1.0+1e-4)" in builds[0], builds[0][:300]


def test_a_lower_bound_above_the_build_speed_keeps_the_build_buffer(tmp_path):
    """A fit that cannot go below the build speed needs no longer buffer than the build speed's."""
    builds = _network_builds(_experiment(tmp_path, domain="domain: {lo: 4.0, hi: 6.0}").render_code("tvboptim"))
    assert "max_delay=float(jnp.max(distances))/3.0*(1.0+1e-4)" in builds[0], builds[0][:300]


def test_an_infinite_build_speed_with_a_fitted_speed_renders(tmp_path):
    """A network built at an infinite speed whose fit frees the speed renders: the buffer size is computed from the speed's value, not from the `jnp.inf` the emitted code spells it as."""
    assert _network_builds(_experiment(tmp_path, build_speed=".inf").render_code("tvboptim"))


def test_a_fitted_speed_without_a_lower_bound_is_refused(tmp_path):
    """No positive lower bound, no longest delay to size the buffer from: refused at render, not clamped at run time."""
    with pytest.raises(Exception, match="positive `lo`"):
        _experiment(tmp_path, domain="domain: {hi: 6.0}").render_code("tvboptim")


def test_a_fit_below_the_build_speed_reads_the_right_history(tmp_path):
    """The fit's loss at 1.2 mm/ms is the same whether the network was built at 3.0 or at 1.2.

    Built at 1.2 the buffer holds the 25 ms delay on any tree. Built at 3.0 it held only 10 ms before the fix, the gather clamped, and the two losses differed.
    """
    slow = _final_loss(_experiment(tmp_path, build_speed=1.2).run("tvboptim", mode="optimization"))
    fast = _final_loss(_experiment(tmp_path, build_speed=3.0).run("tvboptim", mode="optimization"))
    assert slow is not None and fast is not None, "the fit recorded no loss the result could read"
    assert np.isfinite(slow), slow
    np.testing.assert_allclose(fast, slow, rtol=0, atol=1e-9)
