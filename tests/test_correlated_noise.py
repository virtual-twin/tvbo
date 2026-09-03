"""Correlated noise: a declared covariance is imposed on the Wiener increments.

These pin the contract `Noise.covariance` + `Noise.correlated_over` promise — that the
*sampled* increments really carry the declared second-order structure — rather than
merely that a run completes. A diagonal approximation would pass a "does it run" check while quietly changing the science.

The mechanism is a solver wrapper (tvbo's concrete implementation against tvboptim's `NativeSolver` contract), so the tests exercise it through `step`, which is the one path every network shape shares.
"""

import numpy as np
import pytest

from tvbo.classes.correlated_noise import (
    CorrelatedNoiseSolver,
    covariance_factor,
    noise_mixer,
)


def _psd(n, rank=None, seed=0):
    """A random symmetric PSD matrix, optionally rank-deficient."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, rank if rank is not None else n))
    return A @ A.T


# ------------------------------------------------------------------ factorisation


def test_factor_reproduces_the_covariance():
    C = _psd(6)
    L = covariance_factor(C)
    assert np.allclose(L @ L.T, C, atol=1e-10)


def test_rank_deficient_covariance_is_accepted():
    """Fewer independent sources than elements is a legitimate covariance, not an error."""
    C = _psd(6, rank=3)
    L = covariance_factor(C)
    assert np.allclose(L @ L.T, C, atol=1e-8)
    assert np.linalg.matrix_rank(C, tol=1e-8) == 3


def test_asymmetric_matrix_is_rejected_with_a_pointed_message():
    C = _psd(4)
    C[0, 1] += 1.0
    with pytest.raises(ValueError, match="not symmetric"):
        covariance_factor(C)


def test_indefinite_matrix_is_rejected():
    C = _psd(4)
    C -= 2.0 * np.abs(np.linalg.eigvalsh(C)).max() * np.eye(4)
    with pytest.raises(ValueError, match="not positive semi-definite"):
        covariance_factor(C)


def test_non_square_is_rejected():
    with pytest.raises(ValueError, match="square"):
        covariance_factor(np.zeros((3, 4)))


# ----------------------------------------------------------------------- sampling


def test_sampled_increments_carry_the_declared_covariance():
    """The whole point: empirical covariance of the mixed draws converges to C."""
    jax = pytest.importorskip("jax")

    n_nodes, n_states, n_draws = 5, 2, 200_000
    C = _psd(n_nodes, seed=3)
    mix = noise_mixer(covariance_factor(C), "node")

    xi = jax.random.normal(jax.random.key(0), (n_draws, n_states, n_nodes))
    mixed = np.asarray(mix(xi))

    for s in range(n_states):
        emp = np.cov(mixed[:, s, :], rowvar=False)
        assert np.allclose(emp, C, rtol=0.05, atol=0.05 * np.abs(C).max())


def test_state_axis_correlates_states_not_nodes():
    """`correlated_over: state` must index the state axis, not silently the node axis."""
    jax = pytest.importorskip("jax")

    n_states, n_nodes, n_draws = 3, 4, 200_000
    C = _psd(n_states, seed=5)
    mix = noise_mixer(covariance_factor(C), "state")

    xi = jax.random.normal(jax.random.key(1), (n_draws, n_states, n_nodes))
    mixed = np.asarray(mix(xi))

    for node in range(n_nodes):
        emp = np.cov(mixed[:, :, node], rowvar=False)
        assert np.allclose(emp, C, rtol=0.05, atol=0.05 * np.abs(C).max())


def test_identity_covariance_leaves_increments_independent():
    """The correlated path must reduce exactly to the uncorrelated one."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    n = 4
    mix = noise_mixer(covariance_factor(np.eye(n)), "node")
    xi = jax.random.normal(jax.random.key(1), (1000, 2, n))
    assert jnp.allclose(mix(xi), xi, atol=1e-6)


def test_leading_axes_are_untouched():
    """Mixing acts on the correlated axis only; time/block axes pass through."""
    jax = pytest.importorskip("jax")

    C = _psd(3, seed=7)
    mix = noise_mixer(covariance_factor(C), "node")
    xi = jax.random.normal(jax.random.key(2), (9, 2, 3))
    assert np.asarray(mix(xi)).shape == (9, 2, 3)


def test_unknown_axis_is_rejected():
    with pytest.raises(ValueError, match="not an axis"):
        noise_mixer(np.eye(3), "frequency")


def test_mode_axis_is_rejected_with_an_explanation():
    """`mode` is in the schema's vocabulary but is not an axis of the increment."""
    with pytest.raises(NotImplementedError, match="folded into the state"):
        noise_mixer(np.eye(3), "mode")


# ------------------------------------------------------------------------- solver


def _euler():
    solvers = pytest.importorskip("tvboptim.experimental.network_dynamics.solvers")
    return solvers.Euler()


def _zero_dynamics(t, state, params):
    """No drift, no auxiliaries — isolates what the wrapper does to the increment."""
    import jax.numpy as jnp

    return jnp.zeros_like(state), None


def test_solver_delegates_and_imposes_the_covariance():
    """One `step` must equal the base solver's step on the *mixed* increment."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    base = _euler()
    C = _psd(4, seed=11)
    L = covariance_factor(C)
    solver = CorrelatedNoiseSolver(base, L, axis="node")

    state = jnp.zeros((2, 4))
    xi = jax.random.normal(jax.random.key(3), (2, 4))

    got, _ = solver.step(_zero_dynamics, 0.0, state, 0.1, {}, xi)
    want, _ = base.step(_zero_dynamics, 0.0, state, 0.1, {}, noise_mixer(L, "node")(xi))
    assert jnp.allclose(got, want, atol=1e-6)


def test_solver_with_identity_covariance_is_a_no_op():
    """Switching correlation on must change the distribution, nothing else."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    base = _euler()
    solver = CorrelatedNoiseSolver(base, covariance_factor(np.eye(4)), axis="node")

    state = jnp.zeros((2, 4))
    xi = jax.random.normal(jax.random.key(4), (2, 4))

    got, _ = solver.step(_zero_dynamics, 0.0, state, 0.1, {}, xi)
    want, _ = base.step(_zero_dynamics, 0.0, state, 0.1, {}, xi)
    assert jnp.allclose(got, want, atol=1e-6)


def test_solver_passes_a_noiseless_step_through():
    """An ODE step carries the scalar 0.0 as its increment and must survive untouched."""
    pytest.importorskip("jax")
    import jax.numpy as jnp

    base = _euler()
    solver = CorrelatedNoiseSolver(base, covariance_factor(_psd(4, seed=2)), axis="node")

    state = jnp.ones((2, 4))
    got, _ = solver.step(_zero_dynamics, 0.0, state, 0.1, {})
    want, _ = base.step(_zero_dynamics, 0.0, state, 0.1, {})
    assert jnp.allclose(got, want, atol=1e-6)


def test_solver_delegates_scan_settings_to_the_wrapped_solver():
    """Wrapping must not silently drop a block size or a gradient horizon."""
    solvers = pytest.importorskip("tvboptim.experimental.network_dynamics.solvers")

    base = solvers.Euler(block_size=17)
    solver = CorrelatedNoiseSolver(base, covariance_factor(np.eye(3)), axis="node")

    assert solver.block_size == 17
    assert solver.grad_horizon == base.grad_horizon
    assert solver.stage_time_centroid == base.stage_time_centroid
    assert solver.recompute_coupling_per_stage == base.recompute_coupling_per_stage


def test_per_group_factors_only_touch_their_own_group():
    """A heterogeneous network: a group declaring no covariance keeps iid increments."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    base = _euler()
    C = _psd(3, seed=13)
    L = covariance_factor(C)
    solver = CorrelatedNoiseSolver(base, {"cortex": L}, axis="node")

    xi = {
        "cortex": jax.random.normal(jax.random.key(5), (2, 3)),
        "relay": jax.random.normal(jax.random.key(6), (2, 3)),
    }
    mixed = solver._mix(xi)

    assert jnp.allclose(mixed["cortex"], noise_mixer(L, "node")(xi["cortex"]), atol=1e-6)
    assert jnp.allclose(mixed["relay"], xi["relay"], atol=1e-6)


def test_covariance_sized_to_the_wrong_axis_is_rejected():
    """A shape mismatch must name the problem, not fail deep inside the scan."""
    jax = pytest.importorskip("jax")

    solver = CorrelatedNoiseSolver(_euler(), covariance_factor(np.eye(3)), axis="node")
    xi = jax.random.normal(jax.random.key(7), (2, 5))
    with pytest.raises(ValueError, match="must be square in the axis"):
        solver._mix(xi)
