"""Graph-construction primitives render and compute correctly on every backend.

These are the vocabulary a ``Procedural`` GraphGenerator's typed DAG lowers to, so a
paper's network construction (distance kernel, stochastic connection mask, Gaussian
field, axis normalisation) is authored as metadata and emitted natively per backend
instead of living in per-generator Python.

Two things are pinned here:

* **Numerical agreement with the reference implementation.** Each deterministic
  primitive is checked against the scipy/numpy routine a hand-written generator would
  have called, so "expressible in the primitive set" also means "computes the same
  thing".
* **numpy/jax agreement.** The same expression must produce the same array on both, to
  float tolerance — that is what makes a generator backend-independent rather than
  merely re-implemented twice.

The samplers are the exception, and deliberately so: per the RNG contract in
``dev/GenericProcedureEngine.md`` §4, a fixed seed is reproducible *within* a backend and
statistically equivalent *across* backends, but never bit-identical (numpy PCG64 is not
jax Threefry). They are therefore checked for shape, support, distributional moments and
within-backend reproducibility — never for cross-backend equality.
"""

import numpy as np
import pytest

from tvbo.codegen.code import render_expression

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

RNG = np.random.default_rng(20240720)


def _eval_numpy(src, **env):
    return eval(src, {"np": np, "jax": jax}, dict(env))  # noqa: S307 — rendered by us


def _eval_jax(src, **env):
    return np.asarray(
        eval(src, {"jnp": jnp, "jax": jax}, {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v
                                             for k, v in env.items()})  # noqa: S307
    )


def _both(expr, **env):
    """Render `expr` for numpy and jax, evaluate both, assert they agree, return numpy's."""
    out_np = _eval_numpy(render_expression(expr, format="numpy"), **env)
    out_jx = _eval_jax(render_expression(expr, format="jax"), **env)
    np.testing.assert_allclose(np.asarray(out_np), out_jx, rtol=1e-6, atol=1e-6)
    return np.asarray(out_np)


# --------------------------------------------------------------------------- #
# Deterministic primitives                                                     #
# --------------------------------------------------------------------------- #
def test_grid_positions_reproduces_the_reference_node_ORDER():
    """Node order is the contract, not an implementation detail.

    A connectome's row order *is* its node identity, so a layout that produced the same
    coordinate set in a different order would silently permute every per-node quantity
    downstream (in-strength gradients, region labels, recorded traces) while every
    aggregate check still passed. This pins byte-equality against the reference
    meshgrid construction, not just set equality.
    """
    nx, ny, xe, ye = 30, 30, 140.0, 140.0
    got = _both("grid_positions(nx, ny, xe, ye)", nx=nx, ny=ny, xe=xe, ye=ye)
    x, y = np.linspace(0.0, xe, nx), np.linspace(0.0, ye, ny)
    expected = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    assert got.shape == (nx * ny, 2)
    np.testing.assert_array_equal(got, expected)


def test_grid_positions_spans_the_requested_extent():
    got = _both("grid_positions(nx, ny, xe, ye)", nx=4, ny=5, xe=10.0, ye=20.0)
    assert got[:, 0].min() == pytest.approx(0.0) and got[:, 0].max() == pytest.approx(10.0)
    assert got[:, 1].min() == pytest.approx(0.0) and got[:, 1].max() == pytest.approx(20.0)


@pytest.mark.parametrize("nx, ny", [(1, 5), (5, 1), (1, 1)])
def test_grid_positions_handles_a_degenerate_axis(nx, ny):
    """A 1-D lattice (a line of nodes) is a legitimate layout, not an error.

    Spacing is extent / (n - 1), which divides by zero when an axis has one node. The
    reference `linspace(0, extent, 1)` returns [0.0] with no division, so this used to
    crash on a layout the thing it replaces handles.
    """
    got = _both("grid_positions(nx, ny, xe, ye)", nx=nx, ny=ny, xe=140.0, ye=140.0)
    assert got.shape == (nx * ny, 2)
    assert np.isfinite(got).all()
    x, y = np.linspace(0.0, 140.0, nx), np.linspace(0.0, 140.0, ny)
    np.testing.assert_array_equal(got, np.array(np.meshgrid(x, y)).T.reshape(-1, 2))


def test_normalize_leaves_an_all_zero_slice_as_zeros():
    """An unconnected node must not poison the connectome with NaN.

    A stochastic connection mask routinely leaves a peripheral node with no incoming
    edges, giving an all-zero column. Dividing by that sum yields NaN, which then
    propagates silently through every downstream step into the weights.
    """
    M = np.array([[0.0, 1.0], [0.0, 2.0]])  # column 0: a node with no incoming edges
    got = _both("normalize(M, 0)", M=M)
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got[:, 0], [0.0, 0.0])
    np.testing.assert_allclose(got[:, 1], [1 / 3, 2 / 3])


def test_minmax_rescale_maps_a_constant_field_to_the_midpoint():
    """A flat field has zero span; it must stay neutral rather than become NaN.

    Rescaling a constant to [-1, 1] gives 0 — which is what a hand-written generator
    special-cases a zero-span gradient to, so a degenerate field stays neutral instead
    of NaN-ing every node's in-strength.
    """
    got = _both("minmax_rescale(x, -1, 1)", x=np.full(5, 3.0))
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got, np.zeros(5))


def test_minmax_rescale_constant_field_respects_an_asymmetric_range():
    got = _both("minmax_rescale(x, 0, 10)", x=np.full(4, -2.0))
    np.testing.assert_allclose(got, np.full(4, 5.0))


def test_pairwise_distance_matches_scipy_cdist():
    scipy_spatial = pytest.importorskip("scipy.spatial")
    pos = RNG.normal(size=(12, 2)) * 40.0
    got = _both("pairwise_distance(pos)", pos=pos)
    expected = scipy_spatial.distance.cdist(pos, pos)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_fill_diagonal_matches_numpy_fill_diagonal():
    M = RNG.normal(size=(6, 6))
    got = _both("fill_diagonal(M, 0)", M=M)
    expected = M.copy()
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_fill_diagonal_leaves_off_diagonal_untouched():
    """The diagonal is replaced; nothing else may move (a generator relies on this to
    suppress self-connections without perturbing the kernel)."""
    M = RNG.normal(size=(5, 5))
    got = _both("fill_diagonal(M, 7)", M=M)
    off = ~np.eye(5, dtype=bool)
    np.testing.assert_allclose(got[off], M[off], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.diag(got), np.full(5, 7.0), rtol=1e-12, atol=1e-12)


def test_gaussian_pdf_matches_scipy_multivariate_normal():
    scipy_stats = pytest.importorskip("scipy.stats")
    pos = RNG.normal(size=(15, 2)) * 30.0
    mean, cov = (40.0, 40.0), 300.0
    got = _both("gaussian_pdf(pos, mu, cov)", pos=pos, mu=np.asarray(mean), cov=cov)
    expected = scipy_stats.multivariate_normal(np.asarray(mean), cov).pdf(pos)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("axis", [0, 1])
def test_normalize_divides_by_axis_sum(axis):
    M = RNG.uniform(0.1, 1.0, size=(7, 7))
    got = _both(f"normalize(M, {axis})", M=M)
    np.testing.assert_allclose(got, M / M.sum(axis=axis, keepdims=True), rtol=1e-9)
    np.testing.assert_allclose(got.sum(axis=axis), np.ones(7), rtol=1e-9)


def test_minmax_rescale_hits_the_target_range():
    x = RNG.normal(size=40) * 5.0
    got = _both("minmax_rescale(x, -1, 1)", x=x)
    assert got.min() == pytest.approx(-1.0)
    assert got.max() == pytest.approx(1.0)
    # Affine: the rank order and relative spacing are preserved.
    np.testing.assert_allclose(
        (got - got.min()) / (got.max() - got.min()),
        (x - x.min()) / (x.max() - x.min()),
        rtol=1e-9,
    )


def test_eigvals_matches_numpy_linalg():
    M = RNG.normal(size=(8, 8))
    got = np.sort_complex(_eval_numpy(render_expression("eigvals(M)", format="numpy"), M=M))
    np.testing.assert_allclose(got, np.sort_complex(np.linalg.eigvals(M)), rtol=1e-6)


def test_spectral_radius_composes_from_primitives():
    """max(abs(eigvals(M))) — the reservoir rescaling — is expressible without new heads."""
    M = RNG.normal(size=(9, 9))
    src = render_expression("max(abs(eigvals(M)))", format="numpy")
    got = _eval_numpy(src, M=M)
    assert got == pytest.approx(np.max(np.abs(np.linalg.eigvals(M))))


# --------------------------------------------------------------------------- #
# Samplers — within-backend reproducibility, never cross-backend equality      #
# --------------------------------------------------------------------------- #
def test_sample_exponential_shape_support_and_mean():
    src = render_expression("sample_exponential(key, 0, 17.0, n, n)", format="jax")
    got = _eval_jax(src, key=jax.random.key(0), n=64)
    assert got.shape == (64, 64)
    assert (got >= 0).all()
    # Exponential(scale=17) has mean 17; 4096 draws puts the SE near 17/64 ~ 0.27.
    assert got.mean() == pytest.approx(17.0, rel=0.05)


def test_sample_normal_moments():
    src = render_expression("sample_normal(key, 0, 3.0, 2.0, n, n)", format="jax")
    got = _eval_jax(src, key=jax.random.key(1), n=64)
    assert got.mean() == pytest.approx(3.0, abs=0.1)
    assert got.std() == pytest.approx(2.0, rel=0.05)


def test_sample_uniform_support():
    src = render_expression("sample_uniform(key, 0, -1.0, 1.0, n, n)", format="jax")
    got = _eval_jax(src, key=jax.random.key(2), n=32)
    assert got.min() >= -1.0 and got.max() <= 1.0
    assert got.mean() == pytest.approx(0.0, abs=0.05)


def test_same_key_reproduces_within_backend():
    """The contract we DO guarantee: one backend, one key, one draw."""
    src = render_expression("sample_exponential(key, 0, 17.0, n, n)", format="jax")
    a = _eval_jax(src, key=jax.random.key(7), n=16)
    b = _eval_jax(src, key=jax.random.key(7), n=16)
    np.testing.assert_array_equal(a, b)


def test_different_keys_give_different_draws():
    """A per-cell key must actually change the realisation, or an ensemble is a no-op."""
    src = render_expression("sample_exponential(key, 0, 17.0, n, n)", format="jax")
    a = _eval_jax(src, key=jax.random.key(7), n=16)
    b = _eval_jax(src, key=jax.random.key(8), n=16)
    assert np.abs(a - b).max() > 0


def test_fold_in_derives_independent_substreams():
    """fold_in(key, i) is how a per-cell seed becomes a per-cell network realisation."""
    src = render_expression("sample_exponential(key, 0, 17.0, n, n)", format="jax")
    base = jax.random.key(42)
    draws = [_eval_jax(src, key=jax.random.fold_in(base, i), n=16) for i in range(4)]
    for i in range(len(draws)):
        for j in range(i + 1, len(draws)):
            assert np.abs(draws[i] - draws[j]).max() > 0


# --------------------------------------------------------------------------- #
# The composite: Koller2024's stochastic connection mask                       #
# --------------------------------------------------------------------------- #
def test_connection_mask_composes_from_primitives():
    """`d <= abs(Exponential(scale))` — Koller2024's mask — needs no bespoke head.

    A comparison parses to a SymPy relational and prints on every backend, so
    `stochastic_mask` is a typed DAG step that LOWERS to this expression rather than a
    primitive of its own.
    """
    expr = "d <= abs(sample_exponential(key, 0, 17.0, n, n))"
    src = render_expression(expr, format="jax")
    pos = np.stack(np.meshgrid(np.linspace(0, 140, 8), np.linspace(0, 140, 8)), -1).reshape(-1, 2)
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    mask = _eval_jax(src, d=d, key=jax.random.key(0), n=len(pos))
    assert mask.shape == d.shape
    assert mask.dtype == bool
    # Distance-dependent: near pairs connect far more often than distant ones.
    near, far = d < 20.0, d > 100.0
    assert mask[near].mean() > mask[far].mean()
