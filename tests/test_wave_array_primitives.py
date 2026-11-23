"""General array-op primitives for per-timestep detectors / permutation tests.

``take`` (2-D gather), ``sum_axis`` (single-axis reduction), and ``pearson`` (node-collapsing correlation) are backend-abstracted printer primitives, so a wave / graph / significance observable can be authored as declarative equations instead of backend ``source_code``. ``scatter_sum`` (a face field summed onto its vertices) and ``spearman`` (a rank correlation over the entries a mask keeps) complete the chain of a mesh wave detector, and a comparison that arithmetic applies to, ``(p <= alpha) * mask``, reads as the 0/1 field it denotes. Each parses to a SymPy Function and prints to the backend's array algebra; here we render for jax + numpy and execute against a numpy reference.
"""

import numpy as np
import pytest

from tvbo.codegen import render_expression
from tvbo.datamodel.schema import DerivedVariable, Dynamics, Equation, Observation, Parameter, StateVariable
from tvbo.parse.expression import parse_eq
from tvbo.templates.tvboptim.utils import resolve_reduction


def _render(rhs, fmt, params):
    return render_expression(parse_eq(rhs, parameters=params), format=fmt, parameters=params)


def _env(fmt):
    """Eval namespace + array constructor for a backend format."""
    if fmt == "jax":
        import jax.numpy as jnp

        return {"jnp": jnp}, jnp.array
    return {"np": np}, np.array


@pytest.mark.parametrize("fmt,mod", [("jax", "jnp"), ("numpy", "np")])
def test_take_is_a_2d_gather(fmt, mod):
    code = _render("take(x, idx)", fmt, ["x", "idx"])
    assert code == f"{mod}.take(x, idx)"
    env, arr = _env(fmt)
    x = np.array([10.0, 20, 30, 40])
    idx = np.array([[0, 1], [2, 0], [3, 3]])
    got = eval(code, env, {"x": arr(x), "idx": arr(idx)})
    assert np.allclose(np.asarray(got), x[idx])  # == fancy indexing x[idx]


@pytest.mark.parametrize("fmt,mod", [("jax", "jnp"), ("numpy", "np")])
def test_sum_axis_reduces_one_axis(fmt, mod):
    code = _render("sum_axis(x, 1)", fmt, ["x"])
    assert code == f"{mod}.sum(x, axis=1)"
    env, arr = _env(fmt)
    X = np.arange(12.0).reshape(3, 4)
    got = eval(code, env, {"x": arr(X)})
    assert np.allclose(np.asarray(got), X.sum(axis=1))


def test_pearson_matches_numpy_corrcoef():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    code = _render("pearson(a, b)", "jax", ["a", "b"])
    # expands to reduction primitives (backend-agnostic), not a bespoke call
    assert "mean" in code and "sqrt" in code and "sum" in code
    rng = np.random.default_rng(0)
    a = rng.standard_normal(50)
    b = 0.6 * a + 0.4 * rng.standard_normal(50)
    got = float(eval(code, {"jnp": jnp}, {"a": jnp.array(a), "b": jnp.array(b)}))
    assert abs(got - float(np.corrcoef(a, b)[0, 1])) < 1e-12  # exact in float64


@pytest.mark.parametrize("fmt,mod", [("jax", "jnp"), ("numpy", "np")])
def test_clip_any_all_render(fmt, mod):
    """clip/any/all complete the per-timestep detector vocabulary (clip before acos, any over a significance mask)."""
    assert _render("clip(x, -1, 1)", fmt, ["x"]) == f"{mod}.clip(x, -1, 1)"
    assert _render("any(x)", fmt, ["x"]) == f"{mod}.any(x)"
    assert _render("all(x)", fmt, ["x"]) == f"{mod}.all(x)"
    env, arr = _env(fmt)
    x = np.array([-2.0, 0.5, 3.0])
    assert np.allclose(np.asarray(eval(_render("clip(x, -1, 1)", fmt, ["x"]), env, {"x": arr(x)})), np.clip(x, -1, 1))
    assert bool(eval(_render("any(x > 0)", fmt, ["x"]), env, {"x": arr(x)})) is True


def test_masked_mean_form_composes():
    """The axis-1 masked-mean the wave detector uses composes from the primitives.

    ``sum_axis(ang*nbr_mask, 1) / deg`` == ``(ang*nbr_mask).sum(1)/deg``.
    """
    import jax.numpy as jnp

    code = _render("sum_axis(ang * nbr_mask, 1) / deg", "jax", ["ang", "nbr_mask", "deg"])
    assert code.replace(" ", "") == "jnp.sum(ang*nbr_mask,axis=1)/deg"
    rng = np.random.default_rng(1)
    ang = rng.standard_normal((5, 3))
    nbr_mask = (rng.random((5, 3)) > 0.3).astype(float)
    deg = nbr_mask.sum(1)
    got = eval(code, {"jnp": jnp}, {"ang": jnp.array(ang), "nbr_mask": jnp.array(nbr_mask), "deg": jnp.array(deg)})
    assert np.allclose(np.asarray(got), (ang * nbr_mask).sum(1) / deg)


@pytest.mark.parametrize("fmt", ["jax", "numpy"])
def test_scatter_sum_matches_add_at(fmt):
    """Each slot sums every value whose index names it, a repeated index adding each time and a slot no index names staying 0, as `numpy.add.at` accumulates."""
    faces = np.array([[0, 1, 2], [2, 1, 3], [3, 1, 0]])
    values = np.arange(9.0).reshape(3, 3)
    expected = np.zeros(5)
    np.add.at(expected, faces, values)
    code = _render("scatter_sum(values, faces, 5)", fmt, ["values", "faces"])
    env, arr = _env(fmt)

    got = eval(code, env, {"values": arr(values), "faces": arr(faces)})

    np.testing.assert_allclose(np.asarray(got), expected)


@pytest.mark.parametrize("fmt", ["jax", "numpy"])
def test_spearman_masked_matches_scipy(fmt):
    """Spearman's rho over the kept entries is `scipy.stats.spearmanr` of those entries alone, ties included; what a left-out entry holds, NaN or an outlier, changes nothing."""
    from scipy.stats import spearmanr

    rng = np.random.default_rng(3)
    x = rng.integers(0, 6, 40).astype(float)
    y = x + rng.integers(0, 4, 40)
    keep = rng.random(40) < 0.7
    expected = spearmanr(x[keep], y[keep]).statistic
    x[~keep], y[~keep] = np.nan, 1e9
    code = _render("spearman(x, y, keep)", fmt, ["x", "y", "keep"])
    env, arr = _env(fmt)

    got = eval(code, env, {"x": arr(x), "y": arr(y), "keep": arr(keep)})

    assert float(got) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("fmt", ["jax", "numpy"])
def test_relational_times_mask(fmt):
    """A comparison that arithmetic applies to is 1 where it holds and 0 where not, so `(p <= alpha) * mask` is the masked indicator numpy computes."""
    p = np.array([0.01, 0.2, 0.04, 0.5])
    mask = np.array([1.0, 1.0, 0.0, 1.0])
    code = _render("(p <= alpha) * mask", fmt, ["p", "alpha", "mask"])
    env, arr = _env(fmt)

    got = eval(code, env, {"p": arr(p), "alpha": 0.05, "mask": arr(mask)})

    np.testing.assert_array_equal(np.asarray(got), (p <= 0.05) * mask)


@pytest.mark.parametrize("fmt", ["jax", "numpy"])
def test_an_equality_times_a_mask_compares_values(fmt):
    """`==` in arithmetic compares the values entry by entry, not the two symbols, which SymPy alone would find unequal."""
    k = np.array([0.0, 1.0, 2.0, 1.0])
    mask = np.array([1.0, 1.0, 1.0, 0.0])
    code = _render("(k == level) * mask", fmt, ["k", "level", "mask"])
    env, arr = _env(fmt)

    got = eval(code, env, {"k": arr(k), "level": 1.0, "mask": arr(mask)})

    np.testing.assert_array_equal(np.asarray(got), (k == 1.0) * mask)


def test_an_observer_reads_a_comparison_times_a_mask_as_parse_eq_does():
    """An observer parses its per-step chain with its own vocabulary, and lowers the comparison the same way."""
    obs = Observation(
        name="hits",
        source=["p"],
        dynamics=Dynamics(
            name="hits",
            parameters={"alpha": Parameter(name="alpha", value=0.05), "mask": Parameter(name="mask", value=[1.0, 0.0])},
            state_variables={"acc": StateVariable(name="acc", equation=Equation(rhs="acc + p"), equation_type="recurrence")},
            derived_variables={"hit": DerivedVariable(name="hit", equation=Equation(rhs="(p <= alpha) * mask"))},
            output=["hit"],
        ),
    )

    (hit,) = resolve_reduction(obs)["derived"]

    assert hit["expr"] == parse_eq("(p <= alpha) * mask", parameters=["p", "alpha", "mask"])
