"""General array-op primitives for per-timestep detectors / permutation tests.

``take`` (2-D gather), ``sum_axis`` (single-axis reduction), and ``pearson`` (node-collapsing correlation) are backend-abstracted printer primitives, so a
wave / graph / significance observable can be authored as declarative equations instead of backend ``source_code``. Each parses to a SymPy Function and prints to
the backend's array algebra; here we render for jax + numpy and execute against a numpy reference.
"""

import numpy as np
import pytest

from tvbo.parse.expression import parse_eq
from tvbo.codegen import render_expression


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
    """The axis-1 masked-mean the wave detector uses composes from the primitives:
    sum_axis(ang*nbr_mask, 1) / deg  ==  (ang*nbr_mask).sum(1)/deg."""
    import jax.numpy as jnp

    code = _render("sum_axis(ang * nbr_mask, 1) / deg", "jax", ["ang", "nbr_mask", "deg"])
    assert code.replace(" ", "") == "jnp.sum(ang*nbr_mask,axis=1)/deg"
    rng = np.random.default_rng(1)
    ang = rng.standard_normal((5, 3))
    nbr_mask = (rng.random((5, 3)) > 0.3).astype(float)
    deg = nbr_mask.sum(1)
    got = eval(code, {"jnp": jnp}, {"ang": jnp.array(ang), "nbr_mask": jnp.array(nbr_mask), "deg": jnp.array(deg)})
    assert np.allclose(np.asarray(got), (ang * nbr_mask).sum(1) / deg)
