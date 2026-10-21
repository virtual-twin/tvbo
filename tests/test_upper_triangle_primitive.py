"""``upper_triangle(M, k)`` gathers the entries on and above a matrix's ``k``-th diagonal, so an FC agreement is declared as ``pearson(upper_triangle(fc, 1), upper_triangle(empirical_fc, 1))``.

The references are built from the definition, an entry ``M[i, j]`` with ``j - i >= k`` taken row by row, and from ``scipy.stats.pearsonr``, never from the ``triu_indices`` call the printers emit.
"""

import numpy as np
import pytest
import scipy.stats

from tvbo.codegen import render_expression
from tvbo.parse.expression import parse_eq

pytestmark = pytest.mark.backend_core

FORMATS = ["jax", "numpy"]


def _code(rhs, fmt, names):
    return render_expression(parse_eq(rhs, parameters=list(names)), format=fmt, parameters=list(names))


def _namespace(fmt, values):
    """The module and the operands *fmt*'s generated code reads, in float64."""
    if fmt == "jax":
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp

        return {"jnp": jnp, **{k: jnp.asarray(v) for k, v in values.items()}}
    return {"np": np, **values}


def _run(rhs, fmt, **values):
    return np.asarray(eval(_code(rhs, fmt, values), _namespace(fmt, values)))


def _by_definition(M, k):
    return np.array([M[i, j] for i in range(M.shape[0]) for j in range(M.shape[1]) if j - i >= k])


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("k", [1, 0, 2, -1])
@pytest.mark.parametrize("shape", [(5, 5), (4, 6), (6, 4)])
def test_the_entries_above_the_kth_diagonal_row_by_row(fmt, k, shape):
    M = np.random.default_rng(0).standard_normal(shape)

    np.testing.assert_array_equal(_run(f"upper_triangle(M, {k})", fmt, M=M), _by_definition(M, k))


@pytest.fixture(scope="module")
def fcs():
    """A simulated and an empirical FC: symmetric, unit diagonal, from two related noisy signals."""
    rng = np.random.default_rng(1)
    shared = rng.standard_normal((400, 8))
    return np.corrcoef(shared + 0.8 * rng.standard_normal((400, 8)), rowvar=False), np.corrcoef(
        shared + 0.8 * rng.standard_normal((400, 8)), rowvar=False
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_an_fc_agreement_correlates_the_off_diagonal_half_once(fmt, fcs):
    """The diagonal is 1 in both and the lower half mirrors the upper, so either one left in would change the score."""
    fc, empirical_fc = fcs
    off_diagonal = np.triu(np.ones_like(fc, dtype=bool), 1)

    got = _run("pearson(upper_triangle(fc, 1), upper_triangle(empirical_fc, 1))", fmt, fc=fc, empirical_fc=empirical_fc)

    np.testing.assert_allclose(got, scipy.stats.pearsonr(fc[off_diagonal], empirical_fc[off_diagonal])[0], rtol=1e-12)
    assert not np.isclose(got, scipy.stats.pearsonr(fc.ravel(), empirical_fc.ravel())[0]), (
        "the diagonal and the mirrored half were included"
    )


def test_it_traces_under_jit(fcs):
    """``k`` and the matrix's shape are static, so the gather compiles."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    fc, empirical_fc = fcs
    code = _code("pearson(upper_triangle(fc, 1), upper_triangle(empirical_fc, 1))", "jax", ["fc", "empirical_fc"])
    score = jax.jit(lambda a, b: eval(code, {"jnp": jnp, "fc": a, "empirical_fc": b}))

    np.testing.assert_allclose(
        score(fc, empirical_fc),
        _run("pearson(upper_triangle(fc, 1), upper_triangle(empirical_fc, 1))", "numpy", fc=fc, empirical_fc=empirical_fc),
        rtol=1e-12,
    )


def test_a_k_that_is_not_an_integer_literal_is_refused():
    with pytest.raises(ValueError, match=r"upper_triangle\(M, k\): k must be an integer literal"):
        _code("upper_triangle(M, k)", "numpy", ["M", "k"])
