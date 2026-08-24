"""Structural array-op codegen primitives — one declarative op, every backend.

These ops let an observation pipeline that trims a transient, selects a variable of interest, or downsamples be authored as a **declarative equation** rather than backend ``source_code``. Each op is a single backend-agnostic handler in ``tvbo.codegen.code`` that renders through per-backend rendering primitives:

- numpy / jax: Python 0-based slicing (``x[::step]``, ``jnp.take(x, arange, axis)``)
- julia:       1-based, ``end``-relative indexing (``x[1:step:end]``, ``selectdim``)

The two invariants under test:

1. **Cross-backend rendering** — each op emits the idiomatic form for numpy, jax, julia.
2. **Numeric byte-identity** — the generated numpy/jax code produces values *identical* to the hand-written slicing it replaces (so migrating ``source_code`` -> declarative equation changes nothing numerically).
"""

import numpy as np
import pytest

from tvbo.codegen import render_expression


# 1. Cross-backend rendering
@pytest.mark.parametrize(
    ("expr", "numpy", "jax", "julia"),
    [
        # strided downsample of the leading (time) axis
        ("subsample(x, 1000)", "x[::1000]", "x[::1000]", "x[1:1000:end]"),
        ("subsample(x, 5, 1000)", "x[5::1000]", "x[5::1000]", "x[5 + 1:1000:end]"),
        # bounded slice of one axis (keeps ndim)
        (
            "slice_axis(x, 1, 0, 2)",
            "np.take(x, np.arange(0, 2), axis=1)",
            "jnp.take(x, jnp.arange(0, 2), axis=1)",
            "selectdim(x, 2, 0 + 1:2)",
        ),
        (
            "slice_axis(x, 1, 0, 10, 2)",
            "np.take(x, np.arange(0, 10, 2), axis=1)",
            "jnp.take(x, jnp.arange(0, 10, 2), axis=1)",
            "selectdim(x, 2, 0 + 1:2:10)",
        ),
        # open-ended slice of one axis (transient trimming)
        (
            "slice_from(x, 0, 100)",
            "np.take(x, np.arange(100, x.shape[0]), axis=0)",
            "jnp.take(x, jnp.arange(100, x.shape[0]), axis=0)",
            "selectdim(x, 1, 100 + 1:size(x, 1))",
        ),
        # axis length
        ("shape(x, 1)", "x.shape[1]", "x.shape[1]", "size(x, 2)"),
    ],
)
def test_array_op_renders_per_backend(expr, numpy, jax, julia):
    assert render_expression(expr, format="numpy") == numpy
    assert render_expression(expr, format="jax") == jax
    assert render_expression(expr, format="julia") == julia


def test_piecewise_backend_abstracted():
    """Piecewise renders through the shared ``_where3`` primitive: where vs ifelse."""
    expr = "Piecewise((1, x > 0), (0, True))"
    assert render_expression(expr, format="numpy") == "np.where(np.greater(x, 0), 1, 0)"
    assert render_expression(expr, format="jax") == "jnp.where(jnp.greater(x, 0), 1, 0)"
    assert render_expression(expr, format="julia") == "ifelse(x > 0, 1, 0)"


def test_ops_compose():
    """Structural ops nest — select a voi, then downsample (Hopf-BOLD flavour)."""
    code = render_expression("subsample(slice_axis(data, 1, 0, 1), 1000)", format="jax")
    assert code == "jnp.take(data, jnp.arange(0, 1), axis=1)[::1000]"


# 2. Numeric byte-identity vs the hand-written slicing each op replaces
@pytest.mark.parametrize("fmt", ["numpy", "jax"])
def test_new_ops_numerically_identical(fmt):
    if fmt == "jax":
        jnp = pytest.importorskip("jax.numpy")
        mod, xp = jnp, jnp
    else:
        mod, xp = np, np

    rs = np.random.RandomState(0)
    x = xp.asarray(rs.randn(20000, 3, 5))  # (time, voi, nodes)
    ns = {"np": np, "jnp": mod, "x": x}

    def gen(expr):
        return eval(render_expression(expr, format=fmt), ns)

    # subsample(x, step) == x[::step]
    assert bool(xp.array_equal(gen("subsample(x, 1000)"), x[::1000]))
    # subsample(x, start, step) == x[start::step]
    assert bool(xp.array_equal(gen("subsample(x, 5, 1000)"), x[5::1000]))
    # slice_axis(x, 1, 0, 1) == x[:, 0:1, :]  (keeps ndim)
    assert bool(xp.array_equal(gen("slice_axis(x, 1, 0, 1)"), x[:, 0:1, :]))
    # slice_axis(x, 1, 0, 3, 2) == x[:, 0:3:2, :]
    assert bool(xp.array_equal(gen("slice_axis(x, 1, 0, 3, 2)"), x[:, 0:3:2, :]))
    # slice_from(x, 0, 100) == x[100:]  (transient trim)
    assert bool(xp.array_equal(gen("slice_from(x, 0, 100)"), x[100:]))
    # shape(x, 1) == x.shape[1]
    assert gen("shape(x, 1)") == x.shape[1]
    # composition: select voi 0 then subsample == x[:, 0:1, :][::1000]
    assert bool(xp.array_equal(gen("subsample(slice_axis(x, 1, 0, 1), 1000)"), x[:, 0:1, :][::1000]))
