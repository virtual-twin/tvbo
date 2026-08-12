"""The ``mode_dot`` codegen primitive — mode-axis contraction for multi-mode models.

``mode_dot(X, M)`` contracts a state variable's mode axis with a (modes x modes)
matrix, i.e. TVB's ``numpy.dot(xi, A_ik)`` for reduced-set / mode-coupled models.
It renders to ``{np,jnp}.dot(X, M)`` per backend.
"""

import numpy as np
import pytest


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [
        ("numpy", "np.dot(xi, A_ik)"),
        ("jax", "jnp.dot(xi, A_ik)"),
    ],
)
def test_mode_dot_renders(fmt, expected):
    from tvbo.codegen import render_expression

    assert render_expression("mode_dot(xi, A_ik)", format=fmt) == expected


def test_mode_dot_in_expression():
    from tvbo.codegen import render_expression

    assert render_expression("mode_dot(xi, A_ik) + 0.5*xi", format="numpy") == ("0.5*xi + np.dot(xi, A_ik)")


def test_mode_dot_contracts_mode_axis():
    """mode_dot(X, M) == X @ M over the mode axis: result[node,k] = sum_j X[node,j] M[j,k]."""
    from tvbo.codegen import render_expression

    code = render_expression("mode_dot(xi, A_ik)", format="numpy")
    rs = np.random.RandomState(0)
    xi = rs.rand(4, 3)  # (n_nodes, n_modes)
    A_ik = rs.rand(3, 3)  # (n_modes, n_modes)
    got = eval(code, {"np": np, "xi": xi, "A_ik": A_ik})
    np.testing.assert_allclose(got, np.dot(xi, A_ik))
    assert got.shape == (4, 3)
