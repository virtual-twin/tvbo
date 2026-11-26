"""An equation printed in its authored operation order keeps the value of every name bound into it."""

import pytest

from tvbo.codegen.code import render_verbatim

pytestmark = pytest.mark.backend_core


@pytest.mark.parametrize(
    ("source", "bindings", "expected"),
    [
        ("x ** 2", {"x": -3}, "(-3) ** 2"),
        ("k ** 2 * X", {"k": -0.5}, "(-0.5) ** 2 * X"),
        ("k * X", {"k": 2.5}, "2.5 * X"),
    ],
)
def test_a_bound_number_keeps_its_sign_under_a_power(source, bindings, expected):
    """A negative binding is written as a parenthesised negation, which a bare literal under ``**`` is not."""
    assert render_verbatim(source, format="jax", bindings=bindings) == expected


def test_a_bound_axis_stays_an_integer_literal():
    assert render_verbatim("take(t, a)", format="jax", bindings={"a": -1}) == "jnp.take(t, -1)"
