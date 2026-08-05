"""What a coupling's gathered states mean in the JAX cfun, and what it refuses.

``pre`` reads SOURCES: tvboptim folds a factored coupling's declared ``local_states`` into
``vec_states`` and registers them as *incoming*, reaching the target's own value through the
``<state>_i`` alias in ``post`` instead. The JAX template mirrors that — ``theta`` in ``pre``
is the delayed, weight-gathered row, ``theta_i`` in ``post`` is ``current_state``.

What it did not do is notice when a gathered name has no row to read: a state named in
``pre`` that is not transmitted was simply never assigned, and the defect surfaced as a
``NameError`` from generated code rather than as a codegen error naming the spec.

NOT pinned here, because it is an open question rather than a bug: what a BARE
``local_states`` name means inside ``pre``. The schema says the target's value; the factored
Kuramoto form in the database uses it as the source. Until one reading wins, these tests
assert only the behaviour both agree on.
"""
from __future__ import annotations

import re

import pytest

from tvbo.classes.coupling import Coupling
from tvbo.classes.dynamics import Dynamics


def _model():
    return Dynamics(**{
        "name": "twostate",
        "parameters": {"K": {"value": 1.0}},
        "coupling_inputs": {"c": {}},
        "state_variables": {
            "theta": {"equation": {"rhs": "K * c"}, "initial_value": 0.0,
                      "coupling_variable": True},
            "v": {"equation": {"rhs": "-v"}, "initial_value": 0.0},
        },
    })


def _render(pre, local_states, post="gx_0"):
    coupling = Coupling(**{
        "name": "c",
        "delayed": False,
        "local_states": local_states,
        "pre_expression": {"rhs": pre},
        "post_expression": {"rhs": post},
    })
    return coupling.render_code("jax", model=_model())


def _assignment(code, name):
    """The RHS the generated cfun assigns to *name*, or None."""
    m = re.search(rf"^\s*{re.escape(name)} = (.+)$", code, re.M)
    return m.group(1).strip() if m else None


def test_a_gathered_state_reads_the_source_row():
    """A factored pre is evaluated on sources; the target's value arrives via `_i` in post."""
    code = _render("sin(theta)", ["theta"])
    assert "x_j[" in _assignment(code, "theta")


def test_the_target_alias_reads_current_state_not_the_gathered_row():
    code = _render("sin(theta)", ["theta"], post="cos(theta_i) * gx_0")
    assert _assignment(code, "theta_i").startswith("current_state[")


def test_reading_only_the_target_alias_is_not_mistaken_for_a_source_read():
    """`\\b` does not match the `v` inside `v_i`, so the guard must not fire on it."""
    code = _render("sin(theta) * v_i", ["v"])
    assert _assignment(code, "v_i").startswith("current_state[")


def test_a_local_state_the_model_does_not_declare_is_refused():
    with pytest.raises(Exception, match="local_states"):
        _render("sin(theta) * ghost", ["ghost"])


def test_the_generated_coupling_compiles():
    code = _render("sin(theta)", ["theta"], post="cos(theta_i) * gx_0")
    compile(code, "<cfun>", "exec")
