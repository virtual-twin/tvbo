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


def test_a_list_pre_emits_one_row_per_component_however_many_inputs_exist():
    """The rows a multi-output coupling returns come from `pre`, not from the input count.

    `pre: [S*wLRE, S*wFFI]` is two weighted routes over one connectome, so `cfun` returns
    two rows whatever the model declares alongside them. An emitter that counted
    `coupling_inputs` instead counted the local ones too — a `local: true` input is not
    driven by the connectome and has no row — and returned the first route twice, at the
    right shape and with no error to say so.

    The weights are named `wLRE`/`wFFI` because the emitter that got this wrong found its
    per-row weight matrix by matching those literal names, so any other pair leaves the
    defect unreachable and the test pinning nothing.
    """
    model = _model()
    coupling = Coupling(**{
        "name": "c",
        "delayed": False,
        "incoming_states": ["theta"],
        "parameters": {"wLRE": {"value": 1.0}, "wFFI": {"value": 2.0}},
        "pre_expression": {"rhs": "[theta * wLRE, theta * wFFI]"},
        "post_expression": {"rhs": "gx"},
    })
    one_input = coupling.render_code("jax", model=model)

    model.coupling_inputs["local_coupling"] = {"local": True}
    assert coupling.render_code("jax", model=model) == one_input

    assert _assignment(one_input, "gx_0") == "gx[0]"
    assert _assignment(one_input, "gx_1") == "gx[1]"
    assert _assignment(one_input, "gx_2") is None
    assert one_input.count("def op") == 1


def test_the_generated_coupling_compiles():
    code = _render("sin(theta)", ["theta"], post="cos(theta_i) * gx_0")
    compile(code, "<cfun>", "exec")
