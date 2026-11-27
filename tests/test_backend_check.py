"""The backend check: generated code, read back into SymPy, states the specification's equations.

The golden corpus (`test_codegen_golden_corpus.py`) freezes what each backend emits and so reports that an output changed. This module asks whether the frozen output is right: every reference is read back by `tvbo.analysis.backend_check` and each state variable's derivative is compared with the model's own.

The references are read from disk rather than rendered again, because the corpus test already holds a fresh render byte-identical to them; reading them keeps this module to parsing and comparing.

A check that cannot fail proves nothing, so the controls below alter a reference and assert the alteration is reported: a changed coefficient as `differs` on the one state variable it touches, two swapped state indices as `differs` on both, a narrower domain as `differs`, a call the reader does not know or cannot follow as `unread`, and a coupling input the code fixes by name.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

import pytest

from tvbo.analysis.backend_check import DIFFERS, EQUAL, UNREAD, check_backend
from tvbo.classes.dynamics import Dynamics

from .test_codegen_golden_corpus import CASES, CORPUS, MODEL_ROOT, _case_id

UNREAD_CASES: dict[str, str] = {}
"""References the readers refuse, mapped to the construct each one names. A case leaves this table when the reader learns it, which the test reports as a failure."""


@cache
def _model(path: Path) -> Dynamics:
    """One instance per model for all its formats; the check reads a model and leaves it as it found it."""
    return Dynamics.from_file(str(path))


def _reference(case_id: str) -> str:
    return (CORPUS.root / f"{case_id}{CORPUS.suffix}").read_text(encoding="utf-8")


def _needs_reader(fmt: str) -> None:
    if fmt == "julia":
        pytest.importorskip("tree_sitter_julia", reason="reading Julia needs tvbo[julia]")


@pytest.mark.backend_core
@pytest.mark.parametrize(("path", "fmt"), CASES, ids=[_case_id(p, f) for p, f in CASES])
def test_generated_code_states_the_specification(path: Path, fmt: str):
    """Every state variable's derivative in every frozen reference equals the model's."""
    _needs_reader(fmt)
    case_id = _case_id(path, fmt)
    model = _model(path)

    verdicts = check_backend(model, fmt, code=_reference(case_id))

    assert [verdict.name for verdict in verdicts] == [str(name) for name in model.state_variables]
    if case_id in UNREAD_CASES:
        assert verdicts, f"{case_id} has no state variable to refuse"
        for verdict in verdicts:
            assert verdict.status == UNREAD, f"{case_id}: {verdict.name} now reads back as {verdict.status}"
            assert UNREAD_CASES[case_id] in verdict.detail
    else:
        failed = [f"{verdict.name}: {verdict.status} ({verdict.detail})" for verdict in verdicts if not verdict]
        assert not failed, f"{case_id} does not state the specification:\n" + "\n".join(failed)


CONTROL = MODEL_ROOT / "CoombesByrne2D.yaml"
"""`dV/dt` and `dr/dt` over two states and one coupling input, each backend's output a few lines."""

COEFFICIENT = {
    "jax": ("2 * V * r", "3 * V * r"),
    "numpy": ("2 * V * r", "3 * V * r"),
    "tvb": ("2 * V * r", "3 * V * r"),
    "tvboptim": ("2 * V * r", "3 * V * r"),
    "julia": ("2 * V .* r", "3 * V .* r"),
    "neuroml": ("2*V*r - 3.14", "3*V*r - 3.14"),
}
"""The `2 V r` term of `dr/dt` as each format spells it, and the same term with its coefficient changed."""


def _altered(fmt: str, old: str, new: str) -> str:
    reference = _reference(_case_id(CONTROL, fmt))
    assert reference.count(old) == 1, f"the {fmt} reference no longer spells `{old}` exactly once"
    return reference.replace(old, new)


@pytest.mark.backend_core
@pytest.mark.parametrize("fmt", sorted(COEFFICIENT))
def test_a_changed_coefficient_is_reported_on_its_state_variable(fmt: str):
    """Altering one term of `dr/dt` turns `r` to `differs` and leaves `V` `equal`."""
    _needs_reader(fmt)

    verdicts = check_backend(_model(CONTROL), fmt, code=_altered(fmt, *COEFFICIENT[fmt]))

    assert {verdict.name: verdict.status for verdict in verdicts} == {"V": EQUAL, "r": DIFFERS}


@pytest.mark.backend_core
def test_swapped_state_indices_are_reported():
    """Reading `V` from the second state entry and `r` from the first changes both derivatives.

    The names in the emitted code stay as they were, so a reader that trusted them would see nothing: the state array is read by position against the model's declared order.
    """
    swapped = _altered("jax", "V = current_state[0]", "V = current_state[1]").replace(
        "r = current_state[1]", "r = current_state[0]"
    )

    verdicts = check_backend(_model(CONTROL), "jax", code=swapped)

    assert {verdict.name: verdict.status for verdict in verdicts} == {"V": DIFFERS, "r": DIFFERS}


@pytest.mark.backend_core
def test_an_unknown_call_is_refused_rather_than_skipped():
    """A call the reader has no meaning for makes the whole output `unread`, and the verdict names it."""
    unknown = _altered("jax", "Delta / jnp.pi", "jnp.unknown(Delta)")

    verdicts = check_backend(_model(CONTROL), "jax", code=unknown)

    assert {verdict.status for verdict in verdicts} == {UNREAD}
    assert all("jnp.unknown(Delta)" in verdict.detail for verdict in verdicts)


@pytest.mark.backend_core
def test_an_input_the_code_fixes_is_fixed_in_the_specification_too():
    """The JAX dfun sets `local_coupling = 0`, and the verdict records that the comparison holds under it."""
    path = MODEL_ROOT / "ReducedWongWang.yaml"

    (verdict,) = check_backend(_model(path), "jax", code=_reference(_case_id(path, "jax")))

    assert verdict.status == EQUAL
    assert {str(symbol): value for symbol, value in verdict.pinned.items()} == {"local_coupling": 0}
    assert "the code fixes local_coupling = 0" in verdict.detail


@pytest.mark.backend_core
def test_a_coupling_input_the_code_drops_is_named_in_the_verdict():
    """An emitter that writes `c_glob = 0.0` instead of reading the coupling array says so in `dV/dt`'s verdict."""
    (v_verdict, _) = check_backend(_model(CONTROL), "jax", code=_altered("jax", "c_glob = cX[0]", "c_glob = 0.0"))

    assert "the code fixes c_glob = 0" in v_verdict.detail


@pytest.mark.backend_core
def test_validate_backend_fails_on_a_refusal_only_where_the_format_is_named():
    """NeuroML refuses multi-mode models by design, so the default run passes on the other formats and `--format neuroml` fails."""
    from typer.testing import CliRunner

    from tvbo.cli import app

    runner = CliRunner()
    by_default = runner.invoke(app, ["validate", "backend", "ReducedSetFitzHughNagumo"])
    named = runner.invoke(app, ["validate", "backend", "ReducedSetFitzHughNagumo", "--format", "neuroml"])

    assert "neuroml: not checked" in by_default.output
    assert by_default.exit_code == 0, by_default.output
    assert "neuroml: not checked" in named.output
    assert named.exit_code == 1


def _one_state(name: str, rhs: str) -> Dynamics:
    return Dynamics(
        name=name, state_variables={"x": {"equation": {"lhs": "Derivative(x, t)", "rhs": rhs}, "initial_value": 1.0}}
    )


@pytest.mark.backend_core
def test_a_narrower_domain_is_a_difference():
    """An emitted `sqrt(x)` for a specified `sqrt(Abs(x))` agrees wherever `x` is positive and has no real value elsewhere, which is a difference."""
    model = _one_state("Root", "sqrt(Abs(x))")
    code = model.render_code("jax")
    assert code.count("jnp.sqrt(abs(x))") == 1

    (verdict,) = check_backend(model, "jax", code=code.replace("jnp.sqrt(abs(x))", "jnp.sqrt(x)"))

    assert verdict.status == DIFFERS
    assert "emitted no real number" in verdict.detail


@pytest.mark.backend_core
def test_julia_spells_eulers_number_with_its_own_constant():
    """Julia defines `ℯ` and no `e`, so code reading a bare `e` would throw where it runs, and is unread here."""
    _needs_reader("julia")
    model = _one_state("Euler", "E * x")
    code = re.sub(r"\b(e|ℯ) \* x", "{} * x", model.render_code("julia"))
    assert code.count("{} * x") == 1

    assert [verdict.status for verdict in check_backend(model, "julia", code=code.replace("{}", "ℯ"))] == [EQUAL]
    (verdict,) = check_backend(model, "julia", code=code.replace("{}", "e"))
    assert verdict.status == UNREAD
    assert "`e` is read before anything binds it" in verdict.detail


@pytest.mark.backend_core
@pytest.mark.parametrize(
    ("old", "new", "named"),
    [
        ("2 * V * r", "2 * V * r.sum(axis=-1, keepdims=True)", "the method call"),
        ("Delta / jnp.pi", "jnp.sin(Delta, 2) / jnp.pi", "does not take these arguments"),
        ("return derivatives", "return dr_dt", "not one derivative for each"),
    ],
    ids=["method call", "wrong arity", "bare return"],
)
def test_code_the_reader_cannot_follow_is_unread(old: str, new: str, named: str):
    """A method call, a function called with arguments it does not take, and an entry point returning one expression are each `unread`, never an exception out of `check_backend`."""
    verdicts = check_backend(_model(CONTROL), "jax", code=_altered("jax", old, new))

    assert {verdict.status for verdict in verdicts} == {UNREAD}
    assert all(named in verdict.detail for verdict in verdicts)
