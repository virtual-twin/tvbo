"""The parsing namespace is frozen, explicit, and never SymPy's.

These tests pin the three properties that make a `SymbolContext` worth having over the
plain dict it replaces: parsing cannot change it, deriving one cannot change its parent,
and none of it reaches back into `sympy.abc`. Each corresponds to a way the previous
shared-and-mutated namespace produced a wrong parse.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import sympy.abc
from sympy import Function, Symbol, pi, srepr

from tvbo.parse.symbols import AUTO, BUILTIN_SHADOW, SymbolContext

MUTATORS = {
    "__setitem__": lambda ctx: ctx.__setitem__("x", Symbol("x")),
    "__delitem__": lambda ctx: ctx.__delitem__("E"),
    "__ior__": lambda ctx: ctx.__ior__({"x": Symbol("x")}),
    "clear": lambda ctx: ctx.clear(),
    "pop": lambda ctx: ctx.pop("E"),
    "popitem": lambda ctx: ctx.popitem(),
    "setdefault": lambda ctx: ctx.setdefault("x", Symbol("x")),
    "update": lambda ctx: ctx.update({"x": Symbol("x")}),
}


@pytest.mark.parametrize("name", sorted(MUTATORS))
def test_every_mutator_is_refused(name):
    """No route into `dict`'s mutating API is left open.

    Blocking only `__setitem__` would be worthless: SymPy's `auto_symbol` assigns, but
    `parse_expr` reaches for `pop`, and callers reach for `update`.
    """
    with pytest.raises(TypeError, match="frozen"):
        MUTATORS[name](BUILTIN_SHADOW.extend(x=Symbol("x")))


def test_parsing_does_not_change_the_context():
    """SymPy resolves an `AUTO` name by writing its choice back — onto the copy, not here."""
    scope = BUILTIN_SHADOW.extend(x=Symbol("x"))
    before = dict(scope)

    scope.parse("beta * x")

    assert dict(scope) == before


def test_the_same_context_parses_the_same_way_twice():
    """The property the shared namespace could not offer: a parse depends only on its input."""
    scope = BUILTIN_SHADOW.extend(x=Symbol("x"))

    assert srepr(scope.parse("beta * x")) == srepr(scope.parse("beta * x"))


def test_an_earlier_parse_cannot_decide_a_later_one():
    """`E` is a symbol in one expression and a function in the next, in either order.

    Under the mutated global this was order-dependent: whichever expression parsed first
    wrote its choice into the shared dict, and the other silently got that answer.
    """
    scope = BUILTIN_SHADOW.extend(x=Symbol("x"))

    as_symbol_first = (scope.parse("E * x"), scope.parse("E(x)"))
    as_function_first = (scope.parse("E(x)"), scope.parse("E * x"))

    assert as_symbol_first[0] == Symbol("E") * Symbol("x")
    assert as_symbol_first[1] == Function("E")(Symbol("x"))
    assert as_function_first[1] == as_symbol_first[0]
    assert as_function_first[0] == as_symbol_first[1]


def test_extend_and_without_leave_the_receiver_alone():
    scope = BUILTIN_SHADOW.extend(x=Symbol("x"))
    before = dict(scope)

    extended = scope.extend(y=Symbol("y"))
    reduced = scope.without("x")

    assert dict(scope) == before
    assert "y" in extended and "y" not in scope
    assert "x" not in reduced and "x" in scope
    assert isinstance(extended, SymbolContext) and isinstance(reduced, SymbolContext)


def test_later_namespaces_win_over_earlier_ones():
    scope = SymbolContext({"a": Symbol("first")}, {"a": Symbol("second")}, a=Symbol("third"))

    assert scope["a"] == Symbol("third")


@pytest.mark.parametrize("name", ["E", "I", "N", "O", "Q", "S", "beta", "gamma", "zeta"])
def test_builtin_names_parse_as_the_models_own(name):
    """Every shadowed name is the model's quantity, not SymPy's object of that name."""
    scope = BUILTIN_SHADOW.extend(x=Symbol("x"))

    assert scope.parse(f"{name} * x") == Symbol(name) * Symbol("x")


def test_pi_is_still_pi():
    """Shadowing `pi` would turn `2*pi` into a free symbol; no model means that."""
    assert BUILTIN_SHADOW.parse("2*pi") == 2 * pi
    assert "pi" not in BUILTIN_SHADOW


def test_auto_picks_function_or_symbol_from_the_call_syntax():
    """What `AUTO` buys: one declaration covers both uses of a name."""
    scope = BUILTIN_SHADOW.extend(x=Symbol("x"))

    assert scope.parse("gamma(x)") == Function("gamma")(Symbol("x"))
    assert scope.parse("gamma * x") == Symbol("gamma") * Symbol("x")
    assert BUILTIN_SHADOW["gamma"] is AUTO


def test_unshadowed_sympy_would_get_this_wrong():
    """The failure being prevented, stated as a test rather than a comment."""
    from sympy import parse_expr

    with pytest.raises(TypeError):
        parse_expr("beta * x")


def test_importing_tvbo_does_not_touch_sympys_namespace():
    """Run in a clean interpreter: the pollution being removed happened at import time.

    Checked in a subprocess because `sympy.abc._clash1` is process-global — once this
    session's imports have run, an in-process assertion proves nothing about import order.
    """
    source = (
        "import sympy.abc, copy;"
        "before = copy.deepcopy(sympy.abc._clash1), copy.deepcopy(sympy.abc._clash2);"
        "import tvbo, tvbo.classes.equation, tvbo.classes.dynamics, tvbo.parse.symbols;"
        "after = (sympy.abc._clash1, sympy.abc._clash2);"
        "assert before == after, f'tvbo mutated sympy.abc: {before} -> {after}'"
    )
    subprocess.run([sys.executable, "-c", source], check=True)


def test_rendering_a_model_does_not_touch_sympys_namespace():
    """The same guarantee after real work, not just after import."""
    source = (
        "import sympy.abc, copy;"
        "import tvbo;"
        "before = copy.deepcopy(sympy.abc._clash1), copy.deepcopy(sympy.abc._clash2);"
        "from tvbo.classes.dynamics import Dynamics;"
        "from tvbo.data.registry import database_dir;"
        "p = sorted(database_dir('Dynamics').rglob('*.yaml'))[0];"
        "Dynamics.from_file(str(p)).render_code(format='numpy');"
        "after = (sympy.abc._clash1, sympy.abc._clash2);"
        "assert before == after, f'render_code mutated sympy.abc: {before} -> {after}'"
    )
    subprocess.run([sys.executable, "-c", source], check=True)
