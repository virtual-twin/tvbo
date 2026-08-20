"""Parsing namespaces: the names an expression is allowed to mean.

Parsing an expression is not a pure function of its text. `"beta * x"` is a product of two symbols in a model that declares a parameter `beta`, and a call to SymPy's beta function in one that does not; `"I"` is a state variable in one model and the imaginary unit in another.
The namespace handed to the parser decides, so it is part of the input.

TVBO used to keep that namespace in a single dict imported from `sympy.abc._clash1` and mutated in place — by this package at import, by the ontology loader per equation, by the stimulus code per stimulus, and by SymPy's own `auto_symbol`, which rewrites its `local_dict` as a side effect of parsing. Every parse therefore saw a namespace that depended on which parses had already run, in that interpreter, in that order. Two consequences followed: a model could parse differently depending on import order, and any other library in the process that used `sympy.abc._clash1` silently got TVBO's names.

[`SymbolContext`](#tvbo.parse.symbols.SymbolContext) replaces it. A context is frozen, so it cannot pick up names from a previous parse, and it is never handed to SymPy directly — [`parse`](#tvbo.parse.symbols.SymbolContext.parse) passes a private copy, absorbing the mutation. Deriving a namespace is explicit and returns a new context rather than editing a shared one.

Nothing here is a default that applies "unless overridden": a caller states the namespace it means. The only shared piece is [`BUILTIN_SHADOW`](#tvbo.parse.symbols.BUILTIN_SHADOW), which covers the names SymPy would otherwise resolve to its own objects.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import sympy.abc
from sympy import Basic, Symbol
from sympy.core.function import UndefinedFunction
from sympy.parsing.sympy_parser import null as AUTO

from tvbo.parse.expression import parse_eq

__all__ = [
    "AUTO",
    "BUILTIN_SHADOW",
    "SymbolContext",
    "assumptions_of",
    "symbol_in",
]


def assumptions_of(element: Any = None) -> dict[str, bool]:
    """The SymPy assumptions a declared quantity clearly implies.

    A modelled quantity is real. That alone is the difference between SymPy answering a question and not: asked for the fixed points of `Generic2dOscillator` it must otherwise consider complex branches, and does not terminate in 45 s; told the parameters are real it returns in under one.

    A declared `domain` says more — a lower bound at or above zero makes the symbol positive or nonnegative, which is what lets `sqrt(x**2)` reduce and a sign test resolve. Nothing beyond that is inferred: a bound admitting negative values implies realness only, and an absent domain implies nothing further. Assumptions SymPy is told are assumptions it will act on, so an over-claim is a wrong answer rather than a missed simplification.
    """
    assumptions = {"real": True}
    domain = getattr(element, "domain", None) if element is not None else None
    lo = getattr(domain, "lo", None) if domain is not None else None
    if lo is not None and lo >= 0:
        assumptions["positive" if lo > 0 else "nonnegative"] = True
    return assumptions


def symbol_in(scope: Mapping | None, name: Any) -> Any:
    """Whatever a parsed expression uses for *name* — symbol, `y0(t)`, or function head.

    For code that holds an expression and the namespace it was parsed against, but not the model: resolving through the scope is what makes a later `subs` or `free_symbols` test meet the same symbol the parser produced. Falls back to a bare symbol for a name the scope does not declare, which is the right answer for one it never bound — an unbound name is exactly what SymPy's `auto_symbol` turns into a bare symbol while parsing.

    Function heads resolve too, because `Function("f") != Function("f", real=True)` and the two `srepr` identically: rebuilding a head to write `f(x)` on a left-hand side yields a different class from the one every calling equation contains.
    """
    resolved = (scope or {}).get(str(name))
    if isinstance(resolved, (Basic, UndefinedFunction)):
        return resolved
    return Symbol(str(name))


def _rejects_mutation(method: str):
    """Build the replacement for one of `dict`'s mutating methods."""

    def blocked(self, *args, **kwargs):
        raise TypeError(
            f"SymbolContext is frozen; {method}() would rebind a name after construction. "
            "Build a derived namespace with .extend() or .without(), and parse with "
            ".parse(), which hands SymPy a private copy."
        )

    blocked.__name__ = method
    blocked.__qualname__ = f"SymbolContext.{method}"
    return blocked


class SymbolContext(dict):
    """A frozen mapping of name to SymPy object, usable as a parser namespace.

    Subclasses `dict` because that is what SymPy means by a `local_dict`, so a context can be inspected, compared and merged with ordinary mapping code. It is frozen because the two bugs it exists to prevent are both writes: SymPy's `auto_symbol` rewriting the namespace while parsing, and callers extending a shared namespace in place so that a later, unrelated parse sees the addition.

    A value of [`AUTO`](#tvbo.parse.symbols.AUTO) declares a name without committing to what it is: SymPy resolves it to a `Function` when the text calls it and a `Symbol` otherwise.
    That is the only way to say "this name is the model's, whatever SymPy would make of it" without inspecting the expression first.

    Example:
        >>> scope = BUILTIN_SHADOW.extend(x=Symbol("x"))
        >>> scope.parse("beta * x")           # beta is a Symbol, not sympy.beta
        >>> scope.extend(beta=Symbol("beta")) # a new context; `scope` is untouched
    """

    __slots__ = ()

    def __init__(self, *namespaces: Mapping[str, Any], **names: Any):
        """Merge `namespaces` left to right, then `names`; later entries win."""
        merged: dict[str, Any] = {}
        for namespace in namespaces:
            merged.update(namespace)
        merged.update(names)
        super().__init__(merged)

    @classmethod
    def auto(cls, names: Iterable[str]) -> SymbolContext:
        """A context declaring `names` without fixing whether each is a Symbol or Function."""
        return cls({name: AUTO for name in names})

    def extend(self, *namespaces: Mapping[str, Any], **names: Any) -> SymbolContext:
        """A new context with `namespaces` and `names` layered on top of this one."""
        return type(self)(self, *namespaces, **names)

    def without(self, *names: str) -> SymbolContext:
        """A new context with `names` removed, so they parse as plain symbols again."""
        dropped = frozenset(names)
        return type(self)({k: v for k, v in self.items() if k not in dropped})

    def parse(self, expression, **kwargs):
        """Parse `expression` against this namespace, which `parse_eq` copies before use.

        The copy is what makes freezing workable: SymPy resolves an [`AUTO`](#tvbo.parse.symbols.AUTO) name by writing the object it chose back into the `local_dict`, and `parse_expr` pops its bookkeeping key afterwards. Both writes land on `parse_eq`'s own copy, so a context yields the same result however many times it is used — and equally for a caller that hands a context straight to `parse_eq`.
        """
        return parse_eq(expression, local_dict=self, **kwargs)

    def __reduce__(self):
        """Rebuild by construction, because restoring a `dict` subclass assigns its items.

        `copy`, `deepcopy` and `pickle` all reach for `__reduce_ex__`, which replays items through `__setitem__` — exactly what freezing blocks. Without this, any object graph holding a context is uncopyable, and an `Exploration` deep-copies one per cell.
        """
        return type(self), (dict(self),)

    def __deepcopy__(self, memo) -> SymbolContext:
        """Share rather than duplicate: a frozen context has no per-copy state to protect."""
        return self

    def __repr__(self) -> str:
        return f"{type(self).__name__}({dict.__repr__(self)})"


for _method in (
    "__setitem__",
    "__delitem__",
    "__ior__",
    "clear",
    "pop",
    "popitem",
    "setdefault",
    "update",
):
    setattr(SymbolContext, _method, _rejects_mutation(_method))


BUILTIN_SHADOW = SymbolContext.auto(sorted(set(sympy.abc._clash1) | (set(sympy.abc._clash2) - {"pi"})))
"""Names SymPy binds to its own objects, declared as the caller's instead.

`E`, `I`, `N`, `O`, `Q` and `S` are Euler's number, the imaginary unit, `evalf`, big-O, the
assumptions registry and the sympify shortcut; `beta`, `gamma` and `zeta` are functions. All
are ordinary quantity names in a neural mass model, so without this a model that declares one
parses into something else entirely.

Taken from SymPy's own `_clash1` and `_clash2` so the list tracks upstream rather than a
copy that silently goes stale. `pi` is deliberately excluded: no model means anything but the
constant by it, and shadowing it would turn `2*pi` into a free symbol.

This is read at import and never written back — `sympy.abc._clash1` is SymPy's, and
`tests/test_symbol_context.py` asserts TVBO leaves it that way.
"""
