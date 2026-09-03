"""Dimensional analysis of a model's equations.

The symbolic layer knows a quantity is *real*. This is what lets it know a quantity is
*a voltage*: units travel **beside** the expression, as a map from the scope's own
symbols to the unit each was declared in, and propagate through the equation rather than being multiplied into it.

Beside, not inside, for three reasons. 87% of quantities in the curated database declare no unit, so an expression carrying `Quantity` factors would be inconsistent by construction for almost every model; every consumer of the symbolic layer would have to strip those factors before printing; and the frozen codegen corpus would move. None of that buys anything a separate map does not.

The verdict is three-valued — `consistent`, `inconsistent`, `underdetermined` — because "I cannot tell" is a different claim from "this is wrong", and collapsing them is how a checker becomes noise. An undeclared quantity is the common case, and treating it as dimensionless silently corrupts the answer: `hhcell_1` declares units for 92% of its symbols and reports `dv/dt` as `1/capacitance` — not voltage over time — purely because its one undeclared symbol was assumed dimensionless.

Two strictnesses over one map (U2). `dimensional` compares base-dimension vectors, so `mV/ms` and `V/s` agree; `exact` compares the full quantity, so they do not, and the disagreement is reported as the exact rational ratio `1` vs `1000`. The ratio is the part that names the bug — a boolean does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import sympy as sp

from tvbo.utils.units import unit_dimensions, unit_expression

CONSISTENT = "consistent"
INCONSISTENT = "inconsistent"
UNDERDETERMINED = "underdetermined"


class _Underdetermined(Exception):
    """A subexpression contains a symbol with no declared unit."""

    def __init__(self, symbol):
        super().__init__(str(symbol))
        self.symbol = symbol


@dataclass
class Term:
    """One addend of an equation, with the unit propagation assigned to it."""

    expression: Any
    unit: Any = None
    unknown: Any = None


@dataclass
class Verdict:
    """The dimensional standing of one equation."""

    name: str
    equation: Any
    status: str
    detail: str = ""
    unit: Any = None
    ratios: dict = field(default_factory=dict)
    inferred: dict = field(default_factory=dict)
    undeclared: tuple = ()

    def __bool__(self) -> bool:
        return self.status == CONSISTENT


def quantity_name(expression):
    """The name of the quantity *expression* stands for.

    `y0` for all of `y0`, `y0(t)` and `Derivative(y0(t), t)` — a quantity is the same quantity however the notation renders it, and every caller (a verdict's label, a report row keyed by parameter name) wants that one name.
    """
    target = expression.expr if isinstance(expression, sp.Derivative) else expression
    if isinstance(target, sp.core.function.AppliedUndef):
        return target.func.__name__
    return str(target)


def _as_unit(declared):
    """A declared unit string as a SymPy quantity, or `None` when it says nothing."""
    if declared is None:
        return None
    try:
        return unit_expression(declared)
    except ValueError:
        return None


def _dimension_key(unit):
    """The comparison key for `dimensional` strictness: base exponents, scale discarded."""
    if unit is None:
        return None
    exponents = {}
    for base, power in _base_powers(unit).items():
        if power:
            exponents[base] = power
    return tuple(sorted(exponents.items()))


def _base_powers(unit):
    """Base-unit exponents of a propagated unit expression."""
    powers: dict[str, Fraction] = {}
    for factor, exponent in (unit.as_powers_dict() if hasattr(unit, "as_powers_dict") else {}).items():
        if factor.is_number or not exponent.is_number:
            continue
        powers[str(factor)] = powers.get(str(factor), Fraction(0)) + Fraction(str(exponent))
    return powers


class _Propagator:
    """Walks an expression assigning a unit to every subexpression.

    Constraints are exactly the three that hold without solving anything: addends of a sum share a unit, a transcendental function's argument is dimensionless, and a derivative divides by the unit of its variable. Anything needing a general linear solve over base-dimension exponents reports `underdetermined` instead of guessing — the bound is deliberate, because a checker that occasionally invents a constraint is worse than one that admits it does not know.
    """

    def __init__(self, units: dict, time_unit):
        """Index the declared units by name, not by symbol object.

        A unit is declared for a *quantity*, and the same quantity reaches this walk under more than one symbol: an inlined function body carries the codegen view's bare ``Symbol("e0")`` where the equation around it carries the analysis view's ``Symbol("e0", real=True)``. Those compare unequal, so a symbol-keyed lookup reports a declared parameter as undeclared.
        """
        self.units = {str(symbol): unit for symbol, unit in units.items()}
        self.time_unit = time_unit
        self.inferred: dict = {}
        self.undeclared: set = set()

    def unit_of(self, expression):
        """The unit of *expression*, propagated from the declared ones around it.

        Declared quantities are recognised before the transcendental branch below: a state variable bound as ``Function(name)(t)`` is a *quantity*, not a function being applied, and reaching that branch would demand its argument ``t`` be dimensionless.
        """
        expression = sp.sympify(expression)

        if expression.is_number:
            return sp.Integer(1)

        if str(expression) == "t":
            return self.time_unit

        if self._is_declared(expression):
            return self._declared(expression)

        if isinstance(expression, sp.Derivative):
            numerator = self.unit_of(expression.expr)
            for variable, order in expression.variable_count:
                numerator = numerator / self._variable_unit(variable) ** order
            return numerator

        if expression.is_Add:
            return self._add(expression)

        if expression.is_Mul:
            product = sp.Integer(1)
            for factor in expression.args:
                product *= self.unit_of(factor)
            return sp.simplify(product)

        if expression.is_Pow:
            base, exponent = expression.args
            if not exponent.is_number:
                self._require_dimensionless(exponent, "an exponent")
                self._require_dimensionless(base, "a base raised to a non-numeric power")
                return sp.Integer(1)
            return self.unit_of(base) ** sp.Rational(str(exponent)) if exponent.is_Rational else self.unit_of(base) ** exponent

        if isinstance(expression, sp.Piecewise):
            return self._add(sp.Add(*[value for value, _ in expression.args], evaluate=False))

        if expression.is_Function:
            for argument in expression.args:
                self._require_dimensionless(argument, f"an argument of {expression.func}")
            return sp.Integer(1)

        if expression.is_Relational or expression.is_Boolean:
            return sp.Integer(1)

        raise _Underdetermined(expression)

    def _name(self, expression):
        """The declared name an expression stands for: `y0(t)` and `y0` are both `y0`."""
        if getattr(expression, "is_Function", False) and expression.args:
            return getattr(expression.func, "__name__", str(expression.func))
        return str(expression)

    def _is_declared(self, expression):
        return self._name(expression) in self.units

    def _declared(self, symbol):
        unit = _as_unit(self.units.get(self._name(symbol)))
        if unit is not None:
            return unit
        self.undeclared.add(symbol)
        raise _Underdetermined(symbol)

    def _variable_unit(self, variable):
        if str(variable) == "t":
            return self.time_unit
        return self.unit_of(variable)

    def _require_dimensionless(self, expression, where):
        unit = self.unit_of(expression)
        if _dimension_key(unit):
            raise DimensionalClash(f"{where} is not dimensionless: {sp.nsimplify(unit)}")

    def _add(self, expression):
        """Addends share a unit; one unknown among knowns is forced rather than refused."""
        terms = []
        for addend in expression.args:
            try:
                terms.append(Term(addend, self.unit_of(addend)))
            except _Underdetermined as unresolved:
                terms.append(Term(addend, None, unresolved.symbol))

        known = [t for t in terms if t.unit is not None]
        if not known:
            raise _Underdetermined(terms[0].unknown if terms else expression)

        reference = known[0].unit
        clashes = {
            quantity_name(term.expression): sp.nsimplify(sp.simplify(term.unit / reference))
            for term in known[1:]
            if sp.simplify(term.unit / reference) != 1
        }
        if clashes:
            raise DimensionalClash(f"addends disagree: {clashes}")

        for term in terms:
            if term.unit is None and term.unknown is not None:
                self.inferred.setdefault(term.unknown, self._solve_for(term.expression, term.unknown, reference))
        if any(t.unit is None for t in terms):
            raise _Underdetermined(next(t.unknown for t in terms if t.unit is None))
        return reference

    def _solve_for(self, expression, unknown, target):
        """The unit the unknown must carry for this addend to match the others."""
        try:
            rest = expression / unknown
            return sp.nsimplify(sp.simplify(target / self.unit_of(rest)))
        except (_Underdetermined, DimensionalClash, TypeError, ZeroDivisionError):
            return None


class DimensionalClash(Exception):
    """Two quantities that must agree dimensionally do not."""


def declared_units(model, scope: dict | None = None) -> dict:
    """The third projection of the symbolic layer: `{scope symbol: declared unit}`.

    Keyed by the scope's own symbols, like `Dynamics.symbolic["parameters"]` — rebuilt keys look identical, compare unequal, and would silently match nothing. A caller that already holds the analysis scope passes it, so the map is keyed by the very table its equations were parsed against rather than by whichever one the model resolves to.
    """
    if scope is None:
        scope = model.get_symbolic_elements(time_dependent=True)
    collections = (
        getattr(model, "parameters", None) or {},
        getattr(model, "state_variables", None) or {},
        getattr(model, "derived_variables", None) or {},
        getattr(model, "derived_parameters", None) or {},
        getattr(model, "coupling_inputs", None) or {},
    )
    units = {}
    for collection in collections:
        for name, member in collection.items():
            symbol = scope.get(str(name))
            if symbol is None:
                continue
            declared = getattr(member, "unit", None)
            units[symbol] = str(getattr(declared, "text", declared)) if declared is not None else None
            if getattr(symbol, "is_Function", False) and symbol.args:
                units[symbol.func] = units[symbol]
    return units


def _function_bodies(model):
    """The model's own function definitions, or an empty table if it declares none."""
    from tvbo.parse.expression import function_bodies

    try:
        return function_bodies(model)
    except Exception:  # noqa: BLE001 — a model that cannot expose bodies simply inlines nothing
        return {}


def _inline(expression, bodies):
    """Expand model-defined calls before checking.

    A user function carries no unit of its own, so a call to one is opaque to propagation: `Jansen1995` reports its sigmoid's argument as non-dimensionless purely because `Sigm(...)` is undeclared. Inlining resolves it exactly — `r` is `per_mV` and `v0` is `mV`, so the argument really is dimensionless — which is why only four of the curated models declare functions and none of them needs a `Function.unit` slot. It works because the analysis view's function heads are now the same objects the bodies were parsed against; before that, substituting across the two views matched nothing, silently.
    """
    if not bodies:
        return expression
    from tvbo.codegen.code import inline_functions

    return inline_functions(expression, bodies)


def check_units(model, strictness: str = "dimensional", time_unit: str | None = None) -> list[Verdict]:
    """Per-equation dimensional verdicts for *model*.

    Args:
        model: A `Dynamics`, read through its symbolic projection.
        strictness: `"dimensional"` compares base-dimension vectors, so `mV/ms` and `V/s`
            agree. `"exact"` compares the whole quantity, so they do not, and the ratio
            is reported.
        time_unit: The scope's time unit; resolved via `time_unit_of` when omitted.

    Returns:
        One `Verdict` per state equation, each `consistent`, `inconsistent` or
        `underdetermined`, carrying the derived unit, any inferred units, and the exact
        ratios of any disagreement.
    """
    if strictness not in ("dimensional", "exact"):
        raise ValueError(f"strictness must be 'dimensional' or 'exact', not {strictness!r}")

    from tvbo.utils.units import time_unit_of

    units = declared_units(model)
    clock = _as_unit(time_unit or time_unit_of(getattr(model, "integration", None), model))
    bodies = _function_bodies(model)
    verdicts = []

    for equation in model.symbolic["state"]:
        name = quantity_name(equation.lhs)
        propagator = _Propagator(units, clock)
        try:
            left = propagator.unit_of(equation.lhs)
            right = propagator.unit_of(_inline(equation.rhs, bodies))
        except _Underdetermined as unresolved:
            verdicts.append(
                Verdict(
                    name,
                    equation,
                    UNDERDETERMINED,
                    f"{quantity_name(unresolved.symbol)} has no declared unit",
                    inferred=dict(propagator.inferred),
                    undeclared=tuple(sorted(map(str, propagator.undeclared))),
                )
            )
            continue
        except DimensionalClash as clash:
            verdicts.append(Verdict(name, equation, INCONSISTENT, str(clash), inferred=dict(propagator.inferred)))
            continue

        ratio = sp.nsimplify(sp.simplify(right / left))
        agrees = ratio == 1 if strictness == "exact" else _dimension_key(left) == _dimension_key(right)
        verdicts.append(
            Verdict(
                name,
                equation,
                CONSISTENT if agrees else INCONSISTENT,
                "" if agrees else f"left is {sp.nsimplify(left)}, right is {sp.nsimplify(right)}",
                unit=left,
                ratios={} if agrees else {"right/left": ratio},
                inferred=dict(propagator.inferred),
            )
        )
    return verdicts


def dimension_exponents(unit) -> dict:
    """ISO 80000-1 base-dimension exponents of a declared unit, for reporting (U25).

    `dim Q = L²MT⁻³I⁻¹` is what makes an inconsistency legible: `L²MT⁻³I⁻¹` against `L²MT⁻⁴I⁻¹` shows where the discrepancy is, where "voltage vs something else" does not.
    """
    return unit_dimensions(unit)
