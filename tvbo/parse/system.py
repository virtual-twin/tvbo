#
# Module: system.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""The symbolic layer between a model's metadata and everything rendered from it.

[`SymbolicSystem`](#tvbo.parse.system.SymbolicSystem) parses a model's equations once and
holds them, together with the symbol tables they were parsed against. Every consumer —
code generation, the analysis views, the report — is a projection of it, so an equation is
parsed once no matter how many of them ask.

It is a service object rather than a set of methods on `Dynamics` because a scope and the
equations parsed against it have to agree, and that agreement is what the cache is keyed
on. Owning both here makes the one invalidation point visible instead of leaving it as an
attribute convention spread across a model class.
"""

from __future__ import annotations

from sympy import Derivative, Eq, Function, Symbol, latex

from tvbo.parse.expression import parse_eq, states_an_expression
from tvbo.parse.symbols import assumptions_of, symbol_in


def _declared(element, collection: str):
    """*element*'s named collection, empty when it holds nothing.

    LinkML's dataclasses default an unfilled collection to an empty one and Pydantic's
    models default it to `None`, so a layer written against either alone raises on the
    other for a model that simply declares nothing yet. Stating that difference once here
    is what lets this layer read a model — or one of its elements — from either generator,
    rather than as an `or {}` at each of the two dozen reads below.

    An empty mapping stands in for an empty list as well: every list slot read here is
    either iterated or tested for membership, and both answer the same on either.
    """
    return getattr(element, collection, None) or {}


class SymbolicSystem:
    """A model's equations in SymPy form, parsed against a symbol table built from its names.

    Built for a model by
    [`Dynamics.symbolic_system`](../behaviour/dynamics.qmd#symbolic_system), which keeps one
    per model; construct it directly only to work against a model that does not.

    Before this layer existed each caller re-derived from metadata: loading
    `ZerlautAdaptationSecondOrder` parsed its 27 equations 264 times, and every
    `render_code` and `generate_report` parsed all 27 again because nothing was kept.
    """

    _GROUP_COLLECTIONS = {
        "derived-parameters": "derived_parameters",
        "functions": "functions",
        "derived-variables": "derived_variables",
        "state-equations": "state_variables",
        "output-transformations": "output",
    }
    """Which collection each equation group is built from, and takes its order from."""

    def __init__(self, model):
        self.model = model
        self._cache = None

    def scope(self, include_time_symbol: bool = True, time_dependent: bool = False):
        """Build a unified local_dict for parsing model expressions.

        Includes symbols for parameters, coupling terms, derived parameters, derived
        variables, output transforms, state variables, event names, function names, and
        (optionally) the time symbol 't'.

        Every declared name must appear here so it shadows SymPy's own global namespace:
        `Q` is SymPy's assumptions object, `S` its sympify shortcut, `O` big-O, `N`
        numeric evaluation and `I` the imaginary unit, so a model that names a quantity
        after any of them would otherwise fail to parse.

        Args:
            include_time_symbol: Bind `t` to `Symbol("t")`.
            time_dependent: Bind state and derived variables to `Function(name)(t)` rather
                than `Symbol(name)`, so `Derivative(x(t), t)` stays unevaluated and the
                result reads as a system of ODEs. This is the only difference between the
                two symbolic views of a model — everything downstream of the scope is
                shared, which is why it is a parameter here and not a second builder.

        Returns
        -------
        dict
            Mapping of names to SymPy objects suitable for parse_eq(local_dict=...).
            A copy, so a caller may keep or adapt it; the model's own is cached.
        """
        key = (bool(include_time_symbol), bool(time_dependent))
        scopes = self._state()["scopes"]
        if key not in scopes:
            scopes[key] = self._build_scope(include_time_symbol, time_dependent)
        return dict(scopes[key])

    def form(self, notation: str = "symbol", evaluate: bool = True):
        """The model's equations, parsed once per (notation, evaluate) and remembered.

        Both public views — [`get_equations`](../behaviour/dynamics.qmd#get_equations) and
        [`view`](#tvbo.parse.system.SymbolicSystem.view) — are projections of this, as is
        the function-body table the inliner consumes.

        The cache is discarded whole whenever [`_inputs`](#tvbo.parse.system.SymbolicSystem)
        changes, which is what makes it safe on a mutable model. Rendering is a query —
        `render_code` does not run `update_metadata` — so no consumer can invalidate it
        mid-use.

        Args:
            notation: `"symbol"` binds variables to `Symbol(name)`; `"function"` binds
                them to `Function(name)(t)`.
            evaluate: Let SymPy evaluate right-hand sides, or preserve authored term order.

        Returns:
            `{group: {name: Eq}}` over the five groups, each keyed by the variable it
            defines so no consumer has to recover a name from an `Eq`'s left-hand side.
        """
        forms = self._state()["forms"]
        key = (notation, bool(evaluate))
        if key not in forms:
            forms[key] = self._build_form(notation, evaluate)
        return forms[key]

    def view(self):
        """Full symbolic ODE system using proper SymPy conventions.

        State variables are represented as ``Function(name)(t)`` so that
        ``Derivative(theta(t), t)`` stays unevaluated.  Derived variables
        and derived parameters are included as algebraic equations.

        Returns
        -------
        dict
            ``{'state': [...], 'derived': [...], 'parameters': {...}}``
            where each list contains ``sympy.Eq`` objects and parameters
            maps ``Symbol → value``. That map is keyed by the scope's own
            symbols: rebuilt keys look identical, compare unequal, and make
            substituting it into these equations silently replace nothing.

        Example
        -------
        >>> model.symbolic['state']
        [Eq(Derivative(theta(t), t), I + omega)]
        >>> model.symbolic['derived']
        [Eq(signal(t), sin(theta(t)))]
        >>> model.symbolic['units']
        {omega: 'per_ms', I: None}
        """
        from tvbo.analysis.units import declared_units

        form = self.form(notation="function")
        return {
            "state": list(form["state-equations"].values()),
            "functions": list(form["functions"].values()),
            "derived_parameters": list(form["derived-parameters"].values()),
            "derived": list(form["derived-variables"].values()),
            "parameters": self.keyed_parameters(time_dependent=True),
            "units": declared_units(self.model, scope=self.scope(time_dependent=True)),
        }

    def keyed_parameters(self, time_dependent: bool = False):
        """Each declared parameter's symbol mapped to its value, for substitution.

        Keyed through the scope rather than by minting `Symbol(name)`, because the analysis
        view carries assumptions and a rebuilt key would print identically, compare unequal,
        and make `subs` replace nothing. *time_dependent* selects which view's symbols the
        map is keyed by, and must match the equations it will be substituted into.
        """
        scope = self.scope(time_dependent=time_dependent)
        return {
            scope[str(p.name)]: p.value
            for p in _declared(self.model, "parameters").values()
            if str(p.name) in scope
        }

    def symbol_map(self):
        """Display-symbol overrides for report rendering: ``{identifier Symbol: LaTeX str}``.

        For each element that declares a ``symbol`` (e.g. ``w_+`` for the identifier
        ``w_plus``, or ``S^{(E)}`` for ``S_e``), map its identifier Symbol to the LaTeX
        of that override, so ``sympy.latex(expr, symbol_names=model.symbol_map())``
        renders the source's own notation. Elements without an override are omitted (they
        render from their identifier). Fully sympy-native: the override is itself rendered
        through ``sympy.latex(Symbol(...))``, inheriting Greek/sub/superscript handling.

        Keyed by the canonical collection keys (the identifiers used in the equations),
        over the same element collections as [`scope`](#tvbo.parse.system.SymbolicSystem.scope).
        """
        collections = (
            "parameters",
            "state_variables",
            "derived_variables",
            "derived_parameters",
            "coupling_inputs",
        )
        return {
            Symbol(str(key)): latex(Symbol(str(el.symbol)))
            for collection in collections
            for key, el in _declared(self.model, collection).items()
            if getattr(el, "symbol", None)
        }

    def _state(self):
        """The per-content cache the symbol table and the equations share.

        One invalidation point for both, because they have to agree: a scope built from one
        set of names and equations parsed against another is precisely the drift this layer
        exists to remove.

        A reorder keeps the parsed equations and re-keys them; the scopes are dropped
        instead, since rebuilding a symbol table is a few hundred `Symbol` constructions
        while reparsing is the expensive half.
        """
        content, order = self._inputs()
        cache = self._cache
        if cache is None or cache[0] != content:
            cache = (content, order, {"scopes": {}, "forms": {}})
        elif cache[1] != order:
            reordered = {key: self._reordered(form) for key, form in cache[2]["forms"].items()}
            cache = (content, order, {"scopes": {}, "forms": reordered})
        else:
            return cache[2]
        self._cache = cache
        return cache[2]

    def _inputs(self):
        """What `_build_form` reads, split into content and order.

        The cache is sound only if *content* changes whenever a built equation would, so it
        walks the same collections the builder walks rather than a hand-listed subset: a
        slot the builder starts reading without being added here would serve a stale
        equation forever. Content is compared as dicts, which ignore key order.

        *order* is tracked separately because `sort_equations` reorders collections into
        dependency order without changing a single equation — five times over one load.
        Treating that as a content change would re-parse everything to produce the same
        expressions in a different sequence.
        """
        def _equation(element):
            equation = getattr(element, "equation", None)
            if equation is None:
                return None
            return (
                equation.rhs,
                tuple((c.condition, c.expression) for c in _declared(equation, "conditionals")),
                bool(equation.latex),
            )

        def _assumed(element):
            """Keyed on `assumptions_of` itself, so the key cannot drift from what it reads.

            A `domain` is not an equation, but it decides whether a symbol is `positive` or
            merely `real`, and `Symbol('a', positive=True) != Symbol('a', real=True)`. Naming
            the fields here instead would leave the key stale the day `assumptions_of` starts
            reading one more of them.
            """
            return tuple(sorted(assumptions_of(element).items()))

        content = (
            self.model.system_type,
            {str(name): _assumed(p) for name, p in _declared(self.model, "parameters").items()},
            frozenset(str(name) for name in _declared(self.model, "coupling_inputs")),
            frozenset(str(name) for name in _declared(self.model, "events")),
            frozenset(str(name) for name in _declared(self.model, "output")),
            {str(k): _equation(v) for k, v in _declared(self.model, "derived_parameters").items()},
            {
                str(k): (_equation(v), _assumed(v))
                for k, v in _declared(self.model, "derived_variables").items()
            },
            {
                str(k): (
                    _equation(v),
                    int(v.equation_order) if v.equation_order else 1,
                    _assumed(v),
                )
                for k, v in _declared(self.model, "state_variables").items()
            },
            {
                str(k): (tuple(str(a) for a in _declared(v, "arguments")), _equation(v))
                for k, v in _declared(self.model, "functions").items()
            },
        )
        order = tuple(
            tuple(str(name) for name in _declared(self.model, collection))
            for collection in self._GROUP_COLLECTIONS.values()
        )
        return content, order

    def _reordered(self, form):
        """The same equations, re-keyed into their collections' current order."""
        return {
            group: {
                name: equations[name]
                for name in (str(n) for n in _declared(self.model, self._GROUP_COLLECTIONS[group]))
                if name in equations
            }
            for group, equations in form.items()
        }

    def _build_scope(self, include_time_symbol: bool, time_dependent: bool):
        """Assemble the symbol table. See `scope`.

        Holds only the names the *model* declares. A function's formal arguments are bound by
        that function, exactly as a lambda binds its parameters, and are supplied as an
        overlay while its body is parsed — see `_assemble`. Registering them here
        let a formal shadow a variable of the same name: `ReducedWongWangTvboptim` declares
        both `H(x)` and a derived variable `x`, and the formal won, so the analysis view held
        `x` constant and dropped the chain-rule term from every Jacobian through `H`.

        Assumptions ride on the time-dependent view only. They are what SymPy's analysis
        machinery needs — without `real=True` the fixed points of a two-variable model do
        not come back inside a minute — but they also enter `Symbol.sort_key`, so the same
        product prints as `q*alpha` instead of `alpha*q`. That is no gain for a backend
        that parses, inlines and prints without ever simplifying, and every emitted file is
        compared against a frozen reference. The codegen view therefore stays plain, and
        the two are never mixed: a `Symbol` from one does not compare equal to the same name
        from the other, so nothing can substitute across them by accident.

        Function heads are the exception, and carry `assumptions_of()` in both views: a head
        is notation-independent — `Sigm` is the same function whether the variables around it
        are Symbols or Functions of `t`. Building it per view made `Function("Sigm", real=True)`
        and `Function("Sigm")`, which print identically, compare unequal, and make
        `expr.has(Sigm)` False on an expression that visibly calls it, so every inliner
        matched nothing, silently.
        """
        def _assume(element=None):
            return assumptions_of(element) if time_dependent else {}

        def _symbol(name, element=None):
            return Symbol(str(name), **_assume(element))

        t = _symbol("t")
        scope: dict[str, object] = {}

        def _variable(name, element=None):
            if time_dependent:
                return Function(str(name), **_assume(element))(t)
            return _symbol(name, element)

        if include_time_symbol:
            scope["t"] = t

        for p in _declared(self.model, "parameters").values():
            scope[str(p.name)] = _symbol(p.name, p)

        # Coupling inputs (named inputs from coupling function)
        for ci in _declared(self.model, "coupling_inputs"):
            scope[str(ci)] = _symbol(ci)

        # A derived parameter is constant in time, so it stays a Symbol in both views.
        for name in _declared(self.model, "derived_parameters"):
            scope[str(name)] = _symbol(name)
        derived_variables = _declared(self.model, "derived_variables")
        for name, dv in derived_variables.items():
            scope[str(name)] = _variable(name, dv)

        # Output is a list of string references
        for name in _declared(self.model, "output"):
            scope[str(name)] = _variable(name, derived_variables.get(str(name)))

        for name, sv in _declared(self.model, "state_variables").items():
            scope[str(name)] = _variable(name, sv)

        for fname in _declared(self.model, "functions"):
            scope[str(fname)] = Function(str(fname), **assumptions_of())

        for name in _declared(self.model, "events"):
            scope[str(name)] = _symbol(name)

        if "e" not in scope:
            from sympy import E

            scope["e"] = E

        return scope

    def _build_form(self, notation: str, evaluate: bool):
        """Parse every equation the model states, once. See `form`.

        The two views differ in what they are for, and the evaluation policy follows from
        that rather than the other way round.

        `"symbol"` feeds codegen, which parses, inlines and prints. It honours the caller's
        *evaluate* so a backend can keep the term order its author wrote.

        `"function"` is the analysis view — the one `Matrix.jacobian`, `solve` and `dsolve`
        act on — so it is canonical. It used to suppress evaluation globally, which kept
        `Derivative(theta(t), t)` from collapsing but also left the right-hand sides in a
        nested unevaluated form that SymPy's solvers cannot make progress on: asked for the
        fixed points of `Generic2dOscillator` in that form, `solve` returns nothing in 45 s;
        canonical and with real symbols it answers in under one. `Derivative` is built
        explicitly here, so nothing needs the global suppression to survive.
        """
        time_dependent = notation == "function"
        return self._assemble(
            time_dependent=time_dependent,
            evaluate=True if time_dependent else evaluate,
        )

    def _assemble(self, time_dependent: bool, evaluate: bool):
        """Build the five equation groups against one scope. See `_build_form`.

        Every symbol an equation's left-hand side names is resolved through `scope` — the
        same table the right-hand sides were parsed against. Minting one here instead
        produces a name that prints identically and compares unequal once the analysis view
        attaches assumptions, and `subs` across that mismatch replaces nothing rather than
        raising: a derivative taken w.r.t. a freshly built `t` leaves `doit()` returning 0,
        and a derived parameter's definition substitutes into none of its own equations.
        """
        scope = self.scope(time_dependent=time_dependent)
        t = symbol_in(scope, "t")
        discrete = self.model.system_type == "discrete"
        derived_variables = _declared(self.model, "derived_variables")
        state_variables = _declared(self.model, "state_variables")

        def _lhs(name):
            return symbol_in(scope, name)

        def _states(element):
            """Whether the element has anything to parse — see `states_an_expression`.

            An element declared with no `rhs` and no conditionals is skipped rather than
            parsed. Every rendering path funnels through here, so one such element used
            to break all of them at once instead of only `get_equations`.
            """
            return states_an_expression(getattr(element, "equation", None))

        def _parse(element, namespace=None):
            return parse_eq(element.equation, local_dict=namespace or scope, evaluate=evaluate)

        def _formal(name):
            """A function's bound argument — a quantity, never a state, so never `name(t)`."""
            return Symbol(str(name), **(assumptions_of() if time_dependent else {}))

        def _function_scope(function):
            """The model's names with this function's formals bound over them."""
            return {**scope, **{str(a): _formal(a) for a in _declared(function, "arguments")}}

        form = {
            "derived-parameters": {
                str(k): Eq(lhs=_lhs(k), rhs=_parse(dp))
                for k, dp in _declared(self.model, "derived_parameters").items()
                if _states(dp)
            },
            "functions": {
                str(k): Eq(
                    lhs=_lhs(k)(*[_formal(a) for a in _declared(f, "arguments")]),
                    rhs=_parse(f, _function_scope(f)),
                )
                for k, f in _declared(self.model, "functions").items()
                if _states(f) and _declared(f, "arguments")
            },
            "derived-variables": {
                str(k): Eq(lhs=_lhs(k), rhs=_parse(dv))
                for k, dv in derived_variables.items()
                if _states(dv)
            },
            "state-equations": {},
            "output-transformations": {},
        }

        for k, sv in state_variables.items():
            if not _states(sv):
                continue
            order = int(sv.equation_order or 1)
            lhs = _lhs(k) if discrete else Derivative(_lhs(k), *([t] * order))
            form["state-equations"][str(k)] = Eq(lhs=lhs, rhs=_parse(sv))

        # An identity equation for an output that IS a state variable overwrites its real one.
        for name in _declared(self.model, "output"):
            name = str(name)
            if name in derived_variables:
                if _states(derived_variables[name]):
                    form["output-transformations"][name] = Eq(
                        lhs=_lhs(name), rhs=_parse(derived_variables[name])
                    )
            elif name not in state_variables:
                raise ValueError(
                    f"Output variable '{name}' not found in derived_variables or state_variables"
                )

        return form
