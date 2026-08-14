"""One symbolic layer between a model's metadata and everything rendered from it.

`metadata → symbolic → render`. Every consumer — codegen, the report, the function
inliner, `get_equations`, `symbolic` — reads equations that were parsed once, against one
symbol table, and kept. Before this there were three independent builders with three
hand-made scopes: loading `ZerlautAdaptationSecondOrder` parsed its 27 equations 264 times,
and each `render_code` and `generate_report` parsed all 27 again.

The tests here pin the two properties that make the layer worth having and the two that
make caching safe on a mutable model:

* an equation is parsed once, however many consumers ask for it;
* the two public views are projections of that one parse, not re-derivations;
* changing an equation invalidates the cache;
* the cache never reaches the serialised record.
"""

from __future__ import annotations

import pytest

from tvbo.classes.dynamics import Dynamics
from tvbo.data.registry import database_dir

MODEL_ROOT = database_dir("Dynamics")


@pytest.fixture
def model() -> Dynamics:
    """A model with derived variables, derived parameters and several state equations."""
    return Dynamics.from_file(str(MODEL_ROOT / "JansenRit.yaml"))


@pytest.fixture
def count_parses(monkeypatch):
    """Count calls into the parser, wherever they come from."""
    import tvbo.parse.expression as expression

    calls = []
    original = expression.parse_expr

    def counting(expr, *args, **kwargs):
        calls.append(str(expr))
        return original(expr, *args, **kwargs)

    monkeypatch.setattr(expression, "parse_expr", counting)
    return calls


@pytest.mark.backend_core
def test_rendering_reuses_the_parsed_equations(model: Dynamics, count_parses):
    """Rendering a model, in any format and any number of times, parses nothing new.

    The first render populates the layer; everything after reads it. A regression here
    means a consumer has gone back to parsing metadata directly.
    """
    model.render_code(format="jax")
    count_parses.clear()

    model.render_code(format="jax")
    model.render_code(format="numpy")
    model.generate_report()

    assert count_parses == [], f"{len(count_parses)} re-parses: {count_parses[:3]}"


@pytest.mark.backend_core
def test_both_views_come_from_one_parse(model: Dynamics, count_parses):
    """`get_equations` and `symbolic` are projections, so a second call costs nothing."""
    model.get_equations()
    model.symbolic
    count_parses.clear()

    model.get_equations()
    model.get_equations(format="dict")
    model.get_equations(format="state-equations")
    model.symbolic

    assert count_parses == []


@pytest.mark.backend_core
def test_the_two_views_agree_on_every_equation(model: Dynamics):
    """Same equations, differing only in how a variable is written.

    `get_equations` binds `y0` to a `Symbol`; `symbolic` binds it to `y0(t)` so that
    `Derivative(y0(t), t)` survives. If the two ever disagreed on *which* equations exist,
    a report and its generated code would be describing different models.
    """
    from tvbo.utils import report

    flat = model.get_equations()
    ode = model.symbolic

    assert [report.equation_name(eq) for eq in ode["state"]] == list(model.state_variables)
    assert set(model.state_variables) <= set(flat)
    for equation in ode["state"]:
        assert equation.lhs.expr.func.__name__ in model.state_variables


@pytest.mark.backend_core
def test_changing_an_equation_invalidates_the_cache(model: Dynamics):
    """The property that makes caching sound on a mutable model."""
    before = model.get_equations()["y0"]

    model.state_variables["y0"].equation.rhs = "y3 * 2"
    after = model.get_equations()["y0"]

    assert after != before, "edited equation still served from the cache"
    assert str(after.rhs) == "2*y3"


@pytest.mark.backend_core
def test_reordering_does_not_reparse_but_does_reorder(model: Dynamics, count_parses):
    """Sorting into dependency order rearranges equations; it does not change any.

    `update_metadata` sorts three collections on every load, so treating a reorder as a
    content change would re-parse the whole model several times over for no new result.

    The collection is reordered in place, the way `sort_equations` does it: assigning to a
    multivalued slot replaces it with a `JsonObj`, which is not how this is ever reordered.
    """
    model.get_equations()
    count_parses.clear()

    reversed_items = dict(reversed(list(model.derived_variables.items())))
    model.derived_variables.clear()
    model.derived_variables.update(reversed_items)
    reordered = model.get_equations(format="dict")["derived-variables"]

    assert count_parses == [], "a reorder re-parsed the equations"
    assert [str(eq.lhs) for eq in reordered] == list(model.derived_variables)


@pytest.mark.backend_core
def test_the_cache_never_reaches_the_serialised_record(model: Dynamics):
    """`tvbo/database/` is the published record; a SymPy cache must not appear in it.

    `yaml.SafeDumper` cannot represent a SymPy object, so without the `_items` filter this
    fails loudly — but a filter that stopped working would be caught by nothing else.
    """
    model.get_equations()
    assert "_symbolic_cache" in model.__dict__

    dumped = model.to_yaml()
    assert "_symbolic_cache" not in dumped
    assert "sympy" not in dumped


@pytest.mark.backend_core
@pytest.mark.parametrize("name", ["JansenRit", "Epileptor", "ZerlautAdaptationSecondOrder"])
def test_every_group_is_keyed_by_the_name_it_defines(name: str):
    """Consumers read names from the layer instead of recovering them from an `Eq`."""
    model = Dynamics.from_file(str(MODEL_ROOT / f"{name}.yaml"))
    form = model._symbolic_form()

    assert list(form["state-equations"]) == [key for key in model.state_variables if model.state_variables[key].equation]
    assert list(form["derived-variables"]) == list(model.derived_variables)
    assert list(form["derived-parameters"]) == list(model.derived_parameters)
    assert list(form["functions"]) == list(model.functions)


@pytest.mark.backend_core
def test_the_analysis_view_carries_assumptions(model: Dynamics):
    """SymPy is told what the schema already knows: these quantities are real.

    Without it every symbol is `real=None` and SymPy must consider complex branches, which
    is the difference between an analysis returning and not.
    """
    equations = model.symbolic["state"]
    symbols = {s for eq in equations for s in eq.rhs.free_symbols}
    assert symbols, "no free symbols to check"
    assert all(s.is_real for s in symbols), sorted(str(s) for s in symbols if not s.is_real)


@pytest.mark.backend_core
def test_a_declared_domain_becomes_a_sign_assumption():
    """A lower bound at or above zero is the one further thing a domain clearly implies."""
    model = Dynamics.from_file(str(MODEL_ROOT / "Generic2dOscillator.yaml"))
    scope = model.get_symbolic_elements(time_dependent=True)
    bounded = {
        name: parameter
        for name, parameter in model.parameters.items()
        if getattr(parameter, "domain", None) is not None and parameter.domain.lo is not None
    }
    assert bounded, "fixture no longer declares a bounded parameter"
    for name, parameter in bounded.items():
        symbol = scope[str(name)]
        if parameter.domain.lo > 0:
            assert symbol.is_positive, f"{name} has lo={parameter.domain.lo} but is not positive"
        elif parameter.domain.lo == 0:
            assert symbol.is_nonnegative, f"{name} has lo=0 but is not nonnegative"


@pytest.mark.backend_core
def test_the_codegen_view_stays_plain(model: Dynamics):
    """Assumptions enter `Symbol.sort_key`, so they reorder printed products.

    A backend that parses, inlines and prints without ever simplifying gains nothing from
    them and every emitted file is compared against a frozen reference, so the codegen view
    is deliberately plain. This also keeps the two views unmixable: a symbol from one never
    compares equal to the same name from the other.
    """
    codegen = model.get_symbolic_elements()
    analysis = model.get_symbolic_elements(time_dependent=True)
    name = next(iter(model.parameters))
    assert codegen[name].is_real is None
    assert analysis[name].is_real is True
    assert codegen[name] != analysis[name]


@pytest.mark.backend_core
def test_the_analysis_view_is_a_system_of_odes(model: Dynamics):
    """`Derivative(y0(t), t)` must survive `doit()`, which pins the two `t`s together.

    The derivative is taken with respect to a symbol resolved from the scope, so it is the
    same object as the one inside `y0(t)`. Built fresh it is not, and SymPy then reads the
    left-hand side as a derivative with respect to an absent variable and evaluates it to
    **zero** — while `str()` still prints `Derivative(y0(t), t)`, which is why nothing else
    in this suite noticed. Every consumer that treats `symbolic` as a system of ODEs
    (`Matrix.jacobian`, `dsolve`, fixed-point `solve`) is handed `0 = rhs` instead.
    """
    equation = model.symbolic["state"][0]

    assert equation.lhs.doit() == equation.lhs, f"{equation.lhs} evaluates to {equation.lhs.doit()}"
    assert equation.lhs.doit() != 0


@pytest.mark.backend_core
@pytest.mark.parametrize("name", ["Jansen1995", "JansenRit", "ZerlautAdaptationSecondOrder"])
def test_every_left_hand_side_is_the_symbol_the_equations_use(name: str):
    """A left-hand side is resolved from the scope, never minted beside it.

    `Symbol("x") != Symbol("x", real=True)` and `Function("f") != Function("f", real=True)`;
    both pairs `srepr` identically and `subs` across either does nothing rather than
    raising. So a derived parameter whose left-hand side was rebuilt substitutes into none
    of its own equations, and a function's formal argument does not occur in its own body.
    """
    model = Dynamics.from_file(str(MODEL_ROOT / f"{name}.yaml"))
    scope = model.get_symbolic_elements(time_dependent=True)
    form = model._symbolic_form(notation="function")

    for equation in form["derived-parameters"].values():
        assert equation.lhs is scope[str(equation.lhs)]

    for fname, equation in form["functions"].items():
        assert type(equation.lhs) is scope[fname], f"{fname} head was rebuilt"
        for formal in equation.lhs.args:
            assert formal in equation.rhs.free_symbols or not equation.rhs.free_symbols

    definitions = {eq.lhs: eq.rhs for eq in form["derived-parameters"].values()}
    if definitions:
        for equation in form["state-equations"].values():
            names = {str(s) for s in equation.rhs.free_symbols}
            if names & {str(k) for k in definitions}:
                assert equation.rhs.subs(definitions) != equation.rhs


@pytest.mark.backend_core
@pytest.mark.parametrize("name", ["ReducedWongWangTvboptim", "ReducedWongWangFunc"])
def test_a_function_formal_does_not_shadow_a_variable(name: str):
    """`H(x)`'s bound `x` and a derived variable `x` are different quantities.

    Both models declare a function `H(x)` and a derived variable `x`. Registering formals in
    the model-wide table let the formal win, so the analysis view held `x` constant: its own
    definition read `Eq(x, … S(t) …)` — a constant equal to a function of time — and every
    Jacobian taken through `H` silently lost the chain-rule term.

    A formal is bound by its function, like a lambda parameter. Keeping it out of the model's
    namespace is also what lets a function be declared once and reused, where its argument is
    not tied to any one host's variables.
    """
    from sympy import Symbol

    # `ReducedWongWangFunc` is one of the two curated models spelled `.yml`.
    path = next(p for ext in ("yaml", "yml") for p in MODEL_ROOT.rglob(f"{name}.{ext}"))
    model = Dynamics.from_file(str(path))
    scope = model.get_symbolic_elements(time_dependent=True)
    form = model._symbolic_form(notation="function")

    assert scope["x"] == form["derived-variables"]["x"].lhs, "the variable lost its own name"
    assert scope["x"].func.__name__ == "x", "the derived variable is not time-dependent"

    formal = form["functions"]["H"].lhs.args[0]
    assert isinstance(formal, Symbol), "a bound formal must not be a function of time"
    assert formal != scope["x"], "the formal and the variable are the same object"

    state = next(iter(form["state-equations"].values()))
    assert "Derivative(x(t), t)" in str(state.rhs.diff(scope["t"])), "chain-rule term lost"


@pytest.mark.backend_core
def test_editing_a_domain_invalidates_the_cache():
    """A `domain` is not an equation, but it decides a symbol's sign assumption.

    `assumptions_of` reads `domain.lo`, so a cache keyed only on equations serves symbols
    carrying the old assumptions — and those compare unequal to freshly parsed ones, so
    nothing substitutes across the two.
    """
    model = Dynamics.from_file(str(MODEL_ROOT / "Generic2dOscillator.yaml"))
    name = next(
        n
        for n, p in model.parameters.items()
        if getattr(p, "domain", None) is not None and p.domain.lo is not None and p.domain.lo < 0
    )
    assert model.get_symbolic_elements(time_dependent=True)[name].is_positive is not True

    model.parameters[name].domain.lo = 1.0

    assert model.get_symbolic_elements(time_dependent=True)[name].is_positive is True


@pytest.mark.backend_core
def test_the_parameter_map_substitutes_into_its_own_equations(model: Dynamics):
    """`symbolic["parameters"]` must be keyed by the symbols the equations actually use.

    Rebuilding those keys yields names that look identical and compare unequal, so the
    substitution silently replaces nothing — the failure mode assumptions introduce.
    """
    symbolic = model.symbolic
    substituted = symbolic["state"][0].rhs.subs(symbolic["parameters"])
    remaining = {str(s) for s in substituted.free_symbols}
    assert not (remaining & set(model.parameters)), f"parameters left unsubstituted: {remaining}"
