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
    """
    model.get_equations()
    count_parses.clear()

    # In place, the way `sort_equations` does it — assigning to a multivalued slot
    # replaces it with a `JsonObj` and is not how this collection is ever reordered.
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

    assert list(form["state-equations"]) == [
        key for key in model.state_variables if model.state_variables[key].equation
    ]
    assert list(form["derived-variables"]) == list(model.derived_variables)
    assert list(form["derived-parameters"]) == list(model.derived_parameters)
    assert list(form["functions"]) == list(model.functions)
