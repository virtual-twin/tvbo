"""One integration method, one canonical name, whatever the recipe spells it.

The schema advertises ``rk4`` as a method (``Integrator.method``: "Integration method (euler, heun, rk4, etc.)"), and the tvboptim adapter accepted it, so recipes wrote it and ran. The ontology that holds the method's symbolic update expression knew only ``RungeKutta4thOrder``, so the same recipe rendered on tvboptim and died in the tvb template on ``'NoneType' object has no attribute 'equation'`` — the spelling had left ``update_expression`` unfilled three layers up, and nothing on the way down said so.

The spellings were also written out five separate times: once in the adapter and once in each of four mako templates, disagreeing about which they accepted, and two of them resolving an unknown method to ``Euler`` — silently integrating a fourth-order recipe by a first-order scheme. So the table is now one table, and a miss raises.
"""

from __future__ import annotations

import pytest

from tvbo.adapters.tvboptim import SOLVER_MAP, solver_class
from tvbo.utils import INTEGRATION_METHODS, integration_method


@pytest.mark.parametrize(
    ("spelling", "canonical"),
    [
        ("rk4", "RungeKutta4thOrder"),
        ("RK4", "RungeKutta4thOrder"),
        ("runge_kutta", "RungeKutta4thOrder"),
        ("RungeKutta4thOrder", "RungeKutta4thOrder"),
        ("RungeKutta4thOrderDeterministic", "RungeKutta4thOrder"),
        ("euler", "Euler"),
        ("Euler", "Euler"),
        ("heun", "Heun"),
        ("HeunStochastic", "Heun"),
    ],
)
def test_every_spelling_names_one_method(spelling, canonical):
    assert integration_method(spelling) == canonical


def test_an_unknown_spelling_raises():
    """Resolving to a default would integrate by a scheme the recipe did not ask for."""
    with pytest.raises(ValueError, match="unknown integration method"):
        integration_method("rk5")


def test_a_backend_supplied_solver_name_still_loads():
    """`Integrator.method` is an open vocabulary where the backend brings its own solver.

    A NetworkDynamics.jl recipe writes `method: AutoTsit5` and that string is handed to Julia's `solve`; the tvb and tvboptim backends reject it when they are asked to render an update step, but `enrich()` runs at construction, so raising there stops the study loading at all.
    """
    from tvbo.datamodel.schema import Integrator

    assert integration_method("AutoTsit5", strict=False) is None
    integrator = Integrator(method="AutoTsit5", duration=10.0, step_size=0.1)
    assert integrator.enrich().method == "AutoTsit5"
    assert integrator.ontoclass is None


def test_the_canonical_names_are_the_database_entries():
    """The canonical key has to be the registry entry, since that is what carries the update expression."""
    from tvbo.data.registry import list_entries

    assert set(list_entries("Integrator")) <= set(INTEGRATION_METHODS)


def test_rk4_reaches_its_update_expression():
    """The failure this fixes: the spelling resolved for tvboptim and left the symbolic path empty."""
    from tvbo.datamodel.schema import Integrator

    integrator = Integrator(method="rk4", duration=10.0, step_size=1.0)
    integrator.enrich()
    assert integrator.update_expression.equation.rhs == "(dt / 6) * (dX0 + 2.0 * dX1 + 2.0 * dX2 + dX3)"


def test_the_solver_map_is_keyed_by_the_canonical_name():
    assert set(SOLVER_MAP) <= set(INTEGRATION_METHODS)
    assert solver_class("rk4") == solver_class("RungeKutta4thOrder") == "RungeKutta4"


def test_a_method_tvboptim_cannot_integrate_raises():
    """``SOLVER_MAP.get(method, 'Euler')`` — what four templates did — reports a first-order run as the recipe's own."""
    with pytest.raises(NotImplementedError, match="no tvboptim solver"):
        solver_class("VODE")


def test_no_template_carries_its_own_spelling_table():
    """Five copies disagreed about which spellings they accepted; the adapter now holds the only one."""
    from pathlib import Path

    templates = Path(__file__).resolve().parent.parent / "tvbo" / "templates"
    offenders = [p.name for p in templates.rglob("*.mako") if "SOLVER_MAP = {" in p.read_text(encoding="utf-8")]
    assert offenders == []
