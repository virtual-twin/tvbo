"""An optimization's loss calls a name the generated code defines, or the spec is refused.

A loss given as ``callable: {module, name}`` is imported from its module the way a ``functions:`` callable is, whether or not ``functions:`` lists it; a loss given as ``function:`` calls the Function the experiment declares. A loss naming a function nothing defines is refused when the spec is loaded, instead of being generated into a ``loss_fn`` that raises ``NameError`` on its first evaluation.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tvbo import SimulationExperiment

pytest.importorskip("tvboptim")

pytestmark = pytest.mark.backend_tvboptim

MODULE = "g305_study_losses"
"""The study-local module the callable losses are imported from, written per test beside the spec."""

SPEC = """
id: 1
dynamics:
  name: Leak
  parameters:
    k: {value: 1.0}
    tau: {value: 5.0}
  state_variables:
    x: {initial_value: 0.0, equation: {rhs: "(k - x) / tau"}}
  output: [x]
network:
  number_of_nodes: 1
  nodes: [{id: 0, label: r0}]
  edges: []
integration: {method: euler, step_size: 0.1, duration: 4.0, unit: ms}
FUNCTIONS
observations:
  mean_x: {source: [x], aggregation: mean}
optimizations:
  fit:
    loss: {LOSS, arguments: {x: {value: observations.mean_x.data}}}
    free_parameters: [{parameter: Leak.k}]
    algorithm: adam
    learning_rate: 0.01
    max_iterations: 2
"""

DECLARED = 'functions:\n  squared_mean: {source_code: "jnp.mean(x ** 2)", arguments: {x: {}}}'
"""The same loss as the module's, declared under ``functions:``."""

LISTED = f"functions:\n  squared_mean: {{callable: {{module: {MODULE}, name: squared_mean}}, arguments: {{x: {{}}}}}}"
"""The module's loss, also listed under ``functions:``."""


@pytest.fixture
def study_module(tmp_path, monkeypatch):
    """A study-local loss module on ``sys.path``, as a recipe's ``code/`` puts one."""
    (tmp_path / f"{MODULE}.py").write_text("import jax.numpy as jnp\n\n\ndef squared_mean(x):\n    return jnp.mean(x ** 2)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, MODULE, raising=False)


def _experiment(loss: str, functions: str = "") -> SimulationExperiment:
    return SimulationExperiment.from_string(SPEC.replace("FUNCTIONS", functions).replace("LOSS", loss))


def _loss_history(experiment: SimulationExperiment) -> np.ndarray:
    """The fit's recorded loss at each step."""
    ((_, fit),) = experiment.run("tvboptim", mode="optimization", quiet=True).optimizations.items()
    return np.asarray(fit.loss_trajectory, dtype=float)


def test_a_loss_callable_naming_its_module_is_imported_and_runs(study_module):
    """Not listed under ``functions:``, the callable is imported from its module, and the fit it drives matches the same loss declared as a Function."""
    imported = _experiment(f"callable: {{module: {MODULE}, name: squared_mean}}")
    assert f"from {MODULE} import squared_mean\n" in imported.render_code("tvboptim")
    history = _loss_history(imported)
    assert history.size and np.all(np.isfinite(history)), history
    np.testing.assert_allclose(history, _loss_history(_experiment("function: squared_mean", DECLARED)), rtol=1e-12, atol=0)


def test_a_loss_callable_listed_under_functions_renders_as_the_function_it_names(study_module):
    """A callable ``functions:`` already lists is imported once, by that entry, and the code is the code of a ``function:`` reference to it."""
    listed = _experiment(f"callable: {{module: {MODULE}, name: squared_mean}}", LISTED).render_code("tvboptim")
    assert listed.count(f"from {MODULE} import squared_mean\n") == 1
    assert listed == _experiment("function: squared_mean", LISTED).render_code("tvboptim")


def test_a_loss_function_declared_under_functions_imports_nothing():
    """A ``function:`` loss calls the Function the experiment declares, defined in the code and imported from nowhere."""
    code = _experiment("function: squared_mean", DECLARED).render_code("tvboptim")
    assert "def squared_mean(" in code and "import squared_mean" not in code


@pytest.mark.parametrize(
    "loss", ["function: fc_corr_loss", "callable: {name: fc_corr_loss}"], ids=["function", "callable-without-module"]
)
def test_a_loss_naming_a_function_nothing_defines_is_refused(loss):
    """Nothing declares or imports the name, so the spec is refused naming the optimization and the missing function."""
    with pytest.raises(
        ValueError,
        match=r"optimization 'fit': its loss calls 'fc_corr_loss', which no `functions:` entry declares and no `callable: \{module: \.\.\.\}` imports",
    ):
        _experiment(loss, DECLARED)
