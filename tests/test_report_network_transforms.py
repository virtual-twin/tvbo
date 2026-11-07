"""A network transform appears in the experiment report the way `Network._apply_transform` runs it.

A transform is declared by a `callable` or by an `equation`. The report read every transform's ``equation.rhs``, so a callable transform (the NetworkDynamics.jl FitzHugh-Nagumo page's `jax.numpy.transpose`) raised ``AttributeError: 'NoneType' object has no attribute 'rhs'`` and the page did not render. A transform that declares neither leaves the matrix as it is and gets no row.
"""

from tvbo import SimulationExperiment

SPEC = """
label: Transformed network
dynamics:
  name: Relax
  parameters: {k: {value: 0.5}}
  state_variables:
    x: {equation: {rhs: "k - x"}, initial_value: 0.0}
network:
  number_of_nodes: 2
  transforms:
    - name: weight
      callable: {module: jax.numpy, name: power}
      arguments: {x2: {value: 2}}
    - name: length
      equation: {rhs: "length / 2"}
    - name: delay
integration: {method: euler, step_size: 1.0, duration: 10.0}
"""


def _rows() -> dict[str, str]:
    report = SimulationExperiment.from_string(SPEC).generate_report()
    text = report if isinstance(report, str) else str(report)
    return {
        line.split("(", 1)[1].split(")", 1)[0]: line.split("|")[2].strip()
        for line in text.splitlines()
        if line.startswith("| Transform (")
    }


def test_a_callable_transform_shows_the_call_it_runs():
    assert _rows()["weight"] == "`jax.numpy.power(M, x2=2)`"


def test_an_equation_transform_shows_its_equation():
    row = _rows()["length"]
    assert row.startswith("$M_{\\text{out}} = ") and "length" in row


def test_a_transform_that_declares_neither_gets_no_row():
    assert "delay" not in _rows()
