"""A stimulus event may declare its constants under ``equation.parameters``.

Codegen reads an event's constants from ``ev.parameters``, so the loader merges the
equation-level ones onto the event. That merge used to *assign* a dict to the multivalued slot,
which LinkML re-coerces to the list form — after which every ``dict(ev.parameters)`` downstream
raised ``dictionary update sequence element #0 has length N``, an error naming neither the event
nor the slot. The merge mutates in place now.

Both spellings must reach the same emitted signal, since a recipe may put a constant next to the
equation that uses it or on the event itself.
"""

import pytest

pytest.importorskip("jax")
pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment  # noqa: E402

SPEC = """
label: Stimulus constants
dynamics:
  name: Driven
  parameters:
    k: {value: 0.5}
    I_ext: {value: 0.0}
  state_variables:
    x: {equation: {rhs: "-k*x + I_ext + c_in"}, initial_value: 0.1}
  coupling_inputs: {c_in: {}}
network:
  label: Solo
  number_of_nodes: 1
  nodes: [{id: 0, label: A, dynamics: Driven}]
integration: {method: euler, step_size: 1.0, duration: 20.0, unit: ms}
events:
  I_ext:
    event_type: stimulus
    equation:
%(eq_tail)s
%(event_tail)s
observations:
  x_last: {source: [x], aggregation: last}
"""

ON_EQUATION = {
    "eq_tail": '      rhs: "amp * sin(2 * pi * f_inj * t)"\n      parameters:\n        amp: {value: 2.0}\n        f_inj: {value: 0.01}',
    "event_tail": "",
}
ON_EVENT = {
    "eq_tail": '      rhs: "amp * sin(2 * pi * f_inj * t)"',
    "event_tail": "    parameters:\n      amp: {value: 2.0}\n      f_inj: {value: 0.01}",
}


def _signal(spec_kwargs):
    exp = SimulationExperiment.from_string(SPEC % spec_kwargs)
    assert isinstance(exp.events["I_ext"].parameters, dict), "the merge must leave a keyed mapping, not the list form"
    code = exp.render_code("tvboptim")
    return [ln.strip() for ln in code.splitlines() if "jnp.sin" in ln and "signal" in ln]


def test_equation_level_parameters_render_the_same_signal_as_event_level():
    on_equation, on_event = _signal(ON_EQUATION), _signal(ON_EVENT)
    assert on_equation, "no stimulus signal was emitted"
    assert on_equation == on_event


def test_the_experiment_with_equation_level_parameters_runs():
    """The failure this guards was at render time, but a run also exercises the external-input path."""
    SimulationExperiment.from_string(SPEC % ON_EQUATION).run("tvboptim")
