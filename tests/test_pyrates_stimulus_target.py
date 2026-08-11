"""A PyRates stimulus drives the variable the recipe names, not `I_ext` by assumption.

`_build_inputs` read `target_variable` off `experiment.stimulation`. That slot holds a
`Stimulus`, which has no such attribute at all — TVBO names a stimulus target by the
*event*, and `SimulationExperiment._resolve_events` lowers a declarative `target_variable`
onto the event's `name`. So the read returned its `I_ext` default for every experiment ever
run, and PyRates was handed an input key for a variable the model need not declare.
"""

import pytest

pytest.importorskip("pyrates")

from tvbo.adapters.pyrates import PyRatesAdapter, _legacy_input_variable  # noqa: E402
from tvbo.classes.experiment import SimulationExperiment  # noqa: E402

RECIPE = """
label: PyRates stimulus target
network:
  number_of_nodes: 1
  dynamics:
    Drive:
      name: Drive
      parameters: {tau: {value: 10.0}}
      state_variables:
        v: {equation: {rhs: '(-v + P)/tau'}, initial_value: 0.0}
      coupling_inputs: {P: {}}
  nodes:
    - {id: 0, label: N0, dynamics: Drive}
events:
  P:
    event_type: stimulus
    equation: {rhs: 'amplitude'}
    parameters: {amplitude: {value: 1.0}}
    regions: [0]
    weighting: [1.0]
stimulation:
  label: drive
  equation: {rhs: 'amplitude'}
  parameters: {amplitude: {value: 1.0}}
  regions: [0]
  weighting: [1.0]
integration: {method: euler, step_size: 0.1, duration: 5.0}
"""


def _experiment(text, tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text(text)
    return SimulationExperiment.from_file(str(path))


def test_the_target_variable_comes_from_the_stimulus_event(tmp_path):
    """The event is named `P`, which is the variable the dynamics actually reads."""
    adapter = PyRatesAdapter(_experiment(RECIPE, tmp_path))

    assert adapter._stimulus_target_variable() == "P"


def test_the_generated_input_key_addresses_that_variable(tmp_path):
    """The whole point: the key used to end in `/I_ext`, which this model never declares."""
    inputs = PyRatesAdapter(_experiment(RECIPE, tmp_path))._build_inputs()

    assert inputs, "the stimulus produced no PyRates input at all"
    key = next(iter(inputs))
    assert key.endswith("/P"), key
    assert "I_ext" not in key


def test_a_stimulus_with_no_event_and_no_I_ext_says_so(tmp_path):
    """Guessing `I_ext` addressed a variable that need not exist; that is now an error."""
    recipe = RECIPE.replace("event_type: stimulus", "event_type: discrete")
    adapter = PyRatesAdapter(_experiment(recipe, tmp_path))

    with pytest.raises(ValueError, match="No stimulus event names the variable"):
        adapter._build_inputs()


def test_a_model_declaring_I_ext_keeps_the_legacy_path_through_build_inputs(tmp_path):
    """Through `_build_inputs`, not the helper alone.

    The lookup consulted `experiment.dynamics` while `dyn_name` keys the network's library,
    so it always missed and a legacy `stimulation:`-only recipe raised instead of falling
    back — a regression the helper-only test could not see.
    """
    recipe = (RECIPE
              .replace("event_type: stimulus", "event_type: discrete")
              .replace("{tau: {value: 10.0}}", "{tau: {value: 10.0}, I_ext: {value: 0.0}}")
              .replace("(-v + P)/tau", "(-v + I_ext)/tau"))
    inputs = PyRatesAdapter(_experiment(recipe, tmp_path))._build_inputs()

    assert inputs, "the legacy fallback produced no PyRates input at all"
    assert [k for k in inputs if k.endswith("/I_ext")] == list(inputs)


def test_the_legacy_default_is_kept_only_where_the_model_declares_it(tmp_path):
    """`I_ext` stays available to models that really have it, and only to those."""
    from types import SimpleNamespace

    with_it = SimpleNamespace(state_variables={}, parameters={"I_ext": object()},
                              coupling_inputs={}, derived_variables={})
    without = SimpleNamespace(state_variables={"v": object()}, parameters={},
                              coupling_inputs={}, derived_variables={})

    assert _legacy_input_variable(with_it) == "I_ext"
    assert _legacy_input_variable(without) is None
    assert _legacy_input_variable(None) is None
