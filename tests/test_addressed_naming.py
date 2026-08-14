"""An addressed name has to be a name, and it is held to it where the recipe says it.

Three keyed slots carry a name the rest of the spec refers to rather than displays, and
every Python backend turns each into a definition or a binding:

* an ``events`` key is the part before the dot in ``priors: {stimulus.amplitude: ...}``,
  a free symbol in the right-hand side the author writes, and five separate positions in
  the rendered module;
* an ``observations`` key is named by ``source:`` and rendered as a ``def`` under jax;
* a ``functions`` key is named by the pipeline step that calls it, and likewise rendered.

Mangling such a name into an identifier is not open. The author's own equations and
dotted references still spell it the original way, an unrecognised dotted prefix resolves
to the dynamics scope rather than raising, and for events a partial mangling is not even a
syntax error — it is a ``hasattr`` that fails, substituting 0.0 for the stimulus.

These tests assert outcomes rather than spellings: an accepted name reaches the generated
module in each position it is emitted in, and a rejected one is reported against the
recipe key as the recipe is read.
"""

from __future__ import annotations

import pytest
import yaml

from tvbo.classes.experiment import SimulationExperiment

EVENT_RECIPE = """
id: 1
label: event naming
dynamics:
  name: Generic2dOscillator
  iri: tvbo:Generic2dOscillator
  state_variables:
    V:
      equation:
        rhs: "d*tau*(I*gamma - V**3*f + V**2*e + V*g + W*alpha) + {name}"
      coupling_variable: true
      initial_value: 0.0
    W:
      equation:
        rhs: "d*(V**2*c + V*b - W*beta + a)/tau"
      initial_value: 0.0
integration: {{method: Heun, step_size: 0.1, duration: 10.0}}
events:
  {name}:
    label: "Pulse stimulus"
    event_type: stimulus
    equation:
      rhs: "Piecewise((amplitude, (t >= onset) & (t < onset + duration)), (0.0, True))"
    parameters:
      amplitude: {{value: 0.4}}
      onset: {{value: 2.0}}
      duration: {{value: 1.0}}
    nodes: [0]
    weights: [1.0]
"""

MEMBER_RECIPE = """
id: 1
label: member naming
dynamics:
  name: Generic2dOscillator
  iri: tvbo:Generic2dOscillator
integration: {{method: Heun, step_size: 0.1, duration: 10.0}}
functions:
  {function}:
    description: "Subsample the recording."
    source_code: "data[::2, 0, 0]"
    arguments:
      data: {{}}
observations:
  {observation}:
    label: "Recorded V"
    source: [V]
    pipeline:
      - function: {function}
        arguments: {{data: V}}
"""

REJECTED = {
    "a space": "flash burst",
    "a hyphen": "flash-burst",
    "a leading digit": "100Hz",
    "a dot": "flash.burst",
    "a keyword": "lambda",
}


def _event_experiment(name: str) -> SimulationExperiment:
    """The recipe with *name* in every place the author would write it."""
    return SimulationExperiment(**yaml.safe_load(EVENT_RECIPE.format(name=name)))


def _member_experiment(observation: str = "recorded_ts", function: str = "subsample"):
    return SimulationExperiment(**yaml.safe_load(MEMBER_RECIPE.format(observation=observation, function=function)))


def test_an_event_name_reaches_every_position_it_is_emitted_in():
    """One name, five emitted forms — they have to agree or the stimulus is dropped.

    The local is bound from ``external`` by attribute, and that attribute exists only
    because the ``external_input`` dict was keyed with the same string. A disagreement
    between them is not a syntax error but a ``hasattr`` that fails, which substitutes
    0.0 — a simulation that runs to completion having applied no stimulus at all.
    """
    code = _event_experiment("flash_burst").render_code(format="tvboptim")

    assert '"flash_burst": 1,' in code  # EXTERNAL_INPUTS
    assert 'hasattr(external, "flash_burst")' in code  # the binding
    assert "class flash_burstInput(" in code  # the input class
    assert '"flash_burst": flash_burstInput()' in code  # the wiring
    assert "+ flash_burst" in code  # the dfun's free symbol


@pytest.mark.parametrize("fmt", ["tvboptim", "jax"])
def test_observation_and_function_names_are_emitted_as_definitions(fmt: str):
    """Which is why they are held to the same rule as an event name."""
    code = _member_experiment().render_code(format=fmt)

    assert "subsample" in code and "recorded_ts" in code
    compile(code, "<generated>", "exec")


@pytest.mark.parametrize("name", list(REJECTED.values()), ids=list(REJECTED))
def test_an_event_name_that_is_not_a_name_is_refused_when_the_recipe_is_read(name: str):
    """Refused at load, before anything downstream gets a worse way to say it.

    Reading the recipe is the moment the author is still looking at what they wrote. Held
    any later, the same names arrive as somebody else's error: `lambda` and `flash.burst`
    reach SymPy first, which reports a `SyntaxError` against a mangled prefix of the
    author's own right-hand side, and a name with a space reaches black, which reports a
    parse error against a line of generated code.
    """
    with pytest.raises(ValueError, match="event name is not a name") as excinfo:
        _event_experiment(name)

    message = str(excinfo.value)
    assert repr(name) in message
    assert "Pulse stimulus" in message, "the label is shown so the event is recognisable"
    assert "label:" in message, "prose has somewhere to go"


@pytest.mark.parametrize("slot", ["observation", "function"])
def test_every_addressed_slot_is_held_to_the_same_rule(slot: str):
    """Not events alone — the rule is about being addressed, not about being an event."""
    with pytest.raises(ValueError, match=f"{slot} name is not a name"):
        _member_experiment(**{slot: "not a name"})


def test_a_target_variable_is_a_name_too():
    """It becomes the event's name, so checking only the key it is filed under misses it.

    `_resolve_events` replaces the name with `target_variable` when one is given, and
    that happens after construction — so a recipe keyed acceptably could still reach
    codegen carrying an unusable one.
    """
    raw = yaml.safe_load(EVENT_RECIPE.format(name="drive"))
    raw["events"]["drive"]["target_variable"] = "flash burst"

    with pytest.raises(ValueError, match="event name is not a name") as excinfo:
        SimulationExperiment(**raw)
    assert repr("flash burst") in str(excinfo.value)


def test_a_models_own_functions_are_held_to_it_as_well():
    """A `Dynamics` function is emitted as a `def` exactly as an experiment's is."""
    raw = yaml.safe_load(MEMBER_RECIPE.format(observation="recorded_ts", function="subsample"))
    raw["dynamics"]["functions"] = {"sig func": {"source_code": "1/(1+exp(-x))", "arguments": {"x": {}}}}

    with pytest.raises(ValueError, match="function name is not a name"):
        SimulationExperiment(**raw)


def test_the_loader_that_skips_init_checks_too():
    """`from_datamodel` builds through `cls.__new__`, so `__init__`'s checks never run.

    Every `from_file` / `from_string` load takes that path. Without its own call the
    suite stays green while the entry point users actually use accepts the name.
    """
    from tvbo.datamodel import schema as tvbo_schema

    dm = tvbo_schema.SimulationExperiment(id=1, label="loader probe")
    dm.observations = {"recorded ts": tvbo_schema.Observation(name="recorded ts", source=["V"])}

    with pytest.raises(ValueError, match="observation name is not a name"):
        SimulationExperiment.from_datamodel(dm)
