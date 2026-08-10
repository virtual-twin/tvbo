"""``stimulation:`` is the singleton spelling of a stimulus ``Event``.

An experiment may hold several events and only one ``stimulation``, and ``events`` is what
every backend except TVB reads — so the singleton is deprecated and mirrored into
``events`` at load. These tests pin the mirroring, and pin the two things about it that
would be silent if they broke: that a legacy recipe still reaches the event-reading
backends, and that TVB's own stimulus support is not removed on the way.
"""

import warnings

import pytest

from tvbo.classes.experiment import SimulationExperiment

BASE = """
id: 1
label: stim_probe
dynamics: {iri: "tvbo:Generic2dOscillator"}
"""

STIMULATION = """
stimulation:
  label: flash
  regions: [0]
  duration: 0.5
  parameters:
    onset: {value: 1.0}
  equation:
    rhs: "0.5"
"""


def _load(source: str) -> SimulationExperiment:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return SimulationExperiment.from_string(source)


def test_a_legacy_stimulation_reaches_the_event_reading_backends():
    """The whole point: tvboptim and NeuroML read ``events`` and never learned ``stimulation``."""
    exp = _load(BASE + STIMULATION)

    event = (exp.events or {}).get("flash")
    assert event is not None, sorted((exp.events or {}))
    assert str(event.event_type) == "stimulus"
    assert list(event.nodes) == [0]
    assert float(event.duration) == 0.5
    assert list(event.parameters) == ["onset"]


def test_the_singleton_survives_so_tvb_keeps_its_stimulus():
    """Mirrored, not moved.

    TVB renders a stimulus from ``stimulation`` alone. Clearing the slot once the event
    exists would leave every TVB recipe silently stimulus-free — a backend losing a
    feature is not a deprecation.
    """
    exp = _load(BASE + STIMULATION)

    assert exp.stimulation is not None
    assert exp.stimulation.label == "flash"


def test_declaring_stimulation_warns():
    with pytest.warns(DeprecationWarning, match="`stimulation:` is deprecated"):
        SimulationExperiment.from_string(BASE + STIMULATION)


def test_an_existing_event_of_the_same_name_wins():
    """The authored event is the one the author meant; the mirror must not overwrite it."""
    source = BASE + STIMULATION + """
events:
  flash:
    event_type: stimulus
    duration: 99.0
    equation: {rhs: "1.0"}
"""
    exp = _load(source)

    assert float(exp.events["flash"].duration) == 99.0


def test_stimulation_noise_warns_rather_than_vanishing():
    """``noise`` is the one slot with no event counterpart, and no backend renders it.

    A stochastic event is a parameter whose ``distribution`` declares ``axis: time``, so
    there is nothing to carry across — but an authored key disappearing without a word
    is how a recipe comes to mean something other than what it says.
    """
    source = BASE + """
stimulation:
  label: noisy
  equation: {rhs: "0.5"}
  noise:
    noise_type: gaussian
"""
    with pytest.warns(DeprecationWarning, match="stimulation.noise"):
        SimulationExperiment.from_string(source)
