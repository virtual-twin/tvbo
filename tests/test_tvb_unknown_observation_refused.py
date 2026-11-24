"""tvb refuses, by name, an observation it cannot compute in full, where it recorded another quantity under the observation's name.

tvb's observation template files an observation under the monitor its class reference names, else under the first monitor whose key step its pipeline holds and that computes every step it declares, and else, for an observation that declares nothing to compute, under a source or period fallback or ``raw``, a passthrough of every state variable. An observation that declares a pipeline, an analysis, an imaging modality, dynamics or a class reference that no monitor computes in full is refused, as is a pipeline a monitor computes only in part. An observation that declares only its source states is recorded raw, as is the curated ``raw``, and a declared subsample runs as tvb's subsample monitor.
"""

from __future__ import annotations

import copy
import re

import pytest

from tvbo import SimulationExperiment

from .test_subsample_phase import INLINE, PERIOD
from .test_tvb_delayed_clock import _undelayed

ENVELOPE = {"name": "envelope", "equation": {"rhs": "abs(X)"}}
"""A pipeline step no tvb monitor computes."""

REFUSED = {
    "envelope": ({"source": ["y"], "pipeline": [ENVELOPE]}, "the pipeline envelope"),
    "periodic": ({"source": ["y"], "period": PERIOD, "pipeline": [ENVELOPE]}, "the pipeline envelope"),
    "partial": (
        {**INLINE, "pipeline": [*INLINE["pipeline"], ENVELOPE]},
        "the pipeline subsample, envelope; the tvb subsample monitor does not compute envelope",
    ),
    "bold": ({"source": ["y"], "imaging_modality": "BOLD"}, "the imaging modality"),
}
"""Per observation name, a declaration no tvb monitor computes in full and the start of the phrase the refusal names it by: a pipeline step none recognises, the same with a period that would otherwise file it as a subsample, a subsample followed by a step the subsample monitor does not compute, and a BOLD modality with no pipeline or class reference."""

RAW = "Raw passthrough — records all variables at every step."
KEPT = {
    "bare": ({"source": ["y"]}, RAW),
    "curated": ({"iri": "tvbo:observation_models/raw"}, RAW),
    "covered": (INLINE, "Temporal decimation without averaging."),
}
"""Declarations tvb computes and the first docstring line of the monitor it renders for each: the source alone and the curated ``raw`` (class reference ``Raw``) as the raw states, and a declared subsample as tvb's subsample monitor."""

MONITORS = re.compile(r'^class \w+\((?:\w+, )*(?:Monitor|Projection)\):\n    """([^\n]*)"""', re.M)
"""The first docstring line of each monitor class a tvb script defines, whatever mixins its class takes."""


def _render(name, observation):
    """The tvb code of the undelayed probe observed by *observation* under *name*."""
    pytest.importorskip("tvb")
    spec = copy.deepcopy(_undelayed())
    spec["observations"] = {name: copy.deepcopy(observation)}
    return SimulationExperiment(**spec).render_code("tvb")


@pytest.mark.backend_tvb
@pytest.mark.parametrize("name", list(REFUSED))
def test_tvb_refuses_an_observation_no_monitor_computes(name):
    """The render names the observation and what it declares, instead of recording another quantity under its name."""
    observation, phrase = REFUSED[name]
    with pytest.raises(
        NotImplementedError, match=rf"the tvb backend has no monitor for the observation '{name}', which declares {phrase}"
    ):
        _render(name, observation)


@pytest.mark.backend_tvb
@pytest.mark.parametrize("name", list(KEPT))
def test_tvb_renders_the_monitor_an_observation_declares(name):
    """An observation tvb computes in full renders as one monitor, the one that computes it."""
    observation, docstring = KEPT[name]
    assert MONITORS.findall(_render(name, observation)) == [docstring]
