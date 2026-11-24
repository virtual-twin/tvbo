"""Observations assigned to a built experiment, or carried by a shallow copy of one, are the observations it was built with.

LinkML's setter wrapped an assigned mapping in a ``JsonObj``, which has no ``.items`` and iterates its keys: tvb rendered no monitor for the observation and its run returned none, and tvboptim raised ``AttributeError``. `copy.copy` set each attribute of its clone through that setter, so a shallow copy held the same ``JsonObj``. Each case reaches the undelayed probe's observation one of those ways and runs it on tvb and on tvboptim against the same probe built with it; the renamed copy of each names its observation otherwise, so nothing keys on a name. Whatever reaches tvb's render in a shape no keyed collection takes is refused there, instead of rendering no monitor.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from tvbo import SimulationExperiment, templates
from tvbo.datamodel import Observation

from .test_subsample_phase import _stamps_and_y
from .test_tvb_observation_declared_key import OBSERVED, _assert_subsampled, _spec

CASES = {
    "record": ("observed", "record"),
    "record_renamed": ("sampled", "record"),
    "declaration": ("observed", "declaration"),
    "declaration_renamed": ("every_third", "declaration"),
    "shallow_copy": ("observed", "shallow_copy"),
    "shallow_copy_renamed": ("decimated", "shallow_copy"),
}
"""Per case, the observation's name and how it reaches the experiment: assigned as an ``Observation`` record, assigned as its declaration, or carried by `copy.copy` of the probe built with it. The observation is the curated subsample every three steps, which tvb and tvboptim each compute."""

BACKENDS = [
    pytest.param("tvb", marks=pytest.mark.backend_tvb),
    pytest.param("tvboptim", marks=pytest.mark.backend_tvboptim),
]

TVB_TEMPLATES = ["tvbo-tvb-observation.py.mako", "tvbo-tvb-monitor.py.mako"]
"""The two tvb templates that turn an experiment's observations into monitors."""


def _built(name):
    """The probe built with the subsample under *name*."""
    return SimulationExperiment(**_spec({name: OBSERVED}))


def _experiment(case):
    """The probe holding *case*'s observation the way *case* gives it."""
    name, how = CASES[case]
    if how == "shallow_copy":
        return copy.copy(_built(name))
    experiment = SimulationExperiment(**_spec({}))
    declaration = copy.deepcopy(OBSERVED)
    experiment.observations = {name: Observation(name=name, **declaration) if how == "record" else declaration}
    return experiment


@pytest.mark.parametrize("case", list(CASES))
def test_the_collection_is_held_as_a_constructed_one(case):
    """The slot holds a plain dict of ``Observation`` records, filled from the curated model as construction fills them."""
    name, _ = CASES[case]
    held = _experiment(case).observations
    built = _built(name).observations
    assert type(held) is dict
    assert list(held) == [name]
    assert isinstance(held[name], Observation)
    assert held[name].class_reference == built[name].class_reference
    assert held[name].period == built[name].period == OBSERVED["period"]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", list(CASES))
def test_the_observation_is_run_as_a_constructed_one(case, backend):
    """The run reports the observation under its name, sample for sample the observation of the probe built with it."""
    pytest.importorskip(backend)
    name, _ = CASES[case]
    held = _experiment(case).run(backend).integration.observations[name]
    built = _built(name).run(backend).integration.observations[name]
    _assert_subsampled(held)
    for got, expected in zip(_stamps_and_y(held), _stamps_and_y(built), strict=True):
        np.testing.assert_array_equal(got, expected)


@pytest.mark.backend_tvb
@pytest.mark.parametrize("template", TVB_TEMPLATES)
def test_the_tvb_render_refuses_a_collection_of_no_keyed_shape(template):
    """A set of observation names is no keyed collection, and the render raises on it rather than emitting no monitor."""
    experiment = SimpleNamespace(observations=frozenset({"observed"}))
    with pytest.raises(TypeError, match=r"^cannot read observations names from frozenset$"):
        templates.lookup.get_template(template).render(experiment=experiment)
