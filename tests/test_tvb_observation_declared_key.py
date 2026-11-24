"""Every backend keys an observation by its declared name.

tvb runs each observation as a monitor class it generates and names after the observation with its first letter capitalised, and keyed the results by that class name (``m.title``, and the class name in `ExperimentResult.from_tvb`), so an observation declared ``observed`` came back as ``Observed`` on tvb and as ``observed`` on tvboptim and jax. The generated class now carries the declared name, which both tvb paths key by; a monitor tvbo did not generate keeps its class name.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from tvbo import ExperimentResult, SimulationExperiment

from .test_subsample_phase import PERIOD, _stamps_and_y
from .test_tvb_delayed_clock import _undelayed

OBSERVED = {"iri": "tvbo:observation_models/subsample", "period": PERIOD}
"""The curated subsample, which tvb, tvboptim and jax each compute, every three steps."""

BACKENDS = [
    pytest.param("tvb", marks=pytest.mark.backend_tvb),
    pytest.param("tvboptim", marks=pytest.mark.backend_tvboptim),
    pytest.param("jax", marks=pytest.mark.backend_jax),
]


def _spec(observations):
    """The undelayed probe with *observations*."""
    spec = copy.deepcopy(_undelayed())
    spec["observations"] = copy.deepcopy(observations)
    return spec


def _assert_subsampled(observation):
    """*observation* is the subsample: stamped at whole periods over the window."""
    stamps, _ = _stamps_and_y(observation)
    np.testing.assert_allclose(stamps, PERIOD * np.arange(1, len(stamps) + 1), rtol=0, atol=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_an_observation_is_read_by_its_declared_name(backend):
    """The run reports the observation declared ``observed`` under ``observed``."""
    pytest.importorskip(backend)
    observations = SimulationExperiment(**_spec({"observed": OBSERVED})).run(backend).integration.observations
    _assert_subsampled(observations["observed"])


@pytest.mark.backend_tvb
def test_from_tvb_keys_a_generated_monitor_by_its_declared_name():
    """`ExperimentResult.from_tvb` on the simulator tvbo builds keys the monitor after the first by the observation's declared name."""
    pytest.importorskip("tvb")
    sim = SimulationExperiment(**_spec({"raw": {"iri": "tvbo:observation_models/raw"}, "observed": OBSERVED})).execute("tvb")
    _assert_subsampled(ExperimentResult.from_tvb(sim, transient_time=0.0).integration.observations["observed"])
