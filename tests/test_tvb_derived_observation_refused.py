"""tvb refuses, by name, an observation derived from other observations or aggregated over time, where it left the observation out of its result.

Both are computed after the run, from other observations or from the whole trace, and a tvb monitor computes per step. tvb dropped a derived observation before classifying the rest and dropped an aggregation it filed as the raw states; an aggregation on a periodic monitor was recorded as that monitor and the aggregation ignored. One refusal names every such observation, in sorted order, and what each declares. An observation whose source names itself is not derived, and ``aggregation: none`` declares no aggregation: both are recorded.
"""

from __future__ import annotations

import copy

import pytest

from tvbo import SimulationExperiment

from .test_delay_history_from_opening import DT
from .test_tvb_delayed_clock import _undelayed

REFUSED = {
    "derived": (
        {"y_trace": {"source": ["y"]}, "y_again": {"source": ["y_trace"]}},
        "the observation 'y_again', which declares the source observation y_trace",
    ),
    "aggregated": (
        {"y_mean": {"source": ["y"], "aggregation": "mean"}},
        "the observation 'y_mean', which declares the aggregation mean",
    ),
    "periodic": (
        {"y_last": {"source": ["y"], "period": 3 * DT, "aggregation": "last"}},
        "the observation 'y_last', which declares the aggregation last",
    ),
    "several": (
        {"z_mean": {"source": ["y"], "aggregation": "mean"}, "a_trace": {"source": ["z_mean"]}},
        "the observation 'a_trace', which declares the source observation z_mean; for the observation 'z_mean', which declares the aggregation mean",
    ),
}
"""Per case, the observations declared and the observations the refusal names with what each declares: one derived from a raw observation, which is kept, a mean of the raw states, the last sample of a periodic monitor, and two declared out of sorted order."""

RAW = "Raw passthrough — records all variables at every step."
KEPT = {
    "self_reference": ({"y": {"name": "y", "source": ["y"]}}, "Y"),
    "aggregation_none": ({"y_kept": {"source": ["y"], "aggregation": "none"}}, "Y_kept"),
}
"""Declarations tvb records as the raw states, with the monitor class it names after each: an observation named after the state it observes, and one that declares no aggregation."""


def _render(observations):
    """The tvb code of the undelayed probe observed by *observations*."""
    pytest.importorskip("tvb")
    spec = copy.deepcopy(_undelayed())
    spec["observations"] = copy.deepcopy(observations)
    return SimulationExperiment(**spec).render_code("tvb")


@pytest.mark.backend_tvb
@pytest.mark.parametrize("case", list(REFUSED))
def test_tvb_refuses_an_observation_computed_after_the_run(case):
    """The render names every derived or aggregated observation and what it declares, instead of leaving it out of the result."""
    observations, named = REFUSED[case]
    with pytest.raises(NotImplementedError, match=rf"^the tvb backend has no monitor for {named}\. "):
        _render(observations)


@pytest.mark.backend_tvb
@pytest.mark.parametrize("case", list(KEPT))
def test_tvb_records_an_observation_that_declares_no_derivation(case):
    """An observation that names itself as its source, or declares no aggregation, renders as the raw states."""
    observations, class_name = KEPT[case]
    assert f'class {class_name}(Monitor):\n    """{RAW}"""' in _render(observations)
