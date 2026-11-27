"""A tuning loop hands its monitors one short window at a time, so a monitor that reads history from before its window, a kernel's past or an observer's states, has to carry it from one window to the next.

Which monitors hold such memory is decided from the spec (`observation_has_memory`), so the loop carries it for every one of them unconditionally and refuses a monitor that cannot, instead of skipping the carry whenever the monitor happens to lack the method. That silent skip left a streaming BOLD convolving each in-loop sample against an empty ring.
"""

import re

import pytest

from tvbo import Observation
from tvbo.classes.experiment import SimulationExperiment
from tvbo.templates.tvboptim.utils import observation_has_memory

pytestmark = pytest.mark.backend_tvboptim

_FIC_EIB = "tvbo/database/experiments/EI_Tuning_FIC_EIB_Optimization.yaml"


@pytest.mark.parametrize(
    ("spec", "declared", "memory"),
    [
        ("BOLD_HRF_strided", {"reduce": "streaming"}, True),
        ("BOLD_HRF_strided", {}, True),
        ("BalloonWindkesselBold", {}, True),
        ("HRFBold_FirstOrderVolterra", {}, True),
        ("SubSample", {}, False),
        ("TemporalAverage_tvboptim", {}, False),
        ("GlobalAverage", {}, False),
    ],
)
def test_memory_is_read_from_the_spec(spec, declared, memory):
    observation = Observation.from_db(spec)
    observation.source = ["S"]
    for name, value in declared.items():
        setattr(observation, name, value)
    experiment = SimulationExperiment(observations=[observation])
    assert observation_has_memory(experiment.observations[observation.name], experiment) is memory


def test_a_tuning_loop_carries_its_bold_every_window_and_refuses_a_monitor_that_cannot():
    experiment = SimulationExperiment.from_file(_FIC_EIB)
    assert observation_has_memory(experiment.observations["bold"], experiment)
    code = experiment.render_code("tvboptim")
    assert "_bold_monitor = _bold_monitor.carry_warmup(result)" in code
    assert "_bold_monitor = _bold_monitor.open_warmup(n_nodes)" in code
    guards = [
        line.strip() for line in code.splitlines() if re.search(r"hasattr\(_bold_monitor, [\"']carry_warmup[\"']\)", line)
    ]
    assert guards and all(line.startswith("if not hasattr(") for line in guards), guards
