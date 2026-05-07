"""Tests for subplot-mosaic plotting helpers."""

import matplotlib.pyplot as plt
import numpy as np

from tvbo.classes.dynamics import Dynamics
from tvbo.classes.experiment import SimulationExperiment
from tvbo.data.types import ExperimentResult, SimulationResult, ExplorationResult
from tvbo.plot.experiment_layout import _auto_experiment_panels


LINEAR = """
name: LinearScalar
parameters:
  a:
    name: a
    value: -0.5
state_variables:
  x:
    name: x
    domain:
      lo: -2.0
      hi: 2.0
    equation:
      lhs: Derivative(x, t)
      rhs: a*x
    initial_value: 1.0
"""


def test_dynamics_plot_layout_supports_parameter_sweeps():
    dyn = Dynamics.from_string(LINEAR)

    fig = dyn.plot(
        layout="ab",
        panels={
            "a": {
                "kind": "timeseries",
                "dims": ("x",),
                "duration": 20,
                "dt": 0.5,
                "title": "Base dynamics",
            },
            "b": {
                "kind": "parameter_sweep_timeseries",
                "parameter": "a",
                "dims": ("x",),
                "values": [-1.0, 0.0, 1.0],
                "duration": 20,
                "dt": 0.5,
                "title": "Sweep",
            },
        },
    )

    assert len(fig.axes) == 2
    assert fig.axes[0].get_title() == "Base dynamics"
    assert fig.axes[1].get_title() == "Sweep"
    assert len(fig.axes[1].lines) == 3


def test_simulation_experiment_plot_layout_runs_integration_panels():
    dyn = Dynamics.from_string(LINEAR)
    exp = SimulationExperiment(dynamics=dyn)

    fig = exp.plot(
        layout="ab",
        run_kwargs={"format": "jax", "duration": 20, "dt": 0.5},
        panels={
            "a": {
                "kind": "integration",
                "modality": "timeseries",
                "title": "Integration",
            },
            "b": {
                "kind": "timeseries",
                "dims": ("x",),
                "duration": 20,
                "dt": 0.5,
                "title": "Dynamics",
            },
        },
    )

    assert len(fig.axes) == 2
    assert fig.axes[0].get_title() == "Integration"
    assert fig.axes[1].get_title() == "Dynamics"
    assert fig.axes[0].lines


def test_dynamics_plot_layout_can_render_into_existing_figure():
    dyn = Dynamics.from_string(LINEAR)
    fig = plt.figure(figsize=(8, 3))

    out = dyn.plot(
        layout="ab",
        fig=fig,
        panels={
            "a": {"kind": "timeseries", "dims": ("x",), "duration": 20, "dt": 0.5, "title": "Left"},
            "b": {
                "kind": "parameter_sweep_timeseries",
                "parameter": "a",
                "dims": ("x",),
                "values": [-1.0, 0.0, 1.0],
                "duration": 20,
                "dt": 0.5,
                "title": "Right",
            },
        },
    )

    assert out is fig
    assert len(fig.axes) == 2
    assert fig.axes[0].get_title() == "Left"
    assert fig.axes[1].get_title() == "Right"


def test_auto_experiment_panels_overlay_timeseries_exploration():
    sim = SimulationResult(
        data=np.array([[[[1.0]]], [[[0.7]]], [[[0.5]]], [[[0.3]]]]),
        state_names=["x"],
    )
    expl = ExplorationResult(
        name="sweep_a",
        axes=[{"name": "a", "n": 3, "explored_values": [-1.0, 0.0, 1.0]}],
        results=np.array(
            [
                [[1.0], [0.8], [0.6], [0.4]],
                [[1.0], [1.0], [1.0], [1.0]],
                [[1.0], [1.2], [1.4], [1.6]],
            ]
        ),
        dt=1.0,
        output_names=["x"],
    )
    result = ExperimentResult(integration=sim, explorations={"sweep_a": expl}, name="demo")

    panels = _auto_experiment_panels(result)

    assert set(panels) == {"a"}
    assert panels["a"]["kind"] == "integration"
    assert panels["a"]["overlay"][0]["kind"] == "exploration"


def test_auto_experiment_panels_put_continuation_before_timeseries_exploration():
    expl = ExplorationResult(
        name="sweep_a",
        axes=[{"name": "a", "n": 3, "explored_values": [-1.0, 0.0, 1.0]}],
        results=np.array(
            [
                [[1.0], [0.8], [0.6], [0.4]],
                [[1.0], [1.0], [1.0], [1.0]],
                [[1.0], [1.2], [1.4], [1.6]],
            ]
        ),
        dt=1.0,
        output_names=["x"],
    )
    result = ExperimentResult(continuations={"cont_a": object()}, explorations={"sweep_a": expl})

    panels = _auto_experiment_panels(result)

    assert list(panels) == ["a", "b"]
    assert panels["a"]["kind"] == "continuation"
    assert panels["a"]["plot"] == {"VOI": "x"}
    assert panels["b"]["kind"] == "exploration"


def test_simulation_experiment_runs_declared_timeseries_exploration():
    dyn = Dynamics.from_string(LINEAR)
    exp = SimulationExperiment(
        dynamics=dyn,
        explorations=[
            {
                "name": "sweep_a",
                "space": [{"parameter": "a", "explored_values": [-1.0, 0.0, 1.0]}],
                "observable": {"function": "x"},
            }
        ],
    )

    explorations = exp._run_python_explorations(duration=20, dt=0.5)

    assert set(explorations) == {"sweep_a"}
    assert explorations["sweep_a"].is_timeseries
    assert explorations["sweep_a"].output_names == ["x"]
    assert explorations["sweep_a"].results.shape[0] == 3


def test_simulation_experiment_plot_auto_into_existing_figure():
    dyn = Dynamics.from_string(LINEAR)
    exp = SimulationExperiment(dynamics=dyn)
    fig = plt.figure(figsize=(6, 3))

    out = exp.plot(fig=fig, run_kwargs={"format": "jax", "duration": 20, "dt": 0.5})

    assert out is fig
    assert fig.axes
