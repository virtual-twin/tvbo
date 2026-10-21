"""A parameter sourced from another run holds in the measured window, not only in the settle before it.

A settle opens the scan and hands its endpoint to the measured window, so the window does not rebuild its initial condition. Its parameters are another matter: a value read from another run's result with `used:` is the model's own, and a window that skipped it would integrate the placeholder the recipe left beside it, with nothing raised. The model here is a pure drift, so the measured trajectory's slope is the drift rate the run actually integrated, and a coupling gain that should be zero is visible as any departure from it. Each case runs with and without a settle, and with the coupling input named as the coupling and named otherwise, since a sourced coupling parameter is routed to the coupling by that name.
"""

from __future__ import annotations

import pytest
import xarray as xr

pytest.importorskip("tvboptim")

from tvbo.classes.experiment import SimulationExperiment

RATE = xr.DataArray([0.5, 2.0], dims="node", coords={"node": ["r0", "r1"]})
"""The drift rate of each node, as the source run saved it."""

_CONSUMER = """
id: 100
dynamics:
  name: Drift
  parameters:
    rate: {value: 7.0, shape: "(n_nodes,)", used: {experiment: 99, output: estimate__rate}}
  coupling_inputs:
    INPUT: {}
  state_variables:
    x:
      equation: {rhs: rate + INPUT}
      initial_value: 0.0
      coupling_variable: true
network:
  number_of_nodes: 2
  nodes:
    - {id: 0, label: r0}
    - {id: 1, label: r1}
  edges:
    - {source: 0, target: 1, weight: 1.0}
    - {source: 1, target: 0, weight: 1.0}
  coupling:
    LinCoupling:
      parameters:
        G: {value: 1.0, used: {experiment: 99, output: estimate__G}}
      pre_expression: {rhs: x_j}
      post_expression: {rhs: G * gx}
      incoming_states: [x]
      local_states: [x]
integration: {method: Euler, duration: 10.0, step_size: 1.0, transient_time: TRANSIENT}
observations:
  x_trace: {source: [x], aggregation: none, record: true}
"""


def _write_source(dirpath):
    xr.Dataset({"estimate__rate": RATE, "estimate__G": xr.DataArray(0.0)}).to_netcdf(
        dirpath / "exp-99_desc-source_result.h5", engine="h5netcdf"
    )


@pytest.mark.parametrize("transient", [0.0, 5.0])
@pytest.mark.parametrize("coupling_input", ["LinCoupling", "c"])
def test_the_measured_window_integrates_the_sourced_values(tmp_path, transient, coupling_input):
    _write_source(tmp_path)
    spec = tmp_path / "consumer.yaml"
    spec.write_text(_CONSUMER.replace("INPUT", coupling_input).replace("TRANSIENT", str(transient)))
    result = SimulationExperiment.from_file(str(spec)).run(format="tvboptim", results_root=str(tmp_path))

    slope = result.integration.sel(variable="x", drop=True).diff("time")
    xr.testing.assert_allclose(slope, RATE.broadcast_like(slope), rtol=1e-6)
