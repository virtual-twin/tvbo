"""A coupling parameter sourced from another run reaches its coupling when the coupling input carries a different name.

tvboptim keys a state's couplings by coupling-input name, while a recipe keys its couplings by function name, so the seed routing must translate one into the other. The model is a pure drift fed through input ``c`` by coupling ``LinCoupling``: the sourced gain is zero, so the measured slope is the sourced drift rate only when the gain lands on the coupling the input reads.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tvboptim")
xr = pytest.importorskip("xarray")

from tvbo.classes.experiment import SimulationExperiment

RATE = np.array([0.5, 2.0])

_CONSUMER = """
id: 100
dynamics:
  name: Drift
  parameters:
    rate: {value: 7.0, shape: "(n_nodes,)", used: {experiment: 99, output: estimate__rate}}
  coupling_inputs:
    c: {source: LinCoupling}
  state_variables:
    x:
      equation: {rhs: rate + c}
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
integration: {method: Euler, duration: 10.0, step_size: 1.0, transient_time: 5.0}
observations:
  x_trace: {source: [x], aggregation: none, record: true}
"""


@pytest.fixture
def consumer(tmp_path):
    """The consumer recipe beside the result of the run it sources ``rate`` and ``G`` from."""
    xr.Dataset(
        {
            "estimate__rate": xr.DataArray(RATE, dims=["node"], coords={"node": ["r0", "r1"]}),
            "estimate__G": xr.DataArray(np.array(0.0)),
        }
    ).to_netcdf(tmp_path / "exp-99_desc-source_result.h5", engine="h5netcdf")
    spec = tmp_path / "consumer.yaml"
    spec.write_text(_CONSUMER)
    return SimulationExperiment.from_file(str(spec))


def test_the_seed_routing_names_the_coupling_input(consumer):
    code = consumer.render_code(format="tvboptim")
    assert '_SEED_PARAM_COUPLING = {\n    "G": "c",\n}' in code


def test_the_sourced_gain_reaches_the_coupling_the_input_reads(consumer, tmp_path):
    result = consumer.run(format="tvboptim", results_root=str(tmp_path))

    trace = np.asarray(getattr(result.observations.x_trace, "data", result.observations.x_trace))
    slope = np.diff(trace.reshape(trace.shape[0], -1), axis=0)
    np.testing.assert_allclose(slope, np.broadcast_to(RATE, slope.shape), rtol=1e-6)
