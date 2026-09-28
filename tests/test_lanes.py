"""An event parameter sourced as a (lane, node) array runs the experiment once per lane.

Three leaky nodes on a chain r0 - r1 - r2 with undirected weights 1 and 1/4, starting at rest without noise, and a stimulus whose per-node gain comes from a curated container with one row per programme. The network is linear, so the answer is known without a reference run: the undriven baseline stays at rest, a lane with twice the gains responds twice as strongly, the symmetric wiring makes the response reciprocal (r0 driven as seen at r2 equals r2 driven as seen at r0), and a lane run inside a batch matches the same lane run alone. The container is read through the parameter's `used:` reference, reconciled by label, and the lane coordinates it carries come back on the result.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("tvboptim")

from tvbo.classes.experiment import SimulationExperiment
from tvbo.run import lanes

_SPEC = """
id: 1
functions:
  node_mean:
    source_code: jnp.mean(data[:, 0, :], axis=0)
    arguments: {data: {}}
dynamics:
  name: Leak
  parameters:
    k: {value: 0.1}
  coupling_inputs:
    c: {}
  state_variables:
    x:
      equation: {rhs: -k * x + c}
      initial_value: 0.0
      coupling_variable: true
network:
  number_of_nodes: 3
  nodes:
    - {id: 0, label: r0}
    - {id: 1, label: r1}
    - {id: 2, label: r2}
  edges:
    - {source: 0, target: 1, weight: 1.0}
    - {source: 2, target: 1, weight: 0.25}
  coupling:
    LinCoupling:
      parameters:
        G: {value: 0.05}
      pre_expression: {rhs: x_j}
      post_expression: {rhs: G * gx}
      incoming_states: [x]
      local_states: [x]
integration: {method: Euler, duration: 50.0, step_size: 0.1, transient_time: 0.0}
events:
  drive:
    event_type: stimulus
    equation: {rhs: amplitude}
    parameters:
      amplitude:
        shape: "(n_nodes,)"
        USED
    target_variable: x
observations:
  level:
    source: [x]
    dims: [node]
    pipeline:
      - function: node_mean
        arguments: {data: {value: integration.result}}
"""

GAINS = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [2.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
"""Four programmes: r0 alone, r2 alone, r0 twice as hard, and every node at half gain."""


def _container(path, gains=GAINS, labels=("r2", "r0", "r1")):
    """The gains as a curated `programme × node` container, its nodes stored in an order other than the model's so only a by-label read aligns them."""
    order = [("r0", "r1", "r2").index(label) for label in labels]
    ds = xr.Dataset(
        {"amplitude": (("programme", "node"), gains[:, order])},
        coords={
            "node": list(labels),
            "subject": ("programme", ["p1", "p1", "p2", "p3"]),
            "session": ("programme", ["s1", "s2", "s1", "s1"]),
        },
    )
    ds.to_netcdf(path, engine="h5netcdf")
    return path


def _experiment(tmp_path, used: str = "{}"):
    spec = tmp_path / "lanes.yaml"
    spec.write_text(_SPEC.replace("USED", f"used: {used}" if used != "{}" else "value: 0.0"))
    return SimulationExperiment.from_file(str(spec))


@pytest.fixture(scope="module")
def swept(tmp_path_factory):
    root = tmp_path_factory.mktemp("lanes")
    path = _container(root / "drive.h5")
    experiment = _experiment(root, used=f"{{iri: {path}, output: amplitude, reconcile: by_label}}")
    return experiment, lanes.run_lanes(experiment, batch_size=3)


def test_the_referenced_parameter_is_the_lane_parameter(swept):
    experiment, _ = swept
    (param,) = lanes.lane_parameters(experiment)
    assert param.values.dims == ("programme", "node")
    np.testing.assert_array_equal(param.values.values, GAINS)
    assert list(param.values.coords["node"].values) == ["r0", "r1", "r2"]


def test_one_row_per_lane_with_its_coordinates(swept):
    _, ds = swept
    assert ds["level"].dims == ("programme", "node")
    assert ds["level"].shape == (4, 3)
    assert list(ds.coords["subject"].values) == ["p1", "p1", "p2", "p3"]
    assert list(ds.coords["session"].values) == ["s1", "s2", "s1", "s1"]
    assert list(ds.coords["node"].values) == ["r0", "r1", "r2"]
    np.testing.assert_array_equal(ds["axis_points__" + ds.attrs["lane_parameters"][0]].values, GAINS)


def test_the_undriven_baseline_stays_at_rest(swept):
    _, ds = swept
    assert ds["level_baseline"].dims == ("node",)
    np.testing.assert_array_equal(ds["level_baseline"].values, 0.0)


def test_the_response_follows_the_gains_and_the_wiring(swept):
    _, ds = swept
    level = ds["level"].values
    np.testing.assert_allclose(level[2], 2.0 * level[0], rtol=1e-5)
    np.testing.assert_allclose(level[0, 2], level[1, 0], rtol=1e-5)
    assert level[0, 2] > 0.0
    np.testing.assert_allclose(level[0, 1] / level[1, 1], 4.0, rtol=1e-5)


def test_a_lane_in_a_batch_matches_the_lane_run_alone(tmp_path, swept):
    experiment, ds = swept
    one = xr.Dataset(
        {ds.attrs["lane_parameters"][0]: (("programme", "node"), GAINS[3:4])}, coords={"node": ["r0", "r1", "r2"]}
    )
    alone = lanes.run_lanes(_experiment(tmp_path), one, baseline=False)
    np.testing.assert_allclose(alone["level"].values[0], ds["level"].values[3], rtol=1e-6)
    assert "level_baseline" not in alone


def test_lanes_must_name_event_parameters(tmp_path):
    bogus = xr.Dataset({"drive.gain": (("programme", "node"), GAINS)}, coords={"node": ["r0", "r1", "r2"]})
    with pytest.raises(ValueError, match="not event parameters"):
        lanes.lane_parameters(_experiment(tmp_path), bogus)


def test_a_node_axis_out_of_model_order_is_refused(tmp_path):
    shuffled = xr.Dataset({"x.amplitude": (("programme", "node"), GAINS)}, coords={"node": ["r2", "r0", "r1"]})
    with pytest.raises(ValueError, match="model order"):
        lanes.lane_parameters(_experiment(tmp_path), shuffled)
