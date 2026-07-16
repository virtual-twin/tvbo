"""End-to-end for `Experiment._resolve_from_experiment_params` — the consumer side of
the from_experiment parameter warm-start.

Covers, against a source result `.h5`:
  * a dynamics VECTOR sourced from a tuned estimate (``estimate__J_i``), reconciled by
    node label;
  * a coupling per-edge MATRIX sourced from a tuned estimate (``estimate__wLRE``),
    reconciled on BOTH node axes;
  * the legacy path — a dynamics vector sourced from a recorded OBSERVATION
    (``observation__control_mask``, unlabelled) — the Taher `g` case, taken in model
    order with no reconcile.
"""
from __future__ import annotations

import numpy as np
import pytest

from tvbo.classes.experiment import SimulationExperiment

# A 2-node consumer with initial_state.from_experiment and three measure-sourced params.
_CONSUMER = """
id: 100
initial_state: {method: from_experiment, source_experiment: 99}
dynamics:
  name: Kuramoto
  label: Kuramoto
  parameters:
    omega: {name: omega, value: 0.0628, unit: rad_per_ms}
    J_i:   {name: J_i, value: 0.0, free: true, heterogeneous: true, shape: "(n_nodes,)", measure: J_i}
    g:     {name: g,   value: 0.0, free: true, heterogeneous: true, shape: "(n_nodes,)", measure: control_mask}
  coupling_inputs:
    c: {name: c, description: "coupling"}
  state_variables:
    theta:
      name: theta
      unit: rad
      equation: {lhs: "Derivative(theta, t)", rhs: "omega + c"}
      variable_of_interest: true
      coupling_variable: true
  output: [theta]
  number_of_modes: 1
network:
  number_of_nodes: 2
  nodes:
    - {id: 0, label: r0}
    - {id: 1, label: r1}
  edges:
    - {source: 0, target: 1, weight: 0.5}
    - {source: 1, target: 0, weight: 0.5}
coupling:
  name: KuramotoCoupling
  label: KuramotoCoupling
  parameters:
    a: {name: a, value: 0.01}
    N: {name: N, value: 1.0}
    wLRE: {name: wLRE, value: 1.0, free: true, heterogeneous: true, shape: "(n_nodes, n_nodes)", measure: wLRE}
  pre_expression: {rhs: "sin(theta_j - theta_i)"}
  post_expression: {rhs: "a * gx / N"}
  incoming_states: [theta]
  local_states: [theta]
integration: {method: RungeKutta4thOrder, duration: 10.0, step_size: 1.0, transient_time: 0.0}
"""

_J_I = np.array([1.0, 2.0])
_WLRE = np.array([[0.1, 0.2], [0.3, 0.4]])
_MASK = np.array([1.0, 0.0])


def _write_source_h5(dirpath, labels=("r0", "r1")):
    """A source result exposing tuned estimates (labelled) + a recorded observation."""
    xr = pytest.importorskip("xarray")
    labels = list(labels)
    ds = xr.Dataset({
        "estimate__J_i": xr.DataArray(_J_I, dims=["node"], coords={"node": labels}),
        "estimate__wLRE": xr.DataArray(
            _WLRE, dims=["node_i", "node_j"],
            coords={"node_i": labels, "node_j": labels}),
        # legacy: unlabelled observation (Taher control_mask)
        "observation__control_mask": xr.DataArray(_MASK, dims=["control_mask_d0"]),
    })
    path = dirpath / "exp-99_desc-test_result.h5"
    ds.to_netcdf(path, engine="h5netcdf")
    return path


def _consumer(tmp_path):
    spec = tmp_path / "consumer.yaml"
    spec.write_text(_CONSUMER)
    return SimulationExperiment.from_file(str(spec))


def test_resolve_estimate_vector_matrix_and_legacy_observation(tmp_path):
    _write_source_h5(tmp_path)
    exp = _consumer(tmp_path)
    out = exp._resolve_from_experiment_params(results_root=str(tmp_path))
    assert out is not None
    assert set(out) == {"J_i", "g", "wLRE"}

    # vector estimate, reconciled (identity, same 2-node network)
    np.testing.assert_allclose(np.asarray(out["J_i"]).ravel(), _J_I)
    # matrix estimate — shape preserved (NOT ravelled), reconciled on both axes
    wl = np.asarray(out["wLRE"])
    assert wl.shape == (2, 2)
    np.testing.assert_allclose(wl, _WLRE)
    # legacy observation source (unlabelled) — model order, no reconcile
    np.testing.assert_allclose(np.asarray(out["g"]).ravel(), _MASK)


def test_reconcile_by_label_permuted_source(tmp_path):
    """Source records estimates in the OPPOSITE node order; reconcile must realign by
    label (never positional), so J_i/wLRE come back in the model's r0,r1 order."""
    _write_source_h5(tmp_path, labels=("r1", "r0"))   # swapped source order
    # the source arrays are authored in [r1, r0] order:
    #   estimate__J_i = [1,2] means r1->1, r0->2  => model order [r0,r1] -> [2,1]
    exp = _consumer(tmp_path)
    out = exp._resolve_from_experiment_params(results_root=str(tmp_path))
    np.testing.assert_allclose(np.asarray(out["J_i"]).ravel(), [2.0, 1.0])
    # wLRE authored in [r1,r0]x[r1,r0]; realigned to [r0,r1]x[r0,r1] => reverse both axes
    np.testing.assert_allclose(np.asarray(out["wLRE"]), _WLRE[::-1, :][:, ::-1])


def test_no_measure_params_returns_none(tmp_path):
    """An experiment with no measure-sourced params resolves to None (no seed)."""
    _write_source_h5(tmp_path)
    spec = tmp_path / "plain.yaml"
    spec.write_text(_CONSUMER.replace(", measure: J_i", "")
                             .replace(", measure: control_mask", "")
                             .replace(", measure: wLRE", ""))
    exp = SimulationExperiment.from_file(str(spec))
    assert exp._resolve_from_experiment_params(results_root=str(tmp_path)) is None
