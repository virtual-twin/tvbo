"""A streamed trailing window folds the declared span of simulated time, whatever recording period the observation also declares.

A streamed reducer folds every integration step, so a `tail_duration` is a number of steps to it. Converted against a declared `period` instead, the window shrinks by the ratio of the period to the step and the observation averages a span other than the one it names. Checked against the closed-form Euler trajectory of a one-variable relaxation.
"""

import glob

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment  # noqa: E402

_SPEC = """
id: 1
dynamics:
  name: Relax
  label: Relax
  parameters:
    tau: {value: 5.0}
  state_variables:
    x:
      initial_value: 0.0
      equation: {rhs: "(1 - x) / tau"}
  output: [x]
network:
  number_of_nodes: 1
  edges: []
integration: {method: euler, step_size: 1.0, duration: 50.0, unit: ms}
observations:
  x_tail: {source: [x], aggregation: mean, tail_duration: 10.0, period: 5.0, reduce: streaming}
execution: {random_seed: 0, precision: float64}
"""


def _euler_x(n):
    """``x`` after ``n`` Euler steps of ``dx/dt = (1 - x) / 5`` from 0 at ``dt = 1``."""
    return 1.0 - (1.0 - 1.0 / 5.0) ** np.asarray(n, dtype=float)


def test_a_streamed_tail_folds_the_declared_span_not_that_many_periods(tmp_path):
    SimulationExperiment.from_string(_SPEC).run("tvboptim").save(tmp_path / "run")
    (path,) = glob.glob(str(tmp_path / "run" / "**" / "*result.h5"), recursive=True)
    with xr.open_dataset(path, engine="h5netcdf") as ds:
        value = float(np.asarray(ds["observation__x_tail"]).ravel()[0])
    np.testing.assert_allclose(value, _euler_x(np.arange(41, 51)).mean(), rtol=1e-12)
