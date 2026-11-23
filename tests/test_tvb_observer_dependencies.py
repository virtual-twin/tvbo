"""An output the state alone determines is recorded on tvb from everything it is built from, as tvboptim records it.

TVB's observer computes a state-only output from the recorded state. It bound the state-only outputs alone, in a namespace of its own holding `numpy` and two of its names, so one that reads an unrecorded derived variable, calls one of the model's functions, or calls a function that renders to `scipy.special` raised `NameError`. tvboptim's realignment binds the derived variables it needs in dependency order, with the functions inlined. The model is ``x' = -a x`` beside ``y' = x``, so the recorded value can be checked against the trace's own states as well as against tvboptim's.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erfc

pytest.importorskip("tvb")

from tvbo import SimulationExperiment

pytestmark = [pytest.mark.backend_tvb, pytest.mark.backend_tvboptim]

B = 0.5
RTOL = 1e-10
"""Both runs integrate in float64; TVB's float32 delay buffer is not read, the model being uncoupled."""

_SPEC = """
id: 1
dynamics:
  name: Chain
  parameters:
    a: {value: 1.0}
    b: {value: %(b)s}
  state_variables:
    x: {equation: {rhs: "-a*x"}, initial_value: 1.0}
    y: {equation: {rhs: "x"}, initial_value: 0.0}
  derived_variables:
    u: {equation: {rhs: "2*x"}}
    s: {equation: {rhs: "u + y"}}
    r: {equation: {rhs: "squash(x)"}}
    w: {equation: {rhs: "tail(x)"}}
  functions:
    squash:
      arguments:
        z: {}
      equation: {rhs: "z / (b + z)"}
    tail:
      arguments:
        z: {}
      equation: {rhs: "erfc(z / b)"}
  output: [x, y, %(output)s]
network:
  number_of_nodes: 2
  nodes:
    - {id: 0, label: p}
    - {id: 1, label: q}
execution: {precision: float64}
integration: {method: Euler, step_size: 0.1, duration: 1.0}
"""


def _trace(output, backend):
    """The run of the chain recording ``x``, ``y`` and *output* on *backend*, as ``(time, variable, node)``."""
    trace = SimulationExperiment.from_string(_SPEC % {"b": B, "output": output}).run(backend).integration.data
    if "mode" in trace.dims:
        trace = trace.isel(mode=0)
    return trace.transpose("time", "variable", "node")


@pytest.mark.parametrize(
    "output, closed_form",
    [("s", lambda x, y: 2 * x + y), ("r", lambda x, y: x / (B + x)), ("w", lambda x, y: erfc(x / B))],
    ids=["unrecorded-derived", "model-function", "scipy-special-function"],
)
def test_a_state_only_output_is_recorded_from_what_it_reads(output, closed_form):
    """``s = u + y`` reads ``u = 2 x``, which nothing records, ``r = squash(x)`` calls a model function, and ``w = tail(x)`` one whose body renders to ``scipy.special.erfc``; on tvb each is its closed form at the recorded state, sample for sample, and equals tvboptim's."""
    pytest.importorskip("tvboptim")
    tvb = _trace(output, "tvb")
    x, y = tvb.sel(variable="x").values, tvb.sel(variable="y").values
    np.testing.assert_allclose(tvb.sel(variable=output).values, closed_form(x, y), rtol=RTOL)
    tvboptim = _trace(output, "tvboptim")
    np.testing.assert_array_equal(tvb["time"].values, tvboptim["time"].values)
    np.testing.assert_allclose(tvb.sel(variable=output).values, tvboptim.sel(variable=output).values, rtol=RTOL)
