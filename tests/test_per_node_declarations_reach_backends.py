"""What a declared node sets for itself — its label, its initial state, a dynamics parameter — reaches every array backend.

The model is an uncoupled decay ``x' = -a x`` integrated by Euler, whose trajectory is exactly ``x0 (1 - a dt)^n`` per node, so the reference is closed form rather than another backend. Node ``p`` sets its own ``x0`` and ``a``, ``r`` its own ``x0``, and ``q`` nothing: a backend that ignores a node's state or parameter is off by at least 0.5 at the last step.
"""

from __future__ import annotations

import numpy as np
import pytest

from tvbo.classes.experiment import SimulationExperiment

_SPEC = """
id: 1
dynamics:
  name: Decay
  parameters:
    a: {value: 1.0}
  state_variables:
    x: {equation: {rhs: "-a*x"}, initial_value: 1.0}
network:
  number_of_nodes: 3
  nodes:
    - {id: 0, label: p, state: {x: {value: 2.0}}, parameters: [{name: a, value: 0.5}]}
    - {id: 1, label: q}
    - {id: 2, label: r, state: {x: {value: -1.0}}}
integration: {method: Euler, step_size: 0.1, duration: 1.0}
"""

X0 = np.array([2.0, 1.0, -1.0])
RATE = np.array([0.5, 1.0, 1.0])


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("tvb", marks=pytest.mark.backend_tvb),
        pytest.param("jax", marks=pytest.mark.backend_jax),
        pytest.param("tvboptim", marks=pytest.mark.backend_tvboptim),
        pytest.param("python", marks=pytest.mark.backend_core),
    ],
)
def test_a_node_s_own_declarations_reach_the_backend(tmp_path, backend):
    pytest.importorskip({"tvb": "tvb", "jax": "jax", "tvboptim": "tvboptim", "python": "numpy"}[backend])
    spec = tmp_path / "decay.yaml"
    spec.write_text(_SPEC)

    trace = SimulationExperiment.from_file(str(spec)).run(backend).integration
    if "mode" in trace.dims:
        trace = trace.isel(mode=0)
    x = trace.sel(variable="x").transpose("time", "node")

    steps = np.round(x["time"].values / 0.1)
    assert list(map(str, x["node"].values)) == ["p", "q", "r"]
    np.testing.assert_allclose(x.values, X0 * (1 - RATE * 0.1) ** steps[:, None], rtol=0, atol=1e-12)
