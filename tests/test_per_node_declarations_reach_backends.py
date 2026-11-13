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


BACKENDS = [
    pytest.param("tvb", marks=pytest.mark.backend_tvb),
    pytest.param("jax", marks=pytest.mark.backend_jax),
    pytest.param("tvboptim", marks=pytest.mark.backend_tvboptim),
    pytest.param("python", marks=pytest.mark.backend_core),
]


def _decay_trace(tmp_path, backend, spec_text=_SPEC, arrays=None):
    """The decay experiment's ``x`` on *backend* as ``(time, node)``, its network holding *arrays* in memory."""
    pytest.importorskip({"tvb": "tvb", "jax": "jax", "tvboptim": "tvboptim", "python": "numpy"}[backend])
    spec = tmp_path / "decay.yaml"
    spec.write_text(spec_text)
    experiment = SimulationExperiment.from_file(str(spec))
    for path, value in (arrays or {}).items():
        experiment.network.set_array(path, value)
    trace = experiment.run(backend).integration
    if "mode" in trace.dims:
        trace = trace.isel(mode=0)
    return trace.sel(variable="x").transpose("time", "node")


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_node_s_own_declarations_reach_the_backend(tmp_path, backend):
    x = _decay_trace(tmp_path, backend)
    steps = np.round(x["time"].values / 0.1)
    assert list(map(str, x["node"].values)) == ["p", "q", "r"]
    np.testing.assert_allclose(x.values, X0 * (1 - RATE * 0.1) ** steps[:, None], rtol=0, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_categorical_node_attribute_leaves_the_run_unchanged(tmp_path, backend):
    """A categorical attribute declared on the nodes, its codes and its vocabulary of names held in memory and listed beside it, is data about the nodes and no parameter of the model: every backend integrates the same closed form as without it."""
    categorical = _SPEC.replace(
        "  number_of_nodes: 3\n",
        "  number_of_nodes: 3\n"
        "  parameters:\n"
        "    groups: {value: [cortex, thalamus]}\n"
        "    group_names: {measure: nodes/group_names}\n"
        "  node_template:\n"
        "    parameters:\n"
        "      group: {measure: nodes/group, levels: groups}\n",
    )
    arrays = {"nodes/group": np.array([0, 1, 0], dtype="int32"), "nodes/group_names": np.array([b"cortex", b"thalamus"])}
    x = _decay_trace(tmp_path, backend, categorical, arrays)
    steps = np.round(x["time"].values / 0.1)
    np.testing.assert_allclose(x.values, X0 * (1 - RATE * 0.1) ** steps[:, None], rtol=0, atol=1e-12)
