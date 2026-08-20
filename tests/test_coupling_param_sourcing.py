"""A coupling parameter carrying ``source:`` + ``measure:`` is read from storage, not defaulted.

`Parameter.source` promises that "a sourced value is never inlined into generated code: it is
resolved lazily and the backend reads it from storage, so an array of any size costs nothing in
the spec or the emitted module". That held for dynamics parameters only. `materialise_lazy_params`
ran on `model.parameters` alone, while the coupling parameter dicts were built from
`get_param_info`, which reports names, defaults and shapes and never looks at `source`. A
parameter with no literal `value` is simply absent from the defaults, so the emission site's
``.get(name, 1.0)`` fallback won and a sourced per-edge matrix became ``jnp.full(shape, 1.0)``.

Nothing raised. Codegen succeeded, the module imported, the simulation ran to completion and
reported a plausible number, having integrated an all-ones connectome-shaped array in place of
the declared one. Coupling is where per-edge arrays live, so the failure landed precisely on the
case the slot exists for.

These assert on the emitted source rather than on a result, because a run cannot distinguish the
two: that is the whole character of the defect.
"""

import ast
import re

import numpy as np
import pytest

pytest.importorskip("tvboptim")

import h5py

from tvbo import SimulationExperiment

_SPEC = """
id: 7
dynamics:
  name: Drift
  label: Drift
  parameters:
    a: {name: a, value: -0.1}
  coupling_inputs:
    c: {name: c, description: "coupling"}
  state_variables:
    x:
      name: x
      equation: {lhs: "Derivative(x, t)", rhs: "a * x + c"}
      variable_of_interest: true
      coupling_variable: true
  output: [x]
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
  name: LinearCoupling
  label: LinearCoupling
  parameters:
    g: {name: g, value: 0.01}
    w:
      name: w
      heterogeneous: true
      shape: "(n_nodes, n_nodes)"
      # W_BINDING
  pre_expression: {rhs: "x_j * w"}
  post_expression: {rhs: "g * gx"}
  incoming_states: [x]
  local_states: [x]
integration:
  method: Heun
  duration: 20.0
  step_size: 1.0
  transient_time: 0.0
"""

DEPOSITED = np.array([[0.0, 7.25], [3.5, 0.0]])


def _render(tmp_path, w_binding):
    """Render the tvboptim module for a spec whose per-edge `w` is bound as given."""
    with h5py.File(tmp_path / "store.h5", "w") as f:
        f.create_dataset("w_edge", data=DEPOSITED)
    spec = tmp_path / "spec.yaml"
    spec.write_text(_SPEC.replace("      # W_BINDING\n", w_binding))
    exp = SimulationExperiment.from_file(str(spec))
    exp.configure()
    return exp.render_code("tvboptim")


SOURCED = "      source: store.h5\n      measure: w_edge\n"
LITERAL = "      value: 1.0\n"


def _binding(code, name="w"):
    """The emitted right-hand side bound to coupling parameter *name*.

    The generated source is formatted after rendering, so a long call is wrapped across lines
    and the binding cannot be read off one line.
    """
    m = re.search(rf'["\']{name}["\']:\s*(.+?),?\n\s*["\'#}}]', code, re.S)
    assert m, f"no binding for {name} in the emitted module"
    return m.group(1)


def test_a_sourced_coupling_parameter_is_loaded_from_its_store(tmp_path):
    """The emitted binding for `w` must read the artifact, never stand in a default."""
    binding = _binding(_render(tmp_path, SOURCED))
    assert "_load_param(" in binding, binding
    assert "jnp.full" not in binding, binding


def test_the_loader_is_defined_when_only_a_coupling_needs_it(tmp_path):
    """`_load_param` is emitted from the dynamics scope and was gated on dynamics alone.

    With no sourced dynamics parameter and no covariance, a coupling-only spec emitted calls to
    a helper that was never defined — a NameError rather than a silent wrong answer, but only
    reachable once the binding above stopped being defaulted away.
    """
    code = _render(tmp_path, SOURCED)
    assert "def _load_param(" in code
    compile(code, "generated_experiment.py", "exec")


def test_the_emitted_call_points_at_the_deposited_bytes(tmp_path):
    """Resolve the emitted (path, key) and confirm it is the declared array, not a stand-in."""
    code = _render(tmp_path, SOURCED)
    m = re.search(r'["\']w["\']:\s*_load_param\(\s*([^,]+),\s*([^,)]+)', code)
    assert m, "the `w` binding does not call _load_param"
    path, key = (ast.literal_eval(g.strip()) for g in m.groups())
    assert key == "w_edge", key
    with h5py.File(path, "r") as f:
        np.testing.assert_array_equal(f[key][()], DEPOSITED)


def test_a_literal_coupling_parameter_still_inlines(tmp_path):
    """The negative control: `value:` must keep binding inline, not route through the loader."""
    binding = _binding(_render(tmp_path, LITERAL))
    assert "_load_param(" not in binding, binding
    assert "jnp.full" in binding, binding
