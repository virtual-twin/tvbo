"""`BoundedSolver` is emitted only when a state variable is deliberately clamped.

A ``domain`` with ``enforce: clamp`` and a finite bound is the sole signal that the integrator should hard-clip the trajectory. Absent that, the generated tvboptim code must neither import nor use ``BoundedSolver``: a bare descriptive ``domain`` (``enforce: none``, the default) states bounds as metadata but never constrains integration, so wrapping the solver would silently change the dynamics. A stray import is not merely cosmetic here — it advertises clamping the code does not do.

Both codegen paths are checked: the experiment template (the production path taken by ``render_code('tvboptim')``) and the standalone solver template (the component/sim path).
"""

import pytest

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment
from tvbo.classes.experiment import templates

_SPEC = """
id: 7
dynamics:
  name: Kuramoto
  label: Kuramoto
  parameters:
    omega: {name: omega, value: 0.0628, unit: rad_per_ms}
  coupling_inputs:
    c: {name: c, description: "coupling"}
  state_variables:
    theta:
      name: theta
      unit: rad
      equation: {lhs: "Derivative(theta, t)", rhs: "omega + c"}
      variable_of_interest: true
      coupling_variable: true
      # DOMAIN
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
  pre_expression: {rhs: "sin(theta_j - theta_i)"}
  post_expression: {rhs: "a * gx / N"}
  incoming_states: [theta]
  local_states: [theta]
integration:
  method: RungeKutta4thOrder
  duration: 20.0
  step_size: 1.0
  transient_time: 0.0
"""
"""A minimal single-population network; the ``# DOMAIN`` placeholder is replaced with a state-variable ``domain:`` line (or with nothing) to toggle whether clamping is requested."""


def _render(tmp_path, domain_line):
    """Render both tvboptim codegen paths for a spec with the given domain line."""
    spec = _SPEC.replace("      # DOMAIN\n", domain_line)
    p = tmp_path / "spec.yaml"
    p.write_text(spec)
    exp = SimulationExperiment.from_file(str(p))
    exp.configure()
    experiment_code = exp.render_code("tvboptim")
    solver_code = templates.lookup.get_template("tvboptim/tvbo-tvboptim-solver.py.mako").render(experiment=exp)
    return experiment_code, solver_code


def test_no_domain_omits_bounded_solver(tmp_path):
    """No domain at all -> no clamping -> BoundedSolver never appears."""
    experiment_code, solver_code = _render(tmp_path, "")
    assert "BoundedSolver" not in experiment_code
    assert "BoundedSolver" not in solver_code


def test_unenforced_domain_omits_bounded_solver(tmp_path):
    """A descriptive domain (enforce defaults to 'none') is metadata, not a clamp."""
    domain_line = "      domain: {lo: -3.2, hi: 3.2}\n"
    experiment_code, solver_code = _render(tmp_path, domain_line)
    assert "BoundedSolver" not in experiment_code
    assert "BoundedSolver" not in solver_code


def test_clamp_domain_emits_bounded_solver(tmp_path):
    """enforce: clamp with finite bounds -> BoundedSolver is imported and used."""
    domain_line = "      domain: {lo: -3.2, hi: 3.2, enforce: clamp}\n"
    experiment_code, solver_code = _render(tmp_path, domain_line)
    for code in (experiment_code, solver_code):
        assert "import BoundedSolver" in code
        assert "BoundedSolver(" in code
