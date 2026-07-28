"""`CorrelatedNoiseSolver` is emitted exactly when a covariance is declared.

`Noise.covariance` + `Noise.correlated_over` are the sole signal that the Wiener
increment carries structure. Absent them the generated tvboptim code must neither
import nor use the wrapper — noise stays independent, which is what an undecorated
`noise:` block means.

The inverse matters more, and is the reason this file exists: a feature wired into only
one of the two codegen paths is dead code on the other, and the failure is silent —
the run completes and reports success while integrating uncorrelated noise. Both paths
are therefore checked, exactly as `test_bounded_solver_codegen.py` checks both.
"""

import pytest

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment
from tvbo.classes.experiment import templates

# A minimal two-node network whose single state variable is driven by noise.
# `# COVARIANCE` is replaced with the covariance declaration (or nothing).
_SPEC = """
id: 9
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
      noise:
        intensity: {name: intensity, value: 0.01}
        # COVARIANCE
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
    a: {name: a, value: 0.01}
  pre_expression: {rhs: "x_j"}
  post_expression: {rhs: "a * gx"}
  incoming_states: [x]
  local_states: [x]
integration:
  method: Heun
  duration: 20.0
  step_size: 1.0
  transient_time: 0.0
"""

# A 2x2 covariance over the node axis: unit variances, correlation 0.6.
_COVARIANCE = """        correlated_over: node
        covariance:
          name: covariance
          value: [[1.0, 0.6], [0.6, 1.0]]
"""


def _render(tmp_path, covariance_block):
    """Render both tvboptim codegen paths for a spec with the given covariance block."""
    spec = _SPEC.replace("        # COVARIANCE\n", covariance_block)
    p = tmp_path / "spec.yaml"
    p.write_text(spec)
    exp = SimulationExperiment.from_file(str(p))
    exp.configure()
    experiment_code = exp.render_code("tvboptim")
    solver_code = templates.lookup.get_template(
        "tvboptim/tvbo-tvboptim-solver.py.mako"
    ).render(experiment=exp)
    return experiment_code, solver_code


def test_plain_noise_omits_the_correlated_solver(tmp_path):
    """Noise without a covariance is independent; the wrapper must not appear."""
    experiment_code, solver_code = _render(tmp_path, "")
    for code in (experiment_code, solver_code):
        assert "CorrelatedNoiseSolver" not in code


def test_declared_covariance_emits_the_correlated_solver(tmp_path):
    """Both codegen paths must wrap the solver — one alone would be dead code."""
    experiment_code, solver_code = _render(tmp_path, _COVARIANCE)
    for code in (experiment_code, solver_code):
        assert "import CorrelatedNoiseSolver" in code or (
            "CorrelatedNoiseSolver" in code and "covariance_factor" in code
        )
        assert "CorrelatedNoiseSolver(" in code
        assert "axis='node'" in code or 'axis="node"' in code
        assert "[[1.0, 0.6], [0.6, 1.0]]" in code


def test_generated_module_builds_a_wrapping_solver(tmp_path):
    """The emitted `get_solver()` must actually return the wrapper, not just import it.

    Executes the generated solver module, which is what catches a wrapper that is
    constructed and then dropped (`base_solver` returned instead of the wrapped one).
    """
    _, solver_code = _render(tmp_path, _COVARIANCE)
    namespace = {"__name__": "generated_solver"}
    exec(compile(solver_code, "generated_solver.py", "exec"), namespace)

    solver = namespace["get_solver"]()
    assert type(solver).__name__ == "_CorrelatedNoiseSolver"
    # The declared covariance survived factorisation: L Lᵀ == C.
    import numpy as np

    L = np.asarray(solver.factor)
    assert np.allclose(L @ L.T, [[1.0, 0.6], [0.6, 1.0]], atol=1e-10)


def test_the_declared_covariance_reaches_the_integrated_trajectory(tmp_path):
    """End-to-end: run the model and measure the correlation it actually integrated.

    Every check above inspects generated source, which a wrapper that is emitted but
    never reached would still satisfy. This one runs the simulation. The model is a
    pure random walk (zero drift, zero coupling), so the increments of `x` ARE the
    Wiener increments and the declared node-node correlation is directly measurable.
    """
    import numpy as np

    def _run(covariance_block):
        spec = _SPEC.replace("        # COVARIANCE\n", covariance_block)
        # Pure random walk, long enough that the sample correlation is precise to ~0.007.
        spec = spec.replace('rhs: "a * x + c"', 'rhs: "a * x + 0*c"')
        spec = spec.replace("value: -0.1", "value: 0.0")
        spec = spec.replace("duration: 20.0", "duration: 20000.0")
        spec = spec.replace("value: 0.01", "value: 1.0")
        p = tmp_path / f"rw_{len(covariance_block)}.yaml"
        p.write_text(spec + "execution: {random_seed: 0}\n")
        exp = SimulationExperiment.from_file(str(p))
        exp.configure()
        data = exp.run("tvboptim").integration.data
        arr = np.asarray(
            data.sel(variable="x") if "variable" in data.dims else data
        ).squeeze()
        increments = np.diff(arr.reshape(arr.shape[0], -1), axis=0)
        return float(np.corrcoef(increments[:, 0], increments[:, 1])[0, 1])

    assert abs(_run("")) < 0.05, "undecorated noise must stay independent across nodes"
    assert abs(_run(_COVARIANCE) - 0.6) < 0.05, "the declared correlation must be realised"


def test_covariance_without_an_axis_is_rejected(tmp_path):
    """A covariance with no `correlated_over` does not identify a process."""
    block = """        covariance:
          name: covariance
          value: [[1.0, 0.6], [0.6, 1.0]]
"""
    with pytest.raises(ValueError, match="correlated_over"):
        _render(tmp_path, block)
