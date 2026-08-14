"""`initial_conditions.<sv>` exploration axis: a deterministic IC ensemble.

The scope sweeps the initial value of one state variable across grid cells (one trajectory per swept value) — distinct from the stochastic `n_trials` + `StateVariable.distribution` ensemble. Resolution is a pure string parser (`initial_conditions_axis_sv`, mirroring `network_axis_leaf`); codegen lowers the axis to a dummy `_ic_<sv>` grid slot plus a per-cell wrapper that writes each cell's value into the state variable's row of the initial state.
"""

import pytest

from tvbo.templates.tvboptim.utils import initial_conditions_axis_sv


# --- resolver (pure string parse) -------------------------------------------
@pytest.mark.parametrize(
    "ref, sv",
    [
        ("initial_conditions.E", "E"),
        ("initial_conditions.theta", "theta"),
        ("initial_conditions.V_m", "V_m"),
    ],
)
def test_resolves_state_variable(ref, sv):
    assert initial_conditions_axis_sv(ref) == sv


@pytest.mark.parametrize(
    "ref",
    [
        "ReducedWongWang.a",  # dynamics-scoped
        "FastLinearCoupling.G",  # coupling-scoped
        "network.conduction_speed",  # network-scoped
        "execution.random_seed",  # seed-scoped
        "E",  # bare name is not scoped
        "",
        None,
        3.0,
    ],
)
def test_out_of_scope_returns_none(ref):
    """A reference outside the scope routes through the other axis paths."""
    assert initial_conditions_axis_sv(ref) is None


@pytest.mark.parametrize("ref", ["initial_conditions.", "initial_conditions.E.mode0"])
def test_malformed_scope_raises(ref):
    """In-scope but not a single state-variable name fails at codegen, not silently."""
    with pytest.raises(ValueError):
        initial_conditions_axis_sv(ref)


# --- codegen lowering --------------------------------------------------------
pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment  # noqa: E402

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
      initial_value: 0.1
      equation: {lhs: "Derivative(theta, t)", rhs: "omega + c"}
      variable_of_interest: true
      coupling_variable: true
      # DIST
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
explorations:
  ic_sweep:
    name: ic_sweep
    mode: product
    space:
      - parameter: initial_conditions.theta
        # AXIS
"""
"""Minimal 2-node Kuramoto with an initial-condition sweep on theta; the `# DIST` and exploration `# AXIS` placeholders are substituted per test."""


def _render(tmp_path, axis, dist=""):
    spec = _SPEC.replace("      # DIST\n", dist).replace("        # AXIS\n", axis)
    p = tmp_path / "spec.yaml"
    p.write_text(spec)
    exp = SimulationExperiment.from_file(str(p))
    exp.configure()
    return exp.render_code("tvboptim")


def test_domain_axis_lowers_to_slot_and_wrapper(tmp_path, unwrapped):
    """A domain lo/hi/n IC axis emits a GridAxis dummy slot and the injecting wrapper."""
    code = _render(tmp_path, "        domain: {lo: 0.0, hi: 1.0, n: 5}\n")
    assert "grid_state.dynamics._ic_theta = GridAxis(" in code
    # The wrapper writes the swept value into theta's row (row 0) of the initial state.
    assert unwrapped("s.initial_state.dynamics = s.initial_state.dynamics.at[0].set(s.dynamics._ic_theta)") in unwrapped(code)
    # The result dimension keeps the declared dotted name.
    assert "initial_conditions.theta" in code


def test_explored_values_axis_lowers_to_data_slot(tmp_path, unwrapped):
    """Explicit explored_values emit a DataAxis dummy slot."""
    code = _render(tmp_path, "        explored_values: [0.1, 0.2, 0.3]\n")
    assert "grid_state.dynamics._ic_theta = DataAxis(" in code
    assert unwrapped("s.initial_state.dynamics.at[0].set(s.dynamics._ic_theta)") in unwrapped(code)


def test_distribution_on_swept_sv_is_rejected(tmp_path):
    """A distribution on the swept SV would resample and overwrite — fail at codegen."""
    with pytest.raises(ValueError, match="also declares a distribution"):
        _render(
            tmp_path,
            "        domain: {lo: 0.0, hi: 1.0, n: 5}\n",
            dist="      distribution: {name: Uniform, domain: {lo: 0.0, hi: 0.1}}\n",
        )


def test_unknown_sv_message(tmp_path):
    """Sweeping a non-existent state variable fails at codegen, not silently."""
    spec = _SPEC.replace("initial_conditions.theta", "initial_conditions.nope")
    spec = spec.replace("        # AXIS\n", "        domain: {lo: 0.0, hi: 1.0, n: 5}\n")
    p = tmp_path / "bad.yaml"
    p.write_text(spec)
    exp = SimulationExperiment.from_file(str(p))
    exp.configure()
    with pytest.raises((AssertionError, ValueError), match="unknown state variable"):
        exp.render_code("tvboptim")


# --- end-to-end --------------------------------------------------------------
def test_sweep_yields_distinct_per_cell_trajectories(tmp_path):
    """Each swept IC integrates from its own initial value into a distinct trajectory.

    The result is a keyed xarray with a first-class `initial_conditions.theta` dimension; the first post-step sample of each cell is its swept theta(0) plus one RK4 step of the omega drift, so the cells are ordered and distinct.
    """
    import numpy as np

    spec = _SPEC.replace("        # AXIS\n", "        explored_values: [0.0, 0.5, 1.0]\n")
    p = tmp_path / "spec.yaml"
    p.write_text(spec)
    exp = SimulationExperiment.from_file(str(p))
    exp.configure()
    r = exp.run("tvboptim", mode="exploration")

    expl = r.explorations.ic_sweep
    ic_axis = [a for a in expl.axes if str(getattr(a, "name", "")) == "initial_conditions.theta"]
    assert ic_axis, "no initial_conditions.theta axis in the exploration result"
    assert int(getattr(ic_axis[0], "n", 0)) == 3

    grid = expl.as_grid()  # keyed DataArray: (initial_conditions.theta, time, node)
    assert "initial_conditions.theta" in grid.dims
    np.testing.assert_allclose(np.asarray(grid.coords["initial_conditions.theta"]), [0.0, 0.5, 1.0])

    cells = np.asarray(grid.transpose("initial_conditions.theta", ...))
    flat = cells.reshape(cells.shape[0], -1)
    # Distinct trajectories, one per swept IC.
    assert np.unique(np.round(flat, 8), axis=0).shape[0] == 3
    # First sample tracks the swept IC (offset by one identical omega step).
    first = cells[:, 0, 0]
    assert first[0] < first[1] < first[2]
    np.testing.assert_allclose(np.diff(first), 0.5, atol=1e-6)
