"""`noise.sigma` exploration axis: sweeping the noise AMPLITUDE.

Noise amplitude is declared per state variable (`state_variables.<sv>.noise.parameters.sigma`) or once for the integration, and every backend holds it as an ordinary parameter leaf beside the dynamics parameters, so an axis that can NAME it sweeps it like any other parameter. The scope is what makes a noise-strength sweep declarable at all: without it the sweep is a Python driver mutating the spec in a loop.

Resolution is a pure string parser (`noise_axis_param`, mirroring `network_axis_leaf` and `initial_conditions_axis_sv`); codegen binds the grid directly to the backend's amplitude leaf, needing neither a dummy slot nor a wrapper because the amplitude is already numeric.
"""

import pytest

from tvbo.templates.tvboptim.utils import noise_axis_param


# --- resolver (pure string parse) -------------------------------------------
def test_resolves_the_amplitude():
    assert noise_axis_param("noise.sigma") == "sigma"


@pytest.mark.parametrize(
    "ref",
    [
        "ReducedWongWang.a",  # dynamics-scoped
        "FastLinearCoupling.G",  # coupling-scoped
        "network.conduction_speed",  # network-scoped
        "initial_conditions.theta",  # IC-scoped
        "execution.random_seed",  # seed-scoped
        "sigma",  # bare name is not scoped
        "",
        None,
        3.0,
    ],
)
def test_out_of_scope_returns_none(ref):
    """A reference outside the scope routes through the other axis paths."""
    assert noise_axis_param(ref) is None


@pytest.mark.parametrize("ref", ["noise.", "noise.nsig", "noise.intensity", "noise.key"])
def test_unsweepable_noise_attribute_raises(ref):
    """In-scope but not a sweepable amplitude fails at codegen.

    `nsig`/`intensity` are alternate SPELLINGS of the amplitude the reader normalises, not separate leaves; silently accepting one would sweep a slot the backend never reads.
    """
    with pytest.raises(ValueError, match="unknown noise parameter"):
        noise_axis_param(ref)


# --- codegen lowering --------------------------------------------------------
pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment  # noqa: E402

# A 2-node Kuramoto with noise on theta, whose `# AXIS` and `# NOISE` markers each test substitutes.
_SPEC = """
id: 8
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
      # NOISE
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
  method: Heun
  duration: 40.0
  step_size: 1.0
  transient_time: 0.0
execution:
  random_seed: 3
explorations:
  sigma_sweep:
    name: sigma_sweep
    mode: product
    space:
      - parameter: noise.sigma
        # AXIS
"""
"""A minimal 2-node Kuramoto with additive noise on theta and a sweep of its amplitude, where each test substitutes its own `# AXIS` line and `# NOISE` block."""

_NOISE = "      noise: {additive: true, gaussian: true, parameters: {sigma: {value: 0.01}}}\n"


def _render(tmp_path, axis, noise=_NOISE):
    spec = _SPEC.replace("      # NOISE\n", noise).replace("        # AXIS\n", axis)
    p = tmp_path / "spec.yaml"
    p.write_text(spec)
    exp = SimulationExperiment.from_file(str(p))
    exp.configure()
    return exp.render_code("tvboptim")


def test_domain_axis_binds_the_amplitude_leaf(tmp_path):
    """A domain lo/hi/n amplitude axis binds the noise params leaf as a GridAxis."""
    code = _render(tmp_path, "        domain: {lo: 0.005, hi: 0.05, n: 4}\n")
    assert "grid_state.noise.sigma = GridAxis(" in code
    # The result dimension keeps the declared dotted name.
    assert "noise.sigma" in code


def test_explored_values_axis_binds_a_data_axis(tmp_path):
    """Explicit explored_values bind the same leaf as a DataAxis."""
    code = _render(tmp_path, "        explored_values: [0.01, 0.02, 0.04]\n")
    assert "grid_state.noise.sigma = DataAxis(" in code


def test_sweeping_amplitude_without_noise_is_rejected(tmp_path):
    """No declared noise → no amplitude to sweep.

    Binding the leaf anyway would sweep a slot that does not exist, so this fails at codegen rather than silently doing nothing.
    """
    with pytest.raises(ValueError, match="declares no noise"):
        _render(tmp_path, "        domain: {lo: 0.005, hi: 0.05, n: 4}\n", noise="")


def test_a_builder_supplied_amplitude_axis_binds_the_noise_leaf(tmp_path, monkeypatch):
    """A `builder:` on this axis must reach the noise scope, not the dynamics scope.

    The builder branch is resolved before the scope is known, so it claimed the axis first and emitted `grid_state.dynamics.sigma` — an AttributeError on a model with no such parameter, or (worse) a silent sweep of the model's own `sigma` while the noise amplitude stayed constant in every cell.
    """
    import sys

    (tmp_path / "sigma_builder.py").write_text("def log_sigmas(n):\n    return [0.01 * 2 ** i for i in range(int(n))]\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("sigma_builder", None)

    axis = (
        "        builder:\n"
        "          callable: {name: log_sigmas, module: sigma_builder}\n"
        "          arguments: {n: {value: 3}}\n"
    )
    code = _render(tmp_path, axis)
    assert "grid_state.noise.sigma = DataAxis(" in code
    assert "grid_state.dynamics.sigma" not in code


_HETERO_SV = """      noise: {additive: true, gaussian: true, parameters: {sigma: {value: 0.01}}}
    phi:
      name: phi
      unit: rad
      initial_value: 0.0
      equation: {lhs: "Derivative(phi, t)", rhs: "omega"}
      variable_of_interest: true
      noise: {additive: true, gaussian: true, parameters: {sigma: {value: 0.05}}}
"""


def test_a_heterogeneous_per_state_amplitude_is_rejected(tmp_path):
    """One swept scalar cannot stand in for a per-state amplitude PROFILE.

    An experiment declaring different sigmas per state variable renders a sigma VECTOR.
    Overwriting it with one scalar per cell drives every targeted state at the same amplitude and loses the declared ratio between them — the sweep then answers a different question than the one written down, with nothing in the result saying so.
    """
    with pytest.raises(ValueError, match="HETEROGENEOUS"):
        _render(tmp_path, "        explored_values: [0.01, 0.02]\n", noise=_HETERO_SV)


def test_the_amplitude_coexists_with_a_model_parameter_of_the_same_name(tmp_path):
    """`noise.sigma` and a model's own `sigma` are two axes, and both must be placeable.

    These two share a bare leaf name, so an axis keyed on that alone lets one win and drops the other's grid column back to its raw keypath — a coordinate matching no declared axis, which fails the container's per-cell placement check and takes the whole sweep down after the compute. Each axis is therefore keyed by its declared scope, not its leaf.
    Generic2dOscillator, Zerlaut and Larter-Breakspear all carry a `sigma`.
    """
    import numpy as np

    spec = (
        _SPEC.replace("      # NOISE\n", _NOISE)
        .replace(
            "    omega: {name: omega, value: 0.0628, unit: rad_per_ms}\n",
            "    omega: {name: omega, value: 0.0628, unit: rad_per_ms}\n    sigma: {name: sigma, value: 1.0}\n",
        )
        .replace('rhs: "omega + c"', 'rhs: "omega + sigma * c"')
        .replace(
            "        # AXIS\n",
            "        explored_values: [0.01, 0.02]\n      - parameter: Kuramoto.sigma\n        explored_values: [1.0, 2.0]\n",
        )
    )
    p = tmp_path / "spec.yaml"
    p.write_text(spec)
    exp = SimulationExperiment.from_file(str(p))
    exp.configure()
    grid = exp.run("tvboptim", mode="exploration").explorations.sigma_sweep.as_grid()

    assert "noise.sigma" in grid.dims and "Kuramoto.sigma" in grid.dims
    np.testing.assert_allclose(np.asarray(grid.coords["noise.sigma"]), [0.01, 0.02])
    np.testing.assert_allclose(np.asarray(grid.coords["Kuramoto.sigma"]), [1.0, 2.0])


# --- end-to-end --------------------------------------------------------------
def test_sweep_yields_monotonically_noisier_trajectories(tmp_path):
    """Each swept amplitude drives a trajectory whose spread grows with sigma.

    The result is a keyed xarray with a first-class `noise.sigma` dimension. All cells share one noise seed, so the cells differ only through the amplitude — and for additive noise the deviation from the sigma=0 drift scales linearly in sigma, which is a far sharper check than "the cells differ".
    """
    import numpy as np

    code_axis = "        explored_values: [0.0, 0.01, 0.02, 0.04]\n"
    spec = _SPEC.replace("      # NOISE\n", _NOISE).replace("        # AXIS\n", code_axis)
    p = tmp_path / "spec.yaml"
    p.write_text(spec)
    exp = SimulationExperiment.from_file(str(p))
    exp.configure()
    r = exp.run("tvboptim", mode="exploration")

    expl = r.explorations.sigma_sweep
    axis = [a for a in expl.axes if str(getattr(a, "name", "")) == "noise.sigma"]
    assert axis, f"no noise.sigma axis in the exploration result (got {[str(getattr(a, 'name', '')) for a in expl.axes]})"
    assert int(getattr(axis[0], "n", 0)) == 4

    grid = expl.as_grid()
    assert "noise.sigma" in grid.dims
    np.testing.assert_allclose(np.asarray(grid.coords["noise.sigma"]), [0.0, 0.01, 0.02, 0.04])

    cells = np.asarray(grid.transpose("noise.sigma", ...))
    flat = cells.reshape(cells.shape[0], -1)
    assert np.unique(np.round(flat, 10), axis=0).shape[0] == 4, (
        "cells are identical — the amplitude is not reaching the integrator"
    )

    # Deviation from the deterministic (sigma=0) cell grows with sigma.
    dev = np.abs(flat[1:] - flat[0]).mean(axis=1)
    assert dev[0] < dev[1] < dev[2], f"spread not monotone in sigma: {dev}"


# --- grid keypath (the one definition of WHERE an axis binds) ----------------
@pytest.mark.parametrize(
    "ax,expected",
    [
        ({"name": "sigma", "is_noise": True}, "noise.sigma"),
        ({"name": "K"}, "dynamics.K"),
        ({"name": "a", "is_coupling": True, "coupling_key": "linear"}, "coupling.linear.a"),
        ({"name": "conduction_speed", "is_network": True, "graph_leaf": "speed"}, "graph.speed"),
    ],
)
def test_axis_keypath_routes_each_scope_to_its_sub_object(ax, expected):
    """One resolver answers WHERE an axis binds, for every scope.

    The grid binding, the warm-started/adiabatic sweep and the branch-analysis restart all need this string, and a scope missing from any one of them is silent: routing `noise.sigma` to `dynamics.sigma` sweeps a same-named model parameter (Generic2dOscillator and Zerlaut both have one) or nothing at all, and the run still completes with a plausible surface.
    """
    from tvbo.templates.tvboptim.utils import axis_keypath

    assert axis_keypath(ax) == expected


def test_the_sweep_partials_do_not_respell_the_keypath():
    """The warm-start and branch-restart partials must READ the resolver, not re-derive it.

    They have no rendering test of their own, so a second spelling here would drift unnoticed — which is how both partials came to route every non-coupling axis to `dynamics.<name>`.
    """
    from pathlib import Path

    import tvbo.templates.tvboptim as _pkg

    src = (Path(_pkg.__file__).parent / "tvbo-tvboptim-sweep.py.mako").read_text()
    assert src.count("axis_keypath(axis)") == 2, "a sweep partial stopped using the shared resolver"
    assert '"dynamics.%s"' not in src, "a sweep partial spells a grid keypath itself"


# --- reserved-scope gate ------------------------------------------------------
def test_a_misspelled_scope_is_rejected_rather_than_swept_as_a_model_parameter(tmp_path):
    """An unrecognised dotted scope must fail at codegen, not fall through to the dynamics.

    The dynamics branch DISCARDS the prefix and keeps the leaf, so `nosie.sigma` on a model carrying its own `sigma` binds `grid_state.dynamics.sigma` and sweeps the wrong quantity with nothing in the result saying so. The same hole hides a real-but-unimplemented scope.
    """
    spec = _SPEC.replace("      # NOISE\n", _NOISE).replace(
        "      - parameter: noise.sigma\n        # AXIS\n",
        "      - parameter: nosie.sigma\n        explored_values: [0.01, 0.02]\n",
    )
    p = tmp_path / "spec.yaml"
    p.write_text(spec)
    exp = SimulationExperiment.from_file(str(p))
    exp.configure()
    with pytest.raises(ValueError, match="unknown scope 'nosie'"):
        exp.render_code("tvboptim")


def test_the_dynamics_and_coupling_scopes_still_resolve(tmp_path):
    """The gate must not reject the two spellings that carry every ordinary sweep."""
    for parameter in ("Kuramoto.omega", "KuramotoCoupling.a"):
        spec = _SPEC.replace("      # NOISE\n", _NOISE).replace(
            "      - parameter: noise.sigma\n        # AXIS\n",
            f"      - parameter: {parameter}\n        explored_values: [0.01, 0.02]\n",
        )
        p = tmp_path / "spec.yaml"
        p.write_text(spec)
        exp = SimulationExperiment.from_file(str(p))
        exp.configure()
        assert exp.render_code("tvboptim"), parameter
