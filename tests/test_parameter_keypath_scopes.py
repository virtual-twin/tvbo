"""Where a declared parameter binds, for consumers holding it as a raw dotted string.

Several codegen consumers hold a parameter reference as TEXT rather than as an exploration axis the classifier has already split into scope flags: the `initial_state.from_working_point` ramp, the NSGA-II decision axes, an `Optimization`'s free parameters and the marked optimizer parameters, an analysis `wrt`, and an inference prior. Each had grown its own prefix ladder and they disagreed — the ramp knew no scope at all and hard-prefixed `dynamics.`, two knew `noise.` but not `network.`, and the `wrt`/prior grammar knew an `external.` scope none of the others did.

Every disagreement is silent. A `noise.sigma` ramp routed to `dynamics.sigma` ramps a same-named model parameter (Generic2dOscillator and Zerlaut both have one) or writes a slot nothing reads, and the run completes with a plausible working point. A `network.` free parameter marked on `dynamics.conduction_speed` gives the optimiser a leaf outside the gradient path, so the fit converges having moved nothing.

`parameter_keypath` is the one resolution of that question. Its counterpart `axis_keypath` answers it for an axis whose flags are already resolved; the two must agree, which `test_keypath_resolvers_agree.py` pins.
"""

import pytest

from tvbo.templates.tvboptim.utils import parameter_keypath

_COUPLINGS = {"LinCoupling": object(), "KuramotoCoupling": object()}
_CI_KEY = {"LinCoupling": "ci_lin"}.get


# --- resolution (pure string parse) ------------------------------------------
@pytest.mark.parametrize(
    "ref,expected",
    [
        ("w", "dynamics.w"),
        ("ReducedWongWang.w", "dynamics.w"),
        ("noise.sigma", "noise.sigma"),
        ("AdditiveNoise.sigma", "noise.sigma"),
        ("Noise.sigma", "noise.sigma"),
        ("network.conduction_speed", "graph.speed"),
        ("network.edges.weight", "graph.weights"),
        ("network.edges.length", "graph.lengths"),
        ("network.weights", "graph.weights"),
        ("initial_conditions.theta", "dynamics._ic_theta"),
        ("execution.random_seed", "dynamics._noise_seed"),
    ],
)
def test_each_scope_resolves_to_its_own_sub_object(ref, expected):
    assert parameter_keypath(ref) == expected


def test_a_declared_coupling_prefix_routes_to_that_coupling():
    """The prefix→scope decision depends on what the experiment declares."""
    got = parameter_keypath("LinCoupling.G", couplings=_COUPLINGS, coupling_key=_CI_KEY)
    assert got == "coupling.ci_lin.G"


def test_an_undeclared_class_prefix_is_a_dynamics_parameter():
    """`ReducedWongWang.w` names the model that declares `w`; the prefix is not a scope."""
    assert parameter_keypath("ReducedWongWang.w", couplings=_COUPLINGS) == "dynamics.w"


def test_the_coupling_key_mapper_is_optional():
    """Without a mapper the declared coupling name IS the state key."""
    assert parameter_keypath("KuramotoCoupling.a", couplings=_COUPLINGS) == "coupling.KuramotoCoupling.a"


def test_a_declared_event_prefix_routes_to_that_external_input():
    """A stimulus event's parameters are addressable the same way a coupling's are."""
    assert parameter_keypath("stimulus.amplitude", external={"stimulus"}) == "external.stimulus.amplitude"


# --- the analysis/inference grammar is the same grammar -----------------------
def test_the_config_access_grammar_delegates_to_the_one_resolver():
    """Analysis `wrt` and inference priors address a knob the way a fit or a sweep does.

    This grammar knew `coupling.` and `external.` and nothing else, so a `noise.sigma` prior sampled a posterior and wrote it into `dynamics.sigma` — a slot the integrator never reads for the amplitude. The chain converges and reports a distribution for the wrong parameter.
    """
    from tvbo.templates.tvboptim.utils import resolve_config_access

    assert resolve_config_access("noise.sigma", set()) == "noise.sigma"
    assert resolve_config_access("network.conduction_speed", set()) == "graph.speed"
    assert resolve_config_access("LinCoupling.G", {"LinCoupling"}) == "coupling.LinCoupling.G"
    assert resolve_config_access("stimulus.amplitude", set(), {"stimulus"}) == "external.stimulus.amplitude"
    assert resolve_config_access("ReducedWongWang.w", set()) == "dynamics.w"
    assert resolve_config_access("w", set()) == "dynamics.w"


@pytest.mark.parametrize("empty", ["", None])
def test_an_empty_reference_has_no_config_path(empty):
    """Callers distinguish "no knob declared" from "the dynamics scope"."""
    from tvbo.templates.tvboptim.utils import resolve_config_access

    assert resolve_config_access(empty, set()) is None


@pytest.mark.parametrize(
    "ref,match",
    [
        ("noise.nsig", "unknown noise parameter"),
        ("network.conduction_delay", "unknown network attribute"),
        ("network.edges.tract_count", "has no graph leaf to sweep"),
        ("initial_conditions.", "single"),
    ],
)
def test_a_reserved_scope_naming_nothing_bindable_raises(ref, match):
    """In-scope but unbindable fails at codegen, not by falling through to `dynamics.<leaf>`.

    Falling through is the silent case: the reference resolves to a same-named model parameter or creates a dead slot, and nothing in the result says the declared thing never moved.
    """
    with pytest.raises(ValueError, match=match):
        parameter_keypath(ref)


# --- codegen lowering --------------------------------------------------------
pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment  # noqa: E402

_SPEC = """
id: 8
dynamics:
  name: Kuramoto
  label: Kuramoto
  parameters:
    omega: {name: omega, value: 0.0628, unit: rad_per_ms}
    sigma: {name: sigma, value: 7.0}
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
      noise: {additive: true, gaussian: true, parameters: {sigma: {value: 0.01}}}
  output: [theta]
  number_of_modes: 1
network:
  number_of_nodes: 2
  parameters:
    conduction_speed: {name: conduction_speed, value: 3.0, unit: mm_per_ms}
  nodes:
    - {id: 0, label: r0}
    - {id: 1, label: r1}
  edges:
    - {source: 0, target: 1, weight: 0.5, distance: 10.0}
    - {source: 1, target: 0, weight: 0.5, distance: 10.0}
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
  transient_time: 10.0
execution:
  random_seed: 3
initial_state:
  method: from_working_point
  ramp:
    parameter: RAMP_PARAM
    domain: {lo: 0.0, hi: 0.02, n: 3}
"""


def _render(tmp_path, ramp_param):
    spec = _SPEC.replace("RAMP_PARAM", ramp_param)
    p = tmp_path / "spec.yaml"
    p.write_text(spec)
    exp = SimulationExperiment.from_file(str(p))
    exp.configure()
    return exp.render_code("tvboptim")


def test_a_noise_scoped_ramp_reaches_the_noise_amplitude(tmp_path):
    """The ramp accessor must point at the amplitude, not at the model's own `sigma`.

    This model declares both, which is the case that makes the wrong route invisible: the run ramps a dynamics parameter to its target, settles, and seeds a working point that was never the declared one.
    """
    code = _render(tmp_path, "noise.sigma")
    assert "accessor=lambda _c: _c.noise.sigma," in code
    assert "_c.dynamics.sigma" not in code


def test_a_network_scoped_ramp_reaches_the_delay_graph(tmp_path):
    code = _render(tmp_path, "network.conduction_speed")
    assert "accessor=lambda _c: _c.graph.speed," in code
    assert "_c.dynamics.conduction_speed" not in code


def test_a_coupling_scoped_ramp_reaches_its_coupling_instance(tmp_path):
    """The scope that already worked must keep working through the shared resolver."""
    code = _render(tmp_path, "KuramotoCoupling.a")
    assert "accessor=lambda _c: _c.coupling." in code
    assert "_c.dynamics.a" not in code


def test_a_dynamics_ramp_still_binds_the_dynamics(tmp_path):
    code = _render(tmp_path, "Kuramoto.omega")
    assert "accessor=lambda _c: _c.dynamics.omega," in code


def test_an_unbindable_ramp_scope_fails_at_codegen(tmp_path):
    with pytest.raises(ValueError, match="unknown network attribute"):
        _render(tmp_path, "network.conduction_delay")


# --- no consumer respells the ladder ----------------------------------------
def test_no_codegen_site_hardcodes_the_scope_prefixes():
    """Four sites once carried this ladder with three different behaviours.

    They have no shared rendering test, so a fifth copy would drift unnoticed — which is how the ramp came to know no scope at all while the free-parameter path knew two of four.
    """
    from pathlib import Path

    import tvbo.templates.tvboptim as _pkg

    src = (Path(_pkg.__file__).parent / "tvbo-tvboptim-experiment.py.mako").read_text()
    assert "'AdditiveNoise', 'Noise'" not in src, "a codegen site spells the noise scope itself"
    assert 'f"dynamics.{_fpn}"' not in src, "the free-parameter path re-derives its keypath"
    assert "init_state.dynamics.${fp_name}" not in src, "mark_parameters re-derives its keypath"
    assert src.count("parameter_keypath(") >= 4
