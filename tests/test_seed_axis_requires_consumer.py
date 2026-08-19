"""An `execution.random_seed` axis must have something that consumes the seed.

The axis reseeds the stochastic solver's PRNG key (``config.noise.key``). On a deterministic experiment there is no key to reseed, so the swept leaf is read by nothing and every grid cell returns an identical result — a silent no-op that still shows up in the result container as a genuine-looking ensemble dimension. Codegen rejects it instead, so a recipe cannot quietly produce a fake trial ensemble.
"""

import copy

import pytest

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment

# Minimal single-node model with an instantaneous coupling; noise is added per-test.
MINI_EXP = {
    "id": 1,
    "label": "seed-axis consumer unit fixture",
    "dynamics": {
        "name": "MiniOsc",
        "system_type": "continuous",
        "output": ["x"],
        "parameters": {"a": {"value": 1.0}},
        "coupling_inputs": {"c": {}},
        "state_variables": {
            "x": {
                "equation": {"rhs": "-a * x + c"},
                "initial_value": 0.1,
                "coupling_variable": True,
            },
        },
    },
    "network": {
        "number_of_nodes": 2,
        "coupling": {
            "c": {
                "delayed": False,
                "local_states": ["x"],
                "pre_expression": {"rhs": "x"},
                "post_expression": {"rhs": "gx_0"},
            }
        },
    },
    "integration": {
        "method": "heun",
        "step_size": 0.1,
        "duration": 1.0,
        "transient_time": 0.0,
        "unit": "s",
    },
    "explorations": {
        "seed_sweep": {
            "name": "seed_sweep",
            "mode": "product",
            "record": ["x"],
            "space": [
                {
                    "parameter": "execution.random_seed",
                    "domain": {"lo": 0, "hi": 3, "n": 4},
                }
            ],
        }
    },
}


def _with_noise(spec, sigma=0.01):
    spec = copy.deepcopy(spec)
    spec["integration"]["noise"] = {"parameters": {"sigma": {"value": sigma}}}
    return spec


def test_seed_axis_without_noise_is_rejected():
    """No noise => nothing reads the seed => codegen error, not identical cells."""
    with pytest.raises(ValueError, match="has no consumer"):
        SimulationExperiment(**copy.deepcopy(MINI_EXP)).render_code("tvboptim")


def test_seed_axis_error_names_the_axis_and_a_way_forward():
    """The message must identify the axis and point at a mechanism that does apply."""
    with pytest.raises(ValueError) as excinfo:
        SimulationExperiment(**copy.deepcopy(MINI_EXP)).render_code("tvboptim")
    message = str(excinfo.value)
    assert "execution.random_seed" in message
    assert "n_trials" in message


def test_seed_axis_with_noise_still_renders():
    """The positive control: with noise the axis has a consumer and codegen proceeds."""
    code = SimulationExperiment(**_with_noise(MINI_EXP)).render_code("tvboptim")
    assert "_noise_seed" in code
    assert "noise.key" in code


def test_no_seed_axis_is_unaffected_without_noise():
    """A deterministic experiment with no seed axis must still render."""
    spec = copy.deepcopy(MINI_EXP)
    spec["explorations"]["seed_sweep"]["space"] = [{"parameter": "MiniOsc.a", "domain": {"lo": 0.5, "hi": 1.5, "n": 3}}]
    code = SimulationExperiment(**spec).render_code("tvboptim")
    assert "grid_state.dynamics.a" in code


def test_seed_axis_is_rejected_under_a_strategy_that_bypasses_the_grid():
    """Noise alone is not enough: the seed must also reach the per-cell grid binding.

    nsga2 / warm-start / branch-analysis bodies never execute the grid-binding block that applies the swept seed, so the axis is inert there even on a stochastic experiment — the same fake ensemble, just arrived at a different way.
    """
    spec = _with_noise(copy.deepcopy(MINI_EXP))
    spec["explorations"]["seed_sweep"]["strategy"] = "nsga2"
    with pytest.raises(ValueError, match="has no consumer"):
        SimulationExperiment(**spec).render_code("tvboptim")


def test_a_builder_supplied_seed_axis_still_reseeds(tmp_path, monkeypatch):
    """A seed axis whose values come from a `builder:` must reach the PRNG key too.

    The builder branch used to claim this axis first and route it through the generic parameter path, where `execution` has no consumer — so every cell ran the identical noise while the container still reported a seed dimension. That is precisely the fake ensemble the checks above exist to refuse, arrived at by walking past them. Seed values are baked into the grid at codegen, so a builder on this axis is resolved there.
    """
    import sys

    (tmp_path / "seed_builder.py").write_text("def paired_seeds(n):\n    return list(range(int(n))) + list(range(int(n)))\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("seed_builder", None)

    spec = _with_noise(copy.deepcopy(MINI_EXP))
    spec["explorations"]["seed_sweep"]["space"] = [
        {
            "parameter": "execution.random_seed",
            "builder": {"callable": {"name": "paired_seeds", "module": "seed_builder"}, "arguments": {"n": {"value": 3}}},
        }
    ]
    code = SimulationExperiment(**spec).render_code("tvboptim")
    assert "_noise_seed" in code and "noise.key" in code, "the seed must reach the PRNG key"
    assert "random_seed[6]" in code, "the builder's six seeds must be resolved at codegen"
    assert "seed_builder" not in code, "a resolved seed axis carries values, not a call"


def test_a_seed_builder_needing_runtime_data_is_refused(tmp_path, monkeypatch):
    """Refuse rather than defer: the grid's seeds are fixed at codegen, so a builder that cannot answer until run time has no way to supply them."""
    spec = _with_noise(copy.deepcopy(MINI_EXP))
    spec["explorations"]["seed_sweep"]["space"] = [
        {
            "parameter": "execution.random_seed",
            "builder": {
                "callable": {"name": "paired_seeds", "module": "seed_builder"},
                "arguments": {"n": {"value": "observations.rate"}},
            },
        }
    ]
    with pytest.raises(ValueError, match="resolve at run time"):
        SimulationExperiment(**spec).render_code("tvboptim")


def test_two_axis_seed_sweep_maps_the_noise_seed_leaf_to_its_label():
    """A (parameter x seed) product keys results by value, and the seed axis's grid column is the ``dynamics._noise_seed`` state leaf — codegen must map that bare name onto the declared ``execution.random_seed`` label, or cell placement cannot find the axis and the container assembly refuses rather than scrambling."""
    spec = _with_noise(copy.deepcopy(MINI_EXP))
    spec["explorations"]["seed_sweep"]["space"].insert(0, {"parameter": "MiniOsc.a", "domain": {"lo": 0.5, "hi": 1.5, "n": 3}})
    code = SimulationExperiment(**spec).render_code("tvboptim")
    squeezed = "".join(code.split()).replace('"', "'")
    assert "_register('_noise_seed')" in squeezed
    assert "if_name=='execution.random_seed':" in squeezed


_ZIP_STREAM_SPEC = """
id: 9
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
      noise: {additive: true, gaussian: true, parameters: {sigma: {value: 0.5}}}
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
observations:
  m:
    source: theta
    aggregation: mean
    reduce: streaming
execution:
  random_seed: 3
explorations:
  seed_sweep:
    name: seed_sweep
    mode: MODE
    space:
      - parameter: Kuramoto.omega
        explored_values: [0.05, 0.07]
      - parameter: execution.random_seed
        explored_values: SEEDS
"""


def _streamed_cells(tmp_path, mode, seeds, tag):
    """The observation values of a bundled STREAMING exploration, one row per cell."""
    import numpy as np

    spec = _ZIP_STREAM_SPEC.replace("MODE", mode).replace("SEEDS", str(list(seeds)))
    path = tmp_path / f"{tag}.yaml"
    path.write_text(spec)
    exp = SimulationExperiment.from_file(str(path))
    exp.configure()
    obs = exp.run("tvboptim", mode="exploration").explorations.seed_sweep.observations
    return np.asarray(obs["m" if "m" in obs else next(iter(obs))])


@pytest.mark.parametrize("mode", ["zip", "product"])
def test_a_seed_axis_reseeds_a_streamed_observation(tmp_path, mode):
    """The seed must reach `noise.key` on the STREAMING path, and under either grid mode.

    A bundled streaming observation folds into the integrator carry through
    `prepare(reduce=...)` rather than reading a materialised trajectory, and the per-cell
    reseeding wrapper composes on top of it. If it did not, every cell of the sweep would
    integrate the same noise and the container would still report a seed dimension — the
    fake ensemble the checks above refuse at codegen, arrived at after it.
    """
    import numpy as np

    a = _streamed_cells(tmp_path, mode, [0, 1], f"{mode}_a")
    b = _streamed_cells(tmp_path, mode, [5, 6], f"{mode}_b")
    assert not np.allclose(a, b), (
        f"{mode} + streaming: the cells are identical under two different seed sets, so the "
        "seed axis never reached the solver's PRNG key")
