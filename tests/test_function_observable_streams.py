"""A sweep scored by a function of streamed observations folds them in-carry, as a sweep that bundles them does.

An exploration declaring ``observable: {function: ..., arguments: ...}`` built every cell on the materialise path whatever its observations declared, so ``reduce: streaming`` was honoured by the base run and silently dropped by the sweep. A 379-node network at 0.1 ms over 660 s holds about 40 GB of trajectory per cell, and the first such sweep reached an 83 GB footprint before it was killed. The function now reads its inputs from the folded values whenever the base run streams the experiment's observations, and a plain observation beside them puts the sweep back on the materialise path. Both paths hand the function the same value under the same declaration, compared keyed by the swept parameter.
"""

import re

import pytest
import xarray as xr

pytest.importorskip("jax")
pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment  # noqa: E402

SPEC = """
label: Function-scored sweep
dynamics:
  name: Ramp
  parameters: {k: {value: 0.5}}
  state_variables:
    x: {equation: {rhs: "k + c_in"}, initial_value: 0.0}
  coupling_inputs: {c_in: {}}
network:
  label: Pair
  number_of_nodes: 2
  nodes: [{id: 0, label: A, dynamics: Ramp}, {id: 1, label: B, dynamics: Ramp}]
  edges:
    - {source: 0, target: 1, parameters: {weight: {value: 0.3}}, source_var: x_out, target_var: c_in, directed: true}
integration: {method: euler, step_size: 1.0, duration: 50.0, unit: ms}
functions:
  spread:
    source_code: "jnp.max(data) - jnp.min(data)"
    arguments:
      data: {description: "The observation's values."}
  same:
    source_code: "data * 1.0"
    arguments:
      data: {description: "The observation's values."}
  scaled:
    source_code: "data * alpha"
    arguments:
      data: {description: "The observation's values."}
      alpha: {description: "A literal factor."}
observations:
{observations}
explorations:
  sweep:
    mode: product
    space:
      - parameter: Ramp.k
        domain: {lo: 0.1, hi: 0.9, n: 3}
    observable:
      function: {function}
      arguments:
        data: {value: m.data}{extra}
"""

STREAMED = {
    "mean": "  m: {source: [x], aggregation: mean, reduce: streaming}",
    "observer": """  m:
    source: [x]
    period: 10.0
    reduce: streaming
    dynamics:
      name: RunningSum
      system_type: discrete
      state_variables:
        acc: {equation: {rhs: 'acc + x'}, initial_value: 0.0}
      output: [acc]""",
}

PLAIN = "\n  raw: {source: [x]}"


def _experiment(observation: str, function: str = "spread", folded: bool = True, extra: str = "") -> SimulationExperiment:
    observations = STREAMED[observation] + ("" if folded else PLAIN)
    spec = SPEC.replace("{observations}", observations).replace("{function}", function).replace("{extra}", extra)
    return SimulationExperiment.from_string(spec)


def _sweep_body(code: str) -> str:
    """The emitted exploration function, up to the next module-level definition."""
    return re.search(r"^def sweep\(.*?(?=^\S)", code, flags=re.M | re.S).group(0)


@pytest.mark.parametrize("observation", sorted(STREAMED))
def test_a_streamed_observation_is_folded_in_the_sweep(observation):
    assert "reduce=_stream_reduction(" in _sweep_body(_experiment(observation).render_code("tvboptim"))


@pytest.mark.parametrize("observation", sorted(STREAMED))
def test_a_set_the_base_run_cannot_stream_keeps_the_materialise_path(observation):
    assert "reduce=" not in _sweep_body(_experiment(observation, folded=False).render_code("tvboptim"))


@pytest.mark.parametrize("function", ["spread", "same"])
@pytest.mark.parametrize("observation", sorted(STREAMED))
def test_the_function_sees_the_same_value_either_way(observation, function):
    def scores(folded):
        return _experiment(observation, function, folded).run("tvboptim").explorations["sweep"].as_grid()

    folded, materialised = scores(True), scores(False)
    assert "Ramp.k" in folded.dims
    xr.testing.assert_allclose(folded, materialised, rtol=1e-10, atol=1e-12)
    assert float(folded.max() - folded.min()) > 0, "the swept parameter must move the score, or equality proves nothing"


def test_a_scoring_argument_is_a_literal_an_output_or_a_runtime_input():
    from tvbo.templates.tvboptim.utils import parse_observable_args

    extra = "".join(
        f"\n        {name}: {spec}"
        for name, spec in {"a": "{value: 0.5}", "b": "{value: 5}", "c": "{value: fc}", "d": "{}"}.items()
    )
    arguments = _experiment("mean", extra=extra).explorations["sweep"].observable.arguments
    assert parse_observable_args(arguments) == [
        {"name": "data", "obs": "m", "key": "data"},
        {"name": "a", "literal": 0.5},
        {"name": "b", "literal": 5},
        {"name": "c", "obs": "fc", "key": "data"},
        {"name": "d", "obs": None, "key": None},
    ]


@pytest.mark.parametrize("folded", [True, False])
def test_a_numeric_argument_reaches_the_function_as_written(folded):
    """`alpha: {value: 0.5}` reaches the function as 0.5, not as output `5` of an observation named `0`."""

    def scores(function, extra=""):
        return _experiment("mean", function, folded, extra).run("tvboptim").explorations["sweep"].as_grid()

    xr.testing.assert_allclose(scores("scaled", "\n        alpha: {value: 0.5}"), 0.5 * scores("same"), rtol=1e-12)
