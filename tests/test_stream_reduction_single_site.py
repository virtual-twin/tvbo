"""Every streamed run composes its reducers in one place.

The emitted module folds streamed observables in four situations -- a base run's measured window, an exploration cell, a bundle of trajectory-free observables, and the long fold after a tuning algorithm -- and each one used to spell the composition out itself. Four copies of one expression is four chances to pass a different `skip`, a different `settle` or no `progress` at all, and they had already drifted into three combinations of those arguments before this test existed. They now all call `_stream_reduction`, so a change to how a reduction is composed lands once.

The assertions are structural on purpose: what a fold *computes* is pinned by the streaming reducer tests, which compare a streamed observable against its post-scan form. What is pinned here is that no site builds its own.
"""

import pytest

pytest.importorskip("jax")
pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment, database_path  # noqa: E402

BASE = """
label: Streamed probe
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
    - {source: 0, target: 1, parameters: {weight: {value: 0.0}}, source_var: x_out, target_var: c_in, directed: true}
integration: {method: euler, step_size: 1.0, duration: 50.0, unit: ms}
observations:
  m_stream: {source: [x], aggregation: mean, reduce: streaming}
"""

SWEPT = """
explorations:
  sweep:
    name: sweep
    mode: product
    record: [m_stream]
    space:
      - parameter: Ramp.k
        domain: {lo: 0.1, hi: 0.9, n: 3}
"""

BUNDLED = SWEPT.replace("    record: [m_stream]\n", "")


def _tuned_source():
    """The post-tuning fold: a curated optimisation whose settled mean is streamed rather than materialised."""
    exp = SimulationExperiment.from_file(str(database_path / "experiments" / "EI_Tuning_FIC_EIB_Optimization.yaml"))
    exp.observations["mean_S_e"].reduce = "streaming"
    return exp.render_code("tvboptim")


SOURCES = {
    "base window": lambda: SimulationExperiment.from_string(BASE).render_code("tvboptim"),
    "exploration cell": lambda: SimulationExperiment.from_string(BASE + SWEPT).render_code("tvboptim"),
    "bundled observables": lambda: SimulationExperiment.from_string(BASE + BUNDLED).render_code("tvboptim"),
    "post-tuning fold": _tuned_source,
}


@pytest.fixture(scope="module", params=sorted(SOURCES))
def emitted(request):
    return request.param, SOURCES[request.param]()


def test_the_composition_is_written_once(emitted):
    """`_compose_reducers` is called from the helper and from nowhere else, so the sites cannot drift apart again.

    Counted over lines rather than over the raw text because the emitted module is formatted before it is written, and a call the formatter wrapped across lines is the same call.
    """
    where, code = emitted
    assert "def _stream_reduction" in code, f"{where}: streams without emitting the helper"
    calls = [ln for ln in code.splitlines() if "_compose_reducers(" in ln and not ln.lstrip().startswith("def ")]
    assert len(calls) == 1, f"{where}: composed in {len(calls)} places -- {calls}"


def test_every_fold_goes_through_the_helper(emitted):
    """A `prepare(reduce=...)` that streams names its reduction by calling `_stream_reduction`."""
    where, code = emitted
    folds = [line.strip() for line in code.splitlines() if line.strip().startswith("reduce=")]
    assert folds, f"{where}: nothing streams"
    assert all(f.startswith("reduce=_stream_reduction(") for f in folds), f"{where}: {folds}"


def test_the_emitted_module_compiles(emitted):
    """The helper is a plain module-level def, so a site calling it is only correct if the module still parses."""
    where, code = emitted
    compile(code, f"<{where}>", "exec")
