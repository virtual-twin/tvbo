"""A first-passage observation must be able to stream instead of keeping its trajectory.

``aggregation: first_passage`` is computed post-scan as an ``argmax`` over the time axis, which means every cell of a sweep has to deliver its whole trajectory to the host before the reduction runs. That is what caps a declarative first-passage sweep: a 31x31 grid of the Schirner2023 decision circuit was OOM-killed at an 8.3 GB peak, in the phase after the last grid batch, while the same grid through a row-chunked reference integrator held 371 MB.

The crossing index is a two-scalar recurrence, so ``reduce: streaming`` now folds it into the integrator carry. Streaming is only worth having if it agrees with the post-scan form exactly, so that equality is the test — on a source that crosses mid-run, on one that crosses at the very first sample, and on one that never crosses at all.
"""

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment  # noqa: E402

SPEC = """
label: First-passage streaming
dynamics:
  name: Ramp
  parameters:
    k: {value: %(k)s}
  state_variables:
    x: {equation: {rhs: "k + c_in"}, initial_value: %(x0)s}
  coupling_inputs: {c_in: {}}
network:
  label: Pair
  number_of_nodes: 2
  nodes: [{id: 0, label: A, dynamics: Ramp}, {id: 1, label: B, dynamics: Ramp}]
  edges:
    - {source: 0, target: 1, parameters: {weight: {value: 0.0}}, source_var: x_out, target_var: c_in, directed: true}
integration: {method: euler, step_size: 1.0, duration: 50.0, unit: ms}
observations:
  t_host:
    label: First passage, post-scan
    source: [x]
    aggregation: first_passage
    parameters: {threshold: {value: 10.0}}
  t_stream:
    label: First passage, streamed
    source: [x]
    aggregation: first_passage
    reduce: streaming
    parameters: {threshold: {value: 10.0}}
"""


def run(k, x0):
    exp = SimulationExperiment.from_string(SPEC % {"k": k, "x0": x0})
    obs = exp.run("tvboptim").observations
    # Squeezed: the post-scan path keeps a leading singleton axis the streaming path does not, for a streamed mean as much as for this — a shape difference between the two reduction paths that predates streaming first passage and is not what this test is about.
    return tuple(np.squeeze(np.asarray(getattr(obs[n], "values", obs[n]))) for n in ("t_host", "t_stream"))


@pytest.mark.parametrize(
    "case,k,x0",
    [("crosses mid-run", 0.5, 0.0), ("crosses at the first sample", 0.5, 20.0), ("never crosses", 0.0, 0.0)],
)
def test_streamed_first_passage_equals_the_post_scan_form(case, k, x0):
    host, stream = run(k, x0)
    assert np.array_equal(host, stream), f"{case}: post-scan {host} vs streamed {stream}"


def test_streaming_first_passage_needs_a_threshold():
    """The crossing level is the aggregation's one parameter; without it there is nothing to latch on."""
    spec = SPEC.replace("    reduce: streaming\n    parameters: {threshold: {value: 10.0}}", "    reduce: streaming")
    with pytest.raises(ValueError, match="threshold"):
        SimulationExperiment.from_string(spec % {"k": 0.5, "x0": 0.0}).render_code("tvboptim")


def test_streamed_first_passage_keeps_no_trajectory():
    """The point of the fix: the reduction is emitted into the scan carry, not applied to a stack."""
    code = SimulationExperiment.from_string(SPEC % {"k": 0.5, "x0": 0.0}).render_code("tvboptim")
    assert "_fp_idx" in code and "_fp_hit" in code


LAST_SPEC = SPEC.replace(
    """  t_stream:
    label: First passage, streamed
    source: [x]
    aggregation: first_passage
    reduce: streaming
    parameters: {threshold: {value: 10.0}}""",
    """  last_host: {source: [x], aggregation: last}
  last_stream: {source: [x], aggregation: last, reduce: streaming}""",
)


def test_streamed_last_equals_the_post_scan_form():
    """``last`` streams too, because the decision panels need it beside the crossing index.

    A first-passage sweep is only trajectory-free if *every* observation it feeds is: one post-scan ``data[-1]`` left behind puts the whole trajectory back on the host.
    """
    obs = SimulationExperiment.from_string(LAST_SPEC % {"k": 0.5, "x0": 0.0}).run("tvboptim").observations
    host, stream = (np.squeeze(np.asarray(getattr(obs[n], "values", obs[n]))) for n in ("last_host", "last_stream"))
    assert np.array_equal(host, stream), f"post-scan {host} vs streamed {stream}"
