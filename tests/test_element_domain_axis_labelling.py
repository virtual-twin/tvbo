"""An ``element_domains`` exploration axis must be readable back by its declared name.

A heterogeneous parameter swept per element is emitted as one dummy scalar grid leaf per element, ``dynamics._<param>_el<i>``, which is packed back into the array before the run. The declared axis, however, is named ``<ref>.<param>[<i>]``, and the grid's own column carries the leaf's keypath — so the declared label has to travel with the axis object at binding time, which is where the coordinate map reads it back.

Without that label every observation of such an exploration fails to place, because keying by value refuses to fall back to a positional reshape (see ``test_exploration_result_labelling.py``). It is a whole axis kind that silently stops working, so both spellings are pinned here.
"""

import re

import pytest

pytest.importorskip("jax")
pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment  # noqa: E402

SPEC = """
label: Element-domain sweep
dynamics:
  name: ElemDecay
  parameters:
    k:
      value: 0.5
      heterogeneous: true
      shape: "(n_nodes,)"
  state_variables:
    x: {equation: {rhs: "-k*x + c_in"}, initial_value: 0.7}
  coupling_inputs: {c_in: {}}
network:
  label: Pair
  number_of_nodes: 2
  nodes: [{id: 0, label: A, dynamics: ElemDecay}, {id: 1, label: B, dynamics: ElemDecay}]
  edges:
    - {source: 0, target: 1, parameters: {weight: {value: 0.3}}, source_var: x_out, target_var: c_in, directed: true}
    - {source: 1, target: 0, parameters: {weight: {value: 0.2}}, source_var: x_out, target_var: c_in, directed: true}
integration: {method: euler, step_size: 0.5, duration: 20.0, unit: ms}
observations:
  x_last:
    label: Final x
    source: [x]
    aggregation: last
explorations:
  per_element:
    label: k per element
    space:
      ElemDecay.k:
        element_domains:
          - {lo: 0.1, hi: 0.9, n: 2}
          - {lo: 0.2, hi: 0.8, n: 2}
"""


@pytest.fixture(scope="module")
def code():
    return SimulationExperiment.from_string(SPEC).render_code("tvboptim")


def test_element_axes_are_declared_under_their_indexed_names(code):
    """The result's axes carry the declared ``<ref>.<param>[<i>]`` labels."""
    assert "ElemDecay.k[0]" in code
    assert "ElemDecay.k[1]" in code


def test_element_axes_sweep_the_dummy_scalar_leaves(code):
    """And the grid itself sweeps ``_k_el0`` / ``_k_el1``, which is why a bridge is needed."""
    assert "_k_el0" in code
    assert "_k_el1" in code


def test_the_emitted_runtime_bridges_leaf_name_to_declared_axis(code):
    """Each element axis binds its dummy leaf under the declared, element-indexed name.

    Asserted on the emitted source rather than a run because the failure it guards is a label that never reaches the coordinate map, and a run would only report it as an unrelated-looking placement error on whichever observation happens to be processed first.
    """
    assert "element_idx" in code
    for _i in (0, 1):
        assert re.search(rf'_k_el{_i} = _ax\(\s*"ElemDecay\.k\[{_i}\]"', code), f"element {_i} bound without its declared label"
