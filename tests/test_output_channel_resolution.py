"""Tests for resolving ``model.output`` to recorded channel indices.

The solver records state variables followed by the auxiliaries that were actually requested, so an output's channel cannot be inferred from its kind: a state output sits *before* the auxiliaries, not after them. Outputs are resolved against that recorded ordering, in declared order, so the emitted channels and the reported ``output_names`` always agree — including when states and auxiliaries are mixed.
"""

import pytest

from tvbo import Dynamics
from tvbo.templates.tvboptim.utils import (
    format_channel_index,
    resolve_model_output_indices,
)

PENDULUM = """
name: PendulumSystem
parameters:
    c: {value: 0.001}
    omega0: {value: 0.01}
    L: {value: 1.0}
state_variables:
    theta:
        initial_value: 1.0
        equation: {rhs: omega}
    omega:
        initial_value: 0.0
        equation: {rhs: -c*omega - omega0**2 * sin(theta)}
derived_variables:
    x: {equation: {rhs: L * sin(theta)}}
    y: {equation: {rhs: -L * cos(theta)}}
output:
%s
"""


def _model(outputs):
    return Dynamics.from_string(PENDULUM % "\n".join(f"    - {o}" for o in outputs))


@pytest.mark.parametrize(
    "outputs, expected_indices",
    [
        # Only requested auxiliaries are recorded, so they follow the two states.
        (["x", "y"], [2, 3]),
        (["y"], [2]),
        # State outputs precede the auxiliaries — they are not offset by n_states.
        (["theta"], [0]),
        (["theta", "omega"], [0, 1]),
        # Mixed outputs keep the declared order rather than the layout order.
        (["x", "y", "theta", "omega"], [2, 3, 0, 1]),
        (["omega", "x"], [1, 2]),
    ],
)
def test_outputs_resolve_to_recorded_channels(outputs, expected_indices):
    indices, names = resolve_model_output_indices(_model(outputs))
    assert indices == expected_indices
    assert names == outputs


def test_unknown_output_is_rejected_at_construction():
    """An output naming no variable is rejected when the model is built."""
    with pytest.raises(ValueError, match="not found in derived_variables or state_variables"):
        _model(["not_a_variable"])


@pytest.mark.parametrize(
    "indices, n_channels, expected",
    [
        ([2], 4, "2"),  # single channel drops the variable dim
        ([2, 3], 4, "2:"),  # contiguous to the end -> open slice
        ([0, 1], 2, "0:"),
        ([0, 1], 4, "0:2"),  # contiguous, bounded
        ([2, 3, 0, 1], 4, "[2, 3, 0, 1]"),  # reordered -> explicit index list
    ],
)
def test_channel_index_expression(indices, n_channels, expected):
    assert format_channel_index(indices, n_channels) == expected


def test_all_auxiliary_output_keeps_previous_slice():
    """The common all-auxiliaries case still emits the ``n_states:`` slice."""
    indices, _ = resolve_model_output_indices(_model(["x", "y"]))
    assert format_channel_index(indices, 4) == "2:"
