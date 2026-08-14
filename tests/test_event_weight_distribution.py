"""A stimulus Event's `weight_distribution` is sampled, not silently dropped.

`_resolve_events` lowers a declared distribution into the per-node `weights` array the
codegen reads. It used to do that through `graph_generators.builtins`, a module deleted
when the per-generator materialisers went — behind a bare `except`, so the failure was
invisible: every stimulus that declared a distribution got no weighting at all, and the
simulation ran on happily with the wrong drive. It now goes through the same
printer-backed sampler a generator's `sample` step uses.
"""

import numpy as np
import pytest

from tvbo import SimulationExperiment

EXP = {
    "id": 1,
    "label": "stimulus weighting fixture",
    "dynamics": {
        "name": "MiniOsc",
        "system_type": "continuous",
        "output": ["x"],
        "parameters": {"a": {"value": 1.0}},
        "state_variables": {"x": {"equation": {"rhs": "-a * x + drive"}, "initial_value": 0.0}},
    },
    "network": {"number_of_nodes": 6},
    "integration": {"method": "heun", "step_size": 0.1, "duration": 1.0, "transient_time": 0.0, "unit": "s"},
    "events": {
        "drive": {
            "event_type": "stimulus",
            "target_variable": "drive",
            "equation": {"rhs": "Piecewise((1.0, t >= 0.2), (0.0, True))"},
            "weight_distribution": {
                "name": "Normal",
                "parameters": {"mean": {"value": 1.0}, "std": {"value": 0.1}},
                "seed": 3,
            },
        }
    },
}


def _weights(spec=EXP):
    """Resolve the experiment, then read the event's lowered weighting.

    `configure` is the boundary that lowers declarative Event fields into the ones
    codegen reads; plain construction leaves them declarative.
    """
    import copy

    exp = SimulationExperiment(**copy.deepcopy(spec))
    exp.configure()
    return list(getattr(exp.events["drive"], "weights", None) or [])


def test_a_declared_distribution_becomes_a_weights_array():
    """The regression: this was an empty list for as long as the import was broken."""
    weights = _weights()
    assert len(weights) == 6
    assert all(isinstance(w, float) for w in weights)


def test_the_weighting_is_not_uniform():
    """A dropped distribution leaves flat/absent weights, which is the failure to catch."""
    assert len(set(_weights())) > 1


def test_sampling_is_seeded_and_reproducible():
    np.testing.assert_array_equal(_weights(), _weights())


def test_the_seed_is_read_from_the_distribution():
    import copy

    other = copy.deepcopy(EXP)
    other["events"]["drive"]["weight_distribution"]["seed"] = 4
    assert not np.array_equal(_weights(), _weights(other))


def test_explicit_weights_are_not_overwritten_by_a_distribution():
    """A recipe that states its weighting must keep it; the draw only fills a gap."""
    import copy

    explicit = copy.deepcopy(EXP)
    explicit["events"]["drive"]["weights"] = [0.5] * 6
    assert _weights(explicit) == [0.5] * 6


def test_an_omitted_parameter_uses_the_families_standard_form():
    """`Normal` without a std is Normal(mean, 1) — the family's own standard form."""
    import copy

    partial = copy.deepcopy(EXP)
    partial["events"]["drive"]["weight_distribution"]["parameters"].pop("std")
    assert len(_weights(partial)) == 6


def test_a_misspelled_parameter_names_the_event_rather_than_a_generator_step():
    """A typo must fail — the value it carried would be dropped and the parameter it meant
    would silently take its standard form, drawing from a different distribution.

    The sampler's own message talks about generator steps, which says nothing about which
    stimulus is at fault, so `_resolve_events` re-raises with the event named.
    """
    import copy

    broken = copy.deepcopy(EXP)
    broken["events"]["drive"]["weight_distribution"]["parameters"]["stdev"] = {"value": 0.2}
    with pytest.raises(ValueError, match=r"event 'drive'.*weight_distribution"):
        _weights(broken)


def test_an_unsupported_family_names_the_event_too():
    import copy

    broken = copy.deepcopy(EXP)
    broken["events"]["drive"]["weight_distribution"]["name"] = "Cauchy"
    with pytest.raises(ValueError, match=r"event 'drive'"):
        _weights(broken)
