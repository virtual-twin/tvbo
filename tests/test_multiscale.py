#
# Module: test_multiscale.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# License: EUPL v2
#
"""Smoke tests for the reservoir multi-scale flattening (``tvbo.multiscale``).

The graph-generator engine (reservoir recurrence, SC loading, weight sampling)
is monkeypatched with tiny synthetic arrays so the lowering logic — Kronecker
assembly of ``W_global`` and the OOM size guard — is exercised without a real
network file.
"""

import numpy as np
import pytest

from tvbo import multiscale
from tvbo.graph_generators import catalog, procedural

R = 3  # macro regions
N = 4  # reservoir units per region


def _minimal_experiment():
    """A minimal Experiment-A dict matching the pattern flatten_reservoir accepts."""
    return {
        "dynamics": {
            "name": "ESN",
            "parameters": {"tau": {"value": 2.0}},
            "state_variables": {"x": {"equation": {"rhs": "(1/tau) * (-x + tanh(W_int @ x))"}}},
        },
        "network": {
            "iri": "dummy://sc",
            "node_template": {
                "subnetwork": {
                    "graph_generator": {
                        "seed": 0,
                        "parameters": {
                            "n_nodes": {"value": N},
                            "weight_distribution": {
                                "distribution": {
                                    "name": "Normal",
                                    "parameters": {"mean": {"value": 0.0}, "std": {"value": 1.0}},
                                }
                            },
                        },
                    },
                    "edges": [],
                }
            },
            "coupling": {
                "Long": {
                    "parameters": {"kappa": {"value": 0.5}},
                    "post_expression": {"rhs": "kappa * gx"},
                }
            },
        },
        "integration": {"method": "Heun", "step_size": 1.0, "duration": 10, "transient_time": 0},
    }


@pytest.fixture
def patched_engine(monkeypatch):
    """Replace the heavy engine calls with deterministic synthetic arrays."""

    def fake_run_generator(name, params, seed=None):
        n = int(params["n_nodes"])
        return {"weights": np.full((n, n), 0.1)}

    # SC with no self-connections
    sc = np.ones((R, R)) - np.eye(R)
    monkeypatch.setattr(catalog, "run_generator", fake_run_generator)
    monkeypatch.setattr(catalog, "load_matrix", lambda source: sc)
    monkeypatch.setattr(procedural, "draw", lambda dist, shape, seed=None, substream=0: np.ones(shape))
    return sc


def test_flatten_reservoir_assembles_global_matrix(patched_engine):
    fr = multiscale.flatten_reservoir(_minimal_experiment())

    assert fr.n_regions == R
    assert fr.n_units == N
    assert fr.tau == 2.0
    assert fr.activation == "tanh"
    # W_global is the dense (R·n)² Kronecker assembly
    assert fr.weights.shape == (R * N, R * N)
    assert np.isfinite(fr.weights).all()
    # flat scalar model wires coupling inside the activation
    assert "tanh(ReservoirInput)" in fr.dynamics["state_variables"]["x"]["equation"]["rhs"]


def test_n_override_shrinks_reservoir(patched_engine):
    fr = multiscale.flatten_reservoir(_minimal_experiment(), n_override=2)
    assert fr.n_units == 2
    assert fr.weights.shape == (R * 2, R * 2)


def test_the_legacy_size_parameter_is_rejected_not_silently_defaulted(patched_engine):
    """`n` was the size parameter until it was replaced by `n_nodes`.

    Ignoring it would leave the size to fall through to the hardcoded 100-unit default:
    a spec asking for 500 units would build 100, run to completion, and produce a
    plausible reservoir with no indication the size was wrong.
    """
    spec = _minimal_experiment()
    params = spec["network"]["node_template"]["subnetwork"]["graph_generator"]["parameters"]
    params["n"] = params.pop("n_nodes")
    with pytest.raises(ValueError, match=r"`n`.*no longer the size parameter"):
        multiscale.flatten_reservoir(spec)


def test_size_guard_raises_before_allocation(patched_engine):
    # R·N = 12 flat nodes; cap below that must refuse to build W_global
    with pytest.raises(ValueError, match="max_flat_nodes"):
        multiscale.flatten_reservoir(_minimal_experiment(), max_flat_nodes=5)


def test_size_guard_disabled_with_none(patched_engine):
    fr = multiscale.flatten_reservoir(_minimal_experiment(), max_flat_nodes=None)
    assert fr.weights.shape == (R * N, R * N)


class TestCrossScaleClock:
    """Flattening leaves one network, so it leaves one clock.

    The subnetwork's `step_size` and the parent's are numbers on two different
    scales that may be written in two different units. Copying one onto the other
    — which is what happened before — is wrong twice over: it integrates the
    reservoir at the macro step, and it does so without converting the unit.
    """

    def _two_scale(self, *, inner_unit=None, inner_step=0.1, outer_unit=None):
        spec = _minimal_experiment()
        subnet = spec["network"]["node_template"]["subnetwork"]
        subnet["integration"] = {"step_size": inner_step}
        if inner_unit:
            subnet["integration"]["time_unit"] = inner_unit
        if outer_unit:
            spec["integration"]["time_unit"] = outer_unit
        return spec

    def test_a_finer_subnetwork_step_imposes_itself(self, patched_engine):
        """0.1 against the parent's 1.0: the reservoir sets the flat step."""
        fr = multiscale.flatten_reservoir(self._two_scale())

        assert fr.integration["step_size"] == 0.1
        assert fr.integration["duration"] == 10

    def test_a_coarser_subnetwork_step_does_not_slow_the_parent(self, patched_engine):
        """The finer of the two wins; a subnetwork cannot under-resolve the macro net."""
        fr = multiscale.flatten_reservoir(self._two_scale(inner_step=5.0))

        assert fr.integration["step_size"] == 1.0

    def test_a_step_in_another_unit_is_converted_not_copied(self, patched_engine):
        """0.1 s into a network clocked in ms is 100 ms — coarser than the parent's 1.0.

        Copied verbatim it would read 0.1 and silently win, integrating the whole
        network 1000x finer than either scale asked for. The number stays plausible,
        which is exactly why nothing catches this downstream.
        """
        fr = multiscale.flatten_reservoir(self._two_scale(inner_unit="s", outer_unit="ms"))

        assert fr.integration["step_size"] == 1.0

    def test_the_conversion_is_exact(self, patched_engine):
        """0.5 ms into a network clocked in s is 0.0005 s, and finer than 1.0."""
        fr = multiscale.flatten_reservoir(self._two_scale(inner_unit="ms", inner_step=0.5, outer_unit="s"))

        assert fr.integration["step_size"] == 0.0005

    def test_an_undeclared_subnetwork_inherits_the_parents_unit(self, patched_engine):
        """Silence means "same clock", not "the `ms` fallback".

        Resolved against the fallback instead, a subnetwork in a network clocked in
        `s` would be converted by 1/1000 for having said nothing at all.
        """
        fr = multiscale.flatten_reservoir(self._two_scale(outer_unit="s"))

        assert fr.integration["step_size"] == 0.1

    def test_a_subnetwork_with_no_clock_of_its_own_changes_nothing(self, patched_engine):
        """The single-scale case has to stay exactly as it was."""
        fr = multiscale.flatten_reservoir(_minimal_experiment())

        assert fr.integration == {"method": "Heun", "step_size": 1.0, "duration": 10, "transient_time": 0}
