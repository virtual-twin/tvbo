"""A coupling the network declares must reach every backend that renders one.

Coupling belongs to the network, so ``SimulationExperiment`` has no coupling slot and ``experiment.coupling`` is a read of ``network.coupling`` — the first member, which is what a backend expressing a single coupling renders. The collection arrives as a plain dict when the experiment is constructed and as a ``JsonObj`` once the slot is assigned, so the read goes through ``keyed_items``; reading the assigned shape with ``.values()`` raised, and a backend that lost its coupling emits code with none at all while the TVB export substitutes ``Linear``.
"""

from __future__ import annotations

import pytest

from tvbo import Coupling, Dynamics, Network, SimulationExperiment


def _experiment(with_coupling: bool) -> SimulationExperiment:
    exp = SimulationExperiment(dynamics=Dynamics.from_db("Generic2dOscillator"))
    exp.network = Network.from_db("DesikanKilliany")
    if with_coupling:
        exp.network.coupling["Sigmoidal"] = Coupling.from_db("Sigmoidal")
    return exp


def test_the_property_reads_the_network():
    assert _experiment(False).coupling is None
    assert _experiment(True).coupling.name == "Sigmoidal"


def test_an_assigned_coupling_collection_is_still_readable():
    """Assigning the slot rewraps it as a ``JsonObj``, which has no ``.values()``."""
    exp = _experiment(True)
    exp.network.coupling = exp.network.coupling

    assert exp.coupling.name == "Sigmoidal"


def test_a_bare_experiment_resolves_to_no_coupling():
    """No network at all is the degenerate case, and it must answer rather than raise."""
    assert SimulationExperiment(dynamics=Dynamics.from_db("Generic2dOscillator")).coupling is None


def test_the_tvb_export_names_the_declared_coupling():
    """The fallback here is ``Linear``, so a lost coupling reads as a deliberate choice."""
    code = _experiment(True).render("tvb")

    assert "coupling=Sigmoidal(**coupling_kwargs)" in code
    assert "coupling=Linear(**coupling_kwargs)" not in code


def test_the_jax_export_renders_a_declared_coupling():
    with_coupling = _experiment(True).render("jax")
    without = _experiment(False).render("jax")

    assert "state.parameters.coupling" in with_coupling
    assert "state.parameters.coupling" not in without


@pytest.mark.parametrize("fmt", ["jax", "tvb"])
def test_an_experiment_without_coupling_still_renders(fmt):
    assert _experiment(False).render(fmt)
