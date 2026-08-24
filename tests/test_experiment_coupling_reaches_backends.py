"""A coupling the network declares must reach every backend that renders one.

Coupling belongs to the network, so ``SimulationExperiment`` has no coupling slot, and reading the removed one back returns ``None`` rather than raising. A template that asks for it therefore emits code with no coupling at all, and the TVB export substitutes ``Linear`` for whatever was declared. These tests pin the resolution to the place coupling actually lives.
"""

from __future__ import annotations

import pytest

from tvbo import Coupling, Dynamics, Network, SimulationExperiment
from tvbo.templates.base.utils import experiment_coupling


def _experiment(with_coupling: bool) -> SimulationExperiment:
    exp = SimulationExperiment(dynamics=Dynamics.from_db("Generic2dOscillator"))
    exp.network = Network.from_db("DesikanKilliany")
    if with_coupling:
        exp.network.coupling["Sigmoidal"] = Coupling.from_db("Sigmoidal")
    return exp


def test_the_resolver_reads_the_network():
    assert experiment_coupling(_experiment(False)) is None
    assert experiment_coupling(_experiment(True)).name == "Sigmoidal"


def test_a_bare_experiment_resolves_to_no_coupling():
    """No network at all is the degenerate case, and it must answer rather than raise."""
    assert experiment_coupling(SimulationExperiment(dynamics=Dynamics.from_db("Generic2dOscillator"))) is None


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
