"""Acceptance test: custom conductance-based synapse LEMS emission.

Before the ontology grounding, tvbo could only emit current-based synapses
(hardcoded ``extends="baseSynapse"``, exposing ``i``). Deco (2014)'s saturating
NMDA gate (Eqs 3-4) is conductance-based — it exposes ``g`` and extends
``baseConductanceBasedSynapse`` — so it was downgraded to the linear standard
``blockingPlasticSynapse``.

This test authors the *faithful* saturating NMDA as a custom tvbo Dynamics and
checks that it emits, grounded in the ingested NeuroML ontology contract, as a
valid conductance-based LEMS ComponentType: the base type, the ``g`` exposure,
the two-ODE saturating gate with a spike-driven ``OnEvent``, and the Mg2+ block.
"""

from __future__ import annotations

import os
import re
import zipfile

import pytest

from tvbo.classes.experiment import SimulationExperiment

# A two-neuron column whose single edge is the custom saturating NMDA synapse.
# The cell is a minimal LIF; the synapse carries the faithful Deco Eqs 3-4.
NMDA_COLUMN_YAML = """
label: "Custom saturating-NMDA column"
network:
  dynamics:
    LIFCell:
      iri: "extends:baseCellMembPot"
      parameters:
        C: {value: 0.5, unit: nF}
        gL: {value: 25.0, unit: nS}
        EL: {value: -70.0, unit: mV}
        thresh: {value: -50.0, unit: mV}
        reset: {value: -55.0, unit: mV}
        v0: {value: -70.0, unit: mV}
      state_variables:
        v:
          equation: {rhs: "(gL * (EL - v)) / C"}
          initial_value: -70.0
          unit: mV
      events:
        spike:
          condition: {rhs: "v > thresh"}
          affect: {rhs: "v = reset"}
    NMDA:
      iri: "extends:baseConductanceBasedSynapse"
      description: Deco (2014) Eqs 3-4 saturating NMDA gate with Mg2+ voltage block.
      parameters:
        gbase: {value: 0.20, unit: nS}
        erev: {value: 0.0, unit: mV}
        tauDecay: {value: 100.0, unit: ms}
        tauRise: {value: 2.0, unit: ms}
        alpha: {value: 0.5, unit: per_ms}
        mgFactor: {value: 0.28, unit: dimensionless}
        scalingVolt: {value: 16.129, unit: mV}
      coupling_inputs: [v]
      state_variables:
        s:
          equation: {rhs: "-s / tauDecay + alpha * x * (1 - s)"}
          initial_value: 0.0
          unit: dimensionless
        x:
          equation: {rhs: "-x / tauRise"}
          initial_value: 0.0
          unit: dimensionless
      derived_variables:
        block:
          equation: {rhs: "1 / (1 + mgFactor * exp(-v / scalingVolt))"}
          unit: dimensionless
        g:
          equation: {rhs: "gbase * s * block"}
          unit: nS
        i:
          equation: {rhs: "g * (erev - v)"}
          unit: nA
      events:
        spike_in:
          affect: {rhs: "x = x + 1"}
  nodes:
    - {id: 0, dynamics: LIFCell}
    - {id: 1, dynamics: LIFCell}
  edges:
    - source: 0
      target: 1
      dynamics: NMDA
      parameters:
        weight: {value: 1.0}
integration:
  method: euler
  step_size: 0.02
  duration: 500.0
  time_scale: ms
"""


@pytest.fixture(scope="module")
def rendered_lems(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("nmda") / "column.yaml"
    path.write_text(NMDA_COLUMN_YAML)
    exp = SimulationExperiment.from_file(str(path))
    return exp.render("lems")


@pytest.fixture(scope="module")
def nmda_component(rendered_lems) -> str:
    # The edge's synapse ComponentType (id derived from the edge).
    m = re.search(r'<ComponentType name="syn_edge0".*?</ComponentType>', rendered_lems, re.S)
    assert m, "custom synapse ComponentType was not emitted"
    return m.group(0)


class TestCustomConductanceSynapseEmission:
    def test_extends_conductance_based_synapse(self, nmda_component):
        """Grounded in the ontology, not the hardcoded current-based baseSynapse."""
        assert 'extends="baseConductanceBasedSynapse"' in nmda_component

    def test_exposes_conductance(self, nmda_component):
        assert re.search(r'<DerivedVariable name="g"[^>]*exposure="g"', nmda_component)
        assert re.search(r'<DerivedVariable name="i"[^>]*exposure="i"', nmda_component)

    def test_inherited_parameters_not_redeclared(self, nmda_component):
        """gbase/erev come from baseConductanceBasedSynapse; v from the chain."""
        assert '<Parameter name="gbase"' not in nmda_component
        assert '<Parameter name="erev"' not in nmda_component
        assert "<InstanceRequirement" not in nmda_component

    def test_saturating_two_ode_gate(self, nmda_component):
        """Deco Eqs 3-4: the saturating alpha*x*(1-s) term and the rise ODE."""
        assert "alpha*x*(1 - s)" in nmda_component
        assert re.search(r'<TimeDerivative variable="x" value="\(-x/tauRise\)', nmda_component)

    def test_spike_driven_onevent(self, nmda_component):
        """Presynaptic spike increments the rise variable (no linear proxy)."""
        assert "<OnEvent port=\"in\">" in nmda_component
        assert re.search(r'<StateAssignment variable="x" value="1 \+ x"', nmda_component)

    def test_mg_voltage_block(self, nmda_component):
        assert "mgFactor*exp(-v/scalingVolt)" in nmda_component

    def test_pylems_valid_against_core_types(self, rendered_lems, tmp_path):
        """The synapse resolves against the real NeuroML baseConductanceBasedSynapse."""
        pytest.importorskip("lems")
        pyneuroml = pytest.importorskip("pyneuroml")
        from lems.model.model import Model

        import glob

        libdir = os.path.join(os.path.dirname(pyneuroml.__file__), "lib")
        jars = sorted(glob.glob(os.path.join(libdir, "jNeuroML-*-jar-with-dependencies.jar")))
        if not jars:
            pytest.skip("jNeuroML jar not available")
        with zipfile.ZipFile(jars[-1]) as z:
            members = [n for n in z.namelist() if n.startswith("NeuroML2CoreTypes/") and n.endswith(".xml")]
            z.extractall(tmp_path, members=members)
        core = str(tmp_path / "NeuroML2CoreTypes")
        sim = os.path.join(core, "_column_sim.xml")
        with open(sim, "w") as fh:
            fh.write(rendered_lems)

        model = Model(include_includes=True)
        model.add_include_directory(core)
        model.import_from_file(sim)

        ct = model.component_types["syn_edge0"]
        assert ct.extends == "baseConductanceBasedSynapse"
        chain, name = [], ct.extends
        while name and name in model.component_types:
            chain.append(name)
            name = model.component_types[name].extends
        assert "baseConductanceBasedSynapse" in chain
        assert "baseStandalone" in chain  # resolves all the way to the root
