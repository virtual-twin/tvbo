"""Acceptance test: custom conductance-based synapse LEMS emission.

Before the ontology grounding, tvbo could only emit current-based synapses (hardcoded ``extends="baseSynapse"``, exposing ``i``). Deco (2014)'s saturating
NMDA gate (Eqs 3-4) is conductance-based — it exposes ``g`` and extends
``baseConductanceBasedSynapse`` — so it was downgraded to the linear standard
``blockingPlasticSynapse``.

This test authors the *faithful* saturating NMDA as a custom tvbo Dynamics and checks that it emits, grounded in the ingested NeuroML ontology contract, as a
valid conductance-based LEMS ComponentType: the base type, the ``g`` exposure, the two-ODE saturating gate with a spike-driven ``OnEvent``, and the Mg2+ block.
"""

from __future__ import annotations

import os
import re
import sys
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
      coupling_inputs: [iSyn]
      state_variables:
        v:
          equation: {rhs: "(gL * (EL - v) + iSyn) / C"}
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


# The same column, driven: a pulse makes the presynaptic cell fire so the gate is actually exercised.  Structural validity says nothing about whether the emitted model integrates to the right trajectory, which is what this drives.
DRIVEN_COLUMN_YAML = (
    NMDA_COLUMN_YAML.replace(
        """  dynamics:
    LIFCell:""",
        """  dynamics:
    Drive:
      iri: "neuroml:pulseGenerator"
      parameters:
        delay: {value: 50.0, unit: ms}
        duration: {value: 900.0, unit: ms}
        amplitude: {value: 0.6, unit: nA}
    LIFCell:""",
    )
    .replace(
        """  nodes:
    - {id: 0, dynamics: LIFCell}
    - {id: 1, dynamics: LIFCell}
  edges:
    - source: 0
      target: 1""",
        """  nodes:
    - {id: 0, dynamics: Drive}
    - {id: 1, dynamics: LIFCell}
    - {id: 2, dynamics: LIFCell}
  edges:
    - {source: 0, target: 1}
    - source: 1
      target: 2""",
    )
    .replace("  duration: 500.0", "  duration: 1000.0")
)


def _norm(text: str) -> str:
    """Collapse whitespace so assertions survive printer spacing changes."""
    return re.sub(r"\s+", " ", text).strip()


def _time_derivative(component: str, variable: str) -> str:
    """The TimeDerivative expression for *variable*, whitespace-normalised."""
    m = re.search(r'<TimeDerivative variable="%s" value="([^"]+)"' % variable, component)
    assert m, f"no TimeDerivative for {variable} in:\n{component}"
    return _norm(m.group(1))


def _render(tmp_path, yaml_text: str) -> str:
    path = tmp_path / "column.yaml"
    path.write_text(yaml_text)
    return SimulationExperiment.from_file(str(path)).render("lems")


@pytest.fixture(scope="module")
def rendered_lems(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("nmda") / "column.yaml"
    path.write_text(NMDA_COLUMN_YAML)
    exp = SimulationExperiment.from_file(str(path))
    return exp.render("lems")


@pytest.fixture(scope="module")
def nmda_component(rendered_lems) -> str:
    # The synapse ComponentType, named after the dynamics the edge references.
    m = re.search(r'<ComponentType name="NMDA".*?</ComponentType>', rendered_lems, re.S)
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
        """Deco Eqs 3-4: the saturating gate and the spike-driven rise ODE.

        Asserted on the parsed expression rather than an exact string, so a change in the symbolic printer's term order or spacing does not fail a
        run that is still semantically correct.
        """
        decay = _time_derivative(nmda_component, "s")
        assert "-s/tauDecay" in decay, decay
        assert "alpha" in decay and "1 - s" in decay, decay  # the saturating term
        rise = _time_derivative(nmda_component, "x")
        assert "-x/tauRise" in rise, rise

    def test_spike_driven_onevent(self, nmda_component):
        """Presynaptic spike increments the rise variable (no linear proxy)."""
        assert '<OnEvent port="in">' in nmda_component
        m = re.search(r'<OnEvent port="in">(.*?)</OnEvent>', nmda_component, re.S)
        assign = re.search(r'<StateAssignment variable="x" value="([^"]+)"', m.group(1))
        assert assign, m.group(1)
        assert set(_norm(assign.group(1)).replace("+", " ").split()) == {"1", "x"}

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

        ct = model.component_types["NMDA"]
        assert ct.extends == "baseConductanceBasedSynapse"
        chain, name = [], ct.extends
        while name and name in model.component_types:
            chain.append(name)
            name = model.component_types[name].extends
        assert "baseConductanceBasedSynapse" in chain
        assert "baseStandalone" in chain  # resolves all the way to the root


class TestSynapticTransmissionWiring:
    """The synapse must actually reach the postsynaptic membrane equation.

    A valid synapse ComponentType is not enough: the cell has to host it and sum its current, or the synapse is emitted but inert.
    """

    def test_cell_hosts_and_sums_attached_synapses(self, rendered_lems):
        cell = re.search(r'<ComponentType name="LIFCell".*?</ComponentType>', rendered_lems, re.S)
        assert cell, "cell ComponentType was not emitted"
        cell = cell.group(0)
        assert '<Attachments name="synapses" type="basePointCurrent"/>' in cell
        assert re.search(r'<DerivedVariable name="iSyn"[^>]*select="synapses\[\*\]/i"[^>]*reduce="add"', cell)
        assert "iSyn" in _time_derivative(cell, "v")

    def test_connection_targets_the_synapse(self, rendered_lems):
        """The connection must name the synapse Component, not its ComponentType.

        A ``synapse=`` reference resolves against components; pointing it at the bare ComponentType left jNeuroML with "No component found: syn_edge0".
        """
        assert re.search(r'synapse="NMDA_inst"[^>]*destination="synapses"', rendered_lems)

    def test_synapse_component_is_instantiated_with_its_values(self, rendered_lems):
        """The ComponentType alone carries no values — the instance must exist.

        Without it the synapse's parameters (the Deco gate constants, and the gbase/erev inherited from the base type) were emitted nowhere at all.
        """
        m = re.search(r"<Component id=\"NMDA_inst\" type=\"NMDA\"[^>]*/>", rendered_lems)
        assert m, "custom synapse ComponentType was never instantiated"
        inst = m.group(0)
        for attr, value in (
            ("tauDecay", "100"),
            ("tauRise", "2"),
            ("alpha", "0.5"),
            ("mgFactor", "0.28"),
            ("scalingVolt", "16.129"),
            ("gbase", "0.2"),  # inherited from baseConductanceBasedSynapse
            ("erev", "0"),
        ):
            assert re.search(r'%s="%s' % (attr, re.escape(value)), inst), f"{attr} missing from {inst}"

    def test_time_derivatives_are_dimensionally_consistent(self, nmda_component):
        """Gate equations with dimensioned time constants must not be / SEC.

        ``tauDecay``/``alpha`` already carry time dimensions, so the RHS is a rate; dividing by SEC again made it per-time-squared and jNeuroML
        rejected the model outright.
        """
        assert "SEC" not in nmda_component
        assert _time_derivative(nmda_component, "s").endswith("alpha*x*(1 - s)")


class TestUnknownBaseTypeWarns:
    """A base type tvbo cannot resolve must not fail silently.

    Emitting against an unknown type yields a ComponentType extending something
    NeuroML has never heard of; without this warning the failure only surfaces much later inside jNeuroML, with nothing pointing at the bad reference.
    """

    @staticmethod
    def _resolve(ref):
        from tvbo.adapters.neuroml import _base_type_meta

        # The contract lookup is cached per name, so a warning fires only once per process for a given type.
        _base_type_meta.cache_clear()
        return _base_type_meta(ref)

    def test_typo_warns_and_suggests_the_intended_type(self):
        with pytest.warns(UserWarning, match="Unknown NeuroML base type") as rec:
            meta = self._resolve("extends:baseConductnceBasedSynapse")
        assert meta == {}
        assert "baseConductanceBasedSynapse" in str(rec[0].message)

    def test_unrecognisable_name_still_warns(self):
        with pytest.warns(UserWarning, match="totallyBogusType"):
            self._resolve("extends:totallyBogusType")

    def test_known_base_types_are_silent(self):
        import warnings as _w

        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            for ref in ("baseConductanceBasedSynapse", "baseSynapse", "baseCellMembPot"):
                self._resolve(ref)
        assert [str(c.message) for c in caught] == []


# A synapse with no base-type IRI keeps the historical current-based default.
CURRENT_SYNAPSE_YAML = NMDA_COLUMN_YAML.replace(
    """    NMDA:
      iri: "extends:baseConductanceBasedSynapse\"""",
    """    NMDA:""",
).replace(
    """      description: Deco (2014) Eqs 3-4 saturating NMDA gate with Mg2+ voltage block.
""",
    "",
)


class TestCurrentBasedSynapseUnchanged:
    """The default path must not shift when no base type is declared.

    Nothing in the tvbo database exercises the synapse emitter, so this pins the pre-existing current-based behaviour that the ontology grounding defaults to.
    """

    @pytest.fixture(scope="class")
    def component(self, tmp_path_factory) -> str:
        xml = _render(tmp_path_factory.mktemp("cur"), CURRENT_SYNAPSE_YAML)
        m = re.search(r'<ComponentType name="NMDA".*?</ComponentType>', xml, re.S)
        assert m, "synapse ComponentType was not emitted"
        return m.group(0)

    def test_defaults_to_base_synapse(self, component):
        assert 'extends="baseSynapse"' in component

    def test_declares_own_parameters(self, component):
        """baseSynapse inherits no parameters, so none are skipped."""
        assert '<Parameter name="gbase"' in component
        assert '<Parameter name="erev"' in component

    def test_exposes_current_and_requires_voltage(self, component):
        assert re.search(r'<DerivedVariable name="i"[^>]*exposure="i"', component)
        # v is not inherited from baseSynapse, so it is declared explicitly.
        assert '<InstanceRequirement name="v" type="voltage"/>' in component


@pytest.mark.backend_neuroml
class TestDrivenColumnBehaviour:
    """Run the emitted model and check the gate behaves like Deco Eqs 3-4.

    Everything above asserts on the XML.  A model can be perfectly valid XML, resolve against the core types, and still integrate to nothing — which is
    exactly what happened before: the synapse ComponentType was never instantiated and the connection pointed at a type, so jNeuroML refused to
    build the network at all.  This class is the guard against that class of failure: it drives the presynaptic cell and reads the trajectory back.
    """

    @pytest.fixture(scope="class")
    def run_output(self, tmp_path_factory):
        pytest.importorskip("pyneuroml")
        import glob
        import subprocess

        import numpy as np

        tmp = tmp_path_factory.mktemp("driven")
        (tmp / "column.yaml").write_text(DRIVEN_COLUMN_YAML)
        xml = SimulationExperiment.from_file(str(tmp / "column.yaml")).render("lems")

        # tvbo emits OutputColumns for population state variables only; the synapse's own gate is recorded here by path so the trajectory can be asserted on (emitting these declaratively is a separate gap).
        gate_cols = "".join(
            '      <OutputColumn id="%s" quantity="LIFCell_pop[1]/NMDA_inst/%s"/>\n' % (n, n) for n in ("s", "x")
        )
        xml = xml.replace("</OutputFile>", gate_cols + "    </OutputFile>", 1)
        (tmp / "driven.xml").write_text(xml)
        (tmp / "results").mkdir()

        exe = os.path.join(os.path.dirname(sys.executable), "pynml")
        if not os.path.exists(exe):
            pytest.skip("pynml executable not available")
        proc = subprocess.run([exe, "driven.xml", "-nogui"], cwd=str(tmp), capture_output=True, text=True, timeout=900)
        dat = glob.glob(str(tmp / "results" / "*.dat"))
        if not dat:
            pytest.skip(f"jNeuroML produced no output (exit {proc.returncode}):\n{proc.stdout[-2000:]}")
        return np.loadtxt(dat[0])

    def test_presynaptic_cell_fires(self, run_output):
        """The pulse must actually drive spiking, else the gate is never tested."""
        pre = run_output[:, 1]
        crossings = int(((pre[:-1] < -0.051) & (pre[1:] >= -0.051)).sum())
        assert crossings > 10, f"presynaptic cell did not spike ({crossings} crossings)"

    def test_gate_saturates_strictly_below_one(self, run_output):
        """Deco's alpha*x*(1-s) term is what bounds s — a linear gate would not.

        The bound is the whole reason the faithful synapse is needed instead of the linear ``blockingPlasticSynapse`` downgrade.
        """
        s = run_output[:, 3]
        assert s.min() >= 0.0, f"gate went negative: {s.min()}"
        assert s.max() < 1.0, f"gate breached saturation: {s.max()}"
        assert s.max() > 0.5, f"gate never charged, so saturation is untested: {s.max()}"

    def test_synapse_reaches_postsynaptic_membrane(self, run_output):
        """A synapse that renders but is not summed into dv/dt leaves v at rest."""
        post = run_output[:, 2]
        assert post.max() > post[0], "postsynaptic voltage never moved"


# A Poisson background: the component names its synapse by id (<ComponentReference name="synapse" type="baseSynapse"/>) rather than nesting it, and that synapse hangs off no node and no edge.
POISSON_BACKGROUND_YAML = DRIVEN_COLUMN_YAML.replace(
    """    Drive:
      iri: "neuroml:pulseGenerator"
      parameters:
        delay: {value: 50.0, unit: ms}
        duration: {value: 900.0, unit: ms}
        amplitude: {value: 0.6, unit: nA}""",
    """    Drive:
      iri: "neuroml:poissonFiringSynapse"
      parameters:
        averageRate: {value: 2400.0, unit: Hz}
        synapse: {value: AMPA_bg}
        spikeTarget: {value: "./AMPA_bg"}
    AMPA_bg:
      iri: "neuroml:expOneSynapse"
      parameters:
        gbase: {value: 3.37, unit: nS}
        erev: {value: 0.0, unit: mV}
        tauDecay: {value: 2.0, unit: ms}""",
)


class TestComponentReferences:
    """A referenced component must be emitted, or the reference dangles.

    ``poissonFiringSynapse`` names its synapse by id.  That target is attached to neither a node nor an edge, so nothing in the network builder would
    otherwise emit it and jNeuroML failed with "No component found".
    """

    @pytest.fixture(scope="class")
    def rendered(self, tmp_path_factory) -> str:
        return _render(tmp_path_factory.mktemp("poisson"), POISSON_BACKGROUND_YAML)

    def test_contract_exposes_the_reference(self):
        """Ingested from <ComponentReference>, not hardcoded per type."""
        from tvbo.adapters.neuroml import _base_type_meta

        refs = _base_type_meta("poissonFiringSynapse").get("component_references")
        assert refs == {"synapse": "baseSynapse"}

    def test_referenced_component_is_emitted(self, rendered):
        """A library Dynamics reached only by reference still has to be emitted."""
        assert re.search(r'<expOneSynapse id="AMPA_bg"[^>]*gbase="3.37', rendered), rendered

    def test_reference_resolves_to_it(self, rendered):
        assert re.search(r'<poissonFiringSynapse[^>]*synapse="AMPA_bg"', rendered)

    def test_parameterisation_of_a_standard_type_is_not_a_componenttype(self, rendered):
        """A Dynamics with no equations of its own only parameterises the type."""
        assert '<ComponentType name="AMPA_bg"' not in rendered

    def test_unsatisfied_reference_warns(self, tmp_path):
        """Silence here would surface only as a simulator error much later."""
        broken = POISSON_BACKGROUND_YAML.replace("        synapse: {value: AMPA_bg}\n", "")
        with pytest.warns(UserWarning, match="requires a 'synapse' reference"):
            _render(tmp_path, broken)

    def test_reference_to_unknown_dynamics_warns(self, tmp_path):
        broken = POISSON_BACKGROUND_YAML.replace("{value: AMPA_bg}", "{value: NoSuchSynapse}")
        with pytest.warns(UserWarning, match="not in the network's dynamics library"):
            _render(tmp_path, broken)


class TestInputFanOut:
    """One component definition per input, one explicitInput per target.

    A connectivity rule on an input edge attaches an independent copy of the input to every target cell.  Those copies share one component definition —
    emitting it per target would repeat the same id N times.
    """

    @pytest.fixture(scope="class")
    def rendered(self, tmp_path_factory) -> str:
        yaml_text = DRIVEN_COLUMN_YAML.replace(
            "- {id: 1, dynamics: LIFCell}", "- {id: 1, dynamics: LIFCell, size: 4}"
        ).replace("- {source: 0, target: 1}", "- {source: 0, target: 1, connectivity: all_to_all}")
        return _render(tmp_path_factory.mktemp("fanout"), yaml_text)

    def test_component_declared_once(self, rendered):
        assert rendered.count('<pulseGenerator id="Drive"') == 1

    def test_every_target_gets_an_input(self, rendered):
        assert rendered.count("<explicitInput") == 4


# A *standard-types* network (iri: neuroml:*) driving the same input onto several cells. This exercises the standard-NeuroML builder, not the custom-LEMS one, so it pins the input de-duplication the shared small-scale lowering brings to that path — the parity fix from routing both builders through one core.
STD_SHARED_INPUT_YAML = """
label: "Standard-types column with a shared input"
network:
  dynamics:
    FNCell:
      iri: "neuroml:fitzHughNagumoCell"
      parameters: {I: {value: 0.0}}
      state_variables:
        V: {equation: {rhs: "V - V**3/3 - W + I"}, initial_value: 0.0, variable_of_interest: true}
        W: {equation: {rhs: "0.08*(V + 0.7 - 0.8*W)"}, initial_value: 0.0}
    Drive:
      iri: "neuroml:pulseGenerator"
      parameters:
        delay: {value: 50.0, unit: ms}
        duration: {value: 100.0, unit: ms}
        amplitude: {value: 0.5, unit: nA}
  nodes:
    - {id: 0, dynamics: Drive}
    - {id: 1, dynamics: Drive}
    - {id: 2, dynamics: FNCell}
    - {id: 3, dynamics: FNCell}
  edges:
    - {source: 0, target: 2}
    - {source: 1, target: 3}
integration: {method: euler, step_size: 0.01, duration: 10.0, time_scale: s}
"""


class TestStandardPathInputDedup:
    """The standard-NeuroML builder must de-duplicate inputs like the custom one.

    Two input nodes with identical parameters share one component; two with differing parameters each get a unique id. Before the shared lowering, the
    standard-types path used a bare ``safe_id(name)`` with no de-duplication, so a repeated input emitted the same id twice (a duplicate component) and a
    differing one silently collided — the fixes that already lived only in the custom-LEMS builder. Routing both builders through the shared core closes it.
    """

    def test_identical_inputs_share_one_component(self, tmp_path):
        rendered = _render(tmp_path, STD_SHARED_INPUT_YAML)
        assert "<population" in rendered  # confirms the standard-types builder ran
        assert rendered.count('<pulseGenerator id="Drive"') == 1
        assert rendered.count("<explicitInput") == 2

    def test_differing_inputs_get_unique_ids(self, tmp_path):
        # Give node 1's Drive a different amplitude so it cannot share a component.
        yaml_text = STD_SHARED_INPUT_YAML.replace(
            "- {id: 1, dynamics: Drive}",
            "- {id: 1, dynamics: Drive, parameters: {amplitude: {value: 0.9, unit: nA}}}",
        )
        with pytest.warns(UserWarning, match="more than one set of parameters"):
            rendered = _render(tmp_path, yaml_text)
        ids = sorted(set(re.findall(r'<pulseGenerator id="([^"]+)"', rendered)))
        assert ids == ["Drive", "Drive_2"], ids
        assert rendered.count("<explicitInput") == 2


class TestDeclaredRefractoryPeriod:
    """A model's own refractory period must not be overwritten by the default.

    The refractory regime reads `refract`, so the emitter defaults it in when the model is silent.  When the model DOES declare it, emitting both put two
    `refract` attributes on the component — and the synthesised "0 ms" came last, so a declared refractory period was silently discarded.
    """

    @pytest.fixture(scope="class")
    def rendered(self, tmp_path_factory) -> str:
        yaml_text = DRIVEN_COLUMN_YAML.replace(
            "        v0: {value: -70.0, unit: mV}",
            "        v0: {value: -70.0, unit: mV}\n        refract: {value: 2.0, unit: ms}",
        )
        return _render(tmp_path_factory.mktemp("refrac"), yaml_text)

    def test_declared_once(self, rendered):
        assert rendered.count('<Parameter name="refract"') == 1

    def test_declared_value_survives(self, rendered):
        m = re.search(r'<Component id="LIFCell_inst"[^>]*/>', rendered)
        assert m, rendered
        assert m.group(0).count("refract=") == 1, m.group(0)
        assert 'refract="2.0 ms"' in m.group(0), m.group(0)


# A network-mode column written NUMERICALLY: no physical units anywhere, so the
# TimeDerivative RHS is a pure number in model-time units.
NUMERIC_COLUMN_YAML = """
label: "Numeric (unitless) network-mode column"
network:
  dynamics:
    NumericCell:
      parameters:
        tau: {value: 20.0}
        drive: {value: 1.2}
      coupling_inputs: [iSyn]
      state_variables:
        u:
          equation: {rhs: "(-u + drive + iSyn) / tau"}
          initial_value: 0.0
      events:
        spike:
          condition: {rhs: "u > 1.0"}
          affect: {rhs: "u = 0.0"}
    NumericSyn:
      parameters:
        gsyn: {value: 0.1}
        tausyn: {value: 5.0}
      coupling_inputs: [v]
      state_variables:
        r:
          equation: {rhs: "-r / tausyn"}
          initial_value: 0.0
      derived_variables:
        i:
          equation: {rhs: "gsyn * r"}
      events:
        spike_in:
          affect: {rhs: "r = r + 1"}
  nodes:
    - {id: 0, dynamics: NumericCell}
    - {id: 1, dynamics: NumericCell}
  edges:
    - {source: 0, target: 1, dynamics: NumericSyn, parameters: {weight: {value: 1.0}}}
integration:
  method: euler
  step_size: 0.02
  duration: 100.0
  time_scale: ms
"""


class TestNetworkModeTimeScaling:
    """`/ SEC` must follow the equations' dimensions, in network mode too.

    SEC converts a TimeDerivative written in model-time units into SI. Applying it to equations that already carry dimensioned time constants makes the
    derivative per-time-squared and jNeuroML rejects the model; omitting it from equations written as pure numbers integrates 1000x too fast. Network-mode
    ComponentTypes used to apply it unconditionally, and nothing in the shipped database exercises that path — every database NeuroML experiment has a
    `network`, but none defines `network.dynamics` — so these are the only tests holding the rule for cells and synapses alike.
    """

    @pytest.fixture(scope="class")
    def numeric(self, tmp_path_factory) -> str:
        return _render(tmp_path_factory.mktemp("numeric"), NUMERIC_COLUMN_YAML)

    @staticmethod
    def _component(xml: str, name: str) -> str:
        m = re.search(r'<ComponentType name="%s".*?</ComponentType>' % name, xml, re.S)
        assert m, f"{name} was not emitted in:\n{xml[:2000]}"
        return m.group(0)

    def test_dimensioned_cell_is_not_rescaled(self, rendered_lems):
        """mV/nS/nF/ms constants already make the RHS a rate."""
        cell = self._component(rendered_lems, "LIFCell")
        assert "SEC" not in cell, cell

    def test_dimensioned_synapse_is_not_rescaled(self, nmda_component):
        """tauDecay/alpha carry time, so the gate equation is a rate already."""
        assert "SEC" not in nmda_component, nmda_component

    def test_numeric_cell_is_rescaled(self, numeric):
        """A unitless model carries no time scale of its own, so SEC supplies it."""
        cell = self._component(numeric, "NumericCell")
        assert '<Constant name="SEC" dimension="time" value="1ms"/>' in cell, cell
        assert _time_derivative(cell, "u").endswith("/ SEC"), _time_derivative(cell, "u")

    def test_numeric_synapse_is_rescaled(self, numeric):
        """The same rule has to hold for the synapse ComponentType path."""
        syn = self._component(numeric, "NumericSyn")
        assert '<Constant name="SEC" dimension="time" value="1ms"/>' in syn, syn
        assert _time_derivative(syn, "r").endswith("/ SEC"), _time_derivative(syn, "r")


class TestConnectionWeightReachesCurrent:
    """The per-connection weight must scale the emitted current, jLEMS-validly.

    NeuroML's gradedSynapse references `weight` in its continuous current, but only after re-declaring the Property locally — jLEMS does not surface the
    inherited baseSynapse Property to the DerivedVariable checker, so a custom synapse that omits the re-declaration fails to load with "No such variable
    in map: weight". Deco's w_+ = 1.4 multiplies the summed gating variable, i.e.
    it sits OUTSIDE each connection's saturating gate, so it scales the current rather than the spike increment.
    """

    def test_weight_property_is_declared(self, nmda_component):
        assert '<Property name="weight" dimension="none" defaultValue="1"/>' in nmda_component

    def test_weight_scales_the_current(self, nmda_component):
        i = re.search(r'<DerivedVariable name="i"[^>]*value="([^"]+)"', nmda_component)
        assert i, nmda_component
        assert i.group(1).startswith("weight * "), i.group(1)

    def test_weighted_synapse_loads_in_jlems(self, tmp_path):
        """A model whose custom synapse references weight must actually load."""
        pytest.importorskip("pyneuroml")
        import glob
        import subprocess

        yaml_text = DRIVEN_COLUMN_YAML.replace(
            "      dynamics: NMDA\n      parameters:\n        weight: {value: 1.0}",
            "      dynamics: NMDA\n      parameters:\n        weight: {value: 1.4}",
        )
        (tmp_path / "w.yaml").write_text(yaml_text)
        xml = SimulationExperiment.from_file(str(tmp_path / "w.yaml")).render("lems")
        assert "weight * (" in xml
        (tmp_path / "w.xml").write_text(xml)
        (tmp_path / "results").mkdir()
        exe = os.path.join(os.path.dirname(sys.executable), "pynml")
        if not os.path.exists(exe):
            pytest.skip("pynml not available")
        p = subprocess.run([exe, "w.xml", "-nogui"], cwd=str(tmp_path), capture_output=True, text=True, timeout=900)
        assert "No such variable in map: weight" not in (p.stdout + p.stderr), p.stdout[-1500:]
        assert glob.glob(str(tmp_path / "results" / "*.dat")), f"no output (exit {p.returncode}):\n{p.stdout[-1500:]}"
