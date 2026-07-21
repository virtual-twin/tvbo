"""Native Brian2 backend — the Deco 2014 spiking column.

The adapter lowers the fully-connected 160E+40I conductance-based LIF column onto
Brian2 via O(N) summed "hub" population sums (not O(N^2) `Synapses`), and
reproduces the paper's Table-2 spontaneous rates (E 2.92 Hz / I 7.54 Hz). These
tests pin: (1) `render("brian2")` emits a valid, self-contained, runnable script;
(2) `run(format="brian2")` reproduces the Table-2 regime; (3) the rendered script
and the in-process run agree at a fixed seed (render == run).

The column is declared inline (no external recipe) so the test is self-contained.
It is the same structure as the manuscript recipe's experiment 7.
"""

from __future__ import annotations

import pytest

from tvbo.classes.experiment import SimulationExperiment

pytest.importorskip("brian2")

# A self-contained Deco 2014 cortical column: 160 E + 40 I, all-to-all recurrent
# AMPA (rec) + saturating NMDA (Mg block) + GABA-A, independent 2.4 kHz Poisson
# background per neuron. Constants from Deco (2014) Table 2.
DECO_COLUMN_YAML = """
label: "Deco 2014 column (Brian2 backend test)"
network:
  dynamics:
    ExcitatoryCell:
      iri: "extends:baseCellMembPot"
      parameters:
        C: {value: 0.5, unit: nF}
        gL: {value: 25.0, unit: nS}
        EL: {value: -70.0, unit: mV}
        thresh: {value: -50.0, unit: mV}
        reset: {value: -55.0, unit: mV}
        refract: {value: 2.0, unit: ms}
        v0: {value: -70.0, unit: mV}
      coupling_inputs: [iSyn]
      state_variables:
        v: {equation: {rhs: "(gL * (EL - v) + iSyn) / C"}, initial_value: -70.0, unit: mV, record: true}
      events:
        spike: {condition: {rhs: "v > thresh"}, affect: {rhs: "v = reset"}}
    InhibitoryCell:
      iri: "extends:baseCellMembPot"
      parameters:
        C: {value: 0.2, unit: nF}
        gL: {value: 20.0, unit: nS}
        EL: {value: -70.0, unit: mV}
        thresh: {value: -50.0, unit: mV}
        reset: {value: -55.0, unit: mV}
        refract: {value: 1.0, unit: ms}
        v0: {value: -70.0, unit: mV}
      coupling_inputs: [iSyn]
      state_variables:
        v: {equation: {rhs: "(gL * (EL - v) + iSyn) / C"}, initial_value: -70.0, unit: mV, record: true}
      events:
        spike: {condition: {rhs: "v > thresh"}, affect: {rhs: "v = reset"}}
    AMPA_ext_E:
      iri: neuroml:expOneSynapse
      parameters: {gbase: {value: 3.37, unit: nS}, erev: {value: 0.0, unit: mV}, tauDecay: {value: 2.0, unit: ms}}
    AMPA_rec_E:
      iri: neuroml:expOneSynapse
      parameters: {gbase: {value: 0.065, unit: nS}, erev: {value: 0.0, unit: mV}, tauDecay: {value: 2.0, unit: ms}}
    NMDA_E:
      iri: "extends:baseConductanceBasedSynapse"
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
        s: {equation: {rhs: "-s / tauDecay + alpha * x * (1 - s)"}, initial_value: 0.0, unit: dimensionless}
        x: {equation: {rhs: "-x / tauRise"}, initial_value: 0.0, unit: dimensionless}
      derived_variables:
        block: {equation: {rhs: "1 / (1 + mgFactor * exp(-v / scalingVolt))"}, unit: dimensionless}
        g: {equation: {rhs: "gbase * s * block"}, unit: nS}
        i: {equation: {rhs: "g * (erev - v)"}, unit: nA}
      events:
        spike_in: {affect: {rhs: "x = x + 1"}}
    GABA_E:
      iri: neuroml:expOneSynapse
      parameters: {gbase: {value: 10.94, unit: nS}, erev: {value: -70.0, unit: mV}, tauDecay: {value: 10.0, unit: ms}}
    AMPA_rec_I:
      iri: neuroml:expOneSynapse
      parameters: {gbase: {value: 0.051, unit: nS}, erev: {value: 0.0, unit: mV}, tauDecay: {value: 2.0, unit: ms}}
    NMDA_I:
      iri: "extends:baseConductanceBasedSynapse"
      parameters:
        gbase: {value: 0.16, unit: nS}
        erev: {value: 0.0, unit: mV}
        tauDecay: {value: 100.0, unit: ms}
        tauRise: {value: 2.0, unit: ms}
        alpha: {value: 0.5, unit: per_ms}
        mgFactor: {value: 0.28, unit: dimensionless}
        scalingVolt: {value: 16.129, unit: mV}
      coupling_inputs: [v]
      state_variables:
        s: {equation: {rhs: "-s / tauDecay + alpha * x * (1 - s)"}, initial_value: 0.0, unit: dimensionless}
        x: {equation: {rhs: "-x / tauRise"}, initial_value: 0.0, unit: dimensionless}
      derived_variables:
        block: {equation: {rhs: "1 / (1 + mgFactor * exp(-v / scalingVolt))"}, unit: dimensionless}
        g: {equation: {rhs: "gbase * s * block"}, unit: nS}
        i: {equation: {rhs: "g * (erev - v)"}, unit: nA}
      events:
        spike_in: {affect: {rhs: "x = x + 1"}}
    GABA_I:
      iri: neuroml:expOneSynapse
      parameters: {gbase: {value: 8.51, unit: nS}, erev: {value: -70.0, unit: mV}, tauDecay: {value: 10.0, unit: ms}}
    Background_E:
      iri: neuroml:poissonFiringSynapse
      parameters: {averageRate: {value: 2400.0, unit: Hz}, synapse: {value: AMPA_ext_E}, spikeTarget: {value: "./AMPA_ext_E"}}
    Background_I:
      iri: neuroml:poissonFiringSynapse
      parameters: {averageRate: {value: 2400.0, unit: Hz}, synapse: {value: AMPA_ext_I}, spikeTarget: {value: "./AMPA_ext_I"}}
    AMPA_ext_I:
      iri: neuroml:expOneSynapse
      parameters: {gbase: {value: 2.59, unit: nS}, erev: {value: 0.0, unit: mV}, tauDecay: {value: 2.0, unit: ms}}
  nodes:
    - {id: 0, dynamics: Background_E}
    - {id: 1, dynamics: Background_I}
    - {id: 2, dynamics: ExcitatoryCell, size: 160}
    - {id: 3, dynamics: InhibitoryCell, size: 40}
  edges:
    - {source: 0, target: 2, connectivity: all_to_all}
    - {source: 1, target: 3, connectivity: all_to_all}
    - {source: 2, target: 2, dynamics: AMPA_rec_E, connectivity: all_to_all, allow_self_connections: false, parameters: {weight: {value: 1.4}}}
    - {source: 2, target: 2, dynamics: NMDA_E, connectivity: all_to_all, allow_self_connections: false, parameters: {weight: {value: 1.4}}}
    - {source: 2, target: 3, dynamics: AMPA_rec_I, connectivity: all_to_all, parameters: {weight: {value: 1.0}}}
    - {source: 2, target: 3, dynamics: NMDA_I, connectivity: all_to_all, parameters: {weight: {value: 1.0}}}
    - {source: 3, target: 2, dynamics: GABA_E, connectivity: all_to_all, parameters: {weight: {value: 1.0}}}
    - {source: 3, target: 3, dynamics: GABA_I, connectivity: all_to_all, allow_self_connections: false, parameters: {weight: {value: 1.0}}}
integration: {method: euler, step_size: 0.02, duration: 1500.0, time_scale: ms}
"""


@pytest.fixture(scope="module")
def column(tmp_path_factory) -> SimulationExperiment:
    path = tmp_path_factory.mktemp("brian2") / "column.yaml"
    path.write_text(DECO_COLUMN_YAML)
    return SimulationExperiment.from_file(str(path))


class TestBrian2Render:
    """`render("brian2")` emits a valid, self-contained, runnable Brian2 script."""

    @pytest.fixture(scope="class")
    def script(self, column) -> str:
        return column.render("brian2")

    def test_is_valid_python(self, script):
        compile(script, "<generated>", "exec")

    def test_lowers_all_to_all_to_summed_hubs(self, script):
        # The whole point: population sums, not enumerated Synapses.connect().
        assert "(summed)" in script
        assert "linked_var(" in script
        assert ".connect(i" not in script and ".connect(condition" not in script

    def test_has_the_expected_structure(self, script):
        assert "NeuronGroup(\n    160" in script  # E population
        assert "NeuronGroup(\n    40" in script    # I population
        assert "PoissonInput(" in script           # 2.4 kHz background
        assert "SpikeMonitor(" in script
        # NMDA Mg block is emitted through the shared Brian2 printer (bare exp()).
        assert "exp(-v/scalingVolt" in script


@pytest.mark.backend_brian2
@pytest.mark.slow
class TestBrian2ReproducesTable2:
    """The native backend reproduces Deco (2014) Table-2 spontaneous rates."""

    def test_run_reproduces_rates(self, column):
        res = column.run(format="brian2", seed=3, settle_ms=500.0)
        rates = res._extras["rates"]
        rE, rI = rates["ExcitatoryCell"], rates["InhibitoryCell"]
        # Table 2 targets: E 2.92 Hz, I 7.54 Hz. Assert the correct regime with a
        # margin for the finite window; the balance (E < I, ~1:2.6) must hold.
        assert 2.0 <= rE <= 4.0, f"E rate {rE:.3f} outside Table-2 regime"
        assert 6.0 <= rI <= 9.5, f"I rate {rI:.3f} outside Table-2 regime"
        assert rE < rI

    def test_render_and_run_agree(self, column):
        """The generated script and the in-process run are the same computation.

        Both at seed 3 with numpy codegen and the same stationary window, so the
        result is spike-level identical — a regression here means render() drifted
        from run(). (The default settle window matches the script's, so no
        ``settle_ms`` override is passed to run().)
        """
        run_rates = column.run(format="brian2", seed=3)._extras["rates"]
        script = column.render("brian2", seed=3)
        ns = {}
        exec(compile(script, "<generated>", "exec"), ns)
        gen_rates = ns["RATES"]
        assert gen_rates["ExcitatoryCell"] == pytest.approx(run_rates["ExcitatoryCell"], abs=1e-9)
        assert gen_rates["InhibitoryCell"] == pytest.approx(run_rates["InhibitoryCell"], abs=1e-9)
