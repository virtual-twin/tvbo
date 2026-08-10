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

import re

import pytest

from tvbo.classes.experiment import SimulationExperiment


def squashed(script: str) -> str:
    """The script with all whitespace removed, for asserting on call structure.

    Generated Python is formatted before it is returned, so a long call is wrapped
    across lines. Assertions about which arguments a call receives must therefore
    not depend on where the formatter chose to break it.
    """
    return re.sub(r"\s+", "", script)


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
        assert "NeuronGroup(\n    40" in script  # I population
        assert "PoissonInput(" in script  # 2.4 kHz background
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


# A recurrent E population wired with SPARSE random connectivity through a facilitating
# Tsodyks-Markram synapse. Unlike the all-to-all column, a genuinely sparse projection is
# emitted as a real Brian2 `Synapses` (not a summed hub): the conductance decays on the
# post-synaptic neuron, STP (u, x) lives on the synapse as (event-driven) variables.
SPARSE_STP_YAML = """
label: "Sparse facilitating E->E network (Brian2 sparse-path test)"
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
    AMPA_ext: {iri: neuroml:expOneSynapse, parameters: {gbase: {value: 2.5, unit: nS}, erev: {value: 0.0, unit: mV}, tauDecay: {value: 2.0, unit: ms}}}
    Background_E: {iri: neuroml:poissonFiringSynapse, parameters: {averageRate: {value: 1800.0, unit: Hz}, synapse: {value: AMPA_ext}, spikeTarget: {value: "./AMPA_ext"}}}
    STP_fac_E:
      iri: "extends:baseConductanceBasedSynapse"
      parameters:
        gbase: {value: 3.0, unit: nS}
        erev: {value: 0.0, unit: mV}
        tauDecay: {value: 3.0, unit: ms}
        U: {value: 0.15, unit: dimensionless}
        tauF: {value: 1500.0, unit: ms}
        tauD: {value: 200.0, unit: ms}
      coupling_inputs: [v]
      state_variables:
        g: {equation: {rhs: "-g / tauDecay"}, initial_value: 0.0, unit: dimensionless}
        u: {equation: {rhs: "(U - u) / tauF"}, initial_value: 0.15, unit: dimensionless}
        x: {equation: {rhs: "(1 - x) / tauD"}, initial_value: 1.0, unit: dimensionless}
      derived_variables:
        i: {equation: {rhs: "gbase * g * (erev - v)"}, unit: nA}
      events:
        spike_in: {affect: {rhs: "u = u + U*(1 - u); g = g + u*x; x = x - u*x"}}
  nodes:
    - {id: 0, dynamics: Background_E}
    - {id: 2, dynamics: ExcitatoryCell, size: 80}
  edges:
    - {source: 0, target: 2, connectivity: all_to_all}
    - {source: 2, target: 2, dynamics: STP_fac_E, connectivity: random, allow_self_connections: false,
       parameters: {weight: {value: 1.0}, connection_probability: {value: 0.12}}}
integration: {method: euler, step_size: 0.05, duration: 800.0, time_scale: ms}
"""


class TestBrian2SparseConnectivity:
    """A `random` projection lowers to a real sparse `Synapses`, not a summed hub."""

    @pytest.fixture(scope="class")
    def net(self, tmp_path_factory) -> SimulationExperiment:
        path = tmp_path_factory.mktemp("brian2sparse") / "sparse.yaml"
        path.write_text(SPARSE_STP_YAML)
        return SimulationExperiment.from_file(str(path))

    @pytest.fixture(scope="class")
    def script(self, net) -> str:
        return net.render("brian2", seed=7)

    def test_is_valid_python(self, script):
        compile(script, "<generated>", "exec")

    def test_random_emits_sparse_synapses(self, script):
        # Real Synapses with connect(p=...) excluding autapses — not a (summed) hub.
        assert "Synapses(ExcitatoryCell,ExcitatoryCell" in squashed(script)
        assert 'connect(p=0.12,condition="i!=j")' in squashed(script)

    def test_conductance_delivered_postsynaptically(self, script):
        # Decaying conductance on the post-synaptic cell, incremented event-driven by on_pre.
        # Objects are named by (synapse, source, target) so block-structured nets never collide.
        assert "dgsyn_STP_fac_E_from_ExcitatoryCell_to_ExcitatoryCell/dt = -gsyn" in script
        assert "gsyn_STP_fac_E_from_ExcitatoryCell_to_ExcitatoryCell_post +=" in script

    def test_stp_state_is_per_synapse_event_driven(self, script):
        # u, x are per-synapse (event-driven); facilitation runs before release (recipe order).
        assert "du/dt = (U - u)/tauF : 1 (event-driven)" in script
        assert "dx/dt = (1 - x)/tauD : 1 (event-driven)" in script
        assert script.index("u = u + U*(1 - u)") < script.index("x = x - u*x")

    @pytest.mark.backend_brian2
    @pytest.mark.slow
    def test_render_and_run_agree(self, net):
        """Sparse connectivity is seed-identical between the run and the rendered script."""
        run_rates = net.run(format="brian2", seed=7)._extras["rates"]
        script = net.render("brian2", seed=7)
        ns = {}
        exec(compile(script, "<generated>", "exec"), ns)
        assert ns["RATES"]["ExcitatoryCell"] == pytest.approx(run_rates["ExcitatoryCell"], abs=1e-9)


# A deterministic timed current pulse (pulseGenerator) drives a cell only within its window.
# This is the declarative loading / nonspecific-readout primitive: a rectangular current
# (delay, duration, amplitude) summed into the target's membrane current. No Poisson, so the
# behaviour is fully deterministic — the cell fires only while the pulse is on.
PULSE_YAML = """
label: "Timed current pulse (Brian2 pulseGenerator test)"
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
    LoadPulse:
      iri: neuroml:pulseGenerator
      parameters: {delay: {value: 300.0, unit: ms}, duration: {value: 200.0, unit: ms}, amplitude: {value: 0.8, unit: nA}}
  nodes:
    - {id: 2, dynamics: ExcitatoryCell, size: 20}
    - {id: 5, dynamics: LoadPulse}
  edges:
    - {source: 5, target: 2, connectivity: all_to_all}
integration: {method: euler, step_size: 0.05, duration: 800.0, time_scale: ms}
"""


class TestBrian2TimedCurrentPulse:
    """A `pulseGenerator` lowers to a rectangular current window on its target population."""

    @pytest.fixture(scope="class")
    def net(self, tmp_path_factory) -> SimulationExperiment:
        path = tmp_path_factory.mktemp("brian2pulse") / "pulse.yaml"
        path.write_text(PULSE_YAML)
        return SimulationExperiment.from_file(str(path))

    @pytest.fixture(scope="class")
    def script(self, net) -> str:
        return net.render("brian2")

    def test_is_valid_python(self, script):
        compile(script, "<generated>", "exec")

    def test_emits_time_gated_current_term(self, script):
        # A rectangular window int(t>=delay)*int(t<delay+duration), summed into iSyn.
        assert "int(t >= delay_stim_LoadPulse)" in script
        assert "int(t < delay_stim_LoadPulse + dur_stim_LoadPulse)" in script
        assert "amp_stim_LoadPulse" in script

    @pytest.mark.backend_brian2
    @pytest.mark.slow
    def test_fires_only_within_the_pulse_window(self, net):
        # Deterministic: the cell is silent at rest, driven above threshold only 300–500 ms.
        res = net.run(format="brian2")
        t = res._extras["spikes"]["ExcitatoryCell"]["t_ms"]
        before = int(((t >= 50) & (t < 300)).sum())
        during = int(((t >= 300) & (t < 500)).sum())
        after = int(((t >= 500) & (t < 800)).sum())
        assert before == 0 and after == 0, f"pulse leaked outside its window: {before=} {after=}"
        assert during > 0, "pulse did not drive any spikes in its window"

    @pytest.mark.backend_brian2
    @pytest.mark.slow
    def test_render_and_run_agree(self, net):
        """The deterministic pulse is identical between the run and the rendered script."""
        run_rate = net.run(format="brian2")._extras["rates"]["ExcitatoryCell"]
        script = net.render("brian2")
        ns = {}
        exec(compile(script, "<generated>", "exec"), ns)
        assert ns["RATES"]["ExcitatoryCell"] == pytest.approx(run_rate, abs=1e-9)


# The faithful Mongillo/Amit-Brunel form: a CURRENT-BASED LIF (tau_m dV/dt = -V + mu_ext + iSyn,
# V_rest 0) with an instantaneous (delta) PSC recurrent synapse (a spike jumps v_post directly,
# no conductance / no synaptic time constant) carrying short-term facilitation, plus additive
# Gaussian white-noise external drive. Two selective E sub-populations wired sparsely through the
# facilitating delta synapse (potentiated within-population, baseline across) exercise the
# block-structured delta path; a timed pulse loads one population.
DELTA_STP_YAML = """
label: "Current-based delta-PSC facilitating network (Brian2 delta+noise test)"
network:
  dynamics:
    ExcitatoryCell:
      iri: "extends:baseCellMembPot"
      parameters:
        tau_m: {value: 15.0, unit: ms}
        thresh: {value: 20.0, unit: mV}
        reset: {value: 16.0, unit: mV}
        refract: {value: 2.0, unit: ms}
        mu_ext: {value: 17.0, unit: mV}
        v0: {value: 16.0, unit: mV}
      coupling_inputs: [iSyn]
      state_variables:
        v:
          equation: {rhs: "(-v + mu_ext + iSyn) / tau_m"}
          initial_value: 16.0
          unit: mV
          record: true
          noise: {parameters: {sigma: {value: 1.0, unit: mV}}}
      events:
        spike: {condition: {rhs: "v > thresh"}, affect: {rhs: "v = reset"}}
    STP_E:
      iri: "extends:baseConductanceBasedSynapse"
      parameters:
        U: {value: 0.20, unit: dimensionless}
        tauF: {value: 1500.0, unit: ms}
        tauD: {value: 200.0, unit: ms}
      coupling_inputs: [v]
      state_variables:
        u: {equation: {rhs: "(U - u) / tauF"}, initial_value: 0.20, unit: dimensionless}
        x: {equation: {rhs: "(1 - x) / tauD"}, initial_value: 1.0, unit: dimensionless}
      events:
        spike_in: {affect: {rhs: "u = u + U*(1 - u); v = v + u*x; x = x - u*x"}}
    LoadA:
      iri: neuroml:pulseGenerator
      parameters: {delay: {value: 300.0, unit: ms}, duration: {value: 300.0, unit: ms}, amplitude: {value: 6.0, unit: mV}}
  nodes:
    - {id: 10, dynamics: ExcitatoryCell, size: 60}
    - {id: 11, dynamics: ExcitatoryCell, size: 60}
    - {id: 30, dynamics: LoadA}
  edges:
    - {source: 10, target: 10, dynamics: STP_E, connectivity: random, allow_self_connections: false, parameters: {weight: {value: 3.0}, connection_probability: {value: 0.2}}}
    - {source: 11, target: 11, dynamics: STP_E, connectivity: random, allow_self_connections: false, parameters: {weight: {value: 3.0}, connection_probability: {value: 0.2}}}
    - {source: 10, target: 11, dynamics: STP_E, connectivity: random, parameters: {weight: {value: 0.5}, connection_probability: {value: 0.2}}}
    - {source: 11, target: 10, dynamics: STP_E, connectivity: random, parameters: {weight: {value: 0.5}, connection_probability: {value: 0.2}}}
    - {source: 30, target: 10, connectivity: all_to_all}
integration: {method: euler, step_size: 0.05, duration: 1500.0, time_scale: ms}
"""


class TestBrian2DeltaPscAndNoise:
    """Current-based LIF + instantaneous (delta) PSC + Gaussian white-noise membrane."""

    @pytest.fixture(scope="class")
    def net(self, tmp_path_factory) -> SimulationExperiment:
        path = tmp_path_factory.mktemp("brian2delta") / "delta.yaml"
        path.write_text(DELTA_STP_YAML)
        return SimulationExperiment.from_file(str(path))

    @pytest.fixture(scope="class")
    def script(self, net) -> str:
        return net.render("brian2", seed=5)

    def test_is_valid_python(self, script):
        compile(script, "<generated>", "exec")

    def test_current_based_membrane_has_volt_drive_and_noise(self, script):
        # Current-based cell: drive iSyn is a voltage (not amp), and the membrane carries xi noise.
        assert "iSyn = " in script and ": volt" in script
        assert "noise_sigma_v * xi * tau_m**-0.5" in script

    def test_delta_synapse_jumps_v_post_directly(self, script):
        # No decaying conductance for the delta synapse — the spike jumps v_post directly.
        assert "v_post +=" in script
        assert "* mV" in script
        # STP lives on the synapse; facilitation before release (v jump uses the updated u).
        assert "(event-driven)" in script
        assert script.index("u = u + U*(1 - u)") < script.index("v_post +=")
        assert script.index("v_post +=") < script.index("x = x - u*x")

    def test_block_synapse_names_are_target_keyed(self, script):
        # The four block edges (AA, BB, AB, BA) must produce four distinctly-named Synapses.
        for pair in (
            "ExcitatoryCell_10_to_ExcitatoryCell_10",
            "ExcitatoryCell_11_to_ExcitatoryCell_11",
            "ExcitatoryCell_10_to_ExcitatoryCell_11",
            "ExcitatoryCell_11_to_ExcitatoryCell_10",
        ):
            assert f"syn_STP_E_from_{pair}" in script

    @pytest.mark.backend_brian2
    @pytest.mark.slow
    def test_facilitation_holds_after_loading(self, net):
        # Loading population A elevates its presynaptic utilisation u and keeps it above baseline
        # into the delay (the activity-silent memory trace); B stays at baseline.
        import numpy as np

        res = net.run(format="brian2", seed=5)
        sp = res._extras["spikes"]
        U, tauF = 0.20, 1500.0

        def mean_u(pop, t0, t1, n):
            grid = np.arange(0, t1, 2.0)
            acc = np.zeros_like(grid)
            ii, tt = sp[pop]["i"], sp[pop]["t_ms"]
            for j in range(n):
                st = np.sort(tt[ii == j])
                u = U
                last = 0.0
                k = 0
                for gi, tnow in enumerate(grid):
                    while k < len(st) and st[k] <= tnow:
                        u = U + (u - U) * np.exp(-(st[k] - last) / tauF)
                        u += U * (1 - u)
                        last = st[k]
                        k += 1
                    acc[gi] += U + (u - U) * np.exp(-(tnow - last) / tauF)
            acc /= n
            m = (grid >= t0) & (grid < t1)
            return acc[m].mean()

        uA_spont = mean_u("ExcitatoryCell_10", 50, 290, 60)
        uA_delay = mean_u("ExcitatoryCell_10", 700, 1400, 60)
        uB_delay = mean_u("ExcitatoryCell_11", 700, 1400, 60)
        assert uA_delay > uA_spont + 0.03, f"loading did not facilitate A: {uA_spont=:.3f} {uA_delay=:.3f}"
        assert uA_delay > uB_delay + 0.03, f"facilitation not selective: {uA_delay=:.3f} {uB_delay=:.3f}"


# A spiking run persists its raster to the result container so `tvbo run` reproduces from disk:
# ExperimentResult.save writes per-population spike times + neuron indices as flat variables plus
# the firing rates and sizes on a shared axis, with the run window in the Dataset attrs. Two E
# populations (one driven by a pulse, one silent) exercise the per-population keying and the
# save→load round-trip a spiking study's figures bind to.
CONTAINER_YAML = """
label: "Spiking result-container round-trip (Brian2)"
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
    LoadPulse:
      iri: neuroml:pulseGenerator
      parameters: {delay: {value: 100.0, unit: ms}, duration: {value: 300.0, unit: ms}, amplitude: {value: 0.8, unit: nA}}
  nodes:
    - {id: 2, dynamics: ExcitatoryCell, size: 15}
    - {id: 3, dynamics: ExcitatoryCell, size: 10}
    - {id: 5, dynamics: LoadPulse}
  edges:
    - {source: 5, target: 2, connectivity: all_to_all}
integration: {method: euler, step_size: 0.05, duration: 500.0, time_scale: ms}
"""


@pytest.mark.backend_brian2
@pytest.mark.slow
class TestBrian2ResultContainer:
    """A spiking run's raster round-trips through the saved HDF5 result container."""

    def _run_and_save(self, tmp_path):
        import xarray as xr

        path = tmp_path / "container.yaml"
        path.write_text(CONTAINER_YAML)
        exp = SimulationExperiment.from_file(str(path))
        res = exp.run(format="brian2", seed=2)
        res.save(str(tmp_path))
        h5 = next(p for p in sorted(tmp_path.glob("*result.h5")))
        return res, xr.open_dataset(str(h5), engine="h5netcdf")

    def test_raster_rates_and_window_persist(self, tmp_path):
        res, ds = self._run_and_save(tmp_path)
        pops = list(ds.attrs["populations"])
        assert pops == ["ExcitatoryCell_2", "ExcitatoryCell_3"]
        assert float(ds.attrs["duration_ms"]) == 500.0 and float(ds.attrs["dt_ms"]) == 0.05
        # per-population rasters present; sizes and rates on the shared population axis, in order.
        import numpy as np

        assert list(np.asarray(ds["population_size"])) == [15.0, 10.0]
        assert ds["firing_rate"].sizes["population"] == 2
        # the population axis is KEYED (never positional): the coord labels equal attrs and the
        # per-population raster tokens, so a rate is selectable by name and maps to its raster.
        assert list(ds["firing_rate"].coords["population"].values) == pops
        assert float(ds["population_size"].sel(population="ExcitatoryCell_2")) == 15.0
        for p in pops:
            assert f"spikes__{p}__t" in ds.data_vars and f"spikes__{p}__i" in ds.data_vars
        # the driven population fired; the container raster matches the in-memory run exactly.
        driven_t = np.asarray(ds["spikes__ExcitatoryCell_2__t"].values)
        assert driven_t.size > 0
        assert driven_t.size == res._extras["spikes"]["ExcitatoryCell_2"]["t_ms"].size
        np.testing.assert_allclose(np.sort(driven_t), np.sort(res._extras["spikes"]["ExcitatoryCell_2"]["t_ms"]), atol=1e-6)
        ds.close()


# A conductance-based cell (drive is a current; no membrane time constant) that declares membrane
# noise: the sigma*xi/sqrt(tau_m) discretisation has no tau_m to normalise by, so the build must
# reject it with a clear message instead of emitting an equation that references an undefined
# tau_m and fails opaquely inside Brian2.
CONDUCTANCE_NOISE_YAML = """
label: "Conductance cell with membrane noise (unsupported)"
network:
  dynamics:
    NoisyCell:
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
        v:
          equation: {rhs: "(gL * (EL - v) + iSyn) / C"}
          initial_value: -70.0
          unit: mV
          record: true
          noise: {parameters: {sigma: {value: 1.0, unit: mV}}}
      events:
        spike: {condition: {rhs: "v > thresh"}, affect: {rhs: "v = reset"}}
  nodes:
    - {id: 2, dynamics: NoisyCell, size: 5}
integration: {method: euler, step_size: 0.05, duration: 100.0, time_scale: ms}
"""


def test_conductance_cell_with_membrane_noise_is_rejected(tmp_path):
    """A conductance-based cell declaring membrane noise fails loudly (no undefined tau_m)."""
    path = tmp_path / "cond_noise.yaml"
    path.write_text(CONDUCTANCE_NOISE_YAML)
    exp = SimulationExperiment.from_file(str(path))
    with pytest.raises(NotImplementedError, match="tau_m"):
        exp.render("brian2")


# Two pulse edges from the same generator onto the same population: their windows must SUM, each
# keeping its own weight, not collapse to a single window carrying only the last edge's weight.
TWO_PULSE_YAML = """
label: "Two pulse edges onto one population sum (Brian2)"
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
    LoadPulse:
      iri: neuroml:pulseGenerator
      parameters: {delay: {value: 100.0, unit: ms}, duration: {value: 200.0, unit: ms}, amplitude: {value: 0.8, unit: nA}}
  nodes:
    - {id: 2, dynamics: ExcitatoryCell, size: 10}
    - {id: 5, dynamics: LoadPulse}
  edges:
    - {source: 5, target: 2, connectivity: all_to_all, parameters: {weight: {value: 2.0}}}
    - {source: 5, target: 2, connectivity: all_to_all, parameters: {weight: {value: 3.0}}}
integration: {method: euler, step_size: 0.05, duration: 400.0, time_scale: ms}
"""


def test_multiple_pulse_edges_onto_one_population_sum(tmp_path):
    """Two same-source pulse edges of different weight sum into two windows, not one."""
    path = tmp_path / "two_pulse.yaml"
    path.write_text(TWO_PULSE_YAML)
    script = SimulationExperiment.from_file(str(path)).render("brian2")
    window = "amp_stim_LoadPulse * int(t >= delay_stim_LoadPulse)"
    assert script.count(window) == 2, "the two pulse edges did not both survive into iSyn"
    assert "2.0 * amp_stim_LoadPulse" in script and "3.0 * amp_stim_LoadPulse" in script


def test_render_seeds_random_connectivity_from_execution_random_seed(tmp_path):
    """render() with no explicit seed emits seed(execution.random_seed) so the codegen path
    builds the SAME random connectivity as run() — the render≡run invariant for sparse nets."""
    path = tmp_path / "seeded.yaml"
    path.write_text(SPARSE_STP_YAML + "execution: {random_seed: 4242}\n")
    exp = SimulationExperiment.from_file(str(path))
    assert "seed(4242)" in exp.render("brian2"), "the recipe seed did not reach the rendered script"


def test_render_is_unseeded_without_a_declared_seed(tmp_path):
    """No execution.random_seed and no explicit argument → no seed() call (unchanged behaviour)."""
    path = tmp_path / "unseeded.yaml"
    path.write_text(SPARSE_STP_YAML)
    exp = SimulationExperiment.from_file(str(path))
    assert "seed(" not in exp.render("brian2")


# A delta (instantaneous-PSC) synapse whose spike event scales the jump by a unit-valued parameter
# (J in mV): the delivered `v_post += weight*(J*u*x)*mV` would be dimensionally inconsistent (mV^2),
# so the build must reject it and steer the amplitude into the edge weight instead of failing deep
# inside Brian2. STP_bad mirrors the faithful STP_E but moves the mV scale into a synapse parameter.
DELTA_UNITFUL_YAML = """
label: "Delta synapse with unit-valued jump (unsupported)"
network:
  dynamics:
    ExcitatoryCell:
      iri: "extends:baseCellMembPot"
      parameters:
        tau_m: {value: 15.0, unit: ms}
        thresh: {value: 20.0, unit: mV}
        reset: {value: 16.0, unit: mV}
        refract: {value: 2.0, unit: ms}
        mu_ext: {value: 17.0, unit: mV}
        v0: {value: 16.0, unit: mV}
      coupling_inputs: [iSyn]
      state_variables:
        v: {equation: {rhs: "(-v + mu_ext + iSyn) / tau_m"}, initial_value: 16.0, unit: mV, record: true}
      events:
        spike: {condition: {rhs: "v > thresh"}, affect: {rhs: "v = reset"}}
    STP_bad:
      iri: "extends:baseConductanceBasedSynapse"
      parameters:
        U: {value: 0.20, unit: dimensionless}
        J: {value: 0.45, unit: mV}
        tauF: {value: 1500.0, unit: ms}
        tauD: {value: 200.0, unit: ms}
      coupling_inputs: [v]
      state_variables:
        u: {equation: {rhs: "(U - u) / tauF"}, initial_value: 0.20, unit: dimensionless}
        x: {equation: {rhs: "(1 - x) / tauD"}, initial_value: 1.0, unit: dimensionless}
      events:
        spike_in: {affect: {rhs: "u = u + U*(1 - u); v = v + J*u*x; x = x - u*x"}}
  nodes:
    - {id: 10, dynamics: ExcitatoryCell, size: 20}
  edges:
    - {source: 10, target: 10, dynamics: STP_bad, connectivity: random, allow_self_connections: false,
       parameters: {weight: {value: 3.0}, connection_probability: {value: 0.2}}}
integration: {method: euler, step_size: 0.05, duration: 100.0, time_scale: ms}
"""


def test_delta_synapse_with_unit_valued_jump_is_rejected(tmp_path):
    """A delta jump scaled by a unit-valued parameter fails loudly (dimensionless increment only)."""
    path = tmp_path / "delta_bad.yaml"
    path.write_text(DELTA_UNITFUL_YAML)
    exp = SimulationExperiment.from_file(str(path))
    with pytest.raises(NotImplementedError, match="dimensionless"):
        exp.render("brian2")


def test_probe_sample_indices_are_evenly_spaced_and_never_divide_by_zero():
    """The probe sampler is safe at the k==1 edge and evenly spans the population otherwise."""
    from tvbo.adapters.brian2 import _sample_indices

    assert _sample_indices(800, 1) == [0]  # k == 1: no (k-1) division-by-zero
    assert _sample_indices(5, 10) == [0, 1, 2, 3, 4]  # k >= n: all indices
    s = _sample_indices(800, 200)
    assert len(s) == 200 and s[0] == 0 and s[-1] == 799 and s == sorted(set(s))


# A delta STP synapse whose u/x carry `record: true`: the backend adds a clock-driven,
# zero-delivery OBSERVATION PROBE (a sampled copy of the source population's synapses, driven by
# the same presynaptic spikes) and StateMonitors its u/x. This measures the continuous synaptic
# state — including the slow decay held through silent intervals — that an event-driven monitor
# would freeze, without perturbing the network (the probe delivers nothing). A suprathreshold
# mu_ext makes the cells fire a regular train deterministically, so u builds by facilitation.
RECORDED_SYN_YAML = """
label: "Delta STP synapse with recorded internal state (observation probe)"
network:
  dynamics:
    ExcitatoryCell:
      iri: "extends:baseCellMembPot"
      parameters:
        tau_m: {value: 15.0, unit: ms}
        thresh: {value: 20.0, unit: mV}
        reset: {value: 16.0, unit: mV}
        refract: {value: 2.0, unit: ms}
        mu_ext: {value: 24.0, unit: mV}
        v0: {value: 16.0, unit: mV}
      coupling_inputs: [iSyn]
      state_variables:
        v: {equation: {rhs: "(-v + mu_ext + iSyn) / tau_m"}, initial_value: 16.0, unit: mV, record: true}
      events:
        spike: {condition: {rhs: "v > thresh"}, affect: {rhs: "v = reset"}}
    STP_E:
      iri: "extends:baseConductanceBasedSynapse"
      parameters:
        U: {value: 0.20, unit: dimensionless}
        tauF: {value: 1500.0, unit: ms}
        tauD: {value: 200.0, unit: ms}
      coupling_inputs: [v]
      state_variables:
        u: {equation: {rhs: "(U - u) / tauF"}, initial_value: 0.20, unit: dimensionless, record: true}
        x: {equation: {rhs: "(1 - x) / tauD"}, initial_value: 1.0, unit: dimensionless, record: true}
      events:
        spike_in: {affect: {rhs: "u = u + U*(1 - u); v = v + 0.1*u*x; x = x - u*x"}}
  nodes:
    - {id: 10, dynamics: ExcitatoryCell, size: 50}
  edges:
    - {source: 10, target: 10, dynamics: STP_E, connectivity: random, allow_self_connections: false,
       parameters: {weight: {value: 0.1}, connection_probability: {value: 0.2}}}
integration: {method: euler, step_size: 0.05, duration: 400.0, time_scale: ms}
"""


class TestBrian2RecordedSynapseState:
    """A synapse variable with `record: true` is measured by a clock-driven observation probe."""

    @pytest.fixture(scope="class")
    def net(self, tmp_path_factory) -> SimulationExperiment:
        path = tmp_path_factory.mktemp("brian2rec") / "rec.yaml"
        path.write_text(RECORDED_SYN_YAML)
        return SimulationExperiment.from_file(str(path))

    @pytest.fixture(scope="class")
    def script(self, net) -> str:
        return net.render("brian2", seed=3)

    def test_is_valid_python(self, script):
        compile(script, "<generated>", "exec")

    def test_emits_zero_delivery_clock_driven_probe(self, script):
        pname = "probe_STP_E_from_ExcitatoryCell_to_ExcitatoryCell"
        assert f"{pname}=Synapses(ExcitatoryCell,_probe_sink" in squashed(script)
        assert f"{pname}.connect(i=[" in squashed(script) and ",j=0" in squashed(script)
        # clock-driven (recordable continuous trace), and it delivers NOTHING to the sink.
        assert "(clock-driven)" in script
        probe_on_pre = script.split(pname + " = Synapses")[1].split(")")[0]
        assert "v_post +=" not in probe_on_pre
        # the generated script exposes the same shape as the in-process path: a time axis + means.
        assert "SYNAPSE_STATE[" in script and '"t_ms":' in script

    @pytest.mark.backend_brian2
    @pytest.mark.slow
    def test_probe_measures_ux_without_perturbing_the_network(self, net, tmp_path):
        import numpy as np
        import xarray as xr

        res = net.run(format="brian2", seed=3)
        res.save(str(tmp_path))
        ds = xr.open_dataset(str(next(tmp_path.glob("*result.h5"))), engine="h5netcdf")
        # the measured population-mean u/x round-trips, keyed by the source population.
        assert "synapse__ExcitatoryCell__u" in ds.data_vars
        rec = ds.attrs["synapse_recorded"]
        rec = [rec] if isinstance(rec, str) else [str(r) for r in np.atleast_1d(rec)]
        assert "ExcitatoryCell" in rec
        u = np.asarray(ds["synapse__ExcitatoryCell__u"].values)
        x = np.asarray(ds["synapse__ExcitatoryCell__x"].values)
        assert u.size > 50 and u.shape == x.shape
        # FACILITATION measured: u rises above its baseline U=0.20 and varies (not a frozen value);
        # x is depleted below 1. These are the recorded traces, not an analytic reconstruction.
        assert u.max() > 0.20 + 1e-3 and u.std() > 1e-3
        assert x.min() < 1.0 - 1e-3
        ds.close()
        # the observation probe delivers nothing: firing rate is identical to a run whose synapse
        # is NOT recorded (no probe built at all).
        norec_path = tmp_path / "norec.yaml"
        norec_path.write_text(RECORDED_SYN_YAML.replace(", record: true}", "}"))
        r_rec = res._extras["rates"]["ExcitatoryCell"]
        r_plain = (
            SimulationExperiment.from_file(str(norec_path)).run(format="brian2", seed=3)._extras["rates"]["ExcitatoryCell"]
        )
        assert r_rec == pytest.approx(r_plain, abs=1e-9), "the probe perturbed the network"


# A `random` current-pulse edge drives only a random SUBSET of the target's neurons (a per-neuron
# 0/1 mask, fraction = connection_probability), reusing the sparse-synapse convention — the paper's
# "nonspecific input to 15% of the excitatory neurons". A subthreshold mu_ext keeps the unstimulated
# neurons silent, so only the masked fraction fires, and only within the pulse window.
RANDOM_PULSE_YAML = """
label: "Random-subset current pulse (Brian2)"
network:
  dynamics:
    Cell:
      iri: "extends:baseCellMembPot"
      parameters:
        tau_m: {value: 15.0, unit: ms}
        thresh: {value: 20.0, unit: mV}
        reset: {value: 16.0, unit: mV}
        refract: {value: 2.0, unit: ms}
        mu_ext: {value: 15.0, unit: mV}
        v0: {value: 16.0, unit: mV}
      coupling_inputs: [iSyn]
      state_variables:
        v: {equation: {rhs: "(-v + mu_ext + iSyn) / tau_m"}, initial_value: 16.0, unit: mV, record: true}
      events:
        spike: {condition: {rhs: "v > thresh"}, affect: {rhs: "v = reset"}}
    Noise:
      iri: neuroml:pulseGenerator
      parameters: {delay: {value: 100.0, unit: ms}, duration: {value: 200.0, unit: ms}, amplitude: {value: 10.0, unit: mV}}
  nodes:
    - {id: 2, dynamics: Cell, size: 200}
    - {id: 5, dynamics: Noise}
  edges:
    - {source: 5, target: 2, connectivity: random, parameters: {connection_probability: {value: 0.3}}}
integration: {method: euler, step_size: 0.05, duration: 400.0, time_scale: ms}
execution: {random_seed: 0}
"""


class TestBrian2RandomSubsetPulse:
    """A `random` current-pulse edge drives only a fraction of the target (per-neuron mask)."""

    @pytest.fixture(scope="class")
    def net(self, tmp_path_factory) -> SimulationExperiment:
        path = tmp_path_factory.mktemp("brian2subset") / "subset.yaml"
        path.write_text(RANDOM_PULSE_YAML)
        return SimulationExperiment.from_file(str(path))

    @pytest.fixture(scope="class")
    def script(self, net) -> str:
        return net.render("brian2")

    def test_is_valid_python(self, script):
        compile(script, "<generated>", "exec")

    def test_emits_seeded_per_neuron_mask(self, script):
        # a constant per-neuron mask, drawn from the seeded RNG (render≡run), gating the
        # window. Keyed by edge index, so two random pulse edges onto one population are
        # two independent subsets rather than one shared mask.
        assert "stim_mask_Noise_0 : 1 (constant)" in script
        assert 'stim_mask_Noise_0 = "rand() < 0.3"' in script
        assert "stim_mask_Noise_0 * 1.0 * amp_stim_Noise" in script

    @pytest.mark.backend_brian2
    @pytest.mark.slow
    def test_only_the_masked_fraction_fires_within_the_window(self, net):
        import numpy as np

        spk = net.run(format="brian2", seed=0)._extras["spikes"]["Cell"]
        i, t = np.asarray(spk["i"]), np.asarray(spk["t_ms"])
        assert 0.20 < len(np.unique(i)) / 200 < 0.40  # ~30% of neurons driven (fraction 0.3)
        assert np.all((t >= 100.0) & (t < 300.0))  # only within the pulse window
