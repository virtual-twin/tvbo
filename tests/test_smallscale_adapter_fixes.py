"""NeuroML refuses what every other backend refuses, honours a settle as they do, and the small-scale backends read their window from one place.

`NeuroMLAdapter.render_code` passes the declaration through `refuse_unrenderable` before it builds a context, as `BaseAdapter.render_context` does for every other backend, so a delayed coupling with no per-edge delay to lower, and an input timed from the run start behind a settle, are refused instead of emitted as well-formed LEMS for the rest.

A declared settle is the head of the LEMS run: every onset and every reading of ``t`` moves a settle later, and the run is reported on the measurement clock, so an input lands at the same measured time with or without one.

The NeuroML, Brian2 and Gillespie windows are `BaseAdapter.get_integration_info`'s. Patching that one reader moves every one of them; a window re-derived with private defaults would not follow it.
"""

import re
import textwrap

import pytest

from tvbo import database_path
from tvbo.adapters.base import BaseAdapter
from tvbo.adapters.neuroml import NeuroMLAdapter, build_lems_context, build_std_lems_context
from tvbo.classes.experiment import SimulationExperiment

from .test_edge_requirements import _DELAYED, _experiment

NEUROML_EXPERIMENTS = database_path / "experiments" / "neuroml"

EDGE = "  edges:\n    - {{source: 0, target: 1, parameters: {{weight: {{value: 1.0}}, {carrier}: {{value: 2.0}}}}}}\n"
"""One explicit connection, carrying the named delay carrier besides its weight."""

LIF_CELL = """
  name: LIFCell
  iri: "extends:baseCellMembPot"
  parameters:
    C: {value: 0.5, unit: nF}
    gL: {value: 25.0, unit: nS}
    EL: {value: -70.0, unit: mV}
    thresh: {value: -50.0, unit: mV}
    reset: {value: -55.0, unit: mV}
    v0: {value: -70.0, unit: mV}
  state_variables:
    v: {equation: {rhs: "(gL * (EL - v)) / C"}, initial_value: -70.0, unit: mV}
  events:
    spike: {condition: {rhs: "v > thresh"}, affect: {rhs: "v = reset"}}
"""
"""A leaky integrate-and-fire cell extending a LEMS base type, which NeuroML lowers through its hierarchical custom template and Brian2 as one population."""

IAF_CELL = """
  iri: neuroml:iafCell
  parameters:
    leakReversal: {value: -50.0, unit: mV}
    thresh: {value: -55.0, unit: mV}
    reset: {value: -70.0, unit: mV}
    C: {value: 0.2, unit: nF}
    leakConductance: {value: 0.01, unit: uS}
"""
"""A standard NeuroML cell, emitted as a ``<iafCell>`` component; named by its key in a network, by ``name`` on its own."""

IAF_NETWORK = f"""
label: "two standard cells, one synapse"
network:
  dynamics:
    IaF:{textwrap.indent(IAF_CELL, "    ")}
    Syn:
      iri: neuroml:expOneSynapse
      parameters: {{gbase: {{value: 1.0, unit: nS}}, erev: {{value: 0.0, unit: mV}}, tauDecay: {{value: 2.0, unit: ms}}}}
  nodes:
    - {{id: 0, dynamics: IaF}}
    - {{id: 1, dynamics: IaF}}
  edges:
    - {{source: 0, target: 1, dynamics: Syn, parameters: {{weight: {{value: 1.0}}}}}}
"""
"""A standard-type network, which NeuroML lowers through its standard network template."""

TRAIN = "    train: {iri: neuroml:spikeArray, events: {spikes: {event_type: preset_time, trigger_times: [25, 40]}}}\n"
"""A spike train at 25 and 40 ms on the measurement clock."""

TIMED_INPUTS = f"""
label: "a pulse and a spike train onto two resting cells"
dynamics:
  name: RS
  iri: neuroml:izhikevich2007Cell
  parameters:
    v0: {{value: -60, unit: mV}}
    C: {{value: 100, unit: pF}}
    k: {{value: 0.7, unit: nS_per_mV}}
    vr: {{value: -60, unit: mV}}
    vt: {{value: -40, unit: mV}}
    vpeak: {{value: 35, unit: mV}}
    a: {{value: 0.03, unit: per_ms}}
    b: {{value: -2, unit: nS}}
    c: {{value: -50, unit: mV}}
    d: {{value: 100, unit: pA}}
network:
  dynamics:
    syn: {{iri: neuroml:expTwoSynapse, parameters: {{gbase: {{value: 1, unit: nS}}, erev: {{value: 20, unit: mV}}, tauRise: {{value: 0.1, unit: ms}}, tauDecay: {{value: 3, unit: ms}}}}}}
    pulse: {{iri: neuroml:pulseGenerator, parameters: {{delay: {{value: 20, unit: ms}}, duration: {{value: 30, unit: ms}}, amplitude: {{value: 1, unit: nA}}}}}}
{TRAIN}  nodes:
    - {{id: 0, dynamics: RS}}
    - {{id: 1, dynamics: RS}}
    - {{id: 10, dynamics: pulse, record: false}}
    - {{id: 11, dynamics: train, record: false}}
  edges:
    - {{source: 10, target: 0}}
    - {{source: 11, target: 1, coupling: syn}}
integration: {{method: euler, step_size: 0.025, duration: 100.0, time_scale: ms}}
"""
"""Two Izhikevich cells at their exact rest (``v0 = vr``), one driven by a current pulse from 20 to 50 ms, the other by a spike train: a settle leaves them where they started, so the measured window is the same with or without one."""

RUN_ANCHORED = {
    "spikeGenerator": "{period: {value: 20, unit: ms}}",
    "spikeGeneratorRandom": "{minISI: {value: 10, unit: ms}, maxISI: {value: 30, unit: ms}}",
    "spikeGeneratorRefPoisson": "{averageRate: {value: 50, unit: Hz}, minimumISI: {value: 10, unit: ms}}",
}
"""Spike sources timed from the start of the run, with the parameters each needs."""

T_READING = """
label: "relaxation driven by a pulse written in t"
dynamics:
  name: TPulse
  parameters:
    tau: {value: 5.0}
    pulse_on: {value: 0.02}
    pulse_off: {value: 0.05}
  derived_variables:
    I: {equation: {rhs: "Piecewise((1.0, (t >= pulse_on) & (t < pulse_off)), (0.0, True))"}}
  state_variables:
    x: {equation: {lhs: "Derivative(x, t)", rhs: "(-x + I)/tau"}, initial_value: 0.0}
integration: {method: euler, step_size: 0.01, duration: 100.0, time_scale: ms}
"""
"""A custom model whose own equations read the clock ``t``, which LEMS reads in SI seconds: a pulse from 20 to 50 ms."""

RELAXATION = """
label: "relaxation rate model (gillespie)"
dynamics:
  name: Relax
  parameters:
    tau: {value: 0.5}
    drive: {value: 0.4}
  state_variables:
    E: {equation: {lhs: "Derivative(E, t)", rhs: "(-E + drive)/tau"}, initial_value: 0.4, domain: {lo: 0.0, hi: 1.0}}
  output: [E]
execution:
  backend: gillespie
  system_size: 50.0
  random_seed: 0
"""
"""The smallest model the Gillespie backend runs: one activity relaxing onto a constant gain."""


def _window(dt=0.25, duration=12.5, transient_time=0.0):
    """A window `BaseAdapter.get_integration_info` could return, with values no recipe or private default here uses."""
    n_transient, n_measured = round(transient_time / dt), round(duration / dt)
    return {
        "dt": dt,
        "duration": duration,
        "method": "Euler",
        "transient_time": transient_time,
        "total_duration": transient_time + duration,
        "n_transient": n_transient,
        "n_measured": n_measured,
    }


def _patch_window(monkeypatch, **window):
    monkeypatch.setattr(BaseAdapter, "get_integration_info", lambda self: _window(**window))


def _database_experiment(name):
    return SimulationExperiment.from_file(str(NEUROML_EXPERIMENTS / f"{name}.yaml"))


def _inline(recipe):
    return SimulationExperiment.from_string(recipe)


class TestNeuroMLRefuses:
    @pytest.mark.parametrize("standard_types", [False, True])
    def test_a_delayed_coupling_without_per_edge_delays(self, tmp_path, standard_types):
        """A delay is lowered from each edge's own ``delay``; nothing derives one from a tract length, so lengths alone are refused as a missing delay."""
        exp = _experiment(tmp_path, _DELAYED + EDGE.format(carrier="length"))
        with pytest.raises(ValueError, match="the NeuroML backend needs delay for the delayed coupling c_lin"):
            NeuroMLAdapter(exp).render_code(use_standard_types=standard_types)
        exp = _experiment(tmp_path, _DELAYED + EDGE.format(carrier="delay"))
        assert "<Simulation" in NeuroMLAdapter(exp).render_code(use_standard_types=standard_types)

    @pytest.mark.parametrize("standard_types", [False, True])
    @pytest.mark.parametrize("source", sorted(RUN_ANCHORED))
    def test_an_input_timed_from_the_run_start_behind_a_settle(self, source, standard_types):
        """A source with no onset to move would fire differently in the measured window behind a settle, so it is refused by name; without one it renders."""
        exp = _inline(
            TIMED_INPUTS.replace(TRAIN, f"    train: {{iri: neuroml:{source}, parameters: {RUN_ANCHORED[source]}}}\n")
        )
        assert "<Simulation" in NeuroMLAdapter(exp).render_code(use_standard_types=standard_types)
        exp.integration.transient_time = 10.0
        with pytest.raises(NotImplementedError, match=f"a {source} is timed from the start of the run"):
            NeuroMLAdapter(exp).render_code(use_standard_types=standard_types)

    @pytest.mark.parametrize("name", ["FitzHughNagumo_Ex9", "Generic2dOscillator_LEMS"])
    @pytest.mark.parametrize("standard_types", [False, True])
    def test_once_per_render(self, monkeypatch, name, standard_types):
        """Once on the standard-type path, the custom one, and a request for standard types the experiment has none of."""
        calls = []
        refuse = NeuroMLAdapter.refuse_unrenderable
        monkeypatch.setattr(NeuroMLAdapter, "refuse_unrenderable", lambda self: calls.append(1) or refuse(self))
        NeuroMLAdapter(_database_experiment(name)).render_code(use_standard_types=standard_types)
        assert len(calls) == 1


class TestNeuroMLSettle:
    """A declared settle is the head of the LEMS run: the run spans both windows, every onset and every reading of ``t`` moves a settle later, and a zero settle leaves no trace."""

    @staticmethod
    def _settled(recipe, settle):
        exp = _inline(recipe)
        exp.integration.transient_time = settle
        return exp

    @pytest.mark.parametrize("standard_types", [False, True])
    def test_a_zero_settle_leaves_no_trace(self, standard_types):
        xml = NeuroMLAdapter(self._settled(T_READING, 0.0)).render_code(use_standard_types=standard_types)
        assert "SETTLE" not in xml and re.search(r'<Simulation [^>]*length="100.0ms"', xml)

    @pytest.mark.parametrize("standard_types", [False, True])
    def test_the_run_spans_the_settle_and_the_measured_window(self, standard_types):
        exp = _database_experiment("FitzHughNagumo_Ex9")
        exp.integration.transient_time = 10.0
        xml = NeuroMLAdapter(exp).render_code(use_standard_types=standard_types)
        assert re.search(r'<Simulation [^>]*length="210.0s" step="0.01s"', xml)

    def test_every_onset_moves_a_settle_later(self):
        xml = NeuroMLAdapter(self._settled(TIMED_INPUTS, 50.0)).render_code(use_standard_types=True)
        assert re.search(r'<pulseGenerator [^>]*delay="70(\.0)? ms"', xml)
        assert re.findall(r'<spike id="\d+" time="([\d.]+) ms"', xml) == ["75.0", "90.0"]

    @pytest.mark.parametrize("settle", [0.0, 2.0])
    def test_the_measured_window_opens_one_step_after_the_settle(self, settle):
        """LEMS records the initial state at the run start; with or without a settle it stays off `.data`, which holds ``duration / step`` samples from one step in, as on every other backend."""
        import numpy as np
        import xarray as xr

        from tvbo.adapters.neuroml import LemsClock, _cut_the_settle

        seconds = np.arange(0, 6.0 + settle + 0.5, 0.5) * 1e-3
        cut, n_settle = _cut_the_settle(
            xr.DataArray(seconds, dims=["time"], coords={"time": seconds}), LemsClock(0.5, 6.0 + settle, settle, "ms")
        )
        measured = cut.time.values[n_settle:]
        assert n_settle == settle / 0.5 + 1 and len(measured) == 12
        assert measured[0] == pytest.approx(0.5e-3) and measured[-1] == pytest.approx(6.0e-3)

    def test_t_reads_the_measurement_clock(self):
        xml = NeuroMLAdapter(self._settled(T_READING, 30.0)).render_code()
        assert '<Constant name="SETTLE" dimension="time" value="30.0ms"/>' in xml
        condition = re.search(r'<Case condition="([^"]+)"', xml).group(1)
        assert condition.count("(t - SETTLE)") == 2 and not re.search(r"(?<!\()\bt\b(?! - SETTLE)", condition), condition


def test_every_input_is_placed_behind_a_settle_or_refused():
    """Each NeuroML input either declares the onset a settle moves, carries timed children, runs stationary, or is refused; the refused ones are exactly the sources timed from the run start."""
    from tvbo.adapters.smallscale.lowering import CURRENT_INPUT_TYPES, EVENT_SOURCE_TYPES, settle_refusal

    refused = {kind for kind in CURRENT_INPUT_TYPES | EVENT_SOURCE_TYPES if settle_refusal(kind)}
    assert refused == set(RUN_ANCHORED)
    assert settle_refusal("izhikevich2007Cell") is None


class TestOneTimeUnitReader:
    """Every LEMS builder reads the clock's unit through one reader, so a spelling one of them normalises is normalised by all."""

    @pytest.mark.parametrize("standard_types", [False, True])
    def test_a_spelled_out_unit(self, standard_types):
        exp = _database_experiment("FitzHughNagumo_Ex9")
        exp.integration.time_unit = "second"
        xml = NeuroMLAdapter(exp).render_code(use_standard_types=standard_types)
        assert re.search(r'<Simulation [^>]*length="200.0s" step="0.01s"', xml)

    def test_a_unit_lems_cannot_name(self):
        exp = _database_experiment("FitzHughNagumo_Ex9")
        exp.integration.time_unit = "min"
        with pytest.raises(ValueError, match="cannot emit a clock in 'min'"):
            NeuroMLAdapter(exp).render_code()


class TestOneWindowReader:
    @pytest.mark.parametrize(
        "recipe, standard",
        [
            pytest.param(None, "custom", id="custom-lems"),
            pytest.param(None, "is_fhn", id="standard-fhn"),
            pytest.param(f"dynamics:\n  name: IaF{IAF_CELL}", "cell", id="standard-cell"),
            pytest.param(f"dynamics:{LIF_CELL}", "is_hier_custom", id="hierarchical-custom"),
            pytest.param(IAF_NETWORK, "is_network", id="standard-network"),
        ],
    )
    def test_every_lems_builder(self, monkeypatch, recipe, standard):
        if recipe is None:
            exp = _database_experiment("Generic2dOscillator_LEMS" if standard == "custom" else "FitzHughNagumo_Ex9")
        else:
            exp = _inline(recipe)
        _patch_window(monkeypatch)
        if standard == "custom":
            ctx = build_lems_context(exp)
        else:
            ctx = build_std_lems_context(exp)
            reached = {kind for kind in ("is_fhn", "is_hier_custom", "is_network") if ctx and ctx.get(kind)}
            assert ctx is not None and reached == ({standard} - {"cell"}), "the recipe reaches the builder it names"
        if standard == "is_hier_custom":
            assert (ctx["sim_step"], ctx["sim_length"]) == ("0.25ms", "12.5ms")
        else:
            assert (ctx["dt"], ctx["length"]) == (0.25, 12.5)
        xml = NeuroMLAdapter(exp).render_code(use_standard_types=standard != "custom")
        clock = re.findall(r'<Simulation [^>]*length="([\d.]+)([a-z]+)" step="([\d.]+)([a-z]+)"', xml)
        assert len(clock) == 1 and clock[0][0::2] == ("12.5", "0.25") and clock[0][1] == clock[0][3], clock

    def test_brian2(self, monkeypatch):
        pytest.importorskip("brian2")
        from tvbo.adapters.brian2 import Brian2Adapter

        _patch_window(monkeypatch, transient_time=2.5)
        ctx = Brian2Adapter(_inline(f"dynamics:{LIF_CELL}")).prepare_context()
        assert (ctx["dt_ms"], ctx["transient_ms"], ctx["measured_ms"], ctx["total_ms"]) == (0.25, 2.5, 12.5, 15.0)

    def test_gillespie(self, monkeypatch):
        """The settle is simulated ahead of the measured window and cut: it stays on `.transient` and ends at t = 0, and `.data` opens one step later with `duration / dt` samples."""
        _patch_window(monkeypatch, transient_time=2.5)
        sim = _inline(RELAXATION).run(format="gillespie").integration
        measured, settle = sim.data.time.values, sim.transient.data.time.values
        assert len(measured) == 50 and measured[0] == pytest.approx(0.25) and measured[-1] == pytest.approx(12.5)
        assert len(settle) == 10 and settle[0] == pytest.approx(-2.25) and settle[-1] == 0.0
