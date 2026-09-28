"""NeuroML refuses what every other backend refuses, and the small-scale backends read their window from one place.

`NeuroMLAdapter.render_code` passes the declaration through `refuse_unrenderable` before it builds a context, as `BaseAdapter.render_context` does for every other backend, so a delayed coupling with no per-edge delay to lower, and a settle LEMS cannot cut, are refused instead of emitted as well-formed LEMS for the rest.

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
    def test_a_declared_settle(self, standard_types):
        exp = _database_experiment("FitzHughNagumo_Ex9")
        exp.integration.transient_time = 10.0
        with pytest.raises(NotImplementedError, match="the NeuroML backend has no settle"):
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
            assert (ctx["dt"], ctx["duration"]) == (0.25, 12.5)
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
        """The settle is simulated ahead of the measured window and cut: `.data` opens on the state at t = 0 and holds `duration / dt` steps after it, the settle stays on `.transient` at negative times."""
        _patch_window(monkeypatch, transient_time=2.5)
        sim = _inline(RELAXATION).run(format="gillespie").integration
        measured, settle = sim.data.time.values, sim.transient.data.time.values
        assert len(measured) == 51 and measured[0] == 0.0 and measured[-1] == pytest.approx(12.5)
        assert len(settle) == 10 and settle[0] == pytest.approx(-2.5) and settle[-1] == pytest.approx(-0.25)
