"""Native Brian2 backend for small-scale spiking networks.

Consumes the shared small-scale lowering core (:mod:`tvbo.adapters.smallscale`) and emits a Brian2 point-neuron network. The maths is printed through the shared SymPy printer (``render_expression(..., format="brian2")``); this adapter adds only the Brian2 role vocabulary and the synapse rendering.

Two connectivity lowerings, chosen per edge by the ``connectivity`` rule:

**all_to_all → O(N) population sums.** Every post-synaptic neuron sees the same sum over pre-synaptic gating, so the gate lives on the *pre-synaptic* neuron and a size-1 "hub" `NeuronGroup` accumulates the population sum once (via a ``(summed)`` `Synapses`), read by every post-synaptic neuron through a ``linked_var``. This is the hand-written Deco 2014 `deco_column.py` structure — it runs the 160E+40I column in seconds where the enumerated LEMS network needs ~190 s per 100 ms in jLEMS.

**random / one_to_one → real sparse `Synapses`.** A genuinely sparse projection cannot be a single population sum (each target sees a different subset), so it is emitted as a Brian2 `Synapses` with ``connect(p=…)`` / ``connect(j='i')``. Following the canonical Brian2 idioms: the delivered conductance decays on the *post-synaptic* `NeuronGroup` (``dg/dt=-g/tau``) and is incremented event-driven by ``on_pre`` (spike-gated, not summed every step); short-term-plasticity state (u, x) lives *on the synapse* as ``(event-driven)`` variables, mutated in ``on_pre`` in the recipe's declared order, so any facilitation/depression convention is honoured per connection.

Supported synapse forms:
  * ``neuroml:expOneSynapse`` — single-exponential conductance (AMPA, GABA), either lowering;
  * a custom conductance synapse extending ``baseConductanceBasedSynapse`` whose current is *linear* in a single gate: all_to_all lowers any such gate (e.g. the saturating NMDA with Mg block); the sparse path additionally requires that gate to be a pure decaying conductance (``dg/dt=-g/tau``), with the remaining state variables the per-synapse STP;
  * ``neuroml:poissonFiringSynapse`` — independent Poisson background → `PoissonInput`.

Anything outside this set (non-Poisson spike sources, constant-current inputs, a summed-gate current nonlinear in its gate, or a sparse synapse whose gate is not a pure decay) raises a clear ``NotImplementedError`` rather than mis-simulating.
"""

from __future__ import annotations

from tvbo.adapters.base import BaseAdapter
from tvbo.adapters.smallscale.lowering import (
    EVENT_SOURCE_TYPES,
    POISSON_INPUT_TYPES,
    classify_node_role,
    group_nodes_by_dynamics,
    nml_type,
    safe_id,
)
from tvbo.codegen.code import render_expression
from tvbo.utils import edge_param, noise_sigma, normalize_params
from tvbo.utils.units import time_unit_factor, unit_multiplier


def _shift_onto_integration_clock(quantity, transient_ms):
    """A declared time, moved from the measurement clock onto the one Brian2 counts on.

    Brian2's ``t`` opens at the start of integration, which is the start of the settle; a recipe declares its onsets against the measured window, which opens ``transient_ms`` later. The shift is applied in the quantity's OWN unit, so a delay declared in seconds and a settle counted in milliseconds compose exactly rather than through whichever of the two the caller happened to write.
    """
    value, unit = quantity
    if not transient_ms:
        return quantity
    multiplier = unit_multiplier(unit)
    if multiplier is None:
        raise ValueError(
            f"a declared onset in {unit!r} cannot be placed on the integration clock: the unit is "
            "not curated, so the settle prepended to the measured window cannot be converted into it."
        )
    return (float(value) + transient_ms / 1000.0 / float(multiplier), unit)


# ── Brian2 role vocabulary ────────────────────────────────────────────
_PULSE_TYPES = frozenset({"pulseGenerator", "pulseGeneratorDL"})
"""Timed current pulses, which lower to a rectangular current window (delay, duration, amplitude)."""

_BRIAN2_ROLE_VOCAB = {
    "current_input": _PULSE_TYPES | {"sineGenerator", "rampGenerator"},
    "event_source": POISSON_INPUT_TYPES | EVENT_SOURCE_TYPES,
}
"""The inputs Brian2 tells apart from cell populations. Poisson inputs lower to a `PoissonInput` and pulses to a current window; sine and ramp generators and the other spike sources are recognised so that they raise, rather than being mistaken for a cell population with no membrane `v`."""
_EXP_ONE_TYPES = frozenset({"expOneSynapse"})

# Unit strings that denote a dimensionless (unitless) parameter.
_DIMENSIONLESS_UNITS = frozenset({"", "1", "dimensionless", "none"})

# Observation probe for recording synapse-internal state (u, x): how many source neurons to sample for the population trace, and how coarsely to record it. STP variables evolve on the tauF/tauD (100s of ms) scale, so a 1 ms record step is ample and keeps the trace small.
_PROBE_SAMPLE = 200
_PROBE_RECORD_DT_MS = 1.0


def _unit_of(param):
    """Lower-cased unit string of a ``(value, unit)`` parameter tuple ('' if unitless)."""
    try:
        return str(param[1]).strip().lower()
    except (TypeError, IndexError):
        return ""


def _sample_indices(n, k):
    """Up to ``k`` distinct evenly-spaced indices in ``[0, n)`` (all of them if k >= n)."""
    if k >= n:
        return list(range(n))
    if k <= 1:
        return [0]
    return sorted({int(round(i * (n - 1) / (k - 1))) for i in range(k)})


# A generated Brian2 script states every duration in ms, so declared times convert onto it.
_BRIAN2_CLOCK = {"time_unit": "ms"}

# TVBO unit name -> Brian2 unit name (as it appears in a generated script). A unit absent here is rejected (fail loud) rather than silently dropped.
_UNIT_TO_BRIAN2 = {
    "nS": "nS",
    "pS": "psiemens",
    "uS": "usiemens",
    "siemens": "siemens",
    "mV": "mV",
    "V": "volt",
    "volt": "volt",
    "ms": "ms",
    "s": "second",
    "us": "usecond",
    "second": "second",
    "nF": "nF",
    "pF": "pfarad",
    "uF": "ufarad",
    "farad": "farad",
    "nA": "nA",
    "pA": "pamp",
    "amp": "amp",
    "Hz": "Hz",
    "kHz": "kHz",
    "per_ms": "1/ms",
    "per_s": "1/second",
    "dimensionless": "",
    "": "",
}


def _params(dyn):
    """``{name: (value, unit)}`` for a Dynamics/edge's parameters."""
    out = {}
    for name, p in normalize_params(getattr(dyn, "parameters", None)).items():
        val = getattr(p, "value", p)
        unit = str(getattr(p, "unit", "") or "")
        out[str(name)] = (val, unit)
    return out


def _membrane_noise_sigma(v_sv):
    """The Gaussian white-noise amplitude on a membrane variable, as ``(value, unit)``.

    The standard-deviation scale σ of an additive Gaussian white-noise current (the Mongillo ``σ_ext·η(t)`` external drive), with the unit its declaration carries.
    Returns None when the variable declares no noise.
    """
    nz = getattr(v_sv, "noise", None)
    sigma = noise_sigma(nz)
    if not sigma:
        return None
    # The unit belongs to whichever spelling carried the value, not always `intensity`.
    source = normalize_params(getattr(nz, "parameters", None)).get("sigma") or getattr(nz, "intensity", None)
    return float(sigma), str(getattr(source, "unit", "") or "mV")


def _edge_weight(edge):
    """The per-connection ``weight`` on an edge (1.0 if absent or valueless)."""
    return _edge_param(edge, "weight", 1.0)


def _edge_param(edge, name, default):
    """A named scalar off an edge as a float (or ``default``); see `tvbo.utils.edge_param`."""
    val = edge_param(edge, name)
    return float(val) if val is not None else default


def _brian2_unit(unit):
    """Brian2 unit expression for a TVBO unit name, or raise for an unknown one."""
    if unit not in _UNIT_TO_BRIAN2:
        raise ValueError(f"Brian2 backend has no unit mapping for {unit!r}; add it to _UNIT_TO_BRIAN2.")
    return _UNIT_TO_BRIAN2[unit]


def _brian2_const(value, unit):
    """A Brian2 constant literal for a script, e.g. ``0.2 * nS`` or ``0.5 / ms``."""
    bu = _brian2_unit(unit)
    if not bu:
        return f"{value}"
    if bu.startswith("1/"):
        return f"{value} / {bu[2:]}"
    return f"{value} * {bu}"


def _dict_literal(entries):
    """A dict literal for a script, ``{'key': value, ...}``, each value written as the code it already is."""
    return "{" + ", ".join(f"'{k}': {v}" for k, v in entries) + "}"


def _ns_literal(ns):
    """A namespace ``{name: (value, unit)}`` as a script literal of Brian2 constants."""
    return _dict_literal((k, _brian2_const(v, u)) for k, (v, u) in ns.items())


def _monitor_literal(populations):
    """The script's spike monitors of *populations*, as ``{'pop': mon_pop, ...}``."""
    return _dict_literal((p, f"mon_{p}") for p in populations)


# ── Symbolic synapse reduction ────────────────────────────────────────


def _symbols(names):
    """``{name: Symbol}`` for a synapse's declared names and the membrane ``v``: the locals its expressions parse against."""
    import sympy as sp

    return {n: sp.Symbol(n) for n in [*names, "v"]}


def _parse(rhs, syms):
    """A declared right-hand side as a SymPy expression, its names bound to *syms* rather than to SymPy built-ins."""
    import sympy as sp

    return sp.sympify(str(rhs), locals=syms)


def _inline_derived(dvs, syms):
    """The synapse current ``i`` with every derived variable it references substituted in, to a fixed point."""
    dv_exprs = {n: _parse(dv.equation.rhs, syms) for n, dv in dvs.items()}
    i_expr = dv_exprs["i"]
    for _ in range(len(dv_exprs) + 1):
        sub = {syms[n]: e for n, e in dv_exprs.items() if n != "i" and syms[n] in i_expr.free_symbols}
        if not sub:
            break
        i_expr = i_expr.subs(sub)
    return i_expr


def _linear_gate(i_expr, svs, syms):
    """The state variables the current ``i`` references, and whether ``i`` is linear in them.

    Returns ``(gates, linear)``. *linear* holds only for exactly one gate in which ``i`` is linear with no offset, ``i = coeff(v) * gate``: the form both a population sum and a delivered conductance need.
    """
    import sympy as sp

    gates = [n for n in svs if syms[n] in i_expr.free_symbols]
    if len(gates) != 1:
        return gates, False
    g = syms[gates[0]]
    return gates, not (sp.simplify(sp.diff(i_expr, g)).has(g) or sp.simplify(i_expr.subs(g, 0)) != 0)


def _affect_assignments(events, syms):
    """Every ``lhs = rhs`` the synapse's events assign, in declared order, as ``(lhs, parsed rhs)``."""
    assignments = []
    for ev in events.values():
        affect = getattr(getattr(ev, "affect", None), "rhs", None)
        if not affect:
            continue
        for piece in str(affect).split(";"):
            if "=" in piece:
                lhs, rhs = (part.strip() for part in piece.split("=", 1))
                assignments.append((lhs, _parse(rhs, syms)))
    return assignments


def _event_driven_lines(names, svs, syms):
    """The per-synapse ``(event-driven)`` equations of the state variables *names*, with their initial values and every name the equations reference."""
    lines, init, refs = [], {}, set()
    for n in names:
        rhs = _parse(svs[n].equation.rhs, syms)
        refs |= {s.name for s in rhs.free_symbols}
        lines.append(f"d{n}/dt = {render_expression(str(rhs), format='brian2')} : 1 (event-driven)")
        iv = getattr(svs[n], "initial_value", None)
        if iv is not None:
            init[n] = float(iv)
    return lines, init, refs


def _on_pre(steps, deliver):
    """One edge's ``on_pre`` code: each delivered increment written by *deliver*, each synapse-local update as it stands, in declared order."""
    return "\n".join(deliver(increment) if increment is not None else update for increment, update in steps)


def _summed_form(syn, sparams):
    """The edge-independent reduction of an all-to-all custom conductance synapse.

    Its parsed gate ODEs and spike-event assignments, and the current ``i`` with the derived variables inlined, required to be linear in its one gate because an all-to-all conductance is delivered as a population sum.
    """
    svs, dvs = syn.state_variables, syn.derived_variables
    if not svs:
        raise NotImplementedError(f"Synapse {getattr(syn, 'name', syn)!r}: no state variables to render.")
    if "i" not in dvs:
        raise NotImplementedError(f"Synapse {getattr(syn, 'name', syn)!r}: no current derived variable 'i'.")
    syms = _symbols([*svs, *dvs, *sparams])
    odes = {n: _parse(sv.equation.rhs, syms) for n, sv in svs.items()}
    affects = _affect_assignments(syn.events, syms)
    i_expr = _inline_derived(dvs, syms)
    summed, linear = _linear_gate(i_expr, svs, syms)
    if len(summed) != 1:
        raise NotImplementedError(
            f"Synapse {getattr(syn, 'name', syn)!r}: the current must reference exactly one gating "
            f"variable to lower to a population sum, found {summed}."
        )
    if not linear:
        raise NotImplementedError(
            f"Synapse {getattr(syn, 'name', syn)!r}: current is not linear in the gating variable "
            f"{summed[0]!r}; an all-to-all population sum requires linearity."
        )
    return {"syms": syms, "odes": odes, "affects": affects, "i": i_expr, "gate": summed[0]}


def _delta_form(syn, sparams, edge_idx):
    """The edge-independent reduction of an instantaneous (delta) PSC synapse.

    Its per-synapse STP equations, and its spike event as ``on_pre`` steps for `_on_pre`: the membrane jump kept as its bare increment, for each edge to scale by its own weight. The jump is delivered as ``v_post += weight * (increment) * mV``, the weight carrying the mV amplitude, so the increment must be dimensionless: one that holds a residual ``v`` or a parameter with a voltage or current unit is rejected here, as the membrane-noise path does, rather than failing deep inside Brian2.
    """
    import sympy as sp

    svs = syn.state_variables
    syms = _symbols([*svs, *sparams])
    # Short-term-plasticity vars (all state vars — a delta synapse has no conductance gate) become per-synapse (event-driven) equations.
    model_lines, init, syn_ref = _event_driven_lines(list(svs), svs, syms)
    steps, delivered = [], False
    for lhs, expr in _affect_assignments(syn.events, syms):
        if lhs == "v":  # deliver the membrane jump
            incr = sp.simplify(expr - syms["v"])
            unitful = sorted(
                s.name
                for s in incr.free_symbols
                if s.name == "v" or (s.name in sparams and _unit_of(sparams[s.name]) not in _DIMENSIONLESS_UNITS)
            )
            if unitful:
                raise NotImplementedError(
                    f"Delta synapse (edge {edge_idx}): the membrane-jump increment "
                    f"'{incr}' is not dimensionless (unit-valued: {unitful}). The jump is "
                    f"delivered as 'v_post += weight*(increment)*mV', so put the mV "
                    f"amplitude in the edge weight and keep the event increment "
                    f"dimensionless (e.g. 'v = v + u*x')."
                )
            syn_ref |= {s.name for s in incr.free_symbols}
            steps.append((render_expression(str(incr), format="brian2"), None))
            delivered = True
        elif lhs in svs:  # synapse-local STP update
            syn_ref |= {s.name for s in expr.free_symbols}
            steps.append((None, f"{lhs} = {render_expression(str(expr), format='brian2')}"))
    if not delivered:
        raise NotImplementedError(
            f"Delta synapse (edge {edge_idx}): its spike event must assign the post-synaptic "
            f"membrane 'v' (e.g. 'v = v + J*u*x') to deliver a jump; none found."
        )
    syn_consts = {p: sparams[p] for p in sparams if p in syn_ref}
    return {"model": "\n".join(model_lines), "steps": steps, "syn_consts": syn_consts, "init": init}


def _sparse_form(syn, sparams, edge_idx):
    """The edge-independent reduction of a custom conductance synapse on a sparse projection.

    The current ``i`` must be linear in one gate whose ODE is a pure decay; the remaining state is per-synapse STP. Returns the current and the gate ODE for each edge to rename onto its own post-synaptic conductance, the STP equations, and the spike event as ``on_pre`` steps for `_on_pre`, the conductance increment kept bare for each edge to scale by its own weight.
    """
    import sympy as sp

    svs, dvs = syn.state_variables, syn.derived_variables
    if "i" not in dvs:
        raise NotImplementedError(f"Sparse synapse (edge {edge_idx}): no current derived variable 'i'.")
    syms = _symbols([*svs, *dvs, *sparams])
    # Inline derived variables into i; the single state var in i is the gate g (linear).
    i_expr = _inline_derived(dvs, syms)
    gate, linear = _linear_gate(i_expr, svs, syms)
    if len(gate) != 1:
        raise NotImplementedError(
            f"Sparse synapse (edge {edge_idx}): the current must reference exactly one gating variable, found {gate}."
        )
    g = gate[0]
    gsym = syms[g]
    if not linear:
        raise NotImplementedError(f"Sparse synapse (edge {edge_idx}): current is not linear in the gate {g!r}.")

    # The gate ODE must be a pure decay -g/tau (params only) — sparse delivery accumulates onto a decaying post-synaptic conductance; a saturating gate (e.g. NMDA) cannot.
    g_ode = _parse(svs[g].equation.rhs, syms)
    param_syms = {syms[p] for p in sparams}
    if (
        sp.simplify(g_ode.subs(gsym, 0)) != 0
        or sp.simplify(sp.diff(g_ode, gsym)).has(gsym)
        or not (g_ode.free_symbols - {gsym}) <= param_syms
    ):
        raise NotImplementedError(
            f"Sparse synapse (edge {edge_idx}): gate {g!r} ODE {str(svs[g].equation.rhs)!r} "
            f"is not a pure decay -{g}/tau; the sparse path needs a decaying post-synaptic conductance "
            f"(use all_to_all for a saturating summed gate)."
        )

    # Synapse-side: the OTHER state vars (u, x) are per-synapse (event-driven).
    stp_vars = [n for n in svs if n != g]
    model_lines, init, syn_ref = _event_driven_lines(stp_vars, svs, syms)

    # The spike event, in the recipe's declared order: the g-increment is delivered to the post-synaptic conductance, u/x updates run on the synapse.
    steps = []
    for lhs, expr in _affect_assignments(syn.events, syms):
        syn_ref |= {s.name for s in expr.free_symbols} - {g}
        if lhs == g:  # deliver the increment (rhs - g)
            steps.append((render_expression(str(sp.simplify(expr - gsym)), format="brian2"), None))
        elif lhs in stp_vars:  # synapse-local STP update
            steps.append((None, f"{lhs} = {render_expression(str(expr), format='brian2')}"))
    syn_consts = {p: sparams[p] for p in sparams if p in syn_ref}
    return {
        "syms": syms,
        "gate": g,
        "i": i_expr,
        "g_ode": g_ode,
        "model": "\n".join(model_lines),
        "steps": steps,
        "syn_consts": syn_consts,
        "init": init,
    }


class Brian2Adapter(BaseAdapter):
    """Render/run a small-scale spiking network natively in Brian2."""

    TEMPLATE = "brian2/tvbo-brian2-experiment.py.mako"

    def run(self, seed=None, record_v=False, codegen_target="numpy", **kwargs):
        """Build and run the network in Brian2, returning an ExperimentResult.

        Population firing rates (from Brian2 ``SpikeMonitor``) are the primary output — the exact quantity the Deco 2014 replication targets — and are exposed both as ``result.integration.observations.firing_rate_<pop>`` and, raw, under ``result._extras``.

        ``codegen_target`` defaults to ``"numpy"`` (no C compilation, portable); pass ``"cython"`` for the faster compiled path where the toolchain allows.
        """
        import brian2
        import numpy as np
        from brian2 import ms

        from tvbo.data.types import ExperimentResult, SimulationResult

        if "settle_ms" in kwargs:
            raise TypeError(
                "settle_ms is gone: the settle is the recipe's own `integration.transient_time`, prepended to the "
                "measured `duration`. A caller-side override is what let run() and render() measure different windows."
            )
        if codegen_target:
            brian2.prefs.codegen.target = codegen_target
        model = self.prepare_context()
        # An explicit seed argument wins; otherwise fall back to the seed the build description resolved from execution.random_seed — the SAME value the rendered script emits, so run() and render() build seed-identical random connectivity from the recipe alone.
        if seed is None:
            seed = model.get("seed")
        net, meta = _instantiate(model, seed=seed, record_v=record_v)
        if model.get("ramp"):
            return self._run_ramp(net, meta, model)
        # The settle is the offset of the measurement clock, which is what every time payload below is reported on.
        settle, measured = model["transient_ms"], model["measured_ms"]
        net.run(model["total_ms"] * ms)

        rates, spikes = {}, {}
        for name, mon in meta["spike_monitors"].items():
            n = model["sizes"][name]
            t = np.asarray(mon.t / ms)
            counts = int((t >= settle).sum())
            window_s = measured / 1000.0
            rates[name] = counts / (window_s * n) if window_s > 0 and n else 0.0
            # On the measurement clock, like every other time axis a run reports: the settle carries negative timestamps and t = 0 opens the measured window.
            spikes[name] = {"i": np.asarray(mon.i), "t_ms": t - settle}

        integration = SimulationResult(observations={f"firing_rate_{k}": v for k, v in rates.items()})
        result = ExperimentResult(
            integration=integration, name=getattr(self.experiment, "label", None), source=self.experiment
        )
        result._extras["rates"] = rates
        result._extras["spikes"] = spikes
        result._extras["sizes"] = dict(model["sizes"])
        result._extras["duration_ms"] = measured
        result._extras["dt_ms"] = model["dt_ms"]
        # Written into the saved Dataset's attrs, so a file on disk says which clock its times are on rather than leaving a reader to infer it from a settle it cannot see.
        result._extras["transient_ms"] = settle
        if record_v and meta.get("state_monitors"):
            result._extras["v"] = {p: np.asarray(m.v / meta["v_unit"]) for p, m in meta["state_monitors"].items()}
            any_mon = next(iter(meta["state_monitors"].values()))
            result._extras["t_ms"] = np.asarray(any_mon.t / ms) - settle
        # Recorded synapse-internal state (u, x): the population mean over the probed sample, the continuous trace the figures show as MEASURED (not reconstructed from spike trains).
        syn_state = {}
        for pinfo in meta.get("probe_monitors", {}).values():
            mon = pinfo["mon"]
            vals = {v: np.asarray(getattr(mon, v)) for v in pinfo["vars"]}  # [n_sample, n_time]
            syn_state[pinfo["key"]] = {
                "t_ms": np.asarray(mon.t / ms) - settle,
                "vars": {v: arr.mean(axis=0) for v, arr in vals.items()},  # population mean [n_time]
                "source": pinfo["source"],
                "n_sample": int(next(iter(vals.values())).shape[0]) if vals else 0,
            }
        if syn_state:
            result._extras["synapse_state"] = syn_state
        return result

    def _run_ramp(self, net, meta, model):
        """Run the declared continuation ramp and return its per-step population rates.

        The state is never re-initialised: each declared value is written into the namespace constant that carries the labelled projection's weight, the declared settle lets the network relax quasi-statically, and the spikes of the measured window that follows give that step's rate. Listing the parameter up and back down therefore traces both branches of a hysteresis loop in one run, which is what makes the loop evidence of bistability rather than of two different runs. A point costs `transient_time + duration`, of which `duration` is measured — the same split every other backend uses.
        """
        import numpy as np
        from brian2 import ms

        from tvbo.data.types import Bunch, ExperimentResult, ExplorationResult, SimulationResult

        ramp = model["ramp"]
        step, settle = model["measured_ms"], model["transient_ms"]
        groups = {o.name: o for o in net.objects if hasattr(o, "namespace")}
        mons = {n: meta["spike_monitors"][n] for n in ramp["populations"]}
        window_s, rates = step / 1000.0, {n: [] for n in mons}
        for value in ramp["values"]:
            for pop, key in ramp["handles"]:
                groups[pop].namespace[key] = float(value)
            if settle:
                net.run(settle * ms)
            counted = {n: int(m.num_spikes) for n, m in mons.items()}
            net.run(step * ms)
            for n, mon in mons.items():
                size = model["sizes"][n]
                # An empty population has no rate rather than a division; the window itself is validated with the ramp.
                rates[n].append((int(mon.num_spikes) - counted[n]) / (window_s * size) if size else 0.0)
        axis = Bunch(name=ramp["parameter"], explored_values=np.asarray(ramp["values"]), n=len(ramp["values"]))
        expl = ExplorationResult(
            name=ramp["name"],
            axes=[axis],
            observations={f"firing_rate_{n}": np.asarray(v) for n, v in rates.items()},
        )
        result = ExperimentResult(
            integration=SimulationResult(observations={f"firing_rate_{n}": v[-1] for n, v in rates.items()}),
            explorations={ramp["name"]: expl},
            name=getattr(self.experiment, "label", None),
            source=self.experiment,
        )
        result._extras["ramp"] = {
            "parameter": ramp["parameter"],
            "values": np.asarray(ramp["values"]),
            "rates": {n: np.asarray(v) for n, v in rates.items()},
        }
        return result

    # ── Analysis: declarative network → Brian2 build description ────────

    def prepare_context(self):
        """Reduce the experiment to a backend-neutral Brian2 build description.

        Returns a dict the template renders and ``_instantiate`` builds: ``populations`` (per cell pop: eqs data, namespace, poisson, size) and their ``sizes``, ``hubs`` (summed-gate accumulators), ``synapses`` and ``probes`` (sparse projections and their observation probes), the clock (``dt_ms``, the settle ``transient_ms``, the ``measured_ms`` window after it and their sum ``total_ms``), and the validated continuation ``ramp`` or None.
        """
        exp = self.experiment
        network = exp.network
        dyn_lib = getattr(network, "dynamics", None) or {}
        nodes = getattr(network, "nodes", None) or []
        edges = getattr(network, "edges", None) or []

        integration = getattr(exp, "integration", None)
        ts_factor = float(time_unit_factor((network, integration, exp), _BRIAN2_CLOCK))
        dt_ms = float(getattr(integration, "step_size", 0.02) or 0.02) * ts_factor
        measured_ms = float(getattr(integration, "duration", 1000.0) or 1000.0) * ts_factor
        transient_ms = float(getattr(integration, "transient_time", 0.0) or 0.0) * ts_factor

        default_name = getattr(exp.dynamics, "name", None) or "dynamics"
        groups = group_nodes_by_dynamics(nodes, default_name)

        # Classify node roles and record per-node populations / sources.
        cell_pop_of_node = {}  # node_id -> pop name
        poisson_of_node = {}  # node_id -> (dyn_name, dyn_obj)
        pulse_of_node = {}  # node_id -> (dyn_name, dyn_obj)  (timed current pulse)
        populations = {}  # pop name -> descriptor
        for dyn_name, gnodes in groups.items():
            # An experiment with no node library — one population declared by `dynamics:` alone — names its cell type by the default name.
            dyn_obj = dyn_lib.get(dyn_name) or (exp.dynamics if dyn_name == default_name else None)
            role, nml = classify_node_role(dyn_name, dyn_obj, _BRIAN2_ROLE_VOCAB)
            if role == "event_source":
                if nml not in POISSON_INPUT_TYPES:
                    raise NotImplementedError(
                        f"Brian2 backend does not yet handle the spike source {nml!r} "
                        f"(dynamics {dyn_name!r}); only Poisson backgrounds are supported."
                    )
                for node in gnodes:
                    poisson_of_node[getattr(node, "id", 0)] = (dyn_name, dyn_obj)
                continue
            if role == "current_input":
                # A deterministic timed current pulse (delay/duration/amplitude) — the declarative loading / nonspecific-readout drive. Recorded here and lowered to a time-gated current term on its target population(s) in the edge loop.
                if nml not in _PULSE_TYPES:
                    raise NotImplementedError(
                        f"Current input {nml!r} (dynamics {dyn_name!r}) is not yet wired for "
                        f"Brian2; only the timed pulse {sorted(_PULSE_TYPES)} is supported."
                    )
                for node in gnodes:
                    pulse_of_node[getattr(node, "id", 0)] = (dyn_name, dyn_obj)
                continue
            # A cell population — ONE per node, so same-dynamics nodes in different areas stay separate (a two-area network keeps its two E pools distinct and its long-range projection connects only the right pair). The clean dynamics name is kept when a single node uses that dynamics; when several do, the node id disambiguates (ExcitatoryCell_2 / ExcitatoryCell_6) so the single-column output keys are unchanged.
            v_sv = (dyn_obj.state_variables).get("v")
            if v_sv is None:
                raise NotImplementedError(
                    f"Cell dynamics {dyn_name!r} declares no membrane variable 'v'; cannot build a NeuronGroup."
                )
            v_rhs = render_expression(str(v_sv.equation.rhs), format="brian2")
            noise_sigma = _membrane_noise_sigma(v_sv)  # (value, unit) or None
            base = safe_id(dyn_name)
            for node in gnodes:
                node_id = getattr(node, "id", 0)
                pop = base if len(gnodes) == 1 else f"{base}_{node_id}"
                populations[pop] = {
                    "name": pop,
                    "dyn_name": dyn_name,
                    "dyn": dyn_obj,
                    "node_id": node_id,
                    "size": int(getattr(node, "size", 1) or 1),
                    "v_rhs": v_rhs,
                    "noise_sigma": noise_sigma,  # Gaussian white-noise amplitude on v, or None
                    "cell_params": _params(dyn_obj),
                    "gate_odes": {},  # var -> rhs (dimensionless)
                    "gate_increments": {},  # var -> rhs (in reset)
                    "derived": {},  # name -> rhs (dimensionless helpers)
                    "linked": {},  # S_var -> (hub name, hub field)
                    "current_terms": [],  # list of amp-expression strings summed into iSyn
                    "poisson": [],  # {"gate", "rate": (v,u), "weight"}
                    "namespace": {},  # const name -> (value, unit)
                    "masks": {},  # per-neuron 0/1 subset mask var -> fraction (random pulse)
                }
                if noise_sigma is not None:
                    populations[pop]["namespace"]["noise_sigma_v"] = noise_sigma
                cell_pop_of_node[node_id] = pop

        hubs = {}  # hub name -> {"source_pop", "gate", "summed_var"} (all_to_all)
        synapses = []  # sparse Synapses descriptors (random / one_to_one)
        probes = []  # observation probes for synapses with recorded internal state (u, x)
        weight_handles = {}  # edge label -> [(target pop, namespace key holding that edge's weight)]
        forms = {}  # synapse name -> its edge-independent symbolic reductions, shared by every edge it serves

        for edge_idx, edge in enumerate(edges):
            src = getattr(edge, "source", None)
            tgt = getattr(edge, "target", None)
            if src is None or tgt is None:
                continue
            src, tgt = int(src), int(tgt)
            rule = getattr(edge, "connectivity", None)
            rule_norm = str(rule).lower().replace("-", "_") if rule is not None else None
            weight = _edge_weight(edge)

            # ── Poisson background edge → PoissonInput on the target's ext gate ──
            if src in poisson_of_node:
                if tgt not in cell_pop_of_node:
                    continue
                bg_name, bg_obj = poisson_of_node[src]
                self._add_poisson(populations[cell_pop_of_node[tgt]], bg_name, bg_obj, dyn_lib, weight)
                continue

            # ── Timed current pulse edge → a rectangular current window on the target ──
            if src in pulse_of_node:
                if tgt not in cell_pop_of_node:
                    continue
                pulse_name, pulse_obj = pulse_of_node[src]
                self._add_current_pulse(
                    populations[cell_pop_of_node[tgt]],
                    pulse_name,
                    pulse_obj,
                    weight,
                    edge,
                    edge_idx,
                    rule_norm,
                    transient_ms,
                )
                continue

            if src not in cell_pop_of_node or tgt not in cell_pop_of_node:
                continue

            src_pop = cell_pop_of_node[src]
            tgt_pop = cell_pop_of_node[tgt]
            edge_dyn = getattr(edge, "dynamics", None)
            syn = dyn_lib.get(str(edge_dyn)) if edge_dyn is not None else None
            if syn is None:
                raise NotImplementedError(f"Edge {edge_idx} has no resolvable synapse dynamics.")
            prefix = safe_id(getattr(edge_dyn, "name", None) or str(edge_dyn))
            memo = forms.setdefault(str(edge_dyn), {})

            if rule_norm == "all_to_all":
                self._add_conductance_synapse(populations, hubs, src_pop, tgt_pop, syn, prefix, weight, memo)
                if getattr(edge, "label", None):
                    weight_handles.setdefault(str(edge.label), []).append((tgt_pop, f"w_{src_pop}__{prefix}"))
            elif rule_norm in ("random", "one_to_one"):
                self._add_sparse_synapse(
                    populations, synapses, probes, src_pop, tgt_pop, syn, prefix, weight, edge, edge_idx, rule_norm, memo
                )
            else:
                shown = "none (a single explicit connection)" if rule is None else repr(rule)
                raise NotImplementedError(
                    f"Brian2 backend handles all_to_all, random and one_to_one projections; "
                    f"edge {edge_idx} has connectivity {shown}."
                )

        return {
            "label": getattr(exp, "label", None),
            "populations": populations,
            "sizes": {name: pop["size"] for name, pop in populations.items()},
            "hubs": hubs,
            "synapses": synapses,
            "probes": probes,
            # `duration` is the MEASURED window and `transient_time` is prepended to it, so the run is their sum and raising the settle never shortens the data.
            "measured_ms": measured_ms,
            "transient_ms": transient_ms,
            "total_ms": transient_ms + measured_ms,
            "dt_ms": dt_ms,
            "weight_handles": weight_handles,
            "ramp": self._ramp_spec(weight_handles, populations, measured_ms),
            # Resolved once here, so the rendered script and run() build identical connectivity.
            "seed": getattr(getattr(exp, "execution", None), "random_seed", None),
        }

    def _ramp_spec(self, weight_handles, populations, measured_ms):
        """The declared quasi-static ramp, or None when the experiment declares no such protocol.

        An `Exploration` with ``strategy: continuation`` is a protocol rather than a grid: its single axis lists the values in the ORDER they are applied and the simulator state carries from one point to the next, so a hysteresis loop is declared by listing the parameter up and back down. Each point settles for the declared ``transient_time`` and then measures for the integration ``duration``, which must therefore be positive. The axis addresses one labelled projection's weight as ``network.edges.<label>.weight``, which is what keeps the sweep off every other synapse of the same dynamics.
        """
        expls = getattr(self.experiment, "explorations", None) or {}
        items = list(expls.items()) if hasattr(expls, "items") else [(getattr(e, "name", None), e) for e in expls]
        ramps = [(n, e) for n, e in items if str(getattr(e, "strategy", "") or "") == "continuation"]
        if not ramps:
            return None
        if len(ramps) > 1:
            raise NotImplementedError(f"Brian2 backend runs one continuation ramp per experiment; got {len(ramps)}.")
        name, expl = ramps[0]
        space = getattr(expl, "space", None) or []
        axes = list(space.values()) if hasattr(space, "values") else list(space)
        if len(axes) != 1:
            raise NotImplementedError(
                f"A continuation ramp is a single ordered protocol, so exploration {name!r} takes "
                f"exactly one axis; it declares {len(axes)}."
            )
        axis = axes[0]
        values = getattr(axis, "explored_values", None)
        if not values:
            raise ValueError(
                f"Exploration {name!r} is a continuation ramp, so its axis must list "
                f"`explored_values` — the order they are applied IS the protocol."
            )
        parameter = str(getattr(axis, "parameter", ""))
        parts = parameter.split(".")
        if len(parts) != 4 or parts[:2] != ["network", "edges"] or parts[3] != "weight":
            raise NotImplementedError(
                f"Brian2 ramps address a labelled projection's weight as "
                f"'network.edges.<label>.weight'; exploration {name!r} sweeps {parameter!r}."
            )
        label = parts[2]
        if label not in weight_handles:
            known = sorted(weight_handles) or ["(none — no edge carries a label)"]
            raise KeyError(f"No all-to-all edge is labelled {label!r}; labelled projections are {known}.")
        if measured_ms <= 0:
            raise ValueError(
                f"ramp {str(name)!r} measures a firing rate over integration.duration, which is "
                f"{measured_ms:g} — there is no window to count spikes in. `duration` is the MEASURED window "
                "and `transient_time` is the settle prepended to it, so a point that should only relax "
                "wants transient_time, not duration: 0."
            )
        record = [str(r) for r in (getattr(expl, "record", None) or [])]
        keep = {r[len("firing_rate_") :] for r in record if r.startswith("firing_rate_")}
        return {
            "name": str(name),
            "parameter": parameter,
            "values": [float(v) for v in values],
            "handles": weight_handles[label],
            # The populations whose rates the ramp reports: those its `record` names, or every one when it names none.
            "populations": [p for p in populations if not keep or p in keep],
        }

    def _add_poisson(self, pop, bg_name, bg_obj, dyn_lib, weight):
        """Wire a Poisson background: an external gate + a PoissonInput driving it.

        The per-edge ``weight`` scales the delivered conductance (outside the gate), matching the recurrent-synapse convention.
        """
        bp = _params(bg_obj)
        syn_ref = bp.get("synapse", (None, ""))[0]
        ext_syn = dyn_lib.get(str(syn_ref)) if syn_ref else None
        if ext_syn is None:
            raise NotImplementedError(f"Poisson background {bg_name!r} references unknown synapse {syn_ref!r}.")
        sp_ = _params(ext_syn)
        gate = f"s_ext_{safe_id(bg_name)}"
        pop["gate_odes"][gate] = f"-{gate} / tau_{gate}"
        pop["namespace"][f"tau_{gate}"] = sp_.get("tauDecay", (2.0, "ms"))
        pop["namespace"][f"gbase_{gate}"] = sp_.get("gbase", (1.0, "nS"))
        pop["namespace"][f"erev_{gate}"] = sp_.get("erev", (0.0, "mV"))
        pop["namespace"][f"w_{gate}"] = (weight, "dimensionless")
        pop["current_terms"].append(f"w_{gate} * gbase_{gate} * (erev_{gate} - v) * {gate}")
        pop["poisson"].append({"gate": gate, "rate": bp.get("averageRate", (2400.0, "Hz")), "weight": 1.0})

    def _add_current_pulse(self, pop, pulse_name, pulse_obj, weight, edge, edge_idx, rule_norm, transient_ms=0.0):
        """Wire a deterministic timed current pulse onto a target population.

        A ``pulseGenerator`` (delay, duration, amplitude) becomes a rectangular current window summed into ``iSyn``: ``w * amplitude`` for ``delay <= t < delay + duration``, zero otherwise, added to every neuron of the population. This is the declarative loading / nonspecific-readout drive — deterministic, so it is identical between the in-process run and the generated script. The per-edge ``weight`` scales the amplitude (a uniform nonspecific readout uses ``weight = 1``); when several pulse edges target the same population their windows sum, each keeping its own weight.

        A `random` edge instead drives only a random SUBSET, ``connection_probability`` of the population — the paper's nonspecific input to 15% of the excitatory neurons — as a per-neuron 0/1 mask drawn once from the seeded RNG, so run and rendered script are identical. The mask is per EDGE, so two random pulse edges onto one population are two independent subsets. Any other connectivity rule raises, as it does for a synapse edge.

        The declared ``delay`` is on the MEASUREMENT clock, as a declared onset is in every other backend: a pulse at 300 ms fires 300 ms into the measured window, not 300 ms into the settle that precedes it. Brian2's ``t`` opens at the start of integration, so the settle is added here — once, on the constant — rather than left for a modeller to add to every onset by hand and to leave out of one.
        """
        fraction = self._pulse_fraction(edge, edge_idx, rule_norm)
        pp = _params(pulse_obj)
        key = safe_id(pulse_name)
        amp = pp.get("amplitude", (0.0, "nA"))
        delay = pp.get("delay", (0.0, "ms"))
        dur = pp.get("duration", (0.0, "ms"))
        pop["namespace"][f"amp_stim_{key}"] = amp
        pop["namespace"][f"delay_stim_{key}"] = _shift_onto_integration_clock(delay, transient_ms)
        pop["namespace"][f"dur_stim_{key}"] = dur
        # Keyed by edge: independent draws, possibly different fractions.
        gate = ""
        if fraction is not None:
            mask = f"stim_mask_{key}_{edge_idx}"
            pop["masks"][mask] = float(fraction)
            gate = f"{mask} * "
        # Inline the per-edge weight and always append the term. A second edge from the same pulse onto this population shares the (identical) amp/delay/dur constants but adds its own window, so the two sum — rather than the second overwriting a shared ``w_stim`` and its byte-identical term being dropped as a duplicate, silently losing a weight.
        term = f"{gate}{weight} * amp_stim_{key} * int(t >= delay_stim_{key}) * int(t < delay_stim_{key} + dur_stim_{key})"
        pop["current_terms"].append(term)

    @staticmethod
    def _pulse_fraction(edge, edge_idx, rule_norm):
        """The stimulated fraction a current-pulse edge declares, or ``None`` for all-to-all."""
        if rule_norm in (None, "all_to_all"):
            return None
        if rule_norm != "random":
            raise NotImplementedError(
                f"Brian2 backend drives a current pulse over the whole target (all_to_all) "
                f"or a 'random' subset; edge {edge_idx} has connectivity {rule_norm!r}."
            )
        fraction = _edge_param(edge, "connection_probability", None)
        if fraction is None:
            raise NotImplementedError(
                f"Edge {edge_idx}: a 'random' current-pulse edge needs a "
                f"'connection_probability' (the stimulated fraction) in `parameters`."
            )
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                f"Edge {edge_idx}: connection_probability is a fraction in (0, 1]; got {fraction} (15% is 0.15, not 15)."
            )
        return fraction

    def _add_conductance_synapse(self, populations, hubs, src_pop, tgt_pop, syn, prefix, weight, memo):
        """Reduce one all-to-all conductance synapse to gate + hub + current term.

        The gate lives on the *source* population (one per source pop and synapse dynamics, shared across all of that source's projections of this dynamics — e.g. an E pool's recurrent and long-range AMPA read the same pre-synaptic gate). The *target*-side terms (summed gate ``S``, weight, current) are keyed additionally by the source pop, so the same dynamics arriving at one pool from two different sources (recurrent + long-range) don't overwrite each other. *memo* holds the synapse's edge-independent reductions.
        """
        src = populations[src_pop]
        tgt = populations[tgt_pop]
        nml = nml_type(syn, "")
        sparams = _params(syn)

        gate_prefix = prefix  # source-side (shared per source pop)
        cur_prefix = f"{src_pop}__{prefix}"  # target-side (per incoming source)

        def gconst(name):
            return f"{name}_{gate_prefix}"

        def cconst(name):
            return f"{name}_{cur_prefix}"

        if nml in _EXP_ONE_TYPES:
            # Single-exponential conductance: dimensionless gate decays at tauDecay, a pre-synaptic spike adds 1, current is w * gbase * (erev - v) * S.
            gate = f"s_{gate_prefix}"
            summed_gate = gate
            src["gate_odes"][gate] = f"-{gate} / {gconst('tauDecay')}"
            src["gate_increments"].setdefault(gate, f"{gate} + 1")
            src["namespace"][gconst("tauDecay")] = sparams["tauDecay"]
            tgt["namespace"][cconst("gbase")] = sparams["gbase"]
            tgt["namespace"][cconst("erev")] = sparams["erev"]
            tgt["namespace"][cconst("w")] = (weight, "dimensionless")
            current = f"{cconst('w')} * {cconst('gbase')} * ({cconst('erev')} - v) * S_{cur_prefix}"
            summed_var = f"S_{cur_prefix}"
        else:
            r = self._reduce_custom(syn, sparams, gate_prefix, cur_prefix, memo)
            summed_gate = r["summed_gate"]
            src["gate_odes"].update(r["gate_odes"])
            for g, incr in r["increments"].items():
                src["gate_increments"].setdefault(g, incr)
            src["namespace"].update(r["gate_consts"])
            tgt["namespace"].update(r["current_consts"])
            tgt["namespace"][f"w_{cur_prefix}"] = (weight, "dimensionless")
            current = r["current"]
            summed_var = r["summed_var"]

        # Hub: sum the source's gate over the source population; link into the target.
        hub_name = f"hub_{src_pop}_{summed_gate}"
        field = f"Sig_{summed_gate}"
        hubs[hub_name] = {"source_pop": src_pop, "gate": summed_gate, "summed_var": field}
        tgt["linked"][summed_var] = (hub_name, field)
        tgt["current_terms"].append(current)

    def _reduce_custom(self, syn, sparams, gate_prefix, cur_prefix, memo):
        """Reduce a custom conductance synapse's declared dynamics to Brian2 form.

        Renames the pre-synaptic gate ODEs / spike increments with *gate_prefix* (they live on the source pop, shared across its projections of this dynamics) and the post-synaptic current with *cur_prefix* (keyed by source pop, so two sources of the same dynamics onto one target stay distinct). The current ``i`` has its derived variables inlined and, because an all-to-all conductance is delivered as a *population sum*, must be linear in the summed gate; that analysis depends on the synapse alone, so it runs once and is kept in *memo*. Returns the source gate ODEs/increments, the target current (gate replaced by the summed ``S`` and ``weight`` applied outside), the target linked-var name, and which constants belong to the gate vs current.
        """
        import sympy as sp

        if "summed" not in memo:
            memo["summed"] = _summed_form(syn, sparams)
        form = memo["summed"]
        syms, g = form["syms"], form["gate"]
        # Gate side renames with gate_prefix; current side with cur_prefix.
        gate_rename = {syms[n]: sp.Symbol(f"{n}_{gate_prefix}") for n in list(syn.state_variables) + list(sparams)}
        gate_rename[syms["v"]] = syms["v"]

        # Gate ODEs + spike increments (source side); track referenced constant names.
        gate_odes, increments, gate_ref = {}, {}, set()

        def _record(expr):
            renamed = expr.subs(gate_rename)
            gate_ref.update(s.name for s in renamed.free_symbols)
            return render_expression(str(renamed), format="brian2")

        for n, expr in form["odes"].items():
            gate_odes[f"{n}_{gate_prefix}"] = _record(expr)
        for lhs, expr in form["affects"]:
            increments[f"{lhs}_{gate_prefix}"] = _record(expr)

        # Current side: params -> cur_prefix, the gate -> the summed target var S_{cur_prefix}.
        cur_rename = {syms[n]: sp.Symbol(f"{n}_{cur_prefix}") for n in sparams}
        cur_rename[syms["v"]] = syms["v"]
        cur_rename[syms[g]] = sp.Symbol(f"S_{cur_prefix}")
        current_expr = sp.Symbol(f"w_{cur_prefix}") * form["i"].subs(cur_rename)
        current = render_expression(str(current_expr), format="brian2")
        current_ref = {s.name for s in current_expr.free_symbols}

        gate_all = {f"{n}_{gate_prefix}": sparams[n] for n in sparams}
        cur_all = {f"{n}_{cur_prefix}": sparams[n] for n in sparams}
        return {
            "summed_gate": f"{g}_{gate_prefix}",
            "summed_var": f"S_{cur_prefix}",
            "gate_odes": gate_odes,
            "increments": increments,
            "current": current,
            "gate_consts": {k: v for k, v in gate_all.items() if k in gate_ref},
            "current_consts": {k: v for k, v in cur_all.items() if k in current_ref},
        }

    # --------------------------------------------------------------- sparse projections
    def _add_sparse_synapse(
        self, populations, synapses, probes, src_pop, tgt_pop, syn, prefix, weight, edge, edge_idx, rule_norm, memo
    ):
        """Emit one genuinely-sparse projection as a Brian2 ``Synapses``.

        Three synapse forms are lowered, chosen by the synapse's declared dynamics:

        * a **decaying conductance** (``expOneSynapse`` or a custom synapse whose current ``i`` is linear in a single pure-decay gate) — the delivered conductance decays on the *target* ``NeuronGroup`` and is incremented event-driven by ``on_pre``;
        * an **instantaneous (delta) PSC** — a synapse with no continuous current ``i`` whose spike event jumps the post-synaptic membrane ``v`` directly (current-based, no conductance and no synaptic time constant), e.g. the Mongillo/Amit-Brunel form.

        Short-term-plasticity state (u, x) lives on the synapse as ``(event-driven)`` variables, mutated in ``on_pre`` in the recipe's declared order, so any facilitation/depression convention is honoured per connection. ``all_to_all`` keeps the O(N) hub path.

        Each projection's Brian2 objects are named by ``(synapse, source, target)`` so the block-structured networks (several edges sharing one synapse dynamics between different sub-population pairs) never collide on a name, and they share the synapse's edge-independent reductions through *memo*.
        """
        tgt = populations[tgt_pop]
        nml = nml_type(syn, "")
        sparams = _params(syn)
        gkey = f"{prefix}_from_{src_pop}_to_{tgt_pop}"
        gvar = f"gsyn_{gkey}"

        # Brian2 has no allow_self_connections, so autapses are excluded by condition.
        if rule_norm == "one_to_one":
            connect = {"j": "i"}  # source i -> target i
        else:  # random (fixed-probability Erdos-Renyi)
            p = _edge_param(edge, "connection_probability", None)
            if p is None:
                raise NotImplementedError(
                    f"Edge {edge_idx}: connectivity 'random' needs a 'connection_probability' "
                    f"parameter in the edge's `parameters`."
                )
            connect = {"p": float(p)}
            if src_pop == tgt_pop and getattr(edge, "allow_self_connections", True) is False:
                connect["condition"] = "i != j"

        if nml in _EXP_ONE_TYPES:
            # Static single-exponential conductance: decay on target, event-driven on_pre += weight.
            tgt["gate_odes"][gvar] = f"-{gvar} / tauDecay_{gvar}"
            tgt["namespace"][f"tauDecay_{gvar}"] = sparams["tauDecay"]
            tgt["namespace"][f"gbase_{gvar}"] = sparams["gbase"]
            tgt["namespace"][f"erev_{gvar}"] = sparams["erev"]
            tgt["current_terms"].append(f"gbase_{gvar} * (erev_{gvar} - v) * {gvar}")
            synapses.append(
                {
                    "name": f"syn_{gkey}",
                    "source": src_pop,
                    "target": tgt_pop,
                    "model": "",
                    "on_pre": f"{gvar}_post += {float(weight)}",
                    "connect": connect,
                    "namespace": {},
                    "init": {},
                }
            )
            return

        if "i" not in (syn.derived_variables):
            # Instantaneous (delta) PSC: the spike event jumps v_post directly; no conductance.
            r = self._reduce_delta_sparse(syn, sparams, float(weight), edge_idx, memo)
            synapses.append(
                {
                    "name": f"syn_{gkey}",
                    "source": src_pop,
                    "target": tgt_pop,
                    "model": r["model"],
                    "on_pre": r["on_pre"],
                    "connect": connect,
                    "namespace": r["syn_consts"],
                    "init": r["init"],
                }
            )
            self._maybe_add_probe(probes, populations, syn, src_pop, gkey, r)
            return

        # Custom conductance synapse (STP): decaying post-synaptic conductance + per-synapse u/x.
        r = self._reduce_custom_sparse(syn, sparams, gvar, float(weight), edge_idx, memo)
        tgt["gate_odes"][gvar] = r["decay"]
        tgt["namespace"].update(r["cur_consts"])
        tgt["current_terms"].append(r["current"])
        synapses.append(
            {
                "name": f"syn_{gkey}",
                "source": src_pop,
                "target": tgt_pop,
                "model": r["model"],
                "on_pre": r["on_pre"],
                "connect": connect,
                "namespace": r["syn_consts"],
                "init": r["init"],
            }
        )
        self._maybe_add_probe(probes, populations, syn, src_pop, gkey, r)

    def _maybe_add_probe(self, probes, populations, syn, src_pop, gkey, r):
        """Register an observation probe when the synapse declares recorded internal state.

        A synapse state variable with ``record: true`` (e.g. the STP ``u``/``x``) is monitored by a clock-driven, zero-delivery copy of the projection: a representative sample of the source population's neurons carry the same STP dynamics, driven by the same presynaptic spikes, so their ``u``/``x`` equal what the real (event-driven) synapses use — and, being clock-driven, are recordable as the continuous trace (an event-driven StateMonitor would freeze the value between spikes). The probe delivers nothing (its ``_post +=`` line is dropped), so the network's results are byte-identical; only the observation is added.
        """
        svs = syn.state_variables
        recorded = [n for n, sv in svs.items() if bool(getattr(sv, "record", False))]
        if not recorded:
            return
        # Only variables that live ON the synapse (an event-driven ODE in the reduced model) can be probed — a target-side gate (g) is a cell variable, not a synapse one.
        on_synapse = {ln.strip()[1:].split("/dt", 1)[0].strip() for ln in r["model"].splitlines() if "/dt" in ln}
        vars_ = [n for n in recorded if n in on_synapse]
        if not vars_:
            return
        # One probe per (source population, synapse dynamics): a source neuron's u/x is a property of its own spikes and STP params, identical across all its outgoing synapses, so several edges sharing a source + dynamics (the A->A/A->B/A->non-sel fan-out) need only one probe.
        sig = (src_pop, r["model"], tuple(sorted(r["syn_consts"].items())), tuple(vars_))
        if any(p.get("_sig") == sig for p in probes):
            return
        key = src_pop if src_pop not in {p["key"] for p in probes} else f"{src_pop}__{gkey}"
        probes.append(
            {
                "name": f"probe_{gkey}",
                "key": key,
                "source": src_pop,
                "vars": vars_,
                "model": r["model"].replace("(event-driven)", "(clock-driven)"),
                "on_pre": "\n".join(line for line in r["on_pre"].split("\n") if "_post +=" not in line),
                "namespace": r["syn_consts"],
                "init": r["init"],
                "sample_i": _sample_indices(populations[src_pop]["size"], _PROBE_SAMPLE),
                "record_dt_ms": _PROBE_RECORD_DT_MS,
                "_sig": sig,
            }
        )

    def _reduce_delta_sparse(self, syn, sparams, weight, edge_idx, memo):
        """Reduce an instantaneous (delta) PSC synapse to the sparse per-synapse Brian2 form.

        A delta synapse has no continuous current: an arriving spike jumps the post-synaptic membrane by an amount set in the spike event as ``v = v + <expr>`` (current-based, no conductance, no synaptic time constant — the Mongillo/Amit-Brunel form). The jump is delivered as ``v_post += weight * (<expr> - v) * mV`` (so ``weight`` is the PSP jump amplitude in mV, signed: negative for inhibitory projections). Any other event pieces act on the synapse's own short-term-plasticity variables (u, x), which live on the synapse as ``(event-driven)`` equations and are updated in the recipe's declared order — so facilitation-before-release is honoured and the delivered jump uses the updated u. Only the weight is the edge's own; the rest is reduced once per synapse and kept in *memo*.
        """
        if "delta" not in memo:
            memo["delta"] = _delta_form(syn, sparams, edge_idx)
        form = memo["delta"]
        return {
            "model": form["model"],
            "on_pre": _on_pre(form["steps"], lambda incr: f"v_post += {weight} * ({incr}) * mV"),
            "syn_consts": dict(form["syn_consts"]),
            "init": dict(form["init"]),
        }

    def _reduce_custom_sparse(self, syn, sparams, gvar, weight, edge_idx, memo):
        """Reduce a custom conductance synapse to the sparse per-synapse Brian2 form.

        The single gate in the current ``i`` must be a PURE DECAYING conductance (``dg/dt = -g/tau``): it becomes the post-synaptic decaying variable ``gvar``, delivered by ``on_pre``. The remaining state variables (STP u, x) become per-synapse ``(event-driven)`` equations, mutated in ``on_pre``. The spike event's conductance increment is delivered as ``gvar_post += weight*(increment)``; its u/x updates run on the synapse, all emitted in the recipe's declared order so any facilitation/depression convention is honoured. Only ``gvar`` and the weight are the edge's own; the rest is reduced once per synapse and kept in *memo*.
        """
        import sympy as sp

        if "sparse" not in memo:
            memo["sparse"] = _sparse_form(syn, sparams, edge_idx)
        form = memo["sparse"]
        syms = form["syms"]
        # Target-side renames (g -> gvar, params -> gvar-suffixed) for the decay ODE + current.
        cur_rename = {
            syms[form["gate"]]: sp.Symbol(gvar),
            syms["v"]: syms["v"],
            **{syms[p]: sp.Symbol(f"{p}_{gvar}") for p in sparams},
        }
        decay_expr = form["g_ode"].subs(cur_rename)
        current_expr = form["i"].subs(cur_rename)
        cur_ref = {s.name for s in current_expr.free_symbols} | {s.name for s in decay_expr.free_symbols}
        return {
            "decay": render_expression(str(decay_expr), format="brian2"),
            "current": render_expression(str(current_expr), format="brian2"),
            "cur_consts": {f"{p}_{gvar}": sparams[p] for p in sparams if f"{p}_{gvar}" in cur_ref},
            "model": form["model"],
            "on_pre": _on_pre(form["steps"], lambda incr: f"{gvar}_post += {weight} * ({incr})"),
            "syn_consts": dict(form["syn_consts"]),
            "init": dict(form["init"]),
        }


def assemble_eqs(pop):
    """The Brian2 ``Equations`` block for a cell population.

    Membrane ODE + a summed drive ``iSyn`` + the pre-synaptic gate ODEs (dimensionless) + any linked summed-gate variables. Shared by the in-process ``run`` path and the generated script so the two never diverge. A conductance-based cell's drive is a current (``amp``); a current-based cell (one declaring a membrane time constant ``tau_m``, whose membrane is ``(-v + ... + iSyn)/tau_m``) has a voltage drive (``volt``) — the Mongillo/Amit-Brunel form.
    """
    drive_unit = "volt" if "tau_m" in pop["cell_params"] else "amp"
    membrane = f"dv/dt = ({pop['v_rhs']})"
    if pop.get("noise_sigma") is not None:
        # Additive Gaussian white noise on the membrane: the faithful discretisation of the current-based SDE tau*dv = -v + ... + sigma*eta(t) is dv/dt += sigma*xi/sqrt(tau_m) (stationary membrane std sigma/sqrt(2)). The 1/sqrt(tau_m) normalisation needs the membrane time constant, which only the current-based form declares; a conductance- based cell has no single tau_m, so reject it here rather than emit an equation that references an undefined ``tau_m`` and fails deep inside Brian2 with an opaque error.
        if "tau_m" not in pop["cell_params"]:
            raise NotImplementedError(
                f"Cell {pop['name']!r} declares membrane noise but no 'tau_m'. The additive "
                "white-noise discretisation sigma*xi/sqrt(tau_m) is defined only for the "
                "current-based membrane form (tau_m*dv/dt = -v + ...); a conductance-based "
                "cell has no single membrane time constant. Add tau_m or remove the noise."
            )
        membrane += " + noise_sigma_v * xi * tau_m**-0.5"
    lines = [membrane + " : volt (unless refractory)"]
    zero = "0*mV" if drive_unit == "volt" else "0*amp"
    if pop["current_terms"]:
        lines.append("iSyn = " + " + ".join(f"({t})" for t in pop["current_terms"]) + f" : {drive_unit}")
    else:
        lines.append(f"iSyn = {zero} : {drive_unit}")
    for name, rhs in pop["derived"].items():
        lines.append(f"{name} = {rhs} : 1")
    for gate, rhs in pop["gate_odes"].items():
        lines.append(f"d{gate}/dt = {rhs} : 1")
    for svar in pop["linked"]:
        lines.append(f"{svar} : 1 (linked)")
    for mvar in pop["masks"]:
        lines.append(f"{mvar} : 1 (constant)")  # per-neuron 0/1 random-subset stim mask
    return "\n".join(lines)


def reset_code(pop):
    """The Brian2 reset statement: v reset plus pre-synaptic gate increments."""
    parts = ["v = reset"]
    for gate, incr in pop["gate_increments"].items():
        parts.append(f"{gate} = {incr}")
    return "; ".join(parts)


def _instantiate(model, seed=None, record_v=False):
    """Build a Brian2 ``Network`` from a build description (the run() path)."""
    import brian2
    from brian2 import (
        Network,
        NeuronGroup,
        PoissonInput,
        SpikeMonitor,
        StateMonitor,
        Synapses,
        defaultclock,
        linked_var,
        ms,
        mV,
        start_scope,
    )

    start_scope()
    if seed is not None:
        brian2.seed(int(seed))
    defaultclock.dt = model["dt_ms"] * ms

    def qty(value, unit):
        bu = _brian2_unit(unit)
        if not bu:
            return float(value)
        if bu.startswith("1/"):
            return float(value) / getattr(brian2, bu[2:])
        return float(value) * getattr(brian2, bu)

    objects, groups, spike_mons, state_mons = [], {}, {}, {}
    v_unit = mV

    for name, pop in model["populations"].items():
        cp = {k: qty(*v) for k, v in pop["cell_params"].items()}
        ns = {k: qty(*v) for k, v in pop["namespace"].items()}
        refract = cp.get("refract", 0 * ms)
        grp = NeuronGroup(
            pop["size"],
            assemble_eqs(pop),
            threshold="v > thresh",
            reset=reset_code(pop),
            refractory=refract,
            method="euler",
            namespace={**cp, **ns},
            name=name,
        )
        grp.v = cp.get("v0", cp.get("EL", -70 * mV))
        for mvar, frac in pop["masks"].items():
            setattr(grp, mvar, f"rand() < {frac}")  # seeded per-neuron subset mask (render≡run)
        groups[name] = grp
        objects.append(grp)

    # Hubs: sum the gate over the source population, expose Sig_<gate>.
    hub_groups = {}
    for hub_name, hub in model["hubs"].items():
        field = hub["summed_var"]
        hg = NeuronGroup(1, f"{field} : 1", name=hub_name)
        hub_groups[hub_name] = hg
        objects.append(hg)
        syn = Synapses(
            groups[hub["source_pop"]], hg, model=f"{field}_post = {hub['gate']}_pre : 1 (summed)", name=f"sum_{hub_name}"
        )
        syn.connect()
        objects.append(syn)

    # Link summed gates into the target populations.
    for name, pop in model["populations"].items():
        for svar, (hub_name, field) in pop["linked"].items():
            setattr(groups[name], svar, linked_var(hub_groups[hub_name], field))

    # Sparse projections: real Synapses with connect(p=...) / connect(j='i').
    for sd in model.get("synapses", []):
        ns = {k: qty(*v) for k, v in sd["namespace"].items()}
        syn = Synapses(
            groups[sd["source"]],
            groups[sd["target"]],
            model=(sd["model"] or None),
            on_pre=sd["on_pre"],
            namespace=ns,
            method="euler",
            name=sd["name"],
        )
        syn.connect(**sd["connect"])
        for var, val in sd["init"].items():
            setattr(syn, var, val)
        objects.append(syn)

    # Poisson backgrounds.
    for name, pop in model["populations"].items():
        for pin in pop["poisson"]:
            objects.append(PoissonInput(groups[name], pin["gate"], 1, qty(*pin["rate"]), weight=pin["weight"]))

    # Observation probes: clock-driven, zero-delivery copies of a sampled subset of a recorded synapse, StateMonitored for the continuous internal state (u, x). Delivering nothing, they leave the network's results untouched.
    probe_mons = {}
    probes = model.get("probes", [])
    if probes:
        sink = NeuronGroup(1, "x_sink : 1", name="probe_sink")
        objects.append(sink)
        for pr in probes:
            pns = {k: qty(*v) for k, v in pr["namespace"].items()}
            pg = Synapses(
                groups[pr["source"]],
                sink,
                model=pr["model"],
                on_pre=pr["on_pre"],
                namespace=pns,
                method="euler",
                name=pr["name"],
            )
            pg.connect(i=pr["sample_i"], j=0)
            for var, val in pr["init"].items():
                setattr(pg, var, val)
            objects.append(pg)
            pm = StateMonitor(pg, pr["vars"], record=True, dt=pr["record_dt_ms"] * ms, name=f"mon_{pr['name']}")
            objects.append(pm)
            probe_mons[pr["name"]] = {"mon": pm, "key": pr["key"], "source": pr["source"], "vars": pr["vars"]}

    # Monitors.
    for name in model["populations"]:
        spike_mons[name] = SpikeMonitor(groups[name])
        objects.append(spike_mons[name])
        if record_v:
            state_mons[name] = StateMonitor(groups[name], "v", record=True)
            objects.append(state_mons[name])

    return Network(objects), {
        "spike_monitors": spike_mons,
        "state_monitors": state_mons,
        "probe_monitors": probe_mons,
        "v_unit": v_unit,
    }
