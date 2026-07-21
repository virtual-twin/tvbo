"""Native Brian2 backend for small-scale spiking networks.

Consumes the shared small-scale lowering core (:mod:`tvbo.adapters.smallscale`)
and emits a Brian2 point-neuron network. The maths is printed through the shared
SymPy printer (``render_expression(..., format="brian2")``); this adapter adds only
the Brian2 role vocabulary and the synapse rendering.

**All-to-all conductance synapses lower to O(N) population sums, not O(N^2)
`Synapses`.** For a fully-connected projection every post-synaptic neuron sees the
same sum over pre-synaptic gating, so the gate lives on the *pre-synaptic* neuron
and a size-1 "hub" `NeuronGroup` accumulates the population sum once (via a
``(summed)`` `Synapses`), read by every post-synaptic neuron through a ``linked_var``.
This is exactly the structure of the hand-written Deco 2014 reference
`deco_column.py`, and is what lets the 160E+40I column run in seconds where the
enumerated LEMS network needs ~190 s per 100 ms in jLEMS.

Supported synapse forms (the Deco 2014 column):
  * ``neuroml:expOneSynapse`` — single-exponential conductance (AMPA, GABA);
  * a custom conductance synapse extending ``baseConductanceBasedSynapse`` whose
    gate ODEs / block / current are declared (the saturating NMDA with Mg block),
    provided the current is *linear* in the summed gate (population-sum requirement);
  * ``neuroml:poissonFiringSynapse`` — independent Poisson background → `PoissonInput`.

Anything outside this set (sparse/one-to-one connectivity, non-Poisson spike
sources, constant-current inputs, synapses nonlinear in their gate) raises a clear
``NotImplementedError`` rather than mis-simulating.
"""

from __future__ import annotations

from tvbo.adapters.base import BaseAdapter
from tvbo.adapters.smallscale.lowering import (
    classify_node_role,
    group_nodes_by_dynamics,
    normalize_edge_params,
    safe_id,
)
from tvbo.codegen.code import render_expression

# ── Brian2 role vocabulary ────────────────────────────────────────────
_POISSON_TYPES = frozenset({"poissonFiringSynapse", "transientPoissonFiringSynapse"})
# Spike (event) sources other than Poisson — recognised so they raise rather than
# being silently mistaken for a cell population (they carry no membrane `v`).
_SPIKE_SOURCE_TYPES = frozenset(
    {"spikeGenerator", "spikeGeneratorRandom", "spikeGeneratorRefPoisson",
     "spikeGeneratorPoisson", "spikeArray", "SpikeSourcePoisson"}
)
_CURRENT_INPUT_TYPES = frozenset(
    {"pulseGenerator", "pulseGeneratorDL", "sineGenerator", "rampGenerator"}
)
_BRIAN2_ROLE_VOCAB = {
    "current_input": _CURRENT_INPUT_TYPES,
    "event_source": _POISSON_TYPES | _SPIKE_SOURCE_TYPES,
}
_EXP_ONE_TYPES = frozenset({"expOneSynapse"})

# TVBO time-scale → factor to convert a time value into milliseconds.
_TIME_SCALE_TO_MS = {"s": 1000.0, "ms": 1.0, "us": 0.001}

# TVBO unit name -> Brian2 unit name (as it appears in a generated script). A unit
# absent here is rejected (fail loud) rather than silently dropped.
_UNIT_TO_BRIAN2 = {
    "nS": "nS", "pS": "psiemens", "uS": "usiemens", "siemens": "siemens",
    "mV": "mV", "V": "volt", "volt": "volt",
    "ms": "ms", "s": "second", "us": "usecond", "second": "second",
    "nF": "nF", "pF": "pfarad", "uF": "ufarad", "farad": "farad",
    "nA": "nA", "pA": "pamp", "amp": "amp",
    "Hz": "Hz", "kHz": "kHz",
    "per_ms": "1/ms", "per_s": "1/second",
    "dimensionless": "", "": "",
}


def _nml_type(dyn):
    iri = getattr(dyn, "iri", "") or ""
    return iri.split(":", 1)[1] if iri.startswith("neuroml:") else ""


def _params(dyn):
    """``{name: (value, unit)}`` for a Dynamics/edge's parameters."""
    out = {}
    for name, p in normalize_edge_params(getattr(dyn, "parameters", None)).items():
        val = getattr(p, "value", p)
        unit = str(getattr(p, "unit", "") or "")
        out[str(name)] = (val, unit)
    return out


def _edge_weight(edge):
    """The per-connection ``weight`` on an edge (1.0 if absent or valueless)."""
    for name, p in normalize_edge_params(getattr(edge, "parameters", None)).items():
        if str(name) == "weight":
            val = getattr(p, "value", p)
            return float(val) if val is not None else 1.0
    return 1.0


def _brian2_unit(unit):
    """Brian2 unit expression for a TVBO unit name, or raise for an unknown one."""
    if unit not in _UNIT_TO_BRIAN2:
        raise ValueError(
            f"Brian2 backend has no unit mapping for {unit!r}; add it to _UNIT_TO_BRIAN2."
        )
    return _UNIT_TO_BRIAN2[unit]


def _brian2_const(value, unit):
    """A Brian2 constant literal for a script, e.g. ``0.2 * nS`` or ``0.5 / ms``."""
    bu = _brian2_unit(unit)
    if not bu:
        return f"{value}"
    if bu.startswith("1/"):
        return f"{value} / {bu[2:]}"
    return f"{value} * {bu}"


class Brian2Adapter(BaseAdapter):
    """Render/run a small-scale spiking network natively in Brian2."""

    def render_code(self, **kwargs) -> str:
        from tvbo import templates

        ctx = self._build_context()
        ctx.update(kwargs)
        template = templates.lookup.get_template("brian2/tvbo-brian2-experiment.py.mako")
        return template.render(**ctx)

    def run(self, seed=None, record_v=False, settle_ms=None, codegen_target="numpy", **kwargs):
        """Build and run the network in Brian2, returning an ExperimentResult.

        Population firing rates (from Brian2 ``SpikeMonitor``) are the primary
        output — the exact quantity the Deco 2014 replication targets — and are
        exposed both as ``result.integration.observations.firing_rate_<pop>`` and,
        raw, under ``result._extras``.

        ``codegen_target`` defaults to ``"numpy"`` (no C compilation, portable);
        pass ``"cython"`` for the faster compiled path where the toolchain allows.
        """
        import numpy as np
        import brian2
        from brian2 import ms

        from tvbo.data.types import ExperimentResult, SimulationResult

        if codegen_target:
            brian2.prefs.codegen.target = codegen_target
        # An unset seed falls back to the experiment's declared execution.random_seed,
        # so a run reproduces from the recipe without a manual seed argument.
        if seed is None:
            seed = getattr(getattr(self.experiment, "execution", None), "random_seed", None)
        model = self._build_context()
        net, meta = _instantiate(model, seed=seed, record_v=record_v)
        duration = model["duration_ms"]  # milliseconds
        net.run(duration * ms)

        settle = settle_ms if settle_ms is not None else min(1000.0, 0.2 * duration)
        rates, spikes = {}, {}
        for name, mon in meta["spike_monitors"].items():
            n = model["populations"][name]["size"]
            t = np.asarray(mon.t / ms)
            counts = int((t >= settle).sum())
            window_s = (duration - settle) / 1000.0
            rates[name] = counts / (window_s * n) if window_s > 0 and n else 0.0
            spikes[name] = {"i": np.asarray(mon.i), "t_ms": t}

        integration = SimulationResult(observations={f"firing_rate_{k}": v for k, v in rates.items()})
        result = ExperimentResult(
            integration=integration, name=getattr(self.experiment, "label", None), source=self.experiment
        )
        result._extras["rates"] = rates
        result._extras["spikes"] = spikes
        if record_v and meta.get("state_monitors"):
            result._extras["v"] = {p: np.asarray(m.v / meta["v_unit"]) for p, m in meta["state_monitors"].items()}
            any_mon = next(iter(meta["state_monitors"].values()))
            result._extras["t_ms"] = np.asarray(any_mon.t / ms)
        return result

    # ── Analysis: declarative network → Brian2 build description ────────

    def _build_context(self):
        """Reduce the experiment to a backend-neutral Brian2 build description.

        Returns a dict the template renders and ``_instantiate`` builds:
        ``populations`` (per cell pop: eqs data, namespace, poisson, size),
        ``hubs`` (summed-gate accumulators), ``duration_ms``, ``dt_ms``.
        """
        exp = self.experiment
        network = exp.network
        dyn_lib = getattr(network, "dynamics", None) or {}
        nodes = getattr(network, "nodes", None) or []
        edges = getattr(network, "edges", None) or []

        integration = getattr(exp, "integration", None)
        raw_ts = str(getattr(integration, "time_scale", "ms") or "ms") if integration else "ms"
        ts_factor = _TIME_SCALE_TO_MS.get(raw_ts, 1.0)  # model time unit → ms
        dt_ms = float(getattr(integration, "step_size", 0.02) or 0.02) * ts_factor
        duration_ms = float(getattr(integration, "duration", 1000.0) or 1000.0) * ts_factor

        default_name = getattr(exp.dynamics, "name", None) or "dynamics"
        groups = group_nodes_by_dynamics(nodes, default_name)

        # Classify node roles and record per-node populations / sources.
        cell_pop_of_node = {}   # node_id -> pop name
        poisson_of_node = {}    # node_id -> (dyn_name, dyn_obj)
        populations = {}        # pop name -> descriptor
        for dyn_name, gnodes in groups.items():
            dyn_obj = dyn_lib.get(dyn_name)
            role, nml = classify_node_role(dyn_name, dyn_obj, _BRIAN2_ROLE_VOCAB)
            if role == "event_source":
                if nml not in _POISSON_TYPES:
                    raise NotImplementedError(
                        f"Brian2 backend does not yet handle the spike source {nml!r} "
                        f"(dynamics {dyn_name!r}); only Poisson backgrounds are supported."
                    )
                for node in gnodes:
                    poisson_of_node[getattr(node, "id", 0)] = (dyn_name, dyn_obj)
                continue
            if role == "current_input":
                raise NotImplementedError(
                    f"Constant-current input {nml!r} (dynamics {dyn_name!r}) is not yet wired for Brian2."
                )
            # A cell population — ONE per node, so same-dynamics nodes in different
            # areas stay separate (a two-area network keeps its two E pools distinct
            # and its long-range projection connects only the right pair). The clean
            # dynamics name is kept when a single node uses that dynamics; when several
            # do, the node id disambiguates (ExcitatoryCell_2 / ExcitatoryCell_6) so the
            # single-column output keys are unchanged.
            v_sv = (getattr(dyn_obj, "state_variables", None) or {}).get("v")
            if v_sv is None:
                raise NotImplementedError(
                    f"Cell dynamics {dyn_name!r} declares no membrane variable 'v'; cannot build a NeuronGroup."
                )
            v_rhs = render_expression(str(getattr(v_sv, "equation").rhs), format="brian2")
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
                    "cell_params": _params(dyn_obj),
                    "gate_odes": {},        # var -> rhs (dimensionless)
                    "gate_increments": {},  # var -> rhs (in reset)
                    "derived": {},          # name -> rhs (dimensionless helpers)
                    "linked": {},           # S_var -> (hub name, hub field)
                    "current_terms": [],    # list of amp-expression strings summed into iSyn
                    "poisson": [],          # {"gate", "rate": (v,u), "weight"}
                    "namespace": {},        # const name -> (value, unit)
                }
                cell_pop_of_node[node_id] = pop

        hubs = {}  # hub name -> {"source_pop", "gate", "summed_var"}

        for edge_idx, edge in enumerate(edges):
            src = getattr(edge, "source", None)
            tgt = getattr(edge, "target", None)
            if src is None or tgt is None:
                continue
            src, tgt = int(src), int(tgt)
            rule = getattr(edge, "connectivity", None)
            weight = _edge_weight(edge)

            # ── Poisson background edge → PoissonInput on the target's ext gate ──
            if src in poisson_of_node:
                if tgt not in cell_pop_of_node:
                    continue
                bg_name, bg_obj = poisson_of_node[src]
                self._add_poisson(populations[cell_pop_of_node[tgt]], bg_name, bg_obj, dyn_lib, weight)
                continue

            if src not in cell_pop_of_node or tgt not in cell_pop_of_node:
                continue
            if str(rule).lower().replace("-", "_") != "all_to_all":
                shown = "none (a single/one-to-one connection)" if rule is None else repr(rule)
                raise NotImplementedError(
                    f"Brian2 backend handles only all_to_all conductance projections; edge {edge_idx} "
                    f"has connectivity {shown}. Sparse and one-to-one connectivity are not yet supported."
                )

            src_pop = cell_pop_of_node[src]
            tgt_pop = cell_pop_of_node[tgt]
            edge_dyn = getattr(edge, "dynamics", None)
            syn = dyn_lib.get(str(edge_dyn)) if edge_dyn is not None else None
            if syn is None:
                raise NotImplementedError(f"Edge {edge_idx} has no resolvable synapse dynamics.")
            prefix = safe_id(getattr(edge_dyn, "name", None) or str(edge_dyn))

            self._add_conductance_synapse(populations, hubs, src_pop, tgt_pop, syn, prefix, weight)

        return {
            "label": getattr(exp, "label", None),
            "populations": populations,
            "hubs": hubs,
            "duration_ms": duration_ms,
            "dt_ms": dt_ms,
            "seed": None,
        }

    def _add_poisson(self, pop, bg_name, bg_obj, dyn_lib, weight):
        """Wire a Poisson background: an external gate + a PoissonInput driving it.

        The per-edge ``weight`` scales the delivered conductance (outside the gate),
        matching the recurrent-synapse convention.
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

    def _add_conductance_synapse(self, populations, hubs, src_pop, tgt_pop, syn, prefix, weight):
        """Reduce one all-to-all conductance synapse to gate + hub + current term.

        The gate lives on the *source* population (one per source pop and synapse
        dynamics, shared across all of that source's projections of this dynamics —
        e.g. an E pool's recurrent and long-range AMPA read the same pre-synaptic
        gate). The *target*-side terms (summed gate ``S``, weight, current) are keyed
        additionally by the source pop, so the same dynamics arriving at one pool from
        two different sources (recurrent + long-range) don't overwrite each other.
        """
        src = populations[src_pop]
        tgt = populations[tgt_pop]
        nml = _nml_type(syn)
        sparams = _params(syn)

        gate_prefix = prefix                       # source-side (shared per source pop)
        cur_prefix = f"{src_pop}__{prefix}"        # target-side (per incoming source)

        def gconst(name):
            return f"{name}_{gate_prefix}"

        def cconst(name):
            return f"{name}_{cur_prefix}"

        if nml in _EXP_ONE_TYPES:
            # Single-exponential conductance: dimensionless gate decays at tauDecay, a
            # pre-synaptic spike adds 1, current is w * gbase * (erev - v) * S.
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
            r = self._reduce_custom(syn, sparams, gate_prefix, cur_prefix)
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

    def _reduce_custom(self, syn, sparams, gate_prefix, cur_prefix):
        """Reduce a custom conductance synapse's declared dynamics to Brian2 form.

        Renames the pre-synaptic gate ODEs / spike increments with *gate_prefix*
        (they live on the source pop, shared across its projections of this dynamics)
        and the post-synaptic current with *cur_prefix* (keyed by source pop, so two
        sources of the same dynamics onto one target stay distinct). Inlines the
        derived variables into the current ``i`` once, and — because an all-to-all
        conductance is delivered as a *population sum* — requires ``i`` to be linear
        in the summed gate. Returns the source gate ODEs/increments, the target
        current (gate replaced by the summed ``S`` and ``weight`` applied outside),
        the target linked-var name, and which constants belong to the gate vs current.
        """
        import sympy as sp

        svs = getattr(syn, "state_variables", None) or {}
        dvs = getattr(syn, "derived_variables", None) or {}
        events = getattr(syn, "events", None) or {}
        if not svs:
            raise NotImplementedError(f"Synapse {getattr(syn,'name',syn)!r}: no state variables to render.")
        if "i" not in dvs:
            raise NotImplementedError(f"Synapse {getattr(syn,'name',syn)!r}: no current derived variable 'i'.")

        local = list(svs) + list(dvs) + list(sparams)
        syms = {n: sp.Symbol(n) for n in local + ["v"]}
        # Gate side renames with gate_prefix; current side with cur_prefix.
        gate_rename = {syms[n]: sp.Symbol(f"{n}_{gate_prefix}") for n in list(svs) + list(sparams)}
        gate_rename[syms["v"]] = syms["v"]

        def parse(rhs):
            return sp.sympify(str(rhs), locals=syms)

        # Gate ODEs + spike increments (source side); track referenced constant names.
        gate_odes, increments, gate_ref = {}, {}, set()

        def _record(expr):
            renamed = expr.subs(gate_rename)
            gate_ref.update(s.name for s in renamed.free_symbols)
            return render_expression(str(renamed), format="brian2")

        for n, sv in svs.items():
            gate_odes[f"{n}_{gate_prefix}"] = _record(parse(getattr(sv, "equation").rhs))
        for ev in events.values():
            affect = getattr(getattr(ev, "affect", None), "rhs", None)
            if not affect:
                continue
            for piece in str(affect).split(";"):
                if "=" in piece:
                    lhs, rhs = piece.split("=", 1)
                    increments[f"{lhs.strip()}_{gate_prefix}"] = _record(parse(rhs))

        # Inline the derived variables into `i` (fixed point).
        dv_exprs = {n: parse(getattr(dv, "equation").rhs) for n, dv in dvs.items()}
        i_expr = dv_exprs["i"]
        for _ in range(len(dv_exprs) + 1):
            sub = {syms[n]: e for n, e in dv_exprs.items() if n != "i" and syms[n] in i_expr.free_symbols}
            if not sub:
                break
            i_expr = i_expr.subs(sub)

        summed = [n for n in svs if syms[n] in i_expr.free_symbols]
        if len(summed) != 1:
            raise NotImplementedError(
                f"Synapse {getattr(syn,'name',syn)!r}: the current must reference exactly one gating "
                f"variable to lower to a population sum, found {summed}."
            )
        g = syms[summed[0]]
        # The population sum is only valid when i = coeff(v) * gate (linear, no offset).
        if sp.simplify(sp.diff(i_expr, g)).has(g) or sp.simplify(i_expr.subs(g, 0)) != 0:
            raise NotImplementedError(
                f"Synapse {getattr(syn,'name',syn)!r}: current is not linear in the gating variable "
                f"{summed[0]!r}; an all-to-all population sum requires linearity."
            )

        # Current side: params -> cur_prefix, the gate -> the summed target var S_{cur_prefix}.
        cur_rename = {syms[n]: sp.Symbol(f"{n}_{cur_prefix}") for n in sparams}
        cur_rename[syms["v"]] = syms["v"]
        cur_rename[g] = sp.Symbol(f"S_{cur_prefix}")
        current_expr = sp.Symbol(f"w_{cur_prefix}") * i_expr.subs(cur_rename)
        current = render_expression(str(current_expr), format="brian2")
        current_ref = {s.name for s in current_expr.free_symbols}

        gate_all = {f"{n}_{gate_prefix}": sparams[n] for n in sparams}
        cur_all = {f"{n}_{cur_prefix}": sparams[n] for n in sparams}
        return {
            "summed_gate": f"{summed[0]}_{gate_prefix}",
            "summed_var": f"S_{cur_prefix}",
            "gate_odes": gate_odes,
            "increments": increments,
            "current": current,
            "gate_consts": {k: v for k, v in gate_all.items() if k in gate_ref},
            "current_consts": {k: v for k, v in cur_all.items() if k in current_ref},
        }


def assemble_eqs(pop):
    """The Brian2 ``Equations`` block for a cell population.

    Membrane ODE + a summed synaptic current ``iSyn`` + the pre-synaptic gate ODEs
    (dimensionless) + any linked summed-gate variables. Shared by the in-process
    ``run`` path and the generated script so the two never diverge.
    """
    lines = [f"dv/dt = ({pop['v_rhs']}) : volt (unless refractory)"]
    if pop["current_terms"]:
        lines.append("iSyn = " + " + ".join(f"({t})" for t in pop["current_terms"]) + " : amp")
    else:
        lines.append("iSyn = 0*amp : amp")
    for name, rhs in pop["derived"].items():
        lines.append(f"{name} = {rhs} : 1")
    for gate, rhs in pop["gate_odes"].items():
        lines.append(f"d{gate}/dt = {rhs} : 1")
    for svar in pop["linked"]:
        lines.append(f"{svar} : 1 (linked)")
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
        Network, NeuronGroup, PoissonInput, SpikeMonitor, StateMonitor,
        Synapses, defaultclock, linked_var, mV, ms, start_scope,
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
        grp = NeuronGroup(pop["size"], assemble_eqs(pop), threshold="v > thresh", reset=reset_code(pop),
                          refractory=refract, method="euler", namespace={**cp, **ns}, name=name)
        grp.v = cp.get("v0", cp.get("EL", -70 * mV))
        groups[name] = grp
        objects.append(grp)

    # Hubs: sum the gate over the source population, expose Sig_<gate>.
    hub_groups = {}
    for hub_name, hub in model["hubs"].items():
        field = hub["summed_var"]
        hg = NeuronGroup(1, f"{field} : 1", name=hub_name)
        hub_groups[hub_name] = hg
        objects.append(hg)
        syn = Synapses(groups[hub["source_pop"]], hg,
                       model=f"{field}_post = {hub['gate']}_pre : 1 (summed)", name=f"sum_{hub_name}")
        syn.connect()
        objects.append(syn)

    # Link summed gates into the target populations.
    for name, pop in model["populations"].items():
        for svar, (hub_name, field) in pop["linked"].items():
            setattr(groups[name], svar, linked_var(hub_groups[hub_name], field))

    # Poisson backgrounds.
    for name, pop in model["populations"].items():
        for pin in pop["poisson"]:
            objects.append(PoissonInput(groups[name], pin["gate"], 1, qty(*pin["rate"]), weight=pin["weight"]))

    # Monitors.
    for name in model["populations"]:
        spike_mons[name] = SpikeMonitor(groups[name])
        objects.append(spike_mons[name])
        if record_v:
            state_mons[name] = StateMonitor(groups[name], "v", record=True)
            objects.append(state_mons[name])

    return Network(objects), {"spike_monitors": spike_mons, "state_monitors": state_mons, "v_unit": v_unit}
