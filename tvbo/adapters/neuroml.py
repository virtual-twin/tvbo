# -*- coding: utf-8 -*-
"""NeuroML/LEMS adapter for SimulationExperiment.

Renders a self-contained LEMS XML simulation file from any TVBO
SimulationExperiment using a Mako template.  Every Dynamics model
is exported as a custom LEMS ComponentType — no hardcoded mappings
to built-in NeuroML cell types.

Validation is done via PyLEMS (``lems.Model``).
Simulation can be run via pyNeuroML (jnml).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from tvbo.adapters.base import BaseAdapter

if TYPE_CHECKING:
    from tvbo.classes.experiment import SimulationExperiment
    from tvbo.data.types import ExperimentResult

# ── NeuroML dimension / unit mapping ─────────────────────────────────

# TODO: Represent the units in ontology (already started) and retrieve from there. this is semantic knowledge and axioms that are heloful for other backends, too!


DIMENSIONS = {
    "voltage":        (1, 2, -3, -1, 0, 0, 0),
    "time":           (0, 0, 1, 0, 0, 0, 0),
    "per_time":       (0, 0, -1, 0, 0, 0, 0),
    "conductance":    (-1, -2, 3, 2, 0, 0, 0),
    "conductanceDensity": (-1, -4, 3, 2, 0, 0, 0),
    "capacitance":    (-1, -2, 4, 2, 0, 0, 0),
    "specificCapacitance": (-1, -4, 4, 2, 0, 0, 0),
    "resistance":     (1, 2, -3, -2, 0, 0, 0),
    "resistivity":    (1, 3, -3, -2, 0, 0, 0),
    "current":        (0, 0, 0, 1, 0, 0, 0),
    "currentDensity": (0, -2, 0, 1, 0, 0, 0),
    "length":         (0, 1, 0, 0, 0, 0, 0),
    "area":           (0, 2, 0, 0, 0, 0, 0),
    "volume":         (0, 3, 0, 0, 0, 0, 0),
    "concentration":  (0, -3, 0, 0, 0, 0, 1),
    "substance":      (0, 0, 0, 0, 0, 0, 1),
    "charge":         (0, 0, 1, 1, 0, 0, 0),
    "charge_per_mole": (0, 0, 1, 1, 0, -1, 0),
    "temperature":    (0, 0, 0, 0, 1, 0, 0),
    "idealGasConstantDims": (1, 2, -2, 0, -1, -1, 0),
    "rho_factor":     (-1, -4, 3, 2, 0, -1, 0),
    "none":           (0, 0, 0, 0, 0, 0, 0),
}

UNITS = {
    "s": ("time", 0), "ms": ("time", -3), "us": ("time", -6),
    "V": ("voltage", 0), "mV": ("voltage", -3),
    "A": ("current", 0), "mA": ("current", -3), "uA": ("current", -6),
    "nA": ("current", -9), "pA": ("current", -12),
    "S": ("conductance", 0), "mS": ("conductance", -3),
    "uS": ("conductance", -6), "nS": ("conductance", -9),
    "S_per_cm2": ("conductanceDensity", 4),
    "mS_per_cm2": ("conductanceDensity", 1),
    "F": ("capacitance", 0), "uF": ("capacitance", -6),
    "nF": ("capacitance", -9), "pF": ("capacitance", -12),
    "uF_per_cm2": ("specificCapacitance", -2),
    "ohm": ("resistance", 0), "kohm": ("resistance", 3),
    "Mohm": ("resistance", 6),
    "ohm_cm": ("resistivity", -2), "kohm_cm": ("resistivity", 1),
    "m": ("length", 0), "cm": ("length", -2), "um": ("length", -6),
    "mol_per_m3": ("concentration", 0), "mol_per_cm3": ("concentration", 6),
    "M": ("concentration", 3), "mM": ("concentration", 0),
    "per_s": ("per_time", 0), "per_ms": ("per_time", 3),
    "Hz": ("per_time", 0),
    "degC": ("temperature", 0), "K": ("temperature", 0),
}

_ALIASES = {
    "millisecond": "ms", "milliseconds": "ms",
    "second": "s", "seconds": "s",
    "millivolt": "mV", "millivolts": "mV", "volt": "V",
    "milliampere": "mA", "microampere": "uA",
    "nanoampere": "nA", "picoampere": "pA",
    "siemens": "S", "millisiemens": "mS",
    "microsiemens": "uS", "nanosiemens": "nS",
    "S/cm2": "S_per_cm2", "S/cm²": "S_per_cm2",
    "mS/cm2": "mS_per_cm2",
    "uF/cm2": "uF_per_cm2", "µF/cm2": "uF_per_cm2", "µF/cm²": "uF_per_cm2",
    "ohm.cm": "ohm_cm", "ohm·cm": "ohm_cm", "kohm.cm": "kohm_cm",
    "micrometer": "um", "µm": "um",
}


def normalize_unit(unit_str):
    if not unit_str:
        return None
    s = str(unit_str).strip()
    return _ALIASES.get(s, s)


def unit_to_dimension(unit_str):
    norm = normalize_unit(unit_str)
    if norm is None:
        return "none"
    entry = UNITS.get(norm)
    return entry[0] if entry else "none"


# ── SymPy → LEMS expression printer ─────────────────────────────────

def sympy_to_lems(expr_str, parameters=None):
    """Convert a TVBO equation RHS string (or SymPy expr) to LEMS syntax.

    Parameters
    ----------
    expr_str : str or sympy.Basic
        Equation RHS to convert.
    parameters : list of str, optional
        Model symbol names (parameters, state variables, etc.) to inject as
        SymPy Symbols before parsing, overriding any conflicting built-ins
        (e.g. ``I``, ``gamma``, ``lambda``).
    """
    if expr_str is None or (isinstance(expr_str, str) and not expr_str):
        return ""
    from tvbo.codegen.code import render_expression
    return render_expression(expr_str, format="lems", parameters=parameters)


def inline_model_functions(expr, dynamics, all_names):
    """Inline model-defined functions into a SymPy expression.

    LEMS has no user-defined function mechanism, so calls like ``Sigm(y1 - y2)``
    must be expanded to their body (e.g. ``2*e0/(1 + exp(r*(v0 - (y1-y2))))``)
    before the expression is printed.

    Parameters
    ----------
    expr : sympy.Basic
        Already-parsed SymPy expression that may contain calls to model functions.
    dynamics : Dynamics
        The model whose ``functions`` dict holds body + formal arguments.
    all_names : list of str
        All symbol names in scope (parameters, state variables, …) so the body
        is parsed with the correct local dict.
    """
    from sympy import Function, Symbol
    from tvbo.parse.expression import parse_eq

    functions = getattr(dynamics, 'functions', None) or {}
    for fname, fn_obj in functions.items():
        arguments = getattr(fn_obj, 'arguments', None) or []
        arg_names = [getattr(a, 'name', str(a)) for a in arguments]
        rhs_str = getattr(getattr(fn_obj, 'equation', None), 'rhs', None)
        if not rhs_str or not arg_names:
            continue
        arg_syms = [Symbol(n) for n in arg_names]
        body = parse_eq(str(rhs_str), parameters=list(all_names) + arg_names)
        fn_cls = Function(fname)
        expr = expr.replace(
            fn_cls,
            lambda *actual_args, _body=body, _syms=arg_syms:
                _body.xreplace(dict(zip(_syms, actual_args)))
        )
    return expr


# ── Helpers ──────────────────────────────────────────────────────────

def safe_id(s):
    """Make a string safe for XML id attribute."""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", str(s or "id0"))
    return ("_" + s) if s[0].isdigit() else s


def _dynamics_has_time_units(params, svs, dvs):
    """Check if dynamics equations (TimeDerivatives) use time-dimensioned params.

    Only parameters directly referenced in TD equations — or in non-Piecewise
    DerivedVariables that feed into TDs — count.  Parameters that appear *only*
    in Piecewise conditions (e.g. ``pulse_delay``, ``switch_time``) are timing /
    stimulus parameters and do NOT indicate that the model equations carry
    physical time normalisation.
    """
    from tvbo.utils.units import unit_has_time_dimension

    # Collect TD equation RHS strings
    td_rhs_parts = []
    for sv in svs.values():
        eq = getattr(sv, 'equation', None)
        rhs = getattr(eq, 'rhs', None) if eq else None
        if rhs:
            td_rhs_parts.append(str(rhs))
    td_text = ' '.join(td_rhs_parts)

    def _name_in(name, text):
        return bool(re.search(r'\b' + re.escape(name) + r'\b', text))

    # Parameters directly in TD equations
    td_params = set()
    for pname in params:
        if _name_in(str(pname), td_text):
            td_params.add(str(pname))

    # Expand through non-Piecewise DVs that are used in TD equations:
    # if a DV feeds into a TD and is NOT Piecewise, its referenced params
    # effectively contribute to the TD value dimension.
    for dv_name, dv in (dvs or {}).items():
        if not _name_in(str(dv_name), td_text):
            continue
        eq = getattr(dv, 'equation', None)
        rhs = getattr(eq, 'rhs', None) if eq else None
        if not rhs or 'Piecewise' in str(rhs):
            continue
        rhs_str = str(rhs)
        for pname in params:
            if _name_in(str(pname), rhs_str):
                td_params.add(str(pname))

    return any(
        unit_has_time_dimension(getattr(params[k], "unit", None))
        for k in td_params if k in params
    ) or any(
        unit_has_time_dimension(getattr(sv, "unit", None))
        for sv in svs.values()
    )


def _normalize_edge_params(params):
    """Normalize edge parameters to a flat ``{name: param_obj}`` dict.

    Handles both the dict form ``{weight: {value: 1.0}}`` and the list form
    ``[{weight: {value: 1.0}}, ...]`` that YAML may produce.
    """
    if not params:
        return {}
    if isinstance(params, list):
        result = {}
        for item in params:
            if isinstance(item, dict):
                for k, v in item.items():
                    result[str(k)] = v
        return result
    # dict or dict-like (LinkML JsonObj)
    try:
        return {str(k): v for k, v in params.items()}
    except AttributeError:
        return {}


def validate_lems_xml(xml_string):
    """Validate a LEMS XML string using PyLEMS.

    Raises if the XML is not valid LEMS.
    """
    import os
    import tempfile

    from lems.model.model import Model

    with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
        f.write(xml_string)
        fname = f.name
    try:
        Model().import_from_file(fname)
    finally:
        os.unlink(fname)


# ── Network context builder ──────────────────────────────────────────

def _build_network_context(experiment):
    """Extract multi-population network structure for LEMS rendering.

    Inspects the experiment's network for explicit nodes, edges, and
    coupling definitions.  When present, builds the population, synapse,
    connection, and input metadata needed by the LEMS templates.

    Returns None if the network is a simple single-population case
    (no explicit nodes/edges), otherwise returns a dict with keys:
    ``populations``, ``synapses``, ``connections``, ``inputs``,
    ``cell_types``.
    """
    from tvbo.classes.dynamics import Dynamics

    network = getattr(experiment, "network", None)
    if network is None:
        return None

    nodes = getattr(network, "nodes", None) or []
    edges = getattr(network, "edges", None) or []

    if not nodes or not edges:
        return None

    # ── Resolve dynamics for each node ──
    # Build a dict of unique dynamics models and group nodes into populations
    default_dyn = experiment.dynamics
    dynamics_lib = getattr(network, "dynamics", None) or {}

    cell_types = {}   # dyn_name -> Dynamics object
    populations = []  # list of {id, component, size, node_ids}
    node_pop_map = {} # node_id -> (pop_id, index_within_pop)

    # Group nodes by their dynamics name
    from collections import OrderedDict
    groups = OrderedDict()  # dyn_name -> [node_obj, ...]
    for node in nodes:
        node_dyn = getattr(node, "dynamics", None)
        if node_dyn:
            dyn_name = getattr(node_dyn, "name", None) or str(node_dyn)
        else:
            dyn_name = getattr(default_dyn, "name", None) or "dynamics"
        groups.setdefault(dyn_name, []).append(node)

    for dyn_name, group_nodes in groups.items():
        # Resolve the Dynamics object
        if dyn_name in dynamics_lib:
            dyn_obj = dynamics_lib[dyn_name]
        elif dyn_name == getattr(default_dyn, "name", None):
            dyn_obj = default_dyn
        else:
            dyn_obj = Dynamics.from_db(dyn_name)
        cell_types[dyn_name] = dyn_obj

        pop_id = safe_id(dyn_name) + "_pop"
        node_ids = []
        for idx, node in enumerate(group_nodes):
            nid = getattr(node, "id", idx)
            node_pop_map[nid] = (pop_id, idx)
            node_ids.append(nid)

        populations.append({
            "id": pop_id,
            "component": safe_id(dyn_name) + "_inst",
            "size": len(group_nodes),
            "node_ids": node_ids,
            "dyn_name": dyn_name,
        })

    # ── Extract synapse definitions from edges ──
    synapses = []     # list of {id, type, params}
    connections = []  # list of {from_pop, from_idx, to_pop, to_idx, synapse, weight, delay}
    synapse_set = {}  # dedup key -> synapse_id

    for edge_idx, edge in enumerate(edges):
        src = getattr(edge, "source", None)
        tgt = getattr(edge, "target", None)
        if src is None or tgt is None:
            continue

        src = int(src)
        tgt = int(tgt)
        if src not in node_pop_map or tgt not in node_pop_map:
            continue

        src_pop, src_idx = node_pop_map[src]
        tgt_pop, tgt_idx = node_pop_map[tgt]

        # Get edge coupling/synapse info
        edge_coupling = getattr(edge, "coupling", None)
        edge_dynamics = getattr(edge, "dynamics", None)

        # ── Extract ALL edge parameters ────────────────────────────────
        # Supports both dict {weight: {value:1.0}} and list [{weight: ...}]
        # formats.  Separates connection-level params (weight, delay) from
        # synapse definition params (everything else, kept with their units).
        edge_params = _normalize_edge_params(getattr(edge, "parameters", None))
        weight = None
        delay = None
        delay_unit = None
        syn_params = {}  # {name: {'value': v, 'unit': u}}
        for pname, pval in edge_params.items():
            pname = str(pname)
            val = getattr(pval, 'value', pval)
            unit = getattr(pval, 'unit', None)
            if pname == 'weight':
                weight = float(val) if val is not None else None
            elif pname == 'delay':
                delay = float(val) if val is not None else None
                delay_unit = unit
            else:
                syn_params[pname] = {'value': val, 'unit': unit}

        # Merge coupling-definition params into syn_params (edge params take precedence)
        syn_type = None
        if edge_coupling:
            coup_name = getattr(edge_coupling, "name", None) or str(edge_coupling)
            syn_type = coup_name
            coup_params = _normalize_edge_params(getattr(edge_coupling, "parameters", None))
            for k, v in coup_params.items():
                k = str(k)
                if k not in ('weight', 'delay') and k not in syn_params:
                    val = getattr(v, 'value', v)
                    unit = getattr(v, 'unit', None)
                    syn_params[k] = {'value': val, 'unit': unit}

        # Resolve edge dynamics to a Dynamics object
        resolved_edge_dyn = None
        if edge_dynamics:
            dyn_name = getattr(edge_dynamics, 'name', None) or str(edge_dynamics)
            if dyn_name in dynamics_lib:
                resolved_edge_dyn = dynamics_lib[dyn_name]
            elif dyn_name:
                try:
                    resolved_edge_dyn = Dynamics.from_db(dyn_name)
                except Exception:
                    pass

        if syn_type is None and edge_dynamics:
            syn_type = getattr(edge_dynamics, 'name', None) or f"syn_edge{edge_idx}"
        if syn_type is None:
            syn_type = f"syn{edge_idx}"

        # Build synapse dedup key — include param values (not units) for dedup
        syn_key = (
            syn_type,
            tuple(sorted(
                (k, pinfo['value'] if isinstance(pinfo, dict) else pinfo)
                for k, pinfo in syn_params.items()
            )),
        )
        if syn_key not in synapse_set:
            syn_id = safe_id(syn_type)
            synapse_set[syn_key] = syn_id
            synapses.append({
                "id": syn_id,
                "type": syn_type,
                "params": syn_params,       # {name: {'value': v, 'unit': u}}
                "edge_dynamics": edge_dynamics,
                "edge_coupling": edge_coupling,
                "resolved_dyn": resolved_edge_dyn,
            })
        syn_id = synapse_set[syn_key]

        connections.append({
            "from_pop": src_pop,
            "from_idx": src_idx,
            "to_pop": tgt_pop,
            "to_idx": tgt_idx,
            "synapse": syn_id,
            "weight": float(weight) if weight is not None else None,
            "delay": float(delay) if delay is not None else None,
            "delay_unit": delay_unit,
        })

    # ── Extract input specifications ──
    inputs = []
    # Inputs come from node-level parameters or experiment-level stimulus.
    # For now, check if any node dynamics has a Piecewise I_ext that encodes
    # a pulse generator — this is the pattern used in QMD examples.
    # More sophisticated input handling (explicit stimulus objects) can be
    # added later.

    return {
        "populations": populations,
        "synapses": synapses,
        "connections": connections,
        "inputs": inputs,
        "cell_types": cell_types,
        "node_pop_map": node_pop_map,
    }


# ── Shared template context ───────────────────────────────────────────

def build_lems_context(experiment):
    """Build the shared rendering context passed to all LEMS Mako templates.

    Extracts and pre-computes every variable that LEMS templates need —
    model objects, name lists for safe SymPy parsing, expression helpers, and
    integration/network scalars.  All templates receive this dict via
    ``template.render(**build_lems_context(experiment))``.

    Parameters
    ----------
    experiment : SimulationExperiment

    Returns
    -------
    dict
        Keys: ``dyn``, ``dyn_id``, ``params``, ``svs``, ``dvs``, ``events``,
        ``coupling_inputs``, ``coupling_meta``, ``coupling_params``,
        ``coupling_pre_rhs``, ``coupling_post_rhs``, ``coupling_global``,
        ``sv_names_set``, ``n_nodes``, ``dt``, ``duration``,
        ``lems_expr`` (callable), ``_parse_piecewise`` (callable),
        ``lems_dim`` (callable), ``safe_id`` (callable).
    """
    from sympy import Piecewise, S as sympy_S, Eq as sympy_Eq
    from sympy.functions.elementary.piecewise import piecewise_fold
    from sympy.core.basic import Basic as _SympyBasic
    from tvbo.parse.expression import parse_eq

    dyn = experiment.dynamics

    # Resolve model reference: if dynamics has a name but no state variables,
    # it was loaded from a YAML `model: ModelName` reference and needs to be
    # resolved from the database.
    if dyn.name and not (dyn.state_variables or dyn.parameters):
        from tvbo.classes.dynamics import Dynamics
        dyn = Dynamics.from_db(dyn.name)

    dyn_id = safe_id(dyn.name or "dynamics")

    params = dyn.parameters or {}
    svs = dyn.state_variables or {}
    dvs = getattr(dyn, "derived_variables", None) or {}
    events = getattr(dyn, "events", None) or {}
    coupling_inputs = getattr(dyn, "coupling_inputs", None) or []

    # Coupling metadata
    coupling_meta = getattr(experiment, "coupling", None)
    coupling_params = {}
    coupling_pre_rhs = None
    coupling_post_rhs = None
    coupling_global = 1.0
    if coupling_meta is not None:
        coupling_params = dict(getattr(coupling_meta, "parameters", None) or {})
        pre_eq = getattr(getattr(coupling_meta, "pre_expression", None), "rhs", None)
        post_eq = getattr(getattr(coupling_meta, "post_expression", None), "rhs", None)
        coupling_pre_rhs = str(pre_eq) if pre_eq else None
        coupling_post_rhs = str(post_eq) if post_eq else None
        g_param = coupling_params.get("global_coupling") or coupling_params.get("a")
        coupling_global = getattr(g_param, "value", 1.0) if g_param else 1.0

    sv_names_set = set(str(k) for k in svs.keys())
    if coupling_pre_rhs is None:
        first_sv = next(iter(svs), None)
        coupling_pre_rhs = f"{first_sv}_j" if first_sv else "x_j"
    if coupling_post_rhs is None:
        coupling_post_rhs = "global_coupling * pre"

    integration = getattr(experiment, "integration", None)
    network = getattr(experiment, "network", None)
    n_nodes = int(network.number_of_nodes) if network and hasattr(network, "number_of_nodes") else 1
    dt = integration.step_size if integration else 0.01
    duration = integration.duration if integration else 1000.0
    raw_ts = (getattr(integration, "time_scale", None) or "ms") if integration else "ms"
    from tvbo.utils.units import normalize_unit
    ts_enum = normalize_unit(str(raw_ts)) or str(raw_ts)
    # With abbreviation-based enum, ts_enum is already "s", "ms", "us" etc.
    time_scale = ts_enum if ts_enum in ("s", "ms", "us") else "ms"

    # ── Determine whether equations need / SEC ──
    # If parameters used in TimeDerivative equations carry time-bearing
    # units, the equations already contain physical time normalisation
    # (e.g., /tau_e where tau_e has unit=ms).  In that case / SEC would
    # double-count.  Parameters that appear *only* in Piecewise conditions
    # (pulse timing, switch times) are excluded from this check.
    from tvbo.utils.units import unit_has_time_dimension, unit_to_lems_dimension
    _model_has_time_units = _dynamics_has_time_units(params, svs, dvs)

    # All symbol names — override SymPy built-ins (I, gamma, lambda, …)
    all_names = (
        [str(k) for k in params.keys()]
        + [str(k) for k in svs.keys()]
        + [str(k) for k in dvs.keys()]
        + [str(ci) for ci in coupling_inputs]
        + ["c_pop0", "c_pop1", "local_coupling"]
        + ["pre", "gx", "post", "global_coupling"]   # coupling CT vars
        + [f"{sv}_j" for sv in svs.keys()]              # pre-synaptic symbols
        + [str(k) for k in coupling_params.keys()]       # coupling parameters
    )
    fn_names = list((getattr(dyn, "functions", None) or {}).keys())

    def lems_expr(e):
        """Parse (if needed), inline model functions, then print as LEMS."""
        if not isinstance(e, _SympyBasic):
            e = parse_eq(str(e), parameters=all_names, functions=fn_names)
        e = inline_model_functions(e, dyn, all_names)
        return sympy_to_lems(e, parameters=all_names)

    def _parse_piecewise(rhs_str):
        """Return [(condition_str, value_str)] if rhs is Piecewise, else None."""
        try:
            expr = parse_eq(str(rhs_str), parameters=all_names, functions=fn_names)
            # Rewrite Min/Max and similar forms to Piecewise when possible.
            expr = expr.rewrite(Piecewise)
            # Support wrapped forms like Q10*Piecewise(...) by folding to a
            # top-level Piecewise expression first.
            if not isinstance(expr, Piecewise) and expr.has(Piecewise):
                expr = piecewise_fold(expr)
            if not isinstance(expr, Piecewise):
                return None
            cases = []
            for val, cond in expr.args:
                # Exact-equality singularity guards (Eq(v, v0)) are numerically
                # measure-zero and can trigger parser issues in some jLEMS builds.
                # Drop them and keep the regular branch.
                if getattr(cond, "func", None) is sympy_Eq:
                    continue
                cond_str = None if cond == sympy_S.true else lems_expr(cond)
                val_str = lems_expr(val)
                cases.append((cond_str, val_str))
            return cases
        except Exception:
            return None

    label = getattr(experiment, 'label', None)
    sim_id = 'sim_' + (safe_id(label) if label else dyn_id)

    # ── Multi-population network context ──
    net_ctx = _build_network_context(experiment)

    ctx = dict(
        dyn=dyn,
        dyn_id=dyn_id,
        sim_id=sim_id,
        params=params,
        svs=svs,
        dvs=dvs,
        events=events,
        coupling_inputs=coupling_inputs,
        coupling_meta=coupling_meta,
        coupling_params=coupling_params,
        coupling_pre_rhs=coupling_pre_rhs,
        coupling_post_rhs=coupling_post_rhs,
        coupling_global=coupling_global,
        sv_names_set=sv_names_set,
        n_nodes=n_nodes,
        dt=dt,
        duration=duration,
        lems_expr=lems_expr,
        _parse_piecewise=_parse_piecewise,
        lems_dim=unit_to_lems_dimension,
        safe_id=safe_id,
        time_scale=time_scale,
        needs_sec=not _model_has_time_units,
        max_output_nodes=100,
        # Network context (None for single-population)
        net_ctx=net_ctx,
        has_network=net_ctx is not None,
    )

    # When multi-population network exists, build per-cell-type contexts
    if net_ctx:
        cell_contexts = {}
        for ct_name, ct_dyn in net_ctx["cell_types"].items():
            ct_params = ct_dyn.parameters or {}
            ct_svs = ct_dyn.state_variables or {}
            ct_dvs = getattr(ct_dyn, "derived_variables", None) or {}
            ct_events = getattr(ct_dyn, "events", None) or {}
            ct_coupling_inputs = getattr(ct_dyn, "coupling_inputs", None) or []
            ct_sv_names_set = set(str(k) for k in ct_svs.keys())

            ct_all_names = (
                [str(k) for k in ct_params.keys()]
                + [str(k) for k in ct_svs.keys()]
                + [str(k) for k in ct_dvs.keys()]
                + [str(ci) for ci in ct_coupling_inputs]
            )
            ct_fn_names = list((getattr(ct_dyn, "functions", None) or {}).keys())

            ct_has_time_units = _dynamics_has_time_units(
                ct_params, ct_svs, ct_dvs)

            _LEMS_CMP_RE = re.compile(r'\.(gt|lt|geq|leq|eq|neq)\.', re.IGNORECASE)

            def _make_ct_lems_expr(ct_dyn, ct_all_names, ct_fn_names):
                def ct_lems_expr(e):
                    e_str = str(e)
                    # LEMS comparison operators (.gt., .lt., etc.) are not SymPy parseable;
                    # return them as-is.
                    if _LEMS_CMP_RE.search(e_str):
                        return e_str
                    if not isinstance(e, _SympyBasic):
                        e = parse_eq(e_str, parameters=ct_all_names, functions=ct_fn_names)
                    e = inline_model_functions(e, ct_dyn, ct_all_names)
                    return sympy_to_lems(e, parameters=ct_all_names)
                return ct_lems_expr

            def _make_ct_parse_pw(ct_all_names, ct_fn_names, ct_lems_expr_fn):
                def ct_parse_pw(rhs_str):
                    try:
                        expr = parse_eq(str(rhs_str), parameters=ct_all_names, functions=ct_fn_names)
                        expr = expr.rewrite(Piecewise)
                        if not isinstance(expr, Piecewise) and expr.has(Piecewise):
                            expr = piecewise_fold(expr)
                        if not isinstance(expr, Piecewise):
                            return None
                        cases = []
                        for val, cond in expr.args:
                            if getattr(cond, "func", None) is sympy_Eq:
                                continue
                            cond_str = None if cond == sympy_S.true else ct_lems_expr_fn(cond)
                            val_str = ct_lems_expr_fn(val)
                            cases.append((cond_str, val_str))
                        return cases
                    except Exception:
                        return None
                return ct_parse_pw

            ct_lems_expr = _make_ct_lems_expr(ct_dyn, ct_all_names, ct_fn_names)
            ct_parse_pw = _make_ct_parse_pw(ct_all_names, ct_fn_names, ct_lems_expr)

            # Detect threshold (spike-emitting) events for this node type
            threshold_event_names = [
                k for k, v in ct_events.items()
                if getattr(getattr(v, 'condition', None), 'rhs', None) is not None
            ]

            cell_contexts[ct_name] = {
                "dyn": ct_dyn,
                "dyn_id": safe_id(ct_name),
                "params": ct_params,
                "svs": ct_svs,
                "dvs": ct_dvs,
                "events": ct_events,
                "coupling_inputs": ct_coupling_inputs,
                "sv_names_set": ct_sv_names_set,
                "needs_sec": not ct_has_time_units,
                "lems_expr": ct_lems_expr,
                "_parse_piecewise": ct_parse_pw,
                # Node-level flags for LEMS network rendering
                "is_synapse": False,
                "has_threshold_events": bool(threshold_event_names),
                "threshold_event_names": threshold_event_names,
            }
        ctx["cell_contexts"] = cell_contexts

        # ── Also build cell_contexts for edge dynamics (synapse ComponentTypes) ──
        for syn in net_ctx.get("synapses", []):
            rdyn = syn.get("resolved_dyn")
            if rdyn and syn["id"] not in cell_contexts:
                ct_dyn = rdyn
                ct_name = syn["id"]
                ct_params = ct_dyn.parameters or {}
                ct_svs = ct_dyn.state_variables or {}
                ct_dvs = getattr(ct_dyn, "derived_variables", None) or {}
                ct_events = getattr(ct_dyn, "events", None) or {}
                ct_coupling_inputs = getattr(ct_dyn, "coupling_inputs", None) or []
                ct_sv_names_set = set(str(k) for k in ct_svs.keys())
                ct_fn_names = list((getattr(ct_dyn, "functions", None) or {}).keys())
                ct_has_time_units = _dynamics_has_time_units(
                    ct_params, ct_svs, ct_dvs)
                ct_all_names = (
                    [str(k) for k in ct_params.keys()]
                    + [str(k) for k in ct_svs.keys()]
                    + [str(k) for k in ct_dvs.keys()]
                    + [str(ci) for ci in ct_coupling_inputs]
                )
                ct_lems_expr = _make_ct_lems_expr(ct_dyn, ct_all_names, ct_fn_names)
                ct_parse_pw = _make_ct_parse_pw(ct_all_names, ct_fn_names, ct_lems_expr)

                # Synapse-specific: events without a condition are external spike triggers
                external_event_names = [
                    k for k, ev in ct_events.items()
                    if not getattr(getattr(ev, 'condition', None), 'rhs', None)
                ]
                # Exposure / InstanceRequirement detection
                has_i_exposure = 'i' in ct_dvs or any(str(k) == 'i' for k in ct_svs)
                has_v_req = any(str(ci) == 'v' for ci in ct_coupling_inputs)

                cell_contexts[ct_name] = {
                    "dyn": ct_dyn,
                    "dyn_id": ct_name,
                    "params": ct_params,
                    "svs": ct_svs,
                    "dvs": ct_dvs,
                    "events": ct_events,
                    "coupling_inputs": ct_coupling_inputs,
                    "sv_names_set": ct_sv_names_set,
                    "needs_sec": not ct_has_time_units,
                    "lems_expr": ct_lems_expr,
                    "_parse_piecewise": ct_parse_pw,
                    # Synapse-specific extras
                    "is_synapse": True,
                    "has_i_exposure": has_i_exposure,
                    "has_v_req": has_v_req,
                    "external_event_names": external_event_names,
                    "has_threshold_events": False,
                    "threshold_event_names": [],
                }
        # Re-store with synapse contexts included
        ctx["cell_contexts"] = cell_contexts

    return ctx


# ── Adapter ──────────────────────────────────────────────────────────

class NeuroMLAdapter(BaseAdapter):
    """Adapter for exporting a SimulationExperiment (or bare Dynamics) as LEMS XML.

    Supports both a single monolithic file and a canonical three-file split:

    * ``render_dynamics()``   → standalone ComponentType definitions
    * ``render_network()``    → Network component (may include a dynamics file)
    * ``render_simulation()`` → LEMS Simulation block (may include a network file)
    * ``render_code()``       → monolithic all-in-one LEMS file (default)
    * ``render_neuroml()``    → NeuroML v2 document (``<neuroml>`` root)
    * ``render_lems_wrapper()`` → LEMS wrapper for a NeuroML file
    * ``export(dir)``         → write file(s) to disk, optionally validate

    ``render('lems')`` produces a self-contained ``<Lems>`` file.
    ``render('neuroml')`` produces a ``<neuroml>`` document with custom
    ComponentType definitions — no mapping to native NeuroML cell types.

    All ``render_*`` methods pass a fully pre-computed context via
    :func:`build_lems_context` so templates stay logic-free.
    """

    TEMPLATE = "neuroml/tvbo-neuroml-lems.xml.mako"
    NEUROML_TEMPLATE = "neuroml/tvbo-neuroml-document.xml.mako"
    LEMS_WRAPPER_TEMPLATE = "neuroml/tvbo-lems-wrapper.xml.mako"
    DYNAMICS_TEMPLATE = "neuroml/tvbo-neuroml-dynamics.xml.mako"
    NETWORK_TEMPLATE = "neuroml/tvbo-neuroml-network.xml.mako"
    SIMULATION_TEMPLATE = "neuroml/tvbo-neuroml-simulation.xml.mako"

    def __init__(self, source=None):
        from tvbo.classes.dynamics import Dynamics
        from tvbo.classes.experiment import SimulationExperiment

        if source is None:
            self.experiment = None
            return
        if isinstance(source, Dynamics):
            source = SimulationExperiment(dynamics=source)
        super().__init__(source)

    def _ctx(self, **extra):
        """Return ``build_lems_context()`` merged with any caller-supplied extras."""
        ctx = build_lems_context(self.experiment)
        ctx.update(extra)
        return ctx

    def render_code(self, **kwargs) -> str:
        """Render a complete, self-contained LEMS simulation file (``<Lems>`` root)."""
        from tvbo import templates
        template = templates.lookup.get_template(self.TEMPLATE)
        return template.render(experiment=self.experiment, **self._ctx(**kwargs))

    def render_neuroml(self, **kwargs) -> str:
        """Render a NeuroML v2 document (``<neuroml>`` root).

        Uses custom ``<ComponentType>`` definitions for the dynamics model
        rather than mapping to native NeuroML cell types.  The output
        contains ComponentType definitions, Component instances, and
        a ``<network>`` with populations.

        To run the output, pair it with a LEMS simulation wrapper
        generated by :meth:`render_lems_wrapper`.
        """
        from tvbo import templates
        template = templates.lookup.get_template(self.NEUROML_TEMPLATE)
        return template.render(experiment=self.experiment, **self._ctx(**kwargs))

    def render_lems_wrapper(self, neuroml_file=None, **kwargs) -> str:
        """Render a LEMS simulation wrapper for a NeuroML file.

        The wrapper includes standard NeuroML type files and the given
        NeuroML document, then defines a ``<Simulation>`` targeting the
        network defined in the ``.nml`` file.

        Parameters
        ----------
        neuroml_file : str or None
            Filename of the NeuroML document to include.
        """
        from tvbo import templates
        template = templates.lookup.get_template(self.LEMS_WRAPPER_TEMPLATE)
        return template.render(
            experiment=self.experiment,
            neuroml_file=neuroml_file,
            **self._ctx(**kwargs),
        )

    def render_dynamics(self, **kwargs) -> str:
        """Render a standalone LEMS file with only ComponentType definitions.

        The output is a valid LEMS document containing dimensions, units,
        the dynamics ``ComponentType``, the ``Coupling`` ``ComponentType``,
        and the default ``Component`` instances.  No ``Network`` or
        ``Simulation`` elements are included, making it suitable for inclusion
        in larger LEMS documents via ``<Include file="..."/>``.
        """
        from tvbo import templates
        template = templates.lookup.get_template(self.DYNAMICS_TEMPLATE)
        return template.render(experiment=self.experiment, **self._ctx(**kwargs))

    def render_network(self, dynamics_file=None, **kwargs) -> str:
        """Render a LEMS Network document.

        Parameters
        ----------
        dynamics_file : str or None
            If given, an ``<Include file="..."/>`` referencing that filename is
            prepended so the document can be used standalone.
        """
        from tvbo import templates
        template = templates.lookup.get_template(self.NETWORK_TEMPLATE)
        return template.render(
            experiment=self.experiment,
            dynamics_file=dynamics_file,
            **self._ctx(**kwargs),
        )

    def render_simulation(self, network_file=None, **kwargs) -> str:
        """Render a LEMS Simulation document.

        Parameters
        ----------
        network_file : str or None
            If given, an ``<Include file="..."/>`` referencing that filename is
            prepended so the document can be used standalone.
        """
        from tvbo import templates
        template = templates.lookup.get_template(self.SIMULATION_TEMPLATE)
        return template.render(
            experiment=self.experiment,
            network_file=network_file,
            **self._ctx(**kwargs),
        )

    def export(self, dir, format='lems', split=False, validate=True, **kwargs) -> dict:
        """Export LEMS or NeuroML XML to a directory.

        Parameters
        ----------
        dir : str or Path
            Output directory (created if needed).
        format : str
            ``'lems'`` (default) — LEMS output (monolithic or split).
            ``'neuroml'`` — NeuroML ``.nml`` document + LEMS simulation wrapper.
        split : bool
            Only used for ``format='lems'``.
            ``False`` (default) — one monolithic ``{prefix}_simulation.xml``.
            ``True`` — three canonical files:

            * ``{prefix}_dynamics.xml``   — ComponentType definitions
            * ``{prefix}_network.xml``    — Network (includes dynamics)
            * ``{prefix}_simulation.xml`` — Simulation (includes network)
        validate : bool
            Run PyLEMS validation on every written file (default ``True``).

        Returns
        -------
        dict
            Mapping of role to absolute file paths.
        """
        from pathlib import Path

        out_dir = Path(dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = (
            self.experiment.get_experiment_file_prefix()
            if hasattr(self.experiment, "get_experiment_file_prefix")
            else (self.experiment.dynamics.name or "model")
        )

        if format.lower() in ('neuroml', 'nml'):
            nml_name = f"{prefix}.nml"
            lems_name = f"LEMS_{prefix}.xml"
            nml_xml = self.render_neuroml(**kwargs)
            lems_xml = self.render_lems_wrapper(
                neuroml_file=nml_name, **kwargs
            )
            (out_dir / nml_name).write_text(nml_xml)
            (out_dir / lems_name).write_text(lems_xml)
            return {
                'neuroml': str(out_dir / nml_name),
                'simulation': str(out_dir / lems_name),
            }

        if split:
            dyn_name = f"{prefix}_dynamics.xml"
            net_name = f"{prefix}_network.xml"
            sim_name = f"{prefix}_simulation.xml"
            files = [
                (dyn_name, self.render_dynamics(**kwargs)),
                (net_name, self.render_network(dynamics_file=dyn_name, **kwargs)),
                (sim_name, self.render_simulation(network_file=net_name, **kwargs)),
            ]
            roles = ("dynamics", "network", "simulation")
            paths = {}
            for (fname, xml), role in zip(files, roles):
                fpath = out_dir / fname
                fpath.write_text(xml)
                paths[role] = str(fpath)
            # Individual split files aren't standalone-valid LEMS (they
            # reference external ComponentTypes via <Include>).  Validate
            # only the dynamics file, which IS self-contained.
            if validate:
                validate_lems_xml(files[0][1])
        else:
            sim_name = f"{prefix}_simulation.xml"
            xml = self.render_code(**kwargs)
            fpath = out_dir / sim_name
            fpath.write_text(xml)
            if validate:
                validate_lems_xml(xml)
            paths = {"simulation": str(fpath)}

        return paths

    def validate(self, xml_string=None):
        """Validate rendered LEMS XML with PyLEMS. Returns True or raises."""
        if xml_string is None:
            xml_string = self.render_code()
        validate_lems_xml(xml_string)
        return True

    def run(self, **kwargs) -> "ExperimentResult":
        """Run the LEMS simulation via jNeuroML.

        Uses a fully self-contained monolithic LEMS file with all dimensions,
        units, and infrastructure types (Simulation, OutputFile, OutputColumn)
        defined inline.  This avoids the jNeuroML double-read bug that occurs
        when external NeuroML type files are included via ``<Include>``.

        Returns
        -------
        ExperimentResult
            Simulation results loaded from jNeuroML output files.
        """
        import subprocess
        import tempfile
        from pathlib import Path

        import numpy as np
        import xarray as xr

        from tvbo.data.types import ExperimentResult, SimulationResult

        ctx = build_lems_context(self.experiment)
        sv_names = list(ctx['svs'].keys())
        dyn_id = ctx['dyn_id']
        net_ctx = ctx.get('net_ctx')
        cell_contexts = ctx.get('cell_contexts', {})

        xml = self.render_code(**kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            lems_file = tmpdir / "simulation.xml"
            lems_file.write_text(xml)
            (tmpdir / "results").mkdir()

            # Use jNeuroML JAR directly for portability
            from pyneuroml import JNEUROML_VERSION
            import pyneuroml
            jar_dir = Path(pyneuroml.__file__).parent / "lib"
            jar = jar_dir / f"jNeuroML-{JNEUROML_VERSION}-jar-with-dependencies.jar"
            result = subprocess.run(
                ["java", "-jar", str(jar), "simulation.xml", "-nogui"],
                capture_output=True, text=True, cwd=str(tmpdir), timeout=600,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"jNeuroML execution failed (rc={result.returncode}):\n"
                    f"{result.stderr[-1000:]}"
                )

            # Load output: results/{dyn_id}.dat
            dat_files = list(tmpdir.glob("results/*.dat"))
            if not dat_files:
                raise RuntimeError(
                    "jNeuroML produced no output files. stderr:\n"
                    f"{result.stderr[-500:]}"
                )

            raw = np.loadtxt(str(dat_files[0]))

        time_data = raw[:, 0]
        values_data = raw[:, 1:]

        if net_ctx and cell_contexts:
            # Multi-population: rebuild column labels from the ordered out_cols
            # produced by the template (same order as the OutputFile columns).
            col_names = []
            for pop in net_ctx['populations']:
                ct = cell_contexts.get(pop['dyn_name'], {})
                if ct.get('is_synapse'):
                    continue  # synapse populations don't have output columns
                for sv_name in ct.get('svs', {}):
                    for idx in range(pop['size']):
                        col_names.append(f"{pop['id']}[{idx}]/{sv_name}")

            da = xr.DataArray(
                data=values_data,
                dims=['time', 'quantity'],
                coords={'time': time_data, 'quantity': col_names},
            )
        else:
            # Single population: existing flat layout
            da = xr.DataArray(
                data=values_data.reshape(-1, len(sv_names), 1),
                dims=['time', 'variable', 'node'],
                coords={
                    'time': time_data,
                    'variable': sv_names,
                    'node': ['0'],
                },
            )

        sim = SimulationResult(data=da)
        return ExperimentResult(
            integration=sim,
            source=self.experiment,
            name=getattr(self.experiment, 'label', None),
        )