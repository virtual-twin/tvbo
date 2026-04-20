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


def _dynamics_has_physical_units(params, svs, td_param_names=None):
    """Return True if any dynamics-relevant parameter or state variable has a
    non-dimensionless LEMS dimension (voltage, conductance, capacitance, etc.).

    When physical units are present, the equations are fully dimensioned in LEMS
    and do NOT need ``/ SEC`` time scaling.

    Parameters that only appear in Piecewise conditions (e.g. ``pulse_delay``,
    ``pulse_duration``) are NOT dynamics-relevant and should be excluded via
    *td_param_names*.
    """
    from tvbo.utils.units import unit_to_lems_dimension
    for pname, p in params.items():
        if td_param_names is not None and str(pname) not in td_param_names:
            continue
        dim = unit_to_lems_dimension(getattr(p, 'unit', None))
        if dim != 'none':
            return True
    for sv in svs.values():
        dim = unit_to_lems_dimension(getattr(sv, 'unit', None))
        if dim != 'none':
            return True
    return False


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
    ) or _dynamics_has_physical_units(params, svs, td_params)


def _build_regime_data(events):
    """Detect spike events and build Regime rendering data.

    When an event has both a ``condition`` (threshold test) and an ``affect``
    (state reset assignments), we render it as a pair of LEMS Regimes
    (integrating / refractory) instead of a flat ``<OnCondition>``.

    Returns ``None`` when no Regime rendering is needed, otherwise a dict::

        {
            'condition':  str,          # e.g. "v > thresh"
            'assignments': [(lhs, rhs), ...],  # parsed affect assignments
            'reset_vars': set,          # SVs clamped in refractory (no TD)
        }

    A variable is "clamped" (excluded from refractory TimeDerivatives) when
    the assignment is a pure reset (``v = reset``, LHS absent from RHS).
    A variable that receives a bump (``w = w + b``, LHS appears in RHS)
    still evolves via its TimeDerivative in the refractory regime.
    """
    for _ev_name, ev in (events or {}).items():
        cond = getattr(ev, 'condition', None)
        affect = getattr(ev, 'affect', None)
        cond_rhs = getattr(cond, 'rhs', None) if cond else None
        affect_rhs = getattr(affect, 'rhs', None) if affect else None

        if not (cond_rhs and affect_rhs):
            continue

        assignments = []
        reset_vars = set()
        for piece in str(affect_rhs).split(";"):
            piece = piece.strip()
            if "=" not in piece:
                continue
            lhs, rhs_val = piece.split("=", 1)
            lhs, rhs_val = lhs.strip(), rhs_val.strip()
            assignments.append((lhs, rhs_val))
            # If LHS doesn't appear in RHS → it's a reset / clamp
            if not re.search(r'\b' + re.escape(lhs) + r'\b', rhs_val):
                reset_vars.add(lhs)

        return {
            'condition': str(cond_rhs),
            'assignments': assignments,
            'reset_vars': reset_vars,
        }
    return None


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


# ── Standard NeuroML type rendering ──────────────────────────────────

def _uses_neuroml_types(dynamics):
    """Return True if the dynamics uses standard NeuroML types (iri: neuroml:*)."""
    iri = getattr(dynamics, 'iri', None) or ''
    return iri.startswith('neuroml:')


_TVBO_TO_NML_UNIT = {
    'mmol_per_m3': 'mM',
}
"""Map TVBO canonical unit names to NeuroML/LEMS unit symbols where they differ.

Most TVBO UnitEnum names already match NeuroML symbols exactly (mV, ms, pS, …).
Only add entries here for genuine mismatches.  The authoritative NeuroML unit list
lives in ``NeuroML2CoreTypes/NeuroMLCoreDimensions.xml``.
"""


def _nml_attr(param, default=''):
    """Extract a NeuroML XML attribute value from a TVBO Parameter.

    Preferred format: ``{ value: 10, unit: pS }`` → ``"10 pS"``.
    String-valued attributes (ion, species): ``{ description: k }`` → ``"k"``.
    Legacy: ``description: "nml:10pS"`` still works as fallback.
    """
    if param is None:
        return str(default)
    desc = str(getattr(param, 'description', '') or '')
    unit = str(getattr(param, 'unit', '') or '')
    val = getattr(param, 'value', None)

    # Legacy fallback: description with nml: prefix
    if desc.startswith('nml:'):
        return desc[4:]

    # Numeric value + unit
    if val is not None:
        if isinstance(val, float) and val == int(val) and abs(val) < 1e15:
            formatted = str(int(val))
        else:
            formatted = str(val)
        if unit:
            nml_unit = _TVBO_TO_NML_UNIT.get(unit, unit)
            return f"{formatted} {nml_unit}"
        return formatted

    # String-valued attribute via description or label
    if desc:
        return desc
    label = str(getattr(param, 'label', '') or '')
    if label:
        return label

    return str(default)


# ── Generic NeuroML tree walker ──────────────────────────────────────
#
# Role slots use "Pattern B": <key_name type="iri_type" attrs.../>
# All other component slots use "Pattern A": <iri_type id="key_name" attrs...>
#
# Standard NeuroML types (with IRI but no equations) render as plain XML
# elements.  Custom types (with derived_variables) generate a LEMS
# <ComponentType> definition and are referenced by type name.

_NML_ROLE_SLOTS = frozenset({
    'forwardRate', 'reverseRate',
    'steadyState', 'timeCourse',
    'q10Settings',
    'blockMechanism', 'plasticityMechanism',
})

# role slot → (extends_type, exposure_name, exposure_dimension)
_NML_ROLE_BASES = {
    'forwardRate':  ('baseVoltageDepRate', 'r', 'per_time'),
    'reverseRate':  ('baseVoltageDepRate', 'r', 'per_time'),
    'steadyState':  ('baseVoltageDepVariable', 'x', 'none'),
    'timeCourse':   ('baseVoltageDepTime', 't', 'time'),
}

_CHANNEL_TYPES = frozenset({
    'ionChannelHH', 'ionChannelKS', 'ionChannelPassive',
})

# Channel params that go on channelPopulation / channelDensity, not on the channel element
_CHANNEL_LINKING_PARAMS = frozenset({
    'number', 'erev', 'condDensity', 'ion', 'permeability',
})

_CONCENTRATION_MODEL_TYPES = frozenset({
    'fixedFactorConcentrationModel', 'decayingPoolConcentrationModel',
})

# Standard NeuroML synapse types — rendered as standalone XML components
# with parameter attributes (no ComponentType definition needed).
_SYNAPSE_TYPES = frozenset({
    'expOneSynapse', 'expTwoSynapse', 'expThreeSynapse', 'alphaSynapse',
    'alphaCurrentSynapse',
    'blockingPlasticSynapse', 'doubleSynapse',
    'gapJunction', 'linearGradedSynapse', 'gradedSynapse',
    'silentSynapse',
    'expCondSynapse', 'alphaCondSynapse',
    'expCurrSynapse', 'alphaCurrSynapse',
})

# Synapse types that use electrical projections (non-chemical)
_ELECTRICAL_SYNAPSE_TYPES = frozenset({
    'gapJunction',
})

# Synapse types that use continuous projections (graded)
_CONTINUOUS_SYNAPSE_TYPES = frozenset({
    'linearGradedSynapse', 'gradedSynapse', 'silentSynapse',
})


def _nml_type_name(dynamics):
    """Extract the NeuroML type name from a dynamics IRI (e.g. 'neuroml:ionChannelHH' → 'ionChannelHH')."""
    iri = getattr(dynamics, 'iri', None) or ''
    if ':' not in iri:
        return None
    prefix, name = iri.split(':', 1)
    return name if prefix == 'neuroml' else None


def _is_custom_nml_type(dynamics):
    """Return True if this dynamics needs a custom LEMS ComponentType definition."""
    dvs = getattr(dynamics, 'derived_variables', None) or {}
    svs = getattr(dynamics, 'state_variables', None) or {}
    return bool(dvs or svs)


# ── Unit → LEMS dimension mapping ────────────────────────────────────
_UNIT_TO_DIMENSION = {
    # time
    's': 'time', 'ms': 'time', 'us': 'time',
    # voltage
    'V': 'voltage', 'mV': 'voltage',
    # concentration
    'mol_per_m3': 'concentration', 'mol_per_cm3': 'concentration',
    'mmol_per_m3': 'concentration', 'mM': 'concentration',
    # conductance
    'S': 'conductance', 'mS': 'conductance', 'uS': 'conductance', 'pS': 'conductance',
    'S_per_cm2': 'conductance_density', 'mS_per_cm2': 'conductance_density',
    # temperature
    'degC': 'temperature',
    # current
    'A': 'current', 'mA': 'current', 'uA': 'current', 'nA': 'current', 'pA': 'current',
    # capacitance
    'F': 'capacitance', 'mF': 'capacitance', 'uF': 'capacitance', 'nF': 'capacitance', 'pF': 'capacitance',
    # per_time (rates)
    'per_s': 'per_time', 'per_ms': 'per_time', 'Hz': 'per_time',
    # resistance
    'ohm': 'resistance', 'kohm': 'resistance', 'Mohm': 'resistance',
    # specific capacitance
    'F_per_m2': 'specificCapacitance', 'uF_per_cm2': 'specificCapacitance',
}

# Well-known constant names → LEMS dimension (fallback when no unit)
_CONST_NAME_DIMENSION = {
    'TIME_SCALE': 'time',
    'VOLT_SCALE': 'voltage',
    'CONC_SCALE': 'concentration',
    'offset': 'voltage',
}

# Known requirement variables → LEMS dimension
_REQUIREMENT_DIMENSIONS = {
    'alpha': 'per_time',
    'beta': 'per_time',
    'caConc': 'concentration',
    'iCa': 'current',
    'temperature': 'temperature',
}

# LEMS built-in functions and keywords (not requirements)
_LEMS_BUILTINS = frozenset({
    'exp', 'ln', 'log', 'sin', 'cos', 'tan', 'sqrt', 'ceil', 'floor',
    'abs', 'random', 'H', 'Piecewise', 'True', 'False',
    'TIME_SCALE', 'VOLT_SCALE', 'CONC_SCALE',
})


def _parse_piecewise(rhs):
    """Parse ``Piecewise((val, cond), ...)`` → list of (value_str, condition_str|None).

    Returns None if *rhs* is not a Piecewise expression.
    The last entry may have condition ``None`` (default/fallback case).
    """
    rhs = rhs.strip()
    if not rhs.startswith('Piecewise(') or not rhs.endswith(')'):
        return None
    inner = rhs[10:-1].strip()

    # Walk character-by-character, extracting (value, condition) pairs
    pairs = []
    pos = 0
    while pos < len(inner):
        # skip commas/spaces between pairs
        while pos < len(inner) and inner[pos] in ' ,':
            pos += 1
        if pos >= len(inner) or inner[pos] != '(':
            break
        # find matching close-paren for this pair
        depth = 0
        start = pos + 1
        end = None
        for i in range(pos, len(inner)):
            if inner[i] == '(':
                depth += 1
            elif inner[i] == ')':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            break
        pair_str = inner[start:end].strip()
        # split at LAST top-level comma → (value, condition)
        last_comma = -1
        d = 0
        for j, ch in enumerate(pair_str):
            if ch == '(':
                d += 1
            elif ch == ')':
                d -= 1
            elif ch == ',' and d == 0:
                last_comma = j
        if last_comma > 0:
            val = pair_str[:last_comma].strip()
            cond = pair_str[last_comma + 1:].strip()
        else:
            val = pair_str
            cond = 'True'
        # translate Python comparison operators → LEMS
        if cond == 'True':
            pairs.append((val, None))
        else:
            cond = cond.replace('>=', ' .geq. ')
            cond = cond.replace('<=', ' .leq. ')
            cond = cond.replace('==', ' .eq. ')
            # remaining < > are standalone (>= already replaced)
            cond = re.sub(r'(?<!\.)>', ' .gt. ', cond)
            cond = re.sub(r'(?<!\.)<', ' .lt. ', cond)
            # clean up extra whitespace
            cond = ' '.join(cond.split())
            pairs.append((val, cond))
        pos = end + 1

    return pairs if pairs else None


def _detect_requirements(params, dvs):
    """Find variables referenced in expressions but not defined as Constants or DVs.

    Returns dict of {name: dimension} for each detected requirement.
    """
    defined = set(params.keys()) | set(dvs.keys()) | {'v', 't'}
    referenced = set()
    for dv in dvs.values():
        eq = getattr(dv, 'equation', None)
        rhs = getattr(eq, 'rhs', '') if eq else ''
        referenced |= set(re.findall(r'\b([a-zA-Z_]\w*)\b', rhs))
    reqs = referenced - defined - _LEMS_BUILTINS
    return {r: _REQUIREMENT_DIMENSIONS.get(r, 'none') for r in sorted(reqs)}


def _param_dimension(p_name, p_val):
    """Infer LEMS dimension for a Constant from its unit or name."""
    unit = str(getattr(p_val, 'unit', '') or '')
    if unit:
        return _UNIT_TO_DIMENSION.get(unit, 'none')
    return _CONST_NAME_DIMENSION.get(p_name, 'none')


def _render_custom_component_type(dynamics, role_slot=None):
    """Generate a LEMS ``<ComponentType>`` definition from a Dynamics with derived_variables.

    Parameters
    ----------
    dynamics : Dynamics
        Dynamics with derived_variables to render.
    role_slot : str or None
        The role slot this type fills (forwardRate, timeCourse, etc.).
        Determines the base type and exposure attributes.

    Unit handling: Constants honour the parameter's own ``unit``.
    TIME_SCALE / VOLT_SCALE are only auto-added when not present in params.
    """
    type_name = _nml_type_name(dynamics)
    if not type_name:
        return ''

    params = getattr(dynamics, 'parameters', None) or {}
    dvs = getattr(dynamics, 'derived_variables', None) or {}

    base_info = _NML_ROLE_BASES.get(role_slot)
    extends = base_info[0] if base_info else 'baseComponent'
    exposure_name = base_info[1] if base_info else None
    exposure_dim = base_info[2] if base_info else None

    # Detect requirements (undeclared variables in expressions)
    reqs = _detect_requirements(params, dvs)
    # Ca-dependent rate → switch base type
    if 'caConc' in reqs and extends == 'baseVoltageDepRate':
        extends = 'baseVoltageConcDepRate'

    lines = [f'    <ComponentType name="{type_name}" extends="{extends}">']

    # Auto-add scale constants only when not explicitly provided
    if 'TIME_SCALE' not in params:
        lines.append(
            '        <Constant name="TIME_SCALE" dimension="time" value="1 ms"/>')
    if 'VOLT_SCALE' not in params:
        lines.append(
            '        <Constant name="VOLT_SCALE" dimension="voltage" value="1 mV"/>')
    if 'caConc' in reqs and 'CONC_SCALE' not in params:
        lines.append(
            '        <Constant name="CONC_SCALE" dimension="concentration"'
            ' value="1 mol_per_m3"/>')

    # Render explicit parameters as Constants with proper dimension and unit
    for p_name, p_val in params.items():
        dim = _param_dimension(p_name, p_val)
        nml_val = _nml_attr(p_val)
        if not nml_val:
            continue
        lines.append(
            f'        <Constant name="{p_name}" dimension="{dim}"'
            f' value="{nml_val}"/>'
        )

    # Requirements
    for req_name, req_dim in reqs.items():
        lines.append(
            f'        <Requirement name="{req_name}" dimension="{req_dim}"/>')

    # Dynamics block
    lines.append('        <Dynamics>')
    for dv_name, dv in dvs.items():
        eq = getattr(dv, 'equation', None)
        rhs = getattr(eq, 'rhs', '') if eq else ''
        is_exposure = (exposure_name and dv_name == exposure_name)
        dim = exposure_dim if is_exposure else 'none'

        # Check for Piecewise → ConditionalDerivedVariable
        cases = _parse_piecewise(rhs)
        if cases is not None:
            exp_attr = f' exposure="{exposure_name}"' if is_exposure else ''
            lines.append(
                f'            <ConditionalDerivedVariable name="{dv_name}"'
                f'{exp_attr} dimension="{dim}">')
            for val, cond in cases:
                if cond is not None:
                    lines.append(
                        f'                <Case condition="{cond}"'
                        f' value="{val}"/>')
                else:
                    lines.append(f'                <Case value="{val}"/>')
            lines.append('            </ConditionalDerivedVariable>')
        else:
            if is_exposure:
                lines.append(
                    f'            <DerivedVariable name="{dv_name}" '
                    f'exposure="{exposure_name}" dimension="{dim}" '
                    f'value="{rhs}"/>')
            else:
                lines.append(
                    f'            <DerivedVariable name="{dv_name}" '
                    f'dimension="none" value="{rhs}"/>')
    lines.append('        </Dynamics>')
    lines.append('    </ComponentType>')

    return '\n'.join(lines)


def _render_nml_subtree(dynamics, key_name, indent=8, custom_types=None,
                        exclude_params=None):
    """Recursively render a Dynamics node as NeuroML XML.

    Two rendering patterns determined by the ``key_name``:

    - **Pattern A** (component): ``<iri_type id="key_name" attrs...>children</iri_type>``
    - **Pattern B** (role slot):  ``<key_name type="iri_type" attrs.../>``

    Pattern B is used when ``key_name`` is in :data:`_NML_ROLE_SLOTS`.

    If the Dynamics has ``derived_variables``, a LEMS ``<ComponentType>``
    definition is generated and collected in *custom_types*.

    Parameters
    ----------
    dynamics : Dynamics
        The dynamics node to render.
    key_name : str
        Name of this node (parent's modes/components dict key).
    indent : int
        Current indentation (spaces).
    custom_types : dict or None
        Collector for custom ComponentType definitions {type_name: xml_str}.
    exclude_params : set or None
        Parameter names to skip (e.g. channelPopulation attrs).
    """
    type_name = _nml_type_name(dynamics)
    if not type_name:
        return []

    pad = ' ' * indent
    params = getattr(dynamics, 'parameters', None) or {}
    children = getattr(dynamics, 'modes', None) or {}
    is_role = key_name in _NML_ROLE_SLOTS
    is_custom = _is_custom_nml_type(dynamics)

    # Collect custom ComponentType if needed
    if is_custom and custom_types is not None and type_name not in custom_types:
        custom_types[type_name] = _render_custom_component_type(
            dynamics, role_slot=key_name if is_role else None
        )

    # Build XML attributes — standard types use all params; custom types use none
    attrs = []
    if not is_custom:
        for p_name, p_val in params.items():
            if exclude_params and p_name in exclude_params:
                continue
            nml_val = _nml_attr(p_val)
            if nml_val:
                attrs.append(f'{p_name}="{nml_val}"')

    # Recurse into children
    child_lines = []
    for child_key, child_dyn in children.items():
        child_lines.extend(
            _render_nml_subtree(child_dyn, child_key, indent + 4, custom_types)
        )

    # Assemble element
    if is_role:
        # Pattern B: <key_name type="type_name" attrs.../>
        attr_parts = [f'type="{type_name}"'] + attrs
        attr_str = ' '.join(attr_parts)
        if child_lines:
            return [f'{pad}<{key_name} {attr_str}>'] + child_lines + [f'{pad}</{key_name}>']
        return [f'{pad}<{key_name} {attr_str}/>']
    else:
        # Pattern A: <type_name id="key_name" attrs...>
        attr_parts = [f'id="{safe_id(key_name)}"'] + attrs
        attr_str = ' '.join(attr_parts)
        if child_lines:
            return [f'{pad}<{type_name} {attr_str}>'] + child_lines + [f'{pad}</{type_name}>']
        return [f'{pad}<{type_name} {attr_str}/>']


def _render_cell_xml(dyn, dyn_id=None, custom_types=None):
    """Render a single NeuroML cell (channels + cell definition) from a Dynamics.

    Returns a dict with keys:
    - ``channel_xmls``: list of channel XML strings
    - ``cell_xml``: the cell definition XML string
    - ``input_xmls``: list of standalone input component XML strings
    - ``input_refs``: list of explicitInput XML strings (for single-cell)
    - ``conc_xmls``: list of concentration model XML strings
    - ``custom_types``: updated custom_types dict

    Returns None if the Dynamics cannot be rendered as a standard NeuroML cell.
    """
    if custom_types is None:
        custom_types = {}

    iri = getattr(dyn, 'iri', None) or ''
    if not iri.startswith('neuroml:'):
        return None
    cell_type = iri.split(':', 1)[1]

    # Flat cell types: render as simple <type id="..." params.../> with no children
    _FLAT_CELL_TYPES = {
        'iafCell', 'iafRefCell', 'iafTauCell', 'iafTauRefCell',
        'izhikevichCell', 'izhikevich2007Cell',
        'adExIaFCell', 'fitzHughNagumoCell', 'fitzHughNagumo1969Cell',
        'IF_curr_alpha', 'IF_curr_exp', 'IF_cond_alpha', 'IF_cond_exp',
        'EIF_cond_exp_isfa_ista', 'EIF_cond_alpha_isfa_ista',
        'HH_cond_exp',
    }
    if cell_type in _FLAT_CELL_TYPES:
        if dyn_id is None:
            dyn_id = safe_id(dyn.name or 'dynamics')
        params = dyn.parameters or {}
        attr_parts = [f'id="{dyn_id}"']
        for pn, pv in params.items():
            pn = str(pn)
            attr_parts.append(f'{pn}="{_nml_attr(pv)}"')
        cell_xml = f'    <{cell_type} {" ".join(attr_parts)}/>'
        return {
            'channel_xmls': [],
            'cell_xml': cell_xml,
            'input_xmls': [],
            'input_refs': [],
            'conc_xmls': [],
            'custom_types': custom_types,
            'cell_type': cell_type,
            'dyn_id': dyn_id,
        }

    if cell_type not in ('pointCellCondBased', 'cell'):
        return None

    if dyn_id is None:
        dyn_id = safe_id(dyn.name or 'dynamics')

    params = dyn.parameters or {}
    components = dyn.modes or {}

    # ── Classify components by IRI type ──
    channels = {}
    inputs = {}
    conc_models = {}
    segments = {}
    segment_groups = {}
    for comp_name, comp in components.items():
        comp_type = _nml_type_name(comp) or ''
        if comp_type in _CHANNEL_TYPES:
            channels[comp_name] = comp
        elif comp_type in CURRENT_INPUT_TYPES:
            inputs[comp_name] = comp
        elif comp_type in _CONCENTRATION_MODEL_TYPES:
            conc_models[comp_name] = comp
        elif comp_type == 'segment':
            segments[comp_name] = comp
        elif comp_type == 'segmentGroup':
            segment_groups[comp_name] = comp

    # ── Render channels with generic tree walker ──
    channel_xmls = []
    channel_pops = []
    channel_densities = []

    for comp_name, comp in channels.items():
        comp_params = comp.parameters or {}
        ch_lines = _render_nml_subtree(
            comp, comp_name, indent=4, custom_types=custom_types,
            exclude_params=_CHANNEL_LINKING_PARAMS,
        )
        channel_xmls.append('\n'.join(ch_lines))

        if cell_type == 'pointCellCondBased':
            number_attr = _nml_attr(comp_params.get('number'), '1')
            erev_attr = _nml_attr(comp_params.get('erev'), '0mV')
            channel_pops.append(
                f'        <channelPopulation id="{safe_id(comp_name)}_pop" '
                f'ionChannel="{safe_id(comp_name)}" '
                f'number="{number_attr}" erev="{erev_attr}"/>'
            )
        elif cell_type == 'cell':
            erev_attr = _nml_attr(comp_params.get('erev'), '0mV')
            ion_attr = _nml_attr(comp_params.get('ion'), 'non_specific')
            ghk_perm = comp_params.get('permeability')
            if ghk_perm:
                channel_densities.append(
                    f'                <channelDensityGHK permeability="{_nml_attr(ghk_perm)}" '
                    f'id="{safe_id(comp_name)}_all" '
                    f'ionChannel="{safe_id(comp_name)}" ion="{ion_attr}"/>'
                )
            else:
                cd_attr = _nml_attr(comp_params.get('condDensity'), '0.0003 S_per_cm2')
                channel_densities.append(
                    f'                <channelDensity condDensity="{cd_attr}" '
                    f'id="{safe_id(comp_name)}_all" '
                    f'ionChannel="{safe_id(comp_name)}" erev="{erev_attr}" ion="{ion_attr}"/>'
                )

    # ── Render input components ──
    input_xmls = []
    input_refs = []
    for inp_name, inp in inputs.items():
        inp_lines = _render_nml_subtree(
            inp, inp_name, indent=4, custom_types=custom_types,
        )
        input_xmls.append('\n'.join(inp_lines))
        input_refs.append(
            f'        <explicitInput target="pop[0]" '
            f'input="{safe_id(inp_name)}" destination="synapses"/>'
        )

    # Legacy: pulse params on the cell (backward compat)
    if not input_xmls:
        pulse_idx = 0
        for suffix in ('', '_2', '_3', '_4'):
            d_key = f'pulse_delay{suffix}'
            dur_key = f'pulse_duration{suffix}'
            amp_key = f'I_amp{suffix}'
            if params.get(d_key) and params.get(dur_key) and params.get(amp_key):
                pulse_idx += 1
                pid = f'pulseGen{pulse_idx}'
                input_xmls.append(
                    f'    <pulseGenerator id="{pid}" '
                    f'delay="{_nml_attr(params[d_key])}" '
                    f'duration="{_nml_attr(params[dur_key])}" '
                    f'amplitude="{_nml_attr(params[amp_key])}"/>'
                )
                input_refs.append(
                    f'        <explicitInput target="pop[0]" '
                    f'input="{pid}" destination="synapses"/>'
                )

    # ── Render concentration models ──
    conc_xmls = []
    species_xmls = []
    for cm_name, cm in conc_models.items():
        cm_type = _nml_type_name(cm)
        cm_params = cm.parameters or {}
        attrs = [f'id="{safe_id(cm_name)}"', f'type="{cm_type}"']
        for p_name, p_val in cm_params.items():
            if p_name in ('initialConcentration', 'initialExtConcentration'):
                continue
            nml_val = _nml_attr(p_val)
            if nml_val:
                attrs.append(f'{p_name}="{nml_val}"')
        conc_xmls.append(f'    <concentrationModel {" ".join(attrs)}/>')

        ion_attr = _nml_attr(cm_params.get('ion'), 'ca')
        init_conc = _nml_attr(cm_params.get('initialConcentration'), '5e-6 mM')
        init_ext = _nml_attr(cm_params.get('initialExtConcentration'), '2 mM')
        species_xmls.append(
            f'                <species id="{ion_attr}" ion="{ion_attr}" '
            f'concentrationModel="{safe_id(cm_name)}" '
            f'initialConcentration="{init_conc}" '
            f'initialExtConcentration="{init_ext}"/>'
        )

    # ── Build cell definition XML ──
    cell_lines = []
    segment_id_list = []

    if cell_type == 'pointCellCondBased':
        C_attr = _nml_attr(params.get('C'), '10pF')
        v0_attr = _nml_attr(params.get('v0'), '-65mV')
        thresh_attr = _nml_attr(params.get('thresh'), '20mV')
        cell_lines.append(
            f'    <pointCellCondBased id="{dyn_id}" C="{C_attr}" '
            f'v0="{v0_attr}" thresh="{thresh_attr}">'
        )
        for pop in channel_pops:
            cell_lines.append(pop)
        cell_lines.append('    </pointCellCondBased>')

    elif cell_type == 'cell':
        spec_cap = _nml_attr(params.get('specificCapacitance'), '1.0 uF_per_cm2')
        init_v = _nml_attr(
            params.get('initMembPotential') or params.get('v0'), '-65.0 mV')
        spike_thresh = _nml_attr(params.get('spikeThresh'), '0 mV')
        resistivity_val = _nml_attr(params.get('resistivity'), '0.1 kohm_cm')

        cell_lines.append(f'    <cell id="{dyn_id}">')
        cell_lines.append(f'        <morphology id="{dyn_id}_morphology">')

        if segments:
            # Multi-segment morphology from neuroml:segment modes
            seg_list = sorted(
                segments.items(),
                key=lambda x: int(getattr(
                    (x[1].parameters or {}).get('id'), 'value', 0)))
            for seg_name, seg in seg_list:
                sp = seg.parameters or {}
                seg_id = int(getattr(sp.get('id'), 'value', 0))
                segment_id_list.append(seg_id)
                parent_param = sp.get('parent')
                parent_id = (int(getattr(parent_param, 'value', -1))
                             if parent_param else -1)
                cell_lines.append(
                    f'            <segment id ="{seg_id}" name="{seg_name}">')
                if parent_id >= 0:
                    cell_lines.append(
                        f'                <parent segment="{parent_id}"/>')
                px = sp.get('proximal_x')
                if px is not None:
                    cell_lines.append(
                        f'                <proximal '
                        f'x="{getattr(px, "value", 0)}" '
                        f'y="{getattr(sp.get("proximal_y"), "value", 0)}" '
                        f'z="{getattr(sp.get("proximal_z"), "value", 0)}" '
                        f'diameter='
                        f'"{getattr(sp.get("proximal_diameter"), "value", 1)}'
                        f'"/>')
                dx = sp.get('distal_x')
                if dx is not None:
                    cell_lines.append(
                        f'                <distal '
                        f'x="{getattr(dx, "value", 0)}" '
                        f'y="{getattr(sp.get("distal_y"), "value", 0)}" '
                        f'z="{getattr(sp.get("distal_z"), "value", 0)}" '
                        f'diameter='
                        f'"{getattr(sp.get("distal_diameter"), "value", 1)}'
                        f'"/>')
                cell_lines.append('            </segment>')

            # Segment groups from neuroml:segmentGroup modes
            for grp_name, grp in segment_groups.items():
                gp = grp.parameters or {}
                nlid_param = gp.get('neuroLexId')
                nlid = (getattr(nlid_param, 'label', None)
                        if nlid_param else None)
                nlid_attr = f' neuroLexId="{nlid}"' if nlid else ''
                cell_lines.append(
                    f'            <segmentGroup id="{grp_name}"{nlid_attr}>')
                for pn, pv in gp.items():
                    if pn in ('neuroLexId', 'members', 'includes'):
                        continue
                    val = getattr(pv, 'value', pv)
                    if val is not None:
                        cell_lines.append(
                            f'                <property tag="{pn}" '
                            f'value="{int(val)}"/>')
                members_param = gp.get('members')
                if members_param:
                    members_str = getattr(members_param, 'shape', '') or ''
                    for mid in str(members_str).split(','):
                        mid = mid.strip()
                        if mid:
                            cell_lines.append(
                                f'                <member segment="{mid}"/>')
                includes_param = gp.get('includes')
                if includes_param:
                    includes_str = (
                        getattr(includes_param, 'shape', '') or '')
                    for inc in str(includes_str).split(','):
                        inc = inc.strip()
                        if inc:
                            cell_lines.append(
                                f'                <include '
                                f'segmentGroup="{inc}"/>')
                cell_lines.append('            </segmentGroup>')
        else:
            # Single-segment morphology (backward compatible)
            diameter = float(_nml_attr(params.get('diameter'), '10'))
            length = float(_nml_attr(params.get('length'), '0'))
            segment_id_list = [0]
            cell_lines.append('            <segment id="0" name="Soma">')
            cell_lines.append(
                f'                <proximal x="0.0" y="0.0" z="0.0" '
                f'diameter="{diameter}"/>')
            cell_lines.append(
                f'                <distal x="0.0" y="0.0" z="{length}" '
                f'diameter="{diameter}"/>')
            cell_lines.append('            </segment>')
            cell_lines.append('            <segmentGroup id="all">')
            cell_lines.append('                <member segment="0"/>')
            cell_lines.append('            </segmentGroup>')
            cell_lines.append('            <segmentGroup id="soma_group">')
            cell_lines.append('                <member segment="0"/>')
            cell_lines.append('            </segmentGroup>')

        cell_lines.append('        </morphology>')
        cell_lines.append('        <biophysicalProperties id="biophys">')
        cell_lines.append('            <membraneProperties>')
        for cd in channel_densities:
            cell_lines.append(cd)
        cell_lines.append(
            f'                <specificCapacitance value="{spec_cap}"/>')
        cell_lines.append(
            f'                <initMembPotential value="{init_v}"/>')
        cell_lines.append(
            f'                <spikeThresh value="{spike_thresh}"/>')
        cell_lines.append('            </membraneProperties>')
        cell_lines.append('            <intracellularProperties>')
        for sp in species_xmls:
            cell_lines.append(sp)
        cell_lines.append(
            f'                <resistivity value="{resistivity_val}"/>')
        cell_lines.append('            </intracellularProperties>')
        cell_lines.append('        </biophysicalProperties>')
        cell_lines.append('    </cell>')

    return {
        'channel_xmls': channel_xmls,
        'cell_xml': '\n'.join(cell_lines),
        'input_xmls': input_xmls,
        'input_refs': input_refs,
        'conc_xmls': conc_xmls,
        'custom_types': custom_types,
        'cell_type': cell_type,
        'dyn_id': dyn_id,
        'segment_ids': segment_id_list,
    }


# ── Standard NeuroML input type detection ────────────────────────────

# Current injection sources: standalone <Component>, injected via <explicitInput>
# poissonFiringSynapse / transientPoissonFiringSynapse are combined spike+synapse
# components applied via <explicitInput destination="synapses">.
CURRENT_INPUT_TYPES = frozenset({
    'pulseGenerator', 'pulseGeneratorDL',
    'compoundPulseGenerator', 'compoundInput',
    'sineGenerator', 'sineGeneratorDL',
    'rampGenerator', 'rampGeneratorDL',
    'voltageClamp', 'voltageClampTriple',
    'poissonFiringSynapse', 'transientPoissonFiringSynapse',
    'timedSynapticInput',
})

# Event (spike) sources: become <population> with synapticConnection
EVENT_SOURCE_TYPES = frozenset({
    'spikeGenerator', 'spikeGeneratorRandom', 'spikeGeneratorRefPoisson',
    'spikeGeneratorPoisson', 'spikeArray',
    'SpikeSourcePoisson',
})

ALL_INPUT_TYPES = CURRENT_INPUT_TYPES | EVENT_SOURCE_TYPES


def _render_compound_input_children(dyn_obj, indent=8):
    """Render child input components of a compoundInput from Dynamics.components.

    Each child component (pulseGenerator, sineGenerator, etc.) is rendered as
    a self-closing XML element with its parameters as attributes.
    """
    components = getattr(dyn_obj, 'components', None) or getattr(dyn_obj, 'modes', None) or {}
    if not components:
        return ''
    pad = ' ' * indent
    lines = []
    for comp_name, comp_obj in components.items():
        comp_iri = getattr(comp_obj, 'iri', '') or ''
        comp_type = comp_iri.split(':', 1)[1] if comp_iri.startswith('neuroml:') else str(comp_name)
        comp_id = safe_id(str(comp_name))
        attr_parts = [f'id="{comp_id}"']
        params = getattr(comp_obj, 'parameters', None) or {}
        for pn, pv in params.items():
            attr_parts.append(f'{pn}="{_nml_attr(pv)}"')
        lines.append(f'{pad}<{comp_type} {" ".join(attr_parts)}/>')
    return '\n'.join(lines)


def _render_event_children(dyn_obj, time_scale='ms', indent=8):
    """Render preset_time events from a Dynamics as NeuroML child XML elements.

    For ``spikeArray``, each trigger time becomes ``<spike id="N" time="T unit"/>``.
    Returns the child XML string (multiple lines), or empty string if none.
    """
    if dyn_obj is None:
        return ''
    events = getattr(dyn_obj, 'events', None) or {}
    if not events:
        return ''
    pad = ' ' * indent
    nml_unit = _TVBO_TO_NML_UNIT.get(str(time_scale), str(time_scale))
    children = []
    spike_idx = 0
    for ev in events.values():
        if str(getattr(ev, 'event_type', '')) != 'preset_time':
            continue
        times = getattr(ev, 'trigger_times', None) or []
        for t in times:
            children.append(f'{pad}<spike id="{spike_idx}" time="{t} {nml_unit}"/>')
            spike_idx += 1
    return '\n'.join(children)


# ── Hierarchical custom LEMS context builder ─────────────────────────
#
# When a dynamics tree uses ``iri: extends:base*`` on its components,
# the user defines all equations explicitly (no standard NeuroML
# biological types).  The adapter generates custom ComponentTypes that
# extend the LEMS base types for type-system compatibility.
#
# The lookup tables below encode the LEMS *type-system infrastructure*
# (what Child/Children/Attachments slots each base type declares, which
# parameters are inherited, etc.).  This is NOT biological knowledge —
# it's the equivalent of knowing that an abstract base class has certain
# methods.

_BASE_TYPE_META = {
    'baseCellMembPot': {
        'inherited_params': set(),
        'exposures': {'v': 'voltage'},
        'requirements': {},
        'channel_wrapper': True,
        'attachments': [('synapses', 'basePointCurrent')],
        'on_start_refs': {'v': 'v0'},
    },
    'baseIonChannel': {
        'inherited_params': {'conductance'},
        'exposures': {'g': 'conductance'},
        'requirements': {},
        'children': ('gates', 'baseGate'),
    },
    'baseGate': {
        'inherited_params': {'instances'},
        'exposures': {'q': 'none', 'fcond': 'none'},
        'requirements': {},
        'child_roles': {
            'forwardRate': 'baseVoltageDepRate',
            'reverseRate': 'baseVoltageDepRate',
        },
        'on_start_refs': {'q': 'inf'},
    },
    'baseVoltageDepRate': {
        'inherited_params': set(),
        'exposures': {'r': 'per_time'},
        'requirements': {'v': 'voltage'},
    },
}


def _hier_format_attr_value(param):
    """Format a parameter as a compact LEMS attribute value (``10pS``)."""
    val = getattr(param, 'value', None)
    unit = str(getattr(param, 'unit', '') or '')
    if val is None:
        return str(getattr(param, 'description', '') or '')
    if isinstance(val, float) and val == int(val) and abs(val) < 1e15:
        formatted = str(int(val))
    else:
        formatted = str(val)
    if unit:
        nml_unit = _TVBO_TO_NML_UNIT.get(unit, unit)
        return f"{formatted}{nml_unit}"
    return formatted


def _hier_attrs_string(params, skip=None):
    """Build an XML attributes string from a dict of Parameters.

    Parameters in *skip* (e.g. inherited from base type) are excluded.
    """
    skip = skip or set()
    parts = []
    for k, v in params.items():
        if k in skip:
            continue
        parts.append(f'{k}="{_hier_format_attr_value(v)}"')
    return ' '.join(parts)


def _hier_parse_select_label(label_str):
    """Parse a DerivedVariable label like ``select:populations[*]/i reduce:add``.

    Returns a dict with keys ``select``, ``reduce``, ``required_false``
    or None if the label doesn't encode select metadata.
    """
    if not label_str or 'select:' not in label_str:
        return None
    result = {'select': None, 'reduce': None, 'required_false': False}
    for token in label_str.split():
        if token.startswith('select:'):
            result['select'] = token[7:]
        elif token.startswith('reduce:'):
            result['reduce'] = token[7:]
        elif token == 'required:false':
            result['required_false'] = True
    return result if result['select'] else None


def _hier_build_dynamics(dyn, extends, all_params):
    """Build the ``dynamics`` sub-dict for a custom ComponentType.

    Reads state_variables, derived_variables, and events from the
    Dynamics object and converts them to LEMS template data.
    """
    from collections import OrderedDict
    meta = _BASE_TYPE_META.get(extends, {})
    exposures = meta.get('exposures', {})

    # Collect all symbol names for expression conversion
    param_names = list((dyn.parameters or {}).keys())
    sv_names = list((dyn.state_variables or {}).keys())
    dv_names = list((dyn.derived_variables or {}).keys())
    all_names = param_names + sv_names + dv_names

    # Add any requirement names (e.g. v for rates)
    for req_name in meta.get('requirements', {}):
        if req_name not in all_names:
            all_names.append(req_name)

    inherited = meta.get('inherited_params', set())

    dynamics = {
        'derived_variables': [],
        'cdvs': [],
        'state_variables': [],
        'time_derivatives': [],
        'on_start': [],
        'on_condition': [],
    }

    # ── Derived variables ──
    for dv_key, dv in (dyn.derived_variables or {}).items():
        dim = _UNIT_TO_DIMENSION.get(str(getattr(dv, 'unit', '') or ''), 'none')
        exposure = dv_key if dv_key in exposures else None
        if exposure:
            dim = exposures[exposure]

        # Check for select/reduce via label
        sel = _hier_parse_select_label(getattr(dv, 'label', None))
        if sel:
            dynamics['derived_variables'].append({
                'name': dv_key,
                'dimension': dim,
                'exposure': exposure,
                'select': sel['select'],
                'reduce': sel['reduce'],
                'required_false': sel.get('required_false', False),
            })
            continue

        # Check for conditional derived variable
        if getattr(dv, 'conditional', False) and getattr(dv, 'cases', None):
            cases = []
            for case in dv.cases:
                cond_str = case.condition if case.condition else None
                val_str = case.equation.rhs if case.equation else ''
                # Default case (True or None) → no condition in LEMS
                if cond_str and cond_str.strip() == 'True':
                    cond_str = None
                # Convert Python conditions to LEMS
                elif cond_str:
                    cond_str = _python_cond_to_lems(cond_str, all_names)
                val_str = sympy_to_lems(val_str, parameters=all_names)
                cases.append({
                    'condition': cond_str,
                    'value': val_str,
                })
            dynamics['cdvs'].append({
                'name': dv_key,
                'dimension': dim,
                'exposure': exposure or dv_key,
                'cases': cases,
            })
            continue

        # Regular derived variable
        rhs = dv.equation.rhs if dv.equation else '0'
        value = sympy_to_lems(rhs, parameters=all_names)
        dynamics['derived_variables'].append({
            'name': dv_key,
            'dimension': dim,
            'exposure': exposure,
            'value': value,
        })

    # ── State variables & time derivatives ──
    for sv_key, sv in (dyn.state_variables or {}).items():
        dim = _UNIT_TO_DIMENSION.get(str(getattr(sv, 'unit', '') or ''), 'none')
        exposure = sv_key if sv_key in exposures else None
        if exposure:
            dim = exposures[exposure]

        dynamics['state_variables'].append({
            'name': sv_key,
            'dimension': dim,
            'exposure': exposure or sv_key,
        })

        # Time derivative
        rhs = sv.equation.rhs if sv.equation else '0'
        td_value = sympy_to_lems(rhs, parameters=all_names)
        dynamics['time_derivatives'].append({
            'variable': sv_key,
            'value': td_value,
        })

        # OnStart: use named reference from base type meta, fallback to literal
        on_start_refs = meta.get('on_start_refs', {})
        iv = getattr(sv, 'initial_value', None)
        if iv is not None:
            on_start_value = on_start_refs.get(sv_key, str(iv))
            dynamics['on_start'].append({
                'variable': sv_key,
                'value': on_start_value,
            })

    # ── Events → OnCondition ──
    for ev_key, ev in (getattr(dyn, 'events', None) or {}).items():
        cond = getattr(getattr(ev, 'condition', None), 'rhs', None)
        if cond:
            cond_lems = _python_cond_to_lems(str(cond), all_names)
            dynamics['on_condition'].append({
                'test': cond_lems,
                'port': ev_key,
            })

    return dynamics


def _python_cond_to_lems(cond_str, all_names=None):
    """Convert a Python-style condition to LEMS syntax.

    ``x != 0`` → ``x .neq. 0``
    ``v > thresh`` → ``v .gt. thresh``
    ``v >= thresh`` → ``v .geq. thresh``
    """
    s = str(cond_str)
    s = s.replace('!=', ' .neq. ')
    s = s.replace('>=', ' .geq. ')
    s = s.replace('<=', ' .leq. ')
    # Must check > and < AFTER >= and <=
    s = s.replace('>', ' .gt. ')
    s = s.replace('<', ' .lt. ')
    # Clean up multiple spaces
    s = ' '.join(s.split())
    return s


def _build_hier_custom_context(experiment):
    """Build context dict for hierarchical custom LEMS ComponentTypes.

    Used when the root dynamics has ``iri: extends:base*``, meaning the
    user defines all equations explicitly while extending LEMS base types
    for type-system compatibility.

    The context is consumed by ``tvbo-neuroml-hier-custom-lems.xml.mako``.
    """
    from collections import OrderedDict

    dyn = experiment.dynamics
    integration = getattr(experiment, 'integration', None)

    # ── Integration settings ──
    dt = integration.step_size if integration else 0.01
    duration = integration.duration if integration else 1000.0
    raw_ts = str(getattr(integration, 'time_scale', None) or 'ms')
    time_unit = raw_ts if raw_ts in ('s', 'ms', 'us') else 'ms'

    label = getattr(experiment, 'label', None)
    dyn_id = safe_id(dyn.name or 'dynamics')
    sim_id = 'sim_' + (safe_id(label) if label else dyn_id)

    # ── Collect ComponentType definitions (leaf-first) ──
    type_defs = OrderedDict()
    type_order = []

    root_extends = dyn.iri.split(':', 1)[1]
    root_type_name = getattr(dyn, 'label', None) or f'custom{root_extends}'
    root_meta = _BASE_TYPE_META.get(root_extends, {})

    channels = []
    channel_pops = []

    for ch_key, ch_dyn in (dyn.modes or {}).items():
        ch_iri = getattr(ch_dyn, 'iri', '') or ''
        if not ch_iri.startswith('extends:'):
            continue
        ch_extends = ch_iri.split(':', 1)[1]
        ch_meta = _BASE_TYPE_META.get(ch_extends, {})
        ch_type_name = getattr(ch_dyn, 'label', None) or f'custom{ch_extends}'
        ch_params = ch_dyn.parameters or {}

        # channelPopulation params: number, erev (belong to the wrapper)
        number = _hier_format_attr_value(ch_params['number']) if 'number' in ch_params else '1'
        erev = _hier_format_attr_value(ch_params['erev']) if 'erev' in ch_params else '0mV'

        # Channel-level instance attrs (exclude pop params)
        ch_inst_params = {k: v for k, v in ch_params.items()
                         if k not in ('number', 'erev')}
        ch_inherited = ch_meta.get('inherited_params', set())
        ch_attrs_str = _hier_attrs_string(ch_inst_params)

        # ── Process gates (Children of channel) ──
        gate_instances = []
        for g_key, g_dyn in (ch_dyn.modes or {}).items():
            g_iri = getattr(g_dyn, 'iri', '') or ''
            if not g_iri.startswith('extends:'):
                continue
            g_extends = g_iri.split(':', 1)[1]
            g_meta = _BASE_TYPE_META.get(g_extends, {})
            g_type_name = getattr(g_dyn, 'label', None) or f'custom{g_extends}'
            g_params = g_dyn.parameters or {}
            g_inherited = g_meta.get('inherited_params', set())
            g_attrs_str = _hier_attrs_string(g_params, skip=set())

            # ── Process rates (Child of gate) ──
            rate_children = OrderedDict()
            for r_key, r_dyn in (g_dyn.modes or {}).items():
                r_iri = getattr(r_dyn, 'iri', '') or ''
                if not r_iri.startswith('extends:'):
                    continue
                r_extends = r_iri.split(':', 1)[1]
                r_meta = _BASE_TYPE_META.get(r_extends, {})
                r_type_name = getattr(r_dyn, 'label', None) or f'custom{r_extends}'
                r_params = r_dyn.parameters or {}
                r_inherited = r_meta.get('inherited_params', set())
                r_attrs_str = _hier_attrs_string(r_params, skip=r_inherited)

                # Add rate ComponentType definition
                if r_type_name not in type_defs:
                    r_ct_params = [
                        {'name': k, 'dimension': _param_dimension(k, v)}
                        for k, v in r_params.items()
                        if k not in r_inherited
                    ]
                    type_defs[r_type_name] = {
                        'name': r_type_name,
                        'extends': r_extends,
                        'parameters': r_ct_params,
                        'child_slots': [],
                        'children_slots': [],
                        'attachments': [],
                        'dynamics': _hier_build_dynamics(r_dyn, r_extends, r_params),
                    }
                    type_order.append(r_type_name)

                rate_children[r_key] = {
                    'type_name': r_type_name,
                    'attrs_str': r_attrs_str,
                }

            # Add gate ComponentType definition
            if g_type_name not in type_defs:
                g_ct_params = [
                    {'name': k, 'dimension': _param_dimension(k, v)}
                    for k, v in g_params.items()
                    if k not in g_inherited
                ]
                # Child slots from base type meta
                child_roles = g_meta.get('child_roles', {})
                child_slots = [(role, base_t) for role, base_t in child_roles.items()]
                type_defs[g_type_name] = {
                    'name': g_type_name,
                    'extends': g_extends,
                    'parameters': g_ct_params,
                    'child_slots': child_slots,
                    'children_slots': [],
                    'attachments': [],
                    'dynamics': _hier_build_dynamics(g_dyn, g_extends, g_params),
                }
                type_order.append(g_type_name)

            gate_instances.append({
                'type_name': g_type_name,
                'id': g_key,
                'attrs_str': g_attrs_str,
                'role_children': rate_children,
            })

        # Add channel ComponentType definition
        if ch_type_name not in type_defs:
            ch_ct_params = [
                {'name': k, 'dimension': _param_dimension(k, v)}
                for k, v in ch_params.items()
                if k not in ch_inherited and k not in ('number', 'erev')
            ]
            # Children slot from base type meta
            children_info = ch_meta.get('children', None)
            ch_children_slots = [(children_info[0], children_info[1])] if children_info else []
            type_defs[ch_type_name] = {
                'name': ch_type_name,
                'extends': ch_extends,
                'parameters': ch_ct_params,
                'child_slots': [],
                'children_slots': ch_children_slots,
                'attachments': [],
                'dynamics': _hier_build_dynamics(ch_dyn, ch_extends, ch_params),
            }
            type_order.append(ch_type_name)

        channels.append({
            'type_name': ch_type_name,
            'id': ch_key,
            'attrs_str': ch_attrs_str,
            'children': gate_instances,
        })

        # channelPopulation entry
        pop_id = ch_key if ch_key in ('passive', 'leak') else f'{ch_key}Chans'
        channel_pops.append({
            'id': pop_id,
            'ion_channel': ch_key,
            'number': number,
            'erev': erev,
        })

    # ── Root cell ComponentType definition ──
    cell_params = {k: v for k, v in (dyn.parameters or {}).items()
                   if k not in ('pulse_delay', 'pulse_duration', 'I_amp')}
    cell_inherited = root_meta.get('inherited_params', set())
    cell_ct_params = [
        {'name': k, 'dimension': _param_dimension(k, v)}
        for k, v in cell_params.items()
        if k not in cell_inherited
    ]
    # Children: channelPopulations via wrapper
    cell_children_slots = []
    if root_meta.get('channel_wrapper'):
        cell_children_slots.append(('populations', 'baseChannelPopulation'))
    cell_attachments = root_meta.get('attachments', [])

    type_defs[root_type_name] = {
        'name': root_type_name,
        'extends': root_extends,
        'parameters': cell_ct_params,
        'child_slots': [],
        'children_slots': cell_children_slots,
        'attachments': cell_attachments,
        'dynamics': _hier_build_dynamics(dyn, root_extends, cell_params),
    }
    type_order.append(root_type_name)

    # ── Cell instance attributes ──
    cell_attrs_str = _hier_attrs_string(cell_params, skip=cell_inherited)

    # ── Input generators ──
    inputs = []
    ip = dyn.parameters or {}
    if 'pulse_delay' in ip and 'pulse_duration' in ip and 'I_amp' in ip:
        inputs.append({
            'type': 'pulseGenerator',
            'id': 'pulseGen1',
            'delay': _hier_format_attr_value(ip['pulse_delay']),
            'duration': _hier_format_attr_value(ip['pulse_duration']),
            'amplitude': _hier_format_attr_value(ip['I_amp']),
        })

    # ── Output variable ──
    output_var = None
    for sv_key, sv in (dyn.state_variables or {}).items():
        if getattr(sv, 'variable_of_interest', False):
            output_var = sv_key
            break
    if output_var is None:
        output_var = next(iter(dyn.state_variables or {}), 'v')

    # ── Population ID ──
    pop_id = f'{dyn_id}pop'

    return {
        'component_types': [type_defs[tn] for tn in type_order],
        'channels': channels,
        'cell_type_name': root_type_name,
        'cell_id': dyn_id,
        'cell_attrs_str': cell_attrs_str,
        'channel_pops': channel_pops,
        'inputs': inputs,
        'network_id': 'net1',
        'population_id': pop_id,
        'sim_id': sim_id,
        'sim_length': f'{duration}{time_unit}',
        'sim_step': f'{dt}{time_unit}',
        'output_var': output_var,
        'dyn_id': dyn_id,
        'is_network': False,
        'is_hier_custom': True,
    }


# ── Standard-types context builder (for Mako templates) ──────────────

def build_std_lems_context(experiment):
    """Build a context dict for standard NeuroML-type Mako templates.

    Inspects the experiment to determine whether it uses standard NeuroML
    types (``iri: neuroml:*``).  When it does, extracts *all* data that
    the Mako templates need — integration parameters, pre-rendered XML
    fragments for cells/channels/synapses, population lists, connection
    lists, and simulation metadata.

    Returns
    -------
    dict or None
        ``None`` when the experiment cannot be rendered using standard
        NeuroML types.  Otherwise a dict containing:

        * ``'is_network'``  – True for multi-population network template
        * ``'is_fhn'``      – True for FitzHugh-Nagumo cell template
        * All scalar and list variables that the templates iterate over.
    """
    from collections import OrderedDict

    # ── Multi-population network detection ──
    network = getattr(experiment, 'network', None)
    if network:
        nodes = getattr(network, 'nodes', None) or []
        edges = getattr(network, 'edges', None) or []
        dynamics_lib = getattr(network, 'dynamics', None) or {}
        if nodes and edges and dynamics_lib:
            has_nml_cells = any(
                _uses_neuroml_types(d)
                for d in dynamics_lib.values()
                if hasattr(d, 'iri')
            )
            if has_nml_cells:
                return _build_std_network_context(experiment)

    # ── Single-cell path ──
    dyn = experiment.dynamics
    if not dyn:
        return None

    iri = getattr(dyn, 'iri', None) or ''

    # Hierarchical custom types: iri starts with "extends:"
    if iri.startswith('extends:'):
        return _build_hier_custom_context(experiment)

    if not iri.startswith('neuroml:'):
        return None

    cell_type = iri.split(':', 1)[1]

    # Common integration parameters
    integration = getattr(experiment, 'integration', None)
    label = getattr(experiment, 'label', None)

    if cell_type in ('fitzHughNagumoCell', 'fitzHughNagumo1969Cell'):
        return _build_std_fhn_context(experiment, cell_type)

    return _build_std_cell_context(experiment)


def _build_std_fhn_context(experiment, cell_type):
    """Build context for FitzHugh-Nagumo standard cell template."""
    dyn = experiment.dynamics
    params = dyn.parameters or {}
    svs = dyn.state_variables or {}

    integration = getattr(experiment, 'integration', None)
    dt = integration.step_size if integration else 0.01
    duration = integration.duration if integration else 200.0
    raw_ts = (getattr(integration, 'time_scale', None) or 's') if integration else 's'
    time_scale = str(raw_ts) if str(raw_ts) in ('s', 'ms', 'us') else 's'

    dyn_id = safe_id(dyn.name or 'fhn')
    label = getattr(experiment, 'label', None)
    sim_id = 'sim_' + (safe_id(label) if label else dyn_id)
    pop_id = f'{dyn_id}Pop'

    if cell_type == 'fitzHughNagumo1969Cell':
        cell_attrs = {}
        for pname in ('a', 'b', 'I', 'phi'):
            p = params.get(pname)
            cell_attrs[pname] = str(getattr(p, 'value', 0) if p else 0)
        V0 = getattr(svs.get('V'), 'initial_value', None)
        W0 = getattr(svs.get('W'), 'initial_value', None)
        cell_attrs['V0'] = str(V0 if V0 is not None else 0.0)
        cell_attrs['W0'] = str(W0 if W0 is not None else 0.0)
    else:
        I_param = params.get('I')
        I_val = getattr(I_param, 'value', 0.8) if I_param else 0.8
        cell_attrs = {'I': str(I_val)}

    sv_names = list(svs.keys()) if svs else ['V', 'W']
    colors = ['#ee40FF', '#BBA0AA', '#44BBFF', '#22DD44']

    return {
        'is_network': False,
        'is_fhn': True,
        'cell_tag': cell_type,
        'cell_attrs': cell_attrs,
        'dyn_id': dyn_id,
        'dyn_name': dyn.name or 'FitzHugh-Nagumo',
        'sim_id': sim_id,
        'pop_id': pop_id,
        'dt': dt,
        'duration': duration,
        'time_scale': time_scale,
        'sv_names': sv_names,
        'colors': colors,
    }


def _build_std_cell_context(experiment):
    """Build context for a single standard NeuroML cell template."""
    dyn = experiment.dynamics

    cell_result = _render_cell_xml(dyn)
    if cell_result is None:
        return None

    custom_types = cell_result['custom_types']
    dyn_id = cell_result['dyn_id']
    params = dyn.parameters or {}

    integration = getattr(experiment, 'integration', None)
    dt = integration.step_size if integration else 0.01
    duration = integration.duration if integration else 1000.0
    raw_ts = (getattr(integration, 'time_scale', None) or 'ms') if integration else 'ms'
    time_scale = str(raw_ts) if str(raw_ts) in ('s', 'ms', 'us') else 'ms'

    label = getattr(experiment, 'label', None)
    sim_id = 'sim_' + (safe_id(label) if label else dyn_id)

    # Temperature handling
    tissue_start = params.get('tissue_startTemperature')
    tissue_end = params.get('tissue_endTemperature')
    tissue_change = params.get('tissue_changeTime')
    use_tissue = bool(tissue_start and tissue_end and tissue_change)
    net_temp = params.get('network_temperature')

    if use_tissue:
        sim_target = 'slice'
        quantity_prefix = 'net1/'
    else:
        sim_target = 'net1'
        quantity_prefix = ''

    return {
        'is_network': False,
        'is_fhn': False,
        'dyn_id': dyn_id,
        'sim_id': sim_id,
        'dt': dt,
        'duration': duration,
        'time_scale': time_scale,
        'has_inputs': bool(cell_result.get('input_xmls')),
        'custom_type_xmls': list(custom_types.values()),
        'conc_xmls': cell_result['conc_xmls'],
        'channel_xmls': cell_result['channel_xmls'],
        'cell_xml': cell_result['cell_xml'],
        'input_xmls': cell_result['input_xmls'],
        'input_refs': cell_result['input_refs'],
        # Temperature
        'use_tissue': use_tissue,
        'tissue_start': _nml_attr(tissue_start) if use_tissue else '',
        'tissue_end': _nml_attr(tissue_end) if use_tissue else '',
        'tissue_change': _nml_attr(tissue_change) if use_tissue else '',
        'net_temp': _nml_attr(net_temp) if net_temp else None,
        'sim_target': sim_target,
        'quantity_prefix': quantity_prefix,
    }


def _build_std_network_context(experiment):
    """Build context for a multi-population standard NeuroML network template.

    Mirrors the logic of ``_render_network_standard_neuroml_lems()`` but
    returns structured data instead of a rendered XML string.
    """
    from collections import OrderedDict

    network = experiment.network
    dynamics_lib = getattr(network, 'dynamics', None) or {}
    nodes = getattr(network, 'nodes', None) or []
    edges = getattr(network, 'edges', None) or []

    # Integration parameters
    integration = getattr(experiment, 'integration', None)
    dt = integration.step_size if integration else 0.01
    duration = integration.duration if integration else 1000.0
    raw_ts = (getattr(integration, 'time_scale', None) or 'ms') if integration else 'ms'
    time_scale = str(raw_ts) if str(raw_ts) in ('s', 'ms', 'us') else 'ms'

    label = getattr(experiment, 'label', None)
    dyn_id = safe_id(
        (experiment.dynamics.name if experiment.dynamics else None) or 'network'
    )
    sim_id = 'sim_' + (safe_id(label) if label else dyn_id)

    # ── Group nodes by dynamics name → populations ──
    groups = OrderedDict()
    for node in nodes:
        node_dyn = getattr(node, 'dynamics', None)
        if node_dyn:
            dyn_name = getattr(node_dyn, 'name', None) or str(node_dyn)
        else:
            dyn_name = dyn_id
        groups.setdefault(dyn_name, []).append(node)

    # ── Classify each group ──
    custom_types = {}
    cell_xmls_all = []
    input_xmls_all = []
    synapse_xmls = []

    populations = []
    node_pop_map = {}
    input_nodes = {}
    output_pops = []

    for dyn_name, group_nodes in groups.items():
        _dyn_lib_obj = dynamics_lib.get(dyn_name)
        _dyn_iri = getattr(_dyn_lib_obj, 'iri', '') or ''
        _nml_type = (_dyn_iri.split(':', 1)[1]
                     if _dyn_iri.startswith('neuroml:') else dyn_name)
        is_current_input = _nml_type in CURRENT_INPUT_TYPES
        is_event_source = _nml_type in EVENT_SOURCE_TYPES

        if is_current_input:
            for node in group_nodes:
                nid = getattr(node, 'id', 0)
                dyn_params = _normalize_edge_params(
                    getattr(_dyn_lib_obj, 'parameters', None))
                node_params = _normalize_edge_params(
                    getattr(node, 'parameters', None))
                merged_params = {**dyn_params, **node_params}
                param_strs = {}
                for pn, pv in merged_params.items():
                    val = getattr(pv, 'value', pv)
                    unit = getattr(pv, 'unit', None) or ''
                    if val is None:
                        val = getattr(pv, 'description', None)
                    if val is None:
                        continue
                    if unit:
                        nml_unit = _TVBO_TO_NML_UNIT.get(str(unit), str(unit))
                        param_strs[str(pn)] = f"{val} {nml_unit}"
                    else:
                        param_strs[str(pn)] = str(val)
                input_id = safe_id(dyn_name)
                input_nodes[nid] = {
                    'type': _nml_type,
                    'id': input_id,
                    'params': param_strs,
                }
                # Render the input component XML
                attr_parts = [f'id="{input_id}"']
                for pk, pv_str in param_strs.items():
                    attr_parts.append(f'{pk}="{pv_str}"')

                if _nml_type == 'compoundInput':
                    children_xml = _render_compound_input_children(
                        _dyn_lib_obj, indent=8)
                    input_xmls_all.append(
                        f'    <{_nml_type} {" ".join(attr_parts)}>\n'
                        f'{children_xml}\n'
                        f'    </{_nml_type}>'
                    )
                elif _nml_type == 'timedSynapticInput':
                    spike_xml = _render_event_children(
                        _dyn_lib_obj, time_scale)
                    if spike_xml:
                        input_xmls_all.append(
                            f'    <{_nml_type} {" ".join(attr_parts)}>\n'
                            f'{spike_xml}\n'
                            f'    </{_nml_type}>'
                        )
                    else:
                        input_xmls_all.append(
                            f'    <{_nml_type} {" ".join(attr_parts)}/>'
                        )
                else:
                    input_xmls_all.append(
                        f'    <{_nml_type} {" ".join(attr_parts)}/>'
                    )
            continue

        if is_event_source:
            dyn_obj = dynamics_lib.get(dyn_name)
            for sub_idx, node in enumerate(group_nodes):
                nid = getattr(node, 'id', sub_idx)
                dyn_params = _normalize_edge_params(
                    getattr(_dyn_lib_obj, 'parameters', None))
                node_params = _normalize_edge_params(
                    getattr(node, 'parameters', None))
                merged_params = {**dyn_params, **node_params}
                param_strs = {}
                for pn, pv in merged_params.items():
                    val = getattr(pv, 'value', pv)
                    unit = getattr(pv, 'unit', None) or ''
                    if val is not None:
                        if unit:
                            nml_unit = _TVBO_TO_NML_UNIT.get(
                                str(unit), str(unit))
                            param_strs[str(pn)] = f"{val} {nml_unit}"
                        else:
                            param_strs[str(pn)] = str(val)

                comp_id = safe_id(dyn_name)
                pop_id = f"{safe_id(dyn_name)}_pop"
                node_pop_map[nid] = (pop_id, 0)
                populations.append({
                    "id": pop_id,
                    "component": comp_id,
                    "size": 1,
                    "node_ids": [nid],
                    "dyn_name": dyn_name,
                    "is_input": True,
                })

                spike_children_xml = _render_event_children(
                    dyn_obj, time_scale)

                attr_parts = [f'id="{comp_id}"']
                for pk, pv_str in param_strs.items():
                    attr_parts.append(f'{pk}="{pv_str}"')
                if spike_children_xml:
                    input_xmls_all.append(
                        f'    <{_nml_type} {" ".join(attr_parts)}>\n'
                        f'{spike_children_xml}\n'
                        f'    </{_nml_type}>'
                    )
                else:
                    input_xmls_all.append(
                        f'    <{_nml_type} {" ".join(attr_parts)}/>'
                    )
            continue

        # ── Normal cell type ──
        dyn_obj = dynamics_lib.get(dyn_name)
        if dyn_obj is None:
            dyn_obj = experiment.dynamics
        if dyn_obj is None:
            continue

        cell_id = safe_id(dyn_name)

        cell_result = _render_cell_xml(dyn_obj, dyn_id=cell_id,
                                       custom_types=custom_types)
        if cell_result is not None:
            for ch in cell_result['channel_xmls']:
                cell_xmls_all.append(ch)
            for cm in cell_result['conc_xmls']:
                cell_xmls_all.append(cm)
            cell_xmls_all.append(cell_result['cell_xml'])
            for inp_xml in cell_result['input_xmls']:
                input_xmls_all.append(inp_xml)
            custom_types = cell_result['custom_types']
            sv_var = 'v'
        else:
            return None

        pop_id = safe_id(dyn_name) + "_pop"
        node_ids = []
        for idx, node in enumerate(group_nodes):
            nid = getattr(node, 'id', idx)
            node_pop_map[nid] = (pop_id, idx)
            node_ids.append(nid)

        has_positions = any(
            getattr(n, 'position', None) is not None
            for n in group_nodes)
        node_positions = []
        if has_positions:
            for node in group_nodes:
                pos = getattr(node, 'position', None)
                if pos:
                    node_positions.append(
                        (pos.x or 0, pos.y or 0, pos.z or 0))
                else:
                    node_positions.append((0, 0, 0))

        seg_ids = (cell_result.get('segment_ids', [])
                   if cell_result else [])

        populations.append({
            "id": pop_id,
            "component": cell_id,
            "size": len(group_nodes),
            "node_ids": node_ids,
            "dyn_name": dyn_name,
            "is_populationlist": has_positions,
            "node_positions": node_positions,
            "segment_ids": seg_ids,
        })
        recorded_nodes = [
            n for n in group_nodes
            if getattr(n, 'record', None) is not False
        ]
        if recorded_nodes:
            output_pops.append((pop_id, len(recorded_nodes), sv_var))

    # ── Process edges → synapses + connections + inputs ──
    synapse_set = {}
    connections = []
    explicit_inputs = []

    _SEG_TARGETING_KEYS = {
        'preSegmentId', 'preFractionAlong',
        'postSegmentId', 'postFractionAlong',
    }

    for edge_idx, edge in enumerate(edges):
        src = getattr(edge, 'source', None)
        tgt = getattr(edge, 'target', None)
        if src is None or tgt is None:
            continue
        src, tgt = int(src), int(tgt)

        if src in input_nodes:
            if tgt not in node_pop_map:
                continue
            inp_info = input_nodes[src]
            tgt_pop, tgt_idx = node_pop_map[tgt]
            inp_edge_params = _normalize_edge_params(
                getattr(edge, 'parameters', None))
            inp_weight = None
            inp_segmentId = None
            inp_fractionAlong = None
            for pn, pv in inp_edge_params.items():
                pn_str = str(pn)
                val = getattr(pv, 'value', pv)
                if pn_str == 'weight':
                    inp_weight = float(val) if val is not None else None
                elif pn_str == 'segmentId' and val is not None:
                    inp_segmentId = int(float(val))
                elif pn_str == 'fractionAlong' and val is not None:
                    inp_fractionAlong = float(val)
            explicit_inputs.append({
                'input_id': inp_info['id'],
                'target_pop': tgt_pop,
                'target_idx': tgt_idx,
                'weight': inp_weight,
                'segmentId': inp_segmentId,
                'fractionAlong': inp_fractionAlong,
            })
            continue

        if src not in node_pop_map or tgt not in node_pop_map:
            continue

        src_pop, src_idx = node_pop_map[src]
        tgt_pop, tgt_idx = node_pop_map[tgt]

        edge_coupling = getattr(edge, 'coupling', None)
        if not edge_coupling:
            edge_coupling = getattr(edge, 'dynamics', None)
        resolved_syn_dyn = None
        if edge_coupling:
            coup_str = str(edge_coupling)
            if coup_str in dynamics_lib:
                resolved_syn_dyn = dynamics_lib[coup_str]

        edge_params = _normalize_edge_params(
            getattr(edge, 'parameters', None))

        weight = None
        delay = None
        syn_params = {}
        _seg_targeting = {}

        for pname, pval in edge_params.items():
            pname = str(pname)
            val = getattr(pval, 'value', pval)
            unit = getattr(pval, 'unit', None) or ''
            if pname == 'weight':
                weight = float(val) if val is not None else None
            elif pname == 'delay':
                delay = val
                if unit:
                    delay = f"{val} {unit}"
            elif pname in _SEG_TARGETING_KEYS:
                if val is not None:
                    if 'Segment' in pname:
                        _seg_targeting[pname] = int(float(val))
                    else:
                        _seg_targeting[pname] = float(val)
            else:
                if unit:
                    nml_unit = _TVBO_TO_NML_UNIT.get(str(unit), str(unit))
                    syn_params[pname] = f"{val} {nml_unit}"
                else:
                    syn_params[pname] = str(val)

        syn_type = None
        if resolved_syn_dyn:
            nml = _nml_type_name(resolved_syn_dyn)
            syn_type = nml or str(edge_coupling)
        elif edge_coupling:
            coup_name = (getattr(edge_coupling, 'name', None)
                         or str(edge_coupling))
            syn_type = coup_name

        if syn_type is None:
            syn_type = f"syn{edge_idx}"

        if resolved_syn_dyn:
            syn_key = (str(edge_coupling),)
        else:
            syn_key = (syn_type, tuple(sorted(syn_params.items())))
        if syn_key not in synapse_set:
            if resolved_syn_dyn:
                syn_id = safe_id(str(edge_coupling))
            elif any(s_id == safe_id(syn_type)
                     for s_id in synapse_set.values()):
                syn_id = f"{safe_id(syn_type)}_{edge_idx}"
            else:
                syn_id = safe_id(syn_type)
            synapse_set[syn_key] = syn_id

            if resolved_syn_dyn:
                syn_lines = _render_nml_subtree(
                    resolved_syn_dyn, str(edge_coupling),
                    indent=4, custom_types=None)
                if syn_lines:
                    synapse_xmls.append('\n'.join(syn_lines))
            else:
                attr_parts = [f'id="{syn_id}"']
                for pk, pv_str in syn_params.items():
                    attr_parts.append(f'{pk}="{pv_str}"')
                synapse_xmls.append(
                    f'    <{syn_type} {" ".join(attr_parts)}/>')

        syn_id = synapse_set[syn_key]

        conn_class = 'chemical'
        if syn_type in _ELECTRICAL_SYNAPSE_TYPES:
            conn_class = 'electrical'
        elif syn_type in _CONTINUOUS_SYNAPSE_TYPES:
            conn_class = 'continuous'

        pre_component = None
        if conn_class == 'continuous':
            pre_component = f"silent_{syn_id}"

        connections.append({
            'from_pop': src_pop,
            'from_idx': src_idx,
            'to_pop': tgt_pop,
            'to_idx': tgt_idx,
            'synapse': syn_id,
            'syn_type': syn_type,
            'weight': weight,
            'delay': delay,
            'conn_class': conn_class,
            'pre_component': pre_component,
            **_seg_targeting,
        })

    # ── Detect PyNN types ──
    _PYNN_TYPES = {
        'IF_curr_alpha', 'IF_curr_exp', 'IF_cond_alpha', 'IF_cond_exp',
        'EIF_cond_exp_isfa_ista', 'EIF_cond_alpha_isfa_ista', 'HH_cond_exp',
        'expCondSynapse', 'alphaCondSynapse', 'expCurrSynapse',
        'alphaCurrSynapse', 'SpikeSourcePoisson',
    }
    _used_nml_types = set()
    for pop in populations:
        _pop_dyn = dynamics_lib.get(pop['dyn_name'])
        if _pop_dyn:
            _t = _nml_type_name(_pop_dyn)
            if _t:
                _used_nml_types.add(_t)
    for sid in synapse_set.values():
        _used_nml_types.add(sid)
    for inp in input_nodes.values():
        _used_nml_types.add(inp['type'])
    for inp_xml in input_xmls_all:
        for pt in _PYNN_TYPES:
            if f'<{pt} ' in inp_xml or f'<{pt}/' in inp_xml:
                _used_nml_types.add(pt)
    needs_pynn = bool(_used_nml_types & _PYNN_TYPES)

    # Render standalone synapse components from dynamics_lib
    rendered_syn_ids = set(synapse_set.values())
    for dlib_name, dlib_obj in dynamics_lib.items():
        dlib_type = _nml_type_name(dlib_obj)
        if dlib_type and dlib_type in _SYNAPSE_TYPES:
            sid = safe_id(dlib_name)
            if sid not in rendered_syn_ids:
                syn_lines = _render_nml_subtree(
                    dlib_obj, dlib_name, indent=4, custom_types=None)
                if syn_lines:
                    synapse_xmls.append('\n'.join(syn_lines))
                    rendered_syn_ids.add(sid)

    needs_inputs_include = bool(input_nodes) or any(
        p.get('is_input') for p in populations)

    # ── Build silent synapse list for continuous connections ──
    silent_ids = []
    _seen_silent = set()
    for conn in connections:
        pc = conn.get('pre_component')
        if pc and pc not in _seen_silent:
            _seen_silent.add(pc)
            silent_ids.append(pc)

    # ── Build pop lookup ──
    pop_map = {p['id']: p for p in populations}

    # ── Classify connections ──
    chem_conns = [c for c in connections if c['conn_class'] == 'chemical']
    elec_conns = [c for c in connections if c['conn_class'] == 'electrical']
    cont_conns = [c for c in connections if c['conn_class'] == 'continuous']

    # Chemical projection grouping
    _SEG_KEYS = {'preSegmentId', 'preFractionAlong',
                 'postSegmentId', 'postFractionAlong'}
    needs_projection = any(
        any(conn.get(sk) is not None for sk in _SEG_KEYS)
        for conn in chem_conns
    ) if chem_conns else False

    chem_projs = OrderedDict()
    if chem_conns and needs_projection:
        for conn in chem_conns:
            key = (conn['from_pop'], conn['to_pop'], conn['synapse'])
            chem_projs.setdefault(key, []).append(conn)

    elec_projs = OrderedDict()
    if elec_conns:
        for conn in elec_conns:
            key = (conn['from_pop'], conn['to_pop'], conn['synapse'])
            elec_projs.setdefault(key, []).append(conn)

    cont_projs = OrderedDict()
    if cont_conns:
        for conn in cont_conns:
            key = (conn['from_pop'], conn['to_pop'], conn['synapse'])
            cont_projs.setdefault(key, []).append(conn)

    # Explicit input classification
    has_weighted = any(
        inp.get('weight') is not None for inp in explicit_inputs)
    has_segment_targeting = any(
        inp.get('segmentId') is not None for inp in explicit_inputs)
    needs_input_list = (
        has_weighted or has_segment_targeting
        or any(pop_map.get(inp['target_pop'], {}).get('is_populationlist')
               for inp in explicit_inputs))

    input_groups = OrderedDict()
    if needs_input_list:
        for inp in explicit_inputs:
            key = (inp['input_id'], inp['target_pop'])
            input_groups.setdefault(key, []).append(inp)

    # Seed attribute
    seed = getattr(integration, 'seed', None) if integration else None
    if seed is None and integration:
        int_params = getattr(integration, 'parameters', None) or {}
        seed_param = int_params.get('seed')
        if seed_param is not None:
            seed = getattr(seed_param, 'value', seed_param)
    seed_attr = f' seed="{int(seed)}"' if seed is not None else ''

    return {
        'is_network': True,
        'is_fhn': False,
        'sim_id': sim_id,
        'dyn_id': dyn_id,
        'dt': dt,
        'duration': duration,
        'time_scale': time_scale,
        'seed_attr': seed_attr,
        # Pre-rendered XML fragments
        'custom_type_xmls': list(custom_types.values()),
        'cell_xmls_all': cell_xmls_all,
        'input_xmls_all': input_xmls_all,
        'synapse_xmls': synapse_xmls,
        'silent_ids': silent_ids,
        # Network data
        'populations': populations,
        'pop_map': pop_map,
        'output_pops': output_pops,
        # Connections (pre-classified)
        'chem_conns': chem_conns,
        'elec_conns': elec_conns,
        'cont_conns': cont_conns,
        'needs_projection': needs_projection,
        'chem_projs': chem_projs,
        'elec_projs': elec_projs,
        'cont_projs': cont_projs,
        # Inputs
        'explicit_inputs': explicit_inputs,
        'needs_input_list': needs_input_list,
        'input_groups': input_groups,
        # Includes
        'needs_pynn': needs_pynn,
        'needs_inputs_include': needs_inputs_include,
    }


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
    input_nodes = {}  # node_id -> {type, params, id}  (for current sources)

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
        # Resolve NeuroML type from dynamics library IRI
        _dyn_lib_obj = dynamics_lib.get(dyn_name)
        _dyn_iri = getattr(_dyn_lib_obj, 'iri', '') or ''
        _nml_type = _dyn_iri.split(':', 1)[1] if _dyn_iri.startswith('neuroml:') else dyn_name
        is_current_input = _nml_type in CURRENT_INPUT_TYPES
        is_event_source = _nml_type in EVENT_SOURCE_TYPES

        if is_current_input:
            # Current injection sources are NOT populations.
            # Each becomes a standalone component + explicitInput.
            for node in group_nodes:
                nid = getattr(node, "id", 0)
                node_params = _normalize_edge_params(
                    getattr(node, "parameters", None))
                param_strs = {}
                for pn, pv in node_params.items():
                    val = getattr(pv, 'value', pv)
                    unit = getattr(pv, 'unit', None) or ''
                    param_strs[str(pn)] = f"{val}{unit}"
                input_id = safe_id(dyn_name)
                input_nodes[nid] = {
                    'type': _nml_type,
                    'id': input_id,
                    'params': param_strs,
                }
            continue

        if is_event_source:
            # Event sources (spikeGenerator, spikeArray) ARE populations.
            # Use the standard type as component directly (not a custom CT).
            dyn_obj = dynamics_lib.get(dyn_name)
            integration = getattr(experiment, 'integration', None)
            ts = str(getattr(integration, 'time_scale', 'ms') or 'ms') if integration else 'ms'
            for sub_idx, node in enumerate(group_nodes):
                nid = getattr(node, "id", sub_idx)
                node_params = _normalize_edge_params(
                    getattr(node, "parameters", None))
                param_strs = {}
                for pn, pv in node_params.items():
                    val = getattr(pv, 'value', pv)
                    unit = getattr(pv, 'unit', None) or ''
                    if val is not None:
                        param_strs[str(pn)] = f"{val}{unit}"
                comp_id = safe_id(dyn_name)
                pop_id = f"{safe_id(dyn_name)}_pop"
                node_pop_map[nid] = (pop_id, 0)
                spike_children = _render_event_children(dyn_obj, ts)
                populations.append({
                    "id": pop_id,
                    "component": comp_id,
                    "size": 1,
                    "node_ids": [nid],
                    "dyn_name": dyn_name,
                    "is_input": True,
                    "input_type": _nml_type,
                    "input_id": comp_id,
                    "input_params": param_strs,
                    "spike_children_xml": spike_children,
                })
            continue

        # Normal cell type — resolve Dynamics object
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
    inputs = []       # list of {id, type, params, target_pop, target_idx}

    for edge_idx, edge in enumerate(edges):
        src = getattr(edge, "source", None)
        tgt = getattr(edge, "target", None)
        if src is None or tgt is None:
            continue

        src = int(src)
        tgt = int(tgt)

        # ── Handle edges FROM current-input nodes → explicitInput ──
        if src in input_nodes:
            if tgt not in node_pop_map:
                continue
            inp_info = input_nodes[src]
            tgt_pop, tgt_idx = node_pop_map[tgt]
            # Edge params may override input weight
            edge_params = _normalize_edge_params(
                getattr(edge, "parameters", None))
            inp_weight = None
            for pn, pv in edge_params.items():
                if str(pn) == 'weight':
                    inp_weight = float(getattr(pv, 'value', pv))
            inputs.append({
                'id': inp_info['id'],
                'type': inp_info['type'],
                'params': inp_info['params'],
                'target_pop': tgt_pop,
                'target_idx': tgt_idx,
                'weight': inp_weight,
            })
            continue

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
    # Current-source inputs have already been populated from edges above.
    # Additional inputs from experiment.stimulation could be added here.

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
    # SEC scaling is needed when state variables are dimensionless.  In that
    # case the TimeDerivative RHS is treated as pure numerics and must be
    # divided by SEC (a time constant) so that d(x)/dt has dimension per_time.
    #
    # SEC scaling is NOT needed when all state variables with TDs carry
    # physical units (e.g. mV, nA).  In that regime the equations are
    # assumed to be fully dimensioned and / SEC would double-count.
    #
    # NOTE:  Parameter units alone are NOT sufficient to declare the model
    # physically dimensioned — parameters may carry units as physical
    # annotations (e.g. A=3.25 mV, a=0.1 per_ms in JansenRit) without
    # the equations being dimensionally consistent.
    from tvbo.utils.units import unit_has_time_dimension, unit_to_lems_dimension, unit_to_lems_symbol

    def _svs_have_physical_units(svs):
        """True if at least one state variable has a non-dimensionless unit."""
        return any(
            unit_to_lems_dimension(getattr(sv, "unit", None)) != "none"
            for sv in svs.values()
        )

    _model_has_time_units = _svs_have_physical_units(svs)

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

    # ── PyLEMS reserved function names ──
    # These names are hard-coded in PyLEMS's ExprParser and CANNOT be used
    # as variable names in LEMS expressions.  If a model defines a derived
    # variable (or other symbol) with one of these names, we must rename it.
    _PYLEMS_RESERVED = {
        'exp', 'log', 'sqrt', 'sin', 'cos', 'tan', 'sinh', 'cosh',
        'tanh', 'abs', 'ceil', 'factorial', 'random', 'H',
    }
    _lems_rename = {}
    for name in list(str(k) for k in dvs.keys()):
        if name in _PYLEMS_RESERVED:
            _lems_rename[name] = f"{name}_dv"
    if _lems_rename:
        from sympy import Symbol
        _lems_subs = {Symbol(old): Symbol(new) for old, new in _lems_rename.items()}
        # Rebuild dvs with renamed keys
        dvs = {_lems_rename.get(str(k), str(k)): v for k, v in dvs.items()}
        # Add renamed names to all_names for correct printing
        all_names = [_lems_rename.get(n, n) for n in all_names]
    else:
        _lems_subs = {}

    def lems_expr(e):
        """Parse (if needed), inline model functions, then print as LEMS."""
        if not isinstance(e, _SympyBasic):
            # Parse with ORIGINAL names so SymPy recognises the symbols
            parse_names = all_names[:]
            for old in _lems_rename:
                if old not in parse_names:
                    parse_names.append(old)
            e = parse_eq(str(e), parameters=parse_names, functions=fn_names)
        e = inline_model_functions(e, dyn, all_names)
        if _lems_subs:
            e = e.subs(_lems_subs)
        return sympy_to_lems(e, parameters=all_names)

    def _parse_piecewise(rhs_str):
        """Return [(condition_str, value_str)] if rhs is Piecewise, else None."""
        try:
            parse_names = all_names[:]
            for old in _lems_rename:
                if old not in parse_names:
                    parse_names.append(old)
            expr = parse_eq(str(rhs_str), parameters=parse_names, functions=fn_names)
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

    # ── Regime detection for spike events ──
    # When an event has both condition and affect (e.g. spike + reset),
    # render as LEMS Regimes (integrating/refractory) instead of flat
    # OnCondition.  This matches the reference NeuroML execution model.
    regime_data = _build_regime_data(events)

    # ── Multi-population network context ──
    net_ctx = _build_network_context(experiment)

    # For single-cell models (no network), use flat OnCondition UNLESS the
    # model explicitly defines a ``refract`` parameter — mirroring the
    # NeuroML convention where izhikevichCell uses flat OnCondition while
    # adExIaFCell/iafRefCell use Regime with an explicit refractory period.
    # Regime adds a one-timestep delay that drifts phase for flat-reference
    # models.  Network mode always uses Regime for correct EventOut.
    has_refract_param = 'refract' in params
    if regime_data and net_ctx is None and not has_refract_param:
        regime_data = None

    # ── Dimension/symbol helpers ──
    # Use real LEMS dimensions and symbols so that jNeuroML outputs SI.
    # Parameter values in the YAML are in model units (e.g. -50 mV); the
    # LEMS symbol suffix (e.g. "mV") tells LEMS how to convert to SI.
    def _lems_dim(unit):
        return unit_to_lems_dimension(unit)
    def _lems_sym(unit):
        return unit_to_lems_symbol(unit)

    # needs_sec: whether TimeDerivative needs "/ SEC" to convert from
    # model time to SI seconds.  When all parameters and state variables
    # carry real LEMS dimensions, LEMS handles unit conversion natively
    # (e.g. tau="30 ms" → 0.03 s internally), so / SEC would double-count.
    # When any variable is dimensionless ("none"), the equations use raw
    # numbers in model-time units, so / SEC provides the numeric scaling.
    _all_dimensioned = all(
        unit_to_lems_dimension(getattr(p, 'unit', None)) != "none"
        for p in list(params.values()) + list(svs.values()) + list(dvs.values())
    )
    _needs_sec = (time_scale != "s") and not _all_dimensioned

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
        lems_dim=_lems_dim,
        lems_sym=_lems_sym,
        lems_dim_real=unit_to_lems_dimension,
        lems_sym_real=unit_to_lems_symbol,
        safe_id=safe_id,
        time_scale=time_scale,
        needs_sec=_needs_sec,
        all_dimensioned=_all_dimensioned,
        max_output_nodes=100,
        # Regime rendering for spike events
        regime_data=regime_data,
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

            ct_has_time_units = _svs_have_physical_units(ct_svs)

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

            # Use real LEMS dimensions so jNeuroML outputs SI.
            ct_time_scale = str(getattr(getattr(ct_dyn, 'time_scale', None), 'value', time_scale) or time_scale)
            ct_needs_sec = ct_time_scale != "s"
            ct_lems_dim = lambda u: unit_to_lems_dimension(u)
            ct_lems_sym_fn = lambda u: unit_to_lems_symbol(u)

            cell_contexts[ct_name] = {
                "dyn": ct_dyn,
                "dyn_id": safe_id(ct_name),
                "params": ct_params,
                "svs": ct_svs,
                "dvs": ct_dvs,
                "events": ct_events,
                "coupling_inputs": ct_coupling_inputs,
                "sv_names_set": ct_sv_names_set,
                "needs_sec": ct_needs_sec,
                "lems_expr": ct_lems_expr,
                "_parse_piecewise": ct_parse_pw,
                "lems_dim": ct_lems_dim,
                "lems_sym": ct_lems_sym_fn,
                "regime_data": _build_regime_data(ct_events),
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
                ct_has_time_units = _svs_have_physical_units(ct_svs)
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

                # Use real LEMS dimensions so jNeuroML outputs SI.
                syn_needs_sec = True  # synapse CTs always need SEC
                syn_lems_dim = lambda u: unit_to_lems_dimension(u)
                syn_lems_sym_fn = lambda u: unit_to_lems_symbol(u)

                cell_contexts[ct_name] = {
                    "dyn": ct_dyn,
                    "dyn_id": ct_name,
                    "params": ct_params,
                    "svs": ct_svs,
                    "dvs": ct_dvs,
                    "events": ct_events,
                    "coupling_inputs": ct_coupling_inputs,
                    "sv_names_set": ct_sv_names_set,
                    "needs_sec": syn_needs_sec,
                    "lems_expr": ct_lems_expr,
                    "_parse_piecewise": ct_parse_pw,
                    "lems_dim": syn_lems_dim,
                    "lems_sym": syn_lems_sym_fn,
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
    STD_LEMS_TEMPLATE = "neuroml/tvbo-neuroml-std-lems.xml.mako"
    STD_NETWORK_TEMPLATE = "neuroml/tvbo-neuroml-std-network-lems.xml.mako"
    HIER_CUSTOM_TEMPLATE = "neuroml/tvbo-neuroml-hier-custom-lems.xml.mako"

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

    def render_code(self, use_standard_types=False, **kwargs) -> str:
        """Render a complete, self-contained LEMS simulation file (``<Lems>`` root).

        Parameters
        ----------
        use_standard_types : bool
            When True and the dynamics uses NeuroML standard types
            (``iri: neuroml:*``), emit standard components with
            ``<Include file="Cells.xml"/>`` etc.  These includes are
            resolved by jNeuroML at runtime but NOT by the Python ``lems``
            validator, so this should only be True when the output is
            destined for ``run()``.
        """
        if use_standard_types and self.experiment:
            ctx = build_std_lems_context(self.experiment)
            if ctx is not None:
                from tvbo import templates
                if ctx.get('is_hier_custom'):
                    tpl = templates.lookup.get_template(
                        self.HIER_CUSTOM_TEMPLATE)
                elif ctx['is_network']:
                    tpl = templates.lookup.get_template(
                        self.STD_NETWORK_TEMPLATE)
                else:
                    tpl = templates.lookup.get_template(
                        self.STD_LEMS_TEMPLATE)
                return tpl.render(**ctx)
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
                self.validate(xml)
            paths = {"simulation": str(fpath)}

        return paths

    def validate(self, xml_string=None):
        """Validate rendered LEMS XML with PyLEMS. Returns True or raises.

        Standard NeuroML type outputs (``<Include file="Cells.xml"/>`` etc.)
        cannot be validated by PyLEMS because the type definition files are
        bundled with jNeuroML, not PyLEMS.  For those, validation is skipped
        here (jNeuroML validates them at runtime).
        """
        if xml_string is None:
            xml_string = self.render_code()
        if '<Include file="Cells.xml"/>' in xml_string:
            return True
        validate_lems_xml(xml_string)
        return True

    # Mapping of backend names to pyNeuroML runner functions.
    _BACKENDS = {
        'jneuroml': 'run_lems_with_jneuroml',
        'neuron':   'run_lems_with_jneuroml_neuron',
        'brian2':   'run_lems_with_jneuroml_brian2',
        'netpyne':  'run_lems_with_jneuroml_netpyne',
        'eden':     'run_lems_with_eden',
    }

    def run(self, backend='jneuroml', **kwargs) -> "ExperimentResult":
        """Run the LEMS simulation via a downstream simulator.

        Exports a self-contained monolithic LEMS file and executes it using
        one of the pyNeuroML runner functions.

        Parameters
        ----------
        backend : str
            Which simulator to use.  One of:

            * ``'jneuroml'`` (default) — reference LEMS engine (Java)
            * ``'neuron'``  — NEURON via jNeuroML
            * ``'brian2'``  — Brian2 via jNeuroML
            * ``'netpyne'`` — NetPyNE via jNeuroML
            * ``'eden'``    — EDEN simulator
        **kwargs
            Passed through to ``render_code()`` for template rendering.

        Returns
        -------
        ExperimentResult
            Simulation results loaded from output files.

        Raises
        ------
        ValueError
            If *backend* is not one of the supported names.
        RuntimeError
            If the downstream simulator fails.
        """
        import tempfile
        from pathlib import Path

        import numpy as np
        import xarray as xr

        from tvbo.data.types import ExperimentResult, SimulationResult

        backend = backend.lower()
        if backend not in self._BACKENDS:
            raise ValueError(
                f"Unknown NeuroML backend {backend!r}. "
                f"Supported: {', '.join(sorted(self._BACKENDS))}"
            )

        uses_std = (
            self.experiment and self.experiment.dynamics
            and _uses_neuroml_types(self.experiment.dynamics)
        )

        # Check for standard-type network (cell types in network.dynamics)
        _is_std_network = False
        _std_net_output_pops = []  # (pop_id, size, sv_var)
        if self.experiment:
            network = getattr(self.experiment, 'network', None)
            if network:
                _net_nodes = getattr(network, 'nodes', None) or []
                _net_edges = getattr(network, 'edges', None) or []
                _net_dyn_lib = getattr(network, 'dynamics', None) or {}
                if _net_nodes and _net_edges and _net_dyn_lib:
                    if any(_uses_neuroml_types(d) for d in _net_dyn_lib.values()
                           if hasattr(d, 'iri')):
                        _is_std_network = True
                        uses_std = True

        if _is_std_network:
            # Build output column names from network structure
            from collections import OrderedDict
            _groups = OrderedDict()
            for node in _net_nodes:
                nd = getattr(node, 'dynamics', None)
                dname = (getattr(nd, 'name', None) or str(nd)) if nd else 'dynamics'
                _groups.setdefault(dname, []).append(node)

            for dname, gnodes in _groups.items():
                _dobj = _net_dyn_lib.get(dname)
                _diri = getattr(_dobj, 'iri', '') or ''
                _dnml = _diri.split(':', 1)[1] if _diri.startswith('neuroml:') else dname
                if _dnml in CURRENT_INPUT_TYPES or _dnml in EVENT_SOURCE_TYPES:
                    continue
                # Respect record flag on nodes (default True)
                recorded = [
                    n for n in gnodes
                    if getattr(n, 'record', None) is not False
                ]
                if not recorded:
                    continue
                pop_id = safe_id(dname) + "_pop"
                pop_size = len(recorded)
                _std_net_output_pops.append((pop_id, pop_size, 'v'))

            sv_names = ['v']
            net_ctx = None
            cell_contexts = {}
        elif uses_std:
            iri = getattr(self.experiment.dynamics, 'iri', '') or ''
            nml_type = iri.split(':', 1)[1] if ':' in iri else ''
            if nml_type in ('fitzHughNagumoCell', 'fitzHughNagumo1969Cell'):
                sv_names = list(
                    (self.experiment.dynamics.state_variables or {}).keys()
                ) or ['V', 'W']
            else:
                sv_names = ['v']
            net_ctx = None
            cell_contexts = {}
        else:
            ctx = build_lems_context(self.experiment)
            sv_names = list(ctx['svs'].keys())
            net_ctx = ctx.get('net_ctx')
            cell_contexts = ctx.get('cell_contexts', {})

        xml = self.render_code(use_standard_types=True, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            lems_file = tmpdir / "tvbo_lems_sim.xml"
            lems_file.write_text(xml)
            (tmpdir / "results").mkdir()

            self._invoke_runner(backend, lems_file, tmpdir)

            # Different backends write output to different locations:
            # jNeuroML/NEURON respect the LEMS path ("results/*.dat"),
            # Brian2 and EDEN write to the working directory.
            dat_files = sorted(tmpdir.glob("results/*.dat"))
            if not dat_files:
                dat_files = sorted(tmpdir.glob("*.dat"))
            if not dat_files:
                raise RuntimeError(
                    f"Backend {backend!r} produced no .dat output files "
                    f"in {tmpdir}"
                )

            if len(dat_files) == 1:
                raw = np.loadtxt(str(dat_files[0]))
                time_data = raw[:, 0]
                values_data = raw[:, 1:]
            else:
                # Multi-population or multi-compartment: each file has its own
                # columns.  Load all and horizontally concatenate value columns.
                # Sort by stem to get deterministic order matching OutputFile order.
                dat_by_stem = {
                    f.stem: np.loadtxt(str(f))
                    for f in sorted(dat_files, key=lambda f: f.stem)
                }
                first = next(iter(dat_by_stem.values()))
                time_data = first[:, 0]
                value_parts = []
                if _is_std_network and _std_net_output_pops:
                    # Try to match file stems to output pops/components
                    matched_stems = set()
                    for pop_id, pop_size, sv_var in _std_net_output_pops:
                        comp = pop_id.removesuffix("_pop")
                        # Per-cell files: comp_0.dat, comp_1.dat, ...
                        cell_files = sorted(
                            [s for s in dat_by_stem
                             if re.match(rf'^{re.escape(comp)}_\d+$', s)],
                            key=lambda s: int(s.rsplit('_', 1)[1]))
                        if cell_files:
                            for cf in cell_files:
                                value_parts.append(dat_by_stem[cf][:, 1:])
                                matched_stems.add(cf)
                        elif comp in dat_by_stem:
                            value_parts.append(dat_by_stem[comp][:, 1:])
                            matched_stems.add(comp)
                        else:
                            for stem, arr in dat_by_stem.items():
                                if stem not in matched_stems and (
                                    stem.startswith(comp)
                                        or comp.startswith(stem)):
                                    value_parts.append(arr[:, 1:])
                                    matched_stems.add(stem)
                                    break
                    if not value_parts:
                        value_parts = [
                            arr[:, 1:] for arr in dat_by_stem.values()]
                else:
                    value_parts = [
                        arr[:, 1:] for arr in dat_by_stem.values()]
                values_data = np.column_stack(value_parts)

        if _is_std_network and _std_net_output_pops:
            # Build column names from the rendered LEMS OutputColumn quantities
            # — this is the most robust approach as it matches what was written.
            col_names = []
            try:
                import xml.etree.ElementTree as _ET
                _root = _ET.fromstring(xml)
                for _of in _root.iter():
                    if _of.tag.endswith('OutputFile') or _of.tag == 'OutputFile':
                        for _oc in _of:
                            if (_oc.tag.endswith('OutputColumn')
                                    or _oc.tag == 'OutputColumn'):
                                q = _oc.get('quantity', '')
                                if q:
                                    col_names.append(q)
            except Exception:
                col_names = []

            if not col_names or len(col_names) != values_data.shape[1]:
                # Fallback: generate from pop metadata
                col_names = []
                for pop_id, pop_size, sv_var in _std_net_output_pops:
                    for idx in range(pop_size):
                        col_names.append(f"{pop_id}[{idx}]/{sv_var}")

            da = xr.DataArray(
                data=values_data,
                dims=['time', 'quantity'],
                coords={'time': time_data, 'quantity': col_names},
            )
        elif net_ctx and cell_contexts:
            col_names = []
            for pop in net_ctx['populations']:
                if pop.get('is_input'):
                    continue
                ct = cell_contexts.get(pop['dyn_name'], {})
                if ct.get('is_synapse'):
                    continue
                for sv_name in ct.get('svs', {}):
                    for idx in range(pop['size']):
                        col_names.append(f"{pop['id']}[{idx}]/{sv_name}")

            da = xr.DataArray(
                data=values_data,
                dims=['time', 'quantity'],
                coords={'time': time_data, 'quantity': col_names},
            )
        else:
            da = xr.DataArray(
                data=values_data.reshape(-1, len(sv_names)),
                dims=['time', 'variable'],
                coords={
                    'time': time_data,
                    'variable': sv_names,
                },
            )

        sim = SimulationResult(data=da)
        return ExperimentResult(
            integration=sim,
            source=self.experiment,
            name=getattr(self.experiment, 'label', None),
        )

    # ── private helpers ───────────────────────────────────────────────

    @staticmethod
    def _invoke_runner(backend: str, lems_file, tmpdir):
        """Call the appropriate pyNeuroML runner for *backend*."""
        import os
        import sys

        from pyneuroml import pynml

        runner_name = NeuroMLAdapter._BACKENDS[backend]
        runner = getattr(pynml, runner_name)

        # jNeuroML's NEURON and NetPyNE exports both need NEURON_HOME to
        # locate nrnivmodl for .mod file compilation.
        # When running inside a venv the binaries live in $VIRTUAL_ENV/bin,
        # so we derive NEURON_HOME from the nrniv location if unset.
        env_patch = {}
        if backend in ('neuron', 'netpyne') and 'NEURON_HOME' not in os.environ:
            import shutil
            from pathlib import Path
            nrniv = shutil.which('nrniv')
            # Fall back to the venv's bin/ directory when nrniv is not on PATH
            # (common when running from IDE extensions or non-activated venvs).
            if not nrniv:
                candidate = Path(sys.prefix) / 'bin' / 'nrniv'
                if candidate.exists():
                    nrniv = str(candidate)
            if nrniv:
                env_patch['NEURON_HOME'] = str(Path(nrniv).parent.parent)
                # Ensure nrniv and nrnivmodl are on PATH for jNeuroML's
                # ProcessManager.findNeuronHome() which searches PATH.
                nrniv_dir = str(Path(nrniv).parent)
                cur_path = os.environ.get('PATH', '')
                if nrniv_dir not in cur_path.split(os.pathsep):
                    env_patch['PATH'] = nrniv_dir + os.pathsep + cur_path

        # On macOS the JVM grabs foreground / bounces in the Dock even when
        # -Djava.awt.headless=true is set.  -Dapple.awt.UIElement=true
        # suppresses the Dock icon and foreground activation entirely.
        if sys.platform == 'darwin':
            existing = os.environ.get('JDK_JAVA_OPTIONS', '')
            extra = '-Dapple.awt.UIElement=true'
            if extra not in existing:
                env_patch['JDK_JAVA_OPTIONS'] = (existing + ' ' + extra).strip()

        old_env = {k: os.environ.get(k) for k in env_patch}
        os.environ.update(env_patch)

        # pyNeuroML's Brian2 runner accesses sys.argv[1] unconditionally
        # (pyneuroml/runners.py ≈ line 593).  Pad sys.argv so it doesn't
        # crash when invoked from contexts with a short argv (e.g. pytest).
        old_argv = sys.argv[:]
        if len(sys.argv) < 2:
            sys.argv.append("")

        # Brian2 needs special handling: jNeuroML generates buggy Python
        # (empty if-blocks, wrong variable name prefixing).  We run the
        # jNeuroML code-gen step, patch the script, and exec it ourselves.
        try:
            if backend == 'brian2':
                success = NeuroMLAdapter._run_brian2(
                    pynml, lems_file, tmpdir, old_argv
                )
            elif backend == 'netpyne':
                # jNeuroML's NetPyNE path fails for custom LEMS ComponentTypes:
                # libNeuroML can't look up non-standard NeuroML2 components in
                # the generated .net.nml.  We generate the scripts ourselves,
                # compile the .mod files, patch the Python, and exec it.
                success = NeuroMLAdapter._run_netpyne(
                    pynml, lems_file, tmpdir, old_argv
                )
            else:
                # Build kwargs — EDEN has a simpler API than jNeuroML runners.
                if backend == 'eden':
                    kwargs = dict(
                        load_saved_data=False,
                        verbose=False,
                    )
                else:
                    kwargs = dict(
                        nogui=True,
                        load_saved_data=False,
                        exec_in_dir=str(tmpdir),
                        verbose=False,
                        exit_on_fail=False,
                    )

                old_cwd = os.getcwd()
                if backend == 'eden':
                    os.chdir(str(tmpdir))

                try:
                    success = runner(str(lems_file.name), **kwargs)
                finally:
                    if backend == 'eden':
                        os.chdir(old_cwd)
                    sys.argv = old_argv
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        if not success:
            raise RuntimeError(
                f"pyNeuroML runner {runner_name}() failed for "
                f"{lems_file.name} (backend={backend!r})"
            )

    @staticmethod
    def _run_netpyne(pynml, lems_file, tmpdir, old_argv):
        """Run via NetPyNE with workaround for custom LEMS ComponentType cells.

        jNeuroML's NetPyNE export calls ``importNeuroML2SimulateAnalyze``
        which tries to look up the cell component in the exported
        ``.net.nml``.  For custom LEMS ``ComponentType`` cells (which are
        not standard NeuroML2 cell types), ``libNeuroML`` returns ``None``
        and the simulation crashes with ``AttributeError``.

        Work-around:
        1. Generate the NetPyNE ``.py`` + ``.mod`` files (no execution).
        2. Compile ``.mod`` files with ``nrnivmodl``.
        3. Parse ``.net.nml`` to extract population/component names.
        4. Patch the generated ``.py``:
           * Add ``'pointp': component`` to all ``recordTraces`` entries so
             NetPyNE records from the POINT_PROCESS (not a section range).
           * Replace ``importNeuroML2SimulateAnalyze`` with a direct
             ``netParams`` build + ``sim.createSimulateAnalyze``.
        5. Exec the patched script.
        """
        import re
        import os
        import shutil
        import subprocess
        import sys

        fname = str(lems_file.name)
        tmpdir_str = str(tmpdir)

        # Step 1: generate netpyne Python + mod + net.nml (no execution)
        ok = pynml.run_lems_with_jneuroml_netpyne(
            fname,
            only_generate_scripts=True,
            exec_in_dir=tmpdir_str,
            verbose=False,
            exit_on_fail=False,
        )
        if not ok:
            return False

        netpyne_py = tmpdir / (fname.replace('.xml', '_netpyne.py'))
        net_nml = tmpdir / (fname.replace('.xml', '.net.nml'))
        if not netpyne_py.exists() or not net_nml.exists():
            return False

        # Step 2: compile .mod files (only_generate_scripts skips compilation)
        nrnivmodl_bin = shutil.which('nrnivmodl')
        if nrnivmodl_bin:
            subprocess.run(
                [nrnivmodl_bin], cwd=tmpdir_str,
                capture_output=True, check=False,
            )

        # Step 3: parse .net.nml for population/component info
        nml_text = net_nml.read_text()
        pop_match = re.search(
            r'<population\s+id="([^"]+)"\s+component="([^"]+)"\s+size="([^"]+)"',
            nml_text,
        )
        if not pop_match:
            return False
        pop_name = pop_match.group(1)
        component = pop_match.group(2)
        pop_size = int(pop_match.group(3))

        # Step 4: patch the generated Python
        code = netpyne_py.read_text()

        # 4a: add 'pointp' key so NetPyNE records from the POINT_PROCESS
        code = re.sub(
            r"('conds':)",
            f"'pointp': '{component}', \\1",
            code,
        )

        # 4a.5: remove 'cellLabel' from conds — NetPyNE doesn't set cellLabel in
        # cell tags when using direct netParams (so the condition always fails)
        code = re.sub(r",\s*'cellLabel'\s*:\s*\d+", "", code)

        # 4b: replace importNeuroML2SimulateAnalyze with direct netParams build
        direct_build = (
            f"_import_os = __import__('os'); _import_glob = __import__('glob')\n"
            f"        from netpyne import specs as _np_specs\n"
            f"        for _d in _import_glob.glob(_import_os.path.join(_import_os.getcwd(), '*')):\n"
            f"            if not _import_os.path.isdir(_d): continue\n"
            f"            for _lib in ('libnrnmech.dylib', 'libnrnmech.so'):\n"
            f"                _p = _import_os.path.join(_d, _lib)\n"
            f"                if _import_os.path.isfile(_p):\n"
            f"                    if not hasattr(h, '{component}'):\n"
            f"                        h.nrn_load_dll(_p)\n"
            f"                    break\n"
            f"        _netParams = _np_specs.NetParams()\n"
            f"        _netParams.cellParams['{component}'] = {{\n"
            f"            'secs': {{'soma': {{\n"
            f"                'geom': {{'diam': 18.8, 'L': 18.8, 'Ra': 123.0}},\n"
            f"                'pointps': {{'{component}': {{'mod': '{component}', 'loc': 0.5}}}},\n"
            f"            }}}}\n"
            f"        }}\n"
            f"        _netParams.popParams['{pop_name}'] = {{'cellType': '{component}', 'numCells': {pop_size}}}\n"
            f"        from netpyne.sim import run as _np_run_mod\n"
            f"        _orig_preRun = _np_run_mod.preRun\n"
            f"        def _safe_preRun(_opr=_orig_preRun):\n"
            f"            _opr()\n"
            f"            sim.pc.set_maxstep(h.dt + 1)  # ensure mindelay > dt for psolve\n"
            f"        _np_run_mod.preRun = _safe_preRun\n"
            f"        try:\n"
            f"            sim.createSimulateAnalyze(netParams=_netParams, simConfig=self.simConfig)\n"
            f"        finally:\n"
            f"            _np_run_mod.preRun = _orig_preRun\n"
            f"        self.gids = {{'{pop_name}': [c.gid for c in sim.net.cells if c.tags.get('pop') == '{pop_name}']}}"
        )
        code = re.sub(
            r'self\.gids\s*=\s*sim\.importNeuroML2SimulateAnalyze\([^)]+\)',
            direct_build,
            code,
        )
        netpyne_py.write_text(code)

        # Step 5: exec the patched script (define class, instantiate, run)
        exec_globals = {}
        old_cwd = os.getcwd()
        sys.path.insert(0, tmpdir_str)
        os.chdir(tmpdir_str)
        try:
            exec(compile(code, str(netpyne_py), 'exec'), exec_globals)
            ns = exec_globals['NetPyNESimulation']()
            ns.run()
            return True
        except Exception:
            import traceback
            traceback.print_exc()
            return False
        finally:
            os.chdir(old_cwd)
            if tmpdir_str in sys.path:
                sys.path.remove(tmpdir_str)
            sys.argv = old_argv

    @staticmethod
    def _run_brian2(pynml, lems_file, tmpdir, old_argv):
        """Run via Brian2 with workarounds for jNeuroML code-gen bugs.

        jNeuroML's Brian2 exporter (as of v0.14.0) produces Python scripts
        with two known defects:
        1. Empty ``if show_gui:`` blocks (IndentationError).
        2. StateMonitor variable names incorrectly prefixed with the
           component id (e.g. ``'fhn_V'`` instead of ``'V'``).

        We work around both by generating the script via jNeuroML, patching
        it, and executing it ourselves.
        """
        import re
        import sys

        fname = str(lems_file.name)
        tmpdir_str = str(tmpdir)

        # Step 1: generate Brian2 Python via jNeuroML
        ok = pynml.run_jneuroml(
            "", fname, "-brian2",
            exec_in_dir=tmpdir_str, verbose=False, exit_on_fail=False,
        )
        if not ok:
            return False

        # Step 2: locate and patch the generated script
        brian2_py = tmpdir / (fname.replace(".xml", "_brian2.py"))
        if not brian2_py.exists():
            return False

        code = brian2_py.read_text()

        # Fix 1: fill empty ``if …:`` blocks with ``pass``
        code = re.sub(
            r'(if\s+[^:]+:\s*)\n\n(?=\S|#)',
            r'\1\n    pass\n\n',
            code,
        )

        # Fix 2: jNeuroML prefixes component-id to variable names in
        # StateMonitor calls and result-array accesses.  Replace
        # '<id>_<var>' with just '<var>' — the equations use bare names.
        code = re.sub(r"'(\w+?)_(\w+)'(\s*,\s*record=)", r"'\2'\3", code)
        code = re.sub(r'\.(\w+?)_(\w+)\[', r'.\2[', code)

        brian2_py.write_text(code)

        # Step 3: execute the patched script from tmpdir
        import os
        sys.argv[1] = "-nogui"
        sys.path.insert(0, tmpdir_str)
        old_cwd = os.getcwd()
        os.chdir(tmpdir_str)
        try:
            exec(compile(code, str(brian2_py), 'exec'))
            return True
        except Exception:
            return False
        finally:
            os.chdir(old_cwd)
            if tmpdir_str in sys.path:
                sys.path.remove(tmpdir_str)
            sys.argv = old_argv


# ---------------------------------------------------------------------------
# NeuroML reference helpers
# ---------------------------------------------------------------------------
#
# Utilities for running reference NeuroML/LEMS examples via jNeuroML,
# comparing traces, and plotting results.  Previously lived in a
# standalone ``_nml_helpers.py`` doc-local script; moved here so that
# ``from tvbo.adapters.neuroml import run_lems_example`` works everywhere
# (tests, notebooks, CI) without sys.path hacking.
# ---------------------------------------------------------------------------

import os as _os
import shutil as _shutil
import subprocess as _subprocess
import sys as _sys
import tempfile as _tempfile
import time as _time
import xml.etree.ElementTree as _ET
from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path

import numpy as _np

_NML2_REPO = "https://github.com/NeuroML/NeuroML2.git"
_NML2_BRANCH = "master"
_CACHE_DIR = _Path(_os.environ.get("TVBO_CACHE_DIR", _Path.home() / ".cache" / "tvbo"))

# Lazy singleton -- avoids cloning the repo at module import time.
_nml2_root_cache: _Path | None = None


def _resolve_nml2_root() -> _Path:
    """Find or fetch the NeuroML2 repository.

    Resolution order:
    1. ``NEUROML2_DIR`` environment variable (explicit override)
    2. Auto-clone to ``~/.cache/tvbo/NeuroML2`` (works anywhere with git)
    """
    global _nml2_root_cache
    if _nml2_root_cache is not None:
        return _nml2_root_cache

    env_dir = _os.environ.get("NEUROML2_DIR")
    if env_dir:
        p = _Path(env_dir)
        if (p / "LEMSexamples").is_dir():
            _nml2_root_cache = p
            return p
        raise FileNotFoundError(
            f"NEUROML2_DIR={env_dir} does not contain LEMSexamples/"
        )

    cached = _CACHE_DIR / "NeuroML2"
    if (cached / "LEMSexamples").is_dir():
        _nml2_root_cache = cached
        return cached

    print(f"Cloning NeuroML2 reference repo to {cached} ...")
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _subprocess.run(
        ["git", "clone", "--depth", "1", "-b", _NML2_BRANCH, _NML2_REPO, str(cached)],
        check=True, capture_output=True, text=True,
    )
    _nml2_root_cache = cached
    return cached


def get_lems_examples_dir() -> _Path:
    """Return the path to the LEMSexamples directory in the NeuroML2 repo."""
    return _resolve_nml2_root() / "LEMSexamples"


class _LazyLemsExamples:
    """Proxy so ``LEMS_EXAMPLES`` does not trigger a git clone at import."""
    def __repr__(self):
        return str(get_lems_examples_dir())
    def __str__(self):
        return str(get_lems_examples_dir())
    def __fspath__(self):
        return str(get_lems_examples_dir())
    def __truediv__(self, other):
        return get_lems_examples_dir() / other
    def is_dir(self):
        return get_lems_examples_dir().is_dir()
    def glob(self, pattern):
        return get_lems_examples_dir().glob(pattern)


LEMS_EXAMPLES = _LazyLemsExamples()


def run_lems_example(lems_file: str, cwd: str | _Path | None = None) -> dict[str, _np.ndarray]:
    """Run a LEMS XML file via jNeuroML and return {filename: array} for each .dat output.

    Parameters
    ----------
    lems_file : str
        Name of the LEMS file (e.g., 'LEMS_NML2_Ex9_FN.xml').
    cwd : path, optional
        Working directory.  Defaults to the LEMSexamples directory.

    Returns
    -------
    dict mapping output filename to (n_time, n_cols) numpy arrays.
    """
    from pyneuroml import JNEUROML_VERSION
    import pyneuroml
    jar_dir = _Path(pyneuroml.__file__).parent / "lib"
    jar = jar_dir / f"jNeuroML-{JNEUROML_VERSION}-jar-with-dependencies.jar"

    if cwd is None:
        cwd = get_lems_examples_dir()

    cwd = _Path(cwd)

    # Ensure results directory exists and clear stale outputs from prior runs.
    results_dir = cwd / "results"
    results_dir.mkdir(exist_ok=True)
    output_globs = ("*.dat", "*.v.dat", "*.h5", "*.csv")
    for pat in output_globs:
        for f in results_dir.glob(pat):
            f.unlink()

    start_time = _time.time()

    def _discover_neuron_home() -> str | None:
        """Resolve a usable NEURON home directory for jNeuroML NEURON mode."""
        env_home = _os.environ.get("NEURON_HOME") or _os.environ.get("NRNHOME")
        if env_home:
            return env_home

        try:
            import neuron  # type: ignore

            root = _Path(neuron.__file__).resolve().parent / ".data"
            if (root / "bin" / "nrniv").exists():
                return str(root)
        except Exception:
            pass

        # Fall back to _sys.prefix (venv root) when nrniv lives there
        candidate = _Path(_sys.prefix) / "bin" / "nrniv"
        if candidate.exists():
            return str(_Path(_sys.prefix))

        return None

    def _build_neuron_env() -> dict[str, str]:
        env = dict(_os.environ)
        neuron_home = _discover_neuron_home()
        if neuron_home:
            env.setdefault("NEURON_HOME", neuron_home)
            env.setdefault("NRNHOME", neuron_home)
        # Ensure venv bin/ is on PATH so jnml and nrniv are discoverable
        venv_bin = str(_Path(_sys.prefix) / "bin")
        cur_path = env.get("PATH", "")
        if venv_bin not in cur_path.split(_os.pathsep):
            env["PATH"] = venv_bin + _os.pathsep + cur_path
        return env

    def _run(lems_name: str) -> subprocess.CompletedProcess:
        return _subprocess.run(
            ["java", "-jar", str(jar), str(lems_name), "-nogui"],
            capture_output=True, text=True, cwd=str(cwd), timeout=600,
        )

    def _run_neuron_backend(lems_name: str) -> subprocess.CompletedProcess:
        """Generate and execute a LEMS model through NEURON backend."""
        env = _build_neuron_env()

        jnml = _shutil.which("jnml") or _shutil.which(
            "jnml", path=str(_Path(_sys.prefix) / "bin")
        )
        if not jnml:
            return _subprocess.CompletedProcess(
                args=["jnml", lems_name, "-neuron", "-nogui"],
                returncode=127,
                stdout="",
                stderr="jnml command not found in PATH",
            )

        generated = _subprocess.run(
            [jnml, str(lems_name), "-neuron", "-nogui"],
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=600,
            env=env,
        )
        if generated.returncode != 0:
            return generated

        nrnivmodl = None
        nrnhome = env.get("NRNHOME") or env.get("NEURON_HOME")
        if nrnhome:
            candidate = _Path(nrnhome) / "bin" / "nrnivmodl"
            if candidate.exists():
                nrnivmodl = str(candidate)
        if nrnivmodl is None:
            nrnivmodl = _shutil.which("nrnivmodl")

        compile_stdout = ""
        compile_stderr = ""
        if nrnivmodl:
            compiled = _subprocess.run(
                [nrnivmodl],
                capture_output=True,
                text=True,
                cwd=str(cwd),
                timeout=600,
                env=env,
            )
            compile_stdout = compiled.stdout
            compile_stderr = compiled.stderr
            if compiled.returncode != 0:
                return _subprocess.CompletedProcess(
                    args=compiled.args,
                    returncode=compiled.returncode,
                    stdout="\n".join([generated.stdout, compile_stdout]).strip(),
                    stderr="\n".join([generated.stderr, compile_stderr]).strip(),
                )

        nrn_script = cwd / f"{_Path(lems_name).stem}_nrn.py"
        if not nrn_script.exists():
            return _subprocess.CompletedProcess(
                args=[_sys.executable, nrn_script.name],
                returncode=1,
                stdout="\n".join([generated.stdout, compile_stdout]).strip(),
                stderr="\n".join(
                    [generated.stderr, compile_stderr, f"Generated NEURON script not found: {nrn_script.name}"]
                ).strip(),
            )

        ran = _subprocess.run(
            [_sys.executable, nrn_script.name],
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=600,
            env=env,
        )
        return _subprocess.CompletedProcess(
            args=ran.args,
            returncode=ran.returncode,
            stdout="\n".join([generated.stdout, compile_stdout, ran.stdout]).strip(),
            stderr="\n".join([generated.stderr, compile_stderr, ran.stderr]).strip(),
        )

    def _collect_outputs() -> dict[str, _np.ndarray]:
        outputs = {}
        candidates = []
        for pat in output_globs:
            candidates.extend(results_dir.glob(pat))
            candidates.extend(cwd.glob(pat))

        fresh = [p for p in candidates if p.stat().st_mtime >= (start_time - 0.5)]
        if not fresh:
            for pat in output_globs:
                fresh.extend(results_dir.glob(pat))

        for out_file in sorted(fresh):
            # Skip binary outputs for now; this helper is trace-oriented.
            if out_file.suffix.lower() in {".h5"}:
                continue
            try:
                arr = _np.loadtxt(str(out_file))
            except Exception:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            outputs[out_file.name] = arr

        return outputs

    def _inject_probe_output(src: _Path, dst: _Path) -> bool:
        """Create a temporary LEMS file with OutputFile from Display lines."""
        tree = _ET.parse(src)
        root = tree.getroot()

        sim = root.find("Simulation")
        if sim is None:
            return False

        if sim.find("OutputFile") is not None or sim.find("EventOutputFile") is not None:
            return False

        quantities = []
        for disp in sim.findall("Display"):
            for line in disp.findall("Line"):
                q = line.attrib.get("quantity")
                if q and q not in quantities:
                    quantities.append(q)

        if not quantities:
            return False

        of = _ET.SubElement(sim, "OutputFile", id="of_auto", fileName="results/auto.dat")
        for i, q in enumerate(quantities):
            _ET.SubElement(of, "OutputColumn", id=f"c{i}", quantity=q)

        tree.write(dst, encoding="unicode")
        return True

    result = _run(lems_file)
    if result.returncode != 0:
        text = "\n".join([result.stdout or "", result.stderr or ""])
        needs_neuron = (
            "MULTICOMPARTMENTAL_CELL_MODEL" in text
            or "requires Neuron" in text
            or "Ex25" in lems_file
        )

        if needs_neuron:
            result = _run_neuron_backend(lems_file)

        if result.returncode != 0:
            merged = "\n".join([result.stdout or "", result.stderr or ""])
            raise RuntimeError(
                f"jNeuroML failed (rc={result.returncode}):\n{merged[-4000:]}"
            )

    outputs = _collect_outputs()

    # Some canonical examples only define Display lines and no OutputFile.
    # For those, auto-inject an OutputFile and rerun to extract traces.
    if not outputs:
        src = cwd / lems_file
        probe_name = f"__tvbo_probe__{_Path(lems_file).name}"
        probe_path = cwd / probe_name
        try:
            if _inject_probe_output(src, probe_path):
                result = _run(probe_name)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"jNeuroML probe run failed (rc={result.returncode}):\n{result.stderr[-2000:]}"
                    )
                outputs = _collect_outputs()
        finally:
            if probe_path.exists():
                probe_path.unlink()

    if not outputs:
        raise RuntimeError(f"jNeuroML produced no output .dat files for {lems_file}")

    return outputs



def compare_traces(
    ref_data: _np.ndarray,
    tvbo_data: _np.ndarray,
    ref_cols: list[str],
    tvbo_cols: list[str],
    time_col: int = 0,
    rtol: float = 0.05,
    atol: float = 1e-4,
):
    """Compare reference and TVBO traces, print metrics.

    Parameters
    ----------
    ref_data, tvbo_data : (n_time, n_cols) arrays
    ref_cols, tvbo_cols : column names (index 0 is time)
    time_col : which column is time (default 0)
    rtol, atol : tolerances for _np.allclose
    """
    # Interpolate TVBO onto reference time grid
    from scipy.interpolate import interp1d

    ref_time = ref_data[:, time_col]
    tvbo_time = tvbo_data[:, time_col]

    results = {}
    for col_name in ref_cols:
        if col_name == 'time':
            continue
        ref_idx = ref_cols.index(col_name)
        if col_name not in tvbo_cols:
            print(f"  {col_name}: not found in TVBO output, skipping")
            continue
        tvbo_idx = tvbo_cols.index(col_name)

        ref_trace = ref_data[:, ref_idx]
        tvbo_trace_raw = tvbo_data[:, tvbo_idx]

        # Interpolate to common grid
        f_tvbo = interp1d(tvbo_time, tvbo_trace_raw, kind='linear',
                          fill_value='extrapolate')
        tvbo_trace = f_tvbo(ref_time)

        # Metrics
        rmse = _np.sqrt(_np.mean((ref_trace - tvbo_trace) ** 2))
        max_err = _np.max(_np.abs(ref_trace - tvbo_trace))
        corr = _np.corrcoef(ref_trace, tvbo_trace)[0, 1] if _np.std(ref_trace) > 0 else 1.0
        close = _np.allclose(ref_trace, tvbo_trace, rtol=rtol, atol=atol)

        results[col_name] = {
            'rmse': rmse, 'max_err': max_err, 'corr': corr, 'close': close,
        }
        status = "✅" if close else "⚠️"
        print(f"  {col_name}: RMSE={rmse:.6f}  max_err={max_err:.6f}  "
              f"corr={corr:.6f}  {status}")

    return results


def plot_comparison(
    ref_data: _np.ndarray,
    tvbo_data: _np.ndarray,
    ref_cols: list[str],
    tvbo_cols: list[str],
    title: str = "",
    time_scale: float = 1.0,
    time_unit: str = "s",
):
    """Plot overlaid traces: reference vs TVBO.

    Parameters
    ----------
    ref_data, tvbo_data : arrays with time in col 0
    ref_cols, tvbo_cols : column names
    title : plot title
    time_scale : multiply time by this factor for display
    time_unit : label for x axis
    """
    import matplotlib.pyplot as plt

    sv_names = [c for c in ref_cols if c != 'time' and c in tvbo_cols]
    n = len(sv_names)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), squeeze=False, sharex=True)

    ref_time = ref_data[:, 0] * time_scale
    tvbo_time = tvbo_data[:, 0] * time_scale

    for i, name in enumerate(sv_names):
        ax = axes[i, 0]
        ref_idx = ref_cols.index(name)
        tvbo_idx = tvbo_cols.index(name)

        ax.plot(ref_time, ref_data[:, ref_idx], label=f'NeuroML (ref)', alpha=0.8)
        ax.plot(tvbo_time, tvbo_data[:, tvbo_idx], '--', label='TVBO', alpha=0.8)
        ax.set_ylabel(name)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel(f"Time ({time_unit})")
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    plt.show()


# ── Display-aware comparison plotting ────────────────────────────────


@_dataclass
class _DisplayLine:
    """One <Line> inside a <Display>."""
    line_id: str
    quantity: str
    color: str
    scale: str


@_dataclass
class _Display:
    """One <Display> element from a LEMS Simulation."""
    display_id: str
    title: str
    time_scale: str
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    lines: list[_DisplayLine]


def parse_lems_displays(lems_file: str) -> list[_Display]:
    """Parse Display + Line elements from a LEMS XML file.

    Parameters
    ----------
    lems_file : str
        LEMS filename (e.g. 'LEMS_NML2_Ex9_FN.xml') resolved relative
        to the LEMSexamples directory.

    Returns
    -------
    list of _Display objects with their Line children.
    """
    path = LEMS_EXAMPLES / lems_file
    tree = _ET.parse(path)
    root = tree.getroot()
    sim = root.find("Simulation")
    if sim is None:
        return []

    displays = []
    for disp in sim.findall("Display"):
        lines = []
        for ln in disp.findall("Line"):
            lines.append(_DisplayLine(
                line_id=ln.attrib.get("id", ""),
                quantity=ln.attrib.get("quantity", ""),
                color=ln.attrib.get("color", "#000000"),
                scale=ln.attrib.get("scale", "1"),
            ))
        displays.append(_Display(
            display_id=disp.attrib.get("id", ""),
            title=disp.attrib.get("title", ""),
            time_scale=disp.attrib.get("timeScale", "1ms"),
            xmin=float(disp.attrib.get("xmin", "0")),
            xmax=float(disp.attrib.get("xmax", "100")),
            ymin=float(disp.attrib.get("ymin", "-80")),
            ymax=float(disp.attrib.get("ymax", "40")),
            lines=lines,
        ))
    return displays


def _match_quantity_to_col(quantity: str, tvbo_cols: list[str]) -> str | None:
    """Best-effort match a LEMS quantity path to a TVBO column name.

    LEMS uses e.g. ``izpopBurst[0]/v`` while TVBO produces
    ``izBurst_pop[0]/v``.  We try progressively looser matching.
    """
    # Direct match
    if quantity in tvbo_cols:
        return quantity

    # Extract the variable part after last /
    parts = quantity.rsplit("/", 1)
    var_suffix = parts[-1] if len(parts) > 1 else quantity

    # Try matching by suffix
    candidates = [c for c in tvbo_cols if c.endswith("/" + var_suffix)]
    if len(candidates) == 1:
        return candidates[0]

    # Try matching by population index pattern: pop[N]/var
    idx_match = re.search(r'\[(\d+)\]', quantity)
    if idx_match:
        idx = idx_match.group(0)
        candidates = [c for c in tvbo_cols
                      if idx in c and c.endswith("/" + var_suffix)]
        if len(candidates) == 1:
            return candidates[0]

    return None


def _scale_factor(scale_str: str) -> float:
    """Convert a LEMS scale string like '1mV' to a numeric factor."""
    scale_str = scale_str.strip()
    unit_factors = {
        'V': 1.0, 'mV': 1e3, 'uV': 1e6,
        'A': 1.0, 'nA': 1e9, 'pA': 1e12, 'uA': 1e6,
        'S': 1.0, 'nS': 1e9, 'uS': 1e6,
        'ms': 1e3, 's': 1.0,
        'Hz': 1.0,
    }
    for unit, factor in sorted(unit_factors.items(), key=lambda x: -len(x[0])):
        if scale_str.endswith(unit):
            num = scale_str[:-len(unit)].strip()
            num_val = float(num) if num else 1.0
            return num_val * factor
    try:
        return float(scale_str)
    except ValueError:
        return 1.0


def _find_ref_column(quantity: str, ref_outputs: dict,
                     output_columns: dict | None = None
                     ) -> tuple[str, int] | None:
    """Find which reference .dat file and column index contains a quantity.

    Parameters
    ----------
    quantity : str
        LEMS quantity path e.g. ``iafPop[0]/v``
    ref_outputs : dict
        {filename: array} from run_lems_example()
    output_columns : dict, optional
        {filename: [quantity_strings]} parsed from OutputFile/OutputColumn
        If None, uses positional order.

    Returns
    -------
    (filename, col_index) or None
    """
    if output_columns:
        for fname, cols in output_columns.items():
            if quantity in cols:
                idx = cols.index(quantity) + 1  # +1 because col 0 is time
                if fname in ref_outputs and idx < ref_outputs[fname].shape[1]:
                    return (fname, idx)

    # Fallback: try to match by position across all output files
    return None


def parse_lems_output_columns(lems_file: str) -> dict[str, list[str]]:
    """Parse OutputFile → OutputColumn quantities from a LEMS file.

    Returns
    -------
    dict mapping output filename (e.g. 'ex14.dat') to list of quantity strings.
    """
    path = LEMS_EXAMPLES / lems_file
    tree = _ET.parse(path)
    root = tree.getroot()
    sim = root.find("Simulation")
    if sim is None:
        return {}

    result = {}
    for of in sim.findall("OutputFile"):
        fname = of.attrib.get("fileName", "")
        # Strip path prefixes
        if fname.startswith("./"):
            fname = fname[2:]
        if fname.startswith("results/"):
            fname = fname[len("results/"):]
        quantities = [oc.attrib.get("quantity", "")
                      for oc in of.findall("OutputColumn")]
        result[fname] = quantities
    return result


def plot_lems_comparison(
    lems_file: str,
    ref_outputs: dict[str, _np.ndarray],
    tvbo_result=None,
    title_prefix: str = "",
):
    """Create publication-quality comparison plots mirroring LEMS Display layout.

    For each Display in the LEMS file, creates one subplot panel with:
    - Reference traces as solid lines (using original colors from LEMS)
    - TVBO traces as dashed lines (same color, slightly transparent)

    Parameters
    ----------
    lems_file : str
        Reference LEMS filename (e.g. 'LEMS_NML2_Ex2_Izh.xml')
    ref_outputs : dict
        {filename: array} from run_lems_example()
    tvbo_result : xarray.DataArray, optional
        ``result.integration.data`` from ``exp.run("neuroml")``.
        Expects dims ``(time, quantity)``.  If None, only reference is plotted.
    title_prefix : str, optional
        Prefix for figure titles (e.g. 'Ex2')
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    displays = parse_lems_displays(lems_file)
    output_cols = parse_lems_output_columns(lems_file)
    if not displays:
        return

    # Build auto.dat positional map: Display lines → column indices
    # When no explicit OutputFile exists, jNeuroML writes Display quantities
    # to auto.dat in order of appearance.
    auto_position = {}
    col_counter = 1  # col 0 is time
    for d in displays:
        for ln in d.lines:
            q = ln.quantity
            if q not in auto_position:
                auto_position[q] = col_counter
                col_counter += 1

    # Build ref lookup: quantity → (filename, col_idx)
    ref_lookup: dict[str, tuple[str, int]] = {}
    if output_cols:
        for fname, qs in output_cols.items():
            for i, q in enumerate(qs):
                ref_lookup[q] = (fname, i + 1)
    # For auto.dat fallback, use positional mapping
    if not ref_lookup and "auto.dat" in ref_outputs:
        for q, idx in auto_position.items():
            if idx < ref_outputs["auto.dat"].shape[1]:
                ref_lookup[q] = ("auto.dat", idx)

    # Extract TVBO data from xarray DataArray
    tvbo_time = None
    tvbo_lookup: dict[str, _np.ndarray] = {}  # col_name → 1-D array
    if tvbo_result is not None:
        tvbo_time = tvbo_result.coords['time'].values
        qty_dim = [d for d in tvbo_result.dims if d != 'time'][0]
        for col_name in tvbo_result.coords[qty_dim].values:
            tvbo_lookup[str(col_name)] = tvbo_result.sel({qty_dim: col_name}).values

    # Positional TVBO matching: for each variable name (e.g. 'v'), track
    # consumption order to handle multiple populations
    _tvbo_by_var: dict[str, list[str]] = {}
    for col_name in tvbo_lookup:
        var = col_name.rsplit("/", 1)[-1] if "/" in col_name else col_name
        _tvbo_by_var.setdefault(var, []).append(col_name)
    _tvbo_consumed: dict[str, int] = {}

    for display in displays:
        if not display.lines:
            continue

        fig, ax = plt.subplots(figsize=(10, 3.5))
        has_ref = False
        has_tvbo = False

        for line in display.lines:
            color = line.color
            if not color.startswith('#'):
                color = f'#{color}'
            scale = _scale_factor(line.scale)
            label = line.line_id or line.quantity.split("/")[-1]

            # Find reference data
            ref_info = ref_lookup.get(line.quantity)
            if ref_info:
                fname, col_idx = ref_info
                ref_arr = ref_outputs[fname]
                t_ref = ref_arr[:, 0] * _scale_factor(display.time_scale)
                y_ref = ref_arr[:, col_idx] * scale
                ax.plot(t_ref, y_ref, color=color, alpha=0.9,
                        linewidth=1.5, label=f'{label} (ref)')
                has_ref = True

            # Find matching TVBO column — try direct match, then positional
            tvbo_col = _match_quantity_to_col(line.quantity, list(tvbo_lookup.keys()))
            if not tvbo_col:
                var = line.quantity.rsplit("/", 1)[-1] if "/" in line.quantity else line.quantity
                candidates = _tvbo_by_var.get(var, [])
                pos = _tvbo_consumed.get(var, 0)
                if pos < len(candidates):
                    tvbo_col = candidates[pos]
                    _tvbo_consumed[var] = pos + 1

            if tvbo_col and tvbo_time is not None:
                t_tvbo = tvbo_time * _scale_factor(display.time_scale)
                y_tvbo = tvbo_lookup[tvbo_col] * scale
                ax.plot(t_tvbo, y_tvbo, color=color, alpha=0.5,
                        linewidth=1.5, linestyle='--', label=f'{label} (TVBO)')
                has_tvbo = True

        title = display.title
        if title_prefix:
            title = f"{title_prefix}: {title}"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"Time ({display.time_scale})")
        ax.set_ylim(display.ymin, display.ymax)
        ax.set_xlim(display.xmin, display.xmax)
        ax.grid(True, alpha=0.2)

        legend_handles = []
        if has_ref:
            legend_handles.append(
                Line2D([0], [0], color='gray', linewidth=1.5,
                       label='NeuroML ref (solid)'))
        if has_tvbo:
            legend_handles.append(
                Line2D([0], [0], color='gray', linewidth=1.5,
                       linestyle='--', alpha=0.6, label='TVBO (dashed)'))
        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper right', fontsize=8)

        fig.tight_layout()
        plt.show()
