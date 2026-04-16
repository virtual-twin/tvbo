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


def _render_fhn_lems(experiment, cell_type):
    """Generate LEMS XML using built-in NeuroML2 fitzHughNagumo(1969)Cell types.

    Uses ``<Include file="Cells.xml"/>`` etc. so jNeuroML, Brian2, NetPyNE,
    and EDEN all see recognised standard types.
    """
    dyn = experiment.dynamics
    params = dyn.parameters or {}
    svs = dyn.state_variables or {}

    # ── Integration ──
    integration = getattr(experiment, 'integration', None)
    dt = integration.step_size if integration else 0.01
    duration = integration.duration if integration else 200.0
    raw_ts = (getattr(integration, 'time_scale', None) or 's') if integration else 's'
    time_scale = str(raw_ts) if str(raw_ts) in ('s', 'ms', 'us') else 's'

    # ── Cell instance attributes ──
    dyn_id = safe_id(dyn.name or 'fhn')
    label = getattr(experiment, 'label', None)
    sim_id = 'sim_' + (safe_id(label) if label else dyn_id)
    pop_id = f'{dyn_id}Pop'

    if cell_type == 'fitzHughNagumo1969Cell':
        # fitzHughNagumo1969Cell has: a, b, I, phi, V0, W0
        attrs = []
        for pname in ('a', 'b', 'I', 'phi'):
            p = params.get(pname)
            val = getattr(p, 'value', 0) if p else 0
            attrs.append(f'{pname}="{val}"')
        # V0, W0 come from state variable initial values
        V0 = getattr(svs.get('V'), 'initial_value', None)
        W0 = getattr(svs.get('W'), 'initial_value', None)
        attrs.append(f'V0="{V0 if V0 is not None else 0.0}"')
        attrs.append(f'W0="{W0 if W0 is not None else 0.0}"')
        cell_tag = 'fitzHughNagumo1969Cell'
    else:
        # fitzHughNagumoCell has only: I
        I_param = params.get('I')
        I_val = getattr(I_param, 'value', 0.8) if I_param else 0.8
        attrs = [f'I="{I_val}"']
        cell_tag = 'fitzHughNagumoCell'

    cell_attrs = ' '.join(attrs)

    # ── State variable names for output ──
    sv_names = list(svs.keys()) if svs else ['V', 'W']

    lines = [
        '<Lems>',
        f'  <Target component="{sim_id}"/>',
        '',
        '  <Include file="Cells.xml"/>',
        '  <Include file="Networks.xml"/>',
        '  <Include file="Inputs.xml"/>',
        '  <Include file="Simulation.xml"/>',
        '',
        f'  <{cell_tag} id="{dyn_id}" {cell_attrs}/>',
        '',
        '  <network id="net1">',
        f'    <population id="{pop_id}" component="{dyn_id}" size="1"/>',
        '  </network>',
        '',
        f'  <Simulation id="{sim_id}" length="{duration}{time_scale}" '
        f'step="{dt}{time_scale}" target="net1">',
        '',
        f'    <Display id="d1" title="{dyn.name or "FitzHugh-Nagumo"}" '
        f'timeScale="1{time_scale}" xmin="0" xmax="{int(duration)}" '
        f'ymin="-2.5" ymax="2.5">',
    ]
    colors = ['#ee40FF', '#BBA0AA', '#44BBFF', '#22DD44']
    for i, sv_name in enumerate(sv_names):
        color = colors[i % len(colors)]
        lines.append(
            f'      <Line id="{sv_name}" quantity="{pop_id}[0]/{sv_name}" '
            f'scale="1" color="{color}" timeScale="1{time_scale}"/>'
        )
    lines.append('    </Display>')
    lines.append('')
    lines.append(
        f'    <OutputFile id="of1" fileName="results/{dyn_id}.dat">'
    )
    for sv_name in sv_names:
        lines.append(
            f'      <OutputColumn id="{sv_name}" '
            f'quantity="{pop_id}[0]/{sv_name}"/>'
        )
    lines.append('    </OutputFile>')
    lines.append('')
    lines.append('  </Simulation>')
    lines.append('')
    lines.append('</Lems>')

    return '\n'.join(lines)


# ── Cell rendering helper ────────────────────────────────────────────

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


def _render_standard_neuroml_lems(experiment):
    """Generate LEMS XML using standard NeuroML2 types via generic tree walker.

    Walks the Dynamics composition tree using :func:`_render_nml_subtree`.
    Components are classified by IRI type and arranged in the NeuroML structure:

    - Channels → standalone XML + channelPopulation/channelDensity in cell
    - Inputs → standalone XML + explicitInput in network
    - Concentration models → standalone XML + species in intracellularProperties

    Supports cell types: ``pointCellCondBased``, ``cell``.
    Also supports multi-population networks when ``experiment.network`` has
    nodes and edges with cell types using NeuroML IRIs.

    Returns the complete LEMS XML string, or None if the dynamics can't be
    represented using standard types (caller should fall back to flat rendering).
    """
    # ── Multi-population network detection ──
    network = getattr(experiment, 'network', None)
    if network:
        nodes = getattr(network, 'nodes', None) or []
        edges = getattr(network, 'edges', None) or []
        dynamics_lib = getattr(network, 'dynamics', None) or {}
        if nodes and edges and dynamics_lib:
            # Check if any network cell type uses standard NeuroML types
            has_nml_cells = any(
                _uses_neuroml_types(d)
                for d in dynamics_lib.values()
                if hasattr(d, 'iri')
            )
            if has_nml_cells:
                return _render_network_standard_neuroml_lems(experiment)

    dyn = experiment.dynamics
    if not dyn:
        return None

    iri = getattr(dyn, 'iri', None) or ''
    if not iri.startswith('neuroml:'):
        return None

    cell_type = iri.split(':', 1)[1]

    if cell_type in ('fitzHughNagumoCell', 'fitzHughNagumo1969Cell'):
        return _render_fhn_lems(experiment, cell_type)

    # Delegate to cell rendering helper
    cell_result = _render_cell_xml(dyn)
    if cell_result is None:
        return None

    custom_types = cell_result['custom_types']
    dyn_id = cell_result['dyn_id']
    params = dyn.parameters or {}

    # ── Integration parameters ──
    integration = getattr(experiment, 'integration', None)
    dt = integration.step_size if integration else 0.01
    duration = integration.duration if integration else 1000.0
    raw_ts = (getattr(integration, 'time_scale', None) or 'ms') if integration else 'ms'
    time_scale = str(raw_ts) if str(raw_ts) in ('s', 'ms', 'us') else 'ms'

    label = getattr(experiment, 'label', None)
    sim_id = 'sim_' + (safe_id(label) if label else dyn_id)

    # ── Assemble XML ──
    _has_inputs = bool(cell_result.get('input_xmls'))
    lines = [
        '<Lems>',
        f'  <Target component="{sim_id}"/>',
        '',
        '  <Include file="Cells.xml"/>',
        '  <Include file="Networks.xml"/>',
    ]
    if _has_inputs:
        lines.append('  <Include file="Inputs.xml"/>')
    lines += [
        '  <Include file="Simulation.xml"/>',
        '',
    ]

    # Custom ComponentType definitions (before channels that reference them)
    for ct_xml in custom_types.values():
        lines.append(ct_xml)
        lines.append('')

    # Concentration models (before cell that references them)
    for cm_xml in cell_result['conc_xmls']:
        lines.append(cm_xml)
        lines.append('')

    # Channel definitions (standalone, before cell)
    for ch in cell_result['channel_xmls']:
        lines.append(ch)
        lines.append('')

    # Cell definition
    lines.append(cell_result['cell_xml'])
    lines.append('')

    # Input generators (standalone)
    for inp_xml in cell_result['input_xmls']:
        lines.append(inp_xml)
        lines.append('')

    input_refs = cell_result['input_refs']

    # ── Temperature / tissue wrapper ──
    tissue_start = params.get('tissue_startTemperature')
    tissue_end = params.get('tissue_endTemperature')
    tissue_change = params.get('tissue_changeTime')
    use_tissue = tissue_start and tissue_end and tissue_change
    net_temp = params.get('network_temperature')

    if use_tissue:
        lines.append('    <ComponentType name="baseTissue" description="...">')
        lines.append('        <Child name="network" type="network"/>')
        lines.append('    </ComponentType>')
        lines.append('')
        lines.append('    <ComponentType name="tissueWithVaryingTemperature" '
                     'description="..." extends="baseTissue">')
        lines.append('        <Exposure name="temperature" dimension="temperature"/>')
        lines.append('        <Parameter name="startTemperature" dimension="temperature"/>')
        lines.append('        <Parameter name="endTemperature" dimension="temperature"/>')
        lines.append('        <Parameter name="changeTime" dimension="time"/>')
        lines.append('        <Dynamics>')
        lines.append('            <StateVariable name="temperature" '
                     'exposure="temperature" dimension="temperature"/>')
        lines.append('            <OnStart>')
        lines.append('                <StateAssignment variable="temperature" '
                     'value="startTemperature"/>')
        lines.append('            </OnStart>')
        lines.append('            <OnCondition test="t .gt. changeTime">')
        lines.append('                <StateAssignment variable="temperature" '
                     'value="endTemperature"/>')
        lines.append('            </OnCondition>')
        lines.append('        </Dynamics>')
        lines.append('    </ComponentType>')
        lines.append('')
        lines.append(
            f'    <tissueWithVaryingTemperature id="slice" '
            f'startTemperature="{_nml_attr(tissue_start)}" '
            f'endTemperature="{_nml_attr(tissue_end)}" '
            f'changeTime="{_nml_attr(tissue_change)}">'
        )
        lines.append('        <network id="net1">')
        lines.append(f'            <population id="pop" component="{dyn_id}" size="1"/>')
        for inp_ref in input_refs:
            lines.append('    ' + inp_ref)
        lines.append('        </network>')
        lines.append('    </tissueWithVaryingTemperature>')
        sim_target = 'slice'
        quantity_prefix = 'net1/'
    elif net_temp:
        lines.append(
            f'    <network id="net1" type="networkWithTemperature" '
            f'temperature="{_nml_attr(net_temp)}">'
        )
        lines.append(f'        <population id="pop" component="{dyn_id}" size="1"/>')
        for inp_ref in input_refs:
            lines.append(inp_ref)
        lines.append('    </network>')
        sim_target = 'net1'
        quantity_prefix = ''
    else:
        lines.append('    <network id="net1">')
        lines.append(f'        <population id="pop" component="{dyn_id}" size="1"/>')
        for inp_ref in input_refs:
            lines.append(inp_ref)
        lines.append('    </network>')
        sim_target = 'net1'
        quantity_prefix = ''
    lines.append('')

    lines.append(
        f'    <Simulation id="{sim_id}" length="{duration}{time_scale}" '
        f'step="{dt}{time_scale}" target="{sim_target}">'
    )
    lines.append(f'        <OutputFile id="of0" fileName="results/{dyn_id}.dat">')
    lines.append(f'            <OutputColumn id="v" quantity="{quantity_prefix}pop[0]/v"/>')
    lines.append('        </OutputFile>')
    lines.append('    </Simulation>')
    lines.append('')
    lines.append('</Lems>')

    return '\n'.join(lines)


# ── Multi-population network rendering (standard NeuroML types) ──────

def _render_network_standard_neuroml_lems(experiment):
    """Render a multi-population NeuroML network using standard types.

    Uses :func:`_build_network_context` to extract populations, synapses,
    connections, and inputs from the experiment's network.  Each cell type
    is rendered via :func:`_render_cell_xml`.  Synapses use standard
    NeuroML component types (expOneSynapse, etc.).

    Returns the complete LEMS XML string, or None on failure.
    """
    from collections import OrderedDict

    network = experiment.network
    dynamics_lib = getattr(network, 'dynamics', None) or {}
    nodes = getattr(network, 'nodes', None) or []
    edges = getattr(network, 'edges', None) or []

    # ── Integration parameters ──
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
    groups = OrderedDict()  # dyn_name -> [node, ...]
    for node in nodes:
        node_dyn = getattr(node, 'dynamics', None)
        if node_dyn:
            dyn_name = getattr(node_dyn, 'name', None) or str(node_dyn)
        else:
            dyn_name = dyn_id
        groups.setdefault(dyn_name, []).append(node)

    # ── Classify each group: cell, input source, or current source ──
    custom_types = {}
    cell_xmls_all = []   # channel + cell definition XML blocks
    input_xmls_all = []  # standalone input component XML blocks
    synapse_xmls = []    # standalone synapse component XML blocks

    populations = []       # {id, component, size, node_ids, dyn_name}
    node_pop_map = {}      # node_id -> (pop_id, index_within_pop)
    input_nodes = {}       # node_id -> {id, type, params}
    output_pops = []       # (pop_id, size, sv_var_name) for OutputColumns

    for dyn_name, group_nodes in groups.items():
        # Resolve NeuroML type from dynamics library IRI
        _dyn_lib_obj = dynamics_lib.get(dyn_name)
        _dyn_iri = getattr(_dyn_lib_obj, 'iri', '') or ''
        _nml_type = _dyn_iri.split(':', 1)[1] if _dyn_iri.startswith('neuroml:') else dyn_name
        is_current_input = _nml_type in CURRENT_INPUT_TYPES
        is_event_source = _nml_type in EVENT_SOURCE_TYPES

        if is_current_input:
            for node in group_nodes:
                nid = getattr(node, 'id', 0)
                # Merge: dynamics library params as defaults, node params override
                dyn_params = _normalize_edge_params(
                    getattr(_dyn_lib_obj, 'parameters', None))
                node_params = _normalize_edge_params(
                    getattr(node, 'parameters', None))
                merged_params = {**dyn_params, **node_params}
                param_strs = {}
                for pn, pv in merged_params.items():
                    val = getattr(pv, 'value', pv)
                    unit = getattr(pv, 'unit', None) or ''
                    # For string-valued params (e.g. synapse, spikeTarget),
                    # fall back to description
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

                # compoundInput: render children from components
                if _nml_type == 'compoundInput':
                    children_xml = _render_compound_input_children(
                        _dyn_lib_obj, indent=8)
                    input_xmls_all.append(
                        f'    <{_nml_type} {" ".join(attr_parts)}>\n'
                        f'{children_xml}\n'
                        f'    </{_nml_type}>'
                    )
                # timedSynapticInput: render <spike> children from events
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
                # Merge: dynamics library params as defaults, node params override
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
                            nml_unit = _TVBO_TO_NML_UNIT.get(str(unit), str(unit))
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

                # Extract preset_time events → <spike> children
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

        # Check if this cell type uses standard NeuroML types
        cell_result = _render_cell_xml(dyn_obj, dyn_id=cell_id,
                                       custom_types=custom_types)
        if cell_result is not None:
            # Standard NeuroML cell — use tree walker output
            for ch in cell_result['channel_xmls']:
                cell_xmls_all.append(ch)
            for cm in cell_result['conc_xmls']:
                cell_xmls_all.append(cm)
            cell_xmls_all.append(cell_result['cell_xml'])
            # Cell inputs (components on cell, like pulseGenerator children)
            for inp_xml in cell_result['input_xmls']:
                input_xmls_all.append(inp_xml)
            custom_types = cell_result['custom_types']
            sv_var = 'v'
        else:
            # Not a standard cell — skip for now (will fall through to
            # Mako template rendering)
            return None

        pop_id = safe_id(dyn_name) + "_pop"
        node_ids = []
        for idx, node in enumerate(group_nodes):
            nid = getattr(node, 'id', idx)
            node_pop_map[nid] = (pop_id, idx)
            node_ids.append(nid)

        # Detect positioned nodes → populationList
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
        # Respect record flag on nodes (default True)
        recorded_nodes = [
            n for n in group_nodes
            if getattr(n, 'record', None) is not False
        ]
        if recorded_nodes:
            output_pops.append((pop_id, len(recorded_nodes), sv_var))

    # ── Process edges → synapses + connections + inputs ──
    synapse_set = {}    # dedup_key -> synapse_id
    connections = []    # {from_pop, from_idx, to_pop, to_idx, synapse, weight, delay}
    explicit_inputs = []  # {input_id, target_pop, target_idx}

    for edge_idx, edge in enumerate(edges):
        src = getattr(edge, 'source', None)
        tgt = getattr(edge, 'target', None)
        if src is None or tgt is None:
            continue
        src, tgt = int(src), int(tgt)

        # Edge from current input → explicitInput (with optional weight)
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

        # Get synapse type from coupling or dynamics name.
        # If the coupling/dynamics name matches a dynamics library entry, resolve
        # the IRI and parameters from that entry (TVBO-native definition).
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
        syn_params = {}  # {name: "formatted value string"}
        # Segment targeting parameters
        _seg_targeting = {}

        # 1) Edge parameters → weight/delay + segment targeting + synapse params
        _SEG_TARGETING_KEYS = {
            'preSegmentId', 'preFractionAlong',
            'postSegmentId', 'postFractionAlong',
        }
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

        # 2) Resolve synapse type name from IRI or coupling string
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

        # 3) Dedup — use dynamics library name as key when resolved,
        #    else (syn_type + edge params) for inline definitions.
        if resolved_syn_dyn:
            syn_key = (str(edge_coupling),)
        else:
            syn_key = (syn_type, tuple(sorted(syn_params.items())))
        if syn_key not in synapse_set:
            if resolved_syn_dyn:
                # Tree walker uses the dynamics library key as element id
                syn_id = safe_id(str(edge_coupling))
            elif any(s_id == safe_id(syn_type)
                     for s_id in synapse_set.values()):
                syn_id = f"{safe_id(syn_type)}_{edge_idx}"
            else:
                syn_id = safe_id(syn_type)
            synapse_set[syn_key] = syn_id

            # ── Render synapse XML ──
            if resolved_syn_dyn:
                # Dynamics-based: use the generic tree walker (same as
                # channels/gates).  Handles child modes (blockMechanism,
                # plasticityMechanism, etc.) recursively.
                syn_lines = _render_nml_subtree(
                    resolved_syn_dyn, str(edge_coupling),
                    indent=4, custom_types=None)
                if syn_lines:
                    synapse_xmls.append('\n'.join(syn_lines))
            else:
                # Inline string coupling: flat <synType id="..." params.../>
                # Merge edge params into syn_params (already done above)
                attr_parts = [f'id="{syn_id}"']
                for pk, pv_str in syn_params.items():
                    attr_parts.append(f'{pk}="{pv_str}"')
                synapse_xmls.append(
                    f'    <{syn_type} {" ".join(attr_parts)}/>')

        syn_id = synapse_set[syn_key]

        # Classify connection type based on synapse
        conn_class = 'chemical'
        if syn_type in _ELECTRICAL_SYNAPSE_TYPES:
            conn_class = 'electrical'
        elif syn_type in _CONTINUOUS_SYNAPSE_TYPES:
            conn_class = 'continuous'

        # For continuous connections, auto-generate silentSynapse preComponent
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

    # ── Assemble LEMS XML ──

    # Detect if PyNN types are used (cells, synapses, or spike sources)
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
    # Also check input components rendered (current inputs)
    for inp_xml in input_xmls_all:
        for pt in _PYNN_TYPES:
            if f'<{pt} ' in inp_xml or f'<{pt}/' in inp_xml:
                _used_nml_types.add(pt)
    needs_pynn = bool(_used_nml_types & _PYNN_TYPES)

    # Render standalone synapse components from dynamics_lib that weren't
    # already rendered as edge couplings (e.g. synapses referenced by
    # poissonFiringSynapse via synapse="..." attribute).
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
    lines = [
        '<Lems>',
        f'  <Target component="{sim_id}"/>',
        '',
        '  <Include file="Cells.xml"/>',
    ]
    if needs_pynn:
        lines.append('  <Include file="PyNN.xml"/>')
    lines.append('  <Include file="Networks.xml"/>')
    if needs_inputs_include:
        lines.append('  <Include file="Inputs.xml"/>')
    lines += [
        '  <Include file="Simulation.xml"/>',
        '',
    ]

    # Custom ComponentType definitions
    for ct_xml in custom_types.values():
        lines.append(ct_xml)
        lines.append('')

    # Cell type definitions (channels + cell)
    for cell_xml in cell_xmls_all:
        lines.append(cell_xml)
        lines.append('')

    # Input components
    for inp_xml in input_xmls_all:
        lines.append(inp_xml)
        lines.append('')

    # Synapse components
    for syn_xml in synapse_xmls:
        lines.append(syn_xml)
        lines.append('')

    # Auto-generate silentSynapse preComponents for continuous connections
    silent_ids = set()
    for conn in connections:
        if conn.get('pre_component') and conn['pre_component'] not in silent_ids:
            silent_ids.add(conn['pre_component'])
            lines.append(
                f'    <silentSynapse id="{conn["pre_component"]}"/>')
            lines.append('')

    # Network block
    lines.append('    <network id="net1">')

    # Build pop lookup
    pop_map = {p['id']: p for p in populations}

    # Populations
    for pop in populations:
        if pop.get('is_input'):
            continue
        if pop.get('is_populationlist'):
            lines.append(
                f'        <population id="{pop["id"]}" '
                f'type="populationList" component="{pop["component"]}">')
            for idx, pos in enumerate(pop['node_positions']):
                lines.append(
                    f'            <instance id="{idx}">'
                    f'<location x="{pos[0]}" y="{pos[1]}" z="{pos[2]}"/>'
                    f'</instance>')
            lines.append('        </population>')
        else:
            lines.append(
                f'        <population id="{pop["id"]}" '
                f'component="{pop["component"]}" size="{pop["size"]}"/>'
            )

    # Event source populations (spikeGenerator, spikeArray, etc.)
    for pop in populations:
        if pop.get('is_input'):
            lines.append(
                f'        <population id="{pop["id"]}" '
                f'component="{pop["component"]}" size="{pop["size"]}"/>'
            )

    # Synaptic connections — split into chemical, electrical, continuous
    chem_conns = [c for c in connections if c['conn_class'] == 'chemical']
    elec_conns = [c for c in connections if c['conn_class'] == 'electrical']
    cont_conns = [c for c in connections if c['conn_class'] == 'continuous']

    # Chemical connections — use <synapticConnection> for simple
    # connections (universally compatible with all cell types including
    # pointCellCondBased) and <projection>+<connection> only when segment
    # targeting attributes are present (multi-compartment cells).
    if chem_conns:
        _SEG_KEYS = {'preSegmentId', 'preFractionAlong',
                     'postSegmentId', 'postFractionAlong'}
        needs_projection = any(
            any(conn.get(sk) is not None for sk in _SEG_KEYS)
            for conn in chem_conns
        )

        if needs_projection:
            # Projection mode — needed for segment targeting
            chem_projs = OrderedDict()
            for conn in chem_conns:
                key = (conn['from_pop'], conn['to_pop'], conn['synapse'])
                chem_projs.setdefault(key, []).append(conn)
            for pidx, ((fp, tp, syn), proj_conns) in enumerate(
                chem_projs.items()
            ):
                proj_id = f"projection{pidx}"
                lines.append(
                    f'        <projection id="{proj_id}" '
                    f'presynapticPopulation="{fp}" '
                    f'postsynapticPopulation="{tp}" synapse="{syn}">')
                for cidx, conn in enumerate(proj_conns):
                    fp_pop = pop_map.get(fp)
                    tp_pop = pop_map.get(tp)
                    if fp_pop and fp_pop.get('is_populationlist'):
                        pre = (f'../{fp}/{conn["from_idx"]}/'
                               f'{fp_pop["component"]}')
                    else:
                        pre = f'../{fp}[{conn["from_idx"]}]'
                    if tp_pop and tp_pop.get('is_populationlist'):
                        post = (f'../{tp}/{conn["to_idx"]}/'
                                f'{tp_pop["component"]}')
                    else:
                        post = f'../{tp}[{conn["to_idx"]}]'

                    seg_attrs = ''
                    for sk in _SEG_KEYS:
                        sv = conn.get(sk)
                        if sv is not None:
                            seg_attrs += f' {sk}="{sv}"'

                    if (conn.get('weight') is not None
                            or conn.get('delay') is not None):
                        w = conn.get('weight', 1.0) or 1.0
                        d = conn.get('delay', '0ms') or '0ms'
                        lines.append(
                            f'            <connectionWD id="{cidx}" '
                            f'preCellId="{pre}" postCellId="{post}" '
                            f'weight="{w}" delay="{d}"{seg_attrs}/>')
                    else:
                        lines.append(
                            f'            <connection id="{cidx}" '
                            f'preCellId="{pre}" '
                            f'postCellId="{post}"{seg_attrs}/>')
                lines.append('        </projection>')
        else:
            # synapticConnection mode — works with all cell types
            for conn in chem_conns:
                fp = conn['from_pop']
                tp = conn['to_pop']
                syn = conn['synapse']
                fp_pop = pop_map.get(fp)
                tp_pop = pop_map.get(tp)
                if fp_pop and fp_pop.get('is_populationlist'):
                    pre = (f'{fp}/{conn["from_idx"]}/'
                           f'{fp_pop["component"]}')
                else:
                    pre = f'{fp}[{conn["from_idx"]}]'
                if tp_pop and tp_pop.get('is_populationlist'):
                    post = (f'{tp}/{conn["to_idx"]}/'
                            f'{tp_pop["component"]}')
                else:
                    post = f'{tp}[{conn["to_idx"]}]'

                if (conn.get('weight') is not None
                        or conn.get('delay') is not None):
                    w = conn.get('weight', 1.0) or 1.0
                    d = conn.get('delay', '0ms') or '0ms'
                    lines.append(
                        f'        <synapticConnectionWD from="{pre}" '
                        f'to="{post}" synapse="{syn}" '
                        f'destination="synapses" '
                        f'weight="{w}" delay="{d}"/>')
                else:
                    lines.append(
                        f'        <synapticConnection from="{pre}" '
                        f'to="{post}" synapse="{syn}" '
                        f'destination="synapses"/>')

    # Electrical projections (gap junctions)
    if elec_conns:
        # Group by (from_pop, to_pop, synapse) → projection
        elec_projs = OrderedDict()
        for conn in elec_conns:
            key = (conn['from_pop'], conn['to_pop'], conn['synapse'])
            elec_projs.setdefault(key, []).append(conn)
        for pidx, ((fp, tp, syn), proj_conns) in enumerate(elec_projs.items()):
            proj_id = f"elecProj_{pidx}"
            lines.append(
                f'        <electricalProjection id="{proj_id}" '
                f'presynapticPopulation="{fp}" '
                f'postsynapticPopulation="{tp}">')
            for cidx, conn in enumerate(proj_conns):
                w = conn.get('weight')
                if w is not None:
                    lines.append(
                        f'            <electricalConnectionInstanceW id="{cidx}" '
                        f'preCell="../{fp}[{conn["from_idx"]}]" '
                        f'postCell="../{tp}[{conn["to_idx"]}]" '
                        f'synapse="{syn}" weight="{w}"/>')
                else:
                    lines.append(
                        f'            <electricalConnection id="{cidx}" '
                        f'preCell="{conn["from_idx"]}" '
                        f'postCell="{conn["to_idx"]}" '
                        f'synapse="{syn}"/>')
            lines.append('        </electricalProjection>')

    # Continuous projections (graded synapses)
    if cont_conns:
        cont_projs = OrderedDict()
        for conn in cont_conns:
            key = (conn['from_pop'], conn['to_pop'], conn['synapse'])
            cont_projs.setdefault(key, []).append(conn)
        for pidx, ((fp, tp, syn), proj_conns) in enumerate(cont_projs.items()):
            proj_id = f"contProj_{pidx}"
            lines.append(
                f'        <continuousProjection id="{proj_id}" '
                f'presynapticPopulation="{fp}" '
                f'postsynapticPopulation="{tp}">')
            for cidx, conn in enumerate(proj_conns):
                pre_comp = conn.get('pre_component') or 'silent1'
                w = conn.get('weight')
                if w is not None:
                    lines.append(
                        f'            <continuousConnectionInstanceW id="{cidx}" '
                        f'preCell="../{fp}[{conn["from_idx"]}]" '
                        f'postCell="../{tp}[{conn["to_idx"]}]" '
                        f'preComponent="{pre_comp}" '
                        f'postComponent="{syn}" weight="{w}"/>')
                else:
                    lines.append(
                        f'            <continuousConnection id="{cidx}" '
                        f'preCell="{conn["from_idx"]}" '
                        f'postCell="{conn["to_idx"]}" '
                        f'preComponent="{pre_comp}" '
                        f'postComponent="{syn}"/>')
            lines.append('        </continuousProjection>')

    # Explicit inputs — use <inputList> when weighted, segment-targeted,
    # or targeting a populationList; otherwise use <explicitInput>.
    has_weighted = any(
        inp.get('weight') is not None for inp in explicit_inputs)
    has_segment_targeting = any(
        inp.get('segmentId') is not None for inp in explicit_inputs)
    needs_input_list = (
        has_weighted or has_segment_targeting
        or any(pop_map.get(inp['target_pop'], {}).get('is_populationlist')
               for inp in explicit_inputs))

    if needs_input_list:
        from collections import OrderedDict as _OD
        input_groups = _OD()
        for inp in explicit_inputs:
            key = (inp['input_id'], inp['target_pop'])
            input_groups.setdefault(key, []).append(inp)
        for gidx, ((inp_id, tgt_pop), group) in enumerate(
                input_groups.items()):
            lines.append(
                f'        <inputList id="inputList_{gidx}" '
                f'component="{inp_id}" population="{tgt_pop}">')
            for iidx, inp in enumerate(group):
                tp_pop = pop_map.get(tgt_pop)
                if tp_pop and tp_pop.get('is_populationlist'):
                    target = (f'../{tgt_pop}/{inp["target_idx"]}/'
                              f'{tp_pop["component"]}')
                else:
                    target = f'../{tgt_pop}[{inp["target_idx"]}]'
                seg_attrs = ''
                if inp.get('segmentId') is not None:
                    seg_attrs += f' segmentId="{inp["segmentId"]}"'
                if inp.get('fractionAlong') is not None:
                    seg_attrs += f' fractionAlong="{inp["fractionAlong"]}"'
                w = inp.get('weight')
                if w is not None:
                    lines.append(
                        f'            <inputW id="{iidx}" '
                        f'target="{target}"{seg_attrs} '
                        f'destination="synapses" weight="{w}"/>')
                else:
                    lines.append(
                        f'            <input id="{iidx}" '
                        f'target="{target}"{seg_attrs} '
                        f'destination="synapses"/>')
            lines.append('        </inputList>')
    else:
        for inp in explicit_inputs:
            lines.append(
                f'        <explicitInput target="{inp["target_pop"]}'
                f'[{inp["target_idx"]}]" '
                f'input="{inp["input_id"]}" destination="synapses"/>'
            )

    lines.append('    </network>')
    lines.append('')

    # Simulation block with output columns
    seed = getattr(integration, 'seed', None) if integration else None
    if seed is None and integration:
        int_params = getattr(integration, 'parameters', None) or {}
        seed_param = int_params.get('seed')
        if seed_param is not None:
            seed = getattr(seed_param, 'value', seed_param)
    seed_attr = f' seed="{int(seed)}"' if seed is not None else ''
    lines.append(
        f'    <Simulation id="{sim_id}" length="{duration}{time_scale}" '
        f'step="{dt}{time_scale}" target="net1"{seed_attr}>'
    )

    of_idx = 0
    for pop_id, pop_size, sv_var in output_pops:
        pop_obj = pop_map.get(pop_id)
        seg_ids = pop_obj.get('segment_ids', []) if pop_obj else []
        is_poplist = (pop_obj.get('is_populationlist', False)
                      if pop_obj else False)
        comp = pop_obj.get('component', '') if pop_obj else ''

        if seg_ids and len(seg_ids) > 1 and is_poplist:
            # Multi-compartment: separate OutputFile per cell instance
            for idx in range(pop_size):
                lines.append(
                    f'        <OutputFile id="of{of_idx}" '
                    f'fileName="results/{comp}_{idx}.dat">')
                for seg_id in seg_ids:
                    col_id = f"{idx}_{seg_id}"
                    lines.append(
                        f'            <OutputColumn id="{col_id}" '
                        f'quantity="{pop_id}/{idx}/{comp}/{seg_id}/'
                        f'{sv_var}"/>')
                lines.append('        </OutputFile>')
                of_idx += 1
        else:
            lines.append(
                f'        <OutputFile id="of{of_idx}" '
                f'fileName="results/{comp or dyn_id}.dat">')
            for idx in range(pop_size):
                col_id = f"{pop_id}_{idx}_{sv_var}"
                if is_poplist:
                    quantity = f'{pop_id}/{idx}/{comp}/{sv_var}'
                else:
                    quantity = f'{pop_id}[{idx}]/{sv_var}'
                lines.append(
                    f'            <OutputColumn id="{col_id}" '
                    f'quantity="{quantity}"/>')
            lines.append('        </OutputFile>')
            of_idx += 1

    lines.append('    </Simulation>')
    lines.append('')
    lines.append('</Lems>')

    return '\n'.join(lines)


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
            # Check primary dynamics for standard NeuroML types
            if self.experiment.dynamics and _uses_neuroml_types(self.experiment.dynamics):
                xml = _render_standard_neuroml_lems(self.experiment)
                if xml is not None:
                    return xml
            # Also check for standard-type network (cell types in network.dynamics)
            network = getattr(self.experiment, 'network', None)
            if network:
                dyn_lib = getattr(network, 'dynamics', None) or {}
                nodes = getattr(network, 'nodes', None) or []
                edges = getattr(network, 'edges', None) or []
                if nodes and edges and dyn_lib:
                    if any(_uses_neuroml_types(d) for d in dyn_lib.values()
                           if hasattr(d, 'iri')):
                        xml = _render_network_standard_neuroml_lems(self.experiment)
                        if xml is not None:
                            return xml
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
            nrniv = shutil.which('nrniv')
            if nrniv:
                from pathlib import Path
                env_patch['NEURON_HOME'] = str(Path(nrniv).parent.parent)

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
