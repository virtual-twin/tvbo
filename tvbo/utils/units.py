"""
Unit and Dimension Utilities
============================

Central mapping between TVBO's ``UnitEnum`` (QUDT-backed), LEMS dimensions,
SymPy units, and legacy free-text unit strings found in older model YAMLs.

The ``UnitEnum`` values use conventional abbreviations (ms, mV, nA, etc.)
as defined in the LinkML schema (``schema/tvbo_datamodel.yaml``).

* ``unit_to_lems_dimension(unit)`` — map a UnitEnum value to a LEMS dimension name
* ``unit_to_lems_symbol(unit)`` — map a UnitEnum value to the LEMS unit symbol
* ``unit_has_time_dimension(unit)`` — whether the unit carries a time component
* ``normalize_unit(raw)`` — convert legacy free-text strings to UnitEnum values
"""

# ── UnitEnum → LEMS dimension ────────────────────────────────────────

_UNIT_TO_LEMS_DIM = {
    # Time
    "s": "time", "ms": "time", "us": "time",
    # Rates / inverse time
    "per_s": "per_time", "per_ms": "per_time",
    "Hz": "per_time", "kHz": "per_time",
    # Voltage
    "V": "voltage", "mV": "voltage",
    # Inverse voltage
    "per_mV": "none",
    # Voltage rates
    "mV_per_ms": "none", "mV_per_s": "none",
    # Current
    "A": "current", "nA": "current", "pA": "current",
    # Capacitance
    "pF": "capacitance", "nF": "capacitance",
    # Conductance
    "nS": "conductance", "uS": "conductance",
    # Charge (inverse)
    "per_nC": "none", "per_pC": "none",
    # Concentration
    "mol_per_m3": "concentration", "mmol_per_m3": "concentration",
    # Volume
    "um3": "none",
    # Length
    "m": "none", "mm": "none", "cm": "none",
    # Velocity
    "m_per_s": "none", "mm_per_ms": "none",
    # Gain / compound
    "Hz_per_nA": "none",
    # Conductivity / permeability
    "S_per_m": "none", "H_per_m": "none",
    # Angular rate
    "rad_per_ms": "per_time", "rad_per_s": "per_time",
    # Angle
    "rad": "none",
    # Mass
    "kg": "none", "kg_per_s": "none",
    # Acceleration
    "m_per_s2": "none",
    # Force / stiffness
    "N_per_m": "none",
    # Time squared
    "s2": "none",
    # Per-unit (power systems)
    "per_unit": "none",
    # Dimensionless
    "dimensionless": "none", "percent": "none", "arbitrary_unit": "none",
}


def unit_to_lems_dimension(unit):
    """Return the LEMS dimension name for a UnitEnum value (or string).

    Returns the proper LEMS dimension (e.g. ``"voltage"``, ``"capacitance"``)
    when the unit has a known mapping, or ``"none"`` for dimensionless /
    unknown units.
    """
    if unit is None:
        return "none"
    key = str(unit).strip()
    return _UNIT_TO_LEMS_DIM.get(key, "none")


# ── UnitEnum → LEMS symbol (for Component values) ───────────────────

_UNIT_TO_LEMS_SYMBOL = {
    "s": "s", "ms": "ms", "us": "us",
    "per_s": "per_s", "per_ms": "per_ms",
    "Hz": "per_s", "kHz": "per_ms",
    "V": "V", "mV": "mV",
    "A": "A", "nA": "nA", "pA": "pA",
    "pF": "pF", "nF": "nF", "uF": "uF",
    "nS": "nS", "uS": "uS", "mS": "mS", "pS": "pS",
    "kohm": "kohm", "Mohm": "Mohm",
    "uA": "uA",
    "mM": "mM", "M": "M",
    "degC": "degC", "K": "K",
    "dimensionless": "",
}


def unit_to_lems_symbol(unit):
    """Return the LEMS unit symbol string for appending to numeric values.

    For dimensioned parameters (e.g. ``pF``, ``nS``, ``mV``), returns the
    matching LEMS unit symbol.  For dimensionless or unknown units, returns
    ``""``.
    """
    if unit is None:
        return ""
    key = str(unit).strip()
    return _UNIT_TO_LEMS_SYMBOL.get(key, "")


# ── Time-dimension detection (for SEC inference) ─────────────────────

_TIME_UNITS = {
    "s", "ms", "us",
    "per_s", "per_ms",
    "Hz", "kHz",
    "mV_per_ms", "mV_per_s",
    "rad_per_ms", "rad_per_s",
    "m_per_s", "mm_per_ms",
    "Hz_per_nA",
    "kg_per_s", "m_per_s2", "s2",
}


def unit_has_time_dimension(unit):
    """Return True if the unit carries a time component (T or T⁻¹).

    This is the key signal for NeuroML LEMS export: if any parameter
    in the RHS equation has a time dimension, the equation already
    carries time normalisation and ``/ SEC`` is not needed.
    """
    if unit is None:
        return False
    key = unit.value if hasattr(unit, "value") else str(unit)
    return key in _TIME_UNITS


# ── UnitEnum → LaTeX display symbol ──────────────────────────────────

_UNIT_TO_LATEX = {
    # Time
    "s": r"\mathrm{s}",
    "ms": r"\mathrm{ms}",
    "us": r"\mathrm{\mu s}",
    # Rates / inverse time (pure inverse → negative exponent)
    "per_s": r"\mathrm{s}^{-1}",
    "per_ms": r"\mathrm{ms}^{-1}",
    "Hz": r"\mathrm{Hz}",
    "kHz": r"\mathrm{kHz}",
    # Voltage
    "V": r"\mathrm{V}",
    "mV": r"\mathrm{mV}",
    # Inverse voltage
    "per_mV": r"\mathrm{mV}^{-1}",
    # Voltage rates (fraction)
    "mV_per_ms": r"\frac{\mathrm{mV}}{\mathrm{ms}}",
    "mV_per_s": r"\frac{\mathrm{mV}}{\mathrm{s}}",
    # Current
    "A": r"\mathrm{A}",
    "nA": r"\mathrm{nA}",
    "pA": r"\mathrm{pA}",
    "uA": r"\mathrm{\mu A}",
    # Capacitance
    "pF": r"\mathrm{pF}",
    "nF": r"\mathrm{nF}",
    "uF": r"\mathrm{\mu F}",
    # Conductance
    "nS": r"\mathrm{nS}",
    "uS": r"\mathrm{\mu S}",
    "mS": r"\mathrm{mS}",
    "pS": r"\mathrm{pS}",
    # Resistance
    "kohm": r"\mathrm{k\Omega}",
    "Mohm": r"\mathrm{M\Omega}",
    # Charge (inverse)
    "per_nC": r"\mathrm{nC}^{-1}",
    "per_pC": r"\mathrm{pC}^{-1}",
    # Concentration
    "mol_per_m3": r"\frac{\mathrm{mol}}{\mathrm{m}^3}",
    "mmol_per_m3": r"\frac{\mathrm{mmol}}{\mathrm{m}^3}",
    "mM": r"\mathrm{mM}",
    "M": r"\mathrm{M}",
    # Volume
    "um3": r"\mathrm{\mu m}^{3}",
    # Length
    "m": r"\mathrm{m}",
    "mm": r"\mathrm{mm}",
    "cm": r"\mathrm{cm}",
    # Velocity (fraction)
    "m_per_s": r"\frac{\mathrm{m}}{\mathrm{s}}",
    "mm_per_ms": r"\frac{\mathrm{mm}}{\mathrm{ms}}",
    # Gain / compound (fraction)
    "Hz_per_nA": r"\frac{\mathrm{Hz}}{\mathrm{nA}}",
    # Conductivity / permeability (fraction)
    "S_per_m": r"\frac{\mathrm{S}}{\mathrm{m}}",
    "H_per_m": r"\frac{\mathrm{H}}{\mathrm{m}}",
    # Angular rate (fraction)
    "rad_per_ms": r"\frac{\mathrm{rad}}{\mathrm{ms}}",
    "rad_per_s": r"\frac{\mathrm{rad}}{\mathrm{s}}",
    # Angle
    "rad": r"\mathrm{rad}",
    # Temperature
    "degC": r"^{\circ}\mathrm{C}",
    "K": r"\mathrm{K}",
    # Mass
    "kg": r"\mathrm{kg}",
    "kg_per_s": r"\frac{\mathrm{kg}}{\mathrm{s}}",
    # Acceleration (fraction)
    "m_per_s2": r"\frac{\mathrm{m}}{\mathrm{s}^2}",
    # Force / stiffness (fraction)
    "N_per_m": r"\frac{\mathrm{N}}{\mathrm{m}}",
    # Time squared
    "s2": r"\mathrm{s}^{2}",
    # Per-unit
    "per_unit": r"\mathrm{p.u.}",
    # Dimensionless
    "dimensionless": "",
    "percent": r"\%",
    "arbitrary_unit": r"\mathrm{a.u.}",
}


def unit_to_latex(unit):
    """Return a LaTeX string for the unit, suitable for wrapping in ``$...$``.

    Converts enum values like ``per_ms`` → ``\\mathrm{ms}^{-1}``,
    ``rad_per_ms`` → ``\\mathrm{rad}\\,\\mathrm{ms}^{-1}``.
    Returns an empty string for dimensionless / unknown units.
    """
    if unit is None:
        return ""
    key = unit.value if hasattr(unit, "value") else str(unit).strip()
    return _UNIT_TO_LATEX.get(key, r"\mathrm{" + key + r"}")


# ── Legacy string → UnitEnum normalisation ───────────────────────────

_LEGACY_TO_ENUM = {
    # Time
    "s": "s", "sec": "s", "second": "s",
    "ms": "ms", "millisecond": "ms",
    "us": "us", "microsecond": "us",
    # Rates
    "s^-1": "per_s", "s**-1": "per_s", "per_s": "per_s", "per_second": "per_s",
    "1/s": "per_s",
    "ms^-1": "per_ms", "ms**-1": "per_ms", "per_ms": "per_ms", "per_millisecond": "per_ms",
    "1/ms": "per_ms",
    "Hz": "Hz", "hz": "Hz", "hertz": "Hz",
    "kHz": "kHz", "khz": "kHz", "kilohertz": "kHz",
    # Voltage
    "V": "V", "volt": "V",
    "mV": "mV", "mv": "mV", "millivolt": "mV",
    "mV^-1": "per_mV", "mV**-1": "per_mV", "per_mV": "per_mV", "per_millivolt": "per_mV",
    "mV*ms^-1": "mV_per_ms", "mV/ms": "mV_per_ms", "mV_per_ms": "mV_per_ms",
    "millivolt_per_millisecond": "mV_per_ms",
    "mV/s": "mV_per_s", "mV*s^-1": "mV_per_s", "mV_per_s": "mV_per_s",
    "millivolt_per_second": "mV_per_s",
    # Current
    "A": "A", "ampere": "A",
    "nA": "nA", "nanoampere": "nA",
    "pA": "pA", "picoampere": "pA",
    # Capacitance
    "pF": "pF", "picofarad": "pF",
    "nF": "nF", "nanofarad": "nF",
    # Conductance
    "nS": "nS", "nanosiemens": "nS",
    "uS": "uS", "microsiemens": "uS",
    # Charge (inverse)
    "nC^-1": "per_nC", "(nC)^-1": "per_nC", "per_nC": "per_nC", "per_nanocoulomb": "per_nC",
    "(pC)^-1": "per_pC", "pC^-1": "per_pC", "per_pC": "per_pC", "per_picocoulomb": "per_pC",
    # Concentration
    "mol.m**-3": "mol_per_m3", "mol/m**3": "mol_per_m3",
    "mol/m^3": "mol_per_m3", "mol/m3": "mol_per_m3",
    "mol_per_m3": "mol_per_m3", "mole_per_cubic_metre": "mol_per_m3",
    "mMol/m**3": "mmol_per_m3", "mMol/m^3": "mmol_per_m3",
    "mmol/m**3": "mmol_per_m3", "mmol/m3": "mmol_per_m3",
    "mM": "mmol_per_m3", "mmol_per_m3": "mmol_per_m3",
    "millimole_per_cubic_metre": "mmol_per_m3",
    # Volume
    "umeter**3": "um3", "um^3": "um3", "um**3": "um3", "um3": "um3",
    "cubic_micrometre": "um3",
    # Length
    "m": "m", "metre": "m",
    "mm": "mm", "millimetre": "mm",
    "cm": "cm", "centimetre": "cm",
    # Velocity
    "m/s": "m_per_s", "m_per_s": "m_per_s",
    "mm/ms": "mm_per_ms", "mm_per_ms": "mm_per_ms",
    # Gain / compound
    "Hz/nA": "Hz_per_nA", "Hz_per_nA": "Hz_per_nA",
    "n/C": "per_nC",
    # Conductivity / permeability
    "S/m": "S_per_m", "S_per_m": "S_per_m",
    "H/m": "H_per_m", "H_per_m": "H_per_m",
    # Angular
    "rad": "rad", "radian": "rad",
    "rad/s": "rad_per_s", "rad_per_s": "rad_per_s", "radian_per_second": "rad_per_s",
    "rad/ms": "rad_per_ms", "rad_per_ms": "rad_per_ms",
    "radian_per_millisecond": "rad_per_ms",
    # Mass
    "kg": "kg", "kilogram": "kg",
    "kg/s": "kg_per_s", "kg_per_s": "kg_per_s",
    # Acceleration
    "m/s²": "m_per_s2", "m/s^2": "m_per_s2", "m/s**2": "m_per_s2", "m_per_s2": "m_per_s2",
    # Force / stiffness
    "N/m": "N_per_m", "N_per_m": "N_per_m",
    # Time squared
    "s²": "s2", "s^2": "s2", "s**2": "s2", "s2": "s2",
    # Per-unit (power systems)
    "p.u.": "per_unit", "per_unit": "per_unit", "pu": "per_unit",
    # Dimensionless
    "dimensionless": "dimensionless", "1": "dimensionless", "": "dimensionless",
    "unitless": "dimensionless",
    "%": "percent", "percent": "percent",
    "a.u.": "arbitrary_unit", "arbitrary_unit": "arbitrary_unit",
    "r_pearson": "dimensionless",
}


def normalize_unit(raw):
    """Convert a legacy free-text unit string to a UnitEnum value name.

    Accepts both abbreviations and full names.
    Returns ``None`` if the string cannot be mapped.

    >>> normalize_unit("mV")
    'mV'
    >>> normalize_unit("millivolt")
    'mV'
    >>> normalize_unit("ms^-1")
    'per_ms'
    >>> normalize_unit(None)
    """
    if raw is None:
        return None
    raw = str(raw).strip()
    if not raw:
        return None
    return _LEGACY_TO_ENUM.get(raw)


# ── UnitEnum → display symbol ────────────────────────────────────────

# For compound underscore forms, map to the conventional notation
# Uses SymPy-parseable symbols (no unicode) so equations render in any context
_DISPLAY_SYMBOLS = {
    "per_s": "1/s", "per_ms": "1/ms",
    "per_mV": "1/mV",
    "mV_per_ms": "mV/ms", "mV_per_s": "mV/s",
    "per_nC": "1/nC", "per_pC": "1/pC",
    "mol_per_m3": "mol/m^3", "mmol_per_m3": "mmol/m^3",
    "um3": "um^3",
    "m_per_s": "m/s", "mm_per_ms": "mm/ms",
    "Hz_per_nA": "Hz/nA",
    "S_per_m": "S/m", "H_per_m": "H/m",
    "rad_per_ms": "rad/ms", "rad_per_s": "rad/s",
    "kg_per_s": "kg/s",
    "m_per_s2": "m/s²", "N_per_m": "N/m", "s2": "s²",
    "per_unit": "p.u.",
    "dimensionless": "", "percent": "%", "arbitrary_unit": "a.u.",
}


def unit_to_symbol(unit):
    """Return the conventional display symbol for a UnitEnum value.

    For simple abbreviations (ms, mV, nA), the enum value IS the symbol.
    For compound forms (per_ms, mV_per_ms), returns conventional notation.

    >>> unit_to_symbol("ms")
    'ms'
    >>> unit_to_symbol("per_ms")
    '1/ms'
    >>> unit_to_symbol(None)
    ''
    """
    if unit is None:
        return ""
    # PermissibleValue (from getattr(UnitEnum, name)) has .text
    if hasattr(unit, "text"):
        key = unit.text
    else:
        key = str(unit)
    return _DISPLAY_SYMBOLS.get(key, key)
