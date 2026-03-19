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

# ── NeuroML dimension / unit mapping ─────────────────────────────────

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

def sympy_to_lems(expr_str):
    """Convert a Python/SymPy expression string to LEMS syntax."""
    if not expr_str:
        return ""
    s = str(expr_str)
    s = s.replace("**", "^")
    s = re.sub(r'\band\b', '.and.', s)
    s = re.sub(r'\bor\b', '.or.', s)
    s = re.sub(r'\bnot\b', '.not.', s)
    return s


# ── Helpers ──────────────────────────────────────────────────────────

def safe_id(s):
    """Make a string safe for XML id attribute."""
    s = re.sub(r"[^a-zA-Z0-9_]", "_", str(s or "id0"))
    return ("_" + s) if s[0].isdigit() else s


def validate_lems_xml(xml_string):
    """Validate a LEMS XML string using PyLEMS.

    Raises if the XML is not valid LEMS.
    """
    from lems.model.model import Model
    m = Model()
    m.import_from_string(xml_string)


# ── Adapter ──────────────────────────────────────────────────────────

class NeuroMLAdapter(BaseAdapter):
    """Adapter for exporting SimulationExperiment as LEMS XML.

    Uses Pattern A: the Mako template receives ``experiment`` directly
    and reads all metadata from the SimulationExperiment object.
    """

    TEMPLATE = "neuroml/tvbo-neuroml-lems.xml.mako"

    def __init__(self, source=None):
        from tvbo.classes.dynamics import Dynamics
        from tvbo.classes.experiment import SimulationExperiment

        if source is None:
            self.experiment = None
            return
        if isinstance(source, Dynamics):
            source = SimulationExperiment(dynamics=source)
        super().__init__(source)

    def render_code(self, **kwargs) -> str:
        """Render LEMS XML from the experiment."""
        from tvbo import templates
        template = templates.lookup.get_template(self.TEMPLATE)
        return template.render(experiment=self.experiment, **kwargs)

    def validate(self, xml_string=None):
        """Validate the rendered LEMS XML."""
        if xml_string is None:
            xml_string = self.render_code()
        validate_lems_xml(xml_string)
        return True

    def run(self, **kwargs):
        """Run the LEMS simulation via pyNeuroML (jnml)."""
        import tempfile
        from pathlib import Path

        from pyneuroml import pynml

        xml = self.render_code(**kwargs)
        with tempfile.TemporaryDirectory() as tmpdir:
            lems_file = Path(tmpdir) / "simulation.xml"
            lems_file.write_text(xml)
            pynml.run_lems_with_jneuroml(
                str(lems_file), nogui=True, load_saved_data=False
            )
