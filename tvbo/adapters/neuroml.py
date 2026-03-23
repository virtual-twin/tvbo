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
    # If ANY parameter in the model carries a time-bearing unit, the
    # equations already contain physical time normalisation (e.g., /tau_e
    # where tau_e has unit=ms).  In that case / SEC would double-count.
    # When the model is fully dimensionless, / SEC is required to give the
    # LEMS TimeDerivative value the mandatory per_time dimension.
    from tvbo.utils.units import unit_has_time_dimension, unit_to_lems_dimension
    _model_has_time_units = any(
        unit_has_time_dimension(getattr(p, "unit", None))
        for p in list(params.values()) + list(svs.values())
    )

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

    return dict(
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
    )


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

        time = raw[:, 0]
        values = raw[:, 1:]  # (n_t, n_sv)

        da = xr.DataArray(
            data=values.reshape(-1, len(sv_names), 1),
            dims=['time', 'variable', 'node'],
            coords={
                'time': time,
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
