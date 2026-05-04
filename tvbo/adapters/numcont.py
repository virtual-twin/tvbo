# -*- coding: utf-8 -*-
"""Self-contained AUTO-07p (numcont) backend adapter for SimulationExperiment.

This adapter does NOT depend on any external `numcont` package. It uses
the `auto-07p` Python bindings directly (`auto.run`, `auto.sv`,
`auto.loadbd`, `auto.merge`) and the Mako template at
``tvbo/templates/numcont/tvbo-auto7p.py.mako`` to emit the model `.f90`
file consumed by AUTO.

Requires the ``AUTO_DIR`` environment variable to point at an installed
auto-07p tree (validated via :func:`tvbo.utils.auto.check_auto_dir`).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from tvbo.analysis.bifurcation import BifurcationResult
    from tvbo.classes.experiment import SimulationExperiment


# AUTO reserves PAR(11)=PERIOD and PAR(12)=ANGLE; user params occupy
# 1..10 then 13..NPAR.
_RESERVED_LO = 11
_RESERVED_HI = 12


def _auto_par_index(i: int) -> int:
    """Map zero-based user-parameter index to AUTO PAR(.) index."""
    return i + 1 if i + 1 <= 10 else i + 3


def _build_parnames(model) -> dict[int, str]:
    """Build AUTO parnames dict, matching the f90 template's PAR layout."""
    pn = {}
    for i, p in enumerate(model.parameters.values()):
        pn[_auto_par_index(i)] = p.name
    pn[_RESERVED_LO] = "PERIOD"
    pn[_RESERVED_HI] = "ANGLE"
    return pn


def _build_unames(model) -> dict[int, str]:
    return {i + 1: sv.name for i, sv in enumerate(model.state_variables.values())}


def _schema_initial_values(model) -> np.ndarray:
    """Initial state from model state-variable defaults (`.value`) or zeros."""
    n = len(model.state_variables)
    x0 = np.zeros(n)
    for i, sv in enumerate(model.state_variables.values()):
        if getattr(sv, "value", None) is not None:
            x0[i] = float(sv.value)
    return x0


def _warmup_to_steady_state(model, cont) -> np.ndarray:
    """Time-integrate the dynamics to produce a starting state for AUTO.

    Honors ``cont.initial_state.method`` (default ``time_integration``):

    - ``time_integration`` (default): run ``Dynamics.run(format='python')``
      for ``initial_state.duration`` time units and return the final state.
    - ``given``: return the schema defaults verbatim (no integration).
    - other methods: fall back to schema defaults (not yet implemented).
    """
    x0 = _schema_initial_values(model)

    iss = getattr(cont, "initial_state", None) if cont else None
    method = getattr(iss, "method", None)
    method_str = str(method) if method is not None else "time_integration"

    if method_str == "given":
        return x0

    duration = float(getattr(iss, "duration", None) or 2000.0)
    ts = model.run(format="python", u_0=x0, duration=duration, save=False, verbose=0)
    # TimeSeries data is (T, n_sv, 1, 1)
    return np.asarray(ts.data[-1, :, 0, 0], dtype=float)


def _initial_state_dict(model, cont) -> dict[int, float]:
    """Build the U={1: x0, ...} initial-condition dict for AUTO."""
    x0 = _warmup_to_steady_state(model, cont)
    return {i + 1: float(x0[i]) for i in range(len(x0))}


def _param_values(model) -> dict[str, float]:
    """Default parameter values keyed by name (skips None)."""
    out = {}
    for p in model.parameters.values():
        if p.value is not None:
            out[p.name] = float(p.value)
    return out


def _free_parameter(cont, model):
    """Return (name, p_min, p_max) for the primary free parameter."""
    fp_dict = getattr(cont, "free_parameters", None) if cont else None
    if fp_dict:
        fp_first = next(iter(fp_dict.values())) if isinstance(fp_dict, dict) else fp_dict[0]
        name = str(fp_first.name)
        dom = fp_first.domain or model.parameters[name].domain
    else:
        name = next(iter(model.parameters.keys()))
        dom = model.parameters[name].domain
    p_min = float(dom.lo) if dom and dom.lo is not None else -10.0
    p_max = float(dom.hi) if dom and dom.hi is not None else 10.0
    return name, p_min, p_max


def _cont_par(cont, key, default=None):
    """Look up named scalar in cont.parameters."""
    if cont is None:
        return default
    params = getattr(cont, "parameters", None)
    if not params:
        return default
    p = params.get(key) if isinstance(params, dict) else None
    if p and getattr(p, "value", None) is not None:
        return p.value
    return default


class NumContAdapter:
    """Adapter for bifurcation analysis via AUTO-07p (no external deps)."""

    def __init__(self, experiment: "SimulationExperiment"):
        self.experiment = experiment

    # ── Context for the f90 template ─────────────────────────────────────

    @staticmethod
    def _prepare_context(model, cont, **kwargs) -> dict:
        return dict(model=model, continuation=cont)

    def render_code(self, model=None, continuation=None, **kwargs) -> str:
        """Render the AUTO-07p Fortran (.f90) model source as a string."""
        from tvbo import templates

        model = model or self.experiment.dynamics
        if continuation is None:
            conts = getattr(self.experiment, "continuations", None) or {}
            if conts:
                continuation = next(iter(conts.values()))
        ctx = self._prepare_context(model, continuation, **kwargs)
        template = templates.lookup.get_template("tvbo-auto7p.py.mako")
        return template.render(**ctx)

    # ── Public API ───────────────────────────────────────────────────────

    def run(self, **kwargs):
        """Run AUTO-07p continuation for each continuation in the experiment."""
        from tvbo.utils.auto import check_auto_dir

        check_auto_dir()

        exp = self.experiment
        conts = getattr(exp, "continuations", None) or {}
        if not conts:
            raise ValueError(
                "No continuations defined. Add continuation specs via "
                "exp.continuations or load from a bifurcation YAML."
            )

        results = {}
        for name, cont in conts.items():
            model = self._resolve_dynamics(cont)
            results[name] = self._run_one(model, cont, name, **kwargs)

        if len(results) == 1:
            return next(iter(results.values()))
        return results

    # ── Internals ────────────────────────────────────────────────────────

    def _resolve_dynamics(self, cont):
        exp = self.experiment
        dyn_ref = getattr(cont, "dynamics", None)
        if dyn_ref:
            dyn_name = str(dyn_ref)
            if exp.dynamics and getattr(exp.dynamics, "name", None) == dyn_name:
                return exp.dynamics
            net_dyn = getattr(exp.network, "dynamics", None) if exp.network else None
            if isinstance(net_dyn, dict) and dyn_name in net_dyn:
                return net_dyn[dyn_name]
        if exp.dynamics is not None:
            return exp.dynamics
        raise ValueError(f"Cannot resolve dynamics for continuation '{cont}'.")

    def _run_one(self, model, cont, cont_name, **kwargs):
        import contextlib
        import io
        # AUTO-07p prints a Tkinter import warning to stdout even when
        # plotting is unused. Suppress it.
        _buf = io.StringIO()
        with contextlib.redirect_stdout(_buf):
            import auto

        from tvbo.analysis.bifurcation import BifurcationResult

        # 1. Render f90 to a temporary working dir
        workdir = tempfile.mkdtemp(prefix=f"tvbo_numcont_{model.name}_")
        f90_path = os.path.join(workdir, "model.f90")
        with open(f90_path, "w") as fh:
            fh.write(self.render_code(model=model, continuation=cont))

        # 2. Build common AUTO arguments
        parnames = _build_parnames(model)
        unames = _build_unames(model)
        ndim = len(model.state_variables)
        npar = max(_auto_par_index(len(model.parameters) - 1), 12) if model.parameters else 12
        par_vals = _param_values(model)
        u0 = _initial_state_dict(model, cont)
        fp_name, p_min, p_max = _free_parameter(cont, model)

        # AUTO needs the source file path WITHOUT the .f90 extension
        source_stub = os.path.join(workdir, "model")

        # 3. Equilibrium (1-parameter) continuation
        kwargs_eq = dict(
            e=source_stub,
            parnames=parnames,
            unames=unames,
            NDIM=ndim,
            NPAR=npar,
            PAR=par_vals,
            U=u0,
            ICP=[fp_name],
            IPS=1,
            ISP=2,
            ISW=1,
            ILP=1,
            IADS=int(_cont_par(cont, "IADS", 1)),
            IAD=1,
            IID=0,
            EPSL=float(_cont_par(cont, "EPSL", 1e-7)),
            EPSU=float(_cont_par(cont, "EPSU", 1e-7)),
            EPSS=float(_cont_par(cont, "EPSS", 1e-5)),
            RL0=p_min,
            RL1=p_max,
            NMX=int(_cont_par(cont, "NMX", 1000)),
            NPR=int(_cont_par(cont, "NPR", 10)),
            DS=float(_cont_par(cont, "DS", 0.005)),
            MXBF=int(_cont_par(cont, "MXBF", 50)),
        )

        cwd0 = os.getcwd()
        os.chdir(workdir)
        try:
            R_eq = auto.run(**kwargs_eq)
            # Bidirectional sweep: also continue with DS<0 and merge
            if bool(getattr(cont, "bothside", False)):
                kwargs_eq_neg = dict(kwargs_eq, DS=-kwargs_eq["DS"])
                R_eq_neg = auto.run(**kwargs_eq_neg)
                R_eq = auto.merge(R_eq + R_eq_neg)
            auto.sv(R_eq, cont_name)

            # 4. Periodic-orbit continuation from each Hopf point
            po_results = []
            for i, br in enumerate(R_eq):
                hbs = br.labels.by_label.get("HB", {})
                n_hb = len(hbs) if hbs else 0
                for k in range(n_hb):
                    kwargs_po = dict(
                        data=R_eq(f"HB{k + 1}"),
                        EPSL=kwargs_eq["EPSL"],
                        EPSU=kwargs_eq["EPSU"],
                        EPSS=kwargs_eq["EPSS"],
                        IPS=2,
                        ISP=2,
                        ISW=1,
                        ICP=[fp_name, "PERIOD"],
                        RL0=p_min,
                        RL1=p_max,
                        NMX=int(_cont_par(cont, "NMX_PO", 400)),
                        NPR=1,
                        DS=float(_cont_par(cont, "DS_PO", 0.01)),
                        IADS=0,
                        MXBF=int(_cont_par(cont, "MXBF", 50)),
                        IID=0,
                    )
                    R_po = auto.run(**kwargs_po)
                    po_name = f"{cont_name}_HB{k}"
                    auto.sv(R_po, po_name)
                    po_results.append((po_name, R_po))
        finally:
            os.chdir(cwd0)

        return BifurcationResult.from_auto(
            R_eq,
            cont_name=cont_name,
            model=model,
            continuation=cont,
            ICS=fp_name,
            periodic_orbits_raw=po_results,
            workdir=workdir,
        )

    # ── Cleanup ──────────────────────────────────────────────────────────

    @staticmethod
    def cleanup(result):
        """Remove the temporary working directory of a NumCont result."""
        wd = getattr(result, "workdir", None)
        if wd and os.path.isdir(wd):
            shutil.rmtree(wd, ignore_errors=True)
