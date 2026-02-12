# -*- coding: utf-8 -*-
"""PyRates/PyCoBi bifurcation analysis backend adapter for SimulationExperiment.

Uses PyRates to generate Fortran code for AUTO-07p, and PyCoBi as the
Python interface to run parameter continuations and detect bifurcations.

Reuses the same ``Continuation`` schema as the BifurcationKit.jl backend,
so ``exp.run("pyrates-bifurcation")`` and ``exp.run("bifurcationkit.jl")``
accept the same YAML specification.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tvbo.analysis.bifurcation import BifurcationResult
    from tvbo.export.experiment import SimulationExperiment

# Same reserved-name mapping used by the PyRates YAML template
PYRATES_REPL = {
    "I": "I_",
    "gamma": "gamma_",
    "beta": "beta_",
    "zeta": "zeta_",
    "lambda": "lambda_",
    "E": "E_",
    "N": "N_",
    "S": "S_",
    "O": "O_",
    "Q": "Q_",
    "epsilon": "epsilon_",
}


def _pyrates_param_name(name):
    """Apply the same renaming as the PyRates YAML template."""
    return PYRATES_REPL.get(name, name)


class PyRatesBifurcationAdapter:
    """Adapter for running bifurcation analysis via PyRates + PyCoBi (AUTO-07p).

    Like ``BifurcationKitAdapter``, this does not inherit from ``BaseAdapter`` —
    bifurcation analysis operates on individual (Dynamics, Continuation) pairs.
    """

    def __init__(self, experiment: "SimulationExperiment"):
        self.experiment = experiment

    # ── Public API ───────────────────────────────────────────────────────

    def run(self, **kwargs) -> "BifurcationResult | dict[str, BifurcationResult]":
        """Run bifurcation analysis for each continuation in the experiment.

        Returns
        -------
        BifurcationResult or dict[str, BifurcationResult]
            Single result if one continuation, dict if multiple.
        """
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
            results[name] = self._run_single(model, cont, **kwargs)

        if len(results) == 1:
            return next(iter(results.values()))
        return results

    def render_code(self, model=None, continuation=None, **kwargs) -> str:
        """Render Python code for the PyRates/PyCoBi bifurcation workflow.

        Parameters
        ----------
        model : Dynamics, optional
            The dynamics model. Defaults to ``experiment.local_dynamics``.
        continuation : Continuation, optional
            The continuation spec. Defaults to first in experiment.

        Returns
        -------
        str
            Executable Python code string.
        """
        model = model or self.experiment.local_dynamics
        if continuation is None:
            conts = getattr(self.experiment, "continuations", None) or {}
            if conts:
                continuation = next(iter(conts.values()))

        fp = self._get_free_parameter(continuation, model)
        fp_name = fp["name"]
        p_min, p_max = fp["p_min"], fp["p_max"]
        param_order = self._get_param_order(model)
        icp = param_order.index(fp_name) + 1
        auto_kwargs = self._cont_to_auto_kwargs(
            continuation, icp, p_min, p_max
        )

        iss_duration = 2000.0
        if continuation and continuation.initial_state:
            d = getattr(continuation.initial_state, "duration", None)
            if d is not None:
                iss_duration = float(d)

        sv_names = list(model.state_variables.keys())

        code = f'''\
import os, shutil, sys, tempfile, uuid
from pycobi import ODESystem
from pyrates import clear
from pyrates.frontend import CircuitTemplate
from tvbo.export.pyrates import to_pyrates_yaml_string
from tvbo import Dynamics

# Load model
model = Dynamics.from_ontology("{model.name}")

# Export to PyRates YAML
yaml_content = to_pyrates_yaml_string(model)
tmpdir = tempfile.mkdtemp(prefix="tvbo_pyrates_bif_")
pkg_name = f"_tvbo_prbif_{{uuid.uuid4().hex[:8]}}"
pkg_path = os.path.join(tmpdir, pkg_name)
os.makedirs(pkg_path, exist_ok=True)
open(os.path.join(pkg_path, "__init__.py"), "w").close()
with open(os.path.join(pkg_path, "model.yaml"), "w") as f:
    f.write(yaml_content)
sys.path.insert(0, tmpdir)

# Load circuit
circuit = CircuitTemplate.from_yaml(f"{{pkg_name}}.model.{model.name}_circuit")

# Generate Fortran + AUTO constants
circuit.get_run_func(
    func_name="tvbo_rhs", file_name="tvbo_bif",
    step_size=1e-4, auto=True, backend="fortran",
    solver="scipy", vectorize=False, float_precision="float64",
)
clear(circuit)

# Create PyCoBi ODESystem
ode = ODESystem(eq_file="tvbo_bif", working_dir=None, init_cont=False)

# Time continuation to find equilibrium
t_sols, t_cont = ode.run(
    c="ivp", name="time",
    DS=1e-4, DSMIN=1e-10, DSMAX=1e-2,
    EPSL=1e-08, EPSU=1e-08, EPSS=1e-06,
    NMX=10000,
    UZR={{14: {iss_duration / 1000.0}}},
    STOP={{"UZ1"}},
)

# Parameter continuation: {fp_name} (ICP={icp})
p_sols, p_cont = ode.run(
    origin=t_cont, starting_point="UZ1", name="param",
    {self._format_auto_kwargs(auto_kwargs)}
)

# Plot bifurcation diagram
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, {len(sv_names)}, figsize=({5 * len(sv_names)}, 5))
if {len(sv_names)} == 1:
    axes = [axes]
for i, sv in enumerate({sv_names!r}):
    ode.plot_continuation("PAR({icp})", f"U({{i+1}})", cont="param", ax=axes[i])
    axes[i].set_xlabel("{fp_name}")
    axes[i].set_ylabel(sv)
    axes[i].set_title(f"Bifurcation: {{sv}} vs {fp_name}")
plt.tight_layout()
plt.show()

# Cleanup
shutil.rmtree(tmpdir, ignore_errors=True)
for f in ["tvbo_bif.f90", "c.ivp"]:
    if os.path.exists(f):
        os.remove(f)
'''
        return code

    # ── Core workflow ────────────────────────────────────────────────────

    def _run_single(self, model, cont, **kwargs):
        """Run a single continuation analysis."""
        from pycobi import ODESystem
        from pyrates import clear
        from pyrates.frontend import CircuitTemplate

        from tvbo.analysis.bifurcation import PyRatesBifurcationResult

        # Resolve free parameter
        fp = self._get_free_parameter(cont, model)
        fp_name = fp["name"]
        p_min, p_max = fp["p_min"], fp["p_max"]

        # Build circuit from TVBO model
        circuit, tmpdir, pkg_name = self._load_circuit(model)

        try:
            # Get parameter ordering from PyRates to find AUTO-07p ICP index
            param_order = self._get_param_order(model)
            icp = param_order.index(fp_name) + 1  # AUTO uses 1-based

            # Generate Fortran + AUTO constants
            eq_file = "tvbo_bif"
            circuit.get_run_func(
                func_name="tvbo_rhs",
                file_name=eq_file,
                step_size=1e-4,
                auto=True,
                backend="fortran",
                solver="scipy",
                vectorize=False,
                float_precision="float64",
            )

            # Create PyCoBi ODESystem
            ode = ODESystem(eq_file=eq_file, working_dir=None, init_cont=False)

            # Step 1: Time continuation to find equilibrium
            iss_duration = float(getattr(cont.initial_state, "duration", None)
                                 or 2000.0) if cont.initial_state else 2000.0
            t_sols, t_cont = ode.run(
                c="ivp", name="time",
                DS=1e-4, DSMIN=1e-10, DSMAX=1e-2,
                EPSL=1e-08, EPSU=1e-08, EPSS=1e-06,
                NMX=10000,
                UZR={14: iss_duration / 1000.0},  # PAR(14) = time
                STOP={"UZ1"},
            )

            # Step 2: Parameter continuation from equilibrium
            auto_kwargs = self._cont_to_auto_kwargs(cont, icp, p_min, p_max)
            p_sols, p_cont = ode.run(
                origin=t_cont,
                starting_point="UZ1",
                name="param",
                **auto_kwargs,
            )

            # Step 3: Run branch continuations (periodic orbits from Hopf)
            po_results = []
            if cont.branches:
                branches = (
                    list(cont.branches.values())
                    if isinstance(cont.branches, dict)
                    else list(cont.branches)
                )
                for branch in branches:
                    po_res = self._run_branch(
                        ode, p_cont, branch, cont, icp, p_min, p_max
                    )
                    po_results.extend(po_res)

            # Build result
            state_var_names = list(model.state_variables.keys())
            result = PyRatesBifurcationResult(
                ode=ode,
                cont_name="param",
                model=model,
                state_var_names=state_var_names,
                icp=icp,
                fp_name=fp_name,
                periodic_orbit_results=po_results,
            )

            clear(circuit)

        finally:
            # Cleanup temp package (but not working dir — AUTO needs it during run)
            if tmpdir in sys.path:
                sys.path.remove(tmpdir)
            modules_to_remove = [k for k in sys.modules if k.startswith(pkg_name)]
            for mod in modules_to_remove:
                del sys.modules[mod]
            shutil.rmtree(tmpdir, ignore_errors=True)

            # Cleanup AUTO-07p generated files
            for f in [f"{eq_file}.f90", "c.ivp"]:
                if os.path.exists(f):
                    os.remove(f)

        return result

    def _run_branch(self, ode, p_cont, branch, cont, icp, p_min, p_max):
        """Run a branch continuation (e.g., periodic orbits from Hopf)."""
        source = getattr(branch, "source_point", "hopf:all") or "hopf:all"
        all_hopf = source == "hopf:all"

        # Get Hopf points from the continuation
        hopf_points = self._find_special_points(ode, "param", "HB")
        if not hopf_points:
            return []

        if all_hopf:
            indices = list(range(len(hopf_points)))
        else:
            idx_str = source.split(":")[1] if ":" in source else "-1"
            if idx_str.lstrip("-").isdigit():
                idx = int(idx_str)
                indices = [idx if idx >= 0 else len(hopf_points) + idx]
            else:
                indices = list(range(len(hopf_points)))

        bc = getattr(branch, "continuation", None)
        po_results = []

        for i in indices:
            if i < 0 or i >= len(hopf_points):
                continue
            hp_label = hopf_points[i]

            try:
                po_kwargs = self._cont_to_auto_kwargs(
                    bc or cont, icp, p_min, p_max, is_po=True
                )
                po_kwargs.setdefault("ISW", -1)  # Branch switching
                po_kwargs.setdefault("ISP", 2)
                po_kwargs.setdefault("IPS", 2)   # Periodic orbit
                po_kwargs.setdefault("NTST", 400)
                po_kwargs.setdefault("NCOL", 4)

                if branch.bothside:
                    po_kwargs["bidirectional"] = True

                po_sols, po_cont = ode.run(
                    origin=p_cont if isinstance(p_cont, str) else "param",
                    starting_point=hp_label,
                    name=f"po_from_{hp_label}",
                    **po_kwargs,
                )
                po_results.append((f"po_from_{hp_label}", po_cont))
            except Exception:
                # PO continuation may fail for some Hopf points
                pass

        return po_results

    def _find_special_points(self, ode, cont_name, sp_type):
        """Find special point labels from a PyCoBi continuation."""
        try:
            summary = ode.summary
            if not summary:
                return []
            points = []
            for key, info in summary.items():
                if cont_name in str(key):
                    sols = info if isinstance(info, dict) else {}
                    for sol_key, sol_info in sols.items():
                        if sp_type in str(sol_key):
                            points.append(str(sol_key))
            return points
        except Exception:
            return []

    # ── Helpers ──────────────────────────────────────────────────────────

    def _get_free_parameter(self, cont, model):
        """Extract the free parameter info from a Continuation spec."""
        fp_dict = cont.free_parameters if cont else None
        if fp_dict:
            fp_first = (
                next(iter(fp_dict.values()))
                if isinstance(fp_dict, dict)
                else fp_dict[0]
            )
            name = str(fp_first.name)
            if fp_first.domain:
                p_min = float(fp_first.domain.lo) if fp_first.domain.lo else -20
                p_max = float(fp_first.domain.hi) if fp_first.domain.hi else 20
            elif name in model.parameters and model.parameters[name].domain:
                dom = model.parameters[name].domain
                p_min = float(dom.lo or -20)
                p_max = float(dom.hi or 20)
            else:
                p_min, p_max = -20, 20
        else:
            raise ValueError("Continuation has no free_parameters defined.")

        return {"name": name, "p_min": p_min, "p_max": p_max}

    def _get_param_order(self, model):
        """Get parameter names in the order PyRates assigns AUTO-07p PAR indices.

        PyRates assigns PAR(1), PAR(2), ... to parameters in the order they
        appear in the OperatorTemplate variables dict (which follows the order
        in model.parameters). Reserved names are renamed (e.g. I → I_).
        """
        return [_pyrates_param_name(name) for name in model.parameters.keys()]

    def _load_circuit(self, model):
        """Load a PyRates CircuitTemplate from a TVBO Dynamics model."""
        from pyrates.frontend import CircuitTemplate

        from tvbo.export.pyrates import to_pyrates_yaml_string

        yaml_content = to_pyrates_yaml_string(model)

        tmpdir = tempfile.mkdtemp(prefix="tvbo_pyrates_bif_")
        pkg_name = f"_tvbo_prbif_{uuid.uuid4().hex[:8]}"
        pkg_path = os.path.join(tmpdir, pkg_name)
        os.makedirs(pkg_path, exist_ok=True)
        open(os.path.join(pkg_path, "__init__.py"), "w").close()

        yaml_path = os.path.join(pkg_path, "model.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        sys.path.insert(0, tmpdir)

        model_name = getattr(model, "name", None) or "tvbo_model"
        circuit_name = f"{model_name}_circuit"
        circuit = CircuitTemplate.from_yaml(f"{pkg_name}.model.{circuit_name}")
        return circuit, tmpdir, pkg_name

    def _cont_to_auto_kwargs(self, cont, icp, p_min, p_max, is_po=False):
        """Convert Continuation schema fields to AUTO-07p keyword arguments."""
        kw = {}
        kw["ICP"] = icp
        kw["RL0"] = p_min
        kw["RL1"] = p_max
        kw["bidirectional"] = bool(getattr(cont, "bothside", False))

        if not is_po:
            kw["IPS"] = 1   # Equilibrium continuation
            kw["ILP"] = 1   # Detect fold bifurcations
            kw["ISP"] = 2   # Full automatic bifurcation detection
            kw["ISW"] = 1   # Normal continuation

        # Step size
        ds = getattr(cont, "ds", None)
        if ds is not None:
            kw["DS"] = float(ds)
        else:
            kw["DS"] = 1e-4

        ds_min = getattr(cont, "ds_min", None)
        if ds_min is not None:
            kw["DSMIN"] = float(ds_min)
        else:
            kw["DSMIN"] = 1e-8

        ds_max = getattr(cont, "ds_max", None)
        if ds_max is not None:
            kw["DSMAX"] = float(ds_max)

        # Max steps
        max_steps = getattr(cont, "max_steps", None)
        if max_steps is not None:
            kw["NMX"] = int(max_steps)
        else:
            kw["NMX"] = 2000

        # Tolerances
        tol = getattr(cont, "tol_stability", None)
        if tol is not None:
            kw["EPSS"] = float(tol)

        newton_tol = getattr(cont, "newton_tol", None)
        if newton_tol is not None:
            kw["EPSL"] = float(newton_tol)
            kw["EPSU"] = float(newton_tol)

        newton_max = getattr(cont, "newton_max_iterations", None)
        if newton_max is not None:
            kw["ITMX"] = int(newton_max)
            kw["ITNW"] = int(newton_max)

        # Additional standard settings
        kw.setdefault("NTST", 400)
        kw.setdefault("NCOL", 4)
        kw.setdefault("IAD", 3)
        kw.setdefault("IADS", 1)
        kw.setdefault("NPR", 10)

        return kw

    def _resolve_dynamics(self, cont):
        """Resolve the Dynamics model for a continuation spec."""
        exp = self.experiment
        dyn_ref = getattr(cont, "dynamics", None)
        if dyn_ref and str(dyn_ref) in exp.dynamics:
            return exp.dynamics[str(dyn_ref)]
        if exp.local_dynamics is not None:
            return exp.local_dynamics
        raise ValueError(
            f"Cannot resolve dynamics for continuation. "
            f"dynamics='{dyn_ref}' not found in exp.dynamics."
        )

    @staticmethod
    def _format_auto_kwargs(kw):
        """Format AUTO kwargs dict as a string for rendered code."""
        lines = []
        for k, v in kw.items():
            if isinstance(v, bool):
                lines.append(f"{k}={v}")
            elif isinstance(v, str):
                lines.append(f'{k}="{v}"')
            else:
                lines.append(f"{k}={v}")
        return ",\n    ".join(lines)
