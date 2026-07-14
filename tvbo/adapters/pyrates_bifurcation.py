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
import re
import shutil
import sys
import tempfile
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tvbo.analysis.bifurcation import BifurcationResult
    from tvbo.classes.experiment import SimulationExperiment

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
                "No continuations defined. Add continuation specs via exp.continuations or load from a bifurcation YAML."
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
            The dynamics model. Defaults to ``experiment.dynamics``.
        continuation : Continuation, optional
            The continuation spec. Defaults to first in experiment.

        Returns
        -------
        str
            Executable Python code string.
        """
        model = model or self.experiment.dynamics
        if continuation is None:
            conts = getattr(self.experiment, "continuations", None) or {}
            if conts:
                continuation = next(iter(conts.values()))

        fp = self._get_free_parameter(continuation, model)
        fp_name = fp["name"]
        p_min, p_max = fp["p_min"], fp["p_max"]
        pyrates_fp_name = _pyrates_param_name(fp_name)
        auto_kwargs = self._cont_to_auto_kwargs(continuation, pyrates_fp_name, p_min, p_max)

        iss_duration = 10000.0
        if continuation and continuation.initial_state:
            d = getattr(continuation.initial_state, "duration", None)
            if d is not None:
                iss_duration = float(d)

        sv_names = list(model.state_variables.keys())

        code = f'''\
import os, re, shutil, sys, tempfile, uuid
from pycobi import ODESystem
from pyrates import clear
from pyrates.frontend import CircuitTemplate
from tvbo.codegen.pyrates import to_pyrates_yaml_string
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

# Create PyCoBi ODESystem. PyRates emits parnames/unames, so PyCoBi keys
# solutions by variable name — populate only the inverse map, plus a
# name->PAR-index dict for numeric ICP resolution.
ode = ODESystem(eq_file="tvbo_bif", working_dir=None, init_cont=False)
param_idx = {{}}
with open("tvbo_bif.f90") as _f90:
    for m in re.finditer(r"args\\((\\d+)\\)\\s*=\\s*[^!]*!\\s*(\\S+)", _f90.read()):
        idx, name = int(m.group(1)), m.group(2)
        param_idx[name] = idx
        ode._var_map_inv[f"PAR({{idx}})"] = name

# Look up numeric ICP
icp = param_idx["{pyrates_fp_name}"]

# Time continuation to find equilibrium
t_sols, t_cont = ode.run(
    c="ivp", name="time",
    DS=1e-4, DSMIN=1e-10, DSMAX=1.0,
    EPSL=1e-08, EPSU=1e-08, EPSS=1e-06,
    NMX=50000,
    UZR={{14: {iss_duration}}},
    STOP={{"UZ1"}},
)

# Parameter continuation: {fp_name} (mapped to {pyrates_fp_name})
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
    ode.plot_continuation(f"PAR({{icp}})", sv, cont="param", ax=axes[i])
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

        from tvbo.analysis.bifurcation import BifurcationResult

        # Resolve free parameter
        fp = self._get_free_parameter(cont, model)
        fp_name = fp["name"]
        p_min, p_max = fp["p_min"], fp["p_max"]
        pyrates_fp_name = _pyrates_param_name(fp_name)

        # Build circuit from TVBO model
        circuit, tmpdir, pkg_name = self._load_circuit(model)

        try:
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

            # Create PyCoBi ODESystem and populate its parameter mapping
            # from the generated Fortran file (PyCoBi doesn't do this
            # when constructed via eq_file= without params=)
            ode = ODESystem(eq_file=eq_file, working_dir=None, init_cont=False)
            state_var_names = list(model.state_variables.keys())
            param_idx = self._populate_var_map(ode, eq_file, state_var_names)

            # Guard: PyCoBi's _create_summary() crashes on NDIM=1 systems
            # (KeyError: 'U(1)'). PyRates + AUTO-07p both handle 1-D scalar
            # ODEs correctly; the bug is in PyCoBi's summary builder.
            # See https://github.com/pyrates-neuroscience/PyCoBi
            if len(state_var_names) < 2:
                raise NotImplementedError(
                    f"The 'pyrates-bifurcation' backend cannot continue the "
                    f"1-D system '{getattr(model, 'name', '?')}' "
                    f"(state variables: {state_var_names}). "
                    f"PyCoBi's summary builder requires NDIM >= 2 and raises "
                    f"KeyError('U(1)') for scalar ODEs. "
                    f"For 1-D systems use the 'auto-07p' or "
                    f"'bifurcationkit.jl' backend instead."
                )

            # Numeric PAR index for the free parameter (for DataFrame extraction).
            icp = param_idx.get(pyrates_fp_name, pyrates_fp_name)

            # Step 1: Time continuation to find equilibrium
            # PAR(14) = time in model units (AUTO has no unit system)
            iss_duration = float(getattr(cont.initial_state, "duration", None) or 10000.0) if cont.initial_state else 10000.0
            t_sols, t_cont = ode.run(
                c="ivp",
                name="time",
                DS=1e-4,
                DSMIN=1e-10,
                DSMAX=1.0,
                EPSL=1e-08,
                EPSU=1e-08,
                EPSS=1e-06,
                NMX=50000,
                UZR={14: iss_duration},
                STOP={"UZ1"},
            )

            # Step 2: Parameter continuation from equilibrium
            # Pass the parameter name — PyCoBi maps it via _var_map
            auto_kwargs = self._cont_to_auto_kwargs(cont, pyrates_fp_name, p_min, p_max)
            p_sols, p_cont = ode.run(
                origin=t_cont,
                starting_point="UZ1",
                name="param",
                **auto_kwargs,
            )

            # Step 3: Run branch continuations (periodic orbits / codim-2)
            po_results = []
            codim2_results = []
            if cont.branches:
                branches = list(cont.branches.values()) if isinstance(cont.branches, dict) else list(cont.branches)
                for branch in branches:
                    # Detect codim-2 branches (have sub-continuation with free_parameters)
                    bc = getattr(branch, "continuation", None)
                    has_fp2 = bc and getattr(bc, "free_parameters", None)
                    if has_fp2:
                        c2_res = self._run_codim2_branch(
                            ode,
                            p_cont,
                            branch,
                            cont,
                            pyrates_fp_name,
                            p_min,
                            p_max,
                            state_var_names=state_var_names,
                            icp=icp,
                            fp_name=fp_name,
                            param_idx=param_idx,
                        )
                        codim2_results.extend(c2_res)
                    else:
                        po_res = self._run_branch(
                            ode,
                            p_cont,
                            branch,
                            cont,
                            pyrates_fp_name,
                            p_min,
                            p_max,
                        )
                        po_results.extend(po_res)

            # Build result (icp is the numeric index for DataFrame extraction)
            state_var_names = list(model.state_variables.keys())
            result = BifurcationResult.from_pycobi(
                ode=ode,
                cont_name="param",
                model=model,
                state_var_names=state_var_names,
                icp=icp,
                fp_name=fp_name,
                periodic_orbit_results=po_results,
                codim2_results=codim2_results,
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

    def _run_branch(self, ode, p_cont, branch, cont, icp_name, p_min, p_max):
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
                po_kwargs = self._cont_to_auto_kwargs(bc or cont, icp_name, p_min, p_max, is_po=True)
                po_kwargs.setdefault("ISW", -1)  # Branch switching
                po_kwargs.setdefault("ISP", 2)
                po_kwargs.setdefault("IPS", 2)  # Periodic orbit
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

    def _run_codim2_branch(
        self,
        ode,
        p_cont,
        branch,
        cont,
        icp_name,
        p_min,
        p_max,
        state_var_names=None,
        icp=1,
        fp_name="param",
        param_idx=None,
    ):
        """Run a codim-2 continuation branch (fold or Hopf curve in 2-param space).

        Uses AUTO-07p's ``ISW=2`` (branch switching) with two free parameters
        (``ICP=[p1, p2]``) to trace a fold or Hopf curve in the (p1, p2) plane.
        """
        from tvbo.analysis.bifurcation import BifurcationResult

        bc = branch.continuation
        fp2 = getattr(bc, "free_parameters", None) or {}
        if isinstance(fp2, dict) and fp2:
            fp2_first = next(iter(fp2.values()))
        elif isinstance(fp2, list) and fp2:
            fp2_first = fp2[0]
        else:
            return []

        fp2_name = str(fp2_first.name)
        pyrates_fp2_name = _pyrates_param_name(fp2_name)
        p2_min = float(fp2_first.domain.lo) if fp2_first.domain else -20.0
        p2_max = float(fp2_first.domain.hi) if fp2_first.domain else 20.0

        source = getattr(branch, "source_point", None) or "fold:all"
        source_type = source.split(":")[0]  # 'hopf' or 'fold'
        all_source = ":all" in source

        # Map source type to AUTO special point labels
        sp_type_map = {"hopf": "HB", "fold": "LP", "branch_point": "BP"}
        auto_sp = sp_type_map.get(source_type, source_type.upper())

        source_points = self._find_special_points(ode, "param", auto_sp)
        if not source_points:
            return []

        if all_source:
            indices = list(range(len(source_points)))
        else:
            idx_str = source.split(":")[1] if ":" in source else "-1"
            if idx_str.lstrip("-").isdigit():
                idx = int(idx_str)
                indices = [idx if idx >= 0 else len(source_points) + idx]
            else:
                indices = list(range(len(source_points)))

        # Build AUTO kwargs for codim-2
        c2_kwargs = self._cont_to_auto_kwargs(bc or cont, icp_name, p_min, p_max)
        # Override ICP to be [p1, p2] for codim-2
        c2_kwargs["ICP"] = [icp_name, pyrates_fp2_name]
        # ISW=2 for branch switching (codim-2 curve tracing)
        c2_kwargs["ISW"] = 2
        c2_kwargs["IPS"] = 1  # Equilibrium
        c2_kwargs["ILP"] = 0  # Don't detect folds again
        c2_kwargs["ISP"] = 2  # Detect bifurcations
        # Second parameter bounds
        c2_kwargs["RL0"] = min(p_min, p2_min)
        c2_kwargs["RL1"] = max(p_max, p2_max)

        if branch.bothside:
            c2_kwargs["bidirectional"] = True

        results = []
        for i in indices:
            if i < 0 or i >= len(source_points):
                continue
            sp_label = source_points[i]
            try:
                c2_name = f"codim2_{source_type}_{sp_label}"
                c2_sols, c2_cont = ode.run(
                    origin=p_cont if isinstance(p_cont, str) else "param",
                    starting_point=sp_label,
                    name=c2_name,
                    **c2_kwargs,
                )
                c2_res = BifurcationResult.from_pycobi(
                    ode=ode,
                    cont_name=c2_name,
                    model=None,  # Not needed for codim-2 curves
                    state_var_names=state_var_names,
                    icp=icp,
                    fp_name=fp_name,
                )
                c2_res._is_codim2 = True
                c2_res._fp2_name = fp2_name
                c2_res._ics_name = fp_name  # Original param (e.g. 'I')
                c2_res._source_type = source_type
                c2_res._cont_name = c2_name

                # Extract second parameter values
                icp2 = (param_idx or {}).get(pyrates_fp2_name)
                if icp2 is not None:
                    c2_res._icp2 = icp2
                    c2_res._fp2_pyrates = pyrates_fp2_name

                results.append(c2_res)
            except Exception:
                pass

        return results

    def _find_special_points(self, ode, cont_name, sp_type):
        """Find special point labels from a PyCoBi continuation."""
        try:
            summary = ode.get_summary(cont_name)
            if summary is None or len(summary) == 0:
                return []
            if "bifurcation" not in summary.columns:
                return []
            points = []
            bif_col = summary["bifurcation"]
            count = 0
            for idx in summary.index:
                bif_val = str(bif_col.loc[idx]).strip()
                if bif_val == sp_type:
                    count += 1
                    points.append(f"{sp_type}{count}")
            return points
        except Exception:
            return []

    # ── Helpers ──────────────────────────────────────────────────────────

    def _get_free_parameter(self, cont, model):
        """Extract the free parameter info from a Continuation spec."""
        fp_dict = cont.free_parameters if cont else None
        if fp_dict:
            fp_first = next(iter(fp_dict.values())) if isinstance(fp_dict, dict) else fp_dict[0]
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

    def _load_circuit(self, model):
        """Load a PyRates CircuitTemplate from a TVBO Dynamics model."""
        from pyrates.frontend import CircuitTemplate

        from tvbo.adapters.pyrates import _patch_pyrates_networkx_backend
        from tvbo.codegen.pyrates import to_pyrates_yaml_string

        # PyRates threads a ``backend`` kwarg into ComputeGraph that networkx >= 3.4's
        # dispatch decorator intercepts; apply the shared dispatch patch (same one the
        # main PyRates adapter uses) before the circuit is built and compiled.
        _patch_pyrates_networkx_backend()

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

    @staticmethod
    def _populate_var_map(ode, eq_file, state_var_names):
        """Recover the parameter name→PAR-index map from the generated .f90.

        Our systems are PyRates-generated, so AUTO-07p's c.* file carries
        ``parnames``/``unames`` and PyCoBi keys solutions by the variable *name*
        (``V``, ``I_``, …), not by ``U(i)``/``PAR(i)``. Populating the forward
        ``_var_map`` (name → ``("U", i)`` / ``("P", i)``) is therefore harmful:
        ``ODESystem.run`` maps every solution key through ``_map_var(…, "plot")``
        and would rewrite those names to ``U(i)``/``PAR(i)``, which then miss in
        the name-keyed solution (``KeyError: 'U(1)'``). We only populate the
        *inverse* map (used by the result extractor to translate a ``PAR(i)`` /
        ``U(i)`` reference back to a name) and return ``{param_name: PAR_index}``
        for numeric ICP resolution.

        Parses the Fortran ``stpnt`` subroutine's ``args(N) = value  ! name``
        lines.
        """
        param_idx = {}
        f90_path = eq_file + ".f90"
        if not os.path.exists(f90_path):
            return param_idx

        with open(f90_path) as f:
            src = f.read()

        # Parse: args(4) = 0.0  ! I_
        pat = re.compile(r"args\((\d+)\)\s*=\s*[^!]*!\s*(\S+)")
        for m in pat.finditer(src):
            idx = int(m.group(1))
            name = m.group(2)
            param_idx[name] = idx
            ode._var_map_inv[f"PAR({idx})"] = name

        # State variables: U(1), U(2), ...
        if state_var_names:
            for i, sv in enumerate(state_var_names):
                ode._var_map_inv[f"U({i + 1})"] = sv

        return param_idx

    def _cont_to_auto_kwargs(self, cont, icp_name, p_min, p_max, is_po=False):
        """Convert Continuation schema fields to AUTO-07p keyword arguments.

        Parameters
        ----------
        icp_name : str
            The PyRates-renamed parameter name. PyCoBi's ``_map_auto_kwargs``
            resolves this to the correct numeric PAR index internally.
        """
        kw = {}
        kw["ICP"] = icp_name
        kw["RL0"] = p_min
        kw["RL1"] = p_max
        kw["bidirectional"] = bool(getattr(cont, "bothside", False))

        if not is_po:
            kw["IPS"] = 1  # Equilibrium continuation
            kw["ILP"] = 1  # Detect fold bifurcations
            kw["ISP"] = 2  # Full automatic bifurcation detection
            kw["ISW"] = 1  # Normal continuation

        # Step size — AUTO-07p default DS=1e-4 is often too small
        ds = getattr(cont, "ds", None)
        kw["DS"] = float(ds) if ds is not None else 1e-2

        ds_min = getattr(cont, "ds_min", None)
        kw["DSMIN"] = float(ds_min) if ds_min is not None else 1e-8

        ds_max = getattr(cont, "ds_max", None)
        if ds_max is not None:
            kw["DSMAX"] = float(ds_max)
        else:
            kw["DSMAX"] = 0.1

        # Max steps
        max_steps = getattr(cont, "max_steps", None)
        kw["NMX"] = int(max_steps) if max_steps is not None else 2000

        # Tolerances — provide sane defaults; AUTO built-in defaults
        # can be too strict and cause MX (Newton failure) on first step
        tol = getattr(cont, "tol_stability", None)
        kw["EPSS"] = float(tol) if tol is not None else 1e-6

        newton_tol = getattr(cont, "newton_tol", None)
        if newton_tol is not None:
            kw["EPSL"] = float(newton_tol)
            kw["EPSU"] = float(newton_tol)
        else:
            kw["EPSL"] = 1e-7
            kw["EPSU"] = 1e-7

        newton_max = getattr(cont, "newton_max_iterations", None)
        if newton_max is not None:
            kw["ITMX"] = int(newton_max)
            kw["ITNW"] = int(newton_max)
        else:
            kw["ITNW"] = 8  # Max Newton corrections per step

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
        if dyn_ref:
            dyn_name = str(dyn_ref)
            # Check primary dynamics
            if exp.dynamics and getattr(exp.dynamics, "name", None) == dyn_name:
                return exp.dynamics
            # Check network dynamics dict
            net_dyn = getattr(exp.network, "dynamics", None) if exp.network else None
            if isinstance(net_dyn, dict) and dyn_name in net_dyn:
                return net_dyn[dyn_name]
        if exp.dynamics is not None:
            return exp.dynamics
        raise ValueError(f"Cannot resolve dynamics for continuation. dynamics='{dyn_ref}' not found.")

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
