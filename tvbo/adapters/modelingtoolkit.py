# -*- coding: utf-8 -*-
"""Standalone ModelingToolkit.jl backend adapter for SimulationExperiment.

Pure MTK adapter using @component + mtkcompile + ODEProblem + solve.
No dependency on NetworkDynamics.jl.

Key capability: symbolic round-trip.  tvbo's SymPy equations are rendered to
MTK Julia code, MTK's ``mtkcompile`` performs structural transformations (e.g. higher-order ODE lowering), and the resulting equations are extracted
back into SymPy.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import TYPE_CHECKING

from tvbo.adapters.base import BaseAdapter

if TYPE_CHECKING:
    from tvbo.data.types import ExperimentResult

# Julia packages required by the MTK backend
MTK_PACKAGES = [
    "ModelingToolkit",
    "OrdinaryDiffEqTsit5",
    "Plots",
]


def _strip_plot_lines(code: str) -> str:
    """Remove plotting lines from generated Julia code for headless run."""
    lines = code.splitlines()
    filtered = []
    skip = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("using Plots") or stripped.startswith("plot("):
            skip = True
            continue
        if skip and stripped == "":
            skip = False
            continue
        skip = False
        filtered.append(line)
    return "\n".join(filtered)


class ModelingToolkitAdapter(BaseAdapter):
    """Adapter for running SimulationExperiment via pure ModelingToolkit.jl.

    Uses @component syntax (MTK v11+), mtkcompile, ODEProblem, and solve.

    Accepts ``Dynamics``, ``SimulationExperiment``, or no argument::

        adapter = ModelingToolkitAdapter()              # empty, set .experiment later
        adapter = ModelingToolkitAdapter(dynamics)      # wraps in minimal experiment
        adapter = ModelingToolkitAdapter(exp)            # full experiment
    """

    def __init__(self, source=None):
        from tvbo.classes.experiment import SimulationExperiment
        from tvbo.classes.dynamics import Dynamics

        if source is None:
            self.experiment = None
            self._input_dynamics = None
            return
        if isinstance(source, Dynamics):
            self._input_dynamics = source
            source = SimulationExperiment(dynamics=source)
        else:
            self._input_dynamics = None
        super().__init__(source)

    def render_code(self, **kwargs) -> str:
        """Render Julia code using the standalone MTK template."""
        from tvbo import templates

        ctx = self.prepare_context()
        ctx.update(kwargs)
        template = templates.lookup.get_template("tvbo-mtk-experiment.jl.mako")
        return template.render(**ctx)

    def run(self, **kwargs) -> "ExperimentResult":
        """Run simulation using pure ModelingToolkit.jl.

        Returns
        -------
        ExperimentResult
            Simulation results with named dimensions and coordinates.
        """
        import os

        from tvbo.data.types import ExperimentResult, SimulationResult
        from tvbo.run.julia import (
            ensure_packages,
            extract_ode_solution,
            run_julia_code,
            solution_to_dataarray,
        )

        exp = self.experiment

        # 1. Ensure required Julia packages
        ensure_packages(*MTK_PACKAGES)

        # 2. Generate Julia code, strip plotting
        code = self.render_code(**kwargs)
        code = _strip_plot_lines(code)

        # 3. Change Julia working directory to YAML source dir
        source = getattr(exp, "_source_file", None)
        original_cwd = os.getcwd()
        if source:
            from pathlib import Path

            src_dir = str(Path(source).parent)
            run_julia_code(f'cd("{src_dir}")')

        # 4. Execute in Julia
        run_julia_code(code)

        # 5. Extract solution
        t, u, sol = extract_ode_solution()

        # 6. Reshape to TVBO convention
        ctx = self.prepare_context()
        sv_names = ctx["sv_names"]
        n_sv = ctx["n_sv"]
        n_nodes = ctx["n_nodes"]

        # Pure MTK: single model, n_nodes=1 typically u shape from MTK: (n_unknowns, n_t) n_unknowns may differ from n_sv if mtkcompile introduced auxiliary variables (e.g., higher-order ODE lowering)
        n_unknowns = u.shape[0] if u.ndim == 2 else 1
        if n_unknowns != n_sv * n_nodes:
            try:
                unknowns = run_julia_code("string.(unknowns(sys))")
                state_labels = [_mtk_to_python_name(str(s).replace("(t)", "")) for s in list(unknowns)]
            except Exception:
                state_labels = [f"x_{i}" for i in range(n_unknowns)]
            # Flat unknowns — treat as n_unknowns variables, 1 node
            da = solution_to_dataarray(t, u, state_labels, 1)
        else:
            da = solution_to_dataarray(t, u, sv_names, n_nodes)

        # 7. Restore working directory
        os.chdir(original_cwd)

        sim = SimulationResult(data=da)
        return ExperimentResult(
            integration=sim,
            source=exp,
            name=getattr(exp, "label", None),
            sol=sol,
        )

    # ── Symbolic round-trip ────────────────────────────────────────────

    def lower(self, source=None, returns="auto", **kwargs):
        """Lower higher-order ODEs via MTK's ``mtkcompile``.

        Performs a symbolic round-trip: tvbo → MTK Julia → mtkcompile → lowered first-order SymPy equations, optionally wrapped back into
        a tvbo ``Dynamics`` or ``SimulationExperiment``.

        Parameters
        ----------
        source : Dynamics | SimulationExperiment, optional
            The system to lower. Can also be set at init time.
        returns : str
            What to return:

            - ``"sympy"`` — dict of SymPy equations, unknowns, parameters
            - ``"dynamics"`` — tvbo ``Dynamics`` with lowered equations
            - ``"experiment"`` — tvbo ``SimulationExperiment`` with lowered
              dynamics
            - ``"auto"`` (default) — same type as the input: ``Dynamics``
              if given ``Dynamics``, ``SimulationExperiment`` if given an
              experiment, ``"sympy"`` if empty.

        Returns
        -------
        dict | Dynamics | SimulationExperiment
        """
        from tvbo.classes.experiment import SimulationExperiment
        from tvbo.classes.dynamics import Dynamics

        # Accept source at call time — set up adapter state
        if source is not None:
            if isinstance(source, Dynamics):
                self._input_dynamics = source
                self.experiment = SimulationExperiment(dynamics=source)
            else:
                self._input_dynamics = None
                self.experiment = source

        result = self._lower_sympy(**kwargs)

        # Resolve "auto" based on what was passed to __init__
        if returns == "auto":
            if self._input_dynamics is not None:
                returns = "dynamics"
            elif self.experiment is not None:
                returns = "experiment"
            else:
                returns = "sympy"

        if returns == "sympy":
            return result

        if returns in ("dynamics", "experiment"):
            dyn = self._apply_lowered(result)
            if returns == "dynamics":
                return dyn
            return self.experiment.copy(
                dynamics=dyn,
                model=dyn,
            )

        raise ValueError(f"Unknown returns={returns!r}. Use 'sympy', 'dynamics', 'experiment', or 'auto'.")

    # ── Internal helpers for lower() ──────────────────────────────────

    def _lower_sympy(self, **kwargs):
        """Run MTK round-trip, return raw dict of SymPy equations."""
        from tvbo.run.julia import ensure_packages, run_julia_code

        # 1. Ensure packages + render & execute the model (up to mtkcompile)
        ensure_packages(*MTK_PACKAGES)
        code = self.render_code(**kwargs)

        # Only keep code up to (and including) the mtkcompile line — we don't need ODEProblem / solve / plot for equation extraction.
        lines = []
        for line in code.splitlines():
            lines.append(line)
            if "mtkcompile" in line:
                break
        run_julia_code("\n".join(lines))

        # 2. Extract from compiled system
        n_eqs = int(run_julia_code("length(equations(sys))"))
        param_names = list(run_julia_code("string.(parameters(sys))"))
        unknown_strs = list(run_julia_code("string.(unknowns(sys))"))

        # 3. Parse each equation RHS back to SymPy
        equations = {}
        for i in range(1, n_eqs + 1):
            lhs_str = str(run_julia_code(f"string(equations(sys)[{i}].lhs)"))
            rhs_str = str(run_julia_code(f"string(equations(sys)[{i}].rhs)"))
            var_name, eq = _parse_mtk_equation(
                lhs_str,
                rhs_str,
                unknown_strs,
                param_names,
            )
            equations[var_name] = eq

        # Variable names cleaned for Python
        unknowns = [_mtk_to_python_name(s.replace("(t)", "")) for s in unknown_strs]

        return {
            "equations": equations,
            "unknowns": unknowns,
            "parameters": param_names,
        }

    def _apply_lowered(self, lowered):
        """Apply lowered equations onto a deepcopy of the original Dynamics."""
        from tvbo.datamodel.schema import (
            Equation,
            StateVariable,
            StateVariableName,
        )

        original = self.experiment.dynamics
        dyn = deepcopy(original)
        dyn.name = f"{original.name}_FirstOrder"
        dyn.description = f"First-order equivalent of {original.name} (lowered via MTK)."

        # Update existing state variables with lowered equations
        for name, eq in lowered["equations"].items():
            if name in dyn.state_variables:
                sv = dyn.state_variables[name]
                sv.equation.lhs = str(eq.lhs)
                sv.equation.rhs = str(eq.rhs)
                sv.equation_order = 1
                sv.derivative_initial_value = None
            else:
                # Auxiliary variable introduced by MTK (e.g. x_t for dx/dt)
                init_val = _infer_aux_initial_value(name, original)
                sv = StateVariable(
                    name=StateVariableName(name),
                    equation=Equation(
                        lhs=str(eq.lhs),
                        rhs=str(eq.rhs),
                    ),
                    equation_type="differential",
                    equation_order=1,
                    initial_value=init_val,
                    variable_of_interest=False,
                    description="Auxiliary variable introduced by MTK",
                )
                dyn.state_variables[name] = sv

        return dyn


# ── Helpers for parsing MTK equation strings to SymPy ───────────────


def _mtk_to_python_name(name: str) -> str:
    """Convert MTK's Unicode auxiliary names to Python-safe names.

    ``xˍt`` → ``x_t``, ``xˍtt`` → ``x_tt``, etc.
    """
    return name.replace("\u02cd", "_")


def _parse_mtk_equation(lhs_str, rhs_str, unknown_strs, param_names):
    """Parse a single MTK equation string pair into a SymPy Eq.

    Parameters
    ----------
    lhs_str : str
        LHS from Julia, e.g. ``"Differential(t, 1)(x(t))"``.
    rhs_str : str
        RHS from Julia, e.g. ``"sigma*(-x(t) + y(t))"``.
    unknown_strs : list[str]
        Unknown names from Julia, e.g. ``["x(t)", "xˍt(t)", ...]``.
    param_names : list[str]
        Parameter names from Julia.

    Returns
    -------
    tuple[str, sympy.Eq]
        ``(python_var_name, Eq(Derivative(var, t), rhs_expr))``.
    """
    from sympy import Derivative, Eq, Symbol, parse_expr

    # Extract variable name from LHS: "Differential(t, 1)(x(t))" -> "x"
    m = re.search(r"\)\((\w+)\(t\)\)", lhs_str)
    var_name_jl = m.group(1) if m else lhs_str
    var_name = _mtk_to_python_name(var_name_jl)

    # Build local_dict for SymPy parsing
    local_dict = {}
    for p in param_names:
        local_dict[p] = Symbol(p)
    for u_str in unknown_strs:
        u_jl = u_str.replace("(t)", "")
        u_py = _mtk_to_python_name(u_jl)
        local_dict[u_py] = Symbol(u_py)

    # Clean RHS: strip "(t)" from variables, convert Unicode names
    rhs_clean = rhs_str
    for u_str in unknown_strs:
        u_jl = u_str.replace("(t)", "")
        u_py = _mtk_to_python_name(u_jl)
        rhs_clean = rhs_clean.replace(u_str, u_py)

    t = Symbol("t")
    rhs_expr = parse_expr(rhs_clean, local_dict=local_dict)
    eq = Eq(Derivative(Symbol(var_name), t), rhs_expr)

    return var_name, eq


def _infer_aux_initial_value(var_name, original):
    """Infer initial values for MTK-introduced auxiliary variables."""
    if "_t" in var_name:
        base = var_name.split("_t", 1)[0]
        base_sv = original.state_variables[base] if base in original.state_variables else None
        if base_sv is not None:
            deriv_init = getattr(base_sv, "derivative_initial_value", None)
            if var_name == f"{base}_t" and deriv_init is not None:
                return deriv_init
    return 0.0
