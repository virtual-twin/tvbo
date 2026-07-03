# -*- coding: utf-8 -*-
"""DifferentialEquations.jl backend adapter for SimulationExperiment.

Generates a self-contained Julia script from the existing Julia templates
and executes it via juliacall, returning a TVBO TimeSeries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from tvbo.data.types import ExperimentResult
    from tvbo.classes.experiment import SimulationExperiment


# Julia packages required by the base Julia templates
# Use specific sub-packages instead of monolithic DifferentialEquations
# to avoid excessive memory usage / Bus errors during precompilation.
REQUIRED_PACKAGES = [
    "OrdinaryDiffEqTsit5",
    "StochasticDiffEq",
]


def _strip_plot_lines(code: str) -> str:
    """Remove 'using Plots' and plot() calls — plotting is handled in Python."""
    lines = []
    for line in code.splitlines():
        s = line.strip()
        if s.startswith("using Plots"):
            continue
        if s.startswith("plot(") and "sol" in s:
            continue
        lines.append(line)
    return "\n".join(lines)


class DiffEqAdapter:
    """Adapter for running SimulationExperiment via DifferentialEquations.jl.

    Renders the Julia DifferentialEquations template, executes it via
    juliacall, and returns a TVBO TimeSeries.
    """

    def __init__(self, experiment: "SimulationExperiment"):
        self.experiment = experiment

    def render_code(self, **kwargs) -> str:
        """Render Julia code for this experiment."""
        from tvbo import templates

        exp = self.experiment
        model = exp.dynamics
        model.update_metadata()

        ctx = {
            "experiment": exp,
            "model": model,
            "duration": exp.integration.duration,
            "dt": exp.integration.step_size,
            "plot": False,
            "fout": False,
        }
        ctx.update(kwargs)

        template = templates.lookup.get_template("tvbo-julia-DifferentialEquations.jl.mako")
        return template.render(**ctx)

    def run(self, **kwargs) -> "ExperimentResult":
        """Run simulation using DifferentialEquations.jl.

        Returns
        -------
        ExperimentResult
            Simulation results with named dimensions and coordinates.
        """
        from tvbo.data.types import ExperimentResult, SimulationResult
        from tvbo.run.julia import (
            ensure_packages,
            extract_ode_solution,
            run_julia_code,
            solution_to_dataarray,
        )

        exp = self.experiment
        model = exp.dynamics

        # 1. Ensure required Julia packages
        ensure_packages(*REQUIRED_PACKAGES)

        # 2. Generate Julia code, strip plotting
        code = self.render_code(**kwargs)
        code = _strip_plot_lines(code)

        # 3. Execute in Julia
        run_julia_code(code)

        # 4. Extract solution
        t, u, sol = extract_ode_solution()

        # 5. Build xr.DataArray directly — dims (time, variable, node[, mode])
        sv_names = list(model.state_variables.keys())
        n_modes = getattr(model, "number_of_modes", 1) or 1
        da = solution_to_dataarray(t, u, sv_names, 1, n_modes)

        sim = SimulationResult(data=da)
        return ExperimentResult(
            integration=sim,
            source=exp,
            name=getattr(exp, "label", None),
            sol=sol,
        )
