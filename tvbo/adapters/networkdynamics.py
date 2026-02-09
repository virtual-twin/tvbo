# -*- coding: utf-8 -*-
"""NetworkDynamics.jl backend adapter for SimulationExperiment.

Uses pyjulia (tvbo.adapters.julia) to execute generated Julia code
and return full Julia objects alongside a TVBO TimeSeries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tvbo.data.types import TimeSeries
    from tvbo.export.experiment import SimulationExperiment


# Julia packages required by NetworkDynamics.jl templates
REQUIRED_PACKAGES = [
    "Graphs",
    "NetworkDynamics",
    "OrdinaryDiffEqTsit5",
    "SimpleWeightedGraphs",
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


def _extract_graph_data(n_nodes: int) -> dict:
    """Extract graph adjacency, node positions, and edge weights from Julia Main.

    Returns a dict with:
        adjacency : (n_nodes, n_nodes) ndarray
        positions : (n_nodes, 2) ndarray  — spring layout
        weights   : (n_edges,) ndarray or None
    """
    from tvbo.run.julia import run_julia_code

    adj = np.array(
        run_julia_code("Array(adj_matrix)"), dtype=float
    )
    pos = np.array(
        run_julia_code("Array(node_positions)"), dtype=float
    )
    # Edge weights (only present for weighted graphs)
    try:
        w = np.array(
            run_julia_code("Array(edge_weights)"), dtype=float
        )
    except Exception:
        w = None
    return {"adjacency": adj, "positions": pos, "weights": w}


class NetworkDynamicsAdapter:
    """Adapter for running SimulationExperiment via NetworkDynamics.jl (pyjulia)."""

    def __init__(self, experiment: "SimulationExperiment"):
        self.experiment = experiment

    def run(self, **kwargs) -> "TimeSeries":
        """Run simulation using NetworkDynamics.jl.

        Returns the full Julia ``sol`` object attached to the TimeSeries as
        ``ts.sol`` so users can inspect it interactively, just like the
        bifurcation workflow.

        Returns
        -------
        TimeSeries
            Simulation results shaped ``(time, state_vars, nodes, 1)``.
            The raw Julia solution is available as ``ts.sol``.
        """
        from tvbo.data.types import TimeSeries
        from tvbo.run.julia import (
            ensure_packages,
            extract_ode_solution,
            run_julia_code,
            solution_to_array,
        )

        exp = self.experiment

        # 1. Ensure required Julia packages
        ensure_packages(*REQUIRED_PACKAGES)

        # 2. Generate Julia code, strip plotting
        code = exp.render_code("networkdynamics")
        code = _strip_plot_lines(code)

        # 3. Change Julia working directory to YAML source dir
        #    so that readdlm("Norm_G_DTI.txt") etc. resolve correctly.
        #    Save and restore because Julia cd() affects Python cwd via pyjulia.
        source = getattr(exp, '_source_file', None)
        import os
        original_cwd = os.getcwd()
        if source:
            from pathlib import Path
            src_dir = str(Path(source).parent)
            run_julia_code(f'cd("{src_dir}")')

        # 4. Execute in Julia – variables land in Main
        run_julia_code(code)

        # 5. Extract solution
        t, u, sol = extract_ode_solution()

        # 6. Reshape to TVBO convention
        sv_names = list(exp.local_dynamics.state_variables.keys())
        n_sv = len(sv_names)
        n_nodes = (
            getattr(exp.network, "number_of_nodes", None)
            or getattr(exp.network, "number_of_regions", 1)
        )
        data = solution_to_array(t, u, n_sv, n_nodes)

        # 7. Extract graph data from Julia
        graph_data = _extract_graph_data(n_nodes)

        # 8. Restore original working directory
        os.chdir(original_cwd)

        dt = float(exp.integration.step_size) if exp.integration else 0.01
        ts = TimeSeries(
            time=t,
            data=data,
            labels_dimensions={
                "State Variable": sv_names,
                "Region": list(range(n_nodes)),
            },
            sample_period=dt,
        )
        ts.source_experiment = exp
        ts.sol = sol  # keep full Julia object for interactive use
        ts.graph = graph_data  # adjacency, positions, weights
        return ts
