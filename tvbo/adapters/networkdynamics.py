# -*- coding: utf-8 -*-
"""NetworkDynamics.jl backend adapter for SimulationExperiment.

Uses pyjulia (tvbo.adapters.julia) to execute generated Julia code
and return full Julia objects alongside a TVBO TimeSeries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tvbo.adapters.base import BaseAdapter

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


def _extract_edge_observables(
    outsym_names: list[str],
    coupling_observed: dict,
) -> dict[str, np.ndarray]:
    """Extract edge observables from Julia solution using outsym metadata.

    For each symbol in outsym_names, extracts the per-edge time series
    using ``eidxs(sol, :, :sym)``.

    For coupling-defined observed variables (obssym), extracts those too.

    Returns a dict mapping symbol names to arrays of shape ``(n_t, n_edges)``.
    """
    from tvbo.run.julia import run_julia_code

    edge_data = {}
    all_syms = list(outsym_names)

    # Also collect obssym from coupling observed definitions
    for obs_list in coupling_observed.values():
        for obs in obs_list:
            name = getattr(obs, 'name', str(obs))
            if name not in all_syms:
                all_syms.append(name)

    if all_syms:
        # Ensure SymbolicIndexingInterface is available for eidxs
        try:
            run_julia_code("import SymbolicIndexingInterface")
        except Exception:
            pass

    for sym in all_syms:
        try:
            raw = run_julia_code(
                f"hcat(sol(sol.t; idxs=eidxs(sol, :, :{sym})).u...)'"
            )
            edge_data[sym] = np.array(raw, dtype=float)
        except Exception:
            pass  # Symbol not available in solution — skip

    return edge_data


class NetworkDynamicsAdapter(BaseAdapter):
    """Adapter for running SimulationExperiment via NetworkDynamics.jl (pyjulia).

    Inherits metadata processing from BaseAdapter. The prepare_context()
    method pre-computes all template variables so Mako templates stay clean.
    """

    def render_code(self, **kwargs) -> str:
        """Render Julia code with pre-computed context from BaseAdapter."""
        from tvbo import templates

        ctx = self.prepare_context()
        ctx.update(kwargs)
        template = templates.lookup.get_template(
            "tvbo-nd-experiment.jl.mako"
        )
        return template.render(**ctx)

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
        code = self.render_code(**kwargs)
        code = _strip_plot_lines(code)

        # 3. Change Julia working directory to YAML source dir
        #    so that readdlm("Norm_G_DTI.txt") etc. resolve correctly.
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
        ctx = self.prepare_context()
        sv_names = ctx['sv_names']
        n_sv = ctx['n_sv']
        n_nodes = ctx['n_nodes']
        is_hetero = ctx.get('is_heterogeneous', False)

        if is_hetero:
            # Heterogeneous models: nodes have different numbers of SVs,
            # so the total state dimension != n_nodes * n_sv.
            # Return raw (n_t, n_total_states, 1, 1) — no node/SV split.
            n_t = len(t)
            n_total = u.shape[0]
            data = u.T[:, :, np.newaxis, np.newaxis]   # (n_t, n_states, 1, 1)
        else:
            data = solution_to_array(t, u, n_sv, n_nodes)

        # 7. Extract edge observables from outsym metadata
        edge_data = _extract_edge_observables(
            ctx.get('outsym_names', []),
            ctx.get('coupling_observed', {}),
        )

        # 8. Extract graph data from Julia
        graph_data = _extract_graph_data(n_nodes)

        # 9. Restore original working directory
        os.chdir(original_cwd)

        dt = ctx['dt']
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
        ts.edge_data = edge_data  # edge observables from outsym/obssym
        return ts
