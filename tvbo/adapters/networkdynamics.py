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
    "OrdinaryDiffEqSDIRK",
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


def _extract_vertex_observables(
    vertex_dv_names: list[str],
    n_nodes: int,
) -> dict[str, np.ndarray]:
    """Extract vertex derived-variable observables from Julia solution.

    For each symbol in vertex_dv_names, extracts per-node time series
    using ``vidxs(sol, i, :sym)``.

    Returns a dict mapping symbol names to arrays of shape ``(n_t, n_nodes)``.
    """
    from tvbo.run.julia import run_julia_code

    vertex_data = {}
    if not vertex_dv_names:
        return vertex_data

    for sym in vertex_dv_names:
        try:
            raw = run_julia_code(
                f"hcat([sol(sol.t; idxs=vidxs(sol, i, :{sym})).u"
                f" for i in 1:{n_nodes}]...)'"
            )
            vertex_data[sym] = np.array(raw, dtype=float)
        except Exception:
            pass  # Symbol not available for all nodes — skip

    return vertex_data


class NetworkDynamicsAdapter(BaseAdapter):
    """Adapter for running SimulationExperiment via NetworkDynamics.jl (pyjulia).

    Inherits metadata processing from BaseAdapter. The prepare_context()
    method pre-computes all template variables so Mako templates stay clean.
    """

    # ── Spatial / heterogeneous metadata ─────────────────────────────────

    def get_initial_positions(self) -> np.ndarray:
        """Extract initial (x, y, …) positions for ALL nodes from YAML.

        For free (dynamic) nodes: positions come from ``initial_state``
        at the indices marked ``coupling_variable=True``.
        For static (fixed) nodes: positions come from node parameter
        values (in parameter-definition order).

        Returns shape ``(n_nodes, n_coupling_vars)``.
        """
        dynamics_dict = self.build_dynamics_dict()
        node_dynamics_map = self.build_node_dynamics_map()
        nodes = list(self.experiment.network.nodes)
        default_model = self.experiment.dynamics

        # Determine coupling var indices from the default (free) model
        coupling_vars = self.get_coupling_vars(default_model)
        n_cv = len(coupling_vars) or 1
        sv_names = list(default_model.state_variables.keys())
        cv_indices = [i for i, name in enumerate(sv_names)
                      if name in coupling_vars]

        positions = np.zeros((len(nodes), n_cv))
        for node in nodes:
            dyn_name = node_dynamics_map[node.id]
            dyn = dynamics_dict[dyn_name]
            if self.is_static(dyn):
                # Static node: positions from per-node parameter overrides
                params = self.parse_node_parameters(node)
                if params:
                    vals = list(params.values())
                    positions[node.id, :len(vals)] = [
                        float(v) for v in vals[:n_cv]
                    ]
            else:
                # Dynamic node: positions from initial_state at cv indices
                init = getattr(node, 'initial_state', None)
                if init:
                    init_vals = [float(v) for v in init]
                    for j, idx in enumerate(cv_indices):
                        if idx < len(init_vals):
                            positions[node.id, j] = init_vals[idx]
        return positions

    def get_fixed_nodes(self) -> set[int]:
        """Return set of node IDs with static dynamics (no state variables)."""
        dynamics_dict = self.build_dynamics_dict()
        node_dynamics_map = self.build_node_dynamics_map()
        fixed = set()
        for node in self.experiment.network.nodes:
            dyn_name = node_dynamics_map[node.id]
            dyn = dynamics_dict[dyn_name]
            if self.is_static(dyn):
                fixed.add(node.id)
        return fixed

    def get_node_metadata(self) -> dict[int, dict]:
        """Extract per-node metadata: dynamics name, parameters, type.

        Returns ``{node_id: {'dynamics': str, 'params': dict, 'static': bool}}``.
        """
        dynamics_dict = self.build_dynamics_dict()
        node_dynamics_map = self.build_node_dynamics_map()
        meta = {}
        for node in self.experiment.network.nodes:
            dyn_name = node_dynamics_map[node.id]
            dyn = dynamics_dict[dyn_name]
            meta[node.id] = {
                'dynamics': dyn_name,
                'params': self.parse_node_parameters(node),
                'static': self.is_static(dyn),
                'label': getattr(node, 'label', None),
            }
        return meta

    def build_node_positions(
        self, ts: "TimeSeries", ctx: dict,
    ) -> np.ndarray:
        """Build ``(n_t, n_nodes, n_cv)`` position array from simulation data.

        For free nodes: positions come from the coupling-variable columns
        of the flat heterogeneous state vector.
        For fixed nodes: positions are constant (from YAML parameters).
        """
        dynamics_dict = ctx['dynamics_dict']
        node_dynamics_map = ctx['node_dynamics_map']
        nodes = ctx['nodes']
        default_model = ctx['model']
        coupling_vars = self.get_coupling_vars(default_model)
        n_cv = len(coupling_vars) or 1

        n_t = len(ts.time)
        n_nodes = len(nodes)
        positions = np.zeros((n_t, n_nodes, n_cv))

        # Initial positions for fixed nodes
        init_pos = self.get_initial_positions()

        # Walk through state vector to find coupling-var columns per node
        state_offset = 0
        for node in nodes:
            dyn_name = node_dynamics_map[node.id]
            dyn = dynamics_dict[dyn_name]
            if self.is_static(dyn):
                # Constant position for all timesteps
                positions[:, node.id, :] = init_pos[node.id]
            else:
                # Find which state indices correspond to coupling vars
                sv_names = list(dyn.state_variables.keys())
                for j, cv_name in enumerate(coupling_vars):
                    if cv_name in sv_names:
                        sv_idx = sv_names.index(cv_name)
                        col = state_offset + sv_idx
                        positions[:, node.id, j] = ts.data[:, col, 0, 0]
                state_offset += len(sv_names)
        return positions

    # ── Code generation ──────────────────────────────────────────────────

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

            # Build per-state labels from node-dynamics map
            dynamics_dict = ctx['dynamics_dict']
            node_dynamics_map = ctx['node_dynamics_map']
            nodes = ctx['nodes']
            default_name = ctx['model'].name if ctx['model'] else None
            state_labels = []
            for node in nodes:
                dyn_name = node_dynamics_map.get(node.id, default_name)
                dyn = dynamics_dict.get(dyn_name)
                if dyn and dyn.state_variables:
                    for sv_name in dyn.state_variables:
                        state_labels.append(
                            f"{sv_name}_{node.id}"
                        )
            # Fallback if label count doesn't match
            if len(state_labels) != n_total:
                state_labels = [f"x_{i}" for i in range(n_total)]
        else:
            data = solution_to_array(t, u, n_sv, n_nodes)
            state_labels = sv_names

        # 7. Extract edge observables from outsym metadata
        edge_data = _extract_edge_observables(
            ctx.get('outsym_names', []),
            ctx.get('coupling_observed', {}),
        )

        # 7b. Extract vertex derived-variable observables
        vertex_data = _extract_vertex_observables(
            ctx.get('vertex_dv_names', []),
            n_nodes,
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
                "State Variable": state_labels,
                "Region": list(range(n_nodes)),
            },
            sample_period=dt,
        )
        ts.source_experiment = exp
        ts.sol = sol  # keep full Julia object for interactive use
        ts.graph = graph_data  # adjacency, positions, weights
        ts.edge_data = edge_data  # edge observables from outsym/obssym
        ts.vertex_data = vertex_data  # vertex derived-variable observables

        # Spatial metadata (for heterogeneous spatial models)
        if is_hetero and self.get_coupling_vars(ctx['model']):
            ts.node_positions = self.build_node_positions(ts, ctx)
            ts.initial_positions = self.get_initial_positions()
            ts.fixed_nodes = self.get_fixed_nodes()
            ts.node_metadata = self.get_node_metadata()

        return ts
