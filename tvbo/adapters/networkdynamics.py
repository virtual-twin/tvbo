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
    from tvbo.data.types import ExperimentResult, SimulationResult, TimeSeries
    from tvbo.classes.experiment import SimulationExperiment


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

        For free (dynamic) nodes: positions come from per-node ``state``
        overrides (legacy ``initial_state`` arrays are also supported),
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
                # Dynamic node: positions from per-node state at cv indices
                node_state = getattr(node, 'state', None)
                init_vals = []
                if node_state:
                    state_values = node_state.values() if isinstance(node_state, dict) else node_state
                    state_map = {}
                    for state_entry in state_values:
                        if isinstance(state_entry, dict):
                            sv_name = state_entry.get('name')
                            sv_value = state_entry.get('value')
                        else:
                            sv_name = getattr(state_entry, 'name', None)
                            sv_value = getattr(state_entry, 'value', None)
                        if sv_name is not None and sv_value is not None:
                            state_map[str(sv_name)] = float(sv_value)
                    init_vals = [state_map.get(name, None) for name in sv_names]

                if not init_vals:
                    legacy_init = getattr(node, 'initial_state', None)
                    if legacy_init:
                        init_vals = [float(v) for v in legacy_init]

                for j, idx in enumerate(cv_indices):
                    if idx < len(init_vals) and init_vals[idx] is not None:
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
        self, ts: "SimulationResult", ctx: dict,
    ) -> np.ndarray:
        """Build ``(n_t, n_nodes, n_cv)`` position array from simulation data.

        For free nodes: positions come from the coupling-variable columns
        of the properly shaped ``(time, variable, node)`` DataArray.
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
        for node in nodes:
            dyn_name = node_dynamics_map[node.id]
            dyn = dynamics_dict[dyn_name]
            if self.is_static(dyn):
                positions[:, node.id, :] = init_pos[node.id]
            else:
                for j, cv_name in enumerate(coupling_vars):
                    if cv_name in dyn.state_variables:
                        positions[:, node.id, j] = ts.sel(variable=cv_name, node=node.id).values
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

    def run(self, **kwargs) -> "ExperimentResult":
        """Run simulation using NetworkDynamics.jl.

        Returns
        -------
        ExperimentResult
            Simulation results with named dimensions and coordinates.
            Extra attributes: ``sol``, ``graph``, ``edge_data``, ``vertex_data``.
        """
        import xarray as xr

        from tvbo.data.types import ExperimentResult, SimulationResult
        from tvbo.run.julia import (
            ensure_packages,
            extract_ode_solution,
            run_julia_code,
            solution_to_dataarray,
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
            from tvbo.run.julia import run_julia_code

            dynamics_dict = ctx['dynamics_dict']
            node_dynamics_map = ctx['node_dynamics_map']
            nodes = ctx['nodes']
            default_name = ctx['model'].name if ctx['model'] else None

            # Collect all unique state variable names (preserving order)
            all_sv_names = []
            seen_sv = set()
            for node in nodes:
                dyn_name = node_dynamics_map.get(node.id, default_name)
                dyn = dynamics_dict.get(dyn_name)
                if dyn and dyn.state_variables:
                    for sv_name in dyn.state_variables:
                        if sv_name not in seen_sv:
                            all_sv_names.append(sv_name)
                            seen_sv.add(sv_name)

            n_t = len(t)
            n_unique_sv = len(all_sv_names)
            data = np.full((n_t, n_unique_sv, n_nodes), np.nan)

            # Extract per-variable time series via ND.jl vidxs (one Julia
            # call per unique SV — batches all nodes that share that variable)
            for sv_idx, sv_name in enumerate(all_sv_names):
                node_ids = [n.id for n in nodes
                            if (d := dynamics_dict.get(
                                node_dynamics_map.get(n.id, default_name)))
                            and d.state_variables and sv_name in d.state_variables]
                if not node_ids:
                    continue
                jl_ids = ', '.join(str(nid + 1) for nid in node_ids)
                raw = run_julia_code(
                    f"hcat([getindex.(sol(sol.t; idxs=vidxs(sol, i, :{sv_name})).u, 1)"
                    f" for i in [{jl_ids}]]...)"
                )
                vals = np.array(raw, dtype=float)  # (n_t, len(node_ids))
                for k, nid in enumerate(node_ids):
                    data[:, sv_idx, nid] = vals[:, k]

            da = xr.DataArray(
                data=data,
                dims=['time', 'variable', 'node'],
                coords={
                    'time': np.asarray(t),
                    'variable': all_sv_names,
                    'node': [node.id for node in nodes],
                },
            )
            state_labels = all_sv_names
        else:
            da = solution_to_dataarray(t, u, sv_names, n_nodes)
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

        # 10. Build SimulationResult (store graph so TimeSeries.animate() can access it)
        sim = SimulationResult(data=da, graph=graph_data)

        # Collect extra metadata
        extras = dict(sol=sol, graph=graph_data, edge_data=edge_data, vertex_data=vertex_data)
        if is_hetero and self.get_coupling_vars(ctx['model']):
            extras['node_positions'] = self.build_node_positions(sim, ctx)
            extras['initial_positions'] = self.get_initial_positions()
            extras['fixed_nodes'] = self.get_fixed_nodes()
            extras['node_metadata'] = self.get_node_metadata()

        return ExperimentResult(
            integration=sim, source=exp, name=getattr(exp, 'label', None),
            **extras,
        )
