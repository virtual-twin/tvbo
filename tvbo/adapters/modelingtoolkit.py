# -*- coding: utf-8 -*-
"""ModelingToolkit.jl backend adapter for SimulationExperiment.

Extends NetworkDynamicsAdapter to use MTK equation-based templates
instead of function-based templates. Shares the same Julia runtime,
solution extraction, and graph/edge-observable logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tvbo.adapters.networkdynamics import (
    REQUIRED_PACKAGES,
    NetworkDynamicsAdapter,
    _extract_edge_observables,
    _extract_graph_data,
    _extract_vertex_observables,
    _strip_plot_lines,
)

if TYPE_CHECKING:
    from tvbo.data.types import TimeSeries

# Additional Julia packages required by MTK templates
MTK_PACKAGES = [
    "ModelingToolkit",
]


class ModelingToolkitAdapter(NetworkDynamicsAdapter):
    """Adapter for running SimulationExperiment via MTK + NetworkDynamics.jl.

    Uses the same pyjulia runtime as NetworkDynamicsAdapter but renders
    code via @mtkmodel templates instead of function-based templates.
    """

    def render_code(self, **kwargs) -> str:
        """Render Julia code using MTK templates."""
        from tvbo import templates

        ctx = self.prepare_context()
        ctx.update(kwargs)
        template = templates.lookup.get_template(
            "tvbo-mtk-experiment.jl.mako"
        )
        return template.render(**ctx)

    def run(self, **kwargs) -> "TimeSeries":
        """Run simulation using MTK + NetworkDynamics.jl.

        Same logic as NetworkDynamicsAdapter.run() but uses MTK templates
        and ensures ModelingToolkit.jl is installed.
        """
        import os

        import numpy as np

        from tvbo.data.types import TimeSeries
        from tvbo.run.julia import (
            ensure_packages,
            extract_ode_solution,
            run_julia_code,
            solution_to_array,
        )

        exp = self.experiment

        # 1. Ensure required Julia packages (ND + MTK)
        ensure_packages(*REQUIRED_PACKAGES, *MTK_PACKAGES)

        # 2. Generate Julia code, strip plotting
        code = self.render_code(**kwargs)
        code = _strip_plot_lines(code)

        # 3. Change Julia working directory to YAML source dir
        source = getattr(exp, '_source_file', None)
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
        sv_names = ctx['sv_names']
        n_sv = ctx['n_sv']
        n_nodes = ctx['n_nodes']
        is_hetero = ctx.get('is_heterogeneous', False)

        if is_hetero:
            n_total = u.shape[0]
            data = u.T[:, :, np.newaxis, np.newaxis]

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
                        state_labels.append(f"{sv_name}_{node.id}")
            if len(state_labels) != n_total:
                state_labels = [f"x_{i}" for i in range(n_total)]
        else:
            data = solution_to_array(t, u, n_sv, n_nodes)
            state_labels = sv_names

        # 7. Extract edge observables
        edge_data = _extract_edge_observables(
            ctx.get('outsym_names', []),
            ctx.get('coupling_observed', {}),
        )

        # 7b. Extract vertex derived-variable observables
        vertex_data = _extract_vertex_observables(
            ctx.get('vertex_dv_names', []),
            n_nodes,
        )

        # 8. Extract graph data
        graph_data = _extract_graph_data(n_nodes)

        # 9. Restore working directory
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
        ts.sol = sol
        ts.graph = graph_data
        ts.edge_data = edge_data
        ts.vertex_data = vertex_data

        if is_hetero and self.get_coupling_vars(ctx['model']):
            ts.node_positions = self.build_node_positions(ts, ctx)
            ts.initial_positions = self.get_initial_positions()
            ts.fixed_nodes = self.get_fixed_nodes()
            ts.node_metadata = self.get_node_metadata()

        return ts
