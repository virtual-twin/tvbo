"""NetworkDynamics.jl backend adapter for SimulationExperiment.

Uses pyjulia (tvbo.adapters.julia) to execute generated Julia code and return full Julia objects alongside a TVBO TimeSeries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tvbo.adapters.base import BaseAdapter, dense_matrix

if TYPE_CHECKING:
    from tvbo.data.types import ExperimentResult, SimulationResult


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

    adj = np.array(run_julia_code("Array(adj_matrix)"), dtype=float)
    pos = np.array(run_julia_code("Array(node_positions)"), dtype=float)
    # Edge weights (only present for weighted graphs)
    try:
        w = np.array(run_julia_code("Array(edge_weights)"), dtype=float)
    except Exception:
        w = None
    return {"adjacency": adj, "positions": pos, "weights": w}


def _extract_edge_observables(
    outsym_names: list[str],
    coupling_observed: dict,
) -> dict[str, np.ndarray]:
    """Extract edge observables from Julia solution using outsym metadata.

    For each symbol in outsym_names, extracts the per-edge time series using ``eidxs(sol, :, :sym)``.

    For coupling-defined observed variables (obssym), extracts those too.

    Returns a dict mapping symbol names to arrays of shape ``(n_t, n_edges)``.
    """
    from tvbo.run.julia import run_julia_code

    edge_data = {}
    all_syms = list(outsym_names)

    # Also collect obssym from coupling observed definitions
    for obs_list in coupling_observed.values():
        for obs in obs_list:
            name = getattr(obs, "name", str(obs))
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
            raw = run_julia_code(f"hcat(sol(sol.t; idxs=eidxs(sol, :, :{sym})).u...)'")
            edge_data[sym] = np.array(raw, dtype=float)
        except Exception:
            pass  # Symbol not available in solution — skip

    return edge_data


def _extract_vertex_observables(
    vertex_dv_names: list[str],
    n_nodes: int,
) -> dict[str, np.ndarray]:
    """Extract vertex derived-variable observables from Julia solution.

    For each symbol in vertex_dv_names, extracts per-node time series using ``vidxs(sol, i, :sym)``.

    Returns a dict mapping symbol names to arrays of shape ``(n_t, n_nodes)``.
    """
    from tvbo.run.julia import run_julia_code

    vertex_data = {}
    if not vertex_dv_names:
        return vertex_data

    for sym in vertex_dv_names:
        try:
            raw = run_julia_code(f"hcat([sol(sol.t; idxs=vidxs(sol, i, :{sym})).u for i in 1:{n_nodes}]...)'")
            vertex_data[sym] = np.array(raw, dtype=float)
        except Exception:
            pass  # Symbol not available for all nodes — skip

    return vertex_data


def _graph_edges(i: int, j: int, edge) -> list[tuple[int, int]]:
    """The `SimpleDiGraph` edges a listed edge ``i → j`` adds: both directions of an undirected edge, as `Network.matrix` mirrors it."""
    if not edge.directed and i != j:
        return [(i, j), (j, i)]
    return [(i, j)]


def _rhs(equation) -> str | None:
    """*equation*'s right-hand side as text, or ``None`` where there is no equation."""
    rhs = getattr(equation, "rhs", None)
    return None if rhs is None else str(rhs).strip()


def coupling_split(coupling) -> dict:
    """Where each part of *coupling* is evaluated on NetworkDynamics.jl: the pre-expression on every edge, weighted by the connection, and the post-expression once per node, on the edges' sum.

    Every edge carries a weight parameter, the one ``weight`` names, which the template sets from the connectome wherever the graph is built from it. A coupling that declares ``w`` or ``weight`` names that parameter itself: the connectome weight replaces the declared value, and the pre-expression, which already carries the factor, is emitted as declared, so a connection is weighted once. Any other coupling gets an implicit ``w``, default 1, which the edge multiplies into its pre-expression (``implicit_weight``). A declared weight the pre-expression does not read is refused, since the connectome would never reach the node, and so is an undeclared ``w`` it does read, which would be the implicit weight a second time.

    A parameter lives where its expression is evaluated. ``edge_parameters`` are the ``(name, default)`` pairs the edge model carries: those the pre-expression or an edge observable reads and those nothing reads, which a callback may address, then the implicit weight. ``post_parameters`` are the ``(name, symbol, default)`` triples the vertex model carries for the post-expression, under ``<coupling>_<name>`` so they cannot collide with the node's own parameters; a parameter both expressions read is on both. ``post`` is the post-expression, or ``None`` where it is absent or the identity ``gx``, and ``post_function`` the Julia function the edge template defines for it.
    """
    from tvbo.templates.base.utils import referenced

    declared = {str(name): getattr(p, "value", None) for name, p in (coupling.parameters or {}).items()}
    pre = _rhs(coupling.pre_expression)
    post = _rhs(coupling.post_expression)
    post = None if post in (None, "gx") else post
    observed = [_rhs(getattr(obs, "equation", None)) for obs in (getattr(coupling, "observed", None) or {}).values()]
    weight = next((name for name in ("w", "weight") if name in declared), None)
    if weight is not None and not referenced([weight], pre):
        raise NotImplementedError(
            f"coupling {coupling.name} declares `{weight}`, the per-edge weight the NetworkDynamics.jl templates set from the connectome, "
            f"but its pre_expression {pre!r} does not read it, so no connection weight would reach a node. "
            f"Write `{weight}` into the pre_expression, or rename the parameter so each edge is weighted for you."
        )
    if weight is None and referenced(["w"], pre):
        raise NotImplementedError(
            f"coupling {coupling.name}'s pre_expression {pre!r} reads `w` without declaring it, and `w` is the weight the NetworkDynamics.jl templates give every edge, so the connection would be weighted twice. "
            "Declare `w` as the coupling's weight parameter, or rename the symbol."
        )
    on_edge = set(referenced(declared, pre, *observed))
    on_node = referenced(declared, post)
    edge_parameters = [(name, value) for name, value in declared.items() if name in on_edge or name not in on_node]
    return {
        "weight": weight or "w",
        "implicit_weight": weight is None,
        "edge_parameters": edge_parameters + ([("w", 1.0)] if weight is None else []),
        "post": post,
        "post_function": f"{coupling.name}_post",
        "post_parameters": [(name, f"{coupling.name}_{name}", declared[name]) for name in on_node],
    }


class NetworkDynamicsAdapter(BaseAdapter):
    """Adapter for running SimulationExperiment via NetworkDynamics.jl (pyjulia).

    Inherits metadata processing from BaseAdapter. The prepare_context() method pre-computes all template variables so Mako templates stay clean.
    """

    TEMPLATE = "tvbo-nd-experiment.jl.mako"

    # ── Spatial / heterogeneous metadata ─────────────────────────────────

    def get_initial_positions(self) -> np.ndarray:
        """Extract initial (x, y, …) positions for ALL nodes from YAML.

        For free (dynamic) nodes: positions come from per-node ``state`` overrides (legacy ``initial_state`` arrays are also supported), at the indices marked ``coupling_variable=True``.
        For static (fixed) nodes: positions come from node parameter values (in parameter-definition order).

        Returns shape ``(n_nodes, n_coupling_vars)``, one row per node in `node_rows` order.
        """
        dynamics_dict = self.build_dynamics_dict()
        node_dynamics_map = self.build_node_dynamics_map()
        nodes = list(self.experiment.network.nodes)
        rows = self.node_rows()
        default_model = self.experiment.dynamics

        # Determine coupling var indices from the default (free) model
        coupling_vars = self.get_coupling_vars(default_model)
        n_cv = len(coupling_vars) or 1
        sv_names = list(default_model.state_variables.keys())
        cv_indices = [i for i, name in enumerate(sv_names) if name in coupling_vars]

        positions = np.zeros((len(nodes), n_cv))
        for node in nodes:
            dyn_name = node_dynamics_map[node.id]
            dyn = dynamics_dict[dyn_name]
            if self.is_static(dyn):
                # Static node: positions from per-node parameter overrides
                params = self.parse_node_parameters(node)
                if params:
                    vals = list(params.values())
                    positions[rows[node.id], : len(vals)] = [float(v) for v in vals[:n_cv]]
            else:
                # Dynamic node: positions from per-node state at cv indices
                node_state = getattr(node, "state", None)
                init_vals = []
                if node_state:
                    state_values = node_state.values() if isinstance(node_state, dict) else node_state
                    state_map = {}
                    for state_entry in state_values:
                        if isinstance(state_entry, dict):
                            sv_name = state_entry.get("name")
                            sv_value = state_entry.get("value")
                        else:
                            sv_name = getattr(state_entry, "name", None)
                            sv_value = getattr(state_entry, "value", None)
                        if sv_name is not None and sv_value is not None:
                            state_map[str(sv_name)] = float(sv_value)
                    init_vals = [state_map.get(name, None) for name in sv_names]

                if not init_vals:
                    legacy_init = getattr(node, "initial_state", None)
                    if legacy_init:
                        init_vals = [float(v) for v in legacy_init]

                for j, idx in enumerate(cv_indices):
                    if idx < len(init_vals) and init_vals[idx] is not None:
                        positions[rows[node.id], j] = init_vals[idx]
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
                "dynamics": dyn_name,
                "params": self.parse_node_parameters(node),
                "static": self.is_static(dyn),
                "label": getattr(node, "label", None),
            }
        return meta

    def build_node_positions(
        self,
        ts: SimulationResult,
        ctx: dict,
    ) -> np.ndarray:
        """Build ``(n_t, n_nodes, n_cv)`` position array from simulation data.

        For free nodes: positions come from the coupling-variable columns of the properly shaped ``(time, variable, node)`` DataArray.
        For fixed nodes: positions are constant (from YAML parameters).
        Nodes are laid out along the second axis in `node_rows` order.
        """
        dynamics_dict = ctx["dynamics_dict"]
        rows = ctx["node_rows"]
        node_dynamics_map = ctx["node_dynamics_map"]
        nodes = ctx["nodes"]
        default_model = ctx["model"]
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
                positions[:, rows[node.id], :] = init_pos[rows[node.id]]
            else:
                for j, cv_name in enumerate(coupling_vars):
                    if cv_name in dyn.state_variables:
                        positions[:, rows[node.id], j] = ts.sel(variable=cv_name, node=node.id).values
        return positions

    # ── Code generation ──────────────────────────────────────────────────

    def node_rows(self) -> dict[int, int]:
        """``{Node.id: row}``, the 0-based position of each node's vertex, through `Network.node_index_map`, the resolution `listed_edges` and `Network.matrix` apply, so a node whose id is not its position lands on the vertex its edges address."""
        network = getattr(self.experiment, "network", None)
        return network.node_index_map() if network is not None else {}

    def graph_layout(self, ctx: dict) -> dict:
        """The graph the template builds, a `SimpleDiGraph` in every form, and the weight and parameters of each of its edges in the order ``edges(g)`` visits them.

        ``graph_form`` follows one precedence: a curated generator, the weight matrix (`build_weight_matrix`, which an ``edge_matrix_files`` entry is read into when the network loads), the listed edges, a single node, a complete graph. Every form is directed, so the one `Directed` edge model serves every coupling and each node receives only what its incoming edges send: an undirected listed edge adds both directions, as `Network.matrix` mirrors it, the undirected graph a generator builds is converted with ``SimpleDiGraph`` before its edges are counted, and the complete graph is ``complete_digraph``. The matrix form is built from the matrix literal, ``SimpleDiGraph(SimpleWeightedDiGraph(W))`` over its nonzero entries, and the listed form from ``graph_edges``, its ``(source, target)`` rows; both are laid out in the order Graphs.jl visits a `SimpleDiGraph`'s edges, by source and then target, whatever order they were declared in.

        ``edge_weights`` and ``edge_parameters`` follow that order, so position ``k`` is the ``k``-th edge ``edges(g)`` yields. A weight is the network's own matrix, `Network.matrix("weight")` with its declared transforms, read at ``[target, source]``, and goes onto the coupling's weight parameter (`coupling_split`). Any other parameter a listed edge declares lands at each position its edge occupies, the last declaration winning as it does in the matrix; one naming a parameter the edge model does not carry, the weight, a parameter only the post-expression reads or one the coupling does not declare, is refused. A generated, complete or single-node graph has no weights to set, so ``edge_weights`` is ``None`` and every edge keeps its weight parameter's default. ``event_edges``, ``edge_event_names`` and ``line_partners`` place the events attached to edges (`edge_events`).
        """
        listed = ctx["listed_edges"]
        if ctx["has_graph_generator"]:
            form = "generator"
        elif ctx["weight_matrix"] is not None:
            form = "matrix"
        elif ctx["has_explicit_edges"]:
            form = "listed"
        else:
            form = "single" if ctx["n_nodes"] == 1 else "complete"
        layout = {"graph_form": form, "graph_edges": [], "edge_weights": None, "edge_parameters": []}
        placed = [
            (
                i,
                j,
                edge,
                {
                    name: value
                    for name, value in self.parse_node_parameters(edge).items()
                    if name != "weight" and value is not None
                },
            )
            for i, j, edge in listed
        ]
        declared = sorted({name for *_, params in placed for name in params})
        split = ctx["edge_split"]
        if split is not None:
            carried = {name for name, _ in split["edge_parameters"]}
            uncarried = [name for name in declared if name == split["weight"] or name not in carried]
            if uncarried:
                raise NotImplementedError(
                    f"the edges declare a per-edge {', '.join(uncarried)}, which the NetworkDynamics.jl edge model does not carry: "
                    f"`{split['weight']}` is set from the connectome weight, a parameter only the post-expression reads is applied at the node, once, to the summed input, "
                    f"and the edge model carries only the coupling's own parameters ({', '.join(sorted(carried))}). "
                    "Declare the connection strength as the edge's `weight`, and a post-expression parameter on the coupling."
                )
        if form not in ("matrix", "listed"):
            if declared:
                raise NotImplementedError(
                    f"the NetworkDynamics.jl templates build this network's graph from its {form}, not from its listed edges, "
                    f"so the per-edge {', '.join(declared)} those edges declare has no edge to land on. "
                    "Declare the connectome as listed edges or as a weight matrix."
                )
            undirected = form == "complete" or (form == "generator" and not getattr(ctx["graph_gen"], "directed", False))
            layout.update(self.edge_events(ctx["all_events"], [], [], generated=undirected))
            return layout

        if form == "matrix":
            sources, targets = np.nonzero(ctx["weight_matrix"])
            order = list(zip(sources.tolist(), targets.tolist(), strict=True))
        else:
            order = sorted({pair for i, j, edge, _ in placed for pair in _graph_edges(i, j, edge)})
        weights = dense_matrix(self.experiment.network, "weight")
        layout["graph_edges"] = order
        layout["edge_weights"] = [float(weights[j, i]) for i, j in order]
        position = {pair: k for k, pair in enumerate(order, start=1)}
        occupied = [[position[pair] for pair in _graph_edges(i, j, edge) if pair in position] for i, j, edge, _ in placed]
        values = {}
        for (*_, params), positions in zip(placed, occupied, strict=True):
            for k in positions:
                values.update(((k, name), value) for name, value in params.items())
        layout["edge_parameters"] = [(k, name, value) for (k, name), value in sorted(values.items())]
        layout.update(self.edge_events(ctx["all_events"], [edge for *_, edge, _ in placed], occupied))
        return layout

    @staticmethod
    def edge_events(events, edges, occupied, generated: bool = False) -> dict:
        """Where the events attached to edges land, over the listed *edges* whose graph positions are *occupied*.

        ``event_edges`` maps a target naming one listed edge to the graph position of its declared direction, ``source → target``: the edge whose ``label`` it is, or, as ``edge_<n>``, the ``n``-th listed edge counted from 1 in declaration order. ``edge_event_names`` are the events attached to edges, through such a target or ``all_edges``; any other target, a node label, is left to the template. An ``edge_<n>`` naming no listed edge with a place in the graph is refused.

        An undirected line is two directed edges, and a callback changes only the component it runs on, so where an edge event is declared ``line_partners`` maps each direction of every undirected listed line to the other: the template copies an edge affect's parameter changes onto the partner, and the line trips as the one edge it was declared as. A callback on the partner as well would fire a second time at the same instant and save a second pair of points there. The lines of a *generated* undirected graph are known only to Julia, so an edge event on one is refused.
        """
        import re

        labels = {str(edge.label): k for k, edge in enumerate(edges) if getattr(edge, "label", None)}
        targets, names = {}, []
        for event, _ in events:
            target = getattr(event, "target_component", None)
            numbered = re.fullmatch(r"edge_(\d+)", str(target))
            k = labels.get(str(target), int(numbered.group(1)) - 1 if numbered else None)
            if target != "all_edges" and k is None:
                continue
            if generated:
                raise NotImplementedError(
                    f"event {event.name} targets {target} on a generated graph, whose undirected edges the NetworkDynamics.jl templates build as two directed ones, so its affect would act on one direction of a line. "
                    "Declare the connectome as listed edges, whose lines the templates trip as one."
                )
            if k is not None and (not 0 <= k < len(edges) or not occupied[k]):
                raise NotImplementedError(
                    f"event {event.name} targets {target}, which names no listed edge the NetworkDynamics.jl graph carries: "
                    f"`edge_<n>` is the n-th of the {len(edges)} listed edges, counted from 1. Declare the edge in `network.edges`, or target `all_edges`."
                )
            if k is not None:
                targets[str(target)] = occupied[k][:1]
            names.append(str(event.name))
        lines = [
            positions for edge, positions in zip(edges, occupied, strict=True) if not edge.directed and len(positions) == 2
        ]
        partners = {a: b for pair in lines for a, b in (pair, pair[::-1])} if names else {}
        return {"event_edges": targets, "edge_event_names": names, "line_partners": dict(sorted(partners.items()))}

    def prepare_context(self) -> dict:
        """The shared context, plus each coupling's split between edge and node (`coupling_split`), the graph the template builds (`graph_layout`), each node's vertex row (`node_rows`), and whether the network needs its component models copied per index.

        ``coupling_splits`` is keyed like ``all_couplings``, and ``edge_split`` is the default coupling's, the one on the network's edges and so the one whose post-expression every vertex applies.

        ``dealias`` is set wherever the template gives one component a value the others do not carry as a model default: per-node dynamics, events, and every ``set_default!`` a fixpoint search is seeded with.
        """
        ctx = super().prepare_context()
        ctx["coupling_splits"] = {key: coupling_split(c) for key, c in ctx["all_couplings"].items()}
        ctx["edge_split"] = next(iter(ctx["coupling_splits"].values()), None)
        ctx.update(self.graph_layout(ctx))
        ctx["node_rows"] = self.node_rows()
        ctx["dealias"] = bool(ctx["is_heterogeneous"] or ctx["has_events"] or ctx["find_fixpoint"])
        return ctx

    def refuse_unrenderable(self) -> None:
        """Raise where the emitted Julia would quietly integrate something other than what was declared.

        The edge and experiment templates carry no delay path, so a delayed coupling is not lowered — it is dropped, and the run returns a well-formed trajectory of the undelayed network. They carry no observation path either, so a declared monitor is dropped the same way and the result arrives with an empty ``observations``. Refusing is the same contract the Brian2 adapter states for the forms it cannot lower: a clear error beats a plausible answer to a different question.
        """
        delayed = self.delayed_couplings()
        if delayed:
            raise NotImplementedError(
                "the NetworkDynamics.jl templates lower no transmission delay, so "
                f"{', '.join(delayed)} would be integrated as an undelayed coupling. "
                "Run the delayed network on a backend that carries a history buffer "
                "(tvb, tvboptim, jax), or declare the coupling undelayed."
            )
        from tvbo.utils import keyed_items

        observations = sorted(name for name, _ in keyed_items(getattr(self.experiment, "observations", None), "observations"))
        if observations:
            raise NotImplementedError(
                "the NetworkDynamics.jl templates emit no observation, so "
                f"{', '.join(observations)} would be dropped and the run would return the raw trajectory alone. "
                "Run the monitors on a backend that lowers them (tvb, tvboptim, jax), or drop them from the experiment."
            )
        super().refuse_unrenderable()

    def run(self, **kwargs) -> ExperimentResult:
        """Run simulation using NetworkDynamics.jl.

        Returns:
        -------
        ExperimentResult
            Simulation results with named dimensions and coordinates.
            Extra attributes: ``sol``, ``graph``, ``edge_data``, ``vertex_data``.
        """
        import xarray as xr

        ctx = self.render_context(**kwargs)

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
        code = _strip_plot_lines(self.render_template(ctx))

        # 3. Change Julia working directory to YAML source dir so that readdlm("Norm_G_DTI.txt") etc. resolve correctly.
        source = getattr(exp, "_source_file", None)
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

        # 6. Reshape to TVBO convention, by the context the code was rendered from
        sv_names = ctx["sv_names"]
        n_nodes = ctx["n_nodes"]
        is_hetero = ctx.get("is_heterogeneous", False)
        rows = ctx["node_rows"]

        if is_hetero:
            dynamics_dict = ctx["dynamics_dict"]
            node_dynamics_map = ctx["node_dynamics_map"]
            nodes = ctx["nodes"]
            default_name = ctx["model"].name if ctx["model"] else None

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

            # Extract per-variable time series via ND.jl vidxs (one Julia call per unique SV — batches all nodes that share that variable)
            for sv_idx, sv_name in enumerate(all_sv_names):
                node_ids = [
                    n.id
                    for n in nodes
                    if (d := dynamics_dict.get(node_dynamics_map.get(n.id, default_name)))
                    and d.state_variables
                    and sv_name in d.state_variables
                ]
                if not node_ids:
                    continue
                jl_ids = ", ".join(str(rows[nid] + 1) for nid in node_ids)
                raw = run_julia_code(
                    f"hcat([getindex.(sol(sol.t; idxs=vidxs(sol, i, :{sv_name})).u, 1) for i in [{jl_ids}]]...)"
                )
                vals = np.array(raw, dtype=float)  # (n_t, len(node_ids))
                for k, nid in enumerate(node_ids):
                    data[:, sv_idx, rows[nid]] = vals[:, k]

            da = xr.DataArray(
                data=data,
                dims=["time", "variable", "node"],
                coords={
                    "time": np.asarray(t),
                    "variable": all_sv_names,
                    "node": [node.id for node in nodes],
                },
            )
        else:
            da = solution_to_dataarray(t, u, sv_names, n_nodes)

        # 7. Extract edge observables from outsym metadata
        edge_data = _extract_edge_observables(
            ctx.get("outsym_names", []),
            ctx.get("coupling_observed", {}),
        )

        # 7b. Extract vertex derived-variable observables
        vertex_data = _extract_vertex_observables(
            ctx.get("vertex_dv_names", []),
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
        if is_hetero and self.get_coupling_vars(ctx["model"]):
            extras["node_positions"] = self.build_node_positions(sim, ctx)
            extras["initial_positions"] = self.get_initial_positions()
            extras["fixed_nodes"] = self.get_fixed_nodes()
            extras["node_metadata"] = self.get_node_metadata()

        return ExperimentResult(
            integration=sim,
            source=exp,
            name=getattr(exp, "label", None),
            **extras,
        )
