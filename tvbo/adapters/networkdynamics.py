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
    "SymbolicIndexingInterface",
]

LISTED_EDGE_LIMIT = 50
"""The most listed edges the template adds one by one; a network listing more is emitted as its weight matrix."""

COUPLING_STATE_SENTINELS = ("incoming_states", "local_states")
"""A pre-expression that is one of these names reads the declared state itself: tvboptim's identity pre-expression, the vectorised sum of the sources' values."""


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


def _is_julia_body(text: str) -> bool:
    """Whether *text* is Julia statements written for the edge itself, several lines or an ``e_dst``/``obsout`` assignment, rather than one expression."""
    return "\n" in text.strip() or "e_dst[" in text or "obsout[" in text


def vertex_outputs(model, coupling) -> list[str]:
    """The states a vertex of *model* outputs, in the order every edge reads them.

    They are the states the coupling transmits, `gathered_states`: its ``incoming_states``, else the model's coupling variables, so ``x_j[k]`` is the ``k``-th of them here as on every other backend. A model marking none outputs all its states.
    """
    from tvbo.templates.base.utils import gathered_states

    return gathered_states(model, coupling) or list(model.state_variables)


def edge_expression(text: str, coupling, model, parameters: list[str]) -> tuple[str, bool]:
    """*text*, one of *coupling*'s expressions, as Julia over an edge's arguments, and whether it holds one value per vertex output.

    An edge reads the vertex outputs (`vertex_outputs`): ``v_src`` of the source, ``v_dst`` of the target. The names resolve as `coupling_bindings` resolves them for the other backends: ``x_j[k]`` and ``x_i[k]`` are the ``k``-th output (0-based) of the source and target; for a state the coupling declares in ``incoming_states`` or ``local_states``, ``<state>_j`` and ``<state>_i`` are its value at the source and target and its bare name its value at the source; the whole expression ``incoming_states`` or ``local_states`` is the one state the coupling declares, read at the source as tvboptim's identity pre-expression is. Unindexed ``x_j`` and ``x_i`` are the whole output vectors, so the expression is evaluated elementwise (``@.``) where a vertex outputs several states and on the first output otherwise; they keep that meaning where a state is named ``x``.

    Raises:
        NotImplementedError: A name that is neither one of *parameters* nor resolves as above, a state no vertex output carries, a bare local state that other backends read at the target, an index past the outputs, a list-valued expression, or an identity expression over several states. Each would emit Julia that fails or reads another state.
    """
    import sympy as sp

    from tvbo.codegen import render_expression
    from tvbo.codegen.code import ARRAY_FUNCTION_MAPPINGS
    from tvbo.parse.expression import parse_eq
    from tvbo.utils import as_list

    name = coupling.name
    states = [s for s in model.state_variables if s not in parameters]
    outputs = vertex_outputs(model, coupling)
    row = {s: k for k, s in enumerate(outputs, start=1)}
    incoming = [str(s) for s in as_list(getattr(coupling, "incoming_states", None))]
    local = [str(s) for s in as_list(getattr(coupling, "local_states", None))]

    def refuse(what: str):
        raise NotImplementedError(
            f"coupling {name}'s expression {text!r} {what}, and a NetworkDynamics.jl edge reads only the outputs of its two vertices, "
            f"the states {outputs} the coupling transmits. List the state in the coupling's `incoming_states` or mark it `coupling_variable: true`."
        )

    if text.startswith("[") and text.endswith("]"):
        raise NotImplementedError(
            f"coupling {name}'s pre_expression {text!r} is a list of components, which the NetworkDynamics.jl templates do not lower: a vertex applies the post-expression to each summed component on its own. Run it on tvboptim."
        )
    if text in COUPLING_STATE_SENTINELS:
        declared = list(dict.fromkeys(incoming + local))
        if len(declared) != 1:
            raise NotImplementedError(
                f"coupling {name}'s pre_expression `{text}` is the identity over the states it declares, {declared}, and the NetworkDynamics.jl templates lower it over exactly one."
            )
        text = declared[0]
    ambiguous = set(local) - set(incoming) if incoming else set()
    expr = parse_eq(
        text,
        parameters=[*parameters, "x_j", "x_i", *states, *(f"{s}_j" for s in states), *(f"{s}_i" for s in states)],
        functions=list(ARRAY_FUNCTION_MAPPINGS["julia"]),
    )

    def read(vertex: str, state: str) -> sp.Symbol:
        if state not in row:
            refuse(f"reads the state `{state}` at the {'source' if vertex == 'v_src' else 'target'}")
        return sp.Symbol(f"{vertex}[{row[state]}]")

    indexed_reads = {}
    for indexed in expr.atoms(sp.Indexed):
        base = str(indexed.base.label)
        if base not in ("x_j", "x_i"):
            continue
        k = indexed.indices[0]
        if len(indexed.indices) != 1 or not k.is_Integer or not 0 <= int(k) < len(outputs):
            refuse(f"indexes {indexed}, past the {len(outputs)} output(s)")
        indexed_reads[indexed] = read("v_src" if base == "x_j" else "v_dst", outputs[int(k)])
    expr = expr.xreplace(indexed_reads)
    free = {str(s) for s in expr.free_symbols} - {str(s) for s in indexed_reads.values()}
    elementwise = bool(free & {"x_j", "x_i"}) and len(outputs) > 1
    replace = {
        sp.Symbol(x): sp.Symbol(vertex if elementwise else f"{vertex}[1]")
        for x, vertex in (("x_j", "v_src"), ("x_i", "v_dst"))
    }
    for state in dict.fromkeys(incoming + local):
        for alias, vertex in ((f"{state}_j", "v_src"), (f"{state}_i", "v_dst")):
            if alias in free and alias not in ("x_j", "x_i"):
                replace[sp.Symbol(alias)] = read(vertex, state)
        if state in free:
            if state in ambiguous:
                raise NotImplementedError(
                    f"coupling {name}'s expression {text!r} reads the local state `{state}` by its bare name, which tvboptim reads at the target and the jax backend at the source. "
                    f"Write `{state}_i` for the target's value or `{state}_j` for the source's."
                )
            replace[sp.Symbol(state)] = read("v_src", state)
    unresolved = sorted(free - set(parameters) - {str(s) for s in replace} - {"t"})
    if unresolved:
        raise NotImplementedError(
            f"coupling {name}'s expression {text!r} reads {', '.join(f'`{u}`' for u in unresolved)}, which is neither one of its parameters nor a state it declares in "
            "`incoming_states` or `local_states`, so no backend binds it. Declare the parameter, or the state the coupling reads."
        )
    julia = render_expression(expr.xreplace(replace), format="julia", parameters=parameters)
    return (f"@. {julia}" if elementwise else julia), elementwise


def coupling_split(coupling, model) -> dict:
    """What NetworkDynamics.jl evaluates for *coupling* on an edge between vertices of *model*, and what once per node, on the edges' sum.

    Every edge multiplies its pre-expression by the connection's weight, the connectome's entry the template sets on it, as every other backend multiplies ``pre(x_i, x_j)`` by ``W[i, j]``. The weight is an edge parameter of its own: ``w``, or ``weight`` where the coupling declares a ``w``, so a declared ``w`` keeps its declared value inside the pre-expression. A coupling declaring both is refused, and so is a pre-expression reading the weight's name without declaring it, which would weight the connection twice.

    A parameter lives where its expression is evaluated. ``edge_parameters`` are the ``(name, default)`` pairs the edge model carries: those the pre-expression or an edge observable reads and those nothing reads, which a callback may address, then the weight. ``post_parameters`` are the ``(name, symbol, default)`` triples the vertex model carries for the post-expression, under ``<coupling>_<name>`` so they cannot collide with the node's own parameters; a parameter both expressions read is on both. ``post`` is the post-expression, or ``None`` where it is absent or the identity ``gx``, and ``post_function`` the Julia function the edge template defines for it.

    ``edge_body`` is the edge function's statements (`edge_expression`): the weighted pre-expression written into ``e_dst``, or a Julia body written for the edge, followed by the weighting. ``outdim`` is the number of values the edge sends, one per vertex output for an elementwise expression and one otherwise, and ``outsym`` their names. ``observed`` pairs each edge observable's name with its statements.
    """
    import re

    from tvbo.templates.base.utils import referenced

    declared = {str(name): getattr(p, "value", None) for name, p in (coupling.parameters or {}).items()}
    pre = _rhs(coupling.pre_expression) or "x_j - x_i"
    post = _rhs(coupling.post_expression)
    post = None if post in (None, "gx") else post
    observables = list((getattr(coupling, "observed", None) or {}).values())
    observed = [_rhs(getattr(obs, "equation", None)) for obs in observables]
    weight = next((name for name in ("w", "weight") if name not in declared), None)
    if weight is None:
        raise NotImplementedError(
            f"coupling {coupling.name} declares both `w` and `weight`, and the NetworkDynamics.jl templates carry the connectome weight on an edge parameter of one of those names."
        )
    if referenced([weight], pre, *observed):
        raise NotImplementedError(
            f"coupling {coupling.name} reads `{weight}` without declaring it, and `{weight}` is the connectome weight the NetworkDynamics.jl templates give every edge, so the connection would be weighted twice. "
            "Declare the parameter, or rename the symbol."
        )
    on_edge = set(referenced(declared, pre, *observed))
    on_node = referenced(declared, post)
    edge_parameters = [(name, value) for name, value in declared.items() if name in on_edge or name not in on_node]
    carried = [name for name, _ in edge_parameters] + [weight]
    declared_outsym = [str(s) for s in (getattr(coupling, "outsym", None) or [])]
    if _is_julia_body(pre):
        body = [line.strip() for line in pre.strip().splitlines()] + [f"e_dst .*= {weight}"]
        written = [int(k) for k in re.findall(r"e_dst\[(\d+)\]", pre)]
        if not declared_outsym and not written:
            raise NotImplementedError(
                f"coupling {coupling.name}'s pre_expression is a Julia edge body that writes no `e_dst[k]`, so the number of values it sends along each edge is unknown. Declare them as the coupling's `outsym`."
            )
        outdim = len(declared_outsym) or max(written)
    else:
        julia, elementwise = edge_expression(pre, coupling, model, carried)
        body = [f"e_dst .= {weight} .* ({julia})" if elementwise else f"e_dst[1] = {weight} * ({julia})"]
        outdim = len(vertex_outputs(model, coupling)) if elementwise else 1
    if declared_outsym and len(declared_outsym) != outdim:
        raise NotImplementedError(
            f"coupling {coupling.name} names {len(declared_outsym)} outputs, {declared_outsym}, and its pre_expression sends {outdim} value(s) along each edge."
        )
    outsym = declared_outsym or (["coupling"] if outdim == 1 else [f"flow_{s}" for s in vertex_outputs(model, coupling)])
    observed_bodies = []
    for k, (obs, text) in enumerate(zip(observables, observed, strict=True), start=1):
        if _is_julia_body(text):
            observed_bodies.append((str(obs.name), [line.strip() for line in text.splitlines()]))
            continue
        julia, elementwise = edge_expression(text, coupling, model, carried)
        if elementwise:
            raise NotImplementedError(
                f"coupling {coupling.name}'s observable {obs.name} is elementwise over the vertex outputs, and an edge observable holds one value."
            )
        observed_bodies.append((str(obs.name), [f"obsout[{k}] = {julia}"]))
    return {
        "weight": weight,
        "edge_parameters": edge_parameters + [(weight, 1.0)],
        "edge_body": body,
        "outdim": outdim,
        "outsym": outsym,
        "observed": observed_bodies,
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
                state = self.node_state(node, dyn)
                for j, idx in enumerate(cv_indices):
                    if sv_names[idx] in state:
                        positions[rows[node.id], j] = float(state[sv_names[idx]])
        return positions

    @staticmethod
    def node_state(node, dynamics) -> dict:
        """*node*'s declared initial values, ``{state: value}``: its ``state`` entries, else its legacy ``initial_state``, read in the order *dynamics* declares its states."""
        declared = getattr(node, "state", None) or []
        entries = declared.items() if isinstance(declared, dict) else ((None, entry) for entry in declared)
        values = {}
        for key, entry in entries:
            name = entry.get("name", key) if isinstance(entry, dict) else getattr(entry, "name", key)
            value = entry.get("value") if isinstance(entry, dict) else getattr(entry, "value", entry)
            if name is not None and value is not None:
                values[str(name)] = value
        if not values:
            legacy = getattr(node, "initial_state", None) or []
            values = {name: value for name, value in zip(dynamics.state_variables, legacy, strict=False) if value is not None}
        return values

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

        labels = self.node_labels(ctx)
        for node in nodes:
            dyn_name = node_dynamics_map[node.id]
            dyn = dynamics_dict[dyn_name]
            if self.is_static(dyn):
                positions[:, rows[node.id], :] = init_pos[rows[node.id]]
            else:
                for j, cv_name in enumerate(coupling_vars):
                    if cv_name in dyn.state_variables:
                        positions[:, rows[node.id], j] = ts.sel(variable=cv_name, node=labels[rows[node.id]]).values
        return positions

    # ── Code generation ──────────────────────────────────────────────────

    def node_rows(self) -> dict[int, int]:
        """``{Node.id: row}``, the 0-based position of each node's vertex, through `Network.node_index_map`, the resolution `listed_edges` and `Network.matrix` apply, so a node whose id is not its position lands on the vertex its edges address."""
        network = getattr(self.experiment, "network", None)
        return network.node_index_map() if network is not None else {}

    def connectome(self, ctx: dict) -> tuple[np.ndarray, bool]:
        """The weight matrix the template integrates, target-by-source, and whether it is the undirected graph of a library generator, whose edges pair into lines.

        It is `Network.matrix("weight")` with its declared transforms, which holds whatever the network was built from: listed edges, a data file, a matrix, or a generator the `Network` resolves, a curated library generator (Barabási–Albert, Watts–Strogatz, …, one with a ``networkx`` binding) among them, built in Python by `tvbo.graph_generators.catalog.library_weights` with its seed rather than by Graphs.jl's own random generators in the emitted script, so the graph is one Python chose and can name edge by edge.
        """
        network = self.experiment.network
        generator = ctx["graph_gen"] if ctx["has_graph_generator"] else None
        library = (
            generator is not None
            and getattr(generator, "builder", None) is None
            and network._db_networkx_binding_for(generator) is not None
        )
        return dense_matrix(network, "weight"), library and not getattr(generator, "directed", False)

    def graph_layout(self, ctx: dict) -> dict:
        """The graph the template builds, a `SimpleDiGraph` in every form, and the weight and parameters of each of its edges in the order ``edges(g)`` visits them.

        The graph is the one the Python `Network` holds (`connectome`), the graph every other backend integrates. ``graph_form`` is ``listed`` for a network listing at most `LISTED_EDGE_LIMIT` edges and no generator, whose ``graph_edges`` the template adds one by one as ``(source, target)`` rows; ``single`` for one node without edges, which the template gives a self-loop sending nothing; and ``matrix`` for every other network, built from the matrix literal ``weight_matrix`` (source-by-target) as ``SimpleDiGraph(SimpleWeightedDiGraph(W))`` over its nonzero entries, so a network without a connection is one without edges. Every form is directed, so the one `Directed` edge model serves every coupling and each node receives only what its incoming edges send: an undirected edge is both directions, as `Network.matrix` mirrors it. The edges are laid out in the order Graphs.jl visits a `SimpleDiGraph`'s edges, by source and then target, whatever order they were declared in.

        ``edge_weights`` and ``edge_parameters`` follow that order, so position ``k`` is the ``k``-th edge ``edges(g)`` yields. A weight is read off the connectome at ``[target, source]`` and goes onto the coupling's weight parameter (`coupling_split`). Any other parameter a listed edge declares lands at each position its edge occupies, the last declaration winning as it does in the matrix; one naming a parameter the edge model does not carry, the weight, a parameter only the post-expression reads or one the coupling does not declare, is refused, and so is one on a network whose graph a generator builds. ``event_edges``, ``edge_event_names`` and ``line_partners`` place the events attached to edges (`edge_events`), over the lines of the undirected listed edges or of an undirected generated graph.
        """
        listed = ctx["listed_edges"]
        generated = bool(ctx["has_graph_generator"])
        if listed and not generated and len(listed) <= LISTED_EDGE_LIMIT:
            form = "listed"
        elif ctx["n_nodes"] > 1 or generated:
            form = "matrix"
        else:
            form = "single"
        layout = {"graph_form": form, "graph_edges": [], "edge_weights": None, "edge_parameters": [], "weight_matrix": None}
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
        if generated and declared:
            raise NotImplementedError(
                f"the NetworkDynamics.jl templates build this network's graph from its graph_generator, not from its listed edges, "
                f"so the per-edge {', '.join(declared)} those edges declare has no edge to land on. "
                "Declare the connectome as listed edges or as a weight matrix."
            )
        if form == "single":
            layout.update(self.edge_events(ctx["all_events"], [], [], []))
            return layout

        weights, generated_lines = self.connectome(ctx)
        if form == "matrix":
            layout["weight_matrix"] = weights.T
            sources, targets = np.nonzero(weights.T)
            order = list(zip(sources.tolist(), targets.tolist(), strict=True))
        else:
            order = sorted({pair for i, j, edge, _ in placed for pair in _graph_edges(i, j, edge)})
        layout["graph_edges"] = order
        layout["edge_weights"] = [float(weights[j, i]) for i, j in order]
        position = {pair: k for k, pair in enumerate(order, start=1)}
        occupied = [[position[pair] for pair in _graph_edges(i, j, edge) if pair in position] for i, j, edge, _ in placed]
        values = {}
        for (*_, params), positions in zip(placed, occupied, strict=True):
            for k in positions:
                values.update(((k, name), value) for name, value in params.items())
        layout["edge_parameters"] = [(k, name, value) for (k, name), value in sorted(values.items())]
        if generated_lines:
            lines = [[position[(i, j)], position[(j, i)]] for i, j in order if i < j and (j, i) in position]
        else:
            lines = [
                positions
                for (*_, edge, _), positions in zip(placed, occupied, strict=True)
                if not edge.directed and len(positions) == 2
            ]
        layout.update(self.edge_events(ctx["all_events"], [edge for *_, edge, _ in placed], occupied, lines))
        return layout

    @staticmethod
    def edge_events(events, edges, occupied, lines) -> dict:
        """Where the events attached to edges land, over the listed *edges* whose graph positions are *occupied* and the undirected *lines*, each a pair of graph positions.

        ``event_edges`` maps a target naming one listed edge to the graph position of its declared direction, ``source → target``: the edge whose ``label`` it is, or, as ``edge_<n>``, the ``n``-th listed edge counted from 1 in declaration order. ``edge_event_names`` are the events attached to edges, through such a target or ``all_edges``. The templates lower continuous and preset-time events on edges and nothing else, so an event of another type, one targeting a node or the experiment, and an ``edge_<n>`` naming no listed edge with a place in the graph are refused rather than dropped.

        An undirected line is two directed edges, and a callback changes only the component it runs on, so where an edge event is declared ``line_partners`` maps each direction of every line to the other: the template copies an edge affect's parameter changes onto the partner, and the line trips as the one edge it was declared as. A callback on the partner as well would fire a second time at the same instant and save a second pair of points there.
        """
        import re

        labels = {str(edge.label): k for k, edge in enumerate(edges) if getattr(edge, "label", None)}
        targets, names = {}, []
        for event, _ in events:
            target = getattr(event, "target_component", None)
            kind = str(getattr(event.event_type, "text", event.event_type))
            numbered = re.fullmatch(r"edge_(\d+)", str(target))
            k = labels.get(str(target), int(numbered.group(1)) - 1 if numbered else None)
            if kind not in ("continuous", "preset_time") or (target != "all_edges" and k is None):
                raise NotImplementedError(
                    f"event {event.name} is a {kind} event targeting {target}, and the NetworkDynamics.jl templates lower continuous and preset_time events on edges only: "
                    f"`target_component` names an edge's label, `edge_<n>` or `all_edges`. Run it on a backend that lowers it, or drop it from the experiment."
                )
            if k is not None and (not 0 <= k < len(edges) or not occupied[k]):
                raise NotImplementedError(
                    f"event {event.name} targets {target}, which names no listed edge the NetworkDynamics.jl graph carries: "
                    f"`edge_<n>` is the n-th of the {len(edges)} listed edges, counted from 1. Declare the edge in `network.edges`, or target `all_edges`."
                )
            if k is not None:
                targets[str(target)] = occupied[k][:1]
            names.append(str(event.name))
        partners = {a: b for pair in lines for a, b in (pair, pair[::-1])} if names else {}
        return {"event_edges": targets, "edge_event_names": names, "line_partners": dict(sorted(partners.items()))}

    def holds_coupling(self) -> bool:
        """Whether each vertex holds the input its edges send across the stages of a step, as ``coupling_evaluation: per_step`` declares.

        NetworkDynamics.jl evaluates the edges inside the vector field, so a solver of several stages re-evaluates the coupling at each of them, which is ``per_stage``. For ``per_step``, the schema's default, each vertex reads its input off parameters instead, which a callback sets from the edges at the start of every step (`hold_rows`). A single-stage method, ``Euler`` whether it integrates an ODE or, as ``EM``, an SDE (`sde_solver`), sees the step's start either way, and a single node has no edge to hold (`coupling_evaluation_in_effect`).
        """
        from tvbo.adapters.base import coupling_evaluation_in_effect, declared_node_count

        integration = getattr(self.experiment, "integration", None)
        return coupling_evaluation_in_effect(integration, declared_node_count(self.experiment.network)) == "per_step"

    @staticmethod
    def vertex_layout(model, coupling, outsym: list[str], hold: bool) -> dict:
        """How a vertex of *model* meets the edges of *coupling*, which send it the inputs *outsym*: the ``outputs`` it sends (`vertex_outputs`) as the ``mask`` its `StateMask` selects, the ``insym`` naming its inputs, and the ``held`` parameters it reads them from where it holds its input (`holds_coupling`).

        The inputs are named wherever a vertex holds them, since the callback finds each vertex's input by name, and wherever the model has more than one coupling input and the coupling names its outputs.
        """
        outputs = vertex_outputs(model, coupling)
        order = list(model.state_variables)
        rows = [order.index(s) + 1 for s in outputs]
        mask = (
            f"{rows[0]}:{rows[-1]}" if rows == list(range(rows[0], rows[0] + len(rows))) else f"[{', '.join(map(str, rows))}]"
        )
        named = hold or (len(model.coupling_inputs or {}) > 1 and bool(getattr(coupling, "outsym", None)))
        return {
            "outputs": outputs,
            "mask": mask,
            "insym": outsym if named else None,
            "held": [f"held_{s}" for s in outsym] if hold else [],
        }

    def prepare_context(self) -> dict:
        """The shared context, plus each coupling's split between edge and node (`coupling_split`), the graph the template builds (`graph_layout`), each node's vertex row (`node_rows`) and declared initial values (`node_state`), how each dynamics' vertex meets the edges (`vertex_layout`), and whether the network needs its component models copied per index.

        ``coupling_splits`` is keyed like ``all_couplings``, and ``edge_split`` is the default coupling's, the one on the network's edges and so the one whose post-expression every vertex applies and whose ``outdim`` and ``outsym`` size the vertices' input. ``hold_coupling`` is `holds_coupling`, ``hold_rows`` the vertex rows, 1-based, whose input the callback holds, every vertex with dynamics of its own, and ``held_inputs`` pairs each input's name with the parameter it is held on.

        ``node_overrides`` holds the ``(row, name, value)`` triples, rows 1-based, a homogeneous network sets over its model's defaults and samples: under ``states`` each node's declared initial values, under ``parameters`` its declared parameter values, each restricted to the names the model declares, as on tvboptim.

        ``dealias`` is set wherever the template gives one component a value the others do not carry as a model default: per-node dynamics, events, and every ``set_default!`` a fixpoint search is seeded with. A stochastic network's ``solver_method`` is the StochasticDiffEq.jl solver of its declared method (`julia_sde_solver`), which refuses a method with no stochastic counterpart.
        """
        from tvbo.adapters.julia_model import julia_sde_solver

        ctx = super().prepare_context()
        model = ctx["model"]
        if ctx["is_stochastic"]:
            ctx["solver_method"] = julia_sde_solver(self.declared_integration(self.experiment.integration, "method"))
        ctx["coupling_splits"] = {key: coupling_split(c, model) for key, c in ctx["all_couplings"].items()}
        split = ctx["edge_split"] = next(iter(ctx["coupling_splits"].values()), None)
        if split is not None:
            ctx["outdim"], ctx["outsym_names"] = split["outdim"], split["outsym"]
        ctx.update(self.graph_layout(ctx))
        ctx["node_rows"] = rows = self.node_rows()
        dynamics = {node.id: ctx["dynamics_dict"][ctx["node_dynamics_map"].get(node.id, model.name)] for node in ctx["nodes"]}
        ctx["node_states"] = {
            node_id: self.node_state(node, dynamics[node_id]) for node_id, node in ((n.id, n) for n in ctx["nodes"])
        }
        ctx["hold_coupling"] = hold = self.holds_coupling()
        ctx["vertex_layouts"] = {
            name: self.vertex_layout(dyn, ctx["coupling"], ctx["outsym_names"], hold)
            for name, dyn in ctx["dynamics_dict"].items()
            if not self.is_static(dyn)
        }
        dynamic = sorted(rows[node_id] + 1 for node_id, dyn in dynamics.items() if not self.is_static(dyn))
        ctx["hold_rows"] = "1:nv(g)" if len(dynamic) == ctx["n_nodes"] else f"[{', '.join(map(str, dynamic))}]"
        ctx["held_inputs"] = [(name, f"held_{name}") for name in ctx["outsym_names"]] if hold else []
        nodes = sorted(ctx["nodes"], key=lambda node: rows[node.id])
        ctx["node_overrides"] = {
            "states": [
                (rows[n.id] + 1, name, value)
                for n in nodes
                for name, value in ctx["node_states"][n.id].items()
                if name in model.state_variables
            ],
            "parameters": [
                (rows[n.id] + 1, name, value)
                for n in nodes
                for name, value in self.parse_node_parameters(n).items()
                if name in model.parameters and value is not None
            ],
        }
        ctx["dealias"] = bool(ctx["is_heterogeneous"] or ctx["has_events"] or ctx["find_fixpoint"])
        return ctx

    @staticmethod
    def node_labels(ctx: dict) -> list[str]:
        """The result's node coordinates, one per vertex row: each node's label, as tvboptim and TVB label theirs, or ``node_<id>``, `Network.graph`'s name for it, where the node declares none, and ``node_<row>`` for a row no declared node occupies."""
        rows = ctx["node_rows"]
        by_row = {rows[node.id]: node.label or f"node_{node.id}" for node in ctx["nodes"] if node.id in rows}
        return [str(by_row.get(row, f"node_{row}")) for row in range(ctx["n_nodes"])]

    def refuse_unrenderable(self) -> None:
        """Raise where the emitted Julia would quietly integrate something other than what was declared.

        The edge and experiment templates carry no delay path, so a delayed coupling is not lowered — it is dropped, and the run returns a well-formed trajectory of the undelayed network. They carry no observation path either, so a declared monitor is dropped the same way and the result arrives with an empty ``observations``. A fixpoint search solves the network's vector field for its steady state, where a coupling held per step (`holds_coupling`) is a parameter no step has set yet, so ``find_fixpoint`` under ``coupling_evaluation: per_step`` is refused as well. Refusing is the same contract the Brian2 adapter states for the forms it cannot lower: a clear error beats a plausible answer to a different question.
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
        if self.get_execution_info()["find_fixpoint"] and self.holds_coupling():
            raise NotImplementedError(
                "execution.find_fixpoint searches the steady state of the network's vector field, and under integration.coupling_evaluation: per_step "
                "NetworkDynamics.jl reads each vertex's coupling off parameters a callback sets once per step, which no step has set when the search runs. "
                "Declare coupling_evaluation: per_stage, which a steady state does not distinguish from per_step, or integrate by Euler."
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
                    "node": self.node_labels(ctx),
                },
            )
        else:
            da = solution_to_dataarray(t, u, sv_names, n_nodes).assign_coords(node=self.node_labels(ctx))

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
