"""The coupling the NetworkDynamics.jl source integrates: node i receives post(sum_j w_ij * pre(x_i, x_j)), the coupling every other backend integrates, for every form a connectome is declared in.

Every graph is a `SimpleDiGraph` with the `Directed` edge model, and every edge carries the connection weight. ND.jl addresses an edge parameter by the edge's position in ``edges(g)``, so a value lands on the right connection only when it is emitted in the order Graphs.jl iterates the graph. That order was read off Graphs.jl itself (``julia --project=.venv/julia_env``): ``edges(SimpleDiGraph(...))`` visits ``(1,2), (1,4), (2,3), (3,1), (3,4), (4,2), (4,4)`` for `PROBE_W`, whatever order the edges were added in.

Each weight is checked against the network's own matrix, `Network.matrix("weight")`, which is target-by-source: the edge ``i → j`` carries ``matrix[j, i]``. The generated scripts were also run on the standalone Julia, where the RHS each node receives at t=0 equals the closed form for the curated Linear, Sigmoidal and Kuramoto couplings and for a coupling declaring its own ``w`` of 2.5, which stays inside the pre-expression while the connectome weight multiplies it.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from tvbo import SimulationExperiment

DYNAMICS = {"name": "Generic2dOscillator", "iri": "tvbo:Generic2dOscillator"}
WDIFF = {
    "WDiff": {
        "name": "WDiff",
        "delayed": False,
        "parameters": {"w": {"name": "w", "value": 1.0}, "K": {"name": "K", "value": 0.5}},
        "pre_expression": {"rhs": "w * K * (x_j - x_i)"},
    }
}
LINEAR = {"Linear": {"name": "Linear", "iri": "tvbo:Linear"}}
WLIN = {
    "WLin": {
        "name": "WLin",
        "delayed": False,
        "parameters": {"weight": {"name": "weight", "value": 1.0}, "gain": {"name": "gain", "value": 2.0}},
        "pre_expression": {"rhs": "weight * gain * x_j"},
    }
}
DOCS = Path(__file__).resolve().parents[1] / "docs/Interoperability/NetworkDynamics.jl/yaml"

PROBE_W = [[0.0, 1.0, 0.0, 4.0], [0.0, 0.0, 2.0, 0.0], [3.0, 0.0, 0.0, 5.0], [0.0, 6.0, 0.0, 7.0]]
"""Source-by-target: ``PROBE_W[i][j]`` is the weight of the edge ``i+1 → j+1``."""

PROBE_DIGRAPH_ORDER = [(1, 2), (1, 4), (2, 3), (3, 1), (3, 4), (4, 2), (4, 4)]
PROBE_GRAPH_ADDED = [(3, 1), (1, 2), (4, 4), (2, 3), (4, 1), (4, 3)]
PROBE_GRAPH_BOTH_WAYS = [(1, 2), (1, 3), (1, 4), (2, 1), (2, 3), (3, 1), (3, 2), (3, 4), (4, 1), (4, 3), (4, 4)]
"""`PROBE_GRAPH_ADDED` as undirected edges: each one both ways, a self-loop once, in ``edges(SimpleDiGraph)`` order."""
IDS = [40, 10, 30, 20]
"""Node ids that are not positions: row ``k`` is the node declared ``k``-th."""


def _experiment(network, execution=None, **integration):
    spec = {
        "dynamics": DYNAMICS,
        "network": network,
        "integration": {"method": "Heun", "step_size": 0.1, "duration": 1.0, **integration},
    }
    if execution:
        spec["execution"] = execution
    return SimulationExperiment(**spec)


def _edge(i, j, weight, directed, **params):
    """A listed edge between 1-based vertices *i* and *j*, addressed by node id."""
    parameters = {"weight": {"name": "weight", "value": weight}}
    parameters.update({name: {"name": name, "value": value} for name, value in params.items()})
    return {"source": IDS[i - 1], "target": IDS[j - 1], "directed": directed, "parameters": parameters}


def _listed(edges, coupling=WDIFF):
    return {"number_of_nodes": len(IDS), "nodes": [{"id": i} for i in IDS], "edges": edges, "coupling": coupling}


def _probe_digraph_edges(coupling=WDIFF, **params_at):
    """The probe digraph as directed listed edges, declared in reverse of Graphs.jl's order."""
    edges = [
        _edge(i, j, PROBE_W[i - 1][j - 1], True, **params_at.get(f"{i}{j}", {})) for i, j in reversed(PROBE_DIGRAPH_ORDER)
    ]
    return _listed(edges, coupling)


def _emitted_edges(code: str) -> list[tuple[int, int]]:
    """The graph's edges in ``edges(g)`` order: the listed form's ``add_edge!`` lines, or the matrix literal's nonzero entries by source, then target."""
    added = [(int(i), int(j)) for i, j in re.findall(r"^add_edge!\(g, (\d+), (\d+)\)$", code, re.MULTILINE)]
    if added:
        return added
    line = next(line for line in code.splitlines() if line.startswith("W = ["))
    rows = [[float(v) for v in row.split()] for row in line.removeprefix("W = [").removesuffix("]").split("; ")]
    return [(i + 1, j + 1) for i, row in enumerate(rows) for j, v in enumerate(row) if v]


def _emitted_weights(code: str) -> list[float]:
    line = next(line for line in code.splitlines() if line.startswith("edge_weights = Float64["))
    body = line.removeprefix("edge_weights = Float64[").removesuffix("]")
    return [float(v) for v in body.split(", ")] if body else []


def _edge_parameter_lines(code: str, target: str = r"s\.p\.e\[(\d+), :(\w+)\] = (\S+)") -> dict[tuple[int, str], float]:
    return {(int(k), name): float(v) for k, name, v in re.findall(target, code)}


def _block(code: str, start: str) -> str:
    """The Julia function or model that opens with *start*, up to its closing line."""
    return code.split(start, 1)[1].split("\nend\n", 1)[0].split("\n)\n", 1)[0]


def _matrix_weight(experiment, i, j) -> float:
    """The weight of the edge between 1-based vertices ``i → j``, off the network's own target-by-source matrix."""
    return float(experiment.network.matrix("weight", format="dense")[j - 1, i - 1])


# ── Every form puts its weights on the edges, in edges(g) order ──────────


def test_a_weight_matrix_sets_its_weights_in_edges_order():
    experiment = _experiment({"number_of_nodes": 4, "coupling": WDIFF})
    experiment.network.set_matrix("weight", np.array(PROBE_W).T)
    code = experiment.render_code(format="networkdynamics")

    assert "g = SimpleDiGraph(SimpleWeightedDiGraph(W))" in code
    assert _emitted_edges(code) == PROBE_DIGRAPH_ORDER
    assert _emitted_weights(code) == [PROBE_W[i - 1][j - 1] for i, j in PROBE_DIGRAPH_ORDER]
    assert "s.p.e[1:ne(g), :weight] = edge_weights" in code
    assert "pflat(s)" in code and "pflat(p)" not in code


def test_directed_listed_edges_set_their_weights_in_edges_order():
    experiment = _experiment(_probe_digraph_edges())
    code = experiment.render_code(format="networkdynamics")

    assert "g = SimpleDiGraph(4)" in code
    assert _emitted_edges(code) == PROBE_DIGRAPH_ORDER
    assert _emitted_weights(code) == [_matrix_weight(experiment, i, j) for i, j in PROBE_DIGRAPH_ORDER]
    assert _emitted_weights(code) == [PROBE_W[i - 1][j - 1] for i, j in PROBE_DIGRAPH_ORDER]
    assert "s.p.e[1:ne(g), :weight] = edge_weights" in code


def test_undirected_listed_edges_become_both_directions_of_a_digraph():
    declared = {pair: 10.0 * (k + 1) for k, pair in enumerate(PROBE_GRAPH_ADDED)}
    experiment = _experiment(_listed([_edge(i, j, w, False) for (i, j), w in declared.items()], coupling=LINEAR))
    code = experiment.render_code(format="networkdynamics")

    assert "g = SimpleDiGraph(4)" in code and "SimpleGraph(" not in code
    assert "g = Directed(Linear_edge_g!)" in code
    assert _emitted_edges(code) == PROBE_GRAPH_BOTH_WAYS
    either_way = {frozenset(pair): w for pair, w in declared.items()}
    assert _emitted_weights(code) == [either_way[frozenset(pair)] for pair in PROBE_GRAPH_BOTH_WAYS]
    assert _emitted_weights(code) == [_matrix_weight(experiment, i, j) for i, j in PROBE_GRAPH_BOTH_WAYS]


def test_listed_edges_and_their_matrix_set_the_same_weights():
    listed = _experiment(_probe_digraph_edges())
    matrix = _experiment({"number_of_nodes": 4, "coupling": WDIFF})
    matrix.network.set_matrix("weight", listed.network.matrix("weight", format="dense"))

    listed_code, matrix_code = (e.render_code(format="networkdynamics") for e in (listed, matrix))
    assert _emitted_edges(listed_code) == _emitted_edges(matrix_code) == PROBE_DIGRAPH_ORDER
    assert _emitted_weights(listed_code) == _emitted_weights(matrix_code)


def test_explicit_edges_over_the_threshold_set_weights_and_parameters_by_row():
    """More than 50 listed edges are emitted as the matrix; ids are not rows, and the weights and a per-edge parameter follow the rows."""
    n = 10
    ids = [10 * k + 3 for k in range(n)][::-1]
    pairs = [(s, t) for s in range(n) for t in range(n) if s != t and (s + 2 * t) % 3]
    edges = []
    for s, t in pairs[::-1]:
        edge = {
            "source": ids[s],
            "target": ids[t],
            "directed": True,
            "parameters": {"weight": {"name": "weight", "value": float(1 + (7 * s + 3 * t) % 5)}},
        }
        if (s, t) == (4, 6):
            edge["parameters"]["K"] = {"name": "K", "value": 0.7}
        edges.append(edge)
    assert len(edges) > 50
    experiment = _experiment({"number_of_nodes": n, "nodes": [{"id": i} for i in ids], "edges": edges, "coupling": WDIFF})
    code = experiment.render_code(format="networkdynamics")

    order = sorted(pairs)
    assert _emitted_edges(code) == [(s + 1, t + 1) for s, t in order]
    assert _emitted_weights(code) == [float(1 + (7 * s + 3 * t) % 5) for s, t in order]
    assert _edge_parameter_lines(code) == {(order.index((4, 6)) + 1, "K"): 0.7}


# ── Every graph is a SimpleDiGraph with the Directed edge model ─────────────


def test_every_graph_takes_the_directed_edge_model():
    undirected = _listed([_edge(i, j, 1.0, False) for i, j in PROBE_GRAPH_ADDED])
    for network in (_probe_digraph_edges(), {"number_of_nodes": 4, "coupling": WDIFF}, undirected):
        experiment = _experiment(network)
        if "edges" not in network:
            experiment.network.set_matrix("weight", np.array(PROBE_W).T)
        code = experiment.render_code(format="networkdynamics")
        assert "g = Directed(WDiff_edge_g!)" in code
        assert "AntiSymmetric(" not in code and "Symmetric(" not in code
        assert "SimpleGraph(" not in code


def test_an_undirected_edge_beside_directed_ones_couples_both_ways():
    network = _listed([_edge(1, 2, 2.0, True), _edge(2, 3, 5.0, False)])
    code = _experiment(network).render_code(format="networkdynamics")

    assert _emitted_edges(code) == [(1, 2), (2, 3), (3, 2)]
    assert _emitted_weights(code) == [2.0, 5.0, 5.0]


def test_a_generated_graph_is_the_one_python_builds():
    """A library generator is built in Python by its curated ``networkx`` binding and reaches Julia as that graph, each undirected edge both ways with weight 1, not as a Graphs.jl constructor call."""
    ring = {"number_of_nodes": 6, "graph_generator": {"name": "cycle", "type": "Cycle"}, "coupling": LINEAR}
    code = _experiment(ring).render_code(format="networkdynamics")
    assert "cycle_graph" not in code
    assert "g = SimpleDiGraph(SimpleWeightedDiGraph(W))" in code
    assert _emitted_edges(code) == sorted((i + 1, (i + d) % 6 + 1) for i in range(6) for d in (1, 5))
    assert _emitted_weights(code) == [1.0] * 12
    assert "g = Directed(Linear_edge_g!)" in code


def test_a_seeded_generator_draws_the_same_graph_as_networkx():
    import networkx as nx

    network = {
        "number_of_nodes": 12,
        "graph_generator": {"name": "er", "type": "ErdosRenyi", "seed": 7, "parameters": {"p": {"name": "p", "value": 0.3}}},
        "coupling": LINEAR,
    }
    code = _experiment(network).render_code(format="networkdynamics")
    graph = nx.erdos_renyi_graph(12, 0.3, seed=7)
    assert _emitted_edges(code) == sorted((i + 1, j + 1) for u, v in graph.edges for i, j in ((u, v), (v, u)))


def test_a_generator_networkx_cannot_build_at_the_declared_size_raises():
    """The ``star_graph(n)`` of networkx has ``n + 1`` nodes, so the Star entry cannot build a network of the size it declares."""
    star = {"number_of_nodes": 5, "graph_generator": {"name": "star", "type": "Star"}, "coupling": LINEAR}
    with pytest.raises(ValueError, match="builds 6 nodes"):
        _experiment(star).render_code(format="networkdynamics")


def test_a_data_file_is_read_into_the_network_and_emitted_like_its_matrix(tmp_path):
    """An ``edge_matrix_files`` entry is target-by-source, like every matrix a Network takes: the two-node file below couples node 2 into node 1 with 0.2, and node 1 into node 2 with 3."""
    file = np.array([[0.0, 0.2], [3.0, 0.0]])
    np.savetxt(tmp_path / "W.csv", file, delimiter=",")
    spec = {
        "label": "two",
        "dynamics": DYNAMICS,
        "network": {"edge_matrix_files": ["W.csv"], "coupling": LINEAR},
        "integration": {"method": "Heun", "step_size": 0.1, "duration": 1.0},
    }
    (tmp_path / "two.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    experiment = SimulationExperiment.from_file(str(tmp_path / "two.yaml"))
    code = experiment.render_code(format="networkdynamics")

    assert "readdlm" not in code
    assert "W = [0 3; 0.2 0]" in code
    assert _emitted_edges(code) == [(1, 2), (2, 1)]
    assert _emitted_weights(code) == [3.0, 0.2]
    assert _emitted_weights(code) == [_matrix_weight(experiment, i, j) for i, j in [(1, 2), (2, 1)]]
    assert "s.p.e[1:ne(g), :w] = edge_weights" in code


# ── Every edge is weighted once, the post-expression applied once at the node ──


def test_a_coupling_declaring_no_weight_gets_an_implicit_one_it_multiplies_in():
    code = _experiment(_probe_digraph_edges(coupling=LINEAR)).render_code(format="networkdynamics")

    assert "e_dst[1] = w * (v_src[1])" in _block(code, "function Linear_edge_g!(e_dst, v_src, v_dst, (w,), t)")
    assert "psym = [:w => 1.0]," in _block(code, "edge_Linear = EdgeModel(")
    assert _emitted_weights(code) == [PROBE_W[i - 1][j - 1] for i, j in PROBE_DIGRAPH_ORDER]
    assert "s.p.e[1:ne(g), :w] = edge_weights" in code


def test_a_declared_w_keeps_its_value_inside_the_pre_expression():
    """A coupling's own ``w`` or ``weight`` is one of its parameters, at its declared value: the connectome weight is an edge parameter of the other name, multiplying the pre-expression, as ``W[i, j]`` multiplies ``pre`` on tvboptim and TVB."""
    for coupling, name, declared, weight, pre in (
        (WDIFF, "WDiff", "w", "weight", "K .* w .* (v_src[1] .- v_dst[1])"),
        (WLIN, "WLin", "weight", "w", "gain .* v_src[1] .* weight"),
    ):
        code = _experiment(_probe_digraph_edges(coupling=coupling)).render_code(format="networkdynamics")
        edge = _block(code, f"function {name}_edge_g!(")
        assert f"    e_dst[1] = {weight} * ({pre})" in edge
        psym = _block(code, f"edge_{name} = EdgeModel(").split("psym = ", 1)[1].split("\n", 1)[0]
        assert f":{declared} => 1.0" in psym and psym.endswith(f":{weight} => 1.0],")
        assert f"s.p.e[1:ne(g), :{weight}] = edge_weights" in code
        assert f"s.p.e[1:ne(g), :{declared}]" not in code


def test_a_coupling_declaring_both_weight_names_raises():
    both = {
        "Both": {
            **WDIFF["WDiff"],
            "name": "Both",
            "parameters": {**WDIFF["WDiff"]["parameters"], "weight": {"name": "weight", "value": 1.0}},
        }
    }
    with pytest.raises(NotImplementedError, match="declares both `w` and `weight`"):
        _experiment(_probe_digraph_edges(coupling=both)).render_code(format="networkdynamics")


def test_a_weight_the_pre_expression_misreads_raises():
    undeclared = {"Undeclared": {"name": "Undeclared", "delayed": False, "pre_expression": {"rhs": "w * x_j"}}}
    with pytest.raises(NotImplementedError, match="without declaring it"):
        _experiment(_probe_digraph_edges(coupling=undeclared)).render_code(format="networkdynamics")


def test_the_post_expression_is_applied_once_at_the_node():
    """Linear's ``a * gx + b`` is defined once, called once by the vertex on the summed input, and its ``a``, ``b`` sit on the vertex beside the model's own."""
    code = _experiment(_probe_digraph_edges(coupling=LINEAR)).render_code(format="networkdynamics")

    assert code.count("Linear_post(gx, (b, a,)) = b .+ a .* gx") == 1
    assert code.index("Linear_post(gx,") < code.index("function Generic2dOscillator_f!(")
    vertex = _block(code, "function Generic2dOscillator_f!(")
    assert vertex.startswith(
        "dx, x, esum, (I, a, alpha, b, beta, c, d, e, f, g, gamma, tau, Linear_b, Linear_a, held_coupling,), t)"
    )
    assert "    c_glob = Linear_post(esum[1], (Linear_b, Linear_a,))" in vertex
    assert code.count("Linear_post(") == 2
    model = _block(code, "vertex_Generic2dOscillator = VertexModel(")
    assert ":a => -2.0" in model and ":b => -10.0" in model
    assert ":Linear_b => 0.0, :Linear_a => 0.00390625, :held_coupling => 0.0]" in model
    assert "Linear_post" not in _block(code, "function Linear_edge_g!(")


def test_a_parameter_both_expressions_read_is_on_the_edge_and_the_node():
    both = {
        "Both": {
            "name": "Both",
            "delayed": False,
            "parameters": {
                "G": {"name": "G", "value": 0.5},
                "limit": {"name": "limit", "value": 2.0},
                "c0": {"name": "c0", "value": 0.1},
            },
            "pre_expression": {"rhs": "G * x_j"},
            "post_expression": {"rhs": "G * gx + c0"},
        }
    }
    code = _experiment(_probe_digraph_edges(coupling=both)).render_code(format="networkdynamics")

    assert "psym = [:G => 0.5, :limit => 2.0, :w => 1.0]," in _block(code, "edge_Both = EdgeModel(")
    assert ":Both_G => 0.5, :Both_c0 => 0.1, :held_coupling => 0.0]" in _block(
        code, "vertex_Generic2dOscillator = VertexModel("
    )
    assert "Both_post(gx, (G, c0,)) = " in code


def test_an_identity_post_expression_leaves_the_summed_input_as_it_is():
    code = _experiment(_probe_digraph_edges()).render_code(format="networkdynamics")
    assert "_post" not in code
    assert "    c_glob = esum[1]" in code


def test_a_single_node_receives_the_post_expression_of_its_zero_input():
    code = SimulationExperiment(
        dynamics=DYNAMICS,
        network={"number_of_nodes": 1, "coupling": LINEAR},
        integration={"method": "Heun", "step_size": 0.1, "duration": 1.0},
    ).render_code(format="networkdynamics")
    assert "nw = Network(g, vertex_Generic2dOscillator, edge_zero)" in code
    assert "    c_glob = Linear_post(esum[1], (Linear_b, Linear_a,))" in code


def test_a_custom_edge_body_is_weighted_after_it_runs():
    code = SimulationExperiment.from_file(str(DOCS / "stress_on_truss.yaml")).render_code(format="networkdynamics")
    edge = _block(code, "function BeamForce_edge_g!(e_dst, v_src, v_dst, (K, L, w,), t)")
    assert edge.rstrip().endswith("e_dst[2] = Fabs * dy / d\n    e_dst .*= w\n    nothing")
    assert "function BeamForce_obsf!(obsout, u, v_src, v_dst, (K, L, w,), t)" in code


def test_a_multidimensional_coupling_is_weighted_and_post_applied_elementwise(tmp_path):
    spec = yaml.safe_load((DOCS / "diffusion_2d.yaml").read_text())
    code = SimulationExperiment.from_file(str(DOCS / "diffusion_2d.yaml")).render_code(format="networkdynamics")
    assert "    e_dst .= w .* (@. v_src .- v_dst)" in code
    assert "    dx .= esum" in code

    coupling = spec["network"]["coupling"]["DiffusiveCoupling2D"]
    coupling["parameters"] = {"k": {"name": "k", "value": 0.5}}
    coupling["post_expression"] = {"rhs": "k * gx"}
    (tmp_path / "diffusion_2d.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    code = SimulationExperiment.from_file(str(tmp_path / "diffusion_2d.yaml")).render_code(format="networkdynamics")
    assert "DiffusiveCoupling2D_post(gx, (k,)) = gx .* k" in code
    assert "    dx .= DiffusiveCoupling2D_post.(esum, Ref((DiffusiveCoupling2D_k,)))" in code


# ── An event on one line runs on its declared direction and trips both ────


def test_an_event_on_one_undirected_line_runs_on_its_declared_direction():
    """``edge_5`` is the fifth listed line, 2 -- 4, which the digraph carries as its 6th and 10th edges; the callback sits on the declared 2 → 4 only, since `on_line!` trips 4 → 2 with it."""
    code = SimulationExperiment.from_file(str(DOCS / "cascading_failure.yaml")).render_code(format="networkdynamics")

    assert _emitted_edges(code)[5] == (2, 4) and _emitted_edges(code)[9] == (4, 2)
    assert re.findall(r"add_callback!\(nw\[EIndex\((\d+)\)\], initial_perturbation_cb\)", code) == ["6"]
    assert "for i in 1:ne(g)\n    set_callback!(nw[EIndex(i)], edge_trip_cb)\nend" in code


def test_an_edge_event_trips_both_directions_of_an_undirected_line():
    """A callback changes only its own edge, so every edge affect copies its parameter changes onto the line's other direction."""
    code = SimulationExperiment.from_file(str(DOCS / "cascading_failure.yaml")).render_code(format="networkdynamics")

    edges = _emitted_edges(code)
    partner = dict(re.findall(r"(\d+) => (\d+)", code.split("line_partner = Dict{Int, Int}(", 1)[1].split("\n", 1)[0]))
    assert {edges[int(a) - 1]: edges[int(b) - 1] for a, b in partner.items()} == {(i, j): (j, i) for i, j in edges}
    affect = _block(code, "edge_trip_affect = ComponentAffect([], [:K]) do u, p, ctx")
    assert affect == "\n    p[:K] = 0\n    on_line!(p, ctx, (:K, ))"


def test_an_edge_event_on_a_generated_undirected_graph_trips_both_directions(tmp_path):
    """The graph Python builds for an undirected generator names its edges, so an event on ``all_edges`` runs on every one and each line is the pair of its two directions."""
    spec = yaml.safe_load((DOCS / "cascading_failure.yaml").read_text())
    spec["network"].pop("edges")
    spec["network"]["graph_generator"] = {"name": "cycle", "type": "Cycle"}
    del spec["events"]["initial_perturbation"]
    (tmp_path / "cascading_failure.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    code = SimulationExperiment.from_file(str(tmp_path / "cascading_failure.yaml")).render_code(format="networkdynamics")

    edges = _emitted_edges(code)
    assert edges == sorted((i + 1, (i + d) % 5 + 1) for i in range(5) for d in (1, 4))
    partner = dict(re.findall(r"(\d+) => (\d+)", code.split("line_partner = Dict{Int, Int}(", 1)[1].split("\n", 1)[0]))
    assert {edges[int(a) - 1]: edges[int(b) - 1] for a, b in partner.items()} == {(i, j): (j, i) for i, j in edges}
    assert "for i in 1:ne(g)\n    set_callback!(nw[EIndex(i)], edge_trip_cb)\nend" in code


def test_an_event_on_an_edge_the_graph_does_not_list_raises(tmp_path):
    spec = yaml.safe_load((DOCS / "cascading_failure.yaml").read_text())
    spec["events"]["initial_perturbation"]["target_component"] = "edge_8"
    (tmp_path / "cascading_failure.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    with pytest.raises(NotImplementedError, match="edge_8"):
        SimulationExperiment.from_file(str(tmp_path / "cascading_failure.yaml")).render_code(format="networkdynamics")


# ── Per-edge parameters land on their own edge ────────────────────────────


def test_per_edge_parameters_land_at_their_edges_position():
    experiment = _experiment(_probe_digraph_edges(**{"31": {"K": 0.9}, "44": {"K": 0.1}}))
    code = experiment.render_code(format="networkdynamics")

    assert _edge_parameter_lines(code) == {
        (PROBE_DIGRAPH_ORDER.index((3, 1)) + 1, "K"): 0.9,
        (PROBE_DIGRAPH_ORDER.index((4, 4)) + 1, "K"): 0.1,
    }


def test_a_fixpoint_search_starts_from_the_edge_values():
    experiment = _experiment(
        _probe_digraph_edges(**{"31": {"K": 0.9}}), execution={"find_fixpoint": True}, coupling_evaluation="per_stage"
    )
    code = experiment.render_code(format="networkdynamics")

    assert "dealias=true" in code
    assert "for (k, w) in enumerate(edge_weights)\n    set_default!(nw, EIndex(k, :weight), w)\nend" in code
    assert _edge_parameter_lines(code, r"set_default!\(nw, EIndex\((\d+), :(\w+)\), (\S+)\)") == {
        (PROBE_DIGRAPH_ORDER.index((3, 1)) + 1, "K"): 0.9
    }
    assert code.index("EIndex(") < code.index("find_fixpoint(nw)")
    assert "s.p.e[" not in code


def _stochastic(method):
    dynamics = {**DYNAMICS, "state_variables": {"V": {"noise": {"parameters": {"sigma": {"name": "sigma", "value": 0.01}}}}}}
    return SimulationExperiment(
        dynamics=dynamics, network=_probe_digraph_edges(), integration={"method": method, "step_size": 0.1, "duration": 1.0}
    )


def test_a_stochastic_run_carries_the_weights_in_its_parameters():
    code = _stochastic("Heun").render_code(format="networkdynamics")

    assert "s.p.e[1:ne(g), :weight] = edge_weights" in code
    assert "SDEProblem(nw, nw_noise!, uflat(s), tspan, pflat(s); callback=CallbackSet(hold_cb))" in code


@pytest.mark.parametrize(
    ("method", "solver", "holds"), [("Heun", "EulerHeun", True), ("euler", "EM", False), ("SOSRA", "SOSRA", True)]
)
def test_a_stochastic_run_integrates_by_its_declared_methods_sde_solver(method, solver, holds):
    """Euler is Euler–Maruyama and Heun is EulerHeun, TVB's stochastic schemes under additive noise; a name tvbo does not know is a StochasticDiffEq.jl solver the recipe names. Euler–Maruyama has one stage, so it has no coupling to hold."""
    code = _stochastic(method).render_code(format="networkdynamics")

    assert f"sol = solve(prob, {solver}(); dt=0.1, saveat=0.1)" in code
    assert ("hold_cb" in code) is holds


def test_a_stochastic_run_by_a_method_with_no_sde_counterpart_raises():
    with pytest.raises(ValueError, match="RungeKutta4thOrder.*no stochastic counterpart in StochasticDiffEq.jl"):
        _stochastic("RungeKutta4thOrder").render_code(format="networkdynamics")


def test_the_docs_data_file_sets_the_network_matrix_on_its_own_weight_parameter():
    experiment = SimulationExperiment.from_file(str(DOCS / "fitzhugh_nagumo.yaml"))
    code = experiment.render_code(format="networkdynamics")

    edges = _emitted_edges(code)
    matrix = experiment.network.matrix("weight", format="dense")
    assert edges == [(j + 1, i + 1) for j, i in sorted(zip(*np.nonzero(matrix.T), strict=True))]
    assert _emitted_weights(code) == [float(matrix[j - 1, i - 1]) for i, j in edges]
    assert "s.p.e[1:ne(g), :w] = edge_weights" in code
    assert "readdlm" not in code and "NWParameter(nw)" not in code and "pflat(p)" not in code


def test_per_edge_values_the_edge_model_does_not_carry_raise():
    """A per-edge post-expression parameter, a per-edge connectome weight parameter (``w`` of a coupling that declares none) or one the coupling does not declare has no edge parameter to land on."""
    with pytest.raises(NotImplementedError, match=r"per-edge a\b"):
        _experiment(_listed([_edge(1, 2, 1.0, True, a=0.3)], coupling=LINEAR)).render_code(format="networkdynamics")
    with pytest.raises(NotImplementedError, match=r"per-edge w\b"):
        _experiment(_listed([_edge(1, 2, 1.0, True, w=0.3)], coupling=LINEAR)).render_code(format="networkdynamics")
    with pytest.raises(NotImplementedError, match=r"per-edge undeclared\b"):
        _experiment(_listed([_edge(1, 2, 1.0, True, undeclared=0.3)])).render_code(format="networkdynamics")


def test_per_edge_parameters_on_a_graph_not_built_from_them_raise():
    network = {**_listed([_edge(1, 2, 1.0, True, K=0.3)]), "graph_generator": {"name": "ring", "type": "Ring"}}
    with pytest.raises(NotImplementedError, match="K"):
        _experiment(network).render_code(format="networkdynamics")


# ── Heterogeneous vertices land by row, not by id ─────────────────────────


def _heterogeneous_kuramoto(tmp_path, execution=None):
    """The docs' heterogeneous Kuramoto network with every node id replaced by one that is not its position."""
    spec = yaml.safe_load((DOCS / "heterogeneous_kuramoto.yaml").read_text())
    for node in spec["network"]["nodes"]:
        node["id"] = 100 - 10 * node["id"]
    if execution:
        spec["execution"] = execution
    path = tmp_path / "heterogeneous_kuramoto.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return SimulationExperiment.from_file(str(path))


def test_heterogeneous_vertices_land_by_row(tmp_path):
    code = _heterogeneous_kuramoto(tmp_path).render_code(format="networkdynamics")

    assert "vertex_array[1] = vertex_StaticNode" in code
    assert "vertex_array[5] = vertex_KuramotoInertia" in code
    assert "s.p.v[2, :omega0] = -0.3125" in code
    assert "s.p.v[8, :omega0] = 0.4375" in code
    assert not re.search(r"(vertex_array|s\.v|s\.p\.v)\[(\d{2,})", code)


def test_fixpoint_node_defaults_land_by_row(tmp_path):
    code = _heterogeneous_kuramoto(tmp_path, execution={"find_fixpoint": True}).render_code(format="networkdynamics")

    assert "set_default!(nw, VIndex(1, :theta_fix), -0.4375)" in code
    assert "set_default!(nw, VIndex(2, :omega0), -0.3125)" in code
    assert not re.search(r"VIndex\(\d{2,}", code)


# ── The coupling is evaluated per step or per stage, as declared ──────────


def test_a_per_step_coupling_is_held_across_the_stages_of_a_step():
    """Under ``coupling_evaluation: per_step``, the default, each vertex reads its summed input off parameters that a callback copies from the edges at the start of every step."""
    code = _experiment(_probe_digraph_edges(coupling=LINEAR)).render_code(format="networkdynamics")

    vertex = _block(code, "function Generic2dOscillator_f!(")
    assert "held_coupling,), t)" in vertex.split("\n", 1)[0]
    assert "    esum = (held_coupling,)" in vertex
    assert "insym = [:coupling]," in _block(code, "vertex_Generic2dOscillator = VertexModel(")
    assert "coupling_inputs = SII.getu(nw, [VIndex(i, sym) for i in 1:nv(g) for sym in (:coupling, )])" in code
    assert "held_coupling = SII.setp(nw, [VIndex(i, sym) for i in 1:nv(g) for sym in (:held_coupling, )])" in code
    assert "prob = ODEProblem(nw, uflat(s), tspan, pflat(s); add_nw_cb=hold_cb)" in code


def test_a_per_stage_coupling_single_stage_method_or_single_node_holds_nothing():
    per_stage = _experiment(_probe_digraph_edges(coupling=LINEAR), coupling_evaluation="per_stage")
    euler = _experiment(_probe_digraph_edges(coupling=LINEAR), method="Euler")
    single = SimulationExperiment(
        dynamics=DYNAMICS,
        network={"number_of_nodes": 1, "coupling": LINEAR},
        integration={"method": "Heun", "step_size": 0.1, "duration": 1.0},
    )
    for experiment in (per_stage, euler, single):
        code = experiment.render_code(format="networkdynamics")
        assert "held_" not in code and "hold_cb" not in code
        assert "prob = ODEProblem(nw, uflat(s), tspan, pflat(s))" in code


def test_a_fixpoint_search_under_a_per_step_coupling_raises():
    experiment = _experiment(_probe_digraph_edges(), execution={"find_fixpoint": True})
    with pytest.raises(NotImplementedError, match="coupling_evaluation: per_stage"):
        experiment.render_code(format="networkdynamics")


def test_a_held_multidimensional_input_is_named_and_held_per_component(tmp_path):
    spec = yaml.safe_load((DOCS / "diffusion_2d.yaml").read_text())
    spec["integration"].update(method="Heun", coupling_evaluation="per_step")
    (tmp_path / "diffusion_2d.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    code = SimulationExperiment.from_file(str(tmp_path / "diffusion_2d.yaml")).render_code(format="networkdynamics")

    assert "    esum = (held_flow_x, held_flow_phi,)" in code
    assert "insym = [:flow_x, :flow_phi]," in code
    assert "for sym in (:held_flow_x, :held_flow_phi, )" in code


def _trajectory(experiment, backend):
    """*experiment*'s ``(time, state)`` trajectory on *backend*, states by variable then node row, and its time grid."""
    import xarray as xr

    result = experiment.run(format=backend)
    data = next(
        d
        for c in (result, getattr(result, "integration", None))
        for d in (c, getattr(c, "data", None))
        if isinstance(d, xr.DataArray)
    )
    columns = [data.sel(variable=v).isel(node=k).values for v in data["variable"].values for k in range(data.sizes["node"])]
    return np.round(np.asarray(data["time"].values, dtype=float), 9), np.stack(columns, axis=-1)


@pytest.mark.julia
def test_each_coupling_evaluation_matches_tvboptims():
    """Heun on the probe digraph: NetworkDynamics.jl under ``per_step`` and ``per_stage`` each follow tvboptim's trajectory under the same declaration, and the two declarations integrate different systems."""
    pytest.importorskip("juliacall")
    pytest.importorskip("tvboptim")
    states = {"V": [0.3, -0.7, 1.1, 0.45], "W": [0.1, -0.2, 0.05, 0.3]}
    edges = [
        {
            "source": i - 1,
            "target": j - 1,
            "directed": True,
            "parameters": {"weight": {"name": "weight", "value": PROBE_W[i - 1][j - 1]}},
        }
        for i, j in PROBE_DIGRAPH_ORDER
    ]
    nodes = [{"id": k, "state": {s: v[k] for s, v in states.items()}} for k in range(4)]
    network = {"number_of_nodes": 4, "nodes": nodes, "edges": edges, "coupling": LINEAR}
    runs = {}
    for evaluation in ("per_step", "per_stage"):
        for backend in ("networkdynamics", "tvboptim"):
            runs[evaluation, backend] = _trajectory(
                _experiment(network, duration=20.0, step_size=0.05, coupling_evaluation=evaluation), backend
            )

    def gap(a, b):
        (ta, xa), (tb, xb) = runs[a], runs[b]
        common = np.intersect1d(ta, tb)
        assert len(common) >= 399
        return float(np.max(np.abs(xa[np.isin(ta, common)] - xb[np.isin(tb, common)])))

    for evaluation in ("per_step", "per_stage"):
        assert gap((evaluation, "networkdynamics"), (evaluation, "tvboptim")) < 1e-10
    assert gap(("per_step", "networkdynamics"), ("per_stage", "networkdynamics")) > 1e-6


def test_the_result_names_each_node_by_its_label_in_row_order():
    """A node's coordinate is its label, as tvboptim and TVB name theirs, or ``node_<id>`` where it declares none, at the row its id resolves to."""
    from tvbo.adapters.networkdynamics import NetworkDynamicsAdapter

    network = {
        **_probe_digraph_edges(),
        "nodes": [{"id": 40, "label": "PFC"}, {"id": 10}, {"id": 30, "label": "PPC"}, {"id": 20}],
    }
    ctx = NetworkDynamicsAdapter(_experiment(network)).render_context()

    assert NetworkDynamicsAdapter.node_labels(ctx) == ["PFC", "node_10", "PPC", "node_20"]


@pytest.mark.julia
def test_a_run_labels_its_node_axis_as_tvboptim_does():
    pytest.importorskip("juliacall")
    pytest.importorskip("tvboptim")
    import xarray as xr

    nodes = [{"id": k, "label": label, "state": {"V": 0.1 * k}} for k, label in enumerate(["A", "B", "C", "D"])]
    edges = [
        {
            "source": i - 1,
            "target": j - 1,
            "directed": True,
            "parameters": {"weight": {"name": "weight", "value": PROBE_W[i - 1][j - 1]}},
        }
        for i, j in PROBE_DIGRAPH_ORDER
    ]
    network = {"number_of_nodes": 4, "nodes": nodes, "edges": edges, "coupling": LINEAR}
    labels = {}
    for backend in ("networkdynamics", "tvboptim"):
        result = _experiment(network).run(format=backend)
        data = next(
            d
            for c in (result, getattr(result, "integration", None))
            for d in (c, getattr(c, "data", None))
            if isinstance(d, xr.DataArray)
        )
        labels[backend] = [str(v) for v in data["node"].values]

    assert labels["networkdynamics"] == labels["tvboptim"] == ["A", "B", "C", "D"]


# ── Coupling expressions resolve their names as every backend does ────────


def _coupled(dynamics, coupling, **integration):
    edges = [
        {"source": i, "target": j, "directed": True, "parameters": {"weight": {"name": "weight", "value": 0.5 + i + 2 * j}}}
        for i in range(3)
        for j in range(3)
        if i != j
    ]
    network = {
        "number_of_nodes": 3,
        "edges": edges,
        "coupling": {coupling: {"name": coupling, "iri": f"tvbo:{coupling}", "delayed": False}},
    }
    return SimulationExperiment(
        dynamics={"name": dynamics, "iri": f"tvbo:{dynamics}"},
        network=network,
        integration={"method": "Euler", "step_size": 0.1, "duration": 1.0, **integration},
    )


def test_an_indexed_source_state_is_the_transmitted_state_it_indexes():
    """SigmoidalJansenRit's ``x_j[0] - x_j[1]`` is ``y1 - y2`` of the source, JansenRit's two transmitted states, which the vertex outputs in that order."""
    code = _coupled("JansenRit", "SigmoidalJansenRit").render_code(format="networkdynamics")

    edge = _block(code, "function SigmoidalJansenRit_edge_g!(")
    assert "exp(r .* (midpoint .+ v_src[2] .- v_src[1]))" in edge and "][0]" not in edge
    assert "g = StateMask(2:3)," in _block(code, "vertex_JansenRit = VertexModel(")


def test_the_identity_pre_expression_reads_the_declared_state_at_the_source():
    """FastLinearCoupling's ``local_states`` pre-expression is tvboptim's vectorised identity over ``S``: each edge sends the source's ``S``."""
    code = _coupled("ReducedWongWang", "FastLinearCoupling").render_code(format="networkdynamics")

    assert "    e_dst[1] = w * (v_src[1])" in _block(code, "function FastLinearCoupling_edge_g!(")
    assert "local_states" not in code
    assert "FastLinearCoupling_post(gx, (G, b,)) = " in code


def test_a_name_no_vertex_output_carries_raises():
    """``W`` is a state the coupling declares and the vertex does not output, since Generic2dOscillator transmits ``V`` alone; ``U_j`` names nothing."""
    unsent = {
        "Unsent": {
            "name": "Unsent",
            "delayed": False,
            "incoming_states": ["V"],
            "local_states": ["W"],
            "pre_expression": {"rhs": "W_j - x_i"},
        }
    }
    with pytest.raises(NotImplementedError, match="reads the state `W` at the source"):
        _experiment(_probe_digraph_edges(coupling=unsent)).render_code(format="networkdynamics")
    unbound = {"Unbound": {"name": "Unbound", "delayed": False, "pre_expression": {"rhs": "U_j - x_i"}}}
    with pytest.raises(NotImplementedError, match="reads `U_j`, which is neither"):
        _experiment(_probe_digraph_edges(coupling=unbound)).render_code(format="networkdynamics")


# ── Each node starts from its declared state and parameters ───────────────


def test_a_homogeneous_network_sets_each_nodes_declared_state_and_parameters():
    nodes = [
        {"id": 40, "state": {"V": 0.25}, "parameters": [{"name": "a", "value": -1.5}]},
        {"id": 10},
        {"id": 30, "state": {"V": -0.5, "W": 0.125}},
        {"id": 20, "parameters": [{"name": "undeclared", "value": 3.0}]},
    ]
    network = {**_probe_digraph_edges(), "nodes": nodes}
    code = _experiment(network).render_code(format="networkdynamics")

    assert "s.v[1, :V] = 0.25" in code and "s.v[3, :V] = -0.5" in code and "s.v[3, :W] = 0.125" in code
    assert "s.p.v[1, :a] = -1.5" in code
    assert "undeclared" not in code
    assert code.index("s.v[1:nv(g), :V] .=") < code.index("s.v[1, :V] = 0.25")
