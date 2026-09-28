"""The coupling the NetworkDynamics.jl source integrates: node i receives post(sum_j w_ij * pre(x_i, x_j)), the coupling every other backend integrates, for every form a connectome is declared in.

Every graph is a `SimpleDiGraph` with the `Directed` edge model, and every edge carries the connection weight. ND.jl addresses an edge parameter by the edge's position in ``edges(g)``, so a value lands on the right connection only when it is emitted in the order Graphs.jl iterates the graph. That order was read off Graphs.jl itself (``julia --project=.venv/julia_env``): ``edges(SimpleDiGraph(...))`` visits ``(1,2), (1,4), (2,3), (3,1), (3,4), (4,2), (4,4)`` for `PROBE_W`, whatever order the edges were added in.

Each weight is checked against the network's own matrix, `Network.matrix("weight")`, which is target-by-source: the edge ``i → j`` carries ``matrix[j, i]``. The generated scripts were also run on the standalone Julia, where the RHS each node receives at t=0 equals the closed form for the curated Linear, Sigmoidal and Kuramoto couplings and for a coupling declaring its own ``w``.
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


def _experiment(network, execution=None):
    spec = {"dynamics": DYNAMICS, "network": network, "integration": {"method": "Heun", "step_size": 0.1, "duration": 1.0}}
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
    assert "s.p.e[1:ne(g), :w] = edge_weights" in code
    assert "pflat(s)" in code and "pflat(p)" not in code


def test_directed_listed_edges_set_their_weights_in_edges_order():
    experiment = _experiment(_probe_digraph_edges())
    code = experiment.render_code(format="networkdynamics")

    assert "g = SimpleDiGraph(4)" in code
    assert _emitted_edges(code) == PROBE_DIGRAPH_ORDER
    assert _emitted_weights(code) == [_matrix_weight(experiment, i, j) for i, j in PROBE_DIGRAPH_ORDER]
    assert _emitted_weights(code) == [PROBE_W[i - 1][j - 1] for i, j in PROBE_DIGRAPH_ORDER]
    assert "s.p.e[1:ne(g), :w] = edge_weights" in code


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


def test_a_generated_graph_is_converted_to_a_digraph():
    ring = {"number_of_nodes": 6, "graph_generator": {"name": "cycle", "type": "Cycle"}, "coupling": LINEAR}
    code = _experiment(ring).render_code(format="networkdynamics")
    assert "g = SimpleDiGraph(cycle_graph(6))" in code
    assert "g = Directed(Linear_edge_g!)" in code
    assert "edge_weights" not in code


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


def test_a_coupling_declaring_its_weight_is_weighted_once():
    """A declared ``w`` or ``weight`` is the parameter the connectome sets: the pre-expression already carries it, so nothing multiplies it in again."""
    for coupling, name, weight, pre in (
        (WDIFF, "WDiff", "w", "K .* w .* (v_src[1] .- v_dst[1])"),
        (WLIN, "WLin", "weight", "gain .* weight .* v_src[1]"),
    ):
        code = _experiment(_probe_digraph_edges(coupling=coupling)).render_code(format="networkdynamics")
        edge = _block(code, f"function {name}_edge_g!(")
        assert f"    e_dst[1] = {pre}" in edge
        assert f"{weight} * (" not in edge
        psym = _block(code, f"edge_{name} = EdgeModel(").split("psym = ", 1)[1].split("\n", 1)[0]
        assert re.findall(r":(\w+) =>", psym) == [weight, "K" if name == "WDiff" else "gain"]
        assert f"s.p.e[1:ne(g), :{weight}] = edge_weights" in code


def test_a_weight_the_pre_expression_misreads_raises():
    unread = {
        "Unread": {
            "name": "Unread",
            "delayed": False,
            "parameters": {"w": {"name": "w", "value": 1.0}},
            "pre_expression": {"rhs": "x_j"},
        }
    }
    with pytest.raises(NotImplementedError, match="does not read it"):
        _experiment(_probe_digraph_edges(coupling=unread)).render_code(format="networkdynamics")
    undeclared = {"Undeclared": {"name": "Undeclared", "delayed": False, "pre_expression": {"rhs": "w * x_j"}}}
    with pytest.raises(NotImplementedError, match="without declaring it"):
        _experiment(_probe_digraph_edges(coupling=undeclared)).render_code(format="networkdynamics")


def test_the_post_expression_is_applied_once_at_the_node():
    """Linear's ``a * gx + b`` is defined once, called once by the vertex on the summed input, and its ``a``, ``b`` sit on the vertex beside the model's own."""
    code = _experiment(_probe_digraph_edges(coupling=LINEAR)).render_code(format="networkdynamics")

    assert code.count("Linear_post(gx, (b, a,)) = b .+ a .* gx") == 1
    assert code.index("Linear_post(gx,") < code.index("function Generic2dOscillator_f!(")
    vertex = _block(code, "function Generic2dOscillator_f!(")
    assert vertex.startswith("dx, x, esum, (I, a, alpha, b, beta, c, d, e, f, g, gamma, tau, Linear_b, Linear_a,), t)")
    assert "    c_glob = Linear_post(esum[1], (Linear_b, Linear_a,))" in vertex
    assert code.count("Linear_post(") == 2
    model = _block(code, "vertex_Generic2dOscillator = VertexModel(")
    assert ":a => -2.0" in model and ":b => -10.0" in model
    assert ":Linear_b => 0.0, :Linear_a => 0.00390625]" in model
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
    assert ":Both_G => 0.5, :Both_c0 => 0.1]" in _block(code, "vertex_Generic2dOscillator = VertexModel(")
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
    assert "    e_dst .= w .* (v_src .- v_dst)" in code
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


def test_an_edge_event_on_a_generated_undirected_graph_raises(tmp_path):
    spec = yaml.safe_load((DOCS / "cascading_failure.yaml").read_text())
    spec["network"].pop("edges")
    spec["network"]["graph_generator"] = {"name": "cycle", "type": "Cycle"}
    del spec["events"]["initial_perturbation"]
    (tmp_path / "cascading_failure.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    with pytest.raises(NotImplementedError, match="generated graph"):
        SimulationExperiment.from_file(str(tmp_path / "cascading_failure.yaml")).render_code(format="networkdynamics")


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
    experiment = _experiment(_probe_digraph_edges(**{"31": {"K": 0.9}}), execution={"find_fixpoint": True})
    code = experiment.render_code(format="networkdynamics")

    assert "dealias=true" in code
    assert "for (k, w) in enumerate(edge_weights)\n    set_default!(nw, EIndex(k, :w), w)\nend" in code
    assert _edge_parameter_lines(code, r"set_default!\(nw, EIndex\((\d+), :(\w+)\), (\S+)\)") == {
        (PROBE_DIGRAPH_ORDER.index((3, 1)) + 1, "K"): 0.9
    }
    assert code.index("EIndex(") < code.index("find_fixpoint(nw)")
    assert "s.p.e[" not in code


def test_a_stochastic_run_carries_the_weights_in_its_parameters():
    dynamics = {**DYNAMICS, "state_variables": {"V": {"noise": {"parameters": {"sigma": {"name": "sigma", "value": 0.01}}}}}}
    experiment = SimulationExperiment(
        dynamics=dynamics, network=_probe_digraph_edges(), integration={"method": "Heun", "step_size": 0.1, "duration": 1.0}
    )
    code = experiment.render_code(format="networkdynamics")

    assert "s.p.e[1:ne(g), :w] = edge_weights" in code
    assert "SDEProblem(nw, nw_noise!, uflat(s), tspan, pflat(s))" in code


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
    """A per-edge post-expression parameter, a per-edge ``w`` or one the coupling does not declare has no edge parameter to land on."""
    with pytest.raises(NotImplementedError, match=r"per-edge a\b"):
        _experiment(_listed([_edge(1, 2, 1.0, True, a=0.3)], coupling=LINEAR)).render_code(format="networkdynamics")
    with pytest.raises(NotImplementedError, match=r"per-edge w\b"):
        _experiment(_listed([_edge(1, 2, 1.0, True, w=0.3)])).render_code(format="networkdynamics")
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
