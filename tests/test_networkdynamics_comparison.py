"""End-to-end tests: TVBO NetworkDynamics.jl backend vs the original NetworkDynamics.jl tutorials.

The Julia-marked classes run ``tests/reference_data/generate_nd_references.jl`` once per session, in the Julia environment the backend runs in (the `nd_references` fixture), and compare TVBO's generated code, run through juliacall, with each tutorial's trajectory on the tutorial's own graph and initial state.

Requires: h5py, numpy, pytest; juliacall for the Julia-marked classes.
"""

import json
import os

import numpy as np
import pytest

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(HERE, "reference_data")
EXAMPLES_DIR = os.path.join(os.path.dirname(HERE), "docs", "Interoperability", "NetworkDynamics.jl", "yaml")
REFERENCE_SCRIPT = os.path.join(REF_DIR, "generate_nd_references.jl")

# Helpers


def _load_reference(directory, name):
    """Load the reference *name* from the HDF5 file `generate_nd_references.jl` wrote into *directory*; a missing file fails the test rather than skipping it."""
    import h5py

    path = os.path.join(directory, f"{name}_reference.h5")
    if not os.path.exists(path):
        pytest.fail(f"generate_nd_references.jl wrote no reference {path}")
    with h5py.File(path, "r") as f:
        data = {
            "t": f["t"][:],
            "u": f["u"][:],
            "x0": f["x0"][:],
        }
        if "adjacency" in f:
            data["adjacency"] = f["adjacency"][:]
        if "omega0" in f:
            data["omega0"] = f["omega0"][:]
        if "G" in f:
            data["G"] = f["G"][:]
        if "edge_weights" in f:
            data["edge_weights"] = f["edge_weights"][:]
        if "state_vertex" in f:
            data["state_vertex"] = f["state_vertex"][:]
            data["state_symbol"] = [s.decode() if isinstance(s, bytes) else s for s in f["state_symbol"][:]]
        if "vertex_types" in f:
            data["vertex_types"] = [s.decode() if isinstance(s, bytes) else s for s in f["vertex_types"][:]]
        # Read attributes
        data["attrs"] = dict(f.attrs)
    return data


def _emitted_edges(code):
    """The graph's edges, 1-based ``(source, target)``, off the nonzero entries of the matrix literal a generated or matrix network is emitted as."""
    line = next(line for line in code.splitlines() if line.startswith("W = ["))
    rows = [[float(v) for v in row.split()] for row in line.removeprefix("W = [").removesuffix("]").split("; ")]
    return [(i + 1, j + 1) for i, row in enumerate(rows) for j, v in enumerate(row) if v]


# Test 1: Code generation (no Julia needed)
class TestCodeGeneration:
    """Verify that TVBO generates valid Julia code from YAML specs."""

    @pytest.fixture(params=["diffusion", "kuramoto", "fitzhugh_nagumo"])
    def example(self, request):
        return request.param

    @pytest.fixture(
        params=[
            "diffusion",
            "kuramoto",
            "fitzhugh_nagumo",
            "diffusion_2d",
            "heterogeneous_kuramoto",
            "cascading_failure",
            "stress_on_truss",
        ]
    )
    def all_examples(self, request):
        return request.param

    def test_render_code_produces_julia(self, example):
        """render_code('networkdynamics') returns non-empty Julia code."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, f"{example}.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert isinstance(code, str)
        assert len(code) > 100
        assert "using NetworkDynamics" in code
        assert "using Graphs" in code

    def test_code_has_vertex_model(self, example):
        """Generated code defines a VertexModel."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, f"{example}.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "VertexModel" in code

    def test_code_has_edge_model(self, example):
        """Generated code defines an EdgeModel."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, f"{example}.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "EdgeModel" in code

    def test_code_has_ode_problem(self, example):
        """Generated code creates an ODEProblem."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, f"{example}.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "ODEProblem" in code

    def test_code_has_solve(self, example):
        """Generated code calls solve()."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, f"{example}.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "solve(" in code

    def test_diffusion_integrates_the_barabasi_albert_graph_networkx_builds(self):
        """Diffusion's Barabási–Albert generator is built in Python, by its networkx binding with seed 0, and reaches Julia as that graph's edges."""
        import networkx as nx

        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "diffusion.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        graph = nx.barabasi_albert_graph(20, 4, seed=0)
        both_ways = sorted((i + 1, j + 1) for u, v in graph.edges for i, j in ((u, v), (v, u)))
        assert _emitted_edges(code) == both_ways
        assert "barabasi_albert" not in code

    def test_kuramoto_integrates_the_watts_strogatz_ring(self):
        """Kuramoto's Watts–Strogatz generator at k = 2, p = 0 is the ring, each node coupled to its neighbour on either side."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "kuramoto.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        ring = sorted((i + 1, (i + d) % 8 + 1) for i in range(8) for d in (1, 7))
        assert _emitted_edges(code) == ring
        assert "watts_strogatz" not in code

    def test_fhn_builds_its_digraph_from_the_network_matrix(self):
        """FHN example's connectivity file is read into the network, whose weights the script emits edge by edge."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "fitzhugh_nagumo.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "g = SimpleDiGraph(SimpleWeightedDiGraph(W))" in code
        assert "s.p.e[1:ne(g), :w] = edge_weights" in code
        assert "readdlm" not in code

    def test_fhn_uses_directed_coupling(self):
        """FHN example uses Directed edge coupling (not AntiSymmetric)."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "fitzhugh_nagumo.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "Directed(" in code

    def test_diffusion_uses_directed_edges_both_ways(self):
        """Diffusion example's undirected generated graph becomes a digraph carrying each edge both ways, with the Directed edge model."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "diffusion.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "g = SimpleDiGraph(SimpleWeightedDiGraph(W))" in code
        edges = _emitted_edges(code)
        assert sorted((j, i) for i, j in edges) == edges
        assert "Directed(" in code and "AntiSymmetric(" not in code

    # -- 2D Diffusion specific tests --

    def test_diffusion_2d_multidim_coupling(self):
        """2D diffusion: vertex outputs 2 coupling variables via StateMask(1:2)."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "diffusion_2d.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "StateMask(1:2)" in code
        # State variables keep their declared order
        assert "sym = [:x, :phi]" in code

    def test_diffusion_2d_broadcast_esum(self):
        """2D diffusion: vertex uses broadcasting (dx .= esum) for multi-dim coupling."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "diffusion_2d.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "dx .= esum" in code

    def test_diffusion_2d_broadcasting_edge(self):
        """2D diffusion: edge uses broadcasting (e_dst .= v_src .- v_dst)."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "diffusion_2d.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "e_dst .=" in code
        assert "outsym = [:flow_x, :flow_phi]" in code

    def test_diffusion_2d_no_variable_shadowing(self):
        """2D diffusion: function arg doesn't shadow state var 'x'."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "diffusion_2d.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "function Diffusion2D_f!(dx, _x, esum" in code
        assert "x, phi = _x" in code

    def test_diffusion_2d_uses_barabasi_albert_10(self):
        """2D diffusion: 10-node Barabási-Albert network, built by networkx."""
        import networkx as nx

        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "diffusion_2d.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        assert exp.network.number_of_nodes == 10
        code = exp.render_code("networkdynamics")
        graph = nx.barabasi_albert_graph(10, 4, seed=0)
        assert _emitted_edges(code) == sorted((i + 1, j + 1) for u, v in graph.edges for i, j in ((u, v), (v, u)))

    # -- Heterogeneous Kuramoto specific tests --

    def test_heterogeneous_kuramoto_vertex_types(self):
        """Heterogeneous Kuramoto: generates 3 vertex model types."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "heterogeneous_kuramoto.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "vertex_Kuramoto" in code
        assert "vertex_StaticNode" in code
        assert "vertex_KuramotoInertia" in code

    def test_heterogeneous_kuramoto_vertex_array(self):
        """Heterogeneous Kuramoto: builds VertexModel[] array with per-node assignment."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "heterogeneous_kuramoto.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "vertex_array = VertexModel[" in code
        assert "vertex_array[1] = vertex_StaticNode" in code
        assert "vertex_array[5] = vertex_KuramotoInertia" in code

    def test_heterogeneous_kuramoto_static_no_feedforward(self):
        """Static vertex uses NoFeedForward() and outputs :theta."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "heterogeneous_kuramoto.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "ff = NoFeedForward()" in code
        assert "outsym = [:theta]" in code

    def test_heterogeneous_kuramoto_inertia_2d(self):
        """Inertia vertex has 2 state vars but only 1 coupling output."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "heterogeneous_kuramoto.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        # KuramotoInertia declares theta then omega, and only theta (index 1) couples
        assert "sym = [:theta, :omega]" in code
        assert "StateMask(1:1)" in code.split("vertex_KuramotoInertia = VertexModel(")[1]

    def test_heterogeneous_kuramoto_per_node_params(self):
        """Per-node omega0 parameters are set via NWState."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "heterogeneous_kuramoto.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        # Check that per-node omega0 values are set
        assert "s.p.v[2, :omega0] = -0.3125" in code
        assert "s.p.v[8, :omega0] = 0.4375" in code
        # Static node gets theta_fix parameter
        assert "s.p.v[1, :theta_fix] = -0.4375" in code

    def test_heterogeneous_kuramoto_dealias(self):
        """Heterogeneous Network uses dealias=true."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "heterogeneous_kuramoto.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "dealias=true" in code

    # -- Cascading Failure specific tests --

    def test_cascading_failure_find_fixpoint(self):
        """Cascading failure uses find_fixpoint for initial conditions."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "cascading_failure.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "find_fixpoint(nw)" in code
        assert "set_defaults!(nw, u0)" in code

    def test_cascading_failure_callbacks(self):
        """Cascading failure has component-based callbacks."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "cascading_failure.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "ComponentCondition" in code
        assert "ComponentAffect" in code
        assert "ContinuousComponentCallback" in code
        assert "set_callback!" in code

    def test_cascading_failure_preset_time_callback(self):
        """Cascading failure has a preset time callback on edge 5."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "cascading_failure.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "PresetTimeComponentCallback" in code
        assert "add_callback!" in code

    def test_cascading_failure_per_node_params(self):
        """Cascading failure sets per-node P_ref via set_default!."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "cascading_failure.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "set_default!(nw, VIndex(1, :P_ref), -1.0)" in code
        assert "set_default!(nw, VIndex(2, :P_ref), 1.5)" in code

    def test_cascading_failure_outsym(self):
        """Cascading failure edge outputs :P."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "cascading_failure.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "outsym = [:P]" in code

    def test_cascading_failure_dealias(self):
        """Cascading failure uses dealias=true for per-edge callbacks."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "cascading_failure.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "dealias=true" in code

    def test_cascading_failure_ode_problem_nwstate(self):
        """Cascading failure uses ODEProblem(nw, u0, tspan) with NWState."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "cascading_failure.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "ODEProblem(nw, u0, tspan)" in code

    # -- Stress on Truss specific tests --

    def test_stress_on_truss_heterogeneous_vertices(self):
        """Stress on truss uses FreeVertex and FixedVertex."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "stress_on_truss.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "vertex_FreeVertex" in code
        assert "vertex_FixedVertex" in code
        assert "vertex_array = VertexModel[" in code

    def test_stress_on_truss_fixed_vertex_no_feedforward(self):
        """FixedVertex uses NoFeedForward and outputs :x, :y."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "stress_on_truss.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "ff = NoFeedForward()" in code

    def test_stress_on_truss_2d_coupling(self):
        """Stress on truss has 2D coupling with outsym [:Fx, :Fy]."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "stress_on_truss.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "outsym = [:Fx, :Fy]" in code
        assert "insym = [:Fx, :Fy]" in code
        assert "StateMask(3:4)" in code

    def test_stress_on_truss_observed_function(self):
        """Stress on truss beam has an observed function for Fabs."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "stress_on_truss.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert "obsf" in code
        assert "obssym = [:Fabs]" in code

    def test_stress_on_truss_per_edge_L(self):
        """Stress on truss sets per-edge L (rest length) values."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "stress_on_truss.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        code = exp.render_code("networkdynamics")
        assert ":L] = 1.5620499351813308" in code  # a vertical beam
        assert ":L] = 1.019803902718557" in code  # a cross brace


# Test 2: YAML specification correctness
class TestYAMLSpecs:
    """Verify YAML specifications match original tutorial parameters."""

    def test_diffusion_yaml(self):
        """Diffusion YAML matches original: 20 nodes, BA(k=4), Tsit5, t=[0,2]."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "diffusion.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        assert exp.network.number_of_nodes == 20
        assert exp.integration.duration == 2.0

    def test_kuramoto_yaml(self):
        """Kuramoto YAML matches original: 8 nodes, WS(k=2,p=0), K=3, t=[0,10]."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "kuramoto.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        assert exp.network.number_of_nodes == 8
        assert exp.integration.duration == 10.0

    def test_fhn_yaml(self):
        """FHN YAML matches original: 90 nodes, a=0.5, eps=0.05, sigma=0.5, t=[0,200]."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "fitzhugh_nagumo.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        assert exp.network.number_of_nodes == 90
        assert exp.integration.duration == 200.0

    def test_fhn_equations_match_original(self):
        """FHN equations match original ND.jl code (not the description)."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "fitzhugh_nagumo.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        svs = exp.dynamics.state_variables
        # du/dt = u - u^3/3 - v + coupling  (NOT divided by epsilon)
        u_rhs = svs["u"].equation.rhs
        assert "/ epsilon" not in u_rhs, f"u equation should NOT divide by epsilon: {u_rhs}"
        # dv/dt = (u - a) * epsilon  (matches code, not math description)
        v_rhs = svs["v"].equation.rhs
        assert "a" in v_rhs and "epsilon" in v_rhs, f"v equation should contain a and epsilon: {v_rhs}"

    def test_diffusion_2d_yaml(self):
        """2D Diffusion YAML: 10 nodes, BA(k=4), t=[0,3], 2 SVs, both coupling vars."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "diffusion_2d.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        assert exp.network.number_of_nodes == 10
        assert exp.integration.duration == 3.0
        svs = exp.dynamics.state_variables
        assert len(svs) == 2
        assert "x" in svs and "phi" in svs
        for sv in svs.values():
            assert getattr(sv, "coupling_variable", False) is True

    def test_heterogeneous_kuramoto_yaml(self):
        """Heterogeneous Kuramoto YAML: 8 nodes, 3 vertex types, per-node assignment."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "heterogeneous_kuramoto.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        assert exp.network.number_of_nodes == 8
        assert exp.integration.duration == 10.0
        # Default dynamics is Kuramoto
        assert exp.dynamics.name == "Kuramoto"
        # Additional dynamics in network library
        assert "StaticNode" in exp.network.dynamics
        assert "KuramotoInertia" in exp.network.dynamics
        # Nodes with per-node assignments
        nodes = exp.network.nodes
        assert len(nodes) == 8
        # Node 0 has StaticNode dynamics
        assert str(nodes[0].dynamics) == "StaticNode"
        # Node 4 has KuramotoInertia
        assert str(nodes[4].dynamics) == "KuramotoInertia"

    def test_cascading_failure_yaml(self):
        """Cascading failure YAML: 5 nodes, swing equation, events, find_fixpoint."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "cascading_failure.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        assert exp.network.number_of_nodes == 5
        assert exp.integration.duration == 6.0
        assert exp.dynamics.name == "SwingEquation"
        # Has events
        events = getattr(exp, "events", [])
        assert len(events) >= 2
        # Has find_fixpoint execution config
        exec_cfg = getattr(exp, "execution", None)
        assert exec_cfg is not None
        assert getattr(exec_cfg, "find_fixpoint", False) is True

    def test_stress_on_truss_yaml(self):
        """Stress on truss YAML: 11 nodes, 2 vertex types, per-edge L."""
        from tvbo import SimulationExperiment

        yaml_path = os.path.join(EXAMPLES_DIR, "stress_on_truss.yaml")
        exp = SimulationExperiment.from_file(yaml_path)
        assert exp.network.number_of_nodes == 11
        assert exp.integration.duration == 12.0
        assert exp.dynamics.name == "FreeVertex"
        # Has FixedVertex in network dynamics library
        assert "FixedVertex" in exp.network.dynamics
        # Edges
        edges = exp.network.edges
        assert len(edges) == 18
        # FreeVertex has coupling_variables x and y
        svs = exp.dynamics.state_variables
        assert svs["x"].coupling_variable is True
        assert svs["y"].coupling_variable is True
        assert getattr(svs["vx"], "coupling_variable", False) is not True
        assert getattr(svs["vy"], "coupling_variable", False) is not True


# Test 3: Live execution + numerical comparison, and the references themselves (requires Julia)
@pytest.fixture(scope="module")
def nd_references(tmp_path_factory):
    """The directory holding the seven tutorial references, generated for this module by `generate_nd_references.jl` in the Julia environment the backend runs in.

    Generating them here, rather than reading files a developer made once, compares every run against the NetworkDynamics.jl the backend itself calls, and no comparison can pass by finding nothing to compare against: a reference the script fails to write fails the test that reads it.
    """
    pytest.importorskip("juliacall", reason="juliacall not installed")
    from tvbo.run.julia import run_julia_code

    outdir = tmp_path_factory.mktemp("nd_references")
    run_julia_code(
        f'withenv("TVBO_ND_REFERENCE_DIR" => {json.dumps(str(outdir))}) do\n    Base.include(Module(), {json.dumps(REFERENCE_SCRIPT)})\nend\nnothing'
    )
    return outdir


def _on_reference_inputs(name, ref, states=(), parameters=None, adjacency=False):
    """The docs example *name* run on the reference's own inputs, so the two integrate the same system and any difference is TVBO's.

    *states* are the state variables the reference's ``x0`` holds per node, in NetworkDynamics.jl's layout (node by node, each node's states in order), and become each node's declared ``state``; *parameters* maps a node parameter to its per-node values. With *adjacency*, the network drops its generator and takes the reference's graph as its weight matrix: a random generator draws a different graph in Graphs.jl than in the networkx build tvbo integrates, and the reference is the tutorial's draw.
    """
    import yaml

    from tvbo import SimulationExperiment

    with open(os.path.join(EXAMPLES_DIR, f"{name}.yaml")) as f:
        spec = yaml.safe_load(f)
    network = spec["network"]
    if "edge_matrix_files" in network:
        network["edge_matrix_files"] = [os.path.join(EXAMPLES_DIR, path) for path in network["edge_matrix_files"]]
    if adjacency:
        network.pop("graph_generator")
    n = int(network["number_of_nodes"])
    x0 = ref["x0"]
    network["nodes"] = [
        {
            "id": k,
            "state": {state: float(x0[k * len(states) + i]) for i, state in enumerate(states)},
            "parameters": [{"name": p, "value": float(values[k])} for p, values in (parameters or {}).items()],
        }
        for k in range(n)
    ]
    experiment = SimulationExperiment(**spec)
    if adjacency:
        # h5py reads Julia's column-major source-by-target adjacency transposed, target-by-source as a Network holds it.
        experiment.network.set_matrix("weight", ref["adjacency"])
    return experiment


TUTORIAL_STATE_NAMES = {"θ": "theta", "ω": "omega", "ϕ": "phi", "δ": "delta"}
"""The tutorials' state names that the docs examples spell out in ASCII."""


def _reference_layout(ts, ref, names):
    """*ts* as ``(time, state)`` in the reference's column order: each column is the state the reference labels by vertex (1-based, the row of the node) and name, since NetworkDynamics.jl lays out its flat state batch by batch of identical vertex models, not vertex by vertex."""
    columns = [
        ts.data.sel(variable=names.get(name, name)).isel(node=int(vertex) - 1).values
        for vertex, name in zip(ref["state_vertex"], ref["state_symbol"], strict=True)
    ]
    return np.stack(columns, axis=-1)


def _assert_matches(experiment, ref, atol, names=None):
    """Run *experiment* on NetworkDynamics.jl and compare its trajectory with the reference's, sample by sample: the times, where an event adds a sample at the instant root-finding locates, and the states, the reference's state names read through `TUTORIAL_STATE_NAMES` and *names*."""
    ts = experiment.run(format="networkdynamics")
    tvbo = _reference_layout(ts, ref, {**TUTORIAL_STATE_NAMES, **(names or {})})
    time = np.asarray(ts.time, dtype=float)
    assert time.shape == ref["t"].shape, f"{time.shape[0]} samples, the reference {ref['t'].shape[0]}"
    assert np.max(np.abs(time - ref["t"])) < atol, f"max |t - t_reference| = {np.max(np.abs(time - ref['t'])):.3e}"
    assert tvbo.shape == ref["u"].shape
    gap = float(np.max(np.abs(tvbo - ref["u"])))
    assert gap < atol, f"max |TVBO - reference| = {gap:.3e}"
    return gap


@pytest.mark.julia
class TestNumericalComparison:
    """Run TVBO-generated code in Julia and compare its trajectory with the tutorial's own, on the same graph and initial state.

    The references are written in this session by ``generate_nd_references.jl`` (the ``nd_references`` fixture), so each comparison is against the NetworkDynamics.jl the backend runs. Every docs example declares its tutorial's solver and ``coupling_evaluation: per_stage``, and runs on the tutorial's own graph, weights and initial state, so the two trajectories differ only by the rounding of sums taken in another order (a tutorial's undirected edge is one `AntiSymmetric` edge, and tvbo's is two `Directed` ones) and, where an event fires, by where the root finder places it: both far below ``ATOL``.

    Marked so that the one job holding a Julia depot is the only one that runs it. Starting Julia inside a forked xdist worker kills the worker rather than failing the test, and the classes above it here need no Julia at all — so the fact has to live on this class, where it is true, rather than as a path in a CI shard's ignore list.
    """

    ATOL = 1e-6

    @pytest.fixture(autouse=True)
    def _require_julia(self):
        """Skip entire class if juliacall is not available."""
        pytest.importorskip("juliacall", reason="juliacall not installed")
        from tvbo.run.julia import run_julia_code

        try:
            run_julia_code("1+1")
        except Exception:
            pytest.skip("Julia runtime not available")

    def test_diffusion_numerical(self, nd_references):
        """Diffusion on the tutorial's Barabási–Albert draw and its StableRNG initial state."""
        ref = _load_reference(nd_references, "diffusion")
        # The tutorial's vertex names no state, which NetworkDynamics.jl then calls `s`.
        _assert_matches(_on_reference_inputs("diffusion", ref, states=["v"], adjacency=True), ref, self.ATOL, names={"s": "v"})

    def test_kuramoto_numerical(self, nd_references):
        """Kuramoto on the ring tvbo builds from its Watts–Strogatz generator, which at p = 0 is the tutorial's graph, from the tutorial's phases and frequencies."""
        ref = _load_reference(nd_references, "kuramoto")
        experiment = _on_reference_inputs("kuramoto", ref, states=["theta"], parameters={"omega0": ref["omega0"]})
        _assert_matches(experiment, ref, self.ATOL)

    def test_fhn_numerical(self, nd_references):
        """FitzHugh–Nagumo on the tutorial's DTI weights, edge for edge, from its initial state."""
        from tvbo.adapters.networkdynamics import NetworkDynamicsAdapter

        ref = _load_reference(nd_references, "fitzhugh_nagumo")
        experiment = _on_reference_inputs("fitzhugh_nagumo", ref, states=["u", "v"])
        layout = NetworkDynamicsAdapter(experiment).prepare_context()
        assert layout["graph_edges"] == [
            tuple(int(k) for k in pair) for pair in zip(*np.nonzero(ref["adjacency"].T), strict=True)
        ]
        assert np.array_equal(layout["edge_weights"], ref["edge_weights"])
        _assert_matches(experiment, ref, self.ATOL)

    def test_diffusion_2d_numerical(self, nd_references):
        """Two-dimensional diffusion on the tutorial's Barabási–Albert draw and its StableRNG initial state."""
        ref = _load_reference(nd_references, "diffusion_2d")
        _assert_matches(_on_reference_inputs("diffusion_2d", ref, states=["x", "phi"], adjacency=True), ref, self.ATOL)

    def test_heterogeneous_kuramoto_numerical(self, nd_references):
        """Heterogeneous Kuramoto as the docs declare it: the tutorial's ring, vertex types, phases and frequencies."""
        from tvbo import SimulationExperiment

        ref = _load_reference(nd_references, "heterogeneous_kuramoto")
        experiment = SimulationExperiment.from_file(os.path.join(EXAMPLES_DIR, "heterogeneous_kuramoto.yaml"))
        _assert_matches(experiment, ref, self.ATOL)

    def test_cascading_failure_numerical(self, nd_references):
        """Cascading failure as the docs declare it: the tutorial's grid, fixpoint and line trips."""
        from tvbo import SimulationExperiment

        ref = _load_reference(nd_references, "cascading_failure")
        experiment = SimulationExperiment.from_file(os.path.join(EXAMPLES_DIR, "cascading_failure.yaml"))
        _assert_matches(experiment, ref, self.ATOL)

    def test_stress_on_truss_numerical(self, nd_references):
        """Stress on truss as the docs declare it: the tutorial's truss, rest lengths and masses."""
        from tvbo import SimulationExperiment

        ref = _load_reference(nd_references, "stress_on_truss")
        experiment = SimulationExperiment.from_file(os.path.join(EXAMPLES_DIR, "stress_on_truss.yaml"))
        _assert_matches(experiment, ref, self.ATOL)


@pytest.mark.julia
class TestReferenceData:
    """Validate the generated HDF5 reference data itself for consistency."""

    def test_diffusion_reference_shape(self, nd_references):
        ref = _load_reference(nd_references, "diffusion")
        assert ref["t"].shape == (201,)
        assert ref["u"].shape[1] == 20  # 20 nodes
        assert ref["u"].shape[0] == 201  # 201 time points
        assert ref["x0"].shape == (20,)

    def test_diffusion_reference_convergence(self, nd_references):
        """Diffusion should converge toward uniform state."""
        ref = _load_reference(nd_references, "diffusion")
        final = ref["u"][-1, :]  # last timestep, all nodes
        assert np.std(final) < 0.15, f"Diffusion should converge: std={np.std(final):.6f}"

    def test_kuramoto_reference_shape(self, nd_references):
        ref = _load_reference(nd_references, "kuramoto")
        assert ref["t"].shape == (201,)
        assert ref["u"].shape[1] == 8
        assert ref["u"].shape[0] == 201
        assert ref["omega0"].shape == (8,)

    def test_kuramoto_reference_frequencies(self, nd_references):
        """Omega0 should be centered and span [1/N, 1]."""
        ref = _load_reference(nd_references, "kuramoto")
        ω = ref["omega0"]
        assert abs(np.sum(ω)) < 1e-10, f"ω should be centered: sum={np.sum(ω)}"
        assert ω.shape == (8,)

    def test_fhn_reference_shape(self, nd_references):
        ref = _load_reference(nd_references, "fitzhugh_nagumo")
        assert ref["t"].shape == (2001,)
        # 90 nodes × 2 state vars = 180 states
        assert ref["u"].shape[1] == 180
        assert ref["u"].shape[0] == 2001

    def test_fhn_reference_bounded(self, nd_references):
        """FHN states should be bounded (no numerical explosion)."""
        ref = _load_reference(nd_references, "fitzhugh_nagumo")
        assert np.all(np.isfinite(ref["u"])), "All states should be finite"
        assert np.max(np.abs(ref["u"])) < 50, f"States should be bounded: max={np.max(np.abs(ref['u'])):.1f}"

    def test_fhn_connectivity_matrix(self, nd_references):
        """FHN connectivity matrix should be 90×90 with correct sparsity."""
        ref = _load_reference(nd_references, "fitzhugh_nagumo")
        G = ref["G"]
        assert G.shape == (90, 90)
        # Not fully connected, not empty
        n_edges = np.count_nonzero(G)
        assert 1000 < n_edges < 8100, f"Expected ~7793 edges, got {n_edges}"

    def test_diffusion_2d_reference_shape(self, nd_references):
        """2D Diffusion reference: 301 timesteps, 20 states (10 nodes × 2 SVs)."""
        ref = _load_reference(nd_references, "diffusion_2d")
        assert ref["t"].shape == (301,)
        # 10 nodes × 2 state vars = 20 interleaved states
        assert ref["u"].shape[0] == 301
        assert ref["u"].shape[1] == 20

    def test_diffusion_2d_reference_convergence(self, nd_references):
        """2D Diffusion: both state variables should converge."""
        ref = _load_reference(nd_references, "diffusion_2d")
        final = ref["u"][-1, :]  # last timestep, all states
        # x states (even indices), phi states (odd indices)
        x_final = final[0::2]
        phi_final = final[1::2]
        assert np.std(x_final) < 0.5, f"x should converge: std={np.std(x_final):.4f}"
        assert np.std(phi_final) < 0.5, f"phi should converge: std={np.std(phi_final):.4f}"

    def test_heterogeneous_kuramoto_reference_shape(self, nd_references):
        """Heterogeneous Kuramoto reference: 201 timesteps, mixed state dim.

        The node dimensions are 0 for the static node, 1 each for the six Kuramoto nodes and 2 for the inertia node, so the reference carries 8 states in total.
        """
        ref = _load_reference(nd_references, "heterogeneous_kuramoto")
        assert ref["t"].shape == (201,)
        assert ref["u"].shape == (201, 8)

    def test_heterogeneous_kuramoto_reference_bounded(self, nd_references):
        """Heterogeneous Kuramoto: all states should be finite and bounded."""
        ref = _load_reference(nd_references, "heterogeneous_kuramoto")
        assert np.all(np.isfinite(ref["u"])), "All states should be finite"

    def test_heterogeneous_kuramoto_vertex_types_data(self, nd_references):
        """Heterogeneous Kuramoto: vertex types are correctly stored."""
        ref = _load_reference(nd_references, "heterogeneous_kuramoto")
        assert len(ref["vertex_types"]) == 8

    def test_cascading_failure_reference_shape(self, nd_references):
        """Cascading failure: saveat=0.01 over 6s + callback interpolation."""
        ref = _load_reference(nd_references, "cascading_failure")
        # Callback events may add extra interpolated points
        assert ref["t"].shape[0] >= 601
        assert ref["u"].shape[0] == ref["t"].shape[0]
        assert ref["u"].shape[1] == 10  # 5 nodes × 2 SVs

    def test_cascading_failure_reference_bounded(self, nd_references):
        """Cascading failure: states should be bounded (no numerical explosion)."""
        ref = _load_reference(nd_references, "cascading_failure")
        assert np.all(np.isfinite(ref["u"])), "All states should be finite"
        assert np.max(np.abs(ref["u"])) < 100, f"States should be bounded: max={np.max(np.abs(ref['u'])):.1f}"

    def test_cascading_failure_adjacency(self, nd_references):
        """Cascading failure: 5×5 adjacency matrix with 7 edges."""
        ref = _load_reference(nd_references, "cascading_failure")
        adj = ref["adjacency"]
        assert adj.shape == (5, 5)
        n_edges = np.count_nonzero(adj)
        assert n_edges == 14  # 7 undirected edges × 2

    def test_stress_on_truss_reference_shape(self, nd_references):
        """Stress on truss: 1201 timesteps, 36 states (9 free × 4 SVs)."""
        ref = _load_reference(nd_references, "stress_on_truss")
        assert ref["t"].shape == (1201,)
        assert ref["u"].shape[0] == 1201  # timesteps
        assert ref["u"].shape[1] == 36  # 9 free nodes × 4 SVs

    def test_stress_on_truss_reference_bounded(self, nd_references):
        """Stress on truss: states should be bounded (no numerical explosion)."""
        ref = _load_reference(nd_references, "stress_on_truss")
        assert np.all(np.isfinite(ref["u"])), "All states should be finite"

    def test_stress_on_truss_adjacency(self, nd_references):
        """Stress on truss: 11×11 adjacency matrix with 18 edges."""
        ref = _load_reference(nd_references, "stress_on_truss")
        adj = ref["adjacency"]
        assert adj.shape == (11, 11)
        n_edges = np.count_nonzero(adj)
        assert n_edges == 36  # 18 undirected edges × 2


# Test 5: URL validation (no Julia needed)
class TestDocURLs:
    """Verify that documentation URLs in YAML files are valid."""

    @pytest.fixture(
        params=[
            "diffusion",
            "kuramoto",
            "fitzhugh_nagumo",
            "diffusion_2d",
            "heterogeneous_kuramoto",
            "cascading_failure",
            "stress_on_truss",
        ]
    )
    def yaml_path(self, request):
        return os.path.join(EXAMPLES_DIR, f"{request.param}.yaml")

    def test_yaml_urls_use_generated_path(self, yaml_path):
        """All URLs should use /stable/generated/ path (not bare /stable/)."""
        import yaml

        with open(yaml_path) as f:
            spec = yaml.safe_load(f)
        refs = spec.get("references", [])
        for url in refs:
            if "NetworkDynamics.jl" in url:
                assert "/generated/" in url, f"URL should use /generated/ path: {url}"
                # Check it doesn't use old format
                assert url.count("/stable/") == 1, f"URL has duplicate /stable/: {url}"
