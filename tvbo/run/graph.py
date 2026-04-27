import copy

import networkx as nx
import numpy as np

from tvbo.datamodel import schema as tvbo_datamodel
from tvbo.classes.coupling import Coupling
from tvbo.classes import dynamics as localdynamics
from tvbo.run import compgraph


class GraphRunner:
    def __init__(self, connectome, normalize_weights=True):
        if normalize_weights:
            # Normalize using Connectome's schema-safe method
            try:
                connectome.normalize_weights()
            except Exception:
                pass
        # Build a graph snapshot from current connectome
        self.graph = connectome.create_graph()

    def add_local_model(self, model):
        if isinstance(model, localdynamics.Model) or isinstance(model, localdynamics.Dynamics):
            for node in self.graph.nodes:
                self.graph.nodes[node]["model"] = model

        elif isinstance(model, dict):
            for node in model:
                self.graph.nodes[node]["model"] = model[node]

    def add_coupling(self, coupling):
        is_multi = isinstance(self.graph, (nx.MultiDiGraph, nx.MultiGraph))

        if isinstance(coupling, Coupling):
            # Copy same coupling instance to all edges
            if is_multi:
                for src, tgt, key in self.graph.edges(keys=True):
                    self.graph[src][tgt][key]["coupling"] = copy.deepcopy(coupling)
            else:
                for src, tgt in self.graph.edges:
                    self.graph[src][tgt]["coupling"] = copy.deepcopy(coupling)

        elif isinstance(coupling, tvbo_datamodel.Coupling):
            # Wrap a pure datamodel instance
            wrapped = Coupling(metadata=coupling)
            if is_multi:
                for src, tgt, key in self.graph.edges(keys=True):
                    self.graph[src][tgt][key]["coupling"] = copy.deepcopy(wrapped)
            else:
                for src, tgt in self.graph.edges:
                    self.graph[src][tgt]["coupling"] = copy.deepcopy(wrapped)

        elif isinstance(coupling, dict):
            if is_multi:
                for src, tgt, key in self.graph.edges(keys=True):
                    # Support mapping by (src,tgt,key) with fallback to (src,tgt)
                    val = coupling.get((src, tgt, key), coupling.get((src, tgt)))
                    self.graph[src][tgt][key]["coupling"] = val
            else:
                for src, tgt in self.graph.edges:
                    self.graph[src][tgt]["coupling"] = coupling[src, tgt]

    def to_yaml(self, format: str = "tvbo", filepath: str | None = None) -> str:
        """Export Network to YAML format.

        Parameters
        ----------
        format : str
            Output format: "tvbo" (default) or "pyrates" for PyRates CircuitTemplate.
        filepath : str, optional
            Path to write the YAML file. If None, returns the YAML string.

        Returns
        -------
        str
            YAML string (or filepath if written to file).
        """
        if format.lower() == "pyrates":
            from tvbo.codegen.pyrates import network_to_pyrates_yaml_string

            return network_to_pyrates_yaml_string(self, filepath)
        else:
            from tvbo.utils import to_yaml as _to_yaml

            return _to_yaml(self, filepath)

    def add_stimulus(self, node, stimulus, stvar=None, as_derived_variable=False):
        if as_derived_variable:
            self.graph.nodes[node]["model"].add_stimulus(stimulus, as_derived_variable=True)
        else:
            self.graph.nodes[node]["stimulus"] = stimulus
            if stvar is not None:
                if not isinstance(stvar, list):
                    stvar = [stvar]
                for var in stvar:
                    self.graph.nodes[node]["model"].state_variables[var].stimulation_variable = True

    def setup_dfuns(self):
        for node in self.graph.nodes:
            self.graph.nodes[node]["dfun"] = self.graph.nodes[node]["model"].execute("python-network")

    def setup_cfuns(self):
        from sympy import Symbol, lambdify

        is_multi = isinstance(self.graph, (nx.MultiDiGraph, nx.MultiGraph))

        if is_multi:
            for src, tgt, key in self.graph.edges(keys=True):
                coup = self.graph[src][tgt][key]["coupling"]
                self.graph[src][tgt][key]["cfun"] = coup.execute("python")
                self.graph[src][tgt][key]["prefun"] = lambdify(
                    [Symbol("x_j")],
                    coup.pre.subs({k: p.value for k, p in coup.parameters.items()}),
                )
                self.graph[src][tgt][key]["postfun"] = lambdify(
                    [Symbol("gx")],
                    coup.post.subs({k: p.value for k, p in coup.parameters.items()}),
                )
        else:
            for src, tgt in self.graph.edges:
                coup = self.graph[src][tgt]["coupling"]
                self.graph[src][tgt]["cfun"] = coup.execute("python")
                self.graph[src][tgt]["prefun"] = lambdify(
                    [Symbol("x_j")],
                    coup.pre.subs({k: p.value for k, p in coup.parameters.items()}),
                )
                self.graph[src][tgt]["postfun"] = lambdify(
                    [Symbol("gx")],
                    coup.post.subs({k: p.value for k, p in coup.parameters.items()}),
                )

    def setup_initial_conditions(self):
        for node in self.graph.nodes:
            self.graph.nodes[node]["state"] = np.array(
                [sv.initial_value for sv in self.graph.nodes[node]["model"].state_variables.values()]
            )
            # self.graph.nodes[node]["state"] = np.random.uniform(-1, 1, size=2)

    def setup_stimulation(self, sampling_rate=500, duration=2000):
        for node in self.graph.nodes:
            if "stimulus" in self.graph.nodes[node].keys() and self.graph.nodes[node]["stimulus"] is not None:
                stimulus = self.graph.nodes[node]["stimulus"]
                self.graph.nodes[node]["stimfun"] = stimulus.execute(
                    format="python",
                    duration=stimulus.duration,
                    sampling_rate=sampling_rate,
                )

    def run(self, duration=1000, dt=1, format="graph"):
        self.setup_initial_conditions()
        self.setup_stimulation()
        self.setup_dfuns()
        self.setup_cfuns()

        compgraph.initialize_graph_states_with_history(self.graph, delay_buffer=1000)
        time_points = compgraph.simulate_graph_dynamics_with_delay(self.graph, T=duration, dt=dt)

        ts = compgraph.collect_time_series(self.graph, time_points)

        return ts
