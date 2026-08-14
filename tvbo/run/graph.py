"""Graph-based simulation of a connectome as a network of coupled local models.

This module provides [`GraphRunner`](/api/run/graph.qmd#GraphRunner), which turns a connectome into a `networkx` graph, attaches a local dynamics model to each
node and a coupling to each edge, and integrates the resulting network in time using the helpers in [`tvbo.run.compgraph`](/api/run/compgraph.qmd).
"""

import copy

import networkx as nx
import numpy as np

from tvbo.datamodel import schema as tvbo_datamodel
from tvbo.classes.coupling import Coupling
from tvbo.utils import initial_value
from tvbo.classes import dynamics as localdynamics
from tvbo.run import compgraph


class GraphRunner:
    """Assemble and integrate a connectome as a network of coupled local models.

    A `GraphRunner` holds a `networkx` graph snapshot built from a connectome.
    Node attributes carry the local dynamics model, its integrated state and optional stimulus; edge attributes carry the coupling. After the local
    models, couplings and stimuli have been attached, [`run`](/api/run/graph.qmd#GraphRunner.run) compiles per-node and per-edge functions and integrates the network in time.

    The integrator reads one edge per node pair: a multigraph snapshot is
    flattened to a simple digraph at construction, and true parallel edges
    (typed projections between the same pair) are rejected with a `ValueError`.

    Args:
        connectome: Connectome whose weights and node/edge structure define the
            network. Its `create_graph` method supplies the graph snapshot.
        normalize_weights: When `True`, normalize the connectome weights via the
            connectome's schema-safe `normalize_weights` method before building
            the graph. Failures during normalization are ignored.
    """

    def __init__(self, connectome, normalize_weights=True):
        if normalize_weights:
            # Normalize using Connectome's schema-safe method
            try:
                connectome.normalize_weights()
            except Exception:
                pass
        graph = connectome.create_graph()
        if isinstance(graph, (nx.MultiDiGraph, nx.MultiGraph)):
            if any(key > 0 for _, _, key in graph.edges(keys=True)):
                raise ValueError("GraphRunner integrates one edge per node pair; collapse parallel edges before running.")
            graph = nx.DiGraph(graph) if graph.is_directed() else nx.Graph(graph)
        self.graph = graph

    def add_local_model(self, model):
        """Attach a local dynamics model to the graph nodes.

        Args:
            model: A single `Model`/`Dynamics` instance applied to every node,
                or a dict mapping node identifiers to per-node model instances.
        """
        if isinstance(model, localdynamics.Model) or isinstance(model, localdynamics.Dynamics):
            for node in self.graph.nodes:
                self.graph.nodes[node]["model"] = model

        elif isinstance(model, dict):
            for node in model:
                self.graph.nodes[node]["model"] = model[node]

    def add_coupling(self, coupling):
        """Attach a coupling function to the graph edges.

        The snapshot is a simple digraph (see `__init__`), so edges are keyed
        by `(source, target)`.

        Args:
            coupling: A [`Coupling`](../classes/coupling.qmd#Coupling) instance, a
                deep copy of which is assigned to every edge; a bare
                datamodel `Coupling`, which is wrapped in a `Coupling` and then
                copied to every edge; or a dict mapping `(source, target)` pairs
                to per-edge couplings.
        """
        if isinstance(coupling, Coupling):
            for src, tgt in self.graph.edges:
                self.graph[src][tgt]["coupling"] = copy.deepcopy(coupling)

        elif isinstance(coupling, tvbo_datamodel.Coupling):
            wrapped = Coupling(metadata=coupling)
            for src, tgt in self.graph.edges:
                self.graph[src][tgt]["coupling"] = copy.deepcopy(wrapped)

        elif isinstance(coupling, dict):
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
        """Attach a stimulus to a single node.

        Args:
            node: Identifier of the node to stimulate.
            stimulus: Stimulus to apply at that node.
            stvar: State variable name, or list of names, to mark as
                stimulation targets on the node's model. Ignored when
                `as_derived_variable` is `True`.
            as_derived_variable: When `True`, add the stimulus to the node's
                model as a derived variable instead of storing it on the node
                and flagging state variables.
        """
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
        """Compile each node's model into a callable derivative function.

        Stores the compiled `python-network` derivative function under the
        `"dfun"` attribute of every node.
        """
        for node in self.graph.nodes:
            self.graph.nodes[node]["dfun"] = self.graph.nodes[node]["model"].execute("python-network")

    def setup_cfuns(self):
        """Compile each edge's coupling into callable coupling functions.

        For every edge, stores the compiled `python` coupling function under
        `"cfun"`, `"prefun"` and `"postfun"` lambdas obtained by substituting
        the coupling's parameter values into its pre- and post-summation
        expressions, and `"post_src"`, the substituted post expression the
        integrator compares to reject mixed post-transforms on one node.
        """
        from sympy import Symbol, lambdify

        for src, tgt in self.graph.edges:
            coup = self.graph[src][tgt]["coupling"]
            subs = {k: p.value for k, p in coup.parameters.items()}
            post_expr = coup.post.subs(subs)
            self.graph[src][tgt]["cfun"] = coup.execute("python")
            self.graph[src][tgt]["prefun"] = lambdify([Symbol("x_j")], coup.pre.subs(subs))
            self.graph[src][tgt]["postfun"] = lambdify([Symbol("gx")], post_expr)
            self.graph[src][tgt]["post_src"] = str(post_expr)

    def setup_initial_conditions(self):
        """Initialize each node's state from its model's initial values.

        Stores, under each node's `"state"` attribute, an array of the initial values of the model's state variables.
        """
        for node in self.graph.nodes:
            self.graph.nodes[node]["state"] = np.array(
                [initial_value(sv) for sv in self.graph.nodes[node]["model"].state_variables.values()]
            )
            # self.graph.nodes[node]["state"] = np.random.uniform(-1, 1, size=2)

    def setup_stimulation(self, sampling_rate=500, duration=2000):
        """Compile stimulus functions for every stimulated node.

        For each node that carries a non-`None` `"stimulus"`, compiles the stimulus to a `python` callable sampled at `sampling_rate` over the
        stimulus's own duration and stores it under the node's `"stimfun"` attribute.

        Args:
            sampling_rate: Sampling rate, in Hz, at which each stimulus is
                evaluated.
            duration: Unused; each stimulus is sampled over its own duration.
        """
        for node in self.graph.nodes:
            if "stimulus" in self.graph.nodes[node].keys() and self.graph.nodes[node]["stimulus"] is not None:
                stimulus = self.graph.nodes[node]["stimulus"]
                self.graph.nodes[node]["stimfun"] = stimulus.execute(
                    format="python",
                    duration=stimulus.duration,
                    sampling_rate=sampling_rate,
                )

    def run(self, duration=1000, dt=1, format="graph"):
        """Integrate the network in time and return the simulated time series.

        Sets up initial conditions, stimulation, node derivative functions and edge coupling functions, initializes the delay history buffer, then
        integrates the network dynamics with delays and collects the resulting time series.

        Args:
            duration: Total simulation time, in the model's time units.
            dt: Integration time step.
            format: Reserved output-format selector; currently unused.

        Returns:
            The collected per-node time series over the simulated interval.
        """
        self.setup_initial_conditions()
        self.setup_stimulation()
        self.setup_dfuns()
        self.setup_cfuns()

        compgraph.initialize_graph_states_with_history(self.graph, delay_buffer=1000)
        time_points = compgraph.simulate_graph_dynamics_with_delay(self.graph, T=duration, dt=dt)

        ts = compgraph.collect_time_series(self.graph, time_points)

        return ts
