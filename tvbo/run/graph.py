"""Graph-based simulation of a connectome as a network of coupled local models.

This module provides [`GraphRunner`](/api/run/graph.qmd#GraphRunner), which turns a connectome into a `networkx` graph, attaches a local dynamics model to each node, with the states and parameters a node declares for itself, and the coupling every node's afferent input goes through, and integrates the resulting network in time using the helpers in [`tvbo.run.compgraph`](/api/run/compgraph.qmd).
"""

import networkx as nx
import numpy as np

from tvbo.classes import dynamics as localdynamics
from tvbo.classes.coupling import Coupling
from tvbo.run import compgraph
from tvbo.utils import initial_value


class GraphRunner:
    """Assemble and integrate a connectome as a network of coupled local models.

    A `GraphRunner` holds a `networkx` graph snapshot built from a connectome. Node attributes carry the local dynamics model, its integrated state, the parameters it declares for itself, its compiled coupling and optional stimulus; edge attributes carry the weight and delay. After the local models, the coupling and stimuli have been attached, [`run`](/api/run/graph.qmd#GraphRunner.run) compiles the per-node functions and integrates the network in time. The graph lists the nodes in the connectome's declaration order, which is the order the per-node declarations and labels are read in.

    The integrator reads one edge per node pair: a multigraph snapshot is flattened to a simple digraph at construction, and true parallel edges (typed projections between the same pair) are rejected with a `ValueError`.

    Args:
        connectome: Connectome whose weights and node/edge structure define the network. Its `create_graph` method supplies the graph snapshot, and its declared nodes the per-node states, parameters and labels.
        normalize_weights: When `True`, normalize the connectome weights via the connectome's schema-safe `normalize_weights` method before building the graph. Failures during normalization are ignored.
    """

    def __init__(self, connectome, normalize_weights=True):
        if normalize_weights:
            # Normalize using Connectome's schema-safe method
            try:
                connectome.normalize_weights()
            except Exception:
                pass
        self.connectome = connectome
        self.coupling = None
        graph = connectome.create_graph()
        if isinstance(graph, (nx.MultiDiGraph, nx.MultiGraph)):
            if any(key > 0 for _, _, key in graph.edges(keys=True)):
                raise ValueError("GraphRunner integrates one edge per node pair; collapse parallel edges before running.")
            graph = nx.DiGraph(graph) if graph.is_directed() else nx.Graph(graph)
        self.graph = graph

    def add_local_model(self, model):
        """Attach a local dynamics model to the graph nodes.

        Args:
            model: A single `Model`/`Dynamics` instance applied to every node, or a dict mapping node identifiers to per-node model instances.
        """
        if isinstance(model, localdynamics.Dynamics):
            for node in self.graph.nodes:
                self.graph.nodes[node]["model"] = model

        elif isinstance(model, dict):
            for node in model:
                self.graph.nodes[node]["model"] = model[node]

    def add_coupling(self, coupling):
        """Attach the coupling every node's afferent input goes through.

        Args:
            coupling: A [`Coupling`](../classes/coupling.qmd#Coupling), which every node's input goes through, a node without afferents included; or ``None``, which leaves every node uncoupled.

        Raises:
            TypeError: *coupling* is neither.
        """
        if coupling is not None and not isinstance(coupling, Coupling):
            raise TypeError(f"GraphRunner integrates one Coupling through every node's input, not a {type(coupling).__name__}")
        self.coupling = coupling

    def to_yaml(self, format: str = "tvbo", filepath: str | None = None) -> str:
        """Export Network to YAML format.

        Parameters
        ----------
        format : str
            Output format: "tvbo" (default) or "pyrates" for PyRates CircuitTemplate.
        filepath : str, optional
            Path to write the YAML file. If None, returns the YAML string.

        Returns:
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
            stvar: State variable name, or list of names, to mark as stimulation targets on the node's model. Ignored when `as_derived_variable` is `True`.
            as_derived_variable: When `True`, add the stimulus to the node's model as a derived variable instead of storing it on the node and flagging state variables.
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

        Stores the compiled `python-network` derivative function under the `"dfun"` attribute of every node.
        """
        for node in self.graph.nodes:
            self.graph.nodes[node]["dfun"] = self.graph.nodes[node]["model"].execute("python-network")

    def setup_cfuns(self):
        """Compile the coupling for each node's model (`compgraph.coupling_transforms`) and store it under the node's `"coupling"`; without a coupling every node's is ``None``."""
        compiled = {}
        for node in self.graph.nodes:
            model = self.graph.nodes[node]["model"]
            if self.coupling is not None and id(model) not in compiled:
                compiled[id(model)] = compgraph.coupling_transforms(model, self.coupling)
            self.graph.nodes[node]["coupling"] = compiled.get(id(model))

    def setup_initial_conditions(self):
        """Initialize each node's state and dynamics parameters from what it declares for itself.

        A node's `"state"` holds its model's initial values, with each state the node declares in its place (`node_initial_states`), and its `"parameters"` the model's parameters it sets (`node_parameter_arrays`), both read at the node's position in the graph, its declaration order.
        """
        from tvbo.adapters.base import node_initial_states, node_parameter_arrays

        declared = {}
        for position, node in enumerate(self.graph.nodes):
            model = self.graph.nodes[node]["model"]
            if id(model) not in declared:
                defaults = np.array([initial_value(sv) for sv in model.state_variables.values()])
                states = np.repeat(defaults[:, None], self.graph.number_of_nodes(), axis=1)
                declared[id(model)] = (
                    node_initial_states(self.connectome, model, states),
                    node_parameter_arrays(self.connectome, model),
                )
            states, parameters = declared[id(model)]
            self.graph.nodes[node]["state"] = states[:, position]
            self.graph.nodes[node]["parameters"] = {name: float(values[position]) for name, values in parameters.items()}

    def setup_stimulation(self, sampling_rate=500, duration=2000):
        """Compile stimulus functions for every stimulated node.

        For each node that carries a non-`None` `"stimulus"`, compiles the stimulus to a `python` callable sampled at `sampling_rate` over the stimulus's own duration and stores it under the node's `"stimfun"` attribute.

        Args:
            sampling_rate: Sampling rate, in Hz, at which each stimulus is evaluated.
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

    def run(self, duration=1000, dt=1, format="graph", integration=None):
        """Integrate the network in time and return the simulated time series.

        Sets up initial conditions and per-node parameters, stimulation, node derivative functions and each node's coupling, starts each node's trace at its initial state, then integrates the network with delays by the declared method, every node synchronously (`compgraph.simulate_graph_dynamics_with_delay`), and collects the states recorded at dt to `duration`, the nodes labelled by the connectome's `node_labels`.

        Args:
            duration: Total simulation time, in the model's time units.
            dt: Integration time step.
            format: Reserved output-format selector; currently unused.
            integration: The experiment's `Integrator`, whose ``method`` steps every node and whose ``coupling_evaluation`` must be the per-step one this runner integrates; ``None`` integrates by the schema's default method.

        Returns:
            The collected per-node time series over the simulated interval.
        """
        self.setup_initial_conditions()
        self.setup_stimulation()
        self.setup_dfuns()
        self.setup_cfuns()

        compgraph.initialize_graph_states_with_history(self.graph)
        time_points = compgraph.simulate_graph_dynamics_with_delay(self.graph, T=duration, dt=dt, integration=integration)

        return compgraph.collect_time_series(self.graph, time_points, labels=getattr(self.connectome, "node_labels", None))
