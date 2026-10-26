"""Observation models that transform simulation output into observables.

This module provides [`Function`](#tvbo.classes.observation.Function), a named symbolic transformation, and [`ObservationModel`](#tvbo.classes.observation.ObservationModel), a directed graph that chains such functions (e.g. BOLD HRF, filtering, functional connectivity) into an observation pipeline. Helper routines convert Python callables and curated ontology instances into the underlying datamodel shape.
"""

import importlib
import inspect
import logging
from types import FunctionType
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from tvbo.data.types import TimeSeries
from tvbo.datamodel import schema as tvbo_datamodel
from tvbo.plot.ontology import draw_custom_nodes

logger = logging.getLogger(__name__)


def expand_to_4d(array):
    """Expand dimensions of the input array to ensure it has 4 dimensions."""
    while array.ndim < 4:
        array = np.expand_dims(array, axis=-1)  # Add dimensions at the end
    return array


def functioninstance2metadata(function_instance, **kwargs):
    """Normalize a function/ontology instance into datamodel kwargs.

    - For Python callables: infer arguments/parameters, capture source code,
      record callable path (module + qualname), and infer software requirements.
    - For ontology instances: map fields from the ontology to datamodel shape.
    """
    # Python callable path
    if isinstance(function_instance, FunctionType) or callable(function_instance):
        signature = inspect.signature(function_instance)
        arguments = {name: {"name": name} for name, param in signature.parameters.items() if param.default == inspect._empty}
        parameters = {
            name: {"name": name, "value": param.default}
            for name, param in signature.parameters.items()
            if param.default != inspect._empty and isinstance(param.default, (int, float))
        }

        # Base kwargs shared for callables
        merged = {
            **kwargs,
            "name": getattr(function_instance, "__name__", None) or kwargs.get("name"),
            "arguments": arguments,
            "equation": {"parameters": parameters},
        }

        # Capture source code when available
        if hasattr(function_instance, "__code__"):
            try:
                source_code = inspect.getsource(function_instance)
            except OSError:
                source_code = None
        else:
            source_code = None
        merged["source_code"] = source_code.strip() if isinstance(source_code, str) else None

        # Callable path metadata (module + qualname)
        qualname = getattr(function_instance, "__qualname__", getattr(function_instance, "__name__", None))
        merged["callable"] = {
            "module": getattr(function_instance, "__module__", None),
            "qualname": qualname,
            "name": getattr(function_instance, "__name__", None),
        }

        # Software requirements based on the callable's module
        module_name = getattr(function_instance, "__module__", None)
        if module_name:
            module = importlib.import_module(module_name)
            base_module = module.__name__.split(".")[0]
            # Skip local/interactive/builtins
            if base_module not in ("__main__", "builtins"):
                version = getattr(importlib.import_module(base_module), "__version__", None)
                prefix = base_module + "."
                submodule = module.__name__[len(prefix) :] if module.__name__.startswith(prefix) else ""

                requirements = dict(kwargs.get("requirements", {}))
                requirements.update(
                    {
                        base_module: tvbo_datamodel.SoftwareRequirement(
                            name=base_module,
                            version=version,
                            modules=([submodule] if submodule else []),
                        )
                    }
                )
                merged["requirements"] = requirements

        return merged

    # Ontology instance path
    return {
        **kwargs,
        "name": function_instance.name,
        "acronym": function_instance.acronym.first(),
        "arguments": {
            arg.prefLabel.first(): {
                "name": arg.prefLabel.first(),
                "unit": arg.unit.first(),
            }
            for arg in function_instance.has_argument
        },
        "equation": {
            "rhs": function_instance.equation.first(),
            "parameters": {
                p.prefLabel.first(): {
                    "name": p.prefLabel.first(),
                    "value": p.defaultValue.first(),
                }
                for p in function_instance.has_parameter
            },
        },
    }


def instance2metadata(instance, **kwargs):
    """Normalize an ontology transformation instance into datamodel kwargs.

    Maps the instance's name, arguments, equation, parameters and acronym onto the keyword arguments used to construct a datamodel object, nesting the argument and equation metadata under a `transformation` key.

    Args:
        instance: Ontology instance exposing `name`, `has_argument`,
            `equation`, `has_parameter` and `acronym` accessors.
        **kwargs: Extra keyword arguments merged into the result; keys produced
            here take precedence over same-named incoming keys.

    Returns:
        The merged keyword-argument mapping describing the transformation.
    """
    kwargs = {
        **kwargs,  # TODO: remember dict unpacking prioritizes keys from unpacked dict over keys defined later in the same dict!
        "transformation": {
            "name": instance.name,
            "arguments": {arg.name: {"name": arg.name, "unit": arg.unit.first()} for arg in instance.has_argument},
            "equation": {"rhs": instance.equation.first()},
        },
        "parameters": {
            p.prefLabel.first(): {
                "name": p.prefLabel.first(),
                "value": p.defaultValue.first(),
            }
            for p in instance.has_parameter
        },
        "name": instance.name,
        "acronym": instance.acronym.first(),
    }
    return kwargs


Function = tvbo_datamodel.Function
"""The generated class. Imported from here for the callers that always have; its behaviour lives in
:mod:`tvbo.behaviour.function`, which is now the only place a function's behaviour is defined."""


class ObservationModel:
    """A directed graph of `Function`s transforming simulation output to observables.

    `ObservationModel` chains symbolic and numerical operations (e.g. BOLD HRF → low-pass filter → downsample → FC matrix) on a per-region time series. Nodes are `Function`s; edges describe data flow from `Input` to `Output`. Use `add_node(name, function, ...)`, `add_edge(src, dst)` and `run()` to evaluate the pipeline.
    """

    # TODO: Checkout dask for parallel execution

    def __init__(self, data=None):
        self.data = data
        self.graph = nx.DiGraph()
        self.results = {}
        self.graph.add_node("Input", function=None, params={})
        self.graph.add_node("Output", function=None, params={})
        self.graph.nodes["Input"]["data"] = data

        self.last_function_name = None

    def add_data(self, node, data):
        """Attach a data array to a graph node.

        Accepts a `TimeSeries` (whose values and time axis are extracted) or a raw array (for which an integer time axis is generated). Creates the node when it does not yet exist, otherwise updates its stored data.

        Args:
            node: Name of the graph node to attach the data to.
            data: A `TimeSeries` or array-like providing the node's values.
        """
        if isinstance(data, TimeSeries):
            time = data.time
            data = data.data
        else:
            time = np.arange(data.shape[0])
        if node not in self.graph.nodes:
            self.graph.add_node(node, data=data, time=time, function=None, params={})
        else:
            self.graph.nodes[node]["data"] = data
            self.graph.nodes[node]["time"] = time

    def add_function(
        self,
        function,
        argument_mapping=None,
        function_type="",
        select_state=None,
        select_region=None,
        select_mode=0,
        ensure_4d=False,
        apply_on_time=False,
        alt_name=None,
        **kwargs,
    ):
        """Add a `Function` node to the pipeline graph.

        Registers the function as a graph node, records its execution options, overrides parameter values from `kwargs`, and wires edges from the nodes named in `argument_mapping` to this node. Unless the function is a derivative, it becomes the new tail feeding the `Output` node.

        Args:
            function: The `Function` to add as a node.
            argument_mapping: Mapping from each function argument name to the
                graph node supplying that argument.
            function_type: Role of the function (e.g. `"derivative"`,
                `"projection"`); non-derivatives are chained into `Output`.
            select_state: Optional state-variable index sliced from inputs.
            select_region: Optional region selection applied to inputs.
            select_mode: Mode index selected from inputs.
            ensure_4d: When `True`, expand inputs to four dimensions.
            apply_on_time: When `True`, apply the function to the time axis.
            alt_name: Alternative name/acronym used for the node.
            **kwargs: Parameter values; entries matching equation parameters
                override the function's stored values.
        """
        if argument_mapping is None:
            argument_mapping = {}
        if alt_name:
            function.acronym = alt_name

        func_name = alt_name or function.name
        self.graph.add_node(
            func_name,
            function=function,
            params=kwargs,
            argument_mapping=argument_mapping,
            function_type=function_type,
            ensure_4d=ensure_4d,
            apply_on_time=apply_on_time,
            select_state=select_state,
            select_region=select_region,
            select_mode=select_mode,
        )
        for k, v in kwargs.items():
            if k in function.equation.parameters.keys():
                function.equation.parameters[k].value = v

        # Detect dependencies automatically from argument names
        for arg in function.arguments.keys():
            if argument_mapping[arg] not in self.graph.nodes:
                self.graph.add_node(argument_mapping[arg], variable_name=arg)
            self.graph.add_edge(argument_mapping[arg], func_name, argument=arg)

        if function_type != "derivative" and self.last_function_name:
            self.graph.remove_edge(self.last_function_name, "Output")
        if function_type != "derivative":
            self.last_function_name = func_name
            self.graph.add_edge(func_name, "Output")

    def add_derivative(self, function, argument_mapping=None, **kwargs):
        """Add a derivative `Function` node to the pipeline.

        Convenience wrapper around [`add_function`](#tvbo.classes.observation.ObservationModel.add_function) with `function_type="derivative"`, so the node is computed as a side branch rather than chained into `Output`.

        Args:
            function: The `Function` to add as a derivative node.
            argument_mapping: Mapping from argument names to source nodes.
            **kwargs: Additional options forwarded to `add_function`.
        """
        if argument_mapping is None:
            argument_mapping = {}
        self.add_function(
            function,
            argument_mapping=argument_mapping,
            function_type="derivative",
            **kwargs,
        )

    def add_projection_model(self, function, argument_mapping=None, **kwargs):
        """Add a projection `Function` node to the pipeline.

        Convenience wrapper around [`add_function`](#tvbo.classes.observation.ObservationModel.add_function) with `function_type="projection"`.

        Args:
            function: The `Function` to add as a projection node.
            argument_mapping: Mapping from argument names to source nodes.
            **kwargs: Additional options forwarded to `add_function`.
        """
        if argument_mapping is None:
            argument_mapping = {}
        self.add_function(
            function,
            argument_mapping=argument_mapping,
            function_type="projection",
            **kwargs,
        )
        # TODO: Finish implementation
        pass

    def plot_graph(self, ax=None, plot_edge_labels=True, node_kwargs=None, edge_kwargs=None):
        """Draw the pipeline graph, including `Input` and `Output` nodes.

        Lays out the directed graph (falling back to a spring layout when Graphviz is unavailable) and annotates edges with their argument names and any selected state index.

        Args:
            ax: Matplotlib axes to draw into; a new figure is created and
                returned when omitted.
            plot_edge_labels: When `True`, draw argument/state labels on edges.
            node_kwargs: Keyword arguments forwarded to the node renderer.
            edge_kwargs: Keyword arguments forwarded to the edge renderer;
                `font_size` controls the edge-label size.

        Returns:
            The created figure when `ax` is not provided, otherwise `None`.
        """
        if edge_kwargs is None:
            edge_kwargs = {}
        if node_kwargs is None:
            node_kwargs = {}
        try:
            pos = nx.nx_pydot.graphviz_layout(self.graph, prog="dot")  # Layout for graph visualization
        except Exception:
            pos = nx.spring_layout(self.graph)  # Layout for graph visualization

        edge_labels = {}
        for src, dst, _data in self.graph.edges(data=True):
            select_state = self.graph.nodes[dst].get("select_state", None)
            label = self.graph[src][dst].get("argument", "")

            if select_state is not None:
                label += f"\n$State[{select_state}]$"

            edge_labels[(src, dst)] = label

        if ax is None:
            fig, ax = plt.subplots()
            return_fig = True
        else:
            return_fig = False

        edge_font_size = edge_kwargs.pop("font_size", "smaller")
        nx.draw_networkx_edges(self.graph, pos, ax=ax, label=False, **edge_kwargs)

        if plot_edge_labels:
            nx.draw_networkx_edge_labels(
                self.graph,
                pos,
                edge_labels=edge_labels,
                font_size=edge_font_size,
                ax=ax,
            )
        draw_custom_nodes(self.graph, pos, ax=ax, facecolor="white", edgecolor="grey", **node_kwargs)
        ax.axis("off")
        if return_fig:
            plt.title("Observation Model Graph (with inputs and outputs)")
            plt.close()
            return fig

    def _run_node_function(self, node_label, ensure_4d=False, time_mapping=False):
        function = self.graph.nodes[node_label]["function"]
        executable_function = function.execute(fill_in_parameters=False)
        params = function.get_parameters()

        input_values = {}
        for pred in self.graph.predecessors(node_label):
            argument = self.graph.get_edge_data(pred, node_label).get("argument", None)
            if argument == time_mapping:
                pred_data = self.current_time
            else:
                pred_data = self.graph.nodes[pred]["data"]

            if self.graph.nodes[node_label]["select_state"] is not None:
                pred_data = pred_data[
                    :,
                    self.graph.nodes[node_label]["select_state"],
                    :,
                    self.graph.nodes[node_label]["select_mode"],
                ]
            input_values[argument] = expand_to_4d(pred_data) if ensure_4d else pred_data
            params.update(self.graph.nodes[node_label]["params"])

        result = executable_function(**{**input_values, **params})
        return result

    def apply(self, timeseries, mode=0):
        """Run the pipeline on a time series and return the observable.

        Feeds the input into the `Input` node, evaluates every node in topological order, propagates each function's output to its successors, and trims the final `Output` back to the input's shape.

        Args:
            timeseries: A `TimeSeries` or array-like of simulation output; a raw
                array is wrapped in a `TimeSeries` with an integer time axis.
            mode: Mode index (currently unused in slicing).

        Returns:
            A `TimeSeries` holding the pipeline's output data and time axis.
        """
        if isinstance(timeseries, TimeSeries):
            self.data = timeseries.data  # [:, :, :, mode]
            self.time = timeseries.time
        else:
            self.data = timeseries
            self.time = np.arange(self.data.shape[0])
            timeseries = TimeSeries(data=self.data, time=self.time)
        self.orig_timeseries = timeseries

        self.graph.add_node("TimeSeries", data=timeseries, variable_name="data")
        self.graph.add_edge("TimeSeries", "Input")
        # self.data = data.squeeze()

        self.graph.nodes["Input"]["data"] = self.data
        self.graph.nodes["Input"]["time"] = self.time
        self.graph.nodes["Output"]["data"] = {}
        self.current_data = self.data
        self.current_time = self.time

        execution_order = list(nx.topological_sort(self.graph))

        for node_label in execution_order:
            node = self.graph.nodes[node_label]
            ensure_4d = node.get("ensure_4d", False)
            apply_on_time = node.get("apply_on_time", False)
            function_type = self.graph.nodes[node_label].get("function_type", None)

            if "Input" in self.graph.predecessors(node_label):
                pass

            if "function" not in node.keys() or not node["function"]:
                if "Input" in self.graph.predecessors(node_label):
                    node.update({"data": self.current_data})
                elif node["data"] is None:
                    logger.warning("Node %s has no data", node_label)
                continue

            time = (
                self._run_node_function(node_label, ensure_4d=ensure_4d, time_mapping=apply_on_time)
                if apply_on_time
                else self.current_time
            )

            output = self._run_node_function(node_label, ensure_4d=ensure_4d)

            if not function_type == "derivative":
                output = output[: time.shape[0]]

            time = time[: self.current_data.shape[0]]

            self.graph.nodes[node_label]["data"] = output
            self.graph.nodes[node_label]["time"] = time

            if function_type != "derivative":
                self.current_data = output
                self.current_time = time

            self.graph.nodes["Output"]["data"] = self.current_data
            self.graph.nodes["Output"]["time"] = self.current_time

        input_shape = self.graph.nodes["Input"]["data"].shape
        output_data = self.graph.nodes["Output"]["data"][tuple(slice(0, dim) for dim in input_shape)]
        ts = self.orig_timeseries.copy()
        ts.data = output_data
        ts.time = self.current_time
        return ts

    def get_node_data(self, node):
        """Return a node's stored data as a `TimeSeries`.

        Args:
            node: Name of the graph node to read.

        Returns:
            A `TimeSeries` pairing the node's data with its time axis (a
            generated integer axis is used when none was stored).
        """
        data = self.graph.nodes[node].get("data", None)
        time = self.graph.nodes[node].get(
            "time",
            (np.arange(data.shape[0]) if self.graph.nodes[node].get("function_type", None) != "derivative" else np.array([0])),
        )
        return TimeSeries(time, data)

    def get_function_output(self, function_name) -> Any:
        """Get the output of a specific function after execution.

        Args:
            function_name (str): The name of the function whose output to retrieve.

        Returns:
            The result produced by the function.
        """
        return self.results.get(function_name, None)

    def plot_node_data(self, node, ax):
        """Plot a single node's data onto the given axes.

        Args:
            node: Name of the graph node to plot.
            ax: Matplotlib axes to draw into.
        """
        data = self.get_node_data(node)
        ax.plot(data, label=node)

    def plot_graph_data(self, ax=None):
        """Plot the data stored at every pipeline node.

        Iterates the nodes in topological order (skipping the raw input nodes) and overlays each node's time series, highlighting the `Output` trace.

        Args:
            ax: Matplotlib axes to draw into; a new figure is created and
                returned when omitted.

        Returns:
            The created figure when `ax` is not provided, otherwise `None`.
        """
        if ax is None:
            fig, ax = plt.subplots()
            return_fig = True
        else:
            return_fig = False

        for node in nx.topological_sort(self.graph):
            if node in ["TimeSeries", "Timepoints"]:
                continue
            ts = self.get_node_data(node)
            if ts.data is not None:
                ts.plot(
                    ax=ax,
                    label=node,
                    linestyle="dotted" if node == "Output" else "-",
                    zorder=100 if node == "Output" else 0,
                )

        ax.legend(loc="upper right", fontsize=7)

        if return_fig:
            plt.title("Observation Model Graph Data")
            plt.close()
            return fig


def populate_observation_from_iri(obs, functions_sink=None) -> bool:
    """Fill an observation from the curated model its ``iri`` names, and collect its functions.

    The filling itself is :meth:`IriEnrichable.enrich`, which every class the schema gives an ``iri`` carries: the curated record supervenes nowhere, so ``source``/``period`` overrides stay in force while the curated hemodynamic pipeline fills in.

    What is specific to an observation is where its ``functions`` go. A curated model ships the helper functions its pipeline calls by name — an HRF kernel, a downsample, a convolution — and codegen reads those from ``experiment.functions``, not from the observation. Given a ``functions_sink`` (a mutable name -> Function mapping) they are merged there instead, a function the experiment already declares winning.

    Returns True if a curated model was found, False otherwise.
    """
    if not getattr(obs, "iri", None):
        return False
    try:
        obs.enrich(source="database")
    except LookupError:
        if not any(getattr(obs, slot, None) for slot in ("pipeline", "class_reference", "dynamics")):
            raise LookupError(
                f"Observation {getattr(obs, 'name', None)!r} names {obs.iri!r}, which no curated record answers, "
                "and declares no pipeline, class reference or dynamics of its own; every backend would render it as something else."
            ) from None
        return False

    if functions_sink is not None:
        from tvbo.utils import keyed_items

        for name, function in keyed_items(getattr(obs, "functions", None), "functions"):
            if name not in functions_sink:
                functions_sink[name] = function
    return True


Observation = tvbo_datamodel.Observation
"""The generated class. Its factory constructors live in :mod:`tvbo.behaviour.observation`."""
