"""Observation models that transform simulation output into observables.

This module provides [`Function`](#tvbo.classes.observation.Function), a named symbolic transformation, and
[`ObservationModel`](#tvbo.classes.observation.ObservationModel), a directed graph that chains such functions (e.g. BOLD HRF, filtering, functional
connectivity) into an observation pipeline. Helper routines convert Python callables and curated ontology instances into the underlying datamodel shape.
"""

import logging
import importlib
import inspect
from types import FunctionType
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sympy
from sympy import (
    Eq,
    IndexedBase,
    Lambda,
    Symbol,
    lambdify,
    parse_expr,
    latex,
    Rational,
    Float,
)
from tvbo.classes.equation import _clash1
from tvbo.data.types import TimeSeries
from tvbo.datamodel import schema as tvbo_datamodel
from tvbo.codegen.code import render_expression
from tvbo.ontology import owl as ontology
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

    Maps the instance's name, arguments, equation, parameters and acronym onto the keyword arguments used to construct a datamodel object, nesting the
    argument and equation metadata under a `transformation` key.

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


class Function(tvbo_datamodel.Function):
    """A named symbolic transformation applied to simulation outputs.

    `Function` wraps an `equation` (RHS string parseable by SymPy) plus parameters and metadata. Used as the building block of
    [`ObservationModel`](#tvbo.classes.observation.ObservationModel)s (e.g. BOLD HRF, sigmoid firing-rate, band-pass filter) and as derived
    quantities (e.g. coherence, PSD, FC).

    Construct from a callable, from the curated ontology by name, or by passing `equation=`, `parameters=`, etc. inline.
    """

    def __init__(self, instance=None, **kwargs):
        """Initialize Function with datamodel fields only.

        Args:
            instance: Legacy parameter - automatically dispatches to appropriate classmethod
            **kwargs: Datamodel fields
        """
        # Handle legacy instance parameter for backward compatibility
        if instance is not None:
            if isinstance(instance, (FunctionType,)) or callable(instance):
                kwargs = functioninstance2metadata(instance, **kwargs)
            elif isinstance(instance, ontology.onto.Function):
                kwargs = functioninstance2metadata(instance, **kwargs)
            elif isinstance(instance, tvbo_datamodel.Function):
                kwargs = instance._as_dict

        # Initialize the datamodel with normalized kwargs
        super().__init__(**kwargs)

    # Removed _process_python_function and _process_ontology_instance: unified into functioninstance2metadata

    # ---- Factory classmethods ----
    @classmethod
    def from_python(cls, function_instance: FunctionType, **kwargs):
        """Create Function from a Python callable."""
        kwargs = functioninstance2metadata(function_instance, **kwargs)
        return cls(**kwargs)

    @classmethod
    def from_ontology(cls, ontology_instance, **kwargs):
        """Create Function from an ontology instance."""
        kwargs = functioninstance2metadata(ontology_instance, **kwargs)
        return cls(**kwargs)

    @classmethod
    def from_datamodel(cls, datamodel_instance: tvbo_datamodel.Function):
        """Create Function from a datamodel instance."""
        return cls(**datamodel_instance._as_dict)

    @classmethod
    def from_file(cls, filepath: str):
        """Create Function from a file."""
        from tvbo.utils import yaml_loader

        return yaml_loader.load(filepath, target_class=cls)

    @classmethod
    def from_db(cls, name: str) -> "Function":
        """Load a Function by name from the tvbo database."""
        from tvbo.data.registry import resolve

        return cls.from_file(str(resolve("Function", name)))

    @classmethod
    def list_db(cls) -> list[str]:
        """List available observation models in the tvbo database."""
        from tvbo.data.registry import list_entries

        return list_entries("Function")

    # ---- Properties for runtime-only attributes ----
    @property
    def function(self):
        """Access to the underlying callable function if available."""
        # Preferred: resolve via recorded callable path (module + qualname)
        func = self._resolve_function_from_callable_path()
        if func is not None:
            return func

        # Secondary: reconstruct from stored source code
        if hasattr(self, "source_code") and self.source_code:
            return self._reconstruct_function_from_source()

        return None

    def _resolve_function_from_callable_path(self):
        """Resolve the function by importing its module and traversing qualname."""
        if not self.callable:
            return None

        module_name = self.callable.module
        qualname = self.callable.qualname or self.callable.name
        if not module_name or not qualname:
            return None

        module = importlib.import_module(module_name)

        # Traverse qualname (handles nested objects/classes)
        obj = module
        for part in qualname.split("."):
            if part == "<locals>":
                # Skip '<locals>' artifacts from nested function names
                continue
            obj = getattr(obj, part)

        if callable(obj):
            return obj
        return None

    def _reconstruct_function_from_source(self):
        """Reconstruct function from stored source code."""
        # Create a local namespace for execution
        local_namespace = {}
        global_namespace = globals().copy()

        # Import required modules into the namespace
        if hasattr(self, "requirements") and self.requirements:
            for module_name, req in self.requirements.items():
                # Skip local/interactive modules
                if module_name in ("__main__", "builtins"):
                    continue
                # Determine full module name safely
                sub = None
                if hasattr(req, "modules") and req.modules:
                    first = req.modules[0]
                    sub = first if first else None
                full_module_name = f"{module_name}.{sub}" if sub else module_name

                # Import the module and add to namespace
                module = importlib.import_module(full_module_name)
                global_namespace[module_name] = module

                # Also import the alias of the last path segment for convenience
                last_segment = full_module_name.split(".")[-1]
                global_namespace[last_segment] = module

        # Execute the source code to define the function
        exec(self.source_code, global_namespace, local_namespace)

        # Return the function if it was created
        if self.name in local_namespace:
            return local_namespace[self.name]

        return None

    @property
    def ontology(self):
        """Access to the ontology instance if available."""
        # Try to find the ontology instance by name
        if hasattr(ontology.onto, self.name):
            return getattr(ontology.onto, self.name)
        # Try with acronym if available
        if hasattr(self, "acronym") and self.acronym and hasattr(ontology.onto, self.acronym):
            return getattr(ontology.onto, self.acronym)
        return None

    @property
    def metadata(self):
        """Backward compatibility: return self (which is now the datamodel)."""
        return self

    def get_parameters(self, key_as_symbol=False):
        """Return the equation's parameters as a name-to-value mapping.

        Args:
            key_as_symbol: When `True`, use SymPy `Symbol` objects as keys
                instead of plain parameter-name strings.

        Returns:
            Mapping from each parameter name (or `Symbol`) to its value.
        """
        parameters = {Symbol(k) if key_as_symbol else k: v.value for k, v in self.equation.parameters.items()}
        return parameters

    def get_equation(self):
        """Build the function as a SymPy equation.

        Parses the stored right-hand-side string into an expression, treats the function's arguments as `IndexedBase` symbols, and returns an equality
        whose left-hand side is the named function applied to its arguments.

        Returns:
            A SymPy `Eq` relating the function call to its parsed expression.
        """
        parameters = self.get_parameters(key_as_symbol=True)
        clash = {str(p): p for p in parameters.keys()}
        clash.update({str(a): IndexedBase(a) for a in self.arguments})
        expression = parse_expr(self.equation.rhs, clash)
        function = sympy.Function(self.acronym or self.name)(*(Symbol(a) for a in self.arguments))
        return Eq(function, expression)

    def get_symbolic_function(self):
        """Return the function as a callable SymPy `Lambda`.

        Returns:
            A SymPy `Lambda` mapping the function's arguments to its equation.
        """
        equation = self.get_equation()
        self.get_parameters()
        return Lambda(equation.lhs.args, equation)

    def execute(self, format="python", fill_in_parameters=True, parameters={}, **kwargs):
        """Compile the function into an executable callable.

        Returns the recorded Python callable when one is available; otherwise lambdifies the symbolic equation for the requested backend. Supplied
        parameters that do not appear in the equation are discarded, and the function's stored parameter values can optionally be substituted in
        before compilation.

        Args:
            format: Target backend for `lambdify` (e.g. `"python"`/`"numpy"`,
                `"jax"`); also selects the module used for numeric evaluation.
            fill_in_parameters: When `True`, substitute the function's stored
                parameter values into the expression before compiling.
            parameters: Extra parameter values to substitute; entries whose
                symbol is absent from the equation are ignored.
            **kwargs: Backend options; for `format="jax"`, `jit=True` wraps the
                result in `jax.jit` with `stepsize` treated as static.

        Returns:
            A callable evaluating the function over its arguments.
        """
        if self.function:
            return self.function

        if format == "python":
            modules = "numpy"
        else:
            modules = format

        equation = self.get_equation()
        parameters2pop = []
        for p in parameters.keys():
            if Symbol(p) not in equation.rhs.free_symbols:
                parameters2pop.append(p)
        for p in parameters2pop:
            parameters.pop(p)
        parameters.update(self.get_parameters())
        {str(k): v for k, v in parameters.items()}
        eq = equation.rhs
        if fill_in_parameters:
            eq = eq.subs(parameters)
        eq = eq.subs("e", "E")
        arguments = equation.lhs.args + tuple([k for k in parameters.keys() if Symbol(k) in eq.free_symbols])
        function = lambdify(arguments, eq, modules=modules)

        if format == "jax" and kwargs.get("jit", False):
            import jax

            function = jax.jit(
                function,
                static_argnames=[str(arg) for arg in arguments if str(arg) == "stepsize"],
            )
        return function

    def apply(self, **kwargs):
        """Execute the function and call it with the given arguments.

        Args:
            **kwargs: Argument values passed to the compiled callable.

        Returns:
            The result of evaluating the function.
        """
        return self.execute()(**kwargs)

    def render_code(self, format="python", **kwargs):
        """Render the function's equation as backend source code.

        Args:
            format: Target backend passed to the expression renderer.
            **kwargs: Additional options forwarded to the renderer.

        Returns:
            The rendered code for the equation's right-hand side.
        """
        return render_expression(self.get_equation().rhs, format=format, **kwargs)

    def plot(self, format="python", plotting_kwargs={}, **kwargs):
        """Plot the function's output against its input.

        For a single-argument function, the input array (supplied via `kwargs` under the argument name) is plotted against the evaluated output; for
        multi-argument functions the output is plotted directly using the stored parameter values.

        Args:
            format: Backend used to compile the function for evaluation.
            plotting_kwargs: Keyword arguments forwarded to `matplotlib`.
            **kwargs: Input values keyed by argument name.
        """
        function = self.execute(format=format)
        args = self.arguments
        if len(args) == 1:
            fin = kwargs.get(next(iter(args.values())).name)
            plt.plot(fin, function(fin), **plotting_kwargs)
            plt.xlabel(next(iter(self.arguments.values())).unit)
        else:
            plt.plot(function(**{**kwargs, **self.get_parameters()}), **plotting_kwargs)
        pass

    def plot_metadata_graph(self, ax=None, node_kwargs={}, edge_kwargs={}, edge_labels=True):
        """Draw a graph of the function's metadata.

        Builds a directed graph linking the function node to its equation, software requirements and arguments, then renders it with a radial
        layout.

        Args:
            ax: Matplotlib axes to draw into; a new figure is created and
                returned when omitted.
            node_kwargs: Keyword arguments forwarded to the node renderer.
            edge_kwargs: Keyword arguments reserved for edge styling.
            edge_labels: When `True`, annotate edges with their relation
                labels; otherwise fold the relation into the node labels.

        Returns:
            The created figure when `ax` is not provided, otherwise `None`.
        """
        if ax is None:
            fig, ax = plt.subplots()
            return_fig = True
        else:
            return_fig = False
        G = nx.DiGraph()
        func_name = self.acronym or self.name
        G.add_node(func_name, label=f"{func_name}")
        if self.equation and self.equation.rhs:
            expression = parse_expr(self.equation.rhs, _clash1)
            rounded_expression = expression.xreplace({n: Float(round(float(n), 4)) for n in expression.atoms(Float)})
            expression = rounded_expression.subs(0.3333, Rational(1, 3))

            G.add_node(
                "equation",
                label=f"${latex(expression)}$",
            )
            G.add_edge(func_name, "equation", label="equation")

        for req, details in self.requirements.items():
            label = f"${req}$\n{','.join(details.modules)}\n{details['version']}"
            if not edge_labels:
                label = f"requires:\n{label}"

            G.add_node(
                req,
                label=label,
            )
            G.add_edge(func_name, req, label="requires")

        for arg in self.arguments:
            label = f"${arg}$"
            if not edge_labels:
                label = f"argument:\n{label}"
            G.add_node(arg, label=label)
            G.add_edge(func_name, arg, label="arg")
        pos = nx.nx_pydot.graphviz_layout(G, prog="twopi")  # , 'fdp', 'sfdp', 'circo'
        draw_custom_nodes(
            G,
            pos,
            ax=ax,
            facecolor="white",
            edgecolor="grey",
            labels=G.nodes(data="label"),
            **node_kwargs,
        )

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
        )
        if edge_labels:
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=nx.get_edge_attributes(G, "label"),
                ax=ax,
                font_size=node_kwargs.get("font_size", "smaller"),
            )
        ax.axis("off")
        if return_fig:
            plt.close()
            return fig


class ObservationModel:
    """A directed graph of `Function`s transforming simulation output to observables.

    `ObservationModel` chains symbolic and numerical operations (e.g. BOLD
    HRF → low-pass filter → downsample → FC matrix) on a per-region time series. Nodes are `Function`s; edges describe data flow from `Input` to
    `Output`. Use `add_node(name, function, ...)`, `add_edge(src, dst)` and
    `run()` to evaluate the pipeline.
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

        Accepts a `TimeSeries` (whose values and time axis are extracted) or a raw array (for which an integer time axis is generated). Creates the
        node when it does not yet exist, otherwise updates its stored data.

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
        argument_mapping={},
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

        Registers the function as a graph node, records its execution options, overrides parameter values from `kwargs`, and wires edges from the nodes
        named in `argument_mapping` to this node. Unless the function is a derivative, it becomes the new tail feeding the `Output` node.

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

    def add_derivative(self, function, argument_mapping={}, **kwargs):
        """Add a derivative `Function` node to the pipeline.

        Convenience wrapper around [`add_function`](#tvbo.classes.observation.ObservationModel.add_function)
        with `function_type="derivative"`, so the node is computed as a side branch rather than chained into `Output`.

        Args:
            function: The `Function` to add as a derivative node.
            argument_mapping: Mapping from argument names to source nodes.
            **kwargs: Additional options forwarded to `add_function`.
        """
        self.add_function(
            function,
            argument_mapping=argument_mapping,
            function_type="derivative",
            **kwargs,
        )

    def add_projection_model(self, function, argument_mapping={}, **kwargs):
        """Add a projection `Function` node to the pipeline.

        Convenience wrapper around [`add_function`](#tvbo.classes.observation.ObservationModel.add_function)
        with `function_type="projection"`.

        Args:
            function: The `Function` to add as a projection node.
            argument_mapping: Mapping from argument names to source nodes.
            **kwargs: Additional options forwarded to `add_function`.
        """
        self.add_function(
            function,
            argument_mapping=argument_mapping,
            function_type="projection",
            **kwargs,
        )
        # TODO: Finish implementation
        pass

    def plot_graph(self, ax=None, plot_edge_labels=True, node_kwargs={}, edge_kwargs={}):
        """Draw the pipeline graph, including `Input` and `Output` nodes.

        Lays out the directed graph (falling back to a spring layout when
        Graphviz is unavailable) and annotates edges with their argument names and any selected state index.

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
        try:
            pos = nx.nx_pydot.graphviz_layout(self.graph, prog="dot")  # Layout for graph visualization
        except Exception:
            pos = nx.spring_layout(self.graph)  # Layout for graph visualization

        edge_labels = {}
        for src, dst, data in self.graph.edges(data=True):
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

        Feeds the input into the `Input` node, evaluates every node in topological order, propagates each function's output to its successors,
        and trims the final `Output` back to the input's shape.

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
        """
        Get the output of a specific function after execution.

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
    """Fill an observation's missing fields from the curated model its ``iri`` names.

    When an observation declares ``iri: tvbo:BOLD_TVB`` (or any curated entry under
    ``tvbo/database/observation_models/``), its metadata — ``pipeline``, ``parameters``,
    ``class_reference``, ``imaging_modality``, ``label``/``description`` — is loaded from that model and merged **non-destructively**: a field the recipe set locally always
    wins, so ``source``/``period`` overrides stay in force while the curated hemodynamic pipeline fills in. Mirrors :func:`tvbo.classes.coupling._load_coupling_from_database`.

    When ``functions_sink`` (a mutable name→Function mapping) is given, the model's
    ``functions`` block is merged into it too — the helper functions a functional pipeline calls by name, which codegen reads from ``experiment.functions``.

    Returns True if a curated model was found and merged, False otherwise.
    """
    iri = getattr(obs, "iri", None)
    if not iri:
        return False
    from tvbo.data.registry import local_name, resolve

    name = local_name(iri) if ":" in str(iri) else str(iri)
    try:
        path = resolve("Observation", name)
    except Exception:
        return False
    if path is None or not path.exists():
        return False

    import yaml as _yaml

    data = _yaml.safe_load(path.read_text()) or {}

    # Scalar fields: adopt the curated value only where the recipe left it empty.
    for field in ("label", "description", "imaging_modality", "period", "downsample_period", "time_scale"):
        if data.get(field) is not None and not getattr(obs, field, None):
            setattr(obs, field, data[field])

    # source: a list — take the curated one only if the recipe declared none.
    if data.get("source") and not getattr(obs, "source", None):
        obs.source = list(data["source"])

    # pipeline: the heart of a curated observation model. Fill only if absent so a recipe that hand-declares its own pipeline is never silently overridden.
    if data.get("pipeline") and not getattr(obs, "pipeline", None):
        # A step is a FunctionCall, never a bare Function: as Function, `function:`/`callable:` are dropped.
        obs.pipeline = [
            step if isinstance(step, tvbo_datamodel.FunctionCall) else tvbo_datamodel.FunctionCall(**step)
            for step in data["pipeline"]
        ]

    # dynamics: a co-integrated observer (the alternative to a pipeline — the observation computed online as a recurrence). Fill if absent, exactly as the pipeline is; without this an `iri`-referenced observer arrives with no dynamics and codegen emits a pass-through monitor.
    if data.get("dynamics") is not None and not getattr(obs, "dynamics", None):
        dyn = data["dynamics"]
        obs.dynamics = dyn if isinstance(dyn, tvbo_datamodel.Dynamics) else tvbo_datamodel.Dynamics(**dyn)

    # class_reference: a monitor/class handle (e.g. tvb Bold). Fill if absent.
    if data.get("class_reference") is not None and not getattr(obs, "class_reference", None):
        cr = data["class_reference"]
        obs.class_reference = cr if isinstance(cr, tvbo_datamodel.ClassReference) else tvbo_datamodel.ClassReference(**cr)

    # parameters: keyed collection — fill each missing key, keep recipe overrides.
    if data.get("parameters"):
        params = obs.parameters if getattr(obs, "parameters", None) else {}
        for pname, pval in data["parameters"].items():
            if pname in params:
                continue
            if isinstance(pval, dict):
                pval = {"name": pname, **pval}
                params[pname] = tvbo_datamodel.Parameter(**pval)
            else:
                params[pname] = tvbo_datamodel.Parameter(name=pname, value=pval)
        obs.parameters = params

    if data.get("functions") and functions_sink is not None:
        for fname, fdef in data["functions"].items():
            if fname in functions_sink:
                continue
            if isinstance(fdef, tvbo_datamodel.Function):
                functions_sink[fname] = fdef
            else:
                # The dict key is the function's name; a redundant inner ``name`` (against the keyed-collection convention) must not double the keyword. Merge with the key winning.
                functions_sink[fname] = tvbo_datamodel.Function(**{**fdef, "name": fname})

    return True


class Observation(tvbo_datamodel.Observation):
    """Wrapper around the LinkML Observation datamodel with convenience factory methods for loading from file, database, or TVB monitors."""

    @classmethod
    def from_file(cls, path: str) -> "Observation":
        """Load an Observation from a YAML file."""
        from tvbo.utils import yaml_loader

        return yaml_loader.load(str(path), target_class=cls)

    @classmethod
    def from_db(cls, name: str) -> "Observation":
        """Load an Observation by name from the tvbo database."""
        from tvbo.data.registry import resolve

        return cls.from_file(str(resolve("Observation", name)))

    @classmethod
    def list_db(cls) -> list[str]:
        """List available observation models in the tvbo database."""
        from tvbo.data.registry import list_entries

        return list_entries("Observation")

    def render_code(self, format="tvb"):
        """Generate backend code that creates this monitor.

        Parameters
        ----------
        format : str
            Target backend. Currently ``"tvb"`` is supported.

        Returns
        -------
        str
            Executable Python code string.
        """
        if format != "tvb":
            raise ValueError(f"Format {format!r} not supported for Observation. Use 'tvb'.")

        from tvbo import templates
        from tvbo.codegen.templater import format_code

        # Wrap single observation as the template expects experiment.observations dict
        class _Ctx:
            observations = {str(self.name): self}

        template = templates.lookup.get_template("tvbo-tvb-observation.py.mako")
        rendered = template.render(experiment=_Ctx())
        return format_code(rendered)

    def execute(self, format="tvb"):
        """Convert this observation to a backend monitor object.

        Parameters
        ----------
        format : str
            Target backend. Currently ``"tvb"`` is supported.

        Returns
        -------
        tvb.simulator.monitors.Monitor
            Configured TVB monitor instance.
        """
        if format != "tvb":
            raise ValueError(f"Format {format!r} not supported for Observation. Use 'tvb'.")

        code = self.render_code("tvb")
        ns = {}
        exec(code, ns)
        monitors = ns.get("monitors", [])
        if monitors:
            return monitors[0]
        raise RuntimeError("Template produced no monitors")

    # Operation-type labels and face colours for flowchart boxes.
    _OP_COLORS = {
        "kernel": "#dbeafe",  # generates a function over time (e.g. HRF)
        "convolution": "#fef9c3",  # folds two signals (e.g. fftconvolve)
        "projection": "#dcfce7",  # maps over node/space dimension
        "temporal": "#fae8ff",  # averages/subsamples along time
        "transform": "#f1f5f9",  # general equation without dimension tag
        "identity": "#f8fafc",  # passthrough / no-op
    }

    @staticmethod
    def _step_op_type(step) -> str:
        """Return the structural operation type of a single pipeline step."""
        if getattr(step, "time_range", None):
            return "kernel"
        if getattr(step, "callable", None):
            fn = str(getattr(step.callable, "name", "") or "").lower()
            if "convolve" in fn:
                return "convolution"
            return "callable"
        if getattr(step, "equation", None):
            dim = str(getattr(step, "apply_on_dimension", "") or "").lower()
            if dim == "node":
                return "projection"
            if dim == "time":
                return "temporal"
            return "transform"
        return "transform"

    def _classify_pipeline(self) -> str:
        """Classify the whole observation by its dominant pipeline axiom.

        Axioms (first match wins):
        * no pipeline steps            → ``"identity"``
        * any step with ``time_range`` → ``"kernel"``   (generates a kernel fn)
        * any step with ``callable``   → ``"callable"`` (external function)
        * any node-dimension step      → ``"projection"``
        * any time-dimension step      → ``"temporal"``
        * otherwise                    → ``"transform"``
        """
        steps = list(self.pipeline or [])
        if not steps:
            return "identity"
        types = [self._step_op_type(s) for s in steps]
        for dominant in ("kernel", "convolution", "callable", "projection", "temporal"):
            if dominant in types:
                return dominant
        return "transform"

    def plot(self, ax=None, **kwargs):
        """Plot a visual summary of this observation model.

        The plot type is derived purely from the pipeline structure:

        * **kernel step present** (step with ``time_range``): evaluates and
          plots the kernel function.
        * **all other cases**: draws an annotated pipeline flowchart where each
          box is tagged with its structural operation type (projection, temporal, transform, callable, …).

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Axes to draw into. A new figure is returned when ``ax`` is ``None``.
        **kwargs
            Forwarded to the underlying plot call.
        """
        import matplotlib.pyplot as plt

        return_fig = ax is None
        if return_fig:
            fig, ax = plt.subplots(figsize=(3, 2.5))

        obs_class = self._classify_pipeline()
        if obs_class == "kernel":
            self._plot_kernel(ax, **kwargs)
        else:
            self._plot_pipeline_flowchart(ax, **kwargs)

        ax.set_title(str(self.label or self.name))

        if return_fig:
            plt.close(fig)
            return fig

    def _plot_kernel(self, ax, **kwargs):
        """Evaluate and plot the first pipeline step that has a ``time_range``."""
        import numpy as np
        from sympy import lambdify, sympify, Symbol

        kernel_step = next(
            (s for s in (self.pipeline or []) if getattr(s, "time_range", None)),
            None,
        )

        if kernel_step is None or kernel_step.equation is None:
            ax.text(0.5, 0.5, "No kernel step found", ha="center", va="center", transform=ax.transAxes)
            return

        # Collect parameter values: inline equation params + step arguments
        param_subs = {}
        for pname, pobj in (getattr(kernel_step.equation, "parameters", None) or {}).items():
            val = getattr(pobj, "value", None)
            if val is not None:
                try:
                    param_subs[Symbol(str(pname))] = float(val)
                except (TypeError, ValueError):
                    pass

        args = kernel_step.arguments
        arg_iter = args.values() if hasattr(args, "values") else (args or [])
        for arg in arg_iter:
            aname = str(getattr(arg, "name", ""))
            aval = getattr(arg, "value", None)
            if aval is not None:
                try:
                    param_subs[Symbol(aname)] = float(aval)
                except (TypeError, ValueError):
                    pass

        # Build time axis from resolved parameter values
        dt = float(param_subs.get(Symbol("stock_dt"), 0.004))
        t_end = float(param_subs.get(Symbol("duration"), 20.0))
        t_vals = np.arange(dt, t_end, dt)

        t_sym = Symbol("t")
        try:
            expr = sympify(
                str(kernel_step.equation.rhs),
                locals={str(k): k for k in param_subs} | {"t": t_sym},
            )
            expr_sub = expr.subs(param_subs)
            free = expr_sub.free_symbols
            y = lambdify(list(free), expr_sub, modules="numpy")(t_vals) if free else float(expr_sub) * np.ones_like(t_vals)
        except Exception:
            ax.text(0.5, 0.5, "Kernel evaluation failed", ha="center", va="center", transform=ax.transAxes)
            return

        y = np.asarray(y, dtype=float)
        peak = np.nanmax(np.abs(y))
        if peak > 0:
            y = y / peak

        ax.plot(t_vals, y, **kwargs)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("kernel (norm.)")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    def _plot_pipeline_flowchart(self, ax, **kwargs):
        """Draw pipeline steps as a vertical flowchart.

        Each box is colour-coded and tagged with the structural operation type (kernel / projection / temporal / transform / callable / identity).
        """
        import numpy as np
        import matplotlib.patches as mpatches

        steps = list(self.pipeline or [])

        ax.set_axis_off()

        if not steps:
            label = str((self.class_reference.name if self.class_reference else None) or self.name)
            ax.text(
                0.5,
                0.5,
                f"{label}\n(identity)",
                ha="center",
                va="center",
                fontsize=8,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=self._OP_COLORS["identity"], edgecolor="gray", linewidth=0.8),
            )
            return

        n = len(steps)
        y_positions = np.linspace(0.88, 0.08, n)
        box_h = min(0.13, 0.75 / n)
        box_w = 0.78

        for idx, (step, yc) in enumerate(zip(steps, y_positions)):
            op = self._step_op_type(step)
            fc = self._OP_COLORS.get(op, self._OP_COLORS["transform"])
            label = str(getattr(step, "label", None) or getattr(step, "name", None) or getattr(step, "output", f"step {idx}"))
            rect = mpatches.FancyBboxPatch(
                (0.5 - box_w / 2, yc - box_h / 2),
                box_w,
                box_h,
                boxstyle="round,pad=0.02",
                linewidth=0.8,
                edgecolor="#94a3b8",
                facecolor=fc,
                transform=ax.transAxes,
                clip_on=False,
            )
            ax.add_patch(rect)
            # step name (main line) + operation tag (smaller, below)
            ax.text(0.5, yc + 0.012, label, ha="center", va="center", fontsize=7, transform=ax.transAxes)
            ax.text(0.5, yc - 0.022, f"[{op}]", ha="center", va="center", fontsize=5, color="#64748b", transform=ax.transAxes)

            if idx < n - 1:
                gap_start = yc - box_h / 2
                gap_end = y_positions[idx + 1] + box_h / 2
                ax.annotate(
                    "",
                    xy=(0.5, gap_end),
                    xytext=(0.5, gap_start),
                    xycoords="axes fraction",
                    textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#64748b", lw=0.8),
                )
