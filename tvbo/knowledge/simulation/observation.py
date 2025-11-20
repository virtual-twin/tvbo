import importlib
import inspect
import platform
from types import FunctionType

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
from tvbo.knowledge.simulation.equations import _clash1
from tvbo.data.types import TimeSeries
from tvbo.datamodel import tvbo_datamodel
from tvbo.export.code import render_expression
from tvbo.knowledge import ontology
from tvbo.plot.network import draw_custom_arrows, draw_custom_edges, draw_custom_nodes


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
        arguments = {
            name: {"name": name}
            for name, param in signature.parameters.items()
            if param.default == inspect._empty
        }
        parameters = {
            name: {"name": name, "value": param.default}
            for name, param in signature.parameters.items()
            if param.default != inspect._empty
            and isinstance(param.default, (int, float))
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
        merged["source_code"] = (source_code.strip() if isinstance(source_code, str) else None)

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
                submodule = module.__name__[len(prefix):] if module.__name__.startswith(prefix) else ""

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
    kwargs = {
        **kwargs,  # TODO: remember dict unpacking prioritizes keys from unpacked dict over keys defined later in the same dict!
        "transformation": {
            "name": instance.name,
            "arguments": {
                arg.name: {"name": arg.name, "unit": arg.unit.first()}
                for arg in instance.has_argument
            },
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
        from linkml_runtime.loaders import yaml_loader

        return yaml_loader.load(filepath, target_class=cls)

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
        if (
            hasattr(self, "acronym")
            and self.acronym
            and hasattr(ontology.onto, self.acronym)
        ):
            return getattr(ontology.onto, self.acronym)
        return None

    @property
    def metadata(self):
        """Backward compatibility: return self (which is now the datamodel)."""
        return self

    def get_parameters(self, key_as_symbol=False):
        parameters = {
            Symbol(k) if key_as_symbol else k: v.value
            for k, v in self.equation.parameters.items()
        }
        return parameters

    def get_equation(self):
        parameters = self.get_parameters(key_as_symbol=True)
        clash = {str(p): p for p in parameters.keys()}
        clash.update({str(a): IndexedBase(a) for a in self.arguments})
        expression = parse_expr(self.equation.rhs, clash)
        function = sympy.Function(self.acronym or self.name)(
            *(Symbol(a) for a in self.arguments)
        )
        return Eq(function, expression)

    def get_symbolic_function(self):
        equation = self.get_equation()
        parameters = self.get_parameters()
        return Lambda(equation.lhs.args, equation)

    def execute(
        self, format="python", fill_in_parameters=True, parameters={}, **kwargs
    ):
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
        default_values = {str(k): v for k, v in parameters.items()}
        eq = equation.rhs
        if fill_in_parameters:
            eq = eq.subs(parameters)
        eq = eq.subs("e", "E")
        arguments = equation.lhs.args + tuple(
            [k for k in parameters.keys() if Symbol(k) in eq.free_symbols]
        )
        function = lambdify(arguments, eq, modules=modules)

        if format == "jax" and kwargs.get("jit", False):
            import jax

            function = jax.jit(
                function,
                static_argnames=[
                    str(arg) for arg in arguments if str(arg) == "stepsize"
                ],
            )
        return function

    def apply(self, **kwargs):
        return self.execute()(**kwargs)

    def render_code(self, format="python", **kwargs):
        return render_expression(self.get_equation().rhs, format=format, **kwargs)

    def plot(self, format="python", plotting_kwargs={}, **kwargs):
        function = self.execute(format=format)
        args = self.arguments
        if len(args) == 1:
            fin = kwargs.get(next(iter(args.values())).name)
            plt.plot(fin, function(fin, **plotting_kwargs))
            plt.xlabel(next(iter(self.arguments.values())).unit)
        else:
            plt.plot(function(**{**kwargs, **self.get_parameters()}), **plotting_kwargs)
        pass

    def plot_metadata_graph(
        self, ax=None, node_kwargs={}, edge_kwargs={}, edge_labels=True
    ):
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
            rounded_expression = expression.xreplace(
                {n: Float(round(float(n), 4)) for n in expression.atoms(Float)}
            )
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
        if isinstance(data, TimeSeries):
            data = data.data
            time = data.time
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
        self.add_function(
            function,
            argument_mapping=argument_mapping,
            function_type="derivative",
            **kwargs,
        )

    def add_projection_model(self, function, argument_mapping={}, **kwargs):
        self.add_function(
            function,
            argument_mapping=argument_mapping,
            function_type="projection",
            **kwargs,
        )
        # TODO: Finish implementation
        pass

    def plot_graph(
        self, ax=None, plot_edge_labels=True, node_kwargs={}, edge_kwargs={}
    ):
        try:
            pos = nx.nx_pydot.graphviz_layout(
                self.graph, prog="dot"
            )  # Layout for graph visualization
        except:
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
        draw_custom_nodes(
            self.graph, pos, ax=ax, facecolor="white", edgecolor="grey", **node_kwargs
        )
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
                    print("Node", node_label, "has no data")
                continue

            time = (
                self._run_node_function(
                    node_label, ensure_4d=ensure_4d, time_mapping=apply_on_time
                )
                if apply_on_time
                else self.current_time
            )

            # if apply_on_time:
            #     print(node_label, self.current_time.shape, np.max(self.current_time))

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
        output_data = self.graph.nodes["Output"]["data"][
            tuple(slice(0, dim) for dim in input_shape)
        ]
        ts = self.orig_timeseries.copy()
        ts.data = output_data
        ts.time = self.current_time
        return ts

    def get_node_data(self, node):
        data = self.graph.nodes[node].get("data", None)
        time = self.graph.nodes[node].get(
            "time",
            (
                np.arange(data.shape[0])
                if self.graph.nodes[node].get("function_type", None) != "derivative"
                else np.array([0])
            ),
        )
        return TimeSeries(time, data)

    def get_function_output(self, function_name):
        """
        Get the output of a specific function after execution.

        Args:
            function_name (str): The name of the function whose output to retrieve.

        Returns:
            The result produced by the function.
        """
        return self.results.get(function_name, None)

    def plot_node_data(self, node, ax):
        data = self.get_node_data(node)
        ax.plot(data, label=node)
        # plt.title(node)
        # plt.show()

    def plot_graph_data(self, ax=None):
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
