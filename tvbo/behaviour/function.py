#
# Module: behaviour/function.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""What a ``Function`` record does: resolve itself to a callable, render itself as code, and answer for its own symbols.

One declared function had two runtime classes — one in ``classes/function`` that generated code, one in ``classes/observation`` that lambdified an expression — and which one a caller got depended on which module it had imported. They shared three method names and agreed on two of them; the third, ``render_code``, meant a whole ``def`` on one and a bare expression on the other. Both are here, under names that say which is which: `render_code` emits the function, `render_expression` emits its right-hand side.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable

import sympy
from sympy import Eq, IndexedBase, Lambda, Symbol, lambdify


class FunctionBehaviour:
    """Everything a declared function does, on both generated forms."""

    @classmethod
    def from_datamodel(cls, func):
        """Create a Function from a generated record of one."""
        from jsonasobj2 import as_dict

        return cls(**as_dict(func))

    @classmethod
    def from_file(cls, path: str | os.PathLike):
        """Load a Function from a YAML file."""
        from tvbo.utils import yaml_loader

        return yaml_loader.load(str(path), cls)

    @classmethod
    def from_string(cls, yaml_str: str):
        """Load a Function from a YAML string."""
        from tvbo.utils import yaml_loader

        return yaml_loader.loads(yaml_str, cls)

    @classmethod
    def from_python(cls, function_instance, **kwargs):
        """Create Function from a Python callable."""
        from tvbo.classes.observation import functioninstance2metadata

        kwargs = functioninstance2metadata(function_instance, **kwargs)
        return cls(**kwargs)

    @classmethod
    def from_ontology(cls, ontology_instance, **kwargs):
        """Create Function from an ontology instance."""
        from tvbo.classes.observation import functioninstance2metadata

        kwargs = functioninstance2metadata(ontology_instance, **kwargs)
        return cls(**kwargs)

    @classmethod
    def from_db(cls, name: str):
        """Load a Function by name from the tvbo database."""
        from tvbo.data.registry import resolve

        return cls.from_file(str(resolve("Function", name)))

    @classmethod
    def list_db(cls) -> list[str]:
        """List available observation models in the tvbo database."""
        from tvbo.data.registry import list_entries

        return list_entries("Function")

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
        from tvbo.ontology import owl as ontology

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

    def symbol_scope(self):
        """The namespace this function's equation is parsed against.

        Its own parameters, plus its arguments as `IndexedBase` so an argument can be indexed in the body. Shared by every caller that parses this equation, so a function's rendered graph cannot resolve a name differently from its equation.
        """
        from tvbo.parse.symbols import BUILTIN_SHADOW

        return BUILTIN_SHADOW.extend(
            {str(p): p for p in self.get_parameters(key_as_symbol=True)},
            {str(a): IndexedBase(a) for a in self.arguments},
        )

    def get_equation(self):
        """Build the function as a SymPy equation.

        Parses the stored right-hand-side string into an expression, treats the function's arguments as `IndexedBase` symbols, and returns an equality whose left-hand side is the named function applied to its arguments.

        Returns:
            A SymPy `Eq` relating the function call to its parsed expression.
        """
        expression = self.symbol_scope().parse(self.equation.rhs)
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

    @property
    def sympy_expression(self):
        """Return the parsed SymPy expression for this function's equation."""
        if not self.equation or not self.equation.rhs:
            return None
        from tvbo.parse.expression import parse_eq

        return parse_eq(self.equation)

    @property
    def latex(self) -> str:
        """Return LaTeX representation of the function equation."""
        expr = self.sympy_expression
        if expr is None:
            return ""
        from sympy import latex

        return latex(expr)

    def render_code(
        self,
        format: str = "jax",
        user_functions: dict[str, str] | None = None,
        render_func: Callable | None = None,
    ) -> str:
        """Generate Python code for this function.

        Parameters
        ----------
        format : str
            Output format: 'jax', 'numpy', 'python'
        user_functions : dict, optional
            Custom function name mappings for the printer.
            Example: {'sigmoid': 'sigmoid'} to preserve function name
        render_func : callable, optional
            Custom render function for model context.

        Returns:
        -------
        str
            Python code string defining the function

        Examples:
        --------
        >>> func = Function.from_string(yaml_str)
        >>> print(func.render_code())
        def sigmoid(x):
            return 1/(1 + jnp.exp(-x))
        """
        # Lazy import to avoid circular dependency
        from tvbo.codegen.functions import generate_function

        return generate_function(
            self,
            format=format,
            user_functions=user_functions,
            render_func=render_func,
        )

    def render_expression(self, format="python", **kwargs):
        """Render just the function's right-hand side as backend source code.

        `render_code` emits the whole ``def``; this emits the expression alone, for a caller splicing it into code of its own.

        Args:
            format: Target backend passed to the expression renderer.
            **kwargs: Additional options forwarded to the renderer.

        Returns:
            The rendered code for the equation's right-hand side.
        """
        from tvbo.parse.expression import render_expression

        return render_expression(self.get_equation().rhs, format=format, **kwargs)

    def to_jax(self, **kwargs) -> str:
        """Generate JAX code for this function."""
        return self.render_code(format="jax", **kwargs)

    def to_numpy(self, **kwargs) -> str:
        """Generate NumPy code for this function."""
        return self.render_code(format="numpy", **kwargs)

    def to_python(self, **kwargs) -> str:
        """Generate pure Python code for this function."""
        return self.render_code(format="python", **kwargs)

    def to_callable(
        self,
        format: str = "jax",
        user_functions: dict[str, str] | None = None,
        namespace: dict | None = None,
    ) -> Callable:
        """Generate and execute function code, returning the callable.

        Parameters
        ----------
        format : str
            Output format: 'jax', 'numpy'
        user_functions : dict, optional
            Custom function name mappings
        namespace : dict, optional
            Namespace for exec(). If None, creates one with jnp/np imports.

        Returns:
        -------
        callable
            The generated function as a callable

        Examples:
        --------
        >>> func = Function.from_string(sigmoid_yaml)
        >>> sigmoid = func.to_callable()
        >>> sigmoid(0.0)
        0.5
        """
        # Lazy import to avoid circular dependency
        from tvbo.codegen.functions import function_to_callable

        return function_to_callable(
            self,
            format=format,
            user_functions=user_functions,
            namespace=namespace,
        )

    def execute(self, format="python", fill_in_parameters=True, parameters=None, **kwargs):
        """Compile the function into an executable callable.

        Returns the recorded Python callable when one is available; otherwise lambdifies the symbolic equation for the requested backend. Supplied parameters that do not appear in the equation are discarded, and the function's stored parameter values can optionally be substituted in before compilation.

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
        if parameters is None:
            parameters = {}
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

    def plot(self, format="python", plotting_kwargs=None, **kwargs):
        """Plot the function's output against its input.

        For a single-argument function, the input array (supplied via `kwargs` under the argument name) is plotted against the evaluated output; for multi-argument functions the output is plotted directly using the stored parameter values.

        Args:
            format: Backend used to compile the function for evaluation.
            plotting_kwargs: Keyword arguments forwarded to `matplotlib`.
            **kwargs: Input values keyed by argument name.
        """
        import matplotlib.pyplot as plt

        if plotting_kwargs is None:
            plotting_kwargs = {}
        function = self.execute(format=format)
        args = self.arguments
        if len(args) == 1:
            fin = kwargs.get(next(iter(args.values())).name)
            plt.plot(fin, function(fin), **plotting_kwargs)
            plt.xlabel(next(iter(self.arguments.values())).unit)
        else:
            plt.plot(function(**{**kwargs, **self.get_parameters()}), **plotting_kwargs)
        pass

    def plot_metadata_graph(self, ax=None, node_kwargs=None, edge_kwargs=None, edge_labels=True):
        """Draw a graph of the function's metadata.

        Builds a directed graph linking the function node to its equation, software requirements and arguments, then renders it with a radial layout.

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
        import matplotlib.pyplot as plt

        if edge_kwargs is None:
            edge_kwargs = {}
        if node_kwargs is None:
            node_kwargs = {}
        if ax is None:
            fig, ax = plt.subplots()
            return_fig = True
        else:
            return_fig = False
        import matplotlib.pyplot as plt
        import networkx as nx
        from sympy import Float, Rational, latex

        from tvbo.plot.ontology import draw_custom_nodes

        G = nx.DiGraph()
        func_name = self.acronym or self.name
        G.add_node(func_name, label=f"{func_name}")
        if self.equation and self.equation.rhs:
            expression = self.symbol_scope().parse(self.equation.rhs)
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

    def __repr__(self) -> str:
        rhs = self.equation.rhs if self.equation else None
        return f"Function({self.name!r}, equation={rhs!r})"


class LossFunctionBehaviour(FunctionBehaviour):
    """A loss function is a function with an aggregation: everything `FunctionBehaviour` does, plus how a per-element loss is reduced to a scalar."""

    @classmethod
    def from_datamodel(cls, func):
        """Create a LossFunction from a generated record of one."""
        from jsonasobj2 import as_dict

        return cls(**as_dict(func))

    @classmethod
    def from_file(cls, path: str | os.PathLike):
        """Load a LossFunction from a YAML file."""
        from tvbo.utils import yaml_loader

        return yaml_loader.load(str(path), cls)

    @classmethod
    def from_string(cls, yaml_str: str):
        """Load a LossFunction from a YAML string."""
        from tvbo.utils import yaml_loader

        return yaml_loader.loads(yaml_str, cls)

    @property
    def sympy_expression(self):
        """Return the parsed SymPy expression for this function's equation."""
        if not self.equation or not self.equation.rhs:
            return None
        from tvbo.parse.expression import parse_eq

        # For loss functions, inner function names need to be recognized
        inner_funcs = []
        rhs = self.equation.rhs
        # Simple heuristic: look for function calls in the expression
        import re

        func_calls = re.findall(r"(\w+)\s*\(", rhs)
        inner_funcs = [f for f in func_calls if f not in ["Sum", "Mean", "sqrt", "exp", "log", "abs", "mean", "std"]]
        return parse_eq(self.equation, functions=inner_funcs)

    @property
    def latex(self) -> str:
        """Return LaTeX representation of the loss function equation."""
        expr = self.sympy_expression
        if expr is None:
            return ""
        from sympy import latex

        return latex(expr)

    def render_code(
        self,
        format: str = "jax",
        user_functions: dict[str, str] | None = None,
        inner_func_names: list[str] | None = None,
    ) -> str:
        """Generate Python code for this loss function with aggregation.

        Parameters
        ----------
        format : str
            Output format: 'jax', 'numpy'
        user_functions : dict, optional
            Custom function name mappings
        inner_func_names : list, optional
            Names of inner functions that should be recognized.
            Example: ['correlation'] for "1 - correlation(x, y)"

        Returns:
        -------
        str
            Python code string defining the loss function

        Examples:
        --------
        >>> loss = LossFunction.from_string(loss_yaml)
        >>> print(loss.render_code(inner_func_names=['correlation']))
        def spectral_loss(sim, target):
            def _per_element_loss(sim, target):
                return 1 - correlation(sim, target)
            per_element_losses = jax.vmap(_per_element_loss)(sim, target)
            return jnp.mean(per_element_losses)
        """
        # Lazy import to avoid circular dependency
        from tvbo.codegen.functions import generate_loss_function

        return generate_loss_function(
            self,
            format=format,
            user_functions=user_functions,
            inner_func_names=inner_func_names,
        )

    def to_jax(self, **kwargs) -> str:
        """Generate JAX code for this loss function."""
        return self.render_code(format="jax", **kwargs)

    def to_numpy(self, **kwargs) -> str:
        """Generate NumPy code for this loss function."""
        return self.render_code(format="numpy", **kwargs)

    def to_callable(
        self,
        format: str = "jax",
        user_functions: dict[str, str] | None = None,
        inner_func_names: list[str] | None = None,
        namespace: dict | None = None,
    ) -> Callable:
        """Generate and execute loss function code, returning the callable.

        Parameters
        ----------
        format : str
            Output format: 'jax', 'numpy'
        user_functions : dict, optional
            Custom function name mappings
        inner_func_names : list, optional
            Names of inner functions to recognize
        namespace : dict, optional
            Namespace for exec(). If None, creates one with jnp/np/jax imports.

        Returns:
        -------
        callable
            The generated loss function as a callable
        """
        code = self.render_code(
            format=format,
            user_functions=user_functions,
            inner_func_names=inner_func_names,
        )

        # Start with user-provided namespace or empty dict
        if namespace is None:
            namespace = {}
        else:
            # Copy to avoid mutating the user's dict
            namespace = dict(namespace)

        # Always add required imports for the format
        if format == "jax":
            import jax
            import jax.numpy as jnp

            namespace.setdefault("jax", jax)
            namespace.setdefault("jnp", jnp)
        elif format == "numpy":
            import numpy as np

            namespace.setdefault("np", np)

        exec(code, namespace)
        return namespace[str(self.name)]

    def __repr__(self) -> str:
        rhs = self.equation.rhs if self.equation else None
        agg = f", aggregate={self.aggregation_type} over {self.aggregation_dimension}" if self.aggregate else ""
        return f"LossFunction({self.name!r}, equation={rhs!r}{agg})"

    @property
    def aggregation_type(self) -> str | None:
        """Return the aggregation type as a string (e.g., 'mean', 'sum')."""
        if not self.aggregate or not self.aggregate.type:
            return None
        agg_type = self.aggregate.type
        return getattr(agg_type, "text", str(agg_type))

    @property
    def aggregation_dimension(self) -> str | None:
        """Return the aggregation dimension as a string (e.g., 'node')."""
        if not self.aggregate or not self.aggregate.over:
            return None
        return str(self.aggregate.over)
