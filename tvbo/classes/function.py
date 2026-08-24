# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""Function Classes.

Extended Function and LossFunction classes with code generation methods.
Inherits from the LinkML datamodel and adds rendering/execution capabilities.

Usage
-----

From YAML file:

    from tvbo import Function, LossFunction

    func = Function.from_file("correlation.yaml")
    code = func.render_code(format='jax')
    callable_fn = func.to_callable()

From YAML string:

    func = Function.from_string(yaml_string)

From datamodel object:

    from tvbo.datamodel import schema as tvbo_datamodel
    dm_func = tvbo_datamodel.Function(name='sigmoid', ...)
    func = Function.from_datamodel(dm_func)
"""

import os
from collections.abc import Callable

from tvbo.datamodel import schema as tvbo_datamodel
from tvbo.utils import yaml_loader


class Function(tvbo_datamodel.Function):
    """Extended Function class with code generation and execution methods.

    Inherits all schema fields from tvbo_datamodel.Function and adds:
    - Factory constructors: from_file, from_string, from_datamodel
    - Code generation: render_code, to_jax, to_numpy
    - Execution: to_callable
    """

    def __init__(self, name: str = "Function", **kwargs):
        if name is not None:
            kwargs["name"] = str(name)
        super().__init__(**kwargs)

    # Factory constructors
    @classmethod
    def from_datamodel(cls, func: tvbo_datamodel.Function) -> "Function":
        """Create Function from a tvbo_datamodel.Function instance."""
        return cls(**func._as_dict)

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> "Function":
        """Load Function from a YAML file."""
        return yaml_loader.load(str(path), cls)

    @classmethod
    def from_string(cls, yaml_str: str) -> "Function":
        """Load Function from a YAML string."""
        return yaml_loader.loads(yaml_str, cls)

    # Code generation
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

    def to_jax(self, **kwargs) -> str:
        """Generate JAX code for this function."""
        return self.render_code(format="jax", **kwargs)

    def to_numpy(self, **kwargs) -> str:
        """Generate NumPy code for this function."""
        return self.render_code(format="numpy", **kwargs)

    def to_python(self, **kwargs) -> str:
        """Generate pure Python code for this function."""
        return self.render_code(format="python", **kwargs)

    # Execution
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

    # Convenience properties
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

    def __repr__(self) -> str:
        rhs = self.equation.rhs if self.equation else None
        return f"Function({self.name!r}, equation={rhs!r})"


class LossFunction(tvbo_datamodel.LossFunction):
    """Extended LossFunction class with code generation and execution methods.

    Inherits all schema fields from tvbo_datamodel.LossFunction and adds:
    - Factory constructors: from_file, from_string, from_datamodel
    - Code generation: render_code, to_jax, to_numpy
    - Execution: to_callable

    LossFunction extends Function with aggregation specification for per-element losses (e.g., mean over nodes).
    """

    def __init__(self, name: str = "LossFunction", **kwargs):
        if name is not None:
            kwargs["name"] = str(name)
        super().__init__(**kwargs)

    # Factory constructors
    @classmethod
    def from_datamodel(cls, func: tvbo_datamodel.LossFunction) -> "LossFunction":
        """Create LossFunction from a tvbo_datamodel.LossFunction instance."""
        return cls(**func._as_dict)

    @classmethod
    def from_file(cls, path: str | os.PathLike) -> "LossFunction":
        """Load LossFunction from a YAML file."""
        return yaml_loader.load(str(path), cls)

    @classmethod
    def from_string(cls, yaml_str: str) -> "LossFunction":
        """Load LossFunction from a YAML string."""
        return yaml_loader.loads(yaml_str, cls)

    # Code generation
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

    # Execution
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

    # Convenience properties
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

    def __repr__(self) -> str:
        rhs = self.equation.rhs if self.equation else None
        agg = f", aggregate={self.aggregation_type} over {self.aggregation_dimension}" if self.aggregate else ""
        return f"LossFunction({self.name!r}, equation={rhs!r}{agg})"
