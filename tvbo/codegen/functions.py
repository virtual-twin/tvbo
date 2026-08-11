# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""Function Code Generation.

Generate executable Python code from TVBO Function and LossFunction objects.
Supports standalone usage without SimulationExperiment or Dynamics context.

Usage
-----

From YAML string:

    from tvbo.codegen.functions import generate_function, generate_loss_function
    from linkml_runtime.loaders import yaml_loader
    from tvbo.datamodel.schema import Function, LossFunction

    # Load function from YAML
    func = yaml_loader.loads(yaml_string, Function)

    # Generate code
    code = generate_function(func, format='jax')
    print(code)

    # Execute the generated code
    exec(code, globals())
    result = my_function(x, y)

From Function object directly:

    from tvbo.datamodel.schema import Function, Equation, Argument

    func = Function(
        name='correlation',
        equation=Equation(rhs='mean((x - mean(x)) * (y - mean(y))) / (std(x) * std(y))'),
        arguments=[Argument(name='x'), Argument(name='y')]
    )
    code = generate_function(func)

With aggregation (LossFunction):

    loss = yaml_loader.loads(loss_yaml, LossFunction)
    code = generate_loss_function(loss, inner_func_names=['correlation'])
"""

from collections.abc import Callable


def generate_function(
    func,
    format: str = "jax",
    user_functions: dict[str, str] | None = None,
    render_func: Callable | None = None,
) -> str:
    """Generate Python code for a function definition.

    Handles all function types:
    - Equation-based: Symbolic expressions parsed with SymPy
    - Source-code: Raw Python code (func.source_code or equation.pycode)
    - Time-range: Kernel/signal generators
    - Callable: External function references (use generate_callable_function)

    Parameters
    ----------
    func : Function
        A TVBO Function object (from datamodel or loaded from YAML)
    format : str
        Output format: 'jax', 'numpy', 'python'
    user_functions : dict, optional
        Custom function name mappings for the printer.
        Example: {'sigmoid': 'sigmoid'} to preserve function name
    render_func : callable, optional
        Custom render function for model context.
        If provided, uses this instead of render_expression.

    Returns:
    -------
    str
        Python code string defining the function

    Examples:
    --------
    >>> from tvbo.datamodel.schema import Function, Equation
    >>> func = Function(
    ...     name='sigmoid',
    ...     equation=Equation(rhs='1 / (1 + exp(-x))'),
    ...     arguments=[{'name': 'x'}]
    ... )
    >>> print(generate_function(func))
    def sigmoid(x):
        return 1/(1 + jnp.exp(-x))
    """
    # Lazy import to avoid circular dependency
    from tvbo.templates import lookup

    template = lookup.get_template("base/function-def.mako")
    return (
        template.get_def("function_def")
        .render(
            func=func,
            format=format,
            user_functions=user_functions,
            render_func=render_func,
        )
        .strip()
    )


def generate_loss_function(
    func,
    format: str = "jax",
    user_functions: dict[str, str] | None = None,
    inner_func_names: list[str] | None = None,
) -> str:
    """Generate Python code for a loss function with aggregation.

    Supports both:
    - SymPy indexed notation: Sum(f(x[i], y[i]), (i, 0, n-1))
    - Metadata aggregation: aggregate.over + aggregate.type

    Parameters
    ----------
    func : LossFunction
        A TVBO LossFunction object with optional aggregate specification
    format : str
        Output format: 'jax', 'numpy'
    user_functions : dict, optional
        Custom function name mappings
    inner_func_names : list, optional
        Names of inner functions that should be recognized
        Example: ['correlation'] for "1 - correlation(x, y)"

    Returns:
    -------
    str
        Python code string defining the loss function

    Examples:
    --------
    >>> from tvbo.datamodel.schema import LossFunction, Equation, Aggregation
    >>> loss = LossFunction(
    ...     name='spectral_loss',
    ...     equation=Equation(rhs='1 - correlation(sim, target)'),
    ...     aggregate=Aggregation(over='node', type='mean'),
    ... )
    >>> print(generate_loss_function(loss, inner_func_names=['correlation']))
    def spectral_loss(sim, target):
        def _per_element_loss(sim, target):
            return 1 - correlation(sim, target)
        per_element_losses = jax.vmap(_per_element_loss)(sim, target)
        return jnp.mean(per_element_losses)
    """
    # Lazy import to avoid circular dependency
    from tvbo.templates import lookup

    template = lookup.get_template("base/function-def.mako")
    return (
        template.get_def("function_def_with_aggregation")
        .render(
            func=func,
            format=format,
            user_functions=user_functions,
            inner_func_names=inner_func_names,
        )
        .strip()
    )


def generate_indexed_function(
    func,
    format: str = "jax",
    user_functions: dict[str, str] | None = None,
) -> str:
    """Generate Python code for a function with indexed aggregation.

    Properly handles mathematical notation like:
    - Sum(x[i] * y[i], (i, 0, n-1)) -> jnp.sum(x * y)
    - Mean((x[i] - mean(x))**2, (i, 0, n-1)) -> jnp.mean((x - jnp.mean(x))**2)

    Parameters
    ----------
    func : Function
        A TVBO Function object with indexed expression
    format : str
        Output format: 'jax', 'numpy'
    user_functions : dict, optional
        Custom function name mappings

    Returns:
    -------
    str
        Python code string
    """
    # Lazy import to avoid circular dependency
    from tvbo.templates import lookup

    template = lookup.get_template("base/function-def.mako")
    return (
        template.get_def("indexed_function")
        .render(
            func=func,
            format=format,
            user_functions=user_functions,
        )
        .strip()
    )


def generate_callable_function(
    func,
    format: str = "jax",
    callable_ref: str | None = None,
) -> str:
    """Generate Python code for a callable-based function with vmap.

    Handles external function calls with proper array dimension mapping.

    Parameters
    ----------
    func : Function
        A TVBO Function object with callable specification
    format : str
        Output format: 'jax', 'numpy'
    callable_ref : str, optional
        Override the callable reference name

    Returns:
    -------
    str
        Python code string
    """
    # Lazy import to avoid circular dependency
    from tvbo.templates import lookup

    template = lookup.get_template("base/function-def.mako")
    return (
        template.get_def("callable_function")
        .render(
            func=func,
            format=format,
            callable_ref=callable_ref,
        )
        .strip()
    )


def generate_inline_function(
    func,
    format: str = "jax",
    user_functions: dict[str, str] | None = None,
) -> str:
    """Generate a lambda expression for a function.

    Parameters
    ----------
    func : Function
        A TVBO Function object
    format : str
        Output format: 'jax', 'numpy'
    user_functions : dict, optional
        Custom function name mappings

    Returns:
    -------
    str
        Lambda expression string

    Examples:
    --------
    >>> func = Function(name='square', equation=Equation(rhs='x**2'), arguments=[{'name': 'x'}])
    >>> print(generate_inline_function(func))
    lambda x: x**2
    """
    # Lazy import to avoid circular dependency
    from tvbo.templates import lookup

    template = lookup.get_template("base/function-def.mako")
    return (
        template.get_def("inline_function")
        .render(
            func=func,
            format=format,
            user_functions=user_functions,
        )
        .strip()
    )


def function_to_callable(
    func,
    format: str = "jax",
    user_functions: dict[str, str] | None = None,
    namespace: dict | None = None,
) -> Callable:
    """Generate and execute function code, returning the callable.

    Parameters
    ----------
    func : Function
        A TVBO Function object
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
    >>> func = Function(name='sigmoid', equation=Equation(rhs='1/(1+exp(-x))'), ...)
    >>> sigmoid = function_to_callable(func)
    >>> sigmoid(0.0)
    0.5
    """
    code = generate_function(func, format=format, user_functions=user_functions)

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
    return namespace[func.name]


# Convenience alias
generate = generate_function
