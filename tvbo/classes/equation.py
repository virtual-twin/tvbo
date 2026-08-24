# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""Parse, substitute and sort the symbolic equations of a model.

Wraps sympy with the conventions TVBO's models rely on: a `local_dict` that stops names like `e` and `I` being read as constants, coupling-term substitution against the ontology, and a topological sort of derived quantities.
"""

import logging
import re
from collections import deque

import sympy as sp
from sympy import (
    IndexedBase,
    Symbol,
    latex,
    parse_expr,
    sympify,
)
from sympy.core.basic import Basic
from sympy.core.symbol import symbols
from sympy.printing.printer import Printer

from tvbo.ontology import owl as ontology
from tvbo.parse.symbols import BUILTIN_SHADOW

# Term order only — NOT `init_printing`, which also installs global IPython display formatters. Under a Jupyter kernel sympy resolves those to `use_latex='png'` and claims builtin `int`, so a report's inline `{python} n_modes` renders as a base64 PNG instead of a number. This is the setting `init_printing` itself applies, without that side effect.
Printer.set_global_settings(order="none")

ONTOLOGY_SCOPE = BUILTIN_SHADOW.extend(E=IndexedBase("E"), F=IndexedBase("F"))
"""Namespace for equations read out of the OWL ontology.

`E` and `F` are indexed there — an ontology equation writes `E[i]` for a population's
input — so they are bound to `IndexedBase` rather than left to
[`BUILTIN_SHADOW`](../parse/symbols.qmd#BUILTIN_SHADOW), which would make them symbols.
Everything else an ontology equation names is declared per equation by
[`sympify_value`](#tvbo.classes.equation.sympify_value) from that equation's own
parameters, functions and state variables.
"""

coupling_variables = ["lrc", "short_range_coupling", "coupling", "lc_0", "c_0", "lc_1"]
lambda_symbol = sp.symbols("lambda")
E = sp.symbols("E")  # TODO: not used
logger = logging.getLogger(__name__)


def add_spaces_around_operators(expression):
    """Insert surrounding spaces around binary arithmetic operators in a string.

    Wraps each `+`, `-`, `*`, `/` or `%` operator in single spaces so the expression parses cleanly, while leaving the `**` power operator untouched.

    Args:
        expression: The equation string to normalise.

    Returns:
        The expression with spaces added around single-character operators.
    """
    # The lookarounds keep `**` intact while spacing every single-character operator.
    pattern = r"(?<!\*)[\+\-\*/%](?!\*)"
    return re.sub(pattern, r" \g<0> ", expression)


def unify_coupling_terms(eq_string):
    """Rewrite TVB-style coupling terms to the legacy `c_pop*` naming.

    Replaces indexed `coupling[i]` references and `local_range_coupling` with the legacy `c_pop0` / `c_pop1` / `local_coupling` names used elsewhere in the equation pipeline.

    Args:
        eq_string: The equation string to normalise.

    Returns:
        The equation string with coupling terms renamed.
    """
    # TODO: normalises TVB-imported equations (coupling[i]) to the legacy c_pop0/c_pop1 names. The model database now uses c_glob/c_glob0/…, so this should eventually emit c_glob* (the single-vs-indexed choice needs the model's coupling arity, not available at this string-rewrite stage).
    repl_dict = {
        "coupling[0]": "c_pop0",
        "coupling[0, :]": "c_pop0",
        "coupling[0,:]": "c_pop0",
        "coupling[1, :]": "c_pop1",
        "coupling[1,:]": "c_pop1",
        "coupling[1]": "c_pop1",
        "local_range_coupling": "local_coupling",
        # "lrc": "local_coupling",
    }
    for k, v in repl_dict.items():
        eq_string = eq_string.replace(k, v)
    return eq_string


def sympify_value(v, acronym="", evaluate=False):
    """Parse a metadata equation's value into a SymPy expression.

    Collects the equation's referenced functions, parameters and state variables as symbols (stripping `acronym` from their labels), normalises NumPy prefixes and coupling terms, adds operator spacing and parses the result.

    Args:
        v: A metadata equation individual exposing `has_function`,
            `has_parameter`, `has_state_variable`, `value` and `label`.
        acronym: Model acronym to strip from the collected symbol names.
        evaluate: Accepted for API symmetry; the expression is always parsed
            unevaluated.

    Returns:
        The parsed SymPy expression.

    Raises:
        ValueError: If the equation string cannot be parsed, or states a branch as a
            `where(...)` call. A conditional belongs in the equation's `conditionals`,
            the one spelling TVBO builds a `Piecewise` from; a hand-written `where` would
            reach the printers as an opaque function application and silently lose its
            branch semantics.
    """
    eq_parameters = v.has_function + v.has_parameter + v.has_state_variable

    # Create a dictionary of symbols
    symbols_dict = {
        (str(s.label.first().replace(acronym, "")) if s.label.first() else s.name.replace(acronym, "")): Symbol(
            str(s.label.first().replace(acronym, "") if s.label.first() else s.name.replace(acronym, "")),
        )
        for s in eq_parameters
    }
    v = v.value.first()
    v = v.replace("numpy.", "").replace("np.", "") if v else ""
    v = unify_coupling_terms(v)
    eq = add_spaces_around_operators(v)
    scope = ONTOLOGY_SCOPE.extend(symbols_dict)
    # An indexed x_j carries a time dimension the expression does not model
    if "x_j" in eq and "[:" in eq:
        eq = eq.replace("[:,", "[")

    if "where(" in v:
        raise ValueError(
            f"Ontology equation {v!r} states a branch as a `where(...)` call. Write it as "
            "conditionals instead — that is the one spelling TVBO lowers to a Piecewise."
        )

    try:
        eq = scope.parse(eq, evaluate=False)
    except Exception as e:
        logger.debug("Error parsing equation %r: %s", eq, e)
        raise ValueError(f"Failed to parse equation: {eq}. Ensure the equation is in a valid format.") from e

    return eq


def replace_H(eq_dict):
    """Rename the `H` symbol to `h_uc` across a dictionary of equations.

    Avoids clashes with SymPy's built-in `H` by substituting the uppercase `H` symbol (and any `"H"` dictionary key) with `h_uc`.

    Args:
        eq_dict: Mapping of equation names to SymPy expressions.

    Returns:
        A new mapping with `H` replaced by `h_uc` in both keys and expressions.
    """
    hack_functions = {}
    H, h_uc = symbols("H h_uc")
    for k, v in eq_dict.items():
        if k == "H":
            k = "h_uc"
        hack_functions[k] = v.replace(H, h_uc)

    return hack_functions


def rename_uppercase_variables(input_equation):
    """Rename free symbols that start with an uppercase letter to a `*_uc` form.

    Each symbol whose name begins with an uppercase letter is replaced by its lowercased name suffixed with `_uc`; other symbols are left unchanged. A string input is first parsed via `sympify_value`.

    Args:
        input_equation: A SymPy expression, or a string to be parsed.

    Returns:
        The expression with uppercase-leading symbols renamed.

    Raises:
        ValueError: If a string input cannot be converted to a SymPy expression.
    """
    if not isinstance(input_equation, Basic):
        try:
            sympy_equation = sympify_value(input_equation)
        except Exception as e:
            raise ValueError(f"Invalid input for SymPy conversion: {e}") from e
    else:
        sympy_equation = input_equation

    # Parse the equation string into a SymPy expression

    # Extract all symbols in the equation
    all_symbols = sympy_equation.free_symbols

    # Create a mapping for new variable names
    new_names = {}

    for sym in all_symbols:
        name = str(sym)
        if name[0].isupper():
            # Convert to lowercase and append _uc
            new_names[sym] = symbols(name.lower() + "_uc")
        else:
            new_names[sym] = sym  # Keep as is

    # Substitute the variables in the equation with their new names
    new_equation = sympy_equation.subs(new_names)

    return new_equation


def set_specific_symbols_to_zero(
    equation_str,
    symbols_to_zero=coupling_variables,
):
    """Substitute the given symbols with zero in an equation string.

    Parses `equation_str` and replaces every symbol named in `symbols_to_zero` with `0`, for example to drop coupling contributions for isolated-node dynamics.

    Args:
        equation_str: The equation to parse and modify.
        symbols_to_zero: Names of the symbols to set to zero; defaults to the
            module-level coupling variable names.

    Returns:
        The SymPy expression with the listed symbols replaced by zero.
    """
    # Convert the equation string to a SymPy expression
    equation = parse_expr(equation_str, evaluate=False)

    # Create a dictionary for substitutions with each target symbol set to 0
    substitutions = {symbols(symbol): 0 for symbol in symbols_to_zero}

    # Perform the substitution
    modified_equation = equation.subs(substitutions)

    return modified_equation


# ----


def dependency_tree(equations):
    """Build a directed dependency graph from a list of equations.

    For each equation, the right-hand-side free symbols are treated as dependencies of the left-hand side, producing a `networkx.DiGraph` with an edge from each source symbol to its target.

    Args:
        equations: An iterable of SymPy equations exposing `lhs` and `rhs`.

    Returns:
        A `networkx.DiGraph` whose edges point from dependencies to dependents.
    """
    import networkx as nx

    dependencies = {}
    for eq in equations:
        free_symbols = eq.rhs.free_symbols
        dependencies[eq.lhs] = free_symbols

    G = nx.DiGraph()
    node_symbols = {}
    for target, sources in dependencies.items():
        if str(target) not in node_symbols:
            node_symbols[str(target)] = target

        for source in sources:
            if source == target:
                continue
            if str(source) not in node_symbols:
                node_symbols[str(source)] = source

            G.add_edge(node_symbols[str(source)], node_symbols[str(target)])

    return G


def build_dependency_graph(eq_dict):
    # DEPRECATED #TODO: remove
    """Builds a directed graph of dependencies from the eq_dict using SymPy."""
    graph = {}
    symbol_dict = {var: symbols(var) for var in eq_dict.keys()}  # Create a dictionary of SymPy symbols

    for var, expr in eq_dict.items():
        graph[var] = set()
        parsed_expr = sympify(expr, locals=symbol_dict, evaluate=False)
        for atom in parsed_expr.atoms():
            if atom in symbol_dict.values() and atom != symbol_dict[var]:
                graph[var].add(str(atom))

    return graph


def topological_sort(graph):
    """Performs topological sorting on the dependency graph."""
    in_degree = {u: 0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    queue = deque([u for u in graph if in_degree[u] == 0])
    sorted_order = []

    while queue:
        u = queue.popleft()
        sorted_order.append(u)
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(sorted_order) != len(graph):
        raise ValueError("Circular dependency detected or unresolved dependencies remain")

    return sorted_order


def sort_equations_by_dependencies(equations):
    """Return equations ordered so each is defined before it is used.

    Builds a dependency graph over the equation names, topologically sorts it and returns a new dictionary in dependency-respecting order.

    Args:
        equations: Mapping of variable names to their expression strings.

    Returns:
        A new dictionary with the same items ordered by dependency.
    """
    graph = build_dependency_graph(equations)
    sorted_vars = topological_sort(graph)
    sorted_vars.reverse()
    return {var: equations[var] for var in sorted_vars}


# ----


def replace_acronyms(key, cls):
    """Strip model-acronym suffixes from a key.

    Removes the `_<acronym>` suffix contributed by each neural-mass-model ancestor of `cls`, yielding the bare variable name.

    Args:
        key: The name to strip acronym suffixes from.
        cls: The ontology class whose model ancestors provide the acronyms.

    Returns:
        The key with matching `_<acronym>` suffixes removed.
    """
    for c in ontology.intersection(
        list(ontology.onto.NeuralMassModel.descendants()),
        list(cls.ancestors()),
    ):
        a = c.acronym.first() if hasattr(c, "acronym") else ""
        if a:
            key = key.replace(f"_{a}", "")
    return key


def symbolic_model_functions(NMM, zero_coupling=False, **kwargs):
    """Return the model's auxiliary functions as SymPy expressions.

    Sympifies each non-derivative model function (skipping `numpy.exp`), optionally zeroing coupling terms, strips the acronym and model-suffix decorations from the names, and orders the result by inter-equation dependency.

    Args:
        NMM: The neural-mass model identifier or ontology individual.
        zero_coupling: If true, set coupling symbols to zero in each function.
        **kwargs: Forwarded to `sympify_value` (e.g. `acronym`).

    Returns:
        A dependency-ordered mapping of function names to SymPy expressions.
    """
    func_dict = dict()
    suffix = ontology.get_model_suffix(NMM)
    for k, v in ontology.get_model_functions(NMM).items():
        if v.value == ["numpy.exp"]:
            continue
        symeq = sympify_value(v, **kwargs)

        if not symeq:
            continue
        if zero_coupling:
            symeq = set_specific_symbols_to_zero(symeq)

        k = replace_acronyms(k, v)
        func_dict[k.replace(suffix, "")] = symeq

    func_dict = sort_equations_by_dependencies(func_dict)
    return func_dict


def symbolic_differential_equations(NMM, zero_coupling=False, **kwargs):
    """Return the model's time-derivative equations as SymPy expressions.

    Selects the model derivatives whose name contains `dot`, sympifies each right-hand side, optionally zeroing coupling terms, and strips the model suffix from the keys.

    Args:
        NMM: The neural-mass model identifier or ontology individual.
        zero_coupling: If true, set coupling symbols to zero in each equation.
        **kwargs: Forwarded to `sympify_value` (e.g. `acronym`).

    Returns:
        A mapping of derivative names to SymPy expressions.
    """
    td_dict = dict()
    suffix = ontology.get_model_suffix(NMM)
    for k, v in ontology.get_model_derivatives(NMM).items():
        if "dot" not in k:
            continue
        symeq = sympify_value(v, **kwargs)
        if zero_coupling:
            symeq = set_specific_symbols_to_zero(symeq)
        td_dict[k.replace(suffix, "")] = symeq

    return td_dict


def symbolic_conditions(NMM, zero_coupling=False, **kwargs):
    """Return the model's conditional expressions as SymPy expressions.

    Sympifies each model conditional, optionally zeroing coupling terms, and strips the acronym and model-suffix decorations from the names.

    Args:
        NMM: The neural-mass model identifier or ontology individual.
        zero_coupling: If true, set coupling symbols to zero in each expression.
        **kwargs: Forwarded to `sympify_value` (e.g. `acronym`).

    Returns:
        A mapping of conditional names to SymPy expressions.
    """
    cond_dict = dict()
    suffix = ontology.get_model_suffix(NMM)

    for k, v in ontology.get_model_conditionals(NMM).items():
        symeq = sympify_value(v, **kwargs)
        if zero_coupling:
            symeq = set_specific_symbols_to_zero(symeq)
        k = replace_acronyms(k, v)
        cond_dict[k.replace(suffix, "")] = symeq

    return cond_dict


def symbolic_topological_sort(equations):
    """Order equation names so dependencies precede the equations that use them.

    Builds a dependency graph from each expression's free symbols that also appear as equation keys, then performs a Kahn topological sort.

    Args:
        equations: Mapping of variable names to SymPy expressions.

    Returns:
        A list of equation names in dependency-respecting order.

    Raises:
        ValueError: If the equations contain a circular dependency.
    """
    graph = {k: set() for k in equations}
    for eq, expr in equations.items():
        for symbol in expr.free_symbols:
            symbol_str = str(symbol)
            if symbol_str in equations and symbol_str != eq:
                graph[symbol_str].add(eq)
    # TODO: why not use here the topological_sort method defined above
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for dependent in graph[node]:
            in_degree[dependent] += 1

    queue = deque([node for node in graph if in_degree[node] == 0])
    sorted_order = []

    while queue:
        node = queue.popleft()
        sorted_order.append(node)
        for dependent in graph[node]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(sorted_order) != len(graph):
        raise ValueError("A circular dependency exists in the equations")

    return sorted_order


def symbolic_model_equations(NMM, zero_coupling=False, **kwargs):
    """Return all symbolic equations for a model in one mapping.

    Merges the model's auxiliary functions, time-derivative equations and conditionals into a single dictionary.

    Args:
        NMM: The neural-mass model identifier or ontology individual.
        zero_coupling: If true, set coupling symbols to zero throughout.
        **kwargs: Forwarded to `sympify_value` (e.g. `acronym`).

    Returns:
        A combined mapping of names to SymPy expressions.
    """
    func_dict = symbolic_model_functions(NMM, zero_coupling=zero_coupling, **kwargs)
    td_dict = symbolic_differential_equations(NMM, zero_coupling=zero_coupling, **kwargs)
    cond_dict = symbolic_conditions(NMM, zero_coupling=zero_coupling, **kwargs)
    return {**func_dict, **td_dict, **cond_dict}


def sub_equation(eq, model):
    """Substitute an equation's symbols with their ontology display symbols.

    For each free symbol, looks up the corresponding model variable in the ontology (keeping coupling terms by their bare name) and replaces it with the variable's declared symbol, also applying the canonical coupling and conditional renamings.

    Args:
        eq: The SymPy expression to rewrite.
        model: The model identifier used to resolve variable symbols.

    Returns:
        The expression with symbols substituted for their display symbols.

    Raises:
        ValueError: If a symbol cannot be resolved to a model variable.
    """
    acr = ontology.get_model_acronym(model)
    # Derived from the model so any coupling naming works; the literals are a defensive fallback.
    coupling_keep = set(ontology.get_model_coupling_terms(model).keys()) | {
        "local_coupling",
        "c_pop0",
        "c_pop1",
        "c_pop2",
    }
    sub = {
        "local_coupling": "c_loc",
        "c_pop0": "c_glob0",
        "c_pop1": "c_glob1",
        "lambda": lambda_symbol,
        "short_range": "c_short",
    }
    for s in eq.free_symbols:
        name = s.name + "_" + acr if s.name not in coupling_keep else s.name
        c_rhs = ontology.onto[name]
        if isinstance(c_rhs, type(None)):
            search = ontology.intersection(
                ontology.onto[model].descendants(),
                ontology.onto.search(label=f"{s.name}*"),
            )
            if len(search) == 1:
                c_rhs = search[0]
            else:
                raise ValueError(f"Could not find {eq} in {model}")
        # TODO: duplicated below, could a small method be created instead?
        symb = c_rhs.symbol.first()
        if s.name.endswith("cond"):
            sub.update({s.name: sp.symbols(s.name.replace("cond", "_{cond}"))})

        if symb != s.name and not isinstance(symb, type(None)) and not symb == "":
            sub.update({s.name: sp.symbols(symb.replace(" ", "_"))})
    return eq.subs(sub)


def substitute_function_in_state_equations(sv_eqs, funcs):
    """Inline auxiliary function definitions into state-variable equations.

    For each state-variable equation, replaces any function symbol that appears in it with the function's defining expression.

    Args:
        sv_eqs: Mapping of state-variable names to SymPy expressions; mutated in
            place.
        funcs: Mapping of function names to their defining SymPy expressions.

    Returns:
        The updated `sv_eqs` mapping with functions inlined.
    """
    for sv, sv_eq in sv_eqs.items():
        subs = dict()
        for f, f_eq in funcs.items():
            f = sp.symbols(f)
            if f in sv_eq.free_symbols:
                subs[f] = f_eq

        sv_eqs[sv] = sv_eq.subs(subs)

    return sv_eqs


def update_mathematical_relationships(model):
    """Refresh the ontology relationships implied by a model's equations.

    Walks every symbolic equation of the model and, for each free symbol, records the parameter / state-variable / derivative relationship between the symbol's class and the equation's class in the ontology. Equations that are `None` or not valid SymPy expressions are skipped with a message.

    Args:
        model: The model identifier whose relationships are updated.
    """
    for k, v in symbolic_model_equations(model).items():
        k_cls = ontology.find_variables(k, model)
        if k_cls is None:
            logger.warning('did not find "%s" in "%s"', k, model)
            continue

        # Handle cases where v might be None (e.g., for discrete maps with conditionals)
        if v is None:
            logger.debug('Skipping "%s" in "%s" - equation is None', k, model)
            continue

        # Check if v has free_symbols attribute (valid SymPy expression)
        if not hasattr(v, "free_symbols"):
            logger.debug('Skipping "%s" in "%s" - not a valid SymPy expression', k, model)
            continue

        for symbol in v.free_symbols:
            s_cls = ontology.find_variables(symbol.name, model)
            if s_cls is None:
                continue
            update_class_relationships(s_cls, k_cls)


def update_class_relationships(s_cls, k_cls):
    """Append the ontology `is_a` relations linking a variable to its equation.

    Within the ontology world, adds `is_parameter_in` / `is_state_variable_of` / `has_derivative` / `is_derivative_of` axioms between the source variable class and the equation class where they do not already exist, then de-duplicates each class's `is_a` list.

    Args:
        s_cls: The ontology class of the source variable (parameter, function,
            state variable, and similar).
        k_cls: The ontology class of the equation the variable appears in.
    """
    with ontology.onto:
        # Append relationships if they don't already exist
        if (
            ontology.onto.Parameter in s_cls.is_a
            or ontology.onto.Function in s_cls.is_a
            or ontology.onto.ConditionalDerivedVariable in s_cls.is_a
        ) and ontology.onto.is_parameter_in.some(k_cls) not in s_cls.is_a:
            s_cls.is_a.append(ontology.onto.is_parameter_in.some(k_cls))

        if ontology.onto.StateVariable in s_cls.is_a and ontology.onto.is_state_variable_of.some(k_cls) not in s_cls.is_a:
            s_cls.is_a.append(ontology.onto.is_state_variable_of.some(k_cls))

        if (
            ontology.onto.TimeDerivative in k_cls.is_a
            # and ontology.onto.has_derivative.some(k_cls) not in s_cls.is_a
        ):
            if s_cls in k_cls.is_a:
                s_cls.is_a.append(ontology.onto.has_derivative.some(k_cls))
                k_cls.is_a.append(ontology.onto.is_derivative_of.some(s_cls))

        # Ensure unique relationships
        s_cls.is_a = list(set(s_cls.is_a))
        k_cls.is_a = list(set(k_cls.is_a))


# Coupling functions #


def get_symbolic_coupling(coupling_function) -> dict:
    """Get the symbolic coupling expressions for the given coupling function.

    Parameters:
        coupling_function (str or CouplingFunction): The coupling function to retrieve symbolic expressions for.

    Returns:
        dict: A dictionary containing the symbolic expressions for the pre and post functions.
              The keys are 'pre' and 'post', and the values are SymPy expressions.

    Raises:
        SomeException: Description of the exception raised, if any.
    """
    # Get the coupling function from the ontology
    if isinstance(coupling_function, str):
        coupling_function = ontology.get_coupling_function(coupling_function)

    # Get the pre and post functions from the ontology
    fpost = ontology.intersection(coupling_function.subclasses(), ontology.onto.Fpost.descendants())
    fpre = ontology.intersection(coupling_function.subclasses(), ontology.onto.Fpre.descendants())
    # Create symbolic expressions for the pre and post functions
    fpre = sympify_value(fpre[0]) if len(fpre) > 0 else sympify("x_j")
    fpost = sympify_value(fpost[0]) if len(fpost) > 0 else sympify("gx")
    return {"pre": fpre, "post": fpost}


def generate_global_coupling_function(pre_expr, post_expr, j_index_start=0):
    """Generate the global coupling function based on given pre and post expressions.

    :param pre_expr: The 'pre' sympy expression involving x_i and x_j.
    :param post_expr: The 'post' sympy expression involving gx.
    :return: The global coupling function as a sympy expression.
    """
    # Define necessary symbols
    g = IndexedBase("g")
    x = IndexedBase("x")
    i, j, N, gx = symbols("i j N gx")

    # x_i, x_j, N, gx, g_ij = sp.symbols("x_i x_j N gx g_ij")

    gx_sum = sp.Sum(
        g[i, j] * pre_expr.subs({"x_j": x[j], "x_i": x[i]}),
        (j, j_index_start, N - (1 - j_index_start)),
    )

    post_with_gx = post_expr.subs({gx: gx_sum})

    # Return the expression without additional simplification
    return post_with_gx


def topological_sort_equations(variable_dict, dependency_tree):
    """Sort equations topologically according to a dependency graph.

    Verifies the dependency graph is acyclic (raising with the offending cycle when it is not) and returns the equations reordered so each variable follows the variables it depends on.

    Args:
        variable_dict: Mapping of variable names to their equations.
        dependency_tree: A `networkx.DiGraph` of variable dependencies; must be a
            directed acyclic graph.

    Returns:
        A new dictionary of equations in topological order.

    Raises:
        ValueError: If the dependency graph contains a cycle.
    """
    from networkx import (
        draw,
        find_cycle,
        is_directed_acyclic_graph,
        kamada_kawai_layout,
        topological_sort,
    )

    """
    Sorts equations topologically by dependency, breaking ties alphabetically.

    Args:
        variable_dict (dict): A dictionary of variables and their equations.
        dependency_tree (networkx.DiGraph): A directed acyclic graph representing variable dependencies.

    Returns:
        dict: A sorted dictionary of equations.
    """
    # Verify the dependency tree is a DAG
    if not is_directed_acyclic_graph(dependency_tree):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(16, 9))
        draw(
            dependency_tree,
            pos=kamada_kawai_layout(dependency_tree),
            with_labels=True,
            labels={n: f"${latex(n)}$" for n in dependency_tree.nodes},
            ax=ax,
            node_color="lightblue",
        )
        cycles = find_cycle(dependency_tree, orientation="original")
        if cycles:
            cycle_str = " -> ".join(f"{edge[0]} -> {edge[1]}" for edge in cycles)
            raise ValueError(f"Found cycles: {cycle_str}. Dependency tree must be a Directed Acyclic Graph (DAG).")
        else:
            raise ValueError("Dependency tree must be a Directed Acyclic Graph (DAG). Unable to identify cycles.")

    # Perform topological sort and sort ties alphabetically
    sorted_variables = list(topological_sort(dependency_tree))

    sorted_equations = {str(var): variable_dict[str(var)] for var in sorted_variables if str(var) in variable_dict}

    return sorted_equations
