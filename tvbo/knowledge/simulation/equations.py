#  py
#
# Created on Fri Jan 05 2024
# Author: Leon K. Martin
#
# Copyright (c) 2024 Charité Universitätsmedizin Berlin
#

"""
# Handling Equations and Expressions
"""
import re
from collections import deque

import sympy as sp
from sympy import (
    IndexedBase,
    Piecewise,
    init_printing,
    latex,
    sympify,
    parse_expr,
    Symbol,
    pycode,
)
from sympy.abc import _clash1
from sympy.core.basic import Basic
from sympy.core.symbol import symbols

from tvbo.knowledge import ontology

init_printing(order="none")

_clash1.update(
    {
        "gamma": "",
        "beta": "",
        "lambda": "",
        "omega": "",
        "E": Symbol("E"),
        "local_coupling": Symbol("local_coupling"),
        "I": Symbol("I"),
        "Q": Symbol("Q"),
        "x_j": IndexedBase("x_j"),
        "var": Symbol("var"),
        "onset": Symbol("onset"),
        "T": Symbol("T"),
        "tau": Symbol("tau"),
        "amp": Symbol("amp"),
        "Piecewise": Piecewise,  # Add Piecewise for discrete maps
    }
)

coupling_variables = ["lrc", "short_range_coupling", "coupling", "lc_0", "c_0", "lc_1"]
lambda_symbol = sp.symbols("lambda")
E = sp.symbols("E")  # TODO: not used


def add_spaces_around_operators(expression):
    # Pattern explanation:
    # (?<!\*) : Negative lookbehind to ensure there's no * before the current character
    # [\+\-\*/%] : Matches any of the operators +, -, *, /, %
    # (?!\\*) : Negative lookahead to ensure there's no * after the current character
    pattern = r"(?<!\*)[\+\-\*/%](?!\*)"
    return re.sub(pattern, r" \g<0> ", expression)


def unify_coupling_terms(eq_string):
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


def extract_parts_from_numpy_where(python_string):  # TODO: test!
    """
    Extracts the condition, if_true, and if_false parts from a numpy.where or where expression string.
    Args:
    - python_string: A string in the format "numpy.where(condition, if_true, if_false)" or "where(condition, if_true, if_false)"
    Returns:
    - A tuple containing the condition, if_true, and if_false parts as strings.
    """
    # Check and remove the starting part of the string
    if python_string.startswith("numpy.where("):
        parts = python_string[12:-1]
    elif python_string.startswith("where("):
        parts = python_string[6:-1]
    else:
        raise ValueError("The input string does not match the expected format.")

    # Parsing the string to split correctly at top-level commas
    condition, if_true, if_false = "", "", ""
    parentheses_count = 0
    part = 1

    for char in parts:
        if char == "(":
            parentheses_count += 1
        elif char == ")":
            parentheses_count -= 1
        elif char == "," and parentheses_count == 0:
            part += 1
            continue

        if part == 1:
            condition += char
        elif part == 2:
            if_true += char
        elif part == 3:
            if_false += char

    if part != 3:
        raise ValueError("The input string does not have three parts.")

    return condition.strip(), if_true.strip(), if_false.strip()


def convert_ifelse_to_np_where(code_str):
    """
    Convert a sympy.pycode output string from a simple if-else format to a numpy where format.

    Args:
    code_str (str): A string representing a sympy.pycode output,
                    e.g., "((-0.1*z**7) if (z < 0) else (0))".

    Returns:
    str: Converted string using np.where, e.g., "np.where(z < 0, -0.1*z**7, 0)".
    """
    if not "if" in code_str:
        return code_str
    # Regular expression to capture the if-else structure
    pattern = r"\((.*) if (.*) else (.*)\)"
    match = re.match(pattern, code_str)

    if not match:
        raise ValueError("Input string does not match expected if-else format")

    # Extracting the parts of the if-else statement
    true_expr, condition, false_expr = match.groups()

    # Constructing the np.where format
    return f"where({condition}, {true_expr}, {false_expr})"


def convert_numpy_where_to_sympy(python_string):
    """
    Converts a numpy.where expression to a sympy Piecewise expression.
    Args:
    - condition_str: The condition string.
    - if_true_str: The expression if the condition is True.
    - if_false_str: The expression if the condition is False.
    Returns:
    - A sympy Piecewise expression.
    """
    python_string = python_string.replace("numpy.", "").replace("np.", "")
    condition_str, if_true_str, if_false_str = extract_parts_from_numpy_where(
        python_string
    )
    # Parse the strings into sympy expressions
    condition = parse_expr(condition_str, _clash1, evaluate=False)
    if_true = parse_expr(if_true_str, _clash1, evaluate=False)
    if_false = parse_expr(if_false_str, _clash1, evaluate=False)

    # Construct the Piecewise expression
    return Piecewise((if_true, condition), (if_false, True))


def sympify_value(v, acronym="", evaluate=False):
    eq_parameters = v.has_function + v.has_parameter + v.has_state_variable
    # if len(eq_parameters) == 0:
    # print('No parameters found for "{}"'.format(v.label.first()))
    # return None

    # Create a dictionary of symbols
    symbols_dict = {
        (
            str(s.label.first().replace(acronym, ""))
            if s.label.first()
            else s.name.replace(acronym, "")
        ): Symbol(
            str(
                s.label.first().replace(acronym, "")
                if s.label.first()
                else s.name.replace(acronym, "")
            ),
        )
        for s in eq_parameters
    }
    v = v.value.first()
    v = v.replace("numpy.", "").replace("np.", "") if v else ""
    v = unify_coupling_terms(v)
    eq = add_spaces_around_operators(v)
    # If x_j is indexed, remove time-dimension from the expression
    ## TODO: fix this in the ontology or find a better solution
    if "x_j" in eq and "[:" in eq:
        _clash1.update(
            {
                "x_j": IndexedBase("x_j"),
            }
        )
        eq = eq.replace("[:,", "[")
    else:
        _clash1.pop("x_j", None)

    _clash1.update(symbols_dict)
    if "where(" in v:
        return convert_numpy_where_to_sympy(v)

    _clash1.update({"E": IndexedBase("E"), "F": IndexedBase("F")}) # TODO: Remove this hack

    try:
        eq = parse_expr(
            eq,
            _clash1,
            evaluate=False,
        )
    except Exception as e:
        print(f"Error parsing equation: {eq}")
        print(f"Error message: {e}")
        raise ValueError(
            f"Failed to parse equation: {eq}. Ensure the equation is in a valid format."
        )

    return eq


def replace_H(eq_dict):
    hack_functions = {}
    H, h_uc = symbols("H h_uc")
    for k, v in eq_dict.items():
        if k == "H":
            k = "h_uc"
        hack_functions[k] = v.replace(H, h_uc)

    return hack_functions


def rename_uppercase_variables(input_equation):
    if not isinstance(input_equation, Basic):
        try:
            sympy_equation = sympify_value(input_equation)
        except Exception as e:
            raise ValueError(f"Invalid input for SymPy conversion: {e}")
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
    # Convert the equation string to a SymPy expression
    equation = parse_expr(equation_str, evaluate=False)

    # Create a dictionary for substitutions with each target symbol set to 0
    substitutions = {symbols(symbol): 0 for symbol in symbols_to_zero}

    # Perform the substitution
    modified_equation = equation.subs(substitutions)

    return modified_equation


# ----


def dependency_tree(equations):
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
    symbol_dict = {
        var: symbols(var) for var in eq_dict.keys()
    }  # Create a dictionary of SymPy symbols

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
        raise ValueError(
            "Circular dependency detected or unresolved dependencies remain"
        )

    return sorted_order


def sort_equations_by_dependencies(equations):
    graph = build_dependency_graph(equations)
    sorted_vars = topological_sort(graph)
    sorted_vars.reverse()
    return {var: equations[var] for var in sorted_vars}


# ----


def replace_acronyms(key, cls):
    for c in ontology.intersection(
        list(ontology.onto.NeuralMassModel.descendants()),
        list(cls.ancestors()),
    ):
        a = c.acronym.first() if hasattr(c, "acronym") else ""
        if a:
            # print(a)
            key = key.replace(f"_{a}", "")
    return key


def symbolic_model_functions(NMM, zero_coupling=False, **kwargs):
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
    td_dict = dict()
    suffix = ontology.get_model_suffix(NMM)
    for k, v in ontology.get_model_derivatives(NMM).items():
        if not "dot" in k:
            continue
        symeq = sympify_value(v, **kwargs)
        if zero_coupling:
            symeq = set_specific_symbols_to_zero(symeq)
        td_dict[k.replace(suffix, "")] = symeq

    return td_dict


def symbolic_conditions(NMM, zero_coupling=False, **kwargs):
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
    func_dict = symbolic_model_functions(NMM, zero_coupling=zero_coupling, **kwargs)
    td_dict = symbolic_differential_equations(
        NMM, zero_coupling=zero_coupling, **kwargs
    )
    cond_dict = symbolic_conditions(NMM, zero_coupling=zero_coupling, **kwargs)
    return {**func_dict, **td_dict, **cond_dict}


def sub_equation(eq, model):
    acr = ontology.get_model_acronym(model)
    sub = {
        "local_coupling": "c_loc",
        "c_pop0": "c_glob0",
        "c_pop1": "c_glob1",
        "lambda": lambda_symbol,
        "short_range": "c_short",
    }
    for s in eq.free_symbols:
        name = (
            s.name + "_" + acr
            if not s.name in ["local_coupling", "c_pop0", "c_pop1"]
            else s.name
        )
        # print(name)
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
    for sv, sv_eq in sv_eqs.items():
        subs = dict()
        for f, f_eq in funcs.items():
            f = sp.symbols(f)
            if f in sv_eq.free_symbols:
                subs[f] = f_eq

        sv_eqs[sv] = sv_eq.subs(subs)

    return sv_eqs


######################
# Latex equations   #
######################
def get_latex_equation(model, func_dict="all", mul_symbol="dot"):
    if func_dict == "all":
        func_dict = symbolic_model_equations(model)
    sorting = symbolic_topological_sort(func_dict)
    # print(sorting)

    acr = ontology.get_model_acronym(model)
    latex_equations = list()
    sub = {
        "local_coupling": "c_loc",
        "c_pop0": "c_glob0",
        "c_pop1": "c_glob1",
        "lambda": lambda_symbol,
        "short_range": "src",
    }

    for k in sorting:
        v = func_dict[k]
        # print(k, v)
        c_lhs = ontology.onto[k + "_" + acr]
        if isinstance(c_lhs, type(None)):
            search = ontology.intersection(
                ontology.onto[model].descendants(),
                ontology.onto.search(label=f"{k}*"),
            )
            if len(search) == 1:
                c_lhs = search[0]
            else:
                raise ValueError(f"Could not find {k} in {model}")
                # print(search)

        lhs = c_lhs.symbol.first()
        lhs = sp.symbols(lhs)
        if lhs.name.endswith("cond"):
            lhs = lhs.subs({lhs: sp.symbols(lhs.name.replace("cond", "_{cond}"))})
        lhs = lhs.subs(sub)

        for s in v.free_symbols:
            name = (
                s.name + "_" + acr
                if not s.name in ["local_coupling", "c_pop0", "c_pop1"]
                else s.name
            )
            # print(name)
            c_rhs = ontology.onto[name]
            if isinstance(c_rhs, type(None)):
                search = ontology.intersection(
                    ontology.onto[model].descendants(),
                    ontology.onto.search(label=f"{s.name}*"),
                )
                if len(search) == 1:
                    c_rhs = search[0]
                else:
                    print(search)
                    raise ValueError(f"Could not find {s} in {model}")

            symb = c_rhs.symbol.first()
            if s.name.endswith("cond"):
                sub.update({s.name: sp.symbols(s.name.replace("cond", "_{cond}"))})

            if symb != s.name and not isinstance(symb, type(None)) and not symb == "":
                sub.update({s.name: sp.symbols(symb.replace(" ", "_"))})
        # print(sub)
        rhs = v.subs(sub)
        rhs = rhs.subs(sub)
        latex_rhs = latex(rhs, mul_symbol=mul_symbol)
        latex_eq = f"{latex(lhs)} = {latex_rhs}"
        latex_equations.append(latex_eq)
    return latex_equations


def render_latex_equations(
    model,
    odes_first=True,
    evaluate=False,
    separator=r"\text{where }",
    markdown=False,
    subs=None,
):
    NMM = ontology.get_model(model, verbose=False)
    CF = ontology.get_coupling_function(model, verbose=False)
    # print(NMM, CF)
    if NMM and NMM in ontology.onto.NeuralMassModel.descendants():
        func_dict = {
            **symbolic_model_functions(model),
            **symbolic_conditions(model),
        }
        if not isinstance(subs, type(None)):
            func_dict = {k: v.subs(subs) for k, v in func_dict.items()}

        latex_eq = get_latex_equation(
            model,
            func_dict=func_dict,
        )
        if markdown:
            newline = "\n\n"
        else:
            newline = r"\\"
        where = newline.join(latex_eq)
        odes = newline.join(
            get_latex_equation(
                model,
                func_dict=symbolic_differential_equations(
                    model, **{"evaluate": evaluate}
                ),
            )
        )
        if where == "":
            return odes
        if odes_first:
            return f"{newline} {separator} {newline}".join([odes, where])

        expression = newline.join([where, odes])

    elif CF and CF in ontology.onto.Coupling.descendants():
        pre, post = get_symbolic_coupling(CF).values()
        expression = f"$c_{{{CF.label.first()}}} = {latex(generate_global_coupling_function(pre, post))}$"

    return expression


def update_mathematical_relationships(model):
    for k, v in symbolic_model_equations(model).items():
        k_cls = ontology.find_variables(k, model)
        if k_cls is None:
            print('did not find "{}" in "{}"'.format(k, model))
            continue

        # Handle cases where v might be None (e.g., for discrete maps with conditionals)
        if v is None:
            print(f'Skipping "{k}" in "{model}" - equation is None')
            continue

        # Check if v has free_symbols attribute (valid SymPy expression)
        if not hasattr(v, 'free_symbols'):
            print(f'Skipping "{k}" in "{model}" - not a valid SymPy expression')
            continue

        for symbol in v.free_symbols:
            s_cls = ontology.find_variables(symbol.name, model)
            if s_cls is None:
                continue
            update_class_relationships(s_cls, k_cls)


def update_class_relationships(s_cls, k_cls):
    with ontology.onto:
        # Append relationships if they don't already exist
        if (
            ontology.onto.Parameter in s_cls.is_a
            or ontology.onto.Function in s_cls.is_a
            or ontology.onto.ConditionalDerivedVariable in s_cls.is_a
        ) and ontology.onto.is_parameter_in.some(k_cls) not in s_cls.is_a:
            s_cls.is_a.append(ontology.onto.is_parameter_in.some(k_cls))

        if (
            ontology.onto.StateVariable in s_cls.is_a
            and ontology.onto.is_state_variable_of.some(k_cls) not in s_cls.is_a
        ):
            s_cls.is_a.append(ontology.onto.is_state_variable_of.some(k_cls))

        if (
            ontology.onto.TimeDerivative
            in k_cls.is_a
            # and ontology.onto.has_derivative.some(k_cls) not in s_cls.is_a
        ):
            if s_cls in k_cls.is_a:
                s_cls.is_a.append(ontology.onto.has_derivative.some(k_cls))
                k_cls.is_a.append(ontology.onto.is_derivative_of.some(s_cls))

        # Ensure unique relationships
        s_cls.is_a = list(set(s_cls.is_a))
        k_cls.is_a = list(set(k_cls.is_a))


######################
# Coupling functions #
######################


def get_symbolic_coupling(coupling_function) -> dict:
    """
    Get the symbolic coupling expressions for the given coupling function.

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
    fpost = ontology.intersection(
        coupling_function.subclasses(), ontology.onto.Fpost.descendants()
    )
    fpre = ontology.intersection(
        coupling_function.subclasses(), ontology.onto.Fpre.descendants()
    )
    # Create symbolic expressions for the pre and post functions
    fpre = sympify_value(fpre[0]) if len(fpre) > 0 else sympify("x_j")
    fpost = sympify_value(fpost[0]) if len(fpost) > 0 else sympify("gx")
    return {"pre": fpre, "post": fpost}


def generate_global_coupling_function(pre_expr, post_expr, j_index_start=0):
    """
    Generate the global coupling function based on given pre and post expressions.

    :param pre_expr: The 'pre' sympy expression involving x_i and x_j.
    :param post_expr: The 'post' sympy expression involving gx.
    :return: The global coupling function as a sympy expression.
    """
    # Define necessary symbols
    g = IndexedBase("g")
    x = IndexedBase("x")
    i, j, N, gx = symbols("i j N gx")

    # x_i, x_j, N, gx, g_ij = sp.symbols("x_i x_j N gx g_ij")

    # Calculate gx as the sum of the 'pre' function
    # gx_sum = sp.Sum(g[i, j] * pre_expr.subs({"x_j": x[j], "x_i": x[i]}), (j, 1, N))
    gx_sum = sp.Sum(
        g[i, j] * pre_expr.subs({"x_j": x[j], "x_i": x[i]}),
        (j, j_index_start, N - (1 - j_index_start)),
    )

    # Substitute gx in the 'post' expression
    # Keeping the summation in its original form
    post_with_gx = post_expr.subs({gx: gx_sum})

    # Return the expression without additional simplification
    return post_with_gx


def topological_sort_equations(variable_dict, dependency_tree):
    from networkx import (
        topological_sort,
        is_directed_acyclic_graph,
        draw,
        find_cycle,
        kamada_kawai_layout,
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
            raise ValueError(
                f"Found cycles: {cycle_str}. Dependency tree must be a Directed Acyclic Graph (DAG)."
            )
        else:
            raise ValueError(
                "Dependency tree must be a Directed Acyclic Graph (DAG). Unable to identify cycles."
            )

    # Perform topological sort and sort ties alphabetically
    sorted_variables = list(topological_sort(dependency_tree))

    # sorted(
    #     sorted_variables,
    #     key=lambda x: (dependency_tree.in_degree(x), str(x)),
    # )

    # Create a sorted dictionary
    sorted_equations = {
        str(var): variable_dict[str(var)]
        for var in sorted_variables
        if str(var) in variable_dict
    }

    return sorted_equations


#################
# Piecewise     #
#################
def conditionals2piecewise(metadata_equation):

    return Piecewise(
        *[
            (
                parse_expr(cond.expression, _clash1, evaluate=False),
                parse_expr(cond.condition, _clash1, evaluate=False),
            )
            for cond in metadata_equation.conditionals
        ]
        + [
            (
                (
                    parse_expr(metadata_equation.rhs, _clash1, evaluate=False)
                    if metadata_equation.rhs
                    else 0
                ),
                True,
            )
        ]
    )


def piecewise2numpy(piecewise_expr, fully_qualified_modules=False) -> str:
    """
    Convert a sympy Piecewise expression to an equivalent nested numpy.where expression.

    Args:
        piecewise_expr (Piecewise): A sympy Piecewise expression.

    Returns:
        str: A string representing the equivalent numpy.where statement.
    """
    if not isinstance(piecewise_expr, Piecewise):
        raise ValueError("Input must be a sympy Piecewise expression")

    where_expr = None
    for expr, cond in reversed(
        piecewise_expr.args
    ):  # Start from the last argument to build the nested structure
        expr_code = pycode(expr, fully_qualified_modules=False)
        cond_code = (
            pycode(cond, fully_qualified_modules=False) if cond != True else "True"
        )
        if where_expr is None:
            where_expr = expr_code
        else:
            where_expr = f"{'np.' if fully_qualified_modules else ''}where({cond_code}, {expr_code}, {where_expr})"

    return str(where_expr)


#################
# Julia Adapter #
#################


def piecewise2julia(piecewise_expr) -> str:
    """
    Convert a sympy Piecewise expression to a Julia ifelse expression string.

    Parameters:
    piecewise_expr (sympy.Piecewise): A sympy Piecewise object.

    Returns:
    str: A Julia-compatible string representing the Piecewise expression using nested ifelse.
    """
    piecewise_expr = piecewise_expr.replace("^", "**")
    parsed_expr = parse_expr(piecewise_expr, local_dict=_clash1, evaluate=False)
    if not isinstance(parsed_expr, Piecewise):
        return piecewise_expr
    else:
        piecewise_expr = parsed_expr

    def process_piecewise(args):
        """
        Recursively convert sympy Piecewise args to Julia's ifelse syntax.
        """
        if not args:
            return "nothing"  # Julia's fallback for no conditions

        expr, cond = args[0]
        if cond == True:  # Sympy uses True to indicate "otherwise"
            return f"{expr}"
        elif str(cond) == "modification":
            cond = f"{cond} > 0"
        return f"ifelse({cond}, {expr}, {process_piecewise(args[1:])})"

    # Process the Piecewise arguments
    return process_piecewise(piecewise_expr.args)
