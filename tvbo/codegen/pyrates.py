#
# Module: pyrates.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#

"""
PyRates Integration Module
==========================

Provides modular conversion between TVBO Dynamics/Network models and PyRates YAML format.

PyRates uses a YAML-based template system with:
- OperatorTemplate: Contains equations and variables (dynamics)
- NodeTemplate: Contains one or more operators (node)
- EdgeTemplate: Defines edge operators (coupling)
- CircuitTemplate: Contains nodes and edges (network)

TVBO Templates (modular):
- tvbo-pyrates-model.yaml.mako: OperatorTemplate only (dynamics)
- tvbo-pyrates-network.yaml.mako: NodeTemplate + CircuitTemplate (topology)
- tvbo-pyrates-experiment.yaml.mako: Complete runnable YAML (model + network)

Example
-------
>>> from tvbo import Dynamics
>>> model = Dynamics("JansenRitModel")
>>>
>>> # Export model only (OperatorTemplate)
>>> operator_yaml = to_pyrates_model_yaml(model)
>>>
>>> # Export network only (Node + Circuit)
>>> network_yaml = to_pyrates_network_yaml(model)
>>>
>>> # Export complete experiment (ready to run)
>>> experiment_yaml = model.to_yaml(format="pyrates")
>>> model.to_yaml(format="pyrates", filepath="mymodel.yaml")
>>>
>>> # Load a PyRates model
>>> model = Dynamics.from_pyrates("mymodel.yaml")
>>>
>>> # Export a Network (circuit) to PyRates
>>> from tvbo.classes.network import Network
>>> network = Network(connectome)
>>> network.to_yaml(format="pyrates", filepath="circuit.yaml")

References
----------
- PyRates documentation: https://pyrates.readthedocs.io/
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tvbo.classes.dynamics import Dynamics
    from tvbo.classes.network import Network

# Reverse mapping: PyRates-safe names back to original TVBO names.
# Must mirror PYRATES_REPL in tvbo/adapters/pyrates.py and the repl dict
# in tvbo-pyrates-model.yaml.mako. Only Python-keyword / PyRates-internal
# collisions are renamed now; SymPy-colliding names are kept verbatim (the
# parser resolves declared names to symbols), so they need no reversal.
_PYRATES_REPL_REVERSE = {
    "lambda_": "lambda",
    "y_": "y",
    "dy_": "dy",
    "epsilon_": "epsilon",
}


def _unrename_pyrates(name: str) -> str:
    """Reverse PYRATES_REPL: restore original TVBO name from PyRates-safe name."""
    return _PYRATES_REPL_REVERSE.get(name, name)


def _unrename_expr(expr_str: str) -> str:
    """Reverse PYRATES_REPL in an expression string (whole-word replacement)."""
    for pyrates_name, orig_name in _PYRATES_REPL_REVERSE.items():
        # Word-boundary replacement to avoid partial matches
        expr_str = re.sub(r"\b" + re.escape(pyrates_name) + r"\b", orig_name, expr_str)
    return expr_str


def to_pyrates_model_yaml(dynamics: "Dynamics", filepath: str | None = None) -> str:
    """Export a Dynamics model to PyRates OperatorTemplate YAML (model only).

    This generates ONLY the OperatorTemplate (dynamics/equations).
    Use to_pyrates_network_yaml for topology, or to_pyrates_yaml_string for complete.

    Parameters
    ----------
    dynamics : Dynamics
        TVBO Dynamics model to export.
    filepath : str, optional
        Path to write the YAML file. If None, returns the YAML string.

    Returns
    -------
    str
        YAML string (or filepath if written to file).
    """
    from tvbo import templates

    template = templates.lookup.get_template("tvbo-pyrates-model.yaml.mako")
    yaml_str = str(template.render(model=dynamics))

    if filepath:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(yaml_str)
        return filepath

    return yaml_str


def to_pyrates_network_yaml(
    dynamics: "Dynamics | None" = None,
    network: "Network | None" = None,
    filepath: str | None = None,
) -> str:
    """Export to PyRates NodeTemplate + CircuitTemplate YAML (network topology only).

    This generates ONLY the network structure (nodes, edges, circuit).
    Operators are referenced but not defined.

    Parameters
    ----------
    dynamics : Dynamics, optional
        TVBO Dynamics model for single-node circuit.
    network : Network, optional
        TVBO Network for multi-node circuit.
    filepath : str, optional
        Path to write the YAML file. If None, returns the YAML string.

    Returns
    -------
    str
        YAML string (or filepath if written to file).
    """
    from tvbo import templates

    template = templates.lookup.get_template("tvbo-pyrates-network.yaml.mako")
    yaml_str = str(template.render(model=dynamics, network=network))

    if filepath:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(yaml_str)
        return filepath

    return yaml_str


def to_pyrates_yaml_string(
    dynamics: "Dynamics | dict[str, Dynamics] | None" = None,
    network: "Network | None" = None,
    filepath: str | None = None,
) -> str:
    """Export to complete PyRates experiment YAML (model + network, ready to run).

    This generates a self-contained YAML with OperatorTemplate, NodeTemplate,
    and CircuitTemplate - everything needed to run with PyRates.

    Parameters
    ----------
    dynamics : Dynamics or dict[str, Dynamics], optional
        TVBO Dynamics model for single-node experiment, or a dictionary
        of dynamics models keyed by name for heterogeneous networks.
    network : Network, optional
        TVBO Network for multi-node experiment.
    filepath : str, optional
        Path to write the YAML file. If None, returns the YAML string.

    Returns
    -------
    str
        YAML string (or filepath if written to file).
    """
    from tvbo import templates

    template = templates.lookup.get_template("tvbo-pyrates-experiment.yaml.mako")

    # Handle dynamics as either a single model or a dictionary
    if isinstance(dynamics, dict):
        # dynamics is a library for heterogeneous networks
        yaml_str = str(template.render(model=None, network=network, dynamics=dynamics))
    else:
        # dynamics is a single model
        yaml_str = str(template.render(model=dynamics, network=network, dynamics={}))

    if filepath:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(yaml_str)
        return filepath

    return yaml_str


# Alias for backward compatibility
def network_to_pyrates_yaml_string(network: "Network", filepath: str | None = None) -> str:
    """Export a TVBO Network to complete PyRates experiment YAML.

    Alias for to_pyrates_yaml_string(network=network, filepath=filepath).
    """
    return to_pyrates_yaml_string(network=network, filepath=filepath)


def _pyrates_to_python_expr(expr: str) -> str:
    """Convert PyRates expression syntax to Python/sympy compatible syntax.

    PyRates uses:
    - ^ for exponentiation (Python uses **)
    """
    return expr.replace("^", "**")


def from_pyrates_yaml(filepath: str, operator_key: str | None = None) -> dict:
    """Load a PyRates YAML file and return dict for Dynamics constructor.

    Parameters
    ----------
    filepath : str
        Path to PyRates YAML file.
    operator_key : str, optional
        Name of the specific OperatorTemplate to load (without _op suffix).
        If None, loads the first OperatorTemplate found.

    Returns
    -------
    dict
        Dictionary suitable for Dynamics(**dict) constructor.

    Raises
    ------
    ValueError
        If operator_key is specified but not found in the file.
    """
    import yaml

    with open(filepath, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    # Find all OperatorTemplates
    operators = {}
    for template_name, template_def in yaml_data.items():
        if not isinstance(template_def, dict):
            continue
        if template_def.get("base") == "OperatorTemplate":
            parsed = _parse_single_operator(template_name, template_def)
            operators[parsed["name"]] = parsed

    if not operators:
        raise ValueError(f"No OperatorTemplate found in {filepath}")

    if operator_key is not None:
        if operator_key not in operators:
            available = list(operators.keys())
            raise ValueError(f"Operator '{operator_key}' not found. Available: {available}")
        return operators[operator_key]

    # Return first operator if no key specified
    return next(iter(operators.values()))


def _pyrates_yaml_to_dynamics_dict(yaml_data: dict) -> dict:
    """Parse a PyRates YAML structure into a dict suitable for Dynamics constructor."""
    state_variables = {}
    parameters = {}
    derived_variables = {}
    output = []  # List of variable names (references to state_variables or derived_variables)
    operator_name = None  # Name from OperatorTemplate (preferred)
    node_name = None  # Name from NodeTemplate (fallback)
    description = None

    for template_name, template_def in yaml_data.items():
        if not isinstance(template_def, dict):
            continue

        base = template_def.get("base", "")

        if base == "NodeTemplate":
            # Extract name from NodeTemplate, removing common suffixes
            node_name = template_name
            for suffix in ("_node", "_Node", "Node"):
                if node_name.endswith(suffix):
                    node_name = node_name[: -len(suffix)]
                    break
            continue

        if base != "OperatorTemplate":
            continue

        # Extract name from OperatorTemplate, removing _op suffix
        operator_name = template_name
        for suffix in ("_op", "_Op", "Op"):
            if operator_name.endswith(suffix):
                operator_name = operator_name[: -len(suffix)]
                break

        description = template_def.get("description")

        # Parse equations
        equations = template_def.get("equations", [])
        if isinstance(equations, str):
            equations = [equations]

        variables = template_def.get("variables", {})

        for eq in equations:
            eq = str(eq).strip()

            # Check if differential equation (var' = rhs or d/dt * var = rhs)
            match_prime = re.match(r"(\w+)'\s*=\s*(.+)", eq)
            match_ddt = re.match(r"d/dt\s*\*\s*(\w+)\s*=\s*(.+)", eq)

            if match_prime or match_ddt:
                match = match_prime or match_ddt
                var_name = match.group(1)
                rhs = _pyrates_to_python_expr(match.group(2))

                initial_value = None
                var_spec = variables.get(var_name)
                if isinstance(var_spec, str):
                    # Handle variable(value) syntax
                    if "variable(" in var_spec:
                        iv_match = re.search(r"variable\(([\d.e+-]+)\)", var_spec)
                        if iv_match:
                            initial_value = float(iv_match.group(1))
                    # Handle output(value) syntax - still a state variable with initial value
                    elif "output(" in var_spec:
                        iv_match = re.search(r"output\(([\d.e+-]+)\)", var_spec)
                        if iv_match:
                            initial_value = float(iv_match.group(1))

                state_variables[var_name] = {
                    "name": var_name,
                    "equation": {"lhs": var_name, "rhs": rhs},
                    "initial_value": initial_value,
                }
            else:
                match = re.match(r"(\w+)\s*=\s*(.+)", eq)
                if match:
                    var_name = match.group(1)
                    rhs = _pyrates_to_python_expr(match.group(2))

                    var_spec = variables.get(var_name)
                    # All non-differential equations go to derived_variables
                    derived_variables[var_name] = {
                        "name": var_name,
                        "equation": {"lhs": var_name, "rhs": rhs},
                    }
                    # If marked as output, add name to output list
                    if var_spec == "output" or (isinstance(var_spec, str) and "output(" in var_spec):
                        output.append(var_name)

        for var_name, var_spec in variables.items():
            if var_name in state_variables or var_name in derived_variables or var_name in output:
                continue

            if isinstance(var_spec, (int, float)):
                parameters[var_name] = {
                    "name": var_name,
                    "value": float(var_spec),
                }
            elif isinstance(var_spec, str):
                # Handle input(value) syntax - treat as parameter with default value
                input_match = re.match(r"input\(([\d.e+-]+)\)", var_spec)
                if input_match:
                    parameters[var_name] = {
                        "name": var_name,
                        "value": float(input_match.group(1)),
                    }
                # Skip "output" markers - already handled above

    # Prefer operator name, then node name, then default
    name = operator_name or node_name or "pyrates_model"

    return {
        "name": name,
        "description": description,
        "state_variables": state_variables,
        "parameters": parameters,
        "derived_variables": derived_variables,
        "output": output,
    }


def _parse_single_operator(template_name: str, template_def: dict) -> dict:
    """Parse a single OperatorTemplate into a Dynamics-compatible dict.

    Automatically reverses PYRATES_REPL renames (e.g. ``y_`` -> ``y``,
    ``gamma_`` -> ``gamma``) so that round-tripped models recover the
    original TVBO variable names.
    """
    state_variables = {}
    parameters = {}
    derived_variables = {}
    output = []  # List of variable names (references to state_variables or derived_variables)

    # Extract name from OperatorTemplate, removing _op suffix
    name = template_name
    for suffix in ("_op", "_Op", "Op"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    description = template_def.get("description")

    # Parse equations
    equations = template_def.get("equations", [])
    if isinstance(equations, str):
        equations = [equations]

    variables = template_def.get("variables", {})

    for eq in equations:
        eq = str(eq).strip()

        # Check if differential equation (var' = rhs or d/dt * var = rhs)
        match_prime = re.match(r"(\w+)'\s*=\s*(.+)", eq)
        match_ddt = re.match(r"d/dt\s*\*\s*(\w+)\s*=\s*(.+)", eq)

        if match_prime or match_ddt:
            match = match_prime or match_ddt
            raw_var = match.group(1)
            raw_rhs = _pyrates_to_python_expr(match.group(2))
            # Reverse PYRATES_REPL renames
            var_name = _unrename_pyrates(raw_var)
            rhs = _unrename_expr(raw_rhs)

            initial_value = None
            var_spec = variables.get(raw_var)  # YAML uses renamed key
            if isinstance(var_spec, str):
                # Handle variable(value) syntax
                if "variable(" in var_spec:
                    iv_match = re.search(r"variable\(([\d.e+-]+)\)", var_spec)
                    if iv_match:
                        initial_value = float(iv_match.group(1))
                # Handle output(value) syntax - still a state variable with initial value
                elif "output(" in var_spec:
                    iv_match = re.search(r"output\(([\d.e+-]+)\)", var_spec)
                    if iv_match:
                        initial_value = float(iv_match.group(1))

            state_variables[var_name] = {
                "name": var_name,
                "equation": {"lhs": var_name, "rhs": rhs},
                "initial_value": initial_value,
            }
        else:
            match = re.match(r"(\w+)\s*=\s*(.+)", eq)
            if match:
                raw_var = match.group(1)
                raw_rhs = _pyrates_to_python_expr(match.group(2))
                var_name = _unrename_pyrates(raw_var)
                rhs = _unrename_expr(raw_rhs)

                var_spec = variables.get(raw_var)
                # All non-differential equations go to derived_variables
                derived_variables[var_name] = {
                    "name": var_name,
                    "equation": {"lhs": var_name, "rhs": rhs},
                }
                # If marked as output, add name to output list
                if var_spec == "output" or (isinstance(var_spec, str) and "output(" in var_spec):
                    output.append(var_name)

    # Parse variables that are not state/derived/output
    # Track which raw names were already consumed (as state vars or derived)
    _consumed_raw = set()
    for eq in equations:
        eq = str(eq).strip()
        m = re.match(r"(\w+)'\s*=\s*", eq) or re.match(r"d/dt\s*\*\s*(\w+)\s*=\s*", eq) or re.match(r"(\w+)\s*=\s*", eq)
        if m:
            _consumed_raw.add(m.group(1))

    for raw_var, var_spec in variables.items():
        if raw_var in _consumed_raw:
            continue
        # Also skip if unrenamed version was consumed
        var_name = _unrename_pyrates(raw_var)

        if isinstance(var_spec, (int, float)):
            parameters[var_name] = {
                "name": var_name,
                "value": float(var_spec),
            }
        elif isinstance(var_spec, str):
            if var_spec.startswith("variable") or var_spec == "variable":
                continue  # Already handled as state variable
            # Handle input(value) syntax - treat as parameter with default value
            input_match = re.match(r"input\(([\d.e+-]+)\)", var_spec)
            if input_match:
                parameters[var_name] = {
                    "name": var_name,
                    "value": float(input_match.group(1)),
                }

    return {
        "name": name,
        "description": description,
        "state_variables": state_variables,
        "parameters": parameters,
        "derived_variables": derived_variables,
        "output": output,
    }


def from_pyrates_yaml_all(filepath: str) -> dict[str, dict]:
    """Load a PyRates YAML file and return dict of all Dynamics.

    Parameters
    ----------
    filepath : str
        Path to PyRates YAML file.

    Returns
    -------
    dict[str, dict]
        Dictionary mapping operator names to Dynamics-compatible dicts.
        Keys are the clean names (without _op suffix).
    """
    import yaml

    with open(filepath, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    dynamics_dict = {}

    for template_name, template_def in yaml_data.items():
        if not isinstance(template_def, dict):
            continue

        base = template_def.get("base", "")

        if base == "OperatorTemplate":
            dynamics_data = _parse_single_operator(template_name, template_def)
            dynamics_dict[dynamics_data["name"]] = dynamics_data

    return dynamics_dict
