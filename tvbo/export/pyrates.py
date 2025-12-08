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
>>> from tvbo.knowledge.simulation import Network
>>> network = Network(connectome)
>>> network.to_yaml(format="pyrates", filepath="circuit.yaml")

References
----------
- PyRates documentation: https://pyrates.readthedocs.io/
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from tvbo.knowledge.simulation.localdynamics import Dynamics
    from tvbo.knowledge.simulation.network import Network


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
    dynamics: "Dynamics | None" = None,
    network: "Network | None" = None,
    filepath: str | None = None,
) -> str:
    """Export to complete PyRates experiment YAML (model + network, ready to run).

    This generates a self-contained YAML with OperatorTemplate, NodeTemplate,
    and CircuitTemplate - everything needed to run with PyRates.

    Parameters
    ----------
    dynamics : Dynamics, optional
        TVBO Dynamics model for single-node experiment.
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
    yaml_str = str(template.render(model=dynamics, network=network))

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


def from_pyrates_yaml(filepath: str) -> dict:
    """Load a PyRates YAML file and return dict for Dynamics constructor.

    Parameters
    ----------
    filepath : str
        Path to PyRates YAML file.

    Returns
    -------
    dict
        Dictionary suitable for Dynamics(**dict) constructor.
    """
    import yaml

    with open(filepath, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    return _pyrates_yaml_to_dynamics_dict(yaml_data)


def _pyrates_yaml_to_dynamics_dict(yaml_data: dict) -> dict:
    """Parse a PyRates YAML structure into a dict suitable for Dynamics constructor."""
    state_variables = {}
    parameters = {}
    derived_variables = {}
    output_transforms = {}
    name = None
    description = None

    for template_name, template_def in yaml_data.items():
        if not isinstance(template_def, dict):
            continue

        base = template_def.get("base", "")

        if base == "NodeTemplate":
            name = template_name
            continue

        if base != "OperatorTemplate":
            continue

        if name is None:
            name = template_name.replace("_op", "")

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
                rhs = match.group(2)

                initial_value = None
                var_spec = variables.get(var_name)
                if isinstance(var_spec, str) and "variable(" in var_spec:
                    iv_match = re.search(r"variable\(([\d.e+-]+)\)", var_spec)
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
                    rhs = match.group(2)

                    var_spec = variables.get(var_name)
                    if var_spec == "output":
                        output_transforms[var_name] = {
                            "name": var_name,
                            "equation": {"lhs": var_name, "rhs": rhs},
                        }
                    else:
                        derived_variables[var_name] = {
                            "name": var_name,
                            "equation": {"lhs": var_name, "rhs": rhs},
                        }

        for var_name, var_spec in variables.items():
            if var_name in state_variables or var_name in derived_variables or var_name in output_transforms:
                continue

            if isinstance(var_spec, (int, float)):
                parameters[var_name] = {
                    "name": var_name,
                    "value": float(var_spec),
                }

    return {
        "name": name or "pyrates_model",
        "description": description,
        "state_variables": state_variables,
        "parameters": parameters,
        "derived_variables": derived_variables,
        "output_transforms": output_transforms,
    }
