#
# Module: ontology_loader.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""
This module deals with loading elements from the ontology.
"""
from typing import Any

from tvb.simulator import integrators, coupling

from tvbo.export import templater
from tvbo.knowledge import ontology


# TODO: rewrite this without using templater or fix templater to work for all models
def load_tvb_model(model_name):
    """
    Load a specific TVB model by its name.

    Parameters:
    -----------
    model_name : str
        Name of the TVB model to be loaded.

    Returns:
    --------
    tvb_model : object
        The loaded TVB model.

    Example:
    --------
    >>> model = load_tvb_model("ModelName")
    """

    tvb_model = templater.model2class(
        ontology.get_model(model_name), print_source=False
    )

    tvb_model.ontology = ontology.get_model(model_name)  # why do we need this here?
    # Return class instance
    return tvb_model


def load_tvb_integrator(integrator: str = "EulerDeterministic", type: str = "deterministic") -> Any:
    """
    Loads a The Virtual Brain (TVB) integrator.

    This function searches the ontology for the specified integrator,
    and if found, it returns an instance of the integrator with its ontology attribute set.

    Args:
        integrator (str, optional): The name of the integrator to load.
                                    Defaults to "EulerDeterministic".
        type (str, optional): The type of the integrator, either "stochastic" or "deterministic".
                              Defaults to "deterministic".

    Raises:
        ValueError: If the specified integrator is not found in the ontology.

    Returns:
        Any: An instance of the specified integrator.
    """
    if type == "stochastic":
        if integrator == "default":
            integrator = "EulerStochastic"
        possible_integrators = list(ontology.onto.integrationStochastic.subclasses())
    elif type == "deterministic":
        if integrator == "default":
            integrator = "EulerDeterministic"
        possible_integrators = list(ontology.onto.integrationDeterministic.subclasses())

    possible_integrators = {i.label.first(): i for i in possible_integrators}

    if integrator not in possible_integrators.keys():
        raise ValueError(
            "Integrator {} not available for type {}".format(integrator, type)
        )

    tvb_integrator = getattr(integrators, integrator)
    tvb_integrator.ontology = possible_integrators[integrator]

    return tvb_integrator()


def load_tvb_coupling_function(coupling_function: str = "Sigmoidal") -> Any:
    """
    Loads a The Virtual Brain (TVB) coupling function.

    This function searches the ontology for the specified coupling function,
    and if found, it returns an instance of the coupling function with its ontology attribute set.

    Args:
        coupling_function (str, optional): The name of the coupling function to load.
                                           Defaults to "Sigmoidal".

    Raises:
        ValueError: If the specified coupling function is not found in the ontology.

    Returns:
        Any: An instance of the specified coupling function.
    """
    possible_cfs = list(ontology.search_class("Coupling").descendants())

    possible_cfs = {i.label.first(): i for i in possible_cfs}

    if coupling_function not in possible_cfs.keys():
        raise ValueError("Coupling function {} not available".format(coupling))

    tvb_coupling = getattr(coupling, coupling_function)
    tvb_coupling.ontology = possible_cfs[coupling_function]

    return tvb_coupling()


def export_metadata():
    """
    Export metadata to a desired format.

    TODO: Implementation of this function is pending.
    """
    pass
