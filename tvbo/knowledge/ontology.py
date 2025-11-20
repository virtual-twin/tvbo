#  ontology.py
#
# Created on Mon Aug 07 2023
# Author: Leon K. Martin
#
# Copyright (c) 2023 Charité Universitätsmedizin Berlin
#
"""
---
title: "Ontology Module for TVB-O"
author: Leon Martin
---

This module provides a set of functions to interact with the ontology of TVB-O.
It includes functions to:

- Retrieve package version.
- Render ontology objects using labels or IRIs.
- Access and extract various parts of the ontology like models, variables, parameters, etc.
- Compare different models based on their parameters.
- Get properties of parameters.

### Usage:
```python
from tvbo.knowledge import ontology
jansen_rit_model = ontology.get_model("JansenRit")
```

Author:
    Leon K. Martin (2023)

Copyright:
    Copyright (c) 2023 Charité Universitätsmedizin Berlin
"""

import collections
import os
import re
import tempfile
from os.path import abspath, dirname, isfile, join, realpath
from textwrap import wrap
from typing import List, Dict, Tuple, Optional, Union, Any

import numpy as np
import owlready2
import pandas as pd
import simple_colors as sc
from fuzzywuzzy import process
from owlready2 import *
from tvbo.utils import Bunch

from tvbo import knowledge, parse
from tvbo.datamodel import tvbo_datamodel

# %%

ROOT_DIR = abspath(join(abspath(dirname(__file__)), ".."))
np.random.seed(1312)

functional_models = [
    # "CakanObermayer",
    "CoombesByrne",
    "CoombesByrne2D",
    "DumontGutkin",
    "Epileptor2D",
    "Epileptor5D",
    # "EpileptorCodim3",
    # "EpileptorCodim3SlowMod",
    "EpileptorRestingState",
    "GastSchmidtKnosche_SD",
    "GastSchmidtKnosche_SF",
    "Generic2dOscillator",
    "GenericLinear",
    "Hopfield",
    "JansenRit",
    "KIonEx",
    "Kuramoto",
    "LarterBreakspear",
    "MontbrioPazoRoxin",
    # "ReducedSetFitzHughNagumo",
    # "ReducedSetHindmarshRose",
    "ReducedWongWang",
    "ReducedWongWangExcInh",
    "SupHopf",
    "WilsonCowan",
    "ZerlautAdaptationFirstOrder",
    # "ZerlautAdaptationSecondOrder",
    "ZetterbergJansen",
]


def find_version() -> str:
    """
    Retrieves the package version from the `__init__.py` file.

    Returns:
        str: The version of the TVBO package.

    Raises:
        RuntimeError: If the version cannot be found in the `__init__.py` file.
    """

    path_to_init = os.path.join(ROOT_DIR, "__init__.py")
    with open(path_to_init, "r", encoding="utf-8") as f:
        content = f.read()
        version_match = re.search(r"^__version__ = ['\"](.*?)['\"]$", content, re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("version cannot be found!")


DATA_DIR = realpath(join(ROOT_DIR, "data"))
ONTO_DIR = join(DATA_DIR, "ontology")
# %% Load Ontology

original_path = join(
    ONTO_DIR,
    "tvb-o.owl",
)

with open(original_path, "r", encoding="utf-8") as f:
    xml = f.read()

xml = xml.replace(
    """<owl:imports rdf:resource="https://raw.githubusercontent.com/SciCrunch/NIF-Ontology/atlas/ttl/atom.ttl"/>""",
    "",
)

with tempfile.NamedTemporaryFile(
    suffix=".owl", delete=False, mode="w", encoding="utf-8"
) as tmp:
    tmp.write(xml)
    tmp_path = tmp.name

onto = get_ontology("file://" + tmp_path).load()
onto.load()  # TODO: Check if redundant load can be removed

iri = onto.base_iri
namespace = onto.get_namespace(iri)  # TODO: is this used?

df_tvbo = pd.read_csv(join(DATA_DIR, "_tvb-o.csv"), sep=";")


# %% global functions


def get_onto() -> owlready2.namespace.Ontology:
    return onto


def render_using_label(entity) -> str:
    """
    Renders the ontology objects using their labels.

    Parameters:
        entity (owlready2.entity): The ontology class or entity to be rendered.

    Returns:
        str: The label of the given ontology entity."""

    return entity.label.first() or entity.name


def render_using_iri(entity) -> str:
    """
    Renders the ontology objects using their IRIs.

    Parameters:
        entity (owlready2.entity): The ontology class or entity to be rendered.

    Returns:
        str: The IRI of the given ontology entity."""

    return entity.iri


# Needed later for Parameter specification
def intersection(lst1, lst2) -> list:
    """
    Computes the intersection of two lists.

    Parameters:
        lst1 (list): The first list.
        lst2 (list): The second list.

    Returns:
        list: The intersection of the two given lists."""
    lst1 = list(lst1)
    lst2 = list(lst2)
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def disintersection(lst1, lst2) -> list:
    """
    Computes the unique elements of two lists.

    Parameters:
        lst1 (list): The first list.
        lst2 (list): The second list.

    Returns:
        list: The elements that are unique to each list."""
    lst3 = list(set(lst1).symmetric_difference(set(lst2)))
    return lst3


def get_sorted_dict(class_list) -> dict:
    """
    Creates a dictionary from a list of ontology classes. The dictionary's keys are the class labels
    and its values are the class objects. The dictionary is sorted alphabetically based on its keys.

    Parameters:
        class_list (list): The list of ontology classes.

    Returns:
        dict: A sorted dictionary of class labels and their corresponding class objects.
    """

    d = dict()
    for s in class_list:
        d[s.label.first()] = s

    return collections.OrderedDict(
        sorted(
            d.items(),  # key=lambda i: i[0].lower()
        )
    )


# %% OWL Miscellaneous

set_render_func(render_using_label)


# %% Functions for extracting TVB-O variables
# NMM name must match label of model in TVBO
def wrap_text(text, line_length=100, line_breaks="\n") -> str:
    """
    Pretty print a string with automatic line breaks at specified intervals,
    while preserving existing new lines.

    Parameters:
        text (str): The text to be printed.
        line_length (int): The maximum length of each line.
    """

    def wrap_line(line):
        words = line.split()
        wrapped_line = ""
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= line_length:
                wrapped_line += word + " "
                current_length += len(word) + 1
            else:
                wrapped_line += line_breaks + word + " "
                current_length = len(word) + 1

        return wrapped_line

    # Split the text into lines and apply wrapping to each line
    lines = text.split("\n")
    wrapped_text = line_breaks.join(wrap_line(line) for line in lines)

    return wrapped_text


def hangident(text, indent=4) -> str:
    """
    Indent a string by a specified amount.

    Parameters:
        text (str): The text to be indented.
        indent (int): The amount to indent the text by.
    """
    splitted = text.split("\n")
    return splitted[0] + "\n" + "\n".join(" " * indent + line for line in splitted[1:])


def get_info(cls) -> str:
    from tvbo.parse.literature import render_citation

    if isinstance(cls, str):
        cls = onto[cls]
    info = cls.label.first() + "\n"
    info += "=" * len(cls.label.first()) + "\n\n"
    info += wrap_text(cls.definition.first())
    references = cls.has_reference
    if len(references) > 0:
        info += "\n\n"
        info += "References\n"
        info += "-" * 10 + "\n"
        for ref in references:
            info += hangident(wrap_text(render_citation(ref))) + "\n"
    return info


def ontology_info(print_info=True, return_info=False, return_df=False):
    info = Bunch(
        classes=list(onto.classes()),
        tvb_classes=list(onto.TheVirtualBrain.descendants()),
        annotated_go_classes=list(onto.TVBGO.descendants()),
        properties=list(onto.properties()),
        annotation_properties=list(onto.annotation_properties()),
        object_properties=list(onto.object_properties()),
        data_properties=list(onto.data_properties()),
        indivduals=list(onto.individuals()),
        tvb_models=list(get_models().keys()),
        biological_models=list(onto.BiologicalModel.subclasses()),
        phenomenological_models=list(onto.PhenomenologicalModel.subclasses()),
        parameter_categories=get_subclass_list(
            onto.ModelParametersCatalogue,
            level=2,
            exclude_cls=onto.NeuralMassModel,
        ),
    )
    if print_info:
        print("TVBO base IRI:", iri)
        for k, v in info.items():
            print("Number of", sc.blue(k) + ":", sc.green(len(v)))
    if return_info:
        return info
    if return_df:
        df = pd.DataFrame()
        for k, v in info.items():
            df.at[k, "count"] = len(v)
        df.index.name = "TVB-O component"
        return df


def search_class(
    label,
) -> Union[owlready2.ThingClass, owlready2.triplelite._SearchList]:
    """
    Searches for an ontology class using a given label.

    Parameters:
        label (str): The label to search for, with regex support.

    Returns:
        owlready2.ThingClass or owlready2.triplelite._SearchList:
        The ontology class(es) that match the given label."""

    tvbo_classes = onto.search(label=label)
    if len(tvbo_classes) == 1:
        tvbo_classes = tvbo_classes.first()
    return tvbo_classes


def search_in_model(
    search_str, model: owlready2.ThingClass, wildcards=True
) -> Optional[Union[owlready2.ThingClass, List[owlready2.ThingClass]]]:
    if isinstance(model, str):
        model = get_model(model)
    if wildcards:
        search_str = f"*{search_str}*"  # add wildcards
    else:
        search_str = f"{search_str}{get_model_suffix(model)}"
    label_search_result = list(onto.search(label=search_str, _case_sensitive=False))
    def_search = list(onto.search(definition=search_str, _case_sensitive=False))
    search = def_search + label_search_result
    overlap = intersection(list(model.descendants(include_self=False)), search)
    if len(overlap) == 1:
        return overlap[0]
    elif len(overlap) == 0:
        return None
    return overlap


def filter_cls_list(cls_list, by) -> List[owlready2.ThingClass]:
    """
    Filters out classes from a list that have a specific ancestor.

    Parameters:
        cls_list (list): The list of classes to be filtered.
        by (owlready2.ThingClass): The ancestor class to filter by.

    Returns:
        list: The filtered list of classes."""

    filtered_list = list()
    for c in cls_list:
        if by in c.ancestors():
            pass
        else:
            filtered_list.append(c)
    return filtered_list


def get_subclass_list(cls, level=1, exclude_cls=None) -> List[owlready2.ThingClass]:
    """
    Retrieves subclasses for a given ontology class, up to a specified depth.

    Parameters:
        cls (owlready2.ThingClass): The ontology class to retrieve subclasses for.
        level (int, optional): The depth to retrieve subclasses up to. Default is 1.
        exclude_cls (owlready2.ThingClass, optional): A class to exclude from the results. Default is None.

    Returns:
        list: The list of subclasses for the given ontology class."""

    results = list(cls.subclasses())

    if level == 1:
        return list(results)

    for i in range(level):
        for r in results:
            results += get_subclasses(r)
    if exclude_cls is not None:
        results = filter_cls_list(results, exclude_cls)
    return list(results)


def get_type(c: owlready2.ThingClass) -> owlready2.ThingClass:
    """
    Retrieves the type of a TVB class for a given entity.

    Parameters:
        c: The TVB class instance.

    Returns:
        str: The type of the TVB entity.
    """
    try:
        ancestors = c.ancestors()
    except AttributeError:
        ancestors = c.is_a

    types = [
        onto.Coupling,
        onto.NeuralMassModel,
        onto.integration,
        onto.Noise,
        onto.TVBGO,
        onto.JournalArticle,
    ]

    for entity_type in types:
        if entity_type in ancestors:
            return entity_type

    return onto.TheVirtualBrain


def get_def(cls, mode="short") -> str:
    """
    Retrieve the description or definition of a class based on the specified mode.

    This function fetches either a short description or a long definition of the given class.
    If the requested type (short or long) is not available, it attempts to fetch the other type.

    Parameters
    ----------
    cls : class
        The class for which the description or definition is being retrieved.
    mode : str, optional
        The mode specifying the type of text to retrieve: 'short' for a brief description
        or 'long' for a detailed definition. The default is 'short'.

    Returns
    -------
    str
        The description or definition of the class. If neither is available, it returns None.

    """
    desc = cls.description.first() if cls.description else ""
    defi = cls.definition.first() if cls.definition else ""

    if mode == "short" and desc != "":
        return desc
    elif mode == "short" and desc == "" and defi != "":
        return defi.split(".")[0]
    elif mode == "long" and defi != "":
        return defi
    else:
        return ""


def get_subclasses(tvbo_class, recursive=False) -> List[owlready2.ThingClass]:
    """
    Retrieves the subclasses for a given TVB-O class.

    Parameters:
        tvbo_class (owlready2.ThingClass): The TVB-O class to retrieve subclasses for.
        recursive (bool, optional): If True, retrieves subclasses recursively. Default is False.

    Returns:
        list: The list of subclasses for the given TVB-O class."""

    subclasses = onto.get_children_of(tvbo_class)

    if recursive:
        r_subclasses = subclasses.copy()
        for sc in subclasses:
            r_subclasses += onto.get_children_of(sc)

        return r_subclasses

    return subclasses


def get_superclasses(tvbo_class) -> List[owlready2.ThingClass]:
    """
    Retrieves the superclasses for a given TVB-O class.

    Parameters:
        tvbo_class (owlready2.ThingClass): The TVB-O class to retrieve superclasses for.

    Returns:
        list: The list of superclasses for the given TVB-O class."""

    return onto.get_parents_of(tvbo_class)


def get_models(model_type="NMM", from_df=False) -> Dict[str, owlready2.ThingClass]:
    """
    Retrieves all TVB-O models of a given type.

    Parameters:
        model_type (str, optional): The type of model to retrieve. Default is "NMM".
        from_df (bool, optional): If True, retrieves models from a dataframe. Default is False.

    Returns:
        dict: A dictionary of model labels and their corresponding ontology class objects.
    """

    if from_df:
        classes = onto.classes()
        models = dict()
        for cl in classes:
            if cl.name == "Thing":
                continue
            cl_type = get_type(cl)
            if cl_type == model_type:
                models[cl.label[0]] = cl
    else:
        models = dict()
        for NMM in onto.NeuralMassModel.subclasses():
            models[NMM.label.first()] = NMM

    return {m: k for m, k in models.items() if m in functional_models}


def get_model(
    label: str = "JansenRit", model_type="NMM", verbose=False
) -> owlready2.ThingClass:
    """
    Retrieves a specific TVB-O model using its label.

    Parameters:
        label (str, optional): The label of the model to retrieve. Default is "JansenRit".
        model_type (str, optional): The type of model to retrieve. Default is "NMM".

    Returns:
        owlready2.ThingClass: The ontology class for the specified model."""

    if isinstance(label, owlready2.ThingClass):
        return label

    models = get_models(model_type=model_type)
    synonyms = dict()
    for k, model in models.items():
        for synonym in model.synonym:
            synonyms[synonym] = model

    if label in models.keys():
        NMM = models[label]
    elif label in synonyms.keys():
        NMM = synonyms[label]
    else:
        if verbose:
            print(
                f"Model {label} not found in {model_type} models.\n Valid models are {sorted(models.keys())}"
            )
        return onto.NeuralMassModel()  # return empty NMM class
    default_world.full_text_search_properties.append(NMM)
    return NMM


def get_integrator(integration_method="Heun") -> owlready2.ThingClass:
    search_res = knowledge.query.label_search(integration_method)

    available_integrators = onto.IntegrationMethod.descendants(include_self=False)

    av_int = intersection(search_res, available_integrators)

    if len(av_int) == 0:
        print("No integrators found for method: ", integration_method)
        print("Available integrators: ", available_integrators)
        integrator = onto.IntegrationMethod()
    if len(av_int) > 1:
        print("Multiple integrators found for method: ", integration_method)
        print("Available integrators: ", av_int)
        print("Using the first one: ", av_int[0])
        integrator = av_int[0]
    else:
        integrator = av_int[0]
    return integrator


def get_coupling_functions() -> Dict[str, owlready2.ThingClass]:
    return {CF.label.first(): CF for CF in onto.Coupling.subclasses()}


def get_coupling_function(
    label="Linear", verbose=True
) -> Optional[owlready2.ThingClass]:
    coupling_functions = get_coupling_functions()
    synonyms = dict()
    for k, cf in coupling_functions.items():
        synonyms.update({s: cf for s in cf.synonym})
    if label in coupling_functions.keys():
        CF = coupling_functions[label]
    elif label in synonyms.keys():
        CF = synonyms[label]
    else:
        if verbose:
            print(
                f"Coupling function {label} not found.\nValid coupling functions are: {coupling_functions.keys()}"
            )
        return None
    default_world.full_text_search_properties.append(CF)
    return CF


def get_model_acronym(NMM) -> Optional[str]:
    """
    Retrieves the acronym for a given TVB model.

    Parameters:
        NMM (owlready2.ThingClass or str): The TVB model to retrieve the acronym for.

    Returns:
        str: The acronym for the given TVB model."""

    if isinstance(NMM, str):
        NMM = get_model(NMM)

    return NMM.acronym.first()


def get_model_suffix(NMM) -> str:
    """
    Retrieves the suffix for a given TVB model, based on its acronym.

    Parameters:
        NMM (owlready2.ThingClass or str): The TVB model to retrieve the suffix for.

    Returns:
        str: The suffix for the given TVB model."""

    acr = get_model_acronym(NMM)
    if isinstance(acr, type(None)):
        acr = ""
        for char in NMM.label.first():
            if char.isupper() or char.isnumeric():
                acr += char

    return "_" + acr


def replace_suffix(cls) -> str:
    if isinstance(cls, str):
        clsearch = onto.search(label=cls).first()
        if isinstance(clsearch, type(None)):
            return cls
        cls = clsearch
    label = cls.label.first()
    if hasattr(cls, "ancestors"):
        ancestors = cls.ancestors()
    else:
        ancestors = [a for b in cls.is_a for a in b.ancestors()]
    acr_classes = onto.NeuralMassModel.descendants()
    acr_classes.update(onto.Coupling.descendants())
    acr_classes.update(onto.DataTypes.descendants())
    nmm_parents = intersection(ancestors, acr_classes)
    for nmm in nmm_parents:
        suffix = get_model_suffix(nmm)
        if suffix in cls.label.first():
            label = label.replace(suffix, "")
    return label


def get_model_variables(NMM) -> List[owlready2.ThingClass]:
    """
    Retrieves the variables for a given TVB model.

    Parameters:
        NMM (owlready2.ThingClass or str): The TVB model to retrieve the variables for.

    Returns:
        list: The list of variables for the given TVB model."""

    if isinstance(NMM, str):
        NMM = get_model(NMM)

    if isinstance(NMM, owlready2.ThingClass):
        NMM = [NMM]
    variables = list(
        default_world.sparql(
            """
           SELECT ?y
           { ?y rdfs:subClassOf* ?? }
            """,
            NMM,
        )
    )
    variables.remove(NMM)

    return variables


def get_property_annotation(tvbo_class, property) -> List[owlready2.ThingClass]:
    """
    Retrieves annotations for a given ontology class and property.

    Parameters:
        tvbo_class (owlready2.ThingClass or str): The ontology class to retrieve annotations for.
        property (str): The property to retrieve annotations for.

    Returns:
        list: The annotations for the given ontology class and property."""

    if not isinstance(tvbo_class, list):
        tvbo_class = [tvbo_class]

    CE = list(
        default_world.sparql(
            """
                PREFIX tvb-o: <http://www.thevirtualbrain.org/tvb-o/>
                SELECT  ?x    WHERE {

                    ?class owl:someValuesFrom ??
                    ?class owl:onProperty """
            + property
            + """
                    ?x rdfs:subClassOf* ?class
        }
            """,
            tvbo_class,
        )
    )
    return CE


def select_variables(variables, property) -> List[owlready2.ThingClass]:
    """
    Selects variables from a list based on a given property.

    Parameters:
        variables (list): The list of variables to select from.
        property (str): The property to use for selecting variables.

    Returns:
        list: The selected variables."""

    selection = []

    for v in variables:
        CE = get_property_annotation(v, property)
        selection = selection + CE

    return selection


def get_model_parameters(
    NMM, return_as_dict=True
) -> Union[Dict[str, owlready2.ThingClass], List[owlready2.ThingClass]]:
    """
    Retrieves the parameters for a given TVB model.

    Parameters:
        NMM (owlready2.ThingClass or str): The TVB model to retrieve the parameters for.

    Returns:
        dict: A dictionary of parameter labels and their corresponding ontology class objects.
    """
    if isinstance(NMM, str):
        NMM = get_model(NMM)
    suffix = get_model_suffix(NMM)
    if hasattr(NMM, "descendants"):
        parameters = sorted(
            [p for p in NMM.descendants() if onto.Parameter in p.is_a],
            key=lambda x: x.label,
        )
    else:
        parameters = sorted(
            NMM.has_parameter,
            key=lambda x: x.label,
        )

    if return_as_dict:
        parameters = get_sorted_dict(parameters)

    return {
        replace_suffix(p).replace(f"_{NMM.name}", ""): p for k, p in parameters.items()
    }


# TODO: add at least only_global in docstring
def get_model_coupling_terms(
    NMM, only_global=True, return_as_dict=True
) -> Union[Dict[str, owlready2.ThingClass], List[owlready2.ThingClass]]:
    """
    Retrieves the coupling terms for a given TVB model.

    Parameters:
        NMM (owlready2.ThingClass or str): The TVB model to retrieve the coupling terms for.

    Returns:
        dict: A dictionary of coupling term labels and their corresponding ontology class objects.
    """
    if isinstance(NMM, str):
        NMM = get_model(NMM)
    suffix = get_model_suffix(NMM)
    parameters = sorted(
        [td for td in NMM.descendants() if onto.CouplingTerm in td.is_a],
        key=lambda x: x.label,
    )
    if only_global:
        parameters = intersection(parameters, onto.GlobalConnectivity.descendants())

    if return_as_dict:
        parameters = get_sorted_dict(parameters)

    return {k.replace(suffix, ""): p for k, p in parameters.items()}


def get_model_constants(
    NMM, return_as_dict=True
) -> Union[Dict[str, owlready2.ThingClass], List[owlready2.ThingClass]]:
    if isinstance(NMM, str):
        NMM = get_model(NMM)

    constants = sorted(
        [td for td in NMM.descendants() if onto.Constant in td.is_a],
        key=lambda x: x.label,
    )
    if return_as_dict:
        constants = get_sorted_dict(constants)

    return {replace_suffix(k): c for k, c in constants.items()}


def get_model_coefficients(NMM) -> Dict[str, owlready2.ThingClass]:
    """
    Retrieves the coefficients for a given TVB model.

    Parameters:
        NMM (owlready2.ThingClass or str): The TVB model to retrieve the coefficients for.

    Returns:
        dict: A dictionary of coefficient labels and their corresponding ontology class objects.
    """

    variables = get_model_variables(NMM)
    parameters = select_variables(variables, property="tvb-o:is_coefficient_of")
    parameters = intersection(variables, parameters)
    parameters = get_sorted_dict([p[0] for p in parameters])

    return parameters


def get_model_conditionals(NMM) -> Dict[str, owlready2.ThingClass]:
    if isinstance(NMM, str):
        NMM = get_model(NMM)

    conditionals = intersection(
        list(NMM.descendants()),
        list(onto.ConditionalDerivedVariable.subclasses()),
    )
    conditionals = get_sorted_dict(conditionals)

    return {replace_suffix(k): c for k, c in conditionals.items()}


def get_model_functions(NMM) -> Dict[str, owlready2.ThingClass]:
    """
    Retrieves the functions for a given TVB model.

    Parameters:
        NMM (owlready2.ThingClass or str): The TVB model to retrieve the functions for.

    Returns:
        dict: A dictionary of function labels and their corresponding ontology class objects.
    """

    if isinstance(NMM, str):
        NMM = get_model(NMM)
    suffix = get_model_suffix(NMM)
    functions = intersection(
        list(NMM.descendants()),
        list(onto.Function.subclasses()),
    )
    functions = get_sorted_dict(functions)

    # drop = list()
    # for k, v in functions.items():
    #     if "c_0" == k or "coupling" in k or "lrc" in k:
    #         drop.append(k)
    # for k in drop:
    #     del functions[k]

    return {k.replace(suffix, ""): f for k, f in functions.items()}


def get_model_arguments(NMM) -> Dict[str, owlready2.ThingClass]:
    """
    Retrieves the arguments for a given TVB model.

    Parameters:
        NMM (owlready2.ThingClass or str): The TVB model to retrieve the arguments for.

    Returns:
        dict: A dictionary of argument labels and their corresponding ontology class objects.
    """

    variables = get_model_variables(NMM)
    arguments = select_variables(variables, property="tvb-o:is_argument_of")
    arguments = intersection(variables, arguments)
    arguments = get_sorted_dict([f[0] for f in arguments])

    return arguments


def get_model_derivatives(
    NMM, return_as_dict=True
) -> Union[Dict[str, owlready2.ThingClass], List[owlready2.ThingClass]]:
    """
    Retrieves the derivatives for a given TVB model.

    Parameters:
        NMM (owlready2.ThingClass or str): The TVB model to retrieve the derivatives for.

    Returns:
        dict: A dictionary of derivative labels and their corresponding ontology class objects.
    """

    if isinstance(NMM, str):
        NMM = get_model(NMM)

    time_derivatives = sorted(
        [td for td in NMM.descendants() if onto.TimeDerivative in td.is_a],
        key=lambda x: x.label,
    )

    if return_as_dict:
        time_derivatives = get_sorted_dict(time_derivatives)

    return time_derivatives


def get_model_statevariables(
    NMM, return_as_dict=True
) -> Union[Dict[str, owlready2.ThingClass], List[owlready2.ThingClass]]:
    """
    Retrieves the state variables for a given TVB model.

    Parameters:
        NMM (owlready2.ThingClass or str): The TVB model to retrieve the state variables for.
        return_as_dict (bool, optional): If True, returns the state variables as a dictionary. Default is True.

    Returns:
        dict: A dictionary of state variable labels and their corresponding ontology class objects.
    """

    if isinstance(NMM, str):
        NMM = get_model(NMM)

    SV = sorted(
        intersection(list(NMM.subclasses()), list(onto.StateVariable.subclasses())),
        key=lambda x: x.label,
    )
    if return_as_dict:
        SV = get_sorted_dict(SV)
    return {replace_suffix(k): p for k, p in SV.items()}


def get_model_cvars(
    NMM, return_as_dict=True
) -> Union[Dict[str, owlready2.ThingClass], List[owlready2.ThingClass]]:
    """
    Retrieves the cvars (coupling variables) for a given TVB model.

    Parameters:
        NMM (owlready2.ThingClass or str): The TVB model to retrieve the cvars for.

    Returns:
        list: The cvars for the given TVB model."""
    if isinstance(NMM, str):
        NMM = get_model(NMM)
    cvars = NMM.has_cvar
    for k, v in get_model_derivatives(NMM).items():
        if "c_pop" in v.value.first():
            for isa in v.is_a:
                if onto.StateVariable in isa.is_a:
                    cvars.append(isa)
    if len(cvars) == 0:
        cvars = get_model_statevariables(NMM).values()
    if return_as_dict:
        cvars = get_sorted_dict(cvars)
    if NMM == onto.JansenRit:
        cvars.pop("y4_JR")
    return {replace_suffix(k): p for k, p in cvars.items()}


def get_default_values(
    NMM, tvb_name=False, class_as_key=False
) -> Dict[str, Union[float, bool, int]]:
    """
    Retrieves the default values for a given TVB model's parameters.

    Parameters:
        NMM (owlready2.ThingClass or str): The TVB model to retrieve the default values for.
        tvb_name (bool, optional): If True, uses the TVB name for the parameter. Default is False.

    Returns:
        dict: A dictionary of parameter names and their default values."""

    if isinstance(NMM, str):
        NMM = get_model(NMM)
    # TODO: suff not used, remove?
    suff = get_model_suffix(NMM)
    values = dict()
    parameters = get_model_parameters(NMM)
    parameters.update(get_model_constants(NMM))
    for k, v in parameters.items():
        if tvb_name:
            k = v.tvbSourceVariable.first()
        if class_as_key:
            k = v
        else:
            k = replace_suffix(k)
        str_val = v.value.first() if onto.Constant in v.is_a else v.defaultValue.first()
        if str_val == "True":
            values[k] = True
        elif str_val == "False":
            values[k] = False
        else:
            values[k] = float(str_val)

        if onto.NeuralMassModel in v.is_a:
            values["local_coupling"] = 0
            values["c_pop0"] = 0
            values["c_pop1"] = 0

    return values


def contains_math_char(s) -> bool:
    math_chars = ["+", "-", "*", "/", "=", "^"]
    return any(char in s for char in math_chars)


def add_spaces_around_math_chars(s) -> str:
    # Define the pattern: any math character not already surrounded by spaces
    # Math characters included here are +, -, *, /, and =
    # You can add more characters if needed
    pattern = r"(?<!\s)([\+\-\*/=])(?!\s)"

    # Replacement function: add spaces around the math character
    def repl(match):
        return f" {match.group(1)} "

    # Perform the substitution
    return re.sub(pattern, repl, s)


def get_model_vois(model) -> Tuple[str]:
    math_chars = ["+", "-", "*", "/", "=", "^"]

    if isinstance(model, str):
        model = get_model(model)
    suffix = get_model_suffix(model)

    relations = {
        m.label.first().replace(suffix, "")
        for m in model.has_default_voi
        if m.name != "Thing"
    }
    extra_vois = model.VOIs.first()
    if extra_vois:
        relations.update(extra_vois.split(","))

    op_vois = []
    single_vois = []
    for r in set(relations):
        if contains_math_char(r):
            op_vois.append(add_spaces_around_math_chars(r))
        else:
            single_vois.append(r)
    vois = single_vois + op_vois
    if len(vois) == 0:
        vois = list(get_model_statevariables(model).keys())
    return tuple(
        sorted(set([v.replace('"', "").replace("'", "").strip() for v in vois]))
    )


# TODO: integrate biological surrogates in TVB-O
# def get_biological_surrogates(tvbo_class):
#     return tvbo_class.RDzVqsqT7POi88UbtfVuBH1


def get_definition(tvbo_class) -> str:
    """
    Retrieves the definition for a given ontology class.

    Parameters:
        tvbo_class (owlready2.ThingClass or str): The ontology class to retrieve the definition for.

    Returns:
        str: The definition for the given ontology class."""

    return "\n".join(wrap(tvbo_class.definition[0], width=100))


def get_parameters_by_catalogue(
    NMM: owlready2.ThingClass, param_key: str
) -> pd.DataFrame:
    """
    Retrieves parameters for a given TVB model, based on a specified parameter catalogue.

    Parameters:
        NMM (owlready2.ThingClass or str): The TVB model to retrieve parameters for.
        param_key (str): The parameter catalogue to use for retrieving parameters.

    Returns:
        pandas.DataFrame: A dataframe of parameters, their definitions, and their categories.
    """

    if isinstance(NMM, str):
        NMM = get_model(NMM)

    params = pd.DataFrame(columns=["NMM_Parameter", "Definition", "ParameterCategory"])
    i = 0

    for subclass in NMM.descendants():
        for anc in subclass.ancestors():
            if anc.name == "Thing":
                continue
            anc_label = anc.label.first()

            if param_key.lower() in anc_label.lower():
                params.loc[i, "NMM_Parameter"] = subclass.label.first()
                params.loc[i, "Definition"] = subclass.definition.first()
                params.loc[i, "ParameterCategory"] = anc_label
                i += 1

    return params


####################
# Class Properties #
####################
def get_object_properties(ontology_class, include_restriction=True):
    object_properties = []
    for p, o in knowledge.query.get_class_relationships(ontology_class):
        if type(o) == owlready2.class_construct.Restriction:
            if include_restriction:
                if not {o.property.name: o.value} in object_properties:
                    object_properties.append({o.property.name: o.value})

        elif type(o) == owlready2.ThingClass:
            if not {"is_a": o} in object_properties:
                object_properties.append({"is_a": o})

    return object_properties


def get_class_properties(cls):
    if isinstance(cls, str):
        cls = knowledge.query.search_by_label(str(cls))[0]

    prop = dict()
    prop["label"] = cls.label.first()
    prop["identifier"] = cls.identifier.first() if hasattr(cls, "identifier") else None
    prop["annotation_properties"] = {}

    for annot_prop in onto.annotation_properties():
        val = getattr(cls, annot_prop.python_name, [])
        if len(val) > 0:
            val = val[0]
            if annot_prop.python_name in ["value", "defaultValue"]:
                try:
                    val = eval(val)
                except:
                    pass
            prop["annotation_properties"][annot_prop.python_name] = val

    prop["object_properties"] = get_object_properties(cls)
    return prop


def get_class_annotation_properties(cls):
    return get_class_properties(cls)["annotation_properties"]


def get_class_object_properties(cls):
    return get_class_properties(cls)["object_properties"]


def get_class_data_properties(cls):
    return get_class_properties(cls)["data_properties"]


####################
# Model Comparison #
####################
def join_set(a):
    return ", ".join(list(set(a)))


def compare_models(model1, model2, by="ParameterCatalogue") -> pd.DataFrame:
    """
    Compares two TVB models based on their parameters or another specified metric.

    Args:
        model1 (owlready2.ThingClass or str): The first TVB model for comparison.
        model2 (owlready2.ThingClass or str): The second TVB model for comparison.
        by (str, optional): The metric to use for comparison. Default is "ParameterCatalogue".

    Returns:
        pandas.DataFrame: A dataframe comparing the two TVB models based on the specified metric.
    """

    i = 0
    df_comp = pd.DataFrame()

    model1_vars = dict()
    model2_vars = dict()

    model1_suffix = get_model_suffix(model1)
    model2_suffix = get_model_suffix(model2)

    for d in model1.descendants():
        model1_vars.update({d: list(d.ancestors())})

    for d in model2.descendants():
        model2_vars.update({d: list(d.ancestors())})

    if by == "ParameterCatalogue":
        for k1, v1 in model1_vars.items():
            for k2, v2 in model2_vars.items():
                inters = intersection(v1, v2)

                inters = [
                    i
                    for i in inters
                    if i
                    in onto.ModelParametersCatalogue.descendants(include_self=False)
                ]
                if onto.StateVariable in inters:
                    inters.remove(onto.StateVariable)

                if onto.TransferSigmoidFunctionProperties in inters:
                    inters.remove(onto.TransferSigmoidFunctionProperties)

                if len(inters) > 0:
                    df_comp.at[i, model1.label.first()] = k1.label.first().replace(
                        model1_suffix, ""
                    )
                    df_comp.at[i, model2.label.first()] = k2.label.first().replace(
                        model2_suffix, ""
                    )
                    df_comp.at[i, "Parameter Catalogue"] = ", ".join(
                        [i.label.first() for i in inters]
                    )
                    i += 1
        # df_comp.sort_values(by=["Parameter Catalogue", model1.label.first()])
        df_comp = df_comp.groupby("Parameter Catalogue", as_index=False).agg(
            {model1.label.first(): join_set, model2.label.first(): join_set}
        )

    return df_comp


########################
# Parameter Properties #
########################


def get_range(variable, return_array=False) -> Union[Tuple, np.ndarray]:
    """
    Retrieves the range for a given ontology variable.

    Parameters:
        variable (owlready2.ThingClass or str): The ontology variable to retrieve the range for.
        return_array (bool, optional): If True, returns the range as an array. Default is False.

    Returns:
        tuple or numpy.ndarray: The range for the given ontology variable."""

    if isinstance(variable, str):
        vrange = variable
        return tuple(
            val.strip()
            for val in vrange.replace("lo=", "")
            .replace("hi=", "")
            .replace("step=", "")
            .split(",")
        )
    else:
        vrange = variable.range.first()

    if onto.Constant in variable.is_a:
        value = variable.value.first()
        return value, value, 0

    if isinstance(vrange, type(None)):
        vrange = variable.stateVariableRange.first()
        if not isinstance(vrange, type(None)) and vrange != "":
            # Clean the format before splitting
            vrange = vrange.replace("lo=", "").replace("hi=", "").replace("step=", "")
            vrange = vrange.split(",")
            if vrange:
                return tuple([eval(v.strip(), {}, {"pi": np.pi}) for v in vrange])
        else:
            lo = (
                variable.defaultValue.first()
                if not isinstance(variable.defaultValue.first(), type(None))
                and variable.defaultValue.first() != "None"
                else 1e-100
            )
            hi = (
                variable.defaultValue.first()
                if not isinstance(variable.defaultValue.first(), type(None))
                and variable.defaultValue.first() != "None"
                else 1e100
            )
            return lo, hi, 0.0001
    vrange = (
        vrange.replace("lo=", "").replace("hi=", "").replace("step=", "").split(",")
    )

    vrange = [r.strip() for r in vrange]
    if vrange == ["None"] or vrange == [""]:
        return None

    vrange[0] = (
        float(vrange[0].replace("=", "").replace("lo", "").strip())
        if vrange[0] != "None"
        else -1e100
    )
    vrange[1] = (
        float(vrange[1].replace("=", "").replace("hi", "").strip())
        if vrange[1] != "None"
        else 1e100
    )
    step = vrange[2].replace("=", "").replace("step", "").strip()
    step = float(step) if step != "None" else 1
    if return_array:
        return np.arange(vrange[0], vrange[1], step)
    else:
        return vrange[0], vrange[1], step


def find_best_fuzzy_match(target, cls_list) -> owlready2.ThingClass:
    """
    Find the best fuzzy match for a target string in a list of strings,
    prioritizing strings that start with the target followed by an underscore.

    Parameters:
        target (str): The target string to match.
        cls_list (list of str): The list of strings to search.

    Returns:
        str: The string from the list that best matches the target.
    """

    cls2str = {str(cls.label.first()): cls for cls in cls_list}
    string_list = cls2str.keys()
    # Filter strings that start with target followed by an underscore
    filtered_list = [s for s in string_list if s.startswith(target + "_")]

    # If filtered list is not empty, return the shortest string from it
    if filtered_list:
        return onto[min(filtered_list, key=len)]

    # If no specific match, use fuzzy matching
    best_match, _ = process.extractOne(target, string_list)
    return onto["best_match"]


# TODO: update docstrig with the new params
def find_variables(
    var, model, type="all", include_synonyms=False, find_best_match=True
) -> Optional[owlready2.ThingClass]:
    """
    Finds a variable in a TVB model.
    Parameters:
        var (str): The variable to find.
        model (owlready2.ThingClass or str): The TVB model to search in.
        type (str, optional): The type of variable to find. Default is "all".
        include_synonyms (bool, optional): If True, includes synonyms in the search. Default is False.
        find_best_match (bool, optional): If True, finds the best fuzzy match if multiple matches are found. Default is True.
    Returns:
        owlready2.ThingClass or None: The found variable class, or None if not found.
    """

    if isinstance(model, str):
        model = get_model(model)

    scls = model.descendants(include_self=False)
    potential_variables = list(onto.search(label=f"{var}*"))
    if include_synonyms:
        potential_variables += list(onto.search(synonym=f"{var}*"))
    # print(potential_variables)
    var_cls = intersection(scls, potential_variables)
    # print(var_cls)

    if type != "all":
        var_cls = intersection(var_cls, onto.search(label=f"{type}*").subclasses())

    if len(var_cls) == 0:
        return None
    elif len(var_cls) == 1:
        return var_cls[0]
    elif find_best_match:
        return find_best_fuzzy_match(var, var_cls)
    else:
        return var_cls


def get_all_annotations(prop) -> List[str]:
    proplist = []
    for c in onto.classes():
        if c.name == "Thing":
            continue
        proplist.extend(getattr(c, prop))
    return list(set(proplist))


def create_acronym(text) -> str:
    existing_acronyms = get_all_annotations("acronym")
    # Split the text into words based on uppercase letters
    words = re.findall(r"[A-Z][^A-Z]*", text)
    index = 1  # Start from the first letter

    # Initially create an acronym using the first letter of each word
    acronym = "".join(word[0] for word in words)

    # Keep adding letters until the acronym is unique
    while acronym in existing_acronyms:
        acronym = "".join(
            word[: index + 1] if len(word) > index else word for word in words
        )
        index += 1

    return acronym.upper()


####################
# Search Ontology  #
####################


def extract_most_common(searches) -> Optional[owlready2.ThingClass]:
    from collections import Counter

    # Flatten the list of lists
    flat_list = [item for sublist in searches for item in sublist]

    # Count the frequency of each item
    counter = Counter(flat_list)

    # Find the item with the highest frequency
    if len(counter) == 0:
        most_common_item = None
    else:
        most_common_item, _ = counter.most_common(1)[0]
    return most_common_item


def search_all(
    search_term, from_class=None, case_sensitive=False
) -> Optional[owlready2.ThingClass]:
    if from_class is None:
        tree = list(onto.classes())
    else:
        tree = from_class.descendants(include_self=False)

    labelsearch = intersection(
        onto.search(label=f"{search_term}*", _case_sensitive=case_sensitive),
        tree,
    )
    aliassearch = intersection(
        onto.search(synonym=f"{search_term}*", _case_sensitive=case_sensitive),
        tree,
    )
    symbolsearch = intersection(
        onto.search(symbol=f"{search_term}*", _case_sensitive=case_sensitive),
        tree,
    )
    return extract_most_common([labelsearch, aliassearch, symbolsearch])


####################
def import_model(model_metadata, model_name=None) -> owlready2.ThingClass:

    if isinstance(model_metadata, str) and isfile(model_metadata):
        model_metadata = parse.metadata.load_experiment_metadata(model_metadata)
        model_data = model_metadata.model
    elif isinstance(model_metadata, tvbo_datamodel.Dynamics):
        model_data = model_metadata
    elif isinstance(model_metadata, tvbo_datamodel.SimulationExperiment):
        model_data = model_metadata.model
    elif isinstance(model_metadata, dict):
        model_data = tvbo_datamodel.Dynamics(**model_metadata)

    if model_name is None:
        model_name = str(model_data.name)

    return parse.metadata.import_yaml_model(model_data, model_name)
