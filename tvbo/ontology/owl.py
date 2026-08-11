# Copyright © 2023 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""
---
title: "Ontology Module for TVB-O" author: Leon Martin
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
from tvbo.ontology import owl as ontology
jansen_rit_model = ontology.get_model("JansenRit")
```

Author:
    Leon K. Martin (2023)

Copyright:
    Copyright (c) 2023 Charité Universitätsmedizin Berlin
"""

import collections
import functools
import os
import re
import tempfile
from os.path import abspath, dirname, isfile, join, realpath
from textwrap import wrap
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import owlready2
import pandas as pd
from owlready2 import default_world, get_ontology, set_render_func
from tvbo.utils import Bunch

try:
    import simple_colors as sc
    from fuzzywuzzy import process as fuzz_process
except ImportError:
    sc = None
    fuzz_process = None


def _require_fuzzy():
    if fuzz_process is None:
        raise ImportError(
            "Fuzzy matching requires knowledge extras. Install with:\n"
            "  pip install tvbo[knowledge]\n"
            "Or: pip install fuzzywuzzy python-Levenshtein"
        )


from tvbo.ontology import query as _query_mod
from tvbo.datamodel import schema as tvbo_datamodel

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


@functools.cache
def _load_ontology():
    """Parse the TVB-O ontology once and return it (memoised for the process).

    The parse is deferred to first real use, behind the lazy proxy below. The ontology is metadata only — consulted to fill in a missing specification or build a model, never needed to generate or run code — but parsing it at import made every `import tvbo` load the `.owl` file through the class modules that import this one. That is expensive, and it collided with JAX's GC callback badly enough to crash the kernel.

    The public surface is unchanged: `onto` still behaves like the loaded ontology for attribute access, item access, iteration and `with onto:`, `get_onto()` returns it, and `iri` and `namespace` stay importable module attributes, resolved lazily through PEP 562.
    """
    with open(join(ONTO_DIR, "tvb-o.owl"), "r", encoding="utf-8") as f:
        xml = f.read()
    # Drop the remote NIF-Ontology import so the load stays offline.
    xml = xml.replace(
        '<owl:imports rdf:resource="https://raw.githubusercontent.com/SciCrunch/NIF-Ontology/atlas/ttl/atom.ttl"/>',
        "",
    )
    with tempfile.NamedTemporaryFile(suffix=".owl", delete=False, mode="w", encoding="utf-8") as tmp:
        tmp.write(xml)
        tmp_path = tmp.name
    loaded_ontology = get_ontology("file://" + tmp_path).load()
    loaded_ontology.load()  # TODO: check if the redundant reload can be removed
    return loaded_ontology


class _LazyOntologyProxy:
    """Stand-in for the TVB-O ontology that loads it on first use.

    Keeps the ``onto`` API intact — attribute access (``onto.JansenRit``), item access (``onto[iri]``), iteration and ``with onto:`` all forward to the real ontology and
    trigger the parse only when first touched.
    """

    __slots__ = ()

    def __getattr__(self, name):
        return getattr(_load_ontology(), name)

    def __getitem__(self, key):
        return _load_ontology()[key]

    def __iter__(self):
        return iter(_load_ontology())

    def __enter__(self):
        return _load_ontology().__enter__()

    def __exit__(self, *exc):
        return _load_ontology().__exit__(*exc)

    def __repr__(self):
        is_loaded = _load_ontology.cache_info().currsize > 0
        return f"<TVB-O ontology (lazy proxy; loaded={is_loaded})>"


onto = _LazyOntologyProxy()


def __getattr__(name):  # PEP 562: resolve ontology-derived module attributes on demand.
    if name == "iri":
        return _load_ontology().base_iri
    if name == "namespace":
        loaded_ontology = _load_ontology()
        return loaded_ontology.get_namespace(loaded_ontology.base_iri)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


df_tvbo = pd.read_csv(join(DATA_DIR, "_tvb-o.csv"), sep=";")


# %% global functions


def get_onto() -> owlready2.namespace.Ontology:
    """Return the loaded TVB-O ontology object.

    Returns:
        The loaded `owlready2` ontology backing this module.
    """
    return _load_ontology()


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
    Creates a dictionary from a list of ontology classes. The dictionary's keys are the class labels and its values are the class objects. The dictionary is sorted alphabetically based on its keys.

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


# %% Functions for extracting TVB-O variables. An NMM name must match the model's label in TVBO.
def wrap_text(text, line_length=100, line_breaks="\n") -> str:
    """
    Pretty print a string with automatic line breaks at specified intervals, while preserving existing new lines.

    Parameters:
        text (str): The text to be printed.
        line_length (int): The maximum length of each line.
    """

    def wrap_line(line):
        """Wrap a single line to `line_length`, inserting `line_breaks` between words."""
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
    """Build a formatted text summary of an ontology class with its definition and references.

    Args:
        cls: The ontology class, or its label string to look up in the ontology.

    Returns:
        A multi-line string with the class label as a heading, its wrapped
        definition, and a formatted references section when references exist.
    """
    from tvbo.utils.report import render_citation

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
    """Summarize the contents of the TVB-O ontology as component counts.

    Args:
        print_info: If True, print the base IRI and a count of each component group.
        return_info: If True, return the collected components as a `Bunch`.
        return_df: If True, return the component counts as a `pandas.DataFrame`.

    Returns:
        A `Bunch` of component lists when `return_info` is set, or a count
        `DataFrame` indexed by component name when `return_df` is set; otherwise
        nothing is returned.
    """
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
        print("TVBO base IRI:", onto.base_iri)
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
    """Search a model's descendant classes by label or definition for a search string.

    Args:
        search_str: The text to search for.
        model: The TVB model to search within, or its label string.
        wildcards: If True, wrap `search_str` in `*` wildcards; otherwise append
            the model's suffix to the search string.

    Returns:
        The single matching class when exactly one is found, `None` when none
        match, or the list of matching classes otherwise.
    """
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
        onto.IntegrationMethod,
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


def get_model(label: str = "JansenRit", model_type="NMM", verbose=False) -> owlready2.ThingClass:
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
            print(f"Model {label} not found in {model_type} models.\n Valid models are {sorted(models.keys())}")
        return onto.NeuralMassModel()  # return empty NMM class
    default_world.full_text_search_properties.append(NMM)
    return NMM


def get_integrator(integration_method="Heun") -> owlready2.ThingClass:
    """Retrieve the ontology class for a named integration method.

    Args:
        integration_method: The label of the integration method to look up.

    Returns:
        The matching integration-method class, or `None` if none is found. When
        several match, the first is returned (with a message printed).
    """
    search_res = _query_mod.label_search(integration_method)

    available_integrators = onto.IntegrationMethod.descendants(include_self=False)

    av_int = intersection(search_res, available_integrators)

    if len(av_int) == 0:
        return None
    if len(av_int) > 1:
        print("Multiple integrators found for method: ", integration_method)
        print("Available integrators: ", av_int)
        print("Using the first one: ", av_int[0])
    return av_int[0]


def get_coupling_functions() -> Dict[str, owlready2.ThingClass]:
    """Return all coupling-function classes keyed by their label.

    Returns:
        A dictionary mapping each coupling function's label to its ontology class.
    """
    return {CF.label.first(): CF for CF in onto.Coupling.subclasses()}


def get_coupling_function(label="Linear", verbose=True) -> Optional[owlready2.ThingClass]:
    """Retrieve a coupling-function class by its label or a synonym.

    Args:
        label: The label (or synonym) of the coupling function to retrieve.
        verbose: If True, print the valid coupling functions when the label is
            not found.

    Returns:
        The matching coupling-function class, or `None` if it is not found.
    """
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
            print(f"Coupling function {label} not found.\nValid coupling functions are: {coupling_functions.keys()}")
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
    """Strip the model-specific suffix from an ontology class's label.

    The suffix is derived from the acronyms of the class's neural-mass-model, coupling, or data-type ancestors.

    Args:
        cls: The ontology class, or its label string to look up in the ontology.

    Returns:
        The class label with any matching ancestor suffix removed. The original
        label (or string) is returned when no suffix matches.
    """
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
                PREFIX tvb-o: <https://w3id.org/tvbo/>
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


def get_model_parameters(NMM, return_as_dict=True) -> Union[Dict[str, owlready2.ThingClass], List[owlready2.ThingClass]]:
    """
    Retrieves the parameters for a given TVB model.

    Parameters:
        NMM (owlready2.ThingClass or str): The TVB model to retrieve the parameters for.

    Returns:
        dict: A dictionary of parameter labels and their corresponding ontology class objects.
    """
    if isinstance(NMM, str):
        NMM = get_model(NMM)
    get_model_suffix(NMM)
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

    return {replace_suffix(p).replace(f"_{NMM.name}", ""): p for k, p in parameters.items()}


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


def get_model_constants(NMM, return_as_dict=True) -> Union[Dict[str, owlready2.ThingClass], List[owlready2.ThingClass]]:
    """Retrieve the constants for a given TVB model.

    Args:
        NMM: The TVB model to retrieve the constants for, or its label string.
        return_as_dict: If True, sort and key the constants by their label before
            suffix stripping.

    Returns:
        A dictionary mapping each suffix-stripped constant label to its ontology
        class object.
    """
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
    """Retrieve the conditional derived variables for a given TVB model.

    Args:
        NMM: The TVB model to retrieve the conditionals for, or its label string.

    Returns:
        A dictionary mapping each suffix-stripped conditional label to its
        ontology class object.
    """
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


def get_model_derivatives(NMM, return_as_dict=True) -> Union[Dict[str, owlready2.ThingClass], List[owlready2.ThingClass]]:
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


def get_model_statevariables(NMM, return_as_dict=True) -> Union[Dict[str, owlready2.ThingClass], List[owlready2.ThingClass]]:
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


def get_model_cvars(NMM, return_as_dict=True) -> Union[Dict[str, owlready2.ThingClass], List[owlready2.ThingClass]]:
    """
    Retrieves the cvars (coupling variables) for a given TVB model.

    Parameters:
        NMM (owlready2.ThingClass or str): The TVB model to retrieve the cvars for.

    Returns:
        list: The cvars for the given TVB model."""
    if isinstance(NMM, str):
        NMM = get_model(NMM)
    cvars = NMM.has_cvar
    # A state variable is a coupling variable if its derivative consumes a global coupling term.  Match the model's actual coupling-term names rather than a hard-coded prefix, so any naming (c_glob, c_pop, …) works.
    global_coupling_names = [c for c in get_model_coupling_terms(NMM, return_as_dict=True).keys() if c != "local_coupling"]
    for k, v in get_model_derivatives(NMM).items():
        rhs = v.value.first() or ""
        if any(cn in rhs for cn in global_coupling_names):
            for isa in v.is_a:
                if onto.StateVariable in isa.is_a:
                    cvars.append(isa)
    if len(cvars) == 0:
        cvars = get_model_statevariables(NMM).values()
    if return_as_dict:
        cvars = get_sorted_dict(cvars)
    if NMM == onto.JansenRit:
        cvars.pop("y4_JR", None)
    return {replace_suffix(k): p for k, p in cvars.items()}


def get_default_values(NMM, tvb_name=False, class_as_key=False) -> Dict[str, Union[float, bool, int]]:
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
    get_model_suffix(NMM)
    values = dict()
    parameters = get_model_parameters(NMM)
    parameters.update(get_model_constants(NMM))
    # Default to zero for single-node evaluation; names derived from the model so any naming works.
    coupling_input_names = list(get_model_coupling_terms(NMM, return_as_dict=True).keys())
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
            for cn in coupling_input_names:
                values[cn] = 0

    return values


def contains_math_char(s) -> bool:
    """Check whether a string contains a mathematical operator character.

    Args:
        s: The string to inspect.

    Returns:
        True if `s` contains any of `+`, `-`, `*`, `/`, `=`, or `^`.
    """
    math_chars = ["+", "-", "*", "/", "=", "^"]
    return any(char in s for char in math_chars)


def add_spaces_around_math_chars(s) -> str:
    """Insert spaces around mathematical operator characters in a string.

    Args:
        s: The string to reformat.

    Returns:
        The string with a space added on each side of every `+`, `-`, `*`, `/`,
        or `=` that is not already surrounded by whitespace.
    """
    pattern = r"(?<!\s)([\+\-\*/=])(?!\s)"

    def repl(match):
        """Return the matched operator padded with a space on each side."""
        return f" {match.group(1)} "

    # Perform the substitution
    return re.sub(pattern, repl, s)


def get_model_vois(model) -> Tuple[str]:
    """Retrieve the variables of interest (VOIs) for a given TVB model.

    Combines the model's default VOIs with any extra VOIs listed on the model, spacing out operator-based expressions.

    Args:
        model: The TVB model to retrieve the VOIs for, or its label string.

    Returns:
        A sorted tuple of unique VOI names, falling back to the model's state
        variables when no VOIs are defined.
    """
    if isinstance(model, str):
        model = get_model(model)
    suffix = get_model_suffix(model)

    relations = {m.label.first().replace(suffix, "") for m in model.has_default_voi if m.name != "Thing"}
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
    return tuple(sorted(set([v.replace('"', "").replace("'", "").strip() for v in vois])))


def get_definition(tvbo_class) -> str:
    """
    Retrieves the definition for a given ontology class.

    Parameters:
        tvbo_class (owlready2.ThingClass or str): The ontology class to retrieve the definition for.

    Returns:
        str: The definition for the given ontology class."""

    return "\n".join(wrap(tvbo_class.definition[0], width=100))


def get_parameters_by_catalogue(NMM: owlready2.ThingClass, param_key: str) -> pd.DataFrame:
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


# Class Properties #
def get_object_properties(ontology_class, include_restriction=True):
    """Collect the object-property relationships of an ontology class.

    Args:
        ontology_class: The ontology class to inspect.
        include_restriction: If True, include `owl:someValuesFrom`-style
            restriction relationships.

    Returns:
        A list of single-key dictionaries mapping each property name (or
        `"is_a"`) to its target value.
    """
    object_properties = []
    for p, o in _query_mod.get_class_relationships(ontology_class):
        if isinstance(o, owlready2.class_construct.Restriction):
            if include_restriction:
                if {o.property.name: o.value} not in object_properties:
                    object_properties.append({o.property.name: o.value})

        elif isinstance(o, owlready2.ThingClass):
            if {"is_a": o} not in object_properties:
                object_properties.append({"is_a": o})

    return object_properties


def get_class_properties(cls):
    """Collect the label, identifier, annotation, and object properties of a class.

    Args:
        cls: The ontology class, or its label string to look up in the ontology.

    Returns:
        A dictionary with the class's `label`, `identifier`,
        `annotation_properties`, and `object_properties`.
    """
    if isinstance(cls, str):
        cls = _query_mod.search_by_label(str(cls))[0]

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
                except Exception:
                    pass
            prop["annotation_properties"][annot_prop.python_name] = val

    prop["object_properties"] = get_object_properties(cls)
    return prop


def get_class_annotation_properties(cls):
    """Return the annotation properties of an ontology class.

    Args:
        cls: The ontology class, or its label string to look up in the ontology.

    Returns:
        The `annotation_properties` mapping for the class.
    """
    return get_class_properties(cls)["annotation_properties"]


def get_class_object_properties(cls):
    """Return the object properties of an ontology class.

    Args:
        cls: The ontology class, or its label string to look up in the ontology.

    Returns:
        The list of object-property relationships for the class.
    """
    return get_class_properties(cls)["object_properties"]


def get_class_data_properties(cls):
    """Return the data properties of an ontology class.

    Args:
        cls: The ontology class, or its label string to look up in the ontology.

    Returns:
        The `data_properties` entry for the class.
    """
    return get_class_properties(cls)["data_properties"]


# Model Comparison #
def join_set(a):
    """Join the unique elements of an iterable into a comma-separated string.

    Args:
        a: The iterable of strings whose distinct values are joined.

    Returns:
        A comma-separated string of the distinct elements.
    """
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

                inters = [i for i in inters if i in onto.ModelParametersCatalogue.descendants(include_self=False)]
                if onto.StateVariable in inters:
                    inters.remove(onto.StateVariable)

                if onto.TransferSigmoidFunctionProperties in inters:
                    inters.remove(onto.TransferSigmoidFunctionProperties)

                if len(inters) > 0:
                    df_comp.at[i, model1.label.first()] = k1.label.first().replace(model1_suffix, "")
                    df_comp.at[i, model2.label.first()] = k2.label.first().replace(model2_suffix, "")
                    df_comp.at[i, "Parameter Catalogue"] = ", ".join([i.label.first() for i in inters])
                    i += 1
        df_comp = df_comp.groupby("Parameter Catalogue", as_index=False).agg(
            {model1.label.first(): join_set, model2.label.first(): join_set}
        )

    return df_comp


# Parameter Properties #


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
        return tuple(val.strip() for val in vrange.replace("lo=", "").replace("hi=", "").replace("step=", "").split(","))
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
                if not isinstance(variable.defaultValue.first(), type(None)) and variable.defaultValue.first() != "None"
                else 1e-100
            )
            hi = (
                variable.defaultValue.first()
                if not isinstance(variable.defaultValue.first(), type(None)) and variable.defaultValue.first() != "None"
                else 1e100
            )
            return lo, hi, 0.0001
    vrange = vrange.replace("lo=", "").replace("hi=", "").replace("step=", "").split(",")

    vrange = [r.strip() for r in vrange]
    if vrange == ["None"] or vrange == [""]:
        return None

    vrange[0] = float(vrange[0].replace("=", "").replace("lo", "").strip()) if vrange[0] != "None" else -1e100
    vrange[1] = float(vrange[1].replace("=", "").replace("hi", "").strip()) if vrange[1] != "None" else 1e100
    step = vrange[2].replace("=", "").replace("step", "").strip()
    step = float(step) if step != "None" else 1
    if return_array:
        return np.arange(vrange[0], vrange[1], step)
    else:
        return vrange[0], vrange[1], step


def find_best_fuzzy_match(target, cls_list) -> owlready2.ThingClass:
    """
    Find the best fuzzy match for a target string in a list of strings, prioritizing strings that start with the target followed by an underscore.

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
    _require_fuzzy()
    best_match, _ = fuzz_process.extractOne(target, string_list)
    return onto["best_match"]


# TODO: update docstrig with the new params
def find_variables(var, model, type="all", include_synonyms=False, find_best_match=True) -> Optional[owlready2.ThingClass]:
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
    var_cls = intersection(scls, potential_variables)

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
    """Collect all distinct values of an annotation property across every class.

    Args:
        prop: The name of the annotation property to gather.

    Returns:
        A list of the unique annotation values found across all ontology classes.
    """
    proplist = []
    for c in onto.classes():
        if c.name == "Thing":
            continue
        proplist.extend(getattr(c, prop))
    return list(set(proplist))


def create_acronym(text) -> str:
    """Generate a unique upper-case acronym from a camel-case name.

    Letters are taken from each capitalised word, extending the acronym until it no longer collides with an existing acronym in the ontology.

    Args:
        text: The camel-case text (e.g. a model name) to build an acronym from.

    Returns:
        An upper-case acronym that is unique among existing ontology acronyms.
    """
    existing_acronyms = get_all_annotations("acronym")
    # Split the text into words based on uppercase letters
    words = re.findall(r"[A-Z][^A-Z]*", text)
    index = 1  # Start from the first letter

    # Initially create an acronym using the first letter of each word
    acronym = "".join(word[0] for word in words)

    # Keep adding letters until the acronym is unique
    while acronym in existing_acronyms:
        acronym = "".join(word[: index + 1] if len(word) > index else word for word in words)
        index += 1

    return acronym.upper()


# Search Ontology  #


def extract_most_common(searches) -> Optional[owlready2.ThingClass]:
    """Return the most frequently occurring item across several result lists.

    Args:
        searches: An iterable of result lists to flatten and tally.

    Returns:
        The single most common item, or `None` if the combined results are empty.
    """
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


def search_all(search_term, from_class=None, case_sensitive=False) -> Optional[owlready2.ThingClass]:
    """Search the ontology by label, synonym, and symbol and return the best match.

    Args:
        search_term: The prefix to search for.
        from_class: Restrict the search to this class's descendants; if `None`,
            search all ontology classes.
        case_sensitive: If True, perform a case-sensitive search.

    Returns:
        The class matching across the most search dimensions, or `None` if
        nothing matches.
    """
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


def import_model(
    model_metadata: Union[str, dict, "tvbo_datamodel.Dynamics", "tvbo_datamodel.SimulationExperiment"],
    model_name: Optional[str] = None,
) -> owlready2.ThingClass:
    """Import a model from metadata into the ontology.

    Creates ontology subclasses for state variables, parameters, derived variables, and output transforms.

    Args:
        model_metadata: A Dynamics, SimulationExperiment, dict, or path to a YAML file.
        model_name: Name to use for the created ontology class.

    Returns:
        owlready2.entity.ThingClass: The created ontology model class.
    """
    from tvbo.classes import equation as equations

    model_data = None
    if isinstance(model_metadata, str) and isfile(model_metadata):
        from linkml_runtime.loaders import yaml_loader

        experiment_metadata = yaml_loader.load(model_metadata, tvbo_datamodel.SimulationExperiment)
        model_data = experiment_metadata.model
    elif isinstance(model_metadata, tvbo_datamodel.Dynamics):
        model_data = model_metadata
    elif isinstance(model_metadata, tvbo_datamodel.SimulationExperiment):
        model_data = model_metadata.model
    elif isinstance(model_metadata, dict):
        model_data = tvbo_datamodel.Dynamics(**model_metadata)

    if model_name is None:
        model_name = str(model_data.name)

    if ontoclass := onto.search_one(label=model_name):
        return ontoclass

    acr = create_acronym(model_name)
    model_suffix = f"_{acr}"

    def _to_native(val):
        """Convert linkml extended types to native Python types for owlready2."""
        if val is None:
            return None
        if type(val).__name__ == "extended_str":
            return str(val)
        if type(val).__name__ == "extended_int":
            return int(val)
        if type(val).__name__ == "extended_float":
            return float(val)
        if isinstance(val, (list, tuple)):
            return type(val)(_to_native(v) for v in val)
        if isinstance(val, dict):
            return {_to_native(k): _to_native(v) for k, v in val.items()}
        return val

    def _create_subclass(name, base_class, properties, parent_class):
        with onto:
            new_class = type(name, (parent_class, base_class), {})
            for prop_name, prop_value in properties.items():
                if prop_value is not None:
                    native_value = _to_native(prop_value)
                    getattr(new_class, prop_name).append(native_value)
            return new_class

    # Create the main model class in the ontology
    with onto:
        model_class = type(
            model_name,
            (onto.NeuralMassModel,),
            {
                "label": model_name,
                "definition": model_name,
                "acronym": acr,
            },
        )

    # State variables
    for sv in model_data.state_variables.values():
        properties = {
            "label": sv.name + model_suffix,
            "symbol": str(sv.name),
            "stateVariableRange": (f"lo={sv.domain.lo}, hi={sv.domain.hi}" if sv.domain else ""),
        }
        # A clamped domain (enforce='clamp') is the modern equivalent of the former dedicated boundaries slot; export it as stateVariableBoundaries.
        from tvbo.utils import domain_enforcement

        _dom = sv.domain
        if _dom and domain_enforcement(_dom) == "clamp":
            properties["stateVariableBoundaries"] = f"lo={_dom.lo}, hi={_dom.hi}"

        sv_class = _create_subclass(sv.name + model_suffix, onto.StateVariable, properties, model_class)
        if sv.coupling_variable:
            model_class.has_cvar.append(sv_class)

        if sv.equation.rhs:
            td_name = sv.name + "_dot" + model_suffix
            td_class = _create_subclass(
                td_name,
                onto.TimeDerivative,
                {
                    "label": td_name,
                    "value": str(sv.equation.rhs),
                    "symbol": str(sv.equation.lhs) if sv.equation.lhs else str(sv.name),
                },
                sv_class,
            )
            # Link the state variable to its time derivative so consumers that read sv.has_derivative (e.g. class2metadata / from_ontology) work.
            with onto:
                sv_class.has_derivative.append(td_class)

        with onto:
            model_class.has_state_variable.append(sv_class)

    # Parameters
    from tvbo.utils import is_array_valued

    for k, p in model_data.parameters.items():
        properties = {
            "label": k + model_suffix,
            "symbol": getattr(p, "symbol", str(k)),
            "definition": str(p.description),
            # Array-valued constants (list/tuple/ndarray) have no scalar default — fall back to p.default rather than float()-ing the array.
            "defaultValue": (p.default if is_array_valued(p.value) else float(p.value)),
            "range": (f"lo={p.domain.lo}, hi={p.domain.hi}, step={p.domain.step}" if p.domain else ""),
        }
        p_class = _create_subclass(k + model_suffix, onto.Parameter, properties, model_class)
        model_class.has_parameter.append(p_class)

    # Derived parameters
    for dp in model_data.derived_parameters.values():
        properties = {
            "label": dp.name + model_suffix,
            "equation": str(dp.equation.rhs),
            "value": str(dp.equation.rhs),
            "symbol": str(dp.equation.lhs if dp.equation.lhs else dp.name),
        }
        _create_subclass(dp.name + model_suffix, onto.Function, properties, model_class)

    # Derived variables
    for dv in model_data.derived_variables.values():
        properties = {
            "label": dv.name + model_suffix,
            "equation": str(dv.equation.rhs),
            "value": str(dv.equation.rhs),
            "symbol": str(dv.equation.lhs if dv.equation.lhs else dv.name),
        }
        _create_subclass(dv.name + model_suffix, onto.Function, properties, model_class)

    # Outputs
    output_items = []
    if isinstance(model_data.output, list):
        output_items = [(name, model_data.derived_variables.get(name)) for name in model_data.output]
    elif isinstance(model_data.output, dict):
        output_items = list(model_data.output.items())

    for ot_name, ot in output_items:
        if ot is None:
            continue
        if not isinstance(ot_name, str):
            ot_name = str(ot.name) if hasattr(ot, "name") and ot.name else "output"
        if hasattr(ot, "equation") and ot.equation and ot.equation.rhs:
            eq_rhs = str(ot.equation.rhs)
            eq_lhs = str(ot.equation.lhs) if ot.equation.lhs else ot_name
        else:
            eq_rhs = ot_name
            eq_lhs = ot_name
        _create_subclass(
            ot_name + model_suffix,
            onto.Function,
            properties={
                "label": ot_name + model_suffix,
                "equation": eq_rhs,
                "value": eq_rhs,
                "symbol": eq_lhs,
            },
            parent_class=model_class,
        )

    # Coupling terms
    for k, cterm in model_data.coupling_terms.items():
        c_class = _create_subclass(
            str(k),
            onto.CouplingTerm,
            {"label": str(k)},
            model_class,
        )
        c_class.is_a.append(onto.GlobalConnectivity)

    equations.update_mathematical_relationships(model_class)

    if references := model_data.has_reference:
        try:
            references = eval(references)
        except Exception:
            if isinstance(references, str):
                references = [references]
        with onto:
            for ref in references:
                if isinstance(ref, str):
                    ref = onto.search_one(label=str(ref))
                if ref is not None:
                    model_class.has_reference.append(ref)

    return model_class
