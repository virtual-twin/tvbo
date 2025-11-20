#  tvbgo.py
#
# Created on Mon Aug 07 2023
# Author: Leon K. Martin
#
# Copyright (c) 2023 Charité Universitätsmedizin Berlin
#
"""
TVB-GO Module
=============

This module provides utilities for working with TVB and the Gene Ontology.

.. moduleauthor:: Leon K. Martin

Attributes:
-----------
import json
import os
from os.path import abspath, dirname, join
from urllib.request import urlopen, urlretrieve
from urllib.request import urlretrieve
import goatools
import networkx as nx
import numpy as np
import pandas as pd
import wget
from goatools import obo_parser

Functions:
----------
"""
import json
import os
from os.path import abspath, dirname, join
from urllib.request import urlopen, urlretrieve

import networkx as nx
import numpy as np
import pandas as pd
import pybel
from goatools import obo_parser
import logging
import sys

from tvbo.knowledge import constants

ROOT_DIR = abspath(dirname(__file__))

TVBGO_DIR = join(constants.DATA_DIR, "tvb-go")

KEYWORD_DIR = join(TVBGO_DIR, "Tripletts_with_keywords")

df_tvbgo = pd.read_excel(join(TVBGO_DIR, "TVB-O_GO_list_curation_Julie_Courtiol.xlsx"))

kw2tvbo = pd.read_csv(
    join(KEYWORD_DIR, "Model_parameter_keywords_TVB3_2_final.csv"), sep=";"
)
go2kw = pd.read_csv(join(KEYWORD_DIR, "GO_terms_keywords_TVBO3_2_final.csv"), sep=";")

# Keyword cluster
df_clust = pd.read_excel(KEYWORD_DIR + "/Keywords_cluster.xlsx")

kw_clust = dict()
kw_clust["excitation"] = [
    i.lower().strip().replace('"', "") for i in df_clust["Cluster"].dropna().to_list()
]
kw_clust["inhibition"] = [
    i.lower().strip().replace('"', "")
    for i in df_clust["Unnamed: 3"].dropna().to_list()
]
kw_clust["e/i balance"] = [
    i.lower().strip().replace('"', "")
    for i in df_clust["Unnamed: 4"].dropna().to_list()
]
kw_clust["internal coupling"] = [
    i.lower().strip().replace('"', "")
    for i in df_clust["Unnamed: 5"].dropna().to_list()
]
kw_clust["time_scaling"] = [
    i.lower().strip().replace('"', "")
    for i in df_clust["Unnamed: 6"].dropna().to_list()
]


def bel2label(node):
    if isinstance(node, pybel.dsl.node_classes.ComplexAbundance) or isinstance(
        node, pybel.dsl.node_classes.CompositeAbundance
    ):
        complex = list()
        for n in node.members:
            complex.append(n.entity["namespace"] + ":" + n.entity["name"].strip())
        nx_node = " AND ".join(complex)

    elif isinstance(node, pybel.dsl.node_classes.Reaction):
        complex = list()
        for n in node.reactants:
            complex.append(n.entity["namespace"] + ":" + n.entity["name"].strip())
        for n in node.products:
            complex.append(n.entity["namespace"] + ":" + n.entity["name"].strip())

        nx_node = " AND ".join(complex)

    else:
        nx_node = node.entity["namespace"] + ":" + node.entity["name"].strip()

    return nx_node


logger = logging.getLogger(__name__)


def _download_with_progress(url: str, dest_path: str):
    """Download a file with a textual progress bar to stderr.

    Uses urllib.request.urlretrieve reporthook for progress updates.
    """
    bar_width = 40

    def _hook(count, block_size, total_size):
        if total_size <= 0:
            # Indeterminate total size; simple spinner-like percentage based on count
            written = count * block_size
            msg = f"Downloading... {written // 1024} KB"
        else:
            downloaded = min(count * block_size, total_size)
            frac = downloaded / float(total_size)
            filled = int(bar_width * frac)
            bar = "#" * filled + "-" * (bar_width - filled)
            pct = int(frac * 100)
            msg = f"[{bar}] {pct:3d}% ({downloaded // 1024} / {total_size // 1024} KB)"
        sys.stderr.write("\r" + msg)
        sys.stderr.flush()

    # Ensure previous line is cleared and newline after completion
    try:
        urlretrieve(url, dest_path, reporthook=_hook)
    finally:
        sys.stderr.write("\n")
        sys.stderr.flush()


def get_go():
    """
    Retrieve the Gene Ontology DAG (Directed Acyclic Graph) from the online repository.

    Returns
    -------
    go : GODag object
        The Gene Ontology DAG.
    """
    go_obo_url = "https://current.geneontology.org/ontology/go.obo"
    data_folder = join(TVBGO_DIR)

    # Ensure data folder exists (mkdir -p semantics)
    if os.path.isfile(data_folder):
        raise Exception(
            "Data path (" + data_folder + ") exists as a file. "
            "Please rename, remove or change the desired location of the data path."
        )
    os.makedirs(data_folder, exist_ok=True)

    # Check if the file exists already
    go_dest = data_folder + "/go.obo"
    if not os.path.isfile(go_dest):
        logger.info("Downloading Gene Ontology OBO from %s", go_obo_url)
        try:
            print("Downloading Gene Ontology OBO...")
            _download_with_progress(go_obo_url, go_dest)
        except Exception as e:
            logger.exception("Failed to download GO OBO: %s", e)
            raise
        logger.info("Saved GO OBO to %s", go_dest)
    else:
        logger.info("Using cached GO OBO at %s", go_dest)
    go_obo = go_dest

    go = obo_parser.GODag(go_obo)
    return go


go = get_go()


def go_term2id(term):
    """
    Convert a Gene Ontology term to its corresponding ID.

    Parameters
    ----------
    term : str
        The Gene Ontology term to convert.

    Returns
    -------
    str or None
        The corresponding GO ID or None if not found.
    """
    term = term.strip().replace("...", "").strip()
    for i in go.items():
        t = i[1]
        if t.name.lower() == term.lower():
            return t.id


def get_term(go_id):
    """
    This function retrieves the definition of a given Gene Ontology term,
    using EMBL-EBI's QuickGO browser.
    Input: go_id - a valid Gene Ontology ID, e.g. GO:0048527.

    Parameters
    ----------
    go_id : str
        A valid Gene Ontology ID, e.g., GO:0048527.

    Returns
    -------
    dict
        Term information.

    Raises
    ------
    ValueError
        If information from QuickGO is not retrievable.
    """
    quickgo_url = "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/" + go_id
    ret = urlopen(quickgo_url)

    # Check the response
    if ret.getcode() == 200:
        term = json.loads(ret.read())
        return term["results"][0]
    else:
        raise ValueError(
            "Couldn't receive information from QuickGO. Check GO ID and try again."
        )


def retrieve_go_id(term):
    """
    Search for a Gene Ontology term and retrieve its ID.

    Parameters
    ----------
    term : str
        The Gene Ontology term to search.

    Returns
    -------
    str or None
        The corresponding GO ID or None if not found.
    """
    # Search for the term in the Gene Ontology
    for go_id in go:
        go_term = go[go_id]
        if term.lower() in go_term.name.lower():
            return go_id


# TVB-GO Notebook


def rename2clust(kw):
    """
    Rename a keyword based on predefined clusters.

    Parameters
    ----------
    kw : str
        Keyword to rename.

    Returns
    -------
    str
        Renamed keyword.
    """
    for k, v in kw_clust.items():
        if kw.lower() in [i.lower().strip().replace('"', "") for i in v]:
            return k

    return kw


def rmv_extrachars(term):
    """
    Remove special characters from a term.

    Parameters
    ----------
    term : str
        The term from which special characters should be removed.

    Returns
    -------
    str
        Term without special characters.
    """
    return term.replace(":", "").replace('"', "").strip()


def add_node(n, g, type=None, **kwargs):
    """
    Add a node to a given graph.

    Parameters
    ----------
    n : node
        The node to be added.
    g : NetworkX graph
        The graph to which the node will be added.
    type : str, optional
        Type of the node.
    **kwargs
        Additional keyword arguments.
    """
    if n not in g.nodes:
        g.add_node(n, type=type, **kwargs)


def graph_from_table(df_tvbgo=df_tvbgo):
    """
    Generate a graph based on a given table.

    Parameters
    ----------
    df_tvbgo : DataFrame, optional
        Data table used to generate the graph. Defaults to the module-level df_tvbgo.

    Returns
    -------
    NetworkX graph
        Generated graph.
    """
    df_tvbgo = df_tvbgo[
        df_tvbgo["TVB-O_function_clustered"].isin(
            [
                "Excitation",
                "Inhibition",
                "E/I Balance",
                "Internal Coupling",
                "Time_Scaling",
            ]
        )
    ]

    g = nx.Graph()

    for i, r in df_tvbgo.iterrows():
        go_term = r["term"].lower()
        tvb_clust = r["TVB-O_function_clustered"].lower()
        param = r["TVB-Function_detailed"]
        # tvb_clust = rename2clust(tvb_clust)

        add_node(go_term, g, type="GO-term", **{"go_id": r["Go_id"]})
        add_node(tvb_clust, g, type="parameter_type")
        g.add_edge(go_term, tvb_clust, type="go2parameter_type")

        if not isinstance(param, type(np.nan)):
            param = param
            add_node(param, g, type="NMM_parameter")
            g.add_edge(param, go_term, type="go2parameter", linetype="--")

    kws = list()
    for i, r in go2kw.iterrows():
        go_term = r.Class.lower()
        kw = r.Keyword.lower()
        kw = rmv_extrachars(kw)
        kw = rename2clust(kw)
        kws.append(kw)

        add_node(go_term, g, type="GO-term")
        add_node(kw, g, type="parameter_type")
        g.add_edge(go_term, kw, type="go2kw")
        # g.add_edge(go_term, 'GO', type='go2kw')

    for i, r in kw2tvbo.iterrows():
        param = r.Class
        kw = r.Keyword.lower()
        kw = rmv_extrachars(kw)
        kw = rename2clust(kw)

        add_node(param, g, type="NMM_parameter")
        add_node(kw, g, type="keyword")
        g.add_edge(param, kw, type="tvbo2kw")

    g = nx.relabel_nodes(
        g,
        mapping={
            "excitation": "Excitation",
            "inhibition": "Inhibition",
            "e/i balance": "E/I Balance",
            "internal coupling": "Internal Coupling",
            "time_scaling": "Time Scaling",
        },
    )
    return g


def get_annotated_parameters():
    """
    Retrieve parameters that have been annotated.

    Returns
    -------
    list
        List of annotated parameters.
    """
    g = graph_from_table()

    annotated_params = list()
    for n, t in nx.get_node_attributes(g, "type").items():
        if t == "NMM_parameter":
            annotated_params.append(n)

    return annotated_params
