#  nlp.py
#
# Created on Mon Aug 07 2023
# Author: Leon K. Martin
#
# Copyright (c) 2023 Charité Universitätsmedizin Berlin
#
"""
Natural Language Processing with TVB-O.
=======================================

## Example

```python
# Provide the path to the PDF file you want to parse
pdf_file_path = "/Users/leonmartin_bih/Downloads/Deco2014.pdf"
pdf_file_path = "/Users/leonmartin_bih/Downloads/fncom-13-00054 (1).pdf"

#Call the function to extract text from the PDF file
extracted_text = extract_text_from_pdf(pdf_file_path)
methods = extract_methods_from_pdf(pdf_file_path)
```
"""

import re
from os.path import abspath, dirname

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import PyPDF2
import spacy
from spacy.cli import download
# from transformers import BertForTokenClassification, BertTokenizer

from tvbo.knowledge import ontology, graph

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")

import en_core_web_sm

ROOT = dirname(abspath(__file__))

onto = ontology.onto

np.random.seed(1312)


def extract_text_from_pdf(file_path):
    """
    Extract text content from a PDF file.

    Parameters
    ----------
    file_path : str
        Path to the PDF file.

    Returns
    -------
    str
        Extracted text from the PDF. Hyphens at line breaks ("-\\n") are replaced with an empty string.
    """
    # Open the PDF file in binary mode
    with open(file_path, "rb") as file:
        # Create a PDF reader object
        reader = PyPDF2.PdfReader(file)

        # Initialize an empty string to store the extracted text
        text = ""

        # Iterate over each page in the PDF
        for page in reader.pages:
            # Extract the text from the current page
            text += page.extract_text()

        # Return the extracted text
        return text.replace("-\n", "")


def get_pdf_page(pdf_path, pagenum):
    """
    Extract text content from a specific page of a PDF file.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    pagenum : int
        Page number (0-indexed) to extract text from.

    Returns
    -------
    str
        Extracted text from the specified PDF page. Hyphens at line breaks ("-\\n") are replaced with an empty string.
    """
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        page = reader.pages[pagenum]
        return page.extract_text()


def extract_methods_from_pdf(pdf_path):
    """
    Extract the "Methods" or "Materials and Methods" section from a PDF.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.

    Returns
    -------
    str
        Extracted text of the methods section. Hyphens at line breaks ("-\\n") are replaced with an empty string.

    Notes
    -----
    This function searches for the "Methods" or "Materials and Methods" section in the PDF and extracts the text until
    it encounters the "Results" or "Discussion" section.
    """

    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)

        methods_section = ""
        is_methods_section = False

        for page_number in range(num_pages):
            page = reader.pages[page_number]
            text = page.extract_text()

            # Check if the page contains the methods section header or keywords
            if "Methods" in text or "Materials and Methods" in text:
                is_methods_section = True

            # Append the page's text to the methods_section variable if it's within the methods section
            if is_methods_section:
                methods_section += text

            # Stop appending if the end of the methods section is reached
            if "Results" in text or "Discussion" in text:
                is_methods_section = False

        return methods_section.replace("-\n", "")


def find_in_text(text, keyword, standalone=True, equation=False, **kwargs):
    """
    Find occurrences of a keyword in the text using regex.

    Parameters
    ----------
    text : str
        Input text to search within.
    keyword : str
        Keyword to search for.
    standalone : bool, optional (default is True)
        If True, only matches the keyword as a standalone word. Otherwise, matches all occurrences.
    equation : bool, optional (default is False)
        If True, only matches the keyword in sentences containing an equal sign.
    **kwargs
        Additional keyword arguments for `re.finditer`.

    Returns
    -------
    list of tuple
        List of (start, end) indices for each match in the text.
    """
    if standalone:
        pattern = r"\b" + re.escape(keyword) + r"\b"
    else:
        pattern = re.escape(keyword)

    occurrences = []
    if equation:
        # Split the text into sentences
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)

        for s in sentences:
            if "=" in s:
                for match in re.finditer(pattern, s, **kwargs):
                    # Adjust the start and end indices relative to the original text
                    start = match.start() + text.find(s)
                    end = match.end() + text.find(s)
                    occurrences.append((start, end))
    else:
        for match in re.finditer(pattern, text, **kwargs):
            occurrences.append((match.start(), match.end()))

    return occurrences


def ner_by_words(text):
    """
    Named Entity Recognition (NER) by matching terms against an ontology.

    Parameters
    ----------
    text : str
        Input text for NER.

    Returns
    -------
    dict
        Dictionary with ontology classes as keys. Values are dictionaries with named entities as keys and their occurrences as values.
    """
    by_word_res = dict()

    # Load the English model of spaCy
    nlp = en_core_web_sm.load()

    # Split the text into sentences
    sentences = text.split(".")
    for i, s in enumerate(sentences):
        if len(s) < 10:
            continue

        # Process the sentence using spaCy
        doc = nlp(s.replace("-", "::"))

        # Iterate through the terms in the processed sentence
        for w in doc:
            term = w.text.replace("::", "-")

            # Skip terms that are '*' or have less than two characters
            if term == "*" or len(term) < 2:
                continue

            typ = w.pos_

            # Iterate through different annotations (label, synonym, acronym, fullName, symbol)
            for annot in ("label", "synonym", "acronym", "fullName", "symbol"):
                # Search for the term in the ontology based on the current annotation
                if annot == "label":
                    res = onto.search(label=term)
                elif annot == "synonym":
                    res = onto.search(synonym=term)
                elif annot == "acronym":
                    res = onto.search(acronym=term)
                elif annot == "fullName":
                    res = onto.search(fullName=term)
                elif annot == "symbol":
                    res = onto.search(symbol=term)

                # If a match is found in the ontology, add the term to the result dictionary
                if len(res) > 0:
                    cl = res[0]
                    if cl not in by_word_res:
                        by_word_res[cl] = {term: find_in_text(text, term)}
                    else:
                        by_word_res[cl].update({term: find_in_text(text, term)})

    return by_word_res


def ner_by_classes(text, semantic_type="all"):
    """
    Named Entity Recognition (NER) based on ontology classes.

    Parameters
    ----------
    text : str
        Input text for NER.
    semantic_type : str, optional (default is "all")
        Semantic type for filtering: "all", "label", "synonym", "acronym", or "symbol".

    Returns
    -------
    dict
        NER results based on the specified semantic type.

    Raises:
        ValueError: If an incorrect `semantic_type` is specified.

    Example:
        >>> text = "The Jansen-Rit neural mass model is accociated with the alpha frequency in the EEG."
        >>> result = ner_by_classes(text, semantic_type="all")
        >>> print(result)
        {Neural Mass Model: [(15, 32)], EEG: [(79, 82)], Model: [(27, 32)], JansenRit: {'Jansen-Rit': [(4, 14)]}}
    """

    ner_label = dict()
    ner_syn = dict()
    ner_acr = dict()
    ner_sym = dict()

    for cl in onto.classes():
        if cl.name == "Thing":
            continue
        label = cl.label.first()

        # Do not ignore case with classes that are only one letter
        if len(label) > 1:
            by_label = find_in_text(text, label, **{"flags": re.IGNORECASE})
        else:
            by_label = find_in_text(text, label)
        if len(by_label) > 0:
            ner_label[cl] = by_label

        for syn in cl.synonym:
            if len(syn) > 1:
                by_syn = find_in_text(text, syn, **{"flags": re.IGNORECASE})
            else:
                by_syn = find_in_text(text, syn)

            if len(by_syn) > 0:
                if cl in ner_syn.keys():
                    ner_syn[cl] += by_syn  # = {syn: by_syn}
                else:
                    ner_syn[cl] = by_syn

        for acr in cl.acronym:
            if not acr.isupper():
                continue
            by_acr = find_in_text(text, acr)
            if len(by_acr) > 0:
                if cl in ner_acr.keys():
                    ner_acr[cl] += by_acr  # = {acr: by_acr}
                else:
                    ner_acr[cl] = by_acr

    keys = ner_label.copy()
    # keys.update(ner_acr)
    keys.update(ner_syn)
    keys = keys.keys()

    NMM = None
    for k in keys:
        if ontology.onto.search_one(acronym="NMM") in k.is_a:
            NMM = k
            print(NMM)

    if not isinstance(NMM, type(None)):
        for cl in onto.classes():
            if cl.name == "Thing":
                continue
            # if ontology.onto.Parameter not in cl.is_a:
            #     continue

            if NMM in cl.is_a:
                for sym in cl.symbol:
                    by_sym = find_in_text(text, sym, equation=True)
                    if len(by_sym) > 0:
                        print(cl, by_sym)
                        if cl in ner_sym.keys():
                            ner_sym[cl] += by_sym  # = {acr: by_acr}
                        else:
                            ner_sym[cl] = by_sym

    if semantic_type == "all":
        ner = ner_label.copy()
        ner.update(ner_acr)
        ner.update(ner_syn)
        ner.update(ner_sym)
        return ner

    elif semantic_type == "label":
        return ner_label

    elif semantic_type == "synonym":
        return ner_syn

    elif semantic_type == "acronym":
        return ner_acr

    elif semantic_type == "symbol":
        return ner_sym

    else:
        raise ValueError("No correct type specified")


def plot_ner_dict(
    ner_label,
    add_graph=False,
    node_size_factor=10,
    ax=None,
    g_bbox=[0.2, 0.1, 0.8, 0.9],
):
    # Assuming you already have the 'ner_label' dictionary with the data
    if isinstance(list(ner_label.keys())[0], str):
        concepts = list(ner_label.keys())
        freq = list(ner_label.values())
    else:
        concepts = []
        freq = []
        for k, v in ner_label.items():
            if str(k) == "Model":
                continue
            label = k.label.first()
            if not isinstance(label, str):
                continue
            concepts.append(label)
            if isinstance(v, list):
                count = int(len(v))
            else:
                count = int(v)
            freq.append(count)

    # Sort the concepts and frequencies in descending order based on frequencies
    sorted_indices = np.argsort(freq)[::-1]
    concepts = np.array(concepts)[sorted_indices]
    freq = np.array(freq)[sorted_indices]

    # Define a colormap for coloring the bars based on frequency values
    # You can choose any colormap you like. Here, I am using 'viridis'.
    cmap = plt.cm.viridis

    if isinstance(ax, type(None)):
        fig, ax = plt.subplots(figsize=(8, 5))
        return_fig = True
    else:
        return_fig = False

    # Creating the bar plot with sorted heights and colored based on frequency
    bars = ax.bar(concepts, freq, color=cmap(freq / max(freq)))

    # To show the color bar indicating the frequency scale, uncomment the following lines:
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(freq), vmax=max(freq)))
    # sm._A = []
    # cbar = plt.colorbar(sm)

    # Optionally, rotate the x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    plt.xlabel("Concepts")
    plt.ylabel("Frequency")
    plt.title("Stefanovski et al. 2019")

    if add_graph:
        axg = ax.inset_axes(bounds=g_bbox)
        axg.axis("off")
        axg.set_facecolor("none")

        G = graph.owl2networkx(ontology.onto)
        nodes = [ontology.search_class(str(c)) for c in concepts]

        cmap = plt.cm.viridis
        colors = dict()
        for n, f in zip(nodes, freq):
            colors[n] = cmap(f / max(freq))

        subset = nodes.copy()
        for n in nodes:
            subset += list(G.neighbors(n))

        g = G.subgraph(nodes=subset)
        node_degrees = dict(G.degree())
        scaling_factor = lambda x: x * node_size_factor
        node_sizes = {
            node: scaling_factor(degree) for node, degree in node_degrees.items()
        }

        # Plot.
        pos = nx.kamada_kawai_layout(g)
        nx.draw_networkx_nodes(
            g,
            pos=pos,
            ax=axg,
            node_size=[node_sizes[n] if n in colors else 5 for n in g.nodes],
            alpha=0.5,
            node_color=[
                colors[n] if n in colors else (0.5, 0.5, 0.5, 0.3) for n in g.nodes
            ],
        )
        nx.draw_networkx_labels(
            g,
            pos=pos,
            labels={n: str(n) if n in colors else "" for n in g.nodes},
            ax=axg,
            font_size=mpl.rcParams["font.size"] - 2,
        )
        nx.draw_networkx_edges(g, pos=pos, ax=axg, alpha=0.6, edge_color="grey")

    # fig.set_dpi(1000)
    if return_fig:
        plt.tight_layout()
        plt.close()
        return fig
