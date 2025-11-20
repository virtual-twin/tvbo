#  graph.py
#
# Created on Mon Aug 07 2023
# Author: Leon K. Martin
#
# Copyright (c) 2023 Charité Universitätsmedizin Berlin
#
"""
# Graph-based representation of the ontology.

This module contains functions for representing ontology-based structures
as graphs, and various utilities for manipulating and visualizing these graphs.

## Functions:

- [`owl2networkx`](#owl2networkx): Convert an ontology object to a NetworkX graph.
- [`nx2mermaid`](#nx2mermaid): Convert a NetworkX graph to a Mermaid representation.
- [`create_graph_from_dataframe`](#create_graph_from_dataframe): Construct a graph from a pandas DataFrame representing ontology.
- [`get_color_mapping`](#get_color_mapping): Map nodes of a graph to distinct colors based on a node attribute.
- [`get_node_colors`](#get_node_colors): Retrieve the colors associated with nodes in a graph.

"""
# TODO: review the commented code in this file
import re
from itertools import count
from typing import Dict, Any, List, Union

import networkx as nx
import numpy as np
import owlready2 as owl
from sympy import latex, symbols

from tvbo.knowledge import ontology


###################
# networkX Graphs #
###################


def edge_exists(G: nx.MultiDiGraph, source, target, edge_type: str) -> bool:
    """
    Check if an edge with the given type exists between source and target in a MultiDiGraph.

    Args:
        G (nx.MultiDiGraph): The graph.
        source (hashable): Source node.
        target (hashable): Target node.
        edge_type (str): Type attribute of the edge to check.

    Returns:
        bool: True if such an edge exists, False otherwise.
    """
    edge_dict = G.get_edge_data(source, target)
    if edge_dict is None:
        return False
    return any(data.get("type") == edge_type for data in edge_dict.values())


def onto2graph(
    onto="default", add_object_properties: bool = True, storid: bool = False, object2string: bool = True
) -> nx.MultiDiGraph:
    """
    Convert an ontology into a NetworkX directed graph.

    The function generates a directed graph (`DiGraph`) where:

    - Nodes represent ontology classes.
    - Node attributes contain annotation properties of the classes.
    - Edges represent relationships between the classes, either
      hierarchical (`is_a`) or based on object properties.

    Note:
        The function assumes that there's a utility function
        `get_class_properties(c)` available which retrieves properties
        of a given ontology class in a predefined format, especially
        the "annotation_properties" and "object_properties".

    Warning:
        The function omits the "Thing" class and its properties
        to avoid redundant information.

    Returns:
        nx.MultiDiGraph: A directed multigraph representation of the ontology with
        nodes representing ontology classes and edges representing relationships.

    Examples:
        >>> G = owl2nx_digraph()
        >>> print(G.nodes(data=True))
        [('ClassA', {'ID': 'id123', 'label': 'A'}), ...]
        >>> print(G.edges(data=True))
        [('ClassA', 'ClassB', {'type': 'is_a'}), ...]

    Raises:
        KeyError: If expected properties are not found in the ontology class.
        TypeError: If the ontology structure differs from the expected format.
    """
    if onto == "default":
        onto = ontology.onto

    nx_graph = nx.MultiDiGraph()

    for c in ontology.onto.classes():
        if c.name == "Thing":
            continue
        # Add class as node
        props = ontology.get_class_properties(c)
        label = c.label.first()

        node_id = c.storid if storid else (c if not object2string else label)
        nx_graph.add_node(node_id, node_type="class")

        if not object2string:
            nx_graph.nodes[node_id]["ID"] = props["identifier"]

            # Annotation properties
            for k, v in props["annotation_properties"].items():
                if isinstance(v, str):
                    v = " ".join(
                        [
                            s if not s.isupper() else s
                            for s in re.split(r"[ -]", v.strip())
                        ]
                    )
                nx_graph.nodes[node_id][k] = v

        # Class hierarchy
        for o in c.is_a:
            if isinstance(o, owl.class_construct.Restriction) or o.name == "Thing":
                continue
            parent_id = (
                o.storid if storid else (o if not object2string else o.label.first())
            )
            if not edge_exists(nx_graph, node_id, parent_id, edge_type="is_a"):
                nx_graph.add_edge(node_id, parent_id, type="is_a")

        # Object properties
        if add_object_properties:
            for prop in props["object_properties"]:
                p, o = next(iter(prop.items()))
                if p in ["has_data_type", "has_dependency"] or o.name == "Thing":
                    continue
                object_id = (
                    o.storid
                    if storid
                    else (o if not object2string else o.label.first())
                )
                property_id = (
                    p.storid
                    if storid and not isinstance(p, str)
                    else (
                        p
                        if not object2string
                        else (
                            onto.search_one(iri=f"*{p}").label.first()
                            if p != "is_a"
                            else "is_a"
                        )
                    )
                )
                if not edge_exists(nx_graph, node_id, object_id, edge_type=p):
                    nx_graph.add_edge(node_id, object_id, type=property_id)

    # Add individuals as nodes
    for i in ontology.onto.individuals():
        individual_id = (
            i.storid
            if storid
            else (i if not object2string else (i.label.first() or str(i)))
        )
        nx_graph.add_node(individual_id, node_type="individual")

        # Connect individual to its class
        for c in i.is_instance_of:
            if isinstance(c, owl.class_construct.Restriction):

                class_id = (
                    c.value.storid
                    if storid
                    else (c.value if not object2string else c.value.label.first())
                )

                property_id = (
                    c.property if not object2string else c.property.label.first()
                )
                if not edge_exists(
                    nx_graph, individual_id, class_id, edge_type=property_id
                ):
                    nx_graph.add_edge(individual_id, class_id, type=property_id)
            else:
                class_id = (
                    c.storid
                    if storid
                    else (c if not object2string else (c.label.first() or str(c)))
                )
                if not edge_exists(
                    nx_graph, individual_id, class_id, edge_type="is_instance_of"
                ):
                    nx_graph.add_edge(
                        individual_id,
                        class_id,
                        type="is_instance_of",
                    )

    return nx_graph


# TODO: add_object_properties is not used, remove it?
def owl2nx_digraph(onto="default", add_object_properties: bool = True, object2string: bool = True) -> nx.MultiDiGraph:
    """
    Convert an ontology into a NetworkX directed graph.

    The function generates a directed graph (`DiGraph`) where:

    - Nodes represent ontology classes.
    - Node attributes contain annotation properties of the classes.
    - Edges represent relationships between the classes, either
      hierarchical (`is_a`) or based on object properties.

    Note:
        The function assumes that there's a utility function
        `get_class_properties(c)` available which retrieves properties
        of a given ontology class in a predefined format, especially
        the "annotation_properties" and "object_properties".

    Warning:
        The function omits the "Thing" class and its properties
        to avoid redundant information.

    Returns:
        networkx.DiGraph: A directed graph representation of the ontology
        with nodes representing ontology classes and edges representing
        relationships.

    Examples:
        >>> G = owl2nx_digraph()
        >>> print(G.nodes(data=True))
        [('ClassA', {'ID': 'id123', 'label': 'A'}), ...]
        >>> print(G.edges(data=True))
        [('ClassA', 'ClassB', {'type': 'is_a'}), ...]

    Raises:
        KeyError: If expected properties are not found in the ontology class.
        TypeError: If the ontology structure differs from the expected format.
    """
    if onto == "default":
        onto = ontology.onto

    nx_graph = nx.MultiDiGraph()

    for c in ontology.onto.classes():
        if c.name == "Thing":
            continue
        # Add class as node
        props = ontology.get_class_properties(c)
        label = c.label.first()

        nx_graph.add_node(c if object2string else label, node_type="class")

        if not object2string:
            nx_graph.nodes[label]["ID"] = props["identifier"]

            # Annotation properties
            for k, v in props["annotation_properties"].items():
                if isinstance(v, str):
                    v = " ".join(
                        [
                            s if not s.isupper() else s
                            for s in re.split(r"[ -]", v.strip())
                        ]
                    )
                nx_graph.nodes[label][k] = v

        # Class hierarchy
        for o in c.is_a:
            if isinstance(o, owl.class_construct.Restriction) or o.name == "Thing":
                continue
            if object2string:
                nx_graph.add_edge(label, o.label.first(), type="is_a")
            else:
                nx_graph.add_edge(c, o, type="is_a")

        # Object properties
        if object2string:
            for p, o in props["object_properties"].items():
                if o.name == "Thing":
                    continue
                nx_graph.add_edge(label, o.label.first(), type=p)
        else:
            for obj_prop in onto.object_properties():
                val = getattr(c, obj_prop.python_name, [])
                if len(val) > 0:
                    for v in val:
                        nx_graph.add_edge(c, v, type=obj_prop)

    # Add individuals as nodes
    for i in ontology.onto.individuals():
        label = i.label.first() or str(i)
        nx_graph.add_node(label, node_type="individual")

        # Connect individual to its class
        for c in i.is_instance_of:
            if isinstance(c, owl.class_construct.Restriction):
                nx_graph.add_edge(
                    label if object2string else c,
                    c.value.label.first() if object2string else c.value,
                    type=c.property.label.first() if object2string else c.property,
                )
            else:
                class_label = c.label.first() or str(c)
                nx_graph.add_edge(
                    label if object2string else i,
                    class_label if object2string else c,
                    type="is_instance_of",
                )

    return nx_graph


def nx2mermaid(G: nx.Graph, id_as_label: bool = False) -> str:
    """
    Convert a NetworkX graph to a Mermaid representation.

    Parameters:
    -----------
    G : nx.Graph
        NetworkX graph to be converted.
    id_as_label : bool, optional
        Use identifier as label in the Mermaid graph. Default is False.

    Returns:
    --------
    str
        The Mermaid representation of the graph.
    """
    mm_list = list()

    replace_dict = {"(": "-", ")": "-"}

    for edge in G.edges:
        s = edge[0]
        if isinstance(s, str):
            s = ontology.search_class(s)

        o = edge[1]
        if isinstance(o, str):
            o = ontology.search_class(o)

        if s.name == "Thing" or o.name == "Thing":
            continue
        s_id = s.identifier.first()
        o_id = o.identifier.first()
        if id_as_label:
            s = s_id
            o = o_id
        else:
            s = s.label.first()
            o = o.label.first()

        for k, v in replace_dict.items():
            s = s.replace(k, v)
            o = o.replace(k, v)

        mm_list.append("{}[{}] --> {}[{}]".format(s_id, s, o_id, o))

    mm = "\n".join(["\t" + m for m in mm_list])

    mm_graph = """
    flowchart TD
    {}
    """.format(
        mm
    )
    return mm_graph


def get_color_mapping(g: nx.Graph, by: str = "type") -> Dict[Any, int]:
    """
    Map nodes of a graph to distinct colors based on a node attribute.

    Parameters:
    -----------
    g : nx.Graph
        The input graph.
    by : str, optional
        Node attribute to be used for color mapping. Default is "type".

    Returns:
    --------
    dict
        A dictionary mapping each node to a color index.
    """
    groups = set(nx.get_node_attributes(g, by).values())
    mapping = dict(zip(sorted(groups), count()))
    nodes = g.nodes()
    color_mapping = {n: mapping[g.nodes[n]["type"]] for n in nodes}
    return color_mapping


###################
# Model Selection #
###################


def subset2graph(
    subset,
    add_object_properties: bool = True,
    add_annotation_properties: bool = True,
    add_individuals: bool = True,
    individual_relationships: List[str] = ["has_reference"],
    expand_nodes: bool = False,
) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for cls in subset:
        if isinstance(cls, owl.class_construct.Restriction):
            continue
        if add_annotation_properties:
            cls_data = ontology.get_class_properties(cls)["annotation_properties"]
        else:
            cls_data = {}
        G.add_node(cls, **cls_data)
        for isa in cls.is_a:
            if isinstance(isa, owl.ThingClass):
                G.add_edge(cls, isa, type="is_a")
            if add_object_properties and isinstance(isa, owl.Restriction):
                if isinstance(isa.property, owl.prop.DataPropertyClass):
                    continue
                G.add_edge(cls, isa.value, type=isa.property.name)
    if add_individuals:
        edges_to_add = list()
        for n in G.nodes:
            if isinstance(n, owl.entity.ThingClass):
                continue
            for prop in n.get_properties():

                if prop.name in individual_relationships:
                    for ref in prop[n]:
                        edges_to_add.append((ref, n, "has_reference"))

        for edge in edges_to_add:
            G.add_edge(*edge)

    # if not expand_nodes:
    #     G = G.subgraph(subset)
    return G


def hierarchy_graph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    G_isa = nx.MultiDiGraph()

    # Add only 'is_a' edges to the new graph
    for u, v, key, data in G.edges(data=True, keys=True):
        if "type" in data.keys() and data["type"] == "is_a":
            G_isa.add_node(u, **G.nodes[u])
            G_isa.add_node(v, **G.nodes[v])
            G_isa.add_edge(u, v, key=key, **data)
    return G_isa


def model2graph(model) -> nx.MultiDiGraph:
    if isinstance(model, str):
        model = ontology.get_model(model)
    G = nx.MultiDiGraph()
    include_nodes = list()
    for cls in model.descendants(include_self=False):
        category = ontology.intersection(
            cls.is_a,
            [
                ontology.onto.Parameter,
                ontology.onto.StateVariable,
                ontology.onto.TimeDerivative,
                ontology.onto.Function,
                ontology.onto.ConditionalDerivedVariable,
            ],
        )
        if len(category) == 0:
            continue
        else:
            include_nodes.append(cls)
            category = category[0]

        G.add_node(
            cls,
            type=category,
        )

        for isa in cls.is_a:
            if isinstance(isa, owl.ThingClass):
                # print(cls, "is_a", isa)
                G.add_edge(cls, isa, type="is_a")

            if isinstance(isa, owl.Restriction) and isa.value in model.descendants():
                if isinstance(isa.property, owl.prop.DataPropertyClass):
                    continue
                # print(cls, isa.property.name, isa.value)
                G.add_edge(cls, isa.value, type=isa.property.name)

        # for isa in cls.is_a:
        #     if isinstance(isa, owl.ThingClass) and isa in model.descendants(
        #         include_self=False
        #     ):
        #         G.add_edge(cls, isa, type=isa.property)

        # if isinstance(isa, owl.Restriction) and isa.value in model.descendants(
        #     include_self=False
        # ):

        # if isa.value in list(model.descendants(include_self=False)):
        #     if isinstance(isa.property, owl.prop.DataPropertyClass):
        #         continue
        #         # print(type(isa.property))
        #         # if isinstance(isa.property, owl.ObjectPropertyClass):
        #         #     # print(cls, isa.value, isa.property)
        # elif hasattr(isa, "property"):
        #     include_nodes.append(isa.value)
        #     G.add_edge(cls, isa.value, type=isa.property.name)
    G = G.subgraph(include_nodes)
    # isolated = list(nx.isolates(G))
    G = G.copy()
    # G.remove_nodes_from(isolated)

    return G


def adjust_positions(pos: Dict[Any, np.ndarray], threshold_percent: int = 10, direction: str = "xy", mode: str = "outward") -> Dict[Any, np.ndarray]:
    pos_array = np.array(list(pos.values()))

    x_span = np.max(pos_array[:, 0]) - np.min(pos_array[:, 0])
    y_span = np.max(pos_array[:, 1]) - np.min(pos_array[:, 1])

    x_threshold = x_span * threshold_percent / 100 if "x" in direction else 0
    y_threshold = y_span * threshold_percent / 100 if "y" in direction else 0

    def adjust_if_needed(coord_diff, threshold, adjust_outward):
        if adjust_outward and coord_diff < threshold:
            return threshold / 2
        elif not adjust_outward and coord_diff > threshold:
            return -threshold / 2
        return 0

    for i in range(len(pos_array)):
        for j in range(i + 1, len(pos_array)):
            dx = pos_array[j, 0] - pos_array[i, 0]
            dy = pos_array[j, 1] - pos_array[i, 1]
            adjust_outward = mode == "outward"

            x_adjustment = adjust_if_needed(abs(dx), x_threshold, adjust_outward)
            y_adjustment = adjust_if_needed(abs(dy), y_threshold, adjust_outward)

            if "x" in direction:
                pos_array[i, 0] -= np.sign(dx) * x_adjustment
                pos_array[j, 0] += np.sign(dx) * x_adjustment

            if "y" in direction:
                pos_array[i, 1] -= np.sign(dy) * y_adjustment
                pos_array[j, 1] += np.sign(dy) * y_adjustment

    adjusted_pos = {key: pos_array[i] for i, key in enumerate(pos)}
    return adjusted_pos


def labels_as_symbols(G: nx.Graph) -> Dict[Any, str]:
    return {
        node: (
            f"${latex(symbols(node.symbol.first()))}$"
            if hasattr(node, "symbol") and node.symbol.first()
            else node
        )
        for node in G.nodes()
    }
