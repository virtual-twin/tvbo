#
# Module: ontology_api.py
#
# Author: Romina Baila, Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#

"""
This module provides a set of methods, through the OntologyAPI interface, which
retrieves data from the ontology
"""

import owlready2 as owl
from sympy import latex, symbols

import tvbo.data.db as db
from tvbo.export.experiment import SimulationExperiment
from tvbo.knowledge import graph, ontology, query
from tvbo.parse import metadata

# exp = SimulationExperiment(
#     id=1,
#     **{
#         "model": {
#             "label": "Generic2dOscillator",
#             "parameters": {"label": "a", "value": 1},
#         }
#     },
# )

G = graph.onto2graph(storid=True)
onto = ontology.onto
db_studies = db.SimulationStudies()


def uid2int(uid):
    """
    Convert a UID object to an integer
    Args:
        uid: UID object
    Returns:
        int: integer
    """
    return int(uid.first()) if uid else 0


def label2symbol(node, delimiter=""):
    return (
        rf"{delimiter}{latex(symbols(node.symbol.first()))}{delimiter}"
        if hasattr(node, "symbol") and node.symbol.first()
        else ontology.replace_suffix(node)
    )


def ontoclass2dict(ontoclass):
    """
    Convert an OntoClass object to a JSON object
    Args:
        ontoclass: OntoClass object
    Returns:
        dict: JSON object
    """
    # if ontoclass.name in db_studies.files.keys():
    #     study = metadata.load_simulation_study(db_studies.files[ontoclass.name])
    #     exp = study.get_experiment(1)

    #     if exp.metadata.model.name:
    #         model = query.label_search(exp.metadata.model.name, root_class=onto.Model)[
    #             0
    #         ]
    #         if not model in ontoclass.requires:
    #             ontoclass.requires.append(model)
    #     if exp.metadata.coupling.name:
    #         coupling = query.label_search(
    #             exp.metadata.coupling.name, root_class=onto.Coupling
    #         )[0]
    #         if not coupling in ontoclass.requires:
    #             ontoclass.requires.append(coupling)

    #     if exp.metadata.integration.method:
    #         integration = query.label_search(
    #             exp.metadata.integration.method, root_class=onto.IntegrationMethod
    #         )[0]
    #         if not integration in ontoclass.requires:
    #             ontoclass.requires.append(integration)

    class_dict = {
        # "id": uid2int(ontoclass.identifier),
        "id": ontoclass.storid,
        "iri": ontoclass.iri,
        "label": ontology.replace_suffix(ontoclass),
        "symbol": label2symbol(ontoclass),
        "type": ontology.get_type(ontoclass).label.first(),
        "definition": ontoclass.definition.first(),
        "description": (
            ontoclass.title.first()
            if isinstance(ontoclass, onto.JournalArticle)
            else (
                ontoclass.description.first()
                if ontoclass.description
                else (
                    ontoclass.definition.first().split(".")[0]
                    if ontoclass.definition
                    else None
                )
            )
        ),
        "collapsed": True,
        "childNodes": [],
        "childLinks": [],
        "is_a": [p.name for p in ontoclass.is_a if isinstance(p, owl.ThingClass)],
        "requires": [p.storid for p in ontoclass.requires],
        "storid": ontoclass.storid,
    }

    return class_dict


class OntologyAPI:
    def __init__(self):
        self.edges = set()
        self.nodes = dict()
        self.graph = {"nodes": [], "links": []}
        pass

    def search_by_term(self, term="JansenRit", node2str=True):
        """
        Search for a term in the ontology
        Args:
            term: string with term to be searched
        Returns:
            list: list of all nodes containing the search term in their label/definition
        """
        if term.isdigit():
            res = [onto.world._get_by_storid(int(term))]
        else:
            res = query.label_search(term)
        return {r.storid: ontoclass2dict(r) for r in res}

    def query_nodes(self, query_str="noise"):
        """
        Search for a node in the ontology by label and return it together with all its direct connections.
        First node in the nodes list should always be the node being searched for.
        Args:
            label: label of the node to be searched for
        Returns:
            dict: a dictionary containing a list of nodes and a list of edges connecting the nodes
        """
        self.__init__()
        nodes = self.search_by_term(query_str)
        self.nodes = nodes.copy()

        # Add relationships between queried nodes
        # self.update_interrelationships()
        # for node_id, node in nodes.items():
        # self.expand_node_relationships(node_id, add_nodes=False)

        graph = self.update_graph()

        # return graph

    def print_triplets(self):
        for e in self.edges:
            print(
                ontology.onto.world._get_by_storid(e["source"]),
                e["type"],
                ontology.onto.world._get_by_storid(e["target"]),
            )

    # def expand_node_relationships(self, node_id, add_nodes=True):
    #     """
    #     Expand a node by retrieving all its direct connections
    #     Args:
    #         node_id: id of the node to be expanded
    #     Returns:
    #         dict: a dictionary containing a list of nodes and a list of edges connecting the nodes
    #     """

    #     node = onto.world._get_by_storid(node_id)

    #     obj_properties = ontology.get_class_properties(node)["object_properties"]

    #     for prop in obj_properties:
    #         ((p, o),) = prop.items()
    #         if isinstance(o, float) or isinstance(o, int):
    #             print(
    #                 f"print skipping object property '{p}' since it just has a numeric value: {o}"
    #             )
    #             continue
    #         if isinstance(o, str):
    #             continue
    #         if hasattr(o, "name") and (
    #             o.name == "Thing" or o.name == "NamedIndividual"
    #         ):
    #             if p != "is_a":
    #                 print(
    #                     f"{o} is not a valid target for object property. Remove triplet ({node} - {p} - {o}) from ontlogy"
    #                 )
    #                 continue
    #             else:
    #                 continue
    #         else:
    #             continue
    #         o_dict = ontoclass2dict(o)

    #         if not add_nodes and o_dict["id"] not in self.nodes:
    #             continue  # TODO: find another solution sometime

    #         if add_nodes and not o_dict["id"] in self.nodes:
    #             self.nodes.update({o_dict["id"]: o_dict})

    #         if node_id in self.nodes and o.storid in self.nodes:
    #             edge = {
    #                 "source": node_id,
    #                 "target": o.storid,
    #                 "type": p,
    #             }
    #             if not edge in self.edges:
    #                 self.edges.append(edge)

    #     self.update_graph()

    def update_interrelationships(self):
        self.edges.update(set(G.subgraph(self.nodes.keys()).edges(data="type")))

        # self.edges.update(
        #     {
        #         {"source": e[0], "target": e[1], "type": e[2]}
        #         for e in set(
        #             [
        #                 (e[0], e[1], e[2]["type"])
        #                 for e in G.subgraph(self.nodes.keys()).edges(data=True)
        #             ]
        #         )
        #     }
        # )

        # nodes = [onto.world._get_by_storid(n["storid"]) for n in self.nodes.values()]
        # G_sub = G.subgraph(nodes)
        # for s, o, p in G_sub.edges(data=True):
        #     self.edges.append(
        #         {
        #             "source": s.storid,
        #             "target": o.storid,
        #             "type": (
        #                 p["type"].replace("rdfs:subClassOf", "is_a")
        #                 if isinstance(p["type"], str)
        #                 else p["type"].name
        #             ),
        #         }
        #     )

    def update_graph(self):
        self.update_interrelationships()
        # edges = self.edges
        # edges = {
        #     (int(s), int(o), p)
        #     for s, o, p in zip(
        #         [e["source"] for e in edges],
        #         [e["target"] for e in edges],
        #         [e["type"] for e in edges],
        #     )
        # }
        # edges = list(set(edges))
        # self.edges = [{"source": s, "target": o, "type": p} for s, o, p in edges]

        self.graph = {
            "nodes": list(self.nodes.values()),
            "links": [{"source": s, "target": o, "type": p} for s, o, p in self.edges],
        }
        return self.graph

    def add_children(self, node_id):
        self.nodes.update(
            {
                s_id: ontoclass2dict(onto.world._get_by_storid(s_id))
                for s_id in G.successors(node_id)
            }
        )

        for r in onto.world._get_by_storid(node_id).requires:
            self.nodes.update({r.storid: ontoclass2dict(r)})
            self.edges.update({(node_id, r.storid, "requires")})

        # traverse_down = query.get_children(onto.world._get_by_storid(int(node_id)))
        # for prop, cl in traverse_down:
        #     if not isinstance(cl, owl.ThingClass):
        #         continue
        #     cl_dict = ontoclass2dict(cl)
        #     self.nodes.update({cl_dict["id"]: ontoclass2dict(cl)})
        #     self.edges.append(
        #         {
        #             "source": cl.storid,
        #             "target": int(node_id),
        #             "type": prop.replace("rdfs:subClassOf", "is_a"),
        #         }
        #     )

        self.update_graph()

    def add_parents(self, node_id):

        for s_id in G.predecessors(node_id):
            self.nodes.update({s_id: ontoclass2dict(onto.world._get_by_storid(s_id))})

        # node = ontology.onto.world._get_by_storid(node_id)
        # for pred in G.predecessors(node):
        #     self.nodes.update({pred.storid: ontoclass2dict(pred)})
        #     for e in G[pred][node].values():
        #         etype = e["type"]
        #         self.edges.append(
        #             {
        #                 "source": node.storid,
        #                 "target": pred.storid,
        #                 "type": etype if isinstance(etype, str) else etype.name,
        #             }
        #         )

        # traverse_up = query.get_parents(onto.world._get_by_storid(int(node_id)))
        # for prop, cl in traverse_up:
        #     cl_dict = ontoclass2dict(cl)
        #     self.nodes.update({cl_dict["id"]: ontoclass2dict(cl)})
        #     self.edges.append(
        #         {
        #             "source": int(node_id),
        #             "target": cl.storid,
        #             "type": prop.replace("rdfs:subClassOf", "is_a"),
        #         }
        #     )
        self.update_graph()

    def get_child_connections(self, node_id):
        """
        Return all the children nodes of the given node and the corresponding edges
        """
        node_id = int(node_id)
        self.add_children(node_id)
        # child_nodes = []
        # child_links = []
        # for e in self.edges:

        #     if e["source"] == node_id:
        #         pass
        #         # child_links.append(e)
        #         # if e["target"] not in self.nodes:
        #         #     pass
        #         # else:
        #         #     child_nodes.append(self.nodes[e["target"]])
        #     elif e["target"] == node_id:
        #         child_links.append(e)
        #         child_nodes.append(self.nodes[e["source"]])

        child_nodes = [self.nodes[s] for s in G.successors(node_id)]
        child_links = [
            {"source": src, "target": tgt, "type": type_val}
            for src, tgt, type_val in {
                (node_id, s, G[node_id][s][0]["type"]) for s in G.successors(node_id)
            }
        ]
        return {"nodes": child_nodes, "links": child_links}

    def get_parent_connections(self, node_id):
        """
        Return all the direct parent nodes of the given node and the corresponding edges
        """
        node_id = int(node_id)
        self.add_parents(node_id)
        # parent_nodes = []
        # parent_links = []
        # for e in self.edges:
        #     if e["source"] == node_id:
        #         parent_links.append(e)
        #         parent_id = int(e["target"])
        #         parent_nodes.append(self.nodes[parent_id])

        parent_nodes = [self.nodes[s] for s in G.predecessors(node_id)]
        parent_links = [
            {"source": src, "target": tgt, "type": type_val}
            for src, tgt, type_val in {
                (s, node_id, G[s][node_id][0]["type"]) for s in G.predecessors(node_id)
            }
        ]

        return {"nodes": parent_nodes, "links": parent_links}

    def configure_simulation_experiment(self, metadata):
        """
        Configure a simulation experiment based on the metadata configuration.
        The metadata conforms to the tvbo-datamodel and should have the following format:
        ```
        {
            "model": {
                "label": "Generic2dOscillator",
                "parameters": {"label": "a", "value": 1},
            },
            "connectivity": {
                "parcellation": {
                    "label": "DesikanKilliany"
                },
                "tractogram": {"label":"dTOR"},
            },
            "coupling": {"label": "Linear"},
            "integration": {
                "noise": {
                    "additive":  True,
                    "parameters": {"label": "sigma", "value": 0.1},
                }
            },
        }
        ```
        """
        self.experiment = SimulationExperiment(
            id=1,
            **metadata,
        )
