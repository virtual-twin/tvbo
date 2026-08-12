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
from tvbo.classes.experiment import SimulationExperiment
from tvbo.ontology import graph
from tvbo.ontology import owl as ontology
from tvbo.ontology import query

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
    """Render an ontology node's symbol as LaTeX, falling back to its label.

    If the node carries a non-empty `symbol` annotation, it is parsed with
    SymPy and rendered as LaTeX, optionally wrapped in `delimiter` on either
    side. Otherwise the node's suffix-stripped label is returned.

    Args:
        node: Ontology class or individual to render.
        delimiter: String placed on both sides of the LaTeX symbol, e.g. `"$"`
            to produce inline math. Ignored when falling back to the label.

    Returns:
        The LaTeX symbol (wrapped in `delimiter`) or the node's label.
    """
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

    #     if exp.model.name:
    #         model = query.label_search(exp.model.name, root_class=onto.Model)[
    #             0
    #         ]
    #         if not model in ontoclass.requires:
    #             ontoclass.requires.append(model)
    #     if exp.coupling.name:
    #         coupling = query.label_search(
    #             exp.coupling.name, root_class=onto.Coupling
    #         )[0]
    #         if not coupling in ontoclass.requires:
    #             ontoclass.requires.append(coupling)

    #     if exp.integration.method:
    #         integration = query.label_search(
    #             exp.integration.method, root_class=onto.IntegrationMethod
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
                else (ontoclass.definition.first().split(".")[0] if ontoclass.definition else None)
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
    """Query the ontology and build graph data for the front end.

    Wraps the module-level ontology graph and exposes methods to search for
    terms and to expand a node's children or parents, accumulating the visited
    nodes and edges into a `{"nodes", "links"}` structure suitable for
    serialisation to a UI.

    Attributes:
        edges: Set of `(source, target, type)` triplets between node storids.
        nodes: Mapping of storid to a node dictionary (see `ontoclass2dict`).
        graph: Latest assembled graph with `"nodes"` and `"links"` lists.
    """

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

        self.update_graph()

        # return graph

    def print_triplets(self):
        """Print each accumulated edge as a subject-predicate-object triplet.

        Resolves the source and target storids of every edge in `self.edges`
        back to their ontology entities and prints them alongside the edge
        type. Intended for interactive debugging.
        """
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
        """Add edges among the currently held nodes from the ontology graph.

        Takes the subgraph of the ontology graph induced by the storids in
        `self.nodes` and merges its `(source, target, type)` edges into
        `self.edges`, capturing every relationship between nodes already present
        in the graph.
        """
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
        """Rebuild the serialisable graph from the current nodes and edges.

        Refreshes interrelationships via `update_interrelationships`, then
        assembles `self.graph` with a `"nodes"` list of node dictionaries and a
        `"links"` list of `{"source", "target", "type"}` edge dictionaries.

        Returns:
            The assembled graph dictionary.
        """
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
        """Add a node's children and required nodes, then refresh the graph.

        Inserts every successor of `node_id` in the ontology graph into
        `self.nodes`, and adds each entity referenced by the node's `requires`
        property together with a `"requires"` edge. Finally rebuilds the graph
        via `update_graph`.

        Args:
            node_id: Storid of the node whose children to add.
        """
        self.nodes.update({s_id: ontoclass2dict(onto.world._get_by_storid(s_id)) for s_id in G.successors(node_id)})

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
        """Add a node's parent nodes, then refresh the graph.

        Inserts every predecessor of `node_id` in the ontology graph into
        `self.nodes` and rebuilds the graph via `update_graph`.

        Args:
            node_id: Storid of the node whose parents to add.
        """

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
            for src, tgt, type_val in {(node_id, s, G[node_id][s][0]["type"]) for s in G.successors(node_id)}
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
            for src, tgt, type_val in {(s, node_id, G[s][node_id][0]["type"]) for s in G.predecessors(node_id)}
        ]

        return {"nodes": parent_nodes, "links": parent_links}

    def configure_simulation_experiment(self, metadata):
        """
        Configure a simulation experiment based on the metadata configuration.

        Accepts either legacy ``model``/``connectivity`` keys or the current
        ``dynamics``/``network`` keys.  Model and coupling names are resolved
        via ``Dynamics.from_db`` / ``Coupling.from_db`` so that equations,
        parameters, and state variables are fully populated. A coupling named at
        the top level is placed on the network it acts over, which is where the
        schema declares one.

        Example metadata::

            {
                "dynamics": "Generic2dOscillator",
                "network": {
                    "coupling": {"long_range": {"iri": "tvbo:Linear"}},
                    "parcellation": {"atlas": {"name": "DesikanKilliany"}},
                    "tractogram": {"name": "dTOR"},
                },
                "integration": {"method": "Heun"},
            }
        """
        from tvbo.classes.dynamics import Dynamics
        from tvbo.classes.coupling import Coupling

        md = dict(metadata)  # shallow copy to avoid mutating caller's dict

        # --- Resolve dynamics name → full Dynamics object ---
        dyn_raw = md.pop("dynamics", None) or md.pop("model", None)
        if dyn_raw is not None:
            if isinstance(dyn_raw, str):
                dyn_name = dyn_raw
            elif isinstance(dyn_raw, dict):
                dyn_name = dyn_raw.get("name") or dyn_raw.get("label", "Generic2dOscillator")
            else:
                dyn_name = str(dyn_raw)
            md["dynamics"] = Dynamics.from_db(dyn_name)

        # --- Resolve a coupling named at the top level onto the network it acts over ---
        coup_raw = md.pop("coupling", None)
        if coup_raw is not None:
            if isinstance(coup_raw, str):
                coup_name = coup_raw
            elif isinstance(coup_raw, dict):
                coup_name = coup_raw.get("name") or coup_raw.get("label", "Linear")
            else:
                coup_name = str(coup_raw)
            network = md.setdefault("network", {})
            network.setdefault("coupling", {})[coup_name] = Coupling.from_db(coup_name)

        # --- Normalise legacy connectivity → network ---
        if "connectivity" in md and "network" not in md:
            md["network"] = md.pop("connectivity")

        self.experiment = SimulationExperiment(
            id=1,
            **md,
        )

    def get_export_formats(self) -> list:
        """Return supported export format descriptors for UI dropdowns."""
        from tvbo import export as _export
        return _export.list_format_dicts()

    def export_experiment(
        self,
        format: str,
        directory: str,
        metadata_only: bool = True,
        **kwargs,
    ) -> dict:
        """Render the configured experiment and save into *directory*.

        Thin wrapper around :meth:`SimulationExperiment.save`. Returns a
        ``{"format", "files"}`` dict describing what was written. Format
        resolution (canonical key + aliases + extension) lives in
        :mod:`tvbo.export.registry`.
        """
        from pathlib import Path
        from tvbo import export as _export

        fmt = _export.resolve(format)
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        saved_path = self.experiment.save(
            out_dir, format=fmt.key, metadata_only=metadata_only, **kwargs
        )
        files = [saved_path]

        # Network HDF5 sidecar (if save() wrote one)
        if not metadata_only:
            stem = Path(saved_path).stem
            for ext in (".yaml", ".h5"):
                candidate = out_dir / f"{stem}_network{ext}"
                if candidate.exists():
                    files.append(str(candidate))

        return {"format": fmt.key, "files": files}
