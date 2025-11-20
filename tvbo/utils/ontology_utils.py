#
# Module: ontology_utils.py
#
# Author: Leon Martin
# Copyright © 2024 Charité Universitätsmedizin Berlin.
# Licensed under the EUPL-1.2-or-later
#
"""
This module contains helper functions for interacting with the TVBO ontology.
"""
import json

import owlready2 as owl
from tqdm import tqdm

from tvbo.knowledge import ontology


def ontology2dict(onto="default", include_individuals=True, include_GO=False):
    if onto == "default":
        onto = ontology.onto

    j = dict(directed=True, multigraph=True, graph={}, nodes=[], links=[])

    for cls in tqdm(list(ontology.onto.classes()), desc="Processing classes"):

        if cls.name == "Thing":
            continue

        # if not include_GO:
        #     if cls in ontology.onto["TVB-GO"].descedants():
        #         continue

        cls_info = {"id": cls.name, "label": cls.label.first, "type": "class"}
        if cls in onto.Parameter.descendants():
            cls_info.update({"type": "Parameter"})
        elif cls in onto.StateVariable.descendants():
            cls_info.update({"type": "StateVariable"})
        elif cls in onto.Constant.descendants():
            cls_info.update({"type": "Constant"})

        props = ontology.get_class_annotation_properties(cls)
        cls_info.update(props)

        j["nodes"].append(cls_info)

        for parent in cls.is_a:
            if (
                isinstance(parent, owl.class_construct.Restriction)
                or parent.name == "Thing"
            ):
                continue
            link = {
                "source": parent.name,
                "target": cls.name,
                "key": "is_a",
                "color": "#adbcc1",
            }
            j["links"].append(link)

        for p, o in ontology.get_class_object_properties(cls).items():
            if o.name == "Thing":
                continue
            link = {"source": cls.name, "target": o.name, "key": p, "color": "#adbcc1"}
            j["links"].append(link)

    if include_individuals:
        for i in tqdm(list(ontology.onto.individuals()), desc="Processing individuals"):
            label = i.label.first() or str(i)

            ind_info = {"id": i.name, "label": label, "type": "individual"}
            props = ontology.get_class_annotation_properties(i)
            ind_info.update(props)
            j["nodes"].append(ind_info)

            # Connect individual to its class
            for c in i.is_instance_of:
                if (
                    not isinstance(c, owl.class_construct.Restriction)
                    and c.name == "Thing"
                ):
                    continue
                elif isinstance(c, owl.class_construct.Restriction):
                    link = {
                        "source": i.name,
                        "target": c.value.name,
                        "key": c.property.label.first(),
                    }
                else:
                    link = {
                        "source": i.name,
                        "target": c.name,
                        "key": "is_instance_of",
                        "color": "#f0ebe3",
                    }
                j["links"].append(link)
    return j


def ontology2json(filename):
    j = ontology2dict()
    with open(filename, "w") as f:
        json.dump(j, f, indent=4)
