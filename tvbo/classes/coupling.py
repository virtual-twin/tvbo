# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""Coupling functions.

The public import location for :class:`Coupling`, alongside the helpers that read a coupling function out of the database or the ontology.

There is no wrapper class: the coupling's own methods live in :mod:`tvbo.behaviour.coupling` and are attached to the generated class itself, so a coupling carries them however it was built.

```{seealso}
- [Coupling](![wiki]/Coupling/index.html)
```

"""

from tvbo.classes import equation as equations
from tvbo.datamodel.dialect import peer_module
from tvbo.datamodel.schema import Coupling
from tvbo.ontology import owl as ontology

__all__ = [
    "Coupling",
    "coupling_class2metadata",
    "get_global_coupling_functions",
    "get_parameters",
    "peer_module",
]


def _ensure_parameters(coupling) -> None:
    """Give *coupling* a parameters mapping to fill.

    The LinkML dataclass defaults the slot to an empty collection; the Pydantic model defaults it to ``None``. A keyed collection is otherwise always mutated in place, so this is the one assignment.
    """
    if getattr(coupling, "parameters", None) is None:
        coupling.parameters = {}


def get_parameters(CF):
    """Extract parameter metadata from a coupling function ontology class.

    Args:
        CF: A coupling function name or an owlready2 ontology class. If a
            string is given, it is first resolved to the corresponding
            ontology class via the ontology registry.

    Returns:
        A mapping from each ontology parameter to a dict of its properties:
        `domain` (with `lo`, `hi`, and `step`), `value`, `definition`,
        `label`, and `name`.
    """
    if isinstance(CF, str):
        CF = ontology.get_coupling_function(CF)

    parameters = {}
    for p in CF.has_parameter:
        param_props = {"domain": {}}
        (
            param_props["domain"]["lo"],
            param_props["domain"]["hi"],
            param_props["domain"]["step"],
        ) = ontology.get_range(p) if ontology.get_range(p) else ("-inf", "inf", "0.001")
        param_props["value"] = (
            float(p.defaultValue.first()) if len(p.defaultValue) > 0 and p.defaultValue.first() != "None" else 0
        )
        param_props["definition"] = p.definition.first()
        param_props["label"] = ontology.replace_suffix(p.label.first())
        param_props["name"] = p.name
        parameters[p] = param_props
    return parameters


def coupling_class2metadata(ontoclass, metadata, overwrite: bool = False):
    """Populate coupling metadata from an ontology class.

    If overwrite is False (default), only fill missing fields.
    If overwrite is True, always set name and pre/post expressions.
    Parameters are added if missing; existing parameter value/description are only filled if missing regardless of overwrite.

    Members are built in *metadata*'s own generated form — see :func:`peer_module`.
    """
    peer = peer_module(metadata)
    _ensure_parameters(metadata)

    try:
        if overwrite or not getattr(metadata, "name", None):
            metadata.name = ontoclass.label.first()
    except Exception:
        pass

    try:
        eqs = equations.get_symbolic_coupling(ontoclass)
    except Exception:
        eqs = None
    if eqs:
        if overwrite or getattr(metadata, "pre_expression", None) is None:
            metadata.pre_expression = peer.Equation(rhs=str(eqs["pre"]))
        if overwrite or getattr(metadata, "post_expression", None) is None:
            metadata.post_expression = peer.Equation(rhs=str(eqs["post"]))

    for param in get_parameters(ontoclass).values():
        label = param["label"]
        if label not in metadata.parameters:
            metadata.parameters[label] = peer.Parameter(
                name=param["label"],
                value=param["value"],
                description=param["definition"],
            )
        else:
            if getattr(metadata.parameters[label], "value", None) is None:
                metadata.parameters[label].value = param["value"]
            if getattr(metadata.parameters[label], "description", None) is None:
                metadata.parameters[label].description = param["definition"]


def get_global_coupling_functions():
    """Return all coupling function classes defined in the ontology.

    Loads the ontology on demand and collects the subclasses of its `Coupling` class.

    Returns:
        A list of the ontology's `Coupling` subclasses.
    """
    onto = ontology.get_onto()
    CouplingFunctions = onto.Coupling.subclasses()

    #     CF.pre = MethodType(get_pre_summation_coupling_function, CF)
    return list(CouplingFunctions)


# No eager set here: traversing subclasses would load the whole ontology on every `import tvbo`.
