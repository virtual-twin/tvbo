#  coupling.py
#
# Created on Mon Jan 22 2024
# Author: Leon K. Martin
#
# Copyright (c) 2024 Charité Universitätsmedizin Berlin
#
"""
Coupling functions
==================
The public import location for :class:`Coupling`, alongside the helpers that read a
coupling function out of the database or the ontology.

There is no wrapper class: the coupling's own methods live in
:mod:`tvbo.behaviour.coupling` and are attached to the generated class itself, so a
coupling carries them however it was built.

```{seealso}
- [Coupling](![wiki]/Coupling/index.html)
```

"""

from tvbo.datamodel.schema import Coupling
from tvbo.ontology import owl as ontology
from tvbo.classes import equation as equations

__all__ = [
    "Coupling",
    "coupling_class2metadata",
    "get_global_coupling_functions",
    "get_parameters",
]


def peer_module(instance):
    """The generated module *instance*'s class comes from.

    A coupling is filled with ``Equation`` and ``Parameter`` members, and those have to be
    of the same generated form as the coupling itself: the strict Pydantic models validate
    on assignment and reject a LinkML dataclass where they want their own peer.
    """
    import importlib

    return importlib.import_module(type(instance).__module__)


def _ensure_parameters(coupling) -> None:
    """Give *coupling* a parameters mapping to fill.

    The LinkML dataclass defaults the slot to an empty collection; the Pydantic model
    defaults it to ``None``. A keyed collection is otherwise always mutated in place, so
    this is the one assignment.
    """
    if getattr(coupling, "parameters", None) is None:
        coupling.parameters = {}


def _load_coupling_from_database(name, coupling):
    """Fill coupling metadata from a curated database YAML file.

    Resolves ``name`` through the database registry — the one component that knows where
    the database lives — and fills the ``pre_expression``, ``post_expression`` and
    ``parameters`` the coupling does not already carry.

    Reading the curated entry is what makes population deterministic: the ontology answers
    with an unordered set, so the parameters it yields come back in a different order in
    every process, which no frozen record can be written against.

    A curated ``delayed:`` is NOT applied. The slot carries a schema default of ``True``,
    so it is never unset by the time this runs and the guard that read it could not fire.
    Only ``FastLinearCoupling`` declares anything else, and honouring it would change that
    coupling's delays — a measured change of its own, not a side effect of this one.

    Parameters
    ----------
    name : str
        Coupling function name (e.g. ``"KuramotoCoupling"``).
    coupling : Coupling
        Coupling instance to fill (modified in-place), in either generated form.

    Returns
    -------
    bool
        True if a database file was found and applied.
    """
    import yaml as _yaml

    from tvbo.data.registry import resolve

    try:
        db_file = resolve("Coupling", name)
    except (FileNotFoundError, RuntimeError, ValueError):
        return False

    data = _yaml.safe_load(db_file.read_text(encoding="utf-8"))
    peer = peer_module(coupling)
    _ensure_parameters(coupling)

    for slot in ("pre_expression", "post_expression"):
        if slot in data and not getattr(coupling, slot, None):
            written = data[slot]
            setattr(coupling, slot, peer.Equation(**(written if isinstance(written, dict) else {"rhs": written})))
    for pname, pval in (data.get("parameters") or {}).items():
        if pname in coupling.parameters:
            continue
        coupling.parameters[pname] = (
            peer.Parameter(**{"name": pname, **pval}) if isinstance(pval, dict) else peer.Parameter(name=pname, value=pval)
        )

    return True


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
    Parameters are added if missing; existing parameter value/description are
    only filled if missing regardless of overwrite.

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

    for key, param in get_parameters(ontoclass).items():
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

    Loads the ontology on demand and collects the subclasses of its
    `Coupling` class.

    Returns:
        A list of the ontology's `Coupling` subclasses.
    """
    onto = ontology.get_onto()
    CouplingFunctions = onto.Coupling.subclasses()

    # for CF in CouplingFunctions:
    #     CF.pre = MethodType(get_pre_summation_coupling_function, CF)
    return list(CouplingFunctions)


# NOTE: do NOT eagerly compute an ``available_coupling_functions`` set at import time. It has no consumers, and traversing ``onto.Coupling.subclasses()`` forces the (metadata-only) owlready2 ontology to fully load on every ``import tvbo`` — including JAX/codegen processes that never query the ontology. Call
# ``get_global_coupling_functions()`` on demand instead.
