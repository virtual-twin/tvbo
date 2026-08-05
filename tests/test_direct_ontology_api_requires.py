"""``requires`` mixes entity references with bare LEMS exposure names.

The NeuroML-core classes ingested into ``tvbo.owl`` declare their requirements by
name — ``concentrationModel`` requires ``'surfaceArea'``, ``'iCa'`` — so the slot
holds plain strings where the rest of the ontology holds entities. Serialising
those with ``r.storid`` raised ``AttributeError: 'str' object has no attribute
'storid'`` and took out every caller of :meth:`DirectOntologyAPI.search`,
including the platform's whole-graph endpoint.

The names are real data, so they must survive serialisation; they just have no
node behind them, which is why the graph walk skips them.
"""

from __future__ import annotations

import pytest

from tvbo.api.direct_ontology_api import DirectOntologyAPI, _requirements, _serialize_entity


class _FakeEntity:
    """Enough of an owlready2 entity for :func:`_serialize_entity`."""

    def __init__(self, name, requires):
        self.name = name
        self.storid = 4242
        self.iri = f"https://w3id.org/tvbo/{name}"
        self.is_a = []
        self.requires = requires


def test_bare_requirement_names_serialize_instead_of_raising():
    entity = _FakeEntity("concentrationModel", ["surfaceArea", "iCa"])
    assert _serialize_entity(entity)["requires"] == ["surfaceArea", "iCa"]


def test_entity_requirements_are_reported_by_label():
    referenced = _FakeEntity("SurfaceArea", [])
    entity = _FakeEntity("mixedModel", [referenced, "iCa"])
    refs, names = _requirements(entity)
    assert (refs, names) == ([referenced], ["iCa"])
    assert _serialize_entity(entity)["requires"] == ["SurfaceArea", "iCa"]


@pytest.fixture(scope="module")
def api() -> DirectOntologyAPI:
    return DirectOntologyAPI()


def test_search_covers_the_concepts_the_platform_asks_for(api):
    """The KG endpoint searches these in one pass; one bad entity broke all of them."""
    carrying = {}
    for concept in ("Model", "NeuralMassModel", "Coupling", "IntegrationMethod",
                    "StateVariable", "Parameter", "BrainRegion", "Parcellation",
                    "Tractogram", "Monitor", "Noise"):
        for hit in api.search(concept, limit=50):
            if hit["requires"]:
                carrying[hit["name"]] = hit["requires"]

    assert carrying, "no entity carries requirements — the fixture below proves nothing"
    assert "surfaceArea" in carrying["concentrationModel"]
    for requires in carrying.values():
        assert all(isinstance(r, str) for r in requires)


def test_children_never_link_to_a_requirement_without_a_node(api):
    storid = api.search("concentrationModel", limit=5)[0]["storid"]
    children = api.get_children(storid)
    ids = {node["storid"] for node in children["nodes"]}
    for link in children["links"]:
        assert link["target"] in ids or link["target"] == storid
