"""The clinical layer of the shipped ontology counted as a study analysis: the arrays agree with each other and with the totals."""

import pytest

from tvbo import Dynamics
from tvbo.data.registry import NAMESPACE
from tvbo.ontology.clinical import CLINICAL, SKOS, clinical_landscape
from tvbo.ontology.grounding import load_world
from tvbo.ontology.query import sparql_query


@pytest.fixture(scope="module")
def landscape():
    return clinical_landscape()


def test_shapes_and_totals_agree(landscape):
    apps = landscape["applications"]
    assert apps.dims == ("domain", "model")
    assert apps.sizes == {"domain": landscape["n_domains"], "model": landscape["n_models"]}
    assert int(landscape["studies_by_domain"].sum()) >= landscape["n_studies"] > 0


def test_every_domain_and_model_is_used(landscape):
    apps = landscape["applications"]
    assert (apps.sum("model") > 0).all()
    assert (apps.sum("domain") > 0).all()


def test_domains_are_grouped_by_icd11_chapter(landscape):
    chapters = list(landscape["studies_by_domain"].coords["chapter"].values)
    assert chapters == sorted(chapters)
    assert all(code for code in landscape["studies_by_domain"].coords["icd11"].values)


def test_every_model_family_has_a_curated_record():
    """Every clinical model family is tied by a SKOS match to at least one TVB-O model record, and every record it names is in the curated model database, so each clinical application study reaches a runnable model."""
    world = load_world()
    families = {
        m.iri
        for (m,) in sparql_query(
            f"SELECT ?m WHERE {{ ?m a <{CLINICAL}NodeDynamicsModel> . }}", flatten_result=False, world=world
        )
    }
    matches = sparql_query(
        f"SELECT ?m ?r WHERE {{ ?m a <{CLINICAL}NodeDynamicsModel> . VALUES ?p {{ <{SKOS}exactMatch> <{SKOS}narrowMatch> <{SKOS}closeMatch> <{SKOS}relatedMatch> }} ?m ?p ?r . }}",
        flatten_result=False,
        world=world,
    )
    models = NAMESPACE + "models/"
    records = {}
    for family, record in matches:
        if record.iri.startswith(models):
            records.setdefault(family.iri, set()).add(record.iri.removeprefix(models))
    assert families and records
    assert families - records.keys() == set()
    curated = set(Dynamics.list_db())
    assert {r for rs in records.values() for r in rs} - curated == set()
