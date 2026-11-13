"""The clinical layer of the shipped ontology counted as a study analysis: the arrays agree with each other and with the totals."""
import pytest

from tvbo.ontology.clinical import clinical_landscape


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
