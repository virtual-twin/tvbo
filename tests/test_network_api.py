"""tvbo's network API lists the curated networks whose ids, their file stems, spell the atlas and the tractogram asked for exactly, and ``Network.from_platform`` and ``Network.list_platform_networks`` read that listing.

The API is mounted in process with FastAPI's test client and ``requests.get`` is answered by it, so no test calls the network.
"""

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="requires the 'api' extra (pip install tvbo[api])")

import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tvbo import Network
from tvbo.api.network_api import router
from tvbo.data.registry import stem_entities

BASE_URL = "http://testserver"
API = "/api/v1/networks"


@pytest.fixture
def client(monkeypatch):
    """The network API in process, with ``requests.get`` answered by it."""
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    monkeypatch.setattr(requests, "get", lambda url, params=None: client.get(url.removeprefix(BASE_URL), params=params))
    return client


def _listed(client, **query):
    """The ids the API lists for *query*."""
    return [entry["id"] for entry in client.get(API, params=query).json()]


def _spelling(client, **entities):
    """The ids of the API's whole listing whose stems spell each of the BIDS *entities*."""
    return [
        net_id for net_id in _listed(client) if all(stem_entities(net_id).get(key) == value for key, value in entities.items())
    ]


@pytest.mark.parametrize("atlas", ["Lobar", "Lobar8", "HCPMMP1", "DesikanKilliany", "Schaefer2018"])
def test_an_atlas_filter_lists_the_networks_whose_stems_spell_that_atlas_exactly(client, atlas):
    """A network whose stem spells no atlas answers no atlas filter, and ``Lobar`` does not match ``Lobar8``."""
    found = _listed(client, atlas=atlas)
    assert found
    assert found == _spelling(client, atlas=atlas)


def test_a_tractogram_filter_matches_the_rec_entity_so_the_avgmatrix_networks_answer_it(client):
    """The avgMatrix networks state no ``tractogram`` but their stems spell ``rec-avgMatrix``, which the filter matches."""
    found = _listed(client, atlas="Lobar8", tractogram="avgMatrix")
    assert found
    assert found == _spelling(client, atlas="Lobar8", rec="avgMatrix")


def test_from_platform_hands_out_the_one_network_of_the_atlas_and_the_tractogram(client, tmp_path):
    (stem,) = _spelling(client, atlas="DesikanKilliany", rec="dTOR")
    net = Network.from_platform(atlas="DesikanKilliany", base_url=BASE_URL, cache_dir=tmp_path)
    own = Network(iri=f"tvbo:networks/{stem}")
    assert net.node_labels == own.node_labels
    np.testing.assert_array_equal(net.matrix("weight"), own.matrix("weight"))


def test_a_kept_older_spelling_is_matched_as_written_on_both_platform_doors(client, tmp_path):
    """``hcpmmp1`` is the atlas entity of no network's stem, so ``from_db(atlas=…)`` finds none, ``from_platform`` refuses it naming the spelling the platform lists that matches it in any case, and ``list_platform_networks`` lists nothing for it."""
    with pytest.raises(FileNotFoundError, match="No network matching"):
        Network.from_db(atlas="hcpmmp1")
    with pytest.raises(ValueError) as from_platform:
        Network.from_platform(atlas="hcpmmp1", base_url=BASE_URL, cache_dir=tmp_path)
    assert str(from_platform.value) == "No network found for atlas=hcpmmp1, tractogram=dTOR; did you mean atlas=HCPMMP1?"
    assert Network.list_platform_networks(base_url=BASE_URL, atlas="hcpmmp1") == []


@pytest.mark.parametrize(
    ("query", "refusal"),
    [
        (
            {"atlas": "HCPMMP1", "tractogram": "dtor"},
            "No network found for atlas=HCPMMP1, tractogram=dtor; did you mean tractogram=dTOR?",
        ),
        ({"atlas": "NoSuchAtlas"}, "No network found for atlas=NoSuchAtlas, tractogram=dTOR."),
    ],
)
def test_a_query_nothing_answers_is_refused_naming_the_spellings_that_match_in_any_case(client, tmp_path, query, refusal):
    with pytest.raises(ValueError) as refused:
        Network.from_platform(base_url=BASE_URL, cache_dir=tmp_path, **query)
    assert str(refused.value) == refusal


@pytest.mark.parametrize("query", [{"atlas": "HCPMMP1", "tractogram": "dtor"}, {"atlas": "NoSuchAtlas"}])
def test_a_listing_nothing_answers_is_empty(client, query):
    """``list_platform_networks`` lists nothing for a query the platform lists no network for, as ``Network.list_db`` lists nothing for entities no network carries."""
    assert Network.list_platform_networks(base_url=BASE_URL, **query) == []
