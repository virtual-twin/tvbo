"""``Network.from_platform`` hands out the one network the platform lists for an atlas and a tractogram, and refuses several, naming them.

tvbo hands out no archived data (register AR1), so the platform's listing is read with the networks the BIDS rename archived left out (register AR1-a), and an archived network is refused, by the id or the ``bids_filename`` the platform lists it under or by the name its sidecar's BIDS entities spell, with the refusal a lookup of its stem gets: before anything is written to the cache, and again when the cache holds it from an earlier download. An atlas is read as ``Network.from_db`` reads one. ``requests.get`` is replaced by a stand-in platform that serves the curated records, so no test calls the network.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import requests
import yaml

from tvbo import Network
from tvbo.classes.network import NETWORK_DIR
from tvbo.data.cohort import BIDS_ENTITY_SHORT_KEYS
from tvbo.data.registry import renamed_stems, resolve, stem_entities

BASE_URL = "https://platform.invalid"
API = f"{BASE_URL}/api/v1/networks"
ARCHIVED = renamed_stems()["archived"]["networks"]
SPELLED = [stem for stem in ARCHIVED if stem_entities(stem)]
"""The archived networks whose stems are BIDS names, which a sidecar's entities can spell."""
BIDS_SLOTS = {short: slot for slot, short in BIDS_ENTITY_SHORT_KEYS.items()}
HCPMMP1 = "tpl-MNI152NLin2009bAsym_cohort-HCPYA_rec-dTOR_atlas-HCPMMP1_desc-SC_relmat"
ROUTES = [
    *(pytest.param(stem, stem, None, id=f"listed-by-its-stem-{stem}") for stem in ARCHIVED),
    *(pytest.param(stem, "42", f"{stem}.h5", id=f"listed-by-its-file-name-{stem}") for stem in ARCHIVED),
    *(pytest.param(stem, "42", None, id=f"spelled-by-its-sidecar-{stem}") for stem in SPELLED),
]
"""Each archived network as the platform lists it: by its stem; under another id with its file name as the listing's ``bids_filename``; or, for one whose stem its sidecar's entities spell, under another id alone."""


def _sidecar(stem):
    """The sidecar of the curated network *stem* as the platform serves it."""
    return yaml.safe_load((NETWORK_DIR / f"{stem}.yaml").read_text(encoding="utf-8"))


def _archived_sidecar(stem, file_name=None):
    """The sidecar of the archived network *stem* as a platform that kept it serves it: the network that replaced it, with the BIDS entities *stem* spells, or with no ``bids:`` block where the listing gives the archived network's *file_name*, so that the file name alone tells it."""
    sidecar = _sidecar(ARCHIVED[stem])
    if file_name:
        sidecar.pop("bids", None)
    elif entities := stem_entities(stem):
        sidecar["bids"] = {BIDS_SLOTS[key]: value for key, value in entities.items() if key not in ("atlas", "desc")}
    return sidecar


def _refusal(stem):
    """The refusal a lookup of the archived *stem* gets."""
    with pytest.raises(ValueError) as refused:
        resolve("Network", stem)
    return str(refused.value)


class _Platform:
    """A stand-in for the platform's network API that lists the ids *listed*, whatever it is asked, each with its ``bids_filename`` from *files* where that names one, and serves *sidecar* and the companion it names, keeping each URL asked for."""

    def __init__(self, monkeypatch, listed, sidecar, files=None):
        self.listed, self.sidecar, self.files, self.asked = listed, sidecar, files or {}, []
        monkeypatch.setattr(requests, "get", self.get)

    def get(self, url, params=None):
        self.asked.append(url)
        if url == API:
            payload = SimpleNamespace(
                json=lambda: [{"id": net_id, "bids_filename": self.files.get(net_id)} for net_id in self.listed]
            )
        elif url.endswith("/sidecar"):
            payload = SimpleNamespace(text=yaml.safe_dump(self.sidecar, sort_keys=False))
        else:
            payload = SimpleNamespace(content=(NETWORK_DIR / self.sidecar["data_file"]).read_bytes())
        payload.raise_for_status = lambda: None
        return payload


def test_the_one_network_the_platform_lists_is_downloaded_once_then_read_from_the_cache(monkeypatch, tmp_path):
    platform = _Platform(monkeypatch, [HCPMMP1], _sidecar(HCPMMP1))
    first = Network.from_platform(atlas="HCPMMP1", base_url=BASE_URL, cache_dir=tmp_path)
    again = Network.from_platform(atlas="HCPMMP1", base_url=BASE_URL, cache_dir=tmp_path)
    own = Network(iri=f"tvbo:networks/{HCPMMP1}")
    assert first.node_labels == again.node_labels == own.node_labels
    np.testing.assert_array_equal(again.matrix("weight"), own.matrix("weight"))
    assert platform.asked == [API, f"{API}/{HCPMMP1}/sidecar", f"{API}/{HCPMMP1}/data", API]
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"{HCPMMP1}.h5", f"{HCPMMP1}.yaml"]


def test_several_networks_the_platform_lists_are_refused_naming_them(monkeypatch, tmp_path):
    listed = [
        f"tpl-MNI152NLin6Asym_cohort-HCPYA_rec-dTOR_atlas-Schaefer2018_seg-{seg}_scale-100_desc-SC_relmat"
        for seg in ("7Networks", "17Networks")
    ]
    platform = _Platform(monkeypatch, listed, _sidecar(listed[0]))
    with pytest.raises(ValueError) as refused:
        Network.from_platform(atlas="Schaefer2018", base_url=BASE_URL, cache_dir=tmp_path / "cache")
    assert str(refused.value) == (
        f"Several networks found for atlas=Schaefer2018, tractogram=dTOR: {listed[0]}, {listed[1]}; "
        "`from_platform` tells networks apart by atlas and tractogram alone, so load the one meant by its IRI, "
        '`Network(iri="tvbo:networks/<id>")`, from a tvbo installation that holds its data file.'
    )
    assert platform.asked == [API]
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize(("stem", "listed", "file_name"), ROUTES)
def test_an_archived_network_is_refused_before_anything_is_cached(monkeypatch, tmp_path, stem, listed, file_name):
    """Listed by its stem or its file name, an archived network is refused before its sidecar is asked for; listed under another id alone, once its sidecar's BIDS entities spell its stem, before its companion is asked for. Neither writes to the cache."""
    platform = _Platform(monkeypatch, [listed], _archived_sidecar(stem, file_name), {listed: file_name})
    with pytest.raises(ValueError) as refused:
        Network.from_platform(atlas="any", base_url=BASE_URL, cache_dir=tmp_path / "cache")
    assert str(refused.value) == _refusal(stem)
    assert platform.asked == ([API] if file_name or listed == stem else [API, f"{API}/{listed}/sidecar"])
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize(("stem", "listed", "file_name"), ROUTES)
def test_an_archived_network_the_cache_holds_is_refused_without_a_download(monkeypatch, tmp_path, stem, listed, file_name):
    """A cache that holds an archived network from an earlier download, under its stem or under another id, hands it out no more: it is refused by the id or the file name the listing gives it, or by the name its cached sidecar's entities spell, and nothing is downloaded."""
    sidecar = _archived_sidecar(stem, file_name)
    (tmp_path / f"{listed}.h5").symlink_to(NETWORK_DIR / sidecar["data_file"])
    (tmp_path / f"{listed}.yaml").write_text(
        yaml.safe_dump({**sidecar, "data_file": f"{listed}.h5"}, sort_keys=False), encoding="utf-8"
    )
    platform = _Platform(monkeypatch, [listed], sidecar, {listed: file_name})
    with pytest.raises(ValueError) as refused:
        Network.from_platform(atlas="any", base_url=BASE_URL, cache_dir=tmp_path)
    assert str(refused.value) == _refusal(stem)
    assert platform.asked == [API]


def test_an_archived_network_listed_beside_the_one_that_replaced_it_is_left_out(monkeypatch, tmp_path):
    """The listing is read with archived networks left out (register AR1-a), so of an archived network and the one that replaced it, listed side by side, the replacement is the one network listed and is handed out."""
    old = next(stem for stem, new in ARCHIVED.items() if new == HCPMMP1)
    platform = _Platform(monkeypatch, [old, HCPMMP1], _sidecar(HCPMMP1))
    net = Network.from_platform(atlas="HCPMMP1", base_url=BASE_URL, cache_dir=tmp_path)
    assert net.node_labels == Network(iri=f"tvbo:networks/{HCPMMP1}").node_labels
    assert platform.asked == [API, f"{API}/{HCPMMP1}/sidecar", f"{API}/{HCPMMP1}/data"]


def test_the_platform_listing_leaves_archived_networks_out(monkeypatch):
    """``list_platform_networks`` lists no network the BIDS rename archived, whether the platform lists it by its stem or by its ``bids_filename`` under another id (register AR1-a), and lists nothing when the platform lists archived networks alone and nothing was asked for."""
    by_stem, by_file = list(ARCHIVED)[:2]
    platform = _Platform(monkeypatch, [by_stem, "42", HCPMMP1], _sidecar(HCPMMP1), {"42": f"{by_file}.h5"})
    assert [entry["id"] for entry in Network.list_platform_networks(base_url=BASE_URL)] == [HCPMMP1]
    platform.listed = [by_stem, "42"]
    assert Network.list_platform_networks(base_url=BASE_URL) == []
    assert platform.asked == [API, API]


def test_a_query_the_platform_answers_with_archived_networks_alone_is_refused(monkeypatch):
    """Asked for an atlas, ``list_platform_networks`` refuses a listing of archived networks alone, by stem or by file name, with the sentence a lookup of each one's stem gets, as ``from_platform`` does."""
    by_stem, by_file = list(ARCHIVED)[:2]
    _Platform(monkeypatch, [by_stem, "42"], _sidecar(HCPMMP1), {"42": f"{by_file}.h5"})
    with pytest.raises(ValueError) as refused:
        Network.list_platform_networks(base_url=BASE_URL, atlas="any")
    assert str(refused.value) == f"{_refusal(by_stem)} {_refusal(by_file)}"


@pytest.mark.parametrize("spelling", ["Schaefer1000", "hcpmmp1ordered"])
def test_an_older_spelling_of_archived_networks_is_refused_before_the_platform_is_asked(monkeypatch, tmp_path, spelling):
    """``from_platform`` and ``list_platform_networks`` read an atlas as ``from_db`` reads one, so an older spelling that names archived networks gets ``from_db``'s refusal, and the platform is not asked."""
    platform = _Platform(monkeypatch, [HCPMMP1], _sidecar(HCPMMP1))
    with pytest.raises(ValueError) as from_platform:
        Network.from_platform(atlas=spelling, base_url=BASE_URL, cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError) as from_db_rec:
        Network.from_db(atlas=spelling, rec="dTOR")
    with pytest.raises(ValueError) as listed:
        Network.list_platform_networks(base_url=BASE_URL, atlas=spelling)
    with pytest.raises(ValueError) as from_db:
        Network.from_db(atlas=spelling)
    assert str(from_platform.value) == str(from_db_rec.value)
    assert str(listed.value) == str(from_db.value)
    assert platform.asked == []
    assert not (tmp_path / "cache").exists()
