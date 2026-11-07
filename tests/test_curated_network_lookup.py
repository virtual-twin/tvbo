"""The curated network a declaration names by its BIDS entities is the one whose file name carries exactly those entities: ``scale-100`` is never ``scale-1000``, nor ``atlas-Lobar`` ``atlas-Lobar8``.

An entity a declaration leaves out matches any value or none, so a declaration several curated networks carry is refused, naming them and the entities that tell them apart, and an atlas spelled as older records spell it names the network it named when the files were renamed, or is refused, naming the network that replaced it, where the BIDS rename archived that network. ``get_normative_connectome_data`` names a network as a declaration does, by its other entities as well, or by its IRI. A surface network holds a mesh and no connectome, so asking it for one is refused saying so, naming the network whose nodes are its regions. A network that lists its nodes is its own, and its ``parcellation`` and ``bids`` are metadata that look nothing up; a node count is a size, so a network that states one names the curated network its parcellation names. A network that names a curated network, by its parcellation's entities or by an ``iri``, which names one whatever else the network declares, is refused when the name matches none, naming what was asked and the curated names nearest to it, and an ``iri`` beside a file source is refused, naming both.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from tvbo import Network
from tvbo.behaviour import network as network_module
from tvbo.behaviour.network import _find_network_sidecar
from tvbo.classes.network import NETWORK_DIR, get_normative_connectome_data
from tvbo.data.cohort import BIDS_ENTITY_SHORT_KEYS
from tvbo.data.registry import resolve, retired_stems, stem_entities

NAMEABLE = [f for f in sorted(NETWORK_DIR.glob("*.yaml")) if {"atlas", "rec"} <= stem_entities(f.stem).keys()]
WEIGHTED = [f for f in NAMEABLE if any(e.get("label") == "weight" for e in yaml.safe_load(f.read_text()).get("edges") or [])]
BIDS_SLOTS = {short: slot for slot, short in BIDS_ENTITY_SHORT_KEYS.items()}


def _carrying(entities):
    """The curated networks whose file names carry each of *entities*, whatever else they carry."""
    return [f for f in NAMEABLE if stem_entities(f.stem).items() >= entities.items()]


def _one_line(entities):
    """The declaration of BIDS *entities* in one line: the atlas, the tractogram, and every other entity as the network's ``bids``."""
    return {
        "parcellation": {"atlas": {"name": entities["atlas"]}},
        "tractogram": {"name": entities["rec"]},
        "bids": {BIDS_SLOTS[key]: value for key, value in entities.items() if key not in ("atlas", "rec")},
    }


def _named_among(stem):
    """A refusal pattern naming *stem* among the networks it lists."""
    return rf"{re.escape(stem)}[,;]"


ALONE = [f for f in NAMEABLE if _carrying(stem_entities(f.stem)) == [f]]
SCALES = "100, 200, 300, 400, 500, 600, 700, 800, 900, 1000"


def _dense(matrix):
    """*matrix* as a dense array, a sparse layer (scale 1000 is stored csr) included."""
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _said(error: BaseException) -> str:
    """The message *error* carries: the one error a record's validation wraps, or *error*'s own."""
    if isinstance(error, ValidationError) and error.error_count() == 1 and "error" in error.errors()[0].get("ctx", {}):
        return str(error.errors()[0]["ctx"]["error"])
    return str(error)


@pytest.mark.parametrize("tractogram", [{}, {"tractogram": {"name": "dTOR"}}], ids=["no-tractogram", "dTOR"])
@pytest.mark.parametrize("scale", [100, 1000])
@pytest.mark.parametrize("segmentation", ["7Networks", "17Networks"])
def test_a_schaefer_network_is_the_one_at_the_scale_it_names(segmentation, scale, tractogram):
    """Declared in one line, Schaefer2018 at scale 100 is its own 100-node network, never the scale-1000 one whose name contains ``scale-100``."""
    weights, _ = get_normative_connectome_data("Schaefer2018", "dTOR", segmentation=segmentation, scale=str(scale))
    own = Network(
        iri=f"tvbo:networks/tpl-MNI152NLin6Asym_cohort-HCPYA_rec-dTOR_atlas-Schaefer2018_seg-{segmentation}_scale-{scale}_desc-SC_relmat"
    )
    net = Network(
        parcellation={"atlas": {"name": "Schaefer2018"}},
        bids={"segmentation": segmentation, "scale": str(scale)},
        **tractogram,
    )
    assert weights.shape == (scale, scale)
    assert net.number_of_nodes == own.number_of_nodes == scale
    assert net.node_labels == own.node_labels
    np.testing.assert_array_equal(*(_dense(network.matrix("weight")) for network in (net, own)))


def test_an_atlas_is_never_another_whose_name_begins_with_it():
    lobar = Network(parcellation={"atlas": {"name": "Lobar"}}, tractogram={"name": "dTOR"}, bids={"desc": "SC"})
    assert (
        lobar.node_labels
        == Network(iri="tvbo:networks/tpl-MNI152NLin2009cAsym_cohort-HCPYA_rec-dTOR_atlas-Lobar_desc-SC_relmat").node_labels
    )


@pytest.mark.parametrize("desc", ["SC", "surf"])
def test_a_further_entity_tells_apart_two_networks_of_one_atlas_and_tractogram(desc):
    found = _find_network_sidecar({"atlas": "Lobar", "rec": "dTOR", "desc": desc})
    assert stem_entities(found.stem)["desc"] == desc


@pytest.mark.parametrize(
    ("atlas", "bids", "apart"),
    [
        ("Schaefer2018", {}, f"seg- (7Networks, 17Networks) and scale- ({SCALES})"),
        ("Schaefer2018", {"segmentation": "7Networks"}, f"scale- ({SCALES})"),
        ("Schaefer2018", {"scale": "100"}, "seg- (7Networks, 17Networks)"),
        ("Schaefer2018", {"scale": "1000"}, "seg- (7Networks, 17Networks)"),
        ("Lobar", {}, "tpl- (fsLR, MNI152NLin2009cAsym) and desc- (SC, surf) and den- (32k, none)"),
    ],
    ids=[
        "atlas-alone",
        "segmentation-without-scale",
        "scale-100-without-segmentation",
        "scale-1000-without-segmentation",
        "two-descs",
    ],
)
def test_a_declaration_several_curated_networks_carry_is_refused_naming_them(atlas, bids, apart):
    """An entity a declaration leaves out matches any value or none, so Schaefer2018 without its segmentation or its scale, and Lobar without its ``desc``, name no one network: the declaration is refused, naming every network that carries it by its IRI and the entities that tell them apart with their values."""
    asked = {**{BIDS_ENTITY_SHORT_KEYS[slot]: value for slot, value in bids.items()}, "atlas": atlas, "rec": "dTOR"}
    found = _carrying(asked)
    with pytest.raises(ValueError) as refusal:
        Network(parcellation={"atlas": {"name": atlas}}, **({"bids": bids} if bids else {}))
    assert len(found) > 1
    assert _said(refusal.value) == (
        f"Several curated networks have the BIDS entities {', '.join(f'{key}-{value}' for key, value in asked.items())}, told apart by {apart}: "
        f"{', '.join(f'tvbo:networks/{f.stem}' for f in found)}; state a further entity of the one meant, or name it by its `iri`."
    )


OLDER = {
    "Schaefer1000": "tpl-FSLMNI152_cohort-{cohort}_rec-{rec}_atlas-Schaefer2018_scale-1000_desc-SC_relmat",
    "Schaefer100017Networks": "tpl-FSLMNI152_cohort-{cohort}_rec-{rec}_atlas-Schaefer2018_seg-17Networks_scale-1000_desc-SC_relmat",
    "hcpmmp1": "tpl-MNI152NLin2009bAsym_cohort-{cohort}_rec-{rec}_atlas-HCPMMP1_desc-SC_relmat",
    "hcpmmp1ordered": "tpl-MNI152NLin2009bAsym_cohort-{cohort}_rec-{rec}_atlas-HCPMMP1_seg-ordered_desc-SC_relmat",
    "DesikanKillianyranked": "tpl-MNI152NLin2009cAsym_cohort-{cohort}_rec-{rec}_atlas-DesikanKilliany_desc-SC_relmat",
    "Destrieuxranked": "tpl-MNI152NLin2009cAsym_cohort-{cohort}_rec-{rec}_atlas-Destrieux_desc-SC_relmat",
}
"""Each older spelling of an atlas, to the network it named before the BIDS rename."""
COHORTS = {"dTOR": "HCPYA", "MghUscHcp32": "MghUscHcp32", "PPMI85": "PPMI85"}
ARCHIVED = {"Schaefer1000", "hcpmmp1ordered"}
"""The spellings whose network the BIDS rename archived, for every tractogram."""


def test_every_older_spelling_of_an_atlas_is_checked():
    assert set(OLDER) == set(network_module._LEGACY_ATLAS_NAMES)


@pytest.mark.parametrize("rec", sorted(COHORTS))
@pytest.mark.parametrize("older", sorted(set(OLDER) - ARCHIVED))
def test_an_older_spelling_of_an_atlas_reaches_the_network_it_named(older, rec):
    """Each spelling older records use whose network the BIDS rename kept names the network it named when the files were renamed (``scripts/rename_networks.py``), under the name the BIDS rename gave it, with no warning and never a refusal for the networks beside it; the ``ranked`` spellings name the only Desikan-Killiany and Destrieux networks, which the removed ``seg-ranked`` ones duplicated."""
    named = OLDER[older].format(cohort=COHORTS[rec], rec=rec)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        found = _find_network_sidecar({"atlas": older, "rec": rec})
    assert found.stem == retired_stems("networks").get(named, named)
    assert not [w for w in caught if w.category is FutureWarning]


@pytest.mark.parametrize("rec", sorted(COHORTS))
@pytest.mark.parametrize("older", sorted(ARCHIVED))
def test_an_older_spelling_of_an_archived_network_is_refused_as_its_stem_is(older, rec):
    """A spelling that named a network the BIDS rename archived (``Schaefer1000`` the Schaefer2018 scale-1000 network without ``seg-``, ``hcpmmp1ordered`` the ``seg-ordered`` HCPMMP1 network) is refused, since tvbo hands out no archived data (register AR1): by the registry, by a lookup of its entities, by ``Network.from_db`` of them and by a network's ``parcellation``, with the refusal a lookup of the archived network's stem gets, which names the network that replaced it."""
    named = OLDER[older].format(cohort=COHORTS[rec], rec=rec)
    with pytest.raises(ValueError) as by_stem:
        resolve("Network", named)
    assert str(by_stem.value).startswith(f"Network {named!r} was archived")
    for lookup in (
        lambda: resolve("Network", older, rec=rec),
        lambda: _find_network_sidecar({"atlas": older, "rec": rec}),
        lambda: Network.from_db(atlas=older, rec=rec),
        lambda: Network(parcellation={"atlas": {"name": older}}, tractogram={"name": rec}),
    ):
        with pytest.raises(ValueError) as by_spelling:
            lookup()
        assert _said(by_spelling.value) == str(by_stem.value)


@pytest.mark.parametrize("older", sorted(ARCHIVED))
def test_an_older_spelling_several_archived_networks_carry_is_refused_naming_each(older):
    """With no tractogram, ``Schaefer1000`` and ``hcpmmp1ordered`` name the archived network of every cohort, so a lookup of the spelling's entities, and ``Network.from_db`` of them, is refused with the refusals a lookup of each of those stems gets, one after another, each saying that its network was archived and naming the one that replaced it."""
    refusals = []
    for named in sorted(OLDER[older].format(cohort=cohort, rec=rec) for rec, cohort in COHORTS.items()):
        with pytest.raises(ValueError) as by_stem:
            resolve("Network", named)
        refusals.append(str(by_stem.value))
    for lookup in (lambda: _find_network_sidecar({"atlas": older}), lambda: Network.from_db(atlas=older)):
        with pytest.raises(ValueError) as by_spelling:
            lookup()
        assert str(by_spelling.value) == " ".join(refusals)


def test_a_network_declared_by_an_older_spelling_reads_the_network_it_named():
    net = Network(parcellation={"atlas": {"name": "hcpmmp1"}})
    own = Network(iri=f"tvbo:networks/{OLDER['hcpmmp1'].format(cohort='HCPYA', rec='dTOR')}")
    assert net.node_labels == own.node_labels
    np.testing.assert_array_equal(_dense(net.matrix("weight")), _dense(own.matrix("weight")))


def test_an_older_spelling_beside_an_entity_it_names_otherwise_names_no_network():
    """``Schaefer1000`` names the Schaefer2018 scale-1000 network without ``seg-``, so a ``seg`` stated beside it is refused, not read as another network."""
    with pytest.raises(
        FileNotFoundError, match=r"^No curated network has the BIDS entities seg-7Networks, atlas-Schaefer1000, rec-dTOR;"
    ):
        Network(parcellation={"atlas": {"name": "Schaefer1000"}}, bids={"segmentation": "7Networks"})


LOBAR = {
    "SC": "tpl-MNI152NLin2009cAsym_cohort-HCPYA_rec-dTOR_atlas-Lobar_desc-SC_relmat",
    "surf": "tpl-fsLR_cohort-HCPYA_rec-dTOR_atlas-Lobar_den-32k_desc-surf_relmat",
}
SURFACE_REFUSAL = rf"^{LOBAR['surf']}\.h5 holds a surface mesh of 64984 vertices and no connectome; its regions are the nodes of tvbo:networks/{LOBAR['SC']}\.$"


def _lobar(route, desc):
    """The arguments that name the Lobar network of *desc* to :func:`get_normative_connectome_data`, by its entities or by its IRI."""
    if route == "entities":
        return {"atlas": "Lobar", "tractogram": "dTOR", "desc": desc}
    return {"iri": f"tvbo:networks/{LOBAR[desc]}"}


def test_the_normative_connectome_of_entities_several_networks_carry_is_refused_naming_how_to_choose():
    with pytest.raises(
        ValueError,
        match=rf"told apart by tpl- \(fsLR, MNI152NLin2009cAsym\) and desc- \(SC, surf\) and den- \(32k, none\): tvbo:networks/{LOBAR['SC']}, tvbo:networks/{LOBAR['surf']}; state a further entity of the one meant, or name it by its `iri`\.$",
    ):
        get_normative_connectome_data("Lobar", "dTOR")


@pytest.mark.parametrize("route", ["entities", "iri"])
def test_the_normative_connectome_function_picks_each_lobar_network(route):
    """A ``desc`` or the IRI picks each of the two Lobar networks: the connectome gives its weights, and the surface network, a mesh with none, is refused as one, naming the network whose nodes are its regions."""
    weights, _ = get_normative_connectome_data(**_lobar(route, "SC"))
    stored = Network(iri=f"tvbo:networks/{LOBAR['SC']}").matrix("weight", apply_transforms=False)
    np.testing.assert_array_equal(_dense(weights), _dense(stored))
    with pytest.raises(ValueError, match=SURFACE_REFUSAL):
        get_normative_connectome_data(**_lobar(route, "surf"))


def test_a_network_whose_parcellation_names_a_surface_network_is_refused_as_a_mesh_naming_the_network_of_its_regions():
    with pytest.raises(ValueError) as refusal:
        Network(parcellation={"atlas": {"name": "Lobar"}}, tractogram={"name": "dTOR"}, bids={"desc": "surf"})
    assert re.match(SURFACE_REFUSAL, _said(refusal.value))


@pytest.mark.parametrize(
    ("arguments", "refusal", "message"),
    [
        ({}, TypeError, r"names a curated network by its `atlas` or by its `iri`"),
        (
            {"atlas": "Lobar", "iri": f"tvbo:networks/{LOBAR['SC']}"},
            ValueError,
            r"names the curated network on its own; drop the BIDS entities beside it: atlas-Lobar\.",
        ),
        ({"atlas": "Lobar", "colour": "red"}, TypeError, r"takes the BIDS entities .* by name, not \['colour'\]"),
    ],
    ids=["neither", "iri-beside-an-atlas", "unknown-entity"],
)
def test_the_normative_connectome_function_refuses_arguments_that_name_no_one_network(arguments, refusal, message):
    with pytest.raises(refusal, match=message):
        get_normative_connectome_data(**arguments)


@pytest.mark.parametrize("form", ["every-entity", "one-line"])
@pytest.mark.parametrize("sidecar", NAMEABLE, ids=lambda f: f.stem)
def test_every_curated_network_is_found_by_its_own_entities(sidecar, form):
    """Named by every entity of its file name, or by its atlas, rec, seg and scale alone, it is itself when no other curated network carries those, and is refused, named among those that do, when another does."""
    entities = stem_entities(sidecar.stem)
    if form == "one-line":
        entities = {key: value for key, value in entities.items() if key in ("atlas", "rec", "seg", "scale")}
    if _carrying(entities) == [sidecar]:
        assert _find_network_sidecar(entities) == sidecar
    else:
        with pytest.raises(ValueError, match=_named_among(sidecar.stem)):
            _find_network_sidecar(entities)


DESIKAN = "tpl-MNI152NLin2009cAsym_cohort-HCPYA_rec-dTOR_atlas-DesikanKilliany_desc-SC_relmat"


@pytest.mark.parametrize("bids", [{}, {"desc": "SC"}, {"template": "MNI152NLin2009cAsym"}, {"cohort": "HCPYA"}], ids=str)
def test_a_network_whose_entities_a_curated_network_of_its_atlas_carries_reads_that_network(bids):
    net = Network(parcellation={"atlas": {"name": "DesikanKilliany"}}, **({"bids": bids} if bids else {}))
    assert net.node_labels == Network(iri=f"tvbo:networks/{DESIKAN}").node_labels


@pytest.mark.parametrize(
    ("bids", "stated"),
    [({"desc": "mine"}, "desc-mine"), ({"template": "fsaverage"}, "tpl-fsaverage"), ({"acquisition": "x"}, "acq-x")],
    ids=["desc", "template", "acquisition"],
)
def test_a_network_whose_entities_no_curated_network_of_its_atlas_carries_is_refused_naming_them(bids, stated):
    with pytest.raises(
        FileNotFoundError,
        match=rf"No curated network has the BIDS entities {stated}, atlas-DesikanKilliany, rec-dTOR; the curated networks of the atlas 'DesikanKilliany' and the tractogram 'dTOR' are: tvbo:networks/{DESIKAN}\.",
    ):
        Network(parcellation={"atlas": {"name": "DesikanKilliany"}}, bids=bids)


@pytest.mark.parametrize("sidecar", [f for f in WEIGHTED if f in ALONE], ids=lambda f: f.stem)
def test_every_curated_network_resolves_through_its_one_line_form(sidecar):
    """Its atlas, its tractogram and every other entity of its file name, declared in one line, read the network its iri names."""
    net = Network(**_one_line(stem_entities(sidecar.stem)))
    assert net.node_labels == Network(iri=f"tvbo:networks/{sidecar.stem}").node_labels


@pytest.mark.parametrize(
    ("atlas", "nearest"),
    [("DesikanKiliany", "; the closest curated atlases are: DesikanKilliany"), ("NoSuchAtlas", "")],
    ids=["misspelt", "unknown"],
)
def test_an_atlas_no_curated_network_is_of_is_refused_naming_it_and_the_closest_curated_atlases(atlas, nearest):
    with pytest.raises(
        FileNotFoundError,
        match=rf"No curated network has the BIDS entities atlas-{atlas}, rec-dTOR; no curated network is of the atlas '{atlas}'{nearest}\.$",
    ):
        Network(parcellation={"atlas": {"name": atlas}})


def test_an_atlas_with_no_curated_network_of_the_tractogram_is_refused_naming_the_tractograms_it_has():
    with pytest.raises(
        FileNotFoundError,
        match=r"the atlas 'Schaefer2018' has curated networks of the tractograms Hansen2022, MghUscHcp32, PPMI85, dTOR only\.",
    ):
        Network(parcellation={"atlas": {"name": "Schaefer2018"}}, tractogram={"name": "nosuch"})


def test_a_parcellation_naming_no_atlas_on_a_network_without_nodes_is_refused():
    with pytest.raises(ValueError, match=r"names a parcellation with no atlas, so it names no curated network"):
        Network(parcellation={"label": "mine"})


MISSPELT = DESIKAN.replace("DesikanKilliany", "DesikanKiliany")
THREE_NODES = [{"id": i, "label": f"n{i}"} for i in range(3)]
READ = pytest.mark.parametrize("how", ["constructor", "file"])


def _built(how, tmp_path, **declared):
    """The network *declared* gives, built by the constructor or read from a network file holding it."""
    if how == "constructor":
        return Network(**declared)
    path = tmp_path / "net.yaml"
    path.write_text(yaml.safe_dump({"tvbo_class": "tvbo:Network", **declared}))
    return Network.from_file(path)


@pytest.mark.parametrize(
    ("iri", "nearest"),
    [
        (f"tvbo:networks/{MISSPELT}", rf"; the closest are: .*tvbo:networks/{DESIKAN}"),
        ("tvbo:atlases/tpl-MNI152NLin2009cAsym_atlas-DesikanKilliany_dseg", ""),
        ("tvbo:networks/no-such-network", ""),
    ],
    ids=["misspelt", "another-folder", "unknown"],
)
@pytest.mark.parametrize(
    "declared",
    [{}, {"number_of_nodes": 3}, {"nodes": THREE_NODES}, {"data_file": "elsewhere.h5"}],
    ids=["alone", "beside-a-node-count", "beside-its-own-nodes", "beside-a-data-file"],
)
@READ
def test_an_iri_that_names_no_curated_network_is_refused(iri, nearest, declared, how, tmp_path):
    """An ``iri`` always names a curated network, whatever else the network declares."""
    with pytest.raises(FileNotFoundError, match=rf"No curated network has the IRI '{iri}'{nearest}"):
        _built(how, tmp_path, iri=iri, **declared)


@pytest.mark.parametrize(
    ("slot", "value"), [("data_file", "elsewhere.h5"), ("edge_matrix_files", "weights.csv"), ("bids_dir", "elsewhere")]
)
@READ
def test_an_iri_beside_a_file_source_is_refused_naming_both(slot, value, how, tmp_path):
    """An ``iri`` that names a curated network is read whatever else the network declares, so beside a file source of its own the network names two sources."""
    with pytest.raises(
        ValueError, match=rf"names `iri: tvbo:networks/{DESIKAN}` and `{slot}: {value}` as the source of its connectivity"
    ):
        _built(how, tmp_path, iri=f"tvbo:networks/{DESIKAN}", **{slot: value})


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("declared", [{}, {"number_of_nodes": 5}], ids=["alone", "beside-a-node-count"])
def test_a_network_file_naming_a_curated_network_reads_as_the_constructor_builds_it(declared, tmp_path):
    """A network file's ``iri`` names its source as a recipe's does: the file gives the curated network's nodes and weights, cut to a count it states."""
    read, built = (_built(how, tmp_path, iri=f"tvbo:networks/{DESIKAN}", **declared) for how in ("file", "constructor"))
    assert list(read.node_labels) == list(built.node_labels) and len(read.node_labels) == declared.get("number_of_nodes", 87)
    np.testing.assert_array_equal(_dense(read.matrix("weight")), _dense(built.matrix("weight")))


OWN = pytest.mark.parametrize("declared", [{"nodes": THREE_NODES}], ids=["nodes"])


@pytest.mark.parametrize("atlas", ["DesikanKilliany", "DesikanKiliany"], ids=["curated-atlas", "unknown-atlas"])
@pytest.mark.parametrize("bids", [{}, {"desc": "mine"}], ids=["no-bids", "unmatched-entity"])
@OWN
def test_a_network_declaring_its_own_nodes_looks_nothing_up_for_its_parcellation(monkeypatch, atlas, bids, declared):
    """Its parcellation and its bids entities are metadata: nothing is looked up, so neither an atlas nor an entity no curated network carries is an error, and the network is the nodes it declares."""
    monkeypatch.setattr(network_module, "_find_network_sidecar", lambda entities: pytest.fail(f"looked up {entities}"))
    net = Network(parcellation={"atlas": {"name": atlas}}, **({"bids": bids} if bids else {}), **declared)
    assert net.number_of_nodes == 3 and net.array("edges/weight") is None
    assert str(net.parcellation.atlas.name) == atlas


@OWN
def test_a_network_of_its_own_beside_an_entity_no_curated_network_of_its_atlas_carries_is_built(declared):
    """The refusal of a lookup that matches no curated network is never a network's that lists its nodes: it is built as it declares them."""
    net = Network(parcellation={"atlas": {"name": "DesikanKilliany"}}, bids={"desc": "mine"}, **declared)
    assert net.number_of_nodes == 3


def test_a_node_count_beside_a_curated_atlas_keeps_that_many_of_its_networks_nodes():
    """A count is a size, not a network of its own: the curated network the parcellation names is looked up and keeps three of its nodes, with their weights."""
    with pytest.warns(UserWarning, match=r"DesikanKilliany: keeps 3 of its 87 nodes\."):
        net = Network(parcellation={"atlas": {"name": "DesikanKilliany"}}, number_of_nodes=3)
    curated = Network(iri=f"tvbo:networks/{DESIKAN}")
    at = [list(curated.node_labels).index(label) for label in net.node_labels]
    assert net.number_of_nodes == 3 and at == sorted(at)
    np.testing.assert_array_equal(_dense(net.matrix("weight")), _dense(curated.matrix("weight"))[np.ix_(at, at)])


@pytest.mark.parametrize(
    ("atlas", "bids", "refusal"),
    [
        (
            "DesikanKilliany",
            {"desc": "mine"},
            r"No curated network has the BIDS entities desc-mine, atlas-DesikanKilliany, rec-dTOR",
        ),
        ("DesikanKiliany", {}, r"no curated network is of the atlas 'DesikanKiliany'"),
        ("DesikanKiliany", {"desc": "mine"}, r"no curated network is of the atlas 'DesikanKiliany'"),
    ],
    ids=["unmatched-entity-curated-atlas", "no-bids-unknown-atlas", "unmatched-entity-unknown-atlas"],
)
def test_a_node_count_beside_a_parcellation_no_curated_network_matches_is_refused(atlas, bids, refusal):
    """A count declares no network of its own, so the parcellation it sits beside is looked up, and a lookup that matches no curated network is refused."""
    with pytest.raises(FileNotFoundError, match=refusal):
        Network(parcellation={"atlas": {"name": atlas}}, **({"bids": bids} if bids else {}), number_of_nodes=3)


def test_a_network_listing_its_nodes_beside_a_curated_atlas_keeps_them_and_the_matrix_set_on_them():
    """The curated atlas's network does not replace the nodes the network lists, so a weight matrix set on them is the side of the network."""
    net = Network(nodes=THREE_NODES, parcellation={"atlas": {"name": "DesikanKilliany"}})
    net.set_matrix("weight", np.ones((3, 3)) - np.eye(3))
    assert [node.label for node in net.nodes] == ["n0", "n1", "n2"] and net.number_of_nodes == 3
    assert net.matrix("weight").shape == (3, 3)
