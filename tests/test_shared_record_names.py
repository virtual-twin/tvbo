"""A name several curated records answer to is refused, naming each by its IRI and the entities that tell them apart, and BIDS entities narrow it to one (register I11i); an old stem resolves to its renamed record with a warning naming the new IRI, and an archived record's stem is refused, naming the record that replaced it, since tvbo hands out no archived data (register AR1).

A refusal is not "no such record": wherever a lookup reads ``FileNotFoundError`` as a plain reference (register I2), the refusal still reaches the caller, and nothing caches it as an absence.
"""

from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tvbo import Network
from tvbo.classes.atlas import Atlas
from tvbo.data import registry

pytestmark = pytest.mark.backend_core

DK_SC = "tpl-MNI152NLin2009cAsym_cohort-HCPYA_rec-dTOR_atlas-DesikanKilliany_desc-SC_relmat"


def _table() -> dict[str, dict[str, dict[str, str]]]:
    """The stem table as written: ``renamed``, ``archived`` and ``differs``, each per database folder."""
    return yaml.safe_load(Path(registry.__file__).with_name("renamed_stems.yaml").read_text(encoding="utf-8"))


def _retired() -> dict[str, dict[str, str]]:
    """Every stem the table retires, renamed or archived, to the stem of the record that has its name now or replaced it, per database folder."""
    table = _table()
    return {folder: registry.retired_stems(folder) for folder in {*table["renamed"], *table["archived"]}}


def test_an_atlas_name_twenty_atlases_share_is_refused():
    with pytest.raises(ValueError, match="Schaefer2018") as refused:
        registry.resolve("BrainAtlas", "Schaefer2018")
    message = str(refused.value)
    assert message.count("tvbo:atlases/") == 20
    assert "seg-" in message and "scale-" in message


def test_entities_narrow_a_shared_atlas_name_to_one():
    path = registry.resolve("BrainAtlas", "Schaefer2018", seg="7Networks", scale="100")
    assert path.stem == "tpl-MNI152NLin6Asym_atlas-Schaefer2018_seg-7Networks_scale-100_res-1_dseg"


def test_atlas_of_refuses_a_shared_name_and_reads_the_narrowed_atlas():
    with pytest.raises(ValueError, match="Schaefer2018"):
        Atlas.of("Schaefer2018")
    assert len(Atlas.of("Schaefer2018", seg="7Networks", scale="100").region_labels) == 100


def test_atlas_of_reads_hcpmmp1_in_any_case():
    upper, lower = Atlas.of("HCPMMP1"), Atlas.of("hcpmmp1")
    assert len(upper.region_labels) == 379
    assert list(upper.region_labels) == list(lower.region_labels)


@pytest.mark.parametrize("atlas", ["Schaefer2018", "Lobar"])
def test_a_network_atlas_several_connectomes_share_is_refused(atlas: str):
    with pytest.raises(ValueError, match="Several curated networks"):
        registry.resolve("Network", atlas)


def test_a_network_atlas_resolves_to_its_hcpya_dtor_connectome():
    assert registry.resolve("Network", "DesikanKilliany").stem == DK_SC


def test_every_renamed_stem_resolves_to_its_new_record_with_a_warning_naming_it():
    """Of every folder the registry resolves; ``coordinate_spaces`` is not one, so its rows record the rename for the ontology alone."""
    wrong = []
    for folder, stems in _table()["renamed"].items():
        if folder not in registry._IRI_SCOPES:
            continue
        cls_name = registry._IRI_SCOPES[folder]
        for old, new in stems.items():
            with pytest.warns(FutureWarning, match=f"tvbo:{folder}/{new}"):
                found = registry.resolve(cls_name, old)
            if found.stem != new:
                wrong.append(f"{folder}/{old} -> {found.stem}, not {new}")
    assert not wrong, wrong


def test_no_record_is_reached_through_the_table():
    """Every stem the table maps to is a record of its folder, read without the table's warning where the registry resolves the folder, and no stem it maps from still is."""
    for folder, stems in _retired().items():
        held = {p.stem for p in (registry.DATABASE_ROOT / folder).rglob("*.yaml")}
        assert not held & set(stems), f"{folder}/ still holds records under old stems: {sorted(held & set(stems))}"
        assert set(stems.values()) <= held, f"{folder}/ holds no record of {sorted(set(stems.values()) - held)}"
        for new in set(stems.values()) if folder in registry._IRI_SCOPES else ():
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                assert registry.resolve(registry._IRI_SCOPES[folder], new).stem == new


@pytest.mark.parametrize("old", sorted(_table()["archived"]["networks"]))
def test_an_archived_stem_is_refused_naming_the_record_that_replaced_it(old: str):
    """An archived network's stem is refused, since tvbo hands out no archived data (register AR1): a lookup of it, in any case, by the registry or by a network's ``iri``, names the record that replaced it and how that record's data differ where the table says they do."""
    table = _table()
    differs = table["differs"]["networks"].get(old)
    refusal = (
        f"Network {old!r} was archived, and tvbo does not hand out archived data; "
        f"the record that replaced it is `tvbo:networks/{table['archived']['networks'][old]}`"
        + (f", {differs}" if differs else "")
        + "."
    )
    for lookup in (
        lambda: registry.resolve("Network", old),
        lambda: registry.resolve("Network", old.lower()),
        lambda: Network(iri=f"tvbo:networks/{old}"),
    ):
        with pytest.raises(ValueError) as refused:
            lookup()
        assert str(refused.value) == refusal


def test_the_stem_table_holds_each_retired_stem_once_in_any_case():
    """``resolve`` reads a retired stem in any case, so a folder of the table spells no two retired stems the same when case-folded."""
    table = _table()
    for folder in {*table["renamed"], *table["archived"]}:
        stems = [*table["renamed"].get(folder, {}), *table["archived"].get(folder, {})]
        folded = Counter(stem.lower() for stem in stems)
        clashes = sorted(stem for stem in stems if folded[stem.lower()] > 1)
        assert not clashes, f"{folder}/ holds retired stems that differ only in case: {clashes}"


def test_no_stem_is_both_renamed_and_archived():
    """``resolve`` refuses a stem listed under ``archived`` before it reads ``renamed``, so a stem in both would be refused although its record was renamed."""
    table = _table()
    for folder in {*table["renamed"], *table["archived"]}:
        both = table["renamed"].get(folder, {}).keys() & table["archived"].get(folder, {}).keys()
        assert not both, f"{folder}/ stems both renamed and archived: {sorted(both)}"


def test_an_ambiguous_network_iri_is_refused_by_network_loading():
    with pytest.raises(ValueError, match="Several curated networks"):
        Network(iri="tvbo:networks/Schaefer2018")


def test_an_ambiguous_network_iri_is_refused_by_a_generators_source():
    from tvbo.graph_generators.catalog import load_matrix

    with pytest.raises(ValueError, match="Several curated networks"):
        load_matrix("tvbo:networks/Lobar")


@pytest.fixture
def twin_database(tmp_path, monkeypatch):
    """A database whose ``graph_generators/`` and ``models/`` each hold two records named ``Twin``."""
    for folder in ("graph_generators", "models"):
        for stem in ("TwinA", "TwinB"):
            (tmp_path / folder).mkdir(exist_ok=True)
            (tmp_path / folder / f"{stem}.yaml").write_text("name: Twin\n", encoding="utf-8")
    monkeypatch.setattr(registry, "DATABASE_ROOT", tmp_path)
    monkeypatch.setattr(Network, "_GG_ENTRY_CACHE", {})
    return tmp_path


def test_a_generator_type_two_records_share_is_refused_and_not_cached(twin_database):
    with pytest.raises(ValueError, match="Twin"):
        Network._db_generator_entry(SimpleNamespace(type="Twin"))
    assert "Twin" not in Network._GG_ENTRY_CACHE


def test_neuroml_export_refuses_a_model_name_two_records_share(twin_database):
    from tvbo.adapters.neuroml import build_lems_context
    from tvbo.classes.dynamics import Dynamics

    with pytest.raises(ValueError, match="Twin"):
        build_lems_context(SimpleNamespace(dynamics=Dynamics(name="Twin")))
