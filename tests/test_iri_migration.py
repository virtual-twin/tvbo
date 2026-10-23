"""``tvbo migrate-iris`` rewrites curated-record IRIs into their folder form and leaves every other byte alone."""

from __future__ import annotations

from pathlib import Path

import pytest

from tvbo.data.iri_migration import migrate_text

pytestmark = pytest.mark.backend_core


def test_a_singular_scope_takes_its_folder_and_nothing_else_moves():
    text = "# keep this comment\npipeline:\n  - iri: 'tvbo:observation/bold_tvb'  # and this one\nused: {iri: tvbo:result/S/fig}\n"
    migrated, changes, _ = migrate_text(text)

    assert migrated == text.replace("tvbo:observation/bold_tvb", "tvbo:observation_models/bold_tvb")
    assert [(c.line, c.new) for c in changes] == [(3, "tvbo:observation_models/bold_tvb")], (
        "a run's output is not a curated record"
    )


def test_an_unscoped_name_takes_the_one_folder_that_files_it():
    migrated, _, _ = migrate_text("dynamics:\n  iri: tvbo:JansenRit\n")

    assert migrated == "dynamics:\n  iri: tvbo:models/JansenRit\n"


def test_the_slot_decides_a_name_several_folders_file():
    """``Linear`` is both a model and a coupling function; the slot the IRI sits in says which."""
    text = "dynamics: {iri: tvbo:Linear}\nnetwork:\n  coupling:\n    c1: {iri: tvbo:Linear}\n"
    migrated, _, _ = migrate_text(text)

    assert (
        migrated
        == "dynamics: {iri: tvbo:models/Linear}\nnetwork:\n  coupling:\n    c1: {iri: tvbo:coupling_functions/Linear}\n"
    )


def test_a_name_nothing_places_is_reported_and_left_alone():
    text = "- tvbo:Linear\n"
    migrated, changes, _ = migrate_text(text)

    assert migrated == text
    assert changes[0].new is None and "no slot says which" in changes[0].reason


def test_a_name_that_is_also_a_schema_term_needs_its_document_class():
    """``Star`` is a graph generator and a permissible value; the envelope says which this document means."""
    bare = "iri: tvbo:Star\n"
    enveloped = "tvbo_class: tvbo:GraphGenerator\niri: tvbo:Star\n"

    assert migrate_text(bare)[0] == bare
    assert migrate_text(enveloped)[0] == "tvbo_class: tvbo:GraphGenerator\niri: tvbo:graph_generators/Star\n"


def test_a_name_no_folder_files_is_not_a_record():
    text = "tvbo_class: tvbo:Network\nparameters:\n  a: {iri: tvbo:SomeOntologyTerm}\n"
    migrated, changes, _ = migrate_text(text)

    assert (migrated, changes) == (text, [])


def test_a_page_rewrites_its_fenced_yaml_only(tmp_path: Path):
    page = "Name it as `tvbo:theme/default` in prose.\n\n```yaml\ntheme: {iri: tvbo:theme/default}\n```\n\n```python\nx = 'tvbo:theme/default'\n```\n"
    migrated, changes, skipped = migrate_text(page, tmp_path / "page.qmd")

    assert migrated == page.replace("{iri: tvbo:theme/default}", "{iri: tvbo:themes/default}")
    assert [c.line for c in changes] == [4]
    assert skipped == []


def test_a_reference_in_a_value_takes_the_one_folder_that_files_it():
    """A generator argument that names a network is a ``value``, which places nothing, so the name's own folder does."""
    text = "graph_generator:\n  parameters:\n    source: {value: 'tvbo:tpl-MNI152NLin2009cAsym_cohort-HCPYA_rec-dTOR_atlas-DesikanKilliany_desc-SC_relmat'}\n"
    migrated, _, _ = migrate_text(text)

    assert migrated == text.replace("tvbo:tpl-", "tvbo:networks/tpl-")


def test_an_iri_spelled_with_escapes_is_reported_not_claimed_as_rewritten():
    text = 'dynamics:\n  iri: "tvbo:Jansen\\x52it"\n'
    migrated, changes, _ = migrate_text(text)

    assert migrated == text
    assert changes[0].new is None and "escapes" in changes[0].reason


def test_a_hidden_directory_is_not_searched(tmp_path: Path):
    from tvbo.data.iri_migration import spec_files

    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "record.yaml").write_text("iri: tvbo:JansenRit\n")
    (tmp_path / "spec.yaml").write_text("iri: tvbo:JansenRit\n")

    assert spec_files([tmp_path]) == [tmp_path / "spec.yaml"]


def test_a_record_named_below_a_comment_header_is_placed():
    """``bold_balloon.yaml`` opens with 18 lines of comment; its name is still the one the record carries."""
    migrated, _, _ = migrate_text("observations:\n  bold: {iri: tvbo:BOLD_Balloon}\n")

    assert migrated == "observations:\n  bold: {iri: tvbo:observation_models/bold_balloon}\n"


def test_a_record_named_by_its_name_takes_its_file_stem():
    """The registry resolves a record's ``name:`` and its stem in any case, but only the stem is the term the ontology mints."""
    text = "dynamics: {iri: tvbo:models/ModelJansen1995}\nobservations:\n  bold: {iri: tvbo:observation_models/BOLD_TVB}\n"
    migrated, _, _ = migrate_text(text)

    assert (
        migrated == "dynamics: {iri: tvbo:models/Jansen1995}\nobservations:\n  bold: {iri: tvbo:observation_models/bold_tvb}\n"
    )


def test_a_network_named_by_its_atlas_takes_the_connectome_it_resolves_to():
    migrated, _, _ = migrate_text("network: {iri: tvbo:DesikanKilliany}\n")

    assert (
        migrated
        == "network: {iri: tvbo:networks/tpl-MNI152NLin2009cAsym_cohort-HCPYA_rec-dTOR_atlas-DesikanKilliany_desc-SC_relmat}\n"
    )


def test_a_canonical_iri_and_one_naming_no_record_are_left_alone():
    text = "dynamics: {iri: tvbo:models/JansenRit}\nnetwork: {iri: tvbo:networks/none}\n"

    assert migrate_text(text)[:2] == (text, [])


def test_a_name_two_folders_resolve_in_an_untyped_slot_is_reported():
    """``connectome`` is a string; ``DesikanKilliany`` is an atlas by name and the connectome filed under that atlas, so nothing says which."""
    text = "network: {connectome: tvbo:DesikanKilliany}\n"
    migrated, changes, _ = migrate_text(text)

    assert migrated == text
    assert changes[0].new is None and "atlases, networks each resolve" in changes[0].reason
