"""The study-layout record is the single ground truth, so these tests guard the record itself.

Three things can go wrong and none of them is caught by any other test: the record can drift away from what BIDS sanctions, the derived ignore files can stop protecting copyrighted material, and a template the record names can go missing from the scaffolder's seeds.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tvbo.cli import app
from tvbo.cli.study import TEMPLATE_DIR
from tvbo.utils import study_layout as layout_rules

REPLICATION = ("replication",)

BIDS_STUDY_ROOT_SUBDIRS = frozenset({"code", "docs", "rawbids", "derivatives", "logs", "sourcedata", "phenotype"})
"""``study.root.subdirs`` from the released BIDS ``rules/directories.yaml`` (1.11.1).

Hardcoded rather than read from a BIDS checkout, which CI does not have. A directory the record
calls ``sanctioned`` must be in here, so claiming BIDS blessing for a name BIDS does not know
fails the build instead of shipping an invalid dataset.
"""


@pytest.fixture(scope="module")
def record():
    return layout_rules.load_layout()


def test_every_directory_role_is_unique(record):
    """A resolver looks a directory up by role, so two directories cannot share one."""
    roles = [str(d.role) for _, d in layout_rules.walk(record) if d.role]
    assert len(roles) == len(set(roles)), "duplicate directory role in the layout record"


def test_sanctioned_directories_are_actually_sanctioned_by_bids(record):
    """``bids: sanctioned`` is a claim about the standard, checked against the standard."""
    for rel, directory in layout_rules.walk(record, REPLICATION):
        if str(directory.bids) != "sanctioned" or "/" in rel:
            continue
        assert rel in BIDS_STUDY_ROOT_SUBDIRS, f"{rel!r} is called sanctioned but BIDS does not define it"


def test_unsanctioned_entries_state_their_expiry(record):
    """An ignore entry without a stated exit becomes permanent by default."""
    for rel, entry in [*layout_rules.walk(record, REPLICATION), *layout_rules.iter_files(record, REPLICATION)]:
        if str(entry.bids) in {"unsanctioned", "proposed"}:
            assert entry.expires_with, f"{rel!r} needs an `expires_with` saying what would retire its entry"


def test_bidsignore_lists_exactly_the_entries_bids_does_not_know(record):
    """A hidden directory and a nested dataset need no line; everything unknown does."""
    lines = [line for line in layout_rules.bidsignore_lines(record, REPLICATION, "S") if not line.startswith("#")]
    assert lines == ["spec/", "prov/", "/S.yaml"]


def test_gitignore_is_the_copyright_gate(record):
    """The generated rules, verbatim. A change here is a change to what may be published."""
    lines = [line for line in layout_rules.gitignore_lines(record, REPLICATION, "S") if not line.startswith("#")]
    assert lines == [
        "sourcedata/*",
        "!sourcedata/README.md",
        "docs/.quarto/",
        "docs/figures/",
        "docs/notes/",
        "derivatives/*",
        "!derivatives/tvbo/",
        "derivatives/tvbo/*",
        "!derivatives/tvbo/dataset_description.json",
        "logs/",
        ".tvbo/",
        "docs/report.pdf",
        "docs/report_internal.pdf",
    ]


@pytest.mark.skipif(shutil.which("git") is None, reason="git is needed to check the ignore rules")
def test_every_required_file_survives_the_gate(record, tmp_path):
    """A file the record demands must be committable, or a fresh checkout cannot validate.

    ``derivatives/tvbo/dataset_description.json`` is the case: the run reproduces everything else under ``derivatives/``, but the description that declares the derivative dataset is asked for before anything has been run.
    """
    rules = layout_rules.gitignore_lines(record, REPLICATION, "S")
    (tmp_path / ".gitignore").write_text("\n".join(rules) + "\n")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    for rel, f in layout_rules.iter_files(record, REPLICATION):
        if str(f.level) != "required":
            continue
        ignored, rule = _ignored(tmp_path, layout_rules.interpolate(rel, "S"))
        assert not ignored, f"required {rel} is ignored by {rule}"


def test_every_named_template_exists(record):
    """The scaffolder fails at run time on a template the record names but does not have."""
    for rel, entry in layout_rules.iter_files(record, REPLICATION):
        if entry.template:
            assert (TEMPLATE_DIR / str(entry.template)).is_file(), f"{rel} names a missing template"


def test_a_template_only_adds(record):
    """The general layout is a subset of every variant, so a template never removes a directory."""
    general = {rel for rel, _ in layout_rules.walk(record)}
    replication = {rel for rel, _ in layout_rules.walk(record, REPLICATION)}
    assert general < replication


def test_role_lookup_raises_rather_than_guessing(record):
    with pytest.raises(KeyError, match="Known roles"):
        layout_rules.relpath("no_such_role", record)


def test_results_and_figures_resolve_where_the_design_puts_them(record):
    assert layout_rules.relpath("results", record) == "derivatives/tvbo"
    assert layout_rules.relpath("figures", record) == "docs/figures"
    assert layout_rules.relpath("kits", record) == ".tvbo/kits"


def test_a_composite_is_staged_under_the_deposit_it_embeds(record):
    """One directory holds everything the publisher owns, so one rule keeps all of it unpublished."""
    stage = layout_rules.relpath("figures_restricted", record)
    assert stage.startswith(layout_rules.relpath("original_study", record) + "/")


def test_the_only_json_the_layout_declares_marks_a_dataset(record):
    """Per-container metadata is the frozen YAML spec, and only that.

    Two metadata files per result are two things to keep in agreement, and the JSON could only ever restate fields the YAML already carries. BIDS asks for JSON in exactly one place — the file that declares a directory to be a dataset — so that is the only place the record spells it.
    """
    jsons = [rel for rel, _ in layout_rules.iter_files(record, layout_rules.ANY_TEMPLATE) if rel.endswith(".json")]
    assert [Path(rel).name for rel in jsons] == ["dataset_description.json"] * len(jsons)


BIDS_PAGE = Path(__file__).resolve().parent.parent / "docs" / "Interoperability" / "BIDS" / "index.qmd"

GENERATED_BLOCKS = ("STUDY LAYOUT", "IGNORE FILES", "BIDS EXCEPTIONS", "SPEC SUFFIXES", "RESULT NAMES")


def test_the_bids_page_carries_every_generated_reference():
    """The one BIDS page renders each reference from its source rather than restating it.

    A block that stops being carried is how a page silently reverts to a hand-typed copy, so the set is pinned here as well as refreshed in CI.
    """
    page = BIDS_PAGE.read_text()
    for name in GENERATED_BLOCKS:
        begin, end = layout_rules.markers(name)
        assert begin in page, f"the BIDS page no longer renders {name}"
        assert end in page, f"the BIDS page's {name} region is unterminated"


def test_the_bids_page_is_in_sync_with_the_record(tmp_path):
    """Syncing the committed page changes nothing, so what a reader sees is what the record says."""
    copy = tmp_path / "index.qmd"
    shutil.copyfile(BIDS_PAGE, copy)
    assert layout_rules.sync_layout(copy) is False, "the BIDS page is stale; run `tvbo study layout --sync` on it"


def test_the_exceptions_table_is_exactly_what_bidsignore_exempts(record):
    """One list, two renderings: a documented exception and an exempted path cannot diverge."""
    table = layout_rules.exceptions_block(record, "Demo1999", REPLICATION)
    documented = {row.split("`")[1] for row in table.splitlines() if row.startswith("| `")}
    exempted = {
        line.lstrip("/").rstrip("/")
        for line in layout_rules.bidsignore_lines(record, REPLICATION, "Demo1999")
        if not line.startswith("#")
    }
    assert {p.rstrip("/") for p in documented} == exempted


def test_every_documented_exception_names_the_proposal_it_waits_on(record):
    """An entry outside the standard without a stated expiry is a permanent divergence by default."""
    for row in layout_rules.exceptions_block(record, "Demo1999", REPLICATION).splitlines():
        if row.startswith("| `"):
            assert "BEP" in row.rsplit("|", 2)[1], f"no proposal named for {row.split('`')[1]}"


def test_the_result_grammar_documents_every_entity_it_uses():
    """Each entity in the filename patterns is explained, and nothing is explained that is not used."""
    from tvbo.adapters.bids import RESULT_ENTITIES, RESULT_PATTERNS

    used = set()
    for pattern in RESULT_PATTERNS:
        used.update(re.findall(r"([a-z]+)-\{", pattern))
    assert used == set(RESULT_ENTITIES), f"documented {sorted(RESULT_ENTITIES)} but the patterns use {sorted(used)}"


def test_every_spec_suffix_names_a_class_the_datamodel_has():
    """A suffix vocabulary is only checkable while each suffix maps to a class that exists."""
    from tvbo.adapters.bids import SPEC_SUFFIXES
    from tvbo.datamodel import schema as datamodel

    for suffix, cls in SPEC_SUFFIXES.items():
        assert hasattr(datamodel, cls), f"suffix `_{suffix}` names {cls}, which the datamodel does not define"


@pytest.fixture
def scaffold(tmp_path):
    """A study scaffolded by the CLI, with the replication variant."""
    result = CliRunner().invoke(app, ["study", "init", "Demo1999", "--in", str(tmp_path), "-t", "replication"])
    assert result.exit_code == 0, result.output
    return tmp_path / "Demo1999"


def test_init_creates_every_directory_in_the_record(scaffold, record):
    for rel, _ in layout_rules.walk(record, REPLICATION):
        assert (scaffold / rel).is_dir(), f"{rel} was not created"


def test_init_keeps_only_tracked_empty_directories(scaffold, record):
    """A placeholder in an untracked directory would be committed into a tree git ignores."""
    for rel, _ in layout_rules.walk(record, REPLICATION):
        keep = scaffold / rel / ".gitkeep"
        if keep.exists():
            assert layout_rules.is_tracked(rel, record, REPLICATION), f"{rel} is untracked but got a .gitkeep"


def test_init_writes_both_dataset_descriptions(scaffold):
    study = json.loads((scaffold / "dataset_description.json").read_text())
    assert study["DatasetType"] == "study"
    assert study["Name"] == "Demo1999"

    derivative = json.loads((scaffold / "derivatives" / "tvbo" / "dataset_description.json").read_text())
    assert derivative["DatasetType"] == "derivative"
    assert derivative["SourceDatasets"] == [{"URL": "../.."}]
    assert derivative["GeneratedBy"][0]["Name"] == "tvbo"


def test_init_splices_the_layout_into_the_readme(scaffold):
    """The README carries the tree as a generated region, never as prose someone must update."""
    readme = (scaffold / "README.md").read_text()
    assert layout_rules.LAYOUT_BEGIN in readme
    assert "derivatives/" in readme.split(layout_rules.LAYOUT_BEGIN)[1]


def test_a_fresh_scaffold_validates(scaffold):
    """`tvbo study init` must produce a study `tvbo validate study` accepts.

    The entry recipe is `level: required`, so the scaffolder seeds a stub rather than leaving the author with a dataset that cannot pass its own validator.
    """
    result = CliRunner().invoke(app, ["validate", "study", str(scaffold), "-t", "replication"])
    assert result.exit_code == 0, result.output


def test_a_study_validates_without_being_told_its_variant(scaffold):
    """A replication ignores entries a plain study never had, so a check that assumed the plain layout reported a difference that was not a fault. The gate's header names the variant, and the checker reads it back."""
    assert layout_rules.templates_of((scaffold / ".gitignore").read_text()) == ("replication",)
    result = CliRunner().invoke(app, ["validate", "study", str(scaffold)])
    assert result.exit_code == 0, result.output


def test_the_seeded_recipe_declares_its_class(scaffold):
    recipe = (scaffold / "Demo1999.yaml").read_text()
    assert "tvbo_class: tvbo:SimulationStudy" in recipe


def test_layout_sync_is_idempotent(scaffold, record):
    readme = scaffold / "README.md"
    assert layout_rules.sync_layout(readme, record, "Demo1999", REPLICATION) is False


def _ignored(scaffold, rel: str) -> tuple[bool, str]:
    """Whether git ignores ``rel``, and the rule that decided it.

    ``git check-ignore`` exits 0 for any *matching* rule, a negation included, so the exit status alone reports ``!sourcedata/README.md`` as ignored. The rule it prints is what actually decides: a pattern starting with ``!`` re-includes the path.
    """
    done = subprocess.run(["git", "check-ignore", "-v", rel], cwd=scaffold, capture_output=True, text=True)
    if done.returncode != 0:
        return False, "no rule matched"
    rule = done.stdout.strip().split("\t")[0]
    return not rule.rpartition(":")[2].startswith("!"), rule


@pytest.mark.skipif(shutil.which("git") is None, reason="git is needed to check the ignore rules")
def test_the_gate_ignores_copyrighted_material(scaffold):
    """What must never be committed, checked against git rather than by reading the patterns.

    ``sourcedata/original_study/`` holds the reproduced work's own figures, and the A/B composites that embed them are staged inside it, so the single rule that keeps the deposit unpublished covers every composite made from it.
    """
    subprocess.run(["git", "init", "-q"], cwd=scaffold, check=True)
    must_be_ignored = [
        "sourcedata/original_study/paper.pdf",
        "sourcedata/connectomes/weights.h5",
        f"{layout_rules.relpath('figures_restricted')}/fig-3_ab.png",
        "docs/figures/fig-3.png",
        # The composites are staged untracked, and so must the report that renders them into a PDF.
        "docs/report_internal.pdf",
        "derivatives/tvbo/exp-01_result.h5",
        ".tvbo/kits/demo/shards/exp-01_split-0001_result.h5",
    ]
    for rel in must_be_ignored:
        ignored, rule = _ignored(scaffold, rel)
        assert ignored, f"{rel} is NOT ignored ({rule})"

    must_be_tracked = [
        "sourcedata/README.md",
        "derivatives/tvbo/dataset_description.json",
        "README.md",
        "docs/report.qmd",
        "spec/exp-01_experiment.yaml",
        # Provenance is the evidence, not an output: a record no one can read without re-running the study cannot be cited.
        "prov/prov-exp1_act.yaml",
    ]
    for rel in must_be_tracked:
        ignored, rule = _ignored(scaffold, rel)
        assert not ignored, f"{rel} must be tracked but is ignored by {rule}"
