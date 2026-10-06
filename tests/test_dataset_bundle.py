"""A workflow kit carries its data: freezing it copies each enumerated subject's empirical target (sidecar + payload) into the kit's shared ``data/<dataset_id>/`` tree, selecting the exact BIDS variant, and rewrites ``dataset.bids_root`` to a path relative to the frozen spec (like a network ``data_file``), so the kit ships the data its fan-out consumes and nothing else, with no separate upload or ``$TVBO_BIDS_ROOT``."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

from tvbo.classes.experiment import SimulationExperiment as _SE
from tvbo.cli import workflow as _wf


def _write_subject(root: Path, subject: str, atlas: str) -> Path:
    """Write one subject's FC sidecar + (dummy) HDF5 payload; return the sidecar path."""
    stem = f"sub-{subject}_tpl-MNI152NLin2009cAsym_cohort-HCPYA_atlas-{atlas}_desc-FCavg_relmat"
    subdir = root / f"sub-{subject}"
    subdir.mkdir(parents=True, exist_ok=True)
    (subdir / f"{stem}.h5").write_bytes(b"\x89HDF\r\n\x1a\n" + atlas.encode())  # payload marker
    sidecar = subdir / f"{stem}.yaml"
    sidecar.write_text(
        f"label: HCPYA FC | sub-{subject} | atlas {atlas}\nnumber_of_nodes: 3\ndata_file: {stem}.h5\n",
        encoding="utf-8",
    )
    return sidecar


def _stub(dataset, observations, source_file=None):
    """A duck-typed experiment binding the REAL dataset methods, so the test exercises the shipping code paths without constructing a full SimulationExperiment."""
    stub = SimpleNamespace(dataset=dataset, observations=observations, _source_file=source_file)
    stub.dataset_observation_targets = _SE.dataset_observation_targets.fget(stub)
    for name in (
        "_dataset_bids_root",
        "dataset_subject_ids",
        "subject_selection",
        "_dataset_subjects",
        "_reference_datasets",
        "_reference_context",
        "declared_references",
        "_referenced",
        "referenced_cohorts",
        "reference_cohort_files",
        "dataset_bundle_files",
        "_find_subject_file",
        "_find_subject_file_by_entities",
        "_bids_query_dict",
        "_match_subject_files",
        "_sidecar_companions",
    ):
        setattr(stub, name, _SE.__dict__[name].__get__(stub))
    return stub


def _fc_experiment(bids_root, *, subjects=None, source_file=None):
    query = SimpleNamespace(atlas="HCPMMP1", desc="FCavg", suffix="relmat")
    obs = SimpleNamespace(source=["dataset.subject.fc"], query=query)
    dataset = SimpleNamespace(bids_root=str(bids_root), subjects=subjects)
    return _stub(dataset, {"empirical_fc": obs}, source_file=source_file)


@pytest.fixture
def cohort(tmp_path: Path) -> Path:
    """A 2-subject BIDS FC tree, each subject carrying an HCPMMP1 target + a Schaefer decoy."""
    root = tmp_path / "functional_connectomes"
    for subj in ("100206", "100307"):
        _write_subject(root, subj, "HCPMMP1")
        _write_subject(root, subj, "Schaefer400")  # the "much more" that must NOT be bundled
    return root


def test_bundle_selects_variant_and_payload(cohort: Path):
    exp = _fc_experiment(cohort)
    manifest = exp.dataset_bundle_files()

    assert set(manifest) == {"100206", "100307"}
    for files in manifest.values():
        names = sorted(f.name for f in files)
        # exactly the HCPMMP1 sidecar + its payload — no Schaefer decoy, no work/ noise
        assert any(n.endswith(".yaml") and "atlas-HCPMMP1" in n for n in names)
        assert any(n.endswith(".h5") and "atlas-HCPMMP1" in n for n in names)
        assert not any("Schaefer" in n for n in names)
        assert len(files) == 2


def test_entity_override_pins_a_different_variant(cohort: Path):
    exp = _fc_experiment(cohort)
    manifest = exp.dataset_bundle_files({"atlas": "Schaefer400"})
    for files in manifest.values():
        assert all("Schaefer400" in f.name for f in files)
        assert not any("HCPMMP1" in f.name for f in files)


def test_subject_subset_scopes_the_bundle(cohort: Path):
    """An explicit dataset.subjects list curates which subjects the bundle carries."""
    exp = _fc_experiment(cohort, subjects=["100206"])
    assert set(exp.dataset_bundle_files()) == {"100206"}


def test_the_enumeration_follows_the_query_and_the_subject_list_it_is_asked_with(cohort: Path, monkeypatch):
    """Subjects found under one query, or listed at one time, do not answer for another, and an unchanged question lists the tree once."""
    from tvbo.data import cohort as cohort_io

    listings = []
    list_tree = cohort_io.subject_sidecars
    monkeypatch.setattr(cohort_io, "subject_sidecars", lambda *args: listings.append(args) or list_tree(*args))
    _write_subject(cohort, "100408", "Schaefer400")
    exp = _fc_experiment(cohort)
    assert exp.dataset_subject_ids() == ["100206", "100307"]
    assert exp.subject_selection() is exp.subject_selection() and len(listings) == 1

    exp.observations["empirical_fc"].query.atlas = "Schaefer400"
    assert exp.dataset_subject_ids() == ["100206", "100307", "100408"]

    exp.dataset.subjects = ["100408", "100206"]
    assert exp.dataset_subject_ids() == ["100408", "100206"]


def test_bundle_dataset_copies_and_returns_relative_root(cohort: Path, tmp_path: Path):
    exp = _fc_experiment(cohort)
    kit = tmp_path / "kit"
    dest = kit / "data" / "dataset"
    root = _wf._bundle_dataset(exp, kit, kit / "spec" / "exp")

    assert root == "../../data/dataset"  # relative to the frozen spec's directory
    copied = sorted(p.name for p in dest.rglob("*") if p.is_file())
    assert (dest / "sub-100206").is_dir() and (dest / "sub-100307").is_dir()
    assert all("Schaefer" not in n for n in copied)
    assert sum("atlas-HCPMMP1" in n and n.endswith(".h5") for n in copied) == 2


def test_relative_bids_root_rebases_to_spec_dir(tmp_path: Path):
    """A kit records ``bids_root: ../data/<dataset_id>`` and resolves it against the spec file, so `tvbo run spec/exp.yaml` finds the kit's data regardless of the working dir."""
    spec_dir = tmp_path / "kit" / "spec"
    spec_dir.mkdir(parents=True)
    bundled = tmp_path / "kit" / "data" / "dataset"
    _write_subject(bundled, "100206", "HCPMMP1")

    exp = _fc_experiment("../data/dataset", source_file=str(spec_dir / "exp.yaml"))
    assert exp._dataset_bids_root().resolve() == bundled.resolve()
    assert exp.dataset_subject_ids() == ["100206"]


def test_no_dataset_target_bundles_nothing(tmp_path: Path):
    exp = _stub(SimpleNamespace(bids_root=None, subjects=None), {})  # no dataset-sourced obs
    assert _wf._bundle_dataset(exp, tmp_path / "kit", tmp_path / "kit" / "spec") is None
    assert not (tmp_path / "kit" / "data").exists()


def test_missing_payload_raises_not_silently_dropped(tmp_path: Path):
    """A sidecar whose data_file is absent must fail loudly — bundling a sidecar without its matrix would only break at load time on a compute node."""
    root = tmp_path / "fc"
    sidecar = _write_subject(root, "100206", "HCPMMP1")
    sidecar.with_suffix(".h5").unlink()  # payload gone; sidecar still references it
    exp = _fc_experiment(root, subjects=["100206"])
    with pytest.raises(FileNotFoundError):
        exp.dataset_bundle_files()


def test_directory_payload_is_copied_as_tree(tmp_path: Path):
    """A .zarr payload is a directory: the bundle must copytree it, not choke on copy2."""
    root = tmp_path / "fc"
    subdir = root / "sub-100206"
    stem = "sub-100206_atlas-HCPMMP1_desc-FCavg_relmat"
    zarr = subdir / f"{stem}.zarr"
    (zarr / "0").mkdir(parents=True)
    (zarr / "0" / ".zarray").write_text("{}", encoding="utf-8")
    (subdir / f"{stem}.yaml").write_text(f"label: z\nnumber_of_nodes: 3\ndata_file: {stem}.zarr\n", encoding="utf-8")

    exp = _fc_experiment(root, subjects=["100206"])
    kit = tmp_path / "kit"
    assert _wf._bundle_dataset(exp, kit, kit / "spec") == "../data/dataset"
    assert (kit / "data" / "dataset" / "sub-100206" / f"{stem}.zarr" / "0" / ".zarray").is_file()


def test_bundle_dies_on_unresolved_selection(cohort: Path):
    """An over-tight --bundle-select (no subject has the variant) is a hard error, not a silent fallback to the machine-specific bids_root."""
    exp = _fc_experiment(cohort)
    kit = cohort.parent / "kit"
    with pytest.raises(typer.Exit):
        _wf._bundle_dataset(exp, kit, kit / "spec", {"atlas": "DoesNotExist"})


def test_experiments_reading_one_dataset_share_its_tree_and_a_conflicting_file_is_refused(cohort: Path, tmp_path: Path):
    """Two experiments of one emission that read the same dataset land in one ``data/<dataset_id>/`` tree, the second copying only what the first did not; a file the emission wrote, found with other content, stops the freeze."""
    kit, written = tmp_path / "kit", set()
    first = _fc_experiment(cohort, subjects=["100206"])
    assert _wf._bundle_dataset(first, kit, kit / "spec" / "a", written=written) == "../../data/dataset"
    assert _wf._bundle_dataset(_fc_experiment(cohort), kit, kit / "spec" / "b", written=written) == "../../data/dataset"
    assert sorted(p.name for p in (kit / "data" / "dataset").iterdir()) == ["sub-100206", "sub-100307"]
    payload = next((kit / "data" / "dataset" / "sub-100307").glob("*.h5"))
    payload.write_bytes(b"other content")
    with pytest.raises(typer.Exit):
        _wf._bundle_dataset(_fc_experiment(cohort), kit, kit / "spec" / "c", written=written)


def test_a_file_an_earlier_emission_left_is_refreshed(cohort: Path, tmp_path: Path, monkeypatch):
    """A kit emitted again after its data changed carries the new data: what the earlier emission left is stale, not a clash, and the emission says how much it replaced."""
    kit = tmp_path / "kit"
    assert _wf._bundle_dataset(_fc_experiment(cohort), kit, kit / "spec") == "../data/dataset"
    source = next((cohort / "sub-100307").glob("*atlas-HCPMMP1*.h5"))
    source.write_bytes(b"refitted")
    said: list[str] = []
    monkeypatch.setattr(_wf._common, "info", said.append)
    assert _wf._bundle_dataset(_fc_experiment(cohort), kit, kit / "spec") == "../data/dataset"
    assert (kit / "data" / "dataset" / "sub-100307" / source.name).read_bytes() == b"refitted"
    assert "1 file(s) copied, 1 of them refreshing a stale file from an earlier emission" in said[-1]


def test_a_freeze_that_fails_leaves_the_experiments_dataset_as_it_was(cohort: Path, tmp_path: Path, monkeypatch):
    """The freeze points the dataset at the kit's tree and narrows its subjects while it renders; a failure on the way must not leave that in the experiment the caller still holds."""
    experiment = _fc_experiment(cohort, subjects=["100206", "100307"])
    listed = experiment.dataset.subjects

    def refuse(_spec):
        raise RuntimeError("no such workflow slot")

    monkeypatch.setattr(_wf._wf, "workflow_config_from_spec", refuse)
    with pytest.raises(RuntimeError, match="no such workflow slot"):
        _wf._freeze_spec_yaml(experiment, tmp_path / "spec", workflow_spec={"chunk": 2}, dataset_bids_root="../data/dataset")
    assert experiment.dataset.bids_root == str(cohort)
    assert experiment.dataset.subjects is listed


def test_the_frozen_dataset_lists_exactly_the_fan_out():
    """A shared tree may hold other experiments' subjects, so the frozen dataset names its own: the dataset's entries for them, or bare ones for subjects found by discovery."""
    from tvbo.datamodel.schema import Subject

    listed = {"01": Subject(subject_id="01", group="patient"), "02": Subject(subject_id="02"), "03": Subject(subject_id="03")}
    pinned = _wf._fan_out_subjects(listed, ["03", "01"])
    assert list(pinned) == ["03", "01"] and pinned["01"].group == "patient"
    assert [s.subject_id for s in _wf._fan_out_subjects(None, ["07"]).values()] == ["07"]
