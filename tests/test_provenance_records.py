"""A run records what it did, beside the container it produced.

The frozen spec beside a container is the recipe; this record is the run — the command, the clock, the machine, the package versions and the checksum of what came out. None of that is derivable from a recipe, which is why it is kept. It lives under ``provenance:`` in the container's own sidecar rather than in a parallel ``prov/`` tree, so the two halves of the story sit in one file and cannot fall out of step. These pin that the record is written without being asked for, that every field in it is a pointer or read off the artifact, and that it never overwrites what the sidecar already said.
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
import yaml

from tvbo.data import provenance
from tvbo.utils.study_layout import is_tracked, load_layout, relpath, study_path


@pytest.fixture
def study(tmp_path):
    """A study root holding one written container and the frozen spec beside it, as a run would leave it."""
    results = study_path("results", root=tmp_path)
    results.mkdir(parents=True)
    container = results / "exp-3_model-Kuramoto_result.h5"
    xr.Dataset(
        {
            "theta": xr.DataArray(np.zeros(4), dims=["time"]),
            "psd": xr.DataArray(np.ones(2), dims=["freq"]),
        }
    ).to_netcdf(container, engine="h5netcdf")
    provenance_sidecar = container.with_suffix(".yaml")
    provenance_sidecar.write_text(yaml.safe_dump({"id": 3, "label": "the frozen spec"}, sort_keys=False))
    return tmp_path, container


def _record(study, **kw):
    """Emit for the fixture's container and hand back the study root and the provenance block written."""
    root, container = study
    path = provenance.emit(container=container, produced_by="tvbo:exp/Demo/exp-3", outputs=["theta", "psd"], **kw)
    return root, yaml.safe_load(path.read_text())


def test_a_run_records_itself_in_the_containers_own_sidecar(study):
    """One file carries the recipe and the run; a reader who opened one has the other."""
    _root, container = study
    assert provenance.emit(container=container, produced_by="tvbo:exp/Demo/exp-3") == container.with_suffix(".yaml")


def test_no_parallel_provenance_tree_is_written(study):
    """The record travels with its container, so there is no second directory to keep in step."""
    root, _ = _record(study)
    assert not list(root.rglob("prov-*"))
    with pytest.raises(KeyError):
        relpath("provenance", load_layout())


def test_the_frozen_spec_beside_the_container_survives_the_record(study):
    """The sidecar is the recipe first; recording a run must add to that document, never replace it."""
    _root, document = _record(study)
    assert document["id"] == 3
    assert document["label"] == "the frozen spec"
    assert "provenance" in document


def test_an_assertion_this_run_does_not_make_is_left_alone(study):
    """``experiment_yaml_hash`` and the input fingerprints are read by the cross-experiment cache; a re-run must not drop them."""
    _root, container = study
    sidecar = container.with_suffix(".yaml")
    sidecar.write_text(yaml.safe_dump({"id": 3, "provenance": {"experiment_yaml_hash": "abc123", "inputs": [{"iri": "x"}]}}))
    provenance.emit(container=container, produced_by="tvbo:exp/Demo/exp-3")
    record = yaml.safe_load(sidecar.read_text())["provenance"]
    assert record["experiment_yaml_hash"] == "abc123"
    assert record["inputs"] == [{"iri": "x"}]
    assert record["activities"], "and the run's own account is there too"


def test_the_record_carries_the_containers_own_checksum(study):
    """Read off the artifact, so a record cannot claim a digest the file does not have."""
    import hashlib

    root, document = _record(study)
    _root, container = study
    expected = hashlib.sha256(container.read_bytes()).hexdigest()
    assert document["provenance"]["digest"]["sha256"]["value"] == expected


def test_a_claim_this_run_cannot_make_is_dropped_rather_than_inherited(study):
    """A digest is a claim about bytes: a re-run that wrote no container must leave none, not the one that last matched."""
    _root, container = study
    first = _record(study)[1]["provenance"]
    assert first["digest"] and first["outputs"]
    container.unlink()
    provenance.emit(container=container, produced_by="tvbo:exp/Demo/exp-3")
    record = yaml.safe_load(container.with_suffix(".yaml").read_text())["provenance"]
    assert "digest" not in record and "outputs" not in record


def test_the_record_lists_what_the_container_holds(study):
    """Read from the container rather than predicted from the spec: a run that recorded less than it declared is visible."""
    _root, document = _record(study)
    assert sorted(document["provenance"]["outputs"]) == ["psd", "theta"]


def test_the_recorded_command_carries_no_machine_path(study, monkeypatch):
    """Two runs are compared through these records; an interpreter path would differ while saying nothing about what ran."""
    monkeypatch.setattr(sys, "argv", ["/opt/homebrew/Caskroom/miniforge/base/bin/tvbo", "run", "Demo.yaml"])
    _root, document = _record(study)
    assert document["provenance"]["activities"][0]["command"] == "tvbo run Demo.yaml"


def test_the_activity_records_the_clock_and_what_produced_the_container(study):
    _root, document = _record(study, started_at="2026-01-01T00:00:00+00:00")
    activity = document["provenance"]["activities"][0]
    assert activity["started_at"] == "2026-01-01T00:00:00+00:00"
    assert activity["ended_at"] and activity["ended_at"] >= activity["started_at"]
    assert activity["iri"] == "tvbo:exp/Demo/exp-3"


def test_the_environment_reports_versions_that_actually_ran(study):
    """A pinned requirement is what was asked for; what ran is what the interpreter reports."""
    from importlib.metadata import version

    _root, document = _record(study)
    found = {r["name"]: r["version"] for r in document["provenance"]["environment"]["requirements"]}
    assert found["tvbo"] == version("tvbo")
    assert "numpy" in found


def test_a_package_the_study_declared_is_recorded_with_what_ran(study):
    """So a study's own ``requires:`` can be compared against the version that was installed."""
    _root, document = _record(study, requires=("pytest",))
    found = {r["name"] for r in document["provenance"]["environment"]["requirements"]}
    assert "pytest" in found


def test_the_environment_names_the_machine(study):
    _root, document = _record(study)
    environment = document["provenance"]["environment"]
    assert environment["platform"] and environment["version"]


def test_the_environment_travels_in_full_rather_than_by_name(study):
    """This record is one container's sidecar, so a reference to an environment defined elsewhere would point outside the only file a reader has."""
    _root, document = _record(study)
    assert isinstance(document["provenance"]["environment"], dict)


def test_the_label_comes_from_the_producers_own_scope():
    assert provenance.prov_label("tvbo:exp/Demo/exp-3") == "exp3"
    assert provenance.prov_label("tvbo:ana/Demo/fcGradient") == "anafcGradient"


def test_a_missing_container_is_recorded_without_a_checksum(tmp_path):
    """A run that failed to write its container still says it ran; refusing to record that loses the only evidence."""
    results = study_path("results", root=tmp_path)
    results.mkdir(parents=True)
    path = provenance.emit(container=results / "exp-9_model-X_result.h5", produced_by="tvbo:exp/Demo/exp-9")
    record = yaml.safe_load(path.read_text())["provenance"]
    assert "digest" not in record
    assert record["activities"]


# ------------------------------------------------------- driven through `tvbo run`

STUDY = """tvbo_class: tvbo:SimulationStudy
citekey: Demo
title: "Provenance smoke study"
# WORKFLOW
experiments:
  - id: 3
    dynamics:
      name: Kuramoto
      label: Kuramoto
      parameters:
        omega: {name: omega, value: 0.0628, unit: rad_per_ms}
      coupling_inputs:
        c: {name: c, description: "coupling"}
      state_variables:
        theta:
          name: theta
          unit: rad
          equation: {lhs: "Derivative(theta, t)", rhs: "omega + c"}
          variable_of_interest: true
          coupling_variable: true
      output: [theta]
      number_of_modes: 1
    network:
      number_of_nodes: 2
      nodes:
        - {id: 0, label: r0}
        - {id: 1, label: r1}
      edges:
        - {source: 0, target: 1, weight: 0.5}
        - {source: 1, target: 0, weight: 0.5}
      coupling:
        KuramotoCoupling:
          label: KuramotoCoupling
          parameters:
            a: {value: 0.01}
            N: {value: 1.0}
          pre_expression: {rhs: "sin(theta_j - theta_i)"}
          post_expression: {rhs: "a * gx / N"}
          incoming_states: [theta]
          local_states: [theta]
    integration:
      method: RungeKutta4thOrder
      duration: 20.0
      step_size: 1.0
      transient_time: 0.0
"""


@pytest.fixture
def ran(tmp_path):
    """A study actually run by the CLI, which is the only thing that proves the wiring."""
    pytest.importorskip("tvboptim")
    from typer.testing import CliRunner

    from tvbo.cli import app

    def _run(extra=""):
        root = tmp_path / f"study{len(list(tmp_path.iterdir()))}"
        root.mkdir()
        (root / "Demo.yaml").write_text(STUDY.replace("# WORKFLOW\n", extra), encoding="utf-8")
        result = CliRunner().invoke(app, ["run", str(root / "Demo.yaml")])
        assert result.exit_code == 0, result.output
        return root

    return _run


def test_a_run_records_itself_without_being_asked(ran):
    """The sidecar is a product of the run and never tracked, so there is nothing for a study to switch off."""
    root = ran()
    sidecars = list(study_path("results", root=root).glob("*_result.yaml"))
    assert sidecars, "the run wrote no sidecar at all"
    assert any(yaml.safe_load(p.read_text()).get("provenance", {}).get("activities") for p in sidecars)


def test_the_recorded_container_is_the_one_the_run_wrote(ran):
    root = ran()
    sets = provenance.read_records(root)
    assert sets, "the run recorded nothing"
    container = next(iter(sets.values()))["ent"]["container"]
    assert (root / container).exists()


def test_the_outputs_are_read_from_the_container_not_the_spec(ran):
    """A run that recorded less than it declared must be visible in the record."""
    root = ran()
    entity = next(iter(provenance.read_records(root).values()))["ent"]
    with xr.open_dataset(root / entity["container"], engine="h5netcdf") as ds:
        assert entity["outputs"] == sorted(str(v) for v in ds.data_vars)


def test_provenance_is_a_product_of_the_run_and_never_tracked():
    """It records one machine at one moment, so committing it would assert a fact the next person's run contradicts."""
    record = load_layout()
    assert not is_tracked(relpath("results", record), record, ())


# ------------------------------------------------------------------ reading back


def test_a_record_is_read_back_under_its_own_label(study):
    """The label is what joins a run to its container, so reading keys on it and not on the filename."""
    root, _ = _record(study)
    sets = provenance.read_records(root)
    assert sorted(sets) == ["exp3"]
    assert sorted(sets["exp3"]) == ["act", "ent", "env", "soft"]


def test_a_sidecar_that_records_no_run_is_skipped_not_faulted(study):
    """It describes a container written before the run recorded itself; faulting would lose every record that is there."""
    root, container = study
    other = container.parent / "exp-4_model-Kuramoto_result.yaml"
    other.write_text(yaml.safe_dump({"id": 4}))
    _record(study)
    assert sorted(provenance.read_records(root)) == ["exp3"]


def test_two_containers_from_one_producer_are_both_read(study):
    """A cohort experiment fans into one container per subject; keying them all on the shared label would report the last one read as the whole run."""
    root, container = study
    other = container.parent / "exp-3_model-Kuramoto_sub-02_result.h5"
    other.write_bytes(container.read_bytes())
    _record(study)
    provenance.emit(container=other, produced_by="tvbo:exp/Demo/exp-3")
    sets = provenance.read_records(root)
    assert len(sets) == 2
    assert {r["ent"]["container"].rsplit("/", 1)[1] for r in sets.values()} == {container.name, other.name}


def test_an_absent_results_directory_reads_as_no_records(tmp_path):
    assert provenance.read_records(tmp_path / "nowhere") == {}


# ------------------------------------------------------------------------ graph


def test_every_node_carries_its_prov_o_type(study):
    """A consumer draws or queries by what a node IS, never by where a filename put it."""
    root, _ = _record(study)
    graph = provenance.provenance_graph(root)
    assert {n["type"] for n in graph["nodes"]} <= {provenance.PROV_ACTIVITY, provenance.PROV_ENTITY, provenance.PROV_AGENT}
    assert any(n["type"] == provenance.PROV_ACTIVITY for n in graph["nodes"])


def test_the_entity_was_generated_by_the_activity_of_its_own_label(study):
    root, _ = _record(study)
    graph = provenance.provenance_graph(root)
    generated = [e for e in graph["edges"] if e["relation"] == "prov:wasGeneratedBy"]
    assert len(generated) == 1
    assert generated[0]["target"] == "activity:exp3"


def test_the_activity_is_attributed_to_the_packages_that_carried_it_out(study):
    root, _ = _record(study)
    graph = provenance.provenance_graph(root)
    agents = {n["label"]: n.get("version") for n in graph["nodes"] if n["type"] == provenance.PROV_AGENT}
    assert "tvbo" in agents and agents["tvbo"]


def test_a_used_reference_becomes_a_derivation_edge(study):
    """What makes a set of runs one derivation rather than a pile of independent ones."""
    root, _ = _record(study, used=["derivatives/tvbo/exp-1_model-X_result.h5"])
    graph = provenance.provenance_graph(root)
    used = [e for e in graph["edges"] if e["relation"] == "prov:used"]
    assert [e["target"] for e in used] == ["entity:derivatives/tvbo/exp-1_model-X_result.h5"]


def test_both_ends_of_a_used_edge_spell_the_container_the_same_way(study):
    """The entity a run produced and the reference a later run makes to it have to be one node, not two."""
    root, container = study
    produced = provenance.input_containers(
        [SimpleNamespace(experiment="3", analysis=None, iri=None)],
        results_root=container.parent,
        study_root=root,
    )
    _record(study)
    entity = provenance.read_records(root)["exp3"]["ent"]["container"]
    assert produced == [entity]


def test_an_input_this_study_never_produced_is_still_a_node(study):
    """A reference to something this study did not compute is exactly what an external input looks like."""
    root, _ = _record(study, used=["sourcedata/empirical_fc.h5"])
    graph = provenance.provenance_graph(root)
    labels = {n["label"] for n in graph["nodes"] if n["type"] == provenance.PROV_ENTITY}
    assert "sourcedata/empirical_fc.h5" in labels


def test_an_unresolvable_used_binding_is_a_missing_edge_not_a_failed_run(tmp_path):
    """Provenance describes a run that already succeeded; a binding it never had to read is not a reason to fail it."""
    refs = [SimpleNamespace(experiment="404", analysis=None, iri=None)]
    assert provenance.input_containers(refs, results_root=tmp_path, study_root=tmp_path) == []
