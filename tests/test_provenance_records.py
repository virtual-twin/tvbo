"""A run records what it did, not a second copy of what it was asked to do.

The frozen spec beside a container is the recipe; these records are the run — the command, the clock, the machine, the package versions and the checksum of what came out. None of that is derivable from a recipe, which is the whole reason `prov/` is not redundant with the study YAML. These pin that the records are written, that every field in them is a pointer or read off the artifact, and that a study can switch them off.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tvbo.data import provenance
from tvbo.utils.study_layout import is_tracked, load_layout, relpath


@pytest.fixture
def study(tmp_path):
    """A study root holding one written container, as a run would leave it."""
    from tvbo.utils.study_layout import study_path

    results = study_path("results", root=tmp_path)
    results.mkdir(parents=True)
    container = results / "exp-3_model-Kuramoto_result.h5"
    xr.Dataset(
        {
            "theta": xr.DataArray(np.zeros(4), dims=["time"]),
            "psd": xr.DataArray(np.ones(2), dims=["freq"]),
        }
    ).to_netcdf(container, engine="h5netcdf")
    return tmp_path, container


def _records(study, fmt="yaml", **kw):
    root, container = study
    paths = provenance.emit(
        container=container, study_root=root, produced_by="tvbo:exp/Demo/exp-3", outputs=["theta", "psd"], fmt=fmt, **kw
    )
    return root, {p.name: p for p in paths}


def test_a_run_writes_all_four_bep028_record_kinds(study):
    """An entity with no activity, environment or software beside it records nothing checkable."""
    _root, files = _records(study)
    assert sorted(files) == [f"prov-exp3_{kind}.yaml" for kind in ("act", "ent", "env", "soft")]


def test_the_records_land_in_the_layouts_provenance_directory(study):
    root, files = _records(study)
    for path in files.values():
        assert path.parent == root / relpath("provenance", load_layout())


def test_provenance_is_tracked_because_it_is_the_evidence():
    """Every other product of a run is regenerated; a record no one can read cannot be cited."""
    record = load_layout()
    assert is_tracked(relpath("provenance", record), record, ())


def test_the_entity_names_the_container_relative_to_the_study(study):
    """An absolute path would be true only on the machine that ran it."""
    import yaml

    _root, files = _records(study)
    entity = yaml.safe_load(files["prov-exp3_ent.yaml"].read_text())
    assert entity["container"] == "derivatives/tvbo/exp-3_model-Kuramoto_result.h5"
    assert not Path(entity["container"]).is_absolute()


def test_the_entity_carries_the_containers_own_checksum(study):
    """Read off the artifact, so a record cannot describe a file that has since changed."""
    import hashlib

    import yaml

    root, files = _records(study)
    entity = yaml.safe_load(files["prov-exp3_ent.yaml"].read_text())
    expected = hashlib.sha256((root / entity["container"]).read_bytes()).hexdigest()
    assert entity["provenance"]["digest"]["sha256"]["value"] == expected


def test_the_entity_lists_what_the_container_holds(study):
    import yaml

    _root, files = _records(study)
    assert yaml.safe_load(files["prov-exp3_ent.yaml"].read_text())["outputs"] == ["theta", "psd"]


def test_the_recorded_command_carries_no_machine_path(study, monkeypatch):
    """``prov/`` is tracked, so the interpreter's absolute path would make every re-run a diff."""
    import yaml

    monkeypatch.setattr(sys, "argv", ["/opt/some/venv/bin/tvbo", "run", "Demo.yaml", "--experiment", "1"])
    _root, files = _records(study)
    act = yaml.safe_load(files["prov-exp3_act.yaml"].read_text())
    assert act["command"] == "tvbo run Demo.yaml --experiment 1"


def test_the_activity_records_the_clock_and_the_command(study):
    """What a recipe cannot say: when it ran and how it was invoked."""
    import yaml

    _root, files = _records(study, started_at="2026-08-18T10:00:00+00:00", command="tvbo run Demo.yaml")
    act = yaml.safe_load(files["prov-exp3_act.yaml"].read_text())
    assert act["command"] == "tvbo run Demo.yaml"
    assert act["started_at"] == "2026-08-18T10:00:00+00:00"
    assert act["ended_at"] > act["started_at"]


def test_the_software_record_reports_versions_that_actually_ran(study):
    """A pinned requirement is what was asked for; the record has to claim what was imported."""
    from importlib.metadata import version

    import yaml

    _root, files = _records(study)
    packages = {p["name"]: p["version"] for p in yaml.safe_load(files["prov-exp3_soft.yaml"].read_text())["packages"]}
    assert packages["tvbo"] == version("tvbo")


def test_a_package_the_study_declared_is_recorded_with_what_ran(study):
    """`requires:` states what the study needs; the record has to say which version answered it, or the two cannot be compared."""
    from importlib.metadata import version

    import yaml

    _root, files = _records(study, requires=("pytest",))
    packages = {p["name"]: p["version"] for p in yaml.safe_load(files["prov-exp3_soft.yaml"].read_text())["packages"]}
    assert packages["pytest"] == version("pytest")


def test_the_environment_record_names_the_machine(study):
    import platform

    import yaml

    _root, files = _records(study)
    env = yaml.safe_load(files["prov-exp3_env.yaml"].read_text())
    assert env["platform"] == platform.platform()
    assert env["version"] == platform.python_version()


def test_json_is_written_when_the_study_asks_for_it(study):
    """BEP028 names JSON; YAML is the default this study proposes in its place."""
    import json

    _root, files = _records(study, fmt="json")
    assert sorted(files) == [f"prov-exp3_{kind}.json" for kind in ("act", "ent", "env", "soft")]
    assert json.loads(files["prov-exp3_ent.json"].read_text())["container"].endswith("_result.h5")


def test_one_format_at_a_time(study):
    """Two serializations of one record are two things that can disagree."""
    root, _files = _records(study, fmt="json")
    assert not list((root / relpath("provenance", load_layout())).glob("*.yaml"))


def test_an_unknown_format_is_refused(study):
    root, container = study
    with pytest.raises(ValueError, match="yaml.*json"):
        provenance.emit(container=container, study_root=root, produced_by="tvbo:exp/Demo/exp-3", fmt="toml")


def test_the_label_comes_from_the_producers_own_scope():
    """A record set and the DataRef reaching the same container must agree on what produced it."""
    assert provenance.prov_label("tvbo:exp/Demo/exp-3") == "exp3"
    assert provenance.prov_label("tvbo:ana/Demo/fcGradient") == "anafcGradient"


def test_a_missing_container_is_recorded_without_a_checksum(tmp_path):
    """A digest is a claim about bytes; absent bytes must produce no claim rather than a false one."""
    assert provenance.digest_of(tmp_path / "nothing.h5") is None


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
      name: KuramotoCoupling
      label: KuramotoCoupling
      parameters:
        a: {name: a, value: 0.01}
        N: {name: N, value: 1.0}
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


def test_a_run_leaves_its_provenance_beside_the_study(ran):
    """The records describe a run, so a run has to be what writes them."""
    root = ran()
    prov = root / relpath("provenance", load_layout())
    assert sorted(p.name for p in prov.glob("*")) == [f"prov-exp3_{k}.yaml" for k in ("act", "ent", "env", "soft")]


def test_the_recorded_container_is_the_one_the_run_wrote(ran):
    """A record naming a file the run did not produce is worse than no record."""
    import yaml

    root = ran()
    entity = yaml.safe_load((root / relpath("provenance", load_layout()) / "prov-exp3_ent.yaml").read_text())
    assert (root / entity["container"]).is_file()


def test_the_outputs_are_read_from_the_container_not_the_spec(ran):
    """A run that recorded less than it declared must be visible in the record."""
    import yaml

    root = ran()
    entity = yaml.safe_load((root / relpath("provenance", load_layout()) / "prov-exp3_ent.yaml").read_text())
    with xr.open_dataset(root / entity["container"], engine="h5netcdf") as ds:
        assert entity["outputs"] == sorted(str(v) for v in ds.data_vars)


def test_a_study_can_switch_provenance_off(ran):
    """Declared in the spec, so the choice travels with the recipe rather than with the machine."""
    root = ran("workflow:\n  emit_provenance: false\n")
    assert not (root / relpath("provenance", load_layout())).exists()
