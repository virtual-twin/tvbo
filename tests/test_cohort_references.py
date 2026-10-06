"""A cohort is a named subset of a dataset's subjects, and a reference to it reads each member's own file.

Three things are held here. A cohort read returns every member on a ``subject`` axis, aligned by node label, and inside a per-subject run only that run's own member. A group-level quantity over a cohort is an ordinary equation analysis that uses it, whose container is pinned to the cohort's members. And a network can declare where a layer and a transform argument come from, so a normalisation that depends on the whole cohort is stated in the recipe and applied on read to raw per-subject connectomes.

The worked case is a cohort-relative connectome normalisation: keep the pairs connected in at least a given share of members, scale by the largest square-rooted cohort-mean weight among them, and zero the rest. Every expected value is computed directly in numpy from the same raw matrices.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import xarray as xr
import yaml

pytest.importorskip("jax")

from tvbo import Network  # noqa: E402
from tvbo.data import analysis_io, cohort, dataref  # noqa: E402
from tvbo.data.matrix_io import read_edge  # noqa: E402
from tvbo.data.study_manifest import _analysis_fingerprint  # noqa: E402
from tvbo.datamodel.schema import Analysis, DataRef, Dataset  # noqa: E402

LABELS = ["A", "B", "C", "D"]
QUERY = {"atlas": "Toy", "desc": "SC", "suffix": "relmat"}
MEMBERS = ["01", "02", "03", "04"]
SHUFFLED = "03"
OUTSIDER = "05"


def _raw(subject: str):
    """Symmetric raw weights and lengths for one subject, with a pair that only some subjects connect."""
    rng = np.random.default_rng(int(subject))
    w = np.triu(rng.uniform(1.0, 400.0, (4, 4)), 1).astype(np.float32)
    if subject in ("02", "04"):
        w[0, 3] = 0.0
    if subject == "02":
        w[1, 2] = 0.0
    length = np.triu(rng.uniform(10.0, 150.0, (4, 4)), 1).astype(np.float32)
    return w + w.T, length + length.T


def _write_subject(root, subject: str, order=None, atlas: str = "Toy"):
    """Save one subject's connectome under a BIDS datatype folder, optionally with its nodes stored in another order."""
    w, length = _raw(subject)
    order = list(range(4)) if order is None else order
    net = Network.from_matrix(
        weights=w[np.ix_(order, order)], lengths=length[np.ix_(order, order)], labels=[LABELS[i] for i in order]
    )
    folder = root / f"sub-{subject}" / "dwi"
    folder.mkdir(parents=True, exist_ok=True)
    net.save(folder / f"sub-{subject}_atlas-{atlas}_desc-SC_relmat.yaml")


@pytest.fixture
def root(tmp_path):
    """Five subjects' raw connectomes; one is stored with its nodes permuted and each carries a decoy of another atlas."""
    data = tmp_path / "structural_connectomes"
    for subject in [*MEMBERS, OUTSIDER]:
        _write_subject(data, subject, order=[2, 0, 3, 1] if subject == SHUFFLED else None)
        _write_subject(data, subject, atlas="Decoy")
    return data


def _dataset(root, members=MEMBERS, **extra):
    return Dataset(
        dataset_id="toy_sc", bids_root=str(root), cohorts=[{"cohort_id": "four", "members": list(members)}], **extra
    )


def _member(output: str, **extra):
    return {"cohort": "four", "query": QUERY, "output": output, **extra}


def _stack(layer: int):
    return np.stack([_raw(s)[layer] for s in MEMBERS]).astype(float)


def test_a_cohort_read_stacks_its_members_by_label_in_declared_order(root):
    da = dataref.resolve_dataref(DataRef(**_member("weight")), datasets=[_dataset(root)])
    assert da.dims == ("subject", "node_i", "node_j")
    assert list(da["subject"].values) == MEMBERS and list(da["node_i"].values) == LABELS
    np.testing.assert_array_equal(da.values, _stack(0))


def test_member_order_is_the_cohorts_not_the_directory_listing(root):
    reverse = MEMBERS[::-1]
    da = dataref.resolve_dataref(DataRef(**_member("weight")), datasets=[_dataset(root, reverse)])
    assert list(da["subject"].values) == reverse
    np.testing.assert_array_equal(da.values, _stack(0)[::-1])


def test_a_run_reads_its_own_member_and_refuses_a_subject_outside_the_cohort(root):
    one = dataref.resolve_dataref(DataRef(**_member("length")), datasets=[_dataset(root)], subject="sub-02")
    assert one.dims == ("node_i", "node_j")
    np.testing.assert_array_equal(one.values, _raw("02")[1])
    with pytest.raises(LookupError, match="not a member of cohort 'four'"):
        dataref.resolve_dataref(DataRef(**_member("length")), datasets=[_dataset(root)], subject=OUTSIDER)


def test_an_undeclared_or_twice_declared_cohort_is_named_in_the_error(root):
    with pytest.raises(LookupError, match=r"not declared by any dataset.*\['four'\]"):
        dataref.resolve_dataref(DataRef(cohort="five", query=QUERY, output="weight"), datasets=[_dataset(root)])
    other = Dataset(dataset_id="toy_sc_copy", bids_root=str(root), cohorts=[{"cohort_id": "four", "members": MEMBERS}])
    with pytest.raises(LookupError, match="declared by 2 datasets"):
        dataref.resolve_dataref(DataRef(**_member("weight")), datasets=[_dataset(root), other])
    # One dataset reached twice, as an experiment's own that its study also lists, is one declaration.
    assert cohort.find_cohort([_dataset(root), _dataset(root)], "four")[0].dataset_id == "toy_sc"
    with pytest.raises(ValueError, match="one WHERE"):
        dataref.resolve_dataref(DataRef(experiment=3, **_member("weight")), datasets=[_dataset(root)])


def test_a_member_the_dataset_does_not_list_is_refused(root):
    listed = _dataset(root, subjects=[{"subject_id": s} for s in MEMBERS[:3]])
    with pytest.raises(ValueError, match="does not list as subjects"):
        dataref.resolve_dataref(DataRef(**_member("weight")), datasets=[listed])


def test_a_query_matching_several_of_a_subjects_files_is_refused(root):
    with pytest.raises(ValueError, match="ambiguous for sub-01"):
        cohort.subject_file(root, "01", {"desc": "SC"}, "relmat")
    assert cohort.subject_file(root, "sub-01", {"atlas": "Toy", "desc": "SC"}, "relmat").parent.name == "dwi"


def _analyses():
    common = {
        "execution": {"backend": "tvboptim"},
        "aggregate": {"over": "subject", "type": "mean"},
        "arguments": {"weight": {"used": _member("weight")}},
    }
    return [
        Analysis(name="toy_mean", equation={"rhs": "weight"}, **common),
        Analysis(name="toy_presence", equation={"rhs": "Ne(weight, 0)"}, **common),
    ]


def test_cohort_aggregates_are_equation_analyses_at_the_declared_precision(root, tmp_path):
    mean_path, presence_path = analysis_io.run_analyses(_analyses(), tmp_path / "results", datasets=[_dataset(root)])
    mean = xr.open_dataset(mean_path, engine="h5netcdf")["observation__toy_mean"].load()
    presence = xr.open_dataset(presence_path, engine="h5netcdf")["observation__toy_presence"].load()
    assert mean.dims == ("node_i", "node_j") and list(mean["node_i"].values) == LABELS
    assert mean.dtype == np.float64
    np.testing.assert_array_equal(mean.values, _stack(0).mean(0))
    np.testing.assert_array_equal(presence.values, (_stack(0) != 0).mean(0))
    assert presence.values[0, 3] == 0.5 and presence.values[1, 2] == 0.75


def test_the_container_states_its_equation_and_its_cohort(root, tmp_path):
    (path, _) = analysis_io.run_analyses(_analyses(), tmp_path / "results", datasets=[_dataset(root)])
    record = yaml.safe_load(dataref.sidecar_path(path).read_text())
    assert record["equation"] == {"rhs": "weight"} and record["aggregate"] == {"over": "subject", "type": "mean"}
    assert "callable" not in record
    assert record["arguments"]["weight"]["used"] == {
        "cohort": "four",
        "output": "weight",
        "query": {"atlas": "Toy", "desc": "SC", "suffix": "relmat"},
    }
    assert record["cohorts"] == {"four": {"dataset": "toy_sc", "n_members": 4}}


def test_changing_a_cohorts_members_changes_the_analysis_digest(root):
    analysis = _analyses()[0]
    four = analysis_io.used_cohorts(analysis, [_dataset(root)])
    three = analysis_io.used_cohorts(analysis, [_dataset(root, MEMBERS[:3])])
    assert _analysis_fingerprint(analysis, four) != _analysis_fingerprint(analysis, three)
    assert _analysis_fingerprint(analysis, {}) == _analysis_fingerprint(analysis)


def test_verify_names_an_analysis_whose_cohort_the_spec_no_longer_declares(root, tmp_path):
    """A cohort renamed after its container was written is one problem ``verify`` reports for that analysis, not a traceback."""
    from types import SimpleNamespace

    from tvbo.data.study_manifest import _stale_or_missing_analyses

    results = tmp_path / "results"
    analyses = _analyses()[:1]
    analysis_io.run_analyses(analyses, results, datasets=[_dataset(root)])
    renamed = Dataset(dataset_id="toy_sc", bids_root=str(root), cohorts=[{"cohort_id": "renamed", "members": MEMBERS}])

    problems = _stale_or_missing_analyses(
        SimpleNamespace(analyses=analyses, datasets=[renamed]), results, tmp_path / "spec.yaml"
    )

    assert len(problems) == 1 and problems[0].startswith("toy_mean: ") and "four" in problems[0]


KEEP = "cohort_presence >= min_presence"
WEIGHT_RULE = f"Piecewise((sqrt(weight) / max(sqrt(cohort_mean), {KEEP}) / 10, {KEEP}), (0, True))"
LENGTH_RULE = f"Piecewise((length, {KEEP}), (0, True))"


def _network_spec(min_presence: float = 0.75):
    """A model network on its own node order whose layers are the bound subject's and whose transforms read the cohort aggregates."""
    aggregates = {
        "cohort_mean": {"used": {"analysis": "toy_mean", "output": "toy_mean", "reconcile": "by_label"}},
        "cohort_presence": {"used": {"analysis": "toy_presence", "output": "toy_presence", "reconcile": "by_label"}},
        "min_presence": {"value": min_presence},
    }
    return {
        "nodes": [{"id": i, "label": lbl} for i, lbl in enumerate(reversed(LABELS))],
        "edges": [
            {"label": "weight", "used": _member("weight", reconcile="by_label")},
            {"label": "length", "used": _member("length", reconcile="by_label")},
        ],
        "transforms": [
            {"name": "weight", "equation": {"rhs": WEIGHT_RULE}, "arguments": aggregates},
            {
                "name": "length",
                "equation": {"rhs": LENGTH_RULE},
                "arguments": {k: v for k, v in aggregates.items() if k != "cohort_mean"},
            },
        ],
    }


def _expected(subject: str, min_presence: float = 0.75):
    """The normalisation evaluated directly, on the model network's reversed node order."""
    w, length = (m.astype(float)[::-1, ::-1] for m in _raw(subject))
    stack = _stack(0)[:, ::-1, ::-1]
    keep = (stack != 0).mean(0) >= min_presence
    scale = np.sqrt(np.where(keep, stack.mean(0), 0.0)).max()
    return np.where(keep, np.sqrt(w) / scale / 10.0, 0.0), np.where(keep, length, 0.0), w


@pytest.fixture
def results(root, tmp_path):
    out = tmp_path / "results"
    analysis_io.run_analyses(_analyses(), out, datasets=[_dataset(root)])
    return out


@pytest.mark.parametrize("subject", ["01", SHUFFLED])
def test_a_declared_rule_normalises_the_bound_subject_on_read(root, results, subject):
    net = Network(**_network_spec()).bind_references(subject=subject, datasets=[_dataset(root)], results_root=results)
    weight, length, raw = _expected(subject)
    np.testing.assert_allclose(net.matrix("weight", format="dense"), weight, rtol=1e-6, atol=0)
    np.testing.assert_array_equal(net.matrix("length", format="dense"), length)
    np.testing.assert_array_equal(net.matrix("weight", format="dense", apply_transforms=False), raw)
    assert net.matrix("weight", format="dense")[3, 0] == 0.0 and raw[3, 0] != 0.0


def test_rebinding_another_subject_rereads_every_sourced_layer(root, results):
    net = Network(**_network_spec())
    for subject in ("01", "04", "01"):
        net.bind_references(subject=subject, datasets=[_dataset(root)], results_root=results)
        np.testing.assert_allclose(net.matrix("weight", format="dense"), _expected(subject)[0], rtol=1e-6)


def test_the_threshold_is_the_declared_argument(root, results):
    net = Network(**_network_spec(min_presence=0.5)).bind_references(
        subject="01", datasets=[_dataset(root)], results_root=results
    )
    np.testing.assert_allclose(net.matrix("weight", format="dense"), _expected("01", 0.5)[0], rtol=1e-6)
    assert net.matrix("weight", format="dense")[3, 0] != 0.0


def test_a_sourced_layer_with_nothing_bound_raises_instead_of_reading_something_else(root, results):
    net = Network(**_network_spec())
    with pytest.raises(LookupError, match="not declared by any dataset"):
        net.matrix("weight", format="dense")
    net.bind_references(datasets=[_dataset(root)], results_root=results)
    with pytest.raises(ValueError, match="no subject bound"):
        net.matrix("weight", format="dense")


def test_an_authored_layer_survives_on_a_network_that_also_names_a_data_file(root, results, tmp_path):
    w = np.arange(16, dtype=float).reshape(4, 4)
    Network.from_matrix(weights=w, lengths=w + 100.0, labels=LABELS[::-1]).save(tmp_path / "group_relmat.yaml")
    base = yaml.safe_load((tmp_path / "group_relmat.yaml").read_text())
    spec = {
        **base,
        "data_file": str(tmp_path / base["data_file"]),
        "edges": [_network_spec()["edges"][0], {"label": "length"}],
    }
    net = Network(**spec).bind_references(subject="01", datasets=[_dataset(root)], results_root=results)
    np.testing.assert_array_equal(net.matrix("weight", format="dense"), _expected("01")[2])
    np.testing.assert_array_equal(net.matrix("length", format="dense"), w + 100.0)


def test_reading_a_companion_layer_asks_nothing_of_a_sourced_one(root, tmp_path):
    w = np.arange(16, dtype=float).reshape(4, 4)
    Network.from_matrix(weights=w, lengths=w + 100.0, labels=LABELS[::-1]).save(tmp_path / "group_relmat.yaml")
    base = yaml.safe_load((tmp_path / "group_relmat.yaml").read_text())
    spec = {
        **base,
        "data_file": str(tmp_path / base["data_file"]),
        "edges": [_network_spec()["edges"][0], {"label": "length"}],
    }
    net = Network(**spec)
    np.testing.assert_array_equal(net.matrix("length", format="dense"), w + 100.0)
    assert net.sources_layer("weights") and not net.sources_layer("length") and net.declares_references()
    with pytest.raises(LookupError, match="not declared by any dataset"):
        net.matrix("weight", format="dense")


def test_one_argument_name_means_one_reference_within_a_chain(root):
    spec = _network_spec()
    second = {"name": "weights", "equation": {"rhs": "weight * cohort_mean"}, "arguments": {}}
    second["arguments"]["cohort_mean"] = {"used": {"analysis": "toy_presence", "output": "toy_presence"}}
    spec["transforms"].insert(1, second)
    with pytest.raises(ValueError, match="'cohort_mean' twice with different `used:` references"):
        Network(**spec).sourced_transform_arguments("weight")
    assert sorted(Network(**_network_spec()).sourced_transform_arguments("weight")) == ["cohort_mean", "cohort_presence"]
    assert list(Network(**_network_spec()).sourced_transform_arguments("length")) == ["cohort_presence"]


def _study_file(root, tmp_path, dataset=None, observations=None, **network_extra):
    """A study whose one experiment relaxes ``dx/dt = 1 - x + sum_j w_ij x_j`` on the bound subject's normalised connectome, optionally with its own *dataset* of subjects to fan over and *observations*."""
    spec = {
        "key": "ToyCohort",
        "datasets": [yaml.safe_load(yaml.safe_dump({"dataset_id": "toy_sc", "bids_root": str(root)}))],
        "experiments": [
            {
                "id": 1,
                "label": "relaxation on the subject's own connectome",
                "dynamics": {
                    "name": "Relaxation",
                    "system_type": "continuous",
                    "output": ["x"],
                    "coupling_inputs": {"c": {}},
                    "state_variables": {
                        "x": {"equation": {"rhs": "1 - x + c"}, "initial_value": 0.0, "coupling_variable": True}
                    },
                },
                "network": {
                    **_network_spec(),
                    **network_extra,
                    "coupling": {
                        "c": {
                            "delayed": False,
                            "incoming_states": ["x"],
                            "pre_expression": {"rhs": "x"},
                            "post_expression": {"rhs": "gx"},
                        }
                    },
                },
                "integration": {"method": "heun", "step_size": 0.05, "duration": 40.0, "transient_time": 0.0, "unit": "ms"},
                "execution": {"precision": "float64"},
            }
        ],
    }
    spec["datasets"][0]["cohorts"] = [{"cohort_id": "four", "members": list(MEMBERS)}]
    if dataset is not None:
        spec["experiments"][0]["dataset"] = dataset
    if observations is not None:
        spec["experiments"][0]["observations"] = observations
    path = tmp_path / "ToyCohort.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


def test_generated_solver_code_takes_cohort_arguments_from_its_run_and_needs_no_subject_to_render(root, tmp_path):
    pytest.importorskip("tvboptim")
    from tvbo import SimulationStudy

    experiment = SimulationStudy.from_file(str(_study_file(root, tmp_path))).get_experiment(1)
    code = experiment.render_code("tvboptim")
    assert (
        'cohort_mean = _network_datum("cohort_mean")' in code and 'cohort_presence = _network_datum("cohort_presence")' in code
    )
    assert "DenseGraph(" in code

    plain = SimulationStudy.from_file(str(_study_file(root, tmp_path, transforms=[]))).get_experiment(1)
    assert "_network_datum" not in plain.render_code("tvboptim") and "_NETWORK_DATA" not in plain.render_code("tvboptim")


@pytest.mark.parametrize("subject", ["01", SHUFFLED])
def test_a_per_subject_run_integrates_that_subjects_normalised_connectome(root, results, tmp_path, subject):
    pytest.importorskip("tvboptim")
    from tvbo import SimulationStudy

    experiment = SimulationStudy.from_file(str(_study_file(root, tmp_path))).get_experiment(1)
    result = experiment.run(format="tvboptim", active_subject=subject, results_root=results)
    settled = np.asarray(result.integration.data.isel(time=-1)).ravel()
    weight = _expected(subject)[0]
    np.testing.assert_allclose(settled, np.linalg.solve(np.eye(4) - weight, np.ones(4)), rtol=1e-6)
    assert not np.allclose(settled, np.linalg.solve(np.eye(4) - _expected("02")[0], np.ones(4)), rtol=1e-3)


def test_a_run_with_no_subject_refuses_a_connectome_that_is_one_subjects(root, results, tmp_path):
    pytest.importorskip("tvboptim")
    from tvbo import SimulationStudy

    experiment = SimulationStudy.from_file(str(_study_file(root, tmp_path))).get_experiment(1)
    with pytest.raises(ValueError, match="no subject bound"):
        experiment.run(format="tvboptim", results_root=results)
    with pytest.raises(LookupError, match="not a member of cohort 'four'"):
        experiment.run(format="tvboptim", active_subject=OUTSIDER, results_root=results)


LISTED = ["04", OUTSIDER, "01", "07"]


def _fan_out_experiment(root, tmp_path, fc_root=None, **network_extra):
    """The relaxation experiment with its own dataset listing two cohort members and two subjects outside the cohort, fitting each subject's FC under *fc_root* when one is given."""
    from tvbo import SimulationStudy

    dataset = {"dataset_id": "toy_fc", "subjects": [{"subject_id": s} for s in LISTED]}
    observations = None
    if fc_root is not None:
        dataset["bids_root"] = str(fc_root)
        query = {"atlas": "Toy", "desc": "FC", "suffix": "relmat"}
        observations = {"fc_target": {"source": ["dataset.subject.fc"], "query": query, "reconcile": "by_label"}}
    path = _study_file(root, tmp_path, dataset=dataset, observations=observations, **network_extra)
    return SimulationStudy.from_file(str(path)).get_experiment(1)


def test_a_network_read_per_subject_fans_over_the_cohort_members_among_the_dataset_subjects(root, tmp_path):
    experiment = _fan_out_experiment(root, tmp_path)
    assert experiment.network.per_subject_cohorts() == ["four"]
    assert experiment.dataset_subject_ids() == ["04", "01"]
    assert experiment.subject_selection() == {
        "dataset": 4,
        "cohorts": {"four": 4},
        "subjects": ["04", "01"],
        "excluded": [OUTSIDER, "07"],
    }


def test_a_network_reading_no_cohort_fans_over_every_dataset_subject(root, tmp_path):
    experiment = _fan_out_experiment(root, tmp_path, edges=[{"label": "weight"}], transforms=[])
    assert experiment.network.per_subject_cohorts() == []
    assert experiment.dataset_subject_ids() == LISTED
    assert experiment.subject_selection()["cohorts"] == {} and experiment.subject_selection()["excluded"] == []


def test_the_workflow_plan_and_kit_readme_state_the_cohort_restriction(root, tmp_path):
    from tvbo.cli.workflow import _plan_payload, _write_readme
    from tvbo.run.workflow import plan

    built = plan(study_key="ToyCohort", experiment=_fan_out_experiment(root, tmp_path), backend="tvboptim", engine="slurm")
    (axis,) = [ax for ax in built.workflow_axes if ax.name == "subject"]
    assert axis.values == ("04", "01")
    assert _plan_payload(built)["subject_selection"]["excluded"] == [OUTSIDER, "07"]
    line = "2 of 4 dataset subjects, the members of four (4 members); 2 excluded"
    assert built.subject_selection_text == line
    _write_readme(tmp_path, engine="slurm", plans=[built], script_relpath=None, spec_layout="spec/")
    assert f"- subjects      : {line}" in (tmp_path / "README.md").read_text()


def test_a_network_layer_read_from_another_experiment_orders_the_plan_after_it(root, tmp_path):
    from tvbo.run.workflow import plan

    edges = [{"label": "weight", "used": {"experiment": 3, "output": "weight"}}, _network_spec()["edges"][1]]
    experiment = _fan_out_experiment(root, tmp_path, edges=edges, transforms=[])
    built = plan(study_key="ToyCohort", experiment=experiment, backend="tvboptim", engine="slurm")
    assert built.depends_on == ["3"]


def test_planning_a_per_subject_network_whose_cohort_no_dataset_declares_raises(root, tmp_path):
    from tvbo.run.workflow import plan

    experiment = _fan_out_experiment(root, tmp_path)
    experiment._study_datasets = None
    with pytest.raises(LookupError, match="cohort 'four' is not declared by any dataset"):
        plan(study_key="ToyCohort", experiment=experiment, backend="tvboptim", engine="slurm")


def _write_fc(fc_root, subject: str):
    """One subject's functional connectome, a decoy for the structural query to tell apart from its own file."""
    net = Network.from_matrix(weights=np.eye(4), labels=LABELS)
    net.set_matrix("fc", np.full((4, 4), int(subject) / 10.0))
    folder = fc_root / f"sub-{subject}" / "func"
    folder.mkdir(parents=True, exist_ok=True)
    net.save(folder / f"sub-{subject}_atlas-Toy_desc-FC_relmat.yaml")


def test_a_bundled_kit_carries_the_cohort_each_members_connectome_and_the_aggregates(root, tmp_path):
    from tvbo import SimulationExperiment
    from tvbo.cli import workflow as kit
    from tvbo.utils.study_layout import study_path

    (tmp_path / "dataset_description.json").write_text('{"Name": "ToyCohort", "BIDSVersion": "1.9.0"}')
    analysis_io.run_analyses(_analyses(), study_path("results", root=tmp_path), datasets=[_dataset(root)])
    fc_root = tmp_path / "functional_connectomes"
    for subject in LISTED:
        _write_fc(fc_root, subject)
    experiment = _fan_out_experiment(root, tmp_path, fc_root=fc_root)

    out = tmp_path / "kit"
    spec_dir = out / "spec"
    bundle = kit._bundle_dataset(experiment, out, spec_dir, {})
    (spec_dir / "experiment.yaml").write_text(kit._freeze_spec_yaml(experiment, spec_dir, dataset_bids_root=bundle))
    assert kit._stage_reference_containers(experiment, out / "derivatives/tvbo") == ["toy_mean", "toy_presence"]

    data = out / "data/toy_fc"
    assert sorted(p.name for p in data.iterdir()) == ["sub-01", "sub-04"]
    assert sorted(p.name for p in (data / "sub-04").glob("*.yaml")) == [
        "sub-04_atlas-Toy_desc-FC_relmat.yaml",
        "sub-04_atlas-Toy_desc-SC_relmat.yaml",
    ]
    frozen = yaml.safe_load((spec_dir / "experiment.yaml").read_text())
    assert frozen["dataset"]["bids_root"] == "../data/toy_fc"
    assert list(frozen["dataset"]["subjects"]) == ["04", "01"]
    assert [(c["cohort_id"], c["members"]) for c in frozen["dataset"]["cohorts"]] == [("four", ["04", "01"])]

    rerun = SimulationExperiment.from_file(str(spec_dir / "experiment.yaml"))
    assert rerun.dataset_subject_ids() == ["04", "01"]
    rerun._active_subject = "01"
    rerun._bind_network_references(results_root=out / "derivatives/tvbo")
    np.testing.assert_allclose(rerun.network.matrix("weight", format="dense"), _expected("01")[0], rtol=1e-6)
    np.testing.assert_array_equal(rerun.network.matrix("length", format="dense"), _expected("01")[1])

    rerun.freeze_yaml(str(tmp_path / "provenance"), network_stem="sub-01_network")
    with h5py.File(tmp_path / "provenance/sub-01_network.h5") as companion:
        for layer in ("weight", "length"):
            stored, _ = read_edge(companion, layer)
            np.testing.assert_array_equal(
                np.asarray(stored.todense() if hasattr(stored, "todense") else stored), rerun.network.array(layer)
            )


def test_a_bundle_carries_a_network_read_per_subject_without_a_dataset_target(root, tmp_path):
    """A dataset that only lists its subjects has no target to bundle, and its fan-out still needs each member's own connectome."""
    manifest = _fan_out_experiment(root, tmp_path).dataset_bundle_files({})
    assert {s: [f.name for f in files if f.suffix == ".yaml"] for s, files in manifest.items()} == {
        "04": ["sub-04_atlas-Toy_desc-SC_relmat.yaml"],
        "01": ["sub-01_atlas-Toy_desc-SC_relmat.yaml"],
    }


def test_a_kit_stages_the_results_its_experiment_reads_from_experiments_it_does_not_run(root, tmp_path):
    import typer

    from tvbo.cli import workflow as kit
    from tvbo.utils.study_layout import study_path

    (tmp_path / "dataset_description.json").write_text('{"Name": "ToyCohort", "BIDSVersion": "1.9.0"}')
    results = study_path("results", root=tmp_path)
    analysis_io.run_analyses(_analyses(), results, datasets=[_dataset(root)])
    group = [f"exp-9_model-Toy_{part}" for part in ("result.h5", "result.yaml", "network.h5", "network.yaml")]
    shards = [f"sub-{s}_exp-8_model-Toy_result.h5" for s in LISTED]
    for name in [*group, *shards, "exp-90_model-Toy_result.h5"]:
        (results / name).write_bytes(name.encode())
    experiment = _fan_out_experiment(root, tmp_path)

    staged_dir = tmp_path / "kit/derivatives/tvbo"
    staged = kit._stage_reference_containers(experiment, staged_dir, depends_on=["9", "8", "7"], in_kit=["7"])
    assert staged == ["toy_mean", "toy_presence", "exp-9", "exp-8"]
    held = sorted(p.name for p in staged_dir.rglob("*exp-*"))
    assert held == sorted([*group, "sub-01_exp-8_model-Toy_result.h5", "sub-04_exp-8_model-Toy_result.h5"])
    assert dataref.locate_exp_container(staged_dir, "8", subject="01").read_bytes() == b"sub-01_exp-8_model-Toy_result.h5"
    with pytest.raises(typer.Exit):
        kit._stage_reference_containers(experiment, staged_dir, depends_on=["6"])


def test_a_kit_carries_the_layer_a_network_reads_per_subject_without_any_dataset_target(root, tmp_path):
    from tvbo.cli import workflow as kit

    out = tmp_path / "kit"
    assert kit._bundle_dataset(_fan_out_experiment(root, tmp_path), out, out / "spec") == "../data/toy_fc"
    assert sorted(p.name for p in (out / "data/toy_fc/sub-04").iterdir()) == [
        "sub-04_atlas-Toy_desc-SC_relmat.h5",
        "sub-04_atlas-Toy_desc-SC_relmat.yaml",
    ]
