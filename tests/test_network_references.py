"""A reference whose ``iri`` names a network reads one of that network's arrays, as the network delivers it.

Two things are held here. A network on disk, by its sidecar path or its exact curated id, hands a reference any edge layer or per-node quantity by name, labelled by node. A network that itself reads per subject, a declared layer under a declared rule, hands over what a run on it integrates: the bound subject's layer with the rule applied, or every member's stacked along ``subject`` when nothing is bound. An analysis over such a reference therefore measures the network instead of restating its rule, and it is ordered after, and goes stale with, whatever the network reads.

The cohort, the rule and every expected value are those of ``test_cohort_references``: computed in numpy from the same raw matrices.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
import yaml

pytest.importorskip("jax")

from tests import test_cohort_references as cohort_case  # noqa: E402
from tests.test_cohort_references import (  # noqa: E402
    LABELS,
    MEMBERS,
    SHUFFLED,
    _analyses,
    _dataset,
    _expected,
    _network_spec,
    _raw,
    _study_file,
)
from tvbo import Network  # noqa: E402
from tvbo.data import analysis_io, dataref, network_io  # noqa: E402
from tvbo.data.study_manifest import _analysis_fingerprint  # noqa: E402
from tvbo.datamodel.schema import Analysis, DataRef, Dataset  # noqa: E402

root, results = cohort_case.root, cohort_case.results  # the cohort on disk and its two aggregate containers
CURATED = "tvbo:tpl-MNI152NLin2009cAsym_cohort-HCPYA_rec-dTOR_atlas-Lobar_desc-SC_relmat"


def _member_file(root, subject: str = "01"):
    return root / f"sub-{subject}" / "dwi" / f"sub-{subject}_atlas-Toy_desc-SC_relmat.yaml"


@pytest.fixture
def ruled(tmp_path):
    """The model network of ``test_cohort_references`` written as a file: its layers are the bound subject's and its transforms the cohort-relative rule."""
    path = tmp_path / "toy_network.yaml"
    path.write_text(yaml.safe_dump(_network_spec(), sort_keys=False))
    return path


def _through(ruled, output: str = "weight"):
    return DataRef(iri=str(ruled), output=output)


def _ruled_mean(ruled, name: str = "toy_ruled_mean"):
    """The cohort mean of the connectome a run integrates: an analysis over the network's own delivered layer."""
    return Analysis(
        name=name,
        equation={"rhs": "weight"},
        execution={"backend": "tvboptim", "precision": "float64"},
        aggregate={"over": "subject", "type": "mean"},
        arguments={"weight": {"used": {"iri": str(ruled), "output": "weight"}}},
    )


@pytest.mark.parametrize("layer", [0, 1], ids=["weight", "length"])
def test_a_reference_reads_a_layer_of_a_network_file_by_its_edge_name(root, layer):
    name = ["weight", "length"][layer]
    da = dataref.resolve_dataref(DataRef(iri=str(_member_file(root)), output=name))
    assert da.dims == ("node_i", "node_j") and list(da["node_i"].values) == LABELS
    np.testing.assert_array_equal(da.values, _raw("01")[layer])


def test_a_layer_read_from_a_network_is_reconciled_by_node_label_like_any_other(root):
    """The member stored with its nodes permuted comes back on the consumer's order when the reference asks for ``by_label``."""
    ref = DataRef(iri=str(_member_file(root, SHUFFLED)), output="weight", reconcile="by_label")
    da = dataref.resolve_dataref(ref, alias_map={lbl: lbl for lbl in LABELS}, model_labels=LABELS)
    np.testing.assert_array_equal(np.asarray(da), _raw(SHUFFLED)[0])


def test_a_network_path_written_against_the_study_is_found_from_any_directory(root, tmp_path, monkeypatch):
    study = tmp_path / "Study"
    (study / "sourcedata" / "networks").mkdir(parents=True)
    (study / "dataset_description.json").write_text('{"Name": "Study", "BIDSVersion": "1.9.0"}')
    net = Network.from_matrix(weights=_raw("01")[0], lengths=_raw("01")[1], labels=LABELS)
    net.save(study / "sourcedata" / "networks" / "group.yaml")
    monkeypatch.chdir(tmp_path)
    ref = DataRef(iri="sourcedata/networks/group.yaml", output="weight")
    da = dataref.resolve_dataref(ref, source_dir=str(study / "spec"))
    np.testing.assert_array_equal(da.values, _raw("01")[0])
    assert dataref.network_file("sourcedata/networks/group.yaml") is None, "with no study named, a path is read as written"


def test_a_curated_network_is_named_by_its_exact_id_and_by_nothing_looser():
    da = dataref.resolve_dataref(DataRef(iri=CURATED, output="weight"))
    curated = Network.from_db(CURATED.split(":", 1)[1])
    np.testing.assert_array_equal(da.values, curated.matrix("weight", format="dense"))
    assert list(da["node_i"].values) == [str(lbl) for lbl in curated.node_labels]
    assert dataref.network_file("tvbo:Lobar") is None, "an atlas shorthand matches several networks, so it names none"


@pytest.mark.parametrize("iri", ["tvbo:exp/Study/exp-3", "tvbo:ana/Study/toy_mean", "exp-3", "3"])
def test_an_iri_that_names_a_result_is_no_network(iri):
    assert dataref.network_file(iri) is None


def test_a_result_container_named_by_path_is_still_read_as_a_container(root, results):
    container = analysis_io.container_path("toy_mean", results)
    assert dataref.network_file(str(container)) is None
    da = dataref.resolve_dataref(DataRef(iri=str(container), output="toy_mean"))
    assert da.dims == ("node_i", "node_j")


@pytest.mark.parametrize("subject", ["01", SHUFFLED])
def test_a_reference_through_a_network_reads_the_bound_subjects_layer_as_the_network_delivers_it(
    root, results, ruled, subject
):
    """The rule is applied by the network, so the array is what a run of that subject integrates, on the network's own node order."""
    kwargs = {"subject": subject, "datasets": [_dataset(root)], "results_root": results}
    weight = dataref.resolve_dataref(_through(ruled), **kwargs)
    assert list(weight["node_i"].values) == LABELS[::-1]
    np.testing.assert_allclose(weight.values, _expected(subject)[0], rtol=1e-6, atol=0)
    np.testing.assert_array_equal(dataref.resolve_dataref(_through(ruled, "length"), **kwargs).values, _expected(subject)[1])
    assert not np.allclose(weight.values, _expected(subject)[2]), "the raw layer would mean the rule was skipped"


def test_with_no_subject_a_reference_through_a_network_is_every_member_stacked(root, results, ruled):
    da = dataref.resolve_dataref(_through(ruled), datasets=[_dataset(root)], results_root=results)
    assert da.dims == ("subject", "node_i", "node_j") and list(da["subject"].values) == MEMBERS
    for subject in MEMBERS:
        np.testing.assert_allclose(da.sel(subject=subject).values, _expected(subject)[0], rtol=1e-6, atol=0)


def test_an_analysis_over_a_network_measures_what_the_network_delivers(root, results, ruled):
    """The cohort mean of the ruled connectome, with the rule written once, in the network."""
    (path,) = analysis_io.run_analyses([_ruled_mean(ruled)], results, datasets=[_dataset(root)])
    mean = xr.open_dataset(path, engine="h5netcdf")["observation__toy_ruled_mean"].load()
    np.testing.assert_allclose(mean.values, np.mean([_expected(s)[0] for s in MEMBERS], axis=0), rtol=1e-6, atol=0)
    record = yaml.safe_load(dataref.sidecar_path(path).read_text())
    assert record["cohorts"] == {"four": {"dataset": "toy_sc", "n_members": 4}}, (
        "the cohort the network reads is the analysis's"
    )


def test_an_analysis_that_reads_through_a_network_runs_after_what_the_network_reads(root, ruled):
    analysis = _ruled_mean(ruled)
    assert analysis_io.dependencies(analysis)["analyses"] == {"toy_mean", "toy_presence"}
    before, after = analysis_io.schedule([analysis, *_analyses()])
    assert [analysis_io.analysis_name(a) for a in before] == ["toy_mean", "toy_presence", "toy_ruled_mean"] and not after
    assert analysis_io.dependents_of([analysis, *_analyses()], changed_analyses={"toy_presence"}) == [
        "toy_ruled_mean",
        "toy_presence",
    ]
    with pytest.raises(KeyError, match="toy_mean"):
        analysis_io.schedule([analysis])


def _digest(analysis, root, members=MEMBERS):
    datasets = [_dataset(root, members=members)]
    return _analysis_fingerprint(analysis, analysis_io.used_cohorts(analysis, datasets), analysis_io.network_digests(analysis))


def test_an_analysis_goes_stale_with_the_networks_rule_its_cohort_and_its_arrays(root, ruled, tmp_path):
    analysis = _ruled_mean(ruled)
    written = _digest(analysis, root)
    assert _digest(analysis, root) == written
    assert _digest(analysis, root, members=MEMBERS[:3]) != written, "the cohort the network reads per subject"
    ruled.write_text(yaml.safe_dump(_network_spec(min_presence=0.5), sort_keys=False))
    assert _digest(analysis, root) != written, "the rule the network declares"

    held = tmp_path / "group.yaml"
    Network.from_matrix(weights=_raw("01")[0], labels=LABELS).save(held)
    over_a_file = _ruled_mean(held, name="toy_group")
    before = _digest(over_a_file, root)
    Network.from_matrix(weights=_raw("02")[0], labels=LABELS).save(held)
    assert _digest(over_a_file, root) != before, "the arrays a network holds itself"


def test_a_network_reading_cohorts_with_different_members_has_no_one_set_of_subjects(root, results, tmp_path):
    spec = _network_spec()
    spec["edges"][1]["used"] = {**spec["edges"][1]["used"], "cohort": "three"}
    path = tmp_path / "two_cohorts.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    cohorts = [{"cohort_id": "four", "members": MEMBERS}, {"cohort_id": "three", "members": MEMBERS[:3]}]
    dataset = Dataset(dataset_id="toy_sc", bids_root=str(root), cohorts=cohorts)
    with pytest.raises(ValueError, match="whose members differ"):
        dataref.resolve_dataref(_through(path), datasets=[dataset], results_root=results)


def test_a_reference_to_a_network_needs_the_array_it_takes(root):
    with pytest.raises(ValueError, match="needs `output:`"):
        dataref.resolve_dataref(DataRef(iri=str(_member_file(root))))
    with pytest.raises(KeyError, match="neither an edge matrix nor a per-node quantity named 'tracts'"):
        dataref.resolve_dataref(DataRef(iri=str(_member_file(root)), output="tracts"))


def test_a_spec_fragment_is_read_as_the_study_that_includes_it_reads_it(root, tmp_path, monkeypatch):
    """A fragment under ``spec/`` names its data file from the study root, as the recipe that includes it does, wherever the reader stands."""
    study = tmp_path / "Study"
    (study / "spec").mkdir(parents=True)
    (study / "sourcedata").mkdir()
    (study / "dataset_description.json").write_text('{"Name": "Study", "BIDSVersion": "1.9.0"}')
    Network.from_matrix(weights=_raw("01")[0], labels=LABELS).save(study / "sourcedata" / "group.yaml")
    fragment = study / "spec" / "desc-group_network.yaml"
    fragment.write_text("tvbo_class: tvbo:Network\ndata_file: sourcedata/group.yaml\nlabel: the group connectome\n")
    monkeypatch.chdir(tmp_path)
    np.testing.assert_array_equal(network_io.load_declared_network(fragment).matrix("weight", format="dense"), _raw("01")[0])
    np.testing.assert_array_equal(dataref.resolve_dataref(DataRef(iri=str(fragment), output="weight")).values, _raw("01")[0])
    with pytest.raises(
        KeyError, match="spec/sourcedata/group.yaml"
    ):  # read where it stands, the path means a file beside the fragment
        network_io.load_network(fragment).matrix("weight", format="dense")


def test_the_cohort_scale_of_a_ruled_connectome_is_one_number_from_one_expression(root, results, ruled, tmp_path):
    """The use the reference was built for: scale the cohort's ruled connectomes to a reference connectome's mean weight, with neither the rule nor the reference's value restated. The expression reduces every axis, so it names none."""
    reference = tmp_path / "reference.yaml"
    Network.from_matrix(weights=_raw("04")[0] / 1000.0, labels=LABELS).save(reference)
    scale = Analysis(
        name="toy_scale",
        equation={"rhs": "mean(reference) / mean(weight)"},
        execution={"backend": "tvboptim", "precision": "float64"},
        arguments={
            "weight": {"used": {"iri": str(ruled), "output": "weight"}},
            "reference": {"used": {"iri": str(reference), "output": "weight"}},
        },
    )
    (path,) = analysis_io.run_analyses([scale], results, datasets=[_dataset(root)])
    value = xr.open_dataset(path, engine="h5netcdf")["observation__toy_scale"].load()
    assert value.dims == ()
    expected = (_raw("04")[0].astype(float) / 1000.0).mean() / np.mean([_expected(s)[0] for s in MEMBERS])
    np.testing.assert_allclose(float(value), expected, rtol=1e-6)

    scaled = _network_spec()
    scaled["transforms"].insert(
        1,
        {
            "name": "weights",
            "equation": {"rhs": "weight * scale"},
            "arguments": {"scale": {"used": {"analysis": "toy_scale", "output": "toy_scale"}}},
        },
    )
    net = Network(**scaled).bind_references(subject="01", datasets=[_dataset(root)], results_root=results)
    np.testing.assert_allclose(net.matrix("weight", format="dense"), _expected("01")[0] * expected, rtol=1e-6, atol=0)


def test_a_kit_refuses_an_experiment_that_reads_a_network_file_through_a_reference(root, ruled, tmp_path):
    """A kit carries dataset files and result containers, and no network a reference names, so it says so before it is written instead of failing on the node."""
    import typer

    from tvbo import SimulationStudy
    from tvbo.cli import workflow as kit

    plain = SimulationStudy.from_file(str(_study_file(root, tmp_path))).get_experiment(1)
    kit._refuse_network_references(plain)
    through = {
        "name": "weights",
        "equation": {"rhs": "weight * mean(other)"},
        "arguments": {"other": {"used": {"iri": str(ruled), "output": "weight"}}},
    }
    spec = _network_spec()
    reading = SimulationStudy.from_file(
        str(_study_file(root, tmp_path, transforms=[*spec["transforms"], through]))
    ).get_experiment(1)
    with pytest.raises(typer.Exit):
        kit._refuse_network_references(reading)


def test_a_kit_accepts_an_experiment_that_reads_a_curated_network_through_a_reference(root, tmp_path):
    """A curated network is installed with the package on the node, so a reference to it by id is not a file the kit would have to carry."""
    from tvbo import SimulationStudy
    from tvbo.cli import workflow as kit

    curated = "tvbo:tpl-MNI152NLin2009cAsym_cohort-HCPYA_rec-dTOR_atlas-DesikanKilliany_desc-SC_relmat"
    through = {
        "name": "weights",
        "equation": {"rhs": "weight * mean(other)"},
        "arguments": {"other": {"used": {"iri": curated, "output": "weight"}}},
    }
    spec = _network_spec()
    reading = SimulationStudy.from_file(
        str(_study_file(root, tmp_path, transforms=[*spec["transforms"], through]))
    ).get_experiment(1)
    kit._refuse_network_references(reading)


def test_a_function_fragment_is_a_spec_file_the_validator_knows(tmp_path):
    """A transform shared by two networks is one fragment each includes, named for the class it declares like every other fragment."""
    from tvbo.cli.validate import _suffix_problems

    fragment = tmp_path / "desc-publishedRule_function.yaml"
    fragment.write_text('tvbo_class: tvbo:Function\nname: weight\nequation: {rhs: "sqrt(weight)"}\n')
    assert _suffix_problems(fragment) == []
    network = tmp_path / "desc-ruled_network.yaml"
    network.write_text(
        "tvbo_class: tvbo:Network\nnumber_of_nodes: 2\ntransforms:\n  - !include desc-publishedRule_function.yaml\n"
    )
    assert [str(t.name) for t in network_io.load_network(network).transforms] == ["weight"]
