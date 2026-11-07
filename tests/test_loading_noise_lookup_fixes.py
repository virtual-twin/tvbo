"""Loading, noise-amplitude, result-lookup and export contracts that used to hold only by accident.

* An experiment is parsed once, and every mapping reaches the record in the order its author declared it; a dump and reparse between two parses sorted them alphabetically.
* A noise amplitude is read by one reader, `tvbo.utils.noise_sigma`, whether it is spelled `sigma` or `nsig`, and reading it writes nothing back into the record.
* A result lookup that several saved runs answer raises instead of opening whichever sorts first; one that nothing answers still reads as "not run yet".
* A BEP017 export writes every template edge under the name the store keys it by.
* Only a class the registry files entries of offers the database lookups.
"""

from __future__ import annotations

import pytest
import yaml

from tvbo import SimulationExperiment

_SPEC = """
id: 1
label: order
dynamics:
  name: Order
  parameters:
    zeta: {value: 1.0}
    alpha: {value: 2.0}
    mu: {value: 3.0}
  state_variables:
    x:
      equation: {rhs: "-zeta*x + alpha + mu"}
      initial_value: 0.1
integration: {method: Euler, step_size: 0.1, duration: 1.0}
"""

_DECLARED = ["zeta", "alpha", "mu"]


def _noisy(noise_yaml: str, integration_noise: str | None = None) -> SimulationExperiment:
    """A one-state experiment whose state variable declares *noise_yaml*, and whose integration declares *integration_noise*."""
    spec = yaml.safe_load(_SPEC)
    spec["dynamics"]["state_variables"]["x"]["noise"] = yaml.safe_load(noise_yaml)
    if integration_noise is not None:
        spec["integration"]["noise"] = yaml.safe_load(integration_noise)
    return SimulationExperiment.from_dict(spec)


# --------------------------------------------------------------------------- one parse, declared order


def test_from_string_keeps_the_declared_order():
    exp = SimulationExperiment.from_string(_SPEC)

    assert list(exp.dynamics.parameters) == _DECLARED
    dumped = yaml.safe_load(exp.to_yaml())
    assert list(dumped["dynamics"]["parameters"]) == _DECLARED


def test_from_file_keeps_the_declared_order_and_parses_once(tmp_path, monkeypatch):
    from tvbo.utils import yaml_loader

    path = tmp_path / "exp.yaml"
    path.write_text(_SPEC)
    parsed = []
    real = yaml_loader._parse

    def counting(source, base_dir):
        parsed.append(source)
        return real(source, base_dir)

    monkeypatch.setattr(yaml_loader, "_parse", counting)
    exp = SimulationExperiment.from_file(str(path))

    assert list(exp.dynamics.parameters) == _DECLARED
    assert parsed == [str(path)], "the recipe must be parsed exactly once"


def test_from_dict_leaves_the_callers_mapping_untouched():
    data = yaml.safe_load(_SPEC)
    before = yaml.safe_dump(data, sort_keys=False)

    exp = SimulationExperiment.from_dict(data)

    assert yaml.safe_dump(data, sort_keys=False) == before
    assert list(exp.dynamics.parameters) == _DECLARED


# --------------------------------------------------------------------------- one noise-amplitude reader


def test_a_noise_derives_either_spelling_by_one_rule():
    from tvbo.datamodel.schema import Noise

    nsig_only = Noise(parameters={"nsig": {"value": 0.5}})
    assert nsig_only.sigma == pytest.approx(1.0)
    assert nsig_only.nsig == pytest.approx(0.5)

    both = Noise(parameters={"sigma": {"value": 0.2}, "nsig": {"value": 0.5}})
    assert both.sigma == pytest.approx(0.2), "sigma wins, as noise_sigma decides"
    assert both.nsig == pytest.approx(0.02), "nsig is derived from the amplitude that wins"

    assert Noise(additive=True).sigma is None
    assert Noise(additive=True).nsig is None


def test_the_per_state_array_reads_nsig():
    """An `nsig`-only recipe used to read as a zero amplitude, i.e. a deterministic run."""
    exp = _noisy("{additive: true, parameters: {nsig: {value: 0.5}}}")

    assert list(exp.noise_sigma_array) == pytest.approx([1.0])


def test_the_per_state_array_falls_back_to_the_integration_noise():
    exp = _noisy("{additive: true}", integration_noise="{parameters: {nsig: {value: 2.0}}}")

    assert list(exp.noise_sigma_array) == pytest.approx([2.0])


def test_a_declared_zero_amplitude_is_not_replaced_by_the_integration_noise():
    exp = _noisy(
        "{additive: true, parameters: {sigma: {value: 0.0}}}", integration_noise="{parameters: {sigma: {value: 2.0}}}"
    )

    assert list(exp.noise_sigma_array) == [0.0]


def test_reading_the_amplitude_writes_nothing_back():
    """The property used to plant `state_wise_sigma` and a default `integration.noise` on the record it read."""
    exp = _noisy("{additive: true, parameters: {sigma: {value: 0.3}}}")
    before = exp.to_yaml()

    _ = exp.noise_sigma_array
    stand_in = exp.run_noise

    assert exp.to_yaml() == before
    assert exp.integration.noise is None
    assert stand_in is not None and stand_in.additive, "a per-state amplitude still makes the run stochastic"


def test_a_declared_integration_noise_is_the_run_noise():
    exp = _noisy("{additive: true}", integration_noise="{additive: false, parameters: {sigma: {value: 0.1}}}")

    assert exp.run_noise.additive is False


def test_a_deterministic_run_has_no_run_noise():
    exp = SimulationExperiment.from_string(_SPEC)

    assert exp.run_noise is None
    assert list(exp.noise_sigma_array) == [0.0]


def test_the_tvb_noise_template_reads_an_nsig_recipe():
    from tvbo.templates import lookup

    exp = _noisy("{additive: true, parameters: {nsig: {value: 0.5}}}")
    code = lookup.get_template("tvb/tvbo-tvb-noise.py.mako").render(experiment=exp)

    assert "noise = Additive(nsig=np.array([0.5" in code


def test_the_tvboptim_noise_template_reads_an_nsig_recipe():
    from tvbo.templates import lookup

    exp = _noisy("{additive: true, parameters: {nsig: {value: 0.5}}}")
    code = lookup.get_template("tvboptim/tvbo-tvboptim-noise.py.mako").render(experiment=exp)

    assert "sigma = 1.0" in code
    assert "AdditiveNoise(" in code


# --------------------------------------------------------------------------- ambiguous result lookups raise


def _containers(results, *names):
    results.mkdir(parents=True, exist_ok=True)
    for name in names:
        (results / name).write_bytes(b"")


def test_a_report_refuses_two_runs_of_one_experiment(tmp_path):
    from tvbo.data.dataref import AmbiguousContainerError
    from tvbo.utils import report

    _containers(tmp_path, "exp-3_desc-A_result.h5", "exp-3_model-B_result.h5")

    with pytest.raises(AmbiguousContainerError, match="different runs"):
        report.open_result(tmp_path, "3")
    with pytest.raises(AmbiguousContainerError, match="different runs"):
        report.result_sidecar(tmp_path, "3")


def test_a_report_reads_a_missing_result_as_not_run(tmp_path):
    from tvbo.utils import report

    _containers(tmp_path, "exp-30_desc-A_result.h5")

    assert report.open_result(tmp_path, "3") is None
    assert report.result_sidecar(tmp_path, "3") == {}


def test_a_report_reads_the_sidecar_beside_the_one_container(tmp_path):
    from tvbo.utils import report

    _containers(tmp_path, "exp-3_desc-A_result.h5")
    (tmp_path / "exp-3_desc-A_result.yaml").write_text("id: 3\n")

    assert report.result_sidecar(tmp_path, "3") == {"id": 3}


def test_a_figure_refuses_two_runs_of_one_experiment(tmp_path):
    from tvbo.adapters.bsplot import _container_path
    from tvbo.data.dataref import AmbiguousContainerError
    from tvbo.utils.study_layout import study_path

    _containers(study_path("results", root=tmp_path), "exp-3_desc-A_result.h5", "exp-3_model-B_result.h5")

    with pytest.raises(AmbiguousContainerError, match="different runs"):
        _container_path("exp-3", tmp_path)


def test_a_figure_reads_a_missing_result_as_a_placeholder(tmp_path):
    from tvbo.adapters.bsplot import _container_path
    from tvbo.utils.study_layout import study_path

    results = study_path("results", root=tmp_path)
    _containers(results, "exp-3_desc-A_result.h5", "ana-fig2_result.h5")

    assert _container_path("exp-4", tmp_path) == ""
    assert _container_path("tvbo:ana/Study/fig9", tmp_path) == ""
    assert _container_path("exp-3", tmp_path) == str((results / "exp-3_desc-A_result.h5").resolve())
    assert _container_path("tvbo:result/Study/fig2", tmp_path) == str((results / "ana-fig2_result.h5").resolve())


# --------------------------------------------------------------------------- BEP017 export


def test_an_unlabelled_template_edge_survives_a_bep017_export(tmp_path):
    """The store keys an edge by `matrix_io.edge_name` — name, else label, else `weight` — and the export must look it up the same way."""
    import numpy as np

    from tvbo import Network
    from tvbo.data.converters import to_bep017
    from tvbo.datamodel import tvbo_datamodel

    weights = np.arange(9, dtype="float32").reshape(3, 3)
    net = Network(nodes=[tvbo_datamodel.Node(id=i, label=f"r{i}") for i in range(3)], edges=[], number_of_nodes=3)
    net.edges = [tvbo_datamodel.Edge(weighted=True)]
    object.__setattr__(net, "_arrays", {"edges/weight": weights})
    net._store = None

    to_bep017(net, tmp_path)

    tsv = list(tmp_path.glob("*_meas-weight_relmat.dense.tsv"))
    assert len(tsv) == 1, sorted(p.name for p in tmp_path.iterdir())
    np.testing.assert_allclose(np.loadtxt(tsv[0], delimiter="\t"), weights)


# --------------------------------------------------------------------------- database lookups need a category


@pytest.mark.parametrize("form", ["schema", "pydantic"])
@pytest.mark.parametrize("name", ["Stimulus", "Phenotype"])
def test_a_class_the_database_files_nothing_of_offers_no_lookup(form, name):
    import importlib

    cls = getattr(importlib.import_module(f"tvbo.datamodel.{form}"), name)

    assert not hasattr(cls, "from_db")
    assert not hasattr(cls, "list_db")
    assert all(hasattr(cls, m) for m in ("from_file", "from_string", "to_yaml"))


@pytest.mark.parametrize("form", ["schema", "pydantic"])
def test_a_catalogued_class_keeps_its_lookup(form):
    import importlib

    cls = importlib.import_module(f"tvbo.datamodel.{form}").Dynamics

    assert "JansenRit" in cls.list_db()
