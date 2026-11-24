"""A network can be built anew for each trial of an ensemble: `Network.realisation: trial`.

Koller et al. (2024) draw a new random sheet for every simulation, and a new shuffle of the connectome for every run of their shuffled control, so the spread across their simulations includes that draw. Under `realisation: trial` the graph generator, when it declares a seed, and every callable transform with a `seed` argument, are called again for trial k with `trial_seed(seed, k)`, and on tvboptim the trials map over the realisations the run stacks on the host.

These pin the contract: the realisations differ, repeat under the same declaration and equal the builder at the derived seed; a seeded shuffle gives each trial a permutation of the base network; each trial integrates its own realisation; the history buffer covers the longest delay of any trial; an experiment without the slot renders none of the new code; a produced constant reading the weights is refused by name while one reading the node data is not; outside the ensemble the network is trial 0's realisation, the one a result freezes, whose sidecar records it as trial 0 of n; a read of that frozen network refuses the weights, the tract lengths and the in-strength by name and serves the node data; and a recipe frozen for a kit draws the realisations the recipe does.
"""

import importlib.util
import re
import shutil

import numpy as np
import pytest
import xarray as xr
import yaml

jax = pytest.importorskip("jax")
pytest.importorskip("tvboptim")
import jax.numpy as jnp  # noqa: E402

from tvbo import SimulationExperiment  # noqa: E402

pytestmark = pytest.mark.backend_tvboptim

_BUILDERS = """
import numpy as np

TEMPLATE = np.linspace(-1.0, 1.0, 6)


def _mask(rng, n, p):
    weights = (rng.random((n, n)) < p) * 1.0
    np.fill_diagonal(weights, 0.0)
    return weights


def sheet(n=6, p=0.5, seed=0):
    lengths = np.full((n, n), 4.0)
    np.fill_diagonal(lengths, 0.0)
    return {"weights": _mask(np.random.default_rng(seed), n, p), "lengths": lengths, "node_parameters": {"g": TEMPLATE[:n]}}


def stretched(n=6, p=0.5, seed=0):
    rng = np.random.default_rng(seed)
    weights = _mask(rng, n, p)
    lengths = rng.uniform(1.0, 40.0, (n, n))
    lengths = (lengths + lengths.T) / 2.0
    np.fill_diagonal(lengths, 0.0)
    return {"weights": weights, "lengths": lengths, "node_parameters": {"g": TEMPLATE[:n]}}


def drifting(n=6, p=0.5, seed=0):
    built = sheet(n, p, seed)
    built["node_parameters"] = {"g": np.random.default_rng(seed).random(n)}
    return built


def fixed(n=6):
    weights = np.arange(1.0, n * n + 1.0).reshape(n, n)
    np.fill_diagonal(weights, 0.0)
    lengths = np.full((n, n), 4.0)
    np.fill_diagonal(lengths, 0.0)
    return {"weights": weights, "lengths": lengths, "node_parameters": {"g": TEMPLATE[:n]}}


def shuffle(M, seed=0):
    weights = np.asarray(M, dtype=float).copy()
    edges = weights != 0
    weights[edges] = np.random.default_rng(seed).permutation(weights[edges])
    return weights
"""

_EXPERIMENT = """
id: 7
label: Leak driven over a sheet
dynamics:
  name: Leak
  parameters:
    a: {{value: -0.3}}
    g: {{value: 0.0, shape: "(n_nodes,)"}}
  coupling_inputs:
    c: {{name: c, description: "coupling"}}
  state_variables:
    x: {{equation: {{rhs: "a * x + c + 0.1 * g"}}, initial_value: 1.0, coupling_variable: true}}
network:
  label: Sheet
  number_of_nodes: 6
{realisation}  parameters:
    conduction_speed: {{name: conduction_speed, value: 2.0, unit: mm_per_ms}}
  graph_generator:
    name: Sheet
    type: Sheet
{seed}    builder: {{name: {builder}, module: trial_builders}}
{transforms}  coupling:
    LinearCoupling:
      label: LinearCoupling
      delayed: true
      pre_expression: {{rhs: "x_j"}}
      post_expression: {{rhs: "0.05 * gx"}}
      incoming_states: [x]
      local_states: [x]
integration: {{method: euler, step_size: 0.5, duration: 4.0, transient_time: 0.0, unit: ms}}
execution: {{random_seed: 1, precision: float64}}
observations:
  x_last: {{label: Final x, source: [x], aggregation: last}}
explorations:
  trials:
    n_trials: 3
{space}"""
"""A linear leak on six nodes coupled with a 2 ms delay (4 mm tracts at 2 mm/ms) over a generated sheet, in a three-trial ensemble. The initial state is the same on every node, so the coupling reaches each node through its own in-strength and a trial's weights show in its final state."""

PER_TRIAL = "  realisation: trial\n"
SHUFFLE = "  transforms:\n    - name: weight\n      callable: {name: shuffle, module: trial_builders}\n      arguments:\n        - {name: seed, value: 7}\n"


def _spec(builder="sheet", realisation=PER_TRIAL, seed=42, transforms="", space=""):
    """The recipe on *builder*, its generator seeded with *seed* (None for none), with *realisation*, *transforms* and an exploration axis *space* written in."""
    return _EXPERIMENT.format(
        realisation=realisation,
        seed="" if seed is None else f"    seed: {seed}\n",
        builder=builder,
        transforms=transforms,
        space=space,
    )


@pytest.fixture
def load(tmp_path):
    """Load a recipe as *cls* from a file beside the builder module, as a study's recipe is loaded."""
    (tmp_path / "trial_builders.py").write_text(_BUILDERS)

    def _load(spec: str, cls=SimulationExperiment):
        path = tmp_path / "spec.yaml"
        path.write_text(spec)
        return cls.from_file(str(path))

    return _load


@pytest.fixture
def builders(tmp_path):
    """The builder module the recipes name, imported from its file."""
    path = tmp_path / "trial_builders.py"
    path.write_text(_BUILDERS)
    spec = importlib.util.spec_from_file_location("trial_builders_direct", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_trials_of_a_seeded_builder_get_different_weights(load):
    weights = load(_spec()).network_realisations()["weights"]
    assert weights.shape == (3, 6, 6)
    assert not np.array_equal(weights[0], weights[1]), "two trials integrate one sheet"


def test_declaring_the_network_again_gives_the_same_realisations(load):
    first, second = (load(_spec()).network_realisations() for _ in range(2))
    np.testing.assert_array_equal(first["weights"], second["weights"])


def test_trial_k_is_the_builder_called_with_the_seed_derived_for_k(load, builders):
    from tvbo.classes.network import trial_seed

    experiment = load(_spec())
    realised = experiment.network_realisations()
    for k, weights in enumerate(realised["weights"]):
        np.testing.assert_array_equal(weights, builders.sheet(seed=trial_seed(42, k))["weights"])
    assert realised["lengths"] is None, "the tract lengths every trial shares were stacked"
    assert len({trial_seed(42, k) for k in range(3)} | {42}) == 4
    np.testing.assert_array_equal(experiment.network.matrix("weight", format="dense"), realised["weights"][0])


def test_a_seeded_shuffle_per_trial_permutes_the_base_network(load, builders):
    from tvbo.classes.network import trial_seed

    base = builders.fixed()["weights"]
    realised = load(_spec("fixed", seed=None, transforms=SHUFFLE)).network_realisations()["weights"]
    assert len({lane.tobytes() for lane in realised}) == 3, "two trials integrate one shuffle"
    for k, lane in enumerate(realised):
        np.testing.assert_array_equal(lane != 0, base != 0)
        np.testing.assert_array_equal(np.sort(lane[lane != 0]), np.sort(base[base != 0]))
        np.testing.assert_array_equal(lane, builders.shuffle(base, seed=trial_seed(7, k)))


def test_each_trial_integrates_its_own_realisation(load):
    experiment = load(_spec())
    observed = experiment.run("tvboptim", quiet=True).explorations["trials"].observations["x_last"]
    assert "trial" in observed.dims, f"the ensemble ran no trials: {observed.dims}"
    trials = np.asarray(observed).reshape(3, -1)
    assert np.unique(trials.round(12), axis=0).shape[0] == 3, "two trials integrated one network"
    realised = load(_spec()).network_realisations()
    module = experiment.execute("tvboptim")
    lengths = jnp.asarray(experiment.network.matrix("length", format="dense"))
    for row, weights in zip(trials, realised["weights"], strict=True):
        alone = module.run_simulation(module.create_network(jnp.asarray(weights), distances=lengths)).observations["x_last"]
        np.testing.assert_allclose(row, np.asarray(getattr(alone, "data", alone)).reshape(-1), rtol=1e-12)


def test_the_history_buffer_covers_the_longest_delay_of_any_trial(load):
    experiment = load(_spec("stretched"))
    realised = experiment.network_realisations()
    assert realised["lengths"] is not None
    longest = realised["lengths"].reshape(3, -1).max(axis=1) / 2.0
    assert len(set(longest.tolist())) == 3, "the realisations do not differ in their longest delay"
    module = experiment.execute("tvboptim")
    network = experiment.network
    base = module.create_network(
        jnp.asarray(network.matrix("weight", format="dense")), distances=jnp.asarray(network.matrix("length", format="dense"))
    )
    union = module._union_graph(base.graph, {name: jnp.asarray(stack) for name, stack in realised.items()})
    assert union.max_delay_bound >= longest.max()
    assert union.max_delay_bound > longest.min(), "a buffer sized for one trial would hold every trial's delays"
    np.testing.assert_array_equal(np.asarray(union.weights) != 0, (realised["weights"] != 0).any(axis=0))


def test_an_experiment_without_realisation_renders_none_of_its_code(load):
    absent = load(_spec(realisation="")).render_code("tvboptim")
    for name in ("_NETWORK_REALISATIONS", "_trial_network_lanes", "_union_graph", "_on_trial_network", "network_realisations"):
        assert name not in absent, name


def test_an_experiment_realised_per_trial_writes_each_trials_network(load):
    assert "_on_trial_network(s, _ti)" in load(_spec()).render_code("tvboptim")


@pytest.mark.parametrize("reference", ["network.instrength", "network.edges.weight", "network.edges.length"])
def test_a_produced_constant_reading_what_varies_per_trial_is_refused_by_name(load, reference):
    from tvbo.data import param_io
    from tvbo.datamodel import Callable, FunctionCall, Parameter

    network = load(_spec()).network
    producer = FunctionCall(
        callable=Callable(name="asarray", module="numpy"), arguments={"a": {"name": "a", "value": reference}}
    )
    with pytest.raises(ValueError, match=rf"Parameter 'reference' producer: '{reference}' can differ from trial to trial"):
        param_io.resolve(Parameter(name="reference", producer=producer), context=network)


def test_a_produced_constant_reading_the_node_data_is_served(load):
    from tvbo.data import param_io
    from tvbo.datamodel import Callable, FunctionCall, Parameter

    param_io.clear_cache()
    producer = FunctionCall(
        callable=Callable(name="asarray", module="numpy"), arguments={"a": {"name": "a", "value": "network.nodes.g"}}
    )
    served = param_io.resolve(Parameter(name="template", producer=producer), context=load(_spec()).network)
    np.testing.assert_array_equal(served, np.linspace(-1.0, 1.0, 6))


def test_a_realisation_that_changes_the_node_data_is_refused(load):
    with pytest.raises(ValueError, match=r"realisation of trial 1 changes the node data of 'node_\d'"):
        load(_spec("drifting")).network_realisations()


def test_a_network_with_nothing_seeded_is_refused(load):
    with pytest.raises(ValueError, match="no seeded part to build again for each trial"):
        load(_spec("fixed", seed=None)).network_realisations()


def test_a_weight_axis_beside_a_network_built_per_trial_is_refused(load):
    space = "    space:\n      network.weight: {explored_values: [0.5, 1.0]}\n"
    with pytest.raises(ValueError, match="on a network built anew for each trial"):
        load(_spec(space=space)).render_code("tvboptim")


def test_a_backend_that_builds_one_network_per_run_refuses_it(load, monkeypatch):
    from tvbo.export import registry

    probe = registry.ExportFormat(
        "probe-format",
        "Probe",
        ".txt",
        "text/plain",
        lambda experiment, **kwargs: "rendered",
        parameter_starts=(registry.StartKind.NODE,),
    )
    monkeypatch.setitem(registry._REGISTRY, "probe-format", probe)
    assert registry.render(load(_spec(realisation="")), "probe-format") == "rendered"
    with pytest.raises(NotImplementedError, match="the probe-format backend builds one network per run"):
        registry.render(load(_spec()), "probe-format")


def test_the_network_a_result_freezes_is_trial_0_and_its_sidecar_says_so(load, builders, tmp_path):
    """Written as a result's provenance writes it (``freeze_yaml`` under ``get_network_stem``) and read as a reference to the run's network reads it (``network_companion_path``, ``load_network``)."""
    from tvbo.classes.network import realised_per_trial, trial_seed
    from tvbo.data.dataref import network_companion_path
    from tvbo.data.network_io import load_network

    experiment = load(_spec())
    experiment.network_realisations()
    out = tmp_path / "derivatives"
    recipe = yaml.safe_load(experiment.freeze_yaml(out, network_stem=experiment.get_network_stem()))
    assert recipe["network"]["realisation"] == "trial"
    assert "realised_trial" not in recipe["network"] and experiment.network.realised_trial is None
    frozen = load_network(network_companion_path(out / f"{experiment.get_result_stem()}.h5"))
    assert realised_per_trial(frozen)
    assert (frozen.realised_trial.trial, frozen.realised_trial.n_trials) == (0, 3)
    np.testing.assert_array_equal(frozen.matrix("weight", format="dense"), builders.sheet(seed=trial_seed(42, 0))["weights"])
    assert all(frozen.varies_per_trial(name) for name in ("weight", "edges.length", "instrength"))
    assert not any(frozen.varies_per_trial(name) for name in ("positions", "labels", "nodes.g"))


def test_a_read_of_the_network_a_run_froze_refuses_what_varies_per_trial_by_name(load, tmp_path):
    """Read as a figure layer or an analysis argument reads ``network.<name>`` of a run (``output_array`` on its container), from the companion a result's provenance writes (``freeze_yaml`` under ``get_network_stem``)."""
    from tvbo.data.dataref import output_array

    experiment = load(_spec())
    out = tmp_path / "derivatives"
    experiment.freeze_yaml(out, network_stem=experiment.get_network_stem())
    container = out / f"{experiment.get_result_stem()}.h5"
    for name in ("weight", "edges.length", "instrength"):
        said = re.escape(f"'network.{name}' can differ from trial to trial") + ".*" + re.escape("Select `sel: {trial: 0}`")
        with pytest.raises(ValueError, match=said):
            output_array(None, container, f"network.{name}")
    served = output_array(None, container, "network.nodes.g")
    np.testing.assert_array_equal(np.asarray(served).reshape(-1), np.linspace(-1.0, 1.0, 6))


def _frozen_run(load, results):
    """Experiment 7's result as a run writes it under *results*: a container, and the network its provenance freezes beside it (``freeze_yaml``); returns the container."""
    results.mkdir(parents=True)
    container = results / "exp-7_model-Leak_result.h5"
    xr.Dataset({"x_last": ("node", np.zeros(6))}).to_netcdf(container, engine="h5netcdf")
    load(_spec()).freeze_yaml(results, network_stem="exp-7_model-Leak_network")
    return container


def _weight_at_trial(trial):
    """A reference to experiment 7's ``network.weight`` that selects *trial*, as an analysis argument declares one."""
    from tvbo.datamodel import DataRef

    return DataRef(experiment=7, output="network.weight", sel={"trial": {"name": "trial", "value": trial}})


def test_a_reference_selecting_the_trial_the_frozen_network_holds_reads_its_weights(load, tmp_path):
    from tvbo.data.dataref import network_companion_path, resolve_dataref
    from tvbo.data.network_io import load_network

    container = _frozen_run(load, tmp_path / "derivatives")
    served = resolve_dataref(_weight_at_trial(0), results_root=tmp_path / "derivatives")
    assert "trial" not in served.dims, served.dims
    frozen = load_network(network_companion_path(container))
    np.testing.assert_array_equal(np.asarray(served), frozen.matrix("weight", format="dense"))


def test_a_reference_selecting_a_trial_the_frozen_network_does_not_hold_is_refused_by_name(load, tmp_path):
    from tvbo.data.dataref import resolve_dataref

    _frozen_run(load, tmp_path / "derivatives")
    said = re.escape("`sel: {trial: 1}` selects a realisation of 'network.weight'") + ".*holds trial 0 alone"
    with pytest.raises(ValueError, match=said):
        resolve_dataref(_weight_at_trial(1), results_root=tmp_path / "derivatives")


def test_a_recipe_frozen_for_a_kit_draws_the_realisations_the_recipe_does(load, tmp_path):
    """A kit runs from the frozen recipe, whose network is trial 0's realisation in a companion beside it, with the builder module copied in as a kit bundles it."""
    experiment = load(_spec())
    kit = tmp_path / "kit"
    frozen_text = experiment.freeze_yaml(kit, network_stem="network")
    assert yaml.safe_load(frozen_text)["network"]["graph_generator"]["builder"]["module"] == "trial_builders"
    (kit / "spec.yaml").write_text(frozen_text)
    shutil.copy(tmp_path / "trial_builders.py", kit / "trial_builders.py")
    rerun = SimulationExperiment.from_file(str(kit / "spec.yaml"))
    np.testing.assert_array_equal(
        rerun.network.matrix("weight", format="dense"), experiment.network.matrix("weight", format="dense")
    )
    np.testing.assert_array_equal(rerun.network_realisations()["weights"], experiment.network_realisations()["weights"])
    np.testing.assert_array_equal(
        rerun.network.matrix("weight", format="dense"),
        experiment.network.matrix("weight", format="dense"),
        err_msg="the kit's network is not trial 0's realisation again",
    )
