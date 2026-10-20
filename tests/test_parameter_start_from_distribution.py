"""A parameter that declares where its value starts is given that start when the network is built, keyed to the run's seed.

A `Parameter.distribution` sampled over space is drawn once per run: one value per node, or for a coupling parameter of the graph's shape one per connection, on the network's edges in row order with every other entry at the parameter's `value`. `Distribution.seed` overrides `execution.random_seed`, which keys every draw that declares none, so a run is reproducible, another seed is another start, and each cell of an `execution.random_seed` axis starts where a run declared at that seed starts. A `Parameter.equation` computes a parameter once from the other parameters of its owner, which is how two parameters share one draw (`p = 1 + u`, `q = 1 - u`). A tuning algorithm starts from these values and then updates them, one subject at a time or as one on-device cohort. The trials of an `n_trials` ensemble share their cell's draw unless the distribution declares `axis: trial`, which gives each trial its own, under the run's key with the trial index folded in; outside an ensemble that is the run's draw.

What cannot be honoured is refused by name when the code is rendered: on tvboptim a parameter with a second definition of its value or a distribution that does not say what to draw, on every backend that draws no start the declaration itself, and on every backend that draws once per run a start drawn per trial.
"""

import json
import os
import subprocess
import sys
import zlib

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("tvboptim")
xr = pytest.importorskip("xarray")
import jax.numpy as jnp  # noqa: E402

from tvbo import SimulationExperiment  # noqa: E402
from tvbo.run import study as study_run  # noqa: E402

pytestmark = pytest.mark.backend_tvboptim

NODE = """
id: 1
label: Decay with one rate per node
dynamics:
  name: Decay
  parameters:
    k: {k}
  state_variables:
    x: {{equation: {{rhs: "-k*x"}}, initial_value: 1.0}}
network:
  label: Five
  number_of_nodes: 5
integration: {{method: euler, step_size: 0.5, duration: 2.0, unit: ms}}
execution: {{random_seed: {seed}}}
observations:
  x_last: {{label: Final x, source: [x], aggregation: last}}
"""

EDGE = """
id: 2
label: Leak driven through one weight per connection
dynamics:
  name: Leak
  parameters:
    a: {{value: -0.3}}
  coupling_inputs:
    c: {{name: c, description: "coupling"}}
  state_variables:
    x: {{equation: {{rhs: "a * x + c"}}, initial_value: 1.0, coupling_variable: true}}
network:
  number_of_nodes: 3
  nodes: [{{id: 0, label: r0}}, {{id: 1, label: r1}}, {{id: 2, label: r2}}]
  edges:
    - {{source: 0, target: 1, weight: 1.0}}
    - {{source: 1, target: 0, weight: 1.0}}
    - {{source: 0, target: 2, weight: 1.0}}
    - {{source: 2, target: 0, weight: 1.0}}
    - {{source: 1, target: 2, weight: 1.0}}
    - {{source: 2, target: 1, weight: 1.0}}
  coupling:
    LinearCoupling:
      label: LinearCoupling
      parameters:
{parameters}
      pre_expression: {{rhs: "{pre}"}}
      post_expression: {{rhs: "0.05 * gx"}}
      incoming_states: [x]
      local_states: [x]
integration: {{method: euler, step_size: 0.5, duration: 4.0, unit: ms}}
execution: {{random_seed: {seed}}}
observations:
  x_last: {{label: Final x, source: [x], aggregation: last}}
"""

CONSTANT_K = "{value: 0.5}"
SAMPLED_K = "{value: 0.5, heterogeneous: true, distribution: {name: Uniform, domain: {lo: 0.0, hi: 2.0}}}"
MATRIX = 'heterogeneous: true, shape: "(n_nodes, n_nodes)"'
CONSTANT_W = f"        w: {{value: 1.0, {MATRIX}}}"
SAMPLED_W = f"        w: {{value: 1.0, {MATRIX}, distribution: {{name: Uniform, domain: {{lo: 0.0, hi: 2.0}}}}}}"
TIED = f'        u: {{value: 0.0, {MATRIX}, distribution: {{name: Uniform, domain: {{lo: -1.0, hi: 1.0}}}}}}\n        p: {{equation: {{rhs: "1 + u"}}, {MATRIX}}}\n        q: {{equation: {{rhs: "1 - u"}}, {MATRIX}}}'
PAIR_OF_ONES = f"        p: {{value: 1.0, {MATRIX}}}\n        q: {{value: 1.0, {MATRIX}}}"
PER_TRIAL_K = SAMPLED_K.replace("hi: 2.0}}", "hi: 2.0}, axis: trial}")
PER_TRIAL_W = SAMPLED_W.replace("hi: 2.0}}", "hi: 2.0}, axis: trial}")


def _final(tmp_path, spec, tag):
    """Run *spec* on tvboptim and return the final state of every node."""
    out = tmp_path / tag
    out.mkdir()
    recipe = out / "recipe.yaml"
    recipe.write_text(spec, encoding="utf-8")
    study_run.run_experiment(
        SimulationExperiment.from_string(spec), str(recipe), out / "results", options=study_run.RunOptions(backend="tvboptim")
    )
    (container,) = sorted((out / "results").glob("*_result.h5"))
    with xr.open_dataset(container, engine="h5netcdf") as ds:
        return np.asarray(ds["observation__x_last"].values).squeeze()


def test_a_per_node_parameter_starts_from_its_declared_distribution(tmp_path):
    """Five nodes with a rate drawn from Uniform(0, 2) do not all end where a rate of 0.5 ends."""
    sampled = _final(tmp_path, NODE.format(k=SAMPLED_K, seed=1), "sampled")
    constant = _final(tmp_path, NODE.format(k=CONSTANT_K, seed=1), "constant")
    assert np.unique(sampled.round(6)).size > 1, (
        f"every node ran at the same rate: {sampled}, the declared distribution was not sampled (a constant rate gives {constant})"
    )
    assert np.all((sampled >= 0.0) & (sampled <= 1.0)), (
        "a rate inside the declared support [0, 2] keeps the final state inside [0, 1]"
    )


def test_the_draw_follows_the_run_seed(tmp_path):
    """The same seed gives the same start, another seed another one."""
    a = _final(tmp_path, NODE.format(k=SAMPLED_K, seed=1), "a")
    again = _final(tmp_path, NODE.format(k=SAMPLED_K, seed=1), "again")
    b = _final(tmp_path, NODE.format(k=SAMPLED_K, seed=2), "b")
    np.testing.assert_allclose(a, again, rtol=0, atol=0, err_msg="one seed, two starts")
    assert not np.allclose(a, b), f"seeds 1 and 2 ran from the same parameter values: {a}"


def test_a_per_connection_parameter_starts_from_its_declared_distribution(tmp_path):
    """A weight per connection drawn from Uniform(0, 2) drives the nodes differently from a weight of 1 everywhere, and differently per seed."""
    constant = _final(tmp_path, EDGE.format(parameters=CONSTANT_W, pre="x_j * w", seed=1), "constant")
    s1 = _final(tmp_path, EDGE.format(parameters=SAMPLED_W, pre="x_j * w", seed=1), "s1")
    s2 = _final(tmp_path, EDGE.format(parameters=SAMPLED_W, pre="x_j * w", seed=2), "s2")
    assert not np.allclose(s1, constant), f"the per-connection distribution was not sampled: {s1} equals the run at 1.0"
    assert not np.allclose(s1, s2), "seeds 1 and 2 ran from the same weights"


def test_two_parameters_start_from_one_draw(tmp_path):
    """With p = 1 + u and q = 1 - u from one draw u per connection, p + q is 2 on every connection while p alone is not 1."""
    both_ones = _final(tmp_path, EDGE.format(parameters=PAIR_OF_ONES, pre="x_j * (p + q)", seed=1), "ones")
    tied_sum = _final(tmp_path, EDGE.format(parameters=TIED, pre="x_j * (p + q)", seed=1), "sum")
    tied_p = _final(tmp_path, EDGE.format(parameters=TIED, pre="x_j * p * 2", seed=1), "p")
    np.testing.assert_allclose(
        tied_sum, both_ones, rtol=1e-6, err_msg="p + q is not 2 on every connection: the two starts do not share one draw"
    )
    assert not np.allclose(tied_p, both_ones), (
        "p is 1 on every connection: the draw did not reach the parameter defined from it"
    )


def _uniform(lo, hi, extra=""):
    return f"distribution: {{name: Uniform, domain: {{lo: {lo}, hi: {hi}}}{extra}}}"


def _node(k, seed=1, nodes=5, more=""):
    """The node recipe with rate *k* on *nodes* nodes, and *more* parameters beside it."""
    spec = NODE.format(k=k, seed=seed).replace("number_of_nodes: 5", f"number_of_nodes: {nodes}")
    return spec.replace("  state_variables:", f"{more}  state_variables:", 1)


def _edge(parameters, pre="x_j * w", seed=1, pairs=None):
    """The edge recipe, on the undirected connections *pairs* instead of all three where given."""
    spec = EDGE.format(parameters=parameters, pre=pre, seed=seed)
    if pairs is None:
        return spec
    kept = [line for line in spec.splitlines() if not line.startswith("    - {source:")]
    at = kept.index("  edges:") + 1
    kept[at:at] = [f"    - {{source: {i}, target: {j}, weight: 1.0}}" for i, j in pairs]
    return "\n".join(kept) + "\n"


def _built(spec):
    """The generated module of *spec*, the network it builds on the recipe's own weights, and those weights."""
    experiment = SimulationExperiment.from_string(spec)
    module = experiment.execute("tvboptim")
    weights = np.asarray(experiment.network.matrix("weight", format="dense", apply_transforms=False), dtype=float)
    return module, module.create_network(jnp.asarray(weights)), weights


def _rate(spec, name="k"):
    return np.asarray(_built(spec)[1].params.dynamics[name])


def _weight(spec, name="w"):
    return np.asarray(_built(spec)[1].params.coupling["c"][name])


def test_a_uniform_draw_fills_its_declared_support():
    """Two thousand rates from Uniform(0, 2) stay inside [0, 2) around 1 with the spread of a uniform of that width."""
    k = _rate(_node(SAMPLED_K, nodes=2000))
    assert k.shape == (2000,) and k.min() >= 0.0 and k.max() < 2.0
    assert abs(k.mean() - 1.0) < 0.06 and abs(k.std() - 2.0 / np.sqrt(12.0)) < 0.03
    assert k.min() < 0.05 and k.max() > 1.95, "the draw does not reach both ends of the support"


@pytest.mark.parametrize("name", ["Normal", "Gaussian"])
def test_a_normal_draw_has_its_declared_mean_and_spread(name):
    k = _rate(
        _node(
            f"{{value: 0.5, distribution: {{name: {name}, parameters: {{mean: {{value: 3.0}}, std: {{value: 0.5}}}}}}}}",
            nodes=2000,
        )
    )
    assert abs(k.mean() - 3.0) < 0.05 and abs(k.std() - 0.5) < 0.03
    assert k.min() < 2.0 and k.max() > 4.0, "a normal draw of this size reaches two spreads from its mean"


def test_a_truncated_normal_draw_stays_inside_its_domain():
    declared = "{value: 0.5, distribution: {name: TruncatedNormal, domain: {lo: 2.75, hi: 3.5}, parameters: {mean: {value: 3.0}, std: {value: 0.5}}}}"
    k = _rate(_node(declared, nodes=2000))
    assert k.min() >= 2.75 and k.max() <= 3.5
    assert 3.0 < k.mean() < 3.2, "the mean moves toward the wider side of the truncation"


def test_a_per_connection_draw_fills_the_edges_in_row_order_and_leaves_the_rest_at_value():
    """Two networks with as many connections in other places hold the same drawn values, one per edge in (target, source) row order, and the declared value everywhere else."""
    star, chain = [(0, 1), (0, 2)], [(0, 1), (1, 2)]
    drawn = []
    for pairs in (star, chain):
        _, network, weights = _built(_edge(SAMPLED_W, pairs=pairs))
        w = np.asarray(network.params.coupling["c"]["w"])
        assert weights.astype(bool).sum() == 4, "each undirected connection is two edges"
        np.testing.assert_array_equal(w[weights == 0], 1.0)
        on_edges = w[weights != 0]
        assert np.unique(on_edges).size == 4 and on_edges.min() >= 0.0 and on_edges.max() < 2.0
        drawn.append(on_edges)
    np.testing.assert_array_equal(drawn[0], drawn[1])


def test_a_distributions_own_seed_overrides_the_run_seed():
    own = "{value: 0.5, " + _uniform(0.0, 2.0, ", seed: 11") + "}"
    np.testing.assert_array_equal(_rate(_node(own, seed=1)), _rate(_node(own, seed=2)))
    assert not np.allclose(_rate(_node(own, seed=1)), _rate(_node(SAMPLED_K, seed=1)))
    assert not np.allclose(_rate(_node(own, seed=1)), _rate(_node(own.replace("seed: 11", "seed: 12"), seed=1)))


def test_parameters_under_one_seed_draw_independently_and_each_keeps_its_draw():
    """A second sampled parameter draws its own values and does not move the first one's."""
    more = f"    m: {{value: 0.0, {_uniform(0.0, 2.0)}}}\n"
    _, network, _ = _built(_node(SAMPLED_K, more=more).replace('"-k*x"', '"-k*x + 0*m"'))
    k, m = np.asarray(network.params.dynamics["k"]), np.asarray(network.params.dynamics["m"])
    assert np.unique(k).size == 5 and np.unique(m).size == 5 and not np.allclose(k, m)
    np.testing.assert_array_equal(k, _rate(_node(SAMPLED_K)))


def test_a_seed_passed_at_run_time_keys_the_draw_as_the_declared_one_does():
    module, _, weights = _built(_node(SAMPLED_K, seed=1))
    passed = np.asarray(module.create_network(jnp.asarray(weights), random_seed=2).params.dynamics["k"])
    np.testing.assert_array_equal(passed, _rate(_node(SAMPLED_K, seed=2)))


SEED_AXIS = """
explorations:
  seeds:
    space:
      execution.random_seed: {{explored_values: {seeds}}}
"""


def test_every_cell_of_a_seed_axis_starts_where_a_run_at_that_seed_starts():
    """Without noise the drawn parameter is what reads the seed: three cells, three starts, each the one a run declared at that seed integrates from."""
    swept = SimulationExperiment.from_string(_node(SAMPLED_K, seed=7) + SEED_AXIS.format(seeds=[1, 2, 3])).run(
        "tvboptim", quiet=True
    )
    cells = swept.explorations["seeds"].observations["x_last"]
    assert cells.dims[0] == "execution.random_seed"
    cells = np.asarray(cells).reshape(3, 5)
    assert np.unique(cells.round(9), axis=0).shape[0] == 3
    for row, seed in zip(cells, (1, 2, 3), strict=True):
        alone = SimulationExperiment.from_string(_node(SAMPLED_K, seed=seed)).run("tvboptim", quiet=True)
        np.testing.assert_array_equal(row, np.asarray(alone.integration.observations["x_last"].data).reshape(5))


def test_a_seed_axis_with_only_self_seeded_draws_still_has_no_consumer():
    own = "{value: 0.5, " + _uniform(0.0, 2.0, ", seed: 11") + "}"
    with pytest.raises(ValueError, match="has no consumer"):
        SimulationExperiment.from_string(_node(own) + SEED_AXIS.format(seeds=[1, 2])).render_code("tvboptim")


def _start_evaluations(seeds):
    """How often the generated module evaluates the dynamics' starts while it runs a seed axis over *seeds*."""
    module, network, _ = _built(_node(SAMPLED_K) + SEED_AXIS.format(seeds=seeds))
    base = module.run_simulation(network, run_main=False)
    scope, made = module.create_network.__globals__, []
    drawn = scope["_dynamics_parameter_starts"]
    scope["_dynamics_parameter_starts"] = lambda *args: (made.append(args[0]), drawn(*args))[1]
    cells = module.seeds(base.state, base.model_fn, network=network).observations["x_last"]
    assert np.asarray(cells).shape[0] == len(seeds)
    return len(made)


def test_a_seed_axis_draws_inside_the_one_compiled_cell():
    """The draw is traced with the cell, so eight seeds evaluate the generated start function as often as two do."""
    assert _start_evaluations([1, 2]) == _start_evaluations(list(range(8))) == 1


TUNED = """
id: 3
label: Relaxation to a tuned level
dynamics:
  name: Relax
  parameters:
    g: {g}
    tau: {{value: 2.0}}
  state_variables:
    x: {{equation: {{rhs: "(g - x) / tau"}}, initial_value: 0.0}}
  output: [x]
network:
  number_of_nodes: 3
  nodes: [{{id: 0, label: r0}}, {{id: 1, label: r1}}, {{id: 2, label: r2}}]
  edges: []
integration: {{method: euler, step_size: 0.5, duration: 20.0, unit: ms}}
observations:
  level: {{label: Mean level, source: [x], aggregation: mean}}
algorithms:
  tune:
    objective: {{type: activity_target, target_variable: x, target_value: 0.5}}
    observations: [level]
    update_rules:
      - name: g_update
        target_parameter: {{name: g}}
        equation: {{rhs: "g - eta * (level - 0.5)"}}
    hyperparameters:
      - {{name: eta, value: {eta}}}
    n_iterations: 4
    simulation_period: 4.0
execution: {{random_seed: {seed}, precision: float64}}
"""
TUNED_SHAPE = 'heterogeneous: true, free: true, shape: "(n_nodes,)"'
CONSTANT_G = f"{{value: 0.2, {TUNED_SHAPE}}}"
SAMPLED_G = f"{{value: 0.2, {TUNED_SHAPE}, {_uniform(0.0, 1.0)}}}"
WIRED = """
explorations:
  seeds:
    algorithms: [tune]
    space:
      execution.random_seed: {explored_values: [1, 2]}
"""


def _tuned(g, eta, seed=1):
    result = SimulationExperiment.from_string(TUNED.format(g=g, eta=eta, seed=seed)).run("tvboptim", quiet=True)
    return np.asarray(result.algorithms["tune"].state.dynamics.g)


def test_a_tuned_parameter_starts_from_its_draw_and_is_then_updated():
    """A rule that changes nothing leaves the draw, and a rule that pulls every node toward one level moves each node from its own drawn value."""
    draw = _rate(TUNED.format(g=SAMPLED_G, eta=0.0, seed=1), "g")
    assert np.unique(draw.round(9)).size == 3 and draw.min() >= 0.0 and draw.max() < 1.0
    np.testing.assert_array_equal(_tuned(SAMPLED_G, 0.0), draw)
    moved, from_constant = _tuned(SAMPLED_G, 0.2), _tuned(CONSTANT_G, 0.2)
    assert not np.allclose(moved, draw) and not np.allclose(moved, from_constant)
    assert np.all(np.abs(moved - 0.5) < np.abs(draw - 0.5)), "four updates move every node toward the level the rule tunes for"
    assert np.unique(from_constant.round(9)).size == 1, "a constant start is tuned to one value on every node"


def test_a_tuning_wired_into_a_seed_axis_starts_each_cell_from_its_own_draw():
    """Each cell tunes from the start a run at its seed draws, so its tuned values are that run's."""
    swept = SimulationExperiment.from_string(TUNED.format(g=SAMPLED_G, eta=0.2, seed=9) + WIRED).run("tvboptim", quiet=True)
    cells = np.asarray(swept.explorations["seeds"].observations["algorithm_tune_g"]).reshape(2, 3)
    for row, seed in zip(cells, (1, 2), strict=True):
        np.testing.assert_allclose(row, _tuned(SAMPLED_G, 0.2, seed=seed), rtol=0, atol=1e-12)
    assert not np.allclose(cells[0], cells[1])


COHORT = """
id: 4
label: One weight per connection moved by each subject's target
dynamics:
  name: Leak
  parameters:
    a: {{value: -0.3}}
  coupling_inputs:
    c: {{name: c, description: "coupling"}}
  state_variables:
    x: {{equation: {{rhs: "a * x + c"}}, initial_value: 1.0, coupling_variable: true}}
  output: [x]
network:
  number_of_nodes: 3
  nodes: [{{id: 0, label: r0}}, {{id: 1, label: r1}}, {{id: 2, label: r2}}]
  edges:
    - {{source: 0, target: 1, weight: 1.0}}
    - {{source: 0, target: 2, weight: 1.0}}
  coupling:
    c:
      label: c
      parameters:
        w: {{value: 1.0, {matrix}, free: true, distribution: {{name: Uniform, domain: {{lo: 0.0, hi: 2.0}}}}}}
      pre_expression: {{rhs: "x_j * w"}}
      post_expression: {{rhs: "0.05 * gx"}}
      incoming_states: [x]
      local_states: [x]
integration: {{method: euler, step_size: 0.5, duration: 4.0, unit: ms}}
dataset:
  dataset_id: toy_fc
  bids_root: {root}
  batch_mode: {mode}
  subjects: [{{subject_id: "01"}}, {{subject_id: "02"}}, {{subject_id: "03"}}]
observations:
  level: {{label: Mean level, source: [x], aggregation: mean}}
  fc_target: {{source: [dataset.subject.fc], query: {{atlas: Toy, desc: FC, suffix: relmat}}, reconcile: by_label}}
algorithms:
  tune:
    objective: {{type: activity_target, target_variable: x, target_value: 0.5}}
    observations: [level, fc_target]
    update_rules:
      - name: w_update
        target_parameter: {{name: w}}
        equation: {{rhs: "w + eta * fc_target"}}
    hyperparameters:
      - {{name: eta, value: 0.5}}
    n_iterations: 3
    simulation_period: 2.0
execution: {{random_seed: 5, precision: float64}}
"""


def test_an_on_device_cohort_starts_every_subject_from_the_draw_a_run_of_one_subject_starts_from(tmp_path):
    """Three subjects fitted as one batch each end at the run's draw plus their own updates, which is where the same subject ends when fitted alone."""
    from tvbo.classes.network import Network

    for subject in ("01", "02", "03"):
        target = Network.from_matrix(weights=np.eye(3), labels=["r0", "r1", "r2"])
        target.set_matrix("fc", np.full((3, 3), int(subject) / 10.0))
        folder = tmp_path / f"sub-{subject}" / "func"
        folder.mkdir(parents=True)
        target.save(folder / f"sub-{subject}_atlas-Toy_desc-FC_relmat.yaml")

    batch = SimulationExperiment.from_string(COHORT.format(matrix=MATRIX, root=tmp_path, mode="on_device"))
    assert batch.dataset_on_device()
    _, network, weights = _built(COHORT.format(matrix=MATRIX, root=tmp_path, mode="on_device"))
    draw = np.asarray(network.params.coupling["c"]["w"])
    assert np.unique(draw[weights != 0]).size == 4 and np.all(draw[weights == 0] == 1.0)

    fitted = batch.run("tvboptim", quiet=True).algorithms["tune"]
    together = np.asarray(fitted.cohort_state.coupling["c"]["w"])
    assert list(fitted.subject_ids) == ["01", "02", "03"] and together.shape == (3, 3, 3)
    for lane, subject in zip(together, (1, 2, 3), strict=True):
        np.testing.assert_allclose(lane, draw + 3 * 0.5 * subject / 10.0, rtol=0, atol=1e-12)

    alone = SimulationExperiment.from_string(COHORT.format(matrix=MATRIX, root=tmp_path, mode="fan_out"))
    one = alone.run("tvboptim", quiet=True, active_subject="02").algorithms["tune"]
    np.testing.assert_allclose(np.asarray(one.state.coupling["c"]["w"]), together[1], rtol=0, atol=1e-12)


def test_an_equation_gives_a_parameter_its_value():
    """A parameter declared by an equation alone holds what the equation says, from literals as from a draw."""
    scaled = f'        s: {{value: 0.75}}\n        w: {{equation: {{rhs: "2 * s"}}, {MATRIX}}}'
    np.testing.assert_array_equal(_weight(_edge(scaled)), 1.5)
    _, network, weights = _built(_edge(TIED, pre="x_j * (p + q)"))
    u, p, q = (np.asarray(network.params.coupling["c"][name]) for name in "upq")
    np.testing.assert_array_equal(p, 1 + u)
    np.testing.assert_array_equal(q, 1 - u)
    assert np.unique(u[weights != 0]).size == 6 and np.all(u[weights == 0] == 0.0)


def test_a_recipe_without_a_parameter_start_renders_none_of_its_code():
    for spec in (_node(CONSTANT_K), _edge(CONSTANT_W), TUNED.format(g=CONSTANT_G, eta=0.2, seed=1)):
        code = SimulationExperiment.from_string(spec).render_code("tvboptim")
        assert "_parameter_starts" not in code and "_start_seed" not in code and "random_seed: int = None" not in code


AXIS = """
explorations:
  sweep:
    space:
      {axis}: {{explored_values: [0.5, 1.5]}}
"""
NODES_WITH_K = """  number_of_nodes: 5
  nodes:
    - {id: 0, label: r0, parameters: {k: {value: 0.1}}}
    - {id: 1, label: r1}
    - {id: 2, label: r2}
    - {id: 3, label: r3}
    - {id: 4, label: r4}
"""
SCALED = f'        s: {{value: 0.75}}\n        w: {{equation: {{rhs: "2 * s"}}, {MATRIX}}}'
OPTIMIZATION = """
optimizations:
  fit:
    loss: {function: mse, arguments: {x: {value: observations.x_last.data}}}
    free_parameters:
      - {parameter: Decay.k, heterogeneous: true, shape: "(n_nodes,)", initial_value: 0.3}
    algorithm: adam
    learning_rate: 0.01
    max_iterations: 2
"""

REFUSED = {
    "an axis on a drawn dynamics parameter": (
        _node(SAMPLED_K) + AXIS.format(axis="Decay.k"),
        "'k' also declares a distribution",
    ),
    "an axis on a drawn coupling parameter": (
        _edge(SAMPLED_W) + AXIS.format(axis="LinearCoupling.w"),
        "'w' also declares a distribution",
    ),
    "an axis on a parameter with an equation": (
        _edge(SCALED) + AXIS.format(axis="LinearCoupling.w"),
        "'w' also declares an equation",
    ),
    "an axis on what an equation reads": (_edge(SCALED) + AXIS.format(axis="LinearCoupling.s"), "'w' is computed from 's'"),
    "a value beside an equation": (
        _edge(SCALED.replace("w: {equation", "w: {value: 1.0, equation")),
        "'w'.*also declares a `value`.*`value: null` clears a value inherited through a merge key",
    ),
    "a distribution beside an equation": (
        _edge(SCALED.replace("w: {equation", f"w: {{{_uniform(0, 2)}, equation")),
        "'w'.*also declares a `distribution`",
    ),
    "per-node values on the nodes": (
        _node(SAMPLED_K).replace("  number_of_nodes: 5\n", NODES_WITH_K),
        "'k'.*per-node values on the network's nodes",
    ),
    "an equation reading an undeclared name": (
        _edge(SCALED.replace("2 * s", "2 * z")),
        "'w'.*reads z, which coupling 'LinearCoupling' does not declare",
    ),
    "an equation reading a parameter without a value": (
        _edge(SCALED.replace("s: {value: 0.75}", "s: {free: true}")),
        "'w'.*reads 's', which has no literal `value`",
    ),
    "equations reading each other": (
        _edge(
            f'        p: {{equation: {{rhs: "q + 1"}}, {MATRIX}}}\n        q: {{equation: {{rhs: "p - 1"}}, {MATRIX}}}',
            pre="x_j * p * q",
        ),
        "reads itself through p -> q -> p",
    ),
    "an equation narrower than what it reads": (
        _edge(
            TIED.replace(f'p: {{equation: {{rhs: "1 + u"}}, {MATRIX}}}', 'p: {equation: {rhs: "1 + u"}}'), pre="x_j * (p + q)"
        ),
        "'p'.*reads 'u', which holds one value per connection",
    ),
    "a distribution tvboptim has no draw for": (
        _node("{value: 0.5, distribution: {name: Cauchy, domain: {lo: 0.0, hi: 2.0}}}"),
        "'k'.*no draw is implemented for the distribution 'Cauchy'",
    ),
    "a uniform without bounds": (_node("{value: 0.5, distribution: {name: Uniform}}"), "'k'.*declares no lo, hi"),
    "a normal without moments": (
        _node("{value: 0.5, distribution: {name: Normal, domain: {lo: 0.0, hi: 2.0}}}"),
        "'k'.*declares no mean, std",
    ),
    "bounds in the wrong order": (_node("{value: 0.5, " + _uniform(2.0, 0.0) + "}"), "'k'.*needs lo < hi"),
    "a coupling draw without a shape": (
        _edge("        w: {value: 1.0, " + _uniform(0.0, 2.0) + "}"),
        "'w'.*declares shape None",
    ),
    "a per-connection draw on a dynamics parameter": (
        _node('{value: 0.5, shape: "(n_nodes, n_nodes)", ' + _uniform(0.0, 2.0) + "}"),
        "'k'.*one per connection",
    ),
    "a draw of a shape the start cannot size": (
        _node('{value: 0.5, shape: "(n_modes,)", ' + _uniform(0.0, 2.0) + "}"),
        r"'k'.*declares shape '\(n_modes,\)'",
    ),
    "an equation of a shape the start cannot size": (
        _node(CONSTANT_K, more='    m: {equation: {rhs: "2 * k"}, shape: "(3,)"}\n').replace('"-k*x"', '"-k*x + 0*m"'),
        r"'m'.*declares shape '\(3,\)'",
    ),
    "a per-connection draw without a value": (
        _edge(f"        w: {{{MATRIX}, {_uniform(0.0, 2.0)}}}"),
        "'w'.*keep the parameter's `value`, which it does not declare",
    ),
    "a weight axis under a per-connection draw": (
        _edge(SAMPLED_W) + AXIS.format(axis="network.edges.weight"),
        "draws one value per connection",
    ),
    "a used reference beside a draw": (
        _node('{value: 0.5, used: {iri: "tvbo:network/none", output: k}, ' + _uniform(0.0, 2.0) + "}"),
        "'k'.*also declares a `used` reference",
    ),
    "a measure read from another experiment beside a draw": (
        "initial_state: {method: from_experiment, source_experiment: 99}\n"
        + _node("{value: 0.5, measure: k, " + _uniform(0.0, 2.0) + "}"),
        "'k'.*also declares a `measure` read from the experiment the initial state names",
    ),
    "an equation reading a value set on the nodes": (
        _node(CONSTANT_K, more='    m: {equation: {rhs: "2 * k"}}\n')
        .replace('"-k*x"', '"-k*x + 0*m"')
        .replace("  number_of_nodes: 5\n", NODES_WITH_K),
        "'m'.*its equation reads 'k', whose value on the network comes from per-node values on the network's nodes",
    ),
    "an equation reading a used reference": (
        _edge(SCALED.replace("s: {value: 0.75}", 's: {value: 0.75, used: {iri: "tvbo:network/none", output: s}}')),
        "'w'.*its equation reads 's', whose value on the network comes from a `used` reference",
    ),
    "a normal start with a domain": (
        _node(
            "{value: 0.5, distribution: {name: Normal, domain: {lo: 0.0, hi: 6.0}, parameters: {mean: {value: 3.0}, std: {value: 0.5}}}}"
        ),
        "'k'.*a normal draw is unbounded, so the `domain`.*Declare TruncatedNormal",
    ),
    "a free parameter's initial value beside a draw": (
        _node(SAMPLED_K) + OPTIMIZATION,
        "free parameter 'k' declares an `initial_value`, and the parameter also declares where it starts",
    ),
    "a run seed beyond the key's range": (
        _node(SAMPLED_K, seed=2**32),
        r"execution.random_seed = 4294967296 is outside 0 to 4294967295",
    ),
    "a negative run seed": (_node(SAMPLED_K, seed=-1), r"execution.random_seed = -1 is outside 0 to 4294967295"),
    "a seed axis value beyond the key's range": (
        _node(SAMPLED_K) + SEED_AXIS.format(seeds=[1, 2**32]),
        r"exploration 'seeds': execution.random_seed = 4294967296 is outside",
    ),
    "a draw per trial under a wired algorithm": (
        TUNED.format(g=f"{{value: 0.2, {TUNED_SHAPE}, {_uniform(0.0, 1.0, ', axis: trial')}}}", eta=0.2, seed=1)
        + WIRED.replace("    algorithms: [tune]\n", "    algorithms: [tune]\n    n_trials: 2\n"),
        "exploration 'seeds'.*parameter 'g' of dynamics 'Relax'.*drawn per trial.*the wired algorithm tunes each cell once",
    ),
}


@pytest.mark.parametrize("case", list(REFUSED))
def test_tvboptim_refuses_what_it_cannot_honour_by_name(case):
    spec, message = REFUSED[case]
    with pytest.raises(ValueError, match=message):
        SimulationExperiment.from_string(spec).render_code("tvboptim")


def test_a_free_parameter_is_matched_to_a_start_by_the_keypath_it_binds():
    """``noise.sigma`` with an `initial_value` binds the noise, so a dynamics ``sigma`` that starts from a draw is no second definition of its start; ``Decay.sigma`` binds that parameter and is refused."""
    from tvbo.templates.tvboptim.utils import parameter_keypath, refuse_initial_value_on_parameter_start

    groups = [{"owner_kind": "dynamics", "owner": "Decay", "names": ["sigma"]}]
    free = {"name": "sigma", "coupling_key": None, "dynamics_key": "noise", "initial_value": 0.1}
    stages = [{"name": "fit", "free_parameters": [free]}]
    refuse_initial_value_on_parameter_start(stages, groups, parameter_keypath)
    free["dynamics_key"] = "Decay"
    with pytest.raises(ValueError, match="free parameter 'sigma' declares an `initial_value`"):
        refuse_initial_value_on_parameter_start(stages, groups, parameter_keypath)


RENDERED_WITHOUT_STARTS = [
    "tvb",
    "jax",
    "pde",
    "brian2",
    "julia",
    "modelingtoolkit",
    "bifurcationkit",
    "pyrates-bifurcation",
    "numcont",
    "neuroml",
    "lems",
    "rateml",
    "rateml-cuda",
    "rateml-driver",
    "pyrates-yaml",
]


@pytest.mark.parametrize("backend", RENDERED_WITHOUT_STARTS)
@pytest.mark.parametrize("declared", ["a distribution", "a distribution drawn per trial", "an equation"])
def test_a_backend_that_draws_no_start_refuses_the_declaration_by_name(backend, declared):
    spec, name, what = {
        "a distribution": (_node(SAMPLED_K), "k", "a distribution sampled over space"),
        "a distribution drawn per trial": (_node(PER_TRIAL_K), "k", "a distribution sampled over space, drawn per trial"),
        "an equation": (_edge(TIED, pre="x_j * (p + q)"), "p", "an equation"),
    }[declared]
    with pytest.raises(
        NotImplementedError, match=rf"the {backend} backend does not lower .*parameter '{name}' of \w+ '\w+' \({what}\)"
    ):
        SimulationExperiment.from_string(spec).render_code(backend)


@pytest.mark.parametrize("backend", ["python", "pyrates", "gillespie", "cuda"])
def test_a_run_path_that_renders_nothing_refuses_it_too(backend):
    with pytest.raises(NotImplementedError, match="does not lower parameter 'k' of dynamics 'Decay'"):
        SimulationExperiment.from_string(_node(SAMPLED_K)).run(backend)


def _bare(spec):
    """*spec* without its observations, which NetworkDynamics.jl does not lower."""
    return spec.split("observations:")[0]


def test_networkdynamics_keys_its_draws_by_the_run_seed():
    """Its one random stream is seeded with the distribution's own seed, else with `execution.random_seed`, as on tvboptim."""
    own = "{value: 0.5, " + _uniform(0.0, 2.0, ", seed: 11") + "}"
    for seed in (1, 2):
        code = SimulationExperiment.from_string(_bare(_node(SAMPLED_K, seed=seed))).render_code("networkdynamics")
        assert f"rng = MersenneTwister({seed})" in code and "s.p.v[node, :k] = 0.0 .+ (2.0 - 0.0) .* rand(rng)" in code
        assert "rng = MersenneTwister(11)" in SimulationExperiment.from_string(_bare(_node(own, seed=seed))).render_code(
            "networkdynamics"
        )


NORMAL_K = "{value: 0.5, distribution: {name: Normal, parameters: {mean: {value: 3.0}, std: {value: 0.5}}}}"
TRUNCATED_K = "{value: 0.5, distribution: {name: TruncatedNormal, domain: {lo: 2.75, hi: 3.5}, parameters: {mean: {value: 3.0}, std: {value: 0.5}}}}"
ND_REFUSED = {
    "a coupling parameter": (_edge(SAMPLED_W), NotImplementedError, "parameter 'w' of coupling 'LinearCoupling'"),
    "a normal": (_bare(_node(NORMAL_K)), NotImplementedError, "'k'.*declared as a normal, would be drawn uniform"),
    "a truncated normal": (
        _bare(_node(TRUNCATED_K)),
        NotImplementedError,
        "'k'.*declared as a truncated normal, would be drawn uniform",
    ),
    "a family no backend draws": (
        _bare(_node("{value: 0.5, distribution: {name: LogNormal, domain: {lo: 0.1, hi: 3.0}}}")),
        ValueError,
        "'k'.*no draw is implemented for the distribution 'LogNormal'",
    ),
    "a normal without moments": (
        _bare(_node("{value: 0.5, distribution: {name: Gaussian, domain: {lo: 0.0, hi: 2.0}}}")),
        ValueError,
        "'k'.*declares no mean, std",
    ),
    "per-node values on the nodes": (
        _bare(_node(SAMPLED_K).replace("  number_of_nodes: 5\n", NODES_WITH_K)),
        ValueError,
        "'k'.*also declares per-node values on the network's nodes",
    ),
    "a steady-state search": (
        _bare(_node(SAMPLED_K)).replace("execution: {random_seed: 1}", "execution: {random_seed: 1, find_fixpoint: true}"),
        NotImplementedError,
        "which a steady-state search skips, so parameter 'k' of dynamics 'Decay'",
    ),
    "a draw per trial": (
        _bare(_node(PER_TRIAL_K)),
        NotImplementedError,
        r"the networkdynamics backend draws a parameter start once per run, so parameter 'k' of dynamics 'Decay' \(a distribution sampled over space, drawn per trial\) would start every trial",
    ),
    "distributions under two seeds": (
        _bare(_node(SAMPLED_K)).replace(
            "initial_value: 1.0}", "initial_value: 1.0, distribution: {domain: {lo: 0.0, hi: 1.0}, seed: 12}}"
        ),
        NotImplementedError,
        "one random stream seeded once, and state variable 'x' of dynamics 'Decay' resolves to seed 12, parameter 'k' of dynamics 'Decay' resolves to seed 1",
    ),
}


@pytest.mark.parametrize("case", list(ND_REFUSED))
def test_networkdynamics_refuses_the_starts_it_does_not_draw_as_declared(case):
    """It draws a dynamics parameter per node, uniform on the distribution's domain, from one stream under one seed, when it sets the initial state, and nothing else."""
    assert "rand(rng)" in SimulationExperiment.from_string(_bare(_node(SAMPLED_K))).render_code("networkdynamics")
    spec, error, message = ND_REFUSED[case]
    with pytest.raises(error, match=message):
        SimulationExperiment.from_string(spec).render_code("networkdynamics")


def test_a_node_that_restates_the_dynamics_value_is_a_second_definition_on_both_drawing_backends():
    """Node 0 declares k at the dynamics' own value, which pins it there as any other value would; tvboptim and NetworkDynamics.jl refuse the draw over it with one message."""
    spec = _node(SAMPLED_K).replace("  number_of_nodes: 5\n", NODES_WITH_K.replace("value: 0.1", "value: 0.5"))
    messages = []
    for backend, recipe in (("tvboptim", spec), ("networkdynamics", _bare(spec))):
        with pytest.raises(ValueError) as refused:
            SimulationExperiment.from_string(recipe).render_code(backend)
        messages.append(str(refused.value))
    assert messages[0] == messages[1]
    assert (
        "parameter 'k' of dynamics 'Decay' (a distribution sampled over space): the parameter also declares per-node values on the network's nodes"
        in messages[0]
    )


def test_a_format_that_integrates_nothing_writes_the_declaration_out():
    text = SimulationExperiment.from_string(_node(SAMPLED_K)).render("yaml")
    assert "distribution" in text and "Uniform" in text


def _producer_values():
    """The values a producer hands a parameter in `test_a_sourced_or_produced_start_is_a_second_definition`."""
    return {"k": np.ones(5)}


def test_a_sourced_or_produced_start_is_a_second_definition(tmp_path):
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "k.h5", "w") as store:
        store.create_dataset("k", data=np.ones(5))
    (tmp_path / "sourced.yaml").write_text(
        _node("{source: k.h5, measure: k, heterogeneous: true, " + _uniform(0.0, 2.0) + "}")
    )
    produced = (
        "{producer: {callable: {name: _producer_values, module: "
        + __name__
        + "}, output: k}, heterogeneous: true, "
        + _uniform(0.0, 2.0)
        + "}"
    )
    for experiment in (
        SimulationExperiment.from_file(str(tmp_path / "sourced.yaml")),
        SimulationExperiment.from_string(_node(produced)),
    ):
        with pytest.raises(ValueError, match="'k'.*also declares a `source` or `producer`"):
            experiment.render_code("tvboptim")


def test_the_draw_key_is_the_owner_and_the_name():
    """The tag folded into the key is a checksum of the owner and the parameter name, the same in every process."""
    for spec, label in ((_node(SAMPLED_K), "dynamics.k"), (_edge(SAMPLED_W), "LinearCoupling.w")):
        code = SimulationExperiment.from_string(spec).render_code("tvboptim")
        assert f"jax.random.fold_in(_key, {zlib.crc32(label.encode()) & 0x7FFFFFFF})" in code


def test_another_process_draws_the_same_start():
    """Under another hash seed a fresh interpreter draws the same rates, so nothing in the key depends on Python's salted `hash`."""
    script = (
        "import json, sys\n"
        f"sys.path.insert(0, {os.path.dirname(__file__)!r})\n"
        "import test_parameter_start_from_distribution as t\n"
        "print(json.dumps(t._rate(t._node(t.SAMPLED_K)).tolist()))\n"
    )
    env = {**os.environ, "PYTHONHASHSEED": "777"}
    other = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True, timeout=600)
    here = _rate(_node(SAMPLED_K))
    assert np.unique(here).size == 5, "the rates are not drawn"
    np.testing.assert_array_equal(np.asarray(json.loads(other.stdout.strip().splitlines()[-1])), here)


TWO_COUPLINGS = (
    """
id: 5
label: Two couplings with a weight of one name each
dynamics:
  name: Leak
  parameters:
    a: {value: -0.3}
  coupling_inputs:
    c: {name: c, description: "coupling"}
    d: {name: d, description: "coupling"}
  state_variables:
    x: {equation: {rhs: "a * x + c + d"}, initial_value: 1.0, coupling_variable: true}
network:
  number_of_nodes: 3
  nodes: [{id: 0, label: r0}, {id: 1, label: r1}, {id: 2, label: r2}]
  edges:
    - {source: 0, target: 1, weight: 1.0}
    - {source: 1, target: 2, weight: 1.0}
  coupling:
"""
    + "".join(
        f"""    {key}:
      label: {key}
      parameters:
        w: {{value: 1.0, heterogeneous: true, shape: "(n_nodes,)", distribution: {{name: Uniform, domain: {{lo: 0.0, hi: 2.0}}}}}}
      pre_expression: {{rhs: "x_j * w"}}
      post_expression: {{rhs: "0.05 * gx"}}
      incoming_states: [x]
      local_states: [x]
"""
        for key in ("c", "d")
    )
    + """integration: {method: euler, step_size: 0.5, duration: 2.0, unit: ms}
execution: {random_seed: 1}
"""
)


def test_a_coupling_start_is_grouped_under_its_key_in_the_couplings_handed_in():
    """The group, and the function the module emits for it, take the key the template integrates the coupling under, whatever name the network declares it by."""
    from tvbo.templates.tvboptim.utils import resolve_parameter_starts
    from tvbo.utils import network_couplings

    experiment = SimulationExperiment.from_string(_edge(SAMPLED_W))
    (declared,) = network_couplings(experiment.network).values()
    (group,) = resolve_parameter_starts(experiment, 1, couplings={"renamed": declared}, coupling_inputs={"renamed": ["c"]})
    assert (group["owner_kind"], group["owner"], group["function"], group["variable"]) == (
        "coupling",
        "renamed",
        "_renamed_parameter_starts",
        "_renamed_starts",
    )
    assert group["names"] == ["w"] and group["inputs_served"] == ["c"]


def test_two_couplings_draw_a_parameter_of_one_name_independently():
    _, network, _ = _built(TWO_COUPLINGS)
    c, d = (np.asarray(network.params.coupling[key]["w"]) for key in ("c", "d"))
    assert c.shape == d.shape == (3,) and not np.allclose(c, d)


def test_a_dense_and_a_sparse_graph_of_one_matrix_draw_alike():
    """Both list the edges in row order over (target, source), so each edge receives the same value."""
    dense = _edge(SAMPLED_W)
    sparse = dense.replace("  number_of_nodes: 3\n", "  number_of_nodes: 3\n  graph_representation: sparse\n")
    (_, on_dense, weights), (_, on_sparse, _) = _built(dense), _built(sparse)
    assert type(on_dense.graph).__name__ == "DenseGraph" and type(on_sparse.graph).__name__ == "SparseGraph"
    w = np.asarray(on_sparse.params.coupling["c"]["w"])
    np.testing.assert_array_equal(w, np.asarray(on_dense.params.coupling["c"]["w"]))
    assert np.unique(w[weights != 0]).size == 6


def test_a_library_copy_of_the_model_is_one_owner():
    """A network that lists the model again under its own name is homogeneous, and its start is drawn once."""
    spec = _node(SAMPLED_K)
    model = spec[spec.index("dynamics:\n") : spec.index("network:\n")].splitlines()[1:]
    library = "  dynamics:\n    Decay:\n" + "".join(f"    {line}\n" for line in model)
    spec = spec.replace("network:\n  label: Five\n", "network:\n  label: Five\n" + library)
    experiment = SimulationExperiment.from_string(spec)
    assert experiment.network.dynamics["Decay"] is not experiment.dynamics
    k = _rate(spec)
    assert np.unique(k).size == 5, "the rates are not drawn"
    np.testing.assert_array_equal(k, _rate(_node(SAMPLED_K)))


def test_the_heterogeneous_runner_refuses_a_start():
    pytest.importorskip("tvboptim.experimental.network_dynamics")
    import yaml

    from tvbo import Dynamics, Network

    drive = Dynamics.from_string(
        "name: Drive\nparameters: {a: {value: 0.5}}\nstate_variables:\n  x: {equation: {rhs: '-a*x + c_in'}, initial_value: 1.0}\ncoupling_inputs: {c_in: {}}"
    )
    excitable = Dynamics.from_string(
        "name: Excitable\n"
        "parameters: {I_ext: {value: 0.3, distribution: {name: Uniform, domain: {lo: 0.0, hi: 1.0}}}}\n"
        "state_variables:\n  v: {equation: {rhs: '-v + I_ext + c_in'}, initial_value: -1.0}\ncoupling_inputs: {c_in: {}}"
    )
    network = Network(
        **yaml.safe_load(
            """
            label: Two kinds
            number_of_nodes: 2
            nodes: [{id: 0, label: A, dynamics: Drive}, {id: 1, label: B, dynamics: Excitable}]
            edges:
              - {source: 0, target: 1, parameters: {weight: {value: 0.8}}, source_var: x_out, target_var: c_in, directed: true}
            """
        )
    )
    network.dynamics["Drive"] = drive
    network.dynamics["Excitable"] = excitable
    experiment = SimulationExperiment(network=network)
    experiment.integration.duration = 10.0
    experiment.integration.step_size = 0.1
    with pytest.raises(
        NotImplementedError, match="heterogeneous tvboptim backend does not lower parameter 'I_ext' of dynamics 'Excitable'"
    ):
        experiment.run("tvboptim")


TRIALS = """
id: 6
label: A drawn rate under an ensemble of initial conditions
dynamics:
  name: Ramp
  parameters:
    k: {k}
    b: {{value: 0.0}}
  state_variables:
    x: {{equation: {{rhs: "k + b"}}, initial_value: 0.0}}
    y: {{equation: {{rhs: "0 * y"}}, initial_value: 0.0, distribution: {{domain: {{lo: 0.0, hi: 1.0}}}}}}
network:
  number_of_nodes: 4
integration: {{method: euler, step_size: 0.5, duration: 2.0, unit: ms}}
execution: {{random_seed: 3, precision: float64}}
observations:
  x_last: {{source: [x], aggregation: last}}
  y_last: {{source: [y], aggregation: last}}
explorations:
  trials:
    n_trials: 3
    space:
      Ramp.b: {{explored_values: [0.0]}}
"""


@pytest.mark.parametrize(
    "k", [SAMPLED_K, SAMPLED_K.replace("hi: 2.0}}", "hi: 2.0}, axis: space}")], ids=["by default", "under axis space"]
)
def test_the_trials_of_a_cell_share_its_draw(k):
    """Three trials redraw the initial conditions and integrate the one start the cell drew: x ramps at the drawn rate in every trial."""
    spec = TRIALS.format(k=k)
    observed = SimulationExperiment.from_string(spec).run("tvboptim", quiet=True).explorations["trials"].observations
    x, y = (np.asarray(observed[name]).reshape(3, 4) for name in ("x_last", "y_last"))
    assert np.unique(y.round(9), axis=0).shape[0] == 3, "the trials did not redraw their initial conditions"
    k = _rate(spec)
    for row in x:
        np.testing.assert_allclose(row, 2.0 * k, rtol=1e-12)


ZIP_STREAMED = """
explorations:
  seeds:
    mode: zip
    space:
      - {parameter: Decay.b, explored_values: [0.0, 0.0]}
      - {parameter: execution.random_seed, explored_values: [1, 2]}
"""


def test_a_zip_seed_axis_with_a_streamed_observation_draws_per_cell():
    """Under `mode: zip` and an observation streamed into the integrator's carry, each cell still integrates from the start a run at its seed draws."""

    def streamed(seed):
        spec = _node(SAMPLED_K, seed=seed, more="    b: {value: 0.0}\n").replace('"-k*x"', '"-k*x + b"')
        return spec.replace(
            "x_last: {label: Final x, source: [x], aggregation: last}",
            "x_mean: {source: [x], aggregation: mean, reduce: streaming}",
        )

    cells = SimulationExperiment.from_string(streamed(7) + ZIP_STREAMED).run("tvboptim", quiet=True)
    cells = np.asarray(cells.explorations["seeds"].observations["x_mean"]).reshape(2, 5)
    assert not np.allclose(cells[0], cells[1])
    for row, seed in zip(cells, (1, 2), strict=True):
        alone = SimulationExperiment.from_string(streamed(seed)).run("tvboptim", quiet=True)
        np.testing.assert_allclose(row, np.asarray(alone.integration.observations["x_mean"].data).reshape(5), rtol=1e-12)


def test_a_format_registered_without_the_capability_refuses_every_start(monkeypatch):
    from tvbo.export import registry

    probe = registry.ExportFormat("probe-format", "Probe", ".txt", "text/plain", lambda experiment, **kwargs: "rendered")
    monkeypatch.setitem(registry._REGISTRY, "probe-format", probe)
    assert registry.render(SimulationExperiment.from_string(_node(CONSTANT_K)), "probe-format") == "rendered"
    with pytest.raises(NotImplementedError, match="the probe-format backend does not lower parameter 'k' of dynamics 'Decay'"):
        registry.render(SimulationExperiment.from_string(_node(SAMPLED_K)), "probe-format")


IC_DRAW = ", distribution: {domain: {lo: 0.0, hi: 1.0}}"
Y_NOISE = ", noise: {additive: true, parameters: {sigma: {value: 0.1}}}"
HOST_OBSERVATION = "  x_np: {source: [x_last], pipeline: [{name: copy, callable: {name: asarray, module: numpy}}]}\n"
STOCHASTIC_S = "    s: {value: 0.0, distribution: {name: Uniform, domain: {lo: -1.0, hi: 1.0}, axis: time}}\n"


def _ensemble(k, b="{value: 0.0}", y=IC_DRAW, space="Ramp.b: {explored_values: [0.0]}", more=""):
    """The TRIALS recipe with rate *k*, parameter *b* and *more* parameters, *y* declared after the initial value of y, and the exploration axis *space*, or none."""
    spec = TRIALS.format(k=k).replace("    b: {value: 0.0}\n", f"    b: {b}\n{more}").replace(IC_DRAW, y)
    if space is None:
        return spec.split("    space:")[0]
    return spec.replace("Ramp.b: {explored_values: [0.0]}", space)


def _trial_draws(seed, n_trials, shape, label="dynamics.k", lo=0.0, hi=2.0):
    """What each trial of an ensemble under *seed* draws for the parameter *label*: the key a run draws it under, with the trial index folded in."""
    key = jax.random.fold_in(jax.random.key(seed), zlib.crc32(label.encode()) & 0x7FFFFFFF)
    return np.stack(
        [np.asarray(jax.random.uniform(jax.random.fold_in(key, i), shape, minval=lo, maxval=hi)) for i in range(n_trials)]
    )


def _trial_rows(spec, name="x_last", cells=1):
    """The observation *name* of the three-trial ensemble `trials` of *spec* on tvboptim, as (cell, trial, node)."""
    observed = SimulationExperiment.from_string(spec).run("tvboptim", quiet=True).explorations["trials"].observations[name]
    assert "trial" in observed.dims, f"the ensemble ran no trials: {observed.dims}"
    return np.asarray(observed).reshape(cells, 3, -1)


def _on_the_host(spec):
    """*spec* with an observation whose pipeline calls numpy, which cannot trace, so its trials run in a host loop."""
    return spec.replace(
        "  y_last: {source: [y], aggregation: last}\n", "  y_last: {source: [y], aggregation: last}\n" + HOST_OBSERVATION
    )


ENSEMBLES = {
    "initial conditions": _ensemble(PER_TRIAL_K),
    "a stochastic parameter": _ensemble(PER_TRIAL_K, y="", more=STOCHASTIC_S).replace('rhs: "0 * y"', 'rhs: "s"'),
    "noise": _ensemble(PER_TRIAL_K, y=Y_NOISE),
    "nothing else": _ensemble(PER_TRIAL_K, y=""),
}
ENSEMBLES["initial conditions, on the host"] = _on_the_host(ENSEMBLES["initial conditions"])
ENSEMBLES["a stochastic parameter, on the host"] = _on_the_host(ENSEMBLES["a stochastic parameter"])


@pytest.mark.parametrize("ensemble", list(ENSEMBLES))
def test_a_start_drawn_per_trial_gives_each_trial_its_own_draw(ensemble):
    """Whatever else the trials redraw, trial i ramps x at the rate drawn under the run's key with i folded in, and no two trials share it."""
    spec = ENSEMBLES[ensemble]
    x = _trial_rows(spec)[0]
    assert np.unique(x.round(9), axis=0).shape[0] == 3, "two trials integrated one rate"
    np.testing.assert_allclose(x, 2.0 * _trial_draws(3, 3, (4,)), rtol=1e-12)


def test_the_trials_of_a_seed_axis_draw_under_their_cells_seed():
    x = _trial_rows(_ensemble(PER_TRIAL_K, y="", space="execution.random_seed: {explored_values: [1, 2]}"), cells=2)
    for rows, seed in zip(x, (1, 2), strict=True):
        np.testing.assert_allclose(rows, 2.0 * _trial_draws(seed, 3, (4,)), rtol=1e-12)


@pytest.mark.parametrize("seed", [None, 11], ids=["under the run's seed", "under its own seed"])
def test_an_equation_reading_a_start_drawn_per_trial_follows_it_in_every_trial(seed):
    """With b = 1 - k, x ramps at 1 in every trial while y, driven by k alone, ramps at each trial's own rate."""
    k = PER_TRIAL_K if seed is None else PER_TRIAL_K.replace("axis: trial", f"axis: trial, seed: {seed}")
    spec = _ensemble(k, b='{equation: {rhs: "1 - k"}}', y="", space=None).replace('rhs: "0 * y"', 'rhs: "k"')
    x, y = (_trial_rows(spec, name)[0] for name in ("x_last", "y_last"))
    np.testing.assert_allclose(y, 2.0 * _trial_draws(seed or 3, 3, (4,)), rtol=1e-12)
    np.testing.assert_allclose(x, 2.0, rtol=1e-12)


def test_a_weight_drawn_per_trial_is_redrawn_on_the_edges_for_each_trial():
    """Trial i integrates the network whose edges carry the weights drawn under the run's key with i folded in, and the declared value off them."""
    spec = _edge(PER_TRIAL_W).replace("execution: {random_seed: 1}", "execution: {random_seed: 1, precision: float64}")
    ensemble = spec + "explorations:\n  trials:\n    n_trials: 3\n    space:\n      Leak.a: {explored_values: [-0.3]}\n"
    x = _trial_rows(ensemble)[0]
    module, network, weights = _built(spec)
    draws = _trial_draws(1, 3, (int((weights != 0).sum()),), label="LinearCoupling.w")
    assert np.unique(x.round(9), axis=0).shape[0] == 3, "two trials integrated one set of weights"
    for row, drawn in zip(x, draws, strict=True):
        w = np.ones_like(weights)
        w[weights != 0] = drawn
        network.params.coupling["c"]["w"] = jnp.asarray(w)
        alone = module.run_simulation(network).observations["x_last"]
        np.testing.assert_allclose(row, np.asarray(getattr(alone, "data", alone)).reshape(-1), rtol=1e-10)


def test_a_start_drawn_per_trial_under_its_own_seed_ignores_the_run_seed():
    own = PER_TRIAL_K.replace("axis: trial", "axis: trial, seed: 11")
    for seed in (3, 4):
        x = _trial_rows(_ensemble(own, y="").replace("random_seed: 3", f"random_seed: {seed}"))[0]
        np.testing.assert_allclose(x, 2.0 * _trial_draws(11, 3, (4,)), rtol=1e-12)


def test_a_seed_passed_at_run_time_keys_the_trials_draws():
    module, _, weights = _built(_ensemble(PER_TRIAL_K, y=""))
    x = module.run_experiment(jnp.asarray(weights), mode="exploration", random_seed=5, quiet=True).explorations.trials
    np.testing.assert_allclose(np.asarray(x.observations["x_last"]).reshape(3, -1), 2.0 * _trial_draws(5, 3, (4,)), rtol=1e-12)


def test_a_start_drawn_per_trial_is_the_runs_draw_outside_an_ensemble():
    np.testing.assert_array_equal(_rate(_node(PER_TRIAL_K)), _rate(_node(SAMPLED_K)))


def test_an_initial_condition_declared_per_trial_is_redrawn_as_under_space():
    """A state variable's initial condition is redrawn for each trial under either axis, as the initial-condition ensemble always has been."""
    space, trial = (
        _trial_rows(_ensemble(SAMPLED_K, y=y), "y_last")[0]
        for y in (IC_DRAW, IC_DRAW.replace("hi: 1.0}}", "hi: 1.0}, axis: trial}"))
    )
    assert np.unique(space.round(9), axis=0).shape[0] == 3
    np.testing.assert_array_equal(trial, space)


def test_a_format_that_draws_once_per_run_refuses_a_start_drawn_per_trial(monkeypatch):
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
    assert registry.render(SimulationExperiment.from_string(_node(SAMPLED_K)), "probe-format") == "rendered"
    with pytest.raises(
        NotImplementedError,
        match="the probe-format backend draws a parameter start once per run, so parameter 'k' of dynamics 'Decay'",
    ):
        registry.render(SimulationExperiment.from_string(_node(PER_TRIAL_K)), "probe-format")
