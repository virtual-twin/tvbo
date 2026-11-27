"""A sliding-window algorithm can put a stage on one clock that counts the periods filling its window.

Without a declaration the window is filled ahead of a stage's first iteration, by periods in which no update rule is applied and which no count includes. `window_fill: in_stage` makes those periods iterations of the stage: `n_iterations` is the stage's whole length, a rule that reads no window is applied from its algorithm's `update_start` on, a rule that reads the window waits for the first full one, and `window_start` keeps the opening samples of a trajectory out of every window. The clock counts periods from where the trajectory starts, so it restarts with a stage that declares `reset_state` and runs on through one that continues.

The counts are read off rules that add a constant step per application, so they do not depend on what the model does. Period `i` of a stage, counted from zero, ends at `(i + 1) * simulation_period`, which is the time a start is declared as.

The first tests run the curated FIC + FIC_EIB experiment, whose rules are of both kinds. The later ones run a toy model of three relaxing nodes whose windowed observation counts the samples its window holds, so that what a window contains, what is recorded before it is full and what a refused declaration says are read directly.
"""

import copy
import functools
import re
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from tvbo import Network
from tvbo.classes.experiment import SimulationExperiment
from tvbo.data.types import AlgorithmResult
from tvbo.datamodel import Algorithm
from tvbo.templates.tvboptim.utils import stage_clock_settings
from tvbo.utils import yaml_loader
from tvbo.utils.source import loading_from

_EXP = "tvbo/database/experiments/EI_Tuning_FIC_EIB_Optimization.yaml"
_PERIOD = 720.0
_STEP_INCLUDED, _STEP_WINDOWED = 1 / 64, 1 / 32
_TUNED = ("J_i", "wLRE", "wFFI")


def _ends(period, length=_PERIOD):
    """The simulated time at which the stage's period ``period``, counted from zero, ends when a period is ``length`` long."""
    return (period + 1) * length


def _recipe(stages, counting):
    """The curated FIC + FIC_EIB recipe as a mapping, FIC_EIB entering from the declared initial state on ``stages`` of ``(n_iterations, window_size, reset_state)``.

    FIC tunes nothing on its own and no evaluation follows. With ``counting`` every rule adds its rate once per application: the included one reads no observation, and FIC_EIB's own two list the windowed FC in ``requires``. Without it the curated rules stand and the included one has a rate of zero.
    """
    data = yaml_loader.load_as_dict(_EXP)
    fic, fe = data["algorithms"]["fic"], data["algorithms"]["fic_eib"]
    fic["n_iterations"], fic["evaluate"], fe["evaluate"] = 1, False, False
    for hp in fic["hyperparameters"]:
        if hp["name"] == "eta":
            hp["value"] = 0.0
    del fe["depends_on"]
    fe["includes"] = [{"algorithm": "fic", "arguments": [{"name": "eta", "value": _STEP_INCLUDED if counting else 0.0}]}]
    if counting:
        fic["update_rules"][0]["equation"] = {"rhs": "J_i + eta"}
        for rule in fe["update_rules"]:
            rule.update(equation={"rhs": f"{rule['target_parameter']['name']} + eta"}, requires=["fc_inloop"], warmup=False)
    fe["stages"] = [
        {
            "n_iterations": n_iterations,
            **({"reset_state": True} if reset else {}),
            "arguments": [
                {"name": "eta", "value": _STEP_WINDOWED if counting else 0.1},
                {"name": "window_size", "value": window},
            ],
        }
        for n_iterations, window, reset in stages
    ]
    return data


def _declare(algorithm, **slots):
    """Write declarations into one algorithm of a recipe, failing by name where this tvbo's datamodel has no slot to read them from; a value of None declares nothing."""
    known = set(Algorithm.model_fields)
    for name, value in slots.items():
        if value is None:
            continue
        if name not in known:
            pytest.fail(
                f"Algorithm has no `{name}` slot, so `{name}: {value}` declares nothing and a stage cannot be put on one clock"
            )
        algorithm[name] = value


def _experiment(data):
    """The experiment a recipe mapping declares, resolved as if it had been loaded from the curated file."""
    with loading_from(_EXP) as source:
        exp = SimulationExperiment.from_dict(data)
        exp._source_file = source
    return exp


def _tuned(algo):
    return {
        "J_i": np.asarray(algo.state.dynamics.J_i),
        "wLRE": np.asarray(algo.state.coupling.EIBLinearCoupling.wLRE),
        "wFFI": np.asarray(algo.state.coupling.EIBLinearCoupling.wFFI),
    }


def _run(data):
    """The tuned parameters FIC_EIB entered with and the ones it left with."""
    algorithms = _experiment(data).run("tvboptim", mode="algorithms", quiet=True).algorithms
    return _tuned(algorithms["fic"]), _tuned(algorithms["fic_eib"])


def _applications(data):
    """How often each rule of a counting recipe was applied over FIC_EIB's schedule, read off the constant step an application adds."""
    start, end = _run(data)
    counts = {}
    for name, step in zip(_TUNED, (_STEP_INCLUDED, _STEP_WINDOWED, _STEP_WINDOWED), strict=True):
        per_entry = np.unique((end[name] - start[name]) / step)
        assert per_entry.size == 1, f"{name} moved by {per_entry} steps across its entries"
        counts[name] = float(per_entry[0])
    return counts


@pytest.mark.parametrize(
    ("length", "window", "excluded", "included_first", "own_first", "expected"),
    [
        (9, 4, 0, 0, None, {"J_i": 9, "wLRE": 6, "wFFI": 6}),
        (14, 4, 3, 2, 1, {"J_i": 12, "wLRE": 8, "wFFI": 8}),
        (12, 5, 0, 0, 7, {"J_i": 12, "wLRE": 5, "wFFI": 5}),
    ],
)
def test_each_rule_acts_from_its_own_start_on_the_stage_clock(length, window, excluded, included_first, own_first, expected):
    """A stage of ``length`` periods applies the included rule ``length - included_first`` times and a windowed rule ``length - (excluded + window - 1)`` times, or ``length - own_first`` where its algorithm's own start is later.

    That is the counting of a simulator that runs both in one loop: 50 000 samples, a rate rule from sample 10, 51 samples kept out of every window and a window of 150 give 49 990 and 49 800 applications.
    """
    pytest.importorskip("tvboptim")
    data = _recipe(((length, window, False),), counting=True)
    fic, fe = data["algorithms"]["fic"], data["algorithms"]["fic_eib"]
    _declare(fe, window_fill="in_stage", window_start=_ends(excluded) if excluded else None)
    _declare(fe, update_start=None if own_first is None else _ends(own_first))
    _declare(fic, update_start=_ends(included_first) if included_first else None)
    assert _applications(data) == expected


def test_the_first_window_of_a_clocked_stage_holds_no_excluded_sample():
    """A clocked stage that ends on its first full window makes the one update a stage makes whose fill ran ahead and whose rule waited for the same window.

    Both integrate the same periods with no parameter moving before the last one, so the windows are the same samples and the tuned weights agree. An excluded sample in the window, or a window one period early or late, changes how many updates the clocked stage makes or what the one reads.
    """
    pytest.importorskip("tvboptim")
    excluded, window = 5, 4
    clocked = _recipe(((excluded + window, window, False),), counting=False)
    _declare(clocked["algorithms"]["fic_eib"], window_fill="in_stage", window_start=_ends(excluded))
    ahead = _recipe(((excluded, window, False),), counting=False)
    _declare(ahead["algorithms"]["fic_eib"], update_start=_ends(excluded - 1))
    untouched, reference = _run(ahead)
    assert not np.array_equal(reference["wLRE"], untouched["wLRE"])
    _, got = _run(clocked)
    for name in _TUNED:
        np.testing.assert_allclose(got[name], reference[name], rtol=1e-7, atol=0, err_msg=name)


@pytest.mark.parametrize(
    ("reset", "expected"), [(True, {"J_i": 14, "wLRE": 10, "wFFI": 10}), (False, {"J_i": 17, "wLRE": 15, "wFFI": 15})]
)
def test_the_clock_restarts_with_a_reset_stage_and_runs_on_through_a_continuous_one(reset, expected):
    """Two stages of 10 periods, the included rule from period 3, two samples kept out of a window of 4.

    A reset stage starts its trajectory again, so each stage waits anew: 7 and 5 applications per stage. A continuous second stage is later on the same trajectory and holds a full window from its first period, so it applies both rules in all 10.
    """
    pytest.importorskip("tvboptim")
    data = _recipe(((10, 4, reset), (10, 4, reset)), counting=True)
    _declare(data["algorithms"]["fic_eib"], window_fill="in_stage", window_start=_ends(2))
    _declare(data["algorithms"]["fic"], update_start=_ends(3))
    assert _applications(data) == expected


@pytest.mark.parametrize(("reset", "first_full"), [(False, 2), (True, 6)], ids=["continuous", "reset"])
def test_a_window_that_grows_between_stages_holds_the_samples_a_window_of_one_size_holds(reset, first_full):
    """Eight periods and eight more, with no parameter moving, so the trajectory does not depend on the windows. A schedule whose window grows from 4 to 7 slides it over a ring sized for the larger one; a schedule at 7 throughout does not. From the period in which the grown window is complete, the fit score the second stage records is the same in both: the ring's window holds the same seven samples, none of them a row that was never written."""
    pytest.importorskip("tvboptim")
    records = []
    for first_window in (4, 7):
        data = _recipe(((8, first_window, reset), (8, 7, reset)), counting=False)
        fe = data["algorithms"]["fic_eib"]
        for stage in fe["stages"]:
            stage["arguments"][0]["value"] = 0.0
        _declare(fe, window_fill="in_stage")
        algorithms = _experiment(data).run("tvboptim", mode="algorithms", quiet=True).algorithms
        records.append(np.asarray(algorithms["fic_eib"].history.fc_rmse))
    grown, fixed = records
    assert np.isnan(grown[:first_full]).all() and np.isfinite(grown[first_full:]).all()
    assert np.ptp(fixed[first_full:]) > 0
    np.testing.assert_allclose(grown[first_full:], fixed[first_full:], rtol=1e-9, atol=0)


@pytest.mark.parametrize("window", [4, 9])
def test_without_the_declaration_the_fill_runs_ahead_and_applies_no_rule(window):
    """A stage that declares no clock applies every rule in each of its ``n_iterations`` and in no period of the fill, whatever the window."""
    pytest.importorskip("tvboptim")
    assert _applications(_recipe(((5, window, False),), counting=True)) == {"J_i": 5, "wLRE": 5, "wFFI": 5}


def test_without_the_declaration_a_start_counts_from_the_first_iteration_after_the_fill():
    """With the fill ahead of the stage, `update_start` skips iterations and not periods of the fill: six iterations behind a window of 4 apply a rule starting with the third in four of them."""
    pytest.importorskip("tvboptim")
    data = _recipe(((6, 4, False),), counting=True)
    _declare(data["algorithms"]["fic"], update_start=_ends(2))
    assert _applications(data) == {"J_i": 4, "wLRE": 6, "wFFI": 6}


def test_spelling_out_the_default_fill_renders_the_code_of_a_recipe_that_declares_none():
    """`window_fill: before_stage` is what an absent `window_fill` means, so declaring it changes no byte of the generated code; a datamodel without the slot has only the absent form."""
    plain = _recipe(((3, 4, False), (3, 6, True)), counting=False)
    spelled = copy.deepcopy(plain)
    if "window_fill" in Algorithm.model_fields:
        spelled["algorithms"]["fic_eib"]["window_fill"] = "before_stage"
    assert _experiment(spelled).render_code("tvboptim") == _experiment(plain).render_code("tvboptim")


_TOY = """
id: 1
dynamics:
  name: Relax
  label: Relax
  parameters:
    g: {value: 0.5, heterogeneous: true, shape: "(n_nodes,)"}
    h: {value: 0.0, heterogeneous: true, shape: "(n_nodes,)"}
    tau: {value: 1500.0}
  state_variables:
    x:
      initial_value: 0.0
      equation: {rhs: "(g + h - x) / tau"}
      coupling_variable: true
  output: [x]
network:
  number_of_nodes: 3
  nodes:
    - {id: 0, label: A}
    - {id: 1, label: B}
    - {id: 2, label: C}
  edges: []
integration: {method: euler, step_size: 10.0, duration: 200.0, unit: ms}
observations:
  signal:
    iri: tvbo:BOLD_Balloon
    source: [x]
    period: 200.0
  filled:
    source: [signal]
    pipeline:
      - callable: {name: count_nonzero, module: jax.numpy}
        arguments:
          a: {value: signal}
          axis: {value: 0}
algorithms:
  tune:
    objective: {type: activity_target, target_variable: x, target_value: 0.5}
    observations: [signal, filled]
    update_rules:
      - name: g_update
        target_parameter: {name: g}
        equation: {rhs: "g + eta + 0 * signal"}
      - name: h_update
        target_parameter: {name: h}
        equation: {rhs: "h + rate"}
        requires: [filled]
    hyperparameters:
      - {name: eta, value: 0.015625}
      - {name: rate, value: 0.03125}
      - {name: window_size, value: 4}
    n_iterations: 9
    simulation_period: 200.0
execution: {random_seed: 0, precision: float64}
"""
_TOY_PERIOD = 200.0
_TOY_START = {"g": 0.5, "h": 0.0}
_FC_OF_THE_WINDOW = {
    "callable": {"name": "compute_fc", "module": "tvboptim.observations.observation"},
    "arguments": {"timeseries": {"value": "signal"}, "skip_t": {"value": 0}},
}


def _toy(stages, window_of="count", handed=None):
    """The toy recipe as a mapping, its algorithm ``tune`` on ``stages`` of ``(n_iterations, window_size, reset_state)``; one stage that does not reset is a plain call without a schedule.

    Three uncoupled nodes relax towards ``g + h`` without noise, a haemodynamic signal is sampled once per period, and ``filled`` is computed from a window of those samples: with ``window_of="count"`` the number of its rows that hold a sample, which is the window size exactly when the window is full, and with ``"fc"`` the functional connectivity, which has a streaming form and a window whose size is traced. The rule on ``g`` names the period's own sample, which is no window, and the rule on ``h`` lists ``filled`` in ``requires``; each adds a constant step per application. With ``handed`` an algorithm ``settle`` runs first on a window of that many samples, tunes nothing and hands its window on.
    """
    data = yaml.safe_load(_TOY)
    tune = data["algorithms"]["tune"]
    if window_of == "fc":
        data["observations"]["filled"]["pipeline"] = [copy.deepcopy(_FC_OF_THE_WINDOW)]
    if handed is not None:
        settle = copy.deepcopy(tune)
        settle["n_iterations"] = 2
        settle["hyperparameters"] = [
            {"name": "eta", "value": 0.0},
            {"name": "rate", "value": 0.0},
            {"name": "window_size", "value": handed},
        ]
        tune["depends_on"] = ["settle"]
        data["algorithms"] = {"settle": settle, "tune": tune}
    if len(stages) == 1 and not stages[0][2]:
        tune["n_iterations"], window, _ = stages[0]
        tune["hyperparameters"][-1]["value"] = window
    else:
        tune["stages"] = [
            {
                "n_iterations": n_iterations,
                **({"reset_state": True} if reset else {}),
                "arguments": [{"name": "window_size", "value": window}],
            }
            for n_iterations, window, reset in stages
        ]
    return data


def _toy_counts(dynamics):
    """How often each toy rule was applied, read off the constant step it adds to every entry of its parameter."""
    counts = {}
    for (name, start), step in zip(_TOY_START.items(), (_STEP_INCLUDED, _STEP_WINDOWED), strict=True):
        per_entry = np.unique((np.asarray(getattr(dynamics, name)) - start) / step)
        assert per_entry.size == 1, f"{name} moved by {per_entry} steps across its entries"
        counts[name] = float(per_entry[0])
    return counts


def _toy_run(data):
    """The result of the toy's ``tune`` and how often each of its rules was applied."""
    algo = SimulationExperiment.from_dict(data).run("tvboptim", mode="algorithms", quiet=True).algorithms["tune"]
    return algo, _toy_counts(algo.state.dynamics)


def _recorded(length, first_full, window):
    """The record of ``filled`` over a stage of ``length`` periods whose window of ``window`` samples is complete from period ``first_full`` on."""
    return [np.nan] * first_full + [float(window)] * (length - first_full)


@pytest.mark.parametrize(("excluded", "first_full"), [(0, 3), (2, 5)])
def test_what_is_computed_from_the_window_is_recorded_as_not_a_number_until_the_first_full_window(excluded, first_full):
    """Nine periods on a window of 4: the windowed observation is not a number in the record until the period that completes the first window and counts four samples from then on, the period's own sample is recorded throughout, and the rule that reads it is applied in every period."""
    pytest.importorskip("tvboptim")
    data = _toy(((9, 4, False),))
    _declare(
        data["algorithms"]["tune"], window_fill="in_stage", window_start=_ends(excluded, _TOY_PERIOD) if excluded else None
    )
    algo, counts = _toy_run(data)
    np.testing.assert_array_equal(np.asarray(algo.history.filled), _recorded(9, first_full, 4))
    assert np.isfinite(np.asarray(algo.history.signal)).all()
    assert counts == {"g": 9, "h": 9 - first_full}


@pytest.mark.parametrize(("from_iteration", "selected"), [(0, 5), (7, 7)])
def test_an_iteration_before_the_first_full_window_is_not_eligible_for_selection(from_iteration, selected):
    """A selection on the smallest count of window samples would take the first period, whose window holds the fewest; on the clock it takes the first period it may, the one that completes the first window or `from_iteration` where that is later, and restores the parameters in effect there."""
    pytest.importorskip("tvboptim")
    data = _toy(((9, 4, False),))
    tune = data["algorithms"]["tune"]
    _declare(tune, window_fill="in_stage", window_start=_ends(2, _TOY_PERIOD))
    tune["selection"] = {"criterion": "filled", "mode": "min", "from_iteration": from_iteration}
    algo, counts = _toy_run(data)
    assert int(algo.selected_iteration) == selected
    assert float(algo.selected_value) == 4.0
    assert counts == {"g": selected, "h": selected - 5}


@pytest.mark.parametrize(
    ("length", "handed", "window_start", "first_full"),
    [
        (8, 4, None, 0),
        (2, 4, None, 0),
        (8, 2, None, 1),
        (8, 4, _ends(0, _TOY_PERIOD), 3),
        (8, 4, _ends(1, _TOY_PERIOD), 4),
    ],
    ids=["a_full_window", "a_full_window_for_a_stage_shorter_than_it", "half_a_window", "dropped", "dropped_and_one_excluded"],
)
def test_a_dependency_hands_its_samples_to_the_first_window_unless_window_start_is_declared(
    length, handed, window_start, first_full
):
    """A dependency's window was sampled before the clock started. Without `window_start` its samples count towards the first window of 4, so the stage waits only for the ones it lacks; with `window_start` they are dropped and the window is built from the stage's own samples. A stage shorter than its window is accepted where the dependency hands it a full one."""
    pytest.importorskip("tvboptim")
    data = _toy(((length, 4, False),), handed=handed)
    _declare(data["algorithms"]["tune"], window_fill="in_stage", window_start=window_start)
    algo, counts = _toy_run(data)
    np.testing.assert_array_equal(np.asarray(algo.history.filled), _recorded(length, first_full, 4))
    assert counts == {"g": length, "h": length - first_full}


def test_two_clocked_algorithms_each_count_from_their_own_first_stage():
    """A dependency on the clock hands its window on like any other, and the algorithm after it starts its own clock at zero: two samples in hand of a window of 4, so the windowed rule waits one period of eight. The generated module holds the first-full-window arithmetic once for both."""
    pytest.importorskip("tvboptim")
    data = _toy(((8, 4, False),), handed=2)
    for algorithm in data["algorithms"].values():
        _declare(algorithm, window_fill="in_stage")
    assert SimulationExperiment.from_dict(data).render_code("tvboptim").count("def first_full_window(") == 1
    algo, counts = _toy_run(data)
    np.testing.assert_array_equal(np.asarray(algo.history.filled), _recorded(8, 1, 4))
    assert counts == {"g": 8, "h": 7}


@pytest.mark.parametrize(("reset", "waits"), [(False, 2), (True, 6)], ids=["continuous", "reset"])
def test_a_later_stage_with_a_larger_window_waits_inside_the_stage_for_the_samples_it_lacks(reset, waits):
    """Eight periods on a window of 4, then ten on a window of 7. A continuous second stage inherits four samples and waits for three more, so its window is complete in its third period; a reset one starts empty and waits for all seven. The record an algorithm keeps is its last stage's."""
    pytest.importorskip("tvboptim")
    data = _toy(((8, 4, reset), (10, 7, reset)))
    _declare(data["algorithms"]["tune"], window_fill="in_stage")
    algo, counts = _toy_run(data)
    np.testing.assert_array_equal(np.asarray(algo.history.filled), _recorded(10, waits, 7))
    assert counts == {"g": 18, "h": (8 - 3) + (10 - waits)}


def test_the_scan_of_a_clocked_schedule_compiles_once_across_stages(monkeypatch):
    """Clock and sample count enter the tuning scan as traced values, so two stages of one length with different windows share one compilation where the windowed observation has a streaming form, and that form waits like the recomputed one."""
    pytest.importorskip("tvboptim")
    import jax

    jit, compiles = jax.jit, {}

    def counting_jit(fn=None, /, *args, **kwargs):
        def wrap(traced):
            @functools.wraps(traced)
            def counted(*a, **k):
                compiles[traced.__name__] = compiles.get(traced.__name__, 0) + 1
                return traced(*a, **k)

            return jit(counted, *args, **kwargs)

        return wrap if fn is None else wrap(fn)

    monkeypatch.setattr(jax, "jit", counting_jit)
    data = _toy(((8, 4, False), (8, 7, False)), window_of="fc")
    _declare(data["algorithms"]["tune"], window_fill="in_stage")
    _, counts = _toy_run(data)
    assert counts == {"g": 16, "h": (8 - 3) + (8 - 2)}
    cores = {name: n for name, n in compiles.items() if "tune_tuning_core" in name}
    assert cores and set(cores.values()) == {1}, f"the tuning scan was compiled {cores or sorted(compiles)}"


def test_apply_every_counts_the_periods_of_the_stage_and_not_the_clock():
    """With `apply_every: 3` a stage applies its rules in its third, sixth and ninth period, wherever the clock stands: ten periods and then eight continuous ones apply the free rule 3 + 2 times, where a count on the clock would give 3 + 3, and the windowed rule, which waits out the first of those, 2 + 2 times."""
    pytest.importorskip("tvboptim")
    data = _toy(((10, 4, False), (8, 4, False)))
    _declare(data["algorithms"]["tune"], window_fill="in_stage", apply_every=3)
    _, counts = _toy_run(data)
    assert counts == {"g": 5, "h": 4}


@pytest.fixture
def cohort_root(tmp_path):
    """Two subjects' functional connectomes, the per-subject target that makes the toy fit a cohort of two lanes."""
    for subject in ("01", "02"):
        net = Network.from_matrix(weights=np.eye(3), labels=["A", "B", "C"])
        net.set_matrix("fc", np.full((3, 3), int(subject) / 10.0))
        folder = tmp_path / f"sub-{subject}" / "func"
        folder.mkdir(parents=True)
        net.save(folder / f"sub-{subject}_atlas-Toy_desc-FC_relmat.yaml")
    return tmp_path


@pytest.mark.parametrize(
    ("reset", "update_start", "expected"),
    [
        (False, None, {"g": 12, "h": 3 + 6}),
        (True, None, {"g": 12, "h": 3 + 3}),
        (False, 1400.0, {"g": 6, "h": 6}),
        (True, 1400.0, {"g": 0, "h": 0}),
    ],
    ids=["continuous", "reset", "continuous_with_a_start", "reset_with_a_start"],
)
def test_an_on_device_cohort_hands_the_clock_from_stage_to_stage_in_every_lane(cohort_root, reset, update_start, expected):
    """Two stages of six periods on a window of 4, every subject in its own lane. The windowed rule waits three periods where the trajectory starts and none in a stage that continues it. A start at 1400 ms is the seventh period of a trajectory: a continuous schedule reaches it with its second stage and applies both rules in all six of that stage's periods, and a schedule that resets every stage never reaches it."""
    pytest.importorskip("tvboptim")
    data = _toy(((6, 4, reset), (6, 4, reset)))
    data["dataset"] = {
        "dataset_id": "toy_fc",
        "bids_root": str(cohort_root),
        "batch_mode": "on_device",
        "subjects": [{"subject_id": "01"}, {"subject_id": "02"}],
    }
    data["observations"]["fc_target"] = {
        "source": ["dataset.subject.fc"],
        "query": {"atlas": "Toy", "desc": "FC", "suffix": "relmat"},
        "reconcile": "by_label",
    }
    tune = data["algorithms"]["tune"]
    tune["observations"].append("fc_target")
    _declare(tune, window_fill="in_stage", update_start=update_start)
    exp = SimulationExperiment.from_dict(data)
    assert exp.dataset_on_device()
    cohort = exp.run("tvboptim", mode="algorithms", quiet=True).algorithms["tune"].cohort_state.dynamics
    assert np.asarray(cohort.g).shape[0] == 2
    assert _toy_counts(cohort) == expected


@pytest.mark.parametrize(
    ("stages", "slots", "refusal"),
    [
        (((9, 4, False),), {"window_start": 400.0}, "declares window_start 400.0 without `window_fill: in_stage`"),
        (
            ((3, 4, True),),
            {"window_fill": "in_stage"},
            "stage 1 of 1 runs 3 periods, but its first window of 4 samples is complete only in period 4",
        ),
        (
            ((5, 4, False),),
            {"window_fill": "in_stage", "window_start": 600.0},
            "stage 1 of 1 runs 5 periods, but its first window of 4 samples is complete only in period 6",
        ),
        (
            ((6, 4, False), (2, 9, False)),
            {"window_fill": "in_stage"},
            "stage 2 of 2 runs 2 periods, but its first window of 9 samples is complete only in period 5",
        ),
    ],
    ids=[
        "start_without_the_clock",
        "reset_stage_too_short",
        "excluded_samples_outlast_the_stage",
        "larger_window_outlasts_the_stage",
    ],
)
def test_a_declaration_that_cannot_be_honoured_is_refused_when_the_code_is_generated(stages, slots, refusal):
    data = _toy(stages)
    _declare(data["algorithms"]["tune"], **slots)
    with pytest.raises(ValueError, match=re.escape(refusal)):
        SimulationExperiment.from_dict(data).render_code("tvboptim")


def test_the_clock_is_refused_on_an_algorithm_that_slides_no_window():
    data = _toy(((9, 4, False),))
    tune = data["algorithms"]["tune"]
    tune["hyperparameters"] = [hp for hp in tune["hyperparameters"] if hp["name"] != "window_size"]
    _declare(tune, window_fill="in_stage")
    with pytest.raises(ValueError, match="slides no observation window"):
        SimulationExperiment.from_dict(data).render_code("tvboptim")


def test_a_first_stage_too_short_for_its_window_is_refused_before_it_runs():
    """Whether samples are handed to a first stage is known only when it runs, so one that declares no `window_start` is generated without complaint and measured on the host before the scan, which says what it started with: three periods cannot complete a window of 4 that starts empty."""
    pytest.importorskip("tvboptim")
    data = _toy(((3, 4, False),))
    _declare(data["algorithms"]["tune"], window_fill="in_stage")
    SimulationExperiment.from_dict(data).render_code("tvboptim")
    started_with = "tune runs 3 periods on the stage clock (at period 0 of its trajectory, 0 samples in hand)"
    with pytest.raises(ValueError, match=re.escape(started_with)):
        _toy_run(data)


def test_the_schema_refuses_a_negative_window_start(tmp_path):
    """`tvbo validate schema --class Algorithm` accepts a start of zero and refuses a negative one, before any code is generated."""
    from typer.testing import CliRunner

    from tvbo.cli.validate import app

    files = {}
    for name, start in (("zero", 0.0), ("negative", -200.0)):
        files[name] = tmp_path / f"{name}.yaml"
        files[name].write_text(f"name: tune\nwindow_fill: in_stage\nwindow_start: {start}\n")
    runner = CliRunner()
    assert runner.invoke(app, ["schema", str(files["zero"]), "--class", "Algorithm"]).exit_code == 0
    assert runner.invoke(app, ["schema", str(files["negative"]), "--class", "Algorithm"]).exit_code != 0


def test_a_clock_that_excludes_no_sample_renders_no_comparison_of_the_clock_with_zero():
    """The count of samples a window may hold is reset only where `window_start` excludes periods; without it the scan carries the plain count."""
    data = _toy(((9, 4, False),))
    tune = data["algorithms"]["tune"]
    _declare(tune, window_fill="in_stage")
    code = SimulationExperiment.from_dict(data).render_code("tvboptim")
    assert "_nvalid = jnp.minimum(_nvalid + 1, _win_len)" in code
    assert "_clock >= 0" not in code
    _declare(tune, window_start=_ends(2, _TOY_PERIOD))
    code = SimulationExperiment.from_dict(data).render_code("tvboptim")
    assert "_nvalid = jnp.where(_clock >= 2, jnp.minimum(_nvalid + 1, _win_len), 0)" in code


@pytest.mark.parametrize(
    ("record", "delta"),
    [
        ([np.nan, np.nan, 1.0, 3.0], 2.0),
        ([1.0, np.nan, 3.0], 2.0),
        ([1.0, 2.0, 4.0], 3.0),
        ([[np.nan, 1.0], [1.0, 2.0], [2.0, 4.0]], [1.0, 2.0]),
        ([np.nan, np.nan, 3.0], np.nan),
        ([np.nan, 1.0, np.nan], np.nan),
        ([np.nan, np.nan, np.nan], np.nan),
    ],
    ids=[
        "opens_undefined",
        "undefined_in_between",
        "defined_throughout",
        "records_of_two_entries",
        "one_finite_record",
        "ends_undefined",
        "no_finite_record",
    ],
)
def test_the_convergence_delta_runs_from_the_first_finite_record(record, delta):
    """`<quantity>_delta` is the last record minus the first whose entries are all numbers, so a quantity that is not defined when a stage opens still reports how far it moved. With fewer than two finite records, or a last record that is not one, it is not a number and nothing is raised; `<quantity>_final` stays the last record."""
    convergence = AlgorithmResult(name="tune", history={"q": np.asarray(record)}).convergence
    np.testing.assert_array_equal(np.asarray(convergence.q_delta), delta)
    np.testing.assert_array_equal(np.asarray(convergence.q_final), record[-1])


@pytest.mark.parametrize(("length", "delta"), [(9, 0.0), (4, np.nan)], ids=["six_full_windows", "one_full_window"])
def test_a_clocked_stage_states_a_delta_for_what_it_computes_from_the_window(length, delta):
    """The record of a windowed quantity opens on not-a-number under the clock, and the stage's convergence summary still states its delta from the first full window on: nothing over six full windows that each hold four samples, and not a number where only the last period has one."""
    pytest.importorskip("tvboptim")
    data = _toy(((length, 4, False),))
    _declare(data["algorithms"]["tune"], window_fill="in_stage")
    algo, _ = _toy_run(data)
    assert np.isnan(np.asarray(algo.history.filled)[0])
    np.testing.assert_array_equal(np.asarray(algo.convergence.filled_delta), delta)
    assert float(algo.convergence.filled_final) == 4.0


@pytest.mark.parametrize(
    "declared",
    [{"window_fill": "in_stage"}, {"window_fill": "before_stage"}, {"window_start": 2 * _PERIOD}],
    ids=["in_stage", "before_stage", "window_start"],
)
def test_a_clock_declared_on_an_included_algorithm_is_refused(declared):
    """An included rule runs in the loop of the algorithm that includes it, so only that algorithm can declare the clock."""
    outer = SimpleNamespace(name="outer", window_fill="in_stage", window_start=None)
    inner = SimpleNamespace(**{"name": "inner", "window_fill": None, "window_start": None, **declared})
    with pytest.raises(ValueError, match="includes 'inner'"):
        stage_clock_settings(outer, _PERIOD, True, [inner, outer])


def test_a_recipe_whose_included_algorithm_declares_the_clock_does_not_render():
    """The refusal reaches a recipe: FIC_EIB includes FIC, and FIC spelling out even the default fill is a declaration FIC_EIB's loop cannot honour."""
    data = _recipe(((10, 4, False),), True)
    _declare(data["algorithms"]["fic"], window_fill="before_stage")
    with pytest.raises(ValueError, match="includes 'fic'"):
        _experiment(data).render_code("tvboptim")
