"""An ``Observation.dynamics`` observer that declares a ``period`` is a streaming monitor.

The schema has always said an observer's output is "read at the final step for a reduction
... or, when ``period`` is set, sub-sampled over time for a monitor". Only the first half
was implemented: every observer collapsed time into one value per node, so a hemodynamic forward model — an ODE whose whole point is the time course it produces — could not be declared as one. These tests pin the second half:

* the resolver reads ``period`` against the integration grid and tags the reduction
  ``monitor`` (axes ``time x node``), leaving a period-less observer folding as before;
* a curated observer writes its input as the canonical ``x`` and is bound to whatever the
  study's ``source`` names, so an observation model is authored once, not per model;
* an observer state's declared ``initial_value`` is honoured — hemodynamics rest at 1.0,
  not at an accumulator's 0.0;
* the emitted reducer equals tvboptim's ``BalloonWindkesselBold`` monitor, under any
  period-aligned block decomposition, while holding O(nodes) instead of O(time x nodes).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tvbo.classes.observation import Observation as CuratedObservation
from tvbo.data.types import Bunch
from tvbo.datamodel.schema import (
    DerivedParameter,
    DerivedVariable,
    Dynamics,
    Equation,
    Integrator,
    Observation,
    StateVariable,
)
from tvbo.templates.tvboptim.utils import (
    _assert_transient_on_sample_grid,
    assert_measured_window_is_stable,
    observation_dims,
    reduction_dims,
    resolve_reduction,
    streaming_post_eval_plan,
)

from .reducer_harness import OBS_TEMPLATE, reducer_namespace


class _Exp:
    """Minimal experiment stub carrying the integration grid the resolver reads."""

    def __init__(self, step_size=1.0):
        self.integration = Integrator(step_size=step_size) if step_size else None
        self.functions = {}
        self._source_file = None


def _observer(source="x", period=None, states=None, dvs=None, output=None, **obs_kw):
    """An observation whose value is a recurrence over its source."""
    states = states or {"acc": ("acc + x", None)}
    dvs = dvs or {"out": ("acc / count", True)}
    return Observation(
        name="obs",
        source=[source],
        period=period,
        dynamics=Dynamics(
            name="observer",
            state_variables={
                n: StateVariable(
                    name=n,
                    equation=Equation(rhs=rhs),
                    equation_type="recurrence",
                    **({} if init is None else {"initial_value": init}),
                )
                for n, (rhs, init) in states.items()
            },
            derived_variables={
                n: DerivedVariable(name=n, equation=Equation(rhs=rhs), record=rec) for n, (rhs, rec) in dvs.items()
            },
            **({"output": output} if output else {}),
        ),
        **obs_kw,
    )


# ── Resolver: period makes the observer a monitor ───────────────────────────────────────


def test_period_and_step_size_give_the_emission_stride():
    red = resolve_reduction(_observer(period=720.0), _Exp(step_size=1.0))
    assert red["kind"] == "monitor"
    assert red["period_steps"] == 720
    assert reduction_dims(red) == ("time", "node")


def test_an_observer_without_a_period_still_folds_to_one_value_per_node():
    red = resolve_reduction(_observer(), _Exp())
    assert red["kind"] == "recurrence"
    assert red["period_steps"] is None
    assert reduction_dims(red) == ("node",)


def test_a_period_on_a_finer_grid_scales_with_the_step_size():
    red = resolve_reduction(_observer(period=720.0), _Exp(step_size=0.5))
    assert red["period_steps"] == 1440


def test_a_period_without_an_integration_grid_raises_rather_than_collapsing():
    """Silently ignoring the period would return a scalar where a time series was declared."""
    with pytest.raises(ValueError, match="no integration step_size"):
        resolve_reduction(_observer(period=720.0), _Exp(step_size=None))


def test_a_period_shorter_than_the_step_raises():
    with pytest.raises(ValueError, match="shorter than the integration step"):
        resolve_reduction(_observer(period=0.4), _Exp(step_size=1.0))


def test_the_predicate_call_without_an_experiment_stays_truthy():
    # `resolve_reduction(obs)` is used bare as "is this a reduction?"; the period needs the grid, so the bare call answers the boolean without resolving it.
    red = resolve_reduction(_observer(period=720.0))
    assert red is not None and red["period_steps"] is None


def test_post_eval_plan_aligns_the_block_to_the_emission_period():
    obs = _observer(period=720.0, reduce="streaming")
    exp = _Exp()
    exp.observations = {"obs": obs}
    plan = streaming_post_eval_plan(exp)
    assert plan["names"] == ["obs"]
    assert plan["period_in_steps"] == 720
    assert observation_dims(exp) == {"obs": ("time", "node")}


def test_reduce_streaming_on_an_observer_keeps_the_observer_not_the_pipeline_resolver():
    """`reduce: streaming` opts an observer into the post-tuning carry.

    It must not send an observation that already IS a reducer down the pipeline-lifting path, which would reject it for having no HRF kernel.
    """
    red = resolve_reduction(_observer(period=720.0, reduce="streaming"), _Exp())
    assert red["kind"] == "monitor"


# ── The canonical observer input ────────────────────────────────────────────────────────


def test_a_curated_observer_binds_its_canonical_input_to_the_study_source():
    """An observation model is authored once and pointed at any model's state variable.

    The observer writes `x`; the study says `source: [r]`. Without the binding a curated model would only work for a model whose state happens to share the author's name.
    """
    red = resolve_reduction(_observer(source="r", period=720.0), _Exp())
    assert red["source"] == "r"
    assert {str(s) for s in red["states"][0]["update"].free_symbols} == {"acc", "r"}


def test_an_observer_that_declares_its_own_x_means_that_x():
    """The alias never shadows a symbol the observer defines itself."""
    red = resolve_reduction(
        _observer(source="r", states={"x": ("x + r", None)}, dvs={"out": ("x / count", True)}),
        _Exp(),
    )
    assert {str(s) for s in red["states"][0]["update"].free_symbols} == {"x", "r"}


# ── The readout and the initial state ───────────────────────────────────────────────────


def test_the_recorded_derived_variable_is_the_readout():
    red = resolve_reduction(_observer(), _Exp())
    assert red["output_name"] == "out"


def test_two_recorded_derived_variables_are_ambiguous_and_raise():
    obs = _observer(dvs={"a": ("acc", True), "b": ("acc * 2", True)})
    with pytest.raises(ValueError, match="exactly one derived variable"):
        resolve_reduction(obs, _Exp())


def test_a_declared_initial_value_is_honoured():
    red = resolve_reduction(_observer(states={"acc": ("acc + x", 1.0)}), _Exp())
    assert red["states"][0]["init"] == 1.0


def test_an_undeclared_initial_value_keeps_the_reduction_identity():
    """A running sum must start at 0, not at the schema's model-state default."""
    red = resolve_reduction(_observer(), _Exp())
    assert red["states"][0]["init"] == 0.0


# ── The emitted reducer vs tvboptim's Balloon-Windkessel monitor ─────────────────────────


def _bw_reducer(period_steps, dt=1.0, n_var=1):
    """Render the curated BOLD_Balloon observer as a reducer and exec it."""
    obs = CuratedObservation.from_db("BOLD_Balloon")
    obs.source = ["r"]
    exp = _Exp(step_size=dt)
    red = resolve_reduction(obs, exp)
    red["period_steps"] = period_steps
    src = OBS_TEMPLATE.get_def("render_recurrence_reduction").render(
        red=red,
        name="bold",
        s_idx=0,
        dt=dt,
    )
    ns = reducer_namespace()
    exec(compile(src, "<reducer>", "exec"), ns)
    return ns["_reduction_bold"]


def _tvboptim_bw(rates, period_ms, dt):
    """The reference: tvboptim's post-hoc BalloonWindkesselBold over the whole trajectory."""
    from tvboptim.experimental.network_dynamics.result import NativeSolution
    from tvboptim.observations.tvb_monitors import BalloonWindkesselBold

    sol = NativeSolution(
        ts=jnp.arange(rates.shape[0]) * dt,
        ys=rates[:, None, :],
        dt=dt,
    )
    return np.asarray(BalloonWindkesselBold(period=period_ms, dt_bw=dt)(sol).ys[:, 0, :])


@pytest.mark.parametrize("period_steps", [10, 25, 100])
def test_the_reducer_reproduces_the_balloon_windkessel_monitor(period_steps):
    rng = np.random.default_rng(0)
    dt, n_node = 1.0, 6
    n_steps = period_steps * 12
    rates = jnp.asarray(rng.uniform(1.0, 5.0, size=(n_steps, n_node)))

    init, update, finalize = _bw_reducer(period_steps, dt=dt)()
    data = rates[:, None, :]
    got = np.asarray(finalize(update(init(data[0], n_steps), data)))
    expected = _tvboptim_bw(rates, period_ms=period_steps * dt, dt=dt)

    # Measured max|diff| is 0.0 — the declared recurrence is the same sequence of operations the monitor performs. Asserted to f64 rounding rather than exactly, because the emitted `s / tau_s` and the monitor's precomputed `(1/tau_s) * s` coincide only through XLA's reciprocal rewrite.
    assert got.shape == expected.shape
    assert np.allclose(got, expected, rtol=1e-12, atol=1e-14), np.abs(got - expected).max()


@pytest.mark.parametrize("blocks", [1, 3, 12])
def test_the_reducer_is_block_decomposition_invariant(blocks):
    """The value must not depend on how the integrator chops the run into blocks."""
    rng = np.random.default_rng(1)
    period_steps, n_node = 25, 4
    n_steps = period_steps * 12
    data = jnp.asarray(rng.uniform(1.0, 5.0, size=(n_steps, 1, n_node)))

    whole = _bw_reducer(period_steps)()
    ref = np.asarray(whole[2](whole[1](whole[0](data[0], n_steps), data)))

    init, update, finalize = _bw_reducer(period_steps)()
    acc = init(data[0], n_steps)
    block = n_steps // blocks
    for start in range(0, n_steps, block):
        acc = update(acc, data[start : start + block])
    assert np.array_equal(np.asarray(finalize(acc)), ref)


def test_a_short_tail_block_advances_the_observer_without_emitting():
    """Tvboptim's blocked scan ends with one partial block; it holds no whole sample."""
    rng = np.random.default_rng(2)
    period_steps, n_node = 25, 3
    n_steps = period_steps * 8 + 7
    data = jnp.asarray(rng.uniform(1.0, 5.0, size=(n_steps, 1, n_node)))

    init, update, finalize = _bw_reducer(period_steps)()
    acc = init(data[0], n_steps)
    for b in range(n_steps // period_steps):
        acc = update(acc, data[b * period_steps : (b + 1) * period_steps])
    acc = update(acc, data[(n_steps // period_steps) * period_steps :])
    got = np.asarray(finalize(acc))

    assert got.shape == (n_steps // period_steps, n_node)
    assert np.allclose(got, _tvboptim_bw(data[:, 0, :], period_steps * 1.0, 1.0)[: got.shape[0]], rtol=1e-12, atol=1e-14)


def test_a_block_that_is_not_a_whole_number_of_periods_raises():
    """Silently accepting it would shift every later sample off the global grid."""
    period_steps = 25
    data = jnp.zeros((60, 1, 3))
    init, update, _ = _bw_reducer(period_steps)()
    with pytest.raises(ValueError, match="whole number of"):
        update(init(data[0], 60), data)


def test_skip_drops_the_transient_samples():
    """A sweep folds transient and main run into one window and asks for the transient back."""
    rng = np.random.default_rng(3)
    period_steps, n_node = 25, 3
    n_steps = period_steps * 10
    data = jnp.asarray(rng.uniform(1.0, 5.0, size=(n_steps, 1, n_node)))

    every = _bw_reducer(period_steps)()
    full = np.asarray(every[2](every[1](every[0](data[0], n_steps), data)))

    skip = 3 * period_steps
    init, update, finalize = _bw_reducer(period_steps)(skip=skip)
    got = np.asarray(finalize(update(init(data[0], n_steps), data)))

    assert np.array_equal(got, full[3:])


def test_the_reducer_holds_no_trajectory():
    """Peak carry is O(samples + nodes), not O(time x nodes) — the point of streaming."""
    period_steps, n_node = 720, 68
    n_steps = period_steps * 50
    data = jnp.zeros((period_steps, 1, n_node))

    init, update, _ = _bw_reducer(period_steps)()
    acc = update(init(data[0], n_steps), data)
    carried = sum(int(np.asarray(a).size) for a in jax.tree_util.tree_leaves(acc))

    # The 4 hemodynamic states, the output buffer and the counter — nothing else. The BOLD readout is a function of the states alone, so it is not carried through the scan.
    assert carried == 4 * n_node + (n_steps // period_steps) * n_node + 1
    assert carried < n_steps * n_node


def test_a_state_only_readout_is_evaluated_once_per_sample():
    """The readout runs per emitted sample, not per integration step.

    BOLD is a function of the hemodynamic states alone, so evaluating it every step and discarding all but the boundary value is ~`period` times more work than needed. The resolver decides this symbolically: a readout that reads the observed signal or a per-step derived variable cannot be deferred and stays in the scan body.
    """
    obs = CuratedObservation.from_db("BOLD_Balloon")
    obs.source = ["r"]
    assert resolve_reduction(obs, _Exp())["output_per_step"] is False

    # ... whereas one that reads the source itself must stay per-step.
    reads_source = _observer(source="r", period=720.0, states={"acc": ("acc + r", None)}, dvs={"out": ("acc * r", True)})
    assert resolve_reduction(reads_source, _Exp())["output_per_step"] is True


def test_derived_parameters_are_bound_once_outside_the_scan():
    """Observer constants belong in the preamble, not recomputed every step.

    `derived_parameters` is the slot every other backend already uses for this; declaring them as `derived_variables` instead would put them in the per-step body AND force the readout that reads them onto the per-step path.
    """
    obs = CuratedObservation.from_db("BOLD_Balloon")
    obs.source = ["r"]
    red = resolve_reduction(obs, _Exp())
    assert [d["name"] for d in red["derived_constants"]] == ["dt_s", "k1", "k2", "k3"]
    assert red["derived"] == [] or all(d["name"] == "bold_signal" for d in red["derived"])

    src = OBS_TEMPLATE.get_def("render_recurrence_reduction").render(red=red, name="bold", s_idx=0, dt=1.0)
    body = src.split("def _step(")[1]
    assert "k1 = " not in body and "dt_s = " not in body


def test_a_derived_parameter_that_varies_per_step_is_rejected():
    """A constant that reads a state is not a constant; say so instead of emitting it."""
    obs = _observer(period=720.0)
    # multivalued slot: mutate in place, an assignment does not go through the normaliser
    obs.dynamics.derived_parameters["bad"] = DerivedParameter(name="bad", equation=Equation(rhs="acc * 2"))
    with pytest.raises(ValueError, match="which vary per step"):
        resolve_reduction(obs, _Exp())


# ── The settle has to land on the sample grid a scan-anchored consumer emits on ────


def _settled_exp(step_size, transient_time):
    """An experiment stub carrying both halves of the window the resolver checks."""
    exp = _Exp(step_size=step_size)
    exp.integration.transient_time = transient_time
    return exp


@pytest.mark.parametrize(
    "transient_time, aligned",
    [(100.0, True), (0.0, True), (110.0, False), (99.0, False)],
)
def test_a_settle_off_the_monitor_sample_grid_is_refused(transient_time, aligned):
    """A monitor decimates on scan boundaries, so a settle that is not whole samples offsets every reported timestamp.

    It is refused rather than rounded because the offset is a fraction of a period that changes silently whenever `transient_time` does — the reported series would still have the right shape and the wrong times, and the recipe is the only place that can decide which settle it meant.
    """
    obs = _observer(period=25.0)
    exp = _settled_exp(step_size=1.0, transient_time=transient_time)
    if aligned:
        assert resolve_reduction(obs, exp)["period_steps"] == 25
        return
    with pytest.raises(ValueError, match="not a whole number"):
        resolve_reduction(obs, exp)


def test_the_refusal_offers_the_two_settles_that_would_work():
    """A message that only says no leaves the modeller to redo the arithmetic the guard just did."""
    obs = _observer(period=25.0)
    with pytest.raises(ValueError, match=r"nearest are 100 and 125"):
        resolve_reduction(obs, _settled_exp(step_size=1.0, transient_time=110.0))


@pytest.mark.parametrize("kind", ["monitor", "wave"])
def test_the_refusal_names_the_consumer_that_will_emit(kind):
    """The same `period` resolves to a monitor or, under a `partition`, to a wave reducer; both decimate on scan boundaries and both are guarded here.

    Checked against the guard rather than through a resolved observation because building a partitioned one needs a produced gather table, and what is under test is which consumer the message names — telling a modeller their wave detector is misaligned as a "monitor" sends them looking through an observation they do not have.
    """
    with pytest.raises(ValueError, match=f"{kind} samples"):
        _assert_transient_on_sample_grid(_settled_exp(step_size=1.0, transient_time=110.0), "obs", 25, kind)


def test_an_aligned_settle_passes_whatever_the_consumer():
    """The guard is silent where the settle spans whole samples, so it cannot be what refuses a valid recipe."""
    for kind in ("monitor", "wave"):
        _assert_transient_on_sample_grid(_settled_exp(step_size=1.0, transient_time=100.0), "obs", 25, kind)


def test_an_optimization_that_re_times_the_scan_is_refused():
    """A monitor tells settle from measurement by subtraction, so the window it subtracts against has to be the one that runs.

    An `optimization.integration` override re-prepares the scan with its own duration and step size; a longer or finer-stepped tuning window then carries an excess that is not settle, and the monitor would cut real measurement off its front. Shape is the only signal a monitor has, so the two cases cannot be told apart from inside it — which is why this is refused at render rather than handled at runtime.
    """
    exp = _Exp(step_size=1.0)
    exp.integration.duration = 1000.0
    exp.optimization = [Bunch(integration=Integrator(step_size=1.0, duration=4000.0))]
    with pytest.raises(ValueError, match="overrides duration"):
        assert_measured_window_is_stable(exp, "obs")


def test_an_optimization_on_the_same_grid_is_allowed():
    """An override that restates the grid, or none at all, leaves the subtraction sound and must not be refused."""
    exp = _Exp(step_size=1.0)
    exp.integration.duration = 1000.0
    for override in (None, Integrator(step_size=1.0, duration=1000.0)):
        exp.optimization = [Bunch(integration=override)]
        assert_measured_window_is_stable(exp, "obs")  # returns None; the assertion is that it does not raise


def test_an_override_that_names_only_a_method_still_re_times_the_scan():
    """The schema fills every slot an override does not name, so `{method: Heun}` carries a default step size — and the emitted code prepares with it.

    Refusing it is not pedantry: `opt_dt` reads that defaulted value, so the tuning scan really does run on a different grid than the one the monitor subtracts against. The message says where the number came from, because nobody who wrote `{method: Heun}` is looking for a step size.
    """
    exp = _Exp(step_size=1.0)
    exp.integration.duration = 1000.0
    exp.optimization = [Bunch(integration=Integrator(method="Heun"))]
    with pytest.raises(ValueError, match="does not name"):
        assert_measured_window_is_stable(exp, "obs")
