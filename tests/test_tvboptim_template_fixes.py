"""Regressions for four tvboptim codegen defects.

* Generated code whose bytes depended on ``PYTHONHASHSEED``: an algorithm's observation inputs, the ``only=`` sets handed to ``compute_all_observations`` and the derived-observation order were all read out of Python sets.
* A streamed reduction with no time axis (the ``wave`` detector) stamped with time coordinates, because the emitted stamping table asked for the fold's period and not for a time axis.
* A Lyapunov ``segment_time`` that one renderer required and another silently defaulted to 1.0.
* A travelling-wave gate switched on by one study's derived-variable and parameter names, with its threshold hard-coded, in place of a gate the observation declares.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest

from tvbo.templates.tvboptim.utils import (
    analysis_settings,
    emission_time_period,
    emission_times,
    get_observation_dependencies,
    get_transitive_observations_from_algo,
    set_literal,
    streaming_time_periods,
)

# ── 1. generated code is the same bytes under every hash seed ──────────────────────────────


def _include(name, mode="combined"):
    return SimpleNamespace(algorithm=SimpleNamespace(name=name), mode=mode, arguments=None)


def test_transitive_observations_follow_the_declarations():
    """Includes first (recursively, in include order), then the algorithm's own; each name once; nested includes skipped."""
    algorithms = {
        "inner": SimpleNamespace(observations=["c", "a"], includes=None),
        "mid": SimpleNamespace(observations=["b", "a"], includes=[_include("inner")]),
        "solo": SimpleNamespace(observations=["z"], includes=None),
        "outer": SimpleNamespace(observations=["d", "b"], includes=[_include("mid"), _include("solo", mode="nested")]),
    }
    assert get_transitive_observations_from_algo(algorithms["outer"], algorithms) == ["c", "a", "b", "d"]


def test_observation_dependencies_keep_their_declared_order():
    derived = {"fc": SimpleNamespace(source=["bold", "raw", "bold", "mean"])}
    assert get_observation_dependencies("fc", derived, {"bold", "mean", "fc"}) == ["bold", "mean"]


def test_set_literal_is_sorted_and_empty_is_a_set():
    assert set_literal(["b", "a", "b"]) == "{'a', 'b'}"
    assert set_literal([]) == "set()"


_RENDER = textwrap.dedent(
    """
    import copy, os, sys
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    from tvbo import SimulationExperiment
    from tvbo.data.registry import database_dir

    exp = SimulationExperiment.from_file(str(database_dir("SimulationExperiment") / "EI_Tuning_FIC_EIB_Optimization.yaml"))
    for suffix in ("b", "c", "d"):
        clone = copy.deepcopy(exp.observations["fc_target"])
        clone.name = f"fc_target_{suffix}"
        exp.observations[clone.name] = clone
        exp.algorithms["fic_eib"].observations.append(clone.name)
    sys.stdout.write(exp.render_code("tvboptim"))
    """
)


def _render_under(seed):
    env = {**os.environ, "PYTHONHASHSEED": str(seed), "JAX_PLATFORMS": "cpu"}
    out = subprocess.run([sys.executable, "-c", _RENDER], env=env, capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout


def test_generated_code_does_not_depend_on_the_hash_seed():
    """An algorithm handed four network observations renders to the same bytes under three hash seeds."""
    renders = [_render_under(seed) for seed in (0, 1, 2)]
    assert "fc_target_d=fc_target_d" in renders[0]
    assert renders[0] == renders[1] == renders[2]


# ── 2. only a reduction with a time axis is stamped with times ─────────────────────────────


@pytest.mark.parametrize(
    "red, period",
    [
        ({"kind": "wave", "period_steps": 5}, None),
        ({"kind": "recurrence"}, None),
        ({"kind": "comoment"}, None),
        ({"kind": "monitor", "period_steps": 10}, 10),
        ({"kind": "stride", "ds_steps": 4}, 4),
        ({"kind": "convolution", "ds_steps": 5, "tr_stride": 100}, 500),
    ],
)
def test_only_a_time_axis_has_an_emission_period(red, period):
    assert emission_time_period(red) == period
    assert (emission_times(red, 3, 0.5) is None) is (period is None)


def test_the_stamping_table_leaves_a_wave_unstamped():
    obs_list = [
        {"name": "waves", "reduction": {"kind": "wave", "period_steps": 5}},
        {"name": "raw"},
        {"name": "bold", "reduction": {"kind": "convolution", "ds_steps": 5, "tr_stride": 100}},
    ]
    assert streaming_time_periods(obs_list) == {"waves": None, "bold": 500}


_STREAMED = """
label: Streamed time axes
dynamics:
  name: Ramp
  parameters:
    k: {value: 1.0}
  state_variables:
    x: {equation: {rhs: "k + c_in"}, initial_value: 0.0}
  coupling_inputs: {c_in: {}}
network:
  label: Quad
  number_of_nodes: 4
  nodes: [{id: 0, label: A, dynamics: Ramp}, {id: 1, label: B, dynamics: Ramp}, {id: 2, label: C, dynamics: Ramp}, {id: 3, label: D, dynamics: Ramp}]
  edges:
    - {source: 0, target: 1, parameters: {weight: {value: 0.0}}, source_var: x_out, target_var: c_in, directed: true}
integration: {method: euler, step_size: 1.0, duration: 50.0, unit: ms}
observations:
  waves:
    label: Grouped wave metrics
    source: [x]
    period: 5.0
    partition: {gather: grp, over: [A], waves: wave, directed: sig, correlation: val}
    dynamics:
      name: GroupDetector
      parameters:
        grp: {value: [[0, 1], [2, 3]], shape: "(2, 2)"}
        A: {value: [[1.0, 2.0], [0.5, 1.5]], shape: "(2, 2)"}
      derived_variables:
        val: {equation: {rhs: "sum_axis(A * x, 0)"}}
        wave: {equation: {rhs: "val > 0"}}
        sig: {equation: {rhs: "val < 0.5"}}
  total:
    label: Running sum on a 10 ms window
    source: [x]
    period: 10.0
    reduce: streaming
    dynamics:
      name: RunningSum
      system_type: discrete
      state_variables:
        acc:
          initial_value: 0.0
          equation_type: recurrence
          equation: {rhs: "acc + x"}
      derived_variables:
        out:
          record: true
          equation: {rhs: "acc"}
  mean_x:
    label: Mean, streamed
    source: [x]
    aggregation: mean
    reduce: streaming
"""


def test_the_generated_module_stamps_only_the_windowed_monitor():
    """The emitted table and `_stream_axes` give the wave and the running mean no times, and the 10 ms monitor its window ends."""
    pytest.importorskip("tvboptim")
    import jax.numpy as jnp

    from tvbo import SimulationExperiment

    namespace = {"__name__": "streamed_time_axes"}
    code = SimulationExperiment.from_string(_STREAMED).render_code("tvboptim")
    # dont_inherit: this file's future import would turn the module's annotations into strings.
    exec(compile(code, "<streamed>", "exec", dont_inherit=True), namespace)
    assert namespace["_STREAMING_PERIODS"] == {"waves": None, "total": 10, "mean_x": None}
    axes = namespace["_stream_axes"]({"waves": jnp.zeros((2, 3)), "total": jnp.zeros((5, 4)), "mean_x": jnp.zeros(4)})
    assert set(axes) == {"total"}
    np.testing.assert_allclose(axes["total"], [10.0, 20.0, 30.0, 40.0, 50.0])


def test_a_streamed_wave_and_recurrence_carry_no_time_while_a_windowed_monitor_keeps_its_times():
    pytest.importorskip("tvboptim")
    from tvbo import SimulationExperiment

    obs = SimulationExperiment.from_string(_STREAMED).run("tvboptim").observations
    assert obs["waves"].dims == ("group", "metric") and "time" not in obs["waves"].coords
    assert "time" not in obs["mean_x"].dims and "time" not in obs["mean_x"].coords
    np.testing.assert_allclose(obs["total"].coords["time"].values, (np.arange(obs["total"].sizes["time"]) + 1) * 10.0)


# ── 3. a Lyapunov segment is declared, never defaulted ─────────────────────────────────────


def _param(value):
    return SimpleNamespace(value=value)


def test_lyapunov_settings_resolve_names_aliases_and_defaults():
    an = SimpleNamespace(type="lyapunov", parameters={"segment_time": _param(250), "n": _param(7)})
    assert analysis_settings(an, "lyap") == {"segment_time": 250.0, "n_steps": 7, "n_exponents": 1}


def test_lyapunov_without_segment_time_names_the_analysis():
    an = SimpleNamespace(type="lyapunov", parameters={"n_steps": _param(5)})
    with pytest.raises(ValueError, match=r"'lambda_max'.*segment_time"):
        analysis_settings(an, "lambda_max")


def test_other_analysis_types_keep_their_defaults():
    fd = SimpleNamespace(type="finite_difference", parameters={"seeds": _param(4)})
    assert analysis_settings(fd, "fd") == {"delta": 0.3, "seeds": 4, "seed_base": 0, "stat": "mean"}
    psd = SimpleNamespace(type="psd", parameters={"sigma": _param(0.1)})
    assert analysis_settings(psd, "psd") == {"sigma": 0.1}


def test_every_lyapunov_renderer_rejects_a_missing_segment_time():
    """The analysis renderer and the warm-start restart read the same settings, so neither measures on an undeclared segment."""
    from tvbo import SimulationExperiment
    from tvbo.data.registry import database_dir

    exp = SimulationExperiment.from_file(str(database_dir("SimulationExperiment") / "TBPTT_JansenRit_FC_Optimization.yaml"))
    del exp.observations["lyapunov"].analysis.parameters["segment_time"]
    with pytest.raises(ValueError, match=r"'lyapunov'.*segment_time"):
        exp.render_code("tvboptim")


# ── 4. the propagation gate is declared on the partition ───────────────────────────────────


def _gated_observation(gate=None, period=5.0):
    from tvbo.datamodel.schema import DerivedVariable, Dynamics, Equation, Observation, Parameter, Partition

    grp = [[0, 1, 2], [3, 4, 5]]
    dvs = {
        "val": "sum_axis(theta, 0)",
        "wave": "val > -100.0",
        "sig": "val < 0.5",
        "d0": "cos(theta)",
        "d1": "sin(theta)",
    }
    return Observation(
        name="waves",
        source=["theta"],
        period=period,
        partition=Partition(gather="grp", waves="wave", directed="sig", correlation="val", propagation=gate),
        dynamics=Dynamics(
            name="detector",
            parameters={
                "grp": Parameter(name="grp", value=grp, shape="(2, 3)"),
                "elements": Parameter(name="elements", value=[[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]], shape="(2, 3)"),
            },
            derived_variables={n: DerivedVariable(name=n, equation=Equation(rhs=rhs)) for n, rhs in dvs.items()},
        ),
    )


def _gate(**overrides):
    from tvbo.datamodel.schema import PropagationGate

    return PropagationGate(**{"direction": ["d0", "d1"], "mask": "elements", "min_dispersion": 0.25, **overrides})


def _experiment():
    return SimpleNamespace(integration=SimpleNamespace(step_size=1.0), _source_file=None)


def _reducer(red):
    from .reducer_harness import OBS_TEMPLATE, reducer_namespace

    src = OBS_TEMPLATE.get_def("render_reduction").render(red=red, name="waves", s_idx=0, dt=1.0)
    ns = reducer_namespace()
    exec(compile(src, "<wave-reducer>", "exec"), ns)
    return ns["_reduction_waves"], src


def _fold(red, data):
    factory, _ = _reducer(red)
    init, update, finalize = factory(s_var=0, dt=1.0, skip=0)
    return finalize(update(init(data[0], data.shape[0]), data))


def _phases(T=200):
    """Group 0 holds a fixed phase pattern (a standing field); group 1's phases rotate at different rates (its direction moves)."""
    t = np.arange(T)[:, None]
    standing = np.broadcast_to(np.array([0.3, 1.1, 2.0]), (T, 3))
    moving = np.array([0.05, 0.11, 0.17]) * t
    return np.concatenate([standing, moving], axis=1)[:, None, :]


def test_an_undeclared_gate_emits_the_plain_reducer():
    from tvbo.templates.tvboptim.utils import resolve_reduction

    red = resolve_reduction(_gated_observation(), _experiment())
    assert red["propagation"] is None
    _, src = _reducer(red)
    assert "_dir" not in src and "_dd" not in src


def test_a_declared_gate_zeroes_only_the_standing_group():
    import jax

    jax.config.update("jax_enable_x64", True)
    from tvbo.templates.tvboptim.utils import resolve_reduction

    data = _phases()
    plain = _fold(resolve_reduction(_gated_observation(), _experiment()), data)
    red = resolve_reduction(_gated_observation(_gate()), _experiment())
    assert red["propagation"] == {"direction": ["d0", "d1"], "mask": "elements", "min_dispersion": 0.25}
    gated = _fold(red, data)

    theta = data[4::5, 0, :]
    for g, cols in enumerate(([0, 1, 2], [3, 4, 5])):
        mean_dir = np.stack([np.cos(theta[:, cols]), np.sin(theta[:, cols])]).mean(axis=1)
        weights = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])[g]
        dispersion = np.sum(weights * (1.0 - np.linalg.norm(mean_dir, axis=0))) / weights.sum()
        if dispersion < 0.25:
            assert gated[g, 0] == 0.0 and np.isnan(gated[g, 1]) and np.isnan(gated[g, 2])
        else:
            np.testing.assert_array_equal(gated[g], plain[g])
    assert gated[0, 0] == 0.0 and plain[0, 0] > 0.0  # the standing group is the one the gate removes
    assert gated[1, 0] == plain[1, 0] > 0.0


@pytest.mark.parametrize(
    "overrides, message",
    [({"direction": ["d0", "ghost"]}, "ghost"), ({"mask": "nowhere"}, "nowhere")],
)
def test_a_gate_must_name_what_the_observer_defines(overrides, message):
    from tvbo.templates.tvboptim.utils import resolve_reduction

    with pytest.raises(ValueError, match=message):
        resolve_reduction(_gated_observation(_gate(**overrides)), _experiment())
