"""Grouped wave-metrics reducer (``kind: 'wave'``) — per-group scalar output.

The cortical wave detector collapses BOTH time and the node axis into per-hemisphere scalars (``proportion_waves``, ``proportion_directed``, ``rho``), so its output is keyed by (group, metric) — NOT by node, and cannot come from ``template.shape[-1]``. The bespoke ``render_wave_reduction`` emitter carries a monitor-style ``(n_ds, n_groups)`` buffer per named per-step output (``corr`` / ``wave_present`` / ``sig_corr``) and reduces them at finalize to the three metrics via exact masked statistics (``nanmedian`` over wave-present samples — no binning).
The heavy per-step math is the SAME declarative DV chain the other observers use; only the outer carry is bespoke. These tests pin, with a synthetic primitive-only per-step body:

* the emitted reducer's three metrics are byte-identical (to f64) to a numpy reference of
  Koller's finalize (``cortical_wave_metrics``: nw/T, sum(sig & wave)/nw, median(corr[wave]));
* the output is shaped ``(n_groups, 3)`` — per-group, not per-node;
* a group that never has a wave (nw == 0) yields NaN for proportion_directed and rho, exactly
  as the CPU detector's ``if nw else np.nan`` guards;
* the fold is block-decomposition invariant across period-aligned blocks (the grid path).

The per-step body here is synthetic (group-mean of the phase); the real detector plugs the gradient → angular-similarity → surrogate → HHD → correlation chain into the same output slots.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mako.template import Template

jax.config.update("jax_enable_x64", True)

from tvbo.parse.expression import parse_eq

_TEMPLATE = "tvbo/templates/tvboptim/tvbo-tvboptim-observation.py.mako"


def _wave_red(P, period=5):
    """A `kind: 'wave'` reduction whose per-group per-step outputs are a synthetic function of the whole-brain phase: grp_mean = mean of theta over each group's member nodes."""
    G, n = P.shape
    params = ["theta", "P", "grp_mean"]
    return {
        "kind": "wave",
        "source": "theta",
        "n_groups": int(G),
        "period_steps": int(period),
        "parameters": {"P": {"value": P.tolist(), "shape": f"({G}, {n})"}},
        "derived": [
            {"name": "grp_mean", "expr": parse_eq("matmul(P, theta) / sum_axis(P, 1)", parameters=params)},
            {"name": "wave", "expr": parse_eq("grp_mean > 0", parameters=params)},
            {"name": "sig", "expr": parse_eq("grp_mean < 0.5", parameters=params)},
        ],
        "corr": "grp_mean",
        "wave_present": "wave",
        "sig_corr": "sig",
        "functions": {},
    }


def _factory(red, name="wave"):
    src = Template(filename=_TEMPLATE).get_def("render_reduction").render(red=red, name=name, s_idx=0, dt=1.0)
    ns = {"jnp": jnp, "jax": jax}
    exec(compile(src, "<wave-reducer>", "exec"), ns)
    return ns[f"_reduction_{name}"], src


def _reference(block, P, period=5, skip=0):
    """Koller's finalize on the same synthetic per-step body: proportion_waves = nw/T, proportion_directed = sum(sig & wave)/nw, rho = median(corr[wave]) — per group."""
    theta_ds = block[period - 1 :: period, 0, :]  # (T_ds, n) downsampled
    grp = theta_ds @ P.T / P.sum(1)  # (T_ds, G) group means
    grp = grp[skip // period :]
    corr, wave, sig = grp, grp > 0, grp < 0.5
    nw = wave.sum(0)  # (G,)
    pw = nw / wave.shape[0]
    with np.errstate(invalid="ignore"):
        pd = np.where(nw > 0, (sig & wave).sum(0) / nw, np.nan)
        rho = np.array([np.median(corr[wave[:, g], g]) if nw[g] else np.nan for g in range(P.shape[0])])
    return np.stack([pw, pd, rho], axis=-1)  # (G, 3)


def _partition(n, groups):
    """A hard node→group membership mask (G, n), each node in exactly one group."""
    G = len(set(groups))
    P = np.zeros((G, n))
    for i, g in enumerate(groups):
        P[g, i] = 1.0
    return P


def _trajectory(seed, T=200, n=8, offset=0.0):
    rng = np.random.default_rng(seed)
    return offset + rng.standard_normal((T, 1, n))


# ── byte-identity vs the numpy finalize ─────────────────────────────────────────────────────


def test_wave_metrics_match_numpy_finalize():
    P = _partition(8, [0, 0, 0, 0, 1, 1, 1, 1])  # two hemispheres of 4 nodes
    data = _trajectory(seed=1, T=205)
    factory, _ = _factory(_wave_red(P, period=5))
    init, update, finalize = factory(s_var=0, dt=1.0, skip=0)
    got = finalize(update(init(data[0], data.shape[0]), data))

    ref = _reference(data, P, period=5, skip=0)
    assert got.shape == ref.shape == (2, 3)  # (group, metric) — NOT per-node
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-9, atol=1e-12)


def test_group_without_waves_is_nan():
    """A group whose phase is always negative never has a wave (nw == 0); its proportion_directed and rho must be NaN, matching the CPU `if nw else np.nan`."""
    P = _partition(6, [0, 0, 0, 1, 1, 1])
    data = _trajectory(seed=2, T=155, n=6)
    data[:, 0, 3:] = -np.abs(data[:, 0, 3:]) - 1.0  # group 1 strictly negative → no wave
    factory, _ = _factory(_wave_red(P, period=5))
    init, update, finalize = factory(s_var=0, dt=1.0, skip=0)
    got = np.asarray(finalize(update(init(data[0], data.shape[0]), data)))

    assert got[1, 0] == 0.0  # proportion_waves = 0
    assert np.isnan(got[1, 1]) and np.isnan(got[1, 2])  # directed / rho undefined
    ref = _reference(data, P, period=5, skip=0)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("block_size", [10, 25, 50])
def test_wave_reducer_is_block_decomposition_invariant(block_size):
    """The grid path folds period-aligned blocks; per-sample independence makes the buffered metrics bit-exact across any (period-aligned) block boundary."""
    P = _partition(8, [0, 0, 0, 0, 1, 1, 1, 1])
    data = _trajectory(seed=3, T=200)
    factory, _ = _factory(_wave_red(P, period=5))
    init, update, finalize = factory(s_var=0, dt=1.0, skip=0)

    single = finalize(update(init(data[0], data.shape[0]), data))
    acc = init(data[0], data.shape[0])
    for s in range(0, data.shape[0], block_size):  # block_size is a multiple of period
        acc = update(acc, data[s : s + block_size])
    assert float(jnp.nanmax(jnp.abs(finalize(acc) - single))) == 0.0


def test_transient_skip_drops_leading_samples():
    """`skip` (integration steps) drops the leading downsampled samples at finalize, so a swept cell reduces the same post-transient window as the same experiment run alone."""
    P = _partition(8, [0, 0, 0, 0, 1, 1, 1, 1])
    data = _trajectory(seed=4, T=200)
    factory, _ = _factory(_wave_red(P, period=5))
    init, update, finalize = factory(s_var=0, dt=1.0, skip=25)  # drop 25 steps = 5 samples
    got = finalize(update(init(data[0], data.shape[0]), data))
    ref = _reference(data, P, period=5, skip=25)
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-9, atol=1e-12, equal_nan=True)


def _grouped_wave_red(grp_verts, A, period=5):
    """A grouped `wave` reduction: the per-timestep body is written for a SINGLE group (val = sum(A * theta_g)) and vmapped over the partition axis. `grp_verts` (G, nvg) selects each group's vertices from the whole-brain source; `A` (G, nvg) is a group-indexed operator."""
    G, nvg = grp_verts.shape
    params = ["A", "theta", "val"]
    return {
        "kind": "wave",
        "source": "theta",
        "n_groups": int(G),
        "period_steps": int(period),
        "parameters": {
            "grp_verts": {"value": grp_verts.tolist(), "shape": f"({G}, {nvg})"},
            "A": {"value": A.tolist(), "shape": f"({G}, {nvg})"},
        },
        "group_vmap": {"gather": "grp_verts", "over": ["A"]},
        "derived": [
            {"name": "val", "expr": parse_eq("sum_axis(A * theta, 0)", parameters=params)},
            {"name": "wave", "expr": parse_eq("val > 0", parameters=params)},
            {"name": "sig", "expr": parse_eq("val < 0.5", parameters=params)},
        ],
        "corr": "val",
        "wave_present": "wave",
        "sig_corr": "sig",
        "functions": {},
    }


def _grouped_reference(block, grp_verts, A, period=5):
    """Numpy: loop the single-group body over the partition — what the vmap must equal."""
    theta_ds = block[period - 1 :: period, 0, :]  # (T_ds, n_total)
    G = grp_verts.shape[0]
    val = np.stack([theta_ds[:, grp_verts[g]] @ A[g] for g in range(G)], axis=1)  # (T_ds, G)
    corr, wave, sig = val, val > 0, val < 0.5
    nw = wave.sum(0)
    pw = nw / wave.shape[0]
    with np.errstate(invalid="ignore"):
        pd = np.where(nw > 0, (sig & wave).sum(0) / nw, np.nan)
        rho = np.array([np.median(corr[wave[:, g], g]) if nw[g] else np.nan for g in range(G)])
    return np.stack([pw, pd, rho], axis=-1)


def test_group_vmap_matches_numpy_group_loop():
    """The elegant form: ONE single-group body, vmapped over the partition axis, is byte-identical to looping the body over groups. This is what makes the detector general to any partition (2 hemispheres, or N parcels) with the surrogate staying a clean per-vertex test inside vmap."""
    rng = np.random.default_rng(7)
    grp_verts = np.array([[0, 1, 2, 3], [5, 6, 7, 8]])  # 2 groups of 4, from 10 nodes
    A = rng.standard_normal((2, 4))
    data = 0.3 + rng.standard_normal((200, 1, 10))
    factory, src = _factory(_grouped_wave_red(grp_verts, A, period=5))
    init, update, finalize = factory(s_var=0, dt=1.0, skip=0)
    got = finalize(update(init(data[0], data.shape[0]), data))

    ref = _grouped_reference(data, grp_verts, A, period=5)
    assert got.shape == ref.shape == (2, 3)
    assert "jax.vmap(_detect" in src  # single-group body, vmapped over groups
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-9, atol=1e-12, equal_nan=True)


## ── resolver: a declarative Observation with a `partition` emits kind:'wave' ───────────────

from types import SimpleNamespace

from tvbo.datamodel.schema import (
    DerivedVariable,
    Dynamics,
    Equation,
    Observation,
    Parameter,
    Partition,
)
from tvbo.templates.tvboptim.utils import resolve_reduction, streaming_post_eval_plan


def _wave_observation(grp_verts, A, period=5.0):
    """A declarative grouped wave Observation: observer DVs produce the per-group outputs; the `partition` names the gather table, the group-indexed operator, and the metric-role DVs."""
    G, nvg = grp_verts.shape
    return Observation(
        name="wave",
        source=["theta"],
        period=period,
        partition=Partition(gather="grp_verts", over=["A"], waves="wave", directed="sig", correlation="val"),
        dynamics=Dynamics(
            name="obs",
            parameters={
                "grp_verts": Parameter(name="grp_verts", value=grp_verts.tolist(), shape=f"({G}, {nvg})"),
                "A": Parameter(name="A", value=A.tolist(), shape=f"({G}, {nvg})"),
            },
            derived_variables={
                "val": DerivedVariable(name="val", equation=Equation(rhs="sum_axis(A * theta, 0)")),
                "wave": DerivedVariable(name="wave", equation=Equation(rhs="val > 0")),
                "sig": DerivedVariable(name="sig", equation=Equation(rhs="val < 0.5")),
            },
        ),
    )


def _stub_experiment(dt=1.0):
    return SimpleNamespace(integration=SimpleNamespace(step_size=dt), _source_file=None)


def test_partition_resolves_to_wave_kind():
    grp_verts = np.array([[0, 1, 2, 3], [5, 6, 7, 8]])
    A = np.linspace(0.5, 2.0, 8).reshape(2, 4)
    red = resolve_reduction(_wave_observation(grp_verts, A, period=5.0), _stub_experiment(dt=1.0))
    assert red["kind"] == "wave"
    assert red["n_groups"] == 2
    assert red["period_steps"] == 5
    assert red["group_vmap"] == {"gather": "grp_verts", "over": ["A"]}
    assert (red["corr"], red["wave_present"], red["sig_corr"]) == ("val", "wave", "sig")


def test_partition_wave_obs_streams_in_the_base_run():
    """A `partition` (wave) observation must be classed streaming by streaming_post_eval_plan, so the base/post-eval run folds it in-carry per block instead of materialising the whole trajectory and vmapping every frame (which OOM'd Koller exp_41 at 286 GiB in STEP 1).

    `wave` must also be a slotted kind so the block aligns to its downsample period, not 1000.
    """
    grp_verts = np.array([[0, 1, 2, 3], [5, 6, 7, 8]])
    A = np.linspace(0.5, 2.0, 8).reshape(2, 4)
    exp = SimpleNamespace(
        integration=SimpleNamespace(step_size=1.0),
        _source_file=None,
        observations={"wave": _wave_observation(grp_verts, A, period=5.0)},
    )
    plan = streaming_post_eval_plan(exp)
    assert "wave" in plan["names"]  # streamed, not materialised → no full-trajectory vmap
    assert plan["period_in_steps"] == 5  # slotted → block aligns to the wave period, not the 1000 default


def test_partition_without_period_is_rejected():
    grp_verts = np.array([[0, 1], [2, 3]])
    obs = _wave_observation(grp_verts, np.ones((2, 2)), period=5.0)
    obs.period = None
    with pytest.raises(ValueError, match="no `period`"):
        resolve_reduction(obs, _stub_experiment(dt=1.0))


def test_a_period_is_required_even_without_an_experiment_to_resolve_it_against():
    """Whether a `period` was DECLARED does not depend on having an experiment."""
    obs = _wave_observation(np.array([[0, 1], [2, 3]]), np.ones((2, 2)), period=5.0)
    obs.period = None
    with pytest.raises(ValueError, match="no `period`"):
        resolve_reduction(obs)


@pytest.mark.parametrize("role", ["waves", "directed", "correlation"])
def test_every_metric_role_must_name_a_derived_variable(role):
    """An unnamed role reached the renderer as the string 'None' and emitted `None * 1.0`."""
    obs = _wave_observation(np.array([[0, 1], [2, 3]]), np.ones((2, 2)))
    setattr(obs.partition, role, None)
    with pytest.raises(ValueError, match=role):
        resolve_reduction(obs, _stub_experiment(dt=1.0))


def test_the_wave_kind_declares_its_axes():
    """(n_groups, metric) is named at codegen, never guessed from the result's shape."""
    from tvbo.templates.tvboptim.utils import reduction_dims

    red = resolve_reduction(
        _wave_observation(np.array([[0, 1, 2, 3], [5, 6, 7, 8]]), np.linspace(0.5, 2.0, 8).reshape(2, 4), period=5.0),
        _stub_experiment(dt=1.0),
    )
    assert reduction_dims(red) == ("group", "metric")


def test_partition_gather_must_be_a_parameter():
    grp_verts = np.array([[0, 1], [2, 3]])
    obs = _wave_observation(grp_verts, np.ones((2, 2)))
    obs.partition.gather = "ghost"
    with pytest.raises(ValueError, match="gathers on 'ghost'"):
        resolve_reduction(obs, _stub_experiment(dt=1.0))


def test_declarative_wave_observation_end_to_end_matches_numpy():
    """Full path: a declarative Observation with a `partition` resolves to kind:'wave' and the emitted grouped reducer is byte-identical to the numpy group-loop — closing the loop from spec to the vmapped GPU reducer (review finding #6: the wave kind is now reachable)."""
    rng = np.random.default_rng(11)
    grp_verts = np.array([[0, 1, 2, 3], [5, 6, 7, 8]])
    A = rng.standard_normal((2, 4))
    data = 0.3 + rng.standard_normal((200, 1, 10))
    red = resolve_reduction(_wave_observation(grp_verts, A, period=5.0), _stub_experiment(dt=1.0))
    factory, _ = _factory(red)
    init, update, finalize = factory(s_var=0, dt=1.0, skip=0)
    got = finalize(update(init(data[0], data.shape[0]), data))

    ref = _grouped_reference(data, grp_verts, A, period=5)
    assert got.shape == ref.shape == (2, 3)
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-9, atol=1e-12, equal_nan=True)


def test_output_is_grouped_not_per_node():
    """Regression on the architecture: the wave reducer's output axis is the GROUP count, not the model node count — the reduction that motivated the bespoke kind."""
    P = _partition(12, [0] * 6 + [1] * 6)  # 12 nodes, 2 groups
    data = _trajectory(seed=5, T=100, n=12)
    factory, src = _factory(_wave_red(P, period=5))
    init, update, finalize = factory(s_var=0, dt=1.0, skip=0)
    got = finalize(update(init(data[0], data.shape[0]), data))
    assert got.shape == (2, 3)  # groups, not 12 nodes
    assert "jnp.nanmedian" in src and "template.shape[-1]" not in src
