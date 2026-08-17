"""Permutation-significance surrogate in the streaming reducer (DerivedVariable.surrogate).

A derived variable may declare a ``surrogate``: re-evaluate a named statistic under a fixed ``(n_perm, n)`` permutation table and report the per-element exceedance p-value. The resolver (``resolve_reduction``) interleaves the surrogate p-value DV back into the per-step chain at its declaration position, and the observation emitter renders it as a ``jax.vmap`` fold over the table — the ``(vmap(lambda p: stat(field[p]))(perms) <cmp> stat(field)).mean`` form the ``Surrogate`` schema documents. These tests pin:

* the resolver carries the surrogate payload (statistic / permute / perms / compare /
  family_reduce) and splices the p-value DV into ``derived`` in declaration order, so a downstream DV that consumes it is emitted after it;
* the emitted fold is byte-identical (to f64) to a numpy reference under the SAME fixed
  permutation table — both the symmetric per-element test and the Westfall–Young max-T FWE form (``family_reduce: nanmax``), whose permuted statistic is reduced over vertices to one family-wise null each observed element is tested against (Koller Fig-6 wave detection);
* the fold survives any block decomposition (the grid path feeds blocks, not one trajectory);
* malformed surrogates (unknown permute symbol / undeclared permutation table / bad
  family_reduce) are rejected at resolve time.

This is the reusable core of the jittable GPU wave detector: any permutation null — spatial nulls, FC significance, wave detection — emits through this one path.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mako.template import Template

jax.config.update("jax_enable_x64", True)

from tvbo.datamodel.schema import (
    DerivedVariable,
    Dynamics,
    Equation,
    Observation,
    Parameter,
    StateVariable,
    Surrogate,
)
from tvbo.templates.tvboptim.utils import resolve_reduction

_TEMPLATE = "tvbo/templates/tvboptim/tvbo-tvboptim-observation.py.mako"


def _surrogate_observer(perms, w, *, family_wise=False, direction="greater_equal"):
    """An observer over per-node source ``x`` whose value is the time-mean of a per-node permutation p-value. ``stat = w * x`` is the observed statistic; ``pval`` is its surrogate under the fixed ``perms`` table; the accumulator folds ``pval`` over time."""
    return Observation(
        name="obs",
        source=["x"],
        dynamics=Dynamics(
            name="observer",
            parameters={
                "w": Parameter(name="w", value=[float(v) for v in w]),
                "perms": Parameter(name="perms", value=[[int(j) for j in row] for row in perms]),
            },
            state_variables={
                "acc": StateVariable(name="acc", equation=Equation(rhs="acc + pval"), equation_type="recurrence")
            },
            derived_variables={
                "stat": DerivedVariable(name="stat", equation=Equation(rhs="w * x")),
                "pval": DerivedVariable(
                    name="pval",
                    surrogate=Surrogate(
                        statistic="stat",
                        permute="x",
                        permutations="perms",
                        direction=direction,
                        family_wise=family_wise,
                    ),
                ),
                "out": DerivedVariable(name="out", equation=Equation(rhs="acc / count")),
            },
            output=["out"],
        ),
    )


def _render(red, name="obs"):
    return Template(filename=_TEMPLATE).get_def("render_reduction").render(red=red, name=name, s_idx=0, dt=1.0)


def _factory(red, name="obs"):
    ns = {"jnp": jnp, "jax": jax}
    exec(compile(_render(red, name), "<surrogate-reducer>", "exec"), ns)
    return ns[f"_reduction_{name}"]


# ── numpy reference (the SAME permutation table drives both paths) ──────────────────────────


def _pvalue(x_t, w, perms, *, family=None, cmp="ge"):
    """Per-node exceedance p-value at one timepoint under the fixed table."""
    stat = w * x_t  # (n,) observed
    null = w * x_t[perms]  # (n_perm, n) permuted statistic
    if family is not None:
        null = getattr(np, family)(null, axis=1)[:, None]  # (n_perm, 1) family-wise extremum
        stat = stat[None]  # (1, n)
    comp = (null >= stat) if cmp == "ge" else (null <= stat)
    return comp.mean(axis=0)  # (n,)


def _reference_over_time(traj_col, w, perms, *, family=None, cmp="ge"):
    """Time-mean of the per-step p-value over the accumulated samples. The simple recurrence gate (skip=0, non-inclusive) drops sample 0 and folds t=1..T-1."""
    per_t = [_pvalue(traj_col[t], w, perms, family=family, cmp=cmp) for t in range(1, traj_col.shape[0])]
    return np.mean(per_t, axis=0)


def _trajectory(seed, T=64, n=6):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, 1, n))  # [T, n_states=1, n]


def _perm_table(seed, n_perm, n):
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(n) for _ in range(n_perm)])  # (n_perm, n) int


# ── resolver: payload + declaration-order interleave ────────────────────────────────────────


def test_surrogate_payload_is_resolved():
    red = resolve_reduction(_surrogate_observer(_perm_table(0, 8, 6), np.arange(1.0, 7.0)))
    surr = next(d for d in red["derived"] if d["name"] == "pval")["surrogate"]
    assert surr["statistic"] == "stat"
    assert surr["permute"] == "x"
    assert surr["perms"] == "perms"
    assert surr["compare"] == ">="
    assert surr["family_reduce"] is None
    # the statistic is expanded to a self-contained function of the permuted symbol `x`
    assert "x" in {str(s) for s in surr["expr"].free_symbols}


def test_family_wise_derives_max_t():
    """`family_wise: true` on a positive (greater_equal) test resolves to a nan-aware max-T extremum — the schema stays intent-only; the resolver maps it to the backend reducer."""
    red = resolve_reduction(_surrogate_observer(_perm_table(0, 8, 6), np.ones(6), family_wise=True))
    surr = next(d for d in red["derived"] if d["name"] == "pval")["surrogate"]
    assert surr["family_reduce"] == "nanmax"


def test_less_equal_direction():
    red = resolve_reduction(_surrogate_observer(_perm_table(0, 8, 6), np.ones(6), direction="less_equal"))
    surr = next(d for d in red["derived"] if d["name"] == "pval")["surrogate"]
    assert surr["compare"] == "<="


def test_surrogate_is_interleaved_in_declaration_order():
    """`stat` (observed) must precede `pval` (its surrogate), and both precede any consumer — the accumulator reads `pval`, so a flat 'all equations then all surrogates' emission would reference it before it exists."""
    red = resolve_reduction(_surrogate_observer(_perm_table(0, 8, 6), np.ones(6)))
    names = [d["name"] for d in red["derived"]]
    assert names.index("stat") < names.index("pval")
    pval_entry = next(d for d in red["derived"] if d["name"] == "pval")
    assert "surrogate" in pval_entry and "expr" not in pval_entry


# ── resolver: validation ────────────────────────────────────────────────────────────────────


def test_unknown_permute_symbol_is_rejected():
    obs = _surrogate_observer(_perm_table(0, 8, 6), np.ones(6))
    obs.dynamics.derived_variables["pval"].surrogate.permute = "nope"
    with pytest.raises(ValueError, match="permutes 'nope'"):
        resolve_reduction(obs)


def test_undeclared_permutation_table_is_rejected():
    obs = _surrogate_observer(_perm_table(0, 8, 6), np.ones(6))
    obs.dynamics.derived_variables["pval"].surrogate.permutations = "ghost"
    with pytest.raises(ValueError, match="permutation table 'ghost'"):
        resolve_reduction(obs)


def test_statistic_declared_after_surrogate_is_rejected():
    """A statistic declared AFTER its surrogate is rejected at resolve time.

    The surrogate reuses the statistic DV as its observed value (`_obs = <stat>`), so that ordering would emit a forward reference and fail at runtime with NameError.
    """
    perms = _perm_table(0, 8, 6)
    obs = Observation(
        name="obs",
        source=["x"],
        dynamics=Dynamics(
            name="observer",
            parameters={
                "w": Parameter(name="w", value=[1.0] * 6),
                "perms": Parameter(name="perms", value=[[int(j) for j in r] for r in perms]),
            },
            state_variables={
                "acc": StateVariable(name="acc", equation=Equation(rhs="acc + pval"), equation_type="recurrence")
            },
            derived_variables={  # surrogate declared BEFORE its statistic
                "pval": DerivedVariable(name="pval", surrogate=Surrogate(statistic="stat", permute="x", permutations="perms")),
                "stat": DerivedVariable(name="stat", equation=Equation(rhs="w * x")),
                "out": DerivedVariable(name="out", equation=Equation(rhs="acc / count")),
            },
            output=["out"],
        ),
    )
    with pytest.raises(ValueError, match="declared at or after it"):
        resolve_reduction(obs)


def test_typo_direction_is_rejected():
    """`direction` drives both the comparison operator and the family-wise extremum; a typo must fail loudly rather than silently flip the test to less_equal/min-T."""
    obs = _surrogate_observer(_perm_table(0, 8, 6), np.ones(6))
    obs.dynamics.derived_variables["pval"].surrogate.direction = "greater"
    with pytest.raises(ValueError, match="direction 'greater'"):
        resolve_reduction(obs)


def test_family_wise_less_equal_derives_min_t():
    """A negative (less_equal) family-wise test resolves to min-T — the extremum tracks the sidedness, so an incoherent max-extremum/negative-test pairing cannot be expressed."""
    red = resolve_reduction(_surrogate_observer(_perm_table(0, 8, 6), np.ones(6), family_wise=True, direction="less_equal"))
    surr = next(d for d in red["derived"] if d["name"] == "pval")["surrogate"]
    assert surr["family_reduce"] == "nanmin"
    assert surr["compare"] == "<="


# ── emitter: byte-identity vs numpy under the shared table ──────────────────────────────────


@pytest.mark.parametrize("direction,cmp", [("greater_equal", "ge"), ("less_equal", "le")])
def test_symmetric_surrogate_matches_numpy(direction, cmp):
    perms, w = _perm_table(1, 32, 6), np.linspace(0.5, 2.0, 6)
    data = _trajectory(seed=2)
    red = resolve_reduction(_surrogate_observer(perms, w, direction=direction))
    init, update, finalize = _factory(red)(s_var=0, dt=1.0, skip=0)
    got = finalize(update(init(data[0], data.shape[0]), data))

    ref = _reference_over_time(data[:, 0, :], w, perms, cmp=cmp)
    assert got.shape == ref.shape == (6,)
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-9, atol=1e-12)


def test_family_wise_maxT_surrogate_matches_numpy():
    """The Westfall–Young FWE null: the permuted statistic is reduced over vertices (nanmax) to one family-wise extremum per permutation, and each observed vertex is tested against it — Koller's max-over-vertices wave surrogate."""
    perms, w = _perm_table(3, 40, 6), np.linspace(0.5, 2.0, 6)
    data = _trajectory(seed=4)
    red = resolve_reduction(_surrogate_observer(perms, w, family_wise=True))
    init, update, finalize = _factory(red)(s_var=0, dt=1.0, skip=0)
    got = finalize(update(init(data[0], data.shape[0]), data))

    ref = _reference_over_time(data[:, 0, :], w, perms, family="nanmax")
    assert got.shape == ref.shape == (6,)
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("family_wise", [False, True])
@pytest.mark.parametrize("block_size", [7, 16, 31])
def test_surrogate_is_block_decomposition_invariant(family_wise, block_size):
    """The exploration grid folds blocks, not one trajectory; the per-step surrogate is stateless, so the accumulated mean is bit-exact across any block boundary."""
    perms, w = _perm_table(5, 24, 6), np.linspace(0.5, 2.0, 6)
    data = _trajectory(seed=6, T=96)
    red = resolve_reduction(_surrogate_observer(perms, w, family_wise=family_wise))
    init, update, finalize = _factory(red)(s_var=0, dt=1.0, skip=0)

    single = finalize(update(init(data[0], data.shape[0]), data))
    acc = init(data[0], data.shape[0])
    for s in range(0, data.shape[0], block_size):
        acc = update(acc, data[s : s + block_size])
    assert float(jnp.max(jnp.abs(finalize(acc) - single))) == 0.0


def test_emitted_fold_binds_the_permutation_table_by_name():
    """The (n_perm, n) table is a captured constant, gathered once per step, not inlined per comparison; the fold reads `x[perms]` and vmaps the statistic over it."""
    red = resolve_reduction(_surrogate_observer(_perm_table(0, 8, 6), np.ones(6), family_wise=True))
    code = _render(red)
    assert "jax.vmap(_surrstat_pval)(x[perms])" in code
    assert "jnp.nanmax(_null_pval" in code
    assert "_obs_pval = stat" in code  # observed reuses the already-computed statistic DV
