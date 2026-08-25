"""The settle is folded away in-band, not materialised and sliced.

A declared ``transient_time`` opens the scan at ``-transient_time`` so ``t=0`` is the first measured step. Stacking that settle and slicing it off afterwards costs the whole joined trajectory in memory: on a 379-node Jansen-Rit grid with a 60 s settle and a 5.12 s window that is 2.37 GB per cell, of which 2.24 GB is discarded on the next line. The ``(init, update, finalize)`` window reducer rides the native block scan instead and keeps only the measured window, so the same cell costs what the window costs.

The reducer is only sound where blocking is free. Blocking SELECTS the noise realization in tvboptim — a block grain regenerates each block from ``(key, block_index)`` — so an experiment that declared ``noise_draw: fused`` must keep the unblocked stack-and-cut path rather than have its trajectory quietly moved. That constraint is about noise, not about the declaration: a deterministic network has no realization to move, and gating it on the declaration alone silently doubled the memory of every noise-free study the day ``fused`` became the default.
"""

import ast

import numpy as np
import pytest

pytest.importorskip("tvboptim")

from tvbo import SimulationExperiment, database_path

DETERMINISTIC = "Delay_Speed_Synchronization"
STOCHASTIC = "JR_MEG_FrequencyGradient_Optimization"
CALL = "reduce=_window_reducer("


def render(name, transient=2000.0, noise_draw=None):
    """A curated experiment re-declared with a settle, rendered to tvboptim source."""
    exp = SimulationExperiment.from_file(str(database_path / "experiments" / f"{name}.yaml"))
    exp.integration.transient_time = transient
    if noise_draw is not None:
        exp.integration.noise_draw = noise_draw
    exp.configure()
    return exp.render_code("tvboptim")


@pytest.mark.parametrize("noise_draw", ["blocked", "fused"])
def test_a_noise_free_settle_folds_under_either_draw(noise_draw):
    """No noise means no realization to move, so the draw does not decide the path."""
    assert CALL in render(DETERMINISTIC, noise_draw=noise_draw)


def test_a_stochastic_settle_folds_when_blocking_is_declared():
    assert CALL in render(STOCHASTIC, noise_draw="blocked")


def test_a_fused_stochastic_settle_keeps_the_unblocked_path():
    """Folding would impose a block grain, and that grain is the realization."""
    assert CALL not in render(STOCHASTIC, noise_draw="fused")


MATERIALISED = "is materialised, not folded"


@pytest.mark.parametrize(
    "name,noise_draw,folds",
    [(DETERMINISTIC, "blocked", True), (DETERMINISTIC, "fused", True),
     (STOCHASTIC, "blocked", True), (STOCHASTIC, "fused", False)],
)
def test_a_materialised_settle_says_so(name, noise_draw, folds):
    """The expensive branch announces itself, because its cost is otherwise invisible.

    Stacking a settle and cutting it costs memory in proportion to the settle and returns exactly the same numbers, so nothing in a run's output distinguishes it from the folded path. A recipe that loses folding by declaring `noise_draw: fused` should learn that from the log rather than from a machine running out of memory.
    """
    code = render(name, noise_draw=noise_draw)
    assert (CALL in code) is folds
    assert (MATERIALISED in code) is not folds


def test_no_settle_emits_no_reducer():
    code = render(DETERMINISTIC, transient=0.0)
    assert "def _window_reducer(" not in code and CALL not in code


def _reducer_from(code):
    """The emitted ``_window_reducer``, lifted out of the generated module and executed alone.

    Taken from the render rather than restated here: a copy of the definition in the test would keep passing after the emitted one changed, which is the failure this exists to catch.
    """
    tree = ast.parse(code)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_window_reducer")
    import jax.numpy as jnp

    ns = {"jnp": jnp}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<emitted>", "exec"), ns)
    return ns["_window_reducer"]


@pytest.mark.parametrize(
    "n_steps,n_skip,n_keep,block",
    [
        (20, 5, 15, 5),
        (100, 40, 60, 25),
        (100, 40, 60, 7),
        (1000, 900, 100, 128),
        (130240, 120000, 10240, 1000),
    ],
)
def test_folding_the_settle_equals_stacking_and_cutting(n_steps, n_skip, n_keep, block):
    """The reducer's output is the slice the materialise path would have taken.

    Checked over block decompositions that do not divide the settle, since the boundary the fold has to get right is the one falling inside a block rather than between two.
    """
    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    traj = jnp.asarray(rng.standard_normal((n_steps, 2, 3)))
    init, update, finalize = _reducer_from(render(DETERMINISTIC))(n_skip, n_keep)

    acc = init(jnp.zeros((2, 3)), n_steps)
    for lo in range(0, n_steps, block):
        acc = update(acc, traj[lo : lo + block])

    np.testing.assert_array_equal(np.asarray(finalize(acc)), np.asarray(traj[n_skip : n_skip + n_keep]))
