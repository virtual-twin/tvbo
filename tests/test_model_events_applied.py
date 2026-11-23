"""Brian2, tvboptim and jax apply the event a model declares, as NeuroML does, reading the declaration under whatever names it carries.

The curated NeuroML Ex8 AdEx cell (experiment 905) spikes when ``v > thresh`` and then sets ``v = reset; w = w + b``. LEMS applies that reset, so its run is the reference: every other backend that marks itself as applying model events must spike at the same steps. The same cell with every name changed (``v`` to ``x``, ``w`` to ``y``, ``thresh`` to ``top``, ``reset`` to ``back``, the model to ``AnyCell``) must spike at the same steps too, because a backend reads the event, never a name. Where a backend cannot apply an event it refuses the model, naming the event.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import tvbo
from tvbo import SimulationExperiment

ADEX = Path(tvbo.__file__).parent / "database" / "experiments" / "neuroml" / "AdaptiveExponentialIF_Ex8.yaml"
RENAMES = {"v": "x", "w": "y", "thresh": "top", "reset": "back", "AdaptiveExponentialIF": "AnyCell"}
DT = 0.025
DROP = (48.5 - 40.4) / 2
"""Half the gap from the cell's threshold (-40.4) down to its reset (-48.5): a step whose membrane falls further than this is a reset."""
MS_PER_TIME_UNIT = {"neuroml": 1000.0, "tvboptim": 1.0, "jax": 1.0}
"""LEMS reports its run in SI seconds; tvboptim and jax in the experiment's time unit, ms."""


def _text(renamed=False, old=None, new=None):
    """Experiment 905's YAML, with every name changed when *renamed*, and *old* replaced by *new*."""
    text = ADEX.read_text()
    if old is not None:
        assert old in text
        text = text.replace(old, new)
    if renamed:
        text = re.sub(r"\b(" + "|".join(RENAMES) + r")\b", lambda m: RENAMES[m.group(1)], text)
    return text


def _cell(tmp_path, renamed=False, old=None, new=None):
    """Experiment 905 loaded from its YAML, edited as `_text` edits it."""
    path = tmp_path / "cell.yaml"
    path.write_text(_text(renamed, old, new))
    return SimulationExperiment.from_file(str(path))


def _reset_times(experiment, backend, membrane):
    """The time, in ms, of each sample of *backend*'s run whose membrane fell by more than `DROP` since the sample before."""
    trace = experiment.run(backend).integration
    v = np.asarray(trace.sel(variable=membrane).squeeze())
    t = np.asarray(trace.time) * MS_PER_TIME_UNIT[backend]
    return t[1:][np.diff(v) < -DROP]


@pytest.fixture(scope="module", params=[False, True], ids=["curated", "renamed"])
def cell(request, tmp_path_factory):
    """``(renamed, membrane, reference)``: the cell, its membrane's name, and the reset times LEMS gives it."""
    renamed = request.param
    membrane = "x" if renamed else "v"
    reference = _reset_times(_cell(tmp_path_factory.mktemp("lems"), renamed), "neuroml", membrane)
    assert reference.size >= 2, f"the LEMS reference spiked {reference.size} times"
    return renamed, membrane, reference


@pytest.mark.backend_neuroml
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("brian2", marks=pytest.mark.backend_brian2),
        pytest.param("tvboptim", marks=pytest.mark.backend_tvboptim),
        pytest.param("jax", marks=pytest.mark.backend_jax),
    ],
)
def test_the_backend_spikes_where_lems_does(cell, backend, tmp_path):
    """The same number of resets as LEMS, each within one step of its LEMS reset (Brian2 stamps a spike at the step that crossed, its reset lands at the next)."""
    pytest.importorskip(backend)
    renamed, membrane, reference = cell
    experiment = _cell(tmp_path, renamed)
    if backend == "brian2":
        times = np.sort(next(iter(experiment.run("brian2")._extras["spikes"].values()))["t_ms"])
    else:
        times = _reset_times(experiment, backend, membrane)
    assert times.size == reference.size, f"{backend} spiked {times.size} times, LEMS {reference.size}"
    np.testing.assert_allclose(times, reference, atol=1.001 * DT)


SPIKE = 'affect:\n        rhs: "v = reset; w = w + b"'


@pytest.mark.backend_neuroml
@pytest.mark.backend_brian2
@pytest.mark.parametrize(
    "old, new",
    [
        ("v = reset; w = w + b", "v = reset; w = w + b + 0.5 * (I_ext - 800.0)"),
        (
            'rhs: "v = reset; w = w + b"\n\nnetwork:',
            'rhs: "v = reset; w = w + b + drive_in"\n  coupling_inputs:\n    drive_in: {}\n\nnetwork:',
        ),
    ],
    ids=["derived-variable", "coupling-input"],
)
def test_brian2_resets_from_what_its_cells_evaluate(cell, old, new, tmp_path):
    """Brian2's reset reads the cell's derived variables and its coupling input as it reads its state, so a reset adding ``0.5 * (I_ext - 800.0)``, zero at the cell's constant 800, or the unconnected coupling input, which is zero, spikes where LEMS spikes the curated cell."""
    pytest.importorskip("brian2")
    renamed, _, reference = cell
    experiment = _cell(tmp_path, renamed, old, new)
    times = np.sort(next(iter(experiment.run("brian2")._extras["spikes"].values()))["t_ms"])
    assert times.size == reference.size, f"brian2 spiked {times.size} times, LEMS {reference.size}"
    np.testing.assert_allclose(times, reference, atol=1.001 * DT)


NON_STATE = r"cannot apply event 'spike'.*: it assigns I_ext, which is not a state variable"


@pytest.mark.parametrize(
    "backend, old, new, match",
    [
        (
            "tvboptim",
            "method: euler",
            "method: Dopri5",
            r"cannot apply event 'spike' \(v > thresh\): an adaptive Diffrax solver",
        ),
        (
            "brian2",
            SPIKE,
            SPIKE + '\n    burst:\n      condition:\n        rhs: "w > 1000.0"\n      affect:\n        rhs: "w = 0.0"',
            r"the cell 'AdaptiveExponentialIF' declares 2 such events: spike, burst",
        ),
        (
            "tvboptim",
            SPIKE,
            SPIKE + '\n      affect_negative:\n        rhs: "w = w"',
            r"cannot apply event 'spike'.*: it declares an affect_negative",
        ),
        ("jax", "v = reset; w = w + b", "v = reset; I_ext = 0.0", NON_STATE),
        (
            "brian2",
            "v = reset; w = w + b",
            "v = reset; I_ext = 0.0",
            r"the brian2 backend applies a cell's event as its threshold and reset.*" + NON_STATE,
        ),
    ],
    ids=["diffrax", "two-conditioned-events-brian2", "affect-negative", "non-state-jax", "non-state-brian2"],
)
def test_an_event_the_backend_cannot_apply_is_refused_by_name(backend, old, new, match, tmp_path):
    """Each refusal names the event and what in its declaration, or in the run's setting, puts it out of the backend's reach."""
    with pytest.raises(NotImplementedError, match=match):
        _cell(tmp_path, old=old, new=new).render_code(backend)


def test_a_model_with_modes_is_refused_by_the_post_step_backends(tmp_path):
    """The tvboptim and jax backends read events through `post_step_resets`, which refuses an event of a model with more than one mode."""
    from tvbo.adapters.base import post_step_resets

    dynamics = _cell(tmp_path).dynamics
    dynamics.number_of_modes = 2
    for backend in ("tvboptim", "jax"):
        modes = rf"the {backend} backend cannot apply event 'spike' .*: its model has 2 modes"
        with pytest.raises(NotImplementedError, match=modes):
            post_step_resets(dynamics, backend)


@pytest.mark.backend_tvboptim
def test_a_grouped_network_is_refused_by_the_event(tmp_path):
    """The grouped tvboptim scan steps one state per group, which the per-step event wrapper does not reach, so a group whose dynamics declares an event is refused naming it."""
    pytest.importorskip("tvboptim")
    from tvbo.adapters.tvboptim import run_heterogeneous_tvboptim

    experiment = _cell(tmp_path)
    with pytest.raises(NotImplementedError, match=r"the heterogeneous tvboptim backend cannot apply event 'spike'"):
        run_heterogeneous_tvboptim(experiment, dynamics_lib={"AdaptiveExponentialIF": experiment.dynamics})
