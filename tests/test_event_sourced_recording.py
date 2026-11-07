"""A stimulus can play another run's recording: the subject's own, the trial each seed names, the column each node names.

A data-driven stimulus used to read its samples only from a file (`dataLocation`), so one experiment's recorded output could not drive another's populations, and a per-subject run had no way to play its own subject's recording. A `data` parameter now sources the samples with `used:`. The reference is resolved at run time at the run's `--subject` and injected rather than inlined; the recording is laid out (trial, sample, channel); node k plays column `channel[k]`; and cell i of a random-seed axis plays trial i, so an ensemble meets a fresh recorded trial with a fresh noise draw per member.
"""

import copy

import numpy as np
import pytest
import xarray as xr

from tvbo import SimulationExperiment

N_STEPS = 20

EXP = {
    "id": 2,
    "label": "sourced recording fixture",
    "dynamics": {
        "name": "Integrator",
        "system_type": "continuous",
        "output": ["x"],
        "parameters": {"drive": {"value": 0.0}},
        "state_variables": {"x": {"equation": {"rhs": "drive"}, "initial_value": 0.0}},
    },
    "network": {"number_of_nodes": 2},
    "integration": {"method": "euler", "step_size": 1.0, "duration": float(N_STEPS), "transient_time": 0.0, "unit": "ms"},
    "events": {
        "drive": {
            "event_type": "stimulus",
            "parameters": {
                "data": {"used": {"experiment": 7, "output": "recording"}},
                "channel": {"value": [2.0, 0.0], "shape": "(n_nodes,)"},
            },
        }
    },
    "execution": {"backend": "tvboptim"},
}


def _level(subject, trial, channel):
    """The constant a recorded (subject, trial, channel) series holds, distinct for every combination."""
    return 1000.0 * subject + 10.0 * trial + channel


@pytest.fixture
def recordings(tmp_path):
    """Experiment 7 recorded per subject: three trials of 40 samples on three channels, each series a constant."""
    for s in (1, 2):
        rec = np.stack([np.stack([np.full(40, _level(s, t, c)) for c in range(3)], axis=-1) for t in range(3)])
        ds = xr.Dataset({"observation__recording": (("trial", "time", "channel"), rec)})
        ds.to_netcdf(tmp_path / f"sub-0{s}_exp-7_desc-Src_result.h5", engine="h5netcdf")
    return tmp_path


def _experiment(**overrides):
    spec = copy.deepcopy(EXP)
    for key, value in overrides.items():
        spec[key] = value
    return SimulationExperiment(**spec)


def test_a_subject_plays_its_own_recording_on_the_declared_columns(recordings):
    """Euler over constant drive: x after N steps is N times the level each node's column carries."""
    result = _experiment().run("tvboptim", results_root=recordings, active_subject="02")
    final = np.asarray(result.data)[-1, 0, :]
    np.testing.assert_allclose(final, [N_STEPS * _level(2, 0, 2), N_STEPS * _level(2, 0, 0)], rtol=1e-6)


def test_each_seed_plays_its_own_trial(recordings):
    """Cell i of the seed axis plays trial i; the noise the axis requires is too small to matter here."""
    spec = copy.deepcopy(EXP)
    spec["dynamics"]["state_variables"]["x"]["noise"] = {"additive": True, "parameters": {"sigma": {"value": 1e-9}}}
    spec["observations"] = {"x_final": {"source": ["x"], "aggregation": "last"}}
    spec["explorations"] = {
        "trials": {"space": {"execution.random_seed": {"domain": {"lo": 100, "hi": 102, "n": 3}}}},
    }
    exp = SimulationExperiment(**spec)
    written = exp.run("tvboptim", results_root=recordings, active_subject="01").save(str(recordings / "out"))
    container = next(p for p in map(str, written) if p.endswith("_result.h5"))
    with xr.open_dataset(container, engine="h5netcdf") as ds:
        finals = np.asarray(ds["x_final"].transpose("execution.random_seed", ...).values).reshape(3, -1)
    for trial in range(3):
        np.testing.assert_allclose(
            finals[trial, -2:], [N_STEPS * _level(1, trial, 2), N_STEPS * _level(1, trial, 0)], rtol=1e-4
        )


def test_more_seeds_than_recorded_trials_is_refused(recordings):
    spec = copy.deepcopy(EXP)
    spec["dynamics"]["state_variables"]["x"]["noise"] = {"additive": True, "parameters": {"sigma": {"value": 1e-9}}}
    spec["explorations"] = {"trials": {"space": {"execution.random_seed": {"domain": {"lo": 0, "hi": 4, "n": 5}}}}}
    with pytest.raises(ValueError, match="holds 3 trials"):
        SimulationExperiment(**spec).run("tvboptim", results_root=recordings, active_subject="01")


def test_a_data_parameter_without_a_source_is_refused():
    events = copy.deepcopy(EXP["events"])
    events["drive"]["parameters"]["data"] = {"value": 1.0}
    with pytest.raises(ValueError, match="without `used:`"):
        _experiment(events=events).render_code("tvboptim")


def test_a_parameter_the_recording_cannot_evaluate_is_refused():
    events = copy.deepcopy(EXP["events"])
    events["drive"]["parameters"]["frequency"] = {"value": 10.0}
    with pytest.raises(ValueError, match="frequency"):
        _experiment(events=events).render_code("tvboptim")


def test_cubic_interpolation_of_a_recording_is_refused():
    events = copy.deepcopy(EXP["events"])
    events["drive"]["interpolation"] = "cubic"
    with pytest.raises(ValueError, match="interpolates linearly"):
        _experiment(events=events).render_code("tvboptim")
