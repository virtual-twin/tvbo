"""A data-driven stimulus Event is interpolated the way it was declared, and its path is read relative to the spec.

An Event carrying a `dataLocation` is a waveform read from a file rather than an equation evaluated at t: an acoustic drive, a measured stimulation field. The tvboptim codegen used to emit a bespoke class whose `compute` called `jnp.interp`, so a declared `interpolation: cubic` was accepted and never run. The samples now go to tvboptim's `DataInput`, which builds the diffrax interpolation object once and evaluates it inside the scan. The emitted class adds `amplitude` as a live leaf a sweep or a gradient fit can write; a parameter such a stimulus cannot evaluate raises instead of being dropped.
"""

import copy

import numpy as np
import pytest

from tvbo import SimulationExperiment

EXP = {
    "id": 1,
    "label": "data-driven stimulus fixture",
    "dynamics": {
        "name": "MiniOsc",
        "system_type": "continuous",
        "output": ["x"],
        "parameters": {"a": {"value": 1.0}},
        "state_variables": {"x": {"equation": {"rhs": "-a * x + drive"}, "initial_value": 0.0}},
    },
    "network": {"number_of_nodes": 4},
    "integration": {"method": "heun", "step_size": 1.0, "duration": 20.0, "transient_time": 0.0, "unit": "ms"},
    "events": {
        "drive": {
            "event_type": "stimulus",
            "target_variable": "drive",
            "dataLocation": "wave.npy",
            "sampling_rate": 1.0,
            "interpolation": "cubic",
            "nodes": [1],
            "weights": [1.0],
        }
    },
    "execution": {"backend": "tvboptim"},
}


@pytest.fixture
def waveform(tmp_path):
    """A short waveform on disk, and the spec directory a relative `dataLocation` resolves against."""
    path = tmp_path / "wave.npy"
    np.save(path, np.sin(np.arange(64, dtype=np.float32) / 4.0))
    return tmp_path


def _experiment(spec_dir, **event_overrides):
    spec = copy.deepcopy(EXP)
    spec["events"]["drive"].update(event_overrides)
    exp = SimulationExperiment(**spec)
    exp._source_file = str(spec_dir / "study.yaml")
    exp.configure()
    return exp


def test_a_relative_data_location_resolves_against_the_spec(waveform):
    """The generated module inlines the path, so a relative one would resolve against whatever directory the run started in."""
    exp = _experiment(waveform)
    assert exp.events["drive"].dataLocation == str((waveform / "wave.npy").resolve())


def test_an_absolute_data_location_is_left_alone(waveform):
    absolute = str((waveform / "wave.npy").resolve())
    exp = _experiment(waveform, dataLocation=absolute)
    assert exp.events["drive"].dataLocation == absolute


@pytest.mark.parametrize("interpolation", ["linear", "cubic"])
def test_the_declared_interpolation_reaches_the_generated_input(waveform, interpolation):
    """The regression: every rendering said `jnp.interp` whatever the recipe declared."""
    code = _experiment(waveform, interpolation=interpolation).render_code("tvboptim")
    assert "class driveInput(DataInput):" in code
    assert f'interpolation="{interpolation}"' in code
    assert "jnp.interp(" not in code
    assert "from tvboptim.experimental.network_dynamics.external_input.data import DataInput" in code


def test_an_unsupported_interpolation_is_refused(waveform):
    with pytest.raises(ValueError, match="interpolation"):
        _experiment(waveform, interpolation="quintic").render_code("tvboptim")


def test_a_declared_amplitude_scales_the_drive(waveform):
    """`amplitude` is the one knob a waveform has, and it is a live leaf rather than a number baked into the samples."""
    code = _experiment(waveform, parameters={"amplitude": {"value": 0.4}}).render_code("tvboptim")
    assert "amplitude=0.4" in code

    quiet = _experiment(waveform, parameters={"amplitude": {"value": 0.4}}).run("tvboptim")
    loud = _experiment(waveform, parameters={"amplitude": {"value": 0.8}}).run("tvboptim")
    assert np.ptp(np.asarray(loud.data)[:, 0, 1]) > 1.9 * np.ptp(np.asarray(quiet.data)[:, 0, 1])


def test_a_parameter_the_waveform_cannot_evaluate_is_refused(waveform):
    """A data-driven stimulus has no equation to bind a symbol to, so accepting one would mean ignoring it."""
    with pytest.raises(ValueError, match="frequency"):
        _experiment(waveform, parameters={"frequency": {"value": 10.0}}).render_code("tvboptim")


def test_a_data_driven_stimulus_drives_only_its_targeted_nodes(waveform):
    """End to end on tvboptim: the masked node moves, its untargeted neighbours stay at their initial value."""
    result = _experiment(waveform).run("tvboptim")
    trajectory = np.asarray(result.data)[:, 0, :]
    assert np.ptp(trajectory[:, 1]) > 1e-3
    for node in (0, 2, 3):
        assert np.ptp(trajectory[:, node]) < 1e-12
