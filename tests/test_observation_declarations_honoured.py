"""Two observation declarations that used to be accepted and then ignored.

An `equation` stage is rendered inline as the observation's whole value, so any stage declared after it was dropped without a word and the observation came out untransformed (a column selection after a normalisation stored all 379 columns). It is now refused. And `record: false` kept an intermediate out of the declared observations but not out of the base run's copy, so the container still carried the full intermediate trajectory it was declared not to store.
"""

from __future__ import annotations

import copy

import pytest
import xarray as xr

from tvbo import SimulationExperiment

EXP = {
    "id": 3,
    "label": "observation declarations fixture",
    "dynamics": {
        "name": "Integrator",
        "system_type": "continuous",
        "output": ["x"],
        "parameters": {"k": {"value": 1.0}},
        "state_variables": {"x": {"equation": {"rhs": "k"}, "initial_value": 0.0}},
    },
    "network": {"number_of_nodes": 3},
    "integration": {"method": "euler", "step_size": 1.0, "duration": 10.0, "transient_time": 0.0, "unit": "ms"},
    "observations": {
        "trace": {"source": ["x"], "record": False},
        "scaled": {
            "source": ["trace"],
            "pipeline": [{"name": "double", "equation": {"rhs": "2 * trace"}}],
        },
    },
    "execution": {"backend": "tvboptim"},
}


def test_a_stage_after_an_equation_is_refused():
    spec = copy.deepcopy(EXP)
    spec["observations"]["scaled"]["pipeline"].append(
        {"name": "pick", "callable": {"name": "take", "module": "numpy"}, "arguments": [{"name": "indices", "value": [0]}]}
    )
    with pytest.raises(ValueError, match="must be the pipeline's only stage"):
        SimulationExperiment(**spec).render_code("tvboptim")


def test_an_unrecorded_observation_leaves_no_base_run_copy(tmp_path):
    written = SimulationExperiment(**copy.deepcopy(EXP)).run("tvboptim").save(str(tmp_path))
    container = next(p for p in map(str, written) if p.endswith("_result.h5"))
    with xr.open_dataset(container, engine="h5netcdf") as ds:
        names = set(ds.data_vars)
    assert not any(n.endswith("trace") for n in names), names
    assert any(n.endswith("scaled") for n in names), names
