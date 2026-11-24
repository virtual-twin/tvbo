"""`POST /experiment/run` runs the experiment as declared, overriding only what the request states, and returns the state laid out time, variable, node, mode."""

import pytest

pytest.importorskip("fastapi", reason="requires the 'api' extra (pip install tvbo[api])")
pytest.importorskip("jax")
pytest.importorskip("tvboptim")

from fastapi.testclient import TestClient  # noqa: E402

from tvbo.api.main import app  # noqa: E402

EXPERIMENT = {
    "label": "Leak",
    "dynamics": {
        "name": "Leak",
        "parameters": {"k": {"value": 0.5}},
        "state_variables": {"x": {"equation": {"rhs": "-k*x"}, "initial_value": 1.0}},
    },
    "network": {"label": "Solo", "number_of_nodes": 1},
    "integration": {"method": "euler", "step_size": 1.0, "duration": 20.0, "transient_time": 0.0},
}


@pytest.mark.parametrize(
    ("request_fields", "step", "duration"),
    [({}, 1.0, 20.0), ({"step_size": 0.5, "duration": 10.0}, 0.5, 10.0)],
    ids=["declared", "overridden"],
)
def test_a_run_keeps_the_experiments_own_clock_unless_the_request_states_one(request_fields, step, duration):
    body = TestClient(app).post("/experiment/run", json={"experiment": EXPERIMENT, **request_fields}).json()
    assert body["success"], body["error"]
    assert body["sample_period"] == pytest.approx(step)
    assert body["time"][-1] == pytest.approx(duration, abs=step)
    assert len(body["data"][0]) == 1 and len(body["data"][0][0]) == 1 and len(body["data"][0][0][0]) == 1
    assert body["state_variables"] == ["x"]
