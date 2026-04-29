import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tvbo.api.experiment_api import router as experiment_router


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(experiment_router)
    return TestClient(app)


def _first_experiment_id(client: TestClient) -> str:
    resp = client.get("/api/v1/experiments")
    assert resp.status_code == 200
    experiments = resp.json()
    if not experiments:
        pytest.skip("No experiments available in tvbo database")
    return experiments[0]["id"]


def test_experiment_export_formats_endpoint(client: TestClient):
    resp = client.get("/api/v1/experiments/formats")
    assert resp.status_code == 200

    formats = {item["format"] for item in resp.json()}
    assert "yaml" in formats
    assert "neuroml" in formats
    assert "lems" in formats
    assert "openminds" in formats
    assert "tvb" in formats
    assert "tvboptim" in formats


def test_render_stored_experiment_yaml(client: TestClient):
    experiment_id = _first_experiment_id(client)

    resp = client.get(f"/api/v1/experiments/{experiment_id}/render", params={"format": "yaml"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-yaml")
    assert "id:" in resp.text


def test_render_stored_experiment_tvb(client: TestClient):
    experiment_id = _first_experiment_id(client)

    resp = client.get(f"/api/v1/experiments/{experiment_id}/render", params={"format": "tvb"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/x-python")
    assert "define_simulation" in resp.text
