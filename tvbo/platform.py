"""Client for the TVBO platform REST API (load/push saved models & experiments).

This mirrors ``tvbo_platform`` shipped in the tvbo-platform repo
(clients/python), bundled here so users with the ``tvbo`` package can simply::

    from tvbo.platform import TVBOPlatform

    tvbo = TVBOPlatform(base_url="https://platform.example", api_key="tvbo_…")
    exp = tvbo.load_experiment(123)          # -> SimulationExperiment
    tvbo.push_experiment(exp, visibility="shared")

Mint an API key at ``<platform>/my/api-keys``.
"""
from __future__ import annotations

__all__ = ["TVBOPlatform", "TVBOPlatformError"]

import requests


class TVBOPlatformError(RuntimeError):
    """Raised when the platform returns an error response."""


class TVBOPlatform:
    """Client for the TVBO platform REST API.

    Wraps an authenticated `requests` session against a TVBO platform instance,
    exposing helpers to list, fetch, load, and push saved models and
    experiments. Load helpers return live `tvbo` objects (a `Dynamics` or a
    [SimulationExperiment](../classes/experiment.qmd)); push helpers accept YAML
    text, a `dict`, or a `tvbo` object and serialize it for upload.

    Args:
        base_url: Base URL of the platform (a trailing slash is stripped).
        api_key: API key sent as a `Bearer` token; mint one at
            `<platform>/my/api-keys`.
        timeout: Per-request timeout in seconds.

    Raises:
        ValueError: If `base_url` or `api_key` is empty.
    """

    def __init__(self, base_url: str, api_key: str, timeout: int = 60):
        if not base_url or not api_key:
            raise ValueError("base_url and api_key are required")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {api_key}"

    def _get(self, path: str, **params):
        return self._checked(
            self._session.get(self.base_url + path, params=params, timeout=self.timeout))

    def _post(self, path: str, payload: dict):
        return self._checked(
            self._session.post(self.base_url + path, json=payload, timeout=self.timeout))

    @staticmethod
    def _checked(resp):
        if resp.status_code == 401:
            raise TVBOPlatformError("Unauthorized — check your API key.")
        if resp.status_code >= 400:
            detail = resp.text
            try:
                detail = resp.json()
            except ValueError:
                pass
            raise TVBOPlatformError(f"HTTP {resp.status_code}: {detail}")
        return resp

    # -- models ---------------------------------------------------------
    def list_models(self, mine: bool = False) -> list:
        """List the models available on the platform.

        Args:
            mine: If `True`, return only models owned by the authenticated user.

        Returns:
            A list of model metadata dictionaries.
        """
        data = self._get("/api/tvbo/v1/models").json()["data"]
        return [m for m in data if m.get("mine")] if mine else data

    def get_model_yaml(self, model_id: int) -> str:
        """Fetch a model's raw YAML specification.

        Args:
            model_id: Identifier of the model to retrieve.

        Returns:
            The model's YAML text.
        """
        return self._get(f"/api/tvbo/v1/models/{model_id}", format="yaml").text

    def get_model_dict(self, model_id: int) -> dict:
        """Fetch a model's specification as a JSON dictionary.

        Args:
            model_id: Identifier of the model to retrieve.

        Returns:
            The model specification decoded from the JSON `data` payload.
        """
        return self._get(f"/api/tvbo/v1/models/{model_id}", format="json").json()["data"]

    def load_model(self, model_id: int):
        """Load a model from the platform into a `Dynamics` object.

        Fetches the model's YAML and parses it with the `tvbo` pydantic loader.

        Args:
            model_id: Identifier of the model to load.

        Returns:
            The parsed `Dynamics` instance.
        """
        from tvbo.utils import pydantic_loader

        return pydantic_loader.loads(self.get_model_yaml(model_id), "Dynamics")

    def push_model(self, spec, visibility: str = "private") -> dict:
        """Upload a model specification to the platform.

        Args:
            spec: The model to push, given as YAML text, a `dict`, or a `tvbo`
                object (anything `_to_yaml` can serialize).
            visibility: Access level for the created model, e.g. `"private"`,
                `"shared"`, or `"public"`.

        Returns:
            The platform's JSON response describing the created model.
        """
        return self._post(
            "/api/tvbo/v1/models", {"yaml": _to_yaml(spec), "visibility": visibility}
        ).json()

    # -- experiments ----------------------------------------------------
    def list_experiments(self) -> list:
        """List the experiments available on the platform.

        Returns:
            A list of experiment metadata dictionaries.
        """
        return self._get("/api/tvbo/v1/experiments").json()["data"]

    def get_experiment_yaml(self, experiment_id: int) -> str:
        """Fetch an experiment's raw YAML specification.

        Args:
            experiment_id: Identifier of the experiment to retrieve.

        Returns:
            The experiment's YAML text.
        """
        return self._get(f"/api/tvbo/v1/experiments/{experiment_id}", format="yaml").text

    def get_experiment_dict(self, experiment_id: int) -> dict:
        """Fetch an experiment's specification as a JSON dictionary.

        Args:
            experiment_id: Identifier of the experiment to retrieve.

        Returns:
            The experiment specification decoded from the JSON `data` payload.
        """
        return self._get(
            f"/api/tvbo/v1/experiments/{experiment_id}", format="json").json()["data"]

    def load_experiment(self, experiment_id: int):
        """Load an experiment from the platform into a `SimulationExperiment`.

        Fetches the experiment's YAML and parses it via
        [SimulationExperiment.from_string](../classes/experiment.qmd).

        Args:
            experiment_id: Identifier of the experiment to load.

        Returns:
            The parsed `SimulationExperiment` instance.
        """
        from tvbo.classes.experiment import SimulationExperiment

        return SimulationExperiment.from_string(self.get_experiment_yaml(experiment_id))

    def push_experiment(self, spec, visibility: str = "private") -> dict:
        """Upload an experiment specification to the platform.

        Args:
            spec: The experiment to push, given as YAML text, a `dict`, or a
                `tvbo` object (anything `_to_yaml` can serialize).
            visibility: Access level for the created experiment, e.g.
                `"private"`, `"shared"`, or `"public"`.

        Returns:
            The platform's JSON response describing the created experiment.
        """
        return self._post(
            "/api/tvbo/v1/experiments", {"yaml": _to_yaml(spec), "visibility": visibility}
        ).json()


def _to_yaml(spec) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        import yaml

        return yaml.safe_dump(spec, sort_keys=False)
    try:
        from tvbo.utils import pydantic_loader

        if hasattr(spec, "model_dump"):
            return pydantic_loader.dump(spec)
    except Exception:
        pass
    for attr in ("to_string", "to_yaml"):
        if hasattr(spec, attr):
            return getattr(spec, attr)()
    raise TypeError(
        "Unsupported spec type for push; pass YAML text, a dict, or a tvbo object."
    )
