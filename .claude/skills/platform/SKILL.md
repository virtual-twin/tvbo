---
name: platform
description: "How to use the Odoo-based TVBO platform at tvbo.charite.de \u2014 retrieve\
  \ curated knowledge, manage your account, share SimulationExperiments and Studies\
  \ (public or with API key), and load them back into Python."
---

# The TVBO Platform

[**tvbo.charite.de**](https://tvbo.charite.de) is the Odoo-based management layer for everything ontology-backed. It is *not* the Python package — it's a **separate service** that the Python package talks to.

## What the platform does

1. **Displays knowledge** — browse the ontology, curated models, parcellations, and atlases.
2. **Lets users retrieve knowledge** programmatically via the REST API.
3. **Lets users set up their own models** — define `Dynamics`, `Network`, `Experiment` records through the web UI.
4. **Stores and shares `SimulationExperiment`s and `SimulationStudy`s**:
   - **Public** experiments are visible to everyone.
   - **Private** experiments require an **API key** tied to a user account.
5. **Loads experiments back into Python** — fetch a record by ID, get a fully constructed `SimulationExperiment` you can `run()`.

Roadmap: ask-and-discuss layer for the community to comment on shared experiments and studies.

## Accessing the platform from Python

The REST client lives in `tvbo.api` (install with `pip install tvbo[api]`):

```python
from tvbo.api import ...  # see tvbo/api/ for the exact entry points
```

Key modules:

- `tvbo/api/ontology_api.py` — ontology queries
- `tvbo/api/dynamics_api.py` — fetch/store Dynamics
- `tvbo/api/experiment_api.py` — fetch/store SimulationExperiments
- `tvbo/api/network_api.py` — fetch/store Networks
- `tvbo/api/main.py` — FastAPI app entry point (for running a local mirror)

## Authentication

Public reads work without auth. For private records:

1. Create an account at <https://tvbo.charite.de>.
2. Generate an API key from your account page.
3. Provide it to the client (mechanism documented per-endpoint).

Treat API keys like passwords: don't commit them. Use environment variables or a secrets manager.

## When to use the platform vs local files

- Use the platform when you want **other people to be able to reproduce your run** without you mailing them a YAML file.
- Use local files when you're iterating and don't need persistence.
- Hybrid: prototype locally, push the final experiment to the platform with a DOI-like identifier for citation.

## Heads-up

- The platform is the *system of record* for shared experiments. If you re-run a shared experiment locally and get different results, check that you fetched the exact stored revision (the platform versions records).
- The Odoo backend is a separate codebase from the `tvbo` Python package. Issues with the web UI or auth go through platform-side support, not the GitHub repo.
