---
name: linkml-schema
description: How to edit the LinkML schema in schema/*.yaml and why tvbo/datamodel/**
  is generated and must never be hand-edited.
---

# LinkML Schema

The TVBO datamodel is defined in **LinkML** YAML in `schema/` and *generated* into Python types under `tvbo/datamodel/`.

## What is canonical

- `schema/common.yaml`, `schema/SANDS.yaml`, `schema/openMINDS_tvbo/**` — hand-written LinkML definitions.
- `tvbo/datamodel/pydantic.py`, `tvbo/datamodel/schema.py`, `tvbo/datamodel/tvbo_datamodel.schema.json`
  — **generated** at build time by `hatch_build.py` and **gitignored**. Do not edit.

`pyproject.toml` makes this explicit:

- `tool.ruff.extend-exclude` includes `tvbo/datamodel/**`
- `tool.ruff.lint.exclude` includes `tvbo/datamodel/**`
- `tool.mypy.exclude` includes `tvbo/datamodel/`

## Editing the schema

1. Edit the appropriate `schema/*.yaml`.
2. Regenerate the Python types via the LinkML pipeline (see `Makefile` / CI). A build
   regenerates them automatically through the hatch build hook.

The generated types are untracked, so a schema change has nothing to stage alongside it —
`schema/*.yaml` is the whole diff. Never patch a generated file directly: it is not in
git, and the next build silently overwrites it.

## When in doubt

- LinkML 1.11+ is required. See `dependencies` in `pyproject.toml`.
- The LinkML schema is the source of truth for what fields a `SimulationExperiment`, `Dynamics`, `Network`, etc. accept. Class implementations in `tvbo/classes/` should match.
