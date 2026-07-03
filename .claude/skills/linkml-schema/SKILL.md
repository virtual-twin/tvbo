---
name: linkml-schema
description: How to edit the LinkML schema in schema/*.yaml and why tvbo/datamodel/**
  is generated and must never be hand-edited.
---

# LinkML Schema

The TVBO datamodel is defined in **LinkML** YAML in `schema/` and *generated* into Python types under `tvbo/datamodel/`.

## What is canonical

- `schema/common.yaml`, `schema/SANDS.yaml`, `schema/openMINDS_tvbo/**` — hand-written LinkML definitions.
- `tvbo/datamodel/pydantic.py`, `tvbo/datamodel/schema.py` — **generated**. Do not edit.

`pyproject.toml` makes this explicit:

- `tool.ruff.extend-exclude` includes `tvbo/datamodel/**`
- `tool.ruff.lint.exclude` includes `tvbo/datamodel/**`
- `tool.mypy.exclude` includes `tvbo/datamodel/`

If your diff touches `tvbo/datamodel/**` *and* anything else, that is a bug — split the change.

## Editing the schema

1. Edit the appropriate `schema/*.yaml`.
2. Regenerate the Python types via the LinkML pipeline (see `Makefile` / CI).
3. Commit both the schema change *and* the regenerated `tvbo/datamodel/**` in the same commit.

Never patch the generated file directly — the next regeneration will silently revert it.

## When in doubt

- LinkML 1.11+ is required. See `dependencies` in `pyproject.toml`.
- The LinkML schema is the source of truth for what fields a `SimulationExperiment`, `Dynamics`, `Network`, etc. accept. Class implementations in `tvbo/classes/` should match.
