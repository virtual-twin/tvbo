---
name: overview
description: "What TVBO is \u2014 the three pillars (LinkML datamodel, ontology, Python\
  \ package) plus the Odoo platform. Use when working with the tvbo library or deciding\
  \ whether to reach for it."
---

# TVBO Overview

**TVBO (The Virtual Brain Ontology)** is a knowledge-representation and simulation toolkit for large-scale brain network models. It consists of **three pillars** plus a **platform**.

## Pillar 1 — LinkML datamodel

`schema/*.yaml` defines the structured metadata for `SimulationExperiment`, `SimulationStudy`, `Dynamics`, `Network`, `Atlas`, and friends. The Python types in `tvbo/datamodel/` are *generated* from this schema. Users consume them through the higher-level classes in `tvbo/classes/`.

## Pillar 2 — Ontology

`tvb-o.owl` and the TTL/SHACL files in `ontology/` carry the **axioms** about:

- `SimulationExperiment` and `SimulationStudy`
- `Network` and `DynamicalSystem`
- `Software` used to specify and run **Dynamical Network Models**

Loaded via `owlready2` from `tvbo.ontology`.

## Pillar 3 — Python package

The `tvbo` Python package does four things:

1. **Specifies models** — `Dynamics`, `Equation`, `Network`, etc. as Python objects or YAML files.
2. **Loads** `SimulationExperiment`s, `SimulationStudy`s, and all other ontology-backed classes.
3. **Generates code** for all supported backends (JAX, NumPy, TVB, Julia/DifferentialEquations.jl, PyRates, ModelingToolkit, NetworkDynamics, NeuroML, …).
4. **Executes** the generated code and runs experiments against the chosen backend.

Entry points: `from tvbo import Dynamics, SimulationExperiment`. See the `writing-models` and `running-simulations` skills for the everyday workflows.

## Platform — [tvbo.charite.de](https://tvbo.charite.de)

An Odoo-based management layer over the ontology. Users can:

- Display knowledge stored in the ontology and the curated models.
- Retrieve knowledge programmatically.
- Set up their own models, store and share `SimulationExperiment`s / `SimulationStudy`s (public by default; private with an API key).
- Load shared experiments back into Python.

A discussion / Q&A layer for the community is on the roadmap. See the `platform` skill for client-side usage.

## When to use TVBO

- You want to describe a neural-mass / dynamical-network model **once** and run it on multiple simulator backends.
- You want your models to be machine-readable, citable, and shareable (the ontology + LinkML schema make this possible).
- You want a structured way to bundle experiments and studies for reproducibility.

## Installation

```bash
pip install tvbo            # core
pip install tvbo[jax]       # JAX backend
pip install tvbo[tvb]       # The Virtual Brain backend
pip install tvbo[pyrates]   # PyRates backend
pip install tvbo[julia]     # juliacall + Julia backends
pip install tvbo[all]       # everything
```

Documentation: <https://virtual-twin.github.io/tvbo/>
