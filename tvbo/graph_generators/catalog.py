"""The curated GraphGenerator catalog: entry lookup, declared defaults, reference matrices.

What is left here is the residue that no printer should ever emit. Graph construction itself lives in :mod:`tvbo.graph_generators.procedural`, which resolves a generator's typed DAG to SymPy and renders it through the printer tables in ``tvbo/codegen/code.py`` — one primitive definition per backend. This module used to carry a second, numpy-only implementation of those same primitives (sampling, reductions, linear algebra) behind a restricted ``eval``; that table is gone, because two implementations of one vocabulary can only ever agree by coincidence, and the disagreement would show up as a network that differs between a local run and a swept one.

See ``dev/GenericProcedureEngine.md`` for the full design.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def load_matrix(source: str) -> np.ndarray:
    """Resolve ``source`` (IRI / path / bare DB name) into a 2-D adjacency matrix."""
    # Local import avoids a circular dependency with classes.network.
    from tvbo.classes.network import Network

    resolved = Network._resolve_network_iri(source)
    if resolved is not None:
        net = Network.from_file(resolved)
    elif str(source).endswith((".yaml", ".yml", ".h5")):
        net = Network.from_file(source)
    else:
        net = Network.from_db(source)
    net._resolve()

    for attr in ("weights_matrix", "weights", "weight"):
        m = getattr(net, attr, None)
        if m is None:
            continue
        if callable(m):
            try:
                m = m()
            except TypeError:
                continue
        arr = np.asarray(m)
        if arr.ndim == 2:
            return arr
    raise RuntimeError(f"Could not extract a 2-D weights matrix from source {source!r}")


def _load_generator_entry(name: str) -> dict:
    import yaml

    from tvbo.data.registry import resolve as registry_resolve

    with open(registry_resolve("GraphGenerator", str(name))) as f:
        return yaml.safe_load(f) or {}


def declared_defaults(entry: Mapping[str, Any]) -> dict:
    """Default values from a curated entry's ``parameters:`` interface block.

    A curated entry declares each parameter (``datatype``, ``description``, ``default``) while a concrete Network supplies its ``value``. Only the defaults cross over into evaluation — the declaration itself is an interface, not a value, and binding it as one would hand a step a ``{'datatype': ...}`` dict where it expects a number.
    """
    defaults = {}
    for name, spec in (entry.get("parameters") or {}).items():
        if not isinstance(spec, Mapping):
            continue
        # `default` is the Parameter slot; `ifabsent` is the LinkML spelling any entry not yet migrated may still carry. Reading only one would silently drop the other's default and hand the step an unset parameter.
        value = spec.get("default", spec.get("ifabsent"))
        if value is not None:
            defaults[name] = value
    return defaults


def run_generator(name: str, params: dict, seed: int | None = None) -> dict:
    """Materialise a curated generator by name from its typed ``procedure:`` DAG.

    Convenience entry point for scripts and notebooks. ``Network._resolve`` goes through the same resolver, so a generator built here matches the one a recipe builds value for value.
    """
    from tvbo.graph_generators.procedural import materialize

    entry = _load_generator_entry(name)
    procedure = entry.get("procedure")
    if not procedure:
        raise ValueError(f"GraphGenerator {name!r} has no `procedure:` block to evaluate.")
    return materialize(procedure, {**declared_defaults(entry), **params}, seed=seed)
