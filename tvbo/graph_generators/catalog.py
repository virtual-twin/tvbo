"""The curated GraphGenerator catalog: entry lookup, declared defaults, reference matrices.

What is left here is the residue that no printer should ever emit. Graph construction itself lives in :mod:`tvbo.graph_generators.procedural`, which resolves a generator's typed DAG to SymPy and renders it through the printer tables in ``tvbo/codegen/code.py`` — one primitive definition per backend. This module used to carry a second, numpy-only implementation of those same primitives (sampling, reductions, linear algebra) behind a restricted ``eval``; that table is gone, because two implementations of one vocabulary can only ever agree by coincidence, and the disagreement would show up as a network that differs between a local run and a swept one.
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

    m = net.matrix("weight", format="dense")
    if m is not None:
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


def connection_density(weights: np.ndarray) -> float:
    """The fraction of *weights*' off-diagonal entries that are nonzero: the probability of an edge between two distinct nodes, which for a symmetric matrix is the density of the undirected graph it holds.

    Raises:
        ValueError: *weights* has fewer than two nodes, so no pair of distinct nodes to take a density over.
    """
    n = weights.shape[0]
    if n < 2:
        raise ValueError(f"a {n}-node matrix has no pair of distinct nodes to take a connection density over.")
    off_diagonal = ~np.eye(n, dtype=bool)
    return float(np.count_nonzero(weights[off_diagonal]) / off_diagonal.sum())


def library_weights(generator: Any, n_nodes: int) -> np.ndarray:
    """The weight matrix a curated library generator builds, target-by-source like every tvbo connectome: ``W[i, j]`` is the edge ``j → i``, 1 where it exists.

    A library generator (Barabási–Albert, Watts–Strogatz, …) has no ``procedure:`` to resolve; its curated entry maps each backend onto a library constructor through ``bindings``, and this is the Python one, the ``networkx`` binding, called with the arguments it lists in order. The size is *n_nodes*, the network's, bound to the entry's ``n``; the other arguments are the generator's ``parameters`` over the entry's declared defaults, an ``integer`` one cast to ``int`` since a recipe's number reaches here as a float. A ``density_from`` parameter, which ErdosRenyi's entry declares, names a Network (an IRI, a curated name or a file, as `load_matrix` reads it) whose connection density (`connection_density`) is the edge probability ``p``, so the random graph matches the source's density. A generator whose entry declares a ``seed`` is drawn with its own ``seed``, or 0 where it states none, as `procedural.materialize` draws one, so every call builds the same graph.

    Args:
        generator: The network's ``GraphGenerator``, naming a curated ``type``.
        n_nodes: The number of nodes the network declares.

    Returns:
        The ``(n_nodes, n_nodes)`` weight matrix; symmetric unless the generator is ``directed``.

    Raises:
        ValueError: The entry has no ``networkx`` binding, the generator gives both ``p`` and ``density_from``, it gives no value for an argument the binding takes, the binding cannot build a ``directed`` graph the generator asks for, or it builds a graph whose node count is not *n_nodes*.
    """
    import importlib
    import inspect

    import networkx as nx

    name = str(generator.type)
    entry = _load_generator_entry(name)
    binding = (entry.get("bindings") or {}).get("networkx")
    if not binding:
        raise ValueError(f"GraphGenerator {name!r} declares no `networkx` binding, so no graph can be built for it in Python.")
    declared = entry.get("parameters") or {}
    given = {
        str(getattr(parameter, "name", None) or key): parameter.value
        for key, parameter in (getattr(generator, "parameters", None) or {}).items()
        if getattr(parameter, "value", None) is not None
    }
    values = {**declared_defaults(entry), **given}
    source = values.pop("density_from", None) if "density_from" in declared else None
    if source is not None:
        if "p" in given:
            raise ValueError(
                f"GraphGenerator {name!r} gives both p and density_from {source!r}, and density_from computes p: declare one."
            )
        values["p"] = connection_density(load_matrix(str(source)))
    values["n"] = int(n_nodes)
    missing = [a for a in binding.get("args", []) if a not in values]
    if missing:
        raise ValueError(
            f"GraphGenerator {name!r} gives no value for {', '.join(missing)}, which its networkx binding {binding['callable']} takes."
        )
    integer = {key for key, spec in declared.items() if isinstance(spec, Mapping) and spec.get("datatype") == "integer"}
    args = [int(values[a]) if a in integer else values[a] for a in binding.get("args", [])]
    build = getattr(importlib.import_module(binding["library"]), binding["callable"])
    accepts = inspect.signature(build).parameters
    kwargs = {"seed": int(getattr(generator, "seed", None) or 0)} if "seed" in declared else {}
    if getattr(generator, "directed", False):
        if "directed" not in accepts:
            raise ValueError(
                f"GraphGenerator {name!r} is declared directed, and its networkx binding {binding['callable']} builds undirected graphs only."
            )
        kwargs["directed"] = True
    graph = build(*args, **kwargs)
    if graph.number_of_nodes() != n_nodes:
        raise ValueError(
            f"GraphGenerator {name!r}'s networkx binding {binding['callable']}{tuple(args)} builds {graph.number_of_nodes()} nodes, "
            f"and the network declares {n_nodes}."
        )
    return nx.to_numpy_array(graph, nodelist=range(n_nodes), weight=None).T


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
