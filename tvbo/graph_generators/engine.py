"""Generic eager procedure engine for TVBO graph generators.

Interprets the symbolic ``procedure:`` block of a curated ``GraphGenerator``
YAML — an ordered DAG of named intermediates plus ``output:`` expressions — and
evaluates it once, at ``Network`` load time, in numpy, producing a concrete
adjacency matrix. This is the engine that *replaces* the deleted per-generator
Python materialisers (``builtins.py``): a generator is now **pure YAML**
(its ``procedure:`` block) and this one engine is its numpy/eager interpreter.

See ``dev/GenericProcedureEngine.md`` for the full design.

Eager evaluation uses a restricted ``eval`` over a fixed numpy *primitive
namespace* (``__builtins__`` stripped): the procedure's array expressions use
ordinary Python operator / indexing / keyword-argument semantics on numpy
arrays — exactly the syntax the curated procedures are written in
(``sample(dist, shape=(n, n), seed=seed)``, ``M[M != 0]``, ``Uniform(0, 1)``).
The procedure YAMLs are trusted, curated database artefacts, never user input.
The *codegen* mode — emitting the **same** primitive set into jax / julia source
via the printer tables in ``tvbo/codegen/code.py`` — is deferred (Stage 3).

RNG contract (``GenericProcedureEngine.md`` §4)
----------------------------------------------
A generator with a fixed ``seed`` is deterministic *within numpy*. Any seed
offsets a procedure needs (e.g. ``seed`` for the weights, ``seed + 1`` for the
sparsity mask) are written **explicitly** in the ``procedure:`` so every backend
derives sub-streams the same structural way. Cross-backend bit-identical
sampling is *not* guaranteed (numpy PCG64 ≠ jax Threefry); within-backend
reproducibility and statistical equivalence are. A missing seed defaults to 0
so ``seed + 1`` stays well-defined and reproducible.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Distribution descriptors — let a procedure write `Uniform(0, 1)` inline.     #
# --------------------------------------------------------------------------- #
class _Dist:
    """Lightweight inline distribution descriptor (e.g. ``Uniform(0, 1)``)."""

    __slots__ = ("name", "params")

    def __init__(self, name: str, **params: float) -> None:
        self.name = name
        self.params = params


def _Normal(mean: float = 0.0, std: float = 1.0) -> _Dist:
    return _Dist("normal", mean=mean, std=std)


def _Uniform(lo: float = 0.0, hi: float = 1.0) -> _Dist:
    return _Dist("uniform", lo=lo, hi=hi)


def _LogNormal(mu: float = 0.0, sigma: float = 1.0) -> _Dist:
    return _Dist("lognormal", mu=mu, sigma=sigma)


def _Beta(a: float = 1.0, b: float = 1.0) -> _Dist:
    return _Dist("beta", a=a, b=b)


def _dist_name_and_getter(dist: Optional[Any]):
    """Normalise any distribution form to ``(name, getter(key, default))``.

    Accepts a LinkML ``Distribution`` (``dist.name`` + ``dist.parameters[k].value``),
    an inline ``_Dist`` descriptor, or ``None`` (defaults to a standard Normal).
    """
    if dist is None:
        return "normal", (lambda key, default: float(default))
    if isinstance(dist, _Dist):
        return dist.name.lower(), (lambda key, default: float(dist.params.get(key, default)))
    name = (getattr(dist, "name", None) or "Normal").lower()
    params = getattr(dist, "parameters", None) or {}

    def getter(key: str, default: float) -> float:
        p = params.get(key)
        v = getattr(p, "value", None) if p is not None else None
        return float(v if v is not None else default)

    return name, getter


def sample(dist: Optional[Any], shape=None, seed: Optional[int] = None) -> np.ndarray:
    """Sample ``shape`` values from a distribution (LinkML / inline / ``None``→Normal)."""
    rng = np.random.default_rng(seed)
    shp = tuple(shape) if shape is not None else ()
    name, get = _dist_name_and_getter(dist)
    if name == "normal":
        return rng.normal(get("mean", 0.0), get("std", 1.0), size=shp)
    if name == "uniform":
        return rng.uniform(get("lo", 0.0), get("hi", 1.0), size=shp)
    if name == "lognormal":
        return rng.lognormal(get("mu", 0.0), get("sigma", 1.0), size=shp)
    if name == "beta":
        return rng.beta(get("a", 1.0), get("b", 1.0), size=shp)
    raise ValueError(f"sample(): unsupported distribution {name!r}")


# --------------------------------------------------------------------------- #
# Array / graph primitives.                                                    #
# --------------------------------------------------------------------------- #
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


def permute(arr, seed: Optional[int] = None) -> np.ndarray:
    """Random permutation of ``arr`` (reproducible for a fixed ``seed``)."""
    return np.random.default_rng(seed).permutation(np.asarray(arr))


def nonzero_positions(M) -> np.ndarray:
    """Row-major (i, j) indices of the non-zero entries of ``M``.

    Order matches numpy boolean extraction ``M[M != 0]``, so values extracted
    that way scatter back to the same positions.
    """
    return np.argwhere(np.asarray(M) != 0)


def scatter(base, positions, values) -> np.ndarray:
    """Return a copy of ``base`` with ``values`` written at ``positions``."""
    out = np.array(base, dtype=float)
    positions = np.asarray(positions)
    values = np.asarray(values)
    if positions.ndim == 2 and positions.shape[1] == 2:
        out[positions[:, 0], positions[:, 1]] = values
    elif positions.size:
        out[tuple(positions.T)] = values
    return out


def maslov_sneppen(M, seed: Optional[int] = None, n_swaps_per_edge: int = 10) -> np.ndarray:
    """Degree-preserving edge-swap null model (Maslov–Sneppen) on a weighted matrix.

    A reusable graph *primitive* (not generator-specific Python): repeatedly
    swaps two directed edges (i→j), (k→l) to (i→l), (k→j) when the targets are
    free, preserving per-row out-degree while rerandomising topology; weights
    ride along with their edges.
    """
    W = np.array(M, dtype=float)
    rng = np.random.default_rng(seed)
    sources, targets = np.nonzero(W)
    if len(sources) < 2:
        return W
    for _ in range(int(n_swaps_per_edge * len(sources))):
        a, b = rng.integers(0, len(sources), size=2)
        if a == b:
            continue
        i, j = sources[a], targets[a]
        k, l = sources[b], targets[b]
        if len({i, j, k, l}) < 4:
            continue
        if W[i, l] != 0 or W[k, j] != 0:
            continue
        W[i, l], W[k, j] = W[i, j], W[k, l]
        W[i, j] = W[k, l] = 0.0
        targets[a], targets[b] = l, j
    return W


def require(condition: bool, message: str) -> bool:
    """Declarative guard usable in a procedure step; raises if ``condition`` is false."""
    if not condition:
        raise ValueError(message)
    return True


def _primitive_namespace() -> dict:
    """The numpy primitive table the eager evaluator runs procedures against."""
    return {
        # sampling + inline distributions
        "sample": sample,
        "Normal": _Normal,
        "Uniform": _Uniform,
        "LogNormal": _LogNormal,
        "Beta": _Beta,
        # linear algebra / reductions
        "eigvals": np.linalg.eigvals,
        "abs": np.abs,
        "max": np.max,
        "min": np.min,
        "sum": np.sum,
        "mean": np.mean,
        "sqrt": np.sqrt,
        "exp": np.exp,
        "log": np.log,
        # array construction / shape
        "zeros": lambda shp: np.zeros(tuple(shp) if np.iterable(shp) else (int(shp),)),
        "shape": lambda M: np.asarray(M).shape,
        # graph primitives
        "load_matrix": load_matrix,
        "permute": permute,
        "nonzero_positions": nonzero_positions,
        "scatter": scatter,
        "maslov_sneppen": maslov_sneppen,
        # control
        "require": require,
        "np": np,
    }


def _eval(rhs: str, env: dict) -> Any:
    """Evaluate a single procedure expression against ``env`` (no Python builtins)."""
    g = {"__builtins__": {}}
    g.update(env)
    return eval(rhs, g)  # noqa: S307 — trusted curated procedure expression


def materialize(
    procedure: dict,
    params: dict,
    seed: Optional[int] = None,
    declared_params=None,
) -> dict:
    """Evaluate a ``procedure:`` block to ``{weights[, lengths, node_parameters]}``.

    Parameters
    ----------
    procedure : dict
        The generator's ``procedure`` block: ``{steps: [{name, equation.rhs}],
        output: {weights: {equation.rhs}, ...}}``.
    params : dict
        Concrete parameter values (``n``, ``sparsity``, ``source``,
        ``weight_distribution``, …) supplied by the GraphGenerator.
    seed : int | None
        Fallback seed when ``params`` carries none (defaults to 0 so seed
        arithmetic in the procedure stays well-defined and reproducible).
    declared_params : Mapping[str, dict] | Iterable[str] | None
        The generator's declared parameters. Any not present in ``params`` are
        bound so the procedure resolves them to a default rather than NameError:
        to the parameter's ``ifabsent`` value when ``declared_params`` is a
        mapping of specs (e.g. ``preserve`` → ``binary_mask``), otherwise to
        ``None`` (e.g. an optional ``weight_distribution`` → standard Normal).
    """
    env = _primitive_namespace()
    if isinstance(declared_params, dict):
        for name, spec in declared_params.items():
            default = spec.get("ifabsent") if isinstance(spec, dict) else None
            env.setdefault(name, default)
    else:
        for name in (declared_params or []):
            env.setdefault(name, None)
    env.update(params)
    if env.get("seed") is None:
        env["seed"] = 0 if seed is None else seed

    for step in (procedure.get("steps") or []):
        name = step.get("name")
        rhs = (step.get("equation") or {}).get("rhs")
        if name and rhs is not None:
            env[name] = _eval(rhs, env)

    out: dict = {}
    for key, spec in (procedure.get("output") or {}).items():
        rhs = (spec.get("equation") or {}).get("rhs")
        if rhs is not None:
            out[key] = _eval(rhs, env)
    if "weights" in out:
        out["weights"] = np.asarray(out["weights"], dtype=float)
    return out


# --------------------------------------------------------------------------- #
# Public convenience: materialise a named curated generator directly.          #
# --------------------------------------------------------------------------- #
def _load_generator_entry(name: str) -> dict:
    import yaml

    from tvbo.data.registry import resolve as registry_resolve

    with open(registry_resolve("GraphGenerator", str(name))) as f:
        return yaml.safe_load(f) or {}


def run_generator(name: str, params: dict, seed: Optional[int] = None) -> dict:
    """Materialise a curated generator by name from its YAML ``procedure:`` block.

    Convenience entry point for scripts/notebooks (e.g. building a shuffled-SC
    control). ``Network.resolve()`` uses the same engine internally.
    """
    entry = _load_generator_entry(name)
    procedure = entry.get("procedure")
    if not procedure:
        raise ValueError(f"GraphGenerator {name!r} has no `procedure:` block to evaluate.")
    return materialize(procedure, params, seed=seed, declared_params=(entry.get("parameters") or {}))
