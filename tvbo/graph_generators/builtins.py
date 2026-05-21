"""Reference Python materialisers for built-in TVBO graph generators.

These are reference implementations called by `Network.resolve()` at YAML
load time. The symbolic `procedure:` block in each generator's YAML is
the backend-independent specification; this module is the Python
realisation of that specification.

All materialisers return a dict with at least a ``weights`` key:
the constructed adjacency matrix as a NumPy array. Optional keys
(``lengths``, ``node_parameters``) follow the same convention used by
`Network._resolve_from_graph_generator()` for `builder: Callable`.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _resolve_distribution(spec: Optional[Any], rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Sample from a Distribution spec.

    Accepts (a) a dict-like ``{'type': 'Normal', 'mean': 0, 'std': 1}`` or
    similar, (b) the corresponding LinkML-generated Pydantic object, or
    (c) ``None`` → defaults to standard Normal.
    """
    if spec is None:
        return rng.standard_normal(shape)

    # Extract a plain dict regardless of source representation
    if hasattr(spec, "model_dump"):
        data = spec.model_dump()
    elif isinstance(spec, dict):
        data = spec
    else:
        # Pydantic without model_dump or some other object — best-effort attribute read
        data = {k: getattr(spec, k) for k in ("type", "mean", "std", "lo", "hi") if hasattr(spec, k)}

    dist_type = (data.get("type") or "Normal").lower()

    if dist_type == "normal":
        mean = float(data.get("mean", 0.0))
        std = float(data.get("std", 1.0))
        return rng.normal(loc=mean, scale=std, size=shape)
    if dist_type == "uniform":
        lo = float(data.get("lo", 0.0))
        hi = float(data.get("hi", 1.0))
        return rng.uniform(low=lo, high=hi, size=shape)
    if dist_type == "lognormal":
        mu = float(data.get("mu", data.get("mean", 0.0)))
        sigma = float(data.get("sigma", data.get("std", 1.0)))
        return rng.lognormal(mean=mu, sigma=sigma, size=shape)

    raise ValueError(f"Unsupported weight_distribution type: {dist_type!r}")


def random_reservoir(
    n: int,
    sparsity: float = 0.1,
    spectral_radius: float = 0.95,
    weight_distribution: Optional[Any] = None,
    seed: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Build a sparse random recurrent adjacency rescaled to a target spectral radius.

    Echo State Network substrate (Jaeger 2001). The non-zero entries are
    drawn from ``weight_distribution`` (default: standard Normal), a Bernoulli
    mask with success rate ``sparsity`` retains a fraction of them, and the
    resulting matrix is rescaled so its spectral radius equals
    ``spectral_radius`` — *the* defining hyperparameter of RC.

    Parameters
    ----------
    n : int                       — number of reservoir units (matrix is n×n)
    sparsity : float              — fraction of non-zero entries, in [0, 1]
    spectral_radius : float       — target spectral radius after rescaling
    weight_distribution : Distribution | dict | None
                                  — distribution of the non-zero weights
    seed : int | None             — reproducibility seed (uses two streams:
                                    `seed` for weights, `seed + 1` for the mask)

    Returns
    -------
    dict with key ``weights`` mapped to the (n, n) NumPy array.
    """
    rng_w = np.random.default_rng(seed)
    rng_m = np.random.default_rng(None if seed is None else seed + 1)

    raw = _resolve_distribution(weight_distribution, rng_w, (n, n))
    mask = rng_m.uniform(size=(n, n)) < float(sparsity)
    masked = raw * mask

    rho = float(np.max(np.abs(np.linalg.eigvals(masked))))
    if rho == 0.0:
        # Vanishingly rare for n > 1, but defend the divide
        raise RuntimeError(
            f"RandomReservoir: spectral radius of masked matrix is 0 (n={n}, "
            f"sparsity={sparsity}, seed={seed}). Increase sparsity or n."
        )

    weights = masked * (float(spectral_radius) / rho)
    return {"weights": weights}


def weight_shuffle(
    source: str,
    preserve: str = "binary_mask",
    seed: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Null-model derived adjacency: permute non-zero entries of a source matrix.

    Parameters
    ----------
    source : str (URI or path)
                                  — IRI / path to a TVBO Network whose
                                    weights matrix is the reference. Resolved
                                    via the standard registry (atlas / network
                                    BIDS data).
    preserve : str                — preservation mode:
                                      * ``binary_mask``        — keep the
                                        {zero, non-zero} pattern; permute the
                                        non-zero values among their existing
                                        positions.
                                      * ``degree``             — preserve
                                        per-row degree (rewires under
                                        Maslov-Sneppen swap).
                                      * ``weight_distribution`` — preserve the
                                        multiset of weights; randomise the
                                        binary topology too (Erdős-Rényi over
                                        the same density).
    seed : int | None             — reproducibility seed.

    Returns
    -------
    dict with key ``weights`` mapped to the shuffled (n, n) NumPy array.
    """
    rng = np.random.default_rng(seed)
    W = _load_source_weights(source)

    if preserve == "binary_mask":
        mask = W != 0
        values = W[mask]
        permuted = rng.permutation(values)
        out = np.zeros_like(W)
        out[mask] = permuted
        return {"weights": out}

    if preserve == "degree":
        # Maslov-Sneppen edge-rewire: preserves degree sequence while
        # rerandomising connections. Standard null model for connectomes.
        return {"weights": _maslov_sneppen(W, rng)}

    if preserve == "weight_distribution":
        n = W.shape[0]
        density = float((W != 0).sum()) / float(n * (n - 1))
        binary = rng.uniform(size=(n, n)) < density
        np.fill_diagonal(binary, False)
        values = W[W != 0]
        permuted = rng.permutation(values)
        out = np.zeros_like(W)
        positions = np.argwhere(binary)
        # Trim/extend permuted to match exact number of new non-zero positions
        k = min(len(permuted), len(positions))
        for idx in range(k):
            i, j = positions[idx]
            out[i, j] = permuted[idx]
        return {"weights": out}

    raise ValueError(
        f"WeightShuffle.preserve={preserve!r} not recognised. "
        f"Use one of: binary_mask, degree, weight_distribution."
    )


def _load_source_weights(source: str) -> np.ndarray:
    """Resolve `source` (IRI / path) into a NumPy adjacency matrix."""
    # Local import to avoid circular dependency with classes.network
    from tvbo.classes.network import Network

    net = Network.from_db(source) if not source.endswith((".yaml", ".yml", ".h5")) else Network.from_file(source)
    net._resolve()

    # Network may expose its primary weights matrix via different attrs depending on
    # how it was loaded; try the common ones.
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
    raise RuntimeError(f"Could not extract 2-D weights matrix from source {source!r}")


def _maslov_sneppen(W: np.ndarray, rng: np.random.Generator, n_swaps_per_edge: int = 10) -> np.ndarray:
    """Maslov-Sneppen degree-preserving edge swap on a (possibly weighted) matrix.

    Repeatedly picks two non-zero entries (i, j) and (k, l) and swaps them to
    (i, l) and (k, j), provided the new entries are not already occupied.
    Preserves per-row out-degree exactly when the matrix is treated as a
    directed binary graph; weights are carried along with the edges.
    """
    W = W.copy()
    sources, targets = np.nonzero(W)
    if len(sources) < 2:
        return W
    total_swaps = int(n_swaps_per_edge * len(sources))
    for _ in range(total_swaps):
        a, b = rng.integers(low=0, high=len(sources), size=2)
        if a == b:
            continue
        i, j = sources[a], targets[a]
        k, l = sources[b], targets[b]
        if i == k or j == l or i == l or j == k:
            continue
        if W[i, l] != 0 or W[k, j] != 0:
            continue
        W[i, l] = W[i, j]
        W[k, j] = W[k, l]
        W[i, j] = 0
        W[k, l] = 0
        targets[a] = l
        targets[b] = j
    return W
