"""TVBO graph generators — declarative typed DAGs, one resolver.

A curated ``GraphGenerator`` under ``tvbo/database/graph_generators/`` is defined by its ``procedure:`` block: an ordered DAG of typed steps whose options are schema fields. :mod:`tvbo.graph_generators.procedural` resolves that DAG to SymPy and renders it through the printer tables in ``tvbo/codegen/code.py``, so eager construction at ``Network`` load time and emitted backend source are the same expressions rendered twice. There are no per-generator Python materialisers (see ``dev/GenericProcedureEngine.md``).

Two kinds of generator sit outside the DAG, both through the standard ``bindings`` slot: library wrappers (Graphs.jl / NetworkX families) and the documented Callable exception below, for a construction the backend-independent primitive set genuinely cannot express.

The helper below is a *thin convenience wrapper* over the resolver (it holds no generation algorithm) for scripts and notebooks that want a matrix directly.
"""

from __future__ import annotations

from typing import Any, Optional

from .catalog import load_matrix, run_generator


def random_reservoir(
    n_nodes: int,
    sparsity: float = 0.1,
    spectral_radius: float = 0.95,
    weight_distribution: Any | None = None,
    seed: int | None = None,
) -> dict:
    """Materialise a ``RandomReservoir`` adjacency through the typed-DAG resolver."""
    params: dict = {"n_nodes": n_nodes, "sparsity": sparsity, "spectral_radius": spectral_radius}
    # Left unset, the generator's `raw` step falls back to the standard Normal it declares; binding None here would look like a supplied-but-empty distribution.
    if weight_distribution is not None:
        params["weight_distribution"] = weight_distribution
    return run_generator("RandomReservoir", params, seed=seed)


def weight_shuffle(source: str, preserve: str = "binary_mask", seed: int | None = None) -> dict:
    """Materialise a ``WeightShuffle`` null-model adjacency: permute the non-zero weights.

    This is the documented exception to the typed-DAG rule (see ``dev/GenericProcedureEngine.md`` §5): a masked extract, a permutation and a scatter are not expressible in the backend-independent primitive set. Boolean-mask extraction in particular cannot survive expression parsing at all — ``M[M != 0]`` evaluates its comparison to a plain Python ``True`` before an expression tree is ever built. So the algorithm lives here as ordinary Python, reached through the generator's ``bindings.python`` binding like any other library wrapper.

    ``preserve='binary_mask'`` keeps the ``{0, nonzero}`` pattern and permutes the weight values among their existing positions, so density and topology are held fixed while the weight-to-edge assignment is randomised.

    Args:
        source: IRI, path or database name of the reference Network to shuffle.
        preserve: Structural property to hold fixed. Only ``binary_mask`` is implemented.
        seed: PRNG seed; ``None`` means 0, matching the generator's declared default.

    Returns:
        ``{"weights": ndarray}`` — the permuted adjacency matrix.
    """
    if preserve != "binary_mask":
        raise ValueError(
            f"WeightShuffle: preserve={preserve!r} is not implemented. Only "
            f"'binary_mask' is available, and silently falling back to it would produce "
            f"a null model that does not control what the caller asked it to control — "
            f"invalidating the comparison the null model exists to support."
        )
    import numpy as np

    matrix = np.asarray(load_matrix(source), dtype=float)
    nonzero = matrix != 0
    # argwhere and boolean extraction are both row-major, so the permuted values land back on the positions they were taken from, in the same order.
    positions = np.argwhere(nonzero)
    permuted = np.random.default_rng(0 if seed is None else seed).permutation(matrix[nonzero])
    shuffled = np.zeros(matrix.shape, dtype=float)
    shuffled[positions[:, 0], positions[:, 1]] = permuted
    return {"weights": shuffled}


__all__ = ["random_reservoir", "weight_shuffle", "load_matrix", "run_generator"]
