"""TVBO graph generators — pure-YAML procedures, one generic engine.

Each curated ``GraphGenerator`` under ``tvbo/database/graph_generators/`` is
defined entirely by its symbolic ``procedure:`` block. The generic engine
(:mod:`tvbo.graph_generators.engine`) interprets that block in numpy at
``Network`` load time — there are no per-generator Python materialisers (the
old ``builtins.py`` is gone; see ``dev/GenericProcedureEngine.md``).

The helpers below are *thin convenience wrappers* over the engine (they hold no
generation algorithm) for scripts and notebooks that want a matrix directly.
"""

from __future__ import annotations

from typing import Any, Optional

from .engine import load_matrix, run_generator


def random_reservoir(n: int, sparsity: float = 0.1, spectral_radius: float = 0.95,
                     weight_distribution: Optional[Any] = None,
                     seed: Optional[int] = None) -> dict:
    """Materialise a ``RandomReservoir`` adjacency via the generic engine."""
    return run_generator(
        "RandomReservoir",
        {"n": n, "sparsity": sparsity, "spectral_radius": spectral_radius,
         "weight_distribution": weight_distribution},
        seed=seed,
    )


def weight_shuffle(source: str, preserve: str = "binary_mask",
                   seed: Optional[int] = None) -> dict:
    """Materialise a ``WeightShuffle`` null-model adjacency: permute the non-zero weights.

    This is the documented exception to the pure-YAML rule (see
    ``dev/GenericProcedureEngine.md`` §5): a masked extract, a permutation and a scatter
    are not expressible in the backend-independent primitive set. Boolean-mask extraction
    in particular cannot survive expression parsing at all — ``M[M != 0]`` evaluates its
    comparison to a plain Python ``True`` before an expression tree is ever built. So the
    algorithm lives here as ordinary Python rather than being bent into a printer
    vocabulary that no other generator would want.

    ``preserve='binary_mask'`` keeps the ``{0, nonzero}`` pattern and permutes the weight
    values among their existing positions, so density and topology are held fixed while
    the weight-to-edge assignment is randomised.

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
            f"'binary_mask' is available; 'degree' and 'weight_distribution' are "
            f"declared in the generator's parameter documentation but have no "
            f"implementation, and silently falling back to binary_mask would produce a "
            f"null model that does not control what the caller asked it to control."
        )
    import numpy as np

    matrix = np.asarray(load_matrix(source), dtype=float)
    nonzero = matrix != 0
    # argwhere and boolean extraction are both row-major, so the permuted values land
    # back on the positions they were taken from, in the same order.
    positions = np.argwhere(nonzero)
    permuted = np.random.default_rng(0 if seed is None else seed).permutation(matrix[nonzero])
    shuffled = np.zeros(matrix.shape, dtype=float)
    shuffled[positions[:, 0], positions[:, 1]] = permuted
    return {"weights": shuffled}


__all__ = ["random_reservoir", "weight_shuffle", "load_matrix", "run_generator"]
