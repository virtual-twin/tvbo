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
    """Materialise a ``WeightShuffle`` null-model adjacency via the generic engine."""
    return run_generator("WeightShuffle", {"source": source, "preserve": preserve}, seed=seed)


__all__ = ["random_reservoir", "weight_shuffle", "load_matrix", "run_generator"]
