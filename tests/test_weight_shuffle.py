"""``WeightShuffle`` — the documented Callable exception to the pure-YAML rule.

A masked extract, a permutation and a scatter are not expressible in the backend-independent primitive set: ``M[M != 0]`` cannot survive expression parsing at all,
because the comparison evaluates to a plain Python ``True`` before a tree is built. Rather than invent a contorted primitive that only this generator would use, the algorithm stays
as Python — which ``dev/GenericProcedureEngine.md`` §5 explicitly preserves for exactly this case.

These pin the null model's actual guarantees (pattern preserved, weight multiset preserved, topology rerandomised) and the semantics inherited from the procedure it
replaces, so the swap is behaviour-preserving rather than merely plausible.
"""

import numpy as np
import pytest

from tvbo.graph_generators import weight_shuffle

M = np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 4.0], [0.0, 5.0, 0.0]])


@pytest.fixture
def shuffled(monkeypatch):
    """Bind the source matrix directly, so the null model is tested without I/O."""

    def _run(seed=0, matrix=M):
        monkeypatch.setattr("tvbo.graph_generators.load_matrix", lambda _source: matrix, raising=True)
        return weight_shuffle("irrelevant://source", seed=seed)["weights"]

    return _run


def test_matches_the_procedure_it_replaces(shuffled):
    """Bit-identical to the engine's step-for-step semantics, not merely similar.

    Reference values are the procedure's own primitives applied in order: boolean extract
    -> permute(seed) -> scatter onto argwhere positions.
    """
    np.testing.assert_array_equal(
        shuffled(seed=0),
        np.array([[0.0, 4.0, 0.0], [2.0, 0.0, 3.0], [0.0, 5.0, 0.0]]),
    )
    np.testing.assert_array_equal(
        shuffled(seed=7),
        np.array([[0.0, 2.0, 0.0], [4.0, 0.0, 3.0], [0.0, 5.0, 0.0]]),
    )


def test_binary_mask_pattern_is_preserved(shuffled):
    """The whole point of the null model: topology fixed, weight assignment randomised."""
    np.testing.assert_array_equal(shuffled(seed=3) != 0, M != 0)


def test_weight_multiset_is_preserved(shuffled):
    out = shuffled(seed=5)
    np.testing.assert_array_equal(np.sort(out[out != 0]), np.sort(M[M != 0]))


def test_is_reproducible_and_seed_sensitive(shuffled):
    np.testing.assert_array_equal(shuffled(seed=11), shuffled(seed=11))
    # A 5-element permutation collides across seeds often enough to be flaky if asserted on one pair, so require only that some seed in a small sweep differs.
    assert any(not np.array_equal(shuffled(seed=0), shuffled(seed=s)) for s in range(1, 8))


def test_none_seed_matches_seed_zero(shuffled):
    """The engine defaulted a missing seed to 0; the replacement must not drift from it."""
    np.testing.assert_array_equal(shuffled(seed=None), shuffled(seed=0))


def test_an_unimplemented_preserve_mode_is_rejected(monkeypatch):
    """Silently falling back would return a null model controlling the wrong property.

    `degree` and `weight_distribution` are documented in the generator's parameters but have no implementation; a caller asking for a degree-preserving null must not be
    handed a binary-mask one.
    """
    monkeypatch.setattr("tvbo.graph_generators.load_matrix", lambda _s: M, raising=True)
    with pytest.raises(ValueError, match="not implemented"):
        weight_shuffle("irrelevant://source", preserve="degree", seed=0)


def test_an_all_zero_matrix_shuffles_to_itself(shuffled):
    """No non-zero entries to permute — must not raise on the empty index arrays."""
    out = shuffled(seed=0, matrix=np.zeros((3, 3)))
    np.testing.assert_array_equal(out, np.zeros((3, 3)))
