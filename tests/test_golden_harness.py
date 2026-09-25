"""The golden harness refuses a reference that cannot fail.

A corpus is only worth its runtime if each case would go red when the thing it freezes changes. A case whose output is empty, entirely non-finite, or constant does not: every later run compares equal to it whatever the code did, and the corpus reports a pass. That is not hypothetical — ``JR_MEG``'s ``simulated_psd`` was frozen at shape ``(0, 1, 129)`` with ``peak_frequency`` identically ``0``, and every run since compared empty to empty and passed, for as long as the bug existed.

These cases pin the guard behaviourally, in both directions, because writing a dead reference and comparing against one already committed are different failures and each is silent on its own.
"""

from __future__ import annotations

import numpy as np
import pytest

from .golden import GoldenCorpus, array_discriminates


@pytest.mark.parametrize(
    "values,expected",
    [
        (np.zeros((0, 3)), "empty"),
        (np.full((4, 2), np.nan), "finite"),
        (np.full((4, 2), 7.0), "every element"),
        (np.array([[np.nan, np.nan], [np.nan, np.nan]]), "finite"),
        (np.array(["a", "a", "a"]), "every element"),
    ],
)
def test_a_degenerate_array_says_why(values, expected):
    """Each degeneracy is named in its own terms, so the failure message points at the cause."""
    reason = array_discriminates(values)
    assert reason is not None and expected in reason, reason


@pytest.mark.parametrize(
    "values",
    [
        np.arange(6.0).reshape(3, 2),
        np.array([np.nan, 1.0, 2.0]),  # partly non-finite still discriminates on the rest
        np.array([0.0, 0.0, 1e-30]),  # a tiny spread is a spread
    ],
)
def test_an_array_that_varies_discriminates(values):
    assert array_discriminates(values) is None


def _corpus(tmp_path, **kwargs) -> GoldenCorpus:
    """A corpus over plain arrays, comparing by exact equality."""
    return GoldenCorpus(
        tmp_path,
        ".npy",
        write=lambda path, produced: np.save(path, produced),
        read=np.load,
        compare=lambda produced, expected: None if np.array_equal(produced, expected) else "differs",
        discriminates=array_discriminates,
        **kwargs,
    )


def test_a_dead_reference_is_never_written(tmp_path):
    """Regeneration refuses the artifact instead of freezing it, so the case cannot be born dead."""
    corpus = _corpus(tmp_path)
    with pytest.raises(pytest.fail.Exception, match="refusing to freeze"):
        corpus.check("dead", np.zeros((0, 4)), regenerate=True, what="output")
    assert not corpus.path("dead").exists(), "a refused reference must not reach the corpus"


def test_a_live_reference_is_written(tmp_path):
    """The guard is not in the way of an ordinary re-baseline."""
    corpus = _corpus(tmp_path)
    with pytest.raises(pytest.skip.Exception, match="regenerated"):
        corpus.check("live", np.arange(6.0), regenerate=True, what="output")
    assert corpus.path("live").exists()


def test_comparing_two_dead_sides_fails_rather_than_passes(tmp_path):
    """The case that made this necessary: empty against empty, which every comparison calls a match."""
    corpus = _corpus(tmp_path)
    dead = np.zeros((0, 4))
    np.save(corpus.path("dead"), dead)
    assert corpus._compare(dead, dead) is None, "the comparison itself sees no difference — that is the point"
    with pytest.raises(pytest.fail.Exception, match="cannot discriminate"):
        corpus.check("dead", dead, regenerate=False, what="output")


def test_a_dead_reference_under_a_live_output_names_the_reference(tmp_path):
    """Which side is degenerate decides what to fix, so the message says which."""
    corpus = _corpus(tmp_path)
    np.save(corpus.path("case"), np.zeros((0, 4)))
    with pytest.raises(pytest.fail.Exception, match="its reference is degenerate"):
        corpus.check("case", np.arange(6.0), regenerate=False, what="output")


def test_a_dead_output_under_a_live_reference_names_the_output(tmp_path):
    corpus = _corpus(tmp_path)
    np.save(corpus.path("case"), np.arange(6.0))
    with pytest.raises(pytest.fail.Exception, match="its output is degenerate"):
        corpus.check("case", np.full(6, np.nan), regenerate=False, what="output")


def test_a_named_case_is_allowed_to_be_degenerate(tmp_path):
    """The escape hatch is per case and carries its reason, so the claim sits in the diff."""
    corpus = _corpus(tmp_path, degenerate_ok={"quiescent": "the model is at rest by construction"})
    flat = np.zeros((5, 2))
    np.save(corpus.path("quiescent"), flat)
    corpus.check("quiescent", flat, regenerate=False, what="output")


def test_a_corpus_without_the_predicate_is_unchanged(tmp_path):
    """Opting in is per corpus: one that declares no predicate behaves exactly as before."""
    corpus = GoldenCorpus(
        tmp_path,
        ".npy",
        write=lambda path, produced: np.save(path, produced),
        read=np.load,
        compare=lambda produced, expected: None if np.array_equal(produced, expected) else "differs",
    )
    dead = np.zeros((0, 4))
    np.save(corpus.path("dead"), dead)
    corpus.check("dead", dead, regenerate=False, what="output")
