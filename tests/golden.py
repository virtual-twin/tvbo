"""Shared machinery for TVBO's golden corpora.

A golden corpus freezes an artifact TVBO promises to produce — generated source, simulation output — and fails when it changes. Every corpus needs the same five behaviours, and this module owns them once so a new corpus is a storage format plus a comparison, not another copy of the harness:

* resolving a case to its reference file,
* re-baselining under ``--regenerate-golden``, including pruning references whose case no
  longer exists,
* refusing to pass when a case has no reference at all,
* refusing a reference that cannot discriminate, at write and at compare,
* reporting a mismatch as a diagnostic rather than an opaque assertion,
* checking that the case set and the reference set describe the same things.

Regeneration deliberately cannot produce a green run: every regenerated case is skipped rather than passed, and :func:`pytest_sessionfinish` in ``conftest`` fails the session. A re-baseline is a change to what TVBO promises, so it must be reviewed and committed on its own — never mistaken for a suite that passed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pytest


def array_discriminates(values) -> str | None:
    """Why an array cannot tell a right answer from a wrong one, or ``None`` when it can.

    A reference earns its place by failing when the thing it freezes changes. An array that is empty, entirely non-finite, or the same value throughout cannot do that: every later run compares equal to it whatever the code did, so the case passes forever while asserting nothing. Offered here so each corpus states degeneracy in its own terms without re-deriving these three.
    """
    values = np.asarray(values)
    if values.size == 0:
        return f"it is empty (shape {values.shape})"
    if values.dtype.kind in "fc" and not np.isfinite(values).any():
        return "no element of it is finite"
    finite = values[np.isfinite(values)] if values.dtype.kind in "fc" else values
    if finite.size and (finite == finite.flat[0]).all():
        return f"every element of it is {finite.flat[0]!r}"
    return None


def text_discriminates(text) -> str | None:
    """Why a frozen text artifact cannot tell a right answer from a wrong one, or ``None``.

    Generated source and serialized records have one degeneracy between them: nothing was produced. A reference of no content matches an empty render exactly, so a backend that stopped emitting would be frozen as agreeing with itself.
    """
    return "it is empty" if not str(text).strip() else None


class GoldenCorpus:
    """One directory of reference files, keyed by case id.

    Args:
        root: Directory holding the reference files. Created on regeneration.
        suffix: Reference file extension, including the leading dot.
        write: ``(path, produced) -> None``; persists a freshly produced artifact.
        read: ``(path) -> expected``; loads a reference for comparison.
        compare: ``(produced, expected) -> str | None``; returns ``None`` when they match, otherwise a human-readable account of the difference.
        discriminates: ``(artifact) -> str | None``; returns ``None`` when the artifact can tell a right answer from a wrong one, otherwise why it cannot. See :func:`array_discriminates`. Omitted, the corpus is not checked for degeneracy.
        degenerate_ok: ``{case_id: why}`` for cases whose output is legitimately empty or constant. The reason is the point: it puts the claim in the diff, where a reviewer can disagree with it.
    """

    def __init__(
        self,
        root: Path,
        suffix: str,
        *,
        write: Callable[[Path, object], None],
        read: Callable[[Path], object],
        compare: Callable[[object, object], str | None],
        discriminates: Callable[[object], str | None] | None = None,
        degenerate_ok: Mapping[str, str] | None = None,
    ):
        self.root = root
        self.suffix = suffix
        self._write = write
        self._read = read
        self._compare = compare
        self._discriminates = discriminates
        self._degenerate_ok = dict(degenerate_ok or {})

    def _degeneracy(self, case_id: str, artifact) -> str | None:
        """Why ``artifact`` cannot serve as a reference for ``case_id``, or ``None``."""
        if self._discriminates is None or case_id in self._degenerate_ok:
            return None
        return self._discriminates(artifact)

    def path(self, case_id: str) -> Path:
        return self.root / f"{case_id}{self.suffix}"

    def check(self, case_id: str, produced, *, regenerate: bool, what: str) -> None:
        """Assert ``produced`` matches the reference for ``case_id``.

        Skips (never passes) when regenerating. Fails with a pointed message when the reference is absent, so a newly added case cannot slip in unreviewed.

        Refuses a degenerate artifact in both directions, because the two failures are different and both are silent. Writing one commits a reference that can never fail, so the case is dead from its first run. Comparing against one that is already committed is how such a case stays dead: empty compares equal to empty, and the corpus reports a pass. Checking on compare is what surfaces the ones already in the corpus, on the next run rather than at an audit nobody schedules.
        """
        reference = self.path(case_id)

        if regenerate:
            reason = self._degeneracy(case_id, produced)
            if reason is not None:
                pytest.fail(
                    f"{case_id}: refusing to freeze this {what} as a reference, because {reason}. "
                    f"A reference that cannot discriminate passes forever while asserting nothing. "
                    f"Fix what produced it, or — if the output is legitimately degenerate — say so by "
                    f"name in the corpus's degenerate_ok, where the claim can be reviewed."
                )
            self.root.mkdir(parents=True, exist_ok=True)
            self._write(reference, produced)
            pytest.skip("regenerated")

        if not reference.is_file():
            pytest.fail(
                f"No reference for {case_id}. A new {what} needs its output reviewed and "
                f"committed: run --regenerate-golden and inspect what it produced."
            )

        expected = self._read(reference)
        for side, artifact in (("reference", expected), ("output", produced)):
            reason = self._degeneracy(case_id, artifact)
            if reason is not None:
                pytest.fail(
                    f"{case_id}: this case cannot discriminate — its {side} is degenerate, because {reason}. "
                    f"Comparing it proves nothing, whatever the comparison reports. Fix what produced it, or "
                    f"name the case in the corpus's degenerate_ok with the reason it is legitimately so."
                )

        difference = self._compare(produced, expected)
        if difference is None:
            return
        pytest.fail(
            f"{case_id}: {what} changed.\n{difference}\n"
            f"If intended, re-baseline with --regenerate-golden in its own reviewed commit."
        )

    def reconcile(self, case_ids, *, regenerate: bool, what: str) -> None:
        """Assert the reference set and ``case_ids`` describe the same cases.

        Under ``--regenerate-golden`` the references of cases that no longer exist are deleted instead, so renaming a case does not leave the corpus permanently inconsistent and needing a hand-fix.
        """
        expected = set(case_ids)
        present = {p.name[: -len(self.suffix)] for p in self.root.glob(f"*{self.suffix}")}

        if regenerate:
            for orphan in sorted(present - expected):
                self.path(orphan).unlink()
            pytest.skip("regenerated")

        missing = sorted(expected - present)
        orphaned = sorted(present - expected)
        assert not missing, f"{what} with no reference: {missing}"
        assert not orphaned, f"references for {what} that no longer exist: {orphaned}"
