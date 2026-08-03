"""Shared machinery for TVBO's golden corpora.

A golden corpus freezes an artifact TVBO promises to produce — generated source, simulation
output — and fails when it changes. Every corpus needs the same five behaviours, and this
module owns them once so a new corpus is a storage format plus a comparison, not another
copy of the harness:

* resolving a case to its reference file,
* re-baselining under ``--regenerate-golden``, including pruning references whose case no
  longer exists,
* refusing to pass when a case has no reference at all,
* reporting a mismatch as a diagnostic rather than an opaque assertion,
* checking that the case set and the reference set describe the same things.

Regeneration deliberately cannot produce a green run: every regenerated case is skipped
rather than passed, and :func:`pytest_sessionfinish` in ``conftest`` fails the session. A
re-baseline is a change to what TVBO promises, so it must be reviewed and committed on its
own — never mistaken for a suite that passed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest


class GoldenCorpus:
    """One directory of reference files, keyed by case id.

    Args:
        root: Directory holding the reference files. Created on regeneration.
        suffix: Reference file extension, including the leading dot.
        write: ``(path, produced) -> None``; persists a freshly produced artifact.
        read: ``(path) -> expected``; loads a reference for comparison.
        compare: ``(produced, expected) -> str | None``; returns ``None`` when they
            match, otherwise a human-readable account of the difference.
    """

    def __init__(
        self,
        root: Path,
        suffix: str,
        *,
        write: Callable[[Path, object], None],
        read: Callable[[Path], object],
        compare: Callable[[object, object], str | None],
    ):
        self.root = root
        self.suffix = suffix
        self._write = write
        self._read = read
        self._compare = compare

    def path(self, case_id: str) -> Path:
        return self.root / f"{case_id}{self.suffix}"

    def check(self, case_id: str, produced, *, regenerate: bool, what: str) -> None:
        """Assert ``produced`` matches the reference for ``case_id``.

        Skips (never passes) when regenerating. Fails with a pointed message when the
        reference is absent, so a newly added case cannot slip in unreviewed.
        """
        reference = self.path(case_id)

        if regenerate:
            self.root.mkdir(parents=True, exist_ok=True)
            self._write(reference, produced)
            pytest.skip("regenerated")

        if not reference.is_file():
            pytest.fail(
                f"No reference for {case_id}. A new {what} needs its output reviewed and "
                f"committed: run --regenerate-golden and inspect what it produced."
            )

        difference = self._compare(produced, self._read(reference))
        if difference is None:
            return
        pytest.fail(
            f"{case_id}: {what} changed.\n{difference}\n"
            f"If intended, re-baseline with --regenerate-golden in its own reviewed commit."
        )

    def reconcile(self, case_ids, *, regenerate: bool, what: str) -> None:
        """Assert the reference set and ``case_ids`` describe the same cases.

        Under ``--regenerate-golden`` the references of cases that no longer exist are
        deleted instead, so renaming a case does not leave the corpus permanently
        inconsistent and needing a hand-fix.
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
