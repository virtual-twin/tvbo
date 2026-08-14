#!/usr/bin/env python3
"""Flag when the runtime ontology is stale relative to its sources.

`tvbo/data/ontology/tvbo.owl` is the file the platform KG actually loads
(via `tvbo/ontology/owl.py`, which does NOT run a reasoner — it relies on the
asserted axioms baked in by ROBOT's ELK pass). `make gen-merged` rebuilds it
from the sources below and packages it here (a copy of `ontology/tvbo.owl`).
This check catches the case where a source (`tvb-o-axioms.ttl` /
`tvb-o-struct.owl` / the A-box / clinical) was committed after the runtime owl,
so the packaged copy silently lags the sources and the deployed KG is stale.
(The deprecated class-based `tvb-o.owl` is preserved but no longer loaded.)

This makes it answerable. It compares git COMMIT timestamps (stable across
checkouts and CI, unlike filesystem mtime): if any source was committed AFTER the
runtime owl was last committed, the runtime owl is stale. It also reports
working-tree mtime drift as a soft hint for local, not-yet-committed edits.

    python3 scripts/ontology/check_runtime_onto_fresh.py
Exit 0 = current; non-zero = stale (suitable as a CI gate alongside the existing
"committed artifact must match regenerated output" checks).
"""

import os
import subprocess
import sys

RUNTIME = "tvbo/data/ontology/tvbo.owl"
SOURCES = [
    "ontology/tvb-o-struct.owl",  # gen-owl  (LinkML T-box)
    "ontology/tvb-o-axioms.ttl",  # hand-authored OWL axioms
    "ontology/tvb-o-coupling.ttl",  # coupling-evaluation scheme enrichment
    "ontology/tvb-o-data.ttl",  # gen-abox (A-box)
    "ontology/tvb-o-clinical.ttl",  # clinical addon
    "ontology/tvb-o-clinical-nmm.ttl",
    "ontology/clinical-postmerge.ru",  # SPARQL updates applied on merge
    "ontology/fix-punning.ru",
]


def _git_commit_time(path):
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "--", path], capture_output=True, text=True, check=True
        ).stdout.strip()
        return int(out) if out else None
    except Exception:  # noqa: BLE001
        return None


def main():
    if not os.path.exists(RUNTIME):
        print("? %s not found; skipping freshness check" % RUNTIME)
        return 0

    rt_commit = _git_commit_time(RUNTIME)
    stale_committed, newer_uncommitted = [], []
    rt_mtime = os.path.getmtime(RUNTIME)

    for src in SOURCES:
        if not os.path.exists(src):
            continue
        sc = _git_commit_time(src)
        if rt_commit is not None and sc is not None and sc > rt_commit:
            stale_committed.append(src)
        if os.path.getmtime(src) > rt_mtime:
            newer_uncommitted.append(src)

    if newer_uncommitted:
        print("~ working-tree hint: newer than %s by mtime: %s" % (RUNTIME, ", ".join(newer_uncommitted)))

    if stale_committed:
        print("✗ runtime ontology is STALE — committed after %s:" % RUNTIME)
        for s in stale_committed:
            print("    %s" % s)
        print("  Rebuild and re-commit the runtime owl: `make gen-merged` regenerates")
        print("  ontology/tvbo.owl from the sources and packages it to")
        print("  tvbo/data/ontology/tvbo.owl (the file the runtime loads).")
        return 1

    if rt_commit is None:
        print("? %s not committed yet; mtime hints only" % RUNTIME)
        return 0
    print("✓ runtime ontology %s is current with its sources (by commit time)" % RUNTIME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
