#!/usr/bin/env python3
"""Flag when the runtime ontology is stale relative to its sources.

`tvbo/data/ontology/tvbo.owl` is the file the platform KG actually loads (via `tvbo/ontology/owl.py`, which does NOT run a reasoner — it relies on the asserted axioms baked in by ROBOT's ELK pass). `make gen-merged` rebuilds it from the sources below and packages it here (a copy of `ontology/tvbo.owl`). This check catches the case where a source (`tvb-o-axioms.ttl` / the schema / the database / clinical) was committed after the runtime owl, so the copy in the tree describes an older world than the code beside it. It is about a checkout, not a release: `publish-pypi.yml` rebuilds the merged ontology from the released schema and hands it to the build, so a published wheel carries a current graph however far behind the committed copy has fallen. The T-box and the A-box are themselves derived and untracked, so each is represented here by the tree it is generated from — `schema/` and `tvbo/database/` — which is the thing that actually carries a commit. (The deprecated class-based `tvb-o.owl` is preserved but no longer loaded.)

This makes it answerable. It compares git COMMIT timestamps (stable across checkouts and CI, unlike filesystem mtime): if any source was committed AFTER the runtime owl was last committed, the runtime owl is stale. It also reports working-tree mtime drift as a soft hint for local, not-yet-committed edits.

    python3 scripts/ontology/check_runtime_onto_fresh.py
Exit 0 = current; non-zero = stale. Deliberately not a CI gate: passing it after every schema commit would put ROBOT and a JVM in the path of anyone touching the schema, which is the cost the release-time rebuild exists to avoid.
"""

import os
import subprocess
import sys

RUNTIME = "tvbo/data/ontology/tvbo.owl"
SOURCES = [
    "schema/",  # the LinkML T-box source; gen-owl's output is derived and untracked
    "ontology/tvb-o-axioms.ttl",  # hand-authored OWL axioms
    "ontology/tvb-o-coupling.ttl",  # coupling-evaluation scheme enrichment
    "tvbo/database/",  # the A-box source; gen-abox's output is derived and untracked
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


def _newest_mtime(path):
    """Newest mtime at or under `path`.

    A directory's own mtime only moves when an entry is added or removed, so the two tree sources would otherwise look untouched after an edit to a file inside them.
    """
    if not os.path.isdir(path):
        return os.path.getmtime(path)
    return max(
        (os.path.getmtime(os.path.join(root, f)) for root, _dirs, files in os.walk(path) for f in files),
        default=os.path.getmtime(path),
    )


def main():
    if not os.path.exists(RUNTIME):
        print(f"? {RUNTIME} not found; skipping freshness check")
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
        if _newest_mtime(src) > rt_mtime:
            newer_uncommitted.append(src)

    if newer_uncommitted:
        print("~ working-tree hint: newer than {} by mtime: {}".format(RUNTIME, ", ".join(newer_uncommitted)))

    if stale_committed:
        print(f"✗ runtime ontology is STALE — committed after {RUNTIME}:")
        for s in stale_committed:
            print(f"    {s}")
        print("  Rebuild and re-commit the runtime owl: `make gen-merged` regenerates")
        print("  ontology/tvbo.owl from the sources and packages it to")
        print("  tvbo/data/ontology/tvbo.owl (the file the runtime loads).")
        return 1

    if rt_commit is None:
        print(f"? {RUNTIME} not committed yet; mtime hints only")
        return 0
    print(f"✓ runtime ontology {RUNTIME} is current with its sources (by commit time)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
