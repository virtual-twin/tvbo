"""The canonical replication-pairs contract.

A replication study states its findings as *pairs*: a number the paper published beside the number this study reproduced. Every study emits them under one analysis name, in one schema, so a consumer joins on numbers and never parses prose or per-study naming.

A study builds a list of row mappings and hands them to :func:`pairs_payload`, whose return value is the analysis container payload. :func:`conforms` reports what a written container is missing, so a migration can be measured rather than assumed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

ANALYSIS_NAME = "replication_pairs"
"""The analysis name every study declares; the container is ``ana-replicationpairs_result.h5``."""

KIND_VOCABULARY = ("measured", "closed_form", "configured", "degenerate")
"""What the reproduced side is: a pipeline output, a closed-form evaluation, an input we
configured rather than derived, or a pair whose published value cannot carry a deviation."""

PROVENANCE_VOCABULARY = ("printed", "axis_read", "rederived", "bound", "released_array", "not_in_paper")
"""Where the published side came from, which bounds how far a deviation may be read."""

REQUIRED_FIELDS = ("quantity", "published", "reproduced", "kind", "published_provenance", "join_sound")
"""Fields a row must carry. A consumer may rely on these being present in every study."""

OPTIONAL_FIELDS = ("units", "published_source", "reproduced_from")
"""Fields a row should carry where the study can state them; absent ones serialize empty."""

_DERIVED_FIELDS = ("deviation", "abs_deviation")


def _rel_deviation(published: float, reproduced: float) -> float:
    """Reproduced minus published, relative to the published magnitude.

    Undefined where the paper published zero, since no relative scale exists there.
    """
    return float("nan") if published == 0 else (reproduced - published) / abs(published)


def pairs_payload(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The container payload for one study's replication pairs.

    Emits one parallel array per field, ordered as ``rows`` is, with ``deviation`` and ``abs_deviation`` computed here so no study can compute them differently. Raises ``ValueError`` on a missing required field or a term outside its vocabulary.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("a replication-pairs analysis states at least one pair")
    for i, r in enumerate(rows):
        missing = [f for f in REQUIRED_FIELDS if f not in r]
        if missing:
            raise ValueError(f"row {i} ({r.get('quantity', '?')}) lacks {missing}")
        if r["kind"] not in KIND_VOCABULARY:
            raise ValueError(f"row {i}: kind {r['kind']!r} outside {KIND_VOCABULARY}")
        if r["published_provenance"] not in PROVENANCE_VOCABULARY:
            raise ValueError(f"row {i}: published_provenance {r['published_provenance']!r} outside {PROVENANCE_VOCABULARY}")

    pub = np.array([float(r["published"]) for r in rows])
    rep = np.array([float(r["reproduced"]) for r in rows])
    dev = np.array([_rel_deviation(p, q) for p, q in zip(pub, rep, strict=True)])
    payload: dict[str, Any] = {
        "quantity": np.array([str(r["quantity"]) for r in rows]),
        "published": pub,
        "reproduced": rep,
        "deviation": dev,
        "abs_deviation": np.abs(dev),
        "kind": np.array([str(r["kind"]) for r in rows]),
        "published_provenance": np.array([str(r["published_provenance"]) for r in rows]),
        "join_sound": np.array([bool(r["join_sound"]) for r in rows], dtype=bool),
    }
    for field in OPTIONAL_FIELDS:
        payload[field] = np.array([str(r.get(field, "")) for r in rows])
    return payload


def conforms(container: Mapping[str, Any] | Any) -> list[str]:
    """What ``container`` lacks to satisfy the contract, empty where it conforms.

    Accepts anything key-addressable, so an open ``h5py.File`` may be passed directly.
    Fields are looked up both bare and under the ``observation__`` prefix the writer adds.
    """
    keys = set(container.keys())
    want = (*REQUIRED_FIELDS, *_DERIVED_FIELDS, *OPTIONAL_FIELDS)
    return [f for f in want if f not in keys and f"observation__{f}" not in keys]
