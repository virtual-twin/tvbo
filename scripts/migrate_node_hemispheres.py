"""Derive `hemisphere` and `homologue` for every node of the curated networks from their labels, once, under review.

Prints one table per network sidecar under `tvbo/database/networks`: the label convention it detected, how many nodes fell on each side, the nodes it could not place, and the nodes whose homologue is missing. With `--write` it stores the derived values into the YAML sidecars whose every node was placed and paired and that carry no comments (a YAML dump would drop them), leaving the others untouched and listed, so a curated network never carries a half-derived side. The derivation is deliberately naive, a fixed prefix table (`L_`, `L.`, `lh.`, `Left-`, `left-`, `ctx-lh-`, `ctx_lh_`, `LH_` and their right-side twins) plus the Schaefer infix `..._LH_`, and a short midline list (brain stem) that stays side-less, because anything cleverer would be a guess made at runtime by another name; a network these rules do not cover is curated by hand.
"""

import argparse
import re
from pathlib import Path

import yaml

NETWORKS = Path(__file__).resolve().parents[1] / "tvbo" / "database" / "networks"
PREFIXES = {
    "L_": ("left", "R_"),
    "R_": ("right", "L_"),
    "L.": ("left", "R."),
    "R.": ("right", "L."),
    "lh.": ("left", "rh."),
    "rh.": ("right", "lh."),
    "Left-": ("left", "Right-"),
    "Right-": ("right", "Left-"),
    "left-": ("left", "right-"),
    "right-": ("right", "left-"),
    "ctx-lh-": ("left", "ctx-rh-"),
    "ctx-rh-": ("right", "ctx-lh-"),
    "ctx_lh_": ("left", "ctx_rh_"),
    "ctx_rh_": ("right", "ctx_lh_"),
    "LH_": ("left", "RH_"),
    "RH_": ("right", "LH_"),
}
INFIX = re.compile(r"^(\w+?_)(LH|RH)_")
MIDLINE = {"brain-stem", "brainstem", "brain_stem"}


def derive(label):
    """(hemisphere, homologue label, midline) for one label: the side and its counterpart's label, or (None, None, True) for a midline label and (None, None, False) for one no rule matches."""
    if label.lower() in MIDLINE:
        return None, None, True
    for prefix, (side, other) in PREFIXES.items():
        if label.startswith(prefix):
            return side, other + label[len(prefix) :], False
    m = INFIX.match(label)
    if m:
        tag = m.group(2)
        return ("left" if tag == "LH" else "right"), m.group(1) + ("RH" if tag == "LH" else "LH") + label[m.end(2) :], False
    return None, None, False


def process(sidecar, write):
    text = sidecar.read_text()
    doc = yaml.safe_load(text)
    nodes = doc.get("nodes") or []
    if not nodes or not any("label" in n for n in nodes):
        return None
    labels = [str(n.get("label", "")) for n in nodes]
    present = set(labels)
    derived = [derive(lab) for lab in labels]
    unplaced = [lab for lab, (s, _, mid) in zip(labels, derived, strict=True) if s is None and not mid]
    midline = [lab for lab, (_, _, mid) in zip(labels, derived, strict=True) if mid]
    unpaired = [lab for lab, (s, h, _) in zip(labels, derived, strict=True) if s is not None and h not in present]
    counts = {
        "left": sum(s == "left" for s, _, _ in derived),
        "right": sum(s == "right" for s, _, _ in derived),
        "midline": len(midline),
    }
    complete = not unplaced and not unpaired
    commented = any(line.lstrip().startswith("#") for line in text.splitlines())  # a YAML dump would drop them
    if write and complete and not commented:
        for n, (s, h, mid) in zip(nodes, derived, strict=True):
            if not mid:
                n["hemisphere"], n["homologue"] = s, h
        sidecar.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=10**9))
    return {
        "file": sidecar.name,
        "n": len(labels),
        **counts,
        "unplaced": unplaced,
        "unpaired": unpaired,
        "complete": complete,
        "commented": commented,
        "example": labels[:2],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--write",
        action="store_true",
        help="store the derived values into every fully placed and paired sidecar without comments",
    )
    a = ap.parse_args()
    rows = [r for r in (process(p, a.write) for p in sorted(NETWORKS.glob("*.yaml"))) if r]
    for r in rows:
        status = "complete" if r["complete"] else f"INCOMPLETE unplaced={len(r['unplaced'])} unpaired={len(r['unpaired'])}"
        if r["complete"] and r["commented"]:
            status += " but carries comments, left for hand curation"
        print(
            f"{r['file']}\n  n={r['n']} left={r['left']} right={r['right']} midline={r['midline']} {status} example={r['example']}"
        )
        for key in ("unplaced", "unpaired"):
            if r[key]:
                shown = r[key][:8]
                print(f"  {key}: {shown}{' ...' if len(r[key]) > 8 else ''}")
    print(
        f"\n{sum(r['complete'] for r in rows)} of {len(rows)} labelled networks fully derivable"
        + (" (written)" if a.write else " (dry run, nothing written)")
    )


if __name__ == "__main__":
    main()
