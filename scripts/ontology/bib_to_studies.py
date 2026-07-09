"""Emit one slim yaml-per-study under tvbo/database/studies/ from the BibTeX
bibliographies.

A study yaml is a pointer, not a bibliographic record: it carries the citekey
(the join key into references.bib, from which gen_abox.py resolves the full
bibliographic detail) and, when available, the doi (the pointer to the source on
the web). Everything else -- title, authors, journal, volume, pages, ... -- lives
in references.bib and is resolved by citekey, so it is never duplicated here.

Re-running is idempotent: existing study yaml files are overwritten.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

from _bib import DEFAULT_BIBS, load_bib_records

ROOT = pathlib.Path(__file__).resolve().parents[2]
STUDIES = ROOT / "tvbo" / "database" / "studies"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bib", action="append", default=None,
                    help="BibTeX file(s); defaults to both database refs.")
    ap.add_argument("-o", "--out-dir", default=str(STUDIES))
    args = ap.parse_args()

    bibs = [pathlib.Path(p) for p in args.bib] if args.bib else DEFAULT_BIBS
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for bib in bibs:
        if not bib.exists():
            print(f"  ! missing: {bib}", file=sys.stderr)

    records = load_bib_records(bibs)
    written = 0
    for citekey, record in records.items():
        study = {"citekey": citekey}
        if record.get("doi"):
            study["doi"] = record["doi"]
        target = out_dir / f"{citekey}.yaml"
        target.write_text(yaml.safe_dump(study, sort_keys=False, allow_unicode=True))
        written += 1

    print(f"Wrote {written} study yaml files to {out_dir}")
    print(f"  unique citekeys: {len(records)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
