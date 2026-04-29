"""Convert BibTeX bibliography files into one yaml-per-study under
tvbo/database/studies/. Each study yaml is a small, schema-friendly record
that downstream tools (gen_abox.py, KG browser) can ingest, and that other
yaml files (models, experiments, software) can cross-reference via citekey.

Re-running is idempotent: existing study yaml files are overwritten.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

import bibtexparser
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]
DB = ROOT / "tvbo" / "database"
STUDIES = DB / "studies"
DEFAULT_BIBS = [DB / "references.bib", DB / "references_from_julia.bib"]

# bibtex entry type -> ontological/bibliographic kind label
TYPE_MAP = {
    "article": "article",
    "inproceedings": "conference_paper",
    "conference": "conference_paper",
    "incollection": "book_chapter",
    "inbook": "book_chapter",
    "book": "book",
    "phdthesis": "thesis",
    "mastersthesis": "thesis",
    "techreport": "technical_report",
    "misc": "misc",
    "unpublished": "preprint",
}


# Strip {...} wrappers and TeX escapes for yaml-friendly output.
_BRACE_RE = re.compile(r"[{}]")


def clean(value: str | None) -> str | None:
    if not value:
        return None
    v = _BRACE_RE.sub("", value).strip()
    v = re.sub(r"\s+", " ", v)
    return v or None


def split_authors(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [clean(a) or "" for a in raw.split(" and ")]


# IRI-safe sanitiser for citekeys. Strips TeX escapes and non-ASCII so the
# generated yaml filename and the resulting RDF IRI are both well-formed.
_CITEKEY_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]")


def _sanitize_citekey(raw: str) -> str:
    s = _BRACE_RE.sub("", raw)
    s = re.sub(r"\\['`^\"~=.]?\{?([A-Za-z])\}?", r"\1", s)
    s = _CITEKEY_SAFE_RE.sub("_", s)
    return s.strip("_")


# IRI-safe sanitiser for citekeys. Strips TeX escapes and non-ASCII so the
# generated yaml filename and the resulting RDF IRI are both well-formed.
_CITEKEY_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]")


def _sanitize_citekey(raw: str) -> str:
    s = _BRACE_RE.sub("", raw)
    s = re.sub(r"\\['`^\"~=.]?\{?([A-Za-z])\}?", r"\1", s)
    s = _CITEKEY_SAFE_RE.sub("_", s)
    return s.strip("_")


def entry_to_yaml(entry: dict) -> dict:
    citekey = _sanitize_citekey(entry["ID"].strip())
    btype = (entry.get("ENTRYTYPE") or "misc").lower()
    out: dict = {
        "name": citekey,
        "citekey": citekey,
        "type": TYPE_MAP.get(btype, btype),
    }
    title = clean(entry.get("title"))
    if title:
        out["title"] = title
    authors = split_authors(entry.get("author"))
    if authors:
        out["authors"] = authors
    for src, dst in (
        ("year", "year"),
        ("journal", "journal"),
        ("booktitle", "booktitle"),
        ("publisher", "publisher"),
        ("volume", "volume"),
        ("number", "number"),
        ("pages", "pages"),
        ("doi", "doi"),
        ("url", "url"),
        ("editor", "editor"),
        ("school", "school"),
        ("note", "note"),
    ):
        v = clean(entry.get(src))
        if v:
            out[dst] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bib", action="append", default=None,
                    help="BibTeX file(s); defaults to both database refs.")
    ap.add_argument("-o", "--out-dir", default=str(STUDIES))
    args = ap.parse_args()

    bibs = [pathlib.Path(p) for p in (args.bib or DEFAULT_BIBS)]
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seen: dict[str, pathlib.Path] = {}
    written = 0
    for bib in bibs:
        if not bib.exists():
            print(f"  ! missing: {bib}", file=sys.stderr)
            continue
        parser = bibtexparser.bparser.BibTexParser(common_strings=True)
        parser.ignore_nonstandard_types = False
        with bib.open() as fh:
            db = bibtexparser.load(fh, parser=parser)
        for entry in db.entries:
            data = entry_to_yaml(entry)
            citekey = data["citekey"]
            if citekey in seen:
                # First occurrence wins; record source for debugging.
                continue
            target = out_dir / f"{citekey}.yaml"
            target.write_text(
                yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
            )
            seen[citekey] = bib
            written += 1
    print(f"Wrote {written} study yaml files to {out_dir}")
    print(f"  unique citekeys: {len(seen)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
