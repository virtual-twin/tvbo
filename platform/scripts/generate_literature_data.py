#!/usr/bin/env python3
"""Generate Odoo data records for tvbo.literature_reference from a BibTeX file."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from xml.sax.saxutils import escape

import bibtexparser
from bibtexparser.bparser import BibTexParser

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BIB = REPO_ROOT / "manuscript" / "references.bib"
DEFAULT_OUTPUT = REPO_ROOT / "platform" / "odoo-addons" / "tvbo" / "data" / "database_literature.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bib", type=Path, default=DEFAULT_BIB, help="Path to BibTeX file")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output XML path for tvbo.literature_reference records",
    )
    return parser.parse_args()


def load_entries(bib_path: Path) -> List[Dict[str, str]]:
    parser = BibTexParser(common_strings=True)
    with bib_path.open("r", encoding="utf-8") as handle:
        database = bibtexparser.load(handle, parser=parser)
    return database.entries


def clean_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return " ".join(str(value).strip().split())


def extract_year(entry: Dict[str, str]) -> Optional[int]:
    raw_year = entry.get("year") or entry.get("date") or ""
    match = re.search(r"(19|20)\d{2}", raw_year)
    return int(match.group(0)) if match else None


def slugify(text: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_").lower()
    return slug or "ref"


def make_xml_id(key: str, seen: set[str]) -> str:
    base = f"literature_{slugify(key)}"
    candidate = base
    counter = 2
    while candidate in seen:
        candidate = f"{base}_{counter}"
        counter += 1
    seen.add(candidate)
    return candidate


def build_records(entries: Iterable[Dict[str, str]]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    seen_ids: set[str] = set()
    for entry in entries:
        key = entry.get("ID")
        title = clean_text(entry.get("title"))
        if not key or not title:
            continue

        record = {
            "xml_id": make_xml_id(key, seen_ids),
            "key": key,
            "title": title,
            "doi": clean_text(entry.get("doi")),
            "pubmed_id": clean_text(entry.get("pmid") or entry.get("pubmed_id")),
            "year": extract_year(entry),
            "journal": clean_text(entry.get("journal") or entry.get("booktitle") or entry.get("publisher")),
            "abstract": clean_text(entry.get("abstract")),
            "authors": clean_text(entry.get("author") or entry.get("editor")),
        }
        records.append(record)
    records.sort(key=lambda r: str(r["key"]))
    return records


def write_xml(records: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n")
        handle.write("<odoo>\n  <data noupdate=\"1\">\n")
        for rec in records:
            handle.write(
                f"    <record id=\"{rec['xml_id']}\" model=\"tvbo.literature_reference\">\n"
            )
            for field_name in [
                "title",
                "key",
                "doi",
                "pubmed_id",
                "journal",
                "authors",
                "abstract",
            ]:
                value = rec.get(field_name)
                if value:
                    handle.write(
                        f"      <field name=\"{field_name}\">{escape(str(value))}</field>\n"
                    )
            if rec.get("year"):
                handle.write(f"      <field name=\"year\">{rec['year']}</field>\n")
            handle.write("    </record>\n")
        handle.write("  </data>\n</odoo>\n")


def main() -> None:
    args = parse_args()
    entries = load_entries(args.bib)
    records = build_records(entries)
    write_xml(records, args.output)
    print(f"Wrote {len(records)} literature references to {args.output}")


if __name__ == "__main__":
    main()
