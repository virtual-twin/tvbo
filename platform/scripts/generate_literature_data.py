#!/usr/bin/env python3
"""Generate Odoo data records for tvbo.literature_reference from a BibTeX file.

Only imports references that have a DOI or PubMed ID. If a reference has a DOI but
no PMID, attempts to retrieve the PMID from NCBI. Deduplicates based on DOI/PMID.

Requires PUBMED_API_KEY environment variable for NCBI API access.
"""
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from xml.sax.saxutils import escape

import bibtexparser
from bibtexparser.bparser import BibTexParser

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BIB = REPO_ROOT / "manuscript" / "references.bib"
DEFAULT_OUTPUT = REPO_ROOT / "platform" / "odoo-addons" / "tvbo" / "data" / "database_literature.xml"

# NCBI E-utilities base URL
NCBI_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_API_KEY = os.environ.get("PUBMED_API_KEY", "")

# Rate limiting: NCBI allows 10 requests/second with API key, 3/second without
REQUEST_DELAY = 0.1 if PUBMED_API_KEY else 0.34


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bib", type=Path, default=DEFAULT_BIB, help="Path to BibTeX file")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output XML path for tvbo.literature_reference records",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Skip fetching missing PMIDs from NCBI (faster, but fewer PMIDs)",
    )
    return parser.parse_args()


def load_entries(bib_path: Path) -> List[Dict[str, str]]:
    parser = BibTexParser(common_strings=True)
    with bib_path.open("r", encoding="utf-8") as handle:
        database = bibtexparser.load(handle, parser=parser)
    return database.entries


def strip_invalid_xml_chars(text: str) -> str:
    """Remove characters that are invalid in XML 1.0.

    Valid XML 1.0 chars: #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]
    """
    # Pattern matches invalid XML 1.0 characters
    invalid_xml_pattern = re.compile(
        r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]'
    )
    return invalid_xml_pattern.sub('', text)


def clean_text(value: Optional[str]) -> str:
    if not value:
        return ""
    # First strip invalid XML characters, then normalize whitespace
    cleaned = strip_invalid_xml_chars(str(value))
    return " ".join(cleaned.strip().split())


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


def normalize_doi(doi: str) -> str:
    """Normalize DOI to standard format (lowercase, no URL prefix)."""
    if not doi:
        return ""
    # Remove common URL prefixes
    doi = re.sub(r'^https?://(dx\.)?doi\.org/', '', doi.strip())
    # Remove 'doi:' prefix
    doi = re.sub(r'^doi:\s*', '', doi, flags=re.IGNORECASE)
    return doi.strip().lower()


def normalize_pmid(pmid: str) -> str:
    """Extract numeric PMID from various formats."""
    if not pmid:
        return ""
    # Extract just the numeric part
    match = re.search(r'\d+', str(pmid))
    return match.group(0) if match else ""


def fetch_pmid_from_doi(doi: str) -> Optional[str]:
    """Fetch PubMed ID from DOI using NCBI E-utilities."""
    if not REQUESTS_AVAILABLE or not doi:
        return None

    params = {
        "db": "pubmed",
        "term": f"{doi}[doi]",
        "retmode": "json",
        "retmax": 1,
    }
    if PUBMED_API_KEY:
        params["api_key"] = PUBMED_API_KEY

    try:
        time.sleep(REQUEST_DELAY)  # Rate limiting
        response = requests.get(NCBI_ESEARCH_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        id_list = data.get("esearchresult", {}).get("idlist", [])
        if id_list:
            return id_list[0]
    except Exception as e:
        print(f"  Warning: Failed to fetch PMID for DOI {doi}: {e}")

    return None


def build_records(entries: Iterable[Dict[str, str]], fetch_missing_pmids: bool = True) -> List[Dict[str, object]]:
    """Build records from BibTeX entries.

    Only includes entries that have a DOI or PMID.
    Deduplicates based on normalized DOI and PMID.
    Optionally fetches missing PMIDs from NCBI.
    """
    records: List[Dict[str, object]] = []
    seen_xml_ids: set[str] = set()
    seen_dois: set[str] = set()
    seen_pmids: set[str] = set()

    skipped_no_id = 0
    skipped_duplicate = 0
    pmids_fetched = 0

    entries_list = list(entries)
    total = len(entries_list)

    for i, entry in enumerate(entries_list):
        key = entry.get("ID")
        title = clean_text(entry.get("title"))
        if not key or not title:
            continue

        # Get and normalize DOI and PMID
        doi = normalize_doi(entry.get("doi", ""))
        pmid = normalize_pmid(entry.get("pmid") or entry.get("pubmed_id") or "")

        # Skip entries without DOI or PMID
        if not doi and not pmid:
            skipped_no_id += 1
            continue

        # Try to fetch PMID from DOI if missing
        if doi and not pmid and fetch_missing_pmids:
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  Processing {i + 1}/{total}...")
            fetched_pmid = fetch_pmid_from_doi(doi)
            if fetched_pmid:
                pmid = fetched_pmid
                pmids_fetched += 1

        # Check for duplicates
        is_duplicate = False
        if doi and doi in seen_dois:
            is_duplicate = True
        if pmid and pmid in seen_pmids:
            is_duplicate = True

        if is_duplicate:
            skipped_duplicate += 1
            continue

        # Mark as seen
        if doi:
            seen_dois.add(doi)
        if pmid:
            seen_pmids.add(pmid)

        record = {
            "xml_id": make_xml_id(key, seen_xml_ids),
            "key": key,
            "title": title,
            "doi": doi,
            "pubmed_id": pmid,
            "year": extract_year(entry),
            "journal": clean_text(entry.get("journal") or entry.get("booktitle") or entry.get("publisher")),
            "abstract": clean_text(entry.get("abstract")),
            "authors": clean_text(entry.get("author") or entry.get("editor")),
        }
        records.append(record)

    records.sort(key=lambda r: str(r["key"]))

    print(f"  Skipped {skipped_no_id} entries without DOI or PMID")
    print(f"  Skipped {skipped_duplicate} duplicate entries")
    if fetch_missing_pmids:
        print(f"  Fetched {pmids_fetched} PMIDs from DOIs")

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

    print(f"Loading BibTeX from {args.bib}...")
    entries = load_entries(args.bib)
    print(f"  Found {len(entries)} total entries")

    if not args.no_fetch and not REQUESTS_AVAILABLE:
        print("  Warning: requests library not available, skipping PMID fetch")
    if not args.no_fetch and REQUESTS_AVAILABLE and not PUBMED_API_KEY:
        print("  Warning: PUBMED_API_KEY not set, using slower rate limit")

    print("Building records (filtering for DOI/PMID, deduplicating)...")
    fetch_pmids = not args.no_fetch and REQUESTS_AVAILABLE
    records = build_records(entries, fetch_missing_pmids=fetch_pmids)

    print(f"Writing {len(records)} literature references to {args.output}...")
    write_xml(records, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
