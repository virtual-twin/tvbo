"""Shared BibTeX loader for the ontology generators.

`tvbo/database/references.bib` is the single source of truth for bibliographic metadata (also loaded at runtime by `tvbo.data.db.load_bibliography`). Both
generators consume this module so the source file, the citekey sanitiser and the field normalisation stay defined once:

- `bib_to_studies.py` emits a slim `studies/<citekey>.yaml` pointer per entry.
- `gen_abox.py` resolves the full bibliographic record by citekey when it emits
  the study individuals into the knowledge-graph A-box.

Parsing is lenient (pybtex non-strict): a repeated citekey within a file warns and the first occurrence wins, matching the historical behaviour.
"""

from __future__ import annotations

import pathlib
import re
import sys

from pybtex.database import parse_file
from pybtex.errors import set_strict_mode

# Duplicate/malformed entries warn instead of aborting the whole parse; the first occurrence of a repeated citekey wins.
set_strict_mode(False)

ROOT = pathlib.Path(__file__).resolve().parents[2]
DB = ROOT / "tvbo" / "database"
DEFAULT_BIBS = [DB / "references.bib"]

# BibTeX entry type -> ontological/bibliographic kind label. gen_abox maps these labels on to schema.org subclasses (STUDY_TYPE_MAP).
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

_BRACE_RE = re.compile(r"[{}]")
# IRI-safe citekey: strips TeX escapes and non-ASCII so the study filename and the resulting RDF IRI are both well-formed.
_CITEKEY_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]")
# Characters that can never appear in a well-formed BibTeX citekey. Their presence means the entry key is malformed (e.g. a TeX-accented key such as
# `R{"o}ssler1976`), which pybtex truncates into a spurious fragment key.
_MALFORMED_KEY_RE = re.compile(r'[{}"\\]')

# Plain-field name -> record key. Bibliographic detail lives here (in the bib), never duplicated into the slim study YAML.
_SCALAR_FIELDS = (
    "title",
    "year",
    "doi",
    "url",
    "journal",
    "booktitle",
    "publisher",
    "volume",
    "number",
    "pages",
)


def sanitize_citekey(raw: str) -> str:
    """Normalise a raw BibTeX citekey to the IRI/filename-safe form."""
    s = _BRACE_RE.sub("", raw)
    s = re.sub(r"\\['`^\"~=.]?\{?([A-Za-z])\}?", r"\1", s)
    s = _CITEKEY_SAFE_RE.sub("_", s)
    return s.strip("_")


def clean(value: str | None) -> str | None:
    """Strip `{...}` wrappers and collapse whitespace for yaml/RDF-friendly text."""
    if not value:
        return None
    v = _BRACE_RE.sub("", str(value)).strip()
    v = re.sub(r"\s+", " ", v)
    return v or None


def _format_person(person) -> str:
    """Render a pybtex `Person` as a `Last, First` display string."""
    last = " ".join(person.prelast_names + person.last_names).strip()
    first = " ".join(person.first_names + person.middle_names).strip()
    lineage = " ".join(person.lineage_names).strip()
    parts = [p for p in (last, first) if p]
    name = ", ".join(parts) if parts else str(person)
    if lineage:
        name = f"{name}, {lineage}"
    return clean(name) or name


def _entry_to_record(citekey: str, entry) -> dict:
    """Normalise one pybtex entry to a flat, yaml/RDF-friendly record."""
    fields = entry.fields
    record: dict = {
        "citekey": citekey,
        "type": TYPE_MAP.get(entry.type.lower(), entry.type.lower()),
    }
    for src in _SCALAR_FIELDS:
        v = clean(fields.get(src))
        if v is None:
            continue
        if src == "year":
            try:
                record[src] = int(v)
            except ValueError:
                record[src] = v
        else:
            record[src] = v
    authors = [_format_person(p) for p in entry.persons.get("author", [])]
    if authors:
        record["authors"] = authors
    return record


def load_bib_records(bibs: list[pathlib.Path] | None = None) -> dict[str, dict]:
    """Load the database bibliographies into `{sanitised_citekey: record}`.

    Entries are merged across files with first-occurrence-wins; each record is a flat dict carrying `citekey`, `type`, `authors` and the bibliographic
    scalars present in the source entry.
    """
    records: dict[str, dict] = {}
    for bib in bibs or DEFAULT_BIBS:
        if not bib.exists():
            continue
        data = parse_file(str(bib))
        for raw_key, entry in data.entries.items():
            if _MALFORMED_KEY_RE.search(raw_key):
                print(f"  ! skipping malformed citekey {raw_key!r} in {bib.name}", file=sys.stderr)
                continue
            citekey = sanitize_citekey(raw_key.strip())
            records.setdefault(citekey, _entry_to_record(citekey, entry))
    return records
