#!/usr/bin/env python
"""
Auto-generate the 'Data Model' section of _toc.yml from the datamodel/ directory.

Scans datamodel/{schemas,classes,slots,enums}/ for .qmd files and rewrites
the block between the # BEGIN:datamodel-autogen … # END:datamodel-autogen
markers in _toc.yml.

Run automatically as a Quarto pre-render step (after generate_datamodel_docs.py).
"""
from pathlib import Path

DOCS_DIR = Path(__file__).parent.parent
TOC_FILE = DOCS_DIR / "_toc.yml"
BEGIN_MARKER = "# BEGIN:datamodel-autogen"
END_MARKER   = "# END:datamodel-autogen"

# Indentation constants matching the rest of _toc.yml
L1 = "              "   # 14 sp  — direct contents of the sidebar
L2 = "                  "   # 18 sp  — inside a section
L3 = "                      "   # 22 sp  — inside a nested section


import re

# Camel-case tokens whose canonical product casing is acronym-style. The
# splitter below first segments ``BidsEntities`` → ``["Bids","Entities"]``;
# this map then upgrades the leading token to ``BIDS``.
TOKEN_OVERRIDES: dict[str, str] = {
    "Bids":     "BIDS",
    "Tvb":      "TVB",
    "Tvbo":     "TVB-O",
    "Lems":     "LEMS",
    "Owl":      "OWL",
    "Api":      "API",
    "Cli":      "CLI",
    "Nifti":    "NIfTI",
    "Hdf5":     "HDF5",
    "Json":     "JSON",
    "Yaml":     "YAML",
    "Xml":      "XML",
    "Psd":      "PSD",
    "Fc":       "FC",
    "Bold":     "BOLD",
    "Eeg":      "EEG",
    "Meg":      "MEG",
    "Mri":      "MRI",
    "Dwi":      "DWI",
    "Pet":      "PET",
}

# Schema-file stem overrides (these are flat names, not camel-case).
SCHEMA_LABELS: dict[str, str] = {
    "common":         "Common",
    "software":       "Software",
    "types":          "Types",
    "SANDS":          "SANDS",
    "tvb-datamodel":  "TVB Datamodel",
    "tvb_dbs":        "TVB Databases",
    "tvbo_study":     "TVB-O Study",
    "tvbo_units":     "TVB-O Units",
}


def qmd_title(path: Path) -> str:
    """Return a human-readable title from a .qmd filename stem.

    For schema files we look up an explicit label first. For class/slot/enum
    files we split camel-case on the lower→upper boundary, then upgrade any
    acronym-style tokens (``Bids`` → ``BIDS`` …).
    """
    stem = path.stem
    if path.parent.name == "schemas":
        return SCHEMA_LABELS.get(stem, stem)
    tokens = re.split(r"(?<=[a-z])(?=[A-Z])", stem)
    tokens = [TOKEN_OVERRIDES.get(t, t) for t in tokens]
    return " ".join(tokens)


def build_section(pages: list[Path], subdir: str, section_title: str) -> list[str]:
    lines: list[str] = []
    lines.append(f"{L2}- section: \"{section_title}\"")
    lines.append(f"{L2}  contents:")
    for p in sorted(pages, key=lambda x: x.stem.lower()):
        title = qmd_title(p)
        href  = f"datamodel/{subdir}/{p.name}"
        lines.append(f"{L3}- text: \"{title}\"")
        lines.append(f"{L3}  href: {href}")
    return lines


def generate_block() -> str:
    # Discover all subsection directories with .qmd files
    subdirs = [
        ("schemas", "Schemas"),
        ("classes", "Classes"),
        ("slots",   "Slots"),
        ("enums",   "Enumerations"),
    ]

    lines: list[str] = []
    lines.append(f"{BEGIN_MARKER}")
    lines.append(f"{L1}- section: \"Data Model\"")
    lines.append(f"{L1}  href: datamodel/index.qmd")
    lines.append(f"{L1}  contents:")

    for subdir, title in subdirs:
        d = DOCS_DIR / "datamodel" / subdir
        if not d.exists():
            continue
        pages = sorted(d.glob("*.qmd"), key=lambda p: p.stem.lower())
        if pages:
            lines += build_section(pages, subdir, title)

    lines.append(f"{END_MARKER}")
    return "\n".join(lines)


def update_toc() -> None:
    text = TOC_FILE.read_text()

    begin_idx = text.find(BEGIN_MARKER)
    end_idx   = text.find(END_MARKER)

    if begin_idx == -1 or end_idx == -1:
        print(f"✗ Markers not found in {TOC_FILE}. Add {BEGIN_MARKER!r} / {END_MARKER!r}.")
        raise SystemExit(1)

    new_block = generate_block()
    new_text  = text[:begin_idx] + new_block + text[end_idx + len(END_MARKER):]

    if new_text == text:
        print("Data Model TOC section unchanged — skipping write.")
        return

    TOC_FILE.write_text(new_text)
    print(f"✓ Updated Data Model section in {TOC_FILE.name}")


if __name__ == "__main__":
    update_toc()
