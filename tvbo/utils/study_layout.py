"""Read the study-layout record and answer questions about it.

``schema/study_layout.yaml`` is the single ground truth for where a study dataset keeps its specification, its inputs, its results, its figures and its provenance. It is a LinkML instance of :class:`StudyLayout`, authored beside the schema that types it, and materialized into the package tree at ``tvbo/rules/study_layout.yaml`` by ``hatch_build.py`` so a wheel-installed tvbo resolves it too.

Everything that needs to know the layout comes through this module: the scaffolder that creates a study, the resolvers that turn a role into a path, the writers that generate ``.gitignore`` and ``.bidsignore``, the validator that checks a study against the record, and the documentation that renders the tree. Nothing restates the layout, so it cannot drift.

A resolver asks for a *role*, never for a literal path::

    from tvbo.utils.study_layout import study_path
    out = study_path("results", root=study_root)   # <root>/derivatives/tvbo

Moving or renaming a directory is then a one-line edit to the record.

One record covers every kind of study: an entry carrying ``in_templates`` belongs only to the named ``tvbo study init --template`` variants, so ``templates=("replication",)`` selects the replication layout out of the same tree the general one comes from.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from tvbo.datamodel.schema import StudyDirectory, StudyFile, StudyLayout
from tvbo.utils import yaml_loader

RECORD_PATH = Path(__file__).resolve().parent.parent / "rules" / "study_layout.yaml"
"""The materialized record. Authored at ``schema/study_layout.yaml``; see the module docstring."""

SOURCE_NAME = "schema/study_layout.yaml"
"""Where a person edits the layout, named in every file this module generates."""

_TRACK_ALL, _TRACK_DECLARED, _TRACK_NONE = "all", "declared_files", "none"
_BIDS_UNSANCTIONED, _BIDS_PROPOSED, _BIDS_NESTED, _BIDS_HIDDEN = (
    "unsanctioned",
    "proposed",
    "nested_dataset",
    "hidden",
)
_NEEDS_BIDSIGNORE = (_BIDS_UNSANCTIONED, _BIDS_PROPOSED)


@lru_cache(maxsize=1)
def load_layout(path: Path | str | None = None) -> StudyLayout:
    """Return the layout record, parsed and cached."""
    return yaml_loader.load(str(path or RECORD_PATH), StudyLayout)


ANY_TEMPLATE = None
"""Pass as ``templates`` to select every entry the record declares, whatever template names it.

Role lookup uses this. ``in_templates`` says whether ``tvbo study init`` CREATES an entry, not where that entry lives, so a replication study's report must be able to ask the record for ``analysis`` or ``original_study`` without also telling it which template built the study. The tree and ignore-file renderers still pass an explicit tuple, because there what a study HAS is exactly the question.
"""


def _selected(entry: StudyDirectory | StudyFile, templates: tuple[str, ...] | None) -> bool:
    """True when ``entry`` belongs to a study built with ``templates``.

    An entry with no ``in_templates`` is part of every study. One that names templates is included only when a requested template is among them, or when ``templates`` is :data:`ANY_TEMPLATE`.
    """
    if templates is ANY_TEMPLATE:
        return True
    named = [str(t) for t in entry.in_templates or []]
    return not named or any(t in named for t in templates)


def walk(layout: StudyLayout | None = None, templates: tuple[str, ...] | None = ()) -> list[tuple[str, StudyDirectory]]:
    """Every directory as ``(relative_posix_path, directory)``, outermost first."""
    layout = layout or load_layout()
    found: list[tuple[str, StudyDirectory]] = []

    def descend(parent: str, dirs) -> None:
        for d in dirs or []:
            if not _selected(d, templates):
                continue
            rel = f"{parent}/{d.name}" if parent else str(d.name)
            found.append((rel, d))
            descend(rel, d.subdirs)

    descend("", layout.subdirs)
    return found


def iter_files(layout: StudyLayout | None = None, templates: tuple[str, ...] | None = ()) -> list[tuple[str, StudyFile]]:
    """Every file the layout accounts for as ``(relative_posix_path, file)``."""
    layout = layout or load_layout()
    found: list[tuple[str, StudyFile]] = [(str(f.name), f) for f in layout.files or [] if _selected(f, templates)]
    for rel, d in walk(layout, templates):
        found.extend((f"{rel}/{f.name}", f) for f in d.files or [] if _selected(f, templates))
    return found


def _one(matches: list[str], role: str, known: list[str], kind: str) -> str:
    """The single match for ``role``, or a message naming what was actually available."""
    if not matches:
        raise KeyError(f"No {kind} has role {role!r} in the study layout. Known roles: {', '.join(sorted(set(known)))}")
    if len(matches) > 1:
        raise KeyError(f"Role {role!r} is ambiguous in the study layout: {', '.join(matches)}")
    return matches[0]


def relpath(role: str, layout: StudyLayout | None = None) -> str:
    """Relative path of the directory carrying ``role``, e.g. ``results`` to ``derivatives/tvbo``."""
    entries = walk(layout, ANY_TEMPLATE)
    return _one(
        [rel for rel, d in entries if str(d.role) == role],
        role,
        [str(d.role) for _, d in entries if d.role],
        "directory",
    )


def study_path(role: str, root: Path | str | None = None, layout: StudyLayout | None = None) -> Path:
    """Path of the directory carrying ``role``, absolute under ``root`` when given."""
    rel = Path(relpath(role, layout))
    return Path(root) / rel if root is not None else rel


def file_relpath(
    role: str,
    study: str | None = None,
    layout: StudyLayout | None = None,
    parent: str = "",
) -> str:
    """Relative path of the file carrying ``role`` in ``parent``, ``{study}`` interpolated.

    A file role is unique per directory rather than per layout: a README belongs to the dataset root and another to ``sourcedata/``, so the directory is part of the question. ``parent`` defaults to the dataset root.
    """
    entries = [(rel, f) for rel, f in iter_files(layout, ANY_TEMPLATE) if rel.rpartition("/")[0] == parent]
    rel = _one(
        [rel for rel, f in entries if str(f.role) == role],
        role,
        [str(f.role) for _, f in entries if f.role],
        f"file in {parent or 'the dataset root'}",
    )
    return interpolate(rel, study)


def interpolate(name: str, study: str | None) -> str:
    """Substitute the study's own name into a layout entry that is named after it."""
    if "{study}" not in name:
        return name
    if study is None:
        raise ValueError(f"Layout entry {name!r} is named after the study and needs its name to resolve.")
    return name.replace("{study}", study)


def _ancestor_untracked(rel: str, entries: list[tuple[str, StudyDirectory]]) -> bool:
    """True when a strict parent of ``rel`` is already ignored, which decides for ``rel``."""
    return any(rel.startswith(f"{other}/") and str(d.tracked) != _TRACK_ALL for other, d in entries)


def is_tracked(rel: str, layout: StudyLayout | None = None, templates: tuple[str, ...] = ()) -> bool:
    """True when version control keeps this directory's contents.

    An ignored ancestor decides for its descendants: ``.tvbo/kits`` carries the default ``tracked: all`` and is still untracked, because ``.tvbo`` is. Callers that create placeholder files ask this rather than reading ``tracked`` directly.
    """
    entries = walk(layout, templates)
    own = next((d for other, d in entries if other == rel), None)
    if own is None:
        raise KeyError(f"{rel!r} is not a directory in the study layout.")
    return str(own.tracked) == _TRACK_ALL and not _ancestor_untracked(rel, entries)


def gitignore_lines(
    layout: StudyLayout | None = None,
    templates: tuple[str, ...] = (),
    study: str | None = None,
) -> list[str]:
    """The ``.gitignore`` body, derived from each entry's ``tracked`` field.

    A directory under an already-ignored parent is skipped, so each rule is stated once. A ``declared_files`` directory ignores its *contents* and re-includes each declared file, because a negation cannot re-include from an excluded directory.
    """
    layout = layout or load_layout()
    entries = walk(layout, templates)
    lines = [f"# Generated by `tvbo study init` from {SOURCE_NAME}. Edit the record, not this file."]
    files = iter_files(layout, templates)
    for rel, d in entries:
        tracked = str(d.tracked)
        if tracked == _TRACK_ALL or _ancestor_untracked(rel, entries):
            continue
        if tracked == _TRACK_NONE:
            lines.append(f"{rel}/")
        elif tracked == _TRACK_DECLARED:
            lines.extend(_declared_files_rules(rel, files, study))
    lines.extend(interpolate(rel, study) for rel, f in iter_files(layout, templates) if str(f.tracked) == _TRACK_NONE)
    return lines


def _declared_files_rules(rel: str, files: list[tuple[str, StudyFile]], study: str | None = None) -> list[str]:
    """Ignore everything under ``rel`` but the declared files, re-opening each directory on the way.

    Git cannot re-include a path out of an excluded *directory*, so every level between ``rel`` and a declared file has to be excluded by content (``dir/*``) and then re-included itself.
    """
    rules = [f"{rel}/"] if not any(r.startswith(f"{rel}/") for r, _ in files) else [f"{rel}/*"]
    for path, f in files:
        if not path.startswith(f"{rel}/") or str(f.tracked) == _TRACK_NONE:
            continue
        parts = interpolate(path, study).split("/")
        for depth in range(len(rel.split("/")) + 1, len(parts)):
            branch = "/".join(parts[:depth])
            rules.extend(rule for rule in (f"!{branch}/", f"{branch}/*") if rule not in rules)
        rules.append(f"!{'/'.join(parts)}")
    return rules


def bidsignore_lines(
    layout: StudyLayout | None = None,
    templates: tuple[str, ...] = (),
    study: str | None = None,
) -> list[str]:
    """The ``.bidsignore`` body, derived from each entry's standing with BIDS.

    Only what a released validator does not know is listed: an entry outside the BIDS vocabulary, and one a BEP proposes but has not landed. A hidden directory is skipped by convention and a nested dataset is validated as its own type, so neither needs a line.
    Each entry's ``expires_with`` says what would retire it, which is what keeps the surface from becoming permanent by default.
    """
    layout = layout or load_layout()
    lines = [f"# Generated by `tvbo study init` from {SOURCE_NAME}. Edit the record, not this file."]
    for rel, d in walk(layout, templates):
        if str(d.bids) in _NEEDS_BIDSIGNORE:
            lines.append(f"{rel}/")
    lines.extend(f"/{interpolate(rel, study)}" for rel, f in iter_files(layout, templates) if str(f.bids) in _NEEDS_BIDSIGNORE)
    return lines


def tree(
    layout: StudyLayout | None = None,
    study: str = "<Study>",
    templates: tuple[str, ...] = (),
) -> str:
    """The layout as an indented tree, for documentation to render rather than restate."""
    layout = layout or load_layout()
    out = [f"{study}/"]
    out.extend(f"  {interpolate(str(f.name), study)}" for rel, f in iter_files(layout, templates) if "/" not in rel)
    for rel, d in walk(layout, templates):
        depth = rel.count("/") + 1
        out.append(f"{'  ' * depth}{d.name}/")
        out.extend(f"{'  ' * (depth + 1)}{f.name}" for f in d.files or [] if _selected(f, templates))
    return "\n".join(out)


def markers(name: str) -> tuple[str, str]:
    """The comment pair delimiting the generated region named ``name``."""
    return (
        f"<!-- BEGIN {name} (generated by `tvbo study layout --sync`; do not edit) -->",
        f"<!-- END {name} -->",
    )


LAYOUT_BEGIN, LAYOUT_END = markers("STUDY LAYOUT")


def layout_block(
    layout: StudyLayout | None = None,
    study: str = "<Study>",
    templates: tuple[str, ...] = (),
) -> str:
    """The layout as a marker-delimited markdown block, for a document to carry rather than retype.

    Every entry's own ``description`` travels with it, so the tree and its explanation come from the one record and a document showing the layout cannot fall behind it.
    """
    layout = layout or load_layout()
    width = max((len(line) for line in tree(layout, study, templates).splitlines()), default=0)
    lines = ["```"]
    for line in tree(layout, study, templates).splitlines():
        note = _note_for(line, layout, study, templates)
        lines.append(f"{line.ljust(width + 2)}{note}".rstrip() if note else line)
    lines.append("```")
    return _wrap("STUDY LAYOUT", "\n".join(lines))


def _wrap(name: str, body: str) -> str:
    """``body`` between the markers for the generated region ``name``."""
    begin, end = markers(name)
    return "\n".join([begin, "", body, "", end])


def _sentence(entry: Any) -> str:
    """The first sentence of an entry's own ``description``, whole rather than clipped."""
    return " ".join(str(getattr(entry, "description", None) or "").split()).split(". ")[0].rstrip(".")


def exceptions_block(
    layout: StudyLayout | None = None,
    study: str = "<Study>",
    templates: tuple[str, ...] = (),
) -> str:
    """Every entry BIDS does not yet know, and the proposal each waits on.

    Exactly the entries :func:`bidsignore_lines` exempts, so this table explains that file rather than paraphrasing it, and an entry a BEP sanctions leaves both by one change of status. A nested derivative dataset and a dot-prefixed build root are absent because neither is an exception: BIDS sanctions the first and skips the second by convention.
    """
    layout = layout or load_layout()
    entries = [(interpolate(rel, study), e) for rel, e in iter_files(layout, templates)]
    entries += [(f"{interpolate(rel, study)}/", e) for rel, e in walk(layout, templates)]
    rows = ["| path | status | waits on |", "|------|--------|----------|"]
    for path, entry in entries:
        status = str(getattr(entry, "bids", None) or "")
        if status not in _NEEDS_BIDSIGNORE:
            continue
        waits = " ".join(str(getattr(entry, "expires_with", None) or "").split()) or "-"
        rows.append(f"| `{path}` | {status} | {waits} |")
    return _wrap("BIDS EXCEPTIONS", "\n".join(rows))


def ignore_files_block(
    layout: StudyLayout | None = None,
    study: str = "<Study>",
    templates: tuple[str, ...] = (),
) -> str:
    """Both generated ignore files, as a study gets them.

    Rendered by the same functions that write them, so a page cannot show a gate looser than the one a scaffolded study enforces.
    """
    layout = layout or load_layout()
    lines = ["`.gitignore`", "", "```gitignore"]
    lines.extend(gitignore_lines(layout, templates, study))
    lines.extend(["```", "", "`.bidsignore`", "", "```gitignore"])
    lines.extend(bidsignore_lines(layout, templates, study))
    lines.append("```")
    return _wrap("IGNORE FILES", "\n".join(lines))


def result_names_block() -> str:
    """The result filename grammar, rendered from the patterns filenames are built with."""
    from tvbo.adapters.bids import RESULT_ENTITIES, RESULT_PATTERNS

    lines = ["```"]
    lines.extend(RESULT_PATTERNS)
    lines.append("```")
    lines.extend(["", "| entity | identifies |", "|--------|------------|"])
    lines.extend(f"| `{key}-` | {meaning} |" for key, meaning in RESULT_ENTITIES.items())
    return _wrap("RESULT NAMES", "\n".join(lines))


def spec_suffixes_block() -> str:
    """The suffix vocabulary, rendered from the suffix-to-class map itself."""
    from tvbo.adapters.bids import SPEC_SUFFIXES

    lines = ["| suffix | declares |", "|--------|----------|"]
    lines.extend(f"| `_{suffix}` | one `{cls}` |" for suffix, cls in SPEC_SUFFIXES.items())
    return _wrap("SPEC SUFFIXES", "\n".join(lines))


def _note_for(line: str, layout: StudyLayout, study: str, templates: tuple[str, ...]) -> str:
    """Gloss for the entry this tree line renders, keyed by its name and depth.

    The first sentence of the entry's own ``description``, whole rather than clipped, so every note is a complete thought and the record stays the only place the text is written.
    """
    depth = (len(line) - len(line.lstrip())) // 2
    name = line.strip().rstrip("/")
    for rel, entry in [*walk(layout, templates), *iter_files(layout, templates)]:
        if rel.count("/") + 1 == depth and interpolate(str(entry.name), study) == name:
            head = _sentence(entry)
            return head[:1].lower() + head[1:] if head else ""
    return ""


def splice_layout(text: str, block: str) -> str:
    """Replace the marker-delimited layout region of ``text``, or append it under a heading."""
    return _splice(text, "STUDY LAYOUT", block, fallback="## Layout")


def _splice(text: str, name: str, block: str, fallback: str | None = None) -> str:
    """Replace the region named ``name`` in ``text`` with ``block``.

    A document opts into a generated region by carrying its markers; one it does not carry is left alone rather than being appended, so a page decides which references it renders. ``fallback`` appends the block under that heading when the markers are absent, which is how a freshly scaffolded README gets its tree.
    """
    import re

    begin, end = markers(name)
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), flags=re.DOTALL)
    if pattern.search(text):
        return pattern.sub(lambda _: block, text)
    return f"{text.rstrip()}\n\n{fallback}\n\n{block}\n" if fallback else text


def sync_layout(
    dest: Path,
    layout: StudyLayout | None = None,
    study: str = "<Study>",
    templates: tuple[str, ...] = (),
) -> bool:
    """Rewrite every generated region ``dest`` carries. Returns True when the file changed.

    The layout tree is appended when absent; the three reference tables are only refreshed where a document already asks for them.
    """
    layout = layout or load_layout()
    before = dest.read_text(encoding="utf-8")
    after = splice_layout(before, layout_block(layout, study, templates))
    after = _splice(after, "BIDS EXCEPTIONS", exceptions_block(layout, study, templates))
    after = _splice(after, "IGNORE FILES", ignore_files_block(layout, study, templates))
    after = _splice(after, "RESULT NAMES", result_names_block())
    after = _splice(after, "SPEC SUFFIXES", spec_suffixes_block())
    if after == before:
        return False
    dest.write_text(after, encoding="utf-8")
    return True
