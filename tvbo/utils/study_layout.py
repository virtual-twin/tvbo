"""Read the study-layout record and answer questions about it.

``schema/study_layout.yaml`` is the single ground truth for where a study dataset keeps its specification, its inputs, its results, its figures and its provenance. It is a LinkML instance of :class:`StudyLayout`, authored beside the schema that types it, and materialized into the package tree at ``tvbo/rules/study_layout.yaml`` by ``hatch_build.py`` so a wheel-installed tvbo resolves it too.

Everything that needs to know the layout comes through this module: the scaffolder that creates a study, the resolvers that turn a role into a path, the writers that generate ``.gitignore`` and ``.bidsignore``, the validator that checks a study against the record, and the documentation that renders the tree. Nothing restates the layout, so it cannot drift.

A resolver asks for a *role*, never for a literal path::

    from tvbo.utils.study_layout import study_path
    out = study_path("results", root=study_root)   # <root>/derivatives/tvbo

Moving or renaming a directory is then a one-line edit to the record.

One record covers every kind of study: an entry carrying ``in_templates`` belongs only to the named ``tvbo study init --template`` kinds and one carrying ``not_in_templates`` to every kind but those, so ``templates=("replication",)`` selects the replication layout out of the same tree the general one comes from. The kinds themselves are declared in the record's ``templates:``, which is what :func:`check_templates` matches a requested name against.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import cache, lru_cache
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

    An entry with no ``in_templates`` is part of every study. One that names templates is included only when a requested template is among them, or when ``templates`` is :data:`ANY_TEMPLATE`. ``not_in_templates`` is the complement, for an entry every kind has except the one that replaces it.
    """
    if templates is ANY_TEMPLATE:
        return True
    if any(str(t) in templates for t in entry.not_in_templates or []):
        return False
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


def template_names(layout: StudyLayout | None = None) -> list[str]:
    """The study kinds the record declares, in the order it names them."""
    return list((layout or load_layout()).templates or {})


def check_templates(templates: Iterable[str], layout: StudyLayout | None = None) -> tuple[str, ...]:
    """*templates*, or raise naming the kinds the record does declare.

    A template name is matched, never registered: an undeclared one selects no entry, so without this a typo scaffolds the general study and reports success.
    """
    known = template_names(layout)
    unknown = [t for t in templates if t not in known]
    if unknown:
        raise KeyError(f"No study template named {', '.join(repr(t) for t in unknown)} in the layout record. Declared templates: {', '.join(known)}")
    return tuple(templates)


def template_for(entry, templates: tuple[str, ...] | None = ()) -> str | None:
    """The seed *entry* starts from under *templates*, or ``None`` when it has no seed.

    A variant may supersede the default: the same file can need different starting text once the variant says what kind of study this is, and a variant that states only the difference keeps one record entry per file. Requested templates are consulted in the order the caller gave them, so the first one naming a seed wins and the choice does not depend on how the record happens to be ordered.
    """
    variants = getattr(entry, "template_variants", None) or {}
    for name in templates or ():
        chosen = variants.get(name) if hasattr(variants, "get") else None
        if chosen is not None and getattr(chosen, "template", None):
            return str(chosen.template)
    return str(entry.template) if getattr(entry, "template", None) else None


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


def study_root(inside: Path | str, layout: StudyLayout | None = None) -> Path:
    """The study dataset ``inside`` belongs to — the nearest ancestor declaring itself one.

    A BIDS dataset is identified by its own ``dataset_description.json``, so that file is what the walk looks for; a helper handed a figure path can then resolve any other role without its caller passing a root it already implies. Raises rather than guessing, because a wrong root silently writes into the wrong study.
    """
    marker = file_relpath("dataset_description", layout=layout)
    start = Path(inside).resolve()
    for candidate in (start, *start.parents):
        if (candidate / marker).is_file():
            return candidate
    raise FileNotFoundError(
        f"{start} is not inside a study dataset: no {marker} in it or any parent. "
        f"Create one with `tvbo study init`, or pass the study root explicitly."
    )


def outermost_study_root(inside: Path | str, layout: StudyLayout | None = None) -> Path:
    """The outermost study dataset *inside* belongs to — the holder of a study-of-studies, not the member.

    :func:`study_root` answers "which study is this file in", which is what a member's own paths resolve against. This answers "which tree is it part of", which is what a reference from one member to another has to be resolved within: searching wider than the shared root is how a binding silently finds a same-named study in an unrelated checkout.
    """
    marker = file_relpath("dataset_description", layout=layout)
    start = Path(inside).resolve()
    found = None
    for candidate in (start, *start.parents):
        if (candidate / marker).is_file():
            found = candidate
    if found is None:
        raise FileNotFoundError(f"{start} is not inside a study dataset: no {marker} in it or any parent.")
    return found


_DECLARED_CITEKEY = re.compile(r"^citekey:\s*[\"']?([\w.-]+)", re.MULTILINE)
"""A recipe's own identity, read from its root-level ``citekey``. Anchored at column zero so a nested slot cannot be mistaken for the document's."""


def study_names(root: Path | str, layout: StudyLayout | None = None) -> set[str]:
    """Every name a study answers to: its directory, the stem of each entry recipe in it, and the ``citekey`` each recipe declares.

    A reference names a study the way its author knows it, which is the recipe stem (`Jansen1995.yaml`) far more often than anything else, and the directory usually agrees. The declared ``citekey`` is accepted too, because it is the only one of the three the study states about itself rather than inheriting from where it happens to sit: `tvbo-manuscript.yaml` in `tvbo-manuscript/` declares `citekey: tvbo_manuscript`, and its own results are named by that.
    """
    root = Path(root)
    names = {root.name}
    if not (root / file_relpath("dataset_description", layout=layout)).is_file():
        return names
    for entry in root.glob("*.yaml"):
        names.add(entry.stem)
        names.update(_DECLARED_CITEKEY.findall(entry.read_text(encoding="utf-8", errors="ignore")))
    return names


def sibling_study_root(name: str, inside: Path | str) -> Path | None:
    """The root of the study called *name*, resolved from a path inside the tree that holds it.

    A cross-study reference (``tvbo:exp/<study>/exp-N``) names a study rather than a directory, so the name has to be matched against the tree the referring study is itself part of. The search is bounded by :func:`outermost_study_root` and matches on :func:`study_names`, which is the same rule `study_manifest._owning_results_root` applies to a prose binding — the difference is only that this one has no loaded tree to walk and reads the filesystem instead.

    Returns ``None`` when *inside* is not part of a study dataset at all: there is then no tree for the name to be resolved within and none for it to contradict, so the caller reads its own results exactly as it did before study segments meant anything.

    Raises on no match within a real tree, and on an ambiguous one. Both are cases where continuing would bind the reference to *a* container rather than to the one it names, and a figure drawn from the wrong study's run is the failure this whole path exists to prevent.

    The answer is memoised per referring study rather than per path: every file in one study resolves a name identically, so the recursive scan runs once for a study that asks a hundred times, and two studies can no longer read each other's answer the way caching the raw path allowed. ``sibling_study_root.cache_clear()`` drops it, which a caller needs only when the tree itself changes under a live process.
    """
    try:
        own = study_root(Path(inside).resolve())
    except FileNotFoundError:
        return None
    return _sibling_study_root(name, own)


@cache
def _sibling_study_root(name: str, own: Path) -> Path:
    """The study called *name* within the tree holding *own*, which is the whole of the resolution that depends on the filesystem."""
    if name in study_names(own):
        return own
    outer = outermost_study_root(own)
    marker = file_relpath("dataset_description")
    matches = sorted(
        {d.parent for d in outer.rglob(marker) if name in study_names(d.parent)},
        key=lambda p: str(p),
    )
    if not matches:
        raise FileNotFoundError(
            f"cross-study reference names study {name!r}, which is not in the tree under {outer}. "
            "A reference resolves within the study-of-studies that holds the referring study; "
            "a study outside it has to be included as a member before its results can be named."
        )
    if len(matches) > 1:
        listed = ", ".join(str(m.relative_to(outer)) for m in matches)
        raise FileNotFoundError(f"cross-study reference names study {name!r}, which matches more than one member: {listed}.")
    return matches[0]


sibling_study_root.cache_clear = _sibling_study_root.cache_clear
"""The memo lives on the private worker, so the public name carries the hook a caller reaches for."""


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


def _generated_header(templates: tuple[str, ...] = ()) -> str:
    """First line of a generated ignore file, naming the record and the variant it was built for.

    The template belongs in the file because nothing else in a study records it: a replication ignores two entries a plain study has never heard of, so a checker comparing the file against the record has to know which variant to compare against. Stating it here means the file answers that itself, rather than every caller having to be told.
    """
    variant = f" (template: {', '.join(templates)})" if templates else ""
    return f"# Generated by `tvbo study init` from {SOURCE_NAME}{variant}. Edit the record, not this file."


def templates_of(text: str) -> tuple[str, ...]:
    """The templates a generated ignore file was written for, read back from its header.

    Found anywhere in the file, not only on the first line: a project may keep its own rules above the generated ones, and then the header that says what kind of study this is sits in the middle of the file rather than at the top. Keyed on the same :data:`SOURCE_NAME` :func:`_generated_header` writes, so the reader cannot look for a string the writer no longer emits.
    """
    marker = SOURCE_NAME
    for line in text.splitlines():
        if marker in line and "(template: " in line:
            named = line.split("(template: ", 1)[1].split(")", 1)[0]
            return tuple(t.strip() for t in named.split(",") if t.strip())
    return ()


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
    lines = [_generated_header(templates)]
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
    lines = [_generated_header(templates)]
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


def _uncapitalize(text: str) -> str:
    """``text`` as the continuation of the tree line it annotates.

    A gloss reads on from the entry it follows, so an ordinary sentence-initial capital is dropped. A first word carrying capitals of its own — an acronym like ``A/B``, a class name — keeps them: lowercasing those changes what the word is, not just how the sentence starts.
    """
    first = text.split(" ", 1)[0][1:]
    return text[:1].lower() + text[1:] if first.islower() or not first else text


def _note_for(line: str, layout: StudyLayout, study: str, templates: tuple[str, ...]) -> str:
    """Gloss for the entry this tree line renders, keyed by its name and depth.

    The first sentence of the entry's own ``description``, whole rather than clipped, so every note is a complete thought and the record stays the only place the text is written.
    """
    depth = (len(line) - len(line.lstrip())) // 2
    name = line.strip().rstrip("/")
    for rel, entry in [*walk(layout, templates), *iter_files(layout, templates)]:
        if rel.count("/") + 1 == depth and interpolate(str(entry.name), study) == name:
            head = _sentence(entry)
            return _uncapitalize(head) if head else ""
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


def is_network_companion(path) -> bool:
    """Whether *path* is the connectome companion a run writes beside its result, rather than a result itself.

    The companion is named by its trailing ``_network`` entity, so that is what this tests. Matching the bare substring instead — which three call sites did — hides every container whose own name happens to contain the word: an analysis called ``network_scaling`` writes ``ana-networkscaling_result.h5``, and a figure binding it silently found no container at all.
    """
    return Path(path).name.split(".")[0].endswith("_network")
