"""Read a cohort: a named subset of a dataset's subjects, each holding its own files under the dataset root.

A ``Dataset`` says where per-subject files live and a ``Cohort`` (``Dataset.cohorts``) names which subjects count. A ``DataRef`` whose WHERE is ``cohort`` reads one array from each member's own file, the file chosen by a BIDS entity ``query`` and the array by ``output``. This module is that read, and the one definition of how a subject's file is found under a dataset root, shared with the per-subject fit targets of an experiment (``source: [dataset.subject.<measure>]``).

Nothing here aggregates. A cohort read returns every member, labelled, on a leading ``subject`` axis; a mean, a presence count or a normalisation constant over the cohort is an ``Analysis`` that uses it, so the reduction is declared in the recipe rather than performed by the reader.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from tvbo.data.analysis_io import _slot
from tvbo.data.dataref import SUBJECT_DIM, sel_dict
from tvbo.data.registry import stem_entities
from tvbo.utils import as_list, keyed_items

BIDS_ENTITY_SHORT_KEYS = {
    "template": "tpl",
    "cohort": "cohort",
    "reconstruction": "rec",
    "segmentation": "seg",
    "scale": "scale",
    "density": "den",
    "atlas": "atlas",
    "acquisition": "acq",
    "hemi": "hemi",
    "desc": "desc",
}
"""``BidsEntities`` attribute names to the short keys a file name spells them with; ``suffix`` is the trailing file-name component, not an entity."""


def short_entities(obj) -> dict:
    """A ``BidsEntities`` object as ``{short_key: value}`` for the entities it sets."""
    return {short: str(_slot(obj, attr)) for attr, short in BIDS_ENTITY_SHORT_KEYS.items() if _slot(obj, attr) is not None}


def query_entities(query) -> tuple[dict, str | None]:
    """Split a ``BidsEntities`` query into its key-value entities and its ``suffix``."""
    if query is None:
        return {}, None
    suffix = _slot(query, "suffix")
    return short_entities(query), (str(suffix) if suffix is not None else None)


def match_subject_files(files: Iterable[Path], ents: dict, suffix: str | None):
    """Yield ``(subject, path)`` for the files whose parsed entities match *ents* and *suffix*."""
    for f in files:
        file_ents = stem_entities(f.stem)
        if not file_ents.get("sub"):
            continue
        if suffix is not None and not f.stem.endswith(f"_{suffix}"):
            continue
        if all(file_ents.get(k) == v for k, v in ents.items()):
            yield file_ents["sub"], f


def subject_sidecars(root: Path, subject: str = "*") -> list[Path]:
    """Every sidecar in the subject directories under *root*, or in one subject's, at any depth: BIDS keeps a subject's files in session and datatype folders."""
    return sorted(Path(root).glob(f"sub-{subject}/**/*.yaml"))


def subject_file(root: Path, subject: str, ents: dict, suffix: str | None) -> Path:
    """The one sidecar of *subject* under *root* matching *ents* and *suffix*; none raises ``FileNotFoundError``, several ``ValueError``."""
    sub = str(subject).removeprefix("sub-")
    matches = [f for s, f in match_subject_files(subject_sidecars(root, sub), ents, suffix) if s == sub]
    if not matches:
        raise FileNotFoundError(f"No file for sub-{sub} matching query {ents} suffix={suffix!r} under {root}.")
    if len(matches) > 1:
        raise ValueError(
            f"Query {ents} suffix={suffix!r} is ambiguous for sub-{sub}: {[m.name for m in matches]}. Add entities to disambiguate."
        )
    return matches[0]


def dataset_root(dataset, source_dir=None) -> Path:
    """``dataset.bids_root`` as a path, a relative one resolved against the directory of the declaring spec."""
    root = _slot(dataset, "bids_root")
    if not root:
        raise ValueError(
            f"dataset {_slot(dataset, 'dataset_id')!r} declares no `bids_root`, so its subjects' files cannot be found."
        )
    root = Path(str(root))
    if not root.is_absolute() and source_dir is not None:
        root = Path(source_dir) / root
    return root


def find_cohort(datasets, cohort_id: str):
    """The ``(dataset, cohort)`` pair declaring *cohort_id* among *datasets*.

    A cohort identifier is unique within a study, so a reference names it alone and the dataset follows. One dataset reached twice (an experiment's own that the study also lists) is one declaration, so the first copy that declares the cohort is taken, and a copy that does not declare it hides nothing. None declaring it, or two datasets, raises with the identifiers that do exist.
    """
    hits, known, owners = [], set(), set()
    for dataset in as_list(datasets):
        dataset_id = _slot(dataset, "dataset_id")
        for key, cohort in keyed_items(_slot(dataset, "cohorts"), "cohorts"):
            cid = str(_slot(cohort, "cohort_id", key))
            known.add(cid)
            if cid != str(cohort_id) or (dataset_id is not None and str(dataset_id) in owners):
                continue
            owners.add(str(dataset_id))
            hits.append((dataset, cohort))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise LookupError(
            f"cohort {cohort_id!r} is not declared by any dataset of this study (declared: {sorted(known) or 'none'}). "
            "A cohort is a named subset under `datasets: [{cohorts: [...]}]`."
        )
    owners = [str(_slot(dataset, "dataset_id")) for dataset, _ in hits]
    raise LookupError(
        f"cohort {cohort_id!r} is declared by {len(hits)} datasets ({owners}); a cohort identifier must be unique within a study."
    )


def subject_entries(subjects) -> dict:
    """A ``subjects`` slot as ``{subject id: entry}`` in declared order, each id without its ``sub-`` prefix. The slot holds bare ids, ``Subject`` records or a mapping keyed by id, and this is the one place the three are read, so the fan-out, a cohort's member check and a kit's frozen subject list name every subject the same way."""
    return {
        str(s if isinstance(s, str) else _slot(s, "subject_id", key)).removeprefix("sub-"): s
        for key, s in keyed_items(subjects, "subjects")
    }


TABLE_DELIMITERS = {".tsv": "\t", ".csv": ","}
"""File suffixes read as a table with one row per subject under ``participant_id`` (the BIDS participants and phenotype layout), each with the delimiter its cells are split by."""

TABLE_SUFFIXES = tuple(TABLE_DELIMITERS)

RECORD_SUFFIXES = (".yaml", ".yml", ".json")
"""File suffixes read with the recipe loader, so the file holds what the slot it stands in for would hold."""

MISSING_CELLS = ("", "n/a", "na", "nan")
"""What a table cell without a value reads as, compared case-insensitively; ``n/a`` is the BIDS spelling."""

_FILE_READS: dict = {}


def declared_file(path, source_dir=None, *, what: str) -> Path:
    """The file a dataset record names by *path*, found as every such path is: ``members_file``, ``subjects_file`` and ``phenotype_file``.

    An absolute path is taken as written. A relative one is read from *source_dir*, the directory of the experiment or network file the dataset is part of (also for a dataset that file brings in by ``!include`` from elsewhere), and when the file is not there from the root of the study that directory belongs to (:func:`tvbo.utils.study_layout.study_root`), so a study names the lists it keeps by their place in the study and a kit names its own copies by their place beside its frozen spec. With no *source_dir* a relative path is read from the working directory. *what* names the declaring slot in the error a missing file raises, which lists the places tried.
    """
    given = Path(str(path))
    places = [given]
    if source_dir is not None and not given.is_absolute():
        from tvbo.utils.study_layout import study_root

        places = [Path(source_dir) / given]
        try:
            places.append(Path(study_root(source_dir)) / given)
        except (StopIteration, LookupError, OSError):
            pass
    places = list(dict.fromkeys(place.resolve() for place in places))
    for place in places:
        if place.is_file():
            return place
    raise FileNotFoundError(
        f"{what} names the file {str(path)!r}, which is not at {' nor at '.join(str(place) for place in places)}."
    )


def _read_once(path: Path, reader, *key, also: tuple = ()):
    """``reader(path, *key)``, read again only when the size or modification time of the file, or *also*, has changed: a list its producer extends is seen at the next use and an unchanged one is parsed once. On a file system whose timestamps are coarser than the interval between two writes, a rewrite that keeps the file's size within one tick is not seen until the file changes again."""
    stat = path.stat()
    held_under, stamp = (str(path), reader.__name__, *key), (stat.st_mtime_ns, stat.st_size, *also)
    held = _FILE_READS.get(held_under)
    if held is None or held[0] != stamp:
        held = _FILE_READS[held_under] = (stamp, reader(path, *key))
    return held[1]


def _read_table(path: Path) -> tuple[list[str], list[tuple[int, str, list[str]]]]:
    """The header of the table at *path* and its rows as ``(row number, participant, cells)``, the participant without its ``sub-`` prefix.

    The delimiter follows the suffix (:data:`TABLE_DELIMITERS`), a leading byte-order mark is dropped, a row whose cells are all empty is skipped, and rows are numbered as the file numbers them, the header being row 1. The table has to carry a ``participant_id`` column, and a row with a value but no participant raises with its number.
    """
    import csv

    with open(path, newline="", encoding="utf-8-sig") as handle:
        numbered = [
            (number, row)
            for number, row in enumerate(csv.reader(handle, delimiter=TABLE_DELIMITERS[path.suffix.lower()]), start=1)
            if any(cell.strip() for cell in row)
        ]
    header = [cell.strip() for cell in numbered[0][1]] if numbered else []
    if "participant_id" not in header:
        raise ValueError(
            f"{path.name} is read as a participants table and has no `participant_id` column (columns: {header or 'none'})."
        )
    column, rows = header.index("participant_id"), []
    for number, row in numbered[1:]:
        participant = row[column].strip() if len(row) > column else ""
        if not participant:
            raise ValueError(f"{path.name}, row {number} names no participant.")
        rows.append((number, participant.removeprefix("sub-"), row))
    return header, rows


def _read_listing(path: Path) -> dict:
    """The subjects the list file at *path* names, as ``{subject id: Subject}`` in the file's order.

    By suffix the file holds what a recipe would state in the slot (:data:`RECORD_SUFFIXES`, parsed by the recipe loader, so an ``!include`` of the same file gives the same entries), a table read by its ``participant_id`` column (:data:`TABLE_SUFFIXES`), or one identifier per line, blank lines and lines starting with ``#`` skipped. The entries are built the way a dataset builds its ``subjects``. An identifier listed twice raises with a count, and no error repeats an entry.
    """
    import copy

    from tvbo.datamodel.schema import Dataset

    suffix = path.suffix.lower()
    if suffix in RECORD_SUFFIXES:
        from tvbo.utils import yaml_loader

        content = copy.deepcopy(yaml_loader.strip_envelope(yaml_loader.load_as_dict(path)))
    elif suffix in TABLE_SUFFIXES:
        content = [participant for _, participant, _ in _read_table(path)[1]]
    else:
        lines = (line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines())
        content = [line for line in lines if line and not line.startswith("#")]
    if content is None:
        return {}
    if not isinstance(content, (list, dict)):
        raise ValueError(f"{path.name} holds no list of subjects.")
    if isinstance(content, list):
        names = [
            str(_slot(entry, "subject_id") if isinstance(entry, dict) else entry).removeprefix("sub-") for entry in content
        ]
        repeated = sum(1 for n in Counter(names).values() if n > 1)
        if repeated:
            raise ValueError(f"{path.name} lists {repeated} identifier(s) more than once.")
        content = [entry if isinstance(entry, dict) else str(entry) for entry in content]
    try:
        return subject_entries(Dataset(dataset_id="listed", subjects=content).subjects)
    except Exception as exc:
        raise ValueError(f"{path.name} does not hold a list of subjects ({type(exc).__name__}).") from None


def _listed(path, source_dir, what: str) -> dict:
    """The subjects the list file a slot names holds (:func:`declared_file`, :func:`_read_listing`), with *what*, the declaring slot, leading any error."""
    found = declared_file(path, source_dir, what=what)
    try:
        return dict(_read_once(found, _read_listing))
    except ValueError as exc:
        raise ValueError(f"{what}: {exc}") from None


def dataset_subjects(dataset, source_dir=None) -> dict:
    """The subjects *dataset* lists, as ``{subject id: entry}`` in declared order: its ``subjects``, or the content of its ``subjects_file`` (:func:`declared_file`, read against *source_dir*). Empty for a dataset that lists none; one that states both raises, and so does a file that lists nobody."""
    typed = subject_entries(_slot(dataset, "subjects"))
    path = _slot(dataset, "subjects_file")
    if not path:
        return typed
    name = _slot(dataset, "dataset_id")
    if typed:
        raise ValueError(
            f"dataset {name!r} states both `subjects` and `subjects_file`; a dataset lists its subjects in one of the two."
        )
    listed = _listed(path, source_dir, f"dataset {name!r}: `subjects_file`")
    if not listed:
        raise ValueError(f"dataset {name!r}: `subjects_file` {str(path)!r} lists no subject.")
    return listed


def _companion(path: Path) -> str | None:
    """The companion file the ``Phenotype`` sidecar at *path* keeps its values in."""
    from tvbo.classes.phenotype import Phenotype

    return getattr(Phenotype.from_file(path), "_h5_path", None)


def _read_measure(path: Path, measure: str) -> tuple[list[str], list[float]]:
    """The subjects of the phenotype table at *path* and their values of *measure*, in the table's order: a table (:data:`TABLE_SUFFIXES`) by its ``participant_id`` and *measure* columns, a record (:data:`RECORD_SUFFIXES`) as a ``Phenotype`` sidecar with its companion. A cell without a value (:data:`MISSING_CELLS`) is NaN, and no error repeats a row."""
    import math

    if path.suffix.lower() in TABLE_SUFFIXES:
        header, rows = _read_table(path)
        if measure not in header:
            raise ValueError(
                f"{path.name} has no column {measure!r} (columns: {[c for c in header if c != 'participant_id']})."
            )
        where, subjects, values = header.index(measure), [], []
        for number, participant, row in rows:
            cell = row[where].strip() if len(row) > where else ""
            try:
                value = math.nan if cell.lower() in MISSING_CELLS else float(cell)
            except ValueError:
                raise ValueError(
                    f"{path.name}, row {number}: column {measure!r} holds a value that is not a number."
                ) from None
            subjects.append(participant)
            values.append(value)
        return subjects, values
    from tvbo.classes.phenotype import Phenotype

    table = Phenotype.from_file(path)
    if measure not in [str(m) for m in table.measures or []]:
        raise ValueError(f"{path.name} declares no measure {measure!r} (measures: {[str(m) for m in table.measures or []]}).")
    subjects = [str(s).removeprefix("sub-") for s in table.subjects or []]
    values = [float(v) for v in table.get(measure)]
    if len(values) != len(subjects):
        raise ValueError(f"{path.name}: measure {measure!r} holds {len(values)} values for {len(subjects)} subjects.")
    return subjects, values


def phenotype_measure(path, measure: str) -> tuple[list[str], list[float]]:
    """The subjects of the phenotype table at *path* and their values of *measure* (:func:`_read_measure`), read again only when the table or the companion that holds its values has changed. A file of neither kind is refused by its suffix, and a subject the table lists twice raises with a count."""
    path, also = Path(path), ()
    if path.suffix.lower() not in (*TABLE_SUFFIXES, *RECORD_SUFFIXES):
        raise ValueError(
            f"{path.name} is named as a phenotype table, and a {path.suffix or 'suffixless'} file is not read as one: a table is {', '.join(TABLE_SUFFIXES)} and a `Phenotype` sidecar {', '.join(RECORD_SUFFIXES)}."
        )
    if path.suffix.lower() not in TABLE_SUFFIXES:
        companion = _read_once(path, _companion)
        if companion and Path(companion).is_file():
            also = (Path(companion).stat().st_mtime_ns, Path(companion).stat().st_size)
    subjects, values = _read_once(path, _read_measure, str(measure), also=also)
    repeated = sum(1 for n in Counter(subjects).values() if n > 1)
    if repeated:
        raise ValueError(f"{path.name} lists {repeated} subject(s) more than once, so a value cannot be assigned to each.")
    return list(subjects), list(values)


def _rule_members(dataset, cohort, source_dir, chain: tuple) -> list[str]:
    """The members a cohort's ``rule`` admits (``CohortRule``): the candidates whose value of the rule's measure falls in the group it names.

    The candidates are the members of the cohort ``of`` names, else the subjects the dataset lists, else the table's subjects, and the members come in their order. A candidate without a row or with a value that is not finite is counted and refused, or under ``missing: exclude`` left out before any group is formed. The groups are those of :func:`tvbo.analysis.statistics.group_index` over the remaining values, counted from 1 for the lowest: ``equal_width`` cuts the range from their smallest to their largest into bins closed below and open above, the last closed on both sides; ``equal_count`` cuts their ascending order, equal values in the candidates' order, into runs whose sizes differ by at most one, the longer runs first. *chain* holds the cohorts whose rules led here, so a rule that leads back to one of them raises. No message names a subject or a value.
    """
    import numpy as np

    from tvbo.analysis.statistics import GROUPINGS, group_index

    cid, rule, owner = str(_slot(cohort, "cohort_id")), _slot(cohort, "rule"), _slot(dataset, "dataset_id")
    measure, grouping = str(_slot(rule, "measure")), str(_slot(rule, "grouping"))
    n_groups, group = int(_slot(rule, "n_groups")), int(_slot(rule, "group"))
    if grouping not in GROUPINGS:
        raise ValueError(f"cohort {cid!r}: its rule's grouping must be one of {GROUPINGS}, got {grouping!r}.")
    if n_groups < 1 or not 1 <= group <= n_groups:
        raise ValueError(
            f"cohort {cid!r}: its rule asks for group {group} of {n_groups}; groups are counted from 1 to `n_groups`."
        )
    table = _slot(rule, "phenotype_file") or _slot(dataset, "phenotype_file")
    if not table:
        raise ValueError(
            f"cohort {cid!r} is admitted by a rule on {measure!r}, and neither the rule nor the dataset {owner!r} names a `phenotype_file` to read it from."
        )
    path = declared_file(table, source_dir, what=f"cohort {cid!r}: `phenotype_file`")
    try:
        subjects, values = phenotype_measure(path, measure)
    except ValueError as exc:
        raise ValueError(f"cohort {cid!r}: {exc}") from None
    of = _slot(rule, "of")
    if of is not None:
        if str(of) in (*chain, cid):
            raise ValueError(
                f"cohort {cid!r}: its rule splits the cohort {str(of)!r}, whose own rule leads back to it ({' -> '.join((*chain, cid, str(of)))})."
            )
        declared = {str(_slot(c, "cohort_id", key)): c for key, c in keyed_items(_slot(dataset, "cohorts"), "cohorts")}
        if str(of) not in declared:
            raise LookupError(
                f"cohort {cid!r}: its rule splits the cohort {str(of)!r}, which the dataset {owner!r} does not declare (declared: {sorted(declared)})."
            )
        candidates = _members(dataset, declared[str(of)], source_dir, (*chain, cid))
    else:
        candidates = list(dataset_subjects(dataset, source_dir)) or subjects
    value_of = dict(zip(subjects, values, strict=True))
    scores = np.array([value_of.get(candidate, np.nan) for candidate in candidates], dtype=float)
    finite = np.isfinite(scores)
    if not finite.all() and str(_slot(rule, "missing", "refuse")) != "exclude":
        raise ValueError(
            f"cohort {cid!r}: {int((~finite).sum())} of its {len(candidates)} candidates have no finite value of {measure!r} in {path.name}. `missing: exclude` on the rule leaves them out of every group."
        )
    held, scores = [c for c, ok in zip(candidates, finite, strict=True) if ok], scores[finite]
    if not held:
        raise ValueError(
            f"cohort {cid!r}: none of its {len(candidates)} candidates has a finite value of {measure!r} in {path.name}."
        )
    if grouping == "equal_width" and n_groups > 1 and scores.min() == scores.max():
        raise ValueError(
            f"cohort {cid!r}: the {len(held)} values of {measure!r} are all equal, so their range has no width to cut into {n_groups} bins."
        )
    index = group_index(scores, n_groups, grouping)
    members = [c for c, k in zip(held, index, strict=True) if k == group - 1]
    if not members:
        raise ValueError(
            f"cohort {cid!r}: group {group} of the {n_groups} {grouping} groups of {measure!r} holds none of the {len(held)} candidates."
        )
    return members


def _members(dataset, cohort, source_dir, chain: tuple) -> list[str]:
    """:func:`cohort_members`, with the cohorts whose rules led to this one in *chain*."""
    cid = _slot(cohort, "cohort_id")
    typed = [str(m).removeprefix("sub-") for m in as_list(_slot(cohort, "members"))]
    path, rule = _slot(cohort, "members_file"), _slot(cohort, "rule")
    stated = [
        slot for slot, given in (("members", bool(typed)), ("members_file", bool(path)), ("rule", rule is not None)) if given
    ]
    if len(stated) > 1:
        raise ValueError(
            f"cohort {cid!r} states its members through {' and '.join(f'`{slot}`' for slot in stated)}; a cohort names them in one way: typed, read from a file, or admitted by a rule."
        )
    if not stated:
        raise ValueError(f"cohort {cid!r} declares no `members`, `members_file` or `rule`.")
    if rule is not None:
        return _rule_members(dataset, cohort, source_dir, chain)
    if path:
        members = list(_listed(path, source_dir, f"cohort {cid!r}: `members_file`"))
        if not members:
            raise ValueError(f"cohort {cid!r}: `members_file` {str(path)!r} lists no member.")
    else:
        members = typed
        repeated = sorted(m for m, n in Counter(members).items() if n > 1)
        if repeated:
            raise ValueError(f"cohort {cid!r} lists {repeated} more than once.")
    listed = dataset_subjects(dataset, source_dir)
    strangers = [m for m in members if m not in listed] if listed else []
    if strangers:
        raise ValueError(
            f"cohort {cid!r} names {len(strangers)} member(s) the dataset {_slot(dataset, 'dataset_id')!r} does not list as subjects{f', e.g. {strangers[:5]}' if typed else ''}."
        )
    return members


def admission(dataset, cohort) -> str | None:
    """How *cohort* of *dataset* admits its members, in words a frozen record keeps in place of the list: the file that lists them, or the rule and the table it reads, each by file name. ``None`` for a cohort that types its members."""
    rule = _slot(cohort, "rule")
    if rule is None:
        path = _slot(cohort, "members_file")
        return f"listed in {Path(str(path)).name}" if path else None
    table = _slot(rule, "phenotype_file") or _slot(dataset, "phenotype_file")
    of, missing = _slot(rule, "of"), str(_slot(rule, "missing", "refuse"))
    return (
        f"group {_slot(rule, 'group')} of the {_slot(rule, 'n_groups')} {_slot(rule, 'grouping')} groups of {str(_slot(rule, 'measure'))!r}"
        f" in {Path(str(table)).name if table else 'the dataset table'}"
        f"{f' among the members of cohort {str(of)!r}' if of is not None else ''}"
        f"{', candidates without a value left out' if missing == 'exclude' else ''}"
    )


def lists_from_file(cohort) -> bool:
    """Whether *cohort* keeps its members out of the recipe: read from ``members_file`` or admitted by ``rule``. Whatever freezes such a cohort writes its members to a file again and never into a specification."""
    return bool(_slot(cohort, "members_file")) or _slot(cohort, "rule") is not None


def cohort_members(dataset, cohort, source_dir=None) -> list[str]:
    """The cohort's members, checked against the dataset's own subject list when it has one.

    A cohort states them in one way: typed (``members``, in declared order), read from a file (``members_file``, in the file's order, found by :func:`declared_file` against *source_dir*) or admitted by a rule on a phenotype measure (``rule``, :func:`_rule_members`). A list that is read or computed is read or computed on each call, so it follows its file, and a message about it gives counts and never an identifier.
    """
    return _members(dataset, cohort, source_dir, ())


def pinned_members(ref) -> tuple[list[str], bool] | None:
    """The members a cohort reference's ``sel`` on ``subject`` pins it to, as ``(members, several)``, or ``None`` for a reference that follows the run. *several* tells a list, read stacked in the order it gives, from one named member, read as that member's own array."""
    pinned = sel_dict(ref).get(SUBJECT_DIM)
    if pinned is None:
        return None
    several = isinstance(pinned, (list, tuple))
    return [str(m).removeprefix("sub-") for m in (pinned if several else [pinned])], several


def follows_run(ref) -> bool:
    """Whether *ref* reads the running subject's own member of a cohort: it names a cohort and pins no member. Such a reference needs a run with a subject that belongs to the cohort, so it narrows a per-subject fan-out; a pinned one reads the same members whichever subject runs and narrows nothing."""
    return _slot(ref, "cohort") is not None and pinned_members(ref) is None


def members_read(ref, datasets, subject=None, source_dir=None) -> tuple[list[str], bool]:
    """The members of its cohort *ref* reads in a run of *subject*, and whether they come back stacked along ``subject``; *source_dir* is where a member list kept in a file is found (:func:`cohort_members`).

    The one rule the reader, the per-subject fan-out, a kit's bundle and the on-device refusal share. A ``sel`` on ``subject`` pins the reference to the member or members it names, whichever subject runs (:func:`pinned_members`). A reference that pins none follows the run: the run's own member, and in a run with no subject the whole cohort, stacked in declared order. A member the cohort does not declare raises rather than being read anyway.
    """
    cohort_id = _slot(ref, "cohort")
    declared = cohort_members(*find_cohort(datasets, cohort_id), source_dir)
    pinned = pinned_members(ref)
    if pinned is not None:
        members, stacked = pinned
        if not members:
            raise ValueError(f"a reference to cohort {cohort_id!r} pins an empty list of members, so it has nothing to read.")
        strangers = [f"sub-{m}" for m in members if m not in declared]
        if strangers:
            raise LookupError(
                f"a reference pins {', '.join(strangers)}, not a member of cohort {cohort_id!r} ({len(declared)} members), so it has nothing to read there."
            )
        return members, stacked
    if subject is None:
        return declared, True
    sub = str(subject).removeprefix("sub-")
    if sub not in declared:
        raise LookupError(
            f"sub-{sub} is not a member of cohort {cohort_id!r} ({len(declared)} members), so the reference has nothing to read for this run."
        )
    return [sub], False


def describe_selection(selection: dict) -> str:
    """One line stating how cohorts narrowed a per-subject fan-out, from ``SimulationExperiment.subject_selection()``."""
    cohorts = ", ".join(f"{cohort_id} ({n} members)" for cohort_id, n in selection["cohorts"].items())
    return (
        f"{len(selection['subjects'])} of {selection['dataset']} dataset subjects, the members of {cohorts}; "
        f"{len(selection['excluded'])} excluded"
    )


def read_member(root: Path, subject: str, query, output: str):
    """One subject's array *output* from the file *query* selects, as a labelled ``DataArray``.

    An edge matrix comes back on ``(node_i, node_j)`` and a per-node quantity on ``node``, both carrying the file's own node labels, so whatever consumes it aligns by name.
    """
    from tvbo.classes.network import Network

    path = subject_file(root, subject, *query_entities(query))
    return network_array(Network.load(str(path)), output, path.name)


def network_array(net, output: str, name: str):
    """*output* of the network *net* as a labelled ``DataArray``, as the network delivers it; *name* is how an error names the network.

    An edge layer comes back on ``(node_i, node_j)`` with the transforms the network declares on it applied, and a per-node quantity on ``node``. Both carry the network's own node labels, so whatever consumes the array aligns by name.
    """
    import numpy as np
    import xarray as xr

    from tvbo.data.param_io import resolve_network_node

    labels = [str(lbl) for lbl in net.node_labels]
    repeated = sorted(lbl for lbl, n in Counter(labels).items() if n > 1)
    if repeated:
        raise ValueError(f"{name}: node labels {repeated[:5]} occur more than once, so its arrays cannot be aligned by label.")
    # `matrix` answers a weight name with zeros on a file that holds none, which would read as an unconnected subject.
    from tvbo.classes.network import _declares_layer

    spellings = set(net._matrix_names(output))
    held = (
        any(str(n).lower() in spellings for n in net.matrix_names)
        or bool(net._placed_edges() and net.carries(output))
        or _declares_layer(
            net.edges or [], output
        )  # a layer declared with `used:` or `producer:` is read on demand and listed nowhere before
    )
    matrix = net.matrix(output, format="dense") if held else None
    if matrix is not None:
        return xr.DataArray(np.asarray(matrix), dims=("node_i", "node_j"), coords={"node_i": labels, "node_j": labels})
    vector = resolve_network_node(net, output)
    if vector is None:
        raise KeyError(
            f"{name} holds neither an edge matrix nor a per-node quantity named {output!r} (edge matrices: {list(net.matrix_names)})."
        )
    return xr.DataArray(np.asarray(vector), dims=("node",), coords={"node": labels})


def network_members(net, datasets, source_dir=None) -> list[str]:
    """The subjects a network that reads per subject can be bound to: the members of the cohorts it follows the run on, in declared order (:func:`cohort_members`, with *source_dir*).

    Cohorts with different members raise, because no one list then names every subject the network holds a layer for.
    """
    lists = {
        cohort_id: cohort_members(*find_cohort(datasets, cohort_id), source_dir) for cohort_id in net.per_subject_cohorts()
    }
    first_id, first = next(iter(lists.items()))
    for cohort_id, members in lists.items():
        if sorted(members) != sorted(first):
            raise ValueError(
                f"the network reads per subject from cohorts {first_id!r} ({len(first)} members) and {cohort_id!r} ({len(members)} members), whose members differ, so a reference to it has no one set of subjects to read."
            )
    return first


def read_network(ref, path, *, subject=None, datasets=None, results_root=None, source_dir=None):
    """Resolve a reference whose ``iri`` names the network at *path* to that network's array ``output``, as the network delivers it (:func:`network_array`).

    A network that holds its own arrays has one such array. One that reads a layer or a transform argument per subject has one per subject, and the reference follows the rule a ``cohort`` reference does: a ``sel`` on ``subject`` pins it to the member or members it names whichever subject runs (:func:`pinned_members`), and otherwise inside a run of *subject* it is that subject's, and with no subject it is every member's (:func:`network_members`). Several members come back stacked along a leading ``subject`` axis, each laid out on the first one's node labels by name. *datasets*, *results_root* and *source_dir* are what the network's own references resolve against.

    Raises:
        ValueError: If the reference pins no member, or pins members of a network that reads nothing per subject, which holds one array for every subject.
    """
    import xarray as xr

    from tvbo.data.network_io import load_declared_network

    output = _slot(ref, "output")
    if not output:
        raise ValueError(
            f"a reference to the network {path.name} needs `output:`, the edge layer or per-node quantity to take from it."
        )
    net = load_declared_network(path)
    pinned = pinned_members(ref)
    if pinned is not None and not net.per_subject_cohorts():
        raise ValueError(
            f"a reference to the network {path.name} pins `{SUBJECT_DIM}` members, but the network reads no cohort per subject, so it holds one array for every subject."
        )
    if not net.declares_references():
        return network_array(net, str(output), path.name)

    def bound(member):
        net.bind_references(subject=member, datasets=datasets, results_root=results_root, source_dir=source_dir)
        return network_array(net, str(output), path.name)

    if pinned is not None:
        members, stacked = pinned
        if not members:
            raise ValueError(
                f"a reference to the network {path.name} pins an empty list of members, so it has nothing to read."
            )
    elif subject is not None or not net.per_subject_cohorts():
        return bound(subject)
    else:
        members, stacked = network_members(net, datasets, source_dir), True
    if not stacked:
        return bound(members[0])
    parts = []
    for member in members:
        part = bound(member)
        parts.append(_on_labels_of(parts[0], part, member) if parts else part)
    return xr.concat(parts, dim=SUBJECT_DIM, coords="minimal", compat="override", join="exact").assign_coords(
        {SUBJECT_DIM: members}
    )


def _on_labels_of(first, part, subject: str):
    """*part* laid out on *first*'s node labels, in *first*'s order; a member whose label set differs raises."""
    for dim in first.dims:
        ours, theirs = list(first[dim].values), list(part[dim].values)
        if ours == theirs:
            continue
        if sorted(ours) != sorted(theirs):
            missing = sorted(set(ours) - set(theirs))
            raise ValueError(
                f"cohort member sub-{subject} carries a different node set on {dim!r} than the first member "
                f"({len(theirs)} against {len(ours)} nodes; missing e.g. {missing[:5]}), so the cohort cannot be stacked."
            )
        part = part.sel({dim: ours})
    return part


def read_cohort(ref, datasets, *, subject=None, source_dir=None):
    """Resolve a ``cohort`` reference to its labelled array, from the files of the members it reads in a run of *subject* (:func:`members_read`) and no other member's.

    One member comes back as its own array. Several come back stacked along a leading ``subject`` axis in the order they are read, each laid out on the first one's node labels by name.
    """
    import xarray as xr

    cohort_id = _slot(ref, "cohort")
    output = _slot(ref, "output")
    if not output:
        raise ValueError(f"a reference to cohort {cohort_id!r} needs `output:`, the array to take from each member's file.")
    dataset, _ = find_cohort(datasets, cohort_id)
    members, stacked = members_read(ref, datasets, subject, source_dir)
    root = dataset_root(dataset, source_dir)
    query = _slot(ref, "query")

    if not stacked:
        return read_member(root, members[0], query, str(output))

    parts = []
    for sub in members:
        part = read_member(root, sub, query, str(output))
        parts.append(_on_labels_of(parts[0], part, sub) if parts else part)
    return xr.concat(parts, dim=SUBJECT_DIM, coords="minimal", compat="override", join="exact").assign_coords(
        {SUBJECT_DIM: members}
    )
