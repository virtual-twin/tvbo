"""Read a cohort: a named subset of a dataset's subjects, each holding its own files under the dataset root.

A ``Dataset`` says where per-subject files live and a ``Cohort`` (``Dataset.cohorts``) names which subjects count. A ``DataRef`` whose WHERE is ``cohort`` reads one array from each member's own file, the file chosen by a BIDS entity ``query`` and the array by ``output``. This module is that read, and the one definition of how a subject's file is found under a dataset root, shared with the per-subject fit targets of an experiment (``source: [dataset.subject.<measure>]``).

Nothing here aggregates. A cohort read returns every member, labelled, on a leading ``subject`` axis; a mean, a presence count or a normalisation constant over the cohort is an ``Analysis`` that uses it, so the reduction is declared in the recipe rather than performed by the reader.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from tvbo.data.analysis_io import _slot
from tvbo.utils import as_list, keyed_items

BIDS_ENTITY_SHORT_KEYS = {
    "template": "tpl",
    "cohort": "cohort",
    "reconstruction": "rec",
    "segmentation": "seg",
    "scale": "scale",
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
    from tvbo.classes.network import _parse_bids_entities

    for f in files:
        file_ents = _parse_bids_entities(f.stem)
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
    """The one sidecar of *subject* under *root* matching *ents* and *suffix*; none or several raises."""
    sub = str(subject).removeprefix("sub-")
    matches = [f for s, f in match_subject_files(subject_sidecars(root, sub), ents, suffix) if s == sub]
    if not matches:
        raise ValueError(f"No file for sub-{sub} matching query {ents} suffix={suffix!r} under {root}.")
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


def cohort_members(dataset, cohort) -> list[str]:
    """The cohort's members in declared order, checked against the dataset's own subject list when it has one."""
    cid = _slot(cohort, "cohort_id")
    members = [str(m).removeprefix("sub-") for m in as_list(_slot(cohort, "members"))]
    if not members:
        raise ValueError(f"cohort {cid!r} declares no `members`.")
    repeated = sorted(m for m, n in Counter(members).items() if n > 1)
    if repeated:
        raise ValueError(f"cohort {cid!r} lists {repeated} more than once.")
    listed = {
        str(s if isinstance(s, str) else _slot(s, "subject_id", key)).removeprefix("sub-")
        for key, s in keyed_items(_slot(dataset, "subjects"), "subjects")
    }
    strangers = [m for m in members if m not in listed] if listed else []
    if strangers:
        raise ValueError(
            f"cohort {cid!r} names {len(strangers)} member(s) the dataset {_slot(dataset, 'dataset_id')!r} does not list as subjects, e.g. {strangers[:5]}."
        )
    return members


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
    import numpy as np
    import xarray as xr

    from tvbo.classes.network import Network
    from tvbo.data.param_io import resolve_network_node

    path = subject_file(root, subject, *query_entities(query))
    net = Network.load(str(path))
    labels = [str(lbl) for lbl in net.node_labels]
    repeated = sorted(lbl for lbl, n in Counter(labels).items() if n > 1)
    if repeated:
        raise ValueError(
            f"{path.name}: node labels {repeated[:5]} occur more than once, so its arrays cannot be aligned by label."
        )
    # `matrix` answers a weight name with zeros on a file that holds none, which would read as an unconnected subject.
    spellings = set(net._matrix_names(output))
    held = any(str(n).lower() in spellings for n in net.matrix_names) or bool(net._placed_edges() and net.carries(output))
    matrix = net.matrix(output, format="dense") if held else None
    if matrix is not None:
        return xr.DataArray(np.asarray(matrix), dims=("node_i", "node_j"), coords={"node_i": labels, "node_j": labels})
    vector = resolve_network_node(net, output)
    if vector is None:
        raise KeyError(
            f"{path.name} holds neither an edge matrix nor a per-node quantity named {output!r} (edge matrices: {list(net.matrix_names)})."
        )
    return xr.DataArray(np.asarray(vector), dims=("node",), coords={"node": labels})


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
    """Resolve a ``cohort`` reference to its labelled array.

    With *subject*, the reading run's own, the reference is that one member's array, and a subject outside the cohort raises rather than being read anyway. Without one it is the whole cohort: every member's array stacked along a leading ``subject`` axis in declared member order, each laid out on the first member's node labels by name.
    """
    import xarray as xr

    cohort_id = _slot(ref, "cohort")
    output = _slot(ref, "output")
    if not output:
        raise ValueError(f"a reference to cohort {cohort_id!r} needs `output:`, the array to take from each member's file.")
    dataset, cohort = find_cohort(datasets, cohort_id)
    members = cohort_members(dataset, cohort)
    root = dataset_root(dataset, source_dir)
    query = _slot(ref, "query")

    if subject is not None:
        sub = str(subject).removeprefix("sub-")
        if sub not in members:
            raise LookupError(
                f"sub-{sub} is not a member of cohort {cohort_id!r} ({len(members)} members), so the reference has nothing to read for this run."
            )
        return read_member(root, sub, query, str(output))

    parts = []
    for sub in members:
        part = read_member(root, sub, query, str(output))
        parts.append(_on_labels_of(parts[0], part, sub) if parts else part)
    return xr.concat(parts, dim="subject", coords="minimal", compat="override", join="exact").assign_coords(subject=members)
