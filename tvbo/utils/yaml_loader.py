"""YAML loader wrapper used by every ``Network.from_file`` / ``SimulationExperiment.from_file`` / ``SimulationStudy.from_file`` entry point in TVBO.

Extends :class:`linkml_runtime.utils.yamlutils.DupCheckYamlLoader` (the default LinkML loader, which already disallows duplicate keys) with two generally-useful YAML idioms:

* **Merge keys** (``<<: *anchor``) — standard YAML 1.1 semantics. Lets
  an inline override reuse another block with one slot changed:

  .. code-block:: yaml

      kuramoto_10hz: &kuramoto_10hz
        name: Kuramoto
        parameters: {omega: 0.0628}

      kuramoto_20hz:
        <<: *kuramoto_10hz
        parameters: {omega: 0.1257}

* **`!include`** — substitute the value at a directive's position with
  the contents of another YAML file. Paths are resolved relative to the directory of the file containing the directive; absolute paths are
  accepted as-is. Anchors in the included document are scoped to that document only (each include uses a fresh loader instance), so
  fragments are readable in isolation.

  .. code-block:: yaml

      experiments:
        - !include _experiments/exp1.yaml
        - !include _experiments/exp2.yaml

  The two idioms **compose**: an ``!include`` may appear as the value of a merge key, so a fragment file can be merged into a mapping alongside the
  mapping's own entries and alongside other anchors. Without this a shared fragment could only ever *replace* a whole slot, which forces every
  consumer of a partial fragment (a haemodynamic cascade shared by two models) to copy it instead:

  .. code-block:: yaml

      state_variables:
        <<: !include _balloon_states.yaml
        phi: {...}

      parameters:
        <<: [*model_params, !include _balloon_parameters.yaml]

Both idioms are pure data-format machinery; they don't introduce any TVBO-specific semantics into user YAMLs. The wrapper is transparent — any LinkML class can still load through ``yaml_loader.load`` and get back the same datamodel instance it would have produced before.
"""

from __future__ import annotations

import io
import os
import re
import warnings
from pathlib import Path
from typing import Any

import yaml
from linkml_runtime.loaders import yaml_loader as _linkml_yaml_loader
from linkml_runtime.utils.yamlutils import DupCheckYamlLoader

_MERGE_TAG = "tag:yaml.org,2002:merge"
_INCLUDE_TAG = "!include"

ENVELOPE_KEYS = ("tvbo_class", "schema_version")
"""Keys that annotate a serialized FILE's class and schema version rather than the object's slots. TVBO writes them itself — every one of the 121 network sidecars in the
database opens with ``tvbo_class: tvbo:Network`` — so its own loader has to accept them. They are dropped on every route into a class constructor — a document root, a plain
``!include``, and a fragment merged with ``<<: !include`` — and kept by :func:`load_as_dict`, whose callers dispatch on them."""

_ENVELOPE_CLASS = re.compile(r"^tvbo_class:\s*[\"']?([\w:.-]+)", re.MULTILINE)


def declared_class(source) -> str | None:
    """The class a document names in its own ``tvbo_class`` envelope, or ``None``.

    Accepts the parsed mapping or the raw YAML text, because a caller deciding WHICH class to parse as has not parsed yet — and a document that says what it is should never lose that argument to a guess about its shape. Only a root-level key counts, and the CURIE prefix is stripped, so ``tvbo:SimulationStudy`` answers ``SimulationStudy``.
    """
    if isinstance(source, dict):
        declared = source.get("tvbo_class")
    else:
        match = _ENVELOPE_CLASS.search(str(source))
        declared = match.group(1) if match else None
    return str(declared).split(":")[-1] if declared else None


def _flatten_map_constructor(loader: yaml.Loader, node: yaml.MappingNode, deep: bool = False) -> dict:
    """``DupCheckYamlLoader`` map constructor augmented with merge-key support.

    Standard YAML merge semantics: an explicit entry overrides any value pulled in by a ``<<:`` merge. We preserve the original LinkML duplicate-key safety check for *explicit* duplicates (the same key written twice by the author), but suppress it for keys whose collision came from a merge expansion — those are silently overridden by the explicit entry.
    """
    if not isinstance(node, yaml.MappingNode):
        from yaml.constructor import ConstructorError

        raise ConstructorError(None, None, f"expected a mapping node, but found {node.id}", node.start_mark)
    # Catch duplicate explicit keys in the user's YAML before any merge expansion runs. (Merge expansion can introduce key collisions legitimately; only explicit duplicates are an authoring error.)
    explicit_counts: dict = {}
    has_merge = False
    for key_node, _ in node.value:
        if key_node.tag == _MERGE_TAG:
            has_merge = True
            continue
        try:
            k = loader.construct_object(key_node, deep=False)
        except Exception:  # noqa: BLE001
            continue
        explicit_counts[k] = explicit_counts.get(k, 0) + 1
    for k, n in explicit_counts.items():
        if n > 1:
            raise ValueError(f'Duplicate key: "{k}"')

    if has_merge:
        _compose_include_merges(loader, node)
        loader.flatten_mapping(node)

    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        value = loader.construct_object(value_node, deep=deep)
        # Later entries override earlier ones: this matches standard YAML merge-key semantics where explicit values override merged ones.
        mapping[key] = value
    return mapping


def _compose_include_merges(loader: yaml.Loader, node: yaml.MappingNode) -> None:
    """Replace ``!include`` nodes sitting under a ``<<:`` merge key with the composed file.

    ``flatten_mapping`` works on the node tree and rejects anything that is not a mapping node, so an ``!include`` — a scalar node until its constructor runs — cannot be merged.
    Composing the referenced file into a node here, before the flatten, makes the two idioms compose without touching PyYAML's merge semantics: the spliced node is an ordinary mapping and precedence (explicit over merged, earlier merge over later) stays exactly as it was.
    """
    for entry, (key_node, value_node) in enumerate(node.value):
        if key_node.tag != _MERGE_TAG:
            continue
        if isinstance(value_node, yaml.SequenceNode):
            value_node.value = [_compose_included(loader, s) if s.tag == _INCLUDE_TAG else s for s in value_node.value]
        elif value_node.tag == _INCLUDE_TAG:
            node.value[entry] = (key_node, _compose_included(loader, value_node))


def _compose_included(loader: yaml.Loader, node: yaml.ScalarNode) -> yaml.Node:
    """The ``!include`` target composed to a node tree rather than constructed to a dict.

    Same file resolution and same anchor scoping as the ``!include`` constructor — the fragment is composed with its own loader class, so its anchors stay file-local. The file envelope (:data:`ENVELOPE_KEYS`) is dropped: it describes the fragment's file, not the object it is merged into, and would reach the parent class as an unknown slot.
    """
    base_dir = getattr(loader, "_tvbo_base_dir", Path.cwd())
    path = _include_path(loader.construct_scalar(node), base_dir)
    with open(path) as fh:
        composed = yaml.compose(fh, _make_loader_class(path.parent))
    if not isinstance(composed, yaml.MappingNode):
        raise yaml.constructor.ConstructorError(
            None,
            None,
            f"a merged !include must hold a mapping, but {path} holds {composed.id}",
            node.start_mark,
        )
    composed.value = [(k, v) for k, v in composed.value if not (isinstance(k, yaml.ScalarNode) and k.value in ENVELOPE_KEYS)]
    return composed


class IncludedMapping(dict):
    """A mapping spliced in by ``!include``, remembering the file it came from.

    A fragment's relative paths mean what they say *in the fragment*: a figure record naming ``code_modules`` and a captured page states them relative to the study it belongs to, and that must not change because a second spec includes it. A plain dict loses that the moment it is spliced, so the whole fragment silently re-resolves against whoever included it. This is a dict in every other respect, so nothing downstream needs to know it exists.
    """

    def __init__(self, mapping, origin: Path):
        super().__init__(mapping)
        self.include_origin = Path(origin)


for _dumper in (yaml.Dumper, yaml.SafeDumper):
    # A dict subclass has no representer of its own, and the load path dumps the parsed document back to YAML for LinkML: without this an included fragment is a load-time crash rather than a mapping.
    _dumper.add_representer(IncludedMapping, lambda dumper, data: dumper.represent_dict(data))


def include_origin(value: Any) -> Path | None:
    """The directory an ``!include``d value came from, or None for anything loaded in place."""
    return getattr(value, "include_origin", None)


def _include_path(rel: str, base_dir: Path) -> Path:
    """An ``!include`` target resolved against the including file's directory."""
    path = Path(rel)
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"!include target not found: {path}")
    return path


def _make_include_constructor(base_dir: Path):
    """Build a ``!include`` constructor anchored at ``base_dir``.

    The included document's own file envelope (:data:`ENVELOPE_KEYS`) is dropped, as it is for a merged include: the value is spliced into a parent slot, where those keys belong to no class. A file read for its own sake keeps them — see :func:`load_as_dict`.
    """

    def _include(loader: yaml.Loader, node: yaml.Node) -> Any:
        if isinstance(node, yaml.ScalarNode):
            rel = loader.construct_scalar(node)
        else:
            raise yaml.constructor.ConstructorError(None, None, "!include expects a scalar (a file path)", node.start_mark)
        path = _include_path(rel, base_dir)
        # Fresh loader instance for the included document so anchors are file-local (no name capture from or into the parent document).
        with open(path) as fh:
            value = strip_envelope(yaml.load(fh, _make_loader_class(path.parent)))
        return IncludedMapping(value, path.parent) if isinstance(value, dict) else value

    return _include


def _make_loader_class(base_dir: Path) -> type[DupCheckYamlLoader]:
    """Build a fresh loader subclass bound to ``base_dir``.

    A new class per base directory is the simplest way to thread the directory context through PyYAML's class-level constructor registry without leaking state across concurrent loads. The constructors are installed in ``__init__`` (after ``super().__init__``) so they override the instance-level registrations that :class:`DupCheckYamlLoader` performs in its own ``__init__``.
    """
    include_ctor = _make_include_constructor(base_dir)

    class _TVBOLoader(DupCheckYamlLoader):
        _tvbo_base_dir = base_dir

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # Re-register the mapping constructor with merge-key support, overriding the duplicate-only one DupCheckYamlLoader installed.
            self.add_constructor(
                yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                _flatten_map_constructor,
            )
            # Install the !include constructor for this base_dir.
            self.add_constructor(_INCLUDE_TAG, include_ctor)

    return _TVBOLoader


def _looks_like_path(source: Any) -> bool:
    """Heuristic: short string with no newline, treated as a path candidate.

    Avoids ``OSError: File name too long`` when callers pass full YAML content as a string.
    """
    if isinstance(source, os.PathLike):
        return True
    if not isinstance(source, str):
        return False
    if "\n" in source:
        return False
    if len(source) > 4096:
        return False
    try:
        return Path(source).exists()
    except OSError:
        return False


_EDGE_VAR_ALIASES = {"source_variable": "source_var", "target_variable": "target_var"}
"""Edge var-slot aliases, folded ONLY under an ``edges`` / ``edge_template`` key.

They must never join the global, context-free ``_SLOT_ALIASES``: ``target_variable`` is
also the *canonical* slot on stimulus events, so a context-free rename would silently
retarget every stimulus. The key-scoped application is what keeps the two apart.
"""


def resolve_edge_var_aliases(edges: Any) -> None:
    """Fold the ``source_variable`` / ``target_variable`` slot aliases onto the canonical ``source_var`` / ``target_var`` on inline edge dicts, in place.

    ``edges`` may be a single edge dict, a list of them, or ``None``; non-dict entries are left untouched.
    """
    if edges is None:
        return
    for edge in edges if isinstance(edges, (list, tuple)) else [edges]:
        if not isinstance(edge, dict):
            continue
        for alias, canonical in _EDGE_VAR_ALIASES.items():
            if alias not in edge:
                continue
            if canonical in edge:
                warnings.warn(
                    f"Edge has both '{alias}' and its canonical alias target '{canonical}'; ignoring '{alias}'.",
                    stacklevel=2,
                )
                edge.pop(alias)
            else:
                edge[canonical] = edge.pop(alias)


def _fold_edge_var_aliases(obj: Any) -> Any:
    """Recursively apply :func:`resolve_edge_var_aliases` to every ``edges`` / ``edge_template`` value, wherever the network sits in the document.

    Keying on the slot name rather than on the enclosing class keeps the fold scoped to edges while staying agnostic about the document root — the same alias works whether a ``Network``, a ``SimulationExperiment`` or a ``SimulationStudy`` is being loaded.
    """
    if isinstance(obj, dict):
        for key in ("edges", "edge_template"):
            resolve_edge_var_aliases(obj.get(key))
        for v in obj.values():
            _fold_edge_var_aliases(v)
    elif isinstance(obj, list):
        for x in obj:
            _fold_edge_var_aliases(x)
    return obj


def _lift_one_distribution(obj: dict) -> None:
    """Complete every terse ``*distribution`` mapping written directly on *obj*, in place.

    A ``Distribution`` carries its support under ``domain``; a bare ``lo``/``hi``/``step`` on any ``*distribution`` slot is lifted into ``domain``, and the distribution ``name`` is materialised as ``Uniform`` so the lifted form is a complete, valid ``{name: Uniform, domain: {lo, hi}}``. Any other keys (seed, axis, …) are preserved; a value that already states a ``domain`` is left untouched.

    One level, so the dialect can apply it to each object as it is constructed;
    :func:`_lift_distribution_shortcut` walks a whole document with it.
    """
    for key, value in list(obj.items()):
        if (
            isinstance(key, str)
            and key.endswith("distribution")
            and isinstance(value, dict)
            and "domain" not in value
            and any(b in value for b in ("lo", "hi", "step"))
        ):
            bounds = {b: value[b] for b in ("lo", "hi", "step") if b in value}
            lifted = {k: v for k, v in value.items() if k not in ("lo", "hi", "step")}
            lifted["domain"] = bounds
            lifted.setdefault("name", "Uniform")
            obj[key] = lifted


def _lift_distribution_shortcut(obj: Any) -> Any:
    """Apply :func:`_lift_one_distribution` at every depth of *obj*."""
    if isinstance(obj, dict):
        _lift_one_distribution(obj)
        for value in obj.values():
            _lift_distribution_shortcut(value)
    elif isinstance(obj, list):
        for item in obj:
            _lift_distribution_shortcut(item)
    return obj


def _fold_one_state_variable_domain(sv: dict) -> None:
    """Fold a single state variable's legacy domain slots in place.

    * ``range`` (a ``domain`` alias) → ``domain`` when no explicit ``domain`` is set.
    * ``boundaries`` (deprecated hard-clamp slot) → ``domain`` with ``enforce: clamp``;
      a co-existing descriptive ``domain`` is preserved as the sampling ``distribution`` (a terse ``{lo, hi}`` that the distribution-lift then completes) so a half-open
      clamp cannot drop a finite IC-sampling range.

    A ``Range`` object or an ``(lo, hi[, step])`` sequence is accepted as well as a mapping:
    the class carries no ``boundaries`` slot, so a value this cannot read would be popped and lost without a word.
    """
    if "range" in sv:
        if sv.get("domain") is None:
            sv["domain"] = sv.pop("range")
        else:
            warnings.warn(
                "State variable has both 'range' and its canonical alias 'domain'; ignoring 'range'.",
                stacklevel=2,
            )
            sv.pop("range")
    if sv.get("boundaries") is not None:
        bnd = _as_bounds_mapping(sv.pop("boundaries"))
        bnd.setdefault("enforce", "clamp")
        prev = sv.get("domain")
        if isinstance(prev, dict) and sv.get("distribution") is None:
            sv["distribution"] = {k: prev[k] for k in ("lo", "hi", "step") if k in prev}
        sv["domain"] = bnd


def _as_bounds_mapping(bounds: Any) -> dict:
    """*bounds* as a plain ``{lo, hi[, step, enforce]}`` mapping.

    Accepts the mapping the YAML path always hands over, the ``(lo, hi[, step])`` sequence and the ``Range`` object the Python construction path may, and raises on anything else rather than dropping a hard clamp silently. Only the bounds are read off an object — ``boundaries`` means clamp, so the caller supplies ``enforce``.
    """
    bound_keys = ("lo", "hi", "step")
    if isinstance(bounds, dict):
        return dict(bounds)
    if isinstance(bounds, (list, tuple)) and 2 <= len(bounds) <= 3:
        return dict(zip(bound_keys, bounds, strict=False))
    read = {k: getattr(bounds, k, None) for k in bound_keys}
    if read["lo"] is None and read["hi"] is None:
        raise ValueError(
            "State variable 'boundaries' must state lo/hi as a mapping, a (lo, hi[, step]) "
            f"sequence or a Range; got {bounds!r}."
        )
    return {k: v for k, v in read.items() if v is not None}


def _fold_state_variable_domains(obj: Any) -> Any:
    """Recursively fold legacy ``boundaries``/``range`` on state variables into ``domain`` (see :func:`_fold_one_state_variable_domain`), at any nesting depth.

    The schema declares ``range``/``boundaries`` as ``domain`` aliases, but LinkML aliases are metadata only (the loader keys on the canonical slot), so — like the slot-alias and distribution-shortcut folds — this is applied before LinkML sees the data. Runs on both load paths so ``yaml_loader.load``/``loads`` matches ``Dynamics.from_file`` for legacy files.
    """
    if isinstance(obj, dict):
        svs = obj.get("state_variables")
        sv_iter = svs.values() if isinstance(svs, dict) else svs if isinstance(svs, list) else []
        for sv in sv_iter:
            if isinstance(sv, dict):
                _fold_one_state_variable_domain(sv)
        for v in obj.values():
            _fold_state_variable_domains(v)
    elif isinstance(obj, list):
        for x in obj:
            _fold_state_variable_domains(x)
    return obj


_PATH_KEYS = ("bids_dir", "mesh_file", "data_file", "code_source", "path", "file")


def _anchor_paths(obj: Any, origin: Path) -> Any:
    """Rewrite a spliced fragment's relative file references to absolute, against the directory it came from.

    A document moved out of its own directory takes its relative paths with it, and they then resolve against whoever spliced it — which is how an experiment lifted from the database comes to look for its connectome under the including study. Anchoring is guarded by existence: a value is rewritten only when it actually names something at the origin, so a path-shaped string that is not a path is left alone.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in _PATH_KEYS and isinstance(value, str) and not Path(value).is_absolute():
                candidate = (origin / value).resolve()
                if candidate.exists():
                    obj[key] = str(candidate)
                    continue
            _anchor_paths(value, origin)
    elif isinstance(obj, list):
        for item in obj:
            _anchor_paths(item, origin)
    return obj


def _expand_curated_experiments(data: Any) -> Any:
    """Replace a study's curated-experiment IRI references with the documents they name.

    ``experiments: [tvbo:experiment/<name>]`` loads that experiment out of the database in place, with its relative paths anchored at the database directory it came from. A path reference cannot do this job: ``!include`` splices a document without rebasing the paths inside it, so an experiment included out of the database resolves its own ``network.bids_dir`` against the including study and loads the wrong root. The IRI carries no path of its own, so the database resolves it against the database.

    An entry may also be a mapping carrying ``iri:`` plus its own keys, which are merged over the curated document: a variant of a curated experiment — the same model run without its settle, or with one parameter moved — is then declared as the difference from it rather than as a second copy that can drift. Merging is recursive over mappings, and a null drops the curated key outright (``optimizations: null`` for a variant that only simulates).

    Only a top-level ``experiments`` list is expanded. A reference that stays a reference (``model: JansenRit1995``) is left alone, so this cannot turn a pointer into an inlined copy.
    """
    if not isinstance(data, dict):
        return data
    entries = data.get("experiments")
    if not isinstance(entries, list):
        return data

    def _ref(entry):
        """The curated IRI an entry names, or None when it is an ordinary inline experiment."""
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict) and isinstance(entry.get("iri"), str):
            return entry["iri"]
        return None

    from tvbo.data.registry import iri_target

    if not any(iri_target(_ref(e) or "") for e in entries):
        return data

    expanded = []
    for entry in entries:
        iri = _ref(entry)
        if iri is None or iri_target(iri) is None:
            expanded.append(entry)
            continue
        curated = _load_curated(iri)
        if isinstance(entry, dict):
            curated = _merge_curated(curated, {k: v for k, v in entry.items() if k != "iri"})
        expanded.append(curated)
    data["experiments"] = expanded
    return data


def _merge_curated(base: dict, over: dict) -> dict:
    """*over* laid on *base*, recursing into mappings; a null drops the key.

    Shared by every curated-reference expansion so a variant declared as the difference from a curated record means the same thing wherever it is written.
    """
    merged = dict(base)
    for k, v in over.items():
        if v is None:
            merged.pop(k, None)
        elif isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _merge_curated(merged[k], v)
        else:
            merged[k] = v
    return merged


def _load_curated(iri: str) -> dict:
    """The curated document an IRI names, envelope dropped and its relative paths anchored at the database directory it came from.

    Shared by every curated-reference expansion. Anchoring is the half that is easy to forget and impossible to notice: a document spliced out of the database takes its relative ``data_file``/``path`` references with it, and they then resolve against whoever spliced it.
    """
    from tvbo.data.registry import resolve_iri

    source = resolve_iri(iri)
    return _anchor_paths(strip_envelope(load_as_dict(str(source))), source.parent)


def _expand_pipeline_references(data: Any) -> Any:
    """Splice the steps of the curated observation a ``pipeline`` entry names by ``iri``.

    A pipeline step that carries ``iri: tvbo:observation/<name>`` is that observation's pipeline, written once and reused: the BOLD variants differ only in their hemodynamic kernel, and repeating the interim average, the convolution and the output stride in each is how five copies of one pipeline come to drift apart. The entry's own keys are merged over the referenced step, so a reuse states only what differs from it, and a null drops a curated key outright.

    Splicing rather than gap-filling because the target is a list: an observation whose pipeline is several steps contributes all of them, in order. Overriding keys is therefore only meaningful against a single step, and naming a multi-step observation while also overriding raises rather than guessing which step the override was meant for.

    Only the steps travel. A referenced observation's own ``period``, ``parameters`` and ``data_source`` stay behind, so the spliced steps resolve those against the host — which is what a monitor reused on another grid wants, and which is also why a reference to an observation whose steps depend on its own declarations does not carry them along.

    Applies to every ``pipeline`` list in the document, at any depth, so an observation curated in the database and one written inline in a recipe expand alike. The spliced steps are copies: handing a caller the objects the curated document was parsed into lets a later mutation reach every other recipe that named it.
    """
    import copy

    from tvbo.data.registry import iri_target

    def _intact(walked, original) -> bool:
        """Whether the walk handed back the very objects it was given, a rebuilt list of unchanged members included."""
        if walked is original:
            return True
        if isinstance(walked, list) and isinstance(original, list) and len(walked) == len(original):
            return all(a is b for a, b in zip(walked, original, strict=True))
        return False

    def _walk(node):
        """*node* with every ``pipeline`` list under it expanded, or *node* itself where nothing under it names a curated step.

        Returned unchanged rather than rebuilt, because this runs on every document the loader reads and almost none of them carry a reference: copying each one a second time on top of the deepcopy :func:`_normalize_loaded` already took buys nothing.
        """
        if isinstance(node, list):
            walked = [_walk(item) for item in node]
            return node if _intact(walked, node) else walked
        if not isinstance(node, dict):
            return node
        out = {}
        for key, value in node.items():
            if key == "pipeline" and isinstance(value, list):
                out[key] = [step for entry in value for step in _steps_for(entry)]
            else:
                out[key] = _walk(value)
        return node if all(_intact(out[key], value) for key, value in node.items()) else out

    def _steps_for(entry):
        """The steps one pipeline entry contributes: itself, or the curated pipeline it names by ``iri``."""
        target = iri_target(entry.get("iri", "")) if isinstance(entry, dict) else None
        if target is None or target[0] != "Observation":
            return [_walk(entry)]
        referenced = _load_curated(entry["iri"]).get("pipeline") or []
        overrides = {k: v for k, v in entry.items() if k != "iri"}
        if not referenced:
            raise ValueError(
                f"pipeline step names {entry['iri']!r}, which declares no pipeline of its own — there is "
                "nothing to splice. Reference an observation that carries the steps you meant to reuse."
            )
        if len(referenced) != 1 and overrides:
            raise ValueError(
                f"pipeline step names {entry['iri']!r}, whose pipeline has {len(referenced)} steps, "
                f"and also overrides {sorted(overrides)} — an override has no single step to apply to. "
                "Reference it without overrides, or name a single-step observation."
            )
        return [_merge_curated(step, overrides) if overrides else copy.deepcopy(step) for step in referenced]

    return _walk(data)


def _normalize_loaded(data: Any) -> Any:
    """Apply the dict-level TVBO conveniences shared by every load path.

    Slot aliases are folded at construction by the generated datamodel (see ``hatch_build._alias_support``), so this handles only what a class cannot: the edge-template ``source_variable``/``target_variable`` snapshot, the legacy state-variable ``boundaries``/``range`` into ``domain`` (+ ``enforce: clamp`` for boundaries), lifts the terse ``distribution: {lo, hi}`` shortcut into ``distribution: {domain: {lo, hi}}``, loads a study's curated-experiment IRI references from the database, and splices the curated observation a pipeline step names by ``iri``. Both the string path (``load``/``loads`` → LinkML) and the dict path (``load_as_dict`` → ``Dynamics.from_file``/``from_db``) route through here so the two cannot diverge. Order matters: the boundaries fold can create a terse ``distribution`` that the following lift then completes, and the experiments are expanded first so the folds reach inside them.
    """
    import copy

    # Some passes below mutate in place; never reach through to the caller's object.
    data = copy.deepcopy(data)
    data = _expand_curated_experiments(data)
    data = _expand_pipeline_references(data)
    data = _fold_edge_var_aliases(data)
    data = _fold_state_variable_domains(data)
    data = _lift_distribution_shortcut(data)
    return data


def strip_envelope(data: Any) -> Any:
    """Drop :data:`ENVELOPE_KEYS` from a document root bound for a class constructor.

    A file may name its own class and schema version (``tvbo_class: tvbo:SimulationStudy``) so tooling can dispatch on it without being told. Those keys are slots of no class, so they must not survive into the target's ``__init__``.
    """
    if isinstance(data, dict):
        return {k: v for k, v in data.items() if k not in ENVELOPE_KEYS}
    if isinstance(data, list):
        return [strip_envelope(d) for d in data]
    return data


def _preprocess(source: Any, base_dir: Path) -> str:
    """Parse ``source`` with the TVBO loader and re-serialise to plain YAML.

    The LinkML loader expects either a path it can open or a string it can hand to its own ``DupCheckYamlLoader``. To layer our extensions on top, we first parse with our loader, then re-serialise the fully-expanded data structure (no anchors, no includes, no merge keys) and let LinkML consume that.
    """
    LoaderCls = _make_loader_class(base_dir)
    if _looks_like_path(source):
        with open(source) as fh:
            data = yaml.load(fh, LoaderCls)
    elif isinstance(source, str):
        data = yaml.load(io.StringIO(source), LoaderCls)
    elif hasattr(source, "read"):
        data = yaml.load(source, LoaderCls)
    else:
        data = source
    # Fold slot aliases + lift the terse distribution shortcut (shared with the dict path so the two cannot diverge), then drop the file envelope, which only load_as_dict's dispatching callers need.
    data = _normalize_loaded(data)
    data = strip_envelope(data)
    # Re-serialise using safe_dump so the LinkML loader sees pure data with no remaining anchors/merge keys/!include directives.
    return yaml.safe_dump(data, sort_keys=False)


def load(source: Any, target_class: type, **kwargs: Any) -> Any:
    """Drop-in replacement for ``linkml_runtime.loaders.yaml_loader.load``.

    Accepts the same arguments as the LinkML loader. Expands TVBO YAML extensions (``<<:`` merge keys, ``!include``) before delegating to LinkML's constructor-class machinery. Relative ``!include`` paths are resolved against the directory of ``source`` when ``source`` is a path; otherwise against the current working directory.
    """
    base_dir = _base_dir_for(source)
    expanded = _preprocess(source, base_dir)
    return _linkml_yaml_loader.loads(expanded, target_class, **kwargs)


def loads(source: str, target_class: type, **kwargs: Any) -> Any:
    """Drop-in replacement for ``linkml_runtime.loaders.yaml_loader.loads``."""
    base_dir = Path(kwargs.pop("base_dir", Path.cwd())).resolve()
    expanded = _preprocess(source, base_dir)
    return _linkml_yaml_loader.loads(expanded, target_class, **kwargs)


def load_as_dict(source: Any, **kwargs: Any) -> dict:
    """Drop-in replacement for ``yaml_loader.load_as_dict``.

    Returns a plain Python ``dict`` (or ``list`` of dicts) after applying the TVBO YAML extensions. Useful for callers that need to inspect or mutate the parsed structure before handing it to LinkML.
    """
    base_dir = _base_dir_for(source)
    LoaderCls = _make_loader_class(base_dir)
    if _looks_like_path(source):
        with open(source) as fh:
            data = yaml.load(fh, LoaderCls)
    elif isinstance(source, str):
        data = yaml.load(io.StringIO(source), LoaderCls)
    elif hasattr(source, "read"):
        data = yaml.load(source, LoaderCls)
    else:
        data = source
    # Same normalisation as the string path, so the dict path used by from_file/from_db cannot diverge from the LinkML one; the envelope survives, for the caller to dispatch on.
    return _normalize_loaded(data)


def _base_dir_for(source: Any) -> Path:
    if _looks_like_path(source):
        return Path(str(source)).resolve().parent
    if hasattr(source, "name"):
        try:
            return Path(source.name).resolve().parent
        except (TypeError, ValueError):
            pass
    return Path.cwd()
