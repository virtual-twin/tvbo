"""YAML loader wrapper used by every ``Network.from_file`` /
``SimulationExperiment.from_file`` / ``SimulationStudy.from_file`` entry
point in TVBO.

Extends :class:`linkml_runtime.utils.yamlutils.DupCheckYamlLoader` (the
default LinkML loader, which already disallows duplicate keys) with two
generally-useful YAML idioms:

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
  the contents of another YAML file. Paths are resolved relative to the
  directory of the file containing the directive; absolute paths are
  accepted as-is. Anchors in the included document are scoped to that
  document only (each include uses a fresh loader instance), so
  fragments are readable in isolation.

  .. code-block:: yaml

      experiments:
        - !include _experiments/exp1.yaml
        - !include _experiments/exp2.yaml

Both idioms are pure data-format machinery; they don't introduce any
TVBO-specific semantics into user YAMLs. The wrapper is transparent —
any LinkML class can still load through ``yaml_loader.load`` and get
back the same datamodel instance it would have produced before.
"""

from __future__ import annotations

import io
import os
import warnings
from pathlib import Path
from typing import Any, Type

import yaml
from linkml_runtime.loaders import yaml_loader as _linkml_yaml_loader
from linkml_runtime.utils.yamlutils import DupCheckYamlLoader


_MERGE_TAG = "tag:yaml.org,2002:merge"
_INCLUDE_TAG = "!include"


def _flatten_map_constructor(loader: yaml.Loader, node: yaml.MappingNode, deep: bool = False) -> dict:
    """``DupCheckYamlLoader`` map constructor augmented with merge-key support.

    Standard YAML merge semantics: an explicit entry overrides any value
    pulled in by a ``<<:`` merge. We preserve the original LinkML
    duplicate-key safety check for *explicit* duplicates (the same key
    written twice by the author), but suppress it for keys whose
    collision came from a merge expansion — those are silently
    overridden by the explicit entry.
    """
    if not isinstance(node, yaml.MappingNode):
        from yaml.constructor import ConstructorError

        raise ConstructorError(None, None, f"expected a mapping node, but found {node.id}", node.start_mark)
    # Catch duplicate explicit keys in the user's YAML before any merge
    # expansion runs. (Merge expansion can introduce key collisions
    # legitimately; only explicit duplicates are an authoring error.)
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
        loader.flatten_mapping(node)

    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        value = loader.construct_object(value_node, deep=deep)
        # Later entries override earlier ones: this matches standard YAML
        # merge-key semantics where explicit values override merged ones.
        mapping[key] = value
    return mapping


def _make_include_constructor(base_dir: Path):
    """Build a ``!include`` constructor anchored at ``base_dir``."""

    def _include(loader: yaml.Loader, node: yaml.Node) -> Any:
        if isinstance(node, yaml.ScalarNode):
            rel = loader.construct_scalar(node)
        else:
            raise yaml.constructor.ConstructorError(
                None, None, "!include expects a scalar (a file path)", node.start_mark
            )
        path = Path(rel)
        if not path.is_absolute():
            path = base_dir / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"!include target not found: {path}")
        # Fresh loader instance for the included document so anchors are
        # file-local (no name capture from or into the parent document).
        with open(path, "r") as fh:
            return yaml.load(fh, _make_loader_class(path.parent))

    return _include


def _make_loader_class(base_dir: Path) -> Type[DupCheckYamlLoader]:
    """Build a fresh loader subclass bound to ``base_dir``.

    A new class per base directory is the simplest way to thread the
    directory context through PyYAML's class-level constructor registry
    without leaking state across concurrent loads. The constructors are
    installed in ``__init__`` (after ``super().__init__``) so they
    override the instance-level registrations that
    :class:`DupCheckYamlLoader` performs in its own ``__init__``.
    """

    include_ctor = _make_include_constructor(base_dir)

    class _TVBOLoader(DupCheckYamlLoader):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            # Re-register the mapping constructor with merge-key support,
            # overriding the duplicate-only one DupCheckYamlLoader installed.
            self.add_constructor(
                yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                _flatten_map_constructor,
            )
            # Install the !include constructor for this base_dir.
            self.add_constructor(_INCLUDE_TAG, include_ctor)

    return _TVBOLoader


def _looks_like_path(source: Any) -> bool:
    """Heuristic: short string with no newline, treated as a path candidate.

    Avoids ``OSError: File name too long`` when callers pass full YAML
    content as a string.
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


# Convenience YAML keys folded to their canonical (schema) slot names before
# LinkML parsing. LinkML ``aliases:`` are metadata only — the runtime loader
# keys on the canonical slot name — so singular/alternate spellings declared as
# schema aliases are resolved here. Keep in sync with ``aliases:`` in the schema.
_SLOT_ALIASES = {
    "optimization": "optimizations",  # singular convenience for the (multivalued) optimizations slot
}


def _fold_slot_aliases(obj: Any) -> Any:
    """Recursively rename alias keys (e.g. ``optimization``) to their canonical
    slot names (``optimizations``) so the LinkML loader accepts them."""
    if isinstance(obj, dict):
        folded: dict = {}
        for k, v in obj.items():
            canonical = _SLOT_ALIASES.get(k, k) if isinstance(k, str) else k
            if canonical != k and canonical in obj:
                # Both the alias (e.g. ``optimization``) and its canonical
                # (``optimizations``) are present in the same mapping. Keep the
                # canonical value and drop the alias rather than silently
                # overwriting one with the other.
                warnings.warn(
                    f"YAML mapping has both '{k}' and its canonical alias "
                    f"target '{canonical}'; ignoring '{k}'.",
                    stacklevel=2,
                )
                continue
            folded[canonical] = _fold_slot_aliases(v)
        return folded
    if isinstance(obj, list):
        return [_fold_slot_aliases(x) for x in obj]
    return obj


def _lift_distribution_shortcut(obj: Any) -> Any:
    """Allow a terse ``distribution: {lo, hi}`` as a shortcut for a Uniform.

    A ``Distribution`` carries its support under ``domain``; a bare
    ``lo``/``hi``/``step`` on any ``*distribution`` slot is lifted into ``domain``
    here, before the LinkML loader sees it, and the distribution ``name`` is
    materialised as ``Uniform`` (so the lifted form is a complete, valid
    ``{name: Uniform, domain: {lo, hi}}``). Any other keys (seed, axis, …) are
    preserved; if an explicit ``domain`` is already present the value is left
    untouched.
    """
    if isinstance(obj, dict):
        out: dict = {}
        for k, v in obj.items():
            if (
                isinstance(k, str)
                and k.endswith("distribution")
                and isinstance(v, dict)
                and "domain" not in v
                and any(b in v for b in ("lo", "hi", "step"))
            ):
                bounds = {b: v[b] for b in ("lo", "hi", "step") if b in v}
                rest = {kk: vv for kk, vv in v.items() if kk not in ("lo", "hi", "step")}
                v = {**rest, "domain": bounds}
                v.setdefault("name", "Uniform")
            out[k] = _lift_distribution_shortcut(v)
        return out
    if isinstance(obj, list):
        return [_lift_distribution_shortcut(x) for x in obj]
    return obj


def _normalize_loaded(data: Any) -> Any:
    """Apply the dict-level TVBO conveniences shared by every load path.

    Folds slot aliases to their canonical names and lifts the terse
    ``distribution: {lo, hi}`` shortcut into ``distribution: {domain: {lo, hi}}``.
    Both the string path (``load``/``loads`` → LinkML) and the dict path
    (``load_as_dict`` → ``Dynamics.from_file``/``from_db``) route through here so
    the two cannot diverge.
    """
    data = _fold_slot_aliases(data)
    data = _lift_distribution_shortcut(data)
    return data


def _preprocess(source: Any, base_dir: Path) -> str:
    """Parse ``source`` with the TVBO loader and re-serialise to plain YAML.

    The LinkML loader expects either a path it can open or a string it
    can hand to its own ``DupCheckYamlLoader``. To layer our extensions
    on top, we first parse with our loader, then re-serialise the
    fully-expanded data structure (no anchors, no includes, no merge
    keys) and let LinkML consume that.
    """
    LoaderCls = _make_loader_class(base_dir)
    if _looks_like_path(source):
        with open(source, "r") as fh:
            data = yaml.load(fh, LoaderCls)
    elif isinstance(source, str):
        data = yaml.load(io.StringIO(source), LoaderCls)
    elif hasattr(source, "read"):
        data = yaml.load(source, LoaderCls)
    else:
        data = source
    # Fold slot aliases + lift the terse distribution shortcut (shared with the
    # dict path so the two cannot diverge).
    data = _normalize_loaded(data)
    # Re-serialise using safe_dump so the LinkML loader sees pure data
    # with no remaining anchors/merge keys/!include directives.
    return yaml.safe_dump(data, sort_keys=False)


def load(source: Any, target_class: Type, **kwargs: Any) -> Any:
    """Drop-in replacement for ``linkml_runtime.loaders.yaml_loader.load``.

    Accepts the same arguments as the LinkML loader. Expands TVBO YAML
    extensions (``<<:`` merge keys, ``!include``) before delegating to
    LinkML's constructor-class machinery. Relative ``!include`` paths
    are resolved against the directory of ``source`` when ``source`` is
    a path; otherwise against the current working directory.
    """
    base_dir = _base_dir_for(source)
    expanded = _preprocess(source, base_dir)
    return _linkml_yaml_loader.loads(expanded, target_class, **kwargs)


def loads(source: str, target_class: Type, **kwargs: Any) -> Any:
    """Drop-in replacement for ``linkml_runtime.loaders.yaml_loader.loads``."""
    base_dir = Path(kwargs.pop("base_dir", Path.cwd())).resolve()
    expanded = _preprocess(source, base_dir)
    return _linkml_yaml_loader.loads(expanded, target_class, **kwargs)


def load_as_dict(source: Any, **kwargs: Any) -> dict:
    """Drop-in replacement for ``yaml_loader.load_as_dict``.

    Returns a plain Python ``dict`` (or ``list`` of dicts) after applying
    the TVBO YAML extensions. Useful for callers that need to inspect or
    mutate the parsed structure before handing it to LinkML.
    """
    base_dir = _base_dir_for(source)
    LoaderCls = _make_loader_class(base_dir)
    if _looks_like_path(source):
        with open(source, "r") as fh:
            data = yaml.load(fh, LoaderCls)
    elif isinstance(source, str):
        data = yaml.load(io.StringIO(source), LoaderCls)
    elif hasattr(source, "read"):
        data = yaml.load(source, LoaderCls)
    else:
        data = source
    # Route through the same normalisation as the string path (fold slot aliases
    # + lift the terse `distribution: {lo, hi}` shortcut) so the dict path used by
    # from_file/from_db cannot diverge from the LinkML string path.
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
