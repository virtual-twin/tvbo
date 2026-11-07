"""Export-format registry.

Single source of truth for *all* SimulationExperiment export backends:
serialisation formats (YAML, openMINDS), reports (markdown, PDF), and code generators (TVB, JAX, tvboptim, Julia, NeuroML/LEMS, …).

A backend self-registers an :class:`ExportFormat` describing:

* canonical key + aliases  (resolution)
* human label, file extension, MIME type   (UI / API)
* a renderer callable                       (dispatch)
* optional flags                            (``supports_with_data``, …)

Adding a new backend = importing this module and calling :func:`register` once. Dispatch (`SimulationExperiment.render`), discovery (`/api/v1/experiments/formats`, OntologyAPI), and the extension/UI dropdown all light up automatically.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

# Renderer signature: (experiment, **kwargs) -> str
Renderer = Callable[..., str]
# Importer signature: (path) -> SimulationExperiment | SimulationStudy | Network | Dynamics | ...
Importer = Callable[..., Any]


class StartKind(StrEnum):
    """What a backend must do to honour a `tvbo.adapters.base.ParameterStart`: draw a dynamics parameter per node, draw a coupling parameter per node or per connection, evaluate a parameter's equation once, or draw a start again for each trial of an `n_trials` ensemble. ``tuple(StartKind)`` is every kind."""

    NODE = "node"
    COUPLING = "coupling"
    EQUATION = "equation"
    TRIAL = "trial"


@dataclass(frozen=True)
class ExportFormat:
    """Descriptor for an export backend."""

    key: str
    label: str
    extension: str
    media_type: str
    renderer: Renderer
    aliases: tuple[str, ...] = ()
    supports_with_data: bool = False
    description: str = ""
    importer: Importer | None = None
    extensions: tuple[str, ...] = ()  # extra accepted file suffixes (incl. leading dot)
    # Output language for `tvbo.codegen.style`; empty means "return verbatim".
    language: str = ""
    # The `StartKind`s this backend honours, every other one refused by name before rendering; None for a format that integrates nothing and writes the declaration out.
    parameter_starts: tuple[str, ...] | None = ()

    def to_public_dict(self) -> dict[str, Any]:
        """Serialisable view (without the renderer callable)."""
        d = asdict(self)
        d.pop("renderer", None)
        d["aliases"] = list(self.aliases)
        d["format"] = d.pop("key")  # back-compat with previous _EXPORT_FORMATS shape
        return d


_REGISTRY: dict[str, ExportFormat] = {}


def register(fmt: ExportFormat, *, overwrite: bool = False) -> ExportFormat:
    """Register *fmt* under its canonical key and all aliases.

    Raises ``ValueError`` if a key is already taken (unless ``overwrite``).
    """
    keys = (fmt.key, *fmt.aliases)
    if not overwrite:
        for k in keys:
            if k in _REGISTRY:
                raise ValueError(f"Export format key '{k}' is already registered (by '{_REGISTRY[k].key}').")
    for k in keys:
        _REGISTRY[k.lower()] = fmt
    return fmt


def resolve(key: str) -> ExportFormat:
    """Look up an :class:`ExportFormat` by canonical key or alias."""
    fmt = _REGISTRY.get(key.lower())
    if fmt is None:
        valid = sorted({f.key for f in _REGISTRY.values()})
        raise ValueError(f"Unknown export format '{key}'. Valid: {', '.join(valid)}")
    return fmt


def list_formats() -> list[ExportFormat]:
    """Return all registered formats (deduplicated, ordered by canonical key)."""
    seen: dict[str, ExportFormat] = {}
    for fmt in _REGISTRY.values():
        seen.setdefault(fmt.key, fmt)
    return list(seen.values())


def list_format_dicts() -> list[dict[str, Any]]:
    """Public dict view of all formats — for API/UI dropdowns."""
    return [f.to_public_dict() for f in list_formats()]


def has(key: str) -> bool:
    """Report whether a format is registered under the given key or alias.

    Args:
        key: Canonical key or alias to check (matched case-insensitively).

    Returns:
        `True` if a format is registered under `key`, otherwise `False`.
    """
    return key.lower() in _REGISTRY


def keys() -> Iterable[str]:
    """Return a view of every registered key and alias.

    Returns:
        A view over all lookup keys, including canonical keys and aliases
        (all lowercased). The same format may appear under several keys.
    """
    return _REGISTRY.keys()


def refuse_unlowered_parameter_starts(experiment, key: str) -> None:
    """Raise where the backend *key* would integrate a parameter start it does not lower at one constant: the one check `render` and the run path share.

    A key no format registers lowers nothing, so every start is refused; a format whose ``parameter_starts`` is None integrates nothing and is not checked.
    """
    fmt = _REGISTRY.get(key.lower())
    lowered = () if fmt is None else fmt.parameter_starts
    if lowered is not None:
        from tvbo.adapters.base import refuse_parameter_starts

        refuse_parameter_starts(experiment, key if fmt is None else fmt.key, lowered)


def render(experiment, fmt_key: str, **kwargs) -> str:
    """Resolve *fmt_key*, invoke its renderer, prune its dead imports, and format it.

    All three happen here rather than in each renderer so that every backend — including the ones that render through an adapter and never touch the template helpers — is held to the same house style. Pruning precedes formatting because it edits statements and black only edits layout. See :mod:`tvbo.codegen.prune` and :mod:`tvbo.codegen.style`.
    """
    fmt = resolve(fmt_key)
    refuse_unlowered_parameter_starts(experiment, fmt.key)
    rendered = fmt.renderer(experiment, **kwargs)
    if not fmt.language:
        return rendered
    if fmt.language == "python":
        from tvbo.codegen.prune import prune

        rendered = prune(rendered)
    from tvbo.codegen.style import format_source

    return format_source(rendered, fmt.language)


def _all_extensions(fmt: ExportFormat) -> tuple[str, ...]:
    """All file suffixes claimed by *fmt* (canonical + extras), lowercased."""
    exts = (fmt.extension, *fmt.extensions)
    return tuple(e.lower() for e in exts if e)


def resolve_by_extension(suffix: str) -> ExportFormat:
    """Look up a format by file suffix (e.g. ``'.yaml'``, ``'.nml'``).

    Raises ``ValueError`` if no format claims the suffix or if multiple formats claim it ambiguously.
    """
    suffix = suffix.lower()
    if not suffix.startswith("."):
        suffix = "." + suffix
    matches = [f for f in list_formats() if suffix in _all_extensions(f)]
    if not matches:
        raise ValueError(f"No export format registered for suffix {suffix!r}.")
    if len(matches) > 1:
        keys_ = sorted({f.key for f in matches})
        raise ValueError(f"Suffix {suffix!r} is ambiguous — claimed by: {', '.join(keys_)}. Pass an explicit format.")
    return matches[0]


def load(fmt_key: str, path: str | Path, **kwargs) -> Any:
    """Resolve *fmt_key* and invoke its importer on *path*.

    Raises ``ValueError`` if the format has no importer registered.
    """
    fmt = resolve(fmt_key)
    if fmt.importer is None:
        raise ValueError(f"Export format {fmt.key!r} has no importer registered.")
    return fmt.importer(path, **kwargs)
