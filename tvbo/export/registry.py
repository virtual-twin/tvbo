"""Export-format registry.

Single source of truth for *all* SimulationExperiment export backends:
serialisation formats (YAML, openMINDS), reports (markdown, PDF), and code
generators (TVB, JAX, tvboptim, Julia, NeuroML/LEMS, …).

A backend self-registers an :class:`ExportFormat` describing:

* canonical key + aliases  (resolution)
* human label, file extension, MIME type   (UI / API)
* a renderer callable                       (dispatch)
* optional flags                            (``supports_with_data``, …)

Adding a new backend = importing this module and calling
:func:`register` once. Dispatch (`SimulationExperiment.render`),
discovery (`/api/v1/experiments/formats`, OntologyAPI), and the
extension/UI dropdown all light up automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable

# Renderer signature: (experiment, **kwargs) -> str
Renderer = Callable[..., str]


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
                raise ValueError(
                    f"Export format key '{k}' is already registered "
                    f"(by '{_REGISTRY[k].key}')."
                )
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
    return key.lower() in _REGISTRY


def keys() -> Iterable[str]:
    return _REGISTRY.keys()


def render(experiment, fmt_key: str, **kwargs) -> str:
    """Convenience: resolve *fmt_key* and invoke its renderer."""
    return resolve(fmt_key).renderer(experiment, **kwargs)
