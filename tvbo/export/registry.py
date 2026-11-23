"""Export-format registry.

Single source of truth for *all* SimulationExperiment export backends: serialisation formats (YAML, openMINDS), reports (markdown, PDF), and code generators (TVB, JAX, tvboptim, Julia, NeuroML/LEMS, …).

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
    """What a backend must do to honour a `tvbo.adapters.base.ParameterStart`: draw a dynamics parameter per node, draw a coupling parameter per node or per connection, evaluate a parameter's equation once, or draw a start again for each trial of an `n_trials` ensemble; or, for a distribution declared with ``axis: time``, draw again at every step a dynamics parameter (``TIME``), a coupling parameter (``COUPLING_TIME``), the one such parameter of a stimulus whose signal is an equation of time (``STIMULUS_TIME``), or any other event's (``EVENT_TIME``: a state-triggered, recorded or sourced event's, or one of several on one event). ``tuple(StartKind)`` is every kind."""

    NODE = "node"
    COUPLING = "coupling"
    EQUATION = "equation"
    TRIAL = "trial"
    TIME = "time"
    COUPLING_TIME = "coupling_time"
    STIMULUS_TIME = "stimulus_time"
    EVENT_TIME = "event_time"


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
    # How this backend treats an event the experiment declares (`EVENT_HANDLING`).
    events: str = "dropped"
    # Whether this backend applies the affect of an event a model declares, such as a spike's reset (`refuse_unapplied_model_events`).
    model_events: bool = False

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
        A view over all lookup keys, including canonical keys and aliases (all lowercased). The same format may appear under several keys.
    """
    return _REGISTRY.keys()


def refuse_unlowered_parameter_starts(experiment, key: str) -> None:
    """Raise where the backend *key* would integrate a parameter start it does not lower at one constant, or run every trial on one realisation of a network built per trial: the one check `render` and the run path share.

    A key no format registers lowers nothing, so every start and a network built per trial are refused; a format whose ``parameter_starts`` is None integrates nothing and is not checked.
    """
    fmt = _REGISTRY.get(key.lower())
    lowered = () if fmt is None else fmt.parameter_starts
    if lowered is not None:
        from tvbo.adapters.base import refuse_parameter_starts, refuse_trial_network

        backend = key if fmt is None else fmt.key
        refuse_parameter_starts(experiment, backend, lowered)
        refuse_trial_network(experiment, backend, lowered)


#: What a backend does with an event the experiment declares, each value refused by name but the first: "integrated", "dropped" for a backend with no stimulus path, "continuation" for one that continues an equilibrium.
EVENT_HANDLING = {
    "dropped": "has no stimulus path, so it would run the unstimulated model",
    "continuation": "continues an equilibrium, where a stimulus has no place",
}
#: `ExportFormat.events` for the run-only backends, which no format registers: python and pyrates evaluate a stimulus, gillespie none.
RUN_ONLY_EVENTS = {"python": "integrated", "pyrates": "integrated", "gillespie": "dropped", "ssa": "dropped"}


def refuse_undeclared_events(experiment, key: str) -> None:
    """Raise where the backend *key* would drop an event *experiment* declares, in whatever form it is declared, named under its `events` or read by an equation: the one check `render` and the run path share.

    A run-only key reads `RUN_ONLY_EVENTS`, any other key no format registers integrates no event, and a format whose ``parameter_starts`` is None integrates nothing and is not checked. A model rendered on its own carries only its own events, a spike's reset among them, which are part of the model and not checked here.
    """
    from tvbo.utils import keyed_items

    fmt = _REGISTRY.get(key.lower())
    if (fmt is not None and fmt.parameter_starts is None) or not hasattr(experiment, "dynamics"):
        return
    handling = RUN_ONLY_EVENTS.get(key.lower(), "dropped") if fmt is None else fmt.events
    declared = [str(name) for name, _ in keyed_items(getattr(experiment, "events", None), "events")]
    if declared and handling != "integrated":
        raise NotImplementedError(
            f"the {key if fmt is None else fmt.key} backend {EVENT_HANDLING[handling]}, and the experiment declares event {', '.join(map(repr, declared))}. "
            "Run the experiment on a backend that integrates a stimulus (tvboptim, tvb, jax), or remove the event."
        )


def refuse_unapplied_model_events(experiment, key: str) -> None:
    """Raise where the backend *key* has no path for the affect of an event a model of *experiment* declares, such as a spike's reset, and would integrate the model without it: the one check `render` and the run path share.

    The refusal reads the declaration alone, each event that carries an affect and what it sets, so a model is refused by what it declares under whatever name. An event without an affect, a condition that only marks a spike or the name of an experiment event an equation reads, changes no trajectory and passes. A key no format registers applies no affect, and a format whose ``parameter_starts`` is None integrates nothing and is not checked.
    """
    from tvbo.adapters.base import dynamics_library
    from tvbo.utils import keyed_items

    fmt = _REGISTRY.get(key.lower())
    if fmt is not None and (fmt.parameter_starts is None or fmt.model_events):
        return
    unapplied = [
        f"event {str(name)!r} sets {event.affect.rhs}"
        for model in dynamics_library(experiment).values()
        for name, event in keyed_items(getattr(model, "events", None), "events")
        if getattr(getattr(event, "affect", None), "rhs", None)
    ]
    if unapplied:
        applying = ", ".join(sorted(f.key for f in list_formats() if f.model_events))
        raise NotImplementedError(
            f"the {key if fmt is None else fmt.key} backend has no path for the affect of an event a model declares, so it would integrate the model without it: {' and '.join(unapplied)}. "
            f"Run the experiment on a backend that applies a model's events ({applying}), or remove the event."
        )


def render(experiment, fmt_key: str, **kwargs) -> str:
    """Resolve *fmt_key*, lower *experiment* (:func:`tvbo.multiscale.lower`) for a format that integrates, invoke its renderer, prune its dead imports, and format it.

    All three happen here rather than in each renderer so that every backend — including the ones that render through an adapter and never touch the template helpers — is held to the same house style. Pruning precedes formatting because it edits statements and black only edits layout. See :mod:`tvbo.codegen.prune` and :mod:`tvbo.codegen.style`.
    """
    fmt = resolve(fmt_key)
    if fmt.parameter_starts is not None:
        from tvbo.multiscale import lower

        experiment = lower(experiment)
    refuse_unlowered_parameter_starts(experiment, fmt.key)
    refuse_undeclared_events(experiment, fmt.key)
    refuse_unapplied_model_events(experiment, fmt.key)
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
