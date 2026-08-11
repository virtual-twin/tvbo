# Copyright Berlin Institute of Health / Charité University Medicine Berlin
# Department of Neurology and Experimental Neurology
# Brain Simulation Section

"""Central logging configuration for TVBO.

Every part of TVBO logs through the ``tvbo`` logger hierarchy:

* in-package modules use ``logger = logging.getLogger(__name__)`` — their names
  already sit under ``tvbo`` (e.g. ``tvbo.classes.experiment``);
* generated backend scripts (tvboptim, …) use ``logging.getLogger("tvbo.run")``
  so their progress output is controlled by the very same switch, regardless of which backend produced them or whether they run in-process or standalone.

Importing tvbo as a library stays silent: the package installs only a
:class:`~logging.NullHandler`. Entry points that are meant to surface progress —
``tvbo run`` and :meth:`SimulationExperiment.run` — call :func:`configure_logging` (directly, or via :func:`ensure_configured`) to attach a stderr handler.

One switch controls all of it, no matter the entry point:

* the ``TVBO_LOG_LEVEL`` environment variable
  (``DEBUG`` / ``INFO`` / ``WARNING`` / ``ERROR`` / ``CRITICAL`` / ``OFF``), or
* :func:`set_log_level` / :func:`silence` at runtime, or
* an explicit ``configure_logging(level=...)``.

Example:
    >>> import tvbo
    >>> tvbo.set_log_level("WARNING")   # quiet the progress banners everywhere
    >>> tvbo.silence()                  # turn tvbo logging off entirely
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator, Optional, Union

__all__ = [
    "logger",
    "configure_logging",
    "ensure_configured",
    "set_log_level",
    "get_log_level",
    "silence",
    "log_level",
    "LOGGER_NAME",
    "ENV_VAR",
]

LOGGER_NAME = "tvbo"
"""Root of the tvbo logger hierarchy; every package and generated-code logger is a child of this and inherits its level and handlers."""
ENV_VAR = "TVBO_LOG_LEVEL"
"""Environment variable read as the central level switch when nothing is passed."""
DEFAULT_LEVEL = logging.INFO
"""Level used when neither an explicit argument nor the env var is set."""
DEFAULT_FORMAT = "%(levelname)s [%(name)s] %(message)s"
"""Default stderr handler format for tvbo-managed output."""

# Effective "off": above CRITICAL, so no standard record is ever emitted.
_OFF = logging.CRITICAL + 1
# Marks the single handler this module owns, so configuration stays idempotent.
_MANAGED = "_tvbo_managed_handler"

LevelLike = Union[int, str, None]

logger = logging.getLogger(LOGGER_NAME)
# Library default: silent unless an entry point or the embedding app configures a handler. This replaces the old package-wide ``logging.disable(CRITICAL)``, which muted every logger in the process (tvbo's own included).
logger.addHandler(logging.NullHandler())


def _coerce_level(level: LevelLike) -> Optional[int]:
    """Turn a user-supplied level into a numeric level (``None`` passes through).

    Accepts ints, standard level names, and the aliases ``OFF``/``NONE``/
    ``SILENT``/``QUIET`` for "no output".
    """
    if level is None:
        return None
    if isinstance(level, bool):  # bool is an int subclass — treat explicitly
        return logging.DEBUG if level else _OFF
    if isinstance(level, int):
        return level
    text = str(level).strip().upper()
    if text in ("OFF", "NONE"):
        return _OFF
    value = logging.getLevelName(text)  # name → int for known levels
    if isinstance(value, int):
        return value
    raise ValueError(f"Unknown log level {level!r}; use an int or one of DEBUG, INFO, WARNING, ERROR, CRITICAL, OFF.")


def _env_level() -> Optional[int]:
    """Numeric level from ``TVBO_LOG_LEVEL``, or ``None`` if unset/invalid."""
    raw = os.environ.get(ENV_VAR)
    if not raw or not raw.strip():
        return None
    try:
        return _coerce_level(raw)
    except ValueError:
        logging.getLogger(__name__).warning(
            "Ignoring invalid %s=%r; expected DEBUG/INFO/WARNING/ERROR/CRITICAL/OFF.",
            ENV_VAR,
            raw,
        )
        return None


def _resolve_level(level: LevelLike) -> int:
    """Resolve a level with precedence: explicit arg → env var → default."""
    explicit = _coerce_level(level)
    if explicit is not None:
        return explicit
    from_env = _env_level()
    if from_env is not None:
        return from_env
    return DEFAULT_LEVEL


def _managed_handler(stream=None, fmt=None, datefmt=None) -> logging.Handler:
    handler = logging.StreamHandler(stream)  # stream=None → stderr
    handler.setFormatter(logging.Formatter(fmt or DEFAULT_FORMAT, datefmt))
    setattr(handler, _MANAGED, True)
    return handler


def _existing_managed() -> Optional[logging.Handler]:
    return next((h for h in logger.handlers if getattr(h, _MANAGED, False)), None)


def _has_real_handler(target: logging.Logger) -> bool:
    return any(not isinstance(h, logging.NullHandler) for h in target.handlers)


def _install_stream_handler(stream=None, fmt=None, datefmt=None, force=False) -> None:
    """Install (idempotently) the single tvbo-managed stream handler. Level-agnostic."""
    existing = _existing_managed()
    if existing is not None and not force:
        return
    if existing is not None:
        logger.removeHandler(existing)
    logger.addHandler(_managed_handler(stream, fmt, datefmt))
    # We own emission: don't also bubble to the root logger (avoids duplicate lines when the embedding application has configured root logging too).
    logger.propagate = False


def configure_logging(
    level: LevelLike = None,
    *,
    stream=None,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
    force: bool = False,
) -> logging.Logger:
    """Attach a stderr handler to the ``tvbo`` logger and set its level.

    Idempotent: the tvbo logger keeps at most one handler owned by this module.
    When *level* is ``None`` the level falls back to ``TVBO_LOG_LEVEL`` and then to :data:`DEFAULT_LEVEL`. Because the tvbo logger then owns its own output,
    its records stop propagating to the root logger (so an embedding application that also configured root logging does not print every line twice).

    Args:
        level: Desired level (int, name, or ``"OFF"``); ``None`` → env → default.
        stream: Target stream for the handler; ``None`` uses stderr.
        fmt: Handler format string; ``None`` uses :data:`DEFAULT_FORMAT`.
        datefmt: Optional date format for the handler.
        force: Replace an existing tvbo-managed handler (e.g. to change stream).

    Returns:
        The central ``tvbo`` logger.
    """
    logger.setLevel(_resolve_level(level))
    _install_stream_handler(stream, fmt, datefmt, force=force)
    return logger


def ensure_configured(level: LevelLike = None) -> logging.Logger:
    """Make tvbo logs visible for a run without clobbering an app's logging setup.

    Called from the run entry points (``tvbo run``, ``SimulationExperiment.run``) so that logging behaves the same however a run is launched:

    * if the tvbo logger or the root logger already has a real handler (an app,
      notebook, or a prior :func:`configure_logging` set things up), only the level is applied and the existing handlers keep emitting;
    * otherwise a default stderr handler is installed via :func:`configure_logging`.

    A level explicitly set earlier (``set_log_level`` / ``silence`` / a prior
    ``configure_logging``) is preserved: with no explicit *level* this only installs a default level the first time (while the logger is still at
    ``NOTSET``), so the central switch stays put across repeated ``.run()`` calls.

    Args:
        level: Level to apply; ``None`` keeps any level already set, else falls
            back to ``TVBO_LOG_LEVEL`` → :data:`DEFAULT_LEVEL` on first use.

    Returns:
        The central ``tvbo`` logger.
    """
    if level is not None:
        logger.setLevel(_resolve_level(level))
    elif logger.level == logging.NOTSET:
        logger.setLevel(_resolve_level(None))
    if _existing_managed() is not None:
        return logger
    if _has_real_handler(logger) or _has_real_handler(logging.getLogger()):
        # Something already configured logging — let tvbo records flow to it.
        return logger
    _install_stream_handler()
    return logger


def set_log_level(level: LevelLike) -> None:
    """Set the central ``tvbo`` logger level — the global on/off/verbosity switch.

    Affects every tvbo module and every generated backend script in the process.
    ``"OFF"`` (or :func:`silence`) turns tvbo logging off entirely.
    """
    logger.setLevel(_resolve_level(level))


def get_log_level() -> int:
    """Return the effective numeric level of the central ``tvbo`` logger."""
    return logger.getEffectiveLevel()


def silence() -> None:
    """Turn tvbo logging off (equivalent to ``TVBO_LOG_LEVEL=OFF``)."""
    logger.setLevel(_OFF)


@contextmanager
def log_level(level: LevelLike) -> Iterator[logging.Logger]:
    """Temporarily set the ``tvbo`` logger level within a ``with`` block.

    Useful to quiet a noisy section or to force verbosity in a test without leaking the change to the rest of the process.
    """
    previous = logger.level
    logger.setLevel(_resolve_level(level))
    try:
        yield logger
    finally:
        logger.setLevel(previous)
