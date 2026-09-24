"""The file a record is being loaded from, held as scoped context rather than written into the record.

A recipe names its companions by paths relative to itself — a ``data_file:``, an ``edge_matrix_files:`` entry, a callable's module beside the YAML — so whatever resolves them needs the recipe's directory. That directory is context, never data: resolving the paths to absolute ones in the loaded dict would publish a machine-local path into every dump. The loader states the file for the duration of the construction it starts, and whatever the construction builds reads it back, at any depth, without it being threaded through every constructor.

A :class:`contextvars.ContextVar` makes the statement scoped: a nested load (a study materialising its experiments, an experiment building its network) sees its own file and restores the outer one on exit, and two threads loading two recipes never see each other's.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_SOURCE_FILE: ContextVar[str | None] = ContextVar("tvbo_source_file", default=None)


@contextmanager
def loading_from(path: str | os.PathLike | None) -> Iterator[str | None]:
    """State that everything constructed inside the block is being loaded from *path*.

    The path is resolved to an absolute one once, here. ``None`` states nothing: the block sees whatever an enclosing load stated, so a caller that may or may not hold a file wraps its construction the same way in both cases.

    Args:
        path: The recipe file being loaded, or ``None``.

    Yields:
        The absolute source file in effect inside the block.
    """
    if path is None:
        yield _SOURCE_FILE.get()
        return
    token = _SOURCE_FILE.set(str(Path(path).resolve()))
    try:
        yield _SOURCE_FILE.get()
    finally:
        _SOURCE_FILE.reset(token)


def current_source_file() -> str | None:
    """The absolute path of the recipe being loaded, or ``None`` outside any load."""
    return _SOURCE_FILE.get()


def current_source_dir() -> str | None:
    """The directory of the recipe being loaded — what its relative paths resolve against — or ``None`` outside any load."""
    source = _SOURCE_FILE.get()
    return os.path.dirname(source) if source else None
