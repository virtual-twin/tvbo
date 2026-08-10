"""Helpers for locating the auto-07p installation used by the NumCont backend.

TVBO requires the ``AUTO_DIR`` environment variable to be set to the root of
an existing auto-07p installation. There is no fallback file lookup -- the
environment variable is the single source of truth.
"""

from __future__ import annotations

import os


def check_auto_dir() -> str:
    """Return a validated absolute ``AUTO_DIR`` path.

    Raises
    ------
    EnvironmentError
        If ``AUTO_DIR`` is unset or empty.
    FileNotFoundError
        If ``AUTO_DIR`` does not point to an existing directory.
    """
    auto_dir = os.environ.get("AUTO_DIR", "").strip()
    if not auto_dir:
        raise EnvironmentError(
            "AUTO_DIR is not set. Export it to your auto-07p installation, e.g. `export AUTO_DIR=$HOME/Applications/auto-07p`."
        )
    auto_dir = os.path.abspath(os.path.expandvars(os.path.expanduser(auto_dir)))
    if not os.path.isdir(auto_dir):
        raise FileNotFoundError(f"AUTO_DIR={auto_dir!r} does not point to an existing directory.")
    os.environ["AUTO_DIR"] = auto_dir
    return auto_dir
