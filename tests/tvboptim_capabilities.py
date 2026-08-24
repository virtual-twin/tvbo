"""What the installed tvboptim can do, so a test needing more than it ships says which.

TVBO's tvboptim backend targets capabilities a released tvboptim may not carry yet, and CI installs tvboptim from PyPI while a developer's checkout is often an editable one that is ahead of it. A test that needs such a capability must not read as a defect in TVBO, and must not read as "API absent" either: each guard here skips with a reason naming exactly what the installed tvboptim lacks, so a rename, a missing release and a genuinely absent module are three different messages in the skip report, and the test resumes by itself once tvboptim ships the capability.

Every guard decides by asking the installed package rather than by comparing versions, since a version number does not distinguish a release from an editable checkout that shares it — which is what hid these gaps until a catch-all CI shard ran the tests that prove them.
"""

from __future__ import annotations

import functools
import importlib.util

import pytest

HETEROGENEOUS_NAMES = ("HeterogeneousNetwork", "NodeGroup", "SignalRoute")
"""The ``network_dynamics`` names ``tvbo.adapters.tvboptim`` imports to build a heterogeneous run."""


def require_heterogeneous_engine():
    """Skip the whole module unless tvboptim ships the heterogeneous network-dynamics engine.

    Presence of the module is decided by ``find_spec``, which does not execute it, and the import below is deliberately unguarded: a ``network_dynamics`` that fails to load is a defect and must raise rather than read as a version difference.
    """
    if importlib.util.find_spec("tvboptim.experimental.network_dynamics") is None:
        pytest.skip("tvboptim has no heterogeneous network-dynamics API", allow_module_level=True)

    import tvboptim.experimental.network_dynamics as network_dynamics

    missing = [name for name in HETEROGENEOUS_NAMES if not hasattr(network_dynamics, name)]
    if missing:
        pytest.skip(f"installed tvboptim's network_dynamics exposes no {', '.join(missing)}", allow_module_level=True)


@functools.cache
def axis_wrap_reason() -> str | None:
    """Why the installed tvboptim's grid axes cannot take ``wrap=``, or ``None`` when they can.

    Decided by constructing an axis rather than by reading a signature, because that is the call the generated code makes and the only thing that settles whether it will run.
    """
    try:
        from tvboptim.types.spaces import DataAxis
    except Exception as exc:
        return f"tvboptim's axis types are unavailable ({exc})"
    try:
        DataAxis([0.0, 1.0], wrap=lambda value: value)
    except TypeError:
        return "installed tvboptim's axes take no wrap=, which TVBO emits for a swept event onset under a transient and for a swept per-edge network leaf"
    return None


def needs_axis_wrap(test):
    """Mark one test as running only where tvboptim's grid axes take ``wrap=``.

    Per test rather than per module: these files mostly exercise axes TVBO renders without a wrapper, and skipping those too would hide far more than the gap warrants.
    """
    reason = axis_wrap_reason()
    return pytest.mark.skipif(reason is not None, reason=reason or "")(test)
