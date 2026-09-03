"""What the installed tvboptim can do, so a test needing more than it ships says which.

TVBO's tvboptim backend targets capabilities a released tvboptim may not carry yet, and CI installs tvboptim from PyPI while a developer's checkout is often an editable one that is ahead of it. A test that needs such a capability must not read as a defect in TVBO, and must not read as "API absent" either: each guard here names exactly what the installed tvboptim lacks, so a rename, a missing release and a genuinely absent module are three different messages in the report, and the test resumes by itself once tvboptim ships the capability. A guard that can gate a whole module skips; one that gates individual cases marks them xfail, so the cases CI is not running are counted rather than invisible.

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
        return "installed tvboptim's axes take no wrap=, which TVBO emits for a swept per-edge network leaf"
    return None


def needs_axis_wrap(test):
    """Mark one test as expected to fail where tvboptim's grid axes take no ``wrap=``.

    ``xfail`` rather than ``skipif``, and the difference is the point: a skipped test reports nothing, so a run that never exercised the per-edge network axes at all reads exactly like one that exercised them and passed. An xfail is counted and named in the summary, so the gap is visible in every run that has it, and the day tvboptim ships ``wrap=`` the case turns up as an XPASS rather than staying silent.

    Non-strict, because the same suite must stay green on both installs; conditional, so where ``wrap=`` is present the test is an ordinary one and a real failure is a real failure.

    Per test rather than per module: these files mostly exercise axes TVBO renders without a wrapper, and marking those too would hide far more than the gap warrants.
    """
    reason = axis_wrap_reason()
    return pytest.mark.xfail(reason is not None, reason=reason or "", strict=False)(test)
